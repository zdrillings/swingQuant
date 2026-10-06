from __future__ import annotations

import unittest

import pandas as pd

from src.research.orthogonal_ensemble import (
    build_rank_ensemble_predictions,
    inverse_absolute_spearman_weights,
    window_spearman_summary,
)


class OrthogonalEnsembleTests(unittest.TestCase):
    def test_inverse_absolute_spearman_weights_normalize_and_favor_lower_abs_correlation(self) -> None:
        weights = inverse_absolute_spearman_weights({"high": 0.20, "low": -0.05})

        self.assertAlmostEqual(sum(weights.values()), 1.0)
        self.assertGreater(weights["low"], weights["high"])

    def test_inverse_absolute_spearman_weights_floor_near_zero_values(self) -> None:
        weights = inverse_absolute_spearman_weights({"zero": 0.0, "small": 0.002}, floor=0.001)

        self.assertAlmostEqual(sum(weights.values()), 1.0)
        self.assertGreater(weights["zero"], weights["small"])

    def test_rank_ensemble_inner_aligns_members_and_averages_daily_percentile_ranks(self) -> None:
        frame = pd.DataFrame(
            [
                {"snapshot_date": "2026-01-01", "ticker": "AAA", "model_name": "m1", "predicted_alpha": 0.9, "alpha": 0.10},
                {"snapshot_date": "2026-01-01", "ticker": "BBB", "model_name": "m1", "predicted_alpha": 0.1, "alpha": -0.05},
                {"snapshot_date": "2026-01-01", "ticker": "AAA", "model_name": "m2", "predicted_alpha": 0.2, "alpha": 0.10},
                {"snapshot_date": "2026-01-01", "ticker": "BBB", "model_name": "m2", "predicted_alpha": 0.8, "alpha": -0.05},
                {"snapshot_date": "2026-01-01", "ticker": "CCC", "model_name": "m2", "predicted_alpha": 0.7, "alpha": 0.02},
            ]
        )

        ensemble = build_rank_ensemble_predictions(frame, members=("m1", "m2"), target_column="alpha")

        self.assertEqual(ensemble["ticker"].tolist(), ["AAA", "BBB"])
        scores = dict(zip(ensemble["ticker"], ensemble["predicted_alpha"], strict=False))
        self.assertAlmostEqual(scores["AAA"], (1.0 + 0.5) / 2.0)
        self.assertAlmostEqual(scores["BBB"], (0.5 + 1.0) / 2.0)

    def test_rank_ensemble_applies_supplied_weights(self) -> None:
        frame = pd.DataFrame(
            [
                {"snapshot_date": "2026-01-01", "ticker": "AAA", "model_name": "m1", "predicted_alpha": 0.9, "alpha": 0.10},
                {"snapshot_date": "2026-01-01", "ticker": "BBB", "model_name": "m1", "predicted_alpha": 0.1, "alpha": -0.05},
                {"snapshot_date": "2026-01-01", "ticker": "AAA", "model_name": "m2", "predicted_alpha": 0.2, "alpha": 0.10},
                {"snapshot_date": "2026-01-01", "ticker": "BBB", "model_name": "m2", "predicted_alpha": 0.8, "alpha": -0.05},
            ]
        )

        ensemble = build_rank_ensemble_predictions(
            frame,
            members=("m1", "m2"),
            target_column="alpha",
            weights={"m1": 0.75, "m2": 0.25},
        )

        scores = dict(zip(ensemble["ticker"], ensemble["predicted_alpha"], strict=False))
        self.assertAlmostEqual(scores["AAA"], (1.0 * 0.75) + (0.5 * 0.25))
        self.assertAlmostEqual(scores["BBB"], (0.5 * 0.75) + (1.0 * 0.25))

    def test_window_summary_reports_full_last_and_trailing_spearman(self) -> None:
        rows = []
        for index in range(65):
            rows.extend(
                [
                    {"snapshot_date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=index), "ticker": "AAA", "predicted_alpha": 2.0, "alpha": 2.0},
                    {"snapshot_date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=index), "ticker": "BBB", "predicted_alpha": 1.0, "alpha": 1.0},
                    {"snapshot_date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=index), "ticker": "CCC", "predicted_alpha": 0.0, "alpha": 0.0},
                ]
            )

        summary = window_spearman_summary(pd.DataFrame(rows), target_column="alpha")

        self.assertEqual(summary["dates"], 65)
        self.assertAlmostEqual(float(summary["full_oos_spearman"]), 1.0)
        self.assertAlmostEqual(float(summary["last_fold_spearman"]), 1.0)
        self.assertAlmostEqual(float(summary["trailing_3fold_spearman"]), 1.0)


if __name__ == "__main__":
    unittest.main()
