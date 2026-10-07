from __future__ import annotations

import unittest

import pandas as pd

from scripts.basket_calibration import (
    BasketVariant,
    TARGET_COLUMN,
    add_chronological_isotonic_probability,
    build_variants,
    evaluate_selection_window,
    score_threshold_values,
    verdict_text,
)


class BasketCalibrationTests(unittest.TestCase):
    def test_score_threshold_values_uses_requested_quantiles(self) -> None:
        frame = pd.DataFrame({"predicted_alpha": [1.0, 2.0, 3.0, 4.0]})

        thresholds = score_threshold_values(frame, score_column="predicted_alpha", quantiles=(0.0, 0.5, 1.0))

        self.assertEqual([1.0, 2.5, 4.0], thresholds)

    def test_chronological_isotonic_uses_only_prior_dates(self) -> None:
        rows = []
        for day in range(4):
            snapshot_date = pd.Timestamp("2026-01-01") + pd.Timedelta(days=day)
            for ticker, score, target in (
                ("AAA", 0.9, 0.05),
                ("BBB", 0.1, -0.05),
            ):
                rows.append(
                    {
                        "snapshot_date": snapshot_date,
                        "ticker": ticker,
                        "predicted_alpha": score,
                        TARGET_COLUMN: target,
                    }
                )
        frame = pd.DataFrame(rows)

        calibrated = add_chronological_isotonic_probability(frame, target_column=TARGET_COLUMN, min_train_rows=4)

        first_day = calibrated[calibrated["snapshot_date"].eq(pd.Timestamp("2026-01-01"))]
        last_day = calibrated[calibrated["snapshot_date"].eq(pd.Timestamp("2026-01-04"))].set_index("ticker")
        self.assertTrue((first_day["calibrated_p_beat_sector_oos"] == 0.5).all())
        self.assertGreater(
            float(last_day.loc["AAA", "calibrated_p_beat_sector_oos"]),
            float(last_day.loc["BBB", "calibrated_p_beat_sector_oos"]),
        )

    def test_evaluate_selection_window_leaves_gated_slots_empty(self) -> None:
        frame = pd.DataFrame(
            [
                {"snapshot_date": "2026-01-01", "ticker": "AAA", "predicted_alpha": 0.9, TARGET_COLUMN: 0.04},
                {"snapshot_date": "2026-01-01", "ticker": "BBB", "predicted_alpha": 0.8, TARGET_COLUMN: 0.03},
                {"snapshot_date": "2026-01-01", "ticker": "CCC", "predicted_alpha": 0.1, TARGET_COLUMN: -0.02},
                {"snapshot_date": "2026-01-02", "ticker": "AAA", "predicted_alpha": 0.2, TARGET_COLUMN: 0.04},
                {"snapshot_date": "2026-01-02", "ticker": "BBB", "predicted_alpha": 0.1, TARGET_COLUMN: 0.03},
                {"snapshot_date": "2026-01-02", "ticker": "CCC", "predicted_alpha": 0.0, TARGET_COLUMN: -0.02},
            ]
        )
        frame["snapshot_date"] = pd.to_datetime(frame["snapshot_date"])
        variant = BasketVariant("score_gate", "predicted_alpha", top_n=2, min_score=0.8, threshold_label="0.8")

        metrics = evaluate_selection_window(
            frame,
            variant=variant,
            target_column=TARGET_COLUMN,
            selected_dates=sorted(frame["snapshot_date"].unique().tolist()),
            label="toy",
        )

        self.assertEqual(metrics["dates"], 2)
        self.assertEqual(metrics["active_dates"], 1)
        self.assertEqual(metrics["total_picks"], 2)
        self.assertAlmostEqual(float(metrics["avg_pick_count"]), 1.0)
        self.assertGreater(float(metrics["hit_rate_excess"]), 0.0)

    def test_build_variants_includes_score_and_calibrated_probability_cuts(self) -> None:
        frame = pd.DataFrame({"predicted_alpha": [0.1, 0.2, 0.3], "calibrated_p_beat_sector_oos": [0.4, 0.5, 0.6]})

        variants = build_variants(frame)

        names = {variant.variant for variant in variants}
        self.assertIn("as_is_top2_by_score", names)
        self.assertIn("top1_by_score", names)
        self.assertTrue(any(name.startswith("score_gate_") for name in names))
        self.assertTrue(any(name.startswith("calibrated_p_gate_") for name in names))

    def test_verdict_identifies_floor_passer(self) -> None:
        rows = pd.DataFrame(
            [
                {
                    "variant": "as_is_top2_by_score",
                    "full_oos_spearman": 0.01,
                    "trailing_3fold_hit_rate_excess": 0.01,
                    "trailing_3fold_beat_universe_rate": 0.49,
                    "trailing_3fold_total_picks": 120,
                    "clears_trailing_floors": False,
                },
                {
                    "variant": "score_gate",
                    "full_oos_spearman": 0.02,
                    "trailing_3fold_hit_rate_excess": 0.03,
                    "trailing_3fold_beat_universe_rate": 0.55,
                    "trailing_3fold_total_picks": 90,
                    "clears_trailing_floors": True,
                },
            ]
        )

        self.assertIn("PASS: score_gate", verdict_text(rows))


if __name__ == "__main__":
    unittest.main()
