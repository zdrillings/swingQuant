from __future__ import annotations

import math
import unittest

import pandas as pd

from scripts import p3_adaptive_training as p3


class P3AdaptiveTrainingTests(unittest.TestCase):
    def test_recency_weights_give_latest_date_unit_weight(self) -> None:
        dates = pd.Series(pd.bdate_range("2026-01-01", periods=4))

        weights = p3.recency_weights_by_snapshot_date(dates, half_life_dates=2)

        self.assertAlmostEqual(float(weights[-1]), 1.0)
        self.assertAlmostEqual(float(weights[-3]), 0.5)
        self.assertLess(float(weights[0]), float(weights[1]))

    def test_recency_weights_assign_same_weight_within_date(self) -> None:
        dates = pd.Series(
            [
                pd.Timestamp("2026-01-01"),
                pd.Timestamp("2026-01-01"),
                pd.Timestamp("2026-01-02"),
            ]
        )

        weights = p3.recency_weights_by_snapshot_date(dates, half_life_dates=1)

        self.assertAlmostEqual(float(weights[0]), float(weights[1]))
        self.assertAlmostEqual(float(weights[2]), 1.0)

    def test_recency_weights_handle_missing_dates_as_zero_weight(self) -> None:
        dates = pd.Series([pd.Timestamp("2026-01-01"), None, pd.Timestamp("2026-01-02")])

        weights = p3.recency_weights_by_snapshot_date(dates, half_life_dates=1)

        self.assertEqual(float(weights[1]), 0.0)
        self.assertTrue(math.isfinite(float(weights[0])))

    def test_recency_weights_can_age_on_full_reference_calendar(self) -> None:
        reference_dates = list(pd.bdate_range("2026-01-01", periods=4))
        sparse_dates = pd.Series([reference_dates[0], reference_dates[-1]])

        weights = p3.recency_weights_by_snapshot_date(
            sparse_dates,
            half_life_dates=2,
            reference_dates=reference_dates,
        )

        self.assertAlmostEqual(float(weights[0]), 2 ** (-3 / 2))
        self.assertAlmostEqual(float(weights[1]), 1.0)

    def test_verdict_requires_trailing_lift_without_full_oos_drop(self) -> None:
        rows = pd.DataFrame(
            [
                {
                    "model": "ridge_model",
                    "variant": "max_train_252",
                    "full_oos_spearman_delta_vs_252": 0.0,
                    "trailing_3fold_spearman_delta_vs_252": 0.0,
                },
                {
                    "model": "ridge_model",
                    "variant": "weighted_hl_63",
                    "full_oos_spearman_delta_vs_252": -0.0001,
                    "trailing_3fold_spearman_delta_vs_252": 0.0500,
                },
            ]
        )

        self.assertIn("ties/loses", p3._verdict(rows))

        rows.loc[1, "full_oos_spearman_delta_vs_252"] = 0.0
        self.assertIn("wins", p3._verdict(rows))


if __name__ == "__main__":
    unittest.main()
