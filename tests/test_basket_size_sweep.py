from __future__ import annotations

import unittest

import pandas as pd

from src.research.basket_size_sweep import clears_both_beat_windows, summarize_basket_sizes


class BasketSizeSweepTests(unittest.TestCase):
    def test_summarize_basket_sizes_reports_requested_top_n_rows(self) -> None:
        rows = []
        for index in range(5):
            snapshot_date = pd.Timestamp("2026-01-01") + pd.Timedelta(days=index)
            rows.extend(
                [
                    {"snapshot_date": snapshot_date, "ticker": "AAA", "predicted_alpha": 4.0, "alpha": 0.10},
                    {"snapshot_date": snapshot_date, "ticker": "BBB", "predicted_alpha": 3.0, "alpha": 0.08},
                    {"snapshot_date": snapshot_date, "ticker": "CCC", "predicted_alpha": 2.0, "alpha": -0.02},
                    {"snapshot_date": snapshot_date, "ticker": "DDD", "predicted_alpha": 1.0, "alpha": -0.04},
                ]
            )

        summary = summarize_basket_sizes(
            pd.DataFrame(rows),
            basket_sizes=(2, 3),
            target_column="alpha",
            fold_size=5,
            trailing_folds=1,
        )

        self.assertEqual(summary["basket_size"].tolist(), [2, 3])
        top2 = summary.set_index("basket_size").loc[2]
        top3 = summary.set_index("basket_size").loc[3]
        self.assertAlmostEqual(float(top2["full_oos_avg_pick_count"]), 2.0)
        self.assertAlmostEqual(float(top3["full_oos_avg_pick_count"]), 3.0)
        self.assertGreater(float(top2["trailing_3fold_mean_target_excess"]), float(top3["trailing_3fold_mean_target_excess"]))

    def test_clears_both_beat_windows_requires_last_and_trailing(self) -> None:
        self.assertTrue(
            clears_both_beat_windows(
                {
                    "last_fold_beat_universe_rate": 0.50,
                    "trailing_3fold_beat_universe_rate": 0.55,
                }
            )
        )
        self.assertFalse(
            clears_both_beat_windows(
                {
                    "last_fold_beat_universe_rate": 0.49,
                    "trailing_3fold_beat_universe_rate": 0.55,
                }
            )
        )


if __name__ == "__main__":
    unittest.main()
