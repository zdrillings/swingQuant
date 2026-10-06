from __future__ import annotations

import unittest

import pandas as pd

from scripts.c3_binary_beat_label import BINARY_TARGET_COLUMN, add_binary_beat_label, basket_stats


class C3BinaryBeatLabelTests(unittest.TestCase):
    def test_binary_beat_label_uses_two_percent_threshold_and_preserves_missing(self) -> None:
        frame = pd.DataFrame(
            [
                {"ticker": "A", "alpha_vs_sector_60d": 0.0199},
                {"ticker": "B", "alpha_vs_sector_60d": 0.0200},
                {"ticker": "C", "alpha_vs_sector_60d": -0.0100},
                {"ticker": "D", "alpha_vs_sector_60d": None},
            ]
        )

        labelled = add_binary_beat_label(frame)

        values = labelled.set_index("ticker")[BINARY_TARGET_COLUMN]
        self.assertEqual(float(values.loc["A"]), 0.0)
        self.assertEqual(float(values.loc["B"]), 1.0)
        self.assertEqual(float(values.loc["C"]), 0.0)
        self.assertTrue(pd.isna(values.loc["D"]))

    def test_basket_stats_average_top_n_by_date(self) -> None:
        frame = pd.DataFrame(
            [
                {"snapshot_date": "2026-01-02", "ticker": "A", "predicted_alpha": 0.9, "alpha_vs_sector_60d": 0.03},
                {"snapshot_date": "2026-01-02", "ticker": "B", "predicted_alpha": 0.8, "alpha_vs_sector_60d": -0.01},
                {"snapshot_date": "2026-01-02", "ticker": "C", "predicted_alpha": 0.1, "alpha_vs_sector_60d": 0.10},
                {"snapshot_date": "2026-01-05", "ticker": "A", "predicted_alpha": 0.7, "alpha_vs_sector_60d": -0.02},
                {"snapshot_date": "2026-01-05", "ticker": "B", "predicted_alpha": 0.6, "alpha_vs_sector_60d": -0.03},
            ]
        )

        stats = basket_stats(frame, top_n=2)

        self.assertEqual(stats["dates"], 2)
        self.assertEqual(stats["basket_rows"], 4)
        self.assertAlmostEqual(float(stats["hit_rate"]), 0.5)
        self.assertAlmostEqual(float(stats["beat_rate"]), 0.25)
        self.assertAlmostEqual(float(stats["mean_alpha"]), -0.0075)


if __name__ == "__main__":
    unittest.main()
