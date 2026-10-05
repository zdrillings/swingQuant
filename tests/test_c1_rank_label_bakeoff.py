from __future__ import annotations

import unittest

import pandas as pd

from scripts.c1_rank_label_bakeoff import RANK_TARGET_COLUMN, add_rank_percentile_target


class C1RankLabelBakeoffTests(unittest.TestCase):
    def test_rank_percentile_target_groups_by_date_and_preserves_nans(self) -> None:
        frame = pd.DataFrame(
            [
                {"snapshot_date": "2026-01-02", "ticker": "A", "alpha_vs_sector_60d": 0.10},
                {"snapshot_date": "2026-01-02", "ticker": "B", "alpha_vs_sector_60d": 0.20},
                {"snapshot_date": "2026-01-02", "ticker": "C", "alpha_vs_sector_60d": 0.20},
                {"snapshot_date": "2026-01-02", "ticker": "D", "alpha_vs_sector_60d": None},
                {"snapshot_date": "2026-01-05", "ticker": "A", "alpha_vs_sector_60d": -0.05},
                {"snapshot_date": "2026-01-05", "ticker": "B", "alpha_vs_sector_60d": 0.15},
            ]
        )

        ranked = add_rank_percentile_target(frame)

        day_one = ranked[ranked["snapshot_date"].eq(pd.Timestamp("2026-01-02"))].set_index("ticker")
        self.assertAlmostEqual(float(day_one.loc["A", RANK_TARGET_COLUMN]), 1.0 / 3.0)
        self.assertAlmostEqual(float(day_one.loc["B", RANK_TARGET_COLUMN]), 5.0 / 6.0)
        self.assertAlmostEqual(float(day_one.loc["C", RANK_TARGET_COLUMN]), 5.0 / 6.0)
        self.assertTrue(pd.isna(day_one.loc["D", RANK_TARGET_COLUMN]))

        day_two = ranked[ranked["snapshot_date"].eq(pd.Timestamp("2026-01-05"))].set_index("ticker")
        self.assertAlmostEqual(float(day_two.loc["A", RANK_TARGET_COLUMN]), 0.5)
        self.assertAlmostEqual(float(day_two.loc["B", RANK_TARGET_COLUMN]), 1.0)


if __name__ == "__main__":
    unittest.main()
