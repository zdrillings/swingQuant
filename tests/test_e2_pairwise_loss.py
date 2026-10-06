from __future__ import annotations

import unittest

import pandas as pd

from scripts.e2_pairwise_loss import pairwise_group_sizes, same_date_pair_count, same_date_pair_labels


class E2PairwiseLossTests(unittest.TestCase):
    def test_same_date_pair_labels_only_pairs_rows_inside_date(self) -> None:
        frame = pd.DataFrame(
            [
                {"snapshot_date": "2026-01-02", "ticker": "A", "alpha_vs_sector_60d": 0.10},
                {"snapshot_date": "2026-01-02", "ticker": "B", "alpha_vs_sector_60d": 0.05},
                {"snapshot_date": "2026-01-02", "ticker": "C", "alpha_vs_sector_60d": 0.05},
                {"snapshot_date": "2026-01-05", "ticker": "A", "alpha_vs_sector_60d": -0.02},
                {"snapshot_date": "2026-01-05", "ticker": "B", "alpha_vs_sector_60d": 0.04},
            ]
        )

        pairs = same_date_pair_labels(frame)

        self.assertEqual(len(pairs.index), 3)
        self.assertEqual(set(pairs["snapshot_date"]), {pd.Timestamp("2026-01-02"), pd.Timestamp("2026-01-05")})
        first_day = pairs[pairs["snapshot_date"].eq(pd.Timestamp("2026-01-02"))]
        self.assertEqual(set(first_day["winner_ticker"]), {"A"})
        self.assertEqual(set(first_day["loser_ticker"]), {"B", "C"})
        self.assertTrue((first_day["target_delta"] == 0.05).all())
        second_day = pairs[pairs["snapshot_date"].eq(pd.Timestamp("2026-01-05"))].iloc[0]
        self.assertEqual(second_day["winner_ticker"], "B")
        self.assertEqual(second_day["loser_ticker"], "A")
        self.assertAlmostEqual(float(second_day["target_delta"]), 0.06)

    def test_same_date_pair_count_matches_explicit_pairs_without_materializing(self) -> None:
        frame = pd.DataFrame(
            [
                {"snapshot_date": "2026-01-02", "ticker": "A", "alpha_vs_sector_60d": 0.10},
                {"snapshot_date": "2026-01-02", "ticker": "B", "alpha_vs_sector_60d": 0.05},
                {"snapshot_date": "2026-01-02", "ticker": "C", "alpha_vs_sector_60d": 0.05},
                {"snapshot_date": "2026-01-05", "ticker": "A", "alpha_vs_sector_60d": -0.02},
                {"snapshot_date": "2026-01-05", "ticker": "B", "alpha_vs_sector_60d": 0.04},
            ]
        )

        pair_count, date_count = same_date_pair_count(frame)

        self.assertEqual(pair_count, len(same_date_pair_labels(frame).index))
        self.assertEqual(date_count, 2)

    def test_pairwise_group_sizes_are_sorted_by_snapshot_date(self) -> None:
        frame = pd.DataFrame(
            [
                {"snapshot_date": "2026-01-05", "ticker": "B"},
                {"snapshot_date": "2026-01-02", "ticker": "A"},
                {"snapshot_date": "2026-01-05", "ticker": "A"},
                {"snapshot_date": "2026-01-09", "ticker": "A"},
            ]
        )

        self.assertEqual(pairwise_group_sizes(frame), [1, 2, 1])


if __name__ == "__main__":
    unittest.main()
