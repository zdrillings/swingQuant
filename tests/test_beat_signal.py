from __future__ import annotations

import unittest

import pandas as pd

from src.research.beat_signal import add_forward_beat_label, build_rank_hybrid_predictions


class BeatSignalTests(unittest.TestCase):
    def test_add_forward_beat_label_uses_strict_positive_alpha_and_preserves_missing(self) -> None:
        frame = pd.DataFrame(
            [
                {"ticker": "AAA", "alpha_vs_sector_60d": 0.001},
                {"ticker": "BBB", "alpha_vs_sector_60d": 0.0},
                {"ticker": "CCC", "alpha_vs_sector_60d": -0.001},
                {"ticker": "DDD", "alpha_vs_sector_60d": None},
            ]
        )

        labelled = add_forward_beat_label(frame).set_index("ticker")

        self.assertEqual(float(labelled.loc["AAA", "forward_beat_sector_60d"]), 1.0)
        self.assertEqual(float(labelled.loc["BBB", "forward_beat_sector_60d"]), 0.0)
        self.assertEqual(float(labelled.loc["CCC", "forward_beat_sector_60d"]), 0.0)
        self.assertTrue(pd.isna(labelled.loc["DDD", "forward_beat_sector_60d"]))

    def test_rank_hybrid_combines_beat_probability_and_alpha_ranks_by_date(self) -> None:
        beat = pd.DataFrame(
            [
                {"snapshot_date": "2026-01-02", "ticker": "AAA", "predicted_alpha": 0.9, "alpha_vs_sector_60d": 0.01},
                {"snapshot_date": "2026-01-02", "ticker": "BBB", "predicted_alpha": 0.5, "alpha_vs_sector_60d": 0.02},
                {"snapshot_date": "2026-01-02", "ticker": "CCC", "predicted_alpha": 0.1, "alpha_vs_sector_60d": -0.01},
            ]
        )
        alpha = pd.DataFrame(
            [
                {"snapshot_date": "2026-01-02", "ticker": "AAA", "predicted_alpha": 0.1, "alpha_vs_sector_60d": 0.01},
                {"snapshot_date": "2026-01-02", "ticker": "BBB", "predicted_alpha": 0.5, "alpha_vs_sector_60d": 0.02},
                {"snapshot_date": "2026-01-02", "ticker": "CCC", "predicted_alpha": 0.9, "alpha_vs_sector_60d": -0.01},
            ]
        )

        hybrid = build_rank_hybrid_predictions(beat, alpha, beat_weight=0.7).set_index("ticker")

        self.assertGreater(float(hybrid.loc["AAA", "predicted_alpha"]), float(hybrid.loc["CCC", "predicted_alpha"]))
        self.assertAlmostEqual(float(hybrid.loc["BBB", "predicted_alpha"]), 2.0 / 3.0)
        self.assertEqual(str(hybrid.loc["AAA", "model_name"]), "beat_alpha_rank_hybrid")

    def test_rank_hybrid_can_use_calibrated_probability_column(self) -> None:
        beat = pd.DataFrame(
            [
                {"snapshot_date": "2026-01-02", "ticker": "AAA", "calibrated_p": 0.2, "alpha_vs_sector_60d": 0.01},
                {"snapshot_date": "2026-01-02", "ticker": "BBB", "calibrated_p": 0.8, "alpha_vs_sector_60d": 0.02},
            ]
        )
        alpha = pd.DataFrame(
            [
                {"snapshot_date": "2026-01-02", "ticker": "AAA", "predicted_alpha": 0.9, "alpha_vs_sector_60d": 0.01},
                {"snapshot_date": "2026-01-02", "ticker": "BBB", "predicted_alpha": 0.1, "alpha_vs_sector_60d": 0.02},
            ]
        )

        hybrid = build_rank_hybrid_predictions(
            beat,
            alpha,
            beat_weight=0.5,
            beat_score_column="calibrated_p",
        ).set_index("ticker")

        self.assertAlmostEqual(float(hybrid.loc["AAA", "predicted_alpha"]), 0.75)
        self.assertAlmostEqual(float(hybrid.loc["BBB", "predicted_alpha"]), 0.75)


if __name__ == "__main__":
    unittest.main()
