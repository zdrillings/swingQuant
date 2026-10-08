from __future__ import annotations

import unittest

import pandas as pd

from src.research.beat_calibration import (
    calibrated_probability_monotonic_by_score,
    chronological_isotonic_probability,
)


class BeatCalibrationTests(unittest.TestCase):
    def test_chronological_isotonic_probability_is_monotonic_by_score(self) -> None:
        rows = []
        for day in range(4):
            snapshot_date = pd.Timestamp("2026-01-01") + pd.Timedelta(days=day)
            rows.extend(
                [
                    {"snapshot_date": snapshot_date, "ticker": f"A{day}", "predicted_alpha": 0.1, "alpha": -0.03},
                    {"snapshot_date": snapshot_date, "ticker": f"B{day}", "predicted_alpha": 0.5, "alpha": 0.02},
                    {"snapshot_date": snapshot_date, "ticker": f"C{day}", "predicted_alpha": 0.9, "alpha": 0.06},
                ]
            )

        calibrated = chronological_isotonic_probability(
            pd.DataFrame(rows),
            target_column="alpha",
            min_train_rows=6,
        )

        last_day = calibrated[calibrated["snapshot_date"].eq(pd.Timestamp("2026-01-04"))]
        self.assertTrue(
            calibrated_probability_monotonic_by_score(
                last_day,
                score_column="predicted_alpha",
                probability_column="calibrated_p_beat_sector_oos",
            )
        )
        probabilities = last_day.sort_values("predicted_alpha")["calibrated_p_beat_sector_oos"].tolist()
        self.assertLessEqual(probabilities[0], probabilities[1])
        self.assertLessEqual(probabilities[1], probabilities[2])

    def test_chronological_isotonic_probability_uses_only_prior_dates(self) -> None:
        rows = []
        for day in range(3):
            snapshot_date = pd.Timestamp("2026-01-01") + pd.Timedelta(days=day)
            rows.extend(
                [
                    {"snapshot_date": snapshot_date, "ticker": f"H{day}", "predicted_alpha": 0.9, "alpha": -0.05},
                    {"snapshot_date": snapshot_date, "ticker": f"L{day}", "predicted_alpha": 0.1, "alpha": 0.05},
                ]
            )
        future_date = pd.Timestamp("2026-01-04")
        rows.extend(
            [
                {"snapshot_date": future_date, "ticker": "HF", "predicted_alpha": 0.9, "alpha": 0.10},
                {"snapshot_date": future_date, "ticker": "LF", "predicted_alpha": 0.1, "alpha": -0.10},
            ]
        )

        calibrated = chronological_isotonic_probability(
            pd.DataFrame(rows),
            target_column="alpha",
            min_train_rows=4,
        )
        third_day = calibrated[calibrated["snapshot_date"].eq(pd.Timestamp("2026-01-03"))].set_index("ticker")

        self.assertLessEqual(
            float(third_day.loc["H2", "calibrated_p_beat_sector_oos"]),
            float(third_day.loc["L2", "calibrated_p_beat_sector_oos"]),
        )


if __name__ == "__main__":
    unittest.main()
