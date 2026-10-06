from __future__ import annotations

import unittest

import pandas as pd

from src.research.sector_neutral_selection import (
    deoverlap_oos_predictions,
    evaluate_selection_modes,
    select_sector_neutral_top_n,
    verdict_for_model,
)


class SectorNeutralSelectionTests(unittest.TestCase):
    def test_sector_neutral_selects_top_rank_per_sector_before_final_top_n(self) -> None:
        frame = pd.DataFrame(
            [
                {"ticker": "AAA", "sector": "Tech", "predicted_alpha": 0.90},
                {"ticker": "BBB", "sector": "Tech", "predicted_alpha": 0.80},
                {"ticker": "CCC", "sector": "Energy", "predicted_alpha": 0.70},
                {"ticker": "DDD", "sector": "Materials", "predicted_alpha": 0.60},
            ]
        )

        selected = select_sector_neutral_top_n(frame, top_n=2)

        self.assertEqual(selected["ticker"].tolist(), ["AAA", "CCC"])

    def test_sector_neutral_ties_are_ticker_stable(self) -> None:
        frame = pd.DataFrame(
            [
                {"ticker": "BBB", "sector": "Tech", "predicted_alpha": 0.50},
                {"ticker": "AAA", "sector": "Tech", "predicted_alpha": 0.50},
                {"ticker": "CCC", "sector": "Energy", "predicted_alpha": 0.40},
            ]
        )

        selected = select_sector_neutral_top_n(frame, top_n=2)

        self.assertEqual(selected["ticker"].tolist(), ["AAA", "CCC"])

    def test_sector_neutral_groups_missing_sectors_without_dropping_rows(self) -> None:
        frame = pd.DataFrame(
            [
                {"ticker": "AAA", "sector": None, "predicted_alpha": 0.90},
                {"ticker": "BBB", "sector": "", "predicted_alpha": 0.80},
                {"ticker": "CCC", "sector": "Energy", "predicted_alpha": 0.70},
            ]
        )

        selected = select_sector_neutral_top_n(frame, top_n=2)

        self.assertEqual(selected["ticker"].tolist(), ["AAA", "CCC"])

    def test_sector_neutral_empty_universe_returns_empty_frame(self) -> None:
        frame = pd.DataFrame(columns=["ticker", "sector", "predicted_alpha"])

        selected = select_sector_neutral_top_n(frame, top_n=2)

        self.assertTrue(selected.empty)

    def test_deoverlap_keeps_horizon_spaced_rows_per_ticker(self) -> None:
        frame = pd.DataFrame(
            [
                {"snapshot_date": "2026-01-01", "ticker": "AAA", "predicted_alpha": 0.1},
                {"snapshot_date": "2026-01-02", "ticker": "AAA", "predicted_alpha": 0.2},
                {"snapshot_date": "2026-01-04", "ticker": "AAA", "predicted_alpha": 0.3},
                {"snapshot_date": "2026-01-02", "ticker": "BBB", "predicted_alpha": 0.4},
            ]
        )

        selected = deoverlap_oos_predictions(
            frame,
            horizon_days=2,
            calendar_dates=pd.date_range("2026-01-01", periods=4, freq="D").tolist(),
        )

        self.assertEqual(
            selected.sort_values(["ticker", "snapshot_date"])["predicted_alpha"].tolist(),
            [0.1, 0.3, 0.4],
        )

    def test_evaluate_selection_modes_reports_sector_neutral_loss(self) -> None:
        frame = pd.DataFrame(
            [
                {"snapshot_date": "2026-01-01", "ticker": "AAA", "sector": "Tech", "model_name": "m", "predicted_alpha": 0.9, "alpha": 0.10},
                {"snapshot_date": "2026-01-01", "ticker": "BBB", "sector": "Tech", "model_name": "m", "predicted_alpha": 0.8, "alpha": 0.08},
                {"snapshot_date": "2026-01-01", "ticker": "CCC", "sector": "Energy", "model_name": "m", "predicted_alpha": 0.7, "alpha": -0.05},
            ]
        )

        results = evaluate_selection_modes(frame, top_n=2, target_column="alpha")

        self.assertEqual(verdict_for_model(results, "m"), "sector-neutral loses on money statistics")


if __name__ == "__main__":
    unittest.main()
