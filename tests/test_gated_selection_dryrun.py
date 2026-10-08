from __future__ import annotations

import unittest

import pandas as pd

from scripts.gated_selection_dryrun import (
    AUDIT_MODEL,
    TARGET_COLUMN,
    acceptance_windows,
    floor_verdict_rows,
    prepare_oos_predictions,
    render_report,
)
from src.research.shortlist_model_service import ShortlistModelService
from src.utils.shortlist_selection_gate import ShortlistSelectionGate


class GatedSelectionDryRunTests(unittest.TestCase):
    def test_acceptance_windows_apply_non_empty_selection_gate_denominator(self) -> None:
        rows = []
        for snapshot_date, scores in (
            (pd.Timestamp("2026-01-02"), (0.90, 0.80, 0.10)),
            (pd.Timestamp("2026-01-05"), (0.20, 0.10, 0.00)),
        ):
            for ticker, score, target in zip(("AAA", "BBB", "CCC"), scores, (0.04, 0.03, -0.02), strict=True):
                rows.append(
                    {
                        "snapshot_date": snapshot_date,
                        "ticker": ticker,
                        "sector": "Industrials",
                        "model_name": AUDIT_MODEL,
                        "predicted_alpha": score,
                        TARGET_COLUMN: target,
                    }
                )
        gate = ShortlistSelectionGate(enabled=True, quantile=0.0, lookback_sessions=1)

        summaries = acceptance_windows(
            pd.DataFrame(rows),
            service=ShortlistModelService(db_manager=object()),
            gate=gate,
            promotion_top_n=2,
            target_column=TARGET_COLUMN,
        )

        full = summaries[summaries["model"].eq(f"{AUDIT_MODEL}_full_oos")].iloc[0]
        self.assertEqual(full["dates"], 1)
        self.assertEqual(full["total_dates"], 2)
        self.assertEqual(full["active_dates"], 1)
        self.assertEqual(full["empty_gated_dates"], 1)

    def test_floor_verdict_rows_include_full_and_fold_floors(self) -> None:
        summaries = pd.DataFrame(
            [
                {
                    "model": f"{AUDIT_MODEL}_last_fold",
                    "hit_rate_excess": 0.03,
                    "beat_universe_rate": 0.55,
                    "mean_target_excess": 0.01,
                    "spearman": 0.04,
                    "top_ticker_date_rate": 0.10,
                },
                {
                    "model": f"{AUDIT_MODEL}_trailing_3folds",
                    "hit_rate_excess": 0.03,
                    "beat_universe_rate": 0.49,
                    "mean_target_excess": 0.01,
                    "spearman": 0.04,
                    "top_ticker_date_rate": 0.10,
                },
                {"model": f"{AUDIT_MODEL}_full_oos", "spearman": 0.02},
            ]
        )
        promotion_gate = {
            "min_recent_1fold_hit_rate_excess": 0.02,
            "min_recent_1fold_beat_universe_rate": 0.50,
            "min_recent_1fold_mean_target_excess": 0.0,
            "min_recent_1fold_spearman": 0.0,
            "max_recent_1fold_top_ticker_date_rate": 0.40,
            "min_recent_3fold_hit_rate_excess": 0.02,
            "min_recent_3fold_beat_universe_rate": 0.50,
            "min_recent_3fold_mean_target_excess": 0.0,
            "min_recent_3fold_spearman": 0.0,
            "max_recent_3fold_top_ticker_date_rate": 0.40,
            "min_full_oos_spearman": 0.0,
        }

        rows = floor_verdict_rows(summaries, promotion_gate=promotion_gate)

        self.assertEqual(len(rows), 11)
        failing = [row for row in rows if not row["passes"]]
        self.assertEqual(1, len(failing))
        self.assertEqual("beat_universe_rate", failing[0]["metric"])
        self.assertEqual("trailing_3folds", failing[0]["window"])

    def test_render_report_lists_top_two_and_gate_qualified_count(self) -> None:
        gate = ShortlistSelectionGate(enabled=False, quantile=0.95, lookback_sessions=126)
        fixed_gate = ShortlistSelectionGate(enabled=True, quantile=0.95, lookback_sessions=126)
        oos = pd.DataFrame(
            [
                {"snapshot_date": pd.Timestamp("2026-01-01"), "ticker": "AAA", "predicted_alpha": 0.1},
                {"snapshot_date": pd.Timestamp("2026-01-02"), "ticker": "BBB", "predicted_alpha": 0.2},
            ]
        )
        summaries = pd.DataFrame(
            [
                {
                    "model": f"{AUDIT_MODEL}_full_oos",
                    "dates": 2,
                    "active_dates": 2,
                    "empty_gated_dates": 0,
                    "avg_pick_count": 2.0,
                    "hit_rate_excess": 0.03,
                    "beat_universe_rate": 0.55,
                    "mean_target_excess": 0.01,
                    "spearman": 0.02,
                    "top_ticker": "AAA",
                    "top_ticker_date_rate": 0.1,
                }
            ]
        )
        live = pd.DataFrame(
            [
                {"snapshot_date": pd.Timestamp("2026-10-07"), "ticker": "AAA", "sector": "Tech", "predicted_alpha": 0.5},
                {"snapshot_date": pd.Timestamp("2026-10-07"), "ticker": "BBB", "sector": "Health Care", "predicted_alpha": 0.4},
                {"snapshot_date": pd.Timestamp("2026-10-07"), "ticker": "CCC", "sector": "Energy", "predicted_alpha": 0.3},
            ]
        )

        report = render_report(
            gate=gate,
            fixed_gate=fixed_gate,
            raw_oos=oos,
            oos=oos,
            summaries=summaries,
            floor_rows=[],
            passes=True,
            live=live,
            gated_live=live,
            live_threshold=0.25,
            before_gated_live=live,
            before_live_threshold=0.10,
            calendar_dates=[],
        )

        self.assertIn("- fixed_gate_qualified_live_count: 3", report)
        self.assertIn("| 1 | AAA | Tech | +0.500000 | +0.250000 |", report)
        self.assertIn("| 2 | BBB | Health Care | +0.400000 | +0.250000 |", report)
        self.assertIn("- runtime_top_n_after_gate: 2", report)

    def test_prepare_oos_predictions_deoverlaps_by_label_horizon(self) -> None:
        raw = pd.DataFrame(
            [
                {"snapshot_date": pd.Timestamp("2026-01-01"), "ticker": "AAA", "predicted_alpha": 1.0},
                {"snapshot_date": pd.Timestamp("2026-01-02"), "ticker": "AAA", "predicted_alpha": 2.0},
                {"snapshot_date": pd.Timestamp("2026-03-02"), "ticker": "AAA", "predicted_alpha": 3.0},
                {"snapshot_date": pd.Timestamp("2026-01-02"), "ticker": "BBB", "predicted_alpha": 4.0},
            ]
        )

        prepared = prepare_oos_predictions(
            raw,
            calendar_dates=pd.date_range("2026-01-01", periods=90, freq="D").tolist(),
        )

        self.assertEqual(
            [
                ("AAA", pd.Timestamp("2026-01-01"), 1.0),
                ("AAA", pd.Timestamp("2026-03-02"), 3.0),
                ("BBB", pd.Timestamp("2026-01-02"), 4.0),
            ],
            [
                (row.ticker, row.snapshot_date, row.predicted_alpha)
                for row in prepared.sort_values(["ticker", "snapshot_date"]).itertuples(index=False)
            ],
        )


if __name__ == "__main__":
    unittest.main()
