from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from src.cli import build_parser
from src.research.universe_snapshot_service import UniverseSnapshotBackfillService
from src.utils.strategy import ExitRules, ProductionStrategy


class PathLabelBackfillTests(unittest.TestCase):
    def test_parser_accepts_path_label_backfill_args(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "path-label-backfill",
                "--horizon",
                "60",
                "--date-from",
                "2026-01-02",
                "--date-to",
                "2026-03-31",
                "--batch-size-dates",
                "5",
            ]
        )

        self.assertEqual(args.command, "path-label-backfill")
        self.assertEqual(args.horizon, 60)
        self.assertEqual(args.date_from, "2026-01-02")
        self.assertEqual(args.date_to, "2026-03-31")
        self.assertEqual(args.batch_size_dates, 5)

    def test_path_label_backfill_updates_only_missing_path_label_columns(self) -> None:
        class FakeDB:
            def __init__(self) -> None:
                self.updated_rows = []
                self.history_calls = []

            def initialize(self): return None

            def list_missing_universe_path_label_dates(self, *, horizon_days, date_from, date_to):
                self.requested_dates = (horizon_days, date_from, date_to)
                return ["2026-05-01"]

            def load_universe_path_label_inputs(self, *, horizon_days, date_from, date_to):
                self.requested_inputs = (horizon_days, date_from, date_to)
                return pd.DataFrame(
                    [
                        {
                            "snapshot_date": "2026-05-01",
                            "ticker": "AAA",
                            "sector": "Industrials",
                            "passed_slots_json": '["industrials"]',
                        }
                    ]
                )

            def load_price_history(self, tickers, date_from=None, date_to=None):
                self.history_calls.append((tuple(sorted(tickers)), date_from, date_to))
                return pd.DataFrame(
                    [
                        {"ticker": "AAA", "date": pd.Timestamp("2026-05-01"), "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1_000_000, "adj_close": 100.0},
                        {"ticker": "AAA", "date": pd.Timestamp("2026-05-04"), "open": 96.0, "high": 96.0, "low": 93.0, "close": 94.0, "volume": 1_000_000, "adj_close": 94.0},
                        {"ticker": "XLI", "date": pd.Timestamp("2026-05-01"), "open": 50.0, "high": 50.0, "low": 50.0, "close": 50.0, "volume": 1_000_000, "adj_close": 50.0},
                        {"ticker": "XLI", "date": pd.Timestamp("2026-05-04"), "open": 51.0, "high": 51.0, "low": 51.0, "close": 51.0, "volume": 1_000_000, "adj_close": 51.0},
                    ]
                )

            def update_universe_path_labels(self, *, horizon_days, rows):
                self.updated_horizon = horizon_days
                self.updated_rows.extend(rows)
                return len(rows)

        strategy = ProductionStrategy(
            strategy_id=1,
            promoted_at="2026-04-01T17:00:00",
            indicators={},
            exit_rules=ExitRules(
                trailing_stop_pct=None,
                profit_target_pct=None,
                time_limit_days=20,
                hard_stop_pct=0.05,
            ),
            slot="industrials",
            sector="Industrials",
        )
        db = FakeDB()

        with patch("src.research.universe_snapshot_service.load_active_strategies", return_value={"industrials": strategy}):
            report = UniverseSnapshotBackfillService(db).run_path_label_backfill(
                horizon_days=60,
                date_from="2026-05-01",
                date_to="2026-05-01",
                batch_size_dates=1,
            )

        self.assertEqual(report.snapshot_dates_processed, 1)
        self.assertEqual(report.total_rows, 1)
        self.assertEqual(report.updated_rows, 1)
        self.assertEqual(report.unavailable_rows, 0)
        self.assertEqual(db.updated_horizon, 60)
        self.assertIn("XLI", db.history_calls[0][0])
        self.assertEqual(db.history_calls[0][1], "2026-05-01")
        row = db.updated_rows[0]
        self.assertEqual(row["snapshot_date"], "2026-05-01")
        self.assertEqual(row["ticker"], "AAA")
        self.assertAlmostEqual(row["path_return_60d"], -0.05)
        self.assertEqual(row["path_exit_reason_60d"], "hard_stop")
        self.assertEqual(row["path_holding_days_60d"], 1)
        self.assertAlmostEqual(row["path_alpha_vs_sector_60d"], -0.07)


if __name__ == "__main__":
    unittest.main()
