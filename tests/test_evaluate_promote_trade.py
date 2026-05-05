from __future__ import annotations

from pathlib import Path
import json
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from src.evaluate.service import EvaluateService
from src.promote.service import PromoteService
from src.settings import AppPaths
from src.trade.service import TradeService
from src.utils.db_manager import BacktestResultRow, DatabaseManager


class FakeDuckDBConnection:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None

    def execute(self, statement: str):
        return self


class EvaluateServiceTests(unittest.TestCase):
    def test_evaluate_applies_min_max_normalization_before_scoring(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = AppPaths(
                root_dir=root,
                data_dir=root / "data",
                duckdb_path=root / "data" / "market_data.duckdb",
                sqlite_path=root / "data" / "ledger.sqlite",
                reports_dir=root / "reports",
                logs_dir=root / "logs",
                config_path=root / "config.yaml",
                env_path=root / ".env",
                production_strategy_path=root / "production_strategy.json",
            )
            db = DatabaseManager(paths)
            with patch.object(db, "duckdb_connection", return_value=FakeDuckDBConnection()):
                db.initialize()
            db.insert_backtest_results(
                [
                    BacktestResultRow(
                        strategy_id=1,
                        params_json=json.dumps({"indicators": {}, "exit_rules": {}, "sector": "ALL"}),
                        norm_score=None,
                        profit_factor=1.0,
                        expectancy=0.1,
                        mdd=0.2,
                        win_rate=0.5,
                    ),
                    BacktestResultRow(
                        strategy_id=2,
                        params_json=json.dumps({"indicators": {}, "exit_rules": {}, "sector": "ALL"}),
                        norm_score=None,
                        profit_factor=2.0,
                        expectancy=0.3,
                        mdd=0.1,
                        win_rate=0.6,
                    ),
                ]
            )
            service = EvaluateService(db)
            service._build_candidate_links = lambda frame: {}

            with patch.object(db, "duckdb_connection", return_value=FakeDuckDBConnection()):
                report = service.run(top=2)

            self.assertEqual(report.rows_written, 2)
            rows = db.list_backtest_results()
            scores = {row["strategy_id"]: row["norm_score"] for row in rows}
            self.assertAlmostEqual(scores[1], -0.3)
            self.assertAlmostEqual(scores[2], 0.7)
            report_text = (paths.reports_dir / "candidates.md").read_text(encoding="utf-8")
            self.assertIn("norm_score: 0.700000", report_text)


class PromoteAndTradeTests(unittest.TestCase):
    def test_promote_writes_fully_formed_production_strategy(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = AppPaths(
                root_dir=root,
                data_dir=root / "data",
                duckdb_path=root / "data" / "market_data.duckdb",
                sqlite_path=root / "data" / "ledger.sqlite",
                reports_dir=root / "reports",
                logs_dir=root / "logs",
                config_path=root / "config.yaml",
                env_path=root / ".env",
                production_strategy_path=root / "production_strategy.json",
            )
            db = DatabaseManager(paths)
            with patch.object(db, "duckdb_connection", return_value=FakeDuckDBConnection()):
                db.initialize()
            db.insert_backtest_results(
                [
                    BacktestResultRow(
                        strategy_id=402,
                        params_json=json.dumps(
                            {
                                "indicators": {"rsi_14_max": 35, "vol_alpha_min": 1.5},
                                "exit_rules": {
                                    "trailing_stop_pct": 0.05,
                                    "profit_target_pct": 0.12,
                                    "time_limit_days": 20,
                                },
                                "sector": "ALL",
                            }
                        ),
                        norm_score=0.7,
                        profit_factor=1.8,
                        expectancy=0.15,
                        mdd=0.12,
                        win_rate=0.55,
                    )
                ]
            )

            with patch.object(db, "duckdb_connection", return_value=FakeDuckDBConnection()):
                message = PromoteService(db).run(row_id=1)

            payload = json.loads(paths.production_strategy_path.read_text(encoding="utf-8"))
            self.assertEqual(message, "Strategy 1 promoted. production_strategy.json updated.")
            self.assertEqual(payload["strategy_id"], 402)
            self.assertIn("promoted_at", payload)
            self.assertEqual(payload["indicators"]["rsi_14_max"], 35.0)
            self.assertEqual(payload["exit_rules"]["time_limit_days"], 20)

    def test_trade_sell_updates_status_exit_fields_and_reports_pnl(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = AppPaths(
                root_dir=root,
                data_dir=root / "data",
                duckdb_path=root / "data" / "market_data.duckdb",
                sqlite_path=root / "data" / "ledger.sqlite",
                reports_dir=root / "reports",
                logs_dir=root / "logs",
                config_path=root / "config.yaml",
                env_path=root / ".env",
                production_strategy_path=root / "production_strategy.json",
            )
            db = DatabaseManager(paths)
            with patch.object(db, "duckdb_connection", return_value=FakeDuckDBConnection()):
                db.initialize()
            db.open_trade(
                ticker="AAA",
                entry_date="2026-05-01",
                entry_price=100.0,
                shares=10,
                max_price_seen=100.0,
            )

            with patch.object(db, "duckdb_connection", return_value=FakeDuckDBConnection()):
                message = TradeService(db).sell(ticker="AAA", price=112.5)

            self.assertEqual(message, "Realized P&L for AAA: 125.00")
            connection = sqlite3.connect(paths.sqlite_path)
            connection.row_factory = sqlite3.Row
            try:
                row = connection.execute(
                    "SELECT status, exit_date, exit_price FROM Active_Trades WHERE ticker = 'AAA'"
                ).fetchone()
            finally:
                connection.close()
            self.assertEqual(row["status"], "closed")
            self.assertEqual(row["exit_price"], 112.5)
            self.assertIsNotNone(row["exit_date"])
