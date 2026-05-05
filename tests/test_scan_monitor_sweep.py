from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import unittest
from unittest.mock import patch

import pandas as pd

from src.scan.service import ScanService
from src.settings import AppPaths, RuntimeSettings
from src.sweep.service import SweepService
from src.utils.db_manager import DatabaseManager
from src.utils.strategy import ExitRules, ProductionStrategy
from src.monitor.service import MonitorService


@dataclass
class EmailCall:
    subject: str
    html_body: str


class ScanServiceTests(unittest.TestCase):
    def test_scan_sizes_using_adjusted_close(self) -> None:
        class FakeDB:
            def initialize(self): return None
            def list_universe_rows(self, active_only=True): return [{"ticker": "AAA", "sector": "Industrials", "md_volume_30d": 30_000_000}]
            def load_price_history(self, tickers): return pd.DataFrame()

        email_calls: list[EmailCall] = []
        service = ScanService(FakeDB(), email_sender=lambda subject, html_body, settings: email_calls.append(EmailCall(subject, html_body)))
        snapshot = pd.DataFrame(
            [
                {
                    "ticker": "AAA",
                    "sector": "Industrials",
                    "regime_green": True,
                    "regime_etf": "SPY",
                    "close": 100.0,
                    "adj_close": 80.0,
                    "md_volume_30d": 30_000_000,
                }
            ]
        )
        settings = RuntimeSettings(
            paths=AppPaths(
                root_dir=Path("."),
                data_dir=Path("data"),
                duckdb_path=Path("data/market_data.duckdb"),
                sqlite_path=Path("data/ledger.sqlite"),
                reports_dir=Path("reports"),
                logs_dir=Path("logs"),
                config_path=Path("config.yaml"),
                env_path=Path(".env"),
                production_strategy_path=Path("production_strategy.json"),
            ),
            env={},
            total_capital=50_000.0,
            risk_per_trade=0.02,
        )
        strategy = ProductionStrategy(
            strategy_id=1,
            promoted_at="2026-05-05T17:00:00",
            indicators={"rsi_14_max": 35.0},
            exit_rules=ExitRules(0.05, 0.12, 20),
        )

        with patch.object(service, "_download_recent_daily_history", return_value=pd.DataFrame()), \
             patch("src.scan.service.build_analysis_frame", return_value=(pd.DataFrame(), [])), \
             patch("src.scan.service.latest_snapshot", return_value=snapshot), \
             patch("src.scan.service.filter_signal_candidates", return_value=snapshot.copy()), \
             patch("src.scan.service.get_settings", return_value=settings), \
             patch("src.scan.service.load_active_strategy", return_value=strategy):
            report = service.run()

        self.assertTrue(report.emailed)
        self.assertEqual(report.candidate_count, 1)
        self.assertEqual(len(email_calls), 1)
        self.assertIn("<td>250</td>", email_calls[0].html_body)


class MonitorServiceTests(unittest.TestCase):
    def test_monitor_sends_single_consolidated_digest_and_evaluates_exit_paths(self) -> None:
        class FakeDB:
            def __init__(self):
                self.closed = []
                self.max_updates = []
            def initialize(self): return None
            def list_open_trades(self):
                return [
                    {"ticker": "AAA", "entry_date": "2026-04-01", "entry_price": 100.0, "shares": 10, "max_price_seen": 110.0, "status": "open"},
                    {"ticker": "BBB", "entry_date": "2026-04-01", "entry_price": 100.0, "shares": 10, "max_price_seen": 100.0, "status": "open"},
                    {"ticker": "CCC", "entry_date": "2026-04-01", "entry_price": 100.0, "shares": 10, "max_price_seen": 100.0, "status": "open"},
                    {"ticker": "DDD", "entry_date": "2026-03-01", "entry_price": 100.0, "shares": 10, "max_price_seen": 100.0, "status": "open"},
                    {"ticker": "EEE", "entry_date": "2026-05-01", "entry_price": 100.0, "shares": 10, "max_price_seen": 100.0, "status": "open"},
                ]
            def load_price_history(self, tickers):
                rows = []
                for ticker in tickers:
                    for day in pd.bdate_range("2025-06-01", periods=220):
                        rows.append(
                            {
                                "ticker": ticker,
                                "date": day.date(),
                                "open": 100.0,
                                "high": 101.0,
                                "low": 99.0,
                                "close": 100.0,
                                "volume": 1000,
                                "adj_close": 100.0,
                            }
                        )
                return pd.DataFrame(rows)
            def list_universe_rows(self, active_only=False):
                return [
                    {"ticker": "AAA", "sector": "Industrials", "md_volume_30d": 30_000_000},
                    {"ticker": "BBB", "sector": "Industrials", "md_volume_30d": 30_000_000},
                    {"ticker": "CCC", "sector": "Industrials", "md_volume_30d": 30_000_000},
                    {"ticker": "DDD", "sector": "Industrials", "md_volume_30d": 30_000_000},
                    {"ticker": "EEE", "sector": "Information Technology", "md_volume_30d": 30_000_000},
                ]
            def get_latest_open_trade(self, ticker):
                return {"rowid": {"AAA": 1, "BBB": 2, "CCC": 3, "DDD": 4, "EEE": 5}[ticker], "entry_price": 100.0, "shares": 10}
            def update_trade_max_price(self, trade_rowid, max_price_seen):
                self.max_updates.append((trade_rowid, max_price_seen))
            def close_trade(self, trade_rowid, exit_date, exit_price):
                self.closed.append((trade_rowid, exit_price))
            def load_recent_highs(self, ticker, limit=2):
                return pd.DataFrame([{"date": "2026-05-05", "high": 105.0}, {"date": "2026-05-04", "high": 100.0}])

        email_calls: list[EmailCall] = []
        service = MonitorService(FakeDB(), email_sender=lambda subject, html_body, settings: email_calls.append(EmailCall(subject, html_body)))
        settings = RuntimeSettings(
            paths=AppPaths(
                root_dir=Path("."),
                data_dir=Path("data"),
                duckdb_path=Path("data/market_data.duckdb"),
                sqlite_path=Path("data/ledger.sqlite"),
                reports_dir=Path("reports"),
                logs_dir=Path("logs"),
                config_path=Path("config.yaml"),
                env_path=Path(".env"),
                production_strategy_path=Path("production_strategy.json"),
            ),
            env={},
            total_capital=50_000.0,
            risk_per_trade=0.02,
        )
        strategy = ProductionStrategy(
            strategy_id=1,
            promoted_at="2026-05-05T17:00:00",
            indicators={"rsi_14_max": 35.0},
            exit_rules=ExitRules(0.05, 0.12, 20),
        )
        analysis_frame = pd.DataFrame(
            [
                {"ticker": "SPY", "date": pd.Timestamp("2026-05-05"), "spy_sma_200": 100.0, "qqq_sma_200": None},
                {"ticker": "QQQ", "date": pd.Timestamp("2026-05-05"), "spy_sma_200": None, "qqq_sma_200": 120.0},
            ]
        )

        with patch.object(service, "_load_intraday_last_prices", return_value={
            "AAA": 103.0,
            "BBB": 115.0,
            "CCC": 101.0,
            "DDD": 102.0,
            "EEE": 110.0,
            "SPY": 105.0,
            "QQQ": 110.0,
        }), \
             patch("src.monitor.service.get_settings", return_value=settings), \
             patch("src.monitor.service.load_active_strategy", return_value=strategy), \
             patch("src.monitor.service.build_analysis_frame", return_value=(analysis_frame, [])), \
             patch("src.monitor.service.latest_rsi_2_with_intraday", side_effect=lambda price_history, ticker, current_price, as_of: 95.0 if ticker == "CCC" else 20.0):
            report = service.run()

        self.assertTrue(report.emailed)
        self.assertEqual(report.triggered_count, 5)
        self.assertEqual(len(email_calls), 1)
        self.assertEqual(len(service.db_manager.closed), 5)


class SweepServiceTests(unittest.TestCase):
    def test_sweep_uses_polars_engine_not_vectorbt(self) -> None:
        service = SweepService(DatabaseManager(AppPaths(
            root_dir=Path("."),
            data_dir=Path("data"),
            duckdb_path=Path("data/market_data.duckdb"),
            sqlite_path=Path("data/ledger.sqlite"),
            reports_dir=Path("reports"),
            logs_dir=Path("logs"),
            config_path=Path("config.yaml"),
            env_path=Path(".env"),
            production_strategy_path=Path("production_strategy.json"),
        )))
        source = Path("src/sweep/service.py").read_text(encoding="utf-8")
        self.assertIn("import polars as pl", source)
        self.assertNotIn("vectorbt", source.lower())
        grid = service._build_parameter_grid(
            {
                "sweep_grid": {
                    "rsi_14_max": {"min": 25, "max": 35, "step": 5},
                    "vol_alpha_min": {"min": 1.0, "max": 1.5, "step": 0.5},
                }
            }
        )
        self.assertEqual(len(grid), 6)
