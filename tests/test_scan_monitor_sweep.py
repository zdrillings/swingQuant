from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import unittest
from unittest.mock import patch

import pandas as pd

from src.scan.service import ScanService
from src.settings import AppPaths, RuntimeSettings
from src.sweep.service import SweepService, _optional_finite_float
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
                    "atr_14": 4.0,
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
            slot="materials",
            sector="Industrials",
        )

        with patch.object(service, "_download_recent_daily_history", return_value=pd.DataFrame()), \
             patch("src.scan.service.build_analysis_frame", return_value=(pd.DataFrame(), [])), \
             patch("src.scan.service.latest_snapshot", return_value=snapshot), \
             patch("src.scan.service.filter_signal_candidates", return_value=snapshot.copy()), \
             patch("src.scan.service.get_settings", return_value=settings), \
             patch("src.scan.service.load_feature_config", return_value={"scan_policy": {"max_candidates_total": 6, "max_candidates_per_slot": 3, "max_candidates_per_sector": 2, "pre_cap_candidates_per_slot": 5}}), \
             patch("src.scan.service.load_active_strategies", return_value={"default": strategy}):
            report = service.run()

        self.assertTrue(report.emailed)
        self.assertEqual(report.candidate_count, 1)
        self.assertEqual(len(email_calls), 1)
        self.assertIn("Slot: default", email_calls[0].html_body)
        self.assertIn("<td>250</td>", email_calls[0].html_body)

    def test_scan_groups_output_by_slot_and_applies_portfolio_caps(self) -> None:
        class FakeDB:
            def initialize(self): return None
            def list_universe_rows(self, active_only=True):
                return [
                    {"ticker": "MAT1", "sector": "Materials", "md_volume_30d": 40_000_000},
                    {"ticker": "MAT2", "sector": "Materials", "md_volume_30d": 35_000_000},
                    {"ticker": "MAT3", "sector": "Materials", "md_volume_30d": 30_000_000},
                    {"ticker": "TECH1", "sector": "Information Technology", "md_volume_30d": 45_000_000},
                    {"ticker": "TECH2", "sector": "Information Technology", "md_volume_30d": 42_000_000},
                ]
            def load_price_history(self, tickers): return pd.DataFrame()

        email_calls: list[EmailCall] = []
        service = ScanService(FakeDB(), email_sender=lambda subject, html_body, settings: email_calls.append(EmailCall(subject, html_body)))
        snapshot = pd.DataFrame(
            [
                {"ticker": "MAT1", "sector": "Materials", "regime_green": True, "regime_etf": "SPY", "adj_close": 50.0, "atr_14": 2.0, "md_volume_30d": 40_000_000, "signal_score": 38.0},
                {"ticker": "MAT2", "sector": "Materials", "regime_green": True, "regime_etf": "SPY", "adj_close": 48.0, "atr_14": 2.0, "md_volume_30d": 35_000_000, "signal_score": 36.0},
                {"ticker": "MAT3", "sector": "Materials", "regime_green": True, "regime_etf": "SPY", "adj_close": 47.0, "atr_14": 2.0, "md_volume_30d": 30_000_000, "signal_score": 34.0},
                {"ticker": "TECH1", "sector": "Information Technology", "regime_green": True, "regime_etf": "QQQ", "adj_close": 100.0, "atr_14": 4.0, "md_volume_30d": 45_000_000, "signal_score": 37.0},
                {"ticker": "TECH2", "sector": "Information Technology", "regime_green": True, "regime_etf": "QQQ", "adj_close": 95.0, "atr_14": 4.0, "md_volume_30d": 42_000_000, "signal_score": 35.0},
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
        materials = ProductionStrategy(
            strategy_id=1,
            promoted_at="2026-05-05T17:00:00",
            indicators={"rsi_14_max": 35.0},
            exit_rules=ExitRules(0.05, 0.12, 20),
            slot="materials",
            sector="Materials",
        )
        technology = ProductionStrategy(
            strategy_id=2,
            promoted_at="2026-05-05T17:00:00",
            indicators={"rsi_14_max": 35.0},
            exit_rules=ExitRules(0.05, 0.12, 20),
            slot="technology",
            sector="Information Technology",
        )

        with patch.object(service, "_download_recent_daily_history", return_value=pd.DataFrame()), \
             patch("src.scan.service.build_analysis_frame", return_value=(pd.DataFrame(), [])), \
             patch("src.scan.service.latest_snapshot", return_value=snapshot), \
             patch("src.scan.service.filter_signal_candidates", side_effect=lambda frame, indicators: frame.copy()), \
             patch("src.scan.service.get_settings", return_value=settings), \
             patch("src.scan.service.load_feature_config", return_value={"scan_policy": {"max_candidates_total": 3, "max_candidates_per_slot": 2, "max_candidates_per_sector": 2, "pre_cap_candidates_per_slot": 5}}), \
             patch("src.scan.service.load_active_strategies", return_value={"materials": materials, "technology": technology}):
            report = service.run()

        self.assertTrue(report.emailed)
        self.assertEqual(report.candidate_count, 3)
        self.assertEqual(len(email_calls), 1)
        html = email_calls[0].html_body
        self.assertIn("Slot: materials", html)
        self.assertIn("Slot: technology", html)
        self.assertIn("Portfolio caps: total=3, per_slot=2, per_sector=2", html)
        self.assertIn("MAT1", html)
        self.assertIn("MAT2", html)
        self.assertNotIn("MAT3", html)
        self.assertIn("TECH1", html)


class MonitorServiceTests(unittest.TestCase):
    def test_monitor_backfills_legacy_trade_strategy_using_regime_family_fallback(self) -> None:
        class FakeDB:
            def __init__(self):
                self.assigned = []
            def initialize(self): return None
            def list_open_trades(self):
                return [
                    {
                        "ticker": "AMZN",
                        "entry_date": "2026-05-05",
                        "entry_price": 100.0,
                        "shares": 10,
                        "max_price_seen": 100.0,
                        "status": "open",
                        "entry_atr": None,
                        "strategy_id": None,
                        "strategy_slot": None,
                    }
                ]
            def load_price_history(self, tickers):
                rows = []
                for ticker in ("AMZN", "SPY", "QQQ"):
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
                return [{"ticker": "AMZN", "sector": "Consumer Discretionary", "md_volume_30d": 30_000_000}]
            def get_latest_open_trade(self, ticker):
                return {"rowid": 7, "entry_price": 100.0, "entry_atr": None, "shares": 10, "strategy_id": None, "strategy_slot": None}
            def assign_trade_strategy(self, trade_rowid, *, strategy_id, strategy_slot):
                self.assigned.append((trade_rowid, strategy_id, strategy_slot))
            def update_trade_max_price(self, trade_rowid, max_price_seen): return None
            def close_trade(self, trade_rowid, exit_date, exit_price): return None
            def load_recent_highs(self, ticker, limit=2):
                return pd.DataFrame([{"date": "2026-05-05", "high": 105.0}, {"date": "2026-05-04", "high": 105.0}])

        service = MonitorService(FakeDB(), email_sender=lambda subject, html_body, settings: None)
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
        materials = ProductionStrategy(
            strategy_id=10,
            promoted_at="2026-05-05T17:00:00",
            indicators={"rsi_14_max": 35.0},
            exit_rules=ExitRules(0.05, 0.12, 20),
            slot="materials",
            sector="Materials",
        )
        technology = ProductionStrategy(
            strategy_id=11,
            promoted_at="2026-05-05T17:00:00",
            indicators={"rsi_14_max": 35.0},
            exit_rules=ExitRules(0.05, 0.12, 20),
            slot="technology",
            sector="Information Technology",
        )
        analysis_frame = pd.DataFrame(
            [
                {"ticker": "SPY", "date": pd.Timestamp("2026-05-05"), "spy_sma_200": 100.0, "qqq_sma_200": None},
                {"ticker": "QQQ", "date": pd.Timestamp("2026-05-05"), "spy_sma_200": None, "qqq_sma_200": 100.0},
                {"ticker": "AMZN", "date": pd.Timestamp("2026-05-05"), "atr_14": 4.0},
            ]
        )

        with patch.object(service, "_load_intraday_last_prices", return_value={"AMZN": 101.0, "SPY": 105.0, "QQQ": 110.0}), \
             patch.object(service, "_download_recent_daily_history", return_value=pd.DataFrame()), \
             patch("src.monitor.service.get_settings", return_value=settings), \
             patch("src.monitor.service.load_active_strategies", return_value={"materials": materials, "technology": technology}), \
             patch("src.monitor.service.build_analysis_frame", return_value=(analysis_frame, [])), \
             patch("src.monitor.service.latest_rsi_2_with_intraday", return_value=20.0):
            report = service.run()

        self.assertFalse(report.emailed)
        self.assertEqual(report.watchlist_size, 1)
        self.assertEqual(service.db_manager.assigned, [(7, 10, "materials")])

    def test_monitor_fetches_recent_history_for_open_trade_missing_from_duckdb(self) -> None:
        class FakeDB:
            def initialize(self): return None
            def list_open_trades(self):
                return [{"ticker": "NBIS", "entry_date": "2026-05-05", "entry_price": 100.0, "shares": 10, "max_price_seen": 100.0, "status": "open", "entry_atr": None}]
            def load_price_history(self, tickers):
                rows = []
                for ticker in ("SPY", "QQQ"):
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
                return []
            def get_latest_open_trade(self, ticker):
                return {"rowid": 1, "entry_price": 100.0, "entry_atr": None, "shares": 10}
            def update_trade_max_price(self, trade_rowid, max_price_seen): return None
            def close_trade(self, trade_rowid, exit_date, exit_price): return None
            def load_recent_highs(self, ticker, limit=2):
                return pd.DataFrame([{"date": "2026-05-05", "high": 100.0}, {"date": "2026-05-04", "high": 100.0}])

        service = MonitorService(FakeDB(), email_sender=lambda subject, html_body, settings: None)
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
        recent_history = pd.DataFrame(
            [
                {
                    "ticker": "NBIS",
                    "date": day.date(),
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.0,
                    "volume": 1000,
                    "adj_close": 100.0,
                }
                for day in pd.bdate_range("2025-06-01", periods=220)
            ]
        )

        with patch.object(service, "_load_intraday_last_prices", return_value={"NBIS": 100.0, "SPY": 105.0, "QQQ": 110.0}), \
             patch.object(service, "_download_recent_daily_history", return_value=recent_history), \
             patch("src.monitor.service.get_settings", return_value=settings), \
             patch("src.monitor.service.load_active_strategies", return_value={"default": strategy}), \
             patch("src.monitor.service.latest_rsi_2_with_intraday", return_value=20.0):
            report = service.run()

        self.assertFalse(report.emailed)
        self.assertEqual(report.watchlist_size, 1)
        self.assertEqual(report.triggered_count, 0)

    def test_monitor_sends_single_consolidated_digest_and_evaluates_exit_paths_without_closing_trades(self) -> None:
        class FakeDB:
            def __init__(self):
                self.closed = []
                self.max_updates = []
            def initialize(self): return None
            def list_open_trades(self):
                return [
                    {"ticker": "AAA", "entry_date": "2026-04-01", "entry_price": 100.0, "shares": 10, "max_price_seen": 110.0, "status": "open", "entry_atr": None},
                    {"ticker": "BBB", "entry_date": "2026-04-01", "entry_price": 100.0, "shares": 10, "max_price_seen": 100.0, "status": "open", "entry_atr": None},
                    {"ticker": "CCC", "entry_date": "2026-04-01", "entry_price": 100.0, "shares": 10, "max_price_seen": 100.0, "status": "open", "entry_atr": None},
                    {"ticker": "DDD", "entry_date": "2026-03-01", "entry_price": 100.0, "shares": 10, "max_price_seen": 100.0, "status": "open", "entry_atr": None},
                    {"ticker": "EEE", "entry_date": "2026-05-01", "entry_price": 100.0, "shares": 10, "max_price_seen": 100.0, "status": "open", "entry_atr": None},
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
             patch("src.monitor.service.load_active_strategies", return_value={"industrials": strategy, "information_technology": ProductionStrategy(strategy_id=2, promoted_at='2026-05-05T17:00:00', indicators={'rsi_14_max': 35.0}, exit_rules=ExitRules(0.05, 0.12, 20), slot='information_technology', sector='Information Technology')}), \
             patch("src.monitor.service.build_analysis_frame", return_value=(analysis_frame, [])), \
             patch("src.monitor.service.latest_rsi_2_with_intraday", side_effect=lambda price_history, ticker, current_price, as_of: 95.0 if ticker == "CCC" else 20.0):
            report = service.run()

        self.assertTrue(report.emailed)
        self.assertEqual(report.triggered_count, 5)
        self.assertEqual(len(email_calls), 1)
        self.assertEqual(len(service.db_manager.closed), 0)
        self.assertIn("Recommended Action", email_calls[0].html_body)
        self.assertIn("<td>sell</td>", email_calls[0].html_body)

    def test_monitor_uses_atr_exit_thresholds_when_promoted(self) -> None:
        class FakeDB:
            def initialize(self): return None
            def list_open_trades(self):
                return [
                    {
                        "ticker": "AAA",
                        "entry_date": "2026-04-01",
                        "entry_price": 100.0,
                        "entry_atr": 4.0,
                        "shares": 10,
                        "max_price_seen": 100.0,
                        "status": "open",
                    }
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
                return [{"ticker": "AAA", "sector": "Industrials", "md_volume_30d": 30_000_000}]
            def get_latest_open_trade(self, ticker):
                return {"rowid": 1, "entry_price": 100.0, "entry_atr": 4.0, "shares": 10}
            def update_trade_max_price(self, trade_rowid, max_price_seen): return None
            def close_trade(self, trade_rowid, exit_date, exit_price): return None
            def load_recent_highs(self, ticker, limit=2):
                return pd.DataFrame([{"date": "2026-05-05", "high": 100.0}, {"date": "2026-05-04", "high": 100.0}])

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
            exit_rules=ExitRules(
                trailing_stop_pct=None,
                profit_target_pct=None,
                time_limit_days=20,
                trailing_stop_atr_mult=2.0,
                profit_target_atr_mult=3.0,
            ),
        )
        analysis_frame = pd.DataFrame(
            [{"ticker": "SPY", "date": pd.Timestamp("2026-05-05"), "spy_sma_200": 100.0, "qqq_sma_200": None}]
        )

        with patch.object(service, "_load_intraday_last_prices", return_value={"AAA": 91.0, "SPY": 105.0, "QQQ": 110.0}), \
             patch("src.monitor.service.get_settings", return_value=settings), \
             patch("src.monitor.service.load_active_strategies", return_value={"default": strategy}), \
             patch("src.monitor.service.build_analysis_frame", return_value=(analysis_frame, [])), \
             patch("src.monitor.service.latest_rsi_2_with_intraday", return_value=20.0):
            report = service.run()

        self.assertTrue(report.emailed)
        self.assertEqual(report.triggered_count, 1)
        self.assertIn("trailing stop", email_calls[0].html_body)
        self.assertIn("<td>sell</td>", email_calls[0].html_body)

    def test_monitor_uses_central_regime_helper(self) -> None:
        class FakeDB:
            def initialize(self): return None
            def list_open_trades(self):
                return [{"ticker": "AAA", "entry_date": "2026-04-01", "entry_price": 100.0, "shares": 10, "max_price_seen": 100.0, "status": "open", "entry_atr": None}]
            def load_price_history(self, tickers): return pd.DataFrame()
            def list_universe_rows(self, active_only=False):
                return [{"ticker": "AAA", "sector": "Industrials", "md_volume_30d": 30_000_000}]
            def get_latest_open_trade(self, ticker): return {"rowid": 1, "entry_price": 100.0, "shares": 10}
            def update_trade_max_price(self, trade_rowid, max_price_seen): return None
            def close_trade(self, trade_rowid, exit_date, exit_price): return None
            def load_recent_highs(self, ticker, limit=2):
                return pd.DataFrame([{"date": "2026-05-05", "high": 105.0}, {"date": "2026-05-04", "high": 100.0}])

        service = MonitorService(FakeDB(), email_sender=lambda subject, html_body, settings: None)
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
            [{"ticker": "SPY", "date": pd.Timestamp("2026-05-05"), "spy_sma_200": 100.0, "qqq_sma_200": None}]
        )

        with patch.object(service, "_load_intraday_last_prices", return_value={"AAA": 101.0, "SPY": 105.0, "QQQ": 110.0}), \
             patch("src.monitor.service.get_settings", return_value=settings), \
             patch("src.monitor.service.load_active_strategies", return_value={"default": strategy}), \
             patch("src.monitor.service.build_analysis_frame", return_value=(analysis_frame, [])), \
             patch("src.monitor.service.latest_rsi_2_with_intraday", return_value=20.0), \
             patch("src.monitor.service.regime_etf_for_sector", return_value="SPY") as regime_helper:
            service.run()

        regime_helper.assert_any_call("Industrials")


class SweepServiceTests(unittest.TestCase):
    def test_optional_finite_float_treats_nan_as_missing(self) -> None:
        self.assertIsNone(_optional_finite_float(None))
        self.assertIsNone(_optional_finite_float(float("nan")))
        self.assertEqual(_optional_finite_float(4.5), 4.5)

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
                    "trailing_stop_atr_mult": {"min": 2.0, "max": 3.0, "step": 0.5},
                }
            }
        )
        self.assertEqual(len(grid), 18)
        self.assertEqual(grid[0]["exit_rules"]["trailing_stop_atr_mult"], 2.0)
        self.assertIn("rsi_14_max", grid[0]["indicators"])

    def test_sweep_signal_expression_uses_rs_hard_filter_and_score_threshold(self) -> None:
        try:
            import polars as pl
        except ModuleNotFoundError:
            self.skipTest("polars is not installed in this test environment")

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
        frame = pl.DataFrame(
            [
                {
                    "ticker": "AAA",
                    "rsi_14": 45.0,
                    "vol_alpha": 1.8,
                    "sma_200_dist": 0.18,
                    "roc_63": 0.18,
                    "relative_strength_index_vs_spy": 85.0,
                },
                {
                    "ticker": "BBB",
                    "rsi_14": 45.0,
                    "vol_alpha": 1.8,
                    "sma_200_dist": 0.18,
                    "roc_63": 0.18,
                    "relative_strength_index_vs_spy": 70.0,
                },
                {
                    "ticker": "CCC",
                    "rsi_14": 60.0,
                    "vol_alpha": 1.0,
                    "sma_200_dist": 0.02,
                    "roc_63": 0.01,
                    "relative_strength_index_vs_spy": 90.0,
                },
            ]
        )
        signal_expr = service._build_signal_expression(
            pl,
            {
                "rsi_14_max": 35.0,
                "vol_alpha_min": 1.4,
                "sma_200_dist_min": 0.10,
                "roc_63_min": 0.10,
                "relative_strength_index_vs_spy_min": 80.0,
                "signal_score_min": 30.0,
            },
        )

        signal_values = frame.with_columns(signal=signal_expr)["signal"].to_list()

        self.assertEqual(signal_values, [True, False, False])

    def test_backtest_costs_reduce_net_expectancy(self) -> None:
        try:
            import polars as pl
        except ModuleNotFoundError:
            self.skipTest("polars is not installed in this test environment")

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
        validation_frame = pl.DataFrame(
            [
                {
                    "ticker": "AAA",
                    "date": pd.Timestamp("2026-01-02"),
                    "open": 100.0,
                    "high": 100.0,
                    "low": 100.0,
                    "close": 100.0,
                    "rsi_14": 30.0,
                    "vol_alpha": 2.0,
                    "sma_200_dist": 0.20,
                    "roc_63": 0.20,
                    "relative_strength_index_vs_spy": 90.0,
                    "regime_green": True,
                    "rsi_2": 20.0,
                },
                {
                    "ticker": "AAA",
                    "date": pd.Timestamp("2026-01-03"),
                    "open": 100.0,
                    "high": 101.5,
                    "low": 100.0,
                    "close": 101.0,
                    "rsi_14": 30.0,
                    "vol_alpha": 2.0,
                    "sma_200_dist": 0.20,
                    "roc_63": 0.20,
                    "relative_strength_index_vs_spy": 90.0,
                    "regime_green": True,
                    "rsi_2": 20.0,
                },
                {
                    "ticker": "AAA",
                    "date": pd.Timestamp("2026-01-04"),
                    "open": 101.0,
                    "high": 102.5,
                    "low": 101.0,
                    "close": 102.0,
                    "rsi_14": 30.0,
                    "vol_alpha": 2.0,
                    "sma_200_dist": 0.20,
                    "roc_63": 0.20,
                    "relative_strength_index_vs_spy": 90.0,
                    "regime_green": True,
                    "rsi_2": 95.0,
                },
            ]
        )
        indicators = {
            "rsi_14_max": 35.0,
            "vol_alpha_min": 1.4,
            "sma_200_dist_min": 0.10,
            "roc_63_min": 0.10,
            "relative_strength_index_vs_spy_min": 80.0,
            "signal_score_min": 30.0,
        }
        exit_rules = {
            "trailing_stop_pct": 0.08,
            "profit_target_pct": 0.10,
            "time_limit_days": 20,
        }

        gross_metrics = service._run_backtest(
            validation_frame,
            indicators,
            exit_rules,
            backtest_costs=service._load_backtest_costs({}),
        )
        net_metrics = service._run_backtest(
            validation_frame,
            indicators,
            exit_rules,
            backtest_costs=service._load_backtest_costs(
                {"backtest_costs": {"slippage_bps_per_side": 5, "commission_bps_per_side": 0}}
            ),
        )

        self.assertEqual(gross_metrics["trade_count"], 1)
        self.assertLess(net_metrics["expectancy"], gross_metrics["expectancy"])
        self.assertAlmostEqual(gross_metrics["expectancy"] - net_metrics["expectancy"], 0.0010, places=6)
