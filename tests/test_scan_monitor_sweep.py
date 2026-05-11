from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import unittest
from unittest.mock import patch

import pandas as pd

from src.scan.backfill_service import ScanBackfillService
from src.scan.service import LearnedRankerStatus, ScanPolicy, ScanService
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
    def test_scan_policy_supports_slot_specific_learned_ranker_weights(self) -> None:
        policy = ScanPolicy.from_config(
            {
                "scan_policy": {
                    "learned_ranker": {
                        "weight": 1.0,
                        "min_train_rows": 40,
                        "min_train_dates": 8,
                        "slot_weights": {
                            "energy": 2.0,
                            "materials": 0.25,
                        },
                    }
                }
            }
        )

        class FakeDB:
            pass

        service = ScanService(FakeDB(), email_sender=lambda subject, html_body, settings: None)
        self.assertEqual(service._learned_ranker_weight_for_slot("energy", policy), 2.0)
        self.assertEqual(service._learned_ranker_weight_for_slot("materials", policy), 0.25)
        self.assertEqual(service._learned_ranker_weight_for_slot("industrials", policy), 1.0)

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
                    "signal_score": 32.0,
                    "roc_63": 0.10,
                    "relative_strength_index_vs_spy": 85.0,
                    "vol_alpha": 1.2,
                    "sma_200_dist": 0.10,
                    "sma_50_dist": 0.05,
                    "rsi_14": 48.0,
                    "sector_pct_above_50": 0.8,
                    "sector_pct_above_200": 0.8,
                    "sector_median_roc_63": 0.08,
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
             patch("src.scan.service.load_feature_config", return_value={"scan_policy": {"max_candidates_total": 6, "max_candidates_per_slot": 3, "max_candidates_per_sector": 2, "pre_cap_candidates_per_slot": 5, "min_opportunity_score": 0.0}}), \
             patch("src.scan.service.load_active_strategies", return_value={"default": strategy}):
            report = service.run()

        self.assertTrue(report.emailed)
        self.assertEqual(report.candidate_count, 1)
        self.assertEqual(len(email_calls), 1)
        self.assertIn("Slot: default", email_calls[0].html_body)
        self.assertIn("<td>250</td>", email_calls[0].html_body)
        self.assertIn("chart", email_calls[0].html_body)
        self.assertIn("<td>76.00</td>", email_calls[0].html_body)
        self.assertIn("<td>89.60</td>", email_calls[0].html_body)
        self.assertIn("Opportunity", email_calls[0].html_body)

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
             patch("src.scan.service.load_feature_config", return_value={"scan_policy": {"max_candidates_total": 3, "max_candidates_per_slot": 2, "max_candidates_per_sector": 2, "pre_cap_candidates_per_slot": 5, "min_opportunity_score": 0.0}}), \
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
        self.assertIn("Why", html)
        self.assertIn("Chart", html)
        self.assertIn("Rank Context", html)

    def test_scan_skips_email_when_candidates_fail_opportunity_threshold_and_persists_snapshots(self) -> None:
        class FakeDB:
            def __init__(self):
                self.persisted = []
            def initialize(self): return None
            def list_universe_rows(self, active_only=True):
                return [{"ticker": "AAA", "sector": "Industrials", "md_volume_30d": 30_000_000}]
            def load_price_history(self, tickers): return pd.DataFrame()
            def list_open_trades(self): return []
            def replace_scan_candidates(self, *, scan_date, rows):
                self.persisted = list(rows)
                return len(self.persisted)

        service = ScanService(FakeDB(), email_sender=lambda subject, html_body, settings: None)
        snapshot = pd.DataFrame(
            [
                {
                    "ticker": "AAA",
                    "sector": "Industrials",
                    "regime_green": True,
                    "regime_etf": "SPY",
                    "adj_close": 80.0,
                    "atr_14": 4.0,
                    "md_volume_30d": 30_000_000,
                    "signal_score": 12.0,
                    "rsi_14": 78.0,
                    "vol_alpha": 0.9,
                    "sma_50_dist": 0.18,
                    "sma_200_dist": 0.01,
                    "roc_63": 0.01,
                    "relative_strength_index_vs_spy": 60.0,
                    "sector_pct_above_50": 0.2,
                    "sector_pct_above_200": 0.2,
                    "sector_median_roc_63": -0.02,
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
            slot="industrials",
            sector="Industrials",
        )

        with patch.object(service, "_download_recent_daily_history", return_value=pd.DataFrame()), \
             patch("src.scan.service.build_analysis_frame", return_value=(pd.DataFrame(), [])), \
             patch("src.scan.service.latest_snapshot", return_value=snapshot), \
             patch("src.scan.service.filter_signal_candidates", return_value=snapshot.copy()), \
             patch("src.scan.service.get_settings", return_value=settings), \
             patch("src.scan.service.load_feature_config", return_value={"scan_policy": {"min_opportunity_score": 0.95}}), \
             patch("src.scan.service.load_active_strategies", return_value={"industrials": strategy}):
            report = service.run()

        self.assertFalse(report.emailed)
        self.assertEqual(report.candidate_count, 0)
        self.assertEqual(len(service.db_manager.persisted), 1)
        self.assertFalse(service.db_manager.persisted[0]["selected"])

    def test_scan_penalizes_overlap_with_existing_positions(self) -> None:
        class FakeDB:
            def initialize(self): return None
            def list_universe_rows(self, active_only=True):
                return [
                    {"ticker": "AAA", "sector": "Industrials", "md_volume_30d": 40_000_000},
                    {"ticker": "BBB", "sector": "Industrials", "md_volume_30d": 35_000_000},
                ]
            def load_price_history(self, tickers): return pd.DataFrame()
            def list_open_trades(self):
                return [
                    {
                        "ticker": "AAA",
                        "entry_date": "2026-05-05",
                        "entry_price": 100.0,
                        "entry_atr": 2.0,
                        "strategy_id": 1,
                        "strategy_slot": "industrials",
                        "shares": 10,
                        "max_price_seen": 100.0,
                        "status": "open",
                    }
                ]

        email_calls: list[EmailCall] = []
        service = ScanService(FakeDB(), email_sender=lambda subject, html_body, settings: email_calls.append(EmailCall(subject, html_body)))
        snapshot = pd.DataFrame(
            [
                {"ticker": "AAA", "sector": "Industrials", "regime_green": True, "regime_etf": "SPY", "adj_close": 50.0, "atr_14": 2.0, "md_volume_30d": 40_000_000, "signal_score": 40.0, "roc_63": 0.12, "relative_strength_index_vs_spy": 85.0, "vol_alpha": 1.2, "sma_200_dist": 0.12, "sma_50_dist": 0.08, "rsi_14": 50.0, "sector_pct_above_50": 0.8, "sector_pct_above_200": 0.8, "sector_median_roc_63": 0.08},
                {"ticker": "BBB", "sector": "Industrials", "regime_green": True, "regime_etf": "SPY", "adj_close": 48.0, "atr_14": 2.0, "md_volume_30d": 35_000_000, "signal_score": 35.0, "roc_63": 0.11, "relative_strength_index_vs_spy": 84.0, "vol_alpha": 1.2, "sma_200_dist": 0.12, "sma_50_dist": 0.08, "rsi_14": 49.0, "sector_pct_above_50": 0.8, "sector_pct_above_200": 0.8, "sector_median_roc_63": 0.08},
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
            slot="industrials",
            sector="Industrials",
        )

        with patch.object(service, "_download_recent_daily_history", return_value=pd.DataFrame()), \
             patch("src.scan.service.build_analysis_frame", return_value=(pd.DataFrame(), [])), \
             patch("src.scan.service.latest_snapshot", return_value=snapshot), \
             patch("src.scan.service.filter_signal_candidates", side_effect=lambda frame, indicators: frame.copy()), \
             patch("src.scan.service.get_settings", return_value=settings), \
             patch("src.scan.service.load_feature_config", return_value={"scan_policy": {"max_candidates_total": 2, "min_opportunity_score": 0.0}}), \
             patch("src.scan.service.load_active_strategies", return_value={"industrials": strategy}):
            report = service.run()

        self.assertTrue(report.emailed)
        self.assertEqual(report.candidate_count, 1)
        html = email_calls[0].html_body
        self.assertIn("BBB", html)
        self.assertNotIn("AAA</td>", html)

    def test_scan_same_slot_overlap_remains_a_nudge_not_a_veto(self) -> None:
        class FakeDB:
            def initialize(self): return None
            def list_universe_rows(self, active_only=True):
                return [
                    {"ticker": "AAA", "sector": "Industrials", "md_volume_30d": 40_000_000},
                    {"ticker": "BBB", "sector": "Industrials", "md_volume_30d": 35_000_000},
                ]
            def load_price_history(self, tickers): return pd.DataFrame()
            def list_open_trades(self):
                return [
                    {
                        "ticker": "AAA",
                        "entry_date": "2026-05-05",
                        "entry_price": 100.0,
                        "entry_atr": 2.0,
                        "strategy_id": 1,
                        "strategy_slot": "industrials",
                        "shares": 10,
                        "max_price_seen": 100.0,
                        "status": "open",
                    }
                ]

        email_calls: list[EmailCall] = []
        service = ScanService(FakeDB(), email_sender=lambda subject, html_body, settings: email_calls.append(EmailCall(subject, html_body)))
        snapshot = pd.DataFrame(
            [
                {"ticker": "AAA", "sector": "Industrials", "regime_green": True, "regime_etf": "SPY", "adj_close": 50.0, "atr_14": 2.0, "md_volume_30d": 40_000_000, "signal_score": 40.0, "roc_63": 0.12, "relative_strength_index_vs_spy": 85.0, "vol_alpha": 1.2, "sma_200_dist": 0.12, "sma_50_dist": 0.08, "rsi_14": 50.0, "sector_pct_above_50": 0.8, "sector_pct_above_200": 0.8, "sector_median_roc_63": 0.08},
                {"ticker": "BBB", "sector": "Industrials", "regime_green": True, "regime_etf": "SPY", "adj_close": 48.0, "atr_14": 2.0, "md_volume_30d": 35_000_000, "signal_score": 35.0, "roc_63": 0.11, "relative_strength_index_vs_spy": 84.0, "vol_alpha": 1.2, "sma_200_dist": 0.12, "sma_50_dist": 0.08, "rsi_14": 49.0, "sector_pct_above_50": 0.8, "sector_pct_above_200": 0.8, "sector_median_roc_63": 0.08},
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
            slot="industrials",
            sector="Industrials",
        )

        with patch.object(service, "_download_recent_daily_history", return_value=pd.DataFrame()), \
             patch("src.scan.service.build_analysis_frame", return_value=(pd.DataFrame(), [])), \
             patch("src.scan.service.latest_snapshot", return_value=snapshot), \
             patch("src.scan.service.filter_signal_candidates", side_effect=lambda frame, indicators: frame.copy()), \
             patch("src.scan.service.get_settings", return_value=settings), \
             patch("src.scan.service.load_feature_config", return_value={"scan_policy": {"max_candidates_total": 2, "min_opportunity_score": 0.55}}), \
             patch("src.scan.service.load_active_strategies", return_value={"industrials": strategy}):
            report = service.run()

        self.assertTrue(report.emailed)
        self.assertEqual(report.candidate_count, 1)
        html = email_calls[0].html_body
        self.assertIn("BBB", html)
        self.assertIn("overlap -0.08", html)

    def test_scan_uses_selection_score_from_learned_ranker_under_caps(self) -> None:
        class FakeDB:
            def initialize(self): return None
            def list_universe_rows(self, active_only=True):
                return [
                    {"ticker": "AAA", "sector": "Energy", "md_volume_30d": 40_000_000},
                    {"ticker": "BBB", "sector": "Energy", "md_volume_30d": 38_000_000},
                ]
            def load_price_history(self, tickers): return pd.DataFrame()
            def list_open_trades(self): return []
            def load_scan_candidates(self):
                return pd.DataFrame(
                    [
                        {
                            "scan_date": "2026-05-01",
                            "ticker": "HIST1",
                            "strategy_slot": "energy",
                            "strategy_sector": "Energy",
                            "sector": "Energy",
                            "signal_score": 30.0,
                            "setup_quality_score": 0.7,
                            "expected_alpha_score": 0.7,
                            "breadth_score": 0.6,
                            "freshness_score": 0.7,
                            "overlap_penalty": 0.0,
                            "opportunity_score": 0.6,
                            "selected": 1,
                            "alpha_vs_sector_10d": 0.03,
                            "details_json": '{"already_owned": false, "feature_snapshot": {"atr_14": 2.0}}',
                        }
                    ]
                )

        email_calls: list[EmailCall] = []
        service = ScanService(FakeDB(), email_sender=lambda subject, html_body, settings: email_calls.append(EmailCall(subject, html_body)))
        snapshot = pd.DataFrame(
            [
                {"ticker": "AAA", "sector": "Energy", "regime_green": True, "regime_etf": "SPY", "adj_close": 50.0, "atr_14": 2.0, "md_volume_30d": 40_000_000, "signal_score": 37.0, "roc_63": 0.12, "relative_strength_index_vs_spy": 85.0, "vol_alpha": 1.2, "sma_200_dist": 0.12, "sma_50_dist": 0.08, "rsi_14": 50.0, "sector_pct_above_50": 0.8, "sector_pct_above_200": 0.8, "sector_median_roc_63": 0.08},
                {"ticker": "BBB", "sector": "Energy", "regime_green": True, "regime_etf": "SPY", "adj_close": 48.0, "atr_14": 2.0, "md_volume_30d": 38_000_000, "signal_score": 36.0, "roc_63": 0.11, "relative_strength_index_vs_spy": 84.0, "vol_alpha": 1.2, "sma_200_dist": 0.12, "sma_50_dist": 0.08, "rsi_14": 49.0, "sector_pct_above_50": 0.8, "sector_pct_above_200": 0.8, "sector_median_roc_63": 0.08},
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
            slot="energy",
            sector="Energy",
        )

        def fake_apply_learned_ranker(candidates, *, scan_policy, historical_scan_candidates):
            adjusted = candidates.copy()
            adjusted["ranker_score"] = [0.01, 0.06]
            adjusted["selection_score"] = [0.69, 0.74]
            adjusted["ranker_enabled"] = True
            adjusted["ranker_top_positive_reasons"] = [("signal_score (+0.0100)",), ("signal_score (+0.0200)",)]
            adjusted["ranker_top_negative_reasons"] = [tuple(), tuple()]
            return adjusted, LearnedRankerStatus(enabled=True, train_rows=100, train_dates=20, reason=None)

        with patch.object(service, "_download_recent_daily_history", return_value=pd.DataFrame()), \
             patch("src.scan.service.build_analysis_frame", return_value=(pd.DataFrame(), [])), \
             patch("src.scan.service.latest_snapshot", return_value=snapshot), \
             patch("src.scan.service.filter_signal_candidates", side_effect=lambda frame, indicators: frame.copy()), \
             patch("src.scan.service.get_settings", return_value=settings), \
             patch("src.scan.service.load_feature_config", return_value={"scan_policy": {"max_candidates_total": 1, "max_candidates_per_slot": 1, "max_candidates_per_sector": 1, "min_opportunity_score": 0.0}}), \
             patch("src.scan.service.load_active_strategies", return_value={"energy": strategy}), \
             patch.object(service, "_apply_learned_ranker", side_effect=fake_apply_learned_ranker):
            report = service.run()

        self.assertTrue(report.emailed)
        self.assertEqual(report.candidate_count, 1)
        html = email_calls[0].html_body
        self.assertIn("BBB", html)
        self.assertNotIn("AAA</td>", html)
        self.assertIn("learned 0.060", html)

    def test_scan_enables_learned_ranker_for_live_candidates_without_scan_date(self) -> None:
        class FakeDB:
            def initialize(self): return None
            def list_universe_rows(self, active_only=True):
                return [
                    {"ticker": "AAA", "sector": "Energy", "md_volume_30d": 40_000_000},
                    {"ticker": "BBB", "sector": "Energy", "md_volume_30d": 38_000_000},
                ]
            def load_price_history(self, tickers): return pd.DataFrame()
            def list_open_trades(self): return []
            def load_scan_candidates(self):
                rows = []
                history_dates = pd.bdate_range("2026-04-01", periods=10)
                for date_index, scan_day in enumerate(history_dates):
                    for candidate_index in range(5):
                        expected_alpha = 0.20 + (candidate_index * 0.15)
                        rows.append(
                            {
                                "scan_date": scan_day.strftime("%Y-%m-%d"),
                                "ticker": f"H{date_index}{candidate_index}",
                                "strategy_slot": "energy",
                                "strategy_sector": "Energy",
                                "sector": "Energy",
                                "md_volume_30d": 30_000_000 + candidate_index,
                                "signal_score": 25.0 + candidate_index,
                                "setup_quality_score": 0.60 + (candidate_index * 0.02),
                                "expected_alpha_score": expected_alpha,
                                "breadth_score": 0.55,
                                "freshness_score": 0.70,
                                "overlap_penalty": 0.0,
                                "opportunity_score": 0.55 + (candidate_index * 0.05),
                                "selected": 1 if candidate_index < 2 else 0,
                                "alpha_vs_sector_10d": (expected_alpha * 0.04) + (candidate_index * 0.002),
                                "details_json": '{"already_owned": false, "feature_snapshot": {"atr_14": 2.0}}',
                            }
                        )
                return pd.DataFrame(rows)

        email_calls: list[EmailCall] = []
        service = ScanService(FakeDB(), email_sender=lambda subject, html_body, settings: email_calls.append(EmailCall(subject, html_body)))
        snapshot = pd.DataFrame(
            [
                {"ticker": "AAA", "sector": "Energy", "regime_green": True, "regime_etf": "SPY", "adj_close": 50.0, "atr_14": 2.0, "md_volume_30d": 40_000_000, "signal_score": 31.0, "roc_63": 0.12, "relative_strength_index_vs_spy": 85.0, "vol_alpha": 1.2, "sma_200_dist": 0.12, "sma_50_dist": 0.08, "rsi_14": 50.0, "sector_pct_above_50": 0.8, "sector_pct_above_200": 0.8, "sector_median_roc_63": 0.08},
                {"ticker": "BBB", "sector": "Energy", "regime_green": True, "regime_etf": "SPY", "adj_close": 48.0, "atr_14": 2.0, "md_volume_30d": 38_000_000, "signal_score": 38.0, "roc_63": 0.11, "relative_strength_index_vs_spy": 84.0, "vol_alpha": 1.2, "sma_200_dist": 0.12, "sma_50_dist": 0.08, "rsi_14": 49.0, "sector_pct_above_50": 0.8, "sector_pct_above_200": 0.8, "sector_median_roc_63": 0.08},
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
            slot="energy",
            sector="Energy",
        )

        with patch.object(service, "_download_recent_daily_history", return_value=pd.DataFrame()), \
             patch("src.scan.service.build_analysis_frame", return_value=(pd.DataFrame(), [])), \
             patch("src.scan.service.latest_snapshot", return_value=snapshot), \
             patch("src.scan.service.filter_signal_candidates", side_effect=lambda frame, indicators: frame.copy()), \
             patch("src.scan.service.get_settings", return_value=settings), \
             patch("src.scan.service.load_feature_config", return_value={"scan_policy": {"max_candidates_total": 1, "max_candidates_per_slot": 1, "max_candidates_per_sector": 1, "min_opportunity_score": 0.0}}), \
             patch("src.scan.service.load_active_strategies", return_value={"energy": strategy}):
            report = service.run()

        self.assertTrue(report.emailed)
        self.assertTrue(report.learned_ranker_enabled)
        self.assertGreater(report.learned_ranker_train_rows, 0)
        self.assertGreater(report.learned_ranker_train_dates, 0)
        self.assertIsNone(report.learned_ranker_reason)
        html = email_calls[0].html_body
        self.assertIn("learned", html)

    def test_scan_backfill_replays_each_historical_date_without_using_future_rows(self) -> None:
        class FakeDB:
            def __init__(self):
                self.persisted_by_date = {}
            def initialize(self): return None
            def list_universe_rows(self, active_only=True):
                return [{"ticker": "AAA", "sector": "Industrials", "md_volume_30d": 30_000_000}]
            def load_price_history(self, tickers):
                return pd.DataFrame(
                    [
                        {"ticker": "AAA", "date": pd.Timestamp("2026-05-01"), "open": 10.0, "high": 11.0, "low": 9.0, "close": 10.0, "volume": 1_000_000, "adj_close": 10.0},
                        {"ticker": "AAA", "date": pd.Timestamp("2026-05-02"), "open": 20.0, "high": 21.0, "low": 19.0, "close": 20.0, "volume": 1_000_000, "adj_close": 20.0},
                        {"ticker": "SPY", "date": pd.Timestamp("2026-05-01"), "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1_000_000, "adj_close": 100.0},
                        {"ticker": "SPY", "date": pd.Timestamp("2026-05-02"), "open": 101.0, "high": 102.0, "low": 100.0, "close": 101.0, "volume": 1_000_000, "adj_close": 101.0},
                    ]
                )
            def load_scan_candidates(self, scan_date=None):
                return pd.DataFrame()
            def replace_scan_candidates(self, *, scan_date, rows):
                self.persisted_by_date[scan_date] = list(rows)
                return len(rows)

        db = FakeDB()
        service = ScanBackfillService(db)
        analysis_frame = pd.DataFrame(
            [
                {
                    "ticker": "AAA",
                    "date": pd.Timestamp("2026-05-01"),
                    "sector": "Industrials",
                    "regime_green": True,
                    "regime_etf": "SPY",
                    "adj_close": 10.0,
                    "atr_14": 1.0,
                    "md_volume_30d": 30_000_000,
                    "signal_score": 30.0,
                    "roc_63": 0.10,
                    "relative_strength_index_vs_spy": 80.0,
                    "vol_alpha": 1.1,
                    "sma_200_dist": 0.10,
                    "sma_50_dist": 0.05,
                    "rsi_14": 50.0,
                    "sector_pct_above_50": 0.8,
                    "sector_pct_above_200": 0.8,
                    "sector_median_roc_63": 0.08,
                },
                {
                    "ticker": "AAA",
                    "date": pd.Timestamp("2026-05-02"),
                    "sector": "Industrials",
                    "regime_green": True,
                    "regime_etf": "SPY",
                    "adj_close": 20.0,
                    "atr_14": 1.0,
                    "md_volume_30d": 30_000_000,
                    "signal_score": 35.0,
                    "roc_63": 0.12,
                    "relative_strength_index_vs_spy": 85.0,
                    "vol_alpha": 1.2,
                    "sma_200_dist": 0.12,
                    "sma_50_dist": 0.05,
                    "rsi_14": 48.0,
                    "sector_pct_above_50": 0.8,
                    "sector_pct_above_200": 0.8,
                    "sector_median_roc_63": 0.09,
                },
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
            slot="industrials",
            sector="Industrials",
        )

        with patch("src.scan.backfill_service.build_analysis_frame", return_value=(analysis_frame, [])), \
             patch("src.scan.backfill_service.filter_signal_candidates", side_effect=lambda frame, indicators: frame.copy()), \
             patch("src.scan.backfill_service.get_settings", return_value=settings), \
             patch("src.scan.backfill_service.load_feature_config", return_value={"scan_policy": {"max_candidates_total": 2, "min_opportunity_score": 0.0}}), \
             patch("src.scan.backfill_service.load_active_strategies", return_value={"industrials": strategy}):
            report = service.run(date_from="2026-05-01", date_to="2026-05-02")

        self.assertEqual(report.scan_dates_processed, 2)
        self.assertEqual(report.scan_dates_skipped, 0)
        self.assertIn("2026-05-01", db.persisted_by_date)
        self.assertIn("2026-05-02", db.persisted_by_date)
        self.assertEqual(db.persisted_by_date["2026-05-01"][0]["adj_close"], 10.0)
        self.assertEqual(db.persisted_by_date["2026-05-02"][0]["adj_close"], 20.0)


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

    def test_monitor_resolves_inactive_legacy_strategy_from_backtest_results(self) -> None:
        class FakeDB:
            def initialize(self): return None
            def list_open_trades(self):
                return [
                    {
                        "ticker": "RKLB",
                        "entry_date": "2026-05-05",
                        "entry_price": 100.0,
                        "shares": 10,
                        "max_price_seen": 100.0,
                        "status": "open",
                        "entry_atr": None,
                        "strategy_id": 172365,
                        "strategy_slot": "technology",
                    }
                ]
            def load_price_history(self, tickers):
                rows = []
                for ticker in ("RKLB", "SPY", "QQQ"):
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
                return {"rowid": 1, "entry_price": 100.0, "entry_atr": None, "shares": 10, "strategy_id": 172365, "strategy_slot": "technology"}
            def get_backtest_result_by_strategy_id(self, strategy_id):
                if strategy_id != 172365:
                    return None
                return {
                    "strategy_id": 172365,
                    "params_json": '{"sector":"Information Technology","indicators":{"rsi_14_max":35.0},"exit_rules":{"trailing_stop_pct":0.05,"profit_target_pct":0.12,"time_limit_days":20}}',
                }
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
        materials = ProductionStrategy(
            strategy_id=1,
            promoted_at="2026-05-05T17:00:00",
            indicators={"rsi_14_max": 35.0},
            exit_rules=ExitRules(0.05, 0.12, 20),
            slot="materials",
            sector="Materials",
        )

        with patch.object(service, "_load_intraday_last_prices", return_value={"RKLB": 100.0, "SPY": 105.0, "QQQ": 110.0}), \
             patch.object(service, "_download_recent_daily_history", return_value=pd.DataFrame()), \
             patch("src.monitor.service.get_settings", return_value=settings), \
             patch("src.monitor.service.load_active_strategies", return_value={"materials": materials}), \
             patch("src.monitor.service.latest_rsi_2_with_intraday", return_value=20.0):
            report = service.run()

        self.assertFalse(report.emailed)
        self.assertEqual(report.watchlist_size, 1)

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
             patch.object(service, "_download_recent_daily_history", return_value=pd.DataFrame()), \
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
             patch.object(service, "_download_recent_daily_history", return_value=pd.DataFrame()), \
             patch("src.monitor.service.get_settings", return_value=settings), \
             patch("src.monitor.service.load_active_strategies", return_value={"default": strategy}), \
             patch("src.monitor.service.build_analysis_frame", return_value=(analysis_frame, [])), \
             patch("src.monitor.service.latest_rsi_2_with_intraday", return_value=20.0):
            report = service.run()

        self.assertTrue(report.emailed)
        self.assertEqual(report.triggered_count, 1)
        self.assertIn("trailing stop", email_calls[0].html_body)
        self.assertIn("<td>sell</td>", email_calls[0].html_body)

    def test_monitor_uses_watch_breakout_for_breakout_only_rows(self) -> None:
        class FakeDB:
            def initialize(self): return None
            def list_open_trades(self):
                return [
                    {
                        "ticker": "AAA",
                        "entry_date": "2026-05-05",
                        "entry_price": 100.0,
                        "entry_atr": None,
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
                return {"rowid": 1, "entry_price": 100.0, "shares": 10}
            def update_trade_max_price(self, trade_rowid, max_price_seen): return None
            def close_trade(self, trade_rowid, exit_date, exit_price): return None
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
            [{"ticker": "SPY", "date": pd.Timestamp("2026-05-05"), "spy_sma_200": 100.0, "qqq_sma_200": None}]
        )

        with patch.object(service, "_load_intraday_last_prices", return_value={"AAA": 101.0, "SPY": 105.0, "QQQ": 110.0}), \
             patch.object(service, "_download_recent_daily_history", return_value=pd.DataFrame()), \
             patch("src.monitor.service.get_settings", return_value=settings), \
             patch("src.monitor.service.load_active_strategies", return_value={"default": strategy}), \
             patch("src.monitor.service.build_analysis_frame", return_value=(analysis_frame, [])), \
             patch("src.monitor.service.latest_rsi_2_with_intraday", return_value=20.0):
            report = service.run()

        self.assertTrue(report.emailed)
        self.assertEqual(report.triggered_count, 1)
        self.assertIn("breakout alert", email_calls[0].html_body)
        self.assertIn("<td>watch breakout</td>", email_calls[0].html_body)

    def test_monitor_uses_pre_earnings_exit_flag(self) -> None:
        class FakeDB:
            def initialize(self): return None
            def list_open_trades(self):
                return [
                    {
                        "ticker": "AAA",
                        "entry_date": "2026-05-05",
                        "entry_price": 100.0,
                        "entry_atr": None,
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
                return {"rowid": 1, "entry_price": 100.0, "shares": 10}
            def update_trade_max_price(self, trade_rowid, max_price_seen): return None
            def close_trade(self, trade_rowid, exit_date, exit_price): return None
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
            exit_rules=ExitRules(0.05, 0.12, 20, exit_before_earnings_days=2),
        )
        analysis_frame = pd.DataFrame(
            [
                {"ticker": "SPY", "date": pd.Timestamp("2026-05-05"), "spy_sma_200": 100.0, "qqq_sma_200": None},
                {"ticker": "AAA", "date": pd.Timestamp("2026-05-05"), "days_to_next_earnings": 1.0},
            ]
        )

        with patch.object(service, "_load_intraday_last_prices", return_value={"AAA": 101.0, "SPY": 105.0, "QQQ": 110.0}), \
             patch.object(service, "_download_recent_daily_history", return_value=pd.DataFrame()), \
             patch("src.monitor.service.get_settings", return_value=settings), \
             patch("src.monitor.service.load_active_strategies", return_value={"default": strategy}), \
             patch("src.monitor.service.build_analysis_frame", return_value=(analysis_frame, [])), \
             patch("src.monitor.service.latest_rsi_2_with_intraday", return_value=20.0):
            report = service.run()

        self.assertTrue(report.emailed)
        self.assertEqual(report.triggered_count, 1)
        self.assertIn("pre earnings exit", email_calls[0].html_body)
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
             patch.object(service, "_download_recent_daily_history", return_value=pd.DataFrame()), \
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

    def test_resolve_low_drawdown_technology_sweep_mode(self) -> None:
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
        config = {
            "sweep_grid": {
                "rsi_14_max": {"min": 25, "max": 60, "step": 5},
                "vol_alpha_min": {"min": 0.8, "max": 1.8, "step": 0.2},
            },
            "sweep_modes": {
                "low_drawdown_technology": {
                    "sector_whitelist": ["Information Technology"],
                    "grid_overrides": {
                        "rsi_14_max": {"min": 30, "max": 45, "step": 5},
                        "vol_alpha_min": {"min": 0.8, "max": 1.2, "step": 0.2},
                    },
                }
            },
        }

        resolved, sectors = service._resolve_sweep_mode(config, "low_drawdown_technology")
        grid = service._build_parameter_grid(resolved)

        self.assertEqual(sectors, ["Information Technology"])
        self.assertEqual(len(grid), 12)
        self.assertEqual(grid[0]["indicators"]["rsi_14_max"], 30.0)

    def test_resolve_promotable_live_technology_sweep_mode(self) -> None:
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
        config = {
            "sweep_grid": {
                "rsi_14_max": {"min": 25, "max": 60, "step": 5},
                "vol_alpha_min": {"min": 0.8, "max": 1.8, "step": 0.2},
                "sma_200_dist_min": {"min": 0.0, "max": 0.2, "step": 0.02},
                "roc_63_min": {"min": 0.1, "max": 0.15, "step": 0.05},
                "relative_strength_index_vs_spy_min": {"min": 80, "max": 90, "step": 5},
                "signal_score_min": {"min": 30, "max": 34, "step": 2},
                "trailing_stop_atr_mult": {"min": 2.0, "max": 2.5, "step": 0.5},
                "profit_target_atr_mult": {"min": 2.5, "max": 3.0, "step": 0.5},
            },
            "sweep_modes": {
                "promotable_live_technology": {
                    "sector_whitelist": ["Information Technology"],
                    "grid_overrides": {
                        "rsi_14_max": {"min": 40, "max": 45, "step": 5},
                        "vol_alpha_min": {"min": 0.8, "max": 1.0, "step": 0.2},
                        "sma_200_dist_min": {"min": 0.18, "max": 0.20, "step": 0.02},
                        "relative_strength_index_vs_spy_min": {"min": 85, "max": 90, "step": 5},
                        "signal_score_min": {"min": 32, "max": 34, "step": 2},
                        "trailing_stop_atr_mult": {"min": 2.5, "max": 2.5, "step": 0.5},
                        "profit_target_atr_mult": {"min": 3.0, "max": 3.0, "step": 0.5},
                    },
                }
            },
        }

        resolved, sectors = service._resolve_sweep_mode(config, "promotable_live_technology")
        grid = service._build_parameter_grid(resolved)

        self.assertEqual(sectors, ["Information Technology"])
        self.assertEqual(len(grid), 64)
        self.assertEqual(grid[0]["indicators"]["rsi_14_max"], 40.0)
        self.assertEqual(grid[0]["exit_rules"]["trailing_stop_atr_mult"], 2.5)
        self.assertEqual(grid[0]["exit_rules"]["profit_target_atr_mult"], 3.0)

    def test_resolve_energy_earnings_refined_mode(self) -> None:
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
        config = {
            "sweep_grid": {
                "rsi_14_max": {"min": 25, "max": 60, "step": 5},
            },
            "sweep_modes": {
                "high_performance_energy_earnings_refined": {
                    "sector_whitelist": ["Energy"],
                    "grid_overrides": {
                        "rsi_14_max": {"min": 55, "max": 60, "step": 5},
                        "days_to_next_earnings_min": {"min": 0, "max": 6, "step": 2},
                        "days_since_last_earnings_min": {"min": 0, "max": 4, "step": 2},
                        "signal_score_min": {"min": 30, "max": 32, "step": 2},
                        "exit_before_earnings_days": {"min": 0, "max": 2, "step": 1},
                    },
                }
            },
        }

        resolved, sectors = service._resolve_sweep_mode(config, "high_performance_energy_earnings_refined")
        grid = service._build_parameter_grid(resolved)

        self.assertEqual(sectors, ["Energy"])
        self.assertEqual(len(grid), 144)
        self.assertTrue(any("days_to_next_earnings_min" in row["indicators"] for row in grid))
        self.assertTrue(any("days_since_last_earnings_min" in row["indicators"] for row in grid))
        self.assertIsNone(grid[0]["exit_rules"]["exit_before_earnings_days"])

    def test_resolve_production_sleeves_gap_refined_mode(self) -> None:
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
        config = {
            "sweep_grid": {
                "rsi_14_max": {"min": 25, "max": 60, "step": 5},
            },
            "sweep_modes": {
                "production_sleeves_gap_refined": {
                    "sector_whitelist": ["Energy", "Materials", "Industrials"],
                    "grid_overrides": {
                        "rsi_14_max": {"min": 55, "max": 60, "step": 5},
                        "avg_abs_gap_pct_20_max": {"min": 0.02, "max": 0.04, "step": 0.01},
                        "max_gap_down_pct_60_max": {"min": 0.04, "max": 0.08, "step": 0.02},
                        "signal_score_min": {"min": 30, "max": 32, "step": 2},
                    },
                }
            },
        }

        resolved, sectors = service._resolve_sweep_mode(config, "production_sleeves_gap_refined")
        grid = service._build_parameter_grid(resolved)

        self.assertEqual(sectors, ["Energy", "Materials", "Industrials"])
        self.assertEqual(len(grid), 36)
        self.assertIn("avg_abs_gap_pct_20_max", grid[0]["indicators"])
        self.assertIn("max_gap_down_pct_60_max", grid[0]["indicators"])

    def test_resolve_promotable_live_technology_v2_sweep_mode(self) -> None:
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
        config = {
            "sweep_grid": {
                "rsi_14_max": {"min": 25, "max": 60, "step": 5},
                "vol_alpha_min": {"min": 0.8, "max": 1.8, "step": 0.2},
                "sma_200_dist_min": {"min": 0.0, "max": 0.2, "step": 0.02},
                "roc_63_min": {"min": 0.1, "max": 0.15, "step": 0.05},
                "relative_strength_index_vs_spy_min": {"min": 80, "max": 90, "step": 5},
                "signal_score_min": {"min": 30, "max": 34, "step": 2},
                "trailing_stop_atr_mult": {"min": 2.0, "max": 2.5, "step": 0.5},
                "profit_target_atr_mult": {"min": 2.5, "max": 3.0, "step": 0.5},
            },
            "sweep_modes": {
                "promotable_live_technology_v2": {
                    "sector_whitelist": ["Information Technology"],
                    "grid_overrides": {
                        "rsi_14_max": {"min": 40, "max": 40, "step": 5},
                        "vol_alpha_min": {"min": 0.8, "max": 0.8, "step": 0.2},
                        "sma_200_dist_min": {"min": 0.18, "max": 0.20, "step": 0.02},
                        "roc_63_min": {"min": 0.10, "max": 0.10, "step": 0.05},
                        "relative_strength_index_vs_spy_min": {"min": 85, "max": 85, "step": 5},
                        "signal_score_min": {"min": 30, "max": 32, "step": 2},
                        "trailing_stop_atr_mult": {"min": 2.5, "max": 2.5, "step": 0.5},
                        "profit_target_atr_mult": {"min": 3.0, "max": 3.0, "step": 0.5},
                    },
                }
            },
        }

        resolved, sectors = service._resolve_sweep_mode(config, "promotable_live_technology_v2")
        grid = service._build_parameter_grid(resolved)

        self.assertEqual(sectors, ["Information Technology"])
        self.assertEqual(len(grid), 4)
        self.assertEqual(grid[0]["indicators"]["rsi_14_max"], 40.0)
        self.assertEqual(grid[0]["indicators"]["vol_alpha_min"], 0.8)
        self.assertEqual(grid[0]["exit_rules"]["trailing_stop_atr_mult"], 2.5)
        self.assertEqual(grid[0]["exit_rules"]["profit_target_atr_mult"], 3.0)

    def test_resolve_promotable_live_technology_v3_sweep_mode(self) -> None:
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
        config = {
            "sweep_grid": {
                "rsi_14_max": {"min": 25, "max": 60, "step": 5},
                "vol_alpha_min": {"min": 0.8, "max": 1.8, "step": 0.2},
                "sma_200_dist_min": {"min": 0.0, "max": 0.2, "step": 0.02},
                "roc_63_min": {"min": 0.1, "max": 0.15, "step": 0.05},
                "relative_strength_index_vs_spy_min": {"min": 80, "max": 90, "step": 5},
                "signal_score_min": {"min": 30, "max": 34, "step": 2},
                "trailing_stop_atr_mult": {"min": 2.0, "max": 2.5, "step": 0.5},
                "profit_target_atr_mult": {"min": 2.5, "max": 3.0, "step": 0.5},
            },
            "sweep_modes": {
                "promotable_live_technology_v3": {
                    "sector_whitelist": ["Information Technology"],
                    "grid_overrides": {
                        "rsi_14_max": {"min": 40, "max": 40, "step": 5},
                        "vol_alpha_min": {"min": 0.8, "max": 0.8, "step": 0.2},
                        "sma_200_dist_min": {"min": 0.18, "max": 0.20, "step": 0.02},
                        "roc_63_min": {"min": 0.10, "max": 0.10, "step": 0.05},
                        "relative_strength_index_vs_spy_min": {"min": 85, "max": 85, "step": 5},
                        "signal_score_min": {"min": 31, "max": 31, "step": 2},
                        "trailing_stop_atr_mult": {"min": 2.5, "max": 2.5, "step": 0.5},
                        "profit_target_atr_mult": {"min": 3.0, "max": 3.0, "step": 0.5},
                    },
                }
            },
        }

        resolved, sectors = service._resolve_sweep_mode(config, "promotable_live_technology_v3")
        grid = service._build_parameter_grid(resolved)

        self.assertEqual(sectors, ["Information Technology"])
        self.assertEqual(len(grid), 2)
        self.assertEqual(grid[0]["indicators"]["signal_score_min"], 31.0)
        self.assertEqual(grid[0]["exit_rules"]["trailing_stop_atr_mult"], 2.5)
        self.assertEqual(grid[0]["exit_rules"]["profit_target_atr_mult"], 3.0)

    def test_resolve_promotable_live_technology_v4_sweep_mode(self) -> None:
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
        config = {
            "sweep_grid": {
                "rsi_14_max": {"min": 25, "max": 60, "step": 5},
                "vol_alpha_min": {"min": 0.8, "max": 1.8, "step": 0.2},
                "sma_200_dist_min": {"min": 0.0, "max": 0.2, "step": 0.02},
                "roc_63_min": {"min": 0.1, "max": 0.15, "step": 0.05},
                "relative_strength_index_vs_spy_min": {"min": 80, "max": 90, "step": 5},
                "signal_score_min": {"min": 30, "max": 34, "step": 2},
                "trailing_stop_atr_mult": {"min": 2.0, "max": 2.5, "step": 0.5},
                "profit_target_atr_mult": {"min": 2.5, "max": 3.0, "step": 0.5},
            },
            "sweep_modes": {
                "promotable_live_technology_v4": {
                    "sector_whitelist": ["Information Technology"],
                    "grid_overrides": {
                        "rsi_14_max": {"min": 40, "max": 40, "step": 5},
                        "vol_alpha_min": {"min": 0.8, "max": 0.8, "step": 0.2},
                        "sma_200_dist_min": {"min": 0.18, "max": 0.20, "step": 0.02},
                        "roc_63_min": {"min": 0.10, "max": 0.10, "step": 0.05},
                        "relative_strength_index_vs_spy_min": {"min": 85, "max": 90, "step": 5},
                        "signal_score_min": {"min": 30, "max": 30, "step": 2},
                        "trailing_stop_atr_mult": {"min": 2.5, "max": 2.5, "step": 0.5},
                        "profit_target_atr_mult": {"min": 2.5, "max": 3.0, "step": 0.5},
                    },
                }
            },
        }

        resolved, sectors = service._resolve_sweep_mode(config, "promotable_live_technology_v4")
        grid = service._build_parameter_grid(resolved)

        self.assertEqual(sectors, ["Information Technology"])
        self.assertEqual(len(grid), 8)
        self.assertEqual(grid[0]["indicators"]["signal_score_min"], 30.0)
        self.assertIn(grid[0]["indicators"]["relative_strength_index_vs_spy_min"], {85.0, 90.0})
        self.assertIn(grid[0]["exit_rules"]["profit_target_atr_mult"], {2.5, 3.0})

    def test_resolve_promotable_live_technology_v5_sweep_mode(self) -> None:
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
        config = {
            "sweep_grid": {
                "rsi_14_max": {"min": 25, "max": 60, "step": 5},
                "vol_alpha_min": {"min": 0.8, "max": 1.8, "step": 0.2},
                "sma_200_dist_min": {"min": 0.0, "max": 0.2, "step": 0.02},
                "roc_63_min": {"min": 0.1, "max": 0.15, "step": 0.05},
                "relative_strength_index_vs_spy_min": {"min": 80, "max": 90, "step": 5},
                "signal_score_min": {"min": 30, "max": 34, "step": 2},
                "trailing_stop_atr_mult": {"min": 2.0, "max": 2.5, "step": 0.25},
                "profit_target_atr_mult": {"min": 2.5, "max": 3.0, "step": 0.5},
            },
            "sweep_modes": {
                "promotable_live_technology_v5": {
                    "sector_whitelist": ["Information Technology"],
                    "grid_overrides": {
                        "rsi_14_max": {"min": 40, "max": 40, "step": 5},
                        "vol_alpha_min": {"min": 0.8, "max": 0.8, "step": 0.2},
                        "sma_200_dist_min": {"min": 0.18, "max": 0.20, "step": 0.02},
                        "roc_63_min": {"min": 0.10, "max": 0.10, "step": 0.05},
                        "relative_strength_index_vs_spy_min": {"min": 85, "max": 90, "step": 5},
                        "signal_score_min": {"min": 30, "max": 30, "step": 2},
                        "trailing_stop_atr_mult": {"min": 2.0, "max": 2.25, "step": 0.25},
                        "profit_target_atr_mult": {"min": 2.5, "max": 3.0, "step": 0.5},
                    },
                }
            },
        }

        resolved, sectors = service._resolve_sweep_mode(config, "promotable_live_technology_v5")
        grid = service._build_parameter_grid(resolved)

        self.assertEqual(sectors, ["Information Technology"])
        self.assertEqual(len(grid), 16)
        self.assertEqual(grid[0]["indicators"]["signal_score_min"], 30.0)
        self.assertIn(grid[0]["indicators"]["relative_strength_index_vs_spy_min"], {85.0, 90.0})
        self.assertIn(grid[0]["exit_rules"]["trailing_stop_atr_mult"], {2.0, 2.25})
        self.assertIn(grid[0]["exit_rules"]["profit_target_atr_mult"], {2.5, 3.0})

    def test_resolve_high_performance_energy_sweep_mode(self) -> None:
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
        config = {
            "sweep_grid": {
                "rsi_14_max": {"min": 25, "max": 60, "step": 5},
                "vol_alpha_min": {"min": 0.8, "max": 1.8, "step": 0.2},
                "sma_200_dist_min": {"min": 0.0, "max": 0.2, "step": 0.02},
                "roc_63_min": {"min": 0.1, "max": 0.15, "step": 0.05},
                "relative_strength_index_vs_spy_min": {"min": 80, "max": 90, "step": 5},
                "signal_score_min": {"min": 30, "max": 34, "step": 2},
                "trailing_stop_atr_mult": {"min": 2.0, "max": 2.5, "step": 0.25},
                "profit_target_atr_mult": {"min": 2.5, "max": 3.0, "step": 0.5},
            },
            "sweep_modes": {
                "high_performance_energy": {
                    "sector_whitelist": ["Energy"],
                    "grid_overrides": {
                        "rsi_14_max": {"min": 40, "max": 45, "step": 5},
                        "vol_alpha_min": {"min": 0.8, "max": 1.0, "step": 0.2},
                        "sma_200_dist_min": {"min": 0.10, "max": 0.14, "step": 0.04},
                        "roc_63_min": {"min": 0.05, "max": 0.10, "step": 0.05},
                        "relative_strength_index_vs_spy_min": {"min": 75, "max": 80, "step": 5},
                        "oil_corr_60_min": {"min": 0.10, "max": 0.40, "step": 0.30},
                        "signal_score_min": {"min": 30, "max": 32, "step": 2},
                        "trailing_stop_atr_mult": {"min": 2.5, "max": 3.0, "step": 0.5},
                        "profit_target_atr_mult": {"min": 3.0, "max": 3.5, "step": 0.5},
                    },
                }
            },
        }

        resolved, sectors = service._resolve_sweep_mode(config, "high_performance_energy")
        grid = service._build_parameter_grid(resolved)

        self.assertEqual(sectors, ["Energy"])
        self.assertEqual(len(grid), 512)
        self.assertIn("oil_corr_60_min", grid[0]["indicators"])
        self.assertIn(grid[0]["indicators"]["signal_score_min"], {30.0, 32.0})

    def test_resolve_high_performance_energy_refined_sweep_mode(self) -> None:
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
        config = {
            "sweep_grid": {
                "rsi_14_max": {"min": 25, "max": 60, "step": 5},
                "vol_alpha_min": {"min": 0.8, "max": 1.8, "step": 0.2},
                "sma_200_dist_min": {"min": 0.0, "max": 0.2, "step": 0.02},
                "roc_63_min": {"min": 0.1, "max": 0.15, "step": 0.05},
                "relative_strength_index_vs_spy_min": {"min": 80, "max": 90, "step": 5},
                "signal_score_min": {"min": 30, "max": 34, "step": 2},
                "trailing_stop_atr_mult": {"min": 2.0, "max": 2.5, "step": 0.25},
                "profit_target_atr_mult": {"min": 2.5, "max": 3.0, "step": 0.5},
            },
            "sweep_modes": {
                "high_performance_energy_refined": {
                    "sector_whitelist": ["Energy"],
                    "grid_overrides": {
                        "rsi_14_max": {"min": 55, "max": 60, "step": 5},
                        "vol_alpha_min": {"min": 1.0, "max": 1.2, "step": 0.2},
                        "sma_200_dist_min": {"min": 0.10, "max": 0.14, "step": 0.04},
                        "roc_63_min": {"min": 0.05, "max": 0.15, "step": 0.05},
                        "relative_strength_index_vs_spy_min": {"min": 75, "max": 80, "step": 5},
                        "oil_corr_60_min": {"min": 0.10, "max": 0.40, "step": 0.30},
                        "signal_score_min": {"min": 30, "max": 32, "step": 2},
                        "trailing_stop_atr_mult": {"min": 3.0, "max": 3.5, "step": 0.5},
                        "profit_target_atr_mult": {"min": 3.5, "max": 4.0, "step": 0.5},
                    },
                }
            },
        }

        resolved, sectors = service._resolve_sweep_mode(config, "high_performance_energy_refined")
        grid = service._build_parameter_grid(resolved)

        self.assertEqual(sectors, ["Energy"])
        self.assertEqual(len(grid), 768)
        self.assertIn("oil_corr_60_min", grid[0]["indicators"])
        self.assertIn(grid[0]["indicators"]["rsi_14_max"], {55.0, 60.0})
        self.assertIn(grid[0]["exit_rules"]["trailing_stop_atr_mult"], {3.0, 3.5})
        self.assertIn(grid[0]["exit_rules"]["profit_target_atr_mult"], {3.5, 4.0})

    def test_resolve_high_performance_energy_stability_refined_sweep_mode(self) -> None:
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
        config = {
            "sweep_grid": {
                "rsi_14_max": {"min": 25, "max": 60, "step": 5},
                "vol_alpha_min": {"min": 0.8, "max": 1.8, "step": 0.2},
                "sma_200_dist_min": {"min": 0.0, "max": 0.2, "step": 0.02},
                "roc_63_min": {"min": 0.1, "max": 0.15, "step": 0.05},
                "relative_strength_index_vs_spy_min": {"min": 80, "max": 90, "step": 5},
                "signal_score_min": {"min": 30, "max": 34, "step": 2},
                "trailing_stop_atr_mult": {"min": 2.0, "max": 2.5, "step": 0.25},
                "profit_target_atr_mult": {"min": 2.5, "max": 3.0, "step": 0.5},
            },
            "sweep_modes": {
                "high_performance_energy_stability_refined": {
                    "sector_whitelist": ["Energy"],
                    "grid_overrides": {
                        "rsi_14_max": {"min": 45, "max": 55, "step": 5},
                        "vol_alpha_min": {"min": 0.8, "max": 1.2, "step": 0.2},
                        "sma_200_dist_min": {"min": 0.14, "max": 0.14, "step": 0.04},
                        "roc_63_min": {"min": 0.05, "max": 0.10, "step": 0.05},
                        "relative_strength_index_vs_spy_min": {"min": 75, "max": 80, "step": 5},
                        "oil_corr_60_min": {"min": 0.30, "max": 0.50, "step": 0.10},
                        "signal_score_min": {"min": 32, "max": 34, "step": 2},
                        "trailing_stop_atr_mult": {"min": 3.0, "max": 3.5, "step": 0.5},
                        "profit_target_atr_mult": {"min": 3.5, "max": 4.5, "step": 0.5},
                    },
                }
            },
        }

        resolved, sectors = service._resolve_sweep_mode(config, "high_performance_energy_stability_refined")
        grid = service._build_parameter_grid(resolved)

        self.assertEqual(sectors, ["Energy"])
        self.assertEqual(len(grid), 1296)
        self.assertIn(grid[0]["indicators"]["rsi_14_max"], {45.0, 50.0, 55.0})
        self.assertIn(grid[0]["indicators"]["vol_alpha_min"], {0.8, 1.0, 1.2})
        self.assertEqual(grid[0]["indicators"]["sma_200_dist_min"], 0.14)
        self.assertIn(grid[0]["indicators"]["oil_corr_60_min"], {0.3, 0.4, 0.5})
        self.assertIn(grid[0]["indicators"]["signal_score_min"], {32.0, 34.0})
        self.assertIn(grid[0]["exit_rules"]["trailing_stop_atr_mult"], {3.0, 3.5})
        self.assertIn(grid[0]["exit_rules"]["profit_target_atr_mult"], {3.5, 4.0, 4.5})

    def test_resolve_high_performance_materials_refined_sweep_mode(self) -> None:
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
        config = {
            "sweep_grid": {
                "rsi_14_max": {"min": 25, "max": 60, "step": 5},
                "vol_alpha_min": {"min": 0.8, "max": 1.8, "step": 0.2},
                "sma_200_dist_min": {"min": 0.0, "max": 0.2, "step": 0.02},
                "roc_63_min": {"min": 0.1, "max": 0.15, "step": 0.05},
                "relative_strength_index_vs_spy_min": {"min": 80, "max": 90, "step": 5},
                "signal_score_min": {"min": 30, "max": 34, "step": 2},
                "trailing_stop_atr_mult": {"min": 2.0, "max": 2.5, "step": 0.25},
                "profit_target_atr_mult": {"min": 2.5, "max": 3.0, "step": 0.5},
            },
            "sweep_modes": {
                "high_performance_materials_refined": {
                    "sector_whitelist": ["Materials"],
                    "grid_overrides": {
                        "rsi_14_max": {"min": 55, "max": 60, "step": 5},
                        "vol_alpha_min": {"min": 1.0, "max": 1.2, "step": 0.2},
                        "sma_200_dist_min": {"min": 0.14, "max": 0.22, "step": 0.04},
                        "roc_63_min": {"min": 0.05, "max": 0.15, "step": 0.05},
                        "relative_strength_index_vs_spy_min": {"min": 75, "max": 80, "step": 5},
                        "signal_score_min": {"min": 30, "max": 32, "step": 2},
                        "trailing_stop_atr_mult": {"min": 2.5, "max": 2.5, "step": 0.5},
                        "profit_target_atr_mult": {"min": 4.0, "max": 4.5, "step": 0.5},
                    },
                }
            },
        }

        resolved, sectors = service._resolve_sweep_mode(config, "high_performance_materials_refined")
        grid = service._build_parameter_grid(resolved)

        self.assertEqual(sectors, ["Materials"])
        self.assertEqual(len(grid), 288)
        self.assertIn(grid[0]["indicators"]["rsi_14_max"], {55.0, 60.0})
        self.assertIn(grid[0]["indicators"]["sma_200_dist_min"], {0.14, 0.18, 0.22})
        self.assertEqual(grid[0]["exit_rules"]["trailing_stop_atr_mult"], 2.5)
        self.assertIn(grid[0]["exit_rules"]["profit_target_atr_mult"], {4.0, 4.5})

    def test_resolve_high_performance_industrials_refined_sweep_mode(self) -> None:
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
        config = {
            "sweep_grid": {
                "rsi_14_max": {"min": 25, "max": 60, "step": 5},
                "vol_alpha_min": {"min": 0.8, "max": 1.8, "step": 0.2},
                "sma_200_dist_min": {"min": 0.0, "max": 0.2, "step": 0.02},
                "roc_63_min": {"min": 0.1, "max": 0.15, "step": 0.05},
                "relative_strength_index_vs_spy_min": {"min": 80, "max": 90, "step": 5},
                "signal_score_min": {"min": 30, "max": 34, "step": 2},
                "trailing_stop_atr_mult": {"min": 2.0, "max": 2.5, "step": 0.25},
                "profit_target_atr_mult": {"min": 2.5, "max": 3.0, "step": 0.5},
            },
            "sweep_modes": {
                "high_performance_industrials_refined": {
                    "sector_whitelist": ["Industrials"],
                    "grid_overrides": {
                        "rsi_14_max": {"min": 50, "max": 60, "step": 5},
                        "vol_alpha_min": {"min": 0.8, "max": 1.2, "step": 0.2},
                        "sma_200_dist_min": {"min": 0.10, "max": 0.14, "step": 0.04},
                        "roc_63_min": {"min": 0.15, "max": 0.20, "step": 0.05},
                        "relative_strength_index_vs_spy_min": {"min": 85, "max": 90, "step": 5},
                        "signal_score_min": {"min": 32, "max": 34, "step": 2},
                        "trailing_stop_atr_mult": {"min": 3.0, "max": 3.5, "step": 0.5},
                        "profit_target_atr_mult": {"min": 4.0, "max": 4.5, "step": 0.5},
                    },
                }
            },
        }

        resolved, sectors = service._resolve_sweep_mode(config, "high_performance_industrials_refined")
        grid = service._build_parameter_grid(resolved)

        self.assertEqual(sectors, ["Industrials"])
        self.assertEqual(len(grid), 576)
        self.assertIn(grid[0]["indicators"]["rsi_14_max"], {50.0, 55.0, 60.0})
        self.assertIn(grid[0]["indicators"]["relative_strength_index_vs_spy_min"], {85.0, 90.0})
        self.assertIn(grid[0]["exit_rules"]["trailing_stop_atr_mult"], {3.0, 3.5})
        self.assertIn(grid[0]["exit_rules"]["profit_target_atr_mult"], {4.0, 4.5})

    def test_resolve_high_performance_real_economy_sweep_mode(self) -> None:
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
        config = {
            "sweep_grid": {
                "rsi_14_max": {"min": 25, "max": 60, "step": 5},
            },
            "sweep_modes": {
                "high_performance_real_economy": {
                    "sector_whitelist": ["Energy", "Materials", "Industrials", "Financials"],
                    "grid_overrides": {
                        "rsi_14_max": {"min": 40, "max": 45, "step": 5},
                        "signal_score_min": {"min": 28, "max": 30, "step": 2},
                    },
                }
            },
        }

        resolved, sectors = service._resolve_sweep_mode(config, "high_performance_real_economy")
        grid = service._build_parameter_grid(resolved)

        self.assertEqual(sectors, ["Energy", "Materials", "Industrials", "Financials"])
        self.assertEqual(len(grid), 4)
        self.assertIn(grid[0]["indicators"]["rsi_14_max"], {40.0, 45.0})
        self.assertIn(grid[0]["indicators"]["signal_score_min"], {28.0, 30.0})

    def test_resolve_breakout_v1_growth_leaders_sweep_mode(self) -> None:
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
        config = {
            "sweep_grid": {
                "trailing_stop_atr_mult": {"min": 2.0, "max": 2.5, "step": 0.5},
            },
            "sweep_modes": {
                "breakout_v1_growth_leaders": {
                    "sector_whitelist": ["Information Technology", "Industrials", "Financials"],
                    "replace_base_grid": True,
                    "grid_overrides": {
                        "sma_50_dist_min": {"min": 0.0, "max": 0.0, "step": 0.1},
                        "ma_alignment_50_200_min": {"min": 0.0, "max": 0.0, "step": 0.1},
                        "ma_slope_50_20_min": {"min": 0.0, "max": 0.0, "step": 0.1},
                        "ma_slope_200_20_min": {"min": 0.0, "max": 0.0, "step": 0.1},
                        "breakout_above_20d_high_min": {"min": 1.0, "max": 1.0, "step": 1.0},
                        "distance_above_20d_high_max": {"min": 0.01, "max": 0.02, "step": 0.01},
                        "relative_strength_index_vs_spy_min": {"min": 85, "max": 90, "step": 5},
                        "roc_63_min": {"min": 0.10, "max": 0.10, "step": 0.05},
                        "rsi_14_min": {"min": 50, "max": 50, "step": 5},
                        "sma_200_dist_max": {"min": 0.30, "max": 0.30, "step": 0.05},
                        "base_range_pct_20_max": {"min": 0.08, "max": 0.12, "step": 0.04},
                        "base_atr_contraction_20_max": {"min": 0.95, "max": 0.95, "step": 0.10},
                        "base_volume_dryup_ratio_20_max": {"min": 0.95, "max": 0.95, "step": 0.10},
                        "breakout_volume_ratio_50_min": {"min": 2.0, "max": 2.5, "step": 0.5},
                        "signal_score_min": {"min": 34, "max": 36, "step": 2},
                        "trailing_stop_atr_mult": {"min": 1.5, "max": 2.0, "step": 0.5},
                        "profit_target_atr_mult": {"min": 3.0, "max": 3.0, "step": 1.0},
                    },
                }
            },
        }

        resolved, sectors = service._resolve_sweep_mode(config, "breakout_v1_growth_leaders")
        grid = service._build_parameter_grid(resolved)

        self.assertEqual(sectors, ["Information Technology", "Industrials", "Financials"])
        self.assertEqual(len(grid), 64)
        self.assertIn("breakout_above_20d_high_min", grid[0]["indicators"])
        self.assertIn("distance_above_20d_high_max", grid[0]["indicators"])
        self.assertIn("breakout_volume_ratio_50_min", grid[0]["indicators"])
        self.assertIsNone(grid[0]["exit_rules"]["trailing_stop_pct"])
        self.assertIsNone(grid[0]["exit_rules"]["profit_target_pct"])
        self.assertNotIn("rsi_14_max", grid[0]["indicators"])

    def test_resolve_breakout_v1_information_technology_v2_sweep_mode(self) -> None:
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
        config = {
            "sweep_grid": {},
            "sweep_modes": {
                "breakout_v1_information_technology_v2": {
                    "sector_whitelist": ["Information Technology"],
                    "replace_base_grid": True,
                    "grid_overrides": {
                        "sma_50_dist_min": {"min": 0.0, "max": 0.0, "step": 0.1},
                        "ma_alignment_50_200_min": {"min": 0.0, "max": 0.0, "step": 0.1},
                        "ma_slope_50_20_min": {"min": 0.0, "max": 0.0, "step": 0.1},
                        "ma_slope_200_20_min": {"min": 0.0, "max": 0.0, "step": 0.1},
                        "breakout_above_20d_high_min": {"min": 1.0, "max": 1.0, "step": 1.0},
                        "distance_above_20d_high_max": {"min": 0.01, "max": 0.02, "step": 0.01},
                        "relative_strength_index_vs_spy_min": {"min": 85, "max": 90, "step": 5},
                        "roc_63_min": {"min": 0.10, "max": 0.10, "step": 0.05},
                        "rsi_14_min": {"min": 50, "max": 50, "step": 5},
                        "sma_200_dist_max": {"min": 0.30, "max": 0.30, "step": 0.05},
                        "base_range_pct_20_max": {"min": 0.08, "max": 0.12, "step": 0.04},
                        "base_atr_contraction_20_max": {"min": 0.95, "max": 0.95, "step": 0.10},
                        "base_volume_dryup_ratio_20_max": {"min": 0.95, "max": 0.95, "step": 0.10},
                        "breakout_volume_ratio_50_min": {"min": 2.0, "max": 2.5, "step": 0.5},
                        "signal_score_min": {"min": 32, "max": 34, "step": 2},
                        "trailing_stop_atr_mult": {"min": 1.5, "max": 2.0, "step": 0.5},
                        "profit_target_atr_mult": {"min": 3.0, "max": 3.0, "step": 1.0},
                    },
                }
            },
        }

        resolved, sectors = service._resolve_sweep_mode(config, "breakout_v1_information_technology_v2")
        grid = service._build_parameter_grid(resolved)

        self.assertEqual(sectors, ["Information Technology"])
        self.assertEqual(len(grid), 64)
        self.assertEqual({entry["indicators"]["signal_score_min"] for entry in grid}, {32.0, 34.0})
        self.assertEqual({entry["indicators"]["distance_above_20d_high_max"] for entry in grid}, {0.01, 0.02})

    def test_resolve_breakout_v1_information_technology_v3_sweep_mode(self) -> None:
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
        config = {
            "sweep_grid": {},
            "sweep_modes": {
                "breakout_v1_information_technology_v3": {
                    "sector_whitelist": ["Information Technology"],
                    "replace_base_grid": True,
                    "grid_overrides": {
                        "sma_50_dist_min": {"min": 0.0, "max": 0.0, "step": 0.1},
                        "ma_alignment_50_200_min": {"min": 0.0, "max": 0.0, "step": 0.1},
                        "ma_slope_50_20_min": {"min": 0.0, "max": 0.0, "step": 0.1},
                        "ma_slope_200_20_min": {"min": 0.0, "max": 0.0, "step": 0.1},
                        "breakout_above_20d_high_min": {"min": 1.0, "max": 1.0, "step": 1.0},
                        "distance_above_20d_high_max": {"min": 0.02, "max": 0.03, "step": 0.01},
                        "relative_strength_index_vs_spy_min": {"min": 85, "max": 90, "step": 5},
                        "roc_63_min": {"min": 0.10, "max": 0.10, "step": 0.05},
                        "rsi_14_min": {"min": 50, "max": 50, "step": 5},
                        "sma_200_dist_max": {"min": 0.30, "max": 0.30, "step": 0.05},
                        "base_range_pct_20_max": {"min": 0.08, "max": 0.12, "step": 0.04},
                        "base_atr_contraction_20_max": {"min": 0.95, "max": 0.95, "step": 0.10},
                        "base_volume_dryup_ratio_20_max": {"min": 0.95, "max": 0.95, "step": 0.10},
                        "breakout_volume_ratio_50_min": {"min": 2.0, "max": 2.5, "step": 0.5},
                        "signal_score_min": {"min": 32, "max": 34, "step": 2},
                        "trailing_stop_atr_mult": {"min": 1.5, "max": 2.0, "step": 0.5},
                        "profit_target_atr_mult": {"min": 3.0, "max": 3.0, "step": 1.0},
                    },
                }
            },
        }

        resolved, sectors = service._resolve_sweep_mode(config, "breakout_v1_information_technology_v3")
        grid = service._build_parameter_grid(resolved)

        self.assertEqual(sectors, ["Information Technology"])
        self.assertEqual(len(grid), 64)
        self.assertEqual({entry["indicators"]["signal_score_min"] for entry in grid}, {32.0, 34.0})
        self.assertEqual({entry["indicators"]["distance_above_20d_high_max"] for entry in grid}, {0.02, 0.03})

    def test_resolve_breakout_v1_information_technology_v4_sweep_mode(self) -> None:
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
        config = {
            "sweep_grid": {},
            "sweep_modes": {
                "breakout_v1_information_technology_v4": {
                    "sector_whitelist": ["Information Technology"],
                    "replace_base_grid": True,
                    "grid_overrides": {
                        "sma_50_dist_min": {"min": 0.0, "max": 0.0, "step": 0.1},
                        "ma_alignment_50_200_min": {"min": 0.0, "max": 0.0, "step": 0.1},
                        "ma_slope_50_20_min": {"min": 0.0, "max": 0.0, "step": 0.1},
                        "ma_slope_200_20_min": {"min": 0.0, "max": 0.0, "step": 0.1},
                        "breakout_above_20d_high_min": {"min": 1.0, "max": 1.0, "step": 1.0},
                        "distance_above_20d_high_max": {"min": 0.01, "max": 0.02, "step": 0.01},
                        "relative_strength_index_vs_spy_min": {"min": 80, "max": 85, "step": 5},
                        "roc_63_min": {"min": 0.10, "max": 0.10, "step": 0.05},
                        "rsi_14_min": {"min": 50, "max": 50, "step": 5},
                        "sma_200_dist_max": {"min": 0.30, "max": 0.30, "step": 0.05},
                        "base_range_pct_20_max": {"min": 0.08, "max": 0.12, "step": 0.04},
                        "base_atr_contraction_20_max": {"min": 0.95, "max": 0.95, "step": 0.10},
                        "base_volume_dryup_ratio_20_max": {"min": 0.95, "max": 0.95, "step": 0.10},
                        "breakout_volume_ratio_50_min": {"min": 2.0, "max": 2.5, "step": 0.5},
                        "signal_score_min": {"min": 32, "max": 34, "step": 2},
                        "trailing_stop_atr_mult": {"min": 1.5, "max": 2.0, "step": 0.5},
                        "profit_target_atr_mult": {"min": 3.0, "max": 3.0, "step": 1.0},
                    },
                }
            },
        }

        resolved, sectors = service._resolve_sweep_mode(config, "breakout_v1_information_technology_v4")
        grid = service._build_parameter_grid(resolved)

        self.assertEqual(sectors, ["Information Technology"])
        self.assertEqual(len(grid), 64)
        self.assertEqual({entry["indicators"]["signal_score_min"] for entry in grid}, {32.0, 34.0})
        self.assertEqual({entry["indicators"]["distance_above_20d_high_max"] for entry in grid}, {0.01, 0.02})
        self.assertEqual({entry["indicators"]["relative_strength_index_vs_spy_min"] for entry in grid}, {80.0, 85.0})

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
