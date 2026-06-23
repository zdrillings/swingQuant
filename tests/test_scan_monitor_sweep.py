from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

from src.scan.analyst_data import AnalystContext
from src.scan.backfill_service import ScanBackfillService
from src.scan.service import LearnedRankerStatus, ScanPolicy, ScanService
from src.scan.ranker import RankerValidationReport
from src.research.universe_snapshot_service import UniverseSnapshotBackfillService
from src.settings import AppPaths, RuntimeSettings
from src.sweep.service import BenchmarkContext, SweepService, _optional_finite_float
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
                        "weight": 0.0,
                        "min_train_rows": 40,
                        "min_train_dates": 8,
                        "slot_weights": {
                            "energy": 0.5,
                            "materials": 0.0,
                            "industrials": 0.0,
                        },
                        "validation_gate": {
                            "enabled": True,
                            "min_q1_q5_spread": 0.01,
                            "min_ic_mean": 0.02,
                            "min_ic_dates": 6,
                            "min_validation_blocks": 2,
                        },
                    }
                    ,
                    "slot_selection_overlay": {
                        "enabled": True,
                        "slot_weights": {
                            "energy": {
                                "distance_above_20d_high": -0.08,
                            }
                        },
                    },
                    "shortlist_model": {
                        "eligible_universe_mode": "passed_only",
                        "production_eligible_universe_mode": "passed_or_trend",
                        "production_model_scope": "sector_specific",
                        "production_model_name": "xgboost_model",
                        "production_xgboost_config": "balanced_depth4",
                        "min_opportunity_score": 0.31,
                    },
                }
            }
        )

        class FakeDB:
            pass

        service = ScanService(FakeDB(), email_sender=lambda subject, html_body, settings: None)
        self.assertEqual(service._learned_ranker_weight_for_slot("energy", policy), 0.5)
        self.assertEqual(service._learned_ranker_weight_for_slot("materials", policy), 0.0)
        self.assertEqual(service._learned_ranker_weight_for_slot("industrials", policy), 0.0)
        self.assertEqual(service._learned_ranker_weight_for_slot("other_slot", policy), 0.0)
        self.assertTrue(policy.learned_ranker_validation_enabled)
        self.assertEqual(policy.learned_ranker_min_q1_q5_spread, 0.01)
        self.assertEqual(policy.learned_ranker_min_ic_mean, 0.02)
        self.assertEqual(policy.learned_ranker_min_ic_dates, 6)
        self.assertEqual(policy.learned_ranker_min_validation_blocks, 2)
        self.assertTrue(policy.recent_selection_memory_enabled)
        self.assertEqual(policy.recent_selection_memory_scan_dates, 20)
        self.assertEqual(policy.recent_drag_min_picks, 2)
        self.assertEqual(policy.recent_missed_winner_min_count, 2)
        self.assertEqual(policy.recent_swap_feedback_gap, 0.03)
        self.assertEqual(policy.recent_swap_max_opportunity_gap, 0.10)
        self.assertTrue(policy.slot_selection_overlay_enabled)
        self.assertEqual(policy.slot_selection_overlay_weights["energy"]["distance_above_20d_high"], -0.08)
        self.assertTrue(policy.shortlist_model.use_as_candidate_source)
        self.assertEqual(policy.shortlist_model.production_eligible_universe_mode, "passed_or_trend")
        self.assertEqual(policy.shortlist_model.production_model_scope, "sector_specific")
        self.assertEqual(policy.shortlist_model.production_xgboost_config, "balanced_depth4")
        self.assertEqual(policy.shortlist_model.min_opportunity_score, 0.31)

    def test_scan_recent_selection_memory_penalizes_drags_and_boosts_missed_winners(self) -> None:
        policy = ScanPolicy.from_config({"scan_policy": {"recent_selection_memory": {"enabled": True}}})

        class FakeDB:
            pass

        service = ScanService(FakeDB(), email_sender=lambda subject, html_body, settings: None)
        candidates = pd.DataFrame(
            [
                {
                    "ticker": "XOM",
                    "strategy_slot": "energy",
                    "strategy_sector": "Energy",
                    "sector": "Energy",
                    "signal_score": 38.0,
                    "opportunity_score": 0.82,
                    "selection_score": 0.82,
                },
                {
                    "ticker": "AESI",
                    "strategy_slot": "energy",
                    "strategy_sector": "Energy",
                    "sector": "Energy",
                    "signal_score": 35.0,
                    "opportunity_score": 0.78,
                    "selection_score": 0.78,
                },
            ]
        )
        historical = pd.DataFrame(
            [
                {
                    "scan_date": "2026-04-08",
                    "ticker": "XOM",
                    "strategy_slot": "energy",
                    "selected": 1,
                    "opportunity_score": 0.86,
                    "alpha_vs_sector_10d": -0.0170,
                },
                {
                    "scan_date": "2026-04-08",
                    "ticker": "AESI",
                    "strategy_slot": "energy",
                    "selected": 0,
                    "opportunity_score": 0.70,
                    "alpha_vs_sector_10d": 0.3891,
                },
                {
                    "scan_date": "2026-04-09",
                    "ticker": "XOM",
                    "strategy_slot": "energy",
                    "selected": 1,
                    "opportunity_score": 0.84,
                    "alpha_vs_sector_10d": -0.0230,
                },
                {
                    "scan_date": "2026-04-09",
                    "ticker": "AESI",
                    "strategy_slot": "energy",
                    "selected": 0,
                    "opportunity_score": 0.71,
                    "alpha_vs_sector_10d": 0.4058,
                },
            ]
        )

        adjusted = service._apply_recent_selection_memory(
            candidates,
            scan_policy=policy,
            historical_scan_candidates=historical,
        ).sort_values("ticker").reset_index(drop=True)

        aesi = adjusted[adjusted["ticker"] == "AESI"].iloc[0]
        xom = adjusted[adjusted["ticker"] == "XOM"].iloc[0]
        self.assertGreater(float(aesi["recent_missed_winner_boost"]), 0.0)
        self.assertGreater(float(aesi["recent_feedback_adjustment"]), 0.0)
        self.assertGreater(float(xom["recent_drag_penalty"]), 0.0)
        self.assertLess(float(xom["recent_feedback_adjustment"]), 0.0)
        self.assertGreater(float(aesi["selection_score"]), float(xom["selection_score"]))

    def test_scan_uses_signal_score_as_selection_baseline_when_ranker_disabled(self) -> None:
        policy = ScanPolicy.from_config(
            {
                "scan_policy": {
                    "max_candidates_total": 1,
                    "max_candidates_per_slot": 1,
                    "max_candidates_per_sector": 1,
                    "min_opportunity_score": 0.55,
                    "learned_ranker": {
                        "weight": 0.0,
                        "min_train_rows": 40,
                        "min_train_dates": 8,
                    },
                    "recent_selection_memory": {"enabled": False},
                }
            }
        )

        class FakeDB:
            pass

        service = ScanService(FakeDB(), email_sender=lambda subject, html_body, settings: None)
        candidates = pd.DataFrame(
            [
                {
                    "ticker": "AAA",
                    "strategy_slot": "energy",
                    "strategy_sector": "Energy",
                    "sector": "Energy",
                    "signal_score": 38.0,
                    "opportunity_score": 0.70,
                    "md_volume_30d": 20_000_000,
                    "already_owned": False,
                },
                {
                    "ticker": "BBB",
                    "strategy_slot": "energy",
                    "strategy_sector": "Energy",
                    "sector": "Energy",
                    "signal_score": 34.0,
                    "opportunity_score": 0.85,
                    "md_volume_30d": 20_000_000,
                    "already_owned": False,
                },
            ]
        )

        scored, status = service._apply_learned_ranker(
            candidates,
            scan_policy=policy,
            historical_scan_candidates=pd.DataFrame(),
        )
        selected = service._apply_portfolio_caps(scored, policy)

        self.assertFalse(status.enabled)
        self.assertEqual(status.reason, "No historical scan candidates found.")
        self.assertEqual(float(scored.loc[scored["ticker"] == "AAA", "selection_score"].iloc[0]), 38.0)
        self.assertEqual(float(scored.loc[scored["ticker"] == "BBB", "selection_score"].iloc[0]), 34.0)
        self.assertEqual(len(selected.index), 1)
        self.assertEqual(selected.iloc[0]["ticker"], "AAA")

    def test_scan_uses_shortlist_model_candidates_even_when_old_gate_is_empty(self) -> None:
        class FakeDB:
            def initialize(self): return None
            def list_universe_rows(self, active_only=True):
                return [
                    {"ticker": "AAA", "sector": "Energy", "md_volume_30d": 40_000_000},
                    {"ticker": "BBB", "sector": "Energy", "md_volume_30d": 38_000_000},
                ]
            def load_price_history(self, tickers): return pd.DataFrame()
            def list_open_trades(self): return []

        email_calls: list[EmailCall] = []
        service = ScanService(FakeDB(), email_sender=lambda subject, html_body, settings: email_calls.append(EmailCall(subject, html_body)))
        snapshot = pd.DataFrame(
            [
                {"ticker": "AAA", "sector": "Energy", "regime_green": True, "regime_etf": "SPY", "adj_close": 50.0, "atr_14": 2.0, "md_volume_30d": 40_000_000, "roc_63": 0.12, "relative_strength_index_vs_spy": 85.0, "vol_alpha": 1.2, "sma_200_dist": 0.12, "sma_50_dist": 0.08, "rsi_14": 50.0, "sector_pct_above_50": 0.8, "sector_pct_above_200": 0.8, "sector_median_roc_63": 0.08},
                {"ticker": "BBB", "sector": "Energy", "regime_green": True, "regime_etf": "SPY", "adj_close": 48.0, "atr_14": 2.0, "md_volume_30d": 38_000_000, "roc_63": 0.11, "relative_strength_index_vs_spy": 84.0, "vol_alpha": 1.2, "sma_200_dist": 0.12, "sma_50_dist": 0.08, "rsi_14": 49.0, "sector_pct_above_50": 0.8, "sector_pct_above_200": 0.8, "sector_median_roc_63": 0.08},
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
            indicators={"rsi_14_max": 55.0},
            exit_rules=ExitRules(0.05, 0.12, 20),
            slot="energy",
            sector="Energy",
        )
        model_context = type(
            "ModelContext",
            (),
            {
                "generated_at": "2026-05-26T16:33:14+00:00",
                "champion_model": "xgboost_model",
                "live_snapshot_date": "2026-05-19",
                "top_n": 10,
                "live_predictions": pd.DataFrame(
                    [
                        {
                            "ticker": "BBB",
                            "predicted_alpha": 0.20,
                            "model_rank": 1,
                            "md_volume_30d": 38_000_000,
                            "model_reason_summary": "strong 63d momentum, strong RS vs SPY",
                            "model_comparison_summary": "AAA in Energy on strong 63d momentum",
                        },
                        {
                            "ticker": "AAA",
                            "predicted_alpha": 0.10,
                            "model_rank": 2,
                            "md_volume_30d": 40_000_000,
                            "model_reason_summary": "strong RS vs SPY",
                        },
                    ]
                ),
            },
        )()

        with patch.object(service, "_download_recent_daily_history", return_value=pd.DataFrame()), \
             patch("src.scan.service.build_analysis_frame", return_value=(pd.DataFrame(), [])), \
             patch("src.scan.service.latest_snapshot", return_value=snapshot), \
             patch("src.scan.service.filter_signal_candidates", return_value=pd.DataFrame()), \
             patch("src.scan.service.get_settings", return_value=settings), \
             patch("src.scan.service.load_feature_config", return_value={"scan_policy": {"max_candidates_total": 1, "max_candidates_per_slot": 1, "max_candidates_per_sector": 1, "min_opportunity_score": 0.55}}), \
             patch("src.scan.service.load_active_strategies", return_value={"energy": strategy}), \
             patch("src.scan.service.load_live_shortlist_model_context", return_value=model_context):
            report = service.run()

        self.assertTrue(report.emailed)
        self.assertEqual(report.candidate_count, 1)
        html = email_calls[0].html_body
        self.assertIn("BBB", html)
        self.assertIn("Strategy sector scope: Energy | Candidates: 1", html)
        self.assertIn("Legacy Signal", html)
        self.assertIn("<td>n/a</td>", html)
        self.assertIn("model +0.200 rank #1 (xgboost_model)", html)
        self.assertIn("won over AAA in Energy on strong 63d momentum", html)

    def test_scan_refuses_stale_snapshot_when_freshness_gate_enabled(self) -> None:
        class FakeDB:
            def initialize(self): return None
            def list_universe_rows(self, active_only=True):
                return [{"ticker": "AAA", "sector": "Energy", "md_volume_30d": 40_000_000}]
            def load_price_history(self, tickers): return pd.DataFrame()
            def list_open_trades(self): return []

        service = ScanService(FakeDB(), email_sender=lambda subject, html_body, settings: None)
        snapshot = pd.DataFrame(
            [
                {
                    "ticker": "AAA",
                    "date": pd.Timestamp("2000-01-03"),
                    "sector": "Energy",
                    "regime_green": True,
                    "regime_etf": "SPY",
                    "adj_close": 50.0,
                    "atr_14": 2.0,
                    "md_volume_30d": 40_000_000,
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
            indicators={"rsi_14_max": 55.0},
            exit_rules=ExitRules(0.05, 0.12, 20),
            slot="energy",
            sector="Energy",
        )

        with patch.object(service, "_download_recent_daily_history", return_value=pd.DataFrame()), \
             patch("src.scan.service.build_analysis_frame", return_value=(pd.DataFrame(), [])), \
             patch("src.scan.service.latest_snapshot", return_value=snapshot), \
             patch("src.scan.service.get_settings", return_value=settings), \
             patch("src.scan.service.load_feature_config", return_value={"scan_policy": {"max_snapshot_age_days": 4}}), \
             patch("src.scan.service.load_active_strategies", return_value={"energy": strategy}):
            with self.assertRaisesRegex(ValueError, "Scan snapshot is stale"):
                service.run()

    def test_scan_applies_shortlist_model_opportunity_floor_before_caps(self) -> None:
        class FakeDB:
            def initialize(self): return None
            def list_universe_rows(self, active_only=True):
                return [
                    {"ticker": "AAA", "sector": "Energy", "md_volume_30d": 40_000_000},
                    {"ticker": "BBB", "sector": "Energy", "md_volume_30d": 38_000_000},
                ]
            def load_price_history(self, tickers): return pd.DataFrame()
            def list_open_trades(self): return []

        email_calls: list[EmailCall] = []
        service = ScanService(FakeDB(), email_sender=lambda subject, html_body, settings: email_calls.append(EmailCall(subject, html_body)))
        snapshot = pd.DataFrame(
            [
                {"ticker": "AAA", "sector": "Energy", "regime_green": True, "regime_etf": "SPY", "adj_close": 50.0, "atr_14": 2.0, "md_volume_30d": 40_000_000, "roc_63": 0.12, "relative_strength_index_vs_spy": 85.0, "vol_alpha": 1.2, "sma_200_dist": 0.12, "sma_50_dist": 0.08, "rsi_14": 50.0, "sector_pct_above_50": 0.8, "sector_pct_above_200": 0.8, "sector_median_roc_63": 0.08},
                {"ticker": "BBB", "sector": "Energy", "regime_green": True, "regime_etf": "SPY", "adj_close": 48.0, "atr_14": 2.0, "md_volume_30d": 38_000_000, "roc_63": -0.05, "relative_strength_index_vs_spy": 10.0, "vol_alpha": 0.8, "sma_200_dist": -0.12, "sma_50_dist": -0.08, "rsi_14": 49.0, "sector_pct_above_50": 0.0, "sector_pct_above_200": 0.0, "sector_median_roc_63": -0.05},
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
            indicators={"rsi_14_max": 55.0},
            exit_rules=ExitRules(0.05, 0.12, 20),
            slot="energy",
            sector="Energy",
        )
        model_context = type(
            "ModelContext",
            (),
            {
                "generated_at": "2026-05-26T16:33:14+00:00",
                "champion_model": "xgboost_model",
                "live_snapshot_date": "2026-05-19",
                "top_n": 10,
                "live_predictions": pd.DataFrame(
                    [
                        {"ticker": "BBB", "predicted_alpha": 0.30, "model_rank": 1, "md_volume_30d": 38_000_000},
                        {"ticker": "AAA", "predicted_alpha": 0.10, "model_rank": 2, "md_volume_30d": 40_000_000},
                    ]
                ),
            },
        )()

        with patch.object(service, "_download_recent_daily_history", return_value=pd.DataFrame()), \
             patch("src.scan.service.build_analysis_frame", return_value=(pd.DataFrame(), [])), \
             patch("src.scan.service.latest_snapshot", return_value=snapshot), \
             patch("src.scan.service.get_settings", return_value=settings), \
             patch("src.scan.service.load_feature_config", return_value={"scan_policy": {"max_candidates_total": 1, "max_candidates_per_slot": 1, "max_candidates_per_sector": 1, "shortlist_model": {"min_opportunity_score": 0.30}}}), \
             patch("src.scan.service.load_active_strategies", return_value={"energy": strategy}), \
             patch("src.scan.service.load_live_shortlist_model_context", return_value=model_context):
            report = service.run()

        self.assertTrue(report.emailed)
        self.assertEqual(report.candidate_count, 1)
        html = email_calls[0].html_body
        self.assertIn("AAA", html)
        slot_section = html.split("<h2>Slot: energy</h2>", maxsplit=1)[1]
        self.assertNotIn("BBB</td>", slot_section)
        self.assertIn("shortlist_model_min_opportunity_score=0.30", html)

    def test_evening_brief_renders_portfolio_strength_coverage(self) -> None:
        class FakeDB:
            pass

        service = ScanService(FakeDB(), email_sender=lambda subject, html_body, settings: None)
        policy = ScanPolicy.from_config({"scan_policy": {"min_opportunity_score": 0.55}})
        strategy = ProductionStrategy(
            strategy_id=1,
            promoted_at="2026-05-05T17:00:00",
            indicators={"rsi_14_max": 55.0},
            exit_rules=ExitRules(0.05, 0.12, 20),
            slot="energy",
            sector="Energy",
        )
        all_candidates = pd.DataFrame(
            [
                {
                    "ticker": "AAA",
                    "strategy_slot": "energy",
                    "strategy_sector": "Energy",
                    "sector": "Energy",
                    "regime_etf": "SPY",
                    "selection_score": 0.70,
                    "opportunity_score": -0.20,
                    "overlap_penalty": 1.0,
                    "signal_score": 0.0,
                    "md_volume_30d": 100_000_000,
                    "already_owned": True,
                    "selected": 0,
                    "setup_quality_score": 0.0,
                    "expected_alpha_score": 0.8,
                    "breadth_score": 0.7,
                    "freshness_score": 0.6,
                },
                {
                    "ticker": "BBB",
                    "strategy_slot": "energy",
                    "strategy_sector": "Energy",
                    "sector": "Energy",
                    "regime_etf": "SPY",
                    "selection_score": 0.90,
                    "opportunity_score": 0.85,
                    "overlap_penalty": 0.0,
                    "signal_score": 0.0,
                    "md_volume_30d": 120_000_000,
                    "already_owned": False,
                    "selected": 1,
                    "setup_quality_score": 0.0,
                    "expected_alpha_score": 0.9,
                    "breadth_score": 0.7,
                    "freshness_score": 0.6,
                },
            ]
        )
        selected = all_candidates[all_candidates["selected"].astype(int) == 1].copy()
        selected["adj_close"] = 50.0
        selected["shares"] = 10
        selected["atr_14"] = 2.0

        html = service._build_email_html(
            selected,
            policy,
            {"energy": strategy},
            earnings_lookup={},
            all_candidates=all_candidates,
            open_trade_tickers={"AAA", "ZZZ"},
            open_trades=[
                {
                    "ticker": "AAA",
                    "entry_date": "2026-06-12",
                    "entry_price": 40.0,
                    "shares": 10,
                    "max_price_seen": 50.0,
                    "status": "open",
                    "entry_atr": None,
                    "strategy_slot": "energy",
                }
            ],
        )

        self.assertIn("<h2>Current Target Dashboard</h2>", html)
        self.assertIn("<h3>Top Current Bets</h3>", html)
        self.assertIn("<h3>Already Held Targets</h3>", html)
        self.assertIn("<h3>Best New Targets</h3>", html)
        self.assertIn("<td>AAA</td><td>yes (", html)
        self.assertIn("<td>BBB</td><td>no</td><td>yes</td>", html)
        self.assertIn("<h2>Portfolio Strength Coverage</h2>", html)
        self.assertIn("Top 6 already held: 1/6", html)
        self.assertIn("Strongest held: AAA", html)
        self.assertIn("Strongest unheld: BBB", html)
        self.assertIn("Open holdings not in candidate set: ZZZ", html)
        self.assertIn("<td>AAA</td><td>held</td><td>not selected</td><td>0.80</td>", html)

    def test_evening_brief_renders_analyst_target_context(self) -> None:
        class FakeDB:
            pass

        service = ScanService(FakeDB(), email_sender=lambda subject, html_body, settings: None)
        policy = ScanPolicy.from_config({"scan_policy": {"min_opportunity_score": 0.30}})
        strategy = ProductionStrategy(
            strategy_id=1,
            promoted_at="2026-05-05T17:00:00",
            indicators={"rsi_14_max": 55.0},
            exit_rules=ExitRules(0.05, 0.12, 20),
            slot="energy",
            sector="Energy",
        )
        selected = pd.DataFrame(
            [
                {
                    "ticker": "AAA",
                    "strategy_slot": "energy",
                    "strategy_sector": "Energy",
                    "sector": "Energy",
                    "regime_etf": "SPY",
                    "selection_score": 0.90,
                    "opportunity_score": 0.42,
                    "overlap_penalty": 0.0,
                    "signal_score": 0.0,
                    "md_volume_30d": 120_000_000,
                    "already_owned": False,
                    "selected": 1,
                    "setup_quality_score": 0.0,
                    "expected_alpha_score": 0.9,
                    "breadth_score": 0.7,
                    "freshness_score": 0.6,
                    "model_predicted_alpha": 0.12,
                    "adj_close": 50.0,
                    "shares": 10,
                    "atr_14": 2.0,
                },
            ]
        )

        html = service._build_email_html(
            selected,
            policy,
            {"energy": strategy},
            earnings_lookup={},
            analyst_contexts={
                "AAA": AnalystContext(
                    ticker="AAA",
                    target_mean=65.0,
                    target_low=45.0,
                    target_high=80.0,
                    analyst_count=12,
                    recommendation="8 buy, 4 hold",
                )
            },
            all_candidates=selected,
            open_trade_tickers=set(),
            open_trades=[],
        )

        self.assertIn("<th>Analyst Target</th>", html)
        self.assertIn("mean 65.00 (+30.0%)", html)
        self.assertIn("range 45.00-80.00", html)
        self.assertIn("n=12", html)
        self.assertIn("8 buy, 4 hold", html)

    def test_scan_slot_selection_overlay_can_penalize_extended_energy_name(self) -> None:
        policy = ScanPolicy.from_config(
            {
                "scan_policy": {
                    "max_candidates_total": 1,
                    "max_candidates_per_slot": 1,
                    "max_candidates_per_sector": 1,
                    "min_opportunity_score": 0.55,
                    "learned_ranker": {"weight": 0.0, "min_train_rows": 40, "min_train_dates": 8},
                    "recent_selection_memory": {"enabled": False},
                    "slot_selection_overlay": {
                        "enabled": True,
                        "slot_weights": {
                            "energy": {
                                "distance_above_20d_high": -2.0,
                                "roc_63": -1.0,
                            }
                        },
                    },
                }
            }
        )

        class FakeDB:
            pass

        service = ScanService(FakeDB(), email_sender=lambda subject, html_body, settings: None)
        candidates = pd.DataFrame(
            [
                {
                    "ticker": "AAA",
                    "strategy_slot": "energy",
                    "strategy_sector": "Energy",
                    "sector": "Energy",
                    "signal_score": 38.0,
                    "selection_score": 38.0,
                    "opportunity_score": 0.75,
                    "md_volume_30d": 20_000_000,
                    "already_owned": False,
                    "distance_above_20d_high": 0.08,
                    "roc_63": 0.40,
                },
                {
                    "ticker": "BBB",
                    "strategy_slot": "energy",
                    "strategy_sector": "Energy",
                    "sector": "Energy",
                    "signal_score": 36.0,
                    "selection_score": 36.0,
                    "opportunity_score": 0.74,
                    "md_volume_30d": 20_000_000,
                    "already_owned": False,
                    "distance_above_20d_high": 0.00,
                    "roc_63": 0.10,
                },
            ]
        )

        adjusted = service._apply_slot_selection_overlays(candidates, scan_policy=policy)
        selected = service._apply_portfolio_caps(adjusted, policy)

        aaa = adjusted[adjusted["ticker"] == "AAA"].iloc[0]
        bbb = adjusted[adjusted["ticker"] == "BBB"].iloc[0]
        self.assertLess(float(aaa["slot_overlay_adjustment"]), 0.0)
        self.assertGreater(float(bbb["selection_score"]), float(aaa["selection_score"]))
        self.assertEqual(len(selected.index), 1)
        self.assertEqual(selected.iloc[0]["ticker"], "BBB")

    def test_scan_recent_slot_swap_check_can_replace_recent_drag(self) -> None:
        policy = ScanPolicy.from_config(
            {
                "scan_policy": {
                    "max_candidates_total": 1,
                    "max_candidates_per_slot": 1,
                    "max_candidates_per_sector": 1,
                    "min_opportunity_score": 0.0,
                    "recent_selection_memory": {
                        "enabled": True,
                        "swap_feedback_gap": 0.03,
                        "swap_max_opportunity_gap": 0.10,
                    },
                }
            }
        )

        class FakeDB:
            pass

        service = ScanService(FakeDB(), email_sender=lambda subject, html_body, settings: None)
        candidates = pd.DataFrame(
            [
                {
                    "ticker": "XOM",
                    "strategy_slot": "energy",
                    "strategy_sector": "Energy",
                    "sector": "Energy",
                    "selection_score": 0.82,
                    "opportunity_score": 0.82,
                    "signal_score": 38.0,
                    "md_volume_30d": 40_000_000,
                    "already_owned": False,
                    "recent_feedback_adjustment": -0.04,
                },
                {
                    "ticker": "AESI",
                    "strategy_slot": "energy",
                    "strategy_sector": "Energy",
                    "sector": "Energy",
                    "selection_score": 0.80,
                    "opportunity_score": 0.76,
                    "signal_score": 35.0,
                    "md_volume_30d": 30_000_000,
                    "already_owned": False,
                    "recent_feedback_adjustment": 0.08,
                },
            ]
        )

        selected = service._apply_portfolio_caps(candidates, policy)

        self.assertEqual(len(selected.index), 1)
        self.assertEqual(selected.iloc[0]["ticker"], "AESI")

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
        self.assertIn("chart", email_calls[0].html_body)
        self.assertIn("76.00 (-5.0%)", email_calls[0].html_body)
        self.assertIn("89.60 (+12.0%)", email_calls[0].html_body)
        self.assertIn("Opportunity Score", email_calls[0].html_body)
        self.assertIn("Opportunity Breakdown", email_calls[0].html_body)

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
        self.assertIn("Selector caps used for detailed shortlist: total=3, per_slot=2, per_sector=2", html)
        self.assertIn("Strategy sector scope: Materials | Candidates: 2", html)
        self.assertIn("Strategy sector scope: Information Technology | Candidates: 1", html)
        self.assertIn("MAT1", html)
        self.assertIn("MAT2", html)
        self.assertIn("TECH1", html)
        self.assertIn("Signal Evidence", html)
        self.assertIn("Chart", html)
        self.assertIn("Selector Note", html)

    def test_scan_persists_slot_diagnostics_for_runtime_report(self) -> None:
        class FakeDB:
            def __init__(self):
                self.slot_diagnostics = None
            def initialize(self): return None
            def list_universe_rows(self, active_only=True):
                return [
                    {"ticker": "IND1", "sector": "Industrials", "md_volume_30d": 40_000_000},
                    {"ticker": "IND2", "sector": "Industrials", "md_volume_30d": 38_000_000},
                ]
            def load_price_history(self, tickers): return pd.DataFrame()
            def replace_scan_candidates(self, *, scan_date, rows): return len(list(rows))
            def replace_scan_slot_diagnostics(self, *, scan_date, rows):
                self.slot_diagnostics = {"scan_date": scan_date, "rows": list(rows)}
                return len(self.slot_diagnostics["rows"])

        db = FakeDB()
        email_calls: list[EmailCall] = []
        service = ScanService(db, email_sender=lambda subject, html_body, settings: email_calls.append(EmailCall(subject, html_body)))
        snapshot = pd.DataFrame(
            [
                {
                    "ticker": "IND1",
                    "sector": "Industrials",
                    "regime_green": True,
                    "regime_etf": "SPY",
                    "adj_close": 50.0,
                    "atr_14": 2.0,
                    "md_volume_30d": 40_000_000,
                    "signal_score": 35.0,
                    "relative_strength_index_vs_spy": 91.0,
                    "roc_63": 0.21,
                    "rsi_14": 50.0,
                    "vol_alpha": 0.9,
                    "sma_200_dist": 0.15,
                    "sector_pct_above_50": 0.2,
                    "sector_pct_above_200": 0.2,
                    "sector_median_roc_63": 0.02,
                    "distance_above_20d_high": 0.01,
                },
                {
                    "ticker": "IND2",
                    "sector": "Industrials",
                    "regime_green": True,
                    "regime_etf": "SPY",
                    "adj_close": 48.0,
                    "atr_14": 2.0,
                    "md_volume_30d": 38_000_000,
                    "signal_score": 33.0,
                    "relative_strength_index_vs_spy": 92.0,
                    "roc_63": 0.22,
                    "rsi_14": 51.0,
                    "vol_alpha": 0.85,
                    "sma_200_dist": 0.14,
                    "sector_pct_above_50": 0.1,
                    "sector_pct_above_200": 0.1,
                    "sector_median_roc_63": 0.01,
                    "distance_above_20d_high": 0.02,
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
        industrials = ProductionStrategy(
            strategy_id=3,
            promoted_at="2026-05-05T17:00:00",
            indicators={
                "relative_strength_index_vs_spy_min": 90.0,
                "signal_score_min": 32.0,
                "roc_63_min": 0.2,
                "rsi_14_max": 55.0,
                "sma_200_dist_min": 0.14,
                "vol_alpha_min": 0.8,
            },
            exit_rules=ExitRules(0.05, 0.12, 20),
            slot="industrials",
            sector="Industrials",
        )

        with patch.object(service, "_download_recent_daily_history", return_value=pd.DataFrame()), \
             patch("src.scan.service.build_analysis_frame", return_value=(pd.DataFrame(), [])), \
             patch("src.scan.service.latest_snapshot", return_value=snapshot), \
             patch("src.scan.service.filter_signal_candidates", side_effect=lambda frame, indicators: frame.copy()), \
             patch("src.scan.service.get_settings", return_value=settings), \
             patch("src.scan.service.load_feature_config", return_value={"scan_policy": {"max_candidates_total": 3, "max_candidates_per_slot": 3, "max_candidates_per_sector": 3, "pre_cap_candidates_per_slot": 5, "min_opportunity_score": 0.55}}), \
             patch("src.scan.service.load_active_strategies", return_value={"industrials": industrials}):
            service.run()

        self.assertIsNotNone(db.slot_diagnostics)
        row = db.slot_diagnostics["rows"][0]
        self.assertEqual(row["strategy_slot"], "industrials")
        self.assertEqual(row["gated_count"], 2)
        self.assertEqual(row["cleared_opportunity_count"] + row["dropped_after_opportunity_count"], 2)
        self.assertIn("gate_counts", row)
        self.assertIn("component_positive_counts", row)

    def test_scan_email_adds_earnings_columns_and_hides_unavailable_stop_target(self) -> None:
        class FakeDB:
            def initialize(self): return None
            def list_universe_rows(self, active_only=True):
                return [{"ticker": "AAA", "sector": "Information Technology", "md_volume_30d": 30_000_000}]
            def load_price_history(self, tickers): return pd.DataFrame()
            def load_earnings_calendar(self, tickers):
                return pd.DataFrame(
                    [
                        {"ticker": "AAA", "earnings_date": pd.Timestamp("2026-05-08")},
                        {"ticker": "AAA", "earnings_date": pd.Timestamp("2026-05-15")},
                    ]
                )

        email_calls: list[EmailCall] = []
        service = ScanService(FakeDB(), email_sender=lambda subject, html_body, settings: email_calls.append(EmailCall(subject, html_body)))
        snapshot = pd.DataFrame(
            [
                {
                    "ticker": "AAA",
                    "sector": "Information Technology",
                    "regime_green": True,
                    "regime_etf": "QQQ",
                    "adj_close": 100.0,
                    "atr_14": 4.0,
                    "md_volume_30d": 30_000_000,
                    "signal_score": 36.0,
                    "roc_63": 0.12,
                    "relative_strength_index_vs_spy": 85.0,
                    "relative_strength_index_vs_qqq": 84.0,
                    "relative_strength_index_vs_xlk": 86.0,
                    "vol_alpha": 1.2,
                    "sma_200_dist": 0.12,
                    "sma_50_dist": 0.08,
                    "rsi_14": 52.0,
                    "days_to_next_earnings": 3.0,
                    "days_since_last_earnings": 2.0,
                    "last_earnings_gap_pct": 0.045,
                    "close_vs_last_earnings_close": 0.018,
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
            promoted_at="2026-05-12T17:00:00",
            indicators={"rsi_14_max": 35.0},
            exit_rules=ExitRules(0.05, None, 20),
            slot="technology",
            sector="Information Technology",
        )

        with patch.object(service, "_download_recent_daily_history", return_value=pd.DataFrame()), \
             patch("src.scan.service.build_analysis_frame", return_value=(pd.DataFrame(), [])), \
             patch("src.scan.service.latest_snapshot", return_value=snapshot), \
             patch("src.scan.service.filter_signal_candidates", return_value=snapshot.copy()), \
             patch("src.scan.service.get_settings", return_value=settings), \
             patch("src.scan.service.load_feature_config", return_value={"scan_policy": {"max_candidates_total": 6, "max_candidates_per_slot": 3, "max_candidates_per_sector": 3, "pre_cap_candidates_per_slot": 5, "min_opportunity_score": 0.0}}), \
             patch("src.scan.service.load_active_strategies", return_value={"technology": strategy}), \
             patch("src.scan.service.date", wraps=__import__("datetime").date) as mock_date:
            mock_date.today.return_value = __import__("datetime").date(2026, 5, 12)
            report = service.run()

        self.assertTrue(report.emailed)
        html = email_calls[0].html_body
        self.assertIn("Next Earnings", html)
        self.assertIn("Last Earnings", html)
        self.assertIn("Earnings Note", html)
        self.assertIn("Earnings Status", html)
        self.assertIn("2026-05-15 (3bd)", html)
        self.assertIn("2026-05-08 (2bd ago)", html)
        self.assertIn("gap +4.5%, hold +1.8%", html)
        self.assertIn("before earnings (3bd)", html)
        self.assertIn("<th>Stop</th>", html)
        self.assertNotIn("<th>Target</th>", html)

    def test_scan_email_falls_back_to_relative_earnings_timing_when_calendar_dates_missing(self) -> None:
        class FakeDB:
            def initialize(self): return None
            def list_universe_rows(self, active_only=True):
                return [{"ticker": "AAA", "sector": "Information Technology", "md_volume_30d": 30_000_000}]
            def load_price_history(self, tickers): return pd.DataFrame()
            def load_earnings_calendar(self, tickers):
                return pd.DataFrame(columns=["ticker", "earnings_date"])

        email_calls: list[EmailCall] = []
        service = ScanService(FakeDB(), email_sender=lambda subject, html_body, settings: email_calls.append(EmailCall(subject, html_body)))
        snapshot = pd.DataFrame(
            [
                {
                    "ticker": "AAA",
                    "sector": "Information Technology",
                    "regime_green": True,
                    "regime_etf": "QQQ",
                    "adj_close": 100.0,
                    "atr_14": 4.0,
                    "md_volume_30d": 30_000_000,
                    "signal_score": 36.0,
                    "roc_63": 0.12,
                    "relative_strength_index_vs_spy": 85.0,
                    "relative_strength_index_vs_qqq": 84.0,
                    "relative_strength_index_vs_xlk": 86.0,
                    "vol_alpha": 1.2,
                    "sma_200_dist": 0.12,
                    "sma_50_dist": 0.08,
                    "rsi_14": 52.0,
                    "days_to_next_earnings": 4.0,
                    "days_since_last_earnings": 6.0,
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
            promoted_at="2026-05-12T17:00:00",
            indicators={"rsi_14_max": 35.0},
            exit_rules=ExitRules(0.05, 0.12, 20),
            slot="technology",
            sector="Information Technology",
        )

        with patch.object(service, "_download_recent_daily_history", return_value=pd.DataFrame()), \
             patch("src.scan.service.build_analysis_frame", return_value=(pd.DataFrame(), [])), \
             patch("src.scan.service.latest_snapshot", return_value=snapshot), \
             patch("src.scan.service.filter_signal_candidates", return_value=snapshot.copy()), \
             patch("src.scan.service.get_settings", return_value=settings), \
             patch("src.scan.service.load_feature_config", return_value={"scan_policy": {"max_candidates_total": 6, "max_candidates_per_slot": 3, "max_candidates_per_sector": 3, "pre_cap_candidates_per_slot": 5, "min_opportunity_score": 0.0}}), \
             patch("src.scan.service.load_active_strategies", return_value={"technology": strategy}):
            report = service.run()

        self.assertTrue(report.emailed)
        html = email_calls[0].html_body
        self.assertIn("~4bd", html)
        self.assertIn("~6bd ago", html)
        self.assertIn("reaction data unavailable", html)

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
        self.assertIn("Strategy sector scope: Industrials | Candidates: 1", html)

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
        self.assertIn("overlap 0.08", html)

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
        self.assertIn("Strategy sector scope: Energy | Candidates: 1", html)
        self.assertIn("learned +0.060", html)

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
                history_dates = pd.bdate_range("2026-03-02", periods=35)
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

    def test_scan_disables_learned_ranker_when_validation_gate_fails(self) -> None:
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
                history_dates = pd.bdate_range("2026-03-02", periods=35)
                rows = []
                for date_index, scan_day in enumerate(history_dates):
                    for candidate_index in range(5):
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
                                "expected_alpha_score": 0.20 + (candidate_index * 0.15),
                                "breadth_score": 0.55,
                                "freshness_score": 0.70,
                                "overlap_penalty": 0.0,
                                "opportunity_score": 0.55 + (candidate_index * 0.05),
                                "selected": 1 if candidate_index < 2 else 0,
                                "alpha_vs_sector_10d": 0.01,
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
        failed_validation = RankerValidationReport(
            available=True,
            target_column="alpha_vs_sector_10d",
            validation_method="purged_walk_forward",
            embargo_days=10,
            validation_blocks=2,
            train_rows=100,
            validation_rows=20,
            train_dates=20,
            validation_dates=10,
            prediction_correlation=0.01,
            learned_mean_target=0.01,
            learned_hit_rate=0.50,
            handcrafted_mean_target=0.01,
            handcrafted_hit_rate=0.50,
            runtime_mean_target=0.01,
            runtime_hit_rate=0.50,
            latest_scan_date=None,
            latest_learned_tickers=(),
            latest_handcrafted_tickers=(),
            latest_runtime_tickers=(),
            feature_count=10,
            quintile_summaries=(),
            q1_q5_spread=-0.02,
            daily_ic_mean=-0.01,
            daily_ic_std=0.10,
            daily_ic_t_stat=-0.50,
            daily_ic_dates=10,
            learned_turnover_mean=0.20,
            learned_turnover_pairs=9,
            slot_breakdowns=(),
            sector_breakdowns=(),
        )

        with patch.object(service, "_download_recent_daily_history", return_value=pd.DataFrame()), \
             patch("src.scan.service.build_analysis_frame", return_value=(pd.DataFrame(), [])), \
             patch("src.scan.service.latest_snapshot", return_value=snapshot), \
             patch("src.scan.service.filter_signal_candidates", side_effect=lambda frame, indicators: frame.copy()), \
             patch("src.scan.service.get_settings", return_value=settings), \
             patch("src.scan.service.load_feature_config", return_value={"scan_policy": {"max_candidates_total": 1, "max_candidates_per_slot": 1, "max_candidates_per_sector": 1, "min_opportunity_score": 0.0}}), \
             patch("src.scan.service.load_active_strategies", return_value={"energy": strategy}), \
             patch("src.scan.service.CandidateRanker.evaluate", return_value=failed_validation):
            report = service.run()

        self.assertTrue(report.emailed)
        self.assertFalse(report.learned_ranker_enabled)
        self.assertIn("Validation gate failed", report.learned_ranker_reason or "")
        html = email_calls[0].html_body
        self.assertNotIn("learned", html)

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

    def test_scan_backfill_uses_historical_shortlist_model_predictions_when_available(self) -> None:
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
                        {"ticker": "SPY", "date": pd.Timestamp("2026-05-01"), "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1_000_000, "adj_close": 100.0},
                    ]
                )
            def load_scan_candidates(self, scan_date=None):
                return pd.DataFrame()
            def load_shortlist_model_predictions(
                self,
                *,
                generated_at=None,
                horizon_days=None,
                eligible_universe_mode=None,
                model_scope=None,
                dataset_split=None,
                model_name=None,
            ):
                if dataset_split != "oos":
                    return pd.DataFrame()
                return pd.DataFrame(
                    [
                        {
                            "generated_at": "2026-05-10T21:00:00",
                            "horizon_days": 20,
                            "eligible_universe_mode": "passed_or_trend",
                            "model_scope": "sector_specific",
                            "model_name": "xgboost_model",
                            "dataset_split": "oos",
                            "snapshot_date": "2026-05-01",
                            "ticker": "AAA",
                            "sector": "Industrials",
                            "md_volume_30d": 30_000_000.0,
                            "predicted_alpha": 0.12,
                            "actual_alpha_vs_sector": None,
                            "details_json": '{"model_reason_summary":"strong RS vs SPY","model_top_reasons":["strong RS vs SPY"]}',
                        }
                    ]
                )
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
                    "signal_score": 0.0,
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
            indicators={"signal_score_min": 30.0},
            exit_rules=ExitRules(0.05, 0.12, 20),
            slot="industrials",
            sector="Industrials",
        )
        shortlist_context = SimpleNamespace(
            generated_at="2026-05-10T21:00:00",
            champion_model="xgboost_model",
            live_predictions=pd.DataFrame(),
        )

        with patch("src.scan.backfill_service.build_analysis_frame", return_value=(analysis_frame, [])), \
             patch("src.scan.backfill_service.get_settings", return_value=settings), \
             patch("src.scan.backfill_service.load_feature_config", return_value={"scan_policy": {"max_candidates_total": 2, "min_opportunity_score": 0.0, "shortlist_model": {"use_as_candidate_source": True, "production_eligible_universe_mode": "passed_or_trend", "production_model_scope": "sector_specific", "production_model_name": "xgboost_model", "production_xgboost_config": "balanced_depth4"}}}), \
             patch("src.scan.backfill_service.load_active_strategies", return_value={"industrials": strategy}), \
             patch.object(ScanService, "_load_shortlist_model_context", return_value=shortlist_context):
            report = service.run(date_from="2026-05-01", date_to="2026-05-01")

        self.assertEqual(report.scan_dates_processed, 1)
        persisted = db.persisted_by_date["2026-05-01"][0]
        self.assertTrue(persisted["selected"])
        ranking = persisted["details"]["ranking_components"]
        self.assertEqual(ranking["selection_source"], "shortlist_model")
        self.assertEqual(ranking["model_name"], "xgboost_model")
        self.assertAlmostEqual(ranking["model_predicted_alpha"], 0.12, places=6)

    def test_universe_backfill_replays_each_historical_date_without_using_future_rows(self) -> None:
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
                        {"ticker": "AAA", "date": pd.Timestamp("2026-05-03"), "open": 21.0, "high": 22.0, "low": 20.0, "close": 21.0, "volume": 1_000_000, "adj_close": 21.0},
                        {"ticker": "SPY", "date": pd.Timestamp("2026-05-01"), "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1_000_000, "adj_close": 100.0},
                        {"ticker": "SPY", "date": pd.Timestamp("2026-05-02"), "open": 101.0, "high": 102.0, "low": 100.0, "close": 101.0, "volume": 1_000_000, "adj_close": 101.0},
                        {"ticker": "SPY", "date": pd.Timestamp("2026-05-03"), "open": 102.0, "high": 103.0, "low": 101.0, "close": 102.0, "volume": 1_000_000, "adj_close": 102.0},
                    ]
                )
            def list_universe_daily_snapshot_dates(self):
                return []
            def replace_universe_daily_snapshots(self, *, snapshot_date, rows):
                self.persisted_by_date[snapshot_date] = list(rows)
                return len(rows)

        db = FakeDB()
        service = UniverseSnapshotBackfillService(db)
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
                    "signal_score": 35.0,
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
                    "atr_14": 2.0,
                    "md_volume_30d": 30_000_000,
                    "signal_score": 20.0,
                    "roc_63": 0.20,
                    "relative_strength_index_vs_spy": 70.0,
                    "vol_alpha": 1.0,
                    "sma_200_dist": 0.20,
                    "sma_50_dist": 0.10,
                    "rsi_14": 55.0,
                    "sector_pct_above_50": 0.7,
                    "sector_pct_above_200": 0.7,
                    "sector_median_roc_63": 0.06,
                },
            ]
        )
        strategy = ProductionStrategy(
            strategy_id=1,
            promoted_at="2026-05-05T17:00:00",
            indicators={"signal_score_min": 30.0},
            exit_rules=ExitRules(0.05, 0.12, 20),
            slot="industrials",
            sector="Industrials",
        )

        with patch("src.research.universe_snapshot_service.build_analysis_frame", return_value=(analysis_frame, [])), \
             patch("src.research.universe_snapshot_service.load_active_strategies", return_value={"industrials": strategy}), \
             patch("src.research.universe_snapshot_service.filter_signal_candidates", side_effect=lambda frame, indicators: frame[frame["signal_score"] >= 30.0].copy()):
            report = service.run(date_from="2026-05-01", date_to="2026-05-02")

        self.assertEqual(report.snapshot_dates_processed, 2)
        self.assertEqual(db.persisted_by_date["2026-05-01"][0]["adj_close"], 10.0)
        self.assertEqual(db.persisted_by_date["2026-05-02"][0]["adj_close"], 20.0)
        self.assertEqual(db.persisted_by_date["2026-05-01"][0]["passed_slots"], ["industrials"])
        self.assertEqual(db.persisted_by_date["2026-05-02"][0]["passed_slots"], [])
        self.assertAlmostEqual(db.persisted_by_date["2026-05-01"][0]["fwd_return_1d"], 1.0)

    def test_universe_backfill_refreshes_stale_existing_dates_even_when_skip_existing_is_true(self) -> None:
        class FakeDB:
            def __init__(self):
                self.persisted_by_date = {}
            def initialize(self): return None
            def list_universe_rows(self, active_only=True):
                return [{"ticker": "AAA", "sector": "Information Technology", "sub_industry": "Semiconductors", "md_volume_30d": 30_000_000}]
            def load_price_history(self, tickers):
                return pd.DataFrame(
                    [
                        {"ticker": "AAA", "date": pd.Timestamp("2026-05-01"), "open": 10.0, "high": 11.0, "low": 9.0, "close": 10.0, "volume": 1_000_000, "adj_close": 10.0},
                        {"ticker": "SMH", "date": pd.Timestamp("2026-05-01"), "open": 20.0, "high": 21.0, "low": 19.0, "close": 20.0, "volume": 1_000_000, "adj_close": 20.0},
                        {"ticker": "SPY", "date": pd.Timestamp("2026-05-01"), "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1_000_000, "adj_close": 100.0},
                    ]
                )
            def list_universe_daily_snapshot_dates(self):
                return ["2026-05-01"]
            def universe_daily_snapshot_date_needs_refresh(self, *, snapshot_date, required_non_null_columns):
                return snapshot_date == "2026-05-01"
            def replace_universe_daily_snapshots(self, *, snapshot_date, rows):
                self.persisted_by_date[snapshot_date] = list(rows)
                return len(rows)

        db = FakeDB()
        service = UniverseSnapshotBackfillService(db)
        analysis_frame = pd.DataFrame(
            [
                {
                    "ticker": "AAA",
                    "date": pd.Timestamp("2026-05-01"),
                    "sector": "Information Technology",
                    "sub_industry": "Semiconductors",
                    "subindustry_benchmark": "SMH",
                    "regime_green": True,
                    "regime_etf": "QQQ",
                    "adj_close": 10.0,
                    "atr_14": 1.0,
                    "md_volume_30d": 30_000_000,
                    "signal_score": 35.0,
                    "roc_63": 0.10,
                    "relative_strength_index_vs_spy": 80.0,
                    "relative_strength_index_vs_subindustry": 90.0,
                    "vol_alpha": 1.1,
                    "sma_200_dist": 0.10,
                    "sma_50_dist": 0.05,
                    "rsi_14": 50.0,
                    "sector_pct_above_50": 0.8,
                    "sector_pct_above_200": 0.8,
                    "sector_median_roc_63": 0.08,
                },
            ]
        )
        strategy = ProductionStrategy(
            strategy_id=1,
            promoted_at="2026-05-05T17:00:00",
            indicators={"signal_score_min": 30.0},
            exit_rules=ExitRules(0.05, 0.12, 20),
            slot="technology",
            sector="Information Technology",
        )

        with patch("src.research.universe_snapshot_service.build_analysis_frame", return_value=(analysis_frame, [])), \
             patch("src.research.universe_snapshot_service.load_active_strategies", return_value={"technology": strategy}), \
             patch("src.research.universe_snapshot_service.filter_signal_candidates", side_effect=lambda frame, indicators: frame[frame["signal_score"] >= 30.0].copy()):
            report = service.run(date_from="2026-05-01", date_to="2026-05-01", skip_existing=True)

        self.assertEqual(report.snapshot_dates_processed, 1)
        self.assertEqual(report.snapshot_dates_skipped, 0)
        self.assertEqual(db.persisted_by_date["2026-05-01"][0]["sub_industry"], "Semiconductors")
        self.assertEqual(db.persisted_by_date["2026-05-01"][0]["subindustry_benchmark"], "SMH")
        self.assertEqual(db.persisted_by_date["2026-05-01"][0]["relative_strength_index_vs_subindustry"], 90.0)


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

        self.assertTrue(report.emailed)
        self.assertEqual(report.watchlist_size, 1)
        self.assertEqual(service.db_manager.assigned, [(7, 10, "materials")])

    def test_monitor_fetches_recent_history_for_open_trade_missing_from_duckdb(self) -> None:
        entry_date = (pd.Timestamp.today().normalize() - pd.offsets.BDay(5)).date().isoformat()

        class FakeDB:
            def initialize(self): return None
            def list_open_trades(self):
                return [{"ticker": "NBIS", "entry_date": entry_date, "entry_price": 100.0, "shares": 10, "max_price_seen": 100.0, "status": "open", "entry_atr": None}]
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
                return pd.DataFrame([{"date": entry_date, "high": 100.0}])

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
            slot="industrials",
            sector="Industrials",
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

        self.assertTrue(report.emailed)
        self.assertEqual(report.watchlist_size, 1)
        self.assertEqual(report.triggered_count, 0)

    def test_monitor_does_not_skip_target_or_time_limit_when_current_price_missing(self) -> None:
        class FakeDB:
            def initialize(self): return None
            def list_open_trades(self):
                return [
                    {
                        "ticker": "MRVL",
                        "entry_date": "2026-05-15",
                        "entry_price": 100.0,
                        "shares": 10,
                        "max_price_seen": 120.0,
                        "status": "open",
                        "entry_atr": 4.0,
                        "strategy_id": 1,
                        "strategy_slot": "technology",
                    }
                ]
            def load_price_history(self, tickers):
                rows = []
                for ticker in ("MRVL", "SPY", "QQQ"):
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
                return [{"ticker": "MRVL", "sector": "Information Technology", "md_volume_30d": 30_000_000}]
            def get_latest_open_trade(self, ticker):
                return {"rowid": 1, "entry_price": 100.0, "entry_atr": 4.0, "shares": 10}
            def update_trade_max_price(self, trade_rowid, max_price_seen): return None
            def close_trade(self, trade_rowid, exit_date, exit_price): return None
            def load_recent_highs(self, ticker, limit=2):
                return pd.DataFrame([{"date": "2026-05-15", "high": 120.0}])

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
            exit_rules=ExitRules(None, None, 10, trailing_stop_atr_mult=2.0, profit_target_atr_mult=2.5),
            slot="technology",
            sector="Information Technology",
        )
        analysis_frame = pd.DataFrame(
            [
                {"ticker": "SPY", "date": pd.Timestamp("2026-06-22"), "spy_sma_200": 100.0, "qqq_sma_200": None},
                {"ticker": "QQQ", "date": pd.Timestamp("2026-06-22"), "spy_sma_200": None, "qqq_sma_200": 100.0},
                {"ticker": "MRVL", "date": pd.Timestamp("2026-06-22"), "atr_14": 4.0},
            ]
        )

        with patch.object(service, "_load_intraday_last_prices", return_value={"SPY": 105.0, "QQQ": 110.0}), \
             patch.object(service, "_download_recent_daily_history", return_value=pd.DataFrame()), \
             patch("src.monitor.service.get_settings", return_value=settings), \
             patch("src.monitor.service.load_active_strategies", return_value={"technology": strategy}), \
             patch("src.monitor.service.build_analysis_frame", return_value=(analysis_frame, [])), \
             patch("src.monitor.service.latest_rsi_2_with_intraday", side_effect=AssertionError("RSI needs a current price")):
            report = service.run()

        self.assertTrue(report.emailed)
        self.assertEqual(report.triggered_count, 1)
        self.assertIn("MRVL</td>", email_calls[0].html_body)
        self.assertIn("profit target, time limit", email_calls[0].html_body)
        self.assertIn("target already touched; current price unavailable", email_calls[0].html_body)

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

        self.assertTrue(report.emailed)
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
        self.assertIn("All Holdings", email_calls[0].html_body)
        self.assertIn("Sell Now", email_calls[0].html_body)
        self.assertIn("Trade Action", email_calls[0].html_body)
        self.assertIn("Fresh Setup", email_calls[0].html_body)
        self.assertIn("How to read this:", email_calls[0].html_body)
        self.assertIn("Main Risk", email_calls[0].html_body)
        self.assertIn("Price Context", email_calls[0].html_body)
        self.assertIn("<td>sell</td>", email_calls[0].html_body)
        self.assertIn("Fresh Setup Note", email_calls[0].html_body)

    def test_monitor_does_not_sell_small_industrials_winner_on_rsi_2_alone(self) -> None:
        entry_date = (pd.Timestamp.today().normalize() - pd.offsets.BDay(2)).date().isoformat()

        class FakeDB:
            def initialize(self): return None
            def list_open_trades(self):
                return [
                    {
                        "ticker": "AGX",
                        "entry_date": entry_date,
                        "entry_price": 100.0,
                        "shares": 10,
                        "max_price_seen": 101.0,
                        "status": "open",
                        "entry_atr": None,
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
                return [{"ticker": "AGX", "sector": "Industrials", "md_volume_30d": 30_000_000}]
            def get_latest_open_trade(self, ticker):
                return {"rowid": 1, "entry_price": 100.0, "shares": 10}
            def update_trade_max_price(self, trade_rowid, max_price_seen): return None
            def close_trade(self, trade_rowid, exit_date, exit_price): return None
            def load_recent_highs(self, ticker, limit=2):
                return pd.DataFrame([{"date": entry_date, "high": 101.0}])

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
            slot="industrials",
            sector="Industrials",
        )
        analysis_frame = pd.DataFrame(
            [{"ticker": "SPY", "date": pd.Timestamp(entry_date), "spy_sma_200": 100.0, "qqq_sma_200": None}]
        )

        with patch.object(service, "_load_intraday_last_prices", return_value={"AGX": 101.0, "SPY": 105.0, "QQQ": 110.0}), \
             patch.object(service, "_download_recent_daily_history", return_value=pd.DataFrame()), \
             patch("src.monitor.service.get_settings", return_value=settings), \
             patch("src.monitor.service.load_active_strategies", return_value={"industrials": strategy}), \
             patch("src.monitor.service.build_analysis_frame", return_value=(analysis_frame, [])), \
             patch("src.monitor.service.latest_rsi_2_with_intraday", return_value=95.0):
            report = service.run()

        self.assertTrue(report.emailed)
        self.assertEqual(report.triggered_count, 0)
        self.assertEqual(len(email_calls), 1)
        self.assertIn("No current sell signals", email_calls[0].html_body)

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

    def test_monitor_ignores_breakout_only_rows(self) -> None:
        entry_date = (pd.Timestamp.today().normalize() - pd.offsets.BDay(5)).date().isoformat()

        class FakeDB:
            def initialize(self): return None
            def list_open_trades(self):
                return [
                    {
                        "ticker": "AAA",
                        "entry_date": entry_date,
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
                return pd.DataFrame([{"date": entry_date, "high": 105.0}])

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
            [{"ticker": "SPY", "date": pd.Timestamp(entry_date), "spy_sma_200": 100.0, "qqq_sma_200": None}]
        )

        with patch.object(service, "_load_intraday_last_prices", return_value={"AAA": 101.0, "SPY": 105.0, "QQQ": 110.0}), \
             patch.object(service, "_download_recent_daily_history", return_value=pd.DataFrame()), \
             patch("src.monitor.service.get_settings", return_value=settings), \
             patch("src.monitor.service.load_active_strategies", return_value={"default": strategy}), \
             patch("src.monitor.service.build_analysis_frame", return_value=(analysis_frame, [])), \
             patch("src.monitor.service.latest_rsi_2_with_intraday", return_value=20.0):
            report = service.run()

        self.assertTrue(report.emailed)
        self.assertEqual(report.triggered_count, 0)
        self.assertEqual(len(email_calls), 1)
        self.assertIn("No current sell signals", email_calls[0].html_body)
        self.assertIn("All Holdings", email_calls[0].html_body)
        self.assertIn("<td>hold</td>", email_calls[0].html_body)

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

    def test_monitor_includes_buy_setup_context_for_sell_rows(self) -> None:
        class FakeDB:
            def initialize(self): return None
            def list_open_trades(self):
                return [
                    {
                        "ticker": "AAA",
                        "entry_date": "2026-04-01",
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
            indicators={"relative_strength_index_vs_spy_min": 80.0, "signal_score_min": 10.0, "rsi_14_max": 55.0},
            exit_rules=ExitRules(0.05, 0.12, 20),
        )
        analysis_frame = pd.DataFrame(
            [
                {"ticker": "SPY", "date": pd.Timestamp("2026-05-05"), "spy_sma_200": 100.0, "qqq_sma_200": None},
                {
                    "ticker": "AAA",
                    "date": pd.Timestamp("2026-05-05"),
                    "relative_strength_index_vs_spy": 85.0,
                    "rsi_14": 50.0,
                },
            ]
        )

        with patch.object(service, "_load_intraday_last_prices", return_value={"AAA": 115.0, "SPY": 105.0, "QQQ": 110.0}), \
             patch.object(service, "_download_recent_daily_history", return_value=pd.DataFrame()), \
             patch("src.monitor.service.get_settings", return_value=settings), \
             patch("src.monitor.service.load_active_strategies", return_value={"default": strategy}), \
             patch("src.monitor.service.build_analysis_frame", return_value=(analysis_frame, [])), \
             patch("src.monitor.service.latest_rsi_2_with_intraday", return_value=20.0):
            report = service.run()

        self.assertTrue(report.emailed)
        self.assertEqual(report.triggered_count, 1)
        self.assertIn("valid: hard filters passed; signal score", email_calls[0].html_body)
        self.assertIn("sell now: profit target, time limit", email_calls[0].html_body)
        self.assertIn(">buyable<", email_calls[0].html_body)

    def test_monitor_uses_shortlist_model_context_for_buyable_signal(self) -> None:
        class FakeDB:
            def initialize(self): return None
            def list_open_trades(self):
                return [
                    {
                        "ticker": "AAA",
                        "entry_date": "2026-04-01",
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
            indicators={"relative_strength_index_vs_spy_min": 80.0, "signal_score_min": 10.0, "rsi_14_max": 55.0},
            exit_rules=ExitRules(0.05, 0.12, 20),
        )
        analysis_frame = pd.DataFrame(
            [
                {"ticker": "SPY", "date": pd.Timestamp("2026-05-05"), "spy_sma_200": 100.0, "qqq_sma_200": None},
                {
                    "ticker": "AAA",
                    "date": pd.Timestamp("2026-05-05"),
                    "relative_strength_index_vs_spy": 85.0,
                    "rsi_14": 50.0,
                },
            ]
        )
        model_context = type(
            "ModelContext",
            (),
            {
                "generated_at": "2026-05-26T16:33:14+00:00",
                "champion_model": "xgboost_model",
                "live_snapshot_date": "2026-05-19",
                "top_n": 10,
                "live_predictions": pd.DataFrame(
                    [
                        {
                            "ticker": "AAA",
                            "predicted_alpha": 0.12,
                            "model_rank": 1,
                            "md_volume_30d": 30_000_000,
                            "model_reason_summary": "strong 63d momentum, strong RS vs SPY",
                            "model_comparison_summary": "next-ranked BBB on strong 63d momentum",
                        },
                    ]
                ),
            },
        )()

        with patch.object(service, "_load_intraday_last_prices", return_value={"AAA": 101.0, "SPY": 105.0, "QQQ": 110.0}), \
             patch.object(service, "_download_recent_daily_history", return_value=pd.DataFrame()), \
             patch("src.monitor.service.get_settings", return_value=settings), \
             patch("src.monitor.service.load_active_strategies", return_value={"default": strategy}), \
             patch("src.monitor.service.build_analysis_frame", return_value=(analysis_frame, [])), \
             patch("src.monitor.service.latest_rsi_2_with_intraday", return_value=20.0), \
             patch("src.monitor.service.load_live_shortlist_model_context", return_value=model_context):
            report = service.run()

        self.assertTrue(report.emailed)
        html = email_calls[0].html_body
        self.assertIn("model shortlist #1; predicted alpha 12.00% (xgboost_model)", html)
        self.assertIn("won over: next-ranked BBB on strong 63d momentum", html)
        self.assertIn(">buyable<", html)

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

    def test_resolve_breakout_v1_information_technology_continuation_sweep_mode(self) -> None:
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
                "breakout_v1_information_technology_continuation": {
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
                        "relative_strength_index_vs_qqq_min": {"min": 80, "max": 85, "step": 5},
                        "relative_strength_index_vs_xlk_min": {"min": 80, "max": 85, "step": 5},
                        "roc_63_min": {"min": 0.10, "max": 0.15, "step": 0.05},
                        "roc_126_min": {"min": 0.20, "max": 0.30, "step": 0.10},
                        "rsi_14_min": {"min": 50, "max": 55, "step": 5},
                        "sma_200_dist_max": {"min": 0.25, "max": 0.30, "step": 0.05},
                        "base_range_pct_20_max": {"min": 0.08, "max": 0.10, "step": 0.02},
                        "base_atr_contraction_20_max": {"min": 0.85, "max": 0.95, "step": 0.10},
                        "base_volume_dryup_ratio_20_max": {"min": 0.85, "max": 0.95, "step": 0.10},
                        "breakout_volume_ratio_50_min": {"min": 1.8, "max": 2.2, "step": 0.4},
                        "signal_score_min": {"min": 34, "max": 36, "step": 2},
                        "trailing_stop_atr_mult": {"min": 1.5, "max": 2.0, "step": 0.5},
                        "profit_target_atr_mult": {"min": 3.0, "max": 3.5, "step": 0.5},
                    },
                }
            },
        }

        resolved, sectors = service._resolve_sweep_mode(config, "breakout_v1_information_technology_continuation")
        grid = service._build_parameter_grid(resolved)

        self.assertEqual(sectors, ["Information Technology"])
        self.assertTrue(grid)
        self.assertIn("relative_strength_index_vs_qqq_min", grid[0]["indicators"])
        self.assertIn("relative_strength_index_vs_xlk_min", grid[0]["indicators"])
        self.assertIn("roc_126_min", grid[0]["indicators"])
        self.assertEqual({entry["indicators"]["relative_strength_index_vs_qqq_min"] for entry in grid}, {80.0, 85.0})
        self.assertEqual({entry["indicators"]["relative_strength_index_vs_xlk_min"] for entry in grid}, {80.0, 85.0})

    def test_resolve_breakout_v1_information_technology_continuation_v2_sweep_mode(self) -> None:
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
                "breakout_v1_information_technology_continuation_v2": {
                    "sector_whitelist": ["Information Technology"],
                    "replace_base_grid": True,
                    "grid_overrides": {
                        "sma_50_dist_min": {"min": 0.0, "max": 0.0, "step": 0.1},
                        "ma_alignment_50_200_min": {"min": 0.0, "max": 0.0, "step": 0.1},
                        "ma_slope_50_20_min": {"min": 0.0, "max": 0.0, "step": 0.1},
                        "ma_slope_200_20_min": {"min": 0.0, "max": 0.0, "step": 0.1},
                        "breakout_above_20d_high_min": {"min": 1.0, "max": 1.0, "step": 1.0},
                        "distance_above_20d_high_max": {"min": 0.01, "max": 0.01, "step": 0.01},
                        "relative_strength_index_vs_spy_min": {"min": 80, "max": 85, "step": 5},
                        "relative_strength_index_vs_qqq_min": {"min": 80, "max": 85, "step": 5},
                        "relative_strength_index_vs_xlk_min": {"min": 85, "max": 85, "step": 5},
                        "roc_63_min": {"min": 0.10, "max": 0.15, "step": 0.05},
                        "roc_126_min": {"min": 0.20, "max": 0.20, "step": 0.10},
                        "rsi_14_min": {"min": 50, "max": 50, "step": 5},
                        "sma_200_dist_max": {"min": 0.25, "max": 0.25, "step": 0.05},
                        "base_range_pct_20_max": {"min": 0.08, "max": 0.08, "step": 0.02},
                        "base_atr_contraction_20_max": {"min": 0.85, "max": 0.95, "step": 0.10},
                        "base_volume_dryup_ratio_20_max": {"min": 0.85, "max": 0.85, "step": 0.10},
                        "breakout_volume_ratio_50_min": {"min": 1.6, "max": 2.0, "step": 0.2},
                        "signal_score_min": {"min": 34, "max": 36, "step": 2},
                        "trailing_stop_atr_mult": {"min": 1.5, "max": 2.0, "step": 0.5},
                        "profit_target_atr_mult": {"min": 2.0, "max": 3.0, "step": 0.5},
                        "time_limit_days": {"min": 5, "max": 10, "step": 5},
                    },
                }
            },
        }

        resolved, sectors = service._resolve_sweep_mode(config, "breakout_v1_information_technology_continuation_v2")
        grid = service._build_parameter_grid(resolved)

        self.assertEqual(sectors, ["Information Technology"])
        self.assertEqual(len(grid), 1152)
        self.assertEqual({entry["indicators"]["distance_above_20d_high_max"] for entry in grid}, {0.01})
        self.assertEqual({entry["indicators"]["relative_strength_index_vs_xlk_min"] for entry in grid}, {85.0})
        self.assertEqual({entry["indicators"]["breakout_volume_ratio_50_min"] for entry in grid}, {1.6, 1.8, 2.0})
        self.assertEqual({entry["exit_rules"]["time_limit_days"] for entry in grid}, {5, 10})
        self.assertEqual({entry["exit_rules"]["profit_target_atr_mult"] for entry in grid}, {2.0, 2.5, 3.0})

    def test_resolve_tech_post_earnings_drift_v1_sweep_mode(self) -> None:
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
                "tech_post_earnings_drift_v1": {
                    "sector_whitelist": ["Information Technology"],
                    "replace_base_grid": True,
                    "grid_overrides": {
                        "relative_strength_index_vs_spy_min": {"min": 80, "max": 80, "step": 5},
                        "relative_strength_index_vs_qqq_min": {"min": 80, "max": 80, "step": 5},
                        "relative_strength_index_vs_xlk_min": {"min": 80, "max": 80, "step": 5},
                        "days_since_last_earnings_max": {"min": 3, "max": 5, "step": 2},
                        "days_to_next_earnings_min": {"min": 5, "max": 5, "step": 1},
                        "last_earnings_gap_pct_min": {"min": 0.03, "max": 0.05, "step": 0.02},
                        "last_earnings_volume_ratio_20_min": {"min": 1.5, "max": 2.0, "step": 0.5},
                        "last_earnings_open_vs_20d_high_min": {"min": 0.0, "max": 0.0, "step": 0.1},
                        "roc_63_min": {"min": 0.05, "max": 0.10, "step": 0.05},
                        "roc_126_min": {"min": 0.10, "max": 0.10, "step": 0.10},
                        "close_vs_last_earnings_close_min": {"min": 0.00, "max": 0.02, "step": 0.02},
                        "vol_alpha_min": {"min": 1.0, "max": 1.0, "step": 0.2},
                        "signal_score_min": {"min": 32, "max": 34, "step": 2},
                        "trailing_stop_atr_mult": {"min": 1.5, "max": 2.0, "step": 0.5},
                        "profit_target_atr_mult": {"min": 2.0, "max": 2.5, "step": 0.5},
                        "time_limit_days": {"min": 5, "max": 10, "step": 5},
                    },
                }
            },
        }

        resolved, sectors = service._resolve_sweep_mode(config, "tech_post_earnings_drift_v1")
        grid = service._build_parameter_grid(resolved)

        self.assertEqual(sectors, ["Information Technology"])
        self.assertEqual(len(grid), 512)
        self.assertEqual({entry["indicators"]["days_since_last_earnings_max"] for entry in grid}, {3.0, 5.0})
        self.assertEqual({entry["indicators"]["last_earnings_gap_pct_min"] for entry in grid}, {0.03, 0.05})
        self.assertEqual({entry["indicators"]["last_earnings_volume_ratio_20_min"] for entry in grid}, {1.5, 2.0})
        self.assertEqual({entry["exit_rules"]["time_limit_days"] for entry in grid}, {5, 10})
        self.assertEqual({entry["exit_rules"]["profit_target_atr_mult"] for entry in grid}, {2.0, 2.5})

    def test_resolve_tech_post_earnings_drift_v2_sweep_mode(self) -> None:
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
                "tech_post_earnings_drift_v2": {
                    "sector_whitelist": ["Information Technology"],
                    "replace_base_grid": True,
                    "grid_overrides": {
                        "relative_strength_index_vs_spy_min": {"min": 80, "max": 80, "step": 5},
                        "relative_strength_index_vs_qqq_min": {"min": 80, "max": 80, "step": 5},
                        "relative_strength_index_vs_xlk_min": {"min": 80, "max": 80, "step": 5},
                        "days_since_last_earnings_max": {"min": 3, "max": 7, "step": 2},
                        "days_to_next_earnings_min": {"min": 5, "max": 5, "step": 1},
                        "last_earnings_gap_pct_min": {"min": 0.02, "max": 0.05, "step": 0.01},
                        "last_earnings_volume_ratio_20_min": {"min": 1.2, "max": 1.5, "step": 0.3},
                        "last_earnings_open_vs_20d_high_min": {"min": 0.0, "max": 0.0, "step": 0.1},
                        "roc_63_min": {"min": 0.05, "max": 0.10, "step": 0.05},
                        "roc_126_min": {"min": 0.10, "max": 0.10, "step": 0.10},
                        "close_vs_last_earnings_close_min": {"min": 0.00, "max": 0.02, "step": 0.02},
                        "vol_alpha_min": {"min": 1.0, "max": 1.0, "step": 0.2},
                        "signal_score_min": {"min": 32, "max": 34, "step": 2},
                        "trailing_stop_atr_mult": {"min": 1.5, "max": 2.0, "step": 0.5},
                        "profit_target_atr_mult": {"min": 2.0, "max": 2.5, "step": 0.5},
                        "time_limit_days": {"min": 5, "max": 5, "step": 5},
                    },
                }
            },
        }

        resolved, sectors = service._resolve_sweep_mode(config, "tech_post_earnings_drift_v2")
        grid = service._build_parameter_grid(resolved)

        self.assertEqual(sectors, ["Information Technology"])
        self.assertEqual(len(grid), 768)
        self.assertEqual({entry["indicators"]["days_since_last_earnings_max"] for entry in grid}, {3.0, 5.0, 7.0})
        self.assertEqual({entry["indicators"]["last_earnings_gap_pct_min"] for entry in grid}, {0.02, 0.03, 0.04, 0.05})
        self.assertEqual({entry["indicators"]["last_earnings_volume_ratio_20_min"] for entry in grid}, {1.2, 1.5})
        self.assertEqual({entry["exit_rules"]["time_limit_days"] for entry in grid}, {5})
        self.assertEqual({entry["exit_rules"]["profit_target_atr_mult"] for entry in grid}, {2.0, 2.5})

    def test_resolve_tech_post_earnings_followthrough_v1_sweep_mode(self) -> None:
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
                "tech_post_earnings_followthrough_v1": {
                    "sector_whitelist": ["Information Technology"],
                    "replace_base_grid": True,
                    "grid_overrides": {
                        "relative_strength_index_vs_spy_min": {"min": 75, "max": 75, "step": 5},
                        "relative_strength_index_vs_qqq_min": {"min": 75, "max": 75, "step": 5},
                        "relative_strength_index_vs_subindustry_min": {"min": 75, "max": 80, "step": 5},
                        "days_since_last_earnings_max": {"min": 3, "max": 7, "step": 2},
                        "days_to_next_earnings_min": {"min": 7, "max": 7, "step": 1},
                        "last_earnings_gap_pct_min": {"min": 0.00, "max": 0.01, "step": 0.01},
                        "last_earnings_volume_ratio_20_min": {"min": 1.0, "max": 1.2, "step": 0.2},
                        "last_earnings_open_vs_20d_high_min": {"min": 0.0, "max": 0.02, "step": 0.02},
                        "roc_63_min": {"min": 0.05, "max": 0.05, "step": 0.05},
                        "roc_126_min": {"min": 0.10, "max": 0.20, "step": 0.10},
                        "close_vs_last_earnings_close_min": {"min": 0.02, "max": 0.04, "step": 0.02},
                        "signal_score_min": {"min": 30, "max": 32, "step": 2},
                        "trailing_stop_atr_mult": {"min": 1.5, "max": 1.5, "step": 0.5},
                        "profit_target_atr_mult": {"min": 1.5, "max": 2.0, "step": 0.5},
                        "time_limit_days": {"min": 3, "max": 5, "step": 2},
                    },
                }
            },
        }

        resolved, sectors = service._resolve_sweep_mode(config, "tech_post_earnings_followthrough_v1")
        grid = service._build_parameter_grid(resolved)

        self.assertEqual(sectors, ["Information Technology"])
        self.assertEqual(len(grid), 1536)
        self.assertEqual({entry["indicators"]["relative_strength_index_vs_subindustry_min"] for entry in grid}, {75.0, 80.0})
        self.assertEqual({entry["indicators"]["days_since_last_earnings_max"] for entry in grid}, {3.0, 5.0, 7.0})
        self.assertEqual({entry["indicators"]["close_vs_last_earnings_close_min"] for entry in grid}, {0.02, 0.04})
        self.assertEqual({entry["exit_rules"]["time_limit_days"] for entry in grid}, {3, 5})
        self.assertEqual({entry["exit_rules"]["profit_target_atr_mult"] for entry in grid}, {1.5, 2.0})

    def test_resolve_tech_post_earnings_followthrough_v2_sweep_mode(self) -> None:
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
                "tech_post_earnings_followthrough_v2": {
                    "sector_whitelist": ["Information Technology"],
                    "replace_base_grid": True,
                    "grid_overrides": {
                        "relative_strength_index_vs_spy_min": {"min": 75, "max": 75, "step": 5},
                        "relative_strength_index_vs_qqq_min": {"min": 75, "max": 80, "step": 5},
                        "days_since_last_earnings_max": {"min": 5, "max": 10, "step": 5},
                        "days_to_next_earnings_min": {"min": 3, "max": 5, "step": 2},
                        "close_vs_last_earnings_close_min": {"min": 0.00, "max": 0.04, "step": 0.02},
                        "roc_63_min": {"min": 0.05, "max": 0.10, "step": 0.05},
                        "roc_126_min": {"min": 0.10, "max": 0.20, "step": 0.10},
                        "vol_alpha_min": {"min": 0.8, "max": 1.0, "step": 0.2},
                        "signal_score_min": {"min": 26, "max": 30, "step": 2},
                        "trailing_stop_atr_mult": {"min": 1.5, "max": 2.0, "step": 0.5},
                        "profit_target_atr_mult": {"min": 1.5, "max": 2.5, "step": 0.5},
                        "time_limit_days": {"min": 3, "max": 5, "step": 2},
                    },
                }
            },
        }

        resolved, sectors = service._resolve_sweep_mode(config, "tech_post_earnings_followthrough_v2")
        grid = service._build_parameter_grid(resolved)

        self.assertEqual(sectors, ["Information Technology"])
        self.assertEqual(len(grid), 6912)
        self.assertEqual({entry["indicators"]["relative_strength_index_vs_qqq_min"] for entry in grid}, {75.0, 80.0})
        self.assertEqual({entry["indicators"]["days_since_last_earnings_max"] for entry in grid}, {5.0, 10.0})
        self.assertEqual({entry["indicators"]["close_vs_last_earnings_close_min"] for entry in grid}, {0.0, 0.02, 0.04})
        self.assertEqual({entry["exit_rules"]["time_limit_days"] for entry in grid}, {3, 5})
        self.assertEqual({entry["exit_rules"]["profit_target_atr_mult"] for entry in grid}, {1.5, 2.0, 2.5})

    def test_resolve_tech_post_earnings_followthrough_v3_sweep_mode(self) -> None:
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
                "tech_post_earnings_followthrough_v3": {
                    "sector_whitelist": ["Information Technology"],
                    "replace_base_grid": True,
                    "grid_overrides": {
                        "relative_strength_index_vs_spy_min": {"min": 75, "max": 75, "step": 5},
                        "relative_strength_index_vs_qqq_min": {"min": 80, "max": 80, "step": 5},
                        "days_since_last_earnings_max": {"min": 3, "max": 5, "step": 2},
                        "days_to_next_earnings_min": {"min": 3, "max": 5, "step": 2},
                        "close_vs_last_earnings_close_min": {"min": 0.00, "max": 0.02, "step": 0.02},
                        "roc_63_min": {"min": 0.05, "max": 0.05, "step": 0.05},
                        "roc_126_min": {"min": 0.10, "max": 0.20, "step": 0.10},
                        "vol_alpha_min": {"min": 0.8, "max": 0.8, "step": 0.2},
                        "signal_score_min": {"min": 28, "max": 30, "step": 2},
                        "trailing_stop_atr_mult": {"min": 1.5, "max": 2.0, "step": 0.5},
                        "profit_target_atr_mult": {"min": 2.0, "max": 2.0, "step": 0.5},
                        "time_limit_days": {"min": 5, "max": 5, "step": 5},
                    },
                }
            },
        }

        resolved, sectors = service._resolve_sweep_mode(config, "tech_post_earnings_followthrough_v3")
        grid = service._build_parameter_grid(resolved)

        self.assertEqual(sectors, ["Information Technology"])
        self.assertEqual(len(grid), 64)
        self.assertEqual({entry["indicators"]["relative_strength_index_vs_qqq_min"] for entry in grid}, {80.0})
        self.assertEqual({entry["indicators"]["days_since_last_earnings_max"] for entry in grid}, {3.0, 5.0})
        self.assertEqual({entry["indicators"]["days_to_next_earnings_min"] for entry in grid}, {3.0, 5.0})
        self.assertEqual({entry["indicators"]["close_vs_last_earnings_close_min"] for entry in grid}, {0.0, 0.02})
        self.assertEqual({entry["exit_rules"]["time_limit_days"] for entry in grid}, {5})
        self.assertEqual({entry["exit_rules"]["profit_target_atr_mult"] for entry in grid}, {2.0})

    def test_resolve_tech_infrastructure_post_earnings_v1_sweep_mode(self) -> None:
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
                "tech_infrastructure_post_earnings_v1": {
                    "sector_whitelist": ["Information Technology"],
                    "sub_industry_whitelist": [
                        "Communications Equipment",
                        "Semiconductor Materials & Equipment",
                        "Technology Hardware, Storage & Peripherals",
                        "Internet Services & Infrastructure",
                    ],
                    "replace_base_grid": True,
                    "grid_overrides": {
                        "relative_strength_index_vs_spy_min": {"min": 75, "max": 75, "step": 5},
                        "relative_strength_index_vs_qqq_min": {"min": 80, "max": 80, "step": 5},
                        "days_since_last_earnings_max": {"min": 3, "max": 5, "step": 2},
                        "days_to_next_earnings_min": {"min": 3, "max": 5, "step": 2},
                        "close_vs_last_earnings_close_min": {"min": 0.00, "max": 0.02, "step": 0.02},
                        "roc_63_min": {"min": 0.05, "max": 0.05, "step": 0.05},
                        "roc_126_min": {"min": 0.10, "max": 0.20, "step": 0.10},
                        "vol_alpha_min": {"min": 0.8, "max": 0.8, "step": 0.2},
                        "signal_score_min": {"min": 28, "max": 30, "step": 2},
                        "trailing_stop_atr_mult": {"min": 2.0, "max": 2.0, "step": 0.5},
                        "profit_target_atr_mult": {"min": 2.0, "max": 2.0, "step": 0.5},
                        "time_limit_days": {"min": 5, "max": 5, "step": 5},
                    },
                }
            },
        }

        resolved, sectors = service._resolve_sweep_mode(config, "tech_infrastructure_post_earnings_v1")
        grid = service._build_parameter_grid(resolved)

        self.assertEqual(sectors, ["Information Technology"])
        self.assertEqual(len(grid), 32)
        self.assertEqual(
            resolved["_sub_industry_whitelist"],
            [
                "Communications Equipment",
                "Semiconductor Materials & Equipment",
                "Technology Hardware, Storage & Peripherals",
                "Internet Services & Infrastructure",
            ],
        )
        self.assertEqual({entry["indicators"]["relative_strength_index_vs_qqq_min"] for entry in grid}, {80.0})
        self.assertEqual({entry["indicators"]["days_since_last_earnings_max"] for entry in grid}, {3.0, 5.0})
        self.assertEqual({entry["exit_rules"]["time_limit_days"] for entry in grid}, {5})
        self.assertEqual({entry["exit_rules"]["trailing_stop_atr_mult"] for entry in grid}, {2.0})

    def test_resolve_tech_comms_infra_post_earnings_v1_sweep_mode(self) -> None:
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
                "tech_comms_infra_post_earnings_v1": {
                    "sector_whitelist": ["Information Technology"],
                    "sub_industry_whitelist": [
                        "Communications Equipment",
                        "Internet Services & Infrastructure",
                    ],
                    "replace_base_grid": True,
                    "grid_overrides": {
                        "relative_strength_index_vs_spy_min": {"min": 75, "max": 75, "step": 5},
                        "relative_strength_index_vs_qqq_min": {"min": 80, "max": 80, "step": 5},
                        "days_since_last_earnings_max": {"min": 5, "max": 5, "step": 2},
                        "days_to_next_earnings_min": {"min": 3, "max": 5, "step": 2},
                        "close_vs_last_earnings_close_min": {"min": 0.00, "max": 0.02, "step": 0.02},
                        "roc_63_min": {"min": 0.05, "max": 0.05, "step": 0.05},
                        "roc_126_min": {"min": 0.10, "max": 0.20, "step": 0.10},
                        "vol_alpha_min": {"min": 0.8, "max": 0.8, "step": 0.2},
                        "signal_score_min": {"min": 28, "max": 30, "step": 2},
                        "trailing_stop_atr_mult": {"min": 2.0, "max": 2.0, "step": 0.5},
                        "profit_target_atr_mult": {"min": 2.0, "max": 2.0, "step": 0.5},
                        "time_limit_days": {"min": 5, "max": 5, "step": 5},
                    },
                }
            },
        }

        resolved, sectors = service._resolve_sweep_mode(config, "tech_comms_infra_post_earnings_v1")
        grid = service._build_parameter_grid(resolved)

        self.assertEqual(sectors, ["Information Technology"])
        self.assertEqual(len(grid), 16)
        self.assertEqual(
            resolved["_sub_industry_whitelist"],
            [
                "Communications Equipment",
                "Internet Services & Infrastructure",
            ],
        )
        self.assertEqual({entry["indicators"]["days_since_last_earnings_max"] for entry in grid}, {5.0})
        self.assertEqual({entry["indicators"]["close_vs_last_earnings_close_min"] for entry in grid}, {0.0, 0.02})
        self.assertEqual({entry["exit_rules"]["time_limit_days"] for entry in grid}, {5})

    def test_resolve_tech_comms_infra_post_earnings_v2_sweep_mode(self) -> None:
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
                "tech_comms_infra_post_earnings_v2": {
                    "sector_whitelist": ["Information Technology"],
                    "sub_industry_whitelist": [
                        "Communications Equipment",
                        "Internet Services & Infrastructure",
                    ],
                    "replace_base_grid": True,
                    "grid_overrides": {
                        "relative_strength_index_vs_spy_min": {"min": 75, "max": 75, "step": 5},
                        "relative_strength_index_vs_qqq_min": {"min": 80, "max": 80, "step": 5},
                        "days_since_last_earnings_max": {"min": 5, "max": 5, "step": 2},
                        "days_to_next_earnings_min": {"min": 3, "max": 3, "step": 2},
                        "close_vs_last_earnings_close_min": {"min": 0.02, "max": 0.02, "step": 0.02},
                        "roc_63_min": {"min": 0.05, "max": 0.05, "step": 0.05},
                        "roc_126_min": {"min": 0.10, "max": 0.10, "step": 0.10},
                        "vol_alpha_min": {"min": 0.8, "max": 0.8, "step": 0.2},
                        "signal_score_min": {"min": 28, "max": 30, "step": 2},
                        "trailing_stop_atr_mult": {"min": 2.0, "max": 2.0, "step": 0.5},
                        "profit_target_atr_mult": {"min": 2.0, "max": 2.0, "step": 0.5},
                        "time_limit_days": {"min": 5, "max": 7, "step": 2},
                    },
                }
            },
        }

        resolved, sectors = service._resolve_sweep_mode(config, "tech_comms_infra_post_earnings_v2")
        grid = service._build_parameter_grid(resolved)

        self.assertEqual(sectors, ["Information Technology"])
        self.assertEqual(len(grid), 4)
        self.assertEqual(
            resolved["_sub_industry_whitelist"],
            [
                "Communications Equipment",
                "Internet Services & Infrastructure",
            ],
        )
        self.assertEqual({entry["indicators"]["days_to_next_earnings_min"] for entry in grid}, {3.0})
        self.assertEqual({entry["indicators"]["close_vs_last_earnings_close_min"] for entry in grid}, {0.02})
        self.assertEqual({entry["indicators"]["signal_score_min"] for entry in grid}, {28.0, 30.0})
        self.assertEqual({entry["exit_rules"]["time_limit_days"] for entry in grid}, {5, 7})

    def test_resolve_tech_comms_infra_breakout_v1_sweep_mode(self) -> None:
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
                "tech_comms_infra_breakout_v1": {
                    "sector_whitelist": ["Information Technology"],
                    "sub_industry_whitelist": [
                        "Communications Equipment",
                        "Internet Services & Infrastructure",
                    ],
                    "replace_base_grid": True,
                    "grid_overrides": {
                        "sma_50_dist_min": {"min": 0.0, "max": 0.0, "step": 0.1},
                        "ma_alignment_50_200_min": {"min": 0.0, "max": 0.0, "step": 0.1},
                        "ma_slope_50_20_min": {"min": 0.0, "max": 0.0, "step": 0.1},
                        "ma_slope_200_20_min": {"min": 0.0, "max": 0.0, "step": 0.1},
                        "breakout_above_20d_high_min": {"min": 1.0, "max": 1.0, "step": 1.0},
                        "distance_above_20d_high_max": {"min": 0.01, "max": 0.02, "step": 0.01},
                        "relative_strength_index_vs_spy_min": {"min": 75, "max": 80, "step": 5},
                        "relative_strength_index_vs_qqq_min": {"min": 75, "max": 80, "step": 5},
                        "roc_63_min": {"min": 0.05, "max": 0.10, "step": 0.05},
                        "roc_126_min": {"min": 0.10, "max": 0.20, "step": 0.10},
                        "sma_200_dist_max": {"min": 0.25, "max": 0.30, "step": 0.05},
                        "base_range_pct_20_max": {"min": 0.08, "max": 0.12, "step": 0.04},
                        "base_atr_contraction_20_max": {"min": 0.85, "max": 0.95, "step": 0.10},
                        "base_volume_dryup_ratio_20_max": {"min": 0.85, "max": 0.95, "step": 0.10},
                        "breakout_volume_ratio_50_min": {"min": 1.6, "max": 2.0, "step": 0.4},
                        "signal_score_min": {"min": 32, "max": 34, "step": 2},
                        "trailing_stop_atr_mult": {"min": 1.5, "max": 2.0, "step": 0.5},
                        "profit_target_atr_mult": {"min": 2.5, "max": 3.0, "step": 0.5},
                        "time_limit_days": {"min": 10, "max": 10, "step": 5},
                    },
                }
            },
        }

        resolved, sectors = service._resolve_sweep_mode(config, "tech_comms_infra_breakout_v1")
        grid = service._build_parameter_grid(resolved)

        self.assertEqual(sectors, ["Information Technology"])
        self.assertEqual(len(grid), 8192)
        self.assertEqual(
            resolved["_sub_industry_whitelist"],
            [
                "Communications Equipment",
                "Internet Services & Infrastructure",
            ],
        )
        self.assertEqual({entry["indicators"]["breakout_above_20d_high_min"] for entry in grid}, {1.0})
        self.assertEqual({entry["indicators"]["distance_above_20d_high_max"] for entry in grid}, {0.01, 0.02})
        self.assertEqual({entry["indicators"]["relative_strength_index_vs_spy_min"] for entry in grid}, {75.0, 80.0})
        self.assertEqual({entry["indicators"]["relative_strength_index_vs_qqq_min"] for entry in grid}, {75.0, 80.0})
        self.assertEqual({entry["indicators"]["breakout_volume_ratio_50_min"] for entry in grid}, {1.6, 2.0})
        self.assertEqual({entry["exit_rules"]["time_limit_days"] for entry in grid}, {10})

    def test_resolve_tech_hardware_equipment_post_earnings_v1_sweep_mode(self) -> None:
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
                "tech_hardware_equipment_post_earnings_v1": {
                    "sector_whitelist": ["Information Technology"],
                    "sub_industry_whitelist": [
                        "Semiconductor Materials & Equipment",
                        "Technology Hardware, Storage & Peripherals",
                    ],
                    "replace_base_grid": True,
                    "grid_overrides": {
                        "relative_strength_index_vs_spy_min": {"min": 75, "max": 75, "step": 5},
                        "relative_strength_index_vs_qqq_min": {"min": 80, "max": 80, "step": 5},
                        "days_since_last_earnings_max": {"min": 5, "max": 5, "step": 2},
                        "days_to_next_earnings_min": {"min": 3, "max": 5, "step": 2},
                        "close_vs_last_earnings_close_min": {"min": 0.00, "max": 0.02, "step": 0.02},
                        "roc_63_min": {"min": 0.05, "max": 0.05, "step": 0.05},
                        "roc_126_min": {"min": 0.10, "max": 0.20, "step": 0.10},
                        "vol_alpha_min": {"min": 0.8, "max": 0.8, "step": 0.2},
                        "signal_score_min": {"min": 28, "max": 30, "step": 2},
                        "trailing_stop_atr_mult": {"min": 2.0, "max": 2.0, "step": 0.5},
                        "profit_target_atr_mult": {"min": 2.0, "max": 2.0, "step": 0.5},
                        "time_limit_days": {"min": 5, "max": 5, "step": 5},
                    },
                }
            },
        }

        resolved, sectors = service._resolve_sweep_mode(config, "tech_hardware_equipment_post_earnings_v1")
        grid = service._build_parameter_grid(resolved)

        self.assertEqual(sectors, ["Information Technology"])
        self.assertEqual(len(grid), 16)
        self.assertEqual(
            resolved["_sub_industry_whitelist"],
            [
                "Semiconductor Materials & Equipment",
                "Technology Hardware, Storage & Peripherals",
            ],
        )
        self.assertEqual({entry["indicators"]["days_since_last_earnings_max"] for entry in grid}, {5.0})
        self.assertEqual({entry["indicators"]["close_vs_last_earnings_close_min"] for entry in grid}, {0.0, 0.02})
        self.assertEqual({entry["exit_rules"]["time_limit_days"] for entry in grid}, {5})

    def test_resolve_tech_leader_pullback_v1_sweep_mode(self) -> None:
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
                "tech_leader_pullback_v1": {
                    "sector_whitelist": ["Information Technology"],
                    "replace_base_grid": True,
                    "grid_overrides": {
                        "sma_50_dist_min": {"min": 0.0, "max": 0.0, "step": 0.1},
                        "sma_50_dist_max": {"min": 0.02, "max": 0.06, "step": 0.02},
                        "sma_200_dist_min": {"min": 0.10, "max": 0.18, "step": 0.08},
                        "ma_alignment_50_200_min": {"min": 0.0, "max": 0.0, "step": 0.1},
                        "ma_slope_50_20_min": {"min": 0.0, "max": 0.0, "step": 0.1},
                        "ma_slope_200_20_min": {"min": 0.0, "max": 0.0, "step": 0.1},
                        "relative_strength_index_vs_spy_min": {"min": 85, "max": 85, "step": 5},
                        "relative_strength_index_vs_qqq_min": {"min": 85, "max": 85, "step": 5},
                        "relative_strength_index_vs_xlk_min": {"min": 85, "max": 85, "step": 5},
                        "roc_63_min": {"min": 0.10, "max": 0.15, "step": 0.05},
                        "roc_126_min": {"min": 0.20, "max": 0.20, "step": 0.10},
                        "rsi_14_min": {"min": 45, "max": 50, "step": 5},
                        "rsi_14_max": {"min": 55, "max": 60, "step": 5},
                        "vol_alpha_min": {"min": 1.0, "max": 1.2, "step": 0.2},
                        "signal_score_min": {"min": 32, "max": 34, "step": 2},
                        "trailing_stop_atr_mult": {"min": 2.0, "max": 2.5, "step": 0.5},
                        "profit_target_atr_mult": {"min": 2.5, "max": 3.0, "step": 0.5},
                        "time_limit_days": {"min": 10, "max": 15, "step": 5},
                    },
                }
            },
        }

        resolved, sectors = service._resolve_sweep_mode(config, "tech_leader_pullback_v1")
        grid = service._build_parameter_grid(resolved)

        self.assertEqual(sectors, ["Information Technology"])
        self.assertEqual(len(grid), 1536)
        self.assertEqual({entry["indicators"]["sma_50_dist_max"] for entry in grid}, {0.02, 0.04, 0.06})
        self.assertEqual({entry["indicators"]["relative_strength_index_vs_qqq_min"] for entry in grid}, {85.0})
        self.assertEqual({entry["indicators"]["relative_strength_index_vs_xlk_min"] for entry in grid}, {85.0})
        self.assertEqual({entry["exit_rules"]["time_limit_days"] for entry in grid}, {10, 15})
        self.assertEqual({entry["exit_rules"]["profit_target_atr_mult"] for entry in grid}, {2.5, 3.0})

    def test_resolve_tech_leader_pullback_v2_sweep_mode(self) -> None:
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
                "tech_leader_pullback_v2": {
                    "sector_whitelist": ["Information Technology"],
                    "replace_base_grid": True,
                    "grid_overrides": {
                        "sma_50_dist_min": {"min": 0.0, "max": 0.0, "step": 0.1},
                        "sma_50_dist_max": {"min": 0.02, "max": 0.04, "step": 0.02},
                        "sma_200_dist_min": {"min": 0.14, "max": 0.18, "step": 0.04},
                        "ma_alignment_50_200_min": {"min": 0.0, "max": 0.0, "step": 0.1},
                        "ma_slope_50_20_min": {"min": 0.0, "max": 0.0, "step": 0.1},
                        "ma_slope_200_20_min": {"min": 0.0, "max": 0.0, "step": 0.1},
                        "relative_strength_index_vs_spy_min": {"min": 85, "max": 85, "step": 5},
                        "relative_strength_index_vs_qqq_min": {"min": 85, "max": 85, "step": 5},
                        "relative_strength_index_vs_xlk_min": {"min": 85, "max": 85, "step": 5},
                        "roc_63_min": {"min": 0.10, "max": 0.15, "step": 0.05},
                        "roc_126_min": {"min": 0.20, "max": 0.20, "step": 0.10},
                        "rsi_14_min": {"min": 50, "max": 50, "step": 5},
                        "rsi_14_max": {"min": 55, "max": 55, "step": 5},
                        "vol_alpha_min": {"min": 1.2, "max": 1.2, "step": 0.2},
                        "signal_score_min": {"min": 34, "max": 36, "step": 2},
                        "trailing_stop_atr_mult": {"min": 2.0, "max": 2.5, "step": 0.5},
                        "profit_target_atr_mult": {"min": 2.5, "max": 3.0, "step": 0.5},
                        "time_limit_days": {"min": 5, "max": 10, "step": 5},
                    },
                }
            },
        }

        resolved, sectors = service._resolve_sweep_mode(config, "tech_leader_pullback_v2")
        grid = service._build_parameter_grid(resolved)

        self.assertEqual(sectors, ["Information Technology"])
        self.assertEqual(len(grid), 128)
        self.assertEqual({entry["indicators"]["sma_50_dist_max"] for entry in grid}, {0.02, 0.04})
        self.assertEqual({entry["indicators"]["sma_200_dist_min"] for entry in grid}, {0.14, 0.18})
        self.assertEqual({entry["indicators"]["relative_strength_index_vs_qqq_min"] for entry in grid}, {85.0})
        self.assertEqual({entry["indicators"]["relative_strength_index_vs_xlk_min"] for entry in grid}, {85.0})
        self.assertEqual({entry["indicators"]["rsi_14_min"] for entry in grid}, {50.0})
        self.assertEqual({entry["indicators"]["rsi_14_max"] for entry in grid}, {55.0})
        self.assertEqual({entry["indicators"]["vol_alpha_min"] for entry in grid}, {1.2})
        self.assertEqual({entry["indicators"]["signal_score_min"] for entry in grid}, {34.0, 36.0})
        self.assertEqual({entry["exit_rules"]["time_limit_days"] for entry in grid}, {5, 10})
        self.assertEqual({entry["exit_rules"]["profit_target_atr_mult"] for entry in grid}, {2.5, 3.0})

    def test_resolve_tech_leader_pullback_v3_sweep_mode(self) -> None:
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
                "tech_leader_pullback_v3": {
                    "sector_whitelist": ["Information Technology"],
                    "replace_base_grid": True,
                    "grid_overrides": {
                        "sma_50_dist_min": {"min": 0.02, "max": 0.02, "step": 0.01},
                        "sma_200_dist_min": {"min": 0.18, "max": 0.24, "step": 0.06},
                        "ma_alignment_50_200_min": {"min": 0.0, "max": 0.0, "step": 0.1},
                        "ma_slope_50_20_min": {"min": 0.0, "max": 0.0, "step": 0.1},
                        "ma_slope_200_20_min": {"min": 0.0, "max": 0.0, "step": 0.1},
                        "breakout_above_20d_high_min": {"min": 1.0, "max": 1.0, "step": 1.0},
                        "distance_above_20d_high_max": {"min": 0.005, "max": 0.010, "step": 0.005},
                        "relative_strength_index_vs_spy_min": {"min": 85, "max": 85, "step": 5},
                        "relative_strength_index_vs_qqq_min": {"min": 85, "max": 85, "step": 5},
                        "relative_strength_index_vs_xlk_min": {"min": 85, "max": 85, "step": 5},
                        "roc_63_min": {"min": 0.10, "max": 0.15, "step": 0.05},
                        "roc_126_min": {"min": 0.20, "max": 0.20, "step": 0.10},
                        "base_range_pct_20_max": {"min": 0.04, "max": 0.06, "step": 0.02},
                        "base_atr_contraction_20_max": {"min": 0.75, "max": 0.85, "step": 0.10},
                        "base_volume_dryup_ratio_20_max": {"min": 0.75, "max": 0.75, "step": 0.10},
                        "breakout_volume_ratio_50_min": {"min": 1.2, "max": 1.6, "step": 0.4},
                        "signal_score_min": {"min": 34, "max": 36, "step": 2},
                        "trailing_stop_atr_mult": {"min": 1.5, "max": 2.0, "step": 0.5},
                        "profit_target_atr_mult": {"min": 2.5, "max": 3.0, "step": 0.5},
                        "time_limit_days": {"min": 5, "max": 10, "step": 5},
                    },
                }
            },
        }

        resolved, sectors = service._resolve_sweep_mode(config, "tech_leader_pullback_v3")
        grid = service._build_parameter_grid(resolved)

        self.assertEqual(sectors, ["Information Technology"])
        self.assertEqual(len(grid), 1024)
        self.assertEqual({entry["indicators"]["breakout_above_20d_high_min"] for entry in grid}, {1.0})
        self.assertEqual({entry["indicators"]["distance_above_20d_high_max"] for entry in grid}, {0.005, 0.01})
        self.assertEqual({entry["indicators"]["relative_strength_index_vs_qqq_min"] for entry in grid}, {85.0})
        self.assertEqual({entry["indicators"]["relative_strength_index_vs_xlk_min"] for entry in grid}, {85.0})
        self.assertEqual({entry["indicators"]["base_range_pct_20_max"] for entry in grid}, {0.04, 0.06})
        self.assertEqual({entry["indicators"]["base_atr_contraction_20_max"] for entry in grid}, {0.75, 0.85})
        self.assertEqual({entry["indicators"]["breakout_volume_ratio_50_min"] for entry in grid}, {1.2, 1.6})
        self.assertNotIn("vol_alpha_min", grid[0]["indicators"])
        self.assertNotIn("rsi_14_min", grid[0]["indicators"])
        self.assertNotIn("rsi_14_max", grid[0]["indicators"])
        self.assertEqual({entry["exit_rules"]["time_limit_days"] for entry in grid}, {5, 10})
        self.assertEqual({entry["exit_rules"]["profit_target_atr_mult"] for entry in grid}, {2.5, 3.0})

    def test_resolve_tech_leader_pullback_v4_sweep_mode(self) -> None:
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
                "tech_leader_pullback_v4": {
                    "sector_whitelist": ["Information Technology"],
                    "replace_base_grid": True,
                    "grid_overrides": {
                        "sma_50_dist_min": {"min": 0.02, "max": 0.02, "step": 0.01},
                        "sma_200_dist_min": {"min": 0.18, "max": 0.24, "step": 0.06},
                        "ma_alignment_50_200_min": {"min": 0.0, "max": 0.0, "step": 0.1},
                        "ma_slope_50_20_min": {"min": 0.0, "max": 0.0, "step": 0.1},
                        "ma_slope_200_20_min": {"min": 0.0, "max": 0.0, "step": 0.1},
                        "breakout_above_20d_high_min": {"min": 1.0, "max": 1.0, "step": 1.0},
                        "distance_above_20d_high_max": {"min": 0.005, "max": 0.005, "step": 0.005},
                        "relative_strength_index_vs_spy_min": {"min": 85, "max": 85, "step": 5},
                        "relative_strength_index_vs_qqq_min": {"min": 85, "max": 85, "step": 5},
                        "relative_strength_index_vs_xlk_min": {"min": 85, "max": 85, "step": 5},
                        "roc_63_min": {"min": 0.10, "max": 0.15, "step": 0.05},
                        "roc_126_min": {"min": 0.20, "max": 0.20, "step": 0.10},
                        "base_atr_contraction_20_max": {"min": 0.75, "max": 0.75, "step": 0.10},
                        "base_volume_dryup_ratio_20_max": {"min": 0.75, "max": 0.75, "step": 0.10},
                        "breakout_volume_ratio_50_min": {"min": 1.8, "max": 2.0, "step": 0.2},
                        "signal_score_min": {"min": 34, "max": 36, "step": 2},
                        "trailing_stop_atr_mult": {"min": 1.5, "max": 2.0, "step": 0.5},
                        "profit_target_atr_mult": {"min": 2.5, "max": 3.0, "step": 0.5},
                        "time_limit_days": {"min": 10, "max": 10, "step": 5},
                    },
                }
            },
        }

        resolved, sectors = service._resolve_sweep_mode(config, "tech_leader_pullback_v4")
        grid = service._build_parameter_grid(resolved)

        self.assertEqual(sectors, ["Information Technology"])
        self.assertEqual(len(grid), 64)
        self.assertEqual({entry["indicators"]["breakout_above_20d_high_min"] for entry in grid}, {1.0})
        self.assertEqual({entry["indicators"]["distance_above_20d_high_max"] for entry in grid}, {0.005})
        self.assertEqual({entry["indicators"]["relative_strength_index_vs_qqq_min"] for entry in grid}, {85.0})
        self.assertEqual({entry["indicators"]["relative_strength_index_vs_xlk_min"] for entry in grid}, {85.0})
        self.assertEqual({entry["indicators"]["breakout_volume_ratio_50_min"] for entry in grid}, {1.8, 2.0})
        self.assertEqual({entry["indicators"]["base_atr_contraction_20_max"] for entry in grid}, {0.75})
        self.assertNotIn("base_range_pct_20_max", grid[0]["indicators"])
        self.assertNotIn("vol_alpha_min", grid[0]["indicators"])
        self.assertNotIn("rsi_14_min", grid[0]["indicators"])
        self.assertNotIn("rsi_14_max", grid[0]["indicators"])
        self.assertEqual({entry["exit_rules"]["time_limit_days"] for entry in grid}, {10})
        self.assertEqual({entry["exit_rules"]["profit_target_atr_mult"] for entry in grid}, {2.5, 3.0})

    def test_resolve_tech_leader_pullback_v5_sweep_mode(self) -> None:
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
                "tech_leader_pullback_v5": {
                    "sector_whitelist": ["Information Technology"],
                    "replace_base_grid": True,
                    "grid_overrides": {
                        "sma_50_dist_min": {"min": 0.02, "max": 0.02, "step": 0.01},
                        "sma_200_dist_min": {"min": 0.18, "max": 0.24, "step": 0.06},
                        "ma_alignment_50_200_min": {"min": 0.0, "max": 0.0, "step": 0.1},
                        "ma_slope_50_20_min": {"min": 0.0, "max": 0.0, "step": 0.1},
                        "ma_slope_200_20_min": {"min": 0.0, "max": 0.0, "step": 0.1},
                        "breakout_above_20d_high_min": {"min": 1.0, "max": 1.0, "step": 1.0},
                        "distance_above_20d_high_max": {"min": 0.01, "max": 0.01, "step": 0.01},
                        "relative_strength_index_vs_spy_min": {"min": 85, "max": 85, "step": 5},
                        "relative_strength_index_vs_qqq_min": {"min": 85, "max": 85, "step": 5},
                        "relative_strength_index_vs_xlk_min": {"min": 85, "max": 85, "step": 5},
                        "roc_63_min": {"min": 0.10, "max": 0.15, "step": 0.05},
                        "roc_126_min": {"min": 0.20, "max": 0.20, "step": 0.10},
                        "base_atr_contraction_20_max": {"min": 0.75, "max": 0.75, "step": 0.10},
                        "base_volume_dryup_ratio_20_max": {"min": 0.75, "max": 0.75, "step": 0.10},
                        "breakout_volume_ratio_50_min": {"min": 1.8, "max": 2.0, "step": 0.2},
                        "signal_score_min": {"min": 34, "max": 36, "step": 2},
                        "trailing_stop_atr_mult": {"min": 1.5, "max": 2.0, "step": 0.5},
                        "profit_target_atr_mult": {"min": 2.5, "max": 3.0, "step": 0.5},
                        "time_limit_days": {"min": 10, "max": 10, "step": 5},
                    },
                }
            },
        }

        resolved, sectors = service._resolve_sweep_mode(config, "tech_leader_pullback_v5")
        grid = service._build_parameter_grid(resolved)

        self.assertEqual(sectors, ["Information Technology"])
        self.assertEqual(len(grid), 64)
        self.assertEqual({entry["indicators"]["breakout_above_20d_high_min"] for entry in grid}, {1.0})
        self.assertEqual({entry["indicators"]["distance_above_20d_high_max"] for entry in grid}, {0.01})
        self.assertEqual({entry["indicators"]["relative_strength_index_vs_qqq_min"] for entry in grid}, {85.0})
        self.assertEqual({entry["indicators"]["relative_strength_index_vs_xlk_min"] for entry in grid}, {85.0})
        self.assertEqual({entry["indicators"]["breakout_volume_ratio_50_min"] for entry in grid}, {1.8, 2.0})
        self.assertEqual({entry["indicators"]["base_atr_contraction_20_max"] for entry in grid}, {0.75})
        self.assertNotIn("base_range_pct_20_max", grid[0]["indicators"])
        self.assertNotIn("vol_alpha_min", grid[0]["indicators"])
        self.assertNotIn("rsi_14_min", grid[0]["indicators"])
        self.assertNotIn("rsi_14_max", grid[0]["indicators"])
        self.assertEqual({entry["exit_rules"]["time_limit_days"] for entry in grid}, {10})
        self.assertEqual({entry["exit_rules"]["profit_target_atr_mult"] for entry in grid}, {2.5, 3.0})

    def test_resolve_tech_subindustry_reacceleration_v1_sweep_mode(self) -> None:
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
                "tech_subindustry_reacceleration_v1": {
                    "sector_whitelist": ["Information Technology"],
                    "replace_base_grid": True,
                    "grid_overrides": {
                        "sma_50_dist_min": {"min": 0.02, "max": 0.02, "step": 0.01},
                        "sma_200_dist_min": {"min": 0.18, "max": 0.24, "step": 0.06},
                        "ma_alignment_50_200_min": {"min": 0.0, "max": 0.0, "step": 0.1},
                        "ma_slope_50_20_min": {"min": 0.0, "max": 0.0, "step": 0.1},
                        "ma_slope_200_20_min": {"min": 0.0, "max": 0.0, "step": 0.1},
                        "breakout_above_20d_high_min": {"min": 1.0, "max": 1.0, "step": 1.0},
                        "distance_above_20d_high_max": {"min": 0.01, "max": 0.01, "step": 0.01},
                        "relative_strength_index_vs_spy_min": {"min": 85, "max": 85, "step": 5},
                        "relative_strength_index_vs_qqq_min": {"min": 85, "max": 85, "step": 5},
                        "relative_strength_index_vs_xlk_min": {"min": 85, "max": 85, "step": 5},
                        "relative_strength_index_vs_subindustry_min": {"min": 80, "max": 90, "step": 5},
                        "roc_63_min": {"min": 0.10, "max": 0.15, "step": 0.05},
                        "roc_126_min": {"min": 0.20, "max": 0.20, "step": 0.10},
                        "base_atr_contraction_20_max": {"min": 0.75, "max": 0.75, "step": 0.10},
                        "base_volume_dryup_ratio_20_max": {"min": 0.75, "max": 0.75, "step": 0.10},
                        "breakout_volume_ratio_50_min": {"min": 1.8, "max": 2.0, "step": 0.2},
                        "signal_score_min": {"min": 34, "max": 36, "step": 2},
                        "trailing_stop_atr_mult": {"min": 1.5, "max": 2.0, "step": 0.5},
                        "profit_target_atr_mult": {"min": 2.5, "max": 3.0, "step": 0.5},
                        "time_limit_days": {"min": 10, "max": 10, "step": 5},
                    },
                }
            },
        }

        resolved, sectors = service._resolve_sweep_mode(config, "tech_subindustry_reacceleration_v1")
        grid = service._build_parameter_grid(resolved)

        self.assertEqual(sectors, ["Information Technology"])
        self.assertEqual(len(grid), 192)
        self.assertEqual({entry["indicators"]["relative_strength_index_vs_subindustry_min"] for entry in grid}, {80.0, 85.0, 90.0})
        self.assertEqual({entry["indicators"]["relative_strength_index_vs_qqq_min"] for entry in grid}, {85.0})
        self.assertEqual({entry["indicators"]["relative_strength_index_vs_xlk_min"] for entry in grid}, {85.0})
        self.assertEqual({entry["exit_rules"]["time_limit_days"] for entry in grid}, {10})

    def test_resolve_tech_subindustry_reacceleration_v2_sweep_mode(self) -> None:
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
                "tech_subindustry_reacceleration_v2": {
                    "sector_whitelist": ["Information Technology"],
                    "replace_base_grid": True,
                    "grid_overrides": {
                        "sma_50_dist_min": {"min": 0.02, "max": 0.02, "step": 0.01},
                        "sma_200_dist_min": {"min": 0.18, "max": 0.24, "step": 0.06},
                        "ma_alignment_50_200_min": {"min": 0.0, "max": 0.0, "step": 0.1},
                        "ma_slope_50_20_min": {"min": 0.0, "max": 0.0, "step": 0.1},
                        "ma_slope_200_20_min": {"min": 0.0, "max": 0.0, "step": 0.1},
                        "breakout_above_20d_high_min": {"min": 1.0, "max": 1.0, "step": 1.0},
                        "distance_above_20d_high_max": {"min": 0.01, "max": 0.01, "step": 0.01},
                        "relative_strength_index_vs_spy_min": {"min": 85, "max": 85, "step": 5},
                        "relative_strength_index_vs_qqq_min": {"min": 80, "max": 85, "step": 5},
                        "relative_strength_index_vs_xlk_min": {"min": 80, "max": 85, "step": 5},
                        "relative_strength_index_vs_subindustry_min": {"min": 85, "max": 90, "step": 5},
                        "roc_63_min": {"min": 0.10, "max": 0.15, "step": 0.05},
                        "roc_126_min": {"min": 0.20, "max": 0.20, "step": 0.10},
                        "base_atr_contraction_20_max": {"min": 0.75, "max": 0.75, "step": 0.10},
                        "base_volume_dryup_ratio_20_max": {"min": 0.75, "max": 0.75, "step": 0.10},
                        "breakout_volume_ratio_50_min": {"min": 1.8, "max": 2.0, "step": 0.2},
                        "signal_score_min": {"min": 34, "max": 36, "step": 2},
                        "trailing_stop_atr_mult": {"min": 1.5, "max": 2.0, "step": 0.5},
                        "profit_target_atr_mult": {"min": 2.5, "max": 3.0, "step": 0.5},
                        "time_limit_days": {"min": 10, "max": 10, "step": 5},
                    },
                }
            },
        }

        resolved, sectors = service._resolve_sweep_mode(config, "tech_subindustry_reacceleration_v2")
        grid = service._build_parameter_grid(resolved)

        self.assertEqual(sectors, ["Information Technology"])
        self.assertEqual(len(grid), 512)
        self.assertEqual({entry["indicators"]["relative_strength_index_vs_qqq_min"] for entry in grid}, {80.0, 85.0})
        self.assertEqual({entry["indicators"]["relative_strength_index_vs_xlk_min"] for entry in grid}, {80.0, 85.0})
        self.assertEqual({entry["indicators"]["relative_strength_index_vs_subindustry_min"] for entry in grid}, {85.0, 90.0})
        self.assertEqual({entry["exit_rules"]["time_limit_days"] for entry in grid}, {10})

    def test_resolve_tech_subindustry_reacceleration_v3_sweep_mode(self) -> None:
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
                "tech_subindustry_reacceleration_v3": {
                    "sector_whitelist": ["Information Technology"],
                    "replace_base_grid": True,
                    "grid_overrides": {
                        "sma_50_dist_min": {"min": 0.02, "max": 0.02, "step": 0.01},
                        "sma_200_dist_min": {"min": 0.18, "max": 0.24, "step": 0.06},
                        "ma_alignment_50_200_min": {"min": 0.0, "max": 0.0, "step": 0.1},
                        "ma_slope_50_20_min": {"min": 0.0, "max": 0.0, "step": 0.1},
                        "ma_slope_200_20_min": {"min": 0.0, "max": 0.0, "step": 0.1},
                        "breakout_above_20d_high_min": {"min": 1.0, "max": 1.0, "step": 1.0},
                        "distance_above_20d_high_max": {"min": 0.005, "max": 0.01, "step": 0.005},
                        "relative_strength_index_vs_spy_min": {"min": 85, "max": 85, "step": 5},
                        "relative_strength_index_vs_qqq_min": {"min": 80, "max": 85, "step": 5},
                        "relative_strength_index_vs_xlk_min": {"min": 80, "max": 80, "step": 5},
                        "relative_strength_index_vs_subindustry_min": {"min": 75, "max": 85, "step": 5},
                        "roc_63_min": {"min": 0.10, "max": 0.15, "step": 0.05},
                        "roc_126_min": {"min": 0.20, "max": 0.30, "step": 0.10},
                        "base_atr_contraction_20_max": {"min": 0.75, "max": 0.75, "step": 0.10},
                        "base_volume_dryup_ratio_20_max": {"min": 0.75, "max": 0.75, "step": 0.10},
                        "breakout_volume_ratio_50_min": {"min": 1.6, "max": 1.8, "step": 0.2},
                        "signal_score_min": {"min": 34, "max": 36, "step": 2},
                        "trailing_stop_atr_mult": {"min": 1.5, "max": 2.0, "step": 0.5},
                        "profit_target_atr_mult": {"min": 2.5, "max": 3.0, "step": 0.5},
                        "time_limit_days": {"min": 10, "max": 10, "step": 5},
                    },
                }
            },
        }

        resolved, sectors = service._resolve_sweep_mode(config, "tech_subindustry_reacceleration_v3")
        grid = service._build_parameter_grid(resolved)

        self.assertEqual(sectors, ["Information Technology"])
        self.assertEqual(len(grid), 1536)
        self.assertEqual({entry["indicators"]["breakout_above_20d_high_min"] for entry in grid}, {1.0})
        self.assertEqual({entry["indicators"]["distance_above_20d_high_max"] for entry in grid}, {0.005, 0.01})
        self.assertEqual({entry["indicators"]["relative_strength_index_vs_qqq_min"] for entry in grid}, {80.0, 85.0})
        self.assertEqual({entry["indicators"]["relative_strength_index_vs_xlk_min"] for entry in grid}, {80.0})
        self.assertEqual({entry["indicators"]["relative_strength_index_vs_subindustry_min"] for entry in grid}, {75.0, 80.0, 85.0})
        self.assertEqual({entry["indicators"]["roc_126_min"] for entry in grid}, {0.2, 0.3})
        self.assertEqual({entry["indicators"]["breakout_volume_ratio_50_min"] for entry in grid}, {1.6, 1.8})
        self.assertEqual({entry["exit_rules"]["time_limit_days"] for entry in grid}, {10})

    def test_resolve_tech_subindustry_reacceleration_v4_sweep_mode(self) -> None:
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
                "tech_subindustry_reacceleration_v4": {
                    "sector_whitelist": ["Information Technology"],
                    "replace_base_grid": True,
                    "grid_overrides": {
                        "sma_50_dist_min": {"min": 0.02, "max": 0.02, "step": 0.01},
                        "sma_200_dist_min": {"min": 0.18, "max": 0.24, "step": 0.06},
                        "ma_alignment_50_200_min": {"min": 0.0, "max": 0.0, "step": 0.1},
                        "ma_slope_50_20_min": {"min": 0.0, "max": 0.0, "step": 0.1},
                        "ma_slope_200_20_min": {"min": 0.0, "max": 0.0, "step": 0.1},
                        "breakout_above_20d_high_min": {"min": 1.0, "max": 1.0, "step": 1.0},
                        "distance_above_20d_high_max": {"min": 0.005, "max": 0.01, "step": 0.005},
                        "relative_strength_index_vs_spy_min": {"min": 85, "max": 85, "step": 5},
                        "relative_strength_index_vs_qqq_min": {"min": 75, "max": 80, "step": 5},
                        "relative_strength_index_vs_xlk_min": {"min": 75, "max": 80, "step": 5},
                        "relative_strength_index_vs_subindustry_min": {"min": 85, "max": 85, "step": 5},
                        "roc_63_min": {"min": 0.10, "max": 0.15, "step": 0.05},
                        "roc_126_min": {"min": 0.20, "max": 0.30, "step": 0.10},
                        "base_atr_contraction_20_max": {"min": 0.75, "max": 0.75, "step": 0.10},
                        "base_volume_dryup_ratio_20_max": {"min": 0.75, "max": 0.75, "step": 0.10},
                        "breakout_volume_ratio_50_min": {"min": 1.6, "max": 1.8, "step": 0.2},
                        "signal_score_min": {"min": 34, "max": 36, "step": 2},
                        "trailing_stop_atr_mult": {"min": 1.5, "max": 2.0, "step": 0.5},
                        "profit_target_atr_mult": {"min": 2.5, "max": 3.0, "step": 0.5},
                        "time_limit_days": {"min": 10, "max": 10, "step": 5},
                    },
                }
            },
        }

        resolved, sectors = service._resolve_sweep_mode(config, "tech_subindustry_reacceleration_v4")
        grid = service._build_parameter_grid(resolved)

        self.assertEqual(sectors, ["Information Technology"])
        self.assertEqual(len(grid), 1024)
        self.assertEqual({entry["indicators"]["distance_above_20d_high_max"] for entry in grid}, {0.005, 0.01})
        self.assertEqual({entry["indicators"]["relative_strength_index_vs_qqq_min"] for entry in grid}, {75.0, 80.0})
        self.assertEqual({entry["indicators"]["relative_strength_index_vs_xlk_min"] for entry in grid}, {75.0, 80.0})
        self.assertEqual({entry["indicators"]["relative_strength_index_vs_subindustry_min"] for entry in grid}, {85.0})
        self.assertEqual({entry["indicators"]["roc_126_min"] for entry in grid}, {0.2, 0.3})
        self.assertEqual({entry["indicators"]["breakout_volume_ratio_50_min"] for entry in grid}, {1.6, 1.8})
        self.assertEqual({entry["exit_rules"]["time_limit_days"] for entry in grid}, {10})

    def test_resolve_tech_subindustry_reacceleration_v5_sweep_mode(self) -> None:
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
                "tech_subindustry_reacceleration_v5": {
                    "sector_whitelist": ["Information Technology"],
                    "replace_base_grid": True,
                    "grid_overrides": {
                        "sma_50_dist_min": {"min": 0.02, "max": 0.02, "step": 0.01},
                        "sma_200_dist_min": {"min": 0.18, "max": 0.24, "step": 0.06},
                        "ma_alignment_50_200_min": {"min": 0.0, "max": 0.0, "step": 0.1},
                        "ma_slope_50_20_min": {"min": 0.0, "max": 0.0, "step": 0.1},
                        "ma_slope_200_20_min": {"min": 0.0, "max": 0.0, "step": 0.1},
                        "breakout_above_20d_high_min": {"min": 1.0, "max": 1.0, "step": 1.0},
                        "distance_above_20d_high_max": {"min": 0.005, "max": 0.01, "step": 0.005},
                        "relative_strength_index_vs_spy_min": {"min": 75, "max": 75, "step": 5},
                        "relative_strength_index_vs_qqq_min": {"min": 75, "max": 80, "step": 5},
                        "relative_strength_index_vs_subindustry_min": {"min": 85, "max": 85, "step": 5},
                        "roc_63_min": {"min": 0.10, "max": 0.15, "step": 0.05},
                        "roc_126_min": {"min": 0.20, "max": 0.30, "step": 0.10},
                        "base_atr_contraction_20_max": {"min": 0.75, "max": 0.75, "step": 0.10},
                        "base_volume_dryup_ratio_20_max": {"min": 0.75, "max": 0.75, "step": 0.10},
                        "breakout_volume_ratio_50_min": {"min": 1.6, "max": 1.8, "step": 0.2},
                        "signal_score_min": {"min": 34, "max": 36, "step": 2},
                        "trailing_stop_atr_mult": {"min": 1.5, "max": 2.0, "step": 0.5},
                        "profit_target_atr_mult": {"min": 2.5, "max": 3.0, "step": 0.5},
                        "time_limit_days": {"min": 10, "max": 10, "step": 5},
                    },
                }
            },
        }

        resolved, sectors = service._resolve_sweep_mode(config, "tech_subindustry_reacceleration_v5")
        grid = service._build_parameter_grid(resolved)

        self.assertEqual(sectors, ["Information Technology"])
        self.assertEqual(len(grid), 512)
        self.assertEqual({entry["indicators"]["distance_above_20d_high_max"] for entry in grid}, {0.005, 0.01})
        self.assertEqual({entry["indicators"]["relative_strength_index_vs_spy_min"] for entry in grid}, {75.0})
        self.assertEqual({entry["indicators"]["relative_strength_index_vs_qqq_min"] for entry in grid}, {75.0, 80.0})
        self.assertNotIn("relative_strength_index_vs_xlk_min", grid[0]["indicators"])
        self.assertEqual({entry["indicators"]["relative_strength_index_vs_subindustry_min"] for entry in grid}, {85.0})
        self.assertEqual({entry["indicators"]["roc_126_min"] for entry in grid}, {0.2, 0.3})
        self.assertEqual({entry["indicators"]["breakout_volume_ratio_50_min"] for entry in grid}, {1.6, 1.8})
        self.assertEqual({entry["exit_rules"]["time_limit_days"] for entry in grid}, {10})

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
            benchmark_context=BenchmarkContext(spy_ticker="SPY", sector_ticker=None, price_maps={}),
        )
        net_metrics = service._run_backtest(
            validation_frame,
            indicators,
            exit_rules,
            backtest_costs=service._load_backtest_costs(
                {"backtest_costs": {"slippage_bps_per_side": 5, "commission_bps_per_side": 0}}
            ),
            benchmark_context=BenchmarkContext(spy_ticker="SPY", sector_ticker=None, price_maps={}),
        )

        self.assertEqual(gross_metrics["trade_count"], 1)
        self.assertLess(net_metrics["expectancy"], gross_metrics["expectancy"])
        self.assertAlmostEqual(gross_metrics["expectancy"] - net_metrics["expectancy"], 0.0010, places=6)
