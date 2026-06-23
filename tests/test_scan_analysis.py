from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from src.scan.analysis_service import ScanAnalysisService
from src.scan.ranker import CandidateRanker
from src.settings import AppPaths


class ScanAnalysisServiceTests(unittest.TestCase):
    def test_candidate_ranker_builds_purged_walk_forward_folds(self) -> None:
        scan_dates = pd.bdate_range("2026-01-05", periods=30)
        frame = pd.DataFrame(
            [
                {
                    "scan_date": scan_day.strftime("%Y-%m-%d"),
                    "ticker": f"T{index}",
                    "strategy_slot": "energy",
                    "strategy_sector": "Energy",
                    "sector": "Energy",
                    "signal_score": 30.0,
                    "setup_quality_score": 0.70,
                    "expected_alpha_score": 0.60,
                    "breadth_score": 0.50,
                    "freshness_score": 0.65,
                    "overlap_penalty": 0.0,
                    "opportunity_score": 0.62,
                    "selected": 1,
                    "alpha_vs_sector_10d": 0.01,
                    "details_json": '{"already_owned": false}',
                }
                for index, scan_day in enumerate(scan_dates)
            ]
        )

        ranker = CandidateRanker(target_column="alpha_vs_sector_10d")
        folds = ranker._purged_walk_forward_folds(
            frame,
            train_ratio=0.7,
            embargo_days=10,
            max_validation_blocks=5,
        )

        self.assertTrue(folds)
        first_fold = folds[0]
        first_train_dates = sorted(first_fold["train"]["scan_date"].astype(str).unique().tolist())
        first_validation_dates = sorted(first_fold["validation"]["scan_date"].astype(str).unique().tolist())
        self.assertEqual(first_train_dates[-1], scan_dates[10].strftime("%Y-%m-%d"))
        self.assertEqual(first_validation_dates[0], scan_dates[21].strftime("%Y-%m-%d"))
        self.assertEqual(len(first_train_dates), 11)
        self.assertEqual(len(first_validation_dates), 2)

    def test_scan_analysis_writes_forward_attribution_report(self) -> None:
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
            paths.reports_dir.mkdir(parents=True, exist_ok=True)
            paths.logs_dir.mkdir(parents=True, exist_ok=True)

            scan_frame = pd.DataFrame(
                [
                    {
                        "scan_date": "2026-05-01",
                        "ticker": "AAA",
                        "strategy_slot": "materials",
                        "strategy_sector": "Materials",
                        "sector": "Materials",
                        "signal_score": 0.0,
                        "setup_quality_score": 0.8,
                        "expected_alpha_score": 0.7,
                        "breadth_score": 0.6,
                        "freshness_score": 0.8,
                        "overlap_penalty": 0.0,
                        "opportunity_score": 0.75,
                        "selected": 1,
                        "selected_rank": 1,
                        "shares": 100,
                        "details_json": '{"already_owned": false, "ranking_components": {"selection_score": 0.81, "selection_source": "shortlist_model", "model_name": "xgboost_model", "model_predicted_alpha": 0.255, "model_rank": 3, "model_reason_summary": "limited recent downside gap risk, supportive earnings gap", "model_comparison_summary": "BBB in Materials on supportive earnings gap", "recent_feedback_adjustment": 0.03}, "recent_selection_memory": {"drag_picks": 0, "drag_mean_target": null, "missed_winner_count": 2, "missed_winner_mean_gap": 0.12}}',
                    },
                    {
                        "scan_date": "2026-05-01",
                        "ticker": "BBB",
                        "strategy_slot": "materials",
                        "strategy_sector": "Materials",
                        "sector": "Materials",
                        "signal_score": 30.0,
                        "setup_quality_score": 0.6,
                        "expected_alpha_score": 0.5,
                        "breadth_score": 0.6,
                        "freshness_score": 0.5,
                        "overlap_penalty": 0.0,
                        "opportunity_score": 0.55,
                        "selected": 0,
                        "selected_rank": None,
                        "shares": 100,
                        "details_json": '{"already_owned": false, "ranking_components": {"selection_score": 0.50, "recent_feedback_adjustment": -0.02}, "recent_selection_memory": {"drag_picks": 3, "drag_mean_target": -0.04, "missed_winner_count": 0, "missed_winner_mean_gap": null}}',
                    },
                ]
            )
            price_history = pd.DataFrame(
                [
                    {
                        "ticker": ticker,
                        "date": day.date(),
                        "open": 100.0 + index,
                        "high": 101.0 + index,
                        "low": 99.0 + index,
                        "close": 100.5 + index,
                        "volume": 1_000_000,
                        "adj_close": base + index,
                    }
                    for ticker, base in (("AAA", 100.0), ("BBB", 100.0))
                    for index, day in enumerate(pd.bdate_range("2026-05-01", periods=25))
                ]
            )

            class FakeDB:
                def __init__(self, paths, scan_frame, price_history):
                    self.paths = paths
                    self._scan_frame = scan_frame
                    self._price_history = price_history

                def initialize(self): return None
                def load_scan_candidates(self, scan_date=None):
                    if scan_date is None:
                        return self._scan_frame.copy()
                    return self._scan_frame[self._scan_frame["scan_date"] == scan_date].copy()
                def load_price_history(self, tickers):
                    return self._price_history[self._price_history["ticker"].isin(tickers)].copy()

            service = ScanAnalysisService(FakeDB(paths, scan_frame, price_history))
            with patch("src.scan.analysis_service.load_feature_config", return_value={"scan_policy": {"min_opportunity_score": 0.55}}), \
                 patch(
                     "src.scan.analysis_service.load_active_strategies",
                     return_value={
                         "materials": type("Strategy", (), {"sector": "Materials"})(),
                         "industrials": type("Strategy", (), {"sector": "Industrials"})(),
                     },
                 ):
                report = service.run(scan_date="2026-05-01", horizons=(5, 10))

            self.assertEqual(report.scan_date, "2026-05-01")
            report_text = (paths.reports_dir / "scan_analysis.md").read_text(encoding="utf-8")
            self.assertIn("# Scan Analysis", report_text)
            self.assertIn("## Selection Summary", report_text)
            self.assertIn("## Active Slot Coverage", report_text)
            self.assertIn("### industrials", report_text)
            self.assertIn("- status: active slot, but no candidates passed the hard gate and signal threshold", report_text)
            self.assertIn("## Slot Allocation Drivers", report_text)
            self.assertIn("- status: no eligible candidates; this slot dropped out before final ranking", report_text)
            self.assertIn("## Forward Attribution", report_text)
            self.assertIn("### 5-Day Forward Return", report_text)
            self.assertIn("### AAA", report_text)
            self.assertIn("### BBB", report_text)
            self.assertIn("- selection_source: shortlist_model", report_text)
            self.assertIn("- model_name: xgboost_model", report_text)
            self.assertIn("- model_predicted_alpha: 0.2550", report_text)
            self.assertIn("- model_rank: 3", report_text)
            self.assertIn("- model_reason_summary: limited recent downside gap risk, supportive earnings gap", report_text)
            self.assertIn("- model_comparison_summary: BBB in Materials on supportive earnings gap", report_text)
            self.assertIn("- selection_score: 0.8100", report_text)
            self.assertIn("- signal_score: n/a (model-sourced)", report_text)
            self.assertIn("- raw_opportunity_rank: 1", report_text)
            self.assertIn("- adjusted_selection_rank: 1", report_text)
            self.assertIn("- recent_feedback_adjustment: +0.0300", report_text)
            self.assertIn("- recent_missed_winner_count: 2", report_text)
            self.assertIn("- recent_drag_picks: 3", report_text)

    def test_scan_analysis_can_refresh_without_email(self) -> None:
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
            paths.reports_dir.mkdir(parents=True, exist_ok=True)
            paths.logs_dir.mkdir(parents=True, exist_ok=True)

            scan_frame = pd.DataFrame(
                [
                    {
                        "scan_date": "2026-05-11",
                        "ticker": "AAA",
                        "strategy_slot": "materials",
                        "strategy_sector": "Materials",
                        "sector": "Materials",
                        "signal_score": 36.0,
                        "setup_quality_score": 0.8,
                        "expected_alpha_score": 0.7,
                        "breadth_score": 0.6,
                        "freshness_score": 0.8,
                        "overlap_penalty": 0.0,
                        "opportunity_score": 0.75,
                        "selected": 1,
                        "shares": 100,
                        "details_json": '{"already_owned": false}',
                    }
                ]
            )

            class FakeDB:
                def __init__(self, paths, scan_frame):
                    self.paths = paths
                    self._scan_frame = scan_frame

                def initialize(self): return None
                def load_scan_candidates(self, scan_date=None):
                    return self._scan_frame.copy()
                def load_price_history(self, tickers):
                    return pd.DataFrame()

            service = ScanAnalysisService(FakeDB(paths, scan_frame))
            with patch("src.scan.analysis_service.load_feature_config", return_value={"scan_policy": {"min_opportunity_score": 0.55}}), \
                 patch("src.scan.analysis_service.ScanService.run", return_value=None) as run_scan:
                report = service.run(refresh=True, horizons=(5,))

            self.assertTrue(report.refreshed)
            run_scan.assert_called_once_with(dry_run=True)

    def test_scan_analysis_highlights_owned_strength_watchlist(self) -> None:
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
            paths.reports_dir.mkdir(parents=True, exist_ok=True)
            paths.logs_dir.mkdir(parents=True, exist_ok=True)

            scan_frame = pd.DataFrame(
                [
                    {
                        "scan_date": "2026-05-11",
                        "ticker": "AAA",
                        "strategy_slot": "energy",
                        "strategy_sector": "Energy",
                        "sector": "Energy",
                        "signal_score": 38.0,
                        "setup_quality_score": 0.82,
                        "expected_alpha_score": 0.74,
                        "breadth_score": 0.70,
                        "freshness_score": 0.86,
                        "overlap_penalty": 1.0,
                        "opportunity_score": -0.20,
                        "selected": 0,
                        "shares": 100,
                        "details_json": '{"already_owned": true, "pre_penalty_opportunity_score": 0.80, "overlap_components": {"same_ticker": 1.0, "same_slot": 0.08, "same_sector": 0.0, "same_regime": 0.0}}',
                    }
                ]
            )

            class FakeDB:
                def __init__(self, paths, scan_frame):
                    self.paths = paths
                    self._scan_frame = scan_frame

                def initialize(self): return None
                def load_scan_candidates(self, scan_date=None):
                    return self._scan_frame.copy()
                def load_price_history(self, tickers):
                    return pd.DataFrame()

            service = ScanAnalysisService(FakeDB(paths, scan_frame))
            with patch("src.scan.analysis_service.load_feature_config", return_value={"scan_policy": {"min_opportunity_score": 0.55}}):
                service.run(scan_date="2026-05-11", horizons=(5,))

            report_text = (paths.reports_dir / "scan_analysis.md").read_text(encoding="utf-8")
            self.assertIn("## Owned Strength Watchlist", report_text)
            self.assertIn("already owned, setup still valid", report_text)
            self.assertIn("overlap_components: same_ticker=1.00, same_slot=0.08", report_text)

    def test_scan_analysis_renders_portfolio_strength_coverage(self) -> None:
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
            paths.reports_dir.mkdir(parents=True, exist_ok=True)
            paths.logs_dir.mkdir(parents=True, exist_ok=True)

            scan_frame = pd.DataFrame(
                [
                    {
                        "scan_date": "2026-05-11",
                        "ticker": "AAA",
                        "strategy_slot": "energy",
                        "strategy_sector": "Energy",
                        "sector": "Energy",
                        "signal_score": 0.0,
                        "setup_quality_score": 0.0,
                        "expected_alpha_score": 0.8,
                        "breadth_score": 0.7,
                        "freshness_score": 0.6,
                        "overlap_penalty": 1.0,
                        "opportunity_score": -0.20,
                        "selection_score": 0.70,
                        "selected": 0,
                        "shares": 100,
                        "details_json": '{"already_owned": true, "pre_penalty_opportunity_score": 0.80}',
                    },
                    {
                        "scan_date": "2026-05-11",
                        "ticker": "BBB",
                        "strategy_slot": "energy",
                        "strategy_sector": "Energy",
                        "sector": "Energy",
                        "signal_score": 0.0,
                        "setup_quality_score": 0.0,
                        "expected_alpha_score": 0.9,
                        "breadth_score": 0.7,
                        "freshness_score": 0.6,
                        "overlap_penalty": 0.0,
                        "opportunity_score": 0.85,
                        "selection_score": 0.90,
                        "selected": 1,
                        "shares": 100,
                        "details_json": '{"already_owned": false, "pre_penalty_opportunity_score": 0.85}',
                    },
                    {
                        "scan_date": "2026-05-11",
                        "ticker": "CCC",
                        "strategy_slot": "materials",
                        "strategy_sector": "Materials",
                        "sector": "Materials",
                        "signal_score": 0.0,
                        "setup_quality_score": 0.0,
                        "expected_alpha_score": 0.6,
                        "breadth_score": 0.5,
                        "freshness_score": 0.6,
                        "overlap_penalty": 0.0,
                        "opportunity_score": 0.50,
                        "selection_score": 0.50,
                        "selected": 0,
                        "shares": 100,
                        "details_json": '{"already_owned": false, "pre_penalty_opportunity_score": 0.50}',
                    },
                ]
            )

            class FakeDB:
                def __init__(self, paths, scan_frame):
                    self.paths = paths
                    self._scan_frame = scan_frame

                def initialize(self): return None
                def load_scan_candidates(self, scan_date=None):
                    return self._scan_frame.copy()
                def load_price_history(self, tickers):
                    return pd.DataFrame()
                def list_open_trades(self):
                    return [{"ticker": "AAA"}, {"ticker": "ZZZ"}]

            service = ScanAnalysisService(FakeDB(paths, scan_frame))
            with patch("src.scan.analysis_service.load_feature_config", return_value={"scan_policy": {"min_opportunity_score": 0.55}}):
                service.run(scan_date="2026-05-11", horizons=(5,))

            report_text = (paths.reports_dir / "scan_analysis.md").read_text(encoding="utf-8")
            self.assertIn("## Portfolio Strength Coverage", report_text)
            self.assertIn("- top_6_already_held: 1/6", report_text)
            self.assertIn("- held_candidates_in_scan: 1", report_text)
            self.assertIn("- strongest_held_candidate: AAA", report_text)
            self.assertIn("- strongest_unheld_candidate: BBB", report_text)
            self.assertIn("- open_holdings_not_in_candidate_set: ZZZ", report_text)
            self.assertIn("1. BBB - not held, selected, score=0.8500", report_text)
            self.assertIn("2. AAA - held, not selected, score=0.8000", report_text)

    def test_scan_analysis_adds_learned_buy_review_from_prior_scan_history(self) -> None:
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
            paths.reports_dir.mkdir(parents=True, exist_ok=True)
            paths.logs_dir.mkdir(parents=True, exist_ok=True)

            rows: list[dict[str, object]] = []
            history_dates = pd.bdate_range("2026-03-02", periods=35)
            for date_index, scan_day in enumerate(history_dates):
                for candidate_index in range(5):
                    expected_alpha = 0.20 + (candidate_index * 0.15)
                    signal_score = 25.0 + candidate_index
                    target = (expected_alpha * 0.04) + (candidate_index * 0.002)
                    rows.append(
                        {
                            "scan_date": scan_day.strftime("%Y-%m-%d"),
                            "ticker": f"H{date_index}{candidate_index}",
                            "strategy_slot": "energy",
                            "strategy_sector": "Energy",
                            "sector": "Energy",
                            "signal_score": signal_score,
                            "setup_quality_score": 0.60 + (candidate_index * 0.02),
                            "expected_alpha_score": expected_alpha,
                            "breadth_score": 0.55,
                            "freshness_score": 0.70,
                            "overlap_penalty": 0.0,
                            "opportunity_score": 0.55 + (candidate_index * 0.05),
                            "selected": 1 if candidate_index < 2 else 0,
                            "selected_rank": candidate_index + 1 if candidate_index < 2 else None,
                            "shares": 100,
                            "alpha_vs_sector_10d": target,
                            "details_json": '{"already_owned": false, "feature_snapshot": {"atr_14": 2.0}}',
                        }
                    )

            rows.extend(
                [
                    {
                        "scan_date": "2026-05-11",
                        "ticker": "AAA",
                        "strategy_slot": "energy",
                        "strategy_sector": "Energy",
                        "sector": "Energy",
                        "signal_score": 31.0,
                        "setup_quality_score": 0.68,
                        "expected_alpha_score": 0.30,
                        "breadth_score": 0.55,
                        "freshness_score": 0.72,
                        "overlap_penalty": 0.0,
                        "opportunity_score": 0.70,
                        "selected": 1,
                        "selected_rank": 1,
                        "shares": 100,
                        "alpha_vs_sector_10d": None,
                        "details_json": '{"already_owned": false, "feature_snapshot": {"atr_14": 2.0}}',
                    },
                    {
                        "scan_date": "2026-05-11",
                        "ticker": "BBB",
                        "strategy_slot": "energy",
                        "strategy_sector": "Energy",
                        "sector": "Energy",
                        "signal_score": 38.0,
                        "setup_quality_score": 0.82,
                        "expected_alpha_score": 0.90,
                        "breadth_score": 0.55,
                        "freshness_score": 0.88,
                        "overlap_penalty": 0.0,
                        "opportunity_score": 0.69,
                        "selected": 0,
                        "selected_rank": None,
                        "shares": 100,
                        "alpha_vs_sector_10d": None,
                        "details_json": '{"already_owned": false, "feature_snapshot": {"atr_14": 2.0}}',
                    },
                    {
                        "scan_date": "2026-05-11",
                        "ticker": "CCC",
                        "strategy_slot": "energy",
                        "strategy_sector": "Energy",
                        "sector": "Energy",
                        "signal_score": 29.0,
                        "setup_quality_score": 0.66,
                        "expected_alpha_score": 0.20,
                        "breadth_score": 0.55,
                        "freshness_score": 0.70,
                        "overlap_penalty": 0.0,
                        "opportunity_score": 0.68,
                        "selected": 1,
                        "selected_rank": 2,
                        "shares": 100,
                        "alpha_vs_sector_10d": None,
                        "details_json": '{"already_owned": false, "feature_snapshot": {"atr_14": 2.0}}',
                    },
                ]
            )
            scan_frame = pd.DataFrame(rows)

            class FakeDB:
                def __init__(self, paths, scan_frame):
                    self.paths = paths
                    self._scan_frame = scan_frame

                def initialize(self): return None
                def load_scan_candidates(self, scan_date=None):
                    if scan_date is None:
                        return self._scan_frame.copy()
                    return self._scan_frame[self._scan_frame["scan_date"] == scan_date].copy()
                def load_price_history(self, tickers):
                    return pd.DataFrame()

            service = ScanAnalysisService(FakeDB(paths, scan_frame))
            with patch(
                "src.scan.analysis_service.load_feature_config",
                return_value={"scan_policy": {"min_opportunity_score": 0.55, "max_candidates_total": 2, "max_candidates_per_slot": 2, "max_candidates_per_sector": 2}},
            ):
                service.run(scan_date="2026-05-11", horizons=(10,))

            report_text = (paths.reports_dir / "scan_analysis.md").read_text(encoding="utf-8")
            self.assertIn("## Learned Buy Review", report_text)
            self.assertIn("## Slot-Level Selector Attribution", report_text)
            self.assertIn("### Ranker Quintile Tear Sheet", report_text)
            self.assertIn("- Q1: mean_target=", report_text)
            self.assertIn("- q1_minus_q5_spread:", report_text)
            self.assertIn("### Daily IC", report_text)
            self.assertIn("- ic_mean:", report_text)
            self.assertIn("### Slot Breakdown", report_text)
            self.assertIn("### Sector Breakdown", report_text)
            self.assertIn("- validation_method: purged_walk_forward", report_text)
            self.assertIn("- embargo_days: 10", report_text)
            self.assertIn("### Top Learned Buys", report_text)
            self.assertIn("### Best Learned Rejections", report_text)
            self.assertIn("#### BBB", report_text)
            self.assertIn("### energy", report_text)
            self.assertIn("- learned: mean_target=", report_text)
            self.assertIn("runtime_exclusion_reason: portfolio cap / overlap selection loss", report_text)
            self.assertIn("ranker_positive_reasons:", report_text)

    def test_scan_analysis_highlights_recent_selection_mistakes(self) -> None:
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
            paths.reports_dir.mkdir(parents=True, exist_ok=True)
            paths.logs_dir.mkdir(parents=True, exist_ok=True)

            scan_frame = pd.DataFrame(
                [
                    {
                        "scan_date": "2026-04-24",
                        "ticker": "WLK",
                        "strategy_slot": "materials",
                        "strategy_sector": "Materials",
                        "sector": "Materials",
                        "signal_score": 35.0,
                        "setup_quality_score": 0.72,
                        "expected_alpha_score": 0.60,
                        "breadth_score": 0.55,
                        "freshness_score": 0.68,
                        "overlap_penalty": 0.0,
                        "opportunity_score": 0.74,
                        "selected": 1,
                        "selected_rank": 1,
                        "shares": 100,
                        "alpha_vs_sector_10d": -0.14,
                        "details_json": '{"already_owned": false}',
                    },
                    {
                        "scan_date": "2026-04-24",
                        "ticker": "LYB",
                        "strategy_slot": "materials",
                        "strategy_sector": "Materials",
                        "sector": "Materials",
                        "signal_score": 33.0,
                        "setup_quality_score": 0.66,
                        "expected_alpha_score": 0.58,
                        "breadth_score": 0.55,
                        "freshness_score": 0.64,
                        "overlap_penalty": 0.0,
                        "opportunity_score": 0.62,
                        "selected": 0,
                        "selected_rank": None,
                        "shares": 100,
                        "alpha_vs_sector_10d": 0.03,
                        "details_json": '{"already_owned": false}',
                    },
                    {
                        "scan_date": "2026-04-23",
                        "ticker": "WLK",
                        "strategy_slot": "materials",
                        "strategy_sector": "Materials",
                        "sector": "Materials",
                        "signal_score": 34.0,
                        "setup_quality_score": 0.70,
                        "expected_alpha_score": 0.60,
                        "breadth_score": 0.55,
                        "freshness_score": 0.66,
                        "overlap_penalty": 0.0,
                        "opportunity_score": 0.73,
                        "selected": 1,
                        "selected_rank": 1,
                        "shares": 100,
                        "alpha_vs_sector_10d": -0.12,
                        "details_json": '{"already_owned": false}',
                    },
                    {
                        "scan_date": "2026-04-23",
                        "ticker": "DOW",
                        "strategy_slot": "materials",
                        "strategy_sector": "Materials",
                        "sector": "Materials",
                        "signal_score": 32.0,
                        "setup_quality_score": 0.64,
                        "expected_alpha_score": 0.57,
                        "breadth_score": 0.55,
                        "freshness_score": 0.62,
                        "overlap_penalty": 0.0,
                        "opportunity_score": 0.61,
                        "selected": 0,
                        "selected_rank": None,
                        "shares": 100,
                        "alpha_vs_sector_10d": 0.08,
                        "details_json": '{"already_owned": false}',
                    },
                    {
                        "scan_date": "2026-05-11",
                        "ticker": "AAA",
                        "strategy_slot": "materials",
                        "strategy_sector": "Materials",
                        "sector": "Materials",
                        "signal_score": 36.0,
                        "setup_quality_score": 0.80,
                        "expected_alpha_score": 0.70,
                        "breadth_score": 0.60,
                        "freshness_score": 0.80,
                        "overlap_penalty": 0.0,
                        "opportunity_score": 0.75,
                        "selected": 1,
                        "selected_rank": 1,
                        "shares": 100,
                        "alpha_vs_sector_10d": None,
                        "details_json": '{"already_owned": false}',
                    },
                ]
            )

            class FakeDB:
                def __init__(self, paths, scan_frame):
                    self.paths = paths
                    self._scan_frame = scan_frame

                def initialize(self): return None
                def load_scan_candidates(self, scan_date=None):
                    if scan_date is None:
                        return self._scan_frame.copy()
                    return self._scan_frame[self._scan_frame["scan_date"] == scan_date].copy()
                def load_price_history(self, tickers):
                    return pd.DataFrame()

            service = ScanAnalysisService(FakeDB(paths, scan_frame))
            with patch("src.scan.analysis_service.load_feature_config", return_value={"scan_policy": {"min_opportunity_score": 0.55}}):
                service.run(scan_date="2026-05-11", horizons=(10,))

            report_text = (paths.reports_dir / "scan_analysis.md").read_text(encoding="utf-8")
            self.assertIn("## Recent Selection Mistakes", report_text)
            self.assertIn("## Mediocre Setup Diagnostics", report_text)
            self.assertIn("### Low-signal selected ideas", report_text)
            self.assertIn("### Repeated Mediocre Drags", report_text)

    def test_scan_analysis_renders_selector_shadow_comparison(self) -> None:
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
            paths.reports_dir.mkdir(parents=True, exist_ok=True)
            paths.logs_dir.mkdir(parents=True, exist_ok=True)

            scan_frame = pd.DataFrame(
                [
                    {
                        "scan_date": "2026-04-28",
                        "ticker": "AAA",
                        "strategy_slot": "energy",
                        "strategy_sector": "Energy",
                        "sector": "Energy",
                        "signal_score": 35.0,
                        "setup_quality_score": 0.75,
                        "expected_alpha_score": 0.62,
                        "breadth_score": 0.55,
                        "freshness_score": 0.70,
                        "overlap_penalty": 0.0,
                        "opportunity_score": 0.72,
                        "selected": 1,
                        "selected_rank": 1,
                        "shares": 100,
                        "alpha_vs_sector_1d": 0.01,
                        "alpha_vs_sector_5d": None,
                        "alpha_vs_sector_10d": -0.05,
                        "details_json": '{"already_owned": false, "ranking_components": {"selection_score": 0.72, "recent_feedback_adjustment": 0.00}}',
                    },
                    {
                        "scan_date": "2026-04-28",
                        "ticker": "BBB",
                        "strategy_slot": "energy",
                        "strategy_sector": "Energy",
                        "sector": "Energy",
                        "signal_score": 33.0,
                        "setup_quality_score": 0.68,
                        "expected_alpha_score": 0.60,
                        "breadth_score": 0.55,
                        "freshness_score": 0.66,
                        "overlap_penalty": 0.0,
                        "opportunity_score": 0.68,
                        "selected": 0,
                        "selected_rank": None,
                        "shares": 100,
                        "alpha_vs_sector_1d": 0.18,
                        "alpha_vs_sector_5d": None,
                        "alpha_vs_sector_10d": 0.12,
                        "details_json": '{"already_owned": false, "ranking_components": {"selection_score": 0.76, "recent_feedback_adjustment": 0.08}}',
                    },
                ]
            )

            class FakeDB:
                def __init__(self, paths, scan_frame):
                    self.paths = paths
                    self._scan_frame = scan_frame

                def initialize(self): return None
                def load_scan_candidates(self, scan_date=None):
                    if scan_date is None:
                        return self._scan_frame.copy()
                    return self._scan_frame[self._scan_frame["scan_date"] == scan_date].copy()
                def load_price_history(self, tickers):
                    return pd.DataFrame()

            service = ScanAnalysisService(FakeDB(paths, scan_frame))
            with patch(
                "src.scan.analysis_service.load_feature_config",
                return_value={"scan_policy": {"min_opportunity_score": 0.55, "max_candidates_total": 1, "max_candidates_per_slot": 1, "max_candidates_per_sector": 1}},
            ):
                service.run(scan_date="2026-04-28", horizons=(10,))

            report_text = (paths.reports_dir / "scan_analysis.md").read_text(encoding="utf-8")
            self.assertIn("## Selector Bakeoff", report_text)
            self.assertIn("### Window 10", report_text)
            self.assertIn("### Window 20", report_text)
            self.assertIn("### Window 40", report_text)
            self.assertIn("- opportunity: mean_target=", report_text)
            self.assertIn("- signal: mean_target=", report_text)
            self.assertIn("- learned: mean_target=", report_text)
            self.assertIn("- random: mean_target=", report_text)
            self.assertIn("#### Biggest Runtime-vs-Opportunity Disagreements", report_text)
            self.assertIn("- runtime: AAA", report_text)
            self.assertIn("- opportunity: BBB", report_text)
            self.assertIn("## Signal-First Selector Early Read", report_text)
            self.assertIn("- runtime_tickers: AAA", report_text)
            self.assertIn("- opportunity_counterfactual: BBB", report_text)
            self.assertIn("- 1d: runtime_mean_target=0.010000 opportunity_counterfactual_mean_target=0.180000", report_text)
            self.assertIn("- 5d: pending", report_text)
            self.assertIn("## Selector Shadow Comparison", report_text)
            self.assertIn("- runtime: mean_target=", report_text)
            self.assertIn("- shadow_old: mean_target=", report_text)
            self.assertIn("- shadow_new: mean_target=", report_text)
            self.assertIn("### Biggest Old-vs-New Swaps", report_text)
            self.assertIn("- shadow_old: AAA", report_text)
            self.assertIn("- shadow_new: BBB", report_text)
            self.assertIn("#### energy", report_text)
            self.assertIn("#### 2026-04-28 energy", report_text)
            self.assertIn("- delta: 0.170000", report_text)

    def test_scan_analysis_renders_post_change_selector_maturity(self) -> None:
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
            paths.reports_dir.mkdir(parents=True, exist_ok=True)
            paths.logs_dir.mkdir(parents=True, exist_ok=True)

            scan_frame = pd.DataFrame(
                [
                    {
                        "scan_date": "2026-05-13",
                        "ticker": "AAA",
                        "strategy_slot": "energy",
                        "strategy_sector": "Energy",
                        "sector": "Energy",
                        "signal_score": 35.0,
                        "setup_quality_score": 0.75,
                        "expected_alpha_score": 0.62,
                        "breadth_score": 0.55,
                        "freshness_score": 0.70,
                        "overlap_penalty": 0.0,
                        "opportunity_score": 0.72,
                        "selected": 1,
                        "selected_rank": 1,
                        "shares": 100,
                        "alpha_vs_sector_1d": None,
                        "alpha_vs_sector_5d": None,
                        "alpha_vs_sector_10d": None,
                        "details_json": '{"already_owned": false, "ranking_components": {"selection_score": 0.77, "recent_feedback_adjustment": 0.05}}',
                    },
                    {
                        "scan_date": "2026-05-13",
                        "ticker": "BBB",
                        "strategy_slot": "energy",
                        "strategy_sector": "Energy",
                        "sector": "Energy",
                        "signal_score": 33.0,
                        "setup_quality_score": 0.68,
                        "expected_alpha_score": 0.60,
                        "breadth_score": 0.55,
                        "freshness_score": 0.66,
                        "overlap_penalty": 0.0,
                        "opportunity_score": 0.68,
                        "selected": 0,
                        "selected_rank": None,
                        "shares": 100,
                        "alpha_vs_sector_1d": None,
                        "alpha_vs_sector_5d": None,
                        "alpha_vs_sector_10d": None,
                        "details_json": '{"already_owned": false, "ranking_components": {"selection_score": 0.74, "recent_feedback_adjustment": 0.04}}',
                    },
                ]
            )

            class FakeDB:
                def __init__(self, paths, scan_frame):
                    self.paths = paths
                    self._scan_frame = scan_frame

                def initialize(self): return None
                def load_scan_candidates(self, scan_date=None):
                    if scan_date is None:
                        return self._scan_frame.copy()
                    return self._scan_frame[self._scan_frame["scan_date"] == scan_date].copy()
                def load_price_history(self, tickers):
                    return pd.DataFrame()

            service = ScanAnalysisService(FakeDB(paths, scan_frame))
            with patch(
                "src.scan.analysis_service.load_feature_config",
                return_value={"scan_policy": {"min_opportunity_score": 0.55, "max_candidates_total": 1, "max_candidates_per_slot": 1, "max_candidates_per_sector": 1}},
            ):
                service.run(scan_date="2026-05-13", horizons=(10,))

            report_text = (paths.reports_dir / "scan_analysis.md").read_text(encoding="utf-8")
            self.assertIn("## Post-Change Selector Maturity", report_text)
            self.assertIn("- post_change_dates: 2026-05-13", report_text)
            self.assertIn("- 1d: selected_rows_with_outcomes=0/1 dates_with_outcomes=0/1", report_text)
            self.assertIn("  status: pending", report_text)

    def test_scan_analysis_renders_regime_attribution(self) -> None:
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
            paths.reports_dir.mkdir(parents=True, exist_ok=True)
            paths.logs_dir.mkdir(parents=True, exist_ok=True)

            scan_frame = pd.DataFrame(
                [
                    {
                        "scan_date": "2026-05-05",
                        "ticker": "AAA",
                        "strategy_slot": "energy",
                        "strategy_sector": "Energy",
                        "sector": "Energy",
                        "signal_score": 36.0,
                        "setup_quality_score": 0.76,
                        "expected_alpha_score": 0.66,
                        "breadth_score": 0.60,
                        "freshness_score": 0.74,
                        "overlap_penalty": 0.0,
                        "opportunity_score": 0.76,
                        "selected": 1,
                        "selected_rank": 1,
                        "sector_pct_above_50": 0.80,
                        "sector_pct_above_200": 0.75,
                        "alpha_vs_sector_10d": 0.05,
                        "details_json": '{"already_owned": false}',
                    },
                    {
                        "scan_date": "2026-05-05",
                        "ticker": "BBB",
                        "strategy_slot": "energy",
                        "strategy_sector": "Energy",
                        "sector": "Energy",
                        "signal_score": 33.0,
                        "setup_quality_score": 0.70,
                        "expected_alpha_score": 0.61,
                        "breadth_score": 0.60,
                        "freshness_score": 0.69,
                        "overlap_penalty": 0.0,
                        "opportunity_score": 0.69,
                        "selected": 0,
                        "selected_rank": None,
                        "sector_pct_above_50": 0.80,
                        "sector_pct_above_200": 0.75,
                        "alpha_vs_sector_10d": 0.02,
                        "details_json": '{"already_owned": false}',
                    },
                    {
                        "scan_date": "2026-05-12",
                        "ticker": "CCC",
                        "strategy_slot": "energy",
                        "strategy_sector": "Energy",
                        "sector": "Energy",
                        "signal_score": 35.0,
                        "setup_quality_score": 0.74,
                        "expected_alpha_score": 0.64,
                        "breadth_score": 0.45,
                        "freshness_score": 0.68,
                        "overlap_penalty": 0.0,
                        "opportunity_score": 0.72,
                        "selected": 1,
                        "selected_rank": 1,
                        "sector_pct_above_50": 0.30,
                        "sector_pct_above_200": 0.25,
                        "alpha_vs_sector_10d": -0.03,
                        "details_json": '{"already_owned": false}',
                    },
                    {
                        "scan_date": "2026-05-12",
                        "ticker": "DDD",
                        "strategy_slot": "energy",
                        "strategy_sector": "Energy",
                        "sector": "Energy",
                        "signal_score": 31.0,
                        "setup_quality_score": 0.68,
                        "expected_alpha_score": 0.58,
                        "breadth_score": 0.45,
                        "freshness_score": 0.63,
                        "overlap_penalty": 0.0,
                        "opportunity_score": 0.66,
                        "selected": 0,
                        "selected_rank": None,
                        "sector_pct_above_50": 0.30,
                        "sector_pct_above_200": 0.25,
                        "alpha_vs_sector_10d": 0.01,
                        "details_json": '{"already_owned": false}',
                    },
                ]
            )

            market_dates = pd.bdate_range("2025-05-01", periods=280)
            price_rows: list[dict[str, object]] = []
            for ticker in ("SPY", "QQQ", "XLE"):
                for index, trade_date in enumerate(market_dates):
                    if trade_date <= pd.Timestamp("2026-05-05"):
                        adj_close = 100.0 + (index * 0.10)
                    else:
                        adj_close = 80.0 + (index * 0.01)
                    price_rows.append(
                        {
                            "ticker": ticker,
                            "date": trade_date.date(),
                            "open": adj_close,
                            "high": adj_close + 1.0,
                            "low": adj_close - 1.0,
                            "close": adj_close,
                            "volume": 1_000_000,
                            "adj_close": adj_close,
                        }
                    )
            price_history = pd.DataFrame(price_rows)

            class FakeDB:
                def __init__(self, paths, scan_frame, price_history):
                    self.paths = paths
                    self._scan_frame = scan_frame
                    self._price_history = price_history

                def initialize(self): return None
                def load_scan_candidates(self, scan_date=None):
                    if scan_date is None:
                        return self._scan_frame.copy()
                    return self._scan_frame[self._scan_frame["scan_date"] == scan_date].copy()
                def load_price_history(self, tickers):
                    return self._price_history[self._price_history["ticker"].isin(tickers)].copy()

            service = ScanAnalysisService(FakeDB(paths, scan_frame, price_history))
            with patch("src.scan.analysis_service.load_feature_config", return_value={"scan_policy": {"min_opportunity_score": 0.55}}):
                service.run(scan_date="2026-05-12", horizons=(10,))

            report_text = (paths.reports_dir / "scan_analysis.md").read_text(encoding="utf-8")
            self.assertIn("## Regime Attribution", report_text)
            self.assertIn("### SPY 200d Regime", report_text)
            self.assertIn("#### green", report_text)
            self.assertIn("#### red", report_text)
            self.assertIn("### Sector Breadth", report_text)
            self.assertIn("#### high", report_text)
            self.assertIn("#### low", report_text)
            self.assertIn("- energy: selected_mean_target=0.050000 excluded_mean_target=0.020000 selected_rows=1 excluded_rows=1", report_text)

    def test_scan_analysis_renders_slot_internal_attribution(self) -> None:
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
            paths.reports_dir.mkdir(parents=True, exist_ok=True)
            paths.logs_dir.mkdir(parents=True, exist_ok=True)

            rows: list[dict[str, object]] = []
            for index in range(24):
                distance = 0.01 + (index * 0.005)
                target = 0.08 - distance
                rows.append(
                    {
                        "scan_date": "2026-05-05" if index < 12 else "2026-05-06",
                        "ticker": f"E{index}",
                        "strategy_slot": "energy",
                        "strategy_sector": "Energy",
                        "sector": "Energy",
                        "signal_score": 28.0 + index,
                        "setup_quality_score": 0.60 + (index * 0.01),
                        "expected_alpha_score": 0.55 + (index * 0.005),
                        "breadth_score": 0.50,
                        "freshness_score": 0.70 - (index * 0.005),
                        "overlap_penalty": 0.0,
                        "opportunity_score": 0.60 + (index * 0.01),
                        "selected": 1 if index % 2 == 0 else 0,
                        "selected_rank": 1 if index % 2 == 0 else None,
                        "alpha_vs_sector_10d": target,
                        "details_json": json.dumps(
                            {
                                "already_owned": False,
                                "feature_snapshot": {
                                    "distance_above_20d_high": distance,
                                    "sector_pct_above_50": 0.70,
                                    "sector_pct_above_200": 0.65,
                                },
                            }
                        ),
                    }
                )
            scan_frame = pd.DataFrame(rows)

            class FakeDB:
                def __init__(self, paths, scan_frame):
                    self.paths = paths
                    self._scan_frame = scan_frame

                def initialize(self): return None
                def load_scan_candidates(self, scan_date=None):
                    if scan_date is None:
                        return self._scan_frame.copy()
                    return self._scan_frame[self._scan_frame["scan_date"] == scan_date].copy()
                def load_price_history(self, tickers):
                    return pd.DataFrame()

            service = ScanAnalysisService(FakeDB(paths, scan_frame))
            with patch("src.scan.analysis_service.load_feature_config", return_value={"scan_policy": {"min_opportunity_score": 0.55}}):
                service.run(scan_date="2026-05-06", horizons=(10,))

            report_text = (paths.reports_dir / "scan_analysis.md").read_text(encoding="utf-8")
            self.assertIn("## Slot Internal Attribution", report_text)
            self.assertIn("### energy", report_text)
            self.assertIn("#### Strongest Positive Discriminators", report_text)
            self.assertIn("breakout extension", report_text)

    def test_scan_analysis_renders_live_slot_gate_waterfall(self) -> None:
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
            paths.reports_dir.mkdir(parents=True, exist_ok=True)
            paths.logs_dir.mkdir(parents=True, exist_ok=True)

            scan_frame = pd.DataFrame(
                [
                    {
                        "scan_date": "2026-05-20",
                        "ticker": "AAA",
                        "strategy_slot": "energy",
                        "strategy_sector": "Energy",
                        "sector": "Energy",
                        "signal_score": 36.0,
                        "setup_quality_score": 0.8,
                        "expected_alpha_score": 0.7,
                        "breadth_score": 0.6,
                        "freshness_score": 0.8,
                        "overlap_penalty": 0.0,
                        "opportunity_score": 0.75,
                        "selected": 1,
                        "details_json": '{"already_owned": false, "ranking_components": {"selection_score": 36.0}}',
                    }
                ]
            )
            snapshot = pd.DataFrame(
                [
                    {
                        "ticker": "AAA",
                        "date": pd.Timestamp("2026-05-20"),
                        "sector": "Energy",
                        "regime_green": True,
                        "relative_strength_index_vs_spy": 80.0,
                        "roc_63": 0.10,
                        "vol_alpha": 1.1,
                        "sma_200_dist": 0.15,
                        "sector_pct_above_50": 0.60,
                        "sector_pct_above_200": 0.60,
                        "sector_median_roc_63": 0.05,
                        "distance_above_20d_high": 0.01,
                        "signal_score": 36.0,
                    },
                    {
                        "ticker": "BBB",
                        "date": pd.Timestamp("2026-05-20"),
                        "sector": "Industrials",
                        "regime_green": True,
                        "relative_strength_index_vs_spy": 91.0,
                        "roc_63": 0.21,
                        "vol_alpha": 0.8,
                        "sma_200_dist": 0.14,
                        "sector_pct_above_50": 0.10,
                        "sector_pct_above_200": 0.10,
                        "sector_median_roc_63": 0.00,
                        "distance_above_20d_high": 0.02,
                        "signal_score": 33.0,
                    },
                ]
            )

            class FakeDB:
                def __init__(self, paths, scan_frame):
                    self.paths = paths
                    self._scan_frame = scan_frame

                def initialize(self): return None
                def load_scan_candidates(self, scan_date=None):
                    return self._scan_frame.copy()
                def load_price_history(self, tickers):
                    return pd.DataFrame()
                def list_universe_rows(self, active_only=True):
                    return [
                        {"ticker": "AAA", "sector": "Energy"},
                        {"ticker": "BBB", "sector": "Industrials"},
                    ]

            service = ScanAnalysisService(FakeDB(paths, scan_frame))
            with patch("src.scan.analysis_service.load_feature_config", return_value={"scan_policy": {"min_opportunity_score": 0.55}}), \
                 patch(
                     "src.scan.analysis_service.load_active_strategies",
                     return_value={
                         "energy": type("Strategy", (), {"slot": "energy", "sector": "Energy", "indicators": {"relative_strength_index_vs_spy_min": 75.0, "signal_score_min": 32.0}})(),
                         "industrials": type("Strategy", (), {"slot": "industrials", "sector": "Industrials", "indicators": {"relative_strength_index_vs_spy_min": 90.0, "signal_score_min": 32.0}})(),
                     },
                 ), \
                 patch("src.scan.analysis_service.build_analysis_frame", return_value=(snapshot, None)), \
                 patch("src.scan.analysis_service.latest_snapshot", return_value=snapshot.copy()), \
                 patch("src.scan.analysis_service.filter_signal_candidates", side_effect=lambda frame, indicators: frame.copy()):
                service.run(scan_date="2026-05-20", horizons=(5,))

            report_text = (paths.reports_dir / "scan_analysis.md").read_text(encoding="utf-8")
            self.assertIn("## Live Slot Gate Waterfall", report_text)
            self.assertIn("### industrials", report_text)
            self.assertIn("- gate_counts: universe=2, regime_green=2, sector_scope=1, relative_strength_index_vs_spy_min=1", report_text)
            self.assertIn("## Live Post-Gate Dropoff", report_text)
            self.assertIn("- gated_count: 1", report_text)
