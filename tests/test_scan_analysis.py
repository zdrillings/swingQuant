from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from src.scan.analysis_service import ScanAnalysisService
from src.settings import AppPaths


class ScanAnalysisServiceTests(unittest.TestCase):
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
                        "shares": 100,
                        "details_json": '{"already_owned": false}',
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
            with patch("src.scan.analysis_service.load_feature_config", return_value={"scan_policy": {"min_opportunity_score": 0.55}}):
                report = service.run(scan_date="2026-05-01", horizons=(5, 10))

            self.assertEqual(report.scan_date, "2026-05-01")
            report_text = (paths.reports_dir / "scan_analysis.md").read_text(encoding="utf-8")
            self.assertIn("# Scan Analysis", report_text)
            self.assertIn("## Selection Summary", report_text)
            self.assertIn("## Forward Attribution", report_text)
            self.assertIn("### 5-Day Forward Return", report_text)
            self.assertIn("### AAA", report_text)
            self.assertIn("### BBB", report_text)

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
            history_dates = pd.bdate_range("2026-04-01", periods=12)
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
            self.assertIn("### Top Learned Buys", report_text)
            self.assertIn("### Best Learned Rejections", report_text)
            self.assertIn("#### BBB", report_text)
            self.assertIn("### energy", report_text)
            self.assertIn("- learned: mean_target=", report_text)
            self.assertIn("runtime_exclusion_reason: portfolio cap / overlap selection loss", report_text)
            self.assertIn("ranker_positive_reasons:", report_text)
