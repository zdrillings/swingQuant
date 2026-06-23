from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src.cli import build_parser
from src.scan.performance_service import ScanPerformanceService
from src.settings import AppPaths


class ScanPerformanceServiceTests(unittest.TestCase):
    def test_scan_performance_writes_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            reports_dir = root / "reports"
            reports_dir.mkdir(parents=True, exist_ok=True)

            class FakeDB:
                def __init__(self):
                    self.paths = AppPaths(
                        root_dir=root,
                        data_dir=root / "data",
                        duckdb_path=root / "data" / "market_data.duckdb",
                        sqlite_path=root / "data" / "ledger.sqlite",
                        reports_dir=reports_dir,
                        logs_dir=root / "logs",
                        config_path=root / "config.yaml",
                        env_path=root / ".env",
                        production_strategy_path=root / "production_strategy.json",
                    )

                def initialize(self): return None

                def load_scan_candidates(self, scan_date=None):
                    return pd.DataFrame(
                        [
                            {
                                "scan_date": "2026-05-01",
                                "ticker": "AAA",
                                "strategy_slot": "industrials",
                                "strategy_sector": "Industrials",
                                "sector": "Industrials",
                                "selected": 1,
                                "selected_rank": 1,
                                "selection_source": "shortlist_model",
                                "model_name": "xgboost_model",
                                "model_generated_at": "2026-06-02T20:08:10+00:00",
                            },
                            {
                                "scan_date": "2026-05-02",
                                "ticker": "BBB",
                                "strategy_slot": "materials",
                                "strategy_sector": "Materials",
                                "sector": "Materials",
                                "selected": 1,
                                "selected_rank": 1,
                                "selection_source": "heuristic",
                                "model_name": None,
                                "model_generated_at": None,
                            },
                        ]
                    )

                def load_price_history(self, tickers):
                    return pd.DataFrame(
                        [
                            {"ticker": "AAA", "date": "2026-05-01", "adj_close": 10.0},
                            {"ticker": "AAA", "date": "2026-05-02", "adj_close": 10.5},
                            {"ticker": "AAA", "date": "2026-05-05", "adj_close": 11.0},
                            {"ticker": "AAA", "date": "2026-05-08", "adj_close": 12.0},
                            {"ticker": "AAA", "date": "2026-05-15", "adj_close": 13.0},
                            {"ticker": "AAA", "date": "2026-06-02", "adj_close": 14.0},
                            {"ticker": "BBB", "date": "2026-05-02", "adj_close": 20.0},
                            {"ticker": "BBB", "date": "2026-05-05", "adj_close": 20.5},
                            {"ticker": "BBB", "date": "2026-05-06", "adj_close": 21.0},
                            {"ticker": "BBB", "date": "2026-05-09", "adj_close": 22.0},
                            {"ticker": "BBB", "date": "2026-05-16", "adj_close": 23.0},
                            {"ticker": "BBB", "date": "2026-06-03", "adj_close": 24.0},
                            {"ticker": "SPY", "date": "2026-05-01", "adj_close": 100.0},
                            {"ticker": "SPY", "date": "2026-05-02", "adj_close": 101.0},
                            {"ticker": "SPY", "date": "2026-05-05", "adj_close": 102.0},
                            {"ticker": "SPY", "date": "2026-05-06", "adj_close": 103.0},
                            {"ticker": "SPY", "date": "2026-05-08", "adj_close": 104.0},
                            {"ticker": "SPY", "date": "2026-05-09", "adj_close": 105.0},
                            {"ticker": "SPY", "date": "2026-05-15", "adj_close": 106.0},
                            {"ticker": "SPY", "date": "2026-05-16", "adj_close": 107.0},
                            {"ticker": "SPY", "date": "2026-06-02", "adj_close": 108.0},
                            {"ticker": "SPY", "date": "2026-06-03", "adj_close": 109.0},
                            {"ticker": "XLI", "date": "2026-05-01", "adj_close": 50.0},
                            {"ticker": "XLI", "date": "2026-05-02", "adj_close": 50.5},
                            {"ticker": "XLI", "date": "2026-05-05", "adj_close": 51.0},
                            {"ticker": "XLI", "date": "2026-05-08", "adj_close": 52.0},
                            {"ticker": "XLI", "date": "2026-05-15", "adj_close": 53.0},
                            {"ticker": "XLI", "date": "2026-06-02", "adj_close": 54.0},
                            {"ticker": "XLB", "date": "2026-05-02", "adj_close": 60.0},
                            {"ticker": "XLB", "date": "2026-05-05", "adj_close": 60.5},
                            {"ticker": "XLB", "date": "2026-05-06", "adj_close": 61.0},
                            {"ticker": "XLB", "date": "2026-05-09", "adj_close": 61.5},
                            {"ticker": "XLB", "date": "2026-05-16", "adj_close": 62.0},
                            {"ticker": "XLB", "date": "2026-06-03", "adj_close": 63.0},
                        ]
                    )

            report = ScanPerformanceService(FakeDB()).run(recent_scan_dates=60, recent_picks=5, benchmark="sector")

            self.assertTrue(report.output_path.endswith("scan_performance.md"))
            report_text = (reports_dir / "scan_performance.md").read_text(encoding="utf-8")
            self.assertIn("# Scan Performance", report_text)
            self.assertIn("- scope: latest_model", report_text)
            self.assertIn("- scan_dates: 1", report_text)
            self.assertIn("### 2d", report_text)
            self.assertIn("### 60d", report_text)
            self.assertIn("return_p05_p95", report_text)
            self.assertIn("return_range", report_text)
            self.assertIn("mean_alpha_vs_sector", report_text)
            self.assertIn("## Best And Worst Picks", report_text)
            self.assertIn("## Repeated Winners And Losers", report_text)
            self.assertIn("## Recent Picks", report_text)

    def test_scan_performance_can_filter_model_attributed_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            reports_dir = root / "reports"
            reports_dir.mkdir(parents=True, exist_ok=True)

            class FakeDB:
                def __init__(self):
                    self.paths = AppPaths(
                        root_dir=root,
                        data_dir=root / "data",
                        duckdb_path=root / "data" / "market_data.duckdb",
                        sqlite_path=root / "data" / "ledger.sqlite",
                        reports_dir=reports_dir,
                        logs_dir=root / "logs",
                        config_path=root / "config.yaml",
                        env_path=root / ".env",
                        production_strategy_path=root / "production_strategy.json",
                    )

                def initialize(self): return None

                def load_scan_candidates(self, scan_date=None):
                    return pd.DataFrame(
                        [
                            {
                                "scan_date": "2026-05-01",
                                "ticker": "AAA",
                                "strategy_slot": "industrials",
                                "strategy_sector": "Industrials",
                                "sector": "Industrials",
                                "selected": 1,
                                "selected_rank": 1,
                                "selection_source": "shortlist_model",
                                "model_name": "xgboost_model",
                                "model_generated_at": "2026-06-02T20:08:10+00:00",
                            },
                            {
                                "scan_date": "2026-05-02",
                                "ticker": "BBB",
                                "strategy_slot": "materials",
                                "strategy_sector": "Materials",
                                "sector": "Materials",
                                "selected": 1,
                                "selected_rank": 1,
                                "selection_source": "heuristic",
                                "model_name": None,
                                "model_generated_at": None,
                            },
                        ]
                    )

                def load_price_history(self, tickers):
                    return pd.DataFrame(
                        [
                            {"ticker": "AAA", "date": "2026-05-01", "adj_close": 10.0},
                            {"ticker": "AAA", "date": "2026-05-02", "adj_close": 10.5},
                            {"ticker": "AAA", "date": "2026-05-05", "adj_close": 11.0},
                            {"ticker": "AAA", "date": "2026-05-08", "adj_close": 12.0},
                            {"ticker": "AAA", "date": "2026-05-15", "adj_close": 13.0},
                            {"ticker": "AAA", "date": "2026-06-02", "adj_close": 14.0},
                            {"ticker": "SPY", "date": "2026-05-01", "adj_close": 100.0},
                            {"ticker": "SPY", "date": "2026-05-02", "adj_close": 101.0},
                            {"ticker": "SPY", "date": "2026-05-05", "adj_close": 102.0},
                            {"ticker": "SPY", "date": "2026-05-08", "adj_close": 104.0},
                            {"ticker": "SPY", "date": "2026-05-15", "adj_close": 106.0},
                            {"ticker": "SPY", "date": "2026-06-02", "adj_close": 108.0},
                            {"ticker": "XLI", "date": "2026-05-01", "adj_close": 50.0},
                            {"ticker": "XLI", "date": "2026-05-02", "adj_close": 50.5},
                            {"ticker": "XLI", "date": "2026-05-05", "adj_close": 51.0},
                            {"ticker": "XLI", "date": "2026-05-08", "adj_close": 52.0},
                            {"ticker": "XLI", "date": "2026-05-15", "adj_close": 53.0},
                            {"ticker": "XLI", "date": "2026-06-02", "adj_close": 54.0},
                        ]
                    )

            report = ScanPerformanceService(FakeDB()).run(
                recent_scan_dates=60,
                recent_picks=5,
                benchmark="sector",
                selection_source="shortlist_model",
                model_name="xgboost_model",
                model_generated_at="2026-06-02T20:08:10+00:00",
            )

            self.assertEqual(report.selected_rows, 1)
            report_text = (reports_dir / "scan_performance.md").read_text(encoding="utf-8")
            self.assertIn("- scope: explicit", report_text)
            self.assertIn("- selection_source: shortlist_model", report_text)
            self.assertIn("- model_name: xgboost_model", report_text)
            self.assertIn("- model_generated_at: 2026-06-02T20:08:10+00:00", report_text)
            self.assertIn("### AAA", report_text)
            self.assertNotIn("### BBB", report_text)

    def test_scan_performance_renders_20d_opportunity_score_bands(self) -> None:
        service = ScanPerformanceService(db_manager=None)
        frame = pd.DataFrame(
            [
                {
                    "ticker": "AAA",
                    "opportunity_score": 0.25,
                    "fwd_return_20d": -0.01,
                    "alpha_vs_sector_20d": -0.02,
                },
                {
                    "ticker": "BBB",
                    "opportunity_score": 0.42,
                    "fwd_return_20d": 0.05,
                    "alpha_vs_sector_20d": 0.03,
                },
                {
                    "ticker": "CCC",
                    "opportunity_score": 0.52,
                    "fwd_return_20d": 0.14,
                    "alpha_vs_sector_20d": 0.11,
                },
            ]
        )

        lines = service._render_20d_score_bands(frame, benchmark="sector")
        report_text = "\n".join(lines)

        self.assertIn("## 20d Opportunity Score Bands", report_text)
        self.assertIn("- return: fwd_return_20d", report_text)
        self.assertIn("- alpha: alpha_vs_sector_20d", report_text)
        self.assertIn("- observations: 3", report_text)
        self.assertIn("- score < 0.30: n=1, pick_share=33.33%", report_text)
        self.assertIn("0.40 <= score < 0.45: n=1, pick_share=33.33%", report_text)
        self.assertIn("score >= 0.50: n=1, pick_share=33.33%", report_text)
        self.assertIn("mean_return=5.00%", report_text)
        self.assertIn("median_alpha=3.00%", report_text)

    def test_scan_performance_optionally_emails_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            reports_dir = root / "reports"
            reports_dir.mkdir(parents=True, exist_ok=True)

            class FakeDB:
                def __init__(self):
                    self.paths = AppPaths(
                        root_dir=root,
                        data_dir=root / "data",
                        duckdb_path=root / "data" / "market_data.duckdb",
                        sqlite_path=root / "data" / "ledger.sqlite",
                        reports_dir=reports_dir,
                        logs_dir=root / "logs",
                        config_path=root / "config.yaml",
                        env_path=root / ".env",
                        production_strategy_path=root / "production_strategy.json",
                    )

                def initialize(self): return None

                def load_scan_candidates(self, scan_date=None):
                    return pd.DataFrame(
                        [
                            {
                                "scan_date": "2026-05-01",
                                "ticker": "AAA",
                                "strategy_slot": "industrials",
                                "strategy_sector": "Industrials",
                                "sector": "Industrials",
                                "selected": 1,
                                "selected_rank": 1,
                            },
                        ]
                    )

                def load_price_history(self, tickers):
                    return pd.DataFrame(
                        [
                            {"ticker": "AAA", "date": "2026-05-01", "adj_close": 10.0},
                            {"ticker": "AAA", "date": "2026-05-05", "adj_close": 11.0},
                            {"ticker": "AAA", "date": "2026-05-08", "adj_close": 12.0},
                            {"ticker": "AAA", "date": "2026-05-15", "adj_close": 13.0},
                            {"ticker": "AAA", "date": "2026-06-02", "adj_close": 14.0},
                            {"ticker": "SPY", "date": "2026-05-01", "adj_close": 100.0},
                            {"ticker": "SPY", "date": "2026-05-05", "adj_close": 101.0},
                            {"ticker": "SPY", "date": "2026-05-08", "adj_close": 102.0},
                            {"ticker": "SPY", "date": "2026-05-15", "adj_close": 103.0},
                            {"ticker": "SPY", "date": "2026-06-02", "adj_close": 104.0},
                            {"ticker": "XLI", "date": "2026-05-01", "adj_close": 50.0},
                            {"ticker": "XLI", "date": "2026-05-05", "adj_close": 51.0},
                            {"ticker": "XLI", "date": "2026-05-08", "adj_close": 52.0},
                            {"ticker": "XLI", "date": "2026-05-15", "adj_close": 53.0},
                            {"ticker": "XLI", "date": "2026-06-02", "adj_close": 54.0},
                        ]
                    )

            email_calls = []

            class FakeSettings:
                env = {}

            def fake_email_sender(*, subject, html_body, settings):
                email_calls.append((subject, html_body, settings))

            from unittest.mock import patch

            with patch("src.scan.performance_service.get_settings", return_value=FakeSettings()):
                ScanPerformanceService(FakeDB(), email_sender=fake_email_sender).run(
                    recent_scan_dates=60,
                    recent_picks=5,
                    benchmark="sector",
                    email=True,
                )

            self.assertEqual(len(email_calls), 1)
            subject, html_body, _ = email_calls[0]
            self.assertIn("Scan Performance", subject)
            self.assertIn("<html>", html_body)
            self.assertIn("Horizon Summary", html_body)

    def test_scan_performance_parser_accepts_args(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "scan-performance",
                "--recent-scan-dates",
                "45",
                "--recent-picks",
                "12",
                "--benchmark",
                "spy",
                "--selection-source",
                "shortlist_model",
                "--model-name",
                "xgboost_model",
                "--model-generated-at",
                "2026-06-02T20:08:10+00:00",
                "--all-sources",
                "--email",
            ]
        )
        self.assertEqual(args.command, "scan-performance")
        self.assertEqual(args.recent_scan_dates, 45)
        self.assertEqual(args.recent_picks, 12)
        self.assertEqual(args.benchmark, "spy")
        self.assertEqual(args.selection_source, "shortlist_model")
        self.assertEqual(args.model_name, "xgboost_model")
        self.assertEqual(args.model_generated_at, "2026-06-02T20:08:10+00:00")
        self.assertTrue(args.all_sources)
        self.assertTrue(args.email)

    def test_scan_performance_parser_defaults_to_all_dates_latest_model_scope(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["scan-performance"])
        self.assertEqual(args.recent_scan_dates, 0)
        self.assertFalse(args.all_sources)
