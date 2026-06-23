from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src.cli import build_parser
from src.research.shortlist_bakeoff_service import MODEL_FEATURE_COLUMNS, ShortlistBakeoffService
from src.settings import AppPaths


class ShortlistBakeoffServiceTests(unittest.TestCase):
    def test_shortlist_bakeoff_writes_report(self) -> None:
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
                production_strategies_path=root / "production_strategies.json",
            )
            paths.reports_dir.mkdir(parents=True, exist_ok=True)
            paths.logs_dir.mkdir(parents=True, exist_ok=True)

            snapshot_rows: list[dict[str, object]] = []
            base_dates = pd.bdate_range("2026-04-01", periods=8)
            tickers = [
                ("AAA", "Energy"),
                ("BBB", "Materials"),
                ("CCC", "Industrials"),
                ("DDD", "Information Technology"),
            ]
            for date_index, snapshot_date in enumerate(base_dates):
                for ticker_index, (ticker, sector) in enumerate(tickers):
                    row = {
                        "snapshot_date": snapshot_date.strftime("%Y-%m-%d"),
                        "ticker": ticker,
                        "sector": sector,
                        "passed_any_strategy": 1,
                        "md_volume_30d": 50_000_000.0,
                        "adj_close": 100.0 + ticker_index,
                        "alpha_vs_sector_20d": 0.01 * (ticker_index + 1) + 0.002 * date_index,
                    }
                    for feature_index, column in enumerate(MODEL_FEATURE_COLUMNS):
                        row[column] = float((feature_index + 1) * 0.1 + ticker_index + date_index * 0.05)
                    snapshot_rows.append(row)
            snapshot_frame = pd.DataFrame(snapshot_rows)

            scan_rows: list[dict[str, object]] = []
            scan_dates = pd.bdate_range("2026-04-15", periods=5)
            for date_index, scan_date in enumerate(scan_dates):
                for ticker_index, (ticker, sector) in enumerate(tickers[:3]):
                    scan_rows.append(
                        {
                            "scan_date": scan_date.strftime("%Y-%m-%d"),
                            "ticker": ticker,
                            "strategy_slot": sector.lower(),
                            "selected": 1 if ticker_index < 2 else 0,
                            "opportunity_score": 10.0 - ticker_index - date_index * 0.1,
                            "signal_score": 20.0 - ticker_index * 0.5 + date_index * 0.2,
                            "alpha_vs_sector_20d": 0.015 * (ticker_index + 1) + 0.001 * date_index,
                        }
                    )
            scan_frame = pd.DataFrame(scan_rows)

            class FakeDB:
                def __init__(self, paths, snapshot_frame, scan_frame):
                    self.paths = paths
                    self._snapshot_frame = snapshot_frame
                    self._scan_frame = scan_frame

                def initialize(self): return None
                def load_universe_daily_snapshots(self, snapshot_date=None):
                    return self._snapshot_frame.copy()
                def load_scan_candidates(self, scan_date=None):
                    return self._scan_frame.copy()

            service = ShortlistBakeoffService(FakeDB(paths, snapshot_frame, scan_frame))
            report = service.run(top_n=2, horizon_days=20, recent_dates=3)

            self.assertEqual(report.target_column, "alpha_vs_sector_20d")
            self.assertGreater(report.eligible_rows, 0)
            self.assertGreater(report.test_dates, 0)

            report_text = (paths.reports_dir / "shortlist_bakeoff.md").read_text(encoding="utf-8")
            self.assertIn("# Shortlist Bakeoff", report_text)
            self.assertIn("- eligible_universe_mode: passed_only", report_text)
            self.assertIn("## Universe Model Bakeoff", report_text)
            self.assertIn("## Current Scan Policy Bakeoff", report_text)
            self.assertIn("### ridge_model", report_text)
            self.assertIn("### ensemble_model", report_text)
            self.assertIn("### runtime_selected", report_text)

    def test_shortlist_bakeoff_parser_accepts_args(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "shortlist-bakeoff",
                "--top",
                "8",
                "--horizon",
                "20",
                "--recent-dates",
                "25",
                "--eligible-universe-mode",
                "passed_or_trend",
            ]
        )
        self.assertEqual(args.command, "shortlist-bakeoff")
        self.assertEqual(args.top, 8)
        self.assertEqual(args.horizon, 20)
        self.assertEqual(args.recent_dates, 25)
        self.assertEqual(args.eligible_universe_mode, "passed_or_trend")
