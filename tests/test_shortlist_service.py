from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src.cli import build_parser
from src.research.shortlist_service import ShortlistService
from src.settings import AppPaths


class ShortlistServiceTests(unittest.TestCase):
    def test_shortlist_reads_latest_persisted_run(self) -> None:
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

            runs = pd.DataFrame(
                [
                    {
                        "id": 1,
                        "generated_at": "2026-05-26T12:00:00",
                        "horizon_days": 20,
                        "top_n": 10,
                        "min_train_dates": 252,
                        "test_window_dates": 20,
                        "recent_dates": 60,
                        "champion_model": "xgboost_model",
                        "target_column": "alpha_vs_sector_20d",
                        "eligible_rows": 1000,
                        "eligible_dates": 300,
                        "oos_dates": 100,
                        "live_snapshot_date": "2026-05-23",
                        "report_path": str(paths.reports_dir / "shortlist_model.md"),
                    }
                ]
            )
            predictions = pd.DataFrame(
                [
                    {
                        "generated_at": "2026-05-26T12:00:00",
                        "horizon_days": 20,
                        "model_name": "xgboost_model",
                        "dataset_split": "oos",
                        "snapshot_date": "2026-05-20",
                        "ticker": "AAA",
                        "sector": "Energy",
                        "md_volume_30d": 10_000_000,
                        "predicted_alpha": 0.10,
                        "actual_alpha_vs_sector": 0.06,
                    },
                    {
                        "generated_at": "2026-05-26T12:00:00",
                        "horizon_days": 20,
                        "model_name": "xgboost_model",
                        "dataset_split": "oos",
                        "snapshot_date": "2026-05-20",
                        "ticker": "BBB",
                        "sector": "Materials",
                        "md_volume_30d": 11_000_000,
                        "predicted_alpha": 0.08,
                        "actual_alpha_vs_sector": 0.03,
                    },
                    {
                        "generated_at": "2026-05-26T12:00:00",
                        "horizon_days": 20,
                        "model_name": "xgboost_model",
                        "dataset_split": "oos",
                        "snapshot_date": "2026-05-21",
                        "ticker": "AAA",
                        "sector": "Energy",
                        "md_volume_30d": 10_000_000,
                        "predicted_alpha": 0.11,
                        "actual_alpha_vs_sector": 0.04,
                    },
                    {
                        "generated_at": "2026-05-26T12:00:00",
                        "horizon_days": 20,
                        "model_name": "xgboost_model",
                        "dataset_split": "oos",
                        "snapshot_date": "2026-05-21",
                        "ticker": "BBB",
                        "sector": "Materials",
                        "md_volume_30d": 11_000_000,
                        "predicted_alpha": 0.07,
                        "actual_alpha_vs_sector": -0.01,
                    },
                    {
                        "generated_at": "2026-05-26T12:00:00",
                        "horizon_days": 20,
                        "model_name": "xgboost_model",
                        "dataset_split": "live",
                        "snapshot_date": "2026-05-23",
                        "ticker": "AAA",
                        "sector": "Energy",
                        "md_volume_30d": 10_000_000,
                        "predicted_alpha": 0.12,
                        "actual_alpha_vs_sector": None,
                        "details_json": '{"model_top_reasons": ["strong 63d momentum", "strong RS vs SPY"], "model_reason_summary": "strong 63d momentum, strong RS vs SPY"}',
                    },
                    {
                        "generated_at": "2026-05-26T12:00:00",
                        "horizon_days": 20,
                        "model_name": "xgboost_model",
                        "dataset_split": "live",
                        "snapshot_date": "2026-05-23",
                        "ticker": "BBB",
                        "sector": "Materials",
                        "md_volume_30d": 11_000_000,
                        "predicted_alpha": 0.09,
                        "actual_alpha_vs_sector": None,
                        "details_json": '{"model_top_reasons": ["strong volume confirmation"], "model_reason_summary": "strong volume confirmation"}',
                    },
                    {
                        "generated_at": "2026-05-26T12:00:00",
                        "horizon_days": 20,
                        "model_name": "xgboost_model",
                        "dataset_split": "live",
                        "snapshot_date": "2026-05-23",
                        "ticker": "CCC",
                        "sector": "Industrials",
                        "md_volume_30d": 9_000_000,
                        "predicted_alpha": 0.07,
                        "actual_alpha_vs_sector": None,
                        "details_json": '{"model_top_reasons": ["strong RS vs SPY"], "model_reason_summary": "strong RS vs SPY"}',
                    },
                ]
            )

            class FakeDB:
                def __init__(self, paths, runs, predictions):
                    self.paths = paths
                    self._runs = runs
                    self._predictions = predictions

                def initialize(self): return None
                def load_shortlist_model_runs(self, *, horizon_days=None, eligible_universe_mode=None, model_scope=None, xgboost_config=None, limit=None):
                    frame = self._runs.copy()
                    if horizon_days is not None:
                        frame = frame[frame["horizon_days"] == horizon_days].copy()
                    if limit is not None:
                        frame = frame.head(limit).copy()
                    return frame.reset_index(drop=True)
                def list_universe_daily_snapshot_dates(self):
                    return ["2026-05-22", "2026-05-23"]
                def load_shortlist_model_predictions(self, *, generated_at=None, horizon_days=None, eligible_universe_mode=None, model_scope=None, dataset_split=None, model_name=None):
                    frame = self._predictions.copy()
                    if generated_at is not None:
                        frame = frame[frame["generated_at"] == generated_at].copy()
                    if horizon_days is not None:
                        frame = frame[frame["horizon_days"] == horizon_days].copy()
                    if dataset_split is not None:
                        frame = frame[frame["dataset_split"] == dataset_split].copy()
                    if model_name is not None:
                        frame = frame[frame["model_name"] == model_name].copy()
                    return frame.reset_index(drop=True)

            service = ShortlistService(FakeDB(paths, runs, predictions))
            report = service.run(top_n=2, horizon_days=20, refresh_if_stale=False)

            self.assertEqual(report.champion_model, "xgboost_model")
            self.assertEqual(report.candidate_count, 2)
            report_text = (paths.reports_dir / "shortlist.md").read_text(encoding="utf-8")
            self.assertIn("# Shortlist", report_text)
            self.assertIn("- eligible_universe_mode: passed_only", report_text)
            self.assertIn("## Acceptance Windows", report_text)
            self.assertIn("## OOS Sector Contribution", report_text)
            self.assertIn("## Live Candidates", report_text)
            self.assertIn("### AAA", report_text)
            self.assertIn("- why: strong 63d momentum, strong RS vs SPY", report_text)
            self.assertIn("- won over: next-ranked CCC on strong 63d momentum", report_text)

    def test_shortlist_parser_accepts_args(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "shortlist",
                "--top",
                "8",
                "--horizon",
                "20",
                "--min-train-dates",
                "200",
                "--test-window-dates",
                "15",
                "--recent-dates",
                "30",
                "--eligible-universe-mode",
                "passed_or_trend",
                "--model-scope",
                "sector_specific",
                "--no-refresh-if-stale",
            ]
        )
        self.assertEqual(args.command, "shortlist")
        self.assertEqual(args.top, 8)
        self.assertEqual(args.horizon, 20)
        self.assertEqual(args.min_train_dates, 200)
        self.assertEqual(args.test_window_dates, 15)
        self.assertEqual(args.recent_dates, 30)
        self.assertEqual(args.eligible_universe_mode, "passed_or_trend")
        self.assertEqual(args.model_scope, "sector_specific")
        self.assertTrue(args.no_refresh_if_stale)
