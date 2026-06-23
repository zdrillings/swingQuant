from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from src.cli import build_parser
from src.research.shortlist_scoreboard_service import ShortlistScoreboardService
from src.settings import AppPaths


class ShortlistScoreboardServiceTests(unittest.TestCase):
    def test_recommend_model_prefers_stronger_full_beat_universe_profile(self) -> None:
        service = ShortlistScoreboardService(db_manager=object())
        promotion_frame = pd.DataFrame(
            [
                {
                    "model": "xgboost_model",
                    "decision": "promote_strong",
                    "full_status": "strong",
                    "recent40_status": "strong",
                    "full_mean_target": 0.08,
                    "full_beat_universe_rate": 0.72,
                    "full_positive_date_rate": 0.77,
                    "recent40_mean_target": 0.09,
                    "recent40_beat_universe_rate": 0.80,
                    "recent40_positive_date_rate": 0.92,
                    "avg_max_sector_share": 0.46,
                    "avg_sector_hhi": 0.35,
                },
                {
                    "model": "ridge_model",
                    "decision": "promote_strong",
                    "full_status": "strong",
                    "recent40_status": "strong",
                    "full_mean_target": 0.05,
                    "full_beat_universe_rate": 0.68,
                    "full_positive_date_rate": 0.73,
                    "recent40_mean_target": 0.10,
                    "recent40_beat_universe_rate": 0.83,
                    "recent40_positive_date_rate": 0.98,
                    "avg_max_sector_share": 0.50,
                    "avg_sector_hhi": 0.39,
                },
            ]
        )

        self.assertEqual(service._recommend_model(promotion_frame), "xgboost_model")

    def test_shortlist_scoreboard_writes_report(self) -> None:
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
                        "generated_at": "2026-05-26T12:00:00+00:00",
                        "horizon_days": 20,
                        "top_n": 3,
                        "min_train_dates": 252,
                        "test_window_dates": 20,
                        "recent_dates": 60,
                        "champion_model": "ensemble_model",
                        "target_column": "alpha_vs_sector_20d",
                        "eligible_rows": 1200,
                        "eligible_dates": 300,
                        "oos_dates": 80,
                        "live_snapshot_date": "2026-05-23",
                        "report_path": str(paths.reports_dir / "shortlist_model.md"),
                    }
                ]
            )
            predictions = pd.DataFrame(
                [
                    {
                        "generated_at": "2026-05-26T12:00:00+00:00",
                        "horizon_days": 20,
                        "model_name": "xgboost_model",
                        "dataset_split": "oos",
                        "snapshot_date": "2026-05-20",
                        "ticker": "AAA",
                        "sector": "Energy",
                        "md_volume_30d": 40_000_000,
                        "predicted_alpha": 0.30,
                        "actual_alpha_vs_sector": 0.08,
                    },
                    {
                        "generated_at": "2026-05-26T12:00:00+00:00",
                        "horizon_days": 20,
                        "model_name": "xgboost_model",
                        "dataset_split": "oos",
                        "snapshot_date": "2026-05-20",
                        "ticker": "BBB",
                        "sector": "Materials",
                        "md_volume_30d": 30_000_000,
                        "predicted_alpha": 0.20,
                        "actual_alpha_vs_sector": 0.05,
                    },
                    {
                        "generated_at": "2026-05-26T12:00:00+00:00",
                        "horizon_days": 20,
                        "model_name": "xgboost_model",
                        "dataset_split": "oos",
                        "snapshot_date": "2026-05-20",
                        "ticker": "CCC",
                        "sector": "Industrials",
                        "md_volume_30d": 20_000_000,
                        "predicted_alpha": 0.10,
                        "actual_alpha_vs_sector": 0.01,
                    },
                    {
                        "generated_at": "2026-05-26T12:00:00+00:00",
                        "horizon_days": 20,
                        "model_name": "xgboost_model",
                        "dataset_split": "oos",
                        "snapshot_date": "2026-05-21",
                        "ticker": "AAA",
                        "sector": "Energy",
                        "md_volume_30d": 40_000_000,
                        "predicted_alpha": 0.28,
                        "actual_alpha_vs_sector": 0.07,
                    },
                    {
                        "generated_at": "2026-05-26T12:00:00+00:00",
                        "horizon_days": 20,
                        "model_name": "xgboost_model",
                        "dataset_split": "oos",
                        "snapshot_date": "2026-05-21",
                        "ticker": "BBB",
                        "sector": "Materials",
                        "md_volume_30d": 30_000_000,
                        "predicted_alpha": 0.18,
                        "actual_alpha_vs_sector": 0.04,
                    },
                    {
                        "generated_at": "2026-05-26T12:00:00+00:00",
                        "horizon_days": 20,
                        "model_name": "xgboost_model",
                        "dataset_split": "oos",
                        "snapshot_date": "2026-05-21",
                        "ticker": "CCC",
                        "sector": "Industrials",
                        "md_volume_30d": 20_000_000,
                        "predicted_alpha": 0.11,
                        "actual_alpha_vs_sector": 0.02,
                    },
                    {
                        "generated_at": "2026-05-26T12:00:00+00:00",
                        "horizon_days": 20,
                        "model_name": "ensemble_model",
                        "dataset_split": "oos",
                        "snapshot_date": "2026-05-20",
                        "ticker": "AAA",
                        "sector": "Energy",
                        "md_volume_30d": 40_000_000,
                        "predicted_alpha": 0.21,
                        "actual_alpha_vs_sector": 0.02,
                    },
                    {
                        "generated_at": "2026-05-26T12:00:00+00:00",
                        "horizon_days": 20,
                        "model_name": "ensemble_model",
                        "dataset_split": "oos",
                        "snapshot_date": "2026-05-20",
                        "ticker": "BBB",
                        "sector": "Materials",
                        "md_volume_30d": 30_000_000,
                        "predicted_alpha": 0.19,
                        "actual_alpha_vs_sector": 0.01,
                    },
                    {
                        "generated_at": "2026-05-26T12:00:00+00:00",
                        "horizon_days": 20,
                        "model_name": "ensemble_model",
                        "dataset_split": "oos",
                        "snapshot_date": "2026-05-20",
                        "ticker": "CCC",
                        "sector": "Industrials",
                        "md_volume_30d": 20_000_000,
                        "predicted_alpha": 0.18,
                        "actual_alpha_vs_sector": -0.01,
                    },
                    {
                        "generated_at": "2026-05-26T12:00:00+00:00",
                        "horizon_days": 20,
                        "model_name": "ensemble_model",
                        "dataset_split": "oos",
                        "snapshot_date": "2026-05-21",
                        "ticker": "AAA",
                        "sector": "Energy",
                        "md_volume_30d": 40_000_000,
                        "predicted_alpha": 0.20,
                        "actual_alpha_vs_sector": 0.01,
                    },
                    {
                        "generated_at": "2026-05-26T12:00:00+00:00",
                        "horizon_days": 20,
                        "model_name": "ensemble_model",
                        "dataset_split": "oos",
                        "snapshot_date": "2026-05-21",
                        "ticker": "BBB",
                        "sector": "Materials",
                        "md_volume_30d": 30_000_000,
                        "predicted_alpha": 0.17,
                        "actual_alpha_vs_sector": 0.00,
                    },
                    {
                        "generated_at": "2026-05-26T12:00:00+00:00",
                        "horizon_days": 20,
                        "model_name": "ensemble_model",
                        "dataset_split": "oos",
                        "snapshot_date": "2026-05-21",
                        "ticker": "CCC",
                        "sector": "Industrials",
                        "md_volume_30d": 20_000_000,
                        "predicted_alpha": 0.16,
                        "actual_alpha_vs_sector": -0.02,
                    },
                    {
                        "generated_at": "2026-05-26T12:00:00+00:00",
                        "horizon_days": 20,
                        "model_name": "xgboost_model",
                        "dataset_split": "live",
                        "snapshot_date": "2026-05-23",
                        "ticker": "AAA",
                        "sector": "Energy",
                        "md_volume_30d": 40_000_000,
                        "predicted_alpha": 0.33,
                        "actual_alpha_vs_sector": None,
                    },
                    {
                        "generated_at": "2026-05-26T12:00:00+00:00",
                        "horizon_days": 20,
                        "model_name": "xgboost_model",
                        "dataset_split": "live",
                        "snapshot_date": "2026-05-23",
                        "ticker": "BBB",
                        "sector": "Materials",
                        "md_volume_30d": 30_000_000,
                        "predicted_alpha": 0.22,
                        "actual_alpha_vs_sector": None,
                    },
                    {
                        "generated_at": "2026-05-26T12:00:00+00:00",
                        "horizon_days": 20,
                        "model_name": "ensemble_model",
                        "dataset_split": "live",
                        "snapshot_date": "2026-05-23",
                        "ticker": "AAA",
                        "sector": "Energy",
                        "md_volume_30d": 40_000_000,
                        "predicted_alpha": 0.23,
                        "actual_alpha_vs_sector": None,
                    },
                    {
                        "generated_at": "2026-05-26T12:00:00+00:00",
                        "horizon_days": 20,
                        "model_name": "ensemble_model",
                        "dataset_split": "live",
                        "snapshot_date": "2026-05-23",
                        "ticker": "BBB",
                        "sector": "Materials",
                        "md_volume_30d": 30_000_000,
                        "predicted_alpha": 0.21,
                        "actual_alpha_vs_sector": None,
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
                    return ["2026-05-23"]

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

            with patch("src.research.shortlist_scoreboard_service.load_feature_config", return_value={"scan_policy": {"shortlist_model": {"production_model_name": "xgboost_model"}}}):
                report = ShortlistScoreboardService(FakeDB(paths, runs, predictions)).run(
                    top_n=2,
                    horizon_days=20,
                    refresh_if_stale=False,
                )

            self.assertEqual(report.recommended_model, "xgboost_model")
            self.assertEqual(report.production_model, "xgboost_model")
            self.assertEqual(report.model_count, 2)
            report_text = (paths.reports_dir / "shortlist_scoreboard.md").read_text(encoding="utf-8")
            self.assertIn("# Shortlist Scoreboard", report_text)
            self.assertIn("- eligible_universe_mode: passed_only", report_text)
            self.assertIn("## Promotion Decisions", report_text)
            self.assertIn("## Full OOS Scoreboard", report_text)
            self.assertIn("## Recent 40 OOS Dates", report_text)
            self.assertIn("## Window Scorecards", report_text)
            self.assertIn("## Sector Scorecards", report_text)
            self.assertIn("## Live Concentration", report_text)
            self.assertIn("### xgboost_model", report_text)

    def test_shortlist_scoreboard_parser_accepts_args(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "shortlist-scoreboard",
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
        self.assertEqual(args.command, "shortlist-scoreboard")
        self.assertEqual(args.top, 8)
        self.assertEqual(args.horizon, 20)
        self.assertEqual(args.min_train_dates, 200)
        self.assertEqual(args.test_window_dates, 15)
        self.assertEqual(args.recent_dates, 30)
        self.assertEqual(args.eligible_universe_mode, "passed_or_trend")
        self.assertEqual(args.model_scope, "sector_specific")
        self.assertTrue(args.no_refresh_if_stale)
