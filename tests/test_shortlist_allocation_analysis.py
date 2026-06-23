from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src.cli import build_parser
from src.research.shortlist_allocation_analysis_service import ShortlistAllocationAnalysisService
from src.settings import AppPaths


class ShortlistAllocationAnalysisServiceTests(unittest.TestCase):
    def test_shortlist_allocation_analysis_writes_report(self) -> None:
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
                        "eligible_universe_mode": "passed_only",
                        "model_scope": "global",
                    }
                ]
            )

            prediction_rows: list[dict[str, object]] = []
            for snapshot_date, date_offset in [("2026-05-20", 0.0), ("2026-05-21", 0.01)]:
                for ticker, sector, predicted_alpha, actual_alpha in [
                    ("AAA", "Energy", 0.12, 0.06 + date_offset),
                    ("BBB", "Energy", 0.11, 0.05 + date_offset),
                    ("CCC", "Industrials", 0.10, 0.04 + date_offset),
                    ("DDD", "Materials", 0.09, 0.03 + date_offset),
                    ("EEE", "Materials", 0.08, 0.02 + date_offset),
                    ("FFF", "Industrials", 0.07, 0.01 + date_offset),
                ]:
                    prediction_rows.append(
                        {
                            "generated_at": "2026-05-26T12:00:00+00:00",
                            "horizon_days": 20,
                            "model_name": "xgboost_model",
                            "dataset_split": "oos",
                            "snapshot_date": snapshot_date,
                            "ticker": ticker,
                            "sector": sector,
                            "md_volume_30d": 10_000_000,
                            "predicted_alpha": predicted_alpha,
                            "actual_alpha_vs_sector": actual_alpha,
                            "eligible_universe_mode": "passed_only",
                            "model_scope": "global",
                        }
                    )
            for ticker, sector, predicted_alpha in [
                ("AAA", "Energy", 0.13),
                ("CCC", "Industrials", 0.12),
                ("DDD", "Materials", 0.11),
                ("BBB", "Energy", 0.10),
            ]:
                prediction_rows.append(
                    {
                        "generated_at": "2026-05-26T12:00:00+00:00",
                        "horizon_days": 20,
                        "model_name": "xgboost_model",
                        "dataset_split": "live",
                        "snapshot_date": "2026-05-23",
                        "ticker": ticker,
                        "sector": sector,
                        "md_volume_30d": 10_000_000,
                        "predicted_alpha": predicted_alpha,
                        "actual_alpha_vs_sector": None,
                        "eligible_universe_mode": "passed_only",
                        "model_scope": "global",
                    }
                )
            predictions = pd.DataFrame(prediction_rows)

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
                    if eligible_universe_mode is not None and "eligible_universe_mode" in frame.columns:
                        frame = frame[frame["eligible_universe_mode"] == eligible_universe_mode].copy()
                    if model_scope is not None and "model_scope" in frame.columns:
                        frame = frame[frame["model_scope"] == model_scope].copy()
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
                    if eligible_universe_mode is not None and "eligible_universe_mode" in frame.columns:
                        frame = frame[frame["eligible_universe_mode"] == eligible_universe_mode].copy()
                    if model_scope is not None and "model_scope" in frame.columns:
                        frame = frame[frame["model_scope"] == model_scope].copy()
                    if dataset_split is not None:
                        frame = frame[frame["dataset_split"] == dataset_split].copy()
                    if model_name is not None:
                        frame = frame[frame["model_name"] == model_name].copy()
                    return frame.reset_index(drop=True)

            service = ShortlistAllocationAnalysisService(FakeDB(paths, runs, predictions))
            report = service.run(top_n=4, horizon_days=20, refresh_if_stale=False)

            self.assertEqual(report.model_name, "xgboost_model")
            self.assertEqual(report.policy_count, 4)
            self.assertEqual(report.scope_count, 1)
            report_text = (paths.reports_dir / "shortlist_allocation_analysis.md").read_text(encoding="utf-8")
            self.assertIn("# Shortlist Allocation Analysis", report_text)
            self.assertIn("- eligible_universe_mode: passed_only", report_text)
            self.assertIn("- selected_model: xgboost_model", report_text)
            self.assertIn("## Scope Summary", report_text)
            self.assertIn("## Full OOS Allocation Comparison", report_text)
            self.assertIn("## Live Allocation Shapes", report_text)
            self.assertIn("#### raw_top_n", report_text)
            self.assertIn("#### sector_round_robin", report_text)

    def test_shortlist_allocation_analysis_can_compare_scopes_for_same_model(self) -> None:
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
                        "eligible_universe_mode": "passed_or_trend",
                        "model_scope": "global",
                    },
                    {
                        "id": 2,
                        "generated_at": "2026-05-26T13:00:00+00:00",
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
                        "eligible_universe_mode": "passed_or_trend",
                        "model_scope": "sector_specific",
                    },
                ]
            )

            prediction_rows: list[dict[str, object]] = []
            for model_scope, base_pred in [("global", 0.10), ("sector_specific", 0.12)]:
                for dataset_split, actual_alpha in [("oos", 0.05), ("live", None)]:
                    for ticker, sector, pred_offset in [
                        ("AAA", "Energy", 0.00),
                        ("BBB", "Industrials", -0.01),
                        ("CCC", "Materials", -0.02),
                    ]:
                        prediction_rows.append(
                            {
                                "generated_at": "2026-05-26T12:00:00+00:00" if model_scope == "global" else "2026-05-26T13:00:00+00:00",
                                "horizon_days": 20,
                                "model_name": "xgboost_model",
                                "dataset_split": dataset_split,
                                "snapshot_date": "2026-05-23",
                                "ticker": ticker,
                                "sector": sector,
                                "md_volume_30d": 10_000_000,
                                "predicted_alpha": base_pred + pred_offset,
                                "actual_alpha_vs_sector": actual_alpha,
                                "eligible_universe_mode": "passed_or_trend",
                                "model_scope": model_scope,
                            }
                        )
            predictions = pd.DataFrame(prediction_rows)

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
                    if eligible_universe_mode is not None:
                        frame = frame[frame["eligible_universe_mode"] == eligible_universe_mode].copy()
                    if model_scope is not None:
                        frame = frame[frame["model_scope"] == model_scope].copy()
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
                    if eligible_universe_mode is not None:
                        frame = frame[frame["eligible_universe_mode"] == eligible_universe_mode].copy()
                    if model_scope is not None:
                        frame = frame[frame["model_scope"] == model_scope].copy()
                    if dataset_split is not None:
                        frame = frame[frame["dataset_split"] == dataset_split].copy()
                    if model_name is not None:
                        frame = frame[frame["model_name"] == model_name].copy()
                    return frame.reset_index(drop=True)

            service = ShortlistAllocationAnalysisService(FakeDB(paths, runs, predictions))
            report = service.run(
                top_n=2,
                horizon_days=20,
                refresh_if_stale=False,
                eligible_universe_mode="passed_or_trend",
                model_name="xgboost_model",
                compare_scopes=True,
            )

            self.assertEqual(report.model_name, "xgboost_model")
            self.assertEqual(report.scope_count, 2)
            report_text = (paths.reports_dir / "shortlist_allocation_analysis.md").read_text(encoding="utf-8")
            self.assertIn("- model_scope: compare_scopes", report_text)
            self.assertIn("## Full OOS Scope Comparison", report_text)
            self.assertIn("- global: mean_target=", report_text)
            self.assertIn("- sector_specific: mean_target=", report_text)
            self.assertIn("### global", report_text)
            self.assertIn("### sector_specific", report_text)

    def test_shortlist_allocation_parser_accepts_args(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "shortlist-allocation-analysis",
                "--top",
                "6",
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
                "--model-name",
                "xgboost_model",
                "--compare-scopes",
                "--no-refresh-if-stale",
            ]
        )
        self.assertEqual(args.command, "shortlist-allocation-analysis")
        self.assertEqual(args.top, 6)
        self.assertEqual(args.horizon, 20)
        self.assertEqual(args.min_train_dates, 200)
        self.assertEqual(args.test_window_dates, 15)
        self.assertEqual(args.recent_dates, 30)
        self.assertEqual(args.eligible_universe_mode, "passed_or_trend")
        self.assertEqual(args.model_scope, "sector_specific")
        self.assertEqual(args.model_name, "xgboost_model")
        self.assertTrue(args.compare_scopes)
        self.assertTrue(args.no_refresh_if_stale)
