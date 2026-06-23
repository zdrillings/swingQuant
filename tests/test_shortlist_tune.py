from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src.cli import build_parser
from src.research.shortlist_tune_service import ShortlistTuneService
from src.settings import AppPaths


class ShortlistTuneServiceTests(unittest.TestCase):
    def test_shortlist_tune_writes_report(self) -> None:
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

            snapshots = pd.DataFrame(
                [
                    {
                        "snapshot_date": "2026-05-20",
                        "ticker": "AAA",
                        "sector": "Energy",
                        "passed_any_strategy": True,
                        "md_volume_30d": 40_000_000,
                        "adj_close": 50.0,
                        "alpha_vs_sector_20d": 0.08,
                    },
                    {
                        "snapshot_date": "2026-05-20",
                        "ticker": "BBB",
                        "sector": "Materials",
                        "passed_any_strategy": True,
                        "md_volume_30d": 35_000_000,
                        "adj_close": 48.0,
                        "alpha_vs_sector_20d": 0.01,
                    },
                ]
            )

            class FakeDB:
                def __init__(self, paths, snapshots):
                    self.paths = paths
                    self._snapshots = snapshots

                def initialize(self): return None
                def load_universe_daily_snapshots(self): return self._snapshots.copy()

            class TestTuneService(ShortlistTuneService):
                def _walk_forward_predictions(
                    self,
                    frame: pd.DataFrame,
                    *,
                    target_column: str,
                    model_name: str,
                    min_train_dates: int,
                    test_window_dates: int,
                    model_scope: str,
                    xgboost_params=None,
                    feature_columns_override=None,
                ) -> pd.DataFrame | None:
                    score = 0.05
                    if xgboost_params and xgboost_params.get("max_depth") == 3 and xgboost_params.get("min_child_weight") == 3.0:
                        score = 0.09
                    feature_count = len(feature_columns_override or [])
                    if feature_count < 80:
                        score -= 0.02
                    return pd.DataFrame(
                        [
                            {
                                "snapshot_date": pd.Timestamp("2026-05-20"),
                                "ticker": "AAA",
                                "sector": "Energy",
                                "md_volume_30d": 40_000_000,
                                target_column: score,
                                "predicted_alpha": score,
                                "model_top_reasons": ["strong 63d momentum"],
                                "model_reason_summary": "strong 63d momentum",
                            },
                            {
                                "snapshot_date": pd.Timestamp("2026-05-20"),
                                "ticker": "BBB",
                                "sector": "Materials",
                                "md_volume_30d": 35_000_000,
                                target_column: 0.00,
                                "predicted_alpha": 0.00,
                                "model_top_reasons": ["strong volume confirmation"],
                                "model_reason_summary": "strong volume confirmation",
                            },
                        ]
                    )

            report = TestTuneService(FakeDB(paths, snapshots)).run(
                top_n=1,
                horizon_days=20,
                min_train_dates=1,
                test_window_dates=1,
                recent_dates=1,
                eligible_universe_mode="passed_only",
                model_scope="sector_specific",
                mode="full",
                tuning_profile="focused",
                ablation_profile="focused",
            )

            self.assertEqual(report.tuned_candidate, "shallower_regularized")
            self.assertEqual(report.ablation_count, 3)
            report_text = (paths.reports_dir / "shortlist_tune.md").read_text(encoding="utf-8")
            self.assertIn("# Shortlist Tune", report_text)
            self.assertIn("- model_scope: sector_specific", report_text)
            self.assertIn("- mode: full", report_text)
            self.assertIn("- tuning_profile: focused", report_text)
            self.assertIn("- ablation_profile: focused", report_text)
            self.assertIn("## XGBoost Parameter Grid", report_text)
            self.assertIn("### shallower_regularized", report_text)
            self.assertIn("## Feature Ablation", report_text)
            self.assertIn("### no_earnings", report_text)

    def test_shortlist_tune_supports_tune_only_and_ablation_only_modes(self) -> None:
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

            snapshots = pd.DataFrame(
                [
                    {
                        "snapshot_date": "2026-05-20",
                        "ticker": "AAA",
                        "sector": "Energy",
                        "passed_any_strategy": True,
                        "md_volume_30d": 40_000_000,
                        "adj_close": 50.0,
                        "alpha_vs_sector_20d": 0.08,
                    },
                    {
                        "snapshot_date": "2026-05-20",
                        "ticker": "BBB",
                        "sector": "Materials",
                        "passed_any_strategy": True,
                        "md_volume_30d": 35_000_000,
                        "adj_close": 48.0,
                        "alpha_vs_sector_20d": 0.01,
                    },
                ]
            )

            class FakeDB:
                def __init__(self, paths, snapshots):
                    self.paths = paths
                    self._snapshots = snapshots

                def initialize(self): return None
                def load_universe_daily_snapshots(self): return self._snapshots.copy()

            class TestTuneService(ShortlistTuneService):
                def _walk_forward_predictions(
                    self,
                    frame: pd.DataFrame,
                    *,
                    target_column: str,
                    model_name: str,
                    min_train_dates: int,
                    test_window_dates: int,
                    model_scope: str,
                    xgboost_params=None,
                    feature_columns_override=None,
                ) -> pd.DataFrame | None:
                    score = 0.05
                    if xgboost_params and xgboost_params.get("max_depth") == 3 and xgboost_params.get("min_child_weight") == 3.0:
                        score = 0.09
                    feature_count = len(feature_columns_override or [])
                    if feature_count < 80:
                        score -= 0.02
                    return pd.DataFrame(
                        [
                            {
                                "snapshot_date": pd.Timestamp("2026-05-20"),
                                "ticker": "AAA",
                                "sector": "Energy",
                                "md_volume_30d": 40_000_000,
                                target_column: score,
                                "predicted_alpha": score,
                                "model_top_reasons": ["strong 63d momentum"],
                                "model_reason_summary": "strong 63d momentum",
                            },
                            {
                                "snapshot_date": pd.Timestamp("2026-05-20"),
                                "ticker": "BBB",
                                "sector": "Materials",
                                "md_volume_30d": 35_000_000,
                                target_column: 0.00,
                                "predicted_alpha": 0.00,
                                "model_top_reasons": ["strong volume confirmation"],
                                "model_reason_summary": "strong volume confirmation",
                            },
                        ]
                    )

            service = TestTuneService(FakeDB(paths, snapshots))
            tune_only_report = service.run(
                top_n=1,
                horizon_days=20,
                min_train_dates=1,
                test_window_dates=1,
                recent_dates=1,
                eligible_universe_mode="passed_only",
                model_scope="sector_specific",
                mode="tune_only",
            )
            self.assertEqual(tune_only_report.ablation_count, 0)
            tune_only_text = (paths.reports_dir / "shortlist_tune.md").read_text(encoding="utf-8")
            self.assertIn("- skipped in tune_only mode", tune_only_text)

            ablation_only_report = service.run(
                top_n=1,
                horizon_days=20,
                min_train_dates=1,
                test_window_dates=1,
                recent_dates=1,
                eligible_universe_mode="passed_only",
                model_scope="sector_specific",
                mode="ablation_only",
                ablation_params_candidate="shallower_regularized",
            )
            self.assertEqual(ablation_only_report.tuned_candidate, "shallower_regularized")
            self.assertEqual(ablation_only_report.ablation_count, 3)
            ablation_only_text = (paths.reports_dir / "shortlist_tune.md").read_text(encoding="utf-8")
            self.assertIn("- skipped in ablation_only mode", ablation_only_text)

    def test_shortlist_tune_parser_accepts_args(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "shortlist-tune",
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
                "--mode",
                "ablation_only",
                "--tuning-profile",
                "full",
                "--ablation-profile",
                "focused",
                "--ablation-params-candidate",
                "balanced_depth4",
            ]
        )
        self.assertEqual(args.command, "shortlist-tune")
        self.assertEqual(args.top, 6)
        self.assertEqual(args.horizon, 20)
        self.assertEqual(args.min_train_dates, 200)
        self.assertEqual(args.test_window_dates, 15)
        self.assertEqual(args.recent_dates, 30)
        self.assertEqual(args.eligible_universe_mode, "passed_or_trend")
        self.assertEqual(args.model_scope, "sector_specific")
        self.assertEqual(args.mode, "ablation_only")
        self.assertEqual(args.tuning_profile, "full")
        self.assertEqual(args.ablation_profile, "focused")
        self.assertEqual(args.ablation_params_candidate, "balanced_depth4")
