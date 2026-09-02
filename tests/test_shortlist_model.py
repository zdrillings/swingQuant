from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from src.cli import build_parser
from src.research.shortlist_bakeoff_service import MODEL_FEATURE_COLUMNS
from src.research.shortlist_model_service import ShortlistModelService
from src.research.shortlist_universe import filter_eligible_universe
from src.settings import AppPaths
from src.utils.shortlist_runtime import load_live_shortlist_model_context


class ShortlistModelServiceTests(unittest.TestCase):
    def test_filter_eligible_universe_passed_or_trend_broadens_research_set(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "ticker": "AAA",
                    "passed_any_strategy": 1,
                    "md_volume_30d": 30_000_000.0,
                    "adj_close": 100.0,
                    "regime_green": 0,
                    "sma_200_dist": -0.02,
                    "roc_63": -0.01,
                    "relative_strength_index_vs_spy": 40.0,
                },
                {
                    "ticker": "BBB",
                    "passed_any_strategy": 0,
                    "md_volume_30d": 35_000_000.0,
                    "adj_close": 90.0,
                    "regime_green": 1,
                    "sma_200_dist": 0.10,
                    "roc_63": 0.08,
                    "relative_strength_index_vs_spy": 68.0,
                },
                {
                    "ticker": "CCC",
                    "passed_any_strategy": 0,
                    "md_volume_30d": 35_000_000.0,
                    "adj_close": 90.0,
                    "regime_green": 1,
                    "sma_200_dist": -0.01,
                    "roc_63": 0.08,
                    "relative_strength_index_vs_spy": 68.0,
                },
            ]
        )

        passed_only = filter_eligible_universe(frame, eligible_universe_mode="passed_only")
        passed_or_trend = filter_eligible_universe(frame, eligible_universe_mode="passed_or_trend")

        self.assertEqual(sorted(passed_only["ticker"].tolist()), ["AAA"])
        self.assertEqual(sorted(passed_or_trend["ticker"].tolist()), ["AAA", "BBB"])

    def test_filter_eligible_universe_prefers_historical_passed_slots_json(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "ticker": "AAA",
                    "passed_any_strategy": 0,
                    "passed_slots_json": '["energy"]',
                    "md_volume_30d": 30_000_000.0,
                    "adj_close": 100.0,
                },
                {
                    "ticker": "BBB",
                    "passed_any_strategy": 1,
                    "passed_slots_json": "[]",
                    "md_volume_30d": 30_000_000.0,
                    "adj_close": 100.0,
                },
            ]
        )

        passed_only = filter_eligible_universe(frame, eligible_universe_mode="passed_only")

        self.assertEqual(passed_only["ticker"].tolist(), ["AAA"])

    def test_model_reason_summary_uses_relative_language(self) -> None:
        service = ShortlistModelService(db_manager=object())

        reasons = service._top_reason_names(
            {
                "roc_63__rank_all": 0.94,
                "relative_strength_index_vs_spy": 1.8,
                "sma_200_dist": 0.7,
            }
        )

        self.assertIn("top-tier 63d momentum", reasons)
        self.assertIn("strong RS vs SPY", reasons)
        self.assertIn("well above 200d trend", reasons)
        self.assertEqual(
            service._format_reason_summary(reasons),
            "strong RS vs SPY, top-tier 63d momentum, well above 200d trend",
        )

    def test_shortlist_model_writes_walk_forward_report(self) -> None:
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

            dates = pd.bdate_range("2026-01-02", periods=32)
            tickers = [
                ("AAA", "Energy"),
                ("BBB", "Materials"),
                ("CCC", "Industrials"),
                ("DDD", "Information Technology"),
            ]
            rows: list[dict[str, object]] = []
            for date_index, snapshot_date in enumerate(dates):
                for ticker_index, (ticker, sector) in enumerate(tickers):
                    row = {
                        "snapshot_date": snapshot_date.strftime("%Y-%m-%d"),
                        "ticker": ticker,
                        "sector": sector,
                        "passed_any_strategy": 1,
                        "md_volume_30d": 60_000_000.0,
                        "adj_close": 100.0 + ticker_index,
                        "alpha_vs_sector_20d": None if date_index == len(dates) - 1 else 0.02 * (ticker_index + 1) + 0.001 * date_index,
                    }
                    for feature_index, column in enumerate(MODEL_FEATURE_COLUMNS):
                        row[column] = float((feature_index + 1) * 0.05 + ticker_index + date_index * 0.03)
                    rows.append(row)
            snapshot_frame = pd.DataFrame(rows)

            class FakeDB:
                def __init__(self, paths, snapshot_frame):
                    self.paths = paths
                    self._snapshot_frame = snapshot_frame
                    self.run_rows: list[dict[str, object]] = []
                    self.prediction_rows: list[dict[str, object]] = []

                def initialize(self): return None
                def load_universe_daily_snapshots(self, snapshot_date=None):
                    return self._snapshot_frame.copy()
                def insert_shortlist_model_run(self, *, row):
                    self.run_rows.append(dict(row))
                    return len(self.run_rows)
                def replace_shortlist_model_predictions(
                    self,
                    *,
                    generated_at,
                    horizon_days,
                    eligible_universe_mode="passed_only",
                    model_scope="global",
                    rows,
                ):
                    self.prediction_rows = [dict(row) for row in rows]
                    return len(self.prediction_rows)

            fake_db = FakeDB(paths, snapshot_frame)
            service = ShortlistModelService(fake_db)
            report = service.run(
                top_n=2,
                horizon_days=20,
                min_train_dates=6,
                test_window_dates=2,
                recent_dates=4,
                xgboost_config="balanced_depth4",
            )

            self.assertEqual(report.target_column, "alpha_vs_sector_20d")
            self.assertGreater(report.oos_dates, 0)
            self.assertGreater(report.live_candidates, 0)

            report_text = (paths.reports_dir / "shortlist_model.md").read_text(encoding="utf-8")
            self.assertIn("# Shortlist Model", report_text)
            self.assertIn("- eligible_universe_mode: passed_only", report_text)
            self.assertIn("- candidate_models:", report_text)
            self.assertIn("- selected_model:", report_text)
            self.assertIn("## Promotion Gate", report_text)
            self.assertIn("## Full Walk-Forward Evaluation", report_text)
            self.assertIn("## Live Top Candidates", report_text)
            self.assertIn("### signal_proxy", report_text)
            self.assertIn("### lasso_model", report_text)

            self.assertTrue((paths.reports_dir / "shortlist_model_oos_predictions.csv").exists())
            self.assertTrue((paths.reports_dir / "shortlist_model_live_predictions.csv").exists())
            self.assertEqual(len(fake_db.run_rows), 1)
            self.assertGreater(len(fake_db.prediction_rows), 0)

    def test_shortlist_model_parser_accepts_args(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "shortlist-model",
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
                "--xgboost-config",
                "balanced_depth4",
            ]
        )
        self.assertEqual(args.command, "shortlist-model")
        self.assertEqual(args.top, 8)
        self.assertEqual(args.horizon, 20)
        self.assertEqual(args.min_train_dates, 200)
        self.assertEqual(args.test_window_dates, 15)
        self.assertEqual(args.recent_dates, 30)
        self.assertEqual(args.eligible_universe_mode, "passed_or_trend")
        self.assertEqual(args.model_scope, "sector_specific")
        self.assertEqual(args.xgboost_config, "balanced_depth4")

    def test_runtime_loader_returns_lasso_model_context(self) -> None:
        captured: dict[str, object] = {}

        class FakeDB:
            def load_shortlist_model_runs(self, *, horizon_days, eligible_universe_mode=None, model_scope=None, limit=1):
                captured["runs_model_scope"] = model_scope
                return pd.DataFrame(
                    [
                        {
                            "generated_at": "2026-05-26T17:00:00+00:00",
                            "champion_model": "lasso_model",
                            "live_snapshot_date": "2026-05-19",
                        }
                    ]
                )

            def list_universe_daily_snapshot_dates(self):
                return ["2026-05-19"]

            def load_shortlist_model_predictions(self, *, generated_at, horizon_days, eligible_universe_mode=None, model_scope=None, dataset_split, model_name):
                captured.setdefault("prediction_model_scopes", []).append(model_scope)
                if model_name != "lasso_model":
                    return pd.DataFrame()
                if dataset_split == "oos":
                    return pd.DataFrame(
                        [
                            {
                                "snapshot_date": "2026-05-18",
                                "ticker": "AAA",
                                "sector": "Energy",
                                "md_volume_30d": 50_000_000.0,
                                "predicted_alpha": 0.11,
                                "actual_alpha_vs_sector": 0.04,
                            },
                            {
                                "snapshot_date": "2026-05-18",
                                "ticker": "BBB",
                                "sector": "Energy",
                                "md_volume_30d": 40_000_000.0,
                                "predicted_alpha": 0.08,
                                "actual_alpha_vs_sector": 0.01,
                            },
                            {
                                "snapshot_date": "2026-05-18",
                                "ticker": "CCC",
                                "sector": "Energy",
                                "md_volume_30d": 30_000_000.0,
                                "predicted_alpha": 0.02,
                                "actual_alpha_vs_sector": -0.04,
                            },
                        ]
                    )
                return pd.DataFrame(
                    [
                        {
                            "snapshot_date": "2026-05-19",
                            "ticker": "AAA",
                            "sector": "Energy",
                            "md_volume_30d": 50_000_000.0,
                            "predicted_alpha": 0.11,
                            "details_json": '{"model_top_reasons": ["strong 63d momentum", "strong RS vs SPY"], "model_reason_summary": "strong 63d momentum, strong RS vs SPY"}',
                        },
                        {
                            "snapshot_date": "2026-05-19",
                            "ticker": "BBB",
                            "sector": "Energy",
                            "md_volume_30d": 40_000_000.0,
                            "predicted_alpha": 0.08,
                            "details_json": '{"model_top_reasons": ["strong RS vs SPY"], "model_reason_summary": "strong RS vs SPY"}',
                        }
                    ]
                )

        context = load_live_shortlist_model_context(
            FakeDB(),
            top_n=1,
            refresh_if_stale=False,
            eligible_universe_mode="passed_only",
            model_scope="sector_specific",
        )

        self.assertIsNotNone(context)
        assert context is not None
        self.assertEqual(captured["runs_model_scope"], "sector_specific")
        self.assertEqual(context.champion_model, "lasso_model")
        self.assertEqual(context.live_predictions.iloc[0]["ticker"], "AAA")
        self.assertEqual(
            context.live_predictions.iloc[0]["model_comparison_summary"],
            "BBB in Energy on strong 63d momentum",
        )

    def test_walk_forward_predictions_use_sparse_horizon_spaced_oos_dates(self) -> None:
        service = ShortlistModelService(db_manager=object())
        dates = pd.bdate_range("2026-01-02", periods=30)
        rows = []
        for date_index, snapshot_date in enumerate(dates):
            for ticker in ("AAA", "BBB"):
                rows.append(
                    {
                        "snapshot_date": snapshot_date,
                        "ticker": ticker,
                        "sector": "Energy",
                        "md_volume_30d": 50_000_000.0,
                        "relative_strength_index_vs_spy": 70.0 + date_index,
                        "roc_63": 0.1,
                        "sma_200_dist": 0.1,
                        "vol_alpha": 1.0,
                        "alpha_vs_sector_20d": 0.01,
                    }
                )
        frame = pd.DataFrame(rows)

        predictions = service._walk_forward_predictions(
            frame,
            target_column="alpha_vs_sector_20d",
            model_name="signal_proxy",
            min_train_dates=5,
            test_window_dates=2,
            model_scope="global",
            evaluation_stride_dates=10,
        )

        self.assertIsNotNone(predictions)
        assert predictions is not None
        self.assertEqual(len(predictions["snapshot_date"].drop_duplicates()), 3)

    def test_walk_forward_predictions_embargo_overlapping_training_labels(self) -> None:
        service = ShortlistModelService(db_manager=object())
        dates = pd.bdate_range("2026-01-02", periods=18)
        rows = []
        for date_index, snapshot_date in enumerate(dates):
            for ticker in ("AAA", "BBB"):
                rows.append(
                    {
                        "snapshot_date": snapshot_date,
                        "ticker": ticker,
                        "sector": "Energy",
                        "md_volume_30d": 50_000_000.0,
                        "relative_strength_index_vs_spy": 70.0 + date_index,
                        "roc_63": 0.1,
                        "sma_200_dist": 0.1,
                        "vol_alpha": 1.0,
                        "alpha_vs_sector_20d": 0.01,
                    }
                )
        frame = pd.DataFrame(rows)
        seen_folds: list[tuple[pd.Timestamp, pd.Timestamp]] = []

        def fake_score_model(**kwargs):
            train_frame = kwargs["train_frame"]
            test_frame = kwargs["test_frame"]
            seen_folds.append(
                (
                    pd.to_datetime(train_frame["snapshot_date"]).max(),
                    pd.to_datetime(test_frame["snapshot_date"]).min(),
                )
            )
            scored = test_frame.copy()
            scored["predicted_alpha"] = 0.0
            scored["model_top_reasons"] = [[] for _ in range(len(scored.index))]
            scored["model_reason_summary"] = None
            return scored

        with patch.object(service, "_score_model", side_effect=fake_score_model):
            predictions = service._walk_forward_predictions(
                frame,
                target_column="alpha_vs_sector_20d",
                model_name="ridge_model",
                min_train_dates=5,
                test_window_dates=2,
                model_scope="global",
                evaluation_stride_dates=5,
                label_horizon_dates=5,
            )

        self.assertIsNotNone(predictions)
        self.assertTrue(seen_folds)
        for train_end, test_start in seen_folds:
            self.assertLessEqual(
                dates.get_loc(train_end),
                dates.get_loc(test_start) - 5 - 1,
            )

    def test_shortlist_model_writes_failure_report_when_no_candidate_passes_gate(self) -> None:
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
            dates = pd.bdate_range("2026-01-02", periods=30)
            rows = []
            for date_index, snapshot_date in enumerate(dates):
                for ticker_index, ticker in enumerate(("AAA", "BBB")):
                    row = {
                        "snapshot_date": snapshot_date.strftime("%Y-%m-%d"),
                        "ticker": ticker,
                        "sector": "Energy",
                        "passed_any_strategy": 1,
                        "passed_slots_json": '["energy"]',
                        "md_volume_30d": 50_000_000.0,
                        "adj_close": 100.0,
                        "alpha_vs_sector_20d": 0.01 * (ticker_index + 1),
                    }
                    for feature_index, column in enumerate(MODEL_FEATURE_COLUMNS):
                        row[column] = float(feature_index + ticker_index + date_index)
                    rows.append(row)

            class FakeDB:
                def __init__(self, paths, snapshot_frame):
                    self.paths = paths
                    self._snapshot_frame = snapshot_frame

                def initialize(self): return None
                def load_universe_daily_snapshots(self, snapshot_date=None):
                    return self._snapshot_frame.copy()

            gate = {
                "scan_policy": {
                    "shortlist_model": {
                        "promotion_gate": {
                            "enabled": True,
                            "min_recent_20d_hit_rate": 1.01,
                            "min_recent_20d_beat_universe_rate": 1.01,
                            "min_recent_20d_mean_target": 1.01,
                            "min_recent_60d_hit_rate": 1.01,
                            "min_recent_60d_beat_universe_rate": 1.01,
                            "min_recent_60d_mean_target": 1.01,
                        }
                    }
                }
            }
            service = ShortlistModelService(FakeDB(paths, pd.DataFrame(rows)))

            with patch("src.research.shortlist_model_service.load_feature_config", return_value=gate), \
                 patch.object(service, "_score_xgboost_model", return_value=None), \
                 self.assertRaisesRegex(ValueError, "No shortlist model candidate passed"):
                service.run(top_n=1, min_train_dates=4, test_window_dates=2)

            report_text = (paths.reports_dir / "shortlist_model.md").read_text(encoding="utf-8")
            self.assertIn("## Promotion Failure", report_text)
            self.assertIn("- selected_model: n/a", report_text)
            self.assertIn("- oos_evaluation_stride_dates: 20", report_text)
            self.assertTrue((paths.reports_dir / "shortlist_model_oos_predictions.csv").exists())

    def test_runtime_loader_does_not_refresh_stale_model_without_explicit_permission(self) -> None:
        class FakeDB:
            def __init__(self):
                self.refresh_count = 0

            def load_shortlist_model_runs(self, *, horizon_days, eligible_universe_mode=None, model_scope=None, xgboost_config="baseline", limit=1):
                return pd.DataFrame(
                    [
                        {
                            "generated_at": "2026-05-26T17:00:00+00:00",
                            "champion_model": "lasso_model",
                            "live_snapshot_date": "2026-05-18",
                        }
                    ]
                )

            def list_universe_daily_snapshot_dates(self):
                return ["2026-05-19"]

            def load_shortlist_model_predictions(self, **kwargs):
                raise AssertionError("stale runtime loader should not read predictions after declining refresh")

            def load_universe_daily_snapshots(self, snapshot_date=None):
                self.refresh_count += 1
                raise AssertionError("runtime loader should not retrain without allow_refresh=True")

        fake_db = FakeDB()
        context = load_live_shortlist_model_context(
            fake_db,
            refresh_if_stale=True,
            allow_refresh=False,
            eligible_universe_mode="passed_only",
            model_scope="sector_specific",
        )

        self.assertIsNone(context)
        self.assertEqual(fake_db.refresh_count, 0)

    def test_runtime_loader_rejects_champion_with_negative_latest_fold(self) -> None:
        class FakeDB:
            def load_shortlist_model_runs(self, *, horizon_days, eligible_universe_mode=None, model_scope=None, xgboost_config="baseline", limit=1):
                return pd.DataFrame(
                    [
                        {
                            "generated_at": "2026-05-26T17:00:00+00:00",
                            "champion_model": "lasso_model",
                            "live_snapshot_date": "2026-05-22",
                        }
                    ]
                )

            def list_universe_daily_snapshot_dates(self):
                return ["2026-05-22"]

            def load_shortlist_model_predictions(self, *, dataset_split, model_name, **kwargs):
                if model_name != "lasso_model":
                    return pd.DataFrame()
                if dataset_split == "live":
                    return pd.DataFrame(
                        [
                            {
                                "snapshot_date": "2026-05-22",
                                "ticker": "AAA",
                                "sector": "Energy",
                                "md_volume_30d": 50_000_000.0,
                                "predicted_alpha": 0.11,
                            }
                        ]
                    )
                rows = []
                for snapshot_date, targets in [
                    ("2026-05-18", (0.04, 0.03, -0.01)),
                    ("2026-05-19", (0.05, 0.02, -0.01)),
                    ("2026-05-20", (0.04, 0.01, -0.01)),
                    ("2026-05-21", (-0.03, -0.02, -0.04)),
                ]:
                    for ticker, predicted_alpha, target in zip(("AAA", "BBB", "CCC"), (0.11, 0.08, 0.01), targets):
                        rows.append(
                            {
                                "snapshot_date": snapshot_date,
                                "ticker": ticker,
                                "sector": "Energy",
                                "md_volume_30d": 50_000_000.0,
                                "predicted_alpha": predicted_alpha,
                                "actual_alpha_vs_sector": target,
                            }
                        )
                return pd.DataFrame(rows)

        context = load_live_shortlist_model_context(
            FakeDB(),
            top_n=2,
            refresh_if_stale=False,
            eligible_universe_mode="passed_only",
            model_scope="sector_specific",
        )

        self.assertIsNone(context)

    def test_champion_selection_refuses_models_that_fail_promotion_gate(self) -> None:
        service = ShortlistModelService(db_manager=object())
        full_summaries = pd.DataFrame(
            [
                {"model": "xgboost_model", "mean_target": 0.08, "beat_universe_rate": 0.60, "positive_date_rate": 0.70},
                {"model": "lasso_model", "mean_target": 0.04, "beat_universe_rate": 0.55, "positive_date_rate": 0.60},
            ]
        )
        acceptance_summaries = pd.DataFrame(
            [
                {"model": "xgboost_model_20d", "hit_rate": 0.45, "beat_universe_rate": 0.40, "mean_target": -0.01},
                {"model": "xgboost_model_60d", "hit_rate": 0.55, "beat_universe_rate": 0.55, "mean_target": 0.04},
                {"model": "lasso_model_20d", "hit_rate": 0.40, "beat_universe_rate": 0.35, "mean_target": -0.02},
                {"model": "lasso_model_60d", "hit_rate": 0.52, "beat_universe_rate": 0.52, "mean_target": 0.01},
            ]
        )
        promotion_gate = {
            "enabled": True,
            "min_recent_20d_hit_rate": 0.50,
            "min_recent_20d_beat_universe_rate": 0.50,
            "min_recent_20d_mean_target": 0.0,
            "min_recent_60d_hit_rate": 0.50,
            "min_recent_60d_beat_universe_rate": 0.50,
            "min_recent_60d_mean_target": 0.0,
        }

        with self.assertRaisesRegex(ValueError, "No shortlist model candidate passed the promotion gate"):
            service._choose_champion_model(
                full_summaries=full_summaries,
                acceptance_summaries=acceptance_summaries,
                promotion_gate=promotion_gate,
            )

    def test_champion_selection_refuses_models_with_negative_latest_fold(self) -> None:
        service = ShortlistModelService(db_manager=object())
        full_summaries = pd.DataFrame(
            [
                {"model": "lasso_model", "mean_target": 0.08, "beat_universe_rate": 0.80, "positive_date_rate": 0.80},
            ]
        )
        acceptance_summaries = pd.DataFrame(
            [
                {"model": "lasso_model_20d", "hit_rate": 0.85, "beat_universe_rate": 0.85, "mean_target": 0.04},
                {"model": "lasso_model_60d", "hit_rate": 0.85, "beat_universe_rate": 0.85, "mean_target": 0.04},
                {"model": "lasso_model_last_1fold", "hit_rate": 0.0, "beat_universe_rate": 0.0, "mean_target": -0.03},
                {"model": "lasso_model_last_3fold", "hit_rate": 0.67, "beat_universe_rate": 0.67, "mean_target": 0.01},
            ]
        )
        promotion_gate = {
            "enabled": True,
            "min_recent_20d_hit_rate": 0.50,
            "min_recent_20d_beat_universe_rate": 0.50,
            "min_recent_20d_mean_target": 0.0,
            "min_recent_60d_hit_rate": 0.50,
            "min_recent_60d_beat_universe_rate": 0.50,
            "min_recent_60d_mean_target": 0.0,
            "min_recent_1fold_hit_rate": 0.50,
            "min_recent_1fold_beat_universe_rate": 0.50,
            "min_recent_1fold_mean_target": 0.0,
            "min_recent_3fold_hit_rate": 0.50,
            "min_recent_3fold_beat_universe_rate": 0.50,
            "min_recent_3fold_mean_target": 0.0,
        }

        with self.assertRaisesRegex(ValueError, "No shortlist model candidate passed the promotion gate"):
            service._choose_champion_model(
                full_summaries=full_summaries,
                acceptance_summaries=acceptance_summaries,
                promotion_gate=promotion_gate,
            )

    def test_classification_target_preserves_unmatured_alpha_as_missing(self) -> None:
        service = ShortlistModelService(db_manager=object())
        frame = pd.DataFrame(
            [
                {
                    "snapshot_date": "2026-01-02",
                    "ticker": "AAA",
                    "sector": "Energy",
                    "passed_any_strategy": 1,
                    "md_volume_30d": 50_000_000.0,
                    "adj_close": 50.0,
                    "alpha_vs_sector_20d_pos": 1.0,
                },
                {
                    "snapshot_date": "2026-01-03",
                    "ticker": "BBB",
                    "sector": "Energy",
                    "passed_any_strategy": 1,
                    "md_volume_30d": 50_000_000.0,
                    "adj_close": 50.0,
                    "alpha_vs_sector_20d_pos": float("nan"),
                },
            ]
        )

        prepared = service._prepare_snapshot_frame(frame)
        matured = service._build_matured_eligible_universe(
            prepared,
            target_column="alpha_vs_sector_20d_pos",
            eligible_universe_mode="passed_only",
        )

        self.assertEqual(matured["ticker"].tolist(), ["AAA"])
