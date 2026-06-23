from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

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

            dates = pd.bdate_range("2026-01-02", periods=12)
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
            self.assertIn("- xgboost_config: balanced_depth4", report_text)
            self.assertIn("## Full Walk-Forward Evaluation", report_text)
            self.assertIn("## Live Top Candidates", report_text)
            self.assertIn("### signal_proxy", report_text)
            self.assertIn("### ensemble_model", report_text)

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

    def test_runtime_loader_can_prefer_non_champion_live_model(self) -> None:
        captured: dict[str, object] = {}

        class FakeDB:
            def load_shortlist_model_runs(self, *, horizon_days, eligible_universe_mode=None, model_scope=None, xgboost_config=None, limit=1):
                captured["runs_model_scope"] = model_scope
                captured["runs_xgboost_config"] = xgboost_config
                return pd.DataFrame(
                    [
                        {
                            "generated_at": "2026-05-26T17:00:00+00:00",
                            "champion_model": "ensemble_model",
                            "live_snapshot_date": "2026-05-19",
                        }
                    ]
                )

            def list_universe_daily_snapshot_dates(self):
                return ["2026-05-19"]

            def load_shortlist_model_predictions(self, *, generated_at, horizon_days, eligible_universe_mode=None, model_scope=None, dataset_split, model_name):
                captured.setdefault("prediction_model_scopes", []).append(model_scope)
                rows = {
                    "xgboost_model": [
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
                    ],
                    "ensemble_model": [
                        {
                            "snapshot_date": "2026-05-19",
                            "ticker": "BBB",
                            "sector": "Materials",
                            "md_volume_30d": 60_000_000.0,
                            "predicted_alpha": 0.09,
                            "details_json": '{"model_top_reasons": ["strong volume confirmation"], "model_reason_summary": "strong volume confirmation"}',
                        }
                    ],
                }
                return pd.DataFrame(rows.get(model_name, []))

        context = load_live_shortlist_model_context(
            FakeDB(),
            top_n=1,
            preferred_model_name="xgboost_model",
            refresh_if_stale=False,
            eligible_universe_mode="passed_only",
            model_scope="sector_specific",
            xgboost_config="balanced_depth4",
        )

        self.assertIsNotNone(context)
        assert context is not None
        self.assertEqual(captured["runs_model_scope"], "sector_specific")
        self.assertEqual(captured["runs_xgboost_config"], "balanced_depth4")
        self.assertTrue(all(scope == "sector_specific" for scope in captured["prediction_model_scopes"]))
        self.assertEqual(context.champion_model, "xgboost_model")
        self.assertEqual(context.live_predictions.iloc[0]["ticker"], "AAA")
        self.assertEqual(
            context.live_predictions.iloc[0]["model_comparison_summary"],
            "BBB in Energy on strong 63d momentum",
        )
