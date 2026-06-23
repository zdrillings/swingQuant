from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.cli import build_parser
from src.research.shortlist_promote_service import ShortlistPromoteService
from src.settings import AppPaths, RuntimeSettings


class ShortlistPromoteServiceTests(unittest.TestCase):
    def test_shortlist_promote_updates_production_shortlist_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = root / "config.yaml"
            config_path.write_text(
                "scan_policy:\n"
                "  shortlist_model:\n"
                "    production_model_name: xgboost_model\n"
                "    production_eligible_universe_mode: passed_only\n"
                "    production_model_scope: global\n"
                "    production_xgboost_config: baseline\n",
                encoding="utf-8",
            )
            settings = RuntimeSettings(
                paths=AppPaths(
                    root_dir=root,
                    data_dir=root / "data",
                    duckdb_path=root / "data" / "market_data.duckdb",
                    sqlite_path=root / "data" / "ledger.sqlite",
                    reports_dir=root / "reports",
                    logs_dir=root / "logs",
                    config_path=config_path,
                    env_path=root / ".env",
                    production_strategy_path=root / "production_strategy.json",
                    production_strategies_path=root / "production_strategies.json",
                ),
                env={},
                total_capital=None,
                risk_per_trade=None,
            )

            class FakeDB:
                def initialize(self): return None

                def load_shortlist_model_runs(self, *, horizon_days=None, eligible_universe_mode=None, model_scope=None, xgboost_config=None, limit=None):
                    return __import__("pandas").DataFrame(
                        [
                            {
                                "generated_at": "2026-05-26T22:32:40+00:00",
                                "champion_model": "xgboost_model",
                            }
                        ]
                    )

                def load_shortlist_model_predictions(
                    self,
                    *,
                    generated_at=None,
                    horizon_days=None,
                    eligible_universe_mode=None,
                    model_scope=None,
                    dataset_split=None,
                    model_name=None,
                ):
                    return __import__("pandas").DataFrame(
                        [
                            {
                                "ticker": "APA",
                                "dataset_split": "live",
                                "model_name": model_name,
                            }
                        ]
                    )

            with patch("src.research.shortlist_promote_service.get_settings", return_value=settings):
                report = ShortlistPromoteService(FakeDB()).run(
                    model_name="xgboost_model",
                    eligible_universe_mode="passed_or_trend",
                    model_scope="sector_specific",
                    xgboost_config="balanced_depth4",
                    horizon_days=20,
                )

            self.assertEqual(report.production_model_name, "xgboost_model")
            self.assertEqual(report.production_eligible_universe_mode, "passed_or_trend")
            self.assertEqual(report.production_model_scope, "sector_specific")
            self.assertEqual(report.production_xgboost_config, "balanced_depth4")
            updated_text = config_path.read_text(encoding="utf-8")
            self.assertIn("production_model_name: xgboost_model", updated_text)
            self.assertIn("production_eligible_universe_mode: passed_or_trend", updated_text)
            self.assertIn("production_model_scope: sector_specific", updated_text)
            self.assertIn("production_xgboost_config: balanced_depth4", updated_text)

    def test_shortlist_promote_parser_accepts_args(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "shortlist-promote",
                "--model-name",
                "xgboost_model",
                "--eligible-universe-mode",
                "passed_or_trend",
                "--model-scope",
                "sector_specific",
                "--xgboost-config",
                "balanced_depth4",
                "--horizon",
                "20",
            ]
        )
        self.assertEqual(args.command, "shortlist-promote")
        self.assertEqual(args.model_name, "xgboost_model")
        self.assertEqual(args.eligible_universe_mode, "passed_or_trend")
        self.assertEqual(args.model_scope, "sector_specific")
        self.assertEqual(args.xgboost_config, "balanced_depth4")
        self.assertEqual(args.horizon, 20)
