from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from src.cli import build_parser
from src.research.shortlist_sector_reactivation_service import ShortlistSectorReactivationService
from src.settings import AppPaths
from src.utils.strategy import ExitRules, ProductionStrategy


class ShortlistSectorReactivationServiceTests(unittest.TestCase):
    def test_shortlist_sector_reactivation_writes_report(self) -> None:
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

                def load_shortlist_model_runs(self, *, horizon_days=None, eligible_universe_mode=None, model_scope=None, xgboost_config=None, limit=None):
                    return pd.DataFrame(
                        [
                            {
                                "generated_at": "2026-05-28T16:52:57+00:00",
                                "champion_model": "xgboost_model",
                                "live_snapshot_date": "2026-05-19",
                            }
                        ]
                    )

                def list_universe_daily_snapshot_dates(self):
                    return ["2026-05-19"]

                def load_shortlist_model_predictions(self, *, generated_at=None, horizon_days=None, eligible_universe_mode=None, model_scope=None, dataset_split=None, model_name=None):
                    if dataset_split == "live":
                        return pd.DataFrame(
                            [
                                {"snapshot_date": "2026-05-19", "ticker": "AAA", "sector": "Energy", "predicted_alpha": 0.18, "md_volume_30d": 40_000_000},
                                {"snapshot_date": "2026-05-19", "ticker": "BBB", "sector": "Materials", "predicted_alpha": 0.16, "md_volume_30d": 35_000_000},
                                {"snapshot_date": "2026-05-19", "ticker": "CCC", "sector": "Information Technology", "predicted_alpha": 0.22, "md_volume_30d": 50_000_000},
                                {"snapshot_date": "2026-05-19", "ticker": "DDD", "sector": "Industrials", "predicted_alpha": 0.14, "md_volume_30d": 30_000_000},
                            ]
                        )
                    rows = []
                    for snapshot_date, values in (
                        ("2026-05-01", [("AAA", "Energy", 0.18, 0.12), ("BBB", "Materials", 0.16, 0.10), ("CCC", "Information Technology", 0.22, 0.20), ("DDD", "Industrials", 0.14, 0.08)]),
                        ("2026-05-02", [("AAA", "Energy", 0.17, 0.11), ("BBB", "Materials", 0.15, 0.09), ("CCC", "Information Technology", 0.21, 0.18), ("DDD", "Industrials", 0.13, 0.07)]),
                    ):
                        for ticker, sector, predicted_alpha, actual_alpha in values:
                            rows.append(
                                {
                                    "snapshot_date": snapshot_date,
                                    "ticker": ticker,
                                    "sector": sector,
                                    "predicted_alpha": predicted_alpha,
                                    "actual_alpha_vs_sector": actual_alpha,
                                    "md_volume_30d": 30_000_000,
                                }
                            )
                    return pd.DataFrame(rows)

            config = {
                "scan_policy": {
                    "shortlist_model": {
                        "production_eligible_universe_mode": "passed_or_trend",
                        "production_model_scope": "sector_specific",
                        "production_model_name": "xgboost_model",
                        "production_xgboost_config": "balanced_depth4",
                    }
                }
            }
            active_strategies = {
                "energy": ProductionStrategy(1, "2026-05-01T00:00:00", {}, ExitRules(0.05, 0.12, 20), slot="energy", sector="Energy"),
                "materials": ProductionStrategy(2, "2026-05-01T00:00:00", {}, ExitRules(0.05, 0.12, 20), slot="materials", sector="Materials"),
                "industrials": ProductionStrategy(3, "2026-05-01T00:00:00", {}, ExitRules(0.05, 0.12, 20), slot="industrials", sector="Industrials"),
            }

            with patch("src.research.shortlist_sector_reactivation_service.load_feature_config", return_value=config), \
                 patch("src.research.shortlist_sector_reactivation_service.load_active_strategies", return_value=active_strategies):
                report = ShortlistSectorReactivationService(FakeDB()).run(refresh_if_stale=False)

            self.assertTrue(report.output_path.endswith("shortlist_sector_reactivation.md"))
            text = (reports_dir / "shortlist_sector_reactivation.md").read_text(encoding="utf-8")
            self.assertIn("# Shortlist Sector Reactivation Analysis", text)
            self.assertIn("## Step 1: Sector Admission Comparison", text)
            self.assertIn("baseline_plus_candidate", text)
            self.assertIn("Information Technology", text)
            self.assertIn("## Step 2: Expanded Set Allocation Balance", text)
            self.assertIn("## Step 3: Expanded Set Top-N Sensitivity", text)

    def test_shortlist_sector_reactivation_parser_accepts_args(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "shortlist-sector-reactivation",
                "--top",
                "8",
                "--candidate-sector",
                "Information Technology",
                "--candidate-sector",
                "Communication Services",
                "--xgboost-config",
                "balanced_depth4",
            ]
        )
        self.assertEqual(args.command, "shortlist-sector-reactivation")
        self.assertEqual(args.top, 8)
        self.assertEqual(args.candidate_sector, ["Information Technology", "Information Technology", "Communication Services"])
        self.assertEqual(args.xgboost_config, "balanced_depth4")
