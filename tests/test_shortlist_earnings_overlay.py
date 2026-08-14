from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from src.cli import build_parser
from src.research.shortlist_earnings_overlay_service import ShortlistEarningsOverlayService
from src.settings import AppPaths


class ShortlistEarningsOverlayServiceTests(unittest.TestCase):
    def test_shortlist_earnings_overlay_writes_report(self) -> None:
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

            predictions = pd.DataFrame(
                [
                    {
                        "snapshot_date": "2026-05-01",
                        "ticker": "AAA",
                        "sector": "Technology",
                        "predicted_alpha": 0.90,
                        "actual_alpha_vs_sector": 0.05,
                    },
                    {
                        "snapshot_date": "2026-05-01",
                        "ticker": "BBB",
                        "sector": "Technology",
                        "predicted_alpha": 0.80,
                        "actual_alpha_vs_sector": 0.01,
                    },
                    {
                        "snapshot_date": "2026-05-02",
                        "ticker": "AAA",
                        "sector": "Technology",
                        "predicted_alpha": 0.70,
                        "actual_alpha_vs_sector": 0.02,
                    },
                    {
                        "snapshot_date": "2026-05-02",
                        "ticker": "BBB",
                        "sector": "Technology",
                        "predicted_alpha": 0.60,
                        "actual_alpha_vs_sector": 0.08,
                    },
                ]
            )
            snapshots = pd.DataFrame(
                [
                    {
                        "snapshot_date": "2026-05-01",
                        "ticker": "AAA",
                        "last_earnings_gap_pct": 0.02,
                        "last_earnings_volume_ratio_20": 1.2,
                        "last_earnings_open_vs_20d_high": 0.01,
                        "close_vs_last_earnings_close": 0.03,
                        "days_since_last_earnings": 20.0,
                        "days_to_next_earnings": 30.0,
                    },
                    {
                        "snapshot_date": "2026-05-01",
                        "ticker": "BBB",
                        "last_earnings_gap_pct": 0.10,
                        "last_earnings_volume_ratio_20": 2.5,
                        "last_earnings_open_vs_20d_high": 0.05,
                        "close_vs_last_earnings_close": 0.08,
                        "days_since_last_earnings": 10.0,
                        "days_to_next_earnings": 30.0,
                    },
                    {
                        "snapshot_date": "2026-05-02",
                        "ticker": "AAA",
                        "last_earnings_gap_pct": 0.01,
                        "last_earnings_volume_ratio_20": 1.1,
                        "last_earnings_open_vs_20d_high": 0.00,
                        "close_vs_last_earnings_close": 0.01,
                        "days_since_last_earnings": 21.0,
                        "days_to_next_earnings": 29.0,
                    },
                    {
                        "snapshot_date": "2026-05-02",
                        "ticker": "BBB",
                        "last_earnings_gap_pct": 0.09,
                        "last_earnings_volume_ratio_20": 2.0,
                        "last_earnings_open_vs_20d_high": 0.04,
                        "close_vs_last_earnings_close": 0.07,
                        "days_since_last_earnings": 11.0,
                        "days_to_next_earnings": 29.0,
                    },
                ]
            )

            class FakeDB:
                def __init__(self, paths, predictions, snapshots):
                    self.paths = paths
                    self._predictions = predictions
                    self._snapshots = snapshots

                def initialize(self): return None
                def load_shortlist_model_runs(self, **kwargs):
                    return pd.DataFrame(
                        [
                            {
                                "generated_at": "2026-05-03T00:00:00+00:00",
                                "champion_model": "xgboost_model",
                            }
                        ]
                    )

                def load_shortlist_model_predictions(self, **kwargs):
                    return self._predictions.copy()

                def load_universe_daily_snapshots(self):
                    return self._snapshots.copy()

            config = {
                "scan_policy": {
                    "shortlist_model": {
                        "production_model_name": "xgboost_model",
                        "production_eligible_universe_mode": "passed_or_trend",
                        "production_model_scope": "sector_specific",
                        "production_xgboost_config": "balanced_depth4",
                    }
                }
            }
            with patch("src.research.shortlist_earnings_overlay_service.load_feature_config", return_value=config):
                report = ShortlistEarningsOverlayService(FakeDB(paths, predictions, snapshots)).run(
                    top_n=1,
                    horizon_days=20,
                    recent_dates=1,
                )

            self.assertEqual(report.generated_at, "2026-05-03T00:00:00+00:00")
            self.assertEqual(report.row_count, 4)
            self.assertEqual(report.date_count, 2)
            report_text = (paths.reports_dir / "earnings_confirmation_live_comparison.md").read_text(
                encoding="utf-8"
            )
            self.assertIn("# Earnings Confirmation Live Model Comparison", report_text)
            self.assertIn("## Best Earnings Overlay Deltas vs Live", report_text)
            self.assertIn("reaction_rank_30", report_text)
            self.assertIn("## Incremental Regression", report_text)

    def test_shortlist_earnings_overlay_parser_accepts_args(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "shortlist-earnings-overlay",
                "--top",
                "6",
                "--horizon",
                "20",
                "--recent-dates",
                "40",
                "--model-name",
                "xgboost_model",
                "--eligible-universe-mode",
                "passed_or_trend",
                "--model-scope",
                "sector_specific",
                "--xgboost-config",
                "balanced_depth4",
                "--generated-at",
                "2026-05-03T00:00:00+00:00",
            ]
        )
        self.assertEqual(args.command, "shortlist-earnings-overlay")
        self.assertEqual(args.top, 6)
        self.assertEqual(args.horizon, 20)
        self.assertEqual(args.recent_dates, 40)
        self.assertEqual(args.model_name, "xgboost_model")
        self.assertEqual(args.eligible_universe_mode, "passed_or_trend")
        self.assertEqual(args.model_scope, "sector_specific")
        self.assertEqual(args.xgboost_config, "balanced_depth4")
        self.assertEqual(args.generated_at, "2026-05-03T00:00:00+00:00")
