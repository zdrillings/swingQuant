from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src.research.universe_analysis_service import UniverseAnalysisService
from src.settings import AppPaths


class UniverseAnalysisServiceTests(unittest.TestCase):
    def test_universe_analysis_writes_missed_winner_report(self) -> None:
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

            snapshot_frame = pd.DataFrame(
                [
                    {
                        "snapshot_date": "2026-04-01",
                        "ticker": "MAT1",
                        "sector": "Materials",
                        "passed_any_strategy": 1,
                        "strategy_pass_count": 1,
                        "relative_strength_index_vs_spy": 85.0,
                        "roc_63": 0.12,
                        "vol_alpha": 1.20,
                        "sma_200_dist": 0.10,
                        "sma_50_dist": 0.06,
                        "rsi_14": 48.0,
                        "sector_pct_above_50": 0.80,
                        "sector_pct_above_200": 0.78,
                        "sector_median_roc_63": 0.09,
                        "fwd_return_10d": 0.08,
                        "alpha_vs_sector_10d": 0.05,
                    },
                    {
                        "snapshot_date": "2026-04-01",
                        "ticker": "TECH1",
                        "sector": "Information Technology",
                        "passed_any_strategy": 0,
                        "strategy_pass_count": 0,
                        "relative_strength_index_vs_spy": 78.0,
                        "roc_63": 0.10,
                        "vol_alpha": 1.10,
                        "sma_200_dist": 0.08,
                        "sma_50_dist": 0.05,
                        "rsi_14": 52.0,
                        "sector_pct_above_50": 0.70,
                        "sector_pct_above_200": 0.68,
                        "sector_median_roc_63": 0.07,
                        "fwd_return_10d": 0.14,
                        "alpha_vs_sector_10d": 0.10,
                    },
                    {
                        "snapshot_date": "2026-04-01",
                        "ticker": "HC1",
                        "sector": "Health Care",
                        "passed_any_strategy": 0,
                        "strategy_pass_count": 0,
                        "relative_strength_index_vs_spy": 60.0,
                        "roc_63": 0.01,
                        "vol_alpha": 0.90,
                        "sma_200_dist": 0.01,
                        "sma_50_dist": 0.00,
                        "rsi_14": 58.0,
                        "sector_pct_above_50": 0.45,
                        "sector_pct_above_200": 0.40,
                        "sector_median_roc_63": 0.02,
                        "fwd_return_10d": -0.01,
                        "alpha_vs_sector_10d": -0.02,
                    },
                    {
                        "snapshot_date": "2026-04-01",
                        "ticker": "IND1",
                        "sector": "Industrials",
                        "passed_any_strategy": 1,
                        "strategy_pass_count": 1,
                        "relative_strength_index_vs_spy": 80.0,
                        "roc_63": 0.08,
                        "vol_alpha": 1.00,
                        "sma_200_dist": 0.07,
                        "sma_50_dist": 0.04,
                        "rsi_14": 50.0,
                        "sector_pct_above_50": 0.72,
                        "sector_pct_above_200": 0.70,
                        "sector_median_roc_63": 0.06,
                        "fwd_return_10d": 0.04,
                        "alpha_vs_sector_10d": 0.02,
                    },
                    {
                        "snapshot_date": "2026-04-02",
                        "ticker": "MAT2",
                        "sector": "Materials",
                        "passed_any_strategy": 1,
                        "strategy_pass_count": 1,
                        "relative_strength_index_vs_spy": 84.0,
                        "roc_63": 0.11,
                        "vol_alpha": 1.18,
                        "sma_200_dist": 0.09,
                        "sma_50_dist": 0.05,
                        "rsi_14": 47.0,
                        "sector_pct_above_50": 0.79,
                        "sector_pct_above_200": 0.77,
                        "sector_median_roc_63": 0.08,
                        "fwd_return_10d": 0.07,
                        "alpha_vs_sector_10d": 0.04,
                    },
                    {
                        "snapshot_date": "2026-04-02",
                        "ticker": "TECH2",
                        "sector": "Information Technology",
                        "passed_any_strategy": 0,
                        "strategy_pass_count": 0,
                        "relative_strength_index_vs_spy": 79.0,
                        "roc_63": 0.11,
                        "vol_alpha": 1.12,
                        "sma_200_dist": 0.09,
                        "sma_50_dist": 0.05,
                        "rsi_14": 51.0,
                        "sector_pct_above_50": 0.71,
                        "sector_pct_above_200": 0.69,
                        "sector_median_roc_63": 0.07,
                        "fwd_return_10d": 0.12,
                        "alpha_vs_sector_10d": 0.09,
                    },
                    {
                        "snapshot_date": "2026-04-02",
                        "ticker": "HC2",
                        "sector": "Health Care",
                        "passed_any_strategy": 0,
                        "strategy_pass_count": 0,
                        "relative_strength_index_vs_spy": 59.0,
                        "roc_63": 0.00,
                        "vol_alpha": 0.88,
                        "sma_200_dist": 0.01,
                        "sma_50_dist": 0.00,
                        "rsi_14": 57.0,
                        "sector_pct_above_50": 0.44,
                        "sector_pct_above_200": 0.41,
                        "sector_median_roc_63": 0.02,
                        "fwd_return_10d": 0.00,
                        "alpha_vs_sector_10d": -0.01,
                    },
                    {
                        "snapshot_date": "2026-04-02",
                        "ticker": "IND2",
                        "sector": "Industrials",
                        "passed_any_strategy": 1,
                        "strategy_pass_count": 1,
                        "relative_strength_index_vs_spy": 81.0,
                        "roc_63": 0.09,
                        "vol_alpha": 1.01,
                        "sma_200_dist": 0.08,
                        "sma_50_dist": 0.04,
                        "rsi_14": 49.0,
                        "sector_pct_above_50": 0.73,
                        "sector_pct_above_200": 0.71,
                        "sector_median_roc_63": 0.06,
                        "fwd_return_10d": 0.03,
                        "alpha_vs_sector_10d": 0.01,
                    },
                ]
            )

            class FakeDB:
                def __init__(self, paths, snapshot_frame):
                    self.paths = paths
                    self._snapshot_frame = snapshot_frame

                def initialize(self): return None
                def load_universe_daily_snapshots(self, snapshot_date=None):
                    return self._snapshot_frame.copy()

            service = UniverseAnalysisService(FakeDB(paths, snapshot_frame))
            report = service.run(top=5, horizon_days=10, recent_dates=5)

            self.assertEqual(report.snapshot_rows, len(snapshot_frame.index))
            report_text = (paths.reports_dir / "universe_analysis.md").read_text(encoding="utf-8")
            self.assertIn("# Universe Analysis", report_text)
            self.assertIn("## Gate Outcome Summary", report_text)
            self.assertIn("## Sector Pass Rates", report_text)
            self.assertIn("## Missed Winner Sectors", report_text)
            self.assertIn("### Information Technology", report_text)
            self.assertIn("## Recent Missed Winners", report_text)
            self.assertIn("### TECH1", report_text)
            self.assertIn("## Feature Profile Comparison", report_text)
