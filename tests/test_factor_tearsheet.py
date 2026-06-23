from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src.research.factor_tearsheet_service import FactorTearsheetService
from src.settings import AppPaths


class FactorTearsheetServiceTests(unittest.TestCase):
    def test_factor_tearsheet_writes_feature_sections(self) -> None:
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

            rows: list[dict[str, object]] = []
            for day_index, snapshot_date in enumerate(("2026-04-01", "2026-04-02", "2026-04-03")):
                for ticker_index in range(5):
                    rows.append(
                        {
                            "snapshot_date": snapshot_date,
                            "ticker": f"T{day_index}{ticker_index}",
                            "sector": "Information Technology",
                            "regime_green": ticker_index % 2 == 0,
                            "relative_strength_index_vs_spy": 60 + (ticker_index * 5) + day_index,
                            "relative_strength_index_vs_qqq": 58 + (ticker_index * 5) + day_index,
                            "relative_strength_index_vs_xlk": 57 + (ticker_index * 5) + day_index,
                            "relative_strength_index_vs_subindustry": 59 + (ticker_index * 5) + day_index,
                            "roc_63": 0.02 + (ticker_index * 0.03),
                            "roc_126": 0.04 + (ticker_index * 0.04),
                            "vol_alpha": 0.8 + (ticker_index * 0.15),
                            "sma_200_dist": 0.02 + (ticker_index * 0.03),
                            "sma_50_dist": 0.01 + (ticker_index * 0.02),
                            "rsi_14": 40 + (ticker_index * 4),
                            "atr_14": 1.0 + ticker_index,
                            "days_to_next_earnings": 5 + ticker_index,
                            "days_since_last_earnings": 1 + ticker_index,
                            "last_earnings_gap_pct": 0.01 + (ticker_index * 0.01),
                            "last_earnings_volume_ratio_20": 1.0 + (ticker_index * 0.1),
                            "last_earnings_open_vs_20d_high": -0.01 + (ticker_index * 0.01),
                            "close_vs_last_earnings_close": 0.00 + (ticker_index * 0.02),
                            "avg_abs_gap_pct_20": 0.01 + (ticker_index * 0.002),
                            "max_gap_down_pct_60": 0.01 + (ticker_index * 0.003),
                            "distance_above_20d_high": 0.00 + (ticker_index * 0.004),
                            "base_range_pct_20": 0.03 + (ticker_index * 0.005),
                            "base_atr_contraction_20": 0.80 - (ticker_index * 0.10),
                            "base_volume_dryup_ratio_20": 0.90 - (ticker_index * 0.10),
                            "breakout_volume_ratio_50": 0.80 + (ticker_index * 0.20),
                            "sector_pct_above_50": 0.30 + (ticker_index * 0.15),
                            "sector_pct_above_200": 0.25 + (ticker_index * 0.15),
                            "sector_median_roc_63": 0.01 + (ticker_index * 0.02),
                            "alpha_vs_sector_5d": -0.04 + (ticker_index * 0.03),
                            "alpha_vs_sector_10d": -0.05 + (ticker_index * 0.04),
                        }
                    )
            snapshot_frame = pd.DataFrame(rows)

            class FakeDB:
                def __init__(self, paths, snapshot_frame):
                    self.paths = paths
                    self._snapshot_frame = snapshot_frame

                def initialize(self): return None
                def load_universe_daily_snapshots(self, snapshot_date=None):
                    return self._snapshot_frame.copy()

            service = FactorTearsheetService(FakeDB(paths, snapshot_frame))
            report = service.run(sector="Information Technology", horizon_days=10)

            self.assertGreater(report.factor_count, 0)
            report_text = (paths.reports_dir / "factor_tearsheet_information_technology_10d.md").read_text(encoding="utf-8")
            self.assertIn("# Factor Tearsheet", report_text)
            self.assertIn("- sector: Information Technology", report_text)
            self.assertIn("## relative_strength_index_vs_spy", report_text)
            self.assertIn("## relative_strength_index_vs_subindustry", report_text)
            self.assertIn("- q5_minus_q1_spread:", report_text)
            self.assertIn("- daily_ic_mean:", report_text)
            self.assertIn("- ic_t_stat:", report_text)
            self.assertIn("- turnover_q5:", report_text)
            self.assertIn("- best_regime:", report_text)
            self.assertIn("- worst_regime:", report_text)
            self.assertIn("### Mean Forward Sector Alpha By Quintile", report_text)
            self.assertIn("- Q5: mean_target=", report_text)

    def test_factor_tearsheet_skips_constant_input_days_in_ic_calculation(self) -> None:
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
                        "ticker": ticker,
                        "sector": "Energy",
                        "regime_green": True,
                        "relative_strength_index_vs_spy": 80.0,
                        "relative_strength_index_vs_qqq": 80.0,
                        "relative_strength_index_vs_xlk": 80.0,
                        "relative_strength_index_vs_subindustry": None,
                        "roc_63": 0.10,
                        "roc_126": 0.20,
                        "vol_alpha": 1.0,
                        "sma_200_dist": 0.10,
                        "sma_50_dist": 0.05,
                        "rsi_14": 50.0,
                        "atr_14": 2.0,
                        "days_to_next_earnings": 5.0,
                        "days_since_last_earnings": 2.0,
                        "last_earnings_gap_pct": 0.01,
                        "last_earnings_volume_ratio_20": 1.1,
                        "last_earnings_open_vs_20d_high": 0.0,
                        "close_vs_last_earnings_close": 0.02,
                        "avg_abs_gap_pct_20": 0.01,
                        "max_gap_down_pct_60": 0.02,
                        "distance_above_20d_high": 0.01,
                        "base_range_pct_20": 0.04,
                        "base_atr_contraction_20": 0.8,
                        "base_volume_dryup_ratio_20": 0.8,
                        "breakout_volume_ratio_50": 1.2,
                        "sector_pct_above_50": 0.8,
                        "sector_pct_above_200": 0.8,
                        "sector_median_roc_63": 0.08,
                        "alpha_vs_sector_10d": alpha,
                    }
                    for ticker, alpha in (("AAA", 0.01), ("BBB", 0.02), ("CCC", 0.03))
                ]
            )

            class FakeDB:
                def __init__(self, paths, snapshot_frame):
                    self.paths = paths
                    self._snapshot_frame = snapshot_frame

                def initialize(self): return None
                def load_universe_daily_snapshots(self, snapshot_date=None):
                    return self._snapshot_frame.copy()

            service = FactorTearsheetService(FakeDB(paths, snapshot_frame))
            report = service.run(sector="Energy", horizon_days=10)

            self.assertGreater(report.factor_count, 0)
            report_text = (paths.reports_dir / "factor_tearsheet_energy_10d.md").read_text(encoding="utf-8")
            self.assertIn("- daily_ic_mean: nan", report_text)
            self.assertIn("## Unavailable Factors", report_text)
            self.assertIn("relative_strength_index_vs_subindustry: reason=no_non_null_rows", report_text)
