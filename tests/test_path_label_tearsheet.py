from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src.cli import build_parser
from src.research.path_label_tearsheet_service import PathLabelTearsheetService
from src.settings import AppPaths


class PathLabelTearsheetServiceTests(unittest.TestCase):
    def test_path_label_tearsheet_compares_fixed_and_path_labels(self) -> None:
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

            class FakeDB:
                def __init__(self, paths):
                    self.paths = paths

                def initialize(self): return None
                def load_universe_daily_snapshots(self, snapshot_date=None):
                    return pd.DataFrame(
                        [
                            {
                                "snapshot_date": "2026-05-01",
                                "ticker": "AAA",
                                "alpha_vs_sector_20d": 0.10,
                                "path_alpha_vs_sector_20d": 0.04,
                                "path_exit_reason_20d": "profit_target",
                                "path_holding_days_20d": 5,
                                "alpha_vs_sector_60d": 0.20,
                                "path_alpha_vs_sector_60d": 0.05,
                                "path_exit_reason_60d": "time_limit",
                                "path_holding_days_60d": 10,
                            },
                            {
                                "snapshot_date": "2026-05-02",
                                "ticker": "BBB",
                                "alpha_vs_sector_20d": -0.02,
                                "path_alpha_vs_sector_20d": -0.05,
                                "path_exit_reason_20d": "hard_stop",
                                "path_holding_days_20d": 1,
                                "alpha_vs_sector_60d": -0.10,
                                "path_alpha_vs_sector_60d": -0.05,
                                "path_exit_reason_60d": "hard_stop",
                                "path_holding_days_60d": 1,
                            },
                            {
                                "snapshot_date": "2026-05-03",
                                "ticker": "CCC",
                                "alpha_vs_sector_20d": 0.03,
                                "path_alpha_vs_sector_20d": None,
                                "path_exit_reason_20d": None,
                                "path_holding_days_20d": None,
                                "alpha_vs_sector_60d": 0.08,
                                "path_alpha_vs_sector_60d": None,
                                "path_exit_reason_60d": None,
                                "path_holding_days_60d": None,
                            },
                        ]
                    )

            report = PathLabelTearsheetService(FakeDB(paths)).run(horizon_days=20)

            self.assertEqual(report.rows, 3)
            self.assertEqual(report.paired_rows, 2)
            report_text = (paths.reports_dir / "path_label_tearsheet.md").read_text(encoding="utf-8")
            self.assertIn("# Path Label Tearsheet", report_text)
            self.assertIn("- paired_rows: 2", report_text)
            self.assertIn("| 20d | fixed_horizon | 3 | 0.036667 | 0.030000 | 0.666667 | -0.010000 | 0.086000 |", report_text)
            self.assertIn("| 20d | path_aware | 2 | -0.005000 | -0.005000 | 0.500000 | -0.041000 | 0.031000 |", report_text)
            self.assertIn("| 60d | fixed_horizon | 3 | 0.060000 | 0.080000 | 0.666667 | -0.064000 | 0.176000 |", report_text)
            self.assertIn("- mean_path_minus_fixed: -0.045000", report_text)
            self.assertIn("- fixed_60d_mean: 0.050000", report_text)
            self.assertIn("- path_60d_mean: 0.000000", report_text)
            self.assertIn("- mean_holding_days: 5.500000", report_text)
            self.assertIn("| hard_stop | 1 | 0.333333 |", report_text)
            self.assertIn("| time_limit | 1 | 0.333333 |", report_text)

    def test_path_label_tearsheet_parser_accepts_args(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["path-label-tearsheet", "--horizon", "20"])

        self.assertEqual(args.command, "path-label-tearsheet")
        self.assertEqual(args.horizon, 20)


if __name__ == "__main__":
    unittest.main()
