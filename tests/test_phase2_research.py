from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src.cli import build_parser
from src.research.phase2_research_service import Phase2ResearchService
from src.settings import AppPaths


class Phase2ResearchServiceTests(unittest.TestCase):
    def test_phase2_research_report_renders_all_phase_sections(self) -> None:
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
                def load_shortlist_model_runs(self, **kwargs):
                    return pd.DataFrame([{"generated_at": "2026-06-01T00:00:00+00:00"}])
                def load_shortlist_model_predictions(self, *, model_name, **kwargs):
                    rows = []
                    for date_index, snapshot_date in enumerate(pd.bdate_range("2026-01-02", periods=32)):
                        for ticker_index, ticker in enumerate(("AAA", "BBB", "CCC")):
                            rows.append(
                                {
                                    "snapshot_date": snapshot_date,
                                    "ticker": ticker,
                                    "sector": "Energy",
                                    "predicted_alpha": float(3 - ticker_index) + (0.1 if model_name == "ridge_model" else 0.0),
                                    "actual_alpha_vs_sector": 0.02 - ticker_index * 0.01 + date_index * 0.0001,
                                }
                            )
                    return pd.DataFrame(rows) if model_name in {"signal_proxy", "ridge_model"} else pd.DataFrame()
                def load_scan_candidates(self):
                    return pd.DataFrame(
                        [
                            {"scan_date": "2026-01-02", "ticker": "AAA", "opportunity_score": 0.25, "alpha_vs_sector_20d": -0.02},
                            {"scan_date": "2026-01-03", "ticker": "BBB", "opportunity_score": 0.35, "alpha_vs_sector_20d": 0.01},
                            {"scan_date": "2026-01-04", "ticker": "CCC", "opportunity_score": 0.45, "alpha_vs_sector_20d": 0.03},
                            {"scan_date": "2026-01-05", "ticker": "DDD", "opportunity_score": 0.55, "alpha_vs_sector_20d": -0.01},
                        ]
                    )

            report = Phase2ResearchService(FakeDB(paths)).run(horizon_days=20, top_n=2)

            self.assertEqual(report.models, 2)
            self.assertEqual(report.oos_dates, 32)
            report_text = (paths.reports_dir / "phase2_research.md").read_text(encoding="utf-8")
            self.assertIn("## P2.2 Meta-Label Sizing", report_text)
            self.assertIn("## P2.3 Opportunity Score Repair", report_text)
            self.assertIn("## P2.4 Momentum-Crash Conditioning", report_text)
            self.assertIn("## P2.5 Deflation Gates", report_text)
            self.assertIn("- monotone_non_decreasing: false", report_text)
            self.assertIn("- trial_count: 2", report_text)

    def test_phase2_research_parser_accepts_args(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["phase2-research", "--horizon", "20", "--top", "2"])

        self.assertEqual(args.command, "phase2-research")
        self.assertEqual(args.horizon, 20)
        self.assertEqual(args.top, 2)


if __name__ == "__main__":
    unittest.main()
