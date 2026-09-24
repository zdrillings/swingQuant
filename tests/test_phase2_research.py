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
                                    "alpha_vs_sector_20d": 0.02 - ticker_index * 0.01 + date_index * 0.0001,
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
            self.assertIn("## Random-Eligible Baseline and Tilt", report_text)
            self.assertIn("## P2.3 Opportunity Score Repair", report_text)
            self.assertIn("## P2.4 Momentum-Crash Conditioning", report_text)
            self.assertIn("## P2.5 Deflation Gates", report_text)
            self.assertIn("- monotone_non_decreasing: false", report_text)
            self.assertIn("- trial_count: 200", report_text)
            self.assertIn("- provenance_source: latest_db_run_fallback", report_text)
            self.assertIn("- dsr_proxy_basis: Newey-West t-stat", report_text)

    def test_phase2_research_prefers_current_artifact_csv_over_stale_db_champion(self) -> None:
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
            (paths.reports_dir / "shortlist_model.md").write_text(
                "\n".join(
                    [
                        "# Shortlist Model",
                        "- generated_at: 2026-09-08T20:00:00+00:00",
                        "- feature_profile: repaired_v2",
                        "- test_window_dates: 20",
                        "- oos_evaluation_stride_dates: 20",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            artifact_rows = []
            for snapshot_date in pd.bdate_range("2026-08-03", periods=22):
                artifact_rows.extend(
                    [
                        {
                            "snapshot_date": snapshot_date,
                            "ticker": "AAA",
                            "sector": "Energy",
                            "model_name": "artifact_model",
                            "predicted_alpha": 0.20,
                            "path_alpha_vs_sector_60d": 0.03,
                        },
                        {
                            "snapshot_date": snapshot_date,
                            "ticker": "BBB",
                            "sector": "Materials",
                            "model_name": "artifact_model",
                            "predicted_alpha": 0.10,
                            "path_alpha_vs_sector_60d": 0.01,
                        },
                    ]
                )
            pd.DataFrame(artifact_rows).to_csv(paths.reports_dir / "shortlist_model_oos_predictions.csv", index=False)

            class FakeDB:
                def __init__(self, paths):
                    self.paths = paths
                    self.prediction_calls = 0

                def initialize(self): return None
                def load_shortlist_model_runs(self, **kwargs):
                    return pd.DataFrame([{"generated_at": "2026-08-20T00:00:00+00:00"}])
                def load_shortlist_model_predictions(self, **kwargs):
                    self.prediction_calls += 1
                    return pd.DataFrame()
                def load_scan_candidates(self):
                    return pd.DataFrame()

            db = FakeDB(paths)
            report = Phase2ResearchService(db).run(horizon_days=60, top_n=1, trial_count=250)

            self.assertEqual(report.models, 1)
            self.assertEqual(db.prediction_calls, 0)
            report_text = (paths.reports_dir / "phase2_research.md").read_text(encoding="utf-8")
            self.assertIn("- provenance_source: artifact_csv", report_text)
            self.assertIn("- provenance_generated_at: 2026-09-08T20:00:00+00:00", report_text)
            self.assertIn("- provenance_feature_profile: repaired_v2", report_text)
            self.assertIn("- target_column: path_alpha_vs_sector_60d", report_text)
            self.assertIn("- trial_count: 250", report_text)

    def test_phase2_research_refuses_missing_horizon_target_column(self) -> None:
        service = Phase2ResearchService(db_manager=object())
        frame = pd.DataFrame(
            [
                {
                    "snapshot_date": "2026-09-01",
                    "ticker": "AAA",
                    "model_name": "model",
                    "predicted_alpha": 0.1,
                    "path_alpha_vs_sector_60d": 0.02,
                }
            ]
        )

        with self.assertRaisesRegex(
            ValueError,
            "expected path_alpha_vs_sector_20d or alpha_vs_sector_20d",
        ):
            service._target_column(frame, horizon_days=20)

    def test_phase2_research_parser_accepts_args(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["phase2-research", "--horizon", "60", "--top", "2", "--trial-count", "300"])

        self.assertEqual(args.command, "phase2-research")
        self.assertEqual(args.horizon, 60)
        self.assertEqual(args.top, 2)
        self.assertEqual(args.trial_count, 300)


if __name__ == "__main__":
    unittest.main()
