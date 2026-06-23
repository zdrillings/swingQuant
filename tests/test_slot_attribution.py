from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from src.scan.slot_attribution_service import SlotAttributionService
from src.settings import AppPaths
from src.utils.strategy import ExitRules, ProductionStrategy


class SlotAttributionServiceTests(unittest.TestCase):
    def test_slot_attribution_writes_active_slot_report(self) -> None:
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
            for scan_date_value in pd.date_range("2026-03-01", periods=35, freq="D"):
                scan_date = pd.Timestamp(scan_date_value).date().isoformat()
                for slot, sector in (("energy", "Energy"), ("materials", "Materials")):
                    for ticker_index in range(3):
                        rows.append(
                            {
                                "scan_date": scan_date,
                                "ticker": f"{slot[:2].upper()}{scan_date[-2:]}{ticker_index}",
                                "strategy_slot": slot,
                                "strategy_sector": sector,
                                "sector": sector,
                                "signal_score": 28.0 + ticker_index,
                                "setup_quality_score": 0.50 + (ticker_index * 0.10),
                                "expected_alpha_score": 0.20 + (ticker_index * 0.10),
                                "breadth_score": 0.40 + (ticker_index * 0.05),
                                "freshness_score": 0.45 + (ticker_index * 0.05),
                                "overlap_penalty": 0.0,
                                "opportunity_score": 0.55 + (ticker_index * 0.05),
                                "selected": 1 if ticker_index == 2 else 0,
                                "selected_rank": 1 if ticker_index == 2 else None,
                                "md_volume_30d": 5_000_000 + ticker_index,
                                "adj_close": 20.0 + ticker_index,
                                "alpha_vs_sector_10d": -0.01 + (ticker_index * 0.03),
                                "details_json": "{}",
                            }
                        )
            scan_frame = pd.DataFrame(rows)

            class FakeDB:
                def __init__(self, paths, scan_frame):
                    self.paths = paths
                    self._scan_frame = scan_frame

                def initialize(self): return None
                def load_scan_candidates(self, scan_date=None):
                    return self._scan_frame.copy()

            strategies = {
                "energy": ProductionStrategy(1, "2026-05-01T17:00:00", {"rsi_14_max": 35.0}, ExitRules(0.05, 0.12, 20), slot="energy", sector="Energy"),
                "materials": ProductionStrategy(2, "2026-05-01T17:00:00", {"rsi_14_max": 35.0}, ExitRules(0.05, 0.12, 20), slot="materials", sector="Materials"),
            }

            service = SlotAttributionService(FakeDB(paths, scan_frame))
            with patch(
                "src.scan.slot_attribution_service.load_active_strategies",
                return_value=strategies,
            ), patch(
                "src.scan.slot_attribution_service.load_feature_config",
                return_value={"scan_policy": {"max_candidates_per_slot": 1, "learned_ranker": {"min_train_rows": 5, "min_train_dates": 3}}},
            ):
                report = service.run(horizon_days=10)

            self.assertEqual(report.slot_count, 2)
            report_text = (paths.reports_dir / "slot_attribution.md").read_text(encoding="utf-8")
            self.assertIn("# Slot Attribution", report_text)
            self.assertIn("- target_column: alpha_vs_sector_10d", report_text)
            self.assertIn("- validation_method: purged_walk_forward", report_text)
            self.assertIn("## energy", report_text)
            self.assertIn("- eligible_candidate_alpha: mean_target=", report_text)
            self.assertIn("- selected_candidate_alpha: mean_target=", report_text)
            self.assertIn("- learned_ranker_candidate_alpha: mean_target=", report_text)
            self.assertIn("- handcrafted_candidate_alpha: mean_target=", report_text)
            self.assertIn("- random_eligible_alpha: mean_target=", report_text)
