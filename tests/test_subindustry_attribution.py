from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from src.research.subindustry_attribution_service import SubindustryAttributionService
from src.settings import AppPaths


class SubindustryAttributionServiceTests(unittest.TestCase):
    def test_subindustry_attribution_writes_report(self) -> None:
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

            class FakeDB:
                def __init__(self, paths):
                    self.paths = paths

                def initialize(self): return None
                def latest_run_id(self): return 49
                def list_research_universe(self, limit=250):
                    return [
                        {"ticker": "AMD", "sector": "Information Technology", "sub_industry": "Semiconductors", "md_volume_30d": 10_000_000},
                        {"ticker": "AKAM", "sector": "Information Technology", "sub_industry": "Internet Services", "md_volume_30d": 8_000_000},
                    ]
                def load_price_history(self, tickers):
                    return pd.DataFrame(
                        [
                            {"ticker": "AMD", "date": "2026-05-01", "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0, "adj_close": 1.0},
                            {"ticker": "AKAM", "date": "2026-05-01", "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0, "adj_close": 1.0},
                        ]
                    )
                def load_earnings_calendar(self, tickers):
                    return pd.DataFrame(columns=["ticker", "earnings_date"])

            service = SubindustryAttributionService(FakeDB(paths))
            chosen = {
                "strategy_id": 1156518,
                "expectancy": 0.008763,
                "profit_factor": 1.579388,
                "alpha_vs_sector": 0.009096,
                "mdd": 0.248137,
                "trade_count": 76,
                "params_json": "",
            }
            params = {
                "sweep_mode": "tech_post_earnings_followthrough_v3",
                "indicators": {"relative_strength_index_vs_spy_min": 75.0},
                "exit_rules": {"time_limit_days": 5},
                "backtest_costs": {"slippage_bps_per_side": 5.0, "commission_bps_per_side": 0.0},
            }
            chosen["params_json"] = __import__("json").dumps(params, sort_keys=True)
            analysis_frame = pd.DataFrame(
                [
                    {"ticker": "AMD", "date": "2026-05-01", "sector": "Information Technology", "sub_industry": "Semiconductors", "regime_green": True},
                    {"ticker": "AKAM", "date": "2026-05-01", "sector": "Information Technology", "sub_industry": "Internet Services", "regime_green": True},
                ]
            )

            with patch.object(service, "_select_strategy_row", return_value=chosen), patch(
                "src.research.subindustry_attribution_service.build_analysis_frame",
                return_value=(analysis_frame, []),
            ), patch.object(
                service,
                "_build_historical_subindustry_rows",
                return_value=[
                    {
                        "sub_industry": "Semiconductors",
                        "trade_count": 42,
                        "expectancy": 0.012300,
                        "alpha_vs_sector": 0.010100,
                        "profit_factor": 1.900000,
                        "win_rate": 0.6200,
                        "mdd": 0.150000,
                    }
                ],
            ), patch.object(
                service,
                "_build_live_subindustry_rows",
                return_value=[
                    {
                        "sub_industry": "Semiconductors",
                        "live_match_count": 2,
                        "avg_signal_score": 31.5,
                        "tickers": ["AMD", "COHR"],
                    }
                ],
            ):
                report = service.run(sector="Information Technology", run_id=49)

            self.assertEqual(report.strategy_id, 1156518)
            report_text = (paths.reports_dir / "subindustry_attribution.md").read_text(encoding="utf-8")
            self.assertIn("# Subindustry Attribution", report_text)
            self.assertIn("- strategy_id: 1156518", report_text)
            self.assertIn("## Historical By Subindustry", report_text)
            self.assertIn("### Semiconductors", report_text)
            self.assertIn("- trade_count: 42", report_text)
            self.assertIn("## Live Candidates By Subindustry", report_text)
            self.assertIn("- tickers: AMD, COHR", report_text)
