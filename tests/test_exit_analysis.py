from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from src.research.exit_analysis_service import ExitAnalysisService
from src.settings import AppPaths
from src.utils.strategy import ExitRules, ProductionStrategy


class ExitAnalysisServiceTests(unittest.TestCase):
    def test_exit_analysis_writes_counterfactual_report(self) -> None:
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
            )
            paths.reports_dir.mkdir(parents=True, exist_ok=True)
            paths.logs_dir.mkdir(parents=True, exist_ok=True)

            closed_trades = [
                {
                    "rowid": 1,
                    "ticker": "AAA",
                    "entry_date": "2026-05-01",
                    "entry_price": 100.0,
                    "entry_atr": 2.0,
                    "strategy_id": 11,
                    "strategy_slot": "energy",
                    "shares": 10,
                    "max_price_seen": 110.0,
                    "status": "closed",
                    "exit_date": "2026-05-08",
                    "exit_price": 108.0,
                },
                {
                    "rowid": 2,
                    "ticker": "LEGACY",
                    "entry_date": "2026-05-01",
                    "entry_price": 50.0,
                    "entry_atr": 1.5,
                    "strategy_id": 99,
                    "strategy_slot": "technology",
                    "shares": 20,
                    "max_price_seen": 52.0,
                    "status": "closed",
                    "exit_date": "2026-05-08",
                    "exit_price": 48.0,
                }
            ]
            price_history = pd.DataFrame(
                [
                    {
                        "ticker": ticker,
                        "date": day,
                        "open": close - 1.0,
                        "high": close + 1.0,
                        "low": close - 2.0,
                        "close": close,
                        "volume": 1_000_000,
                        "adj_close": close,
                    }
                    for ticker, closes in {
                        "AAA": [100.0, 101.0, 103.0, 105.0, 107.0, 108.0, 109.0],
                        "LEGACY": [50.0, 50.5, 51.0, 50.5, 49.5, 48.0, 47.5],
                        "SPY": [100.0, 100.5, 101.0, 101.5, 102.0, 102.5, 103.0],
                        "XLE": [100.0, 100.8, 101.6, 102.4, 103.2, 104.0, 104.8],
                    }.items()
                    for day, close in zip(pd.bdate_range("2026-05-01", periods=7), closes)
                ]
            )

            class FakeDB:
                def __init__(self, paths, closed_trades, price_history):
                    self.paths = paths
                    self._closed_trades = closed_trades
                    self._price_history = price_history

                def initialize(self): return None
                def list_closed_trades(self): return self._closed_trades
                def list_universe_rows(self, active_only=False):
                    return [
                        {"ticker": "AAA", "sector": "Energy"},
                        {"ticker": "LEGACY", "sector": "Technology"},
                    ]
                def load_price_history(self, tickers):
                    return self._price_history[self._price_history["ticker"].isin(tickers)].copy()
                def get_backtest_result_by_strategy_id(self, strategy_id):
                    return None
                def load_scan_candidates(self, scan_date=None):
                    return pd.DataFrame(
                        [
                            {
                                "scan_date": "2026-04-30",
                                "ticker": "AAA",
                                "strategy_slot": "energy",
                                "selected": 1,
                            }
                        ]
                    )

            strategy = ProductionStrategy(
                strategy_id=11,
                promoted_at="2026-05-01T17:00:00",
                indicators={"relative_strength_index_vs_spy_min": 75.0, "signal_score_min": 32.0},
                exit_rules=ExitRules(0.05, 0.12, 20),
                slot="energy",
                sector="Energy",
            )
            service = ExitAnalysisService(FakeDB(paths, closed_trades, price_history))
            with patch("src.research.exit_analysis_service.load_active_strategies", return_value={"energy": strategy}):
                report = service.run(horizons=(5, 10))

            self.assertEqual(report.closed_trade_count, 2)
            self.assertEqual(report.linked_trade_count, 1)
            self.assertEqual(report.analyzed_trade_count, 1)
            report_text = (paths.reports_dir / "exit_analysis.md").read_text(encoding="utf-8")
            self.assertIn("# Exit Analysis", report_text)
            self.assertIn("- linked_trade_count: 1", report_text)
            self.assertIn("- excluded_non_recommendation_trades: 1", report_text)
            self.assertIn("## Actual Exit Summary", report_text)
            self.assertIn("## Horizon Comparison", report_text)
            self.assertIn("### 5d", report_text)
            self.assertIn("## Slot Breakdown", report_text)
            self.assertIn("### energy", report_text)
            self.assertIn("## Biggest Givebacks", report_text)
            self.assertIn("### AAA", report_text)
            self.assertNotIn("### LEGACY", report_text)
