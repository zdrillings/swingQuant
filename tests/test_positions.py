from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from src.positions.service import PositionsService
from src.utils.strategy import ExitRules, ProductionStrategy


class PositionsServiceTests(unittest.TestCase):
    def test_positions_reports_historical_strategy_and_sell_context(self) -> None:
        class FakeDB:
            def initialize(self): return None
            def list_open_trades(self):
                return [
                    {
                        "ticker": "RKLB",
                        "entry_date": "2026-05-05",
                        "entry_price": 100.0,
                        "entry_atr": None,
                        "strategy_id": 172365,
                        "strategy_slot": "legacy_technology",
                        "shares": 10,
                        "max_price_seen": 100.0,
                        "status": "open",
                    }
                ]
            def load_price_history(self, tickers):
                rows = []
                for ticker in ("RKLB", "SPY", "QQQ"):
                    for day in pd.bdate_range("2025-06-01", periods=220):
                        rows.append(
                            {
                                "ticker": ticker,
                                "date": day.date(),
                                "open": 100.0,
                                "high": 101.0,
                                "low": 99.0,
                                "close": 100.0,
                                "volume": 1000,
                                "adj_close": 100.0,
                            }
                        )
                return pd.DataFrame(rows)
            def list_universe_rows(self, active_only=False):
                return [{"ticker": "RKLB", "sector": "Information Technology", "md_volume_30d": 30_000_000}]
            def load_earnings_calendar(self, tickers):
                return pd.DataFrame()
            def get_backtest_result_by_strategy_id(self, strategy_id):
                if strategy_id != 172365:
                    return None
                return {
                    "strategy_id": 172365,
                    "params_json": (
                        '{"sector":"Information Technology","indicators":{"rsi_14_max":40.0},'
                        '"exit_rules":{"trailing_stop_pct":null,"profit_target_pct":null,'
                        '"time_limit_days":20,"trailing_stop_atr_mult":3.0,"profit_target_atr_mult":4.0}}'
                    ),
                }

        service = PositionsService(FakeDB())
        analysis_frame = pd.DataFrame(
            [
                {"ticker": "SPY", "date": pd.Timestamp("2026-05-05"), "spy_sma_200": 100.0, "qqq_sma_200": None},
                {"ticker": "QQQ", "date": pd.Timestamp("2026-05-05"), "spy_sma_200": None, "qqq_sma_200": 100.0},
                {"ticker": "RKLB", "date": pd.Timestamp("2026-05-05"), "adj_close": 100.0, "atr_14": 4.0},
            ]
        )

        with patch.object(service, "_load_intraday_last_prices", return_value={"RKLB": 87.0, "SPY": 105.0, "QQQ": 110.0}), \
             patch.object(service, "_download_recent_daily_history", return_value=pd.DataFrame()), \
             patch("src.positions.service.load_active_strategies", return_value={"materials": ProductionStrategy(strategy_id=10, promoted_at="2026-05-05T17:00:00", indicators={"rsi_14_max": 35.0}, exit_rules=ExitRules(0.05, 0.12, 20), slot="materials", sector="Materials")}), \
             patch("src.positions.service.build_analysis_frame", return_value=(analysis_frame, [])), \
             patch("src.positions.service.latest_rsi_2_with_intraday", return_value=20.0):
            report = service.run()

        self.assertEqual(report.position_count, 1)
        self.assertEqual(report.sell_count, 1)
        snapshot = report.snapshots[0]
        self.assertEqual(snapshot.strategy_source, "historical_strategy_id")
        self.assertEqual(snapshot.resolved_slot, "legacy_technology")
        self.assertEqual(snapshot.action, "sell")
        self.assertIn("inactive stored slot", snapshot.notes)
        self.assertIn("missing stored entry_atr", snapshot.notes)
        rendered = report.render_console()
        self.assertIn("Open positions: 1", rendered)
        self.assertIn("legacy_technology", rendered)
        self.assertIn("historical_strategy_id", rendered)
        self.assertIn("sell", rendered)
