from __future__ import annotations

import json
import unittest
from unittest.mock import patch

import pandas as pd

from src.cli import build_parser
from src.quote.service import QuoteService
from src.utils.shortlist_runtime import LiveShortlistModelContext
from src.utils.strategy import ExitRules, ProductionStrategy


class QuoteServiceTests(unittest.TestCase):
    def test_quote_reports_live_trade_context_and_model_metadata(self) -> None:
        class FakeDB:
            def initialize(self): return None

            def get_latest_open_trade(self, ticker):
                return {
                    "ticker": "CC",
                    "entry_date": "2026-06-01",
                    "entry_price": 23.62,
                    "entry_atr": 1.4086,
                    "strategy_id": 987486,
                    "strategy_slot": "materials",
                    "shares": 100,
                    "max_price_seen": 25.555,
                    "status": "open",
                }

            def list_universe_rows(self, active_only=False):
                return [{"ticker": "CC", "sector": "Materials", "md_volume_30d": 30_000_000}]

            def load_price_history(self, tickers):
                rows = []
                for ticker in ("CC", "SPY", "QQQ", "XLB"):
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

            def load_earnings_calendar(self, tickers):
                return pd.DataFrame()

            def load_scan_candidates(self, scan_date=None):
                return pd.DataFrame(
                    [
                        {
                            "scan_date": "2026-06-01",
                            "ticker": "CC",
                            "selected": 0,
                            "selected_rank": None,
                            "details_json": json.dumps(
                                {
                                    "ranking_components": {
                                        "selection_source": "shortlist_model",
                                        "model_predicted_alpha": 0.1642,
                                        "model_reason_summary": "strong earnings volume",
                                    }
                                }
                            ),
                        }
                    ]
                )

        service = QuoteService(FakeDB())
        analysis_frame = pd.DataFrame(
            [
                {"ticker": "SPY", "date": pd.Timestamp("2026-06-02"), "adj_close": 100.0, "spy_sma_200": 95.0, "qqq_sma_200": None},
                {"ticker": "QQQ", "date": pd.Timestamp("2026-06-02"), "adj_close": 100.0, "spy_sma_200": None, "qqq_sma_200": 95.0},
                {"ticker": "XLB", "date": pd.Timestamp("2026-06-02"), "adj_close": 100.0, "spy_sma_200": 95.0, "qqq_sma_200": None},
                {
                    "ticker": "CC",
                    "date": pd.Timestamp("2026-06-02"),
                    "adj_close": 22.15,
                    "atr_14": 1.6,
                    "days_to_next_earnings": 12.0,
                    "relative_strength_index_vs_spy": 55.0,
                },
            ]
        )
        model_context = LiveShortlistModelContext(
            generated_at="2026-06-02T18:00:00",
            champion_model="xgboost_model",
            live_snapshot_date="2026-06-02",
            top_n=10,
            live_predictions=pd.DataFrame(
                [
                    {
                        "ticker": "CC",
                        "sector": "Materials",
                        "predicted_alpha": 0.1642,
                        "model_rank": 5,
                        "model_reason_summary": "strong earnings volume",
                        "model_comparison_summary": "SXT in Materials on strong earnings volume",
                    }
                ]
            ),
        )

        with patch.object(service, "_load_intraday_last_prices", return_value={"CC": 22.32, "SPY": 105.0, "QQQ": 110.0, "XLB": 105.0}), \
             patch.object(service, "_download_recent_daily_history", return_value=pd.DataFrame()), \
             patch("src.quote.service.load_active_strategies", return_value={"materials": ProductionStrategy(strategy_id=987486, promoted_at="2026-05-05T17:00:00", indicators={"relative_strength_index_vs_spy_min": 60.0}, exit_rules=ExitRules(trailing_stop_pct=None, profit_target_pct=None, time_limit_days=20, trailing_stop_atr_mult=2.5, profit_target_atr_mult=4.5), slot="materials", sector="Materials")}), \
             patch("src.quote.service.build_analysis_frame", return_value=(analysis_frame, [])), \
             patch("src.quote.service.latest_rsi_2_with_intraday", return_value=20.0), \
             patch.object(service, "_load_shortlist_model_context", return_value=model_context):
            report = service.run(ticker="CC")

        self.assertEqual(report.ticker, "CC")
        self.assertAlmostEqual(report.current_price, 22.32)
        self.assertEqual(report.current_price_source, "intraday")
        self.assertTrue(report.has_open_trade)
        self.assertEqual(report.model_name, "xgboost_model")
        self.assertEqual(report.model_rank, 5)
        self.assertAlmostEqual(report.model_predicted_alpha, 0.1642)
        self.assertEqual(report.latest_scan_selection_source, "shortlist_model")
        self.assertFalse(report.sell_now)
        rendered = report.render_console()
        self.assertIn("Ticker: CC", rendered)
        self.assertIn("Current price: 22.32 (intraday)", rendered)
        self.assertIn("predicted 20d alpha vs sector +16.42%", rendered)
        self.assertIn("Open trade:", rendered)
        self.assertIn("Sell now: no", rendered)

    def test_quote_uses_fresher_scan_close_when_intraday_missing(self) -> None:
        class FakeDB:
            def initialize(self): return None

            def get_latest_open_trade(self, ticker):
                return {
                    "ticker": "VSH",
                    "entry_date": "2026-06-11",
                    "entry_price": 57.34,
                    "entry_atr": 6.5,
                    "strategy_id": 1140440,
                    "strategy_slot": "technology",
                    "shares": 100,
                    "max_price_seen": 58.80,
                    "status": "open",
                }

            def list_universe_rows(self, active_only=False):
                return [{"ticker": "VSH", "sector": "Information Technology", "md_volume_30d": 30_000_000}]

            def load_price_history(self, tickers):
                rows = []
                for ticker in ("VSH", "SPY", "QQQ"):
                    rows.append(
                        {
                            "ticker": ticker,
                            "date": pd.Timestamp("2026-06-02"),
                            "open": 57.99,
                            "high": 64.50,
                            "low": 57.81,
                            "close": 62.84,
                            "volume": 1_000_000,
                            "adj_close": 62.84,
                        }
                    )
                return pd.DataFrame(rows)

            def load_earnings_calendar(self, tickers):
                return pd.DataFrame()

            def load_scan_candidates(self, scan_date=None):
                return pd.DataFrame(
                    [
                        {
                            "scan_date": "2026-06-22",
                            "ticker": "VSH",
                            "adj_close": 57.54,
                            "selected": 0,
                            "selected_rank": None,
                            "details_json": json.dumps({"ranking_components": {"selection_source": "shortlist_model"}}),
                        }
                    ]
                )

        service = QuoteService(FakeDB())
        analysis_frame = pd.DataFrame(
            [
                {"ticker": "SPY", "date": pd.Timestamp("2026-06-02"), "adj_close": 62.84, "spy_sma_200": 50.0, "qqq_sma_200": None},
                {"ticker": "QQQ", "date": pd.Timestamp("2026-06-02"), "adj_close": 62.84, "spy_sma_200": None, "qqq_sma_200": 50.0},
                {
                    "ticker": "VSH",
                    "date": pd.Timestamp("2026-06-02"),
                    "adj_close": 62.84,
                    "atr_14": 6.5,
                    "days_to_next_earnings": 40.0,
                    "relative_strength_index_vs_spy": 70.0,
                },
            ]
        )

        with patch.object(service, "_load_intraday_last_prices", return_value={}), \
             patch.object(service, "_download_recent_daily_history", return_value=pd.DataFrame()), \
             patch("src.quote.service.load_active_strategies", return_value={"technology": ProductionStrategy(strategy_id=1140440, promoted_at="2026-06-01T17:00:00", indicators={"relative_strength_index_vs_spy_min": 60.0}, exit_rules=ExitRules(trailing_stop_pct=None, profit_target_pct=None, time_limit_days=20, trailing_stop_atr_mult=2.5, profit_target_atr_mult=4.5), slot="technology", sector="Information Technology")}), \
             patch("src.quote.service.build_analysis_frame", return_value=(analysis_frame, [])), \
             patch("src.quote.service.latest_rsi_2_with_intraday", return_value=20.0), \
             patch.object(service, "_load_shortlist_model_context", return_value=None):
            report = service.run(ticker="VSH")

        self.assertAlmostEqual(report.current_price, 57.54)
        self.assertEqual(report.current_price_source, "scan_close:2026-06-22")
        self.assertAlmostEqual(report.unrealized_pct, (57.54 / 57.34) - 1.0)
        self.assertFalse(report.sell_now)
        self.assertEqual(report.exit_flags, ())

    def test_quote_refuses_stale_close_before_entry(self) -> None:
        class FakeDB:
            def initialize(self): return None

            def get_latest_open_trade(self, ticker):
                return {
                    "ticker": "VSH",
                    "entry_date": "2026-06-11",
                    "entry_price": 57.34,
                    "entry_atr": 6.5,
                    "strategy_id": 1140440,
                    "strategy_slot": "technology",
                    "shares": 100,
                    "max_price_seen": 58.80,
                    "status": "open",
                }

            def list_universe_rows(self, active_only=False):
                return [{"ticker": "VSH", "sector": "Information Technology", "md_volume_30d": 30_000_000}]

            def load_price_history(self, tickers):
                return pd.DataFrame()

            def load_earnings_calendar(self, tickers):
                return pd.DataFrame()

            def load_scan_candidates(self, scan_date=None):
                return pd.DataFrame()

        service = QuoteService(FakeDB())
        analysis_frame = pd.DataFrame(
            [
                {"ticker": "SPY", "date": pd.Timestamp("2026-06-02"), "adj_close": 62.84, "spy_sma_200": 50.0, "qqq_sma_200": None},
                {"ticker": "QQQ", "date": pd.Timestamp("2026-06-02"), "adj_close": 62.84, "spy_sma_200": None, "qqq_sma_200": 50.0},
                {"ticker": "VSH", "date": pd.Timestamp("2026-06-02"), "adj_close": 62.84, "atr_14": 6.5},
            ]
        )

        with patch.object(service, "_load_intraday_last_prices", return_value={}), \
             patch.object(service, "_download_recent_daily_history", return_value=pd.DataFrame()), \
             patch("src.quote.service.load_active_strategies", return_value={"technology": ProductionStrategy(strategy_id=1140440, promoted_at="2026-06-01T17:00:00", indicators={}, exit_rules=ExitRules(trailing_stop_pct=None, profit_target_pct=None, time_limit_days=20, trailing_stop_atr_mult=2.5, profit_target_atr_mult=4.5), slot="technology", sector="Information Technology")}), \
             patch("src.quote.service.build_analysis_frame", return_value=(analysis_frame, [])), \
             patch.object(service, "_load_shortlist_model_context", return_value=None):
            report = service.run(ticker="VSH")

        self.assertIsNone(report.current_price)
        self.assertEqual(report.current_price_source, "unavailable")
        self.assertIsNone(report.unrealized_pct)
        self.assertFalse(report.sell_now)
        self.assertIn("predates entry", report.notes[0])

    def test_quote_parser_accepts_ticker(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["quote", "CC"])
        self.assertEqual(args.command, "quote")
        self.assertEqual(args.ticker, "CC")
