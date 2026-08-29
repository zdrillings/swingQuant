from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from src.schwab.ledger_sync import SchwabLedgerSyncService
from src.utils.strategy import ExitRules, ProductionStrategy


class FakeSchwabClient:
    def __init__(self, positions):
        self.positions = positions

    def list_positions(self, *, account_hash=None):
        return list(self.positions)


class FakeMarketDataClient:
    def download_intraday_history(self, tickers):
        raise RuntimeError("intraday unavailable")


class CorruptIntradayMarketDataClient:
    def download_intraday_history(self, tickers):
        return pd.DataFrame(
            [
                {
                    "ticker": "AAA",
                    "date": pd.Timestamp("2026-07-01"),
                    "open": 486.0,
                    "high": 506.0,
                    "low": 486.0,
                    "close": 13.0,
                    "volume": 100,
                    "adj_close": 13.0,
                }
            ]
        )


class FakeDB:
    def __init__(self):
        self.open_trades = [
            {
                "rowid": 1,
                "ticker": "AAA",
                "entry_date": "2026-06-01",
                "entry_price": 10.0,
                "entry_atr": None,
                "strategy_id": 1,
                "strategy_slot": "technology",
                "shares": 10,
                "max_price_seen": 12.0,
                "status": "open",
                "exit_date": None,
                "exit_price": None,
            },
            {
                "rowid": 2,
                "ticker": "BBB",
                "entry_date": "2026-06-01",
                "entry_price": 20.0,
                "entry_atr": None,
                "strategy_id": 1,
                "strategy_slot": "technology",
                "shares": 5,
                "max_price_seen": 22.0,
                "status": "open",
                "exit_date": None,
                "exit_price": None,
            },
        ]
        self.closed = []
        self.opened = []
        self.updated = []

    def initialize(self):
        return None

    def list_open_trades(self):
        return list(self.open_trades)

    def get_latest_open_trade(self, ticker):
        for trade in self.open_trades:
            if trade["ticker"] == ticker and trade["status"] == "open":
                return trade
        return None

    def close_trade(self, *, trade_rowid, exit_date, exit_price):
        self.closed.append((trade_rowid, exit_date, exit_price))
        for trade in self.open_trades:
            if trade["rowid"] == trade_rowid:
                trade["status"] = "closed"
                trade["exit_date"] = exit_date
                trade["exit_price"] = exit_price

    def open_trade(self, **kwargs):
        self.opened.append(kwargs)

    def update_open_trade_from_broker(self, trade_rowid, *, entry_price, shares, max_price_seen):
        self.updated.append((trade_rowid, entry_price, shares, max_price_seen))

    def list_universe_rows(self, active_only=False):
        return [
            {"ticker": "CCC", "sector": "Information Technology"},
            {"ticker": "BBB", "sector": "Information Technology"},
        ]

    def load_price_history(self, tickers):
        return pd.DataFrame(
            [
                {
                    "ticker": "AAA",
                    "date": pd.Timestamp("2026-07-01").date(),
                    "open": 11.0,
                    "high": 11.0,
                    "low": 11.0,
                    "close": 11.0,
                    "volume": 100,
                    "adj_close": 11.0,
                }
            ]
        )


class SchwabLedgerSyncServiceTests(unittest.TestCase):
    def test_sync_reconciles_ledger_to_schwab_positions_without_broker_writes(self) -> None:
        db = FakeDB()
        schwab = FakeSchwabClient(
            [
                {
                    "ticker": "BBB",
                    "quantity": 7,
                    "average_price": 21.0,
                    "market_value": 161.0,
                },
                {
                    "ticker": "CCC",
                    "quantity": 3,
                    "average_price": 30.0,
                    "market_value": 96.0,
                },
            ]
        )
        strategy = ProductionStrategy(
            strategy_id=10,
            promoted_at="2026-06-01T17:00:00",
            indicators={},
            exit_rules=ExitRules(0.05, 0.12, 20),
            slot="technology",
            sector="Information Technology",
        )

        with patch("src.schwab.ledger_sync.load_active_strategies", return_value={"technology": strategy}):
            report = SchwabLedgerSyncService(
                db,
                schwab_client=schwab,
                market_data_client=FakeMarketDataClient(),
            ).run()

        self.assertEqual(report.opened, 1)
        self.assertEqual(report.closed, 0)
        self.assertEqual(report.updated, 1)
        self.assertEqual(db.closed, [])
        self.assertTrue(any("warn AAA" in message for message in report.messages))
        self.assertEqual(db.updated, [(2, 21.0, 7, 23.0)])
        self.assertEqual(db.opened[0]["ticker"], "CCC")
        self.assertEqual(db.opened[0]["shares"], 3)
        self.assertEqual(db.opened[0]["entry_price"], 30.0)
        self.assertEqual(db.opened[0]["strategy_id"], 10)
        self.assertEqual(db.opened[0]["strategy_slot"], "technology")

    def test_dry_run_reports_actions_without_writing_ledger(self) -> None:
        db = FakeDB()
        schwab = FakeSchwabClient([])

        with patch("src.schwab.ledger_sync.load_active_strategies", return_value={}):
            report = SchwabLedgerSyncService(
                db,
                schwab_client=schwab,
                market_data_client=FakeMarketDataClient(),
            ).run(dry_run=True)

        self.assertEqual(report.closed, 0)
        self.assertTrue(any("warn AAA" in message for message in report.messages))
        self.assertTrue(any("warn BBB" in message for message in report.messages))
        self.assertEqual(db.closed, [])
        self.assertEqual(db.opened, [])
        self.assertEqual(db.updated, [])

    def test_close_missing_explicitly_closes_ledger_only_positions(self) -> None:
        db = FakeDB()
        schwab = FakeSchwabClient([])

        with patch("src.schwab.ledger_sync.load_active_strategies", return_value={}):
            report = SchwabLedgerSyncService(
                db,
                schwab_client=schwab,
                market_data_client=FakeMarketDataClient(),
            ).run(close_missing=True)

        self.assertEqual(report.closed, 2)
        self.assertEqual(len(db.closed), 2)

    def test_close_missing_rejects_intraday_exit_price_outside_ohlc_range(self) -> None:
        db = FakeDB()
        db.open_trades = [
            {
                "rowid": 1,
                "ticker": "AAA",
                "entry_date": "2026-06-01",
                "entry_price": 483.78,
                "entry_atr": None,
                "strategy_id": 1,
                "strategy_slot": "technology",
                "shares": 13,
                "max_price_seen": 506.0,
                "status": "open",
                "exit_date": None,
                "exit_price": None,
            }
        ]
        schwab = FakeSchwabClient([])

        with patch("src.schwab.ledger_sync.load_active_strategies", return_value={}):
            report = SchwabLedgerSyncService(
                db,
                schwab_client=schwab,
                market_data_client=CorruptIntradayMarketDataClient(),
            ).run(close_missing=True)

        self.assertEqual(report.closed, 1)
        self.assertEqual(db.closed[0][2], 11.0)
        self.assertTrue(any("source=stored_close" in message for message in report.messages))

    def test_sync_skips_etfs_by_default(self) -> None:
        db = FakeDB()
        schwab = FakeSchwabClient(
            [
                {
                    "ticker": "QQQ",
                    "asset_type": "COLLECTIVE_INVESTMENT",
                    "instrument_type": "EXCHANGE_TRADED_FUND",
                    "quantity": 10,
                    "average_price": 500.0,
                    "market_value": 5_100.0,
                }
            ]
        )

        with patch("src.schwab.ledger_sync.load_active_strategies", return_value={}):
            report = SchwabLedgerSyncService(
                db,
                schwab_client=schwab,
                market_data_client=FakeMarketDataClient(),
            ).run(dry_run=True)

        self.assertEqual(report.opened, 0)
        self.assertTrue(any("skip QQQ: ignored fund/ETF position" in message for message in report.messages))

    def test_ignore_ticker_protects_ledger_only_position_from_close(self) -> None:
        db = FakeDB()
        db.open_trades = [
            {
                "rowid": 1,
                "ticker": "TWLO",
                "entry_date": "2026-06-01",
                "entry_price": 200.0,
                "entry_atr": None,
                "strategy_id": 1,
                "strategy_slot": "technology",
                "shares": 5,
                "max_price_seen": 210.0,
                "status": "open",
                "exit_date": None,
                "exit_price": None,
            }
        ]
        schwab = FakeSchwabClient([])

        with patch("src.schwab.ledger_sync.load_active_strategies", return_value={}):
            report = SchwabLedgerSyncService(
                db,
                schwab_client=schwab,
                market_data_client=FakeMarketDataClient(),
            ).run(dry_run=True, ignore_tickers=("TWLO",))

        self.assertEqual(report.closed, 0)

    def test_sync_ignores_subcent_average_price_differences(self) -> None:
        db = FakeDB()
        db.open_trades = [
            {
                "rowid": 46,
                "ticker": "TWLO",
                "entry_date": "2026-07-01",
                "entry_price": 208.71,
                "entry_atr": None,
                "strategy_id": 1,
                "strategy_slot": "technology",
                "shares": 50,
                "max_price_seen": 210.17,
                "status": "open",
                "exit_date": None,
                "exit_price": None,
            }
        ]
        schwab = FakeSchwabClient(
            [
                {
                    "ticker": "TWLO",
                    "quantity": 50,
                    "average_price": 208.714,
                    "market_value": 10_508.50,
                }
            ]
        )

        with patch("src.schwab.ledger_sync.load_active_strategies", return_value={}):
            report = SchwabLedgerSyncService(
                db,
                schwab_client=schwab,
                market_data_client=FakeMarketDataClient(),
            ).run()

        self.assertEqual(report.updated, 0)
        self.assertEqual(db.updated, [])


if __name__ == "__main__":
    unittest.main()
