from __future__ import annotations

from datetime import date
import unittest
from unittest.mock import patch

import pandas as pd

from src.sync.service import SyncService


def make_history_frame(ticker: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": [ticker, ticker],
            "date": [date(2024, 1, 2), date(2024, 1, 3)],
            "open": [10.0, 11.0],
            "high": [10.5, 11.5],
            "low": [9.5, 10.5],
            "close": [10.2, 11.2],
            "volume": [1000, 1200],
            "adj_close": [10.2, 11.2],
        }
    )


class FakeDBManager:
    def __init__(self) -> None:
        self.upserted: list[pd.DataFrame] = []
        self.status_updates: list[dict] = []
        self.all_tickers = ["AAA", "BBB"]
        self.liquidity_windows: dict[str, pd.DataFrame] = {}

    def initialize(self) -> None:
        return None

    def universe_is_empty(self) -> bool:
        return False

    def list_universe_tickers(self, active_only: bool = True) -> list[str]:
        return list(self.all_tickers)

    def build_fetch_plan(self, tickers, lookback_years: int = 5):
        return {date(2024, 1, 1): list(tickers)}

    def upsert_historical_frame(self, frame: pd.DataFrame) -> int:
        self.upserted.append(frame.copy())
        return len(frame.index)

    def set_ticker_status(self, ticker: str, *, is_active: bool, md_volume_30d=None) -> None:
        self.status_updates.append(
            {
                "ticker": ticker,
                "is_active": is_active,
                "md_volume_30d": md_volume_30d,
            }
        )

    def load_recent_liquidity_window(self, ticker: str, window: int = 30) -> pd.DataFrame:
        return self.liquidity_windows[ticker].copy()


class FakeMarketDataClient:
    def __init__(self, responses: dict[tuple[str, ...], object]) -> None:
        self.responses = responses
        self.calls: list[tuple[tuple[str, ...], date]] = []

    def download_daily_history(self, tickers: list[str], start_date: date) -> pd.DataFrame:
        key = tuple(tickers)
        self.calls.append((key, start_date))
        response = self.responses[key]
        if isinstance(response, Exception):
            raise response
        return response


class SyncServiceTests(unittest.TestCase):
    def test_sync_batch_falls_back_to_single_ticker_requests(self) -> None:
        db_manager = FakeDBManager()
        batch_raw = pd.DataFrame({"marker": [1]})
        batch_raw.attrs["name"] = "batch"
        single_raw = pd.DataFrame({"marker": [1]})
        single_raw.attrs["name"] = "single"
        client = FakeMarketDataClient(
            {
                ("AAA", "BBB"): batch_raw,
                ("BBB",): single_raw,
            }
        )
        service = SyncService(db_manager, market_data_client=client)

        def extract_side_effect(raw_frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
            name = raw_frame.attrs.get("name")
            if name == "batch" and ticker == "AAA":
                return make_history_frame("AAA")
            if name == "single" and ticker == "BBB":
                return make_history_frame("BBB")
            return pd.DataFrame()

        with patch("src.sync.service.extract_ticker_history", side_effect=extract_side_effect):
            inserted, failed = service._sync_batch(date(2024, 1, 1), ["AAA", "BBB"])

        self.assertEqual(inserted, 4)
        self.assertEqual(failed, set())
        self.assertEqual([call[0] for call in client.calls], [("AAA", "BBB"), ("BBB",)])
        self.assertEqual(len(db_manager.upserted), 2)

    def test_sync_batch_marks_unrecoverable_ticker_inactive(self) -> None:
        db_manager = FakeDBManager()
        batch_raw = pd.DataFrame({"marker": [1]})
        batch_raw.attrs["name"] = "batch"
        client = FakeMarketDataClient(
            {
                ("AAA", "BBB"): batch_raw,
                ("AAA",): ValueError("permanent failure"),
                ("BBB",): ValueError("permanent failure"),
            }
        )
        service = SyncService(db_manager, market_data_client=client)

        with patch("src.sync.service.extract_ticker_history", return_value=pd.DataFrame()):
            inserted, failed = service._sync_batch(date(2024, 1, 1), ["AAA", "BBB"])

        self.assertEqual(inserted, 0)
        self.assertEqual(failed, {"AAA", "BBB"})
        self.assertEqual(
            {(item["ticker"], item["is_active"]) for item in db_manager.status_updates},
            {("AAA", False), ("BBB", False)},
        )

    def test_liquidity_filter_deactivates_illiquid_names_and_stores_metric(self) -> None:
        db_manager = FakeDBManager()
        liquid_window = pd.DataFrame({"close": [100.0] * 30, "volume": [300000] * 30})
        illiquid_window = pd.DataFrame({"close": [10.0] * 30, "volume": [100000] * 30})
        db_manager.liquidity_windows = {
            "AAA": liquid_window,
            "BBB": illiquid_window,
        }
        service = SyncService(db_manager, market_data_client=FakeMarketDataClient({}))

        inactive = service._apply_liquidity_filter(["AAA", "BBB"])

        self.assertEqual(inactive, ["BBB"])
        self.assertEqual(
            db_manager.status_updates,
            [
                {"ticker": "AAA", "is_active": True, "md_volume_30d": 30000000.0},
                {"ticker": "BBB", "is_active": False, "md_volume_30d": 1000000.0},
            ],
        )
