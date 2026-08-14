from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src.scan.extended_hours_snapshot_service import ExtendedHoursSnapshotService


class ExtendedHoursSnapshotServiceTests(unittest.TestCase):
    def test_capture_persists_relative_extended_hours_returns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_db = FakeExtendedHoursDatabase(Path(tmpdir))
            fake_market = FakeExtendedHoursMarketDataClient()

            report = ExtendedHoursSnapshotService(fake_db, market_data_client=fake_market).run(
                snapshot_date="2026-07-03",
                source="research",
                top=2,
            )
            self.assertTrue(report.output_path.exists())

        self.assertEqual(report.persisted_rows, 2)
        self.assertEqual(report.rows_with_extended_price, 2)
        by_ticker = {row["ticker"]: row for row in fake_db.persisted_rows}
        self.assertAlmostEqual(by_ticker["AAA"]["extended_return"], 0.03, places=6)
        self.assertAlmostEqual(by_ticker["AAA"]["sector_etf_extended_return"], 0.01, places=6)
        self.assertAlmostEqual(by_ticker["AAA"]["relative_extended_return"], 0.02, places=6)
        self.assertEqual(by_ticker["AAA"]["extended_volume"], 1500)
        self.assertIn("XLK", fake_market.requested_tickers)


class FakeExtendedHoursDatabase:
    def __init__(self, root: Path) -> None:
        self.paths = type("Paths", (), {"reports_dir": root / "reports"})()
        self.persisted_rows: list[dict] = []

    def initialize(self) -> None:
        return None

    def list_research_universe(self, limit: int = 250):
        return [
            {"ticker": "AAA", "sector": "Information Technology", "md_volume_30d": 10_000_000},
            {"ticker": "BBB", "sector": "Information Technology", "md_volume_30d": 9_000_000},
        ][:limit]

    def list_universe_rows(self, active_only: bool = False):
        return [
            {"ticker": "AAA", "sector": "Information Technology"},
            {"ticker": "BBB", "sector": "Information Technology"},
        ]

    def replace_extended_hours_snapshots(self, *, snapshot_date: str, rows):
        self.snapshot_date = snapshot_date
        self.persisted_rows = list(rows)
        return len(self.persisted_rows)


class FakeExtendedHoursMarketDataClient:
    def __init__(self) -> None:
        self.requested_tickers: list[str] = []

    def download_intraday_history(self, tickers: list[str], *, include_prepost: bool = False) -> pd.DataFrame:
        self.requested_tickers.extend(tickers)
        self.include_prepost = include_prepost
        timestamps = pd.to_datetime(
            [
                "2026-07-03 15:59:00-04:00",
                "2026-07-03 16:00:00-04:00",
                "2026-07-03 16:01:00-04:00",
                "2026-07-03 19:30:00-04:00",
            ]
        )
        data = {}
        prices = {
            "AAA": [99.0, 100.0, 102.0, 103.0],
            "BBB": [50.0, 50.0, 49.0, 49.5],
            "XLK": [200.0, 200.0, 201.0, 202.0],
            "SPY": [500.0, 500.0, 500.0, 500.5],
            "QQQ": [450.0, 450.0, 450.0, 451.0],
        }
        for ticker in tickers:
            ticker_prices = prices.get(ticker, [10.0, 10.0, 10.0, 10.0])
            data[(ticker, "Close")] = ticker_prices
            data[(ticker, "Volume")] = [1000, 1000, 1000, 500]
        return pd.DataFrame(data, index=timestamps)


if __name__ == "__main__":
    unittest.main()
