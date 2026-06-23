from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src.scan.analyst_data import AnalystContext, AnalystDataClient
from src.scan.analyst_snapshot_service import AnalystSnapshotService


class AnalystDataClientTests(unittest.TestCase):
    def test_price_target_parser_accepts_yfinance_style_payloads(self) -> None:
        client = AnalystDataClient()

        class FakeInstrument:
            def get_analyst_price_targets(self):
                return {
                    "targetMeanPrice": 65.0,
                    "targetMedianPrice": 63.0,
                    "targetLowPrice": 45.0,
                    "targetHighPrice": 80.0,
                    "numberOfAnalystOpinions": 12,
                }
            def get_recommendations_summary(self):
                return pd.DataFrame(
                    [
                        {
                            "period": "0m",
                            "strongBuy": 2,
                            "buy": 6,
                            "hold": 4,
                            "sell": 0,
                            "strongSell": 0,
                        }
                    ]
                )

        targets = client._fetch_price_targets(FakeInstrument())
        recommendation = client._fetch_recommendation_summary(FakeInstrument())

        self.assertEqual(targets["mean"], 65.0)
        self.assertEqual(targets["median"], 63.0)
        self.assertEqual(targets["low"], 45.0)
        self.assertEqual(targets["high"], 80.0)
        self.assertEqual(targets["numberOfAnalystOpinions"], 12)
        self.assertEqual(recommendation, "2 strong buy, 6 buy, 4 hold")


class AnalystSnapshotServiceTests(unittest.TestCase):
    def test_capture_persists_all_requested_tickers_including_missing_contexts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_db = FakeAnalystSnapshotDatabase(Path(tmpdir))
            fake_client = FakeAnalystDataClient(
                {
                    "AAA": AnalystContext(
                        ticker="AAA",
                        target_mean=65.0,
                        target_median=63.0,
                        target_low=45.0,
                        target_high=80.0,
                        analyst_count=12,
                        recommendation="2 strong buy, 6 buy, 4 hold",
                    )
                }
            )

            report = AnalystSnapshotService(fake_db, analyst_data_client=fake_client).run(
                snapshot_date="2026-06-23",
                source="research",
                top=2,
                tickers=["CCC", "AAA"],
            )
            self.assertTrue(report.output_path.exists())

        self.assertEqual(report.requested_tickers, 3)
        self.assertEqual(report.persisted_rows, 3)
        self.assertEqual(report.rows_with_targets, 1)
        self.assertEqual(report.rows_with_recommendations, 1)
        self.assertEqual(fake_client.requested_tickers, ["AAA", "BBB", "CCC"])
        self.assertEqual(fake_db.snapshot_date, "2026-06-23")
        self.assertEqual(fake_db.provider, "yfinance")
        persisted_by_ticker = {row["ticker"]: row for row in fake_db.persisted_rows}
        self.assertEqual(persisted_by_ticker["AAA"]["target_mean"], 65.0)
        self.assertIsNone(persisted_by_ticker["BBB"]["target_mean"])
        self.assertEqual(persisted_by_ticker["BBB"]["details"], {"source": "research", "has_context": False})


class FakeAnalystSnapshotDatabase:
    def __init__(self, root: Path) -> None:
        self.paths = type("Paths", (), {"reports_dir": root / "reports"})()
        self.persisted_rows: list[dict] = []
        self.snapshot_date: str | None = None
        self.provider: str | None = None

    def initialize(self) -> None:
        return None

    def list_research_universe(self, limit: int = 250):
        rows = [
            {"ticker": "AAA", "sector": "Industrials", "md_volume_30d": 30_000_000},
            {"ticker": "BBB", "sector": "Industrials", "md_volume_30d": 20_000_000},
            {"ticker": "DDD", "sector": "Industrials", "md_volume_30d": 10_000_000},
        ]
        return rows[:limit]

    def replace_analyst_snapshots(self, *, snapshot_date: str, provider: str, rows):
        self.snapshot_date = snapshot_date
        self.provider = provider
        self.persisted_rows = list(rows)
        return len(self.persisted_rows)


class FakeAnalystDataClient:
    def __init__(self, contexts: dict[str, AnalystContext]) -> None:
        self.contexts = contexts
        self.requested_tickers: list[str] = []

    def load_contexts(self, tickers: list[str]) -> dict[str, AnalystContext]:
        self.requested_tickers = list(tickers)
        return {ticker: context for ticker, context in self.contexts.items() if ticker in tickers}


if __name__ == "__main__":
    unittest.main()
