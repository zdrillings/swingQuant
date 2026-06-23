from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src.scan.analyst_data import AnalystContext, AnalystDataClient, AnalystRevisionContext
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

    def test_revision_table_parser_preserves_period_rows(self) -> None:
        client = AnalystDataClient()
        raw = pd.DataFrame(
            [
                {"avg": 1.23, "upLast7days": 2, "downLast7days": 0},
                {"avg": 5.67, "upLast7days": 1, "downLast7days": 1},
            ],
            index=["0q", "+1q"],
        )

        records = client._records_from_table(raw)

        self.assertEqual(records[0]["period"], "0q")
        self.assertEqual(records[0]["avg"], 1.23)
        self.assertEqual(records[0]["upLast7days"], 2)
        self.assertEqual(records[1]["period"], "+1q")


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
            fake_client.revision_contexts = {
                "AAA": AnalystRevisionContext(
                    ticker="AAA",
                    earnings_estimate=[{"period": "0q", "avg": 1.23}],
                    revenue_estimate=[{"period": "0q", "avg": 100.0}],
                    eps_trend=[{"period": "0q", "current": 1.23}],
                    eps_revisions=[{"period": "0q", "upLast7days": 2}],
                    growth_estimates=[],
                    upgrades_downgrades=[{"date": "2026-06-23", "action": "up"}],
                )
            }

            report = AnalystSnapshotService(fake_db, analyst_data_client=fake_client).run(
                snapshot_date="2026-06-23",
                source="research",
                top=2,
                tickers=["CCC", "AAA"],
            )
            self.assertTrue(report.output_path.exists())

        self.assertEqual(report.requested_tickers, 3)
        self.assertEqual(report.persisted_rows, 3)
        self.assertEqual(report.persisted_revision_rows, 3)
        self.assertEqual(report.rows_with_targets, 1)
        self.assertEqual(report.rows_with_recommendations, 1)
        self.assertEqual(report.rows_with_estimates, 1)
        self.assertEqual(report.rows_with_revisions, 1)
        self.assertEqual(fake_client.requested_tickers, ["AAA", "BBB", "CCC"])
        self.assertEqual(fake_client.requested_revision_tickers, ["AAA", "BBB", "CCC"])
        self.assertEqual(fake_db.snapshot_date, "2026-06-23")
        self.assertEqual(fake_db.provider, "yfinance")
        persisted_by_ticker = {row["ticker"]: row for row in fake_db.persisted_rows}
        self.assertEqual(persisted_by_ticker["AAA"]["target_mean"], 65.0)
        self.assertIsNone(persisted_by_ticker["BBB"]["target_mean"])
        self.assertEqual(persisted_by_ticker["BBB"]["details"], {"source": "research", "has_context": False})
        revision_by_ticker = {row["ticker"]: row for row in fake_db.persisted_revision_rows}
        self.assertEqual(revision_by_ticker["AAA"]["eps_revisions"], [{"period": "0q", "upLast7days": 2}])
        self.assertEqual(revision_by_ticker["BBB"]["eps_revisions"], [])


class FakeAnalystSnapshotDatabase:
    def __init__(self, root: Path) -> None:
        self.paths = type("Paths", (), {"reports_dir": root / "reports"})()
        self.persisted_rows: list[dict] = []
        self.persisted_revision_rows: list[dict] = []
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

    def replace_analyst_revision_snapshots(self, *, snapshot_date: str, provider: str, rows):
        self.snapshot_date = snapshot_date
        self.provider = provider
        self.persisted_revision_rows = list(rows)
        return len(self.persisted_revision_rows)


class FakeAnalystDataClient:
    def __init__(self, contexts: dict[str, AnalystContext]) -> None:
        self.contexts = contexts
        self.revision_contexts: dict[str, AnalystRevisionContext] = {}
        self.requested_tickers: list[str] = []
        self.requested_revision_tickers: list[str] = []

    def load_contexts(self, tickers: list[str]) -> dict[str, AnalystContext]:
        self.requested_tickers = list(tickers)
        return {ticker: context for ticker, context in self.contexts.items() if ticker in tickers}

    def load_revision_contexts(self, tickers: list[str]) -> dict[str, AnalystRevisionContext]:
        self.requested_revision_tickers = list(tickers)
        return {ticker: context for ticker, context in self.revision_contexts.items() if ticker in tickers}


if __name__ == "__main__":
    unittest.main()
