from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import duckdb
import pandas as pd

from src.utils.ah_history import persist_ah_history, read_markdown_snapshot_rows, upsert_ah_history


class AhHistoryTests(unittest.TestCase):
    def test_upsert_preserves_complete_rows_when_markdown_is_partial(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "ah_history.duckdb"
            full = pd.DataFrame(
                [
                    {
                        "ticker": "AAA",
                        "snapshot_date": "2026-10-05",
                        "ah_price": 101.5,
                        "ah_volume": 200,
                        "rth_close": 100.0,
                        "source_row_count": 2,
                        "captured_at": "2026-10-05T21:00:00+00:00",
                    }
                ]
            )
            partial = pd.DataFrame(
                [
                    {
                        "ticker": "AAA",
                        "snapshot_date": "2026-10-05",
                        "ah_price": None,
                        "ah_volume": 300,
                        "rth_close": None,
                        "source_row_count": 20,
                        "captured_at": "2026-10-05T21:05:00+00:00",
                    }
                ]
            )

            self.assertEqual(upsert_ah_history(ah_duckdb_path=db_path, rows=full), 1)
            self.assertEqual(upsert_ah_history(ah_duckdb_path=db_path, rows=partial), 1)

            with duckdb.connect(str(db_path), read_only=True) as connection:
                row = connection.execute(
                    """
                    SELECT ah_price, ah_volume, rth_close, source_row_count, captured_at
                    FROM ah_snapshot_history
                    WHERE ticker = 'AAA'
                    """
                ).fetchone()

        self.assertEqual(row, (101.5, 300, 100.0, 20, "2026-10-05T21:05:00+00:00"))

    def test_markdown_recovery_reads_snapshot_date_and_top_move_volumes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            reports_dir = Path(tmpdir)
            (reports_dir / "extended_hours_snapshots.md").write_text(
                "\n".join(
                    [
                        "# Extended Hours Snapshot",
                        "",
                        "- snapshot_date: 2026-10-05",
                        "- requested_tickers: 255",
                        "- persisted_rows: 255",
                        "",
                        "## Top Relative Extended-Hours Moves",
                        "",
                        "- AAA: extended=+1.00%, sector=+0.10%, relative=+0.90%, volume=123",
                        "- BBB: extended=+0.80%, sector=+0.00%, relative=+0.80%, volume=0",
                    ]
                ),
                encoding="utf-8",
            )

            frame, reports_read = read_markdown_snapshot_rows(reports_dir)

        self.assertEqual(reports_read, 1)
        self.assertEqual(len(frame), 2)
        self.assertEqual(set(frame["ticker"]), {"AAA", "BBB"})
        self.assertEqual(int(frame.loc[frame["ticker"] == "AAA", "ah_volume"].iloc[0]), 123)
        self.assertEqual(int(frame["source_row_count"].iloc[0]), 255)

    def test_persist_reads_market_snapshot_and_writes_coverage_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            market_db = root / "market_data.duckdb"
            ah_db = root / "ah_history.duckdb"
            reports_dir = root / "reports"
            reports_dir.mkdir()
            with duckdb.connect(str(market_db)) as connection:
                connection.execute(
                    """
                    CREATE TABLE extended_hours_snapshots (
                        snapshot_date DATE,
                        ticker VARCHAR,
                        extended_price DOUBLE,
                        extended_volume BIGINT,
                        regular_close DOUBLE,
                        captured_at VARCHAR
                    )
                    """
                )
                connection.execute(
                    """
                    INSERT INTO extended_hours_snapshots VALUES
                    ('2026-10-05', 'AAA', 101.0, 500, 100.0, '2026-10-05T21:00:00+00:00'),
                    ('2026-10-05', 'BBB',  99.0,   0, 100.0, '2026-10-05T21:00:00+00:00')
                    """
                )

            result = persist_ah_history(
                market_duckdb_path=market_db,
                ah_duckdb_path=ah_db,
                reports_dir=reports_dir,
                output_path=reports_dir / "ah_history_setup.md",
            )
            report_text = result.output_path.read_text(encoding="utf-8")

        self.assertEqual(result.db_rows_read, 2)
        self.assertEqual(result.total_rows, 2)
        self.assertEqual(result.total_nights, 1)
        self.assertEqual(result.volume_coverage_tickers, 1)
        self.assertIn("- table: ah_snapshot_history", report_text)
        self.assertIn("- answer: 1", report_text)


if __name__ == "__main__":
    unittest.main()
