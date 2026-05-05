from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
import sqlite3

from src.settings import AppPaths, get_settings


HISTORICAL_TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS historical_ohlcv (
    ticker VARCHAR NOT NULL,
    date DATE NOT NULL,
    open DOUBLE,
    high DOUBLE,
    low DOUBLE,
    close DOUBLE,
    volume BIGINT,
    adj_close DOUBLE,
    PRIMARY KEY (ticker, date)
);
CREATE INDEX IF NOT EXISTS idx_historical_ohlcv_date
ON historical_ohlcv (date);
"""

SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS Universe (
    ticker TEXT PRIMARY KEY,
    sector TEXT NOT NULL,
    is_active INTEGER NOT NULL DEFAULT 1,
    md_volume_30d REAL
);

CREATE TABLE IF NOT EXISTS Backtest_Results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_id INTEGER NOT NULL,
    params_json TEXT NOT NULL,
    norm_score REAL,
    profit_factor REAL,
    expectancy REAL,
    mdd REAL,
    win_rate REAL
);

CREATE TABLE IF NOT EXISTS Active_Trades (
    ticker TEXT NOT NULL,
    entry_date TEXT NOT NULL,
    entry_price REAL NOT NULL,
    shares INTEGER NOT NULL,
    max_price_seen REAL NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('open', 'closed')),
    exit_date TEXT,
    exit_price REAL
);
"""


@dataclass(frozen=True)
class UniverseRow:
    ticker: str
    sector: str
    is_active: bool = True
    md_volume_30d: float | None = None


class DatabaseManager:
    def __init__(self, paths: AppPaths | None = None) -> None:
        self.paths = paths or get_settings().paths

    def initialize(self) -> None:
        self.paths.data_dir.mkdir(parents=True, exist_ok=True)
        self.paths.reports_dir.mkdir(parents=True, exist_ok=True)
        self.paths.logs_dir.mkdir(parents=True, exist_ok=True)

        with self.sqlite_connection() as sqlite_conn:
            sqlite_conn.executescript(SQLITE_SCHEMA)

        with self.duckdb_connection() as duckdb_conn:
            duckdb_conn.execute(HISTORICAL_TABLE_SCHEMA)

    def sqlite_connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.paths.sqlite_path)
        connection.row_factory = sqlite3.Row
        return connection

    def duckdb_connection(self):
        import duckdb

        return duckdb.connect(str(self.paths.duckdb_path))

    def universe_is_empty(self) -> bool:
        with self.sqlite_connection() as connection:
            row = connection.execute("SELECT COUNT(*) AS count FROM Universe").fetchone()
        return bool(row["count"] == 0)

    def bootstrap_universe(self, members: Iterable[UniverseRow]) -> int:
        rows = [(member.ticker, member.sector, int(member.is_active)) for member in members]
        with self.sqlite_connection() as connection:
            connection.executemany(
                """
                INSERT INTO Universe (ticker, sector, is_active)
                VALUES (?, ?, ?)
                ON CONFLICT(ticker) DO UPDATE SET
                    sector = excluded.sector,
                    is_active = excluded.is_active
                """,
                rows,
            )
        return len(rows)

    def list_universe_tickers(self, active_only: bool = True) -> list[str]:
        query = "SELECT ticker FROM Universe"
        if active_only:
            query += " WHERE is_active = 1"
        query += " ORDER BY ticker"
        with self.sqlite_connection() as connection:
            rows = connection.execute(query).fetchall()
        return [row["ticker"] for row in rows]

    def list_research_universe(self, limit: int = 250) -> list[sqlite3.Row]:
        with self.sqlite_connection() as connection:
            rows = connection.execute(
                """
                SELECT ticker, sector, md_volume_30d
                FROM Universe
                WHERE is_active = 1 AND md_volume_30d IS NOT NULL
                ORDER BY md_volume_30d DESC, ticker ASC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return list(rows)

    def set_ticker_status(
        self,
        ticker: str,
        *,
        is_active: bool,
        md_volume_30d: float | None = None,
    ) -> None:
        with self.sqlite_connection() as connection:
            if md_volume_30d is None:
                connection.execute(
                    "UPDATE Universe SET is_active = ? WHERE ticker = ?",
                    (int(is_active), ticker),
                )
            else:
                connection.execute(
                    """
                    UPDATE Universe
                    SET is_active = ?, md_volume_30d = ?
                    WHERE ticker = ?
                    """,
                    (int(is_active), md_volume_30d, ticker),
                )

    def latest_available_dates(self, tickers: Iterable[str]) -> dict[str, date | None]:
        ticker_list = list(tickers)
        if not ticker_list:
            return {}

        placeholders = ", ".join(["?"] * len(ticker_list))
        query = f"""
            SELECT ticker, MAX(date) AS latest_date
            FROM historical_ohlcv
            WHERE ticker IN ({placeholders})
            GROUP BY ticker
        """

        with self.duckdb_connection() as connection:
            rows = connection.execute(query, ticker_list).fetchall()

        latest_dates = {ticker: None for ticker in ticker_list}
        for ticker, latest_date in rows:
            latest_dates[str(ticker)] = latest_date
        return latest_dates

    def build_fetch_plan(self, tickers: Iterable[str], lookback_years: int = 5) -> dict[date, list[str]]:
        today = date.today()
        default_start = today - timedelta(days=lookback_years * 365)
        latest_dates = self.latest_available_dates(tickers)

        plan: dict[date, list[str]] = defaultdict(list)
        for ticker, latest_date in latest_dates.items():
            if latest_date is None:
                start_date = default_start
            else:
                start_date = latest_date + timedelta(days=1)
            if start_date <= today:
                plan[start_date].append(ticker)
        return dict(plan)

    def upsert_historical_frame(self, frame) -> int:
        if frame.empty:
            return 0

        expected_columns = [
            "ticker",
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "adj_close",
        ]
        frame = frame[expected_columns].copy()

        with self.duckdb_connection() as connection:
            connection.register("incoming_ohlcv", frame)
            connection.execute(
                """
                INSERT OR IGNORE INTO historical_ohlcv
                SELECT ticker, date, open, high, low, close, volume, adj_close
                FROM incoming_ohlcv
                """
            )
            inserted = connection.execute("SELECT COUNT(*) FROM incoming_ohlcv").fetchone()[0]
            connection.unregister("incoming_ohlcv")
        return int(inserted)

    def load_recent_liquidity_window(self, ticker: str, window: int = 30):
        query = """
            SELECT date, close, volume
            FROM historical_ohlcv
            WHERE ticker = ?
            ORDER BY date DESC
            LIMIT ?
        """
        with self.duckdb_connection() as connection:
            return connection.execute(query, [ticker, window]).df()

    def load_price_history(self, tickers: Iterable[str]):
        ticker_list = list(tickers)
        if not ticker_list:
            raise ValueError("At least one ticker is required")

        placeholders = ", ".join(["?"] * len(ticker_list))
        query = f"""
            SELECT ticker, date, open, high, low, close, volume, adj_close
            FROM historical_ohlcv
            WHERE ticker IN ({placeholders})
            ORDER BY date ASC, ticker ASC
        """
        with self.duckdb_connection() as connection:
            return connection.execute(query, ticker_list).df()


def create_default_manager() -> DatabaseManager:
    return DatabaseManager()
