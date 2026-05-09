from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
import json
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
    run_id INTEGER,
    strategy_id INTEGER NOT NULL,
    params_json TEXT NOT NULL,
    norm_score REAL,
    profit_factor REAL,
    expectancy REAL,
    alpha_vs_spy REAL,
    alpha_vs_sector REAL,
    mdd REAL,
    win_rate REAL,
    trade_count INTEGER
);

CREATE TABLE IF NOT EXISTS Active_Trades (
    ticker TEXT NOT NULL,
    entry_date TEXT NOT NULL,
    entry_price REAL NOT NULL,
    entry_atr REAL,
    strategy_id INTEGER,
    strategy_slot TEXT,
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


@dataclass(frozen=True)
class BacktestResultRow:
    run_id: int | None
    strategy_id: int
    params_json: str
    norm_score: float | None
    profit_factor: float
    expectancy: float
    mdd: float
    win_rate: float
    trade_count: int | None = None
    alpha_vs_spy: float | None = None
    alpha_vs_sector: float | None = None


@dataclass(frozen=True)
class ActiveTradeRow:
    ticker: str
    entry_date: str
    entry_price: float
    entry_atr: float | None
    strategy_id: int | None
    strategy_slot: str | None
    shares: int
    max_price_seen: float
    status: str
    exit_date: str | None = None
    exit_price: float | None = None


class DatabaseManager:
    def __init__(self, paths: AppPaths | None = None) -> None:
        self.paths = paths or get_settings().paths

    def initialize(self) -> None:
        self.paths.data_dir.mkdir(parents=True, exist_ok=True)
        self.paths.reports_dir.mkdir(parents=True, exist_ok=True)
        self.paths.logs_dir.mkdir(parents=True, exist_ok=True)

        with self.sqlite_connection() as sqlite_conn:
            sqlite_conn.executescript(SQLITE_SCHEMA)
            self._migrate_sqlite_schema(sqlite_conn)

        with self.duckdb_connection() as duckdb_conn:
            duckdb_conn.execute(HISTORICAL_TABLE_SCHEMA)

    def sqlite_connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.paths.sqlite_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _migrate_sqlite_schema(self, connection: sqlite3.Connection) -> None:
        backtest_columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(Backtest_Results)").fetchall()
        }
        if "run_id" not in backtest_columns:
            connection.execute("ALTER TABLE Backtest_Results ADD COLUMN run_id INTEGER")
        if "trade_count" not in backtest_columns:
            connection.execute("ALTER TABLE Backtest_Results ADD COLUMN trade_count INTEGER")
        if "alpha_vs_spy" not in backtest_columns:
            connection.execute("ALTER TABLE Backtest_Results ADD COLUMN alpha_vs_spy REAL")
        if "alpha_vs_sector" not in backtest_columns:
            connection.execute("ALTER TABLE Backtest_Results ADD COLUMN alpha_vs_sector REAL")
        active_trade_columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(Active_Trades)").fetchall()
        }
        if "entry_atr" not in active_trade_columns:
            connection.execute("ALTER TABLE Active_Trades ADD COLUMN entry_atr REAL")
        if "strategy_id" not in active_trade_columns:
            connection.execute("ALTER TABLE Active_Trades ADD COLUMN strategy_id INTEGER")
        if "strategy_slot" not in active_trade_columns:
            connection.execute("ALTER TABLE Active_Trades ADD COLUMN strategy_slot TEXT")

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

    def load_latest_rows(self, tickers: Iterable[str]):
        ticker_list = list(tickers)
        if not ticker_list:
            return []

        placeholders = ", ".join(["?"] * len(ticker_list))
        query = f"""
            WITH latest_per_ticker AS (
                SELECT ticker, MAX(date) AS latest_date
                FROM historical_ohlcv
                WHERE ticker IN ({placeholders})
                GROUP BY ticker
            )
            SELECT h.ticker, h.date, h.open, h.high, h.low, h.close, h.volume, h.adj_close
            FROM historical_ohlcv h
            INNER JOIN latest_per_ticker l
                ON h.ticker = l.ticker AND h.date = l.latest_date
            ORDER BY h.ticker
        """
        with self.duckdb_connection() as connection:
            return connection.execute(query, ticker_list).df()

    def load_recent_highs(self, ticker: str, limit: int = 2):
        query = """
            SELECT date, high
            FROM historical_ohlcv
            WHERE ticker = ?
            ORDER BY date DESC
            LIMIT ?
        """
        with self.duckdb_connection() as connection:
            return connection.execute(query, [ticker, limit]).df()

    def list_universe_rows(self, active_only: bool = True) -> list[sqlite3.Row]:
        query = "SELECT ticker, sector, is_active, md_volume_30d FROM Universe"
        if active_only:
            query += " WHERE is_active = 1"
        query += " ORDER BY md_volume_30d DESC NULLS LAST, ticker ASC"
        with self.sqlite_connection() as connection:
            return list(connection.execute(query).fetchall())

    def next_strategy_id(self) -> int:
        with self.sqlite_connection() as connection:
            row = connection.execute(
                "SELECT COALESCE(MAX(strategy_id), 0) + 1 AS next_id FROM Backtest_Results"
            ).fetchone()
        return int(row["next_id"])

    def next_run_id(self) -> int:
        with self.sqlite_connection() as connection:
            row = connection.execute(
                "SELECT COALESCE(MAX(run_id), 0) + 1 AS next_id FROM Backtest_Results"
            ).fetchone()
        return int(row["next_id"])

    def latest_run_id(self) -> int | None:
        with self.sqlite_connection() as connection:
            row = connection.execute("SELECT MAX(run_id) AS run_id FROM Backtest_Results").fetchone()
        return int(row["run_id"]) if row["run_id"] is not None else None

    def insert_backtest_results(self, results: Iterable[BacktestResultRow]) -> int:
        rows = [
            (
                result.run_id,
                result.strategy_id,
                result.params_json,
                result.norm_score,
                result.profit_factor,
                result.expectancy,
                result.alpha_vs_spy,
                result.alpha_vs_sector,
                result.mdd,
                result.win_rate,
                result.trade_count,
            )
            for result in results
        ]
        if not rows:
            return 0

        with self.sqlite_connection() as connection:
            connection.executemany(
                """
                INSERT INTO Backtest_Results (
                    run_id, strategy_id, params_json, norm_score, profit_factor, expectancy, alpha_vs_spy, alpha_vs_sector, mdd, win_rate, trade_count
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
        return len(rows)

    def list_backtest_results(self, run_id: int | None = None) -> list[sqlite3.Row]:
        query = """
            SELECT id, run_id, strategy_id, params_json, norm_score, profit_factor, expectancy, alpha_vs_spy, alpha_vs_sector, mdd, win_rate, trade_count
            FROM Backtest_Results
        """
        params: tuple = ()
        if run_id is not None:
            query += " WHERE run_id = ?"
            params = (run_id,)
        query += " ORDER BY id ASC"
        with self.sqlite_connection() as connection:
            return list(
                connection.execute(query, params).fetchall()
            )

    def update_backtest_norm_scores(self, scored_rows: Iterable[tuple[int, float]]) -> None:
        rows = list(scored_rows)
        if not rows:
            return
        with self.sqlite_connection() as connection:
            connection.executemany(
                "UPDATE Backtest_Results SET norm_score = ? WHERE id = ?",
                [(score, row_id) for row_id, score in rows],
            )

    def get_backtest_result(self, row_id: int) -> sqlite3.Row | None:
        with self.sqlite_connection() as connection:
            return connection.execute(
                """
                SELECT id, run_id, strategy_id, params_json, norm_score, profit_factor, expectancy, alpha_vs_spy, alpha_vs_sector, mdd, win_rate, trade_count
                FROM Backtest_Results
                WHERE id = ?
                """,
                (row_id,),
            ).fetchone()

    def open_trade(
        self,
        *,
        ticker: str,
        entry_date: str,
        entry_price: float,
        entry_atr: float | None,
        strategy_id: int | None,
        strategy_slot: str | None,
        shares: int,
        max_price_seen: float,
    ) -> None:
        with self.sqlite_connection() as connection:
            connection.execute(
                """
                INSERT INTO Active_Trades (
                    ticker, entry_date, entry_price, entry_atr, strategy_id, strategy_slot, shares, max_price_seen, status
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'open')
                """,
                (ticker, entry_date, entry_price, entry_atr, strategy_id, strategy_slot, shares, max_price_seen),
            )

    def list_open_trades(self) -> list[sqlite3.Row]:
        with self.sqlite_connection() as connection:
            return list(
                connection.execute(
                    """
                    SELECT ticker, entry_date, entry_price, entry_atr, strategy_id, strategy_slot, shares, max_price_seen, status, exit_date, exit_price
                    FROM Active_Trades
                    WHERE status = 'open'
                    ORDER BY entry_date ASC, ticker ASC
                    """
                ).fetchall()
            )

    def get_latest_open_trade(self, ticker: str) -> sqlite3.Row | None:
        with self.sqlite_connection() as connection:
            return connection.execute(
                """
                SELECT rowid, ticker, entry_date, entry_price, entry_atr, strategy_id, strategy_slot, shares, max_price_seen, status, exit_date, exit_price
                FROM Active_Trades
                WHERE ticker = ? AND status = 'open'
                ORDER BY entry_date DESC, rowid DESC
                LIMIT 1
                """,
                (ticker,),
            ).fetchone()

    def get_latest_trade(self, ticker: str) -> sqlite3.Row | None:
        with self.sqlite_connection() as connection:
            return connection.execute(
                """
                SELECT rowid, ticker, entry_date, entry_price, entry_atr, strategy_id, strategy_slot, shares, max_price_seen, status, exit_date, exit_price
                FROM Active_Trades
                WHERE ticker = ?
                ORDER BY entry_date DESC, rowid DESC
                LIMIT 1
                """,
                (ticker,),
            ).fetchone()

    def update_trade_max_price(self, trade_rowid: int, max_price_seen: float) -> None:
        with self.sqlite_connection() as connection:
            connection.execute(
                "UPDATE Active_Trades SET max_price_seen = ? WHERE rowid = ?",
                (max_price_seen, trade_rowid),
            )

    def assign_trade_strategy(self, trade_rowid: int, *, strategy_id: int, strategy_slot: str) -> None:
        with self.sqlite_connection() as connection:
            connection.execute(
                "UPDATE Active_Trades SET strategy_id = ?, strategy_slot = ? WHERE rowid = ?",
                (strategy_id, strategy_slot, trade_rowid),
            )

    def close_trade(
        self,
        *,
        trade_rowid: int,
        exit_date: str,
        exit_price: float,
    ) -> None:
        with self.sqlite_connection() as connection:
            connection.execute(
                """
                UPDATE Active_Trades
                SET status = 'closed', exit_date = ?, exit_price = ?
                WHERE rowid = ?
                """,
                (exit_date, exit_price, trade_rowid),
            )


def create_default_manager() -> DatabaseManager:
    return DatabaseManager()
