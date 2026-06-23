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

UNIVERSE_SNAPSHOTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS universe_daily_snapshots (
    snapshot_date DATE NOT NULL,
    ticker VARCHAR NOT NULL,
    sector VARCHAR,
    sub_industry VARCHAR,
    subindustry_benchmark VARCHAR,
    regime_etf VARCHAR,
    regime_green BOOLEAN,
    md_volume_30d DOUBLE,
    adj_close DOUBLE,
    atr_14 DOUBLE,
    relative_strength_index_vs_spy DOUBLE,
    relative_strength_index_vs_qqq DOUBLE,
    relative_strength_index_vs_xlk DOUBLE,
    relative_strength_index_vs_subindustry DOUBLE,
    roc_63 DOUBLE,
    roc_126 DOUBLE,
    vol_alpha DOUBLE,
    sma_200_dist DOUBLE,
    sma_50_dist DOUBLE,
    rsi_14 DOUBLE,
    days_to_next_earnings DOUBLE,
    days_since_last_earnings DOUBLE,
    last_earnings_gap_pct DOUBLE,
    last_earnings_volume_ratio_20 DOUBLE,
    last_earnings_open_vs_20d_high DOUBLE,
    close_vs_last_earnings_close DOUBLE,
    avg_abs_gap_pct_20 DOUBLE,
    max_gap_down_pct_60 DOUBLE,
    distance_above_20d_high DOUBLE,
    base_range_pct_20 DOUBLE,
    base_atr_contraction_20 DOUBLE,
    base_volume_dryup_ratio_20 DOUBLE,
    breakout_volume_ratio_50 DOUBLE,
    sector_pct_above_50 DOUBLE,
    sector_pct_above_200 DOUBLE,
    sector_median_roc_63 DOUBLE,
    passed_any_strategy BOOLEAN,
    strategy_pass_count INTEGER,
    passed_slots_json VARCHAR,
    fwd_return_1d DOUBLE,
    fwd_return_3d DOUBLE,
    fwd_return_5d DOUBLE,
    fwd_return_10d DOUBLE,
    fwd_return_20d DOUBLE,
    alpha_vs_spy_1d DOUBLE,
    alpha_vs_spy_3d DOUBLE,
    alpha_vs_spy_5d DOUBLE,
    alpha_vs_spy_10d DOUBLE,
    alpha_vs_spy_20d DOUBLE,
    alpha_vs_sector_1d DOUBLE,
    alpha_vs_sector_3d DOUBLE,
    alpha_vs_sector_5d DOUBLE,
    alpha_vs_sector_10d DOUBLE,
    alpha_vs_sector_20d DOUBLE,
    mfe_20d DOUBLE,
    mae_20d DOUBLE,
    details_json VARCHAR,
    PRIMARY KEY (snapshot_date, ticker)
);
CREATE INDEX IF NOT EXISTS idx_universe_daily_snapshots_date
ON universe_daily_snapshots (snapshot_date);
"""

ANALYST_SNAPSHOTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS analyst_snapshots (
    snapshot_date DATE NOT NULL,
    ticker VARCHAR NOT NULL,
    provider VARCHAR NOT NULL,
    captured_at VARCHAR,
    target_mean DOUBLE,
    target_median DOUBLE,
    target_low DOUBLE,
    target_high DOUBLE,
    analyst_count INTEGER,
    recommendation VARCHAR,
    details_json VARCHAR,
    PRIMARY KEY (snapshot_date, ticker, provider)
);
CREATE INDEX IF NOT EXISTS idx_analyst_snapshots_date
ON analyst_snapshots (snapshot_date);
"""

SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS Universe (
    ticker TEXT PRIMARY KEY,
    sector TEXT NOT NULL,
    sub_industry TEXT,
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

CREATE TABLE IF NOT EXISTS Earnings_Calendar (
    ticker TEXT NOT NULL,
    earnings_date TEXT NOT NULL,
    PRIMARY KEY (ticker, earnings_date)
);

CREATE TABLE IF NOT EXISTS Scan_Candidates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_date TEXT NOT NULL,
    ticker TEXT NOT NULL,
    strategy_slot TEXT NOT NULL,
    strategy_sector TEXT NOT NULL,
    sector TEXT,
    md_volume_30d REAL,
    adj_close REAL,
    regime_etf TEXT,
    signal_score REAL,
    setup_quality_score REAL,
    expected_alpha_score REAL,
    breadth_score REAL,
    freshness_score REAL,
    overlap_penalty REAL,
    opportunity_score REAL,
    selected INTEGER NOT NULL DEFAULT 0,
    selected_rank INTEGER,
    shares INTEGER,
    fwd_return_1d REAL,
    fwd_return_3d REAL,
    fwd_return_5d REAL,
    fwd_return_10d REAL,
    fwd_return_20d REAL,
    alpha_vs_spy_1d REAL,
    alpha_vs_spy_3d REAL,
    alpha_vs_spy_5d REAL,
    alpha_vs_spy_10d REAL,
    alpha_vs_spy_20d REAL,
    alpha_vs_sector_1d REAL,
    alpha_vs_sector_3d REAL,
    alpha_vs_sector_5d REAL,
    alpha_vs_sector_10d REAL,
    alpha_vs_sector_20d REAL,
    mfe_20d REAL,
    mae_20d REAL,
    selection_score REAL,
    selection_source TEXT,
    model_predicted_alpha REAL,
    model_rank INTEGER,
    model_generated_at TEXT,
    model_name TEXT,
    details_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS Scan_Slot_Diagnostics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_date TEXT NOT NULL,
    strategy_slot TEXT NOT NULL,
    strategy_sector TEXT NOT NULL,
    gate_counts_json TEXT NOT NULL,
    first_zero_gate TEXT NOT NULL,
    component_positive_counts_json TEXT NOT NULL,
    gated_count INTEGER NOT NULL DEFAULT 0,
    cleared_opportunity_count INTEGER NOT NULL DEFAULT 0,
    dropped_after_opportunity_count INTEGER NOT NULL DEFAULT 0,
    avg_gated_opportunity_score REAL,
    top_cleared_json TEXT NOT NULL,
    top_dropped_json TEXT NOT NULL,
    drop_examples_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS Shortlist_Model_Runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    generated_at TEXT NOT NULL,
    horizon_days INTEGER NOT NULL,
    eligible_universe_mode TEXT NOT NULL DEFAULT 'passed_only',
    model_scope TEXT NOT NULL DEFAULT 'global',
    xgboost_config TEXT NOT NULL DEFAULT 'baseline',
    top_n INTEGER NOT NULL,
    min_train_dates INTEGER NOT NULL,
    test_window_dates INTEGER NOT NULL,
    recent_dates INTEGER NOT NULL,
    champion_model TEXT NOT NULL,
    target_column TEXT NOT NULL,
    eligible_rows INTEGER NOT NULL,
    eligible_dates INTEGER NOT NULL,
    oos_dates INTEGER NOT NULL,
    live_snapshot_date TEXT,
    report_path TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS Shortlist_Model_Predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    generated_at TEXT NOT NULL,
    horizon_days INTEGER NOT NULL,
    eligible_universe_mode TEXT NOT NULL DEFAULT 'passed_only',
    model_scope TEXT NOT NULL DEFAULT 'global',
    model_name TEXT NOT NULL,
    dataset_split TEXT NOT NULL CHECK(dataset_split IN ('oos', 'live')),
    snapshot_date TEXT NOT NULL,
    ticker TEXT NOT NULL,
    sector TEXT,
    md_volume_30d REAL,
    predicted_alpha REAL,
    actual_alpha_vs_sector REAL,
    details_json TEXT
);
"""


@dataclass(frozen=True)
class UniverseRow:
    ticker: str
    sector: str
    sub_industry: str | None = None
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
            duckdb_conn.execute(UNIVERSE_SNAPSHOTS_SCHEMA)
            duckdb_conn.execute(ANALYST_SNAPSHOTS_SCHEMA)
            self._migrate_duckdb_schema(duckdb_conn)

    def sqlite_connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.paths.sqlite_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _migrate_sqlite_schema(self, connection: sqlite3.Connection) -> None:
        universe_columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(Universe)").fetchall()
        }
        if "sub_industry" not in universe_columns:
            connection.execute("ALTER TABLE Universe ADD COLUMN sub_industry TEXT")
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
        scan_candidate_columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(Scan_Candidates)").fetchall()
        }
        for name, column_type in [
            ("md_volume_30d", "REAL"),
            ("adj_close", "REAL"),
            ("regime_etf", "TEXT"),
            ("selected_rank", "INTEGER"),
            ("fwd_return_1d", "REAL"),
            ("fwd_return_3d", "REAL"),
            ("fwd_return_5d", "REAL"),
            ("fwd_return_10d", "REAL"),
            ("fwd_return_20d", "REAL"),
            ("alpha_vs_spy_1d", "REAL"),
            ("alpha_vs_spy_3d", "REAL"),
            ("alpha_vs_spy_5d", "REAL"),
            ("alpha_vs_spy_10d", "REAL"),
            ("alpha_vs_spy_20d", "REAL"),
            ("alpha_vs_sector_1d", "REAL"),
            ("alpha_vs_sector_3d", "REAL"),
            ("alpha_vs_sector_5d", "REAL"),
            ("alpha_vs_sector_10d", "REAL"),
            ("alpha_vs_sector_20d", "REAL"),
            ("mfe_20d", "REAL"),
            ("mae_20d", "REAL"),
            ("selection_score", "REAL"),
            ("selection_source", "TEXT"),
            ("model_predicted_alpha", "REAL"),
            ("model_rank", "INTEGER"),
            ("model_generated_at", "TEXT"),
            ("model_name", "TEXT"),
        ]:
            if name not in scan_candidate_columns:
                connection.execute(f"ALTER TABLE Scan_Candidates ADD COLUMN {name} {column_type}")
        shortlist_prediction_columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(Shortlist_Model_Predictions)").fetchall()
        }
        shortlist_run_columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(Shortlist_Model_Runs)").fetchall()
        }
        if "eligible_universe_mode" not in shortlist_run_columns:
            connection.execute(
                "ALTER TABLE Shortlist_Model_Runs ADD COLUMN eligible_universe_mode TEXT NOT NULL DEFAULT 'passed_only'"
            )
        if "model_scope" not in shortlist_run_columns:
            connection.execute(
                "ALTER TABLE Shortlist_Model_Runs ADD COLUMN model_scope TEXT NOT NULL DEFAULT 'global'"
            )
        if "xgboost_config" not in shortlist_run_columns:
            connection.execute(
                "ALTER TABLE Shortlist_Model_Runs ADD COLUMN xgboost_config TEXT NOT NULL DEFAULT 'baseline'"
            )
        connection.execute(
            "UPDATE Shortlist_Model_Runs SET eligible_universe_mode = 'passed_only' WHERE eligible_universe_mode IS NULL OR TRIM(eligible_universe_mode) = ''"
        )
        connection.execute(
            "UPDATE Shortlist_Model_Runs SET model_scope = 'global' WHERE model_scope IS NULL OR TRIM(model_scope) = ''"
        )
        connection.execute(
            "UPDATE Shortlist_Model_Runs SET xgboost_config = 'baseline' WHERE xgboost_config IS NULL OR TRIM(xgboost_config) = ''"
        )
        if "eligible_universe_mode" not in shortlist_prediction_columns:
            connection.execute(
                "ALTER TABLE Shortlist_Model_Predictions ADD COLUMN eligible_universe_mode TEXT NOT NULL DEFAULT 'passed_only'"
            )
            shortlist_prediction_columns.add("eligible_universe_mode")
        if "model_scope" not in shortlist_prediction_columns:
            connection.execute(
                "ALTER TABLE Shortlist_Model_Predictions ADD COLUMN model_scope TEXT NOT NULL DEFAULT 'global'"
            )
            shortlist_prediction_columns.add("model_scope")
        connection.execute(
            "UPDATE Shortlist_Model_Predictions SET eligible_universe_mode = 'passed_only' WHERE eligible_universe_mode IS NULL OR TRIM(eligible_universe_mode) = ''"
        )
        connection.execute(
            "UPDATE Shortlist_Model_Predictions SET model_scope = 'global' WHERE model_scope IS NULL OR TRIM(model_scope) = ''"
        )
        if "details_json" not in shortlist_prediction_columns:
            connection.execute("ALTER TABLE Shortlist_Model_Predictions ADD COLUMN details_json TEXT")

    def _migrate_duckdb_schema(self, connection) -> None:
        for statement in [
            "ALTER TABLE universe_daily_snapshots ADD COLUMN IF NOT EXISTS sub_industry VARCHAR",
            "ALTER TABLE universe_daily_snapshots ADD COLUMN IF NOT EXISTS subindustry_benchmark VARCHAR",
            "ALTER TABLE universe_daily_snapshots ADD COLUMN IF NOT EXISTS relative_strength_index_vs_qqq DOUBLE",
            "ALTER TABLE universe_daily_snapshots ADD COLUMN IF NOT EXISTS relative_strength_index_vs_xlk DOUBLE",
            "ALTER TABLE universe_daily_snapshots ADD COLUMN IF NOT EXISTS relative_strength_index_vs_subindustry DOUBLE",
            "ALTER TABLE universe_daily_snapshots ADD COLUMN IF NOT EXISTS roc_126 DOUBLE",
            "ALTER TABLE universe_daily_snapshots ADD COLUMN IF NOT EXISTS last_earnings_gap_pct DOUBLE",
            "ALTER TABLE universe_daily_snapshots ADD COLUMN IF NOT EXISTS last_earnings_volume_ratio_20 DOUBLE",
            "ALTER TABLE universe_daily_snapshots ADD COLUMN IF NOT EXISTS last_earnings_open_vs_20d_high DOUBLE",
            "ALTER TABLE universe_daily_snapshots ADD COLUMN IF NOT EXISTS close_vs_last_earnings_close DOUBLE",
            "CREATE TABLE IF NOT EXISTS analyst_snapshots (snapshot_date DATE NOT NULL, ticker VARCHAR NOT NULL, provider VARCHAR NOT NULL, captured_at VARCHAR, target_mean DOUBLE, target_median DOUBLE, target_low DOUBLE, target_high DOUBLE, analyst_count INTEGER, recommendation VARCHAR, details_json VARCHAR, PRIMARY KEY (snapshot_date, ticker, provider))",
            "CREATE INDEX IF NOT EXISTS idx_analyst_snapshots_date ON analyst_snapshots (snapshot_date)",
            "ALTER TABLE analyst_snapshots ADD COLUMN IF NOT EXISTS captured_at VARCHAR",
            "ALTER TABLE analyst_snapshots ADD COLUMN IF NOT EXISTS target_mean DOUBLE",
            "ALTER TABLE analyst_snapshots ADD COLUMN IF NOT EXISTS target_median DOUBLE",
            "ALTER TABLE analyst_snapshots ADD COLUMN IF NOT EXISTS target_low DOUBLE",
            "ALTER TABLE analyst_snapshots ADD COLUMN IF NOT EXISTS target_high DOUBLE",
            "ALTER TABLE analyst_snapshots ADD COLUMN IF NOT EXISTS analyst_count INTEGER",
            "ALTER TABLE analyst_snapshots ADD COLUMN IF NOT EXISTS recommendation VARCHAR",
            "ALTER TABLE analyst_snapshots ADD COLUMN IF NOT EXISTS details_json VARCHAR",
        ]:
            connection.execute(statement)

    def duckdb_connection(self):
        import duckdb

        return duckdb.connect(str(self.paths.duckdb_path))

    def universe_is_empty(self) -> bool:
        with self.sqlite_connection() as connection:
            row = connection.execute("SELECT COUNT(*) AS count FROM Universe").fetchone()
        return bool(row["count"] == 0)

    def bootstrap_universe(self, members: Iterable[UniverseRow]) -> int:
        rows = [
            (member.ticker, member.sector, member.sub_industry, int(member.is_active))
            for member in members
        ]
        with self.sqlite_connection() as connection:
            connection.executemany(
                """
                INSERT INTO Universe (ticker, sector, sub_industry, is_active)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(ticker) DO UPDATE SET
                    sector = excluded.sector,
                    sub_industry = excluded.sub_industry,
                    is_active = excluded.is_active
                """,
                rows,
            )
        return len(rows)

    def refresh_universe_metadata(self, members: Iterable[UniverseRow]) -> int:
        rows = [(member.ticker, member.sector, member.sub_industry) for member in members]
        with self.sqlite_connection() as connection:
            connection.executemany(
                """
                INSERT INTO Universe (ticker, sector, sub_industry, is_active)
                VALUES (?, ?, ?, 1)
                ON CONFLICT(ticker) DO UPDATE SET
                    sector = excluded.sector,
                    sub_industry = excluded.sub_industry
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
                SELECT ticker, sector, sub_industry, md_volume_30d
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
        query = "SELECT ticker, sector, sub_industry, is_active, md_volume_30d FROM Universe"
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

    def get_backtest_result_by_strategy_id(self, strategy_id: int) -> sqlite3.Row | None:
        with self.sqlite_connection() as connection:
            return connection.execute(
                """
                SELECT id, run_id, strategy_id, params_json, norm_score, profit_factor, expectancy, alpha_vs_spy, alpha_vs_sector, mdd, win_rate, trade_count
                FROM Backtest_Results
                WHERE strategy_id = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (strategy_id,),
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

    def list_closed_trades(self) -> list[sqlite3.Row]:
        with self.sqlite_connection() as connection:
            return list(
                connection.execute(
                    """
                    SELECT rowid, ticker, entry_date, entry_price, entry_atr, strategy_id, strategy_slot, shares, max_price_seen, status, exit_date, exit_price
                    FROM Active_Trades
                    WHERE status = 'closed'
                    ORDER BY exit_date ASC, ticker ASC, rowid ASC
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

    def replace_earnings_dates(self, ticker: str, earnings_dates: Iterable[date]) -> int:
        normalized_dates = sorted({str(earnings_date) for earnings_date in earnings_dates})
        with self.sqlite_connection() as connection:
            connection.execute("DELETE FROM Earnings_Calendar WHERE ticker = ?", (ticker,))
            if normalized_dates:
                connection.executemany(
                    """
                    INSERT INTO Earnings_Calendar (ticker, earnings_date)
                    VALUES (?, ?)
                    """,
                    [(ticker, earnings_date) for earnings_date in normalized_dates],
                )
        return len(normalized_dates)

    def load_earnings_calendar(self, tickers: Iterable[str] | None = None):
        import pandas as pd

        query = "SELECT ticker, earnings_date FROM Earnings_Calendar"
        params: list[str] = []
        if tickers is not None:
            ticker_list = list(tickers)
            if not ticker_list:
                return pd.DataFrame(columns=["ticker", "earnings_date"])
            placeholders = ", ".join(["?"] * len(ticker_list))
            query += f" WHERE ticker IN ({placeholders})"
            params.extend(ticker_list)
        query += " ORDER BY ticker ASC, earnings_date ASC"
        with self.sqlite_connection() as connection:
            rows = connection.execute(query, params).fetchall()
        if not rows:
            return pd.DataFrame(columns=["ticker", "earnings_date"])
        frame = pd.DataFrame(rows, columns=["ticker", "earnings_date"])
        frame["earnings_date"] = pd.to_datetime(frame["earnings_date"]).dt.normalize()
        return frame

    def replace_scan_candidates(self, *, scan_date: str, rows: Iterable[dict]) -> int:
        payload = list(rows)
        with self.sqlite_connection() as connection:
            connection.execute("DELETE FROM Scan_Candidates WHERE scan_date = ?", (scan_date,))
            if not payload:
                return 0
            connection.executemany(
                """
                INSERT INTO Scan_Candidates (
                    scan_date,
                    ticker,
                    strategy_slot,
                    strategy_sector,
                    sector,
                    md_volume_30d,
                    adj_close,
                    regime_etf,
                    signal_score,
                    setup_quality_score,
                    expected_alpha_score,
                    breadth_score,
                    freshness_score,
                    overlap_penalty,
                    opportunity_score,
                    selected,
                    selected_rank,
                    shares,
                    fwd_return_1d,
                    fwd_return_3d,
                    fwd_return_5d,
                    fwd_return_10d,
                    fwd_return_20d,
                    alpha_vs_spy_1d,
                    alpha_vs_spy_3d,
                    alpha_vs_spy_5d,
                    alpha_vs_spy_10d,
                    alpha_vs_spy_20d,
                    alpha_vs_sector_1d,
                    alpha_vs_sector_3d,
                    alpha_vs_sector_5d,
                    alpha_vs_sector_10d,
                    alpha_vs_sector_20d,
                    mfe_20d,
                    mae_20d,
                    selection_score,
                    selection_source,
                    model_predicted_alpha,
                    model_rank,
                    model_generated_at,
                    model_name,
                    details_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        scan_date,
                        str(row["ticker"]),
                        str(row["strategy_slot"]),
                        str(row["strategy_sector"]),
                        row.get("sector"),
                        row.get("md_volume_30d"),
                        row.get("adj_close"),
                        row.get("regime_etf"),
                        row.get("signal_score"),
                        row.get("setup_quality_score"),
                        row.get("expected_alpha_score"),
                        row.get("breadth_score"),
                        row.get("freshness_score"),
                        row.get("overlap_penalty"),
                        row.get("opportunity_score"),
                        int(bool(row.get("selected", False))),
                        int(row["selected_rank"]) if row.get("selected_rank") is not None else None,
                        int(row["shares"]) if row.get("shares") is not None else None,
                        row.get("fwd_return_1d"),
                        row.get("fwd_return_3d"),
                        row.get("fwd_return_5d"),
                        row.get("fwd_return_10d"),
                        row.get("fwd_return_20d"),
                        row.get("alpha_vs_spy_1d"),
                        row.get("alpha_vs_spy_3d"),
                        row.get("alpha_vs_spy_5d"),
                        row.get("alpha_vs_spy_10d"),
                        row.get("alpha_vs_spy_20d"),
                        row.get("alpha_vs_sector_1d"),
                        row.get("alpha_vs_sector_3d"),
                        row.get("alpha_vs_sector_5d"),
                        row.get("alpha_vs_sector_10d"),
                        row.get("alpha_vs_sector_20d"),
                        row.get("mfe_20d"),
                        row.get("mae_20d"),
                        row.get("selection_score"),
                        row.get("selection_source"),
                        row.get("model_predicted_alpha"),
                        int(row["model_rank"]) if row.get("model_rank") is not None else None,
                        row.get("model_generated_at"),
                        row.get("model_name"),
                        json.dumps(row.get("details", {}), sort_keys=True),
                    )
                    for row in payload
                ],
            )
        return len(payload)

    def replace_scan_slot_diagnostics(self, *, scan_date: str, rows: Iterable[dict]) -> int:
        payload = list(rows)
        with self.sqlite_connection() as connection:
            connection.execute("DELETE FROM Scan_Slot_Diagnostics WHERE scan_date = ?", (scan_date,))
            if not payload:
                return 0
            connection.executemany(
                """
                INSERT INTO Scan_Slot_Diagnostics (
                    scan_date,
                    strategy_slot,
                    strategy_sector,
                    gate_counts_json,
                    first_zero_gate,
                    component_positive_counts_json,
                    gated_count,
                    cleared_opportunity_count,
                    dropped_after_opportunity_count,
                    avg_gated_opportunity_score,
                    top_cleared_json,
                    top_dropped_json,
                    drop_examples_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        scan_date,
                        str(row["strategy_slot"]),
                        str(row["strategy_sector"]),
                        json.dumps(row.get("gate_counts", []), sort_keys=False),
                        str(row.get("first_zero_gate", "unavailable")),
                        json.dumps(row.get("component_positive_counts", []), sort_keys=False),
                        int(row.get("gated_count", 0) or 0),
                        int(row.get("cleared_opportunity_count", 0) or 0),
                        int(row.get("dropped_after_opportunity_count", 0) or 0),
                        row.get("avg_gated_opportunity_score"),
                        json.dumps(row.get("top_cleared", []), sort_keys=False),
                        json.dumps(row.get("top_dropped", []), sort_keys=False),
                        json.dumps(row.get("drop_examples", []), sort_keys=False),
                    )
                    for row in payload
                ],
            )
        return len(payload)

    def update_scan_candidate_outcomes(self, *, scan_date: str, rows: Iterable[dict]) -> int:
        payload = list(rows)
        if not payload:
            return 0
        with self.sqlite_connection() as connection:
            connection.executemany(
                """
                UPDATE Scan_Candidates
                SET
                    fwd_return_1d = ?,
                    fwd_return_3d = ?,
                    fwd_return_5d = ?,
                    fwd_return_10d = ?,
                    fwd_return_20d = ?,
                    alpha_vs_spy_1d = ?,
                    alpha_vs_spy_3d = ?,
                    alpha_vs_spy_5d = ?,
                    alpha_vs_spy_10d = ?,
                    alpha_vs_spy_20d = ?,
                    alpha_vs_sector_1d = ?,
                    alpha_vs_sector_3d = ?,
                    alpha_vs_sector_5d = ?,
                    alpha_vs_sector_10d = ?,
                    alpha_vs_sector_20d = ?,
                    mfe_20d = ?,
                    mae_20d = ?
                WHERE scan_date = ? AND ticker = ? AND strategy_slot = ?
                """,
                [
                    (
                        row.get("fwd_return_1d"),
                        row.get("fwd_return_3d"),
                        row.get("fwd_return_5d"),
                        row.get("fwd_return_10d"),
                        row.get("fwd_return_20d"),
                        row.get("alpha_vs_spy_1d"),
                        row.get("alpha_vs_spy_3d"),
                        row.get("alpha_vs_spy_5d"),
                        row.get("alpha_vs_spy_10d"),
                        row.get("alpha_vs_spy_20d"),
                        row.get("alpha_vs_sector_1d"),
                        row.get("alpha_vs_sector_3d"),
                        row.get("alpha_vs_sector_5d"),
                        row.get("alpha_vs_sector_10d"),
                        row.get("alpha_vs_sector_20d"),
                        row.get("mfe_20d"),
                        row.get("mae_20d"),
                        scan_date,
                        str(row["ticker"]),
                        str(row["strategy_slot"]),
                    )
                    for row in payload
                ],
            )
        return len(payload)

    def load_scan_candidates(self, scan_date: str | None = None):
        import pandas as pd

        query = """
            SELECT
                scan_date,
                ticker,
                strategy_slot,
                strategy_sector,
                sector,
                md_volume_30d,
                adj_close,
                regime_etf,
                signal_score,
                setup_quality_score,
                expected_alpha_score,
                breadth_score,
                freshness_score,
                overlap_penalty,
                opportunity_score,
                selected,
                selected_rank,
                shares,
                fwd_return_1d,
                fwd_return_3d,
                fwd_return_5d,
                fwd_return_10d,
                fwd_return_20d,
                alpha_vs_spy_1d,
                alpha_vs_spy_3d,
                alpha_vs_spy_5d,
                alpha_vs_spy_10d,
                alpha_vs_spy_20d,
                alpha_vs_sector_1d,
                alpha_vs_sector_3d,
                alpha_vs_sector_5d,
                alpha_vs_sector_10d,
                alpha_vs_sector_20d,
                mfe_20d,
                mae_20d,
                selection_score,
                selection_source,
                model_predicted_alpha,
                model_rank,
                model_generated_at,
                model_name,
                details_json
            FROM Scan_Candidates
        """
        params: tuple = ()
        if scan_date is not None:
            query += " WHERE scan_date = ?"
            params = (scan_date,)
        query += " ORDER BY scan_date ASC, selected DESC, opportunity_score DESC, ticker ASC"
        with self.sqlite_connection() as connection:
            rows = connection.execute(query, params).fetchall()
        if not rows:
            return pd.DataFrame(
                columns=[
                    "scan_date",
                    "ticker",
                    "strategy_slot",
                    "strategy_sector",
                    "sector",
                    "md_volume_30d",
                    "adj_close",
                    "regime_etf",
                    "signal_score",
                    "setup_quality_score",
                    "expected_alpha_score",
                    "breadth_score",
                    "freshness_score",
                    "overlap_penalty",
                    "opportunity_score",
                    "selected",
                    "selected_rank",
                    "shares",
                    "fwd_return_1d",
                    "fwd_return_3d",
                    "fwd_return_5d",
                    "fwd_return_10d",
                    "fwd_return_20d",
                    "alpha_vs_spy_1d",
                    "alpha_vs_spy_3d",
                    "alpha_vs_spy_5d",
                    "alpha_vs_spy_10d",
                    "alpha_vs_spy_20d",
                    "alpha_vs_sector_1d",
                    "alpha_vs_sector_3d",
                    "alpha_vs_sector_5d",
                    "alpha_vs_sector_10d",
                    "alpha_vs_sector_20d",
                    "mfe_20d",
                    "mae_20d",
                    "selection_score",
                    "selection_source",
                    "model_predicted_alpha",
                    "model_rank",
                    "model_generated_at",
                    "model_name",
                    "details_json",
                ]
            )
        frame = pd.DataFrame(rows, columns=rows[0].keys())
        frame["selected"] = frame["selected"].astype(int)
        return frame

    def load_scan_slot_diagnostics(self, scan_date: str | None = None):
        import pandas as pd

        query = """
            SELECT
                scan_date,
                strategy_slot,
                strategy_sector,
                gate_counts_json,
                first_zero_gate,
                component_positive_counts_json,
                gated_count,
                cleared_opportunity_count,
                dropped_after_opportunity_count,
                avg_gated_opportunity_score,
                top_cleared_json,
                top_dropped_json,
                drop_examples_json
            FROM Scan_Slot_Diagnostics
        """
        params: tuple = ()
        if scan_date is not None:
            query += " WHERE scan_date = ?"
            params = (scan_date,)
        query += " ORDER BY scan_date ASC, strategy_slot ASC"
        with self.sqlite_connection() as connection:
            rows = connection.execute(query, params).fetchall()
        if not rows:
            return pd.DataFrame(
                columns=[
                    "scan_date",
                    "strategy_slot",
                    "strategy_sector",
                    "gate_counts_json",
                    "first_zero_gate",
                    "component_positive_counts_json",
                    "gated_count",
                    "cleared_opportunity_count",
                    "dropped_after_opportunity_count",
                    "avg_gated_opportunity_score",
                    "top_cleared_json",
                    "top_dropped_json",
                    "drop_examples_json",
                ]
            )
        return pd.DataFrame(rows, columns=rows[0].keys())

    def insert_shortlist_model_run(self, *, row: dict) -> int:
        with self.sqlite_connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO Shortlist_Model_Runs (
                    generated_at,
                    horizon_days,
                    eligible_universe_mode,
                    model_scope,
                    xgboost_config,
                    top_n,
                    min_train_dates,
                    test_window_dates,
                    recent_dates,
                    champion_model,
                    target_column,
                    eligible_rows,
                    eligible_dates,
                    oos_dates,
                    live_snapshot_date,
                    report_path
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(row["generated_at"]),
                    int(row["horizon_days"]),
                    str(row.get("eligible_universe_mode") or "passed_only"),
                    str(row.get("model_scope") or "global"),
                    str(row.get("xgboost_config") or "baseline"),
                    int(row["top_n"]),
                    int(row["min_train_dates"]),
                    int(row["test_window_dates"]),
                    int(row["recent_dates"]),
                    str(row["champion_model"]),
                    str(row["target_column"]),
                    int(row["eligible_rows"]),
                    int(row["eligible_dates"]),
                    int(row["oos_dates"]),
                    row.get("live_snapshot_date"),
                    str(row["report_path"]),
                ),
            )
            return int(cursor.lastrowid)

    def replace_shortlist_model_predictions(
        self,
        *,
        generated_at: str,
        horizon_days: int,
        eligible_universe_mode: str = "passed_only",
        model_scope: str = "global",
        rows: Iterable[dict],
    ) -> int:
        payload = list(rows)
        with self.sqlite_connection() as connection:
            connection.execute(
                """
                DELETE FROM Shortlist_Model_Predictions
                WHERE generated_at = ? AND horizon_days = ? AND eligible_universe_mode = ? AND model_scope = ?
                """,
                (
                    generated_at,
                    int(horizon_days),
                    str(eligible_universe_mode or "passed_only"),
                    str(model_scope or "global"),
                ),
            )
            if not payload:
                return 0
            connection.executemany(
                """
                INSERT INTO Shortlist_Model_Predictions (
                    generated_at,
                    horizon_days,
                    eligible_universe_mode,
                    model_scope,
                    model_name,
                    dataset_split,
                    snapshot_date,
                    ticker,
                    sector,
                    md_volume_30d,
                    predicted_alpha,
                    actual_alpha_vs_sector,
                    details_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        str(generated_at),
                        int(horizon_days),
                        str(row.get("eligible_universe_mode") or eligible_universe_mode or "passed_only"),
                        str(row.get("model_scope") or model_scope or "global"),
                        str(row["model_name"]),
                        str(row["dataset_split"]),
                        str(row["snapshot_date"]),
                        str(row["ticker"]),
                        row.get("sector"),
                        row.get("md_volume_30d"),
                        row.get("predicted_alpha"),
                        row.get("actual_alpha_vs_sector"),
                        json.dumps(row.get("details", {}), sort_keys=True),
                    )
                    for row in payload
                ],
            )
        return len(payload)

    def load_shortlist_model_runs(
        self,
        *,
        horizon_days: int | None = None,
        eligible_universe_mode: str | None = None,
        model_scope: str | None = None,
        xgboost_config: str | None = None,
        limit: int | None = None,
    ):
        import pandas as pd

        query = """
            SELECT
                id,
                generated_at,
                horizon_days,
                eligible_universe_mode,
                model_scope,
                xgboost_config,
                top_n,
                min_train_dates,
                test_window_dates,
                recent_dates,
                champion_model,
                target_column,
                eligible_rows,
                eligible_dates,
                oos_dates,
                live_snapshot_date,
                report_path
            FROM Shortlist_Model_Runs
        """
        params: list[object] = []
        filters: list[str] = []
        if horizon_days is not None:
            filters.append("horizon_days = ?")
            params.append(int(horizon_days))
        if eligible_universe_mode is not None:
            filters.append("eligible_universe_mode = ?")
            params.append(str(eligible_universe_mode))
        if model_scope is not None:
            filters.append("model_scope = ?")
            params.append(str(model_scope))
        if xgboost_config is not None:
            filters.append("xgboost_config = ?")
            params.append(str(xgboost_config))
        if filters:
            query += " WHERE " + " AND ".join(filters)
        query += " ORDER BY generated_at DESC, id DESC"
        if limit is not None:
            query += f" LIMIT {int(limit)}"
        with self.sqlite_connection() as connection:
            rows = connection.execute(query, tuple(params)).fetchall()
        if not rows:
            return pd.DataFrame(
                columns=[
                    "id",
                    "generated_at",
                    "horizon_days",
                    "eligible_universe_mode",
                    "model_scope",
                    "xgboost_config",
                    "top_n",
                    "min_train_dates",
                    "test_window_dates",
                    "recent_dates",
                    "champion_model",
                    "target_column",
                    "eligible_rows",
                    "eligible_dates",
                    "oos_dates",
                    "live_snapshot_date",
                    "report_path",
                ]
            )
        return pd.DataFrame(rows, columns=rows[0].keys())

    def load_shortlist_model_predictions(
        self,
        *,
        generated_at: str | None = None,
        horizon_days: int | None = None,
        eligible_universe_mode: str | None = None,
        model_scope: str | None = None,
        dataset_split: str | None = None,
        model_name: str | None = None,
    ):
        import pandas as pd

        query = """
            SELECT
                generated_at,
                horizon_days,
                eligible_universe_mode,
                model_scope,
                model_name,
                dataset_split,
                snapshot_date,
                ticker,
                sector,
                md_volume_30d,
                predicted_alpha,
                actual_alpha_vs_sector,
                details_json
            FROM Shortlist_Model_Predictions
            WHERE 1 = 1
        """
        params: list[object] = []
        if generated_at is not None:
            query += " AND generated_at = ?"
            params.append(str(generated_at))
        if horizon_days is not None:
            query += " AND horizon_days = ?"
            params.append(int(horizon_days))
        if eligible_universe_mode is not None:
            query += " AND eligible_universe_mode = ?"
            params.append(str(eligible_universe_mode))
        if model_scope is not None:
            query += " AND model_scope = ?"
            params.append(str(model_scope))
        if dataset_split is not None:
            query += " AND dataset_split = ?"
            params.append(str(dataset_split))
        if model_name is not None:
            query += " AND model_name = ?"
            params.append(str(model_name))
        query += " ORDER BY snapshot_date ASC, predicted_alpha DESC, ticker ASC"
        with self.sqlite_connection() as connection:
            rows = connection.execute(query, tuple(params)).fetchall()
        if not rows:
            return pd.DataFrame(
                columns=[
                    "generated_at",
                    "horizon_days",
                    "eligible_universe_mode",
                    "model_scope",
                    "model_name",
                    "dataset_split",
                    "snapshot_date",
                    "ticker",
                    "sector",
                    "md_volume_30d",
                    "predicted_alpha",
                    "actual_alpha_vs_sector",
                    "details_json",
                ]
            )
        return pd.DataFrame(rows, columns=rows[0].keys())

    def replace_universe_daily_snapshots(self, *, snapshot_date: str, rows: Iterable[dict]) -> int:
        payload = list(rows)
        columns = [
            "snapshot_date",
            "ticker",
            "sector",
            "sub_industry",
            "subindustry_benchmark",
            "regime_etf",
            "regime_green",
            "md_volume_30d",
            "adj_close",
            "atr_14",
            "relative_strength_index_vs_spy",
            "relative_strength_index_vs_qqq",
            "relative_strength_index_vs_xlk",
            "relative_strength_index_vs_subindustry",
            "roc_63",
            "roc_126",
            "vol_alpha",
            "sma_200_dist",
            "sma_50_dist",
            "rsi_14",
            "days_to_next_earnings",
            "days_since_last_earnings",
            "last_earnings_gap_pct",
            "last_earnings_volume_ratio_20",
            "last_earnings_open_vs_20d_high",
            "close_vs_last_earnings_close",
            "avg_abs_gap_pct_20",
            "max_gap_down_pct_60",
            "distance_above_20d_high",
            "base_range_pct_20",
            "base_atr_contraction_20",
            "base_volume_dryup_ratio_20",
            "breakout_volume_ratio_50",
            "sector_pct_above_50",
            "sector_pct_above_200",
            "sector_median_roc_63",
            "passed_any_strategy",
            "strategy_pass_count",
            "passed_slots_json",
            "fwd_return_1d",
            "fwd_return_3d",
            "fwd_return_5d",
            "fwd_return_10d",
            "fwd_return_20d",
            "alpha_vs_spy_1d",
            "alpha_vs_spy_3d",
            "alpha_vs_spy_5d",
            "alpha_vs_spy_10d",
            "alpha_vs_spy_20d",
            "alpha_vs_sector_1d",
            "alpha_vs_sector_3d",
            "alpha_vs_sector_5d",
            "alpha_vs_sector_10d",
            "alpha_vs_sector_20d",
            "mfe_20d",
            "mae_20d",
            "details_json",
        ]
        placeholders = ", ".join(["?"] * len(columns))
        with self.duckdb_connection() as connection:
            connection.execute("DELETE FROM universe_daily_snapshots WHERE snapshot_date = ?", (snapshot_date,))
            if not payload:
                return 0
            connection.executemany(
                f"""
                INSERT INTO universe_daily_snapshots ({", ".join(columns)})
                VALUES ({placeholders})
                """,
                [
                    (
                        snapshot_date,
                        str(row["ticker"]),
                        row.get("sector"),
                        row.get("sub_industry"),
                        row.get("subindustry_benchmark"),
                        row.get("regime_etf"),
                        row.get("regime_green"),
                        row.get("md_volume_30d"),
                        row.get("adj_close"),
                        row.get("atr_14"),
                        row.get("relative_strength_index_vs_spy"),
                        row.get("relative_strength_index_vs_qqq"),
                        row.get("relative_strength_index_vs_xlk"),
                        row.get("relative_strength_index_vs_subindustry"),
                        row.get("roc_63"),
                        row.get("roc_126"),
                        row.get("vol_alpha"),
                        row.get("sma_200_dist"),
                        row.get("sma_50_dist"),
                        row.get("rsi_14"),
                        row.get("days_to_next_earnings"),
                        row.get("days_since_last_earnings"),
                        row.get("last_earnings_gap_pct"),
                        row.get("last_earnings_volume_ratio_20"),
                        row.get("last_earnings_open_vs_20d_high"),
                        row.get("close_vs_last_earnings_close"),
                        row.get("avg_abs_gap_pct_20"),
                        row.get("max_gap_down_pct_60"),
                        row.get("distance_above_20d_high"),
                        row.get("base_range_pct_20"),
                        row.get("base_atr_contraction_20"),
                        row.get("base_volume_dryup_ratio_20"),
                        row.get("breakout_volume_ratio_50"),
                        row.get("sector_pct_above_50"),
                        row.get("sector_pct_above_200"),
                        row.get("sector_median_roc_63"),
                        bool(row.get("passed_any_strategy", False)),
                        int(row.get("strategy_pass_count", 0)),
                        json.dumps(row.get("passed_slots", [])),
                        row.get("fwd_return_1d"),
                        row.get("fwd_return_3d"),
                        row.get("fwd_return_5d"),
                        row.get("fwd_return_10d"),
                        row.get("fwd_return_20d"),
                        row.get("alpha_vs_spy_1d"),
                        row.get("alpha_vs_spy_3d"),
                        row.get("alpha_vs_spy_5d"),
                        row.get("alpha_vs_spy_10d"),
                        row.get("alpha_vs_spy_20d"),
                        row.get("alpha_vs_sector_1d"),
                        row.get("alpha_vs_sector_3d"),
                        row.get("alpha_vs_sector_5d"),
                        row.get("alpha_vs_sector_10d"),
                        row.get("alpha_vs_sector_20d"),
                        row.get("mfe_20d"),
                        row.get("mae_20d"),
                        json.dumps(row.get("details", {}), sort_keys=True),
                    )
                    for row in payload
                ],
            )
        return len(payload)

    def list_universe_daily_snapshot_dates(self) -> list[str]:
        with self.duckdb_connection() as connection:
            rows = connection.execute(
                "SELECT DISTINCT snapshot_date FROM universe_daily_snapshots ORDER BY snapshot_date ASC"
            ).fetchall()
        return [str(row[0]) for row in rows]

    def universe_daily_snapshot_date_needs_refresh(
        self,
        *,
        snapshot_date: str,
        required_non_null_columns: Iterable[str],
    ) -> bool:
        columns = [str(column).strip() for column in required_non_null_columns if str(column).strip()]
        if not columns:
            return False
        allowed_columns = {
            "sector",
            "sub_industry",
            "subindustry_benchmark",
            "relative_strength_index_vs_subindustry",
        }
        invalid = [column for column in columns if column not in allowed_columns]
        if invalid:
            raise ValueError(f"Unsupported universe snapshot refresh columns: {invalid}")
        select_clauses = ", ".join(
            f"SUM(CASE WHEN {column} IS NOT NULL THEN 1 ELSE 0 END) AS {column}_non_null"
            for column in columns
        )
        query = (
            f"SELECT {select_clauses} "
            "FROM universe_daily_snapshots "
            "WHERE snapshot_date = ?"
        )
        with self.duckdb_connection() as connection:
            cursor = connection.execute(query, (snapshot_date,))
            row = cursor.fetchone()
            column_names = [description[0] for description in cursor.description] if row is not None else []
        if row is None:
            return True
        row_map = {column_names[index]: row[index] for index in range(len(column_names))}
        for column in columns:
            value = row_map.get(f"{column}_non_null")
            if value is None or int(value) == 0:
                return True
        return False

    def load_universe_daily_snapshots(self, snapshot_date: str | None = None):
        import pandas as pd

        query = """
            SELECT
                snapshot_date,
                ticker,
                sector,
                sub_industry,
                subindustry_benchmark,
                regime_etf,
                regime_green,
                md_volume_30d,
                adj_close,
                atr_14,
                relative_strength_index_vs_spy,
                relative_strength_index_vs_qqq,
                relative_strength_index_vs_xlk,
                relative_strength_index_vs_subindustry,
                roc_63,
                roc_126,
                vol_alpha,
                sma_200_dist,
                sma_50_dist,
                rsi_14,
                days_to_next_earnings,
                days_since_last_earnings,
                last_earnings_gap_pct,
                last_earnings_volume_ratio_20,
                last_earnings_open_vs_20d_high,
                close_vs_last_earnings_close,
                avg_abs_gap_pct_20,
                max_gap_down_pct_60,
                distance_above_20d_high,
                base_range_pct_20,
                base_atr_contraction_20,
                base_volume_dryup_ratio_20,
                breakout_volume_ratio_50,
                sector_pct_above_50,
                sector_pct_above_200,
                sector_median_roc_63,
                passed_any_strategy,
                strategy_pass_count,
                passed_slots_json,
                fwd_return_1d,
                fwd_return_3d,
                fwd_return_5d,
                fwd_return_10d,
                fwd_return_20d,
                alpha_vs_spy_1d,
                alpha_vs_spy_3d,
                alpha_vs_spy_5d,
                alpha_vs_spy_10d,
                alpha_vs_spy_20d,
                alpha_vs_sector_1d,
                alpha_vs_sector_3d,
                alpha_vs_sector_5d,
                alpha_vs_sector_10d,
                alpha_vs_sector_20d,
                mfe_20d,
                mae_20d,
                details_json
            FROM universe_daily_snapshots
        """
        params: tuple[object, ...] = ()
        if snapshot_date is not None:
            query += " WHERE snapshot_date = ?"
            params = (snapshot_date,)
        query += " ORDER BY snapshot_date ASC, ticker ASC"
        with self.duckdb_connection() as connection:
            cursor = connection.execute(query, params)
            rows = cursor.fetchall()
            columns = [description[0] for description in cursor.description] if rows else []
        if not rows:
            return pd.DataFrame(
                columns=[
                    "snapshot_date",
                    "ticker",
                    "sector",
                    "sub_industry",
                    "subindustry_benchmark",
                    "regime_etf",
                    "regime_green",
                    "md_volume_30d",
                    "adj_close",
                    "atr_14",
                    "relative_strength_index_vs_spy",
                    "relative_strength_index_vs_qqq",
                    "relative_strength_index_vs_xlk",
                    "relative_strength_index_vs_subindustry",
                    "roc_63",
                    "roc_126",
                    "vol_alpha",
                    "sma_200_dist",
                    "sma_50_dist",
                    "rsi_14",
                    "days_to_next_earnings",
                    "days_since_last_earnings",
                    "last_earnings_gap_pct",
                    "last_earnings_volume_ratio_20",
                    "last_earnings_open_vs_20d_high",
                    "close_vs_last_earnings_close",
                    "avg_abs_gap_pct_20",
                    "max_gap_down_pct_60",
                    "distance_above_20d_high",
                    "base_range_pct_20",
                    "base_atr_contraction_20",
                    "base_volume_dryup_ratio_20",
                    "breakout_volume_ratio_50",
                    "sector_pct_above_50",
                    "sector_pct_above_200",
                    "sector_median_roc_63",
                    "passed_any_strategy",
                    "strategy_pass_count",
                    "passed_slots_json",
                    "fwd_return_1d",
                    "fwd_return_3d",
                    "fwd_return_5d",
                    "fwd_return_10d",
                    "fwd_return_20d",
                    "alpha_vs_spy_1d",
                    "alpha_vs_spy_3d",
                    "alpha_vs_spy_5d",
                    "alpha_vs_spy_10d",
                    "alpha_vs_spy_20d",
                    "alpha_vs_sector_1d",
                    "alpha_vs_sector_3d",
                    "alpha_vs_sector_5d",
                    "alpha_vs_sector_10d",
                    "alpha_vs_sector_20d",
                    "mfe_20d",
                    "mae_20d",
                    "details_json",
                ]
            )
        return pd.DataFrame(rows, columns=columns)

    def replace_analyst_snapshots(
        self,
        *,
        snapshot_date: str,
        provider: str,
        rows: Iterable[dict],
    ) -> int:
        payload = list(rows)
        columns = [
            "snapshot_date",
            "ticker",
            "provider",
            "captured_at",
            "target_mean",
            "target_median",
            "target_low",
            "target_high",
            "analyst_count",
            "recommendation",
            "details_json",
        ]
        placeholders = ", ".join(["?"] * len(columns))
        with self.duckdb_connection() as connection:
            connection.execute(
                "DELETE FROM analyst_snapshots WHERE snapshot_date = ? AND provider = ?",
                (snapshot_date, provider),
            )
            if not payload:
                return 0
            connection.executemany(
                f"""
                INSERT INTO analyst_snapshots ({", ".join(columns)})
                VALUES ({placeholders})
                """,
                [
                    (
                        snapshot_date,
                        str(row["ticker"]).strip().upper(),
                        provider,
                        row.get("captured_at"),
                        row.get("target_mean"),
                        row.get("target_median"),
                        row.get("target_low"),
                        row.get("target_high"),
                        row.get("analyst_count"),
                        row.get("recommendation"),
                        json.dumps(row.get("details", {}), sort_keys=True),
                    )
                    for row in payload
                ],
            )
        return len(payload)

    def load_analyst_snapshots(
        self,
        *,
        snapshot_date: str | None = None,
        provider: str | None = None,
    ):
        import pandas as pd

        columns = [
            "snapshot_date",
            "ticker",
            "provider",
            "captured_at",
            "target_mean",
            "target_median",
            "target_low",
            "target_high",
            "analyst_count",
            "recommendation",
            "details_json",
        ]
        query = f"SELECT {', '.join(columns)} FROM analyst_snapshots"
        clauses: list[str] = []
        params: list[object] = []
        if snapshot_date is not None:
            clauses.append("snapshot_date = ?")
            params.append(snapshot_date)
        if provider is not None:
            clauses.append("provider = ?")
            params.append(provider)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY snapshot_date ASC, ticker ASC, provider ASC"
        with self.duckdb_connection() as connection:
            cursor = connection.execute(query, tuple(params))
            rows = cursor.fetchall()
            result_columns = [description[0] for description in cursor.description] if rows else columns
        if not rows:
            return pd.DataFrame(columns=columns)
        return pd.DataFrame(rows, columns=result_columns)


def create_default_manager() -> DatabaseManager:
    return DatabaseManager()
