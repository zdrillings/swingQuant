from __future__ import annotations

from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from src.settings import AppPaths
from src.utils.db_manager import DatabaseManager


class FakeDuckDBConnection:
    def __init__(self) -> None:
        self.statements: list[str] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def execute(self, statement: str):
        self.statements.append(statement)
        return self


class DatabaseManagerInitializationTests(unittest.TestCase):
    def test_initialize_creates_sqlite_schema_and_executes_duckdb_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = AppPaths(
                root_dir=root,
                data_dir=root / "data",
                duckdb_path=root / "data" / "market_data.duckdb",
                sqlite_path=root / "data" / "ledger.sqlite",
                reports_dir=root / "reports",
                logs_dir=root / "logs",
                config_path=root / "config.yaml",
                env_path=root / ".env",
                production_strategy_path=root / "production_strategy.json",
            )
            manager = DatabaseManager(paths)
            fake_duckdb = FakeDuckDBConnection()

            with patch.object(manager, "duckdb_connection", return_value=fake_duckdb):
                manager.initialize()

            connection = sqlite3.connect(paths.sqlite_path)
            try:
                tables = {
                    row[0]
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    ).fetchall()
                }
                active_trade_columns = {
                    row[1]
                    for row in connection.execute("PRAGMA table_info(Active_Trades)").fetchall()
                }
            finally:
                connection.close()

            self.assertTrue({"Universe", "Backtest_Results", "Active_Trades", "Scan_Candidates"}.issubset(tables))
            self.assertIn("entry_atr", active_trade_columns)
            self.assertTrue(any("historical_ohlcv" in statement for statement in fake_duckdb.statements))
