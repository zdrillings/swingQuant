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
        self.executemany_calls: list[tuple[str, list[tuple]]] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def execute(self, statement: str, params=None):
        self.statements.append(statement)
        return self

    def executemany(self, statement: str, rows: list[tuple]):
        self.executemany_calls.append((statement, rows))
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
                scan_candidate_columns = {
                    row[1]
                    for row in connection.execute("PRAGMA table_info(Scan_Candidates)").fetchall()
                }
            finally:
                connection.close()

            self.assertTrue({"Universe", "Backtest_Results", "Active_Trades", "Scan_Candidates"}.issubset(tables))
            self.assertIn("entry_atr", active_trade_columns)
            self.assertTrue(
                {
                    "md_volume_30d",
                    "adj_close",
                    "regime_etf",
                    "selected_rank",
                    "fwd_return_10d",
                    "alpha_vs_sector_10d",
                    "mfe_20d",
                    "mae_20d",
                    "selection_score",
                    "selection_source",
                    "model_predicted_alpha",
                    "model_rank",
                    "model_generated_at",
                    "model_name",
                }.issubset(scan_candidate_columns)
            )
            self.assertTrue(any("historical_ohlcv" in statement for statement in fake_duckdb.statements))
            self.assertTrue(any("analyst_snapshots" in statement for statement in fake_duckdb.statements))

    def test_scan_candidates_persist_model_attribution(self) -> None:
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

            manager.replace_scan_candidates(
                scan_date="2026-05-01",
                rows=[
                    {
                        "ticker": "AAA",
                        "strategy_slot": "technology",
                        "strategy_sector": "Information Technology",
                        "sector": "Information Technology",
                        "signal_score": 0.0,
                        "setup_quality_score": 0.0,
                        "expected_alpha_score": 0.0,
                        "breadth_score": 0.0,
                        "freshness_score": 0.0,
                        "overlap_penalty": 0.0,
                        "opportunity_score": 0.0,
                        "selected": True,
                        "selected_rank": 1,
                        "shares": 10,
                        "selection_score": 0.123,
                        "selection_source": "shortlist_model",
                        "model_predicted_alpha": 0.123,
                        "model_rank": 4,
                        "model_generated_at": "2026-06-02T20:08:10+00:00",
                        "model_name": "xgboost_model",
                        "details": {},
                    }
                ],
            )

            frame = manager.load_scan_candidates()
            self.assertEqual(frame.loc[0, "selection_source"], "shortlist_model")
            self.assertEqual(frame.loc[0, "model_name"], "xgboost_model")
            self.assertEqual(frame.loc[0, "model_generated_at"], "2026-06-02T20:08:10+00:00")
            self.assertEqual(int(frame.loc[0, "model_rank"]), 4)
            self.assertAlmostEqual(float(frame.loc[0, "model_predicted_alpha"]), 0.123, places=6)
            self.assertAlmostEqual(float(frame.loc[0, "selection_score"]), 0.123, places=6)

    def test_replace_universe_daily_snapshots_builds_matching_insert_placeholders(self) -> None:
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

            row = {
                "ticker": "AAA",
                "sector": "Information Technology",
                "sub_industry": "Semiconductors",
                "subindustry_benchmark": "SMH",
                "regime_etf": "QQQ",
                "regime_green": True,
                "md_volume_30d": 30_000_000.0,
                "adj_close": 100.0,
                "atr_14": 4.0,
                "relative_strength_index_vs_spy": 80.0,
                "relative_strength_index_vs_qqq": 85.0,
                "relative_strength_index_vs_xlk": 84.0,
                "relative_strength_index_vs_subindustry": 90.0,
                "roc_63": 0.1,
                "roc_126": 0.2,
                "vol_alpha": 1.2,
                "sma_200_dist": 0.15,
                "sma_50_dist": 0.05,
                "rsi_14": 52.0,
                "days_to_next_earnings": 5.0,
                "days_since_last_earnings": 3.0,
                "last_earnings_gap_pct": 0.04,
                "last_earnings_volume_ratio_20": 1.5,
                "last_earnings_open_vs_20d_high": 0.02,
                "close_vs_last_earnings_close": 0.01,
                "avg_abs_gap_pct_20": 0.01,
                "max_gap_down_pct_60": 0.02,
                "distance_above_20d_high": 0.01,
                "base_range_pct_20": 0.04,
                "base_atr_contraction_20": 0.8,
                "base_volume_dryup_ratio_20": 0.7,
                "breakout_volume_ratio_50": 1.8,
                "sector_pct_above_50": 0.75,
                "sector_pct_above_200": 0.7,
                "sector_median_roc_63": 0.08,
                "passed_any_strategy": True,
                "strategy_pass_count": 1,
                "passed_slots": ["technology"],
                "fwd_return_1d": 0.01,
                "fwd_return_3d": 0.02,
                "fwd_return_5d": 0.03,
                "fwd_return_10d": 0.04,
                "fwd_return_20d": 0.05,
                "alpha_vs_spy_1d": 0.001,
                "alpha_vs_spy_3d": 0.002,
                "alpha_vs_spy_5d": 0.003,
                "alpha_vs_spy_10d": 0.004,
                "alpha_vs_spy_20d": 0.005,
                "alpha_vs_sector_1d": 0.006,
                "alpha_vs_sector_3d": 0.007,
                "alpha_vs_sector_5d": 0.008,
                "alpha_vs_sector_10d": 0.009,
                "alpha_vs_sector_20d": 0.01,
                "mfe_20d": 0.12,
                "mae_20d": -0.05,
                "details": {"example": True},
            }

            with patch.object(manager, "duckdb_connection", return_value=fake_duckdb):
                inserted = manager.replace_universe_daily_snapshots(snapshot_date="2026-05-12", rows=[row])

            self.assertEqual(inserted, 1)
            statement, rows = fake_duckdb.executemany_calls[-1]
            placeholder_count = statement.count("?")
            self.assertEqual(placeholder_count, len(rows[0]))

    def test_replace_analyst_snapshots_builds_matching_insert_placeholders(self) -> None:
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
                inserted = manager.replace_analyst_snapshots(
                    snapshot_date="2026-06-23",
                    provider="yfinance",
                    rows=[
                        {
                            "ticker": "AAA",
                            "captured_at": "2026-06-23T21:00:00+00:00",
                            "target_mean": 65.0,
                            "target_median": 63.0,
                            "target_low": 45.0,
                            "target_high": 80.0,
                            "analyst_count": 12,
                            "recommendation": "2 strong buy, 6 buy, 4 hold",
                            "details": {"source": "research", "has_context": True},
                        }
                    ],
                )

            self.assertEqual(inserted, 1)
            statement, rows = fake_duckdb.executemany_calls[-1]
            placeholder_count = statement.count("?")
            self.assertEqual(placeholder_count, len(rows[0]))

    def test_replace_analyst_revision_snapshots_builds_matching_insert_placeholders(self) -> None:
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
                inserted = manager.replace_analyst_revision_snapshots(
                    snapshot_date="2026-06-23",
                    provider="yfinance",
                    rows=[
                        {
                            "ticker": "AAA",
                            "captured_at": "2026-06-23T21:00:00+00:00",
                            "earnings_estimate": [{"period": "0q", "avg": 1.23}],
                            "revenue_estimate": [{"period": "0q", "avg": 100.0}],
                            "eps_trend": [{"period": "0q", "current": 1.23}],
                            "eps_revisions": [{"period": "0q", "upLast7days": 2}],
                            "growth_estimates": [{"period": "0q", "growth": 0.1}],
                            "upgrades_downgrades": [{"date": "2026-06-23", "action": "up"}],
                            "details": {"source": "research", "has_context": True},
                        }
                    ],
                )

            self.assertEqual(inserted, 1)
            statement, rows = fake_duckdb.executemany_calls[-1]
            placeholder_count = statement.count("?")
            self.assertEqual(placeholder_count, len(rows[0]))
