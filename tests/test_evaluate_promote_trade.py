from __future__ import annotations

from pathlib import Path
import json
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from src.evaluate.service import EvaluateService
from src.promote.service import PromoteService
from src.settings import AppPaths
from src.trade.service import TradeService
from src.utils.db_manager import BacktestResultRow, DatabaseManager
from src.utils.strategy import ExitRules, ProductionStrategy, clear_strategy_caches


class FakeDuckDBConnection:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None

    def execute(self, statement: str, *args, **kwargs):
        return self

    def df(self) -> pd.DataFrame:
        return pd.DataFrame()


class FakeMarketDataClient:
    def __init__(self, frame: pd.DataFrame | list[pd.DataFrame]) -> None:
        self.frames = frame if isinstance(frame, list) else [frame]
        self.calls = 0

    def download_daily_history(self, tickers: list[str], start_date, end_date=None) -> pd.DataFrame:
        index = min(self.calls, len(self.frames) - 1)
        self.calls += 1
        return self.frames[index].copy()


class EvaluateServiceTests(unittest.TestCase):
    def test_evaluate_caps_top_section_sector_repetition(self) -> None:
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
            db = DatabaseManager(paths)
            with patch.object(db, "duckdb_connection", return_value=FakeDuckDBConnection()):
                db.initialize()
            db.insert_backtest_results(
                [
                    BacktestResultRow(
                        run_id=5,
                        strategy_id=101,
                        params_json=json.dumps({"indicators": {"a": 1}, "exit_rules": {}, "sector": "Energy"}),
                        norm_score=None,
                        profit_factor=2.0,
                        expectancy=0.12,
                        mdd=0.10,
                        win_rate=0.6,
                        trade_count=150,
                    ),
                    BacktestResultRow(
                        run_id=5,
                        strategy_id=102,
                        params_json=json.dumps({"indicators": {"a": 2}, "exit_rules": {}, "sector": "Energy"}),
                        norm_score=None,
                        profit_factor=1.9,
                        expectancy=0.11,
                        mdd=0.11,
                        win_rate=0.59,
                        trade_count=150,
                    ),
                    BacktestResultRow(
                        run_id=5,
                        strategy_id=103,
                        params_json=json.dumps({"indicators": {"a": 3}, "exit_rules": {}, "sector": "Energy"}),
                        norm_score=None,
                        profit_factor=1.8,
                        expectancy=0.10,
                        mdd=0.12,
                        win_rate=0.58,
                        trade_count=150,
                    ),
                    BacktestResultRow(
                        run_id=5,
                        strategy_id=104,
                        params_json=json.dumps({"indicators": {"b": 1}, "exit_rules": {}, "sector": "Materials"}),
                        norm_score=None,
                        profit_factor=1.7,
                        expectancy=0.09,
                        mdd=0.13,
                        win_rate=0.57,
                        trade_count=150,
                    ),
                ]
            )
            service = EvaluateService(db)
            service._build_live_candidate_metadata = lambda frame: ({}, {})

            with patch.object(db, "duckdb_connection", return_value=FakeDuckDBConnection()):
                service.run(top=3, run_id=5, min_trades=12)

            report_text = (paths.reports_dir / "candidates.md").read_text(encoding="utf-8")
            top_section = report_text.split("## Top Ranked Candidates", 1)[1].split("## Top Live Match Candidates", 1)[0]
            self.assertEqual(top_section.count("- sector: Energy"), 2)
            self.assertEqual(top_section.count("- sector: Materials"), 1)

    def test_evaluate_uses_latest_run_and_min_trade_filter_in_report(self) -> None:
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
            db = DatabaseManager(paths)
            with patch.object(db, "duckdb_connection", return_value=FakeDuckDBConnection()):
                db.initialize()
            db.insert_backtest_results(
                [
                    BacktestResultRow(
                        run_id=1,
                        strategy_id=1,
                        params_json=json.dumps({"indicators": {}, "exit_rules": {}, "sector": "ALL"}),
                        norm_score=None,
                        profit_factor=1.0,
                        expectancy=0.1,
                        mdd=0.2,
                        win_rate=0.5,
                        trade_count=5,
                    ),
                    BacktestResultRow(
                        run_id=2,
                        strategy_id=2,
                        params_json=json.dumps({"indicators": {}, "exit_rules": {}, "sector": "ALL"}),
                        norm_score=None,
                        profit_factor=2.0,
                        expectancy=0.3,
                        mdd=0.1,
                        win_rate=0.6,
                        trade_count=12,
                    ),
                    BacktestResultRow(
                        run_id=2,
                        strategy_id=3,
                        params_json=json.dumps({"indicators": {}, "exit_rules": {}, "sector": "ALL"}),
                        norm_score=None,
                        profit_factor=float("inf"),
                        expectancy=0.25,
                        mdd=0.05,
                        win_rate=1.0,
                        trade_count=4,
                    ),
                ]
            )
            service = EvaluateService(db)
            service._build_live_candidate_metadata = lambda frame: ({}, {})

            with patch.object(db, "duckdb_connection", return_value=FakeDuckDBConnection()):
                report = service.run(top=2)

            self.assertEqual(report.rows_written, 1)
            rows = db.list_backtest_results(run_id=2)
            scores = {row["strategy_id"]: row["norm_score"] for row in rows}
            self.assertAlmostEqual(scores[2], 0.0)
            report_text = (paths.reports_dir / "candidates.md").read_text(encoding="utf-8")
            self.assertIn("- run_id: 2", report_text)
            self.assertIn("- min_trades: 12", report_text)
            self.assertIn("- ranking: practical_score desc, then norm_score desc, then expectancy desc", report_text)
            self.assertIn("## Top Ranked Candidates", report_text)
            self.assertIn("## Top Live Match Candidates", report_text)
            self.assertIn("## Best Practical Live Candidates", report_text)
            self.assertIn("## Best Candidate Per Sector", report_text)
            self.assertIn("## Best Live Candidate Per Sector", report_text)
            self.assertIn("## Best Promotable Candidate Per Sector", report_text)
            self.assertIn("## Best Promotable Portfolio Pairs", report_text)
            self.assertIn("No strategies currently have live matches.", report_text)
            self.assertIn("strategy_id: 2", report_text)
            self.assertNotIn("strategy_id: 1", report_text)
            self.assertNotIn("strategy_id: 3", report_text)
            self.assertIn("global_rank: 1", report_text)
            self.assertIn("sector_rank: 1", report_text)
            self.assertIn("trade_count: 12", report_text)
            self.assertIn("live_match_count: 0", report_text)
            self.assertIn("promotion_policy_passed: no", report_text)
            self.assertIn("promotion_policy_violations: trade_count 12 < 100", report_text)
            self.assertIn("gate_counts: unavailable", report_text)
            self.assertIn("first_zero_gate: unavailable", report_text)
            self.assertIn("practical_score:", report_text)
            self.assertIn("warnings: low_trade_count, no_current_live_matches", report_text)

    def test_evaluate_handles_infinite_profit_factor_when_threshold_allows_row(self) -> None:
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
            db = DatabaseManager(paths)
            with patch.object(db, "duckdb_connection", return_value=FakeDuckDBConnection()):
                db.initialize()
            db.insert_backtest_results(
                [
                    BacktestResultRow(
                        run_id=1,
                        strategy_id=10,
                        params_json=json.dumps({"indicators": {}, "exit_rules": {}, "sector": "ALL"}),
                        norm_score=None,
                        profit_factor=float("inf"),
                        expectancy=0.2,
                        mdd=0.01,
                        win_rate=1.0,
                        trade_count=12,
                    ),
                    BacktestResultRow(
                        run_id=1,
                        strategy_id=11,
                        params_json=json.dumps({"indicators": {}, "exit_rules": {}, "sector": "ALL"}),
                        norm_score=None,
                        profit_factor=3.0,
                        expectancy=0.15,
                        mdd=0.02,
                        win_rate=0.7,
                        trade_count=12,
                    ),
                ]
            )
            service = EvaluateService(db)
            service._build_live_candidate_metadata = lambda frame: ({}, {})

            with patch.object(db, "duckdb_connection", return_value=FakeDuckDBConnection()):
                report = service.run(top=2, min_trades=12)

            self.assertEqual(report.rows_written, 2)
            report_text = (paths.reports_dir / "candidates.md").read_text(encoding="utf-8")
            self.assertIn("profit_factor: inf", report_text)
            self.assertIn("practical_score:", report_text)
            self.assertIn("No strategies currently have live matches.", report_text)
            self.assertIn("## Best Practical Live Candidates", report_text)
            self.assertIn("## Best Promotable Portfolio Pairs", report_text)
            self.assertIn("warnings: low_trade_count, profit_factor_infinite, no_current_live_matches", report_text)

    def test_evaluate_dedupes_plateau_equivalent_rows_in_report(self) -> None:
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
            db = DatabaseManager(paths)
            with patch.object(db, "duckdb_connection", return_value=FakeDuckDBConnection()):
                db.initialize()
            db.insert_backtest_results(
                [
                    BacktestResultRow(
                        run_id=3,
                        strategy_id=20,
                        params_json=json.dumps({"indicators": {"a": 1}, "exit_rules": {}, "sector": "Energy"}),
                        norm_score=None,
                        profit_factor=5.0,
                        expectancy=0.2,
                        mdd=0.03,
                        win_rate=0.7,
                        trade_count=12,
                    ),
                    BacktestResultRow(
                        run_id=3,
                        strategy_id=21,
                        params_json=json.dumps({"indicators": {"a": 2}, "exit_rules": {}, "sector": "Energy"}),
                        norm_score=None,
                        profit_factor=5.0,
                        expectancy=0.2,
                        mdd=0.03,
                        win_rate=0.7,
                        trade_count=12,
                    ),
                    BacktestResultRow(
                        run_id=3,
                        strategy_id=22,
                        params_json=json.dumps({"indicators": {"a": 3}, "exit_rules": {}, "sector": "Materials"}),
                        norm_score=None,
                        profit_factor=4.0,
                        expectancy=0.15,
                        mdd=0.02,
                        win_rate=0.65,
                        trade_count=12,
                    ),
                ]
            )
            service = EvaluateService(db)
            service._build_live_candidate_metadata = lambda frame: ({}, {})

            with patch.object(db, "duckdb_connection", return_value=FakeDuckDBConnection()):
                report = service.run(top=10, run_id=3, min_trades=12)

            self.assertEqual(report.rows_written, 2)
            report_text = (paths.reports_dir / "candidates.md").read_text(encoding="utf-8")
            self.assertIn("## Top Ranked Candidates", report_text)
            self.assertIn("## Top Live Match Candidates", report_text)
            self.assertIn("## Best Practical Live Candidates", report_text)
            self.assertIn("## Best Candidate Per Sector", report_text)
            self.assertIn("## Best Promotable Candidate Per Sector", report_text)
            self.assertIn("## Best Promotable Portfolio Pairs", report_text)
            self.assertIn("No strategies currently have live matches.", report_text)
            top_ranked_section = report_text.split("## Top Ranked Candidates", maxsplit=1)[1].split("## Top Live Match Candidates", maxsplit=1)[0]
            self.assertEqual(top_ranked_section.count("### Result "), 2)
            self.assertIn("duplicate_group_size: 2", report_text)
            self.assertIn("collapsed_result_ids: 1, 2", report_text)
            self.assertIn("No portfolio pairs currently satisfy promotion policy.", report_text)

    def test_gate_diagnostic_identifies_first_zero_gate(self) -> None:
        service = EvaluateService(DatabaseManager(AppPaths(
            root_dir=Path("."),
            data_dir=Path("data"),
            duckdb_path=Path("data/market_data.duckdb"),
            sqlite_path=Path("data/ledger.sqlite"),
            reports_dir=Path("reports"),
            logs_dir=Path("logs"),
            config_path=Path("config.yaml"),
            env_path=Path(".env"),
            production_strategy_path=Path("production_strategy.json"),
        )))
        full_snapshot = pd.DataFrame(
            [
                {
                    "ticker": "AAA",
                    "sector": "Energy",
                    "regime_green": True,
                    "rsi_14": 30.0,
                    "vol_alpha": 1.3,
                    "relative_strength_index_vs_spy": 90.0,
                },
                {
                    "ticker": "BBB",
                    "sector": "Energy",
                    "regime_green": True,
                    "rsi_14": 40.0,
                    "vol_alpha": 1.6,
                    "relative_strength_index_vs_spy": 85.0,
                },
                {
                    "ticker": "CCC",
                    "sector": "Industrials",
                    "regime_green": False,
                    "rsi_14": 25.0,
                    "vol_alpha": 2.0,
                    "relative_strength_index_vs_spy": 95.0,
                },
            ]
        )
        regime_snapshot = full_snapshot[full_snapshot["regime_green"]].copy()

        diagnostic = service._build_gate_diagnostic(
            full_snapshot=full_snapshot,
            regime_snapshot=regime_snapshot,
            indicators={
                "rsi_14_max": 35.0,
                "vol_alpha_min": 1.5,
                "relative_strength_index_vs_spy_min": 80.0,
                "signal_score_min": 25.0,
            },
            sector="Energy",
        )

        self.assertEqual(
            diagnostic["counts"],
            [
                ("universe", 3),
                ("regime_green", 2),
                ("sector_scope", 2),
                ("relative_strength_index_vs_spy_min", 2),
                ("signal_score_min", 0),
            ],
        )
        self.assertEqual(
            diagnostic["component_positive_counts"],
            [("rsi_14_max", 2), ("vol_alpha_min", 2)],
        )
        self.assertEqual(diagnostic["first_zero_gate"], "signal_score_min")

    def test_live_match_bonus_surfaces_live_candidate_in_practical_live_section(self) -> None:
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
            db = DatabaseManager(paths)
            with patch.object(db, "duckdb_connection", return_value=FakeDuckDBConnection()):
                db.initialize()
            db.insert_backtest_results(
                [
                    BacktestResultRow(
                        run_id=7,
                        strategy_id=100,
                        params_json=json.dumps({"indicators": {}, "exit_rules": {}, "sector": "ALL"}),
                        norm_score=None,
                        profit_factor=10.0,
                        expectancy=0.20,
                        mdd=0.01,
                        win_rate=0.90,
                        trade_count=13,
                    ),
                    BacktestResultRow(
                        run_id=7,
                        strategy_id=101,
                        params_json=json.dumps({"indicators": {}, "exit_rules": {}, "sector": "ALL"}),
                        norm_score=None,
                        profit_factor=2.0,
                        expectancy=0.10,
                        mdd=0.20,
                        win_rate=0.55,
                        trade_count=30,
                    ),
                ]
            )
            service = EvaluateService(db)
            service._build_live_candidate_metadata = lambda frame: (
                {
                    int(frame.iloc[0]["id"]): {"count": 0, "tickers": [], "links": []},
                    int(frame.iloc[1]["id"]): {"count": 1, "tickers": ["AAA"], "links": ["https://www.tradingview.com/chart/?symbol=AAA"]},
                },
                {},
            )

            with patch.object(db, "duckdb_connection", return_value=FakeDuckDBConnection()):
                report = service.run(top=2, run_id=7, min_trades=12)

            self.assertEqual(report.rows_written, 2)
            report_text = (paths.reports_dir / "candidates.md").read_text(encoding="utf-8")
            self.assertIn("## Best Practical Live Candidates", report_text)
            self.assertIn("## Best Live Candidate Per Sector", report_text)
            self.assertIn("## Best Promotable Portfolio Pairs", report_text)
            best_live_section = report_text.split("## Best Practical Live Candidates", maxsplit=1)[1]
            self.assertIn("strategy_id: 101", best_live_section)
            self.assertIn("## Best Practical Live Candidates", report_text)
            self.assertIn("live_match_tickers: AAA", report_text)

    def test_alpha_bonus_can_lift_higher_alpha_row_in_practical_ranking(self) -> None:
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
            db = DatabaseManager(paths)
            with patch.object(db, "duckdb_connection", return_value=FakeDuckDBConnection()):
                db.initialize()
            db.insert_backtest_results(
                [
                    BacktestResultRow(
                        run_id=8,
                        strategy_id=200,
                        params_json=json.dumps({"indicators": {}, "exit_rules": {}, "sector": "Materials"}),
                        norm_score=None,
                        profit_factor=1.55,
                        expectancy=0.098,
                        mdd=0.10,
                        win_rate=0.55,
                        trade_count=100,
                        alpha_vs_spy=0.020,
                        alpha_vs_sector=0.020,
                    ),
                    BacktestResultRow(
                        run_id=8,
                        strategy_id=201,
                        params_json=json.dumps({"indicators": {}, "exit_rules": {}, "sector": "Materials"}),
                        norm_score=None,
                        profit_factor=1.52,
                        expectancy=0.100,
                        mdd=0.10,
                        win_rate=0.55,
                        trade_count=100,
                        alpha_vs_spy=0.0,
                        alpha_vs_sector=0.0,
                    ),
                ]
            )
            service = EvaluateService(db)
            service._build_live_candidate_metadata = lambda frame: ({}, {})

            with patch.object(db, "duckdb_connection", return_value=FakeDuckDBConnection()):
                service.run(top=2, run_id=8, min_trades=12)

            report_text = (paths.reports_dir / "candidates.md").read_text(encoding="utf-8")
            top_ranked_section = report_text.split("## Top Ranked Candidates", maxsplit=1)[1].split("## Top Live Match Candidates", maxsplit=1)[0]
            self.assertIn("strategy_id: 200", top_ranked_section)
            self.assertLess(top_ranked_section.find("strategy_id: 200"), top_ranked_section.find("strategy_id: 201"))
            self.assertIn("alpha_vs_spy: 0.020000", report_text)
            self.assertIn("alpha_vs_sector: 0.020000", report_text)

    def test_evaluate_can_render_walk_forward_stability_section(self) -> None:
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
            db = DatabaseManager(paths)
            with patch.object(db, "duckdb_connection", return_value=FakeDuckDBConnection()):
                db.initialize()
            db.insert_backtest_results(
                [
                    BacktestResultRow(
                        run_id=9,
                        strategy_id=301,
                        params_json=json.dumps({"indicators": {}, "exit_rules": {}, "sector": "Materials"}),
                        norm_score=None,
                        profit_factor=1.8,
                        expectancy=0.08,
                        mdd=0.10,
                        win_rate=0.6,
                        trade_count=120,
                        alpha_vs_spy=0.01,
                        alpha_vs_sector=0.01,
                    )
                ]
            )
            service = EvaluateService(db)
            service._build_live_candidate_metadata = lambda frame: ({}, {})
            service._build_walk_forward_stability = lambda **kwargs: pd.DataFrame(
                [
                    {
                        "id": 1,
                        "wf_window_count": 5,
                        "wf_median_expectancy": 0.02,
                        "wf_worst_expectancy": -0.01,
                        "wf_positive_window_ratio": 0.8,
                        "wf_positive_alpha_window_ratio": 0.6,
                        "wf_median_alpha_vs_spy": 0.01,
                        "wf_worst_mdd": 0.12,
                        "wf_trade_count_min": 14,
                        "wf_stability_score": 0.55,
                    }
                ]
            )

            with patch.object(db, "duckdb_connection", return_value=FakeDuckDBConnection()):
                report = service.run(top=5, run_id=9, min_trades=12, walk_forward=True)

            self.assertEqual(report.rows_written, 1)
            report_text = (paths.reports_dir / "candidates.md").read_text(encoding="utf-8")
            self.assertIn("- walk_forward: enabled on shortlist=25 with rolling_windows=5", report_text)
            self.assertIn("## Best Walk-Forward Stability Candidates", report_text)
            self.assertIn("wf_stability_score: 0.550000", report_text)
            self.assertIn("wf_window_count: 5", report_text)
            self.assertIn("wf_positive_window_ratio: 0.800000", report_text)
            self.assertIn("wf_positive_alpha_window_ratio: 0.600000", report_text)

    def test_evaluate_walk_forward_metrics_can_fail_promotion_policy(self) -> None:
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
            db = DatabaseManager(paths)
            with patch.object(db, "duckdb_connection", return_value=FakeDuckDBConnection()):
                db.initialize()
            db.insert_backtest_results(
                [
                    BacktestResultRow(
                        run_id=10,
                        strategy_id=302,
                        params_json=json.dumps({"indicators": {}, "exit_rules": {}, "sector": "Materials"}),
                        norm_score=None,
                        profit_factor=1.8,
                        expectancy=0.08,
                        mdd=0.10,
                        win_rate=0.6,
                        trade_count=120,
                        alpha_vs_spy=0.01,
                        alpha_vs_sector=0.01,
                    )
                ]
            )
            service = EvaluateService(db)
            service._build_live_candidate_metadata = lambda frame: ({}, {})
            service._build_walk_forward_stability = lambda **kwargs: pd.DataFrame(
                [
                    {
                        "id": 1,
                        "wf_window_count": 5,
                        "wf_median_expectancy": -0.001,
                        "wf_worst_expectancy": -0.02,
                        "wf_positive_window_ratio": 0.4,
                        "wf_positive_alpha_window_ratio": 0.2,
                        "wf_median_alpha_vs_spy": -0.01,
                        "wf_worst_mdd": 0.35,
                        "wf_trade_count_min": 4,
                        "wf_stability_score": -0.10,
                    }
                ]
            )

            with patch("src.utils.promotion_policy.load_feature_config", return_value={
                "promotion_policy": {
                    "min_profit_factor": 1.30,
                    "min_expectancy": 0.004,
                    "min_trade_count": 100,
                    "max_mdd": 0.30,
                    "walk_forward": {
                        "windows": 5,
                        "min_window_count": 5,
                        "min_positive_window_ratio": 0.60,
                        "min_positive_alpha_window_ratio": 0.55,
                        "min_median_expectancy": 0.0,
                        "max_worst_mdd": 0.30,
                        "min_trade_count_min": 8,
                    },
                }
            }), patch.object(db, "duckdb_connection", return_value=FakeDuckDBConnection()):
                service.run(top=5, run_id=10, min_trades=12, walk_forward=True)

            report_text = (paths.reports_dir / "candidates.md").read_text(encoding="utf-8")
            self.assertIn("promotion_policy_passed: no", report_text)
            self.assertIn("wf_positive_window_ratio 0.4 < 0.600000", report_text)
            self.assertIn("wf_worst_mdd 0.35 > 0.300000", report_text)

    def test_walk_forward_summary_rewards_consistency(self) -> None:
        service = EvaluateService(DatabaseManager(AppPaths(
            root_dir=Path("."),
            data_dir=Path("data"),
            duckdb_path=Path("data/market_data.duckdb"),
            sqlite_path=Path("data/ledger.sqlite"),
            reports_dir=Path("reports"),
            logs_dir=Path("logs"),
            config_path=Path("config.yaml"),
            env_path=Path(".env"),
            production_strategy_path=Path("production_strategy.json"),
        )))

        stronger = service._score_walk_forward_summary(
            median_expectancy=0.01,
            worst_expectancy=0.002,
            positive_window_ratio=0.8,
            positive_alpha_window_ratio=0.8,
            worst_mdd=0.15,
            trade_count_min=15,
        )
        weaker = service._score_walk_forward_summary(
            median_expectancy=0.005,
            worst_expectancy=-0.01,
            positive_window_ratio=0.4,
            positive_alpha_window_ratio=0.2,
            worst_mdd=0.30,
            trade_count_min=5,
        )

        self.assertGreater(stronger, weaker)


class PromoteAndTradeTests(unittest.TestCase):
    def test_promote_writes_fully_formed_production_strategy(self) -> None:
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
            db = DatabaseManager(paths)
            with patch.object(db, "duckdb_connection", return_value=FakeDuckDBConnection()):
                db.initialize()
            db.insert_backtest_results(
                [
                    BacktestResultRow(
                        run_id=1,
                        strategy_id=402,
                        params_json=json.dumps(
                            {
                                "indicators": {"rsi_14_max": 35, "vol_alpha_min": 1.5},
                                "exit_rules": {
                                    "trailing_stop_pct": None,
                                    "profit_target_pct": None,
                                    "time_limit_days": 20,
                                    "trailing_stop_atr_mult": 2.5,
                                    "profit_target_atr_mult": 3.0,
                                },
                                "sector": "ALL",
                            }
                        ),
                        norm_score=0.7,
                        profit_factor=1.8,
                        expectancy=0.15,
                        mdd=0.12,
                        win_rate=0.55,
                        trade_count=150,
                    )
                ]
            )

            with patch.object(db, "duckdb_connection", return_value=FakeDuckDBConnection()), \
                 patch.object(PromoteService, "_compute_walk_forward_metrics", return_value={
                     "wf_window_count": 5,
                     "wf_median_expectancy": 0.01,
                     "wf_worst_expectancy": 0.0,
                     "wf_positive_window_ratio": 0.8,
                     "wf_positive_alpha_window_ratio": 0.6,
                     "wf_worst_mdd": 0.15,
                     "wf_trade_count_min": 12,
                 }):
                message = PromoteService(db).run(row_id=1)

            payload = json.loads(paths.production_strategy_path.read_text(encoding="utf-8"))
            strategies_payload = json.loads((root / "production_strategies.json").read_text(encoding="utf-8"))
            self.assertEqual(message, "Strategy 1 promoted into slot 'default'. production_strategies.json updated.")
            self.assertEqual(payload["strategy_id"], 402)
            self.assertIn("promoted_at", payload)
            self.assertEqual(payload["sector"], "ALL")
            self.assertEqual(payload["indicators"]["rsi_14_max"], 35.0)
            self.assertEqual(payload["exit_rules"]["time_limit_days"], 20)
            self.assertEqual(payload["exit_rules"]["trailing_stop_atr_mult"], 2.5)
            self.assertEqual(payload["exit_rules"]["profit_target_atr_mult"], 3.0)
            self.assertEqual(strategies_payload["strategies"]["default"]["strategy_id"], 402)

    def test_promote_rejects_result_that_fails_promotion_policy(self) -> None:
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
            db = DatabaseManager(paths)
            with patch.object(db, "duckdb_connection", return_value=FakeDuckDBConnection()):
                db.initialize()
            db.insert_backtest_results(
                [
                    BacktestResultRow(
                        run_id=1,
                        strategy_id=501,
                        params_json=json.dumps({"indicators": {"rsi_14_max": 35}, "exit_rules": {"time_limit_days": 20}, "sector": "ALL"}),
                        norm_score=0.2,
                        profit_factor=1.10,
                        expectancy=0.001,
                        mdd=0.40,
                        win_rate=0.51,
                        trade_count=40,
                    )
                ]
            )

            with patch.object(db, "duckdb_connection", return_value=FakeDuckDBConnection()):
                with self.assertRaisesRegex(ValueError, "does not satisfy promotion policy"):
                    PromoteService(db).run(row_id=1)

    def test_promote_rejects_result_that_fails_walk_forward_policy(self) -> None:
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
            db = DatabaseManager(paths)
            with patch.object(db, "duckdb_connection", return_value=FakeDuckDBConnection()):
                db.initialize()
            db.insert_backtest_results(
                [
                    BacktestResultRow(
                        run_id=1,
                        strategy_id=502,
                        params_json=json.dumps({"indicators": {"rsi_14_max": 35}, "exit_rules": {"time_limit_days": 20}, "sector": "ALL"}),
                        norm_score=0.7,
                        profit_factor=1.8,
                        expectancy=0.02,
                        mdd=0.12,
                        win_rate=0.55,
                        trade_count=140,
                        alpha_vs_spy=0.01,
                        alpha_vs_sector=0.01,
                    )
                ]
            )

            with patch("src.utils.promotion_policy.load_feature_config", return_value={
                "promotion_policy": {
                    "min_profit_factor": 1.30,
                    "min_expectancy": 0.004,
                    "min_trade_count": 100,
                    "max_mdd": 0.30,
                    "walk_forward": {
                        "windows": 5,
                        "min_window_count": 5,
                        "min_positive_window_ratio": 0.60,
                        "min_positive_alpha_window_ratio": 0.55,
                        "min_median_expectancy": 0.0,
                        "max_worst_mdd": 0.30,
                        "min_trade_count_min": 8,
                    },
                }
            }), patch.object(db, "duckdb_connection", return_value=FakeDuckDBConnection()), \
                 patch.object(PromoteService, "_compute_walk_forward_metrics", return_value={
                     "wf_window_count": 5,
                     "wf_median_expectancy": -0.001,
                     "wf_worst_expectancy": -0.02,
                     "wf_positive_window_ratio": 0.4,
                     "wf_positive_alpha_window_ratio": 0.2,
                     "wf_worst_mdd": 0.35,
                     "wf_trade_count_min": 4,
                 }):
                with self.assertRaisesRegex(ValueError, "wf_positive_window_ratio"):
                    PromoteService(db).run(row_id=1)

    def test_trade_sell_updates_status_exit_fields_and_reports_pnl(self) -> None:
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
            db = DatabaseManager(paths)
            with patch.object(db, "duckdb_connection", return_value=FakeDuckDBConnection()):
                db.initialize()
            db.open_trade(
                ticker="AAA",
                entry_date="2026-05-01",
                entry_price=100.0,
                entry_atr=None,
                strategy_id=1,
                strategy_slot="default",
                shares=10,
                max_price_seen=100.0,
            )

            with patch.object(db, "duckdb_connection", return_value=FakeDuckDBConnection()):
                message = TradeService(db).sell(ticker="AAA", price=112.5)

            self.assertEqual(message, "Realized P&L for AAA: 125.00")
            connection = sqlite3.connect(paths.sqlite_path)
            connection.row_factory = sqlite3.Row
            try:
                row = connection.execute(
                    "SELECT status, exit_date, exit_price FROM Active_Trades WHERE ticker = 'AAA'"
                ).fetchone()
            finally:
                connection.close()
            self.assertEqual(row["status"], "closed")
            self.assertEqual(row["exit_price"], 112.5)
            self.assertIsNotNone(row["exit_date"])

    def test_trade_buy_reuses_existing_open_trade_strategy_for_off_universe_ticker(self) -> None:
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
                production_strategies_path=root / "production_strategies.json",
            )
            db = DatabaseManager(paths)
            with patch.object(db, "duckdb_connection", return_value=FakeDuckDBConnection()):
                db.initialize()
            db.open_trade(
                ticker="RKLB",
                entry_date="2026-05-06",
                entry_price=75.0,
                entry_atr=None,
                strategy_id=172365,
                strategy_slot="technology",
                shares=10,
                max_price_seen=75.0,
            )
            strategy = ProductionStrategy(
                strategy_id=172365,
                promoted_at="2026-05-06T17:00:00",
                indicators={},
                exit_rules=ExitRules(
                    trailing_stop_pct=0.08,
                    profit_target_pct=0.10,
                    time_limit_days=20,
                ),
                slot="technology",
                sector="Information Technology",
            )

            clear_strategy_caches()
            with patch("src.trade.service.load_active_strategies", return_value={"technology": strategy}):
                with patch.object(db, "duckdb_connection", return_value=FakeDuckDBConnection()):
                    message = TradeService(db).buy(ticker="RKLB", price=79.66, shares=60)

            self.assertEqual(message, "Bought RKLB: 60 shares at 79.66 using strategy slot 'technology'")
            latest_trade = db.get_latest_open_trade("RKLB")
            self.assertIsNotNone(latest_trade)
            self.assertEqual(latest_trade["strategy_slot"], "technology")
            self.assertEqual(latest_trade["strategy_id"], 172365)
            self.assertEqual(latest_trade["shares"], 60)

    def test_trade_buy_fetches_recent_history_for_atr_based_off_universe_ticker(self) -> None:
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
                production_strategies_path=root / "production_strategies.json",
            )
            db = DatabaseManager(paths)
            with patch.object(db, "duckdb_connection", return_value=FakeDuckDBConnection()):
                db.initialize()

            db.open_trade(
                ticker="RKLB",
                entry_date="2026-05-06",
                entry_price=75.0,
                entry_atr=None,
                strategy_id=172365,
                strategy_slot="technology",
                shares=10,
                max_price_seen=75.0,
            )

            strategy = ProductionStrategy(
                strategy_id=172365,
                promoted_at="2026-05-06T17:00:00",
                indicators={},
                exit_rules=ExitRules(
                    trailing_stop_pct=None,
                    profit_target_pct=None,
                    time_limit_days=20,
                    trailing_stop_atr_mult=2.5,
                    profit_target_atr_mult=3.0,
                ),
                slot="technology",
                sector="Information Technology",
            )

            trade_dates = pd.date_range("2026-03-01", periods=40, freq="B")
            raw_history = pd.DataFrame(
                {
                    "Open": [60.0 + index * 0.4 for index in range(len(trade_dates))],
                    "High": [61.0 + index * 0.4 for index in range(len(trade_dates))],
                    "Low": [59.0 + index * 0.4 for index in range(len(trade_dates))],
                    "Close": [60.5 + index * 0.4 for index in range(len(trade_dates))],
                    "Adj Close": [60.5 + index * 0.4 for index in range(len(trade_dates))],
                    "Volume": [1_000_000 for _ in range(len(trade_dates))],
                },
                index=trade_dates,
            )
            raw_history.index.name = "Date"

            clear_strategy_caches()
            with patch("src.trade.service.load_active_strategies", return_value={"technology": strategy}):
                with patch.object(db, "duckdb_connection", return_value=FakeDuckDBConnection()):
                    message = TradeService(
                        db,
                        market_data_client=FakeMarketDataClient(raw_history),
                    ).buy(ticker="RKLB", price=79.66, shares=60)

            self.assertEqual(message, "Bought RKLB: 60 shares at 79.66 using strategy slot 'technology'")
            latest_trade = db.get_latest_open_trade("RKLB")
            self.assertIsNotNone(latest_trade)
            self.assertEqual(latest_trade["strategy_slot"], "technology")
            self.assertEqual(latest_trade["strategy_id"], 172365)
            self.assertEqual(latest_trade["shares"], 60)
            self.assertIsNotNone(latest_trade["entry_atr"])
            self.assertGreater(float(latest_trade["entry_atr"]), 0.0)

    def test_trade_buy_retries_shorter_history_windows_for_atr_based_ticker(self) -> None:
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
                production_strategies_path=root / "production_strategies.json",
            )
            db = DatabaseManager(paths)
            with patch.object(db, "duckdb_connection", return_value=FakeDuckDBConnection()):
                db.initialize()
            db.open_trade(
                ticker="NBIS",
                entry_date="2026-05-05",
                entry_price=150.0,
                entry_atr=None,
                strategy_id=172365,
                strategy_slot="technology",
                shares=75,
                max_price_seen=155.0,
            )
            latest_open = db.get_latest_open_trade("NBIS")
            self.assertIsNotNone(latest_open)
            db.close_trade(
                trade_rowid=int(latest_open["rowid"]),
                exit_date="2026-05-06",
                exit_price=160.0,
            )
            strategy = ProductionStrategy(
                strategy_id=172365,
                promoted_at="2026-05-06T17:00:00",
                indicators={},
                exit_rules=ExitRules(
                    trailing_stop_pct=None,
                    profit_target_pct=None,
                    time_limit_days=20,
                    trailing_stop_atr_mult=2.5,
                    profit_target_atr_mult=3.0,
                ),
                slot="technology",
                sector="Information Technology",
            )
            trade_dates = pd.date_range("2026-03-01", periods=40, freq="B")
            fallback_history = pd.DataFrame(
                {
                    "Open": [140.0 + index * 0.8 for index in range(len(trade_dates))],
                    "High": [141.5 + index * 0.8 for index in range(len(trade_dates))],
                    "Low": [138.5 + index * 0.8 for index in range(len(trade_dates))],
                    "Close": [140.7 + index * 0.8 for index in range(len(trade_dates))],
                    "Adj Close": [140.7 + index * 0.8 for index in range(len(trade_dates))],
                    "Volume": [800_000 for _ in range(len(trade_dates))],
                },
                index=trade_dates,
            )
            fallback_history.index.name = "Date"
            market_data_client = FakeMarketDataClient([pd.DataFrame(), fallback_history])

            clear_strategy_caches()
            with patch("src.trade.service.load_active_strategies", return_value={"technology": strategy}):
                with patch.object(db, "duckdb_connection", return_value=FakeDuckDBConnection()):
                    message = TradeService(
                        db,
                        market_data_client=market_data_client,
                    ).buy(ticker="NBIS", price=178.22, shares=120)

            self.assertEqual(message, "Bought NBIS: 120 shares at 178.22 using strategy slot 'technology'")
            self.assertGreaterEqual(market_data_client.calls, 2)
            latest_trade = db.get_latest_trade("NBIS")
            self.assertIsNotNone(latest_trade)
            self.assertEqual(latest_trade["status"], "open")
            self.assertIsNotNone(latest_trade["entry_atr"])
            self.assertGreater(float(latest_trade["entry_atr"]), 0.0)

    def test_trade_buy_reuses_latest_closed_trade_strategy_for_off_universe_ticker(self) -> None:
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
                production_strategies_path=root / "production_strategies.json",
            )
            db = DatabaseManager(paths)
            with patch.object(db, "duckdb_connection", return_value=FakeDuckDBConnection()):
                db.initialize()
            db.open_trade(
                ticker="NBIS",
                entry_date="2026-05-05",
                entry_price=150.0,
                entry_atr=None,
                strategy_id=172365,
                strategy_slot="technology",
                shares=75,
                max_price_seen=155.0,
            )
            latest_open = db.get_latest_open_trade("NBIS")
            self.assertIsNotNone(latest_open)
            db.close_trade(
                trade_rowid=int(latest_open["rowid"]),
                exit_date="2026-05-06",
                exit_price=160.0,
            )
            strategy = ProductionStrategy(
                strategy_id=172365,
                promoted_at="2026-05-06T17:00:00",
                indicators={},
                exit_rules=ExitRules(
                    trailing_stop_pct=0.08,
                    profit_target_pct=0.10,
                    time_limit_days=20,
                ),
                slot="technology",
                sector="Information Technology",
            )

            clear_strategy_caches()
            with patch("src.trade.service.load_active_strategies", return_value={"technology": strategy}):
                with patch.object(db, "duckdb_connection", return_value=FakeDuckDBConnection()):
                    message = TradeService(db).buy(ticker="NBIS", price=178.22, shares=120)

            self.assertEqual(message, "Bought NBIS: 120 shares at 178.22 using strategy slot 'technology'")
            latest_trade = db.get_latest_trade("NBIS")
            self.assertIsNotNone(latest_trade)
            self.assertEqual(latest_trade["status"], "open")
            self.assertEqual(latest_trade["strategy_slot"], "technology")
            self.assertEqual(latest_trade["strategy_id"], 172365)
            self.assertEqual(latest_trade["shares"], 120)
