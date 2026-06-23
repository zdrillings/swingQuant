from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src.cli import build_parser
from src.scan.portfolio_rotation_service import PortfolioRotationService
from src.settings import AppPaths


class PortfolioRotationServiceTests(unittest.TestCase):
    def test_portfolio_rotation_fills_empty_slots_and_exits_after_max_hold(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            reports_dir = root / "reports"
            reports_dir.mkdir(parents=True, exist_ok=True)

            class FakeDB:
                def __init__(self):
                    self.paths = AppPaths(
                        root_dir=root,
                        data_dir=root / "data",
                        duckdb_path=root / "data" / "market_data.duckdb",
                        sqlite_path=root / "data" / "ledger.sqlite",
                        reports_dir=reports_dir,
                        logs_dir=root / "logs",
                        config_path=root / "config.yaml",
                        env_path=root / ".env",
                        production_strategy_path=root / "production_strategy.json",
                    )

                def initialize(self): return None

                def load_scan_candidates(self, scan_date=None):
                    rows = []
                    scan_dates = pd.bdate_range("2026-05-01", periods=5)
                    for index, scan_date in enumerate(scan_dates):
                        rows.extend(
                            [
                                {
                                    "scan_date": scan_date.strftime("%Y-%m-%d"),
                                    "ticker": "AAA",
                                    "sector": "Technology",
                                    "opportunity_score": 0.45,
                                    "overlap_penalty": 0.0,
                                    "model_predicted_alpha": 0.20,
                                },
                                {
                                    "scan_date": scan_date.strftime("%Y-%m-%d"),
                                    "ticker": "BBB",
                                    "sector": "Technology",
                                    "opportunity_score": 0.44,
                                    "overlap_penalty": 0.0,
                                    "model_predicted_alpha": 0.10,
                                },
                                {
                                    "scan_date": scan_date.strftime("%Y-%m-%d"),
                                    "ticker": "CCC",
                                    "sector": "Industrials",
                                    "opportunity_score": 0.43,
                                    "overlap_penalty": 0.0,
                                    "model_predicted_alpha": 0.08 + index * 0.01,
                                },
                            ]
                        )
                    return pd.DataFrame(rows)

                def load_price_history(self, tickers):
                    rows = []
                    for ticker, base in (("AAA", 10.0), ("BBB", 20.0), ("CCC", 30.0), ("SPY", 100.0)):
                        for index, day in enumerate(pd.bdate_range("2026-05-01", periods=5)):
                            rows.append(
                                {
                                    "ticker": ticker,
                                    "date": day.strftime("%Y-%m-%d"),
                                    "adj_close": base + index,
                                }
                            )
                    return pd.DataFrame(rows)

            report = PortfolioRotationService(FakeDB()).run(target_positions=2, max_hold_days=2)

            self.assertTrue(report.output_path.endswith("portfolio_rotation.md"))
            self.assertGreater(report.final_equity, 1.0)
            report_text = (reports_dir / "portfolio_rotation.md").read_text(encoding="utf-8")
            self.assertIn("target_positions: 2", report_text)
            self.assertIn("sell_rule: max_hold_days only in v1", report_text)
            self.assertIn("transaction_cost_bps: 0.00", report_text)
            self.assertIn("slippage_bps: 0.00", report_text)
            self.assertIn("cooldown_days: 0", report_text)
            self.assertIn("reinvest_gains: true", report_text)
            self.assertIn("max_new_entries_per_scan: unlimited", report_text)
            self.assertIn("max_hold_days", report_text)
            self.assertIn("## Primary Scorecard", report_text)
            self.assertIn("### Rolling Portfolio Windows", report_text)
            self.assertIn("### Calendar Quarter Returns", report_text)
            self.assertTrue((reports_dir / "portfolio_rotation_rolling_windows.csv").exists())
            self.assertTrue((reports_dir / "portfolio_rotation_quarterly_returns.csv").exists())

    def test_portfolio_rotation_parser_accepts_args(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "portfolio-rotation",
                "--target-positions",
                "4",
                "--max-hold-days",
                "15",
                "--min-pre-opportunity",
                "0.45",
                "--min-model-alpha",
                "0.02",
                "--walk-forward",
                "--horizon-days",
                "20",
                "--model-name",
                "xgboost_model",
                "--eligible-universe-mode",
                "passed_or_trend",
                "--model-scope",
                "sector_specific",
                "--transaction-cost-bps",
                "10",
                "--slippage-bps",
                "5",
                "--cooldown-days",
                "3",
                "--no-reinvest-gains",
                "--max-new-entries-per-scan",
                "2",
                "--date-from",
                "2025-01-01",
                "--date-to",
                "2025-12-31",
            ]
        )
        self.assertEqual(args.command, "portfolio-rotation")
        self.assertEqual(args.target_positions, 4)
        self.assertEqual(args.max_hold_days, 15)
        self.assertEqual(args.min_pre_opportunity, 0.45)
        self.assertEqual(args.min_model_alpha, 0.02)
        self.assertTrue(args.walk_forward)
        self.assertEqual(args.horizon_days, 20)
        self.assertEqual(args.model_name, "xgboost_model")
        self.assertEqual(args.eligible_universe_mode, "passed_or_trend")
        self.assertEqual(args.model_scope, "sector_specific")
        self.assertEqual(args.transaction_cost_bps, 10)
        self.assertEqual(args.slippage_bps, 5)
        self.assertEqual(args.cooldown_days, 3)
        self.assertTrue(args.no_reinvest_gains)
        self.assertEqual(args.max_new_entries_per_scan, 2)
        self.assertEqual(args.date_from, "2025-01-01")
        self.assertEqual(args.date_to, "2025-12-31")

    def test_portfolio_rotation_walk_forward_uses_oos_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            reports_dir = root / "reports"
            reports_dir.mkdir(parents=True, exist_ok=True)

            class FakeDB:
                def __init__(self):
                    self.paths = AppPaths(
                        root_dir=root,
                        data_dir=root / "data",
                        duckdb_path=root / "data" / "market_data.duckdb",
                        sqlite_path=root / "data" / "ledger.sqlite",
                        reports_dir=reports_dir,
                        logs_dir=root / "logs",
                        config_path=root / "config.yaml",
                        env_path=root / ".env",
                        production_strategy_path=root / "production_strategy.json",
                    )

                def initialize(self): return None

                def load_scan_candidates(self, scan_date=None):
                    return pd.DataFrame(
                        [
                            {
                                "scan_date": "2026-05-01",
                                "ticker": "AAA",
                                "sector": "Technology",
                                "opportunity_score": 0.45,
                                "overlap_penalty": 0.0,
                                "model_predicted_alpha": 0.50,
                            },
                            {
                                "scan_date": "2026-05-01",
                                "ticker": "BBB",
                                "sector": "Technology",
                                "opportunity_score": 0.44,
                                "overlap_penalty": 0.0,
                                "model_predicted_alpha": 0.01,
                            },
                        ]
                    )

                def load_shortlist_model_runs(self, **kwargs):
                    return pd.DataFrame(
                        [
                            {
                                "generated_at": "2026-05-02T00:00:00+00:00",
                                "champion_model": "xgboost_model",
                                "eligible_universe_mode": "passed_or_trend",
                                "model_scope": "sector_specific",
                            }
                        ]
                    )

                def load_shortlist_model_predictions(self, **kwargs):
                    return pd.DataFrame(
                        [
                            {
                                "generated_at": "2026-05-02T00:00:00+00:00",
                                "snapshot_date": "2026-05-01",
                                "ticker": "AAA",
                                "sector": "Technology",
                                "predicted_alpha": 0.05,
                                "model_name": "xgboost_model",
                            },
                            {
                                "generated_at": "2026-05-02T00:00:00+00:00",
                                "snapshot_date": "2026-05-01",
                                "ticker": "BBB",
                                "sector": "Technology",
                                "predicted_alpha": 0.25,
                                "model_name": "xgboost_model",
                            },
                        ]
                    )

                def load_price_history(self, tickers):
                    return pd.DataFrame(
                        [
                            {"ticker": ticker, "date": "2026-05-01", "adj_close": price}
                            for ticker, price in (("AAA", 10.0), ("BBB", 20.0), ("SPY", 100.0))
                        ]
                    )

            report = PortfolioRotationService(FakeDB()).run(
                target_positions=1,
                max_hold_days=20,
                walk_forward=True,
                eligible_universe_mode="passed_or_trend",
                model_scope="sector_specific",
            )

            self.assertEqual(report.trades, 1)
            trades = pd.read_csv(reports_dir / "portfolio_rotation_walk_forward_trades.csv")
            self.assertEqual(trades.iloc[0]["ticker"], "BBB")
            report_text = (reports_dir / "portfolio_rotation_walk_forward.md").read_text(encoding="utf-8")
            self.assertIn("ranking_mode: walk_forward_oos", report_text)
            self.assertIn("ranking_model: xgboost_model", report_text)

    def test_portfolio_rotation_applies_transaction_costs_and_slippage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            reports_dir = root / "reports"
            reports_dir.mkdir(parents=True, exist_ok=True)

            class FakeDB:
                def __init__(self):
                    self.paths = AppPaths(
                        root_dir=root,
                        data_dir=root / "data",
                        duckdb_path=root / "data" / "market_data.duckdb",
                        sqlite_path=root / "data" / "ledger.sqlite",
                        reports_dir=reports_dir,
                        logs_dir=root / "logs",
                        config_path=root / "config.yaml",
                        env_path=root / ".env",
                        production_strategy_path=root / "production_strategy.json",
                    )

                def initialize(self): return None

                def load_scan_candidates(self, scan_date=None):
                    return pd.DataFrame(
                        [
                            {
                                "scan_date": "2026-05-01",
                                "ticker": "AAA",
                                "sector": "Technology",
                                "opportunity_score": 0.45,
                                "overlap_penalty": 0.0,
                                "model_predicted_alpha": 0.20,
                            }
                        ]
                    )

                def load_price_history(self, tickers):
                    return pd.DataFrame(
                        [
                            {"ticker": "AAA", "date": "2026-05-01", "adj_close": 10.0},
                            {"ticker": "AAA", "date": "2026-05-29", "adj_close": 11.0},
                            {"ticker": "SPY", "date": "2026-05-01", "adj_close": 100.0},
                            {"ticker": "SPY", "date": "2026-05-29", "adj_close": 100.0},
                        ]
                    )

            no_cost = PortfolioRotationService(FakeDB()).run(target_positions=1, max_hold_days=20)
            with_cost = PortfolioRotationService(FakeDB()).run(
                target_positions=1,
                max_hold_days=20,
                transaction_cost_bps=10,
                slippage_bps=5,
            )

            self.assertLess(with_cost.final_equity, no_cost.final_equity)
            trades = pd.read_csv(reports_dir / "portfolio_rotation_friction_trades.csv")
            self.assertGreater(float(trades.iloc[0]["entry_fee"]), 0.0)
            self.assertGreater(float(trades.iloc[0]["exit_fee"]), 0.0)
            report_text = (reports_dir / "portfolio_rotation_friction.md").read_text(encoding="utf-8")
            self.assertIn("transaction_cost_bps: 10.00", report_text)
            self.assertIn("slippage_bps: 5.00", report_text)

    def test_portfolio_rotation_can_disable_gain_reinvestment(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            reports_dir = root / "reports"
            reports_dir.mkdir(parents=True, exist_ok=True)

            class FakeDB:
                def __init__(self):
                    self.paths = AppPaths(
                        root_dir=root,
                        data_dir=root / "data",
                        duckdb_path=root / "data" / "market_data.duckdb",
                        sqlite_path=root / "data" / "ledger.sqlite",
                        reports_dir=reports_dir,
                        logs_dir=root / "logs",
                        config_path=root / "config.yaml",
                        env_path=root / ".env",
                        production_strategy_path=root / "production_strategy.json",
                    )

                def initialize(self): return None

                def load_scan_candidates(self, scan_date=None):
                    rows = []
                    for day in ("2026-05-01", "2026-05-29", "2026-06-26"):
                        rows.append(
                            {
                                "scan_date": day,
                                "ticker": {"2026-05-01": "AAA", "2026-05-29": "BBB", "2026-06-26": "CCC"}[day],
                                "sector": "Technology",
                                "opportunity_score": 0.45,
                                "overlap_penalty": 0.0,
                                "model_predicted_alpha": 0.20,
                            }
                        )
                    return pd.DataFrame(rows)

                def load_price_history(self, tickers):
                    return pd.DataFrame(
                        [
                            {"ticker": "AAA", "date": "2026-05-01", "adj_close": 10.0},
                            {"ticker": "AAA", "date": "2026-05-29", "adj_close": 20.0},
                            {"ticker": "BBB", "date": "2026-05-29", "adj_close": 10.0},
                            {"ticker": "BBB", "date": "2026-06-26", "adj_close": 20.0},
                            {"ticker": "SPY", "date": "2026-05-01", "adj_close": 100.0},
                            {"ticker": "SPY", "date": "2026-05-29", "adj_close": 100.0},
                            {"ticker": "SPY", "date": "2026-06-26", "adj_close": 100.0},
                        ]
                    )

            compounding = PortfolioRotationService(FakeDB()).run(target_positions=1, max_hold_days=20)
            fixed_slot = PortfolioRotationService(FakeDB()).run(
                target_positions=1,
                max_hold_days=20,
                reinvest_gains=False,
            )

            self.assertLess(fixed_slot.final_equity, compounding.final_equity)
            report_text = (reports_dir / "portfolio_rotation_capital.md").read_text(encoding="utf-8")
            self.assertIn("reinvest_gains: false", report_text)


if __name__ == "__main__":
    unittest.main()
