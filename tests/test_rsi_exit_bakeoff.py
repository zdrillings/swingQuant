from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from src.cli import build_parser
from src.research.rsi_exit_bakeoff_service import RsiExitBakeoffService
from src.settings import AppPaths
from src.utils.strategy import ExitRules, ProductionStrategy


class RsiExitBakeoffServiceTests(unittest.TestCase):
    def test_rsi_exit_bakeoff_writes_report(self) -> None:
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
                                "strategy_slot": "energy",
                                "strategy_sector": "Energy",
                                "sector": "Energy",
                                "selected": 1,
                                "adj_close": 100.0,
                                "details_json": '{"feature_snapshot":{"atr_14":2.0}}',
                            }
                        ]
                    )

                def list_universe_rows(self, active_only=False):
                    return [
                        {"ticker": "AAA", "sector": "Energy"},
                    ]

                def load_price_history(self, tickers):
                    dates = pd.bdate_range("2026-05-01", periods=6)
                    closes = {
                        "AAA": [100.0, 101.0, 102.0, 104.0, 105.0, 106.0],
                        "SPY": [100.0, 100.2, 100.4, 100.6, 100.8, 101.0],
                        "XLE": [50.0, 50.2, 50.4, 50.6, 50.8, 51.0],
                        "QQQ": [200.0, 200.5, 201.0, 201.5, 202.0, 202.5],
                    }
                    rows = []
                    for ticker, values in closes.items():
                        for day, close in zip(dates, values):
                            rows.append(
                                {
                                    "ticker": ticker,
                                    "date": day,
                                    "open": close - 0.5,
                                    "high": close + 0.5,
                                    "low": close - 0.5,
                                    "close": close,
                                    "volume": 1_000_000,
                                    "adj_close": close,
                                }
                            )
                    frame = pd.DataFrame(rows)
                    return frame[frame["ticker"].isin(tickers)].copy()

            strategy = ProductionStrategy(
                strategy_id=11,
                promoted_at="2026-05-01T17:00:00",
                indicators={"relative_strength_index_vs_spy_min": 75.0, "signal_score_min": 32.0},
                exit_rules=ExitRules(trailing_stop_pct=0.20, profit_target_pct=0.50, time_limit_days=3),
                slot="energy",
                sector="Energy",
            )

            analysis_frame = pd.DataFrame(
                [
                    {
                        "ticker": "AAA",
                        "date": day,
                        "regime_green": True,
                        "days_to_next_earnings": float("nan"),
                        "atr_14": 2.0,
                        "relative_strength_index_vs_spy": 80.0,
                        "signal_score": 40.0,
                    }
                    for day in pd.bdate_range("2026-05-01", periods=6)
                ]
            )

            def fake_rsi(price_history, ticker, current_price, as_of):
                return 95.0 if str(as_of) == "2026-05-04" else 20.0

            with patch("src.research.rsi_exit_bakeoff_service.load_active_strategies", return_value={"energy": strategy}), \
                 patch("src.research.rsi_exit_bakeoff_service.build_analysis_frame", return_value=(analysis_frame, {})), \
                 patch("src.research.rsi_exit_bakeoff_service.latest_rsi_2_with_intraday", side_effect=fake_rsi):
                report = RsiExitBakeoffService(FakeDB()).run(recent_scan_dates=60, benchmark="sector")

            self.assertTrue(report.output_path.endswith("rsi_exit_bakeoff.md"))
            self.assertEqual(report.selected_rows, 1)
            self.assertEqual(report.mature_rows, 1)
            report_text = (reports_dir / "rsi_exit_bakeoff.md").read_text(encoding="utf-8")
            self.assertIn("# RSI_2 Exit Bakeoff", report_text)
            self.assertIn("## Variant Summary", report_text)
            self.assertIn("### Current Rule", report_text)
            self.assertIn("### No RSI_2 Exit", report_text)
            self.assertIn("## Early RSI_2 Sells", report_text)
            self.assertIn("AAA (energy, 2026-05-01)", report_text)

    def test_rsi_exit_bakeoff_parser_accepts_args(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "rsi-exit-bakeoff",
                "--recent-scan-dates",
                "45",
                "--benchmark",
                "spy",
            ]
        )
        self.assertEqual(args.command, "rsi-exit-bakeoff")
        self.assertEqual(args.recent_scan_dates, 45)
        self.assertEqual(args.benchmark, "spy")
