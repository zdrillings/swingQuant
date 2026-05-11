from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from src.research.alpha_service import AlphaResearchService
from src.settings import AppPaths


class FakeModel:
    def predict(self, frame: pd.DataFrame):
        return [0.02 + index * 0.001 for index in range(len(frame.index))]


class AlphaResearchServiceTests(unittest.TestCase):
    def test_alpha_research_writes_report(self) -> None:
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
            paths.reports_dir.mkdir(parents=True, exist_ok=True)
            paths.logs_dir.mkdir(parents=True, exist_ok=True)

            dates = pd.bdate_range("2025-01-01", periods=40)
            price_rows = []
            for index, day in enumerate(dates):
                for ticker, base, slope in (
                    ("AAA", 100.0, 0.8),
                    ("BBB", 90.0, 0.5),
                    ("SPY", 300.0, 0.3),
                    ("XLI", 110.0, 0.25),
                ):
                    price_rows.append(
                        {
                            "ticker": ticker,
                            "date": day.date(),
                            "open": base + index * slope,
                            "high": base + index * slope + 1.0,
                            "low": base + index * slope - 1.0,
                            "close": base + index * slope + 0.5,
                            "volume": 1_000_000,
                            "adj_close": base + index * slope + 0.5,
                        }
                    )
            price_history = pd.DataFrame(price_rows)
            analysis_frame = pd.DataFrame(
                [
                    {
                        "ticker": ticker,
                        "date": day,
                        "sector": "Industrials",
                        "adj_close": 100.0 + index * 0.8 if ticker == "AAA" else 90.0 + index * 0.5,
                        "roc_63": 0.10 + (0.01 if ticker == "AAA" else 0.0),
                        "relative_strength_index_vs_spy": 80.0 + (5.0 if ticker == "AAA" else 0.0),
                        "rsi_14": 50.0,
                        "vol_alpha": 1.2,
                        "sma_50_dist": 0.1,
                        "sma_200_dist": 0.1,
                        "sector_pct_above_50": 0.8,
                        "sector_pct_above_200": 0.7,
                        "sector_median_roc_63": 0.08,
                        "regime_green": True,
                        "spy_regime_green": True,
                        "qqq_regime_green": True,
                        "md_volume_30d": 2_000_000,
                    }
                    for index, day in enumerate(dates)
                    for ticker in ("AAA", "BBB")
                ]
            )

            class FakeDB:
                def __init__(self, paths, price_history):
                    self.paths = paths
                    self._price_history = price_history

                def initialize(self): return None
                def list_research_universe(self, limit=250):
                    return [
                        {"ticker": "AAA", "sector": "Industrials", "md_volume_30d": 2_000_000},
                        {"ticker": "BBB", "sector": "Industrials", "md_volume_30d": 2_000_000},
                    ]
                def load_price_history(self, tickers): return self._price_history.copy()
                def load_earnings_calendar(self, tickers): return pd.DataFrame()

            service = AlphaResearchService(FakeDB(paths, price_history))
            importance = pd.DataFrame(
                [
                    {"feature": "relative_strength_index_vs_spy", "gain": 1.0},
                    {"feature": "roc_63", "gain": 0.5},
                ]
            )
            with patch("src.research.alpha_service.build_analysis_frame", return_value=(analysis_frame, ["roc_63", "relative_strength_index_vs_spy", "rsi_14", "vol_alpha"])), \
                 patch.object(service, "_fit_regressor", return_value=(FakeModel(), importance)):
                report = service.run(top=2, sector="Industrials", horizon_days=5, benchmark="sector")

            self.assertGreater(report.train_rows, 0)
            self.assertGreater(report.validation_rows, 0)
            report_text = (paths.reports_dir / "alpha_research.md").read_text(encoding="utf-8")
            self.assertIn("# Alpha Research", report_text)
            self.assertIn("## Top Predicted Live Excess Return Candidates", report_text)
            self.assertIn("### BBB", report_text)
            self.assertIn("chart: https://www.tradingview.com/chart/?symbol=BBB", report_text)
            self.assertIn("## Feature Importance", report_text)
