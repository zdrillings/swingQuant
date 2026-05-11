from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from src.settings import AppPaths
from src.sleeves.service import SleeveResearchService


class SleeveResearchServiceTests(unittest.TestCase):
    def test_sleeve_research_writes_rank_based_report(self) -> None:
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

            dates = pd.bdate_range("2026-01-01", periods=8)
            price_rows = []
            for index, day in enumerate(dates):
                price_rows.extend(
                    [
                        {
                            "ticker": "AAA",
                            "date": day.date(),
                            "open": 100.0 + index,
                            "high": 101.0 + index,
                            "low": 99.0 + index,
                            "close": 100.5 + index,
                            "volume": 1_000_000,
                            "adj_close": 100.5 + index,
                        },
                        {
                            "ticker": "BBB",
                            "date": day.date(),
                            "open": 90.0 + index * 0.2,
                            "high": 91.0 + index * 0.2,
                            "low": 89.0 + index * 0.2,
                            "close": 90.1 + index * 0.2,
                            "volume": 1_000_000,
                            "adj_close": 90.1 + index * 0.2,
                        },
                        {
                            "ticker": "SPY",
                            "date": day.date(),
                            "open": 300.0 + index,
                            "high": 301.0 + index,
                            "low": 299.0 + index,
                            "close": 300.5 + index,
                            "volume": 2_000_000,
                            "adj_close": 300.5 + index,
                        },
                        {
                            "ticker": "XLB",
                            "date": day.date(),
                            "open": 80.0 + index * 0.5,
                            "high": 81.0 + index * 0.5,
                            "low": 79.0 + index * 0.5,
                            "close": 80.4 + index * 0.5,
                            "volume": 2_000_000,
                            "adj_close": 80.4 + index * 0.5,
                        },
                    ]
                )
            price_history = pd.DataFrame(price_rows)
            analysis_rows = []
            for index, day in enumerate(dates):
                analysis_rows.extend(
                    [
                        {
                            "ticker": "AAA",
                            "date": day,
                            "sector": "Materials",
                            "regime_green": True,
                            "sma_50_dist": 0.10,
                            "sma_200_dist": 0.12,
                            "roc_63": 0.15,
                            "relative_strength_index_vs_spy": 90.0,
                            "rsi_14": 48.0,
                            "vol_alpha": 1.2,
                            "md_volume_30d": 1_000_000,
                            "sector_pct_above_50": 1.0,
                            "sector_pct_above_200": 1.0,
                            "sector_median_roc_63": 0.10,
                            "open": 100.0 + index,
                            "close": 100.5 + index,
                        },
                        {
                            "ticker": "BBB",
                            "date": day,
                            "sector": "Materials",
                            "regime_green": True,
                            "sma_50_dist": 0.04,
                            "sma_200_dist": 0.05,
                            "roc_63": 0.03,
                            "relative_strength_index_vs_spy": 72.0,
                            "rsi_14": 60.0,
                            "vol_alpha": 0.9,
                            "md_volume_30d": 1_000_000,
                            "sector_pct_above_50": 1.0,
                            "sector_pct_above_200": 1.0,
                            "sector_median_roc_63": 0.10,
                            "open": 90.0 + index * 0.2,
                            "close": 90.1 + index * 0.2,
                        },
                    ]
                )
            analysis_frame = pd.DataFrame(analysis_rows)

            class FakeDB:
                def __init__(self, paths, price_history):
                    self.paths = paths
                    self._price_history = price_history

                def initialize(self): return None
                def list_research_universe(self, limit=250):
                    return [
                        {"ticker": "AAA", "sector": "Materials", "md_volume_30d": 1_000_000},
                        {"ticker": "BBB", "sector": "Materials", "md_volume_30d": 1_000_000},
                    ]
                def load_price_history(self, tickers): return self._price_history.copy()
                def load_earnings_calendar(self, tickers): return pd.DataFrame()

            service = SleeveResearchService(FakeDB(paths, price_history))
            sleeve_config = {
                "sleeve_research": {
                    "sectors": ["Materials"],
                    "top_n_values": [1],
                    "horizon_days_values": [5],
                    "support": {
                        "min_trade_count_for_rank": 1,
                        "target_trade_count_for_score": 2,
                        "min_trade_count_for_live_section": 1,
                        "min_trade_count_for_top_sections": 1,
                        "min_distinct_tickers_for_rank": 1,
                        "target_distinct_tickers_for_score": 1,
                        "min_distinct_tickers_for_top_sections": 1,
                        "target_live_match_count_for_score": 1,
                        "min_live_match_count_for_top_sections": 1,
                    },
                    "breadth_thresholds": {
                        "sector_pct_above_50_min": [0.5],
                        "sector_pct_above_200_min": [0.5],
                        "sector_median_roc_63_min": [0.0],
                    },
                    "base_filters": {
                        "relative_strength_index_vs_spy_min": 70.0,
                        "roc_63_min": 0.0,
                        "vol_alpha_min": 0.8,
                        "rsi_14_min": 40.0,
                        "rsi_14_max": 65.0,
                    },
                    "rank_weights": {
                        "relative_strength_index_vs_spy": 0.40,
                        "roc_63": 0.25,
                        "rsi_14_pullback": 0.20,
                        "vol_alpha": 0.15,
                    },
                },
                "backtest_costs": {
                    "slippage_bps_per_side": 5,
                    "commission_bps_per_side": 0,
                },
            }

            with patch("src.sleeves.service.load_feature_config", return_value=sleeve_config), \
                 patch("src.sleeves.service.build_analysis_frame", return_value=(analysis_frame, [])):
                report = service.run(top=5)

            self.assertEqual(report.configs_evaluated, 1)
            report_text = (paths.reports_dir / "sleeve_research.md").read_text(encoding="utf-8")
            self.assertIn("# Sleeve Research", report_text)
            self.assertIn("## Top Ranked Sleeve Configurations", report_text)
            self.assertIn("## Best Live Configurations With Enough Sample", report_text)
            self.assertIn("## Best Supported Configuration Per Sector", report_text)
            self.assertIn("distinct_tickers_traded: 1", report_text)
            self.assertIn("Materials | top_n=1 | horizon=5d", report_text)
            self.assertIn("live_match_tickers: AAA", report_text)

    def test_sleeve_research_can_render_walk_forward_section(self) -> None:
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

            dates = pd.bdate_range("2026-01-01", periods=8)
            price_history = pd.DataFrame(
                [
                    {
                        "ticker": ticker,
                        "date": day.date(),
                        "open": 100.0 + index,
                        "high": 101.0 + index,
                        "low": 99.0 + index,
                        "close": 100.5 + index,
                        "volume": 1_000_000,
                        "adj_close": 100.5 + index,
                    }
                    for index, day in enumerate(dates)
                    for ticker in ("AAA", "SPY", "XLB")
                ]
            )
            analysis_frame = pd.DataFrame(
                [
                    {
                        "ticker": "AAA",
                        "date": day,
                        "sector": "Materials",
                        "regime_green": True,
                        "sma_50_dist": 0.10,
                        "sma_200_dist": 0.12,
                        "roc_63": 0.15,
                        "relative_strength_index_vs_spy": 90.0,
                        "rsi_14": 48.0,
                        "vol_alpha": 1.2,
                        "md_volume_30d": 1_000_000,
                        "sector_pct_above_50": 1.0,
                        "sector_pct_above_200": 1.0,
                        "sector_median_roc_63": 0.10,
                        "open": 100.0 + index,
                        "close": 100.5 + index,
                    }
                    for index, day in enumerate(dates)
                ]
            )

            class FakeDB:
                def __init__(self, paths, price_history):
                    self.paths = paths
                    self._price_history = price_history

                def initialize(self): return None
                def list_research_universe(self, limit=250):
                    return [{"ticker": "AAA", "sector": "Materials", "md_volume_30d": 1_000_000}]
                def load_price_history(self, tickers): return self._price_history.copy()
                def load_earnings_calendar(self, tickers): return pd.DataFrame()

            service = SleeveResearchService(FakeDB(paths, price_history))
            sleeve_config = {
                "sleeve_research": {
                    "sectors": ["Materials"],
                    "top_n_values": [1],
                    "horizon_days_values": [5],
                    "support": {
                        "min_trade_count_for_rank": 1,
                        "target_trade_count_for_score": 2,
                        "min_trade_count_for_live_section": 1,
                        "min_trade_count_for_top_sections": 1,
                        "min_distinct_tickers_for_rank": 1,
                        "target_distinct_tickers_for_score": 1,
                        "min_distinct_tickers_for_top_sections": 1,
                        "target_live_match_count_for_score": 1,
                        "min_live_match_count_for_top_sections": 1,
                    },
                    "breadth_thresholds": {
                        "sector_pct_above_50_min": [0.5],
                        "sector_pct_above_200_min": [0.5],
                        "sector_median_roc_63_min": [0.0],
                    },
                    "base_filters": {
                        "relative_strength_index_vs_spy_min": 70.0,
                        "roc_63_min": 0.0,
                        "vol_alpha_min": 0.8,
                        "rsi_14_min": 40.0,
                        "rsi_14_max": 65.0,
                    },
                    "rank_weights": {
                        "relative_strength_index_vs_spy": 0.40,
                        "roc_63": 0.25,
                        "rsi_14_pullback": 0.20,
                        "vol_alpha": 0.15,
                    },
                },
                "backtest_costs": {
                    "slippage_bps_per_side": 5,
                    "commission_bps_per_side": 0,
                },
            }

            with patch("src.sleeves.service.load_feature_config", return_value=sleeve_config), \
                 patch("src.sleeves.service.build_analysis_frame", return_value=(analysis_frame, [])), \
                 patch.object(
                     SleeveResearchService,
                     "_build_walk_forward_stability",
                     return_value=pd.DataFrame(
                         [
                             {
                                 "sector": "Materials",
                                 "top_n": 1,
                                 "horizon_days": 5,
                                 "sector_pct_above_50_min": 0.5,
                                 "sector_pct_above_200_min": 0.5,
                                 "sector_median_roc_63_min": 0.0,
                                 "wf_window_count": 5,
                                 "wf_median_expectancy": 0.01,
                                 "wf_worst_expectancy": 0.001,
                                 "wf_positive_window_ratio": 0.8,
                                 "wf_positive_alpha_window_ratio": 0.8,
                                 "wf_median_alpha_vs_spy": 0.02,
                                 "wf_worst_mdd": 0.2,
                                 "wf_trade_count_min": 2,
                                 "wf_stability_score": 0.55,
                             }
                         ]
                     ),
                 ):
                report = service.run(top=5, walk_forward=True, walk_forward_shortlist=5, walk_forward_windows=5)

            self.assertEqual(report.configs_evaluated, 1)
            report_text = (paths.reports_dir / "sleeve_research.md").read_text(encoding="utf-8")
            self.assertIn("## Best Walk-Forward Stable Sleeve Configurations", report_text)
            self.assertIn("## Best Stable Configuration Per Sector", report_text)
            self.assertIn("wf_stability_score: 0.550000", report_text)

    def test_practical_scoring_penalizes_low_trade_support(self) -> None:
        service = SleeveResearchService(db_manager=None)  # type: ignore[arg-type]
        frame = pd.DataFrame(
            [
                {
                    "expectancy": 0.01,
                    "profit_factor": 1.5,
                    "alpha_vs_spy": 0.01,
                    "alpha_vs_sector": 0.01,
                    "mdd": 0.2,
                    "trade_count": 20,
                    "distinct_tickers_traded": 1,
                    "max_ticker_trade_share": 1.0,
                    "live_match_count": 1,
                },
                {
                    "expectancy": 0.01,
                    "profit_factor": 1.5,
                    "alpha_vs_spy": 0.01,
                    "alpha_vs_sector": 0.01,
                    "mdd": 0.2,
                    "trade_count": 120,
                    "distinct_tickers_traded": 5,
                    "max_ticker_trade_share": 0.2,
                    "live_match_count": 1,
                },
            ]
        )
        scored = service._apply_practical_scoring(
            frame,
            {
                "min_trade_count_for_rank": 75,
                "target_trade_count_for_score": 100,
                "min_trade_count_for_live_section": 75,
                "min_trade_count_for_top_sections": 75,
                "min_distinct_tickers_for_rank": 3,
                "target_distinct_tickers_for_score": 5,
                "min_distinct_tickers_for_top_sections": 3,
                "target_live_match_count_for_score": 2,
                "min_live_match_count_for_top_sections": 1,
            },
        )
        self.assertLess(float(scored.iloc[0]["practical_score"]), float(scored.iloc[1]["practical_score"]))

    def test_rank_candidates_for_day_respects_rsi_band_filter(self) -> None:
        service = SleeveResearchService(db_manager=None)  # type: ignore[arg-type]
        frame = pd.DataFrame(
            [
                {
                    "ticker": "AAA",
                    "sector_pct_above_50": 1.0,
                    "sector_pct_above_200": 1.0,
                    "sector_median_roc_63": 0.10,
                    "regime_green": True,
                    "sma_50_dist": 0.10,
                    "sma_200_dist": 0.10,
                    "relative_strength_index_vs_spy": 80.0,
                    "roc_63": 0.10,
                    "vol_alpha": 1.1,
                    "rsi_14": 35.0,
                    "md_volume_30d": 1_000_000,
                },
                {
                    "ticker": "BBB",
                    "sector_pct_above_50": 1.0,
                    "sector_pct_above_200": 1.0,
                    "sector_median_roc_63": 0.10,
                    "regime_green": True,
                    "sma_50_dist": 0.10,
                    "sma_200_dist": 0.10,
                    "relative_strength_index_vs_spy": 82.0,
                    "roc_63": 0.12,
                    "vol_alpha": 1.2,
                    "rsi_14": 50.0,
                    "md_volume_30d": 1_000_000,
                },
            ]
        )
        ranked = service._rank_candidates_for_day(
            frame,
            breadth_filters={
                "sector_pct_above_50_min": 0.5,
                "sector_pct_above_200_min": 0.5,
                "sector_median_roc_63_min": 0.0,
            },
            base_filters={
                "relative_strength_index_vs_spy_min": 75.0,
                "roc_63_min": 0.05,
                "vol_alpha_min": 1.0,
                "rsi_14_min": 40.0,
                "rsi_14_max": 65.0,
            },
            rank_weights={
                "relative_strength_index_vs_spy": 0.40,
                "roc_63": 0.25,
                "rsi_14_pullback": 0.20,
                "vol_alpha": 0.15,
            },
        )
        self.assertEqual(ranked["ticker"].tolist(), ["BBB"])
