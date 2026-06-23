from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from src.utils.feature_engineering import apply_feature_definitions
from src.utils.regime import regime_etf_for_sector
from src.utils.signal_engine import build_analysis_frame, filter_signal_candidates
from src.utils.strategy import (
    ExitRules,
    clear_strategy_caches,
    indicator_score,
    profit_target_price,
    rsi_2_exit_triggered,
    stop_risk_per_share,
    trailing_stop_price,
)


class StrategyHelperTests(unittest.TestCase):
    def test_rsi_2_exit_triggered_supports_slot_specific_profit_gates(self) -> None:
        clear_strategy_caches()
        with patch(
            "src.utils.strategy.load_feature_config",
            return_value={
                "monitor_policy": {
                    "rsi_2_exit": {
                        "threshold": 90,
                        "min_unrealized_pct": 0.05,
                        "min_days_in_trade": 0,
                        "slot_overrides": {
                            "energy": {"min_unrealized_pct": 0.05},
                            "industrials": {"min_unrealized_pct": 0.05},
                        },
                    }
                }
            },
        ):
            clear_strategy_caches()
            self.assertFalse(
                rsi_2_exit_triggered(
                    rsi_2=95.0,
                    unrealized_pct=0.02,
                    days_in_trade=2,
                    strategy_slot="energy",
                )
            )
            self.assertTrue(
                rsi_2_exit_triggered(
                    rsi_2=95.0,
                    unrealized_pct=0.06,
                    days_in_trade=2,
                    strategy_slot="energy",
                )
            )
            self.assertFalse(
                rsi_2_exit_triggered(
                    rsi_2=95.0,
                    unrealized_pct=0.02,
                    days_in_trade=2,
                    strategy_slot="materials",
                )
            )
            self.assertTrue(
                rsi_2_exit_triggered(
                    rsi_2=95.0,
                    unrealized_pct=0.06,
                    days_in_trade=2,
                    strategy_slot="materials",
                )
            )
        clear_strategy_caches()

    def test_filter_signal_candidates_uses_confluence_scoring_and_rs_hard_filter(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "ticker": "AAA",
                    "rsi_14": 45.0,
                    "vol_alpha": 1.8,
                    "sma_200_dist": 0.18,
                    "roc_63": 0.18,
                    "relative_strength_index_vs_spy": 85.0,
                },
                {
                    "ticker": "BBB",
                    "rsi_14": 45.0,
                    "vol_alpha": 1.8,
                    "sma_200_dist": 0.18,
                    "roc_63": 0.18,
                    "relative_strength_index_vs_spy": 70.0,
                },
                {
                    "ticker": "CCC",
                    "rsi_14": 60.0,
                    "vol_alpha": 1.0,
                    "sma_200_dist": 0.02,
                    "roc_63": 0.01,
                    "relative_strength_index_vs_spy": 90.0,
                },
            ]
        )

        candidates = filter_signal_candidates(
            frame,
            {
                "rsi_14_max": 35.0,
                "vol_alpha_min": 1.4,
                "sma_200_dist_min": 0.10,
                "roc_63_min": 0.10,
                "relative_strength_index_vs_spy_min": 80.0,
                "signal_score_min": 30.0,
            },
        )

        self.assertEqual(candidates["ticker"].tolist(), ["AAA"])
        self.assertGreaterEqual(float(candidates.iloc[0]["signal_score"]), 30.0)

    def test_filter_signal_candidates_supports_breakout_model_family(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "ticker": "AAA",
                    "sma_50_dist": 0.04,
                    "ma_alignment_50_200": 0.08,
                    "ma_slope_50_20": 0.06,
                    "ma_slope_200_20": 0.02,
                    "breakout_above_20d_high": 1.0,
                    "distance_above_20d_high": 0.01,
                    "relative_strength_index_vs_spy": 88.0,
                    "roc_63": 0.16,
                    "rsi_14": 58.0,
                    "sma_200_dist": 0.22,
                    "base_range_pct_20": 0.11,
                    "base_atr_contraction_20": 0.92,
                    "base_volume_dryup_ratio_20": 0.88,
                    "breakout_volume_ratio_50": 1.9,
                },
                {
                    "ticker": "BBB",
                    "sma_50_dist": 0.04,
                    "ma_alignment_50_200": 0.08,
                    "ma_slope_50_20": 0.06,
                    "ma_slope_200_20": 0.02,
                    "breakout_above_20d_high": 0.0,
                    "distance_above_20d_high": -0.01,
                    "relative_strength_index_vs_spy": 88.0,
                    "roc_63": 0.16,
                    "rsi_14": 58.0,
                    "sma_200_dist": 0.22,
                    "base_range_pct_20": 0.11,
                    "base_atr_contraction_20": 0.92,
                    "base_volume_dryup_ratio_20": 0.88,
                    "breakout_volume_ratio_50": 1.9,
                },
                {
                    "ticker": "CCC",
                    "sma_50_dist": 0.04,
                    "ma_alignment_50_200": 0.08,
                    "ma_slope_50_20": 0.06,
                    "ma_slope_200_20": 0.02,
                    "breakout_above_20d_high": 1.0,
                    "distance_above_20d_high": 0.01,
                    "relative_strength_index_vs_spy": 72.0,
                    "roc_63": 0.16,
                    "rsi_14": 58.0,
                    "sma_200_dist": 0.22,
                    "base_range_pct_20": 0.11,
                    "base_atr_contraction_20": 0.92,
                    "base_volume_dryup_ratio_20": 0.88,
                    "breakout_volume_ratio_50": 1.9,
                },
                {
                    "ticker": "DDD",
                    "sma_50_dist": 0.04,
                    "ma_alignment_50_200": 0.08,
                    "ma_slope_50_20": 0.06,
                    "ma_slope_200_20": 0.02,
                    "breakout_above_20d_high": 1.0,
                    "distance_above_20d_high": 0.05,
                    "relative_strength_index_vs_spy": 88.0,
                    "roc_63": 0.16,
                    "rsi_14": 58.0,
                    "sma_200_dist": 0.22,
                    "base_range_pct_20": 0.11,
                    "base_atr_contraction_20": 0.92,
                    "base_volume_dryup_ratio_20": 0.88,
                    "breakout_volume_ratio_50": 1.9,
                },
            ]
        )

        candidates = filter_signal_candidates(
            frame,
            {
                "sma_50_dist_min": 0.0,
                "ma_alignment_50_200_min": 0.0,
                "ma_slope_50_20_min": 0.0,
                "ma_slope_200_20_min": 0.0,
                "breakout_above_20d_high_min": 1.0,
                "distance_above_20d_high_max": 0.02,
                "relative_strength_index_vs_spy_min": 80.0,
                "roc_63_min": 0.10,
                "rsi_14_min": 50.0,
                "sma_200_dist_max": 0.30,
                "base_range_pct_20_max": 0.15,
                "base_atr_contraction_20_max": 1.0,
                "base_volume_dryup_ratio_20_max": 1.0,
                "breakout_volume_ratio_50_min": 1.5,
                "signal_score_min": 34.0,
            },
        )

        self.assertEqual(candidates["ticker"].tolist(), ["AAA"])
        self.assertGreaterEqual(float(candidates.iloc[0]["signal_score"]), 34.0)

    def test_filter_signal_candidates_supports_earnings_timing_filters(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "ticker": "AAA",
                    "rsi_14": 55.0,
                    "vol_alpha": 1.2,
                    "sma_200_dist": 0.18,
                    "roc_63": 0.12,
                    "relative_strength_index_vs_spy": 80.0,
                    "days_to_next_earnings": 7.0,
                    "days_since_last_earnings": 3.0,
                },
                {
                    "ticker": "BBB",
                    "rsi_14": 55.0,
                    "vol_alpha": 1.2,
                    "sma_200_dist": 0.18,
                    "roc_63": 0.12,
                    "relative_strength_index_vs_spy": 80.0,
                    "days_to_next_earnings": 1.0,
                    "days_since_last_earnings": 3.0,
                },
                {
                    "ticker": "CCC",
                    "rsi_14": 55.0,
                    "vol_alpha": 1.2,
                    "sma_200_dist": 0.18,
                    "roc_63": 0.12,
                    "relative_strength_index_vs_spy": 80.0,
                    "days_to_next_earnings": 7.0,
                    "days_since_last_earnings": 0.0,
                },
            ]
        )

        candidates = filter_signal_candidates(
            frame,
            {
                "rsi_14_max": 60.0,
                "vol_alpha_min": 1.0,
                "sma_200_dist_min": 0.10,
                "roc_63_min": 0.10,
                "relative_strength_index_vs_spy_min": 75.0,
                "days_to_next_earnings_min": 5.0,
                "days_since_last_earnings_min": 2.0,
                "signal_score_min": 30.0,
            },
        )

        self.assertEqual(candidates["ticker"].tolist(), ["AAA"])

    def test_relative_strength_feature_ranks_tickers_vs_spy(self) -> None:
        dates = pd.bdate_range("2026-01-01", periods=70)
        rows = []
        for index, day in enumerate(dates):
            rows.extend(
                [
                    {
                        "ticker": "SPY",
                        "date": day.date(),
                        "open": 100 + index,
                        "high": 100 + index,
                        "low": 100 + index,
                        "close": 100 + index,
                        "volume": 1_000_000,
                        "adj_close": 100 + index,
                    },
                    {
                        "ticker": "AAA",
                        "date": day.date(),
                        "open": 100 + index * 2,
                        "high": 100 + index * 2,
                        "low": 100 + index * 2,
                        "close": 100 + index * 2,
                        "volume": 1_000_000,
                        "adj_close": 100 + index * 2,
                    },
                    {
                        "ticker": "BBB",
                        "date": day.date(),
                        "open": 100 + index * 0.5,
                        "high": 100 + index * 0.5,
                        "low": 100 + index * 0.5,
                        "close": 100 + index * 0.5,
                        "volume": 1_000_000,
                        "adj_close": 100 + index * 0.5,
                    },
                ]
            )
        frame, _ = apply_feature_definitions(
            pd.DataFrame(rows),
            {
                "features": {
                    "momentum": [
                        {
                            "name": "relative_strength_index_vs_spy",
                            "ticker": "SPY",
                            "type": "relative_strength_percentile",
                            "params": {"window": 63},
                        }
                    ]
                }
            },
        )
        latest = frame.sort_values("date").groupby("ticker").tail(1).set_index("ticker")

        self.assertGreater(
            float(latest.loc["AAA", "relative_strength_index_vs_spy"]),
            float(latest.loc["BBB", "relative_strength_index_vs_spy"]),
        )
        self.assertTrue(pd.isna(latest.loc["SPY", "relative_strength_index_vs_spy"]))

    def test_relative_strength_feature_supports_qqq_and_xlk_references(self) -> None:
        dates = pd.bdate_range("2026-01-01", periods=70)
        rows = []
        for index, day in enumerate(dates):
            rows.extend(
                [
                    {
                        "ticker": "QQQ",
                        "date": day.date(),
                        "open": 100 + index,
                        "high": 100 + index,
                        "low": 100 + index,
                        "close": 100 + index,
                        "volume": 1_000_000,
                        "adj_close": 100 + index,
                    },
                    {
                        "ticker": "XLK",
                        "date": day.date(),
                        "open": 100 + index * 0.9,
                        "high": 100 + index * 0.9,
                        "low": 100 + index * 0.9,
                        "close": 100 + index * 0.9,
                        "volume": 1_000_000,
                        "adj_close": 100 + index * 0.9,
                    },
                    {
                        "ticker": "AAA",
                        "date": day.date(),
                        "open": 100 + index * 2,
                        "high": 100 + index * 2,
                        "low": 100 + index * 2,
                        "close": 100 + index * 2,
                        "volume": 1_000_000,
                        "adj_close": 100 + index * 2,
                    },
                    {
                        "ticker": "BBB",
                        "date": day.date(),
                        "open": 100 + index * 0.4,
                        "high": 100 + index * 0.4,
                        "low": 100 + index * 0.4,
                        "close": 100 + index * 0.4,
                        "volume": 1_000_000,
                        "adj_close": 100 + index * 0.4,
                    },
                ]
            )
        frame, _ = apply_feature_definitions(
            pd.DataFrame(rows),
            {
                "features": {
                    "momentum": [
                        {
                            "name": "relative_strength_index_vs_qqq",
                            "ticker": "QQQ",
                            "type": "relative_strength_percentile",
                            "params": {"window": 63},
                        },
                        {
                            "name": "relative_strength_index_vs_xlk",
                            "ticker": "XLK",
                            "type": "relative_strength_percentile",
                            "params": {"window": 63},
                        },
                    ]
                }
            },
        )
        latest = frame.sort_values("date").groupby("ticker").tail(1).set_index("ticker")

        self.assertGreater(
            float(latest.loc["AAA", "relative_strength_index_vs_qqq"]),
            float(latest.loc["BBB", "relative_strength_index_vs_qqq"]),
        )
        self.assertGreater(
            float(latest.loc["AAA", "relative_strength_index_vs_xlk"]),
            float(latest.loc["BBB", "relative_strength_index_vs_xlk"]),
        )
        self.assertTrue(pd.isna(latest.loc["QQQ", "relative_strength_index_vs_qqq"]))
        self.assertTrue(pd.isna(latest.loc["XLK", "relative_strength_index_vs_xlk"]))

    def test_contextual_relative_strength_feature_supports_subindustry_benchmarks(self) -> None:
        dates = pd.bdate_range("2026-01-01", periods=70)
        rows = []
        for index, day in enumerate(dates):
            rows.extend(
                [
                    {
                        "ticker": "SMH",
                        "date": day.date(),
                        "open": 100 + index,
                        "high": 100 + index,
                        "low": 100 + index,
                        "close": 100 + index,
                        "volume": 1_000_000,
                        "adj_close": 100 + index,
                        "subindustry_benchmark": None,
                    },
                    {
                        "ticker": "IGV",
                        "date": day.date(),
                        "open": 100 + index * 0.8,
                        "high": 100 + index * 0.8,
                        "low": 100 + index * 0.8,
                        "close": 100 + index * 0.8,
                        "volume": 1_000_000,
                        "adj_close": 100 + index * 0.8,
                        "subindustry_benchmark": None,
                    },
                    {
                        "ticker": "NVDA",
                        "date": day.date(),
                        "open": 100 + index * 2.2,
                        "high": 100 + index * 2.2,
                        "low": 100 + index * 2.2,
                        "close": 100 + index * 2.2,
                        "volume": 1_000_000,
                        "adj_close": 100 + index * 2.2,
                        "subindustry_benchmark": "SMH",
                    },
                    {
                        "ticker": "AMD",
                        "date": day.date(),
                        "open": 100 + index * 1.1,
                        "high": 100 + index * 1.1,
                        "low": 100 + index * 1.1,
                        "close": 100 + index * 1.1,
                        "volume": 1_000_000,
                        "adj_close": 100 + index * 1.1,
                        "subindustry_benchmark": "SMH",
                    },
                    {
                        "ticker": "NOW",
                        "date": day.date(),
                        "open": 100 + index * 1.9,
                        "high": 100 + index * 1.9,
                        "low": 100 + index * 1.9,
                        "close": 100 + index * 1.9,
                        "volume": 1_000_000,
                        "adj_close": 100 + index * 1.9,
                        "subindustry_benchmark": "IGV",
                    },
                    {
                        "ticker": "TEAM",
                        "date": day.date(),
                        "open": 100 + index * 0.9,
                        "high": 100 + index * 0.9,
                        "low": 100 + index * 0.9,
                        "close": 100 + index * 0.9,
                        "volume": 1_000_000,
                        "adj_close": 100 + index * 0.9,
                        "subindustry_benchmark": "IGV",
                    },
                ]
            )
        frame, _ = apply_feature_definitions(
            pd.DataFrame(rows),
            {
                "features": {
                    "momentum": [
                        {
                            "name": "relative_strength_index_vs_subindustry",
                            "type": "relative_strength_percentile_contextual",
                            "params": {"window": 63, "benchmark_column": "subindustry_benchmark"},
                        }
                    ]
                }
            },
        )
        latest = frame.sort_values("date").groupby("ticker").tail(1).set_index("ticker")

        self.assertGreater(
            float(latest.loc["NVDA", "relative_strength_index_vs_subindustry"]),
            float(latest.loc["AMD", "relative_strength_index_vs_subindustry"]),
        )
        self.assertGreater(
            float(latest.loc["NOW", "relative_strength_index_vs_subindustry"]),
            float(latest.loc["TEAM", "relative_strength_index_vs_subindustry"]),
        )
        self.assertTrue(pd.isna(latest.loc["SMH", "relative_strength_index_vs_subindustry"]))
        self.assertTrue(pd.isna(latest.loc["IGV", "relative_strength_index_vs_subindustry"]))

    def test_roc_feature_measures_multi_month_return(self) -> None:
        dates = pd.bdate_range("2026-01-01", periods=70)
        rows = []
        for index, day in enumerate(dates):
            rows.append(
                {
                    "ticker": "AAA",
                    "date": day.date(),
                    "open": 100 + index,
                    "high": 100 + index,
                    "low": 100 + index,
                    "close": 100 + index,
                    "volume": 1_000_000,
                    "adj_close": 100 + index,
                }
            )
        frame, _ = apply_feature_definitions(
            pd.DataFrame(rows),
            {
                "features": {
                    "momentum": [
                        {
                            "name": "roc_63",
                            "type": "roc",
                            "params": {"window": 63},
                        }
                    ]
                }
            },
        )
        latest = frame.sort_values("date").iloc[-1]
        expected = (169.0 - 106.0) / 106.0

        self.assertAlmostEqual(float(latest["roc_63"]), expected)

    def test_breakout_feature_family_detects_base_and_breakout(self) -> None:
        dates = pd.bdate_range("2026-01-01", periods=80)
        rows = []
        for index, day in enumerate(dates):
            if index < 60:
                close = 100.0 + index * 0.5
                volume = 1_000_000
            elif index < 79:
                close = 130.0 + ((index % 4) - 1.5)
                volume = 700_000
            else:
                close = 136.0
                volume = 2_200_000
            rows.append(
                {
                    "ticker": "AAA",
                    "date": day.date(),
                    "open": close - 0.5,
                    "high": close + 1.0,
                    "low": close - 1.0,
                    "close": close,
                    "volume": volume,
                    "adj_close": close,
                }
            )
        frame, _ = apply_feature_definitions(
            pd.DataFrame(rows),
            {
                "features": {
                    "trend": [
                        {"name": "sma_50_dist", "type": "pct_diff", "params": {"window": 50}},
                        {
                            "name": "ma_alignment_50_200",
                            "type": "moving_average_gap",
                            "params": {"short_window": 50, "long_window": 70},
                        },
                        {
                            "name": "ma_slope_50_20",
                            "type": "moving_average_slope",
                            "params": {"window": 50, "slope_window": 10},
                        },
                    ],
                    "volume": [
                        {"name": "breakout_volume_ratio_50", "type": "ratio_to_avg", "params": {"window": 50}},
                        {
                            "name": "base_volume_dryup_ratio_20",
                            "type": "volume_dryup_ratio",
                            "params": {"recent_window": 10, "prior_window": 10},
                        },
                    ],
                    "price_structure": [
                        {"name": "base_range_pct_20", "type": "base_range_pct", "params": {"window": 20}},
                        {
                            "name": "base_atr_contraction_20",
                            "type": "base_atr_contraction",
                            "params": {"atr_window": 14, "recent_window": 10, "prior_window": 10},
                        },
                        {"name": "breakout_above_20d_high", "type": "breakout_above_high", "params": {"window": 20}},
                        {
                            "name": "distance_above_20d_high",
                            "type": "distance_above_high",
                            "params": {"window": 20},
                        },
                    ],
                }
            },
        )
        ordered = frame.sort_values("date").reset_index(drop=True)
        pre_breakout = ordered.iloc[-2]
        latest = ordered.iloc[-1]

        self.assertEqual(float(latest["breakout_above_20d_high"]), 1.0)
        self.assertGreater(float(latest["distance_above_20d_high"]), 0.0)
        self.assertGreater(float(latest["breakout_volume_ratio_50"]), 1.5)
        self.assertLess(float(pre_breakout["base_volume_dryup_ratio_20"]), 1.0)

    def test_liquidity_volatility_regime_features_are_point_in_time(self) -> None:
        dates = pd.bdate_range("2025-01-01", periods=320)
        rows = []
        for index, day in enumerate(dates):
            base_close = 100.0 + index * 0.15
            shock = 0.0 if index < 300 else ((index - 299) % 4) * 1.5
            close = base_close + shock
            volume = 1_000_000 if index < 260 else 2_500_000
            high = close + (1.0 if index < 300 else 3.0)
            low = close - (1.0 if index < 300 else 3.0)
            rows.append(
                {
                    "ticker": "AAA",
                    "date": day.date(),
                    "open": close - 0.25,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                    "adj_close": close,
                }
            )
        frame, _ = apply_feature_definitions(
            pd.DataFrame(rows),
            {
                "features": {
                    "trend": [
                        {"name": "atr_pct_14", "type": "atr_pct", "params": {"window": 14}},
                        {
                            "name": "atr_pct_14_percentile_252",
                            "type": "rolling_percentile",
                            "params": {"source_column": "atr_pct_14", "window": 252},
                        },
                        {
                            "name": "realized_vol_20_percentile_252",
                            "type": "realized_volatility_percentile",
                            "params": {"return_window": 20, "percentile_window": 252},
                        },
                    ],
                    "volume": [
                        {
                            "name": "dollar_volume_ratio_20_60",
                            "type": "dollar_volume_ratio",
                            "params": {"short_window": 20, "long_window": 60},
                        },
                        {
                            "name": "volume_percentile_60",
                            "type": "rolling_percentile",
                            "params": {"source_column": "volume", "window": 60},
                        },
                    ],
                    "price_structure": [
                        {
                            "name": "distance_from_52w_high",
                            "type": "distance_from_high",
                            "params": {"window": 252},
                        },
                        {
                            "name": "days_since_52w_high",
                            "type": "days_since_high",
                            "params": {"window": 252},
                        },
                    ],
                }
            },
        )
        ordered = frame.sort_values("date").reset_index(drop=True)
        latest = ordered.iloc[-1]

        self.assertGreater(float(latest["atr_pct_14_percentile_252"]), 0.8)
        self.assertGreater(float(latest["realized_vol_20_percentile_252"]), 0.8)
        self.assertGreater(float(latest["dollar_volume_ratio_20_60"]), 1.0)
        self.assertGreater(float(latest["volume_percentile_60"]), 0.5)
        self.assertLessEqual(float(latest["distance_from_52w_high"]), 0.0)
        self.assertGreaterEqual(float(latest["days_since_52w_high"]), 0.0)

    def test_earnings_timing_features_measure_business_day_distance(self) -> None:
        dates = pd.bdate_range("2026-01-05", periods=6)
        rows = []
        for index, day in enumerate(dates):
            rows.append(
                {
                    "ticker": "AAA",
                    "date": day.date(),
                    "open": 100.0 + index,
                    "high": 101.0 + index,
                    "low": 99.0 + index,
                    "close": 100.5 + index,
                    "volume": 1_000_000,
                    "adj_close": 100.5 + index,
                }
            )
        earnings_calendar = pd.DataFrame(
            [
                {"ticker": "AAA", "earnings_date": pd.Timestamp("2026-01-07")},
                {"ticker": "AAA", "earnings_date": pd.Timestamp("2026-01-14")},
            ]
        )
        frame, _ = apply_feature_definitions(
            pd.DataFrame(rows),
            {
                "features": {
                    "event_risk": [
                        {"name": "days_to_next_earnings", "type": "days_to_next_event"},
                        {"name": "days_since_last_earnings", "type": "days_since_last_event"},
                    ]
                }
            },
            earnings_calendar=earnings_calendar,
        )
        ordered = frame.sort_values("date").reset_index(drop=True)

        self.assertEqual(float(ordered.iloc[0]["days_to_next_earnings"]), 2.0)
        self.assertTrue(pd.isna(ordered.iloc[0]["days_since_last_earnings"]))
        self.assertEqual(float(ordered.iloc[2]["days_to_next_earnings"]), 0.0)
        self.assertEqual(float(ordered.iloc[2]["days_since_last_earnings"]), 0.0)
        self.assertEqual(float(ordered.iloc[3]["days_to_next_earnings"]), 4.0)
        self.assertEqual(float(ordered.iloc[3]["days_since_last_earnings"]), 1.0)

    def test_post_earnings_reaction_features_capture_gap_volume_and_hold(self) -> None:
        dates = pd.bdate_range("2026-01-01", periods=25)
        rows = []
        for index, day in enumerate(dates):
            if index < 20:
                open_price = 100.0
                close_price = 100.0
                volume = 100.0
            elif index == 20:
                open_price = 110.0
                close_price = 111.0
                volume = 250.0
            elif index == 21:
                open_price = 112.0
                close_price = 112.0
                volume = 120.0
            else:
                open_price = 109.0
                close_price = 109.0
                volume = 120.0
            rows.append(
                {
                    "ticker": "AAA",
                    "date": day.date(),
                    "open": open_price,
                    "high": close_price + 1.0,
                    "low": min(open_price, close_price) - 1.0,
                    "close": close_price,
                    "volume": volume,
                    "adj_close": close_price,
                }
            )
        earnings_calendar = pd.DataFrame(
            [{"ticker": "AAA", "earnings_date": dates[20]}]
        )
        frame, _ = apply_feature_definitions(
            pd.DataFrame(rows),
            {
                "features": {
                    "event_risk": [
                        {"name": "days_since_last_earnings", "type": "days_since_last_event"},
                        {"name": "last_earnings_gap_pct", "type": "last_earnings_gap_pct"},
                        {"name": "last_earnings_volume_ratio_20", "type": "last_earnings_volume_ratio", "params": {"window": 20}},
                        {"name": "last_earnings_open_vs_20d_high", "type": "last_earnings_open_vs_high", "params": {"window": 20}},
                        {"name": "close_vs_last_earnings_close", "type": "close_vs_last_earnings_close"},
                    ]
                }
            },
            earnings_calendar=earnings_calendar,
        )
        ordered = frame.sort_values("date").reset_index(drop=True)
        earnings_day = ordered.iloc[20]
        next_day = ordered.iloc[21]
        later_day = ordered.iloc[22]

        self.assertEqual(float(earnings_day["days_since_last_earnings"]), 0.0)
        self.assertAlmostEqual(float(earnings_day["last_earnings_gap_pct"]), 0.10, places=6)
        self.assertAlmostEqual(float(earnings_day["last_earnings_volume_ratio_20"]), 2.5, places=6)
        self.assertAlmostEqual(float(earnings_day["last_earnings_open_vs_20d_high"]), (110.0 / 101.0) - 1.0, places=6)
        self.assertAlmostEqual(float(earnings_day["close_vs_last_earnings_close"]), 0.0, places=6)
        self.assertAlmostEqual(float(next_day["close_vs_last_earnings_close"]), (112.0 / 111.0) - 1.0, places=6)
        self.assertAlmostEqual(float(later_day["close_vs_last_earnings_close"]), (109.0 / 111.0) - 1.0, places=6)

    def test_gap_risk_features_measure_average_and_worst_overnight_gaps(self) -> None:
        dates = pd.bdate_range("2026-01-01", periods=65)
        rows = []
        closes = [100.0]
        for index in range(1, len(dates)):
            closes.append(closes[-1] + 1.0)
        for index, day in enumerate(dates):
            prev_close = closes[index - 1] if index > 0 else closes[index]
            if index == 40:
                open_price = prev_close * 0.95
            elif index == 50:
                open_price = prev_close * 1.03
            else:
                open_price = prev_close * 1.005
            close_price = closes[index]
            rows.append(
                {
                    "ticker": "AAA",
                    "date": day.date(),
                    "open": open_price,
                    "high": max(open_price, close_price) + 1.0,
                    "low": min(open_price, close_price) - 1.0,
                    "close": close_price,
                    "volume": 1_000_000,
                    "adj_close": close_price,
                }
            )
        frame, _ = apply_feature_definitions(
            pd.DataFrame(rows),
            {
                "features": {
                    "gap_risk": [
                        {"name": "avg_abs_gap_pct_20", "type": "avg_abs_gap_pct", "params": {"window": 20}},
                        {"name": "max_gap_down_pct_60", "type": "max_gap_down_pct", "params": {"window": 60}},
                    ]
                }
            },
        )
        latest = frame.sort_values("date").iloc[-1]

        self.assertGreater(float(latest["avg_abs_gap_pct_20"]), 0.005)
        self.assertAlmostEqual(float(latest["max_gap_down_pct_60"]), 0.05, places=6)

    def test_build_analysis_frame_adds_sector_breadth_features(self) -> None:
        dates = pd.bdate_range("2025-01-01", periods=220)
        rows = []
        for index, day in enumerate(dates):
            rows.extend(
                [
                    {
                        "ticker": "AAA",
                        "date": day.date(),
                        "open": 100.0 + index,
                        "high": 101.0 + index,
                        "low": 99.0 + index,
                        "close": 100.0 + index,
                        "volume": 1_000_000,
                        "adj_close": 100.0 + index,
                    },
                    {
                        "ticker": "BBB",
                        "date": day.date(),
                        "open": 140.0 if index < 210 else 60.0,
                        "high": 141.0 if index < 210 else 61.0,
                        "low": 139.0 if index < 210 else 59.0,
                        "close": 140.0 if index < 210 else 60.0,
                        "volume": 1_000_000,
                        "adj_close": 140.0 if index < 210 else 60.0,
                    },
                    {
                        "ticker": "SPY",
                        "date": day.date(),
                        "open": 300.0 + index,
                        "high": 301.0 + index,
                        "low": 299.0 + index,
                        "close": 300.0 + index,
                        "volume": 2_000_000,
                        "adj_close": 300.0 + index,
                    },
                    {
                        "ticker": "QQQ",
                        "date": day.date(),
                        "open": 200.0 + index,
                        "high": 201.0 + index,
                        "low": 199.0 + index,
                        "close": 200.0 + index,
                        "volume": 2_000_000,
                        "adj_close": 200.0 + index,
                    },
                    {
                        "ticker": "USO",
                        "date": day.date(),
                        "open": 70.0 + index * 0.1,
                        "high": 70.5 + index * 0.1,
                        "low": 69.5 + index * 0.1,
                        "close": 70.0 + index * 0.1,
                        "volume": 1_500_000,
                        "adj_close": 70.0 + index * 0.1,
                    },
                ]
            )
        frame, _ = build_analysis_frame(
            pd.DataFrame(rows),
            [
                {"ticker": "AAA", "sector": "Materials", "md_volume_30d": 1_000_000},
                {"ticker": "BBB", "sector": "Materials", "md_volume_30d": 1_000_000},
            ],
        )
        latest_materials = frame[(frame["sector"] == "Materials")].sort_values("date").groupby("ticker").tail(1)

        self.assertIn("sector_pct_above_50", latest_materials.columns)
        self.assertIn("sector_pct_above_200", latest_materials.columns)
        self.assertIn("sector_median_roc_63", latest_materials.columns)
        self.assertAlmostEqual(float(latest_materials.iloc[0]["sector_pct_above_50"]), 0.5)
        self.assertAlmostEqual(float(latest_materials.iloc[0]["sector_pct_above_200"]), 0.5)

    def test_non_tech_sectors_default_to_spy_regime(self) -> None:
        self.assertEqual(regime_etf_for_sector("Industrials"), "SPY")
        self.assertEqual(regime_etf_for_sector("Information Technology"), "QQQ")

    def test_atr_feature_and_exit_helpers_use_entry_atr(self) -> None:
        dates = pd.bdate_range("2026-01-01", periods=20)
        rows = []
        for index, day in enumerate(dates):
            rows.append(
                {
                    "ticker": "AAA",
                    "date": day.date(),
                    "open": 100.0 + index,
                    "high": 102.0 + index,
                    "low": 99.0 + index,
                    "close": 101.0 + index,
                    "volume": 1_000_000,
                    "adj_close": 101.0 + index,
                }
            )
        frame, _ = apply_feature_definitions(
            pd.DataFrame(rows),
            {
                "features": {
                    "trend": [
                        {
                            "name": "atr_14",
                            "type": "atr",
                            "params": {"window": 14},
                        }
                    ]
                }
            },
        )
        latest = frame.sort_values("date").iloc[-1]
        self.assertGreater(float(latest["atr_14"]), 0.0)

        exit_rules = ExitRules(
            trailing_stop_pct=None,
            profit_target_pct=None,
            time_limit_days=20,
            trailing_stop_atr_mult=2.5,
            profit_target_atr_mult=3.0,
        )
        self.assertAlmostEqual(
            trailing_stop_price(max_price_seen=120.0, entry_atr=4.0, exit_rules=exit_rules),
            110.0,
        )
        self.assertAlmostEqual(
            profit_target_price(entry_price=100.0, entry_atr=4.0, exit_rules=exit_rules),
            112.0,
        )
        self.assertAlmostEqual(
            stop_risk_per_share(price=100.0, entry_atr=4.0, exit_rules=exit_rules),
            10.0,
        )

    def test_vol_alpha_score_is_downweighted_relative_to_other_components(self) -> None:
        vol_score = indicator_score("vol_alpha_min", actual_value=1.8, threshold_value=1.4)
        rsi_score = indicator_score("rsi_14_max", actual_value=35.0, threshold_value=35.0)

        self.assertAlmostEqual(vol_score, 4.0)
        self.assertAlmostEqual(rsi_score, 10.0)
