from __future__ import annotations

import unittest

import pandas as pd

from src.utils.feature_engineering import apply_feature_definitions
from src.utils.regime import regime_etf_for_sector
from src.utils.signal_engine import build_analysis_frame, filter_signal_candidates
from src.utils.strategy import ExitRules, indicator_score, profit_target_price, stop_risk_per_share, trailing_stop_price


class StrategyHelperTests(unittest.TestCase):
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
