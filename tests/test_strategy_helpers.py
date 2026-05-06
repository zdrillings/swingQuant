from __future__ import annotations

import unittest

import pandas as pd

from src.utils.feature_engineering import apply_feature_definitions
from src.utils.regime import regime_etf_for_sector
from src.utils.signal_engine import filter_signal_candidates
from src.utils.strategy import ExitRules, profit_target_price, stop_risk_per_share, trailing_stop_price


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
