from __future__ import annotations

import unittest

import pandas as pd

from src.research.features import build_feature_frame, chronological_split


class ResearchFeatureTests(unittest.TestCase):
    def test_chronological_split_preserves_time_order(self) -> None:
        frame = pd.DataFrame(
            {
                "ticker": ["AAA"] * 10 + ["BBB"] * 10,
                "date": list(pd.date_range("2024-01-01", periods=10)) * 2,
                "success": [0] * 20,
                "feature_a": list(range(20)),
            }
        )

        train_frame, validation_frame = chronological_split(frame, train_ratio=0.7)

        self.assertLess(train_frame["date"].max(), validation_frame["date"].min())

    def test_build_feature_frame_drops_rows_without_forward_label(self) -> None:
        base_dates = pd.date_range("2024-01-01", periods=30)
        price_history = pd.DataFrame(
            {
                "ticker": ["AAA"] * 30 + ["USO"] * 30,
                "date": list(base_dates) * 2,
                "open": [100 + index for index in range(30)] * 2,
                "high": [101 + index for index in range(30)] * 2,
                "low": [99 + index for index in range(30)] * 2,
                "close": [100 + index for index in range(30)] * 2,
                "volume": [1000 + index for index in range(30)] * 2,
                "adj_close": [100 + index for index in range(30)] * 2,
            }
        )
        feature_config = {
            "features": {
                "trend": [{"name": "sma_2_dist", "type": "pct_diff", "params": {"window": 2}}],
                "momentum": [{"name": "rsi_2", "type": "rsi", "params": {"window": 2}}],
                "volume": [{"name": "vol_alpha", "type": "ratio_to_avg", "params": {"window": 2}}],
                "commodities": [
                    {
                        "name": "oil_corr_2",
                        "type": "correlation",
                        "ticker": "USO",
                        "params": {"window": 2},
                    }
                ],
            }
        }

        feature_frame, feature_columns = build_feature_frame(price_history, feature_config)

        self.assertEqual(
            feature_columns,
            ["sma_2_dist", "rsi_2", "vol_alpha", "oil_corr_2"],
        )
        unlabeled_tail = set(base_dates[-20:])
        aaa_dates = set(feature_frame.loc[feature_frame["ticker"] == "AAA", "date"])
        self.assertTrue(aaa_dates.isdisjoint(unlabeled_tail))
