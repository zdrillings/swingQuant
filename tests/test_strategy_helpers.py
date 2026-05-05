from __future__ import annotations

import unittest

import pandas as pd

from src.utils.regime import regime_etf_for_sector
from src.utils.signal_engine import filter_signal_candidates


class StrategyHelperTests(unittest.TestCase):
    def test_filter_signal_candidates_requires_strict_and_gate(self) -> None:
        frame = pd.DataFrame(
            [
                {"ticker": "AAA", "rsi_14": 30.0, "vol_alpha": 1.6},
                {"ticker": "BBB", "rsi_14": 36.0, "vol_alpha": 1.6},
                {"ticker": "CCC", "rsi_14": 30.0, "vol_alpha": 1.2},
            ]
        )

        candidates = filter_signal_candidates(
            frame,
            {
                "rsi_14_max": 35.0,
                "vol_alpha_min": 1.5,
            },
        )

        self.assertEqual(candidates["ticker"].tolist(), ["AAA"])

    def test_non_tech_sectors_default_to_spy_regime(self) -> None:
        self.assertEqual(regime_etf_for_sector("Industrials"), "SPY")
        self.assertEqual(regime_etf_for_sector("Information Technology"), "QQQ")
