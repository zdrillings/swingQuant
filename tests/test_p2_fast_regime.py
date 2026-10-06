from __future__ import annotations

import unittest

import pandas as pd

from scripts import p2_fast_regime as p2


class P2FastRegimeTests(unittest.TestCase):
    def test_fast_regime_uses_positive_short_and_long_returns_for_trending(self) -> None:
        dates = pd.bdate_range("2026-01-01", periods=25)
        prices = pd.DataFrame({"date": dates, "adj_close": [100.0 + index for index in range(len(dates))]})

        regime = p2.fast_regime_by_snapshot_date(prices, [dates[-1]])

        self.assertEqual(regime[pd.Timestamp(dates[-1]).normalize()], "trending")

    def test_fast_regime_maps_any_negative_momentum_to_reversal(self) -> None:
        dates = pd.bdate_range("2026-01-01", periods=25)
        prices = pd.DataFrame({"date": dates, "adj_close": [125.0 - index for index in range(len(dates))]})

        regime = p2.fast_regime_by_snapshot_date(prices, [dates[-1]])

        self.assertEqual(regime[pd.Timestamp(dates[-1]).normalize()], "reversal")

    def test_fast_regime_asof_matches_missing_snapshot_date_to_previous_close(self) -> None:
        dates = pd.bdate_range("2026-01-01", periods=25)
        prices = pd.DataFrame({"date": dates, "adj_close": [100.0 + index for index in range(len(dates))]})
        missing_market_date = pd.Timestamp("2026-02-07")

        regime = p2.fast_regime_by_snapshot_date(prices, [missing_market_date])

        self.assertEqual(regime[missing_market_date.normalize()], "trending")

    def test_fast_regime_omits_dates_without_enough_history(self) -> None:
        dates = pd.bdate_range("2026-01-01", periods=10)
        prices = pd.DataFrame({"date": dates, "adj_close": [100.0 + index for index in range(len(dates))]})

        regime = p2.fast_regime_by_snapshot_date(prices, [dates[-1]])

        self.assertEqual(regime, {})

    def test_verdict_requires_five_basis_points_of_spearman_gain(self) -> None:
        bakeoff = pd.DataFrame(
            [
                {"model": "ridge_model", "fast_delta_vs_meter": 0.0049},
                {"model": "lasso_model", "fast_delta_vs_meter": -0.0010},
            ]
        )

        self.assertIn("ties", p2._verdict(bakeoff))

        bakeoff.loc[0, "fast_delta_vs_meter"] = 0.005
        self.assertIn("wins", p2._verdict(bakeoff))


if __name__ == "__main__":
    unittest.main()
