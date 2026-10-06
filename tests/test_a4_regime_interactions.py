from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from scripts import a4_regime_interactions as a4
from src.research.shortlist_bakeoff_service import (
    A4_REGIME_INTERACTION_FEATURES,
    MODEL_FEATURE_COLUMNS,
)
from src.research.shortlist_model_service import ShortlistModelService


class A4RegimeInteractionTests(unittest.TestCase):
    def test_parse_top_feature_ic_features_uses_surviving_rows_in_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "feature_ic_report.md"
            path.write_text(
                "\n".join(
                    [
                        "| feature | rank_ic | abs_rank_ic | observations | duplicate_of | survives |",
                        "|---|---:|---:|---:|---|---|",
                        "| sector_median_roc_63 | -0.12 | 0.12 | 100 |  | true |",
                        "| roc_126__rank_all | 0.04 | 0.04 | 100 |  | true |",
                        "| ignored | 0.01 | 0.01 | 100 |  | false |",
                    ]
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                a4._parse_top_feature_ic_features(path, limit=2),
                ["sector_median_roc_63", "roc_126__rank_all"],
            )

    def test_regime_classifications_match_train_and_flip_lag(self) -> None:
        universe_dates = pd.bdate_range("2026-01-01", periods=70).tolist()
        regime_meter = pd.DataFrame(
            [
                {"snapshot_date": universe_dates[0], "classification": "trending"},
                {"snapshot_date": universe_dates[5], "classification": "reversal"},
            ]
        )

        mapped = a4._regime_classifications_by_prediction_date(
            prediction_dates=[universe_dates[-1]],
            universe_dates=universe_dates,
            regime_meter=regime_meter,
            horizon_sessions=60,
        )

        self.assertEqual(mapped[pd.Timestamp(universe_dates[-1]).normalize()], "reversal")

    def test_build_regime_interaction_frame_adds_main_effects_and_interactions(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "snapshot_date": "2026-01-02",
                    "ticker": "AAA",
                    "sector": "Technology",
                    "relative_strength_index_vs_spy": 0.2,
                    "alpha_vs_sector_60d": 0.01,
                },
                {
                    "snapshot_date": "2026-01-02",
                    "ticker": "BBB",
                    "sector": "Technology",
                    "relative_strength_index_vs_spy": 0.6,
                    "alpha_vs_sector_60d": 0.02,
                },
                {
                    "snapshot_date": "2026-01-05",
                    "ticker": "AAA",
                    "sector": "Technology",
                    "relative_strength_index_vs_spy": 0.4,
                    "alpha_vs_sector_60d": 0.03,
                },
            ]
        )
        regime_by_date = {
            pd.Timestamp("2026-01-02"): "trending",
            pd.Timestamp("2026-01-05"): "reversal",
        }

        enriched, columns = a4.build_regime_interaction_frame(
            frame,
            top_features=["relative_strength_index_vs_spy"],
            regime_by_date=regime_by_date,
        )

        trending_column = "a4_relative_strength_index_vs_spy_x_regime_trending"
        reversal_column = "a4_relative_strength_index_vs_spy_x_regime_reversal"
        self.assertIn("a4_regime_trending", columns)
        self.assertIn(trending_column, columns)
        self.assertIn(reversal_column, columns)
        self.assertEqual(enriched.loc[0, "a4_regime_trending"], 1.0)
        self.assertEqual(enriched.loc[2, "a4_regime_reversal"], 1.0)
        self.assertAlmostEqual(enriched.loc[0, trending_column], 0.2)
        self.assertAlmostEqual(enriched.loc[2, reversal_column], 0.4)
        self.assertEqual(enriched.loc[2, trending_column], 0.0)

    def test_temporary_model_features_restores_global_feature_list(self) -> None:
        before = list(MODEL_FEATURE_COLUMNS)
        with a4._temporary_model_features(["a4_unit_test_feature"]):
            self.assertIn("a4_unit_test_feature", MODEL_FEATURE_COLUMNS)

        self.assertEqual(MODEL_FEATURE_COLUMNS, before)

    def test_production_feature_list_includes_a4_interactions(self) -> None:
        self.assertIn("a4_regime_trending", MODEL_FEATURE_COLUMNS)
        self.assertIn("a4_sector_pct_above_50_x_regime_reversal", MODEL_FEATURE_COLUMNS)
        self.assertTrue(set(A4_REGIME_INTERACTION_FEATURES).issubset(set(MODEL_FEATURE_COLUMNS)))

    def test_shortlist_service_adds_lagged_regime_interaction_features(self) -> None:
        dates = pd.bdate_range("2026-01-01", periods=70)

        class FakeDB:
            def load_regime_meter(self):
                return pd.DataFrame(
                    [
                        {"snapshot_date": dates[0], "classification": "trending"},
                        {"snapshot_date": dates[5], "classification": "reversal"},
                    ]
                )

            def list_universe_daily_snapshot_dates(self):
                return list(dates)

        service = ShortlistModelService(FakeDB())
        frame = pd.DataFrame(
            [
                {
                    "snapshot_date": dates[-1],
                    "ticker": "AAA",
                    "sector": "Technology",
                    "md_volume_30d": 1_000_000.0,
                    "adj_close": 10.0,
                    "passed_any_strategy": True,
                    "sector_pct_above_50": 0.25,
                },
                {
                    "snapshot_date": dates[-1],
                    "ticker": "BBB",
                    "sector": "Technology",
                    "md_volume_30d": 1_000_000.0,
                    "adj_close": 12.0,
                    "passed_any_strategy": True,
                    "sector_pct_above_50": 0.75,
                },
            ]
        )

        enriched = service._add_a4_regime_interaction_features(frame, horizon_sessions=60)

        self.assertEqual(enriched["a4_regime_reversal"].tolist(), [1.0, 1.0])
        self.assertEqual(enriched["a4_regime_trending"].tolist(), [0.0, 0.0])
        self.assertEqual(
            enriched["a4_sector_pct_above_50_x_regime_reversal"].tolist(),
            [0.25, 0.75],
        )
        self.assertEqual(
            enriched["a4_sector_pct_above_50_x_regime_trending"].tolist(),
            [0.0, 0.0],
        )


if __name__ == "__main__":
    unittest.main()
