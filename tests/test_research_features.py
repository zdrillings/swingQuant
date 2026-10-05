from __future__ import annotations

import unittest
import json

import pandas as pd

from src.research.shortlist_bakeoff_service import MODEL_FEATURE_COLUMNS, expand_model_feature_columns
from src.research.features import build_feature_frame, chronological_split
from src.research.universe_snapshot_service import UniverseSnapshotBackfillService
from src.utils.feature_engineering import add_overnight_rth_return_features


class ResearchFeatureTests(unittest.TestCase):
    def test_overnight_rth_features_use_prior_close_and_current_session(self) -> None:
        frame = pd.DataFrame(
            {
                "ticker": ["AAA"] * 6,
                "date": pd.bdate_range("2026-01-02", periods=6),
                "open": [10.0, 11.0, 12.0, 11.0, 13.0, 12.0],
                "high": [10.5, 11.5, 12.5, 11.5, 13.5, 12.5],
                "low": [9.5, 10.5, 11.5, 10.5, 12.5, 11.5],
                "close": [10.0, 12.0, 11.0, 13.0, 12.0, 15.0],
                "volume": [1000] * 6,
            }
        )

        add_overnight_rth_return_features(frame, windows=(5,))

        expected_overnight = (
            (11.0 / 10.0 - 1.0)
            + (12.0 / 12.0 - 1.0)
            + (11.0 / 11.0 - 1.0)
            + (13.0 / 13.0 - 1.0)
            + (12.0 / 12.0 - 1.0)
        )
        expected_rth = (
            (12.0 / 11.0 - 1.0)
            + (11.0 / 12.0 - 1.0)
            + (13.0 / 11.0 - 1.0)
            + (12.0 / 13.0 - 1.0)
            + (15.0 / 12.0 - 1.0)
        )
        self.assertTrue(pd.isna(frame.loc[4, "overnight_ret_5d"]))
        self.assertAlmostEqual(float(frame.loc[5, "overnight_ret_5d"]), expected_overnight)
        self.assertAlmostEqual(float(frame.loc[5, "rth_ret_5d"]), expected_rth)
        self.assertAlmostEqual(
            float(frame.loc[5, "overnight_minus_rth_5d"]),
            expected_overnight - expected_rth,
        )

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

    def test_analyst_revision_acceleration_uses_14_session_point_in_time_lag(self) -> None:
        service = UniverseSnapshotBackfillService(db_manager=None)
        revisions = pd.DataFrame(
            [
                {
                    "snapshot_date": "2026-01-02",
                    "eps_revisions_json": json.dumps(
                        [{"period": "0q", "upLast30days": 2, "downLast30days": 2}]
                    ),
                    "earnings_estimate_json": json.dumps(
                        [{"period": "0q", "avg": 2.0, "high": 2.4, "low": 1.8}]
                    ),
                    "upgrades_downgrades_json": "[]",
                },
                {
                    "snapshot_date": "2026-01-26",
                    "eps_revisions_json": json.dumps(
                        [{"period": "0q", "upLast30days": 6, "downLast30days": 2}]
                    ),
                    "earnings_estimate_json": json.dumps(
                        [{"period": "0q", "avg": 2.0, "high": 2.2, "low": 1.9}]
                    ),
                    "upgrades_downgrades_json": "[]",
                },
            ]
        )
        revisions["snapshot_date"] = pd.to_datetime(revisions["snapshot_date"]).dt.normalize()

        early = service._analyst_feature_payload(
            snapshot_date="2026-01-16",
            ticker="AAA",
            adj_close=10.0,
            analyst_context={},
            analyst_revision_context={"AAA": revisions},
        )
        mature = service._analyst_feature_payload(
            snapshot_date="2026-01-26",
            ticker="AAA",
            adj_close=10.0,
            analyst_context={},
            analyst_revision_context={"AAA": revisions},
        )

        self.assertIsNone(early["analyst_eps_revision_breadth_change_14d"])
        self.assertIsNone(early["analyst_eps_estimate_dispersion_change_14d"])
        self.assertAlmostEqual(mature["analyst_eps_revision_breadth"], 0.5)
        self.assertAlmostEqual(mature["analyst_eps_estimate_dispersion"], 0.15)
        self.assertAlmostEqual(mature["analyst_eps_revision_breadth_change_14d"], 0.5)
        self.assertAlmostEqual(mature["analyst_eps_estimate_dispersion_change_14d"], -0.15)

    def test_analyst_revision_acceleration_missing_history_stays_null(self) -> None:
        service = UniverseSnapshotBackfillService(db_manager=None)

        payload = service._analyst_feature_payload(
            snapshot_date="2026-01-16",
            ticker="AAA",
            adj_close=10.0,
            analyst_context={},
            analyst_revision_context={},
        )

        self.assertIsNone(payload["analyst_eps_revision_breadth"])
        self.assertIsNone(payload["analyst_eps_revision_breadth_change_14d"])
        self.assertIsNone(payload["analyst_eps_estimate_dispersion"])
        self.assertIsNone(payload["analyst_eps_estimate_dispersion_change_14d"])

    def test_revision_acceleration_features_expand_to_rank_variants(self) -> None:
        self.assertIn("analyst_eps_revision_breadth_change_14d", MODEL_FEATURE_COLUMNS)
        self.assertIn("analyst_eps_estimate_dispersion_change_14d", MODEL_FEATURE_COLUMNS)
        expanded = expand_model_feature_columns(MODEL_FEATURE_COLUMNS)
        self.assertIn("analyst_eps_revision_breadth_change_14d__rank_all", expanded)
        self.assertIn("analyst_eps_estimate_dispersion_change_14d__rank_sector", expanded)
