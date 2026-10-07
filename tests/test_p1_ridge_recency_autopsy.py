from __future__ import annotations

import unittest

import pandas as pd

from scripts.p1_ridge_recency_autopsy import (
    B1_OVERNIGHT_FEATURES,
    TARGET_COLUMN,
    has_b1_reason,
    rank_blend_predictions,
    render_report,
    variant_metric_rows,
)


class P1RidgeRecencyAutopsyTests(unittest.TestCase):
    def test_rank_blend_combines_ridge_and_overnight_percentile_ranks(self) -> None:
        rows = []
        for model_name, scores in {
            "ridge_adaptive": {"AAA": 3.0, "BBB": 2.0, "CCC": 1.0},
            "overnight_session_specialist": {"AAA": 1.0, "BBB": 2.0, "CCC": 3.0},
        }.items():
            for ticker, score in scores.items():
                rows.append(
                    {
                        "snapshot_date": pd.Timestamp("2026-01-02"),
                        "ticker": ticker,
                        "sector": "Technology",
                        "model_name": model_name,
                        "predicted_alpha": score,
                        TARGET_COLUMN: 0.01,
                        "model_top_reasons": "",
                    }
                )
        raw = pd.DataFrame(rows)

        blended = rank_blend_predictions(
            raw,
            left_model="ridge_adaptive",
            right_model="overnight_session_specialist",
            left_weight=0.7,
            right_weight=0.3,
        ).set_index("ticker")

        self.assertGreater(blended.loc["AAA", "predicted_alpha"], blended.loc["CCC", "predicted_alpha"])
        self.assertAlmostEqual(float(blended.loc["BBB", "predicted_alpha"]), 2.0 / 3.0)

    def test_variant_metric_rows_marks_gate_passing_blend(self) -> None:
        dates = pd.bdate_range("2026-01-02", periods=65)
        rows = []
        for snapshot_date in dates:
            for ticker, score, target in (
                ("AAA", 3.0, 0.05),
                ("BBB", 2.0, 0.04),
                ("CCC", 1.0, -0.02),
            ):
                rows.append(
                    {
                        "snapshot_date": snapshot_date,
                        "ticker": ticker,
                        "sector": "Technology",
                        "predicted_alpha": score,
                        TARGET_COLUMN: target,
                    }
                )

        metrics = variant_metric_rows({"toy": pd.DataFrame(rows)})
        row = metrics.iloc[0]

        self.assertGreaterEqual(float(row["full_oos_spearman"]), 0.0)
        self.assertGreater(float(row["trailing_3fold_hit_rate_excess"]), 0.02)
        self.assertGreaterEqual(float(row["trailing_3fold_beat_universe_rate"]), 0.50)

    def test_report_names_b1_family_and_audit_only_blend(self) -> None:
        rows = pd.DataFrame(
            [
                {
                    "variant": "ridge_adaptive",
                    "rows": 3,
                    "dates": 1,
                    "full_oos_spearman": 0.01,
                    "trailing_3fold_spearman": 0.02,
                    "trailing_3fold_hit_rate_excess": 0.01,
                    "trailing_3fold_beat_universe_rate": 0.48,
                    "last_fold_hit_rate_excess": 0.03,
                    "last_fold_beat_universe_rate": 0.50,
                }
            ]
        )
        raw = pd.DataFrame({"snapshot_date": [pd.Timestamp("2026-01-02")]})

        report = render_report(raw=raw, calendar_dates=[], rows=rows, ablation=pd.DataFrame())

        self.assertIn("# P1 Ridge Recency Autopsy", report)
        self.assertIn(B1_OVERNIGHT_FEATURES[0], report)
        self.assertIn("audit variant only", report)

    def test_has_b1_reason_detects_overnight_family(self) -> None:
        self.assertTrue(has_b1_reason("['overnight_ret_20d__rank_sector', 'roc_63']"))
        self.assertFalse(has_b1_reason("['roc_63', 'sma_200_dist']"))


if __name__ == "__main__":
    unittest.main()
