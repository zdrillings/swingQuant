from __future__ import annotations

import unittest

import pandas as pd

from scripts.p3w_expected_tonight import gate_checks_for_model, gate_verdict
from src.research.shortlist_model_service import ShortlistModelService


class P3WExpectedTonightTest(unittest.TestCase):
    def test_gate_checks_pass_when_ridge_adaptive_meets_all_active_floors(self) -> None:
        service = ShortlistModelService(db_manager=None)  # type: ignore[arg-type]
        summaries = pd.DataFrame(
            [
                {
                    "model": "ridge_adaptive_last_fold",
                    "hit_rate_excess": 0.03,
                    "beat_universe_rate": 0.51,
                    "mean_target_excess": 0.001,
                    "spearman": 0.01,
                    "top_ticker_date_rate": 0.30,
                },
                {
                    "model": "ridge_adaptive_trailing_3folds",
                    "hit_rate_excess": 0.04,
                    "beat_universe_rate": 0.55,
                    "mean_target_excess": 0.002,
                    "spearman": 0.02,
                    "top_ticker_date_rate": 0.35,
                },
                {
                    "model": "ridge_adaptive_full_oos",
                    "spearman": 0.01,
                },
            ]
        )
        checks = gate_checks_for_model(
            service=service,
            acceptance_summaries=summaries,
            promotion_gate=self._gate(),
            model_name="ridge_adaptive",
            required_fold_windows=(1, 3),
        )

        passed, failures = gate_verdict(checks)

        self.assertTrue(passed)
        self.assertEqual([], failures)

    def test_gate_checks_report_exact_failing_floors(self) -> None:
        service = ShortlistModelService(db_manager=None)  # type: ignore[arg-type]
        summaries = pd.DataFrame(
            [
                {
                    "model": "ridge_adaptive_last_fold",
                    "hit_rate_excess": 0.01,
                    "beat_universe_rate": 0.49,
                    "mean_target_excess": 0.001,
                    "spearman": 0.02,
                    "top_ticker_date_rate": 0.41,
                },
                {
                    "model": "ridge_adaptive_trailing_3folds",
                    "hit_rate_excess": 0.03,
                    "beat_universe_rate": 0.55,
                    "mean_target_excess": -0.001,
                    "spearman": 0.02,
                    "top_ticker_date_rate": 0.35,
                },
                {
                    "model": "ridge_adaptive_full_oos",
                    "spearman": -0.01,
                },
            ]
        )
        checks = gate_checks_for_model(
            service=service,
            acceptance_summaries=summaries,
            promotion_gate=self._gate(),
            model_name="ridge_adaptive",
            required_fold_windows=(1, 3),
        )

        passed, failures = gate_verdict(checks)

        self.assertFalse(passed)
        self.assertEqual(
            [
                "last_fold hit_rate_excess",
                "last_fold beat_universe_rate",
                "last_fold top_ticker_date_rate",
                "trailing_3folds mean_target_excess",
                "full_oos spearman",
            ],
            failures,
        )

    def _gate(self) -> dict[str, float | int]:
        return {
            "min_recent_1fold_hit_rate_excess": 0.02,
            "min_recent_1fold_beat_universe_rate": 0.50,
            "min_recent_1fold_mean_target_excess": 0.0,
            "min_recent_1fold_spearman": 0.0,
            "max_recent_1fold_top_ticker_date_rate": 0.40,
            "min_recent_3fold_hit_rate_excess": 0.02,
            "min_recent_3fold_beat_universe_rate": 0.50,
            "min_recent_3fold_mean_target_excess": 0.0,
            "min_recent_3fold_spearman": 0.0,
            "max_recent_3fold_top_ticker_date_rate": 0.40,
            "min_full_oos_spearman": 0.0,
        }


if __name__ == "__main__":
    unittest.main()
