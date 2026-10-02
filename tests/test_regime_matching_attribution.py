from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from scripts.regime_matching_attribution import (
    _disable_recommendation,
    _render_report,
)


class RegimeMatchingAttributionTests(unittest.TestCase):
    def test_disable_recommendation_when_unmatched_is_neutral_or_better(self) -> None:
        summary = pd.DataFrame(
            [
                {"model": "a", "matched_spearman": 0.0180, "unmatched_spearman": 0.0170, "delta_unmatched_minus_matched": -0.0010},
                {"model": "b", "matched_spearman": -0.0100, "unmatched_spearman": -0.0060, "delta_unmatched_minus_matched": 0.0040},
            ]
        )

        disable, reason = _disable_recommendation(summary, tolerance=0.0025)

        self.assertTrue(disable)
        self.assertIn("unmatched best", reason)

    def test_disable_recommendation_keeps_matching_when_matched_clearly_wins(self) -> None:
        summary = pd.DataFrame(
            [
                {"model": "a", "matched_spearman": 0.0300, "unmatched_spearman": 0.0200, "delta_unmatched_minus_matched": -0.0100},
                {"model": "b", "matched_spearman": 0.0100, "unmatched_spearman": 0.0080, "delta_unmatched_minus_matched": -0.0020},
            ]
        )

        disable, reason = _disable_recommendation(summary, tolerance=0.0025)

        self.assertFalse(disable)
        self.assertIn("matched best", reason)

    def test_render_report_includes_full_oos_and_per_fold_tables(self) -> None:
        summary = pd.DataFrame(
            [
                {
                    "model": "signal_proxy",
                    "dates": 20,
                    "matched_spearman": 0.0133,
                    "unmatched_spearman": 0.0140,
                    "delta_unmatched_minus_matched": 0.0007,
                    "matched_mean_target": 0.01,
                    "unmatched_mean_target": 0.02,
                    "matched_beat_universe_rate": 0.51,
                    "unmatched_beat_universe_rate": 0.52,
                }
            ]
        )
        folds = pd.DataFrame(
            [
                {
                    "fold": 1,
                    "fold_start": "2026-01-02",
                    "fold_end": "2026-01-30",
                    "model": "signal_proxy",
                    "matched_spearman": 0.01,
                    "unmatched_spearman": 0.02,
                    "delta_unmatched_minus_matched": 0.01,
                }
            ]
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "regime_matching_attribution.md"
            text = _render_report(
                output_path=output_path,
                summary=summary,
                fold_summary=folds,
                eligible_rows=100,
                eligible_dates=30,
                oos_dates=20,
                matched_stats={"attempted_folds": 1, "matched_folds": 1, "fallback_folds": 0, "unknown_folds": 0},
                disable_recommended=True,
                disable_reason="neutral",
            )
            self.assertTrue(output_path.exists())

        self.assertIn("## Full-OOS Spearman", text)
        self.assertIn("## Per-Fold Delta", text)
        self.assertIn("| signal_proxy | 20 | +0.0133 | +0.0140 | +0.0007 |", text)
        self.assertIn("disable regime_matching", text)


if __name__ == "__main__":
    unittest.main()
