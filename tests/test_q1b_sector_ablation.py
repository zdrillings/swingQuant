from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from scripts.q1b_sector_ablation import (
    _decile_calibration,
    _mean_per_date_spearman,
    _pooled_spearman,
    _render_report,
    _top_n_summary,
)


class Q1bSectorAblationTests(unittest.TestCase):
    def test_spearman_helpers_average_valid_dates_only(self) -> None:
        frame = pd.DataFrame(
            [
                {"snapshot_date": "2026-01-02", "ticker": "A", "predicted_alpha": 3, "alpha_vs_sector_20d": 0.3},
                {"snapshot_date": "2026-01-02", "ticker": "B", "predicted_alpha": 2, "alpha_vs_sector_20d": 0.2},
                {"snapshot_date": "2026-01-02", "ticker": "C", "predicted_alpha": 1, "alpha_vs_sector_20d": 0.1},
                {"snapshot_date": "2026-01-05", "ticker": "A", "predicted_alpha": 1, "alpha_vs_sector_20d": 0.3},
                {"snapshot_date": "2026-01-05", "ticker": "B", "predicted_alpha": 2, "alpha_vs_sector_20d": 0.2},
                {"snapshot_date": "2026-01-05", "ticker": "C", "predicted_alpha": 3, "alpha_vs_sector_20d": 0.1},
                {"snapshot_date": "2026-01-06", "ticker": "A", "predicted_alpha": 1, "alpha_vs_sector_20d": 0.1},
                {"snapshot_date": "2026-01-06", "ticker": "B", "predicted_alpha": 1, "alpha_vs_sector_20d": 0.2},
                {"snapshot_date": "2026-01-06", "ticker": "C", "predicted_alpha": 1, "alpha_vs_sector_20d": 0.3},
            ]
        )

        self.assertAlmostEqual(
            _mean_per_date_spearman(frame, target_column="alpha_vs_sector_20d"),
            0.0,
        )
        self.assertTrue(-1.0 <= _pooled_spearman(frame, target_column="alpha_vs_sector_20d") <= 1.0)

    def test_top_n_and_deciles_use_high_scores_as_picks(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "snapshot_date": "2026-01-02",
                    "ticker": f"T{index:02d}",
                    "predicted_alpha": float(index),
                    "alpha_vs_sector_20d": float(index) / 100.0,
                }
                for index in range(20)
            ]
        )

        summary = _top_n_summary(frame, target_column="alpha_vs_sector_20d", top_n=2)
        deciles = _decile_calibration(frame, target_column="alpha_vs_sector_20d")

        self.assertAlmostEqual(summary["mean_target"], 0.185)
        self.assertEqual(len(deciles.index), 10)
        self.assertLess(float(deciles.iloc[0]["mean_target"]), float(deciles.iloc[-1]["mean_target"]))

    def test_render_report_writes_verdict_and_shared_grid_table(self) -> None:
        rows = [
            {
                "model": "xgboost_model",
                "path": "sector-specific-20d",
                "target": "alpha_vs_sector_20d",
                "dates": 2,
                "rows": 6,
                "pooled_spearman": 0.2,
                "per_date_spearman": 0.18,
                "top_n": 10,
                "top_mean": 0.04,
            },
            {
                "model": "ridge_model",
                "path": "current-global-60d",
                "target": "alpha_vs_sector_60d",
                "dates": 2,
                "rows": 6,
                "pooled_spearman": 0.01,
                "per_date_spearman": 0.02,
                "top_n": 2,
                "top_mean": 0.01,
            },
        ]
        deciles = pd.DataFrame([{"decile": 0, "rows": 3, "mean_target": -0.01, "hit_rate": 0.3}])
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "q1b.md"
            text = _render_report(
                rows=rows,
                deciles=deciles,
                sector_dates=2,
                shared_dates=2,
                eligible_rows=10,
                eligible_dates=5,
                output_path=output_path,
            )
            self.assertTrue(output_path.exists())

        self.assertIn("| xgboost_model | sector-specific-20d |", text)
        self.assertIn("edge recovered:", text)


if __name__ == "__main__":
    unittest.main()
