from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from scripts.q3_label_overlap import (
    OverlapSummary,
    render_report,
    summarize_label_overlap,
    walk_forward_fold_training_rows,
)


class Q3LabelOverlapTests(unittest.TestCase):
    def test_summarize_label_overlap_detects_adjacent_forward_window_overlap(self) -> None:
        dates = pd.date_range("2026-01-01", periods=90, freq="B")
        frame = pd.DataFrame(
            [
                {"ticker": "AAA", "snapshot_date": dates[0]},
                {"ticker": "AAA", "snapshot_date": dates[20]},
                {"ticker": "AAA", "snapshot_date": dates[60]},
                {"ticker": "BBB", "snapshot_date": dates[0]},
                {"ticker": "BBB", "snapshot_date": dates[60]},
            ]
        )

        summary = summarize_label_overlap(
            frame,
            label="sample",
            horizon_days=60,
            calendar_dates=dates,
        )

        self.assertEqual(summary.raw_rows, 5)
        self.assertAlmostEqual(summary.row_overlap_share, 3 / 5)
        self.assertAlmostEqual(summary.pair_overlap_share, 2 / 3)
        self.assertEqual(summary.max_overlap_days, 40)
        self.assertEqual(summary.independent_rows, 4)

    def test_walk_forward_fold_training_rows_apply_embargo_and_horizon_stride(self) -> None:
        dates = pd.date_range("2026-01-01", periods=16, freq="B")
        frame = pd.DataFrame(
            [
                {"ticker": ticker, "snapshot_date": date}
                for ticker in ("AAA", "BBB")
                for date in dates
            ]
        )

        fold_rows = walk_forward_fold_training_rows(
            frame,
            min_train_dates=6,
            max_train_dates=None,
            test_window_dates=2,
            oos_stride_dates=2,
            label_horizon_dates=4,
        )

        first_fold = fold_rows[fold_rows["fold_start"].eq(pd.Timestamp(dates[10]).normalize())]
        self.assertEqual(set(first_fold["snapshot_date"]), {pd.Timestamp(dates[1]), pd.Timestamp(dates[5])})
        summary = summarize_label_overlap(
            first_fold,
            label="first_fold",
            horizon_days=4,
            calendar_dates=dates,
        )
        self.assertEqual(summary.max_overlap_days, 0)

    def test_render_report_includes_verdicts_for_training_and_oos_grid(self) -> None:
        summaries = [
            OverlapSummary(
                label="raw_labeled_snapshot_rows",
                raw_rows=10,
                tickers=2,
                dates=5,
                row_overlap_share=1.0,
                pair_overlap_share=1.0,
                max_overlap_days=59,
                median_overlap_days=59.0,
                p90_overlap_days=59.0,
                median_gap_days=1.0,
                p10_gap_days=1.0,
                p90_gap_days=1.0,
                independent_rows=2,
            ),
            OverlapSummary(
                label="walk_forward_training_rows_per_fold",
                raw_rows=4,
                tickers=2,
                dates=2,
                row_overlap_share=0.0,
                pair_overlap_share=0.0,
                max_overlap_days=0,
                median_overlap_days=0.0,
                p90_overlap_days=0.0,
                median_gap_days=60.0,
                p10_gap_days=60.0,
                p90_gap_days=60.0,
                independent_rows=4,
            ),
            OverlapSummary(
                label="persisted_oos_prediction_grid",
                raw_rows=8,
                tickers=2,
                dates=4,
                row_overlap_share=1.0,
                pair_overlap_share=1.0,
                max_overlap_days=59,
                median_overlap_days=59.0,
                p90_overlap_days=59.0,
                median_gap_days=1.0,
                p10_gap_days=1.0,
                p90_gap_days=1.0,
                independent_rows=2,
            ),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "q3.md"
            text = render_report(
                output_path=output,
                summaries=summaries,
                target_column="alpha_vs_sector_60d",
                horizon_days=60,
                min_train_dates=252,
                max_train_dates=252,
                test_window_dates=20,
                oos_stride_dates=20,
                raw_label_date_min="2024-01-01",
                raw_label_date_max="2026-10-01",
            )
            self.assertTrue(output.exists())

        self.assertIn("VERDICT raw labels: labels overlap; walk-forward training rows: clean; OOS grid: labels overlap.", text)
        self.assertIn("| persisted_oos_prediction_grid | 8 | 2 | 4 | 1.0000 |", text)
        self.assertIn("label_construction: `src/research/universe_snapshot_service.py`", text)


if __name__ == "__main__":
    unittest.main()
