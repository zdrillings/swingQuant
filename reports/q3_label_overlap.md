# Q3 Label Overlap Audit

- target_column: alpha_vs_sector_60d
- horizon_days: 60
- min_train_dates: 252
- max_train_dates: 252
- test_window_dates: 20
- oos_evaluation_stride_dates: 20
- raw_label_date_range: 2021-05-10 -> 2026-07-09
- label_construction: `src/research/universe_snapshot_service.py` computes `alpha_vs_sector_60d` as ticker 60-session forward return minus sector ETF 60-session forward return.
- walk_forward_sampler: `src/research/shortlist_model_service.py` applies a 60-session train/test label embargo, then `_stride_training_labels(..., label_horizon_dates=60)` within each fold.
- label_window_definition: sessions `t+1` through `t+horizon`; consecutive labels overlap when trading-day gap < horizon

VERDICT raw labels: labels overlap; walk-forward training rows: clean; OOS grid: labels overlap.

## Summary

| sample | rows | tickers | dates | row_overlap_share | adjacent_pair_overlap_share | max_overlap_days | median_overlap_days | p90_overlap_days | median_gap_days | independent_rows | independent_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| raw_labeled_snapshot_rows | 1079819 | 1279 | 882 | 1.0000 | 1.0000 | 59 | 59.0 | 59.0 | 1.0 | 18447 | 0.0171 |
| walk_forward_training_rows_per_fold | 177636 | 1244 | 41 | 0.0000 | 0.0000 | 0 | 0.0 | 0.0 | 60.0 | 177636 | 1.0000 |
| persisted_oos_prediction_grid | 219983 | 1244 | 519 | 0.9996 | 0.9937 | 59 | 59.0 | 59.0 | 1.0 | 7889 | 0.0359 |

## Interpretation

- `raw_labeled_snapshot_rows` audits the stored label table directly; daily stored labels are expected to overlap unless a downstream sampler removes them.
- `walk_forward_training_rows_per_fold` audits the model's training sampler inside each fold after the 60-session embargo and horizon stride.
- `persisted_oos_prediction_grid` audits the latest persisted OOS prediction artifact; dense 20-session test blocks overlap for a 60-session forward label.
- `independent_rows` is a greedy non-overlapping count per ticker using the same 60-session forward window.
