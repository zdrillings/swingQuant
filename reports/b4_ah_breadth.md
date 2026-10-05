# B4 AH Breadth Feature Audit

- idea: market-level after-hours breadth joined to every snapshot row by date
- features: ah_breadth_pct_pos, ah_breadth_zscore_5d
- data_access: DuckDB read_only=True; no data/ mutation
- target_column: alpha_vs_sector_60d
- eligible_rows: 344914
- eligible_dates: 831
- ah_feature_dates: 56
- matured_rows_with_any_b4_feature: 1552
- matured_dates_with_any_b4_feature: 4
- min_feature_ic: 0.0300
- min_feature_ic_observation_fraction: 0.2000
- screen_verdict: failed
- surviving_features_by_mean_abs_ic: none

## Fold-Local IC Summary

| feature | folds | mean_ic | mean_abs_ic | max_abs_ic | clear_folds | clear_rate |
|---|---:|---:|---:|---:|---:|---:|
| ah_breadth_pct_pos | 26 | n/a | n/a | n/a | 0 | +0.0000 |
| ah_breadth_pct_pos__rank_all | 26 | n/a | n/a | n/a | 0 | +0.0000 |
| ah_breadth_pct_pos__rank_sector | 26 | n/a | n/a | n/a | 0 | +0.0000 |
| ah_breadth_zscore_5d | 26 | n/a | n/a | n/a | 0 | +0.0000 |
| ah_breadth_zscore_5d__rank_all | 26 | n/a | n/a | n/a | 0 | +0.0000 |
| ah_breadth_zscore_5d__rank_sector | 26 | n/a | n/a | n/a | 0 | +0.0000 |

## Fold Coverage And IC

| fold_start | feature | train_rows | covered_rows_any_b4 | covered_dates_any_b4 | coverage_rate_any_b4 | observations | min_observations | ic | cleared |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 2024-04-12 | ah_breadth_pct_pos | 2418 | 0 | 0 | +0.0000 | 0 | 484 | n/a | False |
| 2024-04-12 | ah_breadth_zscore_5d | 2418 | 0 | 0 | +0.0000 | 0 | 484 | n/a | False |
| 2024-04-12 | ah_breadth_pct_pos__rank_all | 2418 | 0 | 0 | +0.0000 | 0 | 484 | n/a | False |
| 2024-04-12 | ah_breadth_zscore_5d__rank_all | 2418 | 0 | 0 | +0.0000 | 0 | 484 | n/a | False |
| 2024-04-12 | ah_breadth_pct_pos__rank_sector | 2418 | 0 | 0 | +0.0000 | 0 | 484 | n/a | False |
| 2024-04-12 | ah_breadth_zscore_5d__rank_sector | 2418 | 0 | 0 | +0.0000 | 0 | 484 | n/a | False |
| 2024-05-10 | ah_breadth_pct_pos | 2100 | 0 | 0 | +0.0000 | 0 | 420 | n/a | False |
| 2024-05-10 | ah_breadth_zscore_5d | 2100 | 0 | 0 | +0.0000 | 0 | 420 | n/a | False |
| 2024-05-10 | ah_breadth_pct_pos__rank_all | 2100 | 0 | 0 | +0.0000 | 0 | 420 | n/a | False |
| 2024-05-10 | ah_breadth_zscore_5d__rank_all | 2100 | 0 | 0 | +0.0000 | 0 | 420 | n/a | False |
| 2024-05-10 | ah_breadth_pct_pos__rank_sector | 2100 | 0 | 0 | +0.0000 | 0 | 420 | n/a | False |
| 2024-05-10 | ah_breadth_zscore_5d__rank_sector | 2100 | 0 | 0 | +0.0000 | 0 | 420 | n/a | False |
| 2024-06-10 | ah_breadth_pct_pos | 1981 | 0 | 0 | +0.0000 | 0 | 397 | n/a | False |
| 2024-06-10 | ah_breadth_zscore_5d | 1981 | 0 | 0 | +0.0000 | 0 | 397 | n/a | False |
| 2024-06-10 | ah_breadth_pct_pos__rank_all | 1981 | 0 | 0 | +0.0000 | 0 | 397 | n/a | False |
| 2024-06-10 | ah_breadth_zscore_5d__rank_all | 1981 | 0 | 0 | +0.0000 | 0 | 397 | n/a | False |
| 2024-06-10 | ah_breadth_pct_pos__rank_sector | 1981 | 0 | 0 | +0.0000 | 0 | 397 | n/a | False |
| 2024-06-10 | ah_breadth_zscore_5d__rank_sector | 1981 | 0 | 0 | +0.0000 | 0 | 397 | n/a | False |
| 2024-07-10 | ah_breadth_pct_pos | 2751 | 0 | 0 | +0.0000 | 0 | 551 | n/a | False |
| 2024-07-10 | ah_breadth_zscore_5d | 2751 | 0 | 0 | +0.0000 | 0 | 551 | n/a | False |
| 2024-07-10 | ah_breadth_pct_pos__rank_all | 2751 | 0 | 0 | +0.0000 | 0 | 551 | n/a | False |
| 2024-07-10 | ah_breadth_zscore_5d__rank_all | 2751 | 0 | 0 | +0.0000 | 0 | 551 | n/a | False |
| 2024-07-10 | ah_breadth_pct_pos__rank_sector | 2751 | 0 | 0 | +0.0000 | 0 | 551 | n/a | False |
| 2024-07-10 | ah_breadth_zscore_5d__rank_sector | 2751 | 0 | 0 | +0.0000 | 0 | 551 | n/a | False |
| 2024-08-07 | ah_breadth_pct_pos | 2550 | 0 | 0 | +0.0000 | 0 | 510 | n/a | False |
| 2024-08-07 | ah_breadth_zscore_5d | 2550 | 0 | 0 | +0.0000 | 0 | 510 | n/a | False |
| 2024-08-07 | ah_breadth_pct_pos__rank_all | 2550 | 0 | 0 | +0.0000 | 0 | 510 | n/a | False |
| 2024-08-07 | ah_breadth_zscore_5d__rank_all | 2550 | 0 | 0 | +0.0000 | 0 | 510 | n/a | False |
| 2024-08-07 | ah_breadth_pct_pos__rank_sector | 2550 | 0 | 0 | +0.0000 | 0 | 510 | n/a | False |
| 2024-08-07 | ah_breadth_zscore_5d__rank_sector | 2550 | 0 | 0 | +0.0000 | 0 | 510 | n/a | False |
| 2024-09-05 | ah_breadth_pct_pos | 2239 | 0 | 0 | +0.0000 | 0 | 448 | n/a | False |
| 2024-09-05 | ah_breadth_zscore_5d | 2239 | 0 | 0 | +0.0000 | 0 | 448 | n/a | False |
| 2024-09-05 | ah_breadth_pct_pos__rank_all | 2239 | 0 | 0 | +0.0000 | 0 | 448 | n/a | False |
| 2024-09-05 | ah_breadth_zscore_5d__rank_all | 2239 | 0 | 0 | +0.0000 | 0 | 448 | n/a | False |
| 2024-09-05 | ah_breadth_pct_pos__rank_sector | 2239 | 0 | 0 | +0.0000 | 0 | 448 | n/a | False |
| 2024-09-05 | ah_breadth_zscore_5d__rank_sector | 2239 | 0 | 0 | +0.0000 | 0 | 448 | n/a | False |
| 2024-10-03 | ah_breadth_pct_pos | 3316 | 0 | 0 | +0.0000 | 0 | 664 | n/a | False |
| 2024-10-03 | ah_breadth_zscore_5d | 3316 | 0 | 0 | +0.0000 | 0 | 664 | n/a | False |
| 2024-10-03 | ah_breadth_pct_pos__rank_all | 3316 | 0 | 0 | +0.0000 | 0 | 664 | n/a | False |
| 2024-10-03 | ah_breadth_zscore_5d__rank_all | 3316 | 0 | 0 | +0.0000 | 0 | 664 | n/a | False |
| 2024-10-03 | ah_breadth_pct_pos__rank_sector | 3316 | 0 | 0 | +0.0000 | 0 | 664 | n/a | False |
| 2024-10-03 | ah_breadth_zscore_5d__rank_sector | 3316 | 0 | 0 | +0.0000 | 0 | 664 | n/a | False |
| 2024-10-31 | ah_breadth_pct_pos | 2925 | 0 | 0 | +0.0000 | 0 | 585 | n/a | False |
| 2024-10-31 | ah_breadth_zscore_5d | 2925 | 0 | 0 | +0.0000 | 0 | 585 | n/a | False |
| 2024-10-31 | ah_breadth_pct_pos__rank_all | 2925 | 0 | 0 | +0.0000 | 0 | 585 | n/a | False |
| 2024-10-31 | ah_breadth_zscore_5d__rank_all | 2925 | 0 | 0 | +0.0000 | 0 | 585 | n/a | False |
| 2024-10-31 | ah_breadth_pct_pos__rank_sector | 2925 | 0 | 0 | +0.0000 | 0 | 585 | n/a | False |
| 2024-10-31 | ah_breadth_zscore_5d__rank_sector | 2925 | 0 | 0 | +0.0000 | 0 | 585 | n/a | False |
| 2024-11-29 | ah_breadth_pct_pos | 2692 | 0 | 0 | +0.0000 | 0 | 539 | n/a | False |
| 2024-11-29 | ah_breadth_zscore_5d | 2692 | 0 | 0 | +0.0000 | 0 | 539 | n/a | False |
| 2024-11-29 | ah_breadth_pct_pos__rank_all | 2692 | 0 | 0 | +0.0000 | 0 | 539 | n/a | False |
| 2024-11-29 | ah_breadth_zscore_5d__rank_all | 2692 | 0 | 0 | +0.0000 | 0 | 539 | n/a | False |
| 2024-11-29 | ah_breadth_pct_pos__rank_sector | 2692 | 0 | 0 | +0.0000 | 0 | 539 | n/a | False |
| 2024-11-29 | ah_breadth_zscore_5d__rank_sector | 2692 | 0 | 0 | +0.0000 | 0 | 539 | n/a | False |
| 2024-12-30 | ah_breadth_pct_pos | 3994 | 0 | 0 | +0.0000 | 0 | 799 | n/a | False |
| 2024-12-30 | ah_breadth_zscore_5d | 3994 | 0 | 0 | +0.0000 | 0 | 799 | n/a | False |
| 2024-12-30 | ah_breadth_pct_pos__rank_all | 3994 | 0 | 0 | +0.0000 | 0 | 799 | n/a | False |
| 2024-12-30 | ah_breadth_zscore_5d__rank_all | 3994 | 0 | 0 | +0.0000 | 0 | 799 | n/a | False |
| 2024-12-30 | ah_breadth_pct_pos__rank_sector | 3994 | 0 | 0 | +0.0000 | 0 | 799 | n/a | False |
| 2024-12-30 | ah_breadth_zscore_5d__rank_sector | 3994 | 0 | 0 | +0.0000 | 0 | 799 | n/a | False |
| 2025-01-30 | ah_breadth_pct_pos | 3261 | 0 | 0 | +0.0000 | 0 | 653 | n/a | False |
| 2025-01-30 | ah_breadth_zscore_5d | 3261 | 0 | 0 | +0.0000 | 0 | 653 | n/a | False |
| 2025-01-30 | ah_breadth_pct_pos__rank_all | 3261 | 0 | 0 | +0.0000 | 0 | 653 | n/a | False |
| 2025-01-30 | ah_breadth_zscore_5d__rank_all | 3261 | 0 | 0 | +0.0000 | 0 | 653 | n/a | False |
| 2025-01-30 | ah_breadth_pct_pos__rank_sector | 3261 | 0 | 0 | +0.0000 | 0 | 653 | n/a | False |
| 2025-01-30 | ah_breadth_zscore_5d__rank_sector | 3261 | 0 | 0 | +0.0000 | 0 | 653 | n/a | False |
| 2025-02-28 | ah_breadth_pct_pos | 3170 | 0 | 0 | +0.0000 | 0 | 634 | n/a | False |
| 2025-02-28 | ah_breadth_zscore_5d | 3170 | 0 | 0 | +0.0000 | 0 | 634 | n/a | False |
| 2025-02-28 | ah_breadth_pct_pos__rank_all | 3170 | 0 | 0 | +0.0000 | 0 | 634 | n/a | False |
| 2025-02-28 | ah_breadth_zscore_5d__rank_all | 3170 | 0 | 0 | +0.0000 | 0 | 634 | n/a | False |
| 2025-02-28 | ah_breadth_pct_pos__rank_sector | 3170 | 0 | 0 | +0.0000 | 0 | 634 | n/a | False |
| 2025-02-28 | ah_breadth_zscore_5d__rank_sector | 3170 | 0 | 0 | +0.0000 | 0 | 634 | n/a | False |
| 2025-05-12 | ah_breadth_pct_pos | 4306 | 0 | 0 | +0.0000 | 0 | 862 | n/a | False |
| 2025-05-12 | ah_breadth_zscore_5d | 4306 | 0 | 0 | +0.0000 | 0 | 862 | n/a | False |
| 2025-05-12 | ah_breadth_pct_pos__rank_all | 4306 | 0 | 0 | +0.0000 | 0 | 862 | n/a | False |
| 2025-05-12 | ah_breadth_zscore_5d__rank_all | 4306 | 0 | 0 | +0.0000 | 0 | 862 | n/a | False |
| 2025-05-12 | ah_breadth_pct_pos__rank_sector | 4306 | 0 | 0 | +0.0000 | 0 | 862 | n/a | False |
| 2025-05-12 | ah_breadth_zscore_5d__rank_sector | 4306 | 0 | 0 | +0.0000 | 0 | 862 | n/a | False |
| 2025-06-10 | ah_breadth_pct_pos | 3644 | 0 | 0 | +0.0000 | 0 | 729 | n/a | False |
| 2025-06-10 | ah_breadth_zscore_5d | 3644 | 0 | 0 | +0.0000 | 0 | 729 | n/a | False |
| 2025-06-10 | ah_breadth_pct_pos__rank_all | 3644 | 0 | 0 | +0.0000 | 0 | 729 | n/a | False |
| 2025-06-10 | ah_breadth_zscore_5d__rank_all | 3644 | 0 | 0 | +0.0000 | 0 | 729 | n/a | False |
| 2025-06-10 | ah_breadth_pct_pos__rank_sector | 3644 | 0 | 0 | +0.0000 | 0 | 729 | n/a | False |
| 2025-06-10 | ah_breadth_zscore_5d__rank_sector | 3644 | 0 | 0 | +0.0000 | 0 | 729 | n/a | False |
| 2025-07-10 | ah_breadth_pct_pos | 3419 | 0 | 0 | +0.0000 | 0 | 684 | n/a | False |
| 2025-07-10 | ah_breadth_zscore_5d | 3419 | 0 | 0 | +0.0000 | 0 | 684 | n/a | False |
| 2025-07-10 | ah_breadth_pct_pos__rank_all | 3419 | 0 | 0 | +0.0000 | 0 | 684 | n/a | False |
| 2025-07-10 | ah_breadth_zscore_5d__rank_all | 3419 | 0 | 0 | +0.0000 | 0 | 684 | n/a | False |
| 2025-07-10 | ah_breadth_pct_pos__rank_sector | 3419 | 0 | 0 | +0.0000 | 0 | 684 | n/a | False |
| 2025-07-10 | ah_breadth_zscore_5d__rank_sector | 3419 | 0 | 0 | +0.0000 | 0 | 684 | n/a | False |
| 2025-08-07 | ah_breadth_pct_pos | 4374 | 0 | 0 | +0.0000 | 0 | 875 | n/a | False |
| 2025-08-07 | ah_breadth_zscore_5d | 4374 | 0 | 0 | +0.0000 | 0 | 875 | n/a | False |
| 2025-08-07 | ah_breadth_pct_pos__rank_all | 4374 | 0 | 0 | +0.0000 | 0 | 875 | n/a | False |
| 2025-08-07 | ah_breadth_zscore_5d__rank_all | 4374 | 0 | 0 | +0.0000 | 0 | 875 | n/a | False |
| 2025-08-07 | ah_breadth_pct_pos__rank_sector | 4374 | 0 | 0 | +0.0000 | 0 | 875 | n/a | False |
| 2025-08-07 | ah_breadth_zscore_5d__rank_sector | 4374 | 0 | 0 | +0.0000 | 0 | 875 | n/a | False |
| 2025-09-05 | ah_breadth_pct_pos | 4163 | 0 | 0 | +0.0000 | 0 | 833 | n/a | False |
| 2025-09-05 | ah_breadth_zscore_5d | 4163 | 0 | 0 | +0.0000 | 0 | 833 | n/a | False |
| 2025-09-05 | ah_breadth_pct_pos__rank_all | 4163 | 0 | 0 | +0.0000 | 0 | 833 | n/a | False |
| 2025-09-05 | ah_breadth_zscore_5d__rank_all | 4163 | 0 | 0 | +0.0000 | 0 | 833 | n/a | False |
| 2025-09-05 | ah_breadth_pct_pos__rank_sector | 4163 | 0 | 0 | +0.0000 | 0 | 833 | n/a | False |
| 2025-09-05 | ah_breadth_zscore_5d__rank_sector | 4163 | 0 | 0 | +0.0000 | 0 | 833 | n/a | False |
| 2025-10-03 | ah_breadth_pct_pos | 3784 | 0 | 0 | +0.0000 | 0 | 757 | n/a | False |
| 2025-10-03 | ah_breadth_zscore_5d | 3784 | 0 | 0 | +0.0000 | 0 | 757 | n/a | False |
| 2025-10-03 | ah_breadth_pct_pos__rank_all | 3784 | 0 | 0 | +0.0000 | 0 | 757 | n/a | False |
| 2025-10-03 | ah_breadth_zscore_5d__rank_all | 3784 | 0 | 0 | +0.0000 | 0 | 757 | n/a | False |
| 2025-10-03 | ah_breadth_pct_pos__rank_sector | 3784 | 0 | 0 | +0.0000 | 0 | 757 | n/a | False |
| 2025-10-03 | ah_breadth_zscore_5d__rank_sector | 3784 | 0 | 0 | +0.0000 | 0 | 757 | n/a | False |
| 2025-10-31 | ah_breadth_pct_pos | 4852 | 0 | 0 | +0.0000 | 0 | 971 | n/a | False |
| 2025-10-31 | ah_breadth_zscore_5d | 4852 | 0 | 0 | +0.0000 | 0 | 971 | n/a | False |
| 2025-10-31 | ah_breadth_pct_pos__rank_all | 4852 | 0 | 0 | +0.0000 | 0 | 971 | n/a | False |
| 2025-10-31 | ah_breadth_zscore_5d__rank_all | 4852 | 0 | 0 | +0.0000 | 0 | 971 | n/a | False |
| 2025-10-31 | ah_breadth_pct_pos__rank_sector | 4852 | 0 | 0 | +0.0000 | 0 | 971 | n/a | False |
| 2025-10-31 | ah_breadth_zscore_5d__rank_sector | 4852 | 0 | 0 | +0.0000 | 0 | 971 | n/a | False |
| 2025-12-01 | ah_breadth_pct_pos | 4586 | 0 | 0 | +0.0000 | 0 | 918 | n/a | False |
| 2025-12-01 | ah_breadth_zscore_5d | 4586 | 0 | 0 | +0.0000 | 0 | 918 | n/a | False |
| 2025-12-01 | ah_breadth_pct_pos__rank_all | 4586 | 0 | 0 | +0.0000 | 0 | 918 | n/a | False |
| 2025-12-01 | ah_breadth_zscore_5d__rank_all | 4586 | 0 | 0 | +0.0000 | 0 | 918 | n/a | False |
| 2025-12-01 | ah_breadth_pct_pos__rank_sector | 4586 | 0 | 0 | +0.0000 | 0 | 918 | n/a | False |
| 2025-12-01 | ah_breadth_zscore_5d__rank_sector | 4586 | 0 | 0 | +0.0000 | 0 | 918 | n/a | False |
| 2025-12-30 | ah_breadth_pct_pos | 4392 | 0 | 0 | +0.0000 | 0 | 879 | n/a | False |
| 2025-12-30 | ah_breadth_zscore_5d | 4392 | 0 | 0 | +0.0000 | 0 | 879 | n/a | False |
| 2025-12-30 | ah_breadth_pct_pos__rank_all | 4392 | 0 | 0 | +0.0000 | 0 | 879 | n/a | False |
| 2025-12-30 | ah_breadth_zscore_5d__rank_all | 4392 | 0 | 0 | +0.0000 | 0 | 879 | n/a | False |
| 2025-12-30 | ah_breadth_pct_pos__rank_sector | 4392 | 0 | 0 | +0.0000 | 0 | 879 | n/a | False |
| 2025-12-30 | ah_breadth_zscore_5d__rank_sector | 4392 | 0 | 0 | +0.0000 | 0 | 879 | n/a | False |
| 2026-01-29 | ah_breadth_pct_pos | 5138 | 0 | 0 | +0.0000 | 0 | 1028 | n/a | False |
| 2026-01-29 | ah_breadth_zscore_5d | 5138 | 0 | 0 | +0.0000 | 0 | 1028 | n/a | False |
| 2026-01-29 | ah_breadth_pct_pos__rank_all | 5138 | 0 | 0 | +0.0000 | 0 | 1028 | n/a | False |
| 2026-01-29 | ah_breadth_zscore_5d__rank_all | 5138 | 0 | 0 | +0.0000 | 0 | 1028 | n/a | False |
| 2026-01-29 | ah_breadth_pct_pos__rank_sector | 5138 | 0 | 0 | +0.0000 | 0 | 1028 | n/a | False |
| 2026-01-29 | ah_breadth_zscore_5d__rank_sector | 5138 | 0 | 0 | +0.0000 | 0 | 1028 | n/a | False |
| 2026-02-27 | ah_breadth_pct_pos | 4876 | 0 | 0 | +0.0000 | 0 | 976 | n/a | False |
| 2026-02-27 | ah_breadth_zscore_5d | 4876 | 0 | 0 | +0.0000 | 0 | 976 | n/a | False |
| 2026-02-27 | ah_breadth_pct_pos__rank_all | 4876 | 0 | 0 | +0.0000 | 0 | 976 | n/a | False |
| 2026-02-27 | ah_breadth_zscore_5d__rank_all | 4876 | 0 | 0 | +0.0000 | 0 | 976 | n/a | False |
| 2026-02-27 | ah_breadth_pct_pos__rank_sector | 4876 | 0 | 0 | +0.0000 | 0 | 976 | n/a | False |
| 2026-02-27 | ah_breadth_zscore_5d__rank_sector | 4876 | 0 | 0 | +0.0000 | 0 | 976 | n/a | False |
| 2026-04-15 | ah_breadth_pct_pos | 4771 | 0 | 0 | +0.0000 | 0 | 955 | n/a | False |
| 2026-04-15 | ah_breadth_zscore_5d | 4771 | 0 | 0 | +0.0000 | 0 | 955 | n/a | False |
| 2026-04-15 | ah_breadth_pct_pos__rank_all | 4771 | 0 | 0 | +0.0000 | 0 | 955 | n/a | False |
| 2026-04-15 | ah_breadth_zscore_5d__rank_all | 4771 | 0 | 0 | +0.0000 | 0 | 955 | n/a | False |
| 2026-04-15 | ah_breadth_pct_pos__rank_sector | 4771 | 0 | 0 | +0.0000 | 0 | 955 | n/a | False |
| 2026-04-15 | ah_breadth_zscore_5d__rank_sector | 4771 | 0 | 0 | +0.0000 | 0 | 955 | n/a | False |
| 2026-05-13 | ah_breadth_pct_pos | 5652 | 0 | 0 | +0.0000 | 0 | 1131 | n/a | False |
| 2026-05-13 | ah_breadth_zscore_5d | 5652 | 0 | 0 | +0.0000 | 0 | 1131 | n/a | False |
| 2026-05-13 | ah_breadth_pct_pos__rank_all | 5652 | 0 | 0 | +0.0000 | 0 | 1131 | n/a | False |
| 2026-05-13 | ah_breadth_zscore_5d__rank_all | 5652 | 0 | 0 | +0.0000 | 0 | 1131 | n/a | False |
| 2026-05-13 | ah_breadth_pct_pos__rank_sector | 5652 | 0 | 0 | +0.0000 | 0 | 1131 | n/a | False |
| 2026-05-13 | ah_breadth_zscore_5d__rank_sector | 5652 | 0 | 0 | +0.0000 | 0 | 1131 | n/a | False |
| 2026-06-11 | ah_breadth_pct_pos | 5517 | 0 | 0 | +0.0000 | 0 | 1104 | n/a | False |
| 2026-06-11 | ah_breadth_zscore_5d | 5517 | 0 | 0 | +0.0000 | 0 | 1104 | n/a | False |
| 2026-06-11 | ah_breadth_pct_pos__rank_all | 5517 | 0 | 0 | +0.0000 | 0 | 1104 | n/a | False |
| 2026-06-11 | ah_breadth_zscore_5d__rank_all | 5517 | 0 | 0 | +0.0000 | 0 | 1104 | n/a | False |
| 2026-06-11 | ah_breadth_pct_pos__rank_sector | 5517 | 0 | 0 | +0.0000 | 0 | 1104 | n/a | False |
| 2026-06-11 | ah_breadth_zscore_5d__rank_sector | 5517 | 0 | 0 | +0.0000 | 0 | 1104 | n/a | False |

## Honest-Grid With/Without Bakeoff On AH-Covered Dates

Skipped: no B4 feature reached mean_abs_ic >= min_feature_ic with the configured observation floor.
