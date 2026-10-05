# D2 Revision Acceleration Audit

- idea: 14-session change in analyst EPS revision breadth and EPS estimate dispersion
- data_access: DuckDB read_only=True; no data/ mutation
- target_column: alpha_vs_sector_60d
- lag_policy: latest analyst revision snapshot at or before snapshot_date minus 14 business sessions
- dispersion_source: earnings_estimate_json high/low/avg, priority period 0q then +1q then 0y then +1y
- eligible_rows: 344914
- eligible_dates: 831
- eligible_rows_with_any_d2_feature: 0
- first_eligible_d2_feature_date: n/a
- analyst_revision_history_start: 2026-06-23
- first_possible_14_session_delta_date: 2026-07-13
- latest_matured_eligible_date: 2026-07-09
- min_feature_ic: 0.0300
- min_feature_ic_observation_fraction: 0.2000
- screen_verdict: insufficient_matured_coverage
- surviving_features_by_mean_abs_ic: none

## Fold-Local IC Summary

| feature | folds | mean_ic | mean_abs_ic | max_abs_ic | clear_folds | clear_rate |
|---|---:|---:|---:|---:|---:|---:|
| analyst_eps_estimate_dispersion_change_14d | 26 | n/a | n/a | n/a | 0 | +0.0000 |
| analyst_eps_estimate_dispersion_change_14d__rank_all | 26 | n/a | n/a | n/a | 0 | +0.0000 |
| analyst_eps_estimate_dispersion_change_14d__rank_sector | 26 | n/a | n/a | n/a | 0 | +0.0000 |
| analyst_eps_revision_breadth_change_14d | 26 | n/a | n/a | n/a | 0 | +0.0000 |
| analyst_eps_revision_breadth_change_14d__rank_all | 26 | n/a | n/a | n/a | 0 | +0.0000 |
| analyst_eps_revision_breadth_change_14d__rank_sector | 26 | n/a | n/a | n/a | 0 | +0.0000 |

## Fold Details

| fold_start | feature | observations | min_observations | ic | cleared |
|---|---|---:|---:|---:|---|
| 2024-04-12 | analyst_eps_revision_breadth_change_14d | 0 | 484 | n/a | False |
| 2024-04-12 | analyst_eps_estimate_dispersion_change_14d | 0 | 484 | n/a | False |
| 2024-04-12 | analyst_eps_revision_breadth_change_14d__rank_all | 0 | 484 | n/a | False |
| 2024-04-12 | analyst_eps_estimate_dispersion_change_14d__rank_all | 0 | 484 | n/a | False |
| 2024-04-12 | analyst_eps_revision_breadth_change_14d__rank_sector | 0 | 484 | n/a | False |
| 2024-04-12 | analyst_eps_estimate_dispersion_change_14d__rank_sector | 0 | 484 | n/a | False |
| 2024-05-10 | analyst_eps_revision_breadth_change_14d | 0 | 420 | n/a | False |
| 2024-05-10 | analyst_eps_estimate_dispersion_change_14d | 0 | 420 | n/a | False |
| 2024-05-10 | analyst_eps_revision_breadth_change_14d__rank_all | 0 | 420 | n/a | False |
| 2024-05-10 | analyst_eps_estimate_dispersion_change_14d__rank_all | 0 | 420 | n/a | False |
| 2024-05-10 | analyst_eps_revision_breadth_change_14d__rank_sector | 0 | 420 | n/a | False |
| 2024-05-10 | analyst_eps_estimate_dispersion_change_14d__rank_sector | 0 | 420 | n/a | False |
| 2024-06-10 | analyst_eps_revision_breadth_change_14d | 0 | 397 | n/a | False |
| 2024-06-10 | analyst_eps_estimate_dispersion_change_14d | 0 | 397 | n/a | False |
| 2024-06-10 | analyst_eps_revision_breadth_change_14d__rank_all | 0 | 397 | n/a | False |
| 2024-06-10 | analyst_eps_estimate_dispersion_change_14d__rank_all | 0 | 397 | n/a | False |
| 2024-06-10 | analyst_eps_revision_breadth_change_14d__rank_sector | 0 | 397 | n/a | False |
| 2024-06-10 | analyst_eps_estimate_dispersion_change_14d__rank_sector | 0 | 397 | n/a | False |
| 2024-07-10 | analyst_eps_revision_breadth_change_14d | 0 | 551 | n/a | False |
| 2024-07-10 | analyst_eps_estimate_dispersion_change_14d | 0 | 551 | n/a | False |
| 2024-07-10 | analyst_eps_revision_breadth_change_14d__rank_all | 0 | 551 | n/a | False |
| 2024-07-10 | analyst_eps_estimate_dispersion_change_14d__rank_all | 0 | 551 | n/a | False |
| 2024-07-10 | analyst_eps_revision_breadth_change_14d__rank_sector | 0 | 551 | n/a | False |
| 2024-07-10 | analyst_eps_estimate_dispersion_change_14d__rank_sector | 0 | 551 | n/a | False |
| 2024-08-07 | analyst_eps_revision_breadth_change_14d | 0 | 510 | n/a | False |
| 2024-08-07 | analyst_eps_estimate_dispersion_change_14d | 0 | 510 | n/a | False |
| 2024-08-07 | analyst_eps_revision_breadth_change_14d__rank_all | 0 | 510 | n/a | False |
| 2024-08-07 | analyst_eps_estimate_dispersion_change_14d__rank_all | 0 | 510 | n/a | False |
| 2024-08-07 | analyst_eps_revision_breadth_change_14d__rank_sector | 0 | 510 | n/a | False |
| 2024-08-07 | analyst_eps_estimate_dispersion_change_14d__rank_sector | 0 | 510 | n/a | False |
| 2024-09-05 | analyst_eps_revision_breadth_change_14d | 0 | 448 | n/a | False |
| 2024-09-05 | analyst_eps_estimate_dispersion_change_14d | 0 | 448 | n/a | False |
| 2024-09-05 | analyst_eps_revision_breadth_change_14d__rank_all | 0 | 448 | n/a | False |
| 2024-09-05 | analyst_eps_estimate_dispersion_change_14d__rank_all | 0 | 448 | n/a | False |
| 2024-09-05 | analyst_eps_revision_breadth_change_14d__rank_sector | 0 | 448 | n/a | False |
| 2024-09-05 | analyst_eps_estimate_dispersion_change_14d__rank_sector | 0 | 448 | n/a | False |
| 2024-10-03 | analyst_eps_revision_breadth_change_14d | 0 | 664 | n/a | False |
| 2024-10-03 | analyst_eps_estimate_dispersion_change_14d | 0 | 664 | n/a | False |
| 2024-10-03 | analyst_eps_revision_breadth_change_14d__rank_all | 0 | 664 | n/a | False |
| 2024-10-03 | analyst_eps_estimate_dispersion_change_14d__rank_all | 0 | 664 | n/a | False |
| 2024-10-03 | analyst_eps_revision_breadth_change_14d__rank_sector | 0 | 664 | n/a | False |
| 2024-10-03 | analyst_eps_estimate_dispersion_change_14d__rank_sector | 0 | 664 | n/a | False |
| 2024-10-31 | analyst_eps_revision_breadth_change_14d | 0 | 585 | n/a | False |
| 2024-10-31 | analyst_eps_estimate_dispersion_change_14d | 0 | 585 | n/a | False |
| 2024-10-31 | analyst_eps_revision_breadth_change_14d__rank_all | 0 | 585 | n/a | False |
| 2024-10-31 | analyst_eps_estimate_dispersion_change_14d__rank_all | 0 | 585 | n/a | False |
| 2024-10-31 | analyst_eps_revision_breadth_change_14d__rank_sector | 0 | 585 | n/a | False |
| 2024-10-31 | analyst_eps_estimate_dispersion_change_14d__rank_sector | 0 | 585 | n/a | False |
| 2024-11-29 | analyst_eps_revision_breadth_change_14d | 0 | 539 | n/a | False |
| 2024-11-29 | analyst_eps_estimate_dispersion_change_14d | 0 | 539 | n/a | False |
| 2024-11-29 | analyst_eps_revision_breadth_change_14d__rank_all | 0 | 539 | n/a | False |
| 2024-11-29 | analyst_eps_estimate_dispersion_change_14d__rank_all | 0 | 539 | n/a | False |
| 2024-11-29 | analyst_eps_revision_breadth_change_14d__rank_sector | 0 | 539 | n/a | False |
| 2024-11-29 | analyst_eps_estimate_dispersion_change_14d__rank_sector | 0 | 539 | n/a | False |
| 2024-12-30 | analyst_eps_revision_breadth_change_14d | 0 | 799 | n/a | False |
| 2024-12-30 | analyst_eps_estimate_dispersion_change_14d | 0 | 799 | n/a | False |
| 2024-12-30 | analyst_eps_revision_breadth_change_14d__rank_all | 0 | 799 | n/a | False |
| 2024-12-30 | analyst_eps_estimate_dispersion_change_14d__rank_all | 0 | 799 | n/a | False |
| 2024-12-30 | analyst_eps_revision_breadth_change_14d__rank_sector | 0 | 799 | n/a | False |
| 2024-12-30 | analyst_eps_estimate_dispersion_change_14d__rank_sector | 0 | 799 | n/a | False |
| 2025-01-30 | analyst_eps_revision_breadth_change_14d | 0 | 653 | n/a | False |
| 2025-01-30 | analyst_eps_estimate_dispersion_change_14d | 0 | 653 | n/a | False |
| 2025-01-30 | analyst_eps_revision_breadth_change_14d__rank_all | 0 | 653 | n/a | False |
| 2025-01-30 | analyst_eps_estimate_dispersion_change_14d__rank_all | 0 | 653 | n/a | False |
| 2025-01-30 | analyst_eps_revision_breadth_change_14d__rank_sector | 0 | 653 | n/a | False |
| 2025-01-30 | analyst_eps_estimate_dispersion_change_14d__rank_sector | 0 | 653 | n/a | False |
| 2025-02-28 | analyst_eps_revision_breadth_change_14d | 0 | 634 | n/a | False |
| 2025-02-28 | analyst_eps_estimate_dispersion_change_14d | 0 | 634 | n/a | False |
| 2025-02-28 | analyst_eps_revision_breadth_change_14d__rank_all | 0 | 634 | n/a | False |
| 2025-02-28 | analyst_eps_estimate_dispersion_change_14d__rank_all | 0 | 634 | n/a | False |
| 2025-02-28 | analyst_eps_revision_breadth_change_14d__rank_sector | 0 | 634 | n/a | False |
| 2025-02-28 | analyst_eps_estimate_dispersion_change_14d__rank_sector | 0 | 634 | n/a | False |
| 2025-05-12 | analyst_eps_revision_breadth_change_14d | 0 | 862 | n/a | False |
| 2025-05-12 | analyst_eps_estimate_dispersion_change_14d | 0 | 862 | n/a | False |
| 2025-05-12 | analyst_eps_revision_breadth_change_14d__rank_all | 0 | 862 | n/a | False |
| 2025-05-12 | analyst_eps_estimate_dispersion_change_14d__rank_all | 0 | 862 | n/a | False |
| 2025-05-12 | analyst_eps_revision_breadth_change_14d__rank_sector | 0 | 862 | n/a | False |
| 2025-05-12 | analyst_eps_estimate_dispersion_change_14d__rank_sector | 0 | 862 | n/a | False |
| 2025-06-10 | analyst_eps_revision_breadth_change_14d | 0 | 729 | n/a | False |
| 2025-06-10 | analyst_eps_estimate_dispersion_change_14d | 0 | 729 | n/a | False |
| 2025-06-10 | analyst_eps_revision_breadth_change_14d__rank_all | 0 | 729 | n/a | False |
| 2025-06-10 | analyst_eps_estimate_dispersion_change_14d__rank_all | 0 | 729 | n/a | False |
| 2025-06-10 | analyst_eps_revision_breadth_change_14d__rank_sector | 0 | 729 | n/a | False |
| 2025-06-10 | analyst_eps_estimate_dispersion_change_14d__rank_sector | 0 | 729 | n/a | False |
| 2025-07-10 | analyst_eps_revision_breadth_change_14d | 0 | 684 | n/a | False |
| 2025-07-10 | analyst_eps_estimate_dispersion_change_14d | 0 | 684 | n/a | False |
| 2025-07-10 | analyst_eps_revision_breadth_change_14d__rank_all | 0 | 684 | n/a | False |
| 2025-07-10 | analyst_eps_estimate_dispersion_change_14d__rank_all | 0 | 684 | n/a | False |
| 2025-07-10 | analyst_eps_revision_breadth_change_14d__rank_sector | 0 | 684 | n/a | False |
| 2025-07-10 | analyst_eps_estimate_dispersion_change_14d__rank_sector | 0 | 684 | n/a | False |
| 2025-08-07 | analyst_eps_revision_breadth_change_14d | 0 | 875 | n/a | False |
| 2025-08-07 | analyst_eps_estimate_dispersion_change_14d | 0 | 875 | n/a | False |
| 2025-08-07 | analyst_eps_revision_breadth_change_14d__rank_all | 0 | 875 | n/a | False |
| 2025-08-07 | analyst_eps_estimate_dispersion_change_14d__rank_all | 0 | 875 | n/a | False |
| 2025-08-07 | analyst_eps_revision_breadth_change_14d__rank_sector | 0 | 875 | n/a | False |
| 2025-08-07 | analyst_eps_estimate_dispersion_change_14d__rank_sector | 0 | 875 | n/a | False |
| 2025-09-05 | analyst_eps_revision_breadth_change_14d | 0 | 833 | n/a | False |
| 2025-09-05 | analyst_eps_estimate_dispersion_change_14d | 0 | 833 | n/a | False |
| 2025-09-05 | analyst_eps_revision_breadth_change_14d__rank_all | 0 | 833 | n/a | False |
| 2025-09-05 | analyst_eps_estimate_dispersion_change_14d__rank_all | 0 | 833 | n/a | False |
| 2025-09-05 | analyst_eps_revision_breadth_change_14d__rank_sector | 0 | 833 | n/a | False |
| 2025-09-05 | analyst_eps_estimate_dispersion_change_14d__rank_sector | 0 | 833 | n/a | False |
| 2025-10-03 | analyst_eps_revision_breadth_change_14d | 0 | 757 | n/a | False |
| 2025-10-03 | analyst_eps_estimate_dispersion_change_14d | 0 | 757 | n/a | False |
| 2025-10-03 | analyst_eps_revision_breadth_change_14d__rank_all | 0 | 757 | n/a | False |
| 2025-10-03 | analyst_eps_estimate_dispersion_change_14d__rank_all | 0 | 757 | n/a | False |
| 2025-10-03 | analyst_eps_revision_breadth_change_14d__rank_sector | 0 | 757 | n/a | False |
| 2025-10-03 | analyst_eps_estimate_dispersion_change_14d__rank_sector | 0 | 757 | n/a | False |
| 2025-10-31 | analyst_eps_revision_breadth_change_14d | 0 | 971 | n/a | False |
| 2025-10-31 | analyst_eps_estimate_dispersion_change_14d | 0 | 971 | n/a | False |
| 2025-10-31 | analyst_eps_revision_breadth_change_14d__rank_all | 0 | 971 | n/a | False |
| 2025-10-31 | analyst_eps_estimate_dispersion_change_14d__rank_all | 0 | 971 | n/a | False |
| 2025-10-31 | analyst_eps_revision_breadth_change_14d__rank_sector | 0 | 971 | n/a | False |
| 2025-10-31 | analyst_eps_estimate_dispersion_change_14d__rank_sector | 0 | 971 | n/a | False |
| 2025-12-01 | analyst_eps_revision_breadth_change_14d | 0 | 918 | n/a | False |
| 2025-12-01 | analyst_eps_estimate_dispersion_change_14d | 0 | 918 | n/a | False |
| 2025-12-01 | analyst_eps_revision_breadth_change_14d__rank_all | 0 | 918 | n/a | False |
| 2025-12-01 | analyst_eps_estimate_dispersion_change_14d__rank_all | 0 | 918 | n/a | False |
| 2025-12-01 | analyst_eps_revision_breadth_change_14d__rank_sector | 0 | 918 | n/a | False |
| 2025-12-01 | analyst_eps_estimate_dispersion_change_14d__rank_sector | 0 | 918 | n/a | False |
| 2025-12-30 | analyst_eps_revision_breadth_change_14d | 0 | 879 | n/a | False |
| 2025-12-30 | analyst_eps_estimate_dispersion_change_14d | 0 | 879 | n/a | False |
| 2025-12-30 | analyst_eps_revision_breadth_change_14d__rank_all | 0 | 879 | n/a | False |
| 2025-12-30 | analyst_eps_estimate_dispersion_change_14d__rank_all | 0 | 879 | n/a | False |
| 2025-12-30 | analyst_eps_revision_breadth_change_14d__rank_sector | 0 | 879 | n/a | False |
| 2025-12-30 | analyst_eps_estimate_dispersion_change_14d__rank_sector | 0 | 879 | n/a | False |
| 2026-01-29 | analyst_eps_revision_breadth_change_14d | 0 | 1028 | n/a | False |
| 2026-01-29 | analyst_eps_estimate_dispersion_change_14d | 0 | 1028 | n/a | False |
| 2026-01-29 | analyst_eps_revision_breadth_change_14d__rank_all | 0 | 1028 | n/a | False |
| 2026-01-29 | analyst_eps_estimate_dispersion_change_14d__rank_all | 0 | 1028 | n/a | False |
| 2026-01-29 | analyst_eps_revision_breadth_change_14d__rank_sector | 0 | 1028 | n/a | False |
| 2026-01-29 | analyst_eps_estimate_dispersion_change_14d__rank_sector | 0 | 1028 | n/a | False |
| 2026-02-27 | analyst_eps_revision_breadth_change_14d | 0 | 976 | n/a | False |
| 2026-02-27 | analyst_eps_estimate_dispersion_change_14d | 0 | 976 | n/a | False |
| 2026-02-27 | analyst_eps_revision_breadth_change_14d__rank_all | 0 | 976 | n/a | False |
| 2026-02-27 | analyst_eps_estimate_dispersion_change_14d__rank_all | 0 | 976 | n/a | False |
| 2026-02-27 | analyst_eps_revision_breadth_change_14d__rank_sector | 0 | 976 | n/a | False |
| 2026-02-27 | analyst_eps_estimate_dispersion_change_14d__rank_sector | 0 | 976 | n/a | False |
| 2026-04-15 | analyst_eps_revision_breadth_change_14d | 0 | 955 | n/a | False |
| 2026-04-15 | analyst_eps_estimate_dispersion_change_14d | 0 | 955 | n/a | False |
| 2026-04-15 | analyst_eps_revision_breadth_change_14d__rank_all | 0 | 955 | n/a | False |
| 2026-04-15 | analyst_eps_estimate_dispersion_change_14d__rank_all | 0 | 955 | n/a | False |
| 2026-04-15 | analyst_eps_revision_breadth_change_14d__rank_sector | 0 | 955 | n/a | False |
| 2026-04-15 | analyst_eps_estimate_dispersion_change_14d__rank_sector | 0 | 955 | n/a | False |
| 2026-05-13 | analyst_eps_revision_breadth_change_14d | 0 | 1131 | n/a | False |
| 2026-05-13 | analyst_eps_estimate_dispersion_change_14d | 0 | 1131 | n/a | False |
| 2026-05-13 | analyst_eps_revision_breadth_change_14d__rank_all | 0 | 1131 | n/a | False |
| 2026-05-13 | analyst_eps_estimate_dispersion_change_14d__rank_all | 0 | 1131 | n/a | False |
| 2026-05-13 | analyst_eps_revision_breadth_change_14d__rank_sector | 0 | 1131 | n/a | False |
| 2026-05-13 | analyst_eps_estimate_dispersion_change_14d__rank_sector | 0 | 1131 | n/a | False |
| 2026-06-11 | analyst_eps_revision_breadth_change_14d | 0 | 1104 | n/a | False |
| 2026-06-11 | analyst_eps_estimate_dispersion_change_14d | 0 | 1104 | n/a | False |
| 2026-06-11 | analyst_eps_revision_breadth_change_14d__rank_all | 0 | 1104 | n/a | False |
| 2026-06-11 | analyst_eps_estimate_dispersion_change_14d__rank_all | 0 | 1104 | n/a | False |
| 2026-06-11 | analyst_eps_revision_breadth_change_14d__rank_sector | 0 | 1104 | n/a | False |
| 2026-06-11 | analyst_eps_estimate_dispersion_change_14d__rank_sector | 0 | 1104 | n/a | False |

## Honest-Grid With/Without Bakeoff

Skipped: the current fold-local training windows have no D2 observations above the configured 20% observation floor. Re-run after enough post-2026-06-23 analyst history has matured into 60d labels.
