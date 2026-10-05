# D3 Failed-Breakout Feature Audit

- idea: recent failed breakout above the prior 20d/52w high, then close back inside within 10 sessions
- data_access: DuckDB read_only=True; no data/ mutation
- target_column: alpha_vs_sector_60d
- eligible_rows: 344914
- eligible_dates: 831
- min_feature_ic: 0.0300
- min_feature_ic_observation_fraction: 0.2000
- screen_verdict: survived
- surviving_features_by_mean_abs_ic: failed_breakout_20d, failed_breakout_52w, failed_breakout_20d__rank_all, failed_breakout_52w__rank_all, days_since_failed_breakout_20d, failed_breakout_52w__rank_sector

## Fold-Local IC Summary

| feature | folds | mean_ic | mean_abs_ic | max_abs_ic | clear_folds | clear_rate |
|---|---:|---:|---:|---:|---:|---:|
| failed_breakout_20d | 26 | -0.0404 | +0.0791 | +0.2297 | 18 | +0.6923 |
| failed_breakout_52w | 26 | +0.0340 | +0.0672 | +0.1213 | 24 | +0.9231 |
| failed_breakout_20d__rank_all | 26 | -0.0416 | +0.0538 | +0.1383 | 19 | +0.7308 |
| failed_breakout_52w__rank_all | 26 | -0.0004 | +0.0430 | +0.0915 | 18 | +0.6923 |
| days_since_failed_breakout_20d | 26 | -0.0369 | +0.0371 | +0.1516 | 9 | +0.3462 |
| failed_breakout_52w__rank_sector | 26 | +0.0310 | +0.0310 | +0.0673 | 10 | +0.3846 |
| failed_breakout_20d__rank_sector | 26 | +0.0021 | +0.0296 | +0.0686 | 13 | +0.5000 |
| days_since_failed_breakout_52w | 26 | -0.0046 | +0.0255 | +0.0834 | 7 | +0.2692 |
| days_since_failed_breakout_52w__rank_sector | 26 | +0.0151 | +0.0244 | +0.0540 | 9 | +0.3462 |
| days_since_failed_breakout_20d__rank_sector | 26 | +0.0151 | +0.0224 | +0.0468 | 9 | +0.3462 |
| days_since_failed_breakout_52w__rank_all | 26 | +0.0050 | +0.0220 | +0.0518 | 6 | +0.2308 |
| days_since_failed_breakout_20d__rank_all | 26 | +0.0086 | +0.0110 | +0.0297 | 0 | +0.0000 |

## Conditioned Beat Rates

| feature | target | cohort | rows | mean_alpha | beat_rate |
|---|---|---|---:|---:|---:|
| failed_breakout_20d | alpha_vs_sector_20d | all | 344914 | +0.0002 | +0.4791 |
| failed_breakout_20d | alpha_vs_sector_20d | flagged | 274001 | +0.0001 | +0.4800 |
| failed_breakout_20d | alpha_vs_sector_20d | unflagged | 70913 | +0.0004 | +0.4755 |
| failed_breakout_20d | alpha_vs_sector_60d | all | 344914 | -0.0137 | +0.4179 |
| failed_breakout_20d | alpha_vs_sector_60d | flagged | 274001 | -0.0146 | +0.4153 |
| failed_breakout_20d | alpha_vs_sector_60d | unflagged | 70913 | -0.0100 | +0.4278 |
| failed_breakout_52w | alpha_vs_sector_20d | all | 344914 | +0.0002 | +0.4791 |
| failed_breakout_52w | alpha_vs_sector_20d | flagged | 160865 | +0.0023 | +0.4932 |
| failed_breakout_52w | alpha_vs_sector_20d | unflagged | 184049 | -0.0017 | +0.4668 |
| failed_breakout_52w | alpha_vs_sector_60d | all | 344914 | -0.0137 | +0.4179 |
| failed_breakout_52w | alpha_vs_sector_60d | flagged | 160865 | -0.0087 | +0.4285 |
| failed_breakout_52w | alpha_vs_sector_60d | unflagged | 184049 | -0.0180 | +0.4086 |

## Fold Details

| fold_start | feature | observations | min_observations | ic | cleared |
|---|---|---:|---:|---:|---|
| 2024-04-12 | failed_breakout_20d | 2418 | 484 | -0.1770 | True |
| 2024-04-12 | days_since_failed_breakout_20d | 1830 | 484 | -0.0057 | False |
| 2024-04-12 | failed_breakout_52w | 2418 | 484 | +0.0025 | False |
| 2024-04-12 | days_since_failed_breakout_52w | 725 | 484 | +0.0457 | True |
| 2024-04-12 | failed_breakout_20d__rank_all | 2418 | 484 | -0.1383 | True |
| 2024-04-12 | days_since_failed_breakout_20d__rank_all | 1830 | 484 | +0.0178 | False |
| 2024-04-12 | failed_breakout_52w__rank_all | 2418 | 484 | +0.0320 | True |
| 2024-04-12 | days_since_failed_breakout_52w__rank_all | 725 | 484 | -0.0188 | False |
| 2024-04-12 | failed_breakout_20d__rank_sector | 2418 | 484 | -0.0686 | True |
| 2024-04-12 | days_since_failed_breakout_20d__rank_sector | 1830 | 484 | +0.0068 | False |
| 2024-04-12 | failed_breakout_52w__rank_sector | 2418 | 484 | +0.0440 | True |
| 2024-04-12 | days_since_failed_breakout_52w__rank_sector | 725 | 484 | +0.0034 | False |
| 2024-05-10 | failed_breakout_20d | 2100 | 420 | +0.0086 | False |
| 2024-05-10 | days_since_failed_breakout_20d | 1593 | 420 | -0.1516 | True |
| 2024-05-10 | failed_breakout_52w | 2100 | 420 | +0.1213 | True |
| 2024-05-10 | days_since_failed_breakout_52w | 777 | 420 | -0.0834 | True |
| 2024-05-10 | failed_breakout_20d__rank_all | 2100 | 420 | -0.1016 | True |
| 2024-05-10 | days_since_failed_breakout_20d__rank_all | 1593 | 420 | +0.0062 | False |
| 2024-05-10 | failed_breakout_52w__rank_all | 2100 | 420 | -0.0839 | True |
| 2024-05-10 | days_since_failed_breakout_52w__rank_all | 777 | 420 | +0.0272 | False |
| 2024-05-10 | failed_breakout_20d__rank_sector | 2100 | 420 | -0.0065 | False |
| 2024-05-10 | days_since_failed_breakout_20d__rank_sector | 1593 | 420 | +0.0185 | False |
| 2024-05-10 | failed_breakout_52w__rank_sector | 2100 | 420 | +0.0088 | False |
| 2024-05-10 | days_since_failed_breakout_52w__rank_sector | 777 | 420 | +0.0393 | True |
| 2024-06-10 | failed_breakout_20d | 1981 | 397 | +0.0490 | True |
| 2024-06-10 | days_since_failed_breakout_20d | 1502 | 397 | -0.0026 | False |
| 2024-06-10 | failed_breakout_52w | 1981 | 397 | +0.0859 | True |
| 2024-06-10 | days_since_failed_breakout_52w | 831 | 397 | +0.0129 | False |
| 2024-06-10 | failed_breakout_20d__rank_all | 1981 | 397 | +0.0057 | False |
| 2024-06-10 | days_since_failed_breakout_20d__rank_all | 1502 | 397 | -0.0013 | False |
| 2024-06-10 | failed_breakout_52w__rank_all | 1981 | 397 | +0.0796 | True |
| 2024-06-10 | days_since_failed_breakout_52w__rank_all | 831 | 397 | -0.0170 | False |
| 2024-06-10 | failed_breakout_20d__rank_sector | 1981 | 397 | +0.0309 | True |
| 2024-06-10 | days_since_failed_breakout_20d__rank_sector | 1502 | 397 | +0.0023 | False |
| 2024-06-10 | failed_breakout_52w__rank_sector | 1981 | 397 | +0.0673 | True |
| 2024-06-10 | days_since_failed_breakout_52w__rank_sector | 831 | 397 | -0.0244 | False |
| 2024-07-10 | failed_breakout_20d | 2751 | 551 | -0.1769 | True |
| 2024-07-10 | days_since_failed_breakout_20d | 2106 | 551 | +0.0008 | False |
| 2024-07-10 | failed_breakout_52w | 2751 | 551 | -0.0435 | True |
| 2024-07-10 | days_since_failed_breakout_52w | 952 | 551 | +0.0467 | True |
| 2024-07-10 | failed_breakout_20d__rank_all | 2751 | 551 | -0.1377 | True |
| 2024-07-10 | days_since_failed_breakout_20d__rank_all | 2106 | 551 | +0.0157 | False |
| 2024-07-10 | failed_breakout_52w__rank_all | 2751 | 551 | +0.0156 | False |
| 2024-07-10 | days_since_failed_breakout_52w__rank_all | 952 | 551 | -0.0009 | False |
| 2024-07-10 | failed_breakout_20d__rank_sector | 2751 | 551 | -0.0558 | True |
| 2024-07-10 | days_since_failed_breakout_20d__rank_sector | 2106 | 551 | -0.0047 | False |
| 2024-07-10 | failed_breakout_52w__rank_sector | 2751 | 551 | +0.0197 | False |
| 2024-07-10 | days_since_failed_breakout_52w__rank_sector | 952 | 551 | -0.0003 | False |
| 2024-08-07 | failed_breakout_20d | 2550 | 510 | +0.0179 | False |
| 2024-08-07 | days_since_failed_breakout_20d | 1907 | 510 | -0.1257 | True |
| 2024-08-07 | failed_breakout_52w | 2550 | 510 | +0.0973 | True |
| 2024-08-07 | days_since_failed_breakout_52w | 957 | 510 | -0.0748 | True |
| 2024-08-07 | failed_breakout_20d__rank_all | 2550 | 510 | -0.0406 | True |
| 2024-08-07 | days_since_failed_breakout_20d__rank_all | 1907 | 510 | +0.0297 | False |
| 2024-08-07 | failed_breakout_52w__rank_all | 2550 | 510 | -0.0915 | True |
| 2024-08-07 | days_since_failed_breakout_52w__rank_all | 957 | 510 | +0.0324 | True |
| 2024-08-07 | failed_breakout_20d__rank_sector | 2550 | 510 | +0.0086 | False |
| 2024-08-07 | days_since_failed_breakout_20d__rank_sector | 1907 | 510 | +0.0344 | True |
| 2024-08-07 | failed_breakout_52w__rank_sector | 2550 | 510 | +0.0056 | False |
| 2024-08-07 | days_since_failed_breakout_52w__rank_sector | 957 | 510 | +0.0515 | True |
| 2024-09-05 | failed_breakout_20d | 2239 | 448 | +0.0550 | True |
| 2024-09-05 | days_since_failed_breakout_20d | 1742 | 448 | +0.0019 | False |
| 2024-09-05 | failed_breakout_52w | 2239 | 448 | +0.0862 | True |
| 2024-09-05 | days_since_failed_breakout_52w | 1018 | 448 | +0.0220 | False |
| 2024-09-05 | failed_breakout_20d__rank_all | 2239 | 448 | +0.0018 | False |
| 2024-09-05 | days_since_failed_breakout_20d__rank_all | 1742 | 448 | -0.0055 | False |
| 2024-09-05 | failed_breakout_52w__rank_all | 2239 | 448 | +0.0589 | True |
| 2024-09-05 | days_since_failed_breakout_52w__rank_all | 1018 | 448 | -0.0191 | False |
| 2024-09-05 | failed_breakout_20d__rank_sector | 2239 | 448 | +0.0307 | True |
| 2024-09-05 | days_since_failed_breakout_20d__rank_sector | 1742 | 448 | +0.0185 | False |
| 2024-09-05 | failed_breakout_52w__rank_sector | 2239 | 448 | +0.0573 | True |
| 2024-09-05 | days_since_failed_breakout_52w__rank_sector | 1018 | 448 | +0.0022 | False |
| 2024-10-03 | failed_breakout_20d | 3316 | 664 | -0.2297 | True |
| 2024-10-03 | days_since_failed_breakout_20d | 2367 | 664 | -0.0174 | False |
| 2024-10-03 | failed_breakout_52w | 3316 | 664 | -0.0838 | True |
| 2024-10-03 | days_since_failed_breakout_52w | 1070 | 664 | +0.0256 | False |
| 2024-10-03 | failed_breakout_20d__rank_all | 3316 | 664 | -0.0987 | True |
| 2024-10-03 | days_since_failed_breakout_20d__rank_all | 2367 | 664 | -0.0060 | False |
| 2024-10-03 | failed_breakout_52w__rank_all | 3316 | 664 | -0.0831 | True |
| 2024-10-03 | days_since_failed_breakout_52w__rank_all | 1070 | 664 | -0.0381 | True |
| 2024-10-03 | failed_breakout_20d__rank_sector | 3316 | 664 | -0.0566 | True |
| 2024-10-03 | days_since_failed_breakout_20d__rank_sector | 2367 | 664 | -0.0039 | False |
| 2024-10-03 | failed_breakout_52w__rank_sector | 3316 | 664 | +0.0071 | False |
| 2024-10-03 | days_since_failed_breakout_52w__rank_sector | 1070 | 664 | +0.0022 | False |
| 2024-10-31 | failed_breakout_20d | 2925 | 585 | +0.0182 | False |
| 2024-10-31 | days_since_failed_breakout_20d | 2263 | 585 | -0.1043 | True |
| 2024-10-31 | failed_breakout_52w | 2925 | 585 | +0.0953 | True |
| 2024-10-31 | days_since_failed_breakout_52w | 1244 | 585 | -0.0494 | True |
| 2024-10-31 | failed_breakout_20d__rank_all | 2925 | 585 | -0.0357 | True |
| 2024-10-31 | days_since_failed_breakout_20d__rank_all | 2263 | 585 | +0.0182 | False |
| 2024-10-31 | failed_breakout_52w__rank_all | 2925 | 585 | -0.0512 | True |
| 2024-10-31 | days_since_failed_breakout_52w__rank_all | 1244 | 585 | +0.0518 | True |
| 2024-10-31 | failed_breakout_20d__rank_sector | 2925 | 585 | +0.0017 | False |
| 2024-10-31 | days_since_failed_breakout_20d__rank_sector | 2263 | 585 | +0.0334 | True |
| 2024-10-31 | failed_breakout_52w__rank_sector | 2925 | 585 | +0.0211 | False |
| 2024-10-31 | days_since_failed_breakout_52w__rank_sector | 1244 | 585 | +0.0456 | True |
| 2024-11-29 | failed_breakout_20d | 2692 | 539 | +0.0460 | True |
| 2024-11-29 | days_since_failed_breakout_20d | 2109 | 539 | -0.0473 | True |
| 2024-11-29 | failed_breakout_52w | 2692 | 539 | +0.0690 | True |
| 2024-11-29 | days_since_failed_breakout_52w | 1260 | 539 | -0.0155 | False |
| 2024-11-29 | failed_breakout_20d__rank_all | 2692 | 539 | +0.0223 | False |
| 2024-11-29 | days_since_failed_breakout_20d__rank_all | 2109 | 539 | -0.0023 | False |
| 2024-11-29 | failed_breakout_52w__rank_all | 2692 | 539 | +0.0409 | True |
| 2024-11-29 | days_since_failed_breakout_52w__rank_all | 1260 | 539 | -0.0155 | False |
| 2024-11-29 | failed_breakout_20d__rank_sector | 2692 | 539 | +0.0409 | True |
| 2024-11-29 | days_since_failed_breakout_20d__rank_sector | 2109 | 539 | +0.0227 | False |
| 2024-11-29 | failed_breakout_52w__rank_sector | 2692 | 539 | +0.0431 | True |
| 2024-11-29 | days_since_failed_breakout_52w__rank_sector | 1260 | 539 | +0.0107 | False |
| 2024-12-30 | failed_breakout_20d | 3994 | 799 | -0.2032 | True |
| 2024-12-30 | days_since_failed_breakout_20d | 2950 | 799 | -0.0209 | False |
| 2024-12-30 | failed_breakout_52w | 3994 | 799 | -0.0795 | True |
| 2024-12-30 | days_since_failed_breakout_52w | 1453 | 799 | +0.0020 | False |
| 2024-12-30 | failed_breakout_20d__rank_all | 3994 | 799 | -0.1003 | True |
| 2024-12-30 | days_since_failed_breakout_20d__rank_all | 2950 | 799 | -0.0029 | False |
| 2024-12-30 | failed_breakout_52w__rank_all | 3994 | 799 | -0.0487 | True |
| 2024-12-30 | days_since_failed_breakout_52w__rank_all | 1453 | 799 | -0.0422 | True |
| 2024-12-30 | failed_breakout_20d__rank_sector | 3994 | 799 | -0.0352 | True |
| 2024-12-30 | days_since_failed_breakout_20d__rank_sector | 2950 | 799 | -0.0110 | False |
| 2024-12-30 | failed_breakout_52w__rank_sector | 3994 | 799 | +0.0118 | False |
| 2024-12-30 | days_since_failed_breakout_52w__rank_sector | 1453 | 799 | -0.0237 | False |
| 2025-01-30 | failed_breakout_20d | 3261 | 653 | +0.0243 | False |
| 2025-01-30 | days_since_failed_breakout_20d | 2570 | 653 | -0.1068 | True |
| 2025-01-30 | failed_breakout_52w | 3261 | 653 | +0.1005 | True |
| 2025-01-30 | days_since_failed_breakout_52w | 1495 | 653 | -0.0536 | True |
| 2025-01-30 | failed_breakout_20d__rank_all | 3261 | 653 | -0.0524 | True |
| 2025-01-30 | days_since_failed_breakout_20d__rank_all | 2570 | 653 | +0.0091 | False |
| 2025-01-30 | failed_breakout_52w__rank_all | 3261 | 653 | -0.0362 | True |
| 2025-01-30 | days_since_failed_breakout_52w__rank_all | 1495 | 653 | +0.0297 | False |
| 2025-01-30 | failed_breakout_20d__rank_sector | 3261 | 653 | -0.0137 | False |
| 2025-01-30 | days_since_failed_breakout_20d__rank_sector | 2570 | 653 | +0.0285 | False |
| 2025-01-30 | failed_breakout_52w__rank_sector | 3261 | 653 | +0.0205 | False |
| 2025-01-30 | days_since_failed_breakout_52w__rank_sector | 1495 | 653 | +0.0322 | True |
| 2025-02-28 | failed_breakout_20d | 3170 | 634 | +0.0325 | True |
| 2025-02-28 | days_since_failed_breakout_20d | 2550 | 634 | -0.0339 | True |
| 2025-02-28 | failed_breakout_52w | 3170 | 634 | +0.0454 | True |
| 2025-02-28 | days_since_failed_breakout_52w | 1637 | 634 | -0.0116 | False |
| 2025-02-28 | failed_breakout_20d__rank_all | 3170 | 634 | +0.0162 | False |
| 2025-02-28 | days_since_failed_breakout_20d__rank_all | 2550 | 634 | +0.0131 | False |
| 2025-02-28 | failed_breakout_52w__rank_all | 3170 | 634 | +0.0590 | True |
| 2025-02-28 | days_since_failed_breakout_52w__rank_all | 1637 | 634 | +0.0145 | False |
| 2025-02-28 | failed_breakout_20d__rank_sector | 3170 | 634 | +0.0532 | True |
| 2025-02-28 | days_since_failed_breakout_20d__rank_sector | 2550 | 634 | +0.0321 | True |
| 2025-02-28 | failed_breakout_52w__rank_sector | 3170 | 634 | +0.0601 | True |
| 2025-02-28 | days_since_failed_breakout_52w__rank_sector | 1637 | 634 | +0.0172 | False |
| 2025-05-12 | failed_breakout_20d | 4306 | 862 | -0.1914 | True |
| 2025-05-12 | days_since_failed_breakout_20d | 3150 | 862 | -0.0118 | False |
| 2025-05-12 | failed_breakout_52w | 4306 | 862 | -0.0723 | True |
| 2025-05-12 | days_since_failed_breakout_52w | 1620 | 862 | +0.0166 | False |
| 2025-05-12 | failed_breakout_20d__rank_all | 4306 | 862 | -0.0836 | True |
| 2025-05-12 | days_since_failed_breakout_20d__rank_all | 3150 | 862 | -0.0011 | False |
| 2025-05-12 | failed_breakout_52w__rank_all | 4306 | 862 | -0.0388 | True |
| 2025-05-12 | days_since_failed_breakout_52w__rank_all | 1620 | 862 | -0.0289 | False |
| 2025-05-12 | failed_breakout_20d__rank_sector | 4306 | 862 | -0.0275 | False |
| 2025-05-12 | days_since_failed_breakout_20d__rank_sector | 3150 | 862 | -0.0125 | False |
| 2025-05-12 | failed_breakout_52w__rank_sector | 4306 | 862 | +0.0142 | False |
| 2025-05-12 | days_since_failed_breakout_52w__rank_sector | 1620 | 862 | -0.0198 | False |
| 2025-06-10 | failed_breakout_20d | 3644 | 729 | +0.0223 | False |
| 2025-06-10 | days_since_failed_breakout_20d | 2917 | 729 | -0.0855 | True |
| 2025-06-10 | failed_breakout_52w | 3644 | 729 | +0.0967 | True |
| 2025-06-10 | days_since_failed_breakout_52w | 1665 | 729 | -0.0343 | True |
| 2025-06-10 | failed_breakout_20d__rank_all | 3644 | 729 | -0.0436 | True |
| 2025-06-10 | days_since_failed_breakout_20d__rank_all | 2917 | 729 | -0.0030 | False |
| 2025-06-10 | failed_breakout_52w__rank_all | 3644 | 729 | -0.0331 | True |
| 2025-06-10 | days_since_failed_breakout_52w__rank_all | 1665 | 729 | +0.0154 | False |
| 2025-06-10 | failed_breakout_20d__rank_sector | 3644 | 729 | -0.0037 | False |
| 2025-06-10 | days_since_failed_breakout_20d__rank_sector | 2917 | 729 | +0.0307 | True |
| 2025-06-10 | failed_breakout_52w__rank_sector | 3644 | 729 | +0.0158 | False |
| 2025-06-10 | days_since_failed_breakout_52w__rank_sector | 1665 | 729 | +0.0402 | True |
| 2025-07-10 | failed_breakout_20d | 3419 | 684 | +0.0316 | True |
| 2025-07-10 | days_since_failed_breakout_20d | 2771 | 684 | -0.0246 | False |
| 2025-07-10 | failed_breakout_52w | 3419 | 684 | +0.0552 | True |
| 2025-07-10 | days_since_failed_breakout_52w | 1799 | 684 | -0.0106 | False |
| 2025-07-10 | failed_breakout_20d__rank_all | 3419 | 684 | +0.0174 | False |
| 2025-07-10 | days_since_failed_breakout_20d__rank_all | 2771 | 684 | +0.0225 | False |
| 2025-07-10 | failed_breakout_52w__rank_all | 3419 | 684 | +0.0660 | True |
| 2025-07-10 | days_since_failed_breakout_52w__rank_all | 1799 | 684 | +0.0235 | False |
| 2025-07-10 | failed_breakout_20d__rank_sector | 3419 | 684 | +0.0509 | True |
| 2025-07-10 | days_since_failed_breakout_20d__rank_sector | 2771 | 684 | +0.0370 | True |
| 2025-07-10 | failed_breakout_52w__rank_sector | 3419 | 684 | +0.0660 | True |
| 2025-07-10 | days_since_failed_breakout_52w__rank_sector | 1799 | 684 | +0.0257 | False |
| 2025-08-07 | failed_breakout_20d | 4374 | 875 | -0.1814 | True |
| 2025-08-07 | days_since_failed_breakout_20d | 3179 | 875 | -0.0082 | False |
| 2025-08-07 | failed_breakout_52w | 4374 | 875 | -0.0667 | True |
| 2025-08-07 | days_since_failed_breakout_52w | 1622 | 875 | +0.0189 | False |
| 2025-08-07 | failed_breakout_20d__rank_all | 4374 | 875 | -0.0828 | True |
| 2025-08-07 | days_since_failed_breakout_20d__rank_all | 3179 | 875 | +0.0007 | False |
| 2025-08-07 | failed_breakout_52w__rank_all | 4374 | 875 | -0.0415 | True |
| 2025-08-07 | days_since_failed_breakout_52w__rank_all | 1622 | 875 | -0.0272 | False |
| 2025-08-07 | failed_breakout_20d__rank_sector | 4374 | 875 | -0.0275 | False |
| 2025-08-07 | days_since_failed_breakout_20d__rank_sector | 3179 | 875 | -0.0127 | False |
| 2025-08-07 | failed_breakout_52w__rank_sector | 4374 | 875 | +0.0094 | False |
| 2025-08-07 | days_since_failed_breakout_52w__rank_sector | 1622 | 875 | -0.0192 | False |
| 2025-09-05 | failed_breakout_20d | 4163 | 833 | +0.0352 | True |
| 2025-09-05 | days_since_failed_breakout_20d | 3335 | 833 | -0.0559 | True |
| 2025-09-05 | failed_breakout_52w | 4163 | 833 | +0.0726 | True |
| 2025-09-05 | days_since_failed_breakout_52w | 1760 | 833 | -0.0261 | False |
| 2025-09-05 | failed_breakout_20d__rank_all | 4163 | 833 | -0.0358 | True |
| 2025-09-05 | days_since_failed_breakout_20d__rank_all | 3335 | 833 | +0.0145 | False |
| 2025-09-05 | failed_breakout_52w__rank_all | 4163 | 833 | -0.0095 | False |
| 2025-09-05 | days_since_failed_breakout_52w__rank_all | 1760 | 833 | +0.0209 | False |
| 2025-09-05 | failed_breakout_20d__rank_sector | 4163 | 833 | -0.0051 | False |
| 2025-09-05 | days_since_failed_breakout_20d__rank_sector | 3335 | 833 | +0.0361 | True |
| 2025-09-05 | failed_breakout_52w__rank_sector | 4163 | 833 | +0.0151 | False |
| 2025-09-05 | days_since_failed_breakout_52w__rank_sector | 1760 | 833 | +0.0401 | True |
| 2025-10-03 | failed_breakout_20d | 3784 | 757 | +0.0260 | False |
| 2025-10-03 | days_since_failed_breakout_20d | 3116 | 757 | -0.0253 | False |
| 2025-10-03 | failed_breakout_52w | 3784 | 757 | +0.0547 | True |
| 2025-10-03 | days_since_failed_breakout_52w | 1941 | 757 | -0.0110 | False |
| 2025-10-03 | failed_breakout_20d__rank_all | 3784 | 757 | +0.0179 | False |
| 2025-10-03 | days_since_failed_breakout_20d__rank_all | 3116 | 757 | +0.0144 | False |
| 2025-10-03 | failed_breakout_52w__rank_all | 3784 | 757 | +0.0576 | True |
| 2025-10-03 | days_since_failed_breakout_52w__rank_all | 1941 | 757 | +0.0232 | False |
| 2025-10-03 | failed_breakout_20d__rank_sector | 3784 | 757 | +0.0488 | True |
| 2025-10-03 | days_since_failed_breakout_20d__rank_sector | 3116 | 757 | +0.0299 | False |
| 2025-10-03 | failed_breakout_52w__rank_sector | 3784 | 757 | +0.0585 | True |
| 2025-10-03 | days_since_failed_breakout_52w__rank_sector | 1941 | 757 | +0.0223 | False |
| 2025-10-31 | failed_breakout_20d | 4852 | 971 | -0.1518 | True |
| 2025-10-31 | days_since_failed_breakout_20d | 3543 | 971 | -0.0008 | False |
| 2025-10-31 | failed_breakout_52w | 4852 | 971 | -0.0456 | True |
| 2025-10-31 | days_since_failed_breakout_52w | 1802 | 971 | +0.0257 | False |
| 2025-10-31 | failed_breakout_20d__rank_all | 4852 | 971 | -0.0535 | True |
| 2025-10-31 | days_since_failed_breakout_20d__rank_all | 3543 | 971 | +0.0045 | False |
| 2025-10-31 | failed_breakout_52w__rank_all | 4852 | 971 | -0.0108 | False |
| 2025-10-31 | days_since_failed_breakout_52w__rank_all | 1802 | 971 | -0.0102 | False |
| 2025-10-31 | failed_breakout_20d__rank_sector | 4852 | 971 | -0.0108 | False |
| 2025-10-31 | days_since_failed_breakout_20d__rank_sector | 3543 | 971 | -0.0169 | False |
| 2025-10-31 | failed_breakout_52w__rank_sector | 4852 | 971 | +0.0221 | False |
| 2025-10-31 | days_since_failed_breakout_52w__rank_sector | 1802 | 971 | -0.0120 | False |
| 2025-12-01 | failed_breakout_20d | 4586 | 918 | +0.0302 | True |
| 2025-12-01 | days_since_failed_breakout_20d | 3704 | 918 | -0.0424 | True |
| 2025-12-01 | failed_breakout_52w | 4586 | 918 | +0.0695 | True |
| 2025-12-01 | days_since_failed_breakout_52w | 1935 | 918 | -0.0134 | False |
| 2025-12-01 | failed_breakout_20d__rank_all | 4586 | 918 | -0.0315 | True |
| 2025-12-01 | days_since_failed_breakout_20d__rank_all | 3704 | 918 | +0.0275 | False |
| 2025-12-01 | failed_breakout_52w__rank_all | 4586 | 918 | -0.0108 | False |
| 2025-12-01 | days_since_failed_breakout_52w__rank_all | 1935 | 918 | +0.0354 | True |
| 2025-12-01 | failed_breakout_20d__rank_sector | 4586 | 918 | -0.0048 | False |
| 2025-12-01 | days_since_failed_breakout_20d__rank_sector | 3704 | 918 | +0.0468 | True |
| 2025-12-01 | failed_breakout_52w__rank_sector | 4586 | 918 | +0.0159 | False |
| 2025-12-01 | days_since_failed_breakout_52w__rank_sector | 1935 | 918 | +0.0540 | True |
| 2025-12-30 | failed_breakout_20d | 4392 | 879 | +0.0274 | False |
| 2025-12-30 | days_since_failed_breakout_20d | 3555 | 879 | -0.0180 | False |
| 2025-12-30 | failed_breakout_52w | 4392 | 879 | +0.0519 | True |
| 2025-12-30 | days_since_failed_breakout_52w | 2137 | 879 | -0.0036 | False |
| 2025-12-30 | failed_breakout_20d__rank_all | 4392 | 879 | +0.0352 | True |
| 2025-12-30 | days_since_failed_breakout_20d__rank_all | 3555 | 879 | -0.0005 | False |
| 2025-12-30 | failed_breakout_52w__rank_all | 4392 | 879 | +0.0603 | True |
| 2025-12-30 | days_since_failed_breakout_52w__rank_all | 2137 | 879 | +0.0090 | False |
| 2025-12-30 | failed_breakout_20d__rank_sector | 4392 | 879 | +0.0556 | True |
| 2025-12-30 | days_since_failed_breakout_20d__rank_sector | 3555 | 879 | +0.0156 | False |
| 2025-12-30 | failed_breakout_52w__rank_sector | 4392 | 879 | +0.0587 | True |
| 2025-12-30 | days_since_failed_breakout_52w__rank_sector | 2137 | 879 | +0.0162 | False |
| 2026-01-29 | failed_breakout_20d | 5138 | 1028 | -0.1312 | True |
| 2026-01-29 | days_since_failed_breakout_20d | 3799 | 1028 | -0.0064 | False |
| 2026-01-29 | failed_breakout_52w | 5138 | 1028 | -0.0301 | True |
| 2026-01-29 | days_since_failed_breakout_52w | 1969 | 1028 | +0.0099 | False |
| 2026-01-29 | failed_breakout_20d__rank_all | 5138 | 1028 | -0.0752 | True |
| 2026-01-29 | days_since_failed_breakout_20d__rank_all | 3799 | 1028 | +0.0004 | False |
| 2026-01-29 | failed_breakout_52w__rank_all | 5138 | 1028 | -0.0143 | False |
| 2026-01-29 | days_since_failed_breakout_52w__rank_all | 1969 | 1028 | +0.0041 | False |
| 2026-01-29 | failed_breakout_20d__rank_sector | 5138 | 1028 | -0.0088 | False |
| 2026-01-29 | days_since_failed_breakout_20d__rank_sector | 3799 | 1028 | -0.0154 | False |
| 2026-01-29 | failed_breakout_52w__rank_sector | 5138 | 1028 | +0.0199 | False |
| 2026-01-29 | days_since_failed_breakout_52w__rank_sector | 1969 | 1028 | -0.0092 | False |
| 2026-02-27 | failed_breakout_20d | 4876 | 976 | +0.0316 | True |
| 2026-02-27 | days_since_failed_breakout_20d | 3940 | 976 | -0.0286 | False |
| 2026-02-27 | failed_breakout_52w | 4876 | 976 | +0.0846 | True |
| 2026-02-27 | days_since_failed_breakout_52w | 2101 | 976 | +0.0218 | False |
| 2026-02-27 | failed_breakout_20d__rank_all | 4876 | 976 | -0.0325 | True |
| 2026-02-27 | days_since_failed_breakout_20d__rank_all | 3940 | 976 | +0.0274 | False |
| 2026-02-27 | failed_breakout_52w__rank_all | 4876 | 976 | -0.0060 | False |
| 2026-02-27 | days_since_failed_breakout_52w__rank_all | 2101 | 976 | +0.0350 | True |
| 2026-02-27 | failed_breakout_20d__rank_sector | 4876 | 976 | +0.0180 | False |
| 2026-02-27 | days_since_failed_breakout_20d__rank_sector | 3940 | 976 | +0.0421 | True |
| 2026-02-27 | failed_breakout_52w__rank_sector | 4876 | 976 | +0.0222 | False |
| 2026-02-27 | days_since_failed_breakout_52w__rank_sector | 2101 | 976 | +0.0485 | True |
| 2026-04-15 | failed_breakout_20d | 4771 | 955 | +0.0343 | True |
| 2026-04-15 | days_since_failed_breakout_20d | 3878 | 955 | -0.0117 | False |
| 2026-04-15 | failed_breakout_52w | 4771 | 955 | +0.0592 | True |
| 2026-04-15 | days_since_failed_breakout_52w | 2343 | 955 | +0.0110 | False |
| 2026-04-15 | failed_breakout_20d__rank_all | 4771 | 955 | +0.0422 | True |
| 2026-04-15 | days_since_failed_breakout_20d__rank_all | 3878 | 955 | -0.0092 | False |
| 2026-04-15 | failed_breakout_52w__rank_all | 4771 | 955 | +0.0638 | True |
| 2026-04-15 | days_since_failed_breakout_52w__rank_all | 2343 | 955 | -0.0027 | False |
| 2026-04-15 | failed_breakout_20d__rank_sector | 4771 | 955 | +0.0554 | True |
| 2026-04-15 | days_since_failed_breakout_20d__rank_sector | 3878 | 955 | +0.0135 | False |
| 2026-04-15 | failed_breakout_52w__rank_sector | 4771 | 955 | +0.0648 | True |
| 2026-04-15 | days_since_failed_breakout_52w__rank_sector | 2343 | 955 | +0.0091 | False |
| 2026-05-13 | failed_breakout_20d | 5652 | 1131 | -0.1109 | True |
| 2026-05-13 | days_since_failed_breakout_20d | 4281 | 1131 | -0.0132 | False |
| 2026-05-13 | failed_breakout_52w | 5652 | 1131 | -0.0103 | False |
| 2026-05-13 | days_since_failed_breakout_52w | 2280 | 1131 | -0.0038 | False |
| 2026-05-13 | failed_breakout_20d__rank_all | 5652 | 1131 | -0.0894 | True |
| 2026-05-13 | days_since_failed_breakout_20d__rank_all | 4281 | 1131 | +0.0094 | False |
| 2026-05-13 | failed_breakout_52w__rank_all | 5652 | 1131 | -0.0053 | False |
| 2026-05-13 | days_since_failed_breakout_52w__rank_all | 2280 | 1131 | +0.0094 | False |
| 2026-05-13 | failed_breakout_20d__rank_sector | 5652 | 1131 | -0.0321 | True |
| 2026-05-13 | days_since_failed_breakout_20d__rank_sector | 4281 | 1131 | -0.0180 | False |
| 2026-05-13 | failed_breakout_52w__rank_sector | 5652 | 1131 | +0.0257 | False |
| 2026-05-13 | days_since_failed_breakout_52w__rank_sector | 2280 | 1131 | -0.0133 | False |
| 2026-06-11 | failed_breakout_20d | 5517 | 1104 | +0.0143 | False |
| 2026-06-11 | days_since_failed_breakout_20d | 4516 | 1104 | -0.0142 | False |
| 2026-06-11 | failed_breakout_52w | 5517 | 1104 | +0.0681 | True |
| 2026-06-11 | days_since_failed_breakout_52w | 2533 | 1104 | +0.0137 | False |
| 2026-06-11 | failed_breakout_20d__rank_all | 5517 | 1104 | -0.0074 | False |
| 2026-06-11 | days_since_failed_breakout_20d__rank_all | 4516 | 1104 | +0.0233 | False |
| 2026-06-11 | failed_breakout_52w__rank_all | 5517 | 1104 | +0.0204 | False |
| 2026-06-11 | days_since_failed_breakout_52w__rank_all | 2533 | 1104 | +0.0203 | False |
| 2026-06-11 | failed_breakout_20d__rank_sector | 5517 | 1104 | +0.0170 | False |
| 2026-06-11 | days_since_failed_breakout_20d__rank_sector | 4516 | 1104 | +0.0389 | True |
| 2026-06-11 | failed_breakout_52w__rank_sector | 5517 | 1104 | +0.0324 | True |
| 2026-06-11 | days_since_failed_breakout_52w__rank_sector | 2533 | 1104 | +0.0530 | True |

## Honest-Grid With/Without Bakeoff

| model | without_d3_rows | with_d3_rows | without_d3_full_oos_spearman | with_d3_full_oos_spearman | delta_with_minus_without |
|---|---:|---:|---:|---:|---:|
| ridge_model | 7889 | 7889 | -0.0193 | -0.0261 | -0.0067 |
| lasso_model | 7889 | 7889 | -0.0209 | -0.0262 | -0.0054 |
| xgboost_model | 7889 | 7889 | -0.0268 | -0.0170 | +0.0098 |
