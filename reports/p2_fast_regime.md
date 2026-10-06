# P2 Fast-Regime A4 Interaction Audit

- idea: replace A4's stale lagged regime-meter interaction source with a same-grid OHLC fast regime, then compare honestly
- data_access: DuckDB read_only=True; no `./sq` writes; no data/ mutation
- fast_regime: `SPY` 5d and 20d adjusted-close returns; trending only when both are non-negative, otherwise reversal
- production_wiring: unchanged unless fast regime wins this audit
- top_features: sector_median_roc_63, sector_pct_above_50, sector_median_roc_63__rank_all, sector_pct_above_200, relative_strength_index_vs_subindustry, max_gap_down_pct_60__rank_all, avg_abs_gap_pct_20__rank_all, sector_pct_above_50__rank_sector, sma_200_dist__rank_all, relative_strength_index_vs_subindustry__rank_all, roc_126__rank_all, base_range_pct_20__rank_all, max_gap_down_pct_60__rank_sector, atr_pct_14__rank_all, close_vs_20d_low__rank_all
- meter_surviving_interactions: 32
- fast_surviving_interactions: 32
- meter_regime_counts: {'trending': 317, 'reversal': 262, 'neutral': 253}
- fast_regime_counts: {'trending': 457, 'reversal': 375}
- verdict: fast regime ties: all model deltas versus meter are inside +/-0.005 Spearman

## Honest-Grid Three-Way Bakeoff

| model | without_rows | meter_rows | fast_rows | without_spearman | with_a4_meter | with_a4_fast | meter_delta_vs_without | fast_delta_vs_without | fast_delta_vs_meter |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ridge_model | 7896 | 7896 | 7896 | -0.0168 | +0.0032 | +0.0034 | +0.0200 | +0.0202 | +0.0002 |
| ic_sign_model | 7896 | 7896 | 7896 | -0.0072 | -0.0315 | -0.0332 | -0.0243 | -0.0260 | -0.0017 |

## Interaction IC Survivors

### with_a4_meter

| feature | folds | mean_ic | mean_abs_ic | max_abs_ic | clear_folds | clear_rate |
|---|---:|---:|---:|---:|---:|---:|
| a4_sector_pct_above_50_x_regime_reversal | 26 | -0.1133 | +0.1133 | +0.1963 | 26 | +1.0000 |
| a4_sector_median_roc_63_x_regime_reversal | 26 | -0.0884 | +0.0937 | +0.2542 | 17 | +0.6538 |
| a4_regime_reversal | 26 | -0.0554 | +0.0898 | +0.1541 | 23 | +0.8846 |
| a4_atr_pct_14__rank_all_x_regime_reversal | 26 | -0.0491 | +0.0881 | +0.1524 | 24 | +0.9231 |
| a4_max_gap_down_pct_60__rank_all_x_regime_reversal | 26 | -0.0438 | +0.0860 | +0.1423 | 24 | +0.9231 |
| a4_avg_abs_gap_pct_20__rank_all_x_regime_reversal | 26 | -0.0511 | +0.0843 | +0.1497 | 23 | +0.8846 |
| a4_max_gap_down_pct_60__rank_sector_x_regime_reversal | 26 | -0.0414 | +0.0832 | +0.1392 | 24 | +0.9231 |
| a4_sector_pct_above_200_x_regime_reversal | 26 | -0.0807 | +0.0824 | +0.1673 | 22 | +0.8462 |
| a4_base_range_pct_20__rank_all_x_regime_reversal | 26 | -0.0564 | +0.0805 | +0.1494 | 23 | +0.8846 |
| a4_max_gap_down_pct_60__rank_sector_x_regime_trending | 26 | -0.0096 | +0.0789 | +0.1633 | 22 | +0.8462 |
| a4_close_vs_20d_low__rank_all_x_regime_reversal | 26 | -0.0683 | +0.0785 | +0.1499 | 20 | +0.7692 |
| a4_max_gap_down_pct_60__rank_all_x_regime_trending | 26 | -0.0092 | +0.0773 | +0.1624 | 22 | +0.8462 |
| a4_base_range_pct_20__rank_all_x_regime_trending | 26 | -0.0119 | +0.0761 | +0.1599 | 21 | +0.8077 |
| a4_sector_median_roc_63__rank_all_x_regime_reversal | 26 | -0.0650 | +0.0756 | +0.1378 | 20 | +0.7692 |
| a4_avg_abs_gap_pct_20__rank_all_x_regime_trending | 26 | -0.0116 | +0.0751 | +0.1614 | 22 | +0.8462 |
| a4_atr_pct_14__rank_all_x_regime_trending | 26 | -0.0137 | +0.0750 | +0.1585 | 21 | +0.8077 |
| a4_relative_strength_index_vs_subindustry__rank_all_x_regime_reversal | 26 | -0.0550 | +0.0743 | +0.1367 | 23 | +0.8846 |
| a4_close_vs_20d_low__rank_all_x_regime_trending | 26 | -0.0087 | +0.0742 | +0.1604 | 20 | +0.7692 |
| a4_regime_trending | 26 | -0.0073 | +0.0740 | +0.1542 | 22 | +0.8462 |
| a4_sector_pct_above_50_x_regime_trending | 26 | -0.0064 | +0.0739 | +0.1596 | 22 | +0.8462 |
| a4_relative_strength_index_vs_subindustry_x_regime_trending | 26 | -0.0058 | +0.0736 | +0.1543 | 22 | +0.8462 |
| a4_sector_pct_above_50__rank_sector_x_regime_trending | 26 | -0.0005 | +0.0734 | +0.1500 | 21 | +0.8077 |
| a4_roc_126__rank_all_x_regime_reversal | 26 | -0.0534 | +0.0725 | +0.1336 | 23 | +0.8846 |
| a4_sma_200_dist__rank_all_x_regime_trending | 26 | -0.0044 | +0.0724 | +0.1583 | 21 | +0.8077 |
| a4_relative_strength_index_vs_subindustry__rank_all_x_regime_trending | 26 | -0.0069 | +0.0721 | +0.1545 | 22 | +0.8462 |
| a4_roc_126__rank_all_x_regime_trending | 26 | -0.0037 | +0.0716 | +0.1579 | 21 | +0.8077 |
| a4_sector_pct_above_200_x_regime_trending | 26 | -0.0179 | +0.0713 | +0.1609 | 19 | +0.7308 |
| a4_sma_200_dist__rank_all_x_regime_reversal | 26 | -0.0536 | +0.0709 | +0.1328 | 23 | +0.8846 |
| a4_sector_median_roc_63__rank_all_x_regime_trending | 26 | -0.0133 | +0.0704 | +0.1582 | 20 | +0.7692 |
| a4_sector_pct_above_50__rank_sector_x_regime_reversal | 26 | -0.0636 | +0.0674 | +0.1469 | 21 | +0.8077 |
| a4_relative_strength_index_vs_subindustry_x_regime_reversal | 26 | -0.0521 | +0.0629 | +0.1334 | 20 | +0.7692 |
| a4_sector_median_roc_63_x_regime_trending | 26 | -0.0523 | +0.0619 | +0.1533 | 17 | +0.6538 |

### with_a4_fast

| feature | folds | mean_ic | mean_abs_ic | max_abs_ic | clear_folds | clear_rate |
|---|---:|---:|---:|---:|---:|---:|
| a4_sector_pct_above_50_x_regime_reversal | 26 | -0.0694 | +0.1406 | +0.3415 | 26 | +1.0000 |
| a4_sector_pct_above_50_x_regime_trending | 26 | +0.0081 | +0.1349 | +0.2820 | 25 | +0.9615 |
| a4_sector_pct_above_200_x_regime_reversal | 26 | -0.0577 | +0.1285 | +0.2968 | 26 | +1.0000 |
| a4_sector_pct_above_200_x_regime_trending | 26 | +0.0092 | +0.1271 | +0.2830 | 25 | +0.9615 |
| a4_regime_reversal | 26 | -0.0409 | +0.1249 | +0.2901 | 22 | +0.8462 |
| a4_regime_trending | 26 | +0.0409 | +0.1249 | +0.2901 | 22 | +0.8462 |
| a4_sector_median_roc_63_x_regime_trending | 26 | -0.0321 | +0.1235 | +0.2179 | 26 | +1.0000 |
| a4_max_gap_down_pct_60__rank_sector_x_regime_trending | 26 | +0.0389 | +0.1224 | +0.2925 | 22 | +0.8462 |
| a4_close_vs_20d_low__rank_all_x_regime_trending | 26 | +0.0358 | +0.1215 | +0.2788 | 21 | +0.8077 |
| a4_max_gap_down_pct_60__rank_all_x_regime_trending | 26 | +0.0402 | +0.1214 | +0.2872 | 22 | +0.8462 |
| a4_avg_abs_gap_pct_20__rank_all_x_regime_trending | 26 | +0.0322 | +0.1213 | +0.2810 | 22 | +0.8462 |
| a4_atr_pct_14__rank_all_x_regime_reversal | 26 | -0.0281 | +0.1205 | +0.2638 | 21 | +0.8077 |
| a4_base_range_pct_20__rank_all_x_regime_trending | 26 | +0.0321 | +0.1205 | +0.2832 | 21 | +0.8077 |
| a4_atr_pct_14__rank_all_x_regime_trending | 26 | +0.0289 | +0.1202 | +0.2820 | 22 | +0.8462 |
| a4_avg_abs_gap_pct_20__rank_all_x_regime_reversal | 26 | -0.0281 | +0.1192 | +0.2490 | 22 | +0.8462 |
| a4_base_range_pct_20__rank_all_x_regime_reversal | 26 | -0.0333 | +0.1179 | +0.2566 | 22 | +0.8462 |
| a4_max_gap_down_pct_60__rank_all_x_regime_reversal | 26 | -0.0246 | +0.1166 | +0.2431 | 21 | +0.8077 |
| a4_max_gap_down_pct_60__rank_sector_x_regime_reversal | 26 | -0.0227 | +0.1164 | +0.2361 | 21 | +0.8077 |
| a4_close_vs_20d_low__rank_all_x_regime_reversal | 26 | -0.0405 | +0.1142 | +0.2474 | 22 | +0.8462 |
| a4_sector_median_roc_63__rank_all_x_regime_reversal | 26 | -0.0487 | +0.1135 | +0.2390 | 24 | +0.9231 |
| a4_sector_pct_above_50__rank_sector_x_regime_trending | 26 | +0.0404 | +0.1120 | +0.2747 | 21 | +0.8077 |
| a4_relative_strength_index_vs_subindustry__rank_all_x_regime_reversal | 26 | -0.0251 | +0.1113 | +0.2170 | 22 | +0.8462 |
| a4_sector_median_roc_63__rank_all_x_regime_trending | 26 | +0.0314 | +0.1108 | +0.2792 | 21 | +0.8077 |
| a4_sector_median_roc_63_x_regime_reversal | 26 | -0.1033 | +0.1105 | +0.2375 | 19 | +0.7308 |
| a4_sma_200_dist__rank_all_x_regime_reversal | 26 | -0.0277 | +0.1103 | +0.2181 | 22 | +0.8462 |
| a4_roc_126__rank_all_x_regime_reversal | 26 | -0.0275 | +0.1099 | +0.2248 | 22 | +0.8462 |
| a4_roc_126__rank_all_x_regime_trending | 26 | +0.0490 | +0.1042 | +0.2857 | 20 | +0.7692 |
| a4_sma_200_dist__rank_all_x_regime_trending | 26 | +0.0520 | +0.1041 | +0.2829 | 20 | +0.7692 |
| a4_sector_pct_above_50__rank_sector_x_regime_reversal | 26 | -0.0246 | +0.1025 | +0.1825 | 23 | +0.8846 |
| a4_relative_strength_index_vs_subindustry_x_regime_trending | 26 | +0.0410 | +0.1015 | +0.2669 | 22 | +0.8462 |
| a4_relative_strength_index_vs_subindustry__rank_all_x_regime_trending | 26 | +0.0433 | +0.1013 | +0.2674 | 21 | +0.8077 |
| a4_relative_strength_index_vs_subindustry_x_regime_reversal | 26 | -0.0162 | +0.1008 | +0.1834 | 22 | +0.8462 |
