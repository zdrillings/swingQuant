# E1 Niche Split

- note: niche candidates are included in the promotion gate roster; this report audits their feature profiles and orthogonality.
- target_column: alpha_vs_sector_60d
- promotion_top_n: 2
- oos_evaluation_stride_dates: 20
- label_horizon_dates: 60
- min_feature_ic: 0.0300
- niche_min_observation_policy: 0.2000 of rows with any feature in that niche family, not global OOS rows
- baseline_model: signal_proxy

## Feature Profiles

### overnight_session_specialist
- base_features: overnight_ret_5d, rth_ret_5d, overnight_minus_rth_5d, overnight_ret_20d, rth_ret_20d, overnight_minus_rth_20d, avg_abs_gap_pct_20, max_gap_down_pct_60

### base_pattern_specialist
- base_features: distance_above_20d_high, failed_breakout_20d, days_since_failed_breakout_20d, failed_breakout_52w, days_since_failed_breakout_52w, base_range_pct_20, base_atr_contraction_20, base_volume_dryup_ratio_20, breakout_volume_ratio_50, dollar_volume_ratio_20_60, volume_percentile_60, distance_from_52w_high, days_since_52w_high, close_vs_20d_low

## Full OOS Summary

### overnight_session_specialist
- dates: 421
- avg_pick_count: 1.980998
- gross_mean_target: 0.025261
- net_mean_target: 0.024261
- universe_mean_target: -0.003579
- mean_target_excess: 0.027839
- round_trip_cost: 0.001000
- hit_rate: 0.472684
- universe_hit_rate: 0.446552
- hit_rate_excess: 0.026132
- beat_universe_rate: 0.522565
- spearman: 0.029318
- net_sharpe_ann: 0.289481
- newey_west_t_lag_horizon: 1.486759
- years_for_t_1_96: 45.843048
- positive_date_rate: 0.486936
- ge_2pct_rate: 0.413302
- ge_5pct_rate: 0.353919
- top_ticker: RUN
- top_ticker_date_rate: 0.011876
- top_ticker_pick_share: 0.005995

### signal_proxy
- dates: 517
- avg_pick_count: 1.976789
- gross_mean_target: 0.022068
- net_mean_target: 0.021068
- universe_mean_target: -0.002808
- mean_target_excess: 0.023877
- round_trip_cost: 0.001000
- hit_rate: 0.489362
- universe_hit_rate: 0.450579
- hit_rate_excess: 0.038783
- beat_universe_rate: 0.499033
- spearman: 0.020727
- net_sharpe_ann: 0.244632
- newey_west_t_lag_horizon: 1.182391
- years_for_t_1_96: 64.192778
- positive_date_rate: 0.481625
- ge_2pct_rate: 0.444874
- ge_5pct_rate: 0.382979
- top_ticker: LITE
- top_ticker_date_rate: 0.013540
- top_ticker_pick_share: 0.006849

### base_pattern_specialist
- dates: 517
- avg_pick_count: 1.976789
- gross_mean_target: 0.014121
- net_mean_target: 0.013121
- universe_mean_target: -0.002808
- mean_target_excess: 0.015930
- round_trip_cost: 0.001000
- hit_rate: 0.456480
- universe_hit_rate: 0.450579
- hit_rate_excess: 0.005901
- beat_universe_rate: 0.475822
- spearman: -0.012654
- net_sharpe_ann: 0.157862
- newey_west_t_lag_horizon: 0.869394
- years_for_t_1_96: 154.154100
- positive_date_rate: 0.471954
- ge_2pct_rate: 0.408124
- ge_5pct_rate: 0.334623
- top_ticker: LITE
- top_ticker_date_rate: 0.011605
- top_ticker_pick_share: 0.005871

### structure_factor_signal
- dates: 517
- avg_pick_count: 1.976789
- gross_mean_target: -0.010832
- net_mean_target: -0.011832
- universe_mean_target: -0.002808
- mean_target_excess: -0.009024
- round_trip_cost: 0.001000
- hit_rate: 0.446809
- universe_hit_rate: 0.450579
- hit_rate_excess: -0.003771
- beat_universe_rate: 0.452611
- spearman: -0.014435
- net_sharpe_ann: -0.222135
- newey_west_t_lag_horizon: -1.145350
- years_for_t_1_96: 77.853605
- positive_date_rate: 0.441006
- ge_2pct_rate: 0.369439
- ge_5pct_rate: 0.255319
- top_ticker: L
- top_ticker_date_rate: 0.011605
- top_ticker_pick_share: 0.005871

## Acceptance Windows

### overnight_session_specialist_full_oos
- dates: 421
- avg_pick_count: 1.980998
- gross_mean_target: 0.025261
- net_mean_target: 0.024261
- universe_mean_target: -0.003579
- mean_target_excess: 0.027839
- round_trip_cost: 0.001000
- hit_rate: 0.472684
- universe_hit_rate: 0.446552
- hit_rate_excess: 0.026132
- beat_universe_rate: 0.522565
- spearman: 0.029318
- net_sharpe_ann: 0.289481
- newey_west_t_lag_horizon: 1.486759
- years_for_t_1_96: 45.843048
- positive_date_rate: 0.486936
- ge_2pct_rate: 0.413302
- ge_5pct_rate: 0.353919
- top_ticker: RUN
- top_ticker_date_rate: 0.011876
- top_ticker_pick_share: 0.005995

### signal_proxy_full_oos
- dates: 517
- avg_pick_count: 1.976789
- gross_mean_target: 0.022068
- net_mean_target: 0.021068
- universe_mean_target: -0.002808
- mean_target_excess: 0.023877
- round_trip_cost: 0.001000
- hit_rate: 0.489362
- universe_hit_rate: 0.450579
- hit_rate_excess: 0.038783
- beat_universe_rate: 0.499033
- spearman: 0.020727
- net_sharpe_ann: 0.244632
- newey_west_t_lag_horizon: 1.182391
- years_for_t_1_96: 64.192778
- positive_date_rate: 0.481625
- ge_2pct_rate: 0.444874
- ge_5pct_rate: 0.382979
- top_ticker: LITE
- top_ticker_date_rate: 0.013540
- top_ticker_pick_share: 0.006849

### base_pattern_specialist_full_oos
- dates: 517
- avg_pick_count: 1.976789
- gross_mean_target: 0.014121
- net_mean_target: 0.013121
- universe_mean_target: -0.002808
- mean_target_excess: 0.015930
- round_trip_cost: 0.001000
- hit_rate: 0.456480
- universe_hit_rate: 0.450579
- hit_rate_excess: 0.005901
- beat_universe_rate: 0.475822
- spearman: -0.012654
- net_sharpe_ann: 0.157862
- newey_west_t_lag_horizon: 0.869394
- years_for_t_1_96: 154.154100
- positive_date_rate: 0.471954
- ge_2pct_rate: 0.408124
- ge_5pct_rate: 0.334623
- top_ticker: LITE
- top_ticker_date_rate: 0.011605
- top_ticker_pick_share: 0.005871

### overnight_session_specialist_trailing_3folds
- dates: 60
- avg_pick_count: 1.983333
- gross_mean_target: 0.009677
- net_mean_target: 0.008677
- universe_mean_target: -0.001409
- mean_target_excess: 0.010085
- round_trip_cost: 0.001000
- hit_rate: 0.441667
- universe_hit_rate: 0.456840
- hit_rate_excess: -0.015173
- beat_universe_rate: 0.433333
- spearman: -0.065896
- net_sharpe_ann: 0.096936
- newey_west_t_lag_horizon: 0.495373
- years_for_t_1_96: 408.828429
- positive_date_rate: 0.433333
- ge_2pct_rate: 0.366667
- ge_5pct_rate: 0.350000
- top_ticker: AA
- top_ticker_date_rate: 0.033333
- top_ticker_pick_share: 0.016807

### signal_proxy_last_fold
- dates: 20
- avg_pick_count: 2.000000
- gross_mean_target: 0.004787
- net_mean_target: 0.003787
- universe_mean_target: -0.032088
- mean_target_excess: 0.035875
- round_trip_cost: 0.001000
- hit_rate: 0.475000
- universe_hit_rate: 0.413284
- hit_rate_excess: 0.061716
- beat_universe_rate: 0.550000
- spearman: 0.005117
- net_sharpe_ann: 0.059311
- newey_west_t_lag_horizon: 0.136982
- years_for_t_1_96: 1092.038050
- positive_date_rate: 0.300000
- ge_2pct_rate: 0.300000
- ge_5pct_rate: 0.250000
- top_ticker: OMCL
- top_ticker_date_rate: 0.050000
- top_ticker_pick_share: 0.025000

### structure_factor_signal_trailing_3folds
- dates: 60
- avg_pick_count: 1.983333
- gross_mean_target: 0.004342
- net_mean_target: 0.003342
- universe_mean_target: -0.000522
- mean_target_excess: 0.003863
- round_trip_cost: 0.001000
- hit_rate: 0.458333
- universe_hit_rate: 0.461057
- hit_rate_excess: -0.002724
- beat_universe_rate: 0.416667
- spearman: 0.121225
- net_sharpe_ann: 0.056279
- newey_west_t_lag_horizon: 0.173587
- years_for_t_1_96: 1212.897294
- positive_date_rate: 0.433333
- ge_2pct_rate: 0.366667
- ge_5pct_rate: 0.250000
- top_ticker: CARG
- top_ticker_date_rate: 0.016667
- top_ticker_pick_share: 0.008403

### signal_proxy_trailing_3folds
- dates: 60
- avg_pick_count: 1.983333
- gross_mean_target: -0.010487
- net_mean_target: -0.011487
- universe_mean_target: -0.000522
- mean_target_excess: -0.010965
- round_trip_cost: 0.001000
- hit_rate: 0.425000
- universe_hit_rate: 0.461057
- hit_rate_excess: -0.036057
- beat_universe_rate: 0.400000
- spearman: -0.094222
- net_sharpe_ann: -0.121007
- newey_west_t_lag_horizon: -0.492227
- years_for_t_1_96: 262.356645
- positive_date_rate: 0.333333
- ge_2pct_rate: 0.300000
- ge_5pct_rate: 0.250000
- top_ticker: PTEN
- top_ticker_date_rate: 0.016667
- top_ticker_pick_share: 0.008403

### overnight_session_specialist_last_fold
- dates: 20
- avg_pick_count: 2.000000
- gross_mean_target: -0.010555
- net_mean_target: -0.011555
- universe_mean_target: 0.000776
- mean_target_excess: -0.012331
- round_trip_cost: 0.001000
- hit_rate: 0.425000
- universe_hit_rate: 0.446505
- hit_rate_excess: -0.021505
- beat_universe_rate: 0.400000
- spearman: -0.157551
- net_sharpe_ann: -0.137094
- newey_west_t_lag_horizon: -0.521588
- years_for_t_1_96: 204.398517
- positive_date_rate: 0.450000
- ge_2pct_rate: 0.350000
- ge_5pct_rate: 0.350000
- top_ticker: GLW
- top_ticker_date_rate: 0.050000
- top_ticker_pick_share: 0.025000

### structure_factor_signal_full_oos
- dates: 517
- avg_pick_count: 1.976789
- gross_mean_target: -0.010832
- net_mean_target: -0.011832
- universe_mean_target: -0.002808
- mean_target_excess: -0.009024
- round_trip_cost: 0.001000
- hit_rate: 0.446809
- universe_hit_rate: 0.450579
- hit_rate_excess: -0.003771
- beat_universe_rate: 0.452611
- spearman: -0.014435
- net_sharpe_ann: -0.222135
- newey_west_t_lag_horizon: -1.145350
- years_for_t_1_96: 77.853605
- positive_date_rate: 0.441006
- ge_2pct_rate: 0.369439
- ge_5pct_rate: 0.255319
- top_ticker: L
- top_ticker_date_rate: 0.011605
- top_ticker_pick_share: 0.005871

### base_pattern_specialist_trailing_3folds
- dates: 60
- avg_pick_count: 1.983333
- gross_mean_target: -0.015589
- net_mean_target: -0.016589
- universe_mean_target: -0.000522
- mean_target_excess: -0.016067
- round_trip_cost: 0.001000
- hit_rate: 0.391667
- universe_hit_rate: 0.461057
- hit_rate_excess: -0.069391
- beat_universe_rate: 0.333333
- spearman: -0.162625
- net_sharpe_ann: -0.183143
- newey_west_t_lag_horizon: -0.690983
- years_for_t_1_96: 114.532677
- positive_date_rate: 0.383333
- ge_2pct_rate: 0.333333
- ge_5pct_rate: 0.266667
- top_ticker: PTEN
- top_ticker_date_rate: 0.016667
- top_ticker_pick_share: 0.008403

### structure_factor_signal_last_fold
- dates: 20
- avg_pick_count: 2.000000
- gross_mean_target: -0.033858
- net_mean_target: -0.034858
- universe_mean_target: -0.032088
- mean_target_excess: -0.002770
- round_trip_cost: 0.001000
- hit_rate: 0.375000
- universe_hit_rate: 0.413284
- hit_rate_excess: -0.038284
- beat_universe_rate: 0.350000
- spearman: 0.102954
- net_sharpe_ann: -0.745262
- newey_west_t_lag_horizon: -3.060230
- years_for_t_1_96: 6.916632
- positive_date_rate: 0.300000
- ge_2pct_rate: 0.300000
- ge_5pct_rate: 0.150000
- top_ticker: EQH
- top_ticker_date_rate: 0.050000
- top_ticker_pick_share: 0.025000

### base_pattern_specialist_last_fold
- dates: 20
- avg_pick_count: 2.000000
- gross_mean_target: -0.054216
- net_mean_target: -0.055216
- universe_mean_target: -0.032088
- mean_target_excess: -0.023128
- round_trip_cost: 0.001000
- hit_rate: 0.375000
- universe_hit_rate: 0.413284
- hit_rate_excess: -0.038284
- beat_universe_rate: 0.250000
- spearman: -0.179219
- net_sharpe_ann: -0.959153
- newey_west_t_lag_horizon: -6.833440
- years_for_t_1_96: 4.175765
- positive_date_rate: 0.250000
- ge_2pct_rate: 0.150000
- ge_5pct_rate: 0.150000
- top_ticker: EQH
- top_ticker_date_rate: 0.050000
- top_ticker_pick_share: 0.025000

## Niche Feature IC Diagnostics

| model | feature | rank_ic | abs_rank_ic | observations | family_rows | min_observations | survives |
|---|---|---:|---:|---:|---:|---:|---|
| base_pattern_specialist | failed_breakout_20d__rank_all | 0.129806 | 0.129806 | 6538 | 6538 | 1308 | true |
| base_pattern_specialist | failed_breakout_52w__rank_all | -0.060570 | 0.060570 | 6538 | 6538 | 1308 | true |
| base_pattern_specialist | failed_breakout_52w | -0.056745 | 0.056745 | 6538 | 6538 | 1308 | true |
| base_pattern_specialist | failed_breakout_20d__rank_sector | -0.052044 | 0.052044 | 6538 | 6538 | 1308 | true |
| base_pattern_specialist | days_since_failed_breakout_20d__rank_all | 0.049186 | 0.049186 | 5966 | 5966 | 1194 | true |
| base_pattern_specialist | days_since_failed_breakout_52w__rank_sector | 0.045172 | 0.045172 | 4062 | 4062 | 813 | true |
| base_pattern_specialist | base_range_pct_20__rank_all | 0.040809 | 0.040809 | 221600 | 221600 | 44320 | true |
| base_pattern_specialist | close_vs_20d_low__rank_all | 0.039610 | 0.039610 | 221600 | 221600 | 44320 | true |
| base_pattern_specialist | days_since_failed_breakout_20d__rank_sector | 0.038445 | 0.038445 | 5966 | 5966 | 1194 | true |
| base_pattern_specialist | close_vs_20d_low__rank_sector | 0.031599 | 0.031599 | 221600 | 221600 | 44320 | true |
| base_pattern_specialist | days_since_failed_breakout_52w__rank_all | 0.029236 | 0.029236 | 4062 | 4062 | 813 | false |
| base_pattern_specialist | base_range_pct_20__rank_sector | 0.028300 | 0.028300 | 221600 | 221600 | 44320 | false |
| base_pattern_specialist | days_since_52w_high__rank_sector | -0.023484 | 0.023484 | 221056 | 221056 | 44212 | false |
| base_pattern_specialist | failed_breakout_52w__rank_sector | -0.022947 | 0.022947 | 6538 | 6538 | 1308 | false |
| base_pattern_specialist | distance_from_52w_high__rank_sector | 0.020517 | 0.020517 | 221056 | 221056 | 44212 | false |
| base_pattern_specialist | days_since_52w_high__rank_all | -0.018271 | 0.018271 | 221056 | 221056 | 44212 | false |
| base_pattern_specialist | failed_breakout_20d | 0.017590 | 0.017590 | 6538 | 6538 | 1308 | false |
| base_pattern_specialist | base_range_pct_20 | 0.017338 | 0.017338 | 221600 | 221600 | 44320 | false |
| base_pattern_specialist | close_vs_20d_low | 0.014513 | 0.014513 | 221600 | 221600 | 44320 | false |
| base_pattern_specialist | days_since_failed_breakout_52w | -0.014306 | 0.014306 | 4062 | 4062 | 813 | false |
| base_pattern_specialist | dollar_volume_ratio_20_60__rank_all | 0.012263 | 0.012263 | 221600 | 221600 | 44320 | false |
| base_pattern_specialist | dollar_volume_ratio_20_60__rank_sector | 0.011646 | 0.011646 | 221600 | 221600 | 44320 | false |
| base_pattern_specialist | distance_from_52w_high__rank_all | 0.011130 | 0.011130 | 221056 | 221056 | 44212 | false |
| base_pattern_specialist | days_since_52w_high | -0.010996 | 0.010996 | 221056 | 221056 | 44212 | false |
| base_pattern_specialist | breakout_volume_ratio_50 | -0.010515 | 0.010515 | 221600 | 221600 | 44320 | false |
| base_pattern_specialist | dollar_volume_ratio_20_60 | -0.009142 | 0.009142 | 221600 | 221600 | 44320 | false |
| base_pattern_specialist | base_volume_dryup_ratio_20 | 0.007523 | 0.007523 | 221600 | 221600 | 44320 | false |
| base_pattern_specialist | base_volume_dryup_ratio_20__rank_sector | 0.007065 | 0.007065 | 221600 | 221600 | 44320 | false |
| base_pattern_specialist | base_volume_dryup_ratio_20__rank_all | 0.007063 | 0.007063 | 221600 | 221600 | 44320 | false |
| base_pattern_specialist | volume_percentile_60 | -0.006721 | 0.006721 | 221600 | 221600 | 44320 | false |
| base_pattern_specialist | base_atr_contraction_20 | -0.006079 | 0.006079 | 221600 | 221600 | 44320 | false |
| base_pattern_specialist | distance_above_20d_high__rank_sector | 0.004282 | 0.004282 | 221600 | 221600 | 44320 | false |
| base_pattern_specialist | volume_percentile_60__rank_all | 0.003637 | 0.003637 | 221600 | 221600 | 44320 | false |
| base_pattern_specialist | distance_from_52w_high | 0.003265 | 0.003265 | 221056 | 221056 | 44212 | false |
| base_pattern_specialist | base_atr_contraction_20__rank_sector | 0.003231 | 0.003231 | 221600 | 221600 | 44320 | false |
| base_pattern_specialist | distance_above_20d_high | -0.003165 | 0.003165 | 221600 | 221600 | 44320 | false |
| base_pattern_specialist | volume_percentile_60__rank_sector | 0.002890 | 0.002890 | 221600 | 221600 | 44320 | false |
| base_pattern_specialist | breakout_volume_ratio_50__rank_sector | 0.002019 | 0.002019 | 221600 | 221600 | 44320 | false |
| base_pattern_specialist | distance_above_20d_high__rank_all | -0.001437 | 0.001437 | 221600 | 221600 | 44320 | false |
| base_pattern_specialist | days_since_failed_breakout_20d | -0.001255 | 0.001255 | 5966 | 5966 | 1194 | false |
| base_pattern_specialist | base_atr_contraction_20__rank_all | 0.001120 | 0.001120 | 221600 | 221600 | 44320 | false |
| base_pattern_specialist | breakout_volume_ratio_50__rank_all | 0.000873 | 0.000873 | 221600 | 221600 | 44320 | false |
| overnight_session_specialist | overnight_ret_20d | -0.207958 | 0.207958 | 6538 | 6538 | 1308 | true |
| overnight_session_specialist | overnight_ret_20d__rank_all | -0.194952 | 0.194952 | 6538 | 6538 | 1308 | true |
| overnight_session_specialist | overnight_ret_20d__rank_sector | -0.149083 | 0.149083 | 6538 | 6538 | 1308 | true |
| overnight_session_specialist | overnight_minus_rth_20d__rank_all | -0.133516 | 0.133516 | 6538 | 6538 | 1308 | true |
| overnight_session_specialist | overnight_minus_rth_20d | -0.121905 | 0.121905 | 6538 | 6538 | 1308 | true |
| overnight_session_specialist | overnight_minus_rth_20d__rank_sector | -0.082253 | 0.082253 | 6538 | 6538 | 1308 | true |
| overnight_session_specialist | rth_ret_20d__rank_all | 0.069065 | 0.069065 | 6538 | 6538 | 1308 | true |
| overnight_session_specialist | rth_ret_20d | 0.050431 | 0.050431 | 6538 | 6538 | 1308 | true |
| overnight_session_specialist | max_gap_down_pct_60__rank_all | 0.050179 | 0.050179 | 221600 | 221600 | 44320 | true |
| overnight_session_specialist | avg_abs_gap_pct_20__rank_all | 0.049238 | 0.049238 | 221600 | 221600 | 44320 | true |
| overnight_session_specialist | max_gap_down_pct_60__rank_sector | 0.041342 | 0.041342 | 221600 | 221600 | 44320 | true |
| overnight_session_specialist | rth_ret_5d__rank_all | 0.039657 | 0.039657 | 6538 | 6538 | 1308 | true |
| overnight_session_specialist | max_gap_down_pct_60 | 0.036235 | 0.036235 | 221600 | 221600 | 44320 | true |
| overnight_session_specialist | rth_ret_20d__rank_sector | 0.034465 | 0.034465 | 6538 | 6538 | 1308 | true |
| overnight_session_specialist | overnight_ret_5d__rank_all | 0.032809 | 0.032809 | 6538 | 6538 | 1308 | true |
| overnight_session_specialist | overnight_ret_5d | 0.030241 | 0.030241 | 6538 | 6538 | 1308 | true |
| overnight_session_specialist | avg_abs_gap_pct_20__rank_sector | 0.029515 | 0.029515 | 221600 | 221600 | 44320 | false |
| overnight_session_specialist | rth_ret_5d__rank_sector | 0.025625 | 0.025625 | 6538 | 6538 | 1308 | false |
| overnight_session_specialist | avg_abs_gap_pct_20 | 0.021427 | 0.021427 | 221600 | 221600 | 44320 | false |
| overnight_session_specialist | overnight_minus_rth_5d | 0.016929 | 0.016929 | 6538 | 6538 | 1308 | false |
| overnight_session_specialist | overnight_minus_rth_5d__rank_all | -0.016248 | 0.016248 | 6538 | 6538 | 1308 | false |
| overnight_session_specialist | overnight_minus_rth_5d__rank_sector | -0.013453 | 0.013453 | 6538 | 6538 | 1308 | false |
| overnight_session_specialist | overnight_ret_5d__rank_sector | 0.007714 | 0.007714 | 6538 | 6538 | 1308 | false |
| overnight_session_specialist | rth_ret_5d | -0.000474 | 0.000474 | 6538 | 6538 | 1308 | false |

## Prediction Spearman Correlation

| model | signal_proxy | structure_factor_signal | overnight_session_specialist | base_pattern_specialist |
|---|---:|---:|---:|---:|
| signal_proxy | 1.000000 | -0.188307 | 0.104579 | 0.084634 |
| structure_factor_signal | -0.188307 | 1.000000 | -0.147429 | -0.147202 |
| overnight_session_specialist | 0.104579 | -0.147429 | 1.000000 | 0.486385 |
| base_pattern_specialist | 0.084634 | -0.147202 | 0.486385 | 1.000000 |

## Decile Tables

| model | decile | dates | avg_rows | mean_target | hit_rate |
|---|---:|---:|---:|---:|---:|
| signal_proxy | 0 | 275 | 2.869091 | 0.023201 | 0.486102 |
| signal_proxy | 1 | 275 | 2.330909 | -0.003956 | 0.436730 |
| signal_proxy | 2 | 275 | 2.250909 | -0.015431 | 0.408568 |
| signal_proxy | 3 | 275 | 2.374545 | -0.010001 | 0.389813 |
| signal_proxy | 4 | 275 | 2.381818 | -0.000553 | 0.484866 |
| signal_proxy | 5 | 275 | 2.210909 | -0.006101 | 0.480353 |
| signal_proxy | 6 | 275 | 2.269091 | 0.005313 | 0.462352 |
| signal_proxy | 7 | 275 | 2.356364 | -0.000846 | 0.471154 |
| signal_proxy | 8 | 275 | 2.225455 | -0.031747 | 0.397492 |
| signal_proxy | 9 | 275 | 2.756364 | -0.015940 | 0.424198 |
| structure_factor_signal | 0 | 275 | 2.869091 | -0.018778 | 0.420750 |
| structure_factor_signal | 1 | 275 | 2.330909 | -0.016247 | 0.420721 |
| structure_factor_signal | 2 | 275 | 2.250909 | -0.019782 | 0.391088 |
| structure_factor_signal | 3 | 275 | 2.374545 | -0.001297 | 0.471470 |
| structure_factor_signal | 4 | 275 | 2.381818 | -0.001499 | 0.478740 |
| structure_factor_signal | 5 | 275 | 2.210909 | 0.006874 | 0.449658 |
| structure_factor_signal | 6 | 275 | 2.269091 | -0.033006 | 0.387873 |
| structure_factor_signal | 7 | 275 | 2.356364 | 0.001683 | 0.464206 |
| structure_factor_signal | 8 | 275 | 2.225455 | 0.004009 | 0.460480 |
| structure_factor_signal | 9 | 275 | 2.756364 | 0.032328 | 0.505817 |
| overnight_session_specialist | 0 | 241 | 2.751037 | 0.030686 | 0.472003 |
| overnight_session_specialist | 1 | 241 | 2.207469 | 0.012542 | 0.518170 |
| overnight_session_specialist | 2 | 241 | 2.120332 | -0.002472 | 0.414189 |
| overnight_session_specialist | 3 | 241 | 2.257261 | -0.005214 | 0.445180 |
| overnight_session_specialist | 4 | 241 | 2.253112 | -0.005755 | 0.440780 |
| overnight_session_specialist | 5 | 241 | 2.078838 | -0.018399 | 0.424218 |
| overnight_session_specialist | 6 | 241 | 2.145228 | -0.025713 | 0.383489 |
| overnight_session_specialist | 7 | 241 | 2.232365 | -0.024189 | 0.405423 |
| overnight_session_specialist | 8 | 241 | 2.095436 | -0.004233 | 0.465439 |
| overnight_session_specialist | 9 | 241 | 2.634855 | -0.008855 | 0.461051 |
| base_pattern_specialist | 0 | 275 | 2.869091 | 0.017970 | 0.472486 |
| base_pattern_specialist | 1 | 275 | 2.330909 | -0.000614 | 0.450318 |
| base_pattern_specialist | 2 | 275 | 2.250909 | -0.009000 | 0.438159 |
| base_pattern_specialist | 3 | 275 | 2.374545 | -0.006932 | 0.476491 |
| base_pattern_specialist | 4 | 275 | 2.381818 | -0.023318 | 0.406430 |
| base_pattern_specialist | 5 | 275 | 2.210909 | -0.013530 | 0.412912 |
| base_pattern_specialist | 6 | 275 | 2.269091 | -0.007572 | 0.434240 |
| base_pattern_specialist | 7 | 275 | 2.356364 | -0.006614 | 0.436235 |
| base_pattern_specialist | 8 | 275 | 2.225455 | -0.006274 | 0.444218 |
| base_pattern_specialist | 9 | 275 | 2.756364 | 0.001812 | 0.473060 |
