# E1 Niche Split

- note: research report only; niche candidates do not change promotion gate policy or scan caps.
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

### signal_proxy
- dates: 514
- avg_pick_count: 1.976654
- gross_mean_target: 0.022357
- net_mean_target: 0.021357
- universe_mean_target: -0.002391
- mean_target_excess: 0.023748
- round_trip_cost: 0.001000
- hit_rate: 0.490272
- universe_hit_rate: 0.451583
- hit_rate_excess: 0.038689
- beat_universe_rate: 0.496109
- spearman: 0.021444
- net_sharpe_ann: 0.247464
- newey_west_t_lag_horizon: 1.195877
- years_for_t_1_96: 62.732033
- positive_date_rate: 0.482490
- ge_2pct_rate: 0.445525
- ge_5pct_rate: 0.385214
- top_ticker: LITE
- top_ticker_date_rate: 0.013619
- top_ticker_pick_share: 0.006890

### overnight_session_specialist
- dates: 407
- avg_pick_count: 1.955774
- gross_mean_target: 0.018009
- net_mean_target: 0.017009
- universe_mean_target: -0.004265
- mean_target_excess: 0.021274
- round_trip_cost: 0.001000
- hit_rate: 0.474201
- universe_hit_rate: 0.444036
- hit_rate_excess: 0.030166
- beat_universe_rate: 0.474201
- spearman: 0.018231
- net_sharpe_ann: 0.208245
- newey_west_t_lag_horizon: 1.037937
- years_for_t_1_96: 88.585226
- positive_date_rate: 0.471744
- ge_2pct_rate: 0.405405
- ge_5pct_rate: 0.348894
- top_ticker: RMBS
- top_ticker_date_rate: 0.014742
- top_ticker_pick_share: 0.007538

### base_pattern_specialist
- dates: 514
- avg_pick_count: 1.976654
- gross_mean_target: 0.014914
- net_mean_target: 0.013914
- universe_mean_target: -0.002391
- mean_target_excess: 0.016305
- round_trip_cost: 0.001000
- hit_rate: 0.458171
- universe_hit_rate: 0.451583
- hit_rate_excess: 0.006588
- beat_universe_rate: 0.478599
- spearman: -0.009893
- net_sharpe_ann: 0.167314
- newey_west_t_lag_horizon: 0.928368
- years_for_t_1_96: 137.230276
- positive_date_rate: 0.474708
- ge_2pct_rate: 0.410506
- ge_5pct_rate: 0.336576
- top_ticker: LITE
- top_ticker_date_rate: 0.011673
- top_ticker_pick_share: 0.005906

### structure_factor_signal
- dates: 514
- avg_pick_count: 1.976654
- gross_mean_target: -0.010974
- net_mean_target: -0.011974
- universe_mean_target: -0.002391
- mean_target_excess: -0.009583
- round_trip_cost: 0.001000
- hit_rate: 0.447471
- universe_hit_rate: 0.451583
- hit_rate_excess: -0.004112
- beat_universe_rate: 0.447471
- spearman: -0.016232
- net_sharpe_ann: -0.224335
- newey_west_t_lag_horizon: -1.152286
- years_for_t_1_96: 76.333730
- positive_date_rate: 0.441634
- ge_2pct_rate: 0.369650
- ge_5pct_rate: 0.254864
- top_ticker: L
- top_ticker_date_rate: 0.011673
- top_ticker_pick_share: 0.005906

## Acceptance Windows

### overnight_session_specialist_trailing_3folds
- dates: 60
- avg_pick_count: 1.983333
- gross_mean_target: 0.032237
- net_mean_target: 0.031237
- universe_mean_target: -0.000737
- mean_target_excess: 0.031974
- round_trip_cost: 0.001000
- hit_rate: 0.458333
- universe_hit_rate: 0.456815
- hit_rate_excess: 0.001519
- beat_universe_rate: 0.500000
- spearman: -0.021250
- net_sharpe_ann: 0.304324
- newey_west_t_lag_horizon: 0.800638
- years_for_t_1_96: 41.479997
- positive_date_rate: 0.450000
- ge_2pct_rate: 0.383333
- ge_5pct_rate: 0.366667
- top_ticker: VAL
- top_ticker_date_rate: 0.033333
- top_ticker_pick_share: 0.016807

### signal_proxy_full_oos
- dates: 514
- avg_pick_count: 1.976654
- gross_mean_target: 0.022357
- net_mean_target: 0.021357
- universe_mean_target: -0.002391
- mean_target_excess: 0.023748
- round_trip_cost: 0.001000
- hit_rate: 0.490272
- universe_hit_rate: 0.451583
- hit_rate_excess: 0.038689
- beat_universe_rate: 0.496109
- spearman: 0.021444
- net_sharpe_ann: 0.247464
- newey_west_t_lag_horizon: 1.195877
- years_for_t_1_96: 62.732033
- positive_date_rate: 0.482490
- ge_2pct_rate: 0.445525
- ge_5pct_rate: 0.385214
- top_ticker: LITE
- top_ticker_date_rate: 0.013619
- top_ticker_pick_share: 0.006890

### overnight_session_specialist_full_oos
- dates: 407
- avg_pick_count: 1.955774
- gross_mean_target: 0.018009
- net_mean_target: 0.017009
- universe_mean_target: -0.004265
- mean_target_excess: 0.021274
- round_trip_cost: 0.001000
- hit_rate: 0.474201
- universe_hit_rate: 0.444036
- hit_rate_excess: 0.030166
- beat_universe_rate: 0.474201
- spearman: 0.018231
- net_sharpe_ann: 0.208245
- newey_west_t_lag_horizon: 1.037937
- years_for_t_1_96: 88.585226
- positive_date_rate: 0.471744
- ge_2pct_rate: 0.405405
- ge_5pct_rate: 0.348894
- top_ticker: RMBS
- top_ticker_date_rate: 0.014742
- top_ticker_pick_share: 0.007538

### base_pattern_specialist_full_oos
- dates: 514
- avg_pick_count: 1.976654
- gross_mean_target: 0.014914
- net_mean_target: 0.013914
- universe_mean_target: -0.002391
- mean_target_excess: 0.016305
- round_trip_cost: 0.001000
- hit_rate: 0.458171
- universe_hit_rate: 0.451583
- hit_rate_excess: 0.006588
- beat_universe_rate: 0.478599
- spearman: -0.009893
- net_sharpe_ann: 0.167314
- newey_west_t_lag_horizon: 0.928368
- years_for_t_1_96: 137.230276
- positive_date_rate: 0.474708
- ge_2pct_rate: 0.410506
- ge_5pct_rate: 0.336576
- top_ticker: LITE
- top_ticker_date_rate: 0.011673
- top_ticker_pick_share: 0.005906

### structure_factor_signal_trailing_3folds
- dates: 60
- avg_pick_count: 1.983333
- gross_mean_target: 0.001105
- net_mean_target: 0.000105
- universe_mean_target: 0.000871
- mean_target_excess: -0.000766
- round_trip_cost: 0.001000
- hit_rate: 0.466667
- universe_hit_rate: 0.468610
- hit_rate_excess: -0.001944
- beat_universe_rate: 0.366667
- spearman: 0.133004
- net_sharpe_ann: 0.001762
- newey_west_t_lag_horizon: 0.005847
- years_for_t_1_96: 1237391.626685
- positive_date_rate: 0.433333
- ge_2pct_rate: 0.366667
- ge_5pct_rate: 0.233333
- top_ticker: USFD
- top_ticker_date_rate: 0.016667
- top_ticker_pick_share: 0.008403

### base_pattern_specialist_trailing_3folds
- dates: 60
- avg_pick_count: 1.983333
- gross_mean_target: -0.005123
- net_mean_target: -0.006123
- universe_mean_target: 0.000871
- mean_target_excess: -0.006994
- round_trip_cost: 0.001000
- hit_rate: 0.408333
- universe_hit_rate: 0.468610
- hit_rate_excess: -0.060277
- beat_universe_rate: 0.366667
- spearman: -0.141077
- net_sharpe_ann: -0.066649
- newey_west_t_lag_horizon: -0.245983
- years_for_t_1_96: 864.825009
- positive_date_rate: 0.416667
- ge_2pct_rate: 0.366667
- ge_5pct_rate: 0.300000
- top_ticker: JBLU
- top_ticker_date_rate: 0.033333
- top_ticker_pick_share: 0.016807

### signal_proxy_last_fold
- dates: 20
- avg_pick_count: 2.000000
- gross_mean_target: -0.007807
- net_mean_target: -0.008807
- universe_mean_target: -0.028917
- mean_target_excess: 0.020111
- round_trip_cost: 0.001000
- hit_rate: 0.475000
- universe_hit_rate: 0.409124
- hit_rate_excess: 0.065876
- beat_universe_rate: 0.450000
- spearman: -0.046240
- net_sharpe_ann: -0.128412
- newey_west_t_lag_horizon: -0.246001
- years_for_t_1_96: 232.971362
- positive_date_rate: 0.300000
- ge_2pct_rate: 0.300000
- ge_5pct_rate: 0.250000
- top_ticker: CVLT
- top_ticker_date_rate: 0.050000
- top_ticker_pick_share: 0.025000

### signal_proxy_trailing_3folds
- dates: 60
- avg_pick_count: 1.983333
- gross_mean_target: -0.009168
- net_mean_target: -0.010168
- universe_mean_target: 0.000871
- mean_target_excess: -0.011040
- round_trip_cost: 0.001000
- hit_rate: 0.433333
- universe_hit_rate: 0.468610
- hit_rate_excess: -0.035277
- beat_universe_rate: 0.366667
- spearman: -0.089275
- net_sharpe_ann: -0.106422
- newey_west_t_lag_horizon: -0.436585
- years_for_t_1_96: 339.192493
- positive_date_rate: 0.333333
- ge_2pct_rate: 0.300000
- ge_5pct_rate: 0.266667
- top_ticker: WT
- top_ticker_date_rate: 0.033333
- top_ticker_pick_share: 0.016807

### structure_factor_signal_full_oos
- dates: 514
- avg_pick_count: 1.976654
- gross_mean_target: -0.010974
- net_mean_target: -0.011974
- universe_mean_target: -0.002391
- mean_target_excess: -0.009583
- round_trip_cost: 0.001000
- hit_rate: 0.447471
- universe_hit_rate: 0.451583
- hit_rate_excess: -0.004112
- beat_universe_rate: 0.447471
- spearman: -0.016232
- net_sharpe_ann: -0.224335
- newey_west_t_lag_horizon: -1.152286
- years_for_t_1_96: 76.333730
- positive_date_rate: 0.441634
- ge_2pct_rate: 0.369650
- ge_5pct_rate: 0.254864
- top_ticker: L
- top_ticker_date_rate: 0.011673
- top_ticker_pick_share: 0.005906

### structure_factor_signal_last_fold
- dates: 20
- avg_pick_count: 2.000000
- gross_mean_target: -0.043192
- net_mean_target: -0.044192
- universe_mean_target: -0.028917
- mean_target_excess: -0.015275
- round_trip_cost: 0.001000
- hit_rate: 0.350000
- universe_hit_rate: 0.409124
- hit_rate_excess: -0.059124
- beat_universe_rate: 0.250000
- spearman: 0.084314
- net_sharpe_ann: -0.980781
- newey_west_t_lag_horizon: -3.619883
- years_for_t_1_96: 3.993634
- positive_date_rate: 0.250000
- ge_2pct_rate: 0.250000
- ge_5pct_rate: 0.100000
- top_ticker: IBOC
- top_ticker_date_rate: 0.050000
- top_ticker_pick_share: 0.025000

### base_pattern_specialist_last_fold
- dates: 20
- avg_pick_count: 2.000000
- gross_mean_target: -0.050137
- net_mean_target: -0.051137
- universe_mean_target: -0.028917
- mean_target_excess: -0.022220
- round_trip_cost: 0.001000
- hit_rate: 0.425000
- universe_hit_rate: 0.409124
- hit_rate_excess: 0.015876
- beat_universe_rate: 0.300000
- spearman: -0.119388
- net_sharpe_ann: -0.922385
- newey_west_t_lag_horizon: -4.321644
- years_for_t_1_96: 4.515308
- positive_date_rate: 0.250000
- ge_2pct_rate: 0.150000
- ge_5pct_rate: 0.150000
- top_ticker: MUR
- top_ticker_date_rate: 0.050000
- top_ticker_pick_share: 0.025000

### overnight_session_specialist_last_fold
- dates: 20
- avg_pick_count: 2.000000
- gross_mean_target: -0.056508
- net_mean_target: -0.057508
- universe_mean_target: 0.005671
- mean_target_excess: -0.063180
- round_trip_cost: 0.001000
- hit_rate: 0.400000
- universe_hit_rate: 0.475954
- hit_rate_excess: -0.075954
- beat_universe_rate: 0.350000
- spearman: -0.175100
- net_sharpe_ann: -0.773943
- newey_west_t_lag_horizon: -2.831019
- years_for_t_1_96: 6.413494
- positive_date_rate: 0.300000
- ge_2pct_rate: 0.250000
- ge_5pct_rate: 0.250000
- top_ticker: AAON
- top_ticker_date_rate: 0.050000
- top_ticker_pick_share: 0.025000

## Niche Feature IC Diagnostics

| model | feature | rank_ic | abs_rank_ic | observations | family_rows | min_observations | survives |
|---|---|---:|---:|---:|---:|---:|---|
| base_pattern_specialist | failed_breakout_20d__rank_all | 0.107521 | 0.107521 | 5311 | 5311 | 1063 | true |
| base_pattern_specialist | failed_breakout_52w__rank_all | -0.074079 | 0.074079 | 5311 | 5311 | 1063 | true |
| base_pattern_specialist | failed_breakout_20d__rank_sector | -0.062572 | 0.062572 | 5311 | 5311 | 1063 | true |
| base_pattern_specialist | failed_breakout_52w | -0.061754 | 0.061754 | 5311 | 5311 | 1063 | true |
| base_pattern_specialist | days_since_failed_breakout_52w__rank_sector | 0.055428 | 0.055428 | 3304 | 3304 | 661 | true |
| base_pattern_specialist | days_since_failed_breakout_20d__rank_sector | 0.051756 | 0.051756 | 4869 | 4869 | 974 | true |
| base_pattern_specialist | days_since_failed_breakout_20d__rank_all | 0.049074 | 0.049074 | 4869 | 4869 | 974 | true |
| base_pattern_specialist | base_range_pct_20__rank_all | 0.041795 | 0.041795 | 220373 | 220373 | 44075 | true |
| base_pattern_specialist | close_vs_20d_low__rank_all | 0.039372 | 0.039372 | 220373 | 220373 | 44075 | true |
| base_pattern_specialist | close_vs_20d_low__rank_sector | 0.031306 | 0.031306 | 220373 | 220373 | 44075 | true |
| base_pattern_specialist | base_range_pct_20__rank_sector | 0.028764 | 0.028764 | 220373 | 220373 | 44075 | false |
| base_pattern_specialist | days_since_failed_breakout_52w | -0.026716 | 0.026716 | 3304 | 3304 | 661 | false |
| base_pattern_specialist | days_since_failed_breakout_52w__rank_all | 0.024421 | 0.024421 | 3304 | 3304 | 661 | false |
| base_pattern_specialist | days_since_52w_high__rank_sector | -0.024003 | 0.024003 | 219829 | 219829 | 43966 | false |
| base_pattern_specialist | distance_from_52w_high__rank_sector | 0.020510 | 0.020510 | 219829 | 219829 | 43966 | false |
| base_pattern_specialist | days_since_52w_high__rank_all | -0.018451 | 0.018451 | 219829 | 219829 | 43966 | false |
| base_pattern_specialist | base_range_pct_20 | 0.018282 | 0.018282 | 220373 | 220373 | 44075 | false |
| base_pattern_specialist | close_vs_20d_low | 0.014298 | 0.014298 | 220373 | 220373 | 44075 | false |
| base_pattern_specialist | failed_breakout_52w__rank_sector | -0.013787 | 0.013787 | 5311 | 5311 | 1063 | false |
| base_pattern_specialist | dollar_volume_ratio_20_60__rank_all | 0.012525 | 0.012525 | 220373 | 220373 | 44075 | false |
| base_pattern_specialist | dollar_volume_ratio_20_60__rank_sector | 0.011926 | 0.011926 | 220373 | 220373 | 44075 | false |
| base_pattern_specialist | days_since_52w_high | -0.011256 | 0.011256 | 219829 | 219829 | 43966 | false |
| base_pattern_specialist | breakout_volume_ratio_50 | -0.010916 | 0.010916 | 220373 | 220373 | 44075 | false |
| base_pattern_specialist | distance_from_52w_high__rank_all | 0.010505 | 0.010505 | 219829 | 219829 | 43966 | false |
| base_pattern_specialist | dollar_volume_ratio_20_60 | -0.008882 | 0.008882 | 220373 | 220373 | 44075 | false |
| base_pattern_specialist | base_volume_dryup_ratio_20 | 0.007259 | 0.007259 | 220373 | 220373 | 44075 | false |
| base_pattern_specialist | base_volume_dryup_ratio_20__rank_all | 0.007075 | 0.007075 | 220373 | 220373 | 44075 | false |
| base_pattern_specialist | volume_percentile_60 | -0.007025 | 0.007025 | 220373 | 220373 | 44075 | false |
| base_pattern_specialist | base_volume_dryup_ratio_20__rank_sector | 0.006744 | 0.006744 | 220373 | 220373 | 44075 | false |
| base_pattern_specialist | base_atr_contraction_20 | -0.005517 | 0.005517 | 220373 | 220373 | 44075 | false |
| base_pattern_specialist | distance_above_20d_high | -0.004462 | 0.004462 | 220373 | 220373 | 44075 | false |
| base_pattern_specialist | base_atr_contraction_20__rank_sector | 0.003645 | 0.003645 | 220373 | 220373 | 44075 | false |
| base_pattern_specialist | distance_above_20d_high__rank_sector | 0.003602 | 0.003602 | 220373 | 220373 | 44075 | false |
| base_pattern_specialist | volume_percentile_60__rank_all | 0.003564 | 0.003564 | 220373 | 220373 | 44075 | false |
| base_pattern_specialist | volume_percentile_60__rank_sector | 0.002846 | 0.002846 | 220373 | 220373 | 44075 | false |
| base_pattern_specialist | distance_above_20d_high__rank_all | -0.002628 | 0.002628 | 220373 | 220373 | 44075 | false |
| base_pattern_specialist | distance_from_52w_high | 0.002575 | 0.002575 | 219829 | 219829 | 43966 | false |
| base_pattern_specialist | breakout_volume_ratio_50__rank_sector | 0.001933 | 0.001933 | 220373 | 220373 | 44075 | false |
| base_pattern_specialist | base_atr_contraction_20__rank_all | 0.001739 | 0.001739 | 220373 | 220373 | 44075 | false |
| base_pattern_specialist | days_since_failed_breakout_20d | -0.001605 | 0.001605 | 4869 | 4869 | 974 | false |
| base_pattern_specialist | failed_breakout_20d | 0.001365 | 0.001365 | 5311 | 5311 | 1063 | false |
| base_pattern_specialist | breakout_volume_ratio_50__rank_all | 0.000796 | 0.000796 | 220373 | 220373 | 44075 | false |
| overnight_session_specialist | overnight_ret_20d | -0.224482 | 0.224482 | 5311 | 5311 | 1063 | true |
| overnight_session_specialist | overnight_ret_20d__rank_all | -0.219959 | 0.219959 | 5311 | 5311 | 1063 | true |
| overnight_session_specialist | overnight_ret_20d__rank_sector | -0.168823 | 0.168823 | 5311 | 5311 | 1063 | true |
| overnight_session_specialist | overnight_minus_rth_20d__rank_all | -0.126466 | 0.126466 | 5311 | 5311 | 1063 | true |
| overnight_session_specialist | overnight_minus_rth_20d | -0.117171 | 0.117171 | 5311 | 5311 | 1063 | true |
| overnight_session_specialist | overnight_minus_rth_20d__rank_sector | -0.071386 | 0.071386 | 5311 | 5311 | 1063 | true |
| overnight_session_specialist | max_gap_down_pct_60__rank_all | 0.051075 | 0.051075 | 220373 | 220373 | 44075 | true |
| overnight_session_specialist | avg_abs_gap_pct_20__rank_all | 0.050263 | 0.050263 | 220373 | 220373 | 44075 | true |
| overnight_session_specialist | rth_ret_20d__rank_all | 0.048711 | 0.048711 | 5311 | 5311 | 1063 | true |
| overnight_session_specialist | overnight_ret_5d__rank_all | 0.043859 | 0.043859 | 5311 | 5311 | 1063 | true |
| overnight_session_specialist | max_gap_down_pct_60__rank_sector | 0.041733 | 0.041733 | 220373 | 220373 | 44075 | true |
| overnight_session_specialist | rth_ret_20d | 0.037699 | 0.037699 | 5311 | 5311 | 1063 | true |
| overnight_session_specialist | max_gap_down_pct_60 | 0.037073 | 0.037073 | 220373 | 220373 | 44075 | true |
| overnight_session_specialist | overnight_ret_5d | 0.033950 | 0.033950 | 5311 | 5311 | 1063 | true |
| overnight_session_specialist | avg_abs_gap_pct_20__rank_sector | 0.030038 | 0.030038 | 220373 | 220373 | 44075 | true |
| overnight_session_specialist | overnight_minus_rth_5d | 0.028640 | 0.028640 | 5311 | 5311 | 1063 | false |
| overnight_session_specialist | avg_abs_gap_pct_20 | 0.022398 | 0.022398 | 220373 | 220373 | 44075 | false |
| overnight_session_specialist | rth_ret_5d__rank_all | 0.017724 | 0.017724 | 5311 | 5311 | 1063 | false |
| overnight_session_specialist | rth_ret_5d | -0.016705 | 0.016705 | 5311 | 5311 | 1063 | false |
| overnight_session_specialist | overnight_ret_5d__rank_sector | 0.013322 | 0.013322 | 5311 | 5311 | 1063 | false |
| overnight_session_specialist | rth_ret_20d__rank_sector | 0.012340 | 0.012340 | 5311 | 5311 | 1063 | false |
| overnight_session_specialist | overnight_minus_rth_5d__rank_sector | 0.007306 | 0.007306 | 5311 | 5311 | 1063 | false |
| overnight_session_specialist | overnight_minus_rth_5d__rank_all | 0.002912 | 0.002912 | 5311 | 5311 | 1063 | false |
| overnight_session_specialist | rth_ret_5d__rank_sector | -0.002692 | 0.002692 | 5311 | 5311 | 1063 | false |

## Prediction Spearman Correlation

| model | signal_proxy | structure_factor_signal | overnight_session_specialist | base_pattern_specialist |
|---|---:|---:|---:|---:|
| signal_proxy | 1.000000 | -0.186821 | 0.099697 | 0.084131 |
| structure_factor_signal | -0.186821 | 1.000000 | -0.128696 | -0.146046 |
| overnight_session_specialist | 0.099697 | -0.128696 | 1.000000 | 0.492680 |
| base_pattern_specialist | 0.084131 | -0.146046 | 0.492680 | 1.000000 |

## Decile Tables

| model | decile | dates | avg_rows | mean_target | hit_rate |
|---|---:|---:|---:|---:|---:|
| signal_proxy | 0 | 274 | 2.872263 | 0.023639 | 0.486051 |
| signal_proxy | 1 | 274 | 2.335766 | -0.003828 | 0.438323 |
| signal_proxy | 2 | 274 | 2.255474 | -0.015044 | 0.410060 |
| signal_proxy | 3 | 274 | 2.379562 | -0.009553 | 0.391236 |
| signal_proxy | 4 | 274 | 2.386861 | -0.000258 | 0.486635 |
| signal_proxy | 5 | 274 | 2.215328 | -0.005819 | 0.482106 |
| signal_proxy | 6 | 274 | 2.273723 | 0.005215 | 0.460390 |
| signal_proxy | 7 | 274 | 2.361314 | -0.000546 | 0.472874 |
| signal_proxy | 8 | 274 | 2.229927 | -0.030987 | 0.398943 |
| signal_proxy | 9 | 274 | 2.759124 | -0.015396 | 0.425746 |
| structure_factor_signal | 0 | 274 | 2.872263 | -0.018543 | 0.422285 |
| structure_factor_signal | 1 | 274 | 2.335766 | -0.016201 | 0.422256 |
| structure_factor_signal | 2 | 274 | 2.255474 | -0.019711 | 0.392515 |
| structure_factor_signal | 3 | 274 | 2.379562 | -0.000817 | 0.473191 |
| structure_factor_signal | 4 | 274 | 2.386861 | -0.001207 | 0.480487 |
| structure_factor_signal | 5 | 274 | 2.215328 | 0.006782 | 0.447649 |
| structure_factor_signal | 6 | 274 | 2.273723 | -0.032683 | 0.389288 |
| structure_factor_signal | 7 | 274 | 2.361314 | 0.001313 | 0.462250 |
| structure_factor_signal | 8 | 274 | 2.229927 | 0.004900 | 0.462161 |
| structure_factor_signal | 9 | 274 | 2.759124 | 0.033536 | 0.507663 |
| overnight_session_specialist | 0 | 228 | 3.250000 | 0.022554 | 0.471866 |
| overnight_session_specialist | 1 | 228 | 2.723684 | 0.012552 | 0.480154 |
| overnight_session_specialist | 2 | 228 | 2.618421 | 0.000722 | 0.442814 |
| overnight_session_specialist | 3 | 228 | 2.758772 | -0.009157 | 0.441414 |
| overnight_session_specialist | 4 | 228 | 2.807018 | -0.004430 | 0.422725 |
| overnight_session_specialist | 5 | 228 | 2.561404 | -0.020031 | 0.410643 |
| overnight_session_specialist | 6 | 228 | 2.657895 | -0.027105 | 0.417994 |
| overnight_session_specialist | 7 | 228 | 2.719298 | -0.021041 | 0.412357 |
| overnight_session_specialist | 8 | 228 | 2.622807 | -0.005329 | 0.454212 |
| overnight_session_specialist | 9 | 228 | 3.153509 | -0.016258 | 0.441885 |
| base_pattern_specialist | 0 | 274 | 2.872263 | 0.018733 | 0.474210 |
| base_pattern_specialist | 1 | 274 | 2.335766 | -0.000131 | 0.451962 |
| base_pattern_specialist | 2 | 274 | 2.255474 | -0.007951 | 0.439758 |
| base_pattern_specialist | 3 | 274 | 2.379562 | -0.006514 | 0.478230 |
| base_pattern_specialist | 4 | 274 | 2.386861 | -0.023101 | 0.407913 |
| base_pattern_specialist | 5 | 274 | 2.215328 | -0.013955 | 0.410769 |
| base_pattern_specialist | 6 | 274 | 2.273723 | -0.007494 | 0.435825 |
| base_pattern_specialist | 7 | 274 | 2.361314 | -0.005763 | 0.437827 |
| base_pattern_specialist | 8 | 274 | 2.229927 | -0.005993 | 0.445839 |
| base_pattern_specialist | 9 | 274 | 2.759124 | 0.001831 | 0.472962 |
