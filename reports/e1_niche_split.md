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
- dates: 408
- avg_pick_count: 1.955882
- gross_mean_target: 0.018633
- net_mean_target: 0.017633
- universe_mean_target: -0.004237
- mean_target_excess: 0.021869
- round_trip_cost: 0.001000
- hit_rate: 0.475490
- universe_hit_rate: 0.444185
- hit_rate_excess: 0.031305
- beat_universe_rate: 0.475490
- spearman: 0.018104
- net_sharpe_ann: 0.215532
- newey_west_t_lag_horizon: 1.085500
- years_for_t_1_96: 82.696765
- positive_date_rate: 0.473039
- ge_2pct_rate: 0.406863
- ge_5pct_rate: 0.350490
- top_ticker: RMBS
- top_ticker_date_rate: 0.014706
- top_ticker_pick_share: 0.007519

### signal_proxy
- dates: 408
- avg_pick_count: 1.955882
- gross_mean_target: 0.017249
- net_mean_target: 0.016249
- universe_mean_target: -0.004237
- mean_target_excess: 0.020486
- round_trip_cost: 0.001000
- hit_rate: 0.470588
- universe_hit_rate: 0.444185
- hit_rate_excess: 0.026403
- beat_universe_rate: 0.492647
- spearman: 0.008635
- net_sharpe_ann: 0.185678
- newey_west_t_lag_horizon: 0.770969
- years_for_t_1_96: 111.427022
- positive_date_rate: 0.492647
- ge_2pct_rate: 0.448529
- ge_5pct_rate: 0.375000
- top_ticker: CNR
- top_ticker_date_rate: 0.012255
- top_ticker_pick_share: 0.006266

### base_pattern_specialist
- dates: 408
- avg_pick_count: 1.955882
- gross_mean_target: 0.012003
- net_mean_target: 0.011003
- universe_mean_target: -0.004237
- mean_target_excess: 0.015240
- round_trip_cost: 0.001000
- hit_rate: 0.450980
- universe_hit_rate: 0.444185
- hit_rate_excess: 0.006796
- beat_universe_rate: 0.460784
- spearman: -0.009326
- net_sharpe_ann: 0.132989
- newey_west_t_lag_horizon: 0.615026
- years_for_t_1_96: 217.210452
- positive_date_rate: 0.470588
- ge_2pct_rate: 0.424020
- ge_5pct_rate: 0.328431
- top_ticker: TLN
- top_ticker_date_rate: 0.012255
- top_ticker_pick_share: 0.006266

### structure_factor_signal
- dates: 408
- avg_pick_count: 1.955882
- gross_mean_target: -0.008142
- net_mean_target: -0.009142
- universe_mean_target: -0.004237
- mean_target_excess: -0.004905
- round_trip_cost: 0.001000
- hit_rate: 0.444853
- universe_hit_rate: 0.444185
- hit_rate_excess: 0.000668
- beat_universe_rate: 0.441176
- spearman: -0.008227
- net_sharpe_ann: -0.177941
- newey_west_t_lag_horizon: -1.041762
- years_for_t_1_96: 121.327404
- positive_date_rate: 0.460784
- ge_2pct_rate: 0.357843
- ge_5pct_rate: 0.242647
- top_ticker: L
- top_ticker_date_rate: 0.014706
- top_ticker_pick_share: 0.007519

## Acceptance Windows

### overnight_session_specialist_trailing_3folds
- dates: 60
- avg_pick_count: 1.983333
- gross_mean_target: 0.030627
- net_mean_target: 0.029627
- universe_mean_target: -0.000213
- mean_target_excess: 0.029840
- round_trip_cost: 0.001000
- hit_rate: 0.466667
- universe_hit_rate: 0.460469
- hit_rate_excess: 0.006198
- beat_universe_rate: 0.500000
- spearman: -0.029678
- net_sharpe_ann: 0.291775
- newey_west_t_lag_horizon: 0.907418
- years_for_t_1_96: 45.124758
- positive_date_rate: 0.450000
- ge_2pct_rate: 0.383333
- ge_5pct_rate: 0.366667
- top_ticker: VAL
- top_ticker_date_rate: 0.033333
- top_ticker_pick_share: 0.016807

### base_pattern_specialist_trailing_3folds
- dates: 60
- avg_pick_count: 1.983333
- gross_mean_target: 0.026098
- net_mean_target: 0.025098
- universe_mean_target: -0.000213
- mean_target_excess: 0.025311
- round_trip_cost: 0.001000
- hit_rate: 0.425000
- universe_hit_rate: 0.460469
- hit_rate_excess: -0.035469
- beat_universe_rate: 0.516667
- spearman: -0.051711
- net_sharpe_ann: 0.216181
- newey_west_t_lag_horizon: 1.005213
- years_for_t_1_96: 82.200953
- positive_date_rate: 0.450000
- ge_2pct_rate: 0.450000
- ge_5pct_rate: 0.366667
- top_ticker: VAL
- top_ticker_date_rate: 0.033333
- top_ticker_pick_share: 0.016807

### overnight_session_specialist_full_oos
- dates: 408
- avg_pick_count: 1.955882
- gross_mean_target: 0.018633
- net_mean_target: 0.017633
- universe_mean_target: -0.004237
- mean_target_excess: 0.021869
- round_trip_cost: 0.001000
- hit_rate: 0.475490
- universe_hit_rate: 0.444185
- hit_rate_excess: 0.031305
- beat_universe_rate: 0.475490
- spearman: 0.018104
- net_sharpe_ann: 0.215532
- newey_west_t_lag_horizon: 1.085500
- years_for_t_1_96: 82.696765
- positive_date_rate: 0.473039
- ge_2pct_rate: 0.406863
- ge_5pct_rate: 0.350490
- top_ticker: RMBS
- top_ticker_date_rate: 0.014706
- top_ticker_pick_share: 0.007519

### signal_proxy_full_oos
- dates: 408
- avg_pick_count: 1.955882
- gross_mean_target: 0.017249
- net_mean_target: 0.016249
- universe_mean_target: -0.004237
- mean_target_excess: 0.020486
- round_trip_cost: 0.001000
- hit_rate: 0.470588
- universe_hit_rate: 0.444185
- hit_rate_excess: 0.026403
- beat_universe_rate: 0.492647
- spearman: 0.008635
- net_sharpe_ann: 0.185678
- newey_west_t_lag_horizon: 0.770969
- years_for_t_1_96: 111.427022
- positive_date_rate: 0.492647
- ge_2pct_rate: 0.448529
- ge_5pct_rate: 0.375000
- top_ticker: CNR
- top_ticker_date_rate: 0.012255
- top_ticker_pick_share: 0.006266

### base_pattern_specialist_full_oos
- dates: 408
- avg_pick_count: 1.955882
- gross_mean_target: 0.012003
- net_mean_target: 0.011003
- universe_mean_target: -0.004237
- mean_target_excess: 0.015240
- round_trip_cost: 0.001000
- hit_rate: 0.450980
- universe_hit_rate: 0.444185
- hit_rate_excess: 0.006796
- beat_universe_rate: 0.460784
- spearman: -0.009326
- net_sharpe_ann: 0.132989
- newey_west_t_lag_horizon: 0.615026
- years_for_t_1_96: 217.210452
- positive_date_rate: 0.470588
- ge_2pct_rate: 0.424020
- ge_5pct_rate: 0.328431
- top_ticker: TLN
- top_ticker_date_rate: 0.012255
- top_ticker_pick_share: 0.006266

### structure_factor_signal_full_oos
- dates: 408
- avg_pick_count: 1.955882
- gross_mean_target: -0.008142
- net_mean_target: -0.009142
- universe_mean_target: -0.004237
- mean_target_excess: -0.004905
- round_trip_cost: 0.001000
- hit_rate: 0.444853
- universe_hit_rate: 0.444185
- hit_rate_excess: 0.000668
- beat_universe_rate: 0.441176
- spearman: -0.008227
- net_sharpe_ann: -0.177941
- newey_west_t_lag_horizon: -1.041762
- years_for_t_1_96: 121.327404
- positive_date_rate: 0.460784
- ge_2pct_rate: 0.357843
- ge_5pct_rate: 0.242647
- top_ticker: L
- top_ticker_date_rate: 0.014706
- top_ticker_pick_share: 0.007519

### signal_proxy_trailing_3folds
- dates: 60
- avg_pick_count: 1.983333
- gross_mean_target: -0.013967
- net_mean_target: -0.014967
- universe_mean_target: -0.000213
- mean_target_excess: -0.014754
- round_trip_cost: 0.001000
- hit_rate: 0.408333
- universe_hit_rate: 0.460469
- hit_rate_excess: -0.052135
- beat_universe_rate: 0.433333
- spearman: -0.009712
- net_sharpe_ann: -0.141801
- newey_west_t_lag_horizon: -0.738058
- years_for_t_1_96: 191.051705
- positive_date_rate: 0.383333
- ge_2pct_rate: 0.366667
- ge_5pct_rate: 0.300000
- top_ticker: MATX
- top_ticker_date_rate: 0.033333
- top_ticker_pick_share: 0.016807

### structure_factor_signal_last_fold
- dates: 20
- avg_pick_count: 2.000000
- gross_mean_target: -0.014618
- net_mean_target: -0.015618
- universe_mean_target: 0.005931
- mean_target_excess: -0.021549
- round_trip_cost: 0.001000
- hit_rate: 0.350000
- universe_hit_rate: 0.476201
- hit_rate_excess: -0.126201
- beat_universe_rate: 0.350000
- spearman: 0.068152
- net_sharpe_ann: -0.252808
- newey_west_t_lag_horizon: -0.679100
- years_for_t_1_96: 60.107951
- positive_date_rate: 0.300000
- ge_2pct_rate: 0.200000
- ge_5pct_rate: 0.150000
- top_ticker: FOX
- top_ticker_date_rate: 0.050000
- top_ticker_pick_share: 0.025000

### structure_factor_signal_trailing_3folds
- dates: 60
- avg_pick_count: 1.983333
- gross_mean_target: -0.026767
- net_mean_target: -0.027767
- universe_mean_target: -0.000213
- mean_target_excess: -0.027554
- round_trip_cost: 0.001000
- hit_rate: 0.416667
- universe_hit_rate: 0.460469
- hit_rate_excess: -0.043802
- beat_universe_rate: 0.333333
- spearman: -0.009813
- net_sharpe_ann: -0.478578
- newey_west_t_lag_horizon: -1.799197
- years_for_t_1_96: 16.772847
- positive_date_rate: 0.416667
- ge_2pct_rate: 0.283333
- ge_5pct_rate: 0.200000
- top_ticker: HR
- top_ticker_date_rate: 0.033333
- top_ticker_pick_share: 0.016807

### base_pattern_specialist_last_fold
- dates: 20
- avg_pick_count: 2.000000
- gross_mean_target: -0.028548
- net_mean_target: -0.029548
- universe_mean_target: 0.005931
- mean_target_excess: -0.035479
- round_trip_cost: 0.001000
- hit_rate: 0.375000
- universe_hit_rate: 0.476201
- hit_rate_excess: -0.101201
- beat_universe_rate: 0.400000
- spearman: -0.131451
- net_sharpe_ann: -0.300996
- newey_west_t_lag_horizon: -1.522308
- years_for_t_1_96: 42.402421
- positive_date_rate: 0.400000
- ge_2pct_rate: 0.400000
- ge_5pct_rate: 0.300000
- top_ticker: SM
- top_ticker_date_rate: 0.050000
- top_ticker_pick_share: 0.025000

### overnight_session_specialist_last_fold
- dates: 20
- avg_pick_count: 2.000000
- gross_mean_target: -0.028933
- net_mean_target: -0.029933
- universe_mean_target: 0.005931
- mean_target_excess: -0.035865
- round_trip_cost: 0.001000
- hit_rate: 0.450000
- universe_hit_rate: 0.476201
- hit_rate_excess: -0.026201
- beat_universe_rate: 0.400000
- spearman: -0.146794
- net_sharpe_ann: -0.384377
- newey_west_t_lag_horizon: -1.866841
- years_for_t_1_96: 26.001377
- positive_date_rate: 0.350000
- ge_2pct_rate: 0.300000
- ge_5pct_rate: 0.300000
- top_ticker: RMBS
- top_ticker_date_rate: 0.050000
- top_ticker_pick_share: 0.025000

### signal_proxy_last_fold
- dates: 20
- avg_pick_count: 2.000000
- gross_mean_target: -0.067976
- net_mean_target: -0.068976
- universe_mean_target: 0.005931
- mean_target_excess: -0.074908
- round_trip_cost: 0.001000
- hit_rate: 0.400000
- universe_hit_rate: 0.476201
- hit_rate_excess: -0.076201
- beat_universe_rate: 0.400000
- spearman: -0.176172
- net_sharpe_ann: -0.856583
- newey_west_t_lag_horizon: -3.864372
- years_for_t_1_96: 5.235680
- positive_date_rate: 0.350000
- ge_2pct_rate: 0.300000
- ge_5pct_rate: 0.200000
- top_ticker: SM
- top_ticker_date_rate: 0.050000
- top_ticker_pick_share: 0.025000

## Niche Feature IC Diagnostics

| model | feature | rank_ic | abs_rank_ic | observations | family_rows | min_observations | survives |
|---|---|---:|---:|---:|---:|---:|---|
| base_pattern_specialist | failed_breakout_20d__rank_all | 0.116700 | 0.116700 | 5710 | 5710 | 1142 | true |
| base_pattern_specialist | failed_breakout_52w__rank_all | -0.063212 | 0.063212 | 5710 | 5710 | 1142 | true |
| base_pattern_specialist | failed_breakout_52w | -0.061301 | 0.061301 | 5710 | 5710 | 1142 | true |
| base_pattern_specialist | failed_breakout_20d__rank_sector | -0.054884 | 0.054884 | 5710 | 5710 | 1142 | true |
| base_pattern_specialist | days_since_failed_breakout_52w__rank_sector | 0.051885 | 0.051885 | 3546 | 3546 | 710 | true |
| base_pattern_specialist | days_since_failed_breakout_20d__rank_all | 0.051674 | 0.051674 | 5228 | 5228 | 1046 | true |
| base_pattern_specialist | days_since_failed_breakout_20d__rank_sector | 0.048615 | 0.048615 | 5228 | 5228 | 1046 | true |
| base_pattern_specialist | base_range_pct_20__rank_all | 0.041554 | 0.041554 | 220772 | 220772 | 44155 | true |
| base_pattern_specialist | close_vs_20d_low__rank_all | 0.039460 | 0.039460 | 220772 | 220772 | 44155 | true |
| base_pattern_specialist | close_vs_20d_low__rank_sector | 0.031387 | 0.031387 | 220772 | 220772 | 44155 | true |
| base_pattern_specialist | days_since_failed_breakout_52w__rank_all | 0.030314 | 0.030314 | 3546 | 3546 | 710 | true |
| base_pattern_specialist | base_range_pct_20__rank_sector | 0.028642 | 0.028642 | 220772 | 220772 | 44155 | false |
| base_pattern_specialist | days_since_failed_breakout_52w | -0.024929 | 0.024929 | 3546 | 3546 | 710 | false |
| base_pattern_specialist | days_since_52w_high__rank_sector | -0.023833 | 0.023833 | 220228 | 220228 | 44046 | false |
| base_pattern_specialist | distance_from_52w_high__rank_sector | 0.020482 | 0.020482 | 220228 | 220228 | 44046 | false |
| base_pattern_specialist | days_since_52w_high__rank_all | -0.018370 | 0.018370 | 220228 | 220228 | 44046 | false |
| base_pattern_specialist | base_range_pct_20 | 0.018119 | 0.018119 | 220772 | 220772 | 44155 | false |
| base_pattern_specialist | failed_breakout_52w__rank_sector | -0.015572 | 0.015572 | 5710 | 5710 | 1142 | false |
| base_pattern_specialist | close_vs_20d_low | 0.014410 | 0.014410 | 220772 | 220772 | 44155 | false |
| base_pattern_specialist | dollar_volume_ratio_20_60__rank_all | 0.012471 | 0.012471 | 220772 | 220772 | 44155 | false |
| base_pattern_specialist | dollar_volume_ratio_20_60__rank_sector | 0.011851 | 0.011851 | 220772 | 220772 | 44155 | false |
| base_pattern_specialist | days_since_52w_high | -0.011170 | 0.011170 | 220228 | 220228 | 44046 | false |
| base_pattern_specialist | breakout_volume_ratio_50 | -0.010784 | 0.010784 | 220772 | 220772 | 44155 | false |
| base_pattern_specialist | distance_from_52w_high__rank_all | 0.010618 | 0.010618 | 220228 | 220228 | 44046 | false |
| base_pattern_specialist | dollar_volume_ratio_20_60 | -0.008854 | 0.008854 | 220772 | 220772 | 44155 | false |
| base_pattern_specialist | base_volume_dryup_ratio_20 | 0.007265 | 0.007265 | 220772 | 220772 | 44155 | false |
| base_pattern_specialist | base_volume_dryup_ratio_20__rank_all | 0.007072 | 0.007072 | 220772 | 220772 | 44155 | false |
| base_pattern_specialist | volume_percentile_60 | -0.006916 | 0.006916 | 220772 | 220772 | 44155 | false |
| base_pattern_specialist | base_volume_dryup_ratio_20__rank_sector | 0.006880 | 0.006880 | 220772 | 220772 | 44155 | false |
| base_pattern_specialist | base_atr_contraction_20 | -0.005649 | 0.005649 | 220772 | 220772 | 44155 | false |
| base_pattern_specialist | failed_breakout_20d | 0.005592 | 0.005592 | 5710 | 5710 | 1142 | false |
| base_pattern_specialist | distance_above_20d_high | -0.004133 | 0.004133 | 220772 | 220772 | 44155 | false |
| base_pattern_specialist | distance_above_20d_high__rank_sector | 0.003821 | 0.003821 | 220772 | 220772 | 44155 | false |
| base_pattern_specialist | volume_percentile_60__rank_all | 0.003665 | 0.003665 | 220772 | 220772 | 44155 | false |
| base_pattern_specialist | base_atr_contraction_20__rank_sector | 0.003507 | 0.003507 | 220772 | 220772 | 44155 | false |
| base_pattern_specialist | volume_percentile_60__rank_sector | 0.002930 | 0.002930 | 220772 | 220772 | 44155 | false |
| base_pattern_specialist | distance_from_52w_high | 0.002702 | 0.002702 | 220228 | 220228 | 44046 | false |
| base_pattern_specialist | distance_above_20d_high__rank_all | -0.002308 | 0.002308 | 220772 | 220772 | 44155 | false |
| base_pattern_specialist | breakout_volume_ratio_50__rank_sector | 0.002020 | 0.002020 | 220772 | 220772 | 44155 | false |
| base_pattern_specialist | base_atr_contraction_20__rank_all | 0.001588 | 0.001588 | 220772 | 220772 | 44155 | false |
| base_pattern_specialist | days_since_failed_breakout_20d | -0.001153 | 0.001153 | 5228 | 5228 | 1046 | false |
| base_pattern_specialist | breakout_volume_ratio_50__rank_all | 0.000921 | 0.000921 | 220772 | 220772 | 44155 | false |
| overnight_session_specialist | overnight_ret_20d | -0.218771 | 0.218771 | 5710 | 5710 | 1142 | true |
| overnight_session_specialist | overnight_ret_20d__rank_all | -0.209929 | 0.209929 | 5710 | 5710 | 1142 | true |
| overnight_session_specialist | overnight_ret_20d__rank_sector | -0.161710 | 0.161710 | 5710 | 5710 | 1142 | true |
| overnight_session_specialist | overnight_minus_rth_20d__rank_all | -0.125922 | 0.125922 | 5710 | 5710 | 1142 | true |
| overnight_session_specialist | overnight_minus_rth_20d | -0.115884 | 0.115884 | 5710 | 5710 | 1142 | true |
| overnight_session_specialist | overnight_minus_rth_20d__rank_sector | -0.074169 | 0.074169 | 5710 | 5710 | 1142 | true |
| overnight_session_specialist | rth_ret_20d__rank_all | 0.053205 | 0.053205 | 5710 | 5710 | 1142 | true |
| overnight_session_specialist | max_gap_down_pct_60__rank_all | 0.050826 | 0.050826 | 220772 | 220772 | 44155 | true |
| overnight_session_specialist | avg_abs_gap_pct_20__rank_all | 0.050023 | 0.050023 | 220772 | 220772 | 44155 | true |
| overnight_session_specialist | overnight_ret_5d__rank_all | 0.049890 | 0.049890 | 5710 | 5710 | 1142 | true |
| overnight_session_specialist | max_gap_down_pct_60__rank_sector | 0.041583 | 0.041583 | 220772 | 220772 | 44155 | true |
| overnight_session_specialist | overnight_ret_5d | 0.039578 | 0.039578 | 5710 | 5710 | 1142 | true |
| overnight_session_specialist | rth_ret_20d | 0.039139 | 0.039139 | 5710 | 5710 | 1142 | true |
| overnight_session_specialist | max_gap_down_pct_60 | 0.036844 | 0.036844 | 220772 | 220772 | 44155 | true |
| overnight_session_specialist | overnight_minus_rth_5d | 0.033512 | 0.033512 | 5710 | 5710 | 1142 | true |
| overnight_session_specialist | avg_abs_gap_pct_20__rank_sector | 0.029890 | 0.029890 | 220772 | 220772 | 44155 | false |
| overnight_session_specialist | avg_abs_gap_pct_20 | 0.022223 | 0.022223 | 220772 | 220772 | 44155 | false |
| overnight_session_specialist | rth_ret_5d__rank_all | 0.019222 | 0.019222 | 5710 | 5710 | 1142 | false |
| overnight_session_specialist | rth_ret_20d__rank_sector | 0.018444 | 0.018444 | 5710 | 5710 | 1142 | false |
| overnight_session_specialist | rth_ret_5d | -0.017984 | 0.017984 | 5710 | 5710 | 1142 | false |
| overnight_session_specialist | overnight_ret_5d__rank_sector | 0.017230 | 0.017230 | 5710 | 5710 | 1142 | false |
| overnight_session_specialist | overnight_minus_rth_5d__rank_sector | 0.005497 | 0.005497 | 5710 | 5710 | 1142 | false |
| overnight_session_specialist | rth_ret_5d__rank_sector | 0.005251 | 0.005251 | 5710 | 5710 | 1142 | false |
| overnight_session_specialist | overnight_minus_rth_5d__rank_all | 0.005239 | 0.005239 | 5710 | 5710 | 1142 | false |

## Prediction Spearman Correlation

| model | signal_proxy | structure_factor_signal | overnight_session_specialist | base_pattern_specialist |
|---|---:|---:|---:|---:|
| signal_proxy | 1.000000 | -0.222651 | 0.112971 | 0.085679 |
| structure_factor_signal | -0.222651 | 1.000000 | -0.148598 | -0.122287 |
| overnight_session_specialist | 0.112971 | -0.148598 | 1.000000 | 0.479370 |
| base_pattern_specialist | 0.085679 | -0.122287 | 0.479370 | 1.000000 |

## Decile Tables

| model | decile | dates | avg_rows | mean_target | hit_rate |
|---|---:|---:|---:|---:|---:|
| signal_proxy | 0 | 229 | 3.283843 | 0.017191 | 0.471808 |
| signal_proxy | 1 | 229 | 2.755459 | -0.010436 | 0.425301 |
| signal_proxy | 2 | 229 | 2.650655 | -0.002878 | 0.420695 |
| signal_proxy | 3 | 229 | 2.790393 | -0.009809 | 0.409680 |
| signal_proxy | 4 | 229 | 2.838428 | -0.003869 | 0.470927 |
| signal_proxy | 5 | 229 | 2.593886 | -0.016999 | 0.464286 |
| signal_proxy | 6 | 229 | 2.689956 | -0.002941 | 0.458672 |
| signal_proxy | 7 | 229 | 2.751092 | -0.008312 | 0.441245 |
| signal_proxy | 8 | 229 | 2.655022 | -0.028382 | 0.386179 |
| signal_proxy | 9 | 229 | 3.183406 | -0.015955 | 0.419137 |
| structure_factor_signal | 0 | 229 | 3.283843 | -0.016938 | 0.422425 |
| structure_factor_signal | 1 | 229 | 2.755459 | -0.024572 | 0.414221 |
| structure_factor_signal | 2 | 229 | 2.650655 | -0.014537 | 0.409972 |
| structure_factor_signal | 3 | 229 | 2.790393 | -0.007691 | 0.456901 |
| structure_factor_signal | 4 | 229 | 2.838428 | -0.009454 | 0.457223 |
| structure_factor_signal | 5 | 229 | 2.593886 | -0.018092 | 0.402645 |
| structure_factor_signal | 6 | 229 | 2.689956 | -0.010827 | 0.419542 |
| structure_factor_signal | 7 | 229 | 2.751092 | -0.007322 | 0.451258 |
| structure_factor_signal | 8 | 229 | 2.655022 | 0.016456 | 0.470627 |
| structure_factor_signal | 9 | 229 | 3.183406 | 0.023498 | 0.478164 |
| overnight_session_specialist | 0 | 229 | 3.283843 | 0.022941 | 0.472187 |
| overnight_session_specialist | 1 | 229 | 2.755459 | 0.012340 | 0.479367 |
| overnight_session_specialist | 2 | 229 | 2.650655 | 0.000457 | 0.442190 |
| overnight_session_specialist | 3 | 229 | 2.790393 | -0.009026 | 0.442106 |
| overnight_session_specialist | 4 | 229 | 2.838428 | -0.004406 | 0.423062 |
| overnight_session_specialist | 5 | 229 | 2.593886 | -0.020073 | 0.411470 |
| overnight_session_specialist | 6 | 229 | 2.689956 | -0.026823 | 0.419662 |
| overnight_session_specialist | 7 | 229 | 2.751092 | -0.020790 | 0.412740 |
| overnight_session_specialist | 8 | 229 | 2.655022 | -0.005165 | 0.454849 |
| overnight_session_specialist | 9 | 229 | 3.183406 | -0.016368 | 0.441702 |
| base_pattern_specialist | 0 | 229 | 3.283843 | 0.017117 | 0.460614 |
| base_pattern_specialist | 1 | 229 | 2.755459 | -0.003837 | 0.451621 |
| base_pattern_specialist | 2 | 229 | 2.650655 | -0.008448 | 0.426332 |
| base_pattern_specialist | 3 | 229 | 2.790393 | -0.013805 | 0.433336 |
| base_pattern_specialist | 4 | 229 | 2.838428 | -0.010177 | 0.452354 |
| base_pattern_specialist | 5 | 229 | 2.593886 | -0.010300 | 0.411891 |
| base_pattern_specialist | 6 | 229 | 2.689956 | -0.011711 | 0.435419 |
| base_pattern_specialist | 7 | 229 | 2.751092 | -0.018206 | 0.410925 |
| base_pattern_specialist | 8 | 229 | 2.655022 | -0.016527 | 0.424825 |
| base_pattern_specialist | 9 | 229 | 3.183406 | -0.004958 | 0.449963 |
