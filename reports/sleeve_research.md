# Sleeve Research

- sectors: Materials, Energy, Industrials
- configs_evaluated: 324
- model: sector breadth filter + within-sector rank score + fixed holding horizon
- note: portfolio simulation is equal-weight within sleeve, max open positions = top_n, no cross-sector capital coupling
- walk_forward: enabled on shortlist=10 with rolling_windows=5

## Top Ranked Sleeve Configurations

No sleeve configurations currently satisfy the top-section support floor.

## Best Configuration Per Sector

No sector-level sleeve configurations currently satisfy the top-section support floor.

## Best Configuration Per Horizon

No horizon-level sleeve configurations currently satisfy the top-section support floor.

## Best Live Configurations With Enough Sample

No live sleeve configurations currently satisfy the minimum trade support floor.

## Best Supported Configuration Per Sector

No sector-level sleeve configurations currently satisfy the supported floor.

## Best Supported Live Configuration Per Sector

No sector-level live sleeve configurations currently satisfy the supported floor.

## Best Walk-Forward Stable Sleeve Configurations

### Industrials | top_n=2 | horizon=15d
- global_rank: 3
- sector_rank: 1
- practical_score: 0.461437
- expectancy: 0.033871
- profit_factor: 2.645487
- alpha_vs_spy: 0.025903
- alpha_vs_sector: 0.029755
- mdd: 0.165011
- win_rate: 0.681159
- trade_count: 69
- distinct_tickers_traded: 28
- max_ticker_trade_share: 0.130435
- live_match_count: 0
- live_match_tickers: none
- trade_support_ratio: 0.690000
- low_support_penalty: 0.080000
- distinct_ticker_support_ratio: 1.000000
- low_distinct_ticker_penalty: 0.000000
- ticker_concentration_penalty: 0.000000
- wf_stability_score: 0.529977
- wf_positive_window_ratio: 0.800000
- wf_positive_alpha_window_ratio: 0.800000
- wf_median_expectancy: 0.034888
- wf_worst_expectancy: -0.137026
- wf_median_alpha_vs_spy: 0.019282
- wf_worst_mdd: 0.160091
- wf_trade_count_min: 2
- breadth_filters: sector_pct_above_50>=0.40, sector_pct_above_200>=0.40, sector_median_roc_63>=0.05
- base_filters: `{'relative_strength_index_vs_spy_min': 75.0, 'roc_63_min': 0.05, 'vol_alpha_min': 1.0, 'rsi_14_min': 40.0, 'rsi_14_max': 65.0}`
- rank_weights: `{'relative_strength_index_vs_spy': 0.4, 'roc_63': 0.25, 'rsi_14_pullback': 0.2, 'vol_alpha': 0.15}`

### Industrials | top_n=2 | horizon=15d
- global_rank: 4
- sector_rank: 1
- practical_score: 0.461437
- expectancy: 0.033871
- profit_factor: 2.645487
- alpha_vs_spy: 0.025903
- alpha_vs_sector: 0.029755
- mdd: 0.165011
- win_rate: 0.681159
- trade_count: 69
- distinct_tickers_traded: 28
- max_ticker_trade_share: 0.130435
- live_match_count: 0
- live_match_tickers: none
- trade_support_ratio: 0.690000
- low_support_penalty: 0.080000
- distinct_ticker_support_ratio: 1.000000
- low_distinct_ticker_penalty: 0.000000
- ticker_concentration_penalty: 0.000000
- wf_stability_score: 0.529977
- wf_positive_window_ratio: 0.800000
- wf_positive_alpha_window_ratio: 0.800000
- wf_median_expectancy: 0.034888
- wf_worst_expectancy: -0.137026
- wf_median_alpha_vs_spy: 0.019282
- wf_worst_mdd: 0.160091
- wf_trade_count_min: 2
- breadth_filters: sector_pct_above_50>=0.40, sector_pct_above_200>=0.50, sector_median_roc_63>=0.05
- base_filters: `{'relative_strength_index_vs_spy_min': 75.0, 'roc_63_min': 0.05, 'vol_alpha_min': 1.0, 'rsi_14_min': 40.0, 'rsi_14_max': 65.0}`
- rank_weights: `{'relative_strength_index_vs_spy': 0.4, 'roc_63': 0.25, 'rsi_14_pullback': 0.2, 'vol_alpha': 0.15}`

### Industrials | top_n=2 | horizon=15d
- global_rank: 5
- sector_rank: 1
- practical_score: 0.461437
- expectancy: 0.033871
- profit_factor: 2.645487
- alpha_vs_spy: 0.025903
- alpha_vs_sector: 0.029755
- mdd: 0.165011
- win_rate: 0.681159
- trade_count: 69
- distinct_tickers_traded: 28
- max_ticker_trade_share: 0.130435
- live_match_count: 0
- live_match_tickers: none
- trade_support_ratio: 0.690000
- low_support_penalty: 0.080000
- distinct_ticker_support_ratio: 1.000000
- low_distinct_ticker_penalty: 0.000000
- ticker_concentration_penalty: 0.000000
- wf_stability_score: 0.529977
- wf_positive_window_ratio: 0.800000
- wf_positive_alpha_window_ratio: 0.800000
- wf_median_expectancy: 0.034888
- wf_worst_expectancy: -0.137026
- wf_median_alpha_vs_spy: 0.019282
- wf_worst_mdd: 0.160091
- wf_trade_count_min: 2
- breadth_filters: sector_pct_above_50>=0.40, sector_pct_above_200>=0.60, sector_median_roc_63>=0.05
- base_filters: `{'relative_strength_index_vs_spy_min': 75.0, 'roc_63_min': 0.05, 'vol_alpha_min': 1.0, 'rsi_14_min': 40.0, 'rsi_14_max': 65.0}`
- rank_weights: `{'relative_strength_index_vs_spy': 0.4, 'roc_63': 0.25, 'rsi_14_pullback': 0.2, 'vol_alpha': 0.15}`

### Industrials | top_n=2 | horizon=15d
- global_rank: 6
- sector_rank: 2
- practical_score: 0.450104
- expectancy: 0.033535
- profit_factor: 2.624553
- alpha_vs_spy: 0.026307
- alpha_vs_sector: 0.029642
- mdd: 0.166224
- win_rate: 0.671642
- trade_count: 67
- distinct_tickers_traded: 27
- max_ticker_trade_share: 0.119403
- live_match_count: 0
- live_match_tickers: none
- trade_support_ratio: 0.670000
- low_support_penalty: 0.106667
- distinct_ticker_support_ratio: 1.000000
- low_distinct_ticker_penalty: 0.000000
- ticker_concentration_penalty: 0.000000
- wf_stability_score: 0.529977
- wf_positive_window_ratio: 0.800000
- wf_positive_alpha_window_ratio: 0.800000
- wf_median_expectancy: 0.036179
- wf_worst_expectancy: -0.137026
- wf_median_alpha_vs_spy: 0.024546
- wf_worst_mdd: 0.160091
- wf_trade_count_min: 2
- breadth_filters: sector_pct_above_50>=0.50, sector_pct_above_200>=0.40, sector_median_roc_63>=0.05
- base_filters: `{'relative_strength_index_vs_spy_min': 75.0, 'roc_63_min': 0.05, 'vol_alpha_min': 1.0, 'rsi_14_min': 40.0, 'rsi_14_max': 65.0}`
- rank_weights: `{'relative_strength_index_vs_spy': 0.4, 'roc_63': 0.25, 'rsi_14_pullback': 0.2, 'vol_alpha': 0.15}`

### Industrials | top_n=2 | horizon=15d
- global_rank: 7
- sector_rank: 2
- practical_score: 0.450104
- expectancy: 0.033535
- profit_factor: 2.624553
- alpha_vs_spy: 0.026307
- alpha_vs_sector: 0.029642
- mdd: 0.166224
- win_rate: 0.671642
- trade_count: 67
- distinct_tickers_traded: 27
- max_ticker_trade_share: 0.119403
- live_match_count: 0
- live_match_tickers: none
- trade_support_ratio: 0.670000
- low_support_penalty: 0.106667
- distinct_ticker_support_ratio: 1.000000
- low_distinct_ticker_penalty: 0.000000
- ticker_concentration_penalty: 0.000000
- wf_stability_score: 0.529977
- wf_positive_window_ratio: 0.800000
- wf_positive_alpha_window_ratio: 0.800000
- wf_median_expectancy: 0.036179
- wf_worst_expectancy: -0.137026
- wf_median_alpha_vs_spy: 0.024546
- wf_worst_mdd: 0.160091
- wf_trade_count_min: 2
- breadth_filters: sector_pct_above_50>=0.50, sector_pct_above_200>=0.50, sector_median_roc_63>=0.05
- base_filters: `{'relative_strength_index_vs_spy_min': 75.0, 'roc_63_min': 0.05, 'vol_alpha_min': 1.0, 'rsi_14_min': 40.0, 'rsi_14_max': 65.0}`
- rank_weights: `{'relative_strength_index_vs_spy': 0.4, 'roc_63': 0.25, 'rsi_14_pullback': 0.2, 'vol_alpha': 0.15}`

### Industrials | top_n=2 | horizon=15d
- global_rank: 8
- sector_rank: 2
- practical_score: 0.450104
- expectancy: 0.033535
- profit_factor: 2.624553
- alpha_vs_spy: 0.026307
- alpha_vs_sector: 0.029642
- mdd: 0.166224
- win_rate: 0.671642
- trade_count: 67
- distinct_tickers_traded: 27
- max_ticker_trade_share: 0.119403
- live_match_count: 0
- live_match_tickers: none
- trade_support_ratio: 0.670000
- low_support_penalty: 0.106667
- distinct_ticker_support_ratio: 1.000000
- low_distinct_ticker_penalty: 0.000000
- ticker_concentration_penalty: 0.000000
- wf_stability_score: 0.529977
- wf_positive_window_ratio: 0.800000
- wf_positive_alpha_window_ratio: 0.800000
- wf_median_expectancy: 0.036179
- wf_worst_expectancy: -0.137026
- wf_median_alpha_vs_spy: 0.024546
- wf_worst_mdd: 0.160091
- wf_trade_count_min: 2
- breadth_filters: sector_pct_above_50>=0.50, sector_pct_above_200>=0.60, sector_median_roc_63>=0.05
- base_filters: `{'relative_strength_index_vs_spy_min': 75.0, 'roc_63_min': 0.05, 'vol_alpha_min': 1.0, 'rsi_14_min': 40.0, 'rsi_14_max': 65.0}`
- rank_weights: `{'relative_strength_index_vs_spy': 0.4, 'roc_63': 0.25, 'rsi_14_pullback': 0.2, 'vol_alpha': 0.15}`

### Materials | top_n=1 | horizon=15d
- global_rank: 1
- sector_rank: 1
- practical_score: 0.490307
- expectancy: 0.056273
- profit_factor: 4.212199
- alpha_vs_spy: 0.049840
- alpha_vs_sector: 0.061610
- mdd: 0.186120
- win_rate: 0.611111
- trade_count: 18
- distinct_tickers_traded: 7
- max_ticker_trade_share: 0.333333
- live_match_count: 0
- live_match_tickers: none
- trade_support_ratio: 0.180000
- low_support_penalty: 0.760000
- distinct_ticker_support_ratio: 1.000000
- low_distinct_ticker_penalty: 0.000000
- ticker_concentration_penalty: 0.000000
- wf_stability_score: 0.465679
- wf_positive_window_ratio: 0.600000
- wf_positive_alpha_window_ratio: 0.800000
- wf_median_expectancy: 0.039398
- wf_worst_expectancy: -0.021884
- wf_median_alpha_vs_spy: 0.024696
- wf_worst_mdd: 0.177286
- wf_trade_count_min: 2
- breadth_filters: sector_pct_above_50>=0.60, sector_pct_above_200>=0.40, sector_median_roc_63>=0.00
- base_filters: `{'relative_strength_index_vs_spy_min': 75.0, 'roc_63_min': 0.05, 'vol_alpha_min': 1.0, 'rsi_14_min': 40.0, 'rsi_14_max': 65.0}`
- rank_weights: `{'relative_strength_index_vs_spy': 0.4, 'roc_63': 0.25, 'rsi_14_pullback': 0.2, 'vol_alpha': 0.15}`

### Materials | top_n=1 | horizon=15d
- global_rank: 2
- sector_rank: 1
- practical_score: 0.490307
- expectancy: 0.056273
- profit_factor: 4.212199
- alpha_vs_spy: 0.049840
- alpha_vs_sector: 0.061610
- mdd: 0.186120
- win_rate: 0.611111
- trade_count: 18
- distinct_tickers_traded: 7
- max_ticker_trade_share: 0.333333
- live_match_count: 0
- live_match_tickers: none
- trade_support_ratio: 0.180000
- low_support_penalty: 0.760000
- distinct_ticker_support_ratio: 1.000000
- low_distinct_ticker_penalty: 0.000000
- ticker_concentration_penalty: 0.000000
- wf_stability_score: 0.465679
- wf_positive_window_ratio: 0.600000
- wf_positive_alpha_window_ratio: 0.800000
- wf_median_expectancy: 0.039398
- wf_worst_expectancy: -0.021884
- wf_median_alpha_vs_spy: 0.024696
- wf_worst_mdd: 0.177286
- wf_trade_count_min: 2
- breadth_filters: sector_pct_above_50>=0.60, sector_pct_above_200>=0.50, sector_median_roc_63>=0.00
- base_filters: `{'relative_strength_index_vs_spy_min': 75.0, 'roc_63_min': 0.05, 'vol_alpha_min': 1.0, 'rsi_14_min': 40.0, 'rsi_14_max': 65.0}`
- rank_weights: `{'relative_strength_index_vs_spy': 0.4, 'roc_63': 0.25, 'rsi_14_pullback': 0.2, 'vol_alpha': 0.15}`

### Materials | top_n=1 | horizon=15d
- global_rank: 9
- sector_rank: 2
- practical_score: 0.449331
- expectancy: 0.051704
- profit_factor: 3.985036
- alpha_vs_spy: 0.047678
- alpha_vs_sector: 0.059519
- mdd: 0.177286
- win_rate: 0.562500
- trade_count: 16
- distinct_tickers_traded: 7
- max_ticker_trade_share: 0.437500
- live_match_count: 0
- live_match_tickers: none
- trade_support_ratio: 0.160000
- low_support_penalty: 0.786667
- distinct_ticker_support_ratio: 1.000000
- low_distinct_ticker_penalty: 0.000000
- ticker_concentration_penalty: 0.000000
- wf_stability_score: 0.415283
- wf_positive_window_ratio: 0.800000
- wf_positive_alpha_window_ratio: 0.800000
- wf_median_expectancy: 0.011823
- wf_worst_expectancy: -0.009457
- wf_median_alpha_vs_spy: 0.007069
- wf_worst_mdd: 0.177286
- wf_trade_count_min: 1
- breadth_filters: sector_pct_above_50>=0.60, sector_pct_above_200>=0.60, sector_median_roc_63>=0.00
- base_filters: `{'relative_strength_index_vs_spy_min': 75.0, 'roc_63_min': 0.05, 'vol_alpha_min': 1.0, 'rsi_14_min': 40.0, 'rsi_14_max': 65.0}`
- rank_weights: `{'relative_strength_index_vs_spy': 0.4, 'roc_63': 0.25, 'rsi_14_pullback': 0.2, 'vol_alpha': 0.15}`

### Industrials | top_n=1 | horizon=15d
- global_rank: 10
- sector_rank: 3
- practical_score: 0.399184
- expectancy: 0.046520
- profit_factor: 3.219272
- alpha_vs_spy: 0.039007
- alpha_vs_sector: 0.043275
- mdd: 0.271073
- win_rate: 0.725000
- trade_count: 40
- distinct_tickers_traded: 20
- max_ticker_trade_share: 0.175000
- live_match_count: 0
- live_match_tickers: none
- trade_support_ratio: 0.400000
- low_support_penalty: 0.466667
- distinct_ticker_support_ratio: 1.000000
- low_distinct_ticker_penalty: 0.000000
- ticker_concentration_penalty: 0.000000
- wf_stability_score: 0.258466
- wf_positive_window_ratio: 0.600000
- wf_positive_alpha_window_ratio: 0.600000
- wf_median_expectancy: 0.012686
- wf_worst_expectancy: -0.093326
- wf_median_alpha_vs_spy: 0.006874
- wf_worst_mdd: 0.301048
- wf_trade_count_min: 2
- breadth_filters: sector_pct_above_50>=0.50, sector_pct_above_200>=0.40, sector_median_roc_63>=0.00
- base_filters: `{'relative_strength_index_vs_spy_min': 75.0, 'roc_63_min': 0.05, 'vol_alpha_min': 1.0, 'rsi_14_min': 40.0, 'rsi_14_max': 65.0}`
- rank_weights: `{'relative_strength_index_vs_spy': 0.4, 'roc_63': 0.25, 'rsi_14_pullback': 0.2, 'vol_alpha': 0.15}`

## Best Stable Configuration Per Sector

No sector-level stable sleeve configurations currently satisfy the supported floor.
