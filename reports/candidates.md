# Ranked Candidates

- run_id: 54
- min_trades: 12
- ranking: practical_score desc, then norm_score desc, then expectancy desc
- practical_score includes live-match bonus, trade-count bonus, and alpha bonuses vs SPY/sector
- walk_forward: enabled on shortlist=25 with rolling_windows=5
- walk_forward note: fixed-parameter rolling validation windows; not per-window re-optimized training

## Top Ranked Candidates

### Result 1156657
- run_id: 54
- strategy_id: 1156657
- global_rank: 1
- sector_rank: 1
- sector: Information Technology
- practical_score: 1.240933
- norm_score: 0.700000
- expectancy: 0.005023
- profit_factor: 1.613268
- alpha_vs_spy: 0.004631
- alpha_vs_sector: 0.003505
- mdd: 0.082025
- win_rate: 0.606061
- trade_count: 33
- wf_stability_score: -0.024459
- wf_window_count: 5
- wf_positive_window_ratio: 0.400000
- wf_positive_alpha_window_ratio: 0.200000
- wf_median_expectancy: 0.000000
- wf_worst_expectancy: -0.002635
- wf_median_alpha_vs_spy: -0.001632
- wf_worst_mdd: 0.381690
- wf_trade_count_min: 0
- duplicate_group_size: 32
- collapsed_result_ids: 1156657, 1156658, 1156721, 1156722, 1156785, 1156786, 1156849, 1156850, 1156913, 1156914, 1156977, 1156978, 1157041, 1157042, 1157105, 1157106, 1157169, 1157170, 1157233, 1157234, 1157297, 1157298, 1157361, 1157362, 1157425, 1157426, 1157489, 1157490, 1157553, 1157554, 1157617, 1157618
- live_match_count: 1
- live_match_tickers: CSCO
- promotion_policy_passed: no
- promotion_policy_violations: trade_count 33 < 100, wf_positive_window_ratio 0.4 < 0.600000, wf_positive_alpha_window_ratio 0.2 < 0.550000, wf_worst_mdd 0.38168992554596937 > 0.300000, wf_trade_count_min 0 < 8
- gate_counts: universe=250 -> regime_green=250 -> sector_scope=56 -> subindustry_scope=8 -> breakout_above_20d_high_min=1 -> distance_above_20d_high_max=1 -> ma_alignment_50_200_min=1 -> ma_slope_200_20_min=1 -> ma_slope_50_20_min=1 -> relative_strength_index_vs_qqq_min=1 -> relative_strength_index_vs_spy_min=1 -> sma_50_dist_min=1 -> signal_score_min=1
- first_zero_gate: none
- component_positive_counts: base_atr_contraction_20_max=1, base_range_pct_20_max=0, base_volume_dryup_ratio_20_max=1, breakout_volume_ratio_50_min=1, roc_126_min=1, roc_63_min=1, sma_200_dist_max=1
- warnings: none
- params: `{"backtest_costs": {"commission_bps_per_side": 0.0, "slippage_bps_per_side": 5.0}, "exit_rules": {"exit_before_earnings_days": null, "profit_target_atr_mult": 2.5, "profit_target_pct": null, "time_limit_days": 10, "trailing_stop_atr_mult": 1.5, "trailing_stop_pct": null}, "indicators": {"base_atr_contraction_20_max": 0.95, "base_range_pct_20_max": 0.08, "base_volume_dryup_ratio_20_max": 0.85, "breakout_above_20d_high_min": 1.0, "breakout_volume_ratio_50_min": 2.0, "distance_above_20d_high_max": 0.01, "ma_alignment_50_200_min": 0.0, "ma_slope_200_20_min": 0.0, "ma_slope_50_20_min": 0.0, "relative_strength_index_vs_qqq_min": 75.0, "relative_strength_index_vs_spy_min": 75.0, "roc_126_min": 0.1, "roc_63_min": 0.05, "signal_score_min": 34.0, "sma_200_dist_max": 0.25, "sma_50_dist_min": 0.0}, "scope_size": 250, "sector": "Information Technology", "sub_industry_whitelist": ["Communications Equipment", "Internet Services & Infrastructure"], "sweep_mode": "tech_comms_infra_breakout_v1"}`
- TradingView: https://www.tradingview.com/chart/?symbol=CSCO

### Result 1156621
- run_id: 54
- strategy_id: 1156621
- global_rank: 2
- sector_rank: 2
- sector: Information Technology
- practical_score: 1.209337
- norm_score: 0.682573
- expectancy: 0.004823
- profit_factor: 1.602606
- alpha_vs_spy: 0.004328
- alpha_vs_sector: 0.003060
- mdd: 0.083675
- win_rate: 0.588235
- trade_count: 34
- wf_stability_score: -0.019744
- wf_window_count: 5
- wf_positive_window_ratio: 0.400000
- wf_positive_alpha_window_ratio: 0.200000
- wf_median_expectancy: 0.000000
- wf_worst_expectancy: -0.002560
- wf_median_alpha_vs_spy: -0.001653
- wf_worst_mdd: 0.371766
- wf_trade_count_min: 0
- duplicate_group_size: 32
- collapsed_result_ids: 1156621, 1156622, 1156685, 1156686, 1156749, 1156750, 1156813, 1156814, 1156877, 1156878, 1156941, 1156942, 1157005, 1157006, 1157069, 1157070, 1157133, 1157134, 1157197, 1157198, 1157261, 1157262, 1157325, 1157326, 1157389, 1157390, 1157453, 1157454, 1157517, 1157518, 1157581, 1157582
- live_match_count: 1
- live_match_tickers: CSCO
- promotion_policy_passed: no
- promotion_policy_violations: trade_count 34 < 100, wf_positive_window_ratio 0.4 < 0.600000, wf_positive_alpha_window_ratio 0.2 < 0.550000, wf_worst_mdd 0.37176639714828996 > 0.300000, wf_trade_count_min 0 < 8
- gate_counts: universe=250 -> regime_green=250 -> sector_scope=56 -> subindustry_scope=8 -> breakout_above_20d_high_min=1 -> distance_above_20d_high_max=1 -> ma_alignment_50_200_min=1 -> ma_slope_200_20_min=1 -> ma_slope_50_20_min=1 -> relative_strength_index_vs_qqq_min=1 -> relative_strength_index_vs_spy_min=1 -> sma_50_dist_min=1 -> signal_score_min=1
- first_zero_gate: none
- component_positive_counts: base_atr_contraction_20_max=1, base_range_pct_20_max=0, base_volume_dryup_ratio_20_max=1, breakout_volume_ratio_50_min=1, roc_126_min=1, roc_63_min=1, sma_200_dist_max=1
- warnings: none
- params: `{"backtest_costs": {"commission_bps_per_side": 0.0, "slippage_bps_per_side": 5.0}, "exit_rules": {"exit_before_earnings_days": null, "profit_target_atr_mult": 2.5, "profit_target_pct": null, "time_limit_days": 10, "trailing_stop_atr_mult": 1.5, "trailing_stop_pct": null}, "indicators": {"base_atr_contraction_20_max": 0.85, "base_range_pct_20_max": 0.08, "base_volume_dryup_ratio_20_max": 0.85, "breakout_above_20d_high_min": 1.0, "breakout_volume_ratio_50_min": 2.0, "distance_above_20d_high_max": 0.01, "ma_alignment_50_200_min": 0.0, "ma_slope_200_20_min": 0.0, "ma_slope_50_20_min": 0.0, "relative_strength_index_vs_qqq_min": 75.0, "relative_strength_index_vs_spy_min": 75.0, "roc_126_min": 0.1, "roc_63_min": 0.05, "signal_score_min": 32.0, "sma_200_dist_max": 0.25, "sma_50_dist_min": 0.0}, "scope_size": 250, "sector": "Information Technology", "sub_industry_whitelist": ["Communications Equipment", "Internet Services & Infrastructure"], "sweep_mode": "tech_comms_infra_breakout_v1"}`
- TradingView: https://www.tradingview.com/chart/?symbol=CSCO

## Top Live Match Candidates

### Result 1156657
- run_id: 54
- strategy_id: 1156657
- global_rank: 1
- sector_rank: 1
- sector: Information Technology
- practical_score: 1.240933
- norm_score: 0.700000
- expectancy: 0.005023
- profit_factor: 1.613268
- alpha_vs_spy: 0.004631
- alpha_vs_sector: 0.003505
- mdd: 0.082025
- win_rate: 0.606061
- trade_count: 33
- wf_stability_score: -0.024459
- wf_window_count: 5
- wf_positive_window_ratio: 0.400000
- wf_positive_alpha_window_ratio: 0.200000
- wf_median_expectancy: 0.000000
- wf_worst_expectancy: -0.002635
- wf_median_alpha_vs_spy: -0.001632
- wf_worst_mdd: 0.381690
- wf_trade_count_min: 0
- duplicate_group_size: 32
- collapsed_result_ids: 1156657, 1156658, 1156721, 1156722, 1156785, 1156786, 1156849, 1156850, 1156913, 1156914, 1156977, 1156978, 1157041, 1157042, 1157105, 1157106, 1157169, 1157170, 1157233, 1157234, 1157297, 1157298, 1157361, 1157362, 1157425, 1157426, 1157489, 1157490, 1157553, 1157554, 1157617, 1157618
- live_match_count: 1
- live_match_tickers: CSCO
- promotion_policy_passed: no
- promotion_policy_violations: trade_count 33 < 100, wf_positive_window_ratio 0.4 < 0.600000, wf_positive_alpha_window_ratio 0.2 < 0.550000, wf_worst_mdd 0.38168992554596937 > 0.300000, wf_trade_count_min 0 < 8
- gate_counts: universe=250 -> regime_green=250 -> sector_scope=56 -> subindustry_scope=8 -> breakout_above_20d_high_min=1 -> distance_above_20d_high_max=1 -> ma_alignment_50_200_min=1 -> ma_slope_200_20_min=1 -> ma_slope_50_20_min=1 -> relative_strength_index_vs_qqq_min=1 -> relative_strength_index_vs_spy_min=1 -> sma_50_dist_min=1 -> signal_score_min=1
- first_zero_gate: none
- component_positive_counts: base_atr_contraction_20_max=1, base_range_pct_20_max=0, base_volume_dryup_ratio_20_max=1, breakout_volume_ratio_50_min=1, roc_126_min=1, roc_63_min=1, sma_200_dist_max=1
- warnings: none
- params: `{"backtest_costs": {"commission_bps_per_side": 0.0, "slippage_bps_per_side": 5.0}, "exit_rules": {"exit_before_earnings_days": null, "profit_target_atr_mult": 2.5, "profit_target_pct": null, "time_limit_days": 10, "trailing_stop_atr_mult": 1.5, "trailing_stop_pct": null}, "indicators": {"base_atr_contraction_20_max": 0.95, "base_range_pct_20_max": 0.08, "base_volume_dryup_ratio_20_max": 0.85, "breakout_above_20d_high_min": 1.0, "breakout_volume_ratio_50_min": 2.0, "distance_above_20d_high_max": 0.01, "ma_alignment_50_200_min": 0.0, "ma_slope_200_20_min": 0.0, "ma_slope_50_20_min": 0.0, "relative_strength_index_vs_qqq_min": 75.0, "relative_strength_index_vs_spy_min": 75.0, "roc_126_min": 0.1, "roc_63_min": 0.05, "signal_score_min": 34.0, "sma_200_dist_max": 0.25, "sma_50_dist_min": 0.0}, "scope_size": 250, "sector": "Information Technology", "sub_industry_whitelist": ["Communications Equipment", "Internet Services & Infrastructure"], "sweep_mode": "tech_comms_infra_breakout_v1"}`
- TradingView: https://www.tradingview.com/chart/?symbol=CSCO

### Result 1156621
- run_id: 54
- strategy_id: 1156621
- global_rank: 2
- sector_rank: 2
- sector: Information Technology
- practical_score: 1.209337
- norm_score: 0.682573
- expectancy: 0.004823
- profit_factor: 1.602606
- alpha_vs_spy: 0.004328
- alpha_vs_sector: 0.003060
- mdd: 0.083675
- win_rate: 0.588235
- trade_count: 34
- wf_stability_score: -0.019744
- wf_window_count: 5
- wf_positive_window_ratio: 0.400000
- wf_positive_alpha_window_ratio: 0.200000
- wf_median_expectancy: 0.000000
- wf_worst_expectancy: -0.002560
- wf_median_alpha_vs_spy: -0.001653
- wf_worst_mdd: 0.371766
- wf_trade_count_min: 0
- duplicate_group_size: 32
- collapsed_result_ids: 1156621, 1156622, 1156685, 1156686, 1156749, 1156750, 1156813, 1156814, 1156877, 1156878, 1156941, 1156942, 1157005, 1157006, 1157069, 1157070, 1157133, 1157134, 1157197, 1157198, 1157261, 1157262, 1157325, 1157326, 1157389, 1157390, 1157453, 1157454, 1157517, 1157518, 1157581, 1157582
- live_match_count: 1
- live_match_tickers: CSCO
- promotion_policy_passed: no
- promotion_policy_violations: trade_count 34 < 100, wf_positive_window_ratio 0.4 < 0.600000, wf_positive_alpha_window_ratio 0.2 < 0.550000, wf_worst_mdd 0.37176639714828996 > 0.300000, wf_trade_count_min 0 < 8
- gate_counts: universe=250 -> regime_green=250 -> sector_scope=56 -> subindustry_scope=8 -> breakout_above_20d_high_min=1 -> distance_above_20d_high_max=1 -> ma_alignment_50_200_min=1 -> ma_slope_200_20_min=1 -> ma_slope_50_20_min=1 -> relative_strength_index_vs_qqq_min=1 -> relative_strength_index_vs_spy_min=1 -> sma_50_dist_min=1 -> signal_score_min=1
- first_zero_gate: none
- component_positive_counts: base_atr_contraction_20_max=1, base_range_pct_20_max=0, base_volume_dryup_ratio_20_max=1, breakout_volume_ratio_50_min=1, roc_126_min=1, roc_63_min=1, sma_200_dist_max=1
- warnings: none
- params: `{"backtest_costs": {"commission_bps_per_side": 0.0, "slippage_bps_per_side": 5.0}, "exit_rules": {"exit_before_earnings_days": null, "profit_target_atr_mult": 2.5, "profit_target_pct": null, "time_limit_days": 10, "trailing_stop_atr_mult": 1.5, "trailing_stop_pct": null}, "indicators": {"base_atr_contraction_20_max": 0.85, "base_range_pct_20_max": 0.08, "base_volume_dryup_ratio_20_max": 0.85, "breakout_above_20d_high_min": 1.0, "breakout_volume_ratio_50_min": 2.0, "distance_above_20d_high_max": 0.01, "ma_alignment_50_200_min": 0.0, "ma_slope_200_20_min": 0.0, "ma_slope_50_20_min": 0.0, "relative_strength_index_vs_qqq_min": 75.0, "relative_strength_index_vs_spy_min": 75.0, "roc_126_min": 0.1, "roc_63_min": 0.05, "signal_score_min": 32.0, "sma_200_dist_max": 0.25, "sma_50_dist_min": 0.0}, "scope_size": 250, "sector": "Information Technology", "sub_industry_whitelist": ["Communications Equipment", "Internet Services & Infrastructure"], "sweep_mode": "tech_comms_infra_breakout_v1"}`
- TradingView: https://www.tradingview.com/chart/?symbol=CSCO

## Best Practical Live Candidates

### Result 1156657
- run_id: 54
- strategy_id: 1156657
- global_rank: 1
- sector_rank: 1
- sector: Information Technology
- practical_score: 1.240933
- norm_score: 0.700000
- expectancy: 0.005023
- profit_factor: 1.613268
- alpha_vs_spy: 0.004631
- alpha_vs_sector: 0.003505
- mdd: 0.082025
- win_rate: 0.606061
- trade_count: 33
- wf_stability_score: -0.024459
- wf_window_count: 5
- wf_positive_window_ratio: 0.400000
- wf_positive_alpha_window_ratio: 0.200000
- wf_median_expectancy: 0.000000
- wf_worst_expectancy: -0.002635
- wf_median_alpha_vs_spy: -0.001632
- wf_worst_mdd: 0.381690
- wf_trade_count_min: 0
- duplicate_group_size: 32
- collapsed_result_ids: 1156657, 1156658, 1156721, 1156722, 1156785, 1156786, 1156849, 1156850, 1156913, 1156914, 1156977, 1156978, 1157041, 1157042, 1157105, 1157106, 1157169, 1157170, 1157233, 1157234, 1157297, 1157298, 1157361, 1157362, 1157425, 1157426, 1157489, 1157490, 1157553, 1157554, 1157617, 1157618
- live_match_count: 1
- live_match_tickers: CSCO
- promotion_policy_passed: no
- promotion_policy_violations: trade_count 33 < 100, wf_positive_window_ratio 0.4 < 0.600000, wf_positive_alpha_window_ratio 0.2 < 0.550000, wf_worst_mdd 0.38168992554596937 > 0.300000, wf_trade_count_min 0 < 8
- gate_counts: universe=250 -> regime_green=250 -> sector_scope=56 -> subindustry_scope=8 -> breakout_above_20d_high_min=1 -> distance_above_20d_high_max=1 -> ma_alignment_50_200_min=1 -> ma_slope_200_20_min=1 -> ma_slope_50_20_min=1 -> relative_strength_index_vs_qqq_min=1 -> relative_strength_index_vs_spy_min=1 -> sma_50_dist_min=1 -> signal_score_min=1
- first_zero_gate: none
- component_positive_counts: base_atr_contraction_20_max=1, base_range_pct_20_max=0, base_volume_dryup_ratio_20_max=1, breakout_volume_ratio_50_min=1, roc_126_min=1, roc_63_min=1, sma_200_dist_max=1
- warnings: none
- params: `{"backtest_costs": {"commission_bps_per_side": 0.0, "slippage_bps_per_side": 5.0}, "exit_rules": {"exit_before_earnings_days": null, "profit_target_atr_mult": 2.5, "profit_target_pct": null, "time_limit_days": 10, "trailing_stop_atr_mult": 1.5, "trailing_stop_pct": null}, "indicators": {"base_atr_contraction_20_max": 0.95, "base_range_pct_20_max": 0.08, "base_volume_dryup_ratio_20_max": 0.85, "breakout_above_20d_high_min": 1.0, "breakout_volume_ratio_50_min": 2.0, "distance_above_20d_high_max": 0.01, "ma_alignment_50_200_min": 0.0, "ma_slope_200_20_min": 0.0, "ma_slope_50_20_min": 0.0, "relative_strength_index_vs_qqq_min": 75.0, "relative_strength_index_vs_spy_min": 75.0, "roc_126_min": 0.1, "roc_63_min": 0.05, "signal_score_min": 34.0, "sma_200_dist_max": 0.25, "sma_50_dist_min": 0.0}, "scope_size": 250, "sector": "Information Technology", "sub_industry_whitelist": ["Communications Equipment", "Internet Services & Infrastructure"], "sweep_mode": "tech_comms_infra_breakout_v1"}`
- TradingView: https://www.tradingview.com/chart/?symbol=CSCO

### Result 1156621
- run_id: 54
- strategy_id: 1156621
- global_rank: 2
- sector_rank: 2
- sector: Information Technology
- practical_score: 1.209337
- norm_score: 0.682573
- expectancy: 0.004823
- profit_factor: 1.602606
- alpha_vs_spy: 0.004328
- alpha_vs_sector: 0.003060
- mdd: 0.083675
- win_rate: 0.588235
- trade_count: 34
- wf_stability_score: -0.019744
- wf_window_count: 5
- wf_positive_window_ratio: 0.400000
- wf_positive_alpha_window_ratio: 0.200000
- wf_median_expectancy: 0.000000
- wf_worst_expectancy: -0.002560
- wf_median_alpha_vs_spy: -0.001653
- wf_worst_mdd: 0.371766
- wf_trade_count_min: 0
- duplicate_group_size: 32
- collapsed_result_ids: 1156621, 1156622, 1156685, 1156686, 1156749, 1156750, 1156813, 1156814, 1156877, 1156878, 1156941, 1156942, 1157005, 1157006, 1157069, 1157070, 1157133, 1157134, 1157197, 1157198, 1157261, 1157262, 1157325, 1157326, 1157389, 1157390, 1157453, 1157454, 1157517, 1157518, 1157581, 1157582
- live_match_count: 1
- live_match_tickers: CSCO
- promotion_policy_passed: no
- promotion_policy_violations: trade_count 34 < 100, wf_positive_window_ratio 0.4 < 0.600000, wf_positive_alpha_window_ratio 0.2 < 0.550000, wf_worst_mdd 0.37176639714828996 > 0.300000, wf_trade_count_min 0 < 8
- gate_counts: universe=250 -> regime_green=250 -> sector_scope=56 -> subindustry_scope=8 -> breakout_above_20d_high_min=1 -> distance_above_20d_high_max=1 -> ma_alignment_50_200_min=1 -> ma_slope_200_20_min=1 -> ma_slope_50_20_min=1 -> relative_strength_index_vs_qqq_min=1 -> relative_strength_index_vs_spy_min=1 -> sma_50_dist_min=1 -> signal_score_min=1
- first_zero_gate: none
- component_positive_counts: base_atr_contraction_20_max=1, base_range_pct_20_max=0, base_volume_dryup_ratio_20_max=1, breakout_volume_ratio_50_min=1, roc_126_min=1, roc_63_min=1, sma_200_dist_max=1
- warnings: none
- params: `{"backtest_costs": {"commission_bps_per_side": 0.0, "slippage_bps_per_side": 5.0}, "exit_rules": {"exit_before_earnings_days": null, "profit_target_atr_mult": 2.5, "profit_target_pct": null, "time_limit_days": 10, "trailing_stop_atr_mult": 1.5, "trailing_stop_pct": null}, "indicators": {"base_atr_contraction_20_max": 0.85, "base_range_pct_20_max": 0.08, "base_volume_dryup_ratio_20_max": 0.85, "breakout_above_20d_high_min": 1.0, "breakout_volume_ratio_50_min": 2.0, "distance_above_20d_high_max": 0.01, "ma_alignment_50_200_min": 0.0, "ma_slope_200_20_min": 0.0, "ma_slope_50_20_min": 0.0, "relative_strength_index_vs_qqq_min": 75.0, "relative_strength_index_vs_spy_min": 75.0, "roc_126_min": 0.1, "roc_63_min": 0.05, "signal_score_min": 32.0, "sma_200_dist_max": 0.25, "sma_50_dist_min": 0.0}, "scope_size": 250, "sector": "Information Technology", "sub_industry_whitelist": ["Communications Equipment", "Internet Services & Infrastructure"], "sweep_mode": "tech_comms_infra_breakout_v1"}`
- TradingView: https://www.tradingview.com/chart/?symbol=CSCO

## Best Candidate Per Sector

### Result 1156657
- run_id: 54
- strategy_id: 1156657
- global_rank: 1
- sector_rank: 1
- sector: Information Technology
- practical_score: 1.240933
- norm_score: 0.700000
- expectancy: 0.005023
- profit_factor: 1.613268
- alpha_vs_spy: 0.004631
- alpha_vs_sector: 0.003505
- mdd: 0.082025
- win_rate: 0.606061
- trade_count: 33
- wf_stability_score: -0.024459
- wf_window_count: 5
- wf_positive_window_ratio: 0.400000
- wf_positive_alpha_window_ratio: 0.200000
- wf_median_expectancy: 0.000000
- wf_worst_expectancy: -0.002635
- wf_median_alpha_vs_spy: -0.001632
- wf_worst_mdd: 0.381690
- wf_trade_count_min: 0
- duplicate_group_size: 32
- collapsed_result_ids: 1156657, 1156658, 1156721, 1156722, 1156785, 1156786, 1156849, 1156850, 1156913, 1156914, 1156977, 1156978, 1157041, 1157042, 1157105, 1157106, 1157169, 1157170, 1157233, 1157234, 1157297, 1157298, 1157361, 1157362, 1157425, 1157426, 1157489, 1157490, 1157553, 1157554, 1157617, 1157618
- live_match_count: 1
- live_match_tickers: CSCO
- promotion_policy_passed: no
- promotion_policy_violations: trade_count 33 < 100, wf_positive_window_ratio 0.4 < 0.600000, wf_positive_alpha_window_ratio 0.2 < 0.550000, wf_worst_mdd 0.38168992554596937 > 0.300000, wf_trade_count_min 0 < 8
- gate_counts: universe=250 -> regime_green=250 -> sector_scope=56 -> subindustry_scope=8 -> breakout_above_20d_high_min=1 -> distance_above_20d_high_max=1 -> ma_alignment_50_200_min=1 -> ma_slope_200_20_min=1 -> ma_slope_50_20_min=1 -> relative_strength_index_vs_qqq_min=1 -> relative_strength_index_vs_spy_min=1 -> sma_50_dist_min=1 -> signal_score_min=1
- first_zero_gate: none
- component_positive_counts: base_atr_contraction_20_max=1, base_range_pct_20_max=0, base_volume_dryup_ratio_20_max=1, breakout_volume_ratio_50_min=1, roc_126_min=1, roc_63_min=1, sma_200_dist_max=1
- warnings: none
- params: `{"backtest_costs": {"commission_bps_per_side": 0.0, "slippage_bps_per_side": 5.0}, "exit_rules": {"exit_before_earnings_days": null, "profit_target_atr_mult": 2.5, "profit_target_pct": null, "time_limit_days": 10, "trailing_stop_atr_mult": 1.5, "trailing_stop_pct": null}, "indicators": {"base_atr_contraction_20_max": 0.95, "base_range_pct_20_max": 0.08, "base_volume_dryup_ratio_20_max": 0.85, "breakout_above_20d_high_min": 1.0, "breakout_volume_ratio_50_min": 2.0, "distance_above_20d_high_max": 0.01, "ma_alignment_50_200_min": 0.0, "ma_slope_200_20_min": 0.0, "ma_slope_50_20_min": 0.0, "relative_strength_index_vs_qqq_min": 75.0, "relative_strength_index_vs_spy_min": 75.0, "roc_126_min": 0.1, "roc_63_min": 0.05, "signal_score_min": 34.0, "sma_200_dist_max": 0.25, "sma_50_dist_min": 0.0}, "scope_size": 250, "sector": "Information Technology", "sub_industry_whitelist": ["Communications Equipment", "Internet Services & Infrastructure"], "sweep_mode": "tech_comms_infra_breakout_v1"}`
- TradingView: https://www.tradingview.com/chart/?symbol=CSCO

## Best Promotable Candidate Per Sector

No sectors currently satisfy promotion policy.

## Best Live Candidate Per Sector

### Result 1156657
- run_id: 54
- strategy_id: 1156657
- global_rank: 1
- sector_rank: 1
- sector: Information Technology
- practical_score: 1.240933
- norm_score: 0.700000
- expectancy: 0.005023
- profit_factor: 1.613268
- alpha_vs_spy: 0.004631
- alpha_vs_sector: 0.003505
- mdd: 0.082025
- win_rate: 0.606061
- trade_count: 33
- wf_stability_score: -0.024459
- wf_window_count: 5
- wf_positive_window_ratio: 0.400000
- wf_positive_alpha_window_ratio: 0.200000
- wf_median_expectancy: 0.000000
- wf_worst_expectancy: -0.002635
- wf_median_alpha_vs_spy: -0.001632
- wf_worst_mdd: 0.381690
- wf_trade_count_min: 0
- duplicate_group_size: 32
- collapsed_result_ids: 1156657, 1156658, 1156721, 1156722, 1156785, 1156786, 1156849, 1156850, 1156913, 1156914, 1156977, 1156978, 1157041, 1157042, 1157105, 1157106, 1157169, 1157170, 1157233, 1157234, 1157297, 1157298, 1157361, 1157362, 1157425, 1157426, 1157489, 1157490, 1157553, 1157554, 1157617, 1157618
- live_match_count: 1
- live_match_tickers: CSCO
- promotion_policy_passed: no
- promotion_policy_violations: trade_count 33 < 100, wf_positive_window_ratio 0.4 < 0.600000, wf_positive_alpha_window_ratio 0.2 < 0.550000, wf_worst_mdd 0.38168992554596937 > 0.300000, wf_trade_count_min 0 < 8
- gate_counts: universe=250 -> regime_green=250 -> sector_scope=56 -> subindustry_scope=8 -> breakout_above_20d_high_min=1 -> distance_above_20d_high_max=1 -> ma_alignment_50_200_min=1 -> ma_slope_200_20_min=1 -> ma_slope_50_20_min=1 -> relative_strength_index_vs_qqq_min=1 -> relative_strength_index_vs_spy_min=1 -> sma_50_dist_min=1 -> signal_score_min=1
- first_zero_gate: none
- component_positive_counts: base_atr_contraction_20_max=1, base_range_pct_20_max=0, base_volume_dryup_ratio_20_max=1, breakout_volume_ratio_50_min=1, roc_126_min=1, roc_63_min=1, sma_200_dist_max=1
- warnings: none
- params: `{"backtest_costs": {"commission_bps_per_side": 0.0, "slippage_bps_per_side": 5.0}, "exit_rules": {"exit_before_earnings_days": null, "profit_target_atr_mult": 2.5, "profit_target_pct": null, "time_limit_days": 10, "trailing_stop_atr_mult": 1.5, "trailing_stop_pct": null}, "indicators": {"base_atr_contraction_20_max": 0.95, "base_range_pct_20_max": 0.08, "base_volume_dryup_ratio_20_max": 0.85, "breakout_above_20d_high_min": 1.0, "breakout_volume_ratio_50_min": 2.0, "distance_above_20d_high_max": 0.01, "ma_alignment_50_200_min": 0.0, "ma_slope_200_20_min": 0.0, "ma_slope_50_20_min": 0.0, "relative_strength_index_vs_qqq_min": 75.0, "relative_strength_index_vs_spy_min": 75.0, "roc_126_min": 0.1, "roc_63_min": 0.05, "signal_score_min": 34.0, "sma_200_dist_max": 0.25, "sma_50_dist_min": 0.0}, "scope_size": 250, "sector": "Information Technology", "sub_industry_whitelist": ["Communications Equipment", "Internet Services & Infrastructure"], "sweep_mode": "tech_comms_infra_breakout_v1"}`
- TradingView: https://www.tradingview.com/chart/?symbol=CSCO

## Best Promotable Portfolio Pairs

No portfolio pairs currently satisfy promotion policy.

## Best Walk-Forward Stability Candidates

### Result 1156627
- run_id: 54
- strategy_id: 1156627
- global_rank: 23
- sector_rank: 23
- sector: Information Technology
- practical_score: 0.902510
- norm_score: 0.379161
- expectancy: 0.002902
- profit_factor: 1.251199
- alpha_vs_spy: 0.003404
- alpha_vs_sector: 0.003440
- mdd: 0.127441
- win_rate: 0.600000
- trade_count: 30
- wf_stability_score: 0.184279
- wf_window_count: 5
- wf_positive_window_ratio: 0.600000
- wf_positive_alpha_window_ratio: 0.400000
- wf_median_expectancy: 0.000625
- wf_worst_expectancy: -0.000356
- wf_median_alpha_vs_spy: 0.000551
- wf_worst_mdd: 0.280207
- wf_trade_count_min: 0
- duplicate_group_size: 16
- collapsed_result_ids: 1156627, 1156691, 1156755, 1156819, 1156883, 1156947, 1157011, 1157075, 1157139, 1157203, 1157267, 1157331, 1157395, 1157459, 1157523, 1157587
- live_match_count: 1
- live_match_tickers: CSCO
- promotion_policy_passed: no
- promotion_policy_violations: profit_factor 1.251199 < 1.300000, expectancy 0.002902 < 0.004000, trade_count 30 < 100, wf_positive_alpha_window_ratio 0.4 < 0.550000, wf_trade_count_min 0 < 8
- gate_counts: universe=250 -> regime_green=250 -> sector_scope=56 -> subindustry_scope=8 -> breakout_above_20d_high_min=1 -> distance_above_20d_high_max=1 -> ma_alignment_50_200_min=1 -> ma_slope_200_20_min=1 -> ma_slope_50_20_min=1 -> relative_strength_index_vs_qqq_min=1 -> relative_strength_index_vs_spy_min=1 -> sma_50_dist_min=1 -> signal_score_min=1
- first_zero_gate: none
- component_positive_counts: base_atr_contraction_20_max=1, base_range_pct_20_max=0, base_volume_dryup_ratio_20_max=1, breakout_volume_ratio_50_min=1, roc_126_min=1, roc_63_min=1, sma_200_dist_max=1
- warnings: none
- params: `{"backtest_costs": {"commission_bps_per_side": 0.0, "slippage_bps_per_side": 5.0}, "exit_rules": {"exit_before_earnings_days": null, "profit_target_atr_mult": 2.5, "profit_target_pct": null, "time_limit_days": 10, "trailing_stop_atr_mult": 2.0, "trailing_stop_pct": null}, "indicators": {"base_atr_contraction_20_max": 0.85, "base_range_pct_20_max": 0.08, "base_volume_dryup_ratio_20_max": 0.85, "breakout_above_20d_high_min": 1.0, "breakout_volume_ratio_50_min": 2.0, "distance_above_20d_high_max": 0.01, "ma_alignment_50_200_min": 0.0, "ma_slope_200_20_min": 0.0, "ma_slope_50_20_min": 0.0, "relative_strength_index_vs_qqq_min": 75.0, "relative_strength_index_vs_spy_min": 75.0, "roc_126_min": 0.1, "roc_63_min": 0.05, "signal_score_min": 34.0, "sma_200_dist_max": 0.25, "sma_50_dist_min": 0.0}, "scope_size": 250, "sector": "Information Technology", "sub_industry_whitelist": ["Communications Equipment", "Internet Services & Infrastructure"], "sweep_mode": "tech_comms_infra_breakout_v1"}`
- TradingView: https://www.tradingview.com/chart/?symbol=CSCO

### Result 1156628
- run_id: 54
- strategy_id: 1156628
- global_rank: 20
- sector_rank: 20
- sector: Information Technology
- practical_score: 0.959508
- norm_score: 0.419557
- expectancy: 0.003363
- profit_factor: 1.291077
- alpha_vs_spy: 0.003865
- alpha_vs_sector: 0.003901
- mdd: 0.127441
- win_rate: 0.600000
- trade_count: 30
- wf_stability_score: 0.184056
- wf_window_count: 5
- wf_positive_window_ratio: 0.600000
- wf_positive_alpha_window_ratio: 0.400000
- wf_median_expectancy: 0.000625
- wf_worst_expectancy: -0.000356
- wf_median_alpha_vs_spy: 0.000420
- wf_worst_mdd: 0.281102
- wf_trade_count_min: 0
- duplicate_group_size: 16
- collapsed_result_ids: 1156628, 1156692, 1156756, 1156820, 1156884, 1156948, 1157012, 1157076, 1157140, 1157204, 1157268, 1157332, 1157396, 1157460, 1157524, 1157588
- live_match_count: 1
- live_match_tickers: CSCO
- promotion_policy_passed: no
- promotion_policy_violations: profit_factor 1.291077 < 1.300000, expectancy 0.003363 < 0.004000, trade_count 30 < 100, wf_positive_alpha_window_ratio 0.4 < 0.550000, wf_trade_count_min 0 < 8
- gate_counts: universe=250 -> regime_green=250 -> sector_scope=56 -> subindustry_scope=8 -> breakout_above_20d_high_min=1 -> distance_above_20d_high_max=1 -> ma_alignment_50_200_min=1 -> ma_slope_200_20_min=1 -> ma_slope_50_20_min=1 -> relative_strength_index_vs_qqq_min=1 -> relative_strength_index_vs_spy_min=1 -> sma_50_dist_min=1 -> signal_score_min=1
- first_zero_gate: none
- component_positive_counts: base_atr_contraction_20_max=1, base_range_pct_20_max=0, base_volume_dryup_ratio_20_max=1, breakout_volume_ratio_50_min=1, roc_126_min=1, roc_63_min=1, sma_200_dist_max=1
- warnings: none
- params: `{"backtest_costs": {"commission_bps_per_side": 0.0, "slippage_bps_per_side": 5.0}, "exit_rules": {"exit_before_earnings_days": null, "profit_target_atr_mult": 3.0, "profit_target_pct": null, "time_limit_days": 10, "trailing_stop_atr_mult": 2.0, "trailing_stop_pct": null}, "indicators": {"base_atr_contraction_20_max": 0.85, "base_range_pct_20_max": 0.08, "base_volume_dryup_ratio_20_max": 0.85, "breakout_above_20d_high_min": 1.0, "breakout_volume_ratio_50_min": 2.0, "distance_above_20d_high_max": 0.01, "ma_alignment_50_200_min": 0.0, "ma_slope_200_20_min": 0.0, "ma_slope_50_20_min": 0.0, "relative_strength_index_vs_qqq_min": 75.0, "relative_strength_index_vs_spy_min": 75.0, "roc_126_min": 0.1, "roc_63_min": 0.05, "signal_score_min": 34.0, "sma_200_dist_max": 0.25, "sma_50_dist_min": 0.0}, "scope_size": 250, "sector": "Information Technology", "sub_industry_whitelist": ["Communications Equipment", "Internet Services & Infrastructure"], "sweep_mode": "tech_comms_infra_breakout_v1"}`
- TradingView: https://www.tradingview.com/chart/?symbol=CSCO
