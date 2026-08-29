# Shortlist Model

- target_column: alpha_vs_sector_20d
- top_n: 10
- eligible_universe_mode: passed_or_trend
- model_scope: sector_specific
- candidate_models: signal_proxy, ridge_model, lasso_model, xgboost_model, ensemble_model
- selected_model: xgboost_model
- selected_model_gate_passed: true
- xgboost_config: balanced_depth4
- min_train_dates: 252
- test_window_dates: 20
- oos_evaluation_stride_dates: 20
- objective: walk-forward cross-sectional ranking of the eligible universe on forward sector-relative alpha
- universe: passed_any_strategy or trend-qualified liquid names (green regime, above 200d, positive 63d momentum, RS vs SPY >= 60)
- feature_matrix: raw features plus date-wise cross-sectional ranks and sector-relative ranks

- eligible_rows: 268005
- eligible_dates: 624
- oos_prediction_dates: 19
- champion_model: xgboost_model
- oos_predictions_csv: /home/zdrillings/code/SwingQuant/reports/shortlist_model_oos_predictions.csv
- live_predictions_csv: /home/zdrillings/code/SwingQuant/reports/shortlist_model_live_predictions.csv
- generated_at: 2026-08-29T18:18:09+00:00

## Full Walk-Forward Evaluation

### xgboost_model
- dates: 19
- avg_pick_count: 9.789474
- mean_target: 0.202743
- hit_rate: 0.850877
- beat_universe_rate: 0.894737
- positive_date_rate: 0.894737
- ge_2pct_rate: 0.894737
- ge_5pct_rate: 0.842105

### ensemble_model
- dates: 19
- avg_pick_count: 9.789474
- mean_target: 0.118083
- hit_rate: 0.719298
- beat_universe_rate: 0.894737
- positive_date_rate: 0.894737
- ge_2pct_rate: 0.842105
- ge_5pct_rate: 0.684211

### ridge_model
- dates: 19
- avg_pick_count: 9.789474
- mean_target: 0.083305
- hit_rate: 0.640351
- beat_universe_rate: 0.736842
- positive_date_rate: 0.789474
- ge_2pct_rate: 0.684211
- ge_5pct_rate: 0.684211

### lasso_model
- dates: 19
- avg_pick_count: 9.789474
- mean_target: 0.081029
- hit_rate: 0.650877
- beat_universe_rate: 0.736842
- positive_date_rate: 0.789474
- ge_2pct_rate: 0.631579
- ge_5pct_rate: 0.578947

### signal_proxy
- dates: 19
- avg_pick_count: 9.789474
- mean_target: 0.017520
- hit_rate: 0.514035
- beat_universe_rate: 0.684211
- positive_date_rate: 0.578947
- ge_2pct_rate: 0.578947
- ge_5pct_rate: 0.263158

## Recent 19 Walk-Forward Dates

### xgboost_model
- dates: 19
- avg_pick_count: 9.789474
- mean_target: 0.202743
- hit_rate: 0.850877
- beat_universe_rate: 0.894737
- positive_date_rate: 0.894737
- ge_2pct_rate: 0.894737
- ge_5pct_rate: 0.842105

### ensemble_model
- dates: 19
- avg_pick_count: 9.789474
- mean_target: 0.118083
- hit_rate: 0.719298
- beat_universe_rate: 0.894737
- positive_date_rate: 0.894737
- ge_2pct_rate: 0.842105
- ge_5pct_rate: 0.684211

### ridge_model
- dates: 19
- avg_pick_count: 9.789474
- mean_target: 0.083305
- hit_rate: 0.640351
- beat_universe_rate: 0.736842
- positive_date_rate: 0.789474
- ge_2pct_rate: 0.684211
- ge_5pct_rate: 0.684211

### lasso_model
- dates: 19
- avg_pick_count: 9.789474
- mean_target: 0.081029
- hit_rate: 0.650877
- beat_universe_rate: 0.736842
- positive_date_rate: 0.789474
- ge_2pct_rate: 0.631579
- ge_5pct_rate: 0.578947

### signal_proxy
- dates: 19
- avg_pick_count: 9.789474
- mean_target: 0.017520
- hit_rate: 0.514035
- beat_universe_rate: 0.684211
- positive_date_rate: 0.578947
- ge_2pct_rate: 0.578947
- ge_5pct_rate: 0.263158

## Promotion Gate

- enabled: true
- min_recent_20d_hit_rate: 0.50
- min_recent_20d_beat_universe_rate: 0.50
- min_recent_20d_mean_target: 0.0000
- min_recent_60d_hit_rate: 0.50
- min_recent_60d_beat_universe_rate: 0.50
- min_recent_60d_mean_target: 0.0000
- min_recent_1fold_hit_rate: 0.50
- min_recent_1fold_beat_universe_rate: 0.50
- min_recent_1fold_mean_target: 0.0000
- min_recent_3fold_hit_rate: 0.50
- min_recent_3fold_beat_universe_rate: 0.50
- min_recent_3fold_mean_target: 0.0000

### Recent Acceptance Windows

### xgboost_model_last_3fold
- dates: 3
- avg_pick_count: 10.000000
- mean_target: 0.278904
- hit_rate: 0.866667
- beat_universe_rate: 1.000000
- positive_date_rate: 1.000000
- ge_2pct_rate: 1.000000
- ge_5pct_rate: 1.000000

### xgboost_model_20d
- dates: 19
- avg_pick_count: 9.789474
- mean_target: 0.202743
- hit_rate: 0.850877
- beat_universe_rate: 0.894737
- positive_date_rate: 0.894737
- ge_2pct_rate: 0.894737
- ge_5pct_rate: 0.842105

### xgboost_model_60d
- dates: 19
- avg_pick_count: 9.789474
- mean_target: 0.202743
- hit_rate: 0.850877
- beat_universe_rate: 0.894737
- positive_date_rate: 0.894737
- ge_2pct_rate: 0.894737
- ge_5pct_rate: 0.842105

### ensemble_model_last_3fold
- dates: 3
- avg_pick_count: 10.000000
- mean_target: 0.133723
- hit_rate: 0.666667
- beat_universe_rate: 1.000000
- positive_date_rate: 1.000000
- ge_2pct_rate: 0.666667
- ge_5pct_rate: 0.666667

### ensemble_model_20d
- dates: 19
- avg_pick_count: 9.789474
- mean_target: 0.118083
- hit_rate: 0.719298
- beat_universe_rate: 0.894737
- positive_date_rate: 0.894737
- ge_2pct_rate: 0.842105
- ge_5pct_rate: 0.684211

### ensemble_model_60d
- dates: 19
- avg_pick_count: 9.789474
- mean_target: 0.118083
- hit_rate: 0.719298
- beat_universe_rate: 0.894737
- positive_date_rate: 0.894737
- ge_2pct_rate: 0.842105
- ge_5pct_rate: 0.684211

### ridge_model_20d
- dates: 19
- avg_pick_count: 9.789474
- mean_target: 0.083305
- hit_rate: 0.640351
- beat_universe_rate: 0.736842
- positive_date_rate: 0.789474
- ge_2pct_rate: 0.684211
- ge_5pct_rate: 0.684211

### ridge_model_60d
- dates: 19
- avg_pick_count: 9.789474
- mean_target: 0.083305
- hit_rate: 0.640351
- beat_universe_rate: 0.736842
- positive_date_rate: 0.789474
- ge_2pct_rate: 0.684211
- ge_5pct_rate: 0.684211

### lasso_model_20d
- dates: 19
- avg_pick_count: 9.789474
- mean_target: 0.081029
- hit_rate: 0.650877
- beat_universe_rate: 0.736842
- positive_date_rate: 0.789474
- ge_2pct_rate: 0.631579
- ge_5pct_rate: 0.578947

### lasso_model_60d
- dates: 19
- avg_pick_count: 9.789474
- mean_target: 0.081029
- hit_rate: 0.650877
- beat_universe_rate: 0.736842
- positive_date_rate: 0.789474
- ge_2pct_rate: 0.631579
- ge_5pct_rate: 0.578947

### xgboost_model_last_1fold
- dates: 1
- avg_pick_count: 10.000000
- mean_target: 0.075968
- hit_rate: 0.700000
- beat_universe_rate: 1.000000
- positive_date_rate: 1.000000
- ge_2pct_rate: 1.000000
- ge_5pct_rate: 1.000000

### lasso_model_last_3fold
- dates: 3
- avg_pick_count: 10.000000
- mean_target: 0.018713
- hit_rate: 0.533333
- beat_universe_rate: 0.333333
- positive_date_rate: 0.666667
- ge_2pct_rate: 0.333333
- ge_5pct_rate: 0.333333

### signal_proxy_20d
- dates: 19
- avg_pick_count: 9.789474
- mean_target: 0.017520
- hit_rate: 0.514035
- beat_universe_rate: 0.684211
- positive_date_rate: 0.578947
- ge_2pct_rate: 0.578947
- ge_5pct_rate: 0.263158

### signal_proxy_60d
- dates: 19
- avg_pick_count: 9.789474
- mean_target: 0.017520
- hit_rate: 0.514035
- beat_universe_rate: 0.684211
- positive_date_rate: 0.578947
- ge_2pct_rate: 0.578947
- ge_5pct_rate: 0.263158

### ensemble_model_last_1fold
- dates: 1
- avg_pick_count: 10.000000
- mean_target: 0.007338
- hit_rate: 0.400000
- beat_universe_rate: 1.000000
- positive_date_rate: 1.000000
- ge_2pct_rate: 0.000000
- ge_5pct_rate: 0.000000

### ridge_model_last_3fold
- dates: 3
- avg_pick_count: 10.000000
- mean_target: -0.000460
- hit_rate: 0.466667
- beat_universe_rate: 0.333333
- positive_date_rate: 0.666667
- ge_2pct_rate: 0.333333
- ge_5pct_rate: 0.333333

### signal_proxy_last_3fold
- dates: 3
- avg_pick_count: 10.000000
- mean_target: -0.019533
- hit_rate: 0.400000
- beat_universe_rate: 0.333333
- positive_date_rate: 0.333333
- ge_2pct_rate: 0.333333
- ge_5pct_rate: 0.333333

### lasso_model_last_1fold
- dates: 1
- avg_pick_count: 10.000000
- mean_target: -0.074030
- hit_rate: 0.400000
- beat_universe_rate: 0.000000
- positive_date_rate: 0.000000
- ge_2pct_rate: 0.000000
- ge_5pct_rate: 0.000000

### ridge_model_last_1fold
- dates: 1
- avg_pick_count: 10.000000
- mean_target: -0.078409
- hit_rate: 0.400000
- beat_universe_rate: 0.000000
- positive_date_rate: 0.000000
- ge_2pct_rate: 0.000000
- ge_5pct_rate: 0.000000

### signal_proxy_last_1fold
- dates: 1
- avg_pick_count: 10.000000
- mean_target: -0.096234
- hit_rate: 0.100000
- beat_universe_rate: 0.000000
- positive_date_rate: 0.000000
- ge_2pct_rate: 0.000000
- ge_5pct_rate: 0.000000

## Champion Rolling Acceptance Windows

### xgboost_model_last_3fold
- dates: 3
- avg_pick_count: 10.000000
- mean_target: 0.278904
- hit_rate: 0.866667
- beat_universe_rate: 1.000000
- positive_date_rate: 1.000000
- ge_2pct_rate: 1.000000
- ge_5pct_rate: 1.000000

### xgboost_model_20d
- dates: 19
- avg_pick_count: 9.789474
- mean_target: 0.202743
- hit_rate: 0.850877
- beat_universe_rate: 0.894737
- positive_date_rate: 0.894737
- ge_2pct_rate: 0.894737
- ge_5pct_rate: 0.842105

### xgboost_model_40d
- dates: 19
- avg_pick_count: 9.789474
- mean_target: 0.202743
- hit_rate: 0.850877
- beat_universe_rate: 0.894737
- positive_date_rate: 0.894737
- ge_2pct_rate: 0.894737
- ge_5pct_rate: 0.842105

### xgboost_model_60d
- dates: 19
- avg_pick_count: 9.789474
- mean_target: 0.202743
- hit_rate: 0.850877
- beat_universe_rate: 0.894737
- positive_date_rate: 0.894737
- ge_2pct_rate: 0.894737
- ge_5pct_rate: 0.842105

### xgboost_model_last_1fold
- dates: 1
- avg_pick_count: 10.000000
- mean_target: 0.075968
- hit_rate: 0.700000
- beat_universe_rate: 1.000000
- positive_date_rate: 1.000000
- ge_2pct_rate: 1.000000
- ge_5pct_rate: 1.000000

## Champion Sector Contribution

### Information Technology
- dates: 13
- avg_pick_count: 3.846154
- mean_target: 0.327741
- hit_rate: 0.924359

### Health Care
- dates: 10
- avg_pick_count: 1.500000
- mean_target: 0.278882
- hit_rate: 0.933333

### Consumer Discretionary
- dates: 5
- avg_pick_count: 3.600000
- mean_target: 0.187500
- hit_rate: 0.942857

### Consumer Staples
- dates: 6
- avg_pick_count: 1.166667
- mean_target: 0.184156
- hit_rate: 0.833333

### Industrials
- dates: 11
- avg_pick_count: 2.000000
- mean_target: 0.183582
- hit_rate: 0.840909

### Materials
- dates: 11
- avg_pick_count: 1.636364
- mean_target: 0.160243
- hit_rate: 0.772727

### Financials
- dates: 9
- avg_pick_count: 1.666667
- mean_target: 0.155023
- hit_rate: 0.888889

### Energy
- dates: 11
- avg_pick_count: 1.818182
- mean_target: 0.150668
- hit_rate: 0.877273

### Communication Services
- dates: 9
- avg_pick_count: 1.555556
- mean_target: 0.149014
- hit_rate: 0.777778

### Real Estate
- dates: 2
- avg_pick_count: 1.000000
- mean_target: 0.143059
- hit_rate: 1.000000

### Utilities
- dates: 3
- avg_pick_count: 1.666667
- mean_target: 0.097400
- hit_rate: 0.666667

## Live Top Candidates

- champion_model: xgboost_model
- snapshot_date: 2026-08-28

### CDW
- sector: Information Technology
- predicted_alpha: 0.146837
- calibrated_p_beat_sector: 95.83%
- why: limited recent downside gap risk, strong earnings gap, earnings breakout open
- md_volume_30d: 206323352
- chart: https://www.tradingview.com/chart/?symbol=CDW

### GDDY
- sector: Information Technology
- predicted_alpha: 0.128910
- calibrated_p_beat_sector: 89.80%
- why: limited recent downside gap risk, strong earnings gap, strong sector momentum backdrop
- md_volume_30d: 143300870
- chart: https://www.tradingview.com/chart/?symbol=GDDY

### PANW
- sector: Information Technology
- predicted_alpha: 0.100816
- calibrated_p_beat_sector: 88.00%
- why: strong earnings gap, well above 200d trend, strong sector momentum backdrop
- md_volume_30d: 1906183268
- chart: https://www.tradingview.com/chart/?symbol=PANW

### CBRL
- sector: Consumer Discretionary
- predicted_alpha: 0.099965
- calibrated_p_beat_sector: 88.00%
- why: dollar volume ratio 20 60, supportive distance above 200d, healthy 200d sector breadth
- md_volume_30d: 39846797
- chart: https://www.tradingview.com/chart/?symbol=CBRL

### DV
- sector: Communication Services
- predicted_alpha: 0.093119
- calibrated_p_beat_sector: 84.48%
- why: strong earnings volume, constructive ATR profile, volume dry-up into setup
- md_volume_30d: 26037013
- chart: https://www.tradingview.com/chart/?symbol=DV

### PATH
- sector: Information Technology
- predicted_alpha: 0.091434
- calibrated_p_beat_sector: 84.48%
- why: supportive atr 14, supportive within-sector atr 14, supportive days since earnings
- md_volume_30d: 933239678
- chart: https://www.tradingview.com/chart/?symbol=PATH

### HOOD
- sector: Financials
- predicted_alpha: 0.090273
- calibrated_p_beat_sector: 84.48%
- why: distance from 52w high, supportive avg gap, atr pct 14
- md_volume_30d: 1682588823
- chart: https://www.tradingview.com/chart/?symbol=HOOD

### ZD
- sector: Communication Services
- predicted_alpha: 0.084124
- calibrated_p_beat_sector: 84.48%
- why: strong earnings volume, strong 126d momentum, earnings breakout open
- md_volume_30d: 25224560
- chart: https://www.tradingview.com/chart/?symbol=ZD

### HPE
- sector: Information Technology
- predicted_alpha: 0.075403
- calibrated_p_beat_sector: 84.48%
- why: well above 200d trend, strong sector momentum backdrop, atr pct 14
- md_volume_30d: 688614736
- chart: https://www.tradingview.com/chart/?symbol=HPE

### COTY
- sector: Consumer Staples
- predicted_alpha: 0.071057
- calibrated_p_beat_sector: 84.48%
- why: limited recent downside gap risk, supportive atr pct 14, holding above earnings close
- md_volume_30d: 20756282
- chart: https://www.tradingview.com/chart/?symbol=COTY
