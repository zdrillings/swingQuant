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
- generated_at: 2026-08-21T23:36:54+00:00

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

### Recent Acceptance Windows

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

## Champion Rolling Acceptance Windows

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
- snapshot_date: 2026-08-21

### DV
- sector: Communication Services
- predicted_alpha: 0.219343
- calibrated_p_beat_sector: 100.00%
- why: constructive ATR profile, strong earnings volume, healthy RSI 14
- md_volume_30d: 20027760
- chart: https://www.tradingview.com/chart/?symbol=DV

### SEZL
- sector: Financials
- predicted_alpha: 0.106034
- calibrated_p_beat_sector: 88.00%
- why: distance from 52w high, limited recent downside gap risk, supportive 126d momentum
- md_volume_30d: 76108931
- chart: https://www.tradingview.com/chart/?symbol=SEZL

### CDW
- sector: Information Technology
- predicted_alpha: 0.096866
- calibrated_p_beat_sector: 84.48%
- why: limited recent downside gap risk, earnings breakout open, strong earnings gap
- md_volume_30d: 206323352
- chart: https://www.tradingview.com/chart/?symbol=CDW

### DELL
- sector: Information Technology
- predicted_alpha: 0.095076
- calibrated_p_beat_sector: 84.48%
- why: well above 200d trend, strong sector momentum backdrop, strong 126d momentum
- md_volume_30d: 2122593235
- chart: https://www.tradingview.com/chart/?symbol=DELL

### PANW
- sector: Information Technology
- predicted_alpha: 0.092457
- calibrated_p_beat_sector: 84.48%
- why: strong earnings gap, well above 200d trend, strong sector momentum backdrop
- md_volume_30d: 1911500626
- chart: https://www.tradingview.com/chart/?symbol=PANW

### PATH
- sector: Information Technology
- predicted_alpha: 0.085802
- calibrated_p_beat_sector: 84.48%
- why: supportive within-sector atr 14, supportive atr 14, supportive days since earnings
- md_volume_30d: 914220365
- chart: https://www.tradingview.com/chart/?symbol=PATH

### ZD
- sector: Communication Services
- predicted_alpha: 0.082050
- calibrated_p_beat_sector: 84.48%
- why: strong earnings volume, earnings breakout open, strong 126d momentum
- md_volume_30d: 25288476
- chart: https://www.tradingview.com/chart/?symbol=ZD

### COTY
- sector: Consumer Staples
- predicted_alpha: 0.077084
- calibrated_p_beat_sector: 84.48%
- why: limited recent downside gap risk, supportive days to earnings, supportive atr pct 14
- md_volume_30d: 21377222
- chart: https://www.tradingview.com/chart/?symbol=COTY

### TECH
- sector: Health Care
- predicted_alpha: 0.060618
- calibrated_p_beat_sector: 82.22%
- why: active price discovery, strong earnings volume, constructive ATR profile
- md_volume_30d: 218151180
- chart: https://www.tradingview.com/chart/?symbol=TECH

### NEO
- sector: Health Care
- predicted_alpha: 0.059261
- calibrated_p_beat_sector: 80.30%
- why: supportive days to earnings, strong earnings volume, supportive atr 14
- md_volume_30d: 27007983
- chart: https://www.tradingview.com/chart/?symbol=NEO
