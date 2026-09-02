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

- eligible_rows: 256752
- eligible_dates: 649
- oos_prediction_dates: 20
- champion_model: xgboost_model
- oos_predictions_csv: /home/zdrillings/code/SwingQuant/reports/shortlist_model_oos_predictions.csv
- live_predictions_csv: /home/zdrillings/code/SwingQuant/reports/shortlist_model_live_predictions.csv
- generated_at: 2026-09-02T00:28:32+00:00

## Full Walk-Forward Evaluation

### xgboost_model
- dates: 20
- avg_pick_count: 9.800000
- mean_target: 0.194660
- hit_rate: 0.863333
- beat_universe_rate: 0.900000
- positive_date_rate: 0.900000
- ge_2pct_rate: 0.850000
- ge_5pct_rate: 0.850000

### ensemble_model
- dates: 20
- avg_pick_count: 9.800000
- mean_target: 0.116939
- hit_rate: 0.728333
- beat_universe_rate: 0.850000
- positive_date_rate: 0.800000
- ge_2pct_rate: 0.800000
- ge_5pct_rate: 0.800000

### ridge_model
- dates: 20
- avg_pick_count: 9.800000
- mean_target: 0.081731
- hit_rate: 0.613333
- beat_universe_rate: 0.800000
- positive_date_rate: 0.750000
- ge_2pct_rate: 0.650000
- ge_5pct_rate: 0.600000

### lasso_model
- dates: 20
- avg_pick_count: 9.800000
- mean_target: 0.075755
- hit_rate: 0.628333
- beat_universe_rate: 0.650000
- positive_date_rate: 0.700000
- ge_2pct_rate: 0.550000
- ge_5pct_rate: 0.550000

### signal_proxy
- dates: 20
- avg_pick_count: 9.800000
- mean_target: 0.009244
- hit_rate: 0.488333
- beat_universe_rate: 0.600000
- positive_date_rate: 0.500000
- ge_2pct_rate: 0.500000
- ge_5pct_rate: 0.300000

## Recent 20 Walk-Forward Dates

### xgboost_model
- dates: 20
- avg_pick_count: 9.800000
- mean_target: 0.194660
- hit_rate: 0.863333
- beat_universe_rate: 0.900000
- positive_date_rate: 0.900000
- ge_2pct_rate: 0.850000
- ge_5pct_rate: 0.850000

### ensemble_model
- dates: 20
- avg_pick_count: 9.800000
- mean_target: 0.116939
- hit_rate: 0.728333
- beat_universe_rate: 0.850000
- positive_date_rate: 0.800000
- ge_2pct_rate: 0.800000
- ge_5pct_rate: 0.800000

### ridge_model
- dates: 20
- avg_pick_count: 9.800000
- mean_target: 0.081731
- hit_rate: 0.613333
- beat_universe_rate: 0.800000
- positive_date_rate: 0.750000
- ge_2pct_rate: 0.650000
- ge_5pct_rate: 0.600000

### lasso_model
- dates: 20
- avg_pick_count: 9.800000
- mean_target: 0.075755
- hit_rate: 0.628333
- beat_universe_rate: 0.650000
- positive_date_rate: 0.700000
- ge_2pct_rate: 0.550000
- ge_5pct_rate: 0.550000

### signal_proxy
- dates: 20
- avg_pick_count: 9.800000
- mean_target: 0.009244
- hit_rate: 0.488333
- beat_universe_rate: 0.600000
- positive_date_rate: 0.500000
- ge_2pct_rate: 0.500000
- ge_5pct_rate: 0.300000

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

### xgboost_model_20d
- dates: 20
- avg_pick_count: 9.800000
- mean_target: 0.194660
- hit_rate: 0.863333
- beat_universe_rate: 0.900000
- positive_date_rate: 0.900000
- ge_2pct_rate: 0.850000
- ge_5pct_rate: 0.850000

### xgboost_model_60d
- dates: 20
- avg_pick_count: 9.800000
- mean_target: 0.194660
- hit_rate: 0.863333
- beat_universe_rate: 0.900000
- positive_date_rate: 0.900000
- ge_2pct_rate: 0.850000
- ge_5pct_rate: 0.850000

### xgboost_model_last_3fold
- dates: 3
- avg_pick_count: 10.000000
- mean_target: 0.142016
- hit_rate: 0.900000
- beat_universe_rate: 1.000000
- positive_date_rate: 1.000000
- ge_2pct_rate: 1.000000
- ge_5pct_rate: 1.000000

### ensemble_model_20d
- dates: 20
- avg_pick_count: 9.800000
- mean_target: 0.116939
- hit_rate: 0.728333
- beat_universe_rate: 0.850000
- positive_date_rate: 0.800000
- ge_2pct_rate: 0.800000
- ge_5pct_rate: 0.800000

### ensemble_model_60d
- dates: 20
- avg_pick_count: 9.800000
- mean_target: 0.116939
- hit_rate: 0.728333
- beat_universe_rate: 0.850000
- positive_date_rate: 0.800000
- ge_2pct_rate: 0.800000
- ge_5pct_rate: 0.800000

### xgboost_model_last_1fold
- dates: 1
- avg_pick_count: 10.000000
- mean_target: 0.113319
- hit_rate: 0.900000
- beat_universe_rate: 1.000000
- positive_date_rate: 1.000000
- ge_2pct_rate: 1.000000
- ge_5pct_rate: 1.000000

### ridge_model_20d
- dates: 20
- avg_pick_count: 9.800000
- mean_target: 0.081731
- hit_rate: 0.613333
- beat_universe_rate: 0.800000
- positive_date_rate: 0.750000
- ge_2pct_rate: 0.650000
- ge_5pct_rate: 0.600000

### ridge_model_60d
- dates: 20
- avg_pick_count: 9.800000
- mean_target: 0.081731
- hit_rate: 0.613333
- beat_universe_rate: 0.800000
- positive_date_rate: 0.750000
- ge_2pct_rate: 0.650000
- ge_5pct_rate: 0.600000

### lasso_model_20d
- dates: 20
- avg_pick_count: 9.800000
- mean_target: 0.075755
- hit_rate: 0.628333
- beat_universe_rate: 0.650000
- positive_date_rate: 0.700000
- ge_2pct_rate: 0.550000
- ge_5pct_rate: 0.550000

### lasso_model_60d
- dates: 20
- avg_pick_count: 9.800000
- mean_target: 0.075755
- hit_rate: 0.628333
- beat_universe_rate: 0.650000
- positive_date_rate: 0.700000
- ge_2pct_rate: 0.550000
- ge_5pct_rate: 0.550000

### ensemble_model_last_1fold
- dates: 1
- avg_pick_count: 10.000000
- mean_target: 0.066471
- hit_rate: 0.800000
- beat_universe_rate: 1.000000
- positive_date_rate: 1.000000
- ge_2pct_rate: 1.000000
- ge_5pct_rate: 1.000000

### ensemble_model_last_3fold
- dates: 3
- avg_pick_count: 10.000000
- mean_target: 0.037451
- hit_rate: 0.633333
- beat_universe_rate: 1.000000
- positive_date_rate: 0.666667
- ge_2pct_rate: 0.666667
- ge_5pct_rate: 0.666667

### signal_proxy_20d
- dates: 20
- avg_pick_count: 9.800000
- mean_target: 0.009244
- hit_rate: 0.488333
- beat_universe_rate: 0.600000
- positive_date_rate: 0.500000
- ge_2pct_rate: 0.500000
- ge_5pct_rate: 0.300000

### signal_proxy_60d
- dates: 20
- avg_pick_count: 9.800000
- mean_target: 0.009244
- hit_rate: 0.488333
- beat_universe_rate: 0.600000
- positive_date_rate: 0.500000
- ge_2pct_rate: 0.500000
- ge_5pct_rate: 0.300000

### ridge_model_last_3fold
- dates: 3
- avg_pick_count: 10.000000
- mean_target: -0.002292
- hit_rate: 0.566667
- beat_universe_rate: 0.666667
- positive_date_rate: 0.333333
- ge_2pct_rate: 0.333333
- ge_5pct_rate: 0.333333

### ridge_model_last_1fold
- dates: 1
- avg_pick_count: 10.000000
- mean_target: -0.016764
- hit_rate: 0.500000
- beat_universe_rate: 1.000000
- positive_date_rate: 0.000000
- ge_2pct_rate: 0.000000
- ge_5pct_rate: 0.000000

### lasso_model_last_1fold
- dates: 1
- avg_pick_count: 10.000000
- mean_target: -0.024060
- hit_rate: 0.700000
- beat_universe_rate: 0.000000
- positive_date_rate: 0.000000
- ge_2pct_rate: 0.000000
- ge_5pct_rate: 0.000000

### lasso_model_last_3fold
- dates: 3
- avg_pick_count: 10.000000
- mean_target: -0.025301
- hit_rate: 0.533333
- beat_universe_rate: 0.000000
- positive_date_rate: 0.333333
- ge_2pct_rate: 0.000000
- ge_5pct_rate: 0.000000

### signal_proxy_last_3fold
- dates: 3
- avg_pick_count: 10.000000
- mean_target: -0.049715
- hit_rate: 0.266667
- beat_universe_rate: 0.333333
- positive_date_rate: 0.333333
- ge_2pct_rate: 0.333333
- ge_5pct_rate: 0.333333

### signal_proxy_last_1fold
- dates: 1
- avg_pick_count: 10.000000
- mean_target: -0.125253
- hit_rate: 0.200000
- beat_universe_rate: 0.000000
- positive_date_rate: 0.000000
- ge_2pct_rate: 0.000000
- ge_5pct_rate: 0.000000

## Champion Rolling Acceptance Windows

### xgboost_model_20d
- dates: 20
- avg_pick_count: 9.800000
- mean_target: 0.194660
- hit_rate: 0.863333
- beat_universe_rate: 0.900000
- positive_date_rate: 0.900000
- ge_2pct_rate: 0.850000
- ge_5pct_rate: 0.850000

### xgboost_model_40d
- dates: 20
- avg_pick_count: 9.800000
- mean_target: 0.194660
- hit_rate: 0.863333
- beat_universe_rate: 0.900000
- positive_date_rate: 0.900000
- ge_2pct_rate: 0.850000
- ge_5pct_rate: 0.850000

### xgboost_model_60d
- dates: 20
- avg_pick_count: 9.800000
- mean_target: 0.194660
- hit_rate: 0.863333
- beat_universe_rate: 0.900000
- positive_date_rate: 0.900000
- ge_2pct_rate: 0.850000
- ge_5pct_rate: 0.850000

### xgboost_model_last_3fold
- dates: 3
- avg_pick_count: 10.000000
- mean_target: 0.142016
- hit_rate: 0.900000
- beat_universe_rate: 1.000000
- positive_date_rate: 1.000000
- ge_2pct_rate: 1.000000
- ge_5pct_rate: 1.000000

### xgboost_model_last_1fold
- dates: 1
- avg_pick_count: 10.000000
- mean_target: 0.113319
- hit_rate: 0.900000
- beat_universe_rate: 1.000000
- positive_date_rate: 1.000000
- ge_2pct_rate: 1.000000
- ge_5pct_rate: 1.000000

## Champion Sector Contribution

### Information Technology
- dates: 17
- avg_pick_count: 3.352941
- mean_target: 0.277922
- hit_rate: 0.916667

### Health Care
- dates: 13
- avg_pick_count: 1.461538
- mean_target: 0.227442
- hit_rate: 0.923077

### Materials
- dates: 13
- avg_pick_count: 1.692308
- mean_target: 0.191870
- hit_rate: 0.807692

### Industrials
- dates: 12
- avg_pick_count: 1.916667
- mean_target: 0.169609
- hit_rate: 0.937500

### Consumer Staples
- dates: 5
- avg_pick_count: 1.200000
- mean_target: 0.157948
- hit_rate: 0.800000

### Energy
- dates: 12
- avg_pick_count: 1.916667
- mean_target: 0.155290
- hit_rate: 0.887500

### Financials
- dates: 8
- avg_pick_count: 1.500000
- mean_target: 0.151630
- hit_rate: 0.875000

### Real Estate
- dates: 2
- avg_pick_count: 1.000000
- mean_target: 0.143059
- hit_rate: 1.000000

### Communication Services
- dates: 9
- avg_pick_count: 1.555556
- mean_target: 0.126994
- hit_rate: 0.722222

### Consumer Discretionary
- dates: 5
- avg_pick_count: 2.600000
- mean_target: 0.115922
- hit_rate: 0.813333

### Utilities
- dates: 3
- avg_pick_count: 1.666667
- mean_target: 0.088460
- hit_rate: 0.833333

## Live Top Candidates

- champion_model: xgboost_model
- snapshot_date: 2026-09-01

### ACHC
- sector: Health Care
- predicted_alpha: 0.202830
- calibrated_p_beat_sector: 95.52%
- why: supportive distance from 52w high, supportive within-sector distance above 20d high, supportive sector breadth 50d
- md_volume_30d: 51532426
- chart: https://www.tradingview.com/chart/?symbol=ACHC

### GDDY
- sector: Information Technology
- predicted_alpha: 0.162819
- calibrated_p_beat_sector: 95.52%
- why: limited recent downside gap risk, strong sector momentum backdrop, strong earnings gap
- md_volume_30d: 149124893
- chart: https://www.tradingview.com/chart/?symbol=GDDY

### DV
- sector: Communication Services
- predicted_alpha: 0.148438
- calibrated_p_beat_sector: 95.52%
- why: supportive atr 14, strong earnings volume, dollar volume ratio 20 60
- md_volume_30d: 26326594
- chart: https://www.tradingview.com/chart/?symbol=DV

### CBRL
- sector: Consumer Discretionary
- predicted_alpha: 0.147014
- calibrated_p_beat_sector: 95.42%
- why: dollar volume ratio 20 60, healthy 50d sector breadth, holding above earnings close
- md_volume_30d: 39846797
- chart: https://www.tradingview.com/chart/?symbol=CBRL

### CDW
- sector: Information Technology
- predicted_alpha: 0.128667
- calibrated_p_beat_sector: 93.68%
- why: limited recent downside gap risk, strong sector momentum backdrop, strong earnings gap
- md_volume_30d: 206452265
- chart: https://www.tradingview.com/chart/?symbol=CDW

### NEO
- sector: Health Care
- predicted_alpha: 0.104675
- calibrated_p_beat_sector: 93.68%
- why: supportive atr pct 14, constructive ATR profile, supportive days to earnings
- md_volume_30d: 27007983
- chart: https://www.tradingview.com/chart/?symbol=NEO

### UGI
- sector: Utilities
- predicted_alpha: 0.093741
- calibrated_p_beat_sector: 91.67%
- why: healthy 200d sector breadth, supportive 126d momentum, active price discovery
- md_volume_30d: 44625746
- chart: https://www.tradingview.com/chart/?symbol=UGI

### WEN
- sector: Consumer Discretionary
- predicted_alpha: 0.076847
- calibrated_p_beat_sector: 90.38%
- why: dollar volume ratio 20 60, healthy 50d sector breadth, healthy 200d sector breadth
- md_volume_30d: 50380780
- chart: https://www.tradingview.com/chart/?symbol=WEN

### KMX
- sector: Consumer Discretionary
- predicted_alpha: 0.073096
- calibrated_p_beat_sector: 82.86%
- why: dollar volume ratio 20 60, healthy 50d sector breadth, healthy 200d sector breadth
- md_volume_30d: 95943530
- chart: https://www.tradingview.com/chart/?symbol=KMX

### VIRT
- sector: Financials
- predicted_alpha: 0.068860
- calibrated_p_beat_sector: 82.86%
- why: supportive 126d momentum, earnings breakout open, healthy 200d sector breadth
- md_volume_30d: 60252464
- chart: https://www.tradingview.com/chart/?symbol=VIRT
