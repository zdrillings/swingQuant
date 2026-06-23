# Shortlist Model

- target_column: alpha_vs_sector_20d
- top_n: 10
- eligible_universe_mode: passed_or_trend
- model_scope: sector_specific
- xgboost_config: balanced_depth4
- min_train_dates: 252
- test_window_dates: 20
- objective: walk-forward cross-sectional ranking of the eligible universe on forward sector-relative alpha
- universe: passed_any_strategy or trend-qualified liquid names (green regime, above 200d, positive 63d momentum, RS vs SPY >= 60)
- feature_matrix: raw features plus date-wise cross-sectional ranks and sector-relative ranks

- eligible_rows: 251637
- eligible_dates: 599
- oos_prediction_dates: 347
- champion_model: xgboost_model
- oos_predictions_csv: /home/zdrillings/code/SwingQuant/reports/shortlist_model_oos_predictions.csv
- live_predictions_csv: /home/zdrillings/code/SwingQuant/reports/shortlist_model_live_predictions.csv
- generated_at: 2026-06-23T01:41:57+00:00

## Full Walk-Forward Evaluation

### xgboost_model
- dates: 347
- avg_pick_count: 9.694524
- mean_target: 0.083073
- hit_rate: 0.648805
- beat_universe_rate: 0.757925
- positive_date_rate: 0.795389
- ge_2pct_rate: 0.688761
- ge_5pct_rate: 0.585014

### ensemble_model
- dates: 347
- avg_pick_count: 9.694524
- mean_target: 0.058794
- hit_rate: 0.604425
- beat_universe_rate: 0.688761
- positive_date_rate: 0.746398
- ge_2pct_rate: 0.648415
- ge_5pct_rate: 0.524496

### ridge_model
- dates: 347
- avg_pick_count: 9.694524
- mean_target: 0.047669
- hit_rate: 0.567249
- beat_universe_rate: 0.662824
- positive_date_rate: 0.708934
- ge_2pct_rate: 0.616715
- ge_5pct_rate: 0.475504

### signal_proxy
- dates: 347
- avg_pick_count: 9.694524
- mean_target: 0.029234
- hit_rate: 0.533819
- beat_universe_rate: 0.602305
- positive_date_rate: 0.662824
- ge_2pct_rate: 0.556196
- ge_5pct_rate: 0.371758

## Recent 60 Walk-Forward Dates

### xgboost_model
- dates: 60
- avg_pick_count: 10.000000
- mean_target: 0.102288
- hit_rate: 0.736667
- beat_universe_rate: 0.866667
- positive_date_rate: 0.916667
- ge_2pct_rate: 0.783333
- ge_5pct_rate: 0.733333

### ensemble_model
- dates: 60
- avg_pick_count: 10.000000
- mean_target: 0.089379
- hit_rate: 0.653333
- beat_universe_rate: 0.783333
- positive_date_rate: 0.816667
- ge_2pct_rate: 0.783333
- ge_5pct_rate: 0.700000

### ridge_model
- dates: 60
- avg_pick_count: 10.000000
- mean_target: 0.072155
- hit_rate: 0.623333
- beat_universe_rate: 0.633333
- positive_date_rate: 0.766667
- ge_2pct_rate: 0.733333
- ge_5pct_rate: 0.616667

### signal_proxy
- dates: 60
- avg_pick_count: 10.000000
- mean_target: 0.044816
- hit_rate: 0.551667
- beat_universe_rate: 0.650000
- positive_date_rate: 0.783333
- ge_2pct_rate: 0.716667
- ge_5pct_rate: 0.450000

## Champion Rolling Acceptance Windows

### xgboost_model_60d
- dates: 60
- avg_pick_count: 10.000000
- mean_target: 0.102288
- hit_rate: 0.736667
- beat_universe_rate: 0.866667
- positive_date_rate: 0.916667
- ge_2pct_rate: 0.783333
- ge_5pct_rate: 0.733333

### xgboost_model_40d
- dates: 40
- avg_pick_count: 10.000000
- mean_target: 0.097928
- hit_rate: 0.692500
- beat_universe_rate: 0.800000
- positive_date_rate: 0.875000
- ge_2pct_rate: 0.750000
- ge_5pct_rate: 0.700000

### xgboost_model_20d
- dates: 20
- avg_pick_count: 10.000000
- mean_target: 0.075456
- hit_rate: 0.655000
- beat_universe_rate: 0.750000
- positive_date_rate: 0.750000
- ge_2pct_rate: 0.600000
- ge_5pct_rate: 0.550000

## Champion Sector Contribution

### Health Care
- dates: 144
- avg_pick_count: 1.569444
- mean_target: 0.145398
- hit_rate: 0.575810

### Consumer Staples
- dates: 60
- avg_pick_count: 1.183333
- mean_target: 0.121557
- hit_rate: 0.769444

### Information Technology
- dates: 235
- avg_pick_count: 3.740426
- mean_target: 0.114909
- hit_rate: 0.679558

### Communication Services
- dates: 166
- avg_pick_count: 1.927711
- mean_target: 0.107270
- hit_rate: 0.669478

### Materials
- dates: 212
- avg_pick_count: 1.830189
- mean_target: 0.092151
- hit_rate: 0.604796

### Energy
- dates: 226
- avg_pick_count: 2.486726
- mean_target: 0.053160
- hit_rate: 0.724661

### Industrials
- dates: 158
- avg_pick_count: 1.835443
- mean_target: 0.050251
- hit_rate: 0.591878

### Financials
- dates: 136
- avg_pick_count: 2.647059
- mean_target: 0.041736
- hit_rate: 0.534366

### Consumer Discretionary
- dates: 139
- avg_pick_count: 1.474820
- mean_target: 0.010296
- hit_rate: 0.507914

### Real Estate
- dates: 11
- avg_pick_count: 1.000000
- mean_target: 0.010244
- hit_rate: 0.545455

### Utilities
- dates: 41
- avg_pick_count: 1.268293
- mean_target: -0.001090
- hit_rate: 0.560976

## Live Top Candidates

- champion_model: xgboost_model
- snapshot_date: 2026-06-05

### CRSR
- sector: Information Technology
- predicted_alpha: 0.478445
- why: holding above recent breakout, tight recent base, supportive atr 14
- md_volume_30d: 20510299
- chart: https://www.tradingview.com/chart/?symbol=CRSR

### SLAB
- sector: Information Technology
- predicted_alpha: 0.420919
- why: supportive atr 14, supportive rs vs group etf, constructive ATR profile
- md_volume_30d: 76506781
- chart: https://www.tradingview.com/chart/?symbol=SLAB

### LUMN
- sector: Communication Services
- predicted_alpha: 0.269070
- why: holding above recent breakout, strong 126d momentum, clear of near-term earnings
- md_volume_30d: 111145269
- chart: https://www.tradingview.com/chart/?symbol=LUMN

### MXL
- sector: Information Technology
- predicted_alpha: 0.150268
- why: holding above recent breakout, well above 200d trend, supportive rs vs group etf
- md_volume_30d: 387947307
- chart: https://www.tradingview.com/chart/?symbol=MXL

### GEN
- sector: Information Technology
- predicted_alpha: 0.148559
- why: supportive atr 14, constructive ATR profile, supportive rsi 14
- md_volume_30d: 179413874
- chart: https://www.tradingview.com/chart/?symbol=GEN

### AMR
- sector: Materials
- predicted_alpha: 0.148320
- why: strong earnings volume, earnings breakout open, limited recent downside gap risk
- md_volume_30d: 48783842
- chart: https://www.tradingview.com/chart/?symbol=AMR

### SITM
- sector: Information Technology
- predicted_alpha: 0.134711
- why: holding above recent breakout, healthy 50d sector breadth, well above 200d trend
- md_volume_30d: 360945413
- chart: https://www.tradingview.com/chart/?symbol=SITM

### PVH
- sector: Consumer Discretionary
- predicted_alpha: 0.132827
- why: limited recent downside gap risk, active price discovery, supportive distance above 20d high
- md_volume_30d: 78416994
- chart: https://www.tradingview.com/chart/?symbol=PVH

### SMCI
- sector: Information Technology
- predicted_alpha: 0.127939
- why: limited recent downside gap risk, tight recent base, holding above recent breakout
- md_volume_30d: 1560757974
- chart: https://www.tradingview.com/chart/?symbol=SMCI

### VICR
- sector: Industrials
- predicted_alpha: 0.125819
- why: strong 126d momentum, constructive ATR profile, limited recent downside gap risk
- md_volume_30d: 213161688
- chart: https://www.tradingview.com/chart/?symbol=VICR
