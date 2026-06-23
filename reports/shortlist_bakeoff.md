# Shortlist Bakeoff

- target_column: alpha_vs_sector_20d
- top_n: 6
- eligible_universe_mode: passed_or_trend
- objective: compare daily shortlist policies on forward sector-relative alpha
- note: universe-model bakeoff uses chronological train/test split on `universe_daily_snapshots`
- note: ridge/xgboost feature matrices use raw features plus date-wise cross-sectional ranks and sector-relative ranks
- note: current scan policy bakeoff uses persisted `Scan_Candidates` rows where forward outcomes are already matured
- eligible_universe: passed_any_strategy or trend-qualified liquid names (green regime, above 200d, positive 63d momentum, RS vs SPY >= 60)

- eligible_rows: 238831
- eligible_dates: 572
- train_dates: 400
- test_dates: 172
- test_date_range: 2025-08-07 -> 2026-04-14

## Universe Model Bakeoff

- policies: equal_weight_eligible, signal_proxy, ridge_model, xgboost_model, ensemble_model

### Full Test Window

### signal_proxy
- dates: 172
- avg_pick_count: 6.000000
- mean_target: 0.068150
- hit_rate: 0.589147
- beat_universe_rate: 0.750000
- positive_date_rate: 0.790698
- ge_2pct_rate: 0.686047
- ge_5pct_rate: 0.523256

### ridge_model
- dates: 172
- avg_pick_count: 6.000000
- mean_target: 0.054029
- hit_rate: 0.618217
- beat_universe_rate: 0.691860
- positive_date_rate: 0.738372
- ge_2pct_rate: 0.668605
- ge_5pct_rate: 0.534884

### ensemble_model
- dates: 172
- avg_pick_count: 6.000000
- mean_target: 0.052784
- hit_rate: 0.609496
- beat_universe_rate: 0.651163
- positive_date_rate: 0.703488
- ge_2pct_rate: 0.656977
- ge_5pct_rate: 0.540698

### xgboost_model
- dates: 172
- avg_pick_count: 6.000000
- mean_target: 0.021457
- hit_rate: 0.522287
- beat_universe_rate: 0.447674
- positive_date_rate: 0.563953
- ge_2pct_rate: 0.470930
- ge_5pct_rate: 0.319767

### equal_weight_eligible
- dates: 172
- avg_pick_count: 440.697674
- mean_target: 0.014756
- hit_rate: 0.527622
- beat_universe_rate: 0.000000
- positive_date_rate: 0.755814
- ge_2pct_rate: 0.418605
- ge_5pct_rate: 0.046512

### Recent 40 Test Dates

### ensemble_model
- dates: 40
- avg_pick_count: 6.000000
- mean_target: 0.052028
- hit_rate: 0.620833
- beat_universe_rate: 0.650000
- positive_date_rate: 0.800000
- ge_2pct_rate: 0.700000
- ge_5pct_rate: 0.550000

### ridge_model
- dates: 40
- avg_pick_count: 6.000000
- mean_target: 0.045304
- hit_rate: 0.641667
- beat_universe_rate: 0.650000
- positive_date_rate: 0.750000
- ge_2pct_rate: 0.650000
- ge_5pct_rate: 0.475000

### signal_proxy
- dates: 40
- avg_pick_count: 6.000000
- mean_target: 0.042916
- hit_rate: 0.604167
- beat_universe_rate: 0.575000
- positive_date_rate: 0.700000
- ge_2pct_rate: 0.600000
- ge_5pct_rate: 0.525000

### equal_weight_eligible
- dates: 40
- avg_pick_count: 344.200000
- mean_target: 0.027285
- hit_rate: 0.558158
- beat_universe_rate: 0.000000
- positive_date_rate: 0.775000
- ge_2pct_rate: 0.650000
- ge_5pct_rate: 0.200000

### xgboost_model
- dates: 40
- avg_pick_count: 6.000000
- mean_target: 0.004751
- hit_rate: 0.491667
- beat_universe_rate: 0.350000
- positive_date_rate: 0.550000
- ge_2pct_rate: 0.425000
- ge_5pct_rate: 0.200000

## Current Scan Policy Bakeoff

### Full Matured Scan Window

### signal_top_n
- dates: 266
- avg_pick_count: 5.819549
- mean_target: 0.012325
- hit_rate: 0.558145
- beat_universe_rate: 0.447368
- positive_date_rate: 0.605263
- ge_2pct_rate: 0.379699
- ge_5pct_rate: 0.157895

### runtime_selected
- dates: 266
- avg_pick_count: 5.785714
- mean_target: 0.003788
- hit_rate: 0.505576
- beat_universe_rate: 0.353383
- positive_date_rate: 0.522556
- ge_2pct_rate: 0.323308
- ge_5pct_rate: 0.105263

### opportunity_top_n
- dates: 266
- avg_pick_count: 5.819549
- mean_target: 0.002065
- hit_rate: 0.492356
- beat_universe_rate: 0.315789
- positive_date_rate: 0.530075
- ge_2pct_rate: 0.319549
- ge_5pct_rate: 0.093985

### Recent 40 Matured Scan Dates

### signal_top_n
- dates: 40
- avg_pick_count: 5.950000
- mean_target: -0.001535
- hit_rate: 0.572917
- beat_universe_rate: 0.325000
- positive_date_rate: 0.475000
- ge_2pct_rate: 0.275000
- ge_5pct_rate: 0.050000

### runtime_selected
- dates: 40
- avg_pick_count: 5.950000
- mean_target: -0.007430
- hit_rate: 0.447917
- beat_universe_rate: 0.250000
- positive_date_rate: 0.375000
- ge_2pct_rate: 0.250000
- ge_5pct_rate: 0.075000

### opportunity_top_n
- dates: 40
- avg_pick_count: 5.950000
- mean_target: -0.009784
- hit_rate: 0.427083
- beat_universe_rate: 0.275000
- positive_date_rate: 0.425000
- ge_2pct_rate: 0.225000
- ge_5pct_rate: 0.075000
