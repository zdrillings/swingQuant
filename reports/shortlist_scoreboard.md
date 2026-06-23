# Shortlist Scoreboard

- generated_at: 2026-05-28T16:49:10+00:00
- horizon_days: 20
- top_n: 10
- eligible_universe_mode: passed_or_trend
- model_scope: sector_specific
- run_champion_model: xgboost_model
- recommended_model: xgboost_model
- production_model: xgboost_model
- production_eligible_universe_mode: passed_or_trend
- production_model_scope: sector_specific
- live_snapshot_date: 2026-05-19

## Promotion Decisions

### signal_proxy
- decision: promote_minimum
- full_status: minimum
- recent40_status: strong
- full_mean_target: 0.029088
- full_beat_universe_rate: 0.590625
- full_positive_date_rate: 0.659375
- recent40_mean_target: 0.048392
- recent40_beat_universe_rate: 0.650000
- recent40_positive_date_rate: 0.800000
- avg_max_sector_share: 0.467711
- avg_sector_hhi: 0.345361

### xgboost_model
- decision: promote_strong
- full_status: strong
- recent40_status: strong
- full_mean_target: 0.081160
- full_beat_universe_rate: 0.753125
- full_positive_date_rate: 0.793750
- recent40_mean_target: 0.094385
- recent40_beat_universe_rate: 0.925000
- recent40_positive_date_rate: 1.000000
- avg_max_sector_share: 0.468336
- avg_sector_hhi: 0.360486

### ensemble_model
- decision: promote_strong
- full_status: strong
- recent40_status: strong
- full_mean_target: 0.058286
- full_beat_universe_rate: 0.693750
- full_positive_date_rate: 0.756250
- recent40_mean_target: 0.098824
- recent40_beat_universe_rate: 0.925000
- recent40_positive_date_rate: 0.975000
- avg_max_sector_share: 0.507711
- avg_sector_hhi: 0.400548

### ridge_model
- decision: promote_strong
- full_status: strong
- recent40_status: strong
- full_mean_target: 0.049611
- full_beat_universe_rate: 0.681250
- full_positive_date_rate: 0.725000
- recent40_mean_target: 0.102028
- recent40_beat_universe_rate: 0.825000
- recent40_positive_date_rate: 0.975000
- avg_max_sector_share: 0.499898
- avg_sector_hhi: 0.385548

## Full OOS Scoreboard

### xgboost_model
- dates: 320
- mean_target: 0.081160
- hit_rate: 0.647298
- beat_universe_rate: 0.753125
- positive_date_rate: 0.793750
- ge_2pct_rate: 0.690625
- ge_5pct_rate: 0.581250
- avg_max_sector_share: 0.468336
- avg_sector_hhi: 0.360486

### ensemble_model
- dates: 320
- mean_target: 0.058286
- hit_rate: 0.611985
- beat_universe_rate: 0.693750
- positive_date_rate: 0.756250
- ge_2pct_rate: 0.650000
- ge_5pct_rate: 0.521875
- avg_max_sector_share: 0.507711
- avg_sector_hhi: 0.400548

### ridge_model
- dates: 320
- mean_target: 0.049611
- hit_rate: 0.575423
- beat_universe_rate: 0.681250
- positive_date_rate: 0.725000
- ge_2pct_rate: 0.628125
- ge_5pct_rate: 0.484375
- avg_max_sector_share: 0.499898
- avg_sector_hhi: 0.385548

### signal_proxy
- dates: 320
- mean_target: 0.029088
- hit_rate: 0.539173
- beat_universe_rate: 0.590625
- positive_date_rate: 0.659375
- ge_2pct_rate: 0.550000
- ge_5pct_rate: 0.371875
- avg_max_sector_share: 0.467711
- avg_sector_hhi: 0.345361

## Recent 40 OOS Dates

### ridge_model
- dates: 40
- mean_target: 0.102028
- hit_rate: 0.735000
- beat_universe_rate: 0.825000
- positive_date_rate: 0.975000
- ge_2pct_rate: 0.925000
- ge_5pct_rate: 0.825000
- avg_max_sector_share: 0.652500
- avg_sector_hhi: 0.545500

### ensemble_model
- dates: 40
- mean_target: 0.098824
- hit_rate: 0.742500
- beat_universe_rate: 0.925000
- positive_date_rate: 0.975000
- ge_2pct_rate: 0.875000
- ge_5pct_rate: 0.800000
- avg_max_sector_share: 0.767500
- avg_sector_hhi: 0.677500

### xgboost_model
- dates: 40
- mean_target: 0.094385
- hit_rate: 0.787500
- beat_universe_rate: 0.925000
- positive_date_rate: 1.000000
- ge_2pct_rate: 0.875000
- ge_5pct_rate: 0.750000
- avg_max_sector_share: 0.607500
- avg_sector_hhi: 0.457000

### signal_proxy
- dates: 40
- mean_target: 0.048392
- hit_rate: 0.600000
- beat_universe_rate: 0.650000
- positive_date_rate: 0.800000
- ge_2pct_rate: 0.725000
- ge_5pct_rate: 0.450000
- avg_max_sector_share: 0.570000
- avg_sector_hhi: 0.438000

## Window Scorecards

### ensemble_model

- last_20d_mean_target: 0.097789
- last_20d_beat_universe_rate: 0.900000
- last_20d_positive_date_rate: 1.000000
- last_40d_mean_target: 0.098824
- last_40d_beat_universe_rate: 0.925000
- last_40d_positive_date_rate: 0.975000
- last_60d_mean_target: 0.109403
- last_60d_beat_universe_rate: 0.933333
- last_60d_positive_date_rate: 0.933333

### ridge_model

- last_20d_mean_target: 0.083244
- last_20d_beat_universe_rate: 0.700000
- last_20d_positive_date_rate: 1.000000
- last_40d_mean_target: 0.102028
- last_40d_beat_universe_rate: 0.825000
- last_40d_positive_date_rate: 0.975000
- last_60d_mean_target: 0.109980
- last_60d_beat_universe_rate: 0.883333
- last_60d_positive_date_rate: 0.933333

### signal_proxy

- last_20d_mean_target: 0.062264
- last_20d_beat_universe_rate: 0.600000
- last_20d_positive_date_rate: 0.900000
- last_40d_mean_target: 0.048392
- last_40d_beat_universe_rate: 0.650000
- last_40d_positive_date_rate: 0.800000
- last_60d_mean_target: 0.041028
- last_60d_beat_universe_rate: 0.683333
- last_60d_positive_date_rate: 0.750000

### xgboost_model

- last_20d_mean_target: 0.091425
- last_20d_beat_universe_rate: 0.850000
- last_20d_positive_date_rate: 1.000000
- last_40d_mean_target: 0.094385
- last_40d_beat_universe_rate: 0.925000
- last_40d_positive_date_rate: 1.000000
- last_60d_mean_target: 0.113561
- last_60d_beat_universe_rate: 0.916667
- last_60d_positive_date_rate: 0.916667

## Sector Scorecards

### ensemble_model

- Health Care: avg_pick_count=1.537313, mean_target=0.101231, hit_rate=0.634328
- Information Technology: avg_pick_count=4.151020, mean_target=0.068736, hit_rate=0.645491
- Materials: avg_pick_count=1.705607, mean_target=0.067551, hit_rate=0.631776
- Communication Services: avg_pick_count=1.765957, mean_target=0.051824, hit_rate=0.552305
- Energy: avg_pick_count=2.468927, mean_target=0.023752, hit_rate=0.616972
- Consumer Staples: avg_pick_count=1.137931, mean_target=0.022835, hit_rate=0.591954
- Financials: avg_pick_count=1.953488, mean_target=0.019779, hit_rate=0.432392
- Industrials: avg_pick_count=2.184049, mean_target=0.018491, hit_rate=0.485612
- Utilities: avg_pick_count=1.118644, mean_target=0.010718, hit_rate=0.593220
- Consumer Discretionary: avg_pick_count=1.820225, mean_target=0.006676, hit_rate=0.466506
- Real Estate: avg_pick_count=1.000000, mean_target=-0.041070, hit_rate=0.000000

### ridge_model

- Materials: avg_pick_count=2.489960, mean_target=0.080745, hit_rate=0.632202
- Industrials: avg_pick_count=1.578431, mean_target=0.067835, hit_rate=0.598039
- Health Care: avg_pick_count=1.364341, mean_target=0.060071, hit_rate=0.563307
- Information Technology: avg_pick_count=3.270642, mean_target=0.053172, hit_rate=0.597888
- Energy: avg_pick_count=2.473373, mean_target=0.031377, hit_rate=0.566481
- Communication Services: avg_pick_count=1.744681, mean_target=0.012268, hit_rate=0.431560
- Utilities: avg_pick_count=1.607143, mean_target=0.004611, hit_rate=0.589286
- Consumer Discretionary: avg_pick_count=1.411765, mean_target=-0.006543, hit_rate=0.438725
- Consumer Staples: avg_pick_count=1.776471, mean_target=-0.009390, hit_rate=0.398235
- Financials: avg_pick_count=2.439306, mean_target=-0.029042, hit_rate=0.289515
- Real Estate: avg_pick_count=1.000000, mean_target=-0.104498, hit_rate=0.000000

### signal_proxy

- Information Technology: avg_pick_count=3.740157, mean_target=0.058141, hit_rate=0.574470
- Energy: avg_pick_count=2.519231, mean_target=0.031542, hit_rate=0.631896
- Materials: avg_pick_count=1.537143, mean_target=0.030382, hit_rate=0.509524
- Real Estate: avg_pick_count=1.000000, mean_target=0.024612, hit_rate=0.727273
- Industrials: avg_pick_count=2.252918, mean_target=0.018435, hit_rate=0.496822
- Financials: avg_pick_count=1.358491, mean_target=0.006082, hit_rate=0.500786
- Communication Services: avg_pick_count=1.367816, mean_target=0.004008, hit_rate=0.444444
- Consumer Discretionary: avg_pick_count=1.780822, mean_target=-0.009094, hit_rate=0.440525
- Consumer Staples: avg_pick_count=1.117647, mean_target=-0.010082, hit_rate=0.460784
- Health Care: avg_pick_count=1.680982, mean_target=-0.012433, hit_rate=0.412986
- Utilities: avg_pick_count=1.151515, mean_target=-0.020493, hit_rate=0.373737

### xgboost_model

- Health Care: avg_pick_count=1.595588, mean_target=0.155668, hit_rate=0.594975
- Consumer Staples: avg_pick_count=1.183333, mean_target=0.121557, hit_rate=0.769444
- Communication Services: avg_pick_count=1.993151, mean_target=0.113093, hit_rate=0.658447
- Information Technology: avg_pick_count=3.336538, mean_target=0.112756, hit_rate=0.675794
- Materials: avg_pick_count=1.855721, mean_target=0.102663, hit_rate=0.637894
- Energy: avg_pick_count=2.578947, mean_target=0.053289, hit_rate=0.721404
- Industrials: avg_pick_count=1.868421, mean_target=0.044534, hit_rate=0.575768
- Financials: avg_pick_count=2.684211, mean_target=0.035892, hit_rate=0.523863
- Consumer Discretionary: avg_pick_count=1.474820, mean_target=0.010296, hit_rate=0.507914
- Real Estate: avg_pick_count=1.000000, mean_target=0.010244, hit_rate=0.545455
- Utilities: avg_pick_count=1.268293, mean_target=-0.001090, hit_rate=0.560976

## Live Concentration

### ensemble_model

- live_max_sector_share: 0.900000
- live_sector_hhi: 0.820000
- live_sector_mix: Information Technology 90%, Energy 10%

### ridge_model

- live_max_sector_share: 0.800000
- live_sector_hhi: 0.660000
- live_sector_mix: Information Technology 80%, Industrials 10%, Materials 10%

### signal_proxy

- live_max_sector_share: 0.900000
- live_sector_hhi: 0.820000
- live_sector_mix: Information Technology 90%, Industrials 10%

### xgboost_model

- live_max_sector_share: 0.500000
- live_sector_hhi: 0.300000
- live_sector_mix: Information Technology 50%, Energy 10%, Communication Services 10%, Materials 10%, Industrials 10%, Health Care 10%
