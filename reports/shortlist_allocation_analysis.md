# Shortlist Allocation Analysis

- generated_at: global=2026-05-26T20:08:23+00:00, sector_specific=2026-05-28T16:49:10+00:00
- selected_model: xgboost_model
- horizon_days: 20
- top_n: 6
- eligible_universe_mode: passed_or_trend
- model_scope: compare_scopes

## Scope Summary

### global
- generated_at: 2026-05-26T20:08:23+00:00
- run_champion_model: xgboost_model
- selected_model: xgboost_model

### sector_specific
- generated_at: 2026-05-28T16:49:10+00:00
- run_champion_model: xgboost_model
- selected_model: xgboost_model

## Full OOS Scope Comparison

### raw_top_n
- global: mean_target=0.067201, beat_universe_rate=0.678125, hit_rate=0.601042, avg_max_sector_share=0.486823
- sector_specific: mean_target=0.089993, beat_universe_rate=0.728125, hit_rate=0.637500, avg_max_sector_share=0.508698

### sector_cap_3
- global: mean_target=0.061158, beat_universe_rate=0.640625, hit_rate=0.583802, avg_max_sector_share=0.424062
- sector_specific: mean_target=0.080224, beat_universe_rate=0.709375, hit_rate=0.625833, avg_max_sector_share=0.435000

### sector_cap_2
- global: mean_target=0.057088, beat_universe_rate=0.662500, hit_rate=0.578698, avg_max_sector_share=0.346979
- sector_specific: mean_target=0.074279, beat_universe_rate=0.696875, hit_rate=0.616719, avg_max_sector_share=0.346979

### sector_round_robin
- global: mean_target=0.046853, beat_universe_rate=0.596875, hit_rate=0.571875, avg_max_sector_share=0.228490
- sector_specific: mean_target=0.068516, beat_universe_rate=0.690625, hit_rate=0.615104, avg_max_sector_share=0.228490

## Recent 60 OOS Dates Scope Comparison

### raw_top_n
- global: mean_target=0.123527, beat_universe_rate=0.866667, hit_rate=0.680556, avg_max_sector_share=0.744444
- sector_specific: mean_target=0.133860, beat_universe_rate=0.766667, hit_rate=0.750000, avg_max_sector_share=0.733333

### sector_cap_3
- global: mean_target=0.094409, beat_universe_rate=0.800000, hit_rate=0.616667, avg_max_sector_share=0.497222
- sector_specific: mean_target=0.106861, beat_universe_rate=0.783333, hit_rate=0.755556, avg_max_sector_share=0.488889

### sector_cap_2
- global: mean_target=0.078550, beat_universe_rate=0.800000, hit_rate=0.586111, avg_max_sector_share=0.333333
- sector_specific: mean_target=0.096085, beat_universe_rate=0.816667, hit_rate=0.755556, avg_max_sector_share=0.333333

### sector_round_robin
- global: mean_target=0.041656, beat_universe_rate=0.583333, hit_rate=0.541667, avg_max_sector_share=0.200000
- sector_specific: mean_target=0.084939, beat_universe_rate=0.816667, hit_rate=0.747222, avg_max_sector_share=0.200000

## Live Allocation Shapes

### global
- selected_model: xgboost_model

#### raw_top_n

- OGN (Health Care) predicted_alpha=0.391333
- CC (Materials) predicted_alpha=0.103704
- VICR (Industrials) predicted_alpha=0.097784
- MU (Information Technology) predicted_alpha=0.089561
- WDC (Information Technology) predicted_alpha=0.072249
- POWL (Industrials) predicted_alpha=0.069390

#### sector_cap_3

- OGN (Health Care) predicted_alpha=0.391333
- CC (Materials) predicted_alpha=0.103704
- VICR (Industrials) predicted_alpha=0.097784
- MU (Information Technology) predicted_alpha=0.089561
- WDC (Information Technology) predicted_alpha=0.072249
- POWL (Industrials) predicted_alpha=0.069390

#### sector_cap_2

- OGN (Health Care) predicted_alpha=0.391333
- CC (Materials) predicted_alpha=0.103704
- VICR (Industrials) predicted_alpha=0.097784
- MU (Information Technology) predicted_alpha=0.089561
- WDC (Information Technology) predicted_alpha=0.072249
- POWL (Industrials) predicted_alpha=0.069390

#### sector_round_robin

- OGN (Health Care) predicted_alpha=0.391333
- CC (Materials) predicted_alpha=0.103704
- VICR (Industrials) predicted_alpha=0.097784
- MU (Information Technology) predicted_alpha=0.089561
- IAC (Communication Services) predicted_alpha=0.065855
- PARR (Energy) predicted_alpha=0.051096

### sector_specific
- selected_model: xgboost_model

#### raw_top_n

- ATEN (Information Technology) predicted_alpha=0.261542
- APA (Energy) predicted_alpha=0.226093
- ENPH (Information Technology) predicted_alpha=0.214732
- UNIT (Communication Services) predicted_alpha=0.169246
- CC (Materials) predicted_alpha=0.164171
- AGX (Industrials) predicted_alpha=0.149234

#### sector_cap_3

- ATEN (Information Technology) predicted_alpha=0.261542
- APA (Energy) predicted_alpha=0.226093
- ENPH (Information Technology) predicted_alpha=0.214732
- UNIT (Communication Services) predicted_alpha=0.169246
- CC (Materials) predicted_alpha=0.164171
- AGX (Industrials) predicted_alpha=0.149234

#### sector_cap_2

- ATEN (Information Technology) predicted_alpha=0.261542
- APA (Energy) predicted_alpha=0.226093
- ENPH (Information Technology) predicted_alpha=0.214732
- UNIT (Communication Services) predicted_alpha=0.169246
- CC (Materials) predicted_alpha=0.164171
- AGX (Industrials) predicted_alpha=0.149234

#### sector_round_robin

- ATEN (Information Technology) predicted_alpha=0.261542
- APA (Energy) predicted_alpha=0.226093
- UNIT (Communication Services) predicted_alpha=0.169246
- CC (Materials) predicted_alpha=0.164171
- AGX (Industrials) predicted_alpha=0.149234
- OGN (Health Care) predicted_alpha=0.143412
