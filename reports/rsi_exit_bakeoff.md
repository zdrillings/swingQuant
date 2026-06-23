# RSI_2 Exit Bakeoff

- benchmark: sector
- recent_scan_dates: 60
- selected_rows: 358
- mature_rows: 228
- scan_date_min: 2026-02-20
- scan_date_max: 2026-06-02
- note: exits are replayed on daily close, not intraday 1-minute prices.

## Variant Summary

### Current Rule
- mature_picks: 228
- mean_return: 1.35%
- median_return: 2.11%
- hit_rate: 79.39%
- mean_alpha_vs_sector: 1.84%
- median_alpha_vs_sector: 2.12%
- positive_alpha_rate: 80.70%
- mean_holding_days: 3.1

### No RSI_2 Exit
- mature_picks: 228
- mean_return: 3.01%
- median_return: 0.95%
- hit_rate: 52.19%
- mean_alpha_vs_sector: 3.77%
- median_alpha_vs_sector: 2.75%
- positive_alpha_rate: 59.21%
- mean_holding_days: 12.7

### RSI_2 Only After +3%
- mature_picks: 228
- mean_return: 2.05%
- median_return: 4.14%
- hit_rate: 78.07%
- mean_alpha_vs_sector: 2.62%
- median_alpha_vs_sector: 3.44%
- positive_alpha_rate: 81.14%
- mean_holding_days: 5.7

### RSI_2 Only After +5%
- mature_picks: 228
- mean_return: 2.46%
- median_return: 5.64%
- hit_rate: 72.81%
- mean_alpha_vs_sector: 3.02%
- median_alpha_vs_sector: 4.24%
- positive_alpha_rate: 76.32%
- mean_holding_days: 7.1

### RSI_2 Only After 3d
- mature_picks: 228
- mean_return: 2.07%
- median_return: 2.68%
- hit_rate: 74.12%
- mean_alpha_vs_sector: 2.61%
- median_alpha_vs_sector: 3.02%
- positive_alpha_rate: 77.63%
- mean_holding_days: 4.7

## Delta Vs Current Rule

### No RSI_2 Exit
- mean_return_delta: 1.66%
- median_return_delta: 0.00%
- mean_alpha_delta_vs_sector: 1.93%
- win_rate_vs_current: 45.61%

### RSI_2 Only After +3%
- mean_return_delta: 0.70%
- median_return_delta: 0.00%
- mean_alpha_delta_vs_sector: 0.77%
- win_rate_vs_current: 39.91%

### RSI_2 Only After +5%
- mean_return_delta: 1.10%
- median_return_delta: 1.22%
- mean_alpha_delta_vs_sector: 1.18%
- win_rate_vs_current: 53.95%

### RSI_2 Only After 3d
- mean_return_delta: 0.72%
- median_return_delta: 0.00%
- mean_alpha_delta_vs_sector: 0.77%
- win_rate_vs_current: 35.96%

## Exit Reason Mix

### Current Rule
- trailing_stop: 3.07%
- rsi_2: 89.47%
- regime_flip: 7.46%

### No RSI_2 Exit
- trailing_stop: 21.05%
- profit_target: 14.04%
- time_limit: 22.81%
- regime_flip: 42.11%

### RSI_2 Only After +3%
- trailing_stop: 5.70%
- rsi_2: 75.00%
- time_limit: 3.95%
- regime_flip: 15.35%

### RSI_2 Only After +5%
- trailing_stop: 8.77%
- rsi_2: 67.54%
- time_limit: 5.26%
- regime_flip: 18.42%

### RSI_2 Only After 3d
- trailing_stop: 4.82%
- rsi_2: 84.21%
- regime_flip: 10.96%

## Slot Breakdown

### energy
- Current Rule: mean_return=2.33%, mean_alpha_vs_sector=1.65%, rsi_exit_rate=95.28%
- No RSI_2 Exit: mean_return=7.93%, mean_alpha_vs_sector=4.99%, rsi_exit_rate=0.00%
- RSI_2 Only After +3%: mean_return=4.10%, mean_alpha_vs_sector=2.62%, rsi_exit_rate=91.51%
- RSI_2 Only After +5%: mean_return=5.05%, mean_alpha_vs_sector=3.20%, rsi_exit_rate=85.85%
- RSI_2 Only After 3d: mean_return=3.72%, mean_alpha_vs_sector=2.68%, rsi_exit_rate=88.68%

### industrials
- Current Rule: mean_return=1.86%, mean_alpha_vs_sector=3.10%, rsi_exit_rate=92.11%
- No RSI_2 Exit: mean_return=5.90%, mean_alpha_vs_sector=8.91%, rsi_exit_rate=0.00%
- RSI_2 Only After +3%: mean_return=2.21%, mean_alpha_vs_sector=4.27%, rsi_exit_rate=63.16%
- RSI_2 Only After +5%: mean_return=2.75%, mean_alpha_vs_sector=4.90%, rsi_exit_rate=60.53%
- RSI_2 Only After 3d: mean_return=2.24%, mean_alpha_vs_sector=3.91%, rsi_exit_rate=92.11%

### materials
- Current Rule: mean_return=-0.11%, mean_alpha_vs_sector=1.51%, rsi_exit_rate=80.95%
- No RSI_2 Exit: mean_return=-4.50%, mean_alpha_vs_sector=-0.09%, rsi_exit_rate=0.00%
- RSI_2 Only After +3%: mean_return=-0.60%, mean_alpha_vs_sector=1.86%, rsi_exit_rate=59.52%
- RSI_2 Only After +5%: mean_return=-0.95%, mean_alpha_vs_sector=1.96%, rsi_exit_rate=47.62%
- RSI_2 Only After 3d: mean_return=-0.09%, mean_alpha_vs_sector=1.95%, rsi_exit_rate=75.00%

## Early RSI_2 Sells

- ECG (industrials, 2026-04-15): current=0.60% on 2026-04-20 vs no_rsi=31.57% on 2026-05-06 (delta 30.97%)
- AESI (energy, 2026-04-15): current=2.05% on 2026-04-16 vs no_rsi=26.80% on 2026-04-27 (delta 24.75%)
- CC (materials, 2026-04-10): current=1.50% on 2026-04-13 vs no_rsi=24.88% on 2026-05-05 (delta 23.38%)
- PTEN (energy, 2026-04-09): current=2.66% on 2026-04-16 vs no_rsi=21.79% on 2026-04-29 (delta 19.13%)
- AGX (industrials, 2026-04-14): current=0.42% on 2026-04-15 vs no_rsi=19.31% on 2026-05-13 (delta 18.89%)
- TALO (energy, 2026-03-02): current=1.47% on 2026-03-03 vs no_rsi=20.00% on 2026-03-20 (delta 18.53%)
- PTEN (energy, 2026-04-16): current=1.44% on 2026-04-21 vs no_rsi=19.31% on 2026-05-15 (delta 17.87%)
- TALO (energy, 2026-03-04): current=1.76% on 2026-03-06 vs no_rsi=18.71% on 2026-03-20 (delta 16.95%)
- TALO (energy, 2026-03-03): current=1.38% on 2026-03-06 vs no_rsi=18.26% on 2026-03-20 (delta 16.88%)
- TALO (energy, 2026-03-05): current=0.23% on 2026-03-06 vs no_rsi=16.92% on 2026-03-20 (delta 16.69%)
