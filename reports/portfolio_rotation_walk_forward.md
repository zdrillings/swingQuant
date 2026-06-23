# Portfolio Rotation Backtest

## Rules
- target_positions: 6
- max_hold_days: 20
- min_pre_penalty_opportunity: 0.40
- min_model_alpha: 0.0000
- transaction_cost_bps: 0.00
- slippage_bps: 0.00
- cooldown_days: 0
- reinvest_gains: true
- max_new_entries_per_scan: unlimited
- date_from: all
- date_to: all
- sell_rule: max_hold_days only in v1
- replacement_rule: fill empty slots from best unheld targets; no churn of existing holdings
- ranking_mode: walk_forward_oos
- ranking_model: xgboost_model
- ranking_generated_at: 2026-06-02T20:08:10+00:00
- ranking_prediction_dates: 2025-01-02 to 2026-04-14

## Caveats
- Walk-forward mode uses persisted out-of-sample shortlist predictions, but still depends on the current stored scan snapshots and v1 portfolio mechanics.
- The v1 exit model only uses max holding days, so it does not yet reflect monitor sell signals, stops, RSI exits, earnings exits, or transaction costs.
- Total return is fully compounded through six equal-weight slots; high 20-day trade returns compound quickly.

## Primary Scorecard

### Rolling Portfolio Windows
- 20d: windows=300, mean=11.58%, median=11.42%, p25_p75=2.43% to 19.77%, p05_p95=-7.83% to 31.31%, hit_rate=80.33%, worst=-13.99% (2025-02-12 to 2025-03-13), best=44.98%
- 60d: windows=260, mean=41.02%, median=48.79%, p25_p75=20.63% to 60.98%, p05_p95=-11.85% to 77.60%, hit_rate=83.08%, worst=-15.66% (2025-01-24 to 2025-04-22), best=97.92%

### Calendar Quarter Returns
- quarters=6, mean=35.65%, median=34.09%, p25_p75=17.41% to 57.14%, hit_rate=83.33%, worst=-4.07% (2025Q1), best=73.12% (2026Q1)


## Summary
- initial_equity: 1.0000
- final_equity: 5.9192
- total_return: 491.92%
- spy_return: 20.51%
- max_drawdown: 19.17%
- eligible_scan_dates: 264
- portfolio_valuation_dates: 320
- completed_or_open_trades: 93
- equity_csv: /home/zdrillings/code/SwingQuant/reports/portfolio_rotation_walk_forward_equity.csv
- trades_csv: /home/zdrillings/code/SwingQuant/reports/portfolio_rotation_walk_forward_trades.csv
- rolling_windows_csv: /home/zdrillings/code/SwingQuant/reports/portfolio_rotation_walk_forward_rolling_windows.csv
- quarterly_returns_csv: /home/zdrillings/code/SwingQuant/reports/portfolio_rotation_walk_forward_quarterly_returns.csv
- avg_positions: 5.39
- pct_days_fully_invested: 89.69%
- closed_trades: 87
- trade_win_rate: 74.71%
- mean_trade_return: 12.77%
- median_trade_return: 9.69%
- avg_holding_days: 20.0

## Closed Trade Returns by Entry Month
- 2025-01: trades=12, mean=+1.87%, median=-2.27%, win_rate=41.67%
- 2025-02: trades=6, mean=-6.45%, median=-7.42%, win_rate=33.33%
- 2025-05: trades=6, mean=+5.00%, median=+6.76%, win_rate=83.33%
- 2025-06: trades=6, mean=+13.09%, median=+6.64%, win_rate=83.33%
- 2025-07: trades=6, mean=+15.75%, median=-5.95%, win_rate=33.33%
- 2025-08: trades=6, mean=+8.47%, median=+6.47%, win_rate=83.33%
- 2025-09: trades=9, mean=+22.19%, median=+18.39%, win_rate=100.00%
- 2025-10: trades=9, mean=+14.18%, median=+11.55%, win_rate=77.78%
- 2025-11: trades=6, mean=+21.20%, median=+18.72%, win_rate=100.00%
- 2025-12: trades=6, mean=+7.75%, median=+9.31%, win_rate=83.33%
- 2026-01: trades=6, mean=+28.44%, median=+20.43%, win_rate=83.33%
- 2026-02: trades=6, mean=+19.22%, median=+20.15%, win_rate=100.00%
- 2026-03: trades=3, mean=+28.75%, median=+31.29%, win_rate=100.00%

## Best Closed Trades
- MP: 2025-07-07 -> 2025-08-04 return=+110.69%, model_alpha=+57.32%, pre_opp=0.44
- LITE: 2026-01-20 -> 2026-02-17 return=+68.27%, model_alpha=+30.51%, pre_opp=0.48
- CIEN: 2025-09-02 -> 2025-09-30 return=+55.65%, model_alpha=+16.15%, pre_opp=0.46
- ICHR: 2026-01-20 -> 2026-02-17 return=+53.97%, model_alpha=+35.38%, pre_opp=0.41
- LITE: 2025-10-29 -> 2025-11-26 return=+43.87%, model_alpha=+7.26%, pre_opp=0.44
- CAR: 2025-06-10 -> 2025-07-08 return=+41.83%, model_alpha=+10.89%, pre_opp=0.46
- CENX: 2025-11-25 -> 2025-12-23 return=+36.83%, model_alpha=+16.19%, pre_opp=0.47
- HL: 2025-09-03 -> 2025-10-01 return=+36.53%, model_alpha=+19.67%, pre_opp=0.41
- TWLO: 2025-01-02 -> 2025-01-30 return=+36.01%, model_alpha=+10.68%, pre_opp=0.45
- CENX: 2025-09-02 -> 2025-09-30 return=+35.36%, model_alpha=+11.13%, pre_opp=0.40

## Worst Closed Trades
- ALK: 2025-02-27 -> 2025-03-27 return=-26.13%, model_alpha=+2.60%, pre_opp=0.47
- BMI: 2025-07-09 -> 2025-08-06 return=-21.55%, model_alpha=+13.96%, pre_opp=0.52
- ALGM: 2025-07-09 -> 2025-08-06 return=-16.48%, model_alpha=+12.99%, pre_opp=0.48
- ATEN: 2025-02-27 -> 2025-03-27 return=-15.55%, model_alpha=+4.68%, pre_opp=0.49
- AAL: 2025-01-30 -> 2025-02-27 return=-14.02%, model_alpha=+6.94%, pre_opp=0.50
- KNTK: 2025-01-30 -> 2025-02-27 return=-13.79%, model_alpha=+6.61%, pre_opp=0.44
- GEO: 2025-01-30 -> 2025-02-27 return=-13.10%, model_alpha=+10.20%, pre_opp=0.49
- SLAB: 2025-02-27 -> 2025-03-27 return=-12.07%, model_alpha=+2.70%, pre_opp=0.43
- CLSK: 2025-07-09 -> 2025-08-06 return=-11.79%, model_alpha=+12.42%, pre_opp=0.46
- RUN: 2025-12-24 -> 2026-01-21 return=-11.61%, model_alpha=+7.53%, pre_opp=0.46

## Recent Trades
- LITE: 2026-01-20 -> 2026-02-17 return=+68.27% reason=max_hold_days
- WDC: 2026-01-20 -> 2026-02-17 return=+27.42% reason=max_hold_days
- TTMI: 2026-01-21 -> 2026-02-18 return=-5.91% reason=max_hold_days
- FORM: 2026-01-21 -> 2026-02-18 return=+13.44% reason=max_hold_days
- MKSI: 2026-01-21 -> 2026-02-18 return=+13.42% reason=max_hold_days
- TPL: 2026-02-17 -> 2026-03-17 return=+24.24% reason=max_hold_days
- CENX: 2026-02-17 -> 2026-03-17 return=+15.55% reason=max_hold_days
- AGX: 2026-02-17 -> 2026-03-17 return=+16.41% reason=max_hold_days
- SNDK: 2026-02-18 -> 2026-03-18 return=+25.53% reason=max_hold_days
- MU: 2026-02-18 -> 2026-03-18 return=+9.69% reason=max_hold_days
- CIEN: 2026-02-18 -> 2026-03-18 return=+23.89% reason=max_hold_days
- ICHR: 2026-03-17 -> 2026-04-14 return=+34.42% reason=max_hold_days
- CC: 2026-03-17 -> 2026-04-14 return=+20.54% reason=max_hold_days
- LITE: 2026-03-17 -> 2026-04-14 return=+31.29% reason=max_hold_days
- UCTT: 2026-03-18 -> 2026-04-14 return=+29.58% reason=open_at_end
- CENX: 2026-03-18 -> 2026-04-14 return=+15.23% reason=open_at_end
- FORM: 2026-03-18 -> 2026-04-14 return=+33.37% reason=open_at_end
- TALO: 2026-04-14 -> 2026-04-14 return=+0.00% reason=open_at_end
- PTEN: 2026-04-14 -> 2026-04-14 return=+0.00% reason=open_at_end
- SM: 2026-04-14 -> 2026-04-14 return=+0.00% reason=open_at_end

## Recent Portfolio States
- 2026-03-31: equity=4.9083, positions=6, holdings=ICHR, CC, LITE, UCTT, CENX, FORM
- 2026-04-01: equity=5.0973, positions=6, holdings=ICHR, CC, LITE, UCTT, CENX, FORM
- 2026-04-02: equity=5.1772, positions=6, holdings=ICHR, CC, LITE, UCTT, CENX, FORM
- 2026-04-06: equity=5.1536, positions=6, holdings=ICHR, CC, LITE, UCTT, CENX, FORM
- 2026-04-07: equity=5.2755, positions=6, holdings=ICHR, CC, LITE, UCTT, CENX, FORM
- 2026-04-08: equity=5.5606, positions=6, holdings=ICHR, CC, LITE, UCTT, CENX, FORM
- 2026-04-09: equity=5.7071, positions=6, holdings=ICHR, CC, LITE, UCTT, CENX, FORM
- 2026-04-10: equity=5.8539, positions=6, holdings=ICHR, CC, LITE, UCTT, CENX, FORM
- 2026-04-13: equity=5.9495, positions=6, holdings=ICHR, CC, LITE, UCTT, CENX, FORM
- 2026-04-14: equity=5.9192, positions=6, holdings=UCTT, CENX, FORM, TALO, PTEN, SM
