# Portfolio Rotation Backtest

## Rules
- target_positions: 6
- max_hold_days: 20
- min_pre_penalty_opportunity: 0.40
- min_model_alpha: 0.0000
- sell_rule: max_hold_days only in v1
- replacement_rule: fill empty slots from best unheld targets; no churn of existing holdings
- ranking_mode: retrospective_persisted_scan_scores

## Caveats
- This is not a walk-forward live simulation; it applies persisted scan model scores across historical scan snapshots.
- The v1 exit model only uses max holding days, so it does not yet reflect monitor sell signals, stops, RSI exits, earnings exits, or transaction costs.
- Total return is fully compounded through six equal-weight slots; high 20-day trade returns compound quickly.

## Summary
- initial_equity: 1.0000
- final_equity: 6.7314
- total_return: 573.14%
- spy_return: 31.75%
- max_drawdown: 19.17%
- eligible_scan_dates: 270
- portfolio_valuation_dates: 360
- completed_or_open_trades: 99
- equity_csv: /home/zdrillings/code/SwingQuant/reports/portfolio_rotation_equity.csv
- trades_csv: /home/zdrillings/code/SwingQuant/reports/portfolio_rotation_trades.csv
- avg_positions: 5.05
- pct_days_fully_invested: 81.39%
- closed_trades: 93
- trade_win_rate: 76.34%
- mean_trade_return: 13.25%
- median_trade_return: 11.55%
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
- 2026-03: trades=6, mean=+27.50%, median=+29.73%, win_rate=100.00%
- 2026-04: trades=3, mean=+14.36%, median=+13.38%, win_rate=100.00%

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
- CENX: 2026-02-17 -> 2026-03-17 return=+15.55% reason=max_hold_days
- AGX: 2026-02-17 -> 2026-03-17 return=+16.41% reason=max_hold_days
- SNDK: 2026-02-18 -> 2026-03-18 return=+25.53% reason=max_hold_days
- MU: 2026-02-18 -> 2026-03-18 return=+9.69% reason=max_hold_days
- CIEN: 2026-02-18 -> 2026-03-18 return=+23.89% reason=max_hold_days
- ICHR: 2026-03-17 -> 2026-04-14 return=+34.42% reason=max_hold_days
- CC: 2026-03-17 -> 2026-04-14 return=+20.54% reason=max_hold_days
- LITE: 2026-03-17 -> 2026-04-14 return=+31.29% reason=max_hold_days
- UCTT: 2026-03-18 -> 2026-04-15 return=+28.17% reason=max_hold_days
- CENX: 2026-03-18 -> 2026-04-15 return=+15.81% reason=max_hold_days
- FORM: 2026-03-18 -> 2026-04-15 return=+34.77% reason=max_hold_days
- TALO: 2026-04-14 -> 2026-05-12 return=+7.64% reason=max_hold_days
- PTEN: 2026-04-14 -> 2026-05-12 return=+22.06% reason=max_hold_days
- SM: 2026-04-14 -> 2026-05-12 return=+13.38% reason=max_hold_days
- MXL: 2026-06-08 -> 2026-06-15 return=+11.97% reason=open_at_end
- SMCI: 2026-06-08 -> 2026-06-15 return=+0.00% reason=open_at_end
- INTC: 2026-06-08 -> 2026-06-15 return=+15.95% reason=open_at_end
- CC: 2026-06-08 -> 2026-06-15 return=+7.53% reason=open_at_end
- SLAB: 2026-06-08 -> 2026-06-15 return=+0.98% reason=open_at_end
- TDC: 2026-06-08 -> 2026-06-15 return=-0.24% reason=open_at_end

## Recent Portfolio States
- 2026-05-28: equity=6.3485, positions=0, holdings=
- 2026-05-29: equity=6.3485, positions=0, holdings=
- 2026-06-01: equity=6.3485, positions=0, holdings=
- 2026-06-02: equity=6.3485, positions=0, holdings=
- 2026-06-08: equity=6.3485, positions=6, holdings=MXL, SMCI, INTC, CC, SLAB, TDC
- 2026-06-09: equity=6.1197, positions=6, holdings=MXL, SMCI, INTC, CC, SLAB, TDC
- 2026-06-10: equity=6.1985, positions=6, holdings=MXL, SMCI, INTC, CC, SLAB, TDC
- 2026-06-11: equity=6.4456, positions=6, holdings=MXL, SMCI, INTC, CC, SLAB, TDC
- 2026-06-12: equity=6.5972, positions=6, holdings=MXL, SMCI, INTC, CC, SLAB, TDC
- 2026-06-15: equity=6.7314, positions=6, holdings=MXL, SMCI, INTC, CC, SLAB, TDC
