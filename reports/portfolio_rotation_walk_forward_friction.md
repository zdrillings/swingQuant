# Portfolio Rotation Backtest

## Rules
- target_positions: 6
- max_hold_days: 20
- min_pre_penalty_opportunity: 0.40
- min_model_alpha: 0.0000
- transaction_cost_bps: 10.00
- slippage_bps: 5.00
- cooldown_days: 5
- sell_rule: max_hold_days only in v1
- replacement_rule: fill empty slots from best unheld targets; no churn of existing holdings
- ranking_mode: walk_forward_oos
- ranking_model: xgboost_model
- ranking_generated_at: 2026-06-02T20:08:10+00:00
- ranking_prediction_dates: 2025-01-02 to 2026-04-14

## Caveats
- Walk-forward mode uses persisted out-of-sample shortlist predictions, but still depends on the current stored scan snapshots and v1 portfolio mechanics.
- The v1 exit model only uses max holding days, so it does not yet reflect monitor sell signals, stops, RSI exits, or earnings exits.
- Total return is fully compounded through six equal-weight slots; high 20-day trade returns compound quickly.

## Summary
- initial_equity: 1.0000
- final_equity: 5.6856
- total_return: 468.56%
- spy_return: 20.51%
- max_drawdown: 19.65%
- eligible_scan_dates: 264
- portfolio_valuation_dates: 320
- completed_or_open_trades: 93
- equity_csv: /home/zdrillings/code/SwingQuant/reports/portfolio_rotation_walk_forward_friction_equity.csv
- trades_csv: /home/zdrillings/code/SwingQuant/reports/portfolio_rotation_walk_forward_friction_trades.csv
- avg_positions: 5.39
- pct_days_fully_invested: 89.69%
- closed_trades: 87
- trade_win_rate: 74.71%
- mean_trade_return: 12.40%
- median_trade_return: 9.36%
- avg_holding_days: 20.0

## Closed Trade Returns by Entry Month
- 2025-01: trades=12, mean=+1.57%, median=-2.57%, win_rate=41.67%
- 2025-02: trades=6, mean=-6.73%, median=-7.70%, win_rate=33.33%
- 2025-05: trades=6, mean=+4.69%, median=+6.44%, win_rate=83.33%
- 2025-06: trades=6, mean=+12.75%, median=+6.32%, win_rate=83.33%
- 2025-07: trades=6, mean=+15.40%, median=-6.23%, win_rate=33.33%
- 2025-08: trades=6, mean=+7.85%, median=+5.25%, win_rate=83.33%
- 2025-09: trades=9, mean=+21.60%, median=+15.94%, win_rate=100.00%
- 2025-10: trades=9, mean=+15.29%, median=+11.21%, win_rate=88.89%
- 2025-11: trades=6, mean=+18.52%, median=+18.37%, win_rate=83.33%
- 2025-12: trades=6, mean=+11.62%, median=+14.24%, win_rate=83.33%
- 2026-01: trades=6, mean=+24.15%, median=+13.09%, win_rate=83.33%
- 2026-02: trades=6, mean=+18.86%, median=+19.79%, win_rate=100.00%
- 2026-03: trades=3, mean=+28.36%, median=+30.89%, win_rate=100.00%

## Best Closed Trades
- MP: 2025-07-07 -> 2025-08-04 return=+110.06%, model_alpha=+57.32%, pre_opp=0.44
- LITE: 2026-01-20 -> 2026-02-17 return=+67.76%, model_alpha=+30.51%, pre_opp=0.48
- CIEN: 2025-09-02 -> 2025-09-30 return=+55.18%, model_alpha=+16.15%, pre_opp=0.46
- ICHR: 2026-01-20 -> 2026-02-17 return=+53.51%, model_alpha=+35.38%, pre_opp=0.41
- LITE: 2025-10-29 -> 2025-11-26 return=+43.44%, model_alpha=+7.26%, pre_opp=0.44
- CAR: 2025-06-10 -> 2025-07-08 return=+41.40%, model_alpha=+10.89%, pre_opp=0.46
- CENX: 2025-11-25 -> 2025-12-23 return=+36.42%, model_alpha=+16.19%, pre_opp=0.47
- HL: 2025-09-03 -> 2025-10-01 return=+36.12%, model_alpha=+19.67%, pre_opp=0.41
- TWLO: 2025-01-02 -> 2025-01-30 return=+35.61%, model_alpha=+10.68%, pre_opp=0.45
- CENX: 2025-09-02 -> 2025-09-30 return=+34.96%, model_alpha=+11.13%, pre_opp=0.40

## Worst Closed Trades
- ALK: 2025-02-27 -> 2025-03-27 return=-26.35%, model_alpha=+2.60%, pre_opp=0.47
- BMI: 2025-07-09 -> 2025-08-06 return=-21.78%, model_alpha=+13.96%, pre_opp=0.52
- ALGM: 2025-07-09 -> 2025-08-06 return=-16.73%, model_alpha=+12.99%, pre_opp=0.48
- ATEN: 2025-02-27 -> 2025-03-27 return=-15.80%, model_alpha=+4.68%, pre_opp=0.49
- AAL: 2025-01-30 -> 2025-02-27 return=-14.28%, model_alpha=+6.94%, pre_opp=0.50
- KNTK: 2025-01-30 -> 2025-02-27 return=-14.05%, model_alpha=+6.61%, pre_opp=0.44
- GEO: 2025-01-30 -> 2025-02-27 return=-13.36%, model_alpha=+10.20%, pre_opp=0.49
- SLAB: 2025-02-27 -> 2025-03-27 return=-12.33%, model_alpha=+2.70%, pre_opp=0.43
- CLSK: 2025-07-09 -> 2025-08-06 return=-12.05%, model_alpha=+12.42%, pre_opp=0.46
- RUN: 2025-12-24 -> 2026-01-21 return=-11.87%, model_alpha=+7.53%, pre_opp=0.46

## Recent Trades
- LITE: 2026-01-20 -> 2026-02-17 return=+67.76% reason=max_hold_days
- TTMI: 2026-01-20 -> 2026-02-17 return=-7.35% reason=max_hold_days
- FORM: 2026-01-21 -> 2026-02-18 return=+13.10% reason=max_hold_days
- MKSI: 2026-01-21 -> 2026-02-18 return=+13.08% reason=max_hold_days
- LRCX: 2026-01-21 -> 2026-02-18 return=+4.81% reason=max_hold_days
- TPL: 2026-02-17 -> 2026-03-17 return=+23.87% reason=max_hold_days
- CENX: 2026-02-17 -> 2026-03-17 return=+15.20% reason=max_hold_days
- AGX: 2026-02-17 -> 2026-03-17 return=+16.06% reason=max_hold_days
- SNDK: 2026-02-18 -> 2026-03-18 return=+25.16% reason=max_hold_days
- MU: 2026-02-18 -> 2026-03-18 return=+9.36% reason=max_hold_days
- CIEN: 2026-02-18 -> 2026-03-18 return=+23.52% reason=max_hold_days
- ICHR: 2026-03-17 -> 2026-04-14 return=+34.02% reason=max_hold_days
- CC: 2026-03-17 -> 2026-04-14 return=+20.18% reason=max_hold_days
- LITE: 2026-03-17 -> 2026-04-14 return=+30.89% reason=max_hold_days
- UCTT: 2026-03-18 -> 2026-04-14 return=+29.19% reason=open_at_end
- FORM: 2026-03-18 -> 2026-04-14 return=+32.98% reason=open_at_end
- TER: 2026-03-18 -> 2026-04-14 return=+21.44% reason=open_at_end
- TALO: 2026-04-14 -> 2026-04-14 return=-0.30% reason=open_at_end
- PTEN: 2026-04-14 -> 2026-04-14 return=-0.30% reason=open_at_end
- SM: 2026-04-14 -> 2026-04-14 return=-0.30% reason=open_at_end

## Recent Portfolio States
- 2026-03-31: equity=4.6275, positions=6, holdings=ICHR, CC, LITE, UCTT, FORM, TER
- 2026-04-01: equity=4.7817, positions=6, holdings=ICHR, CC, LITE, UCTT, FORM, TER
- 2026-04-02: equity=4.8641, positions=6, holdings=ICHR, CC, LITE, UCTT, FORM, TER
- 2026-04-06: equity=4.8295, positions=6, holdings=ICHR, CC, LITE, UCTT, FORM, TER
- 2026-04-07: equity=4.9382, positions=6, holdings=ICHR, CC, LITE, UCTT, FORM, TER
- 2026-04-08: equity=5.3100, positions=6, holdings=ICHR, CC, LITE, UCTT, FORM, TER
- 2026-04-09: equity=5.4544, positions=6, holdings=ICHR, CC, LITE, UCTT, FORM, TER
- 2026-04-10: equity=5.5950, positions=6, holdings=ICHR, CC, LITE, UCTT, FORM, TER
- 2026-04-13: equity=5.6782, positions=6, holdings=ICHR, CC, LITE, UCTT, FORM, TER
- 2026-04-14: equity=5.6856, positions=6, holdings=UCTT, FORM, TER, TALO, PTEN, SM
