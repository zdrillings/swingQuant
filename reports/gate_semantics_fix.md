# Gate Semantics Fix - 2026-10-07

- generated_at: 2026-10-08T03:21:01+00:00
- data_access: read-only CSV artifacts; no `./sq` write command; no `data/` mutation
- oos_artifact: reports/shortlist_model_oos_predictions.csv
- live_artifact: reports/shortlist_model_live_predictions.csv
- model: ridge_adaptive
- target_column: alpha_vs_sector_60d
- selection_gate: score_quantile
- config_selection_gate_enabled_after_revert: False
- dryrun_selection_gate_enabled: True
- selection_gate_quantile: 0.9500
- selection_gate_lookback_sessions: 126
- raw_oos_rows_loaded: 221167
- raw_oos_dates_loaded: 522
- oos_rows_loaded: 7914
- oos_dates_loaded: 516
- calendar_dates_loaded: 1362
- oos_date_min: 2024-04-12
- oos_date_max: 2026-07-14
- latest_live_date: 2026-10-07
- latest_live_rows: 647
- before_live_gate_threshold: +0.446391
- before_gate_qualified_live_count: 388
- before_gate_qualified_live_rate: 59.97%
- fixed_live_gate_threshold: +0.852788
- fixed_gate_qualified_live_count: 42
- fixed_gate_qualified_live_rate: 6.49%
- runtime_top_n_after_gate: 2
- would_be_champion: n/a
- numeric_verdict: FAIL: gate-enabled ridge_adaptive does not clear tonight's promotion floors

## Semantics Delta

| implementation | threshold | qualified_live_rows | qualified_live_rate | score_history_rows | score_history_dates |
|---|---:|---:|---:|---:|---:|
| raw overlapping OOS rows | +0.446391 | 388 | 59.97% | 221167 | 522 |
| fixed session-level OOS rows | +0.852788 | 42 | 6.49% | 7914 | 516 |

## Gated Acceptance Windows

| window | dates | active_dates | empty_gated_dates | avg_pick_count | hit_rate_excess | beat_universe_rate | mean_target_excess | spearman | top_ticker | top_ticker_date_rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| ridge_adaptive_full_oos | 118 | 118 | 398 | +1.6780 | +0.0011 | +0.4746 | +0.0253 | +0.0335 | BKH | +0.0169 |
| ridge_adaptive_last_fold | 0 | 0 | 20 | n/a | n/a | n/a | n/a | n/a | nan | n/a |
| ridge_adaptive_trailing_3folds | 27 | 27 | 33 | +1.6667 | -0.0184 | +0.4444 | -0.0200 | +0.0415 | CF | +0.0370 |

## Floor Verdict

| window | metric | value | floor | pass |
|---|---|---:|---:|---|
| last_fold | hit_rate_excess | n/a | >= +0.0200 | no |
| last_fold | beat_universe_rate | n/a | >= +0.5000 | no |
| last_fold | mean_target_excess | n/a | >= +0.0000 | no |
| last_fold | spearman | n/a | >= +0.0000 | no |
| last_fold | top_ticker_date_rate | n/a | <= +0.4000 | no |
| trailing_3folds | hit_rate_excess | -0.0184 | >= +0.0200 | no |
| trailing_3folds | beat_universe_rate | +0.4444 | >= +0.5000 | no |
| trailing_3folds | mean_target_excess | -0.0200 | >= +0.0000 | no |
| trailing_3folds | spearman | +0.0415 | >= +0.0000 | yes |
| trailing_3folds | top_ticker_date_rate | +0.0370 | <= +0.4000 | yes |
| full_oos | spearman | +0.0335 | >= +0.0000 | yes |

## Would-Be Live Picks

- note: 42 latest-live rows clear the fixed rolling score threshold; the runtime model path remains capped at top-2 picks.

| rank_after_gate | ticker | sector | score | threshold |
|---:|---|---|---:|---:|
| 1 | KRG | Real Estate | +0.957689 | +0.852788 |
| 2 | VTR | Real Estate | +0.949382 | +0.852788 |

## Gate-Qualified Live Roster

| rank_after_gate | ticker | sector | score | threshold |
|---:|---|---|---:|---:|
| 1 | KRG | Real Estate | +0.957689 | +0.852788 |
| 2 | VTR | Real Estate | +0.949382 | +0.852788 |
| 3 | GD | Industrials | +0.944745 | +0.852788 |
| 4 | DOC | Real Estate | +0.943972 | +0.852788 |
| 5 | FRT | Real Estate | +0.934699 | +0.852788 |
| 6 | CALY | Consumer Discretionary | +0.925811 | +0.852788 |
| 7 | KHC | Consumer Staples | +0.920788 | +0.852788 |
| 8 | NNN | Real Estate | +0.916924 | +0.852788 |
| 9 | CTRE | Real Estate | +0.910935 | +0.852788 |
| 10 | INVH | Real Estate | +0.910935 | +0.852788 |
| 11 | PNC | Financials | +0.909196 | +0.852788 |
| 12 | VLY | Financials | +0.907264 | +0.852788 |
| 13 | BBT | Financials | +0.904946 | +0.852788 |
| 14 | JBHT | Industrials | +0.904560 | +0.852788 |
| 15 | BNL | Real Estate | +0.903400 | +0.852788 |
| 16 | MHO | Consumer Discretionary | +0.900309 | +0.852788 |
| 17 | RTX | Industrials | +0.896832 | +0.852788 |
| 18 | BMRN | Health Care | +0.889490 | +0.852788 |
| 19 | USB | Financials | +0.887944 | +0.852788 |
| 20 | BAC | Financials | +0.886012 | +0.852788 |
| 21 | AHR | Real Estate | +0.884853 | +0.852788 |
| 22 | BRX | Real Estate | +0.884853 | +0.852788 |
| 23 | KIM | Real Estate | +0.884080 | +0.852788 |
| 24 | PLD | Real Estate | +0.879057 | +0.852788 |
| 25 | PSA | Real Estate | +0.878284 | +0.852788 |
| 26 | OHI | Real Estate | +0.877898 | +0.852788 |
| 27 | FAF | Financials | +0.874034 | +0.852788 |
| 28 | RF | Financials | +0.872488 | +0.852788 |
| 29 | TFC | Financials | +0.872488 | +0.852788 |
| 30 | GS | Financials | +0.871716 | +0.852788 |
| 31 | KEY | Financials | +0.868238 | +0.852788 |
| 32 | FR | Real Estate | +0.866306 | +0.852788 |
| 33 | HXL | Industrials | +0.865533 | +0.852788 |
| 34 | ALL | Financials | +0.865533 | +0.852788 |
| 35 | SPG | Real Estate | +0.864374 | +0.852788 |
| 36 | GE | Industrials | +0.862828 | +0.852788 |
| 37 | OC | Industrials | +0.861283 | +0.852788 |
| 38 | BXP | Real Estate | +0.857805 | +0.852788 |
| 39 | GNTX | Consumer Discretionary | +0.857419 | +0.852788 |
| 40 | EPR | Real Estate | +0.857032 | +0.852788 |
| 41 | TCBI | Financials | +0.853941 | +0.852788 |
| 42 | AMH | Real Estate | +0.853748 | +0.852788 |
