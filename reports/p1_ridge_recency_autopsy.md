# P1 Ridge Recency Autopsy

- generated_at: 2026-10-07T01:32:59+00:00
- idea: P1 autopsy ridge_adaptive recency and close the 3fold hit/beat gap
- data_access: read-only OOS artifact; no `./sq` write command; no `data/` mutation
- source_artifact: reports/shortlist_model_oos_predictions.csv
- target_column: alpha_vs_sector_60d
- raw_rows_loaded: 350856
- raw_dates_loaded: 421
- calendar_dates_loaded: 1361
- ridge_b1_family: overnight_ret_5d, rth_ret_5d, overnight_minus_rth_5d, overnight_ret_20d, rth_ret_20d, overnight_minus_rth_20d, max_gap_down_pct_60
- tested_blend: 0.7 ridge_adaptive rank + 0.3 overnight_session_specialist rank
- acceptance_floor_3fold_hit_rate_excess: +0.0200
- acceptance_floor_3fold_beat_universe_rate: +0.5000
- verdict: no artifact passer: best ridge_adaptive hit_ex=+0.0145, beat=+0.4833

## Variant Metrics

| variant | rows | dates | full_oos_spearman | trailing_3fold_spearman | trailing_3fold_hit_rate_excess | trailing_3fold_beat_universe_rate | last_fold_hit_rate_excess | last_fold_beat_universe_rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overnight_session_specialist | 7367 | 408 | +0.0181 | -0.0297 | +0.0062 | +0.5000 | -0.0262 | +0.4000 |
| ridge_adaptive | 7367 | 408 | +0.0128 | +0.1136 | +0.0145 | +0.4833 | +0.1238 | +0.6000 |
| ridge_overnight_rank_blend_0.7_0.3 | 7367 | 408 | -0.0015 | +0.0962 | -0.0021 | +0.5000 | -0.0262 | +0.3500 |

## Ridge Trailing-3fold Reason Ablation

| bucket | picks | mean_target | hit_rate | beat_pick_rate |
|---|---:|---:|---:|---:|
| all_ridge_adaptive_top2 | 119 | +0.0162 | +0.4706 | +0.5042 |
| b1_reason_pick | 0 | n/a | n/a | n/a |
| non_b1_reason_pick | 119 | +0.0162 | +0.4706 | +0.5042 |

## Notes

- Production code now gives ridge_adaptive the explicit B1 overnight family and applies B1-specific observation counts during the fold-local IC screen.
- The blend row is an audit variant only; it is not added to the production candidate roster.
- Tomorrow's pipeline should verify whether ridge_adaptive no longer loses B1 columns to the global observation floor and whether the live gate windows improve.
