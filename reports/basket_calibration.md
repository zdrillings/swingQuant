# Basket Calibration

- generated_at: 2026-10-07T02:10:08+00:00
- idea: BASKET-CAL from the 2026-10-06 critique
- data_access: read-only OOS artifact; no `./sq` write command; no `data/` mutation
- source_artifact: reports/shortlist_model_oos_predictions.csv
- model: ridge_adaptive
- target_column: alpha_vs_sector_60d
- raw_rows_loaded: 175428
- pinned_rows_audited: 7367
- pinned_dates_audited: 408
- calendar_dates_loaded: 1361
- variants_tested: 12
- trailing_3fold_floor_hit_rate_excess: +0.0200
- trailing_3fold_floor_beat_universe_rate: +0.5000
- verdict: PASS: score_gate_q950 clears trailing floors with hit_ex=+0.1363, beat=+0.6250, full Spearman=+0.0128, trailing picks=14

## Variant Table

| variant | score | cut | top_n | full_oos_spearman | trailing_hit_excess | trailing_beat_rate | trailing_mean_excess | trailing_picks | trailing_avg_picks | pick_cost_vs_as_is | last_hit_excess | last_beat_rate | last_mean_excess | last_picks | clears |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| calibrated_p_gate_0.600 | calibrated_p_beat_sector_oos | p>=0.600 | 2 | +0.0015 | +0.5204 | +1.0000 | +0.1765 | 4 | +0.0667 | -115 | +0.5204 | +1.0000 | +0.1765 | 4 | yes |
| calibrated_p_gate_0.500 | calibrated_p_beat_sector_oos | p>=0.500 | 2 | +0.0015 | +0.4958 | +0.7500 | +0.1205 | 8 | +0.1333 | -111 | +0.4958 | +0.7500 | +0.1205 | 8 | yes |
| calibrated_p_gate_0.525 | calibrated_p_beat_sector_oos | p>=0.525 | 2 | +0.0015 | +0.4958 | +0.7500 | +0.1205 | 8 | +0.1333 | -111 | +0.4958 | +0.7500 | +0.1205 | 8 | yes |
| calibrated_p_gate_0.550 | calibrated_p_beat_sector_oos | p>=0.550 | 2 | +0.0015 | +0.4958 | +0.7500 | +0.1205 | 8 | +0.1333 | -111 | +0.4958 | +0.7500 | +0.1205 | 8 | yes |
| calibrated_p_gate_0.575 | calibrated_p_beat_sector_oos | p>=0.575 | 2 | +0.0015 | +0.4958 | +0.7500 | +0.1205 | 8 | +0.1333 | -111 | +0.4958 | +0.7500 | +0.1205 | 8 | yes |
| score_gate_q950 | predicted_alpha | q0.950=0.598821 | 2 | +0.0128 | +0.1363 | +0.6250 | +0.0744 | 14 | +0.2333 | -105 | +0.2842 | +0.8333 | +0.1566 | 11 | yes |
| score_gate_q975 | predicted_alpha | q0.975=0.885108 | 2 | +0.0128 | +0.0742 | +0.5714 | +0.0499 | 11 | +0.1833 | -108 | +0.2268 | +0.8000 | +0.1706 | 9 | yes |
| top1_by_score | predicted_alpha | none | 1 | +0.0128 | +0.0229 | +0.4833 | +0.0349 | 60 | +1.0000 | -59 | +0.1738 | +0.6500 | +0.0716 | 20 | no |
| score_gate_q925 | predicted_alpha | q0.925=0.391803 | 2 | +0.0128 | +0.0148 | +0.5000 | +0.0092 | 35 | +0.5833 | -84 | +0.1731 | +0.7143 | +0.0532 | 22 | no |
| as_is_top2_by_score | predicted_alpha | none | 2 | +0.0128 | +0.0145 | +0.4833 | +0.0162 | 119 | +1.9833 | +0 | +0.1238 | +0.6000 | +0.0504 | 40 | no |
| score_gate_q900 | predicted_alpha | q0.900=0.298271 | 2 | +0.0128 | -0.0655 | +0.4103 | -0.0085 | 60 | +1.0000 | -59 | +0.1193 | +0.5556 | +0.0329 | 30 | no |
| score_gate_q990 | predicted_alpha | q0.990=5.955269 | 2 | +0.0128 | n/a | n/a | n/a | 0 | +0.0000 | -119 | n/a | n/a | n/a | 0 | no |

## Notes

- Score-gate thresholds are fixed quantiles of ridge_adaptive OOS scores; names such as `q0.950` identify the scanned cut.
- Calibrated-P rows use chronological isotonic calibration: each OOS date is scored from earlier OOS dates only.
- Empty gated slots stay empty and are reflected in total picks and average picks; floor rates are measured on dates where the cut selected at least one name.
- This is report-only. Production selection, promotion gates, top-2 basket size, confidence basket, rotation exclusion, and strategy files are unchanged.
