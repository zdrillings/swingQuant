# P4 Orthogonal Ensemble Bakeoff

- generated_at: 2026-10-08T03:11:54+00:00
- idea: P4 orthogonal ensemble over ridge_adaptive and overnight_session_specialist
- data_access: read-only OOS artifact; no `./sq` write command; no `data/` mutation
- source_artifact: reports/shortlist_model_oos_predictions.csv
- target_column: alpha_vs_sector_60d
- top_n: 2
- fixed_label_calendar_horizon_days: 60
- fold_size_dates: 20
- trailing_folds: 3
- raw_rows_loaded: 396987
- raw_dates_loaded: 522
- calendar_dates_loaded: 1362
- fixed_evaluation_keys: 7914
- acceptance: full_oos_spearman >= 0, 1fold/3fold hit_ex >= 0.02, beat >= 0.50, mean_excess >= 0, spearman >= 0
- verdict: NO PASS: best 3fold beat is ridge_overnight_rank_blend_0.4_0.6 at +0.5167 with hit_ex=+0.0209
- best_variant: ridge_overnight_rank_blend_0.4_0.6
- best_full_oos_spearman: +0.0142
- ridge_baseline_full_oos_spearman: +0.0182
- best_delta_vs_ridge_full_oos_spearman: -0.0040

## Acceptance Windows

| variant | pass | rows | dates | full_oos_spearman | last_fold_spearman | last_fold_hit_ex | last_fold_beat | last_fold_mean_ex | trailing_3fold_spearman | trailing_3fold_hit_ex | trailing_3fold_beat | trailing_3fold_mean_ex |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ridge_overnight_rank_blend_0.4_0.6 | no | 6487 | 420 | +0.0142 | -0.0634 | +0.0285 | +0.4500 | -0.0021 | +0.0223 | +0.0209 | +0.5167 | +0.0264 |
| ridge_overnight_rank_blend_0.3_0.7 | no | 6487 | 420 | +0.0236 | -0.0986 | +0.0535 | +0.4500 | +0.0001 | -0.0037 | +0.0209 | +0.5000 | +0.0214 |
| ridge_overnight_rank_blend_0.1_0.9 | no | 6487 | 420 | +0.0301 | -0.1347 | +0.0535 | +0.5000 | -0.0077 | -0.0511 | -0.0041 | +0.4833 | +0.0196 |
| ridge_overnight_rank_blend_0.6_0.4 | no | 6487 | 420 | +0.0034 | +0.0271 | +0.0035 | +0.3500 | -0.0141 | +0.0643 | +0.0126 | +0.4500 | +0.0064 |
| ridge_overnight_rank_blend_0.7_0.3 | no | 6487 | 420 | -0.0072 | +0.0232 | +0.0035 | +0.3500 | -0.0110 | +0.0678 | +0.0043 | +0.4500 | +0.0060 |
| overnight_session_specialist | no | 6487 | 420 | +0.0303 | -0.1469 | +0.0035 | +0.4500 | -0.0051 | -0.0568 | -0.0041 | +0.4500 | +0.0197 |
| ridge_overnight_rank_blend_0.2_0.8 | no | 6487 | 420 | +0.0250 | -0.1356 | +0.0535 | +0.4500 | +0.0009 | -0.0373 | -0.0041 | +0.4500 | +0.0158 |
| ridge_overnight_rank_blend_0.9_0.1 | no | 6487 | 420 | -0.0035 | +0.1322 | +0.0785 | +0.4000 | +0.0189 | +0.1015 | +0.0126 | +0.4333 | +0.0045 |
| ridge_overnight_rank_blend_0.5_0.5 | no | 6487 | 420 | +0.0074 | -0.0342 | -0.0215 | +0.3000 | -0.0316 | +0.0354 | +0.0126 | +0.4333 | +0.0136 |
| ridge_adaptive | no | 7914 | 516 | +0.0182 | +0.2545 | +0.1014 | +0.4500 | +0.0489 | +0.1007 | +0.0409 | +0.4167 | +0.0071 |
| ridge_overnight_rank_blend_0.8_0.2 | no | 6487 | 420 | -0.0077 | +0.1099 | +0.0035 | +0.4000 | +0.0008 | +0.1109 | -0.0124 | +0.4167 | -0.0023 |

## Implementation Decision

This is an artifact audit only. The production candidate roster, promotion gate, selection gate, top-2 cap, and scan behavior are unchanged.

Tomorrow's critic can verify the result by checking this report's `verdict`, `best_delta_vs_ridge_full_oos_spearman`, and the acceptance-window table against the 2026-10-07 production OOS artifact.
