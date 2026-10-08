# Basket Size Sweep

- generated_at: 2026-10-08T14:56:20+00:00
- idea: BASKET-SIZE audit of gate floor statistics from top-2 through top-6
- data_access: read-only OOS artifact and DuckDB calendar; no `./sq` write command; no `data/` mutation
- source_artifact: reports/shortlist_model_oos_predictions.csv
- output_report: reports/basket_size_sweep.md
- target_column: alpha_vs_sector_60d
- variants: ridge_adaptive raw score; ridge_adaptive chronological calibrated-P; ridge_overnight_rank_blend_0.4_0.6 raw score
- basket_sizes: 2, 3, 4, 5, 6
- fixed_label_calendar_horizon_days: 60
- raw_rows_loaded: 396987
- raw_dates_loaded: 522
- calendar_dates_loaded: 1362
- fixed_evaluation_keys: 7914
- beat_clearance_test: last_fold_beat_universe_rate >= 0.50 AND trailing_3fold_beat_universe_rate >= 0.50
- verdict: NO BEAT CLEAR: no tested basket size clears beat >= 0.50 on both last-fold and trailing-3fold windows; closest row is ridge_overnight_rank_blend_0.4_0.6 raw_score top-2 (last beat +0.4500, trailing beat +0.5167). Full-OOS Spearman by size: top-2 ridge_adaptive/calibrated_p +0.0393; top-3 ridge_adaptive/calibrated_p +0.0393; top-4 ridge_adaptive/calibrated_p +0.0393; top-5 ridge_adaptive/calibrated_p +0.0393; top-6 ridge_adaptive/calibrated_p +0.0393.

## Basket Matrix

| variant | score_mode | top_n | rows | dates | full_sp | last_hit_ex | last_beat | last_mean_ex | last_sp | last_top_rate | trailing_hit_ex | trailing_beat | trailing_mean_ex | trailing_sp | trailing_top_rate | beat_clear | window_floor_clear |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ridge_overnight_rank_blend_0.4_0.6 | raw_score | 2 | 6487 | 420 | +0.0142 | +0.0285 | +0.4500 | -0.0021 | -0.0634 | +0.0500 | +0.0209 | +0.5167 | +0.0264 | +0.0223 | +0.0333 | no | no |
| ridge_adaptive | raw_score | 2 | 7914 | 516 | +0.0182 | +0.1014 | +0.4500 | +0.0489 | +0.2545 | +0.0500 | +0.0409 | +0.4167 | +0.0071 | +0.1007 | +0.0167 | no | no |
| ridge_adaptive | calibrated_p | 2 | 7914 | 516 | +0.0393 | +0.0264 | +0.4000 | +0.0026 | +0.2578 | +0.0500 | +0.0909 | +0.5333 | +0.0279 | +0.1804 | +0.0333 | no | no |
| ridge_adaptive | raw_score | 3 | 7914 | 516 | +0.0182 | +0.0681 | +0.5000 | +0.0392 | +0.2545 | +0.0500 | +0.0409 | +0.4667 | +0.0077 | +0.1007 | +0.0333 | no | no |
| ridge_overnight_rank_blend_0.4_0.6 | raw_score | 3 | 6487 | 420 | +0.0142 | -0.0048 | +0.4500 | -0.0007 | -0.0634 | +0.0500 | +0.0154 | +0.5167 | +0.0186 | +0.0223 | +0.0333 | no | no |
| ridge_adaptive | calibrated_p | 3 | 7914 | 516 | +0.0393 | +0.0348 | +0.4000 | +0.0008 | +0.2578 | +0.0500 | +0.0687 | +0.5000 | +0.0161 | +0.1804 | +0.0333 | no | no |
| ridge_adaptive | raw_score | 4 | 7914 | 516 | +0.0182 | +0.0514 | +0.5000 | +0.0211 | +0.2545 | +0.0500 | +0.0353 | +0.4167 | -0.0017 | +0.1007 | +0.0333 | no | no |
| ridge_overnight_rank_blend_0.4_0.6 | raw_score | 4 | 6487 | 420 | +0.0142 | +0.0368 | +0.4000 | +0.0083 | -0.0634 | +0.0500 | +0.0168 | +0.4833 | +0.0102 | +0.0223 | +0.0333 | no | no |
| ridge_adaptive | calibrated_p | 4 | 7914 | 516 | +0.0393 | +0.0264 | +0.3500 | +0.0065 | +0.2578 | +0.0500 | +0.0562 | +0.4833 | +0.0176 | +0.1804 | +0.0333 | no | no |
| ridge_overnight_rank_blend_0.4_0.6 | raw_score | 5 | 6487 | 420 | +0.0142 | +0.0418 | +0.4000 | +0.0014 | -0.0634 | +0.0500 | +0.0143 | +0.5333 | +0.0045 | +0.0223 | +0.0333 | no | no |
| ridge_adaptive | raw_score | 5 | 7914 | 516 | +0.0182 | +0.0164 | +0.4000 | +0.0049 | +0.2545 | +0.0500 | +0.0203 | +0.5000 | +0.0001 | +0.1007 | +0.0333 | no | no |
| ridge_adaptive | calibrated_p | 5 | 7914 | 516 | +0.0393 | +0.0364 | +0.3500 | +0.0073 | +0.2578 | +0.0500 | +0.0503 | +0.4667 | +0.0163 | +0.1804 | +0.0333 | no | no |
| ridge_overnight_rank_blend_0.4_0.6 | raw_score | 6 | 6487 | 420 | +0.0142 | +0.0285 | +0.4000 | -0.0008 | -0.0634 | +0.0500 | +0.0170 | +0.5000 | +0.0044 | +0.0223 | +0.0333 | no | no |
| ridge_adaptive | raw_score | 6 | 7914 | 516 | +0.0182 | +0.0148 | +0.4000 | +0.0058 | +0.2545 | +0.0500 | +0.0159 | +0.4667 | +0.0008 | +0.1007 | +0.0333 | no | no |
| ridge_adaptive | calibrated_p | 6 | 7914 | 516 | +0.0393 | +0.0314 | +0.3500 | +0.0118 | +0.2578 | +0.0500 | +0.0520 | +0.5167 | +0.0184 | +0.1804 | +0.0333 | no | no |

## Verdict

NO BEAT CLEAR: no tested basket size clears beat >= 0.50 on both last-fold and trailing-3fold windows; closest row is ridge_overnight_rank_blend_0.4_0.6 raw_score top-2 (last beat +0.4500, trailing beat +0.5167). Full-OOS Spearman by size: top-2 ridge_adaptive/calibrated_p +0.0393; top-3 ridge_adaptive/calibrated_p +0.0393; top-4 ridge_adaptive/calibrated_p +0.0393; top-5 ridge_adaptive/calibrated_p +0.0393; top-6 ridge_adaptive/calibrated_p +0.0393.

Report only. The production candidate roster, promotion gate, selection gate, top-2 cap, confidence basket, rotation exclusion, and scan behavior are unchanged.

Tomorrow's critic can verify this by checking that every requested top-N row appears in the matrix and that the verdict names the first basket size clearing both beat windows, or says none.
