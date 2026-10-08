# Beat Signal

- generated_at: 2026-10-08T15:13:23+00:00
- idea: BEAT-SIGNAL beat-classification on the post-tour feature set
- data_access: read-only DuckDB and OOS artifact; no `./sq` write command; no `data/` mutation
- source_artifact: reports/shortlist_model_oos_predictions.csv
- target_column: alpha_vs_sector_60d
- beat_label: forward_beat_sector_60d = alpha_vs_sector_60d > 0
- feature_set: post-tour MODEL_FEATURE_COLUMNS with B1 overnight features and A4 regime interactions
- model_scope: global
- min_train_dates: 252
- max_train_dates: 126 (ridge_adaptive exact adaptive window)
- test_window_dates: 20
- evaluation_stride_dates: 60
- label_horizon_dates: 60
- eligible_universe_mode: passed_or_trend
- eligible_rows: 346098
- eligible_dates: 834
- raw_oos_rows_loaded: 396987
- raw_oos_dates_loaded: 522
- trained_beat_rows: 13300
- calendar_dates_loaded: 1362
- fixed_evaluation_keys: 7914
- calibration: chronological isotonic per variant; each date uses only earlier OOS dates
- acceptance: full-OOS Spearman >= 0; last-fold and trailing-3fold hit_ex >= 0.02, beat >= 0.50, mean_excess >= 0, Spearman >= 0, top_ticker_date_rate <= 0.40
- verdict: PASS: beat_logistic_ridge_adaptive clears all floors with full-OOS Spearman +0.0070, last beat +0.6000, trailing beat +0.5667
- ridge_baseline_full_oos_spearman: +0.0182
- best_variant: beat_logistic_ridge_adaptive
- best_common_grid_delta_vs_ridge_full_oos_spearman: -0.0410

## Floor Table

| variant | pass | rows | dates | full_sp | common_delta_vs_ridge | last_hit_ex | last_beat | last_mean_ex | last_sp | last_top | last_top_rate | trailing_hit_ex | trailing_beat | trailing_mean_ex | trailing_sp | trailing_top | trailing_top_rate | failed_floors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---|---:|---|
| beat_logistic_ridge_adaptive | yes | 3423 | 180 | +0.0070 | -0.0410 | +0.0795 | +0.6000 | +0.0239 | +0.1138 | CUZ | +0.0500 | +0.0491 | +0.5667 | +0.0274 | +0.0505 | WFRD | +0.0500 | none |
| beat_xgboost_post_tour_calibrated_ridge_rank_hybrid_0.3_0.7 | yes | 3423 | 180 | +0.0426 | -0.0055 | +0.1295 | +0.5500 | +0.0325 | +0.2702 | GRMN | +0.0500 | +0.0241 | +0.5333 | +0.0256 | +0.0421 | SPHR | +0.0500 | none |
| beat_xgboost_post_tour_calibrated_ridge_rank_hybrid_0.7_0.3 | no | 3423 | 180 | +0.0446 | -0.0035 | +0.0795 | +0.5000 | +0.0073 | +0.2330 | FLEX | +0.0500 | +0.0158 | +0.5167 | +0.0172 | +0.0409 | SPHR | +0.0500 | trailing_3fold hit_excess +0.0158 >= +0.0200 |
| beat_xgboost_post_tour_calibrated_ridge_rank_hybrid_0.5_0.5 | no | 3423 | 180 | +0.0491 | +0.0010 | +0.1045 | +0.5500 | +0.0245 | +0.2545 | GRMN | +0.0500 | +0.0158 | +0.5333 | +0.0225 | +0.0433 | SPHR | +0.0500 | trailing_3fold hit_excess +0.0158 >= +0.0200 |
| ridge_beat_xgboost_post_tour_rank_blend_0.4_0.6 | no | 3423 | 180 | +0.0487 | +0.0006 | +0.1045 | +0.5500 | +0.0245 | +0.2447 | FLEX | +0.0500 | +0.0158 | +0.5333 | +0.0225 | +0.0439 | SPHR | +0.0500 | trailing_3fold hit_excess +0.0158 >= +0.0200 |
| beat_logistic_ridge_adaptive_calibrated_ridge_rank_hybrid_0.3_0.7 | no | 3423 | 180 | +0.0488 | +0.0007 | +0.1295 | +0.5500 | +0.0325 | +0.2752 | GRMN | +0.0500 | +0.0158 | +0.5000 | +0.0200 | +0.0298 | SPHR | +0.0500 | trailing_3fold hit_excess +0.0158 >= +0.0200 |
| beat_logistic_ridge_adaptive_calibrated_ridge_rank_hybrid_0.5_0.5 | no | 3423 | 180 | +0.0550 | +0.0069 | +0.1045 | +0.5000 | +0.0250 | +0.2754 | GRMN | +0.0500 | +0.0075 | +0.4833 | +0.0175 | +0.0333 | WFRD | +0.0500 | trailing_3fold hit_excess +0.0075 >= +0.0200, trailing_3fold beat +0.4833 >= +0.5000 |
| beat_logistic_ridge_adaptive_calibrated_ridge_rank_hybrid_0.7_0.3 | no | 3423 | 180 | +0.0557 | +0.0076 | +0.0795 | +0.5000 | +0.0203 | +0.2737 | GRMN | +0.0500 | -0.0009 | +0.4833 | +0.0160 | +0.0354 | WFRD | +0.0500 | trailing_3fold hit_excess -0.0009 >= +0.0200, trailing_3fold beat +0.4833 >= +0.5000 |
| ridge_beat_logistic_ridge_adaptive_rank_blend_0.4_0.6 | no | 3423 | 180 | +0.0545 | +0.0065 | +0.0795 | +0.5000 | +0.0203 | +0.2741 | GRMN | +0.0500 | -0.0009 | +0.4833 | +0.0160 | +0.0352 | WFRD | +0.0500 | trailing_3fold hit_excess -0.0009 >= +0.0200, trailing_3fold beat +0.4833 >= +0.5000 |
| ridge_overnight_rank_blend_0.4_0.6 | no | 6487 | 420 | +0.0142 | +0.0197 | +0.0285 | +0.4500 | -0.0021 | -0.0634 | PGNY | +0.0500 | +0.0209 | +0.5167 | +0.0264 | +0.0223 | TALO | +0.0333 | last_fold beat +0.4500 >= +0.5000, last_fold mean_excess -0.0021 >= +0.0000, last_fold spearman -0.0634 >= +0.0000 |
| ridge_adaptive | no | 7914 | 516 | +0.0182 | +0.0000 | +0.1014 | +0.4500 | +0.0489 | +0.2545 | FCX | +0.0500 | +0.0409 | +0.4167 | +0.0071 | +0.1007 | CF | +0.0167 | last_fold beat +0.4500 >= +0.5000, trailing_3fold beat +0.4167 >= +0.5000 |
| beat_xgboost_post_tour | no | 3423 | 180 | -0.0026 | -0.0506 | -0.0205 | +0.4000 | -0.0071 | +0.1129 | FLEX | +0.0500 | +0.1158 | +0.6500 | +0.0692 | +0.1026 | FLEX | +0.0500 | full_oos spearman -0.0026 >= +0.0000, last_fold hit_excess -0.0205 >= +0.0200, last_fold beat +0.4000 >= +0.5000, last_fold mean_excess -0.0071 >= +0.0000 |
| overnight_session_specialist | no | 6487 | 420 | +0.0303 | +0.0358 | +0.0035 | +0.4500 | -0.0051 | -0.1469 | PDFS | +0.0500 | -0.0041 | +0.4500 | +0.0197 | -0.0568 | AA | +0.0333 | last_fold hit_excess +0.0035 >= +0.0200, last_fold beat +0.4500 >= +0.5000, last_fold mean_excess -0.0051 >= +0.0000, last_fold spearman -0.1469 >= +0.0000, trailing_3fold hit_excess -0.0041 >= +0.0200, trailing_3fold beat +0.4500 >= +0.5000, trailing_3fold spearman -0.0568 >= +0.0000 |
| beat_logistic_ridge_adaptive_calibrated | no | 3423 | 180 | +0.0348 | -0.0132 | -0.0455 | +0.4000 | +0.0027 | +0.1090 | AAON | +0.0500 | -0.0509 | +0.3333 | -0.0270 | +0.0682 | ACMR | +0.0500 | last_fold hit_excess -0.0455 >= +0.0200, last_fold beat +0.4000 >= +0.5000, trailing_3fold hit_excess -0.0509 >= +0.0200, trailing_3fold beat +0.3333 >= +0.5000, trailing_3fold mean_excess -0.0270 >= +0.0000 |
| beat_xgboost_post_tour_calibrated | no | 3423 | 180 | +0.0152 | -0.0328 | -0.0455 | +0.3500 | -0.0052 | -0.1421 | FLEX | +0.0500 | -0.0425 | +0.3167 | -0.0297 | +0.0520 | ACGL | +0.0500 | last_fold hit_excess -0.0455 >= +0.0200, last_fold beat +0.3500 >= +0.5000, last_fold mean_excess -0.0052 >= +0.0000, last_fold spearman -0.1421 >= +0.0000, trailing_3fold hit_excess -0.0425 >= +0.0200, trailing_3fold beat +0.3167 >= +0.5000, trailing_3fold mean_excess -0.0297 >= +0.0000 |
| beat_xgboost_post_tour_overnight_rank_blend_0.4_0.6 | no | 3423 | 180 | +0.0088 | -0.0393 | +0.0295 | +0.3000 | -0.0368 | -0.1464 | FLEX | +0.0500 | +0.0408 | +0.4500 | +0.0118 | +0.0237 | RUN | +0.0500 | last_fold beat +0.3000 >= +0.5000, last_fold mean_excess -0.0368 >= +0.0000, last_fold spearman -0.1464 >= +0.0000, trailing_3fold beat +0.4500 >= +0.5000 |
| beat_logistic_ridge_adaptive_overnight_rank_blend_0.4_0.6 | no | 3423 | 180 | +0.0188 | -0.0292 | +0.0045 | +0.2500 | -0.0460 | -0.1359 | AAON | +0.0500 | +0.0158 | +0.4333 | +0.0022 | +0.0322 | RUN | +0.0500 | last_fold hit_excess +0.0045 >= +0.0200, last_fold beat +0.2500 >= +0.5000, last_fold mean_excess -0.0460 >= +0.0000, last_fold spearman -0.1359 >= +0.0000, trailing_3fold hit_excess +0.0158 >= +0.0200, trailing_3fold beat +0.4333 >= +0.5000 |

## Verdict

PASS: beat_logistic_ridge_adaptive clears all floors with full-OOS Spearman +0.0070, last beat +0.6000, trailing beat +0.5667

Report only. The production candidate roster, promotion gate, selection gate, top-2 cap, confidence basket, rotation exclusion, and scan behavior are unchanged.
