# Beat Calibration

- generated_at: 2026-10-08T14:43:13+00:00
- idea: BEAT-CAL chronological isotonic calibration of score to P(beat sector)
- data_access: read-only OOS artifact and DuckDB calendar; no `./sq` write command; no `data/` mutation
- source_artifact: reports/shortlist_model_oos_predictions.csv
- output_report: reports/beat_calibration.md
- target_column: alpha_vs_sector_60d
- models: ridge_adaptive, ridge_overnight_rank_blend_0.4_0.6
- top_n: 2
- fixed_label_calendar_horizon_days: 60
- raw_rows_loaded: 396987
- raw_dates_loaded: 522
- calendar_dates_loaded: 1362
- fixed_evaluation_keys: 7914
- calibration: chronological isotonic, each date trained only on earlier OOS dates; target is alpha_vs_sector_60d > 0
- acceptance: full-OOS Spearman >= 0; last-fold and trailing-3fold hit_ex >= 0.02, beat >= 0.50, mean_excess >= 0, Spearman >= 0, top_ticker_date_rate <= 0.40
- verdict: NO PASS: best calibrated/raw row is ridge_overnight_rank_blend_0.4_0.6 raw_score_top2; last beat +0.4500, trailing beat +0.5167, full-OOS Spearman +0.0142; failed floors: last_fold beat +0.4500 >= +0.5000, last_fold mean_excess -0.0021 >= +0.0000, last_fold spearman -0.0634 >= +0.0000
- ridge_raw_full_oos_spearman: +0.0182

## Floor Table

| variant | cut | pass | rows | dates | full_sp | delta_vs_ridge_raw | last_hit_ex | last_beat | last_mean_ex | last_sp | last_top | last_top_rate | trailing_hit_ex | trailing_beat | trailing_mean_ex | trailing_sp | trailing_top | trailing_top_rate | failed_floors |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---|---:|---|
| ridge_overnight_rank_blend_0.4_0.6 | raw_score_top2 | no | 6487 | 420 | +0.0142 | -0.0040 | +0.0285 | +0.4500 | -0.0021 | -0.0634 | PGNY | +0.0500 | +0.0209 | +0.5167 | +0.0264 | +0.0223 | TALO | +0.0333 | last_fold beat +0.4500 >= +0.5000, last_fold mean_excess -0.0021 >= +0.0000, last_fold spearman -0.0634 >= +0.0000 |
| ridge_adaptive | raw_score_top2 | no | 7914 | 516 | +0.0182 | +0.0000 | +0.1014 | +0.4500 | +0.0489 | +0.2545 | FCX | +0.0500 | +0.0409 | +0.4167 | +0.0071 | +0.1007 | CF | +0.0167 | last_fold beat +0.4500 >= +0.5000, trailing_3fold beat +0.4167 >= +0.5000 |
| ridge_adaptive | calibrated_p_top2 | no | 7914 | 516 | +0.0393 | +0.0211 | +0.0264 | +0.4000 | +0.0026 | +0.2578 | CURB | +0.0500 | +0.0909 | +0.5333 | +0.0279 | +0.1804 | AMKR | +0.0333 | last_fold beat +0.4000 >= +0.5000 |
| ridge_overnight_rank_blend_0.4_0.6 | calibrated_p_top2 | no | 6487 | 420 | +0.0048 | -0.0134 | +0.0035 | +0.4000 | -0.0141 | -0.0817 | PDFS | +0.0500 | +0.0209 | +0.5000 | +0.0192 | +0.0227 | AA | +0.0333 | last_fold hit_excess +0.0035 >= +0.0200, last_fold beat +0.4000 >= +0.5000, last_fold mean_excess -0.0141 >= +0.0000, last_fold spearman -0.0817 >= +0.0000 |

## Verdict

NO PASS: best calibrated/raw row is ridge_overnight_rank_blend_0.4_0.6 raw_score_top2; last beat +0.4500, trailing beat +0.5167, full-OOS Spearman +0.0142; failed floors: last_fold beat +0.4500 >= +0.5000, last_fold mean_excess -0.0021 >= +0.0000, last_fold spearman -0.0634 >= +0.0000

Report only. The production candidate roster, promotion gate, selection gate, top-2 cap, confidence basket, rotation exclusion, and scan behavior are unchanged.

Tomorrow's critic can verify this by checking the floor table, the chronological-calibration tests, and this report's verdict against the pinned 2026-10-07 production OOS artifact.
