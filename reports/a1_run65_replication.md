# A1 Run 65 Replication

- generated_at: 2026-10-05
- data_access: read-only DuckDB plus existing OOS CSV; no `./sq` writes; no `data/` mutation
- baseline_reference: run 65 / 00144e4, 20d top-10 sector-specific xgboost, pooled Spearman +0.4644 and per-date Spearman +0.4713 on 20 OOS dates
- experiment: 20d endpoint label, top-10 readout, sector-specific xgboost balanced_depth4, FULL feature pool, fold-local IC screen OFF, regime matching off
- control: same sector-specific 20d setup with the current fold-local IC screen ON
- feature_note: the exact 00144e4 fold-local survivor list was not retained in git artifacts; this isolates sector-specific 20d modeling without pretending to recover unavailable survivor state
- eligible_rows: 311864
- eligible_dates: 879
- sector_specific_ic_on_oos_dates: 607
- sector_specific_ic_off_oos_dates: 607
- shared_current_grid_dates: 507
- run65_grid_dates: 20

## Spearman Comparison

| grid | model | path | target | dates | rows | pooled_spearman | per_date_spearman | top_n | top_mean |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| run65-dates | xgboost_model | forensic-run65-stored | alpha_vs_sector_20d | 20 | 7412 | +0.4644 | +0.4713 | 10 | +0.1947 |
| run65-dates | elastic_net_model | current-roster-20d | alpha_vs_sector_20d | 17 | 7023 | +0.0603 | +0.0239 | 10 | +0.0228 |
| run65-dates | ensemble_model | current-roster-20d | alpha_vs_sector_20d | 17 | 7023 | +0.0278 | +0.0318 | 10 | +0.0056 |
| run65-dates | event_ic_model | current-roster-20d | alpha_vs_sector_20d | 17 | 7023 | -0.0250 | -0.0084 | 10 | -0.0015 |
| run65-dates | event_signal | current-roster-20d | alpha_vs_sector_20d | 17 | 7023 | -0.0112 | -0.0136 | 10 | -0.0031 |
| run65-dates | ic_sign_model | current-roster-20d | alpha_vs_sector_20d | 17 | 7023 | +0.0086 | +0.0182 | 10 | -0.0022 |
| run65-dates | lasso_model | current-roster-20d | alpha_vs_sector_20d | 17 | 7023 | +0.0739 | +0.0298 | 10 | +0.0212 |
| run65-dates | reversal_rules | current-roster-20d | alpha_vs_sector_20d | 17 | 7023 | +0.0290 | +0.0305 | 10 | +0.0178 |
| run65-dates | ridge_model | current-roster-20d | alpha_vs_sector_20d | 17 | 7023 | +0.0588 | +0.0259 | 10 | +0.0186 |
| run65-dates | signal_proxy | current-roster-20d | alpha_vs_sector_20d | 17 | 7023 | +0.0287 | +0.0281 | 10 | +0.0196 |
| run65-dates | structure_factor_signal | current-roster-20d | alpha_vs_sector_20d | 17 | 7023 | -0.0254 | -0.0185 | 10 | +0.0039 |
| run65-dates | xgboost_model | current-roster-20d | alpha_vs_sector_20d | 17 | 7023 | +0.0172 | +0.0353 | 10 | +0.0304 |
| run65-dates | xgboost_model | sector-specific-20d-ic-off | alpha_vs_sector_20d | 18 | 6803 | -0.0506 | -0.0156 | 10 | +0.0036 |
| shared-current | elastic_net_model | current-global-60d | alpha_vs_sector_60d | 507 | 215107 | +0.0090 | -0.0180 | 2 | +0.0248 |
| shared-current | ensemble_model | current-global-60d | alpha_vs_sector_60d | 507 | 215107 | -0.0046 | -0.0136 | 2 | -0.0235 |
| shared-current | event_ic_model | current-global-60d | alpha_vs_sector_60d | 507 | 215107 | -0.0292 | -0.0180 | 2 | -0.0176 |
| shared-current | event_signal | current-global-60d | alpha_vs_sector_60d | 507 | 215107 | -0.0062 | -0.0098 | 2 | -0.0156 |
| shared-current | ic_sign_model | current-global-60d | alpha_vs_sector_60d | 507 | 215107 | +0.0115 | +0.0182 | 2 | +0.0654 |
| shared-current | lasso_model | current-global-60d | alpha_vs_sector_60d | 507 | 215107 | +0.0249 | -0.0124 | 2 | +0.0417 |
| shared-current | reversal_rules | current-global-60d | alpha_vs_sector_60d | 507 | 215107 | -0.0206 | -0.0208 | 2 | +0.0362 |
| shared-current | ridge_model | current-global-60d | alpha_vs_sector_60d | 507 | 215107 | +0.0036 | -0.0173 | 2 | +0.0245 |
| shared-current | signal_proxy | current-global-60d | alpha_vs_sector_60d | 507 | 215107 | +0.0257 | +0.0155 | 2 | +0.0793 |
| shared-current | structure_factor_signal | current-global-60d | alpha_vs_sector_60d | 507 | 215107 | -0.0384 | -0.0225 | 2 | -0.0099 |
| shared-current | xgboost_model | current-global-60d | alpha_vs_sector_60d | 507 | 215107 | -0.0124 | +0.0077 | 2 | +0.0282 |
| shared-current | xgboost_model | sector-specific-20d-ic-off | alpha_vs_sector_20d | 507 | 190452 | -0.0074 | +0.0019 | 10 | +0.0175 |
| shared-current | xgboost_model | sector-specific-20d-ic-on | alpha_vs_sector_20d | 507 | 190452 | -0.0059 | -0.0041 | 10 | +0.0135 |

## IC-Off Decile Calibration

| decile | rows | mean_target | hit_rate |
|---:|---:|---:|---:|
| 0 | 19283 | +0.0081 | +0.4911 |
| 1 | 19023 | +0.0020 | +0.4882 |
| 2 | 18952 | +0.0016 | +0.4887 |
| 3 | 19032 | +0.0018 | +0.4902 |
| 4 | 19064 | -0.0006 | +0.4835 |
| 5 | 18927 | +0.0003 | +0.4906 |
| 6 | 18964 | +0.0008 | +0.4874 |
| 7 | 19020 | -0.0012 | +0.4777 |
| 8 | 18955 | +0.0009 | +0.4868 |
| 9 | 19232 | +0.0110 | +0.5093 |

## Verdict

flat: IC-screen-off sector-specific 20d xgboost per-date Spearman is +0.0019 on the shared grid versus IC-screen-on -0.0041, a delta of +0.0060 Spearman, or +1.5% of the ~0.39 forensic gap. On run 65's 20-date grid, the IC-screen-off pooled Spearman is -0.0506 versus the stored forensic +0.4644.
