# Q1b Sector-Specific 20d Ablation

- generated_at: 2026-09-30
- data_access: read-only DuckDB plus existing OOS CSV; no `./sq` writes; no `data/` mutation
- baseline_reference: run 65 / 00144e4, 20d top-10 sector-specific xgboost, pooled Spearman +0.4644 on 20 OOS dates
- experiment: 20d endpoint label, top-10 readout, sector-specific xgboost, horizon-strided training labels, current full feature set with fold-local IC screen, regime matching off
- feature_note: the exact 00144e4 fold-local survivor list was not retained in git artifacts; this isolates sector-specific 20d modeling without pretending to recover unavailable survivor state
- eligible_rows: 310859
- eligible_dates: 877
- sector_specific_oos_dates: 605
- shared_current_grid_dates: 505

## Shared-Grid Spearman

| model | path | target | dates | rows | pooled_spearman | per_date_spearman | top_n | top_mean |
|---|---|---|---:|---:|---:|---:|---:|---:|
| elastic_net_model | current-global-60d | alpha_vs_sector_60d | 505 | 214366 | +0.0089 | -0.0180 | 2 | +0.0254 |
| ensemble_model | current-global-60d | alpha_vs_sector_60d | 505 | 214366 | -0.0047 | -0.0138 | 2 | -0.0240 |
| event_ic_model | current-global-60d | alpha_vs_sector_60d | 505 | 214366 | -0.0291 | -0.0178 | 2 | -0.0182 |
| event_signal | current-global-60d | alpha_vs_sector_60d | 505 | 214366 | -0.0062 | -0.0098 | 2 | -0.0159 |
| ic_sign_model | current-global-60d | alpha_vs_sector_60d | 505 | 214366 | +0.0110 | +0.0178 | 2 | +0.0656 |
| lasso_model | current-global-60d | alpha_vs_sector_60d | 505 | 214366 | +0.0247 | -0.0126 | 2 | +0.0425 |
| reversal_rules | current-global-60d | alpha_vs_sector_60d | 505 | 214366 | -0.0204 | -0.0205 | 2 | +0.0376 |
| ridge_model | current-global-60d | alpha_vs_sector_60d | 505 | 214366 | +0.0034 | -0.0172 | 2 | +0.0248 |
| signal_proxy | current-global-60d | alpha_vs_sector_60d | 505 | 214366 | +0.0261 | +0.0159 | 2 | +0.0808 |
| structure_factor_signal | current-global-60d | alpha_vs_sector_60d | 505 | 214366 | -0.0394 | -0.0234 | 2 | -0.0097 |
| xgboost_model | current-global-60d | alpha_vs_sector_60d | 505 | 214366 | -0.0124 | +0.0079 | 2 | +0.0285 |
| xgboost_model | sector-specific-20d | alpha_vs_sector_20d | 505 | 189711 | -0.0049 | -0.0034 | 10 | +0.0137 |

## Sector-Specific Decile Calibration

| decile | rows | mean_target | hit_rate |
|---:|---:|---:|---:|
| 0 | 19208 | +0.0098 | +0.4998 |
| 1 | 18947 | +0.0038 | +0.4899 |
| 2 | 18881 | +0.0004 | +0.4874 |
| 3 | 18957 | +0.0010 | +0.4865 |
| 4 | 18990 | +0.0001 | +0.4887 |
| 5 | 18853 | +0.0002 | +0.4878 |
| 6 | 18891 | -0.0011 | +0.4828 |
| 7 | 18947 | -0.0009 | +0.4814 |
| 8 | 18881 | +0.0020 | +0.4864 |
| 9 | 19156 | +0.0101 | +0.5038 |

## Verdict

flat: sector-specific 20d xgboost per-date Spearman is -0.0034 on the shared grid versus the current global roster best +0.0178, recovering -0.0212 Spearman, or -5.4% of the ~0.39 forensic gap.
