# P3 Adaptive Training Window

- idea: test whether the recent fold collapse is adaptation lag from a one-year training window
- data_access: DuckDB read_only=True; no `./sq` writes; no data/ mutation
- target_column: alpha_vs_sector_60d
- model_scope: global
- candidate_models: ridge_model, lasso_model, xgboost_model
- baseline_max_train_dates: 252
- short_max_train_dates: 126
- recency_weight_half_life_dates: 63
- test_window_dates: 20
- evaluation_stride_dates: 60
- label_horizon_dates: 60
- eligible_rows: 345304
- eligible_dates: 832
- verdict: wins: adaptive training lifts trailing-3fold Spearman by +0.0848 for ridge_model max_train_126 without dropping full-OOS below baseline

## Variant Table

| model | variant | rows | dates | full_oos_spearman | delta_full_vs_252 | last_fold_spearman | delta_last_vs_252 | trailing_3fold_spearman | delta_trailing3_vs_252 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| lasso_model | max_train_126 | 6650 | 180 | +0.0407 | -0.0045 | +0.0834 | -0.0042 | +0.1083 | -0.0332 |
| lasso_model | max_train_252 | 6650 | 180 | +0.0452 | +0.0000 | +0.0876 | +0.0000 | +0.1414 | +0.0000 |
| lasso_model | weighted_hl_63 | 6650 | 180 | +0.0239 | -0.0213 | +0.0874 | -0.0003 | +0.0851 | -0.0563 |
| ridge_model | max_train_126 | 6650 | 180 | +0.0318 | +0.0478 | +0.0952 | +0.0985 | +0.1063 | +0.0848 |
| ridge_model | max_train_252 | 6650 | 180 | -0.0160 | +0.0000 | -0.0032 | +0.0000 | +0.0215 | +0.0000 |
| ridge_model | weighted_hl_63 | 6650 | 180 | -0.0172 | -0.0012 | +0.0044 | +0.0076 | +0.0407 | +0.0191 |
| xgboost_model | max_train_126 | 6650 | 180 | +0.0233 | -0.0024 | -0.0721 | -0.0304 | +0.0353 | -0.0555 |
| xgboost_model | max_train_252 | 6650 | 180 | +0.0256 | +0.0000 | -0.0416 | +0.0000 | +0.0908 | +0.0000 |

## Verification Notes

- Success rule from the critique: an adaptive variant must lift trailing-3fold Spearman by at least +0.0200 on at least one model without dropping full-OOS below that model's 252-session baseline.
- Production training config was not edited; this is a report-only bakeoff.
