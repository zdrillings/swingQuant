# C3 Binary Beat Label Bakeoff

- generated_at: 2026-10-06
- data_access: read-only DuckDB; no `./sq` writes; no `data/` mutation
- experiment: train raw 60d alpha regression twins versus P(alpha_vs_sector_60d >= +2%) classifiers, evaluate probabilities against raw 60d alpha on the honest stride-60 OOS grid
- target_raw: alpha_vs_sector_60d
- target_binary: beat_sector_2pct_60d_pos
- binary_threshold: 0.0200
- candidate_models: ridge_model, lasso_model, xgboost_model
- feature_set: current production MODEL_FEATURE_COLUMNS with A4 interactions
- eligible_universe_mode: passed_or_trend
- model_scope: global
- min_train_dates: 252
- max_train_dates: 252
- test_window_dates: 20
- evaluation_stride_dates: 20
- label_horizon_dates: 60
- xgboost_config: balanced_depth4
- eligible_rows: 345304
- eligible_dates: 832
- verdict: binary beat label wins (best Spearman delta +0.0128; best hit delta 0.6%; best beat delta 0.4%)

## Honest-Grid Spearman

| model | trained_label | honest_dates | honest_rows | pooled_raw_spearman | per_date_raw_spearman | delta_binary_minus_regression |
|---|---|---:|---:|---:|---:|---:|
| lasso_model | binary_beat_label | 514 | 7896 | -0.0087 | -0.0111 | +0.0128 |
| lasso_model | raw_alpha_regression | 514 | 7896 | -0.0215 | +0.0214 | n/a |
| ridge_model | binary_beat_label | 514 | 7896 | +0.0059 | -0.0028 | -0.0112 |
| ridge_model | raw_alpha_regression | 514 | 7896 | +0.0171 | -0.0130 | n/a |
| xgboost_model | binary_beat_label | 514 | 7896 | -0.0058 | -0.0269 | -0.0005 |
| xgboost_model | raw_alpha_regression | 514 | 7896 | -0.0053 | -0.0009 | n/a |

## Basket Hit/Beat

| model | trained_label | top_n | honest_dates | basket_rows | hit_rate | beat_rate | mean_alpha | delta_hit_rate | delta_beat_rate | delta_mean_alpha |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lasso_model | binary_beat_label | 2 | 514 | 1016 | 44.2% | 39.9% | +0.0009 | -4.1% | -2.3% | -0.0208 |
| lasso_model | raw_alpha_regression | 2 | 514 | 1016 | 48.2% | 42.2% | +0.0217 | 0.0% | 0.0% | +0.0000 |
| lasso_model | binary_beat_label | 10 | 514 | 4041 | 43.8% | 40.2% | +0.0003 | -1.0% | 0.2% | -0.0005 |
| lasso_model | raw_alpha_regression | 10 | 514 | 4041 | 44.7% | 40.0% | +0.0008 | 0.0% | 0.0% | +0.0000 |
| ridge_model | binary_beat_label | 2 | 514 | 1016 | 43.6% | 39.5% | -0.0008 | -3.7% | 0.0% | -0.0057 |
| ridge_model | raw_alpha_regression | 2 | 514 | 1016 | 47.3% | 39.5% | +0.0050 | 0.0% | 0.0% | +0.0000 |
| ridge_model | binary_beat_label | 10 | 514 | 4041 | 43.8% | 40.3% | +0.0005 | 0.4% | 0.4% | +0.0004 |
| ridge_model | raw_alpha_regression | 10 | 514 | 4041 | 43.4% | 39.9% | +0.0001 | 0.0% | 0.0% | +0.0000 |
| xgboost_model | binary_beat_label | 2 | 514 | 1016 | 46.3% | 41.1% | +0.0067 | 0.6% | -0.8% | -0.0075 |
| xgboost_model | raw_alpha_regression | 2 | 514 | 1016 | 45.7% | 41.8% | +0.0143 | 0.0% | 0.0% | +0.0000 |
| xgboost_model | binary_beat_label | 10 | 514 | 4041 | 42.8% | 40.0% | -0.0009 | -2.5% | -0.5% | -0.0028 |
| xgboost_model | raw_alpha_regression | 10 | 514 | 4041 | 45.3% | 40.5% | +0.0019 | 0.0% | 0.0% | +0.0000 |

## Verdict

binary beat label wins: best Spearman delta +0.0128; best hit delta 0.6%; best beat delta 0.4%.
