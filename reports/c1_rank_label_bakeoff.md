# C1 Rank-Percentile Label Bakeoff

- generated_at: 2026-10-05
- data_access: read-only DuckDB; no `./sq` writes; no `data/` mutation
- experiment: train raw 60d alpha versus per-date rank-percentile 60d alpha, evaluate both against original 60d alpha on the honest stride-60 OOS grid
- target_raw: alpha_vs_sector_60d
- target_rank: alpha_vs_sector_60d_pct_rank
- candidate_models: signal_proxy, ridge_model, lasso_model, xgboost_model
- eligible_universe_mode: passed_or_trend
- model_scope: global
- min_train_dates: 252
- max_train_dates: 252
- test_window_dates: 20
- evaluation_stride_dates: 20
- label_horizon_dates: 60
- xgboost_config: balanced_depth4
- eligible_rows: 344914
- eligible_dates: 831
- honest_rows: 63112
- honest_dates: 513
- verdict: rank label ties (0 models cleared the +0.0200 win bar; best delta +0.0185, mean delta -0.0006)

## Honest-Grid Spearman

| model | trained_label | honest_dates | honest_rows | pooled_raw_spearman | per_date_raw_spearman | pooled_rank_spearman | per_date_rank_spearman | raw_decile_spread | rank_decile_spread | delta_rank_minus_raw |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lasso_model | rank_label | 513 | 7889 | -0.0221 | +0.0202 | +0.0112 | +0.0202 | +0.0250 | +0.0352 | -0.0012 |
| lasso_model | raw_label | 513 | 7889 | -0.0209 | +0.0317 | +0.0282 | +0.0317 | +0.0497 | +0.0472 | n/a |
| ridge_model | rank_label | 513 | 7889 | -0.0391 | +0.0140 | +0.0108 | +0.0140 | +0.0264 | +0.0389 | -0.0198 |
| ridge_model | raw_label | 513 | 7889 | -0.0193 | +0.0400 | +0.0253 | +0.0400 | +0.0318 | +0.0446 | n/a |
| signal_proxy | rank_label | 513 | 7889 | +0.0085 | +0.0206 | +0.0198 | +0.0206 | +0.0408 | +0.0248 | +0.0000 |
| signal_proxy | raw_label | 513 | 7889 | +0.0085 | +0.0206 | +0.0198 | +0.0206 | +0.0408 | +0.0248 | n/a |
| xgboost_model | rank_label | 513 | 7889 | -0.0083 | -0.0100 | -0.0108 | -0.0100 | +0.0149 | +0.0161 | +0.0185 |
| xgboost_model | raw_label | 513 | 7889 | -0.0268 | -0.0025 | -0.0058 | -0.0025 | +0.0141 | +0.0024 | n/a |

## Rank-Label Decile Calibration

### signal_proxy

| decile | rows | mean_raw_target | mean_rank_target |
|---:|---:|---:|---:|
| 0 | 787 | -0.0148 | +0.4962 |
| 1 | 640 | -0.0263 | +0.4758 |
| 2 | 618 | -0.0008 | +0.5188 |
| 3 | 652 | +0.0030 | +0.5181 |
| 4 | 654 | +0.0010 | +0.5174 |
| 5 | 607 | -0.0028 | +0.5101 |
| 6 | 623 | -0.0071 | +0.4829 |
| 7 | 647 | -0.0160 | +0.4868 |
| 8 | 611 | +0.0078 | +0.4948 |
| 9 | 756 | +0.0260 | +0.5210 |

### ridge_model

| decile | rows | mean_raw_target | mean_rank_target |
|---:|---:|---:|---:|
| 0 | 787 | -0.0101 | +0.4887 |
| 1 | 640 | +0.0073 | +0.5206 |
| 2 | 618 | -0.0085 | +0.4966 |
| 3 | 652 | -0.0062 | +0.4890 |
| 4 | 654 | +0.0104 | +0.5207 |
| 5 | 607 | -0.0220 | +0.4799 |
| 6 | 623 | -0.0010 | +0.5013 |
| 7 | 647 | -0.0140 | +0.4896 |
| 8 | 611 | -0.0201 | +0.4830 |
| 9 | 756 | +0.0163 | +0.5276 |

### lasso_model

| decile | rows | mean_raw_target | mean_rank_target |
|---:|---:|---:|---:|
| 0 | 787 | -0.0080 | +0.4929 |
| 1 | 640 | +0.0035 | +0.5124 |
| 2 | 618 | -0.0038 | +0.5072 |
| 3 | 652 | -0.0110 | +0.4894 |
| 4 | 654 | -0.0123 | +0.4817 |
| 5 | 607 | +0.0044 | +0.5263 |
| 6 | 623 | -0.0071 | +0.4925 |
| 7 | 647 | -0.0073 | +0.4979 |
| 8 | 611 | -0.0213 | +0.4733 |
| 9 | 756 | +0.0170 | +0.5281 |

### xgboost_model

| decile | rows | mean_raw_target | mean_rank_target |
|---:|---:|---:|---:|
| 0 | 787 | -0.0096 | +0.4881 |
| 1 | 640 | -0.0070 | +0.4961 |
| 2 | 618 | +0.0036 | +0.5168 |
| 3 | 652 | -0.0045 | +0.5069 |
| 4 | 654 | -0.0050 | +0.5040 |
| 5 | 607 | -0.0056 | +0.5040 |
| 6 | 623 | -0.0102 | +0.4916 |
| 7 | 647 | +0.0002 | +0.5066 |
| 8 | 611 | -0.0186 | +0.4759 |
| 9 | 756 | +0.0053 | +0.5042 |

## Verdict

rank label ties: 0 models cleared the +0.0200 win bar; best delta +0.0185, mean delta -0.0006.
