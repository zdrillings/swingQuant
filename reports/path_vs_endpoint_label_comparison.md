# Shortlist Path vs Endpoint Label Comparison

- source_oos_csv: reports/shortlist_model_oos_predictions.csv
- promotion_basket_size: 2

| model | target | dates | spearman | top2_mean | top2_hit | top2_mean_excess | top2_hit_excess | beat_universe | universe_mean | universe_hit |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| elastic_net_model | alpha_vs_sector_60d | 187 | 0.0353 | 0.0791 | 0.4171 | 0.0747 | -0.0279 | 0.4652 | 0.0043 | 0.4450 |
| elastic_net_model | path_alpha_vs_sector_60d | 206 | 0.0114 | -0.0007 | 0.2427 | 0.0000 | -0.1424 | 0.3301 | -0.0007 | 0.3851 |
| ensemble_model | alpha_vs_sector_60d | 187 | 0.0137 | 0.0095 | 0.4572 | 0.0052 | 0.0122 | 0.5134 | 0.0043 | 0.4450 |
| ensemble_model | path_alpha_vs_sector_60d | 206 | 0.0055 | -0.0032 | 0.4053 | -0.0025 | 0.0202 | 0.4029 | -0.0007 | 0.3851 |
| event_ic_model | alpha_vs_sector_60d | 187 | 0.0026 | -0.0112 | 0.4439 | -0.0155 | -0.0011 | 0.4545 | 0.0043 | 0.4450 |
| event_ic_model | path_alpha_vs_sector_60d | 206 | -0.0119 | -0.0051 | 0.2864 | -0.0044 | -0.0987 | 0.3932 | -0.0007 | 0.3851 |
| event_signal | alpha_vs_sector_60d | 187 | 0.0004 | 0.0253 | 0.4733 | 0.0210 | 0.0283 | 0.5401 | 0.0043 | 0.4450 |
| event_signal | path_alpha_vs_sector_60d | 206 | 0.0140 | -0.0111 | 0.3277 | -0.0104 | -0.0575 | 0.3786 | -0.0007 | 0.3851 |
| ic_sign_model | alpha_vs_sector_60d | 187 | -0.0237 | 0.0506 | 0.4465 | 0.0463 | 0.0015 | 0.4920 | 0.0043 | 0.4450 |
| ic_sign_model | path_alpha_vs_sector_60d | 206 | -0.0414 | -0.0104 | 0.3617 | -0.0097 | -0.0235 | 0.3689 | -0.0007 | 0.3851 |
| lasso_model | alpha_vs_sector_60d | 187 | 0.0340 | 0.0780 | 0.3957 | 0.0737 | -0.0493 | 0.4171 | 0.0043 | 0.4450 |
| lasso_model | path_alpha_vs_sector_60d | 206 | 0.0113 | 0.0010 | 0.2476 | 0.0017 | -0.1376 | 0.3592 | -0.0007 | 0.3851 |
| reversal_rules | alpha_vs_sector_60d | 187 | 0.0094 | 0.0542 | 0.4706 | 0.0499 | 0.0256 | 0.4813 | 0.0043 | 0.4450 |
| reversal_rules | path_alpha_vs_sector_60d | 206 | -0.0480 | 0.0043 | 0.3180 | 0.0050 | -0.0672 | 0.3786 | -0.0007 | 0.3851 |
| ridge_model | alpha_vs_sector_60d | 187 | 0.0628 | 0.1257 | 0.4786 | 0.1214 | 0.0336 | 0.5080 | 0.0043 | 0.4450 |
| ridge_model | path_alpha_vs_sector_60d | 206 | 0.0099 | 0.0046 | 0.2937 | 0.0053 | -0.0914 | 0.3398 | -0.0007 | 0.3851 |
| signal_proxy | alpha_vs_sector_60d | 187 | 0.0471 | 0.0649 | 0.5080 | 0.0606 | 0.0630 | 0.5294 | 0.0043 | 0.4450 |
| signal_proxy | path_alpha_vs_sector_60d | 206 | -0.0489 | 0.0061 | 0.3083 | 0.0068 | -0.0769 | 0.3932 | -0.0007 | 0.3851 |
| structure_factor_fresh_signal | alpha_vs_sector_60d | 187 | -0.0491 | -0.0006 | 0.4759 | -0.0049 | 0.0309 | 0.4973 | 0.0043 | 0.4450 |
| structure_factor_fresh_signal | path_alpha_vs_sector_60d | 206 | 0.0607 | -0.0027 | 0.4417 | -0.0020 | 0.0566 | 0.4175 | -0.0007 | 0.3851 |
| structure_factor_signal | alpha_vs_sector_60d | 187 | -0.0457 | 0.0004 | 0.4759 | -0.0039 | 0.0309 | 0.5080 | 0.0043 | 0.4450 |
| structure_factor_signal | path_alpha_vs_sector_60d | 206 | 0.0623 | 0.0006 | 0.4563 | 0.0013 | 0.0712 | 0.4417 | -0.0007 | 0.3851 |
| xgboost_model | alpha_vs_sector_60d | 187 | 0.0163 | 0.0849 | 0.4572 | 0.0806 | 0.0122 | 0.5187 | 0.0043 | 0.4450 |
| xgboost_model | path_alpha_vs_sector_60d | 206 | 0.0036 | 0.0102 | 0.3277 | 0.0109 | -0.0575 | 0.3981 | -0.0007 | 0.3851 |
