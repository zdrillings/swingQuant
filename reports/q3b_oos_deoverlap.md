# Q3b OOS De-Overlap

- source_oos_predictions: /home/zdrillings/code/SwingQuant/reports/shortlist_model_oos_predictions.csv
- target_column: alpha_vs_sector_60d
- horizon_days: 60
- deoverlap_policy: greedy non-overlapping rows per ticker using the 60-session forward label window
- calendar_dates: 1359
- dense_rows: 2419813
- honest_rows: 86779
- dense_dates: 519
- honest_dates: 513

## Full-OOS Spearman

| model | dense_dates | honest_dates | dense_spearman | honest_spearman | delta |
|---|---:|---:|---:|---:|---:|
| elastic_net_model | 519 | 513 | -0.0171 | -0.0015 | 0.0156 |
| ensemble_model | 519 | 513 | -0.0139 | -0.0061 | 0.0078 |
| event_ic_model | 519 | 513 | -0.0173 | -0.0345 | -0.0172 |
| event_signal | 519 | 513 | -0.0116 | 0.0238 | 0.0354 |
| ic_sign_model | 519 | 513 | 0.0188 | 0.0049 | -0.0139 |
| lasso_model | 519 | 513 | -0.0111 | 0.0052 | 0.0163 |
| reversal_rules | 519 | 513 | -0.0173 | -0.0190 | -0.0017 |
| ridge_model | 519 | 513 | -0.0166 | -0.0012 | 0.0154 |
| signal_proxy | 519 | 513 | 0.0131 | 0.0207 | 0.0076 |
| structure_factor_signal | 519 | 513 | -0.0282 | -0.0168 | 0.0114 |
| xgboost_model | 519 | 513 | 0.0082 | -0.0142 | -0.0223 |

## Gate Re-Adjudication

| model | dense_gate | honest_gate | dense_full_oos | honest_full_oos | honest_last_fold | honest_trailing_3folds |
|---|---|---|---:|---:|---:|---:|
| elastic_net_model | FAIL | FAIL | -0.0171 | -0.0015 | 0.0408 | -0.0720 |
| ensemble_model | FAIL | FAIL | -0.0139 | -0.0061 | 0.0528 | -0.1505 |
| event_ic_model | FAIL | FAIL | -0.0173 | -0.0345 | -0.0455 | -0.1786 |
| event_signal | FAIL | FAIL | -0.0116 | 0.0238 | -0.0898 | -0.0766 |
| ic_sign_model | FAIL | FAIL | 0.0188 | 0.0049 | -0.0904 | -0.1313 |
| lasso_model | FAIL | FAIL | -0.0111 | 0.0052 | 0.0498 | -0.0953 |
| reversal_rules | FAIL | FAIL | -0.0173 | -0.0190 | -0.1115 | -0.1060 |
| ridge_model | FAIL | FAIL | -0.0166 | -0.0012 | 0.0294 | -0.0759 |
| signal_proxy | FAIL | FAIL | 0.0131 | 0.0207 | -0.1115 | -0.1060 |
| structure_factor_signal | FAIL | FAIL | -0.0282 | -0.0168 | 0.0892 | 0.1404 |
| xgboost_model | FAIL | FAIL | 0.0082 | -0.0142 | -0.2334 | -0.1552 |
