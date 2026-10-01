## Regime Flip Attribution

- note: diagnostic only; variants are not auto-adopted and do not alter champion selection, OOS artifacts, or live predictions.
- variants: as_is, disabled, unconditional_reversal

| model | variant | full_oos_spearman | mean_target | hit_rate | beat_universe_rate |
|---|---|---:|---:|---:|---:|
| elastic_net_model | as_is | -0.017142 | 0.026538 | 0.416828 | 0.435203 |
| elastic_net_model | disabled | -0.017142 | 0.026538 | 0.416828 | 0.435203 |
| elastic_net_model | unconditional_reversal | -0.022785 | 0.017270 | 0.438104 | 0.454545 |
| ensemble_model | as_is | 0.013453 | 0.066349 | 0.530948 | 0.557060 |
| ensemble_model | disabled | 0.013453 | 0.066349 | 0.530948 | 0.557060 |
| ensemble_model | unconditional_reversal | -0.008000 | 0.041714 | 0.494197 | 0.514507 |
| event_ic_model | as_is | -0.017157 | -0.020901 | 0.423598 | 0.427466 |
| event_ic_model | disabled | -0.017157 | -0.020901 | 0.423598 | 0.427466 |
| event_ic_model | unconditional_reversal | -0.018499 | 0.002966 | 0.449710 | 0.454545 |
| event_signal | as_is | -0.011534 | -0.019348 | 0.407157 | 0.417795 |
| event_signal | disabled | -0.011534 | -0.019348 | 0.407157 | 0.417795 |
| event_signal | unconditional_reversal | -0.017838 | -0.017902 | 0.398453 | 0.410058 |
| ic_sign_model | as_is | 0.018307 | 0.047479 | 0.470019 | 0.514507 |
| ic_sign_model | disabled | 0.018307 | 0.047479 | 0.470019 | 0.514507 |
| ic_sign_model | unconditional_reversal | 0.017664 | 0.040450 | 0.451644 | 0.481625 |
| lasso_model | as_is | -0.011367 | 0.033667 | 0.400387 | 0.437137 |
| lasso_model | disabled | -0.011367 | 0.033667 | 0.400387 | 0.437137 |
| lasso_model | unconditional_reversal | -0.017081 | 0.036539 | 0.441973 | 0.477756 |
| reversal_rules | as_is | -0.016993 | 0.021449 | 0.476789 | 0.487427 |
| reversal_rules | disabled | -0.016993 | 0.021449 | 0.476789 | 0.487427 |
| reversal_rules | unconditional_reversal | 0.019019 | 0.068087 | 0.526112 | 0.551257 |
| ridge_model | as_is | -0.016498 | 0.027439 | 0.436170 | 0.460348 |
| ridge_model | disabled | -0.016498 | 0.027439 | 0.436170 | 0.460348 |
| ridge_model | unconditional_reversal | -0.023065 | 0.011022 | 0.439072 | 0.454545 |
| signal_proxy | as_is | 0.013453 | 0.066349 | 0.530948 | 0.557060 |
| signal_proxy | disabled | 0.013453 | 0.066349 | 0.530948 | 0.557060 |
| signal_proxy | unconditional_reversal | -0.008000 | 0.041714 | 0.494197 | 0.514507 |
| structure_factor_signal | as_is | -0.029166 | -0.012778 | 0.444874 | 0.452611 |
| structure_factor_signal | disabled | -0.029166 | -0.012778 | 0.444874 | 0.452611 |
| structure_factor_signal | unconditional_reversal | 0.020198 | 0.000281 | 0.496132 | 0.510638 |
| xgboost_model | as_is | 0.008301 | 0.022610 | 0.462282 | 0.497099 |
| xgboost_model | disabled | 0.008301 | 0.022610 | 0.462282 | 0.497099 |
| xgboost_model | unconditional_reversal | 0.013532 | 0.038005 | 0.485493 | 0.535783 |
