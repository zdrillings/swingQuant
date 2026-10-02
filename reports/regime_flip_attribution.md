## Regime Flip Attribution

- note: diagnostic only; variants are not auto-adopted and do not alter champion selection, OOS artifacts, or live predictions.
- variants: as_is, disabled, unconditional_reversal

| model | variant | full_oos_spearman | mean_target | hit_rate | beat_universe_rate |
|---|---|---:|---:|---:|---:|
| elastic_net_model | as_is | -0.017089 | 0.026055 | 0.416023 | 0.434363 |
| elastic_net_model | disabled | -0.017089 | 0.026055 | 0.416023 | 0.434363 |
| elastic_net_model | unconditional_reversal | -0.022721 | 0.016805 | 0.437259 | 0.453668 |
| ensemble_model | as_is | 0.013312 | 0.065537 | 0.529923 | 0.555985 |
| ensemble_model | disabled | 0.013312 | 0.065537 | 0.529923 | 0.555985 |
| ensemble_model | unconditional_reversal | -0.008100 | 0.040949 | 0.493243 | 0.513514 |
| event_ic_model | as_is | -0.017169 | -0.020626 | 0.424710 | 0.428571 |
| event_ic_model | disabled | -0.017169 | -0.020626 | 0.424710 | 0.428571 |
| event_ic_model | unconditional_reversal | -0.018509 | 0.003195 | 0.450772 | 0.455598 |
| event_signal | as_is | -0.011576 | -0.019185 | 0.407336 | 0.418919 |
| event_signal | disabled | -0.011576 | -0.019185 | 0.407336 | 0.418919 |
| event_signal | unconditional_reversal | -0.017868 | -0.017742 | 0.398649 | 0.411197 |
| ic_sign_model | as_is | 0.018494 | 0.047528 | 0.470077 | 0.515444 |
| ic_sign_model | disabled | 0.018494 | 0.047528 | 0.470077 | 0.515444 |
| ic_sign_model | unconditional_reversal | 0.017852 | 0.040512 | 0.451737 | 0.482625 |
| lasso_model | as_is | -0.011220 | 0.033171 | 0.399614 | 0.436293 |
| lasso_model | disabled | -0.011220 | 0.033171 | 0.399614 | 0.436293 |
| lasso_model | unconditional_reversal | -0.016923 | 0.036036 | 0.441120 | 0.476834 |
| reversal_rules | as_is | -0.017075 | 0.020723 | 0.475869 | 0.486486 |
| reversal_rules | disabled | -0.017075 | 0.020723 | 0.475869 | 0.486486 |
| reversal_rules | unconditional_reversal | 0.018867 | 0.067272 | 0.525097 | 0.550193 |
| ridge_model | as_is | -0.016546 | 0.027312 | 0.435328 | 0.459459 |
| ridge_model | disabled | -0.016546 | 0.027312 | 0.435328 | 0.459459 |
| ridge_model | unconditional_reversal | -0.023101 | 0.010926 | 0.438224 | 0.453668 |
| signal_proxy | as_is | 0.013312 | 0.065537 | 0.529923 | 0.555985 |
| signal_proxy | disabled | 0.013312 | 0.065537 | 0.529923 | 0.555985 |
| signal_proxy | unconditional_reversal | -0.008100 | 0.040949 | 0.493243 | 0.513514 |
| structure_factor_signal | as_is | -0.028674 | -0.012944 | 0.444981 | 0.451737 |
| structure_factor_signal | disabled | -0.028674 | -0.012944 | 0.444981 | 0.451737 |
| structure_factor_signal | unconditional_reversal | 0.020594 | 0.000089 | 0.496139 | 0.509653 |
| xgboost_model | as_is | 0.008123 | 0.022444 | 0.461390 | 0.496139 |
| xgboost_model | disabled | 0.008123 | 0.022444 | 0.461390 | 0.496139 |
| xgboost_model | unconditional_reversal | 0.013344 | 0.037808 | 0.484556 | 0.534749 |
