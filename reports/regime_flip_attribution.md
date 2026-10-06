## Regime Flip Attribution

- note: diagnostic only; variants are not auto-adopted and do not alter champion selection, OOS artifacts, or live predictions.
- variants: as_is, disabled, unconditional_reversal

| model | variant | full_oos_spearman | mean_target | hit_rate | beat_universe_rate |
|---|---|---:|---:|---:|---:|
| elastic_net_model | as_is | -0.017000 | 0.025717 | 0.416346 | 0.432692 |
| elastic_net_model | disabled | -0.017000 | 0.025717 | 0.416346 | 0.432692 |
| elastic_net_model | unconditional_reversal | -0.022610 | 0.016503 | 0.437500 | 0.451923 |
| ensemble_model | as_is | 0.012889 | 0.064520 | 0.527885 | 0.553846 |
| ensemble_model | disabled | 0.012889 | 0.064520 | 0.527885 | 0.553846 |
| ensemble_model | unconditional_reversal | -0.008440 | 0.040027 | 0.491346 | 0.511538 |
| event_ic_model | as_is | -0.017551 | -0.020008 | 0.426923 | 0.430769 |
| event_ic_model | disabled | -0.017551 | -0.020008 | 0.426923 | 0.430769 |
| event_ic_model | unconditional_reversal | -0.018885 | 0.003720 | 0.452885 | 0.457692 |
| event_signal | as_is | -0.011678 | -0.018501 | 0.408654 | 0.421154 |
| event_signal | disabled | -0.011678 | -0.018501 | 0.408654 | 0.421154 |
| event_signal | unconditional_reversal | -0.017946 | -0.017064 | 0.400000 | 0.413462 |
| ic_sign_model | as_is | 0.019020 | 0.047266 | 0.468269 | 0.515385 |
| ic_sign_model | disabled | 0.019020 | 0.047266 | 0.468269 | 0.515385 |
| ic_sign_model | unconditional_reversal | 0.018381 | 0.040277 | 0.450000 | 0.482692 |
| lasso_model | as_is | -0.011005 | 0.032806 | 0.400000 | 0.434615 |
| lasso_model | disabled | -0.011005 | 0.032806 | 0.400000 | 0.434615 |
| lasso_model | unconditional_reversal | -0.016686 | 0.035660 | 0.441346 | 0.475000 |
| reversal_rules | as_is | -0.017381 | 0.019879 | 0.474038 | 0.484615 |
| reversal_rules | disabled | -0.017381 | 0.019879 | 0.474038 | 0.484615 |
| reversal_rules | unconditional_reversal | 0.018424 | 0.066248 | 0.523077 | 0.548077 |
| ridge_model | as_is | -0.016606 | 0.026969 | 0.435577 | 0.457692 |
| ridge_model | disabled | -0.016606 | 0.026969 | 0.435577 | 0.457692 |
| ridge_model | unconditional_reversal | -0.023136 | 0.010647 | 0.438462 | 0.451923 |
| signal_proxy | as_is | 0.012889 | 0.064520 | 0.527885 | 0.553846 |
| signal_proxy | disabled | 0.012889 | 0.064520 | 0.527885 | 0.553846 |
| signal_proxy | unconditional_reversal | -0.008440 | 0.040027 | 0.491346 | 0.511538 |
| structure_factor_signal | as_is | -0.027782 | -0.012942 | 0.444231 | 0.451923 |
| structure_factor_signal | disabled | -0.027782 | -0.012942 | 0.444231 | 0.451923 |
| structure_factor_signal | unconditional_reversal | 0.021297 | 0.000042 | 0.495192 | 0.509615 |
| xgboost_model | as_is | 0.008225 | 0.022429 | 0.460577 | 0.498077 |
| xgboost_model | disabled | 0.008225 | 0.022429 | 0.460577 | 0.498077 |
| xgboost_model | unconditional_reversal | 0.013426 | 0.037735 | 0.483654 | 0.536538 |
