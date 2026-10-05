## Regime Flip Attribution

- note: diagnostic only; variants are not auto-adopted and do not alter champion selection, OOS artifacts, or live predictions.
- variants: as_is, disabled, unconditional_reversal

| model | variant | full_oos_spearman | mean_target | hit_rate | beat_universe_rate |
|---|---|---:|---:|---:|---:|
| elastic_net_model | as_is | -0.017064 | 0.025883 | 0.416185 | 0.433526 |
| elastic_net_model | disabled | -0.017064 | 0.025883 | 0.416185 | 0.433526 |
| elastic_net_model | unconditional_reversal | -0.022686 | 0.016650 | 0.437380 | 0.452794 |
| ensemble_model | as_is | 0.013066 | 0.064950 | 0.528902 | 0.554913 |
| ensemble_model | disabled | 0.013066 | 0.064950 | 0.528902 | 0.554913 |
| ensemble_model | unconditional_reversal | -0.008304 | 0.040409 | 0.492293 | 0.512524 |
| event_ic_model | as_is | -0.017348 | -0.020326 | 0.425819 | 0.429672 |
| event_ic_model | disabled | -0.017348 | -0.020326 | 0.425819 | 0.429672 |
| event_ic_model | unconditional_reversal | -0.018685 | 0.003448 | 0.451830 | 0.456647 |
| event_signal | as_is | -0.011596 | -0.019031 | 0.407514 | 0.420039 |
| event_signal | disabled | -0.011596 | -0.019031 | 0.407514 | 0.420039 |
| event_signal | unconditional_reversal | -0.017876 | -0.017591 | 0.398844 | 0.412331 |
| ic_sign_model | as_is | 0.018774 | 0.047380 | 0.469171 | 0.514451 |
| ic_sign_model | disabled | 0.018774 | 0.047380 | 0.469171 | 0.514451 |
| ic_sign_model | unconditional_reversal | 0.018133 | 0.040378 | 0.450867 | 0.481696 |
| lasso_model | as_is | -0.011110 | 0.032985 | 0.399807 | 0.435453 |
| lasso_model | disabled | -0.011110 | 0.032985 | 0.399807 | 0.435453 |
| lasso_model | unconditional_reversal | -0.016802 | 0.035845 | 0.441233 | 0.475915 |
| reversal_rules | as_is | -0.017262 | 0.020222 | 0.474952 | 0.485549 |
| reversal_rules | disabled | -0.017262 | 0.020222 | 0.474952 | 0.485549 |
| reversal_rules | unconditional_reversal | 0.018611 | 0.066681 | 0.524085 | 0.549133 |
| ridge_model | as_is | -0.016603 | 0.027137 | 0.435453 | 0.458574 |
| ridge_model | disabled | -0.016603 | 0.027137 | 0.435453 | 0.458574 |
| ridge_model | unconditional_reversal | -0.023145 | 0.010783 | 0.438343 | 0.452794 |
| signal_proxy | as_is | 0.013066 | 0.064950 | 0.528902 | 0.554913 |
| signal_proxy | disabled | 0.013066 | 0.064950 | 0.528902 | 0.554913 |
| signal_proxy | unconditional_reversal | -0.008304 | 0.040409 | 0.492293 | 0.512524 |
| structure_factor_signal | as_is | -0.028221 | -0.013016 | 0.444123 | 0.450867 |
| structure_factor_signal | disabled | -0.028221 | -0.013016 | 0.444123 | 0.450867 |
| structure_factor_signal | unconditional_reversal | 0.020952 | -0.000007 | 0.495183 | 0.508671 |
| xgboost_model | as_is | 0.008171 | 0.022361 | 0.460501 | 0.497110 |
| xgboost_model | disabled | 0.008171 | 0.022361 | 0.460501 | 0.497110 |
| xgboost_model | unconditional_reversal | 0.013382 | 0.037696 | 0.483622 | 0.535645 |
