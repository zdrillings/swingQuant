## Regime Flip Attribution

- note: diagnostic only; variants are not auto-adopted and do not alter champion selection, OOS artifacts, or live predictions.
- variants: as_is, disabled, unconditional_reversal

| model | variant | full_oos_spearman | mean_target | hit_rate | beat_universe_rate |
|---|---|---:|---:|---:|---:|
| base_pattern_specialist | as_is | 0.008784 | 0.021538 | 0.447419 | 0.521989 |
| base_pattern_specialist | disabled | 0.008784 | 0.021538 | 0.447419 | 0.521989 |
| base_pattern_specialist | unconditional_reversal | -0.012234 | -0.003767 | 0.417782 | 0.476099 |
| beat_hybrid | as_is | -0.010593 | 0.010212 | 0.432122 | 0.485660 |
| beat_hybrid | disabled | -0.010593 | 0.010212 | 0.432122 | 0.485660 |
| beat_hybrid | unconditional_reversal | -0.029435 | 0.002507 | 0.441683 | 0.476099 |
| beat_logistic_ridge | as_is | 0.010965 | 0.011210 | 0.463671 | 0.514340 |
| beat_logistic_ridge | disabled | 0.010965 | 0.011210 | 0.463671 | 0.514340 |
| beat_logistic_ridge | unconditional_reversal | -0.009641 | 0.001367 | 0.460803 | 0.504780 |
| elastic_net_model | as_is | -0.019489 | 0.010853 | 0.400574 | 0.435946 |
| elastic_net_model | disabled | -0.019489 | 0.010853 | 0.400574 | 0.435946 |
| elastic_net_model | unconditional_reversal | -0.031098 | 0.000284 | 0.414914 | 0.439771 |
| ensemble_model | as_is | 0.012343 | 0.063931 | 0.526769 | 0.554493 |
| ensemble_model | disabled | 0.012343 | 0.063931 | 0.526769 | 0.554493 |
| ensemble_model | unconditional_reversal | -0.008864 | 0.039579 | 0.490440 | 0.512428 |
| event_ic_model | as_is | -0.018596 | -0.020165 | 0.426386 | 0.428298 |
| event_ic_model | disabled | -0.018596 | -0.020165 | 0.426386 | 0.428298 |
| event_ic_model | unconditional_reversal | -0.019923 | 0.003427 | 0.452199 | 0.455067 |
| event_signal | as_is | -0.011741 | -0.017282 | 0.411090 | 0.422562 |
| event_signal | disabled | -0.011741 | -0.017282 | 0.411090 | 0.422562 |
| event_signal | unconditional_reversal | -0.017973 | -0.015853 | 0.402486 | 0.414914 |
| ic_sign_model | as_is | 0.020012 | 0.045178 | 0.472275 | 0.525813 |
| ic_sign_model | disabled | 0.020012 | 0.045178 | 0.472275 | 0.525813 |
| ic_sign_model | unconditional_reversal | 0.013343 | 0.042407 | 0.461759 | 0.508604 |
| lasso_model | as_is | -0.011086 | 0.026566 | 0.387189 | 0.430210 |
| lasso_model | disabled | -0.011086 | 0.026566 | 0.387189 | 0.430210 |
| lasso_model | unconditional_reversal | -0.017733 | 0.032659 | 0.432122 | 0.485660 |
| overnight_session_specialist | as_is | 0.018576 | 0.033818 | 0.445626 | 0.536643 |
| overnight_session_specialist | disabled | 0.018576 | 0.033818 | 0.445626 | 0.536643 |
| overnight_session_specialist | unconditional_reversal | 0.004215 | 0.024629 | 0.423168 | 0.513002 |
| reversal_rules | as_is | -0.017753 | 0.019546 | 0.473231 | 0.485660 |
| reversal_rules | disabled | -0.017753 | 0.019546 | 0.473231 | 0.485660 |
| reversal_rules | unconditional_reversal | 0.017846 | 0.065649 | 0.521989 | 0.548757 |
| ridge_adaptive | as_is | -0.010593 | 0.010212 | 0.432122 | 0.485660 |
| ridge_adaptive | disabled | -0.010593 | 0.010212 | 0.432122 | 0.485660 |
| ridge_adaptive | unconditional_reversal | -0.029435 | 0.002507 | 0.441683 | 0.476099 |
| ridge_model | as_is | -0.012219 | 0.017812 | 0.417782 | 0.462715 |
| ridge_model | disabled | -0.012219 | 0.017812 | 0.417782 | 0.462715 |
| ridge_model | unconditional_reversal | -0.028378 | -0.001783 | 0.413002 | 0.443595 |
| signal_proxy | as_is | 0.012343 | 0.063931 | 0.526769 | 0.554493 |
| signal_proxy | disabled | 0.012343 | 0.063931 | 0.526769 | 0.554493 |
| signal_proxy | unconditional_reversal | -0.008864 | 0.039579 | 0.490440 | 0.512428 |
| structure_factor_signal | as_is | -0.026590 | -0.012929 | 0.442639 | 0.453155 |
| structure_factor_signal | disabled | -0.026590 | -0.012929 | 0.442639 | 0.453155 |
| structure_factor_signal | unconditional_reversal | 0.022207 | -0.000020 | 0.493308 | 0.510516 |
| xgboost_model | as_is | 0.009185 | 0.025754 | 0.466539 | 0.502868 |
| xgboost_model | disabled | 0.009185 | 0.025754 | 0.466539 | 0.502868 |
| xgboost_model | unconditional_reversal | 0.009169 | 0.033060 | 0.478011 | 0.521989 |
