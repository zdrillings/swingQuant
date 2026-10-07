## Regime Flip Attribution

- note: diagnostic only; variants are not auto-adopted and do not alter champion selection, OOS artifacts, or live predictions.
- variants: as_is, disabled, unconditional_reversal

| model | variant | full_oos_spearman | mean_target | hit_rate | beat_universe_rate |
|---|---|---:|---:|---:|---:|
| base_pattern_specialist | as_is | 0.009023 | 0.005094 | 0.427553 | 0.496437 |
| base_pattern_specialist | disabled | 0.009023 | 0.005094 | 0.427553 | 0.496437 |
| base_pattern_specialist | unconditional_reversal | 0.001254 | -0.004802 | 0.427553 | 0.475059 |
| elastic_net_model | as_is | -0.024080 | 0.012803 | 0.397862 | 0.432304 |
| elastic_net_model | disabled | -0.024080 | 0.012803 | 0.397862 | 0.432304 |
| elastic_net_model | unconditional_reversal | -0.032128 | 0.019133 | 0.440618 | 0.472684 |
| ensemble_model | as_is | 0.020951 | 0.077466 | 0.547506 | 0.577197 |
| ensemble_model | disabled | 0.020951 | 0.077466 | 0.547506 | 0.577197 |
| ensemble_model | unconditional_reversal | 0.003531 | 0.056558 | 0.515439 | 0.534442 |
| event_ic_model | as_is | -0.016323 | -0.027236 | 0.400238 | 0.394299 |
| event_ic_model | disabled | -0.016323 | -0.027236 | 0.400238 | 0.394299 |
| event_ic_model | unconditional_reversal | -0.013742 | 0.001497 | 0.422803 | 0.432304 |
| event_signal | as_is | -0.010095 | -0.020319 | 0.397862 | 0.406176 |
| event_signal | disabled | -0.010095 | -0.020319 | 0.397862 | 0.406176 |
| event_signal | unconditional_reversal | -0.019189 | -0.027271 | 0.375297 | 0.384798 |
| ic_sign_model | as_is | 0.018328 | 0.057971 | 0.489311 | 0.543943 |
| ic_sign_model | disabled | 0.018328 | 0.057971 | 0.489311 | 0.543943 |
| ic_sign_model | unconditional_reversal | 0.027035 | 0.052610 | 0.477435 | 0.524941 |
| lasso_model | as_is | -0.012292 | 0.040181 | 0.387173 | 0.429929 |
| lasso_model | disabled | -0.012292 | 0.040181 | 0.387173 | 0.429929 |
| lasso_model | unconditional_reversal | -0.016705 | 0.058178 | 0.464371 | 0.524941 |
| overnight_session_specialist | as_is | 0.019514 | 0.034071 | 0.445368 | 0.534442 |
| overnight_session_specialist | disabled | 0.019514 | 0.034071 | 0.445368 | 0.534442 |
| overnight_session_specialist | unconditional_reversal | 0.005085 | 0.024839 | 0.422803 | 0.510689 |
| reversal_rules | as_is | -0.014052 | 0.046436 | 0.511876 | 0.529691 |
| reversal_rules | disabled | -0.014052 | 0.046436 | 0.511876 | 0.529691 |
| reversal_rules | unconditional_reversal | 0.026824 | 0.076123 | 0.545131 | 0.567696 |
| ridge_adaptive | as_is | -0.024012 | 0.022670 | 0.446556 | 0.505938 |
| ridge_adaptive | disabled | -0.024012 | 0.022670 | 0.446556 | 0.505938 |
| ridge_adaptive | unconditional_reversal | -0.038722 | 0.021447 | 0.465558 | 0.527316 |
| ridge_model | as_is | -0.015208 | 0.017829 | 0.416865 | 0.470309 |
| ridge_model | disabled | -0.015208 | 0.017829 | 0.416865 | 0.470309 |
| ridge_model | unconditional_reversal | -0.027066 | 0.017366 | 0.441805 | 0.489311 |
| signal_proxy | as_is | 0.020951 | 0.077466 | 0.547506 | 0.577197 |
| signal_proxy | disabled | 0.020951 | 0.077466 | 0.547506 | 0.577197 |
| signal_proxy | unconditional_reversal | 0.003531 | 0.056558 | 0.515439 | 0.534442 |
| structure_factor_signal | as_is | -0.027917 | -0.009669 | 0.483373 | 0.470309 |
| structure_factor_signal | disabled | -0.027917 | -0.009669 | 0.483373 | 0.470309 |
| structure_factor_signal | unconditional_reversal | 0.003478 | -0.000829 | 0.516627 | 0.515439 |
| xgboost_model | as_is | 0.011399 | 0.036180 | 0.485748 | 0.520190 |
| xgboost_model | disabled | 0.011399 | 0.036180 | 0.485748 | 0.520190 |
| xgboost_model | unconditional_reversal | 0.016484 | 0.042155 | 0.488124 | 0.539192 |
