# Shortlist Tune

- target_column: alpha_vs_sector_20d
- top_n: 10
- eligible_universe_mode: passed_or_trend
- model_scope: sector_specific
- mode: full
- tuning_profile: focused
- ablation_profile: focused
- min_train_dates: 252
- test_window_dates: 20
- tuned_model_family: xgboost_model

## Tuned Winner

- experiment: balanced_depth4
- params: {"colsample_bytree": 0.9, "max_depth": 4, "min_child_weight": 3.0, "subsample": 0.9}
- full_mean_target: 0.081160
- full_beat_universe_rate: 0.753125
- recent_mean_target: 0.113561
- recent_beat_universe_rate: 0.916667

## XGBoost Parameter Grid

### balanced_depth4
- params: {"colsample_bytree": 0.9, "max_depth": 4, "min_child_weight": 3.0, "subsample": 0.9}
- full_mean_target: 0.081160
- full_beat_universe_rate: 0.753125
- full_hit_rate: 0.647298
- recent_mean_target: 0.113561
- recent_beat_universe_rate: 0.916667
- recent_hit_rate: 0.745000

### shallower_regularized
- params: {"max_depth": 3, "min_child_weight": 3.0, "reg_lambda": 2.0}
- full_mean_target: 0.072218
- full_beat_universe_rate: 0.728125
- full_hit_rate: 0.625423
- recent_mean_target: 0.107805
- recent_beat_universe_rate: 0.916667
- recent_hit_rate: 0.733333

### baseline
- params: {}
- full_mean_target: 0.078713
- full_beat_universe_rate: 0.718750
- full_hit_rate: 0.641048
- recent_mean_target: 0.111349
- recent_beat_universe_rate: 0.833333
- recent_hit_rate: 0.748333

## Feature Ablation

### full_features
- params: {"colsample_bytree": 0.9, "max_depth": 4, "min_child_weight": 3.0, "subsample": 0.9}
- full_mean_target: 0.081160
- full_beat_universe_rate: 0.753125
- full_hit_rate: 0.647298
- recent_mean_target: 0.113561
- recent_beat_universe_rate: 0.916667
- recent_hit_rate: 0.745000

### no_breadth
- params: {"colsample_bytree": 0.9, "max_depth": 4, "min_child_weight": 3.0, "subsample": 0.9}
- excluded_feature_group: breadth
- full_mean_target: 0.082318
- full_beat_universe_rate: 0.753125
- full_hit_rate: 0.657923
- recent_mean_target: 0.115673
- recent_beat_universe_rate: 0.900000
- recent_hit_rate: 0.768333

### no_earnings
- params: {"colsample_bytree": 0.9, "max_depth": 4, "min_child_weight": 3.0, "subsample": 0.9}
- excluded_feature_group: earnings
- full_mean_target: 0.073529
- full_beat_universe_rate: 0.725000
- full_hit_rate: 0.622923
- recent_mean_target: 0.119039
- recent_beat_universe_rate: 0.933333
- recent_hit_rate: 0.728333
