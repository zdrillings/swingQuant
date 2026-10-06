# P3-W Expected Tonight

- idea: P3-LIVE read-only expected promotion verdict for ridge_adaptive on the production grid
- data_access: DuckDB read_only=True; no `./sq` writes; no data/ mutation
- target_column: alpha_vs_sector_60d
- audit_model: ridge_adaptive
- model_scope: global
- eligible_universe_mode: passed_or_trend
- xgboost_config: balanced_depth4
- feature_profile: full
- regime_matching: train_and_flip
- min_train_dates: 252
- default_max_train_dates: 252
- ridge_adaptive_max_train_dates: 126
- test_window_dates: 20
- evaluation_stride_dates: 60
- label_horizon_dates: 60
- promotion_basket_size: 2
- eligible_rows: 345304
- eligible_dates: 832
- verdict: ridge_adaptive fails on paper (last_fold hit_rate_excess, trailing_3folds hit_rate_excess)

## Acceptance Windows

| model | rows | dates | full_oos_spearman | last_fold_spearman | trailing_3fold_spearman | hit_excess_last | beat_last | mean_excess_last |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ridge_adaptive | 6650 | 180 | +0.0472 | +0.1857 | +0.0451 | +0.0135 | +0.5000 | +0.0169 |
| base_pattern_specialist | 6650 | 180 | +0.0153 | -0.0612 | +0.0673 | +0.0385 | +0.4500 | -0.0119 |
| elastic_net_model | 6650 | 180 | -0.0113 | -0.1167 | -0.0556 | -0.0615 | +0.3500 | -0.0592 |
| ensemble_model | 6650 | 180 | +0.0048 | -0.1842 | -0.0044 | -0.0115 | +0.3000 | -0.0628 |
| event_ic_model | 6650 | 180 | -0.0247 | -0.1434 | -0.0384 | +0.0135 | +0.5000 | -0.0222 |
| event_signal | 6650 | 180 | +0.0235 | -0.0167 | +0.0300 | -0.0115 | +0.5500 | +0.0069 |
| ic_sign_model | 6650 | 180 | +0.0320 | -0.1660 | +0.0226 | -0.1615 | +0.2500 | -0.1035 |
| lasso_model | 6650 | 180 | +0.0134 | -0.1761 | +0.0097 | -0.1115 | +0.2000 | -0.0820 |
| overnight_session_specialist | 6650 | 180 | +0.0301 | -0.0950 | +0.0255 | +0.0135 | +0.3500 | -0.0540 |
| reversal_rules | 6650 | 180 | -0.0286 | -0.0732 | -0.0317 | -0.0115 | +0.5000 | -0.0449 |
| ridge_model | 6650 | 180 | +0.0032 | -0.0741 | -0.0531 | -0.0365 | +0.4000 | -0.0402 |
| signal_proxy | 6650 | 180 | +0.0243 | -0.0732 | +0.0557 | -0.0115 | +0.5000 | -0.0449 |
| structure_factor_signal | 6650 | 180 | -0.0068 | +0.0419 | -0.0790 | -0.0365 | +0.6000 | -0.0113 |
| xgboost_model | 6650 | 180 | +0.0024 | -0.1715 | +0.0108 | +0.0135 | +0.3000 | -0.0441 |

## Ridge Adaptive Gate Floors

| window | metric | value | floor | result |
|---|---|---:|---:|---|
| last_fold | hit_rate_excess | +0.0135 | >= +0.0200 | FAIL |
| last_fold | beat_universe_rate | +0.5000 | >= +0.5000 | PASS |
| last_fold | mean_target_excess | +0.0169 | >= +0.0000 | PASS |
| last_fold | spearman | +0.1857 | >= +0.0000 | PASS |
| last_fold | top_ticker_date_rate | +0.0500 | <= +0.4000 | PASS |
| trailing_3folds | hit_rate_excess | +0.0129 | >= +0.0200 | FAIL |
| trailing_3folds | beat_universe_rate | +0.6000 | >= +0.5000 | PASS |
| trailing_3folds | mean_target_excess | +0.0380 | >= +0.0000 | PASS |
| trailing_3folds | spearman | +0.0451 | >= +0.0000 | PASS |
| trailing_3folds | top_ticker_date_rate | +0.0500 | <= +0.4000 | PASS |
| full_oos | spearman | +0.0472 | >= +0.0000 | PASS |

## Verification Notes

- active_fold_windows: last_fold, trailing_3folds
- active_recent_windows: none
- active_full_window: full_oos
- critic_verify_tomorrow: compare this table to tonight's `reports/shortlist_model.md` acceptance windows; floor verdict should match within measurement noise.
