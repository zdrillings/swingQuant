# GRID-PIN Evaluation Grid

- generated: 2026-10-06
- idea: first PICKS-tier improvement from the 2026-10-06 critique
- verdict: implemented fixed label-calendar OOS evaluation keys for promotion-gate metrics

## Rule

Promotion-gate evaluation no longer intersects model prediction dates across the candidate roster. The gate now computes a deterministic set of non-overlapping `(ticker, snapshot_date)` keys from the matured label calendar, after the train/label embargo boundary, and filters every candidate to that fixed key set.

Candidate common-date alignment remains available as a diagnostic helper, but it is not consumed by the promotion gate.

## Tonight's Grid Sizes

Read-only audit against `data/market_data.duckdb`, using the nightly `passed_or_trend`, `min_train_dates=252`, `label_horizon_dates=60`, and `oos_stride_dates=20` context:

| run context | old artifact-dependent gate grid | fixed label-calendar grid |
|---|---:|---:|
| production endpoint label (`alpha_vs_sector_60d`) | 408 eval dates after common-grid + per-model de-overlap | 6,135 fixed ticker/date keys across 501 dates |
| path-target dry-run (`path_alpha_vs_sector_60d`) | 613 eval dates in the dry-run report | 3,778 fixed ticker/date keys across 570 dates |

## Verification

- Regression test: `test_fixed_oos_gate_grid_is_independent_of_candidate_subset`
- Property covered: two different candidate subsets produce identical rolling acceptance windows for a shared model when the model's predictions are unchanged.
