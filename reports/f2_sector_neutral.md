# F2 Sector-Neutral Top-N Audit

- generated_at: 2026-10-06T14:48:40+00:00
- idea: F2 sector-neutral top-N selection
- data_access: read-only OOS artifact; no `./sq` write command; no `data/` mutation
- source_artifact: reports/shortlist_model_oos_predictions.csv
- target_column: alpha_vs_sector_60d
- top_n: 2
- horizon_deoverlap_days: 60
- raw_rows: 440746
- raw_dates: 520
- deoverlapped_rows: 15020
- deoverlapped_dates: 516
- requested_models: signal_proxy, ridge_model, overnight_session_specialist
- available_models: signal_proxy, ridge_model
- missing_models: overnight_session_specialist
- verdict: sector-neutral loses

## Four-Way Comparison

| model | selection_mode | dates | avg_pick_count | basket_mean_alpha | hit_rate | beat_rate | top_sector_share |
|---|---|---:|---:|---:|---:|---:|---:|
| ridge_model | global_top_n | 516 | +1.986434 | +0.004124 | +0.440891 | +0.455426 | +0.615310 |
| ridge_model | sector_neutral | 516 | +1.980620 | +0.004414 | +0.434109 | +0.441860 | +0.509690 |
| signal_proxy | global_top_n | 516 | +1.986434 | +0.023105 | +0.494186 | +0.527132 | +0.601744 |
| signal_proxy | sector_neutral | 516 | +1.980620 | +0.021397 | +0.487403 | +0.523256 | +0.509690 |

## Deltas

| model | delta_mean_alpha | delta_hit_rate | delta_beat_rate | delta_top_sector_share | verdict |
|---|---:|---:|---:|---:|---|
| signal_proxy | -0.001708 | -0.006783 | -0.003876 | -0.092054 | sector-neutral loses on money statistics |
| ridge_model | +0.000290 | -0.006783 | -0.013566 | -0.105620 | sector-neutral loses on money statistics |
| overnight_session_specialist | n/a | n/a | n/a | n/a | missing from current OOS artifact |

## Implementation Decision

Sector-neutral selection improves concentration, but it does not win or tie on the money statistics in the available honest-grid artifact. The production scan selection mode is therefore left unchanged (`global_top_n`).

## Artifact Divergence

The current `reports/shortlist_model_oos_predictions.csv` was generated before the E1 niche roster was persisted into the OOS artifact, so `overnight_session_specialist` is not present even though current code includes it in the candidate roster. Tomorrow's critic can re-run this audit after the nightly model artifact includes that ranker.
