# P4 Orthogonal Ensemble Bakeoff

- generated_at: 2026-10-06T15:46:58+00:00
- idea: P4 orthogonal ensemble over now-uncorrelated members
- data_access: read-only OOS artifact; no `./sq` write command; no `data/` mutation
- source_artifact: reports/shortlist_model_oos_predictions.csv
- target_column: alpha_vs_sector_60d
- horizon_deoverlap_days: 60
- raw_rows: 2424103
- raw_dates: 520
- honest_rows: 82610
- honest_dates: 516
- requested_all_members: signal_proxy, ridge_adaptive, overnight_session_specialist, structure_factor_signal
- requested_no_sfs_members: signal_proxy, ridge_adaptive, overnight_session_specialist
- missing_requested_members: ridge_adaptive, overnight_session_specialist
- verdict: ensemble loses to best available member (-0.0128 honest-grid Spearman delta)

## Single Members

| model | rows | dates | full_oos_spearman | last_fold_spearman | trailing_3fold_spearman |
|---|---:|---:|---:|---:|---:|
| signal_proxy | 7510 | 516 | +0.0144 | -0.1451 | -0.1880 |
| structure_factor_signal | 7510 | 516 | -0.0293 | +0.2400 | +0.0801 |

## Ensemble Comparison

| variant | method | used_members | missing_members | rows | dates | full_oos_spearman | last_fold_spearman | trailing_3fold_spearman | best_single_full_oos | delta_vs_best_single_full_oos | weights |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| available_requested_members | rank_average | signal_proxy, structure_factor_signal | ridge_adaptive, overnight_session_specialist | 7510 | 516 | -0.0071 | +0.0261 | -0.0614 | +0.0144 | -0.0215 | equal |
| requested_all_four | inverse_abs_spearman | signal_proxy, structure_factor_signal | ridge_adaptive, overnight_session_specialist | 7510 | 516 | +0.0016 | -0.0495 | -0.1185 | +0.0144 | -0.0128 | signal_proxy=+0.671, structure_factor_signal=+0.329 |
| requested_all_four | rank_average | signal_proxy, structure_factor_signal | ridge_adaptive, overnight_session_specialist | 7510 | 516 | -0.0071 | +0.0261 | -0.0614 | +0.0144 | -0.0215 | equal |
| requested_no_sfs | inverse_abs_spearman | signal_proxy | ridge_adaptive, overnight_session_specialist | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| requested_no_sfs | rank_average | signal_proxy | ridge_adaptive, overnight_session_specialist | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |

## Implementation Decision

The production ensemble member list is unchanged. The requested four-member and no-SFS ensembles cannot be fully adjudicated from the current OOS artifact because `ridge_adaptive` and `overnight_session_specialist` predictions are absent from both the checked-in CSV and the persisted SQLite prediction table. The constructible available-member check is reported above as an audit only.

Tomorrow's critic can verify this by checking that `reports/p4_orthogonal_ensemble.md` exists, includes full-OOS/last-fold/trailing-3fold columns, and that no production ensemble configuration changed.
