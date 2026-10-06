# Synthesis Forward Iteration

- generated_at: 2026-10-06
- implemented_change: E1 niche specialists admitted to the promotion gate candidate roster
- data_access: read-only report synthesis; no `./sq` write command; no `data/` mutation
- baseline_report: reports/q3b_oos_deoverlap.md
- niche_report: reports/e1_niche_split.md
- a4_report: reports/a4_regime_interactions.md
- live_gate_reference: reports/shortlist_model.md from the 2026-10-05 nightly run

## Implementation

The production shortlist builder now aligns all candidate predictions, including `overnight_session_specialist` and `base_pattern_specialist`, to the common OOS grid before full-OOS summaries, acceptance windows, and `_choose_champion_model()`. The two specialists therefore participate in the same promotion gate as the legacy roster. Gate floors, top-2 basket policy, scan caps, confidence basket, and rotation exclusion were not changed.

## Honest-Grid Full-OOS Delta Table

| candidate | pre-tour full-OOS Spearman | post-tour/audit full-OOS Spearman | delta | source |
|---|---:|---:|---:|---|
| signal_proxy | +0.0207 | +0.0214 | +0.0007 | Q3b -> 10-05 gate |
| event_signal | +0.0238 | +0.0234 | -0.0004 | Q3b -> 10-05 gate |
| lasso_model | +0.0052 | +0.0055 | +0.0003 | Q3b -> 10-05 gate |
| ic_sign_model | +0.0049 | +0.0050 | +0.0001 | Q3b -> 10-05 gate |
| ridge_model | -0.0012 | +0.0171 | +0.0183 | Q3b -> E2 with A4 |
| elastic_net_model | -0.0015 | -0.0020 | -0.0005 | Q3b -> 10-05 gate |
| ensemble_model | -0.0061 | -0.0063 | -0.0002 | Q3b -> 10-05 gate |
| xgboost_model | -0.0142 | -0.0053 | +0.0089 | Q3b -> E2 with A4 |
| structure_factor_signal | -0.0168 | -0.0162 | +0.0006 | Q3b -> 10-05 gate |
| reversal_rules | -0.0190 | -0.0181 | +0.0009 | Q3b -> 10-05 gate |
| event_ic_model | -0.0345 | -0.0355 | -0.0010 | Q3b -> 10-05 gate |
| overnight_session_specialist | n/a | +0.0182 | n/a | E1 niche audit |
| base_pattern_specialist | n/a | -0.0099 | n/a | E1 niche audit |

## Acceptance Windows

| candidate | full-OOS Spearman | last-fold Spearman | trailing-3fold Spearman | on-paper gate verdict |
|---|---:|---:|---:|---|
| signal_proxy | +0.0214 | -0.0462 | -0.0893 | fail recency Spearman |
| event_signal | +0.0234 | -0.1066 | -0.0776 | fail recency Spearman |
| lasso_model | +0.0055 | +0.0913 | -0.0922 | fail trailing-3fold Spearman |
| ic_sign_model | +0.0050 | -0.0737 | -0.1144 | fail recency Spearman |
| ridge_model | +0.0171 | +0.0309* | -0.0823* | fail trailing-3fold Spearman unless A4 recency rerun improves |
| elastic_net_model | -0.0020 | +0.0512 | -0.0763 | fail full-OOS and trailing-3fold Spearman |
| ensemble_model | -0.0063 | +0.0730 | -0.1427 | fail full-OOS and trailing-3fold Spearman |
| xgboost_model | -0.0053 | -0.2398* | -0.1499* | fail recency Spearman |
| structure_factor_signal | -0.0162 | +0.0843 | +0.1330 | fail full-OOS plus hit/beat/mean floors |
| reversal_rules | -0.0181 | -0.0462 | -0.0893 | fail full-OOS and recency Spearman |
| event_ic_model | -0.0355 | -0.0684 | -0.1812 | fail full-OOS and recency Spearman |
| overnight_session_specialist | +0.0182 | -0.1751 | -0.0213 | fail recency Spearman |
| base_pattern_specialist | -0.0099 | -0.1194 | -0.1411 | fail full-OOS and recency Spearman |

`*` Ridge and xgboost full-OOS post-tour values come from the A4/E2 read-only audits; their listed recency windows are the latest production gate windows before the A4 production rerun. The nightly pipeline should recompute those windows with A4 active and the two niches admitted.

## Expected Tonight Gate Outcome

On paper, the roster is strictly broader but still likely fails closed. The best full-OOS candidates are `signal_proxy` (+0.0214), `event_signal` (+0.0234), `overnight_session_specialist` (+0.0182), and A4 `ridge_model` (+0.0171), but each currently lacks both positive last-fold and trailing-3fold Spearman evidence. `structure_factor_signal` remains the only positive-recency candidate, but its full-OOS Spearman and basket economics fail the gate.

## Verification Hook

Tomorrow's critic should verify that `reports/shortlist_model.md` lists `overnight_session_specialist` and `base_pattern_specialist` in `candidate_models`, the Recent Acceptance Windows include their `_full_oos`, `_last_fold`, and `_trailing_3folds` rows, and no promotion gate floor values changed in the diff.
