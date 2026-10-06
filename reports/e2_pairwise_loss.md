# E2 Pairwise Loss Bakeoff

- generated_at: 2026-10-06
- data_access: read-only DuckDB; no `./sq` writes; no `data/` mutation
- experiment: compare raw-alpha regression with xgboost `rank:pairwise` on same-date groups; ridge raw/rank-label rows are the linear comparison
- target_raw: alpha_vs_sector_60d
- pairwise_label: same-date target ordering, encoded as per-date rank relevance
- eligible_universe_mode: passed_or_trend
- model_scope: global
- min_train_dates: 252
- max_train_dates: 252
- test_window_dates: 20
- evaluation_stride_dates: 20
- label_horizon_dates: 60
- xgboost_config: balanced_depth4
- eligible_rows: 345304
- eligible_dates: 832
- audited_same_date_pairs: 78668222
- audited_pair_dates: 832
- verdict: pairwise wins (best xgboost pairwise delta +0.0155 cleared the +0.0050 bar)

## Honest-Grid Spearman

| feature_set | variant | honest_dates | honest_rows | pooled_raw_spearman | per_date_raw_spearman | delta_vs_xgboost_regression |
|---|---|---:|---:|---:|---:|---:|
| with_a4 | ridge_rank_label | 514 | 7896 | +0.0182 | -0.0120 | +0.0236 |
| with_a4 | ridge_regression | 514 | 7896 | +0.0171 | -0.0130 | +0.0224 |
| with_a4 | xgboost_pairwise | 514 | 7896 | -0.0053 | -0.0066 | +0.0000 |
| with_a4 | xgboost_regression | 514 | 7896 | -0.0053 | -0.0009 | +0.0000 |
| without_a4 | ridge_rank_label | 514 | 7896 | -0.0446 | +0.0259 | -0.0176 |
| without_a4 | ridge_regression | 514 | 7896 | -0.0193 | +0.0396 | +0.0077 |
| without_a4 | xgboost_pairwise | 514 | 7896 | -0.0115 | +0.0062 | +0.0155 |
| without_a4 | xgboost_regression | 514 | 7896 | -0.0270 | -0.0036 | +0.0000 |

## Wiring Plan

Pairwise cleared the research bar. Follow-on: add a gated `xgboost_pairwise` candidate to the shortlist roster, reuse the same fold-local IC screen and promotion gate, and keep regression xgboost as the control until a production dry run passes.

## Verdict

pairwise wins: best xgboost pairwise delta +0.0155 cleared the +0.0050 bar.
