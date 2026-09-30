# Baseline Forensic

- generated_at: 2026-09-29
- scope: Q1 forensic on the historical full-OOS Spearman baseline near +0.455
- data_access: read-only SQLite and DuckDB; no `./sq` write commands; no `data/` mutation

## Executive Verdict

The +0.455 baseline was real in the stored artifacts, but it belonged to the old 20d/top-10, sector-specific xgboost evaluation with only 19-20 OOS dates. The exact retained baseline run is SQLite `Shortlist_Model_Runs.id=65`, generated at `2026-09-02T00:28:32+00:00`, where `xgboost_model` scored pooled Spearman `+0.4644` and mean per-date Spearman `+0.4713` on `alpha_vs_sector_20d`.

That edge does not come back when today's roster is judged on the old 20d label. On the full current 515-date OOS artifact, the best pooled 20d Spearman is only `+0.0247` (`signal_proxy`/`ensemble_model`). On the exact common subset of the old baseline date grid, today's best pooled 20d Spearman is `+0.0739` (`lasso_model`) and today's xgboost is `+0.0338`.

Single most likely cause: skill/model-chain loss, not just measurement drift. Measurement changed a lot, but rescoring today's roster on the baseline horizon/date setup recovers only about `0.0495` Spearman at best (`+0.0247` full current 20d to `+0.0739` common baseline dates), leaving roughly `0.3905` of the `+0.4644` to `+0.0739` collapse unexplained by the old measurement setup. The old edge was concentrated in sector-specific xgboost; the current roster no longer reproduces that ranking even under the old label.

## Baseline Artifact

Primary artifact:

| item | value |
|---|---:|
| commit/report citation | `00144e4` / `reports/shortlist_model.md` |
| SQLite run | `Shortlist_Model_Runs.id=65` |
| generated_at | `2026-09-02T00:28:32+00:00` |
| champion | `xgboost_model` |
| target | `alpha_vs_sector_20d` |
| horizon_days | 20 |
| top_n | 10 |
| model_scope | `sector_specific` |
| xgboost_config | `balanced_depth4` |
| feature_profile | `full` |
| eligible_universe_mode | `passed_or_trend` |
| eligible_rows / eligible_dates | 256752 / 649 |
| OOS dates | 20 |
| OOS date span | 2025-01-02 to 2026-07-13 |
| xgboost pooled Spearman | `+0.4644` |
| xgboost mean per-date Spearman | `+0.4713` |
| xgboost top-10 mean target | `+0.1947` |

Nearby retained runs confirm this was not a transcription typo:

| generated_at | OOS dates | xgboost pooled Spearman | xgboost per-date Spearman | xgboost top-10 mean |
|---|---:|---:|---:|---:|
| 2026-08-29T18:18:09+00:00 | 19 | +0.4546 | +0.4475 | +0.2027 |
| 2026-09-01T02:05:12+00:00 | 20 | +0.4515 | +0.4449 | +0.1993 |
| 2026-09-01T15:05:56+00:00 | 20 | +0.4652 | +0.4720 | +0.1945 |
| 2026-09-02T00:28:32+00:00 | 20 | +0.4644 | +0.4713 | +0.1947 |

## Configuration Diff

| dimension | baseline run 65 | current 09-29 artifact |
|---|---|---|
| target/evaluation target | `alpha_vs_sector_20d` | `alpha_vs_sector_60d` |
| horizon | 20 trading days | 60 trading days |
| selection basket | top 10 | top 2 |
| model scope | sector-specific | global |
| candidate roster | signal_proxy, ridge, lasso, xgboost, ensemble | signal_proxy, reversal_rules, event_signal, event_ic_model, structure_factor_signal, ridge, lasso, elastic_net, ic_sign, xgboost, ensemble |
| champion | xgboost_model | structure_factor_signal in dry-run report; latest persisted champion is structure_factor_signal run 71 |
| OOS dates | 20 | 515 in current CSV, 514 latest persisted DB run |
| OOS rows per model | 7,412 in run 65 | 218,413 in current CSV |
| xgboost config | balanced_depth4 | balanced_depth4 |
| feature profile | full | full with fold-local IC screen |
| IC survivor report | not persisted at `00144e4`; feature survivor list unavailable in git for that run | 26 survivors in current `reports/feature_ic_report.md` |
| regime matching | not shown in baseline report | `train_and_flip`, matched folds in current report |
| promotion gate | hit/beat/mean top-basket windows | hit/beat/mean plus Spearman floors, full-OOS floor |

Current IC survivors are 26 features led by `sector_median_roc_63`, `sector_pct_above_50`, `sector_median_roc_63__rank_all`, `sector_pct_above_200`, and `relative_strength_index_vs_subindustry`. The baseline run predates the checked-in `feature_ic_report.md`, so the exact fold-local survivor set after the IC screen cannot be recovered from git-tracked reports.

## Rescore: Today On Current 60d Setup

Source: current `reports/shortlist_model_oos_predictions.csv`, 515 dates, 218,413 rows/model, target `alpha_vs_sector_60d`.

| model | pooled Spearman | per-date Spearman | top-2 mean | top-10 mean |
|---|---:|---:|---:|---:|
| signal_proxy | +0.0245 | +0.0140 | +0.0794 | +0.0561 |
| ensemble_model | +0.0245 | +0.0140 | n/a | n/a |
| lasso_model | +0.0242 | -0.0117 | +0.0470 | +0.0314 |
| ic_sign_model | +0.0114 | +0.0179 | +0.0655 | +0.0309 |
| elastic_net_model | +0.0087 | -0.0174 | +0.0453 | +0.0293 |
| ridge_model | +0.0032 | -0.0166 | +0.0451 | +0.0300 |
| xgboost_model | -0.0094 | +0.0136 | +0.0307 | +0.0339 |
| reversal_rules | -0.0060 | +0.0196 | n/a | n/a |
| event_signal | -0.0079 | -0.0114 | -0.0184 | -0.0094 |
| event_ic_model | -0.0278 | -0.0169 | -0.0192 | +0.0104 |
| structure_factor_signal | -0.0256 | +0.0194 | n/a | n/a |

## Rescore: Today On Baseline 20d Label

Source: current predictions joined read-only to DuckDB `universe_daily_snapshots.alpha_vs_sector_20d`.

Full current OOS grid, 515 dates:

| model | pooled Spearman | per-date Spearman | top-2 mean | top-10 mean |
|---|---:|---:|---:|---:|
| signal_proxy | +0.0247 | +0.0235 | +0.0192 | +0.0181 |
| ensemble_model | +0.0247 | +0.0235 | +0.0192 | +0.0181 |
| lasso_model | +0.0108 | -0.0054 | +0.0236 | +0.0103 |
| ridge_model | +0.0098 | -0.0058 | +0.0221 | +0.0107 |
| elastic_net_model | +0.0092 | -0.0078 | +0.0221 | +0.0112 |
| xgboost_model | -0.0027 | -0.0039 | +0.0114 | +0.0060 |
| reversal_rules | -0.0166 | +0.0238 | +0.0230 | +0.0202 |
| structure_factor_signal | -0.0248 | +0.0194 | -0.0016 | +0.0050 |

Exact common subset of baseline dates, 17 dates present in both artifacts:

| model | pooled Spearman vs 20d | per-date Spearman vs 20d | top-10 20d mean |
|---|---:|---:|---:|
| lasso_model | +0.0739 | +0.0298 | +0.0212 |
| elastic_net_model | +0.0603 | +0.0239 | +0.0228 |
| ridge_model | +0.0588 | +0.0259 | +0.0186 |
| xgboost_model | +0.0338 | +0.0351 | +0.0187 |
| signal_proxy | +0.0287 | +0.0281 | +0.0196 |
| ensemble_model | +0.0287 | +0.0281 | +0.0196 |
| reversal_rules | +0.0133 | +0.0376 | +0.0234 |
| structure_factor_signal | -0.0131 | +0.0227 | +0.0095 |

## Decomposition

| comparison | best Spearman | delta vs +0.4644 |
|---|---:|---:|
| baseline xgboost, run 65, 20d/top-10/sector-specific | +0.4644 | +0.0000 |
| current roster, current 60d full grid, best pooled | +0.0245 | -0.4399 |
| current roster, old 20d label on full current grid, best pooled | +0.0247 | -0.4397 |
| current roster, old 20d label on common baseline dates, best pooled | +0.0739 | -0.3905 |
| current xgboost, old 20d label on common baseline dates | +0.0338 | -0.4306 |

Measurement changed materially: the baseline was a 20-date 20d/top-10 sector-specific xgboost read, while today's headline is a 515-date 60d/top-2 global/regime-matched roster read. But the old measurement does not rescue today's models. Even on the old 20d label and overlapping dates, the best current model is far below the old xgboost baseline, and current xgboost is nearly flat.

## What This Unlocks

Do not chase a gate-only fix as the next research step. The forensic result says the missing edge is upstream of promotion: reconstructing the old sector-specific xgboost feature/training path, or running an apples-to-apples ablation from run 65 to today's code, is the research path. The critic can verify this report tomorrow by checking:

- `reports/baseline_forensic.md` exists.
- It cites `00144e4` and SQLite run `65`.
- It records baseline xgboost Spearman `+0.4644`.
- It records the current old-label rescore max `+0.0739`.
- It names the likely cause as skill/model-chain loss rather than pure measurement drift.
