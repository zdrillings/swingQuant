# SwingQuant Improvement Brief

**Date:** 2026-09-05
**Scope:** strategy + model performance improvement, informed by a deep dive into [paperswithbacktest/awesome-systematic-trading](https://github.com/paperswithbacktest/awesome-systematic-trading) and the replication record at [paperswithbacktest.com](https://paperswithbacktest.com)
**Status:** research brief — nothing in this document changes trading behavior; implementation decisions are for the owner.

---

## 1. Verdict

The system has two separate problems, and they must be fixed in this order:

**[S1] The evaluation machinery cannot support the decisions being made from it.** The promotion gate that froze the live scan for 16 days runs on **5 out-of-sample dates × 2 picks = 10 observations**, and the deciding fold's model predictions are not persisted anywhere — the gate's own evidence cannot be audited from artifacts. Verified in this research: the report is exactly reproducible for `signal_proxy` (match to 4 decimals), but for every ML model the report's numbers come from fold-time predictions that differ materially from the final-refit predictions stored in the OOS CSV (xgboost full-OOS: report -2.4% vs. +6.4% recomputed from the CSV+DB; xgboost last fold: -20.9% vs. -17.4%). The system is making go/no-go money decisions on evidence that is both statistically trivial (10 observations) and unreproducible.

**[S1] Part of the recent collapse is real, and the system's response to it is the wrong response.** The last 3 months of matured picks: mean 20d return **-2.95%**, hit rate **38.5%**, alpha vs sector **-4.03%**. The most recent 20d window: alpha **-6.04%**, positive-alpha rate **20%**. This is a classic momentum-crash pattern (Daniel-Moskowitz 2016): a momentum-tilted eligible universe (green regime, above-200d, positive 63d momentum) reverses violently. The system's answer — freeze the scan via a promotion gate evaluated on 1-3 dates — converts a regime problem into a statistical-decision problem. Momentum crashes are *the* documented failure mode of this strategy family; the literature answer is vol-scaling and crash conditioning (Barroso-Santa-Clara 2015), not freeze/thaw on noise.

**[S2] The feature set is not what the reports claim it is.** The feature IC report (2026-09-05) shows the entire analyst family (12 features × 3 transforms) with **0 observations** — the analyst snapshots persist (250 tickers on 2026-09-04) but the join into the feature matrix is broken. All four macro features (`spy_roc_20`, `spy_roc_5`, `spy_realized_vol_20`, `qqq_roc_20`) are all-NaN in the IC report. AGENTS.md already documents that the L1 model purges macro features as worthless. The "full feature profile" is neither full nor honest.

**[S2] Hard stops are OFF in production**, in direct violation of AGENTS.md non-negotiable rule 8 (`hard_stop_pct` present in ExitRules, backtest, monitor; hard stop checked before trailing stop). No `hard_stop_pct` exists in `production_strategy.json` or `production_strategies.json`. Position sizing anchors to the *trailing* stop (ATR-based), so a gap-down through the stop has unbounded realized loss. The live book shows the consequence: RKLB open at -33.2%, ASTS at -16.7% while the strategy's 20d time limit and trailing stop sat downstream of a missing hard stop.

**[S3] The historical "edge" is more fragile than the aggregate numbers suggest.** One name — SNDK — was picked 59 times with a mean 20d alpha of +35.5% and mean 60d alpha of +92.6%. SNDK's price history in the DB spans only 2025-02-13 → today (392 rows), so nearly all of its observed performance is one memory-cycle boom in one regime. Remove it (and a handful of similar concentrations) and the full-history alpha curve is materially thinner. The 42.9% of picks with `selection_source = unknown` means even the attribution substrate is incomplete.

**[S3] The system never computes a Sharpe ratio.** No Sharpe anywhere in `src/` (verified by grep). The replication record's core yardstick — `(1.96 / Sharpe)²` years to prove a strategy — cannot even be applied to this system's results. A strategy needs ~24 years of evidence at Sharpe 0.4, ~6 years at 0.8. The current promotion gate effectively asserts it can validate a model in 1-3 dates.

**What is working and should not be thrown away:** the data substrate (DuckDB point-in-time snapshots with matured labels, 598 eligible dates, 225,899 eligible rows), the walk-forward harness design, the fail-closed posture, the per-date scan-outcome telemetry (619 scan dates of outcomes), and the 2026-08-13 rank-calibration finding that ranks 3-10 inverted and only the top-2 basket carried signal. The problem is density of evidence and persistence of predictions — not the shape of the architecture.

---

## 2. How the system works today

```
sq sync (yfinance OHLCV, universe) ──► DuckDB historical_ohlcv
sq universe-backfill               ──► DuckDB universe_daily_snapshots (point-in-time features + matured forward outcomes)
sq analyst-snapshot                ──► DuckDB analyst_snapshots / analyst_revision_snapshots
sq shortlist-model                 ──► walk-forward bakeoff of 7 candidate models
                                       ├─ labels: alpha_vs_sector_20d (20d sector-relative alpha)
                                       ├─ features: 57 raw + rank_all/rank_sector transforms + sector dummies
                                       ├─ training: expanding window, 252 min dates, horizon-strided non-overlapping labels
                                       ├─ OOS eval: stride 60, 20d test window → 5 prediction dates total
                                       ├─ fold-local train-only rank-IC screen (min |IC| 0.03)
                                       └─ promotion gate → persists champion (or fails closed)
sq scan                             ──► model path (top-2 by predicted alpha, quality gate, rotation exclusion)
                                       └─ heuristic path (per-slot signal gates) — diagnostic fallback
sq scan-performance                 ──► matured outcome telemetry by horizon + portfolio ledger section
sq monitor                          ──► intraday alert digest (never closes trades)
```

Key policy facts (AGENTS.md): fail closed when no champion passes the gate; confidence metrics computed on the top-2 basket only (rank-calibration analysis showed ranks 3-10 inverted); quality gate scales picks 1-2 based on recent beat rate / mean target; missing forward alpha stays missing (never converted to negative labels); hard stops are non-negotiable (rule 8).

Current production state (2026-09-05): **no champion persisted — live scan has failed closed for 16 days.** The tuned model family is XGBoost (max_depth 3), which was the worst of the seven candidates on the last fold.

---

## 3. The evidence from the repo (verified against the DB)

### 3.1 Recent collapse (scan_performance.md, matured through 2026-08-07)

| Window | mean 20d return | hit rate | mean alpha vs sector | positive-alpha rate |
|---|---:|---:|---:|---:|
| Full (598 dates) | +5.37% | 59.7% | +4.00% | 57.9% |
| 1y | +9.81% | 67.5% | +7.83% | 65.7% |
| 3m | **-2.95%** | **38.5%** | **-4.03%** | **36.6%** |
| last 20d | **-2.50%** | **30.0%** | **-6.04%** | **20.0%** |

Recent matured-picks detail shows the collapse is broad, not isolated: even picks whose RS improved over the holding window had 0% hit rate (38 observations). This is regime-level hostility to momentum selection, not a bad batch of tickers.

### 3.2 Opportunity-score band inversion (full history, 3,521 obs)

| score band | pick share | mean 20d alpha |
|---|---:|---:|
| < 0.30 | 1.3% | -0.29% |
| 0.30-0.40 | 13.4% | +2.9% |
| **0.40-0.50** | **38.6%** | **+8.2%** |
| **≥ 0.50** | **46.7%** | **+0.79%** |

The most-selected band was the worst-performing band; the cap at 0.45 (commit 95ee61d) was added after the fact. The score as constructed does not order expected alpha at the top.

### 3.3 Concentration (scan_performance.md + DB)

- SNDK: 59 picks, mean 20d alpha +35.5%, mean 60d alpha +92.6% — price history only since 2025-02-13 (392 rows).
- Portfolio ledger: realized +$25,170 (6.8% RoC) offset by unrealized -$6,517 (-7.1%): RKLB -33.2%, ASTS -16.8% open; largest realized losers TTMI -24.9%, VSH -26.7%, SMCI -19.2%.
- `selection_source`: 56.6% shortlist_model, 42.9% unknown, 0.5% heuristic.

### 3.4 Feature-set truth (feature_ic_report.md, 2026-09-05)

- All 36 analyst-family features: **0 observations** (join broken; snapshots exist in DuckDB).
- All 12 macro features (`spy_*`, `qqq_*`): **all-NaN**.
- Three sector features share an identical IC of 0.144402 — cross-sectional structure dominating, not independent signals.
- The 58 "surviving" features survive a screen applied over the same OOS range the champion is evaluated on (see §5).

### 3.5 Reproducibility forensics (this research, OOS CSV + DuckDB)

- OOS CSV has 13,125 rows = 5 dates × 7 models × ~375 tickers. **All six per-model rank columns are populated only on `ensemble_model` rows** (1,875 each); every individual model's rows carry NULL ranks. Top-2 reconstructions by rank are impossible for the ML models.
- Reconstructing top-2 by `predicted_alpha` and joining DB outcomes: `signal_proxy` reproduces the report **exactly** (full-OOS mean +8.26% both ways; last fold -18.59% both ways). The ML models do not reproduce (fold-time predictions used in the report are not persisted; the CSV stores final-refit predictions).
- DB sanity checks performed: MU at $1,016.59 and SNDK at $1,740.00 on 2026-09-04 are **real** OHLCV rows, not corruption; the reported xgboost last-fold -20.9% is consistent in direction with the final-refit -17.4% on the same date; the report's fold-time and the CSV's refit predictions diverge enough to flip lasso's last-fold sign (+4.5% report vs. -3.2% refit).

---

## 4. What the literature says (deep dive: awesome-systematic-trading)

### 4.1 The replication record — the base rates that matter

From the list's own replication catalogue (4,843 papers coded and run over their full history):

- Median replication Sharpe **0.37**; **48% fail to clear t = 1.96** — half the published record cannot be distinguished from zero on its own sample.
- A strategy needs roughly `(1.96 / Sharpe)²` years of evidence: Sharpe 0.4 → ~24 years; Sharpe 0.8 → ~6 years.
- Median strategy beta to the S&P is +0.17; removing index exposure drops the median information ratio to 0.21. **A meaningful slice of published edge is beta, not skill.**
- Across 2,838 papers with pre/post-publication records: no measurable post-publication decay (≤0.2%/yr once the market period is controlled). The edge doesn't vanish because it was published — it vanishes because it was never there or was beta.

Implication for SwingQuant: the system's own bar should be *alpha vs sector after beta removal, on a Sharpe/t-stat basis, with enough dates to matter*. The scan-performance telemetry already has 598 matured scan dates — the substrate for a proper yardstick exists; it just isn't being used that way.

### 4.2 Replicated strategies that are directly relevant to this system

| Strategy (from the list's 61 strongest of 1,687) | Sharpe | t | Relevance to SwingQuant |
|---|---:|---:|---|
| Understanding Momentum and Reversal (equities) | 1.36 | 8.2 | The system *is* a momentum+RS strategy; cross-sectional momentum works but is regime-conditional |
| Analytical Solution for Kelly's Criterion (equities) | 1.31 | 7.9 | Sizing is currently uniform-per-risk; Kelly-fraction sizing is the evidenced alternative |
| Any role for mean reversion in short term asset allocation | 1.30 | 7.4 | Short-horizon mean reversion complements the momentum core (the system already carries rsi_2/ret_1d features) |
| How to Improve Commodity Momentum Using Intra-Market Correlation | 0.65 | 2.8 | Cross-name correlation information helps momentum selection (sector breadth is already in the feature set) |
| Regime-based portfolio optimisation (HMM, fixed income) | 0.62 | 3.6 | Regime-conditioning as a model, not a binary green/red flag |
| When Factor Timing Makes Sense | 0.74 | 4.5 | Caution: factor timing rarely pays — prefer conditioning exposure, not switching strategies |
| Rolling vs. Expanding Windows in Mean-Reversion Strategies | 0.36 | 2.2 | Parameter instability is real even in replicated classics; expect decay, test it |
| Multi-Timeframe Trend Strategy on Bitcoin | 3.39 | 16.2 | Multi-timeframe confirmation has replicated edge in trend contexts (the system has 5/20/63/126d features but no explicit timeframe agreement feature) |

### 4.3 The deflation toolkit (from their course material; Bailey & López de Prado 2014; AFML 2018)

- **Deflated Sharpe Ratio (DSR):** corrects an observed Sharpe for the number of trials N. Expected max Sharpe under the null ≈ `√(2·ln N)`; testing 200 configs means an observed Sharpe below ~3.25 is what you'd get by luck. DSR > 0.95 = the best config's Sharpe is probably real.
- **Probability of Backtest Overfitting (PBO) via Combinatorial Purged CV:** PBO > 0.5 → more likely overfit than not; promote only below ~0.3.
- **Purged k-fold CV with embargo:** removes train/test overlap (purge) and the label horizon bleed (embargo). The shortlist harness has an embargo but no purge, and its fold-local IC screen leaks the OOS range into feature selection.
- Practical thresholds they teach: DSR > 0.95, PBO < 0.30, IS/OOS gap < 30%, Sharpe decay < 50%. **Log every configuration tested — N is an input to DSR.** The system currently logs nothing about the sweep of feature profiles, thresholds, and gates that produced the current configuration.
- Overlapping-return evaluation needs autocorrelation-consistent inference (Newey-West or the de Prado correction), not iid t-stats.

### 4.4 Books in the list that map onto this codebase

- **Advances in Financial Machine Learning (López de Prado):** triple-barrier labeling, meta-labeling (train a second model on "do I take this position and at what size"), sample uniqueness, CV for financial data. Directly applicable to §6 Phase 2.
- **Machine Learning for Asset Managers (López de Prado):** entropy-based feature selection, clustering-based NCO allocation. Relevant if Phase 3 portfolio construction is reached.
- **Machine Learning for Trading (Jansen):** the end-to-end reference pipeline this repo mirrors; its alpha-factory/factor-analysis chapters match the factor-tearsheet tooling already here.
- **Quantitative Momentum (Gray & Vogel):** momentum selection hygiene — universe, rebalance, seasoning/skip rules to avoid the 1-month reversal.
- **Systematic Trading (Carver):** forecasting/vol-targeting/sizing decomposition; the vol-targeting half is the missing risk layer in `sizing.py`.

---

## 5. Diagnosis: what is broken and what fixes it

| # | Severity | Finding | Evidence | Fix (phase) |
|---|---|---|---|---|
| 1 | S1 | OOS evaluation is 5 dates × 2 picks = 10 observations; gate runs on 1-3 of them | shortlist_model.md; AGENTS.md says stride 20, pipeline uses 60 | P1.1 daily overlap-aware evaluation |
| 2 | S1 | Gate evidence is not persisted (fold-time predictions vs. CSV refit predictions diverge; rank columns misplaced onto ensemble rows) | CSV forensics §3.5 | P0.1, P0.2 |
| 3 | S1 | Recent momentum-crash regime: 3m hit rate 38.5%, alpha -4.03%; system response is gate-freeze on noise | scan_performance.md §3.1 | P2.4 crash conditioning / vol scaling |
| 4 | S2 | Analyst features dead in the feature matrix (0 obs) despite snapshots persisting | feature_ic_report.md + analyst_snapshots.md | P0.3 |
| 5 | S2 | Hard stops absent from production strategies — AGENTS.md rule 8 violated; sizing anchors to trailing stop | production_strategies.json | P0.5 |
| 6 | S2 | Fold-local IC screen computed over the full OOS range → feature selection leaks evaluation labels | shortlist_model_service.py:130-141, 552-564 | P1.2 |
| 7 | S2 | Promotion gate config drift: hit/beat/mean_target floors configured but never enforced in code (only Spearman) | shortlist_model_service.py:1911-1938; shortlist_runtime.py:275-290 | P1.3 |
| 8 | S3 | No Sharpe/DSR/t-stat anywhere; costs absent from shortlist-model evaluation | grep; sweep has costs, model eval doesn't | P0.4, P1.4 |
| 9 | S3 | Opportunity score top band historically worst; cap added post-hoc; 46.7% of picks were in that band | §3.2 | P1.1 (re-evaluate under dense evidence), P2.3 |
| 10 | S3 | Concentration: SNDK 59 picks / 392 days of history dominates the winner stats; repeated-winner stats are overlapping observations | §3.3 | P0.6 |
| 11 | S3 | Point-in-time holes: earnings events limited to latest 24/ticker, analyst capture starts 2026-06, md_volume_30d falls back to current, row-positional labels vs calendar dates | Explore agent findings 5-6 | P1.5 (document; fix labels first) |

---

## 6. The plan

Principles: lean on the existing structure (DuckDB substrate, snapshot pipeline, walk-forward harness, scan runtime, report command surface). Change the density and honesty of evidence first; change models second; change architecture only if Phases 0-2 prove it necessary. Every phase has acceptance criteria that are measured, not hoped.

### Phase 0 — Repair the evidence (days; no model changes)

**P0.1 Make `shortlist_model.md` reproducible from persisted artifacts.**
Write fold-time OOS predictions (per model, per fold date, with ranks) to the OOS CSV; include raw `alpha_vs_sector_20d` alongside the binary; ensure each model's rank column lands on its own rows.
*Where:* `src/research/shortlist_model_service.py` (CSV writer + report generation).
*Done when:* a script recomputes every acceptance-window number in the report from CSV + DuckDB within float tolerance, for all 7 candidates.

**P0.2 Kill the gate on unreproducible evidence.**
Until P0.1 lands, treat promotion decisions as advisory and keep the scan failed closed (status quo). Do not "relax the gate" (config escape hatch) to un-freeze production — that would trade on evidence that provably can't be audited.
*Done when:* gate inputs are the persisted fold-time predictions.

**P0.3 Fix the analyst-feature join or cut the family.**
Either repair the snapshot→feature-matrix join (and confirm non-zero observations in the next IC report) or remove the analyst columns from `MODEL_FEATURE_COLUMNS` / `SNAPSHOT_FEATURE_COLUMNS` so the "full" profile stops lying. Same treatment for the all-NaN macro features — AGENTS.md already documents the L1 model purges them; removing them from the matrix also removes a whole failure mode.
*Where:* `src/research/shortlist_bakeoff_service.py` (MODEL_FEATURE_COLUMNS), `src/research/universe_snapshot_service.py` (SNAPSHOT_FEATURE_COLUMNS, analyst join).
*Done when:* next IC report shows every advertised feature family with non-trivial observations, or the family is gone from the model matrix.

**P0.4 Add the missing yardsticks.**
Compute, in both `shortlist-model` and `scan-performance` reports: annualized Sharpe of the date-level top-2 basket (with the configured cost model), t-stat (Newey-West lag = label horizon), `(1.96/Sharpe)²` years required, and beta vs SPY with the beta-removed alpha. This is the replication record's language; adopt it verbatim.
*Where:* `src/research/shortlist_model_service.py`, `src/scan/performance_service.py`, `src/utils/` (new `performance_metrics.py`).
*Done when:* the nightly report states the Sharpe and the years-of-evidence number next to every promotion decision.

**P0.5 Restore hard stops in production strategies (AGENTS.md rule 8).**
Add `hard_stop_pct` to every active slot in `production_strategies.json` consistent with each strategy's sweep history (sweep already simulates hard stops first in the exit chain — reuse those swept values, don't invent new ones). Confirm `sq monitor` must-sell classification and backtest priority.
*Where:* `production_strategies.json`, `src/promote/service.py`.
*Done when:* every active slot has a hard stop sourced from its swept configuration; monitor digest flags hard-stop breaches as must-sell; tests pass.

**P0.6 Concentration and attribution hygiene.**
Add a concentration section to `scan-performance`: share of total realized/unrealized outcomes attributable to the top 3 tickers by pick count; report the SNDK-adjusted aggregate curves; backfill `selection_source` for the 42.9% unknown rows if the raw data exists, else mark the column as incomplete in the report.
*Done when:* the report shows the concentration-adjusted alpha curve alongside the raw one.

### Phase 1 — Make evaluation thick enough to decide anything (weeks)

**P1.1 Dense overlap-aware OOS evaluation.**
Evaluate the model on **every eligible date** (stride 1), not 5 stride-60 dates. Training labels stay horizon-strided and non-overlapping (that discipline is correct and is AGENTS.md rule-adjacent); *evaluation* may use overlapping daily predictions because the unit is a date-level basket outcome. Inference on overlapping 20d returns requires Newey-West (lag 20) or the de Prado overlapping-return correction — never iid t-stats. This converts 10 observations into ~350+ date-level observations overnight, using data already in DuckDB.
*Where:* `src/research/shortlist_model_service.py` (OOS loop), new inference helpers in `src/utils/`.
*Done when:* OOS section of the report covers ≥300 prediction dates with NW t-stats; promotion gate consumes monthly rollups of these (e.g., trailing 60-date windows) instead of 1-3 folds.

**P1.2 Purge the leakage and purge the CV.**
(a) Fold-local IC screens must use train-only slices (the current screen is computed over the full OOS range — verified). (b) Add a purged/embargoed CV path for hyperparameter choice (xgboost configs), with purge width ≥ label horizon.
*Where:* `shortlist_model_service.py` (feature screen, CV).
*Done when:* a regression test proves the selected feature set is invariant to adding/removing future dates' rows beyond the fold boundary.

**P1.3 Enforce the configured gate.**
Implement the hit_rate / beat_universe_rate / mean_target floors from `config.yaml` (currently loaded but never checked) — or delete them from config. A gate that exists on paper and not in code is worse than no gate.
*Where:* `shortlist_model_service.py:1911-1938`, `shortlist_runtime.py:275-290`.
*Done when:* every configured gate key is enforced or removed; test coverage asserts each one.

**P1.4 Costs in the model evaluation.**
Apply the configured slippage (5bps/side in `backtest_costs`) to the OOS basket P&L before any promotion metric is computed.
*Done when:* report states gross and net figures; gate consumes net.

**P1.5 Label-integrity audit.**
Confirm row-positional `index + horizon` labels equal calendar-based labels across the whole history (halt/delisting drift); fix or document. Keep the "missing alpha stays missing" rule untouched.
*Where:* `universe_snapshot_service.py:590-596`.
*Done when:* audit script reports zero material mismatches or the drift is quantified in the report.

**Phase 1 exit gate:** with the repaired harness, re-run the full candidate bakeoff and either promote a champion on dense evidence or keep failing closed. Do not change any model weights, features, or thresholds before this re-run — it is the control measurement.

### Phase 2 — Signal work informed by the literature (weeks-months, only after Phase 1)

**P2.1 Triple-barrier / path-aware labels (AFML ch. 3).**
The strategy's real trade exits are trailing stop / profit target / time limit — the fixed 20d alpha label only weakly proxies them. Add a label family that applies the *actual* ExitRules (ATR stop/target, 20d time limit) to each snapshot's forward path, producing realized-path outcomes. Evaluate the same seven candidates against both label families.
*Where:* `universe_snapshot_service.py` (label builder), sweep exit math reused from `sweep/service.py`.
*Done when:* a tearsheet comparing fixed-horizon vs path-aware labels shows which target the model can actually predict, per model family.

**P2.2 Meta-labeling for sizing (AFML ch. 3).**
Train a secondary classifier on "did the primary model's pick clear its stop-adjusted hurdle" and use it to size (full/half/zero). This replaces the current binary quality gate with a per-pick probability. Kelly-fraction (list: Kelly equities replication, Sharpe 1.31/t 7.9) caps the aggressive side.
*Where:* `src/research/` (new meta-label service), `src/scan/service.py` (sizing hook), `sizing.py`.
*Done when:* OOS comparison of flat-size vs meta-labeled-size baskets shows net-alpha improvement at equal max drawdown, on dense (P1.1) evidence.

**P2.3 Opportunity-score repair.**
The band analysis (§3.2) shows the score's top band was its worst. Rebuild `_score_candidate` weights on train-only band regressions under the dense harness, or replace the hand-weighted sum with the model's own calibrated probability (calibration is already computed; policy says it may not drive live ranking while fold evidence is stale — that prohibition expires once the dense harness makes fold evidence current).
*Where:* `scan/service.py:3211-3270`.
*Done when:* score-band → outcome curve is monotone non-decreasing on OOS, or the score is removed from selection.

**P2.4 Momentum-crash conditioning (Barroso & Santa-Clara 2015; Daniel & Moskowitz 2016).**
Scale aggregate exposure by inverse realized vol of the basket's sector/regime; add market-level drawdown/vol state as explicit conditioning features or as a multiplier on position count (the quality gate is already a crude version of this — make it continuous and evidence-based). The 3m collapse (§3.1) is the motivating dataset.
*Where:* `sizing.py`, `scan/service.py` quality gate, feature set (market-state features, replacing the all-NaN macro family with working ones).
*Done when:* backtested over the 2026-05→08 crash window, the conditioned version's max drawdown is ≤ half the unconditioned version's at ≥ equal net alpha.

**P2.5 Deflation gates on promotion.**
Report DSR (N = logged configuration count) and PBO via CPCV for the promoted candidate; promotion requires DSR > 0.95 and PBO < 0.3 (thresholds from the paperswithbacktest course material). Log every configuration tested — including feature profiles, thresholds, and gate settings — as the N input.
*Where:* promotion gate + a `config_trials` log.
*Done when:* the promotion section of the report shows N, DSR, PBO for the champion.

### Phase 3 — Structural rebuild (conditional; only if Phases 1-2 fail their bars)

Trigger: after Phase 2, the system cannot clear the REDESIGN_BRIEF bars (beat-excluded > 55%, mean 20d alpha > +2%, stable recent window) on dense evidence.

- **P3.1 Drop sector-scoped models for a single global cross-sectional ranker** with sector dummies as features (the data already supports it; sector-specific models currently fall back to full-train anyway when thin). This is the REDESIGN_BRIEF's original framing and removes a layer of fragmentation.
- **P3.2 Retarget the horizon by evidence.** Per-day alpha is stronger at 10d than 20d in the scan telemetry (10d: +2.81% mean alpha / 59.5% hit vs 20d: +4.00% / 59.7% — the 20d number is only 1.4x the 10d number for 2x the holding time). Test a 10d-label model family head-to-head before committing.
- **P3.3 Reconsider the exit layer** (rsi-exit-bakeoff, earnings overlay) only after the selection layer is settled — exit evidence is currently too sparse to optimize (REDESIGN_BRIEF conclusion still holds).
- **P3.4 If the system still cannot produce a tradable shortlist:** the honest outcome is a smaller, slower system (fewer, larger, regime-gated positions sized by vol-target) rather than a more complex one. The replication record says half the published field fails t=1.96 on its own sample; there is no shame in a system that admits it, and no edge in one that can't measure it.

---

## 7. What not to do

1. **Do not un-freeze production by relaxing the promotion gate** before P0.1/P0.2. Trading on unauditable evidence is worse than not trading.
2. **Do not add features or model families** before the Phase 1 control re-run — the current failure mode is measurement, not architecture; adding complexity to an unmeasured system compounds it.
3. **Do not tune anything on the recent 60-date window** — that window already selected the top-2 basket, the 0.35/-0.03 quality thresholds, and the 0.45 cap. Every additional decision made on it is the multiple-testing problem the DSR exists for.
4. **Do not widen the top-2 basket** without re-running the rank-calibration analysis (AGENTS.md rule 9).
5. **Do not chase the memory-cycle concentration** (SNDK/memory names) as "what works" — it is one regime, one sector, 392 days of history.
6. **Do not reintroduce calibration-driven live ranking** while fold evidence is stale (existing policy; expires only under the dense harness).
7. **Do not break AGENTS.md rules 1-11** while implementing any of the above — in particular rule 1 (no random splits), rule 8 (hard stops), rule 11 (fail closed; missing labels stay missing).

---

## 8. Immediate actions (this week)

1. P0.1+P0.2: persist fold-time predictions + fix rank-column placement in the OOS CSV; add a reproducibility check script. *(blocks everything)*
2. P0.5: restore `hard_stop_pct` per slot from sweep history. *(pure config+test, no research risk)*
3. P0.3: fix or cut the analyst/macro feature families. *(makes every future IC report honest)*
4. P0.4: land Sharpe/t-stat/`(1.96/SR)²` in both reports. *(the yardstick the whole plan is gated on)*
5. P1.1: dense overlap-aware evaluation with NW inference. *(turns 10 observations into 350+; enables every later gate)*

---

## 9. Sources

### The deep-dive source

- [paperswithbacktest/awesome-systematic-trading](https://github.com/paperswithbacktest/awesome-systematic-trading) — full README (libraries, strategies table with Sharpe/t-stat, books) read directly; replication-record statistics quoted in §4.1.
- [paperswithbacktest.com wiki](https://paperswithbacktest.com/wiki) — method/caveats index.
- [Deflated Sharpe Ratio course page](https://paperswithbacktest.com/course/deflated-sharpe-ratio) — DSR formulas and thresholds used in §4.3.
- [Backtesting Pitfalls: Overfitting and Selection Bias](https://paperswithbacktest.com/course/backtesting-pitfalls-overfitting) — CPCV/PBO framing.
- Bailey & López de Prado (2014), *The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality*, JPM 40(5) ([SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551)).
- López de Prado (2018), *Advances in Financial Machine Learning* (triple-barrier, meta-labeling, purged CV).
- Daniel & Moskowitz (2016), *Momentum Crashes*, JFE. Barroso & Santa-Clara (2015), *Momentum has its moments*, JFE.
- Grinold & Kahn, *Active Portfolio Management* (Fundamental Law: IR ≈ IC·√breadth).
- Gray & Vogel, *Quantitative Momentum*; Carver, *Systematic Trading*; Jansen, *Machine Learning for Trading* — all from the list's Books section.

### Repo evidence

- `reports/scan_performance.md` (2026-09-04), `reports/shortlist_model.md` (2026-09-05), `reports/shortlist_tune.md` (2026-09-04), `reports/feature_ic_report.md` (2026-09-05), `reports/analyst_snapshots.md` (2026-09-04), `reports/shortlist_model_oos_predictions.csv`, `REDESIGN_BRIEF.md`, `AGENTS.md`, `production_strategies.json`, `config.yaml`.
- DB verification performed directly against `data/market_data.duckdb` (OHLCV, `universe_daily_snapshots`) and `data/ledger.sqlite` — see §3.5.

### Verification notes (what was checked, per the house standard)

- MU at $1,016.59 (2026-09-04) and SNDK at $1,740.00: **real DB rows**, not corruption.
- The -20.9% xgboost last-fold figure: consistent in direction with an independent recompute (-17.4%) from final-refit predictions; the exact fold-time value is unverifiable because those predictions are not persisted — which is itself Finding 2.
- The lasso/ridge/elastic-net near-identical top-2 outcomes across dates: traced to NULL rank columns (alphabetical tie-breaking), not to model convergence. All rank columns are populated only on ensemble rows.
- The 3m collapse (-2.95% mean 20d return, 38.5% hit): directionally confirmed by the recent-scan-dates section and the RS-bucket table; no single corrupt row explains it.
