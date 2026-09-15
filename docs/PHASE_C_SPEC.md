# Phase C + Rules Baseline — Build Spec

**Date:** 2026-09-14
**Builder:** Codex
**Owner:** Zachary Drillings
**Status:** build spec — implement exactly as written; ask rather than inventing where unspecified.

---

## 1. What tonight's run proved (read first)

The 2026-09-14 pipeline run was the first true `train_and_flip` run. Two facts from `reports/shortlist_model.md`:

```
- regime_matching_mode: train_and_flip
- regime_matching_folds: attempted=180, matched=0, fallback=180, unknown=0
- surviving_features: 6
```

**D1 — the regime-matching predicate is unsatisfiable.** `_regime_matched_train_dates` (`shortlist_model_service.py:847`) keeps training dates whose regime equals the test window's majority regime, then requires `len(matched_dates) >= min_train_dates` (252). With `max_train_dates: 252` the training pool is the last 252 dates, and 252 dates span several regime episodes — so `matched` can never reach 252, and **every fold falls back**. The regime-adaptive training layer has never engaged. Tonight's run was the unconditional flip plus a rolling train cap, nothing more.

**D2 — the feature screen is regime-blind and starving.** `surviving_features: 6`, all sector-breadth/momentum. The fold-local IC screen (with the new observation-fraction filter) is computed on the generic training slice, so reversal features (`rsi_2`, `ret_1d`, `close_vs_20d_low` — already in the snapshots) are screened out in every fold, including reversal folds. The "flip" therefore inverts a momentum-only model — the cheap approximation, not the design.

Consequence: the flip neutralized the recent bleed (recent-20d windows ≈ 0% net, beat rates 60–65%, spearmans positive) but cannot pass the gate (hit rates 37–45% < 50%). That is the correct outcome of running the approximation. This spec replaces the approximation.

## 2. Path A — make regime-adaptive training real

### A1. Fix regime-matched training (D1)

Replace the per-fold matched-window logic in `_regime_matched_train_dates`:

- Build the training pool from **all eligible history with a known regime classification** (the meter covers ~1,195 dates; the pool is not capped at `max_train_dates` when mode is `train_only`/`train_and_flip`).
- Regime matching: keep pool dates whose regime equals the test window's majority regime (same predicate as today).
- **Floor: `min_regime_train_dates` (new config, default 120).** If matched pool ≥ floor → train on it (cap at `max_train_dates` most recent matched dates if larger than the cap). Else → fallback to the generic pool and count `fallback_folds`.
- Keep `unknown_folds` for when the test regime cannot be determined (no lagged meter coverage).
- Keep the stats counters and report rendering (already present); the acceptance for this build is **matched > 0 and fallback < 50% of attempted** on the next production run.

### A2. Regime-conditioned feature screen (D2)

In the walk-forward fold loop, the fold-local IC screen (`_feature_ic_survivors_from_frame`) must screen on the **regime-matched training slice**, not the generic slice:

- Determine the test window's regime (lagged, via the existing `_regime_classifications_by_prediction_date` machinery).
- `trending` or `neutral` folds: screen on the matched/trending pool (or generic pool when neutral).
- `reversal` folds: screen on the reversal-matched pool, **including** the reversal family columns that are currently excluded or starved. Concrete reversal family (all exist in `SNAPSHOT_FEATURE_COLUMNS`): `rsi_2`, `rsi_14`, `ret_1d`, `ret_5d`, `close_vs_20d_low`, `distance_above_20d_high`, `base_volume_dryup_ratio_20`, `distance_from_52w_high`, `sma_50_dist`, `roc_63`, `roc_126`.
- The screen must still be train-only per fold (never touch test rows — the existing leak guard stays).
- Keep `min_feature_ic` and the observation-fraction filter; the fraction filter applies within the screened slice.

**Acceptance:** on the next run, `surviving_features` for reversal folds includes at least two of (`rsi_2`, `ret_1d`, `close_vs_20d_low`), and the report renders per-regime survivor lists (add a small "surviving features by fold regime" summary to the report — counts per regime, plus the reversal list).

### A3. Keep the flip as the defined fallback

In `train_and_flip`, the score flip applies only when the fold falls back to the generic pool AND the lagged test regime is `reversal` (current behavior, `fallback_only=True`). After A1, fallback should be rare. Do not change flip semantics in this build.

### A4. Universe de-tilting in reversal

In `_build_matured_eligible_universe` (or its equivalent for the model frame), add a reversal-regime eligibility extension applied **per date using the lagged regime classification**:

- When the date's lagged regime is `reversal`, additionally admit names that fail the trend gate but pass a pullback gate: `sma_200_dist >= -0.10` (within 10% of the 200d), `roc_63 <= 0` (pulled back or flat), `md_volume_30d` above the existing liquidity floor (unchanged), `rsi_14 < 60`.
- Trending/neutral dates: eligibility unchanged.
- This widens the candidate set the model can pick *down* in reversal regimes; without it the reversal model can only rank among recent winners.

### A5. Config visibility

Move the mode into config (the code default is invisible today):

```yaml
scan_policy:
  shortlist_model:
    regime_matching: train_and_flip
    min_regime_train_dates: 120
```

`_load_regime_matching_mode` reads it from there; keep the code default as fallback but log a warning when the config key is missing. Pipeline/CLI unchanged (no new flags).

## 3. Path B — reversal rules baseline candidate

Add one new candidate model, `reversal_rules`, to the walk-forward bakeoff (no training, deterministic, evaluated like `signal_proxy`):

- **Trending/neutral dates** (lagged regime): rank by the existing momentum proxy — mean of cross-sectional percentile ranks of `relative_strength_index_vs_spy`, `roc_63`, `sma_200_dist`, `vol_alpha` (the `signal_proxy` formula).
- **Reversal dates** (lagged regime): rank by the mean of cross-sectional percentile ranks of:
  - `roc_63` **ascending** (lowest momentum ranks highest),
  - `rsi_14` **ascending**,
  - `close_vs_20d_low` **ascending** (`close_vs_20d_low` is the distance above the 20d low, so ascending rank puts names nearest their lows highest),
  - `sma_50_dist` **ascending** (furthest below the 50d ranks highest).
- Regime per date comes from the same lagged regime lookup used by the flip (20-session lag, no lookahead).
- Reports: appears in the Full Walk-Forward Evaluation, Recent Acceptance Windows, and the OOS CSV like every other candidate; participates in the promotion gate on identical terms.

Purpose: a clean, interpretable test of "is the reversal signal real enough to clear the gate under the same evaluation the ML models face."

## 4. Acceptance criteria (all measurable)

1. Next production run with `regime_matching: train_and_flip`: `matched_folds > 0` and `fallback_folds < 50%` of attempted.
2. Reversal folds' survivors include ≥2 of (`rsi_2`, `ret_1d`, `close_vs_20d_low`); the report shows per-regime survivor summaries.
3. `reversal_rules` appears in the report with populated full + recent window metrics.
4. No lookahead: for any date d, the regime used for training-pool matching, feature screening, flip, and universe extension is the meter row ≤ d − 20 sessions (extend the existing lag regression test to cover A4 and the rules baseline).
5. `python3 -m unittest discover -s tests` green (current: 304) with new tests for: matching-pool floor logic, regime-conditioned screen, reversal rules ranking (synthetic), universe extension in reversal, stats rendering.
6. Gate floors and scan behavior are untouched: a failing candidate still fails closed; `enforce` stays `false` (stand-down vs adaptive-model resolution is an owner decision, not this build's).

## 5. Decision date

Per the agreed plan: if by **2026-10-01** no candidate passes the promotion gate on dense evidence, the owner will move to the reduced system (Path C: regime-gated, high-conviction manual trading with vol-targeted sizing). This build is the last full cycle before that date; scope discipline matters more than completeness — if something in this spec conflicts with shipping by Sep 20, cut it and note it rather than slipping the date.

## 6. Non-goals

- No new data sources, no new feature families beyond the existing snapshot columns.
- No changes to the promotion gate floors, the scan runtime, sizing, or the nightly pipeline ordering (except the config addition).
- No changes to the regime meter itself (it is validated and stable).
- Do not build Path C pieces.
