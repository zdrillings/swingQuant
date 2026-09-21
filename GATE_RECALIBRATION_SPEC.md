# Promotion Gate Recalibration — Build Spec

**Date:** 2026-09-21
**Builder:** Codex
**Owner:** Zachary Drillings
**Status:** build spec — implement exactly as written; ask rather than inventing where unspecified.

---

## 1. Problem

The promotion gate floors were calibrated on the fixed-20d label, where the eligible universe's positive rate is **48.9%** — an absolute 50% hit floor meant "slightly better than a coin flip." The system now evaluates path-aware 60d labels, whose universe base rate is **42.8%** (measured 2026-09-21: mean +0.0042, median -0.0108), and path labels structurally depress hit rates: hard stops truncate losers at -5%, profit targets cap winners, and 10–20 session time limits cut trends short. An absolute 50% hit floor on a 43% base demands +7 points of skill where the original demanded +1. The gate can now fail structurally — not because there is no skill, but because the floor tests a base rate the label change moved.

Measured universe base rates (all matured rows, 2026-09-21):

| label | n | mean | positive rate |
|---|---:|---:|---:|
| `alpha_vs_sector_20d` (original calibration base) | 1,612,607 | +0.0018 | 48.9% |
| `alpha_vs_sector_60d` | 1,066,954 | -0.0063 | 44.1% |
| `path_alpha_vs_sector_60d` (current production target) | 829,845 | +0.0042 | **42.8%** |

## 2. Principle

The gate must test **skill relative to the same-window eligible universe**, not absolute performance in a distribution the label family changed. All absolute floors become excess floors; the gate consumes the excess version; absolute values remain printed for context.

This is a re-anchoring, not a loosening: the new floors reproduce the original *stringency intent* (≈ +1–2 points of excess over the universe base) on whatever label family is in use. State this in the report so it cannot be misread as a gate relaxation.

## 3. Changes

### 3.1 Evaluation summaries (`src/research/shortlist_model_service.py`)

In `_evaluate_predictions`, per date, alongside the existing `universe_mean_target`, compute `universe_hit_rate` = positive rate of the date's universe rows (the same `day_frame` rows used for the existing universe mean). The returned summary gains:

- `universe_hit_rate` (mean across dates),
- `hit_rate_excess` = `hit_rate` − `universe_hit_rate`,
- `mean_target_excess` = `mean_target` − `universe_mean_target`.

All rolling summaries (`_rolling_window_summaries`) and the gate windows inherit these automatically since they call `_evaluate_predictions`.

### 3.2 Config (`config.yaml`, `scan_policy.shortlist_model.promotion_gate`)

Replace, for each window key (`20d`, `60d`, `1fold`, `3fold`):

```yaml
# remove:
min_recent_{window}_hit_rate: 0.50
min_recent_{window}_mean_target: 0.0
# add:
min_recent_{window}_hit_rate_excess: 0.02
min_recent_{window}_mean_target_excess: 0.0
```

Keep unchanged: `min_recent_{window}_beat_universe_rate: 0.50` (already a relative metric — date-level picks mean vs universe mean) and `min_recent_{window}_spearman: 0.0`.

Default `hit_rate_excess: 0.02` = two points of excess hit over the same-window universe — equal or slightly stricter than the original +1.1 points over the fixed-20d base.

### 3.3 Gate logic

`_model_passes_promotion_gate` (model service): the hit floor now compares `summary.hit_rate_excess >= min_recent_{window}_hit_rate_excess`; the mean floor compares `summary.mean_target_excess >= min_recent_{window}_mean_target_excess`. Beat-universe and spearman logic unchanged.

**Fail closed on missing universe data:** if a window's `universe_hit_rate` or `universe_mean_target` is unavailable (NaN), that floor fails with a logged reason — never pass by default.

`src/utils/shortlist_runtime.py::_passes_runtime_promotion_gate`: must mirror the exact same semantics and config keys. First verify its current shape (commits `13a7b61`, `d75d3df` added hit/beat enforcement there); whatever metrics it enforces, convert absolute hit/mean floors to the same excess floors so the scan-side gate and the promotion gate cannot diverge.

### 3.4 Report rendering

In the Promotion Gate and Recent Acceptance Windows sections, print per window: `hit_rate`, `universe_hit_rate`, `hit_rate_excess`, `mean_target`, `universe_mean_target`, `mean_target_excess`, plus the existing beat/spearman lines. Add one header note: `gate_note: floors are excess-over-universe by design; calibrated 2026-09-21 to the original fixed-20d stringency intent.`

The `selected_model_gate_passed` decision must be reproducible from the printed excess columns alone.

## 4. Tests (suite currently 325)

1. Synthetic summaries: model at exactly the universe base (excess 0.00) fails the hit floor at default 0.02 and passes mean-excess 0.0 only when mean == universe mean; +0.03 hit excess passes; -0.01 mean excess fails.
2. Boundary: `hit_rate_excess == 0.02` passes (≥, not >) — match the existing `_finite_at_least` convention.
3. Missing universe rows → floor fails closed with logged reason (promotion gate and runtime gate).
4. Runtime gate mirrors the promotion gate: identical inputs → identical verdict.
5. Report renders all six lines per window; gate outcome reproducible from printed excess columns.
6. Existing gate tests updated from absolute to excess keys; full suite green; `compileall` clean.

## 5. Acceptance

1. Next production run's gate section shows excess metrics with the calibration note.
2. A candidate at the universe base rate fails visibly (excess ≈ 0), and the failure is attributable to the excess floors, not the label's base rate.
3. Promotion and runtime gates agree on the same inputs.
4. No changes to: the walk-forward, feature screening, regime matching/flip/purge, the meter, or scan candidate construction. The gate remains the sole promotion arbiter; this spec changes its units, not its authority.

## 6. Non-goals

- No threshold tuning beyond the recalibration defaults (do not adjust 0.02/0.0/0.50/0.0 without owner sign-off).
- No changes to how the gate consumes regime or transition diagnostics (they stay report-only).
- No scan, sizing, or pipeline changes.
