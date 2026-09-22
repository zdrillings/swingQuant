# Regime-Gated Heuristic Production Mode — Build Spec

**Date:** 2026-09-21
**Builder:** Codex
**Owner:** Zachary Drillings
**Status:** build spec — implement exactly as written; ask rather than inventing where unspecified.

---

## 1. Purpose

The scan has been fail-closed for 26+ days because no shortlist champion passes the promotion gate. That is the gate doing its job — but the system's only component with statistically significant out-of-sample performance is the **heuristic scan era** (yardstick t-stats 2.9–4.8 across horizons, realized +$25k), whose losses concentrated in reversal regimes. The regime meter now detects those regimes in advance.

This build makes the production system: **regime meter as the exposure switch, sweep-validated heuristic slots as the selection engine, hard stops (already in place) plus vol-targeted sizing as the risk layer, and the ML shortlist as advisory.** The promotion gate's authority is unchanged: when a champion exists, the model path remains primary; this spec fills the no-champion state with a production-quality mode instead of nothing.

## 2. Exposure policy (regime switch)

`regime_gating.enforce: true` (owner flips on sign-off). Classification from the lagged meter (existing machinery, 20-session lag):

| classification | scan behavior |
|---|---|
| `trending` | heuristic path, full caps (total 6, per-slot 3, per-sector 3) |
| `neutral` | heuristic path, **reduced caps** (total 3, per-slot 1, per-sector 2) |
| `reversal` | **stand down** — zero picks, existing stand-down email/log path (unchanged from the regime-meter build) |
| unavailable/unknown | stand down (fail closed — never trade without a regime read) |

The stand-down path must continue to leave `promotion_failures.txt` and champion state untouched (regime ≠ gate failure).

## 3. Candidate source policy

- **Champion exists + gate passes** → model path, exactly as today (unchanged).
- **No champion** → heuristic path as the *production* source (not the 75% diagnostic cap): `evaluate_signal_gate` per active slot with the promoted slot configurations in `production_strategies.json`.
- **Healthcare slot**: its `indicators` are empty and it can never pass the heuristic gate. Exclude `healthcare` from the heuristic-production slot set until its indicators exist; do not modify its configuration in this build.
- The ML live predictions, when they exist, render in the evening brief as an advisory list (no selection authority in this mode). If they do not exist, the brief simply has no advisory section.

## 4. Sizing (vol-targeted)

New logic in `src/utils/sizing.py`, applied to heuristic-mode candidates only (model-path sizing unchanged):

- **Per-position vol scale**: `multiplier = clip(vol_target_daily_pct / atr_pct_14, vol_scale_floor, 1.0)` using the snapshot's `atr_pct_14` (already persisted). Positions in high-ATR names shrink; nothing is levered up (ceiling 1.0).
- **Portfolio stop-risk cap**: after selection, compute each candidate's stop-risk contribution (`RISK_PER_TRADE` per position, as today) and require `sum(stop_risk) <= portfolio_stop_risk_cap_pct × capital`; if the sum exceeds the cap, drop the weakest candidates (lowest signal score) until it fits. Never silently re-size an existing sized position below one share — drop instead.
- Config:

```yaml
sizing:
  vol_target_daily_pct: 0.025
  vol_scale_floor: 0.50
  portfolio_stop_risk_cap_pct: 0.04
```

Defaults above are starting points; they are config, not constants.

## 5. Visibility

- Evening brief header gains an explicit mode line: `mode: MODEL | HEURISTIC | REDUCED-HEURISTIC | REGIME STAND-DOWN` plus the regime line already present.
- `scan_performance.md` header gains the same mode line per run (persist it on the scan candidate rows if the existing schema has a free text column — otherwise log only).
- Each heuristic-mode pick's sizing multiplier is logged so post-hoc performance can separate selection from sizing.

## 6. Success criteria (the mode earns its keep, or it doesn't)

Evaluated monthly from `scan_performance.md` + the ledger, on matured outcomes:

1. **Zero picks in reversal** — the stand-down holds without exception.
2. Over its first full trending-regime window: net-of-cost realized alpha ≥ 0 vs sector, and max drawdown ≤ half the un-gated historical average for comparable windows (yardsticks section gives the historical reference).
3. Concentration: no single ticker > 20% of total stop-risk contribution in any run (the portfolio cap should enforce this; report it).

If criterion 2 fails after one trending window, the owner re-evaluates before another window — this mode is evidence-gated, not aspirational.

## 7. Tests (suite currently 325)

1. Mode matrix: champion×regime → model/stand-down/reduced/full combinations all correct; unknown regime → stand down.
2. Heuristic-primary: no champion + trending → heuristic candidates at full caps; healthcare excluded; signal gates enforced per slot.
3. Neutral: caps 3/1/2; reversal: zero picks, `promotion_failures.txt` untouched, stand-down email sent.
4. Sizing: vol multiplier math (floor/ceiling); portfolio cap drops weakest by signal score; one-share floor respected.
5. Brief/report mode lines rendered in all four states.
6. Existing scan/monitor tests stay green; full suite + `compileall`.

## 8. Non-goals

- No ML/model changes. No new slots or slot configuration changes (healthcare exclusion only).
- No changes to the promotion gate, the regime meter, or the pipeline ordering.
- No automatic trade closure anywhere (`sq monitor` stays alert-only; `sq sell` remains the only closer).
- This mode never overrides a promoted champion: the model path wins whenever the gate says it may.

## 9. Activation sequence (owner, not Codex)

1. Build + tests green.
2. `regime_gating.enforce: true` in config.
3. First nightly run in `trending` or `neutral` reviews the picks in the brief; the owner decides whether to execute via `sq trade buy` (the scan still only recommends; nothing auto-executes).
