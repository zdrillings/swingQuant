# SwingQuant Operating Model

**Date:** 2026-09-23
**Status:** design document — the target the system is moving toward. Build items below are specs waiting to be written or built; existing specs are referenced, not restated.

---

## 1. The One Question

Everything in this system exists to answer one question, daily:

> **Is the regime favorable, and are the rules being followed?**

- *Regime favorable* → the decision layer allows picks (model or heuristic, per the production policy).
- *Rules being followed* → every position carries its exit rules, every exit records the rule that fired, every override is logged.
- When either answer is **no**, the system stands down, flags a violation, or both.

The edge — as the 2026-09 measurement work established — is regime-gated selection with mechanical exits. Nothing in this model exists to rediscover that; it exists to operate it and to measure it honestly.

## 2. Three Layers, Small Interfaces

### 2.1 Research layer (offline; never touches money)

The walk-forward bakeoff, the structure factor family, IC screens, path labels, tearsheets, phase2 diagnostics, regime meter research.

- **Product:** a promoted champion, or "no champion." Nothing else.
- **Failure mode:** wasted compute. Cheap and recoverable.
- **Policy:** research runs use `persist=False` and never touch champion state. Only the production pipeline promotes. A dirty working tree must never again freeze the pipeline for days — commit after each logical change (the 2026-09-11..13 outage).

### 2.2 Decision layer (nightly)

Regime meter + promotion gate + production policy (`REGIME_GATED_PRODUCTION_SPEC.md`).

- **Product:** "today the system emits N picks" or "stand down," with the reason.
- **Failure mode:** wrong exposure. Expensive.
- **Policy:** the gate is the sole promotion arbiter (excess-over-universe floors per `GATE_RECALIBRATION_SPEC.md`). The meter is the sole exposure switch. Neither may be overridden by hand at scan time.

### 2.3 Execution layer (hourly + human)

Monitor, exit rules, ledger, `sq trade`, the owner.

- **Product:** positions opened and closed per pre-agreed rules, with attribution.
- **Failure mode:** real losses.
- **Policy:** alert-only by rule (AGENTS.md rule 5). The human executes must-sells; nothing auto-closes.

### 2.4 The interfaces (and only these)

| from → to | interface |
|---|---|
| Research → Decision | regime meter classification (lagged, 20 sessions); champion + gate verdict |
| Decision → Execution | the nightly brief: mode, picks, slots, sizing, stop levels |
| Execution → Execution | exit-rule state of each open position, recomputed hourly by the monitor |
| Execution → Research | exit attribution: which rule fired, when, at what outcome |

If a layer starts consuming something other than these interfaces, that is a design change — discuss it, don't sneak it in.

## 3. The Operating Loop

### Nightly (19:05 pipeline)

ingest → meter refresh → research bakeoff (background role) → gate decision → scan → **one-page brief**.

The brief's entire job:

- mode line: `MODEL | HEURISTIC | REDUCED-HEURISTIC | REGIME STAND-DOWN`
- regime line: classification + mom_ic_20d
- picks (or the stand-down reason), each with slot, sizing, stop level
- nothing else

### Hourly (10:30–15:30 monitor)

- Recompute every open position's rule state against its stored `entry_atr` + `strategy_slot`.
- Emit must-sell alerts only for pre-agreed conditions (hard stop, profit target, pre-earnings, regime flip once wired).
- The human's daytime job: execute must-sells. Nothing else.

### Weekly

One review question: **did the system do what it said it would do?**

- rule violations counted as first-class metrics: positions with no rules, late must-sell executions, unlogged overrides
- trailing windows reviewed: gate acceptance, scan performance vs the brief

### Monthly

The success criteria from `REGIME_GATED_PRODUCTION_SPEC.md` §6 against realized outcomes, plus the capital-layer review (build item G5).

## 4. Gaps Between Today and This Model (build order)

**G1 — Exit attribution (Execution → Research interface).** Implemented foundation: closed ledger trades persist `exit_reason`, `sq trade sell` accepts `--exit-reason`, broker-inferred closes default to `manual`, and monitor recommendations render a matching suggested sell command. Monthly review still needs to compare the realized exit mix to the path-label tearsheet's predicted mix.

**G2 — Decision log (human interventions).** The owner is the largest untracked variable. One line per intervention: date, position, what the system said, what was done, why. Monthly review splits *system performance* from *human performance*; today they are inseparable. (The ZScaler 2026-09 trade — no slot linkage, no entry_atr, exit decided on sentiment — is the canonical first entry.)

**G3 — System status page.** One page replacing six reports: regime + gate state + mode + open positions with current rule state + tomorrow's expected behavior. Thirty seconds to read. Everything else feeds it. This is the highest-leverage missing piece.

**G4 — Monitor meter wiring.** Implemented: monitor reads the lagged regime meter and renders `regime_flip` as a must-sell when the current classification is `reversal`. The old sector-ETF red/green context remains contextual, not the production exit switch.

**G5 — Capital-layer review.** `RISK_PER_TRADE`, portfolio stop-risk cap, and vol targets were set once. Monthly: do they still match the strategy's expectancy evidence? Reviewed against realized outcomes, not backtests.

**G6 — Per-slot hard stops.** The uniform 5% floor is the least-justified parameter in the exit stack (flagged since P0.5). Calibrate per-slot hard stops from sweep history.

## 5. Roles

- **The owner:** executes must-sells same-day; approves re-entries; logs overrides; runs the weekly/monthly reviews. Nothing else. Every additional in-loop decision the owner makes weakens the evidence the system produces.
- **The nightly pipeline:** the only promoter of champions; the only exposure decision-maker.
- **The monitor:** rules auditor. It re-derives, never decides.
- **Research:** parallel and non-blocking. Its results enter production only through the gate.

## 6. Design Principles (non-negotiable, earned the hard way)

1. **Measure before iterating.** Three strategy generations converged on the same answer before the structure factor emerged; the honest harness made that visible. No construction changes without a pre-registered hypothesis and both variants reported.
2. **The gate arbitrates; nothing else does.** Floors are excess-over-universe (`GATE_RECALIBRATION_SPEC.md`); no floor changes without owner sign-off; no carve-outs for transitions, regimes, or events.
3. **No discretionary override inside the loop.** Sentiment, headlines, and profit-taking temptation are handled by rules set at entry — not at the moment of decision. Overrides are allowed only outside the loop, logged (G2), and reviewed.
4. **Fail closed is the default posture.** No champion, no regime read, no meter row → no picks. An idle scan is the system working, not the system failing.
5. **Small interfaces.** Layers communicate only through §2.4's four interfaces. Anything else is a design change.

## 7. References

- Product spec: `Spec.md`; agent rules: `AGENTS.md` (authoritative for implementation constraints)
- `REGIME_METER_SPEC.md` — regime meter (built, `enforce: true`)
- `REGIME_GATED_PRODUCTION_SPEC.md` — heuristic production mode (built)
- `GATE_RECALIBRATION_SPEC.md` — excess-floored gate (built)
- `REGIME_TRANSITION_SPEC.md` — path labels, purge, matrix, decomposition (built)
- `PHASE_C_SPEC.md` — regime-adaptive training (built)
- Build items G1–G6 above are unwritten specs; author them in this order when scheduled.
