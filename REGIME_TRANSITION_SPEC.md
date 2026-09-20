# Regime Transition Handling — Build Spec

**Date:** 2026-09-19
**Builder:** Codex
**Owner:** Zachary Drillings
**Status:** build spec — implement exactly as written; ask rather than inventing where unspecified.

---

## 1. Problem

The model conditions on the **entry-date regime** (`R_t`, lagged by the label horizon as today), but the label `y_t = forward return over [t, t+h]` is realized across whatever regimes occur inside the window. Near regime transitions, training rows labeled with `R_t` carry outcomes realized under a different regime — the model learns the wrong mapping. Observed case (2026-09-19): the last 60d fold entered 2026-05-13 → 2026-06-10 while the meter was `neutral`; the window matured through the `reversal` that began ~2026-06-17. Result: net -10.23%, hit 30%, beat 20%, spearman -0.247.

For a 20d target a mid-window regime change is possible; for 60d it is likely. This spec adds four pieces: (1) path-aware labels at the model's actual trade horizon, (2) a configurable regime-boundary purge for fixed-horizon training rows, (3) a regime transition matrix in the regime report, (4) a per-fold contamination decomposition in the shortlist report.

**Hard rule carried over from prior discussion: the promotion gate must NOT be made transition-aware (no carving out transition windows from gate inputs).** A live trade opened before a transition genuinely rides into it; those outcomes are the real distribution. All additions here are labels, training-data hygiene, and reporting — never gate exemptions.

## 2. Part 1 — Path-aware labels at the trade horizon

The strategy never holds 60 sessions: production time limits are 10–20 sessions and stops fire on the way down. A fixed 60d label pretends the trade rides the full reversal; the path label exits at the strategy's actual rules, making the label regime-robust by construction.

**Implementation:** reuse `_path_outcome` (`universe_snapshot_service.py`) with `horizon=60`:

- New columns: `path_return_60d`, `path_alpha_vs_sector_60d`, `path_exit_reason_60d` (plus `path_holding_days_60d` if not derivable from existing fields — check `_path_outcome`'s payload; if `holding_days` is not persisted for the 20d family, persist it for both).
- Exit rules, worst-case same-bar ordering, gap-aware fills, and benchmark-matched-to-holding-days behavior are identical to the 20d family. `max_horizon = min(60, time_limit_days)` — most paths will exit at the 10–20 session time limit; that is correct and the tearsheet must show the exit-reason mix so it is visible.
- Schema: add the columns in `db_manager.py` (DuckDB migration), add to `SNAPSHOT_FEATURE_COLUMNS` and the outcome-required list in `universe_snapshot_service.py` (same pattern as the 20d family in commit `1b9b4d9`).
- Backfill: extend `sq universe-backfill`; run a **full-history backfill** for this family (the nightly 120-day window alone will not populate history — coordinate with the owner before running it, it is the same operation as before).
- Tearsheet: extend `sq path-label-tearsheet` to render `60d` side by side with `20d` (fixed vs path distribution rows, paired difference, exit-reason mix). Add one summary line per horizon: `fixed_60d_mean`, `path_60d_mean`, `mean_holding_days`.
- Model: `sq shortlist-model --target-type path` must accept `--horizon 60` (check the current wiring; the path target currently derives from `horizon_days` — make sure `path_alpha_vs_sector_60d` is selected when `--horizon 60 --target-type path`).

## 3. Part 2 — Regime-boundary purge for fixed-horizon training rows

For the **fixed-horizon** target only (path labels already handle transitions), drop training rows whose forward window crossed a regime boundary.

- Regime for the window: the lagged meter series (existing `_regime_classifications_by_prediction_date` mapping, 20-session lag — this is the regime the model could have known at entry; use it, not the realized entry regime).
- Modes (config key `regime_transition_purge`, under `scan_policy.shortlist_model`):
  - `off` — no purge (default; the report still shows what a purge would remove — see Part 4).
  - `majority` — purge the row if the **majority** of the forward-window sessions (dates t+1..t+h with a known lagged regime) differ from `R_t`.
  - `strict` — purge if **any** forward-window session's regime differs from `R_t`.
- Purge applies to the training pool only, never to test/held-out rows, never to the gate's evaluation rows, never to the live snapshot.
- Add the dry-run count to the report: `regime_transition_purge: off (would purge N rows of M under majority, K under strict)` computed once per run, so the owner can see the cost of each mode before flipping the flag.
- The purge must be implemented in the fold loop's training-pool construction (where `_regime_matched_train_dates` runs), not post-hoc in the frame.

## 4. Part 3 — Regime transition matrix

New section in `reports/regime_report.md`:

- From the `regime_meter` table (1,195+ rows, growing): for horizons 20 and 60 **sessions**, the empirical 3×3 matrix P(next regime | current regime) with counts. Pair row d with the meter row d+h sessions later (next available row ≤ d+h; if none exists, skip the pair).
- Render: two tables (h=20, h=60), rows = entry regime, columns = realized regime, values = probability and count. Plus one line: `current_classification + expected distribution at +20 / +60 sessions`.
- Interpretation note in the report: this matrix feeds *sizing and exposure* decisions (Part 5), not the gate.

## 5. Part 4 — Per-fold contamination decomposition (report-only)

New section in `reports/shortlist_model.md`, after the Recent Acceptance Windows:

- For each acceptance window (20d, 60d, 1fold, 3fold — whatever the report already renders), add:
  - entry regime (majority of the window's entry dates),
  - forward-window regime share: for each entry date d in the window, the share of sessions in [d+1, d+h] classified neutral / trending / reversal (lagged series),
  - the window's net mean target / hit / beat (already rendered; repeat the one line so the table is self-contained).
- Header line: `note: contamination decomposition is diagnostic only and is not consumed by the promotion gate.`
- Implementation: computed in the report builder using the meter series; it may use realized regimes after the window (reporting is retrospective by nature) but must be structurally incapable of feeding the gate — keep it inside the report-rendering path and add the regression test in §7.

## 6. Live-side usage (no code beyond Part 3's matrix)

No live prediction changes. The forward regime is unknowable at entry; any feature encoding it is leakage. The owner uses the transition matrix + meter for exposure decisions: when the meter is `neutral` and the historical P(neutral→reversal within 60) is high, shorten time limits or reduce size. Sizing/scan changes are out of scope for this changeset.

## 7. Tests (AGENTS.md standards; suite currently 313)

1. Path 60d: synthetic OHLCV where a hard stop fires inside the window → `path_return_60d` = stop outcome, exit reason correct; a row with `time_limit_days=10` exits at session 10 even with `horizon=60`; gap-open fill uses the gap price (`hard_stop_gap`).
2. Purge: synthetic regime series + folds → `majority` and `strict` purge exactly the specified rows; `off` purges none; the purge never removes test/live rows; dry-run counts correct.
3. Transition matrix: hand-checkable synthetic meter table (known pairs) → exact probabilities and counts; horizon-60 pairing skips windows without a meter row.
4. Contamination decomposition: synthetic folds + meter → correct entry regime and window shares; **regression test: enabling/disabling the decomposition does not change the promotion gate outcome on identical inputs** (proves report-only).
5. Tearsheet: renders 60d rows and exit-reason mix; `--target-type path --horizon 60` selects `path_alpha_vs_sector_60d`.
6. Full suite + `compileall` green.

## 8. Acceptance

1. `reports/path_label_tearsheet.md` shows 60d fixed-vs-path with exit-reason mix (time_limit expected to dominate).
2. `reports/shortlist_model.md` shows the contamination decomposition; the recent fold's row reads entry=neutral, window majority=reversal — self-explanatory without external analysis.
3. `reports/regime_report.md` shows both transition matrices with counts.
4. The purge dry-run counts render; flag remains `off` until the owner decides with data.
5. No gate behavior changed; no lookahead introduced into training (lagged regime series only).

## 9. Non-goals

- No transition-aware gate carve-outs (hard rule, §1).
- No live regime-prediction features.
- No scan/sizing changes.
- No changes to the regime meter computation itself.
