# Regime Meter — Build Spec

**Date:** 2026-09-09
**Builder:** Codex
**Owner:** Zachary Drillings
**Status:** build spec — implement exactly as written; where a choice is not specified, ask rather than inventing.

---

## 1. Problem

The system is a one-sided momentum bet: the eligible universe (trend-qualified: green regime, above-200d, positive 63d momentum), the feature set (roc/RS/trend), and the model all load on recent strength. Since mid-May 2026 the market has been paying short-term reversal: roc_126's cross-sectional IC vs forward 20d sector alpha flipped to **-0.17** (recent 60 days) from ≈0 full-sample, and the eligible universe itself has run at **-1.9%/20d** vs sector. Every layer of the stack points the same way, so when the regime flips the whole system loses together.

The regime is measurable in advance. Monthly averages of the daily cross-sectional Spearman IC of `roc_126` vs `alpha_vs_sector_20d` (computed from `universe_daily_snapshots`, validated 2026-09-09):

| month | momentum IC | month | momentum IC |
|---|---:|---|---:|
| 2026-01 | +0.142 | 2026-05 | +0.031 |
| 2026-02 | -0.070 | **2026-06** | **-0.225** |
| 2026-03 | +0.179 | **2026-07** | **-0.277** |
| 2026-04 | +0.174 | **2026-08** | **-0.124** |

A stand-down rule with a -0.05 threshold would have classified June 2026 as reversal by roughly mid-June (the 20-day label lag; see §3) and held the system flat through July–August, which was the worst stretch of realized losses.

**Purpose of this build:** a daily momentum-regime meter persisted to DuckDB, reported nightly, and wired into the scan as an optional hard stand-down for momentum slots. The regime-conditional model (reversal features) is a future build, scoped in §8 but NOT part of this one.

---

## 2. Metrics (exact definitions)

All metrics are computed per snapshot date `d` from `universe_daily_snapshots` restricted to rows where `alpha_vs_sector_20d IS NOT NULL AND roc_126 IS NOT NULL` (the matured, eligible rows already persisted). Use the full snapshot table (it is the eligible universe), not an ad-hoc filter.

For each date `d`:

1. **`mom_ic_daily`** — cross-sectional Spearman IC of momentum vs forward alpha:
   - Rank `roc_126` and `alpha_vs_sector_20d` within date `d` (average ranks, ties allowed), then take Pearson correlation of the two rank vectors. (DuckDB: `corr()` over rank-transformed columns per date — do NOT hand-roll Spearman; `corr` of ranks is Spearman.)
2. **`wml_20d_alpha`** — winner-minus-loser on the same day: `avg(alpha_vs_sector_20d of top roc_126 quintile) - avg(alpha_vs_sector_20d of bottom quintile)` within date `d` (use `ntile(5)` partitioned by `snapshot_date`). Secondary confirmation signal; also 20d-lagged.
3. **`wml_1d`** — zero-lag WML on `ret_1d` (top minus bottom roc_126 quintile of same-day 1d return). **Diagnostic only.** Validated 2026-09-09: this signal does not discriminate regimes (weak positive almost every month, including the crash) and must NOT be used in the classification.
4. **Rolling averages** (simple means over the 20 and 60 most recent matured dates): `mom_ic_20d_avg`, `mom_ic_60d_avg`. Both from `mom_ic_daily` only.

**Lag discipline (critical):** `alpha_vs_sector_20d` at date `d` is only known after `d + 20` trading sessions. Therefore the most recent row the meter can know on scan day `T` is `d = T - 20` sessions. The meter is a **slow regime switch, not a fast stop** — accept the lag, do not try to hide it with unvalidated zero-lag proxies. Any code path that consumes the meter on day `T` must use the latest matured row (`snapshot_date <= T - 20 sessions`) and nothing later. This is the single most important rule in this spec.

## 3. Classification rule

From the latest matured row:

- `reversal` if `mom_ic_20d_avg < reversal_ic_threshold` (default **-0.05**)
- `trending` if `mom_ic_20d_avg > trending_ic_threshold` (default **+0.05**)
- else `neutral`

Use `mom_ic_20d_avg` as the primary input (faster than 60d, less jumpy than daily). `mom_ic_60d_avg` and `wml_20d_alpha` are reported alongside but do not vote. Defaults above are calibrated from §1's data; they live in config (§4) and are tunable, not hardcoded in logic.

**Validation requirement (acceptance):** after backfill, the classification series must show `reversal` for the matured dates from roughly 2026-06-20 through at least 2026-08, and `trending` or `neutral` during 2026-01, 2026-03–04. If the backfilled series materially disagrees with the table in §1, stop and report — do not adjust the computation until the discrepancy is explained.

## 4. Persistence and config

**DuckDB table** `regime_meter` (create via migration in `src/utils/db_manager.py`, alongside the existing DuckDB migrations):

| column | type | notes |
|---|---|---|
| `snapshot_date` | DATE, PK | matured snapshot date the metrics describe |
| `mom_ic_daily` | DOUBLE | daily Spearman IC |
| `wml_20d_alpha` | DOUBLE | WML of 20d sector alpha |
| `wml_1d` | DOUBLE | diagnostic only |
| `mom_ic_20d_avg` | DOUBLE | mean of prior 20 matured dates (min 10 else NULL) |
| `mom_ic_60d_avg` | DOUBLE | mean of prior 60 matured dates (min 20 else NULL) |
| `classification` | VARCHAR | trending / neutral / reversal (computed from the avg thresholds) |

Writes are replace-by-date (idempotent), matching `replace_universe_daily_snapshots` convention. Rolling averages must be computed over the *full backfilled series in date order* — a backfill of one date can change subsequent rows' rolling averages, so `--latest` must recompute the trailing window (recompute from the earliest date whose rolling window includes the appended date, i.e., last 60 rows), not just insert one row.

**Config** (`config.yaml`, new section — mirror the style of `scan_policy`):

```yaml
regime_gating:
  reversal_ic_threshold: -0.05
  trending_ic_threshold: 0.05
  enforce: false            # flip to true only after Phase B sign-off (§8)
  stand_down_slots: []      # empty = all active momentum slots stand down
```

## 5. CLI

New subcommand `sq regime-meter` (`src/cli.py`, service in `src/research/regime_meter_service.py` following the existing service pattern — `run()` returning a small dataclass report):

- `--backfill` — compute the full history (from the earliest `universe_daily_snapshots` date with matured alpha) into `regime_meter`; idempotent.
- `--latest` — backfill the trailing window only (for the nightly pipeline).
- `--report` — write `reports/regime_report.md`: current classification, latest matured date, the three metric values, the trailing 60-date table (date, ic_daily, ic_20d_avg, classification), and the historical monthly averages (§1 table shape). Always state the lag explicitly in the report: "metrics as of {latest_matured_date} (20-session label lag by design)."

No flags should let a caller override thresholds at the CLI; thresholds come from config only.

## 6. Scan integration

In `src/scan/service.py`, at the top of the scan run (before candidate construction):

1. Load the latest matured `regime_meter` row (the lag rule from §2).
2. If `regime_gating.enforce` is **false**: log the classification and proceed exactly as today. (This is Phase A behavior — the default.)
3. If `enforce` is **true** and classification is `reversal`: skip both the model path and the heuristic path for the slots named in `stand_down_slots` (all slots when empty); do **not** record a promotion failure (this is not a gate failure — leave `data/promotion_failures.txt` and champion state untouched); exit the scan cleanly with a clear log line and the stand-down line in the email subject/body (e.g., "REGIME STAND-DOWN — momentum IC {value}, no picks emitted").
4. Zero-candidate scans must still render a valid report/email with the stand-down note — never crash on empty selection.
5. The regime line also goes into the scan-performance email header (one line: `regime: {classification} (mom_ic_20d {value:.3f})`), regardless of enforce, so the owner sees the state daily.

## 7. Nightly pipeline

In `ops/nightly_pipeline.sh`, after `universe-backfill` and before `shortlist-model`:

```bash
./sq regime-meter --latest
./sq regime-meter --report
```

Order matters: the backfill must run first so the latest matured date exists. The regime-report step must not be able to fail the pipeline (tolerate errors with a log line, same pattern as the path-target dry-run).

## 8. Usage policy and phasing (for the owner)

- **Phase A (this build):** `enforce: false`. The nightly report shows the meter; the owner watches it against realized performance for 2–4 weeks. The scan email carries the regime line.
- **Phase B:** after retrospective validation (§3 acceptance) and owner sign-off, set `enforce: true`. Momentum slots stand down in reversal regimes; the scan fails *closed-by-regime*, not *failed-by-gate* — the distinction must be visible in the email and logs.
- **Phase C (future build, NOT this one):** regime-conditional selection. In `reversal` regimes, train/select with the reversal feature family already in the snapshots (`rsi_2`, `ret_1d`, `close_vs_20d_low`) — the fold-local IC screen must become regime-conditioned (screen features within regime-matched training slices), and the eligible-universe trend filter must relax in reversal regimes (allow pullback candidates). Scope that separately; do not fold it into this build.
- **Never:** let the meter override the promotion gate (gate = model honesty; meter = regime exposure). Never use `wml_1d` in classification. Never backfill with un-matured alpha.

## 9. Tests (required, AGENTS.md standards)

1. Classification thresholds: synthetic metric series → trending/neutral/reversal at the boundaries (±0.05 exact values fall into neutral; test both sides).
2. Lag discipline: a regression test proving the scan consumes only rows with `snapshot_date <= latest_matured` (no lookahead) — construct a meter table with a future-dated row and assert it is not used.
3. Rolling-average recompute: append one date to an existing backfill and assert the trailing window rows were recomputed, not just appended.
4. Scan stand-down: `enforce=true` + reversal classification → zero candidates, no promotion-failure write, email contains the stand-down line; `enforce=false` → normal path.
5. Pipeline script assertions (extend `tests/test_nightly_pipeline.py` pattern) for the two new steps.
6. Backfill idempotency: running `--backfill` twice yields identical rows.
7. Run `python3 -m unittest discover -s tests` (with `PYTHONPATH=.vendor`, per AGENTS.md) and `python3 -m compileall src` before completion.
