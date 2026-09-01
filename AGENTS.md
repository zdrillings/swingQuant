# AGENTS.md

## Purpose

This file is for coding agents and contributors making changes in this repository. It defines the project's operational rules, architecture boundaries, and non-negotiable implementation constraints.

## Project Summary

SwingQuant is a modular swing-trading system with three core domains:

- data sync
- research and backtesting
- trading operations and notifications

The system is intentionally split between:

- DuckDB for historical market data
- SQLite for mutable ledger state

The authoritative product spec is [Spec.md](/home/zdrillings/code/SwingQuant/Spec.md), with one clarified policy detail:

- `sq monitor` is alert-only and does not close ledger trades automatically

## Code Map

- [src/cli.py](/home/zdrillings/code/SwingQuant/src/cli.py): top-level CLI routing
- [src/settings.py](/home/zdrillings/code/SwingQuant/src/settings.py): paths, `.env`, config loading
- [src/utils/db_manager.py](/home/zdrillings/code/SwingQuant/src/utils/db_manager.py): database initialization and persistence helpers
- [src/utils/feature_engineering.py](/home/zdrillings/code/SwingQuant/src/utils/feature_engineering.py): reusable feature calculations (macro features, trend, momentum, volume, price structure, gap risk, earnings events, regime context)
- [src/utils/signal_engine.py](/home/zdrillings/code/SwingQuant/src/utils/signal_engine.py): analysis frame construction, latest snapshot, signal filtering
- [src/utils/strategy.py](/home/zdrillings/code/SwingQuant/src/utils/strategy.py): promoted strategy loading, ExitRules (including hard stops), indicator gate evaluation, entry_stop_price
- [src/utils/regime.py](/home/zdrillings/code/SwingQuant/src/utils/regime.py): sector-to-regime mapping
- [src/utils/sizing.py](/home/zdrillings/code/SwingQuant/src/utils/sizing.py): position sizing
- [src/utils/shortlist_runtime.py](/home/zdrillings/code/SwingQuant/src/utils/shortlist_runtime.py): live model context loading, confidence metrics, prediction annotations
- [src/sync](/home/zdrillings/code/SwingQuant/src/sync): universe scraping and OHLCV sync
- [src/research](/home/zdrillings/code/SwingQuant/src/research): supervised feature research and shortlist modeling
- [src/research/shortlist_model_service.py](/home/zdrillings/code/SwingQuant/src/research/shortlist_model_service.py): walk-forward shortlist model bakeoff and champion selection
- [src/research/shortlist_bakeoff_service.py](/home/zdrillings/code/SwingQuant/src/research/shortlist_bakeoff_service.py): model policy comparison
- [src/research/universe_snapshot_service.py](/home/zdrillings/code/SwingQuant/src/research/universe_snapshot_service.py): daily snapshot backfill with features and outcomes including binary target
- [src/sweep](/home/zdrillings/code/SwingQuant/src/sweep): parameter sweep backtesting (5 archetype modes, hard stop in exit chain)
- [src/evaluate](/home/zdrillings/code/SwingQuant/src/evaluate): normalization and report ranking
- [src/promote](/home/zdrillings/code/SwingQuant/src/promote): runtime strategy generation
- [src/trade](/home/zdrillings/code/SwingQuant/src/trade): manual ledger updates for fills
- [src/scan](/home/zdrillings/code/SwingQuant/src/scan): end-of-day signal scanning with model-driven selection, quality gate, confidence-based position sizing
- [src/monitor](/home/zdrillings/code/SwingQuant/src/monitor): intraday alert digest generation (includes hard stop monitoring)

## Non-Negotiable Rules

1. Do not introduce random shuffling into any research or validation split.
2. Do not bypass `.env` for secrets or capital settings.
3. Do not use VectorBT in `sq sweep`.
4. Do not duplicate sector-to-regime logic; use [src/utils/regime.py](/home/zdrillings/code/SwingQuant/src/utils/regime.py).
5. Do not close `Active_Trades` from `sq monitor`.
6. Do not weaken the signal model into OR logic.
   - Keep `relative_strength_index_vs_spy_min` as a hard filter.
   - Keep the confluence score threshold driven by `signal_score_min`.
   - Keep `roc_63` available as a scored trend-strength component.
   - Keep breakout and pullback families separate; do not silently merge them into a single hybrid score model.
7. Do not hardcode current strategy thresholds; read `production_strategy.json`.
   - When multiple slots are active, use `production_strategies.json` and preserve slot isolation.
8. Do not remove hard stop from ExitRules, backtest, or monitor.
   - Hard stop (`hard_stop_pct`) anchors to entry price and triggers must-sell.
   - Hard stop is checked BEFORE trailing stop in the backtest exit chain.
   - Hard stop is a must-sell in monitor classification; the portfolio shock guard does not demote it.
9. Do not bypass the quality gate or rotation exclusion in scan candidate selection.
   - The model's confidence metrics are computed on the **top-2 basket** (`CONFIDENCE_BASKET_SIZE = 2` in `shortlist_runtime.py`), NOT the top-10.
     - Rationale: rank-calibration analysis (2026-08-13) showed ranks 3-10 inverted in recent months (negative alpha), dragging the top-10 basket to -1.0% while the top-2 basket was +6.7% with a 70% beat rate.
   - `_confidence_adjusted_max_candidates` is a 2-tier gate for the model path:
     - beat_rate < 0.35 or mean_target < -0.03 → 1 pick
     - otherwise → 2 picks
   - The model path is hard-capped at 2 picks (`min(effective_cap, 2)` in scan run()).
   - `_apply_rotation_exclusion` removes tickers selected in the last 3 scan dates when the effective cap is below the configured cap, so picks rotate through the model's top candidates instead of repeating the same name.
   - Gate floors at 1 (never produces zero candidates silently).
10. Do not add new sweep modes without explicit justification.
    - Keep the 5 archetype modes: `pullback_technology`, `pullback_real_economy`, `breakout_growth`, `post_earnings_drift`, `trend_continuation`.
    - Do not add `_v2`, `_v3`, `_refined`, or subindustry-specific variants.
11. Do not hardcode the live shortlist model.
    - `sq shortlist-model` must evaluate candidate models and persist the selected champion.
    - Runtime must use `production_model_name` only when it is explicitly configured; otherwise use the persisted `champion_model`.
    - Model-driven scan sections must fail closed when the recent promotion gate is not met.
    - Missing forward alpha must remain missing; do not convert immature labels into negative examples.

## Shortlist Model Architecture

The shortlist model is the core prediction engine. It evaluates multiple candidate models in chronological walk-forward windows and persists the selected champion for runtime use.

### Model Training (`sq shortlist-model`)

- **Models**: `signal_proxy`, `ridge_model`, `lasso_model`, and optional `xgboost_model`
- **Target**: `alpha_vs_sector_20d` (regression) or `alpha_vs_sector_20d_pos` (classification: 1 if alpha > 2%, else 0)
- **Features**: 47 raw features from `universe_daily_snapshots` (including `rsi_2`, `ret_1d`, `ret_5d`, `close_vs_20d_low` mean-reversion features), each with `__rank_all` and `__rank_sector` cross-sectional rank variants, plus sector one-hot dummies (~150 total features)
- **Macro features**: `spy_roc_20`, `spy_roc_5`, `spy_realized_vol_20`, `qqq_roc_20` — computed from SPY/QQQ price history, populated across all snapshot dates via DuckDB SQL
- **Binary target**: `alpha_vs_sector_20d_pos` — computed on-the-fly from matured `alpha_vs_sector_20d` if not yet backfilled, or stored in DuckDB; missing alpha stays missing
- **Validation**: Expanding-window walk-forward, 252 min train dates, horizon-spaced OOS snapshot dates (`oos_evaluation_stride_dates=20` for 20d target), chronological split only
- **Eligibility**: use historical `passed_slots_json` when available so the research universe reflects what passed on that snapshot date
- **Calibration**: OOS predictions are calibrated to `calibrated_p_beat_sector` (kept for evaluation/audit). Since 2026-09-01 (00144e4), live selection ranks on raw `predicted_alpha` only — the calibration must not drive live ranking while its fold evidence is frozen or stale
- **Model scopes**: `global` (single model), `sector_specific` (per-sector with fallback), `regime_specific` (2 regimes: trending when `spy_roc_20 > 1%` else choppy; uses `regime_green` when macro features unavailable)
- **Champion selection**: selected from model summaries and recent acceptance windows; it must not be hardcoded to the latest experiment
- **Feature purging**: For L1 models, zero-coefficient features are dropped and the model is re-fit on the reduced set (per-fold, honest)
- **Solver**: `liblinear` for classification, default coordinate descent for regression
- **Output**: Report at `reports/shortlist_model.md`, OOS predictions CSV, live predictions CSV

### Confidence Basket (top-2)

The model's evaluation product is its **top-2 predictions**, not the full top-10:

- **Rank calibration finding (2026-08-13)**: over the recent 60 walk-forward dates, rank 1 averaged +10.6% alpha, rank 2 +2.9%, but ranks 3+ were negative. The top-10 basket was -1.0% while the top-2 basket was +6.7% with a 70% beat rate. Ranks 3-10 have inverted — including them dilutes the model's real signal.
- `CONFIDENCE_BASKET_SIZE = 2` in `src/utils/shortlist_runtime.py` controls the basket used for `recent_20d_beat_rate` and `recent_20d_mean_target`.
- Do not widen the confidence basket back to 5 or 10 without re-running the rank-calibration analysis and documenting the finding.

### Model Runtime (`sq scan`)

- Loaded via `load_live_shortlist_model_context()` in `shortlist_runtime.py`
- Runtime loading is read-only by default; `sq scan` must not retrain or refresh the shortlist model.
- Stale or missing model context makes model-driven production scans fail closed unless `scan_policy.shortlist_model.allow_heuristic_fallback` is explicitly enabled.
- Uses explicit `production_model_name` only when configured; otherwise uses the persisted `champion_model`
- Returns no model context when the selected model fails `scan_policy.shortlist_model.promotion_gate`
- Heuristic fallback is a diagnostic escape hatch, not the production default.
- Confidence metrics (`recent_20d_beat_rate`, `recent_20d_mean_target`) are computed from the top-2 OOS basket, sorted by raw `predicted_alpha`, and threaded into the quality gate
- Model path selects at most 2 candidates; rotation exclusion removes the previous 3 scan dates' picks when the cap is below the configured total

## Command Behavior Expectations

### `sq sync`

- Bootstraps the universe only when `Universe` is empty.
- Fetches 5 years of daily OHLCV.
- Fetches and persists earnings calendar dates for the research universe.
- Uses retries with exponential backoff.
- Marks permanently failed tickers inactive.
- Applies the median 30-day dollar-volume liquidity filter.
- Must remain idempotent for historical upserts.

### `sq universe-backfill`

- Recomputes all feature columns including macro features (`spy_roc_20`, `spy_roc_5`, `spy_realized_vol_20`, `qqq_roc_20`) and binary target (`alpha_vs_sector_20d_pos`) for each date.
- Macro features are also populated historically via direct DuckDB SQL from `historical_ohlcv` (SPY/QQQ price data).
- Use `--skip-existing` to only process new dates; omit to overwrite.

### `sq shortlist-model`

- Trains walk-forward shortlist model.
- Refuses to persist a champion when no candidate passes the configured promotion gate.
- Key flags:
  - `--target-type regression|classification` (default: regression)
  - `--model-scope global|sector_specific|regime_specific` (default: sector_specific for production)
  - `--eligible-universe-mode passed_only|passed_or_trend`
- Classification mode predicts `alpha_vs_sector_20d_pos` (binary: >2% alpha).
- Feature purging logs the number and names of zeroed features per fold.
- Macro features that are NaN (pre-backfill) are filled with 0; the L1 model purges them until enough backfill history accumulates.

### `sq research`

- Operates on the top 250 names by `md_volume_30d`.
- Uses the 20-trading-day forward success label.
- Trains on oldest 70% of dates and validates on newest 30%.

### `sq sweep`

- Uses Polars as the backtest engine.
- Sweeps dynamic exit rules from config including `hard_stop_pct`.
- Hard stop is checked BEFORE trailing stop in the exit priority chain (entry-price-anchored, more protective).
- Supports ATR-based exits; keep sweep and runtime exit semantics aligned.
- Supports earnings-aware entry filters and pre-earnings exits.
- Applies configurable execution costs from `config.yaml` to every simulated trade.
- Stores sector scope inside `params_json`.
- 5 archetype sweep modes only:
  - `pullback_technology` — tech pullback-with-trend
  - `pullback_real_economy` — Energy/Materials/Industrials/Financials pullback
  - `breakout_growth` — breakout entries across Tech/Industrials/Financials
  - `post_earnings_drift` — post-earnings continuation
  - `trend_continuation` — simple trend-following with minimal filters

### `sq evaluate`

- Must min-max normalize expectancy, profit factor, and max drawdown before scoring.
- Writes `reports/candidates.md`.

### `sq promote`

- Must emit a fully formed `production_strategy.json` including `hard_stop_pct` in exit rules.
- Must include `promoted_at`.
- When `--slot` is used, update only that slot inside `production_strategies.json`.

### `sq trade`

- `buy` opens ledger positions.
- When the active strategy uses ATR exits, `buy` must persist `entry_atr` for runtime monitoring.
- `buy` must also persist strategy linkage (`strategy_id` / `strategy_slot`) whenever a slot can be resolved.
- `sell` is the only command that closes ledger positions.

### `sq scan`

- Uses the most recent completed session's adjusted close.
- Two candidate sources (auto-selected):
  - **Model path**: inner-joins snapshot with live model predictions; no signal gate (model output IS the selection signal). Production scans fail closed if model context is missing, stale, failing promotion, or maps to zero active-slot candidates.
  - **Heuristic path**: per-slot signal gate via `filter_signal_candidates`, then scored via `_score_candidate`; use only when shortlist model source is disabled or `allow_heuristic_fallback` is explicitly enabled for diagnostics.
- Quality gate: `_confidence_adjusted_max_candidates` scales position count based on model's recent 20d beat rate and mean target.
  - Model active + confident (beat_rate >= 0.45, mean_target >= 0): full cap (6)
  - Model active + borderline: reduced to 2
  - Model active + poor (beat_rate < 0.40 or mean_target < -0.02): reduced to 1
  - Heuristic fallback: diagnostic only; reduced to 75% of cap (floor 3) when explicitly enabled.
- Portfolio caps: per-slot (3), per-sector (3), total (6, or quality-gate-adjusted).
- Sends one evening brief containing the selected candidates.

### `sq monitor`

- Uses intraday 1-minute data for last-trade price.
- Computes hard stop price via `entry_stop_price()` (anchored to entry price).
- Evaluates:
  - hard stop (must-sell, not demotable by shock guard)
  - breakout alert
  - trailing stop
  - profit target
  - RSI_2 > 90
  - time limit
  - regime flip
  - pre-earnings exit when promoted
- When the active strategy uses ATR exits, evaluate stop and target off stored `entry_atr`.
- Sends one consolidated digest per run.
- Recommends `sell` in the digest when exit conditions are met.
- Does not close the trade in SQLite.

## Exit Rule Priority

In both backtest (`sq sweep`) and live monitoring (`sq monitor`), exits are evaluated in this priority order:

1. **Regime flip** — exit at close
2. **Hard stop** — exit at `entry_price * (1 - hard_stop_pct)`. Anchored to entry price (not trailing). Must-sell in monitor.
3. **Trailing stop** — exit at `max_price_seen - (entry_atr * mult)` (ATR) or `max_price_seen * (1 - pct)` (percent)
4. **Profit target** — exit at `entry_price + (entry_atr * mult)` (ATR) or `entry_price * (1 + pct)` (percent)
5. **RSI_2 > 90** — exit at close
6. **Time limit** — exit at close after `time_limit_days`
7. **Pre-earnings exit** — exit at close when `days_to_next_earnings <= exit_before_earnings_days`

Hard stop, profit target, and pre-earnings exit are **must-sell** (not demoted by the portfolio shock guard).

## Database Guidance

### DuckDB

Tables:

- `historical_ohlcv` — daily OHLCV for all universe + reference tickers
- `universe_daily_snapshots` — point-in-time feature snapshots with forward outcomes and macro features
- `analyst_snapshots` — analyst target captures
- `analyst_revision_snapshots` — analyst estimate/revision captures
- `extended_hours_snapshots` — postmarket movement captures

Primary use:

- bulk historical reads
- missing-date fetch planning
- liquidity calculations

### SQLite

Tables:

- `Universe`
- `Backtest_Results`
- `Active_Trades`
- `Earnings_Calendar`
- `Scan_Candidates`
- `Shortlist_Model_Runs`
- `Shortlist_Model_Predictions`

Primary use:

- mutable system state
- ranked results
- trade lifecycle tracking

If you need new persistence, prefer:

- DuckDB for analytical history
- SQLite for operational state

## Testing Expectations

Before considering a change complete:

1. Run:

```bash
python3 -m unittest discover -s tests -v
```

2. Run:

```bash
python3 -m compileall src
```

3. If you change a command's behavior, add or update a command-specific test under [tests](/home/zdrillings/code/SwingQuant/tests).

4. If you touch regime logic, add a regression test proving the helper path is used.

5. If you touch monitor behavior, verify:
   - one digest only
   - all exit rules still evaluated (including hard stop)
   - hard stop is must-sell
   - no implicit trade closure

6. If you touch scan behavior, verify:
   - model path produces candidates
   - production model path fails closed when model context is unavailable or empty
   - quality gate respects confidence metrics
   - candidate count is >= 1

## Current Known Tradeoffs

- The shortlist model is a single L1 linear model. It cannot capture non-linear interactions that XGBoost could, but is more stable, more interpretable, and faster.
- `regime_specific` model scope trains 3 models per fold and is ~15x slower than `sector_specific`. Use `sector_specific` for production.
- Macro features (`spy_roc_*`, `spy_realized_vol_*`, `qqq_roc_20`) are consistently purged by the L1 model — they show zero predictive value for individual stock selection.
- Binary classification target (`alpha_vs_sector_20d_pos`) uses a 2% threshold; this is arbitrary and could be tuned.
- The quality gate reduces positions when the model is wrong, but cannot fix the underlying feature set problem.
- `sq evaluate --sector` filters using sector metadata stored in `params_json`, not a dedicated database column.
- `sq sweep` is compliant but not fully vectorized internally.
- The launcher supports `.vendor/` automatically for local dependency installs.
- Runtime validations in development may use synthetic data and mocked email delivery.

## Safe Change Patterns

- Reuse central helpers instead of inlining logic.
- Keep services thin and push shared logic into `src/utils`.
- Extend tests whenever you fix a bug, especially for spec compliance.
- Prefer explicit policy decisions over hidden automation in operational commands.
- When adding features to `MODEL_FEATURE_COLUMNS`, always handle missing columns in `_prepare_model_matrices` (fill with NaN) and add the column to `SNAPSHOT_FEATURE_COLUMNS` for backfill support.
- When adding fields to `ExitRules`, thread through all three serializers (`_production_strategy_from_payload`, `build_production_strategy_payload`, `production_strategy_from_backtest_result`), the sweep exit chain, and the monitor exit flags.
