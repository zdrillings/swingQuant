# AGENTS.md

## Purpose

This file is for coding agents and contributors making changes in this repository. It defines the project’s operational rules, architecture boundaries, and non-negotiable implementation constraints.

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
- [src/utils/feature_engineering.py](/home/zdrillings/code/SwingQuant/src/utils/feature_engineering.py): reusable feature calculations
- [src/utils/signal_engine.py](/home/zdrillings/code/SwingQuant/src/utils/signal_engine.py): analysis frame construction, latest snapshot, signal filtering
- [src/utils/strategy.py](/home/zdrillings/code/SwingQuant/src/utils/strategy.py): promoted strategy loading and indicator gate evaluation
- [src/utils/regime.py](/home/zdrillings/code/SwingQuant/src/utils/regime.py): sector-to-regime mapping
- [src/utils/sizing.py](/home/zdrillings/code/SwingQuant/src/utils/sizing.py): position sizing
- [src/sync](/home/zdrillings/code/SwingQuant/src/sync): universe scraping and OHLCV sync
- [src/research](/home/zdrillings/code/SwingQuant/src/research): supervised feature research
- [src/sweep](/home/zdrillings/code/SwingQuant/src/sweep): parameter sweep backtesting
- [src/evaluate](/home/zdrillings/code/SwingQuant/src/evaluate): normalization and report ranking
- [src/promote](/home/zdrillings/code/SwingQuant/src/promote): runtime strategy generation
- [src/trade](/home/zdrillings/code/SwingQuant/src/trade): manual ledger updates for fills
- [src/scan](/home/zdrillings/code/SwingQuant/src/scan): end-of-day signal scanning
- [src/monitor](/home/zdrillings/code/SwingQuant/src/monitor): intraday alert digest generation

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
7. Do not hardcode current strategy thresholds; read `production_strategy.json`.
   - When multiple slots are active, use `production_strategies.json` and preserve slot isolation.

## Command Behavior Expectations

### `sq sync`

- Bootstraps the universe only when `Universe` is empty.
- Fetches 5 years of daily OHLCV.
- Uses retries with exponential backoff.
- Marks permanently failed tickers inactive.
- Applies the median 30-day dollar-volume liquidity filter.
- Must remain idempotent for historical upserts.

### `sq research`

- Operates on the top 250 names by `md_volume_30d`.
- Uses the 20-trading-day forward success label.
- Trains on oldest 70% of dates and validates on newest 30%.

### `sq sweep`

- Uses Polars as the backtest engine in the current implementation.
- Sweeps dynamic exit rules from config; do not silently fall back to fixed stop/target values when the grid provides them.
- Supports ATR-based exits; keep sweep and runtime exit semantics aligned.
- Applies configurable execution costs from `config.yaml` to every simulated trade.
- Stores sector scope inside `params_json` so `sq evaluate --sector` can filter without changing the SQLite schema.
- The current trade simulation still contains row iteration inside ticker partitions.
  - This is acceptable at current scope.
  - Future optimization should target vectorized exits or a more specialized engine path.
- Keep progress logging intact or improve it; do not remove visibility for long runs.

### `sq evaluate`

- Must min-max normalize expectancy, profit factor, and max drawdown before scoring.
- Writes `reports/candidates.md`.

### `sq promote`

- Must emit a fully formed `production_strategy.json`.
- Must include `promoted_at`.
- When `--slot` is used, update only that slot inside `production_strategies.json`.

### `sq trade`

- `buy` opens ledger positions.
- When the active strategy uses ATR exits, `buy` must persist `entry_atr` for runtime monitoring.
- `buy` must also persist strategy linkage (`strategy_id` / `strategy_slot`) whenever a slot can be resolved.
- `sell` is the only command that closes ledger positions.

### `sq scan`

- Uses the most recent completed session’s adjusted close.
- Applies regime filter before signal evaluation.
- Uses the promoted strategy’s thresholds, relative-strength hard filter, and confluence score.
- Current score components include `rsi_14`, `vol_alpha`, `sma_200_dist`, and `roc_63`.
- Position sizing must respect the promoted stop model, including ATR-based stops.
- Sends one evening brief containing the top five signals.

### `sq monitor`

- Uses intraday 1-minute data for last-trade price.
- Evaluates:
  - breakout alert
  - trailing stop
  - profit target
  - RSI_2 > 90
  - time limit
  - regime flip
- When the active strategy uses ATR exits, evaluate stop and target off stored `entry_atr`.
- Sends one consolidated digest per run.
- Recommends `sell` in the digest when exit conditions are met.
- Does not close the trade in SQLite.

## Database Guidance

### DuckDB

Table:

- `historical_ohlcv`

Primary use:

- bulk historical reads
- missing-date fetch planning
- liquidity calculations

### SQLite

Tables:

- `Universe`
- `Backtest_Results`
- `Active_Trades`

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

3. If you change a command’s behavior, add or update a command-specific test under [tests](/home/zdrillings/code/SwingQuant/tests).

4. If you touch regime logic, add a regression test proving the helper path is used.

5. If you touch monitor behavior, verify:
   - one digest only
   - all exit rules still evaluated
   - no implicit trade closure

## Current Known Tradeoffs

- `sq evaluate --sector` filters using sector metadata stored in `params_json`, not a dedicated database column.
- `sq sweep` is compliant but not fully vectorized internally.
- The launcher supports `.vendor/` automatically for local dependency installs.
- Runtime validations in development may use synthetic data and mocked email delivery.

## Safe Change Patterns

- Reuse central helpers instead of inlining logic.
- Keep services thin and push shared logic into `src/utils`.
- Extend tests whenever you fix a bug, especially for spec compliance.
- Prefer explicit policy decisions over hidden automation in operational commands.
