# SwingQuant

SwingQuant is a CLI-driven swing-trading research and operations toolkit built around DuckDB for market history, SQLite for the trading ledger, and a single `sq` command surface.

## Scope

Implemented commands:

- `sq init-db`
- `sq sync`
- `sq research`
- `sq sweep`
- `sq evaluate`
- `sq promote --id <ID> [--slot <name>]`
- `sq trade buy <ticker> <price> [shares]`
- `sq trade sell <ticker> <price>`
- `sq scan`
- `sq monitor`

## Architecture

- DuckDB stores historical OHLCV in `historical_ohlcv`.
- SQLite stores:
  - `Universe`
  - `Backtest_Results`
  - `Active_Trades`
- The main code lives in [src](/home/zdrillings/code/SwingQuant/src).
- Tests live in [tests](/home/zdrillings/code/SwingQuant/tests).

Key modules:

- [src/utils/db_manager.py](/home/zdrillings/code/SwingQuant/src/utils/db_manager.py): schema creation and data access
- [src/sync/service.py](/home/zdrillings/code/SwingQuant/src/sync/service.py): universe bootstrap and OHLCV sync
- [src/research/service.py](/home/zdrillings/code/SwingQuant/src/research/service.py): feature training and importance reporting
- [src/sweep/service.py](/home/zdrillings/code/SwingQuant/src/sweep/service.py): parameter grid backtests with Polars
- [src/evaluate/service.py](/home/zdrillings/code/SwingQuant/src/evaluate/service.py): result normalization and report generation
- [src/scan/service.py](/home/zdrillings/code/SwingQuant/src/scan/service.py): daily post-close signal scan
- [src/monitor/service.py](/home/zdrillings/code/SwingQuant/src/monitor/service.py): intraday breakout and exit monitoring

## Setup

1. Create `.env` from `.env.example` and fill in:
   - `GMAIL_USER`
   - `GMAIL_APP_PASSWORD`
   - `RECIPIENT_EMAIL`
   - `TOTAL_CAPITAL`
   - `RISK_PER_TRADE`
2. Install dependencies from [pyproject.toml](/home/zdrillings/code/SwingQuant/pyproject.toml).
3. Initialize schemas:

```bash
./sq init-db
```

The launcher in [sq](/home/zdrillings/code/SwingQuant/sq) automatically adds `.vendor/` to `PYTHONPATH` if that directory exists.

## Typical Workflow

1. Sync market data:

```bash
./sq sync
```

2. Run feature research:

```bash
./sq research
```

3. Sweep parameter combinations:

```bash
./sq sweep
```

4. Rank candidates:

```bash
./sq evaluate --top 10
```

5. Promote one or more strategies:

```bash
./sq promote --id 244786 --slot materials
./sq promote --id 172365 --slot technology
```

6. Run end-of-day scan:

```bash
./sq scan
```

7. Run intraday monitoring:

```bash
./sq monitor
```

8. Record fills manually in the ledger:

```bash
./sq trade buy DOW 53.25 --slot materials
./sq trade buy AEIS 105.10 --slot technology
./sq trade sell AAPL 192.40
```

## Operational Rules

- All train/validation splits are chronological.
- `sq scan` uses:
  - a hard relative-strength filter via `relative_strength_index_vs_spy_min`
  - a confluence score across the promoted score components
  - `signal_score_min` as the pass threshold
  - scored components currently include `rsi_14`, `vol_alpha`, `sma_200_dist`, and `roc_63`
  - position sizing uses the promoted stop model, including ATR-based stops when present
  - multiple active strategy slots are evaluated independently, then merged under `scan_policy` caps from `config.yaml`
  - the current scan policy limits total ideas and also caps per-slot and per-sector concentration
- Non-tech sectors default to the SPY regime; Information Technology and Communication Services use QQQ.
- `sq monitor` is alert-only.
  - It updates `max_price_seen`.
  - It evaluates all exit rules every run.
  - It sends one consolidated digest.
  - It does not close `Active_Trades`; use `sq trade sell` to close trades in the ledger.
  - Legacy imported holdings without `strategy_slot` now fall back to the best available exact-sector or regime-family strategy and are backfilled into the ledger.
- Multiple active runtime strategies are supported through `production_strategies.json`.
  - `sq promote --slot <name>` updates one named strategy slot without overwriting the others.
  - `sq scan` evaluates each active slot against its own sector scope and thresholds.
  - `sq scan` email output is grouped by strategy slot so each slot's candidate set is visible separately.
  - `sq monitor` resolves each open trade to the correct active strategy by stored slot/id, then by sector fallback for legacy rows.
- `sq sweep` uses Polars and does not use VectorBT.
- `sq sweep` now sweeps both entry parameters and selected exit rules.
- `sq sweep` can evaluate ATR-based exits.
  - `atr_14` is available in the research and signal frame.
  - active strategies can promote `trailing_stop_atr_mult` and `profit_target_atr_mult`.
- `sq sweep` applies configurable execution costs from `config.yaml`.
  - Current defaults are `5 bps` slippage per side and `0 bps` commission per side.
  - Sweep metrics are net of those costs.
- `sq evaluate` now includes:
  - overall ranked candidates
  - best practical live candidates
  - best candidate per sector
  - best live candidate per sector
  - heuristic two-model portfolio pair suggestions
- `sq promote` enforces promotion quality floors from `config.yaml`.
  - Current defaults require minimum profit factor, expectancy, trade count, and maximum drawdown before a row can be promoted.

## Outputs

- Market history: `data/market_data.duckdb`
- Ledger: `data/ledger.sqlite`
- Ranked evaluation report: `reports/candidates.md`
- Active runtime strategy: `production_strategy.json`
- Logs: `logs/swingquant.log`

## Verification

Run the test suite:

```bash
python3 -m unittest discover -s tests -v
```

Compile-check the source tree:

```bash
python3 -m compileall src
```
