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
./sq sweep --mode low_drawdown_technology
./sq sweep --mode promotable_live_technology
./sq sweep --mode promotable_live_technology_v2
./sq sweep --mode promotable_live_technology_v3
./sq sweep --mode promotable_live_technology_v4
./sq sweep --mode promotable_live_technology_v5
./sq sweep --mode high_performance_energy
./sq sweep --mode high_performance_energy_refined
./sq sweep --mode high_performance_materials
./sq sweep --mode high_performance_materials_refined
./sq sweep --mode high_performance_industrials
./sq sweep --mode high_performance_industrials_refined
./sq sweep --mode high_performance_financials
./sq sweep --mode high_performance_real_economy
./sq sweep --mode breakout_v1_information_technology
./sq sweep --mode breakout_v1_industrials
./sq sweep --mode breakout_v1_financials
./sq sweep --mode breakout_v1_growth_leaders
```

4. Rank candidates:

```bash
./sq evaluate --top 10
./sq evaluate --run-id 17 --top 20
```

5. Promote one or more strategies:

```bash
./sq promote --id 611572 --slot materials
./sq promote --id 401181 --slot energy
./sq promote --id 622015 --slot industrials
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
./sq trade buy EOG 118.40 --slot energy
./sq trade buy FIX 392.15 --slot industrials
./sq trade sell AAPL 192.40
```

## Operational Rules

- All train/validation splits are chronological.
- `sq scan` uses:
  - a hard relative-strength filter via `relative_strength_index_vs_spy_min`
  - a confluence score across the promoted score components
  - `signal_score_min` as the pass threshold
  - scored components currently include `rsi_14`, `vol_alpha`, `sma_200_dist`, `roc_63`, and sector-specific signals such as `oil_corr_60`
  - `vol_alpha` is currently downweighted relative to the other score components
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
- `sq sweep --mode low_drawdown_technology` restricts the search to Information Technology with a tighter, lower-drawdown grid.
- `sq sweep --mode promotable_live_technology` targets the narrower gap between promotable and currently-live technology setups.
- `sq sweep --mode promotable_live_technology_v2` is the smallest tech tuning loop and mainly relaxes `signal_score_min` while keeping the lower-drawdown exit shape fixed.
- `sq sweep --mode promotable_live_technology_v3` is the midpoint tech loop with `signal_score_min` fixed at `31` to test the exact boundary between promotable-only and live-but-too-risky.
- `sq sweep --mode promotable_live_technology_v4` is a compact frontier search that keeps `signal_score_min = 30` and tests stronger relative strength plus a slightly tighter ATR target.
- `sq sweep --mode promotable_live_technology_v5` keeps the live-capable tech entries and tests tighter ATR stops as the main drawdown-reduction lever.
- `sq sweep --mode high_performance_energy` is a large Energy-only search that adds `oil_corr_60_min` to favor names moving with the oil complex while still demanding strong relative strength and trend quality.
- `sq sweep --mode high_performance_energy_refined` is the narrower follow-up Energy search centered on the current promotable/live cluster, so iteration is cheaper and more targeted.
- `sq sweep --mode high_performance_materials_refined` is the narrower Materials follow-up search centered between the current live/promotable row and the stronger alpha rows.
- `sq sweep --mode high_performance_materials`, `high_performance_industrials`, and `high_performance_financials` run the same broader real-economy template one sector at a time.
- `sq sweep --mode high_performance_industrials_refined` is the narrower Industrials follow-up search centered between the current live/promotable row and the stronger alpha rows.
- `sq sweep --mode high_performance_real_economy` runs the broader real-economy template across Energy, Materials, Industrials, and Financials in one pass.
- `sq sweep --mode breakout_v1_information_technology`, `breakout_v1_industrials`, and `breakout_v1_financials` run the new breakout-specific model family.
- `sq sweep --mode breakout_v1_information_technology_v2` keeps the breakout freshness cap and lowers the breakout score threshold slightly for a tighter A/B test on current tech setup density.
- `sq sweep --mode breakout_v1_information_technology_v3` keeps the stronger v2 score threshold and slightly widens the breakout freshness band to test whether one extra degree of extension unlocks live setups without reverting to late-chase names.
- `sq sweep --mode breakout_v1_information_technology_v4` restores the stricter freshness band from v2 and lowers only the RS floor to test whether relative-strength gating is the last barrier to current tech setups.
- `sq sweep --mode breakout_v1_growth_leaders` runs the breakout family across Information Technology, Industrials, and Financials in one pass.
- The breakout v1 family is intentionally separate from the pullback family.
  - Breakout modes now replace the base sweep grid instead of inheriting it.
  - Trend hard filters: `close > 50d > 200d`, positive 50d slope, positive 200d slope, breakout above the prior 20-day high, capped distance above the prior 20-day high, and RS percentile.
  - Score components: `roc_63`, `rsi_14_min`, `sma_200_dist_max`, `base_range_pct_20_max`, `base_atr_contraction_20_max`, `base_volume_dryup_ratio_20_max`, and `breakout_volume_ratio_50_min`.
  - Runtime path is the same once promoted: `sq scan` can surface candidates and `sq monitor` can manage open trades under the promoted ATR exits.
  - Breakout v1 is now intentionally constrained for runtime sanity:
    - per-sector modes sweep `64` parameter combinations
    - `breakout_v1_growth_leaders` runs `64` combinations across `3` sectors for `192` sector-runs total
    - only these axes are swept in v1: `relative_strength_index_vs_spy_min`, `distance_above_20d_high_max`, `base_range_pct_20_max`, `breakout_volume_ratio_50_min`, `signal_score_min`, and `trailing_stop_atr_mult`
    - the other breakout design choices are fixed in config so the first research pass finishes quickly enough to iterate
  - Current breakout v1 tuning is intentionally stricter:
    - higher RS floor
    - freshness cap above the breakout trigger
    - tighter base range
    - stronger breakout volume requirement
    - tighter ATR stop
    - goal: surface cleaner, earlier-stage breakouts instead of already-extended leaders
  - Current breakout v1 next-try option:
    - `breakout_v1_information_technology_v2` lowers only `signal_score_min` while preserving the freshness cap, so it can test whether the model is “good but too strict” without reopening the late-chase problem.
    - `breakout_v1_information_technology_v3` then widens only `distance_above_20d_high_max`, so it can test whether current setup scarcity is coming from freshness rather than score strictness.
    - `breakout_v1_information_technology_v4` restores the tighter freshness cap and lowers only `relative_strength_index_vs_spy_min`, so it can test whether RS is the final bottleneck after freshness and score have been tuned.
- `sq sweep` can evaluate ATR-based exits.
  - `atr_14` is available in the research and signal frame.
  - active strategies can promote `trailing_stop_atr_mult` and `profit_target_atr_mult`.
- `sq sweep` applies configurable execution costs from `config.yaml`.
  - Current defaults are `5 bps` slippage per side and `0 bps` commission per side.
  - Sweep metrics are net of those costs.
- `sq sweep` now stores benchmark-relative trade alpha when benchmark history exists locally.
  - `alpha_vs_spy` compares each simulated trade against `SPY` over the same holding window.
  - `alpha_vs_sector` compares each simulated trade against the mapped sector ETF such as `XLB`, `XLE`, `XLI`, `XLK`, or `XLF`.
  - This is computed as average excess return per trade, not regression alpha.
  - `alpha_vs_sector` will remain `unknown` until the relevant sector ETF history has been synced into DuckDB.
- `sq evaluate` now includes:
  - overall ranked candidates
  - best practical live candidates
  - best candidate per sector
  - best promotable candidate per sector
  - best live candidate per sector
  - heuristic promotable two-model portfolio pair suggestions
  - `alpha_vs_spy` and `alpha_vs_sector` on each row when available
  - practical ranking now includes modest alpha bonuses, so excess-return models can outrank otherwise similar sector-beta clones
  - live-match and gate diagnostics are cached by sector + indicator signature to reduce repeated evaluation work on large runs
- `sq promote` enforces promotion quality floors from `config.yaml`.
  - Current defaults require minimum profit factor, expectancy, trade count, and maximum drawdown before a row can be promoted.

## Outputs

- Market history: `data/market_data.duckdb`
- Ledger: `data/ledger.sqlite`
- Ranked evaluation report: `reports/candidates.md`
- Active runtime strategies: `production_strategies.json`
- Logs: `logs/swingquant.log`

## Try Next

Suggested operator sequence while the new breakout family is being tuned:

1. Run the new Energy / real-economy family:

```bash
./sq sweep --mode high_performance_energy
./sq sweep --mode high_performance_energy_refined
./sq evaluate --sector "Energy" --top 10
```

Then, for the other active sleeves:

```bash
./sq sweep --mode high_performance_materials_refined
./sq evaluate --sector "Materials" --top 10

./sq sweep --mode high_performance_industrials_refined
./sq evaluate --sector "Industrials" --top 10
```

If you want alpha metrics to populate fully first, refresh the benchmark ETF history before the next sweep:

```bash
./sq sync
```

Then rerun the sweep and evaluate steps. Without synced sector ETFs, `alpha_vs_spy` can populate but `alpha_vs_sector` may still show as `unknown`.

2. Run the breakout v1 family in the three target sectors:

```bash
./sq sweep --mode breakout_v1_growth_leaders
./sq evaluate --top 20
```

Expected runtime envelope after the breakout grid reduction:
- `breakout_v1_information_technology`, `breakout_v1_industrials`, `breakout_v1_financials`: small enough for fast single-sector iteration
- `breakout_v1_growth_leaders`: a compact three-sector pass, no longer a multi-day run

3. If you want to inspect sectors individually:

```bash
./sq sweep --mode breakout_v1_information_technology
./sq sweep --mode breakout_v1_industrials
./sq sweep --mode breakout_v1_financials
```

4. If breakout v1 gets live matches but weak promotion metrics, the first levers to revisit are:
   - `breakout_volume_ratio_50_min`
   - `signal_score_min`
   - `base_range_pct_20_max`
   - `trailing_stop_atr_mult`

5. If breakout names are still showing up too late in the move, the next model feature to add is breakout freshness:
   - distance above prior 20-day high
   - or days-since-breakout

## Verification

Run the test suite:

```bash
python3 -m unittest discover -s tests -v
```

Compile-check the source tree:

```bash
python3 -m compileall src
```
