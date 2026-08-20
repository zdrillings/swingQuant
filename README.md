# SwingQuant

SwingQuant is a CLI-driven swing-trading research and operations toolkit built around DuckDB for market history, SQLite for the trading ledger, and a single `sq` command surface.

## Scope

Implemented commands:

- `sq init-db`
- `sq sync`
- `sq refresh-universe`
- `sq research`
- `sq alpha-research`
- `sq sweep`
- `sq sweep --mode <mode>` — 5 archetype modes: `pullback_technology`, `pullback_real_economy`, `breakout_growth`, `post_earnings_drift`, `trend_continuation`
- `sq evaluate`
- `sq sleeve-research`
- `sq promote --id <ID> [--slot <name>]`
- `sq trade buy <ticker> <price> [shares]`
- `sq trade sell <ticker> <price>`
- `sq positions`
- `sq quote <ticker>`
- `sq scan`
- `sq shortlist-model [--target-type regression|classification] [--model-scope global|sector_specific|regime_specific]`
- `sq shortlist-bakeoff`
- `sq shortlist` — render latest persisted model-driven shortlist
- `sq shortlist-scoreboard`
- `sq analyst-snapshot`
- `sq scan-backfill`
- `sq scan-performance`
- `sq scan-analysis`
- `sq portfolio-rotation`
- `sq universe-backfill`
- `sq universe-analysis`
- `sq factor-tearsheet`
- `sq exit-analysis`
- `sq rsi-exit-bakeoff`
- `sq subindustry-attribution`
- `sq slot-attribution`
- `sq monitor`

## Architecture

- DuckDB stores historical OHLCV in `historical_ohlcv`.
- DuckDB also stores analytical point-in-time history:
  - `universe_daily_snapshots` — daily features, forward outcomes, macro context, binary target
  - `analyst_snapshots`
  - `analyst_revision_snapshots`
  - `extended_hours_snapshots`
- SQLite stores:
  - `Universe`
  - `Backtest_Results`
  - `Active_Trades`
  - `Earnings_Calendar`
  - `Scan_Candidates`
  - `Shortlist_Model_Runs`
  - `Shortlist_Model_Predictions`
- The main code lives in [src](/home/zdrillings/code/SwingQuant/src).
- Tests live in [tests](/home/zdrillings/code/SwingQuant/tests).

Key modules:

- [src/utils/db_manager.py](/home/zdrillings/code/SwingQuant/src/utils/db_manager.py): schema creation and data access
- [src/sync/service.py](/home/zdrillings/code/SwingQuant/src/sync/service.py): universe bootstrap and OHLCV sync
- [src/research/service.py](/home/zdrillings/code/SwingQuant/src/research/service.py): feature training and importance reporting
- [src/research/shortlist_model_service.py](/home/zdrillings/code/SwingQuant/src/research/shortlist_model_service.py): walk-forward shortlist model bakeoff with signal-proxy, ridge, lasso, and optional XGBoost candidates
- [src/research/universe_snapshot_service.py](/home/zdrillings/code/SwingQuant/src/research/universe_snapshot_service.py): daily snapshot backfill with macro features and binary target
- [src/sweep/service.py](/home/zdrillings/code/SwingQuant/src/sweep/service.py): parameter grid backtests with Polars, hard stop in exit chain
- [src/evaluate/service.py](/home/zdrillings/code/SwingQuant/src/evaluate/service.py): result normalization and report generation
- [src/scan/service.py](/home/zdrillings/code/SwingQuant/src/scan/service.py): daily post-close signal scan with model-driven selection, quality gate, and confidence-based position sizing
- [src/utils/shortlist_runtime.py](/home/zdrillings/code/SwingQuant/src/utils/shortlist_runtime.py): live model context loading with confidence metrics
- [src/monitor/service.py](/home/zdrillings/code/SwingQuant/src/monitor/service.py): intraday breakout and exit monitoring (includes hard stop)
- [src/quote/service.py](/home/zdrillings/code/SwingQuant/src/quote/service.py): latest quote and holding context
- [src/utils/strategy.py](/home/zdrillings/code/SwingQuant/src/utils/strategy.py): ExitRules with hard_stop_pct, entry_stop_price, strategy resolution

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

3. Sweep parameter combinations (5 archetype modes):

```bash
./sq sweep --mode pullback_technology       # Tech pullback-with-trend
./sq sweep --mode pullback_real_economy     # Energy/Materials/Industrials/Financials pullback
./sq sweep --mode breakout_growth           # Breakout entries (Tech/Industrials/Financials)
./sq sweep --mode post_earnings_drift       # Post-earnings continuation (Tech/Health Care)
./sq sweep --mode trend_continuation        # Simple trend-following (broad)
```

Each mode sweeps entry thresholds, exit parameters, and `hard_stop_pct` (5%–10% entry-anchored stop-loss).

4. Train the shortlist model:

```bash
# Regression (predicts continuous alpha_vs_sector_20d)
./sq shortlist-model --eligible-universe-mode passed_or_trend --model-scope sector_specific

# Classification (predicts binary alpha_vs_sector_20d_pos: alpha > 2%)
./sq shortlist-model --target-type classification --model-scope regime_specific

# Quick bakeoff to compare policies
./sq shortlist-bakeoff --eligible-universe-mode passed_or_trend
```

The shortlist model uses an L1-regularized linear model (Lasso for regression, LogisticRegression for classification) trained in walk-forward across ~372 OOS dates with 43 features plus cross-sectional ranks. Features include 4 macro context features (`spy_roc_20`, `spy_roc_5`, `spy_realized_vol_20`, `qqq_roc_20`) computed from SPY/QQQ price history. Zero-coefficient features are purged and the model is re-fit per fold.

Model scopes:
- `global` — single model across all sectors
- `sector_specific` — per-sector models with global fallback (production default)
- `regime_specific` — per-regime models (trending/choppy/correcting) via `spy_roc_20`

5. Rank candidates:

```bash
./sq evaluate --top 10
./sq evaluate --run-id 17 --top 20
./sq evaluate --run-id 31 --top 20 --walk-forward
./sq sleeve-research --top 10
./sq sleeve-research --top 10 --walk-forward
```

`--walk-forward` runs a second-pass rolling validation analysis on a shortlist of candidates instead of the entire sweep. It reports:
- `wf_stability_score`
- `wf_positive_window_ratio`
- `wf_positive_alpha_window_ratio`
- `wf_median_expectancy`
- `wf_worst_expectancy`
- `wf_worst_mdd`

Use `--walk-forward-windows` and `--walk-forward-shortlist` to control runtime. The implementation is fixed-parameter rolling validation, not per-window re-optimization.

`sq sleeve-research` runs a separate research path:
- sector breadth filters
- within-sector daily ranking
- fixed holding horizons
- sleeve-level equal-weight portfolio simulation with max open positions = `top_n`
- low-sample trade penalties in practical scoring
- distinct-ticker support and concentration penalties
- a `Best Live Configurations With Enough Sample` section using the configured trade-support floor

Current sleeve defaults are intentionally stricter than the first prototype:
- horizons are narrowed to `5d`, `10d`, and `15d`
- the pre-rank filter now requires stronger RS / ROC / volume quality
- `rsi_14` must stay in a moderate band instead of accepting any pullback
- headline sections require live breadth as well as support

When `--walk-forward` is enabled, `sq sleeve-research` also computes a second-pass rolling stability section on a bounded shortlist of sleeve configurations. Use:
- `--walk-forward-windows`
- `--walk-forward-shortlist`

It writes [sleeve_research.md](/home/zdrillings/code/SwingQuant/reports/sleeve_research.md).

6. Promote one or more strategies (promoted strategies include `hard_stop_pct` in exit rules):

```bash
./sq promote --id 611572 --slot materials
./sq promote --id 401181 --slot energy
./sq promote --id 622015 --slot industrials
```

7. Run end-of-day scan (model-driven with quality gate):

```bash
./sq scan
```

The scan uses the shortlist model's **top-2 predictions** as its product. A rank-calibration analysis (Aug 2026) showed the model's ranks 1-2 carried all the recent alpha (+6.7% mean, 70% beat rate) while ranks 3-10 had inverted to negative — so the model path selects at most 2 candidates.

Quality gate (2 tiers, evaluated on the top-2 basket's recent 20d walk-forward performance):
- **Full (2 picks)**: beat_rate >= 35% AND mean_target >= -3%
- **Minimal (1 pick)**: beat_rate < 35% or mean_target < -3%
- **Heuristic fallback (75% of cap, floor 3)**: diagnostic only; production model scans fail closed unless `allow_heuristic_fallback: true` is explicitly configured

Rotation exclusion: when the effective cap is below the configured total, tickers picked in the last 3 scan dates are removed from the selection pool, so picks rotate through the model's top candidates instead of repeating the same name every day.

When the model context is stale, failing promotion, unavailable, or produces zero mapped candidates, production `sq scan` aborts instead of silently emitting heuristic picks. That makes nightly failures noisy and keeps the evening brief from presenting non-model picks as model-driven output.

8. Capture analyst target snapshots for future point-in-time research:

```bash
./sq analyst-snapshot --source research --top 250
```

This persists one row per ticker/provider/date in DuckDB `analyst_snapshots`. Rows are written even when the provider has no analyst target for a ticker, so later analysis can distinguish "captured but unavailable" from "not captured." The command is idempotent for a given `snapshot_date` and provider; rerunning the same date replaces that date's rows.

Use `--ticker` to always include specific names outside the default source:

```bash
./sq analyst-snapshot --ticker SLAB --ticker LITE
```

9. Run intraday monitoring (includes hard stop checks):

```bash
./sq monitor
```

9. Capture extended-hours movement after the close:

```bash
./sq extended-hours-snapshot --source all
```

`sq extended-hours-snapshot` stores point-in-time postmarket movement in DuckDB for future analysis. It requests 1-minute intraday data with pre/post-market bars, anchors each ticker to the regular-session close, and persists extended-hours return, sector ETF extended-hours return, relative extended-hours return, postmarket volume, and timestamp metadata. The evening brief reads the current day's persisted snapshot when available, but selection remains close-based until enough history exists to test the signal.

10. Sync the local ledger from Schwab positions:

```bash
./sq schwab sync-ledger --dry-run
./sq schwab sync-ledger
```

`sq schwab sync-ledger` is read-only against Schwab. It pulls broker positions and updates only the local SQLite ledger so `Active_Trades` matches the broker source of truth:

- broker-only ticker -> open a local ledger trade using Schwab average price and share count
- ledger-only ticker -> warn by default; pass `--close-missing` to close the local ledger trade using the latest available price approximation
- ticker in both -> update local shares, average price, and max price seen when Schwab differs
- fund/ETF positions are ignored by default; pass `--include-funds` only if you intentionally want ETF holdings in the trading ledger
- use `--ignore-ticker TICKER` for a known broker/reporting discrepancy that should not be opened, updated, or closed locally

11. Check a current quote and holding context:

```bash
./sq quote VSH
```

12. Record fills manually in the ledger:

```bash
./sq trade buy DOW 53.25 --slot materials
./sq trade buy EOG 118.40 --slot energy
./sq trade buy FIX 392.15 --slot industrials
./sq trade sell AAPL 192.40
```

## Operational Rules

- All train/validation splits are chronological.
- `sq sync` now refreshes both OHLCV and earnings calendar dates for the research universe.
- `sq analyst-snapshot` captures current analyst target, recommendation, estimate, and revision context into DuckDB for future point-in-time studies.
  - Default scope is the top 250 active names by `md_volume_30d`.
  - The current provider is yfinance.
  - The command stores target mean, median, low, high, analyst count, recommendation summary, estimate/revision tables, capture timestamp, and details JSON.
- `sq shortlist-model` trains the prediction engine:
  - L1-regularized linear model (Lasso for regression, LogisticRegression for classification).
  - Walk-forward validation with expanding windows (252 min train, 20-day test).
  - 47 raw features (incl. `rsi_2`, `ret_1d`, `ret_5d`, `close_vs_20d_low` mean-reversion features) plus cross-sectional ranks (`__rank_all`, `__rank_sector`) and sector dummies.
  - 4 macro context features computed from SPY/QQQ: `spy_roc_20`, `spy_roc_5`, `spy_realized_vol_20`, `qqq_roc_20`.
  - Binary target (`alpha_vs_sector_20d_pos`) = 1 if `alpha_vs_sector_20d > 2%`, else 0.
  - Zero-coefficient features are purged and the model is re-fit per fold.
  - `regime_specific` scope splits by SPY 20d return regime (trending when >1% else choppy).
- `sq scan` candidate selection:
  - **Primary**: model-driven — inner-joins snapshot with live model predictions; no signal gate needed.
  - **Fallback**: heuristic — per-slot signal gate when model returns empty predictions.
  - **Confidence basket**: quality gate metrics (`recent_20d_beat_rate`, `recent_20d_mean_target`) are computed on the model's **top-2 predictions** — rank calibration showed ranks 3+ inverted recently (see rank analysis in AGENTS.md).
  - **Quality gate** (model path, 2 tiers): 2 picks when beat_rate >= 35% and mean_target >= -3%; else 1 pick.
  - **Rotation exclusion**: picks from the last 3 scan dates are removed from the pool when the cap is below the configured total, so the pair rotates through the model's top candidates.
  - Heuristic fallback: 75% of cap, floor 3.
  - A hard relative-strength filter via `relative_strength_index_vs_spy_min`
  - A confluence score across the promoted score components
  - `signal_score_min` as the pass threshold
  - scored components currently include `rsi_14`, `vol_alpha`, `sma_200_dist`, `roc_63`, and sector-specific signals such as `oil_corr_60`
  - `vol_alpha` is currently downweighted relative to the other score components
  - earnings-aware strategies can additionally hard-filter on:
    - `days_to_next_earnings`
    - `days_since_last_earnings`
  - gap-aware strategies can additionally score:
    - `avg_abs_gap_pct_20`
    - `max_gap_down_pct_60`
  - liquidity/volatility regime features are available for research and model ranking:
    - `atr_pct_14_percentile_252`
    - `realized_vol_20_percentile_252`
    - `dollar_volume_ratio_20_60`
    - `volume_percentile_60`
    - `distance_from_52w_high`
    - `days_since_52w_high`
  - position sizing uses the promoted stop model, including ATR-based stops when present
  - multiple active strategy slots are evaluated independently, then merged under `scan_policy` caps from `config.yaml`
  - the current scan policy limits total ideas and also caps per-slot and per-sector concentration
  - the final shortlist is model-ranked when a promoted shortlist model is available
  - very low opportunity candidates can be filtered out before selection
  - the evening brief emphasizes current best bets, existing holdings, top unheld targets, and analyst target context when available
- Non-tech sectors default to the SPY regime; Information Technology and Communication Services use QQQ.
- `sq monitor` is alert-only.
  - It updates `max_price_seen`.
  - It evaluates all exit rules every run, in priority order:
    1. **Regime flip** — exit if benchmark below 200 SMA.
    2. **Hard stop** — exit if `current_price <= entry_price * (1 - hard_stop_pct)`. Must-sell, not demotable by shock guard.
    3. **Trailing stop** — ATR-based or percent-based from max price seen.
    4. **Profit target** — ATR-based or percent-based from entry.
    5. **RSI_2 > 90** — requires minimum unrealized gain.
    6. **Time limit** — exit after `time_limit_days`.
    7. **Pre-earnings exit** — exit ahead of upcoming earnings.
  - Hard stop, profit target, and pre-earnings exit are **must-sell** (not demoted by the portfolio shock guard).
  - It uses the latest available quote for pricing-sensitive checks.
  - Earnings-aware strategies can trigger a `pre_earnings_exit` flag ahead of the next scheduled report.
  - RSI_2 exits can require a minimum unrealized gain before triggering.
  - It sends one consolidated digest.
  - It does not close `Active_Trades`; use `sq trade sell` to close trades in the ledger.
  - Legacy imported holdings without `strategy_slot` now fall back to the best available exact-sector or regime-family strategy and are backfilled into the ledger.
- `sq schwab sync-ledger` is the broker reconciliation path.
  - It never submits orders or changes Schwab positions.
  - It should run before `sq monitor` so monitor alerts evaluate the broker-truth holdings.
  - Exact historical sell fill prices are not available from the positions endpoint; missing broker positions are warning-only unless `--close-missing` is explicitly passed.

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
- `sq sweep --mode high_performance_energy_stability_refined` is the robustness-focused Energy follow-up that narrows around the current and older high-alpha Energy families to improve positive-window ratio and worst-window drawdown rather than just raw expectancy.
- `sq sweep --mode high_performance_energy_earnings_refined` adds entry blackout and pre-earnings exit testing to that refined Energy sleeve.
- `sq sweep --mode high_performance_materials_refined` is the narrower Materials follow-up search centered between the current live/promotable row and the stronger alpha rows.
- `sq sweep --mode high_performance_materials_earnings_refined` adds entry blackout and pre-earnings exit testing to the refined Materials sleeve.
- `sq sweep --mode high_performance_materials`, `high_performance_industrials`, and `high_performance_financials` run the same broader real-economy template one sector at a time.
- `sq sweep --mode high_performance_industrials_refined` is the narrower Industrials follow-up search centered between the current live/promotable row and the stronger alpha rows.
- `sq sweep --mode high_performance_industrials_earnings_refined` adds entry blackout and pre-earnings exit testing to the refined Industrials sleeve.
- `sq sweep --mode high_performance_real_economy` runs the broader real-economy template across Energy, Materials, Industrials, and Financials in one pass.
- `sq sweep --mode production_sleeves_earnings_refined` runs one bounded earnings-aware pass across the three current production sleeves: Energy, Materials, and Industrials.
- `sq sweep --mode production_sleeves_gap_refined` runs one bounded gap-risk pass across the three current production sleeves: Energy, Materials, and Industrials.
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
  - active strategies can also promote `exit_before_earnings_days`.
- `sq sweep` can now evaluate earnings timing controls.
  - `days_to_next_earnings_min` expresses the pre-earnings entry blackout.
  - `days_since_last_earnings_min` can keep entries away from immediate post-event noise.
  - `exit_before_earnings_days` forces simulated exits ahead of upcoming earnings.
  - `0` disables the earnings filter or event-exit axis inside the targeted sweep modes.
- `sq sweep` can now evaluate overnight gap-risk controls.
  - `avg_abs_gap_pct_20_max` limits average absolute overnight gap behavior over the last 20 sessions.
  - `max_gap_down_pct_60_max` limits the worst downside overnight gap over the last 60 sessions.
  - these are score components, not hard filters, so the research can decide whether lower gap-risk really improves the sleeve.
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
  - Promoted strategies include `hard_stop_pct` in their `exit_rules`.
- `sq sweep` exit rule priority (in backtest simulation):
  1. Regime flip → close
  2. **Hard stop** → `entry_price * (1 - hard_stop_pct)` — entry-anchored, checked before trailing stop
  3. Trailing stop → ATR-based or percent-based from max price
  4. Profit target → ATR-based or percent-based from entry
  5. RSI_2 > 90 → close
  6. Time limit → close
  7. Pre-earnings exit → close
  - All sweep metrics are net of configured execution costs and hard stops.

## Nightly Pipeline

The nightly pipeline runs via `ops/nightly_pipeline.sh` and executes in order:

1. `./sq sync` — refresh OHLCV and earnings data
2. `./sq universe-backfill` — compute today's features with macro context and binary target
3. `./sq shortlist-model` — train the walk-forward shortlist model and persist only a champion that passes the promotion gate
4. `./sq analyst-snapshot` — capture analyst targets
5. `./sq extended-hours-snapshot` — capture postmarket movement
6. `./sq scan` — produce evening brief with quality-gated candidates
7. `./sq scan-performance --all-sources --email` — email performance summary across currently persisted selection sources

If any step fails, the pipeline sends a failure notification email via the `notify_failure` trap. The pipeline also refuses to run when code/config paths have uncommitted changes, so nightly results can be tied back to committed logic.

## Suggested Trading-Day Schedule

Run Schwab ledger reconciliation once shortly after the market opens. Schwab positions are treated as the morning source of truth, including closing local ledger positions missing from Schwab; intraday fills can still be entered manually when needed.

```cron
TZ=America/New_York

# Broker-truth ledger sync: 5 minutes after open.
35 9 * * 1-5 cd /home/zdrillings/code/SwingQuant && ./sq schwab sync-ledger --close-missing >> logs/schwab-ledger-sync.log 2>&1

# Existing hourly monitor runs after the morning ledger sync.
30 10-15 * * 1-5 cd /home/zdrillings/code/SwingQuant && ./sq monitor >> logs/monitor.log 2>&1

# Postmarket snapshot before the evening brief, then the close-based scan with postmarket context.
25 19 * * 1-5 cd /home/zdrillings/code/SwingQuant && ./sq extended-hours-snapshot --source all >> logs/extended_hours_snapshot_cron.log 2>&1
30 19 * * 1-5 cd /home/zdrillings/code/SwingQuant && ./sq scan >> logs/scan_cron.log 2>&1
```

## Current Model State

The shortlist layer evaluates multiple walk-forward candidate models and persists the selected champion. Current candidates are:

- `signal_proxy`: deterministic rank-average baseline
- `ridge_model`: closed-form linear baseline
- `lasso_model`: L1-regularized linear model, or logistic classification when `--target-type classification` is used
- `xgboost_model`: optional tree model when `xgboost` is installed

Key characteristics:

- **Target**: `alpha_vs_sector_20d` or `alpha_vs_sector_20d_pos`; missing forward alpha remains missing and is not converted into a negative label
- **Selection**: the champion is selected from model summaries, not hardcoded
- **Runtime override**: `scan_policy.shortlist_model.production_model_name`, when set, explicitly selects a preferred model; when omitted, runtime uses the persisted champion
- **No runtime retraining**: `sq scan` loads persisted model runs and predictions; model training belongs in `sq shortlist-model`
- **Promotion gate**: model-driven scan picks are unavailable unless recent top-2 OOS metrics pass `scan_policy.shortlist_model.promotion_gate`
- **Fail-closed production scan**: when `scan_policy.shortlist_model.use_as_candidate_source` is enabled, missing/stale/failing model context aborts the scan unless `allow_heuristic_fallback: true` is explicitly configured
- **Quality throttle**: after the model passes the promotion gate, position count can still scale down based on recent model confidence
- **Hard stops**: entry-anchored stop-loss checked before trailing stops in both backtest and live monitoring

## Outputs

- Market history: `data/market_data.duckdb`
- Ledger: `data/ledger.sqlite`
- Model report: `reports/shortlist_model.md`
- Model predictions: `reports/shortlist_model_oos_predictions.csv`, `reports/shortlist_model_live_predictions.csv`
- Scan performance: `reports/scan_performance.md`
- Ranked evaluation report: `reports/candidates.md`
- Active runtime strategies: `production_strategies.json`
- Logs: `logs/swingquant.log`, `logs/nightly_pipeline.log`

## Try Next

1. Run the nightly pipeline to refresh the model and scan:

```bash
./sq universe-backfill --date-from $(date +%F)
./sq shortlist-model --eligible-universe-mode passed_or_trend --model-scope sector_specific
./sq scan
```

2. Test with classification target and regime splitting (slower but potentially more adaptive):

```bash
./sq shortlist-model --target-type classification --model-scope regime_specific
```

3. Sweep new strategy parameters:

```bash
./sq sweep --mode pullback_technology
./sq sweep --mode breakout_growth
./sq evaluate --top 10
```

4. Monitor holdings with hard stops active:

```bash
./sq monitor
```

## Verification

Run the test suite:

```bash
python3 -m unittest discover -s tests -v
```

Compile-check the source tree:

```bash
python3 -m compileall src
```
