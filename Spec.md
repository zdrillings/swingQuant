

I'll include those as a placeholder you can adjust, clearly marked.
Technical Specification: Project SwingQuant (v1.3)
1. Core Principles & Logic

    Chronological Integrity: All data splits must be time-series based. 70% oldest data for Training; 30% most recent for Validation. No random shuffling, ever.
    Survivorship Bias Note: The universe is point-in-time biased toward current S&P 1500 survivors. Backtest results should be interpreted conservatively — live performance will likely trail backtest figures.
    Signal Gate: A ticker passes the signal gate only when:
        - all hard filters defined in the active production_strategy.json are satisfied on the most recent completed trading day
        - the total confluence score across the promoted score components is greater than or equal to signal_score_min
    Regime Catch-all:a
        QQQ Regime: Information Technology, Communication Services
        SPY Regime: ALL other sectors (Healthcare, Staples, Utilities, Industrials, Financials, Energy, Materials, Real Estate, Consumer Discretionary, Consumer Staples)

2. Configuration Files
2.1 .env (Secrets & Capital Parameters)

GMAIL_USER="your_email@gmail.com"
GMAIL_APP_PASSWORD="xxxx xxxx xxxx xxxx"
RECIPIENT_EMAIL="your_email@gmail.com"
TOTAL_CAPITAL=50000     # Current portfolio size
RISK_PER_TRADE=0.02     # 2% risk per trade ($1,000 at $50k)

2.2 config.yaml (Feature Definitions & Sweep Grid)

features:
  trend:
    - name: "sma_200_dist"
      type: "pct_diff"
      params: { window: 200 }
    - name: "atr_14"
      type: "atr"
      params: { window: 14 }
  momentum:
    - name: "rsi_14"
      type: "rsi"
      params: { window: 14 }
    - name: "rsi_2"
      type: "rsi"
      params: { window: 2 }
    - name: "roc_63"
      type: "roc"
      params: { window: 63 }
    - name: "relative_strength_index_vs_spy"
      ticker: "SPY"
      type: "relative_strength_percentile"
      params: { window: 63 }
  volume:
    - name: "vol_alpha"
      type: "ratio_to_avg"
      params: { window: 20 }
  commodities:
    - name: "oil_corr_60"
      ticker: "USO"
      type: "correlation"
      params: { window: 60 }

signal_model:
  default_pass_score: 30
  hard_filter_indicators:
    - "relative_strength_index_vs_spy_min"
  score_spans:
    rsi_14_max: 30
    vol_alpha_min: 1.2
    sma_200_dist_min: 0.25
    roc_63_min: 0.20

backtest_costs:
  slippage_bps_per_side: 5
  commission_bps_per_side: 0

# ADJUST THESE RANGES before running sq sweep.
# Current defaults assume a mean-reversion / oversold-bounce strategy.
sweep_grid:
  rsi_14_max:
    min: 25
    max: 60
    step: 5
  vol_alpha_min:
    min: 0.8
    max: 1.8
    step: 0.2
  sma_200_dist_min:
    min: 0.0
    max: 0.20
    step: 0.01
  roc_63_min:
    min: 0.10
    max: 0.10
    step: 0.05
  relative_strength_index_vs_spy_min:
    min: 80
    max: 80
    step: 5
  signal_score_min:
    min: 30
    max: 30
    step: 5
  trailing_stop_atr_mult:
    min: 2.0
    max: 3.0
    step: 0.5
  profit_target_atr_mult:
    min: 2.5
    max: 4.0
    step: 0.5

2.3 production_strategy.json / production_strategies.json (Runtime Schema — written by sq promote)

{
  "strategy_id": 402,
  "promoted_at": "2025-01-15T17:00:00",
  "indicators": {
    "rsi_14_max": 45,
    "vol_alpha_min": 1.2,
    "sma_200_dist_min": 0.18,
    "roc_63_min": 0.10,
    "relative_strength_index_vs_spy_min": 80,
    "signal_score_min": 30
  },
  "exit_rules": {
    "trailing_stop_pct": null,
    "profit_target_pct": null,
    "time_limit_days": 20,
    "trailing_stop_atr_mult": 2.5,
    "profit_target_atr_mult": 3.0
  }
}

3. Database Architecture

DuckDB: historical_ohlcv — (ticker, date, open, high, low, close, volume, adj_close)

SQLite Ledger:
Table 	Columns
Universe 	ticker, sector, is_active, md_volume_30d
Backtest_Results 	id, strategy_id, params_json, norm_score, profit_factor, expectancy, mdd, win_rate
Active_Trades 	ticker, entry_date, entry_price, entry_atr, shares, max_price_seen, status, exit_date, exit_price

Active_Trades.status lifecycle: 'open' → set by sq trade buy. 'closed' → set by sq trade sell. sq monitor is alert-only and recommends actions in its digest but does not close trades automatically.
4. CLI Command Specifications
sq sync

    Universe: Scrape S&P 500, 400, and 600 from Wikipedia if Universe table is empty. Populate ticker, sector, is_active.
    Fetch: 5 years of daily OHLCV via yfinance for all universe tickers plus $USO, $CPER, $GLD, $SPY, $QQQ. Use a 3-retry decorator with exponential backoff. On permanent failure, log the ticker and set is_active = False.
    Liquidity Filter: After fetch, calculate md_volume_30d = median(close * volume, last 30 days). Drop any ticker below $20M. Store the calculated value in Universe.
    Storage: Upsert to DuckDB — fetch only missing dates (idempotent).

sq research

    Scope: Top 250 tickers by md_volume_30d.
    Label: success = 1 if close[t+20] > close[t] * 1.05, else 0.
    Model: Train XGBoost Classifier on the first 70% of dates; validate on the last 30%.
    Output: Feature Importance table printed to console, ranked by Information Gain.

sq sweep

    Scope: Top 250 tickers by md_volume_30d.
    Grid: Read entry and exit parameter ranges from sweep_grid block in config.yaml. Run all combinations.
    Engine: Vectorized backtesting using Polars or Backtesting.py. VectorBT is explicitly prohibited.
    Validation: Chronological 70/30 split. Record all runs to Backtest_Results.
    Costs: Apply the configured backtest execution costs to every simulated trade so reported metrics are net of slippage and commissions.

sq evaluate [--top X] [--sector NAME]

    Normalization: Apply Min-Max scaling (0–1) to Expectancy, ProfitFactor, and MaxDrawdown within the current result set.
    Score: norm_score = (NormExpectancy * 0.4) + (NormProfitFactor * 0.3) - (NormMaxDrawdown * 0.3)
    Output: /reports/candidates.md with ranked results, parameter sets, and TradingView chart links.

sq promote --id [ID]

    Read the row from Backtest_Results by id.
    Write a fully-formed production_strategy.json matching the schema in Section 2.3.
    If a slot is supplied, update that slot inside production_strategies.json without overwriting other active slots.
    Include promoted_at timestamp.
    Print confirmation including the resolved slot.

sq trade [buy|sell] <ticker> <price> [shares]

    Buy: Insert a new row into Active_Trades with status = 'open', entry_date = today, entry_price = price, entry_atr = active ATR if ATR exits are in use, strategy_id / strategy_slot when resolvable, shares = shares, max_price_seen = price.
    Sell: Update the matching open row: set status = 'closed', exit_date = today, exit_price = price. Print realized P&L: (exit_price - entry_price) * shares.

sq scan (Cron: 5:00 PM EST)

    Price Source: Use today's adjusted close (the most recent completed session, available at 5 PM EST).
    Regime Filter: Load production_strategy.json. Check SPY > 200 SMA (for SPY-regime sectors) and QQQ > 200 SMA (for QQQ-regime sectors). If the relevant regime is Red for a given ticker's sector, skip that ticker entirely.
    Signal:
        - apply the hard filters first
        - compute confluence score across the promoted score components (`rsi_14`, `vol_alpha`, `sma_200_dist`, `roc_63`)
        - keep the ticker only when signal_score >= signal_score_min
    Output: Send "Evening Brief" HTML email containing Top 5 signal tickers, sector, regime status, and share count per the sizing formula.

sq monitor (Cron: Hourly, 10:30 AM–3:30 PM EST)

    Price Source: Fetch intraday last-trade price via yfinance using download(period='1d', interval='1m').iloc[-1].close.
    Watchlist: All rows in Active_Trades where status = 'open'.
    Trigger condition: current_price > yesterday_high (breakout alert).
    Exit checks: For every open trade, evaluate all four exit conditions (see Section 5). If any condition is met, flag it in the digest with the recommended action.
    Digest: Send ONE consolidated email per hourly run listing all triggered tickers and any exit flags. Do not send if the watchlist is empty and no exits were triggered.

5. Exit Logic (Applied in sq monitor)

    Protective (Trailing Stop): Exit if current_price falls below the active trailing stop threshold. Use max_price_seen * (1 - trailing_stop_pct) for percent exits, or max_price_seen - (entry_atr * trailing_stop_atr_mult) for ATR exits. Update max_price_seen on every run if current_price is a new high.
    Objective (Profit Target): Exit if current_price reaches the active target threshold. Use entry_price * (1 + profit_target_pct) for percent exits, or entry_price + (entry_atr * profit_target_atr_mult) for ATR exits. RSI_2 > 90 still forces an exit.
    Temporal (Time Limit): Exit if trading_days_since(entry_date) > time_limit_days.
    Systemic (Regime Flip): If the relevant regime ETF (SPY or QQQ) crosses below its 200 SMA, flag ALL open Active_Trades for immediate exit in the digest regardless of other conditions.

All thresholds (trailing_stop_pct, profit_target_pct, trailing_stop_atr_mult, profit_target_atr_mult, time_limit_days) are read from the active production_strategy.json at runtime.
6. Position Sizing

Shares = floor((TOTAL_CAPITAL * RISK_PER_TRADE) / stop_risk_per_share)

    TOTAL_CAPITAL and RISK_PER_TRADE from .env
    stop_risk_per_share = Price * trailing_stop_pct for percent exits
    stop_risk_per_share = entry_atr * trailing_stop_atr_mult for ATR exits
    Price = adjusted close of the most recent completed session

Example: ($50,000 × 0.02) / ($100 × 0.05) = 200 shares
7. Verification Checklist for Coding Agent

    [ ] Are all secrets and capital parameters sourced from .env — never hardcoded?
    [ ] Does sq sync batch requests and handle failed tickers with retry + is_active = False?
    [ ] Are all time-series splits strictly chronological (no random shuffling)?
    [ ] Does sq evaluate apply Min-Max normalization before computing norm_score?
    [ ] Does the sq scan signal model enforce relative_strength_index_vs_spy_min as a hard filter and use signal_score_min as the confluence pass rule?
    [ ] Does sq scan use today's adjusted close (not yesterday's) for pricing?
    [ ] Does sq monitor send a single consolidated digest per run (not per ticker)?
    [ ] Does sq monitor evaluate all four exit conditions on every open trade each run?
    [ ] Does sq sweep use Polars or Backtesting.py — no VectorBT?
    [ ] Do non-tech sectors default to the SPY regime filter?
    [ ] Does sq promote write a fully-formed production_strategy.json including promoted_at?
    [ ] Does sq trade sell update status, exit_date, exit_price, and print P&L?
