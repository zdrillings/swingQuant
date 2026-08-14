from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

from src.settings import get_settings, load_feature_config
from src.sync.market_data import MarketDataClient, chunked, extract_ticker_history
from src.sync.service import REFERENCE_TICKERS
from src.utils.db_manager import DatabaseManager
from src.utils.emailer import send_html_email
from src.utils.logging import get_logger
from src.utils.regime import regime_etf_for_sector
from src.utils.shortlist_runtime import load_live_shortlist_model_context
from src.utils.signal_engine import build_analysis_frame, latest_rsi_2_with_intraday, overlay_price_history
from src.utils.strategy import (
    entry_stop_price,
    load_active_strategies,
    profit_target_price,
    resolve_trade_strategy,
    rsi_2_exit_triggered,
    summarize_buy_setup,
    trailing_stop_price,
)


@dataclass(frozen=True)
class MonitorReport:
    watchlist_size: int
    triggered_count: int
    emailed: bool


class MonitorService:
    def __init__(
        self,
        db_manager: DatabaseManager,
        *,
        market_data_client: MarketDataClient | None = None,
        email_sender=send_html_email,
    ) -> None:
        self.db_manager = db_manager
        self.market_data_client = market_data_client or MarketDataClient()
        self.email_sender = email_sender
        self.logger = get_logger("monitor")

    def run(self) -> MonitorReport:
        self.db_manager.initialize()
        strategies = load_active_strategies()
        settings = get_settings()
        monitor_policy = self._load_sell_signal_policy()
        open_trades = self.db_manager.list_open_trades()
        if not open_trades:
            return MonitorReport(watchlist_size=0, triggered_count=0, emailed=False)

        trade_tickers = [row["ticker"] for row in open_trades]
        monitor_tickers = trade_tickers + ["SPY", "QQQ"]
        intraday_prices = self._load_intraday_last_prices(monitor_tickers)
        price_sources = {ticker: "intraday" for ticker in intraday_prices}
        history_tickers = sorted(set(trade_tickers).union(["SPY", "QQQ"]))
        base_history = self.db_manager.load_price_history(history_tickers)
        earnings_loader = getattr(self.db_manager, "load_earnings_calendar", None)
        earnings_calendar = earnings_loader(trade_tickers) if callable(earnings_loader) else pd.DataFrame()
        recent_history = self._download_recent_daily_history(history_tickers)
        price_history = overlay_price_history(base_history, recent_history)
        history_prices = self._latest_prices_from_history(
            price_history=price_history,
            tickers=[ticker for ticker in monitor_tickers if ticker not in intraday_prices],
        )
        for ticker, price in history_prices.items():
            intraday_prices[ticker] = price
            price_sources[ticker] = "latest close"
        universe_rows = self.db_manager.list_universe_rows(active_only=False)
        analysis_frame, _ = build_analysis_frame(
            price_history,
            universe_rows,
            earnings_calendar=earnings_calendar,
        )
        latest_snapshot = analysis_frame.sort_values(["ticker", "date"]).groupby("ticker", as_index=False).tail(1)
        sector_map = {row["ticker"]: row["sector"] for row in universe_rows}
        latest_rows_by_ticker = latest_snapshot.set_index("ticker").to_dict(orient="index")
        shortlist_model_context = self._load_shortlist_model_context()
        shortlist_predictions = (
            shortlist_model_context.live_predictions.set_index("ticker").to_dict(orient="index")
            if shortlist_model_context is not None and not shortlist_model_context.live_predictions.empty
            else {}
        )

        holding_rows = []
        for trade in open_trades:
            trade_row = self.db_manager.get_latest_open_trade(trade["ticker"])
            current_price = intraday_prices.get(trade["ticker"])
            price_source = price_sources.get(trade["ticker"], "unavailable")
            resolution = resolve_trade_strategy(
                trade=trade,
                strategies=strategies,
                sector_map=sector_map,
                backtest_lookup=getattr(self.db_manager, "get_backtest_result_by_strategy_id", None),
            )
            strategy = resolution.strategy
            if strategy is None:
                self.logger.warning("Including trade with unresolved strategy for manual review: ticker=%s", trade["ticker"])
                time_in_trade = self._trading_sessions_held(
                    price_history=price_history,
                    ticker=str(trade["ticker"]),
                    entry_date=str(trade["entry_date"]),
                    as_of=date.today(),
                )
                holding_rows.append(
                    self._manual_review_row(
                        trade=trade,
                        current_price=current_price,
                        price_source=price_source,
                        resolution_source=resolution.source,
                        time_in_trade=time_in_trade,
                    )
                )
                continue
            ticker_sector = sector_map.get(trade["ticker"]) or strategy.sector
            if (
                trade_row is not None
                and hasattr(self.db_manager, "assign_trade_strategy")
                and self._should_backfill_strategy_assignment(trade)
            ):
                self.db_manager.assign_trade_strategy(
                    int(trade_row["rowid"]),
                    strategy_id=int(strategy.strategy_id),
                    strategy_slot=strategy.slot,
                )
            yesterday_high = self._load_yesterday_high(trade["ticker"])
            max_price_seen = float(trade["max_price_seen"])
            if current_price is not None:
                max_price_seen = max(max_price_seen, current_price)
            if trade_row is not None and max_price_seen > float(trade["max_price_seen"]):
                self.db_manager.update_trade_max_price(int(trade_row["rowid"]), max_price_seen)
            has_intraday_price = current_price is not None and price_source == "intraday"
            latest_ticker_row = latest_rows_by_ticker.get(trade["ticker"], {})
            entry_atr = self._row_value(trade, "entry_atr")
            if (entry_atr is None or pd.isna(entry_atr)) and pd.notna(latest_ticker_row.get("atr_14")):
                entry_atr = float(latest_ticker_row["atr_14"])

            regime_etf = regime_etf_for_sector(ticker_sector)
            regime_price = intraday_prices.get(regime_etf)
            benchmark_return_since_entry = self._return_since_entry(
                price_history=price_history,
                ticker=regime_etf,
                entry_date=str(trade["entry_date"]),
                current_price=regime_price,
            )
            regime_row = latest_snapshot.loc[latest_snapshot["ticker"] == regime_etf]
            regime_green = True
            if not regime_row.empty and regime_price is not None:
                sma_column = "qqq_sma_200" if regime_etf == "QQQ" else "spy_sma_200"
                sma_value = regime_row.iloc[-1][sma_column]
                regime_green = pd.notna(sma_value) and regime_price >= float(sma_value)

            rsi_2 = (
                latest_rsi_2_with_intraday(
                    price_history=price_history,
                    ticker=trade["ticker"],
                    current_price=current_price,
                    as_of=date.today(),
                )
                if has_intraday_price
                else float("nan")
            )
            time_in_trade = self._trading_sessions_held(
                price_history=price_history,
                ticker=str(trade["ticker"]),
                entry_date=str(trade["entry_date"]),
                as_of=date.today(),
            )
            try:
                stop_price = trailing_stop_price(
                    max_price_seen=max_price_seen,
                    entry_atr=float(entry_atr) if entry_atr is not None else None,
                    exit_rules=strategy.exit_rules,
                )
            except ValueError as exc:
                self.logger.warning("Unable to calculate stop for %s: %s", trade["ticker"], exc)
                stop_price = None
            try:
                target_price = profit_target_price(
                    entry_price=float(trade["entry_price"]),
                    entry_atr=float(entry_atr) if entry_atr is not None else None,
                    exit_rules=strategy.exit_rules,
                )
            except ValueError as exc:
                self.logger.warning("Unable to calculate target for %s: %s", trade["ticker"], exc)
                target_price = None
            buy_setup_status, buy_setup_note = summarize_buy_setup(
                strategy_indicators=strategy.indicators,
                latest_row=latest_ticker_row,
            )
            model_prediction = shortlist_predictions.get(str(trade["ticker"]), {})
            buy_setup_status, buy_setup_note = self._model_buy_setup_context(
                ticker=str(trade["ticker"]),
                fallback_status=buy_setup_status,
                fallback_note=buy_setup_note,
                model_prediction=model_prediction,
                shortlist_model_context=shortlist_model_context,
            )
            hard_stop_price = entry_stop_price(
                entry_price=float(trade["entry_price"]),
                exit_rules=strategy.exit_rules,
            )
            exit_flags = {
                "hard_stop": bool(
                    current_price is not None
                    and hard_stop_price is not None
                    and current_price <= hard_stop_price
                ),
                "trailing_stop": bool(
                    current_price is not None
                    and stop_price is not None
                    and current_price < stop_price
                ),
                "profit_target": bool(
                    target_price is not None
                    and (
                        (has_intraday_price and current_price >= target_price)
                        or (not has_intraday_price and max_price_seen >= target_price)
                    )
                ),
                "rsi_2": rsi_2_exit_triggered(
                    rsi_2=rsi_2,
                    unrealized_pct=(
                        ((current_price / float(trade["entry_price"])) - 1.0)
                        if current_price is not None
                        else None
                    ),
                    days_in_trade=time_in_trade,
                    strategy_slot=strategy.slot,
                ),
                "time_limit": time_in_trade > strategy.exit_rules.time_limit_days,
                "regime_flip": not regime_green,
                "pre_earnings_exit": (
                    strategy.exit_rules.exit_before_earnings_days is not None
                    and pd.notna(latest_ticker_row.get("days_to_next_earnings"))
                    and float(latest_ticker_row["days_to_next_earnings"]) <= float(strategy.exit_rules.exit_before_earnings_days)
                ),
            }
            setup_now = self._summarize_setup_now(
                buy_setup_status=buy_setup_status,
                buy_setup_note=buy_setup_note,
            )
            unrealized_pct = (
                ((current_price / float(trade["entry_price"])) - 1.0)
                if current_price is not None
                else None
            )
            relative_alpha_since_entry = (
                unrealized_pct - benchmark_return_since_entry
                if unrealized_pct is not None and benchmark_return_since_entry is not None
                else None
            )
            action_context = self._classify_sell_signal(
                exit_flags=exit_flags,
                unrealized_pct=unrealized_pct,
                relative_alpha_since_entry=relative_alpha_since_entry,
                buy_setup_status=buy_setup_status,
                policy=monitor_policy,
            )
            main_risk = self._summarize_main_risk(
                recommended_action=action_context["recommended_action"],
                exit_flags=exit_flags,
                regime_green=regime_green,
                distance_to_stop_pct=(
                    ((current_price / stop_price) - 1.0)
                    if current_price is not None and stop_price is not None and stop_price > 0
                    else None
                ),
                days_to_next_earnings=(
                    int(float(latest_ticker_row["days_to_next_earnings"]))
                    if pd.notna(latest_ticker_row.get("days_to_next_earnings"))
                    else None
                ),
            )
            price_context = self._summarize_price_context(
                distance_to_stop_pct=(
                    ((current_price / stop_price) - 1.0)
                    if current_price is not None and stop_price is not None and stop_price > 0
                    else None
                ),
                distance_to_target_pct=(
                    ((target_price / current_price) - 1.0)
                    if current_price is not None and target_price is not None and current_price > 0
                    else None
                ),
                target_touched=bool(
                    target_price is not None
                    and not has_intraday_price
                    and max_price_seen >= target_price
                ),
            )
            holding_rows.append(
                {
                    "strategy_source": resolution.source,
                    "strategy_slot": strategy.slot,
                    "ticker": trade["ticker"],
                    "sector": ticker_sector or "Unknown",
                    "entry_price": float(trade["entry_price"]),
                    "current_price": current_price,
                    "price_source": price_source,
                    "unrealized_pct": unrealized_pct,
                    "benchmark_ticker": regime_etf,
                    "benchmark_return_since_entry": benchmark_return_since_entry,
                    "relative_alpha_since_entry": relative_alpha_since_entry,
                    "recommended_action": action_context["recommended_action"],
                    "action_tier": action_context["action_tier"],
                    "action_basis": action_context["action_basis"],
                    "raw_exit": action_context["raw_exit"],
                    "chart_link": f"https://www.tradingview.com/chart/?symbol={trade['ticker']}",
                    "stop_price": stop_price,
                    "target_price": target_price,
                    "distance_to_stop_pct": (
                        ((current_price / stop_price) - 1.0)
                        if current_price is not None and stop_price is not None and stop_price > 0
                        else None
                    ),
                    "distance_to_target_pct": (
                        ((target_price / current_price) - 1.0)
                        if current_price is not None and target_price is not None and current_price > 0
                        else None
                    ),
                    "time_in_trade": time_in_trade,
                    "days_to_next_earnings": (
                        int(float(latest_ticker_row["days_to_next_earnings"]))
                        if pd.notna(latest_ticker_row.get("days_to_next_earnings"))
                        else None
                    ),
                    "hard_stop": exit_flags["hard_stop"],
                    "trailing_stop": exit_flags["trailing_stop"],
                    "profit_target": exit_flags["profit_target"],
                    "rsi_2": exit_flags["rsi_2"],
                    "time_limit": exit_flags["time_limit"],
                    "regime_flip": exit_flags["regime_flip"],
                    "pre_earnings_exit": exit_flags["pre_earnings_exit"],
                    "buy_setup_status": "still valid" if buy_setup_status == "yes" else ("not valid" if buy_setup_status == "no" else "unknown"),
                    "buy_setup_note": buy_setup_note,
                    "setup_now": setup_now,
                    "main_risk": main_risk,
                    "price_context": price_context,
                }
            )

        if not holding_rows:
            return MonitorReport(watchlist_size=len(open_trades), triggered_count=0, emailed=False)

        self._apply_portfolio_shock_guard(holding_rows, policy=monitor_policy)
        holding_rows.sort(
            key=lambda row: (
                0 if row["recommended_action"] == "sell" else 1,
                0 if row["recommended_action"] == "review" else 1,
                row["days_to_next_earnings"] if row["days_to_next_earnings"] is not None else 9_999,
                row["ticker"],
            )
        )
        triggered_rows = [row for row in holding_rows if row["recommended_action"] == "sell"]
        html = self._build_digest_html(holding_rows=holding_rows, triggered_rows=triggered_rows)
        self.email_sender(subject="Hourly Monitor Digest", html_body=html, settings=settings)
        return MonitorReport(
            watchlist_size=len(open_trades),
            triggered_count=len(triggered_rows),
            emailed=True,
        )

    def _load_shortlist_model_context(self):
        try:
            config = load_feature_config()
            shortlist_model_config = (
                config.get("scan_policy", {}).get("shortlist_model", {})
                if isinstance(config, dict)
                else {}
            )
            preferred_model_name = shortlist_model_config.get("production_model_name")
            eligible_universe_mode = shortlist_model_config.get(
                "production_eligible_universe_mode",
                shortlist_model_config.get("eligible_universe_mode", "passed_only"),
            )
            model_scope = shortlist_model_config.get("production_model_scope", "global")
            xgboost_config = shortlist_model_config.get("production_xgboost_config", "baseline")
            return load_live_shortlist_model_context(
                self.db_manager,
                preferred_model_name=str(preferred_model_name) if preferred_model_name not in (None, "") else None,
                eligible_universe_mode=str(eligible_universe_mode or "passed_only"),
                model_scope=str(model_scope or "global"),
                xgboost_config=str(xgboost_config or "baseline"),
            )
        except Exception as exc:
            self.logger.warning("Unable to load shortlist model context in monitor: %s", exc)
            return None

    def _load_sell_signal_policy(self) -> dict[str, float | int]:
        config = load_feature_config()
        payload = config.get("monitor_policy", {}).get("sell_signal", {}) if isinstance(config, dict) else {}
        return {
            "max_loss_pct": float(payload.get("max_loss_pct", 0.15)),
            "relative_underperformance_pct": float(payload.get("relative_underperformance_pct", 0.05)),
            "portfolio_shock_trigger_pct": float(payload.get("portfolio_shock_trigger_pct", 0.50)),
            "portfolio_shock_min_triggered": int(payload.get("portfolio_shock_min_triggered", 3)),
        }

    def _classify_sell_signal(
        self,
        *,
        exit_flags: dict[str, bool],
        unrealized_pct: float | None,
        relative_alpha_since_entry: float | None,
        buy_setup_status: str,
        policy: dict[str, float | int],
    ) -> dict[str, object]:
        raw_exit = any(
            exit_flags[name]
            for name in ("hard_stop", "trailing_stop", "profit_target", "rsi_2", "time_limit", "regime_flip", "pre_earnings_exit")
        )
        max_loss_pct = float(policy["max_loss_pct"])
        relative_underperformance_pct = float(policy["relative_underperformance_pct"])
        setup_invalid = buy_setup_status == "no"
        severe_relative_underperformance = (
            relative_alpha_since_entry is not None
            and relative_alpha_since_entry <= -relative_underperformance_pct
        )
        if unrealized_pct is not None and unrealized_pct <= -max_loss_pct:
            return {
                "recommended_action": "sell",
                "action_tier": "must-sell",
                "action_basis": f"capital floor: unrealized {unrealized_pct:.1%} <= -{max_loss_pct:.1%}",
                "raw_exit": raw_exit,
            }
        if exit_flags.get("hard_stop"):
            return {
                "recommended_action": "sell",
                "action_tier": "must-sell",
                "action_basis": "hard stop breached",
                "raw_exit": raw_exit,
            }
        if exit_flags.get("profit_target"):
            return {
                "recommended_action": "sell",
                "action_tier": "must-sell",
                "action_basis": "profit target reached",
                "raw_exit": raw_exit,
            }
        if exit_flags.get("pre_earnings_exit"):
            return {
                "recommended_action": "sell",
                "action_tier": "must-sell",
                "action_basis": "pre-earnings exit window",
                "raw_exit": raw_exit,
            }
        if exit_flags.get("trailing_stop"):
            if severe_relative_underperformance:
                return {
                    "recommended_action": "sell",
                    "action_tier": "sell",
                    "action_basis": f"stop breached with relative alpha {relative_alpha_since_entry:.1%}",
                    "raw_exit": raw_exit,
                }
            if setup_invalid:
                return {
                    "recommended_action": "sell",
                    "action_tier": "sell",
                    "action_basis": "stop breached and setup invalidated",
                    "raw_exit": raw_exit,
                }
            return {
                "recommended_action": "review",
                "action_tier": "review",
                "action_basis": "stop breached, but benchmark-relative damage/setup does not confirm",
                "raw_exit": raw_exit,
            }
        if exit_flags.get("regime_flip"):
            return {
                "recommended_action": "review",
                "action_tier": "review",
                "action_basis": "regime flip; review exposure before selling",
                "raw_exit": raw_exit,
            }
        if exit_flags.get("time_limit"):
            if setup_invalid or severe_relative_underperformance:
                return {
                    "recommended_action": "sell",
                    "action_tier": "sell",
                    "action_basis": "time limit plus weak current setup/relative performance",
                    "raw_exit": raw_exit,
                }
            return {
                "recommended_action": "review",
                "action_tier": "review",
                "action_basis": "time limit reached, setup/relative performance not broken",
                "raw_exit": raw_exit,
            }
        if exit_flags.get("rsi_2"):
            return {
                "recommended_action": "review",
                "action_tier": "review",
                "action_basis": "RSI_2 profit-taking signal; confirm manually",
                "raw_exit": raw_exit,
            }
        return {
            "recommended_action": "hold",
            "action_tier": "hold",
            "action_basis": "no confirmed sell signal",
            "raw_exit": raw_exit,
        }

    def _apply_portfolio_shock_guard(self, holding_rows: list[dict], *, policy: dict[str, float | int]) -> None:
        raw_triggered = [row for row in holding_rows if row.get("raw_exit")]
        if not holding_rows:
            return
        shock_trigger_pct = float(policy["portfolio_shock_trigger_pct"])
        min_triggered = int(policy["portfolio_shock_min_triggered"])
        if len(raw_triggered) < min_triggered:
            return
        if (len(raw_triggered) / len(holding_rows)) < shock_trigger_pct:
            return
        for row in holding_rows:
            if row.get("recommended_action") != "sell":
                continue
            if row.get("action_tier") == "must-sell":
                continue
            if row.get("trailing_stop") and row.get("relative_alpha_since_entry") is not None:
                if float(row["relative_alpha_since_entry"]) > -float(policy["relative_underperformance_pct"]):
                    row["recommended_action"] = "review"
                    row["action_tier"] = "portfolio-shock-review"
                    row["action_basis"] = (
                        "portfolio shock guard: stop breached during broad selloff, "
                        "relative damage not severe"
                    )

    def _return_since_entry(
        self,
        *,
        price_history: pd.DataFrame,
        ticker: str,
        entry_date: str,
        current_price: float | None,
    ) -> float | None:
        if current_price is None or price_history.empty or not {"ticker", "date"}.issubset(price_history.columns):
            return None
        price_column = "adj_close" if "adj_close" in price_history.columns else "close"
        if price_column not in price_history.columns:
            return None
        entry = pd.Timestamp(entry_date).normalize()
        ticker_history = price_history.loc[price_history["ticker"].astype(str) == str(ticker)].copy()
        if ticker_history.empty:
            return None
        ticker_history["date"] = pd.to_datetime(ticker_history["date"], errors="coerce").dt.normalize()
        ticker_history = ticker_history.dropna(subset=["date", price_column]).sort_values("date")
        entry_rows = ticker_history.loc[ticker_history["date"] >= entry]
        if entry_rows.empty:
            return None
        entry_price = float(entry_rows.iloc[0][price_column])
        if entry_price <= 0:
            return None
        return (float(current_price) / entry_price) - 1.0

    def _download_recent_daily_history(self, tickers: list[str]) -> pd.DataFrame:
        frames = []
        start_date = date.today() - timedelta(days=450)
        try:
            for ticker_batch in chunked(tickers, 50):
                raw_batch = self.market_data_client.download_daily_history(ticker_batch, start_date)
                for ticker in ticker_batch:
                    history = extract_ticker_history(raw_batch, ticker)
                    if not history.empty:
                        frames.append(history)
        except Exception as exc:
            self.logger.warning("Falling back to DuckDB-only daily history in monitor: %s", exc)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _load_intraday_last_prices(self, tickers: list[str]) -> dict[str, float]:
        prices: dict[str, float] = {}
        try:
            raw_frame = self.market_data_client.download_intraday_history(tickers)
        except Exception as exc:
            self.logger.warning("Unable to load intraday prices in monitor: %s", exc)
            return prices
        for ticker in tickers:
            history = extract_ticker_history(raw_frame, ticker)
            if history.empty:
                continue
            prices[ticker] = float(history.iloc[-1]["close"])
        return prices

    def _load_yesterday_high(self, ticker: str) -> float | None:
        rows = self.db_manager.load_recent_highs(ticker, limit=2)
        if len(rows.index) < 2:
            return None
        return float(rows.iloc[1]["high"])

    def _build_digest_html(self, *, holding_rows: list[dict], triggered_rows: list[dict]) -> str:
        sell_count = len(triggered_rows)
        review_rows = [row for row in holding_rows if row["recommended_action"] == "review"]
        review_count = len(review_rows)
        valid_count = sum(1 for row in holding_rows if row["buy_setup_status"] == "still valid")
        hold_count = len(holding_rows) - sell_count - review_count
        sell_table = self._build_digest_table(triggered_rows) if triggered_rows else "<p>No current sell signals.</p>"
        review_table = self._build_digest_table(review_rows) if review_rows else "<p>No review-only signals.</p>"
        holdings_table = self._build_digest_table(holding_rows)
        return (
            "<html><body>"
            "<h1>Hourly Monitor Digest</h1>"
            f"<p>Holdings: {len(holding_rows)} | Must-sell/sell: {sell_count} | Review: {review_count} | Holds: {hold_count} | Still-valid setups: {valid_count}</p>"
            "<p><strong>How to read this:</strong> "
            "<em>Trade Action</em> and <em>Exit Reasons</em> apply to your existing position. "
            "<em>Fresh Setup</em> answers whether we would newly buy the stock today. "
            "Trailing stops are classified with capital-loss, benchmark-relative, setup-validity, and portfolio-shock context.</p>"
            "<h2>Sell Now</h2>"
            f"{sell_table}"
            "<h2>Review Before Selling</h2>"
            f"{review_table}"
            "<h2>All Holdings</h2>"
            f"{holdings_table}"
            "</body></html>"
        )

    def _build_digest_table(self, rows: list[dict]) -> str:
        html_rows = []
        for row in rows:
            exit_flags = [
                name.replace("_", " ")
                for name in ("hard_stop", "trailing_stop", "profit_target", "rsi_2", "time_limit", "regime_flip", "pre_earnings_exit")
                if row[name]
            ]
            exit_context = []
            if row["stop_price"] is not None:
                exit_context.append(f"stop {row['stop_price']:.2f}")
            if row["target_price"] is not None:
                exit_context.append(f"target {row['target_price']:.2f}")
            if row["distance_to_stop_pct"] is not None:
                exit_context.append(f"{row['distance_to_stop_pct'] * 100.0:+.1f}% vs stop")
            if row["distance_to_target_pct"] is not None:
                exit_context.append(f"{row['distance_to_target_pct'] * 100.0:+.1f}% to target")
            exit_context.append(f"price source: {row['price_source']}")
            exit_context.append(f"{int(row['time_in_trade'])} sessions in trade")
            if row["time_limit"]:
                exit_context.append(f"time limit exceeded: {int(row['time_in_trade'])} sessions")
            if row["days_to_next_earnings"] is not None:
                exit_context.append(f"{int(row['days_to_next_earnings'])} bd to earnings")
            html_rows.append(
                "<tr>"
                f"<td>{row['strategy_slot']}</td>"
                f"<td>{row['strategy_source']}</td>"
                f"<td>{row['ticker']}</td>"
                f"<td>{row['sector']}</td>"
                f"<td>{row['entry_price']:.2f}</td>"
                f"<td>{self._fmt_optional_price(row['current_price'])}</td>"
                f"<td>{row['price_source']}</td>"
                f"<td>{self._fmt_optional_pct(row['unrealized_pct'])}</td>"
                f"<td>{row['recommended_action']}</td>"
                f"<td>{row.get('action_tier', '')}</td>"
                f"<td>{row.get('action_basis', '')}</td>"
                f"<td>{row.get('benchmark_ticker', '-')}</td>"
                f"<td>{self._fmt_optional_pct(row.get('relative_alpha_since_entry'))}</td>"
                f"<td>{', '.join(exit_flags) if exit_flags else 'none'}</td>"
                f"<td>{row['setup_now']}</td>"
                f"<td>{row['main_risk']}</td>"
                f"<td>{row['price_context']}</td>"
                f"<td>{row['buy_setup_note']}</td>"
                f"<td>{', '.join(exit_context)}</td>"
                f"<td><a href=\"{row['chart_link']}\">chart</a></td>"
                "</tr>"
            )
        return (
            "<table border='1' cellpadding='6' cellspacing='0'>"
            "<tr><th>Strategy Slot</th><th>Resolution</th><th>Ticker</th><th>Sector</th><th>Entry</th><th>Current</th><th>Price Source</th><th>P&L %</th><th>Trade Action</th><th>Tier</th><th>Action Basis</th><th>Benchmark</th><th>Alpha Since Entry</th><th>Exit Reasons</th><th>Fresh Setup</th><th>Main Risk</th><th>Price Context</th><th>Fresh Setup Note</th><th>Exit Context</th><th>Chart</th></tr>"
            f"{''.join(html_rows)}"
            "</table>"
        )

    def _summarize_setup_now(
        self,
        *,
        buy_setup_status: str,
        buy_setup_note: str,
    ) -> str:
        if buy_setup_status == "yes":
            return "buyable"
        if buy_setup_status == "no":
            if buy_setup_note.startswith("close:"):
                return "almost buyable"
            return "not buyable"
        return "unknown"

    def _model_buy_setup_context(
        self,
        *,
        ticker: str,
        fallback_status: str,
        fallback_note: str,
        model_prediction: dict,
        shortlist_model_context,
    ) -> tuple[str, str]:
        if not shortlist_model_context or not model_prediction:
            return fallback_status, fallback_note
        model_rank = model_prediction.get("model_rank")
        predicted_alpha = model_prediction.get("predicted_alpha")
        model_reason_summary = model_prediction.get("model_reason_summary")
        model_comparison_summary = model_prediction.get("model_comparison_summary")
        model_name = shortlist_model_context.champion_model
        top_n = int(shortlist_model_context.top_n)
        rank_value = int(model_rank) if pd.notna(model_rank) else None
        pred_value = float(predicted_alpha) if pd.notna(predicted_alpha) else None
        reason_suffix = f"; why: {model_reason_summary}" if model_reason_summary else ""
        comparison_suffix = f"; won over: {model_comparison_summary}" if model_comparison_summary else ""
        if rank_value is not None and rank_value <= top_n:
            return (
                "yes",
                f"model shortlist #{rank_value}; predicted alpha {pred_value:.2%} ({model_name}){reason_suffix}{comparison_suffix}",
            )
        if rank_value is not None and rank_value <= (top_n * 2):
            return (
                "no",
                f"close: model rank #{rank_value}; predicted alpha {pred_value:.2%} ({model_name}){reason_suffix}{comparison_suffix}",
            )
        if rank_value is not None:
            return (
                "no",
                f"not shortlisted: model rank #{rank_value}; predicted alpha {pred_value:.2%} ({model_name}){reason_suffix}{comparison_suffix}",
            )
        return fallback_status, fallback_note

    def _summarize_main_risk(
        self,
        *,
        recommended_action: str,
        exit_flags: dict[str, bool],
        regime_green: bool,
        distance_to_stop_pct: float | None,
        days_to_next_earnings: int | None,
    ) -> str:
        if recommended_action == "sell":
            reasons = [
                name.replace("_", " ")
                for name in ("hard_stop", "trailing_stop", "profit_target", "rsi_2", "time_limit", "regime_flip", "pre_earnings_exit")
                if exit_flags.get(name)
            ]
            return f"sell now: {', '.join(reasons)}"
        if recommended_action == "review":
            reasons = [
                name.replace("_", " ")
                for name in ("hard_stop", "trailing_stop", "profit_target", "rsi_2", "time_limit", "regime_flip", "pre_earnings_exit")
                if exit_flags.get(name)
            ]
            return f"review: {', '.join(reasons)}" if reasons else "review manually"
        if regime_green is False:
            return "red regime"
        if distance_to_stop_pct is not None and distance_to_stop_pct <= 0.03:
            return f"near stop ({distance_to_stop_pct * 100.0:.1f}% above)"
        if days_to_next_earnings is not None and days_to_next_earnings <= 2:
            return f"earnings in {days_to_next_earnings} bd"
        return "no immediate risk"

    def _summarize_price_context(
        self,
        *,
        distance_to_stop_pct: float | None,
        distance_to_target_pct: float | None,
        target_touched: bool = False,
    ) -> str:
        if target_touched:
            return "target already touched; no intraday price"
        if distance_to_target_pct is not None and distance_to_target_pct <= 0:
            return "through target"
        if distance_to_target_pct is not None and distance_to_target_pct <= 0.03:
            return f"near target ({distance_to_target_pct * 100.0:.1f}%)"
        if distance_to_stop_pct is not None and distance_to_stop_pct <= 0.03:
            return f"near stop ({distance_to_stop_pct * 100.0:.1f}%)"
        if distance_to_target_pct is not None:
            return f"{distance_to_target_pct * 100.0:.1f}% to target"
        return "target unavailable"

    def _trading_sessions_held(
        self,
        *,
        price_history: pd.DataFrame,
        ticker: str,
        entry_date: str,
        as_of: date,
    ) -> int:
        entry = pd.Timestamp(entry_date).normalize()
        end = pd.Timestamp(as_of).normalize()
        if not price_history.empty and {"ticker", "date"}.issubset(price_history.columns):
            ticker_history = price_history.loc[price_history["ticker"].astype(str) == str(ticker)].copy()
            if not ticker_history.empty:
                session_dates = pd.to_datetime(ticker_history["date"], errors="coerce").dt.normalize().dropna()
                if not session_dates.empty and session_dates.max() >= end:
                    held_sessions = session_dates[(session_dates > entry) & (session_dates <= end)].drop_duplicates()
                    return int(len(held_sessions.index))
        return max(len(pd.bdate_range(entry, end)) - 1, 0)

    @staticmethod
    def _fmt_optional_price(value: float | None) -> str:
        return "-" if value is None else f"{value:.2f}"

    @staticmethod
    def _fmt_optional_pct(value: float | None) -> str:
        return "-" if value is None else f"{value * 100.0:+.1f}%"

    def _latest_prices_from_history(self, *, price_history: pd.DataFrame, tickers: list[str]) -> dict[str, float]:
        prices: dict[str, float] = {}
        if price_history.empty or not tickers or not {"ticker", "date"}.issubset(price_history.columns):
            return prices
        price_column = "close" if "close" in price_history.columns else "adj_close"
        if price_column not in price_history.columns:
            return prices
        for ticker in sorted(set(str(value) for value in tickers)):
            ticker_history = price_history.loc[price_history["ticker"].astype(str) == ticker].copy()
            if ticker_history.empty:
                continue
            ticker_history["date"] = pd.to_datetime(ticker_history["date"], errors="coerce")
            ticker_history = ticker_history.dropna(subset=["date", price_column]).sort_values("date")
            if ticker_history.empty:
                continue
            price = float(ticker_history.iloc[-1][price_column])
            if pd.notna(price) and price > 0:
                prices[ticker] = price
        return prices

    def _manual_review_row(
        self,
        *,
        trade,
        current_price: float | None,
        price_source: str,
        resolution_source: str,
        time_in_trade: int,
    ) -> dict:
        entry_price = float(trade["entry_price"])
        ticker = str(trade["ticker"])
        return {
            "strategy_source": resolution_source,
            "strategy_slot": self._row_value(trade, "strategy_slot") or "unresolved",
            "ticker": ticker,
            "sector": "Unknown",
            "entry_price": entry_price,
            "current_price": current_price,
            "price_source": price_source,
            "unrealized_pct": ((current_price / entry_price) - 1.0) if current_price is not None else None,
            "recommended_action": "review",
            "action_tier": "review",
            "action_basis": "strategy unresolved",
            "raw_exit": False,
            "benchmark_ticker": "-",
            "benchmark_return_since_entry": None,
            "relative_alpha_since_entry": None,
            "chart_link": f"https://www.tradingview.com/chart/?symbol={ticker}",
            "stop_price": None,
            "target_price": None,
            "distance_to_stop_pct": None,
            "distance_to_target_pct": None,
            "time_in_trade": time_in_trade,
            "days_to_next_earnings": None,
            "hard_stop": False,
            "trailing_stop": False,
            "profit_target": False,
            "rsi_2": False,
            "time_limit": False,
            "regime_flip": False,
            "pre_earnings_exit": False,
            "buy_setup_status": "unknown",
            "buy_setup_note": "open ledger position has no resolvable active or historical strategy; review manually",
            "setup_now": "unknown",
            "main_risk": "manual review: strategy unresolved",
            "price_context": "exit rules not evaluated",
        }

    def _should_backfill_strategy_assignment(self, trade) -> bool:
        return self._row_value(trade, "strategy_slot") in (None, "") or self._row_value(trade, "strategy_id") is None

    def _row_value(self, row, key: str):
        if hasattr(row, "keys"):
            return row[key] if key in row.keys() else None
        if isinstance(row, dict):
            return row.get(key)
        try:
            return row[key]
        except Exception:
            return None
