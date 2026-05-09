from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

from src.settings import get_settings
from src.sync.market_data import MarketDataClient, chunked, extract_ticker_history
from src.sync.service import REFERENCE_TICKERS
from src.utils.db_manager import DatabaseManager
from src.utils.emailer import send_html_email
from src.utils.logging import get_logger
from src.utils.regime import regime_etf_for_sector
from src.utils.signal_engine import build_analysis_frame, latest_rsi_2_with_intraday, overlay_price_history
from src.utils.strategy import ProductionStrategy, load_active_strategies, profit_target_price, trailing_stop_price


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
        open_trades = self.db_manager.list_open_trades()
        if not open_trades:
            return MonitorReport(watchlist_size=0, triggered_count=0, emailed=False)

        trade_tickers = [row["ticker"] for row in open_trades]
        intraday_prices = self._load_intraday_last_prices(trade_tickers + ["SPY", "QQQ"])
        history_tickers = sorted(set(trade_tickers).union(["SPY", "QQQ"]))
        base_history = self.db_manager.load_price_history(history_tickers)
        recent_history = self._download_recent_daily_history(history_tickers)
        price_history = overlay_price_history(base_history, recent_history)
        universe_rows = self.db_manager.list_universe_rows(active_only=False)
        analysis_frame, _ = build_analysis_frame(price_history, universe_rows)
        latest_snapshot = analysis_frame.sort_values(["ticker", "date"]).groupby("ticker", as_index=False).tail(1)
        sector_map = {row["ticker"]: row["sector"] for row in universe_rows}
        latest_rows_by_ticker = latest_snapshot.set_index("ticker").to_dict(orient="index")

        triggered_rows = []
        for trade in open_trades:
            trade_row = self.db_manager.get_latest_open_trade(trade["ticker"])
            strategy = self._resolve_strategy_for_trade(
                trade=trade,
                strategies=strategies,
                sector_map=sector_map,
            )
            if strategy is None:
                self.logger.warning("Skipping trade with unresolved strategy: ticker=%s", trade["ticker"])
                continue
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
            current_price = intraday_prices.get(trade["ticker"])
            if current_price is None:
                continue
            yesterday_high = self._load_yesterday_high(trade["ticker"])
            max_price_seen = max(float(trade["max_price_seen"]), current_price)
            if trade_row is not None and max_price_seen > float(trade["max_price_seen"]):
                self.db_manager.update_trade_max_price(int(trade_row["rowid"]), max_price_seen)
            latest_ticker_row = latest_rows_by_ticker.get(trade["ticker"], {})
            entry_atr = self._row_value(trade, "entry_atr")
            if (entry_atr is None or pd.isna(entry_atr)) and pd.notna(latest_ticker_row.get("atr_14")):
                entry_atr = float(latest_ticker_row["atr_14"])

            regime_etf = regime_etf_for_sector(sector_map.get(trade["ticker"], ""))
            regime_price = intraday_prices.get(regime_etf)
            regime_row = latest_snapshot.loc[latest_snapshot["ticker"] == regime_etf]
            regime_green = True
            if not regime_row.empty and regime_price is not None:
                sma_column = "qqq_sma_200" if regime_etf == "QQQ" else "spy_sma_200"
                sma_value = regime_row.iloc[-1][sma_column]
                regime_green = pd.notna(sma_value) and regime_price >= float(sma_value)

            rsi_2 = latest_rsi_2_with_intraday(
                price_history=price_history,
                ticker=trade["ticker"],
                current_price=current_price,
                as_of=date.today(),
            )
            time_in_trade = len(pd.bdate_range(trade["entry_date"], date.today().isoformat())) - 1
            stop_price = trailing_stop_price(
                max_price_seen=max_price_seen,
                entry_atr=float(entry_atr) if entry_atr is not None else None,
                exit_rules=strategy.exit_rules,
            )
            target_price = profit_target_price(
                entry_price=float(trade["entry_price"]),
                entry_atr=float(entry_atr) if entry_atr is not None else None,
                exit_rules=strategy.exit_rules,
            )
            exit_flags = {
                "breakout_alert": yesterday_high is not None and current_price > yesterday_high,
                "trailing_stop": current_price < stop_price,
                "profit_target": current_price >= target_price,
                "rsi_2": rsi_2 > 90,
                "time_limit": time_in_trade > strategy.exit_rules.time_limit_days,
                "regime_flip": not regime_green,
            }
            should_exit = any(
                exit_flags[name]
                for name in ("trailing_stop", "profit_target", "rsi_2", "time_limit", "regime_flip")
            )
            if should_exit:
                recommended_action = "sell"
            elif exit_flags["breakout_alert"]:
                recommended_action = "watch breakout"
            else:
                recommended_action = "hold"

            if exit_flags["breakout_alert"] or should_exit:
                triggered_rows.append(
                    {
                        "strategy_slot": strategy.slot,
                        "ticker": trade["ticker"],
                        "current_price": current_price,
                        "recommended_action": recommended_action,
                        "breakout_alert": exit_flags["breakout_alert"],
                        "trailing_stop": exit_flags["trailing_stop"],
                        "profit_target": exit_flags["profit_target"],
                        "rsi_2": exit_flags["rsi_2"],
                        "time_limit": exit_flags["time_limit"],
                        "regime_flip": exit_flags["regime_flip"],
                    }
                )

        if not triggered_rows:
            return MonitorReport(watchlist_size=len(open_trades), triggered_count=0, emailed=False)

        html = self._build_digest_html(triggered_rows)
        self.email_sender(subject="Hourly Monitor Digest", html_body=html, settings=settings)
        return MonitorReport(
            watchlist_size=len(open_trades),
            triggered_count=len(triggered_rows),
            emailed=True,
        )

    def _download_recent_daily_history(self, tickers: list[str]) -> pd.DataFrame:
        frames = []
        start_date = date.today() - timedelta(days=450)
        for ticker_batch in chunked(tickers, 50):
            raw_batch = self.market_data_client.download_daily_history(ticker_batch, start_date)
            for ticker in ticker_batch:
                history = extract_ticker_history(raw_batch, ticker)
                if not history.empty:
                    frames.append(history)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _load_intraday_last_prices(self, tickers: list[str]) -> dict[str, float]:
        raw_frame = self.market_data_client.download_intraday_history(tickers)
        prices: dict[str, float] = {}
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

    def _build_digest_html(self, rows: list[dict]) -> str:
        html_rows = []
        for row in rows:
            flags = [
                name.replace("_", " ")
                for name in ("breakout_alert", "trailing_stop", "profit_target", "rsi_2", "time_limit", "regime_flip")
                if row[name]
            ]
            html_rows.append(
                "<tr>"
                f"<td>{row['strategy_slot']}</td>"
                f"<td>{row['ticker']}</td>"
                f"<td>{row['current_price']:.2f}</td>"
                f"<td>{row['recommended_action']}</td>"
                f"<td>{', '.join(flags)}</td>"
                "</tr>"
            )
        return (
            "<html><body>"
            "<h1>Hourly Monitor Digest</h1>"
            "<table border='1' cellpadding='6' cellspacing='0'>"
            "<tr><th>Strategy Slot</th><th>Ticker</th><th>Current Price</th><th>Recommended Action</th><th>Flags</th></tr>"
            f"{''.join(html_rows)}"
            "</table>"
            "</body></html>"
        )

    def _resolve_strategy_for_trade(
        self,
        *,
        trade,
        strategies: dict[str, ProductionStrategy],
        sector_map: dict[str, str],
    ) -> ProductionStrategy | None:
        strategy_slot = self._row_value(trade, "strategy_slot")
        if strategy_slot and strategy_slot in strategies:
            return strategies[strategy_slot]

        strategy_id = self._row_value(trade, "strategy_id")
        if strategy_id is not None:
            for strategy in strategies.values():
                if strategy.strategy_id == int(strategy_id):
                    return strategy

        ticker_sector = sector_map.get(trade["ticker"])
        exact_matches = [strategy for strategy in strategies.values() if strategy.sector == ticker_sector]
        if len(exact_matches) == 1:
            return exact_matches[0]

        ticker_regime = regime_etf_for_sector(ticker_sector or "")
        regime_matches = [
            strategy for strategy in strategies.values()
            if regime_etf_for_sector(strategy.sector) == ticker_regime
        ]
        if len(regime_matches) == 1:
            return regime_matches[0]

        all_matches = [strategy for strategy in strategies.values() if strategy.sector == "ALL"]
        if len(all_matches) == 1:
            return all_matches[0]
        if len(strategies) == 1:
            return next(iter(strategies.values()))
        return None

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
