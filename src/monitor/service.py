from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from src.settings import get_settings
from src.sync.market_data import MarketDataClient, extract_ticker_history
from src.sync.service import REFERENCE_TICKERS
from src.utils.db_manager import DatabaseManager
from src.utils.emailer import send_html_email
from src.utils.logging import get_logger
from src.utils.signal_engine import build_analysis_frame, latest_rsi_2_with_intraday
from src.utils.strategy import load_active_strategy


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
        strategy = load_active_strategy()
        settings = get_settings()
        open_trades = self.db_manager.list_open_trades()
        if not open_trades:
            return MonitorReport(watchlist_size=0, triggered_count=0, emailed=False)

        trade_tickers = [row["ticker"] for row in open_trades]
        intraday_prices = self._load_intraday_last_prices(trade_tickers + ["SPY", "QQQ"])
        price_history = self.db_manager.load_price_history(sorted(set(trade_tickers).union(["SPY", "QQQ"])))
        universe_rows = self.db_manager.list_universe_rows(active_only=False)
        analysis_frame, _ = build_analysis_frame(price_history, universe_rows)
        latest_snapshot = analysis_frame.sort_values(["ticker", "date"]).groupby("ticker", as_index=False).tail(1)
        sector_map = {row["ticker"]: row["sector"] for row in universe_rows}

        triggered_rows = []
        for trade in open_trades:
            current_price = intraday_prices.get(trade["ticker"])
            if current_price is None:
                continue
            yesterday_high = self._load_yesterday_high(trade["ticker"])
            trade_row = self.db_manager.get_latest_open_trade(trade["ticker"])
            max_price_seen = max(float(trade["max_price_seen"]), current_price)
            if trade_row is not None and max_price_seen > float(trade["max_price_seen"]):
                self.db_manager.update_trade_max_price(int(trade_row["rowid"]), max_price_seen)

            regime_etf = "QQQ" if sector_map.get(trade["ticker"]) == "Information Technology" or sector_map.get(trade["ticker"]) == "Communication Services" else "SPY"
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
            exit_flags = {
                "breakout_alert": yesterday_high is not None and current_price > yesterday_high,
                "trailing_stop": current_price < max_price_seen * (1 - strategy.exit_rules.trailing_stop_pct),
                "profit_target": current_price >= float(trade["entry_price"]) * (1 + strategy.exit_rules.profit_target_pct),
                "rsi_2": rsi_2 > 90,
                "time_limit": time_in_trade > strategy.exit_rules.time_limit_days,
                "regime_flip": not regime_green,
            }
            should_exit = any(
                exit_flags[name]
                for name in ("trailing_stop", "profit_target", "rsi_2", "time_limit", "regime_flip")
            )
            if should_exit and trade_row is not None:
                self.db_manager.close_trade(
                    trade_rowid=int(trade_row["rowid"]),
                    exit_date=date.today().isoformat(),
                    exit_price=current_price,
                )

            if exit_flags["breakout_alert"] or should_exit:
                triggered_rows.append(
                    {
                        "ticker": trade["ticker"],
                        "current_price": current_price,
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
                f"<td>{row['ticker']}</td>"
                f"<td>{row['current_price']:.2f}</td>"
                f"<td>{', '.join(flags)}</td>"
                "</tr>"
            )
        return (
            "<html><body>"
            "<h1>Hourly Monitor Digest</h1>"
            "<table border='1' cellpadding='6' cellspacing='0'>"
            "<tr><th>Ticker</th><th>Current Price</th><th>Flags</th></tr>"
            f"{''.join(html_rows)}"
            "</table>"
            "</body></html>"
        )
