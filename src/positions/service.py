from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

from src.sync.market_data import MarketDataClient, chunked, extract_ticker_history
from src.utils.db_manager import DatabaseManager
from src.utils.logging import get_logger
from src.utils.regime import regime_etf_for_sector
from src.utils.signal_engine import build_analysis_frame, latest_rsi_2_with_intraday, overlay_price_history
from src.utils.strategy import (
    TradeStrategyResolution,
    load_active_strategies,
    profit_target_price,
    resolve_trade_strategy,
    trailing_stop_price,
    uses_atr_trailing_stop,
)


@dataclass(frozen=True)
class PositionSnapshot:
    ticker: str
    stored_slot: str | None
    resolved_slot: str | None
    strategy_source: str
    sector: str
    strategy_sector: str | None
    current_price: float | None
    entry_price: float
    shares: int
    stop_price: float | None
    target_price: float | None
    unrealized_pct: float | None
    unrealized_pnl: float | None
    days_in_trade: int
    days_to_next_earnings: int | None
    regime_etf: str
    regime_green: bool | None
    action: str
    notes: tuple[str, ...]


@dataclass(frozen=True)
class PositionsReport:
    generated_at: str
    position_count: int
    sell_count: int
    hold_count: int
    snapshots: tuple[PositionSnapshot, ...]

    def render_console(self) -> str:
        lines = [
            f"Open positions: {self.position_count}",
            f"Action summary: sell={self.sell_count} hold={self.hold_count}",
            f"As of: {self.generated_at}",
            "",
        ]
        if not self.snapshots:
            lines.append("No open positions.")
            return "\n".join(lines)

        headers = [
            "Ticker",
            "Stored",
            "Resolved",
            "Source",
            "Sector",
            "Price",
            "PnL%",
            "Stop",
            "Target",
            "Days",
            "Earnings",
            "Regime",
            "Action",
            "Notes",
        ]
        rows = []
        for snapshot in self.snapshots:
            rows.append(
                [
                    snapshot.ticker,
                    snapshot.stored_slot or "-",
                    snapshot.resolved_slot or "-",
                    snapshot.strategy_source,
                    snapshot.sector or "-",
                    self._fmt_price(snapshot.current_price),
                    self._fmt_pct(snapshot.unrealized_pct),
                    self._fmt_price(snapshot.stop_price),
                    self._fmt_price(snapshot.target_price),
                    str(snapshot.days_in_trade),
                    str(snapshot.days_to_next_earnings) if snapshot.days_to_next_earnings is not None else "-",
                    self._fmt_regime(snapshot.regime_etf, snapshot.regime_green),
                    snapshot.action,
                    ", ".join(snapshot.notes) if snapshot.notes else "-",
                ]
            )

        widths = [
            max(len(headers[index]), *(len(row[index]) for row in rows))
            for index in range(len(headers))
        ]
        lines.append("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
        lines.append("  ".join("-" * widths[index] for index in range(len(headers))))
        for row in rows:
            lines.append("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))
        return "\n".join(lines)

    def _fmt_price(self, value: float | None) -> str:
        return f"{value:.2f}" if value is not None else "-"

    def _fmt_pct(self, value: float | None) -> str:
        return f"{value * 100.0:.1f}%" if value is not None else "-"

    def _fmt_regime(self, regime_etf: str, regime_green: bool | None) -> str:
        if regime_green is None:
            return f"{regime_etf} ?"
        return f"{regime_etf} {'Green' if regime_green else 'Red'}"


class PositionsService:
    def __init__(
        self,
        db_manager: DatabaseManager,
        *,
        market_data_client: MarketDataClient | None = None,
    ) -> None:
        self.db_manager = db_manager
        self.market_data_client = market_data_client or MarketDataClient()
        self.logger = get_logger("positions")

    def run(self) -> PositionsReport:
        self.db_manager.initialize()
        strategies = load_active_strategies()
        open_trades = self.db_manager.list_open_trades()
        if not open_trades:
            return PositionsReport(
                generated_at=date.today().isoformat(),
                position_count=0,
                sell_count=0,
                hold_count=0,
                snapshots=(),
            )

        trade_tickers = [row["ticker"] for row in open_trades]
        intraday_prices = self._load_intraday_last_prices(trade_tickers + ["SPY", "QQQ"])
        history_tickers = sorted(set(trade_tickers).union(["SPY", "QQQ"]))
        base_history = self.db_manager.load_price_history(history_tickers)
        earnings_loader = getattr(self.db_manager, "load_earnings_calendar", None)
        earnings_calendar = earnings_loader(trade_tickers) if callable(earnings_loader) else pd.DataFrame()
        recent_history = self._download_recent_daily_history(history_tickers)
        price_history = overlay_price_history(base_history, recent_history)
        universe_rows = self.db_manager.list_universe_rows(active_only=False)
        analysis_frame, _ = build_analysis_frame(
            price_history,
            universe_rows,
            earnings_calendar=earnings_calendar,
        )
        latest_snapshot = analysis_frame.sort_values(["ticker", "date"]).groupby("ticker", as_index=False).tail(1)
        latest_rows_by_ticker = latest_snapshot.set_index("ticker").to_dict(orient="index")
        sector_map = {row["ticker"]: row["sector"] for row in universe_rows}

        snapshots: list[PositionSnapshot] = []
        for trade in open_trades:
            resolution = resolve_trade_strategy(
                trade=trade,
                strategies=strategies,
                sector_map=sector_map,
                backtest_lookup=getattr(self.db_manager, "get_backtest_result_by_strategy_id", None),
            )
            snapshots.append(
                self._build_snapshot(
                    trade=trade,
                    resolution=resolution,
                    strategies=strategies,
                    latest_rows_by_ticker=latest_rows_by_ticker,
                    intraday_prices=intraday_prices,
                    sector_map=sector_map,
                    price_history=price_history,
                )
            )

        snapshots.sort(
            key=lambda row: (
                0 if row.action == "sell" else 1,
                row.days_to_next_earnings if row.days_to_next_earnings is not None else 9_999,
                row.ticker,
            )
        )
        sell_count = sum(1 for row in snapshots if row.action == "sell")
        return PositionsReport(
            generated_at=date.today().isoformat(),
            position_count=len(snapshots),
            sell_count=sell_count,
            hold_count=len(snapshots) - sell_count,
            snapshots=tuple(snapshots),
        )

    def _build_snapshot(
        self,
        *,
        trade,
        resolution: TradeStrategyResolution,
        strategies: dict,
        latest_rows_by_ticker: dict[str, dict],
        intraday_prices: dict[str, float],
        sector_map: dict[str, str],
        price_history: pd.DataFrame,
    ) -> PositionSnapshot:
        ticker = str(trade["ticker"])
        latest_ticker_row = latest_rows_by_ticker.get(ticker, {})
        current_price = intraday_prices.get(ticker)
        if current_price is None and pd.notna(latest_ticker_row.get("adj_close")):
            current_price = float(latest_ticker_row["adj_close"])

        strategy = resolution.strategy
        ticker_sector = sector_map.get(ticker) or (strategy.sector if strategy is not None else "")
        regime_etf = regime_etf_for_sector(ticker_sector)
        regime_green = self._resolve_regime_state(
            regime_etf=regime_etf,
            latest_rows_by_ticker=latest_rows_by_ticker,
            intraday_prices=intraday_prices,
        )
        entry_price = float(trade["entry_price"])
        shares = int(trade["shares"])
        entry_atr = self._row_value(trade, "entry_atr")
        if (entry_atr is None or pd.isna(entry_atr)) and pd.notna(latest_ticker_row.get("atr_14")):
            entry_atr = float(latest_ticker_row["atr_14"])

        stop_price = None
        target_price = None
        days_to_next_earnings = (
            int(float(latest_ticker_row["days_to_next_earnings"]))
            if pd.notna(latest_ticker_row.get("days_to_next_earnings"))
            else None
        )
        time_in_trade = len(pd.bdate_range(trade["entry_date"], date.today().isoformat())) - 1
        action = "hold"
        notes: list[str] = []

        if strategy is None:
            notes.append("strategy unresolved")
        else:
            try:
                stop_price = trailing_stop_price(
                    max_price_seen=float(trade["max_price_seen"]),
                    entry_atr=float(entry_atr) if entry_atr is not None else None,
                    exit_rules=strategy.exit_rules,
                )
                target_price = profit_target_price(
                    entry_price=entry_price,
                    entry_atr=float(entry_atr) if entry_atr is not None else None,
                    exit_rules=strategy.exit_rules,
                )
            except Exception:
                notes.append("exit thresholds unavailable")

            if uses_atr_trailing_stop(strategy.exit_rules) and self._row_value(trade, "entry_atr") in (None, ""):
                notes.append("missing stored entry_atr")
            if self._row_value(trade, "strategy_slot") and self._row_value(trade, "strategy_slot") not in strategies:
                notes.append("inactive stored slot")
            if strategy.sector not in ("ALL", "", ticker_sector):
                notes.append("strategy sector mismatch")
            if regime_green is False:
                notes.append("regime red")
            if (
                strategy.exit_rules.exit_before_earnings_days is not None
                and days_to_next_earnings is not None
                and days_to_next_earnings <= int(strategy.exit_rules.exit_before_earnings_days)
            ):
                notes.append("pre earnings exit")

            if current_price is not None:
                rsi_2 = latest_rsi_2_with_intraday(
                    price_history=price_history,
                    ticker=ticker,
                    current_price=current_price,
                    as_of=date.today(),
                )
                should_exit = any(
                    [
                        stop_price is not None and current_price < stop_price,
                        target_price is not None and current_price >= target_price,
                        rsi_2 > 90,
                        time_in_trade > strategy.exit_rules.time_limit_days,
                        regime_green is False,
                        "pre earnings exit" in notes,
                    ]
                )
                if should_exit:
                    action = "sell"

        unrealized_pct = None
        unrealized_pnl = None
        if current_price is not None:
            unrealized_pct = (current_price / entry_price) - 1.0
            unrealized_pnl = (current_price - entry_price) * shares

        return PositionSnapshot(
            ticker=ticker,
            stored_slot=self._row_value(trade, "strategy_slot"),
            resolved_slot=strategy.slot if strategy is not None else None,
            strategy_source=resolution.source,
            sector=ticker_sector or "Unknown",
            strategy_sector=strategy.sector if strategy is not None else None,
            current_price=current_price,
            entry_price=entry_price,
            shares=shares,
            stop_price=stop_price,
            target_price=target_price,
            unrealized_pct=unrealized_pct,
            unrealized_pnl=unrealized_pnl,
            days_in_trade=time_in_trade,
            days_to_next_earnings=days_to_next_earnings,
            regime_etf=regime_etf,
            regime_green=regime_green,
            action=action,
            notes=tuple(notes),
        )

    def _resolve_regime_state(
        self,
        *,
        regime_etf: str,
        latest_rows_by_ticker: dict[str, dict],
        intraday_prices: dict[str, float],
    ) -> bool | None:
        regime_row = latest_rows_by_ticker.get(regime_etf)
        regime_price = intraday_prices.get(regime_etf)
        if not regime_row or regime_price is None:
            return None
        sma_column = "qqq_sma_200" if regime_etf == "QQQ" else "spy_sma_200"
        sma_value = regime_row.get(sma_column)
        if pd.isna(sma_value):
            return None
        return regime_price >= float(sma_value)

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
            self.logger.warning("Falling back to DuckDB-only daily history in positions: %s", exc)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _load_intraday_last_prices(self, tickers: list[str]) -> dict[str, float]:
        prices: dict[str, float] = {}
        try:
            raw_frame = self.market_data_client.download_intraday_history(tickers)
        except Exception as exc:
            self.logger.warning("Unable to load intraday prices in positions: %s", exc)
            return prices
        for ticker in tickers:
            history = extract_ticker_history(raw_frame, ticker)
            if history.empty:
                continue
            prices[ticker] = float(history.iloc[-1]["close"])
        return prices

    def _row_value(self, row, key: str):
        if hasattr(row, "keys"):
            return row[key] if key in row.keys() else None
        if isinstance(row, dict):
            return row.get(key)
        try:
            return row[key]
        except Exception:
            return None
