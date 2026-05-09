from __future__ import annotations

from datetime import date, timedelta
import math

import pandas as pd

from src.settings import get_settings
from src.sync.market_data import MarketDataClient, chunked, extract_ticker_history
from src.utils.db_manager import DatabaseManager
from src.utils.signal_engine import latest_atr_14_with_intraday, overlay_price_history
from src.utils.sizing import compute_position_size
from src.utils.strategy import ProductionStrategy, load_active_strategies, load_active_strategy_for_slot


class TradeService:
    def __init__(
        self,
        db_manager: DatabaseManager,
        *,
        market_data_client: MarketDataClient | None = None,
    ) -> None:
        self.db_manager = db_manager
        self.market_data_client = market_data_client or MarketDataClient()

    def buy(self, *, ticker: str, price: float, shares: int | None = None, strategy_slot: str | None = None) -> str:
        self.db_manager.initialize()
        strategy = self._resolve_strategy_for_buy(ticker=ticker, strategy_slot=strategy_slot)
        settings = get_settings()
        entry_atr = self._load_entry_atr(ticker=ticker, current_price=price) if strategy.exit_rules.trailing_stop_atr_mult is not None else None
        final_shares = shares or compute_position_size(
            price=price,
            exit_rules=strategy.exit_rules,
            settings=settings,
            entry_atr=entry_atr,
        )
        self.db_manager.open_trade(
            ticker=ticker,
            entry_date=date.today().isoformat(),
            entry_price=price,
            entry_atr=entry_atr,
            strategy_id=strategy.strategy_id,
            strategy_slot=strategy.slot,
            shares=final_shares,
            max_price_seen=price,
        )
        return f"Bought {ticker}: {final_shares} shares at {price:.2f} using strategy slot '{strategy.slot}'"

    def sell(self, *, ticker: str, price: float) -> str:
        self.db_manager.initialize()
        trade = self.db_manager.get_latest_open_trade(ticker)
        if trade is None:
            raise ValueError(f"No open trade found for {ticker}")
        self.db_manager.close_trade(
            trade_rowid=int(trade["rowid"]),
            exit_date=date.today().isoformat(),
            exit_price=price,
        )
        pnl = (price - float(trade["entry_price"])) * int(trade["shares"])
        return f"Realized P&L for {ticker}: {pnl:.2f}"

    def _load_entry_atr(self, *, ticker: str, current_price: float) -> float:
        base_history = self.db_manager.load_price_history([ticker])
        recent_history = self._download_recent_daily_history([ticker])
        history = overlay_price_history(base_history, recent_history)
        if history.empty:
            raise ValueError(
                f"Historical prices are unavailable for {ticker}. "
                "Unable to load enough daily history for ATR-based sizing."
            )
        entry_atr = latest_atr_14_with_intraday(
            price_history=history,
            ticker=ticker,
            current_price=current_price,
            as_of=date.today(),
        )
        if not math.isfinite(entry_atr) or entry_atr <= 0:
            raise ValueError(f"ATR_14 is unavailable for {ticker}. More history is required before opening an ATR-based trade.")
        return entry_atr

    def _download_recent_daily_history(self, tickers: list[str]) -> pd.DataFrame:
        frames = []
        unresolved = list(tickers)
        for lookback_days in (450, 180, 90, 45):
            if not unresolved:
                break
            start_date = date.today() - timedelta(days=lookback_days)
            newly_resolved: list[str] = []
            for ticker_batch in chunked(unresolved, 50):
                raw_batch = self.market_data_client.download_daily_history(ticker_batch, start_date)
                for ticker in ticker_batch:
                    history = extract_ticker_history(raw_batch, ticker)
                    if history.empty:
                        continue
                    frames.append(history)
                    newly_resolved.append(ticker)
            unresolved = [ticker for ticker in unresolved if ticker not in newly_resolved]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _resolve_strategy_for_buy(self, *, ticker: str, strategy_slot: str | None) -> ProductionStrategy:
        if strategy_slot:
            return load_active_strategy_for_slot(strategy_slot)

        strategies = load_active_strategies()
        latest_trade = self.db_manager.get_latest_trade(ticker)
        if latest_trade is not None:
            stored_slot = latest_trade["strategy_slot"]
            if stored_slot:
                strategy = strategies.get(str(stored_slot))
                if strategy is not None:
                    return strategy
            stored_strategy_id = latest_trade["strategy_id"]
            if stored_strategy_id is not None:
                for strategy in strategies.values():
                    if strategy.strategy_id == int(stored_strategy_id):
                        return strategy

        universe_rows = self.db_manager.list_universe_rows(active_only=False)
        sector_map = {row["ticker"]: row["sector"] for row in universe_rows}
        ticker_sector = sector_map.get(ticker)
        exact_matches = [strategy for strategy in strategies.values() if strategy.sector == ticker_sector]
        if len(exact_matches) == 1:
            return exact_matches[0]
        if len(exact_matches) > 1:
            slots = ", ".join(sorted(strategy.slot for strategy in exact_matches))
            raise ValueError(f"Ticker {ticker} matches multiple strategy slots: {slots}. Pass --slot explicitly.")

        all_matches = [strategy for strategy in strategies.values() if strategy.sector == "ALL"]
        if len(all_matches) == 1:
            return all_matches[0]
        if len(all_matches) > 1:
            slots = ", ".join(sorted(strategy.slot for strategy in all_matches))
            raise ValueError(f"Ticker {ticker} requires an explicit strategy slot. Available ALL-strategy slots: {slots}.")

        available = ", ".join(f"{slot}:{strategy.sector}" for slot, strategy in sorted(strategies.items()))
        raise ValueError(f"No active strategy slot matches ticker {ticker} (sector={ticker_sector}). Available slots: {available}")
