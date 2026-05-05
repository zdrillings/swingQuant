from __future__ import annotations

from datetime import date

from src.settings import get_settings
from src.utils.db_manager import DatabaseManager
from src.utils.sizing import compute_position_size
from src.utils.strategy import load_active_strategy


class TradeService:
    def __init__(self, db_manager: DatabaseManager) -> None:
        self.db_manager = db_manager

    def buy(self, *, ticker: str, price: float, shares: int | None = None) -> str:
        self.db_manager.initialize()
        strategy = load_active_strategy()
        settings = get_settings()
        final_shares = shares or compute_position_size(
            price=price,
            trailing_stop_pct=strategy.exit_rules.trailing_stop_pct,
            settings=settings,
        )
        self.db_manager.open_trade(
            ticker=ticker,
            entry_date=date.today().isoformat(),
            entry_price=price,
            shares=final_shares,
            max_price_seen=price,
        )
        return f"Bought {ticker}: {final_shares} shares at {price:.2f}"

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
