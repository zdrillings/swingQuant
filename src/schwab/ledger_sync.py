from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import math
from typing import Any

import pandas as pd

from src.schwab.client import SchwabClient
from src.sync.market_data import MarketDataClient, extract_ticker_history
from src.utils.db_manager import DatabaseManager
from src.utils.logging import get_logger
from src.utils.strategy import ProductionStrategy, load_active_strategies


@dataclass(frozen=True)
class SchwabLedgerSyncReport:
    broker_positions: int
    ledger_open_before: int
    opened: int
    closed: int
    updated: int
    skipped: int
    dry_run: bool
    messages: tuple[str, ...]

    def render_console(self) -> str:
        lines = [
            "Schwab ledger sync completed:",
            f"  dry_run={self.dry_run}",
            f"  broker_positions={self.broker_positions}",
            f"  ledger_open_before={self.ledger_open_before}",
            f"  opened={self.opened}",
            f"  closed={self.closed}",
            f"  updated={self.updated}",
            f"  skipped={self.skipped}",
        ]
        lines.extend(f"  - {message}" for message in self.messages)
        return "\n".join(lines)


class SchwabLedgerSyncService:
    def __init__(
        self,
        db_manager: DatabaseManager,
        *,
        schwab_client: SchwabClient | None = None,
        market_data_client: MarketDataClient | None = None,
    ) -> None:
        self.db_manager = db_manager
        self.schwab_client = schwab_client or SchwabClient()
        self.market_data_client = market_data_client or MarketDataClient()
        self.logger = get_logger("schwab_ledger_sync")

    def run(
        self,
        *,
        account_hash: str | None = None,
        dry_run: bool = False,
        ignore_tickers: tuple[str, ...] = (),
        ignore_funds: bool = True,
        close_missing: bool = False,
    ) -> SchwabLedgerSyncReport:
        self.db_manager.initialize()
        raw_positions = self.schwab_client.list_positions(account_hash=account_hash)
        ignored_tickers = {str(ticker).upper().strip() for ticker in ignore_tickers if str(ticker).strip()}
        broker_positions, skipped_messages = self._normalize_positions(
            raw_positions,
            ignored_tickers=ignored_tickers,
            ignore_funds=ignore_funds,
        )
        open_trades = [dict(row) for row in self.db_manager.list_open_trades()]
        open_by_ticker = {
            str(row["ticker"]).upper(): row
            for row in open_trades
            if str(row["ticker"]).upper() not in ignored_tickers
        }
        broker_by_ticker = {position["ticker"]: position for position in broker_positions}

        to_close = sorted(set(open_by_ticker) - set(broker_by_ticker))
        to_open = sorted(set(broker_by_ticker) - set(open_by_ticker))
        matched = sorted(set(open_by_ticker) & set(broker_by_ticker))
        close_prices = self._latest_exit_prices(to_close)
        universe_rows = self.db_manager.list_universe_rows(active_only=False)
        strategies = load_active_strategies()

        messages: list[str] = list(skipped_messages)
        opened = 0
        closed = 0
        updated = 0

        for ticker in to_close:
            trade = self.db_manager.get_latest_open_trade(ticker)
            if trade is None:
                continue
            exit_price, source = close_prices.get(ticker, (float(trade["entry_price"]), "entry_price_fallback"))
            if close_missing:
                messages.append(f"close {ticker}: no Schwab position, exit_price={exit_price:.2f} source={source}")
                closed += 1
            else:
                messages.append(
                    f"warn {ticker}: open in ledger but missing from Schwab positions; "
                    f"would close at {exit_price:.2f} source={source} with --close-missing"
                )
            if close_missing and not dry_run:
                self.db_manager.close_trade(
                    trade_rowid=int(trade["rowid"]),
                    exit_date=date.today().isoformat(),
                    exit_price=float(exit_price),
                )

        for ticker in matched:
            trade = open_by_ticker[ticker]
            broker = broker_by_ticker[ticker]
            broker_price = self._ledger_price(float(broker["average_price"]))
            broker_shares = int(broker["shares"])
            current_price = broker.get("current_price")
            current_ledger_price = (
                self._ledger_price(float(current_price))
                if current_price is not None and math.isfinite(float(current_price))
                else broker_price
            )
            max_price_seen = max(
                self._ledger_price(float(trade["max_price_seen"])),
                broker_price,
                current_ledger_price,
            )
            old_entry_price = self._ledger_price(float(trade["entry_price"]))
            old_max_price_seen = self._ledger_price(float(trade["max_price_seen"]))
            share_changed = int(trade["shares"]) != broker_shares
            entry_changed = not math.isclose(old_entry_price, broker_price, rel_tol=0.0, abs_tol=0.005)
            max_changed = not math.isclose(old_max_price_seen, max_price_seen, rel_tol=0.0, abs_tol=0.005)
            if share_changed or entry_changed or max_changed:
                changes: list[str] = []
                if share_changed:
                    changes.append(f"shares {trade['shares']} -> {broker_shares}")
                if entry_changed:
                    changes.append(f"avg_price {old_entry_price:.2f} -> {broker_price:.2f}")
                if max_changed:
                    changes.append(f"max_price_seen {old_max_price_seen:.2f} -> {max_price_seen:.2f}")
                messages.append(f"update {ticker}: {', '.join(changes)}")
                updated += 1
                if not dry_run:
                    self.db_manager.update_open_trade_from_broker(
                        int(trade["rowid"]),
                        entry_price=broker_price,
                        shares=broker_shares,
                        max_price_seen=max_price_seen,
                    )

        for ticker in to_open:
            broker = broker_by_ticker[ticker]
            strategy = self._resolve_strategy_for_broker_position(
                ticker=ticker,
                universe_rows=universe_rows,
                strategies=strategies,
            )
            current_price = broker.get("current_price")
            entry_price = float(broker["average_price"])
            max_price_seen = max(
                entry_price,
                float(current_price) if current_price is not None and math.isfinite(float(current_price)) else entry_price,
            )
            strategy_text = strategy.slot if strategy is not None else "unresolved"
            messages.append(
                f"open {ticker}: Schwab position not in ledger, shares={broker['shares']}, "
                f"avg_price={entry_price:.2f}, strategy={strategy_text}"
            )
            opened += 1
            if not dry_run:
                self.db_manager.open_trade(
                    ticker=ticker,
                    entry_date=date.today().isoformat(),
                    entry_price=entry_price,
                    entry_atr=None,
                    strategy_id=strategy.strategy_id if strategy is not None else None,
                    strategy_slot=strategy.slot if strategy is not None else None,
                    shares=int(broker["shares"]),
                    max_price_seen=max_price_seen,
                )

        return SchwabLedgerSyncReport(
            broker_positions=len(broker_positions),
            ledger_open_before=len(open_trades),
            opened=opened,
            closed=closed,
            updated=updated,
            skipped=len(skipped_messages),
            dry_run=bool(dry_run),
            messages=tuple(messages),
        )

    def _normalize_positions(
        self,
        raw_positions: list[dict[str, Any]],
        *,
        ignored_tickers: set[str],
        ignore_funds: bool,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        positions: list[dict[str, Any]] = []
        skipped: list[str] = []
        for raw in raw_positions:
            ticker = str(raw.get("ticker") or "").upper().strip()
            if not ticker:
                skipped.append("skip position without ticker")
                continue
            if ticker in ignored_tickers:
                skipped.append(f"skip {ticker}: ignored ticker")
                continue
            if ignore_funds and self._is_fund_position(raw):
                skipped.append(f"skip {ticker}: ignored fund/ETF position")
                continue
            raw_quantity = float(raw.get("quantity") or 0.0)
            if raw_quantity <= 0:
                skipped.append(f"skip {ticker}: non-long quantity={raw_quantity:g}")
                continue
            rounded_quantity = round(raw_quantity)
            if not math.isclose(raw_quantity, float(rounded_quantity), rel_tol=0.0, abs_tol=0.000001):
                skipped.append(f"skip {ticker}: fractional quantity={raw_quantity:g} cannot fit integer ledger shares")
                continue
            average_price = self._average_price(raw, raw_quantity)
            if average_price is None or not math.isfinite(average_price) or average_price <= 0:
                skipped.append(f"skip {ticker}: missing average price")
                continue
            current_price = self._current_price(raw, raw_quantity)
            positions.append(
                {
                    "ticker": ticker,
                    "shares": int(rounded_quantity),
                    "average_price": float(average_price),
                    "current_price": current_price,
                }
            )
        return positions, skipped

    def _is_fund_position(self, position: dict[str, Any]) -> bool:
        asset_type = str(position.get("asset_type") or "").upper()
        instrument_type = str(position.get("instrument_type") or "").upper()
        raw_instrument = (position.get("raw") or {}).get("instrument") or {}
        raw_asset_type = str(raw_instrument.get("assetType") or "").upper()
        raw_instrument_type = str(raw_instrument.get("type") or "").upper()
        fund_markers = {"COLLECTIVE_INVESTMENT", "MUTUAL_FUND"}
        instrument_markers = {"EXCHANGE_TRADED_FUND", "MUTUAL_FUND", "CLOSED_END_FUND"}
        return (
            asset_type in fund_markers
            or raw_asset_type in fund_markers
            or instrument_type in instrument_markers
            or raw_instrument_type in instrument_markers
        )

    def _average_price(self, position: dict[str, Any], quantity: float) -> float | None:
        value = position.get("average_price")
        if value not in (None, ""):
            return float(value)
        market_value = position.get("market_value")
        if market_value not in (None, "") and quantity > 0:
            return abs(float(market_value)) / quantity
        return None

    def _current_price(self, position: dict[str, Any], quantity: float) -> float | None:
        market_value = position.get("market_value")
        if market_value in (None, "") or quantity <= 0:
            return None
        return abs(float(market_value)) / quantity

    @staticmethod
    def _ledger_price(value: float) -> float:
        return round(float(value), 2)

    def _latest_exit_prices(self, tickers: list[str]) -> dict[str, tuple[float, str]]:
        if not tickers:
            return {}
        prices: dict[str, tuple[float, str]] = {}
        try:
            raw_intraday = self.market_data_client.download_intraday_history(tickers)
            for ticker in tickers:
                history = extract_ticker_history(raw_intraday, ticker)
                if history.empty:
                    continue
                latest = history.sort_values("date").tail(1).iloc[0]
                prices[ticker] = (float(latest["close"]), "intraday")
        except Exception as exc:
            self.logger.warning("Unable to download intraday prices for Schwab ledger exits: %s", exc)

        missing = [ticker for ticker in tickers if ticker not in prices]
        if missing:
            try:
                history = self.db_manager.load_price_history(missing)
                if not history.empty:
                    for ticker, group in history.groupby("ticker"):
                        latest = group.sort_values("date").tail(1).iloc[0]
                        prices[str(ticker).upper()] = (float(latest["adj_close"]), "stored_close")
            except Exception as exc:
                self.logger.warning("Unable to load stored prices for Schwab ledger exits: %s", exc)
        return prices

    def _resolve_strategy_for_broker_position(
        self,
        *,
        ticker: str,
        universe_rows: list,
        strategies: dict[str, ProductionStrategy],
    ) -> ProductionStrategy | None:
        sector_map = {str(row["ticker"]).upper(): row["sector"] for row in universe_rows}
        ticker_sector = sector_map.get(ticker)
        exact_matches = [strategy for strategy in strategies.values() if strategy.sector == ticker_sector]
        if len(exact_matches) == 1:
            return exact_matches[0]
        all_matches = [strategy for strategy in strategies.values() if strategy.sector == "ALL"]
        if len(all_matches) == 1:
            return all_matches[0]
        return None
