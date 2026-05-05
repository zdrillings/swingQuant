from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Callable

import pandas as pd

from src.sync.market_data import MarketDataClient, chunked, extract_ticker_history
from src.sync.universe import scrape_sp_universe
from src.utils.db_manager import DatabaseManager
from src.utils.logging import get_logger


REFERENCE_TICKERS = ("USO", "CPER", "GLD", "SPY", "QQQ")
LIQUIDITY_THRESHOLD = 20_000_000
BATCH_SIZE = 50


@dataclass(frozen=True)
class SyncReport:
    universe_size: int
    inserted_rows: int
    failed_tickers: tuple[str, ...]
    inactive_for_liquidity: tuple[str, ...]


class SyncService:
    def __init__(
        self,
        db_manager: DatabaseManager,
        *,
        market_data_client: MarketDataClient | None = None,
        universe_loader: Callable[[], list] = scrape_sp_universe,
    ) -> None:
        self.db_manager = db_manager
        self.market_data_client = market_data_client or MarketDataClient()
        self.universe_loader = universe_loader
        self.logger = get_logger("sync")

    def run(self) -> SyncReport:
        self.db_manager.initialize()
        if self.db_manager.universe_is_empty():
            members = self.universe_loader()
            self.db_manager.bootstrap_universe(members)
            self.logger.info("Bootstrapped universe with %s tickers", len(members))

        universe_tickers = self.db_manager.list_universe_tickers(active_only=False)
        fetch_tickers = sorted(set(universe_tickers).union(REFERENCE_TICKERS))
        fetch_plan = self.db_manager.build_fetch_plan(fetch_tickers)

        inserted_rows = 0
        failed_tickers: set[str] = set()

        for start_date, tickers in sorted(fetch_plan.items(), key=lambda item: item[0]):
            rows_added, failed = self._sync_start_date_group(start_date, tickers)
            inserted_rows += rows_added
            failed_tickers.update(failed)

        inactive_for_liquidity = self._apply_liquidity_filter(universe_tickers)
        return SyncReport(
            universe_size=len(universe_tickers),
            inserted_rows=inserted_rows,
            failed_tickers=tuple(sorted(failed_tickers)),
            inactive_for_liquidity=tuple(sorted(inactive_for_liquidity)),
        )

    def _sync_start_date_group(self, start_date: date, tickers: list[str]) -> tuple[int, set[str]]:
        inserted_rows = 0
        failed_tickers: set[str] = set()

        for ticker_batch in chunked(sorted(tickers), BATCH_SIZE):
            inserted, failed = self._sync_batch(start_date, ticker_batch)
            inserted_rows += inserted
            failed_tickers.update(failed)

        return inserted_rows, failed_tickers

    def _sync_batch(self, start_date: date, tickers: list[str]) -> tuple[int, set[str]]:
        inserted_rows = 0
        unresolved = set(tickers)

        try:
            raw_batch = self.market_data_client.download_daily_history(tickers, start_date)
        except Exception as exc:
            self.logger.warning(
                "Batch download failed for %s tickers starting %s: %s",
                len(tickers),
                start_date,
                exc,
            )
            raw_batch = pd.DataFrame()

        if not raw_batch.empty:
            for ticker in tickers:
                history = extract_ticker_history(raw_batch, ticker)
                if history.empty:
                    continue
                inserted_rows += self.db_manager.upsert_historical_frame(history)
                unresolved.discard(ticker)

        failed_tickers: set[str] = set()
        for ticker in sorted(unresolved):
            try:
                raw_single = self.market_data_client.download_daily_history([ticker], start_date)
                history = extract_ticker_history(raw_single, ticker)
                if history.empty:
                    raise ValueError(f"No history returned for {ticker}")
                inserted_rows += self.db_manager.upsert_historical_frame(history)
            except Exception as exc:
                self.logger.error("Permanent sync failure for %s: %s", ticker, exc)
                failed_tickers.add(ticker)
                if ticker in self.db_manager.list_universe_tickers(active_only=False):
                    self.db_manager.set_ticker_status(ticker, is_active=False)

        return inserted_rows, failed_tickers

    def _apply_liquidity_filter(self, tickers: list[str]) -> list[str]:
        inactive: list[str] = []
        for ticker in tickers:
            frame = self.db_manager.load_recent_liquidity_window(ticker, window=30)
            if len(frame.index) < 30:
                self.db_manager.set_ticker_status(ticker, is_active=False)
                inactive.append(ticker)
                continue

            frame["dollar_volume"] = frame["close"] * frame["volume"]
            median_dollar_volume = float(frame["dollar_volume"].median())
            is_active = median_dollar_volume >= LIQUIDITY_THRESHOLD
            self.db_manager.set_ticker_status(
                ticker,
                is_active=is_active,
                md_volume_30d=median_dollar_volume,
            )
            if not is_active:
                inactive.append(ticker)
        return inactive
