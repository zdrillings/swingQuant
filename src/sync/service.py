from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import math
from typing import Callable

import pandas as pd

from src.sync.market_data import MarketDataClient, chunked, extract_ticker_history
from src.sync.universe import scrape_sp_universe
from src.utils.db_manager import DatabaseManager
from src.utils.logging import get_logger


REFERENCE_TICKERS = ("USO", "CPER", "GLD", "SPY", "QQQ", "XLB", "XLE", "XLI", "XLK", "XLF", "XLC", "XLV", "XLY", "XLP", "XLRE", "XLU")
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
        self.logger.info("Starting sync: initializing schemas and loading universe state")
        self.db_manager.initialize()
        if self.db_manager.universe_is_empty():
            self.logger.info("Universe table is empty; bootstrapping S&P 500/400/600 constituents from Wikipedia")
            members = self.universe_loader()
            self.db_manager.bootstrap_universe(members)
            self.logger.info("Bootstrapped universe with %s tickers", len(members))
        else:
            self.logger.info("Universe table already populated; reusing existing constituents")

        universe_tickers = self.db_manager.list_universe_tickers(active_only=False)
        fetch_tickers = sorted(set(universe_tickers).union(REFERENCE_TICKERS))
        fetch_plan = self.db_manager.build_fetch_plan(fetch_tickers)
        fetch_groups = sorted(fetch_plan.items(), key=lambda item: item[0])
        total_batches = sum(math.ceil(len(tickers) / BATCH_SIZE) for _, tickers in fetch_groups)

        if not fetch_groups:
            self.logger.info("Historical data already up to date; no OHLCV fetches required")
        else:
            self.logger.info(
                "Starting OHLCV sync: total_tickers=%s start_groups=%s total_batches=%s",
                len(fetch_tickers),
                len(fetch_groups),
                total_batches,
            )

        inserted_rows = 0
        retry_tickers: set[str] = set()
        fetch_start_by_ticker = {
            ticker: start_date
            for start_date, tickers in fetch_groups
            for ticker in tickers
        }

        for group_index, (start_date, tickers) in enumerate(fetch_groups, start=1):
            rows_added, failed = self._sync_start_date_group(
                start_date,
                tickers,
                group_index=group_index,
                total_groups=len(fetch_groups),
            )
            inserted_rows += rows_added
            retry_tickers.update(failed)

        final_failed_tickers: set[str] = set()
        if retry_tickers:
            self.logger.info(
                "Starting final retry pass for %s tickers with unresolved fetches",
                len(retry_tickers),
            )
            retry_rows, final_failed_tickers = self._retry_failed_tickers(
                retry_tickers,
                fetch_start_by_ticker=fetch_start_by_ticker,
                universe_tickers=set(universe_tickers),
            )
            inserted_rows += retry_rows
            self.logger.info(
                "Final retry pass complete: recovered_tickers=%s remaining_failures=%s",
                len(retry_tickers) - len(final_failed_tickers),
                len(final_failed_tickers),
            )

        self.logger.info("Applying liquidity filter to %s universe tickers", len(universe_tickers))
        inactive_for_liquidity = self._apply_liquidity_filter(
            [ticker for ticker in universe_tickers if ticker not in final_failed_tickers]
        )
        earnings_tickers = [ticker for ticker in universe_tickers if ticker not in final_failed_tickers]
        self.logger.info("Syncing earnings calendars for %s tickers", len(earnings_tickers))
        self._sync_earnings_calendar(earnings_tickers)
        self.logger.info(
            "Sync complete: universe_size=%s inserted_rows=%s failed_tickers=%s illiquid_tickers=%s",
            len(universe_tickers),
            inserted_rows,
            len(final_failed_tickers),
            len(inactive_for_liquidity),
        )
        return SyncReport(
            universe_size=len(universe_tickers),
            inserted_rows=inserted_rows,
            failed_tickers=tuple(sorted(final_failed_tickers)),
            inactive_for_liquidity=tuple(sorted(inactive_for_liquidity)),
        )

    def _sync_start_date_group(
        self,
        start_date: date,
        tickers: list[str],
        *,
        group_index: int,
        total_groups: int,
    ) -> tuple[int, set[str]]:
        inserted_rows = 0
        failed_tickers: set[str] = set()
        ticker_batches = chunked(sorted(tickers), BATCH_SIZE)

        self.logger.info(
            "Sync group %s/%s: start_date=%s tickers=%s batches=%s",
            group_index,
            total_groups,
            start_date,
            len(tickers),
            len(ticker_batches),
        )

        for batch_index, ticker_batch in enumerate(ticker_batches, start=1):
            self.logger.info(
                "Fetching batch %s/%s for start_date=%s tickers=%s",
                batch_index,
                len(ticker_batches),
                start_date,
                len(ticker_batch),
            )
            inserted, failed = self._sync_batch(start_date, ticker_batch)
            inserted_rows += inserted
            failed_tickers.update(failed)
            self.logger.info(
                "Finished batch %s/%s for start_date=%s inserted_rows=%s failed_tickers=%s",
                batch_index,
                len(ticker_batches),
                start_date,
                inserted,
                len(failed),
            )

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
                self.logger.warning("Deferring final retry for %s after batch failure: %s", ticker, exc)
                failed_tickers.add(ticker)

        return inserted_rows, failed_tickers

    def _retry_failed_tickers(
        self,
        tickers: set[str],
        *,
        fetch_start_by_ticker: dict[str, date],
        universe_tickers: set[str],
    ) -> tuple[int, set[str]]:
        inserted_rows = 0
        final_failed_tickers: set[str] = set()

        for index, ticker in enumerate(sorted(tickers), start=1):
            start_date = fetch_start_by_ticker[ticker]
            self.logger.info(
                "Final retry %s/%s for ticker=%s start_date=%s",
                index,
                len(tickers),
                ticker,
                start_date,
            )
            try:
                raw_single = self.market_data_client.download_daily_history([ticker], start_date)
                history = extract_ticker_history(raw_single, ticker)
                if history.empty:
                    raise ValueError(f"No history returned for {ticker}")
                inserted_rows += self.db_manager.upsert_historical_frame(history)
            except Exception as exc:
                self.logger.error("Permanent sync failure for %s after final retry: %s", ticker, exc)
                final_failed_tickers.add(ticker)
                if ticker in universe_tickers:
                    self.db_manager.set_ticker_status(ticker, is_active=False)

        return inserted_rows, final_failed_tickers

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

    def _sync_earnings_calendar(self, tickers: list[str]) -> None:
        for index, ticker in enumerate(sorted(tickers), start=1):
            try:
                earnings_dates = self.market_data_client.download_earnings_dates(ticker, limit=24)
                self.db_manager.replace_earnings_dates(ticker, earnings_dates)
            except Exception as exc:
                self.logger.warning("Failed to sync earnings calendar for %s: %s", ticker, exc)
            if index == len(tickers) or index % 25 == 0:
                self.logger.info(
                    "Earnings sync progress: completed_tickers=%s total_tickers=%s",
                    index,
                    len(tickers),
                )
