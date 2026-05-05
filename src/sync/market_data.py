from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from src.utils.retry import retry


def chunked(items: list[str], size: int) -> list[list[str]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


class MarketDataClient:
    def __init__(self, logger=None) -> None:
        self.logger = logger

    @retry(retries=3, backoff_seconds=1.0, backoff_multiplier=2.0)
    def download_daily_history(
        self,
        tickers: list[str],
        start_date: date,
        end_date: date | None = None,
    ) -> pd.DataFrame:
        import yfinance as yf

        final_date = end_date or date.today()
        return yf.download(
            tickers=tickers,
            start=start_date.isoformat(),
            end=(final_date + timedelta(days=1)).isoformat(),
            interval="1d",
            auto_adjust=False,
            actions=False,
            group_by="ticker",
            progress=False,
            threads=True,
        )


def extract_ticker_history(raw_frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if raw_frame.empty:
        return pd.DataFrame()

    frame = raw_frame.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        available = set(frame.columns.get_level_values(0))
        if ticker not in available:
            return pd.DataFrame()
        frame = frame[ticker].copy()

    if frame.empty:
        return pd.DataFrame()

    frame = frame.reset_index()
    rename_map = {
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
        "Adj Close": "adj_close",
    }
    frame = frame.rename(columns=rename_map)
    if "adj_close" not in frame.columns and "close" in frame.columns:
        frame["adj_close"] = frame["close"]

    expected = ["date", "open", "high", "low", "close", "volume", "adj_close"]
    missing_columns = [column for column in expected if column not in frame.columns]
    if missing_columns:
        return pd.DataFrame()

    frame = frame[expected].dropna(subset=["date", "close"]).copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.date
    frame["ticker"] = ticker
    return frame[["ticker", "date", "open", "high", "low", "close", "volume", "adj_close"]]
