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
    def download_history(
        self,
        tickers: list[str],
        *,
        start_date: date | None = None,
        end_date: date | None = None,
        period: str | None = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        import yfinance as yf

        final_date = end_date or date.today()
        kwargs = {
            "tickers": tickers,
            "interval": interval,
            "auto_adjust": False,
            "actions": False,
            "group_by": "ticker",
            "progress": False,
            "threads": True,
        }
        if period is not None:
            kwargs["period"] = period
        else:
            if start_date is None:
                raise ValueError("start_date is required when period is not supplied")
            kwargs["start"] = start_date.isoformat()
            kwargs["end"] = (final_date + timedelta(days=1)).isoformat()
        return yf.download(**kwargs)

    @retry(retries=3, backoff_seconds=1.0, backoff_multiplier=2.0)
    def download_daily_history(
        self,
        tickers: list[str],
        start_date: date,
        end_date: date | None = None,
    ) -> pd.DataFrame:
        return self.download_history(tickers, start_date=start_date, end_date=end_date, interval="1d")

    @retry(retries=3, backoff_seconds=1.0, backoff_multiplier=2.0)
    def download_intraday_history(self, tickers: list[str]) -> pd.DataFrame:
        return self.download_history(tickers, period="1d", interval="1m")


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
        "Datetime": "date",
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
