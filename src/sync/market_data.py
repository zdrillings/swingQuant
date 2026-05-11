from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from datetime import date, timedelta
from io import StringIO

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
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
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

    @retry(retries=3, backoff_seconds=1.0, backoff_multiplier=2.0)
    def download_earnings_dates(self, ticker: str, *, limit: int = 24) -> list[date]:
        import yfinance as yf

        instrument = yf.Ticker(ticker)
        earnings_dates: list[date] = []

        try:
            earnings_frame = instrument.get_earnings_dates(limit=limit)
        except Exception:
            earnings_frame = pd.DataFrame()

        if isinstance(earnings_frame, pd.DataFrame) and not earnings_frame.empty:
            if "Earnings Date" in earnings_frame.columns:
                raw_dates = pd.to_datetime(earnings_frame["Earnings Date"], errors="coerce")
            else:
                raw_dates = pd.to_datetime(earnings_frame.index, errors="coerce")
            earnings_dates.extend(
                timestamp.date()
                for timestamp in raw_dates.dropna().to_list()
            )

        if earnings_dates:
            return sorted(set(earnings_dates))

        try:
            calendar_payload = instrument.calendar
        except Exception:
            calendar_payload = None

        if isinstance(calendar_payload, pd.DataFrame) and not calendar_payload.empty:
            flattened = pd.to_datetime(calendar_payload.to_numpy().ravel(), errors="coerce")
            earnings_dates.extend(
                timestamp.date()
                for timestamp in flattened[~pd.isna(flattened)].tolist()
            )
        elif isinstance(calendar_payload, dict):
            raw_value = calendar_payload.get("Earnings Date") or calendar_payload.get("EarningsDate")
            if raw_value is not None:
                if not isinstance(raw_value, (list, tuple)):
                    raw_value = [raw_value]
                flattened = pd.to_datetime(list(raw_value), errors="coerce")
                earnings_dates.extend(
                    timestamp.date()
                    for timestamp in flattened[~pd.isna(flattened)].tolist()
                )

        return sorted(set(earnings_dates))


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
