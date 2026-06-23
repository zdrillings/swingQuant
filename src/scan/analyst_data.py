from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class AnalystContext:
    ticker: str
    target_mean: float | None = None
    target_median: float | None = None
    target_low: float | None = None
    target_high: float | None = None
    analyst_count: int | None = None
    recommendation: str | None = None


class AnalystDataClient:
    def load_contexts(self, tickers: list[str]) -> dict[str, AnalystContext]:
        contexts: dict[str, AnalystContext] = {}
        for ticker in sorted({str(value).strip().upper() for value in tickers if str(value).strip()}):
            context = self.load_context(ticker)
            if context is not None:
                contexts[ticker] = context
        return contexts

    def load_context(self, ticker: str) -> AnalystContext | None:
        try:
            import yfinance as yf

            instrument = yf.Ticker(ticker)
            targets = self._fetch_price_targets(instrument)
            recommendation = self._fetch_recommendation_summary(instrument)
        except Exception:
            return None
        if not targets and recommendation is None:
            return None
        return AnalystContext(
            ticker=ticker,
            target_mean=self._finite_or_none(targets.get("mean")),
            target_median=self._finite_or_none(targets.get("median")),
            target_low=self._finite_or_none(targets.get("low")),
            target_high=self._finite_or_none(targets.get("high")),
            analyst_count=self._int_or_none(
                targets.get("numberOfAnalystOpinions")
                or targets.get("number_of_analyst_opinions")
                or targets.get("analyst_count")
            ),
            recommendation=recommendation,
        )

    def _fetch_price_targets(self, instrument) -> dict[str, Any]:
        raw = None
        getter = getattr(instrument, "get_analyst_price_targets", None)
        if callable(getter):
            raw = getter()
        if raw is None:
            raw = getattr(instrument, "analyst_price_targets", None)
        if raw is None:
            return {}
        if isinstance(raw, dict):
            return {self._normalize_key(key): value for key, value in raw.items()}
        if isinstance(raw, pd.Series):
            return {self._normalize_key(key): value for key, value in raw.to_dict().items()}
        if isinstance(raw, pd.DataFrame) and not raw.empty:
            if len(raw.index) == 1:
                return {self._normalize_key(key): value for key, value in raw.iloc[0].to_dict().items()}
            if len(raw.columns) == 1:
                return {self._normalize_key(key): value for key, value in raw.iloc[:, 0].to_dict().items()}
        return {}

    def _fetch_recommendation_summary(self, instrument) -> str | None:
        getter = getattr(instrument, "get_recommendations_summary", None)
        raw = getter() if callable(getter) else getattr(instrument, "recommendations_summary", None)
        if not isinstance(raw, pd.DataFrame) or raw.empty:
            return None
        row = raw.iloc[0]
        parts = []
        for column, label in (
            ("strongBuy", "strong buy"),
            ("buy", "buy"),
            ("hold", "hold"),
            ("sell", "sell"),
            ("strongSell", "strong sell"),
        ):
            value = row.get(column)
            count = self._int_or_none(value)
            if count is not None and count > 0:
                parts.append(f"{count} {label}")
        return ", ".join(parts) if parts else None

    def _normalize_key(self, key: object) -> str:
        normalized = str(key).strip()
        mapping = {
            "current": "current",
            "targetLowPrice": "low",
            "target_low_price": "low",
            "low": "low",
            "targetHighPrice": "high",
            "target_high_price": "high",
            "high": "high",
            "targetMeanPrice": "mean",
            "target_mean_price": "mean",
            "mean": "mean",
            "targetMedianPrice": "median",
            "target_median_price": "median",
            "median": "median",
        }
        return mapping.get(normalized, normalized)

    @staticmethod
    def _finite_or_none(value: object) -> float | None:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        return numeric if math.isfinite(numeric) else None

    @staticmethod
    def _int_or_none(value: object) -> int | None:
        try:
            if pd.isna(value):
                return None
            return int(value)
        except (TypeError, ValueError):
            return None
