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


@dataclass(frozen=True)
class AnalystRevisionContext:
    ticker: str
    earnings_estimate: list[dict]
    revenue_estimate: list[dict]
    eps_trend: list[dict]
    eps_revisions: list[dict]
    growth_estimates: list[dict]
    upgrades_downgrades: list[dict]


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

    def load_revision_contexts(self, tickers: list[str]) -> dict[str, AnalystRevisionContext]:
        contexts: dict[str, AnalystRevisionContext] = {}
        for ticker in sorted({str(value).strip().upper() for value in tickers if str(value).strip()}):
            context = self.load_revision_context(ticker)
            if context is not None:
                contexts[ticker] = context
        return contexts

    def load_revision_context(self, ticker: str) -> AnalystRevisionContext | None:
        try:
            import yfinance as yf

            instrument = yf.Ticker(ticker)
            context = AnalystRevisionContext(
                ticker=ticker,
                earnings_estimate=self._fetch_table_records(
                    instrument,
                    getter_name="get_earnings_estimate",
                    attribute_name="earnings_estimate",
                ),
                revenue_estimate=self._fetch_table_records(
                    instrument,
                    getter_name="get_revenue_estimate",
                    attribute_name="revenue_estimate",
                ),
                eps_trend=self._fetch_table_records(
                    instrument,
                    getter_name="get_eps_trend",
                    attribute_name="eps_trend",
                ),
                eps_revisions=self._fetch_table_records(
                    instrument,
                    getter_name="get_eps_revisions",
                    attribute_name="eps_revisions",
                ),
                growth_estimates=self._fetch_table_records(
                    instrument,
                    getter_name="get_growth_estimates",
                    attribute_name="growth_estimates",
                ),
                upgrades_downgrades=self._fetch_table_records(
                    instrument,
                    getter_name="get_upgrades_downgrades",
                    attribute_name="upgrades_downgrades",
                ),
            )
        except Exception:
            return None
        if not any(
            [
                context.earnings_estimate,
                context.revenue_estimate,
                context.eps_trend,
                context.eps_revisions,
                context.growth_estimates,
                context.upgrades_downgrades,
            ]
        ):
            return None
        return context

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

    def _fetch_table_records(
        self,
        instrument,
        *,
        getter_name: str,
        attribute_name: str,
    ) -> list[dict]:
        raw = None
        getter = getattr(instrument, getter_name, None)
        if callable(getter):
            raw = getter()
        if raw is None:
            raw = getattr(instrument, attribute_name, None)
        return self._records_from_table(raw)

    def _records_from_table(self, raw) -> list[dict]:
        if raw is None:
            return []
        if isinstance(raw, dict):
            return [self._json_safe_dict({key: value for key, value in raw.items()})]
        if isinstance(raw, pd.Series):
            return [self._json_safe_dict(raw.to_dict())]
        if isinstance(raw, pd.DataFrame):
            if raw.empty:
                return []
            frame = raw.copy()
            if frame.index.name is None:
                frame = frame.reset_index().rename(columns={"index": "period"})
            else:
                frame = frame.reset_index()
            return [self._json_safe_dict(row) for row in frame.to_dict(orient="records")]
        return []

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

    def _json_safe_dict(self, values: dict) -> dict:
        safe: dict[str, object] = {}
        for key, value in values.items():
            safe[str(key)] = self._json_safe_value(value)
        return safe

    def _json_safe_value(self, value: object) -> object:
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        if hasattr(value, "isoformat"):
            try:
                return value.isoformat()
            except TypeError:
                pass
        if isinstance(value, (int, float, str, bool)):
            return value
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return str(value)
        return numeric if math.isfinite(numeric) else None

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
