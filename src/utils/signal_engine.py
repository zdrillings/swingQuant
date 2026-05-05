from __future__ import annotations

from datetime import date

import pandas as pd

from src.settings import load_feature_config
from src.utils.feature_engineering import apply_feature_definitions, compute_rsi
from src.utils.regime import regime_etf_for_sector
from src.utils.strategy import evaluate_indicator_gate


def overlay_price_history(base_history: pd.DataFrame, fresh_history: pd.DataFrame) -> pd.DataFrame:
    if base_history.empty:
        return fresh_history.copy()
    if fresh_history.empty:
        return base_history.copy()

    combined = pd.concat([base_history, fresh_history], ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])
    combined = combined.sort_values(["ticker", "date"]).drop_duplicates(
        subset=["ticker", "date"],
        keep="last",
    )
    return combined.reset_index(drop=True)


def _compute_regime_frame(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    reference = frame.loc[frame["ticker"] == ticker].sort_values("date").copy()
    if reference.empty:
        return pd.DataFrame(columns=["date", "sma_200", "regime_green"])
    reference["sma_200"] = reference["adj_close"].rolling(window=200, min_periods=200).mean()
    reference["regime_green"] = reference["adj_close"] > reference["sma_200"]
    return reference[["date", "sma_200", "regime_green"]]


def build_analysis_frame(price_history: pd.DataFrame, universe_rows: list[dict] | list[pd.Series] | list) -> tuple[pd.DataFrame, list[str]]:
    feature_config = load_feature_config()
    frame, feature_columns = apply_feature_definitions(price_history, feature_config)

    sector_map = {
        row["ticker"] if isinstance(row, dict) else row["ticker"]: (
            row["sector"] if isinstance(row, dict) else row["sector"]
        )
        for row in universe_rows
    }
    liquidity_map = {
        row["ticker"] if isinstance(row, dict) else row["ticker"]: (
            row["md_volume_30d"] if isinstance(row, dict) else row["md_volume_30d"]
        )
        for row in universe_rows
    }
    frame["sector"] = frame["ticker"].map(sector_map)
    frame["md_volume_30d"] = frame["ticker"].map(liquidity_map)

    spy_regime = _compute_regime_frame(frame, "SPY").set_index("date")
    qqq_regime = _compute_regime_frame(frame, "QQQ").set_index("date")
    frame["spy_sma_200"] = frame["date"].map(spy_regime["sma_200"]) if not spy_regime.empty else pd.NA
    frame["spy_regime_green"] = frame["date"].map(spy_regime["regime_green"]) if not spy_regime.empty else pd.NA
    frame["qqq_sma_200"] = frame["date"].map(qqq_regime["sma_200"]) if not qqq_regime.empty else pd.NA
    frame["qqq_regime_green"] = frame["date"].map(qqq_regime["regime_green"]) if not qqq_regime.empty else pd.NA
    frame["regime_etf"] = frame["sector"].map(lambda sector: regime_etf_for_sector(sector) if pd.notna(sector) else None)
    frame["regime_green"] = frame.apply(
        lambda row: row["qqq_regime_green"] if row["regime_etf"] == "QQQ" else row["spy_regime_green"],
        axis=1,
    )
    return frame, feature_columns


def latest_snapshot(frame: pd.DataFrame) -> pd.DataFrame:
    latest_date = frame["date"].max()
    return frame.loc[frame["date"] == latest_date].sort_values(["md_volume_30d", "ticker"], ascending=[False, True])


def filter_signal_candidates(frame: pd.DataFrame, indicators: dict[str, float]) -> pd.DataFrame:
    if frame.empty:
        return frame.iloc[0:0].copy()
    rows = []
    for row in frame.to_dict(orient="records"):
        passed, details = evaluate_indicator_gate(indicators, row)
        if passed:
            payload = row.copy()
            payload["indicator_details"] = details
            rows.append(payload)
    if not rows:
        empty_frame = frame.iloc[0:0].copy()
        empty_frame["indicator_details"] = pd.Series(dtype=object)
        return empty_frame
    return pd.DataFrame(rows)


def append_intraday_price(price_history: pd.DataFrame, ticker: str, current_price: float, as_of: date) -> pd.DataFrame:
    ticker_history = price_history.loc[price_history["ticker"] == ticker].sort_values("date").copy()
    if ticker_history.empty:
        raise ValueError(f"Missing price history for {ticker}")

    last_row = ticker_history.iloc[-1].to_dict()
    appended_row = {
        "ticker": ticker,
        "date": pd.Timestamp(as_of),
        "open": current_price,
        "high": max(float(last_row["high"]), current_price),
        "low": min(float(last_row["low"]), current_price),
        "close": current_price,
        "volume": float(last_row["volume"]),
        "adj_close": current_price,
    }
    ticker_history["date"] = pd.to_datetime(ticker_history["date"])
    if ticker_history.iloc[-1]["date"].date() == as_of:
        ticker_history = ticker_history.iloc[:-1]
    return pd.concat([ticker_history, pd.DataFrame([appended_row])], ignore_index=True)


def latest_rsi_2_with_intraday(price_history: pd.DataFrame, ticker: str, current_price: float, as_of: date) -> float:
    ticker_history = append_intraday_price(price_history, ticker, current_price, as_of)
    rsi_values = compute_rsi(ticker_history["close"], window=2)
    return float(rsi_values.iloc[-1])
