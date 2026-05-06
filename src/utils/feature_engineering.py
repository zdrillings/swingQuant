from __future__ import annotations

import pandas as pd


def compute_rsi(series: pd.Series, window: int) -> pd.Series:
    delta = series.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.rolling(window=window, min_periods=window).mean()
    avg_loss = losses.rolling(window=window, min_periods=window).mean()
    relative_strength = avg_gain / avg_loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + relative_strength))
    rsi = rsi.where(avg_gain.notna() & avg_loss.notna())
    rsi = rsi.mask((avg_loss == 0) & avg_gain.notna(), 100)
    rsi = rsi.mask((avg_gain == 0) & avg_loss.notna(), 0)
    return rsi.astype(float)


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            (high - low).abs(),
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(window=window, min_periods=window).mean().astype(float)


def add_pct_diff_feature(frame: pd.DataFrame, *, feature_name: str, window: int) -> None:
    rolling_average = (
        frame.groupby("ticker", group_keys=False)["close"]
        .transform(lambda series: series.rolling(window=window, min_periods=window).mean())
    )
    frame[feature_name] = (frame["close"] - rolling_average) / rolling_average


def add_rsi_feature(frame: pd.DataFrame, *, feature_name: str, window: int) -> None:
    frame[feature_name] = frame.groupby("ticker", group_keys=False)["close"].transform(
        lambda series: compute_rsi(series, window)
    )


def add_ratio_to_avg_feature(frame: pd.DataFrame, *, feature_name: str, window: int) -> None:
    rolling_volume = (
        frame.groupby("ticker", group_keys=False)["volume"]
        .transform(lambda series: series.rolling(window=window, min_periods=window).mean())
    )
    frame[feature_name] = frame["volume"] / rolling_volume


def add_roc_feature(frame: pd.DataFrame, *, feature_name: str, window: int) -> None:
    frame[feature_name] = frame.groupby("ticker", group_keys=False)["close"].transform(
        lambda series: series.pct_change(periods=window)
    )


def add_atr_feature(frame: pd.DataFrame, *, feature_name: str, window: int) -> None:
    frame[feature_name] = pd.NA
    for _, group in frame.groupby("ticker", sort=False):
        frame.loc[group.index, feature_name] = compute_atr(
            group["high"],
            group["low"],
            group["close"],
            window,
        ).to_numpy()


def add_correlation_feature(
    frame: pd.DataFrame,
    *,
    feature_name: str,
    reference_ticker: str,
    window: int,
) -> None:
    ticker_returns = frame.groupby("ticker", group_keys=False)["close"].transform(lambda series: series.pct_change())
    reference = (
        frame.loc[frame["ticker"] == reference_ticker, ["date", "close"]]
        .sort_values("date")
        .assign(reference_return=lambda df: df["close"].pct_change())
        [["date", "reference_return"]]
    )
    frame["ticker_return"] = ticker_returns
    frame["reference_return"] = frame["date"].map(reference.set_index("date")["reference_return"])
    frame[feature_name] = frame.groupby("ticker", group_keys=False).apply(
        lambda group: group["ticker_return"].rolling(window=window, min_periods=window).corr(group["reference_return"])
    ).reset_index(level=0, drop=True)
    frame.drop(columns=["ticker_return", "reference_return"], inplace=True)


def add_relative_strength_percentile_feature(
    frame: pd.DataFrame,
    *,
    feature_name: str,
    reference_ticker: str,
    window: int,
) -> None:
    frame["roc_value"] = frame.groupby("ticker", group_keys=False)["close"].transform(
        lambda series: series.pct_change(periods=window)
    )
    reference = (
        frame.loc[frame["ticker"] == reference_ticker, ["date", "roc_value"]]
        .sort_values("date")
        .rename(columns={"roc_value": "reference_roc"})
    )
    frame["reference_roc"] = frame["date"].map(reference.set_index("date")["reference_roc"])
    frame["rs_excess_return"] = frame["roc_value"] - frame["reference_roc"]
    eligible_mask = frame["ticker"] != reference_ticker
    frame.loc[eligible_mask, feature_name] = (
        frame.loc[eligible_mask]
        .groupby("date")["rs_excess_return"]
        .rank(method="average", pct=True)
        * 100.0
    )
    frame.loc[~eligible_mask, feature_name] = pd.NA
    frame.drop(columns=["roc_value", "reference_roc", "rs_excess_return"], inplace=True)


def apply_feature_definitions(price_history: pd.DataFrame, feature_config: dict) -> tuple[pd.DataFrame, list[str]]:
    frame = price_history.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values(["ticker", "date"]).reset_index(drop=True)

    for feature_group in feature_config.get("features", {}).values():
        for feature in feature_group:
            feature_name = feature["name"]
            feature_type = feature["type"]
            params = feature.get("params", {})
            if feature_type == "pct_diff":
                add_pct_diff_feature(frame, feature_name=feature_name, window=int(params["window"]))
            elif feature_type == "rsi":
                add_rsi_feature(frame, feature_name=feature_name, window=int(params["window"]))
            elif feature_type == "ratio_to_avg":
                add_ratio_to_avg_feature(frame, feature_name=feature_name, window=int(params["window"]))
            elif feature_type == "roc":
                add_roc_feature(frame, feature_name=feature_name, window=int(params["window"]))
            elif feature_type == "atr":
                add_atr_feature(frame, feature_name=feature_name, window=int(params["window"]))
            elif feature_type == "correlation":
                add_correlation_feature(
                    frame,
                    feature_name=feature_name,
                    reference_ticker=feature["ticker"],
                    window=int(params["window"]),
                )
            elif feature_type == "relative_strength_percentile":
                add_relative_strength_percentile_feature(
                    frame,
                    feature_name=feature_name,
                    reference_ticker=feature["ticker"],
                    window=int(params["window"]),
                )
            else:
                raise ValueError(f"Unsupported feature type: {feature_type}")

    feature_columns = [
        feature["name"]
        for feature_group in feature_config.get("features", {}).values()
        for feature in feature_group
    ]
    frame[feature_columns] = frame[feature_columns].apply(pd.to_numeric, errors="coerce")
    return frame, feature_columns
