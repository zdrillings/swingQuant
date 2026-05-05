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


def build_feature_frame(price_history: pd.DataFrame, feature_config: dict) -> tuple[pd.DataFrame, list[str]]:
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
            elif feature_type == "correlation":
                add_correlation_feature(
                    frame,
                    feature_name=feature_name,
                    reference_ticker=feature["ticker"],
                    window=int(params["window"]),
                )
            else:
                raise ValueError(f"Unsupported feature type: {feature_type}")

    frame["future_close_20d"] = frame.groupby("ticker", group_keys=False)["close"].shift(-20)
    frame["success"] = (frame["future_close_20d"] > frame["close"] * 1.05).astype(int)

    feature_columns = [
        feature["name"]
        for feature_group in feature_config.get("features", {}).values()
        for feature in feature_group
    ]
    frame[feature_columns] = frame[feature_columns].apply(pd.to_numeric, errors="coerce")
    required_columns = ["ticker", "date", "future_close_20d", "success"] + feature_columns
    frame = frame[required_columns].dropna().reset_index(drop=True)
    frame = frame.drop(columns=["future_close_20d"])
    return frame, feature_columns


def chronological_split(frame: pd.DataFrame, train_ratio: float = 0.7) -> tuple[pd.DataFrame, pd.DataFrame]:
    unique_dates = sorted(frame["date"].drop_duplicates().tolist())
    if len(unique_dates) < 2:
        raise ValueError("At least two distinct dates are required for chronological splitting")

    split_index = int(len(unique_dates) * train_ratio)
    split_index = min(max(split_index, 1), len(unique_dates) - 1)

    train_dates = set(unique_dates[:split_index])
    validation_dates = set(unique_dates[split_index:])
    train_frame = frame[frame["date"].isin(train_dates)].copy()
    validation_frame = frame[frame["date"].isin(validation_dates)].copy()
    return train_frame, validation_frame
