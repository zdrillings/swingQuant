from __future__ import annotations

import numpy as np
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


def add_sma_feature(frame: pd.DataFrame, *, feature_name: str, window: int) -> None:
    frame[feature_name] = (
        frame.groupby("ticker", group_keys=False)["close"]
        .transform(lambda series: series.rolling(window=window, min_periods=window).mean())
    )


def add_moving_average_gap_feature(
    frame: pd.DataFrame,
    *,
    feature_name: str,
    short_window: int,
    long_window: int,
) -> None:
    short_sma = (
        frame.groupby("ticker", group_keys=False)["close"]
        .transform(lambda series: series.rolling(window=short_window, min_periods=short_window).mean())
    )
    long_sma = (
        frame.groupby("ticker", group_keys=False)["close"]
        .transform(lambda series: series.rolling(window=long_window, min_periods=long_window).mean())
    )
    frame[feature_name] = (short_sma - long_sma) / long_sma


def add_moving_average_slope_feature(
    frame: pd.DataFrame,
    *,
    feature_name: str,
    window: int,
    slope_window: int,
) -> None:
    frame[feature_name] = (
        frame.groupby("ticker", group_keys=False)["close"]
        .transform(
            lambda series: series.rolling(window=window, min_periods=window)
            .mean()
            .pct_change(periods=slope_window)
        )
    )


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


def add_atr_pct_feature(frame: pd.DataFrame, *, feature_name: str, window: int) -> None:
    frame[feature_name] = pd.NA
    for _, group in frame.groupby("ticker", sort=False):
        atr_values = compute_atr(
            group["high"],
            group["low"],
            group["close"],
            window,
        )
        frame.loc[group.index, feature_name] = (atr_values / group["close"]).to_numpy()


def add_base_range_pct_feature(frame: pd.DataFrame, *, feature_name: str, window: int) -> None:
    rolling_high = (
        frame.groupby("ticker", group_keys=False)["high"]
        .transform(lambda series: series.rolling(window=window, min_periods=window).max())
    )
    rolling_low = (
        frame.groupby("ticker", group_keys=False)["low"]
        .transform(lambda series: series.rolling(window=window, min_periods=window).min())
    )
    frame[feature_name] = (rolling_high - rolling_low) / rolling_low


def add_base_atr_contraction_feature(
    frame: pd.DataFrame,
    *,
    feature_name: str,
    atr_window: int,
    recent_window: int,
    prior_window: int,
) -> None:
    frame[feature_name] = pd.NA
    for _, group in frame.groupby("ticker", sort=False):
        atr_values = compute_atr(
            group["high"],
            group["low"],
            group["close"],
            atr_window,
        )
        atr_pct = atr_values / group["close"]
        recent_avg = atr_pct.rolling(window=recent_window, min_periods=recent_window).mean()
        prior_avg = atr_pct.shift(recent_window).rolling(window=prior_window, min_periods=prior_window).mean()
        frame.loc[group.index, feature_name] = (recent_avg / prior_avg).to_numpy()


def add_volume_dryup_ratio_feature(
    frame: pd.DataFrame,
    *,
    feature_name: str,
    recent_window: int,
    prior_window: int,
) -> None:
    frame[feature_name] = (
        frame.groupby("ticker", group_keys=False)["volume"]
        .transform(
            lambda series: (
                series.rolling(window=recent_window, min_periods=recent_window).mean()
                / series.shift(recent_window).rolling(window=prior_window, min_periods=prior_window).mean()
            )
        )
    )


def add_breakout_above_high_feature(frame: pd.DataFrame, *, feature_name: str, window: int) -> None:
    prior_high = (
        frame.groupby("ticker", group_keys=False)["high"]
        .transform(lambda series: series.shift(1).rolling(window=window, min_periods=window).max())
    )
    frame[feature_name] = (frame["close"] > prior_high).astype(float)


def add_distance_above_high_feature(frame: pd.DataFrame, *, feature_name: str, window: int) -> None:
    prior_high = (
        frame.groupby("ticker", group_keys=False)["high"]
        .transform(lambda series: series.shift(1).rolling(window=window, min_periods=window).max())
    )
    frame[feature_name] = (frame["close"] / prior_high) - 1.0


def add_avg_abs_gap_pct_feature(frame: pd.DataFrame, *, feature_name: str, window: int) -> None:
    prior_close = frame.groupby("ticker", group_keys=False)["close"].shift(1)
    gap_pct = (frame["open"] / prior_close - 1.0).abs()
    frame[feature_name] = (
        gap_pct.groupby(frame["ticker"])
        .transform(lambda series: series.rolling(window=window, min_periods=window).mean())
    )


def add_max_gap_down_pct_feature(frame: pd.DataFrame, *, feature_name: str, window: int) -> None:
    prior_close = frame.groupby("ticker", group_keys=False)["close"].shift(1)
    gap_down_pct = ((prior_close - frame["open"]) / prior_close).clip(lower=0.0)
    frame[feature_name] = (
        gap_down_pct.groupby(frame["ticker"])
        .transform(lambda series: series.rolling(window=window, min_periods=window).max())
    )


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


def add_days_to_next_event_feature(
    frame: pd.DataFrame,
    *,
    feature_name: str,
    earnings_calendar: pd.DataFrame | None,
) -> None:
    frame[feature_name] = pd.NA
    if earnings_calendar is None or earnings_calendar.empty:
        return
    event_map = {
        ticker: np.sort(group["earnings_date"].to_numpy(dtype="datetime64[D]"))
        for ticker, group in earnings_calendar.groupby("ticker", sort=False)
    }
    for ticker, group in frame.groupby("ticker", sort=False):
        event_dates = event_map.get(str(ticker))
        if event_dates is None or len(event_dates) == 0:
            continue
        trading_dates = group["date"].to_numpy(dtype="datetime64[D]")
        next_indices = np.searchsorted(event_dates, trading_dates, side="left")
        feature_values = np.full(len(group.index), np.nan)
        valid_mask = next_indices < len(event_dates)
        if valid_mask.any():
            feature_values[valid_mask] = np.busday_count(
                trading_dates[valid_mask],
                event_dates[next_indices[valid_mask]],
            ).astype(float)
        frame.loc[group.index, feature_name] = feature_values


def add_days_since_last_event_feature(
    frame: pd.DataFrame,
    *,
    feature_name: str,
    earnings_calendar: pd.DataFrame | None,
) -> None:
    frame[feature_name] = pd.NA
    if earnings_calendar is None or earnings_calendar.empty:
        return
    event_map = {
        ticker: np.sort(group["earnings_date"].to_numpy(dtype="datetime64[D]"))
        for ticker, group in earnings_calendar.groupby("ticker", sort=False)
    }
    for ticker, group in frame.groupby("ticker", sort=False):
        event_dates = event_map.get(str(ticker))
        if event_dates is None or len(event_dates) == 0:
            continue
        trading_dates = group["date"].to_numpy(dtype="datetime64[D]")
        prior_indices = np.searchsorted(event_dates, trading_dates, side="right") - 1
        feature_values = np.full(len(group.index), np.nan)
        valid_mask = prior_indices >= 0
        if valid_mask.any():
            feature_values[valid_mask] = np.busday_count(
                event_dates[prior_indices[valid_mask]],
                trading_dates[valid_mask],
            ).astype(float)
        frame.loc[group.index, feature_name] = feature_values


def add_sector_breadth_features(frame: pd.DataFrame) -> None:
    for column in ["sector_pct_above_50", "sector_pct_above_200", "sector_median_roc_63"]:
        frame[column] = pd.NA
    required_columns = {"sector", "date", "sma_50_dist", "sma_200_dist", "roc_63"}
    if not required_columns.issubset(frame.columns):
        return

    eligible = frame[frame["sector"].notna()].copy()
    if eligible.empty:
        return
    eligible["above_50"] = (eligible["sma_50_dist"].astype(float) > 0.0).astype(float)
    eligible["above_200"] = (eligible["sma_200_dist"].astype(float) > 0.0).astype(float)
    breadth = (
        eligible.groupby(["sector", "date"], dropna=False)
        .agg(
            sector_pct_above_50=("above_50", "mean"),
            sector_pct_above_200=("above_200", "mean"),
            sector_median_roc_63=("roc_63", "median"),
        )
        .reset_index()
    )
    base = frame.drop(columns=["sector_pct_above_50", "sector_pct_above_200", "sector_median_roc_63"])
    merged = base.merge(
        breadth,
        on=["sector", "date"],
        how="left",
    )
    for column in ["sector_pct_above_50", "sector_pct_above_200", "sector_median_roc_63"]:
        frame[column] = merged[column]


def apply_feature_definitions(
    price_history: pd.DataFrame,
    feature_config: dict,
    earnings_calendar: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[str]]:
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
            elif feature_type == "sma":
                add_sma_feature(frame, feature_name=feature_name, window=int(params["window"]))
            elif feature_type == "moving_average_gap":
                add_moving_average_gap_feature(
                    frame,
                    feature_name=feature_name,
                    short_window=int(params["short_window"]),
                    long_window=int(params["long_window"]),
                )
            elif feature_type == "moving_average_slope":
                add_moving_average_slope_feature(
                    frame,
                    feature_name=feature_name,
                    window=int(params["window"]),
                    slope_window=int(params["slope_window"]),
                )
            elif feature_type == "rsi":
                add_rsi_feature(frame, feature_name=feature_name, window=int(params["window"]))
            elif feature_type == "ratio_to_avg":
                add_ratio_to_avg_feature(frame, feature_name=feature_name, window=int(params["window"]))
            elif feature_type == "roc":
                add_roc_feature(frame, feature_name=feature_name, window=int(params["window"]))
            elif feature_type == "atr":
                add_atr_feature(frame, feature_name=feature_name, window=int(params["window"]))
            elif feature_type == "atr_pct":
                add_atr_pct_feature(frame, feature_name=feature_name, window=int(params["window"]))
            elif feature_type == "base_range_pct":
                add_base_range_pct_feature(frame, feature_name=feature_name, window=int(params["window"]))
            elif feature_type == "base_atr_contraction":
                add_base_atr_contraction_feature(
                    frame,
                    feature_name=feature_name,
                    atr_window=int(params["atr_window"]),
                    recent_window=int(params["recent_window"]),
                    prior_window=int(params["prior_window"]),
                )
            elif feature_type == "volume_dryup_ratio":
                add_volume_dryup_ratio_feature(
                    frame,
                    feature_name=feature_name,
                    recent_window=int(params["recent_window"]),
                    prior_window=int(params["prior_window"]),
                )
            elif feature_type == "breakout_above_high":
                add_breakout_above_high_feature(frame, feature_name=feature_name, window=int(params["window"]))
            elif feature_type == "distance_above_high":
                add_distance_above_high_feature(frame, feature_name=feature_name, window=int(params["window"]))
            elif feature_type == "avg_abs_gap_pct":
                add_avg_abs_gap_pct_feature(frame, feature_name=feature_name, window=int(params["window"]))
            elif feature_type == "max_gap_down_pct":
                add_max_gap_down_pct_feature(frame, feature_name=feature_name, window=int(params["window"]))
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
            elif feature_type == "days_to_next_event":
                add_days_to_next_event_feature(
                    frame,
                    feature_name=feature_name,
                    earnings_calendar=earnings_calendar,
                )
            elif feature_type == "days_since_last_event":
                add_days_since_last_event_feature(
                    frame,
                    feature_name=feature_name,
                    earnings_calendar=earnings_calendar,
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
