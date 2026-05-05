from __future__ import annotations

import pandas as pd

from src.utils.feature_engineering import apply_feature_definitions


def build_feature_frame(price_history: pd.DataFrame, feature_config: dict) -> tuple[pd.DataFrame, list[str]]:
    frame, feature_columns = apply_feature_definitions(price_history, feature_config)
    frame["future_close_20d"] = frame.groupby("ticker", group_keys=False)["close"].shift(-20)
    frame["success"] = (frame["future_close_20d"] > frame["close"] * 1.05).astype(int)
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
