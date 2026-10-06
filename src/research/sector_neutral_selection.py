from __future__ import annotations

import math

import pandas as pd


UNKNOWN_SECTOR = "UNKNOWN"


def select_global_top_n(
    frame: pd.DataFrame,
    *,
    top_n: int,
    score_column: str = "predicted_alpha",
) -> pd.DataFrame:
    if frame.empty or int(top_n) <= 0 or score_column not in frame.columns:
        return frame.iloc[0:0].copy()
    working = frame.copy()
    working[score_column] = pd.to_numeric(working[score_column], errors="coerce")
    working = working.dropna(subset=[score_column]).copy()
    if working.empty:
        return working
    ticker = working["ticker"].astype(str) if "ticker" in working.columns else pd.Series("", index=working.index)
    working["_ticker_sort"] = ticker
    selected = working.sort_values(
        [score_column, "_ticker_sort"],
        ascending=[False, True],
    ).head(max(int(top_n), 0))
    return selected.drop(columns=["_ticker_sort"], errors="ignore").copy()


def select_sector_neutral_top_n(
    frame: pd.DataFrame,
    *,
    top_n: int,
    sector_column: str = "sector",
    score_column: str = "predicted_alpha",
) -> pd.DataFrame:
    if frame.empty or int(top_n) <= 0 or score_column not in frame.columns:
        return frame.iloc[0:0].copy()
    working = frame.copy()
    working[score_column] = pd.to_numeric(working[score_column], errors="coerce")
    working = working.dropna(subset=[score_column]).copy()
    if working.empty:
        return working
    if sector_column in working.columns:
        sector = working[sector_column].fillna(UNKNOWN_SECTOR).astype(str).str.strip()
        sector = sector.where(sector.ne(""), UNKNOWN_SECTOR)
    else:
        sector = pd.Series(UNKNOWN_SECTOR, index=working.index)
    ticker = working["ticker"].astype(str) if "ticker" in working.columns else pd.Series("", index=working.index)
    working["_sector_bucket"] = sector
    working["_ticker_sort"] = ticker
    sector_winners = (
        working.sort_values(
            ["_sector_bucket", score_column, "_ticker_sort"],
            ascending=[True, False, True],
        )
        .groupby("_sector_bucket", sort=True, dropna=False)
        .head(1)
        .copy()
    )
    limit = min(max(int(top_n), 0), int(sector_winners["_sector_bucket"].nunique(dropna=False)))
    selected = sector_winners.sort_values(
        [score_column, "_ticker_sort"],
        ascending=[False, True],
    ).head(limit)
    return selected.drop(columns=["_sector_bucket", "_ticker_sort"], errors="ignore").copy()


def deoverlap_oos_predictions(
    frame: pd.DataFrame,
    *,
    horizon_days: int,
    date_column: str = "snapshot_date",
    ticker_column: str = "ticker",
    calendar_dates: list | pd.Series | None = None,
) -> pd.DataFrame:
    if frame.empty or date_column not in frame.columns or ticker_column not in frame.columns:
        return frame.copy()
    working = frame.copy()
    working[date_column] = pd.to_datetime(working[date_column], errors="coerce").dt.normalize()
    working[ticker_column] = working[ticker_column].astype(str).str.strip()
    working = working.dropna(subset=[date_column])
    working = working[working[ticker_column].ne("")].copy()
    if working.empty:
        return working
    if calendar_dates is None:
        calendar_values = working[date_column].dropna().drop_duplicates().tolist()
    else:
        parsed = pd.to_datetime(pd.Series(list(calendar_dates)), errors="coerce").dt.normalize().dropna()
        calendar_values = parsed.drop_duplicates().tolist()
    calendar_values = sorted(set(calendar_values) | set(working[date_column].dropna().tolist()))
    date_index = {date_value: index for index, date_value in enumerate(calendar_values)}
    working["_date_index"] = working[date_column].map(date_index)
    working = working.dropna(subset=["_date_index"]).copy()
    working["_date_index"] = working["_date_index"].astype(int)
    horizon = max(int(horizon_days), 1)
    keep_keys: set[tuple[str, pd.Timestamp]] = set()
    for ticker, ticker_frame in working.groupby(ticker_column, sort=True):
        dated = ticker_frame[[date_column, "_date_index"]].drop_duplicates().sort_values("_date_index")
        last_kept: int | None = None
        for _, row in dated.iterrows():
            index_value = int(row["_date_index"])
            snapshot_date = pd.Timestamp(row[date_column])
            if last_kept is None or index_value - last_kept >= horizon:
                keep_keys.add((str(ticker), snapshot_date))
                last_kept = index_value
    mask = [
        (str(row[ticker_column]), pd.Timestamp(row[date_column])) in keep_keys
        for _, row in working[[ticker_column, date_column]].iterrows()
    ]
    return working.loc[mask].drop(columns=["_date_index"], errors="ignore").reset_index(drop=True)


def evaluate_selection_modes(
    predictions: pd.DataFrame,
    *,
    top_n: int,
    target_column: str,
    score_column: str = "predicted_alpha",
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    selectors = {
        "global_top_n": select_global_top_n,
        "sector_neutral": select_sector_neutral_top_n,
    }
    for model_name, model_frame in predictions.groupby("model_name", sort=True):
        for selection_mode, selector in selectors.items():
            daily_rows: list[dict[str, float]] = []
            for _, day_frame in model_frame.groupby("snapshot_date", sort=True):
                picks = selector(day_frame, top_n=top_n, score_column=score_column)
                target = pd.to_numeric(picks.get(target_column), errors="coerce").dropna()
                universe = pd.to_numeric(day_frame.get(target_column), errors="coerce").dropna()
                if target.empty or universe.empty:
                    continue
                sector = picks.get("sector", pd.Series(UNKNOWN_SECTOR, index=picks.index))
                sector = sector.fillna(UNKNOWN_SECTOR).astype(str).str.strip().replace({"": UNKNOWN_SECTOR})
                sector_weights = sector.value_counts(normalize=True)
                daily_rows.append(
                    {
                        "basket_mean_alpha": float(target.mean()),
                        "hit_rate": float((target > 0.0).mean()),
                        "beat_universe": float(target.mean() > universe.mean()),
                        "top_sector_share": float(sector_weights.max()) if not sector_weights.empty else float("nan"),
                        "pick_count": float(len(picks.index)),
                    }
                )
            if not daily_rows:
                rows.append(_empty_result(str(model_name), selection_mode))
                continue
            daily = pd.DataFrame(daily_rows)
            rows.append(
                {
                    "model_name": str(model_name),
                    "selection_mode": selection_mode,
                    "dates": int(len(daily.index)),
                    "avg_pick_count": float(daily["pick_count"].mean()),
                    "basket_mean_alpha": float(daily["basket_mean_alpha"].mean()),
                    "hit_rate": float(daily["hit_rate"].mean()),
                    "beat_rate": float(daily["beat_universe"].mean()),
                    "top_sector_share": float(daily["top_sector_share"].mean()),
                }
            )
    return pd.DataFrame(rows)


def verdict_for_model(results: pd.DataFrame, model_name: str) -> str:
    model = results[results["model_name"].astype(str) == str(model_name)].copy()
    if model.empty or set(model["selection_mode"]) != {"global_top_n", "sector_neutral"}:
        return "unavailable"
    global_row = model[model["selection_mode"] == "global_top_n"].iloc[0]
    sector_row = model[model["selection_mode"] == "sector_neutral"].iloc[0]
    money_columns = ("basket_mean_alpha", "hit_rate", "beat_rate")
    money_deltas = [
        float(sector_row[column]) - float(global_row[column])
        for column in money_columns
        if _finite(sector_row[column]) and _finite(global_row[column])
    ]
    concentration_delta = float(sector_row["top_sector_share"]) - float(global_row["top_sector_share"])
    if money_deltas and all(delta >= -1e-12 for delta in money_deltas) and concentration_delta < -1e-12:
        return "sector-neutral wins or ties with better concentration"
    return "sector-neutral loses on money statistics"


def _empty_result(model_name: str, selection_mode: str) -> dict[str, object]:
    return {
        "model_name": model_name,
        "selection_mode": selection_mode,
        "dates": 0,
        "avg_pick_count": float("nan"),
        "basket_mean_alpha": float("nan"),
        "hit_rate": float("nan"),
        "beat_rate": float("nan"),
        "top_sector_share": float("nan"),
    }


def _finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False
