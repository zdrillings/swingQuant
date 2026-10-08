from __future__ import annotations

import math

import pandas as pd


def chronological_isotonic_probability(
    frame: pd.DataFrame,
    *,
    date_column: str = "snapshot_date",
    score_column: str = "predicted_alpha",
    target_column: str = "alpha_vs_sector_60d",
    output_column: str = "calibrated_p_beat_sector_oos",
    min_train_rows: int = 50,
    fallback_probability: float = 0.5,
) -> pd.DataFrame:
    working = frame.copy()
    working[output_column] = float("nan")
    if working.empty or date_column not in working.columns:
        return working

    working[date_column] = pd.to_datetime(working[date_column], errors="coerce").dt.normalize()
    dates = sorted(working[date_column].dropna().drop_duplicates().tolist())
    for snapshot_date in dates:
        train_mask = working[date_column].lt(snapshot_date)
        test_mask = working[date_column].eq(snapshot_date)
        train_scores = pd.to_numeric(working.loc[train_mask, score_column], errors="coerce")
        train_targets = pd.to_numeric(working.loc[train_mask, target_column], errors="coerce")
        train_labels = train_targets.gt(0.0).astype(float)
        valid_train = train_scores.notna() & train_targets.notna()
        test_scores = pd.to_numeric(working.loc[test_mask, score_column], errors="coerce")

        if int(valid_train.sum()) >= int(min_train_rows) and train_labels[valid_train].nunique(dropna=True) > 1:
            calibrated = _fit_isotonic_predict(
                train_scores[valid_train].astype(float),
                train_labels[valid_train].astype(float),
                test_scores.astype(float),
            )
            working.loc[test_mask, output_column] = calibrated
        else:
            fallback = (
                float(train_labels[valid_train].mean())
                if int(valid_train.sum()) > 0
                else float(fallback_probability)
            )
            working.loc[test_mask, output_column] = fallback

    working[output_column] = pd.to_numeric(working[output_column], errors="coerce").clip(lower=0.0, upper=1.0)
    return working


def _fit_isotonic_predict(train_scores: pd.Series, train_labels: pd.Series, test_scores: pd.Series) -> pd.Series:
    try:
        from sklearn.isotonic import IsotonicRegression
    except ModuleNotFoundError as exc:  # pragma: no cover - sklearn is a project dependency in normal runs.
        raise RuntimeError("scikit-learn is required for isotonic calibration") from exc

    valid_test = test_scores.notna()
    predictions = pd.Series(float("nan"), index=test_scores.index, dtype="float64")
    if not bool(valid_test.any()):
        return predictions
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(train_scores.astype(float), train_labels.astype(float))
    values = calibrator.predict(test_scores[valid_test].astype(float))
    predictions.loc[valid_test] = values
    return predictions


def calibrated_probability_monotonic_by_score(frame: pd.DataFrame, *, score_column: str, probability_column: str) -> bool:
    if frame.empty or score_column not in frame.columns or probability_column not in frame.columns:
        return True
    ordered = frame.copy()
    ordered[score_column] = pd.to_numeric(ordered[score_column], errors="coerce")
    ordered[probability_column] = pd.to_numeric(ordered[probability_column], errors="coerce")
    ordered = ordered.dropna(subset=[score_column, probability_column]).sort_values(score_column)
    if ordered.empty:
        return True
    diffs = ordered[probability_column].diff().dropna()
    return bool(diffs.ge(-1e-12).all())


def finite_float(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")
