from __future__ import annotations

import argparse
from datetime import date
import math
from pathlib import Path
import sys
import warnings

ROOT_DIR = Path(__file__).resolve().parents[1]
VENDOR_DIR = ROOT_DIR / ".vendor"
if VENDOR_DIR.exists():
    sys.path.insert(0, str(VENDOR_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd

from src.research.shortlist_bakeoff_service import MODEL_FEATURE_COLUMNS, expand_model_feature_columns
from src.research.shortlist_model_service import ShortlistModelService
from src.settings import get_settings, load_feature_config
from src.utils.db_manager import DatabaseManager


RAW_TARGET_COLUMN = "alpha_vs_sector_60d"
BINARY_TARGET_COLUMN = "beat_sector_2pct_60d_pos"
BEAT_THRESHOLD = 0.02
CANDIDATE_MODELS = ("ridge_model", "lasso_model", "xgboost_model")


def _fmt(value: object, *, places: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(number):
        return "n/a"
    return f"{number:+.{places}f}"


def _fmt_rate(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(number):
        return "n/a"
    return f"{number:.1%}"


def add_binary_beat_label(
    frame: pd.DataFrame,
    *,
    source_column: str = RAW_TARGET_COLUMN,
    target_column: str = BINARY_TARGET_COLUMN,
    threshold: float = BEAT_THRESHOLD,
) -> pd.DataFrame:
    working = frame.copy()
    source = pd.to_numeric(working[source_column], errors="coerce")
    working[target_column] = np.where(source.notna(), (source >= float(threshold)).astype(float), np.nan)
    return working


def _read_universe_snapshots(*, duckdb_path: Path, columns: list[str]) -> pd.DataFrame:
    import duckdb

    safe_columns = ", ".join(columns)
    with duckdb.connect(str(duckdb_path), read_only=True) as connection:
        return connection.execute(
            f"""
            SELECT {safe_columns}
            FROM universe_daily_snapshots
            WHERE {RAW_TARGET_COLUMN} IS NOT NULL
            ORDER BY snapshot_date, ticker
            """
        ).fetchdf()


def _read_calendar_dates(*, duckdb_path: Path) -> list[pd.Timestamp]:
    import duckdb

    with duckdb.connect(str(duckdb_path), read_only=True) as connection:
        rows = connection.execute(
            "SELECT DISTINCT snapshot_date FROM universe_daily_snapshots ORDER BY snapshot_date ASC"
        ).fetchall()
    return sorted(pd.to_datetime(pd.Series([row[0] for row in rows]), errors="coerce").dropna().dt.normalize().tolist())


def _snapshot_columns(feature_columns: list[str]) -> list[str]:
    required = [
        "snapshot_date",
        "ticker",
        "sector",
        "md_volume_30d",
        "adj_close",
        "passed_any_strategy",
        "passed_slots_json",
        RAW_TARGET_COLUMN,
        "regime_green",
    ]
    return list(dict.fromkeys([*required, *feature_columns]))


def _shortlist_config() -> dict[str, object]:
    payload = load_feature_config().get("scan_policy", {}).get("shortlist_model", {})
    horizon_days = int(payload.get("horizon_days", 60))
    return {
        "eligible_universe_mode": str(payload.get("production_eligible_universe_mode", "passed_or_trend")),
        "model_scope": str(payload.get("production_model_scope", "global")),
        "min_train_dates": int(payload.get("min_train_dates", 252)),
        "max_train_dates": int(payload.get("max_train_dates", payload.get("min_train_dates", 252))),
        "test_window_dates": int(payload.get("test_window_dates", 20)),
        "evaluation_stride_dates": int(payload.get("oos_evaluation_stride_dates", 20)),
        "label_horizon_dates": horizon_days,
        "xgboost_config": str(payload.get("production_xgboost_config", "balanced_depth4")),
    }


def _common_dates(*frames: pd.DataFrame) -> set[pd.Timestamp]:
    date_sets: list[set[pd.Timestamp]] = []
    for frame in frames:
        if frame.empty:
            continue
        dates = pd.to_datetime(frame["snapshot_date"], errors="coerce").dt.normalize().dropna()
        date_sets.append(set(dates.tolist()))
    if not date_sets:
        return set()
    common = set(date_sets[0])
    for dates in date_sets[1:]:
        common &= dates
    return common


def _filter_dates(frame: pd.DataFrame, dates: set[pd.Timestamp]) -> pd.DataFrame:
    working = frame.copy()
    working["snapshot_date"] = pd.to_datetime(working["snapshot_date"], errors="coerce").dt.normalize()
    return working[working["snapshot_date"].isin(dates)].reset_index(drop=True)


def _pooled_spearman(frame: pd.DataFrame, *, target_column: str = RAW_TARGET_COLUMN) -> float:
    score = pd.to_numeric(frame["predicted_alpha"], errors="coerce")
    target = pd.to_numeric(frame[target_column], errors="coerce")
    valid = score.notna() & target.notna()
    if int(valid.sum()) < 3 or score[valid].nunique(dropna=True) < 2 or target[valid].nunique(dropna=True) < 2:
        return float("nan")
    corr = score[valid].corr(target[valid], method="spearman")
    return float(corr) if pd.notna(corr) and math.isfinite(float(corr)) else float("nan")


def _mean_per_date_spearman(frame: pd.DataFrame, *, target_column: str = RAW_TARGET_COLUMN) -> float:
    values = [_pooled_spearman(day, target_column=target_column) for _date, day in frame.groupby("snapshot_date", sort=True)]
    values = [value for value in values if math.isfinite(value)]
    return float(pd.Series(values).mean()) if values else float("nan")


def basket_stats(
    frame: pd.DataFrame,
    *,
    top_n: int,
    raw_target_column: str = RAW_TARGET_COLUMN,
    beat_threshold: float = BEAT_THRESHOLD,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for snapshot_date, day_frame in frame.groupby("snapshot_date", sort=True):
        working = day_frame.copy()
        working["predicted_alpha"] = pd.to_numeric(working["predicted_alpha"], errors="coerce")
        working[raw_target_column] = pd.to_numeric(working[raw_target_column], errors="coerce")
        working = working.dropna(subset=["predicted_alpha", raw_target_column])
        if working.empty:
            continue
        basket = working.sort_values(["predicted_alpha", "ticker"], ascending=[False, True]).head(int(top_n))
        if basket.empty:
            continue
        rows.append(
            {
                "snapshot_date": snapshot_date,
                "basket_rows": int(len(basket.index)),
                "hit": float(basket[raw_target_column].mean() > 0.0),
                "beat_rate": float((basket[raw_target_column] >= float(beat_threshold)).mean()),
                "mean_alpha": float(basket[raw_target_column].mean()),
            }
        )
    if not rows:
        return {"dates": 0, "basket_rows": 0, "hit_rate": float("nan"), "beat_rate": float("nan"), "mean_alpha": float("nan")}
    stats = pd.DataFrame(rows)
    return {
        "dates": int(len(stats.index)),
        "basket_rows": int(stats["basket_rows"].sum()),
        "hit_rate": float(stats["hit"].mean()),
        "beat_rate": float(stats["beat_rate"].mean()),
        "mean_alpha": float(stats["mean_alpha"].mean()),
    }


def _score_logistic_model(
    service: ShortlistModelService,
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    penalty: str,
) -> pd.DataFrame | None:
    try:
        from sklearn.linear_model import LogisticRegression
    except ModuleNotFoundError:
        service.logger.warning("scikit-learn unavailable; skipping logistic classifier.")
        return None
    train_matrix, test_matrix, _feature_names, _standardized_test = service._prepare_model_matrices(
        train_frame,
        test_frame,
        feature_columns_override=feature_columns,
    )
    train_target = pd.to_numeric(train_frame[BINARY_TARGET_COLUMN], errors="coerce").to_numpy(dtype=float)
    finite_mask = np.isfinite(train_target)
    train_matrix = np.nan_to_num(train_matrix[finite_mask], nan=0.0, posinf=0.0, neginf=0.0)
    train_target = train_target[finite_mask]
    test_matrix = np.nan_to_num(test_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    unique_classes = np.unique(train_target)
    scored = test_frame.copy()
    if len(unique_classes) < 2:
        scored["predicted_alpha"] = float(unique_classes[0]) if len(unique_classes) == 1 else np.nan
    else:
        solver = "liblinear"
        model = LogisticRegression(penalty=penalty, solver=solver, C=1.0, max_iter=500, random_state=42)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*penalty.*deprecated.*", category=FutureWarning)
            warnings.filterwarnings("ignore", message=".*Inconsistent values: penalty=.*", category=UserWarning)
            model.fit(train_matrix, train_target)
        scored["predicted_alpha"] = model.predict_proba(test_matrix)[:, 1]
    scored["model_top_reasons"] = [[] for _ in range(len(scored.index))]
    scored["model_reason_summary"] = ""
    return scored


def _score_binary_xgboost(
    service: ShortlistModelService,
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    xgboost_params: dict[str, float | int],
) -> pd.DataFrame | None:
    try:
        from xgboost import XGBClassifier
    except ModuleNotFoundError:
        service.logger.warning("xgboost unavailable; skipping binary xgboost.")
        return None
    train_matrix, test_matrix, _feature_names, _standardized_test = service._prepare_model_matrices(
        train_frame,
        test_frame,
        feature_columns_override=feature_columns,
    )
    train_target = pd.to_numeric(train_frame[BINARY_TARGET_COLUMN], errors="coerce").to_numpy(dtype=float)
    finite_mask = np.isfinite(train_target)
    train_matrix = np.nan_to_num(train_matrix[finite_mask], nan=0.0, posinf=0.0, neginf=0.0)
    train_target = train_target[finite_mask]
    test_matrix = np.nan_to_num(test_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    unique_classes = np.unique(train_target)
    scored = test_frame.copy()
    if len(unique_classes) < 2:
        scored["predicted_alpha"] = float(unique_classes[0]) if len(unique_classes) == 1 else np.nan
    else:
        params: dict[str, float | int | str] = {
            "n_estimators": 150,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 1.0,
            "reg_lambda": 1.0,
            "random_state": 42,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "n_jobs": 1,
        }
        params.update(xgboost_params)
        params["objective"] = "binary:logistic"
        params["eval_metric"] = "logloss"
        model = XGBClassifier(**params)
        model.fit(train_matrix, train_target, verbose=False)
        scored["predicted_alpha"] = model.predict_proba(test_matrix)[:, 1]
    scored["model_top_reasons"] = [[] for _ in range(len(scored.index))]
    scored["model_reason_summary"] = ""
    return scored


def _score_binary_by_scope(
    service: ShortlistModelService,
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    *,
    model_name: str,
    feature_columns: list[str],
    model_scope: str,
    xgboost_params: dict[str, float | int],
) -> pd.DataFrame | None:
    def score(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame | None:
        if model_name == "ridge_model":
            return _score_logistic_model(service, train, test, feature_columns=feature_columns, penalty="l2")
        if model_name == "lasso_model":
            return _score_logistic_model(service, train, test, feature_columns=feature_columns, penalty="l1")
        if model_name == "xgboost_model":
            return _score_binary_xgboost(service, train, test, feature_columns=feature_columns, xgboost_params=xgboost_params)
        raise ValueError(f"Unsupported binary model_name={model_name}")

    if model_scope == "sector_specific":
        frames: list[pd.DataFrame] = []
        for sector, sector_test in test_frame.groupby("sector", sort=False):
            sector_train = train_frame[train_frame["sector"] == sector].copy()
            scoped_train = sector_train if len(sector_train.index) >= 120 and sector_train["snapshot_date"].nunique() >= 40 else train_frame
            scored = score(scoped_train, sector_test.copy())
            if scored is not None and not scored.empty:
                frames.append(scored)
        return pd.concat(frames, ignore_index=True) if frames else None
    return score(train_frame, test_frame)


def _walk_forward_binary_predictions(
    service: ShortlistModelService,
    frame: pd.DataFrame,
    *,
    model_name: str,
    config: dict[str, object],
    feature_columns: list[str],
) -> pd.DataFrame | None:
    dates = sorted(frame["snapshot_date"].drop_duplicates().tolist())
    folds: list[pd.DataFrame] = []
    start_index = int(config["min_train_dates"])
    stride = max(int(config["evaluation_stride_dates"]), 1)
    label_embargo = max(int(config["label_horizon_dates"]), 0)
    xgboost_params = {**service._xgboost_params_for_config(str(config["xgboost_config"])), "n_jobs": 1}
    while start_index < len(dates):
        test_dates = dates[start_index : start_index + max(int(config["test_window_dates"]), 1)]
        train_end_index = max(0, start_index - label_embargo)
        train_pool_dates = dates[:train_end_index]
        if len(train_pool_dates) < int(config["min_train_dates"]):
            start_index += stride
            continue
        if int(config["max_train_dates"]) > 0:
            train_pool_dates = train_pool_dates[-int(config["max_train_dates"]) :]
        train_frame = frame[frame["snapshot_date"].isin(set(train_pool_dates))].copy()
        train_frame = service._stride_training_labels(
            train_frame,
            dates=dates,
            anchor_index=start_index,
            label_horizon_dates=label_embargo,
        )
        test_frame = frame[frame["snapshot_date"].isin(set(test_dates))].copy()
        ic_survivors = service._feature_ic_survivors_from_frame(
            train_frame,
            target_column=BINARY_TARGET_COLUMN,
            min_feature_ic=service._load_min_feature_ic(),
            min_observation_fraction=service._load_min_feature_ic_observation_fraction(),
            feature_columns_override=feature_columns,
        )
        allowed = set(feature_columns)
        fold_feature_columns = [feature for feature in ic_survivors if feature in allowed]
        if not fold_feature_columns:
            start_index += stride
            continue
        scored = _score_binary_by_scope(
            service,
            train_frame,
            test_frame,
            model_name=model_name,
            feature_columns=fold_feature_columns,
            model_scope=str(config["model_scope"]),
            xgboost_params=xgboost_params,
        )
        if scored is not None and not scored.empty:
            scored["model_name"] = model_name
            scored["trained_label"] = "binary_beat_label"
            folds.append(
                scored[
                    [
                        "snapshot_date",
                        "ticker",
                        "sector",
                        "md_volume_30d",
                        RAW_TARGET_COLUMN,
                        BINARY_TARGET_COLUMN,
                        "predicted_alpha",
                        "model_top_reasons",
                        "model_reason_summary",
                    ]
                ].copy()
            )
        start_index += stride
    return pd.concat(folds, ignore_index=True) if folds else None


def _run_regression_model(
    service: ShortlistModelService,
    eligible: pd.DataFrame,
    *,
    model_name: str,
    config: dict[str, object],
    feature_columns: list[str],
) -> pd.DataFrame:
    xgboost_params = None
    if model_name == "xgboost_model":
        xgboost_params = {**service._xgboost_params_for_config(str(config["xgboost_config"])), "n_jobs": 1}
    predictions = service._walk_forward_predictions(
        eligible,
        target_column=RAW_TARGET_COLUMN,
        evaluation_target_column=RAW_TARGET_COLUMN,
        model_name=model_name,
        min_train_dates=int(config["min_train_dates"]),
        max_train_dates=int(config["max_train_dates"]),
        test_window_dates=int(config["test_window_dates"]),
        evaluation_stride_dates=int(config["evaluation_stride_dates"]),
        label_horizon_dates=int(config["label_horizon_dates"]),
        model_scope=str(config["model_scope"]),
        xgboost_params=xgboost_params,
        feature_columns_override=feature_columns,
        min_feature_ic=service._load_min_feature_ic(),
        min_feature_ic_observation_fraction=service._load_min_feature_ic_observation_fraction(),
        regime_matching_mode="off",
        min_regime_train_dates=service._load_min_regime_train_dates(),
        regime_transition_purge_mode="off",
    )
    if predictions is None or predictions.empty:
        raise SystemExit(f"{model_name} regression produced no OOS predictions")
    targets = eligible[["snapshot_date", "ticker", RAW_TARGET_COLUMN, BINARY_TARGET_COLUMN]].copy()
    predictions = predictions.drop(columns=[RAW_TARGET_COLUMN, BINARY_TARGET_COLUMN], errors="ignore").merge(
        targets,
        on=["snapshot_date", "ticker"],
        how="left",
    )
    predictions["model_name"] = model_name
    predictions["trained_label"] = "raw_alpha_regression"
    return predictions


def _prediction_summary(*, model: str, label: str, frame: pd.DataFrame) -> dict[str, object]:
    return {
        "model": model,
        "trained_label": label,
        "dates": int(frame["snapshot_date"].nunique()) if not frame.empty else 0,
        "rows": int(len(frame.index)),
        "pooled_raw_spearman": _pooled_spearman(frame, target_column=RAW_TARGET_COLUMN),
        "per_date_raw_spearman": _mean_per_date_spearman(frame, target_column=RAW_TARGET_COLUMN),
    }


def _comparison_rows(predictions: dict[tuple[str, str], pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    spearman_rows: list[dict[str, object]] = []
    basket_rows: list[dict[str, object]] = []
    for model in CANDIDATE_MODELS:
        regression = predictions.get((model, "raw_alpha_regression"), pd.DataFrame())
        binary = predictions.get((model, "binary_beat_label"), pd.DataFrame())
        dates = _common_dates(regression, binary)
        regression = _filter_dates(regression, dates)
        binary = _filter_dates(binary, dates)
        regression_summary = _prediction_summary(model=model, label="raw_alpha_regression", frame=regression)
        binary_summary = _prediction_summary(model=model, label="binary_beat_label", frame=binary)
        delta = float(binary_summary["pooled_raw_spearman"]) - float(regression_summary["pooled_raw_spearman"])
        spearman_rows.append({**regression_summary, "delta_binary_minus_regression": float("nan")})
        spearman_rows.append({**binary_summary, "delta_binary_minus_regression": delta})
        for label, frame in (("raw_alpha_regression", regression), ("binary_beat_label", binary)):
            for top_n in (2, 10):
                stats = basket_stats(frame, top_n=top_n)
                basket_rows.append({"model": model, "trained_label": label, "top_n": top_n, **stats})
    basket = pd.DataFrame(basket_rows)
    controls = basket[basket["trained_label"].eq("raw_alpha_regression")][
        ["model", "top_n", "hit_rate", "beat_rate", "mean_alpha"]
    ].rename(
        columns={
            "hit_rate": "regression_hit_rate",
            "beat_rate": "regression_beat_rate",
            "mean_alpha": "regression_mean_alpha",
        }
    )
    basket = basket.merge(controls, on=["model", "top_n"], how="left")
    basket["delta_hit_rate"] = pd.to_numeric(basket["hit_rate"], errors="coerce") - pd.to_numeric(
        basket["regression_hit_rate"], errors="coerce"
    )
    basket["delta_beat_rate"] = pd.to_numeric(basket["beat_rate"], errors="coerce") - pd.to_numeric(
        basket["regression_beat_rate"], errors="coerce"
    )
    basket["delta_mean_alpha"] = pd.to_numeric(basket["mean_alpha"], errors="coerce") - pd.to_numeric(
        basket["regression_mean_alpha"], errors="coerce"
    )
    return pd.DataFrame(spearman_rows), basket


def _verdict(spearman: pd.DataFrame, basket: pd.DataFrame, *, spearman_threshold: float = 0.005, rate_threshold: float = 0.02) -> tuple[str, str]:
    binary = spearman[spearman["trained_label"].eq("binary_beat_label")].copy()
    deltas = pd.to_numeric(binary["delta_binary_minus_regression"], errors="coerce").dropna()
    best_delta = float(deltas.max()) if not deltas.empty else float("nan")
    rate_rows = basket[basket["trained_label"].eq("binary_beat_label")].copy()
    hit_deltas = pd.to_numeric(rate_rows["delta_hit_rate"], errors="coerce").dropna()
    beat_deltas = pd.to_numeric(rate_rows["delta_beat_rate"], errors="coerce").dropna()
    best_hit_delta = float(hit_deltas.max()) if not hit_deltas.empty else float("nan")
    best_beat_delta = float(beat_deltas.max()) if not beat_deltas.empty else float("nan")
    if (not deltas.empty and (deltas >= spearman_threshold).any()) or (
        (not hit_deltas.empty and (hit_deltas >= rate_threshold).any())
        or (not beat_deltas.empty and (beat_deltas >= rate_threshold).any())
    ):
        return (
            "binary beat label wins",
            f"best Spearman delta {_fmt(best_delta)}; best hit delta {_fmt_rate(best_hit_delta)}; best beat delta {_fmt_rate(best_beat_delta)}",
        )
    return (
        "binary beat label ties",
        f"best Spearman delta {_fmt(best_delta)}; best hit delta {_fmt_rate(best_hit_delta)}; best beat delta {_fmt_rate(best_beat_delta)}",
    )


def _render_report(
    *,
    output_path: Path,
    spearman: pd.DataFrame,
    basket: pd.DataFrame,
    eligible_rows: int,
    eligible_dates: int,
    config: dict[str, object],
) -> str:
    verdict, reason = _verdict(spearman, basket)
    lines = [
        "# C3 Binary Beat Label Bakeoff",
        "",
        f"- generated_at: {date.today().isoformat()}",
        "- data_access: read-only DuckDB; no `./sq` writes; no `data/` mutation",
        "- experiment: train raw 60d alpha regression twins versus P(alpha_vs_sector_60d >= +2%) classifiers, evaluate probabilities against raw 60d alpha on the honest stride-60 OOS grid",
        f"- target_raw: {RAW_TARGET_COLUMN}",
        f"- target_binary: {BINARY_TARGET_COLUMN}",
        f"- binary_threshold: {BEAT_THRESHOLD:.4f}",
        f"- candidate_models: {', '.join(CANDIDATE_MODELS)}",
        f"- feature_set: current production MODEL_FEATURE_COLUMNS with A4 interactions",
        f"- eligible_universe_mode: {config['eligible_universe_mode']}",
        f"- model_scope: {config['model_scope']}",
        f"- min_train_dates: {config['min_train_dates']}",
        f"- max_train_dates: {config['max_train_dates']}",
        f"- test_window_dates: {config['test_window_dates']}",
        f"- evaluation_stride_dates: {config['evaluation_stride_dates']}",
        f"- label_horizon_dates: {config['label_horizon_dates']}",
        f"- xgboost_config: {config['xgboost_config']}",
        f"- eligible_rows: {eligible_rows}",
        f"- eligible_dates: {eligible_dates}",
        f"- verdict: {verdict} ({reason})",
        "",
        "## Honest-Grid Spearman",
        "",
        "| model | trained_label | honest_dates | honest_rows | pooled_raw_spearman | per_date_raw_spearman | delta_binary_minus_regression |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in spearman.sort_values(["model", "trained_label"]).itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.trained_label} | {int(row.dates)} | {int(row.rows)} | "
            f"{_fmt(row.pooled_raw_spearman)} | {_fmt(row.per_date_raw_spearman)} | "
            f"{_fmt(row.delta_binary_minus_regression)} |"
        )
    lines.extend(
        [
            "",
            "## Basket Hit/Beat",
            "",
            "| model | trained_label | top_n | honest_dates | basket_rows | hit_rate | beat_rate | mean_alpha | delta_hit_rate | delta_beat_rate | delta_mean_alpha |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in basket.sort_values(["model", "top_n", "trained_label"]).itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.trained_label} | {int(row.top_n)} | {int(row.dates)} | {int(row.basket_rows)} | "
            f"{_fmt_rate(row.hit_rate)} | {_fmt_rate(row.beat_rate)} | {_fmt(row.mean_alpha)} | "
            f"{_fmt_rate(row.delta_hit_rate)} | {_fmt_rate(row.delta_beat_rate)} | {_fmt(row.delta_mean_alpha)} |"
        )
    lines.extend(["", "## Verdict", "", f"{verdict}: {reason}.", ""])
    text = "\n".join(lines)
    output_path.write_text(text, encoding="utf-8")
    return text


def run(*, output_path: Path) -> dict[str, object]:
    settings = get_settings()
    service = ShortlistModelService(DatabaseManager(paths=settings.paths))
    feature_columns = service._filter_model_feature_columns(expand_model_feature_columns(MODEL_FEATURE_COLUMNS))
    base_feature_columns = [
        feature
        for feature in service._filter_model_feature_columns(MODEL_FEATURE_COLUMNS)
        if not str(feature).startswith("a4_")
    ]
    frame = _read_universe_snapshots(
        duckdb_path=settings.paths.duckdb_path,
        columns=_snapshot_columns(base_feature_columns),
    )
    config = _shortlist_config()
    frame = service._add_a4_regime_interaction_features(
        service._prepare_snapshot_frame(frame),
        horizon_sessions=int(config["label_horizon_dates"]),
    )
    frame = add_binary_beat_label(frame)
    eligible = service._build_matured_eligible_universe(
        frame,
        target_column=RAW_TARGET_COLUMN,
        eligible_universe_mode=str(config["eligible_universe_mode"]),
    )
    calendar_dates = _read_calendar_dates(duckdb_path=settings.paths.duckdb_path)
    predictions: dict[tuple[str, str], pd.DataFrame] = {}
    for model_name in CANDIDATE_MODELS:
        print(f"training {model_name} raw_alpha_regression", file=sys.stderr, flush=True)
        regression = _run_regression_model(
            service,
            eligible,
            model_name=model_name,
            config=config,
            feature_columns=feature_columns,
        )
        predictions[(model_name, "raw_alpha_regression")] = service._non_overlapping_oos_predictions(
            regression,
            horizon_days=int(config["label_horizon_dates"]),
            calendar_dates=calendar_dates,
        )
        print(f"training {model_name} binary_beat_label", file=sys.stderr, flush=True)
        binary = _walk_forward_binary_predictions(
            service,
            eligible,
            model_name=model_name,
            config=config,
            feature_columns=feature_columns,
        )
        if binary is None or binary.empty:
            raise SystemExit(f"{model_name} binary classifier produced no OOS predictions")
        predictions[(model_name, "binary_beat_label")] = service._non_overlapping_oos_predictions(
            binary,
            horizon_days=int(config["label_horizon_dates"]),
            calendar_dates=calendar_dates,
        )
    spearman, basket = _comparison_rows(predictions)
    _render_report(
        output_path=output_path,
        spearman=spearman,
        basket=basket,
        eligible_rows=len(eligible.index),
        eligible_dates=int(eligible["snapshot_date"].nunique()),
        config=config,
    )
    verdict, reason = _verdict(spearman, basket)
    return {"output_path": str(output_path), "verdict": verdict, "verdict_reason": reason}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the C3 binary beat label bakeoff.")
    parser.add_argument("--output", type=Path, default=Path("reports/c3_binary_beat_label.md"))
    args = parser.parse_args()
    result = run(output_path=args.output)
    print("wrote {output_path} verdict={verdict}: {verdict_reason}".format(**result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
