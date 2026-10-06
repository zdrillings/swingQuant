from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
VENDOR_DIR = ROOT_DIR / ".vendor"
if VENDOR_DIR.exists():
    sys.path.insert(0, str(VENDOR_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd

from src.research.shortlist_bakeoff_service import (
    expand_model_feature_columns,
    model_feature_columns_for_profile,
)
from src.research.shortlist_model_service import PROMOTION_BASKET_SIZE, ShortlistModelService
from src.settings import get_settings, load_feature_config
from src.utils.db_manager import DatabaseManager


TARGET_COLUMN = "alpha_vs_sector_60d"
CANDIDATE_MODELS = ("ridge_model", "lasso_model", "xgboost_model")
WEIGHTED_MODELS = {"ridge_model", "lasso_model"}
BASELINE_TRAIN_DATES = 252
SHORT_TRAIN_DATES = 126
RECENCY_HALF_LIFE_DATES = 63


def _fmt(value: object, *, places: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(number):
        return "n/a"
    return f"{number:+.{places}f}"


def _shortlist_config() -> dict[str, object]:
    payload = load_feature_config().get("scan_policy", {}).get("shortlist_model", {})
    horizon = int(payload.get("horizon_days", 60))
    return {
        "eligible_universe_mode": str(payload.get("production_eligible_universe_mode", "passed_or_trend")),
        "label_horizon_dates": horizon,
        "min_train_dates": int(payload.get("min_train_dates", BASELINE_TRAIN_DATES)),
        "test_window_dates": int(payload.get("test_window_dates", 20)),
        "evaluation_stride_dates": int(payload.get("oos_evaluation_stride_dates", horizon)),
        "model_scope": "global",
        "xgboost_config": str(payload.get("production_xgboost_config", "balanced_depth4")),
        "feature_profile": str(payload.get("production_feature_profile", "full")),
    }


def _read_snapshots(*, duckdb_path: Path) -> pd.DataFrame:
    import duckdb

    with duckdb.connect(str(duckdb_path), read_only=True) as connection:
        return connection.execute(
            f"""
            SELECT *
            FROM universe_daily_snapshots
            WHERE {TARGET_COLUMN} IS NOT NULL
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


def recency_weights_by_snapshot_date(
    snapshot_dates: pd.Series,
    *,
    half_life_dates: int = RECENCY_HALF_LIFE_DATES,
    reference_dates: list[pd.Timestamp] | None = None,
) -> np.ndarray:
    """Return exponential weights where the most recent date receives weight 1.0."""
    parsed = pd.to_datetime(snapshot_dates, errors="coerce").dt.normalize()
    if parsed.empty:
        return np.array([], dtype=float)
    if reference_dates is None:
        unique_dates = sorted(parsed.dropna().drop_duplicates().tolist())
    else:
        unique_dates = sorted(
            pd.to_datetime(pd.Series(reference_dates), errors="coerce").dropna().dt.normalize().drop_duplicates().tolist()
        )
    if not unique_dates:
        return np.ones(len(parsed.index), dtype=float)
    date_index = {date_value: index for index, date_value in enumerate(unique_dates)}
    valid_row_dates = [pd.Timestamp(value).normalize() for value in parsed.dropna().tolist()]
    latest_index = max(date_index.get(value, len(unique_dates) - 1) for value in valid_row_dates) if valid_row_dates else len(unique_dates) - 1
    half_life = max(int(half_life_dates), 1)
    weights: list[float] = []
    for value in parsed.tolist():
        if pd.isna(value):
            weights.append(0.0)
            continue
        normalized = pd.Timestamp(value).normalize()
        if normalized not in date_index:
            weights.append(0.0)
            continue
        age = latest_index - int(date_index[normalized])
        weights.append(float(math.exp(-math.log(2.0) * age / half_life)))
    return np.asarray(weights, dtype=float)


def _score_weighted_ridge(
    service: ShortlistModelService,
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    *,
    target_column: str,
    feature_columns: list[str],
    half_life_dates: int,
    reference_dates: list[pd.Timestamp],
) -> pd.DataFrame:
    train_matrix, test_matrix, feature_names, _ = service._prepare_model_matrices(
        train_frame,
        test_frame,
        feature_columns_override=feature_columns,
    )
    target = pd.to_numeric(train_frame[target_column], errors="coerce").to_numpy(dtype=float)
    sample_weight = recency_weights_by_snapshot_date(
        train_frame["snapshot_date"],
        half_life_dates=half_life_dates,
        reference_dates=reference_dates,
    )
    finite = np.isfinite(target) & np.isfinite(sample_weight) & (sample_weight > 0.0)
    train_matrix = np.nan_to_num(train_matrix[finite], nan=0.0, posinf=0.0, neginf=0.0)
    target = target[finite]
    sample_weight = sample_weight[finite]
    test_matrix = np.nan_to_num(test_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    if len(target) == 0:
        return test_frame.assign(predicted_alpha=0.0, model_top_reasons=[[] for _ in range(len(test_frame.index))], model_reason_summary=None)
    root_weight = np.sqrt(sample_weight)
    weighted_x = train_matrix * root_weight[:, None]
    weighted_y = target * root_weight
    xtx = weighted_x.T @ weighted_x
    identity = np.eye(xtx.shape[0], dtype=float)
    weights = np.linalg.solve(xtx + identity, weighted_x.T @ weighted_y)
    scored = test_frame.copy()
    scored["predicted_alpha"] = test_matrix @ weights
    scored["model_top_reasons"] = [
        service._top_reason_names(dict(zip(feature_names, test_matrix[index] * weights, strict=False)))
        for index in range(len(test_matrix))
    ]
    scored["model_reason_summary"] = scored["model_top_reasons"].apply(service._format_reason_summary)
    return scored


def _score_weighted_lasso(
    service: ShortlistModelService,
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    *,
    target_column: str,
    feature_columns: list[str],
    half_life_dates: int,
    reference_dates: list[pd.Timestamp],
) -> pd.DataFrame:
    try:
        from sklearn.linear_model import Lasso
    except ModuleNotFoundError:
        return _score_weighted_ridge(
            service,
            train_frame,
            test_frame,
            target_column=target_column,
            feature_columns=feature_columns,
            half_life_dates=half_life_dates,
            reference_dates=reference_dates,
        )
    train_matrix, test_matrix, feature_names, _ = service._prepare_model_matrices(
        train_frame,
        test_frame,
        feature_columns_override=feature_columns,
    )
    target = pd.to_numeric(train_frame[target_column], errors="coerce").to_numpy(dtype=float)
    sample_weight = recency_weights_by_snapshot_date(
        train_frame["snapshot_date"],
        half_life_dates=half_life_dates,
        reference_dates=reference_dates,
    )
    finite = np.isfinite(target) & np.isfinite(sample_weight) & (sample_weight > 0.0)
    train_matrix = np.nan_to_num(train_matrix[finite], nan=0.0, posinf=0.0, neginf=0.0)
    target = target[finite]
    sample_weight = sample_weight[finite]
    test_matrix = np.nan_to_num(test_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    if len(target) == 0:
        return test_frame.assign(predicted_alpha=0.0, model_top_reasons=[[] for _ in range(len(test_frame.index))], model_reason_summary=None)
    model = Lasso(alpha=0.001, max_iter=2000, random_state=42, selection="cyclic")
    model.fit(train_matrix, target, sample_weight=sample_weight)
    weights = model.coef_
    nonzero = np.abs(weights) > 1e-8
    used_matrix = test_matrix
    if nonzero.any() and not nonzero.all():
        model = Lasso(alpha=0.001, max_iter=2000, random_state=42, selection="cyclic")
        model.fit(train_matrix[:, nonzero], target, sample_weight=sample_weight)
        full_weights = np.zeros(len(feature_names), dtype=float)
        full_weights[nonzero] = model.coef_
        weights = full_weights
        used_matrix = test_matrix[:, nonzero]
    scored = test_frame.copy()
    scored["predicted_alpha"] = used_matrix @ model.coef_ if nonzero.any() and not nonzero.all() else test_matrix @ weights
    scored["model_top_reasons"] = [
        service._top_reason_names(dict(zip(feature_names, test_matrix[index] * weights, strict=False)))
        for index in range(len(test_matrix))
    ]
    scored["model_reason_summary"] = scored["model_top_reasons"].apply(service._format_reason_summary)
    return scored


def _score_weighted_model(
    service: ShortlistModelService,
    *,
    model_name: str,
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    target_column: str,
    feature_columns: list[str],
    half_life_dates: int,
    reference_dates: list[pd.Timestamp],
) -> pd.DataFrame | None:
    if model_name == "ridge_model":
        return _score_weighted_ridge(
            service,
            train_frame,
            test_frame,
            target_column=target_column,
            feature_columns=feature_columns,
            half_life_dates=half_life_dates,
            reference_dates=reference_dates,
        )
    if model_name == "lasso_model":
        return _score_weighted_lasso(
            service,
            train_frame,
            test_frame,
            target_column=target_column,
            feature_columns=feature_columns,
            half_life_dates=half_life_dates,
            reference_dates=reference_dates,
        )
    return None


def _weighted_walk_forward_predictions(
    service: ShortlistModelService,
    frame: pd.DataFrame,
    *,
    model_name: str,
    feature_columns: list[str],
    config: dict[str, object],
    max_train_dates: int,
    half_life_dates: int,
) -> pd.DataFrame:
    dates = sorted(frame["snapshot_date"].drop_duplicates().tolist())
    folds: list[pd.DataFrame] = []
    start_index = int(config["min_train_dates"])
    stride = max(int(config["evaluation_stride_dates"]), 1)
    horizon = max(int(config["label_horizon_dates"]), 1)
    while start_index < len(dates):
        test_dates = dates[start_index : start_index + int(config["test_window_dates"])]
        train_pool_dates = dates[: max(0, start_index - horizon)]
        if len(train_pool_dates) < int(config["min_train_dates"]):
            start_index += stride
            continue
        train_date_window = train_pool_dates[-max(int(max_train_dates), 1) :]
        train_frame = frame[frame["snapshot_date"].isin(set(train_date_window))].copy()
        train_frame = service._stride_training_labels(
            train_frame,
            dates=dates,
            anchor_index=start_index,
            label_horizon_dates=horizon,
        )
        test_frame = frame[frame["snapshot_date"].isin(set(test_dates))].copy()
        test_frame["regime_matched_training_applied"] = False
        test_frame["regime_matching_test_regime"] = "unknown"
        fold_features = feature_columns
        min_feature_ic = service._load_min_feature_ic()
        if min_feature_ic is not None:
            survivors = service._feature_ic_survivors_from_frame(
                train_frame,
                target_column=TARGET_COLUMN,
                min_feature_ic=float(min_feature_ic),
                min_observation_fraction=service._load_min_feature_ic_observation_fraction(),
            )
            allowed = set(feature_columns)
            fold_features = [feature for feature in survivors if feature in allowed]
            if not fold_features:
                start_index += stride
                continue
        scored = _score_weighted_model(
            service,
            model_name=model_name,
            train_frame=train_frame,
            test_frame=test_frame,
            target_column=TARGET_COLUMN,
            feature_columns=fold_features,
            half_life_dates=half_life_dates,
            reference_dates=dates,
        )
        if scored is not None and not scored.empty:
            folds.append(
                scored[
                    [
                        "snapshot_date",
                        "ticker",
                        "sector",
                        "md_volume_30d",
                        TARGET_COLUMN,
                        "predicted_alpha",
                        "model_top_reasons",
                        "model_reason_summary",
                        "regime_matched_training_applied",
                        "regime_matching_test_regime",
                    ]
                ].copy()
            )
        start_index += stride
    if not folds:
        return pd.DataFrame()
    return pd.concat(folds, axis=0, ignore_index=True)


def _run_unweighted_model(
    service: ShortlistModelService,
    eligible: pd.DataFrame,
    *,
    model_name: str,
    feature_columns: list[str],
    config: dict[str, object],
    max_train_dates: int,
) -> pd.DataFrame:
    xgboost_params = None
    if model_name == "xgboost_model":
        xgboost_params = {**service._xgboost_params_for_config(str(config["xgboost_config"])), "n_jobs": 1}
    predictions = service._walk_forward_predictions(
        eligible,
        target_column=TARGET_COLUMN,
        evaluation_target_column=TARGET_COLUMN,
        model_name=model_name,
        min_train_dates=int(config["min_train_dates"]),
        max_train_dates=max_train_dates,
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
    return predictions if predictions is not None else pd.DataFrame()


def _evaluate_windows(
    service: ShortlistModelService,
    predictions: pd.DataFrame,
    *,
    model_name: str,
    test_window_dates: int,
) -> dict[str, object]:
    if predictions.empty:
        return {
            "full_oos_spearman": float("nan"),
            "last_fold_spearman": float("nan"),
            "trailing_3fold_spearman": float("nan"),
            "honest_rows": 0,
            "honest_dates": 0,
        }
    full = service._evaluate_predictions(
        predictions=predictions,
        top_n=PROMOTION_BASKET_SIZE,
        target_column=TARGET_COLUMN,
        model_name=model_name,
    )
    rolling = service._rolling_window_summaries(
        predictions=predictions,
        target_column=TARGET_COLUMN,
        model_name=model_name,
        top_n=PROMOTION_BASKET_SIZE,
        windows=(),
        fold_windows=(1, 3),
        fold_size=int(test_window_dates),
        include_full_oos=False,
    )
    window_scores = {
        str(row["model"]).removeprefix(f"{model_name}_"): row.get("spearman")
        for row in rolling.to_dict(orient="records")
    }
    return {
        "full_oos_spearman": full.get("spearman"),
        "last_fold_spearman": window_scores.get("last_fold"),
        "trailing_3fold_spearman": window_scores.get("trailing_3folds"),
        "honest_rows": len(predictions.index),
        "honest_dates": int(predictions["snapshot_date"].nunique()),
    }


def _variant_rows(
    service: ShortlistModelService,
    eligible: pd.DataFrame,
    *,
    feature_columns: list[str],
    config: dict[str, object],
    calendar_dates: list[pd.Timestamp],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_name in CANDIDATE_MODELS:
        for max_train_dates in (BASELINE_TRAIN_DATES, SHORT_TRAIN_DATES):
            dense = _run_unweighted_model(
                service,
                eligible,
                model_name=model_name,
                feature_columns=feature_columns,
                config=config,
                max_train_dates=max_train_dates,
            )
            honest = service._non_overlapping_oos_predictions(
                dense,
                horizon_days=int(config["label_horizon_dates"]),
                calendar_dates=calendar_dates,
            )
            rows.append(
                {
                    "model": model_name,
                    "variant": f"max_train_{max_train_dates}",
                    "max_train_dates": max_train_dates,
                    "half_life_dates": None,
                    **_evaluate_windows(
                        service,
                        honest,
                        model_name=f"{model_name}_max_train_{max_train_dates}",
                        test_window_dates=int(config["test_window_dates"]),
                    ),
                }
            )
        if model_name in WEIGHTED_MODELS:
            dense = _weighted_walk_forward_predictions(
                service,
                eligible,
                model_name=model_name,
                feature_columns=feature_columns,
                config=config,
                max_train_dates=BASELINE_TRAIN_DATES,
                half_life_dates=RECENCY_HALF_LIFE_DATES,
            )
            honest = service._non_overlapping_oos_predictions(
                dense,
                horizon_days=int(config["label_horizon_dates"]),
                calendar_dates=calendar_dates,
            )
            rows.append(
                {
                    "model": model_name,
                    "variant": f"weighted_hl_{RECENCY_HALF_LIFE_DATES}",
                    "max_train_dates": BASELINE_TRAIN_DATES,
                    "half_life_dates": RECENCY_HALF_LIFE_DATES,
                    **_evaluate_windows(
                        service,
                        honest,
                        model_name=f"{model_name}_weighted_hl_{RECENCY_HALF_LIFE_DATES}",
                        test_window_dates=int(config["test_window_dates"]),
                    ),
                }
            )
    result = pd.DataFrame(rows)
    baselines = (
        result[result["variant"].eq(f"max_train_{BASELINE_TRAIN_DATES}")]
        .set_index("model")[["full_oos_spearman", "last_fold_spearman", "trailing_3fold_spearman"]]
        .to_dict(orient="index")
    )
    for column in ("full_oos_spearman", "last_fold_spearman", "trailing_3fold_spearman"):
        result[f"{column}_delta_vs_252"] = [
            (
                float(value) - float(baselines.get(str(model), {}).get(column))
                if pd.notna(value) and pd.notna(baselines.get(str(model), {}).get(column))
                else float("nan")
            )
            for model, value in zip(result["model"], result[column], strict=False)
        ]
    return result


def _verdict(rows: pd.DataFrame) -> str:
    if rows.empty:
        return "no verdict: no variant rows produced"
    candidates = rows[~rows["variant"].eq(f"max_train_{BASELINE_TRAIN_DATES}")].copy()
    if candidates.empty:
        return "ties/loses: no adaptive variants produced"
    full_delta = pd.to_numeric(candidates["full_oos_spearman_delta_vs_252"], errors="coerce")
    trailing_delta = pd.to_numeric(candidates["trailing_3fold_spearman_delta_vs_252"], errors="coerce")
    winners = candidates[trailing_delta.ge(0.0200) & full_delta.ge(-1e-12)]
    if winners.empty:
        best = candidates.assign(_trailing_delta=trailing_delta).sort_values("_trailing_delta", ascending=False).head(1)
        if best.empty:
            return "ties/loses: no finite adaptive trailing-3fold lift"
        row = best.iloc[0]
        return (
            "ties/loses: best adaptive trailing-3fold delta "
            f"{_fmt(row.get('trailing_3fold_spearman_delta_vs_252'))} for {row.get('model')} {row.get('variant')}"
        )
    row = winners.assign(_trailing_delta=trailing_delta.loc[winners.index]).sort_values("_trailing_delta", ascending=False).iloc[0]
    return (
        "wins: adaptive training lifts trailing-3fold Spearman by "
        f"{_fmt(row.get('trailing_3fold_spearman_delta_vs_252'))} for {row.get('model')} {row.get('variant')} "
        "without dropping full-OOS below baseline"
    )


def _render_report(
    *,
    output_path: Path,
    rows: pd.DataFrame,
    eligible_rows: int,
    eligible_dates: int,
    config: dict[str, object],
) -> None:
    lines = [
        "# P3 Adaptive Training Window",
        "",
        "- idea: test whether the recent fold collapse is adaptation lag from a one-year training window",
        "- data_access: DuckDB read_only=True; no `./sq` writes; no data/ mutation",
        f"- target_column: {TARGET_COLUMN}",
        f"- model_scope: {config['model_scope']}",
        f"- candidate_models: {', '.join(CANDIDATE_MODELS)}",
        f"- baseline_max_train_dates: {BASELINE_TRAIN_DATES}",
        f"- short_max_train_dates: {SHORT_TRAIN_DATES}",
        f"- recency_weight_half_life_dates: {RECENCY_HALF_LIFE_DATES}",
        f"- test_window_dates: {int(config['test_window_dates'])}",
        f"- evaluation_stride_dates: {int(config['evaluation_stride_dates'])}",
        f"- label_horizon_dates: {int(config['label_horizon_dates'])}",
        f"- eligible_rows: {eligible_rows}",
        f"- eligible_dates: {eligible_dates}",
        f"- verdict: {_verdict(rows)}",
        "",
        "## Variant Table",
        "",
        "| model | variant | rows | dates | full_oos_spearman | delta_full_vs_252 | last_fold_spearman | delta_last_vs_252 | trailing_3fold_spearman | delta_trailing3_vs_252 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    ordered = rows.sort_values(["model", "variant"]).reset_index(drop=True)
    for row in ordered.itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.variant} | {int(row.honest_rows)} | {int(row.honest_dates)} | "
            f"{_fmt(row.full_oos_spearman)} | {_fmt(row.full_oos_spearman_delta_vs_252)} | "
            f"{_fmt(row.last_fold_spearman)} | {_fmt(row.last_fold_spearman_delta_vs_252)} | "
            f"{_fmt(row.trailing_3fold_spearman)} | {_fmt(row.trailing_3fold_spearman_delta_vs_252)} |"
        )
    lines.extend(
        [
            "",
            "## Verification Notes",
            "",
            "- Success rule from the critique: an adaptive variant must lift trailing-3fold Spearman by at least +0.0200 on at least one model without dropping full-OOS below that model's 252-session baseline.",
            "- Production training config was not edited; this is a report-only bakeoff.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(*, output_path: Path) -> dict[str, object]:
    settings = get_settings()
    service = ShortlistModelService(DatabaseManager(paths=settings.paths))
    config = _shortlist_config()
    snapshots = _read_snapshots(duckdb_path=settings.paths.duckdb_path)
    prepared = service._prepare_snapshot_frame(snapshots)
    prepared = service._add_a4_regime_interaction_features(
        prepared,
        horizon_sessions=int(config["label_horizon_dates"]),
    )
    eligible = service._build_matured_eligible_universe(
        prepared,
        target_column=TARGET_COLUMN,
        eligible_universe_mode=str(config["eligible_universe_mode"]),
    )
    base_features = model_feature_columns_for_profile(str(config["feature_profile"]))
    feature_columns = service._filter_model_feature_columns(expand_model_feature_columns(base_features))
    calendar_dates = _read_calendar_dates(duckdb_path=settings.paths.duckdb_path)
    rows = _variant_rows(
        service,
        eligible,
        feature_columns=feature_columns,
        config=config,
        calendar_dates=calendar_dates,
    )
    _render_report(
        output_path=output_path,
        rows=rows,
        eligible_rows=len(eligible.index),
        eligible_dates=int(eligible["snapshot_date"].nunique()) if not eligible.empty else 0,
        config=config,
    )
    return {
        "output_path": str(output_path),
        "eligible_rows": len(eligible.index),
        "eligible_dates": int(eligible["snapshot_date"].nunique()) if not eligible.empty else 0,
        "verdict": _verdict(rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit P3 adaptive shortlist training windows.")
    parser.add_argument("--output", type=Path, default=Path("reports/p3_adaptive_training.md"))
    args = parser.parse_args()
    result = run(output_path=args.output)
    print(
        "wrote {output_path} eligible_dates={eligible_dates} verdict={verdict}".format(**result)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
