from __future__ import annotations

from datetime import UTC, datetime
import argparse
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

from src.research.beat_calibration import chronological_isotonic_probability, finite_float
from src.research.beat_signal import add_forward_beat_label, build_rank_hybrid_predictions
from src.research.orthogonal_ensemble import acceptance_window_summary, build_two_member_rank_blend, passes_acceptance
from src.research.sector_neutral_selection import deoverlap_oos_predictions
from src.research.shortlist_bakeoff_service import MODEL_FEATURE_COLUMNS, expand_model_feature_columns
from src.research.shortlist_model_service import RIDGE_ADAPTIVE_MAX_TRAIN_DATES, ShortlistModelService
from src.settings import get_settings, load_feature_config
from src.utils.db_manager import DatabaseManager


REPORT_PATH = Path("reports/beat_signal.md")
OOS_PATH = Path("reports/shortlist_model_oos_predictions.csv")
TARGET_COLUMN = "alpha_vs_sector_60d"
BEAT_LABEL_COLUMN = "forward_beat_sector_60d"
CALIBRATED_COLUMN = "calibrated_p_beat_sector_oos"
RIDGE_MODEL = "ridge_adaptive"
OVERNIGHT_MODEL = "overnight_session_specialist"
BEAT_LOGISTIC = "beat_logistic_ridge_adaptive"
BEAT_XGBOOST = "beat_xgboost_post_tour"
HORIZON_DAYS = 60
FOLD_SIZE = 20
TRAILING_FOLDS = 3
TOP_N = 2
HYBRID_WEIGHTS = (0.3, 0.5, 0.7)

FLOORS = (
    ("full_oos", "spearman", "full_oos_spearman", ">=", 0.0),
    ("last_fold", "hit_excess", "last_fold_hit_rate_excess", ">=", 0.02),
    ("last_fold", "beat", "last_fold_beat_universe_rate", ">=", 0.50),
    ("last_fold", "mean_excess", "last_fold_mean_target_excess", ">=", 0.0),
    ("last_fold", "spearman", "last_fold_spearman", ">=", 0.0),
    ("last_fold", "top_ticker_rate", "last_fold_top_ticker_date_rate", "<=", 0.40),
    ("trailing_3fold", "hit_excess", "trailing_3fold_hit_rate_excess", ">=", 0.02),
    ("trailing_3fold", "beat", "trailing_3fold_beat_universe_rate", ">=", 0.50),
    ("trailing_3fold", "mean_excess", "trailing_3fold_mean_target_excess", ">=", 0.0),
    ("trailing_3fold", "spearman", "trailing_3fold_spearman", ">=", 0.0),
    ("trailing_3fold", "top_ticker_rate", "trailing_3fold_top_ticker_date_rate", "<=", 0.40),
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the post-tour beat-signal research audit.")
    parser.add_argument("--output", type=Path, default=REPORT_PATH)
    args = parser.parse_args()
    result = run(output_path=args.output)
    print("wrote {output_path} verdict={verdict}".format(**result))
    return 0


def run(*, output_path: Path) -> dict[str, object]:
    settings = get_settings()
    service = ShortlistModelService(DatabaseManager(paths=settings.paths))
    config = _shortlist_config()
    base_feature_columns = [
        feature
        for feature in service._filter_model_feature_columns(MODEL_FEATURE_COLUMNS)
        if not str(feature).startswith("a4_")
    ]
    feature_columns = service._filter_model_feature_columns(expand_model_feature_columns(MODEL_FEATURE_COLUMNS))
    ridge_features = service._feature_columns_for_candidate(
        model_name=RIDGE_MODEL,
        default_feature_columns=feature_columns,
    )
    snapshots = _read_universe_snapshots(
        duckdb_path=settings.paths.duckdb_path,
        columns=_snapshot_columns(base_feature_columns),
    )
    snapshots = service._add_a4_regime_interaction_features(
        service._prepare_snapshot_frame(snapshots),
        horizon_sessions=int(config["label_horizon_dates"]),
    )
    snapshots = add_forward_beat_label(
        snapshots,
        source_column=TARGET_COLUMN,
        target_column=BEAT_LABEL_COLUMN,
        threshold=0.0,
    )
    eligible = service._build_matured_eligible_universe(
        snapshots,
        target_column=TARGET_COLUMN,
        eligible_universe_mode=str(config["eligible_universe_mode"]),
    )
    calendar_dates = _read_calendar_dates(duckdb_path=settings.paths.duckdb_path)

    print("training beat logistic ridge_adaptive", file=sys.stderr, flush=True)
    logistic = _walk_forward_beat_predictions(
        service,
        eligible,
        model_name=BEAT_LOGISTIC,
        feature_screen_model=RIDGE_MODEL,
        feature_columns=ridge_features,
        scorer="logistic",
        config=config,
    )
    print("training beat xgboost post-tour", file=sys.stderr, flush=True)
    xgboost = _walk_forward_beat_predictions(
        service,
        eligible,
        model_name=BEAT_XGBOOST,
        feature_screen_model="xgboost_model",
        feature_columns=feature_columns,
        scorer="xgboost",
        config=config,
    )
    trained = {
        BEAT_LOGISTIC: service._non_overlapping_oos_predictions(
            logistic,
            horizon_days=int(config["label_horizon_dates"]),
            calendar_dates=calendar_dates,
        ),
        BEAT_XGBOOST: service._non_overlapping_oos_predictions(
            xgboost,
            horizon_days=int(config["label_horizon_dates"]),
            calendar_dates=calendar_dates,
        ),
    }

    raw_oos = load_oos_predictions(OOS_PATH)
    evaluation_keys = load_fixed_evaluation_keys(OOS_PATH, calendar_dates=calendar_dates)
    variants = build_variants(raw_oos=raw_oos, beat_predictions=trained)
    rows = summarize_variants(variants, evaluation_keys=evaluation_keys)
    text = render_report(
        rows=rows,
        raw_oos=raw_oos,
        beat_predictions=trained,
        eligible=eligible,
        calendar_dates=calendar_dates,
        evaluation_keys=evaluation_keys,
        config=config,
    )
    output_path.write_text(text, encoding="utf-8")
    return {"output_path": str(output_path), "verdict": verdict_text(rows)}


def _shortlist_config() -> dict[str, object]:
    payload = load_feature_config().get("scan_policy", {}).get("shortlist_model", {})
    return {
        "eligible_universe_mode": str(payload.get("production_eligible_universe_mode", "passed_or_trend")),
        "model_scope": "global",
        "min_train_dates": int(payload.get("min_train_dates", 252)),
        "max_train_dates": RIDGE_ADAPTIVE_MAX_TRAIN_DATES,
        "test_window_dates": int(payload.get("test_window_dates", 20)),
        "evaluation_stride_dates": int(payload.get("oos_evaluation_stride_dates", HORIZON_DAYS)),
        "label_horizon_dates": int(payload.get("horizon_days", HORIZON_DAYS)),
        "xgboost_config": str(payload.get("production_xgboost_config", "balanced_depth4")),
    }


def _snapshot_columns(feature_columns: list[str]) -> list[str]:
    required = [
        "snapshot_date",
        "ticker",
        "sector",
        "md_volume_30d",
        "adj_close",
        "passed_any_strategy",
        "passed_slots_json",
        TARGET_COLUMN,
        "regime_green",
    ]
    return list(dict.fromkeys([*required, *feature_columns]))


def _read_universe_snapshots(*, duckdb_path: Path, columns: list[str]) -> pd.DataFrame:
    import duckdb

    safe_columns = ", ".join(columns)
    with duckdb.connect(str(duckdb_path), read_only=True) as connection:
        return connection.execute(
            f"""
            SELECT {safe_columns}
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
    parsed = pd.to_datetime(pd.Series([row[0] for row in rows]), errors="coerce").dt.normalize().dropna()
    return sorted(parsed.drop_duplicates().tolist())


def _walk_forward_beat_predictions(
    service: ShortlistModelService,
    frame: pd.DataFrame,
    *,
    model_name: str,
    feature_screen_model: str,
    feature_columns: list[str],
    scorer: str,
    config: dict[str, object],
) -> pd.DataFrame:
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
        train_date_window = train_pool_dates[-max(int(config["max_train_dates"]), 1) :]
        train_frame = frame[frame["snapshot_date"].isin(set(train_date_window))].copy()
        train_frame = service._stride_training_labels(
            train_frame,
            dates=dates,
            anchor_index=start_index,
            label_horizon_dates=label_embargo,
        )
        test_frame = frame[frame["snapshot_date"].isin(set(test_dates))].copy()
        family_counts = service._candidate_feature_observation_base_counts(
            model_name=feature_screen_model,
            frame=train_frame,
        )
        ic_survivors = service._feature_ic_survivors_from_frame(
            train_frame,
            target_column=BEAT_LABEL_COLUMN,
            min_feature_ic=service._load_min_feature_ic(),
            min_observation_fraction=service._load_min_feature_ic_observation_fraction(),
            feature_columns_override=None,
            feature_observation_base_counts=family_counts or None,
        )
        allowed = set(feature_columns)
        fold_features = [feature for feature in ic_survivors if feature in allowed]
        if not fold_features:
            start_index += stride
            continue
        if scorer == "logistic":
            scored = _score_logistic_probability(
                service,
                train_frame,
                test_frame,
                feature_columns=fold_features,
            )
        elif scorer == "xgboost":
            scored = _score_xgboost_probability(
                service,
                train_frame,
                test_frame,
                feature_columns=fold_features,
                xgboost_params=xgboost_params,
            )
        else:
            raise ValueError(f"Unsupported scorer {scorer}")
        scored["model_name"] = model_name
        scored["trained_label"] = BEAT_LABEL_COLUMN
        folds.append(
            scored[
                [
                    "snapshot_date",
                    "ticker",
                    "sector",
                    "md_volume_30d",
                    TARGET_COLUMN,
                    BEAT_LABEL_COLUMN,
                    "predicted_alpha",
                    "model_name",
                    "trained_label",
                ]
            ].copy()
        )
        start_index += stride
    if not folds:
        raise SystemExit(f"{model_name} produced no OOS predictions")
    return pd.concat(folds, ignore_index=True)


def _score_logistic_probability(
    service: ShortlistModelService,
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    *,
    feature_columns: list[str],
) -> pd.DataFrame:
    from sklearn.linear_model import LogisticRegression

    train_matrix, test_matrix, _feature_names, _standardized_test = service._prepare_model_matrices(
        train_frame,
        test_frame,
        feature_columns_override=feature_columns,
    )
    target = pd.to_numeric(train_frame[BEAT_LABEL_COLUMN], errors="coerce").to_numpy(dtype=float)
    finite_mask = np.isfinite(target)
    train_matrix = np.nan_to_num(train_matrix[finite_mask], nan=0.0, posinf=0.0, neginf=0.0)
    target = target[finite_mask]
    test_matrix = np.nan_to_num(test_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    scored = test_frame.copy()
    if np.unique(target).size < 2:
        scored["predicted_alpha"] = float(target[0]) if target.size else np.nan
        return scored
    model = LogisticRegression(penalty="l2", solver="liblinear", C=1.0, max_iter=500, random_state=42)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*penalty.*deprecated.*", category=FutureWarning)
        warnings.filterwarnings("ignore", message=".*Inconsistent values: penalty=.*", category=UserWarning)
        model.fit(train_matrix, target)
    scored["predicted_alpha"] = model.predict_proba(test_matrix)[:, 1]
    return scored


def _score_xgboost_probability(
    service: ShortlistModelService,
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    xgboost_params: dict[str, float | int],
) -> pd.DataFrame:
    from xgboost import XGBClassifier

    train_matrix, test_matrix, _feature_names, _standardized_test = service._prepare_model_matrices(
        train_frame,
        test_frame,
        feature_columns_override=feature_columns,
    )
    target = pd.to_numeric(train_frame[BEAT_LABEL_COLUMN], errors="coerce").to_numpy(dtype=float)
    finite_mask = np.isfinite(target)
    train_matrix = np.nan_to_num(train_matrix[finite_mask], nan=0.0, posinf=0.0, neginf=0.0)
    target = target[finite_mask]
    test_matrix = np.nan_to_num(test_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    scored = test_frame.copy()
    if np.unique(target).size < 2:
        scored["predicted_alpha"] = float(target[0]) if target.size else np.nan
        return scored
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
    model.fit(train_matrix, target, verbose=False)
    scored["predicted_alpha"] = model.predict_proba(test_matrix)[:, 1]
    return scored


def load_oos_predictions(path: Path) -> pd.DataFrame:
    columns = {
        "snapshot_date",
        "ticker",
        "sector",
        "model_name",
        "predicted_alpha",
        TARGET_COLUMN,
        "artifact_evaluation_target_column",
    }
    frame = pd.read_csv(path, usecols=lambda column: column in columns)
    missing = {"snapshot_date", "ticker", "model_name", "predicted_alpha", TARGET_COLUMN} - set(frame.columns)
    if missing:
        raise SystemExit(f"OOS artifact is missing required columns: {', '.join(sorted(missing))}")
    if "artifact_evaluation_target_column" in frame.columns:
        declared = set(frame["artifact_evaluation_target_column"].dropna().astype(str).unique().tolist())
        if declared and declared != {TARGET_COLUMN}:
            raise SystemExit("Refusing to audit non-production target artifact: " + ", ".join(sorted(declared)))
    frame = frame[frame["model_name"].astype(str).isin({RIDGE_MODEL, OVERNIGHT_MODEL})].copy()
    frame["snapshot_date"] = pd.to_datetime(frame["snapshot_date"], errors="coerce").dt.normalize()
    frame["ticker"] = frame["ticker"].astype(str).str.strip()
    frame["model_name"] = frame["model_name"].astype(str)
    frame["predicted_alpha"] = pd.to_numeric(frame["predicted_alpha"], errors="coerce")
    frame[TARGET_COLUMN] = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
    return frame.dropna(subset=["snapshot_date", "ticker", "model_name", "predicted_alpha", TARGET_COLUMN]).copy()


def load_fixed_evaluation_keys(path: Path, *, calendar_dates: list[pd.Timestamp] | None = None) -> set[tuple[str, pd.Timestamp]]:
    key_frame = pd.read_csv(path, usecols=["snapshot_date", "ticker"])
    key_frame["snapshot_date"] = pd.to_datetime(key_frame["snapshot_date"], errors="coerce").dt.normalize()
    key_frame["ticker"] = key_frame["ticker"].astype(str).str.strip()
    key_frame = key_frame.dropna(subset=["snapshot_date", "ticker"]).drop_duplicates(["ticker", "snapshot_date"])
    fixed = deoverlap_oos_predictions(
        key_frame,
        horizon_days=HORIZON_DAYS,
        calendar_dates=calendar_dates or key_frame["snapshot_date"].dropna().drop_duplicates().tolist(),
    )
    return {
        (str(row.ticker), pd.Timestamp(row.snapshot_date).normalize())
        for row in fixed[["ticker", "snapshot_date"]].itertuples(index=False)
    }


def build_variants(
    *,
    raw_oos: pd.DataFrame,
    beat_predictions: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    variants: dict[str, pd.DataFrame] = {
        RIDGE_MODEL: raw_oos[raw_oos["model_name"].eq(RIDGE_MODEL)].copy(),
        OVERNIGHT_MODEL: raw_oos[raw_oos["model_name"].eq(OVERNIGHT_MODEL)].copy(),
    }
    ridge = variants[RIDGE_MODEL]
    overnight = variants[OVERNIGHT_MODEL]
    variants["ridge_overnight_rank_blend_0.4_0.6"] = build_two_member_rank_blend(
        raw_oos,
        left_model=RIDGE_MODEL,
        right_model=OVERNIGHT_MODEL,
        left_weight=0.4,
        right_weight=0.6,
        target_column=TARGET_COLUMN,
    )
    for model_name, frame in beat_predictions.items():
        raw = frame.copy()
        raw["model_name"] = model_name
        calibrated = chronological_isotonic_probability(
            raw,
            target_column=TARGET_COLUMN,
            output_column=CALIBRATED_COLUMN,
            min_train_rows=50,
        )
        calibrated_variant = calibrated.copy()
        calibrated_variant["predicted_alpha"] = pd.to_numeric(calibrated_variant[CALIBRATED_COLUMN], errors="coerce")
        calibrated_variant["model_name"] = f"{model_name}_calibrated"
        variants[model_name] = raw
        variants[f"{model_name}_calibrated"] = calibrated_variant
        for beat_weight in HYBRID_WEIGHTS:
            variants[f"{model_name}_calibrated_ridge_rank_hybrid_{beat_weight:.1f}_{1.0 - beat_weight:.1f}"] = (
                build_rank_hybrid_predictions(
                    calibrated,
                    ridge,
                    beat_weight=beat_weight,
                    beat_score_column=CALIBRATED_COLUMN,
                    target_column=TARGET_COLUMN,
                    model_name=f"{model_name}_calibrated_ridge_rank_hybrid_{beat_weight:.1f}_{1.0 - beat_weight:.1f}",
                )
            )
        variants[f"ridge_{model_name}_rank_blend_0.4_0.6"] = build_rank_hybrid_predictions(
            calibrated,
            ridge,
            beat_weight=0.6,
            alpha_weight=0.4,
            beat_score_column=CALIBRATED_COLUMN,
            target_column=TARGET_COLUMN,
            model_name=f"ridge_{model_name}_rank_blend_0.4_0.6",
        )
        variants[f"{model_name}_overnight_rank_blend_0.4_0.6"] = build_rank_hybrid_predictions(
            calibrated,
            overnight,
            beat_weight=0.4,
            alpha_weight=0.6,
            beat_score_column=CALIBRATED_COLUMN,
            target_column=TARGET_COLUMN,
            model_name=f"{model_name}_overnight_rank_blend_0.4_0.6",
        )
    return variants


def summarize_variants(
    variants: dict[str, pd.DataFrame],
    *,
    evaluation_keys: set[tuple[str, pd.Timestamp]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    honest_frames = {
        variant: filter_to_evaluation_keys(frame, evaluation_keys=evaluation_keys)
        for variant, frame in variants.items()
    }
    ridge_honest = honest_frames.get(RIDGE_MODEL, pd.DataFrame())
    for variant, frame in variants.items():
        honest = honest_frames[variant]
        summary = acceptance_window_summary(
            honest,
            target_column=TARGET_COLUMN,
            top_n=TOP_N,
            fold_size=FOLD_SIZE,
            trailing_folds=TRAILING_FOLDS,
        )
        rows.append(
            {
                "variant": variant,
                "rows": len(honest.index),
                "dates": int(honest["snapshot_date"].nunique()) if not honest.empty else 0,
                "passes_acceptance": passes_acceptance(summary),
                "failed_floors": ", ".join(failed_floors(summary)) or "none",
                "max_floor_gap": max_floor_gap(summary),
                "full_oos_spearman_delta_vs_ridge_common": _common_full_oos_spearman_delta(
                    honest,
                    ridge_honest,
                ),
                **summary,
            }
        )
    return pd.DataFrame(rows)


def filter_to_evaluation_keys(predictions: pd.DataFrame, *, evaluation_keys: set[tuple[str, pd.Timestamp]]) -> pd.DataFrame:
    if predictions.empty or not evaluation_keys:
        return predictions.iloc[0:0].copy()
    working = predictions.copy()
    normalized_dates = pd.to_datetime(working["snapshot_date"], errors="coerce").dt.normalize()
    row_keys = list(zip(working["ticker"].astype(str), normalized_dates))
    return working.loc[[key in evaluation_keys for key in row_keys]].copy().reset_index(drop=True)


def _common_full_oos_spearman_delta(left: pd.DataFrame, ridge: pd.DataFrame) -> float:
    if left.empty or ridge.empty:
        return float("nan")
    left_working = left.copy()
    ridge_working = ridge.copy()
    left_working["snapshot_date"] = pd.to_datetime(left_working["snapshot_date"], errors="coerce").dt.normalize()
    ridge_working["snapshot_date"] = pd.to_datetime(ridge_working["snapshot_date"], errors="coerce").dt.normalize()
    left_working["_key"] = list(zip(left_working["ticker"].astype(str), left_working["snapshot_date"]))
    ridge_working["_key"] = list(zip(ridge_working["ticker"].astype(str), ridge_working["snapshot_date"]))
    common_keys = set(left_working["_key"].tolist()) & set(ridge_working["_key"].tolist())
    if not common_keys:
        return float("nan")
    left_common = left_working[left_working["_key"].isin(common_keys)].copy()
    ridge_common = ridge_working[ridge_working["_key"].isin(common_keys)].copy()
    left_summary = acceptance_window_summary(left_common, target_column=TARGET_COLUMN, top_n=TOP_N)
    ridge_summary = acceptance_window_summary(ridge_common, target_column=TARGET_COLUMN, top_n=TOP_N)
    return finite_float(left_summary.get("full_oos_spearman")) - finite_float(ridge_summary.get("full_oos_spearman"))


def failed_floors(summary: dict[str, object]) -> list[str]:
    failures: list[str] = []
    for window, metric, column, operator, threshold in FLOORS:
        value = finite_float(summary.get(column))
        passed = math.isfinite(value) and (
            (operator == ">=" and value >= threshold) or (operator == "<=" and value <= threshold)
        )
        if not passed:
            failures.append(f"{window} {metric} {_fmt(value)} {operator} {_fmt(threshold)}")
    return failures


def max_floor_gap(summary: dict[str, object]) -> float:
    gaps: list[float] = []
    for _, _, column, operator, threshold in FLOORS:
        value = finite_float(summary.get(column))
        if not math.isfinite(value):
            gaps.append(float("inf"))
        elif operator == ">=":
            gaps.append(max(0.0, threshold - value))
        else:
            gaps.append(max(0.0, value - threshold))
    return max(gaps) if gaps else float("inf")


def render_report(
    *,
    rows: pd.DataFrame,
    raw_oos: pd.DataFrame,
    beat_predictions: dict[str, pd.DataFrame],
    eligible: pd.DataFrame,
    calendar_dates: list[pd.Timestamp],
    evaluation_keys: set[tuple[str, pd.Timestamp]],
    config: dict[str, object],
) -> str:
    ridge = _row_for(rows, RIDGE_MODEL)
    best = _best_row(rows)
    lines = [
        "# Beat Signal",
        "",
        f"- generated_at: {datetime.now(UTC).replace(microsecond=0).isoformat()}",
        "- idea: BEAT-SIGNAL beat-classification on the post-tour feature set",
        "- data_access: read-only DuckDB and OOS artifact; no `./sq` write command; no `data/` mutation",
        f"- source_artifact: {OOS_PATH}",
        f"- target_column: {TARGET_COLUMN}",
        f"- beat_label: {BEAT_LABEL_COLUMN} = alpha_vs_sector_60d > 0",
        f"- feature_set: post-tour MODEL_FEATURE_COLUMNS with B1 overnight features and A4 regime interactions",
        f"- model_scope: {config['model_scope']}",
        f"- min_train_dates: {config['min_train_dates']}",
        f"- max_train_dates: {config['max_train_dates']} (ridge_adaptive exact adaptive window)",
        f"- test_window_dates: {config['test_window_dates']}",
        f"- evaluation_stride_dates: {config['evaluation_stride_dates']}",
        f"- label_horizon_dates: {config['label_horizon_dates']}",
        f"- eligible_universe_mode: {config['eligible_universe_mode']}",
        f"- eligible_rows: {len(eligible.index)}",
        f"- eligible_dates: {int(eligible['snapshot_date'].nunique()) if not eligible.empty else 0}",
        f"- raw_oos_rows_loaded: {len(raw_oos.index)}",
        f"- raw_oos_dates_loaded: {int(raw_oos['snapshot_date'].nunique()) if not raw_oos.empty else 0}",
        f"- trained_beat_rows: {sum(len(frame.index) for frame in beat_predictions.values())}",
        f"- calendar_dates_loaded: {len(calendar_dates)}",
        f"- fixed_evaluation_keys: {len(evaluation_keys)}",
        "- calibration: chronological isotonic per variant; each date uses only earlier OOS dates",
        "- acceptance: full-OOS Spearman >= 0; last-fold and trailing-3fold hit_ex >= 0.02, beat >= 0.50, mean_excess >= 0, Spearman >= 0, top_ticker_date_rate <= 0.40",
        f"- verdict: {verdict_text(rows)}",
        f"- ridge_baseline_full_oos_spearman: {_fmt(ridge.get('full_oos_spearman') if ridge else float('nan'))}",
        f"- best_variant: {best.get('variant', 'n/a') if best else 'n/a'}",
        f"- best_common_grid_delta_vs_ridge_full_oos_spearman: {_fmt(best.get('full_oos_spearman_delta_vs_ridge_common') if best else float('nan'))}",
        "",
        "## Floor Table",
        "",
        "| variant | pass | rows | dates | full_sp | common_delta_vs_ridge | last_hit_ex | last_beat | last_mean_ex | last_sp | last_top | last_top_rate | trailing_hit_ex | trailing_beat | trailing_mean_ex | trailing_sp | trailing_top | trailing_top_rate | failed_floors |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---|---:|---|",
    ]
    for row in ordered_rows(rows).itertuples(index=False):
        lines.append(
            f"| {row.variant} | {'yes' if bool(row.passes_acceptance) else 'no'} | "
            f"{int(row.rows)} | {int(row.dates)} | {_fmt(row.full_oos_spearman)} | "
            f"{_fmt(row.full_oos_spearman_delta_vs_ridge_common)} | "
            f"{_fmt(row.last_fold_hit_rate_excess)} | {_fmt(row.last_fold_beat_universe_rate)} | "
            f"{_fmt(row.last_fold_mean_target_excess)} | {_fmt(row.last_fold_spearman)} | "
            f"{_text(row.last_fold_top_ticker)} | {_fmt(row.last_fold_top_ticker_date_rate)} | "
            f"{_fmt(row.trailing_3fold_hit_rate_excess)} | {_fmt(row.trailing_3fold_beat_universe_rate)} | "
            f"{_fmt(row.trailing_3fold_mean_target_excess)} | {_fmt(row.trailing_3fold_spearman)} | "
            f"{_text(row.trailing_3fold_top_ticker)} | {_fmt(row.trailing_3fold_top_ticker_date_rate)} | "
            f"{row.failed_floors} |"
        )
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            verdict_text(rows),
            "",
            "Report only. The production candidate roster, promotion gate, selection gate, top-2 cap, confidence basket, rotation exclusion, and scan behavior are unchanged.",
            "",
        ]
    )
    return "\n".join(lines)


def verdict_text(rows: pd.DataFrame) -> str:
    if rows.empty:
        return "NO VERDICT: no rows were evaluated"
    passers = rows[rows["passes_acceptance"].astype(bool)].copy()
    if not passers.empty:
        best = ordered_rows(passers).iloc[0]
        return (
            f"PASS: {best.variant} clears all floors with full-OOS Spearman {_fmt(best.full_oos_spearman)}, "
            f"last beat {_fmt(best.last_fold_beat_universe_rate)}, trailing beat {_fmt(best.trailing_3fold_beat_universe_rate)}"
        )
    best = ordered_rows(rows).iloc[0]
    return (
        f"NO PASS: closest row is {best.variant}; full-OOS Spearman {_fmt(best.full_oos_spearman)}, "
        f"last beat {_fmt(best.last_fold_beat_universe_rate)}, trailing beat {_fmt(best.trailing_3fold_beat_universe_rate)}; "
        f"failed floors: {best.failed_floors}"
    )


def ordered_rows(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows
    ordered = rows.copy()
    for column in (
        "max_floor_gap",
        "full_oos_spearman",
        "last_fold_beat_universe_rate",
        "trailing_3fold_beat_universe_rate",
        "last_fold_hit_rate_excess",
        "trailing_3fold_hit_rate_excess",
    ):
        ordered[column] = pd.to_numeric(ordered[column], errors="coerce")
    return ordered.sort_values(
        [
            "passes_acceptance",
            "max_floor_gap",
            "last_fold_beat_universe_rate",
            "trailing_3fold_beat_universe_rate",
            "last_fold_hit_rate_excess",
            "trailing_3fold_hit_rate_excess",
            "full_oos_spearman",
        ],
        ascending=[False, True, False, False, False, False, False],
    )


def _best_row(rows: pd.DataFrame) -> dict[str, object]:
    return ordered_rows(rows).iloc[0].to_dict() if not rows.empty else {}


def _row_for(rows: pd.DataFrame, variant: str) -> dict[str, object]:
    if rows.empty:
        return {}
    scoped = rows[rows["variant"].astype(str).eq(str(variant))]
    return scoped.iloc[0].to_dict() if not scoped.empty else {}


def _fmt(value: object, *, places: int = 4) -> str:
    number = finite_float(value)
    if not math.isfinite(number):
        return "n/a"
    return f"{number:+.{places}f}"


def _text(value: object) -> str:
    if value is None:
        return "n/a"
    text = str(value)
    return text if text and text.lower() != "nan" else "n/a"


if __name__ == "__main__":
    raise SystemExit(main())
