from __future__ import annotations

import argparse
from contextlib import contextmanager
import math
from pathlib import Path
import re
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
VENDOR_DIR = ROOT_DIR / ".vendor"
if VENDOR_DIR.exists():
    sys.path.insert(0, str(VENDOR_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd

import src.research.shortlist_bakeoff_service as bakeoff_features
import src.research.shortlist_model_service as shortlist_models
from src.research.shortlist_bakeoff_service import (
    MODEL_FEATURE_COLUMNS,
    build_rank_augmented_feature_frame,
    expand_model_feature_columns,
)
from src.research.shortlist_model_service import ShortlistModelService
from src.settings import get_settings, load_feature_config
from src.utils.db_manager import DatabaseManager


TARGET_COLUMN = "alpha_vs_sector_60d"
TOP_FEATURE_COUNT = 15
REGIME_MAIN_EFFECTS = ("a4_regime_trending", "a4_regime_reversal")
CANDIDATE_MODELS = ("ridge_model", "ic_sign_model")
STANDARD_MODEL_FEATURE_COLUMNS = tuple(feature for feature in MODEL_FEATURE_COLUMNS if not str(feature).startswith("a4_"))


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
    return {
        "eligible_universe_mode": str(payload.get("production_eligible_universe_mode", "passed_or_trend")),
        "label_horizon_dates": int(payload.get("horizon_days", 60)),
        "min_train_dates": int(payload.get("min_train_dates", 252)),
        "max_train_dates": int(payload.get("max_train_dates", payload.get("min_train_dates", 252))),
        "test_window_dates": int(payload.get("test_window_dates", 20)),
        "evaluation_stride_dates": int(payload.get("oos_evaluation_stride_dates", payload.get("test_window_dates", 20))),
        "model_scope": "global",
        "xgboost_config": str(payload.get("production_xgboost_config", "balanced_depth4")),
    }


def _read_table_columns(*, duckdb_path: Path, table: str) -> set[str]:
    import duckdb

    with duckdb.connect(str(duckdb_path), read_only=True) as connection:
        rows = connection.execute(f"PRAGMA table_info('{table}')").fetchall()
    return {str(row[1]) for row in rows}


def _read_snapshots(*, duckdb_path: Path, columns: list[str]) -> pd.DataFrame:
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
    return sorted(pd.to_datetime(pd.Series([row[0] for row in rows]), errors="coerce").dropna().dt.normalize().tolist())


def _read_regime_meter(*, duckdb_path: Path) -> pd.DataFrame:
    import duckdb

    with duckdb.connect(str(duckdb_path), read_only=True) as connection:
        tables = {str(row[0]) for row in connection.execute("SHOW TABLES").fetchall()}
        if "regime_meter" not in tables:
            return pd.DataFrame(columns=["snapshot_date", "classification"])
        return connection.execute(
            "SELECT snapshot_date, classification FROM regime_meter ORDER BY snapshot_date"
        ).fetchdf()


def _snapshot_columns(available: set[str]) -> list[str]:
    required = [
        "snapshot_date",
        "ticker",
        "sector",
        "md_volume_30d",
        "adj_close",
        "passed_any_strategy",
        "passed_slots_json",
        "regime_green",
        TARGET_COLUMN,
    ]
    optional = [column for column in MODEL_FEATURE_COLUMNS if column in available]
    return list(dict.fromkeys([column for column in [*required, *optional] if column in available]))


def _sanitize_feature_name(feature: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(feature)).strip("_")


def _parse_top_feature_ic_features(report_path: Path, *, limit: int = TOP_FEATURE_COUNT) -> list[str]:
    if not report_path.exists():
        return []
    features: list[str] = []
    for line in report_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| ") or " |" not in line or line.startswith("| feature "):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 6 or cells[5].lower() != "true":
            continue
        feature = cells[0]
        if feature and feature not in features:
            features.append(feature)
        if len(features) >= int(limit):
            break
    return features


def _regime_classifications_by_prediction_date(
    *,
    prediction_dates: list[pd.Timestamp],
    universe_dates: list[pd.Timestamp],
    regime_meter: pd.DataFrame,
    horizon_sessions: int,
) -> dict[pd.Timestamp, str]:
    if regime_meter.empty or not {"snapshot_date", "classification"}.issubset(regime_meter.columns):
        return {}
    regime = regime_meter.copy()
    regime["snapshot_date"] = pd.to_datetime(regime["snapshot_date"], errors="coerce").dt.normalize()
    regime = regime.dropna(subset=["snapshot_date"]).sort_values("snapshot_date").reset_index(drop=True)
    if regime.empty or not universe_dates:
        return {}
    from bisect import bisect_right

    normalized_universe = sorted(pd.to_datetime(pd.Series(universe_dates), errors="coerce").dropna().dt.normalize().tolist())
    regime_dates = regime["snapshot_date"].tolist()
    regime_classes = regime["classification"].astype(str).str.lower().tolist()
    horizon = max(int(horizon_sessions or 0), 0)
    output: dict[pd.Timestamp, str] = {}
    for raw_date in prediction_dates:
        prediction_date = pd.Timestamp(raw_date).normalize()
        source_index = bisect_right(normalized_universe, prediction_date) - horizon - 1
        if source_index < 0:
            continue
        cutoff = normalized_universe[source_index]
        regime_index = bisect_right(regime_dates, cutoff) - 1
        if regime_index >= 0:
            output[prediction_date] = regime_classes[regime_index]
    return output


def build_regime_interaction_frame(
    frame: pd.DataFrame,
    *,
    top_features: list[str],
    regime_by_date: dict[pd.Timestamp, str],
) -> tuple[pd.DataFrame, list[str]]:
    working = frame.copy()
    working["snapshot_date"] = pd.to_datetime(working["snapshot_date"], errors="coerce").dt.normalize()
    ranked, _ = build_rank_augmented_feature_frame(working)
    ranked = ranked.drop(columns=[column for column in ranked.columns if str(column).startswith("a4_")], errors="ignore")
    regime = working["snapshot_date"].map(lambda value: regime_by_date.get(pd.Timestamp(value).normalize(), "unknown"))
    interaction_payload: dict[str, pd.Series] = {
        "a4_regime_trending": regime.eq("trending").astype(float),
        "a4_regime_reversal": regime.eq("reversal").astype(float),
    }
    interaction_columns: list[str] = list(REGIME_MAIN_EFFECTS)
    for feature in top_features:
        if feature not in ranked.columns:
            continue
        values = pd.to_numeric(ranked[feature], errors="coerce")
        for regime_name in ("trending", "reversal"):
            column = f"a4_{_sanitize_feature_name(feature)}_x_regime_{regime_name}"
            interaction_payload[column] = values * regime.eq(regime_name).astype(float)
            interaction_columns.append(column)
    return pd.concat([ranked, pd.DataFrame(interaction_payload, index=ranked.index)], axis=1), interaction_columns


@contextmanager
def _temporary_model_features(extra_features: list[str]):
    original = list(MODEL_FEATURE_COLUMNS)
    try:
        for feature in extra_features:
            if feature not in MODEL_FEATURE_COLUMNS:
                MODEL_FEATURE_COLUMNS.append(feature)
        shortlist_models.MODEL_FEATURE_COLUMNS = MODEL_FEATURE_COLUMNS
        bakeoff_features.MODEL_FEATURE_COLUMNS = MODEL_FEATURE_COLUMNS
        yield
    finally:
        MODEL_FEATURE_COLUMNS[:] = original
        shortlist_models.MODEL_FEATURE_COLUMNS = MODEL_FEATURE_COLUMNS
        bakeoff_features.MODEL_FEATURE_COLUMNS = MODEL_FEATURE_COLUMNS


def _fold_ic_rows(
    service: ShortlistModelService,
    frame: pd.DataFrame,
    *,
    interaction_columns: list[str],
    config: dict[str, object],
    min_feature_ic: float,
    observation_fraction: float,
) -> pd.DataFrame:
    dates = sorted(pd.to_datetime(frame["snapshot_date"], errors="coerce").dropna().dt.normalize().drop_duplicates().tolist())
    start_index = int(config["min_train_dates"])
    stride = max(int(config["evaluation_stride_dates"]), 1)
    horizon = int(config["label_horizon_dates"])
    rows: list[dict[str, object]] = []
    while start_index < len(dates):
        test_dates = dates[start_index : start_index + int(config["test_window_dates"])]
        train_pool_dates = dates[: max(0, start_index - horizon)]
        if len(train_pool_dates) < int(config["min_train_dates"]):
            start_index += stride
            continue
        train_frame = frame[frame["snapshot_date"].isin(set(train_pool_dates))].copy()
        train_frame = service._stride_training_labels(
            train_frame,
            dates=dates,
            anchor_index=start_index,
            label_horizon_dates=horizon,
        )
        target = pd.to_numeric(train_frame[TARGET_COLUMN], errors="coerce")
        min_observations = service._feature_ic_min_observations(
            len(train_frame.index),
            min_observation_fraction=observation_fraction,
        )
        for feature_name in interaction_columns:
            values = pd.to_numeric(train_frame.get(feature_name), errors="coerce")
            valid = values.notna() & target.notna()
            observations = int(valid.sum())
            ic = float("nan")
            cleared = False
            if observations >= min_observations and values[valid].nunique(dropna=True) >= 2 and target[valid].nunique(dropna=True) >= 2:
                corr = values[valid].corr(target[valid], method="spearman")
                if pd.notna(corr) and math.isfinite(float(corr)):
                    ic = float(corr)
                    cleared = abs(ic) >= float(min_feature_ic)
            rows.append(
                {
                    "fold_start": pd.Timestamp(test_dates[0]).date().isoformat() if test_dates else "n/a",
                    "feature": feature_name,
                    "train_rows": len(train_frame.index),
                    "observations": observations,
                    "min_observations": min_observations,
                    "ic": ic,
                    "cleared": bool(cleared),
                }
            )
        start_index += stride
    return pd.DataFrame(rows)


def _ic_summary(fold_rows: pd.DataFrame) -> pd.DataFrame:
    if fold_rows.empty:
        return pd.DataFrame(columns=["feature", "folds", "mean_ic", "mean_abs_ic", "max_abs_ic", "clear_folds", "clear_rate"])
    working = fold_rows.copy()
    working["abs_ic"] = pd.to_numeric(working["ic"], errors="coerce").abs()
    summary = (
        working.groupby("feature", sort=False)
        .agg(
            folds=("fold_start", "nunique"),
            mean_ic=("ic", "mean"),
            mean_abs_ic=("abs_ic", "mean"),
            max_abs_ic=("abs_ic", "max"),
            clear_folds=("cleared", "sum"),
            clear_rate=("cleared", "mean"),
        )
        .reset_index()
    )
    return summary.sort_values(["mean_abs_ic", "max_abs_ic", "feature"], ascending=[False, False, True]).reset_index(drop=True)


def _pooled_spearman(frame: pd.DataFrame) -> float:
    score = pd.to_numeric(frame.get("predicted_alpha"), errors="coerce")
    target = pd.to_numeric(frame.get(TARGET_COLUMN), errors="coerce")
    valid = score.notna() & target.notna()
    if int(valid.sum()) < 3 or score[valid].nunique(dropna=True) < 2 or target[valid].nunique(dropna=True) < 2:
        return float("nan")
    corr = score[valid].corr(target[valid], method="spearman")
    return float(corr) if pd.notna(corr) and math.isfinite(float(corr)) else float("nan")


def _run_model(
    service: ShortlistModelService,
    eligible: pd.DataFrame,
    *,
    model_name: str,
    feature_columns: list[str],
    config: dict[str, object],
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
        max_train_dates=None if config["max_train_dates"] in (None, "") else int(config["max_train_dates"]),
        test_window_dates=int(config["test_window_dates"]),
        evaluation_stride_dates=int(config["evaluation_stride_dates"]),
        label_horizon_dates=int(config["label_horizon_dates"]),
        model_scope=str(config["model_scope"]),
        xgboost_params=xgboost_params,
        feature_columns_override=feature_columns,
        min_feature_ic=None,
        min_feature_ic_observation_fraction=service._load_min_feature_ic_observation_fraction(),
        regime_matching_mode="off",
        min_regime_train_dates=service._load_min_regime_train_dates(),
        regime_transition_purge_mode="off",
    )
    return predictions if predictions is not None else pd.DataFrame()


def _bakeoff_rows(
    service: ShortlistModelService,
    eligible: pd.DataFrame,
    *,
    baseline_features: list[str],
    interaction_columns: list[str],
    config: dict[str, object],
    calendar_dates: list[pd.Timestamp],
) -> pd.DataFrame:
    with_interactions = service._filter_model_feature_columns([*baseline_features, *interaction_columns])
    rows: list[dict[str, object]] = []
    for model_name in CANDIDATE_MODELS:
        dense_without = _run_model(service, eligible, model_name=model_name, feature_columns=baseline_features, config=config)
        dense_with = _run_model(service, eligible, model_name=model_name, feature_columns=with_interactions, config=config)
        honest_without = service._non_overlapping_oos_predictions(
            dense_without,
            horizon_days=int(config["label_horizon_dates"]),
            calendar_dates=calendar_dates,
        )
        honest_with = service._non_overlapping_oos_predictions(
            dense_with,
            horizon_days=int(config["label_horizon_dates"]),
            calendar_dates=calendar_dates,
        )
        without_s = _pooled_spearman(honest_without)
        with_s = _pooled_spearman(honest_with)
        rows.append(
            {
                "model": model_name,
                "without_a4_honest_rows": len(honest_without.index),
                "with_a4_honest_rows": len(honest_with.index),
                "without_a4_full_oos_spearman": without_s,
                "with_a4_full_oos_spearman": with_s,
                "delta_with_minus_without": with_s - without_s if math.isfinite(with_s) and math.isfinite(without_s) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _render_report(
    *,
    output_path: Path,
    top_features: list[str],
    interaction_columns: list[str],
    summary: pd.DataFrame,
    fold_rows: pd.DataFrame,
    bakeoff: pd.DataFrame | None,
    eligible_rows: int,
    eligible_dates: int,
    regime_rows: int,
    min_feature_ic: float,
    observation_fraction: float,
    model_scope: str,
) -> None:
    surviving = summary[summary["mean_abs_ic"].ge(float(min_feature_ic))]["feature"].astype(str).tolist() if not summary.empty else []
    best_delta = float("nan")
    if bakeoff is not None and not bakeoff.empty:
        finite = pd.to_numeric(bakeoff["delta_with_minus_without"], errors="coerce").dropna()
        if not finite.empty:
            best_delta = float(finite.max())
    verdict = "wire production builder" if math.isfinite(best_delta) and best_delta >= 0.005 else "do not wire; hypothesis flat"
    lines = [
        "# A4 Regime-Interacted Feature Audit",
        "",
        "- idea: multiply top surviving shortlist features by lagged regime-meter onehots so models can learn regime-conditional sign directly",
        "- data_access: DuckDB read_only=True; no `./sq` writes; no data/ mutation",
        f"- target_column: {TARGET_COLUMN}",
        "- regime_sources_available: `universe_daily_snapshots.regime_green`; lagged `regime_meter.classification` mapped exactly like train_and_flip",
        "- interaction_regime_source_used: lagged `regime_meter.classification` onehots (`trending`, `reversal`)",
        f"- research_model_scope: {model_scope}",
        "- bakeoff_feature_set: current top feature-IC survivors with A4 fold-local survivors added; no second internal IC screen",
        f"- top_feature_count: {len(top_features)}",
        f"- top_features_from_current_feature_ic_report: {', '.join(top_features) if top_features else 'none'}",
        f"- interaction_columns: {len(interaction_columns)}",
        f"- eligible_rows: {eligible_rows}",
        f"- eligible_dates: {eligible_dates}",
        f"- regime_meter_rows: {regime_rows}",
        f"- min_feature_ic: {float(min_feature_ic):.4f}",
        f"- min_feature_ic_observation_fraction: {float(observation_fraction):.4f}",
        f"- screen_verdict: {'survived' if surviving else 'failed'}",
        f"- best_honest_grid_delta: {_fmt(best_delta)}",
        f"- verdict: {verdict}",
        "",
        "## Fold-Local IC Summary",
        "",
        "| feature | folds | mean_ic | mean_abs_ic | max_abs_ic | clear_folds | clear_rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.feature} | {int(row.folds)} | {_fmt(row.mean_ic)} | {_fmt(row.mean_abs_ic)} | "
            f"{_fmt(row.max_abs_ic)} | {int(row.clear_folds)} | {_fmt(row.clear_rate)} |"
        )
    lines.extend(["", "## Fold Details", "", "| fold_start | feature | observations | min_observations | ic | cleared |", "|---|---|---:|---:|---:|---|"])
    for row in fold_rows.itertuples(index=False):
        lines.append(
            f"| {row.fold_start} | {row.feature} | {int(row.observations)} | {int(row.min_observations)} | "
            f"{_fmt(row.ic)} | {bool(row.cleared)} |"
        )
    lines.append("")
    lines.append("## Honest-Grid With/Without Bakeoff")
    lines.append("")
    if bakeoff is None:
        lines.append("Skipped: no A4 interaction reached mean_abs_ic >= min_feature_ic, so A4 is falsified at the IC screen.")
    elif bakeoff.empty:
        lines.append("No bakeoff rows produced.")
    else:
        lines.extend(
            [
                "| model | without_a4_rows | with_a4_rows | without_a4_full_oos_spearman | with_a4_full_oos_spearman | delta_with_minus_without |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in bakeoff.itertuples(index=False):
            lines.append(
                f"| {row.model} | {int(row.without_a4_honest_rows)} | {int(row.with_a4_honest_rows)} | "
                f"{_fmt(row.without_a4_full_oos_spearman)} | {_fmt(row.with_a4_full_oos_spearman)} | {_fmt(row.delta_with_minus_without)} |"
            )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(*, output_path: Path) -> dict[str, object]:
    settings = get_settings()
    service = ShortlistModelService(DatabaseManager(paths=settings.paths))
    config = _shortlist_config()
    available = _read_table_columns(duckdb_path=settings.paths.duckdb_path, table="universe_daily_snapshots")
    snapshots = _read_snapshots(duckdb_path=settings.paths.duckdb_path, columns=_snapshot_columns(available))
    prepared = service._prepare_snapshot_frame(snapshots)
    eligible_base = service._build_matured_eligible_universe(
        prepared,
        target_column=TARGET_COLUMN,
        eligible_universe_mode=str(config["eligible_universe_mode"]),
    )
    calendar_dates = _read_calendar_dates(duckdb_path=settings.paths.duckdb_path)
    regime_meter = _read_regime_meter(duckdb_path=settings.paths.duckdb_path)
    regime_by_date = _regime_classifications_by_prediction_date(
        prediction_dates=sorted(eligible_base["snapshot_date"].drop_duplicates().tolist()),
        universe_dates=calendar_dates,
        regime_meter=regime_meter,
        horizon_sessions=int(config["label_horizon_dates"]),
    )
    top_features = _parse_top_feature_ic_features(settings.paths.reports_dir / "feature_ic_report.md", limit=TOP_FEATURE_COUNT)
    interaction_frame, interaction_columns = build_regime_interaction_frame(
        eligible_base,
        top_features=top_features,
        regime_by_date=regime_by_date,
    )
    min_feature_ic = service._load_min_feature_ic()
    observation_fraction = service._load_min_feature_ic_observation_fraction()
    with _temporary_model_features(interaction_columns):
        fold_rows = _fold_ic_rows(
            service,
            interaction_frame,
            interaction_columns=interaction_columns,
            config=config,
            min_feature_ic=min_feature_ic,
            observation_fraction=observation_fraction,
        )
        summary = _ic_summary(fold_rows)
        bakeoff = None
        surviving_interactions = (
            summary[summary["mean_abs_ic"].ge(float(min_feature_ic))]["feature"].astype(str).tolist()
            if not summary.empty
            else []
        )
        if surviving_interactions:
            bakeoff = _bakeoff_rows(
                service,
                interaction_frame,
                baseline_features=top_features,
                interaction_columns=surviving_interactions,
                config=config,
                calendar_dates=calendar_dates,
            )
    _render_report(
        output_path=output_path,
        top_features=top_features,
        interaction_columns=interaction_columns,
        summary=summary,
        fold_rows=fold_rows,
        bakeoff=bakeoff,
        eligible_rows=len(interaction_frame.index),
        eligible_dates=int(interaction_frame["snapshot_date"].nunique()) if not interaction_frame.empty else 0,
        regime_rows=len(regime_meter.index),
        min_feature_ic=min_feature_ic,
        observation_fraction=observation_fraction,
        model_scope=str(config["model_scope"]),
    )
    best_delta = float("nan")
    if bakeoff is not None and not bakeoff.empty:
        finite = pd.to_numeric(bakeoff["delta_with_minus_without"], errors="coerce").dropna()
        if not finite.empty:
            best_delta = float(finite.max())
    return {
        "output_path": str(output_path),
        "eligible_rows": len(interaction_frame.index),
        "eligible_dates": int(interaction_frame["snapshot_date"].nunique()) if not interaction_frame.empty else 0,
        "survivors": int(summary["mean_abs_ic"].ge(float(min_feature_ic)).sum()) if not summary.empty else 0,
        "best_mean_abs_ic": float(summary["mean_abs_ic"].max()) if not summary.empty else float("nan"),
        "best_delta": best_delta,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit A4 regime-interacted shortlist features.")
    parser.add_argument("--output", type=Path, default=Path("reports/a4_regime_interactions.md"))
    args = parser.parse_args()
    result = run(output_path=args.output)
    print(
        "wrote {output_path} eligible_dates={eligible_dates} survivors={survivors} "
        "best_mean_abs_ic={best_mean_abs_ic:+.4f} best_delta={best_delta:+.4f}".format(**result)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
