from __future__ import annotations

import argparse
from datetime import date
import math
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
VENDOR_DIR = ROOT_DIR / ".vendor"
if VENDOR_DIR.exists():
    sys.path.insert(0, str(VENDOR_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd

from src.research.shortlist_bakeoff_service import MODEL_FEATURE_COLUMNS, expand_model_feature_columns
from src.research.shortlist_model_service import ShortlistModelService
from src.settings import get_settings, load_feature_config
from src.utils.db_manager import DatabaseManager


RAW_TARGET_COLUMN = "alpha_vs_sector_60d"
RANK_TARGET_COLUMN = "alpha_vs_sector_60d_pct_rank"
CANDIDATE_MODELS = ("signal_proxy", "ridge_model", "lasso_model", "xgboost_model")


def _fmt(value: object, *, places: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(number):
        return "n/a"
    return f"{number:+.{places}f}"


def add_rank_percentile_target(
    frame: pd.DataFrame,
    *,
    source_column: str = RAW_TARGET_COLUMN,
    target_column: str = RANK_TARGET_COLUMN,
) -> pd.DataFrame:
    working = frame.copy()
    working["snapshot_date"] = pd.to_datetime(working["snapshot_date"], errors="coerce").dt.normalize()
    source = pd.to_numeric(working[source_column], errors="coerce")
    working[target_column] = source.groupby(working["snapshot_date"]).rank(method="average", pct=True)
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


def _pooled_spearman(frame: pd.DataFrame, *, target_column: str) -> float:
    score = pd.to_numeric(frame["predicted_alpha"], errors="coerce")
    target = pd.to_numeric(frame[target_column], errors="coerce")
    valid = score.notna() & target.notna()
    if int(valid.sum()) < 3 or score[valid].nunique(dropna=True) < 2 or target[valid].nunique(dropna=True) < 2:
        return float("nan")
    corr = score[valid].corr(target[valid], method="spearman")
    return float(corr) if pd.notna(corr) and math.isfinite(float(corr)) else float("nan")


def _mean_per_date_spearman(frame: pd.DataFrame, *, target_column: str) -> float:
    values: list[float] = []
    for _snapshot_date, day_frame in frame.groupby("snapshot_date", sort=True):
        corr = _pooled_spearman(day_frame, target_column=target_column)
        if math.isfinite(corr):
            values.append(corr)
    return float(pd.Series(values).mean()) if values else float("nan")


def _decile_calibration(
    frame: pd.DataFrame,
    *,
    raw_target_column: str = RAW_TARGET_COLUMN,
    rank_target_column: str = RANK_TARGET_COLUMN,
    deciles: int = 10,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _snapshot_date, day_frame in frame.groupby("snapshot_date", sort=True):
        working = day_frame.copy()
        score = pd.to_numeric(working["predicted_alpha"], errors="coerce")
        raw_target = pd.to_numeric(working[raw_target_column], errors="coerce")
        rank_target = pd.to_numeric(working[rank_target_column], errors="coerce")
        valid = score.notna() & raw_target.notna() & rank_target.notna()
        working = working.loc[valid].copy()
        if len(working.index) < deciles:
            continue
        score_rank = pd.to_numeric(working["predicted_alpha"], errors="coerce").rank(method="first", ascending=True)
        working["decile"] = pd.qcut(score_rank, q=deciles, labels=False, duplicates="drop")
        for decile, decile_frame in working.groupby("decile", sort=True):
            rows.append(
                {
                    "decile": int(decile),
                    "rows": int(len(decile_frame.index)),
                    "mean_raw_target": float(pd.to_numeric(decile_frame[raw_target_column], errors="coerce").mean()),
                    "mean_rank_target": float(pd.to_numeric(decile_frame[rank_target_column], errors="coerce").mean()),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["decile", "rows", "mean_raw_target", "mean_rank_target"])
    return (
        pd.DataFrame(rows)
        .groupby("decile", as_index=False)
        .agg(
            rows=("rows", "sum"),
            mean_raw_target=("mean_raw_target", "mean"),
            mean_rank_target=("mean_rank_target", "mean"),
        )
        .sort_values("decile")
        .reset_index(drop=True)
    )


def _calibration_spread(deciles: pd.DataFrame, *, column: str) -> float:
    if deciles.empty or column not in deciles.columns:
        return float("nan")
    values = pd.to_numeric(deciles[column], errors="coerce").dropna()
    if values.empty:
        return float("nan")
    return float(values.iloc[-1] - values.iloc[0])


def _prediction_summary(*, model: str, label: str, frame: pd.DataFrame) -> dict[str, object]:
    deciles = _decile_calibration(frame)
    return {
        "model": model,
        "label": label,
        "dates": int(frame["snapshot_date"].nunique()),
        "rows": int(len(frame.index)),
        "pooled_raw_spearman": _pooled_spearman(frame, target_column=RAW_TARGET_COLUMN),
        "per_date_raw_spearman": _mean_per_date_spearman(frame, target_column=RAW_TARGET_COLUMN),
        "pooled_rank_spearman": _pooled_spearman(frame, target_column=RANK_TARGET_COLUMN),
        "per_date_rank_spearman": _mean_per_date_spearman(frame, target_column=RANK_TARGET_COLUMN),
        "raw_decile_spread": _calibration_spread(deciles, column="mean_raw_target"),
        "rank_decile_spread": _calibration_spread(deciles, column="mean_rank_target"),
    }


def _comparison_rows(predictions: dict[tuple[str, str], pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    rows: list[dict[str, object]] = []
    aligned_rank_deciles: dict[str, pd.DataFrame] = {}
    for model in CANDIDATE_MODELS:
        raw = predictions.get((model, "raw_label"), pd.DataFrame())
        rank = predictions.get((model, "rank_label"), pd.DataFrame())
        dates = _common_dates(raw, rank)
        raw = _filter_dates(raw, dates)
        rank = _filter_dates(rank, dates)
        raw_summary = _prediction_summary(model=model, label="raw_label", frame=raw)
        rank_summary = _prediction_summary(model=model, label="rank_label", frame=rank)
        delta = float(rank_summary["pooled_raw_spearman"]) - float(raw_summary["pooled_raw_spearman"])
        rows.append({**raw_summary, "delta_rank_minus_raw": float("nan")})
        rows.append({**rank_summary, "delta_rank_minus_raw": delta})
        aligned_rank_deciles[model] = _decile_calibration(rank)
    return pd.DataFrame(rows), aligned_rank_deciles


def _verdict(summary: pd.DataFrame, *, win_threshold: float = 0.02) -> tuple[str, str]:
    rank_rows = summary[summary["label"].astype(str).eq("rank_label")].copy()
    deltas = pd.to_numeric(rank_rows["delta_rank_minus_raw"], errors="coerce").dropna()
    wins = int((deltas >= win_threshold).sum()) if not deltas.empty else 0
    best_delta = float(deltas.max()) if not deltas.empty else float("nan")
    mean_delta = float(deltas.mean()) if not deltas.empty else float("nan")
    if wins >= 2:
        return "rank label wins", f"{wins} models improved pooled raw-alpha Spearman by at least {win_threshold:.4f}; best delta {_fmt(best_delta)}, mean delta {_fmt(mean_delta)}"
    if not deltas.empty and (deltas <= -win_threshold).sum() >= 2:
        return "rank label loses", f"{int((deltas <= -win_threshold).sum())} models lost at least {win_threshold:.4f}; best delta {_fmt(best_delta)}, mean delta {_fmt(mean_delta)}"
    return "rank label ties", f"{wins} models cleared the +{win_threshold:.4f} win bar; best delta {_fmt(best_delta)}, mean delta {_fmt(mean_delta)}"


def _render_report(
    *,
    output_path: Path,
    summary: pd.DataFrame,
    rank_deciles: dict[str, pd.DataFrame],
    eligible_rows: int,
    eligible_dates: int,
    honest_rows: int,
    honest_dates: int,
    config: dict[str, object],
) -> str:
    verdict, verdict_reason = _verdict(summary)
    lines = [
        "# C1 Rank-Percentile Label Bakeoff",
        "",
        f"- generated_at: {date.today().isoformat()}",
        "- data_access: read-only DuckDB; no `./sq` writes; no `data/` mutation",
        "- experiment: train raw 60d alpha versus per-date rank-percentile 60d alpha, evaluate both against original 60d alpha on the honest stride-60 OOS grid",
        f"- target_raw: {RAW_TARGET_COLUMN}",
        f"- target_rank: {RANK_TARGET_COLUMN}",
        f"- candidate_models: {', '.join(CANDIDATE_MODELS)}",
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
        f"- honest_rows: {honest_rows}",
        f"- honest_dates: {honest_dates}",
        f"- verdict: {verdict} ({verdict_reason})",
        "",
        "## Honest-Grid Spearman",
        "",
        "| model | trained_label | honest_dates | honest_rows | pooled_raw_spearman | per_date_raw_spearman | pooled_rank_spearman | per_date_rank_spearman | raw_decile_spread | rank_decile_spread | delta_rank_minus_raw |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    ordered = summary.sort_values(["model", "label"]).reset_index(drop=True)
    for row in ordered.itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.label} | {int(row.dates)} | {int(row.rows)} | "
            f"{_fmt(row.pooled_raw_spearman)} | {_fmt(row.per_date_raw_spearman)} | "
            f"{_fmt(row.pooled_rank_spearman)} | {_fmt(row.per_date_rank_spearman)} | "
            f"{_fmt(row.raw_decile_spread)} | {_fmt(row.rank_decile_spread)} | "
            f"{_fmt(row.delta_rank_minus_raw)} |"
        )
    lines.extend(["", "## Rank-Label Decile Calibration", ""])
    for model in CANDIDATE_MODELS:
        deciles = rank_deciles.get(model, pd.DataFrame())
        lines.extend(
            [
                f"### {model}",
                "",
                "| decile | rows | mean_raw_target | mean_rank_target |",
                "|---:|---:|---:|---:|",
            ]
        )
        if deciles.empty:
            lines.append("| n/a | 0 | n/a | n/a |")
        else:
            for row in deciles.itertuples(index=False):
                lines.append(
                    f"| {int(row.decile)} | {int(row.rows)} | {_fmt(row.mean_raw_target)} | {_fmt(row.mean_rank_target)} |"
                )
        lines.append("")
    lines.extend(["## Verdict", "", f"{verdict}: {verdict_reason}.", ""])
    text = "\n".join(lines)
    output_path.write_text(text, encoding="utf-8")
    return text


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


def _run_model(
    service: ShortlistModelService,
    eligible: pd.DataFrame,
    *,
    model_name: str,
    target_column: str,
    config: dict[str, object],
    feature_columns: list[str],
) -> pd.DataFrame:
    xgboost_params = None
    if model_name == "xgboost_model":
        xgboost_params = {**service._xgboost_params_for_config(str(config["xgboost_config"])), "n_jobs": 1}
    predictions = service._walk_forward_predictions(
        eligible,
        target_column=target_column,
        evaluation_target_column=target_column,
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
        raise SystemExit(f"{model_name} produced no OOS predictions for {target_column}")
    targets = eligible[["snapshot_date", "ticker", RAW_TARGET_COLUMN, RANK_TARGET_COLUMN]].copy()
    predictions = predictions.drop(columns=[RAW_TARGET_COLUMN, RANK_TARGET_COLUMN], errors="ignore").merge(
        targets,
        on=["snapshot_date", "ticker"],
        how="left",
    )
    if RANK_TARGET_COLUMN not in predictions.columns:
        ranks = eligible[["snapshot_date", "ticker", RANK_TARGET_COLUMN]].copy()
        predictions = predictions.merge(ranks, on=["snapshot_date", "ticker"], how="left")
    predictions["model_name"] = model_name
    predictions["trained_target"] = target_column
    return predictions


def run(*, output_path: Path) -> dict[str, object]:
    settings = get_settings()
    service = ShortlistModelService(DatabaseManager(paths=settings.paths))
    feature_columns = service._filter_model_feature_columns(expand_model_feature_columns(MODEL_FEATURE_COLUMNS))
    frame = _read_universe_snapshots(
        duckdb_path=settings.paths.duckdb_path,
        columns=_snapshot_columns(service._filter_model_feature_columns(MODEL_FEATURE_COLUMNS)),
    )
    frame = add_rank_percentile_target(service._prepare_snapshot_frame(frame))
    eligible = service._build_matured_eligible_universe(
        frame,
        target_column=RAW_TARGET_COLUMN,
        eligible_universe_mode=str(_shortlist_config()["eligible_universe_mode"]),
    )
    config = _shortlist_config()
    predictions: dict[tuple[str, str], pd.DataFrame] = {}
    calendar_dates = _read_calendar_dates(duckdb_path=settings.paths.duckdb_path)
    for model_name in CANDIDATE_MODELS:
        for label, target_column in (("raw_label", RAW_TARGET_COLUMN), ("rank_label", RANK_TARGET_COLUMN)):
            print(f"training {model_name} {label}", file=sys.stderr, flush=True)
            dense = _run_model(
                service,
                eligible,
                model_name=model_name,
                target_column=target_column,
                config=config,
                feature_columns=feature_columns,
            )
            honest = service._non_overlapping_oos_predictions(
                dense,
                horizon_days=int(config["label_horizon_dates"]),
                calendar_dates=calendar_dates,
            )
            predictions[(model_name, label)] = honest
    summary, rank_deciles = _comparison_rows(predictions)
    honest_rows = int(sum(len(frame.index) for frame in predictions.values()))
    honest_dates = int(
        pd.concat(predictions.values(), ignore_index=True)["snapshot_date"].nunique()
        if predictions
        else 0
    )
    _render_report(
        output_path=output_path,
        summary=summary,
        rank_deciles=rank_deciles,
        eligible_rows=len(eligible.index),
        eligible_dates=int(eligible["snapshot_date"].nunique()),
        honest_rows=honest_rows,
        honest_dates=honest_dates,
        config=config,
    )
    verdict, verdict_reason = _verdict(summary)
    return {
        "output_path": str(output_path),
        "verdict": verdict,
        "verdict_reason": verdict_reason,
        "honest_dates": honest_dates,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the C1 rank-percentile label bakeoff.")
    parser.add_argument("--output", type=Path, default=Path("reports/c1_rank_label_bakeoff.md"))
    args = parser.parse_args()
    result = run(output_path=args.output)
    print(
        "wrote {output_path} honest_dates={honest_dates} verdict={verdict}: {verdict_reason}".format(
            **result
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
