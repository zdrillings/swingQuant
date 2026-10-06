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
from src.research.shortlist_model_service import SHORTLIST_HEURISTIC_MODELS, ShortlistModelService
from src.settings import get_settings, load_feature_config
from src.utils.db_manager import DatabaseManager


RAW_TARGET_COLUMN = "alpha_vs_sector_60d"
RANK_TARGET_COLUMN = "alpha_vs_sector_60d_pct_rank"
MODEL_VARIANTS = (
    "ridge_regression",
    "ridge_rank_label",
    "xgboost_regression",
    "xgboost_pairwise",
)


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


def same_date_pair_labels(frame: pd.DataFrame, *, target_column: str = RAW_TARGET_COLUMN) -> pd.DataFrame:
    """Return explicit same-date winner/loser pairs for auditing the pairwise objective."""
    rows: list[dict[str, object]] = []
    working = frame.copy()
    working["snapshot_date"] = pd.to_datetime(working["snapshot_date"], errors="coerce").dt.normalize()
    working[target_column] = pd.to_numeric(working[target_column], errors="coerce")
    for snapshot_date, day_frame in working.dropna(subset=["snapshot_date", target_column]).groupby("snapshot_date", sort=True):
        day = day_frame.sort_values(["ticker"]).reset_index(drop=True)
        for left_index in range(len(day.index)):
            for right_index in range(left_index + 1, len(day.index)):
                left = day.iloc[left_index]
                right = day.iloc[right_index]
                delta = float(left[target_column]) - float(right[target_column])
                if delta == 0.0:
                    continue
                winner = left if delta > 0 else right
                loser = right if delta > 0 else left
                rows.append(
                    {
                        "snapshot_date": snapshot_date,
                        "winner_ticker": str(winner["ticker"]),
                        "loser_ticker": str(loser["ticker"]),
                        "target_delta": abs(delta),
                    }
                )
    return pd.DataFrame(rows, columns=["snapshot_date", "winner_ticker", "loser_ticker", "target_delta"])


def same_date_pair_count(frame: pd.DataFrame, *, target_column: str = RAW_TARGET_COLUMN) -> tuple[int, int]:
    working = frame.copy()
    working["snapshot_date"] = pd.to_datetime(working["snapshot_date"], errors="coerce").dt.normalize()
    working[target_column] = pd.to_numeric(working[target_column], errors="coerce")
    total = 0
    dates = 0
    for _snapshot_date, day_frame in working.dropna(subset=["snapshot_date", target_column]).groupby("snapshot_date", sort=True):
        values = day_frame[target_column]
        count = int(len(values.index))
        if count < 2:
            continue
        tied_pairs = int(sum(int(tie_count) * (int(tie_count) - 1) // 2 for tie_count in values.value_counts().tolist()))
        date_pairs = count * (count - 1) // 2 - tied_pairs
        if date_pairs > 0:
            total += int(date_pairs)
            dates += 1
    return total, dates


def pairwise_group_sizes(frame: pd.DataFrame) -> list[int]:
    working = frame.copy()
    working["snapshot_date"] = pd.to_datetime(working["snapshot_date"], errors="coerce").dt.normalize()
    return [int(len(group.index)) for _date, group in working.groupby("snapshot_date", sort=True) if len(group.index) > 0]


def _ranker_relevance(frame: pd.DataFrame, *, target_column: str = RAW_TARGET_COLUMN) -> pd.Series:
    working = frame.copy()
    working["snapshot_date"] = pd.to_datetime(working["snapshot_date"], errors="coerce").dt.normalize()
    target = pd.to_numeric(working[target_column], errors="coerce")
    relevance = target.groupby(working["snapshot_date"]).rank(method="average", ascending=True) - 1.0
    return relevance.fillna(0.0).astype(float)


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


def _feature_sets(service: ShortlistModelService) -> dict[str, list[str]]:
    current = service._filter_model_feature_columns(expand_model_feature_columns(MODEL_FEATURE_COLUMNS))
    without_a4 = [feature for feature in current if not str(feature).startswith("a4_")]
    return {"without_a4": without_a4, "with_a4": current}


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


def _summary_row(*, feature_set: str, variant: str, predictions: pd.DataFrame) -> dict[str, object]:
    return {
        "feature_set": feature_set,
        "variant": variant,
        "dates": int(predictions["snapshot_date"].nunique()) if not predictions.empty else 0,
        "rows": int(len(predictions.index)),
        "pooled_raw_spearman": _pooled_spearman(predictions, target_column=RAW_TARGET_COLUMN),
        "per_date_raw_spearman": _mean_per_date_spearman(predictions, target_column=RAW_TARGET_COLUMN),
    }


def _score_pairwise_xgboost_global(
    service: ShortlistModelService,
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    xgboost_params: dict[str, float | int],
) -> pd.DataFrame | None:
    try:
        from xgboost import XGBRanker
    except ModuleNotFoundError:
        service.logger.warning("xgboost unavailable; skipping pairwise ranker.")
        return None
    train_frame = train_frame.sort_values(["snapshot_date", "ticker"]).reset_index(drop=True)
    test_frame = test_frame.sort_values(["snapshot_date", "ticker"]).reset_index(drop=True)
    train_matrix, test_matrix, _feature_names, _standardized_test = service._prepare_model_matrices(
        train_frame,
        test_frame,
        feature_columns_override=feature_columns,
    )
    params = {
        "n_estimators": 150,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1.0,
        "reg_lambda": 1.0,
        "random_state": 42,
        "objective": "rank:pairwise",
        "n_jobs": 1,
    }
    params.update(xgboost_params)
    group = pairwise_group_sizes(train_frame)
    if not group:
        return None
    model = XGBRanker(**params)
    model.fit(train_matrix, _ranker_relevance(train_frame).to_numpy(dtype=float), group=group, verbose=False)
    scored = test_frame.copy()
    scored["predicted_alpha"] = model.predict(test_matrix)
    scored["model_top_reasons"] = [[] for _ in range(len(scored.index))]
    scored["model_reason_summary"] = ""
    return scored


def _score_pairwise_xgboost(
    service: ShortlistModelService,
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    model_scope: str,
    xgboost_params: dict[str, float | int],
) -> pd.DataFrame | None:
    if model_scope == "sector_specific":
        frames: list[pd.DataFrame] = []
        for sector, sector_test in test_frame.groupby("sector", sort=False):
            sector_train = train_frame[train_frame["sector"] == sector].copy()
            scoped_train = sector_train if len(sector_train.index) >= 120 and sector_train["snapshot_date"].nunique() >= 40 else train_frame
            scored = _score_pairwise_xgboost_global(
                service,
                scoped_train,
                sector_test.copy(),
                feature_columns=feature_columns,
                xgboost_params=xgboost_params,
            )
            if scored is not None and not scored.empty:
                frames.append(scored)
        return pd.concat(frames, ignore_index=True) if frames else None
    return _score_pairwise_xgboost_global(
        service,
        train_frame,
        test_frame,
        feature_columns=feature_columns,
        xgboost_params=xgboost_params,
    )


def _walk_forward_pairwise_predictions(
    service: ShortlistModelService,
    frame: pd.DataFrame,
    *,
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
        fold_feature_columns = feature_columns
        ic_survivors = service._feature_ic_survivors_from_frame(
            train_frame,
            target_column=RAW_TARGET_COLUMN,
            min_feature_ic=service._load_min_feature_ic(),
            min_observation_fraction=service._load_min_feature_ic_observation_fraction(),
        )
        allowed = set(feature_columns)
        fold_feature_columns = [feature for feature in ic_survivors if feature in allowed]
        if not fold_feature_columns:
            start_index += stride
            continue
        scored = _score_pairwise_xgboost(
            service,
            train_frame,
            test_frame,
            feature_columns=fold_feature_columns,
            model_scope=str(config["model_scope"]),
            xgboost_params=xgboost_params,
        )
        if scored is not None and not scored.empty:
            scored["model_name"] = "xgboost_pairwise"
            scored["trained_target"] = "same_date_pairwise_rank"
            folds.append(
                scored[
                    [
                        "snapshot_date",
                        "ticker",
                        "sector",
                        "md_volume_30d",
                        RAW_TARGET_COLUMN,
                        "predicted_alpha",
                        "model_top_reasons",
                        "model_reason_summary",
                    ]
                ].copy()
            )
        start_index += stride
    return pd.concat(folds, ignore_index=True) if folds else None


def _run_service_model(
    service: ShortlistModelService,
    eligible: pd.DataFrame,
    *,
    variant: str,
    config: dict[str, object],
    feature_columns: list[str],
) -> pd.DataFrame:
    model_name = "ridge_model" if variant.startswith("ridge") else "xgboost_model"
    target_column = RANK_TARGET_COLUMN if variant == "ridge_rank_label" else RAW_TARGET_COLUMN
    xgboost_params = None
    if model_name == "xgboost_model":
        xgboost_params = {**service._xgboost_params_for_config(str(config["xgboost_config"])), "n_jobs": 1}
    predictions = service._walk_forward_predictions(
        eligible,
        target_column=target_column,
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
        raise SystemExit(f"{variant} produced no OOS predictions")
    targets = eligible[["snapshot_date", "ticker", RAW_TARGET_COLUMN, RANK_TARGET_COLUMN]].copy()
    predictions = predictions.drop(columns=[RAW_TARGET_COLUMN, RANK_TARGET_COLUMN], errors="ignore").merge(
        targets,
        on=["snapshot_date", "ticker"],
        how="left",
    )
    predictions["model_name"] = variant
    predictions["trained_target"] = target_column
    return predictions


def _comparison_summary(rows: pd.DataFrame) -> pd.DataFrame:
    working = rows.copy()
    regression = working[working["variant"].eq("xgboost_regression")][["feature_set", "pooled_raw_spearman"]]
    regression = regression.rename(columns={"pooled_raw_spearman": "xgboost_regression_spearman"})
    working = working.merge(regression, on="feature_set", how="left")
    working["delta_vs_xgboost_regression"] = (
        pd.to_numeric(working["pooled_raw_spearman"], errors="coerce")
        - pd.to_numeric(working["xgboost_regression_spearman"], errors="coerce")
    )
    return working


def _verdict(summary: pd.DataFrame, *, win_threshold: float = 0.005) -> tuple[str, str]:
    pairwise = summary[summary["variant"].eq("xgboost_pairwise")].copy()
    deltas = pd.to_numeric(pairwise["delta_vs_xgboost_regression"], errors="coerce").dropna()
    best_delta = float(deltas.max()) if not deltas.empty else float("nan")
    if not deltas.empty and (deltas >= win_threshold).any():
        return "pairwise wins", f"best xgboost pairwise delta {_fmt(best_delta)} cleared the +{win_threshold:.4f} bar"
    return "pairwise ties", f"best xgboost pairwise delta {_fmt(best_delta)} did not clear the +{win_threshold:.4f} bar"


def _render_report(
    *,
    output_path: Path,
    summary: pd.DataFrame,
    eligible_rows: int,
    eligible_dates: int,
    pair_rows: int,
    pair_dates: int,
    config: dict[str, object],
) -> str:
    verdict, reason = _verdict(summary)
    lines = [
        "# E2 Pairwise Loss Bakeoff",
        "",
        f"- generated_at: {date.today().isoformat()}",
        "- data_access: read-only DuckDB; no `./sq` writes; no `data/` mutation",
        "- experiment: compare raw-alpha regression with xgboost `rank:pairwise` on same-date groups; ridge raw/rank-label rows are the linear comparison",
        f"- target_raw: {RAW_TARGET_COLUMN}",
        f"- pairwise_label: same-date target ordering, encoded as per-date rank relevance",
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
        f"- audited_same_date_pairs: {pair_rows}",
        f"- audited_pair_dates: {pair_dates}",
        f"- verdict: {verdict} ({reason})",
        "",
        "## Honest-Grid Spearman",
        "",
        "| feature_set | variant | honest_dates | honest_rows | pooled_raw_spearman | per_date_raw_spearman | delta_vs_xgboost_regression |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary.sort_values(["feature_set", "variant"]).itertuples(index=False):
        lines.append(
            f"| {row.feature_set} | {row.variant} | {int(row.dates)} | {int(row.rows)} | "
            f"{_fmt(row.pooled_raw_spearman)} | {_fmt(row.per_date_raw_spearman)} | "
            f"{_fmt(row.delta_vs_xgboost_regression)} |"
        )
    pairwise = summary[summary["variant"].eq("xgboost_pairwise")].copy()
    if not pairwise.empty:
        lines.extend(["", "## Wiring Plan", ""])
        best = pairwise.sort_values("delta_vs_xgboost_regression", ascending=False).iloc[0]
        if float(best["delta_vs_xgboost_regression"]) >= 0.005:
            lines.append(
                "Pairwise cleared the research bar. Follow-on: add a gated `xgboost_pairwise` candidate to the shortlist roster, reuse the same fold-local IC screen and promotion gate, and keep regression xgboost as the control until a production dry run passes."
            )
        else:
            lines.append(
                "Pairwise did not clear the +0.005 research bar; leave production training unchanged and keep searching for feature-side rank signal."
            )
    lines.extend(["", "## Verdict", "", f"{verdict}: {reason}.", ""])
    text = "\n".join(lines)
    output_path.write_text(text, encoding="utf-8")
    return text


def run(*, output_path: Path) -> dict[str, object]:
    settings = get_settings()
    service = ShortlistModelService(DatabaseManager(paths=settings.paths))
    base_features = [
        feature
        for feature in service._filter_model_feature_columns(MODEL_FEATURE_COLUMNS)
        if not str(feature).startswith("a4_")
    ]
    frame = _read_universe_snapshots(
        duckdb_path=settings.paths.duckdb_path,
        columns=_snapshot_columns(base_features),
    )
    frame = service._add_a4_regime_interaction_features(
        service._prepare_snapshot_frame(frame),
        horizon_sessions=int(_shortlist_config()["label_horizon_dates"]),
    )
    frame = add_rank_percentile_target(frame)
    config = _shortlist_config()
    eligible = service._build_matured_eligible_universe(
        frame,
        target_column=RAW_TARGET_COLUMN,
        eligible_universe_mode=str(config["eligible_universe_mode"]),
    )
    pair_rows, pair_dates = same_date_pair_count(eligible)
    calendar_dates = _read_calendar_dates(duckdb_path=settings.paths.duckdb_path)
    rows: list[dict[str, object]] = []
    for feature_set, feature_columns in _feature_sets(service).items():
        for variant in MODEL_VARIANTS:
            print(f"training {variant} {feature_set}", file=sys.stderr, flush=True)
            if variant == "xgboost_pairwise":
                dense = _walk_forward_pairwise_predictions(
                    service,
                    eligible,
                    config=config,
                    feature_columns=feature_columns,
                )
                if dense is None or dense.empty:
                    raise SystemExit(f"{variant} produced no OOS predictions")
            else:
                dense = _run_service_model(
                    service,
                    eligible,
                    variant=variant,
                    config=config,
                    feature_columns=feature_columns,
                )
            honest = service._non_overlapping_oos_predictions(
                dense,
                horizon_days=int(config["label_horizon_dates"]),
                calendar_dates=calendar_dates,
            )
            rows.append(_summary_row(feature_set=feature_set, variant=variant, predictions=honest))
    summary = _comparison_summary(pd.DataFrame(rows))
    _render_report(
        output_path=output_path,
        summary=summary,
        eligible_rows=len(eligible.index),
        eligible_dates=int(eligible["snapshot_date"].nunique()),
        pair_rows=pair_rows,
        pair_dates=pair_dates,
        config=config,
    )
    verdict, reason = _verdict(summary)
    return {"output_path": str(output_path), "verdict": verdict, "verdict_reason": reason}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the E2 pairwise learning-to-rank bakeoff.")
    parser.add_argument("--output", type=Path, default=Path("reports/e2_pairwise_loss.md"))
    args = parser.parse_args()
    result = run(output_path=args.output)
    print("wrote {output_path} verdict={verdict}: {verdict_reason}".format(**result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
