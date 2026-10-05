from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.research.shortlist_model_service import PROMOTION_BASKET_SIZE, ShortlistModelService
from src.settings import get_settings


class _ReportDB:
    def __init__(self, *, calendar_dates: list[pd.Timestamp]) -> None:
        self._calendar_dates = calendar_dates

    def list_universe_daily_snapshot_dates(self) -> list[pd.Timestamp]:
        return self._calendar_dates


def _fmt(value: object, *, places: int = 4) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not pd.notna(numeric):
        return "n/a"
    return f"{numeric:.{places}f}"


def _target_column(frame: pd.DataFrame) -> str:
    if "artifact_evaluation_target_column" in frame.columns:
        values = frame["artifact_evaluation_target_column"].dropna().astype(str)
        if not values.empty and values.iloc[0]:
            return str(values.iloc[0])
    for column in frame.columns:
        if column.startswith("alpha_vs_sector_") and column.endswith("d"):
            return column
    raise ValueError("Unable to infer evaluation target column from OOS predictions.")


def _horizon_days(target_column: str) -> int:
    marker = "alpha_vs_sector_"
    if marker in target_column and target_column.endswith("d"):
        suffix = target_column.split(marker, 1)[1].removesuffix("d")
        return max(int(suffix), 1)
    return 60


def _model_frames(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if "model_name" not in frame.columns:
        raise ValueError("OOS predictions must include model_name.")
    return {
        str(model_name): model_frame.copy()
        for model_name, model_frame in frame.groupby("model_name", sort=True)
        if str(model_name) and not model_frame.empty
    }


def _read_calendar_dates(*, duckdb_path: Path | None, fallback_dates: list[pd.Timestamp]) -> list[pd.Timestamp]:
    if duckdb_path is not None and duckdb_path.exists():
        try:
            import duckdb

            with duckdb.connect(str(duckdb_path), read_only=True) as connection:
                rows = connection.execute(
                    "SELECT DISTINCT snapshot_date FROM universe_daily_snapshots ORDER BY snapshot_date ASC"
                ).fetchall()
            parsed = pd.to_datetime(pd.Series([row[0] for row in rows]), errors="coerce").dropna()
            if not parsed.empty:
                return sorted(parsed.dt.normalize().drop_duplicates().tolist())
        except Exception:
            pass
    return sorted(pd.to_datetime(pd.Series(fallback_dates), errors="coerce").dropna().dt.normalize().drop_duplicates().tolist())


def _acceptance_summaries(
    *,
    service: ShortlistModelService,
    predictions_by_model: dict[str, pd.DataFrame],
    target_column: str,
    test_window_dates: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_name, predictions in predictions_by_model.items():
        summaries = service._rolling_window_summaries(
            predictions=predictions,
            target_column=target_column,
            model_name=model_name,
            top_n=PROMOTION_BASKET_SIZE,
            windows=service._promotion_recent_windows(horizon_days=_horizon_days(target_column)),
            fold_windows=service._promotion_fold_windows(horizon_days=_horizon_days(target_column)),
            fold_size=int(test_window_dates),
            include_full_oos=True,
        )
        rows.extend(summaries.to_dict(orient="records"))
    return pd.DataFrame(rows)


def _full_summaries(
    *,
    service: ShortlistModelService,
    predictions_by_model: dict[str, pd.DataFrame],
    target_column: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            service._evaluate_predictions(
                predictions=predictions,
                top_n=PROMOTION_BASKET_SIZE,
                target_column=target_column,
                model_name=model_name,
            )
            for model_name, predictions in predictions_by_model.items()
        ]
    )


def _passes(
    *,
    service: ShortlistModelService,
    model_name: str,
    acceptance_summaries: pd.DataFrame,
    promotion_gate: dict[str, float | int],
    horizon_days: int,
) -> bool:
    return service._model_passes_promotion_gate(
        model_name=model_name,
        acceptance_summaries=acceptance_summaries,
        promotion_gate=promotion_gate,
        required_recent_windows=service._promotion_recent_windows(horizon_days=horizon_days),
        required_fold_windows=service._promotion_fold_windows(horizon_days=horizon_days),
    )


def _summary_by_model(frame: pd.DataFrame) -> dict[str, pd.Series]:
    if frame.empty:
        return {}
    return {str(row.model): row for row in frame.itertuples(index=False)}


def render_report(
    *,
    oos_predictions_path: Path,
    output_path: Path,
    duckdb_path: Path | None = None,
    test_window_dates: int = 20,
) -> str:
    header = pd.read_csv(oos_predictions_path, nrows=0)
    usecols = [
        column
        for column in header.columns
        if column in {"snapshot_date", "ticker", "model_name", "predicted_alpha", "artifact_evaluation_target_column"}
        or (column.startswith("alpha_vs_sector_") and column.endswith("d"))
    ]
    frame = pd.read_csv(oos_predictions_path, usecols=usecols, low_memory=False)
    frame["snapshot_date"] = pd.to_datetime(frame["snapshot_date"], errors="coerce").dt.normalize()
    frame = frame.dropna(subset=["snapshot_date", "ticker", "predicted_alpha"])
    target_column = _target_column(frame)
    horizon = _horizon_days(target_column)
    calendar_dates = _read_calendar_dates(
        duckdb_path=duckdb_path,
        fallback_dates=sorted(frame["snapshot_date"].drop_duplicates().tolist()),
    )
    service = ShortlistModelService(_ReportDB(calendar_dates=calendar_dates))
    dense_by_model = _model_frames(frame)
    honest_by_model = {
        model_name: service._non_overlapping_oos_predictions(
            predictions,
            horizon_days=horizon,
            calendar_dates=calendar_dates,
        )
        for model_name, predictions in dense_by_model.items()
    }
    dense_full = _full_summaries(
        service=service,
        predictions_by_model=dense_by_model,
        target_column=target_column,
    )
    honest_full = _full_summaries(
        service=service,
        predictions_by_model=honest_by_model,
        target_column=target_column,
    )
    dense_acceptance = _acceptance_summaries(
        service=service,
        predictions_by_model=dense_by_model,
        target_column=target_column,
        test_window_dates=test_window_dates,
    )
    honest_acceptance = _acceptance_summaries(
        service=service,
        predictions_by_model=honest_by_model,
        target_column=target_column,
        test_window_dates=test_window_dates,
    )
    gate = service._load_promotion_gate()
    dense_rows = _summary_by_model(dense_full)
    honest_rows = _summary_by_model(honest_full)
    lines = [
        "# Q3b OOS De-Overlap",
        "",
        f"- source_oos_predictions: {oos_predictions_path}",
        f"- target_column: {target_column}",
        f"- horizon_days: {horizon}",
        f"- deoverlap_policy: greedy non-overlapping rows per ticker using the {horizon}-session forward label window",
        f"- calendar_dates: {len(calendar_dates)}",
        f"- dense_rows: {sum(len(frame.index) for frame in dense_by_model.values())}",
        f"- honest_rows: {sum(len(frame.index) for frame in honest_by_model.values())}",
        f"- dense_dates: {frame['snapshot_date'].nunique()}",
        f"- honest_dates: {pd.concat(honest_by_model.values(), ignore_index=True)['snapshot_date'].nunique() if honest_by_model else 0}",
        "",
        "## Full-OOS Spearman",
        "",
        "| model | dense_dates | honest_dates | dense_spearman | honest_spearman | delta |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model_name in sorted(dense_by_model):
        dense = dense_rows.get(model_name)
        honest = honest_rows.get(model_name)
        dense_s = getattr(dense, "spearman", float("nan")) if dense is not None else float("nan")
        honest_s = getattr(honest, "spearman", float("nan")) if honest is not None else float("nan")
        lines.append(
            f"| {model_name} | "
            f"{int(getattr(dense, 'dates', 0)) if dense is not None else 0} | "
            f"{int(getattr(honest, 'dates', 0)) if honest is not None else 0} | "
            f"{_fmt(dense_s)} | {_fmt(honest_s)} | {_fmt(float(honest_s) - float(dense_s) if pd.notna(dense_s) and pd.notna(honest_s) else float('nan'))} |"
        )
    lines.extend(
        [
            "",
            "## Gate Re-Adjudication",
            "",
            "| model | dense_gate | honest_gate | dense_full_oos | honest_full_oos | honest_last_fold | honest_trailing_3folds |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for model_name in sorted(dense_by_model):
        dense_pass = _passes(
            service=service,
            model_name=model_name,
            acceptance_summaries=dense_acceptance,
            promotion_gate=gate,
            horizon_days=horizon,
        )
        honest_pass = _passes(
            service=service,
            model_name=model_name,
            acceptance_summaries=honest_acceptance,
            promotion_gate=gate,
            horizon_days=horizon,
        )
        dense_full_row = dense_acceptance[dense_acceptance["model"].astype(str).eq(f"{model_name}_full_oos")]
        honest_full_row = honest_acceptance[honest_acceptance["model"].astype(str).eq(f"{model_name}_full_oos")]
        last_row = honest_acceptance[honest_acceptance["model"].astype(str).eq(f"{model_name}_last_fold")]
        trailing_row = honest_acceptance[honest_acceptance["model"].astype(str).eq(f"{model_name}_trailing_3folds")]
        lines.append(
            f"| {model_name} | {'PASS' if dense_pass else 'FAIL'} | {'PASS' if honest_pass else 'FAIL'} | "
            f"{_fmt(dense_full_row.iloc[0].get('spearman') if not dense_full_row.empty else float('nan'))} | "
            f"{_fmt(honest_full_row.iloc[0].get('spearman') if not honest_full_row.empty else float('nan'))} | "
            f"{_fmt(last_row.iloc[0].get('spearman') if not last_row.empty else float('nan'))} | "
            f"{_fmt(trailing_row.iloc[0].get('spearman') if not trailing_row.empty else float('nan'))} |"
        )
    lines.append("")
    text = "\n".join(lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return text


def main() -> None:
    paths = get_settings().paths
    render_report(
        oos_predictions_path=paths.reports_dir / "shortlist_model_oos_predictions.csv",
        output_path=paths.reports_dir / "q3b_oos_deoverlap.md",
        duckdb_path=paths.duckdb_path,
    )


if __name__ == "__main__":
    main()
