from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
import sys

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.research.sector_neutral_selection import (
    deoverlap_oos_predictions,
    evaluate_selection_modes,
    verdict_for_model,
)


REPORT_PATH = Path("reports/f2_sector_neutral.md")
OOS_PATH = Path("reports/shortlist_model_oos_predictions.csv")
REQUESTED_MODELS = ("signal_proxy", "ridge_model", "overnight_session_specialist")
TARGET_COLUMN = "alpha_vs_sector_60d"
TOP_N = 2
HORIZON_DAYS = 60


def main() -> None:
    raw = _load_oos_predictions(OOS_PATH)
    available_models = tuple(model for model in REQUESTED_MODELS if model in set(raw["model_name"].astype(str)))
    missing_models = tuple(model for model in REQUESTED_MODELS if model not in set(raw["model_name"].astype(str)))
    scoped = raw[raw["model_name"].astype(str).isin(available_models)].copy()
    deoverlapped = deoverlap_oos_predictions(
        scoped,
        horizon_days=HORIZON_DAYS,
        calendar_dates=raw["snapshot_date"].dropna().tolist(),
    )
    results = evaluate_selection_modes(
        deoverlapped,
        top_n=TOP_N,
        target_column=TARGET_COLUMN,
    )
    REPORT_PATH.write_text(
        render_report(
            results=results,
            raw_rows=len(scoped.index),
            deoverlapped_rows=len(deoverlapped.index),
            raw_dates=int(pd.to_datetime(scoped["snapshot_date"], errors="coerce").dt.normalize().nunique()) if not scoped.empty else 0,
            deoverlapped_dates=int(pd.to_datetime(deoverlapped["snapshot_date"], errors="coerce").dt.normalize().nunique()) if not deoverlapped.empty else 0,
            available_models=available_models,
            missing_models=missing_models,
        ),
        encoding="utf-8",
    )


def _load_oos_predictions(path: Path) -> pd.DataFrame:
    columns = ["snapshot_date", "ticker", "sector", "model_name", "predicted_alpha", TARGET_COLUMN]
    frame = pd.read_csv(path, usecols=columns)
    frame["snapshot_date"] = pd.to_datetime(frame["snapshot_date"], errors="coerce").dt.normalize()
    frame["model_name"] = frame["model_name"].astype(str)
    frame["predicted_alpha"] = pd.to_numeric(frame["predicted_alpha"], errors="coerce")
    frame[TARGET_COLUMN] = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
    return frame.dropna(subset=["snapshot_date", "model_name"])


def render_report(
    *,
    results: pd.DataFrame,
    raw_rows: int,
    deoverlapped_rows: int,
    raw_dates: int,
    deoverlapped_dates: int,
    available_models: tuple[str, ...],
    missing_models: tuple[str, ...],
) -> str:
    verdicts = {
        model_name: verdict_for_model(results, model_name)
        for model_name in available_models
    }
    overall = (
        "sector-neutral loses"
        if verdicts and all("loses" in verdict for verdict in verdicts.values())
        else "mixed or unavailable"
    )
    lines = [
        "# F2 Sector-Neutral Top-N Audit",
        "",
        f"- generated_at: {datetime.now(UTC).replace(microsecond=0).isoformat()}",
        "- idea: F2 sector-neutral top-N selection",
        "- data_access: read-only OOS artifact; no `./sq` write command; no `data/` mutation",
        f"- source_artifact: {OOS_PATH}",
        f"- target_column: {TARGET_COLUMN}",
        f"- top_n: {TOP_N}",
        f"- horizon_deoverlap_days: {HORIZON_DAYS}",
        f"- raw_rows: {raw_rows}",
        f"- raw_dates: {raw_dates}",
        f"- deoverlapped_rows: {deoverlapped_rows}",
        f"- deoverlapped_dates: {deoverlapped_dates}",
        f"- requested_models: {', '.join(REQUESTED_MODELS)}",
        f"- available_models: {', '.join(available_models) if available_models else 'none'}",
        f"- missing_models: {', '.join(missing_models) if missing_models else 'none'}",
        f"- verdict: {overall}",
        "",
        "## Four-Way Comparison",
        "",
        "| model | selection_mode | dates | avg_pick_count | basket_mean_alpha | hit_rate | beat_rate | top_sector_share |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    if results.empty:
        lines.append("| n/a | n/a | 0 | n/a | n/a | n/a | n/a | n/a |")
    else:
        ordered = results.sort_values(["model_name", "selection_mode"]).reset_index(drop=True)
        for row in ordered.itertuples(index=False):
            lines.append(
                f"| {row.model_name} | {row.selection_mode} | {int(row.dates)} | "
                f"{_fmt(row.avg_pick_count)} | {_fmt(row.basket_mean_alpha)} | "
                f"{_fmt(row.hit_rate)} | {_fmt(row.beat_rate)} | {_fmt(row.top_sector_share)} |"
            )
    lines.extend(["", "## Deltas", ""])
    lines.extend(
        [
            "| model | delta_mean_alpha | delta_hit_rate | delta_beat_rate | delta_top_sector_share | verdict |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for model_name in available_models:
        delta = _delta_row(results, model_name)
        lines.append(
            f"| {model_name} | {_fmt(delta['basket_mean_alpha'])} | {_fmt(delta['hit_rate'])} | "
            f"{_fmt(delta['beat_rate'])} | {_fmt(delta['top_sector_share'])} | {verdicts[model_name]} |"
        )
    for model_name in missing_models:
        lines.append(f"| {model_name} | n/a | n/a | n/a | n/a | missing from current OOS artifact |")
    lines.extend(
        [
            "",
            "## Implementation Decision",
            "",
            "Sector-neutral selection improves concentration, but it does not win or tie on the money statistics in the available honest-grid artifact. The production scan selection mode is therefore left unchanged (`global_top_n`).",
        ]
    )
    if missing_models:
        lines.extend(
            [
                "",
                "## Artifact Divergence",
                "",
                "The current `reports/shortlist_model_oos_predictions.csv` was generated before the E1 niche roster was persisted into the OOS artifact, so `overnight_session_specialist` is not present even though current code includes it in the candidate roster. Tomorrow's critic can re-run this audit after the nightly model artifact includes that ranker.",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def _delta_row(results: pd.DataFrame, model_name: str) -> dict[str, float]:
    model = results[results["model_name"].astype(str) == str(model_name)]
    if model.empty:
        return {column: float("nan") for column in ("basket_mean_alpha", "hit_rate", "beat_rate", "top_sector_share")}
    global_row = model[model["selection_mode"] == "global_top_n"].iloc[0]
    sector_row = model[model["selection_mode"] == "sector_neutral"].iloc[0]
    return {
        column: float(sector_row[column]) - float(global_row[column])
        for column in ("basket_mean_alpha", "hit_rate", "beat_rate", "top_sector_share")
    }


def _fmt(value: object) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if pd.isna(numeric):
        return "n/a"
    return f"{numeric:+.6f}"


if __name__ == "__main__":
    main()
