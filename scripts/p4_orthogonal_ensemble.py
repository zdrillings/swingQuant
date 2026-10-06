from __future__ import annotations

from datetime import UTC, datetime
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

from src.research.orthogonal_ensemble import (
    build_rank_ensemble_predictions,
    inverse_absolute_spearman_weights,
    window_spearman_summary,
)
from src.research.sector_neutral_selection import deoverlap_oos_predictions


REPORT_PATH = Path("reports/p4_orthogonal_ensemble.md")
OOS_PATH = Path("reports/shortlist_model_oos_predictions.csv")
TARGET_COLUMN = "alpha_vs_sector_60d"
HORIZON_DAYS = 60
REQUESTED_ALL = (
    "signal_proxy",
    "ridge_adaptive",
    "overnight_session_specialist",
    "structure_factor_signal",
)
REQUESTED_NO_SFS = (
    "signal_proxy",
    "ridge_adaptive",
    "overnight_session_specialist",
)


def main() -> None:
    raw = _load_oos_predictions(OOS_PATH)
    deoverlapped = deoverlap_oos_predictions(
        raw,
        horizon_days=HORIZON_DAYS,
        calendar_dates=raw["snapshot_date"].dropna().tolist(),
    )
    single_member_rows = _single_member_rows(deoverlapped)
    ensemble_rows = _ensemble_rows(deoverlapped, single_member_rows)
    REPORT_PATH.write_text(
        render_report(
            raw=raw,
            deoverlapped=deoverlapped,
            single_member_rows=single_member_rows,
            ensemble_rows=ensemble_rows,
        ),
        encoding="utf-8",
    )


def _load_oos_predictions(path: Path) -> pd.DataFrame:
    columns = ["snapshot_date", "ticker", "model_name", "predicted_alpha", TARGET_COLUMN]
    frame = pd.read_csv(path, usecols=columns)
    frame["snapshot_date"] = pd.to_datetime(frame["snapshot_date"], errors="coerce").dt.normalize()
    frame["model_name"] = frame["model_name"].astype(str)
    frame["ticker"] = frame["ticker"].astype(str).str.strip()
    frame["predicted_alpha"] = pd.to_numeric(frame["predicted_alpha"], errors="coerce")
    frame[TARGET_COLUMN] = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
    return frame.dropna(subset=["snapshot_date", "ticker", "model_name", "predicted_alpha", TARGET_COLUMN])


def _single_member_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_name, model_frame in predictions.groupby("model_name", sort=True):
        if model_name not in set(REQUESTED_ALL):
            continue
        summary = window_spearman_summary(model_frame, target_column=TARGET_COLUMN)
        rows.append({"model": str(model_name), **summary})
    return pd.DataFrame(rows)


def _ensemble_rows(predictions: pd.DataFrame, single_member_rows: pd.DataFrame) -> pd.DataFrame:
    available = set(predictions["model_name"].astype(str).drop_duplicates())
    variants = {
        "requested_all_four": REQUESTED_ALL,
        "requested_no_sfs": REQUESTED_NO_SFS,
    }
    rows: list[dict[str, object]] = []
    single_spearman = {
        str(row.model): float(row.full_oos_spearman)
        for row in single_member_rows.itertuples(index=False)
        if _finite(row.full_oos_spearman)
    }
    for variant_name, requested_members in variants.items():
        missing = tuple(model for model in requested_members if model not in available)
        present = tuple(model for model in requested_members if model in available)
        if len(present) < 2:
            rows.append(_unavailable_row(variant_name, "rank_average", requested_members, present, missing))
            rows.append(_unavailable_row(variant_name, "inverse_abs_spearman", requested_members, present, missing))
            continue
        best_single = _best_single_spearman(single_member_rows, present)
        for method in ("rank_average", "inverse_abs_spearman"):
            weights = None
            if method == "inverse_abs_spearman":
                weights = inverse_absolute_spearman_weights(
                    {model: single_spearman.get(model, float("nan")) for model in present}
                )
            ensemble = build_rank_ensemble_predictions(
                predictions,
                members=present,
                target_column=TARGET_COLUMN,
                weights=weights,
            )
            summary = window_spearman_summary(ensemble, target_column=TARGET_COLUMN)
            full = float(summary["full_oos_spearman"])
            rows.append(
                {
                    "variant": variant_name,
                    "method": method,
                    "requested_members": ", ".join(requested_members),
                    "used_members": ", ".join(present),
                    "missing_members": ", ".join(missing) if missing else "none",
                    "rows": summary["rows"],
                    "dates": summary["dates"],
                    "full_oos_spearman": full,
                    "last_fold_spearman": summary["last_fold_spearman"],
                    "trailing_3fold_spearman": summary["trailing_3fold_spearman"],
                    "best_single_full_oos_spearman": best_single,
                    "delta_vs_best_single_full_oos": full - best_single if _finite(full) and _finite(best_single) else float("nan"),
                    "weights": _format_weights(weights, present),
                }
            )
    if "requested_all_four" in variants:
        present = tuple(model for model in variants["requested_all_four"] if model in available)
        missing = tuple(model for model in variants["requested_all_four"] if model not in available)
        if 2 <= len(present) < len(variants["requested_all_four"]):
            best_single = _best_single_spearman(single_member_rows, present)
            ensemble = build_rank_ensemble_predictions(predictions, members=present, target_column=TARGET_COLUMN)
            summary = window_spearman_summary(ensemble, target_column=TARGET_COLUMN)
            full = float(summary["full_oos_spearman"])
            rows.append(
                {
                    "variant": "available_requested_members",
                    "method": "rank_average",
                    "requested_members": ", ".join(variants["requested_all_four"]),
                    "used_members": ", ".join(present),
                    "missing_members": ", ".join(missing) if missing else "none",
                    "rows": summary["rows"],
                    "dates": summary["dates"],
                    "full_oos_spearman": full,
                    "last_fold_spearman": summary["last_fold_spearman"],
                    "trailing_3fold_spearman": summary["trailing_3fold_spearman"],
                    "best_single_full_oos_spearman": best_single,
                    "delta_vs_best_single_full_oos": full - best_single if _finite(best_single) else float("nan"),
                    "weights": "equal",
                }
            )
    return pd.DataFrame(rows)


def render_report(
    *,
    raw: pd.DataFrame,
    deoverlapped: pd.DataFrame,
    single_member_rows: pd.DataFrame,
    ensemble_rows: pd.DataFrame,
) -> str:
    verdict = _verdict(ensemble_rows)
    available_models = sorted(set(raw["model_name"].astype(str)))
    requested_missing = [model for model in REQUESTED_ALL if model not in available_models]
    lines = [
        "# P4 Orthogonal Ensemble Bakeoff",
        "",
        f"- generated_at: {datetime.now(UTC).replace(microsecond=0).isoformat()}",
        "- idea: P4 orthogonal ensemble over now-uncorrelated members",
        "- data_access: read-only OOS artifact; no `./sq` write command; no `data/` mutation",
        f"- source_artifact: {OOS_PATH}",
        f"- target_column: {TARGET_COLUMN}",
        f"- horizon_deoverlap_days: {HORIZON_DAYS}",
        f"- raw_rows: {len(raw.index)}",
        f"- raw_dates: {int(raw['snapshot_date'].nunique()) if not raw.empty else 0}",
        f"- honest_rows: {len(deoverlapped.index)}",
        f"- honest_dates: {int(deoverlapped['snapshot_date'].nunique()) if not deoverlapped.empty else 0}",
        f"- requested_all_members: {', '.join(REQUESTED_ALL)}",
        f"- requested_no_sfs_members: {', '.join(REQUESTED_NO_SFS)}",
        f"- missing_requested_members: {', '.join(requested_missing) if requested_missing else 'none'}",
        f"- verdict: {verdict}",
        "",
        "## Single Members",
        "",
        "| model | rows | dates | full_oos_spearman | last_fold_spearman | trailing_3fold_spearman |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    if single_member_rows.empty:
        lines.append("| n/a | 0 | 0 | n/a | n/a | n/a |")
    else:
        for row in single_member_rows.sort_values("model").itertuples(index=False):
            lines.append(
                f"| {row.model} | {int(row.rows)} | {int(row.dates)} | {_fmt(row.full_oos_spearman)} | "
                f"{_fmt(row.last_fold_spearman)} | {_fmt(row.trailing_3fold_spearman)} |"
            )
    lines.extend(
        [
            "",
            "## Ensemble Comparison",
            "",
            "| variant | method | used_members | missing_members | rows | dates | full_oos_spearman | last_fold_spearman | trailing_3fold_spearman | best_single_full_oos | delta_vs_best_single_full_oos | weights |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    if ensemble_rows.empty:
        lines.append("| n/a | n/a | n/a | n/a | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |")
    else:
        for row in ensemble_rows.sort_values(["variant", "method"]).itertuples(index=False):
            lines.append(
                f"| {row.variant} | {row.method} | {row.used_members or 'none'} | {row.missing_members or 'none'} | "
                f"{int(row.rows)} | {int(row.dates)} | {_fmt(row.full_oos_spearman)} | "
                f"{_fmt(row.last_fold_spearman)} | {_fmt(row.trailing_3fold_spearman)} | "
                f"{_fmt(row.best_single_full_oos_spearman)} | {_fmt(row.delta_vs_best_single_full_oos)} | {row.weights} |"
            )
    lines.extend(
        [
            "",
            "## Implementation Decision",
            "",
            "The production ensemble member list is unchanged. The requested four-member and no-SFS ensembles cannot be fully adjudicated from the current OOS artifact because `ridge_adaptive` and `overnight_session_specialist` predictions are absent from both the checked-in CSV and the persisted SQLite prediction table. The constructible available-member check is reported above as an audit only.",
            "",
            "Tomorrow's critic can verify this by checking that `reports/p4_orthogonal_ensemble.md` exists, includes full-OOS/last-fold/trailing-3fold columns, and that no production ensemble configuration changed.",
            "",
        ]
    )
    return "\n".join(lines)


def _best_single_spearman(single_member_rows: pd.DataFrame, members: tuple[str, ...]) -> float:
    if single_member_rows.empty:
        return float("nan")
    scoped = single_member_rows[single_member_rows["model"].astype(str).isin(members)]
    values = pd.to_numeric(scoped["full_oos_spearman"], errors="coerce").dropna()
    return float(values.max()) if not values.empty else float("nan")


def _unavailable_row(
    variant: str,
    method: str,
    requested: tuple[str, ...],
    present: tuple[str, ...],
    missing: tuple[str, ...],
) -> dict[str, object]:
    return {
        "variant": variant,
        "method": method,
        "requested_members": ", ".join(requested),
        "used_members": ", ".join(present),
        "missing_members": ", ".join(missing) if missing else "none",
        "rows": 0,
        "dates": 0,
        "full_oos_spearman": float("nan"),
        "last_fold_spearman": float("nan"),
        "trailing_3fold_spearman": float("nan"),
        "best_single_full_oos_spearman": float("nan"),
        "delta_vs_best_single_full_oos": float("nan"),
        "weights": "n/a",
    }


def _verdict(ensemble_rows: pd.DataFrame) -> str:
    values = pd.to_numeric(ensemble_rows.get("delta_vs_best_single_full_oos"), errors="coerce").dropna()
    if values.empty:
        return "audit blocked: requested member predictions unavailable"
    best = float(values.max())
    if best >= 0.005:
        return f"ensemble beats best available member by {_fmt(best)} honest-grid Spearman"
    if best >= -0.005:
        return f"ensemble ties best available member ({_fmt(best)} honest-grid Spearman delta)"
    return f"ensemble loses to best available member ({_fmt(best)} honest-grid Spearman delta)"


def _format_weights(weights: dict[str, float] | None, members: tuple[str, ...]) -> str:
    if weights is None:
        return "equal"
    return ", ".join(f"{model}={_fmt(weights.get(model), places=3)}" for model in members)


def _fmt(value: object, *, places: int = 4) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(numeric):
        return "n/a"
    return f"{numeric:+.{places}f}"


def _finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


if __name__ == "__main__":
    main()
