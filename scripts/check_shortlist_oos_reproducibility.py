from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


PROMOTION_BASKET_SIZE = 2


def _evaluate(frame: pd.DataFrame, *, target_column: str, top_n: int = PROMOTION_BASKET_SIZE) -> dict[str, float | int]:
    rows: list[dict[str, float | int]] = []
    for _snapshot_date, day_frame in frame.groupby("snapshot_date", sort=True):
        ordered = day_frame.sort_values(["predicted_alpha", "ticker"], ascending=[False, True]).copy()
        picks = ordered.head(int(top_n)).copy()
        target = pd.to_numeric(picks[target_column], errors="coerce").dropna()
        universe = pd.to_numeric(day_frame[target_column], errors="coerce").dropna()
        if target.empty or universe.empty:
            continue
        score = pd.to_numeric(ordered["predicted_alpha"], errors="coerce")
        full_target = pd.to_numeric(ordered[target_column], errors="coerce")
        valid = score.notna() & full_target.notna()
        spearman = float("nan")
        if int(valid.sum()) >= 3 and score[valid].nunique(dropna=True) > 1 and full_target[valid].nunique(dropna=True) > 1:
            corr = score[valid].corr(full_target[valid], method="spearman")
            if pd.notna(corr) and math.isfinite(float(corr)):
                spearman = float(corr)
        rows.append(
            {
                "pick_count": len(picks.index),
                "mean_target": float(target.mean()),
                "hit_rate": float((target > 0.0).mean()),
                "universe_mean_target": float(universe.mean()),
                "spearman": spearman,
            }
        )
    if not rows:
        return {"dates": 0}
    summary = pd.DataFrame(rows)
    return {
        "dates": len(summary.index),
        "avg_pick_count": float(summary["pick_count"].mean()),
        "mean_target": float(summary["mean_target"].mean()),
        "hit_rate": float(summary["hit_rate"].mean()),
        "beat_universe_rate": float((summary["mean_target"] > summary["universe_mean_target"]).mean()),
        "spearman": float(summary["spearman"].dropna().mean()) if summary["spearman"].notna().any() else float("nan"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Recompute shortlist model OOS acceptance summaries from persisted CSV rows.")
    parser.add_argument("--csv", type=Path, default=Path("reports/shortlist_model_oos_predictions.csv"))
    parser.add_argument("--target-column", default="alpha_vs_sector_20d")
    args = parser.parse_args()

    frame = pd.read_csv(args.csv)
    required = {"snapshot_date", "ticker", "model_name", "predicted_alpha", args.target_column}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise SystemExit(f"Missing required columns in {args.csv}: {', '.join(missing)}")
    frame["snapshot_date"] = pd.to_datetime(frame["snapshot_date"]).dt.normalize()
    for model_name, model_frame in frame.groupby("model_name", sort=True):
        summary = _evaluate(model_frame, target_column=args.target_column)
        print(f"{model_name}: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
