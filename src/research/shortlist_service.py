from __future__ import annotations

from dataclasses import dataclass
import json
import math

import pandas as pd

from src.research.shortlist_model_service import ShortlistModelService
from src.research.shortlist_universe import normalize_eligible_universe_mode, normalize_model_scope
from src.utils.db_manager import DatabaseManager
from src.utils.shortlist_runtime import _annotate_live_prediction_comparisons


@dataclass(frozen=True)
class ShortlistReport:
    output_path: str
    generated_at: str
    champion_model: str
    live_snapshot_date: str | None
    candidate_count: int


class ShortlistService:
    def __init__(self, db_manager: DatabaseManager) -> None:
        self.db_manager = db_manager

    def run(
        self,
        *,
        top_n: int = 10,
        horizon_days: int = 20,
        min_train_dates: int = 252,
        test_window_dates: int = 20,
        recent_dates: int = 60,
        refresh_if_stale: bool = True,
        eligible_universe_mode: str = "passed_only",
        model_scope: str = "global",
    ) -> ShortlistReport:
        self.db_manager.initialize()
        eligible_universe_mode = normalize_eligible_universe_mode(eligible_universe_mode)
        model_scope = normalize_model_scope(model_scope)
        latest_run = self._latest_or_refreshed_run(
            top_n=top_n,
            horizon_days=horizon_days,
            min_train_dates=min_train_dates,
            test_window_dates=test_window_dates,
            recent_dates=recent_dates,
            refresh_if_stale=refresh_if_stale,
            eligible_universe_mode=eligible_universe_mode,
            model_scope=model_scope,
        )
        generated_at = str(latest_run["generated_at"])
        champion_model = str(latest_run["champion_model"])
        live_snapshot_date = latest_run["live_snapshot_date"]

        live_predictions = self.db_manager.load_shortlist_model_predictions(
            generated_at=generated_at,
            horizon_days=int(horizon_days),
            eligible_universe_mode=eligible_universe_mode,
            model_scope=model_scope,
            dataset_split="live",
            model_name=champion_model,
        )
        if live_predictions.empty:
            raise ValueError("Latest shortlist model run has no live predictions.")
        live_predictions["snapshot_date"] = pd.to_datetime(live_predictions["snapshot_date"]).dt.normalize()
        if "details_json" in live_predictions.columns:
            details = live_predictions["details_json"].apply(self._parse_prediction_details)
            live_predictions["model_top_reasons"] = details.apply(lambda payload: payload.get("model_top_reasons", []))
            live_predictions["model_reason_summary"] = details.apply(lambda payload: payload.get("model_reason_summary"))
        live_predictions = live_predictions.sort_values(
            ["predicted_alpha", "md_volume_30d", "ticker"],
            ascending=[False, False, True],
        ).reset_index(drop=True)
        live_predictions["model_rank"] = range(1, len(live_predictions.index) + 1)
        live_predictions = _annotate_live_prediction_comparisons(live_predictions, top_n=int(top_n))
        live_predictions = live_predictions.head(int(top_n)).reset_index(drop=True)

        oos_predictions = self.db_manager.load_shortlist_model_predictions(
            generated_at=generated_at,
            horizon_days=int(horizon_days),
            eligible_universe_mode=eligible_universe_mode,
            model_scope=model_scope,
            dataset_split="oos",
            model_name=champion_model,
        )
        if oos_predictions.empty:
            raise ValueError("Latest shortlist model run has no out-of-sample predictions.")
        oos_predictions["snapshot_date"] = pd.to_datetime(oos_predictions["snapshot_date"]).dt.normalize()
        oos_predictions["actual_alpha_vs_sector"] = pd.to_numeric(
            oos_predictions["actual_alpha_vs_sector"],
            errors="coerce",
        )

        report_path = self.db_manager.paths.reports_dir / "shortlist.md"
        lines = [
            "# Shortlist",
            "",
            f"- generated_at: {generated_at}",
            f"- horizon_days: {int(horizon_days)}",
            f"- eligible_universe_mode: {eligible_universe_mode}",
            f"- model_scope: {model_scope}",
            f"- champion_model: {champion_model}",
            f"- live_snapshot_date: {live_snapshot_date or 'n/a'}",
            f"- top_n: {int(top_n)}",
            "",
        ]
        lines.extend(self._render_acceptance_windows(oos_predictions, top_n=int(top_n)))
        lines.extend(self._render_sector_mix(oos_predictions, top_n=int(top_n)))
        lines.extend(self._render_live_candidates(live_predictions))
        report_path.write_text("\n".join(lines), encoding="utf-8")

        return ShortlistReport(
            output_path=str(report_path),
            generated_at=generated_at,
            champion_model=champion_model,
            live_snapshot_date=str(live_snapshot_date) if live_snapshot_date is not None else None,
            candidate_count=len(live_predictions.index),
        )

    def _latest_or_refreshed_run(
        self,
        *,
        top_n: int,
        horizon_days: int,
        min_train_dates: int,
        test_window_dates: int,
        recent_dates: int,
        refresh_if_stale: bool,
        eligible_universe_mode: str,
        model_scope: str,
    ) -> pd.Series:
        runs = self.db_manager.load_shortlist_model_runs(
            horizon_days=int(horizon_days),
            eligible_universe_mode=eligible_universe_mode,
            model_scope=model_scope,
            limit=1,
        )
        latest_snapshot_dates = self.db_manager.list_universe_daily_snapshot_dates()
        latest_snapshot_date = latest_snapshot_dates[-1] if latest_snapshot_dates else None

        needs_refresh = runs.empty
        if not needs_refresh and refresh_if_stale and latest_snapshot_date is not None:
            run_snapshot_date = runs.iloc[0]["live_snapshot_date"]
            needs_refresh = str(run_snapshot_date or "") != str(latest_snapshot_date)

        if needs_refresh:
            ShortlistModelService(self.db_manager).run(
                top_n=int(top_n),
                horizon_days=int(horizon_days),
                min_train_dates=int(min_train_dates),
                test_window_dates=int(test_window_dates),
                recent_dates=int(recent_dates),
                eligible_universe_mode=eligible_universe_mode,
                model_scope=model_scope,
            )
            runs = self.db_manager.load_shortlist_model_runs(
                horizon_days=int(horizon_days),
                eligible_universe_mode=eligible_universe_mode,
                model_scope=model_scope,
                limit=1,
            )
            if runs.empty:
                raise ValueError("Unable to create shortlist model run.")
        return runs.iloc[0]

    def _render_acceptance_windows(self, predictions: pd.DataFrame, *, top_n: int) -> list[str]:
        lines = ["## Acceptance Windows", ""]
        unique_dates = sorted(predictions["snapshot_date"].drop_duplicates().tolist())
        for window in (20, 40, 60):
            scoped_dates = unique_dates[-min(window, len(unique_dates)) :]
            scoped = predictions[predictions["snapshot_date"].isin(scoped_dates)].copy()
            summary = self._evaluate_predictions(scoped, top_n=top_n)
            lines.append(f"### Last {len(scoped_dates)} OOS Dates")
            lines.append(f"- mean_target: {self._fmt(summary['mean_target'])}")
            lines.append(f"- hit_rate: {self._fmt(summary['hit_rate'])}")
            lines.append(f"- beat_universe_rate: {self._fmt(summary['beat_universe_rate'])}")
            lines.append(f"- positive_date_rate: {self._fmt(summary['positive_date_rate'])}")
            lines.append(f"- ge_2pct_rate: {self._fmt(summary['ge_2pct_rate'])}")
            lines.append(f"- ge_5pct_rate: {self._fmt(summary['ge_5pct_rate'])}")
            lines.append("")
        return lines

    def _render_sector_mix(self, predictions: pd.DataFrame, *, top_n: int) -> list[str]:
        lines = ["## OOS Sector Contribution", ""]
        rows: list[dict[str, object]] = []
        for snapshot_date, day_frame in predictions.groupby("snapshot_date", sort=True):
            picks = day_frame.sort_values(["predicted_alpha", "ticker"], ascending=[False, True]).head(int(top_n)).copy()
            for sector, sector_frame in picks.groupby("sector", sort=True):
                target = pd.to_numeric(sector_frame["actual_alpha_vs_sector"], errors="coerce").dropna()
                if target.empty:
                    continue
                rows.append(
                    {
                        "sector": sector,
                        "pick_count": len(sector_frame.index),
                        "mean_target": float(target.mean()),
                        "hit_rate": float((target > 0.0).mean()),
                    }
                )
        if not rows:
            lines.append("No sector contribution available.")
            lines.append("")
            return lines
        frame = pd.DataFrame(rows)
        aggregated = (
            frame.groupby("sector", as_index=False)
            .agg(
                avg_pick_count=("pick_count", "mean"),
                mean_target=("mean_target", "mean"),
                hit_rate=("hit_rate", "mean"),
            )
            .sort_values(["mean_target", "hit_rate", "sector"], ascending=[False, False, True])
            .reset_index(drop=True)
        )
        for row in aggregated.itertuples(index=False):
            lines.append(f"### {row.sector}")
            lines.append(f"- avg_pick_count: {self._fmt(row.avg_pick_count)}")
            lines.append(f"- mean_target: {self._fmt(row.mean_target)}")
            lines.append(f"- hit_rate: {self._fmt(row.hit_rate)}")
            lines.append("")
        return lines

    def _render_live_candidates(self, frame: pd.DataFrame) -> list[str]:
        lines = ["## Live Candidates", ""]
        for row in frame.itertuples(index=False):
            lines.append(f"### {row.ticker}")
            lines.append(f"- sector: {row.sector}")
            lines.append(f"- predicted_alpha: {self._fmt(row.predicted_alpha)}")
            model_reason_summary = getattr(row, "model_reason_summary", None)
            if model_reason_summary:
                lines.append(f"- why: {model_reason_summary}")
            model_comparison_summary = getattr(row, "model_comparison_summary", None)
            if model_comparison_summary:
                lines.append(f"- won over: {model_comparison_summary}")
            lines.append(f"- md_volume_30d: {float(row.md_volume_30d):.0f}")
            lines.append(f"- chart: https://www.tradingview.com/chart/?symbol={row.ticker}")
            lines.append("")
        return lines

    def _evaluate_predictions(self, predictions: pd.DataFrame, *, top_n: int) -> dict[str, float]:
        rows: list[dict[str, float]] = []
        for snapshot_date, day_frame in predictions.groupby("snapshot_date", sort=True):
            picks = day_frame.sort_values(["predicted_alpha", "ticker"], ascending=[False, True]).head(int(top_n)).copy()
            target = pd.to_numeric(picks["actual_alpha_vs_sector"], errors="coerce").dropna()
            universe_target = pd.to_numeric(day_frame["actual_alpha_vs_sector"], errors="coerce").dropna()
            if target.empty or universe_target.empty:
                continue
            rows.append(
                {
                    "mean_target": float(target.mean()),
                    "hit_rate": float((target > 0.0).mean()),
                    "universe_mean_target": float(universe_target.mean()),
                }
            )
        if not rows:
            return {
                "mean_target": float("nan"),
                "hit_rate": float("nan"),
                "beat_universe_rate": float("nan"),
                "positive_date_rate": float("nan"),
                "ge_2pct_rate": float("nan"),
                "ge_5pct_rate": float("nan"),
            }
        frame = pd.DataFrame(rows)
        return {
            "mean_target": float(frame["mean_target"].mean()),
            "hit_rate": float(frame["hit_rate"].mean()),
            "beat_universe_rate": float((frame["mean_target"] > frame["universe_mean_target"]).mean()),
            "positive_date_rate": float((frame["mean_target"] > 0.0).mean()),
            "ge_2pct_rate": float((frame["mean_target"] >= 0.02).mean()),
            "ge_5pct_rate": float((frame["mean_target"] >= 0.05).mean()),
        }

    def _fmt(self, value: float) -> str:
        if value is None or not math.isfinite(float(value)):
            return "nan"
        return f"{float(value):.6f}"

    def _parse_prediction_details(self, value) -> dict:
        if value in (None, ""):
            return {}
        if isinstance(value, dict):
            return value
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
