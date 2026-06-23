from __future__ import annotations

from dataclasses import dataclass
import math

import pandas as pd

from src.research.shortlist_model_service import ShortlistModelService
from src.research.shortlist_universe import normalize_eligible_universe_mode, normalize_model_scope
from src.settings import load_feature_config
from src.utils.db_manager import DatabaseManager


@dataclass(frozen=True)
class ShortlistScoreboardReport:
    output_path: str
    generated_at: str
    run_champion_model: str
    recommended_model: str
    production_model: str
    production_eligible_universe_mode: str
    production_model_scope: str
    production_xgboost_config: str
    model_count: int


class ShortlistScoreboardService:
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
    ) -> ShortlistScoreboardReport:
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
        run_champion_model = str(latest_run["champion_model"])
        (
            production_model,
            production_eligible_universe_mode,
            production_model_scope,
            production_xgboost_config,
        ) = self._production_shortlist_config()

        predictions = self.db_manager.load_shortlist_model_predictions(
            generated_at=generated_at,
            horizon_days=int(horizon_days),
            eligible_universe_mode=eligible_universe_mode,
            model_scope=model_scope,
            dataset_split="oos",
        )
        if predictions.empty:
            raise ValueError("Latest shortlist model run has no out-of-sample predictions.")
        predictions["snapshot_date"] = pd.to_datetime(predictions["snapshot_date"]).dt.normalize()
        predictions["actual_alpha_vs_sector"] = pd.to_numeric(predictions["actual_alpha_vs_sector"], errors="coerce")
        predictions["predicted_alpha"] = pd.to_numeric(predictions["predicted_alpha"], errors="coerce")

        live_predictions = self.db_manager.load_shortlist_model_predictions(
            generated_at=generated_at,
            horizon_days=int(horizon_days),
            eligible_universe_mode=eligible_universe_mode,
            model_scope=model_scope,
            dataset_split="live",
        )
        if not live_predictions.empty:
            live_predictions["snapshot_date"] = pd.to_datetime(live_predictions["snapshot_date"]).dt.normalize()
            live_predictions["predicted_alpha"] = pd.to_numeric(live_predictions["predicted_alpha"], errors="coerce")

        model_names = sorted(str(model_name) for model_name in predictions["model_name"].dropna().unique().tolist())
        if not model_names:
            raise ValueError("Latest shortlist model run has no model predictions to score.")

        full_rows: list[dict[str, object]] = []
        recent40_rows: list[dict[str, object]] = []
        window_sections: list[str] = []
        sector_sections: list[str] = []
        live_sections: list[str] = []
        promotion_rows: list[dict[str, object]] = []

        for model_name in model_names:
            model_predictions = predictions[predictions["model_name"] == model_name].copy()
            full_summary = self._evaluate_model(model_predictions, top_n=int(top_n), model_name=model_name)
            recent40_predictions = self._recent_window(model_predictions, 40)
            recent40_summary = self._evaluate_model(recent40_predictions, top_n=int(top_n), model_name=model_name)
            full_rows.append(full_summary)
            recent40_rows.append(recent40_summary)
            promotion_rows.append(
                self._promotion_decision_row(
                    model_name=model_name,
                    full_summary=full_summary,
                    recent40_summary=recent40_summary,
                )
            )
            window_sections.extend(self._render_window_section(model_name=model_name, predictions=model_predictions, top_n=int(top_n)))
            sector_sections.extend(self._render_sector_section(model_name=model_name, predictions=model_predictions, top_n=int(top_n)))
            model_live = live_predictions[live_predictions["model_name"] == model_name].copy()
            live_sections.extend(self._render_live_concentration_section(model_name=model_name, predictions=model_live, top_n=int(top_n)))

        full_frame = pd.DataFrame(full_rows)
        recent40_frame = pd.DataFrame(recent40_rows)
        promotion_frame = pd.DataFrame(promotion_rows)
        recommended_model = self._recommend_model(promotion_frame)

        report_path = self.db_manager.paths.reports_dir / "shortlist_scoreboard.md"
        lines = [
            "# Shortlist Scoreboard",
            "",
            f"- generated_at: {generated_at}",
            f"- horizon_days: {int(horizon_days)}",
            f"- top_n: {int(top_n)}",
            f"- eligible_universe_mode: {eligible_universe_mode}",
            f"- model_scope: {model_scope}",
            f"- run_champion_model: {run_champion_model}",
            f"- recommended_model: {recommended_model}",
            f"- production_model: {production_model}",
            f"- production_eligible_universe_mode: {production_eligible_universe_mode}",
            f"- production_model_scope: {production_model_scope}",
            f"- production_xgboost_config: {production_xgboost_config}",
            f"- live_snapshot_date: {latest_run['live_snapshot_date'] or 'n/a'}",
            "",
        ]
        lines.extend(self._render_promotion_table(promotion_frame))
        lines.extend(self._render_summary_table(full_frame, heading="## Full OOS Scoreboard"))
        lines.extend(self._render_summary_table(recent40_frame, heading="## Recent 40 OOS Dates"))
        lines.extend(["## Window Scorecards", ""])
        lines.extend(window_sections)
        lines.extend(["## Sector Scorecards", ""])
        lines.extend(sector_sections)
        lines.extend(["## Live Concentration", ""])
        lines.extend(live_sections)
        report_path.write_text("\n".join(lines), encoding="utf-8")

        return ShortlistScoreboardReport(
            output_path=str(report_path),
            generated_at=generated_at,
            run_champion_model=run_champion_model,
            recommended_model=recommended_model,
            production_model=production_model,
            production_eligible_universe_mode=production_eligible_universe_mode,
            production_model_scope=production_model_scope,
            production_xgboost_config=production_xgboost_config,
            model_count=len(model_names),
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
            needs_refresh = str(runs.iloc[0]["live_snapshot_date"] or "") != str(latest_snapshot_date)
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

    def _production_shortlist_config(self) -> tuple[str, str, str, str]:
        config = load_feature_config()
        shortlist_model = (config.get("scan_policy", {}) or {}).get("shortlist_model", {}) or {}
        production_model_name = shortlist_model.get("production_model_name")
        production_eligible_universe_mode = shortlist_model.get(
            "production_eligible_universe_mode",
            shortlist_model.get("eligible_universe_mode", "passed_only"),
        )
        production_model_scope = shortlist_model.get("production_model_scope", "global")
        production_xgboost_config = shortlist_model.get("production_xgboost_config", "baseline")
        return (
            str(production_model_name or "xgboost_model"),
            str(production_eligible_universe_mode or "passed_only"),
            str(production_model_scope or "global"),
            str(production_xgboost_config or "baseline"),
        )

    def _recent_window(self, predictions: pd.DataFrame, window: int) -> pd.DataFrame:
        dates = sorted(predictions["snapshot_date"].drop_duplicates().tolist())
        selected_dates = dates[-min(int(window), len(dates)) :]
        return predictions[predictions["snapshot_date"].isin(selected_dates)].copy()

    def _pick_top_n(self, day_frame: pd.DataFrame, *, top_n: int) -> pd.DataFrame:
        return day_frame.sort_values(["predicted_alpha", "ticker"], ascending=[False, True]).head(int(top_n)).copy()

    def _evaluate_model(self, predictions: pd.DataFrame, *, top_n: int, model_name: str) -> dict[str, object]:
        rows: list[dict[str, float]] = []
        for _, day_frame in predictions.groupby("snapshot_date", sort=True):
            picks = self._pick_top_n(day_frame, top_n=int(top_n))
            target = pd.to_numeric(picks["actual_alpha_vs_sector"], errors="coerce").dropna()
            universe_target = pd.to_numeric(day_frame["actual_alpha_vs_sector"], errors="coerce").dropna()
            if target.empty or universe_target.empty:
                continue
            sector_weights = picks["sector"].value_counts(normalize=True)
            rows.append(
                {
                    "mean_target": float(target.mean()),
                    "hit_rate": float((target > 0.0).mean()),
                    "universe_mean_target": float(universe_target.mean()),
                    "max_sector_share": float(sector_weights.max()) if not sector_weights.empty else float("nan"),
                    "sector_hhi": float((sector_weights.pow(2)).sum()) if not sector_weights.empty else float("nan"),
                }
            )
        if not rows:
            return self._empty_summary(model_name)
        frame = pd.DataFrame(rows)
        return {
            "model": model_name,
            "dates": len(frame.index),
            "mean_target": float(frame["mean_target"].mean()),
            "hit_rate": float(frame["hit_rate"].mean()),
            "beat_universe_rate": float((frame["mean_target"] > frame["universe_mean_target"]).mean()),
            "positive_date_rate": float((frame["mean_target"] > 0.0).mean()),
            "ge_2pct_rate": float((frame["mean_target"] >= 0.02).mean()),
            "ge_5pct_rate": float((frame["mean_target"] >= 0.05).mean()),
            "avg_max_sector_share": float(frame["max_sector_share"].mean()),
            "avg_sector_hhi": float(frame["sector_hhi"].mean()),
        }

    def _promotion_decision_row(self, *, model_name: str, full_summary: dict[str, object], recent40_summary: dict[str, object]) -> dict[str, object]:
        full_status = self._acceptance_status(full_summary)
        recent_status = self._acceptance_status(recent40_summary)
        if full_status == "strong" and recent_status == "strong":
            decision = "promote_strong"
        elif full_status in {"strong", "minimum"} and recent_status in {"strong", "minimum"}:
            decision = "promote_minimum"
        else:
            decision = "research_only"
        return {
            "model": model_name,
            "decision": decision,
            "full_status": full_status,
            "recent40_status": recent_status,
            "full_mean_target": full_summary.get("mean_target"),
            "full_beat_universe_rate": full_summary.get("beat_universe_rate"),
            "full_positive_date_rate": full_summary.get("positive_date_rate"),
            "recent40_mean_target": recent40_summary.get("mean_target"),
            "recent40_beat_universe_rate": recent40_summary.get("beat_universe_rate"),
            "recent40_positive_date_rate": recent40_summary.get("positive_date_rate"),
            "avg_max_sector_share": full_summary.get("avg_max_sector_share"),
            "avg_sector_hhi": full_summary.get("avg_sector_hhi"),
        }

    def _acceptance_status(self, summary: dict[str, object]) -> str:
        mean_target = float(summary.get("mean_target", float("nan")))
        beat_universe_rate = float(summary.get("beat_universe_rate", float("nan")))
        positive_date_rate = float(summary.get("positive_date_rate", float("nan")))
        if (
            math.isfinite(mean_target)
            and math.isfinite(beat_universe_rate)
            and math.isfinite(positive_date_rate)
            and mean_target > 0.03
            and beat_universe_rate > 0.60
            and positive_date_rate > 0.55
        ):
            return "strong"
        if (
            math.isfinite(mean_target)
            and math.isfinite(beat_universe_rate)
            and math.isfinite(positive_date_rate)
            and mean_target > 0.02
            and beat_universe_rate > 0.55
            and positive_date_rate > 0.55
        ):
            return "minimum"
        return "fail"

    def _recommend_model(self, promotion_frame: pd.DataFrame) -> str:
        ordered = promotion_frame.copy()
        decision_priority = {"promote_strong": 0, "promote_minimum": 1, "research_only": 2}
        status_priority = {"strong": 0, "minimum": 1, "fail": 2}
        ordered["decision_priority"] = ordered["decision"].map(decision_priority).fillna(9)
        ordered["recent_priority"] = ordered["recent40_status"].map(status_priority).fillna(9)
        ordered = ordered.sort_values(
            [
                "decision_priority",
                "recent_priority",
                "full_beat_universe_rate",
                "recent40_beat_universe_rate",
                "full_mean_target",
                "recent40_positive_date_rate",
                "recent40_mean_target",
                "avg_sector_hhi",
                "avg_max_sector_share",
                "model",
            ],
            ascending=[True, True, False, False, False, False, False, False, True, True],
        ).reset_index(drop=True)
        return str(ordered.iloc[0]["model"])

    def _render_promotion_table(self, frame: pd.DataFrame) -> list[str]:
        lines = ["## Promotion Decisions", ""]
        ordered = frame.sort_values(
            ["decision", "full_beat_universe_rate", "recent40_mean_target", "model"],
            ascending=[True, False, False, True],
        ).reset_index(drop=True)
        for row in ordered.itertuples(index=False):
            lines.append(f"### {row.model}")
            lines.append(f"- decision: {row.decision}")
            lines.append(f"- full_status: {row.full_status}")
            lines.append(f"- recent40_status: {row.recent40_status}")
            lines.append(f"- full_mean_target: {self._fmt(row.full_mean_target)}")
            lines.append(f"- full_beat_universe_rate: {self._fmt(row.full_beat_universe_rate)}")
            lines.append(f"- full_positive_date_rate: {self._fmt(row.full_positive_date_rate)}")
            lines.append(f"- recent40_mean_target: {self._fmt(row.recent40_mean_target)}")
            lines.append(f"- recent40_beat_universe_rate: {self._fmt(row.recent40_beat_universe_rate)}")
            lines.append(f"- recent40_positive_date_rate: {self._fmt(row.recent40_positive_date_rate)}")
            lines.append(f"- avg_max_sector_share: {self._fmt(row.avg_max_sector_share)}")
            lines.append(f"- avg_sector_hhi: {self._fmt(row.avg_sector_hhi)}")
            lines.append("")
        return lines

    def _render_summary_table(self, frame: pd.DataFrame, *, heading: str) -> list[str]:
        lines = [heading, ""]
        ordered = frame.sort_values(
            ["mean_target", "beat_universe_rate", "model"],
            ascending=[False, False, True],
        ).reset_index(drop=True)
        for row in ordered.itertuples(index=False):
            lines.append(f"### {row.model}")
            lines.append(f"- dates: {int(row.dates)}")
            lines.append(f"- mean_target: {self._fmt(row.mean_target)}")
            lines.append(f"- hit_rate: {self._fmt(row.hit_rate)}")
            lines.append(f"- beat_universe_rate: {self._fmt(row.beat_universe_rate)}")
            lines.append(f"- positive_date_rate: {self._fmt(row.positive_date_rate)}")
            lines.append(f"- ge_2pct_rate: {self._fmt(row.ge_2pct_rate)}")
            lines.append(f"- ge_5pct_rate: {self._fmt(row.ge_5pct_rate)}")
            lines.append(f"- avg_max_sector_share: {self._fmt(row.avg_max_sector_share)}")
            lines.append(f"- avg_sector_hhi: {self._fmt(row.avg_sector_hhi)}")
            lines.append("")
        return lines

    def _render_window_section(self, *, model_name: str, predictions: pd.DataFrame, top_n: int) -> list[str]:
        lines = [f"### {model_name}", ""]
        for window in (20, 40, 60):
            scoped = self._recent_window(predictions, window)
            summary = self._evaluate_model(scoped, top_n=int(top_n), model_name=model_name)
            lines.append(f"- last_{min(window, int(predictions['snapshot_date'].nunique()))}d_mean_target: {self._fmt(summary['mean_target'])}")
            lines.append(f"- last_{min(window, int(predictions['snapshot_date'].nunique()))}d_beat_universe_rate: {self._fmt(summary['beat_universe_rate'])}")
            lines.append(f"- last_{min(window, int(predictions['snapshot_date'].nunique()))}d_positive_date_rate: {self._fmt(summary['positive_date_rate'])}")
        lines.append("")
        return lines

    def _render_sector_section(self, *, model_name: str, predictions: pd.DataFrame, top_n: int) -> list[str]:
        lines = [f"### {model_name}", ""]
        rows: list[dict[str, object]] = []
        for _, day_frame in predictions.groupby("snapshot_date", sort=True):
            picks = self._pick_top_n(day_frame, top_n=int(top_n))
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
        frame = (
            pd.DataFrame(rows)
            .groupby("sector", as_index=False)
            .agg(
                avg_pick_count=("pick_count", "mean"),
                mean_target=("mean_target", "mean"),
                hit_rate=("hit_rate", "mean"),
            )
            .sort_values(["mean_target", "sector"], ascending=[False, True])
            .reset_index(drop=True)
        )
        for row in frame.itertuples(index=False):
            lines.append(
                f"- {row.sector}: avg_pick_count={self._fmt(row.avg_pick_count)}, "
                f"mean_target={self._fmt(row.mean_target)}, hit_rate={self._fmt(row.hit_rate)}"
            )
        lines.append("")
        return lines

    def _render_live_concentration_section(self, *, model_name: str, predictions: pd.DataFrame, top_n: int) -> list[str]:
        lines = [f"### {model_name}", ""]
        if predictions.empty:
            lines.append("No live predictions available.")
            lines.append("")
            return lines
        picks = self._pick_top_n(predictions, top_n=int(top_n))
        sector_counts = picks["sector"].value_counts(normalize=True)
        max_sector_share = float(sector_counts.max()) if not sector_counts.empty else float("nan")
        sector_hhi = float((sector_counts.pow(2)).sum()) if not sector_counts.empty else float("nan")
        lines.append(f"- live_max_sector_share: {self._fmt(max_sector_share)}")
        lines.append(f"- live_sector_hhi: {self._fmt(sector_hhi)}")
        if not sector_counts.empty:
            lines.append(f"- live_sector_mix: {', '.join(f'{sector} {share:.0%}' for sector, share in sector_counts.items())}")
        lines.append("")
        return lines

    def _fmt(self, value: float) -> str:
        if value is None or not math.isfinite(float(value)):
            return "nan"
        return f"{float(value):.6f}"
