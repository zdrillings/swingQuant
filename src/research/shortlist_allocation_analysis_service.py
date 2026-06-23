from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import pandas as pd

from src.research.shortlist_model_service import ShortlistModelService
from src.research.shortlist_universe import normalize_eligible_universe_mode, normalize_model_scope
from src.utils.db_manager import DatabaseManager


@dataclass(frozen=True)
class ShortlistAllocationAnalysisReport:
    output_path: str
    generated_at: str
    model_name: str
    policy_count: int
    scope_count: int


@dataclass(frozen=True)
class _ScopeAnalysis:
    scope: str
    generated_at: str
    champion_model: str
    selected_model_name: str
    full_frame: pd.DataFrame
    recent_frame: pd.DataFrame
    live_frames: dict[str, pd.DataFrame]
    recent_oos_date_count: int


class ShortlistAllocationAnalysisService:
    def __init__(self, db_manager: DatabaseManager) -> None:
        self.db_manager = db_manager

    def run(
        self,
        *,
        top_n: int = 6,
        horizon_days: int = 20,
        min_train_dates: int = 252,
        test_window_dates: int = 20,
        recent_dates: int = 60,
        refresh_if_stale: bool = True,
        eligible_universe_mode: str = "passed_only",
        model_scope: str = "global",
        model_name: str | None = None,
        compare_scopes: bool = False,
    ) -> ShortlistAllocationAnalysisReport:
        self.db_manager.initialize()
        eligible_universe_mode = normalize_eligible_universe_mode(eligible_universe_mode)
        model_scope = normalize_model_scope(model_scope)
        scopes = ["global", "sector_specific"] if compare_scopes else [model_scope]

        policy_builders = [
            ("raw_top_n", lambda day: self._pick_raw_top_n(day, top_n=int(top_n))),
            ("sector_cap_3", lambda day: self._pick_sector_capped(day, top_n=int(top_n), cap=3)),
            ("sector_cap_2", lambda day: self._pick_sector_capped(day, top_n=int(top_n), cap=2)),
            ("sector_round_robin", lambda day: self._pick_sector_round_robin(day, top_n=int(top_n))),
        ]

        scope_results = [
            self._analyze_scope(
                scope=scope_name,
                top_n=int(top_n),
                horizon_days=int(horizon_days),
                min_train_dates=int(min_train_dates),
                test_window_dates=int(test_window_dates),
                recent_dates=int(recent_dates),
                refresh_if_stale=refresh_if_stale,
                eligible_universe_mode=eligible_universe_mode,
                model_name=model_name,
                policy_builders=policy_builders,
            )
            for scope_name in scopes
        ]

        if model_name is not None:
            selected_model_name = model_name
        elif len(scope_results) == 1:
            selected_model_name = scope_results[0].selected_model_name
        else:
            selected_model_name = ", ".join(f"{result.scope}={result.selected_model_name}" for result in scope_results)
        generated_at = ", ".join(f"{result.scope}={result.generated_at}" for result in scope_results)

        report_path = self.db_manager.paths.reports_dir / "shortlist_allocation_analysis.md"
        lines = [
            "# Shortlist Allocation Analysis",
            "",
            f"- generated_at: {generated_at}",
            f"- selected_model: {selected_model_name}",
            f"- horizon_days: {int(horizon_days)}",
            f"- top_n: {int(top_n)}",
            f"- eligible_universe_mode: {eligible_universe_mode}",
            f"- model_scope: {'compare_scopes' if compare_scopes else model_scope}",
            "",
        ]
        lines.extend(self._render_scope_summary(scope_results))
        if compare_scopes:
            recent_dates_label = max((result.recent_oos_date_count for result in scope_results), default=0)
            lines.extend(self._render_scope_comparison(scope_results, frame_kind="full", heading="## Full OOS Scope Comparison"))
            lines.extend(
                self._render_scope_comparison(
                    scope_results,
                    frame_kind="recent",
                    heading=f"## Recent {recent_dates_label} OOS Dates Scope Comparison",
                )
            )
        else:
            lines.extend(self._render_policy_table(scope_results[0].full_frame, heading="## Full OOS Allocation Comparison"))
            lines.extend(
                self._render_policy_table(
                    scope_results[0].recent_frame,
                    heading=f"## Recent {scope_results[0].recent_oos_date_count} OOS Dates",
                )
            )
        lines.extend(["## Live Allocation Shapes", ""])
        for result in scope_results:
            lines.extend(self._render_live_scope(result))
        report_path.write_text("\n".join(lines), encoding="utf-8")
        return ShortlistAllocationAnalysisReport(
            output_path=str(report_path),
            generated_at=generated_at,
            model_name=selected_model_name,
            policy_count=len(policy_builders),
            scope_count=len(scope_results),
        )

    def _analyze_scope(
        self,
        *,
        scope: str,
        top_n: int,
        horizon_days: int,
        min_train_dates: int,
        test_window_dates: int,
        recent_dates: int,
        refresh_if_stale: bool,
        eligible_universe_mode: str,
        model_name: str | None,
        policy_builders: list[tuple[str, Callable[[pd.DataFrame], pd.DataFrame]]],
    ) -> _ScopeAnalysis:
        latest_run = self._latest_or_refreshed_run(
            top_n=10,
            horizon_days=horizon_days,
            min_train_dates=min_train_dates,
            test_window_dates=test_window_dates,
            recent_dates=recent_dates,
            refresh_if_stale=refresh_if_stale,
            eligible_universe_mode=eligible_universe_mode,
            model_scope=scope,
        )
        generated_at = str(latest_run["generated_at"])
        champion_model = str(latest_run["champion_model"])
        selected_model_name = str(model_name or champion_model)

        oos_predictions = self.db_manager.load_shortlist_model_predictions(
            generated_at=generated_at,
            horizon_days=int(horizon_days),
            eligible_universe_mode=eligible_universe_mode,
            model_scope=scope,
            dataset_split="oos",
            model_name=selected_model_name,
        )
        live_predictions = self.db_manager.load_shortlist_model_predictions(
            generated_at=generated_at,
            horizon_days=int(horizon_days),
            eligible_universe_mode=eligible_universe_mode,
            model_scope=scope,
            dataset_split="live",
            model_name=selected_model_name,
        )
        if oos_predictions.empty:
            raise ValueError(
                f"Latest shortlist model run has no out-of-sample predictions for "
                f"model_name={selected_model_name} and model_scope={scope}."
            )
        oos_predictions["snapshot_date"] = pd.to_datetime(oos_predictions["snapshot_date"]).dt.normalize()
        oos_predictions["actual_alpha_vs_sector"] = pd.to_numeric(oos_predictions["actual_alpha_vs_sector"], errors="coerce")
        live_predictions["snapshot_date"] = pd.to_datetime(live_predictions["snapshot_date"]).dt.normalize()

        full_rows: list[dict[str, object]] = []
        recent_rows: list[dict[str, object]] = []
        recent_oos_dates = sorted(oos_predictions["snapshot_date"].drop_duplicates().tolist())[-max(int(recent_dates), 1):]
        latest_live_date = live_predictions["snapshot_date"].max() if not live_predictions.empty else None
        latest_live_day = live_predictions[live_predictions["snapshot_date"] == latest_live_date].copy() if latest_live_date is not None else pd.DataFrame()
        live_frames: dict[str, pd.DataFrame] = {}

        for policy_name, picker in policy_builders:
            full_rows.append(self._evaluate_policy(oos_predictions, picker=picker, policy_name=policy_name))
            recent_rows.append(
                self._evaluate_policy(
                    oos_predictions[oos_predictions["snapshot_date"].isin(recent_oos_dates)].copy(),
                    picker=picker,
                    policy_name=policy_name,
                )
            )
            live_frames[policy_name] = picker(latest_live_day) if not latest_live_day.empty else pd.DataFrame()

        return _ScopeAnalysis(
            scope=scope,
            generated_at=generated_at,
            champion_model=champion_model,
            selected_model_name=selected_model_name,
            full_frame=pd.DataFrame(full_rows),
            recent_frame=pd.DataFrame(recent_rows),
            live_frames=live_frames,
            recent_oos_date_count=len(recent_oos_dates),
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

    def _pick_raw_top_n(self, day_frame: pd.DataFrame, *, top_n: int) -> pd.DataFrame:
        return day_frame.sort_values(["predicted_alpha", "md_volume_30d", "ticker"], ascending=[False, False, True]).head(int(top_n)).copy()

    def _pick_sector_capped(self, day_frame: pd.DataFrame, *, top_n: int, cap: int) -> pd.DataFrame:
        ordered = day_frame.sort_values(["predicted_alpha", "md_volume_30d", "ticker"], ascending=[False, False, True]).copy()
        picks: list[dict[str, object]] = []
        sector_counts: dict[str, int] = {}
        for row in ordered.to_dict(orient="records"):
            sector = str(row.get("sector") or "Unknown")
            if sector_counts.get(sector, 0) >= int(cap):
                continue
            picks.append(row)
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
            if len(picks) >= int(top_n):
                break
        return pd.DataFrame(picks)

    def _pick_sector_round_robin(self, day_frame: pd.DataFrame, *, top_n: int) -> pd.DataFrame:
        if day_frame.empty:
            return day_frame.copy()
        grouped = {
            sector: group.sort_values(["predicted_alpha", "md_volume_30d", "ticker"], ascending=[False, False, True]).reset_index(drop=True)
            for sector, group in day_frame.groupby("sector", sort=False)
        }
        sector_order = sorted(
            grouped.keys(),
            key=lambda sector: (
                -float(grouped[sector].iloc[0]["predicted_alpha"]),
                sector,
            ),
        )
        pointers = {sector: 0 for sector in sector_order}
        picks: list[dict[str, object]] = []
        while len(picks) < int(top_n):
            progressed = False
            for sector in sector_order:
                pointer = pointers[sector]
                sector_frame = grouped[sector]
                if pointer >= len(sector_frame.index):
                    continue
                picks.append(sector_frame.iloc[pointer].to_dict())
                pointers[sector] = pointer + 1
                progressed = True
                if len(picks) >= int(top_n):
                    break
            if not progressed:
                break
        return pd.DataFrame(picks)

    def _evaluate_policy(self, predictions: pd.DataFrame, *, picker, policy_name: str) -> dict[str, object]:
        rows: list[dict[str, float]] = []
        sector_rows: list[dict[str, float]] = []
        for _, day_frame in predictions.groupby("snapshot_date", sort=True):
            picks = picker(day_frame)
            target = pd.to_numeric(picks["actual_alpha_vs_sector"], errors="coerce").dropna() if not picks.empty else pd.Series(dtype=float)
            universe_target = pd.to_numeric(day_frame["actual_alpha_vs_sector"], errors="coerce").dropna()
            if target.empty or universe_target.empty:
                continue
            sector_counts = picks["sector"].value_counts(normalize=True) if "sector" in picks.columns and not picks.empty else pd.Series(dtype=float)
            rows.append(
                {
                    "mean_target": float(target.mean()),
                    "hit_rate": float((target > 0.0).mean()),
                    "universe_mean_target": float(universe_target.mean()),
                    "max_sector_share": float(sector_counts.max()) if not sector_counts.empty else float("nan"),
                }
            )
            for sector, count in picks["sector"].value_counts().items():
                sector_rows.append({"sector": sector, "count": float(count)})
        if not rows:
            return self._empty_summary(policy_name)
        frame = pd.DataFrame(rows)
        sector_frame = pd.DataFrame(sector_rows)
        return {
            "policy": policy_name,
            "dates": len(frame.index),
            "mean_target": float(frame["mean_target"].mean()),
            "hit_rate": float(frame["hit_rate"].mean()),
            "beat_universe_rate": float((frame["mean_target"] > frame["universe_mean_target"]).mean()),
            "positive_date_rate": float((frame["mean_target"] > 0.0).mean()),
            "ge_2pct_rate": float((frame["mean_target"] >= 0.02).mean()),
            "ge_5pct_rate": float((frame["mean_target"] >= 0.05).mean()),
            "avg_max_sector_share": float(frame["max_sector_share"].mean()),
            "avg_sector_count": float(sector_frame.groupby("sector")["count"].mean().mean()) if not sector_frame.empty else float("nan"),
        }

    def _empty_summary(self, policy_name: str) -> dict[str, object]:
        return {
            "policy": policy_name,
            "dates": 0,
            "mean_target": float("nan"),
            "hit_rate": float("nan"),
            "beat_universe_rate": float("nan"),
            "positive_date_rate": float("nan"),
            "ge_2pct_rate": float("nan"),
            "ge_5pct_rate": float("nan"),
            "avg_max_sector_share": float("nan"),
            "avg_sector_count": float("nan"),
        }

    def _render_policy_table(self, frame: pd.DataFrame, *, heading: str) -> list[str]:
        lines = [heading, ""]
        ordered = frame.sort_values(
            ["mean_target", "beat_universe_rate", "policy"],
            ascending=[False, False, True],
        ).reset_index(drop=True)
        for row in ordered.itertuples(index=False):
            lines.append(f"### {row.policy}")
            lines.append(f"- dates: {int(row.dates)}")
            lines.append(f"- mean_target: {self._fmt(row.mean_target)}")
            lines.append(f"- hit_rate: {self._fmt(row.hit_rate)}")
            lines.append(f"- beat_universe_rate: {self._fmt(row.beat_universe_rate)}")
            lines.append(f"- positive_date_rate: {self._fmt(row.positive_date_rate)}")
            lines.append(f"- ge_2pct_rate: {self._fmt(row.ge_2pct_rate)}")
            lines.append(f"- ge_5pct_rate: {self._fmt(row.ge_5pct_rate)}")
            lines.append(f"- avg_max_sector_share: {self._fmt(row.avg_max_sector_share)}")
            lines.append("")
        return lines

    def _render_scope_summary(self, scope_results: list[_ScopeAnalysis]) -> list[str]:
        lines = ["## Scope Summary", ""]
        for result in scope_results:
            lines.append(f"### {result.scope}")
            lines.append(f"- generated_at: {result.generated_at}")
            lines.append(f"- run_champion_model: {result.champion_model}")
            lines.append(f"- selected_model: {result.selected_model_name}")
            lines.append("")
        return lines

    def _render_scope_comparison(
        self,
        scope_results: list[_ScopeAnalysis],
        *,
        frame_kind: str,
        heading: str,
    ) -> list[str]:
        lines = [heading, ""]
        if not scope_results:
            return lines
        policy_names = scope_results[0].full_frame["policy"].tolist()
        for policy_name in policy_names:
            lines.append(f"### {policy_name}")
            for result in scope_results:
                frame = result.full_frame if frame_kind == "full" else result.recent_frame
                match = frame[frame["policy"] == policy_name]
                if match.empty:
                    continue
                row = match.iloc[0]
                lines.append(
                    f"- {result.scope}: mean_target={self._fmt(row['mean_target'])}, "
                    f"beat_universe_rate={self._fmt(row['beat_universe_rate'])}, "
                    f"hit_rate={self._fmt(row['hit_rate'])}, "
                    f"avg_max_sector_share={self._fmt(row['avg_max_sector_share'])}"
                )
            lines.append("")
        return lines

    def _render_live_policy(self, *, policy_name: str, frame: pd.DataFrame) -> list[str]:
        lines = [f"#### {policy_name}", ""]
        if frame.empty:
            lines.append("No live candidates.")
            lines.append("")
            return lines
        for row in frame.itertuples(index=False):
            lines.append(f"- {row.ticker} ({row.sector}) predicted_alpha={self._fmt(row.predicted_alpha)}")
        lines.append("")
        return lines

    def _render_live_scope(self, result: _ScopeAnalysis) -> list[str]:
        lines = [
            f"### {result.scope}",
            f"- selected_model: {result.selected_model_name}",
            "",
        ]
        for policy_name, frame in result.live_frames.items():
            lines.extend(self._render_live_policy(policy_name=policy_name, frame=frame))
        return lines

    def _fmt(self, value: float) -> str:
        if value is None or not math.isfinite(float(value)):
            return "nan"
        return f"{float(value):.6f}"
