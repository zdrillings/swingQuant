from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import math

import pandas as pd

from src.research.shortlist_model_service import ShortlistModelService
from src.settings import load_feature_config
from src.utils.db_manager import DatabaseManager
from src.utils.strategy import load_active_strategies


@dataclass(frozen=True)
class ShortlistSectorReactivationReport:
    output_path: str
    generated_at: str
    run_generated_at: str
    selected_model_name: str
    active_sector_count: int
    candidate_sector_count: int


class ShortlistSectorReactivationService:
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
        eligible_universe_mode: str | None = None,
        model_scope: str | None = None,
        model_name: str | None = None,
        xgboost_config: str | None = None,
        candidate_sectors: tuple[str, ...] = ("Information Technology",),
    ) -> ShortlistSectorReactivationReport:
        self.db_manager.initialize()
        resolved = self._resolve_shortlist_config(
            eligible_universe_mode=eligible_universe_mode,
            model_scope=model_scope,
            model_name=model_name,
            xgboost_config=xgboost_config,
        )
        active_sectors = self._active_strategy_sectors()
        candidate_sector_list = [sector for sector in candidate_sectors if sector]
        baseline_sectors = tuple(active_sectors)
        expanded_sectors = tuple(dict.fromkeys([*active_sectors, *candidate_sector_list]))
        candidate_only_sectors = tuple(candidate_sector_list)

        latest_run = self._latest_or_refreshed_run(
            top_n=10,
            horizon_days=int(horizon_days),
            min_train_dates=int(min_train_dates),
            test_window_dates=int(test_window_dates),
            recent_dates=int(recent_dates),
            refresh_if_stale=refresh_if_stale,
            eligible_universe_mode=resolved["eligible_universe_mode"],
            model_scope=resolved["model_scope"],
            xgboost_config=resolved["xgboost_config"],
        )
        run_generated_at = str(latest_run["generated_at"])
        selected_model_name = str(model_name or resolved["model_name"] or latest_run["champion_model"])

        oos_predictions = self.db_manager.load_shortlist_model_predictions(
            generated_at=run_generated_at,
            horizon_days=int(horizon_days),
            eligible_universe_mode=resolved["eligible_universe_mode"],
            model_scope=resolved["model_scope"],
            dataset_split="oos",
            model_name=selected_model_name,
        )
        live_predictions = self.db_manager.load_shortlist_model_predictions(
            generated_at=run_generated_at,
            horizon_days=int(horizon_days),
            eligible_universe_mode=resolved["eligible_universe_mode"],
            model_scope=resolved["model_scope"],
            dataset_split="live",
            model_name=selected_model_name,
        )
        if oos_predictions.empty:
            raise ValueError("No out-of-sample predictions found for shortlist sector reactivation analysis.")
        oos_predictions["snapshot_date"] = pd.to_datetime(oos_predictions["snapshot_date"]).dt.normalize()
        oos_predictions["actual_alpha_vs_sector"] = pd.to_numeric(oos_predictions["actual_alpha_vs_sector"], errors="coerce")
        live_predictions["snapshot_date"] = pd.to_datetime(live_predictions["snapshot_date"]).dt.normalize()

        recent_prediction_dates = sorted(oos_predictions["snapshot_date"].drop_duplicates().tolist())[-max(int(recent_dates), 1):]

        scenario_sets = [
            ("baseline_active", baseline_sectors),
            ("baseline_plus_candidate", expanded_sectors),
            ("candidate_only", candidate_only_sectors),
        ]
        scenario_policy_builders = [
            ("raw_top_n", lambda day, top_n=int(top_n): self._pick_raw_top_n(day, top_n=top_n)),
            ("sector_cap_3", lambda day, top_n=int(top_n): self._pick_sector_capped(day, top_n=top_n, cap=3)),
        ]
        expanded_policy_builders = [
            ("raw_top_n", lambda day, top_n=int(top_n): self._pick_raw_top_n(day, top_n=top_n)),
            ("sector_cap_3", lambda day, top_n=int(top_n): self._pick_sector_capped(day, top_n=top_n, cap=3)),
            ("sector_cap_2", lambda day, top_n=int(top_n): self._pick_sector_capped(day, top_n=top_n, cap=2)),
            ("sector_round_robin", lambda day, top_n=int(top_n): self._pick_sector_round_robin(day, top_n=top_n)),
        ]
        top_n_variants = [6, 8, 10]

        step1_rows: list[dict[str, object]] = []
        for policy_name, picker in scenario_policy_builders:
            for scenario_name, sectors in scenario_sets:
                filtered_oos = self._filter_predictions_to_sectors(oos_predictions, sectors=sectors)
                filtered_recent = filtered_oos[filtered_oos["snapshot_date"].isin(recent_prediction_dates)].copy()
                filtered_live = self._latest_live_day(self._filter_predictions_to_sectors(live_predictions, sectors=sectors))
                summary = self._evaluate_policy(filtered_oos, picker=picker, policy_name=policy_name)
                recent_summary = self._evaluate_policy(filtered_recent, picker=picker, policy_name=policy_name)
                live_picks = picker(filtered_live) if not filtered_live.empty else pd.DataFrame()
                step1_rows.append(
                    {
                        "policy": policy_name,
                        "scenario": scenario_name,
                        "sectors": sectors,
                        "full_mean_target": summary["mean_target"],
                        "full_beat_universe_rate": summary["beat_universe_rate"],
                        "full_hit_rate": summary["hit_rate"],
                        "recent_mean_target": recent_summary["mean_target"],
                        "recent_beat_universe_rate": recent_summary["beat_universe_rate"],
                        "recent_hit_rate": recent_summary["hit_rate"],
                        "live_max_sector_share": self._live_max_sector_share(live_picks),
                        "live_sector_mix": self._format_sector_mix(live_picks),
                        "live_picks": self._format_live_picks(live_picks),
                    }
                )

        expanded_oos = self._filter_predictions_to_sectors(oos_predictions, sectors=expanded_sectors)
        expanded_recent = expanded_oos[expanded_oos["snapshot_date"].isin(recent_prediction_dates)].copy()
        expanded_live = self._latest_live_day(self._filter_predictions_to_sectors(live_predictions, sectors=expanded_sectors))

        step2_rows: list[dict[str, object]] = []
        for policy_name, picker in expanded_policy_builders:
            full_summary = self._evaluate_policy(expanded_oos, picker=picker, policy_name=policy_name)
            recent_summary = self._evaluate_policy(expanded_recent, picker=picker, policy_name=policy_name)
            live_picks = picker(expanded_live) if not expanded_live.empty else pd.DataFrame()
            step2_rows.append(
                {
                    "policy": policy_name,
                    "full_mean_target": full_summary["mean_target"],
                    "full_beat_universe_rate": full_summary["beat_universe_rate"],
                    "full_hit_rate": full_summary["hit_rate"],
                    "recent_mean_target": recent_summary["mean_target"],
                    "recent_beat_universe_rate": recent_summary["beat_universe_rate"],
                    "recent_hit_rate": recent_summary["hit_rate"],
                    "live_max_sector_share": self._live_max_sector_share(live_picks),
                    "live_sector_mix": self._format_sector_mix(live_picks),
                    "live_picks": self._format_live_picks(live_picks),
                }
            )

        step3_rows: list[dict[str, object]] = []
        for variant_top_n in top_n_variants:
            picker = lambda day, top_n=variant_top_n: self._pick_sector_capped(day, top_n=top_n, cap=3)
            full_summary = self._evaluate_policy(expanded_oos, picker=picker, policy_name=f"top_{variant_top_n}")
            recent_summary = self._evaluate_policy(expanded_recent, picker=picker, policy_name=f"top_{variant_top_n}")
            live_picks = picker(expanded_live) if not expanded_live.empty else pd.DataFrame()
            step3_rows.append(
                {
                    "top_n": variant_top_n,
                    "full_mean_target": full_summary["mean_target"],
                    "full_beat_universe_rate": full_summary["beat_universe_rate"],
                    "full_hit_rate": full_summary["hit_rate"],
                    "recent_mean_target": recent_summary["mean_target"],
                    "recent_beat_universe_rate": recent_summary["beat_universe_rate"],
                    "recent_hit_rate": recent_summary["hit_rate"],
                    "live_max_sector_share": self._live_max_sector_share(live_picks),
                    "live_sector_mix": self._format_sector_mix(live_picks),
                    "live_picks": self._format_live_picks(live_picks),
                }
            )

        generated_at = datetime.now(UTC).replace(microsecond=0).isoformat()
        report_path = self.db_manager.paths.reports_dir / "shortlist_sector_reactivation.md"
        lines = [
            "# Shortlist Sector Reactivation Analysis",
            "",
            f"- generated_at: {generated_at}",
            f"- run_generated_at: {run_generated_at}",
            f"- selected_model_name: {selected_model_name}",
            f"- eligible_universe_mode: {resolved['eligible_universe_mode']}",
            f"- model_scope: {resolved['model_scope']}",
            f"- xgboost_config: {resolved['xgboost_config']}",
            f"- active_sectors: {', '.join(baseline_sectors) if baseline_sectors else 'none'}",
            f"- candidate_sectors: {', '.join(candidate_only_sectors) if candidate_only_sectors else 'none'}",
            f"- recent_oos_dates: {len(recent_prediction_dates)}",
            "",
            "## Step 1: Sector Admission Comparison",
            "",
            "Compare today's active-sector shortlist set against the same model with the candidate sector admitted.",
            "",
        ]
        lines.extend(self._render_step1(step1_rows))
        lines.extend(
            [
                "## Step 2: Expanded Set Allocation Balance",
                "",
                "Within the expanded sector set, compare diversification policies at the current shortlist size.",
                "",
            ]
        )
        lines.extend(self._render_step2(step2_rows, top_n=int(top_n)))
        lines.extend(
            [
                "## Step 3: Expanded Set Top-N Sensitivity",
                "",
                "Hold the standard sector cap policy constant and test whether widening beyond 6 names improves balance.",
                "",
            ]
        )
        lines.extend(self._render_step3(step3_rows))
        report_path.write_text("\n".join(lines), encoding="utf-8")
        return ShortlistSectorReactivationReport(
            output_path=str(report_path),
            generated_at=generated_at,
            run_generated_at=run_generated_at,
            selected_model_name=selected_model_name,
            active_sector_count=len(baseline_sectors),
            candidate_sector_count=len(candidate_only_sectors),
        )

    def _resolve_shortlist_config(
        self,
        *,
        eligible_universe_mode: str | None,
        model_scope: str | None,
        model_name: str | None,
        xgboost_config: str | None,
    ) -> dict[str, str]:
        config = load_feature_config()
        shortlist_model = config.get("scan_policy", {}).get("shortlist_model", {}) if isinstance(config, dict) else {}
        return {
            "eligible_universe_mode": str(
                eligible_universe_mode
                or shortlist_model.get("production_eligible_universe_mode")
                or shortlist_model.get("eligible_universe_mode")
                or "passed_only"
            ),
            "model_scope": str(
                model_scope
                or shortlist_model.get("production_model_scope")
                or "global"
            ),
            "model_name": str(
                model_name
                or shortlist_model.get("production_model_name")
                or "xgboost_model"
            ),
            "xgboost_config": str(
                xgboost_config
                or shortlist_model.get("production_xgboost_config")
                or "baseline"
            ),
        }

    def _active_strategy_sectors(self) -> list[str]:
        strategies = load_active_strategies()
        sectors = []
        for strategy in strategies.values():
            sector = str(strategy.sector)
            if sector in ("", "ALL"):
                continue
            sectors.append(sector)
        return sorted(set(sectors))

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
        xgboost_config: str,
    ) -> pd.Series:
        runs = self.db_manager.load_shortlist_model_runs(
            horizon_days=int(horizon_days),
            eligible_universe_mode=eligible_universe_mode,
            model_scope=model_scope,
            xgboost_config=xgboost_config,
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
                xgboost_config=xgboost_config,
            )
            runs = self.db_manager.load_shortlist_model_runs(
                horizon_days=int(horizon_days),
                eligible_universe_mode=eligible_universe_mode,
                model_scope=model_scope,
                xgboost_config=xgboost_config,
                limit=1,
            )
            if runs.empty:
                raise ValueError("Unable to create shortlist model run for sector reactivation analysis.")
        return runs.iloc[0]

    def _filter_predictions_to_sectors(self, predictions: pd.DataFrame, *, sectors: tuple[str, ...]) -> pd.DataFrame:
        if predictions.empty:
            return predictions.copy()
        if not sectors:
            return predictions.iloc[0:0].copy()
        return predictions[predictions["sector"].astype(str).isin(set(sectors))].copy()

    def _latest_live_day(self, predictions: pd.DataFrame) -> pd.DataFrame:
        if predictions.empty:
            return predictions.copy()
        latest_date = predictions["snapshot_date"].max()
        return predictions[predictions["snapshot_date"] == latest_date].copy()

    def _pick_raw_top_n(self, day_frame: pd.DataFrame, *, top_n: int) -> pd.DataFrame:
        return day_frame.sort_values(
            ["predicted_alpha", "md_volume_30d", "ticker"],
            ascending=[False, False, True],
        ).head(int(top_n)).copy()

    def _pick_sector_capped(self, day_frame: pd.DataFrame, *, top_n: int, cap: int) -> pd.DataFrame:
        ordered = day_frame.sort_values(
            ["predicted_alpha", "md_volume_30d", "ticker"],
            ascending=[False, False, True],
        ).copy()
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
            sector: group.sort_values(
                ["predicted_alpha", "md_volume_30d", "ticker"],
                ascending=[False, False, True],
            ).reset_index(drop=True)
            for sector, group in day_frame.groupby("sector", sort=False)
        }
        sector_order = sorted(
            grouped.keys(),
            key=lambda sector: (-float(grouped[sector].iloc[0]["predicted_alpha"]), sector),
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
        for _, day_frame in predictions.groupby("snapshot_date", sort=True):
            picks = picker(day_frame)
            target = pd.to_numeric(picks["actual_alpha_vs_sector"], errors="coerce").dropna() if not picks.empty else pd.Series(dtype=float)
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
                "policy": policy_name,
                "mean_target": float("nan"),
                "hit_rate": float("nan"),
                "beat_universe_rate": float("nan"),
            }
        frame = pd.DataFrame(rows)
        return {
            "policy": policy_name,
            "mean_target": float(frame["mean_target"].mean()),
            "hit_rate": float(frame["hit_rate"].mean()),
            "beat_universe_rate": float((frame["mean_target"] > frame["universe_mean_target"]).mean()),
        }

    def _live_max_sector_share(self, frame: pd.DataFrame) -> float:
        if frame.empty or "sector" not in frame.columns:
            return float("nan")
        weights = frame["sector"].value_counts(normalize=True)
        if weights.empty:
            return float("nan")
        return float(weights.max())

    def _format_sector_mix(self, frame: pd.DataFrame) -> str:
        if frame.empty or "sector" not in frame.columns:
            return "none"
        weights = frame["sector"].value_counts(normalize=True)
        return ", ".join(f"{sector} {weight:.0%}" for sector, weight in weights.items())

    def _format_live_picks(self, frame: pd.DataFrame) -> str:
        if frame.empty:
            return "none"
        ordered = frame.sort_values(
            ["predicted_alpha", "md_volume_30d", "ticker"],
            ascending=[False, False, True],
        )
        return ", ".join(f"{row.ticker} ({row.sector})" for row in ordered.itertuples(index=False))

    def _render_step1(self, rows: list[dict[str, object]]) -> list[str]:
        lines: list[str] = []
        for policy_name in sorted({str(row["policy"]) for row in rows}):
            lines.append(f"### {policy_name}")
            for row in [entry for entry in rows if str(entry["policy"]) == policy_name]:
                lines.append(
                    f"- {row['scenario']}: "
                    f"full_mean_target={self._fmt(row['full_mean_target'])}, "
                    f"full_beat_universe={self._fmt(row['full_beat_universe_rate'])}, "
                    f"recent_mean_target={self._fmt(row['recent_mean_target'])}, "
                    f"recent_beat_universe={self._fmt(row['recent_beat_universe_rate'])}, "
                    f"live_max_sector_share={self._fmt(row['live_max_sector_share'])}"
                )
                lines.append(f"  sectors: {', '.join(row['sectors']) if row['sectors'] else 'none'}")
                lines.append(f"  live_sector_mix: {row['live_sector_mix']}")
                lines.append(f"  live_picks: {row['live_picks']}")
            lines.append("")
        return lines

    def _render_step2(self, rows: list[dict[str, object]], *, top_n: int) -> list[str]:
        lines = [f"- top_n: {int(top_n)}", ""]
        for row in rows:
            lines.append(f"### {row['policy']}")
            lines.append(
                f"- full_mean_target: {self._fmt(row['full_mean_target'])}"
            )
            lines.append(
                f"- full_beat_universe: {self._fmt(row['full_beat_universe_rate'])}"
            )
            lines.append(f"- recent_mean_target: {self._fmt(row['recent_mean_target'])}")
            lines.append(
                f"- recent_beat_universe: {self._fmt(row['recent_beat_universe_rate'])}"
            )
            lines.append(f"- live_max_sector_share: {self._fmt(row['live_max_sector_share'])}")
            lines.append(f"- live_sector_mix: {row['live_sector_mix']}")
            lines.append(f"- live_picks: {row['live_picks']}")
            lines.append("")
        return lines

    def _render_step3(self, rows: list[dict[str, object]]) -> list[str]:
        lines: list[str] = []
        for row in rows:
            lines.append(f"### top_n={int(row['top_n'])}")
            lines.append(f"- full_mean_target: {self._fmt(row['full_mean_target'])}")
            lines.append(
                f"- full_beat_universe: {self._fmt(row['full_beat_universe_rate'])}"
            )
            lines.append(f"- recent_mean_target: {self._fmt(row['recent_mean_target'])}")
            lines.append(
                f"- recent_beat_universe: {self._fmt(row['recent_beat_universe_rate'])}"
            )
            lines.append(f"- live_max_sector_share: {self._fmt(row['live_max_sector_share'])}")
            lines.append(f"- live_sector_mix: {row['live_sector_mix']}")
            lines.append(f"- live_picks: {row['live_picks']}")
            lines.append("")
        return lines

    def _fmt(self, value: float) -> str:
        if value is None or not math.isfinite(float(value)):
            return "nan"
        return f"{float(value):.6f}"
