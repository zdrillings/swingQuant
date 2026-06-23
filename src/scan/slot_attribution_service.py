from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import numpy as np
import pandas as pd

from src.scan.ranker import CandidateRanker
from src.scan.service import ScanPolicy
from src.settings import load_feature_config
from src.utils.db_manager import DatabaseManager
from src.utils.logging import get_logger
from src.utils.strategy import load_active_strategies


@dataclass(frozen=True)
class SlotAttributionReport:
    output_path: str
    target_column: str
    slot_count: int
    scan_dates: int


class SlotAttributionService:
    def __init__(self, db_manager: DatabaseManager) -> None:
        self.db_manager = db_manager
        self.logger = get_logger("slot_attribution")

    def run(self, *, horizon_days: int = 10) -> SlotAttributionReport:
        self.db_manager.initialize()
        active_strategies = load_active_strategies()
        active_slots = tuple(sorted(active_strategies))
        if not active_slots:
            raise ValueError("No active strategies found.")
        target_column = f"alpha_vs_sector_{int(horizon_days)}d"
        candidates = self.db_manager.load_scan_candidates()
        if candidates.empty:
            raise ValueError("No scan snapshots found. Run `sq scan-backfill` or `sq scan` first.")
        if target_column not in candidates.columns:
            raise ValueError(f"Scan snapshots do not include horizon_days={horizon_days}.")

        working = candidates.copy()
        working = working[working["strategy_slot"].astype(str).isin(active_slots)].copy()
        working = working.dropna(subset=[target_column]).copy()
        if working.empty:
            raise ValueError(f"No labeled scan candidate rows found for horizon_days={horizon_days}.")
        working["scan_date"] = pd.to_datetime(working["scan_date"]).dt.normalize()
        config = load_feature_config()
        scan_policy = ScanPolicy.from_config(config)
        ranker = CandidateRanker(
            target_column=target_column,
            min_train_rows=scan_policy.learned_ranker_min_train_rows,
            min_train_dates=scan_policy.learned_ranker_min_train_dates,
        )
        embargo_days = ranker._infer_target_horizon_days(default=int(horizon_days))
        folds = ranker._purged_walk_forward_folds(
            working,
            train_ratio=0.7,
            embargo_days=embargo_days,
            max_validation_blocks=5,
        )
        scored_frames: list[pd.DataFrame] = []
        fold_train_rows: list[int] = []
        fold_train_dates: list[int] = []
        for fold in folds:
            train_frame = fold["train"].copy()
            validation_frame = fold["validation"].copy()
            train_dates = int(train_frame["scan_date"].nunique()) if not train_frame.empty else 0
            if (
                train_frame.empty
                or validation_frame.empty
                or len(train_frame.index) < ranker.min_train_rows
                or train_dates < ranker.min_train_dates
            ):
                continue
            ranker.fit(train_frame)
            scored = ranker.score(validation_frame)
            scored["validation_fold"] = int(fold["fold_index"])
            scored_frames.append(scored)
            fold_train_rows.append(len(train_frame.index))
            fold_train_dates.append(train_dates)
        if not scored_frames:
            raise ValueError("Not enough purged walk-forward validation history for slot attribution.")
        scored = pd.concat(scored_frames, ignore_index=True)

        rows: list[dict[str, object]] = []
        for (scan_date, slot), day_frame in scored.groupby(["scan_date", "strategy_slot"], sort=True):
            runtime = day_frame[day_frame["selected"].astype(int) == 1].copy()
            runtime_count = len(runtime.index)
            pick_count = runtime_count if runtime_count > 0 else min(len(day_frame.index), max(scan_policy.max_candidates_per_slot, 1))
            if pick_count <= 0:
                continue
            learned = self._select_top_n(day_frame, score_column="ranker_score", top_n=pick_count)
            handcrafted = self._select_top_n(day_frame, score_column="opportunity_score", top_n=pick_count)
            random_selected = self._stable_random_select(day_frame, top_n=pick_count)
            rows.extend(
                [
                    self._metric_row(scan_date, slot, "eligible", day_frame, target_column),
                    self._metric_row(scan_date, slot, "selected", runtime, target_column),
                    self._metric_row(scan_date, slot, "learned_ranker", learned, target_column),
                    self._metric_row(scan_date, slot, "handcrafted", handcrafted, target_column),
                    self._metric_row(scan_date, slot, "random_eligible", random_selected, target_column),
                ]
            )
        metrics = pd.DataFrame(rows)
        summary_rows: list[dict[str, object]] = []
        for (slot, method), subset in metrics.groupby(["strategy_slot", "method"], sort=True):
            mean_target = pd.to_numeric(subset["mean_target"], errors="coerce").dropna()
            hit_rate = pd.to_numeric(subset["hit_rate"], errors="coerce").dropna()
            pick_count = pd.to_numeric(subset["pick_count"], errors="coerce").dropna()
            summary_rows.append(
                {
                    "strategy_slot": str(slot),
                    "method": str(method),
                    "mean_target": float(mean_target.mean()) if not mean_target.empty else float("nan"),
                    "hit_rate": float(hit_rate.mean()) if not hit_rate.empty else float("nan"),
                    "avg_pick_count": float(pick_count.mean()) if not pick_count.empty else float("nan"),
                    "validation_days": int(subset["scan_date"].nunique()),
                }
            )
        summary = pd.DataFrame(summary_rows)

        report_path = self.db_manager.paths.reports_dir / "slot_attribution.md"
        report_path.write_text(
            "\n".join(
                self._render_report(
                    target_column=target_column,
                    embargo_days=embargo_days,
                    validation_blocks=len(scored["validation_fold"].drop_duplicates()) if "validation_fold" in scored.columns else len(folds),
                    train_rows=int(round(float(np.mean(fold_train_rows)))) if fold_train_rows else 0,
                    train_dates=int(round(float(np.mean(fold_train_dates)))) if fold_train_dates else 0,
                    scan_dates=int(scored["scan_date"].nunique()),
                    summary=summary,
                    active_slots=active_slots,
                )
            ),
            encoding="utf-8",
        )
        return SlotAttributionReport(
            output_path=str(report_path),
            target_column=target_column,
            slot_count=len(active_slots),
            scan_dates=int(scored["scan_date"].nunique()),
        )

    def _select_top_n(self, frame: pd.DataFrame, *, score_column: str, top_n: int) -> pd.DataFrame:
        if frame.empty or top_n <= 0:
            return frame.iloc[0:0].copy()
        ranked = frame.copy()
        ranked["_score"] = pd.to_numeric(ranked[score_column], errors="coerce")
        ranked = ranked.dropna(subset=["_score"]).copy()
        if ranked.empty:
            return ranked
        ranked = ranked.sort_values(
            ["_score", "opportunity_score", "signal_score", "md_volume_30d", "ticker"],
            ascending=[False, False, False, False, True],
        )
        return ranked.head(int(top_n)).copy()

    def _stable_random_select(self, frame: pd.DataFrame, *, top_n: int) -> pd.DataFrame:
        if frame.empty or top_n <= 0:
            return frame.iloc[0:0].copy()
        ranked = frame.copy()
        ranked["_random_key"] = ranked.apply(
            lambda row: self._stable_hash(f"{pd.Timestamp(row['scan_date']).date()}|{row['strategy_slot']}|{row['ticker']}"),
            axis=1,
        )
        ranked = ranked.sort_values(["_random_key", "ticker"], ascending=[True, True])
        return ranked.head(int(top_n)).copy()

    def _stable_hash(self, value: str) -> float:
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
        integer = int(digest[:16], 16)
        return float(integer)

    def _metric_row(self, scan_date, slot: str, method: str, frame: pd.DataFrame, target_column: str) -> dict[str, object]:
        target = pd.to_numeric(frame[target_column], errors="coerce").dropna()
        return {
            "scan_date": pd.Timestamp(scan_date).date().isoformat(),
            "strategy_slot": str(slot),
            "method": str(method),
            "pick_count": int(len(frame.index)),
            "mean_target": float(target.mean()) if not target.empty else float("nan"),
            "hit_rate": float((target > 0.0).mean()) if not target.empty else float("nan"),
        }

    def _render_report(
        self,
        *,
        target_column: str,
        embargo_days: int,
        validation_blocks: int,
        train_rows: int,
        train_dates: int,
        scan_dates: int,
        summary: pd.DataFrame,
        active_slots: tuple[str, ...],
    ) -> list[str]:
        lines = [
            "# Slot Attribution",
            "",
            f"- target_column: {target_column}",
            "- validation_method: purged_walk_forward",
            f"- embargo_days: {embargo_days}",
            f"- validation_blocks: {validation_blocks}",
            f"- train_rows: {train_rows}",
            f"- train_dates: {train_dates}",
            f"- validation_dates: {scan_dates}",
            "",
        ]
        method_labels = {
            "eligible": "eligible_candidate_alpha",
            "selected": "selected_candidate_alpha",
            "learned_ranker": "learned_ranker_candidate_alpha",
            "handcrafted": "handcrafted_candidate_alpha",
            "random_eligible": "random_eligible_alpha",
        }
        for slot in active_slots:
            lines.append(f"## {slot}")
            lines.append("")
            slot_summary = summary[summary["strategy_slot"].astype(str) == str(slot)].copy()
            if slot_summary.empty:
                lines.append("No attribution rows available.")
                lines.append("")
                continue
            for method in ("eligible", "selected", "learned_ranker", "handcrafted", "random_eligible"):
                subset = slot_summary[slot_summary["method"].astype(str) == method].copy()
                if subset.empty:
                    lines.append(f"- {method_labels[method]}: unavailable")
                    continue
                row = subset.iloc[0]
                lines.append(
                    f"- {method_labels[method]}: mean_target={self._format_float(float(row['mean_target']))} "
                    f"hit_rate={self._format_float(float(row['hit_rate']), digits=4)} "
                    f"avg_pick_count={self._format_float(float(row['avg_pick_count']), digits=2)} "
                    f"validation_days={int(row['validation_days'])}"
                )
            lines.append("")
        return lines

    def _format_float(self, value: float, *, digits: int = 6) -> str:
        if not math.isfinite(float(value)):
            return "nan"
        return f"{float(value):.{digits}f}"
