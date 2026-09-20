from __future__ import annotations

from dataclasses import dataclass
import math

import pandas as pd

from src.utils.db_manager import DatabaseManager


@dataclass(frozen=True)
class PathLabelTearsheetReport:
    output_path: str
    rows: int
    paired_rows: int


class PathLabelTearsheetService:
    def __init__(self, db_manager: DatabaseManager) -> None:
        self.db_manager = db_manager

    def run(self, *, horizon_days: int = 20) -> PathLabelTearsheetReport:
        self.db_manager.initialize()
        requested_horizon = int(horizon_days)
        horizons = tuple(dict.fromkeys((20, requested_horizon, 60)))
        frame = self.db_manager.load_universe_daily_snapshots()
        report_path = self.db_manager.paths.reports_dir / "path_label_tearsheet.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        available_horizons = [
            horizon
            for horizon in horizons
            if f"alpha_vs_sector_{horizon}d" in frame.columns
            and f"path_alpha_vs_sector_{horizon}d" in frame.columns
        ]
        if frame.empty or not available_horizons:
            report_path.write_text(
                "\n".join(
                    [
                        "# Path Label Tearsheet",
                        "",
                        f"- horizon_days: {requested_horizon}",
                        "- rows: 0",
                        "- paired_rows: 0",
                        "",
                        "Path-aware labels are unavailable. Run `sq universe-backfill` after the path-label schema migration.",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            return PathLabelTearsheetReport(output_path=str(report_path), rows=0, paired_rows=0)

        working = frame.copy()
        working["snapshot_date"] = pd.to_datetime(working["snapshot_date"]).dt.normalize()
        primary_horizon = requested_horizon if requested_horizon in available_horizons else available_horizons[0]
        primary_fixed_column = f"alpha_vs_sector_{primary_horizon}d"
        primary_path_column = f"path_alpha_vs_sector_{primary_horizon}d"
        paired = working.dropna(subset=[primary_fixed_column, primary_path_column]).copy()
        lines = [
            "# Path Label Tearsheet",
            "",
            f"- horizon_days: {requested_horizon}",
            f"- rows: {len(working.index)}",
            f"- available_horizons: {', '.join(str(horizon) + 'd' for horizon in available_horizons)}",
            f"- paired_rows: {len(paired.index)}",
            "",
            "## Label Comparison",
            "",
            "| horizon | label | rows | mean | median | hit_rate | p10 | p90 |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        for horizon in available_horizons:
            fixed_column = f"alpha_vs_sector_{horizon}d"
            path_column = f"path_alpha_vs_sector_{horizon}d"
            working[fixed_column] = pd.to_numeric(working[fixed_column], errors="coerce")
            working[path_column] = pd.to_numeric(working[path_column], errors="coerce")
            lines.append(self._distribution_row(f"{horizon}d", "fixed_horizon", working[fixed_column]))
            lines.append(self._distribution_row(f"{horizon}d", "path_aware", working[path_column]))
        lines.extend(["", "## Paired Difference", ""])
        for horizon in available_horizons:
            fixed_column = f"alpha_vs_sector_{horizon}d"
            path_column = f"path_alpha_vs_sector_{horizon}d"
            horizon_paired = working.dropna(subset=[fixed_column, path_column]).copy()
            lines.append(f"### {horizon}d")
            if horizon_paired.empty:
                lines.append("No rows have both labels populated.")
                lines.append("")
                continue
            diff = horizon_paired[path_column] - horizon_paired[fixed_column]
            corr = horizon_paired[fixed_column].corr(horizon_paired[path_column], method="spearman")
            lines.extend(
                [
                    f"- fixed_{horizon}d_mean: {self._fmt(horizon_paired[fixed_column].mean())}",
                    f"- path_{horizon}d_mean: {self._fmt(horizon_paired[path_column].mean())}",
                    f"- mean_path_minus_fixed: {self._fmt(diff.mean())}",
                    f"- median_path_minus_fixed: {self._fmt(diff.median())}",
                    f"- spearman_fixed_vs_path: {self._fmt(corr)}",
                    f"- path_better_rate: {self._fmt((diff > 0).mean())}",
                ]
            )
            holding_column = f"path_holding_days_{horizon}d"
            if holding_column in horizon_paired.columns:
                holding_days = pd.to_numeric(horizon_paired[holding_column], errors="coerce")
                lines.append(f"- mean_holding_days: {self._fmt(holding_days.mean())}")
            lines.append("")
        lines.extend(["", "## Exit Reasons", ""])
        for horizon in available_horizons:
            exit_reason_column = f"path_exit_reason_{horizon}d"
            lines.append(f"### {horizon}d")
            if exit_reason_column not in working.columns or working[exit_reason_column].dropna().empty:
                lines.append("No path exit reasons are populated.")
                lines.append("")
                continue
            counts = working[exit_reason_column].fillna("missing").astype(str).value_counts(dropna=False)
            total = int(counts.sum())
            lines.extend(["| exit_reason | rows | share |", "|---|---:|---:|"])
            for reason, count in counts.items():
                lines.append(f"| {reason} | {int(count)} | {self._fmt(float(count) / total if total else float('nan'))} |")
            lines.append("")
        lines.append("")
        report_path.write_text("\n".join(lines), encoding="utf-8")
        return PathLabelTearsheetReport(
            output_path=str(report_path),
            rows=len(working.index),
            paired_rows=len(paired.index),
        )

    def _distribution_row(self, horizon_label: str, label: str, values: pd.Series) -> str:
        numeric = pd.to_numeric(values, errors="coerce").dropna()
        if numeric.empty:
            return f"| {horizon_label} | {label} | 0 | n/a | n/a | n/a | n/a | n/a |"
        return (
            f"| {horizon_label} | {label} | {len(numeric.index)} | {self._fmt(numeric.mean())} | {self._fmt(numeric.median())} | "
            f"{self._fmt((numeric > 0).mean())} | {self._fmt(numeric.quantile(0.10))} | "
            f"{self._fmt(numeric.quantile(0.90))} |"
        )

    def _fmt(self, value: object) -> str:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return "n/a"
        if not math.isfinite(numeric):
            return "n/a"
        return f"{numeric:.6f}"
