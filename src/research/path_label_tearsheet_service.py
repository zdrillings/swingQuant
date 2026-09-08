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
        fixed_column = f"alpha_vs_sector_{int(horizon_days)}d"
        path_column = f"path_alpha_vs_sector_{int(horizon_days)}d"
        exit_reason_column = f"path_exit_reason_{int(horizon_days)}d"
        frame = self.db_manager.load_universe_daily_snapshots()
        report_path = self.db_manager.paths.reports_dir / "path_label_tearsheet.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        if frame.empty or fixed_column not in frame.columns or path_column not in frame.columns:
            report_path.write_text(
                "\n".join(
                    [
                        "# Path Label Tearsheet",
                        "",
                        f"- horizon_days: {int(horizon_days)}",
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
        working[fixed_column] = pd.to_numeric(working[fixed_column], errors="coerce")
        working[path_column] = pd.to_numeric(working[path_column], errors="coerce")
        paired = working.dropna(subset=[fixed_column, path_column]).copy()
        lines = [
            "# Path Label Tearsheet",
            "",
            f"- horizon_days: {int(horizon_days)}",
            f"- rows: {len(working.index)}",
            f"- fixed_coverage_rows: {int(working[fixed_column].notna().sum())}",
            f"- path_coverage_rows: {int(working[path_column].notna().sum())}",
            f"- paired_rows: {len(paired.index)}",
            "",
            "## Label Comparison",
            "",
            "| label | mean | median | hit_rate | p10 | p90 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        lines.append(self._distribution_row("fixed_horizon", working[fixed_column]))
        lines.append(self._distribution_row("path_aware", working[path_column]))
        lines.extend(["", "## Paired Difference", ""])
        if paired.empty:
            lines.append("No rows have both labels populated.")
        else:
            diff = paired[path_column] - paired[fixed_column]
            corr = paired[fixed_column].corr(paired[path_column], method="spearman")
            lines.extend(
                [
                    f"- mean_path_minus_fixed: {self._fmt(diff.mean())}",
                    f"- median_path_minus_fixed: {self._fmt(diff.median())}",
                    f"- spearman_fixed_vs_path: {self._fmt(corr)}",
                    f"- path_better_rate: {self._fmt((diff > 0).mean())}",
                ]
            )
        lines.extend(["", "## Exit Reasons", ""])
        if exit_reason_column not in working.columns or working[exit_reason_column].dropna().empty:
            lines.append("No path exit reasons are populated.")
        else:
            counts = working[exit_reason_column].fillna("missing").astype(str).value_counts(dropna=False)
            total = int(counts.sum())
            lines.extend(["| exit_reason | rows | share |", "|---|---:|---:|"])
            for reason, count in counts.items():
                lines.append(f"| {reason} | {int(count)} | {self._fmt(float(count) / total if total else float('nan'))} |")
        lines.append("")
        report_path.write_text("\n".join(lines), encoding="utf-8")
        return PathLabelTearsheetReport(
            output_path=str(report_path),
            rows=len(working.index),
            paired_rows=len(paired.index),
        )

    def _distribution_row(self, label: str, values: pd.Series) -> str:
        numeric = pd.to_numeric(values, errors="coerce").dropna()
        if numeric.empty:
            return f"| {label} | n/a | n/a | n/a | n/a | n/a |"
        return (
            f"| {label} | {self._fmt(numeric.mean())} | {self._fmt(numeric.median())} | "
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
