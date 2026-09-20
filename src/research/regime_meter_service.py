from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
from pathlib import Path

import pandas as pd

from src.settings import load_feature_config
from src.utils.db_manager import DatabaseManager


@dataclass(frozen=True)
class RegimeMeterReport:
    output_path: str | None
    rows_written: int
    latest_matured_date: str | None
    classification: str | None
    mom_ic_20d_avg: float | None


class RegimeMeterService:
    def __init__(self, db_manager: DatabaseManager) -> None:
        self.db_manager = db_manager

    def run(self, *, backfill: bool = False, latest: bool = False, report: bool = False) -> RegimeMeterReport:
        self.db_manager.initialize()
        if not backfill and not latest and not report:
            latest = True
            report = True
        rows_written = 0
        if backfill or latest:
            rows = self._build_rows()
            if not rows.empty:
                write_rows = rows if backfill else rows.tail(60)
                rows_written = self.db_manager.replace_regime_meter_rows(
                    write_rows.to_dict(orient="records")
                )
        output_path: Path | None = None
        latest_row = self._latest_known_row()
        if report:
            output_path = self._write_report()
            latest_row = self._latest_known_row()
        latest_date = None
        if latest_row is not None:
            latest_date = pd.to_datetime(latest_row["snapshot_date"]).date().isoformat()
        return RegimeMeterReport(
            output_path=str(output_path) if output_path is not None else None,
            rows_written=rows_written,
            latest_matured_date=latest_date,
            classification=(
                str(latest_row["classification"]) if latest_row is not None else None
            ),
            mom_ic_20d_avg=(
                float(latest_row["mom_ic_20d_avg"])
                if latest_row is not None and pd.notna(latest_row.get("mom_ic_20d_avg"))
                else None
            ),
        )

    def _build_rows(self) -> pd.DataFrame:
        thresholds = self._thresholds()
        daily = self._load_daily_metrics()
        if daily.empty:
            return daily
        daily = daily.sort_values("snapshot_date").reset_index(drop=True)
        daily["mom_ic_20d_avg"] = daily["mom_ic_daily"].rolling(window=20, min_periods=10).mean()
        daily["mom_ic_60d_avg"] = daily["mom_ic_daily"].rolling(window=60, min_periods=20).mean()
        daily["classification"] = daily["mom_ic_20d_avg"].apply(
            lambda value: self.classify(
                value,
                reversal_ic_threshold=thresholds["reversal_ic_threshold"],
                trending_ic_threshold=thresholds["trending_ic_threshold"],
            )
        )
        return daily

    def _load_daily_metrics(self) -> pd.DataFrame:
        query = """
            WITH base AS (
                SELECT
                    snapshot_date,
                    ticker,
                    roc_126,
                    alpha_vs_sector_20d,
                    ret_1d,
                    RANK() OVER (
                        PARTITION BY snapshot_date
                        ORDER BY roc_126
                    ) + (COUNT(*) OVER (
                        PARTITION BY snapshot_date, roc_126
                    ) - 1) / 2.0 AS roc_rank,
                    RANK() OVER (
                        PARTITION BY snapshot_date
                        ORDER BY alpha_vs_sector_20d
                    ) + (COUNT(*) OVER (
                        PARTITION BY snapshot_date, alpha_vs_sector_20d
                    ) - 1) / 2.0 AS alpha_rank,
                    NTILE(5) OVER (
                        PARTITION BY snapshot_date
                        ORDER BY roc_126
                    ) AS momentum_quintile
                FROM universe_daily_snapshots
                WHERE alpha_vs_sector_20d IS NOT NULL
                  AND roc_126 IS NOT NULL
            )
            SELECT
                snapshot_date,
                corr(roc_rank, alpha_rank) AS mom_ic_daily,
                avg(CASE WHEN momentum_quintile = 5 THEN alpha_vs_sector_20d END)
                    - avg(CASE WHEN momentum_quintile = 1 THEN alpha_vs_sector_20d END) AS wml_20d_alpha,
                avg(CASE WHEN momentum_quintile = 5 THEN ret_1d END)
                    - avg(CASE WHEN momentum_quintile = 1 THEN ret_1d END) AS wml_1d
            FROM base
            GROUP BY snapshot_date
            ORDER BY snapshot_date ASC
        """
        with self.db_manager.duckdb_connection() as connection:
            frame = connection.execute(query).fetchdf()
        if frame.empty:
            return frame
        frame["snapshot_date"] = pd.to_datetime(frame["snapshot_date"]).dt.date
        return frame

    def _write_report(self) -> Path:
        frame = self.db_manager.load_regime_meter()
        report_path = self.db_manager.paths.reports_dir / "regime_report.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        if frame.empty:
            report_path.write_text(
                "\n".join(
                    [
                        "# Regime Meter",
                        "",
                        "- status: no persisted regime rows",
                        "- note: metrics use a 20-session label lag by design.",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            return report_path
        frame = frame.copy()
        frame["snapshot_date"] = pd.to_datetime(frame["snapshot_date"])
        frame = frame.sort_values("snapshot_date").reset_index(drop=True)
        latest = frame.iloc[-1]
        lines = [
            "# Regime Meter",
            "",
            f"- generated_at: {datetime.now(timezone.utc).isoformat()}",
            f"- current_classification: {latest['classification']}",
            f"- latest_matured_date: {latest['snapshot_date'].date().isoformat()}",
            f"- lag_note: metrics as of {latest['snapshot_date'].date().isoformat()} (20-session label lag by design).",
            f"- mom_ic_daily: {self._fmt(latest['mom_ic_daily'])}",
            f"- mom_ic_20d_avg: {self._fmt(latest['mom_ic_20d_avg'])}",
            f"- mom_ic_60d_avg: {self._fmt(latest['mom_ic_60d_avg'])}",
            f"- wml_20d_alpha: {self._fmt(latest['wml_20d_alpha'])}",
            f"- wml_1d: {self._fmt(latest['wml_1d'])} (diagnostic only; not used for classification)",
            "",
            "## Trailing 60 Matured Dates",
            "",
            "| date | ic_daily | ic_20d_avg | classification |",
            "|---|---:|---:|---|",
        ]
        for row in frame.tail(60).itertuples(index=False):
            lines.append(
                f"| {row.snapshot_date.date().isoformat()} | {self._fmt(row.mom_ic_daily)} | "
                f"{self._fmt(row.mom_ic_20d_avg)} | {row.classification} |"
            )
        lines.extend(self._render_transition_matrices(frame, current_classification=str(latest["classification"])))
        monthly = (
            frame.assign(month=frame["snapshot_date"].dt.strftime("%Y-%m"))
            .groupby("month", as_index=False)["mom_ic_daily"]
            .mean()
            .tail(24)
        )
        lines.extend(
            [
                "",
                "## Historical Monthly Averages",
                "",
                "| month | momentum IC |",
                "|---|---:|",
            ]
        )
        for row in monthly.itertuples(index=False):
            lines.append(f"| {row.month} | {self._fmt(row.mom_ic_daily)} |")
        lines.append("")
        report_path.write_text("\n".join(lines), encoding="utf-8")
        return report_path

    def _render_transition_matrices(self, frame: pd.DataFrame, *, current_classification: str) -> list[str]:
        lines = [
            "",
            "## Regime Transition Matrix",
            "",
            "- note: transition matrices are diagnostic inputs for sizing and exposure decisions; they are not consumed by the shortlist promotion gate.",
        ]
        expected: list[str] = []
        for horizon in (20, 60):
            matrix = self._transition_matrix(frame, horizon_sessions=horizon)
            lines.extend(
                [
                    "",
                    f"### +{horizon} Sessions",
                    "",
                    "| entry_regime | neutral | trending | reversal |",
                    "|---|---:|---:|---:|",
                ]
            )
            for entry in ("neutral", "trending", "reversal"):
                row = matrix.get(entry, {})
                total = sum(int(row.get(exit_regime, 0)) for exit_regime in ("neutral", "trending", "reversal"))
                cells = []
                for exit_regime in ("neutral", "trending", "reversal"):
                    count = int(row.get(exit_regime, 0))
                    probability = count / total if total else float("nan")
                    cells.append(f"{self._fmt(probability)} ({count})")
                lines.append(f"| {entry} | {' | '.join(cells)} |")
            current_row = matrix.get(str(current_classification), {})
            current_total = sum(int(current_row.get(exit_regime, 0)) for exit_regime in ("neutral", "trending", "reversal"))
            if current_total:
                distribution = ", ".join(
                    f"{exit_regime}={self._fmt(int(current_row.get(exit_regime, 0)) / current_total)}"
                    for exit_regime in ("neutral", "trending", "reversal")
                )
            else:
                distribution = "n/a"
            expected.append(f"+{horizon}: {distribution}")
        lines.extend(
            [
                "",
                f"- current_classification_expected_distribution: {current_classification} -> {'; '.join(expected)}",
                "",
            ]
        )
        return lines

    def _transition_matrix(self, frame: pd.DataFrame, *, horizon_sessions: int) -> dict[str, dict[str, int]]:
        if frame.empty or "classification" not in frame.columns:
            return {}
        working = frame.copy()
        working["snapshot_date"] = pd.to_datetime(working["snapshot_date"], errors="coerce")
        working = working.dropna(subset=["snapshot_date"]).sort_values("snapshot_date").reset_index(drop=True)
        output: dict[str, dict[str, int]] = {
            entry: {exit_regime: 0 for exit_regime in ("neutral", "trending", "reversal")}
            for entry in ("neutral", "trending", "reversal")
        }
        horizon = max(int(horizon_sessions), 1)
        classifications = working["classification"].astype(str).tolist()
        for index, entry in enumerate(classifications):
            exit_index = index + horizon
            if exit_index >= len(classifications):
                continue
            if entry not in output:
                continue
            exit_regime = classifications[exit_index]
            if exit_regime not in output[entry]:
                continue
            output[entry][exit_regime] += 1
        return output

    def _latest_known_row(self) -> dict | None:
        frame = self.db_manager.load_regime_meter()
        if frame.empty:
            return None
        frame = frame.sort_values("snapshot_date")
        return frame.iloc[-1].to_dict()

    @staticmethod
    def classify(
        mom_ic_20d_avg: float | int | None,
        *,
        reversal_ic_threshold: float,
        trending_ic_threshold: float,
    ) -> str:
        if mom_ic_20d_avg is None:
            return "neutral"
        try:
            value = float(mom_ic_20d_avg)
        except (TypeError, ValueError):
            return "neutral"
        if not math.isfinite(value):
            return "neutral"
        if value < float(reversal_ic_threshold):
            return "reversal"
        if value > float(trending_ic_threshold):
            return "trending"
        return "neutral"

    @staticmethod
    def _fmt(value) -> str:
        if value is None or pd.isna(value):
            return "nan"
        return f"{float(value):.4f}"

    @staticmethod
    def _thresholds() -> dict[str, float]:
        config = load_feature_config()
        raw = config.get("regime_gating", {}) if isinstance(config, dict) else {}
        return {
            "reversal_ic_threshold": float(raw.get("reversal_ic_threshold", -0.05)),
            "trending_ic_threshold": float(raw.get("trending_ic_threshold", 0.05)),
        }
