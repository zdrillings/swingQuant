from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from src.research.shortlist_universe import normalize_eligible_universe_mode, normalize_model_scope
from src.settings import load_feature_config
from src.utils.db_manager import DatabaseManager


@dataclass(frozen=True)
class ShortlistEarningsOverlayReport:
    output_path: str
    generated_at: str
    row_count: int
    date_count: int


class ShortlistEarningsOverlayService:
    EARNINGS_COLUMNS = [
        "snapshot_date",
        "ticker",
        "last_earnings_gap_pct",
        "last_earnings_volume_ratio_20",
        "last_earnings_open_vs_20d_high",
        "close_vs_last_earnings_close",
        "days_since_last_earnings",
        "days_to_next_earnings",
    ]

    def __init__(self, db_manager: DatabaseManager) -> None:
        self.db_manager = db_manager

    def run(
        self,
        *,
        top_n: int = 10,
        horizon_days: int = 20,
        recent_dates: int = 60,
        model_name: str | None = None,
        eligible_universe_mode: str | None = None,
        model_scope: str | None = None,
        xgboost_config: str | None = None,
        generated_at: str | None = None,
    ) -> ShortlistEarningsOverlayReport:
        self.db_manager.initialize()
        config = (load_feature_config().get("scan_policy", {}) or {}).get("shortlist_model", {}) or {}
        eligible_universe_mode = normalize_eligible_universe_mode(
            eligible_universe_mode
            or config.get("production_eligible_universe_mode")
            or config.get("eligible_universe_mode")
            or "passed_or_trend"
        )
        model_scope = normalize_model_scope(model_scope or config.get("production_model_scope") or "sector_specific")
        xgboost_config = str(xgboost_config or config.get("production_xgboost_config") or "baseline")

        if generated_at is None:
            runs = self.db_manager.load_shortlist_model_runs(
                horizon_days=int(horizon_days),
                eligible_universe_mode=eligible_universe_mode,
                model_scope=model_scope,
                xgboost_config=xgboost_config,
                limit=1,
            )
            if runs.empty:
                raise ValueError(
                    "No shortlist model runs found for "
                    f"eligible_universe_mode={eligible_universe_mode}, model_scope={model_scope}, "
                    f"xgboost_config={xgboost_config}."
                )
            generated_at = str(runs.iloc[0]["generated_at"])
            if model_name in (None, ""):
                model_name = str(runs.iloc[0]["champion_model"])
        elif model_name in (None, ""):
            model_name = str(config.get("production_model_name") or "")
            if not model_name:
                raise ValueError("model_name is required when generated_at is supplied and no production_model_name is configured.")
        else:
            model_name = str(model_name)

        predictions = self.db_manager.load_shortlist_model_predictions(
            generated_at=str(generated_at),
            horizon_days=int(horizon_days),
            eligible_universe_mode=eligible_universe_mode,
            model_scope=model_scope,
            dataset_split="oos",
            model_name=model_name,
        )
        if predictions.empty:
            raise ValueError("No out-of-sample shortlist model predictions found.")

        snapshots = self.db_manager.load_universe_daily_snapshots()
        frame = self._build_analysis_frame(predictions, snapshots)
        if frame.empty:
            raise ValueError("No joined prediction/snapshot rows found for earnings overlay analysis.")

        variants = self._add_overlay_scores(frame)
        windows = self._window_labels(frame, recent_dates=int(recent_dates))
        report = self._build_report(
            frame=frame,
            variants=variants,
            windows=windows,
            top_n=int(top_n),
            generated_at=str(generated_at),
            horizon_days=int(horizon_days),
            model_name=model_name,
            eligible_universe_mode=eligible_universe_mode,
            model_scope=model_scope,
            xgboost_config=xgboost_config,
        )
        report_path = self.db_manager.paths.reports_dir / "earnings_confirmation_live_comparison.md"
        report_path.write_text("\n".join(report), encoding="utf-8")
        return ShortlistEarningsOverlayReport(
            output_path=str(report_path),
            generated_at=str(generated_at),
            row_count=len(frame.index),
            date_count=int(frame["snapshot_date"].nunique()),
        )

    def _build_analysis_frame(self, predictions: pd.DataFrame, snapshots: pd.DataFrame) -> pd.DataFrame:
        pred = predictions.copy()
        pred["snapshot_date"] = pd.to_datetime(pred["snapshot_date"]).dt.normalize()
        pred["target"] = pd.to_numeric(pred["actual_alpha_vs_sector"], errors="coerce")
        pred["predicted_alpha"] = pd.to_numeric(pred["predicted_alpha"], errors="coerce")
        snap = snapshots.copy()
        snap["snapshot_date"] = pd.to_datetime(snap["snapshot_date"]).dt.normalize()
        available_columns = [column for column in self.EARNINGS_COLUMNS if column in snap.columns]
        merged = pred.merge(
            snap[available_columns].copy(),
            on=["snapshot_date", "ticker"],
            how="left",
        )
        merged = merged.dropna(subset=["target", "predicted_alpha"]).copy()
        merged["base_rank"] = merged.groupby("snapshot_date")["predicted_alpha"].rank(method="average", pct=True)
        return merged

    def _add_overlay_scores(self, frame: pd.DataFrame) -> dict[str, str]:
        variants = {"live_model": "base_rank"}
        frame["gap_rank"] = self._date_rank(frame, "last_earnings_gap_pct")
        frame["volume_rank"] = self._date_rank(frame, "last_earnings_volume_ratio_20")
        frame["open_high_rank"] = self._date_rank(frame, "last_earnings_open_vs_20d_high")
        frame["hold_rank"] = self._date_rank(frame, "close_vs_last_earnings_close")
        frame["reaction_rank_raw"] = frame[
            ["gap_rank", "volume_rank", "open_high_rank", "hold_rank"]
        ].mean(axis=1, skipna=True)

        days_since = pd.to_numeric(frame.get("days_since_last_earnings"), errors="coerce")
        frame["reaction_rank_all"] = frame["reaction_rank_raw"].fillna(0.5)
        for max_days in (30, 45, 63):
            column = f"reaction_rank_{max_days}"
            frame[column] = frame["reaction_rank_raw"].where(
                (days_since >= 0.0) & (days_since <= float(max_days)),
                0.5,
            ).fillna(0.5)

        days_to_next = pd.to_numeric(frame.get("days_to_next_earnings"), errors="coerce")
        safety = np.where((days_to_next >= 0.0) & (days_to_next <= 10.0), 0.0, 1.0)
        frame["pre_earnings_safety_rank"] = pd.Series(safety, index=frame.index).groupby(
            frame["snapshot_date"]
        ).rank(method="average", pct=True).fillna(0.5)
        frame["reaction_safety_45"] = frame[["reaction_rank_45", "pre_earnings_safety_rank"]].mean(
            axis=1,
            skipna=True,
        ).fillna(0.5)

        for overlay in (
            "reaction_rank_all",
            "reaction_rank_30",
            "reaction_rank_45",
            "reaction_rank_63",
            "reaction_safety_45",
        ):
            for weight in (0.03, 0.05, 0.10):
                name = f"{overlay}_{int(weight * 100)}pct"
                frame[name] = ((1.0 - weight) * frame["base_rank"]) + (weight * frame[overlay])
                variants[name] = name
        return variants

    def _date_rank(self, frame: pd.DataFrame, column: str) -> pd.Series:
        if column not in frame.columns:
            return pd.Series(np.nan, index=frame.index)
        return pd.to_numeric(frame[column], errors="coerce").groupby(frame["snapshot_date"]).rank(
            method="average",
            pct=True,
        )

    def _window_labels(self, frame: pd.DataFrame, *, recent_dates: int) -> list[str]:
        labels = ["all", "120", str(max(int(recent_dates), 1)), "20"]
        deduped: list[str] = []
        for label in labels:
            if label not in deduped:
                deduped.append(label)
        return deduped

    def _build_report(
        self,
        *,
        frame: pd.DataFrame,
        variants: dict[str, str],
        windows: list[str],
        top_n: int,
        generated_at: str,
        horizon_days: int,
        model_name: str,
        eligible_universe_mode: str,
        model_scope: str,
        xgboost_config: str,
    ) -> list[str]:
        summaries: dict[str, dict[str, dict[str, float]]] = {}
        deltas: dict[str, dict[str, dict[str, float]]] = {}
        top_values = sorted({6, int(top_n)})
        for top in top_values:
            top_key = str(top)
            summaries[top_key] = {}
            deltas[top_key] = {}
            for variant, score_column in variants.items():
                summaries[top_key][variant] = {
                    window: self._evaluate_score(frame, score_column=score_column, top_n=top, window=window)
                    for window in windows
                }
            for variant in variants:
                if variant == "live_model":
                    continue
                deltas[top_key][variant] = {
                    window: self._delta(summaries[top_key][variant][window], summaries[top_key]["live_model"][window])
                    for window in windows
                }

        regressions = {
            window: {
                overlay: self._ols(
                    frame=frame,
                    overlay_column=overlay,
                    window=window,
                )
                for overlay in (
                    "reaction_rank_all",
                    "reaction_rank_30",
                    "reaction_rank_45",
                    "reaction_rank_63",
                    "reaction_safety_45",
                )
            }
            for window in windows
        }
        best = self._best_delta_rows(deltas=deltas, windows=windows, top_values=top_values)

        dates = sorted(frame["snapshot_date"].drop_duplicates().tolist())
        lines = [
            "# Earnings Confirmation Live Model Comparison",
            "",
            f"- live_model_run: {generated_at}",
            f"- live_model: {model_name} / {eligible_universe_mode} / {model_scope} / {xgboost_config}",
            f"- oos_dates: {len(dates)} ({pd.Timestamp(dates[0]).date()} to {pd.Timestamp(dates[-1]).date()})",
            f"- oos_rows: {len(frame.index)}",
            f"- target: alpha_vs_sector_{int(horizon_days)}d",
            "",
            "## Live Model Baseline",
            "",
            "| Top | Window | Mean | Median | Daily Median | P25 | P10 | Hit |",
            "|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
        for top in top_values:
            for window in windows:
                row = summaries[str(top)]["live_model"][window]
                lines.append(
                    "| {} | {} | {:.2%} | {:.2%} | {:.2%} | {:.2%} | {:.2%} | {:.2%} |".format(
                        top,
                        window,
                        row["mean_alpha"],
                        row["median_alpha"],
                        row["mean_daily_median_alpha"],
                        row["p25_alpha"],
                        row["p10_alpha"],
                        row["hit_rate"],
                    )
                )

        lines.extend(
            [
                "",
                "## Best Earnings Overlay Deltas vs Live",
                "",
                "| Top | Window | Variant | Mean d | Median d | Daily Median d | P25 d | P10 d | Hit d |",
                "|---:|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for item in best:
            for row in item["best"][:3]:
                lines.append(
                    "| {} | {} | {} | {:+.2%} | {:+.2%} | {:+.2%} | {:+.2%} | {:+.2%} | {:+.2%} |".format(
                        item["top"],
                        item["window"],
                        row["variant"],
                        row["mean_alpha"],
                        row["median_alpha"],
                        row["mean_daily_median_alpha"],
                        row["p25_alpha"],
                        row["p10_alpha"],
                        row["hit_rate"],
                    )
                )

        lines.extend(
            [
                "",
                "## Incremental Regression",
                "",
                "OLS: alpha_vs_sector target ~ live_model_rank + earnings_overlay_score over candidate rows.",
                "",
                "| Window | Overlay | n | overlay_beta | overlay_t | model_beta | model_t | r2 |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for window in windows:
            for overlay, row in regressions[window].items():
                lines.append(
                    "| {} | {} | {} | {:+.4f} | {:+.2f} | {:+.4f} | {:+.2f} | {:.4f} |".format(
                        window,
                        overlay,
                        row["n"],
                        row["overlay_beta"],
                        row["overlay_t"],
                        row["model_beta"],
                        row["model_t"],
                        row["r2"],
                    )
                )
        return lines

    def _evaluate_score(
        self,
        frame: pd.DataFrame,
        *,
        score_column: str,
        top_n: int,
        window: str,
    ) -> dict[str, float]:
        chosen_dates = self._selected_dates(frame, window=window)
        rows: list[dict[str, float]] = []
        scoped = frame[frame["snapshot_date"].isin(chosen_dates)].copy()
        for _, day in scoped.groupby("snapshot_date", sort=True):
            picks = day.sort_values([score_column, "predicted_alpha", "ticker"], ascending=[False, False, True]).head(
                int(top_n)
            )
            target = picks["target"].dropna()
            universe = day["target"].dropna()
            if target.empty or universe.empty:
                continue
            rows.append(
                {
                    "mean": float(target.mean()),
                    "median": float(target.median()),
                    "p25": float(target.quantile(0.25)),
                    "p10": float(target.quantile(0.10)),
                    "hit": float((target > 0.0).mean()),
                    "universe_mean": float(universe.mean()),
                }
            )
        daily = pd.DataFrame(rows)
        if daily.empty:
            return {
                "dates": 0,
                "mean_alpha": math.nan,
                "median_alpha": math.nan,
                "mean_daily_median_alpha": math.nan,
                "p25_alpha": math.nan,
                "p10_alpha": math.nan,
                "hit_rate": math.nan,
                "beat_universe_rate": math.nan,
                "positive_date_rate": math.nan,
            }
        return {
            "dates": float(len(daily.index)),
            "mean_alpha": float(daily["mean"].mean()),
            "median_alpha": float(daily["median"].median()),
            "mean_daily_median_alpha": float(daily["median"].mean()),
            "p25_alpha": float(daily["p25"].mean()),
            "p10_alpha": float(daily["p10"].mean()),
            "hit_rate": float(daily["hit"].mean()),
            "beat_universe_rate": float((daily["mean"] > daily["universe_mean"]).mean()),
            "positive_date_rate": float((daily["mean"] > 0.0).mean()),
        }

    def _selected_dates(self, frame: pd.DataFrame, *, window: str) -> list[pd.Timestamp]:
        dates = sorted(frame["snapshot_date"].drop_duplicates().tolist())
        if window == "all":
            return dates
        return dates[-min(int(window), len(dates)) :]

    def _delta(self, candidate: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:
        return {
            key: float(candidate[key] - baseline[key])
            for key in (
                "mean_alpha",
                "median_alpha",
                "mean_daily_median_alpha",
                "p25_alpha",
                "p10_alpha",
                "hit_rate",
                "beat_universe_rate",
                "positive_date_rate",
            )
        }

    def _best_delta_rows(
        self,
        *,
        deltas: dict[str, dict[str, dict[str, dict[str, float]]]],
        windows: list[str],
        top_values: list[int],
    ) -> list[dict[str, object]]:
        results: list[dict[str, object]] = []
        for top in top_values:
            top_key = str(top)
            for window in windows:
                rows: list[dict[str, object]] = []
                for variant, variant_windows in deltas[top_key].items():
                    row = dict(variant_windows[window])
                    row["variant"] = variant
                    rows.append(row)
                rows = sorted(
                    rows,
                    key=lambda row: (
                        float(row["mean_alpha"]) + float(row["p10_alpha"]) + (float(row["hit_rate"]) * 0.05),
                        float(row["mean_daily_median_alpha"]),
                    ),
                    reverse=True,
                )
                results.append({"top": top_key, "window": window, "best": rows[:8]})
        return results

    def _ols(self, *, frame: pd.DataFrame, overlay_column: str, window: str) -> dict[str, float | int]:
        scoped = frame[frame["snapshot_date"].isin(self._selected_dates(frame, window=window))].copy()
        y = scoped["target"].to_numpy(dtype=float)
        x = scoped[["base_rank", overlay_column]].to_numpy(dtype=float)
        mask = np.isfinite(y) & np.isfinite(x).all(axis=1)
        y = y[mask]
        x = x[mask]
        if len(y) == 0:
            return {"n": 0, "r2": math.nan, "overlay_beta": math.nan, "overlay_t": math.nan, "model_beta": math.nan, "model_t": math.nan}
        design = np.column_stack([np.ones(len(x)), x])
        beta = np.linalg.lstsq(design, y, rcond=None)[0]
        residual = y - (design @ beta)
        total = float(((y - y.mean()) @ (y - y.mean())))
        r2 = 1.0 - (float(residual @ residual) / total) if total else math.nan
        dof = max(len(y) - design.shape[1], 1)
        s2 = float((residual @ residual) / dof)
        covariance = s2 * np.linalg.pinv(design.T @ design)
        stderr = np.sqrt(np.diag(covariance))
        t_stats = beta / stderr
        return {
            "n": int(len(y)),
            "r2": float(r2),
            "model_beta": float(beta[1]),
            "model_t": float(t_stats[1]),
            "overlay_beta": float(beta[2]),
            "overlay_t": float(t_stats[2]),
        }
