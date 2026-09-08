from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.research.shortlist_model_service import PROMOTION_BASKET_SIZE
from src.settings import load_feature_config
from src.utils.db_manager import DatabaseManager
from src.utils.performance_metrics import annualized_sharpe, newey_west_t_stat


@dataclass(frozen=True)
class Phase2ResearchReport:
    output_path: str
    oos_dates: int
    models: int


class Phase2ResearchService:
    def __init__(self, db_manager: DatabaseManager) -> None:
        self.db_manager = db_manager

    def run(
        self,
        *,
        horizon_days: int = 20,
        top_n: int = PROMOTION_BASKET_SIZE,
        trial_count: int = 200,
    ) -> Phase2ResearchReport:
        self.db_manager.initialize()
        predictions, provenance = self._load_oos_predictions(horizon_days=int(horizon_days))
        scan_candidates = self._load_scan_candidates()
        report_path = self.db_manager.paths.reports_dir / "phase2_research.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        cost_fraction = self._round_trip_cost_fraction()
        if predictions.empty:
            report_path.write_text(
                "\n".join(
                    [
                        "# Phase 2 Research",
                        "",
                        f"- horizon_days: {int(horizon_days)}",
                        "- oos_dates: 0",
                        "- models: 0",
                        f"- round_trip_cost: {self._fmt(cost_fraction)}",
                        f"- provenance_source: {provenance.get('source', 'none')}",
                        "",
                        "No persisted OOS shortlist predictions are available.",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            return Phase2ResearchReport(output_path=str(report_path), oos_dates=0, models=0)

        target_column = self._target_column(predictions, horizon_days=int(horizon_days))
        baskets = self._date_model_baskets(
            predictions,
            target_column=target_column,
            top_n=int(top_n),
            cost_fraction=cost_fraction,
        )
        lines = [
            "# Phase 2 Research",
            "",
            f"- horizon_days: {int(horizon_days)}",
            f"- target_column: {target_column}",
            f"- promotion_top_n: {int(top_n)}",
            f"- oos_dates: {int(predictions['snapshot_date'].nunique())}",
            f"- models: {int(predictions['model_name'].nunique())}",
            f"- round_trip_cost: {self._fmt(cost_fraction)}",
            f"- provenance_source: {provenance.get('source', 'unknown')}",
            f"- provenance_generated_at: {provenance.get('generated_at', 'n/a')}",
            f"- provenance_feature_profile: {provenance.get('feature_profile', 'n/a')}",
            f"- provenance_test_window_dates: {provenance.get('test_window_dates', 'n/a')}",
            f"- provenance_oos_stride_dates: {provenance.get('oos_evaluation_stride_dates', 'n/a')}",
            "- harness_note: current report artifacts are preferred over champion-only DB state; DB is a fallback.",
            "",
        ]
        lines.extend(self._render_meta_labeling(baskets, horizon_days=int(horizon_days)))
        lines.extend(self._render_random_baseline(baskets))
        lines.extend(
            self._render_opportunity_score_repair(
                scan_candidates,
                horizon_days=int(horizon_days),
                cost_fraction=cost_fraction,
            )
        )
        lines.extend(self._render_crash_conditioning(baskets, horizon_days=int(horizon_days)))
        lines.extend(self._render_deflation_gates(baskets, horizon_days=int(horizon_days), trial_count=int(trial_count)))
        report_path.write_text("\n".join(lines), encoding="utf-8")
        return Phase2ResearchReport(
            output_path=str(report_path),
            oos_dates=int(predictions["snapshot_date"].nunique()),
            models=int(predictions["model_name"].nunique()),
        )

    def _load_oos_predictions(self, *, horizon_days: int) -> tuple[pd.DataFrame, dict[str, object]]:
        csv_path = self.db_manager.paths.reports_dir / "shortlist_model_oos_predictions.csv"
        if csv_path.exists():
            provenance = self._parse_shortlist_report_provenance(self.db_manager.paths.reports_dir / "shortlist_model.md")
            provenance["source"] = "artifact_csv"
            return self._normalize_predictions(pd.read_csv(csv_path)), provenance
        runs_loader = getattr(self.db_manager, "load_shortlist_model_runs", None)
        predictions_loader = getattr(self.db_manager, "load_shortlist_model_predictions", None)
        if callable(runs_loader) and callable(predictions_loader):
            try:
                runs = runs_loader(horizon_days=int(horizon_days), active_only=False, limit=1)
            except TypeError:
                runs = runs_loader(horizon_days=int(horizon_days), limit=1)
            if runs is not None and not runs.empty:
                generated_at = runs.iloc[0].get("generated_at")
                frames = []
                for model_name in ("signal_proxy", "ridge_model", "lasso_model", "elastic_net_model", "ic_sign_model", "xgboost_model", "ensemble_model"):
                    loaded = predictions_loader(
                        generated_at=generated_at,
                        horizon_days=int(horizon_days),
                        dataset_split="oos",
                        model_name=model_name,
                    )
                    if loaded is not None and not loaded.empty:
                        frame = loaded.copy()
                        frame["model_name"] = model_name
                        frames.append(frame)
                if frames:
                    provenance = {"source": "latest_db_run_fallback", "generated_at": generated_at}
                    for key in ("feature_profile", "test_window_dates", "oos_evaluation_stride_dates", "target_column"):
                        if key in runs.columns:
                            provenance[key] = runs.iloc[0].get(key)
                    return self._normalize_predictions(pd.concat(frames, axis=0, ignore_index=True)), provenance
        return pd.DataFrame(), {"source": "none"}

    def _parse_shortlist_report_provenance(self, report_path: Path) -> dict[str, object]:
        provenance: dict[str, object] = {}
        if not report_path.exists():
            return provenance
        for raw_line in report_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line.startswith("- ") or ":" not in line:
                continue
            key, value = line[2:].split(":", 1)
            key = key.strip()
            if key in {
                "generated_at",
                "feature_profile",
                "test_window_dates",
                "oos_evaluation_stride_dates",
                "target_column",
                "evaluation_target_column",
                "selected_model",
                "selected_model_gate_passed",
                "round_trip_cost",
            }:
                provenance[key] = value.strip()
        return provenance

    def _normalize_predictions(self, frame: pd.DataFrame) -> pd.DataFrame:
        working = frame.copy()
        if "actual_alpha_vs_sector" in working.columns and "alpha_vs_sector_20d" not in working.columns:
            working["alpha_vs_sector_20d"] = working["actual_alpha_vs_sector"]
        working["snapshot_date"] = pd.to_datetime(working["snapshot_date"]).dt.normalize()
        working["predicted_alpha"] = pd.to_numeric(working["predicted_alpha"], errors="coerce")
        return working.dropna(subset=["snapshot_date", "model_name", "predicted_alpha"]).copy()

    def _load_scan_candidates(self) -> pd.DataFrame:
        loader = getattr(self.db_manager, "load_scan_candidates", None)
        if not callable(loader):
            return pd.DataFrame()
        frame = loader()
        return frame.copy() if frame is not None else pd.DataFrame()

    def _target_column(self, predictions: pd.DataFrame, *, horizon_days: int) -> str:
        path_column = f"path_alpha_vs_sector_{int(horizon_days)}d"
        fixed_column = f"alpha_vs_sector_{int(horizon_days)}d"
        if path_column in predictions.columns and pd.to_numeric(predictions[path_column], errors="coerce").notna().any():
            return path_column
        if fixed_column in predictions.columns:
            return fixed_column
        return "actual_alpha_vs_sector"

    def _date_model_baskets(
        self,
        predictions: pd.DataFrame,
        *,
        target_column: str,
        top_n: int,
        cost_fraction: float,
    ) -> pd.DataFrame:
        rows = []
        working = predictions.copy()
        working[target_column] = pd.to_numeric(working[target_column], errors="coerce")
        for (model_name, snapshot_date), day in working.groupby(["model_name", "snapshot_date"], sort=True):
            ordered = day.dropna(subset=[target_column]).sort_values(["predicted_alpha", "ticker"], ascending=[False, True])
            picks = ordered.head(int(top_n)).copy()
            if picks.empty:
                continue
            target = pd.to_numeric(picks[target_column], errors="coerce").dropna()
            universe_target = pd.to_numeric(ordered[target_column], errors="coerce").dropna()
            if target.empty or universe_target.empty:
                continue
            sector_share = float("nan")
            if "sector" in picks.columns and not picks["sector"].dropna().empty:
                sector_share = float(picks["sector"].astype(str).value_counts(normalize=True).iloc[0])
            rows.append(
                {
                    "model_name": str(model_name),
                    "snapshot_date": pd.Timestamp(snapshot_date),
                    "gross_mean_target": float(target.mean()),
                    "mean_target": float(target.mean()) - float(cost_fraction),
                    "hit_rate": float((target - float(cost_fraction) > 0.0).mean()),
                    "avg_score": float(picks["predicted_alpha"].mean()),
                    "universe_mean_target": float(universe_target.mean()) - float(cost_fraction),
                    "top_sector_share": sector_share,
                }
            )
        return pd.DataFrame(rows)

    def _render_meta_labeling(self, baskets: pd.DataFrame, *, horizon_days: int) -> list[str]:
        lines = ["## P2.2 Meta-Label Sizing", ""]
        if baskets.empty:
            return lines + ["No OOS baskets available.", ""]
        rows = []
        for model_name, model_frame in baskets.groupby("model_name", sort=True):
            ordered = model_frame.sort_values("snapshot_date").reset_index(drop=True)
            flat = pd.to_numeric(ordered["mean_target"], errors="coerce")
            weighted_returns = []
            for index, row in ordered.iterrows():
                prior = ordered.iloc[:index].copy()
                if len(prior.index) < 20:
                    weight = 0.5
                else:
                    threshold = float(pd.to_numeric(prior["avg_score"], errors="coerce").median())
                    prior_high = prior[pd.to_numeric(prior["avg_score"], errors="coerce") >= threshold]
                    prior_low = prior[pd.to_numeric(prior["avg_score"], errors="coerce") < threshold]
                    high_hit = float((pd.to_numeric(prior_high["mean_target"], errors="coerce") > 0.0).mean()) if not prior_high.empty else 0.5
                    low_hit = float((pd.to_numeric(prior_low["mean_target"], errors="coerce") > 0.0).mean()) if not prior_low.empty else 0.5
                    score = float(row["avg_score"])
                    probability = high_hit if score >= threshold else low_hit
                    weight = 0.0 if probability < 0.45 else 0.5 if probability < 0.55 else 1.0
                weighted_returns.append(float(row["mean_target"]) * weight)
            weighted = pd.Series(weighted_returns, dtype=float)
            rows.append(
                {
                    "model": model_name,
                    "dates": len(ordered.index),
                    "flat_mean": flat.mean(),
                    "meta_mean": weighted.mean(),
                    "flat_drawdown": self._max_drawdown(self._non_overlapping_returns(flat, horizon_days=horizon_days)),
                    "meta_drawdown": self._max_drawdown(self._non_overlapping_returns(weighted, horizon_days=horizon_days)),
                }
            )
        lines.append(
            f"- drawdown_basis: non-overlapping every-{int(horizon_days)}th basket observation; overlapping observations are not compounded as daily P&L."
        )
        lines.append("")
        lines.extend(["| model | dates | flat_mean | meta_mean | flat_max_dd | meta_max_dd |", "|---|---:|---:|---:|---:|---:|"])
        for row in rows:
            lines.append(
                f"| {row['model']} | {int(row['dates'])} | {self._fmt(row['flat_mean'])} | {self._fmt(row['meta_mean'])} | "
                f"{self._fmt(row['flat_drawdown'])} | {self._fmt(row['meta_drawdown'])} |"
            )
        lines.append("")
        return lines

    def _render_random_baseline(self, baskets: pd.DataFrame) -> list[str]:
        lines = ["## Random-Eligible Baseline and Tilt", ""]
        if baskets.empty or "universe_mean_target" not in baskets.columns:
            return lines + ["No OOS universe baseline is available.", ""]
        rows = []
        for model_name, model_frame in baskets.groupby("model_name", sort=True):
            selected = pd.to_numeric(model_frame["mean_target"], errors="coerce")
            baseline = pd.to_numeric(model_frame["universe_mean_target"], errors="coerce")
            spread = selected - baseline
            rows.append(
                {
                    "model": model_name,
                    "dates": len(model_frame.index),
                    "selected_mean": selected.mean(),
                    "random_eligible_mean": baseline.mean(),
                    "selection_spread": spread.mean(),
                    "beat_random_rate": float((spread > 0.0).mean()),
                    "avg_top_sector_share": pd.to_numeric(
                        model_frame["top_sector_share"] if "top_sector_share" in model_frame.columns else pd.Series(dtype=float),
                        errors="coerce",
                    ).mean(),
                }
            )
        lines.extend(
            [
                "| model | dates | selected_mean | random_eligible_mean | selection_spread | beat_random_rate | avg_top_sector_share |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in rows:
            lines.append(
                f"| {row['model']} | {int(row['dates'])} | {self._fmt(row['selected_mean'])} | "
                f"{self._fmt(row['random_eligible_mean'])} | {self._fmt(row['selection_spread'])} | "
                f"{self._fmt(row['beat_random_rate'])} | {self._fmt(row['avg_top_sector_share'])} |"
            )
        lines.extend(
            [
                "",
                "- baseline_definition: equal-weight random eligible universe return on each model/date, net of the same round-trip cost.",
                "- tilt_note: avg_top_sector_share is a coarse concentration diagnostic, not a sector-neutral attribution.",
                "",
            ]
        )
        return lines

    def _render_opportunity_score_repair(
        self,
        scan_candidates: pd.DataFrame,
        *,
        horizon_days: int,
        cost_fraction: float,
    ) -> list[str]:
        lines = ["## P2.3 Opportunity Score Repair", ""]
        target_column = f"alpha_vs_sector_{int(horizon_days)}d"
        if scan_candidates.empty or not {"opportunity_score", target_column}.issubset(scan_candidates.columns):
            return lines + ["No scan-candidate opportunity score outcomes are available.", ""]
        scoped = scan_candidates.copy()
        scoped["opportunity_score"] = pd.to_numeric(scoped["opportunity_score"], errors="coerce")
        scoped[target_column] = pd.to_numeric(scoped[target_column], errors="coerce") - float(cost_fraction)
        scoped = scoped.dropna(subset=["opportunity_score", target_column])
        if scoped.empty:
            return lines + ["No matured opportunity score outcomes are available.", ""]
        bands = [("<0.30", -math.inf, 0.30), ("0.30-0.40", 0.30, 0.40), ("0.40-0.50", 0.40, 0.50), (">=0.50", 0.50, math.inf)]
        means = []
        lines.extend(["| band | rows | mean_alpha | hit_rate |", "|---|---:|---:|---:|"])
        for label, lower, upper in bands:
            band = scoped[(scoped["opportunity_score"] >= lower) & (scoped["opportunity_score"] < upper)]
            mean_alpha = float(band[target_column].mean()) if not band.empty else float("nan")
            means.append(mean_alpha)
            hit = float((band[target_column] > 0.0).mean()) if not band.empty else float("nan")
            lines.append(f"| {label} | {len(band.index)} | {self._fmt(mean_alpha)} | {self._fmt(hit)} |")
        finite_means = [value for value in means if math.isfinite(float(value))]
        monotone = all(left <= right for left, right in zip(finite_means, finite_means[1:]))
        lines.extend(
            [
                "",
                f"- monotone_non_decreasing: {str(monotone).lower()}",
                "- recommendation: replace hand score with calibrated model probability if monotonicity remains false on dense OOS.",
                "",
            ]
        )
        return lines

    def _render_crash_conditioning(self, baskets: pd.DataFrame, *, horizon_days: int) -> list[str]:
        lines = ["## P2.4 Momentum-Crash Conditioning", ""]
        if baskets.empty:
            return lines + ["No OOS baskets available.", ""]
        rows = []
        for model_name, model_frame in baskets.groupby("model_name", sort=True):
            ordered = model_frame.sort_values("snapshot_date").reset_index(drop=True)
            returns = pd.to_numeric(ordered["mean_target"], errors="coerce")
            lagged_vol = returns.rolling(20, min_periods=5).std().shift(1)
            median_vol = float(lagged_vol.dropna().median()) if lagged_vol.notna().any() else float("nan")
            if not math.isfinite(median_vol) or median_vol <= 0:
                conditioned = returns * 0.5
            else:
                weights = (median_vol / lagged_vol).clip(lower=0.25, upper=1.0).fillna(0.5)
                conditioned = returns * weights
            crash = self._non_overlapping_returns(returns.tail(min(60, len(returns.index))), horizon_days=horizon_days)
            conditioned_crash = self._non_overlapping_returns(conditioned.tail(min(60, len(conditioned.index))), horizon_days=horizon_days)
            rows.append(
                {
                    "model": model_name,
                    "raw_mean": returns.mean(),
                    "conditioned_mean": conditioned.mean(),
                    "raw_crash_dd": self._max_drawdown(crash),
                    "conditioned_crash_dd": self._max_drawdown(conditioned_crash),
                }
            )
        lines.extend(["| model | raw_mean | conditioned_mean | raw_last60_dd | conditioned_last60_dd |", "|---|---:|---:|---:|---:|"])
        for row in rows:
            lines.append(
                f"| {row['model']} | {self._fmt(row['raw_mean'])} | {self._fmt(row['conditioned_mean'])} | "
                f"{self._fmt(row['raw_crash_dd'])} | {self._fmt(row['conditioned_crash_dd'])} |"
            )
        lines.extend(
            [
                "",
                "- drawdown_basis: last-60 section compounds non-overlapping every-horizon observations only.",
                "- lookahead_note: conditioning scale uses the full-sample median of prior realized basket volatility; diagnostic only.",
                "",
            ]
        )
        return lines

    def _render_deflation_gates(self, baskets: pd.DataFrame, *, horizon_days: int, trial_count: int) -> list[str]:
        lines = ["## P2.5 Deflation Gates", ""]
        if baskets.empty:
            return lines + ["No OOS baskets available.", ""]
        trial_count = max(int(trial_count), int(baskets["model_name"].nunique()), 1)
        rows = []
        for model_name, model_frame in baskets.groupby("model_name", sort=True):
            returns = pd.to_numeric(model_frame.sort_values("snapshot_date")["mean_target"], errors="coerce").dropna()
            periods_per_year = max(252.0 / float(horizon_days), 1.0)
            sharpe = annualized_sharpe(returns, periods_per_year=periods_per_year)
            nw_t = newey_west_t_stat(returns, lag=int(horizon_days))
            dsr = self._deflated_t_probability(nw_t=nw_t, trial_count=trial_count)
            rows.append({"model": model_name, "dates": len(returns.index), "sharpe": sharpe, "nw_t": nw_t, "dsr": dsr})
        pbo = self._pbo_proxy(baskets)
        lines.extend(
            [
                f"- trial_count: {trial_count}",
                "- trial_count_definition: user/default estimate of the broader research search space, not just models present in this frame.",
                f"- pbo_block_rank_stability_proxy: {self._fmt(pbo)}",
                "- dsr_proxy_basis: Newey-West t-stat deflated by expected maximum of trial_count normal trials.",
                "",
                "| model | dates | sharpe | nw_t | dsr_proxy | passes |",
                "|---|---:|---:|---:|---:|---|",
            ]
        )
        for row in rows:
            passes = math.isfinite(float(row["dsr"])) and float(row["dsr"]) >= 0.95 and math.isfinite(pbo) and pbo < 0.30
            lines.append(
                f"| {row['model']} | {int(row['dates'])} | {self._fmt(row['sharpe'])} | {self._fmt(row['nw_t'])} | "
                f"{self._fmt(row['dsr'])} | {str(passes).lower()} |"
            )
        lines.append("")
        return lines

    def _deflated_t_probability(self, *, nw_t: float, trial_count: int) -> float:
        if trial_count < 1 or not math.isfinite(float(nw_t)):
            return float("nan")
        null_max = math.sqrt(2.0 * math.log(max(float(trial_count), 1.0)))
        adjusted = float(nw_t) - null_max
        return 0.5 * (1.0 + math.erf(adjusted / math.sqrt(2.0)))

    def _pbo_proxy(self, baskets: pd.DataFrame, *, blocks: int = 4) -> float:
        dates = sorted(baskets["snapshot_date"].drop_duplicates().tolist())
        if len(dates) < blocks * 2:
            return float("nan")
        date_blocks = np.array_split(np.array(dates, dtype=object), blocks)
        outcomes = []
        models = sorted(baskets["model_name"].astype(str).unique().tolist())
        for test_block in date_blocks:
            test_dates = set(pd.Timestamp(date).normalize() for date in test_block.tolist())
            train = baskets[~baskets["snapshot_date"].isin(test_dates)]
            test = baskets[baskets["snapshot_date"].isin(test_dates)]
            train_means = train.groupby("model_name")["mean_target"].mean().sort_values(ascending=False)
            if train_means.empty:
                continue
            selected = str(train_means.index[0])
            test_means = test.groupby("model_name")["mean_target"].mean().reindex(models).dropna().sort_values(ascending=False)
            if selected not in test_means.index or len(test_means.index) < 2:
                continue
            rank = list(test_means.index).index(selected)
            outcomes.append(rank >= len(test_means.index) / 2.0)
        return float(np.mean(outcomes)) if outcomes else float("nan")

    def _max_drawdown(self, returns: pd.Series) -> float:
        numeric = pd.to_numeric(returns, errors="coerce").dropna()
        if numeric.empty:
            return float("nan")
        equity = (1.0 + numeric).cumprod()
        peak = equity.cummax()
        drawdown = (equity / peak) - 1.0
        return float(drawdown.min())

    def _non_overlapping_returns(self, returns: pd.Series, *, horizon_days: int) -> pd.Series:
        numeric = pd.to_numeric(returns, errors="coerce").dropna().reset_index(drop=True)
        if numeric.empty:
            return numeric
        stride = max(int(horizon_days), 1)
        return numeric.iloc[::stride].reset_index(drop=True)

    def _round_trip_cost_fraction(self) -> float:
        config = load_feature_config()
        payload = config.get("backtest_costs", {}) if isinstance(config, dict) else {}
        slippage = float(payload.get("slippage_bps_per_side", 0.0) or 0.0)
        commission = float(payload.get("commission_bps_per_side", 0.0) or 0.0)
        return ((slippage + commission) * 2.0) / 10_000.0

    def _fmt(self, value: object) -> str:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return "n/a"
        if not math.isfinite(numeric):
            return "n/a"
        return f"{numeric:.6f}"
