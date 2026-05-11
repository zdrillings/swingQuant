from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import json
import math

import numpy as np
import pandas as pd

from src.sweep.service import BenchmarkContext, SweepService
from src.sync.service import REFERENCE_TICKERS
from src.utils.db_manager import DatabaseManager
from src.utils.logging import get_logger
from src.utils.promotion_policy import promotion_policy_violations
from src.utils.signal_engine import build_analysis_frame, filter_signal_candidates, latest_snapshot
from src.utils.strategy import SIGNAL_SCORE_MIN_KEY, evaluate_signal_gate, split_signal_indicators


@dataclass(frozen=True)
class EvaluateReport:
    output_path: str
    rows_written: int


class EvaluateService:
    def __init__(self, db_manager: DatabaseManager) -> None:
        self.db_manager = db_manager
        self.logger = get_logger("evaluate")

    def run(
        self,
        *,
        top: int = 10,
        sector: str | None = None,
        run_id: int | None = None,
        min_trades: int = 12,
        walk_forward: bool = False,
        walk_forward_windows: int = 5,
        walk_forward_shortlist: int = 25,
    ) -> EvaluateReport:
        self.db_manager.initialize()
        selected_run_id = run_id if run_id is not None else self.db_manager.latest_run_id()
        if selected_run_id is None:
            raise ValueError("No backtest results found. Run `sq sweep` first.")

        result_rows = self.db_manager.list_backtest_results(run_id=selected_run_id)
        if not result_rows:
            raise ValueError("No backtest results found. Run `sq sweep` first.")

        records = []
        for row in result_rows:
            params = json.loads(row["params_json"])
            row_sector = params.get("sector", "ALL")
            if sector is not None and row_sector != sector:
                continue
            records.append(
                {
                    "id": row["id"],
                    "run_id": row["run_id"],
                    "strategy_id": row["strategy_id"],
                    "params_json": row["params_json"],
                    "sector": row_sector,
                    "profit_factor": float(row["profit_factor"]),
                    "expectancy": float(row["expectancy"]),
                    "alpha_vs_spy": float(row["alpha_vs_spy"]) if row["alpha_vs_spy"] is not None else np.nan,
                    "alpha_vs_sector": float(row["alpha_vs_sector"]) if row["alpha_vs_sector"] is not None else np.nan,
                    "mdd": float(row["mdd"]),
                    "win_rate": float(row["win_rate"]),
                    "trade_count": int(row["trade_count"]) if row["trade_count"] is not None else None,
                }
            )

        if not records:
            raise ValueError("No backtest results match the requested filter.")

        frame = pd.DataFrame(records)
        if min_trades > 0:
            frame = frame[frame["trade_count"].notna() & (frame["trade_count"] >= min_trades)].copy()
        if frame.empty:
            raise ValueError(
                f"No backtest results remain after applying filters for run_id={selected_run_id} and min_trades={min_trades}."
            )

        frame["norm_expectancy"] = self._min_max(frame["expectancy"])
        frame["norm_profit_factor"] = self._min_max(frame["profit_factor"])
        frame["norm_max_drawdown"] = self._min_max(frame["mdd"])
        frame["norm_score"] = (
            frame["norm_expectancy"] * 0.4
            + frame["norm_profit_factor"] * 0.3
            - frame["norm_max_drawdown"] * 0.3
        )
        self.db_manager.update_backtest_norm_scores(
            [(int(row.id), float(row.norm_score)) for row in frame.itertuples(index=False)]
        )

        deduped = self._dedupe_plateaus(frame)
        candidate_links, gate_diagnostics = self._build_live_candidate_metadata(deduped)
        deduped = self._apply_practical_scoring(deduped, candidate_links)
        deduped = self._apply_promotion_policy_metadata(deduped)

        ranked = deduped.sort_values(
            ["practical_score", "norm_score", "expectancy"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
        ranked["global_rank"] = range(1, len(ranked.index) + 1)
        ranked["sector_rank"] = ranked.groupby("sector")["practical_score"].rank(method="dense", ascending=False).astype(int)
        top_ranked = self._cap_sector_repetition(ranked, max_per_sector=2, limit=top)
        live_ranked = (
            self._cap_sector_repetition(
                ranked[ranked["live_match_count"] > 0].sort_values(
                    ["live_match_count", "practical_score", "norm_score"],
                    ascending=[False, False, False],
                ),
                max_per_sector=2,
                limit=top,
            )
        )
        practical_live_ranked = (
            self._cap_sector_repetition(
                ranked[ranked["live_match_count"] > 0].sort_values(
                    ["practical_score", "live_match_count", "norm_score"],
                    ascending=[False, False, False],
                ),
                max_per_sector=2,
                limit=top,
            )
        )
        walk_forward_available = False
        if walk_forward:
            walk_forward_metrics = self._build_walk_forward_stability(
                ranked=ranked,
                top=top,
                shortlist_size=walk_forward_shortlist,
                windows=walk_forward_windows,
            )
            if not walk_forward_metrics.empty:
                walk_forward_available = True
                ranked = ranked.merge(walk_forward_metrics, on="id", how="left")
                top_ranked = top_ranked.merge(walk_forward_metrics, on="id", how="left")
                live_ranked = live_ranked.merge(walk_forward_metrics, on="id", how="left")
                practical_live_ranked = practical_live_ranked.merge(walk_forward_metrics, on="id", how="left")
        report_path = self.db_manager.paths.reports_dir / "candidates.md"
        lines = [
            "# Ranked Candidates",
            "",
            f"- run_id: {selected_run_id}",
            f"- min_trades: {min_trades}",
            "- ranking: practical_score desc, then norm_score desc, then expectancy desc",
            "- practical_score includes live-match bonus, trade-count bonus, and alpha bonuses vs SPY/sector",
        ]
        if walk_forward:
            lines.append(
                f"- walk_forward: enabled on shortlist={walk_forward_shortlist} with rolling_windows={walk_forward_windows}"
            )
            lines.append(
                "- walk_forward note: fixed-parameter rolling validation windows; not per-window re-optimized training"
            )
        lines.append("")
        lines.extend(
            self._render_report_section(
                "Top Ranked Candidates",
                top_ranked,
                candidate_links,
                gate_diagnostics,
            )
        )
        lines.extend(
            self._render_report_section(
                "Top Live Match Candidates",
                live_ranked,
                candidate_links,
                gate_diagnostics,
                empty_message="No strategies currently have live matches.",
            )
        )
        lines.extend(
            self._render_report_section(
                "Best Practical Live Candidates",
                practical_live_ranked,
                candidate_links,
                gate_diagnostics,
                empty_message="No strategies currently have live matches.",
            )
        )
        lines.extend(
            self._render_report_section(
                "Best Candidate Per Sector",
                self._best_per_sector(ranked),
                candidate_links,
                gate_diagnostics,
            )
        )
        lines.extend(
            self._render_report_section(
                "Best Promotable Candidate Per Sector",
                self._best_per_sector(ranked[ranked["promotion_policy_passed"]].copy()),
                candidate_links,
                gate_diagnostics,
                empty_message="No sectors currently satisfy promotion policy.",
            )
        )
        lines.extend(
            self._render_report_section(
                "Best Live Candidate Per Sector",
                self._best_per_sector(ranked[ranked["live_match_count"] > 0].copy()),
                candidate_links,
                gate_diagnostics,
                empty_message="No sectors currently have live matches.",
            )
        )
        lines.extend(
            self._render_pair_section(
                "Best Promotable Portfolio Pairs",
                self._build_portfolio_pairs(ranked[ranked["promotion_policy_passed"]].copy()),
                empty_message="No portfolio pairs currently satisfy promotion policy.",
            )
        )
        if walk_forward_available:
            stability_ranked = (
                self._cap_sector_repetition(
                    ranked[ranked["wf_stability_score"].notna()].sort_values(
                        ["wf_stability_score", "wf_positive_window_ratio", "wf_median_expectancy"],
                        ascending=[False, False, False],
                    ),
                    max_per_sector=2,
                    limit=top,
                )
            )
            lines.extend(
                self._render_report_section(
                    "Best Walk-Forward Stability Candidates",
                    stability_ranked,
                    candidate_links,
                    gate_diagnostics,
                    empty_message="No walk-forward shortlist metrics are available.",
                )
            )
        report_path.write_text("\n".join(lines), encoding="utf-8")
        return EvaluateReport(output_path=str(report_path), rows_written=len(top_ranked.index))

    def _build_live_candidate_metadata(
        self,
        ranked_frame: pd.DataFrame,
    ) -> tuple[dict[int, dict[str, list[str] | int]], dict[int, dict[str, object]]]:
        universe_rows = self.db_manager.list_research_universe(limit=250)
        if not universe_rows:
            return {}, {}
        universe_tickers = [row["ticker"] for row in universe_rows]
        tickers = sorted(set(universe_tickers).union(REFERENCE_TICKERS))
        history = self.db_manager.load_price_history(tickers)
        if history.empty:
            return {}, {}
        earnings_loader = getattr(self.db_manager, "load_earnings_calendar", None)
        earnings_calendar = earnings_loader(universe_tickers) if callable(earnings_loader) else pd.DataFrame()
        analysis_frame, _ = build_analysis_frame(
            history,
            universe_rows,
            earnings_calendar=earnings_calendar,
        )
        full_snapshot = latest_snapshot(analysis_frame)
        full_snapshot = full_snapshot[full_snapshot["ticker"].isin(universe_tickers)].copy()
        regime_snapshot = full_snapshot[full_snapshot["regime_green"].fillna(False)].copy()

        links: dict[int, dict[str, list[str] | int]] = {}
        diagnostics: dict[int, dict[str, object]] = {}
        signature_cache: dict[tuple[str, str], tuple[dict[str, list[str] | int], dict[str, object]]] = {}
        for row in ranked_frame.itertuples(index=False):
            params = json.loads(row.params_json)
            signature = self._metadata_signature(row.sector, params["indicators"])
            cached = signature_cache.get(signature)
            if cached is None:
                row_diagnostic = self._build_gate_diagnostic(
                    full_snapshot=full_snapshot,
                    regime_snapshot=regime_snapshot,
                    indicators=params["indicators"],
                    sector=row.sector,
                )
                candidates = filter_signal_candidates(regime_snapshot, params["indicators"])
                if row.sector != "ALL":
                    candidates = candidates[candidates["sector"] == row.sector]
                if "signal_score" not in candidates.columns:
                    candidates["signal_score"] = 0.0
                candidates = candidates.sort_values(
                    ["signal_score", "md_volume_30d", "ticker"],
                    ascending=[False, False, True],
                )
                live_match_count = len(candidates.index)
                top_candidates = candidates.head(5).copy()
                tickers = [candidate.ticker for candidate in top_candidates.itertuples(index=False)]
                row_links = {
                    "count": live_match_count,
                    "tickers": tickers,
                    "links": [
                        f"https://www.tradingview.com/chart/?symbol={candidate.ticker}"
                        for candidate in top_candidates.itertuples(index=False)
                    ],
                }
                cached = (row_links, row_diagnostic)
                signature_cache[signature] = cached
            row_links, row_diagnostic = cached
            links[int(row.id)] = row_links
            diagnostics[int(row.id)] = row_diagnostic
        return links, diagnostics

    def _build_walk_forward_stability(
        self,
        *,
        ranked: pd.DataFrame,
        top: int,
        shortlist_size: int,
        windows: int,
    ) -> pd.DataFrame:
        try:
            import polars as pl
        except ModuleNotFoundError as exc:
            raise RuntimeError("polars is required for walk-forward evaluation. Install project dependencies first.") from exc

        shortlist = self._build_walk_forward_shortlist(ranked, top=top, shortlist_size=shortlist_size)
        if shortlist.empty:
            return pd.DataFrame(columns=["id"])

        universe_rows = self.db_manager.list_research_universe(limit=250)
        if not universe_rows:
            return pd.DataFrame(columns=["id"])
        universe_tickers = [row["ticker"] for row in universe_rows]
        tickers = sorted(set(universe_tickers).union(REFERENCE_TICKERS))
        price_history = self.db_manager.load_price_history(tickers)
        if price_history.empty:
            return pd.DataFrame(columns=["id"])

        earnings_loader = getattr(self.db_manager, "load_earnings_calendar", None)
        earnings_calendar = earnings_loader(universe_tickers) if callable(earnings_loader) else pd.DataFrame()
        analysis_frame, _ = build_analysis_frame(price_history, universe_rows, earnings_calendar=earnings_calendar)
        analysis_frame = analysis_frame[analysis_frame["ticker"].isin(universe_tickers)].copy()
        if analysis_frame.empty:
            return pd.DataFrame(columns=["id"])

        pl_frame = pl.from_dicts(analysis_frame.to_dict(orient="records"))
        unique_dates = sorted(pl_frame.select("date").unique().to_series().to_list())
        if len(unique_dates) < max(windows + 1, 4):
            return pd.DataFrame(columns=["id"])

        date_chunks = [
            list(chunk)
            for chunk in np.array_split(np.array(unique_dates, dtype=object), windows + 1)
            if len(chunk) > 0
        ]
        test_chunks = date_chunks[1:]
        if len(test_chunks) < windows:
            return pd.DataFrame(columns=["id"])

        sweep_service = SweepService(self.db_manager)
        benchmark_price_maps = sweep_service._build_benchmark_price_maps(price_history)
        stability_rows: list[dict[str, float | int]] = []
        for row in shortlist.itertuples(index=False):
            params = json.loads(row.params_json)
            row_sector = params.get("sector", "ALL")
            backtest_costs = sweep_service._load_backtest_costs(
                {"backtest_costs": params.get("backtest_costs", {})}
            )

            window_metrics: list[dict[str, float]] = []
            for test_dates in test_chunks:
                start_date = test_dates[0]
                end_date = test_dates[-1]
                scoped_frame = pl_frame.filter((pl.col("date") >= start_date) & (pl.col("date") <= end_date))
                if row_sector != "ALL":
                    scoped_frame = scoped_frame.filter(pl.col("sector") == row_sector)
                metrics = sweep_service._run_backtest(
                    scoped_frame,
                    params["indicators"],
                    params["exit_rules"],
                    backtest_costs,
                    benchmark_context=BenchmarkContext(
                        spy_ticker="SPY",
                        sector_ticker=None if row_sector == "ALL" else self._benchmark_ticker_for_sector(row_sector),
                        price_maps=benchmark_price_maps,
                    ),
                )
                window_metrics.append(metrics)

            stability_rows.append({"id": int(row.id), **self._summarize_walk_forward_metrics(window_metrics)})
        return pd.DataFrame(stability_rows)

    def _build_walk_forward_shortlist(
        self,
        ranked: pd.DataFrame,
        *,
        top: int,
        shortlist_size: int,
    ) -> pd.DataFrame:
        shortlist_parts = [
            ranked.head(top),
            ranked[ranked["live_match_count"] > 0].head(top),
            self._best_per_sector(ranked).head(top),
            self._best_per_sector(ranked[ranked["promotion_policy_passed"]].copy()).head(top),
            ranked.sort_values(
                ["alpha_vs_sector", "alpha_vs_spy", "practical_score"],
                ascending=[False, False, False],
            ).head(top),
        ]
        shortlist = pd.concat(shortlist_parts, ignore_index=True).drop_duplicates(subset=["id"], keep="first")
        return shortlist.head(shortlist_size).reset_index(drop=True)

    def _summarize_walk_forward_metrics(self, window_metrics: list[dict[str, float]]) -> dict[str, float | int]:
        expectancy_values = [float(metrics["expectancy"]) for metrics in window_metrics]
        alpha_values = [
            float(metrics["alpha_vs_spy"])
            for metrics in window_metrics
            if metrics.get("alpha_vs_spy") is not None and np.isfinite(float(metrics["alpha_vs_spy"]))
        ]
        mdd_values = [float(metrics["mdd"]) for metrics in window_metrics]
        trade_counts = [int(metrics["trade_count"]) for metrics in window_metrics]

        positive_window_ratio = (
            sum(1 for value in expectancy_values if value > 0) / len(expectancy_values)
            if expectancy_values
            else 0.0
        )
        positive_alpha_window_ratio = (
            sum(1 for value in alpha_values if value > 0) / len(window_metrics)
            if window_metrics
            else 0.0
        )
        median_expectancy = float(np.median(expectancy_values)) if expectancy_values else 0.0
        worst_expectancy = float(min(expectancy_values)) if expectancy_values else 0.0
        median_alpha_vs_spy = float(np.median(alpha_values)) if alpha_values else np.nan
        worst_mdd = float(max(mdd_values)) if mdd_values else 0.0
        trade_count_min = min(trade_counts) if trade_counts else 0
        return {
            "wf_window_count": len(window_metrics),
            "wf_median_expectancy": median_expectancy,
            "wf_worst_expectancy": worst_expectancy,
            "wf_positive_window_ratio": positive_window_ratio,
            "wf_positive_alpha_window_ratio": positive_alpha_window_ratio,
            "wf_median_alpha_vs_spy": median_alpha_vs_spy,
            "wf_worst_mdd": worst_mdd,
            "wf_trade_count_min": trade_count_min,
            "wf_stability_score": self._score_walk_forward_summary(
                median_expectancy=median_expectancy,
                worst_expectancy=worst_expectancy,
                positive_window_ratio=positive_window_ratio,
                positive_alpha_window_ratio=positive_alpha_window_ratio,
                worst_mdd=worst_mdd,
                trade_count_min=trade_count_min,
            ),
        }

    def _score_walk_forward_summary(
        self,
        *,
        median_expectancy: float,
        worst_expectancy: float,
        positive_window_ratio: float,
        positive_alpha_window_ratio: float,
        worst_mdd: float,
        trade_count_min: int,
    ) -> float:
        expectancy_term = max(min(median_expectancy * 40.0, 0.40), -0.40)
        worst_expectancy_term = max(min(worst_expectancy * 30.0, 0.20), -0.20)
        trade_support_term = min(max(trade_count_min, 0), 20) / 20.0 * 0.10
        return (
            expectancy_term
            + worst_expectancy_term
            + positive_window_ratio * 0.30
            + positive_alpha_window_ratio * 0.15
            + trade_support_term
            - worst_mdd * 0.25
        )

    def _benchmark_ticker_for_sector(self, sector: str) -> str | None:
        from src.utils.regime import benchmark_etf_for_sector

        return benchmark_etf_for_sector(sector)

    def _build_gate_diagnostic(
        self,
        *,
        full_snapshot: pd.DataFrame,
        regime_snapshot: pd.DataFrame,
        indicators: dict[str, float],
        sector: str,
    ) -> dict[str, object]:
        counts: list[tuple[str, int]] = [
            ("universe", len(full_snapshot.index)),
            ("regime_green", len(regime_snapshot.index)),
        ]
        hard_filters, score_components, pass_score = split_signal_indicators(indicators)
        working = regime_snapshot.copy()
        if sector != "ALL":
            working = working[working["sector"] == sector].copy()
        counts.append(("sector_scope", len(working.index)))

        first_zero_gate: str | None = "sector_scope" if len(working.index) == 0 else None
        for threshold_name, threshold_value in hard_filters.items():
            feature_name = threshold_name[:-4] if threshold_name.endswith(("_min", "_max")) else threshold_name
            if threshold_name.endswith("_min"):
                mask = working[feature_name].astype(float) >= float(threshold_value)
            elif threshold_name.endswith("_max"):
                mask = working[feature_name].astype(float) <= float(threshold_value)
            else:
                mask = working[feature_name].astype(float) == float(threshold_value)
            working = working.loc[mask].copy()
            counts.append((threshold_name, len(working.index)))
            if first_zero_gate is None and len(working.index) == 0:
                first_zero_gate = threshold_name

        component_counter = {threshold_name: 0 for threshold_name in score_components}
        if not working.empty:
            passing_rows = 0
            for row in working.to_dict(orient="records"):
                passed, details, _ = evaluate_signal_gate(
                    {
                        **hard_filters,
                        **score_components,
                        SIGNAL_SCORE_MIN_KEY: pass_score,
                    },
                    row,
                )
                for threshold_name in score_components:
                    score = float(details[threshold_name]["score"])
                    if score > 0:
                        component_counter[threshold_name] += 1
                if passed:
                    passing_rows += 1
            counts.append((SIGNAL_SCORE_MIN_KEY, passing_rows))
            if first_zero_gate is None and passing_rows == 0:
                first_zero_gate = SIGNAL_SCORE_MIN_KEY
        else:
            counts.append((SIGNAL_SCORE_MIN_KEY, 0))
            if first_zero_gate is None:
                first_zero_gate = SIGNAL_SCORE_MIN_KEY

        return {
            "counts": counts,
            "first_zero_gate": first_zero_gate or "none",
            "component_positive_counts": list(component_counter.items()),
        }

    def _min_max(self, series: pd.Series) -> pd.Series:
        normalized = series.astype(float).copy()
        finite_mask = np.isfinite(normalized)
        finite_values = normalized[finite_mask]
        if finite_values.empty:
            return pd.Series([0.0] * len(normalized.index), index=normalized.index)
        if not finite_values.empty and (~finite_mask).any():
            finite_min = float(finite_values.min())
            finite_max = float(finite_values.max())
            spread = finite_max - finite_min
            replacement = finite_max + (spread if spread > 0 else 1.0)
            normalized.loc[~finite_mask] = replacement
        min_value = normalized.min()
        max_value = normalized.max()
        if min_value == max_value:
            return pd.Series([0.0] * len(normalized.index), index=normalized.index)
        return (normalized - min_value) / (max_value - min_value)

    def _build_warnings(
        self,
        *,
        profit_factor: float,
        trade_count: int | None,
        live_match_count: int,
    ) -> list[str]:
        warnings: list[str] = []
        if trade_count is not None and trade_count <= 12:
            warnings.append("low_trade_count")
        if math.isinf(profit_factor):
            warnings.append("profit_factor_infinite")
        elif profit_factor >= 20:
            warnings.append("profit_factor_outlier")
        if live_match_count == 0:
            warnings.append("no_current_live_matches")
        return warnings

    def _apply_practical_scoring(
        self,
        frame: pd.DataFrame,
        candidate_links: dict[int, dict[str, list[str] | int]],
    ) -> pd.DataFrame:
        scored = frame.copy()
        scored["live_match_count"] = scored["id"].map(
            lambda row_id: int(candidate_links.get(int(row_id), {}).get("count", 0))
        )
        scored["trade_count_value"] = scored["trade_count"].fillna(0).astype(int)
        scored["norm_alpha_vs_spy"] = self._min_max(scored["alpha_vs_spy"])
        scored["norm_alpha_vs_sector"] = self._min_max(scored["alpha_vs_sector"])
        scored["live_match_bonus"] = scored["live_match_count"].map(
            lambda count: 0.0 if int(count) <= 0 else 0.30 + (min(int(count), 5) - 1) * 0.05
        )
        scored["trade_count_bonus"] = (
            (scored["trade_count_value"].clip(lower=12, upper=30) - 12) / 18
        ) * 0.05
        scored["alpha_bonus"] = (
            scored["norm_alpha_vs_spy"] * 0.08
            + scored["norm_alpha_vs_sector"] * 0.12
        )
        scored["low_trade_penalty"] = scored["trade_count_value"].map(lambda count: 0.08 if count <= 12 else 0.0)
        scored["infinite_pf_penalty"] = scored["profit_factor"].map(lambda value: 0.08 if math.isinf(value) else 0.0)
        scored["no_live_penalty"] = scored["live_match_count"].map(lambda count: 0.12 if count == 0 else 0.0)
        scored["practical_score"] = (
            scored["norm_score"]
            + scored["alpha_bonus"]
            + scored["live_match_bonus"]
            + scored["trade_count_bonus"]
            - scored["low_trade_penalty"]
            - scored["infinite_pf_penalty"]
            - scored["no_live_penalty"]
        )
        return scored

    def _metadata_signature(self, sector: str, indicators: dict[str, float]) -> tuple[str, str]:
        return sector, json.dumps(indicators, sort_keys=True)

    def _dedupe_plateaus(self, frame: pd.DataFrame) -> pd.DataFrame:
        dedupe_frame = frame.copy()
        dedupe_frame["profit_factor_key"] = dedupe_frame["profit_factor"].map(self._metric_key)
        dedupe_frame["expectancy_key"] = dedupe_frame["expectancy"].round(12)
        dedupe_frame["mdd_key"] = dedupe_frame["mdd"].round(12)
        dedupe_frame["win_rate_key"] = dedupe_frame["win_rate"].round(12)
        dedupe_frame["trade_count_key"] = dedupe_frame["trade_count"].fillna(-1).astype(int)

        dedupe_keys = [
            "sector",
            "profit_factor_key",
            "expectancy_key",
            "mdd_key",
            "win_rate_key",
            "trade_count_key",
        ]
        dedupe_frame["duplicate_group_size"] = dedupe_frame.groupby(dedupe_keys)["id"].transform("count")
        collapsed_ids = (
            dedupe_frame.groupby(dedupe_keys)["id"]
            .agg(lambda values: [int(value) for value in values.tolist()])
            .rename("collapsed_result_ids")
            .reset_index()
        )
        dedupe_frame = dedupe_frame.merge(collapsed_ids, on=dedupe_keys, how="left")
        dedupe_frame = dedupe_frame.drop_duplicates(subset=dedupe_keys, keep="first").copy()
        return dedupe_frame.drop(
            columns=[
                "profit_factor_key",
                "expectancy_key",
                "mdd_key",
                "win_rate_key",
                "trade_count_key",
            ]
        ).reset_index(drop=True)

    def _metric_key(self, value: float) -> str:
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return f"{value:.12f}"

    def _best_per_sector(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame.copy()
        return (
            frame.sort_values(
                ["practical_score", "norm_score", "expectancy"],
                ascending=[False, False, False],
            )
            .drop_duplicates(subset=["sector"], keep="first")
            .reset_index(drop=True)
        )

    def _cap_sector_repetition(self, frame: pd.DataFrame, *, max_per_sector: int, limit: int) -> pd.DataFrame:
        if frame.empty:
            return frame.copy()
        selected_indices: list[int] = []
        sector_counts: dict[str, int] = {}
        for row in frame.reset_index(drop=True).itertuples():
            if len(selected_indices) >= limit:
                break
            sector = str(row.sector)
            if sector_counts.get(sector, 0) >= max_per_sector:
                continue
            selected_indices.append(int(row.Index))
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
        if not selected_indices:
            return frame.iloc[0:0].copy()
        return frame.reset_index(drop=True).loc[selected_indices].copy()

    def _build_portfolio_pairs(self, ranked: pd.DataFrame) -> pd.DataFrame:
        sector_best = self._best_per_sector(ranked)
        candidates = sector_best.to_dict(orient="records")
        pair_rows: list[dict[str, object]] = []
        for left, right in combinations(candidates, 2):
            left_trades = max(int(left["trade_count"]) if pd.notna(left["trade_count"]) else 0, 1)
            right_trades = max(int(right["trade_count"]) if pd.notna(right["trade_count"]) else 0, 1)
            total_trades = left_trades + right_trades
            weighted_expectancy = (
                (float(left["expectancy"]) * left_trades) + (float(right["expectancy"]) * right_trades)
            ) / total_trades
            weighted_profit_factor = (
                (float(left["profit_factor"]) * left_trades) + (float(right["profit_factor"]) * right_trades)
            ) / total_trades
            weighted_alpha_vs_spy = (
                (float(left["alpha_vs_spy"]) * left_trades) + (float(right["alpha_vs_spy"]) * right_trades)
            ) / total_trades
            sector_alpha_values = [
                (float(left["alpha_vs_sector"]), left_trades),
                (float(right["alpha_vs_sector"]), right_trades),
            ]
            finite_sector_alpha = [(value, weight) for value, weight in sector_alpha_values if np.isfinite(value)]
            weighted_alpha_vs_sector = (
                sum(value * weight for value, weight in finite_sector_alpha) / sum(weight for _, weight in finite_sector_alpha)
                if finite_sector_alpha
                else np.nan
            )
            worst_mdd = max(float(left["mdd"]), float(right["mdd"]))
            combined_live = int(left["live_match_count"]) + int(right["live_match_count"])
            pair_score = (
                (float(left["practical_score"]) + float(right["practical_score"])) / 2.0
                + min(combined_live, 6) * 0.02
                + 0.05
                - worst_mdd * 0.10
            )
            pair_rows.append(
                {
                    "left_id": int(left["id"]),
                    "right_id": int(right["id"]),
                    "left_sector": str(left["sector"]),
                    "right_sector": str(right["sector"]),
                    "combined_live_match_count": combined_live,
                    "combined_trade_count": total_trades,
                    "weighted_expectancy": weighted_expectancy,
                    "weighted_profit_factor": weighted_profit_factor,
                    "weighted_alpha_vs_spy": weighted_alpha_vs_spy,
                    "weighted_alpha_vs_sector": weighted_alpha_vs_sector,
                    "worst_mdd": worst_mdd,
                    "pair_score": pair_score,
                }
            )
        if not pair_rows:
            return pd.DataFrame()
        return pd.DataFrame(pair_rows).sort_values(
            ["pair_score", "combined_live_match_count", "weighted_expectancy"],
            ascending=[False, False, False],
        ).reset_index(drop=True)

    def _apply_promotion_policy_metadata(self, frame: pd.DataFrame) -> pd.DataFrame:
        enriched = frame.copy()
        enriched["promotion_policy_violations"] = enriched.apply(
            lambda row: promotion_policy_violations(
                profit_factor=float(row["profit_factor"]),
                expectancy=float(row["expectancy"]),
                mdd=float(row["mdd"]),
                trade_count=int(row["trade_count"]) if pd.notna(row["trade_count"]) else None,
            ),
            axis=1,
        )
        enriched["promotion_policy_passed"] = enriched["promotion_policy_violations"].map(lambda violations: len(violations) == 0)
        return enriched

    def _render_report_section(
        self,
        title: str,
        frame: pd.DataFrame,
        candidate_links: dict[int, dict[str, list[str] | int]],
        gate_diagnostics: dict[int, dict[str, object]],
        empty_message: str | None = None,
    ) -> list[str]:
        lines = [f"## {title}", ""]
        if frame.empty:
            lines.append(empty_message or "No results.")
            lines.append("")
            return lines
        for row in frame.itertuples(index=False):
            params = json.loads(row.params_json)
            candidate_info = candidate_links.get(int(row.id), {"links": [], "tickers": [], "count": 0})
            gate_debug = gate_diagnostics.get(
                int(row.id),
                {"counts": [], "first_zero_gate": "unavailable", "component_positive_counts": []},
            )
            trade_count = int(row.trade_count) if pd.notna(row.trade_count) else None
            warnings = self._build_warnings(
                profit_factor=row.profit_factor,
                trade_count=trade_count,
                live_match_count=int(candidate_info["count"]),
            )
            lines.append(f"### Result {row.id}")
            lines.append(f"- run_id: {row.run_id}")
            lines.append(f"- strategy_id: {row.strategy_id}")
            lines.append(f"- global_rank: {row.global_rank}")
            lines.append(f"- sector_rank: {row.sector_rank}")
            lines.append(f"- sector: {row.sector}")
            lines.append(f"- practical_score: {row.practical_score:.6f}")
            lines.append(f"- norm_score: {row.norm_score:.6f}")
            lines.append(f"- expectancy: {row.expectancy:.6f}")
            lines.append(f"- profit_factor: {row.profit_factor:.6f}")
            lines.append(
                f"- alpha_vs_spy: {row.alpha_vs_spy:.6f}" if np.isfinite(float(row.alpha_vs_spy)) else "- alpha_vs_spy: unknown"
            )
            lines.append(
                f"- alpha_vs_sector: {row.alpha_vs_sector:.6f}" if np.isfinite(float(row.alpha_vs_sector)) else "- alpha_vs_sector: unknown"
            )
            lines.append(f"- mdd: {row.mdd:.6f}")
            lines.append(f"- win_rate: {row.win_rate:.6f}")
            lines.append(f"- trade_count: {trade_count if trade_count is not None else 'unknown'}")
            if hasattr(row, "wf_stability_score") and pd.notna(getattr(row, "wf_stability_score", np.nan)):
                lines.append(f"- wf_stability_score: {float(row.wf_stability_score):.6f}")
                lines.append(f"- wf_window_count: {int(row.wf_window_count)}")
                lines.append(f"- wf_positive_window_ratio: {float(row.wf_positive_window_ratio):.6f}")
                lines.append(
                    f"- wf_positive_alpha_window_ratio: {float(row.wf_positive_alpha_window_ratio):.6f}"
                )
                lines.append(f"- wf_median_expectancy: {float(row.wf_median_expectancy):.6f}")
                lines.append(f"- wf_worst_expectancy: {float(row.wf_worst_expectancy):.6f}")
                lines.append(
                    f"- wf_median_alpha_vs_spy: {float(row.wf_median_alpha_vs_spy):.6f}"
                    if np.isfinite(float(row.wf_median_alpha_vs_spy))
                    else "- wf_median_alpha_vs_spy: unknown"
                )
                lines.append(f"- wf_worst_mdd: {float(row.wf_worst_mdd):.6f}")
                lines.append(f"- wf_trade_count_min: {int(row.wf_trade_count_min)}")
            lines.append(f"- duplicate_group_size: {row.duplicate_group_size}")
            lines.append(f"- collapsed_result_ids: {', '.join(str(value) for value in row.collapsed_result_ids)}")
            lines.append(f"- live_match_count: {candidate_info['count']}")
            lines.append(f"- live_match_tickers: {', '.join(candidate_info['tickers']) if candidate_info['tickers'] else 'none'}")
            lines.append(f"- promotion_policy_passed: {'yes' if bool(row.promotion_policy_passed) else 'no'}")
            lines.append(
                f"- promotion_policy_violations: {', '.join(row.promotion_policy_violations) if row.promotion_policy_violations else 'none'}"
            )
            if gate_debug["counts"]:
                counts_rendered = " -> ".join(
                    f"{label}={count}" for label, count in gate_debug["counts"]
                )
                lines.append(f"- gate_counts: {counts_rendered}")
                lines.append(f"- first_zero_gate: {gate_debug['first_zero_gate']}")
                if gate_debug["component_positive_counts"]:
                    support_rendered = ", ".join(
                        f"{label}={count}" for label, count in gate_debug["component_positive_counts"]
                    )
                    lines.append(f"- component_positive_counts: {support_rendered}")
            else:
                lines.append("- gate_counts: unavailable")
                lines.append("- first_zero_gate: unavailable")
                lines.append("- component_positive_counts: unavailable")
            lines.append(f"- warnings: {', '.join(warnings) if warnings else 'none'}")
            lines.append(f"- params: `{json.dumps(params, sort_keys=True)}`")
            if candidate_info["links"]:
                lines.append(f"- TradingView: {', '.join(candidate_info['links'])}")
            else:
                lines.append("- TradingView: none")
            lines.append("")
        return lines

    def _render_pair_section(
        self,
        title: str,
        frame: pd.DataFrame,
        empty_message: str | None = None,
    ) -> list[str]:
        lines = [f"## {title}", ""]
        if frame.empty:
            lines.append(empty_message or "No portfolio pairs available.")
            lines.append("")
            return lines
        for row in frame.head(10).itertuples(index=False):
            lines.append(f"### Pair {row.left_id} + {row.right_id}")
            lines.append(f"- sectors: {row.left_sector} + {row.right_sector}")
            lines.append(f"- pair_score: {row.pair_score:.6f}")
            lines.append(f"- combined_live_match_count: {row.combined_live_match_count}")
            lines.append(f"- combined_trade_count: {row.combined_trade_count}")
            lines.append(f"- weighted_expectancy: {row.weighted_expectancy:.6f}")
            lines.append(f"- weighted_profit_factor: {row.weighted_profit_factor:.6f}")
            lines.append(f"- weighted_alpha_vs_spy: {row.weighted_alpha_vs_spy:.6f}")
            lines.append(
                f"- weighted_alpha_vs_sector: {row.weighted_alpha_vs_sector:.6f}"
                if np.isfinite(float(row.weighted_alpha_vs_sector))
                else "- weighted_alpha_vs_sector: unknown"
            )
            lines.append(f"- worst_mdd: {row.worst_mdd:.6f}")
            lines.append("- note: heuristic pair score based on sector-best models; no covariance model is stored yet")
            lines.append("")
        return lines
