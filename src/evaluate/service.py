from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import json
import math

import numpy as np
import pandas as pd

from src.sync.service import REFERENCE_TICKERS
from src.utils.db_manager import DatabaseManager
from src.utils.logging import get_logger
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

    def run(self, *, top: int = 10, sector: str | None = None, run_id: int | None = None, min_trades: int = 12) -> EvaluateReport:
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

        ranked = deduped.sort_values(
            ["practical_score", "norm_score", "expectancy"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
        ranked["global_rank"] = range(1, len(ranked.index) + 1)
        ranked["sector_rank"] = ranked.groupby("sector")["practical_score"].rank(method="dense", ascending=False).astype(int)
        top_ranked = ranked.head(top).copy()
        live_ranked = (
            ranked[ranked["live_match_count"] > 0]
            .sort_values(
                ["live_match_count", "practical_score", "norm_score"],
                ascending=[False, False, False],
            )
            .head(top)
            .copy()
        )
        practical_live_ranked = (
            ranked[ranked["live_match_count"] > 0]
            .sort_values(
                ["practical_score", "live_match_count", "norm_score"],
                ascending=[False, False, False],
            )
            .head(top)
            .copy()
        )
        report_path = self.db_manager.paths.reports_dir / "candidates.md"
        lines = [
            "# Ranked Candidates",
            "",
            f"- run_id: {selected_run_id}",
            f"- min_trades: {min_trades}",
            "- ranking: practical_score desc, then norm_score desc, then expectancy desc",
            "",
        ]
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
                "Best Live Candidate Per Sector",
                self._best_per_sector(ranked[ranked["live_match_count"] > 0].copy()),
                candidate_links,
                gate_diagnostics,
                empty_message="No sectors currently have live matches.",
            )
        )
        lines.extend(
            self._render_pair_section(
                "Best Portfolio Pairs",
                self._build_portfolio_pairs(ranked),
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

        analysis_frame, _ = build_analysis_frame(history, universe_rows)
        full_snapshot = latest_snapshot(analysis_frame)
        full_snapshot = full_snapshot[full_snapshot["ticker"].isin(universe_tickers)].copy()
        regime_snapshot = full_snapshot[full_snapshot["regime_green"].fillna(False)].copy()

        links: dict[int, dict[str, list[str] | int]] = {}
        diagnostics: dict[int, dict[str, object]] = {}
        for row in ranked_frame.itertuples(index=False):
            params = json.loads(row.params_json)
            diagnostics[int(row.id)] = self._build_gate_diagnostic(
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
            links[int(row.id)] = {
                "count": live_match_count,
                "tickers": tickers,
                "links": [
                    f"https://www.tradingview.com/chart/?symbol={candidate.ticker}"
                    for candidate in top_candidates.itertuples(index=False)
                ],
            }
        return links, diagnostics

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
        scored["live_match_bonus"] = scored["live_match_count"].map(
            lambda count: 0.0 if int(count) <= 0 else 0.30 + (min(int(count), 5) - 1) * 0.05
        )
        scored["trade_count_bonus"] = (
            (scored["trade_count_value"].clip(lower=12, upper=30) - 12) / 18
        ) * 0.05
        scored["low_trade_penalty"] = scored["trade_count_value"].map(lambda count: 0.08 if count <= 12 else 0.0)
        scored["infinite_pf_penalty"] = scored["profit_factor"].map(lambda value: 0.08 if math.isinf(value) else 0.0)
        scored["no_live_penalty"] = scored["live_match_count"].map(lambda count: 0.12 if count == 0 else 0.0)
        scored["practical_score"] = (
            scored["norm_score"]
            + scored["live_match_bonus"]
            + scored["trade_count_bonus"]
            - scored["low_trade_penalty"]
            - scored["infinite_pf_penalty"]
            - scored["no_live_penalty"]
        )
        return scored

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
            lines.append(f"- mdd: {row.mdd:.6f}")
            lines.append(f"- win_rate: {row.win_rate:.6f}")
            lines.append(f"- trade_count: {trade_count if trade_count is not None else 'unknown'}")
            lines.append(f"- duplicate_group_size: {row.duplicate_group_size}")
            lines.append(f"- collapsed_result_ids: {', '.join(str(value) for value in row.collapsed_result_ids)}")
            lines.append(f"- live_match_count: {candidate_info['count']}")
            lines.append(f"- live_match_tickers: {', '.join(candidate_info['tickers']) if candidate_info['tickers'] else 'none'}")
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
            lines.append(f"- worst_mdd: {row.worst_mdd:.6f}")
            lines.append("- note: heuristic pair score based on sector-best models; no covariance model is stored yet")
            lines.append("")
        return lines
