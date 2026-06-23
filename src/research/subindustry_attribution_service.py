from __future__ import annotations

from dataclasses import dataclass
import json
import math

import pandas as pd

from src.evaluate.service import EvaluateService
from src.sweep.service import BenchmarkContext, SweepService
from src.sync.service import REFERENCE_TICKERS
from src.utils.db_manager import DatabaseManager
from src.utils.logging import get_logger
from src.utils.regime import benchmark_etf_for_sector
from src.utils.signal_engine import build_analysis_frame, filter_signal_candidates, latest_snapshot


@dataclass(frozen=True)
class SubindustryAttributionReport:
    output_path: str
    strategy_id: int
    subindustries_written: int


class SubindustryAttributionService:
    def __init__(self, db_manager: DatabaseManager) -> None:
        self.db_manager = db_manager
        self.logger = get_logger("subindustry_attribution")

    def run(
        self,
        *,
        sector: str,
        run_id: int | None = None,
        strategy_id: int | None = None,
        min_trades: int = 12,
    ) -> SubindustryAttributionReport:
        self.db_manager.initialize()
        selected_run_id = run_id if run_id is not None else self.db_manager.latest_run_id()
        if selected_run_id is None:
            raise ValueError("No backtest results found. Run `sq sweep` first.")

        chosen = self._select_strategy_row(
            sector=sector,
            run_id=selected_run_id,
            strategy_id=strategy_id,
            min_trades=min_trades,
        )
        params = json.loads(chosen["params_json"])

        universe_rows = self.db_manager.list_research_universe(limit=250)
        if not universe_rows:
            raise ValueError("Universe is empty or liquidity metrics are unavailable. Run `sq sync` first.")
        universe_tickers = [row["ticker"] for row in universe_rows]
        tickers = sorted(set(universe_tickers).union(REFERENCE_TICKERS))
        price_history = self.db_manager.load_price_history(tickers)
        if price_history.empty:
            raise ValueError("Historical prices are unavailable. Run `sq sync` first.")
        earnings_loader = getattr(self.db_manager, "load_earnings_calendar", None)
        earnings_calendar = earnings_loader(universe_tickers) if callable(earnings_loader) else pd.DataFrame()
        analysis_frame, _ = build_analysis_frame(
            price_history,
            universe_rows,
            earnings_calendar=earnings_calendar,
        )
        if analysis_frame.empty:
            raise ValueError("No analysis frame could be built for the selected universe.")
        universe_analysis_frame = analysis_frame[analysis_frame["ticker"].isin(universe_tickers)].copy()
        if universe_analysis_frame.empty:
            raise ValueError("No analysis frame could be built for the selected universe.")

        historical_rows = self._build_historical_subindustry_rows(
            analysis_frame=universe_analysis_frame,
            benchmark_source_frame=analysis_frame,
            sector=sector,
            indicators=params["indicators"],
            exit_rules=params["exit_rules"],
            backtest_costs=params.get("backtest_costs", {}),
        )
        live_rows = self._build_live_subindustry_rows(
            analysis_frame=universe_analysis_frame,
            sector=sector,
            indicators=params["indicators"],
        )

        report_path = self.db_manager.paths.reports_dir / "subindustry_attribution.md"
        report_path.write_text(
            self._render_report(
                run_id=selected_run_id,
                sector=sector,
                chosen=chosen,
                params=params,
                historical_rows=historical_rows,
                live_rows=live_rows,
            ),
            encoding="utf-8",
        )
        return SubindustryAttributionReport(
            output_path=str(report_path),
            strategy_id=int(chosen["strategy_id"]),
            subindustries_written=len(historical_rows),
        )

    def _select_strategy_row(
        self,
        *,
        sector: str,
        run_id: int,
        strategy_id: int | None,
        min_trades: int,
    ) -> dict:
        result_rows = self.db_manager.list_backtest_results(run_id=run_id)
        if not result_rows:
            raise ValueError(f"No backtest results found for run_id={run_id}.")

        rows: list[dict] = []
        for row in result_rows:
            params = json.loads(row["params_json"])
            row_sector = params.get("sector", "ALL")
            if row_sector != sector:
                continue
            if strategy_id is not None and int(row["strategy_id"]) != int(strategy_id):
                continue
            rows.append(
                {
                    "id": int(row["id"]),
                    "run_id": int(row["run_id"]),
                    "strategy_id": int(row["strategy_id"]),
                    "params_json": row["params_json"],
                    "sector": row_sector,
                    "profit_factor": float(row["profit_factor"]),
                    "expectancy": float(row["expectancy"]),
                    "alpha_vs_spy": float(row["alpha_vs_spy"]) if row["alpha_vs_spy"] is not None else float("nan"),
                    "alpha_vs_sector": float(row["alpha_vs_sector"]) if row["alpha_vs_sector"] is not None else float("nan"),
                    "mdd": float(row["mdd"]),
                    "win_rate": float(row["win_rate"]),
                    "trade_count": int(row["trade_count"]) if row["trade_count"] is not None else None,
                }
            )
        if not rows:
            raise ValueError("No backtest results match the requested sector or strategy filter.")

        frame = pd.DataFrame(rows)
        if min_trades > 0:
            frame = frame[frame["trade_count"].notna() & (frame["trade_count"] >= min_trades)].copy()
        if frame.empty:
            raise ValueError(
                f"No backtest results remain after applying filters for run_id={run_id}, sector={sector}, and min_trades={min_trades}."
            )

        evaluator = EvaluateService(self.db_manager)
        frame["norm_expectancy"] = evaluator._min_max(frame["expectancy"])
        frame["norm_profit_factor"] = evaluator._min_max(frame["profit_factor"])
        frame["norm_max_drawdown"] = evaluator._min_max(frame["mdd"])
        frame["norm_score"] = (
            frame["norm_expectancy"] * 0.4
            + frame["norm_profit_factor"] * 0.3
            - frame["norm_max_drawdown"] * 0.3
        )
        deduped = evaluator._dedupe_plateaus(frame)
        candidate_links, _ = evaluator._build_live_candidate_metadata(deduped)
        ranked = evaluator._apply_practical_scoring(deduped, candidate_links).sort_values(
            ["practical_score", "norm_score", "expectancy"],
            ascending=[False, False, False],
        )
        return ranked.iloc[0].to_dict()

    def _build_historical_subindustry_rows(
        self,
        *,
        analysis_frame: pd.DataFrame,
        benchmark_source_frame: pd.DataFrame,
        sector: str,
        indicators: dict[str, float],
        exit_rules: dict[str, float | None],
        backtest_costs: dict[str, float],
    ) -> list[dict]:
        try:
            import polars as pl
        except ModuleNotFoundError as exc:
            raise RuntimeError("polars is required for subindustry attribution. Install project dependencies first.") from exc

        sector_frame = analysis_frame[analysis_frame["sector"] == sector].copy()
        if sector_frame.empty:
            return []

        unique_dates = sorted(pd.to_datetime(sector_frame["date"]).dropna().unique())
        if len(unique_dates) < 2:
            return []
        split_index = min(max(int(len(unique_dates) * 0.7), 1), len(unique_dates) - 1)
        validation_start = unique_dates[split_index]
        validation_frame = sector_frame[pd.to_datetime(sector_frame["date"]) >= validation_start].copy()

        sweep_service = SweepService(self.db_manager)
        benchmark_price_maps = sweep_service._build_benchmark_price_maps(benchmark_source_frame)
        pl_frame = pl.from_dicts(validation_frame.to_dict(orient="records"))

        subindustries = sorted(
            subindustry
            for subindustry in validation_frame["sub_industry"].dropna().astype(str).unique().tolist()
            if subindustry.strip()
        )
        rows: list[dict] = []
        for subindustry in subindustries:
            scoped_frame = pl_frame.filter(pl.col("sub_industry") == subindustry)
            metrics = sweep_service._run_backtest(
                scoped_frame,
                indicators,
                exit_rules,
                sweep_service._load_backtest_costs({"backtest_costs": backtest_costs}),
                benchmark_context=BenchmarkContext(
                    spy_ticker="SPY",
                    sector_ticker=benchmark_etf_for_sector(sector),
                    price_maps=benchmark_price_maps,
                ),
            )
            rows.append(
                {
                    "sub_industry": subindustry,
                    "trade_count": int(metrics["trade_count"]),
                    "expectancy": float(metrics["expectancy"]),
                    "profit_factor": float(metrics["profit_factor"]),
                    "alpha_vs_sector": self._optional_float(metrics.get("alpha_vs_sector")),
                    "win_rate": float(metrics["win_rate"]),
                    "mdd": float(metrics["mdd"]),
                }
            )
        rows.sort(
            key=lambda row: (
                row["alpha_vs_sector"] if row["alpha_vs_sector"] is not None else float("-inf"),
                row["expectancy"],
                row["trade_count"],
            ),
            reverse=True,
        )
        return rows

    def _build_live_subindustry_rows(
        self,
        *,
        analysis_frame: pd.DataFrame,
        sector: str,
        indicators: dict[str, float],
    ) -> list[dict]:
        snapshot = latest_snapshot(analysis_frame)
        snapshot = snapshot[
            snapshot["ticker"].isin(analysis_frame["ticker"].unique())
            & snapshot["regime_green"].fillna(False)
            & (snapshot["sector"] == sector)
        ].copy()
        if snapshot.empty:
            return []

        candidates = filter_signal_candidates(snapshot, indicators)
        if candidates.empty:
            return []
        if "signal_score" not in candidates.columns:
            candidates["signal_score"] = 0.0

        rows: list[dict] = []
        for subindustry, group in candidates.groupby("sub_industry", dropna=False):
            sorted_group = group.sort_values(
                ["signal_score", "md_volume_30d", "ticker"],
                ascending=[False, False, True],
            )
            tickers = sorted_group["ticker"].astype(str).head(5).tolist()
            rows.append(
                {
                    "sub_industry": str(subindustry) if pd.notna(subindustry) and str(subindustry).strip() else "Unknown",
                    "live_match_count": len(sorted_group.index),
                    "tickers": tickers,
                    "avg_signal_score": float(sorted_group["signal_score"].mean()),
                }
            )
        rows.sort(key=lambda row: (row["live_match_count"], row["avg_signal_score"]), reverse=True)
        return rows

    def _render_report(
        self,
        *,
        run_id: int,
        sector: str,
        chosen: dict,
        params: dict,
        historical_rows: list[dict],
        live_rows: list[dict],
    ) -> str:
        lines = [
            "# Subindustry Attribution",
            "",
            f"- run_id: {run_id}",
            f"- sector: {sector}",
            f"- strategy_id: {int(chosen['strategy_id'])}",
            f"- selected_by: practical_score desc, then norm_score desc, then expectancy desc",
            f"- sweep_mode: {params.get('sweep_mode', 'unknown')}",
            "",
            "## Chosen Strategy",
            "",
            f"- expectancy: {float(chosen['expectancy']):.6f}",
            f"- profit_factor: {float(chosen['profit_factor']):.6f}",
            f"- alpha_vs_sector: {self._format_optional(chosen.get('alpha_vs_sector'))}",
            f"- mdd: {float(chosen['mdd']):.6f}",
            f"- trade_count: {int(chosen['trade_count']) if chosen.get('trade_count') is not None else 0}",
            f"- params: `{json.dumps(params, sort_keys=True)}`",
            "",
            "## Historical By Subindustry",
            "",
        ]
        if not historical_rows:
            lines.append("No historical subindustry rows were available.")
        else:
            for row in historical_rows:
                lines.extend(
                    [
                        f"### {row['sub_industry']}",
                        f"- trade_count: {row['trade_count']}",
                        f"- expectancy: {row['expectancy']:.6f}",
                        f"- alpha_vs_sector: {self._format_optional(row['alpha_vs_sector'])}",
                        f"- profit_factor: {row['profit_factor']:.6f}",
                        f"- win_rate: {row['win_rate']:.4f}",
                        f"- mdd: {row['mdd']:.6f}",
                        "",
                    ]
                )

        lines.extend(
            [
                "## Live Candidates By Subindustry",
                "",
            ]
        )
        if not live_rows:
            lines.append("No current live candidates passed the chosen strategy.")
        else:
            for row in live_rows:
                lines.extend(
                    [
                        f"### {row['sub_industry']}",
                        f"- live_match_count: {row['live_match_count']}",
                        f"- avg_signal_score: {row['avg_signal_score']:.2f}",
                        f"- tickers: {', '.join(row['tickers']) if row['tickers'] else 'none'}",
                        "",
                    ]
                )
        return "\n".join(lines)

    def _optional_float(self, value) -> float | None:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        return numeric if math.isfinite(numeric) else None

    def _format_optional(self, value) -> str:
        numeric = self._optional_float(value)
        return f"{numeric:.6f}" if numeric is not None else "n/a"
