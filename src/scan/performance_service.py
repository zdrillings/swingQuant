from __future__ import annotations

from dataclasses import dataclass
from html import escape
import json
import math

import pandas as pd

from src.settings import get_settings
from src.utils.db_manager import DatabaseManager
from src.utils.emailer import send_html_email
from src.utils.logging import get_logger
from src.utils.regime import benchmark_etf_for_sector


DEFAULT_PERFORMANCE_HORIZONS = (1, 2, 3, 5, 10, 20, 60)


@dataclass(frozen=True)
class ScanPerformanceReport:
    output_path: str
    selected_rows: int
    scan_dates: int
    benchmark: str


class ScanPerformanceService:
    def __init__(
        self,
        db_manager: DatabaseManager,
        *,
        email_sender=send_html_email,
    ) -> None:
        self.db_manager = db_manager
        self.email_sender = email_sender
        self.logger = get_logger("scan_performance")

    def run(
        self,
        *,
        recent_scan_dates: int = 0,
        recent_picks: int = 20,
        benchmark: str = "sector",
        selection_source: str | None = None,
        model_name: str | None = None,
        model_generated_at: str | None = None,
        latest_model_only: bool = True,
        horizons: tuple[int, ...] = DEFAULT_PERFORMANCE_HORIZONS,
        email: bool = False,
    ) -> ScanPerformanceReport:
        self.db_manager.initialize()
        settings = get_settings()
        benchmark = self._normalize_benchmark(benchmark)
        candidates = self.db_manager.load_scan_candidates()
        if candidates.empty:
            raise ValueError("No scan snapshots found. Run `sq scan` or `sq scan-backfill` first.")

        selected = candidates[candidates["selected"].astype(int) == 1].copy()
        if selected.empty:
            raise ValueError("No selected scan picks found. Run `sq scan` or `sq scan-backfill` first.")
        selected, resolved_scope = self._resolve_scope(
            selected,
            latest_model_only=latest_model_only,
            selection_source=selection_source,
            model_name=model_name,
            model_generated_at=model_generated_at,
        )
        selected = self._filter_selected(
            selected,
            selection_source=resolved_scope["selection_source"],
            model_name=resolved_scope["model_name"],
            model_generated_at=resolved_scope["model_generated_at"],
        )
        if selected.empty:
            raise ValueError("No selected scan picks matched the requested source/model filters.")

        selected["scan_date"] = pd.to_datetime(selected["scan_date"]).dt.normalize()
        unique_dates = sorted(selected["scan_date"].drop_duplicates().tolist())
        if int(recent_scan_dates) > 0:
            scoped_dates = unique_dates[-int(recent_scan_dates) :]
            window_label = str(len(scoped_dates))
        else:
            scoped_dates = unique_dates
            window_label = "all"
        selected = selected[selected["scan_date"].isin(scoped_dates)].copy().reset_index(drop=True)
        if selected.empty:
            raise ValueError("No selected scan picks matched the requested recent window.")

        enriched = self._attach_outcomes(selected, horizons=horizons)
        self._persist_selected_outcomes(enriched)

        report_path = self.db_manager.paths.reports_dir / "scan_performance.md"
        lines = [
            "# Scan Performance",
            "",
            f"- benchmark: {benchmark}",
            f"- scope: {resolved_scope['scope']}",
            f"- selection_source: {resolved_scope['selection_source'] or 'all'}",
            f"- model_name: {resolved_scope['model_name'] or 'all'}",
            f"- model_generated_at: {resolved_scope['model_generated_at'] or 'all'}",
            f"- latest_model_generated_at: {resolved_scope.get('latest_model_generated_at') or 'n/a'}",
            f"- recent_scan_dates: {window_label}",
            f"- scan_dates: {len(scoped_dates)}",
            f"- selected_rows: {len(enriched.index)}",
            f"- scan_date_min: {min(scoped_dates).date()}",
            f"- scan_date_max: {max(scoped_dates).date()}",
            "",
        ]
        lines.extend(self._render_selection_source_coverage(enriched))
        lines.extend(self._render_latest_model_selection_audit(candidates))
        lines.extend(self._render_horizon_summary(enriched, horizons=horizons, benchmark=benchmark))
        lines.extend(self._render_20d_timeframe_summary(enriched, benchmark=benchmark))
        lines.extend(self._render_20d_score_bands(enriched, benchmark=benchmark))
        lines.extend(self._render_market_turn_diagnostics(enriched, benchmark=benchmark))
        lines.extend(self._render_portfolio_performance())
        lines.extend(self._render_best_and_worst_picks(enriched, horizons=horizons, benchmark=benchmark))
        lines.extend(self._render_repeated_winners_and_losers(enriched, horizons=horizons, benchmark=benchmark))
        lines.extend(self._render_recent_scan_dates(enriched, horizons=horizons, benchmark=benchmark))
        lines.extend(self._render_recent_picks(enriched, horizons=horizons, benchmark=benchmark, recent_picks=recent_picks))
        report_text = "\n".join(lines)
        report_path.write_text(report_text, encoding="utf-8")
        if email:
            dashboard = self._build_performance_dashboard(
                enriched=enriched,
                horizons=horizons,
                benchmark=benchmark,
                resolved_scope=resolved_scope,
                window_label=window_label,
            )
            forward_predictions = self._load_forward_predictions()
            self.email_sender(
                subject=f"SwingQuant Performance ({benchmark})",
                html_body=self._render_performance_email(enriched, dashboard, horizons, benchmark, forward_predictions),
                settings=settings,
            )

        return ScanPerformanceReport(
            output_path=str(report_path),
            selected_rows=len(enriched.index),
            scan_dates=len(scoped_dates),
            benchmark=benchmark,
        )

    def _persist_selected_outcomes(self, enriched: pd.DataFrame) -> None:
        if enriched.empty or not hasattr(self.db_manager, "update_scan_candidate_outcomes"):
            return
        required_columns = {"scan_date", "ticker", "strategy_slot"}
        if not required_columns.issubset(enriched.columns):
            return
        outcome_columns = [
            "fwd_return_1d",
            "fwd_return_3d",
            "fwd_return_5d",
            "fwd_return_10d",
            "fwd_return_20d",
            "alpha_vs_spy_1d",
            "alpha_vs_spy_3d",
            "alpha_vs_spy_5d",
            "alpha_vs_spy_10d",
            "alpha_vs_spy_20d",
            "alpha_vs_sector_1d",
            "alpha_vs_sector_3d",
            "alpha_vs_sector_5d",
            "alpha_vs_sector_10d",
            "alpha_vs_sector_20d",
            "mfe_20d",
            "mae_20d",
        ]
        present_outcome_columns = [column for column in outcome_columns if column in enriched.columns]
        if not present_outcome_columns:
            return
        for scan_date, group in enriched.groupby("scan_date", sort=True):
            rows = []
            for row in group.to_dict(orient="records"):
                payload = {
                    "ticker": row["ticker"],
                    "strategy_slot": row["strategy_slot"],
                }
                for column in present_outcome_columns:
                    value = row.get(column)
                    payload[column] = None if pd.isna(value) else value
                rows.append(payload)
            if rows:
                self.db_manager.update_scan_candidate_outcomes(
                    scan_date=str(pd.Timestamp(scan_date).date()),
                    rows=rows,
                )

    def _resolve_scope(
        self,
        selected: pd.DataFrame,
        *,
        latest_model_only: bool,
        selection_source: str | None,
        model_name: str | None,
        model_generated_at: str | None,
    ) -> tuple[pd.DataFrame, dict[str, str | None]]:
        explicit_filter = any(value not in (None, "") for value in (selection_source, model_name, model_generated_at))
        if explicit_filter or not latest_model_only:
            return selected.copy(), {
                "scope": "explicit" if explicit_filter else "all",
                "selection_source": selection_source,
                "model_name": model_name,
                "model_generated_at": model_generated_at,
            }
        required_columns = {"selection_source", "model_name", "model_generated_at"}
        if not required_columns.issubset(selected.columns):
            return selected.copy(), {
                "scope": "all",
                "selection_source": None,
                "model_name": None,
                "model_generated_at": None,
            }
        model_rows = selected[
            (selected["selection_source"].astype(str) == "shortlist_model")
            & selected["model_generated_at"].notna()
            & selected["model_name"].notna()
        ].copy()
        if model_rows.empty:
            return selected.copy(), {
                "scope": "all",
                "selection_source": None,
                "model_name": None,
                "model_generated_at": None,
            }
        latest_generated_at = str(sorted(model_rows["model_generated_at"].astype(str).unique())[-1])
        latest_rows = model_rows[model_rows["model_generated_at"].astype(str) == latest_generated_at].copy()
        model_names = sorted(latest_rows["model_name"].astype(str).unique())
        latest_model_name = model_names[0]
        return selected.copy(), {
            "scope": "latest_model_family",
            "selection_source": "shortlist_model",
            "model_name": latest_model_name,
            "model_generated_at": None,
            "latest_model_generated_at": latest_generated_at,
        }

    def _filter_selected(
        self,
        selected: pd.DataFrame,
        *,
        selection_source: str | None,
        model_name: str | None,
        model_generated_at: str | None,
    ) -> pd.DataFrame:
        filtered = selected.copy()
        filters = {
            "selection_source": selection_source,
            "model_name": model_name,
            "model_generated_at": model_generated_at,
        }
        for column, value in filters.items():
            if value in (None, ""):
                continue
            if column not in filtered.columns:
                return filtered.iloc[0:0].copy()
            filtered = filtered[filtered[column].astype(str) == str(value)].copy()
        return filtered

    def _render_selection_source_coverage(self, enriched: pd.DataFrame) -> list[str]:
        lines = ["## Selection Source Coverage", ""]
        if enriched.empty or "selection_source" not in enriched.columns:
            lines.extend(["No selection source metadata is available.", ""])
            return lines
        counts = (
            enriched["selection_source"]
            .fillna("unknown")
            .astype(str)
            .replace({"": "unknown"})
            .value_counts()
            .sort_index()
        )
        total = int(counts.sum())
        for source, count in counts.items():
            pct = (float(count) / total) if total else 0.0
            lines.append(f"- {source}: {int(count)} ({pct:.1%})")
        if "shortlist_model" not in counts.index:
            lines.append("- warning: no selected picks in this report window are model-attributed.")
        lines.append("")
        return lines

    def _render_latest_model_selection_audit(self, candidates: pd.DataFrame) -> list[str]:
        lines = ["## Latest Model Selection Audit", ""]
        required = {"scan_date", "ticker", "selected", "selection_source", "model_rank"}
        if candidates.empty or not required.issubset(candidates.columns):
            lines.extend(["No model selection audit metadata is available.", ""])
            return lines
        working = candidates[candidates["selection_source"].astype(str) == "shortlist_model"].copy()
        if working.empty:
            lines.extend(["No model-attributed candidates are available.", ""])
            return lines
        working["scan_date"] = pd.to_datetime(working["scan_date"]).dt.normalize()
        latest_date = working["scan_date"].max()
        latest = working[working["scan_date"] == latest_date].copy()
        latest["model_rank"] = pd.to_numeric(latest["model_rank"], errors="coerce")
        selected = latest[latest["selected"].astype(int) == 1].sort_values(["selected_rank", "model_rank", "ticker"])
        if selected.empty:
            lines.extend([f"- scan_date: {latest_date.date()}", "- selected_model_picks: 0", ""])
            return lines
        worst_selected_model_rank = pd.to_numeric(selected["model_rank"], errors="coerce").max()
        skipped = latest[
            (latest["selected"].astype(int) == 0)
            & latest["model_rank"].notna()
            & (latest["model_rank"].astype(float) < float(worst_selected_model_rank))
        ].sort_values(["model_rank", "ticker"])
        lines.append(f"- scan_date: {latest_date.date()}")
        lines.append(
            "- note: final picks are chosen after opportunity floor, recent rotation, current holdings, and portfolio caps; this section surfaces model-rank divergences."
        )
        lines.append("- selected:")
        for row in selected.head(10).itertuples(index=False):
            lines.append(
                f"  - {row.ticker}: selected_rank={self._fmt_int(getattr(row, 'selected_rank', None))}, "
                f"model_rank={self._fmt_int(getattr(row, 'model_rank', None))}, "
                f"slot={getattr(row, 'strategy_slot', 'n/a')}, "
                f"selection_score={self._fmt_score(getattr(row, 'selection_score', float('nan')))}, "
                f"opportunity={self._fmt_score(getattr(row, 'opportunity_score', float('nan')))}"
            )
        if skipped.empty:
            lines.append("- higher_model_rank_unselected: none")
        else:
            lines.append("- higher_model_rank_unselected:")
            for row in skipped.head(10).itertuples(index=False):
                lines.append(
                    f"  - {row.ticker}: model_rank={self._fmt_int(getattr(row, 'model_rank', None))}, "
                    f"slot={getattr(row, 'strategy_slot', 'n/a')}, "
                    f"selection_score={self._fmt_score(getattr(row, 'selection_score', float('nan')))}, "
                    f"opportunity={self._fmt_score(getattr(row, 'opportunity_score', float('nan')))}"
                )
        lines.append("")
        return lines

    def _normalize_benchmark(self, benchmark: str) -> str:
        normalized = str(benchmark or "sector").strip().lower()
        if normalized not in {"sector", "spy"}:
            raise ValueError("benchmark must be one of: sector, spy")
        return normalized

    def _attach_outcomes(self, frame: pd.DataFrame, *, horizons: tuple[int, ...]) -> pd.DataFrame:
        tickers = sorted(set(frame["ticker"].astype(str)).union({"SPY"}))
        sector_benchmarks = {
            benchmark_etf_for_sector(str(row.get("sector") or row.get("strategy_sector") or ""))
            for row in frame.to_dict(orient="records")
        }
        tickers = sorted(set(tickers).union({ticker for ticker in sector_benchmarks if ticker}))
        history = self.db_manager.load_price_history(tickers)
        if history.empty:
            raise ValueError("Historical prices are unavailable. Run `sq sync` first.")
        history_context = self._history_context(history)
        rows: list[dict[str, object]] = []
        for row in frame.to_dict(orient="records"):
            payload = dict(row)
            outcomes = self._row_outcomes(row, history_context=history_context, horizons=horizons)
            payload.update(outcomes)
            rows.append(payload)
        return pd.DataFrame(rows)

    def _history_context(self, history: pd.DataFrame) -> dict[str, dict[str, object]]:
        working = history.copy()
        working["date"] = pd.to_datetime(working["date"]).dt.normalize()
        context: dict[str, dict[str, object]] = {}
        for ticker, group in working.groupby("ticker", sort=False):
            ordered = group.sort_values("date").reset_index(drop=True)
            context[str(ticker)] = {
                "frame": ordered,
                "index_by_date": {
                    pd.Timestamp(date_value).normalize().strftime("%Y-%m-%d"): int(index)
                    for index, date_value in enumerate(ordered["date"])
                },
            }
        return context

    def _row_outcomes(
        self,
        row: dict,
        *,
        history_context: dict[str, dict[str, object]],
        horizons: tuple[int, ...],
    ) -> dict[str, float]:
        ticker = str(row["ticker"])
        scan_date = str(pd.Timestamp(row["scan_date"]).date())
        ticker_context = history_context.get(ticker)
        if ticker_context is None:
            return {}
        ticker_frame = ticker_context["frame"]
        index = ticker_context["index_by_date"].get(scan_date)
        if index is None:
            return {}
        benchmark_ticker = benchmark_etf_for_sector(str(row.get("sector") or row.get("strategy_sector") or ""))
        payload: dict[str, float] = {}
        for horizon in horizons:
            payload[f"fwd_return_{horizon}d"] = self._forward_return(
                ticker_frame=ticker_frame,
                index=int(index),
                horizon=int(horizon),
            )
            payload[f"alpha_vs_spy_{horizon}d"] = self._alpha_vs_benchmark(
                history_context=history_context,
                ticker_frame=ticker_frame,
                scan_date=scan_date,
                index=int(index),
                horizon=int(horizon),
                benchmark_ticker="SPY",
            )
            payload[f"alpha_vs_sector_{horizon}d"] = self._alpha_vs_benchmark(
                history_context=history_context,
                ticker_frame=ticker_frame,
                scan_date=scan_date,
                index=int(index),
                horizon=int(horizon),
                benchmark_ticker=benchmark_ticker,
            )
        return payload

    def _forward_return(self, *, ticker_frame: pd.DataFrame, index: int, horizon: int) -> float:
        future_index = index + int(horizon)
        if future_index >= len(ticker_frame.index):
            return float("nan")
        entry_price = float(ticker_frame.loc[index, "adj_close"])
        future_price = float(ticker_frame.loc[future_index, "adj_close"])
        return (future_price / entry_price) - 1.0

    def _alpha_vs_benchmark(
        self,
        *,
        history_context: dict[str, dict[str, object]],
        ticker_frame: pd.DataFrame,
        scan_date: str,
        index: int,
        horizon: int,
        benchmark_ticker: str | None,
    ) -> float:
        raw_return = self._forward_return(ticker_frame=ticker_frame, index=index, horizon=horizon)
        if benchmark_ticker in (None, "") or not math.isfinite(raw_return):
            return float("nan")
        benchmark_context = history_context.get(str(benchmark_ticker))
        if benchmark_context is None:
            return float("nan")
        benchmark_index = benchmark_context["index_by_date"].get(scan_date)
        if benchmark_index is None:
            return float("nan")
        benchmark_return = self._forward_return(
            ticker_frame=benchmark_context["frame"],
            index=int(benchmark_index),
            horizon=horizon,
        )
        if not math.isfinite(benchmark_return):
            return float("nan")
        return raw_return - benchmark_return

    def _render_horizon_summary(
        self,
        frame: pd.DataFrame,
        *,
        horizons: tuple[int, ...],
        benchmark: str,
    ) -> list[str]:
        lines = ["## Horizon Summary", ""]
        for horizon in horizons:
            return_column = f"fwd_return_{horizon}d"
            alpha_column = f"alpha_vs_{benchmark}_{horizon}d"
            lines.append(f"### {horizon}d")
            scoped = self._matured_outcome_frame(
                frame,
                return_column=return_column,
                alpha_column=alpha_column,
            )
            if scoped.empty:
                lines.append("- matured_picks: 0")
                lines.append("")
                continue
            returns = pd.to_numeric(scoped[return_column], errors="coerce").dropna()
            alphas = pd.to_numeric(scoped[alpha_column], errors="coerce").dropna()
            matured_dates = int(scoped["scan_date"].nunique())
            latest_selected_scan_date = pd.to_datetime(frame["scan_date"]).dt.normalize().max()
            latest_matured_scan_date = pd.to_datetime(scoped["scan_date"]).dt.normalize().max()
            return_p25 = float(returns.quantile(0.25))
            return_p75 = float(returns.quantile(0.75))
            return_p05 = float(returns.quantile(0.05))
            return_p95 = float(returns.quantile(0.95))
            alpha_p25 = float(alphas.quantile(0.25))
            alpha_p75 = float(alphas.quantile(0.75))
            lines.append(f"- matured_picks: {len(scoped.index)}")
            lines.append(f"- matured_scan_dates: {matured_dates}")
            lines.append(f"- latest_selected_scan_date: {latest_selected_scan_date.date()}")
            lines.append(f"- latest_matured_scan_date: {latest_matured_scan_date.date()}")
            lines.append(f"- mean_return: {self._fmt_pct(returns.mean())}")
            lines.append(f"- median_return: {self._fmt_pct(returns.median())}")
            lines.append(f"- return_iqr: {self._fmt_pct(return_p25)} to {self._fmt_pct(return_p75)}")
            lines.append(f"- return_p05_p95: {self._fmt_pct(return_p05)} to {self._fmt_pct(return_p95)}")
            lines.append(f"- return_range: {self._fmt_pct(returns.min())} to {self._fmt_pct(returns.max())}")
            lines.append(f"- hit_rate: {self._fmt_pct((returns > 0.0).mean())}")
            lines.append(f"- mean_alpha_vs_{benchmark}: {self._fmt_pct(alphas.mean())}")
            lines.append(f"- median_alpha_vs_{benchmark}: {self._fmt_pct(alphas.median())}")
            lines.append(f"- alpha_iqr: {self._fmt_pct(alpha_p25)} to {self._fmt_pct(alpha_p75)}")
            lines.append(f"- positive_alpha_rate: {self._fmt_pct((alphas > 0.0).mean())}")
            lines.append("")
        return lines

    def _render_20d_timeframe_summary(
        self,
        frame: pd.DataFrame,
        *,
        benchmark: str,
    ) -> list[str]:
        lines = ["## 20d Timeframe Summary", ""]
        return_column = "fwd_return_20d"
        alpha_column = f"alpha_vs_{benchmark}_20d"
        required_columns = ["scan_date", return_column, alpha_column]
        if any(column not in frame.columns for column in required_columns):
            lines.append("- observations: 0")
            lines.append("- note: 20d return or alpha is unavailable.")
            lines.append("")
            return lines
        scoped = frame.dropna(subset=required_columns).copy()
        if scoped.empty:
            lines.append("- observations: 0")
            lines.append("")
            return lines
        scoped["scan_date"] = pd.to_datetime(scoped["scan_date"]).dt.normalize()
        scoped[return_column] = pd.to_numeric(scoped[return_column], errors="coerce")
        scoped[alpha_column] = pd.to_numeric(scoped[alpha_column], errors="coerce")
        scoped = scoped.dropna(subset=[return_column, alpha_column]).copy()
        if scoped.empty:
            lines.append("- observations: 0")
            lines.append("")
            return lines
        anchor = scoped["scan_date"].max()
        latest_selected_scan_date = pd.to_datetime(frame["scan_date"]).dt.normalize().max()
        lines.append(f"- latest_selected_scan_date: {latest_selected_scan_date.date()}")
        lines.append(f"- latest_matured_20d_scan_date: {anchor.date()}")
        lines.append("- note: 20d summaries are anchored to scan dates with a full 20 trading sessions of forward data.")
        lines.append("")
        windows = [
            ("1y", anchor - pd.DateOffset(years=1)),
            ("ytd", pd.Timestamp(year=int(anchor.year), month=1, day=1)),
            ("3m", anchor - pd.DateOffset(months=3)),
            ("20d", anchor - pd.DateOffset(days=20)),
        ]
        for label, start_date in windows:
            window = scoped[scoped["scan_date"] >= pd.Timestamp(start_date).normalize()].copy()
            lines.append(f"### {label}")
            lines.append(f"- start_date: {pd.Timestamp(start_date).date()}")
            lines.append(f"- end_date: {anchor.date()}")
            if window.empty:
                lines.append("- matured_picks: 0")
                lines.append("")
                continue
            returns = window[return_column].astype(float)
            alphas = window[alpha_column].astype(float)
            lines.append(f"- matured_picks: {len(window.index)}")
            lines.append(f"- matured_scan_dates: {int(window['scan_date'].nunique())}")
            lines.append(f"- mean_return: {self._fmt_pct(returns.mean())}")
            lines.append(f"- median_return: {self._fmt_pct(returns.median())}")
            lines.append(f"- hit_rate: {self._fmt_pct((returns > 0.0).mean())}")
            lines.append(f"- mean_alpha_vs_{benchmark}: {self._fmt_pct(alphas.mean())}")
            lines.append(f"- median_alpha_vs_{benchmark}: {self._fmt_pct(alphas.median())}")
            lines.append(f"- positive_alpha_rate: {self._fmt_pct((alphas > 0.0).mean())}")
            lines.append("")
        return lines

    def _render_20d_score_bands(
        self,
        frame: pd.DataFrame,
        *,
        benchmark: str,
    ) -> list[str]:
        lines = ["## 20d Opportunity Score Bands", ""]
        return_column = "fwd_return_20d"
        alpha_column = f"alpha_vs_{benchmark}_20d"
        required_columns = ["opportunity_score", return_column, alpha_column]
        if any(column not in frame.columns for column in required_columns):
            lines.append("- observations: 0")
            lines.append("- note: opportunity score, 20d return, or 20d alpha is unavailable.")
            lines.append("")
            return lines
        scoped = frame.dropna(subset=required_columns).copy()
        if scoped.empty:
            lines.append("- observations: 0")
            lines.append("")
            return lines
        scoped["opportunity_score"] = pd.to_numeric(scoped["opportunity_score"], errors="coerce")
        scoped[return_column] = pd.to_numeric(scoped[return_column], errors="coerce")
        scoped[alpha_column] = pd.to_numeric(scoped[alpha_column], errors="coerce")
        scoped = scoped.dropna(subset=["opportunity_score", return_column, alpha_column]).copy()
        if scoped.empty:
            lines.append("- observations: 0")
            lines.append("")
            return lines
        lines.append("- score: opportunity_score")
        lines.append(f"- return: {return_column}")
        lines.append(f"- alpha: {alpha_column}")
        lines.append(f"- observations: {len(scoped.index)}")
        lines.append("")
        for label, lower, upper in self._opportunity_score_bands():
            band = scoped[scoped["opportunity_score"].astype(float) >= lower].copy()
            if upper is not None:
                band = band[band["opportunity_score"].astype(float) < upper].copy()
            if band.empty:
                lines.append(f"- {label}: n=0, pick_share=0.00%")
                continue
            returns = band[return_column].astype(float)
            alpha = band[alpha_column].astype(float)
            pick_share = len(band.index) / len(scoped.index)
            lines.append(
                f"- {label}: n={len(band.index)}, "
                f"pick_share={self._fmt_pct(pick_share)}, "
                f"mean_return={self._fmt_pct(returns.mean())}, "
                f"median_return={self._fmt_pct(returns.median())}, "
                f"hit_rate={self._fmt_pct((returns > 0.0).mean())}, "
                f"mean_alpha={self._fmt_pct(alpha.mean())}, "
                f"median_alpha={self._fmt_pct(alpha.median())}, "
                f"positive_alpha_rate={self._fmt_pct((alpha > 0.0).mean())}"
            )
        lines.append("")
        return lines

    def _render_portfolio_performance(self) -> list[str]:
        lines = ["## Portfolio Performance", ""]
        if not hasattr(self.db_manager, "list_closed_trades") or not hasattr(self.db_manager, "list_open_trades"):
            lines.append("- note: ledger helpers are unavailable.")
            lines.append("")
            return lines
        closed_trades = [dict(row) for row in self.db_manager.list_closed_trades()]
        open_trades = [dict(row) for row in self.db_manager.list_open_trades()]
        if not closed_trades and not open_trades:
            lines.append("- trades: 0")
            lines.append("")
            return lines

        exit_ohlc = self._exit_ohlc_for_closed_trades(closed_trades)
        realized_rows = []
        suspect_closed_trades = 0
        for trade in closed_trades:
            entry_price = self._coerce_float(trade.get("entry_price"))
            exit_price = self._coerce_float(trade.get("exit_price"))
            shares = self._coerce_int(trade.get("shares"))
            if not (math.isfinite(entry_price) and entry_price > 0 and math.isfinite(exit_price) and shares > 0):
                continue
            if not self._closed_trade_exit_is_plausible(trade, exit_price=exit_price, exit_ohlc=exit_ohlc):
                suspect_closed_trades += 1
                continue
            cost = entry_price * shares
            pnl = (exit_price - entry_price) * shares
            realized_rows.append(
                {
                    "ticker": str(trade.get("ticker") or ""),
                    "entry_date": str(trade.get("entry_date") or ""),
                    "exit_date": str(trade.get("exit_date") or ""),
                    "shares": shares,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "cost": cost,
                    "pnl": pnl,
                    "return_pct": (exit_price / entry_price) - 1.0,
                }
            )

        latest_prices = self._latest_prices_for_open_trades(open_trades)
        unrealized_rows = []
        for trade in open_trades:
            ticker = str(trade.get("ticker") or "")
            entry_price = self._coerce_float(trade.get("entry_price"))
            shares = self._coerce_int(trade.get("shares"))
            price_context = latest_prices.get(ticker)
            current_price = self._coerce_float(price_context.get("price") if price_context else None)
            if not (math.isfinite(entry_price) and entry_price > 0 and math.isfinite(current_price) and shares > 0):
                unrealized_rows.append(
                    {
                        "ticker": ticker,
                        "entry_date": str(trade.get("entry_date") or ""),
                        "shares": shares,
                        "entry_price": entry_price,
                        "current_price": current_price,
                        "price_date": price_context.get("date") if price_context else "n/a",
                        "cost": float("nan"),
                        "pnl": float("nan"),
                        "return_pct": float("nan"),
                    }
                )
                continue
            cost = entry_price * shares
            pnl = (current_price - entry_price) * shares
            unrealized_rows.append(
                {
                    "ticker": ticker,
                    "entry_date": str(trade.get("entry_date") or ""),
                    "shares": shares,
                    "entry_price": entry_price,
                    "current_price": current_price,
                    "price_date": price_context.get("date") if price_context else "n/a",
                    "cost": cost,
                    "pnl": pnl,
                    "return_pct": (current_price / entry_price) - 1.0,
                }
            )

        realized_pnl = sum(row["pnl"] for row in realized_rows if math.isfinite(float(row["pnl"])))
        realized_cost = sum(row["cost"] for row in realized_rows if math.isfinite(float(row["cost"])))
        unrealized_pnl = sum(row["pnl"] for row in unrealized_rows if math.isfinite(float(row["pnl"])))
        unrealized_cost = sum(row["cost"] for row in unrealized_rows if math.isfinite(float(row["cost"])))
        total_cost = realized_cost + unrealized_cost
        total_pnl = realized_pnl + unrealized_pnl

        lines.append(f"- closed_trades: {len(realized_rows)}")
        if suspect_closed_trades:
            lines.append(f"- suspect_closed_trades_skipped: {suspect_closed_trades}")
        lines.append(f"- open_trades: {len(open_trades)}")
        lines.append(f"- realized_pnl: {self._fmt_dollars(realized_pnl)}")
        lines.append(f"- realized_return_on_cost: {self._fmt_pct(realized_pnl / realized_cost) if realized_cost > 0 else 'n/a'}")
        lines.append(f"- unrealized_pnl: {self._fmt_dollars(unrealized_pnl)}")
        lines.append(f"- unrealized_return_on_cost: {self._fmt_pct(unrealized_pnl / unrealized_cost) if unrealized_cost > 0 else 'n/a'}")
        lines.append(f"- total_pnl: {self._fmt_dollars(total_pnl)}")
        lines.append(f"- total_return_on_cost: {self._fmt_pct(total_pnl / total_cost) if total_cost > 0 else 'n/a'}")
        lines.append("")

        lines.extend(self._render_portfolio_realized_rows(realized_rows))
        lines.extend(self._render_portfolio_unrealized_rows(unrealized_rows))
        return lines

    def _exit_ohlc_for_closed_trades(self, closed_trades: list[dict]) -> dict[tuple[str, str], dict[str, float]]:
        if not closed_trades or not hasattr(self.db_manager, "load_price_history"):
            return {}
        tickers = sorted({str(trade.get("ticker") or "") for trade in closed_trades if trade.get("ticker")})
        if not tickers:
            return {}
        history = self.db_manager.load_price_history(tickers)
        if history.empty or not {"ticker", "date", "low", "high"}.issubset(history.columns):
            return {}
        working = history.copy()
        working["date"] = pd.to_datetime(working["date"]).dt.normalize()
        result: dict[tuple[str, str], dict[str, float]] = {}
        for row in working.to_dict(orient="records"):
            ticker = str(row.get("ticker") or "")
            date_value = str(pd.Timestamp(row["date"]).date())
            low = self._coerce_float(row.get("low"))
            high = self._coerce_float(row.get("high"))
            if math.isfinite(low) and math.isfinite(high) and low > 0 and high >= low:
                result[(ticker, date_value)] = {"low": low, "high": high}
        return result

    def _closed_trade_exit_is_plausible(
        self,
        trade: dict,
        *,
        exit_price: float,
        exit_ohlc: dict[tuple[str, str], dict[str, float]],
        tolerance_pct: float = 0.01,
    ) -> bool:
        ticker = str(trade.get("ticker") or "")
        exit_date = str(trade.get("exit_date") or "")
        if not ticker or not exit_date:
            return True
        ohlc = exit_ohlc.get((ticker, exit_date))
        if ohlc is None:
            return True
        low = float(ohlc["low"])
        high = float(ohlc["high"])
        return (low * (1.0 - float(tolerance_pct))) <= float(exit_price) <= (high * (1.0 + float(tolerance_pct)))

    def _latest_prices_for_open_trades(self, open_trades: list[dict]) -> dict[str, dict[str, object]]:
        tickers = sorted({str(trade.get("ticker") or "") for trade in open_trades if trade.get("ticker")})
        if not tickers or not hasattr(self.db_manager, "load_price_history"):
            return {}
        history = self.db_manager.load_price_history(tickers)
        if history.empty:
            return {}
        working = history.copy()
        working["date"] = pd.to_datetime(working["date"]).dt.normalize()
        result: dict[str, dict[str, object]] = {}
        for ticker, group in working.groupby("ticker", sort=False):
            ordered = group.sort_values("date")
            latest = ordered.iloc[-1]
            result[str(ticker)] = {
                "date": pd.Timestamp(latest["date"]).date(),
                "price": self._coerce_float(latest.get("adj_close")),
            }
        return result

    def _render_portfolio_realized_rows(self, rows: list[dict]) -> list[str]:
        lines = ["### Realized By Stock"]
        if not rows:
            lines.append("No closed trades.")
            lines.append("")
            return lines
        frame = pd.DataFrame(rows)
        grouped = (
            frame.groupby("ticker", as_index=False)
            .agg(
                trades=("ticker", "count"),
                realized_pnl=("pnl", "sum"),
                cost=("cost", "sum"),
                mean_return=("return_pct", "mean"),
                min_return=("return_pct", "min"),
                max_return=("return_pct", "max"),
                first_entry=("entry_date", "min"),
                last_exit=("exit_date", "max"),
            )
            .sort_values(["realized_pnl", "ticker"], ascending=[False, True])
        )
        for row in grouped.itertuples(index=False):
            return_on_cost = float(row.realized_pnl) / float(row.cost) if float(row.cost) > 0 else float("nan")
            lines.append(
                f"- {row.ticker}: trades={int(row.trades)}, "
                f"realized_pnl={self._fmt_dollars(row.realized_pnl)}, "
                f"return_on_cost={self._fmt_pct(return_on_cost)}, "
                f"mean_trade_return={self._fmt_pct(row.mean_return)}, "
                f"range={self._fmt_pct(row.min_return)} to {self._fmt_pct(row.max_return)}, "
                f"first_entry={row.first_entry}, last_exit={row.last_exit}"
            )
        lines.append("")
        return lines

    def _render_portfolio_unrealized_rows(self, rows: list[dict]) -> list[str]:
        lines = ["### Unrealized Open Positions"]
        if not rows:
            lines.append("No open trades.")
            lines.append("")
            return lines
        ordered = sorted(rows, key=lambda row: (float(row["pnl"]) if math.isfinite(float(row["pnl"])) else float("-inf")), reverse=True)
        for row in ordered:
            lines.append(
                f"- {row['ticker']}: shares={row['shares']}, "
                f"entry_date={row['entry_date']}, "
                f"entry={self._fmt_price(row['entry_price'])}, "
                f"latest={self._fmt_price(row['current_price'])} ({row['price_date']}), "
                f"unrealized_pnl={self._fmt_dollars(row['pnl'])}, "
                f"return={self._fmt_pct(row['return_pct'])}"
            )
        lines.append("")
        return lines

    def _opportunity_score_bands(self) -> list[tuple[str, float, float | None]]:
        return [
            ("score < 0.30", float("-inf"), 0.30),
            ("0.30 <= score < 0.35", 0.30, 0.35),
            ("0.35 <= score < 0.40", 0.35, 0.40),
            ("0.40 <= score < 0.45", 0.40, 0.45),
            ("0.45 <= score < 0.50", 0.45, 0.50),
            ("score >= 0.50", 0.50, None),
        ]

    def _render_market_turn_diagnostics(
        self,
        frame: pd.DataFrame,
        *,
        benchmark: str,
    ) -> list[str]:
        lines = ["## Market Turn Diagnostics", ""]
        lines.append("- purpose: surface stale-leadership risk without hiding candidates from the scanner")
        latest_scan_date = pd.to_datetime(frame["scan_date"], errors="coerce").dropna().max()
        if pd.isna(latest_scan_date):
            lines.append("- latest_scan_date: n/a")
            lines.append("")
            return lines
        latest_scan_date = latest_scan_date.normalize()
        latest = frame[pd.to_datetime(frame["scan_date"], errors="coerce").dt.normalize() == latest_scan_date].copy()
        lines.append(f"- latest_scan_date: {latest_scan_date.date()}")
        lines.append("")
        lines.extend(self._render_market_etf_snapshot())
        lines.extend(self._render_latest_selection_concentration(latest))
        lines.extend(self._render_rs_deterioration_snapshot(latest))
        lines.extend(self._render_20d_rs_deterioration_bands(frame, benchmark=benchmark))
        return lines

    def _render_market_etf_snapshot(self) -> list[str]:
        lines = ["### Market ETF Snapshot"]
        if not hasattr(self.db_manager, "load_price_history"):
            lines.append("- note: price history helper is unavailable.")
            lines.append("")
            return lines
        tickers = ["SPY", "QQQ", "XLK", "SMH", "XLI", "XLB", "XLV"]
        history = self.db_manager.load_price_history(tickers)
        if history.empty:
            lines.append("- observations: 0")
            lines.append("")
            return lines
        working = history.copy()
        working["date"] = pd.to_datetime(working["date"], errors="coerce").dt.normalize()
        working["adj_close"] = pd.to_numeric(working["adj_close"], errors="coerce")
        rows: list[str] = []
        latest_date = working["date"].dropna().max()
        if pd.isna(latest_date):
            lines.append("- observations: 0")
            lines.append("")
            return lines
        lines.append(f"- latest_price_date: {latest_date.date()}")
        for ticker in tickers:
            group = working[working["ticker"].astype(str) == ticker].dropna(subset=["date", "adj_close"]).sort_values("date")
            if group.empty:
                continue
            latest_index = len(group.index) - 1
            parts = [ticker]
            for horizon in (5, 10, 20):
                prior_index = latest_index - horizon
                if prior_index < 0:
                    parts.append(f"{horizon}d=n/a")
                    continue
                latest_price = float(group.iloc[latest_index]["adj_close"])
                prior_price = float(group.iloc[prior_index]["adj_close"])
                value = (latest_price / prior_price) - 1.0 if prior_price else float("nan")
                parts.append(f"{horizon}d={self._fmt_pct(value)}")
            rows.append("- " + ", ".join(parts))
        lines.extend(rows if rows else ["- observations: 0"])
        lines.append("")
        return lines

    def _render_latest_selection_concentration(self, latest: pd.DataFrame) -> list[str]:
        lines = ["### Latest Selection Concentration"]
        if latest.empty:
            lines.append("- selected_picks: 0")
            lines.append("")
            return lines
        lines.append(f"- selected_picks: {len(latest.index)}")
        for column, label in [("sector", "sector"), ("strategy_slot", "slot")]:
            if column not in latest.columns:
                continue
            counts = latest[column].fillna("unknown").astype(str).value_counts()
            if counts.empty:
                continue
            summary = ", ".join(f"{name}={int(count)}" for name, count in counts.items())
            lines.append(f"- by_{label}: {summary}")
        if "opportunity_score" in latest.columns:
            values = pd.to_numeric(latest["opportunity_score"], errors="coerce").dropna()
            if not values.empty:
                lines.append(f"- median_opportunity_score: {float(values.median()):.4f}")
                lines.append(f"- opportunity_ge_0_40: {int((values >= 0.40).sum())}/{len(values.index)}")
                lines.append(f"- opportunity_ge_0_45: {int((values >= 0.45).sum())}/{len(values.index)}")
                lines.append(f"- opportunity_ge_0_50: {int((values >= 0.50).sum())}/{len(values.index)}")
        lines.append("")
        return lines

    def _render_rs_deterioration_snapshot(self, latest: pd.DataFrame) -> list[str]:
        lines = ["### Latest RS Deterioration"]
        if latest.empty:
            lines.append("- observations: 0")
            lines.append("")
            return lines
        enriched = self._attach_feature_snapshot_columns(
            latest,
            [
                "relative_strength_index_vs_spy",
                "relative_strength_index_vs_subindustry",
                "rs_vs_spy_5d_change",
                "rs_vs_subindustry_5d_change",
                "rs_vs_subindustry_10d_change",
            ],
        )
        if "rs_vs_spy_5d_change" not in enriched.columns and "rs_vs_subindustry_5d_change" not in enriched.columns:
            lines.append("- observations: 0")
            lines.append("- note: RS change fields are unavailable until universe snapshots are refreshed.")
            lines.append("")
            return lines
        spy_change = pd.to_numeric(enriched.get("rs_vs_spy_5d_change"), errors="coerce")
        group_change = pd.to_numeric(enriched.get("rs_vs_subindustry_5d_change"), errors="coerce")
        deterioration = spy_change.le(-10.0).fillna(False) | group_change.le(-10.0).fillna(False)
        severe = spy_change.le(-20.0).fillna(False) | group_change.le(-20.0).fillna(False)
        observed = spy_change.notna() | group_change.notna()
        lines.append(f"- observations: {int(observed.sum())}/{len(enriched.index)}")
        if int(observed.sum()) == 0:
            lines.append("- note: RS change fields are unavailable until universe snapshots are refreshed.")
            lines.append("")
            return lines
        lines.append(f"- rs_deteriorating_ge_10pts: {int(deterioration.sum())}/{len(enriched.index)}")
        lines.append(f"- rs_deteriorating_ge_20pts: {int(severe.sum())}/{len(enriched.index)}")
        risky = enriched.loc[deterioration].copy()
        if not risky.empty:
            risky["_worst_rs_change"] = pd.concat([spy_change, group_change], axis=1).min(axis=1)
            risky = risky.sort_values(["_worst_rs_change", "ticker"], ascending=[True, True]).head(5)
            lines.append("- watch_only_candidates:")
            for row in risky.itertuples(index=False):
                lines.append(
                    f"  - {row.ticker}: "
                    f"rs_vs_spy_5d_change={self._fmt_points(getattr(row, 'rs_vs_spy_5d_change', float('nan')))}, "
                    f"rs_vs_group_5d_change={self._fmt_points(getattr(row, 'rs_vs_subindustry_5d_change', float('nan')))}, "
                    f"opportunity={self._fmt_score(getattr(row, 'opportunity_score', float('nan')))}"
                )
        lines.append("")
        return lines

    def _render_20d_rs_deterioration_bands(
        self,
        frame: pd.DataFrame,
        *,
        benchmark: str,
    ) -> list[str]:
        lines = ["### Matured 20d Outcomes By RS Change"]
        return_column = "fwd_return_20d"
        alpha_column = f"alpha_vs_{benchmark}_20d"
        enriched = self._attach_feature_snapshot_columns(
            frame,
            ["rs_vs_spy_5d_change", "rs_vs_subindustry_5d_change"],
        )
        required_columns = [return_column, alpha_column, "rs_vs_spy_5d_change", "rs_vs_subindustry_5d_change"]
        if any(column not in enriched.columns for column in required_columns):
            lines.append("- observations: 0")
            lines.append("- note: RS change fields are unavailable until universe snapshots are refreshed.")
            lines.append("")
            return lines
        scoped = enriched.copy()
        scoped[return_column] = pd.to_numeric(scoped[return_column], errors="coerce")
        scoped[alpha_column] = pd.to_numeric(scoped[alpha_column], errors="coerce")
        scoped["rs_vs_spy_5d_change"] = pd.to_numeric(scoped["rs_vs_spy_5d_change"], errors="coerce")
        scoped["rs_vs_subindustry_5d_change"] = pd.to_numeric(scoped["rs_vs_subindustry_5d_change"], errors="coerce")
        scoped = scoped.dropna(subset=[return_column, alpha_column]).copy()
        scoped = scoped[scoped["rs_vs_spy_5d_change"].notna() | scoped["rs_vs_subindustry_5d_change"].notna()].copy()
        if scoped.empty:
            lines.append("- observations: 0")
            lines.append("- note: RS change fields are unavailable until universe snapshots are refreshed.")
            lines.append("")
            return lines
        scoped["worst_rs_5d_change"] = scoped[["rs_vs_spy_5d_change", "rs_vs_subindustry_5d_change"]].min(axis=1)
        bands = [
            ("severe deterioration <= -20pts", float("-inf"), -20.0),
            ("deterioration -20pts to -10pts", -20.0, -10.0),
            ("stable -10pts to +10pts", -10.0, 10.0),
            ("improving >= +10pts", 10.0, float("inf")),
        ]
        lines.append(f"- observations: {len(scoped.index)}")
        for label, lower, upper in bands:
            band = scoped[(scoped["worst_rs_5d_change"] >= lower) & (scoped["worst_rs_5d_change"] < upper)].copy()
            if band.empty:
                lines.append(f"- {label}: n=0, pick_share=0.00%")
                continue
            returns = band[return_column].astype(float)
            alphas = band[alpha_column].astype(float)
            lines.append(
                f"- {label}: n={len(band.index)}, "
                f"pick_share={self._fmt_pct(len(band.index) / len(scoped.index))}, "
                f"median_return={self._fmt_pct(returns.median())}, "
                f"hit_rate={self._fmt_pct((returns > 0.0).mean())}, "
                f"median_alpha={self._fmt_pct(alphas.median())}, "
                f"positive_alpha_rate={self._fmt_pct((alphas > 0.0).mean())}"
            )
        lines.append("")
        return lines

    def _attach_feature_snapshot_columns(self, frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        working = frame.copy()
        if "details_json" not in working.columns:
            return self._merge_universe_snapshot_features(working, columns)
        parsed = working["details_json"].map(self._parse_details_json)
        snapshots = parsed.map(lambda payload: payload.get("feature_snapshot", {}) if isinstance(payload.get("feature_snapshot"), dict) else {})
        for column in columns:
            if column in working.columns and working[column].notna().any():
                continue
            working[column] = snapshots.map(lambda payload, key=column: payload.get(key))
        missing_columns = [
            column
            for column in columns
            if column not in working.columns or not pd.to_numeric(working[column], errors="coerce").notna().any()
        ]
        if missing_columns:
            working = self._merge_universe_snapshot_features(working, missing_columns)
        return working

    def _merge_universe_snapshot_features(self, frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        if not columns or not hasattr(self.db_manager, "load_universe_daily_snapshots"):
            return frame
        required = {"scan_date", "ticker"}
        if not required.issubset(frame.columns):
            return frame
        try:
            snapshots = self.db_manager.load_universe_daily_snapshots()
        except Exception as exc:
            self.logger.warning("Unable to load universe snapshots for market-turn diagnostics: %s", exc)
            return frame
        if snapshots.empty:
            return frame
        available_columns = ["snapshot_date", "ticker"] + [column for column in columns if column in snapshots.columns]
        if len(available_columns) <= 2:
            return frame
        left = frame.copy()
        left["_snapshot_date_key"] = pd.to_datetime(left["scan_date"], errors="coerce").dt.normalize()
        left["_ticker_key"] = left["ticker"].astype(str)
        right = snapshots[available_columns].copy()
        right["_snapshot_date_key"] = pd.to_datetime(right["snapshot_date"], errors="coerce").dt.normalize()
        right["_ticker_key"] = right["ticker"].astype(str)
        right = right.drop(columns=["snapshot_date", "ticker"])
        merged = left.merge(right, on=["_snapshot_date_key", "_ticker_key"], how="left", suffixes=("", "__snapshot"))
        for column in columns:
            snapshot_column = f"{column}__snapshot"
            if snapshot_column not in merged.columns:
                continue
            if column not in merged.columns:
                merged[column] = merged[snapshot_column]
            else:
                merged[column] = merged[column].where(merged[column].notna(), merged[snapshot_column])
            merged = merged.drop(columns=[snapshot_column])
        return merged.drop(columns=["_snapshot_date_key", "_ticker_key"])

    def _parse_details_json(self, value) -> dict[str, object]:
        if isinstance(value, dict):
            return value
        if value in (None, ""):
            return {}
        try:
            parsed = json.loads(str(value))
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def _render_recent_scan_dates(
        self,
        frame: pd.DataFrame,
        *,
        horizons: tuple[int, ...],
        benchmark: str,
    ) -> list[str]:
        lines = ["## Recent Scan Dates", ""]
        recent_dates = sorted(frame["scan_date"].drop_duplicates().tolist())[-10:]
        for scan_date in reversed(recent_dates):
            day_frame = frame[frame["scan_date"] == scan_date].copy()
            tickers = ", ".join(day_frame.sort_values("selected_rank")["ticker"].astype(str).tolist())
            lines.append(f"### {scan_date.date()}")
            lines.append(f"- picks: {tickers}")
            for horizon in horizons:
                return_column = f"fwd_return_{horizon}d"
                alpha_column = f"alpha_vs_{benchmark}_{horizon}d"
                scoped = self._matured_outcome_frame(
                    day_frame,
                    return_column=return_column,
                    alpha_column=alpha_column,
                )
                if scoped.empty:
                    continue
                returns = pd.to_numeric(scoped[return_column], errors="coerce").dropna()
                alphas = pd.to_numeric(scoped[alpha_column], errors="coerce").dropna()
                winners = int((returns > 0.0).sum())
                pick_count = int(len(returns.index))
                lines.append(
                    f"- {horizon}d: median_return={self._fmt_pct(returns.median())}, "
                    f"median_alpha_vs_{benchmark}={self._fmt_pct(alphas.median())}, "
                    f"winners={winners}/{pick_count}, "
                    f"range={self._fmt_pct(returns.min())} to {self._fmt_pct(returns.max())}"
                )
            lines.append("")
        return lines

    def _render_best_and_worst_picks(
        self,
        frame: pd.DataFrame,
        *,
        horizons: tuple[int, ...],
        benchmark: str,
    ) -> list[str]:
        lines = ["## Best And Worst Picks", ""]
        for horizon in horizons:
            return_column = f"fwd_return_{horizon}d"
            alpha_column = f"alpha_vs_{benchmark}_{horizon}d"
            lines.append(f"### {horizon}d")
            scoped = self._matured_outcome_frame(
                frame,
                return_column=return_column,
                alpha_column=alpha_column,
            )
            if scoped.empty:
                lines.append("No matured picks.")
                lines.append("")
                continue
            best = self._distinct_ticker_extremes(
                scoped,
                sort_columns=[alpha_column, return_column, "scan_date", "ticker"],
                ascending=[False, False, False, True],
                top_n=3,
            )
            worst = self._distinct_ticker_extremes(
                scoped,
                sort_columns=[alpha_column, return_column, "scan_date", "ticker"],
                ascending=[True, True, False, True],
                top_n=3,
            )
            lines.append("- best:")
            for row in best.itertuples(index=False):
                lines.append(
                    f"  - {row.ticker} ({pd.Timestamp(row.scan_date).date()}): "
                    f"return={self._fmt_pct(getattr(row, return_column))}, "
                    f"alpha_vs_{benchmark}={self._fmt_pct(getattr(row, alpha_column))}"
                )
            lines.append("- worst:")
            for row in worst.itertuples(index=False):
                lines.append(
                    f"  - {row.ticker} ({pd.Timestamp(row.scan_date).date()}): "
                    f"return={self._fmt_pct(getattr(row, return_column))}, "
                    f"alpha_vs_{benchmark}={self._fmt_pct(getattr(row, alpha_column))}"
                )
            lines.append("")
        return lines

    def _render_repeated_winners_and_losers(
        self,
        frame: pd.DataFrame,
        *,
        horizons: tuple[int, ...],
        benchmark: str,
    ) -> list[str]:
        lines = ["## Repeated Winners And Losers", ""]
        for horizon in horizons:
            return_column = f"fwd_return_{horizon}d"
            alpha_column = f"alpha_vs_{benchmark}_{horizon}d"
            lines.append(f"### {horizon}d")
            scoped = self._matured_outcome_frame(
                frame,
                return_column=return_column,
                alpha_column=alpha_column,
            )
            if scoped.empty:
                lines.append("No matured picks.")
                lines.append("")
                continue
            grouped = (
                scoped.groupby("ticker", as_index=False)
                .agg(
                    observations=(return_column, "count"),
                    mean_return=(return_column, "mean"),
                    mean_alpha=(alpha_column, "mean"),
                )
                .sort_values(["observations", "mean_alpha", "ticker"], ascending=[False, False, True])
                .reset_index(drop=True)
            )
            repeated = grouped[grouped["observations"] >= 2].copy()
            if repeated.empty:
                lines.append("No repeated tickers.")
                lines.append("")
                continue
            winners = repeated.sort_values(["mean_alpha", "observations", "ticker"], ascending=[False, False, True]).head(3)
            losers = repeated.sort_values(["mean_alpha", "observations", "ticker"], ascending=[True, False, True]).head(3)
            lines.append("- repeated_winners:")
            for row in winners.itertuples(index=False):
                lines.append(
                    f"  - {row.ticker}: n={int(row.observations)}, "
                    f"mean_return={self._fmt_pct(row.mean_return)}, "
                    f"mean_alpha_vs_{benchmark}={self._fmt_pct(row.mean_alpha)}"
                )
            lines.append("- repeated_losers:")
            for row in losers.itertuples(index=False):
                lines.append(
                    f"  - {row.ticker}: n={int(row.observations)}, "
                    f"mean_return={self._fmt_pct(row.mean_return)}, "
                    f"mean_alpha_vs_{benchmark}={self._fmt_pct(row.mean_alpha)}"
                )
            lines.append("")
        return lines

    def _distinct_ticker_extremes(
        self,
        frame: pd.DataFrame,
        *,
        sort_columns: list[str],
        ascending: list[bool],
        top_n: int,
    ) -> pd.DataFrame:
        ordered = frame.sort_values(sort_columns, ascending=ascending).copy()
        distinct = ordered.drop_duplicates(subset=["ticker"], keep="first")
        return distinct.head(int(top_n)).copy()

    def _matured_outcome_frame(
        self,
        frame: pd.DataFrame,
        *,
        return_column: str,
        alpha_column: str,
    ) -> pd.DataFrame:
        required_columns = [return_column, alpha_column]
        if any(column not in frame.columns for column in required_columns):
            return frame.iloc[0:0].copy()
        return frame.dropna(subset=required_columns).copy()

    def _render_recent_picks(
        self,
        frame: pd.DataFrame,
        *,
        horizons: tuple[int, ...],
        benchmark: str,
        recent_picks: int,
    ) -> list[str]:
        lines = ["## Recent Picks", ""]
        ordered = frame.sort_values(["scan_date", "selected_rank", "ticker"], ascending=[False, True, True]).head(int(recent_picks)).copy()
        for row in ordered.itertuples(index=False):
            lines.append(f"### {row.ticker}")
            lines.append(f"- scan_date: {pd.Timestamp(row.scan_date).date()}")
            lines.append(f"- sector: {row.sector}")
            lines.append(f"- selected_rank: {int(row.selected_rank) if pd.notna(row.selected_rank) else 'n/a'}")
            for horizon in horizons:
                return_value = getattr(row, f"fwd_return_{horizon}d", float('nan'))
                alpha_value = getattr(row, f"alpha_vs_{benchmark}_{horizon}d", float('nan'))
                if not (math.isfinite(float(return_value)) and math.isfinite(float(alpha_value))):
                    continue
                lines.append(
                    f"- {horizon}d: return={self._fmt_pct(return_value)}, "
                    f"alpha_vs_{benchmark}={self._fmt_pct(alpha_value)}"
                )
            lines.append("")
        return lines

    def _fmt(self, value: float) -> str:
        if value is None or not math.isfinite(float(value)):
            return "n/a"
        return f"{float(value):.4f}"

    def _fmt_pct(self, value: float) -> str:
        if value is None or not math.isfinite(float(value)):
            return "n/a"
        return f"{float(value) * 100.0:.2f}%"

    def _fmt_dollars(self, value: float) -> str:
        if value is None or not math.isfinite(float(value)):
            return "n/a"
        return f"${float(value):,.2f}"

    def _fmt_price(self, value: float) -> str:
        if value is None or not math.isfinite(float(value)):
            return "n/a"
        return f"{float(value):.2f}"

    def _fmt_points(self, value: float) -> str:
        if value is None or not math.isfinite(float(value)):
            return "n/a"
        return f"{float(value):+.1f}pts"

    def _fmt_score(self, value: float) -> str:
        if value is None or not math.isfinite(float(value)):
            return "n/a"
        return f"{float(value):.4f}"

    def _fmt_int(self, value) -> str:
        if value is None or pd.isna(value):
            return "n/a"
        try:
            return str(int(float(value)))
        except (TypeError, ValueError):
            return "n/a"

    def _coerce_float(self, value) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("nan")

    def _coerce_int(self, value) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    def _build_performance_dashboard(
        self,
        *,
        enriched: pd.DataFrame,
        horizons: tuple[int, ...],
        benchmark: str,
        resolved_scope: dict,
        window_label: str,
    ) -> dict:
        dashboard: dict[str, object] = {
            "model_name": resolved_scope.get("model_name") or "unknown",
            "scope": resolved_scope.get("scope") or "all",
            "scan_dates": int(enriched["scan_date"].nunique()),
            "total_picks": len(enriched.index),
            "window_label": window_label,
        }
        return_column = "fwd_return_20d"
        alpha_column = f"alpha_vs_{benchmark}_20d"
        if return_column in enriched.columns and alpha_column in enriched.columns:
            scoped = enriched.dropna(subset=[return_column, alpha_column]).copy()
            if not scoped.empty:
                scoped[return_column] = pd.to_numeric(scoped[return_column], errors="coerce")
                scoped[alpha_column] = pd.to_numeric(scoped[alpha_column], errors="coerce")
                anchor = scoped["scan_date"].max()
                windows_def = {
                    "20d": anchor - pd.DateOffset(days=20),
                    "3m": anchor - pd.DateOffset(months=3),
                    "1y": anchor - pd.DateOffset(years=1),
                }
                for label, start in windows_def.items():
                    window = scoped[scoped["scan_date"] >= pd.Timestamp(start).normalize()]
                    if window.empty:
                        continue
                    returns = window[return_column].astype(float)
                    alphas = window[alpha_column].astype(float)
                    beats = sum(
                        1 for d in sorted(window["scan_date"].unique())
                        if window[window["scan_date"] == d][alpha_column].mean() > 0
                    )
                    unique_dates = int(window["scan_date"].nunique())
                    dashboard[f"window_{label}"] = {
                        "picks": len(window.index),
                        "dates": unique_dates,
                        "mean_return": float(returns.mean()),
                        "median_return": float(returns.median()),
                        "hit_rate": float((returns > 0).mean()),
                        "mean_alpha": float(alphas.mean()),
                        "positive_alpha_rate": float((alphas > 0).mean()),
                        "beat_rate": float(beats / unique_dates) if unique_dates > 0 else 0.0,
                    }
                recent = dashboard.get("window_20d", {})
                older = dashboard.get("window_3m", {})
                if recent and older:
                    recent_alpha = float(recent.get("mean_alpha", 0))
                    older_alpha = float(older.get("mean_alpha", 0))
                    recent_hit = float(recent.get("hit_rate", 0))
                    older_hit = float(older.get("hit_rate", 0))
                    if recent_alpha > older_alpha and recent_hit > older_hit:
                        dashboard["trend"] = "improving"
                    elif recent_alpha < older_alpha and recent_hit < older_hit:
                        dashboard["trend"] = "worsening"
                    else:
                        dashboard["trend"] = "mixed"
                    if recent_alpha < -0.02 or recent_hit < 0.45:
                        dashboard["recommendation"] = "push_harder"
                    elif recent_alpha > 0.03 and recent_hit > 0.55:
                        dashboard["recommendation"] = "on_track"
                    else:
                        dashboard["recommendation"] = "monitor"
                else:
                    dashboard["trend"] = "insufficient_data"
                    dashboard["recommendation"] = "monitor"
        return dashboard

    def _load_forward_predictions(self) -> pd.DataFrame:
        if not hasattr(self.db_manager, "load_shortlist_model_predictions"):
            return pd.DataFrame()
        try:
            runs = self.db_manager.load_shortlist_model_runs(
                horizon_days=20,
                eligible_universe_mode="passed_or_trend",
                model_scope="sector_specific",
                limit=1,
            )
            if runs.empty:
                return pd.DataFrame()
            latest_run = runs.iloc[0]
            predictions = self.db_manager.load_shortlist_model_predictions(
                generated_at=str(latest_run["generated_at"]),
                horizon_days=20,
                eligible_universe_mode="passed_or_trend",
                model_scope="sector_specific",
                dataset_split="live",
                model_name=str(latest_run["champion_model"]),
            )
            if predictions.empty:
                return pd.DataFrame()
            predictions["predicted_alpha"] = pd.to_numeric(predictions["predicted_alpha"], errors="coerce")
            predictions["md_volume_30d"] = pd.to_numeric(predictions["md_volume_30d"], errors="coerce")
            return predictions.sort_values("predicted_alpha", ascending=False).reset_index(drop=True)
        except Exception:
            return pd.DataFrame()

    def _render_performance_email(
        self,
        enriched: pd.DataFrame,
        dashboard: dict,
        horizons: tuple[int, ...],
        benchmark: str,
        forward_predictions: pd.DataFrame | None = None,
    ) -> str:
        rec = str(dashboard.get("recommendation", "monitor"))
        rec_colors = {
            "push_harder": ("#dc3545", "PUSH HARDER — recent alpha negative, model not working"),
            "monitor": ("#ffc107", "MONITOR — performance is borderline, watch closely"),
            "on_track": ("#28a745", "ON TRACK — model is delivering, stay the course"),
        }
        rec_color, rec_text = rec_colors.get(rec, rec_colors["monitor"])
        trend = str(dashboard.get("trend", "unknown"))
        trend_icons = {"improving": "↑ improving", "worsening": "↓ worsening", "mixed": "→ mixed", "insufficient_data": "? not enough data"}
        trend_text = trend_icons.get(trend, trend)

        sections: list[str] = []

        # header
        sections.append(f"""
        <div style="background:{rec_color};color:#fff;padding:16px 20px;border-radius:6px;margin-bottom:16px;">
            <h2 style="margin:0 0 4px 0;font-size:18px;">{rec_text}</h2>
            <p style="margin:0;font-size:13px;opacity:0.9;">trend: {trend_text} | model: {dashboard.get('model_name', 'unknown')} | scope: {dashboard.get('scope', 'all')}</p>
        </div>
        """)

        # performance summary cards
        cards_html = ""
        for label in ("20d", "3m", "1y"):
            w = dashboard.get(f"window_{label}")
            if not w:
                continue
            alpha = float(w["mean_alpha"])
            hit = float(w["hit_rate"])
            beat = float(w["beat_rate"])
            alpha_color = "#28a745" if alpha > 0.02 else ("#dc3545" if alpha < 0 else "#6c757d")
            cards_html += f"""
            <div style="flex:1;min-width:140px;background:#f8f9fa;border-radius:6px;padding:12px;text-align:center;margin:4px;">
                <div style="font-size:11px;color:#6c757d;text-transform:uppercase;letter-spacing:0.5px;">{label} window</div>
                <div style="font-size:24px;font-weight:700;color:{alpha_color};margin:4px 0;">{alpha:+.1%}</div>
                <div style="font-size:12px;color:#495057;">mean alpha</div>
                <div style="margin-top:6px;font-size:12px;color:#6c757d;">
                    hit {hit:.0%} &middot; beat {beat:.0%}
                </div>
            </div>
            """
        sections.append(f'<div style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:16px;">{cards_html}</div>')

        # horizon summary table
        return_col = "fwd_return_20d"
        alpha_col = f"alpha_vs_{benchmark}_20d"
        table_rows = ""
        for h in sorted(horizons):
            rcol = f"fwd_return_{h}d"
            acol = f"alpha_vs_{benchmark}_{h}d"
            if rcol not in enriched.columns:
                continue
            data = enriched.dropna(subset=[rcol, acol])
            if data.empty:
                continue
            rets = pd.to_numeric(data[rcol], errors="coerce")
            alps = pd.to_numeric(data[acol], errors="coerce")
            table_rows += f"""
            <tr>
                <td style="padding:6px 12px;border-bottom:1px solid #dee2e6;font-weight:600;">{h}d</td>
                <td style="padding:6px 12px;border-bottom:1px solid #dee2e6;">{len(data.index)}</td>
                <td style="padding:6px 12px;border-bottom:1px solid #dee2e6;color:{'#28a745' if rets.mean() > 0 else '#dc3545'};">{rets.mean():+.1%}</td>
                <td style="padding:6px 12px;border-bottom:1px solid #dee2e6;">{rets.median():+.1%}</td>
                <td style="padding:6px 12px;border-bottom:1px solid #dee2e6;">{(rets > 0).mean():.0%}</td>
                <td style="padding:6px 12px;border-bottom:1px solid #dee2e6;color:{'#28a745' if alps.mean() > 0 else '#dc3545'};">{alps.mean():+.1%}</td>
                <td style="padding:6px 12px;border-bottom:1px solid #dee2e6;">{(alps > 0).mean():.0%}</td>
            </tr>
            """
        sections.append(f"""
        <h3 style="font-size:14px;color:#495057;margin-bottom:8px;">Horizon Summary</h3>
        <table style="width:100%;border-collapse:collapse;font-size:12px;margin-bottom:16px;">
            <tr style="background:#e9ecef;">
                <th style="padding:6px 12px;text-align:left;">Horizon</th>
                <th style="padding:6px 12px;text-align:left;">Picks</th>
                <th style="padding:6px 12px;text-align:left;">Mean Ret</th>
                <th style="padding:6px 12px;text-align:left;">Median</th>
                <th style="padding:6px 12px;text-align:left;">Hit</th>
                <th style="padding:6px 12px;text-align:left;">Alpha</th>
                <th style="padding:6px 12px;text-align:left;">Pos Alpha</th>
            </tr>
            {table_rows}
        </table>
        """)

        # recent picks
        recent_picks_html = ""
        scan_dates = sorted(enriched["scan_date"].drop_duplicates().tolist())[-5:]
        for sd in reversed(scan_dates):
            day = enriched[enriched["scan_date"] == sd]
            picks = day[day.get("selected", 0) == 1] if "selected" in day.columns else day.head(6)
            if picks.empty:
                picks = day.head(6)
            tickers = ", ".join(str(t) for t in picks["ticker"].tolist())
            recent_picks_html += f'<tr><td style="padding:4px 12px;border-bottom:1px solid #dee2e6;">{pd.Timestamp(sd).date()}</td><td style="padding:4px 12px;border-bottom:1px solid #dee2e6;">{tickers}</td></tr>'

        # forward predictions
        if forward_predictions is not None and not forward_predictions.empty:
            top_n = forward_predictions.head(6)
            fp_rows = ""
            for _, row in top_n.iterrows():
                alpha = float(row["predicted_alpha"])
                alpha_color = "#28a745" if alpha > 0 else "#dc3545"
                fp_rows += f"""
                <tr>
                    <td style="padding:4px 10px;font-weight:600;">{row['ticker']}</td>
                    <td style="padding:4px 10px;">{row.get('sector', '')}</td>
                    <td style="padding:4px 10px;color:{alpha_color};">{alpha:+.1%}</td>
                    <td style="padding:4px 10px;">{row.get('model_reason_summary', '')}</td>
                </tr>
                """
            sections.append(f"""
            <h3 style="font-size:14px;color:#495057;margin-bottom:8px;">Forward Predictions (20d alpha)</h3>
            <p style="font-size:11px;color:#6c757d;margin:0 0 6px 0;">Top model-ranked candidates and their predicted sector-relative alpha over the next 20 trading days.</p>
            <table style="width:100%;border-collapse:collapse;font-size:12px;margin-bottom:16px;">
                <tr style="background:#e9ecef;">
                    <th style="padding:4px 10px;text-align:left;">Ticker</th>
                    <th style="padding:4px 10px;text-align:left;">Sector</th>
                    <th style="padding:4px 10px;text-align:left;">Pred Alpha</th>
                    <th style="padding:4px 10px;text-align:left;">Why</th>
                </tr>
                {fp_rows}
            </table>
            """)

        sections.append(f"""
        <h3 style="font-size:14px;color:#495057;margin-bottom:8px;">Recent Picks</h3>
        <table style="width:100%;border-collapse:collapse;font-size:12px;margin-bottom:16px;">
            <tr style="background:#e9ecef;"><th style="padding:4px 12px;text-align:left;">Date</th><th style="padding:4px 12px;text-align:left;">Picks</th></tr>
            {recent_picks_html}
        </table>
        """)

        # portfolio summary
        portfolio_lines = self._render_portfolio_performance()
        portfolio_text = "\n".join(portfolio_lines) if portfolio_lines else ""
        if portfolio_text.strip():
            sections.append(f"""
            <h3 style="font-size:14px;color:#495057;margin-bottom:8px;">Portfolio</h3>
            <pre style="font-size:11px;color:#495057;background:#f8f9fa;padding:12px;border-radius:4px;line-height:1.4;">{escape(portfolio_text.strip())}</pre>
            """)

        # full report link
        sections.append("""
        <hr style="border:none;border-top:1px solid #dee2e6;margin:16px 0;">
        <p style="font-size:11px;color:#6c757d;">Full report: reports/scan_performance.md</p>
        """)

        body = "".join(sections)
        return f"""<html><body style="font-family:-apple-system,BlinkMacSystemFont,sans-serif;max-width:640px;margin:0 auto;padding:16px;color:#212529;">{body}</body></html>"""
