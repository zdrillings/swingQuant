from __future__ import annotations

from dataclasses import dataclass
from html import escape
import math

import pandas as pd

from src.settings import get_settings
from src.utils.db_manager import DatabaseManager
from src.utils.emailer import send_html_email
from src.utils.logging import get_logger
from src.utils.regime import benchmark_etf_for_sector


DEFAULT_PERFORMANCE_HORIZONS = (2, 5, 10, 20, 60)


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

        report_path = self.db_manager.paths.reports_dir / "scan_performance.md"
        lines = [
            "# Scan Performance",
            "",
            f"- benchmark: {benchmark}",
            f"- scope: {resolved_scope['scope']}",
            f"- selection_source: {resolved_scope['selection_source'] or 'all'}",
            f"- model_name: {resolved_scope['model_name'] or 'all'}",
            f"- model_generated_at: {resolved_scope['model_generated_at'] or 'all'}",
            f"- recent_scan_dates: {window_label}",
            f"- scan_dates: {len(scoped_dates)}",
            f"- selected_rows: {len(enriched.index)}",
            f"- scan_date_min: {min(scoped_dates).date()}",
            f"- scan_date_max: {max(scoped_dates).date()}",
            "",
        ]
        lines.extend(self._render_horizon_summary(enriched, horizons=horizons, benchmark=benchmark))
        lines.extend(self._render_20d_score_bands(enriched, benchmark=benchmark))
        lines.extend(self._render_best_and_worst_picks(enriched, horizons=horizons, benchmark=benchmark))
        lines.extend(self._render_repeated_winners_and_losers(enriched, horizons=horizons, benchmark=benchmark))
        lines.extend(self._render_recent_scan_dates(enriched, horizons=horizons, benchmark=benchmark))
        lines.extend(self._render_recent_picks(enriched, horizons=horizons, benchmark=benchmark, recent_picks=recent_picks))
        report_text = "\n".join(lines)
        report_path.write_text(report_text, encoding="utf-8")
        if email:
            self.email_sender(
                subject=f"Scan Performance ({benchmark})",
                html_body=self._render_email_html(report_text),
                settings=settings,
            )

        return ScanPerformanceReport(
            output_path=str(report_path),
            selected_rows=len(enriched.index),
            scan_dates=len(scoped_dates),
            benchmark=benchmark,
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
        latest_model_name = "xgboost_model" if "xgboost_model" in model_names else model_names[0]
        return selected.copy(), {
            "scope": "latest_model",
            "selection_source": "shortlist_model",
            "model_name": latest_model_name,
            "model_generated_at": latest_generated_at,
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
            scoped = frame.dropna(subset=[return_column, alpha_column]).copy()
            lines.append(f"### {horizon}d")
            if scoped.empty:
                lines.append("- matured_picks: 0")
                lines.append("")
                continue
            returns = pd.to_numeric(scoped[return_column], errors="coerce").dropna()
            alphas = pd.to_numeric(scoped[alpha_column], errors="coerce").dropna()
            matured_dates = int(scoped["scan_date"].nunique())
            return_p25 = float(returns.quantile(0.25))
            return_p75 = float(returns.quantile(0.75))
            return_p05 = float(returns.quantile(0.05))
            return_p95 = float(returns.quantile(0.95))
            alpha_p25 = float(alphas.quantile(0.25))
            alpha_p75 = float(alphas.quantile(0.75))
            lines.append(f"- matured_picks: {len(scoped.index)}")
            lines.append(f"- matured_scan_dates: {matured_dates}")
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

    def _opportunity_score_bands(self) -> list[tuple[str, float, float | None]]:
        return [
            ("score < 0.30", float("-inf"), 0.30),
            ("0.30 <= score < 0.35", 0.30, 0.35),
            ("0.35 <= score < 0.40", 0.35, 0.40),
            ("0.40 <= score < 0.45", 0.40, 0.45),
            ("0.45 <= score < 0.50", 0.45, 0.50),
            ("score >= 0.50", 0.50, None),
        ]

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
                scoped = day_frame.dropna(subset=[return_column, alpha_column]).copy()
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
            scoped = frame.dropna(subset=[return_column, alpha_column]).copy()
            lines.append(f"### {horizon}d")
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
            scoped = frame.dropna(subset=[return_column, alpha_column]).copy()
            lines.append(f"### {horizon}d")
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

    def _render_email_html(self, report_text: str) -> str:
        escaped = escape(report_text)
        return (
            "<html><body>"
            "<div style=\"font-family:Menlo,Consolas,monospace;font-size:13px;white-space:pre-wrap;line-height:1.45;\">"
            f"{escaped}"
            "</div>"
            "</body></html>"
        )
