from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import json
import math

import pandas as pd

from src.sync.service import REFERENCE_TICKERS
from src.utils.db_manager import DatabaseManager
from src.utils.logging import get_logger
from src.utils.regime import benchmark_etf_for_sector
from src.utils.signal_engine import build_analysis_frame, latest_rsi_2_with_intraday
from src.utils.strategy import (
    load_active_strategies,
    profit_target_price,
    trailing_stop_price,
)


VARIANT_ORDER = (
    "current_rule",
    "no_rsi_2",
    "profit_gate_3pct",
    "profit_gate_5pct",
    "min_hold_3d",
)

VARIANT_LABELS = {
    "current_rule": "Current Rule",
    "no_rsi_2": "No RSI_2 Exit",
    "profit_gate_3pct": "RSI_2 Only After +3%",
    "profit_gate_5pct": "RSI_2 Only After +5%",
    "min_hold_3d": "RSI_2 Only After 3d",
}

EXIT_REASON_ORDER = (
    "trailing_stop",
    "profit_target",
    "rsi_2",
    "time_limit",
    "regime_flip",
    "pre_earnings_exit",
    "end_of_history",
)


@dataclass(frozen=True)
class RsiExitBakeoffReport:
    output_path: str
    selected_rows: int
    mature_rows: int
    scan_dates: int
    benchmark: str


class RsiExitBakeoffService:
    def __init__(self, db_manager: DatabaseManager) -> None:
        self.db_manager = db_manager
        self.logger = get_logger("rsi_exit_bakeoff")

    def run(
        self,
        *,
        recent_scan_dates: int = 60,
        benchmark: str = "sector",
    ) -> RsiExitBakeoffReport:
        self.db_manager.initialize()
        benchmark = self._normalize_benchmark(benchmark)
        strategies = load_active_strategies()
        candidates = self.db_manager.load_scan_candidates()
        if candidates.empty:
            raise ValueError("No scan snapshots found. Run `sq scan` or `sq scan-backfill` first.")

        selected = candidates[candidates["selected"].astype(int) == 1].copy()
        if selected.empty:
            raise ValueError("No selected scan picks found. Run `sq scan` or `sq scan-backfill` first.")

        selected["scan_date"] = pd.to_datetime(selected["scan_date"]).dt.normalize()
        unique_dates = sorted(selected["scan_date"].drop_duplicates().tolist())
        scoped_dates = unique_dates[-max(int(recent_scan_dates), 1) :]
        selected = selected[selected["scan_date"].isin(scoped_dates)].copy().reset_index(drop=True)
        if selected.empty:
            raise ValueError("No selected scan picks matched the requested recent window.")

        selected = selected[selected["strategy_slot"].astype(str).isin(strategies.keys())].copy().reset_index(drop=True)
        if selected.empty:
            raise ValueError("No selected scan picks matched the active strategy slots.")

        tickers = sorted(set(selected["ticker"].astype(str)).union(set(REFERENCE_TICKERS)))
        benchmarks = {
            benchmark_etf_for_sector(str(row.get("sector") or row.get("strategy_sector") or ""))
            for row in selected.to_dict(orient="records")
        }
        tickers = sorted(set(tickers).union({ticker for ticker in benchmarks if ticker}))
        price_history = self.db_manager.load_price_history(tickers)
        if price_history.empty:
            raise ValueError("Historical prices are unavailable. Run `sq sync` first.")
        price_history["date"] = pd.to_datetime(price_history["date"]).dt.normalize()
        history_context = self._history_context(price_history)

        universe_rows = self.db_manager.list_universe_rows(active_only=False)
        selected_tickers = sorted(set(selected["ticker"].astype(str)))
        earnings_loader = getattr(self.db_manager, "load_earnings_calendar", None)
        earnings_calendar = earnings_loader(selected_tickers) if callable(earnings_loader) else pd.DataFrame()
        analysis_frame, _ = build_analysis_frame(
            price_history,
            universe_rows,
            earnings_calendar=earnings_calendar,
        )
        analysis_frame["date"] = pd.to_datetime(analysis_frame["date"]).dt.normalize()
        analysis_lookup = {
            (str(row["ticker"]), str(pd.Timestamp(row["date"]).date())): row
            for row in analysis_frame.to_dict(orient="records")
        }

        simulated_rows: list[dict[str, object]] = []
        for row in selected.to_dict(orient="records"):
            simulated = self._simulate_pick_variants(
                row=row,
                strategies=strategies,
                history_context=history_context,
                analysis_lookup=analysis_lookup,
                benchmark=benchmark,
            )
            if simulated:
                simulated_rows.extend(simulated)

        simulated_frame = pd.DataFrame(simulated_rows)
        if simulated_frame.empty:
            raise ValueError("No mature recommendation picks were available for the requested bakeoff.")

        output_path = self.db_manager.paths.reports_dir / "rsi_exit_bakeoff.md"
        lines = [
            "# RSI_2 Exit Bakeoff",
            "",
            f"- benchmark: {benchmark}",
            f"- recent_scan_dates: {len(scoped_dates)}",
            f"- selected_rows: {len(selected.index)}",
            f"- mature_rows: {int(simulated_frame['pick_id'].nunique())}",
            f"- scan_date_min: {min(scoped_dates).date()}",
            f"- scan_date_max: {max(scoped_dates).date()}",
            "- note: exits are replayed on daily close, not intraday 1-minute prices.",
            "",
        ]
        lines.extend(self._render_variant_summary(simulated_frame, benchmark=benchmark))
        lines.extend(self._render_delta_vs_current(simulated_frame, benchmark=benchmark))
        lines.extend(self._render_exit_reason_mix(simulated_frame))
        lines.extend(self._render_slot_breakdown(simulated_frame, benchmark=benchmark))
        lines.extend(self._render_early_rsi_sells(simulated_frame, benchmark=benchmark))
        output_path.write_text("\n".join(lines), encoding="utf-8")
        return RsiExitBakeoffReport(
            output_path=str(output_path),
            selected_rows=len(selected.index),
            mature_rows=int(simulated_frame["pick_id"].nunique()),
            scan_dates=len(scoped_dates),
            benchmark=benchmark,
        )

    def _normalize_benchmark(self, benchmark: str) -> str:
        normalized = str(benchmark or "sector").strip().lower()
        if normalized not in {"sector", "spy"}:
            raise ValueError("benchmark must be one of: sector, spy")
        return normalized

    def _history_context(self, history: pd.DataFrame) -> dict[str, dict[str, object]]:
        context: dict[str, dict[str, object]] = {}
        for ticker, group in history.groupby("ticker", sort=False):
            ordered = group.sort_values("date").reset_index(drop=True)
            context[str(ticker)] = {
                "frame": ordered,
                "index_by_date": {
                    pd.Timestamp(date_value).normalize().strftime("%Y-%m-%d"): int(index)
                    for index, date_value in enumerate(ordered["date"])
                },
            }
        return context

    def _simulate_pick_variants(
        self,
        *,
        row: dict[str, object],
        strategies: dict[str, object],
        history_context: dict[str, dict[str, object]],
        analysis_lookup: dict[tuple[str, str], dict[str, object]],
        benchmark: str,
    ) -> list[dict[str, object]]:
        ticker = str(row["ticker"])
        slot = str(row["strategy_slot"])
        strategy = strategies.get(slot)
        if strategy is None:
            return []
        ticker_context = history_context.get(ticker)
        if ticker_context is None:
            return []
        scan_date = str(pd.Timestamp(row["scan_date"]).date())
        entry_index = ticker_context["index_by_date"].get(scan_date)
        if entry_index is None:
            return []
        ticker_frame = ticker_context["frame"]
        if entry_index + 1 >= len(ticker_frame.index):
            return []

        details = self._parse_details_json(row.get("details_json"))
        feature_snapshot = details.get("feature_snapshot", {})
        entry_price = self._coerce_float(row.get("adj_close"))
        if not math.isfinite(entry_price) or entry_price <= 0:
            entry_price = float(ticker_frame.loc[entry_index, "adj_close"])
        entry_atr = self._coerce_float(feature_snapshot.get("atr_14"))
        benchmark_ticker = "SPY" if benchmark == "spy" else benchmark_etf_for_sector(str(row.get("sector") or row.get("strategy_sector") or ""))
        scan_date_analysis_row = analysis_lookup.get((ticker, scan_date), {})
        if not math.isfinite(entry_atr):
            entry_atr = self._coerce_float(scan_date_analysis_row.get("atr_14"))
        path = self._build_trade_path(
            ticker=ticker,
            ticker_frame=ticker_frame,
            entry_index=int(entry_index),
            entry_price=float(entry_price),
            entry_atr=entry_atr,
            strategy=strategy,
            analysis_lookup=analysis_lookup,
        )
        if not path:
            return []
        required_future_bars = int(strategy.exit_rules.time_limit_days) + 1
        if len(path) < required_future_bars:
            return []

        pick_id = f"{scan_date}|{ticker}|{slot}"
        variant_rows: list[dict[str, object]] = []
        for variant_name in VARIANT_ORDER:
            simulated = self._simulate_variant(
                variant_name=variant_name,
                path=path,
                benchmark_ticker=benchmark_ticker,
                benchmark=benchmark,
                history_context=history_context,
                time_limit_days=int(strategy.exit_rules.time_limit_days),
                ticker=ticker,
                slot=slot,
                sector=str(row.get("sector") or row.get("strategy_sector") or ""),
                scan_date=scan_date,
                pick_id=pick_id,
            )
            if simulated is not None:
                variant_rows.append(simulated)
        if len(variant_rows) != len(VARIANT_ORDER):
            return []
        return variant_rows

    def _build_trade_path(
        self,
        *,
        ticker: str,
        ticker_frame: pd.DataFrame,
        entry_index: int,
        entry_price: float,
        entry_atr: float,
        strategy,
        analysis_lookup: dict[tuple[str, str], dict[str, object]],
    ) -> list[dict[str, object]]:
        path: list[dict[str, object]] = []
        max_price_seen = float(entry_price)
        for current_index in range(entry_index + 1, len(ticker_frame.index)):
            current_row = ticker_frame.iloc[current_index]
            current_date = pd.Timestamp(current_row["date"]).normalize()
            current_date_str = str(current_date.date())
            current_close = float(current_row["adj_close"])
            stop_price = trailing_stop_price(
                max_price_seen=max_price_seen,
                entry_atr=entry_atr if math.isfinite(entry_atr) else None,
                exit_rules=strategy.exit_rules,
            )
            target_price = profit_target_price(
                entry_price=entry_price,
                entry_atr=entry_atr if math.isfinite(entry_atr) else None,
                exit_rules=strategy.exit_rules,
            )
            price_slice = ticker_frame.iloc[: current_index + 1].copy()
            rsi_2 = latest_rsi_2_with_intraday(
                price_history=price_slice,
                ticker=ticker,
                current_price=current_close,
                as_of=current_date.date(),
            )
            analysis_row = analysis_lookup.get((ticker, current_date_str), {})
            buy_setup_valid = self._current_setup_valid(strategy=strategy, analysis_row=analysis_row)
            regime_green = analysis_row.get("regime_green")
            if pd.isna(regime_green):
                regime_green = True
            days_to_next_earnings = self._coerce_float(analysis_row.get("days_to_next_earnings"))
            unrealized_pct = (current_close / entry_price) - 1.0
            path.append(
                {
                    "date": current_date_str,
                    "index": int(current_index),
                    "current_close": current_close,
                    "unrealized_pct": unrealized_pct,
                    "days_in_trade": int(current_index - entry_index),
                    "stop_price": float(stop_price),
                    "target_price": float(target_price),
                    "rsi_2": float(rsi_2),
                    "regime_flip": bool(regime_green is False),
                    "pre_earnings_exit": bool(
                        strategy.exit_rules.exit_before_earnings_days is not None
                        and math.isfinite(days_to_next_earnings)
                        and days_to_next_earnings <= float(strategy.exit_rules.exit_before_earnings_days)
                    ),
                    "buy_setup_valid": bool(buy_setup_valid),
                }
            )
            max_price_seen = max(max_price_seen, current_close)
        return path

    def _current_setup_valid(self, *, strategy, analysis_row: dict[str, object]) -> bool:
        if not analysis_row:
            return False
        indicators = strategy.indicators
        from src.utils.strategy import evaluate_signal_gate

        passed, _, _ = evaluate_signal_gate(indicators, analysis_row)
        return bool(passed)

    def _simulate_variant(
        self,
        *,
        variant_name: str,
        path: list[dict[str, object]],
        benchmark_ticker: str | None,
        benchmark: str,
        history_context: dict[str, dict[str, object]],
        time_limit_days: int,
        ticker: str,
        slot: str,
        sector: str,
        scan_date: str,
        pick_id: str,
    ) -> dict[str, object] | None:
        for state in path:
            rsi_exit = self._variant_rsi_exit_trigger(variant_name=variant_name, state=state)
            exit_flags = {
                "trailing_stop": bool(state["current_close"] < state["stop_price"]),
                "profit_target": bool(state["current_close"] >= state["target_price"]),
                "rsi_2": rsi_exit,
                "time_limit": bool(int(state["days_in_trade"]) > int(time_limit_days)),
                "regime_flip": bool(state["regime_flip"]),
                "pre_earnings_exit": bool(state["pre_earnings_exit"]),
            }
            # Keep current runtime semantics for time limit: trigger only after the configured limit.
            # The precomputed state is variant-agnostic; time limit depends on the active slot config.
            exit_reason = next((name for name in EXIT_REASON_ORDER if exit_flags.get(name)), None)
            if exit_reason is None:
                continue
            trade_return = float(state["unrealized_pct"])
            benchmark_return = self._benchmark_return(
                history_context=history_context,
                benchmark_ticker=benchmark_ticker,
                entry_date=scan_date,
                exit_date=str(state["date"]),
            )
            alpha = trade_return - benchmark_return if math.isfinite(benchmark_return) else float("nan")
            return {
                "pick_id": pick_id,
                "scan_date": scan_date,
                "ticker": ticker,
                "slot": slot,
                "sector": sector,
                "variant_name": variant_name,
                "variant_label": VARIANT_LABELS[variant_name],
                "exit_date": str(state["date"]),
                "holding_days": int(state["days_in_trade"]),
                "return_pct": trade_return,
                f"alpha_vs_{benchmark}": alpha,
                "exit_reason": exit_reason,
                "rsi_2_at_exit": float(state["rsi_2"]),
            }
        last_state = path[-1]
        trade_return = float(last_state["unrealized_pct"])
        benchmark_return = self._benchmark_return(
            history_context=history_context,
            benchmark_ticker=benchmark_ticker,
            entry_date=scan_date,
            exit_date=str(last_state["date"]),
        )
        alpha = trade_return - benchmark_return if math.isfinite(benchmark_return) else float("nan")
        return {
            "pick_id": pick_id,
            "scan_date": scan_date,
            "ticker": ticker,
            "slot": slot,
            "sector": sector,
            "variant_name": variant_name,
            "variant_label": VARIANT_LABELS[variant_name],
            "exit_date": str(last_state["date"]),
            "holding_days": int(last_state["days_in_trade"]),
            "return_pct": trade_return,
            f"alpha_vs_{benchmark}": alpha,
            "exit_reason": "end_of_history",
            "rsi_2_at_exit": float(last_state["rsi_2"]),
        }

    def _variant_rsi_exit_trigger(self, *, variant_name: str, state: dict[str, object]) -> bool:
        rsi_trigger = bool(float(state["rsi_2"]) > 90.0)
        if not rsi_trigger:
            return False
        if variant_name == "current_rule":
            return True
        if variant_name == "no_rsi_2":
            return False
        if variant_name == "profit_gate_3pct":
            return bool(float(state["unrealized_pct"]) >= 0.03)
        if variant_name == "profit_gate_5pct":
            return bool(float(state["unrealized_pct"]) >= 0.05)
        if variant_name == "min_hold_3d":
            return bool(int(state["days_in_trade"]) >= 3)
        raise ValueError(f"Unsupported variant: {variant_name}")

    def _benchmark_return(
        self,
        *,
        history_context: dict[str, dict[str, object]],
        benchmark_ticker: str | None,
        entry_date: str,
        exit_date: str,
    ) -> float:
        if benchmark_ticker in (None, ""):
            return float("nan")
        context = history_context.get(str(benchmark_ticker))
        if context is None:
            return float("nan")
        index_by_date = context["index_by_date"]
        frame = context["frame"]
        entry_index = index_by_date.get(str(entry_date))
        exit_index = index_by_date.get(str(exit_date))
        if entry_index is None or exit_index is None:
            return float("nan")
        entry_price = float(frame.loc[int(entry_index), "adj_close"])
        exit_price = float(frame.loc[int(exit_index), "adj_close"])
        return (exit_price / entry_price) - 1.0

    def _render_variant_summary(self, frame: pd.DataFrame, *, benchmark: str) -> list[str]:
        lines = ["## Variant Summary", ""]
        alpha_column = f"alpha_vs_{benchmark}"
        for variant_name in VARIANT_ORDER:
            scoped = frame[frame["variant_name"] == variant_name].copy()
            lines.append(f"### {VARIANT_LABELS[variant_name]}")
            if scoped.empty:
                lines.append("- mature_picks: 0")
                lines.append("")
                continue
            returns = pd.to_numeric(scoped["return_pct"], errors="coerce").dropna()
            alphas = pd.to_numeric(scoped[alpha_column], errors="coerce").dropna()
            lines.append(f"- mature_picks: {len(scoped.index)}")
            lines.append(f"- mean_return: {self._fmt_pct(returns.mean())}")
            lines.append(f"- median_return: {self._fmt_pct(returns.median())}")
            lines.append(f"- hit_rate: {self._fmt_pct((returns > 0.0).mean())}")
            lines.append(f"- mean_alpha_vs_{benchmark}: {self._fmt_pct(alphas.mean())}")
            lines.append(f"- median_alpha_vs_{benchmark}: {self._fmt_pct(alphas.median())}")
            lines.append(f"- positive_alpha_rate: {self._fmt_pct((alphas > 0.0).mean())}")
            lines.append(f"- mean_holding_days: {float(pd.to_numeric(scoped['holding_days'], errors='coerce').mean()):.1f}")
            lines.append("")
        return lines

    def _render_delta_vs_current(self, frame: pd.DataFrame, *, benchmark: str) -> list[str]:
        lines = ["## Delta Vs Current Rule", ""]
        alpha_column = f"alpha_vs_{benchmark}"
        pivot = frame.pivot_table(
            index="pick_id",
            columns="variant_name",
            values=["return_pct", alpha_column],
            aggfunc="first",
        )
        for variant_name in VARIANT_ORDER:
            if variant_name == "current_rule":
                continue
            lines.append(f"### {VARIANT_LABELS[variant_name]}")
            if ("return_pct", "current_rule") not in pivot or ("return_pct", variant_name) not in pivot:
                lines.append("- unavailable")
                lines.append("")
                continue
            return_delta = pd.to_numeric(pivot[("return_pct", variant_name)] - pivot[("return_pct", "current_rule")], errors="coerce").dropna()
            alpha_delta = pd.to_numeric(pivot[(alpha_column, variant_name)] - pivot[(alpha_column, "current_rule")], errors="coerce").dropna()
            lines.append(f"- mean_return_delta: {self._fmt_pct(return_delta.mean())}")
            lines.append(f"- median_return_delta: {self._fmt_pct(return_delta.median())}")
            lines.append(f"- mean_alpha_delta_vs_{benchmark}: {self._fmt_pct(alpha_delta.mean())}")
            lines.append(f"- win_rate_vs_current: {self._fmt_pct((return_delta > 0.0).mean())}")
            lines.append("")
        return lines

    def _render_exit_reason_mix(self, frame: pd.DataFrame) -> list[str]:
        lines = ["## Exit Reason Mix", ""]
        for variant_name in VARIANT_ORDER:
            scoped = frame[frame["variant_name"] == variant_name].copy()
            lines.append(f"### {VARIANT_LABELS[variant_name]}")
            if scoped.empty:
                lines.append("- no exits")
                lines.append("")
                continue
            counts = scoped["exit_reason"].value_counts(normalize=True)
            for reason in EXIT_REASON_ORDER:
                if reason in counts.index:
                    lines.append(f"- {reason}: {self._fmt_pct(float(counts[reason]))}")
            lines.append("")
        return lines

    def _render_slot_breakdown(self, frame: pd.DataFrame, *, benchmark: str) -> list[str]:
        lines = ["## Slot Breakdown", ""]
        alpha_column = f"alpha_vs_{benchmark}"
        current_scoped = frame[frame["variant_name"] == "current_rule"].copy()
        for slot in sorted(current_scoped["slot"].dropna().astype(str).unique()):
            lines.append(f"### {slot}")
            slot_frame = frame[frame["slot"].astype(str) == slot].copy()
            for variant_name in VARIANT_ORDER:
                scoped = slot_frame[slot_frame["variant_name"] == variant_name].copy()
                if scoped.empty:
                    continue
                lines.append(
                    f"- {VARIANT_LABELS[variant_name]}: "
                    f"mean_return={self._fmt_pct(pd.to_numeric(scoped['return_pct'], errors='coerce').mean())}, "
                    f"mean_alpha_vs_{benchmark}={self._fmt_pct(pd.to_numeric(scoped[alpha_column], errors='coerce').mean())}, "
                    f"rsi_exit_rate={self._fmt_pct((scoped['exit_reason'] == 'rsi_2').mean())}"
                )
            lines.append("")
        return lines

    def _render_early_rsi_sells(self, frame: pd.DataFrame, *, benchmark: str) -> list[str]:
        lines = ["## Early RSI_2 Sells", ""]
        alpha_column = f"alpha_vs_{benchmark}"
        current_rule = frame[frame["variant_name"] == "current_rule"].copy()
        no_rsi = frame[frame["variant_name"] == "no_rsi_2"].copy()
        if current_rule.empty or no_rsi.empty:
            lines.append("No comparisons available.")
            lines.append("")
            return lines
        merged = current_rule.merge(
            no_rsi[["pick_id", "return_pct", alpha_column, "exit_date", "holding_days"]],
            on="pick_id",
            how="inner",
            suffixes=("_current", "_no_rsi"),
        )
        scoped = merged[
            (merged["exit_reason"] == "rsi_2")
            & (pd.to_numeric(merged["return_pct_current"], errors="coerce") < 0.03)
        ].copy()
        if scoped.empty:
            lines.append("No tiny-profit RSI_2 exits were found in the scoped history.")
            lines.append("")
            return lines
        scoped["return_delta"] = pd.to_numeric(scoped["return_pct_no_rsi"], errors="coerce") - pd.to_numeric(scoped["return_pct_current"], errors="coerce")
        scoped = scoped.sort_values("return_delta", ascending=False).head(10)
        for row in scoped.to_dict(orient="records"):
            lines.append(
                "- "
                f"{row['ticker']} ({row['slot']}, {row['scan_date']}): "
                f"current={self._fmt_pct(row['return_pct_current'])} on {row['exit_date_current']} "
                f"vs no_rsi={self._fmt_pct(row['return_pct_no_rsi'])} on {row['exit_date_no_rsi']} "
                f"(delta {self._fmt_pct(row['return_delta'])})"
            )
        lines.append("")
        return lines

    def _parse_details_json(self, payload) -> dict[str, object]:
        if isinstance(payload, dict):
            return payload
        if not payload:
            return {}
        try:
            parsed = json.loads(payload)
        except Exception:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def _coerce_float(self, value) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return float("nan")
        return result

    def _fmt_pct(self, value: float) -> str:
        if value is None or not math.isfinite(float(value)):
            return "n/a"
        return f"{float(value) * 100:.2f}%"
