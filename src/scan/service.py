from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import math

import pandas as pd

from src.scan.ranker import CandidateRanker
from src.settings import get_settings, load_feature_config
from src.sync.market_data import MarketDataClient, chunked, extract_ticker_history
from src.sync.service import REFERENCE_TICKERS
from src.utils.db_manager import DatabaseManager
from src.utils.emailer import send_html_email
from src.utils.logging import get_logger
from src.utils.signal_engine import build_analysis_frame, filter_signal_candidates, latest_snapshot, overlay_price_history
from src.utils.sizing import compute_position_size
from src.utils.strategy import (
    ProductionStrategy,
    load_active_strategies,
    profit_target_price,
    resolve_trade_strategy,
    trailing_stop_price,
)


@dataclass(frozen=True)
class ScanReport:
    candidate_count: int
    emailed: bool
    learned_ranker_enabled: bool
    learned_ranker_train_rows: int
    learned_ranker_train_dates: int
    learned_ranker_reason: str | None


@dataclass(frozen=True)
class LearnedRankerStatus:
    enabled: bool
    train_rows: int
    train_dates: int
    reason: str | None = None


@dataclass(frozen=True)
class ScanPolicy:
    max_candidates_total: int
    max_candidates_per_slot: int
    max_candidates_per_sector: int
    pre_cap_candidates_per_slot: int
    min_opportunity_score: float
    signal_score_weight: float
    expected_alpha_weight: float
    freshness_weight: float
    breadth_weight: float
    same_ticker_penalty: float
    same_slot_penalty: float
    same_sector_penalty: float
    same_regime_penalty: float
    learned_ranker_weight: float
    learned_ranker_min_train_rows: int
    learned_ranker_min_train_dates: int
    learned_ranker_slot_weights: dict[str, float]

    @classmethod
    def from_config(cls, config: dict) -> "ScanPolicy":
        policy = config.get("scan_policy", {})
        ranking_weights = policy.get("ranking_weights", {})
        overlap_penalties = policy.get("overlap_penalties", {})
        learned_ranker = policy.get("learned_ranker", {})
        raw_slot_weights = learned_ranker.get("slot_weights", {})
        return cls(
            max_candidates_total=int(policy.get("max_candidates_total", 6)),
            max_candidates_per_slot=int(policy.get("max_candidates_per_slot", 3)),
            max_candidates_per_sector=int(policy.get("max_candidates_per_sector", 3)),
            pre_cap_candidates_per_slot=int(policy.get("pre_cap_candidates_per_slot", 5)),
            min_opportunity_score=float(policy.get("min_opportunity_score", 0.55)),
            signal_score_weight=float(ranking_weights.get("signal_score", 0.35)),
            expected_alpha_weight=float(ranking_weights.get("expected_alpha", 0.30)),
            freshness_weight=float(ranking_weights.get("freshness", 0.20)),
            breadth_weight=float(ranking_weights.get("breadth", 0.15)),
            same_ticker_penalty=float(overlap_penalties.get("same_ticker", 1.0)),
            same_slot_penalty=float(overlap_penalties.get("same_slot", 0.08)),
            same_sector_penalty=float(overlap_penalties.get("same_sector", 0.00)),
            same_regime_penalty=float(overlap_penalties.get("same_regime", 0.00)),
            learned_ranker_weight=float(learned_ranker.get("weight", 1.0)),
            learned_ranker_min_train_rows=int(learned_ranker.get("min_train_rows", 40)),
            learned_ranker_min_train_dates=int(learned_ranker.get("min_train_dates", 8)),
            learned_ranker_slot_weights={
                str(slot): float(weight)
                for slot, weight in raw_slot_weights.items()
            } if isinstance(raw_slot_weights, dict) else {},
        )


class ScanService:
    def __init__(
        self,
        db_manager: DatabaseManager,
        *,
        market_data_client: MarketDataClient | None = None,
        email_sender=send_html_email,
    ) -> None:
        self.db_manager = db_manager
        self.market_data_client = market_data_client or MarketDataClient()
        self.email_sender = email_sender
        self.logger = get_logger("scan")

    def run(self, *, dry_run: bool = False) -> ScanReport:
        self.db_manager.initialize()
        strategies = load_active_strategies()
        settings = get_settings()
        config = load_feature_config()
        scan_policy = ScanPolicy.from_config(config)
        universe_rows = self.db_manager.list_universe_rows(active_only=True)
        if not universe_rows:
            raise ValueError("Universe is empty. Run `sq sync` first.")
        open_trade_loader = getattr(self.db_manager, "list_open_trades", None)
        open_trades = open_trade_loader() if callable(open_trade_loader) else []
        scan_candidate_writer = getattr(self.db_manager, "replace_scan_candidates", None)

        universe_tickers = [row["ticker"] for row in universe_rows]
        tickers = sorted(set(universe_tickers).union(REFERENCE_TICKERS))
        base_history = self.db_manager.load_price_history(tickers)
        earnings_loader = getattr(self.db_manager, "load_earnings_calendar", None)
        earnings_calendar = earnings_loader(universe_tickers) if callable(earnings_loader) else pd.DataFrame()
        recent_history = self._download_recent_daily_history(tickers)
        analysis_frame, _ = build_analysis_frame(
            overlay_price_history(base_history, recent_history),
            universe_rows,
            earnings_calendar=earnings_calendar,
        )
        snapshot = latest_snapshot(analysis_frame)
        snapshot = snapshot[
            snapshot["ticker"].isin(universe_tickers)
            & snapshot["regime_green"].fillna(False)
        ].copy()
        sector_map = {row["ticker"]: row["sector"] for row in universe_rows}
        overlap_context = self._build_overlap_context(
            open_trades=open_trades,
            strategies=strategies,
            sector_map=sector_map,
        )
        candidate_frames: list[pd.DataFrame] = []
        persisted_candidate_frames: list[pd.DataFrame] = []
        for slot, strategy in strategies.items():
            scoped_snapshot = self._scope_snapshot(snapshot, strategy)
            if scoped_snapshot.empty:
                continue
            candidates = filter_signal_candidates(scoped_snapshot, strategy.indicators)
            if candidates.empty:
                continue
            if "signal_score" not in candidates.columns:
                candidates["signal_score"] = 0.0
            candidates = candidates.copy()
            candidates["strategy_slot"] = slot
            candidates["strategy_sector"] = strategy.sector
            scored_records = [
                self._score_candidate(
                    row=row,
                    strategy_slot=slot,
                    strategy=strategy,
                    scan_policy=scan_policy,
                    overlap_context=overlap_context,
                )
                for row in candidates.to_dict(orient="records")
            ]
            scored_frame = pd.DataFrame(scored_records)
            candidates = candidates.merge(
                scored_frame,
                on=["ticker", "strategy_slot", "strategy_sector"],
                how="left",
            )
            candidates["shares"] = candidates.apply(
                lambda row: compute_position_size(
                    price=float(row["adj_close"]),
                    exit_rules=strategy.exit_rules,
                    settings=settings,
                    entry_atr=float(row["atr_14"]) if pd.notna(row.get("atr_14")) else None,
                ),
                axis=1,
            )
            persisted_candidate_frames.append(candidates.copy())
            candidate_frames.append(candidates.copy())

        if not candidate_frames:
            if callable(scan_candidate_writer):
                scan_candidate_writer(scan_date=date.today().isoformat(), rows=[])
            return ScanReport(
                candidate_count=0,
                emailed=False,
                learned_ranker_enabled=False,
                learned_ranker_train_rows=0,
                learned_ranker_train_dates=0,
                learned_ranker_reason="No eligible candidates.",
            )
        candidates = pd.concat(candidate_frames, ignore_index=True).reset_index(drop=True)
        persisted_candidates = pd.concat(persisted_candidate_frames, ignore_index=True).reset_index(drop=True)
        historical_scan_candidates = self.db_manager.load_scan_candidates() if hasattr(self.db_manager, "load_scan_candidates") else pd.DataFrame()
        candidates, ranker_status = self._apply_learned_ranker(
            candidates,
            scan_policy=scan_policy,
            historical_scan_candidates=historical_scan_candidates,
        )
        persisted_candidates = persisted_candidates.merge(
            candidates[
                [
                    "ticker",
                    "strategy_slot",
                    "strategy_sector",
                    "ranker_score",
                    "selection_score",
                    "ranker_enabled",
                    "ranker_top_positive_reasons",
                    "ranker_top_negative_reasons",
                ]
            ],
            on=["ticker", "strategy_slot", "strategy_sector"],
            how="left",
        )
        selected = self._apply_portfolio_caps(candidates, scan_policy)
        selected = selected[selected["opportunity_score"] >= scan_policy.min_opportunity_score].copy()

        selection_keys = {
            (str(row.ticker), str(row.strategy_slot))
            for row in selected.itertuples(index=False)
        }
        selection_ranks = {
            (str(row.ticker), str(row.strategy_slot)): index + 1
            for index, row in enumerate(selected.itertuples(index=False))
        }
        persisted_rows = []
        for row in persisted_candidates.to_dict(orient="records"):
            persisted_rows.append(
                {
                    "ticker": row["ticker"],
                    "strategy_slot": row["strategy_slot"],
                    "strategy_sector": row["strategy_sector"],
                    "sector": row.get("sector"),
                    "md_volume_30d": float(row.get("md_volume_30d", 0.0)) if pd.notna(row.get("md_volume_30d")) else None,
                    "adj_close": float(row.get("adj_close", 0.0)) if pd.notna(row.get("adj_close")) else None,
                    "regime_etf": row.get("regime_etf"),
                    "signal_score": float(row.get("signal_score", 0.0)),
                    "setup_quality_score": float(row.get("setup_quality_score", 0.0)),
                    "expected_alpha_score": float(row.get("expected_alpha_score", 0.0)),
                    "breadth_score": float(row.get("breadth_score", 0.0)),
                    "freshness_score": float(row.get("freshness_score", 0.0)),
                    "overlap_penalty": float(row.get("overlap_penalty", 0.0)),
                    "opportunity_score": float(row.get("opportunity_score", 0.0)),
                    "selected": (str(row["ticker"]), str(row["strategy_slot"])) in selection_keys,
                    "selected_rank": selection_ranks.get((str(row["ticker"]), str(row["strategy_slot"]))),
                    "shares": int(row["shares"]),
                    "details": {
                        "why": self._summarize_candidate_why(row.get("indicator_details", {}) or {}),
                        "already_owned": bool(row.get("already_owned", False)),
                        "pre_penalty_opportunity_score": float(
                            row.get("opportunity_score", 0.0) + row.get("overlap_penalty", 0.0)
                        ),
                        "overlap_components": row.get("overlap_components", {}),
                        "feature_snapshot": self._candidate_feature_snapshot(row),
                        "ranking_components": {
                            "signal_score": float(row.get("signal_score", 0.0)),
                            "setup_quality_score": float(row.get("setup_quality_score", 0.0)),
                            "expected_alpha_score": float(row.get("expected_alpha_score", 0.0)),
                            "breadth_score": float(row.get("breadth_score", 0.0)),
                            "freshness_score": float(row.get("freshness_score", 0.0)),
                            "overlap_penalty": float(row.get("overlap_penalty", 0.0)),
                            "opportunity_score": float(row.get("opportunity_score", 0.0)),
                            "selection_score": float(row.get("selection_score", row.get("opportunity_score", 0.0))),
                            "ranker_score": float(row.get("ranker_score", 0.0)) if pd.notna(row.get("ranker_score")) else None,
                            "ranker_enabled": bool(row.get("ranker_enabled", False)),
                        },
                        "ranker_top_positive_reasons": list(row.get("ranker_top_positive_reasons", ()) or ()),
                        "ranker_top_negative_reasons": list(row.get("ranker_top_negative_reasons", ()) or ()),
                    },
                }
            )
        if callable(scan_candidate_writer):
            scan_candidate_writer(scan_date=date.today().isoformat(), rows=persisted_rows)

        self._log_owned_strength_watchlist(persisted_candidates, scan_policy)

        if selected.empty:
            self.logger.info("Scan produced eligible candidates but none cleared min_opportunity_score=%.2f", scan_policy.min_opportunity_score)
            return ScanReport(
                candidate_count=0,
                emailed=False,
                learned_ranker_enabled=ranker_status.enabled,
                learned_ranker_train_rows=ranker_status.train_rows,
                learned_ranker_train_dates=ranker_status.train_dates,
                learned_ranker_reason=ranker_status.reason,
            )

        selected = selected.sort_values(
            ["strategy_slot", "selection_score", "opportunity_score", "signal_score", "md_volume_30d", "ticker"],
            ascending=[True, False, False, False, False, True],
        ).reset_index(drop=True)
        if dry_run:
            return ScanReport(
                candidate_count=len(selected.index),
                emailed=False,
                learned_ranker_enabled=ranker_status.enabled,
                learned_ranker_train_rows=ranker_status.train_rows,
                learned_ranker_train_dates=ranker_status.train_dates,
                learned_ranker_reason=ranker_status.reason,
            )
        html = self._build_email_html(selected, scan_policy, strategies)
        self.email_sender(
            subject="Evening Brief",
            html_body=html,
            settings=settings,
        )
        return ScanReport(
            candidate_count=len(selected.index),
            emailed=True,
            learned_ranker_enabled=ranker_status.enabled,
            learned_ranker_train_rows=ranker_status.train_rows,
            learned_ranker_train_dates=ranker_status.train_dates,
            learned_ranker_reason=ranker_status.reason,
        )

    def _scope_snapshot(self, snapshot: pd.DataFrame, strategy: ProductionStrategy) -> pd.DataFrame:
        if strategy.sector == "ALL":
            return snapshot.copy()
        return snapshot.loc[snapshot["sector"] == strategy.sector].copy()

    def _download_recent_daily_history(self, tickers: list[str]) -> pd.DataFrame:
        frames = []
        start_date = date.today() - timedelta(days=10)
        try:
            for ticker_batch in chunked(tickers, 50):
                raw_batch = self.market_data_client.download_daily_history(ticker_batch, start_date)
                for ticker in ticker_batch:
                    history = extract_ticker_history(raw_batch, ticker)
                    if not history.empty:
                        frames.append(history)
        except Exception as exc:
            self.logger.warning("Falling back to DuckDB-only daily history in scan: %s", exc)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _apply_portfolio_caps(self, candidates: pd.DataFrame, scan_policy: ScanPolicy) -> pd.DataFrame:
        ranked_candidates = candidates.copy()
        if "selection_score" not in ranked_candidates.columns:
            ranked_candidates["selection_score"] = ranked_candidates["opportunity_score"]
        ranked = ranked_candidates.sort_values(
            ["already_owned", "selection_score", "opportunity_score", "signal_score", "md_volume_30d", "ticker"],
            ascending=[True, False, False, False, False, True],
        ).reset_index(drop=True)
        selected_indices: list[int] = []
        slot_counts: dict[str, int] = {}
        sector_counts: dict[str, int] = {}
        for candidate in ranked.itertuples():
            if len(selected_indices) >= scan_policy.max_candidates_total:
                break
            if bool(candidate.already_owned):
                continue
            slot = str(candidate.strategy_slot)
            sector = str(candidate.sector)
            if slot_counts.get(slot, 0) >= scan_policy.max_candidates_per_slot:
                continue
            if sector_counts.get(sector, 0) >= scan_policy.max_candidates_per_sector:
                continue
            selected_indices.append(int(candidate.Index))
            slot_counts[slot] = slot_counts.get(slot, 0) + 1
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
        if not selected_indices:
            return ranked.iloc[0:0].copy()
        return ranked.loc[selected_indices].copy()

    def _build_email_html(
        self,
        candidates: pd.DataFrame,
        scan_policy: ScanPolicy,
        strategies: dict[str, ProductionStrategy],
    ) -> str:
        sections: list[str] = []
        ordered_slots = (
            candidates.groupby(["strategy_slot", "strategy_sector"], sort=False)["opportunity_score"]
            .max()
            .reset_index()
            .sort_values(["opportunity_score", "strategy_slot"], ascending=[False, True])
        )
        for slot_row in ordered_slots.itertuples(index=False):
            slot_candidates = candidates.loc[candidates["strategy_slot"] == slot_row.strategy_slot].copy()
            slot_candidates = slot_candidates.sort_values(
                ["selection_score", "opportunity_score", "signal_score", "md_volume_30d", "ticker"],
                ascending=[False, False, False, False, True],
            )
            strategy = strategies[str(slot_row.strategy_slot)]
            rows = []
            for candidate in slot_candidates.itertuples(index=False):
                chart_link = f"https://www.tradingview.com/chart/?symbol={candidate.ticker}"
                stop_price, target_price = self._candidate_exit_plan(candidate, strategy)
                why = self._summarize_candidate_why(getattr(candidate, "indicator_details", {}) or {})
                rank_context = (
                    f"select {float(getattr(candidate, 'selection_score', candidate.opportunity_score)):.2f}, "
                    f"alpha {float(candidate.expected_alpha_score):.2f}, "
                    f"fresh {float(candidate.freshness_score):.2f}, "
                    f"breadth {float(candidate.breadth_score):.2f}, "
                    f"overlap -{float(candidate.overlap_penalty):.2f}"
                )
                ranker_score = getattr(candidate, "ranker_score", math.nan)
                if self._is_finite(ranker_score):
                    rank_context += f", learned {float(ranker_score):.3f}"
                rows.append(
                    "<tr>"
                    f"<td>{candidate.ticker}</td>"
                    f"<td>{candidate.sector}</td>"
                    f"<td>{candidate.regime_etf} Green</td>"
                    f"<td>{candidate.opportunity_score:.2f}</td>"
                    f"<td>{candidate.signal_score:.1f}</td>"
                    f"<td>{candidate.adj_close:.2f}</td>"
                    f"<td>{stop_price}</td>"
                    f"<td>{target_price}</td>"
                    f"<td>{candidate.shares}</td>"
                    f"<td>{why}</td>"
                    f"<td>{rank_context}</td>"
                    f"<td><a href=\"{chart_link}\">chart</a></td>"
                    "</tr>"
                )
            sections.append(
                "<section>"
                f"<h2>Slot: {slot_row.strategy_slot}</h2>"
                f"<p>Strategy sector scope: {slot_row.strategy_sector} | Candidates: {len(slot_candidates.index)}</p>"
                "<table border='1' cellpadding='6' cellspacing='0'>"
                "<tr><th>Ticker</th><th>Sector</th><th>Regime</th><th>Opportunity</th><th>Signal Score</th><th>Adj Close</th><th>Stop</th><th>Target</th><th>Shares</th><th>Why</th><th>Rank Context</th><th>Chart</th></tr>"
                f"{''.join(rows)}"
                "</table>"
                "</section>"
            )
        return (
            "<html><body>"
            "<h1>Evening Brief</h1>"
            f"<p>Portfolio caps: total={scan_policy.max_candidates_total}, per_slot={scan_policy.max_candidates_per_slot}, per_sector={scan_policy.max_candidates_per_sector} | min_opportunity_score={scan_policy.min_opportunity_score:.2f}</p>"
            f"{''.join(sections)}"
            "</body></html>"
        )

    def _apply_learned_ranker(
        self,
        candidates: pd.DataFrame,
        *,
        scan_policy: ScanPolicy,
        historical_scan_candidates: pd.DataFrame,
    ) -> tuple[pd.DataFrame, LearnedRankerStatus]:
        scored = candidates.copy()
        scored["ranker_score"] = math.nan
        scored["selection_score"] = scored["opportunity_score"].astype(float)
        scored["ranker_enabled"] = False
        scored["ranker_top_positive_reasons"] = [tuple() for _ in range(len(scored.index))]
        scored["ranker_top_negative_reasons"] = [tuple() for _ in range(len(scored.index))]
        if historical_scan_candidates.empty:
            status = LearnedRankerStatus(
                enabled=False,
                train_rows=0,
                train_dates=0,
                reason="No historical scan candidates found.",
            )
            self.logger.info("Learned ranker disabled in scan: %s", status.reason)
            return scored, status
        ranker = CandidateRanker(
            target_column="alpha_vs_sector_10d",
            min_train_rows=scan_policy.learned_ranker_min_train_rows,
            min_train_dates=scan_policy.learned_ranker_min_train_dates,
        )
        labeled = historical_scan_candidates.dropna(subset=["alpha_vs_sector_10d"]).copy()
        if labeled.empty:
            status = LearnedRankerStatus(
                enabled=False,
                train_rows=0,
                train_dates=0,
                reason="No labeled alpha_vs_sector_10d scan history found.",
            )
            self.logger.info("Learned ranker disabled in scan: %s", status.reason)
            return scored, status
        if "scan_date" in labeled.columns:
            today_str = date.today().isoformat()
            labeled = labeled[pd.to_datetime(labeled["scan_date"]).dt.normalize() < pd.Timestamp(today_str).normalize()].copy()
        train_rows = len(labeled.index)
        train_dates = int(labeled["scan_date"].nunique()) if "scan_date" in labeled.columns and not labeled.empty else 0
        if train_rows < ranker.min_train_rows or train_dates < ranker.min_train_dates:
            reason = (
                f"Insufficient labeled history: train_rows={train_rows} min_rows={ranker.min_train_rows} "
                f"train_dates={train_dates} min_dates={ranker.min_train_dates}"
            )
            self.logger.info(
                "Learned ranker disabled in scan: %s",
                reason,
            )
            return scored, LearnedRankerStatus(
                enabled=False,
                train_rows=train_rows,
                train_dates=train_dates,
                reason=reason,
            )
        try:
            ranker.fit(labeled)
            scored = ranker.score_details(scored, top_features=2)
        except Exception as exc:
            reason = f"Scoring failure: {exc}"
            self.logger.warning("Learned ranker disabled in scan after scoring failure: %s", exc)
            return (
                candidates.assign(
                    ranker_score=math.nan,
                    selection_score=candidates["opportunity_score"].astype(float),
                    ranker_enabled=False,
                    ranker_top_positive_reasons=[tuple() for _ in range(len(candidates.index))],
                    ranker_top_negative_reasons=[tuple() for _ in range(len(candidates.index))],
                ),
                LearnedRankerStatus(
                    enabled=False,
                    train_rows=train_rows,
                    train_dates=train_dates,
                    reason=reason,
                ),
            )
        ranker_values = pd.to_numeric(scored["ranker_score"], errors="coerce").clip(lower=-0.10, upper=0.10)
        scored["learned_ranker_weight"] = scored["strategy_slot"].map(
            lambda slot: self._learned_ranker_weight_for_slot(str(slot), scan_policy)
        )
        scored["selection_score"] = (
            scored["opportunity_score"].astype(float)
            + (ranker_values * scored["learned_ranker_weight"].astype(float))
        )
        scored["ranker_enabled"] = True
        self.logger.info(
            "Learned ranker applied in scan: train_rows=%s train_dates=%s base_weight=%.2f slot_weights=%s",
            train_rows,
            train_dates,
            float(scan_policy.learned_ranker_weight),
            scan_policy.learned_ranker_slot_weights,
        )
        return scored, LearnedRankerStatus(
            enabled=True,
            train_rows=train_rows,
            train_dates=train_dates,
            reason=None,
        )

    def _learned_ranker_weight_for_slot(self, strategy_slot: str, scan_policy: ScanPolicy) -> float:
        if strategy_slot in scan_policy.learned_ranker_slot_weights:
            return float(scan_policy.learned_ranker_slot_weights[strategy_slot])
        return float(scan_policy.learned_ranker_weight)

    def _log_owned_strength_watchlist(self, candidates: pd.DataFrame, scan_policy: ScanPolicy) -> None:
        if candidates.empty or "already_owned" not in candidates.columns:
            return
        owned = candidates[candidates["already_owned"].fillna(False)].copy()
        if owned.empty:
            return
        owned["pre_penalty_opportunity_score"] = (
            owned["opportunity_score"].astype(float) + owned["overlap_penalty"].astype(float)
        )
        strong_owned = owned[owned["pre_penalty_opportunity_score"] >= scan_policy.min_opportunity_score].copy()
        if strong_owned.empty:
            return
        ranked = strong_owned.sort_values(
            ["pre_penalty_opportunity_score", "signal_score", "ticker"],
            ascending=[False, False, True],
        )
        summary = ", ".join(
            f"{row.ticker} pre={float(row.pre_penalty_opportunity_score):.2f} post={float(row.opportunity_score):.2f}"
            for row in ranked.head(5).itertuples(index=False)
        )
        self.logger.info("Owned strength watchlist: %s", summary)

    def _candidate_exit_plan(self, candidate, strategy: ProductionStrategy) -> tuple[str, str]:
        entry_atr = float(candidate.atr_14) if pd.notna(getattr(candidate, "atr_14", pd.NA)) else None
        try:
            stop_price = trailing_stop_price(
                max_price_seen=float(candidate.adj_close),
                entry_atr=entry_atr,
                exit_rules=strategy.exit_rules,
            )
            target_price = profit_target_price(
                entry_price=float(candidate.adj_close),
                entry_atr=entry_atr,
                exit_rules=strategy.exit_rules,
            )
            return f"{stop_price:.2f}", f"{target_price:.2f}"
        except Exception:
            return "n/a", "n/a"

    def _summarize_candidate_why(self, indicator_details: dict) -> str:
        if not indicator_details:
            return "n/a"
        scored = []
        hard = []
        for name, detail in indicator_details.items():
            if name == "signal_score_min":
                continue
            mode = detail.get("mode")
            passed = bool(detail.get("passed"))
            if mode == "hard_filter" and passed:
                hard.append(name.replace("_", " "))
            elif mode == "score_component" and passed:
                scored.append((float(detail.get("score", 0.0)), name.replace("_", " ")))
        scored.sort(reverse=True)
        parts = hard[:2] + [name for _, name in scored[:2]]
        return ", ".join(parts) if parts else "n/a"

    def _candidate_feature_snapshot(self, row: dict) -> dict[str, object]:
        snapshot: dict[str, object] = {}
        for name in [
            "adj_close",
            "atr_14",
            "md_volume_30d",
            "regime_etf",
            "relative_strength_index_vs_spy",
            "roc_63",
            "vol_alpha",
            "sma_200_dist",
            "sma_50_dist",
            "rsi_14",
            "days_to_next_earnings",
            "days_since_last_earnings",
            "avg_abs_gap_pct_20",
            "max_gap_down_pct_60",
            "distance_above_20d_high",
            "base_range_pct_20",
            "base_atr_contraction_20",
            "base_volume_dryup_ratio_20",
            "breakout_volume_ratio_50",
            "sector_pct_above_50",
            "sector_pct_above_200",
            "sector_median_roc_63",
        ]:
            value = row.get(name)
            if isinstance(value, (str, bool)) or pd.isna(value):
                if value in (None, "") or pd.isna(value):
                    continue
                snapshot[name] = value
                continue
            try:
                snapshot[name] = float(value)
            except (TypeError, ValueError):
                continue
        if row.get("indicator_details"):
            snapshot["indicator_details"] = row.get("indicator_details")
        return snapshot

    def _build_overlap_context(
        self,
        *,
        open_trades: list,
        strategies: dict[str, ProductionStrategy],
        sector_map: dict[str, str],
    ) -> dict[str, set[str]]:
        held_tickers: set[str] = set()
        held_slots: set[str] = set()
        held_sectors: set[str] = set()
        held_regimes: set[str] = set()
        backtest_lookup = getattr(self.db_manager, "get_backtest_result_by_strategy_id", None)
        for trade in open_trades:
            ticker = str(trade["ticker"])
            held_tickers.add(ticker)
            resolution = resolve_trade_strategy(
                trade=trade,
                strategies=strategies,
                sector_map=sector_map,
                backtest_lookup=backtest_lookup,
            )
            if resolution.strategy is not None:
                held_slots.add(str(resolution.strategy.slot))
                if resolution.strategy.sector not in ("", "ALL"):
                    held_sectors.add(str(resolution.strategy.sector))
                    held_regimes.add(str(self._regime_for_sector(resolution.strategy.sector)))
                continue
            stored_slot = trade["strategy_slot"] if "strategy_slot" in trade.keys() else None
            if stored_slot:
                held_slots.add(str(stored_slot))
            sector = sector_map.get(ticker)
            if sector:
                held_sectors.add(str(sector))
                held_regimes.add(str(self._regime_for_sector(sector)))
        return {
            "tickers": held_tickers,
            "slots": held_slots,
            "sectors": held_sectors,
            "regimes": held_regimes,
        }

    def _score_candidate(
        self,
        *,
        row: dict,
        strategy_slot: str,
        strategy: ProductionStrategy,
        scan_policy: ScanPolicy,
        overlap_context: dict[str, set[str]],
    ) -> dict[str, object]:
        ticker = str(row["ticker"])
        sector = str(row.get("sector", ""))
        regime = str(row.get("regime_etf", self._regime_for_sector(sector)))
        pass_score = self._indicator_pass_score(row.get("indicator_details", {}) or {}, float(row.get("signal_score", 0.0)))
        setup_quality_score = self._clamp(float(row.get("signal_score", 0.0)) / max(pass_score * 1.5, 1.0))
        expected_alpha_score = self._expected_alpha_score(row)
        breadth_score = self._breadth_score(row)
        freshness_score = self._freshness_score(row)
        overlap_components = {
            "same_ticker": 0.0,
            "same_slot": 0.0,
            "same_sector": 0.0,
            "same_regime": 0.0,
        }
        overlap_penalty = 0.0
        already_owned = ticker in overlap_context["tickers"]
        if already_owned:
            overlap_components["same_ticker"] = scan_policy.same_ticker_penalty
            overlap_penalty += scan_policy.same_ticker_penalty
        if strategy.slot in overlap_context["slots"]:
            overlap_components["same_slot"] = scan_policy.same_slot_penalty
            overlap_penalty += scan_policy.same_slot_penalty
        if sector in overlap_context["sectors"]:
            overlap_components["same_sector"] = scan_policy.same_sector_penalty
            overlap_penalty += scan_policy.same_sector_penalty
        if regime in overlap_context["regimes"]:
            overlap_components["same_regime"] = scan_policy.same_regime_penalty
            overlap_penalty += scan_policy.same_regime_penalty
        overlap_penalty = min(overlap_penalty, 1.0)
        opportunity_score = (
            (setup_quality_score * scan_policy.signal_score_weight)
            + (expected_alpha_score * scan_policy.expected_alpha_weight)
            + (freshness_score * scan_policy.freshness_weight)
            + (breadth_score * scan_policy.breadth_weight)
            - overlap_penalty
        )
        return {
            "ticker": ticker,
            "strategy_slot": strategy_slot,
            "strategy_sector": strategy.sector,
            "setup_quality_score": setup_quality_score,
            "expected_alpha_score": expected_alpha_score,
            "breadth_score": breadth_score,
            "freshness_score": freshness_score,
            "overlap_penalty": overlap_penalty,
            "overlap_components": overlap_components,
            "opportunity_score": opportunity_score,
            "already_owned": already_owned,
        }

    def _indicator_pass_score(self, indicator_details: dict, default_signal_score: float) -> float:
        threshold_detail = indicator_details.get("signal_score_min")
        if isinstance(threshold_detail, dict):
            threshold = threshold_detail.get("threshold")
            if threshold is not None:
                try:
                    return float(threshold)
                except (TypeError, ValueError):
                    return max(default_signal_score, 1.0)
        return max(default_signal_score, 1.0)

    def _expected_alpha_score(self, row: dict) -> float:
        rs_component = self._clamp(self._numeric(row.get("relative_strength_index_vs_spy")) / 100.0)
        roc_component = self._clamp(self._numeric(row.get("roc_63")) / 0.25)
        volume_component = self._clamp((self._numeric(row.get("vol_alpha")) - 1.0) / 1.0)
        trend_component = self._clamp((self._numeric(row.get("sma_200_dist")) + 0.02) / 0.22)
        return (
            (rs_component * 0.35)
            + (roc_component * 0.30)
            + (volume_component * 0.15)
            + (trend_component * 0.20)
        )

    def _breadth_score(self, row: dict) -> float:
        above_50 = self._clamp(self._numeric(row.get("sector_pct_above_50")))
        above_200 = self._clamp(self._numeric(row.get("sector_pct_above_200")))
        median_roc = self._clamp((self._numeric(row.get("sector_median_roc_63")) + 0.02) / 0.17)
        return (above_50 * 0.35) + (above_200 * 0.35) + (median_roc * 0.30)

    def _freshness_score(self, row: dict) -> float:
        breakout_extension = row.get("distance_above_20d_high")
        breakout_component = (
            1.0 - self._clamp(self._numeric(breakout_extension) / 0.05)
            if self._is_finite(breakout_extension)
            else math.nan
        )
        rsi_component = (
            1.0 - self._clamp(abs(self._numeric(row.get("rsi_14")) - 50.0) / 25.0)
            if self._is_finite(row.get("rsi_14"))
            else math.nan
        )
        extension_component = (
            1.0 - self._clamp(max(self._numeric(row.get("sma_50_dist")), 0.0) / 0.15)
            if self._is_finite(row.get("sma_50_dist"))
            else math.nan
        )
        gap_component = (
            1.0 - self._clamp(self._numeric(row.get("avg_abs_gap_pct_20")) / 0.05)
            if self._is_finite(row.get("avg_abs_gap_pct_20"))
            else math.nan
        )
        components = [value for value in [breakout_component, rsi_component, extension_component, gap_component] if self._is_finite(value)]
        if not components:
            return 0.5
        return float(sum(components) / len(components))

    def _regime_for_sector(self, sector: str) -> str:
        return "QQQ" if sector in {"Information Technology", "Communication Services"} else "SPY"

    def _numeric(self, value) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("nan")

    def _is_finite(self, value) -> bool:
        try:
            return math.isfinite(float(value))
        except (TypeError, ValueError):
            return False

    def _clamp(self, value: float, lower: float = 0.0, upper: float = 1.0) -> float:
        if not math.isfinite(value):
            return lower
        return max(lower, min(upper, value))
