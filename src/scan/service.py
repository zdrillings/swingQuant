from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import json
import math

import numpy as np
import pandas as pd

from src.scan.analyst_data import AnalystContext, AnalystDataClient
from src.scan.ranker import CandidateRanker, RankerValidationReport
from src.settings import get_settings, load_feature_config
from src.sync.market_data import MarketDataClient, chunked, extract_ticker_history
from src.sync.service import REFERENCE_TICKERS
from src.utils.db_manager import DatabaseManager
from src.utils.emailer import send_html_email
from src.utils.logging import get_logger
from src.utils.shortlist_runtime import load_live_shortlist_model_context
from src.utils.signal_engine import build_analysis_frame, filter_signal_candidates, latest_snapshot, overlay_price_history
from src.utils.sizing import compute_position_size
from src.utils.strategy import (
    ProductionStrategy,
    SIGNAL_SCORE_MIN_KEY,
    evaluate_signal_gate,
    load_active_strategies,
    profit_target_price,
    resolve_trade_strategy,
    split_signal_indicators,
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
class ShortlistModelPolicy:
    enabled: bool
    horizon_days: int
    top_n: int
    min_opportunity_score: float
    min_train_dates: int
    test_window_dates: int
    recent_dates: int
    refresh_if_stale: bool
    use_as_candidate_source: bool
    eligible_universe_mode: str
    production_eligible_universe_mode: str
    production_model_scope: str
    production_model_name: str | None
    production_xgboost_config: str


@dataclass(frozen=True)
class ScanPolicy:
    max_candidates_total: int
    max_candidates_per_slot: int
    max_candidates_per_sector: int
    pre_cap_candidates_per_slot: int
    max_snapshot_age_days: int | None
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
    learned_ranker_validation_enabled: bool
    learned_ranker_min_q1_q5_spread: float
    learned_ranker_min_ic_mean: float
    learned_ranker_min_ic_dates: int
    learned_ranker_min_validation_blocks: int
    recent_selection_memory_enabled: bool
    recent_selection_memory_scan_dates: int
    recent_drag_min_picks: int
    recent_drag_penalty_scale: float
    recent_drag_penalty_cap: float
    recent_missed_winner_min_count: int
    recent_missed_winner_boost_scale: float
    recent_missed_winner_boost_cap: float
    recent_missed_winner_min_gap: float
    recent_swap_feedback_gap: float
    recent_swap_max_opportunity_gap: float
    slot_selection_overlay_enabled: bool
    slot_selection_overlay_weights: dict[str, dict[str, float]]
    shortlist_model: ShortlistModelPolicy

    @classmethod
    def from_config(cls, config: dict) -> "ScanPolicy":
        policy = config.get("scan_policy", {})
        ranking_weights = policy.get("ranking_weights", {})
        overlap_penalties = policy.get("overlap_penalties", {})
        learned_ranker = policy.get("learned_ranker", {})
        validation_gate = learned_ranker.get("validation_gate", {})
        raw_slot_weights = learned_ranker.get("slot_weights", {})
        selection_memory = policy.get("recent_selection_memory", {})
        slot_selection_overlay = policy.get("slot_selection_overlay", {})
        raw_overlay_weights = slot_selection_overlay.get("slot_weights", {})
        shortlist_model = policy.get("shortlist_model", {})
        return cls(
            max_candidates_total=int(policy.get("max_candidates_total", 6)),
            max_candidates_per_slot=int(policy.get("max_candidates_per_slot", 3)),
            max_candidates_per_sector=int(policy.get("max_candidates_per_sector", 3)),
            pre_cap_candidates_per_slot=int(policy.get("pre_cap_candidates_per_slot", 5)),
            max_snapshot_age_days=(
                int(policy["max_snapshot_age_days"])
                if policy.get("max_snapshot_age_days") not in (None, "")
                else None
            ),
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
            learned_ranker_validation_enabled=bool(validation_gate.get("enabled", True)),
            learned_ranker_min_q1_q5_spread=float(validation_gate.get("min_q1_q5_spread", 0.0)),
            learned_ranker_min_ic_mean=float(validation_gate.get("min_ic_mean", 0.0)),
            learned_ranker_min_ic_dates=int(validation_gate.get("min_ic_dates", 5)),
            learned_ranker_min_validation_blocks=int(validation_gate.get("min_validation_blocks", 1)),
            recent_selection_memory_enabled=bool(selection_memory.get("enabled", True)),
            recent_selection_memory_scan_dates=int(selection_memory.get("recent_scan_dates", 20)),
            recent_drag_min_picks=int(selection_memory.get("drag_min_picks", 2)),
            recent_drag_penalty_scale=float(selection_memory.get("drag_penalty_scale", 1.0)),
            recent_drag_penalty_cap=float(selection_memory.get("drag_penalty_cap", 0.10)),
            recent_missed_winner_min_count=int(selection_memory.get("missed_winner_min_count", 2)),
            recent_missed_winner_boost_scale=float(selection_memory.get("missed_winner_boost_scale", 0.25)),
            recent_missed_winner_boost_cap=float(selection_memory.get("missed_winner_boost_cap", 0.10)),
            recent_missed_winner_min_gap=float(selection_memory.get("missed_winner_min_gap", 0.05)),
            recent_swap_feedback_gap=float(selection_memory.get("swap_feedback_gap", 0.03)),
            recent_swap_max_opportunity_gap=float(selection_memory.get("swap_max_opportunity_gap", 0.10)),
            slot_selection_overlay_enabled=bool(slot_selection_overlay.get("enabled", False)),
            slot_selection_overlay_weights={
                str(slot): {
                    str(name): float(weight)
                    for name, weight in weights.items()
                }
                for slot, weights in raw_overlay_weights.items()
                if isinstance(weights, dict)
            } if isinstance(raw_overlay_weights, dict) else {},
            shortlist_model=ShortlistModelPolicy(
                enabled=bool(shortlist_model.get("enabled", True)),
                horizon_days=int(shortlist_model.get("horizon_days", 20)),
                top_n=int(shortlist_model.get("top_n", 10)),
                min_opportunity_score=float(shortlist_model.get("min_opportunity_score", 0.30)),
                min_train_dates=int(shortlist_model.get("min_train_dates", 252)),
                test_window_dates=int(shortlist_model.get("test_window_dates", 20)),
                recent_dates=int(shortlist_model.get("recent_dates", 60)),
                refresh_if_stale=bool(shortlist_model.get("refresh_if_stale", True)),
                use_as_candidate_source=bool(shortlist_model.get("use_as_candidate_source", True)),
                eligible_universe_mode=str(shortlist_model.get("eligible_universe_mode", "passed_only")),
                production_eligible_universe_mode=str(
                    shortlist_model.get(
                        "production_eligible_universe_mode",
                        shortlist_model.get("eligible_universe_mode", "passed_only"),
                    )
                ),
                production_model_scope=str(shortlist_model.get("production_model_scope", "global")),
                production_model_name=(
                    str(shortlist_model.get("production_model_name")).strip()
                    if shortlist_model.get("production_model_name") not in (None, "")
                    else "xgboost_model"
                ),
                production_xgboost_config=str(shortlist_model.get("production_xgboost_config", "baseline") or "baseline"),
            ),
        )


class ScanService:
    def __init__(
        self,
        db_manager: DatabaseManager,
        *,
        market_data_client: MarketDataClient | None = None,
        analyst_data_client: AnalystDataClient | None = None,
        email_sender=send_html_email,
    ) -> None:
        self.db_manager = db_manager
        self.market_data_client = market_data_client or MarketDataClient()
        self.analyst_data_client = analyst_data_client or AnalystDataClient()
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
        scan_slot_diagnostic_writer = getattr(self.db_manager, "replace_scan_slot_diagnostics", None)

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
        self._validate_snapshot_freshness(snapshot=snapshot, scan_policy=scan_policy)
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
        shortlist_model_context = self._load_shortlist_model_context(scan_policy)
        use_model_candidate_source = (
            shortlist_model_context is not None
            and scan_policy.shortlist_model.use_as_candidate_source
        )
        if use_model_candidate_source:
            candidates = self._build_shortlist_model_candidates(
                snapshot=snapshot,
                strategies=strategies,
                shortlist_model_context=shortlist_model_context,
                scan_policy=scan_policy,
                overlap_context=overlap_context,
                settings=settings,
            )
            persisted_candidates = candidates.copy()
            slot_gate_diagnostics = self._build_shortlist_model_slot_diagnostics(
                candidates=candidates,
                strategies=strategies,
            )
        else:
            slot_gate_diagnostics = self._build_slot_gate_diagnostics(
                full_snapshot=snapshot,
                regime_snapshot=snapshot,
                strategies=strategies,
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

            if candidate_frames:
                candidates = pd.concat(candidate_frames, ignore_index=True).reset_index(drop=True)
                persisted_candidates = pd.concat(persisted_candidate_frames, ignore_index=True).reset_index(drop=True)
            else:
                candidates = pd.DataFrame()
                persisted_candidates = pd.DataFrame()

        if candidates.empty:
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
        historical_scan_candidates = self.db_manager.load_scan_candidates() if hasattr(self.db_manager, "load_scan_candidates") else pd.DataFrame()
        if shortlist_model_context is not None:
            candidates = self._apply_shortlist_model_selection(candidates, shortlist_model_context)
            ranker_status = LearnedRankerStatus(
                enabled=False,
                train_rows=0,
                train_dates=0,
                reason=f"Using shortlist model {shortlist_model_context.champion_model}.",
            )
        else:
            candidates, ranker_status = self._apply_learned_ranker(
                candidates,
                scan_policy=scan_policy,
                historical_scan_candidates=historical_scan_candidates,
            )
            candidates = self._apply_recent_selection_memory(
                candidates,
                scan_policy=scan_policy,
                historical_scan_candidates=historical_scan_candidates,
            )
            candidates = self._apply_slot_selection_overlays(
                candidates,
                scan_policy=scan_policy,
            )
        selection_opportunity_floor = self._selection_opportunity_floor(
            scan_policy=scan_policy,
            shortlist_model_context=shortlist_model_context,
        )
        slot_post_gate_dropoff = self._build_slot_post_gate_dropoff(
            candidates,
            strategies=strategies,
            min_opportunity_score=selection_opportunity_floor,
        )
        if "selection_source" not in candidates.columns:
            candidates["selection_source"] = "heuristic"
        if "model_predicted_alpha" not in candidates.columns:
            candidates["model_predicted_alpha"] = pd.NA
        if "model_rank" not in candidates.columns:
            candidates["model_rank"] = pd.NA
        if "model_generated_at" not in candidates.columns:
            candidates["model_generated_at"] = pd.NA
        if "model_name" not in candidates.columns:
            candidates["model_name"] = pd.NA
        if "model_reason_summary" not in candidates.columns:
            candidates["model_reason_summary"] = pd.NA
        if "model_comparison_summary" not in candidates.columns:
            candidates["model_comparison_summary"] = pd.NA
        persisted_candidates = self._attach_selection_metadata_to_persisted_candidates(
            persisted_candidates,
            candidates,
        )
        eligible_candidates = candidates[
            pd.to_numeric(candidates["opportunity_score"], errors="coerce") >= selection_opportunity_floor
        ].copy()
        selected = self._apply_portfolio_caps(eligible_candidates, scan_policy)
        persisted_rows = self._build_persisted_scan_rows(persisted_candidates, selected)
        if callable(scan_candidate_writer):
            scan_candidate_writer(scan_date=date.today().isoformat(), rows=persisted_rows)
        if callable(scan_slot_diagnostic_writer):
            diagnostic_rows = []
            for slot, strategy in strategies.items():
                gate_payload = slot_gate_diagnostics.get(str(slot), {})
                dropoff_payload = slot_post_gate_dropoff.get(str(slot), {})
                diagnostic_rows.append(
                    {
                        "strategy_slot": str(slot),
                        "strategy_sector": str(strategy.sector),
                        "gate_counts": gate_payload.get("counts", []),
                        "first_zero_gate": gate_payload.get("first_zero_gate", "unavailable"),
                        "component_positive_counts": gate_payload.get("component_positive_counts", []),
                        "gated_count": dropoff_payload.get("gated_count", 0),
                        "cleared_opportunity_count": dropoff_payload.get("cleared_count", 0),
                        "dropped_after_opportunity_count": dropoff_payload.get("dropped_count", 0),
                        "avg_gated_opportunity_score": dropoff_payload.get("avg_opportunity_score"),
                        "top_cleared": dropoff_payload.get("top_cleared", []),
                        "top_dropped": dropoff_payload.get("top_dropped", []),
                        "drop_examples": dropoff_payload.get("drop_examples", []),
                    }
                )
            scan_slot_diagnostic_writer(scan_date=date.today().isoformat(), rows=diagnostic_rows)

        self._log_owned_strength_watchlist(persisted_candidates, scan_policy)

        if selected.empty:
            self.logger.info("Scan produced eligible candidates but none cleared min_opportunity_score=%.2f", selection_opportunity_floor)
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
        earnings_lookup = self._build_earnings_lookup(
            earnings_calendar=earnings_calendar,
            as_of=date.today(),
        )
        html = self._build_email_html(
            selected,
            scan_policy,
            strategies,
            earnings_lookup=earnings_lookup,
            analyst_contexts=self._load_analyst_contexts(selected=selected, all_candidates=candidates),
            all_candidates=candidates,
            open_trade_tickers=self._open_trade_tickers(open_trades),
            open_trades=open_trades,
        )
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

    def _build_shortlist_model_candidates(
        self,
        *,
        snapshot: pd.DataFrame,
        strategies: dict[str, ProductionStrategy],
        shortlist_model_context,
        scan_policy: ScanPolicy,
        overlap_context: dict[str, set[str]],
        settings,
    ) -> pd.DataFrame:
        if snapshot.empty:
            return snapshot.iloc[0:0].copy()
        strategy_map = self._build_model_strategy_map(strategies)
        if not strategy_map:
            return snapshot.iloc[0:0].copy()
        predictions = shortlist_model_context.live_predictions.copy()
        if predictions.empty:
            return snapshot.iloc[0:0].copy()
        predictions = predictions[["ticker", "predicted_alpha", "model_rank"]].copy()
        merged = snapshot.merge(predictions, on="ticker", how="inner")
        if merged.empty:
            return merged
        if "signal_score" not in merged.columns:
            merged["signal_score"] = 0.0
        fallback_strategy = strategy_map.get("__fallback__")
        strategy_keys = merged["sector"].map(lambda sector: strategy_map.get(str(sector), fallback_strategy))
        merged = merged.assign(_strategy_key=strategy_keys)
        merged = merged[merged["_strategy_key"].notna()].copy()
        if merged.empty:
            return merged
        merged["strategy_slot"] = merged["_strategy_key"].map(lambda key: key[0])
        merged["strategy_sector"] = merged["_strategy_key"].map(lambda key: key[1])
        merged["indicator_details"] = [{} for _ in range(len(merged.index))]
        scored_records = []
        for row in merged.to_dict(orient="records"):
            slot = str(row["strategy_slot"])
            strategy = strategies[slot]
            scored_records.append(
                self._score_candidate(
                    row=row,
                    strategy_slot=slot,
                    strategy=strategy,
                    scan_policy=scan_policy,
                    overlap_context=overlap_context,
                )
            )
        scored_frame = pd.DataFrame(scored_records)
        merged = merged.merge(
            scored_frame,
            on=["ticker", "strategy_slot", "strategy_sector"],
            how="left",
        )
        merged["shares"] = merged.apply(
            lambda row: compute_position_size(
                price=float(row["adj_close"]),
                exit_rules=strategies[str(row["strategy_slot"])].exit_rules,
                settings=settings,
                entry_atr=float(row["atr_14"]) if pd.notna(row.get("atr_14")) else None,
            ),
            axis=1,
        )
        return merged.drop(columns=["_strategy_key"])

    def _build_model_strategy_map(
        self,
        strategies: dict[str, ProductionStrategy],
    ) -> dict[str, tuple[str, str]]:
        exact: dict[str, tuple[str, str]] = {}
        fallback_slot: tuple[str, str] | None = None
        ordered = sorted(
            ((str(slot), strategy) for slot, strategy in strategies.items()),
            key=lambda item: item[0],
        )
        for slot, strategy in ordered:
            strategy_sector = str(strategy.sector)
            if strategy_sector == "ALL":
                if fallback_slot is None:
                    fallback_slot = (slot, strategy_sector)
                continue
            exact.setdefault(strategy_sector, (slot, strategy_sector))
        if fallback_slot is None:
            return exact
        mapped = dict(exact)
        return mapped | {"__fallback__": fallback_slot}

    def _build_shortlist_model_slot_diagnostics(
        self,
        *,
        candidates: pd.DataFrame,
        strategies: dict[str, ProductionStrategy],
    ) -> dict[str, dict[str, object]]:
        diagnostics: dict[str, dict[str, object]] = {}
        for slot, strategy in strategies.items():
            slot_frame = candidates[candidates["strategy_slot"].astype(str) == str(slot)].copy()
            counts = [
                ("regime_green", int(len(candidates.index))),
                ("sector_scope", int(len(slot_frame.index))),
                ("model_scored", int(len(slot_frame.index))),
            ]
            first_zero_gate = "sector_scope" if slot_frame.empty else "none"
            diagnostics[str(slot)] = {
                "counts": counts,
                "first_zero_gate": first_zero_gate,
                "component_positive_counts": [],
            }
        return diagnostics

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

    def _load_analyst_contexts(
        self,
        *,
        selected: pd.DataFrame,
        all_candidates: pd.DataFrame | None,
    ) -> dict[str, AnalystContext]:
        tickers = {str(ticker).upper() for ticker in selected["ticker"].dropna().astype(str).tolist()}
        if all_candidates is not None and not all_candidates.empty:
            working = all_candidates.copy()
            if "model_predicted_alpha" not in working.columns:
                working["model_predicted_alpha"] = working.get("selection_score", 0.0)
            working["pre_penalty_opportunity_score"] = (
                pd.to_numeric(working.get("opportunity_score", 0.0), errors="coerce").fillna(float("-inf"))
                + pd.to_numeric(working.get("overlap_penalty", 0.0), errors="coerce").fillna(0.0)
            )
            working["model_predicted_alpha"] = pd.to_numeric(working["model_predicted_alpha"], errors="coerce")
            targetable = working[
                (working["pre_penalty_opportunity_score"].astype(float) >= 0.40)
                & (working["model_predicted_alpha"].fillna(float("-inf")) > 0.0)
            ].copy()
            targetable = targetable.sort_values(
                ["model_predicted_alpha", "pre_penalty_opportunity_score", "ticker"],
                ascending=[False, False, True],
            ).head(10)
            tickers.update(str(ticker).upper() for ticker in targetable["ticker"].dropna().astype(str).tolist())
        if not tickers:
            return {}
        try:
            return self.analyst_data_client.load_contexts(sorted(tickers))
        except Exception as exc:
            self.logger.warning("Unable to load analyst context for scan email: %s", exc)
            return {}

    def _validate_snapshot_freshness(self, *, snapshot: pd.DataFrame, scan_policy: ScanPolicy) -> None:
        if scan_policy.max_snapshot_age_days is None:
            return
        if snapshot.empty or "date" not in snapshot.columns:
            raise ValueError("Scan snapshot is empty; refusing to run nightly scan without current prices.")
        snapshot_dates = pd.to_datetime(snapshot["date"], errors="coerce").dropna()
        if snapshot_dates.empty:
            raise ValueError("Scan snapshot has no valid dates; refusing to run nightly scan without current prices.")
        latest_date = snapshot_dates.max().date()
        age_days = (date.today() - latest_date).days
        if age_days > scan_policy.max_snapshot_age_days:
            raise ValueError(
                "Scan snapshot is stale; refusing to run nightly scan. "
                f"latest_snapshot={latest_date.isoformat()}, "
                f"age_days={age_days}, "
                f"max_snapshot_age_days={scan_policy.max_snapshot_age_days}"
            )

    def _build_slot_gate_diagnostics(
        self,
        *,
        full_snapshot: pd.DataFrame,
        regime_snapshot: pd.DataFrame,
        strategies: dict[str, ProductionStrategy],
    ) -> dict[str, dict[str, object]]:
        return {
            str(slot): self._build_gate_diagnostic(
                full_snapshot=full_snapshot,
                regime_snapshot=regime_snapshot,
                indicators=strategy.indicators,
                sector=strategy.sector,
            )
            for slot, strategy in strategies.items()
        }

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
            if feature_name not in working.columns:
                working = working.iloc[0:0].copy()
            elif threshold_name.endswith("_min"):
                working = working.loc[working[feature_name].astype(float) >= float(threshold_value)].copy()
            elif threshold_name.endswith("_max"):
                working = working.loc[working[feature_name].astype(float) <= float(threshold_value)].copy()
            else:
                working = working.loc[working[feature_name].astype(float) == float(threshold_value)].copy()
            counts.append((threshold_name, len(working.index)))
            if first_zero_gate is None and len(working.index) == 0:
                first_zero_gate = threshold_name
        component_counter = {threshold_name: 0 for threshold_name in score_components}
        if not working.empty:
            passing_rows = 0
            for row in working.to_dict(orient="records"):
                passed, details, _ = evaluate_signal_gate(
                    {**hard_filters, **score_components, SIGNAL_SCORE_MIN_KEY: pass_score},
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

    def _build_slot_post_gate_dropoff(
        self,
        candidates: pd.DataFrame,
        *,
        strategies: dict[str, ProductionStrategy],
        min_opportunity_score: float,
    ) -> dict[str, dict[str, object]]:
        diagnostics: dict[str, dict[str, object]] = {}
        for slot, strategy in strategies.items():
            slot_frame = candidates[candidates["strategy_slot"].astype(str) == str(slot)].copy()
            if slot_frame.empty:
                diagnostics[str(slot)] = {
                    "gated_count": 0,
                    "cleared_count": 0,
                    "dropped_count": 0,
                    "avg_opportunity_score": None,
                    "top_cleared": [],
                    "top_dropped": [],
                    "drop_examples": [],
                }
                continue
            slot_frame["opportunity_score"] = pd.to_numeric(slot_frame["opportunity_score"], errors="coerce")
            cleared = slot_frame[slot_frame["opportunity_score"] >= float(min_opportunity_score)].copy()
            dropped = slot_frame[slot_frame["opportunity_score"] < float(min_opportunity_score)].copy()
            cleared = cleared.sort_values(["opportunity_score", "signal_score", "ticker"], ascending=[False, False, True])
            dropped = dropped.sort_values(["opportunity_score", "signal_score", "ticker"], ascending=[False, False, True])
            diagnostics[str(slot)] = {
                "gated_count": len(slot_frame.index),
                "cleared_count": len(cleared.index),
                "dropped_count": len(dropped.index),
                "avg_opportunity_score": float(slot_frame["opportunity_score"].dropna().mean()) if not slot_frame["opportunity_score"].dropna().empty else None,
                "top_cleared": cleared["ticker"].astype(str).head(3).tolist(),
                "top_dropped": dropped["ticker"].astype(str).head(3).tolist(),
                "drop_examples": [
                    {
                        "ticker": str(row["ticker"]),
                        "opportunity_score": float(row["opportunity_score"]),
                        "signal_score": float(row["signal_score"]),
                    }
                    for _, row in dropped.head(3).iterrows()
                ],
            }
        return diagnostics

    def _selection_opportunity_floor(self, *, scan_policy: ScanPolicy, shortlist_model_context) -> float:
        if shortlist_model_context is not None:
            return float(scan_policy.shortlist_model.min_opportunity_score)
        return float(scan_policy.min_opportunity_score)

    def _apply_portfolio_caps(self, candidates: pd.DataFrame, scan_policy: ScanPolicy) -> pd.DataFrame:
        ranked_candidates = candidates.copy()
        if "selection_score" not in ranked_candidates.columns:
            ranked_candidates["selection_score"] = ranked_candidates["signal_score"]
        if "recent_feedback_adjustment" not in ranked_candidates.columns:
            ranked_candidates["recent_feedback_adjustment"] = 0.0
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
        selected = ranked.loc[selected_indices].copy()
        return self._apply_recent_slot_swap_check(selected, ranked, scan_policy=scan_policy)

    def _load_shortlist_model_context(self, scan_policy: ScanPolicy):
        if not scan_policy.shortlist_model.enabled:
            return None
        try:
            return load_live_shortlist_model_context(
                self.db_manager,
                horizon_days=scan_policy.shortlist_model.horizon_days,
                top_n=scan_policy.shortlist_model.top_n,
                min_train_dates=scan_policy.shortlist_model.min_train_dates,
                test_window_dates=scan_policy.shortlist_model.test_window_dates,
                recent_dates=scan_policy.shortlist_model.recent_dates,
                refresh_if_stale=scan_policy.shortlist_model.refresh_if_stale,
                eligible_universe_mode=scan_policy.shortlist_model.production_eligible_universe_mode,
                model_scope=scan_policy.shortlist_model.production_model_scope,
                preferred_model_name=scan_policy.shortlist_model.production_model_name,
                xgboost_config=scan_policy.shortlist_model.production_xgboost_config,
            )
        except Exception as exc:
            self.logger.warning("Unable to load shortlist model context; falling back to heuristic selector: %s", exc)
            return None

    def _apply_shortlist_model_selection(self, candidates: pd.DataFrame, shortlist_model_context) -> pd.DataFrame:
        scored = candidates.copy()
        scored = scored.drop(columns=["predicted_alpha", "model_predicted_alpha", "model_rank"], errors="ignore")
        prediction_columns = ["ticker", "predicted_alpha", "model_rank"]
        if "model_reason_summary" in shortlist_model_context.live_predictions.columns:
            prediction_columns.append("model_reason_summary")
        if "model_comparison_summary" in shortlist_model_context.live_predictions.columns:
            prediction_columns.append("model_comparison_summary")
        predictions = shortlist_model_context.live_predictions[prediction_columns].copy().rename(
            columns={
                "predicted_alpha": "model_predicted_alpha",
            }
        )
        merged = scored.merge(predictions, on="ticker", how="left")
        merged["selection_score"] = pd.to_numeric(merged["model_predicted_alpha"], errors="coerce")
        fallback_selection = merged.get("selection_score")
        if fallback_selection is None:
            if "signal_score" in merged.columns:
                fallback_selection = merged["signal_score"]
            else:
                fallback_selection = pd.Series(0.0, index=merged.index)
        merged["selection_score"] = merged["selection_score"].where(
            merged["selection_score"].notna(),
            pd.to_numeric(fallback_selection, errors="coerce"),
        )
        merged["selection_source"] = merged["model_predicted_alpha"].apply(
            lambda value: "shortlist_model" if pd.notna(value) else "heuristic_fallback"
        )
        merged["model_generated_at"] = shortlist_model_context.generated_at
        merged["model_name"] = shortlist_model_context.champion_model
        merged["ranker_score"] = pd.NA
        merged["ranker_enabled"] = False
        merged["recent_drag_penalty"] = 0.0
        merged["recent_missed_winner_boost"] = 0.0
        merged["recent_feedback_adjustment"] = 0.0
        merged["slot_overlay_adjustment"] = 0.0
        merged["slot_overlay_components"] = [{} for _ in range(len(merged.index))]
        merged["recent_drag_picks"] = 0
        merged["recent_drag_mean_target"] = pd.NA
        merged["recent_missed_winner_count"] = 0
        merged["recent_missed_winner_mean_gap"] = pd.NA
        merged["ranker_top_positive_reasons"] = [tuple() for _ in range(len(merged.index))]
        merged["ranker_top_negative_reasons"] = [tuple() for _ in range(len(merged.index))]
        return merged

    def _attach_selection_metadata_to_persisted_candidates(
        self,
        persisted_candidates: pd.DataFrame,
        candidates: pd.DataFrame,
    ) -> pd.DataFrame:
        if persisted_candidates.empty or candidates.empty:
            return persisted_candidates.copy()
        return persisted_candidates.merge(
            candidates[
                [
                    "ticker",
                    "strategy_slot",
                    "strategy_sector",
                    "selection_source",
                    "model_predicted_alpha",
                    "model_rank",
                    "model_generated_at",
                    "model_name",
                    "model_reason_summary",
                    "model_comparison_summary",
                    "ranker_score",
                    "selection_score",
                    "ranker_enabled",
                    "recent_drag_penalty",
                    "recent_missed_winner_boost",
                    "recent_feedback_adjustment",
                    "slot_overlay_adjustment",
                    "slot_overlay_components",
                    "recent_drag_picks",
                    "recent_drag_mean_target",
                    "recent_missed_winner_count",
                    "recent_missed_winner_mean_gap",
                    "ranker_top_positive_reasons",
                    "ranker_top_negative_reasons",
                ]
            ],
            on=["ticker", "strategy_slot", "strategy_sector"],
            how="left",
        )

    def _build_persisted_scan_rows(
        self,
        persisted_candidates: pd.DataFrame,
        selected: pd.DataFrame,
    ) -> list[dict]:
        selection_keys = {
            (str(row.ticker), str(row.strategy_slot))
            for row in selected.itertuples(index=False)
        }
        selection_ranks = {
            (str(row.ticker), str(row.strategy_slot)): index + 1
            for index, row in enumerate(selected.itertuples(index=False))
        }
        persisted_rows: list[dict] = []
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
                    "selection_score": float(row.get("selection_score", row.get("signal_score", 0.0)))
                    if pd.notna(row.get("selection_score", row.get("signal_score", 0.0)))
                    else None,
                    "selection_source": row.get("selection_source"),
                    "model_predicted_alpha": float(row.get("model_predicted_alpha"))
                    if pd.notna(row.get("model_predicted_alpha"))
                    else None,
                    "model_rank": int(row.get("model_rank"))
                    if pd.notna(row.get("model_rank"))
                    else None,
                    "model_generated_at": row.get("model_generated_at"),
                    "model_name": row.get("model_name"),
                    "details": {
                        "why": self._candidate_signal_evidence(row),
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
                            "selection_score": float(row.get("selection_score", row.get("signal_score", 0.0))),
                            "selection_source": row.get("selection_source"),
                            "model_predicted_alpha": float(row.get("model_predicted_alpha", 0.0)) if pd.notna(row.get("model_predicted_alpha")) else None,
                            "model_rank": int(row.get("model_rank")) if pd.notna(row.get("model_rank")) else None,
                            "model_generated_at": row.get("model_generated_at"),
                            "model_name": row.get("model_name"),
                            "model_reason_summary": row.get("model_reason_summary"),
                            "model_comparison_summary": row.get("model_comparison_summary"),
                            "ranker_score": float(row.get("ranker_score", 0.0)) if pd.notna(row.get("ranker_score")) else None,
                            "ranker_enabled": bool(row.get("ranker_enabled", False)),
                            "recent_drag_penalty": float(row.get("recent_drag_penalty", 0.0)),
                            "recent_missed_winner_boost": float(row.get("recent_missed_winner_boost", 0.0)),
                            "recent_feedback_adjustment": float(row.get("recent_feedback_adjustment", 0.0)),
                            "slot_overlay_adjustment": float(row.get("slot_overlay_adjustment", 0.0)),
                        },
                        "slot_overlay_components": row.get("slot_overlay_components", {}),
                        "recent_selection_memory": {
                            "drag_picks": int(row.get("recent_drag_picks", 0) or 0),
                            "drag_mean_target": float(row.get("recent_drag_mean_target", 0.0)) if pd.notna(row.get("recent_drag_mean_target")) else None,
                            "missed_winner_count": int(row.get("recent_missed_winner_count", 0) or 0),
                            "missed_winner_mean_gap": float(row.get("recent_missed_winner_mean_gap", 0.0)) if pd.notna(row.get("recent_missed_winner_mean_gap")) else None,
                        },
                        "ranker_top_positive_reasons": list(row.get("ranker_top_positive_reasons", ()) or ()),
                        "ranker_top_negative_reasons": list(row.get("ranker_top_negative_reasons", ()) or ()),
                    },
                }
            )
        return persisted_rows

    def _build_email_html(
        self,
        candidates: pd.DataFrame,
        scan_policy: ScanPolicy,
        strategies: dict[str, ProductionStrategy],
        *,
        earnings_lookup: dict[str, dict[str, str]],
        analyst_contexts: dict[str, AnalystContext] | None = None,
        all_candidates: pd.DataFrame | None = None,
        open_trade_tickers: set[str] | None = None,
        open_trades: list | None = None,
    ) -> str:
        sections: list[str] = []
        analyst_contexts = analyst_contexts or {}
        ordered_slots = (
            candidates.groupby(["strategy_slot", "strategy_sector"], sort=False)["selection_score"]
            .max()
            .reset_index()
            .sort_values(["selection_score", "strategy_slot"], ascending=[False, True])
        )
        for slot_row in ordered_slots.itertuples(index=False):
            slot_candidates = candidates.loc[candidates["strategy_slot"] == slot_row.strategy_slot].copy()
            slot_candidates = slot_candidates.sort_values(
                ["selection_score", "opportunity_score", "signal_score", "md_volume_30d", "ticker"],
                ascending=[False, False, False, False, True],
            )
            strategy = strategies[str(slot_row.strategy_slot)]
            preview_rows = []
            for candidate in slot_candidates.itertuples(index=False):
                exit_plan = self._candidate_exit_plan(candidate, strategy)
                preview_rows.append((candidate, exit_plan))
            show_stop = any(item[1]["stop_display"] is not None for item in preview_rows)
            show_target = any(item[1]["target_display"] is not None for item in preview_rows)
            rows = []
            for candidate, exit_plan in preview_rows:
                chart_link = f"https://www.tradingview.com/chart/?symbol={candidate.ticker}"
                why = self._candidate_signal_evidence(candidate)
                earnings_context = earnings_lookup.get(str(candidate.ticker), {})
                next_earnings = earnings_context.get("next_earnings") or self._candidate_event_fallback(
                    candidate,
                    field_name="days_to_next_earnings",
                    past=False,
                )
                last_earnings = earnings_context.get("last_earnings") or self._candidate_event_fallback(
                    candidate,
                    field_name="days_since_last_earnings",
                    past=True,
                )
                earnings_note = self._candidate_earnings_note(candidate)
                earnings_status = self._candidate_earnings_status(candidate)
                analyst_note = self._candidate_analyst_note(
                    candidate,
                    analyst_contexts.get(str(candidate.ticker).upper()),
                )
                group_label = self._candidate_group_label(candidate)
                setup_label = self._candidate_setup_label(candidate)
                selector_note = self._candidate_selector_note(candidate)
                opportunity_breakdown = self._candidate_opportunity_breakdown(candidate, scan_policy)
                cells = [
                    f"<td>{candidate.ticker}</td>",
                    f"<td>{setup_label}</td>",
                    f"<td>{candidate.sector}</td>",
                    f"<td>{group_label}</td>",
                    f"<td>{candidate.regime_etf} Green</td>",
                    f"<td>{candidate.opportunity_score:.2f}</td>",
                    f"<td>{opportunity_breakdown}</td>",
                    f"<td>{self._candidate_signal_score_display(candidate)}</td>",
                    f"<td>{candidate.adj_close:.2f}</td>",
                ]
                if show_stop:
                    cells.append(f"<td>{self._format_level_with_distance(candidate, exit_plan, kind='stop')}</td>")
                if show_target:
                    cells.append(f"<td>{self._format_level_with_distance(candidate, exit_plan, kind='target')}</td>")
                cells.extend(
                    [
                        f"<td>{earnings_status}</td>",
                        f"<td>{next_earnings}</td>",
                        f"<td>{last_earnings}</td>",
                        f"<td>{earnings_note}</td>",
                        f"<td>{analyst_note}</td>",
                        f"<td>{why}</td>",
                        f"<td>{selector_note}</td>",
                    ]
                )
                cells.append(f"<td><a href=\"{chart_link}\">chart</a></td>")
                rows.append(
                    "<tr>"
                    f"{''.join(cells)}"
                    "</tr>"
                )
            header_cells = [
                "<th>Ticker</th>",
                "<th>Setup</th>",
                "<th>Sector</th>",
                "<th>Group</th>",
                "<th>Regime</th>",
                "<th>Opportunity Score</th>",
                "<th>Opportunity Breakdown</th>",
                "<th>Legacy Signal</th>",
                "<th>Adj Close</th>",
            ]
            if show_stop:
                header_cells.append("<th>Stop</th>")
            if show_target:
                header_cells.append("<th>Target</th>")
            header_cells.extend(
                [
                    "<th>Earnings Status</th>",
                    "<th>Next Earnings</th>",
                    "<th>Last Earnings</th>",
                    "<th>Earnings Note</th>",
                    "<th>Analyst Target</th>",
                    "<th>Signal Evidence</th>",
                    "<th>Selector Note</th>",
                ]
            )
            header_cells.append("<th>Chart</th>")
            sections.append(
                "<section>"
                f"<h2>Slot: {slot_row.strategy_slot}</h2>"
                f"<p>Strategy sector scope: {slot_row.strategy_sector} | Candidates: {len(slot_candidates.index)}</p>"
                "<table border='1' cellpadding='6' cellspacing='0'>"
                f"<tr>{''.join(header_cells)}</tr>"
                f"{''.join(rows)}"
                "</table>"
                "</section>"
            )
        return (
            "<html><body>"
            "<h1>Evening Brief</h1>"
            "<p>Portfolio view: strongest current bets, existing exposure, and best new targets.</p>"
            f"{self._build_target_dashboard_html(all_candidates, open_trades=open_trades, analyst_contexts=analyst_contexts)}"
            f"{self._build_portfolio_strength_coverage_html(all_candidates, open_trade_tickers=open_trade_tickers)}"
            f"<p>Selector caps used for detailed shortlist: total={scan_policy.max_candidates_total}, per_slot={scan_policy.max_candidates_per_slot}, per_sector={scan_policy.max_candidates_per_sector} | min_opportunity_score={scan_policy.min_opportunity_score:.2f} | shortlist_model_min_opportunity_score={scan_policy.shortlist_model.min_opportunity_score:.2f}</p>"
            f"{''.join(sections)}"
            "</body></html>"
        )

    def _build_target_dashboard_html(
        self,
        candidates: pd.DataFrame | None,
        *,
        open_trades: list | None,
        analyst_contexts: dict[str, AnalystContext] | None = None,
        top_n: int = 10,
        new_target_count: int = 3,
    ) -> str:
        if candidates is None or candidates.empty:
            return ""
        working = candidates.copy()
        if "already_owned" not in working.columns:
            working["already_owned"] = False
        if "selection_score" not in working.columns:
            working["selection_score"] = working.get("signal_score", 0.0)
        if "model_predicted_alpha" not in working.columns:
            working["model_predicted_alpha"] = working["selection_score"]
        if "model_reason_summary" not in working.columns:
            working["model_reason_summary"] = pd.NA
        if "details_json" in working.columns:
            missing_reason = working["model_reason_summary"].isna()
            if missing_reason.any():
                working.loc[missing_reason, "model_reason_summary"] = working.loc[missing_reason, "details_json"].apply(
                    self._model_reason_from_details_json
                )
        working["pre_penalty_opportunity_score"] = (
            pd.to_numeric(working["opportunity_score"], errors="coerce").fillna(float("-inf"))
            + pd.to_numeric(working.get("overlap_penalty", 0.0), errors="coerce").fillna(0.0)
        )
        working["model_predicted_alpha"] = pd.to_numeric(working["model_predicted_alpha"], errors="coerce")
        targetable = working[
            (working["pre_penalty_opportunity_score"].astype(float) >= 0.40)
            & (working["model_predicted_alpha"].fillna(float("-inf")) > 0.0)
        ].copy()
        if targetable.empty:
            return (
                "<section>"
                "<h2>Current Target Dashboard</h2>"
                "<p>No candidates currently clear the 0.40 pre-penalty opportunity floor with positive model alpha.</p>"
                "</section>"
            )

        trade_lookup = self._open_trade_lookup(open_trades or [])
        held_tickers = set(trade_lookup)
        targetable["dashboard_held"] = (
            targetable["already_owned"].astype(bool)
            | targetable["ticker"].astype(str).isin(held_tickers)
        )
        targetable = targetable.sort_values(
            ["model_predicted_alpha", "pre_penalty_opportunity_score", "ticker"],
            ascending=[False, False, True],
        ).reset_index(drop=True)
        held = targetable[targetable["dashboard_held"].astype(bool)].copy()
        unheld = targetable[~targetable["dashboard_held"].astype(bool)].copy()
        top_bets = targetable.head(int(top_n))
        new_targets = unheld.head(int(new_target_count))
        held_targets = held.head(int(top_n))
        held_missing = sorted(set(trade_lookup) - set(working["ticker"].astype(str)))

        return (
            "<section>"
            "<h2>Current Target Dashboard</h2>"
            f"<p>Target filter: pre-penalty opportunity >= 0.40 and model alpha > 0. Top bets already held: {int(top_bets['dashboard_held'].astype(bool).sum())}/{len(top_bets.index)}. "
            f"Best new targets: {self._ticker_list(new_targets)}.</p>"
            f"{self._target_table_html('Top Current Bets', top_bets, trade_lookup=trade_lookup, analyst_contexts=analyst_contexts or {})}"
            f"{self._target_table_html('Already Held Targets', held_targets, trade_lookup=trade_lookup, analyst_contexts=analyst_contexts or {})}"
            f"{self._target_table_html('Best New Targets', new_targets, trade_lookup=trade_lookup, analyst_contexts=analyst_contexts or {})}"
            f"{'<p>Open holdings outside current target set: ' + ', '.join(held_missing) + '</p>' if held_missing else ''}"
            "</section>"
        )

    def _target_table_html(
        self,
        title: str,
        frame: pd.DataFrame,
        *,
        trade_lookup: dict[str, dict],
        analyst_contexts: dict[str, AnalystContext],
    ) -> str:
        if frame.empty:
            return f"<h3>{title}</h3><p>None.</p>"
        rows = []
        for index, row in enumerate(frame.itertuples(index=False), start=1):
            ticker = str(row.ticker)
            trade = trade_lookup.get(ticker, {})
            held = bool(getattr(row, "dashboard_held", False)) or ticker in trade_lookup
            selected = "yes" if int(getattr(row, "selected", 0) or 0) == 1 else "no"
            entry_price = trade.get("entry_price")
            current_price = getattr(row, "adj_close", None)
            pnl_text = "n/a"
            if self._is_finite(entry_price) and self._is_finite(current_price) and float(entry_price) != 0:
                pnl_text = f"{((float(current_price) / float(entry_price)) - 1.0) * 100.0:+.1f}%"
            held_text = "yes" if held else "no"
            if held and trade.get("days_held") is not None:
                held_text = f"yes ({trade['days_held']}bd)"
            analyst_note = self._candidate_analyst_note(
                row,
                analyst_contexts.get(ticker.upper()),
            )
            rows.append(
                "<tr>"
                f"<td>{index}</td>"
                f"<td>{ticker}</td>"
                f"<td>{held_text}</td>"
                f"<td>{selected}</td>"
                f"<td>{getattr(row, 'sector', '')}</td>"
                f"<td>{float(row.pre_penalty_opportunity_score):.2f}</td>"
                f"<td>{float(getattr(row, 'opportunity_score')):.2f}</td>"
                f"<td>{self._format_pct_cell(getattr(row, 'model_predicted_alpha', None))}</td>"
                f"<td>{self._format_price_cell(current_price)}</td>"
                f"<td>{self._format_price_cell(entry_price)}</td>"
                f"<td>{pnl_text}</td>"
                f"<td>{analyst_note}</td>"
                f"<td>{self._target_reason_text(row)}</td>"
                "</tr>"
            )
        return (
            f"<h3>{title}</h3>"
            "<table border='1' cellpadding='6' cellspacing='0'>"
            "<tr><th>Rank</th><th>Ticker</th><th>Held</th><th>Selected</th><th>Sector</th><th>Pre-Opp</th><th>Post-Opp</th><th>Model Alpha</th><th>Price</th><th>Basis</th><th>PnL</th><th>Analyst Target</th><th>Why</th></tr>"
            f"{''.join(rows)}"
            "</table>"
        )

    def _ticker_list(self, frame: pd.DataFrame) -> str:
        if frame.empty:
            return "none"
        return ", ".join(str(ticker) for ticker in frame["ticker"].head(5).tolist())

    def _format_pct_cell(self, value) -> str:
        if not self._is_finite(value):
            return "n/a"
        return f"{float(value) * 100.0:+.2f}%"

    def _format_price_cell(self, value) -> str:
        if not self._is_finite(value):
            return "n/a"
        return f"{float(value):.2f}"

    def _optional_text_cell(self, value) -> str:
        if value is None or pd.isna(value) or value == "":
            return "n/a"
        return str(value)

    def _target_reason_text(self, row) -> str:
        reason = getattr(row, "model_reason_summary", None)
        if reason is not None and not pd.isna(reason) and reason != "":
            return str(reason)
        return self._optional_text_cell(self._model_reason_from_details_json(getattr(row, "details_json", None)))

    def _candidate_analyst_note(self, candidate, context: AnalystContext | None) -> str:
        if context is None:
            return "n/a"
        current_price = getattr(candidate, "adj_close", None)
        target = context.target_mean if context.target_mean is not None else context.target_median
        parts = []
        if self._is_finite(target):
            target_label = "mean" if context.target_mean is not None else "median"
            if self._is_finite(current_price) and float(current_price) > 0:
                upside = (float(target) / float(current_price)) - 1.0
                parts.append(f"{target_label} {float(target):.2f} ({upside * 100.0:+.1f}%)")
            else:
                parts.append(f"{target_label} {float(target):.2f}")
        range_values = []
        if self._is_finite(context.target_low):
            range_values.append(f"{float(context.target_low):.2f}")
        if self._is_finite(context.target_high):
            range_values.append(f"{float(context.target_high):.2f}")
        if len(range_values) == 2:
            parts.append(f"range {range_values[0]}-{range_values[1]}")
        if context.analyst_count is not None:
            parts.append(f"n={context.analyst_count}")
        if context.recommendation:
            parts.append(context.recommendation)
        return "; ".join(parts) if parts else "n/a"

    def _model_reason_from_details_json(self, value) -> str | None:
        if value in (None, "") or pd.isna(value):
            return None
        try:
            details = json.loads(str(value))
        except Exception:
            return None
        ranking_components = details.get("ranking_components", {}) if isinstance(details, dict) else {}
        reason = ranking_components.get("model_reason_summary")
        return str(reason) if reason not in (None, "") else None

    def _build_portfolio_strength_coverage_html(
        self,
        candidates: pd.DataFrame | None,
        *,
        open_trade_tickers: set[str] | None,
        top_n: int = 10,
    ) -> str:
        if candidates is None or candidates.empty:
            return ""
        working = candidates.copy()
        if "already_owned" not in working.columns:
            working["already_owned"] = False
        if "selection_score" not in working.columns:
            working["selection_score"] = working.get("signal_score", 0.0)
        working["pre_penalty_opportunity_score"] = (
            pd.to_numeric(working["opportunity_score"], errors="coerce").fillna(float("-inf"))
            + pd.to_numeric(working.get("overlap_penalty", 0.0), errors="coerce").fillna(0.0)
        )
        ranked = working.sort_values(
            ["pre_penalty_opportunity_score", "selection_score", "signal_score", "ticker"],
            ascending=[False, False, False, True],
        ).reset_index(drop=True)
        if ranked.empty:
            return ""
        top_six = ranked.head(6)
        top_slice = ranked.head(int(top_n))
        held_top_six = int(top_six["already_owned"].astype(bool).sum())
        held_top_n = int(top_slice["already_owned"].astype(bool).sum())
        owned_candidates = ranked[ranked["already_owned"].astype(bool)].copy()
        strongest_held = str(owned_candidates.iloc[0]["ticker"]) if not owned_candidates.empty else "none"
        strongest_unheld_frame = ranked[~ranked["already_owned"].astype(bool)].copy()
        strongest_unheld = str(strongest_unheld_frame.iloc[0]["ticker"]) if not strongest_unheld_frame.empty else "none"
        candidate_tickers = set(working["ticker"].astype(str))
        missing_open = sorted((open_trade_tickers or set()) - candidate_tickers)
        rows = []
        for index, row in enumerate(top_slice.itertuples(index=False), start=1):
            status = "held" if bool(row.already_owned) else "not held"
            selected = "selected" if int(getattr(row, "selected", 0) or 0) == 1 else "not selected"
            selection_score = getattr(row, "selection_score", np.nan)
            selection_display = f"{float(selection_score):.4f}" if pd.notna(selection_score) else "n/a"
            rows.append(
                "<tr>"
                f"<td>{index}</td>"
                f"<td>{row.ticker}</td>"
                f"<td>{status}</td>"
                f"<td>{selected}</td>"
                f"<td>{float(row.pre_penalty_opportunity_score):.2f}</td>"
                f"<td>{float(row.opportunity_score):.2f}</td>"
                f"<td>{selection_display}</td>"
                f"<td>{row.sector}</td>"
                "</tr>"
            )
        missing_line = (
            f"<p>Open holdings not in candidate set: {', '.join(missing_open)}</p>"
            if missing_open
            else ""
        )
        return (
            "<section>"
            "<h2>Portfolio Strength Coverage</h2>"
            f"<p>Top 6 already held: {held_top_six}/6 | Top {min(int(top_n), len(ranked.index))} already held: {held_top_n}/{min(int(top_n), len(ranked.index))} | "
            f"Strongest held: {strongest_held} | Strongest unheld: {strongest_unheld}</p>"
            f"{missing_line}"
            "<table border='1' cellpadding='6' cellspacing='0'>"
            "<tr><th>Rank</th><th>Ticker</th><th>Status</th><th>Selected</th><th>Pre-Penalty Opportunity</th><th>Post-Penalty Opportunity</th><th>Selection Score</th><th>Sector</th></tr>"
            f"{''.join(rows)}"
            "</table>"
            "</section>"
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
        scored["selection_score"] = scored["signal_score"].astype(float)
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
        validation_report = ranker.evaluate(
            labeled,
            scan_policy=scan_policy,
            top_n=scan_policy.max_candidates_total,
        )
        validation_reason = self._learned_ranker_validation_reason(validation_report, scan_policy)
        if validation_reason is not None:
            self.logger.info(
                "Learned ranker disabled in scan after validation gate: %s",
                validation_reason,
            )
            return scored, LearnedRankerStatus(
                enabled=False,
                train_rows=train_rows,
                train_dates=train_dates,
                reason=validation_reason,
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
                    selection_score=candidates["signal_score"].astype(float),
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
            scored["signal_score"].astype(float)
            + (ranker_values * scored["learned_ranker_weight"].astype(float))
        )
        scored["ranker_enabled"] = True
        self.logger.info(
            "Learned ranker applied in scan: train_rows=%s train_dates=%s base_weight=%.2f slot_weights=%s q1_q5_spread=%.6f ic_mean=%.6f ic_dates=%s validation_blocks=%s",
            train_rows,
            train_dates,
            float(scan_policy.learned_ranker_weight),
            scan_policy.learned_ranker_slot_weights,
            float(validation_report.q1_q5_spread),
            float(validation_report.daily_ic_mean),
            int(validation_report.daily_ic_dates),
            int(validation_report.validation_blocks),
        )
        return scored, LearnedRankerStatus(
            enabled=True,
            train_rows=train_rows,
            train_dates=train_dates,
            reason=None,
        )

    def _apply_recent_selection_memory(
        self,
        candidates: pd.DataFrame,
        *,
        scan_policy: ScanPolicy,
        historical_scan_candidates: pd.DataFrame,
    ) -> pd.DataFrame:
        scored = candidates.copy()
        scored["recent_drag_penalty"] = 0.0
        scored["recent_missed_winner_boost"] = 0.0
        scored["recent_feedback_adjustment"] = 0.0
        scored["recent_drag_picks"] = 0
        scored["recent_drag_mean_target"] = math.nan
        scored["recent_missed_winner_count"] = 0
        scored["recent_missed_winner_mean_gap"] = math.nan
        if not scan_policy.recent_selection_memory_enabled or historical_scan_candidates.empty:
            return scored
        feedback = self._build_recent_selection_feedback(
            historical_scan_candidates,
            scan_policy=scan_policy,
        )
        if feedback.empty:
            return scored
        merged = scored.merge(
            feedback,
            on=["ticker", "strategy_slot"],
            how="left",
            suffixes=("", "_feedback"),
        )
        for column, default in (
            ("recent_drag_penalty", 0.0),
            ("recent_missed_winner_boost", 0.0),
            ("recent_feedback_adjustment", 0.0),
            ("recent_drag_picks", 0),
            ("recent_missed_winner_count", 0),
        ):
            feedback_column = f"{column}_feedback"
            if feedback_column in merged.columns:
                merged[column] = merged[feedback_column].fillna(default)
                merged = merged.drop(columns=[feedback_column])
            else:
                merged[column] = merged[column].fillna(default)
        for column in ("recent_drag_mean_target", "recent_missed_winner_mean_gap"):
            feedback_column = f"{column}_feedback"
            if feedback_column in merged.columns:
                merged[column] = merged[feedback_column]
                merged = merged.drop(columns=[feedback_column])
        merged["selection_score"] = (
            pd.to_numeric(merged["selection_score"], errors="coerce").fillna(0.0)
            + pd.to_numeric(merged["recent_feedback_adjustment"], errors="coerce").fillna(0.0)
        )
        return merged

    def _apply_slot_selection_overlays(
        self,
        candidates: pd.DataFrame,
        *,
        scan_policy: ScanPolicy,
    ) -> pd.DataFrame:
        scored = candidates.copy()
        scored["slot_overlay_adjustment"] = 0.0
        scored["slot_overlay_components"] = [dict() for _ in range(len(scored.index))]
        if not scan_policy.slot_selection_overlay_enabled:
            return scored
        adjustments: list[float] = []
        components_list: list[dict[str, float]] = []
        for row in scored.to_dict(orient="records"):
            adjustment, components = self._slot_selection_overlay_adjustment(row, scan_policy=scan_policy)
            adjustments.append(float(adjustment))
            components_list.append(components)
        scored["slot_overlay_adjustment"] = adjustments
        scored["slot_overlay_components"] = components_list
        scored["selection_score"] = (
            pd.to_numeric(scored["selection_score"], errors="coerce").fillna(0.0)
            + pd.to_numeric(scored["slot_overlay_adjustment"], errors="coerce").fillna(0.0)
        )
        return scored

    def _slot_selection_overlay_adjustment(
        self,
        row: dict,
        *,
        scan_policy: ScanPolicy,
    ) -> tuple[float, dict[str, float]]:
        slot = str(row.get("strategy_slot") or "")
        weights = scan_policy.slot_selection_overlay_weights.get(slot, {})
        if not weights:
            return 0.0, {}
        components: dict[str, float] = {}
        total = 0.0
        for feature_name, weight in weights.items():
            normalized = self._slot_overlay_feature_value(row, feature_name)
            if not math.isfinite(normalized):
                continue
            contribution = float(weight) * float(normalized)
            components[feature_name] = contribution
            total += contribution
        return float(total), components

    def _slot_overlay_feature_value(self, row: dict, feature_name: str) -> float:
        raw_value = row.get(feature_name)
        value = self._numeric(raw_value)
        if feature_name in {"freshness_score", "setup_quality_score", "expected_alpha_score", "sector_pct_above_50", "sector_pct_above_200"}:
            return self._clamp(value)
        if feature_name == "signal_score":
            return self._clamp(value / 100.0)
        if feature_name == "distance_above_20d_high":
            return self._clamp(value / 0.10)
        if feature_name in {"roc_63", "sma_200_dist"}:
            return self._clamp(value / 0.50)
        return self._clamp(value)

    def _build_recent_selection_feedback(
        self,
        historical_scan_candidates: pd.DataFrame,
        *,
        scan_policy: ScanPolicy,
    ) -> pd.DataFrame:
        if "alpha_vs_sector_10d" not in historical_scan_candidates.columns:
            return pd.DataFrame(columns=["ticker", "strategy_slot"])
        matured = historical_scan_candidates.dropna(subset=["alpha_vs_sector_10d"]).copy()
        if matured.empty:
            return pd.DataFrame(columns=["ticker", "strategy_slot"])
        recent_dates = (
            matured.loc[matured["selected"].astype(int) == 1, "scan_date"]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .sort_values(ascending=False)
            .head(scan_policy.recent_selection_memory_scan_dates)
            .tolist()
        )
        if not recent_dates:
            return pd.DataFrame(columns=["ticker", "strategy_slot"])
        recent = matured[matured["scan_date"].astype(str).isin(recent_dates)].copy()
        if recent.empty:
            return pd.DataFrame(columns=["ticker", "strategy_slot"])

        selected = recent[recent["selected"].astype(int) == 1].copy()
        drags = selected.groupby(["ticker", "strategy_slot"], dropna=False).agg(
            recent_drag_picks=("alpha_vs_sector_10d", "count"),
            recent_drag_mean_target=("alpha_vs_sector_10d", "mean"),
        ).reset_index()
        drags["recent_drag_penalty"] = 0.0
        drag_mask = (
            drags["recent_drag_picks"].astype(int) >= int(scan_policy.recent_drag_min_picks)
        ) & (drags["recent_drag_mean_target"].astype(float) < 0.0)
        drags.loc[drag_mask, "recent_drag_penalty"] = (
            (-drags.loc[drag_mask, "recent_drag_mean_target"].astype(float))
            * float(scan_policy.recent_drag_penalty_scale)
        ).clip(upper=float(scan_policy.recent_drag_penalty_cap))

        swap_rows: list[dict[str, object]] = []
        for (scan_date, strategy_slot), subset in recent.groupby(["scan_date", "strategy_slot"], sort=True):
            selected_subset = subset[subset["selected"].astype(int) == 1].dropna(subset=["alpha_vs_sector_10d"]).copy()
            excluded_subset = subset[subset["selected"].astype(int) == 0].dropna(subset=["alpha_vs_sector_10d"]).copy()
            if selected_subset.empty or excluded_subset.empty:
                continue
            worst_selected = selected_subset.sort_values(
                ["alpha_vs_sector_10d", "opportunity_score", "ticker"],
                ascending=[True, False, True],
            ).iloc[0]
            best_excluded = excluded_subset.sort_values(
                ["alpha_vs_sector_10d", "opportunity_score", "ticker"],
                ascending=[False, False, True],
            ).iloc[0]
            gap = float(best_excluded["alpha_vs_sector_10d"]) - float(worst_selected["alpha_vs_sector_10d"])
            if gap < float(scan_policy.recent_missed_winner_min_gap):
                continue
            swap_rows.append(
                {
                    "ticker": str(best_excluded["ticker"]),
                    "strategy_slot": str(strategy_slot),
                    "gap": gap,
                }
            )
        boosts = pd.DataFrame(swap_rows)
        if boosts.empty:
            merged = drags.copy()
            merged["recent_missed_winner_count"] = 0
            merged["recent_missed_winner_mean_gap"] = math.nan
            merged["recent_missed_winner_boost"] = 0.0
        else:
            boosts = boosts.groupby(["ticker", "strategy_slot"], dropna=False).agg(
                recent_missed_winner_count=("gap", "count"),
                recent_missed_winner_mean_gap=("gap", "mean"),
            ).reset_index()
            boosts["recent_missed_winner_boost"] = 0.0
            boost_mask = boosts["recent_missed_winner_count"].astype(int) >= int(scan_policy.recent_missed_winner_min_count)
            boosts.loc[boost_mask, "recent_missed_winner_boost"] = (
                boosts.loc[boost_mask, "recent_missed_winner_mean_gap"].astype(float)
                * float(scan_policy.recent_missed_winner_boost_scale)
            ).clip(upper=float(scan_policy.recent_missed_winner_boost_cap))
            merged = drags.merge(boosts, on=["ticker", "strategy_slot"], how="outer")
        for column, default in (
            ("recent_drag_picks", 0),
            ("recent_drag_mean_target", math.nan),
            ("recent_drag_penalty", 0.0),
            ("recent_missed_winner_count", 0),
            ("recent_missed_winner_mean_gap", math.nan),
            ("recent_missed_winner_boost", 0.0),
        ):
            if column not in merged.columns:
                merged[column] = default
            if default == 0:
                merged[column] = merged[column].fillna(default)
            else:
                merged[column] = merged[column].fillna(default)
        merged["recent_feedback_adjustment"] = (
            pd.to_numeric(merged["recent_missed_winner_boost"], errors="coerce").fillna(0.0)
            - pd.to_numeric(merged["recent_drag_penalty"], errors="coerce").fillna(0.0)
        )
        return merged[
            [
                "ticker",
                "strategy_slot",
                "recent_drag_picks",
                "recent_drag_mean_target",
                "recent_drag_penalty",
                "recent_missed_winner_count",
                "recent_missed_winner_mean_gap",
                "recent_missed_winner_boost",
                "recent_feedback_adjustment",
            ]
        ].copy()

    def _apply_recent_slot_swap_check(
        self,
        selected: pd.DataFrame,
        ranked: pd.DataFrame,
        *,
        scan_policy: ScanPolicy,
    ) -> pd.DataFrame:
        if selected.empty or not scan_policy.recent_selection_memory_enabled:
            return selected
        working = selected.copy()
        selected_keys = {
            (str(row["ticker"]), str(row["strategy_slot"]))
            for row in working.to_dict(orient="records")
        }
        for strategy_slot in sorted(set(working["strategy_slot"].astype(str))):
            slot_selected = working[working["strategy_slot"].astype(str) == strategy_slot].copy()
            slot_excluded = ranked[
                (ranked["strategy_slot"].astype(str) == strategy_slot)
                & (~ranked["already_owned"].astype(bool))
                & (~ranked.apply(lambda row: (str(row["ticker"]), str(row["strategy_slot"])) in selected_keys, axis=1))
                & (ranked["opportunity_score"].astype(float) >= float(scan_policy.min_opportunity_score))
            ].copy()
            if slot_selected.empty or slot_excluded.empty:
                continue
            worst_selected = slot_selected.sort_values(
                ["recent_feedback_adjustment", "selection_score", "opportunity_score", "signal_score", "ticker"],
                ascending=[True, True, True, True, True],
            ).iloc[0]
            best_excluded = slot_excluded.sort_values(
                ["recent_feedback_adjustment", "selection_score", "opportunity_score", "signal_score", "ticker"],
                ascending=[False, False, False, False, True],
            ).iloc[0]
            feedback_gap = float(best_excluded.get("recent_feedback_adjustment", 0.0)) - float(worst_selected.get("recent_feedback_adjustment", 0.0))
            opportunity_gap = float(worst_selected["opportunity_score"]) - float(best_excluded["opportunity_score"])
            if feedback_gap < float(scan_policy.recent_swap_feedback_gap):
                continue
            if opportunity_gap > float(scan_policy.recent_swap_max_opportunity_gap):
                continue
            worst_key = (str(worst_selected["ticker"]), str(worst_selected["strategy_slot"]))
            best_key = (str(best_excluded["ticker"]), str(best_excluded["strategy_slot"]))
            selected_keys.discard(worst_key)
            selected_keys.add(best_key)
            working = ranked[
                ranked.apply(lambda row: (str(row["ticker"]), str(row["strategy_slot"])) in selected_keys, axis=1)
            ].copy()
        return working

    def _learned_ranker_validation_reason(
        self,
        report: RankerValidationReport,
        scan_policy: ScanPolicy,
    ) -> str | None:
        if not scan_policy.learned_ranker_validation_enabled:
            return None
        if not report.available:
            return (
                "Validation gate unavailable: "
                f"validation_method={report.validation_method} "
                f"validation_blocks={report.validation_blocks} "
                f"validation_dates={report.validation_dates}"
            )
        if int(report.validation_blocks) < int(scan_policy.learned_ranker_min_validation_blocks):
            return (
                "Validation gate failed: "
                f"validation_blocks={report.validation_blocks} "
                f"min_validation_blocks={scan_policy.learned_ranker_min_validation_blocks}"
            )
        if int(report.daily_ic_dates) < int(scan_policy.learned_ranker_min_ic_dates):
            return (
                "Validation gate failed: "
                f"ic_dates={report.daily_ic_dates} "
                f"min_ic_dates={scan_policy.learned_ranker_min_ic_dates}"
            )
        if not math.isfinite(report.q1_q5_spread):
            return "Validation gate failed: q1_q5_spread is not finite."
        if float(report.q1_q5_spread) < float(scan_policy.learned_ranker_min_q1_q5_spread):
            return (
                "Validation gate failed: "
                f"q1_q5_spread={report.q1_q5_spread:.6f} "
                f"min_q1_q5_spread={scan_policy.learned_ranker_min_q1_q5_spread:.6f}"
            )
        if not math.isfinite(report.daily_ic_mean):
            return "Validation gate failed: ic_mean is not finite."
        if float(report.daily_ic_mean) < float(scan_policy.learned_ranker_min_ic_mean):
            return (
                "Validation gate failed: "
                f"ic_mean={report.daily_ic_mean:.6f} "
                f"min_ic_mean={scan_policy.learned_ranker_min_ic_mean:.6f}"
            )
        return None

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

    def _candidate_exit_plan(self, candidate, strategy: ProductionStrategy) -> dict[str, str | None]:
        entry_atr = float(candidate.atr_14) if pd.notna(getattr(candidate, "atr_14", pd.NA)) else None
        stop_display: str | None = None
        target_display: str | None = None
        try:
            stop_price = trailing_stop_price(
                max_price_seen=float(candidate.adj_close),
                entry_atr=entry_atr,
                exit_rules=strategy.exit_rules,
            )
            stop_display = f"{stop_price:.2f}"
        except Exception:
            stop_display = None
        try:
            target_price = profit_target_price(
                entry_price=float(candidate.adj_close),
                entry_atr=entry_atr,
                exit_rules=strategy.exit_rules,
            )
            target_display = f"{target_price:.2f}"
        except Exception:
            target_display = None
        return {"stop_display": stop_display, "target_display": target_display}

    def _format_level_with_distance(
        self,
        candidate,
        exit_plan: dict[str, str | None],
        *,
        kind: str,
    ) -> str:
        display = exit_plan.get(f"{kind}_display")
        if display is None:
            return "n/a"
        price = float(candidate.adj_close)
        level = float(display)
        if price <= 0:
            return display
        pct = ((level / price) - 1.0) * 100.0
        sign = "+" if pct >= 0 else ""
        return f"{display} ({sign}{pct:.1f}%)"

    def _build_earnings_lookup(
        self,
        *,
        earnings_calendar: pd.DataFrame | None,
        as_of: date,
    ) -> dict[str, dict[str, str]]:
        lookup: dict[str, dict[str, str]] = {}
        if earnings_calendar is None or earnings_calendar.empty:
            return lookup
        as_of_ts = pd.Timestamp(as_of).normalize()
        for ticker, group in earnings_calendar.groupby("ticker", sort=False):
            ordered = pd.to_datetime(group["earnings_date"]).dt.normalize().sort_values()
            if ordered.empty:
                continue
            prior = ordered[ordered <= as_of_ts]
            upcoming = ordered[ordered >= as_of_ts]
            payload: dict[str, str] = {}
            if not upcoming.empty:
                next_date = upcoming.iloc[0]
                payload["next_earnings"] = self._format_event_date(next_date, as_of_ts)
            if not prior.empty:
                last_date = prior.iloc[-1]
                payload["last_earnings"] = self._format_event_date(last_date, as_of_ts, past=True)
            if payload:
                lookup[str(ticker)] = payload
        return lookup

    def _format_event_date(self, event_date: pd.Timestamp, as_of_date: pd.Timestamp, *, past: bool = False) -> str:
        event_date = pd.Timestamp(event_date).normalize()
        as_of_date = pd.Timestamp(as_of_date).normalize()
        if past:
            business_days = int(np.busday_count(event_date.date(), as_of_date.date())) if event_date < as_of_date else 0
            return f"{event_date.date()} ({business_days}bd ago)"
        business_days = int(np.busday_count(as_of_date.date(), event_date.date())) if event_date > as_of_date else 0
        if event_date == as_of_date:
            return f"{event_date.date()} (today)"
        return f"{event_date.date()} ({business_days}bd)"

    def _candidate_earnings_note(self, candidate) -> str:
        gap_pct = getattr(candidate, "last_earnings_gap_pct", pd.NA)
        hold_pct = getattr(candidate, "close_vs_last_earnings_close", pd.NA)
        volume_ratio = getattr(candidate, "last_earnings_volume_ratio_20", pd.NA)
        open_vs_high = getattr(candidate, "last_earnings_open_vs_20d_high", pd.NA)
        if (
            not self._is_finite(gap_pct)
            and not self._is_finite(hold_pct)
            and not self._is_finite(volume_ratio)
            and not self._is_finite(open_vs_high)
        ):
            days_since_last = getattr(candidate, "days_since_last_earnings", pd.NA)
            if self._is_finite(days_since_last):
                return f"recent earnings (~{int(float(days_since_last))}bd ago), reaction data unavailable"
            return "n/a"
        parts: list[str] = []
        if self._is_finite(gap_pct):
            gap_value = float(gap_pct) * 100.0
            parts.append(f"gap {'+' if gap_value >= 0 else ''}{gap_value:.1f}%")
        if self._is_finite(hold_pct):
            hold_value = float(hold_pct) * 100.0
            parts.append(f"hold {'+' if hold_value >= 0 else ''}{hold_value:.1f}%")
        if self._is_finite(volume_ratio):
            parts.append(f"vol {float(volume_ratio):.1f}x")
        if self._is_finite(open_vs_high):
            open_value = float(open_vs_high) * 100.0
            parts.append(f"open {'+' if open_value >= 0 else ''}{open_value:.1f}% vs 20d high")
        return ", ".join(parts) if parts else "n/a"

    def _candidate_earnings_status(self, candidate) -> str:
        days_to_next = getattr(candidate, "days_to_next_earnings", pd.NA)
        days_since_last = getattr(candidate, "days_since_last_earnings", pd.NA)
        if self._is_finite(days_to_next) and float(days_to_next) <= 5.0:
            return f"before earnings ({int(float(days_to_next))}bd)"
        if self._is_finite(days_since_last) and float(days_since_last) <= 5.0:
            return f"after earnings ({int(float(days_since_last))}bd)"
        return "clear of earnings"

    def _candidate_event_fallback(self, candidate, *, field_name: str, past: bool) -> str:
        value = getattr(candidate, field_name, pd.NA)
        if not self._is_finite(value):
            return "n/a"
        business_days = int(float(value))
        if past:
            return f"~{business_days}bd ago"
        return f"~{business_days}bd"

    def _candidate_group_label(self, candidate) -> str:
        sub_industry = getattr(candidate, "sub_industry", None)
        benchmark = getattr(candidate, "subindustry_benchmark", None)
        if sub_industry and benchmark:
            return f"{sub_industry} vs {benchmark}"
        if sub_industry:
            return str(sub_industry)
        if benchmark:
            return f"vs {benchmark}"
        return "n/a"

    def _candidate_setup_label(self, candidate) -> str:
        days_since_last = getattr(candidate, "days_since_last_earnings", pd.NA)
        earnings_gap = getattr(candidate, "last_earnings_gap_pct", pd.NA)
        breakout = getattr(candidate, "breakout_above_20d_high", pd.NA)
        distance_above = getattr(candidate, "distance_above_20d_high", pd.NA)
        if (
            self._is_finite(days_since_last)
            and self._is_finite(earnings_gap)
            and float(days_since_last) <= 5.0
            and float(earnings_gap) > 0.0
        ):
            return "post-ER strength"
        if self._is_finite(breakout) and float(breakout) >= 1.0:
            if self._is_finite(distance_above) and float(distance_above) <= 0.01:
                return "fresh breakout"
            return "breakout"
        if self._is_finite(getattr(candidate, "sma_50_dist", pd.NA)) and self._is_finite(getattr(candidate, "roc_63", pd.NA)):
            if float(getattr(candidate, "sma_50_dist")) > 0.0 and float(getattr(candidate, "roc_63")) > 0.0:
                return "trend continuation"
        return "setup"

    def _candidate_model_note(self, candidate) -> str:
        reasons = getattr(candidate, "ranker_top_positive_reasons", ()) or ()
        if not reasons:
            return "n/a"
        trimmed = [str(reason).replace(" (+", " ").replace(")", "") for reason in reasons[:2]]
        return ", ".join(trimmed) if trimmed else "n/a"

    def _candidate_signal_score_display(self, candidate) -> str:
        if getattr(candidate, "selection_source", None) == "shortlist_model":
            return "n/a"
        signal_score = getattr(candidate, "signal_score", math.nan)
        if self._is_finite(signal_score):
            return f"{float(signal_score):.1f}"
        return "n/a"

    def _candidate_selector_note(self, candidate) -> str:
        selection_score = float(getattr(candidate, "selection_score", candidate.signal_score))
        selected_rank = getattr(candidate, "selected_rank", None)
        parts = [f"final {selection_score:.2f}"]
        selection_source = getattr(candidate, "selection_source", None)
        model_predicted_alpha = getattr(candidate, "model_predicted_alpha", math.nan)
        model_rank = getattr(candidate, "model_rank", math.nan)
        model_name = getattr(candidate, "model_name", None)
        model_reason_summary = getattr(candidate, "model_reason_summary", None)
        model_comparison_summary = getattr(candidate, "model_comparison_summary", None)
        if pd.isna(model_reason_summary):
            model_reason_summary = None
        if pd.isna(model_comparison_summary):
            model_comparison_summary = None
        if selection_source == "shortlist_model" and self._is_finite(model_predicted_alpha):
            model_part = f"model {float(model_predicted_alpha):+.3f}"
            if self._is_finite(model_rank):
                model_part += f" rank #{int(float(model_rank))}"
            if model_name:
                model_part += f" ({model_name})"
            parts.append(model_part)
            if model_reason_summary:
                parts.append(str(model_reason_summary))
            if model_comparison_summary:
                parts.append(f"won over {model_comparison_summary}")
        if selected_rank is not None and self._is_finite(selected_rank):
            parts.append(f"rank #{int(float(selected_rank))}")
        ranker_score = getattr(candidate, "ranker_score", math.nan)
        learned_weight = getattr(candidate, "learned_ranker_weight", math.nan)
        if self._is_finite(ranker_score):
            learned_note = f"learned {float(ranker_score):+.3f}"
            if self._is_finite(learned_weight) and float(learned_weight) != 0.0:
                learned_note += f" x {float(learned_weight):.2f}"
            parts.append(learned_note)
        model_note = self._candidate_model_note(candidate)
        if model_note != "n/a":
            parts.append(model_note)
        return "; ".join(parts)

    def _candidate_opportunity_breakdown(self, candidate, scan_policy: ScanPolicy) -> str:
        setup = float(getattr(candidate, "setup_quality_score", 0.0)) * float(scan_policy.signal_score_weight)
        alpha = float(getattr(candidate, "expected_alpha_score", 0.0)) * float(scan_policy.expected_alpha_weight)
        freshness = float(getattr(candidate, "freshness_score", 0.0)) * float(scan_policy.freshness_weight)
        breadth = float(getattr(candidate, "breadth_score", 0.0)) * float(scan_policy.breadth_weight)
        overlap = float(getattr(candidate, "overlap_penalty", 0.0))
        return (
            f"setup {setup:.2f} + alpha {alpha:.2f} + fresh {freshness:.2f} + "
            f"breadth {breadth:.2f} - overlap {overlap:.2f}"
        )

    def _candidate_signal_evidence(self, candidate) -> str:
        if isinstance(candidate, dict):
            indicator_details = candidate.get("indicator_details", {}) or {}
        else:
            indicator_details = getattr(candidate, "indicator_details", {}) or {}
        summarized = self._summarize_candidate_why(indicator_details)
        if summarized != "n/a":
            return summarized
        feature_parts: list[str] = []
        rs_spy = self._candidate_attr(candidate, "relative_strength_index_vs_spy")
        if self._is_finite(rs_spy) and float(rs_spy) >= 80.0:
            feature_parts.append("RS vs SPY")
        roc_63 = self._candidate_attr(candidate, "roc_63")
        if self._is_finite(roc_63) and float(roc_63) >= 0.10:
            feature_parts.append("63d momentum")
        sma_200_dist = self._candidate_attr(candidate, "sma_200_dist")
        if self._is_finite(sma_200_dist) and float(sma_200_dist) > 0.0:
            feature_parts.append("above 200d")
        vol_alpha = self._candidate_attr(candidate, "vol_alpha")
        if self._is_finite(vol_alpha) and float(vol_alpha) >= 1.0:
            feature_parts.append("volume confirmation")
        sector_breadth = self._candidate_attr(candidate, "sector_pct_above_50")
        if self._is_finite(sector_breadth) and float(sector_breadth) >= 0.70:
            feature_parts.append("strong sector breadth")
        breakout = self._candidate_attr(candidate, "breakout_above_20d_high")
        if self._is_finite(breakout) and float(breakout) >= 1.0:
            feature_parts.append("20d breakout")
        return ", ".join(feature_parts[:4]) if feature_parts else "n/a"

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
                hard.append(self._humanize_indicator_name(name))
            elif mode == "score_component" and passed:
                scored.append((float(detail.get("score", 0.0)), self._humanize_indicator_name(name)))
        scored.sort(reverse=True)
        parts = hard[:2] + [name for _, name in scored[:2]]
        return ", ".join(parts) if parts else "n/a"

    def _humanize_indicator_name(self, name: str) -> str:
        labels = {
            "relative_strength_index_vs_spy_min": "RS vs SPY",
            "relative_strength_index_vs_qqq_min": "RS vs QQQ",
            "relative_strength_index_vs_xlk_min": "RS vs XLK",
            "relative_strength_index_vs_subindustry_min": "RS vs group ETF",
            "signal_score_min": "signal score",
            "roc_63_min": "63d momentum",
            "roc_126_min": "126d momentum",
            "sma_200_dist_min": "above 200d",
            "sma_50_dist_min": "above 50d",
            "sma_50_dist_max": "shallow pullback",
            "rsi_14_max": "RSI range",
            "rsi_14_min": "RSI support",
            "vol_alpha_min": "volume confirmation",
            "days_to_next_earnings_min": "earnings clearance",
            "days_to_next_earnings_max": "earnings timing",
            "days_since_last_earnings_max": "fresh post-earnings setup",
            "avg_abs_gap_pct_20_max": "gap discipline",
            "max_gap_down_pct_60_max": "gap risk control",
            "breakout_above_20d_high_min": "20d breakout",
            "distance_above_20d_high_max": "fresh breakout distance",
            "base_range_pct_20_max": "tight base",
            "base_atr_contraction_20_max": "ATR contraction",
            "base_volume_dryup_ratio_20_max": "volume dry-up",
            "breakout_volume_ratio_50_min": "breakout volume",
            "ma_alignment_50_200_min": "50d above 200d",
            "ma_slope_50_20_min": "50d slope",
            "ma_slope_200_20_min": "200d slope",
            "oil_corr_60_min": "oil correlation",
        }
        return labels.get(name, name.replace("_", " "))

    def _candidate_feature_snapshot(self, row: dict) -> dict[str, object]:
        snapshot: dict[str, object] = {}
        for name in [
            "adj_close",
            "atr_14",
            "md_volume_30d",
            "regime_etf",
            "subindustry_benchmark",
            "relative_strength_index_vs_spy",
            "relative_strength_index_vs_qqq",
            "relative_strength_index_vs_xlk",
            "relative_strength_index_vs_subindustry",
            "roc_63",
            "roc_126",
            "vol_alpha",
            "sma_200_dist",
            "sma_50_dist",
            "rsi_14",
            "atr_14",
            "atr_pct_14",
            "atr_pct_14_percentile_252",
            "realized_vol_20_percentile_252",
            "days_to_next_earnings",
            "days_since_last_earnings",
            "last_earnings_gap_pct",
            "last_earnings_volume_ratio_20",
            "last_earnings_open_vs_20d_high",
            "close_vs_last_earnings_close",
            "avg_abs_gap_pct_20",
            "max_gap_down_pct_60",
            "distance_above_20d_high",
            "distance_from_52w_high",
            "days_since_52w_high",
            "base_range_pct_20",
            "base_atr_contraction_20",
            "base_volume_dryup_ratio_20",
            "breakout_volume_ratio_50",
            "dollar_volume_ratio_20_60",
            "volume_percentile_60",
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
        if row.get("sub_industry"):
            snapshot["sub_industry"] = row.get("sub_industry")
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

    def _open_trade_tickers(self, open_trades: list) -> set[str]:
        tickers: set[str] = set()
        for trade in open_trades:
            ticker = None
            if isinstance(trade, dict):
                ticker = trade.get("ticker")
            else:
                try:
                    ticker = trade["ticker"]
                except (KeyError, TypeError, IndexError):
                    ticker = getattr(trade, "ticker", None)
            if ticker not in (None, ""):
                tickers.add(str(ticker))
        return tickers

    def _open_trade_lookup(self, open_trades: list) -> dict[str, dict]:
        lookup: dict[str, dict] = {}
        today = date.today()
        for trade in open_trades:
            ticker = self._trade_value(trade, "ticker")
            if ticker in (None, ""):
                continue
            entry_date = self._trade_value(trade, "entry_date")
            entry_ts = pd.to_datetime(entry_date, errors="coerce")
            days_held = None
            if pd.notna(entry_ts):
                days_held = int(np.busday_count(entry_ts.date(), today))
            lookup[str(ticker)] = {
                "entry_date": entry_date,
                "entry_price": self._trade_value(trade, "entry_price"),
                "days_held": days_held,
            }
        return lookup

    def _trade_value(self, trade, key: str):
        if isinstance(trade, dict):
            return trade.get(key)
        if hasattr(trade, "keys") and key in trade.keys():
            return trade[key]
        try:
            return trade[key]
        except Exception:
            return getattr(trade, key, None)

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

    def _candidate_attr(self, candidate, field_name: str):
        if isinstance(candidate, dict):
            return candidate.get(field_name, pd.NA)
        return getattr(candidate, field_name, pd.NA)

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
