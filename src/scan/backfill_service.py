from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pandas as pd

from src.scan.service import ScanPolicy, ScanService
from src.settings import get_settings, load_feature_config
from src.sync.service import REFERENCE_TICKERS
from src.utils.db_manager import DatabaseManager
from src.utils.logging import get_logger
from src.utils.shortlist_runtime import _annotate_live_prediction_comparisons, _parse_prediction_details
from src.utils.signal_engine import build_analysis_frame, filter_signal_candidates
from src.utils.sizing import compute_position_size
from src.utils.strategy import load_active_strategies


@dataclass(frozen=True)
class ScanBackfillReport:
    scan_dates_processed: int
    scan_dates_skipped: int
    total_candidates: int
    total_selected: int


class ScanBackfillService:
    def __init__(self, db_manager: DatabaseManager) -> None:
        self.db_manager = db_manager
        self.logger = get_logger("scan_backfill")

    def run(
        self,
        *,
        date_from: str,
        date_to: str | None = None,
        skip_existing: bool = False,
    ) -> ScanBackfillReport:
        self.db_manager.initialize()
        strategies = load_active_strategies()
        settings = get_settings()
        config = load_feature_config()
        scan_policy = ScanPolicy.from_config(config)
        universe_rows = self.db_manager.list_universe_rows(active_only=True)
        if not universe_rows:
            raise ValueError("Universe is empty. Run `sq sync` first.")

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
            raise ValueError("No analysis frame could be built for historical scan backfill.")

        scan_dates = self._scan_dates(
            analysis_frame=analysis_frame,
            universe_tickers=universe_tickers,
            date_from=date_from,
            date_to=date_to,
        )
        if not scan_dates:
            raise ValueError("No trading dates matched the requested scan-backfill range.")

        scan_service = ScanService(self.db_manager, email_sender=lambda subject, html_body, settings: None)
        sector_map = {row["ticker"]: row["sector"] for row in universe_rows}
        overlap_context = scan_service._build_overlap_context(
            open_trades=[],
            strategies=strategies,
            sector_map=sector_map,
        )
        shortlist_model_context = scan_service._load_shortlist_model_context(scan_policy)
        historical_model_predictions = self._load_historical_shortlist_predictions(
            scan_service=scan_service,
            scan_policy=scan_policy,
            shortlist_model_context=shortlist_model_context,
        )

        processed = 0
        skipped = 0
        total_candidates = 0
        total_selected = 0
        existing_dates: set[str] = set()
        if skip_existing:
            existing = self.db_manager.load_scan_candidates()
            existing_dates = set(existing["scan_date"].astype(str)) if not existing.empty else set()

        for scan_date in scan_dates:
            scan_date_str = scan_date.strftime("%Y-%m-%d")
            if skip_existing and scan_date_str in existing_dates:
                skipped += 1
                continue
            snapshot = analysis_frame[
                (pd.to_datetime(analysis_frame["date"]).dt.normalize() == scan_date)
                & analysis_frame["ticker"].isin(universe_tickers)
                & analysis_frame["regime_green"].fillna(False)
            ].copy()
            persisted_rows, selected_count = self._persist_scan_for_date(
                scan_service=scan_service,
                snapshot=snapshot,
                scan_date=scan_date_str,
                strategies=strategies,
                scan_policy=scan_policy,
                settings=settings,
                overlap_context=overlap_context,
                historical_model_predictions=historical_model_predictions,
            )
            processed += 1
            total_candidates += persisted_rows
            total_selected += selected_count
            if processed == 1 or processed % 20 == 0 or processed == len(scan_dates):
                self.logger.info(
                    "Historical scan backfill progress: processed=%s/%s current_date=%s candidates=%s selected=%s",
                    processed,
                    len(scan_dates),
                    scan_date_str,
                    persisted_rows,
                    selected_count,
                )

        return ScanBackfillReport(
            scan_dates_processed=processed,
            scan_dates_skipped=skipped,
            total_candidates=total_candidates,
            total_selected=total_selected,
        )

    def _scan_dates(
        self,
        *,
        analysis_frame: pd.DataFrame,
        universe_tickers: list[str],
        date_from: str,
        date_to: str | None,
    ) -> list[pd.Timestamp]:
        working = analysis_frame[analysis_frame["ticker"].isin(set(universe_tickers).union({"SPY", "QQQ"}))].copy()
        if working.empty:
            return []
        all_dates = sorted(pd.to_datetime(working["date"]).dt.normalize().drop_duplicates().tolist())
        start = pd.Timestamp(date_from).normalize()
        end = pd.Timestamp(date_to).normalize() if date_to is not None else all_dates[-1]
        return [date_value for date_value in all_dates if start <= date_value <= end]

    def _persist_scan_for_date(
        self,
        *,
        scan_service: ScanService,
        snapshot: pd.DataFrame,
        scan_date: str,
        strategies: dict,
        scan_policy: ScanPolicy,
        settings,
        overlap_context: dict[str, set[str]],
        historical_model_predictions: pd.DataFrame | None,
    ) -> tuple[int, int]:
        historical_model_context = self._historical_shortlist_context_for_date(
            historical_model_predictions=historical_model_predictions,
            scan_date=scan_date,
        )
        if historical_model_context is not None:
            candidates = scan_service._build_shortlist_model_candidates(
                snapshot=snapshot,
                strategies=strategies,
                shortlist_model_context=historical_model_context,
                scan_policy=scan_policy,
                overlap_context=overlap_context,
                settings=settings,
            )
            if not candidates.empty:
                persisted_candidates = candidates.copy()
                candidates = scan_service._apply_shortlist_model_selection(candidates, historical_model_context)
                if "selection_source" not in candidates.columns:
                    candidates["selection_source"] = "shortlist_model"
                persisted_candidates = scan_service._attach_selection_metadata_to_persisted_candidates(
                    persisted_candidates,
                    candidates,
                )
                selected_candidates = candidates[
                    pd.to_numeric(candidates["opportunity_score"], errors="coerce")
                    >= float(scan_policy.shortlist_model.min_opportunity_score)
                ].copy()
                selected = scan_service._apply_portfolio_caps(selected_candidates, scan_policy)
                persisted_rows = scan_service._build_persisted_scan_rows(persisted_candidates, selected)
                self.db_manager.replace_scan_candidates(scan_date=scan_date, rows=persisted_rows)
                return len(persisted_rows), len(selected.index)

        candidate_frames: list[pd.DataFrame] = []
        persisted_candidate_frames: list[pd.DataFrame] = []
        for slot, strategy in strategies.items():
            scoped_snapshot = scan_service._scope_snapshot(snapshot, strategy)
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
                scan_service._score_candidate(
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
            self.db_manager.replace_scan_candidates(scan_date=scan_date, rows=[])
            return 0, 0

        candidates = pd.concat(candidate_frames, ignore_index=True).reset_index(drop=True)
        persisted_candidates = pd.concat(persisted_candidate_frames, ignore_index=True).reset_index(drop=True)
        selected = scan_service._apply_portfolio_caps(candidates, scan_policy)
        selected = selected[selected["opportunity_score"] >= scan_policy.min_opportunity_score].copy()
        persisted_rows = scan_service._build_persisted_scan_rows(persisted_candidates, selected)
        self.db_manager.replace_scan_candidates(scan_date=scan_date, rows=persisted_rows)
        return len(persisted_rows), len(selected.index)

    def _load_historical_shortlist_predictions(
        self,
        *,
        scan_service: ScanService,
        scan_policy: ScanPolicy,
        shortlist_model_context,
    ) -> pd.DataFrame | None:
        if shortlist_model_context is None:
            return None
        if not hasattr(self.db_manager, "load_shortlist_model_predictions"):
            return None
        prediction_frames: list[pd.DataFrame] = []
        base_kwargs = {
            "generated_at": shortlist_model_context.generated_at,
            "horizon_days": int(scan_policy.shortlist_model.horizon_days),
            "eligible_universe_mode": scan_policy.shortlist_model.production_eligible_universe_mode,
            "model_scope": scan_policy.shortlist_model.production_model_scope,
            "model_name": scan_policy.shortlist_model.production_model_name or shortlist_model_context.champion_model,
        }
        for dataset_split in ("oos", "live"):
            frame = self.db_manager.load_shortlist_model_predictions(
                dataset_split=dataset_split,
                **base_kwargs,
            )
            if frame.empty:
                continue
            frame = frame.copy()
            frame["snapshot_date"] = pd.to_datetime(frame["snapshot_date"]).dt.normalize()
            frame["predicted_alpha"] = pd.to_numeric(frame["predicted_alpha"], errors="coerce")
            prediction_frames.append(frame)
        if not prediction_frames:
            self.logger.info("Historical scan backfill falling back to heuristic gate because no shortlist-model predictions were available.")
            return None
        combined = pd.concat(prediction_frames, ignore_index=True)
        combined = combined.sort_values(
            ["snapshot_date", "predicted_alpha", "md_volume_30d", "ticker"],
            ascending=[True, False, False, True],
        ).reset_index(drop=True)
        if "details_json" in combined.columns:
            details = combined["details_json"].map(_parse_prediction_details)
            combined["model_top_reasons"] = details.apply(lambda payload: payload.get("model_top_reasons", []))
            combined["model_reason_summary"] = details.apply(lambda payload: payload.get("model_reason_summary"))
        grouped_frames: list[pd.DataFrame] = []
        for _, day_frame in combined.groupby("snapshot_date", sort=True):
            ranked = day_frame.copy().reset_index(drop=True)
            ranked["model_rank"] = range(1, len(ranked.index) + 1)
            ranked = _annotate_live_prediction_comparisons(ranked, top_n=int(scan_policy.shortlist_model.top_n))
            grouped_frames.append(ranked)
        return pd.concat(grouped_frames, ignore_index=True) if grouped_frames else None

    def _historical_shortlist_context_for_date(
        self,
        *,
        historical_model_predictions: pd.DataFrame | None,
        scan_date: str,
    ):
        if historical_model_predictions is None or historical_model_predictions.empty:
            return None
        prediction_date = pd.Timestamp(scan_date).normalize()
        date_predictions = historical_model_predictions[
            historical_model_predictions["snapshot_date"] == prediction_date
        ].copy()
        if date_predictions.empty:
            return None
        model_name = str(date_predictions["model_name"].iloc[0]) if "model_name" in date_predictions.columns else "xgboost_model"
        generated_at = str(date_predictions["generated_at"].iloc[0]) if "generated_at" in date_predictions.columns else scan_date
        return SimpleNamespace(
            generated_at=generated_at,
            champion_model=model_name,
            live_predictions=date_predictions,
        )
