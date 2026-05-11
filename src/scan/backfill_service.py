from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.scan.service import ScanPolicy, ScanService
from src.settings import get_settings, load_feature_config
from src.sync.service import REFERENCE_TICKERS
from src.utils.db_manager import DatabaseManager
from src.utils.logging import get_logger
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
    ) -> tuple[int, int]:
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
                        "why": scan_service._summarize_candidate_why(row.get("indicator_details", {}) or {}),
                        "already_owned": bool(row.get("already_owned", False)),
                        "pre_penalty_opportunity_score": float(
                            row.get("opportunity_score", 0.0) + row.get("overlap_penalty", 0.0)
                        ),
                        "overlap_components": row.get("overlap_components", {}),
                        "feature_snapshot": scan_service._candidate_feature_snapshot(row),
                        "ranking_components": {
                            "signal_score": float(row.get("signal_score", 0.0)),
                            "setup_quality_score": float(row.get("setup_quality_score", 0.0)),
                            "expected_alpha_score": float(row.get("expected_alpha_score", 0.0)),
                            "breadth_score": float(row.get("breadth_score", 0.0)),
                            "freshness_score": float(row.get("freshness_score", 0.0)),
                            "overlap_penalty": float(row.get("overlap_penalty", 0.0)),
                            "opportunity_score": float(row.get("opportunity_score", 0.0)),
                        },
                    },
                }
            )
        self.db_manager.replace_scan_candidates(scan_date=scan_date, rows=persisted_rows)
        return len(persisted_rows), len(selected.index)
