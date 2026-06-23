from __future__ import annotations

import argparse
from html import escape
import sys

from src.research.alpha_service import AlphaResearchService
from src.research.exit_analysis_service import ExitAnalysisService
from src.research.factor_tearsheet_service import FactorTearsheetService
from src.research.rsi_exit_bakeoff_service import RsiExitBakeoffService
from src.research.shortlist_bakeoff_service import ShortlistBakeoffService
from src.research.shortlist_allocation_analysis_service import ShortlistAllocationAnalysisService
from src.research.shortlist_model_service import ShortlistModelService
from src.research.shortlist_promote_service import ShortlistPromoteService
from src.research.shortlist_scoreboard_service import ShortlistScoreboardService
from src.research.shortlist_sector_reactivation_service import ShortlistSectorReactivationService
from src.research.shortlist_service import ShortlistService
from src.research.shortlist_tune_service import ShortlistTuneService
from src.research.subindustry_attribution_service import SubindustryAttributionService
from src.research.universe_analysis_service import UniverseAnalysisService
from src.research.universe_snapshot_service import UniverseSnapshotBackfillService
from src.evaluate.service import EvaluateService
from src.monitor.service import MonitorService
from src.positions.service import PositionsService
from src.promote.service import PromoteService
from src.quote.service import QuoteService
from src.research.service import ResearchService
from src.scan.analyst_snapshot_service import AnalystSnapshotService
from src.scan.analysis_service import ScanAnalysisService
from src.scan.backfill_service import ScanBackfillService
from src.scan.performance_service import ScanPerformanceService
from src.scan.portfolio_rotation_service import PortfolioRotationService
from src.scan.slot_attribution_service import SlotAttributionService
from src.scan.service import ScanService
from src.sleeves.service import SleeveResearchService
from src.sweep.service import SweepService
from src.sync.refresh_service import RefreshUniverseService
from src.sync.service import SyncService
from src.trade.service import TradeService
from src.research.shortlist_universe import VALID_ELIGIBLE_UNIVERSE_MODES, VALID_MODEL_SCOPES
from src.utils.db_manager import DatabaseManager
from src.utils.emailer import send_html_email
from src.utils.logging import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sq")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("init-db", help="Initialize DuckDB and SQLite schemas.")
    subparsers.add_parser("sync", help="Bootstrap the universe and sync OHLCV history.")
    subparsers.add_parser("refresh-universe", help="Refresh sector and sub-industry metadata for the tracked universe.")
    subparsers.add_parser("research", help="Run the research model and print feature importance.")
    alpha_research_parser = subparsers.add_parser("alpha-research", help="Model forward excess return directly.")
    alpha_research_parser.add_argument("--top", type=int, default=10)
    alpha_research_parser.add_argument("--sector", type=str, default=None)
    alpha_research_parser.add_argument("--horizon-days", type=int, default=10)
    alpha_research_parser.add_argument("--benchmark", type=str, default="sector")
    sweep_parser = subparsers.add_parser("sweep", help="Run the sweep grid and persist backtest results.")
    sweep_parser.add_argument("--mode", type=str, default="default")

    evaluate_parser = subparsers.add_parser("evaluate", help="Rank backtest results and write candidates.md.")
    evaluate_parser.add_argument("--top", type=int, default=10)
    evaluate_parser.add_argument("--sector", type=str, default=None)
    evaluate_parser.add_argument("--run-id", type=int, default=None)
    evaluate_parser.add_argument("--min-trades", type=int, default=12)
    evaluate_parser.add_argument("--walk-forward", action="store_true")
    evaluate_parser.add_argument("--walk-forward-windows", type=int, default=5)
    evaluate_parser.add_argument("--walk-forward-shortlist", type=int, default=25)
    sleeve_parser = subparsers.add_parser("sleeve-research", help="Run breadth-aware rank-based sleeve research.")
    sleeve_parser.add_argument("--top", type=int, default=10)
    sleeve_parser.add_argument("--sector", action="append", default=None)
    sleeve_parser.add_argument("--walk-forward", action="store_true")
    sleeve_parser.add_argument("--walk-forward-windows", type=int, default=5)
    sleeve_parser.add_argument("--walk-forward-shortlist", type=int, default=10)

    promote_parser = subparsers.add_parser("promote", help="Promote a backtest result to production.")
    promote_parser.add_argument("--id", dest="row_id", type=int, required=True)
    promote_parser.add_argument("--slot", type=str, default=None)

    trade_parser = subparsers.add_parser("trade", help="Record buy or sell trades in the ledger.")
    trade_parser.add_argument("action", choices=("buy", "sell"))
    trade_parser.add_argument("ticker")
    trade_parser.add_argument("price", type=float)
    trade_parser.add_argument("shares", type=int, nargs="?")
    trade_parser.add_argument("--slot", dest="strategy_slot", type=str, default=None)

    subparsers.add_parser("positions", help="Summarize open positions and current sell context.")
    quote_parser = subparsers.add_parser("quote", help="Show current price and live trade context for a ticker.")
    quote_parser.add_argument("ticker")
    subparsers.add_parser("scan", help="Run the daily signal scan and email the evening brief.")
    analyst_snapshot_parser = subparsers.add_parser("analyst-snapshot", help="Capture point-in-time analyst target snapshots.")
    analyst_snapshot_parser.add_argument("--snapshot-date", type=str, default=None)
    analyst_snapshot_parser.add_argument("--source", choices=("research", "active"), default="research")
    analyst_snapshot_parser.add_argument("--top", type=int, default=250)
    analyst_snapshot_parser.add_argument("--ticker", action="append", default=None)
    scan_backfill_parser = subparsers.add_parser("scan-backfill", help="Replay daily scans across historical dates without using future prices.")
    scan_backfill_parser.add_argument("--date-from", required=True, type=str)
    scan_backfill_parser.add_argument("--date-to", default=None, type=str)
    scan_backfill_parser.add_argument("--skip-existing", action="store_true")
    scan_performance_parser = subparsers.add_parser("scan-performance", help="Summarize realized performance for prior selected scan picks.")
    scan_performance_parser.add_argument("--recent-scan-dates", type=int, default=0)
    scan_performance_parser.add_argument("--recent-picks", type=int, default=20)
    scan_performance_parser.add_argument("--benchmark", choices=("sector", "spy"), default="sector")
    scan_performance_parser.add_argument("--selection-source", type=str, default=None)
    scan_performance_parser.add_argument("--model-name", type=str, default=None)
    scan_performance_parser.add_argument("--model-generated-at", type=str, default=None)
    scan_performance_parser.add_argument("--all-sources", action="store_true")
    scan_performance_parser.add_argument("--email", action="store_true")
    portfolio_rotation_parser = subparsers.add_parser("portfolio-rotation", help="Backtest a fixed-slot portfolio rotation from scan targets.")
    portfolio_rotation_parser.add_argument("--target-positions", type=int, default=6)
    portfolio_rotation_parser.add_argument("--max-hold-days", type=int, default=20)
    portfolio_rotation_parser.add_argument("--min-pre-opportunity", type=float, default=0.40)
    portfolio_rotation_parser.add_argument("--min-model-alpha", type=float, default=0.0)
    portfolio_rotation_parser.add_argument("--initial-equity", type=float, default=1.0)
    portfolio_rotation_parser.add_argument("--walk-forward", action="store_true")
    portfolio_rotation_parser.add_argument("--horizon-days", type=int, default=20)
    portfolio_rotation_parser.add_argument("--model-generated-at", type=str, default=None)
    portfolio_rotation_parser.add_argument("--model-name", type=str, default=None)
    portfolio_rotation_parser.add_argument("--eligible-universe-mode", type=str, default=None)
    portfolio_rotation_parser.add_argument("--model-scope", type=str, default=None)
    portfolio_rotation_parser.add_argument("--transaction-cost-bps", type=float, default=0.0)
    portfolio_rotation_parser.add_argument("--slippage-bps", type=float, default=0.0)
    portfolio_rotation_parser.add_argument("--cooldown-days", type=int, default=0)
    portfolio_rotation_parser.add_argument("--no-reinvest-gains", action="store_true")
    portfolio_rotation_parser.add_argument("--max-new-entries-per-scan", type=int, default=None)
    portfolio_rotation_parser.add_argument("--date-from", type=str, default=None)
    portfolio_rotation_parser.add_argument("--date-to", type=str, default=None)
    universe_backfill_parser = subparsers.add_parser("universe-backfill", help="Persist daily broad-universe feature snapshots with future outcome labels for research.")
    universe_backfill_parser.add_argument("--date-from", required=True, type=str)
    universe_backfill_parser.add_argument("--date-to", default=None, type=str)
    universe_backfill_parser.add_argument("--skip-existing", action="store_true")
    universe_analysis_parser = subparsers.add_parser("universe-analysis", help="Analyze broad-universe snapshots to find missed winners and gate blind spots.")
    universe_analysis_parser.add_argument("--top", type=int, default=10)
    universe_analysis_parser.add_argument("--horizon-days", type=int, default=10)
    universe_analysis_parser.add_argument("--recent-dates", type=int, default=20)
    factor_tearsheet_parser = subparsers.add_parser("factor-tearsheet", help="Render a sector-specific factor tearsheet from broad-universe snapshots.")
    factor_tearsheet_parser.add_argument("--sector", required=True, type=str)
    factor_tearsheet_parser.add_argument("--horizon", type=int, default=10)
    shortlist_bakeoff_parser = subparsers.add_parser("shortlist-bakeoff", help="Compare shortlist policies directly on forward sector alpha.")
    shortlist_bakeoff_parser.add_argument("--top", type=int, default=6)
    shortlist_bakeoff_parser.add_argument("--horizon", type=int, default=20)
    shortlist_bakeoff_parser.add_argument("--recent-dates", type=int, default=40)
    shortlist_bakeoff_parser.add_argument("--eligible-universe-mode", choices=VALID_ELIGIBLE_UNIVERSE_MODES, default="passed_only")
    shortlist_model_parser = subparsers.add_parser("shortlist-model", help="Train and evaluate walk-forward shortlist models on forward sector alpha.")
    shortlist_model_parser.add_argument("--top", type=int, default=10)
    shortlist_model_parser.add_argument("--horizon", type=int, default=20)
    shortlist_model_parser.add_argument("--min-train-dates", type=int, default=252)
    shortlist_model_parser.add_argument("--test-window-dates", type=int, default=20)
    shortlist_model_parser.add_argument("--recent-dates", type=int, default=60)
    shortlist_model_parser.add_argument("--eligible-universe-mode", choices=VALID_ELIGIBLE_UNIVERSE_MODES, default="passed_only")
    shortlist_model_parser.add_argument("--model-scope", choices=VALID_MODEL_SCOPES, default="global")
    shortlist_model_parser.add_argument("--xgboost-config", choices=["baseline", "balanced_depth4", "shallower_regularized"], default="baseline")
    shortlist_scoreboard_parser = subparsers.add_parser("shortlist-scoreboard", help="Render model scorecards and explicit promotion decisions for shortlist candidates.")
    shortlist_scoreboard_parser.add_argument("--top", type=int, default=10)
    shortlist_scoreboard_parser.add_argument("--horizon", type=int, default=20)
    shortlist_scoreboard_parser.add_argument("--min-train-dates", type=int, default=252)
    shortlist_scoreboard_parser.add_argument("--test-window-dates", type=int, default=20)
    shortlist_scoreboard_parser.add_argument("--recent-dates", type=int, default=60)
    shortlist_scoreboard_parser.add_argument("--eligible-universe-mode", choices=VALID_ELIGIBLE_UNIVERSE_MODES, default="passed_only")
    shortlist_scoreboard_parser.add_argument("--model-scope", choices=VALID_MODEL_SCOPES, default="global")
    shortlist_scoreboard_parser.add_argument("--no-refresh-if-stale", action="store_true")
    shortlist_parser = subparsers.add_parser("shortlist", help="Render the latest persisted model-driven shortlist.")
    shortlist_parser.add_argument("--top", type=int, default=10)
    shortlist_parser.add_argument("--horizon", type=int, default=20)
    shortlist_parser.add_argument("--min-train-dates", type=int, default=252)
    shortlist_parser.add_argument("--test-window-dates", type=int, default=20)
    shortlist_parser.add_argument("--recent-dates", type=int, default=60)
    shortlist_parser.add_argument("--eligible-universe-mode", choices=VALID_ELIGIBLE_UNIVERSE_MODES, default="passed_only")
    shortlist_parser.add_argument("--model-scope", choices=VALID_MODEL_SCOPES, default="global")
    shortlist_parser.add_argument("--no-refresh-if-stale", action="store_true")
    shortlist_allocation_parser = subparsers.add_parser("shortlist-allocation-analysis", help="Compare portfolio construction policies on persisted shortlist model predictions.")
    shortlist_allocation_parser.add_argument("--top", type=int, default=6)
    shortlist_allocation_parser.add_argument("--horizon", type=int, default=20)
    shortlist_allocation_parser.add_argument("--min-train-dates", type=int, default=252)
    shortlist_allocation_parser.add_argument("--test-window-dates", type=int, default=20)
    shortlist_allocation_parser.add_argument("--recent-dates", type=int, default=60)
    shortlist_allocation_parser.add_argument("--eligible-universe-mode", choices=VALID_ELIGIBLE_UNIVERSE_MODES, default="passed_only")
    shortlist_allocation_parser.add_argument("--model-scope", choices=VALID_MODEL_SCOPES, default="global")
    shortlist_allocation_parser.add_argument("--model-name", type=str, default=None)
    shortlist_allocation_parser.add_argument("--compare-scopes", action="store_true")
    shortlist_allocation_parser.add_argument("--no-refresh-if-stale", action="store_true")
    shortlist_reactivation_parser = subparsers.add_parser("shortlist-sector-reactivation", help="Research whether a candidate sector should be reintroduced into the shortlist.")
    shortlist_reactivation_parser.add_argument("--top", type=int, default=6)
    shortlist_reactivation_parser.add_argument("--horizon", type=int, default=20)
    shortlist_reactivation_parser.add_argument("--min-train-dates", type=int, default=252)
    shortlist_reactivation_parser.add_argument("--test-window-dates", type=int, default=20)
    shortlist_reactivation_parser.add_argument("--recent-dates", type=int, default=60)
    shortlist_reactivation_parser.add_argument("--eligible-universe-mode", choices=VALID_ELIGIBLE_UNIVERSE_MODES, default=None)
    shortlist_reactivation_parser.add_argument("--model-scope", choices=VALID_MODEL_SCOPES, default=None)
    shortlist_reactivation_parser.add_argument("--model-name", type=str, default=None)
    shortlist_reactivation_parser.add_argument("--xgboost-config", choices=["baseline", "balanced_depth4", "shallower_regularized"], default=None)
    shortlist_reactivation_parser.add_argument("--candidate-sector", action="append", default=["Information Technology"])
    shortlist_reactivation_parser.add_argument("--no-refresh-if-stale", action="store_true")
    shortlist_promote_parser = subparsers.add_parser("shortlist-promote", help="Pin a shortlist model configuration for production scan and monitor.")
    shortlist_promote_parser.add_argument("--model-name", required=True, type=str)
    shortlist_promote_parser.add_argument("--eligible-universe-mode", choices=VALID_ELIGIBLE_UNIVERSE_MODES, required=True)
    shortlist_promote_parser.add_argument("--model-scope", choices=VALID_MODEL_SCOPES, required=True)
    shortlist_promote_parser.add_argument("--xgboost-config", choices=["baseline", "balanced_depth4", "shallower_regularized"], default="baseline")
    shortlist_promote_parser.add_argument("--horizon", type=int, default=20)
    shortlist_tune_parser = subparsers.add_parser("shortlist-tune", help="Tune sector-specific xgboost shortlist parameters and run feature ablations.")
    shortlist_tune_parser.add_argument("--top", type=int, default=10)
    shortlist_tune_parser.add_argument("--horizon", type=int, default=20)
    shortlist_tune_parser.add_argument("--min-train-dates", type=int, default=252)
    shortlist_tune_parser.add_argument("--test-window-dates", type=int, default=20)
    shortlist_tune_parser.add_argument("--recent-dates", type=int, default=60)
    shortlist_tune_parser.add_argument("--eligible-universe-mode", choices=VALID_ELIGIBLE_UNIVERSE_MODES, default="passed_or_trend")
    shortlist_tune_parser.add_argument("--model-scope", choices=VALID_MODEL_SCOPES, default="sector_specific")
    shortlist_tune_parser.add_argument("--mode", choices=["full", "tune_only", "ablation_only"], default="full")
    shortlist_tune_parser.add_argument("--tuning-profile", choices=["focused", "full"], default="focused")
    shortlist_tune_parser.add_argument("--ablation-profile", choices=["focused", "full"], default="focused")
    shortlist_tune_parser.add_argument("--ablation-params-candidate", type=str, default=None)
    exit_analysis_parser = subparsers.add_parser("exit-analysis", help="Compare realized exits against simple fixed-horizon counterfactual holds.")
    exit_analysis_parser.add_argument("--horizons", type=int, nargs="*", default=[5, 10, 15, 20])
    rsi_exit_bakeoff_parser = subparsers.add_parser("rsi-exit-bakeoff", help="Compare RSI_2 exit variants on historical selected scan picks.")
    rsi_exit_bakeoff_parser.add_argument("--recent-scan-dates", type=int, default=60)
    rsi_exit_bakeoff_parser.add_argument("--benchmark", choices=("sector", "spy"), default="sector")
    subindustry_attribution_parser = subparsers.add_parser("subindustry-attribution", help="Break the best sector strategy down by subindustry.")
    subindustry_attribution_parser.add_argument("--sector", required=True, type=str)
    subindustry_attribution_parser.add_argument("--run-id", type=int, default=None)
    subindustry_attribution_parser.add_argument("--strategy-id", type=int, default=None)
    subindustry_attribution_parser.add_argument("--min-trades", type=int, default=12)
    scan_analysis_parser = subparsers.add_parser("scan-analysis", help="Analyze scan snapshots and selection attribution.")
    scan_analysis_parser.add_argument("--date", type=str, default=None)
    scan_analysis_parser.add_argument("--refresh", action="store_true")
    scan_analysis_parser.add_argument("--horizons", type=int, nargs="*", default=[5, 10, 20])
    slot_attribution_parser = subparsers.add_parser("slot-attribution", help="Compare active-slot candidate selection methods on historical scan snapshots.")
    slot_attribution_parser.add_argument("--horizon", type=int, default=10)
    subparsers.add_parser("monitor", help="Run the intraday monitor and send a consolidated digest.")
    return parser


def main(argv: list[str] | None = None) -> int:
    from src.settings import get_settings

    settings = get_settings()
    configure_logging(settings.paths.logs_dir)
    logger = __import__("logging").getLogger("swingquant.cli")

    parser = build_parser()
    args = parser.parse_args(argv)
    db_manager = DatabaseManager(settings.paths)

    try:
        if args.command == "init-db":
            db_manager.initialize()
            print("Schemas initialized.")
            return 0

        if args.command == "sync":
            report = SyncService(db_manager).run()
            print(
                "Universe synced:",
                f"tickers={report.universe_size}",
                f"rows={report.inserted_rows}",
                f"failed={len(report.failed_tickers)}",
                f"illiquid={len(report.inactive_for_liquidity)}",
            )
            return 0

        if args.command == "refresh-universe":
            report = RefreshUniverseService(db_manager).run()
            print(f"Universe metadata refreshed: rows={report.refreshed_rows}")
            return 0

        if args.command == "research":
            result = ResearchService(db_manager).run()
            print(result.render_console_table())
            print(f"Train rows: {result.train_rows}")
            print(f"Validation rows: {result.validation_rows}")
            return 0

        if args.command == "alpha-research":
            report = AlphaResearchService(db_manager).run(
                top=args.top,
                sector=args.sector,
                horizon_days=args.horizon_days,
                benchmark=args.benchmark,
            )
            print(
                f"Alpha research written to {report.output_path} "
                f"(train_rows={report.train_rows}, validation_rows={report.validation_rows})"
            )
            return 0

        if args.command == "sweep":
            report = SweepService(db_manager).run(mode=args.mode)
            print(
                f"Sweep completed: mode={args.mode} combinations={report.combinations} "
                f"inserted_results={report.inserted_results}"
            )
            return 0

        if args.command == "evaluate":
            report = EvaluateService(db_manager).run(
                top=args.top,
                sector=args.sector,
                run_id=args.run_id,
                min_trades=args.min_trades,
                walk_forward=args.walk_forward,
                walk_forward_windows=args.walk_forward_windows,
                walk_forward_shortlist=args.walk_forward_shortlist,
            )
            print(f"Candidates report written to {report.output_path} with {report.rows_written} rows.")
            return 0

        if args.command == "sleeve-research":
            report = SleeveResearchService(db_manager).run(
                top=args.top,
                sectors=args.sector,
                walk_forward=args.walk_forward,
                walk_forward_windows=args.walk_forward_windows,
                walk_forward_shortlist=args.walk_forward_shortlist,
            )
            print(f"Sleeve research written to {report.output_path} with {report.configs_evaluated} configurations.")
            return 0

        if args.command == "promote":
            print(PromoteService(db_manager).run(row_id=args.row_id, slot=args.slot))
            return 0

        if args.command == "trade":
            service = TradeService(db_manager)
            if args.action == "buy":
                print(service.buy(ticker=args.ticker, price=args.price, shares=args.shares, strategy_slot=args.strategy_slot))
            else:
                print(service.sell(ticker=args.ticker, price=args.price))
            return 0

        if args.command == "positions":
            report = PositionsService(db_manager).run()
            print(report.render_console())
            return 0

        if args.command == "quote":
            report = QuoteService(db_manager).run(ticker=args.ticker)
            print(report.render_console())
            return 0

        if args.command == "scan":
            report = ScanService(db_manager).run()
            ranker_state = "enabled" if report.learned_ranker_enabled else "disabled"
            ranker_reason = f" reason={report.learned_ranker_reason}" if report.learned_ranker_reason else ""
            print(
                "Scan completed:",
                f"candidates={report.candidate_count}",
                f"emailed={report.emailed}",
                f"learned_ranker={ranker_state}",
                f"train_rows={report.learned_ranker_train_rows}",
                f"train_dates={report.learned_ranker_train_dates}{ranker_reason}",
            )
            return 0

        if args.command == "analyst-snapshot":
            report = AnalystSnapshotService(db_manager).run(
                snapshot_date=args.snapshot_date,
                source=args.source,
                top=args.top,
                tickers=args.ticker or [],
            )
            print(
                f"Analyst snapshot written to {report.output_path} "
                f"(date={report.snapshot_date}, provider={report.provider}, source={report.source}, "
                f"rows={report.persisted_rows}, revision_rows={report.persisted_revision_rows}, "
                f"with_targets={report.rows_with_targets}, with_estimates={report.rows_with_estimates})"
            )
            return 0

        if args.command == "scan-backfill":
            report = ScanBackfillService(db_manager).run(
                date_from=args.date_from,
                date_to=args.date_to,
                skip_existing=args.skip_existing,
            )
            print(
                "Scan backfill completed:",
                f"processed={report.scan_dates_processed}",
                f"skipped={report.scan_dates_skipped}",
                f"candidates={report.total_candidates}",
                f"selected={report.total_selected}",
            )
            return 0

        if args.command == "scan-performance":
            report = ScanPerformanceService(db_manager).run(
                recent_scan_dates=args.recent_scan_dates,
                recent_picks=args.recent_picks,
                benchmark=args.benchmark,
                selection_source=args.selection_source,
                model_name=args.model_name,
                model_generated_at=args.model_generated_at,
                latest_model_only=not args.all_sources,
                email=args.email,
            )
            print(
                f"Scan performance written to {report.output_path} "
                f"(selected_rows={report.selected_rows}, scan_dates={report.scan_dates}, benchmark={report.benchmark})"
            )
            return 0

        if args.command == "portfolio-rotation":
            report = PortfolioRotationService(db_manager).run(
                target_positions=args.target_positions,
                max_hold_days=args.max_hold_days,
                min_pre_opportunity=args.min_pre_opportunity,
                min_model_alpha=args.min_model_alpha,
                initial_equity=args.initial_equity,
                walk_forward=args.walk_forward,
                horizon_days=args.horizon_days,
                generated_at=args.model_generated_at,
                model_name=args.model_name,
                eligible_universe_mode=args.eligible_universe_mode,
                model_scope=args.model_scope,
                transaction_cost_bps=args.transaction_cost_bps,
                slippage_bps=args.slippage_bps,
                cooldown_days=args.cooldown_days,
                reinvest_gains=not args.no_reinvest_gains,
                max_new_entries_per_scan=args.max_new_entries_per_scan,
                date_from=args.date_from,
                date_to=args.date_to,
            )
            print(
                f"Portfolio rotation report written to {report.output_path} "
                f"(scan_dates={report.scan_dates}, trades={report.trades}, total_return={report.total_return:.2%}, max_drawdown={report.max_drawdown:.2%})"
            )
            return 0

        if args.command == "universe-backfill":
            report = UniverseSnapshotBackfillService(db_manager).run(
                date_from=args.date_from,
                date_to=args.date_to,
                skip_existing=args.skip_existing,
            )
            print(
                "Universe backfill completed:",
                f"processed={report.snapshot_dates_processed}",
                f"skipped={report.snapshot_dates_skipped}",
                f"rows={report.total_rows}",
            )
            return 0

        if args.command == "universe-analysis":
            report = UniverseAnalysisService(db_manager).run(
                top=args.top,
                horizon_days=args.horizon_days,
                recent_dates=args.recent_dates,
            )
            print(
                f"Universe analysis written to {report.output_path} "
                f"(rows={report.snapshot_rows}, dates={report.snapshot_dates}, horizon_days={report.horizon_days})"
            )
            return 0

        if args.command == "factor-tearsheet":
            report = FactorTearsheetService(db_manager).run(
                sector=args.sector,
                horizon_days=args.horizon,
            )
            print(
                f"Factor tearsheet written to {report.output_path} "
                f"(sector={report.sector}, horizon_days={report.horizon_days}, factors={report.factor_count}, dates={report.snapshot_dates})"
            )
            return 0

        if args.command == "shortlist-bakeoff":
            report = ShortlistBakeoffService(db_manager).run(
                top_n=args.top,
                horizon_days=args.horizon,
                recent_dates=args.recent_dates,
                eligible_universe_mode=args.eligible_universe_mode,
            )
            print(
                f"Shortlist bakeoff written to {report.output_path} "
                f"(target_column={report.target_column}, eligible_rows={report.eligible_rows}, test_dates={report.test_dates})"
            )
            return 0

        if args.command == "shortlist-model":
            report = ShortlistModelService(db_manager).run(
                top_n=args.top,
                horizon_days=args.horizon,
                min_train_dates=args.min_train_dates,
                test_window_dates=args.test_window_dates,
                recent_dates=args.recent_dates,
                eligible_universe_mode=args.eligible_universe_mode,
                model_scope=args.model_scope,
                xgboost_config=args.xgboost_config,
            )
            print(
                f"Shortlist model written to {report.output_path} "
                f"(target_column={report.target_column}, champion_model={report.champion_model}, oos_dates={report.oos_dates}, live_candidates={report.live_candidates})"
            )
            return 0

        if args.command == "shortlist-scoreboard":
            report = ShortlistScoreboardService(db_manager).run(
                top_n=args.top,
                horizon_days=args.horizon,
                min_train_dates=args.min_train_dates,
                test_window_dates=args.test_window_dates,
                recent_dates=args.recent_dates,
                refresh_if_stale=not args.no_refresh_if_stale,
                eligible_universe_mode=args.eligible_universe_mode,
                model_scope=args.model_scope,
            )
            print(
                f"Shortlist scoreboard written to {report.output_path} "
                f"(recommended_model={report.recommended_model}, production_model={report.production_model}, "
                f"run_champion_model={report.run_champion_model}, models={report.model_count})"
            )
            return 0

        if args.command == "shortlist":
            report = ShortlistService(db_manager).run(
                top_n=args.top,
                horizon_days=args.horizon,
                min_train_dates=args.min_train_dates,
                test_window_dates=args.test_window_dates,
                recent_dates=args.recent_dates,
                refresh_if_stale=not args.no_refresh_if_stale,
                eligible_universe_mode=args.eligible_universe_mode,
                model_scope=args.model_scope,
            )
            print(
                f"Shortlist written to {report.output_path} "
                f"(champion_model={report.champion_model}, live_snapshot_date={report.live_snapshot_date}, candidates={report.candidate_count})"
            )
            return 0

        if args.command == "shortlist-allocation-analysis":
            report = ShortlistAllocationAnalysisService(db_manager).run(
                top_n=args.top,
                horizon_days=args.horizon,
                min_train_dates=args.min_train_dates,
                test_window_dates=args.test_window_dates,
                recent_dates=args.recent_dates,
                refresh_if_stale=not args.no_refresh_if_stale,
                eligible_universe_mode=args.eligible_universe_mode,
                model_scope=args.model_scope,
                model_name=args.model_name,
                compare_scopes=args.compare_scopes,
            )
            print(
                f"Shortlist allocation analysis written to {report.output_path} "
                f"(model_name={report.model_name}, scopes={report.scope_count}, policies={report.policy_count})"
            )
            return 0

        if args.command == "shortlist-sector-reactivation":
            report = ShortlistSectorReactivationService(db_manager).run(
                top_n=args.top,
                horizon_days=args.horizon,
                min_train_dates=args.min_train_dates,
                test_window_dates=args.test_window_dates,
                recent_dates=args.recent_dates,
                refresh_if_stale=not args.no_refresh_if_stale,
                eligible_universe_mode=args.eligible_universe_mode,
                model_scope=args.model_scope,
                model_name=args.model_name,
                xgboost_config=args.xgboost_config,
                candidate_sectors=tuple(args.candidate_sector),
            )
            print(
                f"Shortlist sector reactivation analysis written to {report.output_path} "
                f"(model_name={report.selected_model_name}, active_sectors={report.active_sector_count}, "
                f"candidate_sectors={report.candidate_sector_count})"
            )
            return 0

        if args.command == "shortlist-promote":
            report = ShortlistPromoteService(db_manager).run(
                model_name=args.model_name,
                eligible_universe_mode=args.eligible_universe_mode,
                model_scope=args.model_scope,
                xgboost_config=args.xgboost_config,
                horizon_days=args.horizon,
            )
            print(
                f"Shortlist production config updated in {report.config_path} "
                f"(model_name={report.production_model_name}, "
                f"eligible_universe_mode={report.production_eligible_universe_mode}, "
                f"model_scope={report.production_model_scope}, "
                f"xgboost_config={report.production_xgboost_config})"
            )
            return 0

        if args.command == "shortlist-tune":
            report = ShortlistTuneService(db_manager).run(
                top_n=args.top,
                horizon_days=args.horizon,
                min_train_dates=args.min_train_dates,
                test_window_dates=args.test_window_dates,
                recent_dates=args.recent_dates,
                eligible_universe_mode=args.eligible_universe_mode,
                model_scope=args.model_scope,
            )
            print(
                f"Shortlist tune report written to {report.output_path} "
                f"(tuned_candidate={report.tuned_candidate}, ablations={report.ablation_count})"
            )
            return 0

        if args.command == "exit-analysis":
            report = ExitAnalysisService(db_manager).run(horizons=tuple(args.horizons))
            print(
                f"Exit analysis written to {report.output_path} "
                f"(closed_trades={report.closed_trade_count}, linked_trades={report.linked_trade_count}, analyzed_trades={report.analyzed_trade_count})"
            )
            return 0

        if args.command == "rsi-exit-bakeoff":
            report = RsiExitBakeoffService(db_manager).run(
                recent_scan_dates=args.recent_scan_dates,
                benchmark=args.benchmark,
            )
            print(
                f"RSI exit bakeoff written to {report.output_path} "
                f"(selected_rows={report.selected_rows}, mature_rows={report.mature_rows}, scan_dates={report.scan_dates})"
            )
            return 0

        if args.command == "subindustry-attribution":
            report = SubindustryAttributionService(db_manager).run(
                sector=args.sector,
                run_id=args.run_id,
                strategy_id=args.strategy_id,
                min_trades=args.min_trades,
            )
            print(
                f"Subindustry attribution written to {report.output_path} "
                f"(strategy_id={report.strategy_id}, subindustries={report.subindustries_written})"
            )
            return 0

        if args.command == "scan-analysis":
            report = ScanAnalysisService(db_manager).run(
                scan_date=args.date,
                refresh=args.refresh,
                horizons=tuple(args.horizons),
            )
            print(
                f"Scan analysis written to {report.output_path} "
                f"(scan_date={report.scan_date}, candidates={report.candidate_count}, selected={report.selected_count}, refreshed={report.refreshed})"
            )
            return 0

        if args.command == "slot-attribution":
            report = SlotAttributionService(db_manager).run(horizon_days=args.horizon)
            print(
                f"Slot attribution written to {report.output_path} "
                f"(target_column={report.target_column}, slots={report.slot_count}, validation_dates={report.scan_dates})"
            )
            return 0

        if args.command == "monitor":
            report = MonitorService(db_manager).run()
            print(
                f"Monitor completed: watchlist={report.watchlist_size} "
                f"triggered={report.triggered_count} emailed={report.emailed}"
            )
            return 0

        parser.error(f"Unsupported command: {args.command}")
        return 2
    except Exception as exc:
        logger.exception("Command failed: %s", args.command)
        if args.command == "scan":
            try:
                send_html_email(
                    subject="SwingQuant Evening Brief Failed",
                    html_body=(
                        "<html><body>"
                        "<h1>Evening Brief Failed</h1>"
                        "<p>The nightly scan did not complete, so no picks were sent.</p>"
                        f"<p><strong>Error:</strong> {escape(str(exc))}</p>"
                        "</body></html>"
                    ),
                    settings=settings,
                )
            except Exception as email_exc:
                logger.exception("Unable to send scan failure email.")
                print(f"Warning: failed to send scan failure email: {email_exc}", file=sys.stderr)
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
