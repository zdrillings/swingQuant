from __future__ import annotations

import argparse
import sys

from src.research.alpha_service import AlphaResearchService
from src.evaluate.service import EvaluateService
from src.monitor.service import MonitorService
from src.positions.service import PositionsService
from src.promote.service import PromoteService
from src.research.service import ResearchService
from src.scan.analysis_service import ScanAnalysisService
from src.scan.service import ScanService
from src.sleeves.service import SleeveResearchService
from src.sweep.service import SweepService
from src.sync.service import SyncService
from src.trade.service import TradeService
from src.utils.db_manager import DatabaseManager
from src.utils.logging import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sq")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("init-db", help="Initialize DuckDB and SQLite schemas.")
    subparsers.add_parser("sync", help="Bootstrap the universe and sync OHLCV history.")
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
    subparsers.add_parser("scan", help="Run the daily signal scan and email the evening brief.")
    scan_analysis_parser = subparsers.add_parser("scan-analysis", help="Analyze scan snapshots and selection attribution.")
    scan_analysis_parser.add_argument("--date", type=str, default=None)
    scan_analysis_parser.add_argument("--refresh", action="store_true")
    scan_analysis_parser.add_argument("--horizons", type=int, nargs="*", default=[5, 10, 20])
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

        if args.command == "scan":
            report = ScanService(db_manager).run()
            print(f"Scan completed: candidates={report.candidate_count} emailed={report.emailed}")
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
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
