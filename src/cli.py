from __future__ import annotations

import argparse

from src.evaluate.service import EvaluateService
from src.monitor.service import MonitorService
from src.promote.service import PromoteService
from src.research.service import ResearchService
from src.scan.service import ScanService
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
    subparsers.add_parser("sweep", help="Run the sweep grid and persist backtest results.")

    evaluate_parser = subparsers.add_parser("evaluate", help="Rank backtest results and write candidates.md.")
    evaluate_parser.add_argument("--top", type=int, default=10)
    evaluate_parser.add_argument("--sector", type=str, default=None)

    promote_parser = subparsers.add_parser("promote", help="Promote a backtest result to production.")
    promote_parser.add_argument("--id", dest="row_id", type=int, required=True)

    trade_parser = subparsers.add_parser("trade", help="Record buy or sell trades in the ledger.")
    trade_parser.add_argument("action", choices=("buy", "sell"))
    trade_parser.add_argument("ticker")
    trade_parser.add_argument("price", type=float)
    trade_parser.add_argument("shares", type=int, nargs="?")

    subparsers.add_parser("scan", help="Run the daily signal scan and email the evening brief.")
    subparsers.add_parser("monitor", help="Run the intraday monitor and send a consolidated digest.")
    return parser


def main(argv: list[str] | None = None) -> int:
    from src.settings import get_settings

    settings = get_settings()
    configure_logging(settings.paths.logs_dir)

    parser = build_parser()
    args = parser.parse_args(argv)
    db_manager = DatabaseManager(settings.paths)

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

    if args.command == "sweep":
        report = SweepService(db_manager).run()
        print(f"Sweep completed: combinations={report.combinations} inserted_results={report.inserted_results}")
        return 0

    if args.command == "evaluate":
        report = EvaluateService(db_manager).run(top=args.top, sector=args.sector)
        print(f"Candidates report written to {report.output_path} with {report.rows_written} rows.")
        return 0

    if args.command == "promote":
        print(PromoteService(db_manager).run(row_id=args.row_id))
        return 0

    if args.command == "trade":
        service = TradeService(db_manager)
        if args.action == "buy":
            print(service.buy(ticker=args.ticker, price=args.price, shares=args.shares))
        else:
            print(service.sell(ticker=args.ticker, price=args.price))
        return 0

    if args.command == "scan":
        report = ScanService(db_manager).run()
        print(f"Scan completed: candidates={report.candidate_count} emailed={report.emailed}")
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
