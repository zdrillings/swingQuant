from __future__ import annotations

import argparse

from src.research.service import ResearchService
from src.sync.service import SyncService
from src.utils.db_manager import DatabaseManager
from src.utils.logging import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sq")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("init-db", help="Initialize DuckDB and SQLite schemas.")
    subparsers.add_parser("sync", help="Bootstrap the universe and sync OHLCV history.")
    subparsers.add_parser("research", help="Run the research model and print feature importance.")
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

    parser.error(f"Unsupported command: {args.command}")
    return 2
