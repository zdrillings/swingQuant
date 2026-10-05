from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
VENDOR_DIR = ROOT_DIR / ".vendor"
if VENDOR_DIR.exists():
    sys.path.insert(0, str(VENDOR_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.settings import get_settings
from src.utils.ah_history import persist_ah_history


def main() -> int:
    parser = argparse.ArgumentParser(description="Persist replace-by-date extended-hours snapshots into AH history.")
    parser.add_argument("--snapshot-date", default=None)
    parser.add_argument("--ah-db", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    settings = get_settings()
    result = persist_ah_history(
        market_duckdb_path=settings.paths.duckdb_path,
        ah_duckdb_path=args.ah_db or settings.paths.data_dir / "ah_history.duckdb",
        reports_dir=settings.paths.reports_dir,
        output_path=args.report or settings.paths.reports_dir / "ah_history_setup.md",
        snapshot_date=args.snapshot_date,
    )
    print(
        "AH history persisted: "
        f"db_rows={result.db_rows_read} "
        f"markdown_reports={result.markdown_reports_read} "
        f"markdown_rows={result.markdown_rows_read} "
        f"upserted={result.rows_upserted} "
        f"nights={result.total_nights} "
        f"volume_coverage_80pct={result.volume_coverage_tickers}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
