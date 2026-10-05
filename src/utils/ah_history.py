from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re

import pandas as pd


AH_HISTORY_SCHEMA = """
CREATE TABLE IF NOT EXISTS ah_snapshot_history (
    ticker VARCHAR NOT NULL,
    snapshot_date DATE NOT NULL,
    ah_price DOUBLE,
    ah_volume BIGINT,
    rth_close DOUBLE,
    source_row_count INTEGER,
    captured_at VARCHAR,
    PRIMARY KEY (ticker, snapshot_date)
);
CREATE INDEX IF NOT EXISTS idx_ah_snapshot_history_date
ON ah_snapshot_history (snapshot_date);
"""


@dataclass(frozen=True)
class AhHistoryPersistResult:
    db_rows_read: int
    markdown_reports_read: int
    markdown_rows_read: int
    rows_upserted: int
    total_rows: int
    total_nights: int
    volume_coverage_tickers: int
    output_path: Path


def read_extended_hours_snapshot_rows(
    *,
    market_duckdb_path: Path,
    snapshot_date: str | None = None,
) -> pd.DataFrame:
    import duckdb

    columns = [
        "ticker",
        "snapshot_date",
        "extended_price AS ah_price",
        "extended_volume AS ah_volume",
        "regular_close AS rth_close",
        "captured_at",
    ]
    params: list[object] = []
    query = f"SELECT {', '.join(columns)} FROM extended_hours_snapshots"
    if snapshot_date is not None:
        query += " WHERE snapshot_date = ?"
        params.append(snapshot_date)
    query += " ORDER BY snapshot_date ASC, ticker ASC"
    try:
        with duckdb.connect(str(market_duckdb_path), read_only=True) as connection:
            frame = connection.execute(query, params).df()
    except duckdb.CatalogException:
        return _empty_history_frame()
    except duckdb.IOException:
        return _empty_history_frame()
    if frame.empty:
        return _empty_history_frame()
    counts = frame.groupby("snapshot_date")["ticker"].transform("count")
    frame["source_row_count"] = counts.astype(int)
    return _normalize_history_frame(frame)


def read_markdown_snapshot_rows(reports_dir: Path) -> tuple[pd.DataFrame, int]:
    frames: list[pd.DataFrame] = []
    reports_read = 0
    for path in sorted(reports_dir.glob("extended_hours_snapshots*.md")):
        frame = _parse_markdown_snapshot_report(path)
        if not frame.empty:
            reports_read += 1
            frames.append(frame)
    if not frames:
        return _empty_history_frame(), 0
    return _normalize_history_frame(pd.concat(frames, ignore_index=True)), reports_read


def upsert_ah_history(*, ah_duckdb_path: Path, rows: pd.DataFrame) -> int:
    import duckdb

    normalized = _normalize_history_frame(rows)
    ah_duckdb_path.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(ah_duckdb_path)) as connection:
        connection.execute(AH_HISTORY_SCHEMA)
        if normalized.empty:
            return 0
        connection.register("incoming_ah_history", normalized)
        connection.execute(
            """
            INSERT INTO ah_snapshot_history (
                ticker,
                snapshot_date,
                ah_price,
                ah_volume,
                rth_close,
                source_row_count,
                captured_at
            )
            SELECT
                ticker,
                snapshot_date,
                ah_price,
                ah_volume,
                rth_close,
                source_row_count,
                captured_at
            FROM incoming_ah_history
            ON CONFLICT (ticker, snapshot_date) DO UPDATE SET
                ah_price = COALESCE(EXCLUDED.ah_price, ah_snapshot_history.ah_price),
                ah_volume = COALESCE(EXCLUDED.ah_volume, ah_snapshot_history.ah_volume),
                rth_close = COALESCE(EXCLUDED.rth_close, ah_snapshot_history.rth_close),
                source_row_count = COALESCE(EXCLUDED.source_row_count, ah_snapshot_history.source_row_count),
                captured_at = COALESCE(EXCLUDED.captured_at, ah_snapshot_history.captured_at)
            """
        )
    return len(normalized)


def persist_ah_history(
    *,
    market_duckdb_path: Path,
    ah_duckdb_path: Path,
    reports_dir: Path,
    output_path: Path,
    snapshot_date: str | None = None,
) -> AhHistoryPersistResult:
    db_rows = read_extended_hours_snapshot_rows(
        market_duckdb_path=market_duckdb_path,
        snapshot_date=snapshot_date,
    )
    markdown_rows, markdown_reports_read = read_markdown_snapshot_rows(reports_dir)
    combined = _prefer_complete_rows(db_rows=db_rows, markdown_rows=markdown_rows)
    rows_upserted = upsert_ah_history(ah_duckdb_path=ah_duckdb_path, rows=combined)
    total_rows, total_nights, coverage_tickers = summarize_ah_history(ah_duckdb_path)
    result = AhHistoryPersistResult(
        db_rows_read=len(db_rows),
        markdown_reports_read=markdown_reports_read,
        markdown_rows_read=len(markdown_rows),
        rows_upserted=rows_upserted,
        total_rows=total_rows,
        total_nights=total_nights,
        volume_coverage_tickers=coverage_tickers,
        output_path=output_path,
    )
    write_ah_history_report(
        result=result,
        ah_duckdb_path=ah_duckdb_path,
        market_duckdb_path=market_duckdb_path,
    )
    return result


def summarize_ah_history(ah_duckdb_path: Path) -> tuple[int, int, int]:
    import duckdb

    if not ah_duckdb_path.exists():
        return 0, 0, 0
    with duckdb.connect(str(ah_duckdb_path), read_only=True) as connection:
        total_rows, total_nights = connection.execute(
            """
            SELECT
                COUNT(*)::INTEGER AS total_rows,
                COUNT(DISTINCT snapshot_date)::INTEGER AS total_nights
            FROM ah_snapshot_history
            """
        ).fetchone()
        coverage_tickers = connection.execute(
            """
            WITH night_count AS (
                SELECT COUNT(DISTINCT snapshot_date)::DOUBLE AS nights
                FROM ah_snapshot_history
            ),
            ticker_counts AS (
                SELECT ticker, SUM(CASE WHEN COALESCE(ah_volume, 0) > 0 THEN 1 ELSE 0 END)::DOUBLE AS nonzero_nights
                FROM ah_snapshot_history
                GROUP BY ticker
            )
            SELECT COUNT(*)::INTEGER
            FROM ticker_counts, night_count
            WHERE nights > 0 AND nonzero_nights >= CEIL(nights * 0.80)
            """
        ).fetchone()[0]
    return int(total_rows or 0), int(total_nights or 0), int(coverage_tickers or 0)


def write_ah_history_report(
    *,
    result: AhHistoryPersistResult,
    ah_duckdb_path: Path,
    market_duckdb_path: Path,
) -> None:
    result.output_path.parent.mkdir(parents=True, exist_ok=True)
    first_date, last_date = _history_date_range(ah_duckdb_path)
    lines = [
        "# After-Hours History Setup",
        "",
        "## Schema",
        "",
        f"- history_db: {ah_duckdb_path}",
        "- table: ah_snapshot_history",
        "- columns: ticker, snapshot_date, ah_price, ah_volume, rth_close, source_row_count, captured_at",
        "- primary_key: (ticker, snapshot_date)",
        "",
        "## Ingestion",
        "",
        f"- source_snapshot_db: {market_duckdb_path} (read-only)",
        f"- db_rows_read: {result.db_rows_read}",
        f"- markdown_reports_recoverable: {result.markdown_reports_read}",
        f"- markdown_rows_recoverable: {result.markdown_rows_read}",
        f"- rows_upserted_this_run: {result.rows_upserted}",
        f"- history_total_rows: {result.total_rows}",
        f"- history_total_nights: {result.total_nights}",
        f"- history_first_snapshot_date: {first_date or 'n/a'}",
        f"- history_latest_snapshot_date: {last_date or 'n/a'}",
        "",
        "## Coverage Query",
        "",
        "- question: tickers with non-zero AH volume on at least 80% of stored nights",
        f"- answer: {result.volume_coverage_tickers}",
        "",
        "## Model Effect",
        "",
        "- OOS Spearman delta: n/a",
        "- audit_verdict: data prerequisite only; this unlocks B2/B3/B4/B5 after enough nightly history accumulates.",
        "",
    ]
    result.output_path.write_text("\n".join(lines), encoding="utf-8")


def _history_date_range(ah_duckdb_path: Path) -> tuple[str | None, str | None]:
    import duckdb

    if not ah_duckdb_path.exists():
        return None, None
    with duckdb.connect(str(ah_duckdb_path), read_only=True) as connection:
        row = connection.execute(
            "SELECT MIN(snapshot_date)::VARCHAR, MAX(snapshot_date)::VARCHAR FROM ah_snapshot_history"
        ).fetchone()
    if not row:
        return None, None
    return row[0], row[1]


def _parse_markdown_snapshot_report(path: Path) -> pd.DataFrame:
    text = path.read_text(encoding="utf-8")
    meta = dict(re.findall(r"^- ([a-zA-Z_]+): (.+)$", text, flags=re.MULTILINE))
    snapshot_date = meta.get("snapshot_date")
    if not snapshot_date:
        return _empty_history_frame()
    source_row_count = _parse_int(meta.get("persisted_rows"))
    captured_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
    rows = []
    for ticker, volume in re.findall(
        r"^- ([A-Z0-9.\-]+): .*?volume=([0-9]+|n/a)",
        text,
        flags=re.MULTILINE,
    ):
        rows.append(
            {
                "ticker": ticker,
                "snapshot_date": snapshot_date,
                "ah_price": None,
                "ah_volume": _parse_int(volume),
                "rth_close": None,
                "source_row_count": source_row_count,
                "captured_at": captured_at,
            }
        )
    if not rows:
        return _empty_history_frame()
    return _normalize_history_frame(pd.DataFrame(rows))


def _prefer_complete_rows(*, db_rows: pd.DataFrame, markdown_rows: pd.DataFrame) -> pd.DataFrame:
    frames = [frame for frame in [markdown_rows, db_rows] if not frame.empty]
    if not frames:
        return _empty_history_frame()
    combined = pd.concat(frames, ignore_index=True)
    completeness = combined[["ah_price", "rth_close"]].notna().sum(axis=1)
    combined["_completeness"] = completeness
    combined = combined.sort_values(
        ["snapshot_date", "ticker", "_completeness"],
        ascending=[True, True, True],
    )
    combined = combined.drop_duplicates(["ticker", "snapshot_date"], keep="last")
    combined = combined.drop(columns=["_completeness"])
    return _normalize_history_frame(combined)


def _normalize_history_frame(frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["ticker", "snapshot_date", "ah_price", "ah_volume", "rth_close", "source_row_count", "captured_at"]
    if frame.empty:
        return _empty_history_frame()
    normalized = frame.copy()
    for column in columns:
        if column not in normalized.columns:
            normalized[column] = None
    normalized["ticker"] = normalized["ticker"].astype(str).str.strip().str.upper()
    normalized["snapshot_date"] = pd.to_datetime(normalized["snapshot_date"], errors="coerce").dt.date
    normalized["ah_price"] = pd.to_numeric(normalized["ah_price"], errors="coerce")
    normalized["ah_volume"] = pd.to_numeric(normalized["ah_volume"], errors="coerce").astype("Int64")
    normalized["rth_close"] = pd.to_numeric(normalized["rth_close"], errors="coerce")
    normalized["source_row_count"] = pd.to_numeric(normalized["source_row_count"], errors="coerce").astype("Int64")
    normalized["captured_at"] = normalized["captured_at"].where(pd.notna(normalized["captured_at"]), None)
    normalized = normalized[normalized["ticker"].ne("") & normalized["snapshot_date"].notna()]
    return normalized[columns].reset_index(drop=True)


def _empty_history_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["ticker", "snapshot_date", "ah_price", "ah_volume", "rth_close", "source_row_count", "captured_at"]
    )


def _parse_int(value: object) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None
