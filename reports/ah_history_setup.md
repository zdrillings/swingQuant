# After-Hours History Setup

## Schema

- history_db: /home/zdrillings/code/SwingQuant/data/ah_history.duckdb
- table: ah_snapshot_history
- columns: ticker, snapshot_date, ah_price, ah_volume, rth_close, source_row_count, captured_at
- primary_key: (ticker, snapshot_date)

## Ingestion

- source_snapshot_db: /home/zdrillings/code/SwingQuant/data/market_data.duckdb (read-only)
- db_rows_read: 255
- markdown_reports_recoverable: 1
- markdown_rows_recoverable: 20
- rows_upserted_this_run: 255
- history_total_rows: 15109
- history_total_nights: 59
- history_first_snapshot_date: 2026-07-03
- history_latest_snapshot_date: 2026-10-05

## Coverage Query

- question: tickers with non-zero AH volume on at least 80% of stored nights
- answer: 24

## Model Effect

- OOS Spearman delta: n/a
- audit_verdict: data prerequisite only; this unlocks B2/B3/B4/B5 after enough nightly history accumulates.
