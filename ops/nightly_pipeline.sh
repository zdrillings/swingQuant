#!/usr/bin/env bash
set -euo pipefail

cd /home/zdrillings/code/SwingQuant

run_date="$(date +%F)"

notify_failure() {
  local exit_code="$?"
  local failed_command="${BASH_COMMAND}"
  python3 - "${exit_code}" "${failed_command}" <<'PY'
from html import escape
import sys

from src.settings import get_settings
from src.utils.emailer import send_html_email

exit_code = sys.argv[1]
failed_command = sys.argv[2]
send_html_email(
    subject="SwingQuant Nightly Pipeline Failed",
    html_body=(
        "<html><body>"
        "<h1>Nightly Pipeline Failed</h1>"
        "<p>The ordered nightly refresh did not complete, so downstream scan output may be missing or stale.</p>"
        f"<p><strong>Exit code:</strong> {escape(exit_code)}</p>"
        f"<p><strong>Failed command:</strong> <code>{escape(failed_command)}</code></p>"
        "</body></html>"
    ),
    settings=get_settings(),
)
PY
}

trap notify_failure ERR

echo "[$(date --iso-8601=seconds)] nightly pipeline start run_date=${run_date}"

echo "[$(date --iso-8601=seconds)] sync"
./sq sync

echo "[$(date --iso-8601=seconds)] universe-backfill ${run_date}"
./sq universe-backfill --date-from "${run_date}" --date-to "${run_date}" --skip-existing

echo "[$(date --iso-8601=seconds)] shortlist-model"
./sq shortlist-model \
  --top 10 \
  --horizon 20 \
  --min-train-dates 252 \
  --test-window-dates 20 \
  --recent-dates 60 \
  --eligible-universe-mode passed_or_trend \
  --model-scope sector_specific \
  --xgboost-config balanced_depth4

echo "[$(date --iso-8601=seconds)] analyst-snapshot"
./sq analyst-snapshot --source research --top 250

echo "[$(date --iso-8601=seconds)] extended-hours-snapshot"
./sq extended-hours-snapshot --source all

echo "[$(date --iso-8601=seconds)] scan"
./sq scan

echo "[$(date --iso-8601=seconds)] scan-performance"
./sq scan-performance --email

echo "[$(date --iso-8601=seconds)] nightly pipeline complete"
