#!/usr/bin/env bash
set -euo pipefail

cd /home/zdrillings/code/SwingQuant

run_date="$(date +%F)"
universe_refresh_start="$(date -d "${run_date} - 120 days" +%F)"

send_failure_email() {
  local exit_code="$1"
  local failed_command="$2"
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

notify_failure() {
  local exit_code="$?"
  local failed_command="${BASH_COMMAND}"
  send_failure_email "${exit_code}" "${failed_command}" || true
}

trap notify_failure ERR

if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  guarded_paths=(src tests ops config.yaml production_strategies.json pyproject.toml README.md AGENTS.md)
  if ! git diff --quiet -- "${guarded_paths[@]}" || ! git diff --cached --quiet -- "${guarded_paths[@]}"; then
    echo "Refusing to run nightly pipeline with uncommitted code/config changes." >&2
    echo "Commit or stash changes under: ${guarded_paths[*]}" >&2
    send_failure_email 2 "dirty working tree guard" || true
    exit 2
  fi
fi

echo "[$(date --iso-8601=seconds)] nightly pipeline start run_date=${run_date}"

echo "[$(date --iso-8601=seconds)] sync"
./sq sync

echo "[$(date --iso-8601=seconds)] universe-backfill ${universe_refresh_start}..${run_date}"
./sq universe-backfill --date-from "${universe_refresh_start}" --date-to "${run_date}" --skip-existing

echo "[$(date --iso-8601=seconds)] shortlist-model"
shortlist_log="$(mktemp)"
shortlist_promotion_failed=0
trap - ERR
set +e
./sq shortlist-model \
  --top 10 \
  --horizon 20 \
  --min-train-dates 252 \
  --test-window-dates 20 \
  --recent-dates 60 \
  --eligible-universe-mode passed_or_trend \
  --model-scope sector_specific \
  --xgboost-config balanced_depth4 2>&1 | tee "${shortlist_log}"
shortlist_status="${PIPESTATUS[0]}"
set -e
trap notify_failure ERR
if [[ "${shortlist_status}" -ne 0 ]]; then
  if grep -Fq "No shortlist model candidate passed the promotion gate" "${shortlist_log}"; then
    shortlist_promotion_failed=1
    echo "[$(date --iso-8601=seconds)] shortlist-model produced no promotable champion; continuing with previously persisted model context"
  else
    rm -f "${shortlist_log}"
    exit "${shortlist_status}"
  fi
fi
rm -f "${shortlist_log}"

echo "[$(date --iso-8601=seconds)] analyst-snapshot"
./sq analyst-snapshot --source research --top 250

echo "[$(date --iso-8601=seconds)] extended-hours-snapshot"
./sq extended-hours-snapshot --source all

if [[ "${shortlist_promotion_failed}" -eq 0 ]]; then
  echo "[$(date --iso-8601=seconds)] scan"
  ./sq scan
else
  echo "[$(date --iso-8601=seconds)] scan skipped because shortlist-model produced no promotable champion"
fi

echo "[$(date --iso-8601=seconds)] scan-performance"
./sq scan-performance --all-sources --email

echo "[$(date --iso-8601=seconds)] nightly pipeline complete"
