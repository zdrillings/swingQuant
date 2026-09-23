#!/usr/bin/env bash
set -euo pipefail

cd /home/zdrillings/code/SwingQuant

run_date="$(date +%F)"
universe_refresh_start="$(date -d "${run_date} - 120 days" +%F)"
pipeline_lock_file="data/nightly_pipeline.lock"

send_failure_email() {
  local exit_code="$1"
  local failed_command="$2"
  local subject="${3:-SwingQuant Nightly Pipeline Failed}"
  local heading="${4:-Nightly Pipeline Failed}"
  local message="${5:-The ordered nightly refresh did not complete, so downstream scan output may be missing or stale.}"
  python3 - "${exit_code}" "${failed_command}" "${subject}" "${heading}" "${message}" <<'PY'
from html import escape
import sys

from src.settings import get_settings
from src.utils.emailer import send_html_email

exit_code = sys.argv[1]
failed_command = sys.argv[2]
subject = sys.argv[3]
heading = sys.argv[4]
message = sys.argv[5]
send_html_email(
    subject=subject,
    html_body=(
        "<html><body>"
        f"<h1>{escape(heading)}</h1>"
        f"<p>{escape(message)}</p>"
        f"<p><strong>Exit code:</strong> {escape(exit_code)}</p>"
        f"<p><strong>Failed command:</strong> <code>{escape(failed_command)}</code></p>"
        "</body></html>"
    ),
    settings=get_settings(),
)
PY
}

mkdir -p data
exec 9>"${pipeline_lock_file}"
if ! flock -n 9; then
  echo "Another nightly pipeline run is already active; refusing to overlap." >&2
  send_failure_email \
    3 \
    "nightly pipeline lock" \
    "SwingQuant Nightly Pipeline Already Running" \
    "Nightly Pipeline Already Running" \
    "Another nightly pipeline run is active; refusing to overlap." || true
  exit 3
fi

promotion_failures_file="data/promotion_failures.txt"
last_champion_days_file="data/days_since_last_champion.txt"
shortlist_report_file="reports/shortlist_model.md"

read_shortlist_model_config() {
  PYTHONPATH=.vendor python3 - <<'PY'
from src.settings import load_feature_config

shortlist_model = (
    load_feature_config()
    .get("scan_policy", {})
    .get("shortlist_model", {})
    or {}
)
print(int(shortlist_model.get("horizon_days", 20)), int(shortlist_model.get("top_n", 10)))
PY
}

read -r shortlist_horizon_days shortlist_top_n < <(read_shortlist_model_config)

latest_champion_date() {
  PYTHONPATH=.vendor python3 - <<'PY'
from __future__ import annotations

from datetime import datetime, timezone
import sqlite3

from src.settings import load_feature_config

try:
    config = load_feature_config()
    horizon_days = int(
        config.get("scan_policy", {})
        .get("shortlist_model", {})
        .get("horizon_days", 20)
    )
    conn = sqlite3.connect("file:data/ledger.sqlite?mode=ro", uri=True)
    row = conn.execute(
        """
        SELECT generated_at
        FROM Shortlist_Model_Runs
        WHERE horizon_days = ?
          AND champion_model IS NOT NULL
          AND TRIM(champion_model) != ''
          AND LOWER(TRIM(champion_model)) != 'n/a'
        ORDER BY generated_at DESC
        LIMIT 1
        """,
        (horizon_days,),
    ).fetchone()
    conn.close()
except Exception:
    row = None

if row and row[0]:
    text = str(row[0]).replace("Z", "+00:00")
    try:
        print(datetime.fromisoformat(text).astimezone(timezone.utc).date().isoformat())
    except ValueError:
        pass
PY
}

sync_promotion_failure_state() {
  local champion_date
  local latest_failure_date
  if [[ ! -f "${promotion_failures_file}" ]]; then
    return
  fi
  champion_date="$(latest_champion_date || true)"
  latest_failure_date="$(tail -n 1 "${promotion_failures_file}" || true)"
  if [[ "${champion_date}" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] && [[ "${latest_failure_date}" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] && [[ "${champion_date}" > "${latest_failure_date}" || "${champion_date}" == "${latest_failure_date}" ]]; then
    clear_promotion_failures
  fi
}

read_days_since_last_champion() {
  if [[ -f "${shortlist_report_file}" ]]; then
    grep -E '^- days_since_last_champion:' "${shortlist_report_file}" | tail -n 1 | awk '{print $3}' || true
  fi
}

record_promotion_failure() {
  mkdir -p data
  sync_promotion_failure_state
  if [[ ! -f "${promotion_failures_file}" ]] || ! grep -Fxq "${run_date}" "${promotion_failures_file}"; then
    printf '%s\n' "${run_date}" >> "${promotion_failures_file}"
  fi
  recorded_failures="$(tail -n 30 "${promotion_failures_file}" | wc -l)"
  days_since_last_champion=0
  if [[ -f "${last_champion_days_file}" ]]; then
    days_since_last_champion="$(tr -cd '0-9' < "${last_champion_days_file}")"
  fi
  if [[ -z "${days_since_last_champion}" ]]; then
    days_since_last_champion=0
  fi
  if [[ "${days_since_last_champion}" -gt "${recorded_failures}" ]]; then
    echo "${days_since_last_champion}"
  else
    echo "${recorded_failures}"
  fi
}

clear_promotion_failures() {
  rm -f "${promotion_failures_file}"
  rm -f "${last_champion_days_file}"
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

echo "[$(date --iso-8601=seconds)] analyst-snapshot"
./sq analyst-snapshot --source research --top 250

echo "[$(date --iso-8601=seconds)] universe-backfill ${universe_refresh_start}..${run_date}"
./sq universe-backfill --date-from "${universe_refresh_start}" --date-to "${run_date}" --skip-existing

echo "[$(date --iso-8601=seconds)] regime-meter latest"
./sq regime-meter --latest || echo "[$(date --iso-8601=seconds)] regime-meter latest failed; continuing"

echo "[$(date --iso-8601=seconds)] regime-meter report"
./sq regime-meter --report --email || echo "[$(date --iso-8601=seconds)] regime-meter report failed; continuing"

echo "[$(date --iso-8601=seconds)] path-label-tearsheet"
./sq path-label-tearsheet --horizon 20

echo "[$(date --iso-8601=seconds)] shortlist-model path-target dry-run"
path_shortlist_log="$(mktemp)"
trap - ERR
set +e
./sq shortlist-model \
  --horizon "${shortlist_horizon_days}" \
  --top "${shortlist_top_n}" \
  --target-type path \
  --min-train-dates 252 \
  --max-train-dates 252 \
  --test-window-dates 20 \
  --oos-stride-dates 20 \
  --recent-dates 60 \
  --eligible-universe-mode passed_or_trend \
  --xgboost-config balanced_depth4 \
  --dry-run 2>&1 | tee "${path_shortlist_log}"
path_shortlist_status="${PIPESTATUS[0]}"
set -e
trap notify_failure ERR
if [[ "${path_shortlist_status}" -ne 0 ]]; then
  echo "[$(date --iso-8601=seconds)] path-target dry-run failed status=${path_shortlist_status}; continuing production shortlist flow" >&2
fi
rm -f "${path_shortlist_log}"

echo "[$(date --iso-8601=seconds)] shortlist-model"
shortlist_log="$(mktemp)"
shortlist_promotion_failed=0
trap - ERR
set +e
./sq shortlist-model \
  --horizon "${shortlist_horizon_days}" \
  --top "${shortlist_top_n}" \
  --min-train-dates 252 \
  --max-train-dates 252 \
  --test-window-dates 20 \
  --oos-stride-dates 20 \
  --recent-dates 60 \
  --eligible-universe-mode passed_or_trend \
  --xgboost-config balanced_depth4 2>&1 | tee "${shortlist_log}"
shortlist_status="${PIPESTATUS[0]}"
set -e
trap notify_failure ERR
if [[ "${shortlist_status}" -ne 0 ]]; then
  if grep -Fq "No shortlist model candidate passed the promotion gate" "${shortlist_log}"; then
    shortlist_promotion_failed=1
    recorded_promotion_failures="$(record_promotion_failure)"
    days_since_last_champion="$(read_days_since_last_champion)"
    if [[ "${days_since_last_champion}" =~ ^[0-9]+$ ]]; then
      printf '%s\n' "${days_since_last_champion}" > "${last_champion_days_file}"
    else
      days_since_last_champion="${recorded_promotion_failures}"
    fi
    echo "[$(date --iso-8601=seconds)] shortlist-model produced no promotable champion; scan will be skipped"
    if send_failure_email \
      0 \
      "scan skipped; ${days_since_last_champion} days since the last promoted champion" \
      "SwingQuant scan skipped - no promotable shortlist champion" \
      "Scan Skipped" \
      "The shortlist model promotion gate failed tonight. It has been ${days_since_last_champion} days since the last promoted champion. Recorded promotion-failure nights: ${recorded_promotion_failures}. Scan will be skipped until a champion is promoted."; then
      echo "[$(date --iso-8601=seconds)] scan-skip email sent"
    else
      echo "[$(date --iso-8601=seconds)] scan-skip email failed" >&2
    fi
  else
    rm -f "${shortlist_log}"
    exit "${shortlist_status}"
  fi
else
  clear_promotion_failures
fi
rm -f "${shortlist_log}"

echo "[$(date --iso-8601=seconds)] extended-hours-snapshot"
./sq extended-hours-snapshot --source all

if [[ "${shortlist_promotion_failed}" -eq 0 ]]; then
  echo "[$(date --iso-8601=seconds)] scan"
  ./sq scan
else
  echo "[$(date --iso-8601=seconds)] scan skipped because shortlist-model produced no promotable champion"
fi

echo "[$(date --iso-8601=seconds)] phase2-research"
./sq phase2-research --horizon 20 --top 2 --trial-count 200

echo "[$(date --iso-8601=seconds)] scan-performance"
./sq scan-performance --all-sources --email

echo "[$(date --iso-8601=seconds)] nightly pipeline complete"
