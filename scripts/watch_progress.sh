#!/usr/bin/env bash
# Live progress tracker for PRISM trace generation.
# Shows per-domain phase, completion %, vLLM throughput, and JSONL output.
#
# Usage:
#   bash scripts/watch_progress.sh [ROUND] [REFRESH_SECS]
#   bash scripts/watch_progress.sh 1       # watch Round 1, refresh every 15s
#   bash scripts/watch_progress.sh 2 30    # watch Round 2, refresh every 30s

source "$(dirname "$0")/setup/env.sh" 2>/dev/null || true
cd "${PRISM_ROOT:-.}"

ROUND="${1:-1}"
REFRESH="${2:-15}"
LOG_DIR="${PRISM_LOG_DIR:-${PRISM_ROOT}/results/logs}"
TRACES_DIR="${PRISM_ROOT}/results/traces"

# Domain totals per round
declare -A DOMAIN_TOTAL
case "$ROUND" in
  1) DOMAINS=(algebra geometry combinatorics number_theory)
     for d in "${DOMAINS[@]}"; do DOMAIN_TOTAL[$d]=2500; done
     DOMAIN_TOTAL[geometry]=727; DOMAIN_TOTAL[combinatorics]=602; DOMAIN_TOTAL[number_theory]=691
     ;;
  2) DOMAINS=(miscellaneous algebra_cv_miscellaneous geometry_cv_algebra combinatorics_cv_number_theory)
     for d in "${DOMAINS[@]}"; do DOMAIN_TOTAL[$d]=2500; done ;;
  3) DOMAINS=(number_theory_cv_algebra miscellaneous_cv_combinatorics algebra_cv_geometry geometry_cv_combinatorics)
     for d in "${DOMAINS[@]}"; do DOMAIN_TOTAL[$d]=2500; done ;;
esac

# Phase totals: guided P1 = N×3 passes, plus free-solve adds same N
# Each phase fires N requests; P1 has N, P2 has N, P3 has N → 3N total HTTP 200s

phase_of() {
  local log="$1"
  local p
  p=$(grep -E "Pass [123] \(" "$log" 2>/dev/null | tail -1 | grep -oP "Pass \d \(\w+\)")
  echo "${p:-Phase 1}"
}

pct_bar() {
  local val=$1 total=$2 width=20
  [ "$total" -eq 0 ] && { printf "[%-${width}s] ???" ""; return; }
  local filled=$(( val * width / total ))
  local bar=""
  for ((i=0; i<filled; i++)); do bar+="█"; done
  for ((i=filled; i<width; i++)); do bar+="░"; done
  local pct=$(( val * 100 / total ))
  printf "[%s] %3d%%" "$bar" "$pct"
}

tput_clear() { command -v tput &>/dev/null && tput clear || printf '\033[2J\033[H'; }

while true; do
  tput_clear
  echo "═══════════════════════════════════════════════════════════════════"
  printf "  PRISM Trace Generation — Round %s          %s\n" "$ROUND" "$(date '+%H:%M:%S')"
  echo "═══════════════════════════════════════════════════════════════════"
  echo ""

  # ── Per-domain status ────────────────────────────────────────────────
  procs=$(pgrep -fc "trace_generator" 2>/dev/null || echo 0)
  printf "  %-35s  %-22s  %s\n" "Domain" "Phase" "Responses"
  echo "  ─────────────────────────────────────────────────────────────"

  total_200=0
  for d in "${DOMAINS[@]}"; do
    log="${LOG_DIR}/traces_r${ROUND}_${d}.log"
    cnt=0
    if [ -f "$log" ]; then
      cnt=$(grep -c "HTTP/1.1 200" "$log" 2>/dev/null || echo 0)
    fi
    total_200=$((total_200 + cnt))
    ph=$(phase_of "$log")
    tot=${DOMAIN_TOTAL[$d]:-2500}

    # JSONL size
    jname="${d%%_cv*}_traces.jsonl"
    jsz=""
    if [ -s "${TRACES_DIR}/${jname}" ]; then
      jsz=" ✓$(wc -l < "${TRACES_DIR}/${jname}" 2>/dev/null) traces"
    fi

    bar=$(pct_bar "$cnt" "$((tot * 3))")   # 3 passes × N
    printf "  %-35s  %-22s  %5d %s%s\n" "$d" "$ph" "$cnt" "$bar" "$jsz"
  done

  echo ""
  echo "  Processes alive: ${procs}"

  # ── vLLM server ──────────────────────────────────────────────────────
  echo ""
  echo "  ── vLLM Server ──────────────────────────────────────────────"
  metrics=$(curl -sf "http://localhost:8000/metrics" 2>/dev/null)
  if [ -n "$metrics" ]; then
    running=$(echo "$metrics" | grep "num_requests_running" | grep -v "#" | grep -oP "[\d.]+" | tail -1)
    waiting=$(echo "$metrics" | grep "num_requests_waiting" | grep -v "#" | grep -oP "[\d.]+" | tail -1)
    kv=$(echo "$metrics" | grep "gpu_cache_usage_perc" | grep -v "#" | grep -oP "[\d.]+" | tail -1)
    tput_line=$(grep "throughput" "${LOG_DIR}/vllm_server.log" 2>/dev/null | tail -1 | grep -oP "generation throughput: [\d.]+" | grep -oP "[\d.]+")
    printf "  Running: %-6s  Waiting: %-6s  KV cache: %-6s%%  Gen: %s tok/s\n" \
           "$running" "$waiting" "${kv:-?}" "${tput_line:-?}"
  else
    echo "  Server not reachable on port 8000"
  fi

  # ── JSONL outputs ────────────────────────────────────────────────────
  echo ""
  echo "  ── Output Files ─────────────────────────────────────────────"
  has_output=0
  for f in "${TRACES_DIR}"/*.jsonl; do
    [ -f "$f" ] || continue
    sz=$(wc -c < "$f" 2>/dev/null || echo 0)
    lines=$(wc -l < "$f" 2>/dev/null || echo 0)
    if [ "$sz" -gt 0 ]; then
      neg=$(python3 -c "
import json, sys
n=0
for l in open('$f'):
    try:
        d=json.loads(l)
        if d.get('free_solve') and not d.get('solve_correct'): n+=1
    except: pass
print(n)
" 2>/dev/null || echo "?")
      printf "  %-45s  %5d traces  %s neg\n" "$(basename $f)" "$lines" "$neg"
      has_output=1
    fi
  done
  [ "$has_output" -eq 0 ] && echo "  (none written yet)"

  # ── Chain status ─────────────────────────────────────────────────────
  chain_alive=$(pgrep -f "wait_and_chain\|rounds_chain" | wc -l)
  echo ""
  echo "  Chain watcher: $( [ "$chain_alive" -gt 0 ] && echo "running (Rounds 2+3 auto-queued)" || echo "not running" )"

  echo ""
  echo "  Refreshing every ${REFRESH}s — Ctrl+C to exit"
  sleep "$REFRESH"
done
