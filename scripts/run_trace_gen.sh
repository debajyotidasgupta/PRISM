#!/usr/bin/env bash
# Run one round of PRISM trace generation against the single TP=4 vLLM server.
# Server must already be running (bash scripts/start_vllm_servers.sh).
#
# All domains in a round run in parallel; each fires all its prompts concurrently
# to the same server. The server's scheduler batches everything optimally.
#
# Usage:
#   bash scripts/run_trace_gen.sh ROUND [N_PROBLEMS] [PORT]
#   bash scripts/run_trace_gen.sh 1          # Round 1: 4 primary domains
#   bash scripts/run_trace_gen.sh 2 2500     # Round 2: misc + cross-verify
#   bash scripts/run_trace_gen.sh 3 2500     # Round 3: remaining cross-verify
#
# Round layout:
#   Round 1  algebra / geometry / combinatorics / number_theory
#   Round 2  miscellaneous | algebra×misc | geometry×algebra | combinatorics×nt
#   Round 3  nt×algebra | misc×comb | algebra×geometry | geometry×comb

set -eo pipefail
source "$(dirname "$0")/setup/env.sh" || true
cd "${PRISM_ROOT}"

ROUND="${1:?Usage: bash scripts/run_trace_gen.sh ROUND [N_PROBLEMS] [PORT]}"
N="${2:-2500}"
PORT="${3:-8000}"
SERVER_URL="http://localhost:${PORT}"
TEACHER="Qwen/Qwen3.5-35B-A3B"
LOG_DIR="${PRISM_LOG_DIR:-${PRISM_ROOT}/results/logs}"
mkdir -p "${LOG_DIR}"

echo "======================================================"
echo "PRISM Trace Generation — Round ${ROUND}"
echo "N per domain : ${N}"
echo "Server       : ${SERVER_URL}"
echo "Started      : $(date)"
echo "======================================================"

# ── Verify server is up ───────────────────────────────────────────────────
if ! curl -sf "${SERVER_URL}/health" > /dev/null 2>&1; then
    echo "ERROR: vLLM server at ${SERVER_URL} is not responding."
    echo "       Start it first:  bash scripts/start_vllm_servers.sh"
    exit 1
fi
echo "Server healthy. Launching Round ${ROUND}..."
echo ""

# ── Helper: launch one domain job (pure HTTP client, no GPU assignment) ───
# Appends the child PID directly to the global PIDS array (no subshell).
launch() {
    local DOMAIN=$1
    local CV="${2:-}"
    local SUFFIX="" CV_ARG=""

    if [[ -n "${CV}" ]]; then
        SUFFIX="_cv_${CV}"
        CV_ARG="--cross-verify-domain ${CV}"
    fi

    local LOGFILE="${LOG_DIR}/traces_r${ROUND}_${DOMAIN}${SUFFIX}.log"

    python -m prism.generation.trace_generator \
        --teacher    "${TEACHER}" \
        --domain     "${DOMAIN}" \
        --n-problems "${N}" \
        --output-dir "${PRISM_ROOT}/results/traces" \
        --max-tokens 4096 \
        --filter-tokens 65536 \
        --server-url "${SERVER_URL}" \
        ${CV_ARG} \
        > "${LOGFILE}" 2>&1 &

    local PID=$!
    PIDS+=("${PID}")
    echo "  ${DOMAIN}${SUFFIX}  PID=${PID}  log=$(basename ${LOGFILE})"
}

# ── Launch jobs for requested round ──────────────────────────────────────
PIDS=()
case "${ROUND}" in
1)
    echo "Round 1: primary traces"
    launch algebra
    launch geometry
    launch combinatorics
    launch number_theory
    ;;
2)
    echo "Round 2: miscellaneous primary + cross-domain verification pairs"
    launch miscellaneous
    launch algebra        miscellaneous
    launch geometry       algebra
    launch combinatorics  number_theory
    ;;
3)
    echo "Round 3: remaining cross-domain verification pairs"
    launch number_theory  algebra
    launch miscellaneous  combinatorics
    launch algebra        geometry
    launch geometry       combinatorics
    ;;
*)
    echo "ERROR: unknown round '${ROUND}'. Use 1, 2, or 3."
    exit 1
    ;;
esac

echo ""
echo "Waiting for Round ${ROUND} jobs (PIDs: ${PIDS[*]})..."
FAILED=0
for pid in "${PIDS[@]}"; do
    wait "${pid}" || { echo "WARNING: PID ${pid} exited with error"; FAILED=$((FAILED+1)); }
done

echo ""
echo "======================================================"
echo "Round ${ROUND} complete: $(date)   (${FAILED} failures)"
echo "======================================================"
ls -lh "${PRISM_ROOT}/results/traces/" 2>/dev/null || true

# ── Push stats to GitHub ──────────────────────────────────────────────────
cd "${PRISM_ROOT}"
git add results/traces/*_stats.json 2>/dev/null || true
git diff --cached --quiet || \
    git commit -m "chore: Round ${ROUND} trace stats [$(date +%Y-%m-%d)]" && \
    git push origin main && echo "Stats pushed to origin/main" || \
    echo "WARNING: git push failed"
