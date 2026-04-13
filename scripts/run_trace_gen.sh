#!/usr/bin/env bash
# Run one round of PRISM trace generation against the 4 vLLM servers.
# Servers must already be running (bash scripts/start_vllm_servers.sh).
#
# All prompts for each domain+phase are sent concurrently via async HTTP.
# The server's continuous-batching scheduler queues and batches them optimally.
#
# Usage:
#   bash scripts/run_trace_gen.sh ROUND [N_PROBLEMS] [BASE_PORT]
#   bash scripts/run_trace_gen.sh 1            # Round 1: 4 primary domains, 2500 each
#   bash scripts/run_trace_gen.sh 2 2500       # Round 2: miscellaneous + cross-verify
#   bash scripts/run_trace_gen.sh 3 2500       # Round 3: remaining cross-verify pairs
#
# Round layout:
#   Round 1  GPU0=algebra  GPU1=geometry  GPU2=combinatorics  GPU3=number_theory
#   Round 2  GPU0=miscellaneous  GPU1=algebra×misc  GPU2=geometry×algebra  GPU3=combinatorics×nt
#   Round 3  GPU0=nt×algebra  GPU1=misc×comb  GPU2=algebra×geometry  GPU3=geometry×comb

set -eo pipefail
source "$(dirname "$0")/setup/env.sh" || true
cd "${PRISM_ROOT}"

ROUND="${1:?Usage: bash scripts/run_trace_gen.sh ROUND [N_PROBLEMS] [BASE_PORT]}"
N="${2:-2500}"
BASE_PORT="${3:-8000}"
TEACHER="Qwen/Qwen3.5-35B-A3B"
LOG_DIR="${PRISM_LOG_DIR:-${PRISM_ROOT}/results/logs}"
mkdir -p "${LOG_DIR}"

echo "======================================================"
echo "PRISM Trace Generation — Round ${ROUND}"
echo "N per domain : ${N}"
echo "Base port    : ${BASE_PORT}"
echo "Started      : $(date)"
echo "======================================================"

# ── Verify all 4 servers are up ───────────────────────────────────────────
echo "Checking server health..."
for GPU in 0 1 2 3; do
    PORT=$((BASE_PORT + GPU))
    if ! curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then
        echo "ERROR: vLLM server on port ${PORT} is not responding."
        echo "       Start servers first:  bash scripts/start_vllm_servers.sh"
        exit 1
    fi
    echo "  port ${PORT} OK"
done
echo ""

# ── Helper: launch one domain against one server ───────────────────────────
# Each job is a pure HTTP client — no GPU assignment needed here.
launch() {
    local DOMAIN=$1
    local GPU=$2           # selects port, not CUDA_VISIBLE_DEVICES
    local CV="${3:-}"
    local PORT=$((BASE_PORT + GPU))
    local SUFFIX=""
    local CV_ARG=""

    if [[ -n "${CV}" ]]; then
        SUFFIX="_cv_${CV}"
        CV_ARG="--cross-verify-domain ${CV}"
    fi

    local LOGFILE="${LOG_DIR}/traces_r${ROUND}_${DOMAIN}${SUFFIX}.log"

    nohup python -m prism.generation.trace_generator \
        --teacher  "${TEACHER}" \
        --domain   "${DOMAIN}" \
        --n-problems "${N}" \
        --output-dir "${PRISM_ROOT}/results/traces" \
        --max-tokens 2048 \
        --filter-tokens 65536 \
        --server-url "http://localhost:${PORT}" \
        ${CV_ARG} \
        > "${LOGFILE}" 2>&1 &

    local PID=$!
    echo "  [port${PORT}] ${DOMAIN}${SUFFIX}  PID=${PID}  log=$(basename ${LOGFILE})"
    echo "${PID}"
}

# ── Launch jobs for the requested round ───────────────────────────────────
PIDS=()
case "${ROUND}" in
1)
    echo "Round 1: primary traces (algebra / geometry / combinatorics / number_theory)"
    PIDS+=( $(launch algebra        0) )
    PIDS+=( $(launch geometry       1) )
    PIDS+=( $(launch combinatorics  2) )
    PIDS+=( $(launch number_theory  3) )
    ;;
2)
    echo "Round 2: miscellaneous primary + cross-domain verification pairs"
    PIDS+=( $(launch miscellaneous  0) )
    PIDS+=( $(launch algebra        1  miscellaneous) )
    PIDS+=( $(launch geometry       2  algebra) )
    PIDS+=( $(launch combinatorics  3  number_theory) )
    ;;
3)
    echo "Round 3: remaining cross-domain verification pairs"
    PIDS+=( $(launch number_theory  0  algebra) )
    PIDS+=( $(launch miscellaneous  1  combinatorics) )
    PIDS+=( $(launch algebra        2  geometry) )
    PIDS+=( $(launch geometry       3  combinatorics) )
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
