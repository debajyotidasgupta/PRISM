#!/usr/bin/env bash
# Rounds 2 and 3 of trace generation.
# Waits for the 4 Round-1 PIDs passed as arguments, then launches cross-verify pairs.
#
# Usage:
#   bash scripts/generate_rounds23.sh PID1 PID2 PID3 PID4 [TEACHER] [N_PROBLEMS]
#   bash scripts/generate_rounds23.sh 259270 259271 259272 259273
#   bash scripts/generate_rounds23.sh 259270 259271 259272 259273 Qwen/Qwen3.5-35B-A3B 2500

set -eo pipefail
source "$(dirname "$0")/setup/env.sh" || true
cd "${PRISM_ROOT}"

ROUND1_PIDS=("$1" "$2" "$3" "$4")
TEACHER="${5:-Qwen/Qwen3.5-35B-A3B}"
N_PROBLEMS="${6:-2500}"

LOG_DIR=/tmp/prism_logs
mkdir -p "${LOG_DIR}"

# Always use vLLM
export USE_VLLM=1
VLLM_FLAG="--use-vllm"

echo "======================================================"
echo "PRISM Rounds 2+3 — waiting for Round 1 to complete"
echo "Round-1 PIDs: ${ROUND1_PIDS[*]}"
echo "Teacher     : ${TEACHER}"
echo "Started     : $(date)"
echo "======================================================"

# ── Helper: launch one domain on one GPU ──────────────────────────────────
launch_domain() {
    local DOMAIN=$1
    local GPU=$2
    local CROSS_VERIFY="${3:-}"

    local SUFFIX=""
    local CV_ARG=""
    if [[ -n "${CROSS_VERIFY}" ]]; then
        SUFFIX="_cv_${CROSS_VERIFY}"
        CV_ARG="--cross-verify-domain ${CROSS_VERIFY}"
    fi

    nohup python -m prism.generation.trace_generator \
        --teacher "${TEACHER}" \
        --domain  "${DOMAIN}" \
        --n-problems "${N_PROBLEMS}" \
        --gpu     "${GPU}" \
        --output-dir "${PRISM_ROOT}/results/traces" \
        ${CV_ARG} ${VLLM_FLAG} \
        > "${LOG_DIR}/traces_${DOMAIN}${SUFFIX}.log" 2>&1 &
    echo "  [GPU${GPU}] ${DOMAIN}${SUFFIX}  PID=$!"
}

# ── Wait for Round 1 to complete ──────────────────────────────────────────
echo ""
echo "Polling for Round 1 PIDs: ${ROUND1_PIDS[*]}"
while true; do
    all_done=1
    for pid in "${ROUND1_PIDS[@]}"; do
        if kill -0 "${pid}" 2>/dev/null; then
            all_done=0
            break
        fi
    done
    if [[ "${all_done}" == "1" ]]; then
        break
    fi
    sleep 60
    echo "  $(date) — still waiting for: $(for p in ${ROUND1_PIDS[*]}; do kill -0 $p 2>/dev/null && echo $p; done | tr '\n' ' ')"
done
echo "Round 1 complete: $(date)"

# ══════════════════════════════════════════════════════════════════════════
# Round 2: miscellaneous primary + cross-domain verification pairs
# ══════════════════════════════════════════════════════════════════════════
echo ""
echo "── Round 2: miscellaneous + cross-verify pairs ──────────"
launch_domain miscellaneous  0
launch_domain algebra        1  miscellaneous
launch_domain geometry       2  algebra
launch_domain combinatorics  3  number_theory

ROUND2_PIDS=()
# Capture last 4 background PIDs
wait
echo "Round 2 complete: $(date)"

# ══════════════════════════════════════════════════════════════════════════
# Round 3: remaining cross-domain verification pairs
# ══════════════════════════════════════════════════════════════════════════
echo ""
echo "── Round 3: remaining cross-verify pairs ────────────────"
launch_domain number_theory  0  algebra
launch_domain miscellaneous  1  combinatorics
launch_domain algebra        2  geometry
launch_domain geometry       3  combinatorics

wait
echo "Round 3 complete: $(date)"

echo ""
echo "======================================================"
echo "All rounds complete: $(date)"
echo "======================================================"
ls -lh "${PRISM_ROOT}/results/traces/"
echo ""
echo "Stats files:"
ls "${PRISM_ROOT}/results/traces/"*_stats.json 2>/dev/null || echo "  (none found)"
