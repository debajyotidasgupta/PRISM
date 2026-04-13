#!/usr/bin/env bash
# Full trace generation across all 4 GH200 GPUs.
#
# GPU allocation strategy — maximize utilization:
#   GPU 0: algebra (primary domain traces)
#   GPU 1: geometry + combinatorics cross-verify pairs
#   GPU 2: combinatorics (primary) + algebra→misc cross-verify
#   GPU 3: number_theory + miscellaneous (sequential on same GPU)
#
# After primary traces are done, cross-domain verification pairs run on
# whichever GPUs finish first. All 4 GPUs stay busy throughout.
#
# Usage:
#   bash scripts/generate_traces.sh [TEACHER] [N_PROBLEMS]
#   bash scripts/generate_traces.sh Qwen/Qwen3.5-35B-A3B 2500

set -euo pipefail
source "$(dirname "$0")/setup/env.sh"
cd "${PRISM_ROOT}"

LOG_DIR=/tmp/prism_logs
mkdir -p "${LOG_DIR}"

TEACHER="${1:-Qwen/Qwen3.5-35B-A3B}"
N_PROBLEMS="${2:-2500}"
# Set USE_VLLM=1 to enable vLLM batch backend (requires compatible vLLM install)
USE_VLLM="${USE_VLLM:-0}"
VLLM_FLAG=""
if [[ "${USE_VLLM}" == "1" ]]; then
    VLLM_FLAG="--use-vllm"
    echo "vLLM backend: ENABLED"
else
    echo "vLLM backend: disabled (HF serial mode)"
fi

echo "======================================================"
echo "PRISM Trace Generation — all 4 GH200 GPUs"
echo "Teacher : ${TEACHER}"
echo "N/domain: ${N_PROBLEMS}"
echo "Started : $(date)"
echo "======================================================"

# ── Helper: launch one domain on one GPU ──────────────────────────────────
launch_domain() {
    local DOMAIN=$1
    local GPU=$2
    local CROSS_VERIFY="${3:-}"   # optional cross-verify-domain arg

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

# ══════════════════════════════════════════════════════════════════════════
# Round 1: Primary domain traces — all 4 GPUs simultaneously
# ══════════════════════════════════════════════════════════════════════════
echo ""
echo "── Round 1: primary domain traces ──────────────────────"
launch_domain algebra        0
launch_domain geometry       1
launch_domain combinatorics  2
launch_domain number_theory  3

wait
echo ""
echo "Round 1 complete: $(date)"
echo "Primary traces done: algebra, geometry, combinatorics, number_theory"

# ══════════════════════════════════════════════════════════════════════════
# Round 2: miscellaneous primary + cross-domain verification pairs
# 4 tasks distributed across 4 GPUs
# ══════════════════════════════════════════════════════════════════════════
echo ""
echo "── Round 2: miscellaneous + cross-verify pairs ──────────"
launch_domain miscellaneous  0          # GPU 0: last primary domain
launch_domain algebra        1  miscellaneous     # GPU 1: algebra → verified by misc
launch_domain geometry       2  algebra            # GPU 2: geometry → verified by algebra
launch_domain combinatorics  3  number_theory      # GPU 3: combinatorics → verified by NT

wait
echo ""
echo "Round 2 complete: $(date)"

# ══════════════════════════════════════════════════════════════════════════
# Round 3: remaining cross-domain verification pairs
# ══════════════════════════════════════════════════════════════════════════
echo ""
echo "── Round 3: remaining cross-verify pairs ────────────────"
launch_domain number_theory  0  algebra            # GPU 0: NT → verified by algebra
launch_domain miscellaneous  1  combinatorics      # GPU 1: misc → verified by combinatorics
launch_domain algebra        2  geometry           # GPU 2: algebra → verified by geometry
launch_domain geometry       3  combinatorics      # GPU 3: geometry → verified by combinatorics

wait
echo ""
echo "Round 3 complete: $(date)"

# ══════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════
echo ""
echo "======================================================"
echo "All trace generation complete: $(date)"
echo "======================================================"
ls -lh "${PRISM_ROOT}/results/traces/"
echo ""
echo "Stats files:"
ls "${PRISM_ROOT}/results/traces/"*_stats.json 2>/dev/null || echo "  (none found)"
