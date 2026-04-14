#!/usr/bin/env bash
# Resume trace generation after an interruption.
# Run this after the vLLM server is back up.
# Usage: bash scripts/resume_trace_gen.sh [ROUND]   (default: starts at Round 2)
#
# State at interruption (2026-04-14 ~02:00):
#   Round 1: COMPLETE
#     algebra_traces.jsonl       2492 traces
#     combinatorics_traces.jsonl  602 traces
#     geometry_traces.jsonl       721 traces
#     number_theory_traces.jsonl  691 traces
#   Round 2: INCOMPLETE (~7% through Phase 1, safe to restart)
#   Round 3: NOT STARTED

source "$(dirname "$0")/setup/env.sh" || true
cd "${PRISM_ROOT}"

START_ROUND="${1:-2}"

echo "=================================================="
echo "PRISM Trace Generation Resume — starting at Round ${START_ROUND}"
echo "$(date)"
echo "=================================================="

# Verify vLLM server is up
if ! curl -sf "http://localhost:8000/health" > /dev/null 2>&1; then
    echo "ERROR: vLLM server not running. Start it first:"
    echo "  bash scripts/start_vllm_servers.sh"
    exit 1
fi
echo "vLLM server OK"

LOG_DIR="${PRISM_LOG_DIR:-${PRISM_ROOT}/results/logs}"

# Skip rounds that already have complete JSONL output
for ROUND in $(seq "${START_ROUND}" 3); do
    echo ""
    echo "=== Launching Round ${ROUND} ==="
    bash "${PRISM_ROOT}/scripts/run_trace_gen.sh" "${ROUND}" 2500 \
        >> "${LOG_DIR}/round${ROUND}_resume.log" 2>&1
    STATUS=$?
    echo "Round ${ROUND} finished (exit=${STATUS}) at $(date)"
done

echo ""
echo "All rounds complete."
ls -lh "${PRISM_ROOT}/results/traces/"*.jsonl 2>/dev/null
