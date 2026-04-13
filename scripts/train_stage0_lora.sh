#!/usr/bin/env bash
# Stage 0: LoRA baseline training on expert traces.
# Trains 5 domain-specific LoRA adapters on Qwen3.5-0.8B.
# Run after trace generation completes.
#
# Usage:
#   bash scripts/train_stage0_lora.sh [GPU_BASE] [DOMAINS...]
#   bash scripts/train_stage0_lora.sh 0 algebra geometry combinatorics number_theory miscellaneous

set -eo pipefail
source "$(dirname "$0")/setup/env.sh" || true
cd "${PRISM_ROOT}"

GPU_BASE="${1:-0}"
shift || true
DOMAINS=("${@:-algebra geometry combinatorics number_theory miscellaneous}")
if [[ ${#DOMAINS[@]} -eq 0 ]] || [[ "${DOMAINS[0]}" == "" ]]; then
    DOMAINS=(algebra geometry combinatorics number_theory miscellaneous)
fi

TRACES_DIR="${PRISM_ROOT}/results/traces"
OUTPUT_DIR="${PRISM_ROOT}/results/stage0/lora_adapters"
LOG_DIR="${PRISM_LOG_DIR:-${PRISM_ROOT}/results/logs}"
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

echo "======================================================"
echo "PRISM Stage 0 LoRA Training"
echo "Domains : ${DOMAINS[*]}"
echo "Traces  : ${TRACES_DIR}"
echo "Output  : ${OUTPUT_DIR}"
echo "Started : $(date)"
echo "======================================================"

# Check which trace files exist
echo ""
echo "Checking trace files:"
VALID_DOMAINS=()
for domain in "${DOMAINS[@]}"; do
    trace_file="${TRACES_DIR}/${domain}_traces.jsonl"
    if [[ -f "${trace_file}" ]] && [[ -s "${trace_file}" ]]; then
        count=$(wc -l < "${trace_file}")
        echo "  ${domain}: ${count} traces ✓"
        VALID_DOMAINS+=("${domain}")
    else
        echo "  ${domain}: MISSING or empty — skipping"
    fi
done

if [[ ${#VALID_DOMAINS[@]} -eq 0 ]]; then
    echo "ERROR: No valid trace files found in ${TRACES_DIR}"
    exit 1
fi

echo ""
echo "Training ${#VALID_DOMAINS[@]} domain adapters..."

# ── Train domains in parallel, one per GPU ──────────────────────────────────
PIDS=()
for i in "${!VALID_DOMAINS[@]}"; do
    domain="${VALID_DOMAINS[$i]}"
    gpu=$((GPU_BASE + i % 4))  # wrap around 4 GPUs
    trace_file="${TRACES_DIR}/${domain}_traces.jsonl"
    log_file="${LOG_DIR}/train_lora_${domain}.log"

    echo "  [GPU${gpu}] Training ${domain} adapter  (traces: ${trace_file})"
    nohup python -m prism.training.train_lora \
        --domain "${domain}" \
        --traces "${trace_file}" \
        --config "${PRISM_ROOT}/configs/training/stage0_lora.yaml" \
        --gpu "${gpu}" \
        --output-dir "${OUTPUT_DIR}" \
        > "${log_file}" 2>&1 &
    PIDS+=($!)
    echo "  [GPU${gpu}] ${domain} PID=${PIDS[-1]}"
done

echo ""
echo "All ${#PIDS[@]} training jobs launched. Waiting..."
for pid in "${PIDS[@]}"; do
    wait "${pid}" || echo "WARNING: PID ${pid} exited with error"
done

echo ""
echo "======================================================"
echo "Stage 0 LoRA training complete: $(date)"
echo "======================================================"
ls -lh "${OUTPUT_DIR}"

# ── Push adapters and eval results to upstream ────────────────────────────
echo ""
echo "Pushing LoRA adapters to GitHub..."
cd "${PRISM_ROOT}"
git add results/stage0/ 2>/dev/null || true
git diff --cached --quiet || \
    git commit -m "feat: Stage 0 LoRA adapters [$(date +%Y-%m-%d)]" && \
    git push origin main && echo "Pushed adapters to origin/main" || \
    echo "WARNING: git push failed (check credentials)"
