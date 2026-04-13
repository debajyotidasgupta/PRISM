#!/usr/bin/env bash
# Stage 0: LoRA hypothesis validation
# Runs 5 domain-specific LoRA adapters in parallel across 4 GPUs
# Also runs general LoRA on GPU3
# Trace generation runs on GPU0+1 simultaneously

set -euo pipefail

source "$(dirname "$0")/setup/env.sh"
cd "${PRISM_ROOT}"

LOG_DIR=/tmp/prism_logs
mkdir -p "${LOG_DIR}"

echo "=========================================="
echo "PRISM Stage 0: LoRA Hypothesis Validation"
echo "Started: $(date)"
echo "=========================================="

# ─── Step 1: Generate expert traces (2,500 per domain) ───────────────────────
# Uses GPU 0+1 for teacher (Qwen2.5-VL-7B-Instruct)
echo "[Step 1] Generating expert traces..."

TEACHER="Qwen/Qwen2.5-VL-7B-Instruct"

for domain in algebra geometry combinatorics number_theory miscellaneous; do
    TRACE_FILE="${PRISM_ROOT}/results/traces/${domain}_traces.jsonl"
    if [ -f "${TRACE_FILE}" ]; then
        echo "  Traces already exist: ${TRACE_FILE} — skipping"
        continue
    fi

    GPU_ID=$((RANDOM % 4))  # Distribute across GPUs
    echo "  Generating ${domain} traces on GPU ${GPU_ID}..."
    nohup python -m prism.generation.trace_generator \
        --teacher "${TEACHER}" \
        --domain "${domain}" \
        --n-problems 2500 \
        --gpu "${GPU_ID}" \
        --output-dir "${PRISM_ROOT}/results/traces" \
        --max-tokens 4096 \
        > "${LOG_DIR}/traces_${domain}.log" 2>&1 &
    echo "    PID $! → ${LOG_DIR}/traces_${domain}.log"
done

echo "Waiting for all trace generation jobs..."
wait
echo "[Step 1] Done. Traces in ${PRISM_ROOT}/results/traces/"

# ─── Step 2: Train domain-specific LoRA adapters ────────────────────────────
echo "[Step 2] Training LoRA adapters..."

# Algebra on GPU 0
nohup python -m prism.training.train_lora \
    --domain algebra \
    --traces results/traces/algebra_traces.jsonl \
    --config configs/training/stage0_lora.yaml \
    --gpu 0 --output-dir results/stage0/lora_adapters \
    > "${LOG_DIR}/lora_algebra.log" 2>&1 &
echo "  Algebra LoRA: PID $! (GPU 0)"

# Geometry on GPU 1
nohup python -m prism.training.train_lora \
    --domain geometry \
    --traces results/traces/geometry_traces.jsonl \
    --config configs/training/stage0_lora.yaml \
    --gpu 1 --output-dir results/stage0/lora_adapters \
    > "${LOG_DIR}/lora_geometry.log" 2>&1 &
echo "  Geometry LoRA: PID $! (GPU 1)"

# Combinatorics + Number Theory on GPU 2
nohup python -m prism.training.train_lora \
    --domain combinatorics \
    --traces results/traces/combinatorics_traces.jsonl \
    --config configs/training/stage0_lora.yaml \
    --gpu 2 --output-dir results/stage0/lora_adapters \
    > "${LOG_DIR}/lora_combinatorics.log" 2>&1 &
echo "  Combinatorics LoRA: PID $! (GPU 2)"

# Misc + General on GPU 3
nohup python -m prism.training.train_lora \
    --domain miscellaneous \
    --traces results/traces/miscellaneous_traces.jsonl \
    --config configs/training/stage0_lora.yaml \
    --gpu 3 --output-dir results/stage0/lora_adapters \
    > "${LOG_DIR}/lora_misc.log" 2>&1 &
echo "  Miscellaneous LoRA: PID $! (GPU 3)"

echo "Waiting for LoRA training jobs..."
wait

# Number Theory and General LoRA (sequentially on GPU 0 and 1 after others finish)
nohup python -m prism.training.train_lora \
    --domain number_theory \
    --traces results/traces/number_theory_traces.jsonl \
    --config configs/training/stage0_lora.yaml \
    --gpu 0 --output-dir results/stage0/lora_adapters \
    > "${LOG_DIR}/lora_nt.log" 2>&1 &
echo "  Number Theory LoRA: PID $! (GPU 0)"

nohup python -m prism.training.train_lora \
    --domain general \
    --traces "" \
    --config configs/training/stage0_lora.yaml \
    --gpu 1 --output-dir results/stage0/lora_adapters \
    > "${LOG_DIR}/lora_general.log" 2>&1 &
echo "  General LoRA: PID $! (GPU 1)"

wait
echo "[Step 2] Done. LoRA adapters in results/stage0/lora_adapters/"

# ─── Step 3: Evaluate all adapters ──────────────────────────────────────────
echo "[Step 3] Evaluating LoRA adapters..."

nohup python -m prism.training.eval_lora \
    --adapters-dir results/stage0/lora_adapters \
    --benchmark math500 \
    --gpu 0 \
    --output-dir results/stage0/eval \
    > "${LOG_DIR}/eval_lora.log" 2>&1

echo "[Step 3] Done. Eval results in results/stage0/eval/"

echo "=========================================="
echo "Stage 0 complete: $(date)"
echo "Check PROGRESS.md for pass/fail determination"
echo "=========================================="
