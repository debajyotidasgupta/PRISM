#!/usr/bin/env bash
# Run all benchmarks + ablations in parallel across 4 GPUs

set -euo pipefail
source "$(dirname "$0")/setup/env.sh"
cd "${PRISM_ROOT}"

LOG_DIR=/tmp/prism_logs
mkdir -p "${LOG_DIR}"

MODEL_DIR="${1:-results/stage2}"
echo "Evaluating model: ${MODEL_DIR}"
echo "Started: $(date)"

# GPU 0: OlymMATH (primary)
nohup prism-eval \
    --model "${MODEL_DIR}" --benchmark olymmath \
    --gpu 0 --max-samples 100 \
    --output-dir results --model-name prism \
    > "${LOG_DIR}/eval_olymmath.log" 2>&1 &
echo "OlymMATH GPU0 PID=$!"

# GPU 1: OlympiadBench (primary)
nohup prism-eval \
    --model "${MODEL_DIR}" --benchmark olympiadbench \
    --gpu 1 --max-samples 2126 \
    --output-dir results --model-name prism \
    > "${LOG_DIR}/eval_olympiadbench.log" 2>&1 &
echo "OlympiadBench GPU1 PID=$!"

# GPU 2: MATH-500 (validation)
nohup prism-eval \
    --model "${MODEL_DIR}" --benchmark math500 \
    --gpu 2 --max-samples 500 \
    --output-dir results --model-name prism \
    > "${LOG_DIR}/eval_math500.log" 2>&1 &
echo "MATH-500 GPU2 PID=$!"

# GPU 3: Baselines (qwen35_0.8b no-think)
nohup prism-eval \
    --model "Qwen/Qwen3.5-0.8B" --benchmark math500 \
    --gpu 3 --max-samples 500 \
    --output-dir results --model-name baseline_qwen35_08b_nothink \
    > "${LOG_DIR}/eval_baseline_nothink.log" 2>&1 &
echo "Baseline GPU3 PID=$!"

wait
echo "Primary evals complete: $(date)"

# Ablation runs
echo "Running ablations..."
python -m prism.eval.ablations \
    --model-dir "${MODEL_DIR}" \
    --backbone "Qwen/Qwen3.5-0.8B" \
    --benchmark math500 \
    --max-samples 200 \
    --output-dir results/ablations

echo "All evaluation complete: $(date)"
