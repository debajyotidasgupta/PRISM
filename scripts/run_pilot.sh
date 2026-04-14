#!/usr/bin/env bash
# Pilot end-to-end: train 5 domain LoRAs + 1 general LoRA on 100 samples/domain,
# then evaluate and check Stage 0 Gate.
#
# For 100 samples the dataset is tiny — run all domains sequentially on GPU 0
# with large batch size to maximize GPU utilization per job.
# CUDA_VISIBLE_DEVICES is set in the shell (not Python) so it takes effect
# before any CUDA initialization.
#
# Usage: bash scripts/run_pilot.sh

set -e
cd "$(dirname "$0")/.."
source /users/dasgupta/miniconda3/bin/activate /iopsstor/scratch/cscs/dasgupta/research/ideas/PRISM/.venv

PILOT_DIR="results/traces/pilot"
ADAPTER_DIR="results/pilot/lora_adapters"
LOG_DIR="results/logs"
CONFIG="configs/training/stage0_lora.yaml"

mkdir -p "$ADAPTER_DIR" "$LOG_DIR"

# Use all 4 GPUs in DDP for each domain — 100 samples × 4 GPUs = 25 per GPU
# Better utilization than sequential single-GPU given 95 GB available each.
# But for simplicity and reliability, run each domain on a single GPU 0 with
# large batch size (the 0.8B model + LoRA uses ~10 GB; 85 GB headroom for big batches).
# batch_size=64, grad_accum=1 → effective batch=64, no accumulation overhead.

echo "[$(date)] Starting pilot training — 6 jobs sequentially on GPU 0"

MAX_STEPS=1000  # same budget for all LoRAs (domain + general)

train_domain() {
    local domain=$1
    local traces=$2
    echo "[$(date)] Training: $domain (max_steps=$MAX_STEPS)"
    CUDA_VISIBLE_DEVICES=0 python -m prism.training.train_lora \
        --domain "$domain" \
        --traces "$traces" \
        --config "$CONFIG" \
        --gpu 0 \
        --max-steps "$MAX_STEPS" \
        --output-dir "$ADAPTER_DIR" \
        > "$LOG_DIR/pilot_train_${domain}.log" 2>&1
    echo "[$(date)] Done: $domain — $(tail -1 $LOG_DIR/pilot_train_${domain}.log)"
}

train_domain algebra      "$PILOT_DIR/algebra_traces.jsonl"
train_domain geometry     "$PILOT_DIR/geometry_traces.jsonl"
train_domain combinatorics "$PILOT_DIR/combinatorics_traces.jsonl"
train_domain number_theory "$PILOT_DIR/number_theory_traces.jsonl"
train_domain miscellaneous "$PILOT_DIR/miscellaneous_traces.jsonl"

# General LoRA: combined traces from all 5 domains
echo "[$(date)] Building combined trace file for general LoRA"
cat "$PILOT_DIR"/{algebra,geometry,combinatorics,number_theory,miscellaneous}_traces.jsonl \
    > "$PILOT_DIR/general_traces.jsonl"
train_domain general "$PILOT_DIR/general_traces.jsonl"

echo ""
echo "[$(date)] === ALL 6 LoRAs TRAINED ==="
ls -la "$ADAPTER_DIR"
echo ""
echo "Next: eval with: python -m prism.training.train_lora --eval ..."
