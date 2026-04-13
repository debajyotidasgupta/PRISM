#!/usr/bin/env bash
# PRISM environment setup — source this before any training/eval script
# Usage: source scripts/setup/env.sh

export PRISM_ROOT=/iopsstor/scratch/cscs/dasgupta/research/ideas/PRISM
export HF_HOME=${PRISM_ROOT}/.cache/huggingface
export HF_HUB_CACHE=${PRISM_ROOT}/.cache/huggingface/hub
export PRISM_MODEL_DIR=/tmp/prism_models
export HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null || echo "")

# Activate venv (only if not already active)
if [[ "$VIRTUAL_ENV" != "${PRISM_ROOT}/.venv" ]]; then
    source /users/dasgupta/miniconda3/bin/activate ${PRISM_ROOT}/.venv
fi

# Ensure /tmp directories exist
mkdir -p /tmp/prism_models
mkdir -p /tmp/prism_logs

# CUDA settings for GH200
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "PRISM_ROOT: ${PRISM_ROOT}"
echo "HF_HOME: ${HF_HOME}"
echo "PRISM_MODEL_DIR: ${PRISM_MODEL_DIR}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES}"
