#!/usr/bin/env bash
# PRISM environment setup — source this before any training/eval script
# Usage: source scripts/setup/env.sh

export PRISM_ROOT=/iopsstor/scratch/cscs/dasgupta/research/ideas/PRISM
export HF_HOME=${PRISM_ROOT}/.cache/huggingface
export HF_HUB_CACHE=${PRISM_ROOT}/.cache/huggingface/hub
export PRISM_MODEL_DIR=/tmp/prism_models
export HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null || echo "")

# Activate venv (only if not already active)
if [[ "${VIRTUAL_ENV:-}" != "${PRISM_ROOT}/.venv" ]]; then
    source /users/dasgupta/miniconda3/bin/activate ${PRISM_ROOT}/.venv
fi

# Ensure /tmp directories exist
mkdir -p /tmp/prism_models
mkdir -p /tmp/prism_logs

# CUDA settings for GH200
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# vLLM: provide real libcudart.so.12 (vLLM 0.19.0 compiled for CUDA 12)
# System now has CUDA 13 (cu130); libcudart.so.12 from sibling AMSD env satisfies VERNEEDED
# Also add torch lib dir so vLLM's flash_attn extensions can find libtorch.so
AMSD_CUDART="/iopsstor/scratch/cscs/dasgupta/research/ideas/AMSD/.venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib"
TORCH_LIB="${PRISM_ROOT}/.venv/lib/python3.12/site-packages/torch/lib"
if [[ -f "${AMSD_CUDART}/libcudart.so.12" ]]; then
    export LD_LIBRARY_PATH="${AMSD_CUDART}:${TORCH_LIB}:/usr/lib64:${LD_LIBRARY_PATH:-}"
fi

echo "PRISM_ROOT: ${PRISM_ROOT}"
echo "HF_HOME: ${HF_HOME}"
echo "PRISM_MODEL_DIR: ${PRISM_MODEL_DIR}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES}"
