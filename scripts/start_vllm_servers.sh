#!/usr/bin/env bash
# Start 4 vLLM OpenAI-compatible servers, one per GH200 GPU.
# Each server is independent (tensor_parallel_size=1) on ports 8000-8003.
# All trace generation jobs hit these servers via HTTP — no in-process model loading.
#
# Usage:
#   bash scripts/start_vllm_servers.sh [MODEL_PATH] [BASE_PORT]
#   bash scripts/start_vllm_servers.sh                              # defaults below
#   bash scripts/start_vllm_servers.sh /tmp/prism_models/Qwen--Qwen3.5-35B-A3B 8000

set -eo pipefail
source "$(dirname "$0")/setup/env.sh" || true
cd "${PRISM_ROOT}"

MODEL_PATH="${1:-/tmp/prism_models/Qwen--Qwen3.5-35B-A3B}"
BASE_PORT="${2:-8000}"
LOG_DIR="${PRISM_LOG_DIR:-${PRISM_ROOT}/results/logs}"
mkdir -p "${LOG_DIR}"

echo "======================================================"
echo "Starting 4 vLLM servers"
echo "Model   : ${MODEL_PATH}"
echo "Ports   : ${BASE_PORT}-$((BASE_PORT+3))"
echo "Started : $(date)"
echo "======================================================"

SERVER_PIDS=()
for GPU in 0 1 2 3; do
    PORT=$((BASE_PORT + GPU))
    LOGFILE="${LOG_DIR}/vllm_server_gpu${GPU}.log"

    CUDA_VISIBLE_DEVICES=${GPU} \
    TORCHDYNAMO_DISABLE=1 \
    VLLM_ATTENTION_BACKEND=FLASHINFER \
    VLLM_FLASH_ATTN_VERSION=2 \
    nohup vllm serve "${MODEL_PATH}" \
        --port "${PORT}" \
        --host 0.0.0.0 \
        --max-model-len 8192 \
        --enforce-eager \
        --no-enable-prefix-caching \
        --gpu-memory-utilization 0.90 \
        --limit-mm-per-prompt '{"image": 0, "video": 0}' \
        --max-num-seqs 1024 \
        --disable-uvicorn-access-log \
        > "${LOGFILE}" 2>&1 &

    SERVER_PIDS+=($!)
    echo "  [GPU${GPU}] port ${PORT}  PID=${SERVER_PIDS[-1]}  log=${LOGFILE}"
done

echo ""
echo "Server PIDs: ${SERVER_PIDS[*]}"
printf "%s\n" "${SERVER_PIDS[@]}" > /tmp/prism_vllm_server_pids.txt

echo ""
echo "Waiting for all 4 servers to pass health check..."
for GPU in 0 1 2 3; do
    PORT=$((BASE_PORT + GPU))
    printf "  GPU%d (port %d) " "${GPU}" "${PORT}"
    until curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; do
        # Bail out if the server process died
        if ! kill -0 "${SERVER_PIDS[$GPU]}" 2>/dev/null; then
            echo ""
            echo "ERROR: server for GPU${GPU} (PID ${SERVER_PIDS[$GPU]}) died during startup."
            echo "       Check ${LOG_DIR}/vllm_server_gpu${GPU}.log"
            exit 1
        fi
        printf "."
        sleep 5
    done
    echo " ready"
done

echo ""
echo "======================================================"
echo "All 4 vLLM servers healthy — ready for trace generation"
echo "Ports: ${BASE_PORT} $((BASE_PORT+1)) $((BASE_PORT+2)) $((BASE_PORT+3))"
echo ""
echo "Next step:"
echo "  bash scripts/run_trace_gen.sh 1    # Round 1 — primary traces"
echo "======================================================"
