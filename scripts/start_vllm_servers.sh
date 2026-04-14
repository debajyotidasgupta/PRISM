#!/usr/bin/env bash
# Start a single vLLM OpenAI-compatible server using all 4 GH200 GPUs (TP=4).
# One server, tensor_parallel_size=4 — 4× the KV cache, better utilisation.
# All trace generation jobs hit this server via HTTP on port 8000.
#
# Usage:
#   bash scripts/start_vllm_servers.sh [MODEL_PATH] [PORT]
#   bash scripts/start_vllm_servers.sh
#   bash scripts/start_vllm_servers.sh /tmp/prism_models/Qwen--Qwen3.5-35B-A3B 8000

set -eo pipefail
source "$(dirname "$0")/setup/env.sh" || true
cd "${PRISM_ROOT}"

MODEL_PATH="${1:-/tmp/prism_models/Qwen--Qwen3.5-35B-A3B}"
PORT="${2:-8000}"
LOG_DIR="${PRISM_LOG_DIR:-${PRISM_ROOT}/results/logs}"
mkdir -p "${LOG_DIR}"
LOGFILE="${LOG_DIR}/vllm_server.log"

echo "======================================================"
echo "Starting vLLM server (TP=4, all 4 GH200 GPUs)"
echo "Model  : ${MODEL_PATH}"
echo "Port   : ${PORT}"
echo "Log    : ${LOGFILE}"
echo "Started: $(date)"
echo "======================================================"

CUDA_VISIBLE_DEVICES=0,1,2,3 \
nohup python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --port "${PORT}" \
    --host 0.0.0.0 \
    --tensor-parallel-size 4 \
    --max-model-len 8192 \
    --enforce-eager \
    --trust-remote-code \
    --no-enable-prefix-caching \
    --gpu-memory-utilization 0.90 \
    --limit-mm-per-prompt '{"image": 0, "video": 0}' \
    --max-num-seqs 512 \
    --disable-uvicorn-access-log \
    > "${LOGFILE}" 2>&1 &

SERVER_PID=$!
echo "Server PID: ${SERVER_PID}"
echo "${SERVER_PID}" > /tmp/prism_vllm_server_pid.txt

echo ""
echo "Waiting for server to pass health check..."
printf "  port ${PORT} "
until curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo ""
        echo "ERROR: server (PID ${SERVER_PID}) died during startup."
        echo "       Check ${LOGFILE}"
        exit 1
    fi
    printf "."
    sleep 5
done
echo " ready"

echo ""
echo "======================================================"
echo "vLLM server healthy on port ${PORT}  (PID ${SERVER_PID})"
echo ""
echo "Next step:"
echo "  bash scripts/run_trace_gen.sh 1    # Round 1 — primary traces"
echo "======================================================"
