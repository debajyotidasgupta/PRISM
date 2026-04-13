#!/usr/bin/env bash
# Parallel trace generation across all 4 GPUs
# Each GPU generates one domain's traces simultaneously

set -euo pipefail
source "$(dirname "$0")/setup/env.sh"
cd "${PRISM_ROOT}"

LOG_DIR=/tmp/prism_logs
mkdir -p "${LOG_DIR}"

TEACHER="${1:-Qwen/Qwen2.5-VL-7B-Instruct}"
N_PROBLEMS="${2:-2500}"

echo "Generating traces: teacher=${TEACHER}, n_per_domain=${N_PROBLEMS}"
echo "Started: $(date)"

# GPU 0: algebra
nohup python -m prism.generation.trace_generator \
    --teacher "${TEACHER}" --domain algebra \
    --n-problems "${N_PROBLEMS}" --gpu 0 \
    --output-dir "${PRISM_ROOT}/results/traces" \
    > "${LOG_DIR}/traces_algebra.log" 2>&1 &
echo "Algebra GPU0 PID=$!"

# GPU 1: geometry
nohup python -m prism.generation.trace_generator \
    --teacher "${TEACHER}" --domain geometry \
    --n-problems "${N_PROBLEMS}" --gpu 1 \
    --output-dir "${PRISM_ROOT}/results/traces" \
    > "${LOG_DIR}/traces_geometry.log" 2>&1 &
echo "Geometry GPU1 PID=$!"

# GPU 2: combinatorics
nohup python -m prism.generation.trace_generator \
    --teacher "${TEACHER}" --domain combinatorics \
    --n-problems "${N_PROBLEMS}" --gpu 2 \
    --output-dir "${PRISM_ROOT}/results/traces" \
    > "${LOG_DIR}/traces_combinatorics.log" 2>&1 &
echo "Combinatorics GPU2 PID=$!"

# GPU 3: number_theory (start first, then miscellaneous after)
nohup python -m prism.generation.trace_generator \
    --teacher "${TEACHER}" --domain number_theory \
    --n-problems "${N_PROBLEMS}" --gpu 3 \
    --output-dir "${PRISM_ROOT}/results/traces" \
    > "${LOG_DIR}/traces_nt.log" 2>&1 &
echo "NumberTheory GPU3 PID=$!"

wait

# GPU 3: miscellaneous (runs after number_theory)
nohup python -m prism.generation.trace_generator \
    --teacher "${TEACHER}" --domain miscellaneous \
    --n-problems "${N_PROBLEMS}" --gpu 3 \
    --output-dir "${PRISM_ROOT}/results/traces" \
    > "${LOG_DIR}/traces_misc.log" 2>&1 &
echo "Miscellaneous GPU3 PID=$!"

wait
echo "All trace generation complete: $(date)"
ls -la "${PRISM_ROOT}/results/traces/"
