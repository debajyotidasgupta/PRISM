#!/usr/bin/env bash
# Auto-chain watcher: polls for Round completion then launches next round.
# Usage: bash scripts/chain_watcher.sh [R1_PIDS...]
#   If no PIDs given, waits for any trace_generator to die, then checks JSONL output.

source "$(dirname "$0")/setup/env.sh" 2>/dev/null || true
cd "${PRISM_ROOT:-.}"

LOG_DIR="${PRISM_LOG_DIR:-${PRISM_ROOT}/results/logs}"
mkdir -p "${LOG_DIR}"

WATCH_LOG="${LOG_DIR}/chain_watcher.log"
exec >> "${WATCH_LOG}" 2>&1

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

R1_PIDS=("$@")

# ── Wait for Round 1 ─────────────────────────────────────────────────────────
if [ "${#R1_PIDS[@]}" -gt 0 ]; then
    log "Watching Round 1 PIDs: ${R1_PIDS[*]}"
    for pid in "${R1_PIDS[@]}"; do
        while kill -0 "$pid" 2>/dev/null; do
            sleep 30
        done
        log "PID ${pid} finished."
    done
else
    log "No PIDs given — polling for trace_generator processes to clear."
    while pgrep -f "trace_generator" > /dev/null 2>&1; do
        sleep 30
    done
fi
log "Round 1 complete. Launching Round 2..."

# ── Launch Round 2 ────────────────────────────────────────────────────────────
bash "${PRISM_ROOT}/scripts/run_trace_gen.sh" 2 2500 \
    > "${LOG_DIR}/round2_launcher.log" 2>&1
R2_EXIT=$?
log "Round 2 finished (exit=${R2_EXIT}). Launching Round 3..."

# ── Launch Round 3 ────────────────────────────────────────────────────────────
bash "${PRISM_ROOT}/scripts/run_trace_gen.sh" 3 2500 \
    > "${LOG_DIR}/round3_launcher.log" 2>&1
R3_EXIT=$?
log "Round 3 finished (exit=${R3_EXIT}). All rounds complete."

# ── Update PROGRESS.md ───────────────────────────────────────────────────────
python3 - << 'PYEOF'
import json, pathlib, datetime

root = pathlib.Path("${PRISM_ROOT}")
traces = root / "results" / "traces"
lines = ["", "## Trace Generation Summary (auto-updated)", ""]
lines.append(f"Completed: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
lines.append("")
for f in sorted(traces.glob("*.jsonl")):
    n = sum(1 for _ in f.open())
    sf = f.with_suffix(".json").name.replace("traces", "stats")
    lines.append(f"- {f.name}: {n} traces")
print("\n".join(lines))
PYEOF

log "Chain watcher done."
