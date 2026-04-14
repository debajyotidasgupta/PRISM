#!/usr/bin/env bash
# Train all 15 PRISM expert blocks on 100-sample pilot traces.
# Parallelizes across 4 GH200 GPUs: 4 blocks concurrently per phase.
#
# Schedule:
#   Phase 0 (Solve):   algebra(0), geometry(1), combinatorics(2), NT(3) → parallel
#                      miscellaneous(0) → sequential after phase 0 batch
#   Phase 1 (Verify):  same layout
#   Phase 2 (Correct): same layout
#
# Usage: bash scripts/train_expert_blocks_pilot.sh

set -e
cd "$(dirname "$0")/.."
source /users/dasgupta/miniconda3/bin/activate /iopsstor/scratch/cscs/dasgupta/research/ideas/PRISM/.venv

PILOT_DIR="results/traces/pilot"
OUTPUT_DIR="results/pilot/expert_blocks"
LOG_DIR="results/logs"
CONFIG="configs/model/prism_0.8b.yaml"
EPOCHS=3

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

echo "[$(date)] === PRISM Expert Block Pilot Training ==="
echo "  Domains: algebra, geometry, combinatorics, number_theory, miscellaneous"
echo "  Phases:  solve (0), verify (1), correct (2)"
echo "  Traces:  100 samples/domain"
echo ""

# Trace file mapping
declare -A TRACE_FILE
TRACE_FILE[algebra]="$PILOT_DIR/algebra_traces.jsonl"
TRACE_FILE[geometry]="$PILOT_DIR/geometry_traces.jsonl"
TRACE_FILE[combinatorics]="$PILOT_DIR/combinatorics_traces.jsonl"
TRACE_FILE[number_theory]="$PILOT_DIR/number_theory_traces.jsonl"
TRACE_FILE[miscellaneous]="$PILOT_DIR/miscellaneous_traces.jsonl"

# GPU assignment for parallel batch: 4 domains per round
declare -A GPU_ASSIGN
GPU_ASSIGN[algebra]=0
GPU_ASSIGN[geometry]=1
GPU_ASSIGN[combinatorics]=2
GPU_ASSIGN[number_theory]=3
GPU_ASSIGN[miscellaneous]=0  # re-use GPU 0 sequentially

train_block() {
    local domain=$1
    local phase=$2
    local gpu=$3
    local logfile="$LOG_DIR/pilot_expert_${domain}_phase${phase}.log"
    echo "[$(date)] Starting: ${domain} phase=${phase} on GPU ${gpu}"
    CUDA_VISIBLE_DEVICES=$gpu python -u -c "
import sys, os
sys.path.insert(0, 'src')
os.environ['CUDA_VISIBLE_DEVICES'] = '$gpu'
os.environ.setdefault('HF_HOME', '.cache/huggingface')
import torch, yaml, logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
from prism.model.config import PRISMConfig
from prism.model.prism_model import PRISMModel
from prism.training.train_expert import ExpertTrainer
from transformers import AutoTokenizer

DOMAIN_TO_IDX = {'algebra': 0, 'geometry': 1, 'combinatorics': 2, 'number_theory': 3, 'miscellaneous': 4}
PHASE_TO_IDX  = {'solve': 0, 'verify': 1, 'correct': 2}
PHASES = ['solve', 'verify', 'correct']

domain = '${domain}'
phase  = ${phase}
epochs = ${EPOCHS}

with open('${CONFIG}') as f:
    cfg_dict = yaml.safe_load(f)
config = PRISMConfig(**cfg_dict)

model = PRISMModel(config)._load_backbone(device='cuda:0')
model = model.to('cuda:0')

tok = AutoTokenizer.from_pretrained('Qwen/Qwen3.5-0.8B', trust_remote_code=True, cache_dir='.cache/huggingface')
tok.padding_side = 'left'
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

trainer = ExpertTrainer(
    prism_model=model,
    domain_idx=DOMAIN_TO_IDX[domain],
    phase_idx=phase,
    tokenizer=tok,
    device='cuda:0',
    lr=5e-5,
    warmup_steps=10,
    output_dir='${OUTPUT_DIR}',
    grad_accum_steps=4,
)

trace_file = '${TRACE_FILE[$domain]}'
history = trainer.train(trace_file=trace_file, epochs=epochs, batch_size=4)
print(f'DONE {domain} phase{phase}: final_loss={history[-1][\"loss\"]:.4f}')
" > "$logfile" 2>&1
    echo "[$(date)] Done:     ${domain} phase=${phase} — $(tail -1 $logfile)"
}

# ─── PHASE 0 (Solve) ─────────────────────────────────────────────────────────
echo "[$(date)] --- Phase 0: Solve ---"
# Launch algebra, geometry, combinatorics, NT in parallel
for domain in algebra geometry combinatorics number_theory; do
    train_block "$domain" 0 "${GPU_ASSIGN[$domain]}" &
done
wait
echo "[$(date)] Phase 0 batch 1 done (algebra, geometry, combinatorics, NT)"

# Miscellaneous on GPU 0 (sequential)
train_block miscellaneous 0 0
echo "[$(date)] Phase 0 complete"

# ─── PHASE 1 (Verify) ────────────────────────────────────────────────────────
echo "[$(date)] --- Phase 1: Verify ---"
for domain in algebra geometry combinatorics number_theory; do
    train_block "$domain" 1 "${GPU_ASSIGN[$domain]}" &
done
wait
train_block miscellaneous 1 0
echo "[$(date)] Phase 1 complete"

# ─── PHASE 2 (Correct) ───────────────────────────────────────────────────────
echo "[$(date)] --- Phase 2: Correct ---"
for domain in algebra geometry combinatorics number_theory; do
    train_block "$domain" 2 "${GPU_ASSIGN[$domain]}" &
done
wait
train_block miscellaneous 2 0
echo "[$(date)] Phase 2 complete"

echo ""
echo "[$(date)] === ALL 15 EXPERT BLOCKS TRAINED ==="
ls -la "$OUTPUT_DIR"/*.pt 2>/dev/null | awk '{print $9, $5}' || echo "(no .pt files found)"
