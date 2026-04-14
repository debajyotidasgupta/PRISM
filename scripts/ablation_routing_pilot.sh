#!/usr/bin/env bash
# Run routing ablations on the 100-sample pilot expert blocks.
#
# Ablation matrix:
#   A1: Uniform routing (no router — equal 1/5 weights, new phase-agg forward)
#   A2: Trained router + new phase-agg forward (fixes w_phase bug)
#   A3: Trained router + new forward + CrossPhase enabled
#   A4: Hard routing (argmax — ablate soft vs hard)
#
# Each ablation evaluates 25 problems/domain (125 total) on MATH-500
# using the same pilot expert blocks from results/pilot/expert_blocks/
#
# Output: results/pilot/ablation/{A1,A2,A3,A4}_eval.json
#
# Usage: bash scripts/ablation_routing_pilot.sh

set -e
cd "$(dirname "$0")/.."
source /users/dasgupta/miniconda3/bin/activate /iopsstor/scratch/cscs/dasgupta/research/ideas/PRISM/.venv

EXPERT_DIR="results/pilot/expert_blocks"
ROUTER_PT="results/pilot/router/router_final.pt"
CONFIG="configs/model/prism_0.8b.yaml"
OUT_DIR="results/pilot/ablation"
LOG_DIR="results/logs"
N_PER_DOMAIN=25
MAX_NEW_TOKENS=128    # keep same as baseline eval for apples-to-apples comparison

mkdir -p "$OUT_DIR" "$LOG_DIR"

echo "[$(date)] === PRISM Routing Ablation Study ==="
echo "  Expert blocks: $EXPERT_DIR"
echo "  Router: $ROUTER_PT"
echo "  N/domain: $N_PER_DOMAIN, max_new_tokens: $MAX_NEW_TOKENS"
echo ""

run_ablation() {
    local NAME=$1
    local USE_UNIFORM=$2      # true/false
    local USE_HARD=$3         # true/false
    local USE_CROSSPHASE=$4   # true/false
    local ROUTER_FILE=$5      # path or "none"
    local AGG_MODE=$6         # "mean" or "last"
    local GPU=${7:-0}          # GPU to use
    local DISABLE_CROSSMIX=${8:-True}  # True = safer for ablation (CrossMix untrained)

    local LOG="$LOG_DIR/${NAME}_eval.log"
    local OUT="$OUT_DIR/${NAME}_eval.json"

    echo "[$(date)] Starting ablation: $NAME (GPU $GPU, crossmix_disabled=${DISABLE_CROSSMIX})"
    CUDA_VISIBLE_DEVICES=$GPU python -u -c "
import sys, os, json
sys.path.insert(0, 'src')
os.environ.setdefault('HF_HOME', '.cache/huggingface')
import torch, yaml, logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

from prism.model.config import PRISMConfig
from prism.model.prism_model import PRISMModel
from prism.eval.math_eval import evaluate_model, DOMAINS
from transformers import AutoTokenizer

# ── Load model ──────────────────────────────────────────────────────────────
with open('${CONFIG}') as f:
    cfg_dict = yaml.safe_load(f)
cfg_dict['phase_aggregate_mode'] = '${AGG_MODE}'
config = PRISMConfig(**cfg_dict)

model = PRISMModel(config)._load_backbone(device='cuda:0')
model = model.to('cuda:0')

# ── Load expert blocks ───────────────────────────────────────────────────────
expert_dir = '${EXPERT_DIR}'
n_loaded = 0
for pi in range(config.n_phases):
    for di, domain in enumerate(config.domains):
        pt = f'{expert_dir}/phase{pi}_{domain}_final.pt'
        if os.path.exists(pt):
            model.expert_blocks[pi][di].load_state_dict(torch.load(pt, map_location='cuda:0'))
            n_loaded += 1
logging.info(f'Loaded {n_loaded}/15 expert blocks')

# ── Load router ─────────────────────────────────────────────────────────────
router_file = '${ROUTER_FILE}'
if router_file != 'none' and os.path.exists(router_file):
    model.router.load_state_dict(torch.load(router_file, map_location='cuda:0'))
    logging.info(f'Loaded router from {router_file}')

# ── Set ablation flags ────────────────────────────────────────────────────────
# CrossMix is disabled for A1-A4 (untrained random weights corrupt direction).
# Residual alpha=0.3: expert output blended 30/70 with original h_K.
#   Expert blocks trained on 100 samples produce partially random directions.
#   30% expert + 70% h_K keeps generation coherent while preserving routing signal.
#   alpha→1.0 after joint fine-tuning (A5) trains the full model end-to-end.
model._use_uniform_routing = ${USE_UNIFORM}
model._use_hard_routing = ${USE_HARD}
model._disable_crossphase = not ${USE_CROSSPHASE}
model._disable_crossmix = ${DISABLE_CROSSMIX}
model._residual_alpha = 0.3   # blend: 30% expert, 70% h_K
logging.info(f'uniform={model._use_uniform_routing}, hard={model._use_hard_routing}, crossphase={not model._disable_crossphase}, crossmix={not model._disable_crossmix}, alpha=0.3, agg=${AGG_MODE}')

tok = AutoTokenizer.from_pretrained('Qwen/Qwen3.5-0.8B', trust_remote_code=True, cache_dir='.cache/huggingface')
tok.padding_side = 'left'
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# ── Evaluate with fixed answer extraction and subject-based classification ───
metrics = evaluate_model(
    model=model, tokenizer=tok,
    n_per_domain=${N_PER_DOMAIN}, max_new_tokens=${MAX_NEW_TOKENS},
    device='cuda:0', log_fn=logging.info,
)

result = {
    'ablation': '${NAME}',
    'flags': {'uniform_routing': ${USE_UNIFORM}, 'hard_routing': ${USE_HARD}, 'crossphase': ${USE_CROSSPHASE}, 'crossmix': not ${DISABLE_CROSSMIX}, 'agg_mode': '${AGG_MODE}'},
    **metrics,
}
with open('${OUT}', 'w') as f:
    json.dump(result, f, indent=2)

print(json.dumps(result, indent=2))
print(f'\\nAblation ${NAME}: {metrics[\"overall\"]*100:.1f}%')
for d in DOMAINS:
    c = metrics['correct'][d]; n = metrics['total'][d]
    print(f'  {d}: {c}/{n} = {c/max(n,1)*100:.0f}%')
" > "$LOG" 2>&1
    local RESULT=$(python -c "import json; d=json.load(open('${OUT}')); print(f'{d[\"overall\"]*100:.1f}%')" 2>/dev/null || echo "FAILED")
    echo "[$(date)] Done: $NAME → $RESULT  (log: $LOG)"
}
export -f run_ablation

# Run all 4 ablations in parallel across 4 GPUs
# NOTE: use "last" agg mode for pre-trained blocks — they were trained with
# old single-phase final aggregation. "mean" only correct after joint fine-tuning.
#
# CrossMix is DISABLED (arg 8 = True) for A1-A4.
# CrossMix was never trained — its random weights corrupt h_K_prime's direction
# even after norm stabilization (all 5 domains attend to each other via random
# Q/K/V projections, producing incoherent cross-domain information).
# Expert blocks alone (residual: h_out ≈ h_K + learned correction) are well-
# behaved. CrossMix is enabled in A5 after joint fine-tuning trains it.

# A1: Uniform routing + no CrossPhase
# Tests: expert blocks with equal weights — pure expert contribution without routing
run_ablation "A1_uniform" "True" "False" "False" "none" "last" 0 "True" &

# A2: Trained router, no CrossPhase
# Tests: does routing signal (trained on 100 samples) improve expert utilization?
run_ablation "A2_trained_router" "False" "False" "False" "$ROUTER_PT" "last" 1 "True" &

# A3: Trained router + CrossPhase
# Tests: does cross-phase temporal attention add value on top of routing?
run_ablation "A3_crossphase" "False" "False" "True" "$ROUTER_PT" "last" 2 "True" &

# A4: Hard routing (argmax) — ablate soft vs hard routing
# Tests: is soft routing better than committing to a single expert per phase?
run_ablation "A4_hard_routing" "False" "True" "False" "$ROUTER_PT" "last" 3 "True" &

echo "[$(date)] All 4 ablations launched in parallel (GPU 0-3). Waiting..."
wait
echo "[$(date)] All ablations complete."

# ─── Summary ─────────────────────────────────────────────────────────────────
echo ""
echo "[$(date)] === ABLATION SUMMARY ==="
echo "  Baseline Qwen3.5-0.8B: ~4.4%"
echo "  Prior PRISM (random router, old forward): 2.4%"
echo "  Prior PRISM (trained router, old forward): 2.4%"
echo ""
for NAME in A1_uniform A2_trained_router A3_crossphase A4_hard_routing; do
    F="$OUT_DIR/${NAME}_eval.json"
    if [ -f "$F" ]; then
        RESULT=$(python -c "import json; d=json.load(open('$F')); print(f'{d[\"ablation\"]}: {d[\"overall\"]*100:.1f}%  |  ' + '  '.join(f'{k}={v*100:.0f}%' for k,v in d['domain'].items()))" 2>/dev/null)
        echo "  $RESULT"
    else
        echo "  $NAME: MISSING"
    fi
done
echo ""
echo "[$(date)] Results saved to $OUT_DIR/"
