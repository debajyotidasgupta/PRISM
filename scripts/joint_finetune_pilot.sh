#!/usr/bin/env bash
# Joint end-to-end fine-tuning of all PRISM modules on pilot traces.
#
# Loads pretrained expert blocks + router, then trains ALL modules jointly
# with LM loss flowing through the full 3-phase chain. This fixes:
#   1. Phase mismatch from sequential block training
#   2. Router gets gradient from task performance, not domain classification
#   3. CrossPhase learns temporal attention between phase outputs
#
# Uses pilot 100-sample traces (500 total examples across all domains).
# Runs on 1 GPU (GPU 0) — full PRISM forward is memory intensive.
#
# Usage: bash scripts/joint_finetune_pilot.sh

set -e
cd "$(dirname "$0")/.."
source /users/dasgupta/miniconda3/bin/activate /iopsstor/scratch/cscs/dasgupta/research/ideas/PRISM/.venv

EXPERT_DIR="results/pilot/expert_blocks"
ROUTER_PT="results/pilot/router/router_final.pt"
TRACE_DIR="results/traces/pilot"
CONFIG="configs/model/prism_0.8b.yaml"
OUT_DIR="results/pilot/joint_ft"
LOG="results/logs/joint_finetune_pilot.log"
EVAL_OUT="results/pilot/ablation/A5_joint_ft_eval.json"

mkdir -p "$OUT_DIR" results/logs results/pilot/ablation

echo "[$(date)] === PRISM Joint Fine-tuning (Pilot) ==="
echo "  Expert blocks: $EXPERT_DIR"
echo "  Router: $ROUTER_PT"
echo "  Traces: $TRACE_DIR"
echo "  Output: $OUT_DIR"
echo ""

CUDA_VISIBLE_DEVICES=0 python -u -c "
import sys, os, json
sys.path.insert(0, 'src')
os.environ.setdefault('HF_HOME', '.cache/huggingface')
import torch, yaml, logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

from prism.model.config import PRISMConfig
from prism.model.prism_model import PRISMModel
from prism.training.joint_finetune import JointFinetuneTrainer
from transformers import AutoTokenizer

# ── Load model ───────────────────────────────────────────────────────────────
with open('${CONFIG}') as f:
    cfg_dict = yaml.safe_load(f)
config = PRISMConfig(**cfg_dict)

model = PRISMModel(config)._load_backbone(device='cuda:0')
model = model.to('cuda:0')

# ── Load pretrained expert blocks ─────────────────────────────────────────────
n_loaded = 0
for pi in range(config.n_phases):
    for di, domain in enumerate(config.domains):
        pt = f'${EXPERT_DIR}/phase{pi}_{domain}_final.pt'
        if os.path.exists(pt):
            model.expert_blocks[pi][di].load_state_dict(torch.load(pt, map_location='cuda:0'))
            n_loaded += 1
logging.info(f'Loaded {n_loaded}/15 expert blocks')

# ── Load router checkpoint ────────────────────────────────────────────────────
if os.path.exists('${ROUTER_PT}'):
    model.router.load_state_dict(torch.load('${ROUTER_PT}', map_location='cuda:0'))
    logging.info('Loaded router checkpoint')

tok = AutoTokenizer.from_pretrained('Qwen/Qwen3.5-0.8B', trust_remote_code=True, cache_dir='.cache/huggingface')
tok.padding_side = 'left'
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# ── Train ─────────────────────────────────────────────────────────────────────
trainer = JointFinetuneTrainer(
    prism_model=model,
    tokenizer=tok,
    device='cuda:0',
    lr=2e-5,
    router_lr=5e-5,
    warmup_steps=10,
    output_dir='${OUT_DIR}',
    grad_accum_steps=8,
    entropy_weight=0.02,
)

history = trainer.train(
    trace_dirs=['${TRACE_DIR}'],
    epochs=3,
    batch_size=1,        # full 3-phase forward is memory intensive
    max_length=1024,     # truncate for pilot
)
print(f'Joint FT complete. Final loss: {history[-1][\"loss\"]:.4f}')
" > "$LOG" 2>&1

echo "[$(date)] Joint fine-tuning done. Running eval..."
tail -5 "$LOG"

# ─── Evaluate the jointly fine-tuned model ────────────────────────────────────
# Uses evaluate_model() from math_eval.py for consistency with A1-A4 ablations.
# Joint FT uses full expert output (alpha=1.0, no residual blending needed since
# CrossMix is now trained) and mean agg mode (all 3 phase aggregates contribute).
CUDA_VISIBLE_DEVICES=0 python -u -c "
import sys, os, json
sys.path.insert(0, 'src')
os.environ.setdefault('HF_HOME', '.cache/huggingface')
import torch, yaml, logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

from prism.model.config import PRISMConfig
from prism.model.prism_model import PRISMModel
from prism.eval.math_eval import evaluate_model, DOMAINS
from transformers import AutoTokenizer

with open('${CONFIG}') as f:
    cfg_dict = yaml.safe_load(f)
cfg_dict['phase_aggregate_mode'] = 'mean'   # mean: all 3 phases contribute after joint FT
config = PRISMConfig(**cfg_dict)

model = PRISMModel(config)._load_backbone(device='cuda:0')
model = model.to('cuda:0')

# Load all jointly fine-tuned weights (expert blocks + router + CrossPhase + CrossMix)
jft_dir = '${OUT_DIR}'
model.router.load_state_dict(torch.load(f'{jft_dir}/router_final.pt', map_location='cuda:0'))
model.cross_phase.load_state_dict(torch.load(f'{jft_dir}/cross_phase_final.pt', map_location='cuda:0'))
for pi in range(config.n_phases):
    for di, domain in enumerate(config.domains):
        pt = f'{jft_dir}/phase{pi}_{domain}_final.pt'
        if os.path.exists(pt):
            model.expert_blocks[pi][di].load_state_dict(torch.load(pt, map_location='cuda:0'))
for i in range(config.n_phases):
    pt = f'{jft_dir}/crossmix_{i}_final.pt'
    if os.path.exists(pt):
        model.cross_mix[i].load_state_dict(torch.load(pt, map_location='cuda:0'))
logging.info('Loaded all jointly fine-tuned checkpoints (expert blocks + router + CrossPhase + CrossMix)')

# Full expert output: joint FT trains all modules e2e.
# Norm stabilization MUST remain ON during eval to match training conditions —
# training had norm_stabilize=True (default). Expert blocks learn to produce meaningful
# DIRECTIONS under norm constraint; disabling it at eval time changes the distribution.
model._residual_alpha = 1.0
model._disable_norm_stabilize = False  # KEEP ON: matches training conditions
model._disable_crossmix = False        # CrossMix trained in joint FT
model._disable_crossphase = False      # CrossPhase trained in joint FT

tok = AutoTokenizer.from_pretrained('Qwen/Qwen3.5-0.8B', trust_remote_code=True, cache_dir='.cache/huggingface')
tok.padding_side = 'left'
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

metrics = evaluate_model(
    model=model, tokenizer=tok,
    n_per_domain=25, max_new_tokens=256,
    device='cuda:0', log_fn=logging.info,
)

result = {
    'ablation': 'A5_joint_ft',
    'flags': {'joint_finetuned': True, 'crossphase': True, 'crossmix': True, 'agg_mode': 'mean', 'alpha': 1.0},
    **metrics,
}
with open('${EVAL_OUT}', 'w') as f:
    json.dump(result, f, indent=2)

print(json.dumps(result, indent=2))
print(f'\\nA5 Joint FT: {metrics[\"overall\"]*100:.1f}% overall')
for d in DOMAINS:
    c = metrics['correct'][d]; n = metrics['total'][d]
    print(f'  {d}: {c}/{n} = {c/max(n,1)*100:.0f}%')
" >> "$LOG" 2>&1

echo "[$(date)] Eval complete."
tail -10 "$LOG"
