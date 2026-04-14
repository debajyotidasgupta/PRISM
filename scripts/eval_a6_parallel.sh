#!/usr/bin/env bash
# Evaluate A6 clean pilot JFT across all 4 GPUs in parallel.
# Each GPU handles 1/4 of the 125 problems → 4× speedup.
# Results are merged at the end.
set -e
cd "$(dirname "$0")/.."
source /users/dasgupta/miniconda3/bin/activate /iopsstor/scratch/cscs/dasgupta/research/ideas/PRISM/.venv

JFT_DIR="results/pilot/joint_ft_v2"
EVAL_OUT="results/pilot/ablation/A8_clean_pilot_jft_fullfix_eval.json"
CONFIG="configs/model/prism_0.8b.yaml"
LOG_DIR="results/logs"
SHARD_DIR="/tmp/a6_shards"

mkdir -p "$SHARD_DIR" "$LOG_DIR" results/pilot/ablation

echo "[$(date)] === A6 Parallel Evaluation (4 GPUs) ==="
echo "  JFT checkpoint: $JFT_DIR"
echo "  Output: $EVAL_OUT"
echo ""

# ── Run one shard on one GPU ─────────────────────────────────────────────────
run_shard() {
    local gpu=$1
    local shard_idx=$2      # 0-3
    local n_shards=4
    local out="${SHARD_DIR}/shard_${shard_idx}.json"
    local log="${LOG_DIR}/a6_eval_gpu${gpu}.log"

    echo "[$(date)] GPU $gpu: shard $shard_idx/$n_shards → $log"

    CUDA_VISIBLE_DEVICES=$gpu python -u -c "
import sys, os, json, torch, yaml, logging
sys.path.insert(0, 'src')
os.environ.setdefault('HF_HOME', '.cache/huggingface')
os.environ['CUDA_VISIBLE_DEVICES'] = '$gpu'
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

from prism.model.config import PRISMConfig
from prism.model.prism_model import PRISMModel
from prism.eval.math_eval import (
    load_math500_by_domain, extract_boxed, answers_match,
    DOMAINS, SUBJECT_TO_DOMAIN, _dedup_repetition
)
from transformers import AutoTokenizer

# ── Load model ────────────────────────────────────────────────────────────────
with open('${CONFIG}') as f:
    cfg_dict = yaml.safe_load(f)
cfg_dict['phase_aggregate_mode'] = 'mean'
config = PRISMConfig(**cfg_dict)

model = PRISMModel(config)
model._load_backbone(device='cuda:0')
model = model.to('cuda:0')

jft = '${JFT_DIR}'
model.router.load_state_dict(torch.load(f'{jft}/router_final.pt', map_location='cuda:0'))
model.cross_phase.load_state_dict(torch.load(f'{jft}/cross_phase_final.pt', map_location='cuda:0'))
for pi in range(config.n_phases):
    for di, domain in enumerate(config.domains):
        pt = f'{jft}/phase{pi}_{domain}_final.pt'
        if os.path.exists(pt):
            model.expert_blocks[pi][di].load_state_dict(torch.load(pt, map_location='cuda:0'))
for i in range(config.n_phases):
    pt = f'{jft}/crossmix_{i}_final.pt'
    if os.path.exists(pt):
        model.cross_mix[i].load_state_dict(torch.load(pt, map_location='cuda:0'))

model._residual_alpha = 1.0
model._disable_norm_stabilize = False
model._disable_crossmix = False
model._disable_crossphase = False
logging.info('A6 checkpoints loaded on GPU $gpu (shard $shard_idx)')

tok = AutoTokenizer.from_pretrained('Qwen/Qwen3.5-0.8B', trust_remote_code=True, cache_dir='.cache/huggingface')
tok.padding_side = 'left'
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# ── Load & shard problems ─────────────────────────────────────────────────────
all_problems = load_math500_by_domain(n_per_domain=25)
n_shards = ${n_shards}
shard_idx = ${shard_idx}

# Distribute evenly
my_problems = all_problems[shard_idx::n_shards]
logging.info(f'Shard $shard_idx: {len(my_problems)} problems (indices {shard_idx}, {shard_idx}+{n_shards}, ...)')

# ── Evaluate ──────────────────────────────────────────────────────────────────
results = []
model.eval()
with torch.no_grad():
    for local_i, ex in enumerate(my_problems):
        subj = ex.get('subject', '')
        domain = SUBJECT_TO_DOMAIN.get(subj, 'miscellaneous')
        global_i = shard_idx + local_i * n_shards

        # Match training format exactly: raw problem + "\n\n" (no "Problem:"/"Solution:" prefix)
        prompt = ex['problem'] + '\n\n'
        enc = tok(prompt, return_tensors='pt', truncation=True, max_length=768, add_special_tokens=True).to('cuda:0')

        out_ids = model.generate(
            enc['input_ids'],
            attention_mask=enc['attention_mask'],
            max_new_tokens=1536,
            temperature=0.0,
            repetition_penalty=1.15,
        )
        generated = tok.decode(out_ids[0][enc['input_ids'].shape[1]:], skip_special_tokens=True)

        pred = extract_boxed(generated)
        gold = ex.get('answer', '')
        correct = answers_match(pred, gold)

        results.append({
            'global_idx': global_i,
            'domain': domain,
            'correct': correct,
            'pred': pred,
            'gold': gold,
            'generated_len': len(generated),
        })

        if (local_i + 1) % 8 == 0 or (local_i + 1) == len(my_problems):
            n_corr = sum(1 for r in results if r['correct'])
            logging.info(f'  GPU$gpu shard$shard_idx: {local_i+1}/{len(my_problems)} done — {n_corr}/{local_i+1} = {n_corr/(local_i+1)*100:.1f}%')

n_corr = sum(1 for r in results if r['correct'])
logging.info(f'GPU$gpu FINAL: {n_corr}/{len(my_problems)} = {n_corr/max(len(my_problems),1)*100:.1f}%')

with open('${out}', 'w') as f:
    json.dump(results, f, indent=2)
logging.info(f'Shard saved: ${out}')
" > "$log" 2>&1
    echo "[$(date)] GPU $gpu shard $shard_idx done. $(tail -2 $log)"
}

# ── Launch all 4 shards in parallel ──────────────────────────────────────────
run_shard 0 0 &
run_shard 1 1 &
run_shard 2 2 &
run_shard 3 3 &

echo "[$(date)] All 4 GPU shards launched in parallel. Waiting..."
wait
echo "[$(date)] All shards complete. Merging results..."

# ── Merge shard results ───────────────────────────────────────────────────────
python -u -c "
import json, os, sys
from pathlib import Path

shard_dir = '${SHARD_DIR}'
shards = []
for i in range(4):
    f = f'{shard_dir}/shard_{i}.json'
    if not os.path.exists(f):
        print(f'ERROR: missing shard {i}: {f}', file=sys.stderr)
        sys.exit(1)
    shards.extend(json.load(open(f)))

# Sort by global_idx to restore original order
shards.sort(key=lambda r: r['global_idx'])

DOMAINS = ['algebra', 'geometry', 'combinatorics', 'number_theory', 'miscellaneous']
correct_by_domain = {d: 0 for d in DOMAINS}
total_by_domain   = {d: 0 for d in DOMAINS}

for r in shards:
    d = r['domain']
    total_by_domain[d]   += 1
    correct_by_domain[d] += int(r['correct'])

total_correct = sum(correct_by_domain.values())
total         = sum(total_by_domain.values())
overall       = total_correct / max(total, 1)

result = {
    'ablation': 'A8_clean_pilot_jft_fullfix_eval',
    'flags': {
        'clean_traces': True,
        'expert_epochs': 3,
        'jft_epochs': 5,
        'total_traces': 341,
        'eval_gpus': 4,
        'max_new_tokens': 1536,
        'temperature': 0.0,
        'repetition_penalty': 1.15,
        'prompt_format': 'raw_problem_plus_newlines',
        'extract': 'phase_aware_last_section',
    },
    'overall': round(overall, 4),
    'domain': {d: round(correct_by_domain[d] / max(total_by_domain[d], 1), 4) for d in DOMAINS},
    'correct': correct_by_domain,
    'total': total_by_domain,
    'n_total': total,
}

with open('${EVAL_OUT}', 'w') as f:
    json.dump(result, f, indent=2)

print(json.dumps(result, indent=2))
print()
print(f'=== A6 CLEAN PILOT RESULT ===')
print(f'  Overall: {overall*100:.1f}%  ({total_correct}/{total})')
print()
for d in DOMAINS:
    c = correct_by_domain[d]; n = total_by_domain[d]
    pct = c/max(n,1)*100
    bar = '#'*c + '-'*(n-c)
    print(f'  {d:20s}: {c:2d}/{n} = {pct:5.1f}%  [{bar}]')
print()
print(f'  Baseline Qwen3.5-0.8B : 0.8%  (1/125)')
print(f'  A5 contaminated JFT   : 1.6%  (2/125)')
print(f'  A6 (256tok, greedy, wrong prompt)    : 3.2%  (4/125)')
print(f'  A7 (512tok, sampling, kwargs ignored): 1.6%  (2/125)')
print(f'  A8 (1536tok, greedy+rep_pen, fixed)  : {overall*100:.1f}%  ({total_correct}/{total})')
print(f'  vs baseline                          : {overall/0.008:.1f}x')
"

echo ""
echo "[$(date)] === A6 Evaluation Complete ==="
echo "  Results: ${EVAL_OUT}"
