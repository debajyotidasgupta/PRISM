#!/usr/bin/env bash
# Regenerate geometry pilot traces with the fixed chat-completions API
# (thinking chain now separated from answer), then retrain geometry expert
# blocks and re-run joint fine-tuning.
#
# Root cause of geometry regression (A5: 0% vs baseline 1%):
#   VLLMServerGenerator used raw completions API → Qwen3 thinking chain
#   was mixed into output text, consuming all 4096 tokens with meta-commentary
#   ("Here's a thinking process...", "Analyze the Request: Role: Expert Geometer...")
#   instead of actual math. The geometry expert learned meta-vocabulary, not geometry.
#   Fix: chat completions API routes thinking to reasoning_content; content = clean answer.
#
# Usage: (vLLM server must already be running)
#   bash scripts/start_vllm_servers.sh   # if not already up
#   bash scripts/regen_geometry_pilot.sh

set -e
cd "$(dirname "$0")/.."
source /users/dasgupta/miniconda3/bin/activate /iopsstor/scratch/cscs/dasgupta/research/ideas/PRISM/.venv

PILOT_DIR="results/traces/pilot"
EXPERT_DIR="results/pilot/expert_blocks"
ROUTER_PT="results/pilot/router/router_final.pt"
JFT_DIR="results/pilot/joint_ft"
EVAL_OUT="results/pilot/ablation/A5v2_joint_ft_eval.json"
CONFIG="configs/model/prism_0.8b.yaml"
LOG_DIR="results/logs"
SERVER_URL="http://localhost:8000"

mkdir -p "$LOG_DIR" "$EXPERT_DIR" "$JFT_DIR" results/pilot/ablation

echo "[$(date)] === Geometry Pilot Regeneration ==="
echo "  Traces → $PILOT_DIR/geometry_traces.jsonl"
echo "  Server → $SERVER_URL"
echo ""

# ── Smoke test: verify server is up and chat completions API works ────────────
echo "[$(date)] Smoke test: verifying chat completions API + thinking separation..."
python -u -c "
import sys, os, asyncio, json
sys.path.insert(0, 'src')
os.environ.setdefault('HF_HOME', '.cache/huggingface')
from openai import AsyncOpenAI

async def test():
    import urllib.request
    with urllib.request.urlopen('${SERVER_URL}/v1/models', timeout=10) as r:
        model_id = json.loads(r.read())['data'][0]['id']
    print(f'Server: {model_id}')

    client = AsyncOpenAI(base_url='${SERVER_URL}/v1', api_key='dummy', timeout=120.0)
    resp = await client.chat.completions.create(
        model=model_id,
        messages=[
            {'role': 'system', 'content': 'You are a math expert. Be brief.'},
            {'role': 'user', 'content': 'What is 2+2? Give only the number.'},
        ],
        max_tokens=512,
        temperature=0.1,
        extra_body={'chat_template_kwargs': {'enable_thinking': False}},
    )
    msg = resp.choices[0].message
    has_rc = hasattr(msg, 'reasoning_content') and msg.reasoning_content is not None
    print(f'reasoning_content present: {has_rc}')
    print(f'content: {repr((msg.content or \"\")[:200])}')
    if has_rc:
        print(f'reasoning_content (first 100): {repr((msg.reasoning_content or \"\")[:100])}')
    if '</think>' in (msg.content or ''):
        print('WARNING: </think> leaked into content — fallback splitting will apply')
    else:
        print('OK: thinking properly separated from answer')

asyncio.run(test())
" || { echo "ERROR: Server not reachable or chat completions failed. Start with: bash scripts/start_vllm_servers.sh"; exit 1; }
echo ""

# ── Step 1: Regenerate geometry pilot traces ──────────────────────────────────
echo "[$(date)] Step 1: Generating 100 geometry traces via chat completions API..."
python -u -c "
import sys, os, json, logging
sys.path.insert(0, 'src')
os.environ.setdefault('HF_HOME', '.cache/huggingface')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

from datasets import load_dataset
from prism.generation.trace_generator import VLLMServerGenerator

# Load 100 geometry problems from MATH-500
ds = load_dataset('HuggingFaceH4/MATH-500', split='test', cache_dir='.cache/huggingface')
geo_problems = [ex for ex in ds if ex.get('subject','').lower() in ('geometry',)]
# Pad with other problems if needed (shouldn't be necessary)
problems = geo_problems[:100]
logging.info(f'Loaded {len(problems)} geometry problems')

# Format as expected by generate_dataset
formatted = [
    {
        'problem': ex['problem'],
        'solution': ex.get('solution', ''),
        'answer': ex['answer'],
        'id': ex.get('unique_id', str(i)),
    }
    for i, ex in enumerate(problems)
]

gen = VLLMServerGenerator(
    teacher_model_name='Qwen/Qwen3.5-35B-A3B',
    server_url='${SERVER_URL}',
    max_new_tokens=4096,
    concurrency=200,
    negative_fraction=0.3,
).load()

stats = gen.generate_dataset(
    problems=formatted,
    domain='geometry',
    output_file='${PILOT_DIR}/geometry_traces.jsonl',
)
print(json.dumps(stats, indent=2))
logging.info(f'Geometry traces regenerated: {stats[\"kept\"]} kept / {stats[\"total\"]} total')
" 2>&1 | tee "$LOG_DIR/regen_geometry_traces.log"
echo "[$(date)] Trace gen done."

# ── Quick contamination check ─────────────────────────────────────────────────
echo ""
echo "[$(date)] Contamination check on new traces:"
python -c "
import json
traces = [json.loads(l) for l in open('${PILOT_DIR}/geometry_traces.jsonl')]
meta_kw = ['Thinking Process', 'Analyze the Request', 'user wants', 'thinking process', 'I need to']
st_meta = sum(1 for t in traces if any(k.lower() in t.get('solve_trace','').lower() for k in meta_kw))
ct_meta = sum(1 for t in traces if any(k.lower() in t.get('correct_trace','').lower() for k in meta_kw))
print(f'  Total traces: {len(traces)}')
print(f'  solve_trace contaminated:   {st_meta}/{len(traces)} ({st_meta*100//max(len(traces),1)}%)')
print(f'  correct_trace contaminated: {ct_meta}/{len(traces)} ({ct_meta*100//max(len(traces),1)}%)')
# Show first solve_trace snippet
if traces:
    print(f'  First solve_trace start: {repr(traces[0][\"solve_trace\"][:150])}')
"
echo ""

# ── Step 2: Retrain geometry expert blocks (phases 0, 1, 2) ──────────────────
echo "[$(date)] Step 2: Retraining geometry expert blocks (3 phases × GPU 1)..."

retrain_geo() {
    local phase=$1
    local logfile="$LOG_DIR/regen_geo_expert_phase${phase}.log"
    echo "[$(date)]   Phase $phase → $logfile"
    CUDA_VISIBLE_DEVICES=1 python -u -c "
import sys, os
sys.path.insert(0, 'src')
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ.setdefault('HF_HOME', '.cache/huggingface')
import torch, yaml, logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
from prism.model.config import PRISMConfig
from prism.model.prism_model import PRISMModel
from prism.training.train_expert import ExpertBlockTrainer
from transformers import AutoTokenizer

with open('${CONFIG}') as f:
    config = PRISMConfig(**__import__('yaml').safe_load(f))

model = PRISMModel(config)
model._load_backbone(device='cuda:0')
model = model.to('cuda:0')

tok = AutoTokenizer.from_pretrained('Qwen/Qwen3.5-0.8B', trust_remote_code=True, cache_dir='.cache/huggingface')
tok.padding_side = 'left'
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

trainer = ExpertBlockTrainer(
    model=model,
    tokenizer=tok,
    device='cuda:0',
    lr=3e-4,
    output_dir='${EXPERT_DIR}',
)
history = trainer.train(
    domain='geometry',
    phase=$phase,
    trace_file='${PILOT_DIR}/geometry_traces.jsonl',
    epochs=3,
    batch_size=4,
    max_length=512,
)
print(f'geometry phase$phase done. Losses: {[round(e[\"loss\"],4) for e in history]}')
" > "$logfile" 2>&1
    echo "[$(date)]   Phase $phase done: $(tail -1 $logfile)"
}

# Phases must be sequential (phase 1 uses phase 0's block in forward)
retrain_geo 0
retrain_geo 1
retrain_geo 2
echo "[$(date)] Geometry expert blocks retrained."

# ── Step 3: Re-run joint fine-tuning with new geometry blocks ─────────────────
echo ""
echo "[$(date)] Step 3: Re-running joint fine-tuning with updated geometry blocks..."

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

with open('${CONFIG}') as f:
    config = PRISMConfig(**__import__('yaml').safe_load(f))

model = PRISMModel(config)._load_backbone(device='cuda:0').to('cuda:0')

# Load ALL expert blocks (including newly retrained geometry)
n_loaded = 0
for pi in range(config.n_phases):
    for di, domain in enumerate(config.domains):
        pt = f'${EXPERT_DIR}/phase{pi}_{domain}_final.pt'
        if os.path.exists(pt):
            model.expert_blocks[pi][di].load_state_dict(torch.load(pt, map_location='cuda:0'))
            n_loaded += 1
logging.info(f'Loaded {n_loaded}/15 expert blocks (geometry freshly retrained)')

if os.path.exists('${ROUTER_PT}'):
    model.router.load_state_dict(torch.load('${ROUTER_PT}', map_location='cuda:0'))
    logging.info('Loaded router checkpoint')

tok = AutoTokenizer.from_pretrained('Qwen/Qwen3.5-0.8B', trust_remote_code=True, cache_dir='.cache/huggingface')
tok.padding_side = 'left'
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

trainer = JointFinetuneTrainer(
    prism_model=model,
    tokenizer=tok,
    device='cuda:0',
    lr=2e-5,
    router_lr=5e-5,
    warmup_steps=10,
    output_dir='${JFT_DIR}',
    grad_accum_steps=8,
    entropy_weight=0.02,
)
history = trainer.train(
    trace_dirs=['${PILOT_DIR}'],
    epochs=3,
    batch_size=1,
    max_length=1024,
)
print(f'Joint FT v2 complete. Losses: {[round(e[\"loss\"],4) for e in history]}')
" 2>&1 | tee "$LOG_DIR/joint_finetune_v2.log"

echo "[$(date)] Joint fine-tuning v2 done. Running eval..."

# ── Step 4: Evaluate A5v2 ─────────────────────────────────────────────────────
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
    cfg_dict = __import__('yaml').safe_load(f)
cfg_dict['phase_aggregate_mode'] = 'mean'
config = PRISMConfig(**cfg_dict)

model = PRISMModel(config)._load_backbone(device='cuda:0').to('cuda:0')

jft_dir = '${JFT_DIR}'
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
logging.info('Loaded all A5v2 checkpoints')

model._residual_alpha = 1.0
model._disable_norm_stabilize = False
model._disable_crossmix = False
model._disable_crossphase = False

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
    'ablation': 'A5v2_joint_ft_clean_geo',
    'flags': {'joint_finetuned': True, 'crossphase': True, 'crossmix': True,
              'agg_mode': 'mean', 'alpha': 1.0, 'geometry_retrained_clean': True},
    **metrics,
}
with open('${EVAL_OUT}', 'w') as f:
    json.dump(result, f, indent=2)
print(json.dumps(result, indent=2))
print(f'A5v2: {metrics[\"overall\"]*100:.1f}% overall')
for d in DOMAINS:
    c = metrics['correct'][d]; n = metrics['total'][d]
    print(f'  {d}: {c}/{n} = {c/max(n,1)*100:.0f}%')
" 2>&1 | tee -a "$LOG_DIR/joint_finetune_v2.log"

echo ""
echo "[$(date)] === A5v2 COMPLETE ==="
echo "  Results: ${EVAL_OUT}"
echo "  Compare: A5 original 1.6% (geo 0/25), baseline 0.8% (geo 1/25)"
python -c "
import json
r = json.load(open('${EVAL_OUT}'))
print(f'  A5v2: {r[\"overall\"]*100:.1f}%  |  ', end='')
for d, v in r['domain'].items():
    print(f'{d}={v*100:.0f}%', end='  ')
print()
" 2>/dev/null || true
