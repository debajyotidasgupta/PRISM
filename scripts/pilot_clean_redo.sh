#!/usr/bin/env bash
# Full clean pilot redo — regenerate ALL 5 domain traces with fixed chat-completions
# API (thinking separated from answer), retrain all 15 expert blocks + router,
# then joint fine-tune and evaluate.
#
# All previous pilot traces were 100% contaminated: VLLMServerGenerator used raw
# completions API so Qwen3's thinking chain (2000-4000 tokens of meta-commentary)
# consumed the entire max_tokens budget, leaving no room for actual math content.
#
# With the fix, resp.choices[0].message.content = clean math answer only.
#
# Usage: (vLLM server must be running on port 8000)
#   bash scripts/start_vllm_servers.sh
#   bash scripts/pilot_clean_redo.sh
#
# Runtime estimate:
#   Trace gen (500 problems × 3 phases, 200 concurrent): ~30-60 min
#   Expert block training (15 blocks × 3 epochs):        ~60 min
#   Router training:                                      ~10 min
#   Joint fine-tuning (3 epochs):                        ~30 min
#   Evaluation (125 problems):                           ~75 min

set -e
cd "$(dirname "$0")/.."
source /users/dasgupta/miniconda3/bin/activate /iopsstor/scratch/cscs/dasgupta/research/ideas/PRISM/.venv

PILOT_DIR="results/traces/pilot"
EXPERT_DIR="results/pilot/expert_blocks"
ROUTER_DIR="results/pilot/router"
JFT_DIR="results/pilot/joint_ft_v2"
ABLATION_DIR="results/pilot/ablation"
LOG_DIR="results/logs"
CONFIG="configs/model/prism_0.8b.yaml"
SERVER_URL="http://localhost:8000"
EPOCHS=3
N_PILOT=100   # problems per domain

mkdir -p "$PILOT_DIR" "$EXPERT_DIR" "$ROUTER_DIR" "$JFT_DIR" "$ABLATION_DIR" "$LOG_DIR"

echo "========================================================"
echo "  PRISM Pilot Clean Redo — $(date)"
echo "  All 5 domains × 100 samples, clean chat-completions"
echo "========================================================"
echo ""

# ── 0. Smoke test ────────────────────────────────────────────────────────────
echo "[$(date)] Step 0: Verifying server + thinking separation..."
python -u -c "
import sys, os, asyncio, json
sys.path.insert(0, 'src')
os.environ.setdefault('HF_HOME', '.cache/huggingface')
from openai import AsyncOpenAI
import urllib.request

with urllib.request.urlopen('${SERVER_URL}/v1/models', timeout=10) as r:
    model_id = json.loads(r.read())['data'][0]['id']
print(f'  Server model: {model_id}')

async def test():
    client = AsyncOpenAI(base_url='${SERVER_URL}/v1', api_key='dummy', timeout=60.0)
    resp = await client.chat.completions.create(
        model=model_id,
        messages=[
            {'role': 'system', 'content': 'You are a math expert.'},
            {'role': 'user', 'content': 'What is 15 * 8? Give only the number.'},
        ],
        max_tokens=256,
        temperature=0.1,
        extra_body={'chat_template_kwargs': {'enable_thinking': False}},
    )
    msg = resp.choices[0].message
    has_rc = hasattr(msg, 'reasoning_content') and msg.reasoning_content is not None
    content = (msg.content or '').strip()
    print(f'  reasoning_content separated: {has_rc}')
    print(f'  content: {repr(content[:120])}')
    meta_kw = ['thinking process', 'analyze the request', 'reformulat']
    contaminated = any(k in content.lower() for k in meta_kw)
    if contaminated:
        print('  WARNING: meta-commentary still in content — check vLLM version')
    else:
        print('  OK: content is clean answer')
    assert not contaminated, 'Meta-commentary in content — aborting'

asyncio.run(test())
" || { echo "Smoke test failed. Check server."; exit 1; }
echo ""

# ── 1. Regenerate all 5 domain pilot traces ───────────────────────────────────
echo "[$(date)] Step 1: Regenerating all 5 domain pilot traces (chat completions API)..."

regen_domain() {
    local domain=$1
    local log="$LOG_DIR/regen_traces_${domain}.log"
    echo "  [$(date)] → $domain"
    python -u -c "
import sys, os, json, logging
sys.path.insert(0, 'src')
os.environ.setdefault('HF_HOME', '.cache/huggingface')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

from datasets import load_dataset
from prism.generation.trace_generator import VLLMServerGenerator

DOMAIN_TO_SUBJECT = {
    'algebra': ['algebra'],
    'geometry': ['geometry'],
    'combinatorics': ['counting_and_probability'],
    'number_theory': ['number_theory'],
    'miscellaneous': ['intermediate_algebra', 'prealgebra', 'precalculus'],
}
subjects = DOMAIN_TO_SUBJECT['$domain']

ds = load_dataset('HuggingFaceH4/MATH-500', split='test', cache_dir='.cache/huggingface')
problems_raw = [ex for ex in ds if ex.get('subject','').lower().replace(' ','_') in subjects]
if not problems_raw:
    # fallback: try direct match
    problems_raw = [ex for ex in ds if any(s in ex.get('subject','').lower() for s in subjects)]
import random; rng = random.Random(123)
rng.shuffle(problems_raw)
problems_raw = problems_raw[:${N_PILOT}]
logging.info(f'[$domain] {len(problems_raw)} problems (subjects={subjects})')

formatted = [
    {'problem': ex['problem'], 'solution': ex.get('solution',''),
     'answer': ex['answer'], 'id': ex.get('unique_id', str(i))}
    for i, ex in enumerate(problems_raw)
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
    domain='$domain',
    output_file='${PILOT_DIR}/${domain}_traces.jsonl',
)
logging.info(f'[$domain] Done: {stats[\"kept\"]} kept, phase1_correct={stats[\"phase1_correct\"]}')
print(json.dumps(stats, indent=2))
" > "$log" 2>&1
    local kept=$(python3 -c "import json; d=json.loads(open('$log').read().split('{',1)[1].rsplit('}',1)[0].join(['{}','{'])); print('?')" 2>/dev/null || grep -o '"kept": [0-9]*' "$log" | tail -1 | grep -o '[0-9]*')
    echo "  [$(date)] ✓ $domain ($(wc -l < ${PILOT_DIR}/${domain}_traces.jsonl 2>/dev/null || echo '?') traces)"
}

for domain in algebra geometry combinatorics number_theory miscellaneous; do
    regen_domain "$domain"
done

# Rebuild general traces (combined)
cat "$PILOT_DIR"/{algebra,geometry,combinatorics,number_theory,miscellaneous}_traces.jsonl \
    > "$PILOT_DIR/general_traces.jsonl"
echo "  [$(date)] Combined general_traces.jsonl: $(wc -l < $PILOT_DIR/general_traces.jsonl) traces"

# ── Validation: reject bad traces before any training ─────────────────────────
echo ""
echo "[$(date)] Validating regenerated traces (quality gate: <20% bad, ≥50 usable/domain)..."
python -u -m prism.data.validate_traces \
    --dir "${PILOT_DIR}" \
    --min-usable 50 \
    --max-bad-rate 0.20 \
    --verbose \
    2>&1 | tee "$LOG_DIR/trace_validation.log"

VALIDATE_EXIT=${PIPESTATUS[0]}
if [ $VALIDATE_EXIT -ne 0 ]; then
    echo ""
    echo "ERROR: Trace validation FAILED. Training aborted."
    echo "  Fix the trace generator and re-run. See $LOG_DIR/trace_validation.log"
    exit 1
fi
echo "[$(date)] Trace validation PASSED. Proceeding to training."
echo ""

# ── 2. Train all 15 expert blocks ────────────────────────────────────────────
echo "[$(date)] Step 2: Training all 15 expert blocks (5 domains × 3 phases)..."

train_expert() {
    local domain=$1
    local phase=$2
    local gpu=$3
    local log="$LOG_DIR/expert_${domain}_phase${phase}.log"
    CUDA_VISIBLE_DEVICES=$gpu python -u -c "
import sys, os
sys.path.insert(0, 'src')
os.environ['CUDA_VISIBLE_DEVICES'] = '$gpu'
os.environ.setdefault('HF_HOME', '.cache/huggingface')
import yaml, logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
from prism.model.config import PRISMConfig
from prism.model.prism_model import PRISMModel
from prism.training.train_expert import ExpertBlockTrainer
from transformers import AutoTokenizer

with open('${CONFIG}') as f:
    config = PRISMConfig(**yaml.safe_load(f))
model = PRISMModel(config)._load_backbone(device='cuda:0').to('cuda:0')
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3.5-0.8B', trust_remote_code=True, cache_dir='.cache/huggingface')
tok.padding_side = 'left'
if tok.pad_token is None: tok.pad_token = tok.eos_token

trainer = ExpertBlockTrainer(model=model, tokenizer=tok, device='cuda:0',
    lr=3e-4, output_dir='${EXPERT_DIR}')
history = trainer.train(
    domain='$domain', phase=$phase,
    trace_file='${PILOT_DIR}/${domain}_traces.jsonl',
    epochs=${EPOCHS}, batch_size=4, max_length=512,
)
losses = [round(e['loss'],4) for e in history]
print(f'${domain} phase$phase: {losses}')
" > "$log" 2>&1
    local result=$(tail -1 "$log" 2>/dev/null)
    echo "    ✓ $domain phase$phase (GPU $gpu): $result"
}

# Round A: 4 domains in parallel (GPU 0-3), phases 0→1→2 sequentially
for phase in 0 1 2; do
    echo "  [$(date)] Phase $phase — 4 domains parallel..."
    for d_gpu in "algebra 0" "geometry 1" "combinatorics 2" "number_theory 3"; do
        d=$(echo $d_gpu | cut -d' ' -f1)
        g=$(echo $d_gpu | cut -d' ' -f2)
        train_expert "$d" "$phase" "$g" &
    done
    wait
    echo "  [$(date)] Phase $phase (algebra/geometry/comb/NT) done."
    # miscellaneous on GPU 0 (sequential to avoid OOM)
    train_expert "miscellaneous" "$phase" "0"
done
echo "[$(date)] All 15 expert blocks trained."
echo ""

# ── 3. Train router ───────────────────────────────────────────────────────────
echo "[$(date)] Step 3: Training router on domain classification..."
mkdir -p "$ROUTER_DIR"
CUDA_VISIBLE_DEVICES=0 python -u -c "
import sys, os
sys.path.insert(0, 'src')
os.environ.setdefault('HF_HOME', '.cache/huggingface')
import yaml, logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
from prism.model.config import PRISMConfig
from prism.model.prism_model import PRISMModel
from prism.training.train_router import RouterTrainer
from transformers import AutoTokenizer

with open('${CONFIG}') as f:
    config = PRISMConfig(**yaml.safe_load(f))
model = PRISMModel(config)._load_backbone(device='cuda:0').to('cuda:0')
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3.5-0.8B', trust_remote_code=True, cache_dir='.cache/huggingface')
tok.padding_side = 'left'
if tok.pad_token is None: tok.pad_token = tok.eos_token

trainer = RouterTrainer(model=model, tokenizer=tok, device='cuda:0',
    lr=1e-3, output_dir='${ROUTER_DIR}')
history = trainer.train(
    trace_dirs=['${PILOT_DIR}'],
    epochs=5, batch_size=16, max_length=256,
)
losses = [round(e['loss'],4) for e in history]
accs   = [round(e.get('accuracy',0),4) for e in history]
print(f'Router: loss={losses}')
print(f'Router: accuracy={accs}')
" 2>&1 | tee "$LOG_DIR/router_v2.log"
echo "[$(date)] Router trained."
echo ""

# ── 4. Joint fine-tuning ──────────────────────────────────────────────────────
echo "[$(date)] Step 4: Joint fine-tuning (all 15 blocks + router + CrossMix + CrossPhase)..."
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
    config = PRISMConfig(**yaml.safe_load(f))
model = PRISMModel(config)._load_backbone(device='cuda:0').to('cuda:0')

# Load freshly trained expert blocks
n_loaded = 0
for pi in range(config.n_phases):
    for di, domain in enumerate(config.domains):
        pt = f'${EXPERT_DIR}/phase{pi}_{domain}_final.pt'
        if os.path.exists(pt):
            model.expert_blocks[pi][di].load_state_dict(torch.load(pt, map_location='cuda:0'))
            n_loaded += 1
logging.info(f'Loaded {n_loaded}/15 expert blocks (clean traces)')

router_pt = '${ROUTER_DIR}/router_final.pt'
if os.path.exists(router_pt):
    model.router.load_state_dict(torch.load(router_pt, map_location='cuda:0'))
    logging.info('Loaded router checkpoint')

tok = AutoTokenizer.from_pretrained('Qwen/Qwen3.5-0.8B', trust_remote_code=True, cache_dir='.cache/huggingface')
tok.padding_side = 'left'
if tok.pad_token is None: tok.pad_token = tok.eos_token

trainer = JointFinetuneTrainer(
    prism_model=model, tokenizer=tok, device='cuda:0',
    lr=2e-5, router_lr=5e-5, warmup_steps=20,
    output_dir='${JFT_DIR}',
    grad_accum_steps=8, entropy_weight=0.02,
)
history = trainer.train(
    trace_dirs=['${PILOT_DIR}'],
    epochs=5,          # more epochs on clean data
    batch_size=1,
    max_length=1024,
)
for e in history:
    logging.info(f'  Epoch {e[\"epoch\"]}: loss={e[\"loss\"]:.4f}')
print(f'Joint FT done. Final loss: {history[-1][\"loss\"]:.4f}')
" 2>&1 | tee "$LOG_DIR/joint_ft_v2.log"
echo "[$(date)] Joint fine-tuning done."
echo ""

# ── 5. Evaluate A6 (clean pilot joint FT) ────────────────────────────────────
echo "[$(date)] Step 5: Evaluating A6 clean pilot (125 MATH-500 problems, 25/domain)..."
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
cfg_dict['phase_aggregate_mode'] = 'mean'
config = PRISMConfig(**cfg_dict)

model = PRISMModel(config)._load_backbone(device='cuda:0').to('cuda:0')

jft = '${JFT_DIR}'
model.router.load_state_dict(torch.load(f'{jft}/router_final.pt', map_location='cuda:0'))
model.cross_phase.load_state_dict(torch.load(f'{jft}/cross_phase_final.pt', map_location='cuda:0'))
for pi in range(config.n_phases):
    for di, domain in enumerate(config.domains):
        pt = f'{jft}/phase{pi}_{domain}_final.pt'
        if os.path.exists(pt): model.expert_blocks[pi][di].load_state_dict(torch.load(pt, map_location='cuda:0'))
for i in range(config.n_phases):
    pt = f'{jft}/crossmix_{i}_final.pt'
    if os.path.exists(pt): model.cross_mix[i].load_state_dict(torch.load(pt, map_location='cuda:0'))
logging.info('All A6 checkpoints loaded (clean pilot, 5 epochs joint FT)')

model._residual_alpha = 1.0
model._disable_norm_stabilize = False
model._disable_crossmix = False
model._disable_crossphase = False

tok = AutoTokenizer.from_pretrained('Qwen/Qwen3.5-0.8B', trust_remote_code=True, cache_dir='.cache/huggingface')
tok.padding_side = 'left'
if tok.pad_token is None: tok.pad_token = tok.eos_token

metrics = evaluate_model(
    model=model, tokenizer=tok,
    n_per_domain=25, max_new_tokens=256,
    device='cuda:0', log_fn=logging.info,
)
result = {
    'ablation': 'A6_clean_pilot_jft',
    'flags': {'clean_traces': True, 'joint_finetuned': True, 'crossphase': True,
              'crossmix': True, 'agg_mode': 'mean', 'alpha': 1.0,
              'jft_epochs': 5, 'expert_epochs': 3},
    **metrics,
}
out = '${ABLATION_DIR}/A6_clean_pilot_jft_eval.json'
with open(out, 'w') as f:
    json.dump(result, f, indent=2)
print(json.dumps(result, indent=2))
print()
print(f'=== A6 CLEAN PILOT: {metrics[\"overall\"]*100:.1f}% overall ===')
for d in DOMAINS:
    c = metrics['correct'][d]; n = metrics['total'][d]
    print(f'  {d:15s}: {c}/{n} = {c/max(n,1)*100:.0f}%')
print()
print('Comparison:')
print('  Baseline Qwen3.5-0.8B:  0.8%  (1/125)')
print('  A5 contaminated JFT:    1.6%  (2/125, all trace data was garbage)')
print(f'  A6 clean pilot JFT:    {metrics[\"overall\"]*100:.1f}%  ({int(metrics[\"overall\"]*125)}/125)')
" 2>&1 | tee "$LOG_DIR/A6_eval.log"

echo ""
echo "========================================================"
echo "  PRISM PILOT COMPLETE — $(date)"
echo "========================================================"
tail -15 "$LOG_DIR/A6_eval.log"
