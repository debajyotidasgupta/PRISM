#!/usr/bin/env bash
# Pilot SOG evaluation using batched inference (batch_size=32).
# Each adapter evaluated on 50 problems per domain (250 total from MATH-500).
# Usage: bash scripts/eval_pilot.sh

set -e
cd "$(dirname "$0")/.."
source /users/dasgupta/miniconda3/bin/activate /iopsstor/scratch/cscs/dasgupta/research/ideas/PRISM/.venv

mkdir -p results/pilot/eval results/logs

echo "[$(date)] Running pilot SOG evaluation (batched)"

CUDA_VISIBLE_DEVICES=0 python -u - <<'PYEOF'
import os, sys, json, logging, torch
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, "src")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ.setdefault("HF_HOME", ".cache/huggingface")

ADAPTER_DIR = Path("results/pilot/lora_adapters")
EVAL_DIR = Path("results/pilot/eval")
EVAL_DIR.mkdir(parents=True, exist_ok=True)
BACKBONE = "Qwen/Qwen3.5-0.8B"
BATCH_SIZE = 32
N_PER_DOMAIN = 50   # 50 problems per domain = 250 total, fast enough

from prism.model.backbone import _get_model_dir
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from prism.eval.metrics import extract_answer_from_text
from datasets import load_dataset

model_path = _get_model_dir(BACKBONE)

# MATH-500 subject → pilot domain mapping
MATH_SUBJECT_MAP = {
    "Algebra": "algebra",
    "Intermediate Algebra": "algebra",
    "Prealgebra": "algebra",
    "Geometry": "geometry",
    "Counting & Probability": "combinatorics",
    "Number Theory": "number_theory",
    "Precalculus": "miscellaneous",
}

def load_math500():
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test",
                      cache_dir=".cache/huggingface")
    by_domain = defaultdict(list)
    for ex in ds:
        domain = MATH_SUBJECT_MAP.get(ex.get("subject", ""), "miscellaneous")
        by_domain[domain].append({
            "problem": ex["problem"],
            "answer": ex["answer"],
            "domain": domain,
        })
    logger.info(f"MATH-500 loaded: {sum(len(v) for v in by_domain.values())} problems")
    for d, probs in by_domain.items():
        logger.info(f"  {d}: {len(probs)} problems")
    return by_domain

def sample_problems(by_domain, n_per_domain):
    """Take up to n_per_domain from each domain."""
    problems, answers, domains = [], [], []
    for domain, probs in by_domain.items():
        subset = probs[:n_per_domain]
        for p in subset:
            problems.append(p["problem"])
            answers.append(p["answer"])
            domains.append(domain)
    return problems, answers, domains

def format_prompt(problem, tokenizer):
    messages = [
        {"role": "system", "content": "You are a math expert. Solve the problem and put the final answer in \\boxed{}."},
        {"role": "user", "content": problem},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def safe_match(pred: str, ans: str) -> bool:
    """Safe answer matching without eval() to prevent hangs."""
    import sys
    if not pred or not ans:
        return False
    # Normalize both: strip whitespace + LaTeX formatting
    def norm(s):
        import re
        s = str(s).strip()
        # Remove \boxed{}
        m = re.search(r'\\boxed\{', s)
        if m:
            # Extract content with brace matching
            depth, start, i = 0, m.end(), m.end()
            while i < len(s):
                if s[i] == '{': depth += 1
                elif s[i] == '}':
                    if depth == 0: s = s[start:i].strip(); break
                    depth -= 1
                i += 1
        # LaTeX substitutions
        s = re.sub(r'\\dfrac\s*\{([^{}]*)\}\s*\{([^{}]*)\}', r'\1/\2', s)
        s = re.sub(r'\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}', r'\1/\2', s)
        s = re.sub(r'\\left[\(\[]', '(', s)
        s = re.sub(r'\\right[\)\]]', ')', s)
        s = re.sub(r'\\[a-zA-Z]+\{([^{}]*)\}', r'\1', s)
        s = re.sub(r'\\[a-zA-Z]+', '', s)
        s = re.sub(r'[{}\$]', '', s)
        s = re.sub(r'\s+', '', s)
        return s.lower().strip()

    p, a = norm(pred), norm(ans)
    if p == a:
        return True
    # Safe numeric comparison — try parsing as fractions only
    try:
        from fractions import Fraction
        return Fraction(p) == Fraction(a)
    except Exception:
        pass
    # Try float parsing (no eval — only built-in float())
    try:
        return abs(float(p) - float(a)) < 1e-6
    except Exception:
        pass
    return False

def batched_eval(model, tokenizer, problems, answers, domains, batch_size=32):
    """Run batched inference and return per-domain accuracy."""
    import sys
    model.eval()
    device = next(model.parameters()).device

    domain_correct = defaultdict(int)
    domain_total = defaultdict(int)
    n_processed = 0

    for i in range(0, len(problems), batch_size):
        batch_probs = problems[i:i+batch_size]
        batch_ans = answers[i:i+batch_size]
        batch_doms = domains[i:i+batch_size]
        prompts = [format_prompt(p, tokenizer) for p in batch_probs]
        enc = tokenizer(prompts, return_tensors="pt", padding=True,
                        truncation=True, max_length=768).to(device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        for j, ids in enumerate(out):
            new_toks = ids[enc["input_ids"].shape[1]:]
            text = tokenizer.decode(new_toks, skip_special_tokens=True)
            pred = extract_answer_from_text(text)
            ans = batch_ans[j]
            dom = batch_doms[j]
            domain_total[dom] += 1
            if safe_match(pred, ans):
                domain_correct[dom] += 1
        n_processed += len(batch_probs)
        logger.info(f"  {n_processed}/{len(problems)} evaluated")
        sys.stderr.flush()

    domain_acc = {d: domain_correct[d] / domain_total[d] for d in domain_total}
    overall = sum(domain_correct.values()) / max(sum(domain_total.values()), 1)
    return {"accuracy": overall, "domain_accuracy": domain_acc,
            "n_correct": sum(domain_correct.values()),
            "n_total": sum(domain_total.values())}

def _remap_adapter(adapter_path: Path) -> Path:
    """Remap Unsloth-saved keys (language_model.layers) → PEFT standard (layers).
    Returns a patched adapter dir in /tmp that PEFT can load cleanly."""
    import safetensors.torch, shutil, json, tempfile
    tmp_dir = Path(tempfile.mkdtemp(prefix="prism_adapter_"))
    # Copy all non-weights files
    for fname in ("adapter_config.json", "tokenizer.json", "tokenizer_config.json"):
        src = adapter_path / fname
        if src.exists():
            shutil.copy(str(src), str(tmp_dir / fname))
    # Patch adapter_config: remove auto_mapping (Unsloth-specific) and unsloth_fixed
    cfg_path = tmp_dir / "adapter_config.json"
    with open(cfg_path) as f:
        cfg = json.load(f)
    cfg.pop("auto_mapping", None)  # was pointing to Qwen3_5ForConditionalGeneration
    cfg.pop("unsloth_fixed", None)
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    # Remap tensor keys
    sd = safetensors.torch.load_file(str(adapter_path / "adapter_model.safetensors"))
    remapped = {}
    for k, v in sd.items():
        new_k = k.replace(".language_model.layers.", ".layers.")
        remapped[new_k] = v
    safetensors.torch.save_file(remapped, str(tmp_dir / "adapter_model.safetensors"))
    logger.info(f"  Remapped {len(sd)} adapter keys → {tmp_dir}")
    return tmp_dir

def load_model_with_adapter(adapter_path=None):
    base = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    ).to("cuda:0")
    if adapter_path:
        remapped_path = _remap_adapter(adapter_path)
        model = PeftModel.from_pretrained(base, str(remapped_path))
        logger.info(f"Loaded adapter: {adapter_path}")
    else:
        model = base
        logger.info("Using base model (no adapter)")
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tok.padding_side = "left"  # required for correct batched generation
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok

# --- Main ---
by_domain = load_math500()
problems, answers, domains = sample_problems(by_domain, N_PER_DOMAIN)
logger.info(f"Evaluating on {len(problems)} problems ({N_PER_DOMAIN}/domain)")

results = {}
adapters_to_eval = [
    ("baseline",    None),
    ("general",     ADAPTER_DIR / "lora_general"),
    ("algebra",     ADAPTER_DIR / "lora_algebra"),
    ("geometry",    ADAPTER_DIR / "lora_geometry"),
    ("combinatorics", ADAPTER_DIR / "lora_combinatorics"),
    ("number_theory", ADAPTER_DIR / "lora_number_theory"),
    ("miscellaneous", ADAPTER_DIR / "lora_miscellaneous"),
]

for name, adapter_path in adapters_to_eval:
    if adapter_path and not adapter_path.exists():
        logger.warning(f"Skipping {name}: adapter not found")
        continue
    logger.info(f"\n{'='*50}\nEvaluating: {name}\n{'='*50}")
    model, tok = load_model_with_adapter(adapter_path)
    r = batched_eval(model, tok, problems, answers, domains, BATCH_SIZE)
    results[name] = r
    logger.info(f"{name}: overall={r['accuracy']:.3f} | {r['domain_accuracy']}")
    import sys; sys.stderr.flush()
    del model; tok = None
    torch.cuda.empty_cache()
    import gc; gc.collect()

# SOG gate
general_acc = results.get("general", {}).get("accuracy", 0.0)
baseline_acc = results.get("baseline", {}).get("accuracy", 0.0)

domain_results = {}
for domain in ["algebra", "geometry", "combinatorics", "number_theory", "miscellaneous"]:
    if domain not in results:
        continue
    dom_acc = results[domain].get("domain_accuracy", {}).get(domain, 0.0)
    gen_dom_acc = results.get("general", {}).get("domain_accuracy", {}).get(domain, 0.0)
    domain_results[domain] = {
        "domain_lora_acc": dom_acc,
        "general_acc_on_domain": gen_dom_acc,
        "beats_general": dom_acc > gen_dom_acc,
    }

wins = sum(1 for v in domain_results.values() if v["beats_general"])
passed = wins >= 3

summary = {
    "sog_passed": passed,
    "domain_wins": wins,
    "verdict": f"{'PASS ✓' if passed else 'FAIL ✗'}: {wins}/5 domain LoRAs beat general on their domain",
    "baseline_overall": baseline_acc,
    "general_overall": general_acc,
    "domain_breakdown": domain_results,
}

print("\n" + "="*60)
print("PILOT SOG GATE RESULT")
print("="*60)
print(f"  Verdict: {summary['verdict']}")
print(f"  Baseline accuracy:  {baseline_acc:.1%}")
print(f"  General LoRA accuracy: {general_acc:.1%}")
print("\n  Domain breakdown:")
for d, v in domain_results.items():
    mark = "✓" if v["beats_general"] else "✗"
    print(f"  {mark} {d}: domain_lora={v['domain_lora_acc']:.1%}  general={v['general_acc_on_domain']:.1%}")
print("="*60)

with open(EVAL_DIR / "pilot_sog_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
logger.info(f"Saved: {EVAL_DIR}/pilot_sog_summary.json")
PYEOF

echo "[$(date)] Done"
