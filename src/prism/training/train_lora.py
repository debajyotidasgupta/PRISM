"""
Stage 0: LoRA adapter training for hypothesis validation.

Trains a domain-specific LoRA adapter on Qwen3.5-0.8B using expert traces.
Also trains a general LoRA on all traces combined.

Optimisations applied:
  - bfloat16 (GH200-native, more numerically stable than float16)
  - Flash Attention 2  (already installed as flash_attn 2.8.3)
  - Full linear-layer LoRA targets: q/k/v/o + gate/up/down proj
  - Gradient checkpointing (lower peak VRAM, enables larger batch)
  - trl SFTTrainer with proper cosine LR scheduler and gradient accumulation
  - Unsloth fast-path (2× throughput via custom Triton kernels) when available

Usage:
  python -m prism.training.train_lora --domain algebra \
    --traces results/traces/algebra_traces.jsonl \
    --config configs/training/stage0_lora.yaml \
    --gpu 0 --output-dir results/stage0/lora_adapters
"""

import os
import json
import logging
import argparse
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

DOMAIN_NAMES = ["algebra", "geometry", "combinatorics", "number_theory", "miscellaneous", "general"]

# All linear projection layers — better coverage than just q+v
_DEFAULT_LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"]


# ──────────────────────────────────────────────────────────────────────────────
# Model loading helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_model_unsloth(model_path, lora_r, lora_alpha, lora_target_modules,
                        lora_dropout, max_seq_len):
    """Fast path: load model + LoRA via Unsloth (2× throughput, lower VRAM)."""
    from unsloth import FastLanguageModel  # noqa: import-error

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(model_path),
        max_seq_length=max_seq_len,
        dtype=torch.bfloat16,
        load_in_4bit=False,  # full precision LoRA for 0.8B — fits fine
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",  # Unsloth's optimised checkpointing
        random_state=42,
    )
    return model, tokenizer, "unsloth"


def _load_model_standard(model_path, lora_r, lora_alpha, lora_target_modules,
                         lora_dropout, gpu_id):
    """Standard path: HF + PEFT with Flash Attention 2 and bfloat16."""
    from peft import get_peft_model, LoraConfig, TaskType
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_token = os.environ.get("HF_TOKEN", None)

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16,      # bfloat16: GH200-native, more stable than fp16
        attn_implementation="flash_attention_2",  # FlashAttention-2 for speed/memory
        trust_remote_code=True,
        token=hf_token,
    )
    model.gradient_checkpointing_enable()  # lower peak VRAM

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        token=hf_token,
    )

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    device = torch.device(f"cuda:{gpu_id}")
    model = model.to(device)
    return model, tokenizer, "standard"


# ──────────────────────────────────────────────────────────────────────────────
# Main training function
# ──────────────────────────────────────────────────────────────────────────────

def train_lora(
    domain: str,
    traces_file: str,
    backbone_name: str = "Qwen/Qwen3.5-0.8B",
    gpu_id: int = 0,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_target_modules: list = None,
    lora_dropout: float = 0.05,
    epochs: int = 3,
    batch_size: int = 16,
    grad_accum_steps: int = 4,
    lr: float = 5e-5,
    max_seq_len: int = 1024,
    warmup_steps: int = 50,
    weight_decay: float = 0.01,
    output_dir: str = "results/stage0/lora_adapters",
    max_steps: int = -1,  # -1 = use epochs; >0 overrides epochs
) -> dict:
    """
    Train a LoRA adapter for one domain using SFTTrainer.

    Args:
        domain: Domain name or "general" (trains on all domains combined).
        traces_file: JSONL file of TraceExample records. Empty = load all.
        backbone_name: HF model ID.
        gpu_id: GPU device index.

    Returns:
        Dict with training history and checkpoint path.
    """
    from prism.model.backbone import _get_model_dir

    if lora_target_modules is None:
        lora_target_modules = _DEFAULT_LORA_TARGETS

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    adapter_path = output_path / f"lora_{domain}"

    logger.info(f"Loading backbone: {backbone_name}")
    model_path = _get_model_dir(backbone_name)

    # Try Unsloth first for 2× training speed; fall back to standard HF path
    try:
        model, tokenizer, backend = _load_model_unsloth(
            model_path, lora_r, lora_alpha, lora_target_modules, lora_dropout, max_seq_len,
        )
        logger.info("Using Unsloth fast-path")
        # Unsloth sets the device internally; no explicit .to(device) needed
    except ImportError:
        logger.info("Unsloth not available — using standard HF+PEFT (Flash Attention 2, bfloat16)")
        model, tokenizer, backend = _load_model_standard(
            model_path, lora_r, lora_alpha, lora_target_modules, lora_dropout, gpu_id,
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"LoRA trainable: {trainable_params:,} / {total_params:,} "
                f"({100 * trainable_params / total_params:.2f}%)")

    # ── Build dataset ──────────────────────────────────────────────────────────
    if domain == "general" or not traces_file:
        from prism.data.datasets import get_stage0_training_data
        import tempfile
        logger.info("Loading all domain data for general LoRA")
        domain_data = get_stage0_training_data(n_per_domain=500)
        tmp_file = tempfile.mktemp(suffix=".jsonl")
        _create_combined_traces(domain_data, tokenizer, tmp_file)
        traces_file = tmp_file

    # Build HF Dataset with "text" field so SFTTrainer can tokenize it
    hf_dataset = _build_hf_dataset(traces_file, tokenizer, max_seq_len)
    if len(hf_dataset) == 0:
        logger.warning(f"No training data for {domain} — falling back to raw MATH")
        hf_dataset = _fallback_hf_dataset(domain, tokenizer, max_seq_len)

    logger.info(f"Training on {len(hf_dataset)} examples")

    # ── SFTTrainer ─────────────────────────────────────────────────────────────
    from trl import SFTTrainer, SFTConfig

    # Only set CUDA_VISIBLE_DEVICES if gpu_id is specified and not in DDP mode
    if "LOCAL_RANK" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    effective_max_steps = max_steps if max_steps > 0 else -1
    effective_epochs = 999 if max_steps > 0 else epochs

    training_args = SFTConfig(
        output_dir=str(adapter_path),
        max_steps=effective_max_steps,
        num_train_epochs=effective_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        learning_rate=float(lr),
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        bf16=True,           # bfloat16 mixed precision
        fp16=False,
        max_seq_length=max_seq_len,
        logging_steps=10,
        save_strategy="no",          # don't save mid-training checkpoints
        save_total_limit=1,
        report_to="none",    # no wandb/tensorboard
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        remove_unused_columns=True,
        gradient_checkpointing=True,
        dataset_text_field="text",
        # packing=True,      # pack short sequences — enable for large datasets
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=hf_dataset,
        args=training_args,
    )

    train_result = trainer.train()
    metrics = train_result.metrics

    # ── Save adapter ───────────────────────────────────────────────────────────
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    logger.info(f"Saved LoRA adapter: {adapter_path}")

    history = {
        "domain": domain,
        "backend": backend,
        "train_loss": metrics.get("train_loss"),
        "train_runtime": metrics.get("train_runtime"),
        "train_samples_per_second": metrics.get("train_samples_per_second"),
    }
    with open(output_path / f"lora_{domain}_history.json", "w") as f:
        json.dump(history, f, indent=2)

    return {"history": history, "adapter_path": str(adapter_path)}


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _build_hf_dataset(traces_file: str, tokenizer, max_seq_len: int):
    """
    Build a HuggingFace Dataset with a 'text' field from a JSONL trace file.
    Formats each trace as:  Problem: {problem}\n\nSolution:\n{correct_trace}
    SFTTrainer will tokenize the 'text' field with its own tokenizer.
    """
    from datasets import Dataset as HFDataset
    from prism.data.trace_format import TraceExample

    records = []
    path = Path(traces_file)
    if not path.exists():
        return HFDataset.from_list([])

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ex = TraceExample.from_jsonl(line)
                if not ex.is_valid():
                    continue
                # Use Phase 3 correct_trace as training target, cleaned of meta-commentary.
                # If the trace has no recoverable math, fall back to ground_truth.
                # If ground_truth also lacks math, skip the example.
                target = _clean_trace(ex.correct_trace)
                if not target.strip():
                    # correct_trace was entirely meta — fall back to ground_truth.
                    # Accept any ground_truth ≥50 chars (prose math is still valid signal).
                    gt = (ex.ground_truth or "").strip()
                    if len(gt) >= 50:
                        target = gt
                    else:
                        continue  # too short to be useful

                text = (
                    f"<|im_start|>system\nYou are a domain-expert mathematician. "
                    f"Solve the following problem step by step, showing expert reasoning.<|im_end|>\n"
                    f"<|im_start|>user\n{ex.problem}<|im_end|>\n"
                    f"<|im_start|>assistant\n{target}<|im_end|>"
                )
                records.append({"text": text})
            except Exception:
                pass

    return HFDataset.from_list(records)


def _has_math_content(text: str) -> bool:
    """Returns True if text contains substantial mathematical content."""
    import re
    patterns = [
        r'\\boxed', r'\\frac', r'\\sum', r'\\sqrt', r'\\begin',
        r'\\align', r'align\*', r'\d+\s*=\s*\d', r'\\\(',
    ]
    return sum(1 for p in patterns if re.search(p, text)) >= 2


def _clean_trace(trace: str) -> str:
    """
    Extract the mathematical solution from a Phase 3 correct_trace.

    Round 1 traces are contaminated: the 35B teacher output meta-commentary before
    the actual solution. The teacher consistently uses section headers like
    "Expert Solution", "Final Solution", "FINAL SOLUTION" to mark where the actual
    math begins. We extract everything after the last such marker.

    Falls back to the first paragraph containing display math (\\[ or \\begin{)
    if no section header is found.

    Returns empty string if no clean content can be extracted — caller should fall
    back to ground_truth.
    """
    import re

    # Strip <think>...</think> blocks
    trace = re.sub(r"<think>.*?</think>", "", trace, flags=re.DOTALL).strip()

    _META_STARTS = (
        "Thinking Process", "Analyze the Request", "The user wants me",
        "I need to act as", "My task is", "My role", "I am asked",
        "Looking at the prompt", "The prompt",
    )

    def _starts_with_meta(text: str) -> bool:
        head = text[:400]
        return any(phrase in head for phrase in _META_STARTS)

    # Primary anchor: section headers the teacher uses to introduce the solution.
    # Take the LAST match (occasionally the header appears twice; last = real solution).
    section_header_re = re.compile(
        r'(?:'
        r'\*{0,2}(?:Expert|Final|FINAL)\s+(?:Solution|SOLUTION)\*{0,2}'
        r'|FINAL\s+POLISHED\s+(?:Expert\s+)?SOLUTION'
        r'|Phase\s+3[:\s]*(?:Final\s+)?(?:Expert\s+)?Solution'
        r')[:\s]*\n',
        re.IGNORECASE,
    )
    matches = list(section_header_re.finditer(trace))
    if matches:
        solution_start = matches[-1].end()
        content = trace[solution_start:].strip()
        if content and not _starts_with_meta(content):
            return content

    # Fallback: find first paragraph containing display math (\[ or \begin{)
    # These only appear in real mathematical exposition, not meta-commentary.
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", trace) if p.strip()]
    display_math_re = re.compile(r'\\\[|\\begin\{|\\frac\{.*?\\frac\{')  # \[ or nested \frac
    for i, para in enumerate(paragraphs):
        if display_math_re.search(para):
            # Include the previous paragraph only if it looks like a section header
            start = i
            if i > 0 and len(paragraphs[i - 1]) < 100:
                prev = paragraphs[i - 1]
                if not any(phrase in prev for phrase in [
                    "Thinking Process", "Analyze the Request", "The user wants",
                    "I need to act", "My task", "Phase 1", "Phase 2",
                ]):
                    start = i - 1
            content = "\n\n".join(paragraphs[start:])
            if not _starts_with_meta(content):
                return content
            # This display-math para is still inside a meta block; keep searching
            continue

    return ""  # no clean content found — caller uses ground_truth


def _fallback_hf_dataset(domain: str, tokenizer, max_seq_len: int):
    """Fallback: build HF Dataset from raw MATH problems when no traces available."""
    from datasets import Dataset as HFDataset
    from prism.data.datasets import load_math_dataset

    ds = load_math_dataset(domain=domain if domain != "general" else None, max_samples=500)
    records = []
    for ex in ds:
        problem = ex.get("problem", "")
        solution = ex.get("solution", "")
        text = (
            f"<|im_start|>system\nYou are a domain-expert mathematician.<|im_end|>\n"
            f"<|im_start|>user\n{problem}<|im_end|>\n"
            f"<|im_start|>assistant\n{solution}<|im_end|>"
        )
        records.append({"text": text})
    return HFDataset.from_list(records)


def _create_combined_traces(domain_data: dict, tokenizer, output_file: str):
    from prism.data.trace_format import TraceExample

    with open(output_file, "w") as f:
        for domain, ds in domain_data.items():
            for ex in ds:
                problem = ex.get("problem", ex.get("question", ""))
                solution = ex.get("solution", ex.get("answer", ""))
                trace = TraceExample(
                    problem_id=str(ex.get("id", "")),
                    problem=problem,
                    domain=domain,
                    ground_truth=str(solution),
                    solve_trace=solution,
                    verify_trace="The solution is correct.",
                    correct_trace=solution,
                    correct_correct=True,
                    total_tokens=len(solution) // 4,
                )
                f.write(trace.to_jsonl() + "\n")


def _fallback_dataset(domain: str, tokenizer, max_seq_len: int):
    from prism.data.datasets import load_math_dataset
    from prism.training.train_expert import TraceDataset
    from prism.data.trace_format import TraceExample
    import tempfile

    ds = load_math_dataset(domain=domain if domain != "general" else None, max_samples=500)
    tmp_file = tempfile.mktemp(suffix=".jsonl")

    with open(tmp_file, "w") as f:
        for ex in ds:
            problem = ex.get("problem", "")
            solution = ex.get("solution", "")
            trace = TraceExample(
                problem_id="",
                problem=problem,
                domain=domain if domain != "general" else "miscellaneous",
                ground_truth="",
                solve_trace=solution,
                verify_trace="The solution is correct.",
                correct_trace=solution,
                correct_correct=True,
                total_tokens=len(solution) // 4,
            )
            f.write(trace.to_jsonl() + "\n")

    return TraceDataset(tmp_file, tokenizer, phase=0, max_length=max_seq_len)


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────

def eval_lora_adapters(
    adapters_dir: str,
    backbone_name: str,
    benchmark: str,
    gpu_id: int = 0,
    max_samples: int = 500,
    output_dir: str = "results/stage0/eval",
) -> dict:
    """Evaluate all trained LoRA adapters; check Stage 0 pass gate."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from prism.model.backbone import _get_model_dir
    from prism.eval.eval_prism import run_benchmark

    device = f"cuda:{gpu_id}"
    adapters_path = Path(adapters_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    hf_token = os.environ.get("HF_TOKEN", None)
    model_path = _get_model_dir(backbone_name)

    results = {}

    for adapter_dir in sorted(adapters_path.glob("lora_*")):
        domain = adapter_dir.name.replace("lora_", "")
        logger.info(f"Evaluating LoRA adapter: {domain}")

        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
            token=hf_token,
        )
        model = PeftModel.from_pretrained(base_model, str(adapter_dir))
        model = model.to(device)
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, token=hf_token)

        r = run_benchmark(
            model=model, tokenizer=tokenizer,
            benchmark=benchmark, gpu_id=gpu_id,
            max_samples=max_samples,
            output_dir=output_dir,
            model_name=f"lora_{domain}",
        )
        results[domain] = {
            "accuracy": r["accuracy"],
            "n_correct": r["n_correct"],
            "n_total": r["n_total"],
        }

        del model, base_model
        torch.cuda.empty_cache()

    # Pass gate: at least 3 of 5 domain adapters must beat the general adapter
    general_acc = results.get("general", {}).get("accuracy", 0.0)
    domain_wins = sum(
        1 for d in ["algebra", "geometry", "combinatorics", "number_theory", "miscellaneous"]
        if results.get(d, {}).get("accuracy", 0.0) > general_acc
    )
    pass_gate = domain_wins >= 3
    results["pass_gate"] = {
        "passed": pass_gate,
        "domain_wins": domain_wins,
        "general_accuracy": general_acc,
        "message": f"{'PASS' if pass_gate else 'FAIL'}: {domain_wins}/5 domain adapters beat general LoRA",
    }
    logger.info(f"Stage 0 pass gate: {results['pass_gate']['message']}")

    with open(output_path / "stage0_eval_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train Stage 0 LoRA adapters")
    parser.add_argument("--domain", required=True, choices=DOMAIN_NAMES)
    parser.add_argument("--traces", default="", help="JSONL trace file (empty = auto-load)")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output-dir", default="results/stage0/lora_adapters")
    parser.add_argument("--max-steps", type=int, default=-1,
                        help="-1 = use epochs from config; >0 overrides epochs")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    result = train_lora(
        domain=args.domain,
        traces_file=args.traces,
        backbone_name=cfg.get("backbone_name", "Qwen/Qwen3.5-0.8B"),
        gpu_id=args.gpu,
        lora_r=cfg.get("lora_r", 16),
        lora_alpha=cfg.get("lora_alpha", 32),
        lora_target_modules=cfg.get("lora_target_modules", _DEFAULT_LORA_TARGETS),
        lora_dropout=cfg.get("lora_dropout", 0.05),
        epochs=cfg.get("epochs", 3),
        batch_size=cfg.get("batch_size", 16),
        grad_accum_steps=cfg.get("grad_accum_steps", 4),
        lr=cfg.get("lr", 5e-5),
        max_seq_len=cfg.get("max_seq_len", 1024),
        warmup_steps=cfg.get("warmup_steps", 50),
        weight_decay=cfg.get("weight_decay", 0.01),
        output_dir=args.output_dir,
        max_steps=args.max_steps,
    )
    print(f"LoRA {args.domain} training complete: {result['adapter_path']}")


if __name__ == "__main__":
    main()
