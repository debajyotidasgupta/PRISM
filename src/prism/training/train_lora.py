"""
Stage 0: LoRA adapter training for hypothesis validation.

Trains a domain-specific LoRA adapter on Qwen3.5-0.8B using expert traces.
Also trains a general LoRA on all traces combined.

This is NOT part of the final PRISM architecture — it's a validation step
to confirm that domain-specific reasoning pathways actually help.

Usage:
  python -m prism.training.train_lora --domain algebra \
    --traces results/traces/algebra_traces.jsonl \
    --config configs/training/stage0_lora.yaml \
    --gpu 0 --output-dir results/stage0/lora_adapters
"""

import os
import json
import math
import logging
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)

DOMAIN_NAMES = ["algebra", "geometry", "combinatorics", "number_theory", "miscellaneous", "general"]


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
    lr: float = 5e-5,
    max_seq_len: int = 1024,
    warmup_steps: int = 50,
    output_dir: str = "results/stage0/lora_adapters",
) -> dict:
    """
    Train a LoRA adapter for one domain.

    Args:
        domain: Domain name or "general" (trains on all domains combined).
        traces_file: JSONL file of TraceExample records. Empty string = load all.
        backbone_name: HF model ID.
        gpu_id: GPU device.
        ...

    Returns:
        Dict with training history and checkpoint path.
    """
    from peft import get_peft_model, LoraConfig, TaskType
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from prism.model.backbone import _get_model_dir

    device = torch.device(f"cuda:{gpu_id}")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if lora_target_modules is None:
        lora_target_modules = ["q_proj", "v_proj"]

    logger.info(f"Loading backbone: {backbone_name}")
    model_path = _get_model_dir(backbone_name)

    hf_token = os.environ.get("HF_TOKEN", None)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        token=hf_token,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        token=hf_token,
    )

    # Apply LoRA
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model = model.to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"LoRA trainable params: {trainable_params:,}")

    # Load training data
    from prism.training.train_expert import TraceDataset
    from prism.data.collator import PRISMDataCollator

    if domain == "general" or not traces_file:
        # Load all domain traces for general adapter
        from prism.data.datasets import get_stage0_training_data
        logger.info("Loading all domain data for general LoRA")
        domain_data = get_stage0_training_data(n_per_domain=500)
        # Create a combined trace file temporarily
        import tempfile
        tmp_file = tempfile.mktemp(suffix=".jsonl")
        _create_combined_traces(domain_data, tokenizer, tmp_file)
        traces_file = tmp_file

    # Use TraceDataset for phase 0 (solve) traces — these are the primary supervision
    dataset = TraceDataset(traces_file, tokenizer, phase=0, max_length=max_seq_len)
    if len(dataset) == 0:
        logger.warning(f"No training data for {domain} — using raw data from MATH")
        dataset = _fallback_dataset(domain, tokenizer, max_seq_len)

    collator = PRISMDataCollator(tokenizer, include_domain_labels=False)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collator, num_workers=4, pin_memory=True,
    )

    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.01,
    )

    total_steps = epochs * len(dataloader)
    history = []
    global_step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in tqdm(dataloader, desc=f"LoRA {domain} epoch {epoch}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = out.loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # LR warmup
            global_step += 1
            if global_step <= warmup_steps:
                scale = global_step / warmup_steps
            else:
                progress = (global_step - warmup_steps) / max(total_steps - warmup_steps, 1)
                scale = 0.5 * (1.0 + math.cos(math.pi * progress))
            for pg in optimizer.param_groups:
                pg["lr"] = lr * scale

            total_loss += loss.item()
            n_batches += 1

        epoch_loss = total_loss / max(n_batches, 1)
        metrics = {"epoch": epoch, "domain": domain, "loss": epoch_loss}
        history.append(metrics)
        logger.info(f"LoRA {domain} epoch {epoch}: loss={epoch_loss:.4f}")

    # Save adapter
    adapter_path = output_path / f"lora_{domain}"
    model.save_pretrained(str(adapter_path))
    logger.info(f"Saved LoRA adapter: {adapter_path}")

    with open(output_path / f"lora_{domain}_history.json", "w") as f:
        json.dump(history, f, indent=2)

    return {"history": history, "adapter_path": str(adapter_path)}


def _create_combined_traces(domain_data: dict, tokenizer, output_file: str):
    """Create a combined trace JSONL file from domain data (for general LoRA)."""
    from prism.data.trace_format import TraceExample
    import random

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
    """Fallback: create training data from MATH dataset (no teacher traces)."""
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


def eval_lora_adapters(
    adapters_dir: str,
    backbone_name: str,
    benchmark: str,
    gpu_id: int = 0,
    max_samples: int = 500,
    output_dir: str = "results/stage0/eval",
) -> dict:
    """
    Evaluate all trained LoRA adapters on a benchmark.
    Key Stage 0 pass gate check.
    """
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

        # Load base model + adapter
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, trust_remote_code=True, token=hf_token
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
        results[domain] = {"accuracy": r["accuracy"], "n_correct": r["n_correct"], "n_total": r["n_total"]}

        del model, base_model
        torch.cuda.empty_cache()

    # Pass gate check
    general_acc = results.get("general", {}).get("accuracy", 0.0)
    domain_wins = 0
    for domain in ["algebra", "geometry", "combinatorics", "number_theory", "miscellaneous"]:
        domain_acc = results.get(domain, {}).get("accuracy", 0.0)
        if domain_acc > general_acc:
            domain_wins += 1

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


def main():
    """CLI: python -m prism.training.train_lora"""
    parser = argparse.ArgumentParser(description="Train Stage 0 LoRA adapters")
    parser.add_argument("--domain", required=True, choices=DOMAIN_NAMES)
    parser.add_argument("--traces", default="", help="JSONL trace file (empty = auto-load)")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output-dir", default="results/stage0/lora_adapters")
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
        lora_target_modules=cfg.get("lora_target_modules", ["q_proj", "v_proj"]),
        lora_dropout=cfg.get("lora_dropout", 0.05),
        epochs=cfg.get("epochs", 3),
        batch_size=cfg.get("batch_size", 16),
        lr=cfg.get("lr", 5e-5),
        max_seq_len=cfg.get("max_seq_len", 1024),
        warmup_steps=cfg.get("warmup_steps", 50),
        output_dir=args.output_dir,
    )
    print(f"LoRA {args.domain} training complete: {result['adapter_path']}")


if __name__ == "__main__":
    main()
