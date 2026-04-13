"""
Main evaluation script for PRISM.

Usage:
  prism-eval --model results/stage2/ --benchmark olympiadbench --gpu 0
  prism-eval --model results/stage2/ --benchmark math500 --gpu 0

Outputs a JSON file with per-example predictions and aggregate accuracy.
"""

import os
import json
import logging
import argparse
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm

from prism.eval.metrics import extract_answer_from_text, compute_accuracy

logger = logging.getLogger(__name__)


BENCHMARK_CONFIGS = {
    "math500": {
        "dataset": "HuggingFaceH4/MATH-500",
        "split": "test",
        "problem_key": "problem",
        "answer_key": "answer",
    },
    "olympiadbench": {
        "dataset": "Hothan/OlympiadBench",
        "split": "test",
        "problem_key": "question",
        "answer_key": "answer",
    },
    "olymmath": {
        "dataset": "RUC-AIBOX/OlymMATH",
        "split": "test",
        "problem_key": "problem",
        "answer_key": "answer",
        "filter": {"language": "en", "difficulty": "hard"},
    },
    "omnimath": {
        "dataset": "KbsdJames/Omni-MATH",
        "split": "test",
        "problem_key": "problem",
        "answer_key": "answer",
    },
}


class PRISMEvaluator:
    """
    Evaluates a model (PRISM or baseline) on math benchmarks.

    Supports:
      - PRISMModel (custom)
      - Any HF CausalLM (for baselines)
    """

    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cuda:0",
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        enable_thinking: bool = False,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.enable_thinking = enable_thinking

    def _format_problem(self, problem: str) -> str:
        """Format problem as chat message and apply template."""
        messages = [
            {
                "role": "system",
                "content": "You are a mathematical expert. Solve the following olympiad problem step by step. State your final answer clearly using \\boxed{...}.",
            },
            {"role": "user", "content": problem},
        ]

        try:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
        except TypeError:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return text

    @torch.no_grad()
    def predict(self, problem: str) -> str:
        """Generate answer for one problem. Returns extracted answer string."""
        formatted = self._format_problem(problem)
        inputs = self.tokenizer(
            formatted,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(self.device)

        # Use model's generate
        if hasattr(self.model, "backbone") and self.model.backbone is not None:
            # PRISMModel — use custom generate
            output_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=(self.temperature > 0),
            )
        else:
            # Standard HF model
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=(self.temperature > 0),
                temperature=self.temperature if self.temperature > 0 else 1.0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        new_tokens = output_ids[0][input_len:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        return extract_answer_from_text(response), response

    def evaluate_dataset(
        self,
        problems: list[str],
        ground_truths: list[str],
        domain_labels: Optional[list[str]] = None,
        max_samples: Optional[int] = None,
    ) -> dict:
        """
        Evaluate on a list of problems.

        Returns dict with accuracy metrics and per-example results.
        """
        if max_samples is not None:
            problems = problems[:max_samples]
            ground_truths = ground_truths[:max_samples]
            if domain_labels:
                domain_labels = domain_labels[:max_samples]

        predictions = []
        full_responses = []

        for i, problem in enumerate(tqdm(problems, desc="Evaluating")):
            try:
                pred, response = self.predict(problem)
            except Exception as e:
                logger.warning(f"Prediction failed for problem {i}: {e}")
                pred, response = "", ""
            predictions.append(pred)
            full_responses.append(response)

        # Overall accuracy
        accuracy_result = compute_accuracy(predictions, ground_truths)

        # Per-domain accuracy (if labels provided)
        domain_accuracy = {}
        if domain_labels is not None:
            from collections import defaultdict
            domain_results = defaultdict(lambda: {"correct": 0, "total": 0})
            for pred, gt, domain in zip(predictions, ground_truths, domain_labels):
                from prism.eval.metrics import exact_match
                domain_results[domain]["total"] += 1
                if exact_match(pred, gt):
                    domain_results[domain]["correct"] += 1
            domain_accuracy = {
                d: v["correct"] / max(v["total"], 1)
                for d, v in domain_results.items()
            }

        result = {
            **accuracy_result,
            "domain_accuracy": domain_accuracy,
            "predictions": predictions,
            "responses": full_responses,
            "ground_truths": ground_truths,
        }
        return result


def run_benchmark(
    model,
    tokenizer,
    benchmark: str,
    gpu_id: int = 0,
    max_samples: Optional[int] = None,
    enable_thinking: bool = False,
    output_dir: str = "results",
    model_name: str = "prism",
) -> dict:
    """
    Load a benchmark dataset and evaluate the model.
    """
    import os
    os.environ.setdefault("PRISM_ROOT", "/iopsstor/scratch/cscs/dasgupta/research/ideas/PRISM")
    hf_home = os.environ.get("HF_HOME", None)

    config = BENCHMARK_CONFIGS.get(benchmark)
    if config is None:
        raise ValueError(f"Unknown benchmark: {benchmark}. Choose from {list(BENCHMARK_CONFIGS)}")

    from datasets import load_dataset
    logger.info(f"Loading benchmark: {benchmark}")
    ds = load_dataset(
        config["dataset"],
        split=config["split"],
        cache_dir=hf_home,
        trust_remote_code=True,
    )

    # Apply filters
    if "filter" in config:
        for key, val in config["filter"].items():
            ds = ds.filter(lambda x: x.get(key, "") == val)

    problems = [ex[config["problem_key"]] for ex in ds]
    ground_truths = [str(ex[config["answer_key"]]) for ex in ds]
    domain_labels = [ex.get("subject", ex.get("domain", "")) for ex in ds]

    evaluator = PRISMEvaluator(
        model=model,
        tokenizer=tokenizer,
        device=f"cuda:{gpu_id}",
        enable_thinking=enable_thinking,
    )

    results = evaluator.evaluate_dataset(
        problems=problems,
        ground_truths=ground_truths,
        domain_labels=domain_labels,
        max_samples=max_samples,
    )

    # Save results
    out_path = Path(output_dir) / f"{model_name}_{benchmark}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {k: v for k, v in results.items() if k not in ("responses",)},
            f, indent=2,
        )

    logger.info(f"Benchmark {benchmark}: accuracy={results['accuracy']:.3f} ({results['n_correct']}/{results['n_total']})")
    logger.info(f"Results saved to {out_path}")
    return results


def main():
    """CLI entry point: prism-eval"""
    parser = argparse.ArgumentParser(description="Evaluate PRISM model on math benchmarks")
    parser.add_argument("--model", required=True, help="Path to model directory or HF model ID")
    parser.add_argument("--benchmark", required=True, choices=list(BENCHMARK_CONFIGS))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--thinking", action="store_true", help="Enable thinking mode")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--model-name", default="prism")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    import os
    os.environ.setdefault("PRISM_ROOT", "/iopsstor/scratch/cscs/dasgupta/research/ideas/PRISM")

    device = f"cuda:{args.gpu}"

    # Try to load as PRISMModel, fall back to standard HF
    try:
        from prism.model.prism_model import PRISMModel
        from prism.model.config import PRISMConfig

        config = PRISMConfig.from_pretrained(args.model, trust_remote_code=True)
        model = PRISMModel(config)._load_backbone(device=device)
        model = model.to(device)
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.backbone_name, trust_remote_code=True)
        model_name = args.model_name or "prism"
    except Exception as e:
        logger.info(f"Loading as standard HF model: {e}")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float16, trust_remote_code=True, device_map=device
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        model_name = args.model_name or Path(args.model).name

    results = run_benchmark(
        model=model,
        tokenizer=tokenizer,
        benchmark=args.benchmark,
        gpu_id=args.gpu,
        max_samples=args.max_samples,
        enable_thinking=args.thinking,
        output_dir=args.output_dir,
        model_name=model_name,
    )

    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"Benchmark: {args.benchmark}")
    print(f"Accuracy: {results['accuracy']*100:.1f}% ({results['n_correct']}/{results['n_total']})")
    print(f"{'='*60}")
    if results.get("domain_accuracy"):
        for domain, acc in sorted(results["domain_accuracy"].items()):
            print(f"  {domain:<20}: {acc*100:.1f}%")


if __name__ == "__main__":
    main()
