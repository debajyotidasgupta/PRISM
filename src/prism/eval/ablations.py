"""
Run all ablations from Section 11.3 of program.md in parallel across GPUs.

Ablations:
  A1: Backbone only, thinking=False
  A2: Backbone + thinking=True
  A3: PRISM, 1 general domain (no specialization)
  A4: PRISM, 4 domains only (no Miscellaneous)
  A5: PRISM, no Phase 2 (no verify level)
  A6: PRISM, no Phase 3 (no correct level)
  A7: PRISM, no cross-mixing
  A8: PRISM, hard routing (argmax)
  A9: PRISM, generic traces (not expert-aligned)
  A10: PRISM + thinking=True
  A11: PRISM, Misc expert uses only inequality traces
"""

import os
import json
import logging
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


ABLATION_CONFIGS = {
    "A1": {
        "description": "Backbone only, thinking=False",
        "model": "backbone",
        "thinking": False,
        "n_domains": None,
        "n_phases": None,
        "use_crossmix": True,
        "routing": "soft",
    },
    "A2": {
        "description": "Backbone + thinking=True",
        "model": "backbone",
        "thinking": True,
        "n_domains": None,
        "n_phases": None,
        "use_crossmix": True,
        "routing": "soft",
    },
    "A3": {
        "description": "PRISM, 1 general domain",
        "model": "prism",
        "thinking": False,
        "n_domains": 1,
        "n_phases": 3,
        "use_crossmix": False,
        "routing": "soft",
    },
    "A4": {
        "description": "PRISM, 4 domains (no Misc)",
        "model": "prism",
        "thinking": False,
        "n_domains": 4,
        "n_phases": 3,
        "use_crossmix": True,
        "routing": "soft",
    },
    "A5": {
        "description": "PRISM, no Phase 2 (verify)",
        "model": "prism",
        "thinking": False,
        "n_domains": 5,
        "n_phases": 2,
        "use_crossmix": True,
        "routing": "soft",
        "phases": ["solve", "correct"],
    },
    "A6": {
        "description": "PRISM, no Phase 3 (correct)",
        "model": "prism",
        "thinking": False,
        "n_domains": 5,
        "n_phases": 2,
        "use_crossmix": True,
        "routing": "soft",
        "phases": ["solve", "verify"],
    },
    "A7": {
        "description": "PRISM, no cross-mixing",
        "model": "prism",
        "thinking": False,
        "n_domains": 5,
        "n_phases": 3,
        "use_crossmix": False,
        "routing": "soft",
    },
    "A8": {
        "description": "PRISM, hard routing (argmax)",
        "model": "prism",
        "thinking": False,
        "n_domains": 5,
        "n_phases": 3,
        "use_crossmix": True,
        "routing": "hard",
    },
    "A10": {
        "description": "PRISM + thinking=True",
        "model": "prism",
        "thinking": True,
        "n_domains": 5,
        "n_phases": 3,
        "use_crossmix": True,
        "routing": "soft",
    },
}


def run_ablation_eval(
    ablation_id: str,
    benchmark: str,
    model_dir: str,
    backbone_name: str,
    gpu_id: int,
    output_dir: str = "results/ablations",
    max_samples: Optional[int] = None,
) -> dict:
    """
    Run a single ablation evaluation.

    Args:
        ablation_id: Ablation identifier (A1, A2, ..., A11).
        benchmark: Benchmark name.
        model_dir: Directory with trained PRISM checkpoints.
        backbone_name: HF model ID for backbone.
        gpu_id: GPU to use.
        output_dir: Where to save results.

    Returns:
        Dict with evaluation results.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = ABLATION_CONFIGS.get(ablation_id)
    if config is None:
        raise ValueError(f"Unknown ablation: {ablation_id}")

    logger.info(f"Running ablation {ablation_id}: {config['description']}")
    device = f"cuda:{gpu_id}"

    if config["model"] == "backbone":
        # Standard backbone evaluation
        model = AutoModelForCausalLM.from_pretrained(
            backbone_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(backbone_name, trust_remote_code=True)

    else:
        # PRISM model with ablation modifications
        from prism.model.config import PRISMConfig
        from prism.model.prism_model import PRISMModel

        n_domains = config.get("n_domains", 5)
        n_phases = config.get("n_phases", 3)
        phases = config.get("phases", None)

        prism_config = PRISMConfig(
            backbone_name=backbone_name,
            n_domains=n_domains,
            n_phases=n_phases,
            phases=phases,
        )
        model = PRISMModel(prism_config)._load_backbone(device=device)

        # Load checkpoints if available
        _load_prism_checkpoints(model, model_dir, n_domains, n_phases)

        # Apply hard routing if needed
        if config.get("routing") == "hard":
            model._use_hard_routing = True

        # Disable cross-mix if needed
        if not config.get("use_crossmix", True):
            model._disable_crossmix = True

        model = model.to(device)
        tokenizer = AutoTokenizer.from_pretrained(backbone_name, trust_remote_code=True)

    from prism.eval.eval_prism import run_benchmark
    results = run_benchmark(
        model=model,
        tokenizer=tokenizer,
        benchmark=benchmark,
        gpu_id=gpu_id,
        max_samples=max_samples,
        enable_thinking=config.get("thinking", False),
        output_dir=output_dir,
        model_name=f"{ablation_id}_{benchmark}",
    )
    results["ablation_id"] = ablation_id
    results["description"] = config["description"]
    return results


def _load_prism_checkpoints(model, model_dir: str, n_domains: int, n_phases: int):
    """Load available expert block checkpoints into model."""
    import torch
    domain_names = ["algebra", "geometry", "combinatorics", "number_theory", "miscellaneous"][:n_domains]
    model_path = Path(model_dir)

    for phase_idx in range(n_phases):
        for domain_idx, domain_name in enumerate(domain_names):
            ckpt = model_path / f"expert_blocks/phase{phase_idx}_{domain_name}_final.pt"
            if ckpt.exists():
                state = torch.load(str(ckpt), map_location="cpu")
                model.expert_blocks[phase_idx][domain_idx].load_state_dict(state)
                logger.info(f"Loaded checkpoint: {ckpt.name}")

    # Load router
    router_ckpt = model_path / "router/router_final.pt"
    if router_ckpt.exists():
        state = torch.load(str(router_ckpt), map_location="cpu")
        model.router.load_state_dict(state)
        logger.info("Loaded router checkpoint")


def run_all_ablations(
    benchmark: str,
    model_dir: str,
    backbone_name: str,
    output_dir: str = "results/ablations",
    max_samples: Optional[int] = None,
    ablations: Optional[list[str]] = None,
) -> dict:
    """
    Run all ablations sequentially (use the shell script for parallel execution).
    """
    if ablations is None:
        ablations = list(ABLATION_CONFIGS.keys())

    results = {}
    for i, abl_id in enumerate(ablations):
        gpu_id = i % 4  # Distribute across 4 GPUs
        try:
            r = run_ablation_eval(
                ablation_id=abl_id,
                benchmark=benchmark,
                model_dir=model_dir,
                backbone_name=backbone_name,
                gpu_id=gpu_id,
                output_dir=output_dir,
                max_samples=max_samples,
            )
            results[abl_id] = r
        except Exception as e:
            logger.error(f"Ablation {abl_id} failed: {e}")
            results[abl_id] = {"error": str(e)}

    # Save combined results
    out_path = Path(output_dir) / f"ablations_{benchmark}_summary.json"
    with open(out_path, "w") as f:
        json.dump(
            {k: {kk: vv for kk, vv in v.items() if kk not in ("predictions", "responses", "ground_truths", "wrong_indices")}
             for k, v in results.items()},
            f, indent=2,
        )
    logger.info(f"All ablation results saved to {out_path}")
    return results
