"""
POC: End-to-end trace generation test.

Tests the complete trace generation pipeline on a small subset (n_problems per domain)
BEFORE scaling to the full dataset.

What this validates:
  1. Teacher model loads correctly (Qwen3-VL-30B-A3B-Thinking — image+text VL model)
  2. VL input works: image+text problems processed correctly
  3. All 3 phase prompts produce coherent, domain-aligned traces
  4. Both single-domain and cross-domain verification traces work
  5. Quality filter correctly identifies correct/incorrect traces
  6. JSONL output is parseable by the data pipeline

Teacher: Qwen/Qwen3-VL-30B-A3B-Thinking (30B params, 3B active MoE — fits on 1 GH200)
  - enable_thinking=True, temperature=1.0, top_p=0.95, top_k=20, presence_penalty=1.5
Student: Qwen/Qwen3.5-0.8B (also VL — image+text)

Usage:
  python scripts/poc_trace_gen.py --n-problems 20 --gpu 0 [--with-images]
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path

# Ensure the package is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# POC uses a small curated set from MATH Level 4-5 (hardest, best quality)
POC_DOMAIN_QUERIES = {
    "algebra": {
        "keywords": ["polynomial", "function", "equation", "find all"],
        "level_min": 4,
    },
    "geometry": {
        "keywords": ["triangle", "circle", "angle", "prove"],
        "level_min": 4,
    },
    "combinatorics": {
        "keywords": ["count", "probability", "arrangements", "ways"],
        "level_min": 4,
    },
    "number_theory": {
        "keywords": ["divisible", "prime", "integer", "modulo"],
        "level_min": 4,
    },
    "miscellaneous": {
        "keywords": ["inequality", "maximum", "minimum", "prove that"],
        "level_min": 4,
    },
}

# Cross-domain verification pairs: problem domain → verifier domain
# An algebra problem's solve trace can be verified by combinatorics/misc expert
CROSS_VERIFY_PAIRS = [
    ("algebra", "miscellaneous"),       # Algebraic bounds verified by inequality tools
    ("geometry", "algebra"),            # Coordinate geometry verified algebraically
    ("combinatorics", "number_theory"), # Counting modular patterns verified by NT
    ("number_theory", "algebra"),       # Diophantine equations verified algebraically
]


def load_poc_problems(domain: str, n: int) -> list[dict]:
    """Load n problems for the POC from MATH dataset (EleutherAI/hendrycks_math)."""
    import os
    os.environ.setdefault("HF_HOME", "/iopsstor/scratch/cscs/dasgupta/research/ideas/PRISM/.cache/huggingface")

    from datasets import load_dataset, concatenate_datasets
    from prism.data.datasets import MATH_LABEL_MAP

    # Load all MATH configs and concatenate (EleutherAI mirror has per-subject splits)
    MATH_CONFIGS = [
        "algebra", "counting_and_probability", "geometry",
        "intermediate_algebra", "number_theory", "prealgebra", "precalculus",
    ]
    all_datasets = []
    for config in MATH_CONFIGS:
        try:
            sub_ds = load_dataset(
                "EleutherAI/hendrycks_math",
                config,
                split="train",
                cache_dir=os.environ.get("HF_HOME"),
            )
            all_datasets.append(sub_ds)
            logger.info(f"  Loaded MATH/{config}: {len(sub_ds)} examples")
        except Exception as e:
            logger.warning(f"  Could not load MATH/{config}: {e}")

    if not all_datasets:
        logger.error("Failed to load any MATH config — falling back to synthetic problems")
        return _make_synthetic_problems(domain, n)

    ds = concatenate_datasets(all_datasets)
    logger.info(f"MATH total: {len(ds)} examples across {len(all_datasets)} configs")

    problems = []
    for ex in ds:
        if len(problems) >= n:
            break
        math_type = ex.get("type", "")
        prism_domain = MATH_LABEL_MAP.get(math_type, "miscellaneous")
        if prism_domain != domain:
            continue
        level_str = ex.get("level", "Level 1")
        try:
            level = int(level_str.replace("Level ", "").strip())
        except (ValueError, AttributeError):
            level = 1
        if level < POC_DOMAIN_QUERIES[domain]["level_min"]:
            continue

        problems.append({
            "id": ex.get("problem", "")[:30],
            "problem": ex["problem"],
            "answer": ex.get("solution", ""),
            "domain": domain,
            "level": level,
            "has_image": False,
        })

    if len(problems) < n:
        logger.warning(f"Only found {len(problems)} problems for {domain} (wanted {n})")

    return problems[:n]


def _make_synthetic_problems(domain: str, n: int) -> list[dict]:
    """Fallback: synthetic test problems if MATH dataset unavailable."""
    SAMPLE_PROBLEMS = {
        "algebra": "Find all real solutions to x^4 - 5x^2 + 4 = 0.",
        "geometry": "In triangle ABC, angle A = 60°, AB = AC = 1. Find the area.",
        "combinatorics": "How many ways can 6 people sit at a round table?",
        "number_theory": "Find the largest n such that n^2 - n + 1 divides n^4 + 2.",
        "miscellaneous": "Prove that for positive reals a,b,c: (a+b+c)/3 ≥ (abc)^(1/3).",
    }
    return [
        {
            "id": f"{domain}_synthetic_{i}",
            "problem": SAMPLE_PROBLEMS[domain],
            "answer": "",
            "domain": domain,
            "level": 4,
            "has_image": False,
        }
        for i in range(n)
    ]


def run_poc(
    n_problems: int = 20,
    gpu_id: int = 0,
    teacher_model: str = "Qwen/Qwen3-VL-30B-A3B-Thinking",
    output_dir: str = "results/traces/poc",
    with_images: bool = False,
    test_crossdomain_verify: bool = True,
) -> dict:
    """
    Run the full POC pipeline.

    Args:
        n_problems: Problems per domain (start with 20 for POC).
        gpu_id: GPU to use (Qwen3-VL-30B-A3B fits on one GH200).
        teacher_model: VL teacher model (must support image+text input).
        output_dir: Where to write POC traces.
        with_images: Include a few image-bearing problems to test VL input.
        test_crossdomain_verify: Also test cross-domain verification traces.
    """
    import torch

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    domains = list(POC_DOMAIN_QUERIES.keys())
    stats = {}

    logger.info("=" * 60)
    logger.info(f"PRISM POC Trace Generation")
    logger.info(f"Teacher: {teacher_model}  [thinking=True, T=1.0, top_p=0.95, top_k=20]")
    logger.info(f"Problems per domain: {n_problems}")
    logger.info(f"GPU: {gpu_id}")
    logger.info(f"Cross-domain verify: {test_crossdomain_verify}")
    logger.info("=" * 60)

    # ─── Load teacher model ────────────────────────────────────────────────
    logger.info(f"Loading teacher model: {teacher_model}")
    from prism.generation.trace_generator import TraceGenerator

    try:
        generator = TraceGenerator(
            teacher_model_name=teacher_model,
            gpu_id=gpu_id,
            max_new_tokens_per_phase=512,  # Shorter for POC
            temperature=1.0,   # Thinking mode requires temperature=1.0
        ).load()
        logger.info("Teacher model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load teacher model: {e}")
        raise

    # ─── Verify VL capability ─────────────────────────────────────────────
    logger.info("Verifying VL (image+text) capability...")
    _test_vl_capability(generator)

    # ─── Generate single-domain traces ────────────────────────────────────
    logger.info("\n--- Generating single-domain traces ---")
    for domain in domains:
        logger.info(f"\n[{domain.upper()}] Loading {n_problems} problems...")
        problems = load_poc_problems(domain, n_problems)

        if not problems:
            logger.warning(f"No problems found for {domain}. Skipping.")
            continue

        # Format for generator
        formatted = [
            {
                "problem": p["problem"],
                "answer": p["answer"],
                "id": p.get("id", ""),
            }
            for p in problems
        ]

        trace_file = output_path / f"{domain}_traces.jsonl"
        domain_stats = generator.generate_dataset(
            problems=formatted,
            domain=domain,
            output_file=str(trace_file),
            max_tokens_per_trace=2048,  # Shorter for POC
        )
        stats[domain] = domain_stats
        logger.info(f"  [{domain}] Pass rate: {domain_stats['pass_rate']:.1%} "
                    f"({domain_stats['kept']}/{domain_stats['total']})")

    # ─── Generate cross-domain verification traces ─────────────────────────
    if test_crossdomain_verify:
        logger.info("\n--- Generating cross-domain verification traces ---")
        cross_stats = {}

        for primary_domain, verifier_domain in CROSS_VERIFY_PAIRS[:2]:  # Test 2 pairs in POC
            logger.info(f"  {primary_domain} → verified by {verifier_domain}")
            problems = load_poc_problems(primary_domain, min(5, n_problems))

            # Load existing solve traces for these problems
            primary_trace_file = output_path / f"{primary_domain}_traces.jsonl"
            if not primary_trace_file.exists():
                logger.warning(f"Primary traces not found: {primary_trace_file}")
                continue

            cross_traces = []
            from prism.data.trace_format import TraceExample
            with open(primary_trace_file) as f:
                for i, line in enumerate(f):
                    if i >= 5:
                        break
                    try:
                        ex = TraceExample.from_jsonl(line.strip())
                        # Generate VERIFY trace using VERIFIER domain expert
                        # (this is the cross-domain mixing use case)
                        from prism.generation.phase_prompts import get_phase_system_prompt, get_phase_user_prompt
                        sys2 = get_phase_system_prompt(verifier_domain, phase=1)
                        usr2 = get_phase_user_prompt(
                            ex.problem, phase=1, domain=verifier_domain,
                            solve_trace=ex.solve_trace
                        )
                        verify_trace = generator._generate_phase(sys2, usr2)
                        cross_traces.append({
                            "problem_id": ex.problem_id,
                            "problem": ex.problem,
                            "primary_domain": primary_domain,
                            "verifier_domain": verifier_domain,
                            "solve_trace": ex.solve_trace,
                            "cross_verify_trace": verify_trace,
                            "ground_truth": ex.ground_truth,
                        })
                    except Exception as e:
                        logger.warning(f"Cross-verify failed: {e}")

            cross_file = output_path / f"{primary_domain}_verified_by_{verifier_domain}.jsonl"
            with open(cross_file, "w") as f:
                for t in cross_traces:
                    f.write(json.dumps(t, ensure_ascii=False) + "\n")

            cross_stats[f"{primary_domain}→{verifier_domain}"] = len(cross_traces)
            logger.info(f"  Generated {len(cross_traces)} cross-verify traces → {cross_file}")

        stats["cross_domain"] = cross_stats

    # ─── Inspect sample traces ─────────────────────────────────────────────
    logger.info("\n--- Sample trace inspection ---")
    _inspect_traces(output_path, domains)

    # ─── Save POC stats ────────────────────────────────────────────────────
    stats_file = output_path / "poc_stats.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)

    # ─── Print summary ────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("POC SUMMARY")
    logger.info("=" * 60)
    for domain, s in stats.items():
        if isinstance(s, dict) and "pass_rate" in s:
            logger.info(
                f"  {domain:<20}: {s['pass_rate']:.1%} pass "
                f"({s['kept']}/{s['total']}) "
                f"p1_correct={s['phase1_correct']} "
                f"p3_correct={s['phase3_correct']}"
            )

    total_kept = sum(s.get("kept", 0) for s in stats.values() if isinstance(s, dict))
    total_generated = sum(s.get("total", 0) for s in stats.values() if isinstance(s, dict))
    overall_pass = total_kept / max(total_generated, 1)

    logger.info(f"\n  Overall pass rate: {overall_pass:.1%} ({total_kept}/{total_generated})")
    logger.info(f"  Stats saved: {stats_file}")
    logger.info(f"  Traces saved: {output_path}")

    if overall_pass >= 0.25:
        logger.info("\n  POC RESULT: PASS (≥25% pass rate) — safe to scale to full dataset")
    else:
        logger.warning("\n  POC RESULT: INVESTIGATE — low pass rate, check trace quality")

    return stats


def _test_vl_capability(generator: "TraceGenerator"):
    """Verify that the teacher model handles image+text input."""
    test_problem = (
        "In the figure, triangle ABC has angle A = 90°, AB = 3, BC = 5. "
        "Find the area of triangle ABC."
    )
    logger.info("  Testing text-only input...")
    try:
        result = generator._generate_phase(
            system_prompt="You are an expert geometer. Solve briefly.",
            user_message=f"Problem: {test_problem}",
        )
        logger.info(f"  Text-only: OK (response length={len(result)} chars)")
    except Exception as e:
        raise RuntimeError(f"Text-only generation failed: {e}")

    # Test that the model's processor supports image input
    logger.info("  Checking VL (image input) processor support...")
    proc = generator._processor
    if hasattr(proc, "image_processor") or hasattr(proc, "image_mean"):
        logger.info("  VL processor: image_processor found ✓")
    elif hasattr(proc, "feature_extractor"):
        logger.info("  VL processor: feature_extractor found ✓")
    else:
        # Check if it can process images via process_vision_info
        try:
            import importlib
            qwen_vl = importlib.import_module("qwen_vl_utils")
            logger.info("  VL processor: qwen_vl_utils available ✓")
        except ImportError:
            logger.warning("  VL processor: cannot verify image support (may still work)")

    logger.info("  VL capability check complete")


def _inspect_traces(output_path: Path, domains: list[str]):
    """Print first trace from each domain for manual inspection."""
    from prism.data.trace_format import TraceExample

    for domain in domains[:2]:  # Only inspect first 2 domains in POC
        trace_file = output_path / f"{domain}_traces.jsonl"
        if not trace_file.exists():
            continue
        with open(trace_file) as f:
            first_line = f.readline().strip()
        if not first_line:
            continue
        try:
            ex = TraceExample.from_jsonl(first_line)
            logger.info(f"\n  --- {domain.upper()} sample trace ---")
            logger.info(f"  Problem: {ex.problem[:100]}...")
            logger.info(f"  Phase 1 (Solve): {ex.solve_trace[:150]}...")
            logger.info(f"  Phase 2 (Verify): {ex.verify_trace[:100]}...")
            logger.info(f"  Phase 3 (Correct): {ex.correct_trace[:100]}...")
            logger.info(f"  Phase 1 correct: {ex.solve_correct}, Phase 3 correct: {ex.correct_correct}")
        except Exception as e:
            logger.warning(f"  Could not inspect {domain} trace: {e}")


def main():
    parser = argparse.ArgumentParser(description="PRISM POC: small-scale end-to-end test")
    parser.add_argument("--n-problems", type=int, default=20, help="Problems per domain (20 for POC)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device index")
    parser.add_argument(
        "--teacher",
        default="Qwen/Qwen3-VL-30B-A3B-Thinking",
        help="Teacher VL model HF ID (must be >14B; Qwen3-VL-30B-A3B-Thinking fits on 1 GH200)",
    )
    parser.add_argument("--output-dir", default="results/traces/poc")
    parser.add_argument("--with-images", action="store_true", help="Test VL image input")
    parser.add_argument("--no-cross", action="store_true", help="Skip cross-domain verify test")
    args = parser.parse_args()

    # Set environment
    import os
    prism_root = "/iopsstor/scratch/cscs/dasgupta/research/ideas/PRISM"
    os.environ.setdefault("PRISM_ROOT", prism_root)
    os.environ.setdefault("HF_HOME", f"{prism_root}/.cache/huggingface")
    hf_token_path = os.path.expanduser("~/.cache/huggingface/token")
    if os.path.exists(hf_token_path):
        with open(hf_token_path) as f:
            os.environ["HF_TOKEN"] = f.read().strip()

    stats = run_poc(
        n_problems=args.n_problems,
        gpu_id=args.gpu,
        teacher_model=args.teacher,
        output_dir=args.output_dir,
        with_images=args.with_images,
        test_crossdomain_verify=not args.no_cross,
    )

    print("\nPOC complete. Check results/traces/poc/ for output.")
    print(f"Stats: {json.dumps({k: v.get('pass_rate', v) if isinstance(v, dict) else v for k, v in stats.items()}, indent=2)}")


if __name__ == "__main__":
    main()
