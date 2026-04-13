"""
POC: End-to-end trace generation test.

Tests the complete trace generation pipeline on a small subset (n_problems per domain)
BEFORE scaling to the full dataset.

What this validates:
  1. Teacher model loads correctly on each GPU
  2. Reference-solution reformulation works (no solving from scratch)
  3. All 3 phase prompts produce coherent, domain-aligned expert traces
  4. Both single-domain and cross-domain verification traces work
  5. Quality filter correctly identifies correct/incorrect traces
  6. JSONL output is parseable by the data pipeline

Teacher: Qwen/Qwen3.5-35B-A3B (primary, 35B MoE, ~3.5B active — fits on 1 GH200)
  Fallback: Qwen/Qwen3-VL-30B-A3B-Thinking
  Both: enable_thinking=True, temperature=1.0, top_p=0.95, top_k=20, presence_penalty=1.5

GPU strategy:
  Default: 4 GPUs in parallel — one teacher instance per GPU, each handling one domain.
  5 domains → round 1: 4 domains on GPUs 0-3, round 2: 5th domain on GPU 0.
  This maximises GPU utilisation even in the POC.

Usage:
  # Single GPU (for debugging):
  python scripts/poc_trace_gen.py --n-problems 20 --gpu 0 --no-parallel

  # Multi-GPU (default, uses all 4 GPUs):
  python scripts/poc_trace_gen.py --n-problems 20
"""

import os
import sys
import json
import time
import logging
import argparse
import subprocess
from pathlib import Path

# Ensure the package is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# POC uses Level 4-5 problems (hardest) from the MATH dataset
POC_DOMAIN_QUERIES = {
    "algebra": {"level_min": 4},
    "geometry": {"level_min": 4},
    "combinatorics": {"level_min": 4},
    "number_theory": {"level_min": 4},
    "miscellaneous": {"level_min": 4},
}

# Cross-domain verification pairs for trace diversity
CROSS_VERIFY_PAIRS = [
    ("algebra", "miscellaneous"),        # Algebraic bounds verified by inequality tools
    ("geometry", "algebra"),             # Coordinate geometry verified algebraically
    ("combinatorics", "number_theory"),  # Counting patterns verified by NT
    ("number_theory", "algebra"),        # Diophantine equations verified algebraically
]


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_poc_problems(domain: str, n: int) -> list[dict]:
    """Load n problems for the POC from MATH dataset (EleutherAI/hendrycks_math)."""
    os.environ.setdefault("HF_HOME", "/iopsstor/scratch/cscs/dasgupta/research/ideas/PRISM/.cache/huggingface")

    from datasets import load_dataset, concatenate_datasets
    from prism.data.datasets import MATH_LABEL_MAP

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
        except Exception as e:
            logger.warning(f"  Could not load MATH/{config}: {e}")

    if not all_datasets:
        logger.error("Failed to load any MATH config — using synthetic fallback")
        return _make_synthetic_problems(domain, n)

    ds = _concatenate(all_datasets)
    level_min = POC_DOMAIN_QUERIES[domain]["level_min"]

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
        if level < level_min:
            continue

        problems.append({
            "id": ex.get("problem", "")[:30],
            "problem": ex["problem"],
            # Full worked solution: provided to Phase 1 for expert reformulation
            "solution": ex.get("solution", ""),
            "answer": ex.get("solution", ""),   # answer = full solution; answer extracted downstream
            "domain": domain,
            "level": level,
            "has_image": False,
        })

    if len(problems) < n:
        logger.warning(f"Only {len(problems)} problems found for {domain} (wanted {n})")
    return problems[:n]


def _concatenate(datasets):
    from datasets import concatenate_datasets
    return concatenate_datasets(datasets)


def _make_synthetic_problems(domain: str, n: int) -> list[dict]:
    """Emergency fallback: synthetic test problems if MATH dataset is unavailable."""
    SAMPLE = {
        "algebra": ("Find all real solutions to x^4 - 5x^2 + 4 = 0.",
                    "x^4 - 5x^2 + 4 = (x^2-1)(x^2-4) = 0, so x = ±1, ±2."),
        "geometry": ("In triangle ABC, angle A = 60°, AB = AC = 1. Find the area.",
                     "Height h = sin(60°) = √3/2. Area = 1/2 · 1 · √3/2 = √3/4."),
        "combinatorics": ("How many ways can 6 people sit at a round table?",
                          "(6-1)! = 5! = 120."),
        "number_theory": ("Find the largest n such that n^2 - n + 1 divides n^4 + 2.",
                          "n^4 + 2 = (n^2-n+1)(n^2+n) + n + 2. So n^2-n+1 | n+2. For n≥3 this fails. Check n=1,2,3."),
        "miscellaneous": ("Prove that for positive reals a,b,c: (a+b+c)/3 ≥ (abc)^(1/3).",
                          "By AM-GM: (a+b+c)/3 ≥ (abc)^(1/3)."),
    }
    prob_text, sol_text = SAMPLE.get(domain, SAMPLE["miscellaneous"])
    return [
        {
            "id": f"{domain}_synthetic_{i}",
            "problem": prob_text,
            "solution": sol_text,
            "answer": sol_text,
            "domain": domain,
            "level": 4,
            "has_image": False,
        }
        for i in range(n)
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Single-GPU run (used by worker subprocesses)
# ──────────────────────────────────────────────────────────────────────────────

def run_single_domain(
    domain: str,
    gpu_id: int,
    teacher_model: str,
    n_problems: int,
    output_dir: str,
    cross_verify_domain: str = None,
) -> dict:
    """
    Generate traces for one domain on one GPU.
    Called either directly (single-GPU mode) or as a subprocess worker.
    """
    from prism.generation.trace_generator import TraceGenerator

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"[GPU{gpu_id}] Loading teacher for domain={domain}...")
    generator = TraceGenerator(
        teacher_model_name=teacher_model,
        gpu_id=gpu_id,
        max_new_tokens_per_phase=512,   # Shorter for POC
        temperature=1.0,
    ).load()

    # Quick VL sanity check on first worker only (gpu_id == 0)
    if gpu_id == 0 and cross_verify_domain is None:
        _test_generation(generator)

    problems = load_poc_problems(domain, n_problems)
    if not problems:
        logger.warning(f"[GPU{gpu_id}] No problems for {domain}")
        return {"domain": domain, "kept": 0, "total": 0, "pass_rate": 0.0}

    suffix = f"_cv_{cross_verify_domain}" if cross_verify_domain else ""
    trace_file = output_path / f"{domain}{suffix}_traces.jsonl"

    stats = generator.generate_dataset(
        problems=problems,
        domain=domain,
        output_file=str(trace_file),
        max_tokens_per_trace=2048,
        cross_verify_domain=cross_verify_domain,
    )
    logger.info(
        f"[GPU{gpu_id}] {domain}{suffix}: "
        f"pass={stats['pass_rate']:.1%} ({stats['kept']}/{stats['total']})"
    )

    # Write stats to a JSON sidecar for the orchestrator to collect
    stats_file = output_path / f"{domain}{suffix}_stats.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)

    return stats


# ──────────────────────────────────────────────────────────────────────────────
# Multi-GPU orchestrator
# ──────────────────────────────────────────────────────────────────────────────

def run_poc_parallel(
    n_problems: int = 20,
    n_gpus: int = 4,
    teacher_model: str = "Qwen/Qwen3.5-35B-A3B",
    output_dir: str = "results/traces/poc",
    test_crossdomain_verify: bool = True,
) -> dict:
    """
    Orchestrate POC across n_gpus GPUs using subprocesses.

    Each GPU gets its own teacher model instance and generates one domain.
    Round 1: first 4 domains in parallel (GPUs 0-3).
    Round 2: 5th domain on GPU 0 (+ cross-verify pairs if enabled).
    """
    prism_root = "/iopsstor/scratch/cscs/dasgupta/research/ideas/PRISM"
    log_dir = Path("/tmp/prism_logs")
    log_dir.mkdir(exist_ok=True)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    domains = list(POC_DOMAIN_QUERIES.keys())  # 5 domains
    all_tasks = []

    # Build task list: (domain, gpu_id, cross_verify_domain)
    # Round 1: domains 0..n_gpus-1 on GPUs 0..n_gpus-1 simultaneously
    for i in range(min(n_gpus, len(domains))):
        all_tasks.append((domains[i], i, None))

    # Remaining domains run on GPU 0 after round 1
    for i in range(n_gpus, len(domains)):
        all_tasks.append((domains[i], 0, None))

    # Cross-verify pairs (run after primary traces)
    cv_tasks = []
    if test_crossdomain_verify:
        for j, (prim, verif) in enumerate(CROSS_VERIFY_PAIRS[:4]):
            cv_tasks.append((prim, j % n_gpus, verif))

    logger.info("=" * 60)
    logger.info("PRISM POC — parallel multi-GPU trace generation")
    logger.info(f"Teacher : {teacher_model}")
    logger.info(f"N/domain: {n_problems}  |  GPUs: {n_gpus}")
    logger.info(f"Tasks   : {len(all_tasks)} primary + {len(cv_tasks)} cross-verify")
    logger.info("=" * 60)

    def _launch(task_list, round_name):
        """Launch a batch of tasks as subprocesses, wait for all to finish."""
        logger.info(f"\n── {round_name} ──")
        procs = {}
        for domain, gpu_id, cv in task_list:
            suffix = f"_cv_{cv}" if cv else ""
            log_file = log_dir / f"poc_{domain}{suffix}.log"
            cmd = [
                sys.executable, "-m", "prism.generation.trace_generator",
                "--domain", domain,
                "--gpu", str(gpu_id),
                "--teacher", teacher_model,
                "--n-problems", str(n_problems),
                "--output-dir", output_dir,
                "--max-tokens", "2048",
            ]
            if cv:
                cmd += ["--cross-verify-domain", cv]
            env = {**os.environ, "PRISM_ROOT": prism_root}
            proc = subprocess.Popen(
                cmd,
                stdout=open(log_file, "w"),
                stderr=subprocess.STDOUT,
                env=env,
            )
            key = f"{domain}{suffix}"
            procs[key] = (proc, log_file)
            logger.info(f"  Launched {key} on GPU{gpu_id}  PID={proc.pid}  log={log_file}")

        for key, (proc, log_file) in procs.items():
            rc = proc.wait()
            status = "OK" if rc == 0 else f"FAILED(rc={rc})"
            logger.info(f"  {key}: {status}")
        return list(procs.keys())

    # ── Round 1: first n_gpus primary domains ─────────────────────────────
    round1_tasks = [t for t in all_tasks if t[1] != 0 or len(all_tasks) <= n_gpus][:n_gpus]
    _launch(round1_tasks, f"Round 1: {n_gpus} primary domains in parallel")

    # ── Round 2: remaining primary domains (if any) ────────────────────────
    round2_tasks = all_tasks[n_gpus:]
    if round2_tasks:
        _launch(round2_tasks, "Round 2: remaining primary domains")

    # ── Round 3: cross-verify pairs ────────────────────────────────────────
    if cv_tasks:
        _launch(cv_tasks, "Round 3: cross-domain verification pairs")

    # ── Collect stats ──────────────────────────────────────────────────────
    stats = {}
    for domain, _, cv in all_tasks + cv_tasks:
        suffix = f"_cv_{cv}" if cv else ""
        stats_file = output_path / f"{domain}{suffix}_stats.json"
        if stats_file.exists():
            with open(stats_file) as f:
                stats[f"{domain}{suffix}"] = json.load(f)

    # ── Inspect sample traces ──────────────────────────────────────────────
    logger.info("\n--- Sample trace inspection ---")
    _inspect_traces(output_path, domains[:2])

    # ── Summary ────────────────────────────────────────────────────────────
    _print_summary(stats, output_path)
    return stats


# ──────────────────────────────────────────────────────────────────────────────
# Single-GPU sequential mode (debugging / single-GPU machines)
# ──────────────────────────────────────────────────────────────────────────────

def run_poc_sequential(
    n_problems: int = 20,
    gpu_id: int = 0,
    teacher_model: str = "Qwen/Qwen3.5-35B-A3B",
    output_dir: str = "results/traces/poc",
    test_crossdomain_verify: bool = True,
) -> dict:
    """Single-GPU sequential fallback (use --no-parallel for debugging)."""
    domains = list(POC_DOMAIN_QUERIES.keys())
    stats = {}

    logger.info("=" * 60)
    logger.info("PRISM POC — single-GPU sequential mode")
    logger.info(f"Teacher : {teacher_model}")
    logger.info(f"GPU     : {gpu_id}  |  N/domain: {n_problems}")
    logger.info("=" * 60)

    for domain in domains:
        s = run_single_domain(domain, gpu_id, teacher_model, n_problems, output_dir)
        stats[domain] = s

    if test_crossdomain_verify:
        for prim, verif in CROSS_VERIFY_PAIRS[:2]:
            s = run_single_domain(prim, gpu_id, teacher_model, min(5, n_problems),
                                   output_dir, cross_verify_domain=verif)
            stats[f"{prim}_cv_{verif}"] = s

    _inspect_traces(Path(output_dir), domains[:2])
    _print_summary(stats, Path(output_dir))
    return stats


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _test_generation(generator):
    """Verify basic text generation works."""
    try:
        result = generator._generate_phase(
            system_prompt="You are an expert mathematician. Be concise.",
            user_message="Problem: What is 2+2?\nCorrect solution (reformulate): The answer is 4.",
        )
        logger.info(f"  Generation test: OK (len={len(result)} chars)")
    except Exception as e:
        raise RuntimeError(f"Generation test failed: {e}")


def _inspect_traces(output_path: Path, domains: list):
    """Print first trace from each domain for manual inspection."""
    from prism.data.trace_format import TraceExample

    for domain in domains:
        trace_file = output_path / f"{domain}_traces.jsonl"
        if not trace_file.exists():
            continue
        with open(trace_file) as f:
            first_line = f.readline().strip()
        if not first_line:
            continue
        try:
            ex = TraceExample.from_jsonl(first_line)
            logger.info(f"\n  ── {domain.upper()} sample ──")
            logger.info(f"  Problem : {ex.problem[:120]}...")
            logger.info(f"  Phase 1 : {ex.solve_trace[:180]}...")
            logger.info(f"  Phase 2 : {ex.verify_trace[:120]}...")
            logger.info(f"  Phase 3 : {ex.correct_trace[:120]}...")
            logger.info(f"  P1_ok={ex.solve_correct}  P3_ok={ex.correct_correct}")
        except Exception as e:
            logger.warning(f"  Could not inspect {domain}: {e}")


def _print_summary(stats: dict, output_path: Path):
    stats_file = output_path / "poc_stats.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info("\n" + "=" * 60)
    logger.info("POC SUMMARY")
    logger.info("=" * 60)
    total_kept = total_gen = 0
    for key, s in stats.items():
        if isinstance(s, dict) and "pass_rate" in s:
            kept, total = s.get("kept", 0), s.get("total", 0)
            total_kept += kept
            total_gen += total
            logger.info(
                f"  {key:<30}: {s['pass_rate']:.1%} pass "
                f"({kept}/{total})  "
                f"p1={s.get('phase1_correct', '?')} p3={s.get('phase3_correct', '?')}"
            )
    overall = total_kept / max(total_gen, 1)
    logger.info(f"\n  Overall: {overall:.1%} ({total_kept}/{total_gen})")
    logger.info(f"  Stats  : {stats_file}")
    if overall >= 0.25:
        logger.info("  POC RESULT: PASS — safe to scale to full dataset")
    else:
        logger.warning("  POC RESULT: INVESTIGATE — low pass rate")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PRISM POC: multi-GPU end-to-end test")
    parser.add_argument("--n-problems", type=int, default=20, help="Problems per domain")
    parser.add_argument("--gpu", type=int, default=0, help="GPU for single-GPU mode only")
    parser.add_argument("--n-gpus", type=int, default=4, help="Number of GPUs for parallel mode")
    parser.add_argument(
        "--teacher",
        default="Qwen/Qwen3.5-35B-A3B",
        help="Teacher model (>14B; fallback: Qwen/Qwen3-VL-30B-A3B-Thinking)",
    )
    parser.add_argument("--output-dir", default="results/traces/poc")
    parser.add_argument("--no-parallel", action="store_true",
                        help="Single-GPU sequential mode (for debugging)")
    parser.add_argument("--no-cross", action="store_true",
                        help="Skip cross-domain verification traces")
    args = parser.parse_args()

    prism_root = "/iopsstor/scratch/cscs/dasgupta/research/ideas/PRISM"
    os.environ.setdefault("PRISM_ROOT", prism_root)
    os.environ.setdefault("HF_HOME", f"{prism_root}/.cache/huggingface")
    hf_token_path = os.path.expanduser("~/.cache/huggingface/token")
    if os.path.exists(hf_token_path):
        with open(hf_token_path) as f:
            os.environ["HF_TOKEN"] = f.read().strip()

    if args.no_parallel:
        stats = run_poc_sequential(
            n_problems=args.n_problems,
            gpu_id=args.gpu,
            teacher_model=args.teacher,
            output_dir=args.output_dir,
            test_crossdomain_verify=not args.no_cross,
        )
    else:
        stats = run_poc_parallel(
            n_problems=args.n_problems,
            n_gpus=args.n_gpus,
            teacher_model=args.teacher,
            output_dir=args.output_dir,
            test_crossdomain_verify=not args.no_cross,
        )

    print(f"\nPOC complete. Traces in {args.output_dir}/")


if __name__ == "__main__":
    main()
