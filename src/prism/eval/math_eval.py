"""
Math evaluation utilities for PRISM pilot experiments.

Uses MATH-500 dataset with proper subject classification and answer matching.
"""

import re
import torch
from datasets import load_dataset


# MATH-500 subject → PRISM domain mapping
SUBJECT_TO_DOMAIN = {
    "Algebra": "algebra",
    "Intermediate Algebra": "algebra",
    "Prealgebra": "algebra",
    "Geometry": "geometry",
    "Counting & Probability": "combinatorics",
    "Number Theory": "number_theory",
    "Precalculus": "miscellaneous",
}

DOMAINS = ["algebra", "geometry", "combinatorics", "number_theory", "miscellaneous"]


def load_math500_by_domain(n_per_domain: int = 25) -> list[dict]:
    """
    Load MATH-500 problems with correct subject-based domain classification.
    Returns exactly n_per_domain problems per domain (or all available if fewer).
    """
    import os
    os.environ.setdefault("HF_HOME", ".cache/huggingface")
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")

    domain_pools = {d: [] for d in DOMAINS}
    for ex in ds:
        subj = ex.get("subject", "")
        domain = SUBJECT_TO_DOMAIN.get(subj, "miscellaneous")
        domain_pools[domain].append(ex)

    problems = []
    for d in DOMAINS:
        problems.extend(domain_pools[d][:n_per_domain])

    return problems


def extract_boxed(text: str) -> str:
    """
    Extract the content of the last \\boxed{...} in generated text.
    Handles nested braces up to depth 3.
    Falls back to last token if no boxed found.
    """
    # Try to find \boxed{...} — note single backslash in the actual string
    # r'\\boxed' in raw string = regex \\boxed = matches literal \boxed
    matches = list(re.finditer(r'\\boxed\{', text))
    if matches:
        # Take the LAST match (most likely the final answer)
        m = matches[-1]
        start = m.end()  # position after {
        depth = 1
        i = start
        while i < len(text) and depth > 0:
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
            i += 1
        if depth == 0:
            return text[start:i-1].strip()

    # Fallback: last non-empty token
    tokens = text.strip().split()
    return tokens[-1] if tokens else ""


def normalize_answer(ans: str) -> str:
    """
    Normalize a math answer string for comparison.
    Removes whitespace, normalizes common LaTeX patterns.
    """
    # Remove all whitespace
    ans = re.sub(r'\s+', '', ans)
    # Normalize negative signs
    ans = ans.replace('−', '-')
    # Lowercase
    ans = ans.lower()
    # Remove trailing/leading dollar signs
    ans = ans.strip('$')
    return ans


def answers_match(pred: str, gold: str) -> bool:
    """Check if predicted answer matches gold answer."""
    p = normalize_answer(pred)
    g = normalize_answer(gold)
    if p == g:
        return True
    # Try numeric comparison for simple numbers
    try:
        return abs(float(p) - float(g)) < 1e-6
    except (ValueError, TypeError):
        pass
    return False


def evaluate_model(
    model,
    tokenizer,
    n_per_domain: int = 25,
    max_new_tokens: int = 256,
    batch_size: int = 1,
    device: str = "cuda:0",
    log_fn=None,
) -> dict:
    """
    Evaluate PRISM model on MATH-500.

    Args:
        model: PRISMModel (eval mode, loaded with blocks).
        tokenizer: Backbone tokenizer.
        n_per_domain: Problems per domain.
        max_new_tokens: Max generation length.
        batch_size: Evaluation batch size (1 for PRISM due to no KV-cache).
        device: CUDA device.
        log_fn: Optional logging function.

    Returns:
        Dict with overall and per-domain accuracy.
    """
    if log_fn is None:
        import logging
        log_fn = logging.getLogger(__name__).info

    problems = load_math500_by_domain(n_per_domain)
    log_fn(f"Evaluating on {len(problems)} problems ({n_per_domain}/domain)")

    correct_by_domain = {d: 0 for d in DOMAINS}
    total_by_domain = {d: 0 for d in DOMAINS}

    model.eval()
    with torch.no_grad():
        for i, ex in enumerate(problems):
            subj = ex.get("subject", "")
            domain = SUBJECT_TO_DOMAIN.get(subj, "miscellaneous")
            total_by_domain[domain] += 1

            # Prompt: problem → solution
            prompt = f"Problem: {ex['problem']}\n\nSolution:"
            enc = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(device)

            out_ids = model.generate(
                enc["input_ids"],
                attention_mask=enc["attention_mask"],
                max_new_tokens=max_new_tokens,
                temperature=0.0,
            )
            generated = tokenizer.decode(
                out_ids[0][enc["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )

            pred = extract_boxed(generated)
            gold = ex.get("answer", "")

            if answers_match(pred, gold):
                correct_by_domain[domain] += 1

            if (i + 1) % 16 == 0:
                so_far = sum(correct_by_domain.values())
                log_fn(f"  {i+1}/{len(problems)} done — running: {so_far}/{i+1} = {so_far/(i+1)*100:.1f}%")

    total_correct = sum(correct_by_domain.values())
    total = sum(total_by_domain.values())
    overall = total_correct / max(total, 1)

    return {
        "overall": round(overall, 4),
        "domain": {d: round(correct_by_domain[d] / max(total_by_domain[d], 1), 4) for d in DOMAINS},
        "correct": {d: correct_by_domain[d] for d in DOMAINS},
        "total": {d: total_by_domain[d] for d in DOMAINS},
        "n_total": total,
    }
