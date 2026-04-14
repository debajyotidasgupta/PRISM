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
    Extract the final answer from PRISM-generated text.

    PRISM generates three phases separated by double-newlines:
        [solve_trace] \\n\\n [verify_trace] \\n\\n [correct_trace \\boxed{ans}]

    Strategy:
      1. Deduplicate repetition loops first.
      2. Prefer the LAST phase (correct_trace) — most likely to have the final answer.
      3. Extract the last \\boxed{...} from the correct_trace section first;
         if not found there, scan earlier phases (fallback).
      4. Handle truncated \\boxed{X (no closing brace) from token cutoff.
      5. If no boxed at all, return last numeric-looking token.
    """
    text = _dedup_repetition(text)

    # Split by double-newline to separate phases
    # Take the LAST non-empty section as the most likely correct_trace
    sections = [s.strip() for s in text.split('\n\n') if s.strip()]

    def _extract_last_boxed(s: str):
        """Extract last \\boxed{} from a string; None if not found."""
        matches = list(re.finditer(r'\\boxed\{', s))
        if not matches:
            return None
        m = matches[-1]
        start = m.end()
        depth, i = 1, start
        while i < len(s) and depth > 0:
            if s[i] == '{':
                depth += 1
            elif s[i] == '}':
                depth -= 1
            i += 1
        if depth == 0:
            return s[start:i - 1].strip()
        # Truncated — take whatever is inside
        content = s[start:].rstrip()
        return content if content else None

    # Try last section first, then work backwards (correct → verify → solve)
    for section in reversed(sections):
        result = _extract_last_boxed(section)
        if result is not None:
            return result

    # No boxed found anywhere — last numeric token
    tokens = text.strip().split()
    for tok in reversed(tokens):
        tok_clean = tok.strip('.,;:')
        if tok_clean and (tok_clean.lstrip('-').replace('.', '', 1).isdigit()
                          or re.match(r'^[\d\\/\^_{}]+$', tok_clean)):
            return tok_clean
    return tokens[-1] if tokens else ""


def _dedup_repetition(text: str, min_len: int = 8, max_reps: int = 2) -> str:
    """
    Detect and truncate repetition loops in generated text.
    If a substring of length >= min_len repeats > max_reps times consecutively,
    keep only the first occurrence.
    """
    # Find the first repeated block of meaningful length
    n = len(text)
    for block_len in range(min(200, n // 3), min_len - 1, -1):
        for start in range(n - block_len * (max_reps + 1) + 1):
            block = text[start:start + block_len]
            # Check if block repeats at least max_reps+1 times starting here
            pos = start
            reps = 0
            while pos + block_len <= n and text[pos:pos + block_len] == block:
                reps += 1
                pos += block_len
            if reps > max_reps:
                # Keep text up to and including the first occurrence of the block
                return text[:start + block_len]
    return text


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
    max_new_tokens: int = 1536,
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

            # Prompt format MUST match training format exactly:
            #   training: [BOS][raw problem]\n\n[solve]\n\n[verify]\n\n[correct][EOS]
            #   eval:     [BOS][raw problem]\n\n  ← model continues with solve phase
            # "Problem:" prefix and "Solution:" suffix never appear in training.
            prompt = ex['problem'] + "\n\n"
            enc = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=768,        # longer problem budget (some problems are verbose)
                add_special_tokens=True,
            ).to(device)

            # Generate the full solve→verify→correct chain.
            # avg chain length: 520-1040 tokens; 1536 covers 99th percentile.
            # Greedy (temp=0) is more stable for multi-phase structured output;
            # repetition_penalty=1.15 gently discourages loops without scrambling math.
            out_ids = model.generate(
                enc["input_ids"],
                attention_mask=enc["attention_mask"],
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                repetition_penalty=1.15,
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
