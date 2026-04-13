"""
Evaluation metrics for PRISM.

Primary: exact match accuracy (after answer normalization).
Secondary: partial credit for multi-part problems.
"""

import re
import math
import logging
from typing import Optional, Union

logger = logging.getLogger(__name__)


def normalize_answer(answer: str) -> str:
    """
    Normalize an answer string for comparison.

    Handles:
    - LaTeX formatting: \\boxed{x}, \\frac{a}{b}, etc.
    - Whitespace
    - Trailing zeros in decimals
    - Common equivalent forms
    """
    if answer is None:
        return ""

    s = str(answer).strip()

    # Extract from \\boxed{...}
    boxed = re.search(r"\\boxed\{([^}]+)\}", s)
    if boxed:
        s = boxed.group(1).strip()

    # Remove LaTeX commands but keep content
    s = re.sub(r"\\dfrac\{([^}]*)\}\{([^}]*)\}", r"\1/\2", s)
    s = re.sub(r"\\frac\{([^}]*)\}\{([^}]*)\}", r"\1/\2", s)
    s = re.sub(r"\\left\(", "(", s)
    s = re.sub(r"\\right\)", ")", s)
    s = re.sub(r"\\left\[", "[", s)
    s = re.sub(r"\\right\]", "]", s)
    s = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\[a-zA-Z]+", "", s)
    s = re.sub(r"[{}\$]", "", s)

    # Normalize whitespace
    s = re.sub(r"\s+", "", s)

    # Lowercase
    s = s.lower()

    # Remove trailing zeros: 3.50 → 3.5, 3.0 → 3
    s = re.sub(r"\.0+$", "", s)
    s = re.sub(r"(\.\d*[1-9])0+$", r"\1", s)

    return s


def exact_match(predicted: str, ground_truth: str) -> bool:
    """
    Check if predicted answer exactly matches ground truth after normalization.
    """
    pred_norm = normalize_answer(predicted)
    gt_norm = normalize_answer(ground_truth)

    if pred_norm == gt_norm:
        return True

    # Try numeric comparison
    try:
        from fractions import Fraction
        pred_frac = Fraction(pred_norm)
        gt_frac = Fraction(gt_norm)
        return pred_frac == gt_frac
    except (ValueError, ZeroDivisionError):
        pass

    # Try float comparison with small tolerance
    try:
        pred_float = float(eval(pred_norm.replace("^", "**")))
        gt_float = float(eval(gt_norm.replace("^", "**")))
        return abs(pred_float - gt_float) < 1e-6
    except Exception:
        pass

    return False


def extract_answer_from_text(text: str) -> str:
    """
    Extract the final answer from a model's generated text.
    Tries multiple patterns in order of confidence.
    """
    # \\boxed{...} — highest confidence
    boxed = re.search(r"\\boxed\{([^}]+)\}", text)
    if boxed:
        return boxed.group(1).strip()

    # "The answer is X"
    pattern1 = re.search(
        r"(?:the\s+(?:final\s+)?answer\s+is|answer:\s*)\**([^\n.,]+)",
        text,
        re.IGNORECASE,
    )
    if pattern1:
        return pattern1.group(1).strip()

    # "= X" at end of solution
    eq_pattern = re.search(r"=\s*([^\n=]+)$", text.strip())
    if eq_pattern:
        return eq_pattern.group(1).strip()

    # Last standalone number
    numbers = re.findall(r"(?<!\w)(-?\d+(?:\.\d+)?(?:/\d+)?)(?!\w)", text)
    if numbers:
        return numbers[-1]

    return ""


def partial_credit(predicted_parts: list[str], ground_truth_parts: list[str]) -> float:
    """
    Compute partial credit for multi-part problems.

    Args:
        predicted_parts: List of predicted answers for each part.
        ground_truth_parts: List of ground truth answers.

    Returns:
        Fraction of parts correct (0.0 to 1.0).
    """
    if not ground_truth_parts:
        return 0.0
    if len(predicted_parts) != len(ground_truth_parts):
        # Pad or truncate predictions
        predicted_parts = (predicted_parts + [""] * len(ground_truth_parts))[: len(ground_truth_parts)]

    correct = sum(
        1 for p, g in zip(predicted_parts, ground_truth_parts) if exact_match(p, g)
    )
    return correct / len(ground_truth_parts)


def compute_accuracy(
    predictions: list[str],
    ground_truths: list[str],
) -> dict:
    """
    Compute exact match accuracy over a list of predictions.

    Returns dict with: accuracy, n_correct, n_total, wrong_indices.
    """
    assert len(predictions) == len(ground_truths)
    n_correct = 0
    wrong_indices = []

    for i, (pred, gt) in enumerate(zip(predictions, ground_truths)):
        if exact_match(pred, gt):
            n_correct += 1
        else:
            wrong_indices.append(i)

    return {
        "accuracy": n_correct / max(len(predictions), 1),
        "n_correct": n_correct,
        "n_total": len(predictions),
        "wrong_indices": wrong_indices,
    }
