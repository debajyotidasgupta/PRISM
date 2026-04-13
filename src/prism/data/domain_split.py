"""
Domain classification: map problem text → PRISM domain label.

Two approaches:
  1. Rule-based classifier (fast, uses keyword matching) — used for initial filtering
  2. DomainClassifier (neural, fine-tuned on labeled data) — used for unlabeled datasets

The 5 PRISM domains:
  0 = algebra
  1 = geometry
  2 = combinatorics
  3 = number_theory
  4 = miscellaneous
"""

import re
import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

DOMAIN_NAMES = ["algebra", "geometry", "combinatorics", "number_theory", "miscellaneous"]
DOMAIN_TO_IDX = {d: i for i, d in enumerate(DOMAIN_NAMES)}

# Keyword sets for rule-based classification
_ALGEBRA_KEYWORDS = {
    "polynomial", "roots", "equation", "function", "substitut", "factor",
    "vieta", "functional equation", "coefficient", "quadratic", "expand",
    "simplif", "algebraic", "expression", "variable", "sum of roots",
    "completing the square", "rational root", "remainder theorem",
}
_GEOMETRY_KEYWORDS = {
    "triangle", "circle", "angle", "perpendicular", "parallel", "polygon",
    "inscribed", "circumscribed", "tangent", "chord", "radius", "diameter",
    "similarity", "congruent", "area", "perimeter", "altitude", "median",
    "orthocenter", "incenter", "circumcenter", "centroid", "cyclic", "ceva",
    "menelaus", "inversion", "harmonic", "cross-ratio", "projective",
}
_COMBINATORICS_KEYWORDS = {
    "count", "arrange", "choose", "select", "permutation", "combination",
    "bijection", "pigeonhole", "graph", "coloring", "tournament", "tree",
    "path", "cycle", "hamilton", "bipartite", "matching", "hall",
    "inclusion-exclusion", "expected value", "probability", "sample space",
    "recursion", "recurrence", "induction", "tiling", "partition",
}
_NUMBER_THEORY_KEYWORDS = {
    "prime", "divisib", "modular", "congruence", "gcd", "lcm", "coprime",
    "fermat", "euler", "chinese remainder", "diophantine", "p-adic",
    "quadratic residue", "reciprocity", "lifting", "lte", "valuation",
    "integer", "number theory", "floor", "ceiling",
}
_MISC_KEYWORDS = {
    "inequality", "am-gm", "cauchy-schwarz", "jensen", "holder", "muirhead",
    "generating function", "power series", "calculus", "derivative", "integral",
    "convex", "concave", "linear algebra", "matrix", "determinant", "complex",
    "roots of unity", "probability distribution", "variance",
}


def classify_domain(problem_text: str) -> str:
    """
    Rule-based domain classification using keyword matching.

    Returns the most likely PRISM domain string.
    """
    text = problem_text.lower()
    scores = {
        "algebra": 0,
        "geometry": 0,
        "combinatorics": 0,
        "number_theory": 0,
        "miscellaneous": 0,
    }
    for kw in _ALGEBRA_KEYWORDS:
        if kw in text:
            scores["algebra"] += 1
    for kw in _GEOMETRY_KEYWORDS:
        if kw in text:
            scores["geometry"] += 1
    for kw in _COMBINATORICS_KEYWORDS:
        if kw in text:
            scores["combinatorics"] += 1
    for kw in _NUMBER_THEORY_KEYWORDS:
        if kw in text:
            scores["number_theory"] += 1
    for kw in _MISC_KEYWORDS:
        if kw in text:
            scores["miscellaneous"] += 1

    # Miscellaneous wins only if its score is significantly higher
    # or if no other domain has a clear signal
    max_domain = max(scores, key=scores.get)
    max_score = scores[max_domain]

    if max_score == 0:
        return "miscellaneous"
    if max_domain == "miscellaneous" and any(
        scores[d] >= scores["miscellaneous"] - 1
        for d in ["algebra", "geometry", "combinatorics", "number_theory"]
    ):
        # Tie: prefer a primary domain, not miscellaneous
        primary_scores = {
            d: scores[d]
            for d in ["algebra", "geometry", "combinatorics", "number_theory"]
        }
        max_domain = max(primary_scores, key=primary_scores.get)

    return max_domain


def get_soft_domain_label(problem_text: str, n_domains: int = 5) -> list[float]:
    """
    Produce a soft domain label vector for mixed-domain problems.
    Returns a list of floats summing to 1.0.

    For a problem requiring, e.g., geometry + AM-GM:
      → [0.0, 0.6, 0.0, 0.0, 0.4]
    """
    text = problem_text.lower()
    raw_scores = {
        "algebra": 0,
        "geometry": 0,
        "combinatorics": 0,
        "number_theory": 0,
        "miscellaneous": 0,
    }
    for kw in _ALGEBRA_KEYWORDS:
        if kw in text:
            raw_scores["algebra"] += 1
    for kw in _GEOMETRY_KEYWORDS:
        if kw in text:
            raw_scores["geometry"] += 1
    for kw in _COMBINATORICS_KEYWORDS:
        if kw in text:
            raw_scores["combinatorics"] += 1
    for kw in _NUMBER_THEORY_KEYWORDS:
        if kw in text:
            raw_scores["number_theory"] += 1
    for kw in _MISC_KEYWORDS:
        if kw in text:
            raw_scores["miscellaneous"] += 1

    # Convert to soft labels
    scores_vec = [
        raw_scores["algebra"],
        raw_scores["geometry"],
        raw_scores["combinatorics"],
        raw_scores["number_theory"],
        raw_scores["miscellaneous"],
    ][:n_domains]

    total = sum(scores_vec)
    if total == 0:
        # Uniform
        return [1.0 / n_domains] * n_domains

    # Apply temperature to sharpen distribution
    import math
    t = 0.5
    exp_scores = [math.exp(s / t) for s in scores_vec]
    total_exp = sum(exp_scores)
    return [s / total_exp for s in exp_scores]


class DomainClassifier(nn.Module):
    """
    Neural domain classifier trained on labeled MATH/NuminaMath data.
    Uses a frozen backbone encoder + linear head.

    This is a simple TF-IDF or embedding-based classifier for fast domain
    prediction on unlabeled problems.

    Not used in the main PRISM forward pass — only for data preprocessing.
    """

    def __init__(self, n_domains: int = 5, vocab_size: int = 32000, embed_dim: int = 128):
        super().__init__()
        self.n_domains = n_domains
        # Simple bag-of-words style classifier
        self.embed = nn.EmbeddingBag(vocab_size, embed_dim, mode="mean", sparse=False)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, n_domains),
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: [batch, seq_len] token ids

        Returns:
            logits: [batch, n_domains]
        """
        h = self.embed(input_ids)
        return self.classifier(h)

    def predict(self, input_ids: torch.Tensor) -> list[str]:
        """Return domain name strings."""
        with torch.no_grad():
            logits = self.forward(input_ids)
            indices = logits.argmax(dim=-1)
        return [DOMAIN_NAMES[i] for i in indices.tolist()]
