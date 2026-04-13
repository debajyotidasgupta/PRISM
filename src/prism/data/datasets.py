"""
Dataset loaders for PRISM training data sources.

Datasets:
  - NuminaMath-CoT (860K problems with domain labels)
  - MATH (Hendrycks, 7.5K train) with 7-category labels → mapped to 5
  - OpenR1-Math-220K
  - MetaMathQA (395K, no domain labels)
"""

import os
import logging
from typing import Optional, Iterator
from datasets import load_dataset, Dataset

logger = logging.getLogger(__name__)

# Mapping from MATH's 7 categories to PRISM's 5 domains
MATH_LABEL_MAP = {
    "Algebra": "algebra",
    "Intermediate Algebra": "algebra",
    "Pre-calculus": "algebra",
    "Precalculus": "algebra",
    "Geometry": "geometry",
    "Number Theory": "number_theory",
    "Counting & Probability": "combinatorics",
    "Counting and Probability": "combinatorics",
    "Statistics": "miscellaneous",
    "Prealgebra": "algebra",
}

# Test-set datasets — NEVER use for training
HELD_OUT_DATASETS = {
    "OlympiadBench",
    "OlymMATH",
    "Omni-MATH",
    "MATH-500",
}


def _hf_cache_dir() -> Optional[str]:
    return os.environ.get("HF_HOME", None)


def load_numinamath(
    split: str = "train",
    domain: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> Dataset:
    """
    Load NuminaMath-CoT dataset.

    Args:
        split: "train" only (no validation split in original).
        domain: If provided, filter to this PRISM domain.
        max_samples: Cap at this many samples.

    Returns:
        HF Dataset with columns: problem, solution, source, domain
    """
    logger.info(f"Loading NuminaMath-CoT [{split}]")
    ds = load_dataset(
        "AI-MO/NuminaMath-CoT",
        split=split,
        cache_dir=_hf_cache_dir(),
        trust_remote_code=True,
    )

    # Add PRISM domain label by mapping from 'topic' or 'source' field
    def _add_domain(example):
        # NuminaMath uses 'type' or topic fields
        topic = example.get("topic", example.get("type", "")).lower()
        domain_label = _classify_numinamath_topic(topic)
        return {"prism_domain": domain_label}

    ds = ds.map(_add_domain, desc="Adding domain labels")

    if domain is not None:
        ds = ds.filter(lambda x: x["prism_domain"] == domain)

    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    logger.info(f"NuminaMath loaded: {len(ds)} examples")
    return ds


def _classify_numinamath_topic(topic: str) -> str:
    """Map NuminaMath topic string to PRISM domain."""
    topic = topic.lower()
    if any(k in topic for k in ["algebra", "polynomial", "function", "equation", "calculus"]):
        if any(k in topic for k in ["inequality", "ineq", "cauchy", "am-gm"]):
            return "miscellaneous"
        return "algebra"
    if any(k in topic for k in ["geometry", "triangle", "circle", "angle", "conic"]):
        return "geometry"
    if any(k in topic for k in ["combinatorics", "counting", "probability", "permutation", "graph"]):
        return "combinatorics"
    if any(k in topic for k in ["number theory", "divisib", "prime", "modular", "diophantine"]):
        return "number_theory"
    if any(k in topic for k in ["inequality", "generating function", "linear algebra", "complex"]):
        return "miscellaneous"
    return "miscellaneous"  # default


def load_math_dataset(
    split: str = "train",
    domain: Optional[str] = None,
    level_min: int = 1,
    level_max: int = 5,
    max_samples: Optional[int] = None,
) -> Dataset:
    """
    Load Hendrycks MATH dataset.

    Args:
        split: "train" or "test" (we use "train" only for training).
        domain: Filter to this PRISM domain.
        level_min/max: Filter by difficulty level (1=easiest, 5=hardest).

    Returns:
        Dataset with columns: problem, solution, type, level, prism_domain
    """
    logger.info(f"Loading MATH [{split}]")
    ds = load_dataset(
        "hendrycks/competition_math",
        split=split,
        cache_dir=_hf_cache_dir(),
        trust_remote_code=True,
    )

    def _process(example):
        math_type = example.get("type", "")
        prism_domain = MATH_LABEL_MAP.get(math_type, "miscellaneous")
        level_str = example.get("level", "Level 1")
        try:
            level = int(level_str.replace("Level ", "").strip())
        except (ValueError, AttributeError):
            level = 1
        return {
            "prism_domain": prism_domain,
            "difficulty_level": level,
            "problem": example.get("problem", ""),
            "solution": example.get("solution", ""),
        }

    ds = ds.map(_process, desc="Processing MATH")

    # Filter by difficulty level
    ds = ds.filter(lambda x: level_min <= x["difficulty_level"] <= level_max)

    if domain is not None:
        ds = ds.filter(lambda x: x["prism_domain"] == domain)

    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    logger.info(f"MATH loaded: {len(ds)} examples")
    return ds


def load_openr1(
    split: str = "train",
    domain: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> Dataset:
    """
    Load OpenR1-Math-220K dataset.
    """
    logger.info(f"Loading OpenR1-Math-220K [{split}]")
    try:
        ds = load_dataset(
            "open-r1/OpenR1-Math-220k",
            split=split,
            cache_dir=_hf_cache_dir(),
            trust_remote_code=True,
        )
    except Exception as e:
        logger.warning(f"OpenR1 load failed: {e}. Returning empty dataset.")
        return Dataset.from_dict({"problem": [], "solution": [], "prism_domain": []})

    def _process(example):
        # Try to infer domain from available metadata
        subject = example.get("subject", example.get("topic", "")).lower()
        prism_domain = _classify_numinamath_topic(subject)
        return {"prism_domain": prism_domain}

    ds = ds.map(_process, desc="Processing OpenR1")

    if domain is not None:
        ds = ds.filter(lambda x: x["prism_domain"] == domain)

    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    logger.info(f"OpenR1 loaded: {len(ds)} examples")
    return ds


def load_olympiadbench(split: str = "test") -> Dataset:
    """
    Load OlympiadBench. ALWAYS test-only. Never use for training.
    """
    logger.info(f"Loading OlympiadBench [TEST SET — evaluation only]")
    ds = load_dataset(
        "Hothan/OlympiadBench",
        split=split,
        cache_dir=_hf_cache_dir(),
        trust_remote_code=True,
    )
    return ds


def load_math500() -> Dataset:
    """
    Load MATH-500. Validation set only.
    """
    logger.info("Loading MATH-500 [validation set]")
    ds = load_dataset(
        "HuggingFaceH4/MATH-500",
        split="test",
        cache_dir=_hf_cache_dir(),
        trust_remote_code=True,
    )
    return ds


def get_stage0_training_data(
    n_per_domain: int = 2500,
    domains: list = None,
) -> dict:
    """
    Load Stage 0 training data: n_per_domain problems per domain.
    Source: MATH train (first) then NuminaMath (remainder).

    Returns:
        dict mapping domain_name → Dataset
    """
    if domains is None:
        domains = ["algebra", "geometry", "combinatorics", "number_theory", "miscellaneous"]

    result = {}
    for domain in domains:
        # Try MATH first (highest quality labels)
        math_ds = load_math_dataset(domain=domain, level_min=3, level_max=5)
        math_count = min(len(math_ds), n_per_domain)

        if math_count < n_per_domain:
            # Fill remainder from NuminaMath
            remaining = n_per_domain - math_count
            numa_ds = load_numinamath(domain=domain, max_samples=remaining)
            from datasets import concatenate_datasets
            combined = concatenate_datasets([
                math_ds.select(range(math_count)),
                numa_ds,
            ])
        else:
            combined = math_ds.select(range(math_count))

        result[domain] = combined
        logger.info(f"Stage 0 data — {domain}: {len(combined)} examples")

    return result
