"""
DataCollator for PRISM training batches.

Handles variable-length trace sequences with proper padding and label masking.
"""

import torch
from dataclasses import dataclass
from typing import Optional, List, Dict, Any


@dataclass
class PRISMDataCollator:
    """
    Collate PRISM trace examples into training batches.

    Pads sequences to the longest in the batch.
    Sets label=-100 for padding tokens and for problem tokens
    (we only compute loss on the trace tokens, not the problem statement).

    Args:
        tokenizer: HF tokenizer (from backbone processor).
        max_length: Maximum sequence length (tokens beyond this are truncated).
        pad_to_multiple_of: Pad to multiple of this value (efficient for tensor cores).
        include_domain_labels: Whether to include soft domain label tensors.
    """

    tokenizer: Any
    max_length: int = 2048
    pad_to_multiple_of: int = 8
    include_domain_labels: bool = True

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: List of dicts, each with:
                - input_ids: List[int] — tokenized problem + trace
                - labels: List[int] — -100 for problem tokens, trace token ids
                - attention_mask: List[int] — 1 for real, 0 for pad
                - domain_label: List[float] (optional) — soft domain weights
                - pixel_values: Tensor (optional) — for VL problems

        Returns:
            Batched tensors.
        """
        batch_size = len(features)

        # Gather sequences
        input_ids = [f["input_ids"] for f in features]
        labels = [f["labels"] for f in features]
        attention_masks = [f["attention_mask"] for f in features]

        # Truncate to max_length
        input_ids = [ids[: self.max_length] for ids in input_ids]
        labels = [lbs[: self.max_length] for lbs in labels]
        attention_masks = [am[: self.max_length] for am in attention_masks]

        # Find max length in this batch
        max_len = max(len(ids) for ids in input_ids)

        # Pad to multiple of pad_to_multiple_of
        if self.pad_to_multiple_of > 0:
            max_len = self.pad_to_multiple_of * ((max_len + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of)

        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

        # Pad everything
        padded_input_ids = []
        padded_labels = []
        padded_masks = []

        for ids, lbs, mask in zip(input_ids, labels, attention_masks):
            pad_len = max_len - len(ids)
            padded_input_ids.append(ids + [pad_id] * pad_len)
            padded_labels.append(lbs + [-100] * pad_len)
            padded_masks.append(mask + [0] * pad_len)

        batch = {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
            "attention_mask": torch.tensor(padded_masks, dtype=torch.long),
        }

        # Domain labels (soft)
        if self.include_domain_labels and "domain_label" in features[0]:
            domain_labels = [f["domain_label"] for f in features]
            batch["domain_labels"] = torch.tensor(domain_labels, dtype=torch.float)

        # Pixel values (for VL)
        if "pixel_values" in features[0] and features[0]["pixel_values"] is not None:
            pixel_values = [f["pixel_values"] for f in features]
            try:
                batch["pixel_values"] = torch.stack(pixel_values, dim=0)
            except Exception:
                pass  # Skip if shapes don't match

        return batch


def tokenize_trace_example(
    example: "TraceExample",
    tokenizer,
    phase: int,
    max_length: int = 2048,
    domain_label: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    Tokenize a TraceExample for a specific phase.

    The input is: [BOS] problem [SEP] phase_trace [EOS]
    Labels have -100 for problem tokens, trace token ids for trace tokens.

    Args:
        example: TraceExample instance.
        tokenizer: Backbone tokenizer.
        phase: 0=solve, 1=verify, 2=correct. Determines which trace to use.
        max_length: Maximum sequence length.
        domain_label: Soft domain weights (for router supervision).

    Returns:
        Dict with input_ids, labels, attention_mask, domain_label.
    """
    phase_traces = [example.solve_trace, example.verify_trace, example.correct_trace]
    trace_text = phase_traces[phase]

    # Tokenize problem
    problem_tokens = tokenizer.encode(
        example.problem,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length // 2,
    )

    # Tokenize trace
    sep = tokenizer.encode("\n\n", add_special_tokens=False)
    trace_tokens = tokenizer.encode(
        trace_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length - len(problem_tokens) - len(sep) - 1,
    )
    eos = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []

    input_ids = problem_tokens + sep + trace_tokens + eos

    # Labels: -100 for problem + sep tokens, actual ids for trace tokens
    labels = (
        [-100] * (len(problem_tokens) + len(sep))
        + trace_tokens
        + eos
    )

    attention_mask = [1] * len(input_ids)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "domain_label": domain_label,
    }
