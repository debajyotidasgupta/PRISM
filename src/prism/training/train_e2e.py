"""
Stage 3: Optional end-to-end fine-tuning.

Unfreezes: all PRISM blocks (router + expert blocks + cross-mix)
Keeps frozen: backbone
LR: very low (1e-6) with cosine schedule

Only run if Stage 2 shows a significant gap between PRISM and 7B baseline
that could be closed by joint optimization.
"""

import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


def train_e2e(
    prism_model,
    tokenizer,
    traces_dir: str,
    domains: list[str],
    gpu_id: int = 0,
    epochs: int = 2,
    batch_size: int = 8,
    lr: float = 1e-6,
    output_dir: str = "results/stage3",
) -> list[dict]:
    """
    End-to-end joint fine-tuning of all PRISM blocks.

    Args:
        prism_model: PRISMModel with pre-trained blocks loaded.
        tokenizer: Backbone tokenizer.
        traces_dir: Directory containing domain JSONL trace files.
        domains: List of domain names to include.
        ...

    Returns:
        Per-epoch metrics.
    """
    device = torch.device(f"cuda:{gpu_id}")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Unfreeze all PRISM params, keep backbone frozen
    for p in prism_model.get_prism_params():
        p.requires_grad_(True)
    for p in prism_model.backbone.parameters():
        p.requires_grad_(False)

    trainable_params = sum(p.numel() for p in prism_model.parameters() if p.requires_grad)
    logger.info(f"E2E fine-tuning: {trainable_params:,} trainable params")

    optimizer = optim.AdamW(
        [p for p in prism_model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.01,
    )

    from prism.training.train_expert import TraceDataset
    from prism.data.collator import PRISMDataCollator

    all_datasets = []
    for domain in domains:
        for phase_idx in range(prism_model.config.n_phases):
            trace_file = Path(traces_dir) / f"{domain}_traces.jsonl"
            if trace_file.exists():
                ds = TraceDataset(str(trace_file), tokenizer, phase=phase_idx)
                if len(ds) > 0:
                    all_datasets.append(ds)

    if not all_datasets:
        logger.warning("No data found for E2E training.")
        return []

    combined_ds = ConcatDataset(all_datasets)
    collator = PRISMDataCollator(tokenizer, include_domain_labels=True)
    dataloader = DataLoader(
        combined_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collator, num_workers=4, pin_memory=True,
    )

    history = []
    for epoch in range(1, epochs + 1):
        prism_model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in tqdm(dataloader, desc=f"E2E epoch {epoch}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            domain_labels = batch.get("domain_labels")
            if domain_labels is not None:
                domain_labels = domain_labels.to(device)

            out = prism_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                domain_labels=domain_labels,
            )
            loss = out["loss"]

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                [p for p in prism_model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        metrics = {"epoch": epoch, "loss": total_loss / max(n_batches, 1)}
        history.append(metrics)
        logger.info(f"E2E epoch {epoch}: loss={metrics['loss']:.4f}")

        ckpt = output_path / f"prism_e2e_epoch{epoch}.pt"
        torch.save(
            {k: v for k, v in prism_model.state_dict().items() if "backbone" not in k},
            str(ckpt),
        )

    with open(output_path / "e2e_history.json", "w") as f:
        json.dump(history, f, indent=2)

    return history
