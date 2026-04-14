"""
Joint end-to-end fine-tuning of all PRISM modules simultaneously.

After per-block pretraining (which suffers from sequential mismatch because
each block is trained in isolation), this stage trains ALL modules together:
  - All 15 expert blocks (5 domains × 3 phases)
  - All 3 CrossMix modules
  - CrossPhase module
  - Domain router

ALL receive gradients from the final LM loss flowing back through the full
solve→verify→correct chain. This fixes:
  1. Phase mismatch: Phase 0 now gets gradient signal from Phase 2 loss
  2. Router utility: Router learns which experts help generation, not just domain
  3. CrossPhase: Learns to leverage prior phase states effectively

Training data: same phase-specific traces used for per-block training.
The trace for a problem is concatenated: [problem + solve_trace + verify_trace + correct_trace]
and the model learns to predict all of it autoregressively.

Usage:
  python -m prism.training.joint_finetune \
    --traces results/traces/pilot/ \
    --expert-blocks results/pilot/expert_blocks/ \
    --router results/pilot/router/router_final.pt \
    --config configs/model/prism_0.8b.yaml \
    --output-dir results/pilot/joint_ft/ \
    --gpu 0 --epochs 3 --lr 2e-5
"""

import os
import json
import math
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset

logger = logging.getLogger(__name__)

DOMAIN_NAMES = ["algebra", "geometry", "combinatorics", "number_theory", "miscellaneous"]
PHASE_NAMES = ["solve", "verify", "correct"]


class FullTraceDataset(Dataset):
    """
    Dataset for joint fine-tuning: full traces (all 3 phases concatenated).

    Each example: [problem → solve_trace → verify_trace → correct_trace]
    Supervision: all tokens (the model learns the full reasoning chain).
    """

    def __init__(self, trace_file: str, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        path = Path(trace_file)
        if not path.exists():
            logger.warning(f"Trace file not found: {trace_file}")
            return

        from prism.data.trace_format import TraceExample
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ex = TraceExample.from_jsonl(line)
                    if ex.is_valid():
                        self.examples.append(ex)
                except Exception as e:
                    logger.debug(f"Skip invalid trace: {e}")

        logger.info(f"Loaded {len(self.examples)} traces from {trace_file}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        from prism.data.collator import tokenize_full_trace
        ex = self.examples[idx]
        return tokenize_full_trace(ex, self.tokenizer, max_length=self.max_length)


class JointFinetuneTrainer:
    """
    End-to-end fine-tuning of ALL PRISM modules on full traces.

    All modules trained simultaneously with single LM loss.
    Router gets gradient from task performance, not domain classification.

    Args:
        prism_model: PRISMModel with loaded backbone + pretrained blocks.
        tokenizer: Backbone tokenizer.
        device: Training device.
        lr: Learning rate (lower than per-block training: 2e-5).
        router_lr: Separate (higher) LR for router (learns from scratch on task signal).
        warmup_steps: LR warmup steps.
        output_dir: Checkpoint save directory.
        grad_accum_steps: Gradient accumulation steps.
        grad_clip: Gradient clip norm.
        entropy_weight: Router entropy regularization weight (prevent collapse).
    """

    def __init__(
        self,
        prism_model,
        tokenizer,
        device: str = "cuda:0",
        lr: float = 2e-5,
        router_lr: float = 5e-5,
        warmup_steps: int = 20,
        output_dir: str = "results/pilot/joint_ft",
        grad_accum_steps: int = 8,
        grad_clip: float = 1.0,
        entropy_weight: float = 0.02,
    ):
        self.model = prism_model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.lr = lr
        self.router_lr = router_lr
        self.warmup_steps = warmup_steps
        self.grad_accum_steps = grad_accum_steps
        self.grad_clip = grad_clip
        self.entropy_weight = entropy_weight
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Backbone always frozen
        if self.model.backbone is not None:
            for p in self.model.backbone.parameters():
                p.requires_grad_(False)

        # Unfreeze ALL PRISM modules
        self.model.unfreeze_all_prism()

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Joint fine-tune: {trainable:,} trainable params")

        # Two parameter groups: router gets higher LR (learns from scratch on task signal)
        router_params = list(self.model.router.parameters())
        expert_params = (
            [p for pb in self.model.expert_blocks for b in pb for p in b.parameters()]
            + [p for cm in self.model.cross_mix for p in cm.parameters()]
            + list(self.model.cross_phase.parameters())
        )

        self.optimizer = optim.AdamW([
            {"params": expert_params, "lr": lr},
            {"params": router_params, "lr": router_lr},
        ], weight_decay=0.01)
        self._global_step = 0

    def _lr_scale(self, step: int, total_steps: int) -> float:
        """Cosine schedule with linear warmup."""
        if step < self.warmup_steps:
            return (step + 1) / max(self.warmup_steps, 1)
        progress = (step - self.warmup_steps) / max(total_steps - self.warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    def train_epoch(self, dataloader: DataLoader, epoch: int, total_steps: int) -> dict:
        self.model.train()
        # Backbone always in eval (frozen)
        if self.model.backbone is not None:
            self.model.backbone.eval()

        total_loss = 0.0
        total_lm = 0.0
        total_ent = 0.0
        n_batches = 0
        self.optimizer.zero_grad()

        from torch.nn import CrossEntropyLoss
        loss_fn = CrossEntropyLoss(ignore_index=-100)

        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            # Full PRISM forward (all phases, all routing applied)
            out = self.model.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_router_weights=True,
            )
            logits = out["logits"]

            # LM loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            lm_loss = loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

            # Router entropy regularization (maximize entropy → prevent collapse to 1 expert)
            phase_weights = out["router_weights"]  # [B, n_phases, n_domains]
            entropy_loss = self.model.router.entropy_loss(phase_weights)  # negative entropy
            # entropy_loss is already negated (minimizing it = maximizing entropy)

            loss = lm_loss + self.entropy_weight * entropy_loss.abs()
            (loss / self.grad_accum_steps).backward()

            total_loss += loss.item()
            total_lm += lm_loss.item()
            total_ent += entropy_loss.abs().item()
            n_batches += 1

            if (step + 1) % self.grad_accum_steps == 0:
                nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.grad_clip,
                )
                self.optimizer.step()
                self.optimizer.zero_grad()

                self._global_step += 1
                scale = self._lr_scale(self._global_step, total_steps)
                for pg in self.optimizer.param_groups:
                    pg["lr"] = pg["initial_lr"] * scale if "initial_lr" in pg else pg["lr"]

        # Final step if leftover gradient
        nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad],
            self.grad_clip,
        )
        self.optimizer.step()
        self.optimizer.zero_grad()

        return {
            "epoch": epoch,
            "loss": total_loss / max(n_batches, 1),
            "lm_loss": total_lm / max(n_batches, 1),
            "entropy_loss": total_ent / max(n_batches, 1),
        }

    def save_checkpoint(self, epoch: int):
        """Save all PRISM module checkpoints."""
        epoch_dir = self.output_dir / f"epoch{epoch}"
        epoch_dir.mkdir(exist_ok=True)

        torch.save(self.model.router.state_dict(), epoch_dir / "router.pt")
        torch.save(self.model.cross_phase.state_dict(), epoch_dir / "cross_phase.pt")
        for pi, phase_blocks in enumerate(self.model.expert_blocks):
            for di, block in enumerate(phase_blocks):
                domain = self.model.config.domains[di]
                fname = f"phase{pi}_{domain}.pt"
                torch.save(block.state_dict(), epoch_dir / fname)
        for i, cm in enumerate(self.model.cross_mix):
            torch.save(cm.state_dict(), epoch_dir / f"crossmix_{i}.pt")

        logger.info(f"Saved checkpoint → {epoch_dir}")

    def save_final(self):
        """Save final checkpoints (flat, matches per-block naming convention)."""
        torch.save(self.model.router.state_dict(), self.output_dir / "router_final.pt")
        torch.save(self.model.cross_phase.state_dict(), self.output_dir / "cross_phase_final.pt")
        for pi, phase_blocks in enumerate(self.model.expert_blocks):
            for di, block in enumerate(phase_blocks):
                domain = self.model.config.domains[di]
                fname = f"phase{pi}_{domain}_final.pt"
                torch.save(block.state_dict(), self.output_dir / fname)
        for i, cm in enumerate(self.model.cross_mix):
            torch.save(cm.state_dict(), self.output_dir / f"crossmix_{i}_final.pt")

    def train(
        self,
        trace_dirs: list[str],
        domains: list[str] = None,
        epochs: int = 3,
        batch_size: int = 2,
        max_length: int = 2048,
    ) -> list[dict]:
        """
        Joint fine-tuning on full traces from all domains.

        Args:
            trace_dirs: List of directories containing {domain}_traces.jsonl files.
            domains: Domains to train on (default: all).
            epochs: Fine-tuning epochs.
            batch_size: Per-GPU batch size (small due to full 3-phase forward).
            max_length: Max token length per example.

        Returns:
            Per-epoch metrics.
        """
        from prism.data.collator import PRISMDataCollator

        if domains is None:
            domains = self.model.config.domains

        # Build dataset from all domain trace files
        all_datasets = []
        for trace_dir in trace_dirs:
            for domain in domains:
                trace_file = Path(trace_dir) / f"{domain}_traces.jsonl"
                ds = FullTraceDataset(str(trace_file), self.tokenizer, max_length=max_length)
                if len(ds) > 0:
                    all_datasets.append(ds)

        if not all_datasets:
            logger.error("No trace data found! Check trace_dirs.")
            return []

        combined = ConcatDataset(all_datasets)
        logger.info(f"Joint fine-tune dataset: {len(combined)} total examples")

        collator = PRISMDataCollator(self.tokenizer, include_domain_labels=False)
        dataloader = DataLoader(
            combined,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collator,
            num_workers=2,
            pin_memory=True,
        )

        total_steps = epochs * (len(dataloader) // self.grad_accum_steps)
        # Store initial LRs for scale scheduling
        for pg in self.optimizer.param_groups:
            pg["initial_lr"] = pg["lr"]

        history = []
        for epoch in range(1, epochs + 1):
            metrics = self.train_epoch(dataloader, epoch, total_steps)
            history.append(metrics)
            logger.info(
                f"Joint FT epoch {epoch}: "
                f"loss={metrics['loss']:.4f} "
                f"lm={metrics['lm_loss']:.4f} "
                f"entropy={metrics['entropy_loss']:.4f}"
            )
            self.save_checkpoint(epoch)

        self.save_final()
        with open(self.output_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)

        logger.info(f"Joint fine-tuning complete → {self.output_dir}")
        return history
