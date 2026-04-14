"""
Expert block training: train one (domain, phase) expert block at a time.

Training rule (strictly enforced per program.md):
  - Backbone:                      FROZEN (never touches weights)
  - Domain router:                 FROZEN after router pre-training
  - All OTHER expert blocks:       FROZEN (but still run in forward pass)
  - CrossMix modules:              FROZEN during expert training
  - TARGET expert block (d, p):    TRAINABLE

Critical design note: ALL domain experts still run in every forward pass,
even when only one is trainable. The frozen experts provide their outputs
for CrossMix aggregation. Gradients only flow through the trainable block.
This is NOT equivalent to training in isolation — the trainable block must
produce outputs that mix well with the (frozen) outputs of other domains.

Training schedule for Stage 2 (parallelize across GPUs):
  Level 0 (Solve):
    GPU 0: Algebra (phase=0, domain=0) + Combinatorics (phase=0, domain=2)
    GPU 1: Geometry (phase=0, domain=1) + Miscellaneous (phase=0, domain=4)
    GPU 2: NumberTheory (phase=0, domain=3)
    GPU 3: Evaluation + router training
  [Then: CrossMix L0 training on GPU 3]
  Level 1 (Verify): same GPU assignment
  Level 2 (Correct): same GPU assignment

Usage:
  python -m prism.training.train_expert --domain algebra --phase 0 --gpu 0 \
    --traces results/traces/algebra_traces.jsonl \
    --config configs/model/prism_0.8b.yaml \
    --output-dir results/stage2/expert_blocks
"""

import os
import json
import math
import logging
import argparse
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)

DOMAIN_NAMES = ["algebra", "geometry", "combinatorics", "number_theory", "miscellaneous"]
PHASE_NAMES = ["solve", "verify", "correct"]


class TraceDataset(Dataset):
    """Dataset wrapping JSONL trace files for one (domain, phase)."""

    def __init__(self, trace_file: str, tokenizer, phase: int, max_length: int = 2048):
        self.phase = phase
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

        logger.info(f"Loaded {len(self.examples)} valid traces from {trace_file}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        from prism.data.collator import tokenize_trace_example
        ex = self.examples[idx]
        return tokenize_trace_example(ex, self.tokenizer, phase=self.phase)


class ExpertTrainer:
    """
    Trains a single expert block (domain d, phase p) with careful freezing.

    ALL domain experts participate in every forward pass.
    Only the target block's gradients are computed. All others run with no_grad.

    Args:
        prism_model: PRISMModel with backbone and all blocks initialized.
        domain_idx: Domain index (0-4).
        phase_idx: Phase index (0-2). This determines WHICH trace supervision is used.
        tokenizer: Backbone tokenizer.
        device: Training device.
        lr: Learning rate (5e-5 as per program.md).
        weight_decay: Adam weight decay.
        grad_clip: Gradient clip norm.
        warmup_steps: LR warmup steps.
        output_dir: Checkpoint save directory.
        grad_accum_steps: Gradient accumulation for effective batch size.
    """

    def __init__(
        self,
        prism_model,
        domain_idx: int,
        phase_idx: int,
        tokenizer,
        device: str = "cuda:0",
        lr: float = 5e-5,
        weight_decay: float = 0.01,
        grad_clip: float = 1.0,
        warmup_steps: int = 100,
        output_dir: str = "results/stage2/expert_blocks",
        grad_accum_steps: int = 4,
    ):
        self.model = prism_model
        self.domain_idx = domain_idx
        self.phase_idx = phase_idx
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.lr = lr
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.warmup_steps = warmup_steps
        self.grad_accum_steps = grad_accum_steps
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        domain_name = self.model.config.domains[domain_idx]
        phase_name = self.model.config.phases[phase_idx]
        self.block_name = f"phase{phase_idx}_{domain_name}"
        logger.info(f"Training: {self.block_name} ({phase_name} phase, {domain_name} domain)")

        # ─── FREEZE ALL EXCEPT TARGET BLOCK ──────────────────────────────────
        # Backbone: always frozen (never changes)
        if self.model.backbone is not None:
            for p in self.model.backbone.parameters():
                p.requires_grad_(False)

        # Router: frozen during expert training
        for p in self.model.router.parameters():
            p.requires_grad_(False)

        # Expert blocks: freeze all, then unfreeze only the target
        for pi, phase_blocks in enumerate(self.model.expert_blocks):
            for di, block in enumerate(phase_blocks):
                is_target = (pi == phase_idx and di == domain_idx)
                for p in block.parameters():
                    p.requires_grad_(is_target)

        # CrossMix: frozen during expert training (trained separately)
        for cm in self.model.cross_mix:
            for p in cm.parameters():
                p.requires_grad_(False)

        # Verify freezing
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_prism = sum(p.numel() for g in self.model.expert_blocks for b in g for p in b.parameters())
        logger.info(f"Trainable params: {trainable:,} / {total_prism:,} PRISM params")

        target_block = self.model.expert_blocks[phase_idx][domain_idx]
        self.optimizer = optim.AdamW(
            target_block.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        self._global_step = 0

    def _lr_scale(self, step: int, total_steps: int) -> float:
        """Cosine schedule with linear warmup."""
        if step < self.warmup_steps:
            return (step + 1) / max(self.warmup_steps, 1)
        progress = (step - self.warmup_steps) / max(total_steps - self.warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    def _forward_with_target_expert(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        Forward pass with careful gradient flow.

        ALL experts run. Only the target block computes gradients.
        This ensures the target block learns to produce outputs that
        mix well with the frozen outputs of other domain experts.

        Levels 0..phase_idx-1 use trained+frozen blocks.
        Level phase_idx: target block is trainable, others frozen.
        Levels phase_idx+1..: not run (we train sequentially per level).

        Returns:
            loss: Scalar LM loss.
            info: Dict with loss components for logging.
        """
        n_domains = self.model.config.n_domains
        device = input_ids.device

        # ─ Backbone to insertion point (frozen) ─
        with torch.no_grad():
            h_K = self.model._backbone_forward_to_K(
                input_ids, attention_mask, pixel_values=None, image_grid_thw=None
            )
            # Router: get phase weights (frozen during expert training)
            phase_weights, _ = self.model.router(h_K, attention_mask, phase_idx=None)

        # ─ Run all phases 0..phase_idx (matching updated forward logic) ─
        domain_states = [h_K.clone() for _ in range(n_domains)]
        phase_history = []    # stores domain state lists after each phase's CrossMix
        phase_aggregates = [] # routing-weighted aggregate per phase

        for pi in range(self.phase_idx + 1):
            phase_blocks = self.model.expert_blocks[pi]
            w_phase = phase_weights[:, pi, :].detach()  # [B, n_domains] — routing constants

            # Cross-phase: attend to prior phase states (frozen, but participates in autograd
            # if target expert's output is in phase_history from a prior iteration)
            if pi > 0 and not self.model._disable_crossphase:
                with torch.no_grad():
                    domain_states = self.model.cross_phase(domain_states, phase_history, attention_mask)

            phase_outputs = []
            for di in range(n_domains):
                is_target = (pi == self.phase_idx and di == self.domain_idx)
                if is_target:
                    # Trainable: compute with full grad
                    out = phase_blocks[di](domain_states[di], attention_mask)
                else:
                    with torch.no_grad():
                        out = phase_blocks[di](domain_states[di], attention_mask)
                phase_outputs.append(out)

            # CrossMix: frozen params, but autograd must track this for grad to reach
            # the target expert (loss → logits → h_K_prime → CrossMix → expert[target])
            mixed_outputs = self.model.cross_mix[pi](phase_outputs, attention_mask)

            domain_states = mixed_outputs
            phase_history.append([s.detach() for s in mixed_outputs])  # detached history

            # Compute routing-weighted aggregate for this phase (matches updated forward)
            w = w_phase.unsqueeze(-1).unsqueeze(-1)       # [B, N, 1, 1]
            stacked = torch.stack(mixed_outputs, dim=1)    # [B, N, T, D]
            phase_agg = (w * stacked).sum(dim=1)           # [B, T, D]
            phase_aggregates.append(phase_agg)

        # ─ Aggregate across phases (matches updated prism_model.py forward) ─
        agg_mode = getattr(self.model.config, "phase_aggregate_mode", "mean")
        if agg_mode == "last" or len(phase_aggregates) == 1:
            h_K_prime = phase_aggregates[-1]
        else:
            h_K_prime = torch.stack(phase_aggregates, dim=0).mean(dim=0)  # [B, T, D]

        # ─ Backbone from insertion point (frozen but participates in autograd) ─
        # Do NOT use torch.no_grad() here: gradient must flow from loss → logits →
        # h_K_prime → expert block parameters. Backbone params (frozen) get zero grad.
        logits = self.model._backbone_forward_from_K(h_K_prime, attention_mask)

        # ─ LM loss on phase-specific traces ─
        from torch.nn import CrossEntropyLoss
        loss_fn = CrossEntropyLoss(ignore_index=-100)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        lm_loss = loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

        return lm_loss, {"lm_loss": lm_loss.item()}

    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
        total_steps: int,
    ) -> dict:
        target_block = self.model.expert_blocks[self.phase_idx][self.domain_idx]
        target_block.train()
        # Ensure all other blocks are in eval mode (affects dropout)
        for pi, phase_blocks in enumerate(self.model.expert_blocks):
            for di, block in enumerate(phase_blocks):
                if not (pi == self.phase_idx and di == self.domain_idx):
                    block.eval()
        for cm in self.model.cross_mix:
            cm.eval()

        total_loss = 0.0
        n_batches = 0
        self.optimizer.zero_grad()

        for step, batch in enumerate(tqdm(dataloader, desc=f"{self.block_name} ep{epoch}")):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            loss, info = self._forward_with_target_expert(input_ids, attention_mask, labels)
            # Scale for gradient accumulation
            (loss / self.grad_accum_steps).backward()

            total_loss += loss.item()
            n_batches += 1

            if (step + 1) % self.grad_accum_steps == 0:
                nn.utils.clip_grad_norm_(target_block.parameters(), self.grad_clip)
                self.optimizer.step()
                self.optimizer.zero_grad()

                self._global_step += 1
                scale = self._lr_scale(self._global_step, total_steps)
                for pg in self.optimizer.param_groups:
                    pg["lr"] = self.lr * scale

        # Final step if leftover gradient
        nn.utils.clip_grad_norm_(target_block.parameters(), self.grad_clip)
        self.optimizer.step()
        self.optimizer.zero_grad()

        return {
            "epoch": epoch,
            "block": self.block_name,
            "loss": total_loss / max(n_batches, 1),
        }

    def train(
        self,
        trace_file: str,
        epochs: int = 3,
        batch_size: int = 16,
    ) -> list[dict]:
        """
        Train this expert block on phase-specific traces.

        Args:
            trace_file: JSONL file with TraceExample records.
            epochs: Number of training epochs.
            batch_size: Per-GPU batch size.

        Returns:
            Per-epoch metrics.
        """
        from prism.data.collator import PRISMDataCollator

        dataset = TraceDataset(trace_file, self.tokenizer, phase=self.phase_idx)
        if len(dataset) == 0:
            logger.warning(f"Empty dataset for {self.block_name}. Skipping.")
            return []

        collator = PRISMDataCollator(self.tokenizer, include_domain_labels=False)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            collate_fn=collator, num_workers=4, pin_memory=True,
        )

        total_steps = epochs * (len(dataloader) // self.grad_accum_steps)
        history = []

        for epoch in range(1, epochs + 1):
            metrics = self.train_epoch(dataloader, epoch, total_steps)
            history.append(metrics)
            logger.info(f"{self.block_name} ep{epoch}: loss={metrics['loss']:.4f}")

            ckpt = self.output_dir / f"{self.block_name}_epoch{epoch}.pt"
            torch.save(
                self.model.expert_blocks[self.phase_idx][self.domain_idx].state_dict(),
                str(ckpt),
            )

        # Save final
        final_path = self.output_dir / f"{self.block_name}_final.pt"
        torch.save(
            self.model.expert_blocks[self.phase_idx][self.domain_idx].state_dict(),
            str(final_path),
        )
        with open(self.output_dir / f"{self.block_name}_history.json", "w") as f:
            json.dump(history, f, indent=2)

        logger.info(f"Saved {self.block_name} → {final_path}")
        return history


def train_expert_block(
    prism_config,
    domain: str,
    phase: int,
    trace_file: str,
    gpu_id: int = 0,
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 5e-5,
    output_dir: str = "results/stage2/expert_blocks",
) -> dict:
    """Convenience function: build model and train one expert block."""
    from prism.model.prism_model import PRISMModel
    from transformers import AutoTokenizer

    device = f"cuda:{gpu_id}"
    model = PRISMModel(prism_config)._load_backbone(device=device)
    model = model.to(device)

    domain_idx = prism_config.domain_to_idx[domain]
    phase_idx = prism_config.phase_to_idx[PHASE_NAMES[phase]]

    tokenizer = AutoTokenizer.from_pretrained(prism_config.backbone_name, trust_remote_code=True)

    trainer = ExpertTrainer(
        prism_model=model,
        domain_idx=domain_idx,
        phase_idx=phase_idx,
        tokenizer=tokenizer,
        device=device,
        lr=lr,
        output_dir=output_dir,
    )
    history = trainer.train(trace_file=trace_file, epochs=epochs, batch_size=batch_size)
    return {"history": history, "output_dir": output_dir}


def main():
    """CLI: prism-train"""
    parser = argparse.ArgumentParser(description="Train one PRISM expert block")
    parser.add_argument("--domain", required=True, choices=DOMAIN_NAMES)
    parser.add_argument("--phase", type=int, required=True, choices=[0, 1, 2])
    parser.add_argument("--traces", required=True, help="JSONL trace file")
    parser.add_argument("--config", required=True, help="PRISMConfig YAML path")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--output-dir", default="results/stage2/expert_blocks")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    import yaml
    from prism.model.config import PRISMConfig

    with open(args.config) as f:
        cfg_dict = yaml.safe_load(f)
    config = PRISMConfig(**cfg_dict)

    result = train_expert_block(
        prism_config=config,
        domain=args.domain,
        phase=args.phase,
        trace_file=args.traces,
        gpu_id=args.gpu,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_dir=args.output_dir,
    )
    print(f"Done: {result['output_dir']}")


if __name__ == "__main__":
    main()
