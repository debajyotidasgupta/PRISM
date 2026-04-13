"""
Cross-domain mixing training.

Trained AFTER all expert blocks at a given level are frozen.
Teaches the CrossMixModule to optimally blend domain expert outputs.

Freezing during CrossMix training:
  - Backbone: FROZEN
  - Router: FROZEN
  - ALL expert blocks (all phases, all domains): FROZEN
  - CrossMix levels 0..level_idx-1: FROZEN (already trained)
  - CrossMix level_idx: TRAINABLE
  - CrossMix levels level_idx+1..: FROZEN (not yet trained)

The CrossMix learns to produce mixed domain states that improve the
final aggregate output. It is a true information exchange module —
the verify phase's geometry expert can query the combinatorics expert's
solve outputs to apply modular counting tools to geometric configurations.
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


class CrossMixTrainer:
    """
    Trains the CrossMixModule at a given level.

    Args:
        prism_model: PRISMModel with all expert blocks already trained and frozen.
        level_idx: Which CrossMix level to train (0, 1, or 2).
        tokenizer: Backbone tokenizer.
        device: Training device.
        lr: Learning rate (2e-5 as per program.md).
        warmup_steps: LR warmup steps.
        output_dir: Checkpoint save directory.
    """

    def __init__(
        self,
        prism_model,
        level_idx: int,
        tokenizer,
        device: str = "cuda:0",
        lr: float = 2e-5,
        warmup_steps: int = 100,
        output_dir: str = "results/stage2/cross_mix",
    ):
        self.model = prism_model
        self.level_idx = level_idx
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._global_step = 0

        # ─── FREEZING (critical: must be exact) ──────────────────────────────
        # Backbone: frozen
        if self.model.backbone is not None:
            for p in self.model.backbone.parameters():
                p.requires_grad_(False)

        # Router: frozen
        for p in self.model.router.parameters():
            p.requires_grad_(False)

        # ALL expert blocks: frozen
        for phase_blocks in self.model.expert_blocks:
            for block in phase_blocks:
                for p in block.parameters():
                    p.requires_grad_(False)

        # CrossMix: freeze all, then unfreeze only target level
        for i, cm in enumerate(self.model.cross_mix):
            for p in cm.parameters():
                p.requires_grad_(i == level_idx)

        target_cm = self.model.cross_mix[level_idx]
        trainable = sum(p.numel() for p in target_cm.parameters())
        logger.info(f"CrossMix L{level_idx}: {trainable:,} trainable params")

        self.optimizer = optim.AdamW(
            target_cm.parameters(), lr=lr, weight_decay=0.01
        )

    def _forward_with_target_crossmix(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass where only CrossMix at level_idx computes gradients.

        Runs all expert blocks (frozen) up to and including level_idx,
        applies the trainable CrossMix, then aggregates and continues backbone.
        """
        n_domains = self.model.config.n_domains

        # Backbone (frozen)
        with torch.no_grad():
            h_K = self.model._backbone_forward_to_K(
                input_ids, attention_mask, pixel_values=None, image_grid_thw=None
            )
            phase_weights, _ = self.model.router(h_K, attention_mask, phase_idx=None)

        # Run all phases up to and including level_idx
        domain_states = [h_K.clone() for _ in range(n_domains)]

        for pi in range(self.level_idx + 1):
            phase_blocks = self.model.expert_blocks[pi]

            # All expert blocks at this level are frozen
            with torch.no_grad():
                phase_outputs = [
                    phase_blocks[d](domain_states[d], attention_mask)
                    for d in range(n_domains)
                ]

            if pi < self.level_idx:
                # Previous CrossMix levels: frozen
                with torch.no_grad():
                    domain_states = self.model.cross_mix[pi](phase_outputs, attention_mask)
            else:
                # Target CrossMix level: TRAINABLE (gradients flow here)
                domain_states = self.model.cross_mix[self.level_idx](phase_outputs, attention_mask)

        # Aggregate using target phase's routing weights
        w = phase_weights[:, self.level_idx, :]           # [B, n_domains]
        w_expanded = w.unsqueeze(-1).unsqueeze(-1)        # [B, N, 1, 1]
        stacked = torch.stack(domain_states, dim=1)        # [B, N, T, D]
        h_K_prime = (w_expanded * stacked).sum(dim=1)      # [B, T, D]

        # Backbone continuation (frozen)
        with torch.no_grad():
            logits = self.model._backbone_forward_from_K(h_K_prime, attention_mask)

        # LM loss
        from torch.nn import CrossEntropyLoss
        loss_fn = CrossEntropyLoss(ignore_index=-100)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        return loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

    def train(
        self,
        traces_dir: str,
        domains: list[str],
        epochs: int = 2,
        batch_size: int = 8,
        grad_accum_steps: int = 4,
    ) -> list[dict]:
        """
        Train the CrossMix module using traces from ALL domains at this level.

        Using all domains together is essential: CrossMix must learn to blend
        outputs from different domain experts, so it needs to see diverse
        domain combinations.
        """
        from prism.training.train_expert import TraceDataset
        from prism.data.collator import PRISMDataCollator

        # Combine traces from ALL domains (critical for learning cross-domain mixing)
        all_datasets = []
        for domain in domains:
            trace_file = Path(traces_dir) / f"{domain}_traces.jsonl"
            if trace_file.exists():
                ds = TraceDataset(
                    str(trace_file), self.tokenizer,
                    phase=self.level_idx  # Use target level's traces
                )
                if len(ds) > 0:
                    logger.info(f"CrossMix L{self.level_idx}: adding {len(ds)} {domain} traces")
                    all_datasets.append(ds)

        if not all_datasets:
            logger.warning(f"No data for CrossMix L{self.level_idx}")
            return []

        combined_ds = ConcatDataset(all_datasets)
        collator = PRISMDataCollator(self.tokenizer, include_domain_labels=False)
        dataloader = DataLoader(
            combined_ds, batch_size=batch_size, shuffle=True,
            collate_fn=collator, num_workers=4, pin_memory=True,
        )

        history = []
        for epoch in range(1, epochs + 1):
            self.model.cross_mix[self.level_idx].train()
            # All other modules in eval mode
            for pi, phase_blocks in enumerate(self.model.expert_blocks):
                for block in phase_blocks:
                    block.eval()
            for i, cm in enumerate(self.model.cross_mix):
                if i != self.level_idx:
                    cm.eval()

            total_loss = 0.0
            n_batches = 0
            self.optimizer.zero_grad()

            for step, batch in enumerate(tqdm(dataloader, desc=f"CrossMix L{self.level_idx} ep{epoch}")):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                loss = self._forward_with_target_crossmix(input_ids, attention_mask, labels)
                (loss / grad_accum_steps).backward()

                total_loss += loss.item()
                n_batches += 1

                if (step + 1) % grad_accum_steps == 0:
                    nn.utils.clip_grad_norm_(
                        self.model.cross_mix[self.level_idx].parameters(), 1.0
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self._global_step += 1

            # Final step
            self.optimizer.step()
            self.optimizer.zero_grad()

            metrics = {
                "epoch": epoch,
                "level": self.level_idx,
                "loss": total_loss / max(n_batches, 1),
            }
            history.append(metrics)
            logger.info(f"CrossMix L{self.level_idx} ep{epoch}: loss={metrics['loss']:.4f}")

            ckpt = self.output_dir / f"crossmix_level{self.level_idx}_epoch{epoch}.pt"
            torch.save(self.model.cross_mix[self.level_idx].state_dict(), str(ckpt))

        final = self.output_dir / f"crossmix_level{self.level_idx}_final.pt"
        torch.save(self.model.cross_mix[self.level_idx].state_dict(), str(final))

        with open(self.output_dir / f"crossmix_level{self.level_idx}_history.json", "w") as f:
            json.dump(history, f, indent=2)

        logger.info(f"CrossMix L{self.level_idx} saved → {final}")
        return history
