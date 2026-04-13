"""
Router training: train the DomainRouter on domain-labeled problems.

The router is a lightweight MLP that takes mean-pooled backbone hidden states
and produces 5-dimensional soft domain weights.

Training procedure:
  - Backbone is frozen
  - Expert blocks and cross-mix are frozen (not yet trained)
  - Only router parameters are updated
  - Loss: cross-entropy (hard labels) + KL divergence (soft labels for mixed-domain)
  - Epochs: 3-5, LR=1e-4
"""

import os
import json
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


class DomainLabelDataset(Dataset):
    """
    Dataset for router training.

    Each example: tokenized problem → soft domain label vector.
    """

    def __init__(self, examples: list[dict], tokenizer, max_length: int = 256):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        problem = ex["problem"]
        domain_label = ex["domain_label"]  # list of 5 floats

        tokens = self.tokenizer(
            problem,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "domain_label": torch.tensor(domain_label, dtype=torch.float),
        }


class RouterTrainer:
    """
    Trainer for the DomainRouter component.

    Args:
        prism_model: PRISMModel with backbone and router.
        tokenizer: Backbone tokenizer.
        device: Training device.
        lr: Learning rate.
        warmup_steps: Linear LR warmup steps.
        output_dir: Where to save router checkpoint.
    """

    def __init__(
        self,
        prism_model,
        tokenizer,
        device: str = "cuda:0",
        lr: float = 1e-4,
        warmup_steps: int = 100,
        output_dir: str = "results/stage1/router",
    ):
        self.model = prism_model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Freeze everything except router
        self.model.freeze_all_except_router()

        # Optimizer for router only
        router_params = list(self.model.router.parameters())
        self.optimizer = optim.AdamW(router_params, lr=lr, weight_decay=0.01)

    def _get_hidden_states(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Run backbone to insertion point, return h_K."""
        with torch.no_grad():
            h_K = self.model._backbone_forward_to_K(
                input_ids, attention_mask, pixel_values=None, image_grid_thw=None
            )
        return h_K

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> dict:
        self.model.router.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for step, batch in enumerate(tqdm(dataloader, desc=f"Router epoch {epoch}")):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            domain_labels = batch["domain_label"].to(self.device)  # [B, 5]

            # Get backbone hidden states (frozen)
            h_K = self._get_hidden_states(input_ids, attention_mask)

            # Router forward
            weights, logits = self.model.router(h_K, attention_mask)

            # Loss: KL divergence from soft labels
            import torch.nn.functional as F
            log_probs = F.log_softmax(logits, dim=-1)
            loss = F.kl_div(log_probs, domain_labels, reduction="batchmean")

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.router.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()

            # Accuracy: argmax of prediction vs argmax of label
            pred_domain = logits.argmax(dim=-1)
            true_domain = domain_labels.argmax(dim=-1)
            correct += (pred_domain == true_domain).sum().item()
            total += input_ids.size(0)

            # LR warmup
            global_step = (epoch - 1) * len(dataloader) + step
            if global_step < self.warmup_steps:
                lr_scale = (global_step + 1) / self.warmup_steps
                for pg in self.optimizer.param_groups:
                    pg["lr"] = self.lr * lr_scale

        return {
            "epoch": epoch,
            "loss": total_loss / len(dataloader),
            "accuracy": correct / max(total, 1),
        }

    def train(self, train_data: list[dict], epochs: int = 5, batch_size: int = 32) -> list[dict]:
        """
        Train the router.

        Args:
            train_data: List of dicts with 'problem' and 'domain_label' keys.
            epochs: Number of training epochs.
            batch_size: Batch size.

        Returns:
            List of per-epoch metrics dicts.
        """
        dataset = DomainLabelDataset(train_data, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

        history = []
        for epoch in range(1, epochs + 1):
            metrics = self.train_epoch(dataloader, epoch)
            history.append(metrics)
            logger.info(f"Router epoch {epoch}: {metrics}")

            # Save checkpoint
            ckpt_path = self.output_dir / f"router_epoch{epoch}.pt"
            torch.save(self.model.router.state_dict(), str(ckpt_path))

        # Save final
        final_path = self.output_dir / "router_final.pt"
        torch.save(self.model.router.state_dict(), str(final_path))

        # Save history
        with open(self.output_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)

        return history


def train_router(
    prism_config,
    train_data: list[dict],
    tokenizer,
    gpu_id: int = 0,
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 1e-4,
    output_dir: str = "results/stage1/router",
) -> dict:
    """
    Convenience function: build model + train router.

    Returns final metrics dict.
    """
    from prism.model.prism_model import PRISMModel

    device = f"cuda:{gpu_id}"
    model = PRISMModel(prism_config)._load_backbone(device=device)
    model = model.to(device)

    trainer = RouterTrainer(
        prism_model=model,
        tokenizer=tokenizer,
        device=device,
        lr=lr,
        output_dir=output_dir,
    )
    history = trainer.train(train_data, epochs=epochs, batch_size=batch_size)
    return {"history": history, "output_dir": output_dir}


def main():
    """CLI entry point: prism-router"""
    parser = argparse.ArgumentParser(description="Train PRISM domain router")
    parser.add_argument("--config", required=True, help="Path to model YAML config")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output-dir", default="results/stage1/router")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    import yaml
    from prism.model.config import PRISMConfig
    from prism.data.datasets import load_math_dataset, load_numinamath
    from prism.data.domain_split import get_soft_domain_label

    with open(args.config) as f:
        cfg_dict = yaml.safe_load(f)
    config = PRISMConfig(**cfg_dict)

    # Load training data
    logger.info("Loading training data for router...")
    train_data = []
    for domain in config.domains:
        ds = load_math_dataset(domain=domain, max_samples=1000)
        for ex in ds:
            problem = ex.get("problem", "")
            label = get_soft_domain_label(problem, n_domains=config.n_domains)
            train_data.append({"problem": problem, "domain_label": label})

    logger.info(f"Router training data: {len(train_data)} examples")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.backbone_name, trust_remote_code=True)

    result = train_router(
        prism_config=config,
        train_data=train_data,
        tokenizer=tokenizer,
        gpu_id=args.gpu,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_dir=args.output_dir,
    )
    print(f"Router training complete. Output: {result['output_dir']}")


if __name__ == "__main__":
    main()
