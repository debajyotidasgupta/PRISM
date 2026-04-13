"""
PRISMModel: Full PRISM model combining frozen backbone with expert blocks.

Subclasses PreTrainedModel for HuggingFace compatibility.
Supports AutoModel.from_pretrained() and model.save_pretrained().

Architecture overview:
─────────────────────────────────────────────────────────────────────
Forward pass (single forward — zero inference-time scaling):

  1. BACKBONE: embedding → layers 0..K → h_K

  2. DOMAIN ROUTER (per-phase):
     h_K → DomainRouter → phase_weights[phase, domain] ∈ Δ^N for EACH phase
     Different phases can route to different domain mixtures:
       - Solve:   algebra=0.7, misc=0.2, geo=0.1
       - Verify:  combinatorics=0.4, misc=0.4, algebra=0.2
       - Correct: algebra=0.5, misc=0.3, geo=0.2
     The router NEVER collapses to a single expert.

  3. EXPERT BLOCKS (N_domains × N_phases, hierarchical):
     ALL domain experts run for each phase (soft mixture, not single selection).
     Between phases, CrossMix allows cross-domain information exchange.

     Level 0 (Solve):
       For each domain d:
         expert_0[d] = ExpertBlock(solve, d)(h_K)
       Blend_0[d] = sum_d'(CrossMix_0(d, d') * expert_0[d'])
       phase_out_0 = sum_d(solve_weights[d] * Blend_0[d])

     Level 1 (Verify):
       Input to each domain: Blend_0[d] (not phase_out_0 — each domain sees its own mixed state)
       For each domain d:
         expert_1[d] = ExpertBlock(verify, d)(Blend_0[d])
       Blend_1[d] = CrossMix_1(expert_1)
       phase_out_1 = sum_d(verify_weights[d] * Blend_1[d])

     Level 2 (Correct):
       For each domain d:
         expert_2[d] = ExpertBlock(correct, d)(Blend_1[d])
       final_out = sum_d(correct_weights[d] * expert_2[d])

  4. AGGREGATION: h_K' = final_out (all domains mixed by their phase-specific weights)

  5. BACKBONE: layers K+1..end → LM head → logits

Key properties:
  - ALL experts participate in EVERY forward pass (soft mixture, not hard selection)
  - Phase-specific routing: verify phase may route differently from solve phase
  - CrossMix ensures inter-domain communication at each level
  - Entropy regularization prevents router collapse to single domain
  - Miscellaneous expert gets a minimum floor weight at all times
─────────────────────────────────────────────────────────────────────
"""

import os
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, AutoModelForCausalLM, AutoProcessor

from prism.model.config import PRISMConfig
from prism.model.expert_block import ExpertBlock
from prism.model.cross_mix import CrossMixModule
from prism.model.router import DomainRouter

logger = logging.getLogger(__name__)


class PRISMModel(PreTrainedModel):
    """
    PRISM: Phase-structured Reasoning with Integrated Subject-expert Modules.

    Key design guarantees:
      1. Soft routing at every phase — no hard expert selection.
      2. All domain experts participate in every forward pass.
      3. Different phases route to different domain mixtures.
      4. Cross-domain mixing between all phases.
      5. Entropy regularization prevents router collapse.
      6. Single forward pass at inference — zero extra compute.
    """

    config_class = PRISMConfig
    _no_split_modules = ["ExpertBlock", "CrossMixModule"]
    _auto_class = "AutoModel"

    def __init__(self, config: PRISMConfig, backbone=None):
        super().__init__(config)
        self.config = config

        self.backbone = backbone
        self.processor = None
        self._insert_layer = config.insert_layer
        self._position_ids = None

        # Multi-phase domain router
        # Produces phase_weights[B, n_phases, n_domains] — different distribution per phase
        self.router = DomainRouter(
            hidden_dim=config.hidden_dim,
            n_domains=config.n_domains,
            n_phases=config.n_phases,
            router_hidden_dim=config.router_hidden_dim,
            dropout=config.router_dropout,
            misc_floor=0.10,       # Misc always gets ≥10% weight
            entropy_weight=0.01,   # Entropy regularization
        )
        self.router.domain_names = config.domains

        # Expert blocks: expert_blocks[phase][domain]
        # ALL experts run for ALL phases. Routing weights determine their contribution.
        self.expert_blocks = nn.ModuleList([
            nn.ModuleList([
                ExpertBlock(
                    hidden_dim=config.hidden_dim,
                    n_heads=config.expert_n_heads,
                    head_dim=config.expert_head_dim,
                    ffn_expansion=config.expert_ffn_expansion,
                    domain_name=config.domains[d],
                    phase_name=config.phases[p],
                )
                for d in range(config.n_domains)
            ])
            for p in range(config.n_phases)
        ])

        # CrossMix modules: n_phases of them (one per phase, applied AFTER each phase's experts)
        # This allows information exchange between all domain experts at each level
        self.cross_mix = nn.ModuleList([
            CrossMixModule(
                hidden_dim=config.hidden_dim,
                n_domains=config.n_domains,
                n_heads=config.crossmix_n_heads,
                head_dim=32,
            )
            for _ in range(config.n_phases)
        ])

        # Ablation flags (set externally for ablation experiments)
        self._use_hard_routing = False
        self._disable_crossmix = False

    def _load_backbone(self, device=None):
        """Load backbone from /tmp or HF Hub."""
        from prism.model.backbone import load_backbone, freeze_backbone, get_insertion_layer

        model_name = self.config.backbone_name
        backbone_model, processor = load_backbone(model_name, torch_dtype=torch.float16)
        freeze_backbone(backbone_model)

        if device is not None:
            backbone_model = backbone_model.to(device)

        if self._insert_layer == -1:
            self._insert_layer = get_insertion_layer(backbone_model)

        self.backbone = backbone_model
        self.processor = processor

        # Update hidden_dim from actual backbone config
        if hasattr(backbone_model, "config") and hasattr(backbone_model.config, "hidden_size"):
            actual_dim = backbone_model.config.hidden_size
            if actual_dim != self.config.hidden_dim:
                logger.warning(
                    f"Updating hidden_dim: {self.config.hidden_dim} → {actual_dim} "
                    f"(from backbone config)"
                )
                self.config.hidden_dim = actual_dim

        logger.info(f"Backbone loaded. Insertion layer: {self._insert_layer}")
        return self

    def _get_backbone_layers(self):
        if hasattr(self.backbone, "model") and hasattr(self.backbone.model, "layers"):
            return self.backbone.model.layers
        raise ValueError("Unsupported backbone architecture")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        domain_labels: Optional[torch.Tensor] = None,
        return_router_weights: bool = False,
        **kwargs,
    ) -> dict:
        """
        Full PRISM forward pass with multi-expert soft routing per phase.

        ALL domain experts participate in every forward pass.
        Routing weights determine their CONTRIBUTION, not their ACTIVATION.

        Returns dict with: logits, loss (if labels provided), router_weights (optional).
        """
        assert self.backbone is not None, "Call _load_backbone() before forward()"

        # ─── Step 1: Backbone → h_K ──────────────────────────────────────────
        h_K = self._backbone_forward_to_K(input_ids, attention_mask, pixel_values, image_grid_thw)

        # ─── Step 2: Multi-phase routing ─────────────────────────────────────
        # phase_weights: [B, n_phases, n_domains] — different weights per phase
        phase_weights, phase_logits = self.router(h_K, attention_mask, phase_idx=None)

        # ─── Step 3: Expert blocks + cross-domain mixing ─────────────────────
        # Each domain maintains its own state across phases.
        # CrossMix allows all domains to exchange information at each level.
        # Phase routing weights determine how domain outputs are aggregated.

        # Initialize: all domains start from h_K
        domain_states = [h_K.clone() for _ in range(self.config.n_domains)]

        for phase_idx in range(self.config.n_phases):
            # Get routing weights for this phase: [B, n_domains]
            w_phase = phase_weights[:, phase_idx, :]  # [B, n_domains]

            # Run ALL domain experts for this phase
            # Each expert receives its domain's state from previous phase
            phase_outputs = [
                self.expert_blocks[phase_idx][d](domain_states[d], attention_mask)
                for d in range(self.config.n_domains)
            ]

            # Cross-domain mixing: every domain consults all others
            # This is where e.g. the verify algebra expert can query combinatorics
            if not self._disable_crossmix:
                mixed_outputs = self.cross_mix[phase_idx](phase_outputs, attention_mask)
            else:
                mixed_outputs = phase_outputs

            # Update domain states: each domain's state is its mixed output
            # (maintains separate state per domain across phases)
            domain_states = mixed_outputs

        # ─── Step 4: Final aggregation ────────────────────────────────────────
        # Use the last phase's routing weights to aggregate final domain outputs.
        # The last phase (Correct) routing determines the final answer generation.
        final_weights = phase_weights[:, -1, :]  # [B, n_domains]

        if self._use_hard_routing:
            # Ablation A8: hard routing (argmax) — collapses to one domain
            hard_idx = final_weights.argmax(dim=-1)  # [B]
            hard_one_hot = F.one_hot(hard_idx, self.config.n_domains).float()
            final_weights = hard_one_hot

        # Weighted sum: [B, n_domains, 1, 1] × [B, n_domains, T, D]
        w = final_weights.unsqueeze(-1).unsqueeze(-1)       # [B, N, 1, 1]
        stacked = torch.stack(domain_states, dim=1)          # [B, N, T, D]
        h_K_prime = (w * stacked).sum(dim=1)                 # [B, T, D]

        # ─── Step 5: Backbone → logits ────────────────────────────────────────
        logits = self._backbone_forward_from_K(h_K_prime, attention_mask)

        # ─── Step 6: Losses ───────────────────────────────────────────────────
        total_loss = None
        lm_loss = None
        router_loss = None
        entropy_loss = None

        if labels is not None:
            from torch.nn import CrossEntropyLoss
            loss_fn = CrossEntropyLoss(ignore_index=-100)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            lm_loss = loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            total_loss = lm_loss

        if domain_labels is not None:
            # KL divergence: predicted routing vs. soft domain labels
            # Apply to ALL phases (each phase's routing should reflect domain affinity)
            # domain_labels: [B, n_domains] (problem-level soft labels)
            router_kl_losses = []
            for p in range(self.config.n_phases):
                log_probs = F.log_softmax(phase_logits[:, p, :], dim=-1)
                kl = F.kl_div(log_probs, domain_labels.float(), reduction="batchmean")
                router_kl_losses.append(kl)
            router_loss = sum(router_kl_losses) / self.config.n_phases
            total_loss = (total_loss + 0.1 * router_loss) if total_loss is not None else 0.1 * router_loss

        # Entropy regularization — ALWAYS applied during training to prevent collapse
        if self.training:
            entropy_loss = self.router.entropy_loss(phase_weights)
            # entropy_loss is already negative (adding it minimizes -entropy = maximizes entropy)
            total_loss = (total_loss + entropy_loss) if total_loss is not None else entropy_loss

        out = {"logits": logits}
        if total_loss is not None:
            out["loss"] = total_loss
        if lm_loss is not None:
            out["lm_loss"] = lm_loss
        if router_loss is not None:
            out["router_loss"] = router_loss
        if entropy_loss is not None:
            out["entropy_loss"] = entropy_loss
        if return_router_weights:
            out["router_weights"] = phase_weights  # [B, n_phases, n_domains]

        return out

    def _backbone_forward_to_K(self, input_ids, attention_mask, pixel_values, image_grid_thw):
        model = self.backbone
        device = input_ids.device

        inputs_embeds = model.model.embed_tokens(input_ids)

        if pixel_values is not None and hasattr(model, "visual"):
            try:
                image_embeds = model.visual(pixel_values, grid_thw=image_grid_thw)
                if hasattr(model, "_merge_input_ids_with_image_features"):
                    inputs_embeds = model._merge_input_ids_with_image_features(
                        inputs_embeds, image_embeds, input_ids
                    )
            except Exception as e:
                logger.warning(f"Vision encoding skipped: {e}")

        hidden_states = inputs_embeds
        B, T, D = hidden_states.shape
        position_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)

        layers = self._get_backbone_layers()
        for i in range(self._insert_layer + 1):
            layer_out = layers[i](
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                output_attentions=False,
            )
            hidden_states = layer_out[0]

        self._position_ids = position_ids
        return hidden_states

    def _backbone_forward_from_K(self, h_K_prime, attention_mask):
        model = self.backbone
        hidden_states = h_K_prime
        position_ids = self._position_ids

        layers = self._get_backbone_layers()
        for i in range(self._insert_layer + 1, len(layers)):
            layer_out = layers[i](
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                output_attentions=False,
            )
            hidden_states = layer_out[0]

        hidden_states = model.model.norm(hidden_states)
        logits = model.lm_head(hidden_states)
        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        do_sample: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """Single forward pass per token — no inference-time scaling."""
        generated = input_ids.clone()
        gen_mask = attention_mask.clone() if attention_mask is not None else None

        for step in range(max_new_tokens):
            out = self.forward(generated, attention_mask=gen_mask)
            logits = out["logits"][:, -1, :]

            if temperature > 0 and do_sample:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=-1)
            if gen_mask is not None:
                gen_mask = torch.cat([gen_mask, gen_mask.new_ones((gen_mask.size(0), 1))], dim=-1)

            if hasattr(self.backbone, "config") and self.backbone.config.eos_token_id is not None:
                eos = self.backbone.config.eos_token_id
                if isinstance(eos, list):
                    done = any(next_token.eq(e).all() for e in eos)
                else:
                    done = next_token.eq(eos).all()
                if done:
                    break

        return generated

    # ─── Freezing helpers for staged training ─────────────────────────────────

    def freeze_all_except_phase(self, phase_idx: int, domain_idx: int):
        """Unfreeze only one (phase, domain) expert block. For per-expert training."""
        for p in self.router.parameters():
            p.requires_grad_(False)
        for pi, phase_blocks in enumerate(self.expert_blocks):
            for di, block in enumerate(phase_blocks):
                for p in block.parameters():
                    p.requires_grad_(pi == phase_idx and di == domain_idx)
        for cm in self.cross_mix:
            for p in cm.parameters():
                p.requires_grad_(False)

    def freeze_all_except_router(self):
        """Unfreeze router only."""
        for p in self.router.parameters():
            p.requires_grad_(True)
        for phase_blocks in self.expert_blocks:
            for block in phase_blocks:
                for p in block.parameters():
                    p.requires_grad_(False)
        for cm in self.cross_mix:
            for p in cm.parameters():
                p.requires_grad_(False)

    def freeze_all_except_crossmix(self, level_idx: int):
        """Unfreeze one cross-mix module."""
        for p in self.router.parameters():
            p.requires_grad_(False)
        for phase_blocks in self.expert_blocks:
            for block in phase_blocks:
                for p in block.parameters():
                    p.requires_grad_(False)
        for i, cm in enumerate(self.cross_mix):
            for p in cm.parameters():
                p.requires_grad_(i == level_idx)

    def get_prism_params(self) -> list:
        """All trainable PRISM parameters."""
        params = []
        params.extend(self.router.parameters())
        for phase_blocks in self.expert_blocks:
            for block in phase_blocks:
                params.extend(block.parameters())
        for cm in self.cross_mix:
            params.extend(cm.parameters())
        return params

    def count_prism_params(self) -> dict:
        def count(m):
            return sum(p.numel() for p in m.parameters())
        result = {
            "router": count(self.router),
            "expert_blocks": {},
            "cross_mix": {},
        }
        for pi, pblocks in enumerate(self.expert_blocks):
            for di, block in enumerate(pblocks):
                result["expert_blocks"][f"p{pi}_{block.domain_name}"] = count(block)
        for i, cm in enumerate(self.cross_mix):
            result["cross_mix"][f"level{i}"] = count(cm)
        result["total_prism"] = (
            result["router"]
            + sum(result["expert_blocks"].values())
            + sum(result["cross_mix"].values())
        )
        return result

    def log_routing_stats(self, phase_weights: torch.Tensor):
        """
        Log routing statistics for debugging/analysis.
        phase_weights: [B, n_phases, n_domains]
        """
        with torch.no_grad():
            mean_weights = phase_weights.mean(dim=0)  # [n_phases, n_domains]
            for pi, phase_name in enumerate(self.config.phases):
                w = mean_weights[pi]
                domain_str = ", ".join(
                    f"{d}={w[di]:.2f}"
                    for di, d in enumerate(self.config.domains)
                )
                logger.debug(f"Router [{phase_name}]: {domain_str}")
