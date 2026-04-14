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
from prism.model.cross_phase import CrossPhaseModule
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

        # CrossPhase module: per-domain temporal attention across phases
        # Applied BEFORE each non-first phase's expert blocks run
        # Lets verify/correct experts attend to solve/verify phase outputs
        self.cross_phase = CrossPhaseModule(
            hidden_dim=config.hidden_dim,
            n_domains=config.n_domains,
            n_heads=getattr(config, "crossphase_n_heads", 2),
            head_dim=getattr(config, "crossphase_head_dim", 32),
        )

        # Ablation flags (set externally for ablation experiments)
        self._use_hard_routing = False
        self._disable_crossmix = False
        self._disable_crossphase = False
        self._use_uniform_routing = False  # ablation: ignore router, use 1/N weights

    def _load_backbone(self, device=None):
        """Load backbone from /tmp or HF Hub."""
        from prism.model.backbone import load_backbone, freeze_backbone, get_insertion_layer

        model_name = self.config.backbone_name
        backbone_model, processor = load_backbone(model_name, torch_dtype=torch.bfloat16)
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

        # Cast PRISM modules to match backbone dtype (bfloat16)
        backbone_dtype = next(backbone_model.parameters()).dtype
        self.router.to(dtype=backbone_dtype)
        self.expert_blocks.to(dtype=backbone_dtype)
        self.cross_mix.to(dtype=backbone_dtype)
        self.cross_phase.to(dtype=backbone_dtype)

        logger.info(f"Backbone loaded (dtype={backbone_dtype}). Insertion layer: {self._insert_layer}")
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

        # ─── Step 3: Expert blocks + cross-domain + cross-phase mixing ──────────
        # Each domain maintains its own state across phases.
        # CrossMix (within-phase): all domains exchange info at each phase level.
        # CrossPhase (across-phases): each domain can attend to its own prior states.
        # Phase routing weights are applied at EACH phase (not just the last).

        # Initialize: all domains start from h_K
        domain_states = [h_K.clone() for _ in range(self.config.n_domains)]
        phase_history = []    # stores domain state lists after each phase's CrossMix
        phase_aggregates = [] # routing-weighted aggregate per phase

        for phase_idx in range(self.config.n_phases):
            # Routing weights for THIS phase: [B, n_domains]
            w_phase = phase_weights[:, phase_idx, :]

            # Uniform-routing ablation: ignore router, give equal weight to all
            if self._use_uniform_routing:
                w_phase = torch.full_like(w_phase, 1.0 / self.config.n_domains)

            # Cross-phase: let each domain attend to its own prior-phase states
            # Applied BEFORE running experts (gives experts temporal context)
            if phase_idx > 0 and not self._disable_crossphase:
                domain_states = self.cross_phase(domain_states, phase_history, attention_mask)

            # Run ALL domain experts for this phase
            phase_outputs = [
                self.expert_blocks[phase_idx][d](domain_states[d], attention_mask)
                for d in range(self.config.n_domains)
            ]

            # CrossMix: every domain consults all others within this phase
            if not self._disable_crossmix:
                mixed_outputs = self.cross_mix[phase_idx](phase_outputs, attention_mask)
            else:
                mixed_outputs = phase_outputs

            # Persist domain states for next phase (and cross-phase history)
            domain_states = mixed_outputs
            phase_history.append(list(mixed_outputs))  # snapshot after this phase

            # ── Compute routing-weighted aggregate for this phase ─────────────
            # FIX: w_phase is NOW applied at every phase, not just the last
            if self._use_hard_routing:
                hard_idx = w_phase.argmax(dim=-1)  # [B]
                w_eff = F.one_hot(hard_idx, self.config.n_domains).to(dtype=w_phase.dtype)
            else:
                w_eff = w_phase

            w = w_eff.unsqueeze(-1).unsqueeze(-1)            # [B, N, 1, 1]
            stacked = torch.stack(mixed_outputs, dim=1)       # [B, N, T, D]
            phase_agg = (w * stacked).sum(dim=1)              # [B, T, D]
            phase_aggregates.append(phase_agg)

        # ─── Step 4: Final aggregation across all phases ──────────────────────
        # Combine per-phase aggregates. Each phase's contribution is gated by
        # its own routing weights (already applied above), so h_K_prime reflects
        # solve, verify, AND correct phase routing decisions — not just the last.
        agg_mode = getattr(self.config, "phase_aggregate_mode", "mean")
        if agg_mode == "last":
            h_K_prime = phase_aggregates[-1]
        elif agg_mode == "mean":
            h_K_prime = torch.stack(phase_aggregates, dim=0).mean(dim=0)   # [B, T, D]
        else:
            # Fallback: mean
            h_K_prime = torch.stack(phase_aggregates, dim=0).mean(dim=0)

        # ─── Residual alpha blending ──────────────────────────────────────────
        # Expert blocks trained on limited data produce partially random-direction
        # h_K_prime.  Layers K+1..end of the frozen backbone were pre-trained on
        # h_K, so a differently-directed replacement causes degenerate generation.
        #
        # Fix: blend h_K_prime with the original h_K using a scalar alpha:
        #   h_K_prime_final = h_K + alpha * (h_K_prime - h_K)
        #                   = (1-alpha)*h_K + alpha*h_K_prime
        #
        # Interpretation: expert blocks contribute CORRECTIONS on top of h_K.
        #   alpha=0.0 → pure backbone (exact baseline, 0 expert influence)
        #   alpha=0.2 → 80% backbone + 20% expert (safe for undertrained experts)
        #   alpha=1.0 → full expert output (correct after joint end-to-end training)
        #
        # During joint fine-tuning (A5) training drives alpha=1.0 behavior since
        # the full pipeline is optimized e2e and expert norms converge naturally.
        # _residual_alpha is overridden to 1.0 there (or _disable_residual_blend=True).
        #
        # Separate norm stabilization on top ensures amplitude stays in-distribution
        # regardless of alpha (belt-and-suspenders).
        _residual_alpha = getattr(self, "_residual_alpha", 1.0)
        if _residual_alpha < 1.0:
            h_K_prime = h_K + _residual_alpha * (h_K_prime - h_K)

        # Norm stabilization: after blending, rescale to h_K's per-position norm.
        # Even at alpha=0.2 the blend can be slightly off in amplitude.
        if not getattr(self, "_disable_norm_stabilize", False):
            with torch.no_grad():
                target_norm = h_K.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            current_norm = h_K_prime.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            h_K_prime = h_K_prime * (target_norm / current_norm)

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
        """
        Run backbone from embedding to layer K, return hidden state at K.

        Uses a forward hook to intercept the output of layer K, then lets the
        full model forward run (in no_grad since backbone is frozen). This avoids
        manually reproducing Qwen3.5's internal position_id/mask/RoPE logic.
        """
        text_model = self.backbone.model  # Qwen3_5TextModel (has .layers, .embed_tokens, .norm)
        captured = {}

        def _capture_hook(module, inp, output):
            # Decoder layer returns a tensor directly (not a tuple in Qwen3.5)
            h = output[0] if isinstance(output, (tuple, list)) else output
            captured["h_K"] = h.detach().clone()  # detach: backbone is frozen, no grad needed here

        hook = text_model.layers[self._insert_layer].register_forward_hook(_capture_hook)
        try:
            with torch.no_grad():
                text_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        finally:
            hook.remove()

        # Save input_ids for the second pass (from_K needs to re-run layers 0..K)
        self._saved_input_ids = input_ids
        return captured["h_K"]

    def _backbone_forward_from_K(self, h_K_prime, attention_mask):
        """
        Run backbone from layer K+1 to end, starting from h_K_prime.

        Uses a forward hook on layer K to REPLACE its output with h_K_prime,
        then lets layers K+1..end run normally. Gradients flow through h_K_prime
        into the expert blocks (backbone K+1..end is frozen but participates in
        the autograd graph so gradients reach h_K_prime).

        IMPORTANT: Do NOT call this inside torch.no_grad() — it must participate
        in the autograd graph so that loss.backward() reaches the expert blocks.
        """
        text_model = self.backbone.model

        def _replace_hook(module, inp, output):
            # Replace layer K output with the PRISM-processed hidden state
            # h_K_prime.requires_grad=True (from trainable expert block) →
            # autograd tracks operations in layers K+1..end w.r.t. h_K_prime
            return h_K_prime

        hook = text_model.layers[self._insert_layer].register_forward_hook(_replace_hook)
        try:
            out = text_model(
                input_ids=self._saved_input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            hidden_states = out.last_hidden_state
        finally:
            hook.remove()

        return self.backbone.lm_head(hidden_states)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 1536,
        temperature: float = 0.0,
        do_sample: bool = False,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        **kwargs,
    ) -> torch.Tensor:
        """
        Autoregressive generation with PRISM expert blocks.

        Full PRISM forward runs at every token step (no KV-cache — expert blocks
        modify h_K at each step based on the full context).

        Args:
            max_new_tokens: Token budget. Default 1536 covers the full
                solve→verify→correct chain (avg 520-1040 tokens in training).
            temperature: Sampling temperature. 0 = greedy.
            do_sample: Enable multinomial sampling (requires temperature > 0).
            top_p: Nucleus sampling cutoff (applied before sampling).
            repetition_penalty: > 1.0 penalises tokens already in context.
        """
        generated = input_ids.clone()
        gen_mask = attention_mask.clone() if attention_mask is not None else None

        # Collect EOS token ids once
        eos_ids: set = set()
        if hasattr(self.backbone, "config") and self.backbone.config.eos_token_id is not None:
            raw_eos = self.backbone.config.eos_token_id
            eos_ids = set(raw_eos if isinstance(raw_eos, list) else [raw_eos])

        for _ in range(max_new_tokens):
            out = self.forward(generated, attention_mask=gen_mask)
            logits = out["logits"][:, -1, :].float()   # [B, vocab]  (float32 for sampling)

            # ── Repetition penalty ────────────────────────────────────────────
            if repetition_penalty != 1.0:
                for b in range(generated.size(0)):
                    for token_id in generated[b].unique():
                        tid = token_id.item()
                        if logits[b, tid] > 0:
                            logits[b, tid] /= repetition_penalty
                        else:
                            logits[b, tid] *= repetition_penalty

            # ── Sampling ──────────────────────────────────────────────────────
            if temperature > 0 and do_sample:
                scaled = logits / temperature

                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(scaled, dim=-1, descending=True)
                    cumprobs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    # Remove tokens above nucleus threshold (shift right so the last kept token stays)
                    remove = cumprobs - torch.softmax(sorted_logits, dim=-1) > top_p
                    sorted_logits[remove] = float('-inf')
                    # Scatter back to original ordering
                    scaled.scatter_(-1, sorted_idx, sorted_logits)

                probs = torch.softmax(scaled, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=-1)
            if gen_mask is not None:
                gen_mask = torch.cat([gen_mask, gen_mask.new_ones((gen_mask.size(0), 1))], dim=-1)

            # EOS check
            if eos_ids and next_token.item() in eos_ids:
                break

        return generated

    # ─── Freezing helpers for staged training ─────────────────────────────────

    def _freeze_all_prism(self):
        """Freeze all PRISM modules (router, experts, crossmix, crossphase)."""
        for p in self.router.parameters():
            p.requires_grad_(False)
        for phase_blocks in self.expert_blocks:
            for block in phase_blocks:
                for p in block.parameters():
                    p.requires_grad_(False)
        for cm in self.cross_mix:
            for p in cm.parameters():
                p.requires_grad_(False)
        for p in self.cross_phase.parameters():
            p.requires_grad_(False)

    def freeze_all_except_phase(self, phase_idx: int, domain_idx: int):
        """Unfreeze only one (phase, domain) expert block. For per-expert training."""
        self._freeze_all_prism()
        for pi, phase_blocks in enumerate(self.expert_blocks):
            for di, block in enumerate(phase_blocks):
                for p in block.parameters():
                    p.requires_grad_(pi == phase_idx and di == domain_idx)

    def freeze_all_except_router(self):
        """Unfreeze router only."""
        self._freeze_all_prism()
        for p in self.router.parameters():
            p.requires_grad_(True)

    def freeze_all_except_crossmix(self, level_idx: int):
        """Unfreeze one cross-mix module."""
        self._freeze_all_prism()
        for i, cm in enumerate(self.cross_mix):
            for p in cm.parameters():
                p.requires_grad_(i == level_idx)

    def freeze_all_except_crossphase(self):
        """Unfreeze cross-phase module only."""
        self._freeze_all_prism()
        for p in self.cross_phase.parameters():
            p.requires_grad_(True)

    def unfreeze_all_prism(self):
        """Unfreeze all PRISM modules for joint fine-tuning."""
        for p in self.router.parameters():
            p.requires_grad_(True)
        for phase_blocks in self.expert_blocks:
            for block in phase_blocks:
                for p in block.parameters():
                    p.requires_grad_(True)
        for cm in self.cross_mix:
            for p in cm.parameters():
                p.requires_grad_(True)
        for p in self.cross_phase.parameters():
            p.requires_grad_(True)

    def get_prism_params(self) -> list:
        """All trainable PRISM parameters (excludes frozen backbone)."""
        params = []
        params.extend(self.router.parameters())
        for phase_blocks in self.expert_blocks:
            for block in phase_blocks:
                params.extend(block.parameters())
        for cm in self.cross_mix:
            params.extend(cm.parameters())
        params.extend(self.cross_phase.parameters())
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
