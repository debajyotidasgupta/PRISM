"""
DomainRouter: produces soft domain routing weights per phase.

Key design decisions (per user requirement):
1. SOFT routing always — the router outputs a DISTRIBUTION over domains,
   never a single expert selection.
2. PER-PHASE routing — different phases (solve/verify/correct) may activate
   different domain mixtures. An algebra problem may be solved by the algebra
   expert but VERIFIED by the combinatorics or miscellaneous expert.
3. ENTROPY regularization — prevents the router from collapsing to a
   single expert (degenerate solution). We add an entropy bonus to keep
   the distribution spread.
4. The Miscellaneous expert always receives a minimum floor weight to ensure
   cross-domain tools are always accessible.

Architecture:
  Global router: h_K → [B, n_domains]  (problem-level domain affinity)
  Phase routers: for each phase p, h_K → [B, n_domains]  (phase-specific weights)
  Final weight for (phase p, domain d) = softmax(global[d] + phase_p[d])

Parameters:
  Global router:        hidden_dim → 256 → n_domains     ~0.5M params
  Per-phase routers:    hidden_dim → 128 → n_domains × n_phases   ~0.3M params
  Total router params:  ~0.8M
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DomainRouter(nn.Module):
    """
    Multi-phase soft domain router.

    Produces a SOFT weight distribution over domains for each phase.
    The distribution is NEVER collapsed to a single expert.

    Args:
        hidden_dim: Input feature dimension (must match backbone hidden_dim).
        n_domains: Number of domain experts.
        n_phases: Number of reasoning phases (one routing vector per phase).
        router_hidden_dim: Intermediate MLP dimension for global router.
        dropout: Dropout rate.
        misc_floor: Minimum weight guaranteed to Miscellaneous expert.
        entropy_weight: Entropy regularization coefficient (promotes diversity).
    """

    def __init__(
        self,
        hidden_dim: int,
        n_domains: int = 5,
        n_phases: int = 3,
        router_hidden_dim: int = 256,
        dropout: float = 0.1,
        misc_floor: float = 0.1,
        entropy_weight: float = 0.01,
    ):
        super().__init__()
        self.n_domains = n_domains
        self.n_phases = n_phases
        self.misc_floor = misc_floor
        self.entropy_weight = entropy_weight

        # Global router: overall domain affinity for this problem
        self.global_router = nn.Sequential(
            nn.Linear(hidden_dim, router_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(router_hidden_dim, n_domains),
        )

        # Per-phase routers: phase-specific domain preference adjustments
        # These learn WHICH domain is most relevant for each reasoning phase
        # E.g., Verify phase may prefer Miscellaneous for cross-domain checking
        self.phase_routers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 128),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(128, n_domains),
            )
            for _ in range(n_phases)
        ])

    def forward(
        self,
        h: torch.Tensor,
        attention_mask: torch.Tensor = None,
        phase_idx: int = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute soft domain routing weights.

        Args:
            h: Hidden states at insertion point [batch, seq_len, hidden_dim].
            attention_mask: [batch, seq_len], 1=real, 0=pad.
            phase_idx: If provided, returns weights for this specific phase.
                       If None, returns weights for ALL phases: [batch, n_phases, n_domains].

        Returns:
            weights: [batch, n_domains] if phase_idx given,
                     [batch, n_phases, n_domains] if phase_idx is None.
            logits: Same shape as weights (before softmax).
        """
        # Mean pooling over real tokens
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            h_pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        else:
            h_pooled = h.mean(dim=1)  # [B, hidden_dim]

        # Global routing logits
        global_logits = self.global_router(h_pooled)  # [B, n_domains]

        if phase_idx is not None:
            # Single phase
            phase_logits = self.phase_routers[phase_idx](h_pooled)  # [B, n_domains]
            combined_logits = global_logits + phase_logits
            weights = self._apply_floor_and_normalize(combined_logits)
            return weights, combined_logits
        else:
            # All phases
            all_logits = []
            all_weights = []
            for p in range(self.n_phases):
                phase_logits = self.phase_routers[p](h_pooled)
                combined = global_logits + phase_logits
                w = self._apply_floor_and_normalize(combined)
                all_logits.append(combined)
                all_weights.append(w)
            # Stack: [B, n_phases, n_domains]
            return torch.stack(all_weights, dim=1), torch.stack(all_logits, dim=1)

    def _apply_floor_and_normalize(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply softmax then enforce minimum floor on Miscellaneous expert.

        The Miscellaneous expert (last domain, index -1) always gets at least
        `misc_floor` weight so cross-domain tools are always accessible.
        Other domains are renormalized accordingly.
        """
        raw_weights = F.softmax(logits, dim=-1)  # [B, n_domains]

        if self.misc_floor > 0 and self.n_domains >= 5:
            # Miscellaneous is always the last domain
            misc_idx = self.n_domains - 1
            misc_w = raw_weights[:, misc_idx].unsqueeze(-1)  # [B, 1]
            # Apply floor
            floored_misc = misc_w.clamp(min=self.misc_floor)  # [B, 1]
            # Scale other domains to fill remaining weight
            other_w = raw_weights[:, :misc_idx]  # [B, n_domains-1]
            remaining = (1.0 - floored_misc)  # [B, 1]
            other_total = other_w.sum(dim=-1, keepdim=True).clamp(min=1e-9)
            other_normalized = other_w * remaining / other_total
            # Concatenate
            weights = torch.cat([other_normalized, floored_misc], dim=-1)
        else:
            weights = raw_weights

        return weights

    def entropy_loss(self, weights: torch.Tensor) -> torch.Tensor:
        """
        Compute negative entropy of routing distribution.
        Minimizing this loss encourages MORE diverse routing (anti-collapse).

        weights: [batch, n_domains] or [batch, n_phases, n_domains]
        Returns: scalar entropy bonus (mean entropy over batch, negated for minimization)
        """
        eps = 1e-9
        if weights.dim() == 3:
            # [B, n_phases, n_domains] → flatten phases
            weights = weights.view(-1, self.n_domains)
        # Entropy: -sum(p * log(p))
        entropy = -(weights * (weights + eps).log()).sum(dim=-1).mean()
        # We want to MAXIMIZE entropy (minimize -entropy) to prevent collapse
        # Return negative entropy as a loss to be added (with negative sign in total loss)
        return -entropy * self.entropy_weight

    def hard_route(self, h: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """Hard routing via argmax (for ablation A8 only). Returns [batch]."""
        weights, _ = self.forward(h, attention_mask, phase_idx=0)
        return weights.argmax(dim=-1)

    @torch.no_grad()
    def predict_domain(self, h: torch.Tensor, attention_mask: torch.Tensor = None) -> list[str]:
        """Diagnostic: top domain per example (not used in training)."""
        indices = self.hard_route(h, attention_mask)
        if hasattr(self, "domain_names"):
            return [self.domain_names[i] for i in indices.tolist()]
        return indices.tolist()
