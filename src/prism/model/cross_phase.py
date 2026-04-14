"""
CrossPhaseModule: per-domain temporal attention across reasoning phases.

Between phases, each domain expert can attend to its OWN state from prior
phases. This allows the verify phase algebra expert to query what the solve
phase algebra expert computed — explicit phase-to-phase memory.

Design:
  - Applied BEFORE phase P experts run (P > 0)
  - Each domain d's current state queries its concatenated prior-phase states
  - Keys/values from all prior phases of the SAME domain
  - Lightweight: n_heads=2, head_dim=32 per domain

Input:  current_states: list[N] of [B, T, D]  (domain states entering phase P)
        history_states: list of list[N] of [B, T, D]  (one list per prior phase)
Output: updated current_states: list[N] of [B, T, D]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossPhaseModule(nn.Module):
    """
    Per-domain cross-phase temporal attention.

    Each domain d in the current phase attends to its own state from all
    previous phases. This gives each expert temporal context of what it (and
    other phases of the same domain) previously computed.

    Args:
        hidden_dim: Feature dimension (matches expert block output).
        n_domains: Number of domain experts.
        n_heads: Attention heads for cross-phase attention.
        head_dim: Dimension per head.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_domains: int,
        n_heads: int = 2,
        head_dim: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_domains = n_domains
        self.n_heads = n_heads
        self.head_dim = head_dim
        inner_dim = n_heads * head_dim

        # Per-domain query projections (current phase state → query)
        self.q_projs = nn.ModuleList([
            nn.Linear(hidden_dim, inner_dim, bias=False)
            for _ in range(n_domains)
        ])
        # Shared key/value projections (applied to historical states)
        self.k_proj = nn.Linear(hidden_dim, inner_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, inner_dim, bias=False)
        # Per-domain output projections
        self.out_projs = nn.ModuleList([
            nn.Linear(inner_dim, hidden_dim, bias=False)
            for _ in range(n_domains)
        ])
        # Per-domain layer norms for residual
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(n_domains)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        current_states: list,
        history_states: list,
        attention_mask: torch.Tensor = None,
    ) -> list:
        """
        Update each domain's current state by attending to its prior-phase history.

        Args:
            current_states: list of N tensors [B, T, D] — domain states entering current phase
            history_states: list of lists — history_states[phase][domain] = [B, T, D]
                            Contains all prior phase domain states.
            attention_mask: [B, T] padding mask (1=real, 0=pad)

        Returns:
            updated_states: list of N tensors [B, T, D]
        """
        if not history_states:
            return current_states  # phase 0: nothing to attend to

        B, T, D = current_states[0].shape
        H, Dh = self.n_heads, self.head_dim
        n_prior = len(history_states)  # number of prior phases

        updated = []
        for d in range(self.n_domains):
            x = current_states[d]  # [B, T, D] — current state for domain d

            # Collect this domain's states from ALL prior phases
            # Shape: [B, n_prior * T, D]
            prior_d = torch.cat([history_states[p][d] for p in range(n_prior)], dim=1)

            # Query from current state, keys/values from prior phases
            q = self.q_projs[d](x).view(B, T, H, Dh).transpose(1, 2)         # [B, H, T, Dh]
            k = self.k_proj(prior_d).view(B, n_prior * T, H, Dh).transpose(1, 2)   # [B, H, P*T, Dh]
            v = self.v_proj(prior_d).view(B, n_prior * T, H, Dh).transpose(1, 2)   # [B, H, P*T, Dh]

            # Cross-phase attention: Q (current) → K/V (all prior phases concatenated)
            attn_mask = None
            if attention_mask is not None:
                full_mask = attention_mask.repeat(1, n_prior)  # [B, P*T]
                attn_mask = (1.0 - full_mask[:, None, None, :].to(dtype=q.dtype)) * -1e4

            dropout_p = self.dropout.p if self.training else 0.0
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p)
            # [B, H, T, Dh]
            out = out.transpose(1, 2).contiguous().view(B, T, H * Dh)
            out = self.out_projs[d](out)                          # [B, T, D]

            # Residual + norm
            updated.append(self.norms[d](x + self.dropout(out)))

        return updated
