"""
CrossMixModule: cross-domain mixing via lightweight cross-attention.

Between each reasoning level, this module allows domain experts to query
each other's representations, capturing inter-domain dependencies:
  - Geometry queries Miscellaneous (AM-GM bounds, trig inequalities)
  - Combinatorics queries Number Theory (modular counting)
  - Number Theory queries Algebra (polynomial Diophantine techniques)
  etc.

Input: expert_outputs — list of N tensors, each [batch, seq_len, hidden_dim]
Output: mixed_outputs — list of N tensors, same shape
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossMixModule(nn.Module):
    """
    Cross-domain attention mixing.

    For each domain d, computes cross-attention where:
      - query = linear projection of expert_d's output
      - keys/values = concatenated outputs of all N experts

    This is very lightweight: n_heads=4, head_dim=32.

    Args:
        hidden_dim: Feature dimension (must match expert block output).
        n_domains: Number of expert domains.
        n_heads: Number of attention heads for cross-attention.
        head_dim: Dimension per head.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_domains: int,
        n_heads: int = 4,
        head_dim: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_domains = n_domains
        self.n_heads = n_heads
        self.head_dim = head_dim
        inner_dim = n_heads * head_dim

        # Per-domain projections
        self.q_projs = nn.ModuleList([
            nn.Linear(hidden_dim, inner_dim, bias=False)
            for _ in range(n_domains)
        ])
        self.k_proj = nn.Linear(hidden_dim, inner_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, inner_dim, bias=False)
        self.out_projs = nn.ModuleList([
            nn.Linear(inner_dim, hidden_dim, bias=False)
            for _ in range(n_domains)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(n_domains)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        expert_outputs: list[torch.Tensor],
        attention_mask: torch.Tensor = None,
    ) -> list[torch.Tensor]:
        """
        Args:
            expert_outputs: list of N tensors, each [batch, seq_len, hidden_dim]
            attention_mask: [batch, seq_len]

        Returns:
            mixed_outputs: list of N tensors, each [batch, seq_len, hidden_dim]
        """
        B, T, D = expert_outputs[0].shape
        H, Dh = self.n_heads, self.head_dim

        # Concatenate all experts along sequence dim for keys/values
        # Shape: [batch, N*seq_len, hidden_dim]
        all_experts = torch.cat(expert_outputs, dim=1)
        keys = self.k_proj(all_experts).view(B, self.n_domains * T, H, Dh).transpose(1, 2)
        vals = self.v_proj(all_experts).view(B, self.n_domains * T, H, Dh).transpose(1, 2)

        mixed_outputs = []
        for d in range(self.n_domains):
            x = expert_outputs[d]
            # Query from domain d
            q = self.q_projs[d](x).view(B, T, H, Dh).transpose(1, 2)  # [B, H, T, Dh]

            # Cross-attention: each position queries all N*T positions
            # Build additive mask over the full N*T key sequence
            attn_mask = None
            if attention_mask is not None:
                full_mask = attention_mask.repeat(1, self.n_domains)  # [B, N*T]
                attn_mask = (1.0 - full_mask[:, None, None, :].to(dtype=q.dtype)) * -1e4

            dropout_p = self.dropout.p if self.training else 0.0
            out = F.scaled_dot_product_attention(q, keys, vals, attn_mask=attn_mask, dropout_p=dropout_p)
            # [B, H, T, Dh]
            out = out.transpose(1, 2).contiguous().view(B, T, H * Dh)
            out = self.out_projs[d](out)

            # Residual + norm
            mixed = self.norms[d](x + self.dropout(out))
            mixed_outputs.append(mixed)

        return mixed_outputs
