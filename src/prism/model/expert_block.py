"""
ExpertBlock: a single domain-expert reasoning block.

Each expert block is a compact transformer consisting of:
  - LayerNorm → Multi-head Self-Attention (n_heads=8, head_dim=64)
  - LayerNorm → FFN (4× expansion, SwiGLU activation)
  - Residual connections throughout
  - Input/output: [batch, seq_len, hidden_dim]
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """SwiGLU activation: gate(x) * silu(x)."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        # Use 2/3 of hidden_dim to keep parameter count similar to standard FFN
        hidden_dim = int(2 * hidden_dim / 3)
        # Round to multiple of 64 for efficiency
        hidden_dim = 64 * ((hidden_dim + 63) // 64)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with causal masking support."""

    def __init__(self, hidden_dim: int, n_heads: int, head_dim: int, dropout: float = 0.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        inner_dim = n_heads * head_dim

        self.q_proj = nn.Linear(hidden_dim, inner_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, inner_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, inner_dim, bias=False)
        self.out_proj = nn.Linear(inner_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        B, T, D = x.shape
        H, Dh = self.n_heads, self.head_dim

        q = self.q_proj(x).view(B, T, H, Dh).transpose(1, 2)   # [B, H, T, Dh]
        k = self.k_proj(x).view(B, T, H, Dh).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, Dh).transpose(1, 2)

        # Build additive attention mask: [B, 1, 1, T] in query dtype
        attn_mask = None
        if attention_mask is not None:
            attn_mask = (1.0 - attention_mask[:, None, None, :].to(dtype=q.dtype)) * -1e4

        # F.scaled_dot_product_attention dispatches to FlashAttention-2 automatically
        # when flash_attn is installed, falling back to efficient SDPA otherwise.
        # Handles scaling (1/sqrt(Dh)) and softmax internally.
        dropout_p = self.dropout.p if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p)

        out = out.transpose(1, 2).contiguous().view(B, T, H * Dh)
        return self.out_proj(out)


class ExpertBlock(nn.Module):
    """
    A single expert block for one (domain, phase) pair.

    Architecture:
        h = h + Attention(LayerNorm(h))
        h = h + FFN(LayerNorm(h))

    Input/output shape: [batch, seq_len, hidden_dim]
    """

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int = 8,
        head_dim: int = 64,
        ffn_expansion: int = 4,
        dropout: float = 0.0,
        domain_name: str = "unknown",
        phase_name: str = "unknown",
    ):
        super().__init__()
        self.domain_name = domain_name
        self.phase_name = phase_name
        self.hidden_dim = hidden_dim

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = MultiHeadSelfAttention(hidden_dim, n_heads, head_dim, dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = SwiGLU(hidden_dim, hidden_dim * ffn_expansion)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, hidden_dim]
            attention_mask: [batch, seq_len] with 1=attend, 0=mask

        Returns:
            h: [batch, seq_len, hidden_dim]
        """
        # Self-attention with residual
        h = x + self.dropout(self.attn(self.norm1(x), attention_mask))
        # FFN with residual
        h = h + self.dropout(self.ffn(self.norm2(h)))
        return h

    def extra_repr(self) -> str:
        return f"domain={self.domain_name}, phase={self.phase_name}, dim={self.hidden_dim}"
