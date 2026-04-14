"""
PRISMConfig: configuration class for the PRISM model.
Subclasses PretrainedConfig for full HuggingFace compatibility.
"""

from transformers import PretrainedConfig


DOMAIN_NAMES = ["algebra", "geometry", "combinatorics", "number_theory", "miscellaneous"]
PHASE_NAMES = ["solve", "verify", "correct"]


class PRISMConfig(PretrainedConfig):
    """
    Configuration for PRISMModel.

    The PRISM model inserts N_domains × N_phases expert blocks into a frozen
    Qwen3.5-0.8B backbone at a single insertion point (~60-70% depth).

    Args:
        backbone_name: HF model ID of the base VLM (Qwen3.5-0.8B).
        n_domains: Number of domain experts (5: Algebra, Geometry, Combinatorics, NT, Misc).
        n_phases: Number of reasoning phases (3: Solve, Verify, Correct).
        hidden_dim: Hidden dimension of the backbone and expert blocks. Must match backbone.
        insert_layer: Layer index after which PRISM blocks are inserted. Default -1 = auto.
        expert_n_heads: Number of attention heads in each expert block.
        expert_head_dim: Head dimension in each expert block.
        expert_ffn_expansion: FFN expansion ratio in expert blocks (SwiGLU).
        crossmix_n_heads: Number of attention heads in cross-domain mixing.
        router_hidden_dim: Hidden dimension of the router MLP.
        router_dropout: Dropout in the domain router.
        domains: List of domain name strings (length = n_domains).
        phases: List of phase name strings (length = n_phases).
    """

    model_type = "prism"
    _auto_class = "AutoModel"

    def __init__(
        self,
        backbone_name: str = "Qwen/Qwen3.5-0.8B",
        n_domains: int = 5,
        n_phases: int = 3,
        hidden_dim: int = 1024,
        insert_layer: int = -1,
        expert_n_heads: int = 8,
        expert_head_dim: int = 64,
        expert_ffn_expansion: int = 4,
        crossmix_n_heads: int = 4,
        crossphase_n_heads: int = 2,
        crossphase_head_dim: int = 32,
        phase_aggregate_mode: str = "mean",  # "mean" | "last" | "weighted"
        router_hidden_dim: int = 256,
        router_dropout: float = 0.1,
        router_temperature: float = 1.0,     # <1 = sharper routing, >1 = softer
        domains: list = None,
        phases: list = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.backbone_name = backbone_name
        self.n_domains = n_domains
        self.n_phases = n_phases
        self.hidden_dim = hidden_dim
        self.insert_layer = insert_layer
        self.expert_n_heads = expert_n_heads
        self.expert_head_dim = expert_head_dim
        self.expert_ffn_expansion = expert_ffn_expansion
        self.crossmix_n_heads = crossmix_n_heads
        self.crossphase_n_heads = crossphase_n_heads
        self.crossphase_head_dim = crossphase_head_dim
        self.phase_aggregate_mode = phase_aggregate_mode
        self.router_hidden_dim = router_hidden_dim
        self.router_dropout = router_dropout
        self.router_temperature = router_temperature
        self.domains = domains if domains is not None else DOMAIN_NAMES[:n_domains]
        self.phases = phases if phases is not None else PHASE_NAMES[:n_phases]

    @property
    def domain_to_idx(self) -> dict:
        return {d: i for i, d in enumerate(self.domains)}

    @property
    def phase_to_idx(self) -> dict:
        return {p: i for i, p in enumerate(self.phases)}

    def __repr__(self) -> str:
        return (
            f"PRISMConfig(backbone={self.backbone_name}, "
            f"n_domains={self.n_domains}, n_phases={self.n_phases}, "
            f"hidden_dim={self.hidden_dim}, insert_layer={self.insert_layer})"
        )
