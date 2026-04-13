"""
Backbone loader and insertion point patcher for Qwen3.5-0.8B.

Loads the backbone from /tmp (RAM-backed, fast), identifies the insertion
layer, and provides a forward function that:
  1. Runs layers 0 to K (inclusive)
  2. Returns h_K for PRISM blocks
  3. Accepts h_K' (refined) and runs layers K+1 to end → LM head

The backbone is always frozen. PRISM blocks modify h_K → h_K' only.
"""

import os
import shutil
import logging
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _get_model_dir(model_name: str) -> str:
    """
    Resolve model path. Checks /tmp first, then HF_HOME cache, then returns model_name
    for HF Hub download.
    """
    prism_root = os.environ.get("PRISM_ROOT", "")
    tmp_dir = Path("/tmp/prism_models")
    safe_name = model_name.replace("/", "--")

    tmp_path = tmp_dir / safe_name
    if tmp_path.exists():
        logger.info(f"Loading backbone from /tmp: {tmp_path}")
        return str(tmp_path)

    if prism_root:
        cache_path = Path(prism_root) / ".cache" / "models" / safe_name
        if cache_path.exists():
            logger.info(f"Copying backbone from cache to /tmp: {cache_path} → {tmp_path}")
            tmp_dir.mkdir(parents=True, exist_ok=True)
            shutil.copytree(str(cache_path), str(tmp_path))
            return str(tmp_path)

    logger.info(f"Will download backbone from HF Hub: {model_name}")
    return model_name


def load_backbone(model_name: str, torch_dtype=torch.float16, device_map=None):
    """
    Load Qwen3.5 backbone (or any compatible VLM).
    Returns (model, processor).

    Args:
        model_name: HF model ID or local path.
        torch_dtype: Model dtype. Default fp16.
        device_map: Accelerate device map. None = single GPU.
    """
    from transformers import AutoModelForCausalLM, AutoProcessor

    hf_home = os.environ.get("HF_HOME", None)
    hf_token = os.environ.get("HF_TOKEN", None)

    model_path = _get_model_dir(model_name)

    kwargs = dict(
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    if device_map is not None:
        kwargs["device_map"] = device_map
    if hf_token:
        kwargs["token"] = hf_token

    logger.info(f"Loading backbone: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        token=hf_token,
    )

    return model, processor


def freeze_backbone(model: nn.Module) -> None:
    """Freeze all backbone parameters."""
    for param in model.parameters():
        param.requires_grad_(False)
    logger.info(f"Backbone frozen: {sum(p.numel() for p in model.parameters()):,} params")


def get_num_layers(model: nn.Module) -> int:
    """
    Get number of transformer layers in the backbone.
    Works for Qwen2.5/Qwen3.5 and similar models.
    """
    # Try common attribute patterns
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return len(model.model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return len(model.transformer.h)
    if hasattr(model, "layers"):
        return len(model.layers)
    raise ValueError(f"Cannot determine layer count for {type(model).__name__}")


def get_insertion_layer(model: nn.Module, insert_layer: int = -1) -> int:
    """
    Compute the actual insertion layer index.
    If insert_layer == -1, uses 65% of total layers (empirically optimal).
    """
    n_layers = get_num_layers(model)
    if insert_layer == -1:
        # 65% depth: captures task-relevant representations
        k = int(n_layers * 0.65)
        logger.info(f"Auto-selecting insertion layer: {k}/{n_layers} ({100*k//n_layers}%)")
        return k
    return insert_layer


class BackboneWithInsertionPoint:
    """
    Wrapper that splits backbone forward pass at layer K.

    Usage:
        backbone = BackboneWithInsertionPoint(model, processor, insert_layer=k)
        h_K, cache = backbone.forward_to_insertion(input_ids, attention_mask, pixel_values)
        h_K_prime = prism_blocks(h_K)
        logits = backbone.forward_from_insertion(h_K_prime, cache, attention_mask)
    """

    def __init__(self, model: nn.Module, processor, insert_layer: int):
        self.model = model
        self.processor = processor
        self.insert_layer = insert_layer
        self.model_type = type(model).__name__

        # Detect Qwen architecture
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            self._layers = model.model.layers
            self._embed_tokens = model.model.embed_tokens
            self._norm = model.model.norm
            self._lm_head = model.lm_head
            self._arch = "qwen"
        else:
            raise ValueError(f"Unsupported backbone architecture: {self.model_type}")

    @torch.no_grad()
    def forward_to_insertion(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        pixel_values: torch.Tensor = None,
        image_grid_thw: torch.Tensor = None,
        **kwargs,
    ) -> tuple[torch.Tensor, dict]:
        """
        Run backbone from embedding to insertion point layer K.

        Returns:
            h_K: Hidden states at layer K, shape [batch, seq_len, hidden_dim].
            state: Dict with intermediate state needed for forward_from_insertion.
        """
        if self._arch == "qwen":
            return self._qwen_forward_to(
                input_ids, attention_mask, pixel_values, image_grid_thw, **kwargs
            )

    def forward_from_insertion(
        self,
        h_K_prime: torch.Tensor,
        state: dict,
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Continue backbone from layer K+1 onwards using refined h_K'.

        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        if self._arch == "qwen":
            return self._qwen_forward_from(h_K_prime, state, attention_mask)

    def _qwen_forward_to(
        self,
        input_ids,
        attention_mask,
        pixel_values,
        image_grid_thw,
        **kwargs,
    ):
        """Run Qwen3.5 forward pass up to insertion layer."""
        model = self.model
        device = input_ids.device

        # Embedding
        inputs_embeds = model.model.embed_tokens(input_ids)

        # Vision encoding (if pixel_values provided)
        if pixel_values is not None and hasattr(model, "visual"):
            image_features = model.visual(pixel_values, grid_thw=image_grid_thw)
            # Merge image tokens into sequence (Qwen3.5-VL style)
            # This is handled by the model's own merge logic
            inputs_embeds = model._merge_input_ids_with_image_features(
                inputs_embeds, image_features, input_ids
            )

        # Build causal attention mask
        hidden_states = inputs_embeds
        position_ids = torch.arange(hidden_states.shape[1], device=device).unsqueeze(0)

        # Run layers 0 to K
        past_key_values = None
        for i, layer in enumerate(self._layers[: self.insert_layer + 1]):
            layer_out = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=False,
                output_attentions=False,
            )
            hidden_states = layer_out[0]

        state = {
            "position_ids": position_ids,
            "insert_layer": self.insert_layer,
        }
        return hidden_states, state

    def _qwen_forward_from(self, h_K_prime, state, attention_mask):
        """Continue Qwen3.5 from layer K+1 with refined hidden states."""
        position_ids = state["position_ids"]
        hidden_states = h_K_prime

        for i, layer in enumerate(self._layers[self.insert_layer + 1 :]):
            layer_out = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=None,
                use_cache=False,
                output_attentions=False,
            )
            hidden_states = layer_out[0]

        hidden_states = self._norm(hidden_states)
        logits = self._lm_head(hidden_states)
        return logits
