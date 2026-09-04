"""Architecture-specific module paths for state controls.

`ModelLayout` names the decoder layer prefix, attention/output-projection suffixes, and
normalization sub-module attributes for one model family. `resolve_model_layout` maps a
`PreTrainedModel` to its layout by trying an ordered registry of family detectors.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from transformers import PreTrainedModel


@dataclass(frozen=True)
class ModelLayout:
    """Architecture-specific module paths for one model family.

    Attributes:
        family: Family label (`"llama_style"` or `"gpt2_style"`).
        layer_prefix: Dotted prefix of the decoder layer list (`"model.layers"` or
            `"transformer.h"`).
        num_layers: Number of decoder layers.
        attn_suffix: Suffix of the per-layer attention module (`".self_attn"` or `".attn"`).
        oproj_suffix: Suffix of the attention output projection (`".self_attn.o_proj"` or
            `".attn.c_proj"`).
        norm_attrs: Per-layer normalization sub-module attribute names
            (`("input_layernorm", "post_attention_layernorm")` or `("ln_1", "ln_2")`).
    """

    family: str
    layer_prefix: str
    num_layers: int
    attn_suffix: str
    oproj_suffix: str
    norm_attrs: tuple[str, ...]

    @property
    def layer_names(self) -> list[str]:
        """Dotted paths of every decoder layer, `[f"{layer_prefix}.{i}"]`."""
        return [f"{self.layer_prefix}.{i}" for i in range(self.num_layers)]

    @property
    def oproj_names(self) -> list[str]:
        """Dotted paths of every layer's attention output projection."""
        return [name + self.oproj_suffix for name in self.layer_names]

    @property
    def attn_names(self) -> list[str]:
        """Dotted paths of every layer's attention module."""
        return [name + self.attn_suffix for name in self.layer_names]


def _detect_llama_style(model: PreTrainedModel) -> ModelLayout | None:
    """Llama/Mistral/Qwen/Gemma-style: `model.model.layers[i]`."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return ModelLayout(
            family="llama_style",
            layer_prefix="model.layers",
            num_layers=len(model.model.layers),
            attn_suffix=".self_attn",
            oproj_suffix=".self_attn.o_proj",
            norm_attrs=("input_layernorm", "post_attention_layernorm"),
        )
    return None


def _detect_gpt2_style(model: PreTrainedModel) -> ModelLayout | None:
    """GPT-2-style: `model.transformer.h[i]`."""
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return ModelLayout(
            family="gpt2_style",
            layer_prefix="transformer.h",
            num_layers=len(model.transformer.h),
            attn_suffix=".attn",
            oproj_suffix=".attn.c_proj",
            norm_attrs=("ln_1", "ln_2"),
        )
    return None


_DETECTORS: list[Callable[[PreTrainedModel], ModelLayout | None]] = [
    _detect_llama_style,
    _detect_gpt2_style,
]


def resolve_model_layout(model: PreTrainedModel) -> ModelLayout:
    """Resolve the `ModelLayout` for a HuggingFace causal LM.

    Tries each registered family detector in order and returns the first match.

    Args:
        model: A HuggingFace causal LM.

    Returns:
        The matching `ModelLayout`.

    Raises:
        ValueError: If no registered family matches the model.
    """
    for detector in _DETECTORS:
        layout = detector(model)
        if layout is not None:
            return layout
    raise ValueError(
        f"Cannot determine model layout for {type(model).__name__}. Supported families: "
        f"llama-style (model.model.layers) and GPT-2-style (model.transformer.h)."
    )
