"""Structural facts of a loaded model.

`model_facts` builds the `ModelFacts` of a loaded model. Since the Hugging Face session's `layout`
and the state controls' `resolve_layout(model=...)` both return its result, the facts that a
control reads through a session agree with the facts derived from the model directly.
"""
from __future__ import annotations

from transformers import PreTrainedModel

from steerability.algorithms.core.execution.payloads import ModelFacts

from .fingerprint import model_fingerprint
from .model_layout import config_head_geometry, resolve_model_layout, text_config


def model_facts(model: PreTrainedModel) -> ModelFacts:
    """Build the structural facts of a loaded model.

    The facts are read from the following sources:

    - `num_layers`: the resolved decoder layer list.
    - `hidden_size`, `num_attention_heads`, and `head_dim`: the text config (`text_config(model)`,
      the text sub-config on composite multimodal models). A value that the config declares per
      layer is None, and `head_dim` falls back to `hidden_size // num_attention_heads` when the
      config declares no `head_dim`.
    - `dtype`: the model's dtype, as a string such as `"bfloat16"`.
    - `model_fingerprint`: the digest of the model's config, dtype, and sampled weights.
    - `model_type`: the composite config (a multimodal checkpoint keeps its wrapper `model_type`).
    - `model_ref`: the model's `name_or_path`.

    Args:
        model: A loaded causal LM (or PEFT wrapper).

    Returns:
        The model's `ModelFacts`.

    Raises:
        ValueError: If the decoder stack cannot be resolved.
        AttributeError: If the text config lacks `hidden_size`.
    """
    num_layers = resolve_model_layout(model).num_layers
    text_cfg = text_config(model)
    hidden_size = text_cfg.hidden_size
    num_heads, head_dim = config_head_geometry(text_cfg)
    return ModelFacts(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        head_dim=head_dim,
        dtype=str(model.dtype).removeprefix("torch."),
        model_fingerprint=model_fingerprint(model),
        model_type=getattr(model.config, "model_type", None),
        model_ref=getattr(model, "name_or_path", None),
    )
