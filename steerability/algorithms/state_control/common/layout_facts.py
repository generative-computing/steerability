"""Structural model facts for steer-time preparation.

State controls consume structural facts (layer count, dtype, hidden size) from the steering
session's `ModelFacts` so preparation works the same whether the steering backend holds a live
model or only a layout. Module-path resolution stays out of this module; hook module names are
resolved from the module tree at `get_hooks()` time.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from steerability.algorithms.core.execution.payloads import ModelFacts
from steerability.algorithms.core.internals.fingerprint import model_fingerprint
from steerability.algorithms.core.internals.model_layout import text_config

from .hook_utils import get_model_layer_list

if TYPE_CHECKING:
    from transformers import PreTrainedModel

    from steerability.algorithms.core.execution.backend import SteeringSession

    from .steering_vector import SteeringVector


def resolve_layout(model: PreTrainedModel | None = None, session: SteeringSession | None = None) -> ModelFacts:
    """Structural facts from the session's layout, else derived from the live model.

    Args:
        model: A live model, consulted only when `session` is None.
        session: A `SteeringSession` whose `layout` property carries the facts.

    Returns:
        The structural `ModelFacts`.

    Raises:
        ValueError: If neither a session nor a model is available.
    """
    if session is not None:
        return session.layout
    if model is None:
        raise ValueError(
            "Structural facts require a steering session (session.layout) or a live model; "
            "vector-supplied configurations may steer with model=None only when a session is given."
        )
    _, layer_names = get_model_layer_list(model)
    text_cfg = text_config(model)
    hidden_size = text_cfg.hidden_size
    num_heads = getattr(text_cfg, "num_attention_heads", None)
    head_dim = getattr(text_cfg, "head_dim", None)
    if head_dim is None and num_heads:
        head_dim = hidden_size // num_heads
    return ModelFacts(
        num_layers=len(layer_names),
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        head_dim=head_dim,
        dtype=str(model.dtype).removeprefix("torch."),
        model_fingerprint=model_fingerprint(model),
        model_type=getattr(model.config, "model_type", None),
        model_ref=getattr(model, "name_or_path", None),
    )


def cast_steering_vector(steering_vector: SteeringVector, layout: ModelFacts) -> SteeringVector:
    """A clone of `steering_vector` with per-layer directions cast to the layout dtype.

    Device placement is untouched; transforms move tensors to the stream device at apply time.

    Args:
        steering_vector: The `SteeringVector` to clone and cast.
        layout: The structural layout naming the target dtype.

    Returns:
        The cast clone.
    """
    clone = steering_vector.clone()
    dtype = layout_torch_dtype(layout)
    for layer_id, direction in clone.directions.items():
        clone.directions[layer_id] = direction.to(dtype=dtype)
    return clone


def layout_torch_dtype(layout: ModelFacts) -> torch.dtype:
    """The torch dtype named by `layout.dtype`.

    Raises:
        ValueError: If `layout.dtype` does not name a torch dtype.
    """
    dtype = getattr(torch, layout.dtype, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Layout dtype {layout.dtype!r} does not name a torch dtype.")
    return dtype
