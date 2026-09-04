"""Context handed to transforms (and transform factories) at steer() time."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from ..hook_utils import get_model_layer_list
from ..sources import ArtifactSource, _as_artifact_source
from ..steering_vector import SteeringVector
from .base import BaseTransform


@dataclass(frozen=True)
class TransformContext:
    """Fully-resolved layer/sizing context passed to a transform's `bind` or a factory.

    A transform slot may be a `BaseTransform` instance (bound, or source-carrying and bound via
    `bind(ctx)`) or a factory `Callable[[TransformContext], BaseTransform]`. Either way it receives
    this context after the behavior layers are resolved, so it can turn a source into a concrete
    artifact (`ctx.resolve(...)`) and size any per-layer or per-head buffers to the model.

    Transforms never touch the model directly: fit orchestration lives behind `resolve`, a closure
    the adapter builds over `(model, tokenizer, device, dtype)` that coerces the input, resolves the
    source (memoized master, cloned), and moves the clone onto the model's device/dtype.

    Attributes:
        layer_ids: The resolved behavior layer(s).
        num_layers: Total number of transformer layers in the model.
        hidden_size: Residual-stream width `H`.
        num_heads: Number of attention heads per layer, or None if unavailable from the config.
        head_dim: Dimension of each attention head, or None if unavailable from the config.
        dtype: The model's parameter dtype.
        device: The model's device.
        resolve: Turns a `SteeringVector` / directions mapping / `ArtifactSource` into a concrete
            `SteeringVector` on the model's device and dtype.
    """

    layer_ids: list[int]
    num_layers: int
    hidden_size: int
    num_heads: int | None
    head_dim: int | None
    dtype: torch.dtype
    device: torch.device
    resolve: Callable[[ArtifactSource | SteeringVector | Mapping[int, torch.Tensor]], SteeringVector]


def _build_context(
    model: PreTrainedModel | None,
    tokenizer: PreTrainedTokenizerBase | None,
    layer_ids: Sequence[int],
    layout=None,
    session=None,
) -> TransformContext:
    """Build the `TransformContext` for the given behavior layers.

    With a live model, reads device/dtype/layer-count from the model and
    `hidden_size`/`num_heads`/`head_dim` from its config (deriving `head_dim` as
    `hidden_size // num_heads` when absent), then wraps a resolve closure that coerces any
    artifact to a source, resolves it against the model, and moves the result onto the model's
    device and dtype. With `model=None`, sizes come from `layout` (a structural
    `core.execution.ModelFacts`), the device is CPU, and the resolve closure serves concrete
    artifacts only, since fitting a source requires a live model.
    """
    if model is not None:
        device = next(model.parameters()).device
        dtype = model.dtype
        _, layer_names = get_model_layer_list(model)
        num_layers = len(layer_names)

        config = model.config
        hidden_size = getattr(config, "hidden_size")
        num_heads = getattr(config, "num_attention_heads", None)
        head_dim = getattr(config, "head_dim", None)
        if head_dim is None and num_heads:
            head_dim = hidden_size // num_heads
    else:
        if layout is None:
            raise ValueError("Building a TransformContext requires a live model or a structural layout.")
        device = torch.device("cpu")
        dtype = getattr(torch, layout.dtype, None)
        if not isinstance(dtype, torch.dtype):
            raise ValueError(f"Layout dtype {layout.dtype!r} does not name a torch dtype.")
        num_layers = layout.num_layers
        hidden_size = layout.hidden_size
        num_heads = layout.num_attention_heads
        head_dim = layout.head_dim

    def resolve(artifact) -> SteeringVector:
        source = _as_artifact_source(artifact)
        try:
            resolved = source.resolve(model, tokenizer, session=session)
        except TypeError:
            resolved = source.resolve(model, tokenizer)  # sources without capture support
        return resolved.to(device, dtype)

    return TransformContext(
        layer_ids=list(layer_ids),
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_heads=num_heads,
        head_dim=head_dim,
        dtype=dtype,
        device=device,
        resolve=resolve,
    )


def resolve_transform_slot(
    slot: BaseTransform | Callable[[TransformContext], BaseTransform],
    model: PreTrainedModel | None,
    tokenizer: PreTrainedTokenizerBase | None,
    layer_ids: Sequence[int],
    layout=None,
    require_coverage: bool = True,
    session=None,
) -> BaseTransform:
    """Turn a transform slot into a bound, coverage-checked `BaseTransform` for the given model.

    Shared by every control that exposes a transform slot (`ActivationAdapter.transform`, `CAST`'s
    `behavior_transform`), so the resolution contract lives in one place. It runs three stages:

    1. **Context construction**: introspect the model (device, dtype, layer count, hidden size,
       head count/dim) and build a `TransformContext` whose `resolve` closure fits/resolves a source
       on the steered model and moves the result onto its device/dtype (`_build_context`).
    2. **Slot resolution**: a `BaseTransform` instance is bound if unbound (else used as-is); a
       factory is invoked with the context and its result is bound if unbound. Either branch is
       validated to yield a bound `BaseTransform`, so a factory returning a source-carrying transform
       is bound here rather than failing later in `apply`.
    3. **Coverage check**: when the resolved transform reports `covered_layer_ids`, every layer in
       `layer_ids` must be covered; a transform reporting `None` opts out of the check.

    Args:
        slot: A `BaseTransform` (bound or source-carrying) or a factory taking the context.
        model: The steered model to introspect and resolve sources against, or None to build the
            context from `layout` (concrete artifacts only; fitting a source requires a model).
        tokenizer: Tokenizer used when a source fits from data; may be None for concrete artifacts.
        layer_ids: The resolved behavior layers the transform must cover.
        layout: Structural `core.execution.ModelFacts` consulted when `model` is None.
        require_coverage: When False, skip the coverage check; uncovered layers are hooked and
            pass through unchanged at apply time.
        session: Optional `SteeringSession` forwarded to sources whose fit runs through
            session capture.

    Returns:
        A bound `BaseTransform` ready for `apply`.

    Raises:
        TypeError: If a factory returns a non-`BaseTransform`, or a `bind` returns an unbound
            transform.
        ValueError: If the transform covers only some of `layer_ids`.
    """
    ctx = _build_context(model, tokenizer, layer_ids, layout=layout, session=session)

    if isinstance(slot, BaseTransform):
        built = slot if slot.is_bound else slot.bind(ctx)
    else:  # factory
        built = slot(ctx)
        if not isinstance(built, BaseTransform):
            raise TypeError(f"transform factory must return a BaseTransform; got {type(built).__name__}.")
        if not built.is_bound:
            built = built.bind(ctx)

    if not isinstance(built, BaseTransform) or not built.is_bound:
        raise TypeError(
            f"{type(slot).__name__}.bind() must return a bound BaseTransform; got "
            f"{type(built).__name__} (is_bound={getattr(built, 'is_bound', None)!r})."
        )

    coverage = built.covered_layer_ids
    if require_coverage and coverage is not None:
        missing = [lid for lid in layer_ids if lid not in coverage]
        if missing:
            raise ValueError(
                f"Transform has no direction for layer(s) {missing} (covers {sorted(coverage)})."
            )

    return built
