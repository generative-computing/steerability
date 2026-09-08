"""Architecture-specific module paths for reading and steering a model's decoder stack.

`ModelLayout` names the decoder layer prefix, attention/output-projection suffixes, and
normalization sub-module attributes for one model family. `resolve_model_layout` maps a
`PreTrainedModel` to its layout by locating the decoder stack at one of an ordered set of dotted
roots and probing the first decoder layer's attributes to select a naming convention.

Decoder-stack roots, in probe order:

- `model.layers`: text-only decoder language models (Llama, Mistral, Qwen, Gemma text).
- `model.language_model.layers`: composite multimodal wrappers (the transformers >= 5 layout,
  e.g. Gemma 3/4, Llava, Qwen-VL), where the toolkit steers the text decoder under text-only
  prompting.
- `transformer.h`: GPT-2-style models.

Per-layer naming conventions, in probe order (`gemma_style` precedes `llama_style` because a Gemma
decoder layer also carries the Llama markers):

- `gemma_style`: `self_attn`, residual-stream norms `input_layernorm` and `pre_feedforward_layernorm`.
- `llama_style`: `self_attn`, residual-stream norms `input_layernorm` and `post_attention_layernorm`.
- `gpt2_style`: `attn`, residual-stream norms `ln_1` and `ln_2`.

A convention matches when its norm markers exist on the first decoder layer and its attention
module exists on at least one decoder layer. Hybrid stacks that interleave attention layers with
another token mixer (Qwen3.5 and Qwen3-Next, where three Gated DeltaNet `linear_attn` layers
precede each `self_attn` layer) therefore resolve to the convention of their attention layers,
and the layout records which layers carry attention in `attention_layer_ids`. Residual-stream
sites (decoder-layer boundaries and the norm inputs) exist on every layer of such a stack;
attention sites (`attn_names`, `oproj_names`, `head_geometry`) exist only on the attention
layers, and consumers check `has_attention` before resolving them.

PEFT wrappers are peeled before resolution: `PeftModel.base_model` is the tuner and the tuner's
`model` attribute is the wrapped model, both registered submodules, so each wrapper contributes the
prefix `"base_model.model."` and the resulting paths resolve through `get_submodule` without relying
on attribute forwarding.

`register_layout_detector` adds a callable consulted before the built-in resolution, so a user on an
unlisted family is not blocked on a toolkit release.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import torch.nn as nn
from peft import PeftModel
from transformers import PretrainedConfig, PreTrainedModel

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelLayout:
    """Architecture-specific module paths for one model family.

    Attributes:
        family: Family label (`"gemma_style"`, `"llama_style"`, or `"gpt2_style"`).
        layer_prefix: Dotted prefix of the decoder layer list (e.g. `"model.layers"`,
            `"model.language_model.layers"`, `"transformer.h"`, prefixed with
            `"base_model.model."` per PEFT wrapper).
        num_layers: Number of decoder layers.
        attn_suffix: Suffix of the per-layer attention module (`".self_attn"` or `".attn"`).
        oproj_suffix: Suffix of the attention output projection (`".self_attn.o_proj"` or
            `".attn.c_proj"`).
        norm_attrs: The per-layer normalization sub-module attribute names whose input is the
            residual stream (`("input_layernorm", "pre_feedforward_layernorm")` on Gemma,
            `("input_layernorm", "post_attention_layernorm")` on Llama, `("ln_1", "ln_2")` on
            GPT-2).
        attention_layer_ids: The decoder layers carrying the attention module at `attn_suffix`,
            ascending. None means every layer (homogeneous stacks, and layouts built by hand);
            a hybrid stack lists only its attention layers, and an empty tuple declares that
            no layer carries attention.
    """

    family: str
    layer_prefix: str
    num_layers: int
    attn_suffix: str
    oproj_suffix: str
    norm_attrs: tuple[str, ...]
    attention_layer_ids: tuple[int, ...] | None = None

    @property
    def layer_names(self) -> list[str]:
        """Dotted paths of every decoder layer, `[f"{layer_prefix}.{i}"]`."""
        return [f"{self.layer_prefix}.{i}" for i in range(self.num_layers)]

    @property
    def oproj_names(self) -> list[str]:
        """Dotted paths of every layer's attention output projection, index-aligned with
        `layer_names`. On a hybrid stack, entries outside `attention_layers` name modules the
        model does not have."""
        return [name + self.oproj_suffix for name in self.layer_names]

    @property
    def attn_names(self) -> list[str]:
        """Dotted paths of every layer's attention module, index-aligned with `layer_names`. On a
        hybrid stack, entries outside `attention_layers` name modules the model does not have."""
        return [name + self.attn_suffix for name in self.layer_names]

    @property
    def attention_layers(self) -> tuple[int, ...]:
        """The decoder layers carrying an attention module, ascending; every layer when
        `attention_layer_ids` is None."""
        if self.attention_layer_ids is None:
            return tuple(range(self.num_layers))
        return self.attention_layer_ids

    @property
    def is_hybrid(self) -> bool:
        """True when some decoder layer carries no attention module."""
        return len(self.attention_layers) < self.num_layers

    def has_attention(self, layer_id: int) -> bool:
        """True when decoder layer `layer_id` carries the attention module at `attn_suffix`."""
        return layer_id in self.attention_layers


@dataclass(frozen=True)
class _Conventions:
    """Per-layer naming convention for one family, selected by probing the decoder stack.

    Attributes:
        family: The `ModelLayout.family` label.
        attn_suffix: The attention module suffix.
        oproj_suffix: The output-projection suffix.
        norm_attrs: The residual-stream normalization sub-module names.
        marker_attrs: The norm attributes that must all exist on the first decoder layer for this
            convention to match.
    """

    family: str
    attn_suffix: str
    oproj_suffix: str
    norm_attrs: tuple[str, ...]
    marker_attrs: tuple[str, ...]

    @property
    def attn_attr(self) -> str:
        """The attention module's attribute name on a decoder layer."""
        return self.attn_suffix.split(".")[1]

    def attention_layers(self, stack: nn.ModuleList) -> tuple[int, ...]:
        """The indices of the layers in `stack` carrying this convention's attention module."""
        return tuple(i for i, layer in enumerate(stack) if hasattr(layer, self.attn_attr))

    def matches(self, stack: nn.ModuleList) -> bool:
        """True when every marker attribute exists on the first layer and the attention module
        exists on at least one layer of `stack`."""
        if not all(hasattr(stack[0], attr) for attr in self.marker_attrs):
            return False
        return any(hasattr(layer, self.attn_attr) for layer in stack)


_DECODER_ROOTS: tuple[str, ...] = (
    "model.layers",
    "model.language_model.layers",
    "transformer.h",
)

_CONVENTIONS: tuple[_Conventions, ...] = (
    _Conventions(
        family="gemma_style",
        attn_suffix=".self_attn",
        oproj_suffix=".self_attn.o_proj",
        norm_attrs=("input_layernorm", "pre_feedforward_layernorm"),
        marker_attrs=("input_layernorm", "pre_feedforward_layernorm"),
    ),
    _Conventions(
        family="llama_style",
        attn_suffix=".self_attn",
        oproj_suffix=".self_attn.o_proj",
        norm_attrs=("input_layernorm", "post_attention_layernorm"),
        marker_attrs=("input_layernorm",),
    ),
    _Conventions(
        family="gpt2_style",
        attn_suffix=".attn",
        oproj_suffix=".attn.c_proj",
        norm_attrs=("ln_1", "ln_2"),
        marker_attrs=("ln_1",),
    ),
)


def text_config(model_or_config: PreTrainedModel | PretrainedConfig) -> PretrainedConfig:
    """The config carrying the text decoder's structural facts.

    Composite multimodal configs (text plus vision or audio sub-configs) return their text
    sub-config; plain configs return themselves. Numeric structural facts (`hidden_size`,
    `num_attention_heads`, `head_dim`, `num_hidden_layers`, `intermediate_size`) are read from the
    returned config; identity facts (`model_type`, the fingerprint JSON, `_attn_implementation`)
    stay on the composite `model.config`.

    Args:
        model_or_config: A model (its `config` is used) or a config.

    Returns:
        The text config.
    """
    config = getattr(model_or_config, "config", model_or_config)
    return config.get_text_config()


_DETECTORS: list[Callable[[PreTrainedModel], ModelLayout | None]] = []


def register_layout_detector(
    detector: Callable[[PreTrainedModel], ModelLayout | None], *, prepend: bool = True
) -> None:
    """Register a callable mapping a model to a `ModelLayout` (or None to decline).

    Registered detectors run before the built-in root and convention probing, in registration
    order (`prepend=True` puts the new detector first). Registration is process-global.

    Args:
        detector: A callable that returns a `ModelLayout` for models it recognizes, or None to
            fall through to the next detector and then the built-in resolution.
        prepend: When True (the default), the detector runs before previously registered ones.
    """
    if prepend:
        _DETECTORS.insert(0, detector)
    else:
        _DETECTORS.append(detector)


def _unwrap_peft(model) -> tuple[str, nn.Module]:
    """Peel PEFT wrappers off `model`.

    Returns the dotted prefix that reaches the innermost wrapped model from `model` (`""` when
    `model` is not wrapped) and that inner model. `PeftModel.base_model` is the tuner and the
    tuner's `model` attribute is the wrapped model, both registered submodules, so each wrapper
    contributes `"base_model.model."` and the resulting paths resolve through `get_submodule`
    without attribute forwarding.

    Args:
        model: A model, possibly wrapped in one or more `PeftModel` layers.

    Returns:
        A `(prefix, inner_model)` pair.
    """
    prefix = ""
    inner = model
    while isinstance(inner, PeftModel):
        prefix += "base_model.model."
        inner = inner.base_model.model
    return prefix, inner


def _find_decoder_stack(model) -> tuple[str, nn.ModuleList] | None:
    """The first decoder-stack root that resolves to a non-empty `nn.ModuleList`, or None.

    Args:
        model: The (unwrapped) model to probe.

    Returns:
        A `(root, stack)` pair, or None when no root resolves.
    """
    for root in _DECODER_ROOTS:
        try:
            stack = model.get_submodule(root)
        except AttributeError:
            continue
        if isinstance(stack, nn.ModuleList) and len(stack) > 0:
            return root, stack
    return None


def _resolve_builtin(model) -> ModelLayout | None:
    """The built-in `ModelLayout` for `model`, or None when no root or convention matches."""
    prefix, inner = _unwrap_peft(model)
    found = _find_decoder_stack(inner)
    if found is None:
        return None
    root, stack = found
    for conventions in _CONVENTIONS:
        if not conventions.matches(stack):
            continue
        config = getattr(inner, "config", None)
        expected = getattr(text_config(config), "num_hidden_layers", None) if config is not None else None
        if expected is not None and expected != len(stack):
            logger.warning(
                "Decoder stack at %r has %d layers but the text config declares %d.",
                root, len(stack), expected,
            )
        attention_layer_ids = conventions.attention_layers(stack)
        if len(attention_layer_ids) < len(stack):
            logger.info(
                "Decoder stack at %r is a hybrid: %d of %d layers carry %r; attention sites "
                "resolve on those layers only.",
                root, len(attention_layer_ids), len(stack), conventions.attn_attr,
            )
        return ModelLayout(
            family=conventions.family,
            layer_prefix=prefix + root,
            num_layers=len(stack),
            attn_suffix=conventions.attn_suffix,
            oproj_suffix=conventions.oproj_suffix,
            norm_attrs=conventions.norm_attrs,
            attention_layer_ids=attention_layer_ids,
        )
    return None


@dataclass(frozen=True)
class HeadGeometry:
    """Attention head geometry of one decoder layer.

    Attributes:
        num_heads: Number of attention heads.
        head_dim: Per-head dimension.
    """

    num_heads: int
    head_dim: int


def head_geometry(model: PreTrainedModel, layout: ModelLayout, layer_id: int) -> HeadGeometry:
    """Per-layer attention head geometry, read from the module tree.

    `head_dim` comes from the attention module's `head_dim` attribute, else the text config;
    `num_heads` is the output projection's input width divided by `head_dim`. GPT-2's `Conv1D`
    stores its weight as `[in, out]`, so the width is read from `weight.shape[0]` when
    `in_features` is absent.

    Args:
        model: The live model (or PEFT wrapper) carrying the attention modules.
        layout: The resolved `ModelLayout` naming the per-layer attention and output projections.
        layer_id: The decoder layer index.

    Returns:
        The layer's `HeadGeometry`.

    Raises:
        ValueError: If `layer_id` carries no attention module (a non-attention layer of a hybrid
            stack), or the projection width is not a multiple of `head_dim`.
    """
    if not layout.has_attention(layer_id):
        raise ValueError(
            f"Decoder layer {layer_id} carries no attention module ({layout.attn_suffix!r}); "
            f"attention layers of this model are {list(layout.attention_layers)}."
        )
    attn = model.get_submodule(layout.attn_names[layer_id])
    head_dim = getattr(attn, "head_dim", None)
    if head_dim is None:
        text_cfg = text_config(model)
        head_dim = getattr(text_cfg, "head_dim", None)
        if head_dim is None:
            head_dim = text_cfg.hidden_size // text_cfg.num_attention_heads

    oproj = model.get_submodule(layout.oproj_names[layer_id])
    width = getattr(oproj, "in_features", None)
    if width is None:
        width = oproj.weight.shape[0]  # gpt-2 Conv1D stores weight as [in, out]

    if width % head_dim != 0:
        raise ValueError(
            f"Output-projection input width {width} at layer {layer_id} is not a multiple of "
            f"head_dim {head_dim}; cannot infer the head count."
        )
    return HeadGeometry(num_heads=width // head_dim, head_dim=head_dim)


def resolve_model_layout(model: PreTrainedModel) -> ModelLayout:
    """Resolve the `ModelLayout` for a HuggingFace causal LM.

    Consults registered detectors first, then locates the decoder stack at one of the built-in
    roots and selects a naming convention whose norm markers exist on the first decoder layer and
    whose attention module exists on at least one layer. A hybrid stack that interleaves attention
    layers with another token mixer resolves to its attention layers' family, with those layers
    recorded in `attention_layer_ids`. PEFT wrappers are peeled before resolution and their prefix
    is carried on the resolved `layer_prefix`.

    Args:
        model: A HuggingFace causal LM, a multimodal wrapper whose text decoder sits at a known
            root, or a PEFT-wrapped model.

    Returns:
        The matching `ModelLayout`.

    Raises:
        ValueError: If no detector and no built-in root/convention matches the model.
    """
    for detector in _DETECTORS:
        layout = detector(model)
        if layout is not None:
            return layout
    layout = _resolve_builtin(model)
    if layout is not None:
        return layout
    raise ValueError(
        f"Cannot determine model layout for {type(model).__name__}. Probed decoder-stack roots "
        f"{_DECODER_ROOTS} for a stack matching one of the families "
        f"{tuple(convention.family for convention in _CONVENTIONS)} (the family's norm "
        "sub-modules on the first layer and its attention module on at least one layer); "
        "register a detector with register_layout_detector() for other architectures."
    )
