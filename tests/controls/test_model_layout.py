"""Tests for `ModelLayout` resolution and the `hook_utils` delegation wrappers.

Pins the single-source-of-truth layout registry (roots times conventions across gemma/llama/gpt2),
resolution through composite multimodal wrappers and unmerged PEFT adapters, hybrid attention
stacks that resolve to their attention layers' family (Qwen3.5 / Qwen3-Next style, where only
some decoder layers carry an attention module), the detector registration hook, the `text_config`
fact-derivation helper, the config head-geometry readers on configs that declare `head_dim` per
layer (Gemma 4), the unified unsupported-architecture error, and that the two `hook_utils`
wrappers still produce the same output they did before delegating (so the pure consumers cannot
silently change behavior).
"""
import pytest
import torch
import torch.nn as nn

from steerability.algorithms.core.internals import model_layout as layout_mod
from steerability.algorithms.core.internals.model_layout import (
    ModelLayout,
    config_head_geometry,
    config_layer_head_dim,
    head_geometry,
    register_layout_detector,
    resolve_model_layout,
    text_config,
)
from steerability.algorithms.state_control.common.hook_utils import (
    extract_hidden_states,
    get_model_layer_list,
    get_norm_module_names,
)
from tests.utils.tiny_models import (
    gemma4_declares_per_layer_head_dim,
    hybrid_attention_stub,
    tiny_gemma3_conditional,
    tiny_gemma4_causal,
    tiny_gemma4_conditional,
    tiny_gemma4_text_config,
    tiny_gpt2,
    tiny_llama,
    tiny_lora,
)

LAYERS = 4
HIDDEN = 32
HEADS = 4


@pytest.fixture
def restore_detectors():
    """Snapshot and restore the process-global detector registry around a test."""
    snapshot = list(layout_mod._DETECTORS)
    try:
        yield
    finally:
        layout_mod._DETECTORS.clear()
        layout_mod._DETECTORS.extend(snapshot)


def test_llama_layout_fields():
    layout = resolve_model_layout(tiny_llama(num_layers=LAYERS, hidden=HIDDEN, heads=HEADS))
    assert layout.family == "llama_style"
    assert layout.layer_prefix == "model.layers"
    assert layout.num_layers == LAYERS
    assert layout.attn_suffix == ".self_attn"
    assert layout.oproj_suffix == ".self_attn.o_proj"
    assert layout.norm_attrs == ("input_layernorm", "post_attention_layernorm")
    assert layout.layer_names == [f"model.layers.{i}" for i in range(LAYERS)]
    assert layout.oproj_names == [f"model.layers.{i}.self_attn.o_proj" for i in range(LAYERS)]
    assert layout.attn_names == [f"model.layers.{i}.self_attn" for i in range(LAYERS)]


def test_gpt2_layout_fields():
    layout = resolve_model_layout(tiny_gpt2(num_layers=LAYERS, hidden=HIDDEN, heads=HEADS))
    assert layout.family == "gpt2_style"
    assert layout.layer_prefix == "transformer.h"
    assert layout.num_layers == LAYERS
    assert layout.attn_suffix == ".attn"
    assert layout.oproj_suffix == ".attn.c_proj"
    assert layout.norm_attrs == ("ln_1", "ln_2")
    assert layout.layer_names == [f"transformer.h.{i}" for i in range(LAYERS)]
    assert layout.oproj_names == [f"transformer.h.{i}.attn.c_proj" for i in range(LAYERS)]
    assert layout.attn_names == [f"transformer.h.{i}.attn" for i in range(LAYERS)]


def test_unsupported_architecture_raises():
    class Bare(nn.Module):
        pass

    with pytest.raises(ValueError, match="Cannot determine model layout") as excinfo:
        resolve_model_layout(Bare())
    assert "register_layout_detector" in str(excinfo.value)


def test_gemma3_conditional_nested_root():
    """A composite multimodal wrapper resolves to its text decoder at the nested root."""
    model = tiny_gemma3_conditional(num_layers=LAYERS, hidden=HIDDEN, heads=HEADS)
    layout = resolve_model_layout(model)
    assert layout.family == "gemma_style"
    assert layout.layer_prefix == "model.language_model.layers"
    assert layout.num_layers == LAYERS
    assert layout.attn_suffix == ".self_attn"
    assert layout.oproj_suffix == ".self_attn.o_proj"
    assert layout.norm_attrs == ("input_layernorm", "pre_feedforward_layernorm")
    for name in layout.layer_names + layout.oproj_names + layout.attn_names:
        model.get_submodule(name)  # raises if the path is wrong


def test_gemma3_conditional_norm_module_names():
    """`get_norm_module_names` lists exactly the two residual-stream norms per layer."""
    model = tiny_gemma3_conditional(num_layers=LAYERS, hidden=HIDDEN, heads=HEADS)
    norms = get_norm_module_names(model)
    expected = sorted(
        (i, f"model.language_model.layers.{i}.{attr}")
        for i in range(LAYERS)
        for attr in ("input_layernorm", "pre_feedforward_layernorm")
    )
    assert norms == expected


def test_text_config_identity_and_subconfig():
    """`text_config` is identity on a plain config and the text sub-config on a composite one."""
    llama = tiny_llama(num_layers=LAYERS, hidden=HIDDEN, heads=HEADS)
    assert text_config(llama) is llama.config
    assert text_config(llama.config) is llama.config

    gemma = tiny_gemma3_conditional(num_layers=LAYERS, hidden=HIDDEN, heads=HEADS)
    sub = text_config(gemma)
    assert sub is not gemma.config
    assert sub.hidden_size == HIDDEN
    assert not hasattr(gemma.config, "hidden_size")


def test_detector_precedence(restore_detectors):
    """A registered detector's non-None result takes precedence over the built-in resolution."""
    fixed = ModelLayout(
        family="custom", layer_prefix="model.layers", num_layers=1, attn_suffix=".self_attn",
        oproj_suffix=".self_attn.o_proj", norm_attrs=("input_layernorm",),
    )
    register_layout_detector(lambda _model: fixed)
    assert resolve_model_layout(tiny_llama()).family == "custom"


def test_detector_none_falls_through(restore_detectors):
    """A detector returning None falls through to the built-in resolution."""
    register_layout_detector(lambda _model: None)
    assert resolve_model_layout(tiny_llama()).family == "llama_style"


def test_peft_layer_prefix_and_shared_modules():
    """An unmerged LoRA wrapper resolves through `base_model.model.` to the inner llama layers."""
    inner = tiny_llama(num_layers=LAYERS, hidden=HIDDEN, heads=HEADS)
    wrapped = tiny_lora(inner)
    layout = resolve_model_layout(wrapped)
    assert layout.family == "llama_style"
    assert layout.layer_prefix == "base_model.model.model.layers"
    assert layout.num_layers == LAYERS

    inner_layout = resolve_model_layout(inner)
    for outer_name, inner_name in zip(
        layout.layer_names + layout.oproj_names + layout.attn_names,
        inner_layout.layer_names + inner_layout.oproj_names + inner_layout.attn_names,
    ):
        assert wrapped.get_submodule(outer_name) is inner.get_submodule(inner_name)


def test_peft_over_gemma3_conditional():
    """A LoRA-wrapped composite wrapper accumulates the PEFT prefix onto the nested root."""
    wrapped = tiny_lora(tiny_gemma3_conditional(num_layers=LAYERS, hidden=HIDDEN, heads=HEADS))
    layout = resolve_model_layout(wrapped)
    assert layout.family == "gemma_style"
    assert layout.layer_prefix == "base_model.model.model.language_model.layers"
    for name in layout.layer_names + layout.oproj_names + layout.attn_names:
        wrapped.get_submodule(name)


def test_double_wrapped_peft_accumulates_prefix():
    """PEFT over PEFT accumulates the `base_model.model.` prefix twice."""
    doubly = tiny_lora(tiny_lora(tiny_llama(num_layers=LAYERS, hidden=HIDDEN, heads=HEADS)))
    layout = resolve_model_layout(doubly)
    assert layout.layer_prefix == "base_model.model.base_model.model.model.layers"
    for name in layout.layer_names:
        doubly.get_submodule(name)


def test_resolved_module_paths_exist_on_both_families():
    for factory in (tiny_llama, tiny_gpt2):
        model = factory(num_layers=LAYERS, hidden=HIDDEN, heads=HEADS)
        layout = resolve_model_layout(model)
        for name in layout.layer_names + layout.oproj_names + layout.attn_names:
            model.get_submodule(name)  # raises if the path is wrong


# hybrid attention stacks


def test_homogeneous_layout_lists_every_layer_as_attention():
    """A homogeneous stack reports every layer as an attention layer and is not hybrid."""
    layout = resolve_model_layout(tiny_llama(num_layers=LAYERS, hidden=HIDDEN, heads=HEADS))
    assert layout.attention_layer_ids == tuple(range(LAYERS))
    assert layout.attention_layers == tuple(range(LAYERS))
    assert not layout.is_hybrid
    assert all(layout.has_attention(lid) for lid in range(LAYERS))


def test_hybrid_stack_resolves_to_its_attention_layers_family():
    """A stack whose first layer carries only linear attention resolves to `llama_style`, with
    the full-attention layers recorded and the attention sites resolving only there."""
    stub = hybrid_attention_stub(num_layers=8, hidden=HIDDEN, heads=HEADS)
    layout = resolve_model_layout(stub)
    assert layout.family == "llama_style"
    assert layout.layer_prefix == "model.layers"
    assert layout.num_layers == 8
    assert layout.norm_attrs == ("input_layernorm", "post_attention_layernorm")
    assert layout.attention_layer_ids == (3, 7)
    assert layout.is_hybrid
    assert layout.has_attention(3)
    assert not layout.has_attention(0)

    for name in layout.layer_names:
        stub.get_submodule(name)  # every residual-stream boundary exists
    for layer_id in (3, 7):
        stub.get_submodule(layout.attn_names[layer_id])
        stub.get_submodule(layout.oproj_names[layer_id])
    with pytest.raises(AttributeError):
        stub.get_submodule(layout.attn_names[0])  # linear-attention layer has no self_attn


def test_hybrid_stack_norm_module_names_cover_every_layer():
    """`get_norm_module_names` lists both residual-stream norms on every layer of a hybrid stack."""
    stub = hybrid_attention_stub(num_layers=8, hidden=HIDDEN, heads=HEADS)
    norms = get_norm_module_names(stub)
    expected = sorted(
        (i, f"model.layers.{i}.{attr}")
        for i in range(8)
        for attr in ("input_layernorm", "post_attention_layernorm")
    )
    assert norms == expected


def test_hybrid_head_geometry_reads_attention_layers_and_rejects_others():
    """Head geometry reads a full-attention layer and refuses a linear-attention layer with a
    message naming the attention layers."""
    stub = hybrid_attention_stub(num_layers=8, hidden=HIDDEN, heads=HEADS)
    layout = resolve_model_layout(stub)
    assert (head_geometry(stub, layout, 3).num_heads, head_geometry(stub, layout, 3).head_dim) == (
        HEADS,
        HIDDEN // HEADS,
    )
    with pytest.raises(ValueError, match="carries no attention module") as excinfo:
        head_geometry(stub, layout, 0)
    assert "[3" in str(excinfo.value)


def test_stack_without_any_attention_layer_is_unsupported():
    """A stack with no attention module on any layer matches no convention."""
    stub = hybrid_attention_stub(num_layers=3, hidden=HIDDEN, heads=HEADS, full_attention_interval=4)
    with pytest.raises(ValueError, match="Cannot determine model layout"):
        resolve_model_layout(stub)


def test_hand_built_layout_defaults_to_attention_on_every_layer():
    """A `ModelLayout` built without `attention_layer_ids` treats every layer as attention."""
    layout = ModelLayout(
        family="llama_style", layer_prefix="model.layers", num_layers=3, attn_suffix=".self_attn",
        oproj_suffix=".self_attn.o_proj", norm_attrs=("input_layernorm", "post_attention_layernorm"),
    )
    assert layout.attention_layer_ids is None
    assert layout.attention_layers == (0, 1, 2)
    assert not layout.is_hybrid


def test_hook_utils_wrapper_fidelity_llama():
    """`get_model_layer_list` / `get_norm_module_names` outputs are unchanged on Llama."""
    model = tiny_llama(num_layers=LAYERS, hidden=HIDDEN, heads=HEADS)

    modules, names = get_model_layer_list(model)
    assert names == [f"model.layers.{i}" for i in range(LAYERS)]
    assert modules == list(model.model.layers)

    norms = get_norm_module_names(model)
    expected = sorted(
        [(i, f"model.layers.{i}.{attr}")
         for i in range(LAYERS)
         for attr in ("input_layernorm", "post_attention_layernorm")]
    )
    assert norms == expected


def test_hook_utils_wrappers_gpt2():
    model = tiny_gpt2(num_layers=LAYERS, hidden=HIDDEN, heads=HEADS)

    modules, names = get_model_layer_list(model)
    assert names == [f"transformer.h.{i}" for i in range(LAYERS)]
    assert modules == list(model.transformer.h)

    norms = get_norm_module_names(model)
    expected = sorted(
        [(i, f"transformer.h.{i}.{attr}") for i in range(LAYERS) for attr in ("ln_1", "ln_2")]
    )
    assert norms == expected


@pytest.mark.parametrize("suffix", ["attn.c_proj", "ln_1"])
def test_gpt2_prehook_extract_hidden_states(suffix):
    """Hidden state arrives as the first positional arg on GPT-2 c_proj and ln_1 pre-hooks."""
    model = tiny_gpt2(num_layers=LAYERS, hidden=HIDDEN, heads=HEADS)
    seen: list = []

    module = model.get_submodule(f"transformer.h.0.{suffix}")

    def pre_hook(_module, args, kwargs):
        seen.append(extract_hidden_states(args, kwargs))
        return None

    handle = module.register_forward_pre_hook(pre_hook, with_kwargs=True)
    try:
        input_ids = torch.arange(1, 6, dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            model(input_ids=input_ids)
    finally:
        handle.remove()

    assert seen, f"pre-hook on {suffix} never fired"
    hidden = seen[0]
    assert isinstance(hidden, torch.Tensor)
    assert hidden.ndim == 3 and hidden.size(-1) == HIDDEN


# per-layer head geometry

requires_per_layer_head_dim = pytest.mark.skipif(
    not gemma4_declares_per_layer_head_dim(),
    reason="the installed transformers does not declare Gemma 4's head_dim per layer",
)


def _attention_stub_without_head_dim(config, oproj_widths):
    """A llama-layout stub whose attention modules have no `head_dim` attribute.

    Layer `i` has an `o_proj` with input width `oproj_widths[i]`, so `head_geometry` reads the
    layer's `head_dim` from `config`.
    """

    class _Attn(nn.Module):
        def __init__(self, width):
            super().__init__()
            self.o_proj = nn.Linear(width, HIDDEN, bias=False)

    class _Layer(nn.Module):
        def __init__(self, width):
            super().__init__()
            self.self_attn = _Attn(width)
            self.input_layernorm = nn.Identity()
            self.post_attention_layernorm = nn.Identity()

    class _Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList(_Layer(width) for width in oproj_widths)

    class _Stub(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _Inner()
            self.config = config

    return _Stub()


class TestConfigHeadGeometry:
    """`config_head_geometry` and `config_layer_head_dim` on homogeneous and heterogeneous configs."""

    def test_declared_head_dim(self):
        assert config_head_geometry(text_config(tiny_llama(hidden=HIDDEN, heads=HEADS))) == (HEADS, HIDDEN // HEADS)

    def test_attribute_map_without_head_dim_falls_back_to_quotient(self):
        cfg = text_config(tiny_gpt2(hidden=HIDDEN, heads=HEADS))
        assert config_head_geometry(cfg) == (HEADS, HIDDEN // HEADS)
        assert config_layer_head_dim(cfg, 0) == HIDDEN // HEADS

    def test_composite_reads_text_subconfig(self):
        cfg = text_config(tiny_gemma3_conditional(hidden=HIDDEN, heads=HEADS))
        assert config_head_geometry(cfg) == (HEADS, HIDDEN // HEADS)

    def test_config_without_heads_or_head_dim_reports_none(self):
        from transformers import PretrainedConfig

        assert config_head_geometry(PretrainedConfig(hidden_size=HIDDEN)) == (None, None)
        assert config_layer_head_dim(PretrainedConfig(hidden_size=HIDDEN), 0) is None

    @requires_per_layer_head_dim
    def test_per_layer_head_dim_reports_none_globally(self):
        cfg = tiny_gemma4_text_config(num_layers=6, hidden=HIDDEN, heads=HEADS, head_dim=8, global_head_dim=16)
        with pytest.raises(RuntimeError):
            _ = getattr(cfg, "head_dim", None)  # a global read of a per-layer attribute raises despite the default
        assert config_head_geometry(cfg) == (HEADS, None)

    @requires_per_layer_head_dim
    def test_per_layer_head_dim_reads_each_layer(self):
        cfg = tiny_gemma4_text_config(num_layers=6, hidden=HIDDEN, heads=HEADS, head_dim=8, global_head_dim=16)
        per_layer = [config_layer_head_dim(cfg, i) for i in range(6)]
        assert per_layer == [16 if layer_type == "full_attention" else 8 for layer_type in cfg.layer_types]
        assert per_layer == [8, 8, 8, 8, 8, 16]

    @requires_per_layer_head_dim
    def test_equal_global_and_sliding_head_dim_is_one_value(self):
        cfg = tiny_gemma4_text_config(num_layers=6, hidden=HIDDEN, heads=HEADS, head_dim=8, global_head_dim=8)
        assert config_head_geometry(cfg) == (HEADS, 8)


@requires_per_layer_head_dim
def test_gemma4_head_geometry_reads_each_layer_from_the_module():
    model = tiny_gemma4_causal(num_layers=6, hidden=HIDDEN, heads=HEADS, head_dim=8, global_head_dim=16)
    layout = resolve_model_layout(model)
    assert layout.family == "gemma_style"
    assert layout.num_layers == 6
    geometries = [head_geometry(model, layout, i) for i in range(6)]
    assert [(g.num_heads, g.head_dim) for g in geometries] == [(HEADS, 8)] * 5 + [(HEADS, 16)]


@requires_per_layer_head_dim
def test_head_geometry_config_fallback_reads_the_layer_value():
    """An attention module without `head_dim` falls back to the config's value for its layer."""
    cfg = tiny_gemma4_text_config(num_layers=6, hidden=HIDDEN, heads=HEADS, head_dim=8, global_head_dim=16)
    head_dims = [8, 8, 8, 8, 8, 16]
    stub = _attention_stub_without_head_dim(cfg, [HEADS * head_dim for head_dim in head_dims])
    layout = resolve_model_layout(stub)
    geometries = [head_geometry(stub, layout, i) for i in range(6)]
    assert [g.head_dim for g in geometries] == head_dims
    assert all(g.num_heads == HEADS for g in geometries)


def test_head_geometry_without_any_head_dim_raises():
    """Head geometry raises when neither the attention module nor the config declares `head_dim`."""
    from transformers import PretrainedConfig

    stub = _attention_stub_without_head_dim(PretrainedConfig(hidden_size=HIDDEN, num_hidden_layers=2), [HIDDEN] * 2)
    with pytest.raises(ValueError, match="per-head dimension"):
        head_geometry(stub, resolve_model_layout(stub), 0)


@requires_per_layer_head_dim
def test_gemma4_conditional_nested_root_and_per_layer_geometry():
    model = tiny_gemma4_conditional(num_layers=6, hidden=HIDDEN, heads=HEADS, head_dim=8, global_head_dim=16)
    layout = resolve_model_layout(model)
    assert layout.family == "gemma_style"
    assert layout.layer_prefix == "model.language_model.layers"
    assert [head_geometry(model, layout, i).head_dim for i in range(6)] == [8, 8, 8, 8, 8, 16]
    assert config_head_geometry(text_config(model)) == (HEADS, None)
