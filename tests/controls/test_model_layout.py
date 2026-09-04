"""Tests for `ModelLayout` resolution and the `hook_utils` delegation wrappers.

Pins the single-source-of-truth layout registry (llama-style vs gpt2-style), the unified
unsupported-architecture error, and that the two `hook_utils` wrappers still produce the same
output they did before delegating (so the pure consumers cannot silently change behavior).
"""
import pytest
import torch
import torch.nn as nn

from aisteer360.algorithms.state_control.common.hook_utils import (
    extract_hidden_states,
    get_model_layer_list,
    get_norm_module_names,
)
from aisteer360.algorithms.state_control.common.model_layout import resolve_model_layout
from tests.utils.tiny_models import tiny_gpt2, tiny_llama

LAYERS = 4
HIDDEN = 32
HEADS = 4


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

    with pytest.raises(ValueError, match="Cannot determine model layout"):
        resolve_model_layout(Bare())


def test_resolved_module_paths_exist_on_both_families():
    for factory in (tiny_llama, tiny_gpt2):
        model = factory(num_layers=LAYERS, hidden=HIDDEN, heads=HEADS)
        layout = resolve_model_layout(model)
        for name in layout.layer_names + layout.oproj_names + layout.attn_names:
            model.get_submodule(name)  # raises if the path is wrong


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
