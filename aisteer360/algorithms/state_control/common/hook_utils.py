"""Utilities for hook registration and model inspection."""
import torch
from transformers import PreTrainedModel

from .model_layout import resolve_model_layout


def get_model_layer_list(model: PreTrainedModel) -> tuple[list, list[str]]:
    """Return (layer_modules, layer_name_strings) for a HuggingFace model.

    Supports llama/mistral/gemma-style (model.model.layers) and
    GPT2-style (model.transformer.h) architectures.

    Args:
        model: A HuggingFace causal LM.

    Returns:
        Tuple of (list of nn.Module layers, list of dotted name strings).

    Raises:
        ValueError: If model architecture is not recognized.
    """
    layout = resolve_model_layout(model)
    names = layout.layer_names
    modules = [model.get_submodule(name) for name in names]
    return modules, names


def get_norm_module_names(model: PreTrainedModel) -> list[tuple[int, str]]:
    """Return (layer_id, module_path) pairs for the per-layer normalization sub-modules.

    Supports:

    - llama/mistral/qwen/gemma-style (`model.model.layers[i]`): `input_layernorm`,
        `post_attention_layernorm`.
    - GPT-2-style (`model.transformer.h[i]`): `ln_1`, `ln_2`.

    Only names that exist on the module are returned, sorted by (layer_id, module_path).

    Args:
        model: A HuggingFace causal LM.

    Returns:
        List of `(layer_id, dotted_module_path)` pairs, one per normalization sub-module.

    Raises:
        ValueError: If the model architecture is not recognized.
    """
    layout = resolve_model_layout(model)

    result: list[tuple[int, str]] = []
    for layer_id, layer_name in enumerate(layout.layer_names):
        layer = model.get_submodule(layer_name)
        for attr in layout.norm_attrs:
            if hasattr(layer, attr) and getattr(layer, attr) is not None:
                result.append((layer_id, f"{layer_name}.{attr}"))
    result.sort(key=lambda pair: (pair[0], pair[1]))
    return result


def extract_hidden_states(input_args: tuple, input_kwargs: dict) -> torch.Tensor | None:
    """Extract hidden_states tensor from a pre-hook's arguments.

    HuggingFace transformer layers receive hidden_states either as the
    first positional argument or as a keyword argument.

    Args:
        input_args: Positional args from the pre-hook.
        input_kwargs: Keyword args from the pre-hook.

    Returns:
        The hidden_states tensor, or None if not found.
    """
    if input_args:
        return input_args[0]
    return input_kwargs.get("hidden_states")


def replace_hidden_states(
    input_args: tuple,
    input_kwargs: dict,
    new_hidden: torch.Tensor,
) -> tuple[tuple, dict]:
    """Return modified (input_args, input_kwargs) with hidden_states replaced.

    Args:
        input_args: Original positional args.
        input_kwargs: Original keyword args.
        new_hidden: Replacement hidden states tensor.

    Returns:
        Tuple of (new_input_args, new_input_kwargs).
    """
    if input_args:
        return (new_hidden, *input_args[1:]), input_kwargs
    input_kwargs = dict(input_kwargs)
    input_kwargs["hidden_states"] = new_hidden
    return input_args, input_kwargs
