"""Sequence-classification heads for the Granite and GraniteMoeHybrid architectures.

transformers ships no `AutoModelForSequenceClassification` head for the Granite families, so a Granite
causal-LM checkpoint cannot be loaded as a scalar-head reward model out of the box. These two classes
supply the head by mixing `GenericForSequenceClassification` (the pooled-last-token classifier the
Llama and Qwen heads use) with each family's `PreTrainedModel`. Callers select the loading class with
`sequence_classifier_class`, which reads the checkpoint config and returns the toolkit head for a
Granite family and `AutoModelForSequenceClassification` otherwise, so a head shipped by a future
transformers version wins.
"""
from __future__ import annotations

from transformers import (
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
    AutoConfig,
    AutoModelForSequenceClassification,
    PreTrainedModel,
)
from transformers.modeling_layers import GenericForSequenceClassification
from transformers.models.granite.configuration_granite import GraniteConfig
from transformers.models.granite.modeling_granite import GranitePreTrainedModel
from transformers.models.granitemoehybrid.configuration_granitemoehybrid import GraniteMoeHybridConfig
from transformers.models.granitemoehybrid.modeling_granitemoehybrid import GraniteMoeHybridPreTrainedModel


class GraniteForSequenceClassification(GenericForSequenceClassification, GranitePreTrainedModel):
    """Sequence-classification head for the `granite` architecture."""


class GraniteMoeHybridForSequenceClassification(GenericForSequenceClassification, GraniteMoeHybridPreTrainedModel):
    """Sequence-classification head for the `granitemoehybrid` architecture."""


GRANITE_SEQUENCE_CLASSIFIERS: dict[type, type[PreTrainedModel]] = {
    GraniteConfig: GraniteForSequenceClassification,
    GraniteMoeHybridConfig: GraniteMoeHybridForSequenceClassification,
}

# from_pretrained kwargs that affect which config.json is read
_HUB_KWARGS = ("cache_dir", "force_download", "local_files_only", "proxies", "revision", "subfolder", "token",
               "trust_remote_code")


def sequence_classifier_class(model_id: str, **from_pretrained_kwargs) -> type:
    """The class that loads `model_id` as a sequence classifier.

    Reads the checkpoint config (the hub-related entries of `from_pretrained_kwargs` are forwarded) and
    returns `AutoModelForSequenceClassification` when its mapping covers the config class, the toolkit
    head from `GRANITE_SEQUENCE_CLASSIFIERS` when the config is a Granite family the mapping does not
    cover, and `AutoModelForSequenceClassification` otherwise (its `from_pretrained` then raises the
    standard unrecognized-config error). The resolution is explicit rather than through
    `AutoModelForSequenceClassification.register`, which transformers 5.13 and later ignore for native
    config classes.

    Args:
        model_id: HF hub id or local path.
        **from_pretrained_kwargs: The kwargs the caller will pass to `from_pretrained`.

    Returns:
        The class to call `from_pretrained` on.
    """
    hub_kwargs = {name: from_pretrained_kwargs[name] for name in _HUB_KWARGS if name in from_pretrained_kwargs}
    config = AutoConfig.from_pretrained(model_id, **hub_kwargs)
    if type(config) in MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING:
        return AutoModelForSequenceClassification
    return GRANITE_SEQUENCE_CLASSIFIERS.get(type(config), AutoModelForSequenceClassification)
