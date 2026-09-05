"""The one auxiliary sequence-classifier loader.

Loads and configures an `AutoModelForSequenceClassification` reward model / attribute classifier
the way the output category needs it: eval mode, pad-token fallback to EOS, right padding, and a
clamp of the sentinel `model_max_length` some tokenizers carry onto a plain `max_length` attribute
that `RewardModelValue` reads. Used by the `reward_model` / `classifier` value loaders, the
`reward_model` scorer loader, and RAD's HF-classifier path. Granite checkpoints, for which transformers
ships no head, resolve through the toolkit heads selected by `sequence_classifier_class`.
"""
from __future__ import annotations

import logging

import torch
from transformers import AutoTokenizer

from steerability.algorithms.output_control.common.granite_heads import sequence_classifier_class

logger = logging.getLogger(__name__)


def load_sequence_classifier(
    model_id: str,
    *,
    device: str | torch.device,
    hf_model_kwargs: dict | None = None,
    max_length_clamp: int = 512,
) -> tuple:
    """Load an HF sequence classifier and its tokenizer, configured for candidate/continuation scoring.

    Args:
        model_id: HF hub id or local path for an `AutoModelForSequenceClassification`.
        device: Device to move the loaded model onto.
        hf_model_kwargs: Extra kwargs forwarded to `from_pretrained`.
        max_length_clamp: Value to use for `tokenizer.max_length` when the tokenizer's
            `model_max_length` is unset or an absurd sentinel (`> 100_000`).

    Returns:
        A tuple `(model, tokenizer)`; `model` is in eval mode on `device`, and `tokenizer` has
        `pad_token` set (falling back to `eos_token`), `padding_side="right"`, and a `max_length`
        attribute.
    """
    kwargs = dict(hf_model_kwargs or {})
    model = sequence_classifier_class(model_id, **kwargs).from_pretrained(model_id, **kwargs)
    model = model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    max_len = getattr(tokenizer, "model_max_length", None)
    if max_len is None or max_len > 100_000:
        max_len = max_length_clamp
    tokenizer.max_length = max_len

    logger.debug("Loaded sequence classifier from %s", model_id)
    return model, tokenizer
