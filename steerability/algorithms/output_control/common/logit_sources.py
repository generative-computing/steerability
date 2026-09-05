"""Full-vocabulary logit sources for contrastive mixing.

A logit source produces next-token log-probs `[B, V]` for the current prefix from some
distribution, either an auxiliary LM (expert / anti-expert / amateur) or the pipeline's own model on
a transformed prompt (classifier-free guidance's unconditional branch). Sources own their model
handles and any prefix-keyed KV caching.
"""
from __future__ import annotations

import gc
from abc import ABC, abstractmethod
from typing import Callable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from steerability.algorithms.core.utils.auxiliary_pass import auxiliary_pass
from steerability.utils.tokenization import infer_attention_mask_from_ids


class BaseLogitSource(ABC):
    """Produce next-token log-probs `[B, V]` for the current prefix from another distribution.

    Class attributes:
        same_model_forwards: Whether this source issues additional forward passes through the
            pipeline's own model during decoding. Such passes must be wrapped in
            `auxiliary_pass()` (see `steerability.algorithms.core.utils.auxiliary_pass`), which
            keeps them out of state-control condition scoring, gate updates, and fallback
            position counting. Defaults to False; the flag is declarative metadata and is not
            read by the pipeline.
    """

    same_model_forwards: bool = False

    @abstractmethod
    def logprobs(self, prefix_ids: torch.Tensor) -> torch.Tensor:
        """Return next-token log-probabilities `[B, V]` for `prefix_ids`."""
        ...

    def prepare(self, model=None, tokenizer=None, **kwargs) -> None:
        """Optional offline setup (e.g. load the auxiliary model), invoked from the owning control's
        `steer()`. Default no-op (mirrors the input-control selector `prepare()` convention)."""
        pass

    def cleanup(self) -> None:
        """Release any resources allocated during `prepare()`."""
        pass


class AuxModelSource(BaseLogitSource):
    """A separate causal LM (expert, anti-expert, amateur, class-conditional guide).

    Loads its own model in `prepare()` and maintains its own tokenizer. Optionally applies a prompt
    transform (e.g. prepend a control code) before its tokenization. The auxiliary vocabulary must
    match the base model's; pass `shared_vocab=True` to assert this, since the contrast family in the
    literature assumes shared vocabularies.

    Args:
        name_or_path: HF hub id or local path for the auxiliary LM.
        base_tokenizer: The base model's tokenizer (candidate ids are the base vocabulary).
        prompt_transform: Optional `str -> str` transform applied before aux tokenization.
        shared_vocab: When True (default), validate that the auxiliary vocabulary matches the base
            vocabulary and raise on mismatch. The mixture assumes shared vocabularies; setting False
            only skips the check.
        hf_model_kwargs: Extra kwargs for `from_pretrained`.
    """

    def __init__(
        self,
        name_or_path: str,
        base_tokenizer: PreTrainedTokenizerBase | None = None,
        prompt_transform: Callable[[str], str] | None = None,
        shared_vocab: bool = True,
        hf_model_kwargs: dict | None = None,
    ):
        self.name_or_path = name_or_path
        self.base_tokenizer = base_tokenizer
        self.prompt_transform = prompt_transform
        self.shared_vocab = shared_vocab
        self.hf_model_kwargs = hf_model_kwargs or {}
        self.model = None
        self.tokenizer = None
        self._device = None
        self._base_pad_id = None

    def prepare(self, model=None, tokenizer=None, **kwargs) -> None:
        if self.base_tokenizer is None:
            self.base_tokenizer = tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(self.name_or_path, **self.hf_model_kwargs)
        if model is not None:
            self.model = self.model.to(next(model.parameters()).device)
        self.model.eval()
        self._device = next(self.model.parameters()).device
        self.tokenizer = AutoTokenizer.from_pretrained(self.name_or_path)
        self.tokenizer.padding_side = "left"
        self._base_pad_id = getattr(self.base_tokenizer, "pad_token_id", None)

        if self.shared_vocab:
            aux_vocab = getattr(self.model.config, "vocab_size", None)
            # the base vocabulary is the model's output dimension (what the mixed logits must align
            # with); fall back to the tokenizer length only when no base model is available
            if model is not None:
                base_vocab = getattr(model.config, "vocab_size", None)
            elif self.base_tokenizer is not None:
                base_vocab = len(self.base_tokenizer)
            else:
                base_vocab = None
            if aux_vocab is not None and base_vocab is not None and aux_vocab != base_vocab:
                raise ValueError(
                    f"AuxModelSource vocab mismatch: aux={aux_vocab} vs base={base_vocab}. The "
                    "contrast family requires shared vocabularies; provide a token-id mapping or a "
                    "shared-vocab auxiliary model."
                )

    def set_model(self, model, tokenizer) -> None:
        """Bind an already-loaded auxiliary model/tokenizer directly (bypasses `prepare` loading)."""
        self.model = model
        self.tokenizer = tokenizer or self.tokenizer
        self.model.eval()
        self._device = next(self.model.parameters()).device
        if self.tokenizer is not None:
            self.tokenizer.padding_side = "left"
        self._base_pad_id = getattr(self.base_tokenizer, "pad_token_id", None)

    @torch.no_grad()
    def logprobs(self, prefix_ids: torch.Tensor) -> torch.Tensor:
        """Next-token log-probs `[B, V]`. Mask-correct for left-padded batches; ragged-batch
        equivalence is exact only up to the aux model's own padding-invariance."""
        if self.model is None:
            raise RuntimeError("AuxModelSource is not prepared; call prepare() from steer().")
        if self.prompt_transform is not None and self.base_tokenizer is not None:
            texts = self.base_tokenizer.decode(prefix_ids, skip_special_tokens=True)
            texts = [self.prompt_transform(t) for t in texts]
            enc = self.tokenizer(texts, return_tensors="pt", padding=True).to(self._device)
            ids = enc["input_ids"]
            mask = enc["attention_mask"]
        else:
            ids = prefix_ids.to(self._device)
            mask = infer_attention_mask_from_ids(ids, self._base_pad_id).to(self._device)
        logits = self.model(input_ids=ids, attention_mask=mask).logits[:, -1, :]
        return torch.log_softmax(logits, dim=-1)

    def cleanup(self) -> None:
        """Drop the auxiliary model / tokenizer references and reclaim memory."""
        self.model = None
        self.tokenizer = None
        self._device = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class PromptVariantSource(BaseLogitSource):
    """The pipeline's own model on a transformed prompt (CFG's unconditional branch, CAD's
    context-free branch, self-debias's biased branch).

    This source forwards the pipeline's own model, so it declares `same_model_forwards=True` and
    marks its forwards via `auxiliary_pass(aligned=False)`. The variant prompt is a different
    sequence whose positions do not map onto the generation, so state-control transforms, condition
    scoring, and gate updates all skip these passes, and the variant branch reflects the unsteered
    model.

    Args:
        prompt_transform: `str -> str` transform producing the variant prompt.
        base_tokenizer: The base model's tokenizer.
    """

    same_model_forwards: bool = True

    def __init__(self, prompt_transform: Callable[[str], str], base_tokenizer: PreTrainedTokenizerBase | None = None):
        self.prompt_transform = prompt_transform
        self.base_tokenizer = base_tokenizer
        self.model = None
        self._device = None

    def prepare(self, model=None, tokenizer=None, **kwargs) -> None:
        self.model = model
        if self.base_tokenizer is None:
            self.base_tokenizer = tokenizer
        if self.base_tokenizer is not None:
            self.base_tokenizer.padding_side = "left"
        self._device = next(model.parameters()).device

    @torch.no_grad()
    def logprobs(self, prefix_ids: torch.Tensor) -> torch.Tensor:
        """Next-token log-probs `[B, V]`. Mask-correct for left-padded batches; ragged-batch
        equivalence is exact only up to the model's own padding-invariance."""
        if self.model is None:
            raise RuntimeError("PromptVariantSource is not prepared; call prepare() from steer().")
        texts = self.base_tokenizer.decode(prefix_ids, skip_special_tokens=True)
        texts = [self.prompt_transform(t) for t in texts]
        enc = self.base_tokenizer(texts, return_tensors="pt", padding=True).to(self._device)
        with auxiliary_pass(aligned=False):
            logits = self.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits[:, -1, :]
        return torch.log_softmax(logits, dim=-1)

    def cleanup(self) -> None:
        """Drop the (pipeline-owned) model reference; the pipeline owns the model's lifecycle."""
        self.model = None
        self.base_tokenizer = None
        self._device = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class CallableSource(BaseLogitSource):
    """Adapt a `(prefix_ids) -> Tensor[B, V]` log-prob callable into a `BaseLogitSource`.

    The escape hatch of the source family: any function returning next-token log-probs for the
    current prefix becomes a source without a subclass. The wrapped function must return a
    `[B, V]` tensor aligned with the base model's vocabulary.

    `same_model_forwards` keeps the `BaseLogitSource` default of False, since it cannot be inferred
    from a bare callable. A callable that forwards the pipeline's model should be wrapped in a
    component that declares `same_model_forwards`.

    Args:
        fn: A callable mapping `prefix_ids [B, T]` to next-token log-probs `[B, V]`.
    """

    def __init__(self, fn: Callable[[torch.Tensor], torch.Tensor]):
        if not callable(fn):
            raise TypeError("CallableSource requires a callable (prefix_ids) -> Tensor[B, V].")
        self.fn = fn

    def logprobs(self, prefix_ids: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(self.fn(prefix_ids))
