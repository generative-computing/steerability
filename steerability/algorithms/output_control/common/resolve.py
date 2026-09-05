"""Component-spec resolution for the generic output controls.

The generics expose the `common` slots (values, sources, scorers) through flat `Args`, and resolve
a spec into a live component at `steer()` time, when the base model and tokenizer are available. A
spec is one of:

    - an instance of the target protocol (its `prepare(model=, tokenizer=)` is called, then it is
      returned);
    - a callable (wrapped: values -> `CallableValue`, sources -> `CallableSource`, scorers are
      already `SequenceScorer`s and are returned as-is);
    - a str (sources only: shorthand for an auxiliary model);
    - a dict with a `"kind"` key (dispatched to a loader; see the tables below).

Sweepability rule: `ControlSpec` sweeps flat constructor kwargs, so scalar knobs (`beta`, `alpha`,
`k`, `policy`, weights, budgets) are top-level `Args` fields and dict specs are treated as
identities; you sweep over configs by listing whole dicts in a spec's `vars`, not by reaching
inside them.

`SampleSequenceScorer` is served by passing an instance (it wraps a per-row `SampleScorer`
callable, which has no meaningful string form); no dict kind is added for it.
"""
from __future__ import annotations

from typing import Any

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from steerability.algorithms.output_control.common.loading import load_sequence_classifier
from steerability.algorithms.output_control.common.logit_sources import (
    AuxModelSource,
    BaseLogitSource,
    CallableSource,
    PromptVariantSource,
)
from steerability.algorithms.output_control.common.scorers.base import SequenceScorer
from steerability.algorithms.output_control.common.scorers.majority_vote import MajorityVoteScorer
from steerability.algorithms.output_control.common.scorers.reward_model import RewardModelScorer
from steerability.algorithms.output_control.common.values.base import BaseCandidateValue
from steerability.algorithms.output_control.common.values.callable import CallableValue
from steerability.algorithms.output_control.common.values.classifier import ClassifierValue
from steerability.algorithms.output_control.common.values.reward_model import RewardModelValue
from steerability.algorithms.output_control.common.values.subspace_margin import SubspaceMarginValue


def _require(spec: dict, key: str, kind: str):
    """Read a required key from a dict spec, raising a `ValueError` that names the offending kind."""
    if key not in spec or spec[key] is None:
        raise ValueError(f"{kind!r} spec requires a {key!r} key.")
    return spec[key]


def resolve_value(
    spec: Any,
    *,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    device: torch.device | str,
) -> BaseCandidateValue:
    """Resolve a value spec into a `BaseCandidateValue`.

    Accepted forms: a `BaseCandidateValue` instance; a `(StepContext) -> Tensor[B, K]` callable
    (wrapped in `CallableValue`); or a dict with a `"kind"` key:

        - `"reward_model"`: `model_id` (required); `score_index=0`, `score_transform="none"`,
          `hf_model_kwargs`. Loads an `AutoModelForSequenceClassification` and wraps it in a
          `RewardModelValue` (the text-round-trip path; `shared_vocab` defaults False).
        - `"classifier"`: `model_id` or `fn` (required); `label_index=1`, `hf_model_kwargs`.
          Wraps a loaded classifier (or a `list[str] -> Tensor` callable) in a `ClassifierValue`.
        - `"subspace_margin"`: `probe_path` or `data` (required); `batch_size=4`,
          `max_length=1024`, `save_path`. Loads a probe from a directory artifact (`Probe.load`)
          or a single-file checkpoint (`.probe` JSON or legacy `{'wv', 'mu_mu'}` tensor), or fits
          one on the base model via `fit_probe` (fisher direction over last-token features at the
          raw final-layer boundary, midpoint calibration); `save_path` writes the fitted probe's
          directory artifact.
        - `"callable"`: `fn` (required); `supports_batching`, `scoring_cost`. Wraps `fn` in a
          `CallableValue` with explicit flags.

    Args:
        spec: The value spec.
        model: The pipeline's base model.
        tokenizer: The base tokenizer.
        device: Device to load auxiliary models / align probes onto.

    Returns:
        A prepared `BaseCandidateValue`.

    Raises:
        ValueError: On an unknown kind or a malformed dict spec.
    """
    if isinstance(spec, BaseCandidateValue):
        spec.prepare(model=model, tokenizer=tokenizer)
        return spec

    if callable(spec) and not isinstance(spec, dict):
        return CallableValue(spec)

    if isinstance(spec, dict):
        kind = spec.get("kind")
        if kind == "reward_model":
            model_id = _require(spec, "model_id", "reward_model")
            rm, rm_tokenizer = load_sequence_classifier(
                model_id, device=device, hf_model_kwargs=spec.get("hf_model_kwargs"),
            )
            return RewardModelValue(
                reward_model=rm,
                rm_tokenizer=rm_tokenizer,
                score_index=spec.get("score_index", 0),
                score_transform=spec.get("score_transform", "none"),
            )
        if kind == "classifier":
            label_index = spec.get("label_index", 1)
            if spec.get("fn") is not None:
                return ClassifierValue(spec["fn"], label_index=label_index)
            model_id = _require(spec, "model_id", "classifier")
            clf, clf_tokenizer = load_sequence_classifier(
                model_id, device=device, hf_model_kwargs=spec.get("hf_model_kwargs"),
            )
            return ClassifierValue(clf, classifier_tokenizer=clf_tokenizer, label_index=label_index)
        if kind == "subspace_margin":
            # imported lazily to keep probe-fitting imports out of unrelated resolves
            import os

            from steerability.algorithms.core.internals.data import as_labeled_examples
            from steerability.algorithms.core.internals.model_layout import resolve_model_layout
            from steerability.algorithms.core.internals.probes.fitting import ProbeFitSpec, fit_probe
            from steerability.algorithms.core.internals.probes.probe import Probe
            from steerability.algorithms.output_control.common.values.subspace_margin import load_single_file_probe

            final_layer = resolve_model_layout(model).num_layers - 1
            if spec.get("probe_path") is not None:
                path = spec["probe_path"]
                if os.path.isdir(path):
                    probe = Probe.load(path)
                else:
                    probe = load_single_file_probe(path, layer_id=final_layer)
            elif spec.get("data") is not None:
                fit_spec = ProbeFitSpec(
                    method="fisher",
                    pooling="last",
                    location="layer_output",
                    prompt_format="raw",
                    candidate_layers=[final_layer],
                    calibration="midpoint",
                )
                probe = fit_probe(
                    model,
                    tokenizer,
                    data=as_labeled_examples(spec["data"]),
                    spec=fit_spec,
                    batch_size=spec.get("batch_size", 4),
                    max_length=spec.get("max_length", 1024),
                )
                if spec.get("save_path") is not None:
                    probe.save(spec["save_path"])
            else:
                raise ValueError("'subspace_margin' spec requires a 'probe_path' or 'data' key.")
            return SubspaceMarginValue(probe)
        if kind == "callable":
            fn = _require(spec, "fn", "callable")
            return CallableValue(
                fn,
                supports_batching=spec.get("supports_batching", False),
                scoring_cost=spec.get("scoring_cost", "cheap"),
            )
        raise ValueError(
            f"Unknown value kind {kind!r}; accepted kinds are 'reward_model', 'classifier', "
            "'subspace_margin', 'callable' (or pass a BaseCandidateValue instance or a callable)."
        )

    raise ValueError(
        "A value spec must be a BaseCandidateValue instance, a callable, or a dict with a 'kind' key."
    )


def resolve_source(spec: Any, *, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase) -> BaseLogitSource:
    """Resolve a source spec into a prepared `BaseLogitSource`.

    Accepted forms: a `BaseLogitSource` instance; a `(prefix_ids) -> Tensor[B, V]` callable (wrapped
    in `CallableSource`); a `str` (shorthand for `{"kind": "aux_model", "name_or_path": <str>}`); or
    a dict with a `"kind"` key:

        - `"aux_model"`: `name_or_path` (required); `prompt_transform`, `hf_model_kwargs`,
          `shared_vocab=True`. Builds an `AuxModelSource` (its `prepare` loads and vocab-checks).
        - `"prompt_variant"`: `prompt_transform` (required). Builds a `PromptVariantSource`
          (see its compatibility note on same-model forwards).

    Args:
        spec: The source spec.
        model: The pipeline's base model.
        tokenizer: The base tokenizer.

    Returns:
        A prepared `BaseLogitSource`.

    Raises:
        ValueError: On an unknown kind or a malformed dict spec.
    """
    if isinstance(spec, BaseLogitSource):
        spec.prepare(model=model, tokenizer=tokenizer)
        return spec

    if isinstance(spec, str):
        spec = {"kind": "aux_model", "name_or_path": spec}

    if callable(spec) and not isinstance(spec, dict):
        source = CallableSource(spec)
        source.prepare(model=model, tokenizer=tokenizer)
        return source

    if isinstance(spec, dict):
        kind = spec.get("kind")
        if kind == "aux_model":
            name_or_path = _require(spec, "name_or_path", "aux_model")
            source = AuxModelSource(
                name_or_path,
                base_tokenizer=tokenizer,
                prompt_transform=spec.get("prompt_transform"),
                shared_vocab=spec.get("shared_vocab", True),
                hf_model_kwargs=spec.get("hf_model_kwargs"),
            )
            source.prepare(model=model, tokenizer=tokenizer)
            return source
        if kind == "prompt_variant":
            prompt_transform = _require(spec, "prompt_transform", "prompt_variant")
            source = PromptVariantSource(prompt_transform, base_tokenizer=tokenizer)
            source.prepare(model=model, tokenizer=tokenizer)
            return source
        raise ValueError(
            f"Unknown source kind {kind!r}; accepted kinds are 'aux_model', 'prompt_variant' "
            "(or pass a BaseLogitSource instance, a callable, or an aux-model name/path string)."
        )

    raise ValueError(
        "A source spec must be a BaseLogitSource instance, a callable, a str, or a dict with a "
        "'kind' key."
    )


def resolve_scorer(spec: Any, *, device: torch.device | str) -> SequenceScorer:
    """Resolve a scorer spec into a `SequenceScorer`.

    Accepted forms: any `SequenceScorer` callable (returned as-is, such as a plain function, a
    `MajorityVoteScorer`, or a `SampleSequenceScorer`); or a dict with a `"kind"` key:

        - `"reward_model"`: `model_id` (required); `score_index=0`, `batch_size=8`,
          `hf_model_kwargs`. Loads a classifier and wraps it in a `RewardModelScorer`.
        - `"majority_vote"`: `answer_extractor` (optional). Builds a `MajorityVoteScorer`.

    Args:
        spec: The scorer spec.
        device: Device to load an auxiliary reward model onto.

    Returns:
        A `SequenceScorer` (a callable `(prompt, continuations, params) -> list[float]`).

    Raises:
        ValueError: On an unknown kind or a malformed dict spec.
    """
    if isinstance(spec, dict):
        kind = spec.get("kind")
        if kind == "reward_model":
            model_id = _require(spec, "model_id", "reward_model")
            rm, rm_tokenizer = load_sequence_classifier(
                model_id, device=device, hf_model_kwargs=spec.get("hf_model_kwargs"),
            )
            return RewardModelScorer(
                rm,
                rm_tokenizer,
                score_index=spec.get("score_index", 0),
                batch_size=spec.get("batch_size", 8),
            )
        if kind == "majority_vote":
            return MajorityVoteScorer(answer_extractor=spec.get("answer_extractor"))
        raise ValueError(
            f"Unknown scorer kind {kind!r}; accepted kinds are 'reward_model', 'majority_vote' "
            "(or pass a SequenceScorer callable / SampleSequenceScorer instance)."
        )

    if callable(spec):
        return spec

    raise ValueError(
        "A scorer spec must be a SequenceScorer callable (e.g. a function, MajorityVoteScorer, or "
        "SampleSequenceScorer) or a dict with a 'kind' key."
    )
