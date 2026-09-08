"""Sources: recipes that resolve to concrete steering elements for a given model.

This module provides the `ArtifactSource` protocol with its fit recipes (`ContrastiveFit`,
`SinglePairFit`) and the gate source `ConditionPointSearch`. A transform holds either a
concrete artifact (a `SteeringVector` or a per-layer directions mapping) or a source; an
`Intervention`'s gate slot holds either a concrete gate or a gate source. `Intervention.bind`
resolves sources, so steer-time computations (fits, searches) have a declarative home.
`resolve` returns a defensive clone, and the underlying fit is memoized per model.
"""
from __future__ import annotations

import warnings
import weakref
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, Mapping, Protocol, Sequence, runtime_checkable

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from steerability.algorithms.core.execution.access import ModelAccess
from steerability.algorithms.core.internals.capture import HiddenStateLocation
from steerability.algorithms.core.internals.data import ContrastivePairs, as_contrastive_pairs
from steerability.algorithms.state_control.common.estimators import (
    ContrastiveDirectionEstimator,
    MeanDifferenceEstimator,
)
from steerability.algorithms.state_control.common.estimators.base import BaseEstimator
from steerability.algorithms.state_control.common.fit_specs import (
    Comparator,
    CompMode,
    ConditionSearchSpec,
    VectorTrainSpec,
)
from steerability.algorithms.state_control.common.steering_vector import SteeringVector
from steerability.utils.rendering import PromptFormat

if TYPE_CHECKING:
    from steerability.algorithms.core.execution.backend import SteeringSession
    from steerability.algorithms.core.execution.payloads import ModelFacts
    from steerability.algorithms.state_control.common.gating import Gate


@runtime_checkable
class ArtifactSource(Protocol):
    """A recipe for obtaining a steering artifact for a specific model.

    Implementations MUST return a defensive clone from `resolve` (callers may move/mutate their
    copy) and SHOULD memoize the underlying fit per model so repeated resolves against one model
    (e.g., a parameter sweep) fit only once. Implementations whose fitted directions are
    positional (`[T, H]` with `T > 1`) declare it with a class-level `produces_positional =
    True`, which transform factories read for kind planning before the fit runs.
    """

    def resolve(
        self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, *, session: "SteeringSession | None" = None
    ) -> SteeringVector:
        """Return the steering artifact for this model (a fresh clone each call).

        `session` is a `SteeringSession` a capture-backed fit may extract hidden states
        through; sources whose fit requires a live model ignore it.
        """
        ...


@dataclass
class ContrastiveFit:
    """A fit recipe: contrastive pairs plus how to extract a per-layer direction.

    The five spec fields (`method`, `accumulate`, `batch_size`, `prompt_format`, `location`) drive
    the built-in estimators. `"mean_diff"` dispatches to `MeanDifferenceEstimator`, everything else
    to `ContrastiveDirectionEstimator`, and the fitted artifact records its extraction boundary as
    `meta["location"]`, which `Intervention.bind` checks against the consuming intervention's
    boundary. When a custom `estimator` is supplied, the spec fields are ignored (a warning is
    emitted), no location is recorded, and fitting delegates to
    `estimator.fit(model, tokenizer, data=<coerced pairs>, **(estimator_kwargs or {}))`.

    The fitted master vector is memoized in a single weakref slot keyed by model identity. The same
    model resolved repeatedly fits once, a different model refits, and alternating models (A→B→A)
    refit on each switch. Every `resolve` returns an independent clone, so a consumer may freely
    `.to(...)` or mutate its copy without touching the master.

    Attributes:
        data: Contrastive pairs (or a dict coerced via `as_contrastive_pairs`).
        method: Direction-extraction method used to fit the per-layer direction.
        accumulate: Hidden-state span selection for the fit.
        batch_size: Forward-pass batch size used during fitting.
        prompt_format: How to render pairs into model-ready text.
        location: Residual-stream boundary to fit at.
        normalize: L2-normalize the fitted master per layer once, before caching.
        estimator: Optional custom `BaseEstimator`; when set, the spec fields are ignored.
        estimator_kwargs: Extra kwargs forwarded to a custom `estimator.fit(...)`.
    """

    produces_positional: ClassVar[bool] = False
    artifact_class: ClassVar[str] = "direction"

    data: ContrastivePairs | dict
    method: str = "pca_pairwise"
    accumulate: str = "all"
    batch_size: int = 8
    prompt_format: PromptFormat = "chat_completion"
    location: HiddenStateLocation = "layer_output"
    normalize: bool = False
    estimator: BaseEstimator | None = None
    estimator_kwargs: dict | None = None

    _model_ref: "weakref.ref | None" = field(default=None, init=False, repr=False, compare=False)
    _master: SteeringVector | None = field(default=None, init=False, repr=False, compare=False)

    @property
    def access(self) -> ModelAccess:
        """`ModelAccess.CAPTURE` for the built-in estimators, whose extraction runs through
        session capture; a custom estimator may need the live model, so it declares
        `ModelAccess.MODULE`."""
        return ModelAccess.MODULE if self.estimator is not None else ModelAccess.CAPTURE

    def __post_init__(self):
        if not isinstance(self.data, ContrastivePairs):
            self.data = as_contrastive_pairs(self.data)

        spec_customized = (
            self.method != "pca_pairwise"
            or self.accumulate != "all"
            or self.batch_size != 8
            or self.prompt_format != "chat_completion"
            or self.location != "layer_output"
        )
        if self.estimator is not None and spec_customized:
            warnings.warn(
                "method/accumulate/batch_size/prompt_format/location are ignored when a custom "
                "estimator is supplied; the estimator owns its config via estimator_kwargs.",
                UserWarning,
            )
        if self.estimator is None and self.estimator_kwargs is not None:
            warnings.warn("estimator_kwargs is inert without a custom estimator.", UserWarning)

    def _fit(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, session=None) -> SteeringVector:
        """Fit the master steering vector (no caching, no cloning)."""
        if self.estimator is not None:
            try:
                master = self.estimator.fit(
                    model, tokenizer, data=self.data, session=session, **(self.estimator_kwargs or {})
                )
            except TypeError:
                master = self.estimator.fit(model, tokenizer, data=self.data, **(self.estimator_kwargs or {}))
        else:
            spec = VectorTrainSpec(
                method=self.method,
                accumulate=self.accumulate,
                batch_size=self.batch_size,
                prompt_format=self.prompt_format,
                location=self.location,
            )
            estimator = MeanDifferenceEstimator() if self.method == "mean_diff" else ContrastiveDirectionEstimator()
            master = estimator.fit(model, tokenizer, data=self.data, spec=spec, session=session)
            master.meta["location"] = self.location

        if self.normalize:
            master = master.normalized()
        return master

    def resolve(
        self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, *, session: "SteeringSession | None" = None
    ) -> SteeringVector:
        """Return a fresh clone of the fitted artifact for `model`, fitting once and memoizing.

        Args:
            model: The model to fit against (or a memo hit for the same model), or None to
                fit through `session` capture.
            tokenizer: Tokenizer used to encode the contrastive pairs when fitting.
            session: Optional `SteeringSession` the estimator extracts hidden states through
                when `model` is None (capture-backed fitting).

        Returns:
            An independent `SteeringVector` clone the caller owns.
        """
        if model is not None and self._model_ref is not None and self._model_ref() is model \
                and self._master is not None:
            return self._master.clone()
        master = self._fit(model, tokenizer, session=session)
        if model is not None:
            self._model_ref = weakref.ref(model)
            self._master = master
        return master.clone()


class _Precomputed:
    """A trivially-resolved source wrapping a concrete `SteeringVector` (internal).

    Lets resolvers treat concrete artifacts and sources uniformly, so precomputed vectors take
    the same bind path as fitted ones (defensive clone, device/dtype cast). Resolution is
    model-free, so the source declares `ModelAccess.FACTS` and runs no fit. Not part of the
    public API; users pass vectors, mappings, or sources directly.
    """

    access: ClassVar[ModelAccess] = ModelAccess.FACTS

    def __init__(self, steering_vector: SteeringVector):
        self._steering_vector = steering_vector

    @property
    def produces_positional(self) -> bool:
        return self._steering_vector.is_positional

    def resolve(
        self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, *, session: "SteeringSession | None" = None
    ) -> SteeringVector:
        return self._steering_vector.clone()


class VerifiedPrecomputed:
    """A precomputed source that enforces recorded provenance at resolve time.

    Wraps a concrete `SteeringVector` together with the fingerprints of the side that produced
    it. Resolution is model-free (`ModelAccess.FACTS`) and reads structural facts from the
    session layout (or the live model when one is given): the vector's `model_type` and width
    are always checked, and the recorded model fingerprint is checked per the policy. A
    `"calibrated"` artifact on a mismatched model raises under `policy="strict"` and warns
    under `"warn"`; a `"direction"` artifact warns under both, since transferring a direction
    across fine-tunes of one architecture is a deliberate act. `policy="off"` skips every
    check. Checks that cannot run (no layout available, or no recorded value) are skipped.

    Args:
        steering_vector: The concrete artifact.
        provenance: Producing-side fingerprints (mapping or `ArtifactProvenance`-shaped), with
            keys `model_fingerprint`, `backend_spec_hash`, `tokenizer_fingerprint`.
        artifact_class: `"direction"` or `"calibrated"`.
        policy: `"strict"`, `"warn"`, or `"off"`.
    """

    access: ClassVar[ModelAccess] = ModelAccess.FACTS

    def __init__(
        self,
        steering_vector: SteeringVector,
        provenance: Mapping | None = None,
        artifact_class: str = "direction",
        policy: str = "strict",
    ):
        if policy not in ("strict", "warn", "off"):
            raise ValueError(f"policy must be 'strict', 'warn', or 'off'; got {policy!r}.")
        self.steering_vector = steering_vector
        self.provenance = dict(provenance or {})
        self.artifact_class = artifact_class
        self.policy = policy

    @property
    def produces_positional(self) -> bool:
        return self.steering_vector.is_positional

    def _report(self, message: str, *, hard: bool) -> None:
        if self.policy == "off":
            return
        if hard and self.policy == "strict":
            raise ValueError(message)
        warnings.warn(message, UserWarning)

    def _check(self, layout) -> None:
        vector = self.steering_vector
        if layout is None:
            return

        layout_type = getattr(layout, "model_type", None)
        if vector.model_type not in ("unknown", None) and layout_type not in (None, "unknown") \
                and vector.model_type != layout_type:
            self._report(
                f"Precomputed steering artifact was produced for model_type "
                f"{vector.model_type!r} but this pipeline serves {layout_type!r}.",
                hard=True,
            )

        if vector.num_heads is not None and vector.head_dim is not None:
            num_heads = getattr(layout, "num_attention_heads", None)
            head_dim = getattr(layout, "head_dim", None)
            if (num_heads is not None and vector.num_heads != num_heads) or \
                    (head_dim is not None and vector.head_dim != head_dim):
                self._report(
                    f"Precomputed per-head artifact has num_heads={vector.num_heads}, "
                    f"head_dim={vector.head_dim} but the model has num_heads={num_heads}, "
                    f"head_dim={head_dim}.",
                    hard=True,
                )
        elif vector.directions:
            width = next(iter(vector.directions.values())).size(-1)
            hidden_size = getattr(layout, "hidden_size", None)
            if hidden_size and width != hidden_size:
                self._report(
                    f"Precomputed steering artifact has width {width} but the model's hidden "
                    f"size is {hidden_size}.",
                    hard=True,
                )

        recorded = self.provenance.get("model_fingerprint")
        live = getattr(layout, "model_fingerprint", None)
        if recorded and live and recorded != live:
            self._report(
                f"Precomputed {self.artifact_class} artifact was produced on a different "
                f"model (fingerprint {recorded!r} vs {live!r})."
                + ("" if self.artifact_class == "calibrated"
                   else " Direction artifacts may transfer across fine-tunes of one "
                        "architecture; verify the behavior."),
                hard=(self.artifact_class == "calibrated"),
            )

    def resolve(
        self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, *, session: "SteeringSession | None" = None
    ) -> SteeringVector:
        """Check provenance against the resolution venue and return a fresh clone.

        Args:
            model: The live model, or None when a session layout supplies the facts.
            tokenizer: Unused.
            session: Optional `SteeringSession` whose `layout` supplies structural facts.

        Returns:
            An independent `SteeringVector` clone the caller owns.

        Raises:
            ValueError: Under `policy="strict"`, if the model type or width mismatches, or a
                `"calibrated"` artifact's recorded fingerprint differs from the live model's.
        """
        layout = None
        if session is not None:
            try:
                layout = session.layout
            except Exception:
                layout = None
        if layout is None and model is not None:
            from steerability.algorithms.state_control.common.layout_facts import resolve_layout

            layout = resolve_layout(model, None)
        self._check(layout)
        return self.steering_vector.clone()


def _as_artifact_source(x) -> ArtifactSource:
    """Coerce a concrete artifact or source into an `ArtifactSource` (internal).

    Accepts an `ArtifactSource` (returned as-is), a `SteeringVector` (wrapped in `_Precomputed`), or
    a `Mapping[int, torch.Tensor]` of per-layer directions (wrapped in a `SteeringVector` with
    `model_type="unknown"` then `_Precomputed`). Anything else raises `TypeError`.
    """
    if isinstance(x, ArtifactSource):
        return x
    if isinstance(x, SteeringVector):
        return _Precomputed(x)
    if isinstance(x, Mapping):
        directions = {int(k): v for k, v in x.items()}
        if not all(isinstance(v, torch.Tensor) for v in directions.values()):
            raise TypeError("Mapping artifact must map layer ids to torch.Tensor directions.")
        return _Precomputed(SteeringVector(model_type="unknown", directions=directions))
    raise TypeError(
        f"Expected a SteeringVector, a Mapping[int, Tensor], or an ArtifactSource; got "
        f"{type(x).__name__}."
    )


@dataclass
class SinglePairFit:
    """A fit recipe for a positional steering vector: per-token differences of one contrast pair.

    Produces `[T, H]` directions per layer (the ActAdd extraction), captured at the layer-input
    boundary and recorded as `meta["location"] = "layer_input"`. Fitting requires a live model,
    since the pair is tokenized and co-padded as one batch and remote capture serves per-prompt
    layouts that do not preserve the co-padded positional structure. The fitted master is
    memoized per model; every `resolve` returns an independent clone.

    Attributes:
        positive_prompt: The steering-direction prompt.
        negative_prompt: The contrast prompt.
        normalize: L2-normalize each per-position vector once, before caching.
    """

    produces_positional: ClassVar[bool] = True
    access: ClassVar[ModelAccess] = ModelAccess.MODULE
    artifact_class: ClassVar[str] = "direction"

    positive_prompt: str
    negative_prompt: str
    normalize: bool = False

    _model_ref: "weakref.ref | None" = field(default=None, init=False, repr=False, compare=False)
    _master: SteeringVector | None = field(default=None, init=False, repr=False, compare=False)

    def resolve(
        self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, *, session: "SteeringSession | None" = None
    ) -> SteeringVector:
        """Return a fresh clone of the fitted artifact for `model`, fitting once and memoizing.

        Args:
            model: The model to fit against (or a memo hit for the same model).
            tokenizer: Tokenizer used to encode the prompt pair.
            session: Ignored; the positional fit requires a live model.

        Returns:
            An independent `SteeringVector` clone the caller owns.

        Raises:
            ValueError: If `model` is None.
        """
        if model is None:
            raise ValueError("Fitting ActAdd from a prompt pair requires a live model at steer time.")
        if self._model_ref is not None and self._model_ref() is model and self._master is not None:
            return self._master.clone()

        from steerability.algorithms.state_control.common.estimators import SinglePairEstimator

        master = SinglePairEstimator().fit(
            model, tokenizer,
            positive_prompt=self.positive_prompt,
            negative_prompt=self.negative_prompt,
        )
        if self.normalize:
            master = master.clone()
            for layer_id, direction in master.directions.items():
                norms = direction.norm(dim=-1, keepdim=True)
                master.directions[layer_id] = direction / (norms + 1e-8)
        self._model_ref = weakref.ref(model)
        self._master = master
        return master.clone()


@dataclass
class ConditionPointSearch:
    """A gate recipe: contrastive condition data plus how to find the gate point.

    Occupies an `Intervention`'s gate slot. `resolve_gate` fits (or clones) the condition
    vector, resolves the condition point by grid search when `search` enables it and no manual
    layers are given, and assembles the gate: projected-cosine evidence over the condition
    directions at the resolved layers, decided by a per-layer threshold rule. An unconditional
    configuration (no condition vector or no resolved point) resolves to None.

    The projected-cosine readout and the per-layer threshold rule both have wire forms, so an
    intervention gated this way lowers to intervention-capable backends
    (`wire_readouts`/`wire_rules` declare the kinds for pre-steer support checks).

    Attributes:
        condition_vector: Precomputed condition directions, cloned rather than refit.
        condition_data: Contrastive pairs used to fit the condition vector and calibrate the
            search.
        condition_fit: Fit configuration for the condition vector.
        search: Condition point search configuration.
        layer_ids: Manual condition layers; disables the search when set.
        threshold: Manual gate threshold, required with `layer_ids`.
        comparator: Gate comparator for the manual point (`"ge"` opens when score >= threshold,
            `"le"` when score <= threshold).
        comparison_mode: Runtime token aggregation for condition scoring.
        resolved_point: The `(layer_ids, threshold, comparator)` the last resolve produced, or
            None before resolution or for unconditional configurations.
    """

    wire_readouts: ClassVar[frozenset[str] | None] = frozenset({"projected_cosine"})
    wire_rules: ClassVar[frozenset[str] | None] = frozenset({"per_key_threshold"})
    access: ClassVar[ModelAccess] = ModelAccess.MODULE
    artifact_class: ClassVar[str] = "calibrated"

    condition_vector: SteeringVector | None = None
    condition_data: ContrastivePairs | dict | None = None
    condition_fit: VectorTrainSpec = field(
        default_factory=lambda: VectorTrainSpec(prompt_format="chat_prompt", location="layer_input")
    )
    search: ConditionSearchSpec = field(default_factory=ConditionSearchSpec)
    layer_ids: Sequence[int] | None = None
    threshold: float | None = None
    comparator: Comparator = "ge"
    comparison_mode: CompMode = "mean"

    resolved_point: dict | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self):
        if self.condition_data is not None and not isinstance(self.condition_data, ContrastivePairs):
            self.condition_data = as_contrastive_pairs(self.condition_data)
        if self.comparator not in ("ge", "le"):
            raise ValueError(f"comparator must be 'ge' or 'le'; got {self.comparator!r}.")

    def resolve_gate(
        self,
        model: PreTrainedModel | None,
        tokenizer: PreTrainedTokenizerBase,
        *,
        layout: "ModelFacts | None" = None,
        session: "SteeringSession | None" = None,
    ) -> "Gate | None":
        """Resolve the gate for `model`.

        Args:
            model: The model to search against; required when the search runs or the condition
                vector is fitted from data.
            tokenizer: Tokenizer used to encode the condition data.
            layout: Structural facts, used to bounds-check condition layers.
            session: Optional `SteeringSession` for capture-backed fitting and calibration.

        Returns:
            The gate, or None for unconditional configurations.

        Raises:
            ValueError: If a manual threshold is set without a condition vector, or a condition
                layer lacks a direction.
        """
        from steerability.algorithms.state_control.common.gating import (
            Evidence,
            Gate,
            PerKeyThreshold,
            ProjectedCosineReadout,
        )
        from steerability.algorithms.state_control.common.selectors import ConditionPointSelector

        condition_vec = self.condition_vector.clone() if self.condition_vector is not None else None
        condition_supplied = condition_vec is not None or self.condition_data is not None
        if condition_supplied and condition_vec is None:
            if self.condition_fit.method == "mean_diff" and self.condition_fit.accumulate == "suffix-only":
                raise ValueError(
                    "method='mean_diff' does not support accumulate='suffix-only'; "
                    "use accumulate='all' or 'last_token', or method='pca_pairwise'/'pca_center'."
                )
            estimator = (
                MeanDifferenceEstimator()
                if self.condition_fit.method == "mean_diff"
                else ContrastiveDirectionEstimator()
            )
            condition_vec = estimator.fit(
                model, tokenizer, data=self.condition_data, spec=self.condition_fit, session=session,
            )
            if model is not None:
                device = next(model.parameters()).device
                condition_vec = condition_vec.to(device, dtype=model.dtype)

        layer_ids = self.layer_ids
        threshold = self.threshold
        comparator = self.comparator

        if condition_supplied and condition_vec is not None:
            if self.search.auto_find and layer_ids is None and self.condition_data is not None:
                result = ConditionPointSelector().select(
                    model=model,
                    tokenizer=tokenizer,
                    condition_directions=condition_vec.directions,
                    data=self.condition_data,
                    fit_spec=self.condition_fit,
                    search_spec=self.search,
                    comparison_mode=self.comparison_mode,
                    session=session,
                )
                layer_ids = [result.layer_id]
                threshold = result.threshold
                comparator = result.comparator

        layer_set = sorted(set(int(lid) for lid in (layer_ids or [])))
        conditional = bool(layer_set) and threshold is not None
        if conditional and condition_vec is None:
            raise ValueError("Conditional gating requires a condition vector.")
        if not conditional:
            self.resolved_point = None
            return None

        missing = [lid for lid in layer_set if lid not in condition_vec.directions]
        if missing:
            raise ValueError(f"Condition vector has no direction for condition layer(s) {missing}.")

        readout = ProjectedCosineReadout(
            {lid: condition_vec.directions[lid] for lid in layer_set},
        )
        rule = PerKeyThreshold(threshold=threshold, comparator=comparator, aggregate="any")
        self.resolved_point = {
            "layer_ids": layer_set,
            "threshold": threshold,
            "comparator": comparator,
            "comparison_mode": self.comparison_mode,
        }
        return Gate(
            Evidence(tuple(layer_set), readout, pooling=self.comparison_mode),
            rule,
        )


@dataclass
class LayerFilteredFit:
    """Wraps a source and restricts the resolved directions to a layer range.

    Access, artifact class, and positional-ness delegate to the wrapped source. The filtered
    result keeps the inner artifact's metadata and per-layer statistics for the surviving
    layers.

    Attributes:
        inner: The wrapped source.
        layer_range: 0-based half-open `(start, end)` range; None passes every layer through.
    """

    inner: "ArtifactSource"
    layer_range: tuple[int, int] | None = None

    @property
    def access(self) -> ModelAccess | None:
        return getattr(self.inner, "access", None)

    @property
    def artifact_class(self) -> str | None:
        return getattr(self.inner, "artifact_class", None)

    @property
    def produces_positional(self) -> bool:
        return bool(getattr(self.inner, "produces_positional", False))

    def resolve(
        self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, *, session: "SteeringSession | None" = None
    ) -> SteeringVector:
        """Resolve the inner source and filter its directions to `layer_range`."""
        resolved = self.inner.resolve(model, tokenizer, session=session)
        if self.layer_range is None:
            return resolved
        start, end = self.layer_range
        directions = {
            layer_id: direction
            for layer_id, direction in resolved.directions.items()
            if start <= layer_id < end
        }
        return SteeringVector(
            model_type=resolved.model_type,
            directions=directions,
            num_heads=resolved.num_heads,
            head_dim=resolved.head_dim,
            explained_variances=resolved.explained_variances,
            probe_accuracies=resolved.probe_accuracies,
            meta=dict(resolved.meta),
        )
