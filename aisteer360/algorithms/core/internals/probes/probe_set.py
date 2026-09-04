"""Batched probe reads, scoring every probe in a set in one read-only forward."""
import logging
from dataclasses import dataclass, field
from typing import Mapping

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from aisteer360.algorithms.core.internals.data import ContrastivePairs
from aisteer360.algorithms.core.internals.fingerprint import model_fingerprint
from aisteer360.algorithms.core.internals.pooling import aggregate_condition_hidden
from aisteer360.algorithms.core.internals.probes.fitting import ProbeFitSpec, fit_probe
from aisteer360.algorithms.core.internals.probes.probe import Probe
from aisteer360.algorithms.core.internals.stats import ActivationStats, StatsSpec
from aisteer360.algorithms.core.utils.auxiliary_pass import auxiliary_pass

logger = logging.getLogger(__name__)


@dataclass
class ProbeReadings:
    """One `ProbeSet.read()` call's result.

    Attributes:
        scores: Mapping from probe name to signed decision scores, `[B]` per probe.
        decisions: Mapping from probe name to boolean decisions, `[B]` per probe
            (`scores >= 0`, ties open).
    """

    scores: dict[str, torch.Tensor]
    decisions: dict[str, torch.Tensor]


@dataclass
class ProbeSetFit:
    """Deferred fitting recipe; holds every fitting input except the model.

    `fit(model, tokenizer)` estimates `stats` first when it is a `StatsSpec`, fits one probe
    per name, and returns a `ProbeSet`. `RoutedDecoding.steer()` calls `fit` on the model the
    pipeline provides, so pipelines whose structural controls produce the final weights fit on
    those weights, and `ProbeFitSpec` fields stay grid-searchable through `ControlSpec`.

    Attributes:
        data: Mapping from probe name to the contrastive pairs it is fitted on.
        spec: A shared `ProbeFitSpec`, or a mapping from probe name to a per-name spec covering
            every name in `data`.
        stats: Ambient activation statistics, or a `StatsSpec` estimated on the model at fit
            time. Required when any effective spec's method is `"lda"` or `"logreg"`.
        calibration_data: Optional mapping from probe name to calibration pairs; names absent
            from the mapping calibrate on their fit pairs.
    """

    data: Mapping[str, ContrastivePairs]
    spec: ProbeFitSpec | Mapping[str, ProbeFitSpec] = field(default_factory=ProbeFitSpec)
    stats: ActivationStats | StatsSpec | None = None
    calibration_data: Mapping[str, ContrastivePairs] | None = None

    def __post_init__(self):
        if not self.data:
            raise ValueError("data must map at least one probe name to contrastive pairs.")
        if isinstance(self.spec, Mapping):
            missing = [name for name in self.data if name not in self.spec]
            if missing:
                raise ValueError(f"spec mapping is missing entries for probe(s) {missing}.")
        if self.calibration_data is not None:
            unknown = [name for name in self.calibration_data if name not in self.data]
            if unknown:
                raise ValueError(
                    f"calibration_data names unknown probe(s) {unknown}; "
                    f"available: {sorted(self.data)}."
                )
        if self.stats is None:
            needy = [
                name for name in self.data
                if self._spec_for(name).method in ("lda", "logreg")
            ]
            if needy:
                raise ValueError(
                    f"probe(s) {needy} use a method that requires ambient activation "
                    "statistics; supply stats (an ActivationStats or a StatsSpec)."
                )

    def _spec_for(self, name: str) -> ProbeFitSpec:
        """The effective spec for one probe name."""
        return self.spec[name] if isinstance(self.spec, Mapping) else self.spec

    @property
    def names(self) -> tuple[str, ...]:
        """The probe names, available before fitting so routing rules validate at construction."""
        return tuple(self.data)

    def fit(self, model: PreTrainedModel | None, tokenizer: PreTrainedTokenizerBase, session=None) -> "ProbeSet":
        """Fit the recipe on `model`, resolving a `StatsSpec` on it first.

        Args:
            model: Model the probes are fitted on.
            tokenizer: Tokenizer for rendering and encoding the pairs.

        Returns:
            The fitted `ProbeSet`.
        """
        stats = self.stats
        if isinstance(stats, StatsSpec):
            stats = stats.estimate(model, tokenizer, session=session)
        return ProbeSet.fit(
            model,
            tokenizer,
            data=self.data,
            spec=self.spec,
            stats=stats,
            calibration_data=self.calibration_data,
            session=session,
        )


def _decoder_layer_names(model: PreTrainedModel) -> list[str]:
    """Dotted module paths of the decoder layers, for llama-style and GPT-2-style models.

    Raises:
        ValueError: If the model architecture is not recognized.
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return [f"model.layers.{i}" for i in range(len(model.model.layers))]
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return [f"transformer.h.{i}" for i in range(len(model.transformer.h))]
    raise ValueError(
        f"Unrecognized model architecture {type(model).__name__}; expected llama-style "
        "(model.layers) or GPT-2-style (transformer.h) decoder layers."
    )


class ProbeSet:
    """Named probes scored on a batch of prompts in one read-only forward.

    All probes must share one `model_type` and one capture `location`, since a single forward
    supplies every probe's features. `read()` registers forward pre-hooks on the union of the
    probes' layers (the `layer_input` boundary), runs one forward, and returns per-probe signed
    scores and decisions. The call is self-contained, i.e., it registers no standing hooks and
    never edits hidden states.

    Args:
        probes: Mapping from probe name to a fitted `Probe`.

    Attributes:
        probes: The probes, keyed by name.
        names: The probe names, in mapping order.
        layer_ids: Sorted union of the probes' layers.
        latest: The most recent `ProbeReadings`, overwritten by each `read()` call; None
            before the first. Diagnostics only.

    Raises:
        ValueError: If `probes` is empty, the probes disagree on `model_type` or `location`, or
            two probes record different model fingerprints (probes with no recorded fingerprint
            mix freely).
    """

    def __init__(self, probes: Mapping[str, Probe]):
        if not probes:
            raise ValueError("ProbeSet requires at least one probe.")
        self.probes: dict[str, Probe] = dict(probes)

        model_types = {probe.model_type for probe in self.probes.values()}
        if len(model_types) > 1:
            raise ValueError(
                f"probes disagree on model_type ({sorted(model_types)}); one read() capture "
                "cannot serve probes fitted on different model families."
            )
        locations = {probe.location for probe in self.probes.values()}
        if len(locations) > 1:
            raise ValueError(
                f"probes disagree on location ({sorted(locations)}); one read() capture "
                "cannot serve probes fitted at different boundaries."
            )
        fingerprints = {
            probe.meta["model_fingerprint"]
            for probe in self.probes.values()
            if probe.meta.get("model_fingerprint") is not None
        }
        if len(fingerprints) > 1:
            raise ValueError(
                f"probes record different model fingerprints ({sorted(fingerprints)}); "
                "mixed-model probes cannot share one read() capture."
            )

        self.names: tuple[str, ...] = tuple(self.probes)
        self.layer_ids: list[int] = sorted({lid for probe in self.probes.values() for lid in probe.layer_ids})
        self.latest: ProbeReadings | None = None

    @property
    def model_type(self) -> str:
        """The shared `model_type` of the probes."""
        return next(iter(self.probes.values())).model_type

    @property
    def location(self) -> str:
        """The shared capture boundary of the probes."""
        return next(iter(self.probes.values())).location

    @classmethod
    def fit(
        cls,
        model: PreTrainedModel | None,
        tokenizer: PreTrainedTokenizerBase,
        *,
        data: Mapping[str, ContrastivePairs],
        spec: ProbeFitSpec | Mapping[str, ProbeFitSpec] | None = None,
        stats: ActivationStats | None = None,
        calibration_data: Mapping[str, ContrastivePairs] | None = None,
        session=None,
    ) -> "ProbeSet":
        """Fit one probe per name and return the set.

        Args:
            model: Model the probes are fitted on.
            tokenizer: Tokenizer for rendering and encoding the pairs.
            data: Mapping from probe name to the contrastive pairs it is fitted on.
            spec: A shared `ProbeFitSpec` (None uses the defaults), or a mapping from probe name
                to a per-name spec.
            stats: Ambient activation statistics for methods that require them.
            calibration_data: Optional mapping from probe name to calibration pairs.

        Returns:
            The fitted `ProbeSet`.

        Raises:
            ValueError: If `data` is empty, a spec mapping misses a name, or `fit_probe` raises
                for any probe.
        """
        if not data:
            raise ValueError("data must map at least one probe name to contrastive pairs.")
        if isinstance(spec, Mapping):
            missing = [name for name in data if name not in spec]
            if missing:
                raise ValueError(f"spec mapping is missing entries for probe(s) {missing}.")

        probes: dict[str, Probe] = {}
        for name, pairs in data.items():
            probe_spec = spec[name] if isinstance(spec, Mapping) else (spec or ProbeFitSpec())
            probes[name] = fit_probe(
                model,
                tokenizer,
                data=pairs,
                spec=probe_spec,
                stats=stats,
                calibration_data=(calibration_data or {}).get(name),
                session=session,
            )
        return cls(probes)

    def read(
        self,
        model: PreTrainedModel | None,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        session=None,
    ) -> ProbeReadings:
        """Score a batch of prompts against every probe in one read-only forward.

        Registers one forward pre-hook per layer in the union (the `layer_input` boundary, the
        only boundary `read()` serves), runs a single forward under `torch.no_grad()` with
        `use_cache=False` wrapped in `auxiliary_pass(aligned=True)`, removes the hooks, then
        pools each probe's layers per its `pooling` with the attention mask and applies its
        decision function. The auxiliary marking means co-resident state controls' condition
        scorers, gates, and position counters ignore the pass, while `"all"`-scoped behavior
        transforms still apply, so scores measure the stream as deployed. A set fitted at
        `"layer_output"` raises, since a pre-hook capture would score features one block away
        from where those probes were fitted; score such probes offline via
        `layerwise_tokenwise_hidden` and `Probe.score_hidden`.

        Args:
            model: The model to read (must match the probes' `model_type`).
            input_ids: Prompt token ids of shape `[B, T]` (a 1-D tensor is treated as one row).
            attention_mask: Prompt attention mask matching `input_ids` (1 at real tokens). When
                None, every position is treated as real.

        Returns:
            A `ProbeReadings` with per-probe signed scores and decisions, also stashed on `latest`.

        Raises:
            ValueError: If the model's `model_type` does not match the probes', the set's
                `location` is not `"layer_input"`, a probe layer is out of range, or the
                architecture is unrecognized.
        """
        if model is not None:
            live_model_type = getattr(model.config, "model_type", "unknown")
            if live_model_type != self.model_type:
                raise ValueError(
                    f"ProbeSet was fitted on model_type {self.model_type!r} but read() received "
                    f"{live_model_type!r}."
                )
        elif session is None:
            raise ValueError("ProbeSet.read() requires a live model or a capture-capable session.")
        if self.location != "layer_input":
            raise ValueError(
                f"ProbeSet.read() serves the 'layer_input' boundary, but this set was fitted at "
                f"{self.location!r}; refit with location='layer_input', or score offline via "
                "layerwise_tokenwise_hidden and Probe.score_hidden."
            )

        device = next(model.parameters()).device if model is not None else torch.device("cpu")
        ids = input_ids if isinstance(input_ids, torch.Tensor) else torch.as_tensor(input_ids, dtype=torch.long)
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)
        ids = ids.to(device)

        if attention_mask is not None:
            mask = attention_mask if isinstance(attention_mask, torch.Tensor) else torch.as_tensor(attention_mask)
            if mask.ndim == 1:
                mask = mask.unsqueeze(0)
            mask = mask.to(device)
        else:
            mask = torch.ones_like(ids)

        if model is None:
            return self._read_via_session(session, ids, mask)

        layer_names = _decoder_layer_names(model)
        for lid in self.layer_ids:
            if not 0 <= lid < len(layer_names):
                raise ValueError(f"probe layer {lid} out of range [0, {len(layer_names)}).")

        captured: dict[int, torch.Tensor] = {}

        def _pre_capture(layer_id: int):
            def _pre_hook(module, input_args, input_kwargs):
                hidden = input_args[0] if input_args else input_kwargs.get("hidden_states")
                if hidden is not None:
                    captured[layer_id] = hidden.detach().to("cpu", torch.float32)
                return input_args, input_kwargs

            return _pre_hook

        handles = []
        try:
            for lid in self.layer_ids:
                module = model.get_submodule(layer_names[lid])
                handles.append(module.register_forward_pre_hook(_pre_capture(lid), with_kwargs=True))
            with torch.no_grad(), auxiliary_pass(aligned=True):
                model(input_ids=ids, attention_mask=mask, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()

        cpu_mask = mask.to("cpu")
        scores: dict[str, torch.Tensor] = {}
        decisions: dict[str, torch.Tensor] = {}
        for name, probe in self.probes.items():
            features = {
                lid: aggregate_condition_hidden(captured[lid], probe.pooling, attention_mask=cpu_mask)
                for lid in probe.layer_ids
            }
            probe_scores = probe.decision_function(features)
            scores[name] = probe_scores
            decisions[name] = probe_scores >= 0

        readings = ProbeReadings(scores=scores, decisions=decisions)
        self.latest = readings
        return readings

    def _read_via_session(self, session, ids: torch.Tensor, mask: torch.Tensor) -> ProbeReadings:
        """Score through a capture-capable session's `capture` at the layer-input boundary."""
        from aisteer360.algorithms.core.execution.payloads import PreparedPrompt

        prompts = [
            PreparedPrompt.from_token_ids(ids[index:index + 1], mask[index:index + 1])
            for index in range(ids.size(0))
        ]
        result = session.capture(
            prompts, layers=list(self.layer_ids), mode="all_tokens", location="layer_input"
        )
        cpu_mask = result.attention_mask
        scores: dict[str, torch.Tensor] = {}
        decisions: dict[str, torch.Tensor] = {}
        for name, probe in self.probes.items():
            features = {
                lid: aggregate_condition_hidden(
                    result.hidden[lid].to(torch.float32), probe.pooling, attention_mask=cpu_mask
                )
                for lid in probe.layer_ids
            }
            probe_scores = probe.decision_function(features)
            scores[name] = probe_scores
            decisions[name] = probe_scores >= 0
        readings = ProbeReadings(scores=scores, decisions=decisions)
        self.latest = readings
        return readings

    def summary(self) -> dict[str, dict]:
        """Per-probe diagnostic table.

        Returns:
            Mapping from probe name to a dict with keys `"layer_ids"`, `"method"`, `"bias"`,
            `"calibration"`, `"f1"`, `"model_fingerprint"`, and `"stats_fingerprint"` (all but
            `"layer_ids"` and `"bias"` are None when the probe's `meta` does not record them).
        """
        return {
            name: {
                "layer_ids": list(probe.layer_ids),
                "method": probe.meta.get("method"),
                "bias": probe.bias,
                "calibration": probe.meta.get("calibration"),
                "f1": (probe.meta.get("calibration") or {}).get("f1"),
                "model_fingerprint": probe.meta.get("model_fingerprint"),
                "stats_fingerprint": probe.meta.get("stats_fingerprint"),
            }
            for name, probe in self.probes.items()
        }
