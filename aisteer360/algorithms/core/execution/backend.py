"""The backend seam: `Backend`, `SteeringSession`, the backend registry, and the
session wrapper drivers roll out through.

A `Backend` owns a loaded model, engine, or connection pool and its lifecycle, advertises
capabilities, and creates sessions. A `SteeringSession` is one logical operation's scope on a
backend and the unit of concurrency; sessions register hooks, execute items, and serve capture.
The registry is a fixed mapping over the core-owned backend kinds; backend modules are imported
on first resolution, so `core` carries no module-level dependency on `aisteer360.backends`.
"""
from abc import ABC, abstractmethod
from collections.abc import Sequence
from importlib import import_module
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

import torch

from aisteer360.algorithms.core.execution.contracts import (
    BackendCapabilities,
    Capability,
    CaptureKinds,
    InterventionKinds,
    ProcessorKinds,
)
from aisteer360.algorithms.core.execution.params import GenerationParams
from aisteer360.algorithms.core.execution.payloads import (
    CaptureResult,
    GenerationItem,
    ItemResult,
    ModelFacts,
    PreparedPrompt,
    ScoringItem,
)
from aisteer360.algorithms.core.execution.spec import BackendSpec


@runtime_checkable
class SteeringSession(Protocol):
    """One logical operation's scope on a backend; the unit of concurrency.

    A session is opened per logical operation (one generation fan-out, one scoring call, one
    steer-phase fit) on every backend. The `Backend` owns the loaded model or engine; a session
    holds only per-operation state. Session-contract facts, provided by every backend and
    therefore never capability atoms, include token-id prompts, stop rules, minimum tokens,
    multiple candidates, seeded sampling, prompt-logprob scoring, and the model layout.
    """

    @property
    def layout(self) -> ModelFacts:
        """Structural facts about the session's model."""
        ...

    def generate(
        self,
        items: Sequence[GenerationItem],
        params: GenerationParams,
    ) -> list[ItemResult]:
        """Generate one result per item, in item order."""
        ...

    def score(
        self,
        items: Sequence[ScoringItem],
        params: GenerationParams,
    ) -> torch.Tensor:
        """Teacher-forced log-probabilities of each item's reference tokens, shape
        `[num_items, ref_len]`."""
        ...

    def capture(
        self,
        prompts: list[PreparedPrompt],
        layers: list[int],
        mode: Literal["all_tokens", "last_token"],
        location: Literal["layer_output", "layer_input"] = "layer_output",
    ) -> CaptureResult:
        """Capture hidden states for `prompts` at `layers`; requires
        `Capability.HIDDEN_CAPTURE`."""
        ...


class SteeredSession:
    """A `SteeringSession` whose `generate` and `score` inject a generation's control entries
    into every item, so a driver's rollouts carry the pipeline's steering without the driver
    knowing entries exist.

    The pipeline builds one wrapper per logical generation and hands it to the decoding
    driver. On an in-process backend the injected tuple is empty, since the session hosts the
    generation's hooks ambiently for the span of the driver's decode; on spec-consuming
    backends the injected entries are the generation's lowered interventions with
    prompt-relative scopes rewritten to absolute positions at the generation's original prompt
    boundary, so re-prefilled continuation tokens are steered at their original positions.

    Attributes:
        inner: The wrapped backend session.
        state_entries: Entries injected ahead of each item's own.
    """

    def __init__(self, inner, state_entries: tuple = ()):
        self.inner = inner
        self.state_entries = tuple(state_entries)

    @property
    def layout(self):
        return self.inner.layout

    @property
    def tokenizer(self):
        return getattr(self.inner, "tokenizer", None)

    def _inject(self, item):
        if not self.state_entries:
            return item
        import dataclasses

        return dataclasses.replace(
            item, state_entries=self.state_entries + tuple(item.state_entries)
        )

    def generate(self, items, params):
        """Generate with the wrapper's entries injected into every item."""
        return self.inner.generate([self._inject(item) for item in items], params)

    def score(self, items, params):
        """Score with the wrapper's entries injected into every item."""
        return self.inner.score([self._inject(item) for item in items], params)

    def capture(self, prompts, layers, mode, location="layer_output"):
        """Capture through the wrapped session, unsteered."""
        return self.inner.capture(prompts, layers, mode, location=location)


class Backend(ABC):
    """A backend owns a loaded model, engine, or connection pool and its lifecycle, advertises
    capabilities, and creates sessions.

    Long-lived consumers hold backends in a cache keyed by `BackendSpec` so configurations
    differing only in per-request steering share one resource.

    Attributes:
        spec: The frozen identity of this backend configuration.
    """

    spec: BackendSpec

    @classmethod
    @abstractmethod
    def capabilities_for_spec(cls, spec: BackendSpec) -> BackendCapabilities:
        """The capability advertisement implied by `spec`, computable without constructing the
        backend. Constructed backends advertise the same sets, verified against the live
        resource where a discovery surface exists."""

    @abstractmethod
    def open_session(self) -> SteeringSession:
        """Open a session for one logical operation."""

    @property
    def capabilities(self) -> frozenset[Capability]:
        """The advertised capability atoms."""
        return self.capabilities_for_spec(self.spec).atoms

    @property
    def intervention_kinds(self) -> InterventionKinds | None:
        """The advertised intervention kinds, when `Capability.INTERVENTION_SPECS` is present."""
        return self.capabilities_for_spec(self.spec).intervention_kinds

    @property
    def processor_kinds(self) -> ProcessorKinds | None:
        """The advertised processor kinds, when `Capability.PER_STEP_LOGIT_SPECS` is present."""
        return self.capabilities_for_spec(self.spec).processor_kinds

    @property
    def capture_kinds(self) -> CaptureKinds | None:
        """The advertised capture kinds, when `Capability.HIDDEN_CAPTURE` is present."""
        return self.capabilities_for_spec(self.spec).capture_kinds

    def stage_artifacts(self, payloads) -> None:
        """Make each content-addressed artifact available to the execution side.

        Called by the pipeline at the end of `steer()` with the tensor payloads of every
        lowered intervention spec, keyed by content-addressed artifact id. Staging is
        idempotent: an artifact that already exists at the destination is success. The
        in-process backend keeps live tensors and needs no staging; engine-backed backends
        write into the registry the serving engine reads.

        Args:
            payloads: Mapping from artifact id to a name-to-tensor mapping.
        """
        return None

    def release(self) -> None:
        """Free the resources this backend owns beyond what the caller owns.

        The default is a no-op. Engine-owning backends override this to shut their engines down
        deterministically. Release is idempotent.
        """
        return None


if TYPE_CHECKING:
    from aisteer360.algorithms.core.execution.backend import Backend

_BACKEND_CLASSES: dict[str, tuple[str, str]] = {
    "huggingface": ("aisteer360.backends.huggingface", "HFBackend"),
    "vllm": ("aisteer360.backends.vllm", "VLLMBackend"),
    "vllm-serve": ("aisteer360.backends.vllm", "VLLMServeBackend"),
}


def resolve_backend_class(spec: BackendSpec) -> "type[Backend]":
    """The backend class registered for `spec.kind`.

    Args:
        spec: The backend spec to resolve.

    Returns:
        The backend class. Importing the class does not require the backend's optional
        dependencies; constructing an instance may.

    Raises:
        ValueError: If no backend class is registered for the spec's kind.
    """
    entry = _BACKEND_CLASSES.get(spec.kind)
    if entry is None:
        raise ValueError(f"No backend class is registered for kind {spec.kind!r}.")
    module_name, attribute = entry
    return getattr(import_module(module_name), attribute)


def capabilities_for_spec(spec: BackendSpec) -> BackendCapabilities:
    """The capability advertisement implied by `spec`, without constructing a backend."""
    return resolve_backend_class(spec).capabilities_for_spec(spec)
