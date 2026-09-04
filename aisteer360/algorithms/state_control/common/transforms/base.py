"""Base class for hidden-state transforms."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

import torch

if TYPE_CHECKING:
    from ..specs import WireForm
    from .context import TransformContext


class BaseTransform(ABC):
    """Applies a modification to hidden states at a given layer.

    All transforms receive:
        - hidden_states shaped [B, T, H]
        - the layer_id so the transform can index per-layer artifacts
        - a token_mask shaped [B, T] (True where the transform should apply)

    Transforms MUST NOT modify hidden_states in-place if the original tensor
    is needed later (e.g., for norm-preserving wrappers); return a new tensor.

    Artifact-carrying transforms take their artifact as the first positional argument, typed
    `SteeringVector | Mapping[int, Tensor] | ArtifactSource` and required. A concrete artifact
    binds the transform immediately (validated at `__init__`); an `ArtifactSource` leaves it
    unbound until `bind(ctx)` resolves the source (validated then). Subclass recipe:

        - store the concrete artifact and set `is_bound=True`, or store the source and set
          `is_bound=False`;
        - override `bind(ctx)` to return a freshly constructed bound instance, never mutating
          `self` and never using `copy.copy`, so that derived caches stay valid;
        - override `covered_layer_ids` to report the layers the (bound) artifact covers;
        - call `self._require_bound()` as the first line of `apply`.

    Transforms with no steering artifact keep the defaults (always bound, no coverage).

    Class attributes:
        wire_kind: The permanent wire kind name this class serializes to, or None when the
            class has no wire form. Wire names mirror toolkit class names, so the mapping is
            definitional rather than maintained.
        is_modifier: True for wrapper transforms that hold an `inner` transform and serialize
            as a wire modifier rather than a transform kind.
    """

    wire_kind: ClassVar[str | None] = None
    is_modifier: ClassVar[bool] = False

    @property
    def artifact_meta(self) -> dict | None:
        """Provenance metadata of the transform's steering artifact, or None.

        Populated when the artifact was supplied or resolved as a `SteeringVector` carrying
        `meta` (fit fingerprints); consumers cross-check it against a serving engine's model
        identity. Wrappers delegate to their inner transform. The default returns None.
        """
        return None

    @abstractmethod
    def apply(
        self,
        hidden_states: torch.Tensor,
        *,
        layer_id: int,
        token_mask: torch.BoolTensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Apply the transform and return modified hidden states.

        Args:
            hidden_states: Shape [B, T, H].
            layer_id: Which layer this is being applied at.
            token_mask: Shape [B, T]. True at positions to modify.
            **kwargs: Transform-specific extra arguments.

        Returns:
            Modified hidden states, same shape as input.
        """
        ...

    @property
    def is_bound(self) -> bool:
        """True when the transform can be applied as-is (artifact concrete).

        Default True: transforms that take no steering artifact are always bound.
        """
        return True

    @property
    def source(self):
        """The unresolved `ArtifactSource` this transform carries, or None when concrete."""
        return getattr(self, "_source", None)

    def bind(self, ctx: "TransformContext") -> "BaseTransform":
        """Return a fully-bound transform for this context.

        Contract:

            - MUST NOT mutate `self`; instances and sources are shared across adapters and
              `Benchmark`/`ControlSpec` grid points, whose params objects are reused per point.
            - Returns `self` when already bound (idempotent).
            - When source-carrying, returns a NEW instance of the same class constructed with
              `ctx.resolve(self._source)` and all hyperparameters copied, built by fresh
              construction rather than `copy.copy`.

        Default: returns self.
        """
        return self

    def _require_bound(self) -> None:
        """Raise a clear `RuntimeError` if this transform is unbound.

        Called as the first line of every artifact-consuming `apply`.
        """
        if not self.is_bound:
            raise RuntimeError(
                f"{type(self).__name__} was constructed with an ArtifactSource and is unbound; pass "
                f"it through a control that resolves and binds it during steer() (ActivationAdapter, "
                f"or CAST via behavior_transform)."
            )

    @property
    def covered_layer_ids(self) -> set[int] | None:
        """Layers this transform can act on; None = unknown (opts out of adapter validation).

        Artifact transforms report `set(directions.keys())` when bound and None when unbound;
        wrappers delegate to their inner transform.
        """
        return None

    def export(self, layer_id: int) -> "WireForm | None":
        """This configuration's wire form at `layer_id`, or None when the configuration is
        not expressible in the wire vocabulary.

        Exportability is a property of a configuration, not a class; an artifact whose shape
        has no wire semantics (e.g. a positional direction) returns None even though the
        class names a `wire_kind`. The default returns None.
        """
        return None

    def wire_plan(self) -> str | None:
        """The wire transform kind this configuration serializes to, or None when hook-only.

        Readable on the unbound form: a source-carrying transform consults its source's
        declared shape rather than resolving it. The default returns the class `wire_kind`.
        """
        return type(self).wire_kind

    def modifier_wire_kind(self, core_kind: str) -> str | None:
        """The wire modifier kind this wrapper contributes over a core transform kind, or
        None when the combination has no wire form.

        Meaningful only on wrapper transforms (`is_modifier` True). The default returns the
        class `wire_kind`.
        """
        return type(self).wire_kind

    def export_modifier(self, layer_id: int) -> "WireForm | None":
        """The wrapper's wire modifier form at `layer_id`, or None to contribute no modifier
        at that layer.

        Meaningful only on wrapper transforms (`is_modifier` True); kind-level
        inexpressibility is reported by `modifier_wire_kind`. The default returns None.
        """
        return None


def unwrap_modifiers(transform: BaseTransform) -> tuple[BaseTransform, tuple[BaseTransform, ...]]:
    """Split a possibly wrapper-chained transform into its core and its modifiers.

    Args:
        transform: The transform, possibly wrapped in modifier transforms.

    Returns:
        The core transform and the modifiers in application order, innermost wrapper first,
        matching the wire interpreter's composition order.
    """
    wrappers: list[BaseTransform] = []
    current = transform
    while type(current).is_modifier:
        wrappers.append(current)
        current = current.inner
    return current, tuple(reversed(wrappers))
