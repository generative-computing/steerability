"""Gating for state control interventions.

A gate is an operation on evidence decided by a rule. `Evidence` names where hidden states are
read (condition layers and pooling) and the `Readout` that turns pooled hidden states into one
value per row; a `Rule` is a pure decision over the per-layer values; `Gate` is the single
stateful shell that collects evidence during a generation, freezes the decision when the rule
reports complete, and answers per-row open/closed queries from behavior hooks. Provenance
(probes, contrastive fits, manual weights) produces readouts; it never appears in gate types.

Wire expressibility is a per-component property: a readout or rule with a `wire_kind` exports
its wire form, and `Gate.export` composes them into the wire gate object. A `CallableReadout`
has no wire form and keeps the intervention in process.

This module also holds the score math shared with `selectors/condition_point.py`
(`rank_one_projector`, `projected_cosine_similarity_tensor`), so selector calibration and
runtime scoring provably agree.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, ClassVar, Literal, Mapping, Protocol, runtime_checkable

import torch
import torch.nn.functional as F

from aisteer360.algorithms.core.internals.probes.probe import Probe

from .steering_vector import SteeringVector

if TYPE_CHECKING:
    from .specs import WireForm


def rank_one_projector(direction: torch.Tensor) -> torch.Tensor:
    """Build the rank-one projector `cc^T / (c^T c)` for a direction.

    Args:
        direction: Shape `[H]`.

    Returns:
        Projection matrix of shape `[H, H]`.
    """
    if direction.ndim != 1:
        raise ValueError(f"direction must be 1-D [H]; got shape {tuple(direction.shape)}.")
    c = direction.float()
    return torch.outer(c, c) / (c @ c + 1e-8)


@torch.no_grad()
def projected_cosine_similarity_tensor(
    hidden: torch.Tensor,
    projector: torch.Tensor,
) -> torch.Tensor:
    """Cosine similarity between rows of `hidden` and their tanh'd projections, one score per row.

    Args:
        hidden: Shape `[..., H]`.
        projector: Shape `[H, H]` outer-product projection matrix.

    Returns:
        Scores of shape `[...]` (float32).
    """
    hidden = hidden.float()
    projector = projector.float()
    projected = torch.tanh(hidden @ projector)  # projector is symmetric
    numerator = (hidden * projected).sum(dim=-1)
    denominator = hidden.norm(dim=-1) * projected.norm(dim=-1) + 1e-8
    return numerator / denominator


@torch.no_grad()
def projected_cosine_similarity(
    hidden_state: torch.Tensor,
    projector: torch.Tensor,
) -> float:
    """Projected-cosine score of one aggregated hidden state.

    Projects the hidden state through the condition subspace projector, applies tanh, then
    computes cosine similarity with the original.

    Args:
        hidden_state: Shape `[H]`, an aggregated hidden state.
        projector: Shape `[H, H]` outer-product projection matrix.

    Returns:
        Cosine similarity as a float.
    """
    score = projected_cosine_similarity_tensor(hidden_state.unsqueeze(0), projector)[0]
    return float(score.item())


def _extract_directions(
    artifact: SteeringVector | Mapping[int, torch.Tensor], who: str
) -> dict[int, torch.Tensor]:
    """Validate and extract a concrete per-layer directions mapping from an artifact."""
    if isinstance(artifact, SteeringVector):
        directions = dict(artifact.directions)
    elif isinstance(artifact, Mapping):
        directions = dict(artifact)
    else:
        raise TypeError(
            f"{who} expects a concrete SteeringVector or Mapping[int, Tensor]; "
            f"got {type(artifact).__name__}."
        )
    if not all(isinstance(v, torch.Tensor) for v in directions.values()):
        raise TypeError(f"{who} directions must be torch.Tensor values.")
    return {int(lid): direction for lid, direction in directions.items()}


def _feature_row(direction: torch.Tensor) -> torch.Tensor:
    """The `[H]` feature row of a direction; `[K, H]` artifacts use row 0."""
    if direction.ndim == 2:
        return direction[0]
    return direction


@runtime_checkable
class Readout(Protocol):
    """Per-layer map from pooled hidden state to a per-row value.

    Callable as `(pooled [B, H], layer_id) -> Tensor [B]` (float32, cpu). Readouts are
    stateless per-layer functions of pooled hidden states; pooling is declared once, on
    `Evidence`. `location` and `model_fingerprint` are validated by `Intervention.bind` when
    non-None. `export(layer_ids)` returns the wire form (kind plus row-aligned tensors) or
    None; `wire_kind` is the class-level kind name, or None for readouts with no wire form.
    """

    wire_kind: ClassVar[str | None]
    location: str | None
    model_fingerprint: str | None

    def __call__(self, pooled: torch.Tensor, layer_id: int) -> torch.Tensor: ...

    def export(self, layer_ids: tuple[int, ...]) -> "WireForm | None": ...


class AffineReadout:
    """Signed linear readout: `value = weights[layer] . pooled`.

    Constructed from a per-layer weight mapping or a `SteeringVector`; probe-backed gates build
    one from the probe's weights via `gate_from_probe`. Returns zeros for layers absent from
    the weights. The wire form stacks the weight rows aligned with the evidence layer order.

    Args:
        weights: Per-layer weight vectors (`[H]`, or `[K, H]` using row 0), as a
            `SteeringVector` or mapping.
        location: The boundary the weights were fitted at, or None to skip the boundary check.
        model_fingerprint: The fitted model's identity, or None to skip the identity check.
    """

    wire_kind: ClassVar[str | None] = "affine"

    def __init__(
        self,
        weights: SteeringVector | Mapping[int, torch.Tensor],
        location: str | None = None,
        model_fingerprint: str | None = None,
    ):
        self.weights = {
            lid: _feature_row(direction).detach().reshape(-1).to(torch.float32)
            for lid, direction in _extract_directions(weights, type(self).__name__).items()
        }
        self.location = location
        self.model_fingerprint = model_fingerprint

    @torch.no_grad()
    def __call__(self, pooled: torch.Tensor, layer_id: int) -> torch.Tensor:
        """Return `[B]` values `weights[layer_id] . pooled` (zeros if the layer is absent)."""
        weights = self.weights.get(layer_id)
        if weights is None:
            return torch.zeros(pooled.size(0), dtype=torch.float32)
        return (pooled.to(torch.float32) @ weights.to(pooled.device)).float().cpu()

    def export(self, layer_ids: tuple[int, ...]) -> "WireForm | None":
        """The `affine` wire form: `weights [L, H]` row-aligned with `layer_ids`."""
        from .specs import WireForm

        if any(lid not in self.weights for lid in layer_ids):
            return None
        stacked = torch.stack([self.weights[lid] for lid in layer_ids])
        return WireForm(kind="affine", tensors={"weights": stacked})


class CosineReadout:
    """Signed cosine similarity between the pooled state and a per-layer direction.

    `[K, H]` artifacts use row 0. Returns zeros for layers absent from the directions. The
    score is signed: for a mean-difference direction, positives score high, negatives score
    low, and unrelated content scores near zero, so a `"ge"` threshold fails closed on
    out-of-distribution inputs. Prefer this readout for topic or domain gates whose calibration
    negatives cannot cover the deployment input space; signed scores can be negative, so pair
    it with a threshold range that admits negative values, e.g. `(-1.0, 1.0)`.

    Args:
        directions: Per-layer directions, a `SteeringVector` or `Mapping[int, torch.Tensor]`.
        location: The boundary the directions were fitted at, or None to skip the check.
        model_fingerprint: The fitted model's identity, or None to skip the check.

    Reference:

    - "Steering Llama 2 via Contrastive Activation Addition"
      Nina Panickssery, Nick Gabrieli, Julian Schulz, Meg Tong, Evan Hubinger, Alexander Matt Turner
      [https://arxiv.org/abs/2312.06681](https://arxiv.org/abs/2312.06681)
    """

    wire_kind: ClassVar[str | None] = "cosine"

    def __init__(
        self,
        directions: SteeringVector | Mapping[int, torch.Tensor],
        location: str | None = None,
        model_fingerprint: str | None = None,
    ):
        self.directions = _extract_directions(directions, type(self).__name__)
        self.location = location
        self.model_fingerprint = model_fingerprint

    @torch.no_grad()
    def __call__(self, pooled: torch.Tensor, layer_id: int) -> torch.Tensor:
        """Return `[B]` signed cosine values for `layer_id` (zeros if the layer is absent)."""
        direction = self.directions.get(layer_id)
        if direction is None:
            return torch.zeros(pooled.size(0), dtype=torch.float32)
        direction = _feature_row(direction).to(dtype=pooled.dtype, device=pooled.device)
        return F.cosine_similarity(pooled, direction.unsqueeze(0), dim=-1).float().cpu()

    def export(self, layer_ids: tuple[int, ...]) -> "WireForm | None":
        """The `cosine` wire form: `directions [L, H]` row-aligned with `layer_ids`."""
        from .specs import WireForm

        if any(lid not in self.directions for lid in layer_ids):
            return None
        stacked = torch.stack(
            [_feature_row(self.directions[lid]).reshape(-1).to(torch.float32) for lid in layer_ids]
        )
        return WireForm(kind="cosine", tensors={"directions": stacked})


class ProjectedCosineReadout:
    """Projected-cosine score: `cosine(pooled, tanh(pooled @ P_layer))` with
    `P = dd^T / (d . d)`.

    `[K, H]` artifacts use row 0. Rank-one projectors are built lazily and cached per
    `(layer_id, device)`. Returns zeros for layers absent from the directions. The score is
    approximately `|cos(pooled, d)|` (alignment with the line spanned by the direction,
    erasing which side of the direction a state lies on).

    Args:
        directions: Per-layer condition directions, a `SteeringVector` or
            `Mapping[int, torch.Tensor]`.
        location: The boundary the directions were fitted at, or None to skip the check.
        model_fingerprint: The fitted model's identity, or None to skip the check.

    Reference:

    - "Programming Refusal with Conditional Activation Steering"
      Bruce W. Lee, Inkit Padhi, Karthikeyan Natesan Ramamurthy, Erik Miehling, Pierre Dognin,
      Manish Nagireddy, Amit Dhurandhar
      [https://arxiv.org/abs/2409.05907](https://arxiv.org/abs/2409.05907)
    """

    wire_kind: ClassVar[str | None] = "projected_cosine"

    def __init__(
        self,
        directions: SteeringVector | Mapping[int, torch.Tensor],
        location: str | None = None,
        model_fingerprint: str | None = None,
    ):
        self.directions = _extract_directions(directions, type(self).__name__)
        self.location = location
        self.model_fingerprint = model_fingerprint
        self._projector_cache: dict[tuple[int, torch.device], torch.Tensor] = {}

    def _projector(self, layer_id: int, device: torch.device) -> torch.Tensor | None:
        """Cached `[H, H]` rank-one projector for a layer, or None when the layer is absent."""
        key = (layer_id, device)
        cached = self._projector_cache.get(key)
        if cached is None:
            direction = self.directions.get(layer_id)
            if direction is None:
                return None
            cached = rank_one_projector(_feature_row(direction).to(device=device)).to(device)
            self._projector_cache[key] = cached
        return cached

    @torch.no_grad()
    def __call__(self, pooled: torch.Tensor, layer_id: int) -> torch.Tensor:
        """Return `[B]` projected-cosine values for `layer_id` (zeros if the layer is absent)."""
        projector = self._projector(layer_id, pooled.device)
        if projector is None:
            return torch.zeros(pooled.size(0), dtype=torch.float32)
        return projected_cosine_similarity_tensor(pooled, projector.to(pooled.dtype)).float().cpu()

    def export(self, layer_ids: tuple[int, ...]) -> "WireForm | None":
        """The `projected_cosine` wire form: `directions [L, H]` row-aligned with `layer_ids`."""
        from .specs import WireForm

        if any(lid not in self.directions for lid in layer_ids):
            return None
        stacked = torch.stack(
            [_feature_row(self.directions[lid]).reshape(-1).to(torch.float32) for lid in layer_ids]
        )
        return WireForm(kind="projected_cosine", tensors={"directions": stacked})


class CallableReadout:
    """Escape hatch wrapping any `(pooled [B, H], layer_id) -> Tensor [B]` callable.

    Has no wire form, so a gate built on it never lowers; support verdicts attribute the
    in-process-only status to the readout.

    Args:
        fn: The readout callable.
        location: The boundary the callable expects features at, or None to skip the check.
        model_fingerprint: The fitted model's identity, or None to skip the check.
    """

    wire_kind: ClassVar[str | None] = None

    def __init__(
        self,
        fn: Callable[[torch.Tensor, int], torch.Tensor],
        location: str | None = None,
        model_fingerprint: str | None = None,
    ):
        self.fn = fn
        self.location = location
        self.model_fingerprint = model_fingerprint

    def __call__(self, pooled: torch.Tensor, layer_id: int) -> torch.Tensor:
        return self.fn(pooled, layer_id)

    def export(self, layer_ids: tuple[int, ...]) -> "WireForm | None":
        """None; an arbitrary callable has no wire form."""
        return None


@dataclass(frozen=True)
class Evidence:
    """Where gate evidence is read and how hidden states become values.

    Attributes:
        layer_ids: Condition layers (0-based decoder-layer indices at the intervention's
            boundary).
        readout: Per-layer map from pooled hidden states to per-row values.
        pooling: Token aggregation over real (non-pad) prompt positions: `"mean"` pools,
            `"last"` selects the last real token per row.
    """

    layer_ids: tuple[int, ...]
    readout: Readout
    pooling: Literal["mean", "last"] = "mean"

    def __post_init__(self):
        object.__setattr__(self, "layer_ids", tuple(int(lid) for lid in self.layer_ids))
        if not self.layer_ids:
            raise ValueError("Evidence requires at least one condition layer.")
        if self.pooling not in ("mean", "last"):
            raise ValueError(f"pooling must be 'mean' or 'last'; got {self.pooling!r}.")


@runtime_checkable
class Rule(Protocol):
    """Pure decision over per-layer evidence values. Stateless.

    `decide` maps the collected values (one `[num_rows]` tensor per layer) to a `[num_rows]`
    boolean row tensor; `is_complete` reports whether the collected evidence suffices to
    decide; `export()` returns the wire form or None; `wire_kind` is the class-level kind
    name, or None for rules with no wire form.
    """

    wire_kind: ClassVar[str | None]

    def decide(self, values: Mapping[int, torch.Tensor], num_rows: int) -> torch.BoolTensor: ...

    def is_complete(self, seen: frozenset[int], expected: frozenset[int]) -> bool: ...

    def export(self) -> "WireForm | None": ...


class SumThreshold:
    """Open where `sum over layers of values + bias >= 0` (ties open).

    Args:
        bias: Offset added to the summed values before the zero comparison.
    """

    wire_kind: ClassVar[str | None] = "sum_threshold"

    def __init__(self, bias: float = 0.0):
        self.bias = float(bias)

    def decide(self, values: Mapping[int, torch.Tensor], num_rows: int) -> torch.BoolTensor:
        """Per-row decision at the bias; all-closed when no values have arrived."""
        if not values:
            return torch.zeros(num_rows, dtype=torch.bool)
        total = torch.stack(list(values.values()), dim=0).sum(dim=0)
        return total + self.bias >= 0

    def is_complete(self, seen: frozenset[int], expected: frozenset[int]) -> bool:
        """True once every expected layer has reported."""
        return seen >= expected

    def export(self) -> "WireForm | None":
        """The `sum_threshold` wire form with the inline `bias` param."""
        from .specs import WireForm

        return WireForm(kind="sum_threshold", params={"bias": self.bias})


class PerKeyThreshold:
    """Per-layer threshold comparison, combined across layers with any/all.

    `"ge"` opens a layer's vote when its value is at or above the threshold; `"le"` when at or
    below. `aggregate="any"` opens a row when any layer passes for that row; `"all"` requires
    every layer.

    Args:
        threshold: Score threshold for the per-layer comparison.
        comparator: `"ge"` (value >= threshold) or `"le"` (value <= threshold).
        aggregate: `"any"` or `"all"` across layers.
    """

    wire_kind: ClassVar[str | None] = "per_key_threshold"

    def __init__(
        self,
        threshold: float,
        comparator: Literal["ge", "le"] = "ge",
        aggregate: Literal["any", "all"] = "any",
    ):
        if comparator not in ("ge", "le"):
            raise ValueError(f"comparator must be 'ge' or 'le'; got {comparator!r}.")
        if aggregate not in ("any", "all"):
            raise ValueError(f"aggregate must be 'any' or 'all'; got {aggregate!r}.")
        self.threshold = float(threshold)
        self.comparator = comparator
        self.aggregate = aggregate

    def decide(self, values: Mapping[int, torch.Tensor], num_rows: int) -> torch.BoolTensor:
        """Per-row decision across layers; all-closed when no values have arrived."""
        if not values:
            return torch.zeros(num_rows, dtype=torch.bool)
        stacked = torch.stack(list(values.values()), dim=0)  # [L, num_rows]
        if self.comparator == "ge":
            passed = stacked >= self.threshold
        else:
            passed = stacked <= self.threshold
        if self.aggregate == "any":
            return passed.any(dim=0)
        return passed.all(dim=0)

    def is_complete(self, seen: frozenset[int], expected: frozenset[int]) -> bool:
        """True once every expected layer has reported."""
        return seen >= expected

    def export(self) -> "WireForm | None":
        """The `per_key_threshold` wire form with inline `threshold`, `comparator`, and
        `aggregate` params."""
        from .specs import WireForm

        return WireForm(
            kind="per_key_threshold",
            params={
                "threshold": self.threshold,
                "comparator": self.comparator,
                "aggregate": self.aggregate,
            },
        )


class Gate:
    """The stateful shell around evidence and a rule; holds one decision per logical batch row.

    Lifecycle per generation call:

        1. `reset(num_rows)` clears state from the previous generation and sizes the gate to
           the logical batch (number of prompts, not the beam-expanded batch). Reset is
           idempotent: re-resetting an already-reset gate to the same size leaves it in the
           same cleared state, which shared-instance composition (one gate read by several
           interventions) relies on.
        2. `update(values, key=layer_id)` records one condition layer's per-row values as
           evidence arrives from condition hooks. When the rule first reports complete
           (typically once every evidence layer has reported on the prefill pass), the
           decision is computed once, frozen for the generation, and the stored values are
           dropped after copying into the diagnostics snapshot.
        3. `open_rows()` is queried by behavior hooks; all rows are closed before evidence.

    `is_ready()` reports True once the decision is frozen; the runtime stops condition scoring
    then, so the prompt is scored once and the decision holds. A rule whose `is_complete`
    never returns True re-scores every pass (in-process only; wire gates are prompt-decided by
    construction).

    Args:
        evidence: Where evidence is read and how hidden states become values.
        rule: The decision over the collected values.
    """

    def __init__(self, evidence: Evidence, rule: Rule):
        self.evidence = evidence
        self.rule = rule
        self.num_rows: int = 1
        self._values: dict[int, torch.Tensor] = {}
        self._decision: torch.BoolTensor | None = None
        self._snapshot: dict[int, torch.Tensor] = {}

    def reset(self, num_rows: int = 1) -> None:
        """Clear all per-generation state and size the gate to `num_rows` logical rows."""
        if num_rows < 1:
            raise ValueError(f"num_rows must be >= 1; got {num_rows}.")
        self.num_rows = int(num_rows)
        self._values.clear()
        self._decision = None
        self._snapshot = {}

    def _coerce_values(self, values: torch.Tensor | float) -> torch.Tensor:
        """Normalize `values` to a float32 `[num_rows]` CPU tensor, enforcing the row contract."""
        if isinstance(values, (int, float)):
            if self.num_rows != 1:
                raise ValueError(
                    f"Gate has {self.num_rows} rows but received a scalar value; readouts "
                    f"must return per-row values ([num_rows]) for batched generation."
                )
            return torch.tensor([float(values)], dtype=torch.float32)
        t = torch.as_tensor(values, dtype=torch.float32).reshape(-1).cpu()
        if t.numel() != self.num_rows:
            raise ValueError(
                f"Gate has {self.num_rows} rows but received {t.numel()} values."
            )
        return t

    def update(self, values: torch.Tensor | float, *, key: int) -> None:
        """Record one condition layer's per-row values; freeze the decision on completion.

        Updates after the freeze are ignored.

        Args:
            values: Per-row values of shape `[num_rows]` (a bare float is accepted only when
                `num_rows == 1`).
            key: The condition layer id the values belong to.
        """
        if self._decision is not None:
            return
        self._values[int(key)] = self._coerce_values(values)
        expected = frozenset(self.evidence.layer_ids)
        if self.rule.is_complete(frozenset(self._values), expected):
            decision = self.rule.decide(self._values, self.num_rows)
            self._decision = torch.as_tensor(decision, dtype=torch.bool).reshape(-1)
            self._snapshot = {lid: rows.clone() for lid, rows in self._values.items()}
            self._values.clear()

    def open_rows(self) -> torch.BoolTensor:
        """Per-row decision of shape `[num_rows]`.

        Returns the frozen decision once the rule has reported complete; before that, the rule
        is evaluated live over the values collected so far (all-closed before any evidence,
        since both shipped rules decide all-closed on empty values).
        """
        if self._decision is not None:
            return self._decision
        decision = self.rule.decide(dict(self._values), self.num_rows)
        return torch.as_tensor(decision, dtype=torch.bool).reshape(-1)

    def is_open(self) -> bool:
        """Scalar convenience: True if any row is open (exact for `num_rows == 1`)."""
        return bool(self.open_rows().any())

    def is_ready(self) -> bool:
        """True once the decision is frozen."""
        return self._decision is not None

    def evidence_values(self) -> dict[int, torch.Tensor]:
        """Per-layer value tensors (`[num_rows]` each) behind the frozen decision.

        Empty before the freeze; retained for diagnostics after the stored values are dropped.
        """
        return dict(self._snapshot)

    def wire_kinds(self) -> tuple[frozenset[str], frozenset[str]] | None:
        """The `({readout kind}, {rule kind})` pair, or None when either has no wire form."""
        readout_kind = type(self.evidence.readout).wire_kind
        rule_kind = type(self.rule).wire_kind
        if readout_kind is None or rule_kind is None:
            return None
        return frozenset({readout_kind}), frozenset({rule_kind})

    def export(self, register) -> dict | None:
        """The wire gate object, or None when the configuration has no wire form.

        The readout's tensors are content-addressed through `register`; rule params inline.
        The caller maps `layers` to wire indices before export, so the layer values here are
        the caller's.

        Args:
            register: Callable mapping a tensor payload to its content-addressed artifact id.

        Returns:
            The wire gate dict (`layers`, `pooling`, `readout`, `rule`), with `layers` left as
            the evidence layers for the caller to remap, or None.
        """
        readout_form = self.evidence.readout.export(self.evidence.layer_ids)
        rule_form = self.rule.export()
        if readout_form is None or rule_form is None:
            return None
        readout_wire: dict = {"kind": readout_form.kind, **readout_form.params}
        if readout_form.tensors:
            readout_wire["artifact"] = register(readout_form.tensors)
        return {
            "layers": [int(lid) for lid in self.evidence.layer_ids],
            "pooling": self.evidence.pooling,
            "readout": readout_wire,
            "rule": {"kind": rule_form.kind, **rule_form.params},
        }


@runtime_checkable
class GateSource(Protocol):
    """A recipe resolving to a `Gate` (or None for unconditional) for a model.

    Occupies an `Intervention`'s gate slot; `Intervention.bind` resolves it once. The declared
    wire kinds are class-level facts so `Intervention.wire_kinds()` can run before binding;
    None marks the resolved gating hook-only.
    """

    wire_readouts: ClassVar[frozenset[str] | None]
    wire_rules: ClassVar[frozenset[str] | None]

    def resolve_gate(self, model, tokenizer, *, layout=None, session=None) -> "Gate | None":
        """Return the resolved gate, or None for unconditional configurations."""
        ...


def gate_from_probe(probe: Probe, *, allow_model_mismatch: bool = False) -> Gate:
    """A gate reproducing the probe's decision: affine evidence summed at the calibrated bias.

    The gate reads the probe's layers at the probe's pooling, scores each layer's pooled state
    as `weights[layer] . pooled`, and opens where the sum plus `probe.bias` is at or above
    zero (ties open), matching `Probe.predict` bit-for-bit for the same hidden states (affine
    pooling commutes with scoring).

    Args:
        probe: The fitted probe whose decision admits the intervention.
        allow_model_mismatch: When True, the readout's `model_fingerprint` is set to None,
            which disarms the intervention's model-identity check.

    Returns:
        The assembled `Gate`.
    """
    readout = AffineReadout(
        dict(probe.weights),
        location=probe.location,
        model_fingerprint=None if allow_model_mismatch else probe.meta.get("model_fingerprint"),
    )
    return Gate(
        Evidence(tuple(probe.layer_ids), readout, pooling=probe.pooling),
        SumThreshold(bias=probe.bias),
    )
