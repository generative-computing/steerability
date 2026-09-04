"""Payload types crossing the pipeline/backend seam.

Structural model facts, prepared prompts, steer-time artifacts, declarative constraint
sources, serialized interventions and processors, and the per-item units of session work with
the per-category control contributions they carry. A field on an item holds either an
artifact, named for what it is (`prompt`, `ref_output_ids`, `seed`), or the per-call
contributions of one control category, named `<category>_entries`. An entry is one enabled
control's contribution for this call, in controls-list order, in whichever representation the
session consumes. An item never holds a control object.
"""
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import torch

from aisteer360.algorithms.core.execution.contracts import InterventionKinds
from aisteer360.algorithms.core.output import Output
from aisteer360.utils.optional import require

CONSTRAINT_KINDS = ("json_schema", "regex", "grammar", "choice")


@dataclass(frozen=True, slots=True)
class ConstraintSource:
    """A declarative constrained-decoding specification.

    The portable form of the constraint class: one source renders per execution arm, compiled
    into a client-side automaton in process and onto the engine's native structured-output
    request parameters on vLLM backends.

    Attributes:
        kind: The constraint kind: `"json_schema"`, `"regex"`, `"grammar"` (EBNF), or
            `"choice"`.
        value: The constraint payload: a schema string or mapping for `"json_schema"`, a
            pattern string for `"regex"`, a grammar string for `"grammar"`, or a sequence of
            candidate strings for `"choice"`.
    """

    kind: Literal["json_schema", "regex", "grammar", "choice"]
    value: str | Mapping | Sequence[str]

    def __post_init__(self) -> None:
        if self.kind not in CONSTRAINT_KINDS:
            raise ValueError(
                f"Unknown constraint kind {self.kind!r}; kinds are {', '.join(CONSTRAINT_KINDS)}."
            )
        if self.kind == "json_schema":
            if not isinstance(self.value, (str, Mapping)):
                raise TypeError("A json_schema constraint takes a schema string or mapping.")
        elif self.kind in ("regex", "grammar"):
            if not isinstance(self.value, str):
                raise TypeError(f"A {self.kind} constraint takes a string.")
        else:
            if isinstance(self.value, str) or not isinstance(self.value, Sequence) or not self.value:
                raise TypeError("A choice constraint takes a non-empty sequence of strings.")
            if not all(isinstance(item, str) for item in self.value):
                raise TypeError("A choice constraint takes a non-empty sequence of strings.")
            object.__setattr__(self, "value", tuple(self.value))


def as_constraint_source(value: "ConstraintSource | Mapping[str, Any]") -> ConstraintSource:
    """Coerce a mapping with `kind` and `value` keys into a `ConstraintSource`."""
    if isinstance(value, ConstraintSource):
        return value
    if isinstance(value, Mapping):
        return ConstraintSource(kind=value["kind"], value=value["value"])
    raise TypeError(
        f"Expected a ConstraintSource or a mapping with 'kind' and 'value'; got {type(value).__name__}."
    )


@dataclass(frozen=True, slots=True)
class ModelFacts:
    """Structural facts about the pipeline model, available without a live module tree.

    Layer indices are the canonical coordinates for steer-phase layer selection; module names
    are an in-process serialization detail resolved only at hook construction time. Client-side
    tensor preparation uses the layout's dtype, and device placement is handled in process or by
    the worker rather than at steer time.


    Attributes:
        num_layers: Number of decoder layers.
        hidden_size: Residual-stream width.
        num_attention_heads: Number of attention heads, or None when the model config does not
            state one.
        head_dim: Per-head dimension (the config's value, else `hidden_size` divided by
            `num_attention_heads`), or None when neither is derivable.
        dtype: Canonical dtype string, e.g. `"bfloat16"`.
        model_fingerprint: A 16-character hex digest identifying the model weights and config.
        model_type: The config's `model_type`, or None when unknown.
        model_ref: The served model reference on engine backends, the loaded model's
            `name_or_path` in process, or None when unknown.
    """

    num_layers: int
    hidden_size: int
    num_attention_heads: int | None
    head_dim: int | None
    dtype: str
    model_fingerprint: str
    model_type: str | None = None
    model_ref: str | None = None


@dataclass(frozen=True, slots=True)
class ArtifactProvenance:
    """Identity of the side that produced an artifact.

    Attributes:
        backend_spec_hash: `BackendSpec.spec_hash` of the producing backend.
        model_fingerprint: Fingerprint of the producing model.
        tokenizer_fingerprint: Fingerprint of the producing tokenizer and chat template.
    """

    backend_spec_hash: str | None = None
    model_fingerprint: str | None = None
    tokenizer_fingerprint: str | None = None


@dataclass(frozen=True, slots=True, eq=False)
class ModelArtifact:
    """An in-memory model handed across the role boundary; only the in-process backend can
    consume it.

    Attributes:
        model: The loaded model.
        provenance: Identity of the producing side.
    """

    model: Any
    provenance: ArtifactProvenance = field(default_factory=ArtifactProvenance)


@dataclass(frozen=True, slots=True)
class CheckpointArtifact:
    """A checkpoint directory handed across the role boundary; consuming it requires
    `Capability.SERVE_CHECKPOINT`.

    Attributes:
        path: Checkpoint directory path.
        provenance: Identity of the producing side.
    """

    path: str
    provenance: ArtifactProvenance = field(default_factory=ArtifactProvenance)


@dataclass(frozen=True, slots=True)
class LoRAArtifact:
    """A LoRA adapter handed across the role boundary; consuming it requires
    `Capability.SERVE_LORA`.

    Attributes:
        path: Adapter directory path.
        base_model: Model reference the adapter applies to.
        provenance: Identity of the producing side.
    """

    path: str
    base_model: str
    provenance: ArtifactProvenance = field(default_factory=ArtifactProvenance)


Artifact = ModelArtifact | CheckpointArtifact | LoRAArtifact


@dataclass(frozen=True, slots=True, eq=False)
class PreparedPrompt:
    """A sum type over `messages | text | token_ids` for one prompt, plus metadata.

    Exactly one of `text`, `messages`, or `token_ids` is set at construction. Tokenization is
    forced only when a consumer needs token ids (`resolve_token_ids`); the in-process resolution
    reproduces the pipeline's tokenization calls, so resolved ids match the early-tokenized path.

    Attributes:
        text: A plain-text prompt, or None.
        messages: One conversation as a tuple of message mappings, or None.
        token_ids: Token ids of shape `[1, seq_len]`, or None until resolved.
        attention_mask: Attention mask matching `token_ids`, or None.
        is_single: Whether the originating call passed a single (non-batched) prompt.
        message_handled: `id()`s of input controls whose `adapt_messages` already performed the
            adaptation for this prompt.
    """

    text: str | None = None
    messages: tuple[Mapping, ...] | None = None
    token_ids: torch.Tensor | None = None
    attention_mask: torch.Tensor | None = None
    is_single: bool = True
    message_handled: frozenset[int] = frozenset()

    def __post_init__(self) -> None:
        sources = [
            name for name, value in (
                ("text", self.text), ("messages", self.messages), ("token_ids", self.token_ids),
            ) if value is not None
        ]
        if len(sources) != 1:
            raise ValueError(
                f"PreparedPrompt requires exactly one of text, messages, or token_ids; got "
                f"{', '.join(sources) or 'none'}."
            )

    @classmethod
    def from_text(cls, text: str) -> "PreparedPrompt":
        """Build a text-form prompt."""
        return cls(text=text)

    @classmethod
    def from_messages(cls, messages: list[Mapping] | tuple[Mapping, ...]) -> "PreparedPrompt":
        """Build a message-form prompt from one conversation."""
        return cls(messages=tuple(messages))

    @classmethod
    def from_token_ids(
        cls,
        token_ids: torch.Tensor | list[int],
        attention_mask: torch.Tensor | None = None,
    ) -> "PreparedPrompt":
        """Build a token-form prompt from a 1-D or `[1, seq_len]` tensor or a `list[int]`.

        Raises:
            ValueError: If `token_ids` carries more than one row; a prompt is one row.
        """
        if isinstance(token_ids, list):
            token_ids = torch.tensor(token_ids, dtype=torch.long)
        if token_ids.ndim == 1:
            token_ids = token_ids.unsqueeze(0)
        if token_ids.ndim != 2 or token_ids.size(0) != 1:
            raise ValueError(
                f"A PreparedPrompt holds one prompt row; got shape {tuple(token_ids.shape)}."
            )
        if attention_mask is not None and attention_mask.ndim == 1:
            attention_mask = attention_mask.unsqueeze(0)
        return cls(token_ids=token_ids, attention_mask=attention_mask)

    def resolve_token_ids(self, tokenizer) -> "PreparedPrompt":
        """Return a token-form copy of this prompt, tokenizing text or messages when needed.

        Text prompts tokenize via `tokenizer(...)`; message prompts via
        `tokenizer.apply_chat_template(..., add_generation_prompt=True)`. Both match the
        pipeline's own tokenization calls. A prompt already in token form is returned unchanged.

        Args:
            tokenizer: The pipeline tokenizer.

        Returns:
            A `PreparedPrompt` with `token_ids` (and, when available, `attention_mask`) set.

        Raises:
            ValueError: If tokenization is required but `tokenizer` is None.
        """
        if self.token_ids is not None:
            return self
        if tokenizer is None:
            raise ValueError("A tokenizer is required to resolve this prompt to token ids.")

        if self.text is not None:
            encoded = tokenizer([self.text], return_tensors="pt", padding=True)
            return replace(
                self,
                text=None,
                token_ids=encoded["input_ids"],
                attention_mask=encoded.get("attention_mask"),
            )

        encoded = tokenizer.apply_chat_template(
            [list(self.messages)],
            return_tensors="pt",
            padding=True,
            add_generation_prompt=True,
            return_dict=True,
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask")
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
            if attention_mask is not None:
                attention_mask = attention_mask.unsqueeze(0)
        return replace(self, messages=None, token_ids=input_ids, attention_mask=attention_mask)


def _plain(value: Any) -> Any:
    """Recursively convert mappings and sequences to plain dicts and lists."""
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _collect_artifact_ids(value: Any, found: set[str]) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key == "artifact" and isinstance(item, str):
                found.add(item)
            else:
                _collect_artifact_ids(item, found)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _collect_artifact_ids(item, found)


@dataclass(frozen=True, slots=True)
class InterventionSpec:
    """A serialized activation intervention for intervention-capable backends.

    Each op names its target layers, a transform (kind, scalar parameters, tensor payloads by
    artifact reference, and an ordered modifier list), a token scope, and an optional gate. Kind
    names are the advertised wire names; a worker rejects a spec containing a kind or field it
    does not list.

    Attributes:
        ops: The intervention ops, each a mapping with keys `"layers"`, `"transform"`,
            `"scope"`, and `"gate"`.
        artifacts: Tensor payloads keyed by the content-addressed artifact ids the ops
            reference, each a mapping from tensor name to a float32 contiguous CPU tensor.
            Sessions materialize these into the registry the serving engine reads before
            submission. Excluded from equality, the wire form, and the canonical form.
    """

    ops: tuple[Mapping[str, Any], ...] = ()
    artifacts: Mapping[str, Mapping[str, Any]] = field(default_factory=dict, compare=False)

    def to_wire(self) -> dict[str, Any]:
        """The plain-data wire form, `{"ops": [...]}`, with nested mappings and sequences
        converted to dicts and lists."""
        return _plain({"ops": list(self.ops)})

    def artifact_ids(self) -> tuple[str, ...]:
        """Sorted unique artifact ids referenced anywhere in the ops (transform payloads,
        modifiers, and gate readouts)."""
        found: set[str] = set()
        _collect_artifact_ids(self.to_wire(), found)
        return tuple(sorted(found))

    def required_kinds(self) -> InterventionKinds:
        """The kind names this spec requires a backend to serve, as an `InterventionKinds`.

        Collects transform, modifier, scope, and gate readout/rule kind names from the ops; a
        backend whose negotiated kinds contain them can execute the spec.
        """
        transforms: set[str] = set()
        modifiers: set[str] = set()
        scopes: set[str] = set()
        readouts: set[str] = set()
        rules: set[str] = set()
        for op in self.to_wire()["ops"]:
            transform = op.get("transform", {})
            if "kind" in transform:
                transforms.add(transform["kind"])
            for modifier in transform.get("modifiers", []):
                if "kind" in modifier:
                    modifiers.add(modifier["kind"])
            scope = op.get("scope", {})
            if "kind" in scope:
                scopes.add(scope["kind"])
            gate = op.get("gate")
            if gate is not None:
                readout = gate.get("readout", {})
                if "kind" in readout:
                    readouts.add(readout["kind"])
                rule = gate.get("rule", {})
                if "kind" in rule:
                    rules.add(rule["kind"])
        return InterventionKinds(
            transforms=frozenset(transforms),
            modifiers=frozenset(modifiers),
            scopes=frozenset(scopes),
            readouts=frozenset(readouts),
            rules=frozenset(rules),
        )

    def canonical(self) -> str:
        """The canonical serialization, the form hashed for cache salting and provenance.

        Delegates to `vllm_hook_plugins.core.canonical.canonical_bytes` (sorted keys, compact
        separators, UTF-8), so the toolkit and the plugin agree byte-for-byte on the canonical
        form of a spec.

        Raises:
            ModuleNotFoundError: If `vllm_hook_plugins` is not installed. The message names
                the `aisteer360[vllm]` extra.
            TypeError: If an op contains a value with no JSON form. Tensors belong in
                artifacts, never inline.
        """
        canonical = require("vllm_hook_plugins.core.canonical")
        return canonical.canonical_bytes(self.to_wire()).decode("utf-8")

    def salt(self) -> str:
        """The reference cache salt for requests carrying this spec.

        Delegates to `vllm_hook_plugins.core.canonical.request_salt` over the wire form and
        the referenced artifact ids. Returns the 64-char lowercase-hex digest.

        Raises:
            ModuleNotFoundError: If `vllm_hook_plugins` is not installed. The message names
                the `aisteer360[vllm]` extra.
        """
        canonical = require("vllm_hook_plugins.core.canonical")
        return canonical.request_salt(self.to_wire(), list(self.artifact_ids()))


@dataclass(frozen=True, slots=True)
class ProcessorSpec:
    """A serialized per-step logit processor for backends advertising engine-hosted logit math.

    Attributes:
        kind: The advertised processor kind name, e.g. `"constraint"`.
        params: Processor parameters.
    """

    kind: str
    params: Mapping[str, Any] = field(default_factory=dict)


def remap_prompt_relative_scopes(spec: InterventionSpec, anchor: int) -> InterventionSpec:
    """A rollout copy of `spec` with prompt-relative scopes rewritten to absolute positions.

    Prompt-relative scope kinds are client-side sugar: the worker anchors them at the
    request's own prompt length, and a driver rollout's request prompt is the accumulated
    prefix rather than the user prompt. The wire form of a scope inside a driver generation is
    therefore absolute: `after_prompt` becomes `from_position` at `anchor` (the generation's
    original prompt boundary). The rewrite changes one scalar per op, so artifact ids are
    untouched; the cache salt varies with the anchor, which is correct, since differently
    anchored requests compute different hidden states. The `last_k` kind is refused: its
    in-process semantics are relative to each forwarded pass, which no absolute position can
    reproduce across rollouts.

    Args:
        spec: The lowered spec.
        anchor: The generation's original prompt length, in absolute positions.

    Returns:
        The rewritten spec (the same object when nothing changed), sharing `artifacts`.

    Raises:
        ValueError: If an op carries a `last_k` scope.
    """
    ops = []
    changed = False
    for op in spec.to_wire()["ops"]:
        scope = op.get("scope", {})
        kind = scope.get("kind")
        if kind == "after_prompt":
            op = {**op, "scope": {"kind": "from_position", "position": int(anchor)}}
            changed = True
        elif kind == "last_k":
            raise ValueError(
                "last_k has no absolute rollout form: it is relative to each forwarded pass "
                "in process, which no fixed position reproduces across rollouts; use "
                "from_position, or run this driver on the huggingface backend."
            )
        ops.append(op)
    if not changed:
        return spec
    return InterventionSpec(ops=tuple(ops), artifacts=spec.artifacts)


@dataclass(frozen=True, slots=True, eq=False)
class HookEntry:
    """One state control's torch-hook contribution, consumed by in-process sessions.

    Attributes:
        hooks: Hook specifications keyed by phase (`"pre"`, `"forward"`, `"backward"`), as
            returned by `StateControl.get_hooks`.
    """

    hooks: Mapping[str, list]


@dataclass(frozen=True, slots=True)
class InterventionEntry:
    """One state control's intervention-spec contribution, consumed by intervention-capable
    backends.

    Attributes:
        spec: The serialized intervention.
    """

    spec: InterventionSpec


StateControlEntry = HookEntry | InterventionEntry


@dataclass(frozen=True, slots=True, eq=False)
class StackEntry:
    """One output control's live processor and criteria contribution, consumed by in-process
    sessions.

    Attributes:
        logits_processors: HF `LogitsProcessor`-style objects, in contribution order.
        stopping_criteria: HF `StoppingCriteria`-style objects, in contribution order.
    """

    logits_processors: tuple = ()
    stopping_criteria: tuple = ()


@dataclass(frozen=True, slots=True)
class ConstraintEntry:
    """An output control's contribution as a declarative constrained-decoding source.

    Consumed by backends advertising `Capability.GUIDED_DECODING`, rendered onto the engine's
    native structured-output request parameters in place of the control's live processor.

    Attributes:
        source: The declarative constraint.
    """

    source: ConstraintSource


@dataclass(frozen=True, slots=True)
class ProcessorSpecEntry:
    """One output control's engine-hosted processor contribution.

    Attributes:
        spec: The serialized processor.
    """

    spec: ProcessorSpec


OutputControlEntry = StackEntry | ProcessorSpecEntry | ConstraintEntry


@dataclass(frozen=True, slots=True, eq=False)
class GenerationItem:
    """One prompt's unit of generation work.

    Input controls have no entry because their contribution is already folded into `prompt`;
    structural controls have none because they contribute at steer time through artifacts.

    Attributes:
        prompt: The prepared prompt.
        state_entries: Enabled state controls' contributions, in controls-list order.
        output_entries: Enabled output controls' contributions, in controls-list order.
        seed: Per-item sampling seed, or None for unseeded operation.
    """

    prompt: PreparedPrompt
    state_entries: tuple[StateControlEntry, ...] = ()
    output_entries: tuple[OutputControlEntry, ...] = ()
    seed: int | None = None


@dataclass(frozen=True, slots=True, eq=False)
class ScoringItem:
    """One prompt's unit of scoring work (teacher-forced reference tokens).

    Only controls participating in scoring contribute entries, and stopping criteria are never
    applied (there is no loop to stop).

    Attributes:
        prompt: The prepared prompt.
        ref_output_ids: Reference tokens to score, shape `[ref_len]` or `[1, ref_len]`.
        state_entries: Enabled state controls' contributions, in controls-list order.
        output_entries: Scoring-participant output controls' contributions, in controls-list
            order.
    """

    prompt: PreparedPrompt
    ref_output_ids: torch.Tensor
    state_entries: tuple[StateControlEntry, ...] = ()
    output_entries: tuple[OutputControlEntry, ...] = ()


@dataclass(frozen=True, slots=True, eq=False)
class ItemResult:
    """The result of one generation item.

    Attributes:
        index: Position of the item in the submitted sequence.
        output: The generation record. For `n > 1` the record's batch dimension holds the
            candidates in request order and `finish_reason` reflects the first candidate.
    """

    index: int
    output: Output


@dataclass(frozen=True, slots=True, eq=False)
class CaptureResult:
    """Hidden states captured by `SteeringSession.capture`.

    Attributes:
        hidden: Tensors keyed by 0-based layer id. Shape `[N, T, H]` in `"all_tokens"` mode and
            `[N, H]` in `"last_token"` mode, on CPU, in the model's native dtype.
        attention_mask: Mask of shape `[N, T]` matching the captured prompts, on CPU.
        mode: The capture mode the tensors were produced under.
        location: The capture location (`"layer_output"` or `"layer_input"`).
    """

    hidden: Mapping[int, torch.Tensor]
    attention_mask: torch.Tensor
    mode: str
    location: str
