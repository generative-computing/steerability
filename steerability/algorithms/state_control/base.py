"""State control base classes.

This module provides the abstract base classes for methods that steer through hooks into the
model's forward pass (modifying intermediate representations during inference); state controls
do not change model weights.

Three classes are provided:

- `StateControl`: Abstract root the pipeline type-checks.
- `InterventionControl`: A state control that is a tuple of declarative interventions.
- `HookControl`: A state control that writes its own torch hooks.

State controls implement steering through runtime intervention in the model's forward pass, modifying internal states
(activations, attention patterns) to produce generations following y ~ p_θᵃ(x), where "p_θᵃ" is the model with state
controls.

Examples of state controls:

- Activation steering (e.g., adding direction vectors)
- Attention head manipulation and pruning
- Layer-wise activation editing
- Dynamic routing between components
- Representation engineering techniques

Hooks travel only as `HookEntry` contributions on session items; the session that executes
forwards owns registration. Controls never register hooks and hold no model reference.

See Also:

- `steerability.algorithms.state_control`: Implementations of state control methods
- `steerability.core.steering_pipeline`: Integration with steering pipeline
"""
import copy
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Callable

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from steerability.algorithms.core.base_args import BaseArgs
from steerability.algorithms.core.base_control import BaseControl, NotFreezableError
from steerability.algorithms.core.execution.access import ModelAccess
from steerability.algorithms.core.execution.contracts import Requirements

if TYPE_CHECKING:
    from steerability.algorithms.core.execution.backend import SteeringSession
    from steerability.algorithms.core.execution.payloads import InterventionSpec


def _fit_ingredients(source) -> Any:
    """The encodable fit-identity form of one fit source.

    Sources exposing `fit_ingredients()` return that; dataclass sources pass through
    unchanged.
    """
    ingredients = getattr(source, "fit_ingredients", None)
    if callable(ingredients):
        return ingredients()
    return source


def _core_artifact_view(core):
    """A `SteeringVector` view of a bound core transform's artifact, or None.

    Transforms storing a per-layer directions mapping are viewed as a `SteeringVector` with
    `model_type="unknown"`; transforms storing a `SteeringVector` return it directly.
    """
    from steerability.algorithms.state_control.common.steering_vector import SteeringVector

    vector = getattr(core, "steering_vector", None)
    if vector is not None:
        return vector
    directions = getattr(core, "directions", None)
    if directions is not None:
        return SteeringVector(
            model_type="unknown", directions=dict(directions), meta=dict(core.artifact_meta or {}),
        )
    return None


PreHook = Callable[[nn.Module, tuple], tuple | torch.Tensor]
ForwardHook = Callable[[nn.Module, tuple, torch.Tensor], torch.Tensor]
BackwardHook = Callable[[nn.Module, tuple, tuple], tuple]
HookSpec = dict[str, str | PreHook | ForwardHook | BackwardHook]


class StateControl(BaseControl):
    """Abstract root for state control steering methods; the class the pipeline type-checks.

    Concrete state controls are either `InterventionControl` (a declarative intervention
    tuple, compiled to hooks by `build_hooks` and to `InterventionSpec` payloads by
    `lower_interventions`) or `HookControl` (raw torch hooks). Hooks are per-generation
    products the pipeline collects as `HookEntry` contributions; controls never register
    hooks and hold no model reference.

    Methods:
        get_hooks(input_ids, runtime_kwargs, **kwargs) -> dict: Create hook specs (required)
        steer(model, tokenizer, **kwargs) -> None: One-time preparation (optional)
    """

    Args: type[BaseArgs] | None = None
    RUNTIME_KWARGS_SCHEMA: list[dict] = []

    enabled: bool = True
    supports_batching: bool = False

    @abstractmethod
    def get_hooks(
        self,
        input_ids: torch.Tensor,
        runtime_kwargs: dict | None,
        **kwargs,
    ) -> dict[str, list[HookSpec]]:
        """Create hook specifications for the current generation.

        The pipeline forwards `attention_mask` (the prompt attention mask matching `input_ids`,
        or None) through `**kwargs`; controls that score prompt tokens may consume it to align
        with the real (non-pad) positions instead of re-deriving a mask by token identity.

        Args:
            input_ids: Prompt token ids of shape [batch, seq_len].
            runtime_kwargs: Per-call parameters for the control.
        """
        pass

    def steer(self,
              model: PreTrainedModel,
              tokenizer: PreTrainedTokenizerBase = None,
              session=None,
              **kwargs) -> None:
        """Optional steering/preparation.

        `session` is a `SteeringSession` on the steering backend, provided by the pipeline.
        """
        pass

    def export_intervention_spec(self, runtime_kwargs: dict | None = None) -> "InterventionSpec | None":
        """The control's `InterventionSpec` for intervention-capable backends, or None.

        The spec is the second serialization of the tuple the control's hooks close over,
        emitted from the same transform, gate, and scope objects. Must be called after
        `steer()`. Returns None when the configuration has no wire form (the configuration is
        then hook-only) or when the control does not implement spec export at all.

        Args:
            runtime_kwargs: Per-call parameters, mirroring `get_hooks`; per-item values
                (strengths, positions) serialize into the returned spec.

        Returns:
            The validated `InterventionSpec` with tensor payloads attached, or None.
        """
        return None

def _is_concrete_gate(gate) -> bool:
    """True when `gate` is a resolved gate rather than a gate source."""
    from steerability.algorithms.state_control.common.gating import Gate

    return isinstance(gate, Gate)


class HookControl(StateControl):
    """A state control that writes its own torch hooks.

    Keeps the abstract `get_hooks(input_ids, runtime_kwargs, **kwargs)` and must fully
    re-derive its per-generation state inside `get_hooks` on every call. Controls whose
    behavior is a tuple of residual-stream interventions subclass `InterventionControl`
    instead; this class is for methods hooking other mechanisms (e.g. attention weights).

    Keeps the conservative in-process generate requirement unless the subclass overrides
    `requirements()`.
    """


class InterventionControl(StateControl):
    """A state control that is a tuple of interventions.

    Subclasses declare an unbound intervention template, usually in `_configure()`; the base
    `steer()` binds it. There is no per-generation protocol on the control: hook construction,
    gate sizing, and position state are owned by `build_hooks`, and lowering to
    `InterventionSpec` is owned by `lower_interventions`.

    Class attributes:
        hook_only_hint: Fix text used in unsupported-generate verdicts when the template has
            no wire form.

    Attributes:
        interventions: The bound interventions, populated by `steer()`.
    """

    supports_batching = True
    hook_only_hint: str | None = None

    tokenizer = None
    interventions: tuple = ()
    _template: tuple = ()

    def steer(
        self,
        model: PreTrainedModel | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        session: "SteeringSession | None" = None,
        **kwargs,
    ) -> PreTrainedModel | None:
        """Bind the intervention template against the model (or the session's layout).

        Structural facts come from the steering session's layout when a session is given, so a
        fully concrete template (precomputed vectors, manual thresholds) binds with
        `model=None`. Templates carrying sources (fits, searches) resolve them here.

        Args:
            model: The base language model, or None for concrete templates bound against a
                session layout.
            tokenizer: Tokenizer used when fitting sources.
            session: `SteeringSession` on the steering backend, provided by the pipeline.

        Returns:
            The input model, unchanged.
        """
        from steerability.algorithms.core.internals.model_layout import resolve_model_layout
        from steerability.algorithms.state_control.common.layout_facts import resolve_layout

        layout = resolve_layout(model, session)
        self._num_layers = layout.num_layers
        self._module_layout = resolve_model_layout(model) if model is not None else None
        if tokenizer is not None:
            self.tokenizer = tokenizer
        self.interventions = tuple(
            intervention.bind(model, tokenizer, layout=layout, session=session)
            for intervention in self._template
        )
        return model

    @property
    def _transform(self):
        """The first intervention's transform (None before `steer()`).

        Assignment replaces the transform on the first intervention, so a wrapped or
        instrumented transform takes effect in subsequently built hooks.
        """
        return self.interventions[0].transform if self.interventions else None

    @_transform.setter
    def _transform(self, value) -> None:
        import dataclasses

        if not self.interventions:
            raise AttributeError("No bound interventions; call steer() first.")
        first, *rest = self.interventions
        self.interventions = (dataclasses.replace(first, transform=value), *rest)

    @property
    def _gate(self):
        """The first intervention's gate (None before `steer()` or when unconditional)."""
        return self.interventions[0].gate if self.interventions else None

    def _resolve_module_layout(self, model=None):
        """The module-path layout, resolved from the module tree on first use."""
        layout = getattr(self, "_module_layout", None)
        if layout is None:
            from steerability.algorithms.core.internals.model_layout import resolve_model_layout

            if model is None:
                raise RuntimeError(
                    f"{type(self).__name__} was steered without a live model, so hook module "
                    "names are unresolved; provide the model (the pipeline does) or steer with "
                    "a model."
                )
            layout = resolve_model_layout(model)
            self._module_layout = layout
        return layout

    def get_hooks(
        self,
        input_ids: torch.Tensor | dict,
        runtime_kwargs: dict | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, list[HookSpec]]:
        """Compile the bound interventions to hooks for the current generation.

        Delegates to `build_hooks`: a fresh hook runtime is created, gates reset to the
        logical batch, and one behavior hook is emitted per (intervention, layer).

        Generation-time context arrives through `**kwargs`; `model` is consulted to resolve
        hook module names when steering ran without a live model.

        Args:
            input_ids: Prompt token ids of shape `[B, T]` or `[T]`.
            runtime_kwargs: Unused.
            attention_mask: The prompt attention mask matching `input_ids`, forwarded to
                gate evidence pooling on the prefill pass. When None and the tokenizer defines a
                pad token, a mask is inferred from leading and trailing pad runs.

        Returns:
            Hook specifications with `"pre"`, `"forward"`, `"backward"` keys.
        """
        from steerability.algorithms.state_control.common.runtime import build_hooks
        from steerability.algorithms.state_control.common.token_scope import compute_prompt_lens
        from steerability.utils.tokenization import infer_attention_mask_from_ids

        ids = input_ids if isinstance(input_ids, torch.Tensor) else input_ids["input_ids"]
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)

        layout = self._resolve_module_layout(kwargs.get("model"))
        pad_token_id = getattr(self.tokenizer, "pad_token_id", None) if self.tokenizer is not None else None
        prompt_lens = compute_prompt_lens(ids, pad_token_id)

        if attention_mask is not None:
            mask = attention_mask if isinstance(attention_mask, torch.Tensor) else torch.as_tensor(attention_mask)
            prompt_mask = (mask.unsqueeze(0) if mask.ndim == 1 else mask).to(torch.bool)
        elif pad_token_id is not None:
            prompt_mask = infer_attention_mask_from_ids(ids, pad_token_id).to(torch.bool)
        else:
            prompt_mask = None

        return build_hooks(self.interventions, layout, prompt_lens, prompt_mask, model=kwargs.get("model"))

    def export_intervention_spec(self, runtime_kwargs: dict | None = None):
        """The control's `InterventionSpec`, lowered from the bound interventions, or None.

        Must be called after `steer()`. Returns None when the configuration has no wire form.
        """
        from steerability.algorithms.state_control.common.lowering import lower_interventions

        if not self.interventions or getattr(self, "_num_layers", None) is None:
            return None
        kinds = self.wire_kinds()
        if kinds is None:
            return None
        return lower_interventions(self.interventions, num_layers=self._num_layers)

    def wire_kinds(self):
        """The combined wire kinds of the bound interventions (or the template before
        `steer()`), or None when any intervention is hook-only."""
        from steerability.algorithms.state_control.common.specs import combine_kinds

        source = self.interventions or self._template
        return combine_kinds(intervention.wire_kinds() for intervention in source)

    def _unbound_sources(self):
        """Yield each unbound template element's source (or undeclared factory slot).

        Yields the transform sources of unbound transform elements, factory transform slots
        themselves (which declare their own `access` or default to the live model), and
        unresolved gate sources, in template order.
        """
        from steerability.algorithms.state_control.common.transforms.base import BaseTransform, unwrap_modifiers

        for intervention in self._template:
            transform = intervention.transform
            if isinstance(transform, BaseTransform):
                core, wrappers = unwrap_modifiers(transform)
                for element in (core, *wrappers):
                    if not element.is_bound and element.source is not None:
                        yield element.source
            else:
                yield transform
            gate = intervention.gate
            if gate is not None and not _is_concrete_gate(gate):
                yield gate

    def steer_access(self) -> ModelAccess:
        """The model access the template's steer step requires, folded over its sources.

        A fully bound template requires `ModelAccess.FACTS`, since pure layer selectors
        resolve from structural facts available on any session. Otherwise the strongest
        declared source access wins; a source or factory slot without an `access` declaration
        builds against the live model, so it counts as `ModelAccess.MODULE`.
        """
        access = ModelAccess.FACTS
        for source in self._unbound_sources():
            access = max(access, getattr(source, "access", ModelAccess.MODULE))
        return access

    def _fit_sources(self):
        """Yield the template's fit sources: unbound sources carrying an `artifact_class`
        whose resolution does model-side work (`access` above `ModelAccess.FACTS`)."""
        for source in self._unbound_sources():
            if getattr(source, "artifact_class", None) is None:
                continue
            if getattr(source, "access", ModelAccess.MODULE) == ModelAccess.FACTS:
                continue
            yield source

    def steer_fits(self) -> tuple[tuple[str, str], ...]:
        """The template's fit artifacts, i.e. every unbound source carrying an
        `artifact_class`, as `(artifact, artifact_class)` pairs in template order."""
        return tuple(
            (type(source).__name__, source.artifact_class) for source in self._fit_sources()
        )

    def fit_identity(self) -> Any | None:
        """The template's fit sources in template order, each in its encodable form, or None
        when the template declares no fits."""
        sources = list(self._fit_sources())
        if not sources:
            return None
        return tuple(_fit_ingredients(source) for source in sources)

    def export_state(self) -> dict[str, Any]:
        """Bound steering artifacts, keyed by intervention position.

        For intervention `i`, the core transform's artifact exports as
        `"intervention_{i}/transform"` (a `SteeringVector` view), each wrapper's own artifact
        as `"intervention_{i}/modifier_{j}"`, and the resolved gate as
        `"intervention_{i}/gate"` when the template's gate slot held a source. Must be called
        after `steer()`.

        Returns:
            Mapping from logical name to artifact value.
        """
        from steerability.algorithms.state_control.common.gating import Gate
        from steerability.algorithms.state_control.common.transforms.base import BaseTransform, unwrap_modifiers

        state: dict[str, Any] = {}
        for i, intervention in enumerate(self.interventions):
            if isinstance(intervention.transform, BaseTransform):
                core, wrappers = unwrap_modifiers(intervention.transform)
                view = _core_artifact_view(core)
                if view is not None:
                    state[f"intervention_{i}/transform"] = view
                for j, wrapper in enumerate(wrappers):
                    own = getattr(wrapper, "steering_vector", None)
                    if own is not None:
                        state[f"intervention_{i}/modifier_{j}"] = own
            template_gate = self._template[i].gate if i < len(self._template) else None
            if isinstance(intervention.gate, Gate) and template_gate is not None \
                    and not _is_concrete_gate(template_gate):
                state[f"intervention_{i}/gate"] = intervention.gate
        return state

    def frozen_form(self, state: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
        """One `activation_adapter` entry per bound intervention, in intervention order.

        Each entry's args mirror the intervention: the bound transform, the resolved behavior
        layers, the hook point, the resolved gate, and the token scope. Must be called after
        `steer()`.

        Returns:
            List of `("state_control/activation_adapter", kwargs)` pairs.

        Raises:
            NotFreezableError: If the control is unsteered, an intervention hooks the
                `norm_input` site (no `activation_adapter` form), or an intervention follows
                an externally driven shared gate (an in-memory relationship that does not
                serialize).
        """
        if not self.interventions:
            raise NotFreezableError(
                f"{type(self).__name__} has no bound interventions; call steer() before freezing."
            )
        entries: list[tuple[str, dict[str, Any]]] = []
        for intervention in self.interventions:
            if intervention.resolved_site() == "norm_input":
                raise NotFreezableError(
                    f"{type(self).__name__} hooks the norm_input site, which has no "
                    "activation_adapter form; configure the layer_output intervention point "
                    "to freeze this control."
                )
            if intervention.gate_driven_externally:
                raise NotFreezableError(
                    f"{type(self).__name__} follows an externally driven shared gate, an "
                    "in-memory relationship that does not serialize; freeze the driving "
                    "control's pipeline without the follower, or gate this control directly."
                )
            entries.append(("state_control/activation_adapter", {
                "transform": intervention.transform,
                "layer_ids": [int(lid) for lid in intervention.layers],
                "hook_point": intervention.boundary,
                "gate": intervention.gate,
                "token_scope": intervention.scope.kind,
                "last_k": intervention.scope.last_k,
                "from_position": intervention.scope.from_position,
            }))
        return entries

    def requirements(self) -> Requirements:
        """Backend requirements derived from the declared interventions, per phase.

        Generate offers the intervention-spec alternative whenever every component of every
        intervention has a wire form; hook-only configurations require the in-process backend.
        Score is in-process: remote prompt-logprob scoring anchors token scopes at the
        request's prompt end (the end of the prompt-plus-reference concatenation), which would
        silently unanchor prompt-relative interventions.
        """
        from steerability.algorithms.core.execution.contracts import Capability, Requirements, any_of, needs

        kinds = self.wire_kinds()
        in_process = needs(Capability.IN_PROCESS_TORCH)
        score = needs(
            Capability.IN_PROCESS_TORCH,
            hint=(
                "remote prompt-logprob scoring anchors token scopes at the request's prompt "
                "end, so scoped interventions would not cover the reference; score on the "
                "huggingface backend"
            ),
        )
        if kinds is None:
            return Requirements(
                generate=needs(Capability.IN_PROCESS_TORCH, hint=self.hook_only_hint),
                score=score,
            )
        return Requirements(
            generate=any_of(
                in_process,
                needs(
                    Capability.INTERVENTION_SPECS,
                    kinds=kinds,
                    hint="serve this intervention through the vLLM-Hook plugin",
                ),
            ),
            score=score,
        )

    def clone_for_call(self, seed: int | None = None):
        """A per-call clone whose interventions carry independent gate state.

        Gates are deep-copied with one shared memo across the control's interventions, so a
        gate instance shared by several interventions stays shared inside the clone while
        being isolated from the original and from sibling clones. Transforms, scorers, and
        steer-time artifacts stay shared.
        """
        import dataclasses

        clone = super().clone_for_call(seed)
        if self.interventions:
            memo: dict = {}
            clone.interventions = tuple(
                dataclasses.replace(intervention, gate=copy.deepcopy(intervention.gate, memo))
                for intervention in self.interventions
            )
        return clone
