"""Shared hook runtime for transform-based state controls.

`TransformHookRuntime` builds the hook closures used by residual-stream state controls and
owns one control's per-generation mutable state. It covers hidden-state extraction and
re-wrapping, KV-cache position tracking, token-scope masking, condition scoring, and gated
transform application. Four behaviors define the runtime's contract:

1. **Position tracking**: Each pass's absolute position offset is read from the `cache_position`
    kwarg when the hooked module receives it (decoder layers do, throughout the supported
    `transformers` range), so positions are exact per forwarded sequence even when a decoding
    driver issues several `generate` calls or an output control forwards the model mid-step.
    Hook points whose modules do not receive the kwarg (attention output projections, norm
    sub-modules) fall back to counting, which assumes the model processes the full prompt on
    the prefill pass and one new token per decode pass; exactly one designated pass-opener hook
    advances the shared offset by the observed sequence length once per pass, and every other
    hook in that pass reads the opener's snapshot. The fallback assumes a single `generate`
    call per generation.

2. **Row gating**: Gates hold one decision per logical row, one per prompt. HuggingFace
    `generate` may expand the hidden batch to `B_logical * num_beams` via
    `repeat_interleave`. The runtime collapses condition scores from the expanded batch to
    logical rows before calling `gate.update()`, and expands `gate.open_rows()` back to the
    hidden batch before masking hidden states. Beam siblings of one prompt share that
    prompt's decision, and each prompt in a batch is gated independently.

3. **Condition scoring**: A condition hook computes evidence values only while its gate has
    not frozen its decision. Scoring stops for the remainder of the generation once
    `gate.is_ready()` returns True.

4. **Auxiliary passes**: Forwards marked via `auxiliary_pass()` (same-model candidate scoring,
    variant-prompt branches) never feed condition scorers or gates and never advance the
    fallback counter. Trajectory-aligned auxiliary passes are transformed at their true
    positions when `cache_position` is available; detached ones are never transformed.
"""
from __future__ import annotations

import warnings
from typing import Callable, Literal

import torch

from aisteer360.algorithms.core.internals.pooling import aggregate_condition_hidden
from aisteer360.algorithms.core.utils.auxiliary_pass import current_auxiliary_pass

from .gating import Gate
from .hook_utils import extract_hidden_states, replace_hidden_states
from .token_scope import ScopeKind, align_mask_to_batch, make_token_mask
from .transforms.base import BaseTransform

HookPoint = Literal["layer_output", "layer_input"]


class TransformHookRuntime:
    """Builds hook closures and owns the per-generation position/prefill/mask state for one control.

    Args:
        hook_point: Where the control intervenes. ``"layer_output"`` builds forward hooks on the
            layer output (residual stream after the layer). ``"layer_input"`` builds forward
            pre-hooks; hidden states are extracted from the module's inputs via
            `extract_hidden_states` and re-injected via `replace_hidden_states`. Valid for any
            module receiving `hidden_states` as its first positional argument or as the
            ``hidden_states=`` kwarg, such as decoder layers, attention output projections
            (`o_proj`/`c_proj`), and per-layer norm sub-modules.
    """

    def __init__(self, *, hook_point: HookPoint = "layer_output"):
        if hook_point not in ("layer_output", "layer_input"):
            raise ValueError(f"hook_point must be 'layer_output' or 'layer_input'; got {hook_point!r}.")
        self.hook_point = hook_point

        # per-generation state (set/cleared by reset)
        self._prompt_lens: torch.LongTensor | None = None
        self._prompt_mask: torch.BoolTensor | None = None
        self._offset: int = 0
        self._pass_offset: int = 0
        self._prefill_seen: bool = False
        self._opener_built: bool = False
        self._clock_seen: bool = False
        self._warned: set[str] = set()

    def reset(
        self,
        prompt_lens: torch.LongTensor,
        prompt_mask: torch.Tensor | None = None,
    ) -> None:
        """Clear position/prefill state and store the prompt lengths/mask for this generation.

        Args:
            prompt_lens: Per-row prompt lengths of shape ``[B_logical]`` (from
                `compute_prompt_lens`). Defines the logical batch size for row gating.
            prompt_mask: Optional pad-aware prompt attention mask of shape
                ``[B_logical, T_prompt]`` (True/1 at real tokens). Forwarded to condition scorers
                on the prefill pass so condition scores align with the real (non-pad) prompt
                positions, matching the mask the selector calibrated on.
        """
        self._prompt_lens = prompt_lens
        if prompt_mask is not None:
            mask = torch.as_tensor(prompt_mask)
            if mask.ndim == 1:
                mask = mask.unsqueeze(0)
            if mask.size(0) != prompt_lens.size(0):
                raise ValueError(
                    f"prompt_mask has {mask.size(0)} rows but prompt_lens has "
                    f"{prompt_lens.size(0)}; these must describe the same logical batch."
                )
            self._prompt_mask = mask.bool()
        else:
            self._prompt_mask = None
        self._offset = 0
        self._pass_offset = 0
        self._prefill_seen = False
        self._opener_built = False
        self._clock_seen = False
        self._warned = set()

    @property
    def num_logical_rows(self) -> int:
        """Logical batch size (one row per prompt); 0 before `reset`."""
        return 0 if self._prompt_lens is None else int(self._prompt_lens.size(0))

    def _claim_opener(self, is_pass_opener: bool) -> None:
        """Enforce that at most one hook per generation is designated the pass opener.

        Two openers would advance the shared offset twice per forward pass, silently skewing
        every position-dependent token scope (e.g. `after_prompt` would steer the whole
        prompt). Controls must designate exactly one opener; when a layer hosts both a
        condition and a behavior hook, the one registered first opens the pass. The opener only
        matters for the fallback counter; when positions come from `cache_position`, every hook
        resolves its offset independently.
        """
        if not is_pass_opener:
            return
        if self._opener_built:
            raise ValueError(
                "A pass-opener hook was already built for this generation; exactly one hook "
                "may advance the position offset. When a layer hosts both a condition and a "
                "behavior hook, designate only the first-registered one as the opener."
            )
        self._opener_built = True

    @staticmethod
    def _extract_cache_position(forward_kwargs: dict | None) -> torch.Tensor | None:
        """The `cache_position` kwarg of the hooked module's call, when present and non-empty."""
        if not forward_kwargs:
            return None
        positions = forward_kwargs.get("cache_position")
        if positions is None or not torch.is_tensor(positions) or positions.numel() == 0:
            return None
        return positions

    def _warn_once(self, key: str, message: str) -> None:
        """Emit `message` as a UserWarning at most once per generation (keyed by `key`)."""
        if key in self._warned:
            return
        self._warned.add(key)
        warnings.warn(message, UserWarning)

    def _position_offset(
        self, seq_len: int, cache_position: torch.Tensor | None, is_pass_opener: bool
    ) -> int | None:
        """Resolve the absolute position offset for the current pass, or None to skip the pass.

        Auxiliary passes (marked via `auxiliary_pass()`) never advance the fallback counter. An
        aligned auxiliary pass is positioned by `cache_position` when the hooked module receives it
        and skipped otherwise; a detached auxiliary pass is always skipped. Ordinary passes take
        their offset from `cache_position` when present. The opener maintains the fallback counter
        on every ordinary pass, so hook points without the kwarg keep the one-pass-one-step
        accounting unchanged and an anomalous pass missing the kwarg degrades to counting.

        Args:
            seq_len: The sequence length seen by this hook on this call.
            cache_position: The pass's `cache_position` kwarg, when the hooked module receives it.
            is_pass_opener: Whether this hook is the designated pass opener.

        Returns:
            The absolute position offset to use when building the token mask, or None when the
            pass must be skipped.

        Warns:
            UserWarning: Once per generation for each of: an aligned auxiliary pass at a hook point
                without `cache_position` (its transform is skipped); a multi-token ordinary pass
                after prefill at such a hook point (a multi-call decode pattern that counting cannot
                place); `cache_position` disappearing after having been observed.
        """
        aux = current_auxiliary_pass()
        if aux is not None:
            if not aux.aligned:
                return None
            if cache_position is not None:
                return int(cache_position[0])
            self._warn_once(
                "aux_without_cache_position",
                "Auxiliary same-model passes cannot be position-mapped at this hook point (the "
                "hooked module does not receive `cache_position`); their transforms are skipped. "
                "Hook decoder layers for exact composition with same-model output controls.",
            )
            return None

        if is_pass_opener:
            if self._prefill_seen:
                if seq_len > 1 and cache_position is None:
                    self._warn_once(
                        "multi_call_without_cache_position",
                        "Multiple generate calls detected at a hook point that does not receive "
                        "`cache_position`; position scoping may be skewed for this generation.",
                    )
                self._pass_offset = self._offset
                self._offset += seq_len
            else:
                self._pass_offset = 0
                self._offset = seq_len
                self._prefill_seen = True

        if cache_position is not None:
            self._clock_seen = True
            return int(cache_position[0])
        if self._clock_seen:
            self._warn_once(
                "inconsistent_cache_position",
                "`cache_position` was available on earlier passes but missing on this one; "
                "falling back to pass counting for this pass.",
            )
        return self._pass_offset

    def _collapse_to_rows(self, values: torch.Tensor | float, hidden_batch: int) -> torch.Tensor | float:
        """Collapse per-hidden-row evidence values down to the logical rows the gate holds.

        Beam search expands the batch via `repeat_interleave` (`[i0, i0, i1, i1]`), so the first
        member of each group represents its logical row, and that row is taken as the group's
        value. Values already at logical size pass through; a bare float passes through for the
        gate to validate (accepted only when `num_rows == 1`).
        """
        if isinstance(values, (int, float)):
            return values
        rows = self.num_logical_rows
        t = torch.as_tensor(values).squeeze()
        if t.ndim > 1:
            raise ValueError(
                f"Gate readout returned a tensor of shape {tuple(torch.as_tensor(values).shape)}; "
                f"expected per-row values of shape [B] (extra dimensions must be size 1)."
            )
        flat = t.reshape(-1)
        if flat.numel() == rows:
            return flat
        if flat.numel() == hidden_batch and rows and hidden_batch % rows == 0:
            factor = hidden_batch // rows
            return flat[::factor]
        raise ValueError(
            f"Gate readout returned {flat.numel()} values for a hidden batch of "
            f"{hidden_batch} and {rows} logical row(s); return one value per hidden row or per "
            f"logical row."
        )

    def _row_mask_for(self, gate: Gate, hidden: torch.Tensor) -> torch.BoolTensor | None:
        """Per-row gate decision expanded to the hidden batch as a `[B_hidden, 1]` mask.

        Returns None when every row is closed (caller short-circuits). `align_mask_to_batch`
        performs the beam expansion and validates divisibility.
        """
        open_rows = gate.open_rows()
        if not bool(open_rows.any()):
            return None
        row_mask = align_mask_to_batch(open_rows.unsqueeze(1), hidden.size(0))  # [B_hidden, 1]
        return row_mask.to(hidden.device)

    def _prefill_prompt_mask(self, hidden: torch.Tensor, pass_offset: int) -> torch.Tensor | None:
        """The stored prompt mask aligned to the hidden batch, on the prefill pass only.

        The first pass may be longer than the prompt when a teacher-forced continuation is
        appended (e.g. `compute_logprobs` forwards `[prompt; ref]` in one pass). The continuation
        columns are not prompt, so the mask is extended with False there, and condition
        aggregation then covers exactly the real prompt tokens, reproducing generation-time
        scoring. A first pass shorter than the stored mask indicates misuse and raises.
        """
        if self._prompt_mask is None or pass_offset != 0:
            return None
        width = self._prompt_mask.size(1)
        seq_len = hidden.size(1)
        if width > seq_len:
            raise ValueError(
                f"Prompt mask length {width} exceeds prefill sequence length {seq_len}."
            )
        mask = self._prompt_mask
        if width < seq_len:  # teacher-forced continuation appended after the prompt
            pad = torch.zeros(mask.size(0), seq_len - width, dtype=torch.bool)
            mask = torch.cat([mask, pad], dim=1)
        mask = align_mask_to_batch(mask, hidden.size(0))
        return mask.to(hidden.device)

    def build_behavior_hook(
        self,
        *,
        layer_id: int,
        transform: BaseTransform,
        gate: Gate | None,
        token_scope: ScopeKind,
        last_k: int | None = None,
        from_position: int | None = None,
        is_pass_opener: bool = False,
        hook_point: HookPoint | None = None,
    ) -> Callable:
        """Build a hook that applies `transform` to the residual stream at `layer_id`, gated by `gate`.

        The transform fires at the intersection of the token-scope mask and the gate's per-row
        decision (expanded across beams); a fully closed gate is a no-op, and a None gate
        leaves every row open.

        Args:
            layer_id: Index of the hooked layer (used to index per-layer transform artifacts).
            transform: The transform to apply at masked positions of open rows.
            gate: Gate consulted per call, or None for unconditional application; row `r` of
                the hidden batch fires only when the gate's logical row `r // beam_factor` is
                open.
            token_scope: Which positions to steer (see `make_token_mask`).
            last_k: Required when `token_scope == "last_k"`.
            from_position: Required when `token_scope == "from_position"`.
            is_pass_opener: Whether this hook advances the shared position offset.
            hook_point: Per-hook boundary override; defaults to the runtime's constructor
                value, so one runtime can host hooks at both boundaries.

        Returns:
            A hook callable suitable for the effective hook point (a forward hook for
            ``"layer_output"``, a forward pre-hook for ``"layer_input"``).
        """
        self._claim_opener(is_pass_opener)
        if (hook_point or self.hook_point) == "layer_output":

            def _forward_hook(module, args, kwargs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                if hidden is None:
                    return output
                hidden = self._apply(hidden, layer_id, transform, gate, token_scope, last_k,
                                     from_position, is_pass_opener, forward_kwargs=kwargs)
                return (hidden,) + output[1:] if isinstance(output, tuple) else hidden

            return _forward_hook

        def _pre_hook(module, input_args, input_kwargs):
            hidden = extract_hidden_states(input_args, input_kwargs)
            if hidden is None:
                return input_args, input_kwargs
            hidden = self._apply(hidden, layer_id, transform, gate, token_scope, last_k,
                                 from_position, is_pass_opener, forward_kwargs=input_kwargs)
            return replace_hidden_states(input_args, input_kwargs, hidden)

        return _pre_hook

    def build_condition_hook(
        self,
        *,
        layer_id: int,
        gate: Gate,
        is_pass_opener: bool = False,
        hook_point: HookPoint | None = None,
    ) -> Callable:
        """Build a read-only hook that scores the residual stream at `layer_id` and updates `gate`.

        The hook never modifies hidden states. On each pass where the gate has not frozen its
        decision (`not gate.is_ready()`), it pools the hidden states per the gate's evidence
        (passing the pad-aware prompt mask on the prefill pass), computes per-row values via
        the evidence readout, collapses beam-expanded values to logical rows, and calls
        `gate.update(rows, key=layer_id)`. Once the gate is ready, scoring is skipped
        entirely, and a gate whose rule never reports complete keeps re-scoring every pass.
        The hook still participates in pass-opener bookkeeping when the lowest hooked layer is
        a condition layer. Auxiliary passes are ignored entirely: no scoring, no gate update,
        no accounting.

        Args:
            layer_id: Index of the hooked layer.
            gate: Gate whose evidence is read at this layer and fed the per-row values.
            is_pass_opener: Whether this hook advances the shared position offset.
            hook_point: Per-hook boundary override; defaults to the runtime's constructor
                value.

        Returns:
            A hook callable suitable for the effective hook point.
        """
        self._claim_opener(is_pass_opener)

        def _score(hidden: torch.Tensor, forward_kwargs: dict | None) -> None:
            if current_auxiliary_pass() is not None:
                return
            cache_position = self._extract_cache_position(forward_kwargs)
            pass_offset = self._position_offset(hidden.size(1), cache_position, is_pass_opener)
            if gate.is_ready():
                return
            prompt_mask = self._prefill_prompt_mask(hidden, pass_offset)
            pooled = aggregate_condition_hidden(
                hidden, gate.evidence.pooling, attention_mask=prompt_mask
            )
            values = gate.evidence.readout(pooled, layer_id)
            gate.update(self._collapse_to_rows(values, hidden.size(0)), key=layer_id)

        if (hook_point or self.hook_point) == "layer_output":

            def _forward_hook(module, args, kwargs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                if hidden is None:
                    return output
                _score(hidden, kwargs)
                return output

            return _forward_hook

        def _pre_hook(module, input_args, input_kwargs):
            hidden = extract_hidden_states(input_args, input_kwargs)
            if hidden is None:
                return input_args, input_kwargs
            _score(hidden, input_kwargs)
            return input_args, input_kwargs

        return _pre_hook

    def _apply(
        self,
        hidden: torch.Tensor,
        layer_id: int,
        transform: BaseTransform,
        gate: Gate | None,
        token_scope: ScopeKind,
        last_k: int | None,
        from_position: int | None,
        is_pass_opener: bool,
        forward_kwargs: dict | None = None,
    ) -> torch.Tensor:
        """Mask the current pass by token scope and per-row gate decision, then apply the transform.

        The pass's absolute position offset is forwarded to `transform.apply` as
        `position_offset`, so position-dependent transforms place their edits in absolute
        sequence coordinates. A None gate leaves every row open. Auxiliary passes without a
        resolvable position are returned unchanged.
        """
        seq_len = hidden.size(1)
        cache_position = self._extract_cache_position(forward_kwargs)
        pass_offset = self._position_offset(seq_len, cache_position, is_pass_opener)
        if pass_offset is None:
            return hidden

        row_mask = None
        if gate is not None:
            row_mask = self._row_mask_for(gate, hidden)  # [B_hidden, 1] or None (all closed)
            if row_mask is None:
                return hidden

        mask = make_token_mask(
            token_scope,
            seq_len=seq_len,
            prompt_lens=self._prompt_lens.to(hidden.device),
            last_k=last_k,
            from_position=from_position,
            position_offset=pass_offset,
        )
        mask = align_mask_to_batch(mask, hidden.size(0))  # beam search expands the batch
        if row_mask is not None:
            mask = mask & row_mask
        if not bool(mask.any()):
            return hidden
        return transform.apply(hidden, layer_id=layer_id, token_mask=mask, position_offset=pass_offset)


def build_hooks(
    interventions,
    layout,
    prompt_lens: torch.Tensor,
    prompt_mask: torch.Tensor | None = None,
    model=None,
) -> dict[str, list]:
    """Compile bound interventions to torch hooks for one logical generation.

    Creates a fresh `TransformHookRuntime` (per-generation position state is born here), resets
    every intervention's gate to the logical batch size (gate reset is idempotent, so a gate
    instance shared across interventions is reset harmlessly more than once), and emits one
    behavior hook per (intervention, layer) plus one condition hook per
    (intervention.gate.evidence, layer). An intervention whose gate is None builds no condition
    hooks and its behavior hooks apply with every row open; an intervention marked
    `gate_driven_externally` builds no condition hooks either, since another intervention's
    hooks feed its shared gate. Condition hooks precede behavior hooks so a gate update runs
    before the transform at a shared layer. Exactly one hook opens each pass: the first-firing
    hook of the lowest hooked layer across the tuple.

    Module paths derive from each intervention's resolved site: decoder layers for residual
    transforms, the attention output projection for `head_additive`, and each layer's
    normalization sub-modules for the `"norm_input"` site. The intervention's `boundary` picks
    the hook phase (`"layer_output"` builds forward hooks, `"layer_input"` forward pre-hooks);
    the `o_proj` and `"norm_input"` sites hook module inputs.

    Args:
        interventions: Bound interventions, in application order.
        layout: The module-path `ModelLayout` naming decoder layers, output projections, and
            norm sub-modules.
        prompt_lens: Per-row prompt lengths of shape `[B_logical]` (from
            `compute_prompt_lens`); defines the logical batch size for row gating.
        prompt_mask: Optional pad-aware prompt attention mask of shape `[B_logical, T_prompt]`,
            forwarded to evidence pooling on the prefill pass.
        model: Optional live model, consulted only to skip norm sub-modules a layer does not
            define at the `"norm_input"` site.

    Returns:
        Hook specifications keyed by phase (`"pre"`, `"forward"`, `"backward"`), each entry a
        mapping with `"module"` and `"hook_func"`.

    Raises:
        ValueError: If an intervention is unbound, or a layer has no module path in `layout`.
    """
    from .specs import Intervention

    runtime = TransformHookRuntime()
    runtime.reset(prompt_lens, prompt_mask)
    num_rows = int(prompt_lens.size(0))

    # hook units in module firing order: (layer, site_rank, condition_before_behavior)
    site_rank = {"pre": 0, "norm": 1, "o_proj": 2, "forward": 3}
    units: list[tuple[tuple, dict]] = []

    for intervention in interventions:
        if not isinstance(intervention, Intervention) or not isinstance(intervention.layers, tuple):
            raise ValueError("build_hooks requires bound interventions; call bind() first.")
        gate = intervention.gate
        if gate is not None and not isinstance(gate, Gate):
            raise ValueError("build_hooks requires a resolved gate; call bind() first.")
        if gate is not None:
            gate.reset(num_rows)

        site = intervention.resolved_site()
        boundary = intervention.boundary

        if gate is not None and not intervention.gate_driven_externally:
            for layer_id in gate.evidence.layer_ids:
                phase = "forward" if boundary == "layer_output" else "pre"
                units.append((
                    (layer_id, site_rank[phase if phase == "pre" else "forward"], 0),
                    {
                        "kind": "condition", "phase": phase, "layer_id": layer_id,
                        "module": layout.layer_names[layer_id],
                        "gate": gate, "hook_point": boundary,
                    },
                ))

        for layer_id in intervention.layers:
            if site == "norm_input":
                for norm_attr in layout.norm_attrs:
                    module = f"{layout.layer_names[layer_id]}.{norm_attr}"
                    if model is not None and not _submodule_exists(model, module):
                        continue
                    units.append((
                        (layer_id, site_rank["norm"], 1, module),
                        {
                            "kind": "behavior", "phase": "pre", "layer_id": layer_id,
                            "module": module, "intervention": intervention, "gate": gate,
                            "hook_point": "layer_input",
                        },
                    ))
            elif site == "o_proj":
                units.append((
                    (layer_id, site_rank["o_proj"], 1),
                    {
                        "kind": "behavior", "phase": "pre", "layer_id": layer_id,
                        "module": layout.oproj_names[layer_id], "intervention": intervention,
                        "gate": gate, "hook_point": "layer_input",
                    },
                ))
            else:
                phase = "forward" if boundary == "layer_output" else "pre"
                units.append((
                    (layer_id, site_rank[phase], 1),
                    {
                        "kind": "behavior", "phase": phase, "layer_id": layer_id,
                        "module": layout.layer_names[layer_id], "intervention": intervention,
                        "gate": gate, "hook_point": boundary,
                    },
                ))

    hooks: dict[str, list] = {"pre": [], "forward": [], "backward": []}
    if not units:
        return hooks

    # exactly one opener: the first-registered unit among those sharing the minimal firing key
    opener_index = min(range(len(units)), key=lambda index: units[index][0])

    for index, (key, unit) in enumerate(units):
        is_opener = index == opener_index
        if unit["kind"] == "condition":
            hook_func = runtime.build_condition_hook(
                layer_id=unit["layer_id"],
                gate=unit["gate"],
                is_pass_opener=is_opener,
                hook_point=unit["hook_point"],
            )
        else:
            intervention = unit["intervention"]
            hook_func = runtime.build_behavior_hook(
                layer_id=unit["layer_id"],
                transform=intervention.transform,
                gate=unit["gate"],
                token_scope=intervention.scope.kind,
                last_k=intervention.scope.last_k,
                from_position=intervention.scope.from_position,
                is_pass_opener=is_opener,
                hook_point=unit["hook_point"],
            )
        hooks[unit["phase"]].append({"module": unit["module"], "hook_func": hook_func})

    return hooks


def _submodule_exists(model, module_path: str) -> bool:
    """True when `module_path` resolves on the model's module tree."""
    try:
        model.get_submodule(module_path)
    except AttributeError:
        return False
    return True
