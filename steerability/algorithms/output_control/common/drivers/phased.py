"""Phase-plan types (`Fixed`, `Generated`) and `PhasedDriver`, which executes a plan by splicing
token streams.

A phase plan is a list of `Fixed` (append text without generating) and `Generated` (generate until a
boundary) phases. Every `Generated` phase delegates to `model.generate` with the received stacks
(the driver contract satisfied per phase). An `extract_after` output rule reproduces
ThinkingIntervention's tail extraction.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from transformers import PreTrainedModel, StoppingCriteriaList

from steerability.algorithms.core.execution.contracts import Requirements
from steerability.algorithms.output_control.base import DecodingDriver, resolve_generate_callable, stack_generate_kwargs
from steerability.algorithms.output_control.common.criteria import BudgetTokens, StopOnSubstring, StopOnTokens


@dataclass(frozen=True)
class Fixed:
    """Splice this text's tokens into the stream without generating.

    Attributes:
        text: A literal string, or a callable `(prompt_text, params) -> str`.
        replace: When True, this phase's tokens replace the current stream rather than being
            appended. Used to model a prompt-rewriting transform (ThinkingIntervention) whose text
            already contains the prompt.
        add_special_tokens: Whether to add special tokens when tokenizing (True for a replacing
            phase that becomes the new prompt; False for an appended snippet).
    """

    text: str | Callable[[str, dict], str]
    replace: bool = False
    add_special_tokens: bool = False


@dataclass(frozen=True)
class Generated:
    """Generate until a boundary: the `until` substring, any token in `until_token_ids`, or the
    token `budget`, whichever occurs first.

    All three compose with the pipeline's criteria for the phase. `until_token_ids` is the
    backend-portable form of a delimiter that tokenizes to a special token, which a stop string
    cannot express (`skip_special_tokens=True` strips it before a vLLM stop-string match).

    Attributes:
        until: Substring that ends the phase (via `StopOnSubstring`). None disables.
        until_token_ids: Token ids any one of which ends the phase once it is the last generated
            token (via `StopOnTokens`). On the session path the ids extend `stop_token_ids` the
            way `until` extends `stop_strings`. Empty disables.
        budget: Max new tokens for the phase (via `BudgetTokens`). None disables.
    """

    until: str | None = None
    until_token_ids: tuple[int, ...] = ()
    budget: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "until_token_ids", tuple(int(i) for i in self.until_token_ids))


class PhasedDriver(DecodingDriver):
    """Execute a per-example phase plan by splicing token streams.

    Subclasses / callers provide `plan(prompt_text, params) -> list[Fixed | Generated]`. Every
    `Generated` phase delegates to `model.generate` with the received stacks. An `extract_after`
    output rule reproduces ThinkingIntervention's tail extraction: decode, rsplit on the marker, keep
    the original prompt's token prefix + the re-tokenized remainder.

    Plans are per example; batched inputs are handled by looping over rows.
    """

    RUNTIME_KWARGS_SCHEMA = [
        {
            "name": "params",
            "type": "dict",
            "scope": "call",
            "help": (
                "Phase-plan parameters for this call: a mapping whose scalar values apply to every prompt row and "
                "whose list values carry one entry per row, of batch length."
            ),
        },
    ]

    def __init__(self, extract_after: str | None = None):
        self.extract_after = extract_after
        self.tokenizer = None  # injected by the pipeline

    def max_rollouts_per_query(self) -> int | None:
        """None: the phase plan is per example, so the base class declares no static bound.
        Subclasses with a fixed plan (e.g. `PhasedDecoding`, `BudgetForcing`) override it."""
        return None

    def requirements(self) -> Requirements:
        """Phase splicing is client-side and generated phases run through the session, so no
        phase requires anything beyond the session contract."""
        return Requirements()

    def plan(self, prompt_text: str, params: dict) -> list:
        """Return the phase plan for one example. Subclasses override."""
        raise NotImplementedError

    def _params_per_example(self, runtime_kwargs: dict, batch_size: int) -> list[dict]:
        """Slice a dict-of-lists `params` per example (ThinkingIntervention's convention)."""
        params_agg = runtime_kwargs.get("params", None)
        if params_agg is None:
            return [{} for _ in range(batch_size)]
        if isinstance(params_agg, dict) and any(isinstance(v, (list, tuple)) for v in params_agg.values()):
            out = []
            for i in range(batch_size):
                p_i = {}
                for k, v in params_agg.items():
                    if isinstance(v, (list, tuple)):
                        if len(v) != batch_size:
                            raise ValueError(
                                f"params['{k}'] has length {len(v)}, but batch size is {batch_size}."
                            )
                        p_i[k] = v[i]
                    else:
                        p_i[k] = v
                out.append(p_i)
            return out
        return [params_agg] * batch_size

    def decode(self, input_ids, attention_mask, model: PreTrainedModel | None, logits_processors,
               stopping_criteria, runtime_kwargs, session=None, **gen_kwargs) -> torch.Tensor:
        if self.tokenizer is None:
            raise RuntimeError("PhasedDriver requires a tokenizer; steer() must run first.")

        runtime_kwargs = runtime_kwargs or {}
        via_session = session is not None
        base_generate = resolve_generate_callable(model, runtime_kwargs, session=session)

        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        batch_size = input_ids.size(0)
        params_per_example = self._params_per_example(runtime_kwargs, batch_size)
        original_prompts = self.tokenizer.decode(input_ids, skip_special_tokens=True)
        original_lengths = [row.size(0) for row in input_ids]

        final_sequences: list[torch.Tensor] = []
        for i in range(batch_size):
            row_ids = input_ids[i:i + 1]
            prompt_text = original_prompts[i]
            params = params_per_example[i]
            plan = self.plan(prompt_text, params)
            full = self._run_plan(
                plan, row_ids, prompt_text, params, base_generate, via_session,
                logits_processors, stopping_criteria, gen_kwargs,
            )
            final_sequences.append(self._finalize(full[0], original_lengths[i]))

        padded = self.tokenizer.pad(
            {"input_ids": [seq.tolist() for seq in final_sequences]},
            padding=True,
            return_tensors="pt",
        ).to(input_ids.device)
        return padded["input_ids"]

    def _run_plan(self, plan, row_ids, prompt_text, params, base_generate, via_session,
                  logits_processors, stopping_criteria, gen_kwargs) -> torch.Tensor:
        """Execute one example's plan; return the full spliced sequence `[1, L]`."""
        current = row_ids
        for phase in plan:
            if isinstance(phase, Fixed):
                text = phase.text(prompt_text, params) if callable(phase.text) else phase.text
                fixed_ids = self.tokenizer(
                    text, add_special_tokens=phase.add_special_tokens, return_tensors="pt"
                )["input_ids"].to(current.device)
                current = fixed_ids if phase.replace else torch.cat([current, fixed_ids], dim=1)
            elif isinstance(phase, Generated):
                current = self._generate_phase(
                    phase, current, base_generate, via_session,
                    logits_processors, stopping_criteria, gen_kwargs,
                )
            else:
                raise TypeError(f"Unknown phase type: {type(phase).__name__}")
        return current

    def _generate_phase(self, phase: Generated, current, base_generate, via_session,
                        logits_processors, stopping_criteria, gen_kwargs) -> torch.Tensor:
        """Run one Generated phase, composing its boundary with the pipeline's stop rules.

        On the session path the boundary lowers to normalized parameters (`until` as a stop
        string, `until_token_ids` as extra stop token ids, `budget` as a tightened
        `max_new_tokens`), so the phase runs on any backend; a raw generate callable receives the
        boundary as prompt-anchored criteria instead.
        """
        criteria = list(stopping_criteria) if stopping_criteria is not None else []
        kwargs = dict(gen_kwargs)

        if via_session:
            if phase.until is not None:
                existing = kwargs.get("stop_strings") or ()
                if isinstance(existing, str):
                    existing = (existing,)
                kwargs["stop_strings"] = (*existing, phase.until)
            if phase.until_token_ids:
                existing_ids = tuple(kwargs.get("stop_token_ids") or ())
                kwargs["stop_token_ids"] = (*existing_ids, *phase.until_token_ids)
            if phase.budget is not None:
                cap = kwargs.get("max_new_tokens")
                kwargs["max_new_tokens"] = phase.budget if cap is None else min(cap, phase.budget)
        else:
            current_len = current.size(1)
            if phase.until is not None:
                criteria.append(StopOnSubstring(self.tokenizer, phase.until, current_len))
            if phase.until_token_ids:
                criteria.append(StopOnTokens(phase.until_token_ids))
            if phase.budget is not None:
                criteria.append(BudgetTokens(phase.budget, current_len))
                if "max_new_tokens" not in kwargs:
                    kwargs["max_new_tokens"] = phase.budget

        extra = stack_generate_kwargs(logits_processors, StoppingCriteriaList(criteria))

        attention_mask = torch.ones_like(current)
        outputs = base_generate(input_ids=current, attention_mask=attention_mask, **extra, **kwargs)
        if not isinstance(outputs, torch.Tensor):
            outputs = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
        return outputs

    def _finalize(self, out_ids: torch.Tensor, original_length: int) -> torch.Tensor:
        """Apply the extract_after output rule (or return the full spliced sequence)."""
        if self.extract_after is None:
            return out_ids
        keep_prefix = out_ids[:original_length]
        decoded = self.tokenizer.decode(out_ids, skip_special_tokens=False)
        remainder_txt = decoded.rsplit(self.extract_after, 1)[-1].lstrip()
        remainder_ids = (
            self.tokenizer(remainder_txt, add_special_tokens=False, return_tensors="pt")["input_ids"]
            .to(out_ids.device)
            .squeeze(0)
        )
        return torch.cat([keep_prefix, remainder_ids], dim=0)
