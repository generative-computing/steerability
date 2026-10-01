"""Phase plans (`Fixed`, `Generated`) and `PhasedDriver`, the decoding driver that runs them.

A phase plan is a list of `Fixed` phases (append text without generating) and `Generated` phases
(generate until a boundary). A plan entry may also be a callable that receives a `PlanState` and
returns the phase to apply, or None to skip the entry. `PhasedDriver` runs the plans of all rows of a
batch together, one phase at a time. Every `Generated` phase generates through the session with the
composed logits processors and stopping criteria. When the decoded continuation contains an optional
`extract_after` marker (e.g., `"</think>"`), only the text after the marker's last occurrence is
returned as the continuation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from steerability.algorithms.core.execution.contracts import Requirements
from steerability.algorithms.core.execution.session_utils import session_generate_items
from steerability.algorithms.output_control.base import DecodingDriver, stack_generate_kwargs


@dataclass(frozen=True)
class Fixed:
    """A phase that appends fixed text to the stream without generating.

    Attributes:
        text: The text to append, as a string or a callable `(prompt_text, params) -> str`. A callable
            receives the row's decoded prompt and the row's plan parameters.
        replace: When True, the tokens of the phase replace the stream instead of being appended. This
            is used for a phase whose text is a rewritten prompt that contains the original prompt. A
            replacing phase does not count against `max_new_tokens`. The continuation returned for the
            sequence is the part of its final stream after the length of the original prompt.
        add_special_tokens: Whether the tokenizer adds special tokens to the text. Set it to True for a
            replacing phase whose text becomes the new prompt, and leave it False (default) for
            appended text.
    """

    text: str | Callable[[str, dict], str]
    replace: bool = False
    add_special_tokens: bool = False


@dataclass(frozen=True)
class Generated:
    """A phase that generates until the first of its boundaries is reached.

    The boundaries are the `until` text, a token in `until_token_ids`, and the `budget`. They apply
    together with the pipeline's stopping criteria and the caller's stop rules. The tokens that end
    the phase stay in the stream. `until_token_ids` is the portable way to end a phase on a delimiter
    that tokenizes to a special token. A stop string cannot match such a delimiter on vLLM, since vLLM
    removes special tokens (`skip_special_tokens=True`) before it matches stop strings.

    Attributes:
        until: Text that ends the phase once the generated text contains it, added to the call's stop
            strings. None disables it.
        until_token_ids: Token ids that end the phase when one of them is the last generated token,
            added to the call's stop token ids. The ids are converted to a tuple of ints. Empty
            disables it.
        budget: Maximum number of new tokens for the phase. The phase generates at most the smaller
            of `budget` and the tokens that remain under the caller's `max_new_tokens`. None disables
            it.
    """

    until: str | None = None
    until_token_ids: tuple[int, ...] = ()
    budget: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "until_token_ids", tuple(int(i) for i in self.until_token_ids))


StopCause = Literal["until", "until_token", "budget", "eos", "length"]


@dataclass(frozen=True, slots=True)
class PlanState:
    """A read-only view of one candidate sequence, passed to a callable plan entry.

    Attributes:
        last_stop: What ended the sequence's most recent `Generated` phase, or None before the
            first one ends. The values are checked in the following order:

                - `"until_token"`: the last generated token is one of the phase's
                  `until_token_ids`.
                - `"until"`: the phase's decoded continuation contains the phase's `until` text.
                - `"budget"`: the phase generated `budget` tokens, and `budget` was smaller than
                  the tokens that remained under the caller's `max_new_tokens`.
                - `"length"`: the phase generated the tokens that remained under the caller's
                  `max_new_tokens`.
                - `"eos"`: any other end, e.g., an eos token or a stop rule of the caller or of
                  the pipeline.
        appended: Number of tokens the plan has appended to the sequence, counted against
            `max_new_tokens`.
        text: The decoded stream after the prompt, with special tokens kept.
    """

    last_stop: StopCause | None
    appended: int
    text: str


@dataclass(slots=True, eq=False)
class _Sequence:
    """The state of one candidate sequence while its row's plan runs.

    Attributes:
        row: Index of the prompt row the sequence continues.
        ids: The unpadded 1-D stream so far, i.e., the prompt (or the text of the last replacing
            `Fixed` phase) followed by every appended and generated token.
        plan: The row's phase plan.
        prompt_length: The length of the row's unpadded prompt.
        cursor: Index of the next plan entry to apply.
        appended: Number of tokens the phases have appended, counted against `max_new_tokens`.
        done: Whether the sequence applies no further phases.
        last_stop: What ended the sequence's most recent `Generated` phase (see `PlanState`), or
            None before the first one ends.
        pending: The `Generated` phase at the cursor, once the entry at the cursor has been
            resolved, or None.
    """

    row: int
    ids: torch.Tensor
    plan: list
    prompt_length: int
    cursor: int = 0
    appended: int = 0
    done: bool = False
    last_stop: StopCause | None = None
    pending: Generated | None = None


def _phase_ceiling(budget: int | None, max_new_tokens: int | None, appended: int) -> int | None:
    """Return the token limit of a `Generated` phase.

    Args:
        budget: The phase's `budget`, or None.
        max_new_tokens: The caller's token limit per candidate, or None.
        appended: Number of tokens the sequence has appended.

    Returns:
        The smaller of `budget` and `max_new_tokens - appended`, ignoring a bound that is None, or None
        when both are None.
    """
    remaining = None if max_new_tokens is None else max_new_tokens - appended
    bounds = [bound for bound in (budget, remaining) if bound is not None]
    return min(bounds) if bounds else None


class PhasedDriver(DecodingDriver):
    """Decoding driver that runs a plan of fixed and generated phases for each prompt.

    A subclass implements `plan(prompt_text, params)`, which returns the plan for one prompt row.
    A plan is a list of `Fixed` and `Generated` phases, and an entry may also be a callable
    `(state) -> Fixed | Generated | None`. The subclass is constructed with the arguments listed
    under `Args:`. Alternatively, it sets an `Args` dataclass and overrides `_configure()` to set
    `extract_after`, and the constructor then validates the subclass's `Args`. `decode()` runs the
    plans in three steps:

    1. **Preparation**: pad positions are removed from each row (using the attention mask), and
       `plan()` is called once per row with the decoded prompt. Each row is expanded into
       `num_return_sequences` candidate sequences that share the row's plan.
    2. **Phases**: the plans run one phase at a time over all sequences. A `Fixed` phase appends the
       tokens of its text, or replaces the stream when `replace=True`. A `Generated` phase generates
       through the session with the composed logits processors and stopping criteria, one candidate
       per sequence. Its `until` is passed as a stop string, its `until_token_ids` as stop token ids,
       and its token limit as `max_new_tokens`. Because the boundaries are generation parameters,
       generated phases run on any backend. Sequences whose current phases have the same boundaries
       and token limit, and which have appended the same number of tokens, share one session call.
       After each `Generated` phase, the sequence records what ended the phase
       (`PlanState.last_stop`). A callable entry is evaluated when a sequence reaches it, with that
       sequence's `PlanState`, and returns the phase to apply or None to skip the entry. Each
       candidate therefore follows its own path through a plan with callable entries.
    3. **Output**: when `extract_after` is set and a candidate's decoded continuation contains the
       marker, the continuation is replaced by the re-tokenized text after the marker's last
       occurrence. A candidate whose continuation does not contain the marker (e.g., one whose
       thinking reached the token limit) keeps its full continuation.

    The caller's `max_new_tokens` limits the tokens appended to each candidate across all phases.
    Fixed and generated tokens both count, and a replacing `Fixed` phase counts nothing. A
    `Generated` phase generates at most the smaller of its `budget` and the tokens that remain. A
    `Fixed` phase is always appended in full. A sequence that reaches the limit skips the rest of its
    plan. Its continuation then contains at least `max_new_tokens` tokens before `extract_after`
    applies (more when a `Fixed` phase crosses the limit). The pipeline classifies such a
    continuation as finish reason `"length"` unless it contains a stop string or ends in a stop or
    eos token. The caller's `min_new_tokens` applies to each `Generated` phase and is reduced to the
    phase's token limit when larger.

    Each returned row is the padded prompt row followed by one candidate's continuation, right-padded
    to a common length. Rows are ordered by prompt row and then by candidate.

    Each candidate generates as its own sequence. Under sampling, the candidates of a prompt draw
    different samples under either `seed_scope`. Under greedy decoding they are identical, and beam
    search (`num_beams > 1`) with more than one candidate raises `ValueError`. Later phases are
    seeded differently from the first, since each session call derives its own seeds from the
    call-level `seed`.

    `whole_batch_rollouts` is False because the session calls cover subsets of the batch. With
    enabled in-process state controls, the pipeline accepts one prompt row per call.

    The following `runtime_kwargs` are accepted:

    - `"params"`: A mapping of plan parameters passed to `plan()` and to callable `Fixed` texts. A
      list or tuple value contains one entry per row, and any other value applies to every row.

    Args:
        extract_after: A marker, e.g., `"</think>"`. When a candidate's decoded continuation contains
            it, only the text after its last occurrence is returned. None (default) returns the full
            continuation.

    Attributes:
        tokenizer: The tokenizer that decodes prompts and tokenizes `Fixed` texts, attached by the
            pipeline or by a subclass's `steer()`, or None before it is attached.
    """

    whole_batch_rollouts: bool = False
    tokenizer: PreTrainedTokenizerBase | None = None

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

    def __init__(self, *args, **kwargs):
        # a subclass with an `Args` dataclass validates it and sets its fields in `_configure()`
        if self.Args is not None:
            super().__init__(*args, **kwargs)
        else:
            self._init_fields(*args, **kwargs)

    def _init_fields(self, extract_after: str | None = None) -> None:
        """Set the fields of a driver constructed directly with the arguments listed under `Args:`."""
        self.extract_after = extract_after

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
        """Split the `params` runtime kwarg into one mapping per row.

        Args:
            runtime_kwargs: The call's runtime kwargs.
            batch_size: The number of prompt rows.

        Returns:
            One parameter mapping per row. A mapping that contains a list or tuple value is split by
            row, with each list or tuple indexed by row and every other value shared. Any other
            `params` value is used as is for every row, and a missing or None value gives an empty
            mapping for every row.

        Raises:
            ValueError: If a list or tuple value does not have one entry per row.
        """
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
        """Run each row's phase plan over its candidate sequences.

        Args:
            input_ids: Prompt token ids of shape `[B, T]`, left-padded by the pipeline. A 1-D tensor
                is treated as one row.
            attention_mask: Prompt attention mask with the same shape as `input_ids`, or None when no
                row is padded.
            model: The pipeline's model, used only to read the eos ids of its generation config, or
                None.
            logits_processors: The composed `LogitsProcessorList`, applied in every generated phase.
            stopping_criteria: The composed `StoppingCriteriaList`, applied in every generated phase.
            runtime_kwargs: Per-call parameters. The `params` entry is passed to `plan()` (see the
                class docstring).
            session: The `SteeringSession` on which every generated phase runs.
            **gen_kwargs: Generation keyword arguments. `max_new_tokens` limits each candidate across
                all phases, and `num_return_sequences` sets the number of candidates per row.

        Returns:
            A tensor of shape `[B * n, T + L]`, where `n` is `num_return_sequences` and `L` is the
            length of the longest continuation. Rows are ordered by prompt row and then by
            candidate. Each row is the padded prompt followed by one candidate's continuation,
            right-padded.

        Raises:
            RuntimeError: If no tokenizer is attached because `steer()` has not run.
            ValueError: If no session was provided, if beam search (`num_beams > 1`) is requested
                with more than one candidate, or if a list or tuple value in the `params` runtime
                kwarg does not have one entry per row.
            TypeError: If a plan entry is neither a `Fixed` phase, a `Generated` phase, nor a
                callable, or if a callable entry returns something other than one of these
                phases or None.
        """
        self._check_ready(session)
        input_ids, attention_mask = self._as_batch(input_ids, attention_mask)
        rows = self._unpadded_rows(input_ids, attention_mask)
        prompt_texts = [self.tokenizer.decode(row, skip_special_tokens=True) for row in rows]
        params_per_example = self._params_per_example(runtime_kwargs or {}, len(rows))
        plans = [self.plan(text, params) for text, params in zip(prompt_texts, params_per_example)]
        return self._run_plans(
            input_ids, rows, plans, prompt_texts, params_per_example, model, session,
            logits_processors, stopping_criteria, gen_kwargs,
        )

    def _check_ready(self, session) -> None:
        """Check that a tokenizer is attached and a session was provided.

        Args:
            session: The session passed to `decode()`, or None.

        Raises:
            RuntimeError: If no tokenizer is attached.
            ValueError: If `session` is None.
        """
        if self.tokenizer is None:
            raise RuntimeError(f"{type(self).__name__} requires a tokenizer; steer() must run first.")
        if session is None:
            raise ValueError(f"{type(self).__name__} generates through a session; decode() received none.")

    @staticmethod
    def _as_batch(input_ids: torch.Tensor, attention_mask: torch.Tensor | None):
        """Convert a 1-D prompt and attention mask to a batch of one row.

        Args:
            input_ids: Prompt token ids of shape `[T]` or `[B, T]`.
            attention_mask: Attention mask of shape `[T]` or `[B, T]`, or None.

        Returns:
            The `(input_ids, attention_mask)` pair, with each 1-D tensor given a leading batch
            dimension. 2-D tensors and None are returned unchanged.
        """
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if attention_mask is not None and attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)
        return input_ids, attention_mask

    @staticmethod
    def _unpadded_rows(input_ids: torch.Tensor, attention_mask: torch.Tensor | None) -> list[torch.Tensor]:
        """Return each row's prompt tokens with the pad positions removed.

        Args:
            input_ids: Prompt token ids of shape `[B, T]`.
            attention_mask: Attention mask with the same shape as `input_ids`, or None when no row is
                padded.

        Returns:
            One 1-D tensor per row without the positions where the mask is 0. Every row is returned
            in full when `attention_mask` is None.
        """
        if attention_mask is None:
            return [row for row in input_ids]
        return [row[mask.bool()] for row, mask in zip(input_ids, attention_mask)]

    def _run_plans(self, input_ids, rows, plans, prompt_texts, params_per_example, model, session,
                   logits_processors, stopping_criteria, gen_kwargs) -> torch.Tensor:
        """Run one plan per row over the row's candidate sequences and return the padded output.

        Each row is expanded into `num_return_sequences` sequences. In each round, every sequence
        applies its `Fixed` phases up to its next `Generated` phase. The pending sequences are then
        grouped by the phase's `until` and `until_token_ids`, the phase's token limit, and the number
        of tokens the sequence has appended. Each group generates in one session call, with its
        members ordered by row and then by candidate. Equal appended counts place the end of every
        member's original prompt at the same position of the left-padded call, which is where
        in-process state hooks anchor prompt-relative token scopes. The rounds repeat until no
        sequence has a pending `Generated` phase.

        Args:
            input_ids: The padded prompt batch of shape `[B, T]` that prefixes the returned rows.
            rows: Each row's unpadded prompt ids as a 1-D tensor.
            plans: One phase plan per row.
            prompt_texts: Each row's decoded prompt, passed to callable `Fixed` texts.
            params_per_example: Each row's plan parameters, passed to callable `Fixed` texts.
            model: The pipeline's model, used to read the eos ids of its generation config, or None.
            session: The `SteeringSession` on which the generated phases run.
            logits_processors: The composed `LogitsProcessorList`.
            stopping_criteria: The composed `StoppingCriteriaList`.
            gen_kwargs: The caller's generation keyword arguments. The dict is copied and not
                modified.

        Returns:
            A tensor of shape `[B * n, T + L]`, where `n` is `num_return_sequences` and `L` is the
            length of the longest continuation. Rows are ordered by prompt row and then by
            candidate. Each row is the padded prompt followed by one candidate's continuation,
            right-padded.

        Raises:
            ValueError: If beam search (`num_beams > 1`) is requested with more than one
                candidate.
            TypeError: If a plan entry is neither a `Fixed` phase, a `Generated` phase, nor a
                callable, or if a callable entry returns something other than one of these
                phases or None.
        """
        gen_kwargs = dict(gen_kwargs)
        num_candidates = gen_kwargs.pop("num_return_sequences", None) or 1
        max_new_tokens = gen_kwargs.pop("max_new_tokens", None)
        if num_candidates > 1 and (gen_kwargs.get("num_beams") or 1) > 1:
            raise ValueError(
                f"{type(self).__name__} generates each of the {num_candidates} candidates as its own sequence, so "
                "beam search cannot return distinct beams as candidates; request one candidate with num_beams > 1, "
                "or sample the candidates."
            )
        eos_token_ids = self._eos_token_ids(model, gen_kwargs)
        terminal_eos_ids = set(eos_token_ids)
        if self.tokenizer.eos_token_id is not None:
            terminal_eos_ids.add(int(self.tokenizer.eos_token_id))
        stacks = stack_generate_kwargs(logits_processors, stopping_criteria)
        sequences = [
            _Sequence(row=index, ids=rows[index], plan=plans[index], prompt_length=rows[index].numel())
            for index in range(len(rows))
            for _ in range(num_candidates)
        ]
        fixed_ids: dict[tuple[str, bool], torch.Tensor] = {}

        while True:
            pending = [
                sequence for sequence in sequences
                if self._advance(sequence, max_new_tokens, prompt_texts, params_per_example, fixed_ids)
            ]
            if not pending:
                break
            # insertion order keeps every group's rows in row-major, candidate-minor order, and
            # the appended count in the key ends every member's prompt at the same packed position
            groups: dict[tuple, list[_Sequence]] = {}
            for sequence in pending:
                phase = sequence.pending
                ceiling = _phase_ceiling(phase.budget, max_new_tokens, sequence.appended)
                key = (phase.until, phase.until_token_ids, ceiling, sequence.appended)
                groups.setdefault(key, []).append(sequence)
            for (until, until_token_ids, ceiling, _), members in groups.items():
                phase_kwargs = self._phase_kwargs(until, until_token_ids, ceiling, gen_kwargs)
                results = session_generate_items(
                    session, [member.ids for member in members], eos_token_ids=eos_token_ids,
                    **stacks, **phase_kwargs,
                )
                for member, (continuation, reason) in zip(members, results):
                    remaining = None if max_new_tokens is None else max_new_tokens - member.appended
                    member.last_stop = self._stop_cause(
                        member.pending, continuation, reason, ceiling, remaining, terminal_eos_ids,
                    )
                    member.ids = torch.cat([member.ids, continuation.to(member.ids.device)])
                    member.appended += continuation.numel()
                    member.cursor += 1
                    member.pending = None

        return self._assemble(input_ids, rows, sequences)

    def _stop_cause(self, phase: Generated, continuation: torch.Tensor, reason: str | None,
                    ceiling: int | None, remaining: int | None, eos_ids: set[int]) -> StopCause:
        """Classify what ended one `Generated` phase of one sequence.

        A stop on the phase's `until_token_ids` or `until` text comes first. The session's
        `"stop"` and `"eos"` finish reasons, and a last token that is an eos id, then count as
        `"eos"`. Otherwise a continuation that reached the phase's token limit is classified by
        the bound that set the limit, and a `budget` equal to the remaining tokens counts as
        `"length"`.

        Args:
            phase: The `Generated` phase that ran.
            continuation: The phase's generated ids as a 1-D tensor.
            reason: The finish reason the session reported for the phase, or None.
            ceiling: The phase's token limit, or None for no limit.
            remaining: The tokens that remained under the caller's `max_new_tokens` before the
                phase, or None when the caller set no limit.
            eos_ids: The eos ids that end a phase (the tokenizer's eos and the generation
                config's eos ids).

        Returns:
            The stop cause, as described under `PlanState.last_stop`.
        """
        last = int(continuation[-1]) if continuation.numel() else None
        if last is not None and last in phase.until_token_ids:
            return "until_token"
        if phase.until is not None and phase.until in self.tokenizer.decode(continuation, skip_special_tokens=False):
            return "until"
        ended_on_stop = reason in ("stop", "eos") or (last is not None and last in eos_ids)
        if not ended_on_stop and ceiling is not None and continuation.numel() >= ceiling:
            if phase.budget is not None and (remaining is None or phase.budget < remaining):
                return "budget"
            return "length"
        return "eos"

    def _advance(self, sequence: _Sequence, max_new_tokens: int | None, prompt_texts: list[str],
                 params_per_example: list[dict], fixed_ids: dict[tuple[str, bool], torch.Tensor]) -> bool:
        """Apply the `Fixed` phases of `sequence` up to its next `Generated` phase.

        A callable entry is evaluated with the sequence's `PlanState` when the cursor reaches it.
        It is skipped when it returns None, and its returned phase is applied otherwise. The
        sequence is marked done when its plan is exhausted or when its appended tokens reach
        `max_new_tokens`.

        Args:
            sequence: The sequence to advance, updated in place.
            max_new_tokens: The caller's token limit per candidate, or None.
            prompt_texts: Each row's decoded prompt, passed to callable `Fixed` texts.
            params_per_example: Each row's plan parameters, passed to callable `Fixed` texts.
            fixed_ids: The tokenized `Fixed` texts keyed by `(text, add_special_tokens)`, updated in
                place with each new text.

        Returns:
            True when the sequence's current phase is a `Generated` phase, which is then set as
            `sequence.pending`, otherwise False.

        Raises:
            TypeError: If a plan entry is neither a `Fixed` phase, a `Generated` phase, nor a
                callable, or if a callable entry returns something other than one of these
                phases or None.
        """
        while not sequence.done:
            if sequence.pending is not None:
                return True
            if sequence.cursor >= len(sequence.plan):
                sequence.done = True
                break
            if max_new_tokens is not None and sequence.appended >= max_new_tokens:
                sequence.done = True
                break
            phase = sequence.plan[sequence.cursor]
            if callable(phase):
                phase = phase(self._plan_state(sequence))
                if phase is None:
                    sequence.cursor += 1
                    continue
                if not isinstance(phase, (Fixed, Generated)):
                    raise TypeError(
                        f"A callable plan entry returned {type(phase).__name__}; return a Fixed or Generated "
                        "phase, or None to skip the entry."
                    )
            if isinstance(phase, Generated):
                sequence.pending = phase
                return True
            if not isinstance(phase, Fixed):
                raise TypeError(f"Unknown phase type: {type(phase).__name__}")
            text = phase.text
            if callable(text):
                text = text(prompt_texts[sequence.row], params_per_example[sequence.row])
            key = (text, phase.add_special_tokens)
            if key not in fixed_ids:
                fixed_ids[key] = self.tokenizer(
                    text, add_special_tokens=phase.add_special_tokens, return_tensors="pt",
                )["input_ids"][0]
            ids = fixed_ids[key].to(sequence.ids.device)
            sequence.cursor += 1
            if phase.replace:
                sequence.ids = ids
                continue
            sequence.ids = torch.cat([sequence.ids, ids])
            sequence.appended += ids.numel()
        return False

    def _plan_state(self, sequence: _Sequence) -> PlanState:
        """Return the `PlanState` of `sequence`, decoding its stream after the prompt."""
        text = self.tokenizer.decode(sequence.ids[sequence.prompt_length:], skip_special_tokens=False)
        return PlanState(last_stop=sequence.last_stop, appended=sequence.appended, text=text)

    @staticmethod
    def _eos_token_ids(model: PreTrainedModel | None, gen_kwargs: dict) -> tuple[int, ...]:
        """Return the eos ids of the generated phases in addition to the tokenizer's eos id.

        The ids decide whether a trailing pad token of a phase's continuation is kept as an emitted
        eos when the pad token is also an eos token.

        Args:
            model: The pipeline's model, or None.
            gen_kwargs: The caller's generation keyword arguments.

        Returns:
            The caller's `eos_token_id` when `gen_kwargs` contains that key, otherwise the
            `eos_token_id` of the model's generation config, as a tuple of ints. The tuple is empty
            when the value is None or no model is given.
        """
        if "eos_token_id" in gen_kwargs:
            configured = gen_kwargs["eos_token_id"]
        else:
            configured = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
        if configured is None:
            return ()
        if isinstance(configured, int):
            return (configured,)
        return tuple(int(token_id) for token_id in configured)

    @staticmethod
    def _phase_kwargs(until: str | None, until_token_ids: tuple[int, ...], ceiling: int | None,
                      gen_kwargs: dict) -> dict:
        """Return the session generation kwargs for one `Generated` phase.

        `until` is appended to the caller's stop strings, and `until_token_ids` are appended to the
        caller's stop token ids. `ceiling` becomes `max_new_tokens`, and `min_new_tokens` is reduced
        to `ceiling` when larger.

        Args:
            until: The phase's stop text, or None.
            until_token_ids: The phase's stop token ids.
            ceiling: The phase's token limit, or None for no limit.
            gen_kwargs: The caller's generation keyword arguments, which are not modified.

        Returns:
            A new dict of generation keyword arguments.
        """
        kwargs = dict(gen_kwargs)
        if until is not None:
            existing = kwargs.get("stop_strings") or ()
            if isinstance(existing, str):
                existing = (existing,)
            kwargs["stop_strings"] = (*existing, until)
        if until_token_ids:
            existing_ids = tuple(kwargs.get("stop_token_ids") or ())
            kwargs["stop_token_ids"] = (*existing_ids, *until_token_ids)
        if ceiling is not None:
            kwargs["max_new_tokens"] = ceiling
            if kwargs.get("min_new_tokens") is not None and kwargs["min_new_tokens"] > ceiling:
                kwargs["min_new_tokens"] = ceiling
        return kwargs

    def _assemble(self, input_ids: torch.Tensor, rows: list[torch.Tensor],
                  sequences: list[_Sequence]) -> torch.Tensor:
        """Build the output tensor from the finished sequences.

        The `extract_after` rule is applied to each sequence, and the part of its stream after the
        length of the unpadded prompt is appended to the row's padded prompt. The rows are
        right-padded to a common length with the tokenizer's pad token id, its eos token id when no
        pad token is set, or 0 when neither is set.

        Args:
            input_ids: The padded prompt batch of shape `[B, T]`.
            rows: Each row's unpadded prompt ids as a 1-D tensor.
            sequences: The finished sequences, ordered by row and then by candidate.

        Returns:
            A tensor of shape `[len(sequences), T + L]`, where `L` is the length of the longest
            continuation.
        """
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id or 0
        full_rows = []
        for sequence in sequences:
            prompt_length = rows[sequence.row].numel()
            finalized = self._finalize(sequence.ids, prompt_length)
            continuation = finalized[prompt_length:].to(device=input_ids.device, dtype=input_ids.dtype)
            full_rows.append(torch.cat([input_ids[sequence.row], continuation]))
        width = max(row.numel() for row in full_rows)
        return torch.stack([
            torch.nn.functional.pad(row, (0, width - row.numel()), value=pad_token_id) for row in full_rows
        ])

    def _finalize(self, out_ids: torch.Tensor, original_length: int) -> torch.Tensor:
        """Apply the `extract_after` rule to one sequence.

        Args:
            out_ids: The sequence's full 1-D stream.
            original_length: The length of the row's unpadded prompt.

        Returns:
            `out_ids` unchanged when `extract_after` is None or when the decoded continuation (the
            tokens after `original_length`) does not contain the marker. Otherwise, the first
            `original_length` tokens followed by the text after the marker's last occurrence, with
            leading whitespace removed and tokenized without special tokens.
        """
        if self.extract_after is None:
            return out_ids
        continuation = self.tokenizer.decode(out_ids[original_length:], skip_special_tokens=False)
        if self.extract_after not in continuation:
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
