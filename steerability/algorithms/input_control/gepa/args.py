"""Arguments for the GEPA input control."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from steerability.algorithms.core.base_args import BaseArgs
from steerability.algorithms.input_control.common.memory.text import TextMemory


@dataclass
class GEPAArgs(BaseArgs):
    """Arguments for GEPA, reflective prompt optimization (single-system-prompt variant).

    Optimizes a single system prompt via GEPA's reflective genetic search (Agrawal et al.,
    2025). Scoring ingredients are supplied directly (`row_scorer`, optional `feedback_fn`,
    optional `format_query`); GEPA runs generation, scoring, and feedback internally. See the
    `GEPA` control docstring for scope.

    Attributes:
        seed_instruction: Seed system prompt; a non-empty string.
        train_set: Task instances (`list[dict]`); split internally into D_feedback and
            D_pareto. Rows are forwarded to `row_scorer`, `feedback_fn`, and `format_query`.
        row_scorer: Per-instance metric `(output: str, row: dict) -> float` (the paper's `µ`).
            Required, because Pareto selection needs per-instance scores.
        feedback_fn: Optional `(output: str, row: dict, score: float) -> str` returning the
            textual feedback GEPA reflects on (the paper's `µf`). Defaults to a stringified
            score when None.
        format_query: Optional `(row: dict) -> str` mapping a row to the user query. Falls
            back to `row["input"]` when None.
        gen_kwargs: Optional task-model generation kwargs.
        reflection_lm: LLM used for the reflective proposer. Defaults to the task model.
        reflection_tokenizer: Tokenizer paired with `reflection_lm`. Defaults to the task
            tokenizer. Must be supplied iff `reflection_lm` is supplied.
        budget: Maximum number of rollouts (sum of internal batch sizes) consumed by the
            search.
        minibatch_size: Size of D_feedback minibatch sampled per genetic step.
        pareto_set_size: |D_pareto|, the instances reserved for the persistent score matrix.
        meta_prompt: Override the reflective meta-prompt template; defaults to
            `gepa.utils.reflective_meta_prompt.GEPA_DEFAULT` (the paper's Appendix C prompt,
            which asks the reflection LM to return the new instruction within fenced blocks).
        proposer_gen_kwargs: Generation kwargs for the reflection LLM proposer. The default
            permits long outputs (`max_new_tokens=1024`) since GEPA evolves long, multi-section
            instructions; responses are parsed with `parse_fenced_or_whole`, which preserves
            their full structure. Override to cap or otherwise tune generation.
        progress_callback: Optional callback invoked once per search step with a small dict
            (`step`, `event`, `parent_idx`, `parent_score`, `candidate_score`, `accepted`,
            `pool_size`, `best_mean`, `proposed`). Opt-in; no-op when None.
        seed: Optional RNG seed for the genetic loop (sampling + minibatching).
    """

    seed_instruction: str = field(
        default="",
        metadata={"help": "Seed system prompt; a non-empty string."},
    )

    train_set: list[dict] = field(
        default_factory=list,
        metadata={"help": "Task instances; split internally into D_feedback and D_pareto."},
    )

    row_scorer: Callable[[str, dict], float] | None = field(
        default=None,
        metadata={"help": "Per-instance metric (output, row) -> float (the paper's µ)."},
    )

    feedback_fn: Callable[[str, dict, float], str] | None = field(
        default=None,
        metadata={"help": "Optional textual feedback (output, row, score) -> str (the paper's µf)."},
    )

    format_query: Callable[[dict], str] | None = field(
        default=None,
        metadata={"help": "Optional (row) -> query; falls back to row['input']."},
    )

    gen_kwargs: dict | None = field(
        default=None,
        metadata={"help": "Task-model generation kwargs."},
    )

    reflection_lm: Any = field(
        default=None,
        metadata={"help": "LLM used for reflection. Defaults to the task model when None."},
    )

    reflection_tokenizer: Any = field(
        default=None,
        metadata={"help": "Tokenizer paired with `reflection_lm`. Defaults to the task tokenizer."},
    )

    budget: int = field(
        default=300,
        metadata={"help": "Maximum number of rollouts consumed by the search."},
    )

    minibatch_size: int = field(
        default=3,
        metadata={"help": "Size of D_feedback minibatch sampled per genetic step."},
    )

    pareto_set_size: int = field(
        default=30,
        metadata={"help": "|D_pareto|, the instances reserved for the persistent score matrix."},
    )

    meta_prompt: str | None = field(
        default=None,
        metadata={"help": "Override the reflective meta-prompt; defaults to GEPA_DEFAULT."},
    )

    proposer_gen_kwargs: dict | None = field(
        default=None,
        metadata={
            "help": "Generation kwargs for the reflection LLM proposer. Defaults to long outputs "
            "(max_new_tokens=1024) so evolved instructions are not truncated."
        },
    )

    progress_callback: Callable[[dict], None] | None = field(
        default=None,
        metadata={"help": "Optional callback invoked once per search step with a small dict "
                          "(step, event, scores, accepted, pool_size, best_mean, proposed). "
                          "Opt-in; no-op when None."},
    )

    seed: int | None = field(
        default=None,
        metadata={"help": "Optional RNG seed for the genetic loop (sampling + minibatching)."},
    )

    memory: TextMemory | dict | None = field(
        default=None,
        metadata={
            "help": (
                "Precomputed memory (`TextMemory(slots={'instruction': ...})` or a plain "
                "`{'slots': {...}}` dict). When provided, steer() installs it directly and "
                "skips the reflective search; the search-only args stay inert."
            )
        },
    )

    def __post_init__(self) -> None:
        if isinstance(self.memory, dict):
            self.memory = TextMemory(slots=dict(self.memory.get("slots", self.memory)))
        if self.memory is not None:
            if "instruction" not in self.memory:
                raise ValueError("`memory` must carry an 'instruction' slot.")
            return  # a precomputed memory makes the search args inert
        if not isinstance(self.seed_instruction, str) or not self.seed_instruction:
            raise ValueError("`seed_instruction` must be a non-empty str.")

        if self.row_scorer is None or not callable(self.row_scorer):
            raise ValueError("`row_scorer` must be a callable (output, row) -> float.")

        if not self.train_set:
            raise ValueError("`train_set` must be non-empty.")

        if (self.reflection_lm is None) != (self.reflection_tokenizer is None):
            raise ValueError(
                "`reflection_lm` and `reflection_tokenizer` must both be provided or both be None."
            )

        if self.budget <= 0:
            raise ValueError("budget must be positive.")
        if self.minibatch_size <= 0:
            raise ValueError("minibatch_size must be positive.")
        if self.pareto_set_size <= 0:
            raise ValueError("pareto_set_size must be positive.")
