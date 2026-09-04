"""GEPA, reflective prompt optimization (single-system-prompt variant)."""
from __future__ import annotations

import logging
import random
from statistics import mean
from typing import Any

import torch

from aisteer360.algorithms.core.execution.access import ModelAccess
from aisteer360.algorithms.core.execution.session_utils import SessionLM
from aisteer360.algorithms.input_control.base import InputControl
from aisteer360.algorithms.input_control.common.budget import RolloutBudget
from aisteer360.algorithms.input_control.common.formatters.system_prompt import SystemPromptFormatter
from aisteer360.algorithms.input_control.common.generation import generate_with_system_prompt
from aisteer360.algorithms.input_control.common.memory.text import TextMemory
from aisteer360.algorithms.input_control.common.proposers.llm_meta_prompt import LLMMetaPromptProposer
from aisteer360.algorithms.input_control.common.proposers.utils.parsing import parse_fenced_or_whole
from aisteer360.algorithms.input_control.gepa.args import GEPAArgs
from aisteer360.algorithms.input_control.gepa.utils import pareto_sampling, reflective_meta_prompt
from aisteer360.algorithms.input_control.gepa.utils.pool import CandidatePool
from aisteer360.algorithms.input_control.gepa.utils.reflective_dataset import build_records
from aisteer360.algorithms.input_control.gepa.utils.reflective_meta_prompt import render_records

logger = logging.getLogger(__name__)


class GEPA(InputControl):
    """GEPA, reflective prompt optimization (single-system-prompt variant).

    Optimizes a single system prompt via GEPA's reflective genetic search (Agrawal et al.,
    2025), restricted to the single-module (|M| = 1) case. It implements:

      - Reflective prompt mutation: an LLM rewrites the current instruction using
        natural-language feedback drawn from rollouts.
      - Pareto-based candidate selection: GEPA keeps an instance-wise Pareto frontier over a
        held-out set and samples parents by win-frequency (paper section 3.1, Algorithm 2).
      - A genetic candidate pool that accumulates lessons across successive mutations.

    Scope:

      - This control optimizes ONE system prompt. It does NOT support compound, multi-module
        AI systems, and it does NOT implement the system-aware merge / crossover operator
        (GEPA+Merge). Merge is a multi-module operation that recombines complementary modules
        across lineages, and is vacuous with a single prompt.
      - For compound, multi-module / multi-prompt optimization with module-level credit
        assignment and merge, use the reference implementation at
        [https://github.com/gepa-ai/gepa](https://github.com/gepa-ai/gepa) (or DSPy's GEPA
        integration).

    Scoring is supplied directly through a per-example `row_scorer` (the paper's metric `µ` at
    the instance level, required because Pareto selection needs per-instance scores), an
    optional `feedback_fn` returning the textual feedback GEPA reflects on (the paper's `µf`),
    and a `format_query` mapping a dataset row to the user query. GEPA runs generation, scoring,
    and feedback internally; there is no separate Evaluator object.

    `adapt_messages()` / `adapt()` inject the optimized instruction as the system prompt at
    inference time. Memory: `TextMemory(slots={"instruction": best_instruction})`.

    `reflection_lm` is optional and defaults to the task model. A separate, stronger reflection
    LM can help when the task model is small, but is not required.

    Reference:

      - "GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning"
        Agrawal et al., 2025
        [https://arxiv.org/abs/2507.19457](https://arxiv.org/abs/2507.19457)
    """

    Args = GEPAArgs
    supports_batching: bool = True

    # placeholders
    seed_instruction: str = ""
    train_set: list[dict] = []
    row_scorer: Any = None
    feedback_fn: Any = None
    format_query: Any = None
    gen_kwargs: dict | None = None
    reflection_lm: Any = None
    reflection_tokenizer: Any = None
    budget: int = 300
    minibatch_size: int = 3
    pareto_set_size: int = 30
    meta_prompt: str | None = None
    proposer_gen_kwargs: dict | None = None
    progress_callback: Any = None
    seed: int | None = None

    # state
    memory: TextMemory | None = None
    tokenizer: Any = None
    _formatter: SystemPromptFormatter | None = None

    def steer_access(self) -> ModelAccess:
        """`ModelAccess.ROLLOUTS`; task rollouts and reflection generate through the session,
        and adaptation is a formatter."""
        return ModelAccess.ROLLOUTS

    def steer(
        self,
        model=None,
        tokenizer=None,
        session=None,
        **kwargs,
    ) -> None:
        rng = random.Random(self.seed) if self.seed is not None else random.Random()
        self.tokenizer = tokenizer
        task_lm = SessionLM(session) if session is not None else model

        if self.reflection_lm is None:
            logger.info("GEPA: no `reflection_lm` supplied; reflection falls back to the task model.")
        reflection_lm = self.reflection_lm if self.reflection_lm is not None else task_lm
        reflection_tok = self.reflection_tokenizer if self.reflection_tokenizer is not None else tokenizer
        proposer = LLMMetaPromptProposer(
            llm=reflection_lm,
            tokenizer=reflection_tok,
            meta_prompt_template=self.meta_prompt or reflective_meta_prompt.GEPA_DEFAULT,
            parse_fn=parse_fenced_or_whole,
            gen_kwargs=self.proposer_gen_kwargs or {
                "max_new_tokens": 1024,
                "do_sample": True,
                "temperature": 1.0,
                "top_p": 0.95,
            },
        )

        d_feedback, d_pareto = self._split_dataset(rng)
        budget = RolloutBudget(self.budget)

        pool = CandidatePool()
        _, seed_scores, _ = self._run(task_lm, self.seed_instruction, d_pareto, with_feedback=False)
        budget.charge(len(d_pareto))
        pool.add(self.seed_instruction, seed_scores)

        step = 0
        self._emit_progress(step=0, event="seed", parent_idx=None, parent_score=None,
                            candidate_score=None, accepted=True, pool_size=len(pool.candidates),
                            best_mean=float(pool.scores.mean(axis=1).max()),
                            proposed=self.seed_instruction)

        while budget:
            parent_idx = pareto_sampling.sample(pool.frontier(), len(pool.candidates), rng=rng)

            mb_size = min(self.minibatch_size, len(d_feedback))
            minibatch = rng.sample(d_feedback, mb_size) if mb_size <= len(d_feedback) else list(d_feedback)
            if not minibatch:
                break
            if budget.remaining < len(minibatch):
                break

            parent_outputs, parent_scores, parent_feedback = self._run(
                task_lm, pool.candidates[parent_idx], minibatch, with_feedback=True
            )
            budget.charge(len(minibatch))
            parent_mb_mean = mean(parent_scores)

            query_texts = [self._format_query(row) for row in minibatch]
            records = build_records(query_texts, parent_outputs, parent_feedback)
            new_candidates = proposer.propose(
                seed=pool.candidates[parent_idx],
                n=1,
                context={"records": render_records(records)},
            )
            new_text = (new_candidates[0].strip() if new_candidates else "")
            if not new_text:
                continue

            if budget.remaining < len(minibatch):
                break
            _, cand_scores, _ = self._run(task_lm, new_text, minibatch, with_feedback=False)
            budget.charge(len(minibatch))
            cand_mb_mean = mean(cand_scores)
            if cand_mb_mean <= parent_mb_mean:  # strict improvement
                step += 1
                self._emit_progress(step=step, event="reject", parent_idx=parent_idx,
                                    parent_score=parent_mb_mean, candidate_score=cand_mb_mean,
                                    accepted=False, pool_size=len(pool.candidates),
                                    best_mean=float(pool.scores.mean(axis=1).max()),
                                    proposed=new_text)
                continue

            if budget.remaining < len(d_pareto):
                break
            _, cand_full, _ = self._run(task_lm, new_text, d_pareto, with_feedback=False)
            budget.charge(len(d_pareto))
            pool.add(new_text, cand_full)
            step += 1
            self._emit_progress(step=step, event="accept", parent_idx=parent_idx,
                                parent_score=parent_mb_mean, candidate_score=cand_mb_mean,
                                accepted=True, pool_size=len(pool.candidates),
                                best_mean=float(pool.scores.mean(axis=1).max()),
                                proposed=new_text)

        best = pool.candidates[pool.best_index()]
        self.memory = TextMemory(slots={"instruction": best})
        self._formatter = SystemPromptFormatter()

    def _run(
        self,
        task_lm,
        instruction: str,
        batch: list[dict],
        *,
        with_feedback: bool,
    ) -> tuple[list[str], list[float], list[str] | None]:
        queries = [self._format_query(row) for row in batch]
        outputs = generate_with_system_prompt(
            task_lm, self.tokenizer, instruction, queries, gen_kwargs=self.gen_kwargs
        )
        scores = [float(self.row_scorer(out, row)) for out, row in zip(outputs, batch)]
        feedback: list[str] | None = None
        if with_feedback:
            if self.feedback_fn is not None:
                feedback = [
                    self.feedback_fn(out, row, score)
                    for out, row, score in zip(outputs, batch, scores)
                ]
            else:
                feedback = [f"score={score:.4f}" for score in scores]
        return outputs, scores, feedback

    def _emit_progress(self, **info) -> None:
        if self.progress_callback is not None:
            self.progress_callback(info)

    def _format_query(self, row: dict) -> str:
        if self.format_query is not None:
            return self.format_query(row)
        return row["input"]

    def _split_dataset(self, rng: random.Random) -> tuple[list[dict], list[dict]]:
        train = list(self.train_set)
        rng.shuffle(train)
        pareto_n = min(self.pareto_set_size, len(train))
        if pareto_n >= len(train):
            d_pareto = train[:]
            d_feedback = train[:]  # fall back to overlap when train_set is tiny
        else:
            d_pareto = train[:pareto_n]
            d_feedback = train[pareto_n:] or train[:]
        return d_feedback, d_pareto

    def adapt_messages(
        self,
        messages: list[list[dict]],
        runtime_kwargs: dict | None = None,
    ) -> list[list[dict]] | None:
        if self.memory is None or self._formatter is None:
            raise RuntimeError("GEPA.adapt_messages called before .steer().")
        slot_memory = TextMemory(slots={"instruction": self.memory["instruction"]})
        return self._formatter.apply_to_messages(messages, slot_memory, runtime_kwargs)

    def adapt(
        self,
        input_ids: list[int] | torch.Tensor,
        runtime_kwargs: dict | None = None,
    ) -> list[int] | torch.Tensor:
        if self.memory is None or self._formatter is None or self.tokenizer is None:
            raise RuntimeError("GEPA.adapt called before .steer().")
        slot_memory = TextMemory(slots={"instruction": self.memory["instruction"]})
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids, dtype=torch.long)
        return self._formatter.apply_to_ids(input_ids, slot_memory, self.tokenizer, runtime_kwargs)
