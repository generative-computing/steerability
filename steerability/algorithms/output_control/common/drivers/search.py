"""Segment-search decoding driver: propose continuations, score them, keep the top k, iterate.

Every rollout receives the composed stacks, so step-level controls steer every lookahead of every
segment-search driver.
"""
from __future__ import annotations

import copy
from typing import Any, Mapping, Sequence

import torch
from transformers import PreTrainedModel

from steerability.algorithms.core.execution.contracts import Capability, Requirements, needs
from steerability.algorithms.output_control.base import DecodingDriver, resolve_generate_callable
from steerability.algorithms.output_control.common.drivers.frontier import Frontier
from steerability.algorithms.output_control.common.drivers.proposer import SegmentProposer
from steerability.algorithms.output_control.common.scorers import SequenceScorer
from steerability.utils.tokenization import infer_attention_mask_from_ids


def _resolve_reward_params(runtime_kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """The `reward_params` mapping for this call, from either delivery form.

    A mapping is one row's value (a direct call); a sequence is the row-aligned form (a batched
    caller or the evaluation collator) and must hold exactly one element, since the driver handles
    one prompt per call. A missing key or a None value gives an empty mapping.

    Args:
        runtime_kwargs: The call's runtime kwargs.

    Returns:
        A new mapping of the row's reward params, empty when the key is absent or None.

    Raises:
        ValueError: If a sequence does not hold exactly one element.
        TypeError: If the value, or the sequence's element, is not a mapping.
    """
    value = runtime_kwargs.get("reward_params")
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != 1:
            raise ValueError(
                f"reward_params is row-scoped and the driver handles one prompt per call; "
                f"got a sequence of length {len(value)}."
            )
        row = value[0]
        if row is None:
            return {}
        if not isinstance(row, Mapping):
            raise TypeError(f"reward_params rows must be mappings; got {type(row).__name__}.")
        return dict(row)
    raise TypeError(
        f"reward_params must be a mapping or a one-element sequence of mappings; got {type(value).__name__}."
    )


class SearchDriver(DecodingDriver):
    """Segment-search decoding driver: propose, sequence-score, keep top-k, iterate.

    Supports batch size 1 only (raises otherwise). `decode()` pops `max_new_tokens` as the global
    budget, builds a `SegmentProposer` with the received stacks, and runs the loop. A kept beam is
    finished when its continuation, with trailing pad tokens stripped, ends in a token of the eos
    set (the tokenizer's eos plus any ids on the model's generation config) or when its stripped
    length reaches the global budget. Finished beams leave the frontier, and a beam cut by a
    caller-supplied stopping criterion classifies as unfinished. The `reward_params` runtime kwarg
    is row-scoped, so its per-row form is one mapping, merged into the scorer's params on every
    scoring call, and a batched delivery is a one-element sequence holding that mapping (the driver
    handles one prompt per call).

    Can be constructed directly (its positional constructor below) or as a preset: a subclass with an
    `Args` dataclass maps its mirrored args onto these fields in `_configure()` (see DeAL), so it never
    bypasses this `__init__`.

    Args:
        scorer: A `SequenceScorer` `(prompt, continuations, params) -> list[float]`.
        segment_len: Max new tokens per rollout.
        num_candidates: Number of continuations proposed per iteration.
        keep_k: Number of beams retained each iteration.
        max_iterations: Maximum search iterations.
        propose_mode: `"beam"` or `"sample"`.
    """

    RUNTIME_KWARGS_SCHEMA = [
        {
            "name": "reward_params",
            "type": "dict",
            "scope": "row",
            "help": (
                "Entries merged into the scorer's params mapping on every scoring call of this "
                "generation. The per-row form is one mapping; the driver handles one prompt per "
                "call, so a batched delivery is a one-element sequence holding that mapping. A "
                "per-sample mapping (for example a reference answer under 'reference') reaches "
                "the scorer's row through SampleSequenceScorer."
            ),
        },
    ]

    def __init__(
        self,
        scorer: SequenceScorer,
        segment_len: int,
        num_candidates: int,
        keep_k: int,
        max_iterations: int,
        propose_mode: str = "beam",
    ):
        self.scorer = scorer
        self.segment_len = segment_len
        self.num_candidates = num_candidates
        self.keep_k = keep_k
        self.max_iterations = max_iterations
        self.propose_mode = propose_mode
        self.tokenizer = None  # injected by the pipeline

    def max_rollouts_per_query(self) -> int:
        """`num_candidates * (1 + (max_iterations - 1) * keep_k)`.

        The first iteration proposes `num_candidates` continuations from the single-row prompt;
        each later iteration proposes `num_candidates` from each of up to `keep_k` retained
        beams.
        """
        return self.num_candidates * (1 + (self.max_iterations - 1) * self.keep_k)

    def requirements(self) -> Requirements:
        """Rollouts run through the session, so sampled proposals require nothing beyond the
        session contract; beam proposals require `Capability.BEAM_PROPOSALS`."""
        if getattr(self, "propose_mode", "sample") == "beam":
            return Requirements(generate=needs(
                Capability.BEAM_PROPOSALS,
                hint="use propose_mode='sample' or run this pipeline on the huggingface backend",
            ))
        return Requirements()

    def decode(self, input_ids, attention_mask, model: PreTrainedModel | None, logits_processors,
               stopping_criteria, runtime_kwargs, session=None, **gen_kwargs) -> torch.Tensor:
        if input_ids.dim() != 2 or input_ids.size(0) != 1:
            raise NotImplementedError("SearchDriver handles one prompt at a time (batch size 1).")
        if self.tokenizer is None:
            raise RuntimeError("SearchDriver requires a tokenizer; steer() must run first.")

        runtime_kwargs = runtime_kwargs or {}
        base_generate = resolve_generate_callable(model, runtime_kwargs, session=session)

        prompt_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        input_length = input_ids.size(1)
        global_budget = gen_kwargs.pop("max_new_tokens", None)

        # segment_len=None means "one full-budget segment" (best-of-N as a config); each rollout then
        # spans the whole generation budget in a single iteration
        segment_len = self.segment_len if self.segment_len is not None else global_budget
        if segment_len is None:
            raise ValueError(
                "SearchDriver requires a segment length: set `segment_len` or pass `max_new_tokens` "
                "as the generation budget (segment_len=None uses the budget as the segment length)."
            )

        reward_params = {
            **_resolve_reward_params(runtime_kwargs),
            "segment_len": segment_len,
            "num_candidates": self.num_candidates,
            "keep_k": self.keep_k,
            "max_iterations": self.max_iterations,
        }
        tokenizer_eos = getattr(self.tokenizer, "eos_token_id", None)
        eos_ids = {tokenizer_eos} if tokenizer_eos is not None else set()
        if model is not None:
            configured = getattr(model.generation_config, "eos_token_id", None)
            eos_ids.update([configured] if isinstance(configured, int) else (configured or []))

        proposer = SegmentProposer(mode=self.propose_mode)
        frontier = Frontier(
            keep_k=self.keep_k,
            eos_token_id=sorted(eos_ids) or None,
            input_length=input_length,
            max_new_tokens=global_budget,
            pad_token_id=getattr(self.tokenizer, "pad_token_id", None),
        )

        current_ids = input_ids
        kept = None
        for _ in range(self.max_iterations):
            # safe to deepcopy: the composed stacks travel as explicit decode() parameters, never inside gen_kwargs
            rollout_kwargs = copy.deepcopy(gen_kwargs)
            frontier_mask = infer_attention_mask_from_ids(current_ids, self.tokenizer.pad_token_id)
            beams = proposer.propose(
                current_ids,
                n=self.num_candidates,
                segment_len=segment_len,
                processors=logits_processors,
                criteria=stopping_criteria,
                model=model,
                base_generate=base_generate,
                attention_mask=frontier_mask,
                **rollout_kwargs,
            )
            continuations = self.tokenizer.decode(
                beams[:, input_length:], skip_special_tokens=True
            )
            scores = self.scorer(prompt_text, continuations, reward_params)
            if len(scores) != beams.size(0):
                raise RuntimeError(f"Scorer returned {len(scores)} scores for {beams.size(0)} beams.")

            step = frontier.keep(beams, scores)
            kept = step
            if all(step.finished_flags):
                break

            unfinished = [i for i, f in enumerate(step.finished_flags) if not f]
            if not unfinished:
                break
            current_ids = step.kept_ids[unfinished]

        final = frontier.best_ids if frontier.best_ids is not None else kept.kept_ids[0]
        return final.unsqueeze(0) if final.dim() == 1 else final
