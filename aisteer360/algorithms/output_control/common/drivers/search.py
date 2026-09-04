"""Segment-search decoding driver: propose continuations, score them, keep the top k, iterate.

Every rollout receives the composed stacks, so step-level controls steer every lookahead of every
segment-search driver.
"""
from __future__ import annotations

import copy

import torch
from transformers import PreTrainedModel

from aisteer360.algorithms.core.execution.contracts import Capability, Requirements, needs
from aisteer360.algorithms.output_control.base import DecodingDriver, resolve_generate_callable
from aisteer360.algorithms.output_control.common.drivers.frontier import Frontier
from aisteer360.algorithms.output_control.common.drivers.proposer import SegmentProposer
from aisteer360.utils.tokenization import infer_attention_mask_from_ids


class SearchDriver(DecodingDriver):
    """Segment-search decoding driver: propose, sequence-score, keep top-k, iterate.

    Supports batch size 1 only (raises otherwise). `decode()` pops `max_new_tokens` as the global
    budget, builds a `SegmentProposer` with the received stacks, and runs the loop. `runtime_kwargs`
    pass-throughs (`reward_params`) are preserved.

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

    def __init__(
        self,
        scorer,
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
            **runtime_kwargs.get("reward_params", {}),
            "segment_len": segment_len,
            "num_candidates": self.num_candidates,
            "keep_k": self.keep_k,
            "max_iterations": self.max_iterations,
        }
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)

        proposer = SegmentProposer(mode=self.propose_mode)
        frontier = Frontier(
            keep_k=self.keep_k,
            eos_token_id=eos_token_id,
            input_length=input_length,
            max_new_tokens=global_budget,
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
            continuations = self.tokenizer.batch_decode(
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
