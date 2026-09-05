"""Top-k, finished, and best-so-far bookkeeping for the segment search in `SearchDriver`."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from steerability.algorithms.core.output import infer_finish_reasons


@dataclass
class FrontierStep:
    """Result of keeping the top-k beams for one search iteration.

    Attributes:
        kept_ids: The top-k beam sequences `[k, T]`.
        kept_scores: Their scores.
        finished_flags: Per-kept-beam finished flag (EOS hit or budget reached).
    """

    kept_ids: torch.Tensor
    kept_scores: list[float]
    finished_flags: list[bool]


class Frontier:
    """Keep top-k beams by score, track per-beam finished flags and best-so-far.

    Because beams arrive right-padded to a common length, a kept beam's continuation is first
    stripped of trailing `pad_token_id` positions to recover its true length. The beam is
    finished when the stripped continuation ends in a token of the eos set, or when its
    stripped length reaches `max_new_tokens`. A beam cut by a caller-supplied stopping
    criterion classifies as unfinished.

    Args:
        keep_k: Number of beams to retain each iteration.
        eos_token_id: Finished-beam token id(s), as an int, a list of ints, or None (disables
            the EOS check).
        input_length: Prompt length, used to slice each beam's continuation.
        max_new_tokens: Global token budget (None disables the budget check).
        pad_token_id: Padding token id stripped from continuation tails, or None (no stripping).
    """

    def __init__(self, keep_k: int, eos_token_id: int | list[int] | None, input_length: int,
                 max_new_tokens: int | None, pad_token_id: int | None = None):
        self.keep_k = keep_k
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.input_length = input_length
        self.max_new_tokens = max_new_tokens
        self.best_ids: torch.Tensor | None = None
        self.best_score = float("-inf")

    def keep(self, beams: torch.Tensor, scores: list[float]) -> FrontierStep:
        """Select the top-k beams, update best-so-far, and compute finished flags.

        Args:
            beams: Candidate beam sequences `[N, T]`, right-padded to a common length.
            scores: One score per beam.

        Returns:
            A `FrontierStep` with the kept beams, their scores, and their finished flags.
        """
        score_tensor = torch.tensor(scores, device=beams.device)
        k = min(self.keep_k, score_tensor.numel())
        top_idx = torch.topk(score_tensor, k).indices
        kept = beams[top_idx]
        kept_scores = score_tensor[top_idx].tolist()

        reasons = infer_finish_reasons(
            kept[:, self.input_length:],
            {"max_new_tokens": self.max_new_tokens},
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
        )
        finished_flags = [reason is not None for reason in reasons]

        best_local = int(torch.argmax(torch.tensor(kept_scores)))
        if kept_scores[best_local] > self.best_score:
            self.best_score = kept_scores[best_local]
            self.best_ids = kept[best_local]

        return FrontierStep(kept_ids=kept, kept_scores=kept_scores, finished_flags=finished_flags)
