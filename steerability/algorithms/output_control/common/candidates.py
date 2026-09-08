"""Candidate token policies for step-shaped output controls.

A candidate policy turns a full-vocabulary score row into the small set of tokens a per-candidate
value function will evaluate. Composition position matters and is a feature: a value-guided
processor with `policy="surviving"` placed after other processors steers exactly the distribution
they left behind (SASA's behavior with a composed processor stack).
"""
from __future__ import annotations

from typing import Literal

import torch

CandidatePolicy = Literal["top_k", "top_p", "surviving"]


def select_candidates(
    scores: torch.Tensor,
    policy: CandidatePolicy,
    *,
    k: int | None = None,
    p: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return `(candidate_ids [B, K], candidate_scores [B, K])` under `policy`.

    Policies:

        - `top_k`: the `k` highest-scoring tokens per row (batch-safe).
        - `top_p`: the smallest nucleus with cumulative probability >= `p`. Batch size 1 only
          (K varies per row); raises on `B > 1`.
        - `surviving`: every token with a finite score, i.e. whatever earlier processors in the
          composed stack left finite. Batch size 1 only, for the same reason.

    Args:
        scores: Next-token logits `[B, V]`.
        policy: One of `top_k`, `top_p`, `surviving`.
        k: Number of candidates for `top_k`.
        p: Nucleus threshold for `top_p`.

    Returns:
        A tuple `(candidate_ids, candidate_scores)`, each shaped `[B, K]`.

    Raises:
        ValueError: On an unknown policy, a missing `k`/`p`, or `B > 1` for `top_p` / `surviving`.
    """
    batch_size = scores.size(0)

    if policy == "top_k":
        if k is None or k <= 0:
            raise ValueError("top_k policy requires a positive `k`.")
        k = min(k, scores.size(-1))
        candidate_scores, candidate_ids = torch.topk(scores, k, dim=-1)
        return candidate_ids, candidate_scores

    if policy == "top_p":
        if p is None:
            raise ValueError("top_p policy requires `p`.")
        if batch_size != 1:
            raise ValueError("top_p candidate policy supports batch size 1 only (K varies per row).")
        sorted_logits, sorted_indices = torch.sort(scores, descending=False)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        keep = cumulative_probs > (1 - p)
        keep = keep.scatter(1, sorted_indices, keep)
        cand_ids = torch.nonzero(keep)[:, 1].unsqueeze(0)
        return cand_ids, scores.gather(1, cand_ids)

    if policy == "surviving":
        if batch_size != 1:
            raise ValueError("surviving candidate policy supports batch size 1 only (K varies per row).")
        cand_ids = torch.nonzero(scores > -torch.inf)[:, 1].unsqueeze(0)
        return cand_ids, scores.gather(1, cand_ids)

    raise ValueError(f"Unknown candidate policy: {policy!r}.")
