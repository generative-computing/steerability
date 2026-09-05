"""LM-based labeling of (anchor, candidate) pairs for EPR.

For each training row `(x_i, y_i)`, compute, for each candidate row `(x_j, y_j)` from the BM25 set, the
log-probability `log p_LM(y_i | (x_j, y_j), x_i)`. The top-`k_pos` highest-scoring candidates become
positives, the bottom-`k_neg` become hard negatives.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import torch

logger = logging.getLogger(__name__)


@dataclass
class LabeledPair:
    """One labeled training pair: an anchor row + a list of (candidate_index, polarity) labels."""
    anchor_index: int
    positives: list[int]
    negatives: list[int]


def _format_demo(item: dict, input_field: str, output_field: str) -> str:
    return f"Input: {item.get(input_field, '')}\nOutput: {item.get(output_field, '')}"


def _format_query(item: dict, input_field: str) -> str:
    return f"Input: {item.get(input_field, '')}\nOutput:"


@torch.no_grad()
def _score_candidate_for_anchor(
    scoring_lm,
    scoring_tokenizer,
    anchor: dict,
    candidate: dict,
    input_field: str,
    output_field: str,
) -> float:
    """`log p_LM(y_anchor | demo(candidate), x_anchor)` averaged across target tokens."""
    demo = _format_demo(candidate, input_field, output_field)
    query = _format_query(anchor, input_field)
    target = str(anchor.get(output_field, ""))
    if not target:
        return float("-inf")

    prefix_text = demo + "\n\n" + query + " "
    prefix_ids = scoring_tokenizer(prefix_text, return_tensors="pt").input_ids.to(scoring_lm.device)
    target_ids = scoring_tokenizer(target, return_tensors="pt", add_special_tokens=False).input_ids.to(scoring_lm.device)
    if target_ids.size(1) == 0:
        return float("-inf")

    full_ids = torch.cat([prefix_ids, target_ids], dim=1)
    outputs = scoring_lm(full_ids)
    logits = outputs.logits  # [1, T, V]

    # logits[t] predicts token at position t+1 -> the slice predicting the target tokens is
    # [prefix_len - 1 : prefix_len - 1 + target_len].
    p_len = prefix_ids.size(1)
    t_len = target_ids.size(1)
    slice_logits = logits[0, p_len - 1 : p_len - 1 + t_len, :]
    log_probs = torch.log_softmax(slice_logits, dim=-1)
    target_log_probs = log_probs.gather(dim=-1, index=target_ids[0].unsqueeze(-1)).squeeze(-1)
    return float(target_log_probs.mean().item())


def label_pairs(
    items: Sequence[dict],
    candidate_sets: dict[int, list[int]],
    scoring_lm,
    scoring_tokenizer,
    k_pos: int = 5,
    k_neg: int = 5,
    input_field: str = "input",
    output_field: str = "output",
    max_anchors: int | None = None,
) -> list[LabeledPair]:
    """Label each anchor with its `k_pos` best and `k_neg` worst candidates by LM scoring.

    `max_anchors` caps the number of anchors processed (useful for fast CI); when None, scores every
    anchor in `candidate_sets`.
    """
    items = list(items)
    anchor_indices = list(candidate_sets.keys())
    if max_anchors is not None:
        anchor_indices = anchor_indices[:max_anchors]

    out: list[LabeledPair] = []
    for anchor_idx in anchor_indices:
        candidates = candidate_sets.get(anchor_idx, [])
        if not candidates:
            continue

        scored: list[tuple[int, float]] = []
        anchor = items[anchor_idx]
        for cand_idx in candidates:
            score = _score_candidate_for_anchor(
                scoring_lm,
                scoring_tokenizer,
                anchor,
                items[cand_idx],
                input_field,
                output_field,
            )
            scored.append((cand_idx, score))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        positives = [idx for idx, _ in scored[:k_pos]]
        negatives = [idx for idx, _ in scored[-k_neg:]]
        out.append(LabeledPair(anchor_index=anchor_idx, positives=positives, negatives=negatives))
    return out
