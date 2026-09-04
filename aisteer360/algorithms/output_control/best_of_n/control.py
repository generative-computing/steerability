from __future__ import annotations

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from aisteer360.algorithms.output_control.base import OutputControl
from aisteer360.algorithms.output_control.best_of_n.args import BestOfNArgs
from aisteer360.algorithms.output_control.common.drivers.search import SearchDriver


class BestOfN(SearchDriver):
    """
    Best-of-N sampling (rejection sampling / re-ranking), the standard inference-time alignment
    baseline.

    Best-of-N samples `n` full-length continuations from the base model and returns the single
    highest-scoring one under a supplied `SequenceScorer`. It is the simplest segment-shape driver:
    a single search iteration whose one segment spans the whole generation budget, proposed by
    sampling, keeping the argmax.

    BestOfN is a decoding driver: a thin preset of the generic `SearchDriver`, mapping its args
    onto `(scorer, num_candidates=n, keep_k=1, max_iterations=1, propose_mode="sample")`. Each of the
    `n` samples is a full rollout, so the composed logits/stopping stacks steer every sample (a
    step-level control such as RAD applies to every candidate). Pairing the scorer with
    `MajorityVoteScorer` recovers self-consistency (Wang et al., 2022); pairing it with `MetricScorer`
    gives metric-guided reranking.

    Args:
        n (int): Number of full-length continuations to sample and rank. Defaults to 8.
        scorer (Callable): A `SequenceScorer` `(prompt, continuations, params) -> list[float]`; the
            highest-scoring sample is returned.

    Reference:

    - "WebGPT: Browser-assisted question-answering with human feedback"
      Reiichiro Nakano, Jacob Hilton, Suchir Balaji, Jeff Wu, Long Ouyang, Christina Kim, Christopher
      Hesse, Shantanu Jain, Vineet Kosaraju, William Saunders, Xu Jiang, Karl Cobbe, Tyna Eloundou,
      Gretchen Krueger, Kevin Button, Matthew Knight, Benjamin Chess, John Schulman
      [https://arxiv.org/abs/2112.09332](https://arxiv.org/abs/2112.09332)
    """

    Args = BestOfNArgs

    tokenizer: PreTrainedTokenizer | None = None

    def __init__(self, *args, **kwargs):
        # route through OutputControl (validate BestOfNArgs, mirror fields, then _configure)
        OutputControl.__init__(self, *args, **kwargs)

    def _configure(self) -> None:
        """Map Best-of-N's mirrored args onto the generic `SearchDriver` fields."""
        # self.scorer is already mirrored from BestOfNArgs
        self.num_candidates = self.n
        self.keep_k = 1
        self.max_iterations = 1
        self.propose_mode = "sample"
        self.segment_len = None  # resolved from the runtime budget in decode()

    def steer(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer | None = None, **_) -> PreTrainedModel:
        """Lightweight preparation; attach the tokenizer used to decode continuations."""
        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        return model

    def decode(self, input_ids, attention_mask, model, logits_processors, stopping_criteria,
               runtime_kwargs, session=None, **gen_kwargs) -> torch.Tensor:
        """Resolve the full-length segment from the runtime budget, then run one search iteration.

        The budget defaults to 256 new tokens when the caller sets none; `segment_len` stays
        None on the instance (per-operation state lives in the call, so concurrent calls do not
        race).
        """
        gen_kwargs.setdefault("max_new_tokens", 256)
        return super().decode(
            input_ids, attention_mask, model, logits_processors, stopping_criteria,
            runtime_kwargs, session=session, **gen_kwargs,
        )
