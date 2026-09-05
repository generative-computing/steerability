from __future__ import annotations

from transformers import PreTrainedModel, PreTrainedTokenizerBase

from steerability.algorithms.output_control.base import OutputControl
from steerability.algorithms.output_control.common.drivers.search import SearchDriver
from steerability.algorithms.output_control.deal.args import DeALArgs


class DeAL(SearchDriver):
    """
    Implementation of DeAL (Decoding-time Alignment) from Huang et al., 2024.

    DeAL performs controlled text generation through iterative lookahead search and reward-guided beam selection. Unlike
    training-time alignment methods, DeAL operates purely at inference time to steer language model outputs toward
    desired behaviors.

    The algorithm works in three phases:

    1. **Lookahead Generation**: Generate multiple candidate continuations using beam search from the current context.

    2. **Reward-based Scoring**: Evaluate each candidate continuation using a provided reward function that measures
    alignment with the desired objective (e.g., helpfulness, safety).

    3. **Iterative Refinement**: Select the top-k highest-scoring beams and repeat the process until termination
    conditions are met (EOS token, max length, or max iterations reached).

    DeAL is a decoding driver implemented as a preset of the generic `SearchDriver`, mapping its arguments onto the
    search fields (`scorer`, `segment_len`, `num_candidates`, `keep_k`, `max_iterations`, and `propose_mode="beam"`).
    The composed logits processors and stopping criteria apply inside every lookahead rollout, which means that a
    step-level control such as RAD steers every DeAL rollout. The `reward_params` runtime kwarg is honored per row (one
    mapping merged into the scorer's params).

    Reference:

    - "DeAL: Decoding-time Alignment for Large Language Models"
    James Y. Huang, Sailik Sengupta, Daniele Bonadiman, Yi-an Lai, Arshit Gupta, Nikolaos Pappas, Saab Mansour,
    Katrin Kirchhoff, Dan Roth
    https://arxiv.org/abs/2402.06147
    """

    Args = DeALArgs

    tokenizer: PreTrainedTokenizerBase | None = None

    def __init__(self, *args, **kwargs):
        # route through OutputControl (validate DeALArgs, mirror fields, then _configure)
        OutputControl.__init__(self, *args, **kwargs)

    def _configure(self) -> None:
        """Map DeAL's mirrored args onto the generic `SearchDriver` fields."""
        self.scorer = self.reward_func
        self.segment_len = self.lookahead
        self.num_candidates = self.init_beams
        self.keep_k = self.topk
        # self.max_iterations is already mirrored from DeALArgs
        self.propose_mode = "beam"

    def steer(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase | None = None, **_) -> PreTrainedModel:
        """Lightweight preparation; attach the tokenizer used to decode continuations."""
        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        return model
