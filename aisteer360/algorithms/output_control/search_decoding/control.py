from __future__ import annotations

from transformers import PreTrainedModel, PreTrainedTokenizer

from aisteer360.algorithms.output_control.base import OutputControl
from aisteer360.algorithms.output_control.common.drivers.search import SearchDriver
from aisteer360.algorithms.output_control.common.resolve import resolve_scorer
from aisteer360.algorithms.output_control.search_decoding.args import SearchDecodingArgs


class SearchDecoding(SearchDriver):
    """Config-first segment-shape driver: propose -> score -> keep -> iterate.

    `SearchDecoding` is the generic over the segment shape, a thin `Args`-configured preset of the
    `common` `SearchDriver`. Its defaults are best-of-N: with no arguments beyond a scorer, it
    samples `num_candidates` full-budget continuations once and returns the scorer's argmax. A
    method from the literature is an assignment of a config:

        - Best-of-N: defaults + `scorer={"kind": "reward_model", ...}` (or any callable).
        - Self-consistency: defaults + `scorer={"kind": "majority_vote"}`.
        - Blockwise controlled decoding: `segment_len=block, max_iterations=⌈budget/block⌉`.
        - Metric-guided reranking: defaults + `scorer=MetricScorer(metric, score_key)`.
        - DeAL-equivalent: `propose_mode="beam", segment_len=lookahead, num_candidates=init_beams,
          keep_k=topk, max_iterations=...`.

    `SearchDecoding` is a decoding driver: at most one enabled driver runs per pipeline, and the
    driver forwards the composed logits/stopping stacks into every rollout, so a step-level
    control (e.g. `ValueGuidance`) steers every proposed continuation. Batch size 1 and the runtime
    pass-throughs (`reward_params`) are inherited from `SearchDriver` unchanged.

    Args:
        scorer: A `SequenceScorer` (callable / instance) or a dict spec with a `"kind"` key.
        segment_len (int | None): Max new tokens per rollout; `None` uses the call's `max_new_tokens`.
            Defaults to None.
        num_candidates (int): Number of continuations proposed per iteration. Defaults to 8.
        keep_k (int): Beams retained each iteration. Defaults to 1.
        max_iterations (int): Maximum search iterations. Defaults to 1.
        propose_mode (str): `"sample"` or `"beam"`. Defaults to `"sample"`.

    Reference:

    - "DeAL: Decoding-time Alignment for Large Language Models"
      James Y. Huang, Sailik Sengupta, Daniele Bonadiman, Yi-an Lai, Arshit Gupta, Nikolaos Pappas,
      Saab Mansour, Katrin Kirchhoff, Dan Roth
      [https://arxiv.org/abs/2402.06147](https://arxiv.org/abs/2402.06147)

    - "Self-Consistency Improves Chain of Thought Reasoning in Language Models"
      Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery,
      Denny Zhou
      [https://arxiv.org/abs/2203.11171](https://arxiv.org/abs/2203.11171)
    """

    Args = SearchDecodingArgs

    tokenizer: PreTrainedTokenizer | None = None

    def __init__(self, *args, **kwargs):
        # route through OutputControl (validate SearchDecodingArgs, mirror fields, then _configure)
        OutputControl.__init__(self, *args, **kwargs)

    def _configure(self) -> None:
        """Map the mirrored args onto the generic `SearchDriver` fields (name-identical here)."""
        # self.scorer / segment_len / num_candidates / keep_k / max_iterations / propose_mode are
        # already mirrored from SearchDecodingArgs; the driver reads them under the same names
        self.tokenizer = None

    def steer(self, model: PreTrainedModel | None = None, tokenizer: PreTrainedTokenizer | None = None,
              **_) -> PreTrainedModel | None:
        """Attach the tokenizer and resolve the scorer spec (a device is needed for reward models)."""
        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        device = next(model.parameters()).device if model is not None else None
        self.scorer = resolve_scorer(self.scorer, device=device)
        return model
