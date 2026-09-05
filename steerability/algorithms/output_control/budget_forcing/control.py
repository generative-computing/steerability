from __future__ import annotations

from transformers import PreTrainedModel, PreTrainedTokenizerBase

from steerability.algorithms.output_control.base import OutputControl
from steerability.algorithms.output_control.budget_forcing.args import BudgetForcingArgs
from steerability.algorithms.output_control.common.drivers.phased import Fixed, Generated, PhasedDriver


class BudgetForcing(PhasedDriver):
    """
    Implementation of Budget Forcing (s1) from Muennighoff et al., 2025.

    Budget Forcing controls the length of a reasoning model's thinking at test time. It caps the
    thinking phase at a token budget, and can either shorten reasoning (force the closing think tag
    once the budget is hit) or lengthen it (suppress the early stop and append an `extension_text`
    such as "Wait" to prompt continued reasoning) before generating the final answer.

    BudgetForcing is a decoding driver: a thin preset of the generic `PhasedDriver`. Its plan is:

    1. A thinking phase generating until the closing think tag or `max_thinking_tokens`, whichever
       first.

    2. Up to `num_extensions` extension rounds, each appending `extension_text` and generating another
       bounded thinking segment.

    3. A forced closing think tag (`Fixed`) followed by the answer phase (unbounded `Generated`).

    Every `Generated` phase delegates to `model.generate` with the received stacks, so a step-level
    control steers each phase. The `Fixed` phases are plain appends. Plans are
    constructed per example (the driver loops over rows).

    Reference:

    - "s1: Simple test-time scaling"
      Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi,
      Luke Zettlemoyer, Percy Liang, Emmanuel Candès, Tatsunori Hashimoto
      [https://arxiv.org/abs/2501.19393](https://arxiv.org/abs/2501.19393)
    """

    Args = BudgetForcingArgs

    tokenizer: PreTrainedTokenizerBase | None = None

    def __init__(self, *args, **kwargs):
        # route through OutputControl (validate BudgetForcingArgs, mirror fields, then _configure)
        OutputControl.__init__(self, *args, **kwargs)

    def _configure(self) -> None:
        """Budget forcing keeps the full thinking + answer stream (no extract rule)."""
        self.extract_after = None

    def steer(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase | None = None, **_) -> PreTrainedModel:
        """Lightweight preparation; attach the tokenizer used to splice phase boundaries."""
        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        return model

    def max_rollouts_per_query(self) -> int:
        """`num_extensions + 2`: the initial thinking phase, one per extension round, and the
        answer phase."""
        return self.num_extensions + 2

    def plan(self, prompt_text: str, params: dict) -> list:
        """Build the thinking-budget plan: bounded thinking, optional extensions, forced tag, answer.

        Each thinking phase ends at the `end_think` string, any token in `end_think_token_ids`, or
        `max_thinking_tokens`; the forced closing tag before the answer is the `end_think` text.
        """
        thinking = lambda: Generated(
            until=self.end_think, until_token_ids=self.end_think_token_ids, budget=self.max_thinking_tokens,
        )
        plan = [thinking()]
        for _ in range(self.num_extensions):
            plan.append(Fixed(self.extension_text))
            plan.append(thinking())
        plan.append(Fixed(self.end_think))
        plan.append(Generated())
        return plan
