from __future__ import annotations

import warnings

from transformers import PreTrainedModel, PreTrainedTokenizerBase

from steerability.algorithms.output_control.base import OutputControl
from steerability.algorithms.output_control.common.drivers.phased import Fixed, Generated, PhasedDriver
from steerability.algorithms.output_control.phased_decoding.args import PhasedDecodingArgs

_FIXED_KEYS = {"fixed", "replace", "add_special_tokens"}
_GENERATE_SUBKEYS = {"until", "until_token_ids", "budget"}


def _parse_phase(entry: dict):
    """Parse one plan entry into a `Fixed` / `Generated`, validating the declarative grammar."""
    if not isinstance(entry, dict):
        raise ValueError(f"Each plan entry must be a dict, got {type(entry).__name__}.")
    has_fixed = "fixed" in entry
    has_generate = "generate" in entry
    if has_fixed == has_generate:
        raise ValueError(
            f"Each plan entry must have exactly one of 'fixed' or 'generate', got keys {sorted(entry)}."
        )

    if has_fixed:
        unknown = set(entry) - _FIXED_KEYS
        if unknown:
            raise ValueError(f"Unknown key(s) {sorted(unknown)} in a 'fixed' phase.")
        text = entry["fixed"]
        if not (isinstance(text, str) or callable(text)):
            raise ValueError("'fixed' must be a str or a (prompt_text, params) -> str callable.")
        return Fixed(
            text,
            replace=bool(entry.get("replace", False)),
            add_special_tokens=bool(entry.get("add_special_tokens", False)),
        )

    gen = entry["generate"]
    if not isinstance(gen, dict):
        raise ValueError("'generate' must map to a dict with optional 'until' / 'budget' keys.")
    unknown = set(gen) - _GENERATE_SUBKEYS
    if unknown:
        raise ValueError(f"Unknown subkey(s) {sorted(unknown)} in a 'generate' phase.")
    budget = gen.get("budget")
    if budget is not None and (not isinstance(budget, int) or budget <= 0):
        raise ValueError(f"'generate' budget must be a positive integer when set, got {budget!r}.")
    until = gen.get("until")
    if until is not None and not isinstance(until, str):
        raise ValueError(f"'generate' until must be a string when set, got {type(until).__name__}.")
    until_token_ids = gen.get("until_token_ids") or ()
    if isinstance(until_token_ids, (str, bytes)) or not isinstance(until_token_ids, (list, tuple)):
        raise ValueError(
            f"'generate' until_token_ids must be a sequence of ints when set, got "
            f"{type(until_token_ids).__name__}."
        )
    if any(not isinstance(token_id, int) or isinstance(token_id, bool) for token_id in until_token_ids):
        raise ValueError("'generate' until_token_ids must contain only ints.")
    return Generated(until=until, until_token_ids=tuple(until_token_ids), budget=budget)


class PhasedDecoding(PhasedDriver):
    """Config-first phase-shape driver: forced / generated segments spliced into one stream.

    `PhasedDecoding` is the generic over the phase shape, a thin `Args`-configured preset of the
    `common` `PhasedDriver`. A declarative `plan` grammar (str-or-callable forced text and bounded
    generated segments) makes a method from the literature an assignment of a config:

        - Budget forcing (s1): a bounded thinking phase, a forced `"Wait"`, an extended thinking
          phase, a forced closing tag, then an unbounded answer phase.
        - Response prefill: `plan=[{"fixed": "Sure, here is the answer:\\n"}, {"generate": {}}]`.
        - Scaffolded output: alternating `{"fixed": <header>}` / `{"generate": {"until": "\\n\\n"}}`.
        - ThinkingIntervention-equivalent: a single replacing `Fixed` intervention + `Generated`,
          with `extract_after="</think>"`.

    `PhasedDecoding` is a decoding driver: at most one enabled driver runs per pipeline, and every
    `Generated` phase delegates to `model.generate` with the composed stacks, so a step-level
    control steers each generated phase. Plans are constructed per example (the driver
    loops over rows), so `supports_batching` is True.

    Reference:

    - "s1: Simple test-time scaling"
      Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi,
      Luke Zettlemoyer, Percy Liang, Emmanuel Candès, Tatsunori Hashimoto
      [https://arxiv.org/abs/2501.19393](https://arxiv.org/abs/2501.19393)

    - "Effectively Controlling Reasoning Models through Thinking Intervention"
      Tong Wu, Chong Xiang, Jiachen T. Wang, G. Edward Suh, Prateek Mittal
      [https://arxiv.org/abs/2503.24370](https://arxiv.org/abs/2503.24370)
    """

    Args = PhasedDecodingArgs

    supports_batching: bool = True

    tokenizer: PreTrainedTokenizerBase | None = None

    def __init__(self, *args, **kwargs):
        # route through OutputControl (validate PhasedDecodingArgs, mirror fields, then _configure)
        OutputControl.__init__(self, *args, **kwargs)

    def _configure(self) -> None:
        """Parse and validate the declarative plan into `Fixed` / `Generated` phases.

        The `plan` arg is mirrored onto the instance by `OutputControl.__init__`, which would shadow
        this class's `plan()` method (the driver calls `self.plan(...)`). Capture the raw list into
        `_parsed_plan`, then drop the instance attribute so the method is visible again.
        """
        raw_plan = self.plan
        self._parsed_plan = [_parse_phase(entry) for entry in raw_plan]
        del self.__dict__["plan"]  # unshadow the plan() method mirrored over by __init__
        if not any(isinstance(phase, Generated) for phase in self._parsed_plan):
            warnings.warn(
                "PhasedDecoding plan contains no 'generate' phase; the model will not produce any "
                "tokens beyond the spliced fixed text.",
                UserWarning,
            )
        self.tokenizer = None
        # self.extract_after is already mirrored from PhasedDecodingArgs

    def steer(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase | None = None, **_) -> PreTrainedModel:
        """Lightweight preparation; attach the tokenizer used to splice phase boundaries."""
        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        return model

    def plan(self, prompt_text: str, params: dict) -> list:
        """Return the parsed phase plan (per-example callables in `Fixed` are invoked by `_run_plan`)."""
        return self._parsed_plan

    def max_rollouts_per_query(self) -> int:
        """The number of `Generated` phases in the parsed plan (each phase is one rollout)."""
        return sum(isinstance(phase, Generated) for phase in self._parsed_plan)
