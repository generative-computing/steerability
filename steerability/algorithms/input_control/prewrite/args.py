"""Arguments for the PRewrite input control."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from steerability.algorithms.core.base_args import BaseArgs
from steerability.algorithms.input_control.common.memory.text import TextMemory


@dataclass
class PRewriteArgs(BaseArgs):
    """Arguments for PRewrite, an RL-trained instruction rewriter.

    Two strategies are supported:

      - `"inference"` (PRewrite-I): generate one rewrite greedily.
      - `"search"` (PRewrite-S): sample `k_candidates` rewrites, score each on `dev_set`, keep the best.
    """

    initial_instruction: str = field(
        metadata={"help": "Seed instruction to rewrite."},
        default="",
    )

    rewriter_model_name_or_path: str | None = field(
        default=None,
        metadata={
            "help": (
                "HF model id or local path for the rewriter LLM. None means reuse the task model as "
                "rewriter (forbidden when `train_rewriter=True`, see `rewriter_model`)."
            )
        },
    )

    rewriter_model: Any = field(
        default=None,
        metadata={
            "help": (
                "Pre-loaded rewriter model instance. Mutually exclusive with "
                "`rewriter_model_name_or_path`. Useful for reusing a rewriter across multiple PRewrite "
                "runs without round-tripping through disk."
            )
        },
    )

    rewriter_tokenizer: Any = field(
        default=None,
        metadata={
            "help": (
                "Pre-loaded tokenizer for `rewriter_model`. Required if `rewriter_model` is supplied "
                "without an attached tokenizer."
            )
        },
    )

    train_rewriter: bool = field(
        default=False,
        metadata={"help": "If True, GRPO-train the rewriter (scorer-in-the-loop reward) before proposing rewrites."},
    )

    meta_prompt: str | None = field(
        default=None,
        metadata={
            "help": (
                "Custom meta-prompt template for the rewriter. Must contain `{seed}` to reference the "
                "initial instruction. If None, defaults to PRewriteMetaPrompts.DEFAULT."
            )
        },
    )

    strategy: Literal["inference", "search"] = field(
        default="search",
        metadata={"help": "PRewrite-I (greedy) or PRewrite-S (best-of-k)."},
    )

    k_candidates: int = field(
        default=10,
        metadata={"help": "Number of candidate rewrites to sample (search strategy only)."},
    )

    dev_set: list[dict] | None = field(
        default=None,
        metadata={"help": "Dev set used to score candidates under the search strategy."},
    )

    row_scorer: Callable[[str, dict], float] | None = field(
        default=None,
        metadata={"help": "Per-row scorer (response, row) -> float used to aggregate dev-set responses into a scalar."},
    )

    training_seeds: list[str] | None = field(
        default=None,
        metadata={
            "help": (
                "Pool of seed instructions used as GRPO rollout prompts. Defaults to "
                "`[initial_instruction]`. Only consulted when `train_rewriter=True`."
            )
        },
    )

    reward_fn: Callable | None = field(
        default=None,
        metadata={
            "help": (
                "Callable GRPO reward, with signature `reward_func(prompts, completions, **kwargs) -> "
                "list[float]`. Takes precedence over the `row_scorer` + `dev_set` reward when both are set. "
                "If unset, the reward is built from `row_scorer` + `dev_set` (the paper's reward)."
            )
        },
    )

    grpo_config: dict | None = field(
        default=None,
        metadata={"help": "Configuration forwarded to TRL's GRPO trainer (e.g. num_generations, beta)."},
    )

    reward_dev_size: int | None = field(
        default=None,
        metadata={
            "help": (
                "Optional cap on dev rows used per reward evaluation during GRPO training (cost "
                "control). Applied as a deterministic head slice of `dev_set`."
            )
        },
    )

    trust_remote_code: bool = field(
        default=False,
        metadata={"help": "Trust remote code when loading the rewriter model/tokenizer."},
    )

    rewriter_gen_kwargs: dict | None = field(
        default=None,
        metadata={"help": "Generation kwargs for the rewriter LLM (e.g. temperature, max_new_tokens)."},
    )

    eval_gen_kwargs: dict | None = field(
        default=None,
        metadata={"help": "Generation kwargs used when scoring candidates against the dev set."},
    )

    memory: TextMemory | dict | None = field(
        default=None,
        metadata={
            "help": (
                "Precomputed memory (`TextMemory(slots={'instruction': ...})` or a plain "
                "`{'slots': {...}}` dict). When provided, steer() installs it directly and "
                "skips rewriting and selection; the search-only args stay inert."
            )
        },
    )

    def __post_init__(self) -> None:
        if isinstance(self.memory, dict):
            self.memory = TextMemory(slots=dict(self.memory.get("slots", self.memory)))
        if self.memory is not None:
            if "instruction" not in self.memory:
                raise ValueError("`memory` must carry an 'instruction' slot.")
            return  # a precomputed memory makes the search args inert
        if not isinstance(self.initial_instruction, str) or not self.initial_instruction:
            raise ValueError("`initial_instruction` must be a non-empty str.")
        if self.strategy not in ("inference", "search"):
            raise ValueError(f"strategy must be 'inference' or 'search'; got {self.strategy!r}.")
        if self.strategy == "search":
            if self.dev_set is None or not self.dev_set:
                raise ValueError("`dev_set` is required for search strategy.")
            if self.row_scorer is None:
                raise ValueError("`row_scorer` is required for search strategy.")
            if self.k_candidates <= 0:
                raise ValueError("`k_candidates` must be positive for search strategy.")
        if self.rewriter_model is not None and self.rewriter_model_name_or_path is not None:
            raise ValueError(
                "Specify at most one of `rewriter_model` and `rewriter_model_name_or_path`."
            )
        if self.train_rewriter:
            if self.rewriter_model is None and self.rewriter_model_name_or_path is None:
                raise ValueError(
                    "`train_rewriter=True` requires an explicit rewriter via `rewriter_model` or "
                    "`rewriter_model_name_or_path`. Reusing the task LM as rewriter is unsafe under "
                    "training (it would mutate the task model)."
                )
            has_callable_reward = self.reward_fn is not None
            has_scorer_reward = self.row_scorer is not None and bool(self.dev_set)
            if not (has_callable_reward or has_scorer_reward):
                raise ValueError(
                    "`train_rewriter=True` requires a GRPO reward source: provide a callable `reward_fn`, "
                    "or (`row_scorer` and `dev_set`) for the scorer-in-the-loop reward."
                )
