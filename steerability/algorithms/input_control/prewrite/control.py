"""PRewrite, an RL-trained instruction rewriter.

Reference:

  - "PRewrite: Prompt Rewriting with Reinforcement Learning"
    Weize Kong, Spurthi Amba Hombaiah, Mingyang Zhang, Qiaozhu Mei, Michael Bendersky
    [https://arxiv.org/abs/2401.08189](https://arxiv.org/abs/2401.08189)
"""
from __future__ import annotations

import logging
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from steerability.algorithms.core.execution.access import ModelAccess
from steerability.algorithms.core.execution.session_utils import SessionLM
from steerability.algorithms.input_control.base import InputControl
from steerability.algorithms.input_control.common.formatters.system_prompt import SystemPromptFormatter
from steerability.algorithms.input_control.common.memory.text import TextMemory
from steerability.algorithms.input_control.common.proposers.llm_meta_prompt import LLMMetaPromptProposer
from steerability.algorithms.input_control.common.proposers.utils.parsing import parse_concise_instruction
from steerability.algorithms.input_control.common.scorers.task_evaluation import TaskEvaluationScorer
from steerability.algorithms.input_control.common.selectors.top_k import TopKSelector
from steerability.algorithms.input_control.prewrite.args import PRewriteArgs
from steerability.algorithms.input_control.prewrite.utils import meta_prompts
from steerability.algorithms.input_control.prewrite.utils.reward import make_scorer_reward_func

logger = logging.getLogger(__name__)


class PRewrite(InputControl):
    """Train (or load) an LLM rewriter that rewrites a seed instruction; optionally select the best
    rewrite by dev-set evaluation, then apply it as the system prompt at inference time.

    When `train_rewriter=True`, the rewriter is trained with GRPO (group-relative policy optimization).
    The reward is downstream task performance, computed by applying each rewrite with the frozen task
    model over a dev set and scoring with a per-row `SampleScorer`, or a user-supplied `reward_fn`.

    Memory shape: `TextMemory(slots={"instruction": str})`.

    Reference:

      - "PRewrite: Prompt Rewriting with Reinforcement Learning"
        Weize Kong, Spurthi Amba Hombaiah, Mingyang Zhang, Qiaozhu Mei, Michael Bendersky
        [https://arxiv.org/abs/2401.08189](https://arxiv.org/abs/2401.08189)
    """

    Args = PRewriteArgs
    supports_batching: bool = True

    # placeholder fields; injected from PRewriteArgs at __init__
    initial_instruction: str = ""
    rewriter_model_name_or_path: str | None = None
    rewriter_model: Any = None
    rewriter_tokenizer: Any = None
    train_rewriter: bool = False
    meta_prompt: str | None = None
    strategy: str = "search"
    k_candidates: int = 10
    dev_set: list[dict] | None = None
    row_scorer: Any = None
    training_seeds: list[str] | None = None
    reward_fn: Any = None
    grpo_config: dict | None = None
    reward_dev_size: int | None = None
    trust_remote_code: bool = False
    rewriter_gen_kwargs: dict | None = None
    eval_gen_kwargs: dict | None = None

    # method-owned state
    memory: TextMemory | None = None
    tokenizer: Any = None
    _formatter: SystemPromptFormatter | None = None

    def steer_access(self) -> ModelAccess:
        """`ModelAccess.ROLLOUTS`; rewriting and dev-set scoring generate through the
        session, and adaptation is a formatter. With a precomputed `memory`, steering is
        model-free (`ModelAccess.FACTS`)."""
        if getattr(getattr(self, "args", None), "memory", None) is not None:
            return ModelAccess.FACTS
        return ModelAccess.ROLLOUTS

    def export_state(self) -> dict:
        """The optimized instruction memory under the `"memory"` key (after `steer()`)."""
        return {"memory": self.memory} if self.memory is not None else {}

    def frozen_form(self, state: dict) -> tuple[str, dict]:
        """A same-class frozen form: the recipe args with `memory=` set to the optimized
        memory (the search-only args stay inert)."""
        from dataclasses import fields

        kwargs = {f.name: getattr(self.args, f.name) for f in fields(self.args) if f.init}
        kwargs["memory"] = state["memory"]
        return "input_control/prewrite", kwargs

    def fit_identity(self):
        """The optimizer-relevant args (everything except `memory`), or None when the recipe
        already carries a memory."""
        from dataclasses import fields

        if self.args.memory is not None:
            return None
        return {
            f.name: getattr(self.args, f.name)
            for f in fields(self.args) if f.init and f.name != "memory"
        }

    def steer(
        self,
        model=None,
        tokenizer=None,
        session=None,
        **kwargs,
    ) -> None:
        self.tokenizer = tokenizer

        if self.args.memory is not None:
            self.memory = self.args.memory
            self._formatter = SystemPromptFormatter()
            return

        task_lm = SessionLM(session) if session is not None else model
        rewriter_lm, rewriter_tok = self._resolve_rewriter(task_lm, tokenizer)
        meta_prompt = self.meta_prompt or meta_prompts.DEFAULT

        if self.train_rewriter:
            reward_fn = self._build_reward_fn(task_lm=task_lm, task_tok=tokenizer)
            rewriter_lm = self._grpo_train_rewriter(rewriter_lm, rewriter_tok, meta_prompt, reward_fn)

        proposer = LLMMetaPromptProposer(
            llm=rewriter_lm,
            tokenizer=rewriter_tok,
            meta_prompt_template=meta_prompt,
            gen_kwargs=self.rewriter_gen_kwargs,
            use_chat_template=False if self.train_rewriter else None,
            parse_fn=parse_concise_instruction,
        )

        if self.strategy == "inference":
            candidates = proposer.propose(seed=self.initial_instruction, n=1)
            best = candidates[0] if candidates else self.initial_instruction
        else:
            n = max(self.k_candidates, 1)
            candidates = proposer.propose(seed=self.initial_instruction, n=n)
            if not candidates:
                logger.warning("PRewrite proposer returned no candidates; falling back to initial instruction.")
                best = self.initial_instruction
            else:
                scorer = TaskEvaluationScorer(
                    task_lm=task_lm,
                    tokenizer=tokenizer,
                    dev_set=self.dev_set,
                    row_scorer=self.row_scorer,
                    gen_kwargs=self.eval_gen_kwargs or {"max_new_tokens": 32, "do_sample": False},
                )
                selector = TopKSelector(scorer=scorer)
                best = selector.select(candidates, k=1)[0]

        stripped = (best or "").strip()
        if not stripped:
            logger.info(
                "PRewrite: chosen rewrite was empty/whitespace; falling back to initial_instruction."
            )
            best = self.initial_instruction
        else:
            best = stripped

        self.memory = TextMemory(slots={"instruction": best})
        self._formatter = SystemPromptFormatter()

    def _resolve_rewriter(self, task_lm, tokenizer) -> tuple[Any, Any]:
        """Pick the rewriter LLM.

        Resolution order:

          1. Pre-loaded `rewriter_model` (+ `rewriter_tokenizer`) if supplied.
          2. Load from `rewriter_model_name_or_path` if supplied.
          3. Default: reuse the task model through the session (forbidden under
             `train_rewriter=True`; rejected at args time).
        """
        if self.rewriter_model is not None:
            rewriter_tok = self.rewriter_tokenizer
            if rewriter_tok is None:
                source = getattr(self.rewriter_model, "name_or_path", None) or getattr(
                    getattr(self.rewriter_model, "config", None), "_name_or_path", None
                )
                if source is None:
                    raise ValueError(
                        "PRewrite: `rewriter_model` was supplied without `rewriter_tokenizer` "
                        "and no name_or_path could be inferred from the model."
                    )
                rewriter_tok = AutoTokenizer.from_pretrained(source, trust_remote_code=self.trust_remote_code)
            return self.rewriter_model, rewriter_tok
        if self.rewriter_model_name_or_path is None:
            return task_lm, tokenizer
        rewriter_lm = AutoModelForCausalLM.from_pretrained(
            self.rewriter_model_name_or_path,
            device_map="auto",
            trust_remote_code=self.trust_remote_code,
        )
        rewriter_tok = AutoTokenizer.from_pretrained(
            self.rewriter_model_name_or_path, trust_remote_code=self.trust_remote_code
        )
        if rewriter_tok.pad_token_id is None and rewriter_tok.eos_token_id is not None:
            rewriter_tok.pad_token = rewriter_tok.eos_token
        return rewriter_lm, rewriter_tok

    def _build_reward_fn(self, task_lm, task_tok):
        """Build the GRPO reward callable for rewriter training.

        Uses a user-supplied `reward_fn` if present. Otherwise builds a `TaskEvaluationScorer` that
        applies each rewrite with the frozen task model over `dev_set` and aggregates `row_scorer`
        to a scalar. The reward's `task_lm` generates through the steering session and stays frozen;
        only the rewriter is trained.
        """
        if self.reward_fn is not None:
            return self.reward_fn
        if self.row_scorer is not None and self.dev_set:
            scorer = TaskEvaluationScorer(
                task_lm=task_lm,
                tokenizer=task_tok,
                dev_set=self.dev_set,
                row_scorer=self.row_scorer,
                gen_kwargs=self.eval_gen_kwargs or {"max_new_tokens": 32, "do_sample": False},
                max_dev_size=self.reward_dev_size,
            )
            return make_scorer_reward_func(scorer, parse_fn=parse_concise_instruction)
        raise ValueError(
            "PRewrite.train_rewriter requires a GRPO reward source: a callable `reward_fn` or "
            "(`row_scorer` and `dev_set`)."
        )

    def _grpo_train_rewriter(self, rewriter_lm, rewriter_tok, meta_prompt: str, reward_fn):
        """GRPO-train the rewriter on a pool of seed instructions, using a callable/scorer reward.

        Returns the trained rewriter.
        """
        from datasets import Dataset

        from steerability.algorithms.structural_control.wrappers.trl.grpotrainer import GRPO, GRPOArgs

        seeds = self.training_seeds or [self.initial_instruction]
        train_dataset = Dataset.from_dict(
            {"prompt": [meta_prompt.format(seed=s) for s in seeds]}
        )

        grpo_args = GRPOArgs(
            train_dataset=train_dataset,
            reward_funcs=[reward_fn],
            **(self.grpo_config or {}),
        )
        grpo = GRPO(grpo_args)
        return grpo.steer(rewriter_lm, rewriter_tok)

    def adapt_messages(
        self,
        messages: list[list[dict]],
        runtime_kwargs: dict | None = None,
    ) -> list[list[dict]] | None:
        if self.memory is None or self._formatter is None:
            raise RuntimeError("PRewrite.adapt_messages called before .steer().")
        return self._formatter.apply_to_messages(messages, self.memory, runtime_kwargs)

    def adapt(
        self,
        input_ids: list[int] | torch.Tensor,
        runtime_kwargs: dict | None = None,
    ) -> list[int] | torch.Tensor:
        if self.memory is None or self._formatter is None or self.tokenizer is None:
            raise RuntimeError("PRewrite.adapt called before .steer().")
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids, dtype=torch.long)
        return self._formatter.apply_to_ids(input_ids, self.memory, self.tokenizer, runtime_kwargs)
