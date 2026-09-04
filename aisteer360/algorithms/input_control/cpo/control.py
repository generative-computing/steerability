"""CPO, causal prompt optimization.

Reference:

  - "Causal Prompt Optimization"
    Chen et al., 2026
    [https://arxiv.org/abs/2602.01711](https://arxiv.org/abs/2602.01711)
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from aisteer360.algorithms.core.execution.access import ModelAccess
from aisteer360.algorithms.core.execution.contracts import Capability, Requirements, needs
from aisteer360.algorithms.core.execution.session_utils import SessionLM
from aisteer360.algorithms.input_control.base import InputControl
from aisteer360.algorithms.input_control.common.formatters.system_prompt import SystemPromptFormatter
from aisteer360.algorithms.input_control.common.memory.text import TextMemory
from aisteer360.algorithms.input_control.common.proposers.llm_meta_prompt import LLMMetaPromptProposer
from aisteer360.algorithms.input_control.common.proposers.utils.parsing import parse_concise_instruction
from aisteer360.algorithms.input_control.common.scorers.task_evaluation import TaskEvaluationScorer
from aisteer360.algorithms.input_control.cpo.args import CPOArgs
from aisteer360.algorithms.input_control.cpo.utils import causal_reward, refinement_meta_prompt
from aisteer360.algorithms.input_control.cpo.utils.causal_reward import CausalRewardScorer
from aisteer360.algorithms.input_control.cpo.utils.embeddings import TextEncoder

logger = logging.getLogger(__name__)


@dataclass
class CPOMemory:
    """Custom Memory for CPO; holds the trained scorer and a query→best-prompt cache."""
    causal_scorer: CausalRewardScorer
    query_cache: dict[str, str] = field(default_factory=dict)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.causal_scorer.save(path / "scorer.pkl")
        (path / "cache.json").write_text(json.dumps(self.query_cache))

    @classmethod
    def load(cls, path: Path, encoder: TextEncoder) -> "CPOMemory":
        path = Path(path)
        scorer = CausalRewardScorer.load(path / "scorer.pkl", encoder=encoder)
        cache = json.loads((path / "cache.json").read_text())
        return cls(causal_scorer=scorer, query_cache=cache)


class CPO(InputControl):
    """Causal prompt optimization.

    Stage 1 (`steer`): build a `⟨query, prompt, score⟩` dataset (either supplied or generated from
    `train_dataset`), then train a `CausalRewardScorer` over PCA-reduced embeddings.

    Stage 2 (`adapt_messages`): per-query, run a B-wide K-retained R-round tree search using the prompt
    proposer to expand parents and the causal scorer to retain survivors. Returns the best survivor as
    the system prompt.

    The query cache keys on a sha256 hash of the query text. By default the raw query is hashed, so
    whitespace, casing, and punctuation differences yield distinct cache entries. Supply
    `cache_key_normalizer` (a `str -> str` callable) to normalize the query before hashing so
    near-duplicate queries share a cached prompt.

    Reference:

      - "Causal Prompt Optimization"
        Chen et al., 2026
        [https://arxiv.org/abs/2602.01711](https://arxiv.org/abs/2602.01711)
    """

    Args = CPOArgs
    supports_batching: bool = True

    # placeholder fields; injected from CPOArgs at __init__
    seed_prompt: str = ""
    train_dataset: list[dict] = []
    offline_data: list[dict] | None = None
    n_prompts_per_query: int = 200
    embedding_model: str = "nomic-ai/nomic-embed-text-v1.5"
    trust_remote_code: bool = True
    pca_query_dim: int = 40
    pca_prompt_dim: int = 15
    prompt_lm: Any = None
    rounds: int = 3
    candidates_per_parent: int = 5
    retained_per_round: int = 3
    metric: Any = None
    score_key: str | None = None
    cache_queries: bool = True
    cache_key_normalizer: Any = None
    use_dml: bool | None = None
    refinement_meta_prompt: str | None = None
    proposer_gen_kwargs: dict | None = None
    eval_gen_kwargs: dict | None = None

    # method-owned state
    memory: CPOMemory | None = None
    tokenizer: Any = None
    _formatter: SystemPromptFormatter | None = None
    _proposer: LLMMetaPromptProposer | None = None
    _encoder: TextEncoder | None = None

    def requirements(self) -> Requirements:
        """Backend requirements computed from this instance's configuration, per phase.

        With `prompt_lm` supplied the proposer is a control-owned auxiliary and every phase is
        prompt-only. With `prompt_lm` unset the live pipeline model is bound as the proposer at
        steer and consulted per query at adapt, so the generate phase requires
        `Capability.IN_PROCESS_TORCH`."""
        if self.prompt_lm is not None:
            return Requirements()
        return Requirements(generate=needs(
            Capability.IN_PROCESS_TORCH,
            hint="set prompt_lm to run CPO's per-query search off the pipeline model",
        ))

    def steer_access(self) -> ModelAccess:
        """`ModelAccess.ROLLOUTS` with `prompt_lm` supplied (offline data generation rides the
        session); `ModelAccess.MODULE` with `prompt_lm` unset (the live model is bound as the
        proposer)."""
        if self.prompt_lm is not None:
            return ModelAccess.ROLLOUTS
        return ModelAccess.MODULE

    def steer(
        self,
        model=None,
        tokenizer=None,
        session=None,
        **kwargs,
    ) -> None:
        self.tokenizer = tokenizer

        if self.prompt_lm is not None:
            proposer_lm = self.prompt_lm
            encoder_device = None
        else:
            proposer_lm = model
            encoder_device = next(model.parameters()).device if model is not None else None
        self._encoder = TextEncoder(
            self.embedding_model,
            device=encoder_device,
            trust_remote_code=self.trust_remote_code,
        )

        self._proposer = LLMMetaPromptProposer(
            llm=proposer_lm,
            tokenizer=tokenizer,
            meta_prompt_template=self.refinement_meta_prompt or refinement_meta_prompt.CPO_DEFAULT,
            gen_kwargs=self.proposer_gen_kwargs,
            parse_fn=parse_concise_instruction,
        )

        task_lm = model if model is not None else (SessionLM(session) if session is not None else None)
        offline_data = self.offline_data or self._generate_offline_data(task_lm, tokenizer)
        scorer = causal_reward.train(
            offline_data=offline_data,
            embedding_model=self.embedding_model,
            pca_query_dim=self.pca_query_dim,
            pca_prompt_dim=self.pca_prompt_dim,
            seed_prompt=self.seed_prompt,
            use_dml=self.use_dml,
            encoder=self._encoder,
            trust_remote_code=self.trust_remote_code,
        )

        self.memory = CPOMemory(causal_scorer=scorer)
        self._formatter = SystemPromptFormatter()

    def _generate_offline_data(self, task_lm, tokenizer) -> list[dict]:
        """Build ⟨query, prompt, score⟩ rows from `train_dataset` × proposer × metric.

        Each training row contributes `n_prompts_per_query` ⟨q, p, s⟩ triples (including the seed
        prompt as one of them, so the seed is always in-distribution for the reward model). Candidate
        prompts are scored with `TaskEvaluationScorer` using a one-row dev set per training query.
        """
        if not self.train_dataset:
            raise RuntimeError("offline_data is None and train_dataset is empty; nothing to fit on.")

        gen_kwargs = self.eval_gen_kwargs or {"max_new_tokens": 32, "do_sample": False}
        rows: list[dict] = []

        for example in self.train_dataset:
            query = example.get("input", example.get("query", ""))
            dev_row: dict[str, Any] = {"input": query}
            if "reference" in example:
                dev_row["reference"] = example["reference"]

            candidate_prompts = [self.seed_prompt]
            if self.n_prompts_per_query > 1:
                candidate_prompts.extend(
                    self._proposer.propose(seed=self.seed_prompt, n=self.n_prompts_per_query - 1)
                )

            scorer = TaskEvaluationScorer(
                task_lm=task_lm,
                tokenizer=tokenizer,
                dev_set=[dev_row],
                metric=self.metric,
                score_key=self.score_key,
                gen_kwargs=gen_kwargs,
            )
            scores = scorer.score(candidate_prompts)
            rows.extend(
                {"query": query, "prompt": prompt, "score": score}
                for prompt, score in zip(candidate_prompts, scores)
            )
        return rows

    def adapt_messages(
        self,
        messages: list[list[dict]],
        runtime_kwargs: dict | None = None,
    ) -> list[list[dict]] | None:
        if self.memory is None or self._formatter is None:
            raise RuntimeError("CPO.adapt_messages called before .steer().")
        out: list[list[dict]] = []
        for chat in messages:
            query_text = self._extract_query(chat)
            best = self._best_prompt_for_query(query_text)
            slot_memory = TextMemory(slots={"instruction": best})
            adapted = self._formatter.apply_to_messages([chat], slot_memory, runtime_kwargs)
            out.append(adapted[0])
        return out

    def adapt(
        self,
        input_ids: list[int] | torch.Tensor,
        runtime_kwargs: dict | None = None,
    ) -> list[int] | torch.Tensor:
        if self.memory is None or self._formatter is None or self.tokenizer is None:
            raise RuntimeError("CPO.adapt called before .steer().")
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids, dtype=torch.long)
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)

        adapted_rows: list[torch.Tensor] = []
        for row in input_ids:
            query_text = self.tokenizer.decode(row, skip_special_tokens=True)
            best = self._best_prompt_for_query(query_text)
            slot_memory = TextMemory(slots={"instruction": best})
            adapted = self._formatter.apply_to_ids(
                row.unsqueeze(0), slot_memory, self.tokenizer, runtime_kwargs
            )
            adapted_rows.append(adapted[0])

        # left-pad to max length (causal LM convention) and stack
        max_len = max(r.size(0) for r in adapted_rows)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        if pad_id is None:
            raise RuntimeError(
                "CPO.adapt: tokenizer has neither pad_token_id nor eos_token_id; cannot pad batch."
            )
        padded = torch.stack([
            torch.cat([torch.full((max_len - r.size(0),), pad_id, dtype=r.dtype, device=r.device), r])
            for r in adapted_rows
        ])
        return padded

    @staticmethod
    def _extract_query(chat: list[dict]) -> str:
        for msg in reversed(chat):
            if msg.get("role") == "user":
                return str(msg.get("content", ""))
        return str(chat[-1].get("content", "")) if chat else ""

    def _best_prompt_for_query(self, query: str) -> str:
        if self.memory is None:
            raise RuntimeError("CPO not steered; call .steer() first.")
        key_source = self.cache_key_normalizer(query) if self.cache_key_normalizer else query
        cache_key = hashlib.sha256(key_source.encode("utf-8")).hexdigest()
        if self.cache_queries and cache_key in self.memory.query_cache:
            return self.memory.query_cache[cache_key]
        best = self._tree_search(query)
        if self.cache_queries:
            self.memory.query_cache[cache_key] = best
        return best

    def _tree_search(self, query: str) -> str:
        """B-wide, K-retained, R-round search from `seed_prompt`.

        Survivors of each round are carried into the next round's candidate pool (elitism), so the
        search never returns a prompt that the trained reward model scores below the seed. When every
        proposal scores worse, the seed survives every round and is returned.
        """
        survivors: list[str] = [self.seed_prompt]
        for _ in range(self.rounds):
            candidates: list[str] = list(survivors)  # elitism: parents compete with their children
            for parent in survivors:
                spawned = self._proposer.propose(seed=parent, n=self.candidates_per_parent)
                candidates.extend([c.strip() for c in spawned if isinstance(c, str) and c.strip()])
            candidates = list(dict.fromkeys(candidates))  # dedupe, preserve order
            scores = self.memory.causal_scorer.score(
                candidates,
                queries=[{"text": query}] * len(candidates),
            )
            order = np.argsort(scores)[::-1]
            survivors = [candidates[i] for i in order[: self.retained_per_round]]
        if not survivors:
            return self.seed_prompt
        final_scores = self.memory.causal_scorer.score(
            survivors,
            queries=[{"text": query}] * len(survivors),
        )
        return survivors[int(np.argmax(final_scores))]
