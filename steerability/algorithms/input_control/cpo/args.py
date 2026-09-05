"""Arguments for the CPO input control."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from steerability.algorithms.core.base_args import BaseArgs


@dataclass
class CPOArgs(BaseArgs):
    """Arguments for CPO, causal prompt optimization (Chen et al. 2026)."""

    seed_prompt: str = field(
        default="",
        metadata={"help": "Base prompt used as the DML treatment baseline and tree-search root."},
    )

    train_dataset: list[dict] = field(
        default_factory=list,
        metadata={
            "help": (
                "Task instances for offline data generation (each row is consumed by the prompt LM at "
                "training time). Required when `offline_data` is None."
            )
        },
    )

    offline_data: list[dict] | None = field(
        default=None,
        metadata={
            "help": (
                "Pre-built ⟨query, prompt, score⟩ triplets. If None, will be generated from "
                "`train_dataset` × proposer."
            )
        },
    )

    n_prompts_per_query: int = field(
        default=200,
        metadata={"help": "When generating offline data, how many candidate prompts per training query."},
    )

    embedding_model: str = field(
        default="nomic-ai/nomic-embed-text-v1.5",
        metadata={"help": "HF text encoder used to featurize queries and prompts."},
    )

    trust_remote_code: bool = field(
        default=True,
        metadata={"help": (
            "Trust remote code when loading `embedding_model`. Defaults to True for the default "
            "encoder, which ships custom modeling code. Set False for standard encoders."
        )},
    )

    pca_query_dim: int = field(
        default=40,
        metadata={"help": "PCA target dimensionality for query embeddings."},
    )

    pca_prompt_dim: int = field(
        default=15,
        metadata={"help": "PCA target dimensionality for prompt embeddings."},
    )

    prompt_lm: Any = field(
        default=None,
        metadata={"help": "LLM used to propose candidate prompts in Stage 2 (tree search)."},
    )

    rounds: int = field(
        default=3,
        metadata={"help": "Number of search rounds (R in the paper)."},
    )

    candidates_per_parent: int = field(
        default=5,
        metadata={"help": "Number of candidates spawned from each retained parent (B)."},
    )

    retained_per_round: int = field(
        default=3,
        metadata={"help": "Number of survivors kept after each round (K)."},
    )

    row_scorer: Callable[[str, dict], float] | None = field(
        default=None,
        metadata={"help": "Per-row scorer (response, row) -> float used to score offline-generated rows."},
    )

    cache_queries: bool = field(
        default=True,
        metadata={"help": "Cache the chosen prompt per-query so identical queries skip the search."},
    )

    cache_key_normalizer: Callable[[str], str] | None = field(
        default=None,
        metadata={
            "help": (
                "Optional `str -> str` applied to a query before hashing its cache key, so near-duplicate "
                "queries can share a cached prompt."
            )
        },
    )

    use_dml: bool | None = field(
        default=None,
        metadata={
            "help": (
                'If True, require econml\'s CausalForestDML (installed separately: `pip install "econml>=0.16,'
                '<0.17"`). If False, force the GradientBoostingRegressor fallback. None auto-detects.'
            )
        },
    )

    refinement_meta_prompt: str | None = field(
        default=None,
        metadata={"help": "Override the CPO refinement template; defaults to refinement_meta_prompt.CPO_DEFAULT."},
    )

    proposer_gen_kwargs: dict | None = field(
        default=None,
        metadata={"help": "Generation kwargs for the prompt proposer LLM."},
    )

    eval_gen_kwargs: dict | None = field(
        default=None,
        metadata={"help": "Generation kwargs used during offline data scoring."},
    )

    memory: Any = field(
        default=None,
        metadata={
            "help": (
                "Precomputed memory: a `CPOMemory`, or a path to a directory saved with "
                "`CPOMemory.save`. When provided, steer() installs (or loads) it directly and "
                "skips offline-data generation and scorer training."
            )
        },
    )

    def __post_init__(self) -> None:
        if not isinstance(self.seed_prompt, str) or not self.seed_prompt:
            raise ValueError("`seed_prompt` must be a non-empty str.")
        if self.memory is None and self.offline_data is None:
            if not self.train_dataset:
                raise ValueError("Either `offline_data` or `train_dataset` must be supplied.")
            if self.row_scorer is None:
                raise ValueError("`row_scorer` is required when offline_data is generated from train_dataset.")
            if self.prompt_lm is None:
                raise ValueError("`prompt_lm` is required when offline_data is generated from train_dataset.")
        if self.rounds <= 0 or self.candidates_per_parent <= 0 or self.retained_per_round <= 0:
            raise ValueError("rounds, candidates_per_parent, retained_per_round must all be positive.")
