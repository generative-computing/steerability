"""Causal reward model training (CPO Stage 1).

The CPO paper (Chen et al. 2026, arXiv:2602.01711) trains the reward model with Double Machine Learning
(DML) on PCA-reduced query and prompt embeddings, treating the prompt as a continuous treatment relative
to a fixed seed prompt. We expose a single `CausalRewardScorer` that conforms to `BaseScorer` and a
`train(...)` factory.

When `econml` is installed we use `CausalForestDML`. Otherwise we fall back to a plain
`GradientBoostingRegressor` over the concatenated (query, prompt) features, regressing on score
directly. The fallback drops the causal interpretation but keeps the `[cpo]` extra genuinely optional.
"""
from __future__ import annotations

import logging
import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor

from aisteer360.algorithms.input_control.common.scorers.base import BaseScorer
from aisteer360.algorithms.input_control.cpo.utils.embeddings import TextEncoder, fit_pca
from aisteer360.utils.optional import require

logger = logging.getLogger(__name__)


def _try_import_dml():
    try:
        from econml.dml import CausalForestDML  # type: ignore
        return CausalForestDML
    except ImportError:
        return None


@dataclass
class CausalRewardScorer(BaseScorer):
    """Trained CPO reward model. Implements `BaseScorer.score(prompts, queries)`.

    Internally:

      - embeds prompts and queries with the supplied encoder
      - applies separately-fitted PCAs to reduce dimensionality
      - holds either an econml `CausalForestDML` (`mode="dml"`) or a `GradientBoostingRegressor`
        (`mode="gbr"`) over the concatenated reduced features
    """

    encoder: TextEncoder
    query_pca: PCA
    prompt_pca: PCA
    seed_prompt_emb: np.ndarray  # the seed prompt's reduced embedding (treatment baseline for DML)
    estimator: object  # CausalForestDML or GradientBoostingRegressor
    mode: str  # "dml" or "gbr"

    def _featurize(self, prompts: Sequence[str], queries: Sequence[dict] | None) -> tuple[np.ndarray, np.ndarray]:
        prompt_full = self.encoder.encode_batch(list(prompts))
        prompt_red = self.prompt_pca.transform(prompt_full)

        if queries is not None:
            query_texts = [q.get("text", str(q)) for q in queries]
        else:
            query_texts = [""] * len(prompts)
        query_full = self.encoder.encode_batch(query_texts)
        query_red = self.query_pca.transform(query_full)
        return query_red, prompt_red

    def score(
        self,
        prompts: Sequence[str],
        queries: Sequence[dict] | None = None,
    ) -> list[float]:
        if not prompts:
            return []
        if queries is not None and len(queries) != len(prompts):
            raise ValueError(
                f"`queries` (len={len(queries)}) must align with `prompts` (len={len(prompts)})."
            )

        query_red, prompt_red = self._featurize(prompts, queries)

        if self.mode == "dml":
            # DML treatment effect of (prompt - seed_prompt) given query covariate
            treatment = prompt_red - self.seed_prompt_emb[None, :]
            # CausalForestDML.effect(X, T0, T1) is signed; we want effect of moving from seed to candidate
            t0 = np.zeros_like(treatment)
            effects = self.estimator.effect(query_red, T0=t0, T1=treatment)  # type: ignore[attr-defined]
            return [float(e) for e in np.atleast_1d(effects)]

        # GBR fallback: regress score on concatenated features
        x = np.concatenate([query_red, prompt_red], axis=1)
        preds = self.estimator.predict(x)  # type: ignore[attr-defined]
        return [float(p) for p in np.atleast_1d(preds)]

    def save(self, path: Path) -> None:
        path = Path(path)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "query_pca": self.query_pca,
                    "prompt_pca": self.prompt_pca,
                    "seed_prompt_emb": self.seed_prompt_emb,
                    "estimator": self.estimator,
                    "mode": self.mode,
                    "encoder_name_or_path": getattr(self.encoder, "_name_or_path", None),
                },
                f,
            )

    @classmethod
    def load(cls, path: Path, encoder: TextEncoder) -> "CausalRewardScorer":
        with open(path, "rb") as f:
            payload = pickle.load(f)
        return cls(
            encoder=encoder,
            query_pca=payload["query_pca"],
            prompt_pca=payload["prompt_pca"],
            seed_prompt_emb=payload["seed_prompt_emb"],
            estimator=payload["estimator"],
            mode=payload["mode"],
        )


def train(
    offline_data: Sequence[dict],
    embedding_model: str,
    pca_query_dim: int,
    pca_prompt_dim: int,
    seed_prompt: str,
    use_dml: bool | None = None,
    encoder: TextEncoder | None = None,
    device=None,
    trust_remote_code: bool = False,
) -> CausalRewardScorer:
    """Fit the CPO reward model on a list of `{query, prompt, score}` rows.

    Args:
        offline_data: list of dicts with keys `"query"`, `"prompt"`, `"score"`.
        embedding_model: HF model id for the encoder. Ignored if `encoder` is supplied.
        pca_query_dim, pca_prompt_dim: PCA target dimensions for queries and prompts.
        seed_prompt: The base prompt used as the treatment baseline.
        use_dml: If True, require econml; if False, force the GBR fallback. If None (default), use DML
            when econml is importable, otherwise fall back.
        encoder: Optional pre-built `TextEncoder`. Useful for tests to avoid reloading.
        device: Optional device for the encoder.
        trust_remote_code: Trust remote code when loading `embedding_model`. Ignored if `encoder` is supplied.
    """
    if not offline_data:
        raise ValueError("offline_data is empty.")
    enc = (
        encoder
        if encoder is not None
        else TextEncoder(embedding_model, device=device, trust_remote_code=trust_remote_code)
    )

    queries = [row["query"] for row in offline_data]
    prompts = [row["prompt"] for row in offline_data]
    scores = np.array([row["score"] for row in offline_data], dtype=float)

    if scores.size:
        values, counts = np.unique(scores, return_counts=True)
        top_value = float(values[int(np.argmax(counts))])
        top_frac = float(counts.max()) / float(scores.size)
        if values.size == 1:
            warnings.warn(
                f"CPO offline_data scores are constant (value={top_value:.4f} over {scores.size} rows). "
                "A reward model fit on constant targets predicts that constant for every prompt, so the "
                "tree search degenerates to arbitrary tie-breaking. Use a metric/dev set that produces "
                "score variance.",
                UserWarning,
                stacklevel=2,
            )
        elif top_frac >= 0.95:
            warnings.warn(
                f"CPO offline_data scores are nearly saturated: value {top_value:.4f} covers "
                f"{100.0 * top_frac:.0f}% of {scores.size} rows. The reward model will have almost no "
                "signal to rank prompts; consider harder queries, stricter or graded scoring, or more "
                "diverse candidate prompts.",
                UserWarning,
                stacklevel=2,
            )

    query_full = enc.encode_batch(queries)
    prompt_full = enc.encode_batch(prompts + [seed_prompt])

    query_pca = fit_pca(query_full, pca_query_dim)
    prompt_pca = fit_pca(prompt_full, pca_prompt_dim)

    query_red = query_pca.transform(query_full)
    prompt_red = prompt_pca.transform(prompt_full[:-1])
    seed_prompt_red = prompt_pca.transform(prompt_full[-1:])[0]

    DML = _try_import_dml() if use_dml is None else (_try_import_dml() if use_dml else None)
    if use_dml is True and DML is None:
        require("econml")  # raises ImportError naming the [cpo] extra when econml is absent
        raise ImportError("CPO with use_dml=True requires `econml.dml.CausalForestDML`.")

    if DML is not None:
        treatment = prompt_red - seed_prompt_red[None, :]
        try:
            estimator = DML(
                model_y=GradientBoostingRegressor(),
                model_t=GradientBoostingRegressor(),
                discrete_treatment=False,
                random_state=0,
            )
            estimator.fit(Y=scores, T=treatment, X=query_red)
            mode = "dml"
        except Exception as exc:  # econml occasionally fails on tiny / degenerate data
            logger.warning("CausalForestDML.fit failed (%s); falling back to GBR.", exc)
            estimator = GradientBoostingRegressor()
            estimator.fit(np.concatenate([query_red, prompt_red], axis=1), scores)
            mode = "gbr"
    else:
        estimator = GradientBoostingRegressor()
        estimator.fit(np.concatenate([query_red, prompt_red], axis=1), scores)
        mode = "gbr"

    return CausalRewardScorer(
        encoder=enc,
        query_pca=query_pca,
        prompt_pca=prompt_pca,
        seed_prompt_emb=seed_prompt_red,
        estimator=estimator,
        mode=mode,
    )
