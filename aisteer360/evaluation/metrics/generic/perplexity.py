"""Unconditional perplexity of each response, computed through the scoring seam."""
from __future__ import annotations

import math
import warnings
from collections import defaultdict
from typing import Any

import torch

from aisteer360.algorithms.core.execution.backend import Backend
from aisteer360.algorithms.core.execution.params import GenerationParams
from aisteer360.algorithms.core.execution.payloads import PreparedPrompt, ScoringItem
from aisteer360.algorithms.core.execution.spec import BackendSpec
from aisteer360.evaluation.metrics.backend_utils import resolve_metric_backend
from aisteer360.evaluation.metrics.base import Metric


class Perplexity(Metric):
    """Unconditional perplexity of each response.

    Perplexity is the exponentiated mean cross-entropy between the model's predicted distribution
    and the reference tokens; lower is better. The computation is the seam's `score` operation
    (teacher-forced log-probabilities of reference tokens), so the metric runs on every backend the
    seam supports.

    The judge model is configured by a model reference and a backend, never by a live model object.
    Backends are cached by spec (see `resolve_metric_backend`), so a `Perplexity` and a judge
    configured with equal specs share one loaded model or engine. The backend is resolved per
    `compute()`, so after `release_metric_backends()` the next call constructs it again.

    Each response is tokenized with the backend tokenizer (`add_special_tokens=False`) and scored in
    one of two conditioning modes:

    - `add_bos=True` and the tokenizer has a BOS token: the prompt is the single BOS token and every
      response token is scored.
    - otherwise: the prompt is the response's first token (conditioning context) and the remaining
      tokens are scored.

    Degenerate rows, an empty tokenization or a single token in the no-BOS mode, contribute
    `float("nan")` with one `UserWarning`, keeping the output length aligned with `responses`.
    Session scoring is decoder-only, matching the metric's causal-LM assumption.

    Args:
        model: Model reference (hub id or local path), or None when `backend` carries the identity.
        backend: A `BackendSpec`, a backend-kind string (`"huggingface"` or `"vllm"`), a live
            `Backend`, or None (in-process Hugging Face). A bare `"vllm-serve"` string is rejected;
            pass a `BackendSpec` with `base_url`.
        batch_size: Number of same-length references scored per `session.score` call. Defaults to 8.
        add_bos: Whether to prepend the tokenizer's BOS token so the first response token is also
            scored. Ignored when the tokenizer has no BOS token. Defaults to True.
        max_length: Truncate each tokenized response to this length when set. Defaults to None.
        name: Metric name; defaults to the class name.

    Attributes:
        add_bos: Whether a BOS token is prepended before scoring.
        batch_size: Number of same-length references scored per call.
        max_length: Truncation length for tokenized responses, or None.
    """

    def __init__(
        self,
        model: str | None = None,
        *,
        backend: "BackendSpec | str | Backend | None" = None,
        batch_size: int = 8,
        add_bos: bool = True,
        max_length: int | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self._model_ref = model
        self._backend_ref = backend
        resolve_metric_backend(model, backend)  # validate the identity now; the backend is re-resolved per compute
        self.batch_size = int(batch_size)
        self.add_bos = bool(add_bos)
        self.max_length = max_length

    @property
    def _backend(self) -> Backend:
        """The configured backend, resolved through the metric cache on each access.

        A cache lookup while the backend is cached; after `release_metric_backends()` the next
        access constructs it again, so a released metric stays usable. A live `Backend` passed at
        construction is returned as is.
        """
        return resolve_metric_backend(self._model_ref, self._backend_ref)

    def _tokenize(self, tokenizer, responses: list[str]) -> list[list[int]]:
        """Tokenize each response with `add_special_tokens=False`, truncating to `max_length`."""
        token_lists: list[list[int]] = []
        for response in responses:
            ids = tokenizer(response, add_special_tokens=False)["input_ids"]
            if self.max_length is not None:
                ids = ids[: self.max_length]
            token_lists.append(list(ids))
        return token_lists

    def _scoring_item(self, tokens: list[int], bos_id: int | None) -> ScoringItem | None:
        """Build the `ScoringItem` for one tokenized response, or None for a degenerate row."""
        if not tokens:
            return None
        if self.add_bos and bos_id is not None:
            prompt = PreparedPrompt.from_token_ids([bos_id])
            ref = tokens
        else:
            if len(tokens) < 2:
                return None
            prompt = PreparedPrompt.from_token_ids([tokens[0]])
            ref = tokens[1:]
        return ScoringItem(prompt=prompt, ref_output_ids=torch.tensor(ref, dtype=torch.long))

    def compute(
        self,
        responses: list[str],
        prompts: list[str] | None = None,
    ) -> dict[str, float | list[float]]:
        """Compute per-response perplexity and the mean across the batch.

        Responses are tokenized, degenerate rows recorded as `nan`, and the remainder grouped by
        reference length (the seam scores one reference length per call), chunked by `batch_size`,
        and scored through one session. Per response, perplexity is `exp(-mean(logprobs))` over that
        row's returned log-probabilities.

        Args:
            responses: Text sequences to score.
            prompts: Unused; present for the uniform metric API.

        Returns:
            Dict with keys:

                - `"mean_perplexity"`: Mean perplexity over all responses (nan rows excluded).
                - `"perplexities"`: Per-response perplexities in input order (nan for degenerate rows).
        """
        if not responses:
            return {"mean_perplexity": 0.0, "perplexities": []}

        perplexities: list[float] = [float("nan")] * len(responses)
        with self._backend.open_session() as session:
            tokenizer = session.tokenizer
            bos_id = getattr(tokenizer, "bos_token_id", None)
            token_lists = self._tokenize(tokenizer, responses)

            items: dict[int, ScoringItem] = {}
            degenerate = False
            for index, tokens in enumerate(token_lists):
                item = self._scoring_item(tokens, bos_id)
                if item is None:
                    degenerate = True
                else:
                    items[index] = item
            if degenerate:
                warnings.warn(
                    "One or more responses were too short to score (empty tokenization, or a single "
                    "token without a BOS token); they contribute float('nan').",
                    UserWarning,
                )

            by_length: dict[int, list[int]] = defaultdict(list)
            for index, item in items.items():
                by_length[item.ref_output_ids.shape[-1]].append(index)

            for indices in by_length.values():
                for start in range(0, len(indices), self.batch_size):
                    chunk = indices[start:start + self.batch_size]
                    logprobs = session.score([items[index] for index in chunk], GenerationParams())
                    for row, index in enumerate(chunk):
                        mean_logprob = float(logprobs[row].mean())
                        perplexities[index] = math.exp(-mean_logprob)

        finite = [value for value in perplexities if not math.isnan(value)]
        mean_perplexity = sum(finite) / len(finite) if finite else float("nan")
        return {"mean_perplexity": mean_perplexity, "perplexities": perplexities}
