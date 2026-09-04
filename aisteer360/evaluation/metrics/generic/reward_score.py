from collections.abc import Mapping
from typing import Any, Literal

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from aisteer360.evaluation.metrics.base import Metric


class RewardScore(Metric):
    """
    Compute (pointwise) reward scores using a pretrained reward model.

    This metric expects a Hugging Face sequence-classification model. The typical case for reward models is
    `num_labels == 1`, where the single logit is taken as the reward. If `num_labels > 1`, you can select a class index
    and/or apply a probability transform.

    Args:
        model_or_id: HF model id (str) or an already-instantiated
            `PreTrainedModel` (sequence-classification head).
        tokenizer: Optional tokenizer. If None, loaded from `model_or_id`.
        device: 'cuda' | 'mps' | 'cpu'. Defaults to an available accelerator.
        batch_size: Batch size for scoring.
        max_length: Truncation length for encoding. If None, no truncation.
        score_transform: How to map logits to a scalar:
            - 'identity' -> use raw logit (default; good for num_labels==1)
            - 'sigmoid' -> sigmoid(logit) in [0,1] (num_labels==1)
            - 'softmax' -> softmax(logits)[label_index]
            - 'log_softmax'-> log_softmax(logits)[label_index]
        label_index: Class index to select when `num_labels > 1`.
        return_logits: If True, also return raw logits per sample (for debugging).

    Notes:

        - If your reward model was trained to take both prompt and response, pass `prompts=[...]`. If not, omit
        `prompts` and only responses are encoded.
        - To add pairwise comparisons, compute two calls (candidate vs. baseline) and take the difference externally,
        or extend this class to accept a `reference_responses` kwarg and return margins.
    """

    def __init__(
        self,
        model_or_id: str | PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase | None = None,
        device: str | None = None,
        batch_size: int = 8,
        max_length: int | None = 1024,
        score_transform: Literal["identity", "sigmoid", "softmax", "log_softmax"] = "identity",
        label_index: int = 0,
        return_logits: bool = False,
        **extras: Any,
    ) -> None:
        super().__init__(**extras)

        # load model/tokenizer
        if isinstance(model_or_id, PreTrainedModel):
            self.model: PreTrainedModel = model_or_id
            if tokenizer is None:
                raise ValueError("If passing a model instance, you must also pass its tokenizer.")
            self.tokenizer = tokenizer
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_or_id)
            self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_or_id)

        # device selection mirrors the base judge/perplexity defaults
        self.device = device or (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        self.model.to(self.device).eval()

        self.batch_size = int(batch_size)
        self.max_length = max_length
        self.score_transform = score_transform
        self.label_index = int(label_index)
        self.return_logits = bool(return_logits)

        # ensure we have a pad token for batching
        if self.tokenizer.pad_token is None:
            # fall back to eos/sep if pad is unset
            self.tokenizer.pad_token = getattr(self.tokenizer, "eos_token", None) or getattr(self.tokenizer, "sep_token", None)

    def _score_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Map logits -> scalar rewards according to `score_transform`.
        Supports both [B, 1] and [B, C] shapes.
        """
        if logits.ndim != 2:
            raise ValueError(f"Expected logits to be 2D [B, C], got shape={tuple(logits.shape)}")
        batch_size, num_labels = logits.shape

        if num_labels == 1:
            scores = logits.squeeze(-1)
            if self.score_transform == "sigmoid":
                scores = torch.sigmoid(scores)
            elif self.score_transform == "identity":
                pass
            elif self.score_transform in ("softmax", "log_softmax"):
                raise ValueError("softmax/log_softmax require num_labels > 1.")
            else:
                raise ValueError(f"Unknown score_transform: {self.score_transform}")
            return scores

        # num_labels > 1
        if not (0 <= self.label_index < num_labels):
            raise IndexError(f"label_index={self.label_index} out of range for num_labels={num_labels}")
        if self.score_transform == "softmax":
            probs = torch.softmax(logits, dim=-1)
            return probs[:, self.label_index]
        elif self.score_transform == "log_softmax":
            log_probs = F.log_softmax(logits, dim=-1)
            return log_probs[:, self.label_index]
        elif self.score_transform == "identity":
            return logits[:, self.label_index]
        elif self.score_transform == "sigmoid":
            # rarely meaningful for multi-logit heads, but keep for completeness
            return torch.sigmoid(logits[:, self.label_index])
        else:
            raise ValueError(f"Unknown score_transform: {self.score_transform}")

    @torch.no_grad()
    def compute(
        self,
        responses: list[str] | list[dict] | None = None,
        prompts: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Score each response (optionally conditioned on its prompt).

        Args:
            responses: Text to score, or list of generation dicts (with keys 'response' and optionally 'prompt').
            prompts: Optional list of prompts (same length as responses) that will be encoded as text pairs.

        Returns:
            dict[str, Any]: A dict with keys:

                - ``"mean_reward"``: mean reward score over all responses.
                - ``"rewards"``: list of per-sample reward scores in input order.
                - ``"logits"``: (optional) list of raw logits per sample, only included if ``return_logits=True``.
        """
        if not responses:
            return {"mean_reward": 0.0, "rewards": []}

        # normalize input: allow either list[str] or list[dict]
        if isinstance(responses[0], Mapping):
            gen_dicts = responses
            texts = [d.get("response", "") for d in gen_dicts]

            if prompts is None:
                extracted_prompts = [d.get("prompt") for d in gen_dicts]
                if all(isinstance(p, str) for p in extracted_prompts):
                    prompts = extracted_prompts
                else:
                    prompts = None
        else:
            texts = responses

        if prompts is not None and len(prompts) != len(texts):
            raise AssertionError("If provided, `prompts` must be the same length as `responses`.")

        rewards: list[float] = []
        all_logits: list[list[float]] = []

        for batch_start in range(0, len(texts), self.batch_size):
            response_batch = texts[batch_start : batch_start + self.batch_size]
            if prompts is not None:
                prompt_batch = prompts[batch_start : batch_start + self.batch_size]
                encoding = self.tokenizer(
                    prompt_batch,
                    response_batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
            else:
                encoding = self.tokenizer(
                    response_batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )

            encoding = {key: value.to(self.device) for key, value in encoding.items()}
            output = self.model(**encoding)
            logits = output.logits  # [B, C]
            batch_scores = self._score_logits(logits)

            rewards.extend(batch_scores.detach().cpu().tolist())
            if self.return_logits:
                all_logits.extend(logits.detach().cpu().tolist())

        result: dict[str, Any] = {
            "mean_reward": float(sum(rewards) / len(rewards)) if rewards else 0.0,
            "rewards": rewards,
        }
        if self.return_logits:
            result["logits"] = all_logits
        return result
