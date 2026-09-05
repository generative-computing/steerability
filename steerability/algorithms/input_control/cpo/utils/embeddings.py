"""Sentence embeddings + PCA reduction for CPO.

Uses a single-tower transformer with mean-pooled last_hidden_state. The CPO paper specifies
nomic-embed-text-v1.5 as the default, but any HF text encoder works.
"""
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import torch
from sklearn.decomposition import PCA
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


class TextEncoder:
    """Mean-pooled HF encoder. Not a `BaseScorer`; this is purely a featurizer for CPO's reward model."""

    def __init__(
        self,
        model_name_or_path: str,
        device: str | torch.device | None = None,
        trust_remote_code: bool = True,
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
        self.model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
        if device is not None:
            self.model = self.model.to(device)
        self.model.eval()

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @torch.no_grad()
    def encode_batch(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.model.config.hidden_size), dtype=np.float32)
        inputs = self.tokenizer(
            list(texts), return_tensors="pt", truncation=True, padding=True
        ).to(self.device)
        outputs = self.model(**inputs)
        hidden = outputs.last_hidden_state  # [B, T, H]
        mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return pooled.cpu().numpy()

    def encode(self, text: str) -> np.ndarray:
        return self.encode_batch([text])[0]


def fit_pca(matrix: np.ndarray, n_components: int) -> PCA:
    """Fit a PCA whose number of components is clipped to the available rank."""
    n_samples, n_features = matrix.shape
    max_components = max(1, min(n_components, n_samples, n_features))
    pca = PCA(n_components=max_components)
    pca.fit(matrix)
    return pca
