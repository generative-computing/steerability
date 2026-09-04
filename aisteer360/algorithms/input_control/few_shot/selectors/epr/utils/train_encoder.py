"""Contrastive (DPR-style) training of the EPR dual encoder.

The encoder is a single transformer (we use the same model for both query and document encodings, sharing
weights). Training uses an InfoNCE-style objective with in-batch negatives and the LM-labeled hard
negatives.
"""
from __future__ import annotations

import logging
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from aisteer360.algorithms.input_control.few_shot.selectors.epr.utils.lm_labeling import LabeledPair

logger = logging.getLogger(__name__)


class EPREncoder:
    """A thin encoder wrapper exposing `encode(text) -> np.ndarray`.

    Uses mean-pooled last_hidden_state with attention masking. Backbone is shared between queries and
    pool items (a single-tower retriever, like DPR's text-only variant).
    """

    def __init__(self, model_name_or_path: str = "bert-base-uncased", device: str | torch.device | None = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path)
        if device is not None:
            self.model = self.model.to(device)
        self.model.eval()

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @torch.no_grad()
    def encode(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)
        outputs = self.model(**inputs)
        hidden = outputs.last_hidden_state  # [1, T, H]
        mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)  # [1, T, 1]
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return pooled[0].cpu().numpy()


class _ContrastiveDataset(Dataset):
    def __init__(
        self,
        items: Sequence[dict],
        labeled: Sequence[LabeledPair],
        input_field: str,
        output_field: str,
    ):
        self.items = list(items)
        self.labeled = list(labeled)
        self.input_field = input_field
        self.output_field = output_field

    def __len__(self) -> int:
        return len(self.labeled)

    def _format_query(self, item: dict) -> str:
        return f"{item.get(self.input_field, '')}"

    def _format_demo(self, item: dict) -> str:
        return f"Input: {item.get(self.input_field, '')}\nOutput: {item.get(self.output_field, '')}"

    def __getitem__(self, idx: int) -> dict:
        pair = self.labeled[idx]
        anchor = self.items[pair.anchor_index]
        # take one random positive and one random negative; this matches the in-batch + hard-neg recipe
        pos_idx = pair.positives[np.random.randint(len(pair.positives))] if pair.positives else pair.anchor_index
        neg_idx = pair.negatives[np.random.randint(len(pair.negatives))] if pair.negatives else None
        return {
            "query": self._format_query(anchor),
            "positive": self._format_demo(self.items[pos_idx]),
            "negative": self._format_demo(self.items[neg_idx]) if neg_idx is not None else None,
        }


def _collate(batch: list[dict], tokenizer) -> dict:
    queries = [row["query"] for row in batch]
    positives = [row["positive"] for row in batch]
    negatives = [row["negative"] for row in batch if row["negative"] is not None]
    out = {
        "queries": tokenizer(queries, return_tensors="pt", truncation=True, padding=True),
        "positives": tokenizer(positives, return_tensors="pt", truncation=True, padding=True),
    }
    if negatives:
        out["negatives"] = tokenizer(negatives, return_tensors="pt", truncation=True, padding=True)
    return out


def _mean_pool(model, batch_inputs):
    outputs = model(**batch_inputs)
    hidden = outputs.last_hidden_state
    mask = batch_inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
    return pooled


def train_encoder(
    items: Sequence[dict],
    labeled: Sequence[LabeledPair],
    base_encoder: str = "bert-base-uncased",
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    input_field: str = "input",
    output_field: str = "output",
    device: str | torch.device | None = None,
) -> EPREncoder:
    """Train (or fine-tune) a dual encoder for EPR.

    Returns:
        An `EPREncoder` ready for inference-time `encode()`.
    """
    encoder = EPREncoder(model_name_or_path=base_encoder, device=device)
    if not labeled:
        logger.warning("EPR train_encoder: no labeled pairs supplied; returning untrained encoder.")
        return encoder

    encoder.model.train()
    optimizer = torch.optim.AdamW(encoder.model.parameters(), lr=learning_rate)

    dataset = _ContrastiveDataset(items, labeled, input_field, output_field)

    def collate(batch):
        out = _collate(batch, encoder.tokenizer)
        for key, tensors in out.items():
            out[key] = {k: v.to(encoder.device) for k, v in tensors.items()}
        return out

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)

    for epoch in range(epochs):
        epoch_loss = 0.0
        steps = 0
        for batch in loader:
            optimizer.zero_grad()
            q_emb = _mean_pool(encoder.model, batch["queries"])  # [B, H]
            p_emb = _mean_pool(encoder.model, batch["positives"])  # [B, H]

            # in-batch negatives
            logits = q_emb @ p_emb.T  # [B, B]; row i should pick column i (the positive)
            if "negatives" in batch:
                n_emb = _mean_pool(encoder.model, batch["negatives"])  # [Bn, H]
                neg_logits = q_emb @ n_emb.T  # [B, Bn]
                logits = torch.cat([logits, neg_logits], dim=1)

            targets = torch.arange(q_emb.size(0), device=q_emb.device)
            loss = nn.functional.cross_entropy(logits, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            steps += 1
        logger.debug("EPR epoch %d loss=%.4f", epoch, epoch_loss / max(steps, 1))

    encoder.model.eval()
    return encoder


# convenience name aligning with the design doc: `train_encoder.train(...)`.
def train(
    labeled,
    items,
    base_encoder: str = "bert-base-uncased",
    epochs: int = 3,
    batch_size: int = 8,
    **kwargs,
) -> EPREncoder:
    return train_encoder(
        items=items,
        labeled=labeled,
        base_encoder=base_encoder,
        epochs=epochs,
        batch_size=batch_size,
        **kwargs,
    )
