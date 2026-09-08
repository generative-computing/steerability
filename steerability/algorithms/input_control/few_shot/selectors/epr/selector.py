"""EPRSelector, a learned dense retriever for few-shot example selection.

Reference:

  - "Learning To Retrieve Prompts for In-Context Learning"
    Ohad Rubin, Jonathan Herzig, Jonathan Berant
    [https://arxiv.org/abs/2112.08633](https://arxiv.org/abs/2112.08633)
"""
from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from steerability.algorithms.input_control.common.memory.pool import PoolMemory
from steerability.algorithms.input_control.common.selectors.dense_retrieval import DenseRetrievalSelector
from steerability.algorithms.input_control.few_shot.selectors.epr.utils import bm25_index, lm_labeling, train_encoder

logger = logging.getLogger(__name__)


class EPRSelector(DenseRetrievalSelector):
    """Learned dense retriever for few-shot example selection (EPR).

    At runtime, `select()` encodes the query and returns the nearest items in the example pool by
    cosine similarity over precomputed embeddings. Its `prepare()` runs the EPR training procedure:

      1. For each training pair `(x, y)`, use BM25 over `y` to retrieve a candidate set of size
         `candidate_set_size`.
      2. Use the scoring LM to label `k_pos` positives and `k_neg` negatives in the candidate set.
      3. Train a single-tower BERT-base encoder via contrastive loss with in-batch and hard negatives.

    The trained encoder is then used at inference time to embed the query and rank pool items.

    Args:
        scoring_lm: Causal LM used to score (anchor, candidate) pairs during offline labeling.
        scoring_tokenizer: Tokenizer paired with `scoring_lm`.
        base_encoder: Hugging Face model id of the encoder backbone.
        candidate_set_size: Number of BM25-nearest candidates considered per anchor (paper's `L`).
        k_pos: Number of positives extracted per anchor.
        k_neg: Number of hard negatives extracted per anchor.
        train_epochs: Epochs of contrastive training over the labeled pairs.
        batch_size: Mini-batch size during contrastive training.
        max_anchors: Optional cap on anchors used during labeling (useful for fast CI).
        input_field: Item key holding the input text.
        output_field: Item key holding the output / target text.
        device: Optional device override for the encoder.
    """

    def __init__(
        self,
        scoring_lm: PreTrainedModel,
        scoring_tokenizer: PreTrainedTokenizerBase,
        base_encoder: str = "bert-base-uncased",
        candidate_set_size: int = 50,
        k_pos: int = 5,
        k_neg: int = 5,
        train_epochs: int = 3,
        batch_size: int = 8,
        max_anchors: int | None = None,
        input_field: str = "input",
        output_field: str = "output",
        device: Any = None,
    ) -> None:
        # skip DenseRetrievalSelector's __init__; we own the encoder and create it later in prepare()
        self.encoder = None  # filled in prepare()
        self.similarity = "cosine"
        self.item_to_text = self._format_item
        self.embedding_key = "_epr_embedding"

        self.scoring_lm = scoring_lm
        self.scoring_tokenizer = scoring_tokenizer
        self.base_encoder_name = base_encoder
        self.candidate_set_size = candidate_set_size
        self.k_pos = k_pos
        self.k_neg = k_neg
        self.train_epochs = train_epochs
        self.batch_size = batch_size
        self.max_anchors = max_anchors
        self.input_field = input_field
        self.output_field = output_field
        self.device = device

    def _format_item(self, item: Any) -> str:
        if isinstance(item, dict):
            return f"Input: {item.get(self.input_field, '')}\nOutput: {item.get(self.output_field, '')}"
        return str(item)

    def _extract_query_text(self, query: Any) -> str:
        # when called from FewShot.adapt_messages, `query` is the chat (a list of dicts).
        if isinstance(query, list) and query and isinstance(query[0], dict):
            for msg in reversed(query):
                if msg.get("role") == "user":
                    return str(msg.get("content", ""))
            return str(query[-1].get("content", ""))
        if isinstance(query, dict):
            return str(query.get(self.input_field, "") or query.get("query", ""))
        if isinstance(query, str):
            return query
        return str(query)

    def prepare(
        self,
        model=None,
        tokenizer=None,
        data: PoolMemory[dict] | Sequence[dict] | None = None,
        **kwargs,
    ) -> None:
        if data is None:
            logger.warning("EPRSelector.prepare called with no data; nothing to train on.")
            return

        if isinstance(data, PoolMemory):
            items = list(data.items)
        else:
            items = list(data)

        if not items:
            logger.warning("EPRSelector.prepare: empty pool; skipping training.")
            return

        candidate_sets = bm25_index.build_and_query(
            items,
            query_field=self.output_field,
            candidate_set_size=self.candidate_set_size,
        )
        labeled = lm_labeling.label_pairs(
            items=items,
            candidate_sets=candidate_sets,
            scoring_lm=self.scoring_lm,
            scoring_tokenizer=self.scoring_tokenizer,
            k_pos=self.k_pos,
            k_neg=self.k_neg,
            input_field=self.input_field,
            output_field=self.output_field,
            max_anchors=self.max_anchors,
        )
        self.encoder = train_encoder.train(
            labeled=labeled,
            items=items,
            base_encoder=self.base_encoder_name,
            epochs=self.train_epochs,
            batch_size=self.batch_size,
            input_field=self.input_field,
            output_field=self.output_field,
            device=self.device,
        )

        # pre-compute pool embeddings into PoolMemory metadata for inference-time lookup.
        embeddings = [self.encoder.encode(self._format_item(item)) for item in items]
        if isinstance(data, PoolMemory):
            data.metadata[self.embedding_key] = embeddings
            for item, emb in zip(data.items, embeddings):
                if isinstance(item, dict):
                    item[self.embedding_key] = emb
        else:
            for item, emb in zip(items, embeddings):
                if isinstance(item, dict):
                    item[self.embedding_key] = emb

    def select(
        self,
        items: Sequence[dict],
        query: Any = None,
        k: int = 1,
        context: dict | None = None,
    ) -> list[dict]:
        if self.encoder is None:
            raise RuntimeError("EPRSelector.select called before prepare(); the encoder is not trained.")
        query_text = self._extract_query_text(query)
        query_emb = np.asarray(self.encoder.encode(query_text))
        return self._top_k_by_similarity(items, query_emb, k)
