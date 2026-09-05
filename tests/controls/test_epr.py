"""Tests for EPR — learned dense retriever for few-shot example selection."""
from __future__ import annotations

from typing import Any

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from steerability.algorithms.core.steering_pipeline import SteeringPipeline
from steerability.algorithms.input_control.common.memory.pool import PoolMemory
from steerability.algorithms.input_control.common.selectors.base import BaseSelector
from steerability.algorithms.input_control.few_shot import FewShot
from steerability.algorithms.input_control.few_shot.selectors.epr import EPRSelector
from steerability.algorithms.input_control.few_shot.selectors.epr.utils import bm25_index

TINY_LM = "hf-internal-testing/tiny-random-LlamaForCausalLM"
TINY_BERT = "hf-internal-testing/tiny-random-BertModel"


@pytest.fixture(scope="module")
def tiny_scoring_lm():
    model = AutoModelForCausalLM.from_pretrained(TINY_LM, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(TINY_LM, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


class TestBM25Index:
    def test_naive_bm25_ranks_word_overlap(self):
        docs = ["the cat sat", "the dog ran", "the cat ran fast", "a bird flew"]
        idx = bm25_index.build_index(docs)
        scores = list(idx.get_scores(["cat"]))
        assert scores[0] > 0
        assert scores[2] > 0
        assert scores[3] == 0
        assert scores[1] == 0

    def test_query_returns_top_k_indices(self):
        docs = ["cat", "dog", "cat dog", "fish"]
        idx = bm25_index.build_index(docs)
        top = bm25_index.query(idx, "cat", top_k=2)
        assert len(top) == 2
        # both items containing "cat" should be at the top
        assert set(top) == {0, 2}

    def test_build_and_query_excludes_self(self):
        items = [
            {"input": "x1", "output": "cat"},
            {"input": "x2", "output": "dog"},
            {"input": "x3", "output": "cat"},
        ]
        result = bm25_index.build_and_query(items, query_field="output", candidate_set_size=2)
        for anchor_idx, candidate_indices in result.items():
            assert anchor_idx not in candidate_indices


class TestEPRSelector:
    def test_select_before_prepare_raises(self, tiny_scoring_lm):
        scoring_lm, scoring_tok = tiny_scoring_lm
        selector = EPRSelector(
            scoring_lm=scoring_lm,
            scoring_tokenizer=scoring_tok,
            base_encoder=TINY_BERT,
        )
        with pytest.raises(RuntimeError, match="prepare"):
            selector.select([{"input": "a", "output": "b"}], query="q", k=1)

    def test_subclass_is_dense_retrieval_selector(self):
        from inspect import isclass

        from steerability.algorithms.input_control.common.selectors.dense_retrieval import DenseRetrievalSelector
        assert issubclass(EPRSelector, DenseRetrievalSelector)
        assert issubclass(EPRSelector, BaseSelector)
        assert isclass(EPRSelector)

    def test_prepare_then_select(self, tiny_scoring_lm):
        scoring_lm, scoring_tok = tiny_scoring_lm
        items = [
            {"input": "what is 1+1", "output": "2"},
            {"input": "capital of france", "output": "paris"},
            {"input": "color of grass", "output": "green"},
            {"input": "what is 2+2", "output": "4"},
        ]
        pool = PoolMemory[dict]()
        for item in items:
            pool.add(item, polarity="pos")

        selector = EPRSelector(
            scoring_lm=scoring_lm,
            scoring_tokenizer=scoring_tok,
            base_encoder=TINY_BERT,
            candidate_set_size=2,
            k_pos=1,
            k_neg=1,
            train_epochs=1,
            batch_size=2,
        )
        selector.prepare(model=scoring_lm, tokenizer=scoring_tok, data=pool)
        assert selector.encoder is not None

        out = selector.select(pool.items, query="what is 3+3", k=2)
        assert len(out) == 2
        assert all(isinstance(item, dict) for item in out)

    def test_query_extracted_from_chat(self, tiny_scoring_lm):
        scoring_lm, scoring_tok = tiny_scoring_lm
        items = [{"input": f"q{i}", "output": f"a{i}"} for i in range(3)]
        pool = PoolMemory[dict]()
        for item in items:
            pool.add(item, polarity="pos")

        selector = EPRSelector(
            scoring_lm=scoring_lm,
            scoring_tokenizer=scoring_tok,
            base_encoder=TINY_BERT,
            candidate_set_size=2,
            k_pos=1,
            k_neg=1,
            train_epochs=1,
            batch_size=2,
        )
        selector.prepare(model=scoring_lm, tokenizer=scoring_tok, data=pool)

        chat = [
            {"role": "system", "content": "ignore me"},
            {"role": "user", "content": "the real query"},
        ]
        out = selector.select(pool.items, query=chat, k=1)
        assert len(out) == 1


class TestEPRWithFewShot:
    """End-to-end: EPRSelector slotted into FewShot and exercised through the pipeline."""

    def test_tensor_input_passes_query_to_selector(self, tiny_scoring_lm):
        """Regression: FewShot.adapt must pass the decoded prompt as `query` so query-conditioned
        selectors (EPR) rank against real text, not against the literal string "None"."""
        scoring_lm, scoring_tok = tiny_scoring_lm

        # deterministic, distinguishable items so the selector's output reflects the query
        items = [
            {"input": "alpha alpha alpha", "output": "A"},
            {"input": "beta beta beta", "output": "B"},
            {"input": "gamma gamma gamma", "output": "C"},
            {"input": "delta delta delta", "output": "D"},
        ]
        epr = EPRSelector(
            scoring_lm=scoring_lm,
            scoring_tokenizer=scoring_tok,
            base_encoder=TINY_BERT,
            candidate_set_size=2,
            k_pos=1,
            k_neg=1,
            train_epochs=1,
            batch_size=2,
        )
        fewshot = FewShot(
            positive_example_pool=items,
            k_positive=1,
            selector=epr,
        )
        pipeline = SteeringPipeline(controls=[fewshot], model=scoring_lm, tokenizer=scoring_tok)
        pipeline.steer()

        # capture the actual query passed into the selector during adapt
        captured: dict[str, Any] = {}
        original_select = epr.select

        def spy(items, query=None, k=1, context=None):
            captured["query"] = query
            return original_select(items, query=query, k=k, context=context)

        epr.select = spy

        prompt_text = "alpha alpha alpha"
        prompt_ids = scoring_tok(prompt_text, return_tensors="pt").input_ids
        fewshot.adapt(prompt_ids, runtime_kwargs={})

        # the selector must have received the decoded prompt text, not None
        assert captured.get("query") is not None
        assert prompt_text in str(captured["query"])

    def test_runs_through_pipeline(self, tiny_scoring_lm):
        scoring_lm, scoring_tok = tiny_scoring_lm
        examples = [
            {"input": f"q{i}", "output": f"a{i}"}
            for i in range(4)
        ]
        epr = EPRSelector(
            scoring_lm=scoring_lm,
            scoring_tokenizer=scoring_tok,
            base_encoder=TINY_BERT,
            candidate_set_size=2,
            k_pos=1,
            k_neg=1,
            train_epochs=1,
            batch_size=2,
        )
        fewshot = FewShot(
            positive_example_pool=examples,
            k_positive=1,
            selector=epr,
        )
        pipeline = SteeringPipeline(controls=[fewshot], model=scoring_lm, tokenizer=scoring_tok)
        pipeline.steer()

        # adapt_messages should now use the trained encoder for retrieval
        adapted = fewshot.adapt_messages([[{"role": "user", "content": "what about q2?"}]])
        assert adapted is not None
        # one system block (selected example) + the final user turn
        roles = [msg["role"] for msg in adapted[0]]
        assert roles == ["system", "user"]
        assert adapted[0][-1]["content"] == "what about q2?"
