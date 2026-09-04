"""Tests for the PRewrite input control."""
from __future__ import annotations

import pytest
import torch

from aisteer360.algorithms.core.steering_pipeline import SteeringPipeline
from aisteer360.algorithms.input_control.prewrite import PRewrite, PRewriteArgs
from aisteer360.evaluation.metrics.base import Metric


class _ConstantMetric(Metric):
    """Trivial metric that returns the same scalar regardless of input. Used for fast smoke tests."""
    def __init__(self, value: float = 0.5, **extras):
        super().__init__(**extras)
        self._value = value

    def compute(self, responses, prompts=None, **kwargs):
        return {"score": self._value}


class TestPRewriteArgs:
    def test_inference_minimal(self):
        args = PRewriteArgs(initial_instruction="be brief", strategy="inference")
        assert args.strategy == "inference"

    def test_search_requires_devset(self):
        with pytest.raises(ValueError, match="dev_set"):
            PRewriteArgs(initial_instruction="x", strategy="search", metric=_ConstantMetric())

    def test_search_requires_metric(self):
        with pytest.raises(ValueError, match="metric"):
            PRewriteArgs(initial_instruction="x", strategy="search", dev_set=[{"input": "a"}])

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="strategy"):
            PRewriteArgs(initial_instruction="x", strategy="bogus")

    def test_train_requires_explicit_rewriter(self):
        with pytest.raises(ValueError, match="rewriter_model"):
            PRewriteArgs(
                initial_instruction="x",
                strategy="inference",
                train_rewriter=True,
                reward_fn=lambda prompts, completions, **_: [0.0 for _ in completions],
            )

    def test_train_accepts_callable_reward_fn(self):
        args = PRewriteArgs(
            initial_instruction="x",
            strategy="inference",
            train_rewriter=True,
            rewriter_model_name_or_path="some/rewriter",
            reward_fn=lambda prompts, completions, **_: [0.0 for _ in completions],
        )
        assert args.reward_fn is not None

    def test_train_accepts_metric_devset_reward(self):
        args = PRewriteArgs(
            initial_instruction="x",
            strategy="inference",
            train_rewriter=True,
            rewriter_model_name_or_path="some/rewriter",
            metric=_ConstantMetric(),
            dev_set=[{"input": "a"}],
        )
        assert args.metric is not None and args.dev_set

    def test_train_requires_a_reward_source(self):
        with pytest.raises(ValueError, match="reward source"):
            PRewriteArgs(
                initial_instruction="x",
                strategy="inference",
                train_rewriter=True,
                rewriter_model_name_or_path="some/rewriter",
            )

    def test_at_most_one_rewriter_source(self):
        with pytest.raises(ValueError, match="at most one"):
            PRewriteArgs(
                initial_instruction="x",
                strategy="inference",
                rewriter_model_name_or_path="some/path",
                rewriter_model=object(),
            )

    def test_empty_initial_raises(self):
        with pytest.raises(ValueError, match="initial_instruction"):
            PRewriteArgs(initial_instruction="", strategy="inference")


class TestPRewriteInferenceStrategy:
    """End-to-end smoke test of PRewrite-I (greedy single rewrite)."""

    def test_runs_end_to_end(self, model_and_tokenizer, device: torch.device):
        base_model, tokenizer = model_and_tokenizer
        model = base_model.to(device)

        prewrite = PRewrite(
            initial_instruction="be helpful",
            strategy="inference",
            rewriter_gen_kwargs={"max_new_tokens": 4, "do_sample": False},
        )
        pipeline = SteeringPipeline(controls=[prewrite], model=model, tokenizer=tokenizer)
        pipeline.steer()

        assert prewrite.memory is not None
        assert isinstance(prewrite.memory["instruction"], str)
        assert len(prewrite.memory["instruction"]) > 0


class TestPRewriteSearchStrategy:
    def test_runs_end_to_end(self, model_and_tokenizer, device: torch.device):
        base_model, tokenizer = model_and_tokenizer
        model = base_model.to(device)

        prewrite = PRewrite(
            initial_instruction="be helpful",
            strategy="search",
            k_candidates=2,
            dev_set=[{"input": "hi"}, {"input": "world"}],
            metric=_ConstantMetric(value=1.0),
            rewriter_gen_kwargs={"max_new_tokens": 4, "do_sample": True, "temperature": 0.9},
            eval_gen_kwargs={"max_new_tokens": 2, "do_sample": False},
        )
        pipeline = SteeringPipeline(controls=[prewrite], model=model, tokenizer=tokenizer)
        pipeline.steer()

        assert prewrite.memory is not None
        assert isinstance(prewrite.memory["instruction"], str)


class TestPRewriteAdaptMessages:
    """`adapt_messages` should set/replace the system message with the chosen instruction."""

    def test_inserts_system_prompt(self, model_and_tokenizer, device: torch.device):
        base_model, tokenizer = model_and_tokenizer
        model = base_model.to(device)

        prewrite = PRewrite(
            initial_instruction="be helpful",
            strategy="inference",
            rewriter_gen_kwargs={"max_new_tokens": 2, "do_sample": False},
        )
        pipeline = SteeringPipeline(controls=[prewrite], model=model, tokenizer=tokenizer)
        pipeline.steer()

        adapted = prewrite.adapt_messages([[{"role": "user", "content": "?"}]])
        assert adapted is not None
        assert adapted[0][0]["role"] == "system"
        assert adapted[0][0]["content"] == prewrite.memory["instruction"]

    def test_replaces_existing_system(self, model_and_tokenizer, device: torch.device):
        base_model, tokenizer = model_and_tokenizer
        model = base_model.to(device)

        prewrite = PRewrite(
            initial_instruction="be helpful",
            strategy="inference",
            rewriter_gen_kwargs={"max_new_tokens": 2, "do_sample": False},
        )
        pipeline = SteeringPipeline(controls=[prewrite], model=model, tokenizer=tokenizer)
        pipeline.steer()

        chat = [
            {"role": "system", "content": "OLD"},
            {"role": "user", "content": "?"},
        ]
        adapted = prewrite.adapt_messages([chat])
        assert adapted[0][0]["content"] != "OLD"
        assert adapted[0][0]["content"] == prewrite.memory["instruction"]


class TestPRewriteSessionOnlySteer:
    """The steer phase completes with model=None against a session-only fake (ROLLOUTS)."""

    def test_steer_completes_with_model_none(self):
        from tests.utils.runtime_helpers import ScriptedSession
        from tests.utils.tiny_models import wordlevel_tokenizer

        tokenizer = wordlevel_tokenizer()

        def fake_generate(input_ids=None, attention_mask=None, **gen_kwargs):
            continuation = torch.full((input_ids.size(0), 2), 3, dtype=torch.long)
            return torch.cat([input_ids, continuation], dim=1)

        prewrite = PRewrite(
            initial_instruction="be helpful",
            strategy="inference",
            rewriter_gen_kwargs={"max_new_tokens": 2, "do_sample": False},
        )
        prewrite.steer(model=None, tokenizer=tokenizer, session=ScriptedSession(fake_generate, tokenizer=tokenizer))
        assert prewrite.memory is not None
        assert len(prewrite.memory["instruction"]) > 0
