"""Tests for the GRPO TRL wrapper and the PRewrite metric-reward adapter.

The args-validation and reward-adapter tests are model-free and always run. The end-to-end training
smoke test is gated behind `RUN_GRPO_SMOKE=1` (it loads a tiny model and runs one GRPO step).
"""
from __future__ import annotations

import os

import pytest

from aisteer360.algorithms.input_control.prewrite.utils.reward import _completion_text, make_metric_reward_func
from aisteer360.algorithms.structural_control.wrappers.trl.grpotrainer import GRPO, GRPOArgs


def _reward_stub(prompts, completions, **kwargs):
    return [float(len(c)) for c in completions]


class TestGRPOArgs:
    def test_requires_reward_funcs(self):
        with pytest.raises(ValueError, match="reward_funcs"):
            GRPOArgs()

    def test_rejects_empty_reward_funcs(self):
        with pytest.raises(ValueError, match="reward_funcs"):
            GRPOArgs(reward_funcs=[])

    def test_num_generations_must_be_at_least_two(self):
        with pytest.raises(ValueError, match="num_generations"):
            GRPOArgs(reward_funcs=[_reward_stub], num_generations=1, per_device_train_batch_size=2)

    def test_batch_must_be_divisible_by_num_generations(self):
        with pytest.raises(ValueError, match="divisible"):
            GRPOArgs(reward_funcs=[_reward_stub], num_generations=3, per_device_train_batch_size=8)

    def test_valid_args_populate_training_args(self):
        args = GRPOArgs(
            reward_funcs=[_reward_stub],
            num_generations=2,
            per_device_train_batch_size=4,
            max_completion_length=16,
            beta=0.0,
            temperature=0.7,
        )
        assert args.training_args["num_generations"] == 2
        assert args.training_args["max_completion_length"] == 16
        assert args.training_args["beta"] == 0.0
        assert args.training_args["temperature"] == 0.7
        # GRPO has no eval split by default
        assert args.training_args["load_best_model_at_end"] is False

    def test_single_callable_reward_funcs_allowed(self):
        args = GRPOArgs(reward_funcs=_reward_stub, num_generations=2, per_device_train_batch_size=2)
        assert args.reward_funcs is _reward_stub


class TestCompletionText:
    def test_plain_string(self):
        assert _completion_text("hello") == "hello"

    def test_conversational(self):
        completion = [{"role": "assistant", "content": "rewrite me"}]
        assert _completion_text(completion) == "rewrite me"

    def test_conversational_missing_content(self):
        assert _completion_text([{"role": "assistant"}]) == ""

    def test_fallback(self):
        assert _completion_text(123) == "123"


class _StubScorer:
    """Duck-typed stand-in for `TaskEvaluationScorer`; records the prompts it scores."""

    def __init__(self, fn):
        self._fn = fn
        self.calls: list[list[str]] = []

    def score(self, prompts):
        prompts = list(prompts)
        self.calls.append(prompts)
        return self._fn(prompts)


class TestMakeMetricRewardFunc:
    def test_aligns_scores_to_string_completions(self):
        scorer = _StubScorer(lambda ps: [float(len(p)) for p in ps])
        reward_func = make_metric_reward_func(scorer)
        rewards = reward_func(prompts=["seed", "seed"], completions=["aa", "bbb"])
        assert rewards == [2.0, 3.0]

    def test_handles_conversational_completions(self):
        scorer = _StubScorer(lambda ps: [float(len(p)) for p in ps])
        reward_func = make_metric_reward_func(scorer)
        completions = [
            [{"role": "assistant", "content": "aa"}],
            [{"role": "assistant", "content": "bbbb"}],
        ]
        rewards = reward_func(prompts=["s", "s"], completions=completions)
        assert rewards == [2.0, 4.0]

    def test_deduplicates_identical_rewrites(self):
        scorer = _StubScorer(lambda ps: [float(len(p)) for p in ps])
        reward_func = make_metric_reward_func(scorer)
        rewards = reward_func(prompts=["s"] * 3, completions=["x", "x", "yy"])
        # one float per completion, but the scorer is only asked about the unique rewrites
        assert rewards == [1.0, 1.0, 2.0]
        assert scorer.calls == [["x", "yy"]]

    def test_parse_fn_applied_before_scoring(self):
        scorer = _StubScorer(lambda ps: [float(len(p)) for p in ps])
        reward_func = make_metric_reward_func(scorer, parse_fn=lambda _t: ["PARSED"])
        rewards = reward_func(prompts=["s"], completions=["some long wrapped prose"])
        assert rewards == [float(len("PARSED"))]
        assert scorer.calls == [["PARSED"]]

    def test_parse_fn_empty_falls_back_to_raw(self):
        scorer = _StubScorer(lambda ps: [float(len(p)) for p in ps])
        reward_func = make_metric_reward_func(scorer, parse_fn=lambda _t: [])
        rewards = reward_func(prompts=["s"], completions=["  raw  "])
        assert rewards == [float(len("raw"))]
        assert scorer.calls == [["raw"]]

    def test_empty_completions(self):
        scorer = _StubScorer(lambda ps: [float(len(p)) for p in ps])
        reward_func = make_metric_reward_func(scorer)
        assert reward_func(prompts=[], completions=[]) == []
        assert scorer.calls == []


@pytest.mark.skipif(
    os.environ.get("RUN_GRPO_SMOKE") != "1",
    reason="set RUN_GRPO_SMOKE=1 to run the GRPO training smoke test (loads a tiny model)",
)
class TestGRPOWrapperSmoke:
    def test_trains_one_step_and_generates(self):
        import torch
        from datasets import Dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_id = "hf-internal-testing/tiny-random-LlamaForCausalLM"
        try:
            model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        except Exception as exc:
            pytest.skip(f"could not load {model_id}: {exc}")

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        args = GRPOArgs(
            train_dataset=Dataset.from_dict({"prompt": ["Rewrite: be brief", "Rewrite: be clear"]}),
            reward_funcs=[_reward_stub],
            num_generations=2,
            per_device_train_batch_size=2,
            max_completion_length=4,
            max_prompt_length=32,
            beta=0.0,
            training_args={"max_steps": 1, "logging_steps": 1},
        )
        trained = GRPO(args).steer(model, tokenizer)

        device = next(trained.parameters()).device
        input_ids = tokenizer("hello", return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            out = trained.generate(input_ids, max_new_tokens=3)
        assert out.shape[1] >= input_ids.shape[1]
