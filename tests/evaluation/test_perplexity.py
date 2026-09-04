"""Tests for the scoring-seam `Perplexity`: exact perplexities against a stub backend, the
length-bucketing that keeps input order under more than one `score` call, both conditioning modes,
degenerate rows producing nan plus a warning, clean-break rejections, and cache sharing with a
judge. An optional engine-gated pin checks HF/vLLM perplexity parity."""
from __future__ import annotations

import math

import pytest
import torch

from aisteer360.algorithms.core.execution.backend import Backend
from aisteer360.algorithms.core.execution.spec import BackendSpec
from aisteer360.evaluation.metrics import backend_utils
from aisteer360.evaluation.metrics.generic.perplexity import Perplexity
from tests.utils.tiny_models import wordlevel_tokenizer

# wordlevel vocab: <s>=0 </s>=1 <pad>=2 the=3 cat=4 sat=5 on=6 mat=7 dog=8 ran=9 fast=10 ...


@pytest.fixture(scope="module")
def tokenizer():
    return wordlevel_tokenizer()


class ScoreSession:
    """A session double whose `score` returns fixed per-token log-probs and records ref lengths."""

    def __init__(self, backend: "ScoreBackend") -> None:
        self._backend = backend

    @property
    def tokenizer(self):
        return self._backend.tokenizer

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def score(self, items, params):
        assert params.extra == {}, "Perplexity must pass empty GenerationParams to score()."
        ref_lens = {item.ref_output_ids.shape[-1] for item in items}
        assert len(ref_lens) == 1, "score() receives one reference length per call."
        self._backend.score_calls.append(sorted(item.ref_output_ids.tolist() for item in items))
        rows = []
        for item in items:
            ref = item.ref_output_ids
            if ref.ndim == 1:
                ref = ref.unsqueeze(0)
            rows.append(self._backend.logprob_fn(ref[0]))
        return torch.stack(rows, dim=0)


class ScoreBackend(Backend):
    """A backend double whose scoring log-probs are a deterministic function of the ref tokens."""

    def __init__(self, tokenizer, logprob_fn) -> None:
        self.tokenizer = tokenizer
        self.logprob_fn = logprob_fn
        self.score_calls: list = []

    @classmethod
    def capabilities_for_spec(cls, spec):
        raise NotImplementedError

    def open_session(self):
        return ScoreSession(self)


def _constant_logprob(value: float):
    """A log-prob function assigning the same `value` to every reference token."""
    return lambda ref: torch.full((ref.shape[-1],), float(value), dtype=torch.float32)


def _per_token_logprob(ref: torch.Tensor) -> torch.Tensor:
    """A deterministic per-token log-prob: -(token_id / 10) for each reference token."""
    return -(ref.to(torch.float32) / 10.0)


class TestExactPerplexity:

    def test_constant_logprob_gives_exp_neg_value(self, tokenizer):
        backend = ScoreBackend(tokenizer, _constant_logprob(-0.5))
        perplexity = Perplexity(backend=backend, add_bos=True)
        result = perplexity.compute(responses=["the cat sat"])
        assert result["perplexities"][0] == pytest.approx(math.exp(0.5))
        assert result["mean_perplexity"] == pytest.approx(math.exp(0.5))

    def test_per_token_logprob(self, tokenizer):
        backend = ScoreBackend(tokenizer, _per_token_logprob)
        perplexity = Perplexity(backend=backend, add_bos=True)
        # "the cat" -> tokens [3, 4]; add_bos scores both; logprobs -0.3, -0.4; mean -0.35
        result = perplexity.compute(responses=["the cat"])
        assert result["perplexities"][0] == pytest.approx(math.exp(0.35))


class TestConditioningModes:

    def test_bos_mode_scores_all_tokens(self, tokenizer):
        backend = ScoreBackend(tokenizer, _constant_logprob(-1.0))
        perplexity = Perplexity(backend=backend, add_bos=True)
        perplexity.compute(responses=["the cat sat"])
        # one score call; the reference is all three response tokens [3, 4, 5]
        assert backend.score_calls == [[[3, 4, 5]]]

    def test_no_bos_mode_scores_tail(self, tokenizer):
        backend = ScoreBackend(tokenizer, _constant_logprob(-1.0))
        perplexity = Perplexity(backend=backend, add_bos=False)
        perplexity.compute(responses=["the cat sat"])
        # first token is conditioning context; reference is [4, 5]
        assert backend.score_calls == [[[4, 5]]]


class TestLengthBucketing:

    def test_mixed_lengths_multiple_calls_input_order_preserved(self, tokenizer):
        backend = ScoreBackend(tokenizer, _per_token_logprob)
        perplexity = Perplexity(backend=backend, add_bos=True, batch_size=8)
        responses = ["the cat", "the cat sat", "dog ran"]  # lengths 2, 3, 2
        result = perplexity.compute(responses=responses)
        # two distinct reference lengths -> at least two score calls
        assert len(backend.score_calls) == 2
        # per-response perplexities computed from _per_token_logprob, in input order
        expected = []
        for text in responses:
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            logs = [-(i / 10.0) for i in ids]
            expected.append(math.exp(-sum(logs) / len(logs)))
        assert result["perplexities"] == pytest.approx(expected)

    def test_batch_size_chunks_within_length_group(self, tokenizer):
        backend = ScoreBackend(tokenizer, _constant_logprob(-1.0))
        perplexity = Perplexity(backend=backend, add_bos=True, batch_size=2)
        # four same-length responses -> two chunks of two
        perplexity.compute(responses=["the cat", "dog ran", "cat sat", "on mat"])
        assert len(backend.score_calls) == 2
        assert all(len(call) == 2 for call in backend.score_calls)


class TestDegenerateRows:

    def test_empty_response_is_nan_with_warning(self, tokenizer):
        backend = ScoreBackend(tokenizer, _constant_logprob(-1.0))
        perplexity = Perplexity(backend=backend, add_bos=True)
        with pytest.warns(UserWarning, match="too short"):
            result = perplexity.compute(responses=["", "the cat"])
        assert math.isnan(result["perplexities"][0])
        assert not math.isnan(result["perplexities"][1])
        # mean excludes the nan row
        assert result["mean_perplexity"] == pytest.approx(result["perplexities"][1])

    def test_single_token_no_bos_is_nan(self, tokenizer):
        backend = ScoreBackend(tokenizer, _constant_logprob(-1.0))
        perplexity = Perplexity(backend=backend, add_bos=False)
        with pytest.warns(UserWarning, match="too short"):
            result = perplexity.compute(responses=["cat"])
        assert math.isnan(result["perplexities"][0])
        assert math.isnan(result["mean_perplexity"])

    def test_empty_responses_list(self, tokenizer):
        backend = ScoreBackend(tokenizer, _constant_logprob(-1.0))
        perplexity = Perplexity(backend=backend, add_bos=True)
        assert perplexity.compute(responses=[]) == {"mean_perplexity": 0.0, "perplexities": []}


class TestCleanBreakRejections:

    def test_model_or_id_rejected(self, tokenizer):
        with pytest.raises(TypeError):
            Perplexity(model_or_id="m")

    def test_tokenizer_kwarg_rejected(self, tokenizer):
        backend = ScoreBackend(tokenizer, _constant_logprob(-1.0))
        with pytest.raises(TypeError):
            Perplexity(backend=backend, tokenizer=tokenizer)

    def test_device_kwarg_rejected(self, tokenizer):
        backend = ScoreBackend(tokenizer, _constant_logprob(-1.0))
        with pytest.raises(TypeError):
            Perplexity(backend=backend, device="cpu")


class TestCacheSharingWithJudge:

    def setup_method(self):
        backend_utils._METRIC_BACKENDS.clear()

    def test_perplexity_and_judge_share_equal_spec(self, monkeypatch):
        from aisteer360.evaluation.metrics.base_judge import LLMJudgeMetric

        class FakeBackend:
            def __init__(self, spec):
                self.spec = spec

        monkeypatch.setattr(backend_utils, "resolve_backend_class", lambda spec: FakeBackend)
        perplexity = Perplexity(backend=BackendSpec(kind="vllm", model="shared"))
        judge = LLMJudgeMetric(
            backend=BackendSpec(kind="vllm", model="shared"),
            prompt_template="r {response}", scale=(0, 1), structured_output=False,
            parser=lambda text: 0.0,
        )
        assert perplexity._backend is judge._backend


class TestEngineGatedParity:
    """Optional HF/vLLM parity pin; skips cleanly without `vllm` or a bootable engine."""

    def test_hf_vllm_perplexity_parity(self):
        pytest.importorskip("vllm")
        from aisteer360.backends.huggingface import HFBackend

        model_id = "JackFram/llama-68m"
        responses = ["The quick brown fox jumps."]
        hf_spec = BackendSpec(kind="huggingface", model=model_id)
        vllm_spec = BackendSpec(
            kind="vllm", model=model_id,
            options={"engine_kwargs": {"enforce_eager": True, "max_model_len": 512}},
        )
        try:
            hf_backend = HFBackend(hf_spec)
            hf_ppl = Perplexity(backend=hf_backend).compute(responses=responses)
        except Exception as exception:
            pytest.skip(f"Could not build the HF backend: {exception}")
        try:
            from aisteer360.backends.vllm import VLLMBackend

            vllm_backend = VLLMBackend(vllm_spec)
            vllm_ppl = Perplexity(backend=vllm_backend).compute(responses=responses)
        except Exception as exception:
            pytest.skip(f"Could not boot the vLLM engine: {exception}")
        assert vllm_ppl["perplexities"][0] == pytest.approx(hf_ppl["perplexities"][0], rel=0.05)
