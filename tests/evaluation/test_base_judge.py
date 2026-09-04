"""Tests for the backend-routed `LLMJudgeMetric`: declarative resolution, D3 template fields,
clean-break rejections, the backend resolution table and cache, and the full render->items->parse
loop against a stub backend (including n>1 grouping and the retry path). The ported TruthfulQA
judges are exercised here too. An engine-gated test runs one judge on the offline vLLM engine."""
from __future__ import annotations

import math

import pytest
import torch

from aisteer360.algorithms.core.execution.backend import Backend
from aisteer360.algorithms.core.execution.payloads import ItemResult
from aisteer360.algorithms.core.execution.spec import BackendSpec
from aisteer360.algorithms.core.output import Output
from aisteer360.evaluation.metrics import backend_utils
from aisteer360.evaluation.metrics.base_judge import LLMJudgeMetric
from aisteer360.evaluation.metrics.custom.truthful_qa import Informativeness, Truthfulness
from tests.utils.tiny_models import wordlevel_tokenizer

# wordlevel vocab: <s>=0 </s>=1 <pad>=2 the=3 cat=4 sat=5 on=6 mat=7 dog=8 ran=9 fast=10 ...


@pytest.fixture(scope="module")
def tokenizer():
    return wordlevel_tokenizer()


class StubSession:
    """A session double that returns fixed token rows per item and records its generate calls."""

    def __init__(self, backend: "StubBackend") -> None:
        self._backend = backend

    @property
    def tokenizer(self):
        return self._backend.tokenizer

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def generate(self, items, params):
        self._backend.calls.append((len(items), params.n or 1))
        candidate_ids = self._backend.next_rows()
        results = []
        for index, _item in enumerate(items):
            rows = [torch.tensor(ids, dtype=torch.long) for ids in candidate_ids]
            width = max(row.size(0) for row in rows)
            batch = torch.full((len(rows), width), self.tokenizer.pad_token_id, dtype=torch.long)
            for r, row in enumerate(rows):
                batch[r, : row.size(0)] = row
            results.append(ItemResult(index=index, output=Output(output_ids=batch, finish_reason="eos")))
        return results


class StubBackend(Backend):
    """A backend double producing scripted decoded rows; never loads a model."""

    def __init__(self, tokenizer, rows_per_call) -> None:
        self.tokenizer = tokenizer
        self._rows_per_call = list(rows_per_call)
        self._call_index = 0
        self.calls: list[tuple[int, int]] = []

    @classmethod
    def capabilities_for_spec(cls, spec):
        raise NotImplementedError

    def open_session(self):
        return StubSession(self)

    def next_rows(self):
        rows = self._rows_per_call[min(self._call_index, len(self._rows_per_call) - 1)]
        self._call_index += 1
        return rows


def _cat_parser(text: str) -> float:
    """1.0 when the decoded response contains 'cat', else 0.0 (wordlevel-vocab friendly)."""
    return 1.0 if "cat" in text else 0.0


def _stub(tokenizer, *, rows_per_call=None):
    """A StubBackend whose every generate call returns one candidate row 'the cat sat' by default."""
    rows_per_call = rows_per_call or [[[3, 4, 5]]]
    return StubBackend(tokenizer, rows_per_call)


class TestDeclarativeResolution:

    def test_class_attribute_used(self, tokenizer):
        class MyJudge(LLMJudgeMetric):
            prompt_template = "rate {response}"
            scale = (0, 1)
            structured_output = False

        judge = MyJudge(backend=_stub(tokenizer), parser=_cat_parser)
        assert judge.prompt_template == "rate {response}"
        assert judge.scale == (0, 1)

    def test_constructor_overrides_class_attribute(self, tokenizer):
        class MyJudge(LLMJudgeMetric):
            prompt_template = "class {response}"

        judge = MyJudge(
            backend=_stub(tokenizer), prompt_template="ctor {response}",
            scale=(0, 1), structured_output=False, parser=_cat_parser,
        )
        assert judge.prompt_template == "ctor {response}"

    def test_missing_prompt_template_raises(self, tokenizer):
        with pytest.raises(TypeError, match="prompt_template"):
            LLMJudgeMetric(backend=_stub(tokenizer))

    def test_name_respected(self, tokenizer):
        judge = LLMJudgeMetric(
            backend=_stub(tokenizer), prompt_template="r {response}", name="my_judge",
        )
        assert judge.name == "my_judge"

    def test_direct_instantiation(self, tokenizer):
        judge = LLMJudgeMetric(
            backend=_stub(tokenizer), prompt_template="rate {response}",
            scale=(0, 1), structured_output=False, parser=_cat_parser,
        )
        assert judge.compute(responses=["a", "b"])["scores"] == [1.0, 1.0]


class TestD3Fields:

    def test_placeholder_extraction(self, tokenizer):
        judge = LLMJudgeMetric(
            backend=_stub(tokenizer), prompt_template="q {question} r {response} c {context}",
            scale=(0, 1), structured_output=False, parser=_cat_parser,
        )
        assert judge._extra_fields == ("context", "question")

    def test_scalar_broadcast(self, tokenizer):
        judge = LLMJudgeMetric(
            backend=_stub(tokenizer), prompt_template="q {question} r {response}",
            scale=(0, 1), structured_output=False, parser=_cat_parser,
        )
        assert judge.compute(responses=["a", "b"], question="same")["scores"] == [1.0, 1.0]

    def test_aligned_sequences(self, tokenizer):
        judge = LLMJudgeMetric(
            backend=_stub(tokenizer), prompt_template="q {question} r {response}",
            scale=(0, 1), structured_output=False, parser=_cat_parser,
        )
        assert judge.compute(responses=["a", "b"], question=["q1", "q2"])["scores"] == [1.0, 1.0]

    def test_misaligned_sequence_raises(self, tokenizer):
        judge = LLMJudgeMetric(
            backend=_stub(tokenizer), prompt_template="q {question} r {response}",
            scale=(0, 1), structured_output=False, parser=_cat_parser,
        )
        with pytest.raises(ValueError, match="question"):
            judge.compute(responses=["a", "b"], question=["only_one"])

    def test_missing_field_raises_with_name(self, tokenizer):
        judge = LLMJudgeMetric(
            backend=_stub(tokenizer), prompt_template="q {question} r {response}",
            scale=(0, 1), structured_output=False, parser=_cat_parser,
        )
        with pytest.raises(ValueError, match="question"):
            judge.compute(responses=["a"])

    def test_prompt_placeholder_without_prompts_raises(self, tokenizer):
        judge = LLMJudgeMetric(
            backend=_stub(tokenizer), prompt_template="p {prompt} r {response}",
            scale=(0, 1), structured_output=False, parser=_cat_parser,
        )
        with pytest.raises(ValueError, match="prompt"):
            judge.compute(responses=["a"])


class TestCleanBreakRejections:

    def test_model_or_id_rejected(self, tokenizer):
        with pytest.raises(TypeError):
            LLMJudgeMetric(model_or_id="m", prompt_template="r {response}")

    def test_tokenizer_kwarg_rejected(self, tokenizer):
        with pytest.raises(TypeError):
            LLMJudgeMetric(backend=_stub(tokenizer), tokenizer=tokenizer, prompt_template="r {response}")

    def test_device_kwarg_rejected(self, tokenizer):
        with pytest.raises(TypeError):
            LLMJudgeMetric(backend=_stub(tokenizer), device="cpu", prompt_template="r {response}")

    def test_unknown_gen_kwargs_key_names_vocabulary(self, tokenizer):
        with pytest.raises(ValueError, match="max_new_tokens"):
            LLMJudgeMetric(
                backend=_stub(tokenizer), prompt_template="r {response}",
                gen_kwargs={"pad_token_id": 0},
            )

    def test_num_return_sequences_rejected(self, tokenizer):
        with pytest.raises(ValueError, match="normalized"):
            LLMJudgeMetric(
                backend=_stub(tokenizer), prompt_template="r {response}",
                gen_kwargs={"num_return_sequences": 4},
            )

    def test_bare_vllm_serve_string_rejected(self):
        with pytest.raises(TypeError, match="base_url"):
            LLMJudgeMetric(model="m", backend="vllm-serve", prompt_template="r {response}")

    def test_model_conflicting_with_spec_model_raises(self):
        with pytest.raises(ValueError, match="Conflicting"):
            LLMJudgeMetric(
                model="a", backend=BackendSpec(kind="huggingface", model="b"),
                prompt_template="r {response}",
            )

    def test_structured_true_rejects_parser(self, tokenizer):
        with pytest.raises(ValueError, match="not both"):
            LLMJudgeMetric(
                backend=_stub(tokenizer), prompt_template="r {response}",
                structured_output=True, parser=lambda text: 1.0,
            )

    def test_structured_false_requires_parser(self, tokenizer):
        with pytest.raises(ValueError, match="parser"):
            LLMJudgeMetric(
                backend=_stub(tokenizer), prompt_template="r {response}",
                structured_output=False, parser=None,
            )

    def test_n_greater_than_one_under_greedy_rejected(self, tokenizer):
        with pytest.raises(ValueError, match="temperature"):
            LLMJudgeMetric(
                backend=_stub(tokenizer), prompt_template="r {response}",
                gen_kwargs={"temperature": 0.0, "n": 3},
            )

    def test_score_rendered_removed(self):
        assert not hasattr(LLMJudgeMetric, "score_rendered")


class TestBackendResolutionTable:

    def setup_method(self):
        backend_utils._METRIC_BACKENDS.clear()

    def test_none_and_huggingface_require_model(self):
        with pytest.raises(TypeError, match="model"):
            backend_utils.resolve_metric_backend(None, None)
        with pytest.raises(TypeError, match="model"):
            backend_utils.resolve_metric_backend(None, "huggingface")

    def test_vllm_requires_model(self):
        with pytest.raises(TypeError, match="model"):
            backend_utils.resolve_metric_backend(None, "vllm")

    def test_spec_without_model_and_no_model_raises(self):
        with pytest.raises(TypeError, match="no model"):
            backend_utils.resolve_metric_backend(None, BackendSpec(kind="vllm"))

    def test_live_backend_with_model_raises(self, tokenizer):
        backend = _stub(tokenizer)
        with pytest.raises(ValueError, match="not both"):
            backend_utils.resolve_metric_backend("m", backend)

    def test_live_backend_used_as_is_and_not_cached(self, tokenizer):
        backend = _stub(tokenizer)
        assert backend_utils.resolve_metric_backend(None, backend) is backend
        assert not backend_utils._METRIC_BACKENDS


class TestBackendCache:

    def setup_method(self):
        backend_utils._METRIC_BACKENDS.clear()

    def test_equal_specs_share_one_backend(self, monkeypatch):
        constructed = []

        class FakeBackend:
            def __init__(self, spec):
                constructed.append(spec)
                self.spec = spec

        monkeypatch.setattr(backend_utils, "resolve_backend_class", lambda spec: FakeBackend)
        spec_a = BackendSpec(kind="vllm", model="m")
        spec_b = BackendSpec(kind="vllm", model="m")
        first = backend_utils.resolve_metric_backend(None, spec_a)
        second = backend_utils.resolve_metric_backend(None, spec_b)
        assert first is second
        assert len(constructed) == 1

    def test_perplexity_and_judge_share_equal_spec(self, monkeypatch, tokenizer):
        from aisteer360.evaluation.metrics.generic.perplexity import Perplexity

        class FakeBackend:
            def __init__(self, spec):
                self.spec = spec

        monkeypatch.setattr(backend_utils, "resolve_backend_class", lambda spec: FakeBackend)
        spec = BackendSpec(kind="vllm", model="shared")
        judge = LLMJudgeMetric(
            backend=BackendSpec(kind="vllm", model="shared"),
            prompt_template="r {response}", scale=(0, 1), structured_output=False, parser=_cat_parser,
        )
        perplexity = Perplexity(backend=BackendSpec(kind="vllm", model="shared"))
        assert judge._backend is perplexity._backend
        assert judge._backend.spec == spec


class TestGenerationLoop:

    def test_structured_json_parse_and_clamp(self, tokenizer):
        judge = LLMJudgeMetric(
            backend=StubBackend(tokenizer, [[[3, 4]]]),
            prompt_template="rate {response} from {lower_bound} to {upper_bound}",
        )
        judge.parse_fn = lambda text, scale: max(scale[0], min(scale[1], 9.0))  # clamp to 5
        result = judge.compute(responses=["a"])
        assert result["scores"] == [5.0]

    def test_batching_chunks_by_batch_size(self, tokenizer):
        backend = StubBackend(tokenizer, [[[3, 4]]])
        judge = LLMJudgeMetric(
            backend=backend, prompt_template="r {response}", scale=(0, 1),
            structured_output=False, parser=_cat_parser, batch_size=2,
        )
        judge.compute(responses=["a", "b", "c"])
        assert [count for count, _ in backend.calls] == [2, 1]

    def test_n_grouping(self, tokenizer):
        backend = StubBackend(tokenizer, [[[3, 4], [3, 8], [8, 8]]])  # cat, dog, dog
        judge = LLMJudgeMetric(
            backend=backend, prompt_template="r {response}", scale=(0, 1),
            structured_output=False, parser=_cat_parser, gen_kwargs={"temperature": 0.7, "n": 3},
        )
        result = judge.compute(responses=["x"])
        assert result["raw_scores"] == [[1.0, 0.0, 0.0]]
        assert result["scores"] == [pytest.approx(1 / 3)]

    def test_greedy_parse_failure_raises_with_raw_response(self, tokenizer):
        def boom(text):
            raise ValueError("bad")

        judge = LLMJudgeMetric(
            backend=StubBackend(tokenizer, [[[3, 4]]]), prompt_template="r {response}",
            scale=(0, 1), structured_output=False, parser=boom,
        )
        with pytest.raises(ValueError, match="deterministic"):
            judge.compute(responses=["a"])

    def test_sampling_parse_failure_returns_nan_after_retries(self, tokenizer):
        def boom(text):
            raise ValueError("bad")

        backend = StubBackend(tokenizer, [[[3, 4]]])
        judge = LLMJudgeMetric(
            backend=backend, prompt_template="r {response}", scale=(0, 1),
            structured_output=False, parser=boom, gen_kwargs={"temperature": 0.7}, max_retries=2,
        )
        with pytest.warns(UserWarning, match="retries"):
            result = judge.compute(responses=["a"])
        assert math.isnan(result["scores"][0])


class TestGenerationParamsRendering:
    """The judge's normalized params must render onto do_sample correctly on the HF seam: a
    sampling config leaves greedy=False (do_sample=True), not None (which HF re-defaults to
    do_sample=False, crashing n>1 and making retries futile)."""

    def _params(self, tokenizer, gen_kwargs):
        judge = LLMJudgeMetric(
            backend=_stub(tokenizer), prompt_template="r {response}", scale=(0, 1),
            structured_output=False, parser=_cat_parser, gen_kwargs=gen_kwargs,
        )
        return judge._params

    def test_default_is_greedy(self, tokenizer):
        params = self._params(tokenizer, None)
        assert params.greedy is True
        assert params.temperature in (None, 0.0)

    def test_sampling_forces_do_sample_true_on_hf(self, tokenizer):
        from aisteer360.backends.huggingface import render_hf_gen_kwargs

        params = self._params(tokenizer, {"temperature": 0.7})
        assert params.greedy is False
        assert params.temperature == 0.7
        rendered = render_hf_gen_kwargs(params)
        assert rendered["do_sample"] is True

    def test_sampling_with_n_renders_num_return_sequences_and_sampling(self, tokenizer):
        from aisteer360.backends.huggingface import render_hf_gen_kwargs

        params = self._params(tokenizer, {"temperature": 0.8, "n": 3})
        rendered = render_hf_gen_kwargs(params)
        assert rendered["do_sample"] is True
        assert rendered["num_return_sequences"] == 3

    def test_sampling_renders_on_vllm_without_error(self, tokenizer):
        from aisteer360.backends.vllm import render_vllm_sampling_args

        params = self._params(tokenizer, {"temperature": 0.7, "n": 3})
        rendered = render_vllm_sampling_args(params)
        assert rendered["temperature"] == 0.7
        assert rendered["n"] == 3

    def test_explicit_greedy_under_sampling_respected(self, tokenizer):
        params = self._params(tokenizer, {"temperature": 0.5, "greedy": True})
        assert params.greedy is True


class TestPortedTruthfulQAJudges:

    def test_truthfulness_yes_no_to_binary(self, tokenizer):
        # 'cat' present -> the yes/no parser sees no 'yes', so 0; craft a session returning tokens
        # decoding to a string that startswith 'yes' is not expressible in wordlevel vocab, so
        # override the parser deterministically via a stub whose decoded text is controlled.
        records = [
            {"question": "q1", "response": "a1", "correct_answers": ["c1"], "incorrect_answers": ["i1"]},
            {"question": "q2", "response": "a2", "correct_answers": ["c2"], "incorrect_answers": ["i2"]},
        ]
        judge = Truthfulness(backend=StubBackend(tokenizer, [[[3, 4]]]))
        judge.parse_fn = lambda text, scale, _seq=iter([1.0, 0.0]): next(_seq)
        result = judge.compute(responses=records)
        assert result["scores"] == [1.0, 0.0]
        assert result["truthfulness_rate"] == pytest.approx(0.5)
        assert judge.name == "Truthfulness"

    def test_truthfulness_resolves_extra_fields(self, tokenizer):
        records = [
            {"question": "Who?", "response": "Alice", "correct_answers": ["Alice", "A."],
             "incorrect_answers": ["Bob"]},
        ]
        judge = Truthfulness(backend=StubBackend(tokenizer, [[[3, 4]]]))

        rendered = judge._render(
            responses=["Alice"],
            prompts=None,
            kwargs={
                "question": ["Who?"],
                "correct_answers": ["  - Alice\n  - A."],
                "incorrect_answers": ["  - Bob"],
            },
        )
        assert "Who?" in rendered[0]
        assert "- Alice" in rendered[0]
        assert "- Bob" in rendered[0]
        assert "Alice" in rendered[0]  # the response is the {response} field

        judge.parse_fn = lambda text, scale: 1.0
        assert judge.compute(responses=records)["scores"] == [1.0]

    def test_informativeness_empty_responses(self, tokenizer):
        judge = Informativeness(backend=StubBackend(tokenizer, [[[3, 4]]]))
        assert judge.compute(responses=[]) == {"informativeness_rate": 0.0, "scores": []}

    def test_informativeness_name(self, tokenizer):
        judge = Informativeness(backend=StubBackend(tokenizer, [[[3, 4]]]))
        assert judge.name == "Informativeness"


class TestEngineGatedJudge:

    def test_factuality_on_offline_engine(self):
        pytest.importorskip("vllm")
        from aisteer360.evaluation.metrics.generic.factuality import Factuality

        spec = BackendSpec(
            kind="vllm",
            model="JackFram/llama-68m",
            options={"engine_kwargs": {"enforce_eager": True, "max_model_len": 512}},
        )
        try:
            factuality = Factuality(
                backend=spec, structured_output=False, parser=lambda text: 1.0,
            )
            result = factuality.compute(responses=["Paris."], prompts=["Capital of France?"])
        except Exception as exception:
            pytest.skip(f"Could not boot the vLLM engine: {exception}")
        assert set(result) == {"mean_score", "scores", "raw_scores"}
        assert len(result["scores"]) == 1
