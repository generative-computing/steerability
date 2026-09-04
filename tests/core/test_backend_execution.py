"""Tests for P1 backend execution: strict parameter rendering, the fan-out machinery, the
pinned stop-string and finish-reason semantics, session-routed pipeline inference, driver
sessions, portable requirements, and structural artifact derivation."""
import dataclasses

import pytest
import torch

from aisteer360.algorithms.core.execution import (
    BackendSpec,
    Capability,
    CheckpointArtifact,
    GenerationItem,
    GenerationParams,
    LoRAArtifact,
    PartialBatchError,
    PreparedPrompt,
    TransportError,
    derive_item_seed,
    merge_lowered_params,
    run_bounded,
    with_transport_retries,
)
from aisteer360.algorithms.core.execution.access import ModelAccess
from aisteer360.algorithms.core.execution.session_utils import session_generate
from aisteer360.algorithms.core.output import Output, infer_finish_reasons, truncate_at_stop_strings
from aisteer360.algorithms.core.steering_pipeline import SteeringPipeline
from aisteer360.algorithms.input_control.base import InputControl
from aisteer360.algorithms.input_control.gepa.control import GEPA
from aisteer360.algorithms.input_control.prewrite.control import PRewrite
from aisteer360.algorithms.output_control.base import DecodingDriver, stack_generate_kwargs
from aisteer360.algorithms.output_control.best_of_n.control import BestOfN
from aisteer360.algorithms.output_control.budget_forcing.control import BudgetForcing
from aisteer360.algorithms.output_control.deal.control import DeAL
from aisteer360.algorithms.output_control.phased_decoding.control import PhasedDecoding
from aisteer360.algorithms.output_control.search_decoding.control import SearchDecoding
from aisteer360.algorithms.output_control.stopping_rules.control import StoppingRules
from aisteer360.algorithms.state_control.activation_adapter.control import ActivationAdapter
from aisteer360.algorithms.state_control.base import StateControl
from aisteer360.algorithms.state_control.common.runtime import TransformHookRuntime
from aisteer360.algorithms.structural_control.base import StructuralControl
from aisteer360.backends.huggingface import HFBackend
from aisteer360.backends.vllm import extract_ref_logprobs, map_vllm_finish_reason, render_vllm_sampling_args
from tests.utils.runtime_helpers import RecordingTransform
from tests.utils.tiny_models import tiny_llama, wordlevel_tokenizer

HF_SPEC = BackendSpec(kind="huggingface")
VLLM_SPEC = BackendSpec(kind="vllm", model="m")


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return tiny_llama(num_layers=2, hidden=16, heads=2)


@pytest.fixture(scope="module")
def tokenizer():
    return wordlevel_tokenizer()


@pytest.fixture()
def backend(model, tokenizer):
    return HFBackend.adopt(HF_SPEC, lambda: model, lambda: tokenizer)


def _pipeline(model, tokenizer, controls=()):
    pipeline = SteeringPipeline(controls=list(controls), model=model, tokenizer=tokenizer)
    pipeline.steer()
    return pipeline


class _ForceSequence:
    """Force a fixed token sequence past the prompt (wordlevel vocab: the=3 cat=4 sat=5 on=6)."""

    def __init__(self, prompt_len, sequence):
        self._prompt_len = prompt_len
        self._sequence = sequence

    def __call__(self, input_ids, scores):
        step = input_ids.shape[1] - self._prompt_len
        forced = torch.full_like(scores, -1e9)
        forced[:, self._sequence[step % len(self._sequence)]] = 0.0
        return forced


class TestGenerationParamsStops:

    def test_from_gen_kwargs_captures_stop_fields(self):
        params = GenerationParams.from_gen_kwargs(
            stop_strings="END", stop_token_ids=[5, 7], max_new_tokens=4,
        )
        assert params.stop_strings == ("END",)
        assert params.stop_token_ids == (5, 7)
        assert params.extra == {}

    def test_to_gen_kwargs_round_trips(self):
        params = GenerationParams.from_gen_kwargs(
            max_new_tokens=5, do_sample=False, num_return_sequences=2,
            stop_strings=["a"], foo="bar",
        )
        assert GenerationParams.from_gen_kwargs(**params.to_gen_kwargs()) == params

    def test_merge_lowered_unions_stops_and_tightens_bounds(self):
        params = GenerationParams(stop_strings=("a",), max_new_tokens=10)
        merged = merge_lowered_params(params, {
            "stop_strings": ("b", "a"), "stop_token_ids": (3,),
            "max_new_tokens": 4, "min_new_tokens": 2,
        })
        assert merged.stop_strings == ("a", "b")
        assert merged.stop_token_ids == (3,)
        assert merged.max_new_tokens == 4
        assert merged.min_new_tokens == 2

    def test_merge_lowered_never_relaxes_max(self):
        params = GenerationParams(max_new_tokens=3)
        merged = merge_lowered_params(params, {"max_new_tokens": 100})
        assert merged.max_new_tokens == 3

    def test_merge_lowered_rejects_unknown_keys(self):
        with pytest.raises(ValueError, match="temperature"):
            merge_lowered_params(GenerationParams(), {"temperature": 0.5})


class TestVLLMRendering:

    def test_table_maps_normalized_fields(self):
        args = render_vllm_sampling_args(GenerationParams(
            max_new_tokens=8, min_new_tokens=2, temperature=0.7, top_p=0.9, top_k=40,
            n=3, repetition_penalty=1.1,
        ))
        assert args == {
            "max_tokens": 8, "min_tokens": 2, "temperature": 0.7, "top_p": 0.9,
            "top_k": 40, "n": 3, "repetition_penalty": 1.1,
        }

    def test_greedy_renders_as_zero_temperature(self):
        assert render_vllm_sampling_args(GenerationParams(greedy=True)) == {"temperature": 0.0}

    def test_stop_strings_request_inclusion(self):
        args = render_vllm_sampling_args(GenerationParams(stop_strings=("END",), stop_token_ids=(9,)))
        assert args["stop"] == ["END"]
        assert args["include_stop_str_in_output"] is True
        assert args["stop_token_ids"] == [9]

    def test_unmapped_key_raises_with_name(self):
        with pytest.raises(ValueError, match="num_beams"):
            render_vllm_sampling_args(GenerationParams(extra={"num_beams": 4}))

    def test_greedy_with_nonzero_temperature_rejected(self):
        with pytest.raises(ValueError, match="greedy"):
            render_vllm_sampling_args(GenerationParams(greedy=True, temperature=0.8))

    def test_seed_never_rendered_by_table(self):
        assert "seed" not in render_vllm_sampling_args(GenerationParams(seed=7))


class TestFinishReasonMapping:

    def test_vllm_stop_reason_disambiguates_eos_from_stop(self):
        assert map_vllm_finish_reason("stop", None) == "eos"
        assert map_vllm_finish_reason("stop", "END") == "stop"
        assert map_vllm_finish_reason("stop", 7) == "stop"
        assert map_vllm_finish_reason("length", None) == "length"
        assert map_vllm_finish_reason("abort", None) is None


class TestExtractRefLogprobs:

    def test_offline_shape(self):
        class _Record:
            def __init__(self, logprob):
                self.logprob = logprob

        prompt_logprobs = [None, {3: _Record(-0.5)}, {5: _Record(-1.0)}, {6: _Record(-2.0)}]
        assert extract_ref_logprobs(prompt_logprobs, [5, 6]) == [-1.0, -2.0]

    def test_serve_json_shape(self):
        prompt_logprobs = [None, {"3": {"logprob": -0.5}}, {"5": {"logprob": -1.5}}]
        assert extract_ref_logprobs(prompt_logprobs, [5]) == [-1.5]

    def test_missing_token_entry_rejected(self):
        with pytest.raises(ValueError, match="missing"):
            extract_ref_logprobs([None, {3: object()}], [5])


class TestFanout:

    def test_seed_derivation_vectors(self):
        assert derive_item_seed(42, "generate-0", 0) == 9171175973360618330
        assert derive_item_seed(42, "generate-0", 1) == 7488875411253355286
        assert derive_item_seed(42, "generate-1", 0) == 1372555530269073761
        assert derive_item_seed(0, "op", 0) == 6517267240548121353

    def test_seed_derivation_range_and_distinctness(self):
        seeds = {derive_item_seed(1, "op", index) for index in range(64)}
        assert len(seeds) == 64
        assert all(0 <= seed < 2 ** 63 for seed in seeds)

    def test_run_bounded_preserves_order_and_captures_errors(self):
        def ok(value):
            return lambda: value

        def boom():
            raise ValueError("boom")

        outcomes = run_bounded([ok(1), boom, ok(3)], max_concurrency=2)
        assert outcomes[0] == 1
        assert isinstance(outcomes[1], ValueError)
        assert outcomes[2] == 3

    def test_transport_retries_then_succeeds(self):
        attempts = []
        sleeps = []

        def flaky():
            attempts.append(1)
            if len(attempts) < 3:
                raise TransportError("down")
            return "up"

        assert with_transport_retries(flaky, max_attempts=3, sleep=sleeps.append) == "up"
        assert len(attempts) == 3
        assert sleeps == [0.5, 1.0]

    def test_transport_retries_exhaust(self):
        def always_down():
            raise TransportError("down")

        with pytest.raises(TransportError):
            with_transport_retries(always_down, max_attempts=2, sleep=lambda _: None)

    def test_application_errors_never_retry(self):
        attempts = []

        def rejected():
            attempts.append(1)
            raise ValueError("bad param")

        with pytest.raises(ValueError):
            with_transport_retries(rejected, max_attempts=3, sleep=lambda _: None)
        assert len(attempts) == 1

    def test_partial_batch_error_carries_remainder(self):
        results = [object(), object()]
        failures = [(1, ValueError("x")), (3, TransportError("y"))]
        error = PartialBatchError(results, failures)
        assert error.failed_indices == (1, 3)
        assert len(error.results) == 2
        assert "2 of 4" in str(error)


class TestStopSemantics:

    def test_truncate_at_earliest_occurrence(self):
        assert truncate_at_stop_strings("alpha STOP beta END", ["END", "STOP"]) == "alpha "
        assert truncate_at_stop_strings("no stops here", ["END"]) == "no stops here"
        assert truncate_at_stop_strings("text", []) == "text"

    def test_stop_precedes_eos_and_length(self, tokenizer):
        new_tokens = torch.tensor([[6, 5, 1]])  # "on sat" then eos, at the length cap
        reasons = infer_finish_reasons(
            new_tokens, {"max_new_tokens": 3}, eos_token_id=1, pad_token_id=2,
            stop_strings=("sat",), tokenizer=tokenizer,
        )
        assert reasons == ["stop"]

    def test_stop_token_ids_classify_as_stop(self):
        reasons = infer_finish_reasons(
            torch.tensor([[6, 5]]), {}, eos_token_id=1, pad_token_id=2, stop_token_ids=(5,),
        )
        assert reasons == ["stop"]

    def test_eos_precedes_length_at_boundary(self):
        # eos generated exactly at the cap classifies as eos under the pinned precedence
        reasons = infer_finish_reasons(
            torch.tensor([[6, 6, 1]]), {"max_new_tokens": 3}, eos_token_id=1, pad_token_id=2,
        )
        assert reasons == ["eos"]

    def test_without_stop_rules_reduces_to_prior_labels(self):
        reasons = infer_finish_reasons(
            torch.tensor([[5, 7, 2], [5, 6, 8]]), {"max_new_tokens": 3},
            eos_token_id=7, pad_token_id=2,
        )
        assert reasons == ["eos", "length"]


class TestSessionBatchedFastPath:

    def test_batched_matches_direct_batched_generate(self, backend, model, tokenizer):
        encoded = tokenizer(["the cat", "the dog ran"], return_tensors="pt", padding=True)
        items = [
            GenerationItem(prompt=PreparedPrompt.from_token_ids(
                encoded["input_ids"][i:i + 1], encoded["attention_mask"][i:i + 1],
            ))
            for i in range(2)
        ]
        with backend.open_session() as session:
            results = session.generate(items, GenerationParams(max_new_tokens=4, greedy=True))
        direct = model.generate(
            input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"],
            max_new_tokens=4, do_sample=False,
        )
        prompt_len = encoded["input_ids"].size(1)
        for i, result in enumerate(results):
            assert torch.equal(result.output.output_ids, direct[i:i + 1, prompt_len:])
            assert torch.equal(result.output.adapted_input_ids, encoded["input_ids"][i:i + 1])

    def test_shared_params_seed_derives_distinct_item_seeds(self, backend, tokenizer):
        encoded = tokenizer(["the cat", "the cat"], return_tensors="pt", padding=True)
        items = [
            GenerationItem(prompt=PreparedPrompt.from_token_ids(
                encoded["input_ids"][i:i + 1], encoded["attention_mask"][i:i + 1],
            ))
            for i in range(2)
        ]
        params = GenerationParams(max_new_tokens=8, greedy=False, temperature=1.0, seed=42)
        with backend.open_session() as session:
            first = session.generate(items, params)
        with backend.open_session() as session:
            second = session.generate(items, params)
        # reproducible across sessions, and the two identical prompts sample distinct streams
        assert torch.equal(first[0].output.output_ids, second[0].output.output_ids)
        assert torch.equal(first[1].output.output_ids, second[1].output.output_ids)
        assert not torch.equal(first[0].output.output_ids, first[1].output.output_ids)

    def test_stop_strings_compose_and_classify(self, backend, tokenizer):
        item = GenerationItem(prompt=PreparedPrompt.from_text("the cat"))
        params = GenerationParams(
            max_new_tokens=6, greedy=True, stop_strings=("sat",),
            extra={"logits_processor": [_ForceSequence(3, [6, 5, 6, 6])]},
        )
        with backend.open_session() as session:
            result = session.generate([item], params)[0]
        decoded = tokenizer.decode(result.output.output_ids[0], skip_special_tokens=True)
        assert "sat" in decoded  # ids returned as generated
        assert result.output.finish_reason == "stop"
        assert result.output.finish_reasons == ("stop",)

    def test_score_batched_matches_serial(self, backend, tokenizer):
        encoded = tokenizer(["the cat", "the dog ran"], return_tensors="pt", padding=True)
        ref = torch.tensor([[5, 6], [7, 3]])
        from aisteer360.algorithms.core.execution import ScoringItem

        items = [
            ScoringItem(
                prompt=PreparedPrompt.from_token_ids(
                    encoded["input_ids"][i:i + 1], encoded["attention_mask"][i:i + 1],
                ),
                ref_output_ids=ref[i:i + 1],
            )
            for i in range(2)
        ]
        with backend.open_session() as session:
            batched = session.score(items, GenerationParams())
        serial_rows = []
        with backend.open_session() as session:
            for item in items:
                serial_rows.append(session.score([item], GenerationParams()))
        serial = torch.cat(serial_rows, dim=0)
        assert torch.allclose(batched, serial, atol=1e-4)

    def test_uneven_text_prompts_batch_like_serial(self, backend):
        prompts = ["the cat", "the dog ran fast on the mat"]
        params = GenerationParams(max_new_tokens=4, greedy=True, extra={"eos_token_id": None})
        with backend.open_session() as session:
            batched = session.generate(
                [GenerationItem(prompt=PreparedPrompt.from_text(p)) for p in prompts], params,
            )
        serial = []
        for prompt in prompts:
            with backend.open_session() as session:
                serial.append(
                    session.generate([GenerationItem(prompt=PreparedPrompt.from_text(prompt))], params)[0]
                )
        for one, many in zip(serial, batched):
            assert torch.equal(one.output.output_ids, many.output.output_ids)


class TestPadTokenDefaulting:
    """The session defaults pad_token_id per call without mutating the model's generation config."""

    @staticmethod
    def _capture_generate_kwargs(fresh_model):
        """Wrap `fresh_model.generate` to record the kwargs of the most recent call."""
        seen = {}
        original = fresh_model.generate

        def _spy(*args, **kwargs):
            seen.clear()
            seen.update(kwargs)
            return original(*args, **kwargs)

        fresh_model.generate = _spy
        return seen

    def _run(self, fresh_model, fresh_tokenizer, params):
        seen = self._capture_generate_kwargs(fresh_model)
        backend = HFBackend.adopt(HF_SPEC, lambda: fresh_model, lambda: fresh_tokenizer)
        item = GenerationItem(prompt=PreparedPrompt.from_text("the cat"))
        with backend.open_session() as session:
            session.generate([item], params)
        return seen

    def test_defaults_pad_token_when_config_unset(self):
        fresh_model = tiny_llama(num_layers=2, hidden=16, heads=2)
        fresh_tokenizer = wordlevel_tokenizer()
        assert fresh_model.generation_config.pad_token_id is None
        assert fresh_tokenizer.pad_token_id is not None

        seen = self._run(fresh_model, fresh_tokenizer, GenerationParams(max_new_tokens=3, greedy=True))

        assert seen["pad_token_id"] == fresh_tokenizer.pad_token_id
        # the model's generation config is not mutated by the defaulting
        assert fresh_model.generation_config.pad_token_id is None

    def test_generation_config_object_is_unchanged_after_generation(self):
        fresh_model = tiny_llama(num_layers=2, hidden=16, heads=2)
        fresh_tokenizer = wordlevel_tokenizer()
        config_before = fresh_model.generation_config

        self._run(fresh_model, fresh_tokenizer, GenerationParams(max_new_tokens=3, greedy=True))

        assert fresh_model.generation_config is config_before  # identity preserved
        assert fresh_model.generation_config.pad_token_id is None  # value preserved

    def test_model_configured_pad_token_takes_precedence_over_tokenizer(self):
        fresh_model = tiny_llama(num_layers=2, hidden=16, heads=2)
        fresh_tokenizer = wordlevel_tokenizer()
        fresh_model.generation_config.pad_token_id = 0  # differs from tokenizer.pad_token_id (2)

        seen = self._run(fresh_model, fresh_tokenizer, GenerationParams(max_new_tokens=3, greedy=True))

        # a model-configured value is left to model.generate; the session adds no override
        assert "pad_token_id" not in seen

    def test_caller_pad_token_takes_precedence(self):
        fresh_model = tiny_llama(num_layers=2, hidden=16, heads=2)
        fresh_tokenizer = wordlevel_tokenizer()
        assert fresh_model.generation_config.pad_token_id is None

        seen = self._run(
            fresh_model, fresh_tokenizer,
            GenerationParams(max_new_tokens=3, greedy=True, extra={"pad_token_id": 1}),
        )

        assert seen["pad_token_id"] == 1  # caller kwarg wins over the tokenizer default


class TestPipelineStopRules:

    def test_decoded_text_truncates_at_stop_string(self, model, tokenizer):
        pipeline = _pipeline(model, tokenizer, [StoppingRules(stop_texts=["sat"])])
        text = pipeline.generate(
            text="the cat", max_new_tokens=6, do_sample=False,
            logits_processor=[_ForceSequence(3, [6, 5, 6, 6])],
        )
        assert "sat" not in text
        assert text.startswith("on")

    def test_output_ids_keep_stop_text_and_reason_is_stop(self, model, tokenizer):
        pipeline = _pipeline(model, tokenizer, [StoppingRules(stop_texts=["sat"])])
        out = pipeline.generate(
            text="the cat", max_new_tokens=6, do_sample=False, return_output=True,
            logits_processor=[_ForceSequence(3, [6, 5, 6, 6])],
        )
        assert "sat" in tokenizer.decode(out.output_ids[0], skip_special_tokens=True)
        assert out.finish_reason == "stop"

    def test_budget_lowers_to_length(self, model, tokenizer):
        pipeline = _pipeline(model, tokenizer, [StoppingRules(budget=2)])
        out = pipeline.generate(
            text="the cat", max_new_tokens=6, do_sample=False, return_output=True,
            logits_processor=[_ForceSequence(3, [6, 6, 6, 6])],
        )
        assert out.output_ids.size(1) <= 2
        assert out.finish_reason == "length"

    def test_caller_stop_strings_flow_without_stopping_rules(self, model, tokenizer):
        pipeline = _pipeline(model, tokenizer)
        text = pipeline.generate(
            text="the cat", max_new_tokens=6, do_sample=False, stop_strings=["sat"],
            logits_processor=[_ForceSequence(3, [6, 5, 6, 6])],
        )
        assert "sat" not in text

    def test_per_candidate_finish_reasons(self, model, tokenizer):
        pipeline = _pipeline(model, tokenizer)
        out = pipeline.generate(
            text="the cat", max_new_tokens=3, do_sample=True, num_return_sequences=3,
            seed=11, return_output=True,
        )
        assert out.finish_reasons is not None
        assert len(out.finish_reasons) == 3
        assert out.finish_reason == out.finish_reasons[0]

    def test_output_return_exposes_every_candidate(self, model, tokenizer):
        pipeline = _pipeline(model, tokenizer)
        out = pipeline.generate(
            text="the cat", max_new_tokens=3, do_sample=True, num_return_sequences=3,
            seed=11, return_output=True,
        )
        assert out.output_ids.size(0) == 3
        assert len(out.decode(tokenizer)) == 3

    def test_decoded_single_with_multiple_candidates_rejected(self, model, tokenizer):
        pipeline = _pipeline(model, tokenizer)
        with pytest.raises(ValueError, match="exactly one candidate per prompt"):
            pipeline.generate(
                text="the cat", max_new_tokens=3, do_sample=True, num_return_sequences=3,
            )

    def test_decoded_batched_with_multiple_candidates_rejected(self, model, tokenizer):
        pipeline = _pipeline(model, tokenizer)
        with pytest.raises(ValueError, match="return_output=True"):
            pipeline.generate(
                text=["the cat", "the dog"], max_new_tokens=3, do_sample=True,
                num_return_sequences=2,
            )

    def test_decoded_n_alias_with_multiple_candidates_rejected(self, model, tokenizer):
        pipeline = _pipeline(model, tokenizer)
        with pytest.raises(ValueError, match="exactly one candidate per prompt"):
            pipeline.generate(text="the cat", max_new_tokens=3, do_sample=True, n=2)

    def test_decoded_single_candidate_allowed(self, model, tokenizer):
        pipeline = _pipeline(model, tokenizer)
        text = pipeline.generate(
            text="the cat", max_new_tokens=3, do_sample=True, num_return_sequences=1, seed=11,
        )
        assert isinstance(text, str)

    def test_token_return_carries_candidates_in_shape(self, model, tokenizer):
        pipeline = _pipeline(model, tokenizer)
        ids = pipeline.generate(
            input_ids=torch.tensor([3, 4]), max_new_tokens=3, do_sample=True,
            num_return_sequences=3, seed=11,
        )
        assert ids.size(0) == 3

    def test_seeded_pipeline_generation_is_repeatable(self, model, tokenizer):
        pipeline = _pipeline(model, tokenizer)
        first = pipeline.generate(
            input_ids=torch.tensor([[0, 3, 4]]), max_new_tokens=6, do_sample=True, seed=5,
        )
        second = pipeline.generate(
            input_ids=torch.tensor([[0, 3, 4]]), max_new_tokens=6, do_sample=True, seed=5,
        )
        assert torch.equal(first, second)


class _SessionProbeDriver(DecodingDriver):
    """Driver double recording the session it received and generating through it."""

    Args = None

    def _configure(self):
        self.seen_session = None

    def decode(self, input_ids, attention_mask, model, logits_processors,
               stopping_criteria, runtime_kwargs, session=None, **gen_kwargs):
        self.seen_session = session
        extra = stack_generate_kwargs(logits_processors, stopping_criteria)
        return session_generate(session, input_ids, attention_mask, **extra, **gen_kwargs)


class TestDriversOverSessions:

    def test_driver_receives_session_and_generates_through_it(self, model, tokenizer):
        driver = _SessionProbeDriver()
        pipeline = _pipeline(model, tokenizer, [driver])
        out = pipeline.generate(input_ids=torch.tensor([[0, 3, 4]]), max_new_tokens=3, do_sample=False)
        assert driver.seen_session is not None
        assert out.ndim == 2

    def test_session_generate_matches_model_generate(self, backend, model, tokenizer):
        input_ids = torch.tensor([[0, 3, 4]])
        with backend.open_session() as session:
            via_session = session_generate(
                session, input_ids, torch.ones_like(input_ids), max_new_tokens=4, do_sample=False,
            )
        direct = model.generate(
            input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
            max_new_tokens=4, do_sample=False,
        )
        assert torch.equal(via_session, direct)

class TestPortableRequirements:

    def _generate_ok_on_vllm(self, control) -> bool:
        pipeline = SteeringPipeline(model_name_or_path="m", controls=[control])
        report = pipeline.check(backend=VLLM_SPEC)
        return report.supported("generate")

    def test_stopping_rules_supported_everywhere(self):
        control = StoppingRules(stop_texts=["x"])
        assert control.requirements().generate == ()
        assert self._generate_ok_on_vllm(control)

    def test_phase_drivers_supported_on_vllm(self):
        assert self._generate_ok_on_vllm(BudgetForcing(max_thinking_tokens=4))
        assert self._generate_ok_on_vllm(
            PhasedDecoding(
                plan=[{"fixed": lambda prompt, params: prompt, "replace": True, "add_special_tokens": True},
                      {"generate": {}}],
                extract_after="</think>",
            )
        )

    def test_sampled_search_supported_beam_not(self):
        scorer = lambda prompt, continuations, params: [0.0] * len(continuations)  # noqa: E731
        assert self._generate_ok_on_vllm(BestOfN(n=2, scorer=scorer))
        assert self._generate_ok_on_vllm(
            SearchDecoding(scorer=scorer, num_candidates=2, propose_mode="sample")
        )
        beam = SearchDecoding(scorer=scorer, num_candidates=2, propose_mode="beam")
        assert not self._generate_ok_on_vllm(beam)
        deal = DeAL(reward_func=scorer)
        report = SteeringPipeline(model_name_or_path="m", controls=[deal]).check(backend=VLLM_SPEC)
        assert not report.supported("generate")
        assert any("BEAM_PROPOSALS" in failure.message for failure in report.failures)

    def test_input_controls_are_prompt_only_at_generate(self):
        class _Passthrough(InputControl):
            def adapt(self, input_ids, runtime_kwargs=None):
                return input_ids

        assert _Passthrough().requirements().generate == ()

    def test_refinement_input_controls_declare_rollouts_access(self):
        for cls in (PRewrite, GEPA):
            control = object.__new__(cls)
            requirements = control.requirements()
            assert requirements.generate == ()
            assert control.steer_access() is ModelAccess.ROLLOUTS


class _CheckpointProducingControl(StructuralControl):
    """Structural double whose configuration produces a checkpoint artifact."""

    Args = None

    def steer(self, model, tokenizer=None, **kwargs):
        return model

    def artifact_capability(self):
        return Capability.SERVE_CHECKPOINT

    def export_artifact(self):
        return CheckpointArtifact(path="/tmp/ckpt")


class TestStructuralArtifacts:

    def test_requirements_gain_serving_alternative(self):
        control = _CheckpointProducingControl()
        requirements = control.requirements()
        assert len(requirements.generate) == 2
        assert Capability.SERVE_CHECKPOINT in requirements.generate[1].atoms

    def test_check_passes_with_staged_steer_and_vllm_serving(self):
        pipeline = SteeringPipeline(model_name_or_path="m", controls=[_CheckpointProducingControl()])
        report = pipeline.check(backend=VLLM_SPEC)
        assert report.supported("generate")
        assert report.ok
        assert report.plan.steps[0].venue == "stage"
        assert report.plan.stages is True

    def test_pipeline_collects_and_stamps_artifacts(self, model, tokenizer):
        control = _CheckpointProducingControl()
        pipeline = SteeringPipeline(controls=[control], model=model, tokenizer=tokenizer)
        pipeline.steer()
        artifacts = pipeline._structural_artifacts
        assert len(artifacts) == 1
        artifact = artifacts[0]
        assert artifact.path == "/tmp/ckpt"
        assert artifact.provenance.backend_spec_hash is not None
        assert artifact.provenance.model_fingerprint is not None
        assert len(artifact.provenance.model_fingerprint) == 16


class TestTRLArtifactDerivation:

    def _mixin(self, **attrs):
        from peft import PeftType

        from aisteer360.algorithms.structural_control.wrappers.trl.base_mixin import TRLMixin

        control = object.__new__(type("_TRLDouble", (TRLMixin,), {}))
        control.training_args = {}
        control.output_dir = "./out"
        control.use_peft = False
        control.peft_type = PeftType.LORA
        control.merge_lora_after_train = False
        control.merged_output_dir = None
        control.base_model_name_or_path = "base/model"
        control.model = None
        control.train_dataset = object()
        for name, value in attrs.items():
            setattr(control, name, value)
        return control

    def test_full_finetune_yields_checkpoint(self):
        control = self._mixin(use_peft=False)
        assert control.artifact_capability() == Capability.SERVE_CHECKPOINT
        artifact = control.export_artifact()
        assert isinstance(artifact, CheckpointArtifact)
        assert artifact.path == "./out"

    def test_lora_yields_adapter(self):
        control = self._mixin(use_peft=True)
        assert control.artifact_capability() == Capability.SERVE_LORA
        artifact = control.export_artifact()
        assert isinstance(artifact, LoRAArtifact)
        assert artifact.path == "./out"
        assert artifact.base_model == "base/model"

    def test_merged_lora_yields_checkpoint_only_with_merged_dir(self):
        merged = self._mixin(use_peft=True, merge_lora_after_train=True, merged_output_dir="./merged")
        assert merged.artifact_capability() == Capability.SERVE_CHECKPOINT
        assert merged.export_artifact().path == "./merged"
        unmerged = self._mixin(use_peft=True, merge_lora_after_train=True)
        assert unmerged.artifact_capability() is None
        assert unmerged.export_artifact() is None

    def test_no_training_yields_nothing(self):
        control = self._mixin(train_dataset=None)
        assert control.artifact_capability() is None
        assert control.export_artifact() is None


class TestOutputRecord:

    def test_finish_reasons_field_defaults_to_none(self):
        out = Output(output_ids=torch.tensor([[1, 2]]))
        assert out.finish_reasons is None

    def test_finish_reasons_field_holds_per_candidate_reasons(self):
        out = Output(
            output_ids=torch.tensor([[1, 2], [3, 4]]),
            finish_reason="eos",
            finish_reasons=("eos", "length"),
        )
        assert out.finish_reasons == ("eos", "length")
        assert dataclasses.fields(Output)[3].name == "finish_reasons"


class _RowRecordingStateControl(StateControl):
    """Records the input_ids shape of every get_hooks call into a shared list."""

    Args = None

    def _configure(self):
        self.seen_shapes: list[tuple[int, ...]] = []

    def get_hooks(self, input_ids, runtime_kwargs, **kwargs):
        self.seen_shapes.append(tuple(input_ids.shape))
        return {"pre": [], "forward": [], "backward": []}


class TestSerialSeedStateHooks:
    """Distinct per-item derived seeds force the serial session path; state hooks are computed
    per row there rather than once on the batch."""

    def test_seeded_multi_prompt_batch_computes_hooks_per_row(self, model, tokenizer):
        control = _RowRecordingStateControl()
        pipeline = _pipeline(model, tokenizer, controls=[control])
        pipeline.generate(text=["the cat sat on the mat", "the dog"], seed=7, max_new_tokens=2)
        assert len(control.seen_shapes) == 2
        assert all(shape[0] == 1 for shape in control.seen_shapes)

    def test_unseeded_multi_prompt_batch_keeps_batch_hooks(self, model, tokenizer):
        control = _RowRecordingStateControl()
        pipeline = _pipeline(model, tokenizer, controls=[control])
        pipeline.generate(text=["the cat sat on the mat", "the dog"], max_new_tokens=2)
        assert len(control.seen_shapes) == 1
        assert control.seen_shapes[0][0] == 2

    def test_seeded_batch_runs_runtime_backed_control_per_row(self, model, tokenizer):
        transform = RecordingTransform()
        control = ActivationAdapter(transform=transform, layer_ids=[1], token_scope="after_prompt")
        pipeline = _pipeline(model, tokenizer, controls=[control])
        pipeline.generate(text=["the cat sat on the mat", "the dog"], seed=7, max_new_tokens=2)
        assert transform.masks
        assert all(mask.size(0) == 1 for mask in transform.masks)

    def test_clone_for_call_isolates_gate_state(self, model, tokenizer):
        from aisteer360.algorithms.state_control.common.gating import CallableReadout, Evidence, Gate, PerKeyThreshold

        gate = Gate(
            Evidence((0,), CallableReadout(lambda pooled, layer_id: pooled.mean(dim=-1))),
            PerKeyThreshold(threshold=0.0, comparator="ge"),
        )
        control = ActivationAdapter(transform=RecordingTransform(), layer_ids=[1], gate=gate)
        control.steer(model, tokenizer)
        clone = control.clone_for_call()
        assert clone._gate is not control._gate  # per-row gate state never shared across clones
        assert type(clone._gate) is type(control._gate)
        assert clone.interventions[0].transform is control.interventions[0].transform  # artifacts shared

    def test_clone_for_call_keeps_ungated_interventions_ungated(self, model, tokenizer):
        control = ActivationAdapter(transform=RecordingTransform(), layer_ids=[1])
        control.steer(model, tokenizer)
        clone = control.clone_for_call()
        assert control._gate is None and clone._gate is None
