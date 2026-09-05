"""Output-control mechanisms in `SteeringPipeline`.

Covers the mechanism-based composition of the output category: step-level controls
(`get_logits_processors` / `get_stopping_criteria`) compose in `controls`-list order; the decode
loop is exclusive and owned by at most one `DecodingDriver` (default: the pipeline's
`HFGenerateDriver`). Step-level controls also apply during `compute_logprobs`; drivers and stopping
criteria do not.

Runs hub-free on a tiny randomly-initialized Llama with module-local fixture controls (no
`output_control/common` dependency).
"""
import math

import pytest
import torch
from transformers import LogitsProcessorList, StoppingCriteria

from steerability.algorithms.core.steering_pipeline import SteeringPipeline
from steerability.algorithms.core.utils.controls import merge_controls
from steerability.algorithms.output_control.base import DecodingDriver, OutputControl
from tests.utils.tiny_models import tiny_llama, wordlevel_tokenizer

HIDDEN = 32
HEADS = 4
LAYERS = 4
VOCAB = 100


# fixture controls (module-local; no common)
class _ForceTokenControl(OutputControl):
    """Contributes a processor that masks all logits to -inf except token `k`."""

    Args = None

    def __init__(self, k: int):
        super().__init__()
        self._k = k

    def get_logits_processors(self, input_ids, runtime_kwargs, **kwargs):
        k = self._k

        def _force(prefix_ids, scores):
            out = torch.full_like(scores, float("-inf"))
            out[:, k] = scores[:, k]
            return out

        return [_force]


class _MarkerControl(OutputControl):
    """Records (name, scores[0, probe]) then boosts `token_id`.

    Observing `scores[0, probe]` lets a downstream marker see whether an upstream marker's edit
    landed; recording call order shows the composition order.
    """

    Args = None

    def __init__(self, name: str, token_id: int, recorder: list, probe: int = 0):
        super().__init__()
        self._name = name
        self._token_id = token_id
        self._recorder = recorder
        self._probe = probe

    def get_logits_processors(self, input_ids, runtime_kwargs, **kwargs):
        name, token_id, recorder, probe = self._name, self._token_id, self._recorder, self._probe

        def _mark(prefix_ids, scores):
            recorder.append((name, float(scores[0, probe].item())))
            scores = scores.clone()
            scores[:, token_id] += 100.0
            return scores

        return [_mark]


class _StopAfterControl(OutputControl):
    """Contributes a stopping criterion that halts at prompt_len + n new tokens."""

    Args = None

    def __init__(self, n: int):
        super().__init__()
        self._n = n

    def get_stopping_criteria(self, input_ids, runtime_kwargs, **kwargs):
        stop_len = input_ids.size(1) + self._n

        class _Stop(StoppingCriteria):
            def __call__(self, ids, scores, **kw):
                fired = ids.size(1) >= stop_len
                return torch.full((ids.size(0),), fired, dtype=torch.bool, device=ids.device)

        return [_Stop()]


class _CapturingDriver(DecodingDriver):
    """Records the arguments it received, then delegates to model.generate with the stacks."""

    Args = None
    supports_batching = True

    def __init__(self):
        super().__init__()
        self.captured = None

    def decode(self, input_ids, attention_mask, model, logits_processors,
               stopping_criteria, runtime_kwargs, session=None, **gen_kwargs):
        self.captured = {
            "logits_processors": logits_processors,
            "stopping_criteria": stopping_criteria,
            "runtime_kwargs": runtime_kwargs,
            "session": session,
            "gen_kwargs": dict(gen_kwargs),
        }
        extra = {}
        if len(logits_processors):
            extra["logits_processor"] = logits_processors
        if len(stopping_criteria):
            extra["stopping_criteria"] = stopping_criteria
        return model.generate(input_ids=input_ids, attention_mask=attention_mask, **extra, **gen_kwargs)


class _UniformControl(OutputControl):
    """Scoring step-level control that sets all logits equal (uniform next-token distribution)."""

    Args = None
    supports_batching = True

    def get_logits_processors(self, input_ids, runtime_kwargs, **kwargs):
        def _uniform(prefix_ids, scores):
            return torch.zeros_like(scores)

        return [_uniform]


def _pipeline(controls, model=None):
    """Build a steered pipeline (hub-free tiny Llama, lazy-init inject pattern)."""
    if model is None:
        model = tiny_llama(num_layers=LAYERS, hidden=HIDDEN, heads=HEADS, vocab=VOCAB)
    tokenizer = wordlevel_tokenizer()
    pipeline = SteeringPipeline(controls=controls, model=model, tokenizer=tokenizer)
    pipeline.steer()
    return pipeline, model


def _prompt_ids():
    return torch.tensor([[0, 3, 4, 5]], dtype=torch.long)  # <s> the cat sat


# merge semantics
class TestMergeOutput:
    def test_two_step_level_controls_encounter_order(self):
        rec = []
        a = _MarkerControl("a", 3, rec)
        b = _MarkerControl("b", 4, rec)
        result = merge_controls([a, b])
        assert result["output_controls"] == [a, b]

    def test_step_level_control_plus_driver_ok(self):
        step_level_control = _ForceTokenControl(3)
        driver = _CapturingDriver()
        result = merge_controls([step_level_control, driver])
        assert result["output_controls"] == [step_level_control, driver]

    def test_two_drivers_raise(self):
        with pytest.raises(ValueError, match="decoding drivers"):
            merge_controls([_CapturingDriver(), _CapturingDriver()])

    def test_empty_stays_empty(self):
        assert merge_controls([])["output_controls"] == []

    def test_disabled_driver_does_not_count(self):
        d1 = _CapturingDriver()
        d2 = _CapturingDriver()
        d2.enabled = False
        # only one enabled driver -> no error
        result = merge_controls([d1, d2])
        assert result["output_controls"] == [d1, d2]


# type-level claims
class TestTypeLevelClaims:
    def test_driver_without_decode_is_abstract(self):
        with pytest.raises(TypeError):
            class _BadDriver(DecodingDriver):
                Args = None
            _BadDriver()

    def test_isinstance_is_ownership_predicate(self):
        assert isinstance(_CapturingDriver(), DecodingDriver)
        assert not isinstance(_ForceTokenControl(3), DecodingDriver)


# default-driver composition
class TestDefaultDriverFold:
    def test_force_token_control_steers_every_new_token(self):
        pipeline, _ = _pipeline([_ForceTokenControl(7)])
        out = pipeline.generate(
            input_ids=_prompt_ids(), max_new_tokens=5, do_sample=False, eos_token_id=None,
        )
        assert out.ndim == 2
        assert torch.all(out == 7)


# composition order & chaining
class TestFoldOrderAndChaining:
    def test_markers_fire_in_list_order_and_chain(self):
        rec = []
        # b probes token 3 (which a boosts): b observes a's edit
        a = _MarkerControl("a", 3, rec, probe=3)
        b = _MarkerControl("b", 4, rec, probe=3)
        pipeline, _ = _pipeline([a, b])
        pipeline.generate(input_ids=_prompt_ids(), max_new_tokens=1, do_sample=False, eos_token_id=None)

        assert [name for name, _ in rec] == ["a", "b"]
        a_probe = rec[0][1]
        b_probe = rec[1][1]
        assert b_probe == pytest.approx(a_probe + 100.0, abs=1e-3)

    def test_swapped_list_swaps_order(self):
        rec = []
        a = _MarkerControl("a", 3, rec, probe=3)
        b = _MarkerControl("b", 4, rec, probe=3)
        pipeline, _ = _pipeline([b, a])
        pipeline.generate(input_ids=_prompt_ids(), max_new_tokens=1, do_sample=False, eos_token_id=None)
        assert [name for name, _ in rec] == ["b", "a"]


# caller extras
class TestCallerExtras:
    def test_user_processor_runs_after_composed_stacks_and_driver_kwargs_clean(self):
        driver = _CapturingDriver()
        pipeline, _ = _pipeline([_ForceTokenControl(3), driver])

        call_log = []

        def _user(prefix_ids, scores):
            call_log.append("user")
            return scores

        pipeline.generate(
            input_ids=_prompt_ids(), max_new_tokens=2, do_sample=False, eos_token_id=None,
            logits_processor=[_user],
        )

        # user processor appended AFTER the step-level control
        assert len(driver.captured["logits_processors"]) == 2
        # gen_kwargs handed to the driver contains no processor/criteria keys
        assert "logits_processor" not in driver.captured["gen_kwargs"]
        assert "stopping_criteria" not in driver.captured["gen_kwargs"]
        assert call_log  # user processor did run

    def test_no_step_level_controls_no_extras_calls_model_generate_without_stack_kwargs(self):
        driver = _CapturingDriver()
        pipeline, _ = _pipeline([driver])
        pipeline.generate(input_ids=_prompt_ids(), max_new_tokens=2, do_sample=False, eos_token_id=None)
        assert len(driver.captured["logits_processors"]) == 0
        assert len(driver.captured["stopping_criteria"]) == 0


# driver receives the stacks
class TestDriverReceivesStacks:
    def test_step_level_control_composed_into_driver_stack(self):
        driver = _CapturingDriver()
        pipeline, _ = _pipeline([_ForceTokenControl(9), driver])
        out = pipeline.generate(
            input_ids=_prompt_ids(), max_new_tokens=3, do_sample=False, eos_token_id=None,
        )
        assert isinstance(driver.captured["logits_processors"], LogitsProcessorList)
        assert len(driver.captured["logits_processors"]) == 1
        assert torch.all(out == 9)


# stopping criterion
class TestStoppingCriterion:
    def test_stop_after_limits_new_tokens(self):
        pipeline, _ = _pipeline([_StopAfterControl(2)])
        out = pipeline.generate(
            input_ids=_prompt_ids(), max_new_tokens=8, do_sample=False,
            eos_token_id=None, return_full_sequence=True,
        )
        prompt_len = _prompt_ids().size(1)
        assert out.size(1) - prompt_len <= 2


# disabled driver fallback
class TestDisabledDriverFallback:
    def test_disabled_driver_not_called_but_step_level_control_still_steers(self):
        driver = _CapturingDriver()
        driver.enabled = False
        pipeline, _ = _pipeline([_ForceTokenControl(5), driver])
        out = pipeline.generate(
            input_ids=_prompt_ids(), max_new_tokens=3, do_sample=False, eos_token_id=None,
        )
        assert driver.captured is None  # driver.decode never invoked
        assert torch.all(out == 5)  # default driver still applied the step-level control


# supports_batching
class TestSupportsBatching:
    def test_and_across_controls(self):
        pipeline, _ = _pipeline([_UniformControl()])  # supports_batching=True
        assert pipeline.supports_batching is True

        class _NonBatch(OutputControl):
            Args = None
            supports_batching = False

        pipeline2, _ = _pipeline([_NonBatch()])
        assert pipeline2.supports_batching is False

    def test_empty_output_is_vacuously_true(self):
        pipeline, _ = _pipeline([])
        assert pipeline.supports_batching is True


# freshness
class TestFreshness:
    def test_distinct_processor_instances_per_call(self):
        seen = []  # holds references so identities cannot be recycled by GC

        class _Recording(OutputControl):
            Args = None

            def get_logits_processors(self, input_ids, runtime_kwargs, **kwargs):
                def _p(prefix_ids, scores):
                    return scores
                seen.append(_p)
                return [_p]

        pipeline, _ = _pipeline([_Recording()])
        pipeline.generate(input_ids=_prompt_ids(), max_new_tokens=1, do_sample=False, eos_token_id=None)
        pipeline.generate(input_ids=_prompt_ids(), max_new_tokens=1, do_sample=False, eos_token_id=None)
        assert len(seen) == 2
        assert seen[0] is not seen[1]  # fresh instance each call


# compute_logprobs
class TestComputeLogprobs:
    @pytest.mark.parametrize("batched", [True, False])
    def test_uniform_control_yields_uniform_logprobs(self, batched):
        controls = [_UniformControl()]
        if not batched:
            # a non-batch sibling forces the sequential compute_logprobs path
            class _NonBatch(OutputControl):
                Args = None
                supports_batching = False
            controls.append(_NonBatch())

        pipeline, _ = _pipeline(controls)
        assert pipeline.supports_batching is batched

        prompt = _prompt_ids()
        ref = torch.tensor([[3, 4, 5]], dtype=torch.long)
        logprobs = pipeline.compute_logprobs(input_ids=prompt, ref_output_ids=ref)

        expected = math.log(1.0 / VOCAB)
        assert torch.allclose(logprobs, torch.full_like(logprobs, expected), atol=1e-4)

    def test_opt_out_not_applied_and_driver_not_invoked(self):
        class _OptOutUniform(_UniformControl):
            include_in_scoring = False

        driver = _CapturingDriver()
        opt_out = _OptOutUniform()
        pipeline, model = _pipeline([opt_out, driver])

        prompt = _prompt_ids()
        ref = torch.tensor([[3, 4, 5]], dtype=torch.long)

        # baseline logprobs with no output controls on the same model
        baseline_pipe, _ = _pipeline([], model=model)
        baseline = baseline_pipe.compute_logprobs(input_ids=prompt, ref_output_ids=ref)

        scored = pipeline.compute_logprobs(input_ids=prompt, ref_output_ids=ref)

        # opt-out step-level control NOT applied -> equals baseline
        assert torch.allclose(scored, baseline, atol=1e-5)
        # driver never invoked during scoring
        assert driver.captured is None

    def test_scoring_skip_logs_at_info(self, caplog):
        class _OptOutUniform(_UniformControl):
            include_in_scoring = False

        pipeline, _ = _pipeline([_OptOutUniform()])
        prompt = _prompt_ids()
        ref = torch.tensor([[3, 4, 5]], dtype=torch.long)

        with caplog.at_level("INFO", logger="steerability.algorithms.core.utils.assembly"):
            pipeline.compute_logprobs(input_ids=prompt, ref_output_ids=ref)
        assert any("_OptOutUniform" in r.message and "include_in_scoring" in r.message
                   for r in caplog.records)

        caplog.clear()
        with caplog.at_level("INFO", logger="steerability.algorithms.core.utils.assembly"):
            pipeline.generate(input_ids=prompt, max_new_tokens=2, do_sample=False, eos_token_id=None)
        # the skip log is a scoring concern only; generate must not emit it
        assert not any("_OptOutUniform" in r.message for r in caplog.records)
