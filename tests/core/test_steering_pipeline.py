"""
Tests for `SteeringPipeline` construction, steering, and scoring.

Tests cover:

- Initialization: Hugging Face loading, device handling, lazy mode, control sorting, and
  tokenizer injection
- `steer()`: bottom-up ordering, idempotence, kwarg forwarding, model replacement, and
  failure modes
- `generate()`: `runtime_kwargs` propagation to controls, prompt adaptation, and hook cleanup
- `compute_logprobs()`: input normalization, output shapes, and teacher-forced values
- The `supports_batching` property
- The duplicate-BOS guard
- `same_model_forwards` metadata

Generation dispatch (`text=`/`messages=`/`input_ids=`), output-stack composition, and
multi-control composition are covered in `test_polymorphic_generate.py`,
`test_output_mechanisms.py`, `test_input_structural_multiplicity.py`, and
`test_state_multiplicity.py`.
"""
import logging
import warnings
from unittest.mock import MagicMock

import pytest
import torch

from steerability.algorithms.core.steering_pipeline import SteeringPipeline
from steerability.algorithms.core.utils.assembly import _warn_on_provenance_mismatch
from steerability.algorithms.input_control.base import InputControl
from steerability.algorithms.output_control.base import OutputControl
from steerability.algorithms.structural_control.base import StructuralControl
from tests.conftest import (
    MockInputControl,
    MockOutputControl,
    MockStateControl,
    MockStructuralControl,
    create_mock_model,
    create_mock_tokenizer,
)
from tests.utils.tiny_models import tiny_llama, wordlevel_tokenizer


def _patch_hf_loaders(monkeypatch):
    """Replace the Hugging Face loader classes in the pipeline module with recording mocks.

    Returns:
        A tuple `(model_loader, tokenizer_loader, model, tokenizer)` where the loaders are
        `MagicMock` classes whose `from_pretrained` returns the given mock model and tokenizer.
    """
    model = create_mock_model()
    model.to.return_value = model
    tokenizer = create_mock_tokenizer()

    model_loader = MagicMock()
    model_loader.from_pretrained.return_value = model
    tokenizer_loader = MagicMock()
    tokenizer_loader.from_pretrained.return_value = tokenizer

    monkeypatch.setattr(
        "steerability.algorithms.core.steering_pipeline.AutoModelForCausalLM", model_loader
    )
    monkeypatch.setattr(
        "steerability.algorithms.core.steering_pipeline.AutoTokenizer", tokenizer_loader
    )
    return model_loader, tokenizer_loader, model, tokenizer


def _tiny_pipeline(controls=()) -> SteeringPipeline:
    """Build a pipeline over a hub-free tiny Llama model and WordLevel tokenizer."""
    torch.manual_seed(0)
    pipeline = SteeringPipeline(
        controls=list(controls), model=tiny_llama(num_layers=2, hidden=16, heads=2), tokenizer=wordlevel_tokenizer(),
    )
    return pipeline


# Pipeline Initialization Tests
class TestPipelineInitialization:
    """Tests for `SteeringPipeline` construction."""

    def test_steer_loads_model_and_tokenizer(self, monkeypatch):
        model_loader, tokenizer_loader, model, tokenizer = _patch_hf_loaders(monkeypatch)

        pipeline = SteeringPipeline(model_name_or_path="test-model")
        pipeline.steer()

        model_loader.from_pretrained.assert_called_once()
        assert model_loader.from_pretrained.call_args.args == ("test-model",)
        assert model_loader.from_pretrained.call_args.kwargs["device_map"] == "auto"
        tokenizer_loader.from_pretrained.assert_called_once_with(
            "test-model", trust_remote_code=False
        )
        assert pipeline.model is model
        assert pipeline.tokenizer is tokenizer
        assert pipeline._is_steered

    def test_model_source_required_without_structural_control(self):
        with pytest.raises(ValueError, match="model_name_or_path"):
            SteeringPipeline()

    def test_device_and_device_map_mutually_exclusive(self):
        with pytest.raises(ValueError, match="Cannot specify both"):
            SteeringPipeline(
                model_name_or_path="test-model",
                device="cpu",
                device_map={"": 0},
            )

    def test_device_moves_model(self, monkeypatch):
        model_loader, _, model, _ = _patch_hf_loaders(monkeypatch)

        pipeline = SteeringPipeline(model_name_or_path="test-model", device="cpu")
        pipeline.steer()

        assert "device_map" not in model_loader.from_pretrained.call_args.kwargs
        model.to.assert_called_once_with("cpu")
        assert pipeline.device == model.device

    def test_hf_model_kwargs_forwarded(self, monkeypatch):
        model_loader, _, _, _ = _patch_hf_loaders(monkeypatch)

        SteeringPipeline(
            model_name_or_path="test-model",
            hf_model_kwargs={"torch_dtype": "float16"},
        ).steer()

        kwargs = model_loader.from_pretrained.call_args.kwargs
        assert kwargs["torch_dtype"] == "float16"

    def test_trust_remote_code_forwarded_to_tokenizer(self, monkeypatch):
        _, tokenizer_loader, _, _ = _patch_hf_loaders(monkeypatch)

        SteeringPipeline(model_name_or_path="test-model", trust_remote_code=True).steer()

        assert tokenizer_loader.from_pretrained.call_args.kwargs["trust_remote_code"] is True

    def test_tokenizer_name_or_path_used(self, monkeypatch):
        _, tokenizer_loader, _, _ = _patch_hf_loaders(monkeypatch)

        SteeringPipeline(model_name_or_path="test-model", tokenizer_name_or_path="test-tokenizer").steer()

        assert tokenizer_loader.from_pretrained.call_args.args == ("test-tokenizer",)

    def test_construction_defers_loading(self, monkeypatch):
        model_loader, tokenizer_loader, _, _ = _patch_hf_loaders(monkeypatch)

        pipeline = SteeringPipeline(model_name_or_path="test-model")

        model_loader.from_pretrained.assert_not_called()
        tokenizer_loader.from_pretrained.assert_not_called()
        assert pipeline.model is None
        assert pipeline.tokenizer is None

    def test_controls_sorted_into_categories(self):
        input_ctrl = MockInputControl()
        state_ctrl = MockStateControl()

        pipeline = SteeringPipeline(model_name_or_path="test-model", controls=[input_ctrl, state_ctrl])

        assert pipeline.input_controls == [input_ctrl]
        assert pipeline.state_controls == [state_ctrl]
        assert pipeline.structural_controls == []
        assert pipeline.output_controls == []

    def test_all_four_categories(self):
        input_ctrl = MockInputControl()
        structural_ctrl = MockStructuralControl()
        state_ctrl = MockStateControl()
        output_ctrl = MockOutputControl()

        pipeline = SteeringPipeline(
            controls=[input_ctrl, structural_ctrl, state_ctrl, output_ctrl],
        )

        assert pipeline.input_controls == [input_ctrl]
        assert pipeline.structural_controls == [structural_ctrl]
        assert pipeline.state_controls == [state_ctrl]
        assert pipeline.output_controls == [output_ctrl]

    def test_tokenizer_injected_into_controls(self, monkeypatch):
        _, _, _, tokenizer = _patch_hf_loaders(monkeypatch)
        control = MockInputControl()  # class-level `tokenizer` is None

        SteeringPipeline(model_name_or_path="test-model", controls=[control]).steer()

        assert control.tokenizer is tokenizer


# Pipeline Steer Tests
class TestPipelineSteer:
    """Tests for `SteeringPipeline.steer()`."""

    def test_generate_requires_steer(self):
        pipeline = _tiny_pipeline()

        with pytest.raises(RuntimeError, match="steer"):
            pipeline.generate(input_ids=torch.tensor([[3, 4, 5]]), max_new_tokens=1)

    def test_steer_marks_as_steered(self):
        pipeline = _tiny_pipeline()

        assert not pipeline._is_steered
        pipeline.steer()
        assert pipeline._is_steered

    def test_second_steer_is_noop(self):
        steer_calls = []

        class _CountingStructural(MockStructuralControl):
            def steer(self, model, tokenizer=None, **kwargs):
                steer_calls.append(1)
                return super().steer(model, tokenizer=tokenizer, **kwargs)

        pipeline = _tiny_pipeline([_CountingStructural()])

        pipeline.steer()
        pipeline.steer()

        assert len(steer_calls) == 1

    def test_steer_calls_control_steer_methods(self):
        input_ctrl = MockInputControl()
        structural_ctrl = MockStructuralControl()
        state_ctrl = MockStateControl()

        pipeline = _tiny_pipeline([input_ctrl, structural_ctrl, state_ctrl])
        pipeline.steer()

        assert structural_ctrl._steer_called
        assert input_ctrl.model is pipeline.model
        assert state_ctrl.model is pipeline.model

    def test_steer_order(self):
        """Controls are steered bottom-up: structural, input, state, output."""
        call_order = []

        class TrackingInputControl(MockInputControl):
            def steer(self, model=None, tokenizer=None, **kwargs):
                call_order.append("input")
                super().steer(model, tokenizer, **kwargs)

        class TrackingStructuralControl(MockStructuralControl):
            def steer(self, model, tokenizer=None, **kwargs):
                call_order.append("structural")
                return super().steer(model, tokenizer=tokenizer, **kwargs)

        class TrackingStateControl(MockStateControl):
            def steer(self, model, tokenizer=None, **kwargs):
                call_order.append("state")
                super().steer(model, tokenizer=tokenizer, **kwargs)

        class TrackingOutputControl(MockOutputControl):
            def steer(self, model, tokenizer=None, **kwargs):
                call_order.append("output")
                super().steer(model, tokenizer=tokenizer, **kwargs)

        pipeline = _tiny_pipeline([
            TrackingInputControl(),
            TrackingStructuralControl(),
            TrackingStateControl(),
            TrackingOutputControl(),
        ])
        pipeline.steer()

        assert call_order == ["structural", "input", "state", "output"]

    def test_steer_passes_kwargs(self):
        received_kwargs = {}

        class KwargsCapturingControl(MockInputControl):
            def steer(self, model=None, tokenizer=None, **kwargs):
                received_kwargs.update(kwargs)
                super().steer(model, tokenizer, **kwargs)

        pipeline = _tiny_pipeline([KwargsCapturingControl()])
        pipeline.steer(custom_param="value")

        assert received_kwargs.get("custom_param") == "value"

    def test_structural_control_replaces_model(self):
        """A structural control returning an `nn.Module` replaces the pipeline model."""
        torch.manual_seed(1)
        replacement = tiny_llama(num_layers=2, hidden=16, heads=2)

        class _ReplacingStructural(StructuralControl):
            def __init__(self, new_model):
                super().__init__()
                self.new_model = new_model

            def steer(self, model, tokenizer=None, **kwargs):
                return self.new_model

        pipeline = _tiny_pipeline([_ReplacingStructural(replacement)])
        original = pipeline.model
        pipeline.steer()

        assert pipeline.model is replacement
        assert pipeline.model is not original

    def test_structural_control_returning_no_model_raises(self):
        class _NoModelStructural(MockStructuralControl):
            def steer(self, model=None, tokenizer=None, **kwargs):
                return None

        pipeline = SteeringPipeline(controls=[_NoModelStructural()])

        with pytest.raises(RuntimeError, match="No model is available after steering"):
            pipeline.steer()


# Pipeline Generate Tests
class TestPipelineGenerate:
    """Tests for control wiring during `SteeringPipeline.generate()`."""

    def test_runtime_kwargs_reach_input_control(self):
        control = MockInputControl()
        pipeline = _tiny_pipeline([control])
        pipeline.steer()

        runtime_kwargs = {"key": "value", "param": 123}
        pipeline.generate(
            input_ids=torch.tensor([[3, 4, 5]]),
            max_new_tokens=1,
            runtime_kwargs=runtime_kwargs,
        )

        assert control._adapt_call_count == 1
        assert control._runtime_kwargs_received == runtime_kwargs

    def test_runtime_kwargs_reach_state_control(self):
        control = MockStateControl(target_layers=[0])
        pipeline = _tiny_pipeline([control])
        pipeline.steer()

        runtime_kwargs = {"key": "value"}
        pipeline.generate(
            input_ids=torch.tensor([[3, 4, 5]]),
            max_new_tokens=1,
            runtime_kwargs=runtime_kwargs,
        )

        assert control._hooks_created
        assert control._runtime_kwargs_received == runtime_kwargs

    def test_runtime_kwargs_reach_output_control(self):
        control = MockOutputControl()
        pipeline = _tiny_pipeline([control])
        pipeline.steer()

        runtime_kwargs = {"constraint": "test"}
        pipeline.generate(
            input_ids=torch.tensor([[3, 4, 5]]),
            max_new_tokens=1,
            runtime_kwargs=runtime_kwargs,
        )

        assert control._processors_requested
        assert control._runtime_kwargs_received == runtime_kwargs

    def test_hooks_removed_after_generate(self):
        """No hooks leak onto the model once the session's execution of the work ends."""
        control = MockStateControl(target_layers=[0])
        pipeline = _tiny_pipeline([control])
        pipeline.steer()

        pipeline.generate(input_ids=torch.tensor([[3, 4, 5]]), max_new_tokens=1)

        assert len(pipeline.model.model.layers[0]._forward_pre_hooks) == 0

    def test_adapted_prompt_returned_in_output(self):
        """`Output.adapted_input_ids` reflects the input controls' token-level adaptation."""

        class _AppendControl(InputControl):
            supports_batching = True

            def __init__(self, token_id: int):
                super().__init__()
                self.token_id = token_id

            def adapt(self, input_ids, runtime_kwargs=None):
                ids = input_ids if isinstance(input_ids, torch.Tensor) else torch.as_tensor(input_ids)
                if ids.ndim == 1:
                    ids = ids.unsqueeze(0)
                suffix = torch.full((ids.size(0), 1), self.token_id, dtype=ids.dtype)
                return torch.cat([ids, suffix], dim=1)

        pipeline = _tiny_pipeline([_AppendControl(token_id=5)])
        pipeline.steer()

        prompt = torch.tensor([3, 4])
        out = pipeline.generate(
            input_ids=prompt, max_new_tokens=2, do_sample=False, return_output=True
        )

        assert out.adapted_input_ids.shape[1] == prompt.numel() + 1
        assert out.adapted_input_ids[0, -1].item() == 5


# Pipeline Compute Logprobs Tests
class TestPipelineComputeLogprobs:
    """Tests for `SteeringPipeline.compute_logprobs()`."""

    def test_compute_logprobs_requires_steer(self):
        pipeline = _tiny_pipeline()

        with pytest.raises(RuntimeError, match="steer"):
            pipeline.compute_logprobs(
                input_ids=torch.tensor([[3, 4, 5]]),
                ref_output_ids=torch.tensor([[6, 7, 8]]),
            )

    def test_compute_logprobs_requires_ref_output_ids(self):
        pipeline = _tiny_pipeline()
        pipeline.steer()

        with pytest.raises(ValueError, match="ref_output_ids"):
            pipeline.compute_logprobs(
                input_ids=torch.tensor([[3, 4, 5]]),
                ref_output_ids=None,
            )

    def test_compute_logprobs_basic_shape(self):
        pipeline = _tiny_pipeline()
        pipeline.steer()

        logprobs = pipeline.compute_logprobs(
            input_ids=torch.tensor([[3, 4, 5]]),
            ref_output_ids=torch.tensor([[6, 7, 8]]),
        )

        assert isinstance(logprobs, torch.Tensor)
        assert logprobs.shape == (1, 3)

    @pytest.mark.parametrize(
        "input_ids, ref_output_ids",
        [
            ([3, 4, 5], torch.tensor([[6, 7, 8]])),
            (torch.tensor([[3, 4, 5]]), [6, 7, 8]),
            (torch.tensor([3, 4, 5]), torch.tensor([[6, 7, 8]])),
            (torch.tensor([[3, 4, 5]]), torch.tensor([6, 7, 8])),
        ],
        ids=["list_input", "list_ref", "1d_input", "1d_ref"],
    )
    def test_compute_logprobs_normalizes_inputs(self, input_ids, ref_output_ids):
        """Lists and 1-D tensors are accepted for both arguments."""
        pipeline = _tiny_pipeline()
        pipeline.steer()

        logprobs = pipeline.compute_logprobs(
            input_ids=input_ids, ref_output_ids=ref_output_ids
        )

        assert logprobs.shape == (1, 3)

    def test_compute_logprobs_empty_ref_output_ids(self):
        pipeline = _tiny_pipeline()
        pipeline.steer()

        logprobs = pipeline.compute_logprobs(
            input_ids=torch.tensor([[3, 4, 5]]),
            ref_output_ids=torch.empty((1, 0), dtype=torch.long),
        )

        assert logprobs.shape == (1, 0)

    def test_compute_logprobs_broadcasts_ref_output_ids(self):
        """A single reference row is scored against every batch row."""
        pipeline = _tiny_pipeline()
        pipeline.steer()

        logprobs = pipeline.compute_logprobs(
            input_ids=torch.tensor([[3, 4, 5], [6, 7, 8]]),
            ref_output_ids=torch.tensor([[7, 8]]),
        )

        assert logprobs.shape == (2, 2)

    def test_compute_logprobs_batched_shape(self):
        pipeline = _tiny_pipeline()
        pipeline.steer()

        batch_size = 4
        logprobs = pipeline.compute_logprobs(
            input_ids=torch.tensor([[3, 4, 5]] * batch_size),
            ref_output_ids=torch.tensor([[6, 7, 8]] * batch_size),
        )

        assert logprobs.shape == (batch_size, 3)

    def test_compute_logprobs_values_are_nonpositive(self):
        pipeline = _tiny_pipeline()
        pipeline.steer()

        logprobs = pipeline.compute_logprobs(
            input_ids=torch.tensor([[3, 4, 5]]),
            ref_output_ids=torch.tensor([[6, 7, 8]]),
        )

        assert (logprobs <= 0).all()

    def test_compute_logprobs_matches_teacher_forced_forward(self):
        """Values equal a manual log-softmax gather over one forward pass on the combined
        sequence."""
        pipeline = _tiny_pipeline()
        pipeline.steer()

        input_ids = torch.tensor([[3, 4, 5]])
        ref_output_ids = torch.tensor([[6, 7]])

        logprobs = pipeline.compute_logprobs(
            input_ids=input_ids, ref_output_ids=ref_output_ids
        )

        with torch.no_grad():
            combined = torch.cat([input_ids, ref_output_ids], dim=1)
            logits = pipeline.model(
                input_ids=combined, attention_mask=torch.ones_like(combined)
            ).logits
        prompt_len, ref_len = input_ids.size(1), ref_output_ids.size(1)
        expected = torch.log_softmax(
            logits[:, prompt_len - 1: prompt_len + ref_len - 1, :], dim=-1
        ).gather(-1, ref_output_ids.unsqueeze(-1)).squeeze(-1)

        torch.testing.assert_close(logprobs, expected)

    def test_compute_logprobs_sequential_path_matches_batched(self):
        """A non-batch-safe control routes scoring through the sequential path; with an
        identity adaptation, values match the batched path."""
        batched = _tiny_pipeline()
        batched.steer()

        sequential = SteeringPipeline(controls=[MockInputControl()], model=batched.model, tokenizer=batched.tokenizer)
        sequential.steer()
        assert not sequential.supports_batching

        input_ids = torch.tensor([[3, 4, 5], [6, 7, 8]])
        ref_output_ids = torch.tensor([[7, 8], [4, 5]])

        expected = batched.compute_logprobs(input_ids=input_ids, ref_output_ids=ref_output_ids)
        actual = sequential.compute_logprobs(input_ids=input_ids, ref_output_ids=ref_output_ids)

        torch.testing.assert_close(actual, expected)

    def test_compute_logprobs_accepts_forward_kwargs(self):
        """Extra forward kwargs are passed through without changing the scores."""
        pipeline = _tiny_pipeline()
        pipeline.steer()

        input_ids = torch.tensor([[3, 4, 5]])
        ref_output_ids = torch.tensor([[6, 7]])

        plain = pipeline.compute_logprobs(input_ids=input_ids, ref_output_ids=ref_output_ids)
        with_kwargs = pipeline.compute_logprobs(
            input_ids=input_ids, ref_output_ids=ref_output_ids, output_hidden_states=True
        )

        torch.testing.assert_close(with_kwargs, plain)


# Supports Batching Property Tests
class TestPipelineSupportsBatching:
    """Tests for the `supports_batching` property."""

    def test_default_controls_support_batching(self):
        pipeline = SteeringPipeline(model_name_or_path="test-model", controls=[])
        assert pipeline.supports_batching

    def test_non_batching_control_disables_batching(self):
        pipeline = SteeringPipeline(model_name_or_path="test-model", controls=[MockInputControl()])
        assert not pipeline.supports_batching

    def test_all_batching_controls_enables_batching(self):
        pipeline = SteeringPipeline(model_name_or_path="test-model", controls=[MockStateControl()])
        assert pipeline.supports_batching

    def test_mixed_batching_support(self):
        pipeline = SteeringPipeline(
            model_name_or_path="test-model", controls=[MockStateControl(), MockInputControl()]
        )
        assert not pipeline.supports_batching

    def test_disabled_control_ignored_for_batching(self):
        control = MockInputControl()
        control.enabled = False

        pipeline = SteeringPipeline(model_name_or_path="test-model", controls=[control])
        assert pipeline.supports_batching


# Duplicate-BOS guard: uses a tiny hub-free model.
class TestDuplicateBosGuard:
    """`warn_if_duplicate_bos` warns once when a prompt starts with two BOS tokens."""

    def _steered_pipeline(self):
        torch.manual_seed(0)
        model = tiny_llama(num_layers=2, hidden=16, heads=2)
        tokenizer = wordlevel_tokenizer()  # bos_token_id == 0
        pipeline = SteeringPipeline(model=model, tokenizer=tokenizer)
        pipeline.steer()
        return pipeline, tokenizer

    def test_double_bos_warns_once_across_two_calls(self, caplog):
        pipeline, tokenizer = self._steered_pipeline()
        bos = tokenizer.bos_token_id
        ids = torch.tensor([[bos, bos, 3, 4]])
        with caplog.at_level(logging.WARNING, logger="steerability.utils.tokenization"):
            pipeline.generate(input_ids=ids, max_new_tokens=1)
            pipeline.generate(input_ids=ids, max_new_tokens=1)
        dup_warnings = [r for r in caplog.records if "Duplicate BOS" in r.getMessage()]
        assert len(dup_warnings) == 1  # warn-once per pipeline lifecycle

    def test_single_bos_does_not_warn(self, caplog):
        pipeline, tokenizer = self._steered_pipeline()
        bos = tokenizer.bos_token_id
        ids = torch.tensor([[bos, 3, 4]])
        with caplog.at_level(logging.WARNING, logger="steerability.utils.tokenization"):
            pipeline.generate(input_ids=ids, max_new_tokens=1)
        assert not [r for r in caplog.records if "Duplicate BOS" in r.getMessage()]

    def test_left_padded_double_bos_warns(self, caplog):
        # a left-padded batch [pad, pad, bos, bos, x] with the correct mask must still fire the
        # guard, proving the first-real-token (argmax) logic rather than a fixed position-0 check
        pipeline, tokenizer = self._steered_pipeline()
        bos = tokenizer.bos_token_id
        pad = tokenizer.pad_token_id
        ids = torch.tensor([[pad, pad, bos, bos, 3]])
        attention_mask = torch.tensor([[0, 0, 1, 1, 1]])
        with caplog.at_level(logging.WARNING, logger="steerability.utils.tokenization"):
            pipeline.generate(input_ids=ids, attention_mask=attention_mask, max_new_tokens=1)
        assert [r for r in caplog.records if "Duplicate BOS" in r.getMessage()]


class TestSameModelForwardsMetadata:
    """`same_model_forwards` is declarative component metadata on the declaring classes."""

    def test_declared_flags(self):
        from steerability.algorithms.output_control.common.logit_sources import PromptVariantSource
        from steerability.algorithms.output_control.common.values.subspace_margin import SubspaceMarginValue
        from steerability.algorithms.output_control.sasa.control import SASA

        assert SASA.same_model_forwards is True
        assert SubspaceMarginValue.same_model_forwards is True
        assert PromptVariantSource.same_model_forwards is True
        assert OutputControl.same_model_forwards is False

    def test_prompt_variant_source_construction_emits_no_warning(self):
        from steerability.algorithms.output_control.common.logit_sources import PromptVariantSource

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            PromptVariantSource(lambda text: text)


class _RecorderBackend:
    """Fake backend recording `release()` calls; release is idempotent."""

    def __init__(self):
        self.release_calls = 0
        self.spec = None

    def release(self):
        self.release_calls += 1


class TestReleaseBackends:
    """`release_backends()`, reconstruct-on-next-use, steer-failure release, and the context manager."""

    def test_release_backends_releases_and_empties_cache(self):
        pipeline = _tiny_pipeline()
        pipeline.steer()
        recorder = _RecorderBackend()
        pipeline._backends["dummy"] = recorder

        pipeline.release_backends()

        assert recorder.release_calls == 1
        assert pipeline._backends == {}

        pipeline.release_backends()  # second call is a no-op
        assert recorder.release_calls == 1

    def test_release_backends_survives_a_failing_release(self):
        class _FailingBackend(_RecorderBackend):
            def release(self):
                super().release()
                raise RuntimeError("boom")

        pipeline = _tiny_pipeline()
        pipeline.steer()
        failing = _FailingBackend()
        pipeline._backends["dummy"] = failing

        pipeline.release_backends()  # swallows the failure and empties the cache

        assert failing.release_calls == 1
        assert pipeline._backends == {}

    def test_reconstruct_on_next_use(self):
        """After releasing, the in-process backend re-adopts the live model and generate() works."""
        pipeline = _tiny_pipeline()
        pipeline.steer()
        pipeline.generate(input_ids=torch.tensor([[3, 4, 5]]), max_new_tokens=1)

        pipeline.release_backends()
        assert pipeline._backends == {}

        out = pipeline.generate(input_ids=torch.tensor([[3, 4, 5]]), max_new_tokens=1)
        assert out.shape[0] == 1

    def test_steer_failure_releases_constructed_backends(self):
        """A control whose steer() raises leaves the backend cache empty and re-raises unchanged."""

        class _RaisingInputControl(MockInputControl):
            def steer(self, model=None, tokenizer=None, **kwargs):
                raise ValueError("steer failed")

        pipeline = _tiny_pipeline([_RaisingInputControl()])

        with pytest.raises(ValueError, match="steer failed"):
            pipeline.steer()

        assert pipeline._backends == {}
        assert not pipeline._is_steered

    def test_context_manager_releases_on_exit(self):
        pipeline = _tiny_pipeline()
        pipeline.steer()
        recorder = _RecorderBackend()
        pipeline._backends["dummy"] = recorder

        with pipeline as entered:
            assert entered is pipeline
        assert recorder.release_calls == 1
        assert pipeline._backends == {}

    def test_context_manager_releases_when_body_raises(self):
        pipeline = _tiny_pipeline()
        pipeline.steer()
        recorder = _RecorderBackend()
        pipeline._backends["dummy"] = recorder

        with pytest.raises(RuntimeError, match="body error"):
            with pipeline:
                raise RuntimeError("body error")
        assert recorder.release_calls == 1


class TestProvenanceMismatchWarnings:
    """`_warn_on_provenance_mismatch` against a serving engine's model block."""

    @staticmethod
    def _control_with_meta(meta):
        artifact = MagicMock()
        artifact.meta = meta
        control = MagicMock()
        control._steering_vector = artifact
        return control

    @staticmethod
    def _absent_fingerprint():
        try:
            from vllm_hook_plugins.core.fingerprints import chat_template_fingerprint
        except ImportError:
            import hashlib
            return f"sha256:{hashlib.sha256(b'').hexdigest()}"
        return chat_template_fingerprint(None)

    def test_differing_chat_template_fingerprints_warn(self):
        control = self._control_with_meta({"chat_template_fingerprint": "sha256:aaa"})
        with pytest.warns(UserWarning, match="chat_template_fingerprint"):
            _warn_on_provenance_mismatch(
                control, {"chat_template_fingerprint": "sha256:bbb"},
            )

    def test_absent_served_chat_template_fingerprint_does_not_warn(self):
        control = self._control_with_meta({"chat_template_fingerprint": "sha256:aaa"})
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _warn_on_provenance_mismatch(
                control, {"chat_template_fingerprint": self._absent_fingerprint()},
            )

    def test_differing_config_fingerprints_still_warn(self):
        control = self._control_with_meta({"config_fingerprint": "sha256:aaa"})
        with pytest.warns(UserWarning, match="config_fingerprint"):
            _warn_on_provenance_mismatch(
                control, {"config_fingerprint": "sha256:bbb"},
            )
