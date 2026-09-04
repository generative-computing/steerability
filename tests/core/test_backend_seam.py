"""Tests for the execution seam: `BackendSpec`, capability tables, the requirement language,
and `SteeringPipeline.check()` with its steer-time enforcement."""
import dataclasses
import importlib.util
from pathlib import Path

import pytest
import torch

from aisteer360.algorithms.core.execution import (
    Backend,
    BackendCapabilities,
    BackendSpec,
    Capability,
    GenerationParams,
    InterventionKinds,
    InterventionSpec,
    ModelAccess,
    Requirements,
    SupportFailure,
    UnsupportedPipelineError,
    any_of,
    capabilities_for_spec,
    needs,
)
from aisteer360.algorithms.core.execution.session_utils import ScopedSession
from aisteer360.algorithms.core.steering_pipeline import SteeringPipeline
from aisteer360.algorithms.input_control.base import InputControl
from aisteer360.algorithms.output_control.base import OutputControl
from aisteer360.algorithms.state_control.pasta import PASTA
from aisteer360.algorithms.structural_control.base import StructuralControl
from aisteer360.backends.huggingface import ExclusiveSession
from tests.utils.tiny_models import tiny_llama, wordlevel_tokenizer


class _TokenPassthroughControl(InputControl):
    """Enabled input control declaring the conservative in-process generate requirement
    (the `BaseControl` default; the `InputControl` base is prompt-only and requires nothing)."""

    def __init__(self):
        self._steer_called = False
        self._steer_kwargs = None

    def adapt(self, input_ids, runtime_kwargs=None):
        return input_ids

    def steer(self, model=None, tokenizer=None, **kwargs):
        self._steer_called = True
        self._steer_kwargs = kwargs

    def requirements(self) -> Requirements:
        return Requirements(generate=needs(Capability.IN_PROCESS_TORCH))


class _ModelSwappingControl(StructuralControl):
    """Structural control that replaces the pipeline model with a fresh three-layer model."""

    Args = None

    def steer(self, model, tokenizer=None, **kwargs):
        return tiny_llama(num_layers=3, hidden=16, heads=2)


class _LayoutReadingControl(InputControl):
    """Input control that records the session layout observed during its steer phase."""

    def __init__(self):
        self.observed_num_layers = None

    def adapt(self, input_ids, runtime_kwargs=None):
        return input_ids

    def steer(self, model=None, tokenizer=None, session=None, **kwargs):
        self.observed_num_layers = session.layout.num_layers


class TestBackendSpec:

    def test_option_order_does_not_change_identity(self):
        first = BackendSpec(kind="huggingface", model="m", options={"a": 1, "b": {"c": 2, "d": 3}})
        second = BackendSpec(kind="huggingface", model="m", options={"b": {"d": 3, "c": 2}, "a": 1})
        assert first == second
        assert first.spec_hash == second.spec_hash

    def test_dtype_device_and_path_canonicalize_to_strings(self):
        spec = BackendSpec(
            kind="huggingface",
            model=Path("/tmp/model"),
            options={"hf_model_kwargs": {"torch_dtype": torch.bfloat16}, "device": torch.device("cpu")},
        )
        assert spec.model == "/tmp/model"
        assert spec.get_option("hf_model_kwargs", "torch_dtype") == "bfloat16"
        assert spec.get_option("device") == "cpu"

    def test_unknown_kind_rejected(self):
        with pytest.raises(ValueError, match="Unknown backend kind"):
            BackendSpec(kind="openai", model="m")

    def test_frozen(self):
        spec = BackendSpec(kind="vllm", model="m")
        with pytest.raises(dataclasses.FrozenInstanceError):
            spec.model = "other"

    def test_get_option_default_and_options_dict(self):
        spec = BackendSpec(kind="vllm", model="m", options={"engine_kwargs": {"max_model_len": 2048}})
        assert spec.get_option("engine_kwargs", "max_model_len") == 2048
        assert spec.get_option("engine_kwargs", "absent") is None
        assert spec.get_option("absent", default=7) == 7
        assert spec.options_dict() == {"engine_kwargs": {"max_model_len": 2048}}

    def test_empty_options_equal_default(self):
        assert BackendSpec(kind="vllm", model="m") == BackendSpec(kind="vllm", model="m", options={})

    @pytest.mark.parametrize("kind", ["vllm", "vllm-serve"])
    @pytest.mark.parametrize("options", [
        {"hook_plugin": True, "speculative_config": {"model": "draft"}},
        {"hook_plugin": True, "engine_kwargs": {"speculative_config": {"model": "draft"}}},
    ])
    def test_speculative_decoding_with_plugin_rejected(self, kind, options):
        with pytest.raises(ValueError, match="[Ss]peculative decoding"):
            BackendSpec(kind=kind, model="m", options=options)

    def test_speculative_decoding_without_plugin_allowed(self):
        spec = BackendSpec(kind="vllm", model="m", options={"speculative_config": {"model": "draft"}})
        assert spec.get_option("speculative_config") == {"model": "draft"}

    def test_integer_mapping_keys_round_trip(self):
        spec = BackendSpec(
            kind="huggingface", model="m",
            options={"hf_model_kwargs": {"max_memory": {0: "10GiB", "cpu": "2GiB"}}},
        )
        assert spec.get_option("hf_model_kwargs", "max_memory") == {0: "10GiB", "cpu": "2GiB"}

    def test_mixed_key_types_construct_and_stay_distinct(self):
        spec = BackendSpec(kind="huggingface", model="m", options={"m": {1: {"a": 1}, "1": "x"}})
        assert spec.get_option("m", 1) == {"a": 1}
        assert spec.get_option("m", "1") == "x"

    def test_equality_and_spec_hash_agree_on_value_types(self):
        flag_bool = BackendSpec(kind="vllm", model="m", options={"flag": True})
        flag_int = BackendSpec(kind="vllm", model="m", options={"flag": 1})
        assert flag_bool != flag_int
        assert hash(flag_bool) != hash(flag_int)
        assert flag_bool.spec_hash != flag_int.spec_hash
        assert flag_bool == BackendSpec(kind="vllm", model="m", options={"flag": True})


class TestCapabilityTables:

    def test_huggingface_atoms(self):
        capabilities = capabilities_for_spec(BackendSpec(kind="huggingface", model="m"))
        assert capabilities.atoms == frozenset({
            Capability.IN_PROCESS_TORCH,
            Capability.HIDDEN_CAPTURE,
            Capability.BEAM_PROPOSALS,
        })
        assert capabilities.capture_kinds is not None
        assert "layer_input" in capabilities.capture_kinds.locations

    def test_vllm_baseline_atoms(self):
        capabilities = capabilities_for_spec(BackendSpec(kind="vllm", model="m"))
        assert capabilities.atoms == frozenset({
            Capability.SERVE_CHECKPOINT,
            Capability.SERVE_LORA,
            Capability.GUIDED_DECODING,
        })
        assert capabilities.intervention_kinds is None
        assert capabilities.constraint_kinds.constraints == frozenset(
            {"json_schema", "regex", "grammar", "choice"}
        )

    def test_vllm_plugin_adds_interventions_and_offline_capture(self):
        capabilities = capabilities_for_spec(
            BackendSpec(kind="vllm", model="m", options={"hook_plugin": True})
        )
        assert Capability.INTERVENTION_SPECS in capabilities.atoms
        assert Capability.HIDDEN_CAPTURE in capabilities.atoms
        assert Capability.IN_PROCESS_TORCH not in capabilities.atoms
        assert "additive" in capabilities.intervention_kinds.transforms
        assert "affine" in capabilities.intervention_kinds.readouts
        assert "sum_threshold" in capabilities.intervention_kinds.rules
        # no processor kinds are advertised until a control exports a ProcessorSpec
        assert Capability.PER_STEP_LOGIT_SPECS not in capabilities.atoms
        assert capabilities.processor_kinds is None

    def test_vllm_serve_plugin_has_no_hidden_capture(self):
        capabilities = capabilities_for_spec(
            BackendSpec(kind="vllm-serve", model="m", options={"hook_plugin": True})
        )
        assert Capability.INTERVENTION_SPECS in capabilities.atoms
        assert Capability.HIDDEN_CAPTURE not in capabilities.atoms


class TestRequirementLanguage:

    def test_needs_atoms_satisfaction(self):
        requirement = needs(Capability.IN_PROCESS_TORCH)
        hf = capabilities_for_spec(BackendSpec(kind="huggingface", model="m"))
        vllm = capabilities_for_spec(BackendSpec(kind="vllm", model="m"))
        assert requirement[0].satisfied_by(hf)
        assert not requirement[0].satisfied_by(vllm)
        assert requirement[0].missing(vllm) == ["IN_PROCESS_TORCH"]

    def test_any_of_satisfied_by_either_alternative(self):
        requirement = any_of(
            needs(Capability.IN_PROCESS_TORCH),
            needs(
                Capability.INTERVENTION_SPECS,
                kinds=InterventionKinds(
                    transforms=frozenset({"additive"}), scopes=frozenset({"after_prompt"}),
                ),
            ),
        )
        hf = capabilities_for_spec(BackendSpec(kind="huggingface", model="m"))
        vllm_plugin = capabilities_for_spec(
            BackendSpec(kind="vllm", model="m", options={"hook_plugin": True})
        )
        vllm_bare = capabilities_for_spec(BackendSpec(kind="vllm", model="m"))
        assert any(alternative.satisfied_by(hf) for alternative in requirement)
        assert any(alternative.satisfied_by(vllm_plugin) for alternative in requirement)
        assert not any(alternative.satisfied_by(vllm_bare) for alternative in requirement)

    def test_kind_containment_rejects_unadvertised_kind(self):
        alternative = needs(
            Capability.INTERVENTION_SPECS,
            kinds=InterventionKinds(transforms=frozenset({"a_new_transform"})),
        )[0]
        vllm_plugin = capabilities_for_spec(
            BackendSpec(kind="vllm", model="m", options={"hook_plugin": True})
        )
        assert not alternative.satisfied_by(vllm_plugin)

    def test_base_control_default_requirements(self):
        control = _TokenPassthroughControl()
        requirements = control.requirements()
        assert requirements.score == ()
        assert len(requirements.generate) == 1
        assert requirements.generate[0].atoms == frozenset({Capability.IN_PROCESS_TORCH})
        assert control.steer_access() is ModelAccess.FACTS

    def test_output_control_score_requirement_follows_include_in_scoring(self):
        class _StepControl(OutputControl):
            pass

        scoring = _StepControl()
        assert scoring.requirements().score
        non_scoring = _StepControl()
        non_scoring.include_in_scoring = False
        assert non_scoring.requirements().score == ()

    def test_structural_control_declares_module_access(self):
        control = _ModelSwappingControl()
        assert control.steer_access() is ModelAccess.MODULE
        assert control.requirements().generate[0].atoms == frozenset({Capability.IN_PROCESS_TORCH})

    def test_unknown_phase_rejected(self):
        with pytest.raises(ValueError, match="Unknown phase"):
            Requirements().for_phase("deploy")


class TestGenerationParamsNormalization:

    def test_from_gen_kwargs_split(self):
        params = GenerationParams.from_gen_kwargs(
            max_new_tokens=5, do_sample=False, num_return_sequences=2, foo="bar",
        )
        assert params.max_new_tokens == 5
        assert params.greedy is True
        assert params.n == 2
        assert params.extra == {"foo": "bar"}


class TestCheck:

    def test_defaults_only_pipeline_supported_on_vllm(self):
        pipeline = SteeringPipeline(model_name_or_path="m")
        report = pipeline.check(backend=BackendSpec(kind="vllm", model="m"))
        assert report.ok
        assert report.supported("generate", "score")
        assert report.plan.steps == ()
        assert report.plan.stages is False

    def test_enabled_control_unsupported_on_vllm_with_stable_message(self):
        pipeline = SteeringPipeline(model_name_or_path="m", controls=[_TokenPassthroughControl()])
        report = pipeline.check(backend=BackendSpec(kind="vllm", model="m"))
        assert not report.ok
        assert len(report.failures) == 1
        failure = report.failures[0]
        assert failure.control == "_TokenPassthroughControl"
        assert failure.phase == "generate"
        assert failure.message == (
            "_TokenPassthroughControl is unsupported at generate on backend kind 'vllm': "
            "missing IN_PROCESS_TORCH; run this pipeline on the huggingface backend."
        )

    def test_default_hf_backend_supported(self):
        pipeline = SteeringPipeline(model_name_or_path="m", controls=[_TokenPassthroughControl()])
        assert pipeline.check().ok

    def test_steer_raises_before_any_control_runs(self):
        control = _TokenPassthroughControl()
        pipeline = SteeringPipeline(
            controls=[control], backend=BackendSpec(kind="vllm", model="m"),
        )
        with pytest.raises(UnsupportedPipelineError, match="IN_PROCESS_TORCH"):
            pipeline.steer()
        assert control._steer_called is False

    @pytest.mark.skipif(
        importlib.util.find_spec("vllm") is not None,
        reason="vLLM installed; steer() would boot an engine instead of raising.",
    )
    def test_steer_on_vllm_backend_requires_vllm_extra(self):
        pipeline = SteeringPipeline(backend=BackendSpec(kind="vllm", model="m"))
        with pytest.raises(ModuleNotFoundError, match=r"aisteer360\[vllm\]"):
            pipeline.steer()

    def test_compute_logprobs_raises_on_score_failure(self):
        pipeline = SteeringPipeline(
            controls=[], model=tiny_llama(num_layers=2, hidden=16, heads=2), tokenizer=wordlevel_tokenizer(),
        )
        pipeline.steer()
        pipeline._support_report = dataclasses.replace(
            pipeline._support_report,
            failures=(SupportFailure(
                control="_StepControl", phase="score", message="synthetic score failure",
            ),),
        )
        with pytest.raises(UnsupportedPipelineError, match="synthetic score failure"):
            pipeline.compute_logprobs(input_ids=[3, 4], ref_output_ids=[5])

    def test_invalid_backend_value_rejected(self):
        pipeline = SteeringPipeline(model_name_or_path="m")
        with pytest.raises(TypeError, match="backend must be"):
            pipeline.check(backend=3.14)

    def test_removed_constructor_parameters_rejected(self):
        for removed in ("steer" + "_backend", "inference" + "_backend"):
            with pytest.raises(TypeError):
                SteeringPipeline(**{removed: "huggingface"})


class TestPastaSpecConstraint:

    def _pasta_pipeline(self, attn_implementation):
        hf_model_kwargs = (
            {"attn_implementation": attn_implementation} if attn_implementation else {}
        )
        return SteeringPipeline(
            model_name_or_path="m",
            controls=[PASTA(head_config=[0])],
            hf_model_kwargs=hf_model_kwargs,
        )

    def test_check_reports_flash_attention_conflict(self):
        pipeline = self._pasta_pipeline("flash_attention_2")
        report = pipeline.check()
        assert not report.ok
        failure = report.failures[0]
        assert failure.control == "PASTA"
        assert failure.phase == "generate"
        assert "eager" in failure.message
        assert "attn_implementation" in failure.message

    def test_steer_raises_on_flash_attention_config(self):
        pipeline = self._pasta_pipeline("flash_attention_2")
        with pytest.raises(UnsupportedPipelineError, match="eager"):
            pipeline.steer()

    @pytest.mark.parametrize("attn_implementation", [None, "eager", "sdpa"])
    def test_supported_attention_configurations_pass(self, attn_implementation):
        assert self._pasta_pipeline(attn_implementation).check().ok

    def test_vllm_verdict_is_capability_not_constraint(self):
        pipeline = self._pasta_pipeline(None)
        report = pipeline.check(backend=BackendSpec(kind="vllm", model="m"))
        assert len(report.failures) == 1
        assert "IN_PROCESS_TORCH" in report.failures[0].message


class TestSteerSessionPlumbing:

    def _steered_pipeline(self, controls, **steer_kwargs):
        pipeline = SteeringPipeline(
            controls=controls, model=tiny_llama(num_layers=2, hidden=16, heads=2), tokenizer=wordlevel_tokenizer(),
        )
        pipeline.steer(**steer_kwargs)
        return pipeline

    def test_controls_receive_scoped_session_over_exclusive(self):
        control = _TokenPassthroughControl()
        self._steered_pipeline([control])
        session = control._steer_kwargs["session"]
        assert isinstance(session, ScopedSession)
        assert isinstance(session.inner, ExclusiveSession)

    def test_layout_reflects_model(self):
        control = _LayoutReadingControl()
        self._steered_pipeline([control])
        assert control.observed_num_layers == 2

    def test_caller_supplied_session_wins(self):
        control = _TokenPassthroughControl()
        self._steered_pipeline([control], session="sentinel")
        assert control._steer_kwargs["session"] == "sentinel"

    def test_session_closed_after_steer(self):
        control = _TokenPassthroughControl()
        self._steered_pipeline([control])
        session = control._steer_kwargs["session"]
        assert session.inner.closed
        with pytest.raises(RuntimeError, match="closed"):
            _ = session.layout

    def test_structural_replacement_visible_through_session(self):
        swapper = _ModelSwappingControl()
        reader = _LayoutReadingControl()
        pipeline = self._steered_pipeline([swapper, reader])
        assert reader.observed_num_layers == 3
        assert pipeline.model.config.num_hidden_layers == 3

    def test_intervention_spec_canonical_is_deterministic(self):
        spec = InterventionSpec(ops=({"layers": [1], "transform": {"kind": "additive"}},))
        assert spec.canonical() == spec.canonical()


class _MinimalBackend(Backend):
    """Concrete backend implementing only the two abstract members trivially."""

    def __init__(self, spec: BackendSpec) -> None:
        self.spec = spec

    @classmethod
    def capabilities_for_spec(cls, spec: BackendSpec) -> BackendCapabilities:
        return BackendCapabilities(atoms=frozenset())

    def open_session(self):
        raise NotImplementedError


class TestBackendRelease:
    """The `Backend.release()` lifecycle default and the serve backend's inheritance of it."""

    def test_default_release_is_a_noop_and_idempotent(self):
        backend = _MinimalBackend(BackendSpec(kind="huggingface", model="m"))
        backend.release()
        backend.release()

    def test_vllm_serve_inherits_the_noop_default(self):
        from aisteer360.backends.vllm import VLLMServeBackend

        assert VLLMServeBackend.release is Backend.release
