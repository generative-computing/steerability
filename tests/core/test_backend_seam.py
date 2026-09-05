"""Tests for the execution seam: `BackendSpec`, capability tables, the requirement language,
and `SteeringPipeline.check()` with its steer-time enforcement."""
import dataclasses
import importlib.util
from pathlib import Path

import pytest
import torch

from steerability.algorithms.core.execution import (
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
from steerability.algorithms.core.execution.session_utils import ScopedSession
from steerability.algorithms.core.steering_pipeline import SteeringPipeline
from steerability.algorithms.input_control.base import InputControl
from steerability.algorithms.output_control.base import OutputControl
from steerability.algorithms.state_control.pasta import PASTA
from steerability.algorithms.structural_control.base import StructuralControl
from steerability.backends.huggingface import ExclusiveSession
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
        with pytest.raises(ModuleNotFoundError, match=r"steerability\[vllm\]"):
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
        from steerability.backends.vllm import VLLMServeBackend

        assert VLLMServeBackend.release is Backend.release


# plain serve spec: a vLLM server with no hook_plugin, so it advertises neither INTERVENTION_SPECS
# nor HIDDEN_CAPTURE. A non-huggingface spec needs no model_name_or_path, capabilities_for_spec is
# static, and steerability.backends.vllm imports without vLLM, so these need no server, model, or GPU.
_PLAIN_SERVE_SPEC = BackendSpec(kind="vllm-serve", model="tiny")

_ZERO_SCORER = lambda prompt, continuations, params: [0.0] * len(continuations)  # noqa: E731


def _system_prompt():
    from steerability.algorithms.input_control.system_prompt.control import SystemPrompt
    return SystemPrompt(text="be brief")


def _user_prefix():
    from steerability.algorithms.input_control.user_prefix.control import UserPrefix
    return UserPrefix(text="Note: ")


def _few_shot():
    from steerability.algorithms.input_control.few_shot.control import FewShot
    return FewShot(directive="d", positive_example_pool=[{"prompt": "a", "response": "b"}], k_positive=1)


def _prewrite():
    from steerability.algorithms.input_control.prewrite.control import PRewrite
    return PRewrite(initial_instruction="be helpful", strategy="inference",
                    rewriter_gen_kwargs={"max_new_tokens": 4, "do_sample": False})


def _gepa():
    from steerability.algorithms.input_control.gepa.control import GEPA
    return GEPA(seed_instruction="be helpful", train_set=[{"input": "hi"}],
                row_scorer=lambda out, row: 0.5, budget=4)


def _cpo_with_prompt_lm():
    from steerability.algorithms.input_control.cpo.control import CPO
    return CPO(
        seed_prompt="be helpful", train_dataset=[{"query": "hi"}],
        row_scorer=lambda out, row: 0.5, prompt_lm="some/model",
    )


def _cpo_without_prompt_lm():
    from steerability.algorithms.input_control.cpo.control import CPO

    # offline_data avoids train-time generation, so prompt_lm stays unset: the live model is bound
    # as the proposer at steer, so generate requires IN_PROCESS_TORCH and steer_access is MODULE
    return CPO(seed_prompt="be helpful", offline_data=[{"query": "hi", "prompt": "p", "reward": 1.0}])


def _stopping_rules():
    from steerability.algorithms.output_control.stopping_rules.control import StoppingRules
    return StoppingRules(stop_texts=["x"])


def _phased_decoding():
    from steerability.algorithms.output_control.phased_decoding.control import PhasedDecoding
    return PhasedDecoding(plan=[{"generate": {}}])


def _budget_forcing():
    from steerability.algorithms.output_control.budget_forcing.control import BudgetForcing
    return BudgetForcing(max_thinking_tokens=4)


def _best_of_n():
    from steerability.algorithms.output_control.best_of_n.control import BestOfN
    return BestOfN(n=4, scorer=_ZERO_SCORER)


def _search_sample():
    from steerability.algorithms.output_control.search_decoding.control import SearchDecoding
    return SearchDecoding(scorer=_ZERO_SCORER, propose_mode="sample")


def _search_beam():
    from steerability.algorithms.output_control.search_decoding.control import SearchDecoding
    return SearchDecoding(scorer=_ZERO_SCORER, num_candidates=2, propose_mode="beam")


def _deal():
    from steerability.algorithms.output_control.deal.control import DeAL
    return DeAL(reward_func=_ZERO_SCORER)


def _constrained_source():
    from steerability.algorithms.output_control.constrained_decoding.control import ConstrainedDecoding
    return ConstrainedDecoding(regex="a+", include_in_scoring=False)


def _constrained_automaton():
    from steerability.algorithms.output_control.constrained_decoding.control import ConstrainedDecoding
    return ConstrainedDecoding(automaton=object())


def _rad():
    from steerability.algorithms.output_control.rad.control import RAD
    return RAD(reward_model_id="unused", beta=0.1)


def _sasa():
    from steerability.algorithms.output_control.sasa.control import SASA
    return SASA(beta=0.1)


def _dexperts():
    from steerability.algorithms.output_control.dexperts.control import DExperts
    return DExperts(expert_name_or_path="e", anti_expert_name_or_path="a", alpha=0.5)


def _contrastive_decoding():
    from steerability.algorithms.output_control.contrastive_decoding.control import ContrastiveDecoding
    return ContrastiveDecoding(amateur_name_or_path="a", alpha=0.5)


def _contrastive_guidance():
    from steerability.algorithms.output_control.contrastive_guidance.control import ContrastiveGuidance
    return ContrastiveGuidance(sources=["a"], weights=[1.0])


def _value_guidance():
    from steerability.algorithms.output_control.value_guidance.control import ValueGuidance
    return ValueGuidance(value=lambda ctx: 0.0, policy="top_k", k=5)


def _routed_decoding_fit():
    from steerability.algorithms.core.internals.probes import ProbeSetFit
    from steerability.algorithms.core.internals.probes.fitting import ProbeFitSpec
    from steerability.algorithms.output_control.routed_decoding import P, Route, RoutedDecoding, Router
    from steerability.algorithms.output_control.routed_decoding.actions import respond
    pairs = [{"prompt": "q", "positive": "a", "negative": "b"}]
    rules = Router(routes=[Route("r", when=P("p"), action=respond("x"))])
    return RoutedDecoding(probes=ProbeSetFit(data={"p": pairs}, spec=ProbeFitSpec(method="mean_diff")), rules=rules)


def _pasta():
    return PASTA(head_config=[0])


class TestServeSupportBoundary:
    """Pin the generate-phase serve-support boundary and `steer_access()` of shipped controls on a
    plain `vllm-serve` spec, so a refactor cannot move a control across the Black-box tier line
    silently. A Black-box arm needs `supported("generate")` and `steer_access() <= ROLLOUTS`."""

    @pytest.mark.parametrize("factory, access", [
        (_system_prompt, ModelAccess.FACTS),
        (_user_prefix, ModelAccess.FACTS),
        (_few_shot, ModelAccess.FACTS),
        (_prewrite, ModelAccess.ROLLOUTS),
        (_gepa, ModelAccess.ROLLOUTS),
        (_cpo_with_prompt_lm, ModelAccess.ROLLOUTS),
        (_stopping_rules, ModelAccess.FACTS),
        (_phased_decoding, ModelAccess.FACTS),
        (_budget_forcing, ModelAccess.FACTS),
        (_best_of_n, ModelAccess.FACTS),
        (_search_sample, ModelAccess.FACTS),
        (_constrained_source, ModelAccess.FACTS),
    ])
    def test_serve_supported_controls(self, factory, access):
        control = factory()
        report = SteeringPipeline(controls=[control], backend=_PLAIN_SERVE_SPEC).check()
        assert report.supported("generate") is True
        assert control.steer_access() == access

    @pytest.mark.parametrize("factory, access", [
        (_cpo_without_prompt_lm, ModelAccess.MODULE),
        (_deal, ModelAccess.FACTS),
        (_search_beam, ModelAccess.FACTS),
        (_routed_decoding_fit, ModelAccess.CAPTURE),
        (_constrained_automaton, ModelAccess.FACTS),
        (_rad, ModelAccess.MODULE),
        (_sasa, ModelAccess.MODULE),
        (_dexperts, ModelAccess.MODULE),
        (_contrastive_decoding, ModelAccess.MODULE),
        (_contrastive_guidance, ModelAccess.MODULE),
        (_value_guidance, ModelAccess.MODULE),
        (_pasta, ModelAccess.MODULE),
    ])
    def test_serve_unsupported_controls(self, factory, access):
        control = factory()
        report = SteeringPipeline(controls=[control], backend=_PLAIN_SERVE_SPEC).check()
        assert report.supported("generate") is False
        assert control.steer_access() == access

    def test_constrained_decoding_default_scoring_fails_score_only(self):
        from steerability.algorithms.output_control.constrained_decoding.control import ConstrainedDecoding

        control = ConstrainedDecoding(regex="a+")  # include_in_scoring=True by default
        report = SteeringPipeline(controls=[control], backend=_PLAIN_SERVE_SPEC).check()
        # the tier rule reads the generate phase only; score fails on serve at the default
        assert report.supported("generate") is True
        assert report.supported("score") is False
