"""The steer plan: venue matrix, fit venues, notices, determinism, and the in-process plan."""
import pytest
import torch

from steerability.algorithms.core.execution import BackendSpec, ModelAccess
from steerability.algorithms.core.internals.probes import ProbeSetFit
from steerability.algorithms.core.internals.probes.fitting import ProbeFitSpec
from steerability.algorithms.core.steering_pipeline import SteeringPipeline
from steerability.algorithms.input_control.base import InputControl
from steerability.algorithms.output_control.routed_decoding import P, Route, RoutedDecoding, Router
from steerability.algorithms.output_control.routed_decoding.actions import respond
from steerability.algorithms.state_control.caa.control import CAA
from steerability.algorithms.state_control.common.steering_vector import SteeringVector

PAIRS = {"prompts": ["q"], "positives": ["a"], "negatives": ["b"]}

HF_SPEC = BackendSpec(kind="huggingface", model="m")
VLLM_PLUGIN_SPEC = BackendSpec(kind="vllm", model="m", options={"hook_plugin": True})
VLLM_BARE_SPEC = BackendSpec(kind="vllm", model="m")
SERVE_PLUGIN_SPEC = BackendSpec(kind="vllm-serve", model="m", options={"hook_plugin": True})


def _access_control(access):
    class _Declared(InputControl):
        def adapt(self, input_ids, runtime_kwargs=None):
            return input_ids

        def steer_access(self):
            return access

    _Declared.__name__ = f"_{access.name.title()}Control"
    return _Declared()


def _fit_caa() -> CAA:
    return CAA(data=PAIRS, layer_id=1)


def _precomputed_caa() -> CAA:
    vector = SteeringVector(model_type="llama", directions={1: torch.zeros(1, 16)})
    return CAA(steering_vector=vector, layer_id=1)


def _routed_fit() -> RoutedDecoding:
    return RoutedDecoding(
        probes=ProbeSetFit(data={"p": PAIRS}, spec=ProbeFitSpec(method="mean_diff")),
        rules=Router(routes=[Route("r", when=P("p"), action=respond("x"))]),
    )


class TestVenueMatrix:

    @pytest.mark.parametrize("access,expected", [
        (ModelAccess.FACTS, "session"),
        (ModelAccess.ROLLOUTS, "session"),
        (ModelAccess.MODULE, "stage"),
    ])
    def test_engine_venues_below_and_above_capture(self, access, expected):
        pipeline = SteeringPipeline(model_name_or_path="m", controls=[_access_control(access)])
        (step,) = pipeline.check(backend=VLLM_PLUGIN_SPEC).plan.steps
        assert step.access is access
        assert step.venue == expected

    def test_capture_rides_the_session_where_advertised(self):
        pipeline = SteeringPipeline(model_name_or_path="m", controls=[_fit_caa()])
        (step,) = pipeline.check(backend=VLLM_PLUGIN_SPEC).plan.steps
        assert step.access is ModelAccess.CAPTURE
        assert step.venue == "session"

    @pytest.mark.parametrize("spec", [VLLM_BARE_SPEC, SERVE_PLUGIN_SPEC])
    def test_capture_stages_where_capture_is_statically_absent(self, spec):
        pipeline = SteeringPipeline(model_name_or_path="m", controls=[_fit_caa()])
        report = pipeline.check(backend=spec)
        (step,) = report.plan.steps
        assert step.venue == "stage"
        assert report.plan.stages is True

    def test_fit_in_process_forces_capture_to_the_stage(self):
        pipeline = SteeringPipeline(model_name_or_path="m", controls=[_fit_caa()], fit="in_process")
        report = pipeline.check(backend=VLLM_PLUGIN_SPEC)
        (step,) = report.plan.steps
        assert step.venue == "stage"
        (fit,) = report.plan.fits
        assert fit.venue == "stage"
        assert report.plan.notices == ()  # direction fits cross without notices

    def test_hugging_face_plan_is_all_live(self):
        pipeline = SteeringPipeline(
            model_name_or_path="m",controls=[_fit_caa(), _access_control(ModelAccess.MODULE)],
        )
        plan = pipeline.check(backend=HF_SPEC).plan
        assert all(step.venue == "live" for step in plan.steps)
        assert all(fit.venue == "live" for fit in plan.fits)
        assert plan.stages is False
        assert plan.notices == ()


class TestFitsAndNotices:

    def test_direction_fit_venue_follows_its_step(self):
        pipeline = SteeringPipeline(model_name_or_path="m", controls=[_fit_caa()])
        plan = pipeline.check(backend=VLLM_PLUGIN_SPEC).plan
        (fit,) = plan.fits
        assert fit.control == "CAA"
        assert fit.artifact == "ContrastiveFit"
        assert fit.artifact_class == "direction"
        assert fit.venue == "session"

    def test_calibrated_fit_on_serve_emits_the_crossing_notice(self):
        pipeline = SteeringPipeline(model_name_or_path="m", controls=[_routed_fit()])
        plan = pipeline.check(backend=SERVE_PLUGIN_SPEC).plan
        (fit,) = plan.fits
        assert fit.artifact_class == "calibrated"
        assert fit.venue == "stage"
        assert plan.notices == (
            "ProbeSetFit for RoutedDecoding is scale-calibrated and will be read on backend "
            "kind 'vllm-serve', but is fitted in process (capture is unavailable on this "
            "backend); calibrated thresholds may shift across execution boundaries.",
        )

    def test_fit_in_process_flag_names_itself_in_the_notice(self):
        pipeline = SteeringPipeline(model_name_or_path="m", controls=[_routed_fit()], fit="in_process")
        plan = pipeline.check(backend=VLLM_PLUGIN_SPEC).plan
        assert plan.notices == (
            "ProbeSetFit for RoutedDecoding is scale-calibrated and will be read on backend "
            "kind 'vllm', but is fitted in process (fit='in_process'); calibrated thresholds "
            "may shift across execution boundaries.",
        )

    def test_session_calibrated_fit_carries_no_notice(self):
        pipeline = SteeringPipeline(model_name_or_path="m", controls=[_routed_fit()])
        plan = pipeline.check(backend=VLLM_PLUGIN_SPEC).plan
        (fit,) = plan.fits
        assert fit.venue == "session"
        assert plan.notices == ()


class TestDeterminism:

    def test_same_configuration_yields_the_same_plan_and_verdicts(self):
        def build():
            return SteeringPipeline(
                model_name_or_path="m",controls=[_precomputed_caa(), _routed_fit()], fit="in_process",
            )

        first = build().check(backend=SERVE_PLUGIN_SPEC)
        second = build().check(backend=SERVE_PLUGIN_SPEC)
        assert first.plan == second.plan
        assert first.failures == second.failures

    def test_plan_is_independent_of_sibling_controls(self):
        """A control's venue never moves because an unrelated control was added."""
        alone = SteeringPipeline(model_name_or_path="m", controls=[_fit_caa()])
        (step_alone,) = alone.check(backend=VLLM_PLUGIN_SPEC).plan.steps
        with_module_sibling = SteeringPipeline(
            model_name_or_path="m",controls=[_fit_caa(), _access_control(ModelAccess.MODULE)],
        )
        plan = with_module_sibling.check(backend=VLLM_PLUGIN_SPEC).plan
        (step_with,) = [step for step in plan.steps if step.control == "CAA"]
        assert step_alone.venue == step_with.venue == "session"
