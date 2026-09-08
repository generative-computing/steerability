"""Phase-derived requirements and plans for intervention controls.

Pins the derived declarations: the steer plan stages a fit-carrying template on a capture-less
backend and keeps a precomputed template on the session, generate offers the intervention-spec
alternative exactly when every component has a wire form, and score is in-process (remote
prompt-logprob scoring anchors token scopes at the request's prompt end). Also pins the eager
steer-time lowering failure naming the intervention and reason.
"""
import pytest
import torch

from steerability.algorithms.core.execution import BackendSpec, Capability, ModelAccess, ModelFacts
from steerability.algorithms.core.steering_pipeline import SteeringPipeline
from steerability.algorithms.state_control.caa.control import CAA
from steerability.algorithms.state_control.common.steering_vector import SteeringVector

HIDDEN = 16
LAYERS = 4

pytest.importorskip("vllm_hook_plugins")

SERVE_SPEC = BackendSpec(kind="vllm-serve", model="tiny", options={
    "base_url": "http://localhost:9", "hook_plugin": True,
})


def _vector(k: int = 1) -> SteeringVector:
    generator = torch.Generator().manual_seed(3)
    return SteeringVector(
        model_type="llama",
        directions={1: torch.randn(k, HIDDEN, generator=generator)},
    )


def _fit_caa() -> CAA:
    return CAA(data={"prompts": ["q"], "positives": ["a"], "negatives": ["b"]}, layer_id=1)


class TestPhaseVerdicts:

    def test_fit_template_stages_on_a_capture_less_backend(self):
        """A template carrying a fit source plans a staged fit where capture is absent."""
        pipeline = SteeringPipeline(model_name_or_path="m", controls=[_fit_caa()])
        report = pipeline.check(backend=SERVE_SPEC)
        (step,) = report.plan.steps
        assert step.control == "CAA"
        assert step.access is ModelAccess.CAPTURE
        assert step.venue == "stage"
        assert report.plan.stages is True
        (fit,) = report.plan.fits
        assert fit.artifact == "ContrastiveFit"
        assert fit.venue == "stage"

    def test_precomputed_template_steers_through_the_session(self):
        """A fully concrete configuration needs only structural facts at steer."""
        pipeline = SteeringPipeline(
            model_name_or_path="m",controls=[CAA(steering_vector=_vector(), layer_id=1)],
        )
        report = pipeline.check(backend=SERVE_SPEC)
        assert report.supported("generate")
        (step,) = report.plan.steps
        assert step.access is ModelAccess.FACTS
        assert step.venue == "session"
        assert report.plan.stages is False

    def test_score_phase_rejects_spec_backend_by_name(self):
        """Scoring an intervention control on a spec backend fails at check, naming the control."""
        pipeline = SteeringPipeline(
            model_name_or_path="m",controls=[CAA(steering_vector=_vector(), layer_id=1)],
        )
        report = pipeline.check(backend=SERVE_SPEC)
        failures = report.failures_for("score")
        assert len(failures) == 1
        assert failures[0].control == "CAA"
        assert "prompt" in failures[0].message

    def test_generate_offers_spec_alternative_only_with_a_wire_form(self):
        from steerability.algorithms.state_control.act_add.control import ActAdd

        exportable = CAA(steering_vector=_vector(), layer_id=1)
        positional = ActAdd(steering_vector=_vector(k=3), layer_id=1)

        def offers_specs(control) -> bool:
            return any(
                Capability.INTERVENTION_SPECS in alternative.atoms
                for alternative in control.requirements().generate
            )

        assert offers_specs(exportable)
        assert not offers_specs(positional)


class _FakeServeSession:
    """Session double serving only structural facts, for engine-session steers."""

    def __init__(self):
        self.closed = False

    @property
    def layout(self) -> ModelFacts:
        return ModelFacts(
            num_layers=LAYERS, hidden_size=HIDDEN, num_attention_heads=2, head_dim=HIDDEN // 2,
            dtype="float32", model_fingerprint="0" * 16, model_type="llama", model_ref="tiny",
        )

    def close(self) -> None:
        self.closed = True


class TestEagerLoweringFailure:

    def test_lowering_failure_names_the_intervention_and_reason(self):
        """A configuration whose inexpressibility is artifact-dependent passes check() and
        fails at the eager steer-time lowering with the intervention named."""
        from steerability.algorithms.core.execution import UnsupportedOperationError

        class _UncoveredSource:
            """Resolves a vector with no direction for the behavior layer."""

            access = ModelAccess.FACTS

            def resolve(self, model, tokenizer, *, session=None):
                generator = torch.Generator().manual_seed(3)
                return SteeringVector(
                    model_type="llama",
                    directions={0: torch.randn(1, HIDDEN, generator=generator)},
                )

        from steerability.algorithms.state_control.base import InterventionControl
        from steerability.algorithms.state_control.common.specs import Intervention, TokenScope
        from steerability.algorithms.state_control.common.transforms import AdditiveTransform
        from tests.utils.tiny_models import wordlevel_tokenizer

        class _DeclaredBroadcast(InterventionControl):
            Args = None
            hook_only_hint = "the behavior layer has no direction; run on the huggingface backend"

            def _configure(self):
                self._template = (Intervention(
                    layers=(1,),
                    transform=AdditiveTransform(_UncoveredSource()),
                    scope=TokenScope("all"),
                    require_coverage=False,
                ),)

        control = _DeclaredBroadcast()
        pipeline = SteeringPipeline(controls=[control], backend=SERVE_SPEC)
        pipeline.tokenizer = wordlevel_tokenizer()

        # check() consults construction-time facts, so the declared kinds pass
        assert pipeline.check(backend=SERVE_SPEC).supported("generate")
        assert pipeline.check(backend=SERVE_SPEC).plan.steps[0].venue == "session"

        class _NullStager:
            _discovery = None

            def open_session(self):
                return _FakeServeSession()

            def stage_artifacts(self, payloads):
                return None

            def release(self):
                return None

        pipeline._backends[SERVE_SPEC] = _NullStager()
        with pytest.raises(UnsupportedOperationError) as excinfo:
            pipeline.steer()
        message = str(excinfo.value)
        assert "_DeclaredBroadcast" in message
        assert "intervention 0" in message
        assert "AdditiveTransform" in message
