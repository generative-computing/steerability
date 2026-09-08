"""The `ModelAccess` ladder: per-control declarations, scoped-session enforcement, and the
model gating below `MODULE`."""
import pytest
import torch

from steerability.algorithms.core.execution import BackendSpec, ModelAccess, UnsupportedOperationError
from steerability.algorithms.core.execution.session_utils import ScopedSession
from steerability.algorithms.core.steering_pipeline import SteeringPipeline
from steerability.algorithms.input_control.base import InputControl
from steerability.algorithms.state_control.common.sources import (
    ContrastiveFit,
    LayerFilteredFit,
    SinglePairFit,
    _Precomputed,
)
from steerability.algorithms.state_control.common.steering_vector import SteeringVector
from steerability.backends.huggingface import HFBackend
from tests.utils.tiny_models import tiny_llama, wordlevel_tokenizer

PAIRS = {"prompts": ["q"], "positives": ["a"], "negatives": ["b"]}


class TestLadder:

    def test_rungs_are_ordered_and_fold_by_max(self):
        assert ModelAccess.FACTS < ModelAccess.ROLLOUTS < ModelAccess.CAPTURE < ModelAccess.MODULE
        assert max(ModelAccess.FACTS, ModelAccess.CAPTURE, ModelAccess.ROLLOUTS) is ModelAccess.CAPTURE

    def test_wire_names_are_lowercase_member_names(self):
        assert [access.name.lower() for access in ModelAccess] == [
            "facts", "rollouts", "capture", "module",
        ]


class TestDeclarations:

    def test_cpo_access_follows_prompt_lm(self):
        from steerability.algorithms.input_control.cpo.control import CPO

        bound = CPO(seed_prompt="s", offline_data=[{"query": "q", "prompt": "p", "score": 1.0}])
        assert bound.steer_access() is ModelAccess.MODULE
        aux = CPO(
            seed_prompt="s",
            offline_data=[{"query": "q", "prompt": "p", "score": 1.0}],
            prompt_lm=object(),
        )
        assert aux.steer_access() is ModelAccess.ROLLOUTS

    def test_contrastive_fit_access_follows_estimator(self):
        assert ContrastiveFit(data=PAIRS).access is ModelAccess.CAPTURE

        class _Estimator:
            def fit(self, model, tokenizer, **kwargs):
                raise NotImplementedError

        custom = ContrastiveFit(data=PAIRS, estimator=_Estimator())
        assert custom.access is ModelAccess.MODULE

    def test_source_declarations(self):
        vector = SteeringVector(model_type="llama", directions={0: torch.zeros(1, 4)})
        assert _Precomputed(vector).access is ModelAccess.FACTS
        assert SinglePairFit(positive_prompt="p", negative_prompt="n").access is ModelAccess.MODULE
        assert SinglePairFit.artifact_class == "direction"
        wrapped = LayerFilteredFit(inner=ContrastiveFit(data=PAIRS), layer_range=(0, 1))
        assert wrapped.access is ModelAccess.CAPTURE
        assert wrapped.artifact_class == "direction"

    def test_routed_decoding_access_follows_probe_form(self):
        from steerability.algorithms.core.internals.probes import ProbeSetFit
        from steerability.algorithms.core.internals.probes.fitting import ProbeFitSpec
        from steerability.algorithms.core.internals.probes.probe import Probe
        from steerability.algorithms.core.internals.probes.probe_set import ProbeSet
        from steerability.algorithms.output_control.routed_decoding import P, Route, RoutedDecoding, Router
        from steerability.algorithms.output_control.routed_decoding.actions import respond

        rules = Router(routes=[Route("r", when=P("p"), action=respond("x"))])
        fit = RoutedDecoding(
            probes=ProbeSetFit(data={"p": PAIRS}, spec=ProbeFitSpec(method="mean_diff")),
            rules=rules,
        )
        assert fit.steer_access() is ModelAccess.CAPTURE
        assert fit.steer_fits() == (("ProbeSetFit", "calibrated"),)

        fitted = RoutedDecoding(
            probes=ProbeSet({"p": Probe(
                model_type="llama", location="layer_input", pooling="mean",
                layer_ids=[0], weights={0: torch.zeros(4)}, bias=0.0,
            )}),
            rules=rules,
        )
        assert fitted.steer_access() is ModelAccess.FACTS
        assert fitted.steer_fits() == ()

    def test_module_declarations_for_retaining_controls(self):
        from steerability.algorithms.output_control.rad.control import RAD
        from steerability.algorithms.output_control.sasa.control import SASA
        from steerability.algorithms.output_control.value_guidance.control import ValueGuidance
        from steerability.algorithms.state_control.pasta.control import PASTA

        assert SASA(beta=0.1).steer_access() is ModelAccess.MODULE
        assert RAD(beta=0.1, reward_model_id="unused").steer_access() is ModelAccess.MODULE
        assert object.__new__(ValueGuidance).steer_access() is ModelAccess.MODULE
        assert PASTA(head_config=[0]).steer_access() is ModelAccess.MODULE


class _SessionProbe:
    """Session double recording which operations were attempted."""

    def __init__(self):
        self.calls = []
        self.tokenizer = None

    @property
    def layout(self):
        self.calls.append("layout")
        return "layout-sentinel"

    def generate(self, items, params):
        self.calls.append("generate")
        return []

    def score(self, items, params):
        self.calls.append("score")
        return torch.zeros(0, 0)

    def capture(self, prompts, layers, mode, location="layer_output"):
        self.calls.append("capture")
        return "capture-sentinel"


class TestScopedSessionDenials:

    def test_facts_denies_generation_with_stable_message(self):
        scoped = ScopedSession(_SessionProbe(), "MyControl", ModelAccess.FACTS)
        with pytest.raises(UnsupportedOperationError) as excinfo:
            scoped.generate([], None)
        assert str(excinfo.value) == (
            "MyControl declared steer access 'facts', which does not include session "
            "generation; declare ModelAccess.ROLLOUTS or higher."
        )
        with pytest.raises(UnsupportedOperationError, match="session generation"):
            scoped.score([], None)

    def test_rollouts_denies_capture_with_stable_message(self):
        scoped = ScopedSession(_SessionProbe(), "MyControl", ModelAccess.ROLLOUTS)
        with pytest.raises(UnsupportedOperationError) as excinfo:
            scoped.capture([], [0], "last_token")
        assert str(excinfo.value) == (
            "MyControl declared steer access 'rollouts', which does not include hidden-state "
            "capture; declare ModelAccess.CAPTURE or higher."
        )

    def test_layout_and_tokenizer_available_at_every_rung(self):
        inner = _SessionProbe()
        scoped = ScopedSession(inner, "MyControl", ModelAccess.FACTS)
        assert scoped.layout == "layout-sentinel"
        assert scoped.tokenizer is None

    def test_declared_rungs_delegate(self):
        inner = _SessionProbe()
        scoped = ScopedSession(inner, "MyControl", ModelAccess.CAPTURE)
        scoped.generate([], None)
        scoped.score([], None)
        scoped.capture([], [0], "last_token")
        assert inner.calls == ["generate", "score", "capture"]

    def test_no_model_attribute_at_any_rung(self):
        scoped = ScopedSession(_SessionProbe(), "MyControl", ModelAccess.MODULE)
        with pytest.raises(AttributeError):
            _ = scoped.model

    def test_in_process_fact_reflects_the_venue_session(self):
        model = tiny_llama(num_layers=2, hidden=16, heads=2)
        backend = HFBackend.adopt(
            BackendSpec(kind="huggingface"), lambda: model, lambda: wordlevel_tokenizer(),
        )
        with backend.open_session() as exclusive:
            assert ScopedSession(exclusive, "C", ModelAccess.FACTS).in_process is True
        assert ScopedSession(_SessionProbe(), "C", ModelAccess.FACTS).in_process is False


class TestModelGating:

    def _recording_control(self, access):
        class _Recording(InputControl):
            def __init__(self):
                self.seen_model = "unset"

            def adapt(self, input_ids, runtime_kwargs=None):
                return input_ids

            def steer_access(self):
                return access

            def steer(self, model=None, tokenizer=None, session=None, **kwargs):
                self.seen_model = model

        return _Recording()

    @pytest.mark.parametrize("access", [ModelAccess.FACTS, ModelAccess.ROLLOUTS, ModelAccess.CAPTURE])
    def test_model_is_none_below_module(self, access):
        control = self._recording_control(access)
        pipeline = SteeringPipeline(
            controls=[control], model=tiny_llama(num_layers=2, hidden=16, heads=2), tokenizer=wordlevel_tokenizer(),
        )
        pipeline.steer()
        assert control.seen_model is None

    def test_model_passes_at_module(self):
        control = self._recording_control(ModelAccess.MODULE)
        pipeline = SteeringPipeline(
            controls=[control], model=tiny_llama(num_layers=2, hidden=16, heads=2), tokenizer=wordlevel_tokenizer(),
        )
        pipeline.steer()
        assert control.seen_model is pipeline.model
