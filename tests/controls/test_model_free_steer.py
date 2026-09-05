"""Tests for model-free steering of vector-supplied state controls: a fully concrete
configuration binds against a session layout with `model=None`, and hook module names resolve
from the module tree at `get_hooks()` time."""
import pytest
import torch

from steerability.algorithms.core.execution import (
    BackendSpec,
    GenerationItem,
    GenerationParams,
    HookEntry,
    ModelFacts,
    PreparedPrompt,
)
from steerability.algorithms.state_control.act_add.control import ActAdd
from steerability.algorithms.state_control.activation_adapter.control import ActivationAdapter
from steerability.algorithms.state_control.angular_steering.control import AngularSteering
from steerability.algorithms.state_control.caa.control import CAA
from steerability.algorithms.state_control.common.steering_vector import SteeringVector
from steerability.algorithms.state_control.common.transforms import AdditiveTransform
from steerability.algorithms.state_control.directional_ablation.control import DirectionalAblation
from steerability.algorithms.state_control.iti.control import ITI
from steerability.backends.huggingface import HFBackend
from tests.utils.tiny_models import tiny_llama, wordlevel_tokenizer

LAYERS = 4
HIDDEN = 32
HEADS = 4


class _LayoutOnlySession:
    """Session double carrying only a structural layout."""

    def __init__(self, layout: ModelFacts):
        self._layout = layout

    @property
    def layout(self) -> ModelFacts:
        return self._layout


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return tiny_llama(num_layers=LAYERS, hidden=HIDDEN, heads=HEADS)


@pytest.fixture(scope="module")
def tokenizer():
    return wordlevel_tokenizer()


@pytest.fixture()
def layout_session():
    return _LayoutOnlySession(ModelFacts(
        num_layers=LAYERS,
        hidden_size=HIDDEN,
        num_attention_heads=HEADS,
        head_dim=HIDDEN // HEADS,
        dtype="float32",
        model_fingerprint="0" * 16,
    ))


def _vector(k: int = 1, seed: int = 0, layers=range(LAYERS)) -> SteeringVector:
    generator = torch.Generator().manual_seed(seed)
    return SteeringVector(
        model_type="llama",
        directions={lid: torch.randn(k, HIDDEN, generator=generator) for lid in layers},
    )


def _generate_with(control, model, tokenizer, prompt="the cat sat"):
    """Run one generation through the in-process session with the control's hooks."""
    hooks = control.get_hooks(
        tokenizer(prompt, return_tensors="pt")["input_ids"], None, model=model
    )
    backend = HFBackend.adopt(BackendSpec(kind="huggingface"), lambda: model, lambda: tokenizer)
    with backend.open_session() as session:
        item = GenerationItem(
            prompt=PreparedPrompt.from_text(prompt),
            state_entries=(HookEntry(hooks=hooks),),
        )
        results = session.generate([item], GenerationParams(max_new_tokens=3, greedy=True))
    return results[0].output.output_ids


class TestModelFreeSteer:

    def test_caa_steers_from_layout_and_generates(self, model, tokenizer, layout_session):
        control = CAA(steering_vector=_vector(), layer_id=1, multiplier=4.0)
        control.steer(model=None, tokenizer=tokenizer, session=layout_session)
        output_ids = _generate_with(control, model, tokenizer)
        assert output_ids.shape[1] == 3

    def test_act_add_steers_from_layout_and_generates(self, model, tokenizer, layout_session):
        control = ActAdd(steering_vector=_vector(k=3), layer_id=1, multiplier=2.0)
        control.steer(model=None, tokenizer=tokenizer, session=layout_session)
        output_ids = _generate_with(control, model, tokenizer)
        assert output_ids.shape[1] == 3

    def test_directional_ablation_steers_from_layout_and_generates(self, model, tokenizer, layout_session):
        control = DirectionalAblation(steering_vector=_vector(), layer_ids=[1, 2])
        control.steer(model=None, tokenizer=tokenizer, session=layout_session)
        output_ids = _generate_with(control, model, tokenizer)
        assert output_ids.shape[1] == 3

    def test_angular_steering_steers_from_layout_and_generates(self, model, tokenizer, layout_session):
        control = AngularSteering(steering_vector=_vector(k=2), angle=0.4, mode="offset")
        control.steer(model=None, tokenizer=tokenizer, session=layout_session)
        output_ids = _generate_with(control, model, tokenizer)
        assert output_ids.shape[1] == 3

    def test_iti_steers_from_layout_and_generates(self, model, tokenizer, layout_session):
        vector = SteeringVector(
            model_type="llama",
            directions={lid: torch.randn(HEADS, HIDDEN // HEADS) for lid in range(LAYERS)},
            num_heads=HEADS,
            head_dim=HIDDEN // HEADS,
        )
        control = ITI(steering_vector=vector, selected_heads=[(1, 0), (2, 3)], alpha=2.0)
        control.steer(model=None, tokenizer=tokenizer, session=layout_session)
        output_ids = _generate_with(control, model, tokenizer)
        assert output_ids.shape[1] == 3

    def test_activation_adapter_steers_from_layout_and_generates(self, model, tokenizer, layout_session):
        control = ActivationAdapter(
            transform=AdditiveTransform(_vector().directions, strength=3.0), layer_ids=[2],
        )
        control.steer(model=None, tokenizer=tokenizer, session=layout_session)
        output_ids = _generate_with(control, model, tokenizer)
        assert output_ids.shape[1] == 3

    def test_steered_generation_differs_from_unsteered(self, model, tokenizer, layout_session):
        control = CAA(steering_vector=_vector(seed=3), layer_id=1, multiplier=50.0)
        control.steer(model=None, tokenizer=tokenizer, session=layout_session)
        steered = _generate_with(control, model, tokenizer)
        backend = HFBackend.adopt(BackendSpec(kind="huggingface"), lambda: model, lambda: tokenizer)
        with backend.open_session() as session:
            item = GenerationItem(prompt=PreparedPrompt.from_text("the cat sat"))
            plain = session.generate([item], GenerationParams(max_new_tokens=3, greedy=True))
        assert not torch.equal(steered, plain[0].output.output_ids)


class TestModelFreeSteerBoundaries:

    def test_steer_without_model_or_session_raises(self):
        control = CAA(steering_vector=_vector(), layer_id=1)
        with pytest.raises(ValueError, match="session"):
            control.steer(model=None, session=None)

    def test_data_fitted_config_requires_capture_capable_session(self, tokenizer, layout_session):
        control = CAA(data={"positives": ["the cat"], "negatives": ["the dog"]}, layer_id=1)
        with pytest.raises(ValueError, match="capture-capable session"):
            control.steer(model=None, tokenizer=tokenizer, session=layout_session)

    def test_get_hooks_without_model_anywhere_raises(self, tokenizer, layout_session):
        control = CAA(steering_vector=_vector(), layer_id=1)
        control.steer(model=None, tokenizer=tokenizer, session=layout_session)
        ids = tokenizer("the cat", return_tensors="pt")["input_ids"]
        with pytest.raises(RuntimeError, match="module names"):
            control.get_hooks(ids, None)

    def test_layout_dtype_governs_vector_preparation(self, tokenizer):
        session = _LayoutOnlySession(ModelFacts(
            num_layers=LAYERS, hidden_size=HIDDEN, num_attention_heads=HEADS,
            head_dim=HIDDEN // HEADS, dtype="float16", model_fingerprint="0" * 16,
        ))
        control = CAA(steering_vector=_vector(), layer_id=1)
        control.steer(model=None, tokenizer=tokenizer, session=session)
        assert control._steering_vector.directions[1].dtype == torch.float16

    def test_caller_vector_is_not_mutated_by_steer(self, tokenizer, layout_session):
        vector = _vector()
        before = {lid: d.clone() for lid, d in vector.directions.items()}
        caa = CAA(steering_vector=vector, layer_id=1, normalize_vector=True)
        caa.steer(model=None, tokenizer=tokenizer, session=layout_session)
        assert all(torch.equal(vector.directions[lid], before[lid]) for lid in before)
