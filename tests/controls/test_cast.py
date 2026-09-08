import pytest
import torch

from steerability.algorithms.core.steering_pipeline import SteeringPipeline
from steerability.algorithms.state_control.cast.control import CAST
from steerability.algorithms.state_control.common.steering_vector import SteeringVector
from tests.utils.sweep import build_param_grid

PROMPT_TEXT = (
    "Answer truthfully. Therefore, when you respond: "
    "First, present your main point. "
    "Second, support it with evidence. "
    "Finally, conclude succinctly."
)

CAST_GRID = {
    'behavior_vector_strength': [1.0],
    'condition_vector_threshold': [0.043],
}


def create_dummy_steering_vector(model_type, hidden_size, num_layer):
    """Creates a dummy steering vector for a given model_type, hidden_size and num_layer."""
    directions = {k: torch.zeros(1, hidden_size) for k in range(num_layer)}
    explained_variances = {k: 0.5 for k in range(num_layer)}
    vec = SteeringVector(model_type=model_type, directions=directions, explained_variances=explained_variances)
    return vec


@pytest.mark.parametrize("conf", build_param_grid(CAST_GRID))
def test_cast(model_and_tokenizer, device: torch.device, conf: dict):
    """Verify that CAST steers and generates on every model/device/param combo."""

    # move model to target device
    base_model, tokenizer = model_and_tokenizer
    model = base_model.to(device)

    # get model_type, hidden_size and num_layer for current model
    model_type = model.config.model_type
    hidden_size = getattr(model.config, 'hidden_size') if model_type != 'gpt2' else getattr(model.config, 'n_embd')
    num_layer = getattr(model.config, 'num_hidden_layers') if model_type != 'gpt2' else getattr(model.config, 'n_layer')

    # create (dummy) behavior and condition vectors
    behavior_vector = create_dummy_steering_vector(model.config.model_type, model.config.hidden_size, num_layer)
    condition_vector = create_dummy_steering_vector(model_type, hidden_size, num_layer)

    # build pipeline with CAST control
    cast = CAST(
        behavior_vector=behavior_vector,
        behavior_layer_ids=[0, 1],
        behavior_vector_strength=conf['behavior_vector_strength'],
        condition_vector=condition_vector,
        condition_layer_ids=[1],
        condition_vector_threshold=conf['condition_vector_threshold'],
        condition_comparator_threshold_is='ge',
    )
    pipeline = SteeringPipeline(
        controls=[cast],
        device_map=device,
        model=model,
        tokenizer=tokenizer,
    )
    pipeline.steer()

    # prepare prompt & runtime kwargs
    prompt_ids = tokenizer(PROMPT_TEXT, return_tensors="pt").input_ids.to(device)

    # generate
    out_ids = pipeline.generate(
        input_ids=prompt_ids,
        max_new_tokens=8,
    )

    # assertions
    assert isinstance(out_ids, torch.Tensor), "Output is not torch.Tensor"
    assert out_ids.ndim == 2, "Expected (batch, seq_len) tensor"
    assert out_ids.size(1) >= 1, "No new tokens generated"


# behavior_transform slot: args validation, construction, and application (hub-free tiny Llama)

HIDDEN = 32
LAYERS = 4


def _unit_vector(seed: int, dim: int = HIDDEN) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(dim, generator=g)
    return v / v.norm()


def _steering_vector(seed: int, layers) -> SteeringVector:
    return SteeringVector(
        model_type="llama",
        directions={l: _unit_vector(seed + l).unsqueeze(0) for l in layers},
        explained_variances={l: 0.5 for l in layers},
    )


def _tiny_pipeline(control, seed: int = 0):
    from tests.utils.tiny_models import tiny_llama, wordlevel_tokenizer

    torch.manual_seed(seed)
    model = tiny_llama(num_layers=LAYERS, hidden=HIDDEN, heads=4)
    tokenizer = wordlevel_tokenizer()
    pipeline = SteeringPipeline(controls=[control], model=model, tokenizer=tokenizer)
    pipeline.steer()
    return pipeline, model, tokenizer


class _ProbeAblation:
    """Wraps a transform, recording the direction-projection of masked rows before/after each apply.

    Records at a single target layer so the pre/post projections describe the same hidden state.
    """

    def __init__(self, inner, direction: torch.Tensor, target_layer: int = 0):
        self._inner = inner
        self._unit = direction / direction.norm()
        self._target_layer = target_layer
        self.pre_projections: list[float] = []
        self.post_projections: list[float] = []

    def apply(self, hidden_states, *, layer_id, token_mask, **kwargs):
        out = self._inner.apply(hidden_states, layer_id=layer_id, token_mask=token_mask, **kwargs)
        if layer_id == self._target_layer and token_mask.any():
            unit = self._unit.to(out.device, out.dtype)
            self.pre_projections.append(float((hidden_states[token_mask] @ unit).abs().max()))
            self.post_projections.append(float((out[token_mask] @ unit).abs().max()))
        return out

    @property
    def covered_layer_ids(self):
        return self._inner.covered_layer_ids


class TestBehaviorTransformValidation:
    def _base(self, **overrides):
        from steerability.algorithms.state_control.cast.args import CASTArgs

        kwargs = dict()
        kwargs.update(overrides)
        return CASTArgs(**kwargs)

    def _ablation(self, layers=(0, 1), **kwargs):
        from steerability.algorithms.state_control.common.transforms import ProjectionTransform

        return ProjectionTransform(_steering_vector(seed=100, layers=layers), **kwargs)

    def test_transform_plus_vector_raises(self):
        with pytest.raises(ValueError, match="carries its own artifact"):
            self._base(behavior_transform=self._ablation(), behavior_vector=_steering_vector(1, [0]))

    def test_transform_plus_data_raises(self):
        with pytest.raises(ValueError, match="carries its own artifact"):
            self._base(
                behavior_transform=self._ablation(),
                behavior_data={"positives": ["a"], "negatives": ["b"]},
            )

    def test_transform_plus_strength_raises(self):
        with pytest.raises(ValueError, match="strength"):
            self._base(behavior_transform=self._ablation(), behavior_vector_strength=2.0)

    def test_transform_plus_explained_variance_raises(self):
        with pytest.raises(ValueError, match="use_explained_variance"):
            self._base(behavior_transform=self._ablation(), use_explained_variance=True)

    def test_transform_plus_ooi_normalization_raises(self):
        with pytest.raises(ValueError, match="NormPreservingTransform"):
            self._base(behavior_transform=self._ablation(), use_ooi_preventive_normalization=True)

    def test_nondefault_behavior_fit_is_inert(self):
        # behavior_fit is only read when fitting from behavior_data (absent here), so it does not raise
        from steerability.algorithms.state_control.common.fit_specs import VectorTrainSpec

        args = self._base(
            behavior_transform=self._ablation(),
            behavior_fit=VectorTrainSpec(method="mean_diff", accumulate="last_token"),
        )
        assert args.behavior_transform is not None

    def test_all_sources_absent_raises_three_way(self):
        with pytest.raises(ValueError, match="behavior_vector, behavior_data, or behavior_transform"):
            self._base()

    def test_non_transform_non_callable_raises_type_error(self):
        with pytest.raises(TypeError, match="behavior_transform must be a BaseTransform"):
            self._base(behavior_transform=42)


class TestBehaviorTransformApplication:
    def test_bound_instance_ablates_along_direction(self):
        from steerability.algorithms.state_control.common.transforms import ProjectionTransform

        direction = _unit_vector(7)
        transform = ProjectionTransform({0: direction.unsqueeze(0), 1: _unit_vector(8).unsqueeze(0)})
        control = CAST(behavior_transform=transform, behavior_layer_ids=[0, 1])
        pipeline, _, _ = _tiny_pipeline(control)

        assert isinstance(control._transform, ProjectionTransform)
        assert control._transform is transform  # bound at construction -> used as-is

        probe = _ProbeAblation(control._transform, direction, target_layer=0)
        control._transform = probe
        pipeline.generate(input_ids=torch.tensor([[3, 4, 5, 6]]), max_new_tokens=3)

        assert probe.post_projections, "transform never applied at a masked position"
        # the un-ablated hidden state has a non-trivial component along the direction ...
        assert max(probe.pre_projections) > 1e-2
        # ... and ablation drives that component down by orders of magnitude at masked positions
        for pre, post in zip(probe.pre_projections, probe.post_projections):
            assert post < 0.02 * pre + 1e-6

    def test_source_carrying_transform_bound_after_steer(self):
        from steerability.algorithms.state_control.common.sources import ContrastiveFit
        from steerability.algorithms.state_control.common.transforms import ProjectionTransform

        source_transform = ProjectionTransform(
            ContrastiveFit(
                data={"positives": ["yes indeed", "sure absolutely"], "negatives": ["no thanks", "never decline"]},
                method="mean_diff",
                accumulate="last_token",
                prompt_format="raw",
                location="layer_input",
            )
        )
        assert source_transform.is_bound is False
        control = CAST(behavior_transform=source_transform, behavior_layer_ids=[0, 1])
        pipeline, _, _ = _tiny_pipeline(control)

        assert control._transform is not source_transform  # a fresh bound instance
        assert control._transform.is_bound is True
        assert source_transform.is_bound is False  # the arg stays unbound
        pipeline.generate(input_ids=torch.tensor([[3, 4, 5]]), max_new_tokens=2)

    def test_factory_receives_context_and_result_applied(self):
        from steerability.algorithms.state_control.common.transforms import ProjectionTransform, TransformContext

        seen = {}

        def _factory(ctx: TransformContext):
            seen["ctx"] = ctx
            sv = _steering_vector(11, ctx.layer_ids)
            return ProjectionTransform(ctx.resolve(sv), alpha=1.0)

        control = CAST(behavior_transform=_factory, behavior_layer_ids=[0, 1])
        pipeline, _, _ = _tiny_pipeline(control)

        ctx = seen["ctx"]
        assert sorted(ctx.layer_ids) == [0, 1]
        assert isinstance(control._transform, ProjectionTransform)
        assert control._transform.is_bound is True
        pipeline.generate(input_ids=torch.tensor([[3, 4, 5]]), max_new_tokens=2)

    def test_coverage_error_when_transform_misses_behavior_layer(self):
        from steerability.algorithms.state_control.common.transforms import ProjectionTransform

        transform = ProjectionTransform(_steering_vector(seed=100, layers=[0]))  # missing layer 1
        control = CAST(behavior_transform=transform, behavior_layer_ids=[0, 1])
        with pytest.raises(ValueError, match="no direction for layer"):
            _tiny_pipeline(control)


# reset() / get_hooks() gate re-sizing across consecutive generations
def _conditional_cast() -> CAST:
    """A conditional CAST whose row gate resizes per batch."""
    return CAST(
        behavior_vector=_steering_vector(seed=0, layers=range(LAYERS)),
        behavior_layer_ids=[0, 1],
        condition_vector=_steering_vector(seed=1, layers=range(LAYERS)),
        condition_layer_ids=[1],
        condition_vector_threshold=0.043,
        condition_comparator_threshold_is="ge",
    )


def _cast_pipeline(control: CAST, model, tokenizer) -> SteeringPipeline:
    pipeline = SteeringPipeline(controls=[control], model=model, tokenizer=tokenizer)
    pipeline.steer()
    return pipeline


def test_consecutive_generations_across_batch_sizes():
    """A steered CAST pipeline generating batch-of-4 then batch-of-2 leaves no state from the first call.

    `reset()` clears the row gate to a single row without knowing the next batch; `get_hooks` re-sizes
    it to the true batch before any gate read. The batch-of-2 outputs must match a freshly steered,
    identically configured pipeline given only the batch-of-2, and the gate must be sized to 2 after.
    """
    from tests.utils.tiny_models import tiny_llama, wordlevel_tokenizer

    torch.manual_seed(0)
    model = tiny_llama(num_layers=LAYERS, hidden=HIDDEN, heads=4)
    tokenizer = wordlevel_tokenizer()

    batch4 = torch.stack([torch.arange(3, 7), torch.arange(4, 8), torch.arange(5, 9), torch.arange(6, 10)])
    batch2 = torch.stack([torch.arange(3, 7), torch.arange(4, 8)])
    gen = dict(max_new_tokens=4, do_sample=False, eos_token_id=None)

    control = _conditional_cast()
    pipeline = _cast_pipeline(control, model, tokenizer)
    pipeline.generate(input_ids=batch4, **gen)
    out_after_4 = pipeline.generate(input_ids=batch2, **gen)

    fresh_model = tiny_llama(num_layers=LAYERS, hidden=HIDDEN, heads=4)
    fresh_model.load_state_dict(model.state_dict())
    fresh_pipeline = _cast_pipeline(_conditional_cast(), fresh_model, wordlevel_tokenizer())
    out_fresh = fresh_pipeline.generate(input_ids=batch2, **gen)

    assert torch.equal(out_after_4, out_fresh)  # no state from the batch-of-4 leaked into the batch-of-2
    assert control._gate.num_rows == 2  # get_hooks re-sized the gate past the unsized reset() clear
