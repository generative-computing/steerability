"""Residual-stream steering of hybrid attention stacks (Qwen3.5 / Qwen3-Next style).

Two tiers:

- Stub tests (no model download) over `hybrid_attention_stub`: assert that residual-stream sites
  hook every layer, that a decoder-layer intervention binds on a linear-attention layer without
  reading head geometry there, and that the attention-site consumers (`o_proj` interventions,
  PASTA, ITI) refuse the non-attention layers with an actionable message rather than an
  `AttributeError`.
- Integration tests over a hub-free tiny `Qwen3NextForCausalLM`: the notebook's Angular Steering
  paths (precomputed plane and estimation) and CAA steer and generate on the real hybrid decoder
  layer, with no hooks left behind.
"""
import pytest
import torch

from steerability.algorithms.core.internals.data import ContrastivePairs, LabeledExamples
from steerability.algorithms.core.internals.model_layout import head_geometry, resolve_model_layout
from steerability.algorithms.core.steering_pipeline import SteeringPipeline
from steerability.algorithms.state_control.activation_adapter.control import ActivationAdapter
from steerability.algorithms.state_control.angular_steering.control import AngularSteering
from steerability.algorithms.state_control.caa.control import CAA
from steerability.algorithms.state_control.common.fit_specs import VectorTrainSpec
from steerability.algorithms.state_control.common.steering_vector import SteeringVector
from steerability.algorithms.state_control.common.transforms import HeadAdditiveTransform
from steerability.algorithms.state_control.iti.utils.estimator import ProbeMassShiftEstimator
from steerability.algorithms.state_control.pasta.control import PASTA
from tests.utils.tiny_models import hybrid_attention_stub, tiny_qwen3_next, wordlevel_tokenizer

LAYERS = 4
HIDDEN = 32
HEADS = 2
NORM_ATTRS = ("input_layernorm", "post_attention_layernorm")


def _plane(hidden, layers, seed=0):
    """A `[2, H]`-per-layer steering plane (the shape Angular Steering consumes)."""
    gen = torch.Generator().manual_seed(seed)
    directions = {lid: torch.randn(2, hidden, generator=gen) for lid in range(layers)}
    return SteeringVector(model_type="test", directions=directions)


def _no_toolkit_hooks(model) -> bool:
    """True when no toolkit forward or pre hooks remain on any module (nothing leaked).

    transformers v5 parks its own context-gated output-capture hooks on modules after any
    forward that requests captured outputs; those are inert outside a capture context and are
    not leaks, so hooks owned by transformers itself are excluded from the check.
    """
    for module in model.modules():
        for registry in (module._forward_hooks, module._forward_pre_hooks):
            for hook in registry.values():
                if not getattr(hook, "__module__", "").startswith("transformers."):
                    return False
    return True


# stub tests


def test_angular_norm_input_site_hooks_every_layer_of_hybrid_stack():
    """The default norm-input placement hooks both residual norms on every layer of a hybrid stack."""
    stub = hybrid_attention_stub(num_layers=LAYERS, hidden=HIDDEN, heads=HEADS)
    angular = AngularSteering(steering_vector=_plane(HIDDEN, LAYERS, seed=1), target_degree=90.0)
    pipeline = SteeringPipeline(controls=[angular], model=stub, tokenizer=wordlevel_tokenizer())
    pipeline.steer()

    hooks = angular.get_hooks(torch.arange(1, 5, dtype=torch.long).unsqueeze(0), {}, model=stub)
    modules = {spec["module"] for spec in hooks["pre"]}
    expected = {f"model.layers.{i}.{attr}" for i in range(LAYERS) for attr in NORM_ATTRS}
    assert modules == expected


def test_decoder_layer_site_binds_on_linear_attention_layer_from_module_tree():
    """A decoder-layer intervention binds on a linear-attention layer without reading head geometry.

    `CAA(layer_id=0).steer(model, tokenizer)` is the module-tree bind that previously called
    `head_geometry` on layer 0; the linear-attention layer has no attention module, so that call
    would raise. The bind must succeed and the hook target the decoder layer itself.
    """
    stub = hybrid_attention_stub(num_layers=LAYERS, hidden=HIDDEN, heads=HEADS)
    caa = CAA(
        steering_vector=SteeringVector(model_type="test", directions={0: torch.ones(1, HIDDEN)}),
        layer_id=0,
    )
    caa.steer(stub, wordlevel_tokenizer())
    assert caa.interventions[0].layers == (0,)

    hooks = caa.get_hooks(torch.arange(1, 5, dtype=torch.long).unsqueeze(0), {}, model=stub)
    assert {spec["module"] for spec in hooks["forward"]} == {"model.layers.0"}


def test_o_proj_site_rejects_linear_attention_layer_at_hook_build():
    """A head-additive intervention steers (geometry read from an attention layer) but refuses to
    build an o_proj hook on a linear-attention layer."""
    stub = hybrid_attention_stub(num_layers=LAYERS, hidden=HIDDEN, heads=HEADS)
    head_dim = HIDDEN // HEADS
    sv = SteeringVector(
        model_type="test",
        directions={0: torch.ones(HEADS, head_dim), 3: torch.ones(HEADS, head_dim)},
        num_heads=HEADS,
        head_dim=head_dim,
    )
    adapter = ActivationAdapter(
        transform=HeadAdditiveTransform(sv, active_heads={0: {0}, 3: {0}}),
        layer_ids=[0, 3],
        hook_point="layer_input",
    )
    adapter.steer(stub, wordlevel_tokenizer())  # geometry read from layer 3

    with pytest.raises(ValueError, match="carries no attention module") as excinfo:
        adapter.get_hooks(torch.arange(1, 5, dtype=torch.long).unsqueeze(0), {}, model=stub)
    assert "[3" in str(excinfo.value)


def test_o_proj_site_hooks_attention_layer_of_hybrid_stack():
    """Restricted to an attention layer, the head-additive intervention hooks its o_proj input."""
    stub = hybrid_attention_stub(num_layers=LAYERS, hidden=HIDDEN, heads=HEADS)
    head_dim = HIDDEN // HEADS
    sv = SteeringVector(
        model_type="test",
        directions={3: torch.ones(HEADS, head_dim)},
        num_heads=HEADS,
        head_dim=head_dim,
    )
    adapter = ActivationAdapter(
        transform=HeadAdditiveTransform(sv, active_heads={3: {0}}),
        layer_ids=[3],
        hook_point="layer_input",
    )
    adapter.steer(stub, wordlevel_tokenizer())

    hooks = adapter.get_hooks(torch.arange(1, 5, dtype=torch.long).unsqueeze(0), {}, model=stub)
    assert {spec["module"] for spec in hooks["pre"]} == {"model.layers.3.self_attn.o_proj"}


def test_pasta_rejects_linear_attention_layer():
    """PASTA refuses a linear-attention layer at steer time."""
    stub = hybrid_attention_stub(num_layers=LAYERS, hidden=HIDDEN, heads=HEADS)
    with pytest.raises(ValueError, match="carries no attention module"):
        PASTA(head_config=[0], alpha=2.0).steer(stub, wordlevel_tokenizer())


def test_pasta_accepts_attention_layer_of_hybrid_stack():
    """PASTA resolves the attention module and head count on an attention layer of a hybrid stack."""
    stub = hybrid_attention_stub(num_layers=LAYERS, hidden=HIDDEN, heads=HEADS)
    pasta = PASTA(head_config=[3], alpha=2.0)
    pasta.steer(stub, wordlevel_tokenizer())
    assert pasta._attn_module_names == {3: "model.layers.3.self_attn"}
    assert pasta._num_heads_by_layer == {3: HEADS}


def test_iti_rejects_hybrid_stack_before_forward():
    """The ITI estimator refuses a hybrid stack before running any forward pass.

    The stub's `forward` raises, so a successful raise of the hybrid error proves no forward ran.
    """
    stub = hybrid_attention_stub(num_layers=LAYERS, hidden=HIDDEN, heads=HEADS)
    data = LabeledExamples(positives=["a", "b"], negatives=["c", "d"])
    spec = VectorTrainSpec(method="mean_diff", accumulate="last_token")
    with pytest.raises(ValueError, match="attention module on every decoder layer") as excinfo:
        ProbeMassShiftEstimator().fit(stub, tokenizer=None, data=data, spec=spec)
    assert "[3" in str(excinfo.value)


# integration tests over a hub-free tiny Qwen3-Next


@pytest.fixture
def qwen3_next():
    pytest.importorskip("transformers.models.qwen3_next")
    return tiny_qwen3_next(num_layers=LAYERS, hidden=HIDDEN, heads=HEADS)


def test_qwen3_next_layout_resolves(qwen3_next):
    """The tiny Qwen3-Next resolves to `llama_style` with only its full-attention layer marked."""
    layout = resolve_model_layout(qwen3_next)
    assert layout.family == "llama_style"
    assert layout.layer_prefix == "model.layers"
    assert layout.num_layers == LAYERS
    assert layout.norm_attrs == NORM_ATTRS
    assert layout.attention_layer_ids == (3,)
    assert layout.is_hybrid

    layer0 = qwen3_next.model.layers[0]
    assert hasattr(layer0, "linear_attn") and not hasattr(layer0, "self_attn")
    assert (head_geometry(qwen3_next, layout, 3).num_heads, head_geometry(qwen3_next, layout, 3).head_dim) == (
        HEADS,
        HIDDEN // HEADS,
    )


def test_angular_precomputed_plane_steers_and_generates_on_qwen3_next(qwen3_next):
    """A precomputed plane rotates across every layer of the hybrid stack and generation runs."""
    angular = AngularSteering(
        steering_vector=_plane(HIDDEN, LAYERS, seed=2),
        target_degree=180.0,
        adaptive=True,
        token_scope="all",
    )
    pipeline = SteeringPipeline(controls=[angular], model=qwen3_next, tokenizer=wordlevel_tokenizer())
    pipeline.steer()

    hooks = angular.get_hooks(torch.arange(1, 5, dtype=torch.long).unsqueeze(0), {}, model=qwen3_next)
    modules = {spec["module"] for spec in hooks["pre"]}
    expected = {f"model.layers.{i}.{attr}" for i in range(LAYERS) for attr in NORM_ATTRS}
    assert modules == expected

    out = pipeline.generate(text="the cat sat on mat", max_new_tokens=4)
    assert isinstance(out, str)
    assert _no_toolkit_hooks(qwen3_next)


def test_angular_estimation_path_fits_plane_over_every_layer_of_qwen3_next(qwen3_next):
    """The notebook's estimation path fits a `[2, H]` plane at every layer and generation runs."""
    angular = AngularSteering(
        data=ContrastivePairs(
            positives=["the cat sat", "the dog ran fast"],
            negatives=["on the mat", "the span"],
        ),
        train_spec=VectorTrainSpec(method="mean_diff", accumulate="last_token"),
        target_degree=90.0,
    )
    pipeline = SteeringPipeline(controls=[angular], model=qwen3_next, tokenizer=wordlevel_tokenizer())
    pipeline.steer()

    directions = angular._steering_vector.directions
    assert set(directions.keys()) == set(range(LAYERS))
    for direction in directions.values():
        assert direction.shape == (2, HIDDEN)

    out = pipeline.generate(text="the cat sat on mat", max_new_tokens=4)
    assert isinstance(out, str)
    assert _no_toolkit_hooks(qwen3_next)


def test_caa_steers_and_generates_on_linear_attention_layer_of_qwen3_next(qwen3_next):
    """CAA binds and generates on a linear-attention layer (layer 0) of the real hybrid stack."""
    caa = CAA(
        steering_vector=SteeringVector(model_type="test", directions={0: torch.ones(1, HIDDEN)}),
        layer_id=0,
    )
    caa.steer(qwen3_next, wordlevel_tokenizer())  # module-tree bind on a non-attention layer

    pipeline = SteeringPipeline(controls=[caa], model=qwen3_next, tokenizer=wordlevel_tokenizer())
    pipeline.steer()

    hooks = caa.get_hooks(torch.arange(1, 5, dtype=torch.long).unsqueeze(0), {}, model=qwen3_next)
    assert "model.layers.0" in {spec["module"] for spec in hooks["forward"]}

    out = pipeline.generate(text="the cat sat on mat", max_new_tokens=4)
    assert isinstance(out, str)
    assert _no_toolkit_hooks(qwen3_next)
