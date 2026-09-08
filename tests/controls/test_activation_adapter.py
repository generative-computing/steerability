"""Tests for `ActivationAdapter` (transforms as the sole artifact carrier, gates as the sole
condition carrier).

Covers behavioral parity with CAA and DirectionalAblation (bound + source-carrying transforms),
the validation surface (placement / gating / scope / follower rules + artifact-kwarg guard),
transform binding and coverage, factory mode over `ctx.resolve`, the packaged `CosineReadout`,
gating, native batch support, registry discovery, a `ControlSpec` sweep with shared-source
memoization, and pipeline integration under state-control multiplicity.

Runs hub-free on a tiny randomly-initialized Llama.
"""
import pytest
import torch

from steerability.algorithms.core.steering_pipeline import SteeringPipeline
from steerability.algorithms.core.utils.assembly import collect_state_entries
from steerability.algorithms.state_control.activation_adapter import (
    ActivationAdapter,
    ActivationAdapterArgs,
    TransformContext,
)
from steerability.algorithms.state_control.activation_adapter.control import ActivationAdapter as _AA
from steerability.algorithms.state_control.caa.control import CAA
from steerability.algorithms.state_control.common.gating import (
    CallableReadout,
    CosineReadout,
    Evidence,
    Gate,
    PerKeyThreshold,
)
from steerability.algorithms.state_control.common.sources import ContrastiveFit
from steerability.algorithms.state_control.common.steering_vector import SteeringVector
from steerability.algorithms.state_control.common.transforms import (
    AdditiveTransform,
    NormPreservingTransform,
    ProjectionTransform,
)
from steerability.algorithms.state_control.common.transforms.base import BaseTransform
from steerability.algorithms.state_control.directional_ablation.control import DirectionalAblation
from tests.utils.tiny_models import tiny_llama, wordlevel_tokenizer

HIDDEN = 32
HEADS = 4
LAYERS = 4


def _sv(seed=1, k=1):
    g = torch.Generator().manual_seed(seed)
    return SteeringVector(
        model_type="llama",
        directions={lid: torch.randn(k, HIDDEN, generator=g) for lid in range(LAYERS)},
    )


class _StubSource:
    """An `ArtifactSource` wrapping a concrete vector, counting resolves (for parity/memo tests)."""

    def __init__(self, steering_vector: SteeringVector):
        self._sv = steering_vector
        self.fits = 0

    def resolve(self, model, tokenizer) -> SteeringVector:
        self.fits += 1
        return self._sv.clone()


def _constant_gate(threshold, value, condition_layer=0):
    """A gate whose readout returns a constant per-row value at one condition layer."""
    readout = CallableReadout(lambda pooled, lid, _v=value: torch.full((pooled.size(0),), float(_v)))
    return Gate(Evidence((condition_layer,), readout), PerKeyThreshold(threshold=threshold, comparator="ge"))


def _pipe(control, model):
    tok = wordlevel_tokenizer()
    p = SteeringPipeline(controls=[control] if not isinstance(control, list) else control, model=model, tokenizer=tok)
    p.steer()
    return p


def _hidden_at(model, layer_id, pipeline, input_ids):
    """Capture the (steered) output of `layer_id` under the pipeline's state controls, single pass."""
    entries = collect_state_entries(
        pipeline.state_controls, input_ids, {},
        hooks_in_process=True, lowered_state=pipeline._lowered_state, model=pipeline.model,
    )
    backend = pipeline._backend_for(pipeline._resolve_backend_spec(None))
    captured = {}

    with backend.open_session() as session, session.entries_applied(entries):
        def _cap(module, args, kwargs, output):
            captured["h"] = (output[0] if isinstance(output, tuple) else output).detach().clone()

        handle = model.model.layers[layer_id].register_forward_hook(_cap, with_kwargs=True)
        try:
            with torch.no_grad():
                model(input_ids=input_ids)
        finally:
            handle.remove()
    return captured["h"]


# parity
class TestParity:
    def test_parity_vs_caa(self):
        sv = _sv(7)
        model = tiny_llama(num_layers=LAYERS, hidden=HIDDEN, heads=HEADS)
        input_ids = torch.arange(3, 7, dtype=torch.long).unsqueeze(0)

        p_caa = _pipe(CAA(steering_vector=sv, layer_id=1, multiplier=2.0, token_scope="after_prompt"), model)
        out_caa = p_caa.generate(input_ids=input_ids, max_new_tokens=6, do_sample=False, eos_token_id=None)
        h_caa = _hidden_at(model, 1, p_caa, input_ids)

        p_ad = _pipe(ActivationAdapter(
            transform=AdditiveTransform(sv, strength=2.0), layer_ids=1, token_scope="after_prompt",
        ), model)
        out_ad = p_ad.generate(input_ids=input_ids, max_new_tokens=6, do_sample=False, eos_token_id=None)
        h_ad = _hidden_at(model, 1, p_ad, input_ids)

        assert torch.equal(out_caa, out_ad)
        assert torch.allclose(h_caa, h_ad, atol=1e-6)

    def test_parity_vs_directional_ablation(self):
        """Bound, source-carrying, and standalone DirectionalAblation produce identical ids."""
        sv = _sv(9)
        model = tiny_llama(num_layers=LAYERS, hidden=HIDDEN, heads=HEADS)
        input_ids = torch.arange(3, 7, dtype=torch.long).unsqueeze(0)

        def _gen(control):
            m = tiny_llama(num_layers=LAYERS, hidden=HIDDEN, heads=HEADS)
            m.load_state_dict(model.state_dict())
            p = _pipe(control, m)
            return p.generate(input_ids=input_ids, max_new_tokens=6, do_sample=False, eos_token_id=None), m, p

        out_da, _, _ = _gen(DirectionalAblation(steering_vector=sv, alpha=1.0, layer_ids=[1, 2], token_scope="all"))

        # bound transform
        out_bound, _, _ = _gen(ActivationAdapter(
            transform=ProjectionTransform(sv, alpha=1.0), layer_ids=[1, 2], token_scope="all",
        ))

        # source-carrying transform (resolved + bound at steer)
        source = _StubSource(sv)
        out_source, m_src, p_src = _gen(ActivationAdapter(
            transform=ProjectionTransform(source, alpha=1.0), layer_ids=[1, 2], token_scope="all",
        ))

        assert torch.equal(out_da, out_bound)
        assert torch.equal(out_da, out_source)
        assert source.fits == 1  # resolved exactly once at steer

    def test_source_bound_directions_match_master(self):
        """The bound transform's directions equal the source's fitted master (allclose, distinct object)."""
        sv = _sv(31)
        source = _StubSource(sv)
        model = tiny_llama(num_layers=LAYERS, hidden=HIDDEN, heads=HEADS)
        adapter = ActivationAdapter(
            transform=ProjectionTransform(source, alpha=1.0), layer_ids=[1], token_scope="all",
        )
        adapter.steer(model, wordlevel_tokenizer())
        bound = adapter._transform
        assert bound.directions is not sv.directions
        assert torch.allclose(bound.directions[1].float(), sv.directions[1].float())


# validation surface (placement / gating / scope / follower + artifact-kwarg guard)
class TestValidationSurface:
    def test_transform_required(self):
        with pytest.raises(ValueError, match="transform is required"):
            ActivationAdapterArgs(layer_ids=1)

    def test_transform_wrong_type(self):
        with pytest.raises(TypeError, match="transform must be a BaseTransform"):
            ActivationAdapterArgs(transform=object(), layer_ids=1)

    @pytest.mark.parametrize("name,value", [
        ("steering_vector", "SV"),
        ("data", {"positives": ["a"], "negatives": ["b"]}),
        ("train_spec", {"method": "mean_diff"}),
        ("estimator", object()),
        ("estimator_kwargs", {"foo": 1}),
        ("strength", 2.0),
        ("normalize_vector", True),
    ])
    def test_artifact_kwarg_guard_kwargs_form(self, name, value):
        if value == "SV":
            value = _sv()
        t = AdditiveTransform(_sv())
        with pytest.raises(TypeError, match="does not accept") as ei:
            ActivationAdapterArgs.validate(transform=t, layer_ids=1, **{name: value})
        assert name in str(ei.value)

    def test_artifact_kwarg_guard_dict_form(self):
        t = AdditiveTransform(_sv())
        with pytest.raises(TypeError, match="does not accept") as ei:
            ActivationAdapterArgs.validate({"transform": t, "layer_ids": 1, "strength": 2.0})
        assert "strength" in str(ei.value)
        assert "AdditiveTransform" in str(ei.value)  # replacement hint for 'strength'

    def test_both_placements(self):
        from steerability.algorithms.state_control.common.selectors import FixedLayerSelector
        with pytest.raises(ValueError, match="exactly one of layer_ids or layer_selector"):
            ActivationAdapterArgs(transform=AdditiveTransform(_sv()), layer_ids=1, layer_selector=FixedLayerSelector(1))

    def test_neither_placement(self):
        with pytest.raises(ValueError, match="exactly one of layer_ids or layer_selector"):
            ActivationAdapterArgs(transform=AdditiveTransform(_sv()))

    def test_gate_wrong_type(self):
        with pytest.raises(TypeError, match="gate must be a Gate"):
            ActivationAdapterArgs(transform=AdditiveTransform(_sv()), layer_ids=1, gate=object())

    def test_follower_flag_permits_shared_gate(self):
        gate = _constant_gate(threshold=0.5, value=0.9)
        ActivationAdapterArgs(transform=AdditiveTransform(_sv()), layer_ids=1, gate=gate, gate_driven_externally=True)

    def test_follower_flag_with_gate_source_raises(self):
        from steerability.algorithms.state_control.common.sources import ConditionPointSearch

        with pytest.raises(ValueError, match="pass the driver's Gate"):
            ActivationAdapterArgs(
                transform=AdditiveTransform(_sv()), layer_ids=1,
                gate=ConditionPointSearch(), gate_driven_externally=True,
            )

    def test_follower_flag_without_gate_warns(self):
        with pytest.warns(UserWarning, match="gate_driven_externally is inert"):
            ActivationAdapterArgs(transform=AdditiveTransform(_sv()), layer_ids=1, gate_driven_externally=True)

    def test_last_k_missing(self):
        with pytest.raises(ValueError, match="last_k must be >= 1"):
            ActivationAdapterArgs(transform=AdditiveTransform(_sv()), layer_ids=1, token_scope="last_k")

    def test_from_position_missing(self):
        with pytest.raises(ValueError, match="from_position must be >= 0"):
            ActivationAdapterArgs(transform=AdditiveTransform(_sv()), layer_ids=1, token_scope="from_position")

    def test_inert_last_k_warns(self):
        with pytest.warns(UserWarning, match="last_k is inert"):
            ActivationAdapterArgs(transform=AdditiveTransform(_sv()), layer_ids=1, token_scope="all", last_k=2)

    def test_inert_from_position_warns(self):
        with pytest.warns(UserWarning, match="from_position is inert"):
            ActivationAdapterArgs(transform=AdditiveTransform(_sv()), layer_ids=1, token_scope="all", from_position=2)

    def test_negative_layer_ids(self):
        with pytest.raises(ValueError, match="layer_ids must all be >= 0"):
            ActivationAdapterArgs(transform=AdditiveTransform(_sv()), layer_ids=[-1])

    def test_duplicate_layer_ids(self):
        with pytest.raises(ValueError, match="must not contain duplicates"):
            ActivationAdapterArgs(transform=AdditiveTransform(_sv()), layer_ids=[1, 1])

    def test_hook_point_invalid(self):
        with pytest.raises(ValueError, match="hook_point must be"):
            ActivationAdapterArgs(transform=AdditiveTransform(_sv()), layer_ids=1, hook_point="middle")

    def test_deferred_condition_layer_out_of_range(self):
        gate = _constant_gate(threshold=0.0, value=0.0, condition_layer=99)
        model = tiny_llama(num_layers=LAYERS, hidden=HIDDEN, heads=HEADS)
        adapter = ActivationAdapter(transform=AdditiveTransform(_sv()), layer_ids=[1], gate=gate)
        with pytest.raises(ValueError, match="condition_layer_id 99 out of range"):
            adapter.steer(model, wordlevel_tokenizer())

    def test_condition_selector_rejected_for_placement(self):
        from steerability.algorithms.state_control.common.selectors import ConditionPointSelector
        with pytest.raises(ValueError, match="ConditionPointSelector returns"):
            ActivationAdapter(transform=AdditiveTransform(_sv()), layer_selector=ConditionPointSelector())


# transform binding, coverage, factory
class TestTransformBinding:
    def test_bound_transform_end_to_end(self):
        model = tiny_llama(num_layers=LAYERS, hidden=HIDDEN, heads=HEADS)
        transform = AdditiveTransform(_sv(3), strength=1.5)
        adapter = ActivationAdapter(transform=transform, layer_ids=1, token_scope="all")
        p = _pipe(adapter, model)
        out = p.generate(input_ids=torch.arange(3, 7).unsqueeze(0), max_new_tokens=3, do_sample=False, eos_token_id=None)
        assert out.size(1) >= 1
        assert adapter._transform is transform  # already bound: used as-is

    def test_factory_receives_populated_context_and_resolves(self):
        model = tiny_llama(num_layers=LAYERS, hidden=HIDDEN, heads=HEADS)
        source = _StubSource(_sv(5))
        seen = {}

        def _factory(ctx: TransformContext):
            seen["ctx"] = ctx
            return NormPreservingTransform(AdditiveTransform(ctx.resolve(source), strength=1.5))

        adapter = ActivationAdapter(transform=_factory, layer_ids=[1, 2], token_scope="all")
        adapter.steer(model, wordlevel_tokenizer())
        ctx = seen["ctx"]
        assert ctx.layer_ids == [1, 2]
        assert ctx.num_layers == LAYERS
        assert ctx.hidden_size == HIDDEN
        assert ctx.num_heads == HEADS
        assert ctx.head_dim == HIDDEN // HEADS
        assert isinstance(adapter._transform, NormPreservingTransform)
        assert source.fits == 1

    def test_factory_returning_non_transform_raises(self):
        model = tiny_llama(num_layers=LAYERS, hidden=HIDDEN, heads=HEADS)
        adapter = ActivationAdapter(transform=lambda ctx: object(), layer_ids=1)
        with pytest.raises(TypeError, match="must return a BaseTransform"):
            adapter.steer(model, wordlevel_tokenizer())

    def test_source_transform_bound_at_steer(self):
        model = tiny_llama(num_layers=LAYERS, hidden=HIDDEN, heads=HEADS)
        source = _StubSource(_sv(5))
        transform = AdditiveTransform(source, strength=1.0)
        assert transform.is_bound is False
        adapter = ActivationAdapter(transform=transform, layer_ids=1, token_scope="all")
        adapter.steer(model, wordlevel_tokenizer())
        assert adapter._transform is not transform  # a freshly-bound instance
        assert adapter._transform.is_bound is True
        assert transform.is_bound is False  # template untouched

    def test_coverage_error_on_uncovered_layer(self):
        model = tiny_llama(num_layers=LAYERS, hidden=HIDDEN, heads=HEADS)
        sv = SteeringVector(model_type="llama", directions={1: torch.randn(1, HIDDEN)})
        adapter = ActivationAdapter(transform=AdditiveTransform(sv), layer_ids=[1, 2], token_scope="all")
        with pytest.raises(ValueError, match="no direction for layer"):
            adapter.steer(model, wordlevel_tokenizer())

    def test_coverage_none_opts_out(self):
        """A transform reporting covered_layer_ids=None skips the coverage check."""
        model = tiny_llama(num_layers=LAYERS, hidden=HIDDEN, heads=HEADS)

        class _NoCoverage(BaseTransform):
            def apply(self, hidden_states, *, layer_id, token_mask, **kwargs):
                return hidden_states

        adapter = ActivationAdapter(transform=_NoCoverage(), layer_ids=[1, 2], token_scope="all")
        adapter.steer(model, wordlevel_tokenizer())  # no error
        assert adapter._transform.covered_layer_ids is None

    def test_bind_result_guard(self):
        """A subclass that stores a source but keeps the base bind returns an unbound self -> TypeError."""
        model = tiny_llama(num_layers=LAYERS, hidden=HIDDEN, heads=HEADS)

        class _BadTransform(BaseTransform):
            @property
            def is_bound(self):
                return False  # never becomes bound (base bind returns self)

            def apply(self, hidden_states, *, layer_id, token_mask, **kwargs):
                return hidden_states

        adapter = ActivationAdapter(transform=_BadTransform(), layer_ids=1, token_scope="all")
        with pytest.raises(TypeError, match="must return a bound BaseTransform"):
            adapter.steer(model, wordlevel_tokenizer())

    def test_caller_vector_not_mutated(self):
        """Bound-mode steer + generate leaves the caller's SV tensors bit-identical."""
        sv = _sv(11)
        original = {lid: d.clone() for lid, d in sv.directions.items()}
        model = tiny_llama(num_layers=LAYERS, hidden=HIDDEN, heads=HEADS)
        adapter = ActivationAdapter(transform=AdditiveTransform(sv, strength=2.0), layer_ids=1, token_scope="all")
        p = _pipe(adapter, model)
        p.generate(input_ids=torch.arange(3, 7).unsqueeze(0), max_new_tokens=3, do_sample=False, eos_token_id=None)
        for lid, d in original.items():
            assert torch.equal(sv.directions[lid], d)


# gating
class TestGating:
    def _gated_adapter(self, threshold, score_value, condition_layer=0, behavior_layer=1):
        return ActivationAdapter(
            transform=AdditiveTransform(_sv(13), strength=1.0), layer_ids=[behavior_layer], token_scope="all",
            gate=_constant_gate(threshold, score_value, condition_layer),
        )

    def test_transform_fires_above_threshold(self):
        model = tiny_llama(num_layers=LAYERS, hidden=HIDDEN, heads=HEADS)
        input_ids = torch.arange(3, 7).unsqueeze(0)

        adapter_on = self._gated_adapter(threshold=0.5, score_value=0.9)
        p_on = _pipe(adapter_on, model)
        h_on = _hidden_at(model, 1, p_on, input_ids)

        adapter_off = self._gated_adapter(threshold=0.5, score_value=0.1)
        p_off = _pipe(adapter_off, model)
        h_off = _hidden_at(model, 1, p_off, input_ids)

        ref = _hidden_at(model, 1, _pipe(ActivationAdapter(
            transform=AdditiveTransform(_sv(13), strength=0.0), layer_ids=1, token_scope="all"), model), input_ids)

        assert not torch.allclose(h_on, ref, atol=1e-5)   # gate open -> steered
        assert torch.allclose(h_off, ref, atol=1e-5)       # gate closed -> unsteered

    def test_gate_cleared_between_generations(self):
        model = tiny_llama(num_layers=LAYERS, hidden=HIDDEN, heads=HEADS)
        adapter = self._gated_adapter(threshold=0.5, score_value=0.9)
        p = _pipe(adapter, model)
        input_ids = torch.arange(3, 7).unsqueeze(0)
        p.generate(input_ids=input_ids, max_new_tokens=3, do_sample=False, eos_token_id=None)
        h1 = _hidden_at(model, 1, p, input_ids)
        p.generate(input_ids=input_ids, max_new_tokens=3, do_sample=False, eos_token_id=None)
        h2 = _hidden_at(model, 1, p, input_ids)
        assert torch.allclose(h1, h2, atol=1e-6)

    def test_same_layer_condition_precedes_behavior(self):
        model = tiny_llama(num_layers=LAYERS, hidden=HIDDEN, heads=HEADS)
        order = []

        def _readout(pooled, layer_id):
            order.append("condition")
            return torch.full((pooled.size(0),), 0.9)

        class _RecordingTransform(BaseTransform):
            def apply(self, hidden_states, *, layer_id, token_mask, **kwargs):
                order.append("behavior")
                return hidden_states

        gate = Gate(Evidence((1,), CallableReadout(_readout)), PerKeyThreshold(threshold=0.5, comparator="ge"))
        adapter = ActivationAdapter(
            transform=_RecordingTransform(), layer_ids=[1], token_scope="all",
            gate=gate,
        )
        p = _pipe(adapter, model)
        _hidden_at(model, 1, p, torch.arange(3, 7).unsqueeze(0))
        assert order[0] == "condition"
        assert "behavior" in order


# reset() / get_hooks() gate re-sizing across consecutive generations
def _row_gated_adapter(threshold=0.5, score_value=0.9, condition_layer=0, behavior_layer=1):
    """A gated adapter whose readout returns per-row values, so it batches natively."""
    return ActivationAdapter(
        transform=AdditiveTransform(_sv(13), strength=1.0), layer_ids=[behavior_layer], token_scope="all",
        gate=_constant_gate(threshold, score_value, condition_layer),
    )


def test_consecutive_generations_across_batch_sizes():
    """A steered pipeline generating batch-of-4 then batch-of-2 leaves no state from the first call.

    `reset()` clears the row gate to a single row without knowing the next batch; `get_hooks` re-sizes
    it to the true batch before any gate read. The batch-of-2 outputs must match a freshly steered,
    identically configured pipeline given only the batch-of-2, and the gate must be sized to 2 after.
    """
    torch.manual_seed(0)
    model = tiny_llama(num_layers=LAYERS, hidden=HIDDEN, heads=HEADS)

    batch4 = torch.stack([torch.arange(3, 7), torch.arange(4, 8), torch.arange(5, 9), torch.arange(6, 10)])
    batch2 = torch.stack([torch.arange(3, 7), torch.arange(4, 8)])
    gen = dict(max_new_tokens=4, do_sample=False, eos_token_id=None)

    adapter = _row_gated_adapter()
    pipeline = _pipe(adapter, model)
    pipeline.generate(input_ids=batch4, **gen)
    out_after_4 = pipeline.generate(input_ids=batch2, **gen)

    fresh_model = tiny_llama(num_layers=LAYERS, hidden=HIDDEN, heads=HEADS)
    fresh_model.load_state_dict(model.state_dict())
    fresh_adapter = _row_gated_adapter()
    fresh_pipeline = _pipe(fresh_adapter, fresh_model)
    out_fresh = fresh_pipeline.generate(input_ids=batch2, **gen)

    assert torch.equal(out_after_4, out_fresh)  # no state from the batch-of-4 leaked into the batch-of-2
    assert adapter._gate.num_rows == 2  # get_hooks re-sized the gate past the unsized reset() clear


# CosineReadout
class TestCosineReadout:
    def test_matches_reference_cosine(self):
        """The readout equals the cosine between the last-token hidden state and the layer direction."""
        import torch.nn.functional as F

        sv = _sv(41)
        hidden = torch.randn(2, 5, HIDDEN)
        pooled = hidden[:, -1, :]  # "last" pooling over an unpadded batch

        def reference_rows(hidden, layer_id):
            direction = sv.directions[layer_id].to(hidden.dtype).to(hidden.device)
            direction = direction.squeeze(0) if direction.ndim == 2 else direction
            last_token = hidden[:, -1, :]
            return F.cosine_similarity(last_token, direction.unsqueeze(0), dim=-1)

        readout = CosineReadout(sv)
        for lid in range(LAYERS):
            rows = readout(pooled, lid)  # per-row [B]
            assert rows.shape == (2,)
            assert torch.allclose(rows, reference_rows(hidden, lid), atol=1e-6)

    def test_absent_layer_returns_zero(self):
        readout = CosineReadout(SteeringVector(model_type="x", directions={0: torch.randn(1, HIDDEN)}))
        out = readout(torch.randn(1, HIDDEN), 99)
        assert torch.equal(out, torch.zeros(1))

    def test_accepts_mapping(self):
        readout = CosineReadout({0: torch.randn(1, HIDDEN)})
        out = readout(torch.randn(1, HIDDEN), 0)
        assert isinstance(out, torch.Tensor) and out.shape == (1,)

    def test_junk_artifact_raises(self):
        with pytest.raises(TypeError, match="concrete SteeringVector or Mapping"):
            CosineReadout(ContrastiveFit(data={"positives": ["a"], "negatives": ["b"]}))

    def test_gated_end_to_end_with_readout(self):
        """A gated adapter using CosineReadout fires above threshold, holds below."""
        sv = _sv(43)
        input_ids = torch.arange(3, 7).unsqueeze(0)

        def _build(threshold):
            gate = Gate(
                Evidence((0,), CosineReadout(sv), pooling="last"),
                PerKeyThreshold(threshold=threshold, comparator="ge"),
            )
            return ActivationAdapter(
                transform=AdditiveTransform(sv, strength=3.0), layer_ids=[1], token_scope="all",
                gate=gate,
            )

        model = tiny_llama(num_layers=LAYERS, hidden=HIDDEN, heads=HEADS)
        ref = _hidden_at(model, 1, _pipe(ActivationAdapter(
            transform=AdditiveTransform(sv, strength=0.0), layer_ids=1, token_scope="all"), model), input_ids)

        # threshold below the achievable score -> gate opens -> steered
        model_on = tiny_llama(num_layers=LAYERS, hidden=HIDDEN, heads=HEADS)
        model_on.load_state_dict(model.state_dict())
        h_on = _hidden_at(model_on, 1, _pipe(_build(threshold=-1.0), model_on), input_ids)

        # threshold above 1.0 (impossible for cosine) -> gate stays closed -> unsteered
        model_off = tiny_llama(num_layers=LAYERS, hidden=HIDDEN, heads=HEADS)
        model_off.load_state_dict(model.state_dict())
        h_off = _hidden_at(model_off, 1, _pipe(_build(threshold=2.0), model_off), input_ids)

        assert not torch.allclose(h_on, ref, atol=1e-5)
        assert torch.allclose(h_off, ref, atol=1e-5)


# supports_batching
class TestSupportsBatching:
    def test_gated_and_ungated_both_batch_natively(self):
        # row-vectorized gates score/gate each prompt independently, so gated adapters no
        # longer fall back to sequential processing
        model = tiny_llama(num_layers=LAYERS, hidden=HIDDEN, heads=HEADS)
        ungated = ActivationAdapter(transform=AdditiveTransform(_sv()), layer_ids=1, token_scope="all")
        ungated.steer(model, wordlevel_tokenizer())
        assert ungated.supports_batching is True

        gated = ActivationAdapter(
            transform=AdditiveTransform(_sv()), layer_ids=[1], token_scope="all",
            gate=_constant_gate(threshold=0.5, value=0.9),
        )
        gated.steer(tiny_llama(num_layers=LAYERS, hidden=HIDDEN, heads=HEADS), wordlevel_tokenizer())
        assert gated.supports_batching is True

    def test_pipeline_batched_logprobs_when_gated(self):
        model = tiny_llama(num_layers=LAYERS, hidden=HIDDEN, heads=HEADS)
        adapter = ActivationAdapter(
            transform=AdditiveTransform(_sv()), layer_ids=[1], token_scope="all",
            gate=_constant_gate(threshold=0.5, value=0.9),
        )
        p = _pipe(adapter, model)
        assert p.supports_batching is True
        lp = p.compute_logprobs(input_ids=torch.arange(3, 7).unsqueeze(0), ref_output_ids=torch.tensor([[7, 8]]))
        assert lp.shape == (1, 2)


# registry discovery
class TestRegistry:
    def test_activation_adapter_registered(self):
        from steerability.algorithms.core.registry import REGISTRY
        assert "activation_adapter" in REGISTRY.get("state_control", {})
        method = REGISTRY["state_control"]["activation_adapter"]
        assert method.control_cls is _AA


# ControlSpec sweep + shared-source memoization
class TestControlSpecSweep:
    def test_grid_over_strength_and_layer(self):
        from steerability.algorithms.core.specs import ControlSpec

        sv = _sv(17)
        spec = ControlSpec(
            control_cls=ActivationAdapter,
            params={"token_scope": "all"},
            vars={
                "transform": [AdditiveTransform(sv, strength=1.0), AdditiveTransform(sv, strength=2.0)],
                "layer_ids": [1, 2],
            },
        )
        points = list(spec.iter_points({}))
        assert len(points) == 4  # 2 transforms x 2 layers

        input_ids = torch.arange(3, 7).unsqueeze(0)
        for point in points:
            kwargs = spec.resolve_params(point, {})
            adapter = ActivationAdapter(**kwargs)
            p = _pipe(adapter, tiny_llama(num_layers=LAYERS, hidden=HIDDEN, heads=HEADS))
            out = p.generate(input_ids=input_ids, max_new_tokens=2, do_sample=False, eos_token_id=None)
            assert out.size(1) >= 1

    def test_shared_source_fits_once_per_model(self):
        """One ContrastiveFit across two adapter configs fits once per model; templates clean."""
        from steerability.algorithms.state_control.common.estimators.base import BaseEstimator

        class _CountingEstimator(BaseEstimator):
            def __init__(self):
                self.calls = 0

            def fit(self, model, tokenizer, *, data, **kwargs):
                self.calls += 1
                return _sv(19)

        est = _CountingEstimator()
        source = ContrastiveFit(data={"positives": ["a"], "negatives": ["b"]}, estimator=est)
        model = tiny_llama(num_layers=LAYERS, hidden=HIDDEN, heads=HEADS)

        t1 = AdditiveTransform(source, strength=1.0)
        t2 = ProjectionTransform(source, alpha=1.0)
        a1 = ActivationAdapter(transform=t1, layer_ids=[1], token_scope="all")
        a2 = ActivationAdapter(transform=t2, layer_ids=[1], token_scope="all")
        a1.steer(model, wordlevel_tokenizer())
        a2.steer(model, wordlevel_tokenizer())

        assert est.calls == 1  # single-slot memo: one fit for the whole sweep on one model
        assert t1.is_bound is False and t2.is_bound is False  # templates unpolluted


# pipeline integration under multiplicity
class TestPipelineIntegration:
    def test_two_adapters_order_sensitive(self):
        sv = _sv(21)
        input_ids = torch.arange(3, 7).unsqueeze(0)

        def _final(order, model):
            controls = []
            for kind in order:
                if kind == "add":
                    controls.append(ActivationAdapter(
                        transform=AdditiveTransform(sv, strength=5.0), layer_ids=[1], token_scope="all"))
                else:
                    controls.append(ActivationAdapter(
                        transform=ProjectionTransform(sv, alpha=1.0), layer_ids=[1], token_scope="all"))
            p = _pipe(controls, model)
            return _hidden_at(model, 1, p, input_ids)

        model_a = tiny_llama(num_layers=LAYERS, hidden=HIDDEN, heads=HEADS)
        h_add_ablate = _final(["add", "ablate"], model_a)
        model_b = tiny_llama(num_layers=LAYERS, hidden=HIDDEN, heads=HEADS)
        model_b.load_state_dict(model_a.state_dict())
        h_ablate_add = _final(["ablate", "add"], model_b)

        assert not torch.allclose(h_add_ablate, h_ablate_add, atol=1e-4)

    def _shared_gate_pipeline(self, sv, model, driver_score, driver_condition_layer=0,
                              driver_layer=1, follower_layer=2):
        shared_gate = _constant_gate(threshold=0.5, value=driver_score, condition_layer=driver_condition_layer)
        driver = ActivationAdapter(
            transform=AdditiveTransform(sv, strength=1.0), layer_ids=[driver_layer], token_scope="all",
            gate=shared_gate,
        )
        follower = ActivationAdapter(
            transform=AdditiveTransform(sv, strength=1.0), layer_ids=[follower_layer], token_scope="all",
            gate=shared_gate, gate_driven_externally=True,
        )
        return driver, follower, _pipe([driver, follower], model)

    def test_shared_gate_follower_fires_iff_driver_opens(self):
        sv = _sv(23)
        input_ids = torch.arange(3, 7).unsqueeze(0)

        model_on = tiny_llama(num_layers=LAYERS, hidden=HIDDEN, heads=HEADS)
        driver_on, follower_on, p_on = self._shared_gate_pipeline(sv, model_on, driver_score=0.9)
        h_follower_on = _hidden_at(model_on, 2, p_on, input_ids)

        model_off = tiny_llama(num_layers=LAYERS, hidden=HIDDEN, heads=HEADS)
        model_off.load_state_dict(model_on.state_dict())
        driver_off, follower_off, p_off = self._shared_gate_pipeline(sv, model_off, driver_score=0.1)
        h_follower_off = _hidden_at(model_off, 2, p_off, input_ids)

        assert driver_on._gate is follower_on._gate
        assert follower_on.supports_batching is True  # row gates batch natively
        assert not torch.allclose(h_follower_on, h_follower_off, atol=1e-5)
