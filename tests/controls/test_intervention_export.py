"""Tests for `export_intervention_spec` across the transform-runtime family: wire shapes,
artifact handling, placement mapping, and the coupling between exports and requirements."""
import pytest
import torch

pytest.importorskip("vllm_hook_plugins")
from vllm_hook_plugins.core.schema import parse_intervention_spec  # noqa: E402

from aisteer360.algorithms.core.execution import Capability, ModelFacts
from aisteer360.algorithms.core.internals.probes import Probe
from aisteer360.algorithms.state_control.act_add.control import ActAdd
from aisteer360.algorithms.state_control.activation_adapter.control import ActivationAdapter
from aisteer360.algorithms.state_control.angular_steering.control import AngularSteering
from aisteer360.algorithms.state_control.caa.control import CAA
from aisteer360.algorithms.state_control.common.gating import CallableReadout, Evidence, Gate, PerKeyThreshold
from aisteer360.algorithms.state_control.common.lowering import artifact_id_for
from aisteer360.algorithms.state_control.common.steering_vector import SteeringVector
from aisteer360.algorithms.state_control.common.transforms import (
    AdditiveTransform,
    AlignmentAdaptiveTransform,
    NormPreservingTransform,
    RotationTransform,
)
from aisteer360.algorithms.state_control.directional_ablation.control import DirectionalAblation
from aisteer360.algorithms.state_control.iti.control import ITI

LAYERS = 6
HIDDEN = 16
HEADS = 4


class _LayoutOnlySession:
    def __init__(self, layout: ModelFacts):
        self.layout = layout


@pytest.fixture()
def session():
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


def _probe(layers=(1, 2), location="layer_input", pooling="mean", bias=0.5) -> Probe:
    generator = torch.Generator().manual_seed(7)
    return Probe(
        model_type="llama",
        location=location,
        pooling=pooling,
        layer_ids=list(layers),
        weights={lid: torch.randn(HIDDEN, generator=generator) for lid in layers},
        bias=bias,
        meta={},
    )


def _supports_specs(control) -> bool:
    alternatives = control.requirements().generate
    return any(Capability.INTERVENTION_SPECS in alt.atoms for alt in alternatives)


class TestFamilyExports:

    def test_caa_exports_additive_op(self, session):
        control = CAA(steering_vector=_vector(), layer_id=2, multiplier=3.0, token_scope="after_prompt")
        control.steer(model=None, session=session)
        spec = control.export_intervention_spec()
        (op,) = spec.ops
        assert op["layers"] == [2]
        assert op["transform"]["kind"] == "additive"
        assert op["transform"]["strength"] == 3.0
        assert op["scope"] == {"kind": "after_prompt"}
        assert op["gate"] is None
        assert op["transform"]["artifact"] in spec.artifacts
        assert _supports_specs(control)

    def test_caa_norm_preservation_adds_modifier(self, session):
        control = CAA(steering_vector=_vector(), layer_id=2, use_norm_preservation=True)
        control.steer(model=None, session=session)
        (op,) = control.export_intervention_spec().ops
        assert op["transform"]["modifiers"] == [{"kind": "norm_preserving"}]

    def test_multi_row_vector_under_caa_raises_at_steer(self, session):
        control = CAA(steering_vector=_vector(k=3, layers=[2]), layer_id=2)
        with pytest.raises(ValueError, match="positional=True"):
            control.steer(model=None, session=session)

    def test_act_add_is_hook_only(self, session):
        control = ActAdd(steering_vector=_vector(), layer_id=2, multiplier=2.0)
        assert not _supports_specs(control)
        control.steer(model=None, session=session)
        assert control.export_intervention_spec() is None

    def test_act_add_layer_zero_is_hook_only(self, session):
        control = ActAdd(steering_vector=_vector(), layer_id=0)
        assert not _supports_specs(control)
        control.steer(model=None, session=session)
        assert control.export_intervention_spec() is None

    def test_act_add_from_prompt_pair_is_hook_only(self):
        control = ActAdd(positive_prompt="love", negative_prompt="hate", layer_id=2)
        assert not _supports_specs(control)

    def test_directional_ablation_shares_one_artifact_across_layers(self, session):
        shared = torch.randn(1, HIDDEN)
        vector = SteeringVector(model_type="llama", directions={2: shared, 3: shared.clone()})
        control = DirectionalAblation(steering_vector=vector, layer_ids=[2, 3])
        control.steer(model=None, session=session)
        spec = control.export_intervention_spec()
        # one op per (intervention, layer); identical content shares one content-addressed artifact
        assert [op["layers"] for op in spec.ops] == [[2], [3]]
        assert len(spec.artifacts) == 1
        assert spec.ops[0]["transform"]["artifact"] == spec.ops[1]["transform"]["artifact"]

    def test_directional_ablation_distinct_tensors_yield_one_op_per_layer(self, session):
        control = DirectionalAblation(steering_vector=_vector(), layer_ids=[2, 3])
        control.steer(model=None, session=session)
        spec = control.export_intervention_spec()
        assert sorted(layer for op in spec.ops for layer in op["layers"]) == [2, 3]
        assert len(spec.ops) == 2
        assert len(spec.artifacts) == 2

    def test_partial_ablation_is_hook_only(self, session):
        control = DirectionalAblation(steering_vector=_vector(), layer_ids=[2], alpha=0.5)
        assert not _supports_specs(control)
        control.steer(model=None, session=session)
        assert control.export_intervention_spec() is None

    def test_angular_steering_layer_output_exports_rotation(self, session):
        control = AngularSteering(
            steering_vector=_vector(k=2), target_degree=40.0, adaptive=True,
            intervention_point="layer_output",
        )
        control.steer(model=None, session=session)
        spec = control.export_intervention_spec()
        assert sorted(layer for op in spec.ops for layer in op["layers"]) == list(range(LAYERS))
        transform = spec.ops[0]["transform"]
        assert transform["kind"] == "rotation"
        assert transform["mode"] == "target"
        assert transform["modifiers"][0]["kind"] == "alignment_adaptive"
        assert transform["modifiers"][0]["threshold"] == 0.0
        assert _supports_specs(control)

    def test_angular_steering_norm_placement_is_hook_only(self, session):
        control = AngularSteering(steering_vector=_vector(k=2), angle=0.3)
        assert not _supports_specs(control)
        control.steer(model=None, session=session)
        assert control.export_intervention_spec() is None

    def test_iti_exports_zero_padded_head_vector(self, session):
        head_dim = HIDDEN // HEADS
        vector = SteeringVector(
            model_type="llama",
            directions={lid: torch.ones(HEADS, head_dim) for lid in range(LAYERS)},
            num_heads=HEADS,
            head_dim=head_dim,
        )
        control = ITI(steering_vector=vector, selected_heads=[(2, 1)], alpha=5.0)
        control.steer(model=None, session=session)
        spec = control.export_intervention_spec()
        (op,) = spec.ops
        assert op["layers"] == [2]
        assert op["transform"]["kind"] == "head_additive"
        padded = spec.artifacts[op["transform"]["artifact"]]["vector"]
        assert torch.equal(padded[1], torch.ones(head_dim))
        assert torch.equal(padded[0], torch.zeros(head_dim))
        assert _supports_specs(control)

    def test_iti_norm_preservation_is_hook_only(self, session):
        head_dim = HIDDEN // HEADS
        vector = SteeringVector(
            model_type="llama",
            directions={2: torch.ones(HEADS, head_dim)},
            num_heads=HEADS,
            head_dim=head_dim,
        )
        control = ITI(steering_vector=vector, selected_heads=[(2, 1)], use_norm_preservation=True)
        assert not _supports_specs(control)
        control.steer(model=None, session=session)
        assert control.export_intervention_spec() is None


class TestAdapterExports:

    def test_probe_gated_adapter_lowers_to_structured_gate(self, session):
        probe = _probe(layers=(1, 2), location="layer_input")
        control = ActivationAdapter(
            transform=AdditiveTransform(_vector().directions, strength=2.0),
            layer_ids=[3],
            hook_point="layer_input",
            token_scope="after_prompt",
            gate=probe.as_gate(),
        )
        control.steer(model=None, session=session)
        spec = control.export_intervention_spec()
        (op,) = spec.ops
        assert op["layers"] == [2]  # layer_input placement maps behavior layer 3 to wire layer 2
        gate = op["gate"]
        assert gate["layers"] == [1, 2]  # layer_input probes map directly
        assert gate["pooling"] == "mean"
        assert gate["readout"]["kind"] == "affine"
        assert gate["rule"] == {"kind": "sum_threshold", "bias": 0.5}
        weights = spec.artifacts[gate["readout"]["artifact"]]["weights"]
        assert weights.shape == (2, HIDDEN)
        assert torch.equal(weights[0], probe.weights[1])
        assert _supports_specs(control)

    def test_layer_output_probe_shifts_condition_layers(self, session):
        probe = _probe(layers=(1, 2), location="layer_output")
        control = ActivationAdapter(
            transform=AdditiveTransform(_vector().directions),
            layer_ids=[3],
            hook_point="layer_output",
            gate=probe.as_gate(),
        )
        control.steer(model=None, session=session)
        spec = control.export_intervention_spec()
        gate = spec.ops[0]["gate"]
        assert gate["layers"] == [2, 3]

    def test_callable_readout_gated_adapter_is_hook_only(self, session):
        gate = Gate(
            Evidence((1,), CallableReadout(lambda pooled, layer_id: pooled.mean(dim=-1))),
            PerKeyThreshold(threshold=0.4, comparator="ge"),
        )
        control = ActivationAdapter(
            transform=AdditiveTransform(_vector().directions),
            layer_ids=[3],
            gate=gate,
        )
        assert not _supports_specs(control)
        control.steer(model=None, session=session)
        assert control.export_intervention_spec() is None

    def test_gate_source_declares_kinds_before_binding(self):
        from aisteer360.algorithms.state_control.common.sources import ConditionPointSearch

        source = ConditionPointSearch(
            condition_vector=SteeringVector(model_type="llama", directions={1: torch.ones(1, HIDDEN)}),
            layer_ids=[1],
            threshold=0.05,
            comparator="ge",
        )
        control = ActivationAdapter(
            transform=AdditiveTransform(_vector().directions),
            layer_ids=[3],
            hook_point="layer_input",
            gate=source,
        )
        assert _supports_specs(control)


class TestExportMechanics:

    def test_modifier_order_is_innermost_first(self):
        from aisteer360.algorithms.state_control.common.lowering import lower_interventions
        from aisteer360.algorithms.state_control.common.specs import Intervention, TokenScope

        vector = _vector(k=2)
        transform = NormPreservingTransform(
            AlignmentAdaptiveTransform(RotationTransform(vector, angle=0.2, mode="offset"), vector)
        )
        spec = lower_interventions(
            [Intervention(layers=(1,), transform=transform, scope=TokenScope("all"))],
            num_layers=LAYERS,
        )
        assert [modifier["kind"] for modifier in spec.ops[0]["transform"]["modifiers"]] == [
            "alignment_adaptive",
            "norm_preserving",
        ]

    def test_exported_spec_passes_plugin_validation(self, session):
        control = CAA(steering_vector=_vector(), layer_id=2)
        control.steer(model=None, session=session)
        spec = control.export_intervention_spec()
        parsed = parse_intervention_spec(spec.to_wire(), num_layers=LAYERS)
        assert parsed.ops[0].transform_kind == "additive"

    def test_artifact_ids_are_dtype_and_device_stable(self):
        tensor = torch.randn(HIDDEN)
        id_f32, _ = artifact_id_for({"vector": tensor})
        id_f64, _ = artifact_id_for({"vector": tensor.to(torch.float64)})
        assert id_f32 == id_f64

    def test_export_and_requirement_share_one_verdict(self, session):
        """Every family configuration exports a spec exactly when its requirement advertises
        the intervention-spec alternative."""
        head_dim = HIDDEN // HEADS
        iti_vector = SteeringVector(
            model_type="llama",
            directions={2: torch.ones(HEADS, head_dim)},
            num_heads=HEADS, head_dim=head_dim,
        )
        configurations = [
            CAA(steering_vector=_vector(), layer_id=2),
            CAA(steering_vector=_vector(), layer_id=2, use_norm_preservation=True),
            ActAdd(steering_vector=_vector(), layer_id=2),
            ActAdd(steering_vector=_vector(), layer_id=0),
            DirectionalAblation(steering_vector=_vector(), layer_ids=[1, 4]),
            DirectionalAblation(steering_vector=_vector(), layer_ids=[1], alpha=0.7),
            AngularSteering(steering_vector=_vector(k=2), angle=0.2, intervention_point="layer_output"),
            AngularSteering(steering_vector=_vector(k=2), angle=0.2),
            ITI(steering_vector=iti_vector, selected_heads=[(2, 0)]),
            ITI(steering_vector=iti_vector, selected_heads=[(2, 0)], use_norm_preservation=True),
            ActivationAdapter(transform=AdditiveTransform(_vector().directions), layer_ids=[3]),
            ActivationAdapter(
                transform=AdditiveTransform(_vector().directions), layer_ids=[3],
                hook_point="layer_input", gate=_probe(location="layer_input").as_gate(),
            ),
            ActivationAdapter(
                transform=AdditiveTransform(_vector().directions), layer_ids=[3],
                gate=Gate(
                    Evidence((1,), CallableReadout(lambda pooled, layer_id: pooled.mean(dim=-1))),
                    PerKeyThreshold(threshold=0.4, comparator="ge"),
                ),
            ),
        ]
        session = _LayoutOnlySession(ModelFacts(
            num_layers=LAYERS, hidden_size=HIDDEN, num_attention_heads=HEADS,
            head_dim=HIDDEN // HEADS, dtype="float32", model_fingerprint="0" * 16,
        ))
        for control in configurations:
            control.steer(model=None, session=session)
            exported = control.export_intervention_spec()
            advertised = _supports_specs(control)
            assert (exported is not None) == advertised, type(control).__name__
