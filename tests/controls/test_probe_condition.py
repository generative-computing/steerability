"""Probe-driven steering tests: `Probe.as_gate` semantics, adapter equivalence, and the guards.

Runs hub-free on a tiny randomly-initialized Llama. Probes are hand-built with fixed weights;
biases derived from a preliminary `ProbeSet.read` split a batch into open and closed rows, so
adapter decisions can be compared row-for-row against direct reads.
"""
import pytest
import torch

from aisteer360.algorithms.core.internals.fingerprint import model_fingerprint
from aisteer360.algorithms.core.internals.probes.probe import Probe
from aisteer360.algorithms.core.internals.probes.probe_set import ProbeSet
from aisteer360.algorithms.core.steering_pipeline import SteeringPipeline
from aisteer360.algorithms.state_control.activation_adapter.control import ActivationAdapter
from aisteer360.algorithms.state_control.common.gating import (
    AffineReadout,
    CallableReadout,
    Evidence,
    Gate,
    PerKeyThreshold,
    SumThreshold,
)
from tests.utils.runtime_helpers import RecordingTransform
from tests.utils.tiny_models import tiny_llama, wordlevel_tokenizer

HIDDEN = 32
LAYERS = 4

PROMPTS = torch.tensor([[3, 4, 5, 6], [7, 8, 9, 10], [11, 12, 3, 5]])
GEN_KWARGS = {"do_sample": False, "eos_token_id": None}


def _unit_vector(seed: int, dim: int = HIDDEN) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(dim, generator=g)
    return v / v.norm()


def _probe(layer_ids, bias=0.0, seed=7, pooling="mean", meta=None):
    return Probe(
        model_type="llama",
        location="layer_input",
        pooling=pooling,
        layer_ids=list(layer_ids),
        weights={lid: _unit_vector(seed + lid) for lid in layer_ids},
        bias=bias,
        meta=meta or {},
    )


def _model_and_tokenizer(seed: int = 0):
    torch.manual_seed(seed)
    model = tiny_llama(num_layers=LAYERS, hidden=HIDDEN, heads=4)
    tokenizer = wordlevel_tokenizer()
    return model, tokenizer


def _steered_pipeline(model, tokenizer, controls) -> SteeringPipeline:
    pipeline = SteeringPipeline(controls=controls, model=model, tokenizer=tokenizer)
    pipeline.steer()
    return pipeline


def _splitting_probe(model, layer_ids, seed=7):
    """A probe whose bias splits PROMPTS into open and closed rows, or None if inseparable."""
    scores = ProbeSet({"p": _probe(layer_ids, seed=seed)}).read(model, PROMPTS).scores["p"]
    ordered = scores.sort().values
    if ordered[1] - ordered[0] < 1e-5:
        return None
    midpoint = float((ordered[0] + ordered[1]) / 2)
    return _probe(layer_ids, bias=-midpoint, seed=seed)


class TestAsGate:
    def test_returns_gate_with_probe_evidence_and_bias(self):
        probe = _probe([1, 2], bias=0.5)
        gate = probe.as_gate()
        assert isinstance(gate, Gate)
        assert gate.evidence.layer_ids == (1, 2)
        assert gate.evidence.pooling == "mean"
        assert isinstance(gate.evidence.readout, AffineReadout)
        assert isinstance(gate.rule, SumThreshold)
        assert gate.rule.bias == 0.5

    def test_allow_model_mismatch_disarms_readout_fingerprint(self):
        probe = _probe([1], meta={"model_fingerprint": "abcd"})
        assert probe.as_gate().evidence.readout.model_fingerprint == "abcd"
        assert probe.as_gate(allow_model_mismatch=True).evidence.readout.model_fingerprint is None

    def test_readout_zero_for_absent_layer(self):
        gate = _probe([1]).as_gate()
        assert gate.evidence.readout(torch.randn(2, HIDDEN), 3).tolist() == [0.0, 0.0]


class TestAdapterEquivalence:
    @pytest.mark.parametrize("layer_ids", [[1], [1, 2]], ids=["single_layer", "multi_layer"])
    def test_adapter_decisions_match_probe_set_read(self, layer_ids):
        model, tokenizer = _model_and_tokenizer()
        probe = _splitting_probe(model, layer_ids)
        if probe is None:
            pytest.skip("tiny-model probe scores not separable for this seed")

        expected = ProbeSet({"p": probe}).read(model, PROMPTS).decisions["p"]
        assert expected.any() and not expected.all()  # the bias genuinely splits the batch

        adapter = ActivationAdapter(
            transform=RecordingTransform(value=0.5),
            layer_ids=[3],
            hook_point="layer_input",
            gate=probe.as_gate(),
        )
        pipeline = _steered_pipeline(model, tokenizer, [adapter])
        pipeline.generate(input_ids=PROMPTS, max_new_tokens=2, **GEN_KWARGS)

        assert adapter._gate.open_rows().tolist() == expected.tolist()

    def test_prompt_scored_once_and_decision_freezes(self):
        model, tokenizer = _model_and_tokenizer()
        probe = _probe([1], bias=1e9)  # always open

        calls: list[tuple] = []
        inner_readout = probe.as_gate().evidence.readout

        class _SpyReadout:
            wire_kind = None
            location = inner_readout.location
            model_fingerprint = inner_readout.model_fingerprint

            def __call__(self, pooled, layer_id):
                calls.append(tuple(pooled.shape))
                return inner_readout(pooled, layer_id)

            def export(self, layer_ids):
                return None

        gate = Gate(Evidence((1,), _SpyReadout()), SumThreshold(bias=probe.bias))
        adapter = ActivationAdapter(
            transform=RecordingTransform(value=0.5),
            layer_ids=[3],
            hook_point="layer_input",
            gate=gate,
        )
        pipeline = _steered_pipeline(model, tokenizer, [adapter])
        pipeline.generate(input_ids=PROMPTS[:1], max_new_tokens=4, **GEN_KWARGS)

        assert len(calls) == 1  # prefill only; the frozen decision stops further scoring
        frozen = adapter._gate.open_rows().clone()
        pipeline.generate(input_ids=PROMPTS[:1], max_new_tokens=4, **GEN_KWARGS)
        assert torch.equal(adapter._gate.open_rows(), frozen)  # re-armed and re-frozen per call
        assert len(calls) == 2


class TestLocationGuard:
    def test_layer_input_probe_on_layer_output_adapter_raises(self):
        model, tokenizer = _model_and_tokenizer()
        adapter = ActivationAdapter(
            transform=RecordingTransform(),
            layer_ids=[3],
            gate=_probe([1]).as_gate(),  # probe location "layer_input"; default hook_point "layer_output"
        )
        with pytest.raises(ValueError, match="expects features at 'layer_input'.*hooks 'layer_output'"):
            adapter.steer(model, tokenizer)

    def test_layer_output_probe_on_layer_input_adapter_raises(self):
        model, tokenizer = _model_and_tokenizer()
        probe = Probe(
            model_type="llama", location="layer_output", pooling="mean",
            layer_ids=[1], weights={1: _unit_vector(8)}, bias=0.0,
        )
        adapter = ActivationAdapter(
            transform=RecordingTransform(),
            layer_ids=[3],
            hook_point="layer_input",
            gate=probe.as_gate(),
        )
        with pytest.raises(ValueError, match="expects features at 'layer_output'.*hooks 'layer_input'"):
            adapter.steer(model, tokenizer)

    def test_matching_location_passes(self):
        model, tokenizer = _model_and_tokenizer()
        adapter = ActivationAdapter(
            transform=RecordingTransform(),
            layer_ids=[3],
            hook_point="layer_input",
            gate=_probe([1]).as_gate(),
        )
        adapter.steer(model, tokenizer)


class TestFingerprintGuard:
    def test_probe_from_other_model_raises_and_escape_disarms(self):
        model_a, _ = _model_and_tokenizer(seed=0)
        model_b, tokenizer = _model_and_tokenizer(seed=1)
        probe = _probe([1], meta={"model_fingerprint": model_fingerprint(model_a)})

        adapter = ActivationAdapter(
            transform=RecordingTransform(),
            layer_ids=[3],
            hook_point="layer_input",
            gate=probe.as_gate(),
        )
        with pytest.raises(ValueError, match="different model"):
            adapter.steer(model_b, tokenizer)

        disarmed = ActivationAdapter(
            transform=RecordingTransform(),
            layer_ids=[3],
            hook_point="layer_input",
            gate=probe.as_gate(allow_model_mismatch=True),
        )
        disarmed.steer(model_b, tokenizer)

    def test_matching_fingerprint_passes(self):
        model, tokenizer = _model_and_tokenizer()
        probe = _probe([1], meta={"model_fingerprint": model_fingerprint(model)})
        adapter = ActivationAdapter(
            transform=RecordingTransform(),
            layer_ids=[3],
            hook_point="layer_input",
            gate=probe.as_gate(),
        )
        adapter.steer(model, tokenizer)

    def test_hand_built_probe_with_empty_meta_never_trips(self):
        model_a, _ = _model_and_tokenizer(seed=0)
        model_b, tokenizer = _model_and_tokenizer(seed=1)
        adapter = ActivationAdapter(
            transform=RecordingTransform(),
            layer_ids=[3],
            hook_point="layer_input",
            gate=_probe([1]).as_gate(),
        )
        adapter.steer(model_b, tokenizer)


class TestCallableReadoutGate:
    def test_readout_without_validation_metadata_passes_steer(self):
        model, tokenizer = _model_and_tokenizer()
        gate = Gate(
            Evidence((1,), CallableReadout(lambda pooled, layer_id: torch.zeros(pooled.size(0)))),
            PerKeyThreshold(threshold=0.5, comparator="ge"),
        )
        adapter = ActivationAdapter(
            transform=RecordingTransform(),
            layer_ids=[3],
            gate=gate,
        )
        adapter.steer(model, tokenizer)
