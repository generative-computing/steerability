"""Part C: `merge_controls` rejects the same control instance supplied twice.

Identity-only guard: a duplicated instance would register its hooks twice per generation
(double-applying the transform and double-advancing the shared position offset). Distinct
instances — even of the same class, and even sharing a gate object — compose freely and must
not regress (this is the composition surface the ActivationAdapter driver/follower pattern
relies on).
"""
import pytest
import torch

from steerability.algorithms.core.utils.controls import merge_controls
from steerability.algorithms.state_control.activation_adapter.control import ActivationAdapter
from steerability.algorithms.state_control.caa.control import CAA
from steerability.algorithms.state_control.common.gating import CallableReadout, Evidence, Gate, PerKeyThreshold
from steerability.algorithms.state_control.common.steering_vector import SteeringVector
from steerability.algorithms.state_control.common.transforms import AdditiveTransform

HIDDEN = 32


def _caa(seed: int) -> CAA:
    g = torch.Generator().manual_seed(seed)
    sv = SteeringVector(model_type="llama", directions={1: torch.randn(1, HIDDEN, generator=g)})
    return CAA(steering_vector=sv, layer_id=1, multiplier=1.0)


def _additive(seed: int) -> AdditiveTransform:
    g = torch.Generator().manual_seed(seed)
    sv = SteeringVector(model_type="llama", directions={1: torch.randn(1, HIDDEN, generator=g)})
    return AdditiveTransform(sv, strength=1.0)


def test_same_instance_twice_raises():
    control = _caa(1)
    with pytest.raises(ValueError, match="more than once"):
        merge_controls([control, control])


def test_duplicate_message_mentions_second_instance():
    control = _caa(1)
    with pytest.raises(ValueError, match="construct a second instance"):
        merge_controls([control, control])


def test_two_distinct_same_class_accepted():
    """Two distinct CAA instances (different vectors) compose; both retained in order."""
    a = _caa(1)
    b = _caa(2)
    assert a is not b
    merged = merge_controls([a, b])
    state_controls = merged["state_controls"]
    assert len(state_controls) == 2
    assert state_controls[0] is a
    assert state_controls[1] is b


def test_shared_gate_across_distinct_adapters_accepted():
    """Driver/follower ActivationAdapters sharing one gate object are distinct instances."""
    gate = Gate(
        Evidence((0,), CallableReadout(lambda pooled, layer_id: torch.ones(pooled.size(0)))),
        PerKeyThreshold(threshold=0.0, comparator="ge"),
    )

    driver = ActivationAdapter(
        transform=_additive(1),
        layer_ids=[2],
        gate=gate,
    )
    follower = ActivationAdapter(
        transform=_additive(2),
        layer_ids=[3],
        gate=gate,
        gate_driven_externally=True,
    )

    assert driver is not follower
    assert driver.gate is follower.gate  # a shared gate object, not a shared control instance

    merged = merge_controls([driver, follower])
    state_controls = merged["state_controls"]
    assert len(state_controls) == 2
    assert state_controls[0] is driver and state_controls[1] is follower
