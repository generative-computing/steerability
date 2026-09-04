"""Mutation-guard tests for caller-supplied SteeringVectors (Issue 2).

A precomputed vector shared across controls or across a Benchmark/ControlSpec sweep must not be
silently rescaled/re-cast by the first control that uses it. CAA, ActAdd, and CAST all clone the
resolved vector before any in-place `.to()` / normalization.

Runs hub-free on a tiny randomly-initialized Llama.
"""
import torch

from aisteer360.algorithms.core.steering_pipeline import SteeringPipeline
from aisteer360.algorithms.state_control.act_add.control import ActAdd
from aisteer360.algorithms.state_control.caa.control import CAA
from aisteer360.algorithms.state_control.cast.control import CAST
from aisteer360.algorithms.state_control.common.steering_vector import SteeringVector
from tests.utils.tiny_models import tiny_llama, wordlevel_tokenizer

HIDDEN = 32
LAYERS = 4


def _caa_vector(seed=0):
    g = torch.Generator().manual_seed(seed)
    # scale directions well away from unit norm so normalization would visibly change them
    dirs = {lid: torch.randn(1, HIDDEN, generator=g) * 4.0 for lid in range(LAYERS)}
    return SteeringVector(model_type="llama", directions=dirs)


def _steered_pipeline(control):
    torch.manual_seed(0)
    model = tiny_llama(num_layers=LAYERS, hidden=HIDDEN)
    tokenizer = wordlevel_tokenizer()
    pipeline = SteeringPipeline(controls=[control], model=model, tokenizer=tokenizer)
    pipeline.steer()
    return pipeline


class TestSteeringVectorClone:
    def test_clone_is_independent(self):
        original = _caa_vector(1)
        original.explained_variances = {0: 0.5}
        original.probe_accuracies = {(0, 0): 0.9}
        clone = original.clone()

        assert clone is not original
        for lid in original.directions:
            assert clone.directions[lid] is not original.directions[lid]
            torch.testing.assert_close(clone.directions[lid], original.directions[lid])
        # metadata dicts are copies, not shared references
        assert clone.explained_variances is not original.explained_variances
        assert clone.probe_accuracies is not original.probe_accuracies

        # mutating the clone leaves the original untouched
        clone.directions[0].mul_(0.0)
        assert original.directions[0].abs().sum() > 0


class TestCAAOwnership:
    def test_normalize_does_not_mutate_caller_vector(self):
        vector = _caa_vector(2)
        snapshot = {lid: d.clone() for lid, d in vector.directions.items()}
        orig_norm = vector.directions[1].norm().item()
        assert abs(orig_norm - 1.0) > 0.1  # far from unit so normalize would change it

        control = CAA(steering_vector=vector, layer_id=1, normalize_vector=True, token_scope="all")
        _steered_pipeline(control)

        # caller's tensors, dtype, and norm are all unchanged
        for lid, d in snapshot.items():
            torch.testing.assert_close(vector.directions[lid], d)
        assert vector.directions[1].dtype == torch.float32
        torch.testing.assert_close(vector.directions[1].norm(), torch.tensor(orig_norm))
        # the control's private copy is a different object
        assert control._steering_vector is not vector

    def test_shared_vector_normalize_on_off_are_independent(self):
        vector = _caa_vector(3)

        normed = CAA(steering_vector=vector, layer_id=1, normalize_vector=True, token_scope="all")
        plain = CAA(steering_vector=vector, layer_id=1, normalize_vector=False, token_scope="all")
        _steered_pipeline(normed)
        _steered_pipeline(plain)

        # the normalized control has a unit-norm direction; the plain one retains the raw norm
        normed_dir = normed._steering_vector.directions[1]
        plain_dir = plain._steering_vector.directions[1]
        torch.testing.assert_close(normed_dir.norm(), torch.tensor(1.0), atol=1e-5, rtol=1e-5)
        assert plain_dir.norm().item() > 1.5
        # the shared source vector was not touched by either
        assert vector.directions[1].norm().item() > 1.5


class TestActAddOwnership:
    def test_normalize_does_not_mutate_caller_vector(self):
        # ActAdd uses positional [T, H] directions; build a small T>1 vector
        g = torch.Generator().manual_seed(4)
        dirs = {lid: torch.randn(3, HIDDEN, generator=g) * 4.0 for lid in range(LAYERS)}
        vector = SteeringVector(model_type="llama", directions=dirs)
        snapshot = {lid: d.clone() for lid, d in vector.directions.items()}

        control = ActAdd(steering_vector=vector, layer_id=1, normalize_vector=True)
        _steered_pipeline(control)

        for lid, d in snapshot.items():
            torch.testing.assert_close(vector.directions[lid], d)
        assert control._steering_vector is not vector


class TestCASTOwnership:
    def test_behavior_vector_not_mutated(self):
        vector = _caa_vector(5)
        snapshot = {lid: d.clone() for lid, d in vector.directions.items()}

        control = CAST(
            behavior_vector=vector,
            behavior_layer_ids=[2, 3],
            behavior_vector_strength=1.0,
            token_scope="all",
        )
        _steered_pipeline(control)

        for lid, d in snapshot.items():
            torch.testing.assert_close(vector.directions[lid], d)
        assert vector.directions[2].dtype == torch.float32
