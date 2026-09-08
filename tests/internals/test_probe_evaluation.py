"""Tests for `evaluate_probe`: held-out scoring against a fitted probe without refitting.

Hub-free on a tiny randomly-initialized Llama with a WordLevel tokenizer. A random model guarantees
no real class separation, so the assertions are structural, plus the identity that scoring the fit
data reproduces the fit-set F1 the probe recorded in its layer sweep.
"""
import pytest
import torch

from steerability.algorithms.core.internals.data import ContrastivePairs
from steerability.algorithms.core.internals.probes import ProbeEvaluation, evaluate_probe, fit_probe
from steerability.algorithms.core.internals.probes.fitting import ProbeFitSpec
from tests.utils.tiny_models import tiny_llama, wordlevel_tokenizer

HIDDEN = 32
LAYERS = 4

DATA = ContrastivePairs(
    positives=["the cat sat on mat", "the cat ran", "cat sat fast", "the mat cat"],
    negatives=["dog ran fast", "the dog ran", "dog sat on span", "fast dog span"],
)


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return tiny_llama(num_layers=LAYERS, hidden=HIDDEN, heads=4)


@pytest.fixture(scope="module")
def tokenizer():
    return wordlevel_tokenizer()


@pytest.fixture(scope="module")
def probe(model, tokenizer):
    spec = ProbeFitSpec(
        method="fisher",
        pooling="last",
        location="layer_output",
        prompt_format="raw",
        calibration="midpoint",
    )
    return fit_probe(model, tokenizer, data=DATA, spec=spec)


def test_returns_scores_of_the_right_lengths(model, tokenizer, probe):
    result = evaluate_probe(
        probe, model, tokenizer, DATA, prompt_format="raw"
    )
    assert isinstance(result, ProbeEvaluation)
    assert result.positive_scores.shape == (len(DATA.positives),)
    assert result.negative_scores.shape == (len(DATA.negatives),)


def test_accuracy_in_unit_interval(model, tokenizer, probe):
    result = evaluate_probe(probe, model, tokenizer, DATA, prompt_format="raw")
    assert 0.0 <= result.accuracy <= 1.0
    assert 0.0 <= result.f1 <= 1.0


def test_reproduces_fit_set_f1(model, tokenizer, probe):
    """Scoring the fit data reproduces the F1 the probe recorded at its chosen layer."""
    chosen = probe.layer_ids[0]
    recorded_f1 = next(
        entry["f1"] for entry in probe.meta["layer_sweep"] if entry["layer_id"] == chosen
    )
    result = evaluate_probe(probe, model, tokenizer, DATA, prompt_format="raw")
    assert result.f1 == pytest.approx(recorded_f1)


def test_never_recalibrates(model, tokenizer, probe):
    """A different score set does not shift the probe's bias (no recalibration)."""
    other = ContrastivePairs(
        positives=["the mat sat", "cat ran fast"],
        negatives=["dog span on", "the dog fast"],
    )
    before = probe.bias
    evaluate_probe(probe, model, tokenizer, other, prompt_format="raw")
    assert probe.bias == before
