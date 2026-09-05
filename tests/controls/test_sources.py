"""Tests for `ArtifactSource` / `ContrastiveFit` (v5: transforms carry their artifact).

Covers `ContrastiveFit.resolve` (fit + single-weakref-slot memo + defensive clone + normalize),
local warnings, ownership, and the private `_as_artifact_source` coercions. Hub-free tiny Llama.
"""
import warnings

import pytest
import torch

from steerability.algorithms.state_control.common.estimators.base import BaseEstimator
from steerability.algorithms.state_control.common.sources import (
    ArtifactSource,
    ContrastiveFit,
    _as_artifact_source,
    _Precomputed,
)
from steerability.algorithms.state_control.common.steering_vector import SteeringVector
from tests.utils.tiny_models import tiny_llama, wordlevel_tokenizer

HIDDEN = 32
HEADS = 4
LAYERS = 4


class _CountingEstimator(BaseEstimator):
    """A fit-counting estimator returning a fixed vector (so memo/ownership are testable)."""

    def __init__(self):
        self.calls = 0

    def fit(self, model, tokenizer, *, data, **kwargs) -> SteeringVector:
        self.calls += 1
        g = torch.Generator().manual_seed(100)
        return SteeringVector(
            model_type="llama",
            directions={lid: torch.randn(1, HIDDEN, generator=g) for lid in range(LAYERS)},
        )


def _pairs():
    return {"positives": ["good"], "negatives": ["bad"]}


class TestContrastiveFitResolve:
    def test_resolve_returns_steering_vector(self):
        est = _CountingEstimator()
        fit = ContrastiveFit(data=_pairs(), estimator=est)
        model = tiny_llama(num_layers=LAYERS, hidden=HIDDEN, heads=HEADS)
        sv = fit.resolve(model, wordlevel_tokenizer())
        assert isinstance(sv, SteeringVector)
        assert set(sv.directions.keys()) == set(range(LAYERS))
        assert est.calls == 1

    def test_memo_same_model_fits_once(self):
        est = _CountingEstimator()
        fit = ContrastiveFit(data=_pairs(), estimator=est)
        model = tiny_llama(num_layers=LAYERS, hidden=HIDDEN, heads=HEADS)
        tok = wordlevel_tokenizer()
        a = fit.resolve(model, tok)
        b = fit.resolve(model, tok)
        assert est.calls == 1                       # one fit for two resolves
        assert a is not b                           # distinct clones
        assert torch.equal(a.directions[0], b.directions[0])  # equal tensors

    def test_memo_different_model_refits(self):
        est = _CountingEstimator()
        fit = ContrastiveFit(data=_pairs(), estimator=est)
        tok = wordlevel_tokenizer()
        fit.resolve(tiny_llama(num_layers=LAYERS, hidden=HIDDEN, heads=HEADS), tok)
        fit.resolve(tiny_llama(num_layers=LAYERS, hidden=HIDDEN, heads=HEADS), tok)
        assert est.calls == 2

    def test_memo_alternating_models_refits_each_switch(self):
        est = _CountingEstimator()
        fit = ContrastiveFit(data=_pairs(), estimator=est)
        tok = wordlevel_tokenizer()
        m_a = tiny_llama(num_layers=LAYERS, hidden=HIDDEN, heads=HEADS)
        m_b = tiny_llama(num_layers=LAYERS, hidden=HIDDEN, heads=HEADS)
        fit.resolve(m_a, tok)
        fit.resolve(m_b, tok)
        fit.resolve(m_a, tok)
        assert est.calls == 3  # A -> B -> A refits on each switch (documented)

    def test_normalize_produces_unit_norm_master(self):
        est = _CountingEstimator()
        fit = ContrastiveFit(data=_pairs(), estimator=est, normalize=True)
        model = tiny_llama(num_layers=LAYERS, hidden=HIDDEN, heads=HEADS)
        sv = fit.resolve(model, wordlevel_tokenizer())
        for direction in sv.directions.values():
            assert float(direction.norm()) == pytest.approx(1.0, abs=1e-5)

    def test_clone_ownership_master_untouched(self):
        est = _CountingEstimator()
        fit = ContrastiveFit(data=_pairs(), estimator=est)
        model = tiny_llama(num_layers=LAYERS, hidden=HIDDEN, heads=HEADS)
        tok = wordlevel_tokenizer()
        a = fit.resolve(model, tok)
        a.directions[0].add_(999.0)                 # mutate the caller's clone
        b = fit.resolve(model, tok)                 # memo hit -> fresh clone
        assert not torch.equal(a.directions[0], b.directions[0])
        assert float(b.directions[0].abs().max()) < 100.0  # master unpolluted


class TestBuiltinDispatch:
    def test_mean_diff_dispatch_fits(self):
        fit = ContrastiveFit(data=_pairs(), method="mean_diff", accumulate="last_token", prompt_format="raw")
        model = tiny_llama(num_layers=LAYERS, hidden=HIDDEN, heads=HEADS)
        sv = fit.resolve(model, wordlevel_tokenizer())
        assert isinstance(sv, SteeringVector) and sv.directions


class TestLocationForwarding:
    def test_location_forwarded_into_built_spec(self, monkeypatch):
        import steerability.algorithms.state_control.common.sources as sources

        captured = {}

        class _SpecCapture(BaseEstimator):
            def fit(self, model, tokenizer, *, data, spec, **kwargs) -> SteeringVector:
                captured["location"] = spec.location
                return SteeringVector(
                    model_type="llama",
                    directions={lid: torch.randn(1, HIDDEN) for lid in range(LAYERS)},
                )

        monkeypatch.setattr(sources, "ContrastiveDirectionEstimator", _SpecCapture)
        fit = ContrastiveFit(data=_pairs(), method="pca_center", prompt_format="raw", location="layer_input")
        fit.resolve(tiny_llama(num_layers=LAYERS, hidden=HIDDEN, heads=HEADS), wordlevel_tokenizer())
        assert captured["location"] == "layer_input"


class TestLocalWarnings:
    def test_estimator_with_spec_fields_warns(self):
        with pytest.warns(UserWarning, match="ignored when a custom estimator"):
            ContrastiveFit(data=_pairs(), estimator=_CountingEstimator(), method="mean_diff")

    def test_estimator_with_custom_location_warns_and_names_location(self):
        with pytest.warns(UserWarning, match="location") as record:
            ContrastiveFit(data=_pairs(), estimator=_CountingEstimator(), location="layer_input")
        assert any("location" in str(w.message) for w in record)

    def test_default_location_with_estimator_no_extra_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ContrastiveFit(data=_pairs(), estimator=_CountingEstimator(), location="layer_output")

    def test_estimator_kwargs_without_estimator_warns(self):
        with pytest.warns(UserWarning, match="estimator_kwargs is inert"):
            ContrastiveFit(data=_pairs(), estimator_kwargs={"foo": 1})

    def test_clean_estimator_no_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ContrastiveFit(data=_pairs(), estimator=_CountingEstimator())

    def test_bad_data_raises(self):
        with pytest.raises(TypeError):
            ContrastiveFit(data=123)


class TestAsArtifactSource:
    def test_source_passthrough(self):
        fit = ContrastiveFit(data=_pairs(), estimator=_CountingEstimator())
        assert _as_artifact_source(fit) is fit
        assert isinstance(fit, ArtifactSource)

    def test_steering_vector_wrapped(self):
        sv = SteeringVector(model_type="x", directions={0: torch.randn(1, HIDDEN)})
        src = _as_artifact_source(sv)
        assert isinstance(src, _Precomputed)
        resolved = src.resolve(None, None)
        assert resolved is not sv                    # clone
        assert torch.equal(resolved.directions[0], sv.directions[0])

    def test_mapping_wrapped(self):
        mapping = {0: torch.randn(1, HIDDEN), 2: torch.randn(1, HIDDEN)}
        src = _as_artifact_source(mapping)
        assert isinstance(src, _Precomputed)
        resolved = src.resolve(None, None)
        assert set(resolved.directions.keys()) == {0, 2}
        assert resolved.model_type == "unknown"

    def test_mapping_non_tensor_raises(self):
        with pytest.raises(TypeError, match="torch.Tensor"):
            _as_artifact_source({0: [1, 2, 3]})

    def test_junk_raises(self):
        with pytest.raises(TypeError, match="SteeringVector, a Mapping"):
            _as_artifact_source(123)
