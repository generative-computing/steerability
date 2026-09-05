"""Deterministic tests for pairwise-PCA construction and sign orientation.

Exercises the pure helpers in `contrastive_direction` (no model forward passes): `pca_pairwise`
preserves the shared positive-minus-negative contrast, and every PCA direction is oriented so the
positive class projects above the negative class.
"""
import pytest
import torch
from sklearn.decomposition import PCA

from steerability.algorithms.core.internals.data import ContrastivePairs
from steerability.algorithms.state_control.common.estimators.contrastive_direction import (
    ContrastiveDirectionEstimator,
    _orient_direction,
    _prepare_pca_samples,
)
from steerability.algorithms.state_control.common.estimators.mean_difference import MeanDifferenceEstimator
from steerability.algorithms.state_control.common.fit_specs import VectorTrainSpec
from tests.utils.tiny_models import tiny_llama, wordlevel_tokenizer


def _fit_direction(Hp: torch.Tensor, Hn: torch.Tensor, method: str) -> torch.Tensor:
    """Replicate the estimator's per-layer PCA + orientation on pooled tensors."""
    samples = _prepare_pca_samples(Hp, Hn, method)
    pca = PCA(n_components=1)
    pca.fit(samples.numpy())
    direction = torch.from_numpy(pca.components_[0]).float()
    return _orient_direction(direction, Hp, Hn)


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-12))


class TestPairwisePreservesContrast:
    def test_shared_contrast_survives_symmetric_centering(self):
        # pair differences carry a large shared component on axis 0 and a small varying one on axis 1
        torch.manual_seed(0)
        N, H = 8, 2
        e0 = torch.tensor([1.0, 0.0])
        e1 = torch.tensor([0.0, 1.0])
        deltas = torch.stack([10.0 * e0 + s * e1 for s in torch.linspace(-1.0, 1.0, N)])  # [N, 2]

        Hn = torch.randn(N, H)
        Hp = Hn + deltas

        direction = _fit_direction(Hp, Hn, "pca_pairwise")

        assert abs(_cos(direction, e0)) > 0.9
        assert abs(_cos(direction, e1)) < 0.3
        assert abs(direction[0]) > abs(direction[1])


class TestOrientation:
    def _pos_neg(self, sign: int):
        # positives on +axis-0, negatives on -axis-0; sign flips the two clusters
        torch.manual_seed(1)
        N, H = 6, 3
        base = torch.randn(N, H) * 0.05
        Hp = base + sign * torch.tensor([1.0, 0.0, 0.0])
        Hn = base - sign * torch.tensor([1.0, 0.0, 0.0])
        return Hp, Hn

    def test_pairwise_positive_projects_above_negative(self):
        Hp, Hn = self._pos_neg(sign=+1)
        d = _fit_direction(Hp, Hn, "pca_pairwise")
        assert float((Hp.mean(0) - Hn.mean(0)) @ d) > 0

    def test_center_positive_projects_above_negative(self):
        Hp, Hn = self._pos_neg(sign=+1)
        d = _fit_direction(Hp, Hn, "pca_center")
        assert float((Hp.mean(0) - Hn.mean(0)) @ d) > 0

    def test_swapping_pos_neg_flips_direction(self):
        Hp, Hn = self._pos_neg(sign=+1)
        d1 = _fit_direction(Hp, Hn, "pca_pairwise")
        d2 = _fit_direction(Hn, Hp, "pca_pairwise")  # swapped roles
        assert _cos(d1, d2) < -0.9

    def test_orientation_flips_reversed_component(self):
        # a reversed component is flipped so positives project higher
        Hp = torch.tensor([[2.0, 0.0], [3.0, 0.0]])
        Hn = torch.tensor([[-2.0, 0.0], [-3.0, 0.0]])
        reversed_component = torch.tensor([-1.0, 0.0])
        oriented = _orient_direction(reversed_component, Hp, Hn)
        assert oriented[0] > 0
        assert float((Hp @ oriented).mean()) > float((Hn @ oriented).mean())

    def test_tie_falls_back_to_mean_margin(self):
        # one pair favors pos, one favors neg -> 0.5 majority tie -> mean-margin sign decides
        d = torch.tensor([1.0])

        # margins +1 and -1 -> mean margin 0 -> unchanged
        Hp_zero = torch.tensor([[2.0], [-2.0]])
        Hn_zero = torch.tensor([[1.0], [-1.0]])
        assert _orient_direction(d, Hp_zero, Hn_zero)[0] == 1.0

        # margins +1 and -4 -> mean margin -1.5 < 0 -> flips
        Hp_neg = torch.tensor([[2.0], [-5.0]])
        Hn_neg = torch.tensor([[1.0], [-1.0]])
        assert _orient_direction(d, Hp_neg, Hn_neg)[0] == -1.0


class TestSamplePreparation:
    def test_pairwise_samples_are_symmetric_and_mean_zero(self):
        Hp = torch.randn(5, 4)
        Hn = torch.randn(5, 4)
        samples = _prepare_pca_samples(Hp, Hn, "pca_pairwise")
        assert samples.shape == (10, 4)
        torch.testing.assert_close(samples.mean(dim=0), torch.zeros(4), atol=1e-6, rtol=0)
        torch.testing.assert_close(samples[:5], -samples[5:])

    def test_center_samples_are_grand_mean_zero(self):
        Hp = torch.randn(5, 4)
        Hn = torch.randn(5, 4)
        samples = _prepare_pca_samples(Hp, Hn, "pca_center")
        assert samples.shape == (10, 4)
        torch.testing.assert_close(samples.mean(dim=0), torch.zeros(4), atol=1e-5, rtol=0)

    def test_shape_mismatch_raises(self):
        try:
            _prepare_pca_samples(torch.randn(3, 4), torch.randn(2, 4), "pca_pairwise")
            assert False, "expected ValueError"
        except ValueError:
            pass

    def test_unsupported_method_raises(self):
        try:
            _prepare_pca_samples(torch.randn(2, 4), torch.randn(2, 4), "mean_diff")
            assert False, "expected ValueError"
        except ValueError:
            pass

    def test_non_finite_input_raises(self):
        Hp = torch.tensor([[float("inf"), 0.0]])
        Hn = torch.tensor([[0.0, 0.0]])
        try:
            _prepare_pca_samples(Hp, Hn, "pca_pairwise")
            assert False, "expected ValueError"
        except ValueError:
            pass


def _pairs():
    return ContrastivePairs(
        positives=["yes indeed", "affirmative reply", "sure absolutely", "of course yes"],
        negatives=["no thanks", "negative reply", "never decline", "of course not"],
    )


class TestSpecLocationValidation:
    def test_default_is_layer_output(self):
        assert VectorTrainSpec().location == "layer_output"

    def test_unsupported_location_raises(self):
        with pytest.raises(ValueError, match="location must be"):
            VectorTrainSpec(location="middle")  # type: ignore[arg-type]


class TestEstimatorFitLocation:
    """`spec.location` is consumed by the estimators: a layer-input fit reproduces the layer-output
    fit shifted down by one layer (adjacent residual-stream boundaries), proving the value flows
    through into `layerwise_tokenwise_hidden`."""

    NUM_LAYERS = 4

    def _model_tok(self, seed: int):
        torch.manual_seed(seed)
        model = tiny_llama(num_layers=self.NUM_LAYERS, hidden=32, heads=4)
        return model, wordlevel_tokenizer()

    def _fit_both(self, estimator, method, prompt_format):
        model, tokenizer = self._model_tok(seed=3)
        data = _pairs()
        out = estimator.fit(
            model, tokenizer, data=data,
            spec=VectorTrainSpec(method=method, accumulate="all", prompt_format=prompt_format,
                                 location="layer_output"),
        )
        inp = estimator.fit(
            model, tokenizer, data=data,
            spec=VectorTrainSpec(method=method, accumulate="all", prompt_format=prompt_format,
                                 location="layer_input"),
        )
        return out, inp

    def test_mean_diff_boundary_shift_exact(self):
        out, inp = self._fit_both(MeanDifferenceEstimator(), "mean_diff", "raw")
        assert sorted(out.directions) == list(range(self.NUM_LAYERS))
        assert sorted(inp.directions) == list(range(self.NUM_LAYERS))
        for l in range(1, self.NUM_LAYERS):
            torch.testing.assert_close(inp.directions[l], out.directions[l - 1])

    def test_pca_boundary_shift_close(self):
        out, inp = self._fit_both(ContrastiveDirectionEstimator(), "pca_center", "raw")
        assert sorted(out.directions) == list(range(self.NUM_LAYERS))
        assert sorted(inp.directions) == list(range(self.NUM_LAYERS))
        for l in range(1, self.NUM_LAYERS):
            assert torch.allclose(inp.directions[l], out.directions[l - 1], atol=1e-5, rtol=1e-4)

    def test_layer_input_layer_zero_present(self):
        model, tokenizer = self._model_tok(seed=4)
        inp = ContrastiveDirectionEstimator().fit(
            model, tokenizer, data=_pairs(),
            spec=VectorTrainSpec(method="pca_center", accumulate="all", prompt_format="raw",
                                 location="layer_input"),
        )
        assert set(inp.directions) == set(range(self.NUM_LAYERS))
        assert inp.directions[0].shape[-1] == 32


class TestDirectionalAblationSignInvariance:
    def test_projector_unchanged_by_sign(self):
        # the rank-one projector cc^T/(c·c) is invariant to a sign flip of c
        c = torch.tensor([0.3, -0.7, 1.1])
        proj_pos = torch.outer(c, c) / (c @ c + 1e-8)
        proj_neg = torch.outer(-c, -c) / ((-c) @ (-c) + 1e-8)
        torch.testing.assert_close(proj_pos, proj_neg)
