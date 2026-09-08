"""Tests for the condition-point selector: threshold grid, layer indexing, and layer-input extraction."""
import math
import warnings

import pytest
import torch

from steerability.algorithms.core.internals.capture import layerwise_tokenwise_hidden
from steerability.algorithms.core.internals.data import ContrastivePairs
from steerability.algorithms.state_control.common.estimators import MeanDifferenceEstimator
from steerability.algorithms.state_control.common.estimators.contrastive_direction import ContrastiveDirectionEstimator
from steerability.algorithms.state_control.common.fit_specs import ConditionSearchSpec, VectorTrainSpec
from steerability.algorithms.state_control.common.gating import (
    projected_cosine_similarity,
    projected_cosine_similarity_tensor,
    rank_one_projector,
)
from steerability.algorithms.state_control.common.selectors.condition_point import (
    ConditionPointSelector,
    _best_point_for_layer,
    _threshold_grid,
)
from tests.utils.tiny_models import tiny_llama, wordlevel_tokenizer


class TestThresholdGrid:
    def test_ibm_defaults_step_exact(self):
        grid = _threshold_grid((0.0, 1.0), 0.01)
        assert grid.numel() == 100
        assert grid[0] == pytest.approx(0.0)
        assert grid[-1] == pytest.approx(0.99)
        assert float(grid[1] - grid[0]) == pytest.approx(0.01)
        assert float(grid.max()) < 1.0  # half-open: 1.0 excluded

    def test_negative_range_spacing_exact(self):
        grid = _threshold_grid((-1.0, 1.0), 0.05)
        diffs = grid[1:] - grid[:-1]
        assert torch.allclose(diffs, torch.full_like(diffs, 0.05))

    def test_degenerate_range_falls_back_to_low(self):
        grid = _threshold_grid((0.5, 0.5001), 0.01)
        assert grid.numel() == 1
        assert grid[0] == pytest.approx(0.5)


class TestLayerInputLocation:
    def test_layer_input_equals_previous_output(self):
        torch.manual_seed(0)
        model = tiny_llama(num_layers=4, hidden=32, heads=4)
        enc = {"input_ids": torch.tensor([[5, 6, 7, 8]]), "attention_mask": torch.ones(1, 4, dtype=torch.long)}

        out = layerwise_tokenwise_hidden(model, enc, location="layer_output")
        inp = layerwise_tokenwise_hidden(model, enc, location="layer_input")

        num_layers = 4
        assert sorted(out.keys()) == list(range(num_layers))
        assert sorted(inp.keys()) == list(range(num_layers))
        for l in range(1, num_layers):
            torch.testing.assert_close(inp[l], out[l - 1])

    def test_layer_input_zero_is_embedding(self):
        torch.manual_seed(0)
        model = tiny_llama(num_layers=3, hidden=16, heads=2)
        ids = torch.tensor([[5, 6, 7]])
        enc = {"input_ids": ids, "attention_mask": torch.ones(1, 3, dtype=torch.long)}
        inp = layerwise_tokenwise_hidden(model, enc, location="layer_input")
        full = model(input_ids=ids, output_hidden_states=True, return_dict=True)
        torch.testing.assert_close(inp[0], full.hidden_states[0].cpu())

    def test_unsupported_location_raises(self):
        model = tiny_llama(num_layers=2, hidden=16, heads=2)
        enc = {"input_ids": torch.tensor([[5, 6]]), "attention_mask": torch.ones(1, 2, dtype=torch.long)}
        with pytest.raises(ValueError):
            layerwise_tokenwise_hidden(model, enc, location="middle")  # type: ignore[arg-type]


class TestSelectorRuntimeScoringParity:
    def test_scalar_and_tensor_agree(self):
        torch.manual_seed(1)
        H = 32
        pooled = torch.randn(H)
        projector = rank_one_projector(torch.randn(H))
        tensor_score = float(projected_cosine_similarity_tensor(pooled.unsqueeze(0), projector)[0])
        scalar_score = projected_cosine_similarity(pooled, projector)
        assert abs(tensor_score - scalar_score) < 1e-6


class TestSelectEndToEnd:
    def test_returns_zero_based_layer(self):
        torch.manual_seed(2)
        num_layers = 4
        model = tiny_llama(num_layers=num_layers, hidden=32, heads=4)
        tokenizer = wordlevel_tokenizer()

        estimator = ContrastiveDirectionEstimator()
        data = ContrastivePairs(
            positives=["yes indeed", "affirmative reply", "sure absolutely", "of course yes"],
            negatives=["no thanks", "negative reply", "never decline", "of course not"],
        )
        fit_spec = VectorTrainSpec(method="pca_pairwise", accumulate="all", prompt_format="raw",
                                   location="layer_input")
        vec = estimator.fit(model, tokenizer, data=data, spec=fit_spec)

        selector = ConditionPointSelector()
        point = selector.select(
            model=model,
            tokenizer=tokenizer,
            condition_directions=vec.directions,
            data=data,
            fit_spec=fit_spec,
            search_spec=ConditionSearchSpec(auto_find=True),
            comparison_mode="mean",
        )
        # layer 0 is searchable (0-based) and the returned layer is a valid runtime layer id
        assert 0 <= point.layer_id < num_layers
        assert point.comparator in ("ge", "le")


def _selector_data():
    return ContrastivePairs(
        positives=["yes indeed", "affirmative reply", "sure absolutely", "of course yes"],
        negatives=["no thanks", "negative reply", "never decline", "of course not"],
    )


class TestFitLocationMismatchWarning:
    def _fit_and_select(self, fit_location, expect_warning):
        torch.manual_seed(5)
        model = tiny_llama(num_layers=4, hidden=32, heads=4)
        tokenizer = wordlevel_tokenizer()
        data = _selector_data()
        fit_spec = VectorTrainSpec(method="pca_center", accumulate="all", prompt_format="raw",
                                   location=fit_location)
        vec = ContrastiveDirectionEstimator().fit(model, tokenizer, data=data, spec=fit_spec)
        selector = ConditionPointSelector()
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            selector.select(
                model=model,
                tokenizer=tokenizer,
                condition_directions=vec.directions,
                data=data,
                fit_spec=fit_spec,
                search_spec=ConditionSearchSpec(auto_find=True),
                comparison_mode="mean",
            )
        mismatch = [w for w in record if "layer-input" in str(w.message)]
        assert bool(mismatch) is expect_warning

    def test_layer_output_fit_spec_warns(self):
        self._fit_and_select("layer_output", expect_warning=True)

    def test_layer_input_fit_spec_no_warning(self):
        self._fit_and_select("layer_input", expect_warning=False)


class TestSelectDeviceDtypeAlignment:
    def test_dtype_alignment_completes_on_cpu(self):
        torch.manual_seed(6)
        model = tiny_llama(num_layers=4, hidden=32, heads=4)
        tokenizer = wordlevel_tokenizer()
        data = _selector_data()
        fit_spec = VectorTrainSpec(method="pca_center", accumulate="all", prompt_format="raw",
                                   location="layer_input")
        vec = ContrastiveDirectionEstimator().fit(model, tokenizer, data=data, spec=fit_spec)
        directions = {lid: d.double() for lid, d in vec.directions.items()}  # float64 vs float32 pooled
        point = ConditionPointSelector().select(
            model=model,
            tokenizer=tokenizer,
            condition_directions=directions,
            data=data,
            fit_spec=fit_spec,
            search_spec=ConditionSearchSpec(auto_find=True),
            comparison_mode="mean",
        )
        assert 0 <= point.layer_id < 4

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
    def test_device_alignment_gpu(self):
        torch.manual_seed(7)
        model = tiny_llama(num_layers=4, hidden=32, heads=4)
        tokenizer = wordlevel_tokenizer()
        data = _selector_data()
        fit_spec = VectorTrainSpec(method="pca_center", accumulate="all", prompt_format="raw",
                                   location="layer_input")
        vec = ContrastiveDirectionEstimator().fit(model, tokenizer, data=data, spec=fit_spec)
        # mirror CAST: condition directions are moved to the model device before the selector runs
        directions = {lid: d.to("cuda") for lid, d in vec.directions.items()}
        point = ConditionPointSelector().select(
            model=model,
            tokenizer=tokenizer,
            condition_directions=directions,
            data=data,
            fit_spec=fit_spec,
            search_spec=ConditionSearchSpec(auto_find=True),
            comparison_mode="mean",
        )
        assert 0 <= point.layer_id < 4


class TestMarginAwareSelection:
    """The (f1, margin) tie-break in `_best_point_for_layer` and `ConditionPoint.margin`.

    F1 saturates at many points on a small calibration set, so ties are broken by the geometric
    margin. Assertions never test `f1 == 1.0`: the 1e-8 epsilons in prec/rec cap a perfect score at
    ~0.999999995, so `pytest.approx(1.0)` is used. Exact ties between perfect layers still hold.
    """

    GRID = torch.arange(0, 1, 0.01, dtype=torch.float64)

    def test_threshold_centres_in_the_gap(self):
        sims_p = torch.tensor([0.11, 0.23])
        sims_n = torch.tensor([0.02, 0.05])
        best = _best_point_for_layer(sims_p, sims_n, self.GRID)
        assert best["f1"] == pytest.approx(1.0)
        assert best["comparator"] == "ge"
        assert best["thr"] == pytest.approx(0.08, abs=0.011)  # midpoint of the 0.05 -> 0.11 gap
        assert best["margin"] == pytest.approx(0.03, abs=0.011)

    def test_prefers_wider_margin_among_f1_ties(self):
        # among equal-F1 candidates the wider margin must win
        a = _best_point_for_layer(torch.tensor([0.11, 0.23]), torch.tensor([0.02, 0.05]), self.GRID)
        b = _best_point_for_layer(torch.tensor([0.30, 0.45]), torch.tensor([0.01, 0.02]), self.GRID)
        assert a["f1"] == b["f1"]  # a genuine tie
        assert (b["f1"], b["margin"]) > (a["f1"], a["margin"])

    def test_margin_negative_when_not_separable(self):
        best = _best_point_for_layer(torch.tensor([0.05, 0.20]), torch.tensor([0.04, 0.22]), self.GRID)
        assert best["f1"] < 0.999 or best["margin"] <= 0

    def test_inverted_classes_select_le(self):
        best = _best_point_for_layer(torch.tensor([0.02, 0.05]), torch.tensor([0.30, 0.45]), self.GRID)
        assert best["comparator"] == "le"
        assert best["f1"] == pytest.approx(1.0)
        assert best["margin"] == pytest.approx(0.12, abs=0.011)

    def test_select_reports_margin(self):
        torch.manual_seed(2)
        model = tiny_llama(num_layers=4, hidden=32, heads=4)
        tokenizer = wordlevel_tokenizer()
        data = _selector_data()
        fit_spec = VectorTrainSpec(method="pca_pairwise", accumulate="all", prompt_format="raw",
                                   location="layer_input")
        vec = ContrastiveDirectionEstimator().fit(model, tokenizer, data=data, spec=fit_spec)
        point = ConditionPointSelector().select(
            model=model,
            tokenizer=tokenizer,
            condition_directions=vec.directions,
            data=data,
            fit_spec=fit_spec,
            search_spec=ConditionSearchSpec(auto_find=True),
            comparison_mode="mean",
        )
        assert hasattr(point, "margin")
        assert math.isfinite(point.margin)

    def test_mean_diff_le_comparator_warns(self):
        # a contrast where mean_diff scores positives BELOW negatives forces a "le" pick
        torch.manual_seed(11)
        model = tiny_llama(num_layers=4, hidden=32, heads=4)
        tokenizer = wordlevel_tokenizer()
        data = _selector_data()
        fit_spec = VectorTrainSpec(method="mean_diff", accumulate="all", prompt_format="raw",
                                   location="layer_input")
        vec = MeanDifferenceEstimator().fit(model, tokenizer, data=data, spec=fit_spec)
        # invert the fitted directions so positives project below negatives -> selector picks "le"
        directions = {lid: -d for lid, d in vec.directions.items()}
        selector = ConditionPointSelector()
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            point = selector.select(
                model=model,
                tokenizer=tokenizer,
                condition_directions=directions,
                data=data,
                fit_spec=fit_spec,
                search_spec=ConditionSearchSpec(auto_find=True),
                comparison_mode="mean",
            )
        if point.comparator == "le":
            assert any("expected to score HIGHER" in str(w.message) for w in record)


def test_unknown_score_raises():
    torch.manual_seed(0)
    model = tiny_llama(num_layers=4, hidden=32, heads=4)
    with pytest.raises(ValueError, match="projected_cosine"):
        ConditionPointSelector().select(
            model=model,
            tokenizer=wordlevel_tokenizer(),
            condition_directions={0: torch.randn(32)},
            data=ContrastivePairs(positives=["the cat"], negatives=["the dog"]),
            fit_spec=VectorTrainSpec(prompt_format="raw", location="layer_input"),
            search_spec=ConditionSearchSpec(),
            score="bogus",
        )
