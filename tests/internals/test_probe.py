"""Probe artifact tests: decision math, pooling parity, serialization, format markers.

Runs model-free on deterministic tensors; the probe is pure feature math.
"""
import json

import pytest
import torch

from steerability.algorithms.core.internals.pooling import aggregate_condition_hidden
from steerability.algorithms.core.internals.probes.probe import Probe

HIDDEN = 8


def _probe(layer_ids, weights, bias=0.0, pooling="mean", meta=None):
    return Probe(
        model_type="llama",
        location="layer_input",
        pooling=pooling,
        layer_ids=layer_ids,
        weights=weights,
        bias=bias,
        meta=meta or {},
    )


class TestDecisionFunction:
    def test_single_layer_affine_score(self):
        w = torch.arange(HIDDEN, dtype=torch.float32)
        probe = _probe([0], {0: w}, bias=1.5)
        features = {0: torch.ones(2, HIDDEN)}
        expected = float(w.sum()) + 1.5
        assert torch.allclose(probe.decision_function(features), torch.tensor([expected, expected]))

    def test_multi_layer_contributions_sum(self):
        w0 = torch.ones(HIDDEN)
        w1 = 2.0 * torch.ones(HIDDEN)
        probe = _probe([0, 1], {0: w0, 1: w1}, bias=-3.0)
        features = {
            0: torch.ones(3, HIDDEN),  # contribution 8
            1: torch.ones(3, HIDDEN),  # contribution 16
        }
        assert torch.allclose(probe.decision_function(features), torch.full((3,), 8.0 + 16.0 - 3.0))

    def test_one_dim_features_treated_as_one_row(self):
        probe = _probe([0], {0: torch.ones(HIDDEN)})
        scores = probe.decision_function({0: torch.ones(HIDDEN)})
        assert scores.shape == (1,)

    def test_missing_layer_raises_keyerror_naming_layers(self):
        probe = _probe([0, 2], {0: torch.ones(HIDDEN), 2: torch.ones(HIDDEN)})
        with pytest.raises(KeyError, match=r"layer 2.*\[0\].*probe\.layer_ids"):
            probe.decision_function({0: torch.ones(1, HIDDEN)})

    def test_predict_ties_open(self):
        w = torch.zeros(HIDDEN)
        probe = _probe([0], {0: w}, bias=0.0)  # every score is exactly 0
        assert probe.predict({0: torch.randn(4, HIDDEN)}).tolist() == [True] * 4

    def test_scores_float32_cpu(self):
        probe = _probe([0], {0: torch.ones(HIDDEN)})
        scores = probe.decision_function({0: torch.randn(2, HIDDEN, dtype=torch.float64)})
        assert scores.dtype == torch.float32
        assert scores.device.type == "cpu"


class TestConstruction:
    def test_weights_coerced_to_float32_vectors(self):
        probe = _probe([0], {0: torch.ones(1, HIDDEN, dtype=torch.float16)})
        assert probe.weights[0].dtype == torch.float32
        assert probe.weights[0].shape == (HIDDEN,)

    def test_missing_weight_for_layer_raises(self):
        with pytest.raises(ValueError, match=r"\[1\]"):
            _probe([0, 1], {0: torch.ones(HIDDEN)})

    def test_empty_layer_ids_raise(self):
        with pytest.raises(ValueError, match="at least one layer"):
            _probe([], {})

    def test_bad_pooling_raises(self):
        with pytest.raises(ValueError, match="pooling"):
            _probe([0], {0: torch.ones(HIDDEN)}, pooling="median")


class TestScoreHidden:
    def test_parity_with_pooling_helpers_on_padded_batch(self):
        g = torch.Generator().manual_seed(0)
        hidden = {0: torch.randn(2, 5, HIDDEN, generator=g)}
        mask = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]])  # row 0 left-padded
        w = torch.randn(HIDDEN, generator=g)

        for pooling in ("mean", "last"):
            probe = _probe([0], {0: w}, bias=0.25, pooling=pooling)
            features = {0: aggregate_condition_hidden(hidden[0], pooling, attention_mask=mask)}
            assert torch.allclose(
                probe.score_hidden(hidden, prompt_mask=mask),
                probe.decision_function(features),
            )

    def test_last_pooling_selects_last_real_token(self):
        hidden = torch.zeros(1, 4, HIDDEN)
        hidden[0, 2] = 1.0  # last real position
        hidden[0, 3] = 99.0  # pad position
        mask = torch.tensor([[1, 1, 1, 0]])
        probe = _probe([0], {0: torch.ones(HIDDEN)}, pooling="last")
        assert float(probe.score_hidden({0: hidden}, prompt_mask=mask)) == pytest.approx(HIDDEN)

    def test_missing_layer_raises(self):
        probe = _probe([1], {1: torch.ones(HIDDEN)})
        with pytest.raises(KeyError, match="layer 1"):
            probe.score_hidden({0: torch.zeros(1, 2, HIDDEN)})


class TestSerialization:
    def _rich_probe(self):
        g = torch.Generator().manual_seed(3)
        return _probe(
            [1, 3],
            {1: torch.randn(HIDDEN, generator=g), 3: torch.randn(HIDDEN, generator=g)},
            bias=-0.75,
            pooling="last",
            meta={"method": "lda", "n_pos": 4, "n_neg": 4, "model_fingerprint": "abc123",
                  "layer_sweep": [{"layer_id": 1, "f1": 1.0}]},
        )

    def test_roundtrip(self, tmp_path):
        probe = self._rich_probe()
        probe.save(tmp_path / "probe")
        loaded = Probe.load(tmp_path / "probe")

        assert loaded.model_type == probe.model_type
        assert loaded.location == probe.location
        assert loaded.pooling == probe.pooling
        assert loaded.layer_ids == probe.layer_ids
        assert loaded.bias == probe.bias
        assert loaded.meta == probe.meta
        for lid in probe.layer_ids:
            assert torch.equal(loaded.weights[lid], probe.weights[lid])

    def test_loaded_scores_match(self, tmp_path):
        probe = self._rich_probe()
        probe.save(tmp_path / "probe")
        loaded = Probe.load(tmp_path / "probe")
        features = {1: torch.randn(3, HIDDEN), 3: torch.randn(3, HIDDEN)}
        assert torch.equal(loaded.decision_function(features), probe.decision_function(features))

    def test_version_marker_verified_on_load(self, tmp_path):
        probe = self._rich_probe()
        probe.save(tmp_path / "probe")
        sidecar = tmp_path / "probe" / "probe.json"
        metadata = json.loads(sidecar.read_text())
        metadata["format_version"] = 99
        sidecar.write_text(json.dumps(metadata))
        with pytest.raises(ValueError, match="format version"):
            Probe.load(tmp_path / "probe")

    def test_polarity_marker_verified_on_load(self, tmp_path):
        probe = self._rich_probe()
        probe.save(tmp_path / "probe")
        sidecar = tmp_path / "probe" / "probe.json"
        metadata = json.loads(sidecar.read_text())
        metadata["polarity"] = "negatives_high"
        sidecar.write_text(json.dumps(metadata))
        with pytest.raises(ValueError, match="polarity"):
            Probe.load(tmp_path / "probe")
