"""Probe fitting tests: calibration math, the anisotropy synthetic, orientation, and provenance.

The calibration and anisotropy cases run model-free on deterministic tensors; the fitting cases
run hub-free on a tiny randomly-initialized Llama with a WordLevel tokenizer and make structural
assertions only (a random model guarantees no class separation).
"""
import pytest
import torch
import torch.nn.functional as F

from steerability.algorithms.core.internals.data import ContrastivePairs, LabeledExamples
from steerability.algorithms.core.internals.fingerprint import model_fingerprint
from steerability.algorithms.core.internals.probes.fitting import (
    ProbeFitSpec,
    _fit_direction,
    calibrate_bias,
    fit_probe,
)
from steerability.algorithms.core.internals.probes.probe import Probe
from steerability.algorithms.core.internals.stats import ActivationStats
from tests.utils.tiny_models import tiny_llama, wordlevel_tokenizer

HIDDEN = 32
LAYERS = 4

DATA = ContrastivePairs(
    positives=["the cat sat on mat", "the cat ran", "cat sat fast", "the mat cat"],
    negatives=["dog ran fast", "the dog ran", "dog sat on span", "fast dog span"],
)

STATS_TEXTS = ["the cat sat on mat", "dog ran fast", "attention span on the mat", "the dog sat"]


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return tiny_llama(num_layers=LAYERS, hidden=HIDDEN, heads=4)


@pytest.fixture(scope="module")
def tokenizer():
    return wordlevel_tokenizer()


@pytest.fixture(scope="module")
def stats(model, tokenizer):
    return ActivationStats.estimate(
        model, tokenizer, STATS_TEXTS, location="layer_input", min_samples=1
    )


class TestCalibrateBias:
    def test_midpoint_exact(self):
        bias = calibrate_bias(torch.tensor([2.0, 4.0]), torch.tensor([0.0, 2.0]), "midpoint")
        assert bias == pytest.approx(-2.0)  # threshold (3 + 1) / 2

    def test_max_f1_exact(self):
        pos = torch.tensor([2.0, 3.0, 4.0])
        neg = torch.tensor([0.0, 1.0, 2.5])
        # thresholds are midpoints of consecutive distinct scores; t=1.5 admits only neg 2.5
        # (F1 = 6/7), t=2.75 keeps two positives and no negatives (F1 = 4/5); t=1.5 wins
        assert calibrate_bias(pos, neg, "max_f1") == pytest.approx(-1.5)

    def test_max_f1_perfect_separation_at_gap_midpoint(self):
        bias = calibrate_bias(torch.tensor([2.0, 3.0]), torch.tensor([0.0, 1.0]), "max_f1")
        assert bias == pytest.approx(-1.5)  # threshold (1 + 2) / 2
        assert torch.tensor([2.0]) + bias > 0
        assert torch.tensor([1.0]) + bias < 0

    def test_max_f1_ties_break_on_margin(self):
        pos = torch.tensor([1.0, 6.0])
        neg = torch.tensor([0.0, 2.0, 3.0])
        # t=0.5 and t=4.5 tie at F1 = 2/3; the symmetric margin min(pos.min() - t, t - neg.max())
        # is -2.5 at t=0.5 and -3.5 at t=4.5
        assert calibrate_bias(pos, neg, "max_f1") == pytest.approx(-0.5)

    def test_max_f1_single_distinct_score(self):
        bias = calibrate_bias(torch.tensor([1.0]), torch.tensor([1.0]), "max_f1")
        assert bias == pytest.approx(-1.0)

    def test_target_fpr_exact_quantile(self):
        neg = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
        bias = calibrate_bias(torch.tensor([10.0]), neg, ("target_fpr", 0.25))
        assert bias == pytest.approx(-float(torch.quantile(neg, 0.75)))  # -3.0
        assert bias == pytest.approx(-3.0)

    def test_inverted_scores_raise(self):
        with pytest.raises(ValueError, match="scores are inverted; negate your direction"):
            calibrate_bias(torch.tensor([0.0, 1.0]), torch.tensor([2.0, 3.0]))

    def test_empty_scores_raise(self):
        with pytest.raises(ValueError, match="at least one score"):
            calibrate_bias(torch.tensor([]), torch.tensor([1.0]))

    def test_malformed_spec_raises(self):
        with pytest.raises(ValueError, match="calibration"):
            calibrate_bias(torch.tensor([1.0]), torch.tensor([0.0]), ("target_fpr", 1.5))


class TestAnisotropySynthetic:
    """Residual-stream geometry: raw mean-difference scoring degenerates, diagonal LDA does not.

    Features follow `h = c + y*s*d + eps`: a large shared component (8 on the first basis
    vector), a rogue high-variance coordinate (scale 5 on the second), a small signed class term
    (`y` in {-1, +1}, `s = 0.4` on the third), and small orthogonal noise. An unrelated cluster
    has `y = 0`.
    """

    H = 6
    N = 16
    S = 0.4

    def _clusters(self):
        g = torch.Generator().manual_seed(42)

        def cluster(y: float) -> torch.Tensor:
            h = torch.zeros(self.N, self.H)
            h[:, 0] = 8.0
            h[:, 1] = 5.0 * torch.randn(self.N, generator=g)  # rogue coordinate
            h[:, 2] = y * self.S
            h[:, 3:] = 0.05 * torch.randn(self.N, self.H - 3, generator=g)
            return h

        return cluster(+1.0), cluster(-1.0), cluster(0.0)

    def _ambient_stats(self, samples: torch.Tensor) -> ActivationStats:
        return ActivationStats(
            model_type="synthetic",
            model_fingerprint="synthetic",
            location="layer_input",
            count=samples.size(0),
            mean={0: samples.mean(dim=0)},
            var={0: samples.var(dim=0).clamp_min(1e-6)},
        )

    def test_raw_cosine_degenerates_and_lda_probe_separates(self):
        pos, neg, unrelated = self._clusters()
        stats = self._ambient_stats(torch.cat([pos, neg, unrelated]))

        # raw cosine against the mean difference: the rogue coordinate's sampling noise leaks
        # into the direction and dominates the class term, so the classes overlap
        delta = _fit_direction(pos, neg, 0, ProbeFitSpec(method="mean_diff"), None)
        pos_cos = F.cosine_similarity(pos, delta.unsqueeze(0), dim=-1)
        neg_cos = F.cosine_similarity(neg, delta.unsqueeze(0), dim=-1)
        assert pos_cos.min() < neg_cos.max()

        # the diagonal-LDA direction crushes the rogue coordinate; calibrated at max_f1 with the
        # unrelated cluster among the covering negatives, the probe is canonically polarized and
        # the unrelated cluster falls on the closed side
        w = _fit_direction(pos, neg, 0, ProbeFitSpec(method="lda"), stats)
        bias = calibrate_bias(pos @ w, torch.cat([neg, unrelated]) @ w, "max_f1")
        probe = Probe(
            model_type="synthetic", location="layer_input", pooling="mean",
            layer_ids=[0], weights={0: w}, bias=bias,
        )
        assert probe.predict({0: pos}).all()
        assert not probe.predict({0: neg}).any()
        assert (probe.decision_function({0: unrelated}) < 0).all()


class TestMethodRequirements:
    def test_lda_without_stats_raises(self, model, tokenizer):
        with pytest.raises(ValueError, match="estimate ActivationStats once per model"):
            fit_probe(model, tokenizer, data=DATA, spec=ProbeFitSpec(method="lda"))

    def test_logreg_without_stats_raises(self, model, tokenizer):
        with pytest.raises(ValueError, match="estimate ActivationStats once per model"):
            fit_probe(model, tokenizer, data=DATA, spec=ProbeFitSpec(method="logreg"))

    def test_mean_diff_ignores_stats_for_direction(self, model, tokenizer, stats):
        spec = ProbeFitSpec(method="mean_diff", candidate_layers=[1])
        without = fit_probe(model, tokenizer, data=DATA, spec=spec)
        with_stats = fit_probe(model, tokenizer, data=DATA, spec=spec, stats=stats)
        assert torch.equal(without.weights[1], with_stats.weights[1])
        assert with_stats.meta["stats_used"] is False
        assert "stats_fingerprint" not in with_stats.meta

    def test_wrong_model_stats_raise_and_escape_proceeds(self, model, tokenizer):
        torch.manual_seed(1)
        other = tiny_llama(num_layers=LAYERS, hidden=HIDDEN, heads=4)
        other_stats = ActivationStats.estimate(
            other, tokenizer, STATS_TEXTS, location="layer_input", min_samples=1
        )
        spec = ProbeFitSpec(method="lda", candidate_layers=[1])
        with pytest.raises(ValueError, match="different model"):
            fit_probe(model, tokenizer, data=DATA, spec=spec, stats=other_stats)
        probe = fit_probe(
            model, tokenizer, data=DATA, spec=spec, stats=other_stats, allow_model_mismatch=True
        )
        assert probe.layer_ids == [1]

    def test_wrong_location_stats_raise(self, model, tokenizer):
        output_stats = ActivationStats.estimate(
            model, tokenizer, STATS_TEXTS, location="layer_output", min_samples=1
        )
        spec = ProbeFitSpec(method="lda", candidate_layers=[1])
        with pytest.raises(ValueError, match="location 'layer_output'.*'layer_input'"):
            fit_probe(model, tokenizer, data=DATA, spec=spec, stats=output_stats)

    def test_stats_layer_subset_raises_before_the_sweep(self, model, tokenizer):
        subset_stats = ActivationStats.estimate(
            model, tokenizer, STATS_TEXTS, layer_ids=[1], location="layer_input", min_samples=1
        )
        with pytest.raises(ValueError, match=r"candidate layer\(s\) \[0, 2, 3\]"):
            fit_probe(model, tokenizer, data=DATA, spec=ProbeFitSpec(method="lda"), stats=subset_stats)
        probe = fit_probe(
            model, tokenizer, data=DATA,
            spec=ProbeFitSpec(method="lda", candidate_layers=[1]), stats=subset_stats,
        )
        assert probe.layer_ids == [1]


class TestOrientation:
    def test_mean_diff_and_lda_never_flip_on_fit_set(self, model, tokenizer, stats):
        swapped = ContrastivePairs(positives=DATA.negatives, negatives=DATA.positives)
        for method in ("mean_diff", "lda"):
            spec = ProbeFitSpec(method=method, candidate_layers=[1])
            kwargs = {"stats": stats} if method == "lda" else {}
            for data in (DATA, swapped):
                probe = fit_probe(model, tokenizer, data=data, spec=spec, **kwargs)
                assert probe.meta["orientation_flipped"] is False

    def test_label_flipped_logreg_triggers_orientation(self, model, tokenizer, stats, monkeypatch):
        class _InvertedLogReg:
            """Stand-in whose coefficients point from the positive class to the negative."""

            def __init__(self, **kwargs):
                pass

            def fit(self, X, y):
                X = torch.as_tensor(X, dtype=torch.float32)
                y = torch.as_tensor(y)
                delta = X[y == 1].mean(dim=0) - X[y == 0].mean(dim=0)
                self.coef_ = (-delta).unsqueeze(0).numpy()

        import steerability.algorithms.core.internals.probes.fitting as fitting_module
        monkeypatch.setattr(fitting_module, "LogisticRegression", _InvertedLogReg)

        spec = ProbeFitSpec(method="logreg", candidate_layers=[1])
        probe = fit_probe(model, tokenizer, data=DATA, spec=spec, stats=stats)
        assert probe.meta["orientation_flipped"] is True
        # the flip restored canonical polarity, so calibration succeeded and the fitted weights
        # equal the (un-negated) standardized mean difference folded to raw coordinates
        expected = _fit_direction(*_features_for_layer(model, tokenizer, DATA, spec, 1),
                                  1, ProbeFitSpec(method="lda"), stats)
        assert torch.allclose(probe.weights[1], expected, atol=1e-5)

    def test_orientation_not_evaluated_on_shifted_calibration_data(self, model, tokenizer):
        swapped = ContrastivePairs(positives=DATA.negatives, negatives=DATA.positives)
        spec = ProbeFitSpec(method="mean_diff", candidate_layers=[1])
        with pytest.raises(ValueError, match="scores are inverted"):
            fit_probe(model, tokenizer, data=DATA, spec=spec, calibration_data=swapped)


def _features_for_layer(model, tokenizer, data, spec, layer_id):
    """Pooled positive/negative features at one layer, via the fitting module's own path."""
    from steerability.algorithms.core.internals.probes.fitting import _pooled_features

    pos, neg = _pooled_features(model, tokenizer, data, spec, [layer_id])
    return pos[layer_id], neg[layer_id]


class TestLayerSelection:
    def test_candidate_sweep_recorded_and_best_kept(self, model, tokenizer):
        spec = ProbeFitSpec(method="mean_diff", candidate_layers=[1, 2])
        probe = fit_probe(model, tokenizer, data=DATA, spec=spec)

        sweep = probe.meta["layer_sweep"]
        assert [row["layer_id"] for row in sweep] == [1, 2]
        assert probe.layer_ids[0] in (1, 2)
        best = max(sweep, key=lambda row: (row["f1"], row["margin"]))
        assert probe.layer_ids == [best["layer_id"]]
        assert probe.meta["layer_id"] == best["layer_id"]

    def test_default_sweeps_all_layers(self, model, tokenizer):
        probe = fit_probe(model, tokenizer, data=DATA, spec=ProbeFitSpec(method="mean_diff"))
        assert [row["layer_id"] for row in probe.meta["layer_sweep"]] == list(range(LAYERS))

    def test_layer_range_windows_the_sweep(self, model, tokenizer):
        spec = ProbeFitSpec(method="mean_diff", layer_range=(0.0, 0.5))
        probe = fit_probe(model, tokenizer, data=DATA, spec=spec)
        assert [row["layer_id"] for row in probe.meta["layer_sweep"]] == [0, 1]

    def test_out_of_range_candidate_raises(self, model, tokenizer):
        spec = ProbeFitSpec(method="mean_diff", candidate_layers=[99])
        with pytest.raises(ValueError, match="out of range"):
            fit_probe(model, tokenizer, data=DATA, spec=spec)


class TestPromptFormatting:
    def test_raw_and_chat_prompt_produce_different_encodings(self, model, monkeypatch):
        chat_tokenizer = wordlevel_tokenizer()
        chat_tokenizer.chat_template = (
            "{{ bos_token }}"
            "{% for message in messages %}"
            "[{{ message['role'] | upper }}] {{ message['content'] }}"
            "{% endfor %}"
            "{% if add_generation_prompt %}[ASSISTANT] {% endif %}"
        )

        import steerability.algorithms.core.internals.probes.fitting as fitting_module
        recorded: list[tuple[tuple[str, ...], bool]] = []
        original = fitting_module.tokenize_texts

        def spy(tokenizer, texts, device=None, *, add_special_tokens=True, max_length=None):
            recorded.append((tuple(texts), add_special_tokens))
            return original(
                tokenizer, texts, device, add_special_tokens=add_special_tokens, max_length=max_length
            )

        monkeypatch.setattr(fitting_module, "tokenize_texts", spy)

        spec_raw = ProbeFitSpec(method="mean_diff", candidate_layers=[1], prompt_format="raw")
        fit_probe(model, chat_tokenizer, data=DATA, spec=spec_raw)
        raw_calls = list(recorded)
        recorded.clear()

        spec_chat = ProbeFitSpec(method="mean_diff", candidate_layers=[1], prompt_format="chat_prompt")
        fit_probe(model, chat_tokenizer, data=DATA, spec=spec_chat)
        chat_calls = list(recorded)

        assert raw_calls[0][0] != chat_calls[0][0]  # rendered texts differ
        assert raw_calls[0][1] is True  # raw text takes the tokenizer's special tokens
        assert chat_calls[0][1] is False  # templated text already contains them


class TestMetaRecord:
    def test_meta_provenance_fields(self, model, tokenizer, stats):
        spec = ProbeFitSpec(method="lda", candidate_layers=[1], calibration=("target_fpr", 0.25))
        probe = fit_probe(model, tokenizer, data=DATA, spec=spec, stats=stats)

        meta = probe.meta
        assert meta["method"] == "lda"
        assert meta["pooling"] == "last"
        assert meta["location"] == "layer_input"
        assert meta["n_pos"] == 4 and meta["n_neg"] == 4
        assert meta["calibration"]["kind"] == ["target_fpr", 0.25]
        assert meta["calibration"]["on"] == "data"
        assert 0.0 <= meta["calibration"]["f1"] <= 1.0
        assert 0.0 <= meta["calibration"]["fpr"] <= 1.0
        assert meta["stats_used"] is True
        assert meta["stats_fingerprint"] == stats.fingerprint()
        assert meta["model_fingerprint"] == model_fingerprint(model)
        assert meta["polarity"] == "positives_high"
        assert "package_version" in meta

    def test_calibration_split_recorded(self, model, tokenizer):
        spec = ProbeFitSpec(method="mean_diff", candidate_layers=[1])
        probe = fit_probe(model, tokenizer, data=DATA, spec=spec, calibration_data=DATA)
        assert probe.meta["calibration"]["on"] == "calibration_data"

    def test_weights_raw_coordinates_float32(self, model, tokenizer, stats):
        spec = ProbeFitSpec(method="lda", candidate_layers=[2])
        probe = fit_probe(model, tokenizer, data=DATA, spec=spec, stats=stats)
        assert probe.weights[2].dtype == torch.float32
        assert probe.weights[2].shape == (HIDDEN,)
        assert isinstance(probe.bias, float)


class TestFisher:
    def test_matches_closed_form_pooled_covariance_discriminant(self):
        # full-rank case: the direction equals the pooled-covariance solve of the class-mean
        # difference, unit-normalized (reference computed without the SVD path)
        g = torch.Generator().manual_seed(7)
        pos = torch.randn(12, 8, generator=g) + torch.tensor([1.0] + [0.0] * 7)
        neg = torch.randn(10, 8, generator=g)
        w = _fit_direction(pos, neg, 0, ProbeFitSpec(method="fisher"), None)

        cov = (torch.cov(pos.T) * (pos.size(0) - 1) + torch.cov(neg.T) * (neg.size(0) - 1)) / (
            pos.size(0) + neg.size(0) - 2
        )
        expected = torch.linalg.solve(cov, pos.mean(dim=0) - neg.mean(dim=0))
        expected = expected / torch.linalg.norm(expected)
        assert torch.allclose(w, expected, atol=1e-4)
        assert torch.linalg.norm(w).item() == pytest.approx(1.0, abs=1e-5)

    def test_rank_deficient_uses_truncated_pseudo_inverse(self):
        # fewer samples than dimensions: the direction stays finite and equals the truncated
        # pseudo-inverse applied to the class-mean difference
        g = torch.Generator().manual_seed(3)
        pos = torch.randn(3, 16, generator=g) + 0.5
        neg = torch.randn(3, 16, generator=g)
        w = _fit_direction(pos, neg, 0, ProbeFitSpec(method="fisher"), None)
        assert torch.isfinite(w).all()

        cov = (torch.cov(pos.T) * (pos.size(0) - 1) + torch.cov(neg.T) * (neg.size(0) - 1)) / (
            pos.size(0) + neg.size(0) - 2
        )
        expected = torch.linalg.pinv(cov, atol=1e-6) @ (pos.mean(dim=0) - neg.mean(dim=0))
        expected = expected / torch.linalg.norm(expected)
        assert torch.allclose(w, expected, atol=1e-4)

    def test_fit_probe_without_stats(self, model, tokenizer):
        spec = ProbeFitSpec(method="fisher", candidate_layers=[1])
        probe = fit_probe(model, tokenizer, data=DATA, spec=spec)
        assert probe.layer_ids == [1]
        assert probe.meta["stats_used"] is False
        assert probe.meta["orientation_flipped"] is False
        assert torch.linalg.norm(probe.weights[1]).item() == pytest.approx(1.0, abs=1e-4)


class TestUnpairedData:
    def test_labeled_examples_with_unequal_classes(self, model, tokenizer):
        data = LabeledExamples(
            positives=["the cat sat", "the dog ran", "the cat ran on"],
            negatives=["mat on fast", "span attention"],
        )
        spec = ProbeFitSpec(
            method="fisher", pooling="last", location="layer_output",
            prompt_format="raw", candidate_layers=[LAYERS - 1], calibration="midpoint",
        )
        probe = fit_probe(model, tokenizer, data=data, spec=spec)
        assert probe.layer_ids == [LAYERS - 1]
        assert probe.meta["n_pos"] == 3 and probe.meta["n_neg"] == 2

    def test_chat_completion_with_unpaired_data_raises(self, model, tokenizer):
        data = LabeledExamples(positives=["the cat sat"], negatives=["mat on fast", "dog ran"])
        spec = ProbeFitSpec(
            method="mean_diff", candidate_layers=[1], prompt_format="chat_completion"
        )
        with pytest.raises(ValueError, match="unpaired"):
            fit_probe(model, tokenizer, data=data, spec=spec)


class TestChunkedExtraction:
    def test_fitted_probe_is_chunking_invariant(self, model, tokenizer):
        # per-chunk tokenization pads independently; mask-aware pooling makes the fitted probe
        # independent of the chunking
        spec = ProbeFitSpec(method="fisher", candidate_layers=[1], calibration="midpoint")
        one_chunk = fit_probe(model, tokenizer, data=DATA, spec=spec, batch_size=8)
        split = fit_probe(model, tokenizer, data=DATA, spec=spec, batch_size=1)
        assert torch.allclose(one_chunk.weights[1], split.weights[1], atol=1e-4)
        assert one_chunk.bias == pytest.approx(split.bias, abs=1e-4)


class TestSpecValidation:
    def test_bad_method_raises(self):
        with pytest.raises(ValueError, match="method"):
            ProbeFitSpec(method="pca")

    def test_bad_layer_range_raises(self):
        with pytest.raises(ValueError, match="layer_range"):
            ProbeFitSpec(layer_range=(0.5, 0.2))

    def test_bad_calibration_raises(self):
        with pytest.raises(ValueError, match="calibration"):
            ProbeFitSpec(calibration="best_effort")

    def test_nonpositive_c_raises(self):
        with pytest.raises(ValueError, match="C must be positive"):
            ProbeFitSpec(C=0.0)
