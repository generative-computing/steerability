"""Tests for ambient activation statistics (`core/internals/stats.py`).

Streaming accumulation is pinned to a direct two-pass computation over the same features, and the
`exclude_first_n` and `count` semantics are verified against hand-derived token positions on a
left-padded batch.
"""
import warnings

import pytest
import torch

from aisteer360.algorithms.core.internals.capture import layerwise_tokenwise_hidden
from aisteer360.algorithms.core.internals.encoding import tokenize_texts
from aisteer360.algorithms.core.internals.fingerprint import model_fingerprint
from aisteer360.algorithms.core.internals.pooling import get_last_token_positions, masked_mean, select_at_positions
from aisteer360.algorithms.core.internals.stats import ActivationStats, StatsSpec
from tests.utils.tiny_models import tiny_llama, wordlevel_tokenizer

TEXTS = [
    "the cat sat on mat",
    "dog ran fast",
    "attention span on the mat",
    "the dog sat",
    "mat cat dog ran fast attention",
]


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return tiny_llama()


@pytest.fixture(scope="module")
def tokenizer():
    return wordlevel_tokenizer()


def _collect_features(model, tokenizer, texts, batch_size, pooling, exclude_first_n, location="layer_input"):
    """Replicate the estimate loop but materialize the full per-layer feature matrices."""
    features: dict[int, list[torch.Tensor]] = {}
    for start in range(0, len(texts), batch_size):
        chunk = list(texts[start:start + batch_size])
        enc = tokenize_texts(tokenizer, chunk)
        hidden = layerwise_tokenwise_hidden(model, enc, batch_size=len(chunk), location=location)
        mask = enc["attention_mask"].bool()
        for lid, states in hidden.items():
            states = states.to(torch.float32)
            if pooling == "tokens":
                rank = mask.long().cumsum(dim=1)
                feats = states[mask & (rank > exclude_first_n)]
            elif pooling == "mean":
                feats = masked_mean(states, mask.to(states.dtype))
            else:
                positions = get_last_token_positions(mask.long(), states.size(1), states.size(0))
                feats = select_at_positions(states, positions)
            features.setdefault(lid, []).append(feats)
    return {lid: torch.cat(chunks, dim=0) for lid, chunks in features.items()}


class TestStreamingAccumulation:
    def test_streaming_equals_two_pass_mean_and_var(self, model, tokenizer):
        stats = ActivationStats.estimate(
            model, tokenizer, TEXTS, batch_size=2, min_samples=1
        )
        reference = _collect_features(model, tokenizer, TEXTS, 2, "tokens", 1)
        for lid, feats in reference.items():
            torch.testing.assert_close(stats.mean[lid], feats.mean(dim=0), rtol=1e-4, atol=1e-6)
            torch.testing.assert_close(
                stats.var[lid], feats.var(dim=0).clamp_min(stats.var_floor), rtol=1e-3, atol=1e-6
            )
        assert stats.count == next(iter(reference.values())).size(0)

    def test_layer_subset_and_out_of_range(self, model, tokenizer):
        stats = ActivationStats.estimate(model, tokenizer, TEXTS, layer_ids=[1, 2], min_samples=1)
        assert sorted(stats.mean) == [1, 2]
        with pytest.raises(ValueError, match="out of range"):
            ActivationStats.estimate(model, tokenizer, TEXTS, layer_ids=[99], min_samples=1)

    def test_duplicate_layer_ids_are_collapsed(self, model, tokenizer):
        deduped = ActivationStats.estimate(model, tokenizer, TEXTS, layer_ids=[1, 1], min_samples=1)
        single = ActivationStats.estimate(model, tokenizer, TEXTS, layer_ids=[1], min_samples=1)
        assert sorted(deduped.mean) == [1]
        assert torch.equal(deduped.mean[1], single.mean[1])
        assert torch.equal(deduped.var[1], single.var[1])

    def test_all_pad_rows_contribute_no_sample(self, model):
        # a template without BOS lets the empty string tokenize to zero real tokens
        tokenizer_no_bos = wordlevel_tokenizer(single="$A")
        texts = ["the cat sat on mat", "", "dog ran fast"]
        for pooling in ("mean", "last"):
            stats = ActivationStats.estimate(
                model, tokenizer_no_bos, texts, pooling=pooling, min_samples=1
            )
            clean = ActivationStats.estimate(
                model, tokenizer_no_bos, [texts[0], texts[2]], pooling=pooling, min_samples=1
            )
            assert stats.count == 2
            for lid in stats.mean:
                torch.testing.assert_close(stats.mean[lid], clean.mean[lid], rtol=1e-4, atol=1e-6)

    def test_provenance_recorded(self, model, tokenizer):
        stats = ActivationStats.estimate(model, tokenizer, TEXTS, min_samples=1)
        assert stats.model_fingerprint == model_fingerprint(model)
        assert stats.model_type == model.config.model_type
        assert stats.location == "layer_input"


class TestExcludeFirstN:
    def test_drops_exactly_first_n_real_positions_left_padded(self, model, tokenizer):
        tokenizer_left = wordlevel_tokenizer()
        tokenizer_left.padding_side = "left"
        n = 2
        stats = ActivationStats.estimate(
            model, tokenizer_left, TEXTS, exclude_first_n=n, batch_size=len(TEXTS), min_samples=1
        )

        enc = tokenize_texts(tokenizer_left, TEXTS)
        hidden = layerwise_tokenwise_hidden(model, enc, batch_size=len(TEXTS), location="layer_input")
        mask = enc["attention_mask"]
        expected_rows = []
        for i in range(mask.size(0)):
            real_positions = [t for t in range(mask.size(1)) if mask[i, t] == 1]
            for t in real_positions[n:]:
                expected_rows.append(hidden[0][i, t].to(torch.float32))
        expected = torch.stack(expected_rows)

        assert stats.count == expected.size(0)
        torch.testing.assert_close(stats.mean[0], expected.mean(dim=0), rtol=1e-4, atol=1e-6)

    def test_inert_for_mean_and_last(self, model, tokenizer):
        for pooling in ("mean", "last"):
            a = ActivationStats.estimate(
                model, tokenizer, TEXTS, pooling=pooling, exclude_first_n=0, min_samples=1
            )
            b = ActivationStats.estimate(
                model, tokenizer, TEXTS, pooling=pooling, exclude_first_n=3, min_samples=1
            )
            for lid in a.mean:
                assert torch.equal(a.mean[lid], b.mean[lid])
                assert torch.equal(a.var[lid], b.var[lid])


class TestCountSemantics:
    def test_tokens_counts_kept_positions_and_prompt_poolings_count_prompts(self, model, tokenizer):
        enc = tokenize_texts(tokenizer, TEXTS)
        real = int(enc["attention_mask"].sum().item())
        tokens_stats = ActivationStats.estimate(model, tokenizer, TEXTS, min_samples=1)
        assert tokens_stats.count == real - len(TEXTS)  # default exclude_first_n=1
        for pooling in ("mean", "last"):
            stats = ActivationStats.estimate(model, tokenizer, TEXTS, pooling=pooling, min_samples=1)
            assert stats.count == len(TEXTS)

    def test_min_samples_warning_counts_pooled_samples_not_texts(self, model, tokenizer):
        with pytest.warns(UserWarning, match="pooled samples"):
            ActivationStats.estimate(model, tokenizer, TEXTS, min_samples=10_000)

        # more tokens than min_samples but fewer texts: counting texts would warn, tokens must not
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            ActivationStats.estimate(model, tokenizer, TEXTS, min_samples=10)
        assert not [w for w in record if "pooled samples" in str(w.message)]


class TestVarianceFloor:
    def test_single_sample_variance_is_floored(self, model, tokenizer):
        stats = ActivationStats.estimate(
            model, tokenizer, TEXTS[:1], pooling="mean", min_samples=1
        )
        assert stats.count == 1
        for var in stats.var.values():
            assert torch.equal(var, torch.full_like(var, stats.var_floor))


class TestSerialization:
    def test_save_load_roundtrip_is_exact(self, model, tokenizer, tmp_path):
        stats = ActivationStats.estimate(model, tokenizer, TEXTS, min_samples=1)
        stats.save(tmp_path / "stats")
        loaded = ActivationStats.load(tmp_path / "stats")

        assert loaded.model_type == stats.model_type
        assert loaded.model_fingerprint == stats.model_fingerprint
        assert loaded.location == stats.location
        assert loaded.count == stats.count
        assert loaded.var_floor == stats.var_floor
        assert sorted(loaded.mean) == sorted(stats.mean)
        for lid in stats.mean:
            assert torch.equal(loaded.mean[lid], stats.mean[lid])
            assert torch.equal(loaded.var[lid], stats.var[lid])
        assert loaded.fingerprint() == stats.fingerprint()

    def test_artifact_fingerprint_tracks_contents(self, model, tokenizer):
        a = ActivationStats.estimate(model, tokenizer, TEXTS, min_samples=1)
        b = ActivationStats.estimate(model, tokenizer, TEXTS[:3], min_samples=1)
        assert a.fingerprint() != b.fingerprint()


class TestStandardize:
    def test_shape_and_dtype_contract(self, model, tokenizer):
        stats = ActivationStats.estimate(model, tokenizer, TEXTS, min_samples=1)
        lid = min(stats.mean)
        H = stats.mean[lid].numel()
        features = torch.randn(3, H, dtype=torch.float64)
        out = stats.standardize(features, lid)
        assert out.shape == features.shape
        assert out.dtype == torch.float32
        expected = (features.to(torch.float32) - stats.mean[lid]) / stats.var[lid].sqrt()
        torch.testing.assert_close(out, expected)

    def test_missing_layer_raises(self, model, tokenizer):
        stats = ActivationStats.estimate(model, tokenizer, TEXTS, layer_ids=[0], min_samples=1)
        with pytest.raises(KeyError, match="no statistics for layer"):
            stats.standardize(torch.randn(2, stats.mean[0].numel()), 3)


class TestStatsSpec:
    def test_estimate_delegates_with_recipe_fields(self, model, tokenizer):
        spec = StatsSpec(texts=TEXTS, layer_ids=[0, 1], pooling="mean", batch_size=2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            from_spec = spec.estimate(model, tokenizer)
            direct = ActivationStats.estimate(
                model, tokenizer, TEXTS, layer_ids=[0, 1], pooling="mean", batch_size=2
            )
        assert sorted(from_spec.mean) == [0, 1]
        assert from_spec.count == direct.count
        for lid in from_spec.mean:
            assert torch.equal(from_spec.mean[lid], direct.mean[lid])
            assert torch.equal(from_spec.var[lid], direct.var[lid])


class TestValidation:
    def test_empty_texts_raises(self, model, tokenizer):
        with pytest.raises(ValueError, match="at least one text"):
            ActivationStats.estimate(model, tokenizer, [])

    def test_bad_pooling_raises(self, model, tokenizer):
        with pytest.raises(ValueError, match="pooling must be"):
            ActivationStats.estimate(model, tokenizer, TEXTS, pooling="max")  # type: ignore[arg-type]

    def test_negative_exclude_raises(self, model, tokenizer):
        with pytest.raises(ValueError, match="exclude_first_n"):
            ActivationStats.estimate(model, tokenizer, TEXTS, exclude_first_n=-1)
