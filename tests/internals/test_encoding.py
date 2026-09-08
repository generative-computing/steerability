"""Tests for tokenization entry points (`core/internals/encoding.py`).

The parity class pins the merged `tokenize_texts` to the exact outputs of the single-purpose
tokenize-and-move helper it absorbed, at both of that helper's call-site configurations (an
explicit device move, and `add_special_tokens=False` for text that already carries its special
tokens), plus the prior default behavior when the new parameters are unset.
"""
import torch

from steerability.algorithms.core.internals.encoding import tokenize_pairs, tokenize_texts
from tests.utils.tiny_models import wordlevel_tokenizer

TEXTS = ["the cat sat on mat", "dog ran fast", "attention span"]


def _reference_tokenize(tokenizer, texts, device, *, add_special_tokens=True):
    """Tokenize a list of texts and move to device (the absorbed helper's exact behavior)."""
    enc = tokenizer(
        list(texts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        add_special_tokens=add_special_tokens,
    )
    return {k: v.to(device) for k, v in enc.items()}


def _assert_enc_equal(got: dict, expected: dict):
    assert set(got.keys()) == set(expected.keys())
    for key in expected:
        assert got[key].device == expected[key].device
        assert got[key].dtype == expected[key].dtype
        assert torch.equal(got[key], expected[key])


class TestTokenizeTextsMergeParity:
    def test_device_move_configuration(self):
        tokenizer = wordlevel_tokenizer()
        got = tokenize_texts(tokenizer, TEXTS, torch.device("cpu"))
        expected = _reference_tokenize(tokenizer, TEXTS, torch.device("cpu"))
        _assert_enc_equal(got, expected)

    def test_no_special_tokens_configuration(self):
        # texts standing in for chat-templated renders that already carry their special tokens
        tokenizer = wordlevel_tokenizer()
        got = tokenize_texts(tokenizer, TEXTS, "cpu", add_special_tokens=False)
        expected = _reference_tokenize(tokenizer, TEXTS, "cpu", add_special_tokens=False)
        _assert_enc_equal(got, expected)
        # the BOS-prepending template was actually bypassed
        bos_id = tokenizer.bos_token_id
        assert not (got["input_ids"][:, 0] == bos_id).all()

    def test_default_behavior_with_new_parameters_unset(self):
        tokenizer = wordlevel_tokenizer()
        got = tokenize_texts(tokenizer, TEXTS)
        expected = _reference_tokenize(tokenizer, TEXTS, "cpu")
        _assert_enc_equal(got, expected)


class TestTokenizeTexts:
    def test_independent_padding(self):
        tokenizer = wordlevel_tokenizer()
        enc = tokenize_texts(tokenizer, TEXTS)
        # padded to the longest text, mask marks real tokens only
        assert enc["input_ids"].shape == enc["attention_mask"].shape
        lengths = enc["attention_mask"].sum(dim=1)
        assert lengths[0] > lengths[2]

    def test_special_tokens_added_by_default(self):
        tokenizer = wordlevel_tokenizer()
        enc = tokenize_texts(tokenizer, TEXTS)
        assert (enc["input_ids"][:, 0] == tokenizer.bos_token_id).all()


class TestTokenizePairs:
    def test_pairs_share_padding_length(self):
        tokenizer = wordlevel_tokenizer()
        pos = ["the cat sat on mat", "dog ran"]
        neg = ["dog ran fast", "attention span on the mat"]
        enc_pos, enc_neg = tokenize_pairs(tokenizer, pos, neg, "cpu")
        assert enc_pos["input_ids"].shape == enc_neg["input_ids"].shape

    def test_deinterleave_matches_flat_tokenization(self):
        tokenizer = wordlevel_tokenizer()
        pos = ["the cat sat on mat", "dog ran"]
        neg = ["dog ran fast", "attention span on the mat"]
        enc_pos, enc_neg = tokenize_pairs(tokenizer, pos, neg, "cpu")

        interleaved = [pos[0], neg[0], pos[1], neg[1]]
        flat = tokenizer(interleaved, return_tensors="pt", padding=True, truncation=True)
        assert torch.equal(enc_pos["input_ids"], flat["input_ids"][0::2])
        assert torch.equal(enc_neg["input_ids"], flat["input_ids"][1::2])
        assert torch.equal(enc_pos["attention_mask"], flat["attention_mask"][0::2])
        assert torch.equal(enc_neg["attention_mask"], flat["attention_mask"][1::2])
