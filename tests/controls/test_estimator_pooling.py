"""Pad-position pooling invariance tests for direction estimators (Issue 5).

With variable-length contrastive pairs, activations at pad positions are garbage and the amount
of padding differs per pair. Pooling must be mask-driven so poisoning pad positions cannot bias
the learned direction, regardless of padding side.
"""
import pytest
import torch

from aisteer360.algorithms.core.internals.pooling import pool_over_spans as _pool_over_spans
from aisteer360.algorithms.core.internals.pooling import select_spans as _select_spans
from aisteer360.algorithms.state_control.common.estimators.mean_difference import _masked_mean


def _poison_pads(hidden: torch.Tensor, attention_mask: torch.Tensor, value: float = 1e6) -> torch.Tensor:
    """Set every pad position's hidden state to a huge constant."""
    poisoned = hidden.clone()
    pad = attention_mask == 0
    poisoned[pad] = value
    return poisoned


class TestMaskedMean:
    def test_invariant_to_pad_poison_both_sides(self):
        for side in ("left", "right"):
            torch.manual_seed(0)
            N, T, H = 3, 6, 8
            hidden = torch.randn(N, T, H)
            # each row has a different number of real tokens
            real_lens = [2, 4, 5]
            attention_mask = torch.zeros(N, T, dtype=torch.long)
            for i, n in enumerate(real_lens):
                if side == "left":
                    attention_mask[i, T - n:] = 1
                else:
                    attention_mask[i, :n] = 1

            clean = _masked_mean(hidden, attention_mask)
            poisoned = _masked_mean(_poison_pads(hidden, attention_mask), attention_mask)
            torch.testing.assert_close(clean, poisoned, msg=f"padding_side={side}")

    def test_matches_plain_mean_when_no_pads(self):
        hidden = torch.randn(2, 5, 4)
        full_mask = torch.ones(2, 5, dtype=torch.long)
        torch.testing.assert_close(_masked_mean(hidden, full_mask), hidden.mean(dim=1))

    def test_none_mask_is_plain_mean(self):
        hidden = torch.randn(2, 5, 4)
        torch.testing.assert_close(_masked_mean(hidden, None), hidden.mean(dim=1))


class TestSelectSpansPooling:
    def _build(self, side, real_lens, T):
        N = len(real_lens)
        input_ids = torch.zeros(N, T, dtype=torch.long)
        attention_mask = torch.zeros(N, T, dtype=torch.long)
        for i, n in enumerate(real_lens):
            if side == "left":
                attention_mask[i, T - n:] = 1
                input_ids[i, T - n:] = torch.arange(1, n + 1)
            else:
                attention_mask[i, :n] = 1
                input_ids[i, :n] = torch.arange(1, n + 1)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def test_pool_invariant_to_pad_poison_both_sides(self):
        for side in ("left", "right"):
            torch.manual_seed(1)
            N, T, H = 3, 7, 8
            real_lens = [3, 5, 6]
            enc = self._build(side, real_lens, T)
            hidden = torch.randn(N, T, H)

            spans = _select_spans(enc, prompt_enc=None, accumulate="all")
            clean = _pool_over_spans(hidden, spans)
            poisoned = _pool_over_spans(_poison_pads(hidden, enc["attention_mask"]), spans)
            torch.testing.assert_close(clean, poisoned, msg=f"padding_side={side}")

    def test_span_excludes_pads_both_sides(self):
        for side in ("left", "right"):
            real_lens = [3, 5]
            T = 7
            enc = self._build(side, real_lens, T)
            spans = _select_spans(enc, prompt_enc=None, accumulate="all")
            for i, n in enumerate(real_lens):
                start, end = spans[i]
                # the span covers exactly the n real positions
                assert end - start == n, f"side={side} item={i}: span {start}:{end} != {n} real tokens"
                # and every position inside it is a real (non-pad) token
                assert enc["attention_mask"][i, start:end].all()

    def test_suffix_only_selects_completion_tokens(self):
        # shared prompt of 2 tokens; completions of length 2 and 3; right-padded
        real_lens = [4, 5]
        T = 6
        enc = self._build("right", real_lens, T)
        prompt_enc = {
            "input_ids": torch.zeros(2, 2, dtype=torch.long),
            "attention_mask": torch.ones(2, 2, dtype=torch.long),
        }
        spans = _select_spans(enc, prompt_enc=prompt_enc, accumulate="suffix-only")
        # start skips the 2 prompt tokens, end is the last real token + 1
        assert spans[0] == (2, 4)
        assert spans[1] == (2, 5)

    def test_last_token_selects_final_real_token(self):
        # ragged right-padded batch, real lengths 5, 3, 7
        am = torch.tensor([[1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1]])
        enc = {"input_ids": torch.zeros(3, 7, dtype=torch.long), "attention_mask": am}
        spans = _select_spans(enc, prompt_enc=None, accumulate="last_token")
        assert spans == [(4, 5), (2, 3), (6, 7)]

    def test_last_token_differs_from_all(self):
        # regression guard: last_token must not silently fall through to the full-span "all" behavior
        am = torch.tensor([[1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1]])
        enc = {"input_ids": torch.zeros(3, 7, dtype=torch.long), "attention_mask": am}
        assert _select_spans(enc, None, "last_token") != _select_spans(enc, None, "all")

    def test_last_token_pooling_equals_final_real_token(self):
        # every position distinct so a mean over the span differs from picking the final token
        N, T, H = 3, 7, 4
        real_lens = [5, 3, 7]
        am = torch.zeros(N, T, dtype=torch.long)
        for i, n in enumerate(real_lens):
            am[i, :n] = 1
        enc = {"input_ids": torch.zeros(N, T, dtype=torch.long), "attention_mask": am}
        hidden = torch.arange(N * T * H, dtype=torch.float).reshape(N, T, H)

        spans_last = _select_spans(enc, None, "last_token")
        pooled_last = _pool_over_spans(hidden, spans_last)
        expected = torch.stack([hidden[i, n - 1, :] for i, n in enumerate(real_lens)], dim=0)
        torch.testing.assert_close(pooled_last, expected)

        spans_all = _select_spans(enc, None, "all")
        pooled_mean = _pool_over_spans(hidden, spans_all)
        assert not torch.allclose(pooled_last, pooled_mean)

    def test_last_token_left_padding(self):
        # left-padded row: the final real index is the last position, proving pad-side agnosticism
        am = torch.tensor([[0, 0, 1, 1, 1]])
        enc = {"input_ids": torch.zeros(1, 5, dtype=torch.long), "attention_mask": am}
        spans = _select_spans(enc, prompt_enc=None, accumulate="last_token")
        assert spans == [(4, 5)]

    def test_unknown_accumulate_raises(self):
        enc = {"input_ids": torch.zeros(1, 3, dtype=torch.long), "attention_mask": torch.ones(1, 3, dtype=torch.long)}
        with pytest.raises(ValueError):
            _select_spans(enc, prompt_enc=None, accumulate="bogus")
