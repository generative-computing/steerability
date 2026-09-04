"""Tests for the shared condition-scoring helpers (pooling and projected-cosine score math)."""
import torch

from aisteer360.algorithms.core.internals.pooling import aggregate_condition_hidden, masked_mean
from aisteer360.algorithms.state_control.common.gating import (
    projected_cosine_similarity,
    projected_cosine_similarity_tensor,
    rank_one_projector,
)


def _poison_pads(hidden: torch.Tensor, attention_mask: torch.Tensor, value: float = 1e6) -> torch.Tensor:
    poisoned = hidden.clone()
    poisoned[attention_mask == 0] = value
    return poisoned


def _build_masked(side: str, real_lens: list[int], T: int, H: int):
    N = len(real_lens)
    hidden = torch.randn(N, T, H)
    mask = torch.zeros(N, T, dtype=torch.long)
    for i, n in enumerate(real_lens):
        if side == "left":
            mask[i, T - n:] = 1
        else:
            mask[i, :n] = 1
    return hidden, mask


class TestProjectedCosineSimilarity:
    def test_batched_matches_scalar(self):
        torch.manual_seed(0)
        H = 6
        projector = rank_one_projector(torch.randn(H))
        hidden = torch.randn(4, H)
        batched = projected_cosine_similarity_tensor(hidden, projector)
        for i in range(hidden.size(0)):
            scalar = projected_cosine_similarity(hidden[i], projector)
            assert abs(float(batched[i]) - scalar) < 1e-6

    def test_all_scores_finite(self):
        torch.manual_seed(1)
        projector = rank_one_projector(torch.randn(5))
        scores = projected_cosine_similarity_tensor(torch.randn(3, 5), projector)
        assert torch.isfinite(scores).all()


class TestRankOneProjector:
    def test_matches_outer_product(self):
        c = torch.tensor([0.3, -0.7, 1.1])
        expected = torch.outer(c, c) / (c @ c + 1e-8)
        torch.testing.assert_close(rank_one_projector(c), expected)

    def test_symmetric(self):
        p = rank_one_projector(torch.randn(7))
        torch.testing.assert_close(p, p.T)

    def test_sign_invariant(self):
        c = torch.randn(5)
        torch.testing.assert_close(rank_one_projector(c), rank_one_projector(-c))


class TestAggregateConditionHidden:
    def test_mean_invariant_to_pad_poison_both_sides(self):
        for side in ("left", "right"):
            torch.manual_seed(2)
            hidden, mask = _build_masked(side, [2, 4, 5], T=6, H=8)
            clean = aggregate_condition_hidden(hidden, "mean", mask)
            poisoned = aggregate_condition_hidden(_poison_pads(hidden, mask), "mean", mask)
            torch.testing.assert_close(clean, poisoned, msg=f"side={side}")

    def test_mean_none_mask_is_plain_mean(self):
        hidden = torch.randn(2, 5, 4)
        torch.testing.assert_close(aggregate_condition_hidden(hidden, "mean", None), hidden.mean(dim=1))

    def test_last_selects_final_real_token_both_sides(self):
        T = 7
        for side in ("left", "right"):
            torch.manual_seed(3)
            real_lens = [3, 5]
            hidden, mask = _build_masked(side, real_lens, T=T, H=4)
            out = aggregate_condition_hidden(hidden, "last", mask)
            for i, n in enumerate(real_lens):
                last_idx = T - 1 if side == "left" else n - 1
                torch.testing.assert_close(out[i], hidden[i, last_idx], msg=f"side={side} row={i}")

    def test_last_none_mask_is_final_position(self):
        hidden = torch.randn(2, 5, 4)
        torch.testing.assert_close(aggregate_condition_hidden(hidden, "last", None), hidden[:, -1, :])

    def test_all_padding_row_raises(self):
        hidden = torch.randn(1, 4, 3)
        mask = torch.zeros(1, 4, dtype=torch.long)
        try:
            aggregate_condition_hidden(hidden, "last", mask)
            assert False, "expected ValueError"
        except ValueError:
            pass

    def test_unsupported_mode_raises(self):
        try:
            aggregate_condition_hidden(torch.randn(1, 2, 3), "median", None)  # type: ignore[arg-type]
            assert False, "expected ValueError"
        except ValueError:
            pass


class TestMaskedMeanReimport:
    def test_reimport_path_resolves_and_matches(self):
        from aisteer360.algorithms.state_control.common.estimators.mean_difference import _masked_mean
        hidden = torch.randn(2, 5, 4)
        mask = torch.ones(2, 5, dtype=torch.long)
        torch.testing.assert_close(_masked_mean(hidden, mask), masked_mean(hidden, mask))
