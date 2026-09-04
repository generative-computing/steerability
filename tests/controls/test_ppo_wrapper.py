"""Model-free tests for the PPO TRL wrapper's tokenizer/vocab guard.

`PPOTrainer` scores the policy's own token ids with a single shared tokenizer, so the reward and value
models must share the policy's vocabulary. These tests concern this guard.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from aisteer360.algorithms.structural_control.wrappers.trl.ppotrainer.base_mixin import PPOTrainerMixin


class _TokenizerStub:
    """Minimal stand-in: only `len()` is consulted by the guard."""

    def __init__(self, vocab_size: int) -> None:
        self._vocab_size = vocab_size

    def __len__(self) -> int:
        return self._vocab_size


def _model_stub(vocab_size: int | None) -> SimpleNamespace:
    return SimpleNamespace(config=SimpleNamespace(vocab_size=vocab_size))


def _make_mixin(policy_vocab: int, reward_path: str = "reward/path", value_path: str | None = None):
    mixin = PPOTrainerMixin.__new__(PPOTrainerMixin)
    mixin.tokenizer = _TokenizerStub(policy_vocab)
    mixin.reward_model_name_or_path = reward_path
    mixin.value_model_name_or_path = value_path
    return mixin


class TestCheckScoringVocab:
    def test_reward_vocab_smaller_raises(self):
        mixin = _make_mixin(policy_vocab=128256)
        with pytest.raises(ValueError, match=r"reward model .* vocab_size 128100, smaller"):
            mixin._check_scoring_vocab(
                reward_model=_model_stub(128100),
                value_model=_model_stub(128256),
            )

    def test_value_vocab_smaller_raises(self):
        mixin = _make_mixin(policy_vocab=128256, value_path="value/path")
        with pytest.raises(ValueError, match=r"value model 'value/path' .* smaller"):
            mixin._check_scoring_vocab(
                reward_model=_model_stub(128256),
                value_model=_model_stub(128100),
            )

    def test_matched_vocab_passes(self):
        mixin = _make_mixin(policy_vocab=128256)
        # no exception; reward/value cover the policy vocab exactly
        mixin._check_scoring_vocab(
            reward_model=_model_stub(128256),
            value_model=_model_stub(128256),
        )

    def test_larger_scoring_vocab_passes(self):
        """A scoring model with a strictly larger vocab still covers every policy id."""
        mixin = _make_mixin(policy_vocab=32000)
        mixin._check_scoring_vocab(
            reward_model=_model_stub(50000),
            value_model=_model_stub(50000),
        )

    def test_missing_vocab_size_is_skipped(self):
        """A scoring model whose config lacks `vocab_size` is not flagged (nothing to compare)."""
        mixin = _make_mixin(policy_vocab=128256)
        mixin._check_scoring_vocab(
            reward_model=_model_stub(None),
            value_model=_model_stub(None),
        )
