"""Tests for the prefix-reward training helper (`rad/utils/reward_training.py`).

Covers the cumulative prefix loss against a hand-computed value, `prefix_rewards` shape and range on
the tiny Granite and tiny Llama checkpoints, and a one-epoch CPU training run whose output directory
reloads through `load_sequence_classifier` and backs RAD's cached path.

Hub-free: classifiers are built from config classes saved to `tmp_path` and share the wordlevel
tokenizer, so the reward model and the language model share a vocabulary (`tests/utils/tiny_models.py`).
"""
import pytest
import torch
from transformers import GraniteConfig, GraniteForCausalLM, LlamaConfig, LlamaForSequenceClassification

from steerability.algorithms.core.steering_pipeline import SteeringPipeline
from steerability.algorithms.output_control.common.loading import load_sequence_classifier
from steerability.algorithms.output_control.rad.control import RAD
from steerability.algorithms.output_control.rad.utils.reward_training import (
    PrefixRewardTrainSpec,
    prefix_reward_loss,
    prefix_rewards,
    train_prefix_reward_model,
)
from tests.utils.tiny_models import tiny_llama, wordlevel_tokenizer

VOCAB = 100


def _granite_backbone_path(tmp_path, vocab=VOCAB):
    """A hub-free `GraniteForCausalLM` backbone sharing the wordlevel vocabulary."""
    cfg = GraniteConfig(
        hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=2, vocab_size=vocab, pad_token_id=2,
    )
    GraniteForCausalLM(cfg).eval().save_pretrained(str(tmp_path))
    wordlevel_tokenizer().save_pretrained(str(tmp_path))
    return str(tmp_path)


def _llama_scalar_classifier(vocab=VOCAB):
    """A `LlamaForSequenceClassification` with a single-logit head."""
    cfg = LlamaConfig(
        hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=2, vocab_size=vocab,
        num_labels=1, pad_token_id=2,
    )
    return LlamaForSequenceClassification(cfg).eval()


def test_prefix_reward_loss_hand_computed():
    """The cumulative prefix loss matches a hand computation on a right-padded two-row batch."""
    predictions = torch.tensor([[0.1, 0.2, 0.3], [0.5, 0.6, 0.9]])
    labels = torch.tensor([1.0, 0.0])
    attention_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])

    row0 = (1 * (0.1 - 1) ** 2 + 2 * (0.2 - 1) ** 2 + 3 * (0.3 - 1) ** 2) / 6  # S_l = 3*4/2
    row1 = (1 * (0.5 - 0) ** 2 + 2 * (0.6 - 0) ** 2) / 3  # S_l = 2*3/2, padded position dropped
    expected = (row0 + row1) / 2

    got = prefix_reward_loss(predictions, labels, attention_mask)
    assert got.item() == pytest.approx(expected, abs=1e-6)


def test_prefix_reward_loss_ignores_fully_masked_rows():
    """A fully masked row contributes nothing to the mean."""
    predictions = torch.tensor([[0.2, 0.4], [0.9, 0.9]])
    labels = torch.tensor([0.0, 1.0])
    attention_mask = torch.tensor([[1, 1], [0, 0]])
    only_row0 = (1 * 0.2 ** 2 + 2 * 0.4 ** 2) / 3
    assert prefix_reward_loss(predictions, labels, attention_mask).item() == pytest.approx(only_row0, abs=1e-6)


def test_prefix_reward_loss_computes_in_fp32_from_bf16_inputs():
    """bf16 inputs over a sequence longer than bf16's exact-integer range give the fp32 loss, in fp32."""
    torch.manual_seed(0)
    length = 300
    predictions = torch.rand(2, length).to(torch.bfloat16)
    labels = torch.tensor([0.7, 0.2]).to(torch.bfloat16)
    attention_mask = torch.ones(2, length, dtype=torch.long)
    reference = prefix_reward_loss(predictions.float(), labels.float(), attention_mask)
    got = prefix_reward_loss(predictions, labels, attention_mask)
    assert got.dtype == torch.float32
    torch.testing.assert_close(got, reference)


class TestPrefixRewards:
    """`prefix_rewards` returns `[B, T]` sigmoid outputs on both checkpoint families."""

    def test_shape_and_range_on_llama(self):
        model = _llama_scalar_classifier()
        input_ids = torch.tensor([[0, 4, 5, 6], [0, 7, 8, 2]])
        attention_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]])
        rewards = prefix_rewards(model, input_ids, attention_mask)
        assert rewards.shape == (2, 4)
        assert torch.all(rewards >= 0) and torch.all(rewards <= 1)

    def test_shape_and_range_on_granite(self, tmp_path):
        model, _ = load_sequence_classifier(
            _granite_backbone_path(tmp_path), device="cpu", hf_model_kwargs={"num_labels": 1}
        )
        input_ids = torch.tensor([[0, 4, 5, 6], [0, 7, 8, 2]])
        attention_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]])
        rewards = prefix_rewards(model, input_ids, attention_mask)
        assert rewards.shape == (2, 4)
        assert torch.all(rewards >= 0) and torch.all(rewards <= 1)


def test_train_prefix_reward_model_round_trip(tmp_path):
    """Training on tiny Granite writes a checkpoint that reloads and backs RAD's cached path."""
    backbone = _granite_backbone_path(tmp_path / "backbone")
    texts = ["the cat sat", "the dog ran fast", "on the mat", "the cat ran",
             "the dog sat on the mat", "fast cat", "the mat", "dog on the mat"]
    labels = [0.9, 0.1, 0.2, 0.8, 0.15, 0.85, 0.3, 0.05]
    output_dir = tmp_path / "reward_model"

    spec = PrefixRewardTrainSpec(max_length=16, batch_size=4, epochs=1, learning_rate=1e-4, seed=0, log_every=1)
    result = train_prefix_reward_model(backbone, texts, labels, output_dir, spec=spec, device="cpu")
    assert result == output_dir

    reward_model, _ = load_sequence_classifier(str(output_dir), device="cpu")
    assert reward_model.config.num_labels == 1

    model = tiny_llama(num_layers=2, hidden=16, heads=2, vocab=VOCAB)
    rad = RAD(reward_model_id=str(output_dir), beta=5.0, top_k=8, score_transform="sigmoid", invert=True)
    pipeline = SteeringPipeline(controls=[rad], model=model, tokenizer=wordlevel_tokenizer())
    pipeline.steer()
    assert rad.scoring_path == "cached"

    prompt = wordlevel_tokenizer()("the cat", return_tensors="pt").input_ids
    out = pipeline.generate(input_ids=prompt, max_new_tokens=3, do_sample=False, eos_token_id=None)
    assert isinstance(out, torch.Tensor)
