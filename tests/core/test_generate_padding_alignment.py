"""Pipeline-seam padding alignment on the batched generate path.

`SteeringPipeline._execute_generation` left-packs the steered prompt tensors after the input
chain, matching the layout the Hugging Face session executes every batched forward in. These
tests drive ragged (unequal-length) batches through the full pipeline-plus-session path, the
only place a right-padded build-time layout and the left-packed forward-time layout can
diverge.

Runs hub-free across the tiny CI models and the `device` fixture; forces `padding_side="right"`
so the divergence appears regardless of a model's tokenizer default.
"""
import torch

from steerability.algorithms.core.internals.model_layout import text_config
from steerability.algorithms.core.steering_pipeline import SteeringPipeline
from steerability.algorithms.state_control.cast.control import CAST
from steerability.algorithms.state_control.common.steering_vector import SteeringVector
from steerability.utils.tokenization import to_left_pad


def _cast_control(hidden: int, condition_layer: int) -> CAST:
    """A conditional CAST with a precomputed condition vector and a manual condition point.

    The behavior direction is tiny so generation stays close to the base model; the condition
    layer scores the prompt and gates the (broadcast) behavior addition per row.
    """
    torch.manual_seed(0)
    condition_vector = SteeringVector(
        model_type="test", directions={condition_layer: torch.randn(1, hidden)},
    )
    behavior_vector = SteeringVector(
        model_type="test", directions={0: 0.01 * torch.randn(1, hidden)},
    )
    return CAST(
        behavior_vector=behavior_vector,
        behavior_layer_ids=[0],
        condition_vector=condition_vector,
        condition_point={
            "layer_ids": [condition_layer],
            "threshold": 0.0,
            "comparator": "ge",
            "comparison_mode": "mean",
        },
    )


def test_batched_gate_scores_match_single(model_and_tokenizer, device):
    """Per-row condition scores from a ragged batch match the single-prompt scores.

    The condition hook pools the prefill hidden states over the stored prompt mask. When the
    build-time layout (right-padded) and the forward-time layout (left-packed) disagree, the
    pooled mean for every row shorter than the batch maximum covers pad positions, so the
    batched scores drift from the single-prompt scores for the same prompts.
    """
    base_model, tokenizer = model_and_tokenizer
    model = base_model.to(device)
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    try:
        cfg = text_config(model)
        condition_layer = cfg.num_hidden_layers - 1
        control = _cast_control(cfg.hidden_size, condition_layer)
        pipeline = SteeringPipeline(controls=[control], model=model, tokenizer=tokenizer)
        pipeline.steer()

        prompts = ["a b c d e f g h", "a b c", "a b c d e"]  # ragged lengths
        gen = {"max_new_tokens": 2, "do_sample": False}

        single_scores = []
        for prompt in prompts:
            pipeline.generate(text=prompt, **gen)
            single_scores.append(float(control.latest_decision.scores[condition_layer]))

        pipeline.generate(text=prompts, **gen)
        batched_scores = [
            float(score) for score in control.latest_decision.scores_per_row[condition_layer]
        ]

        torch.testing.assert_close(
            torch.tensor(batched_scores), torch.tensor(single_scores), atol=2e-3, rtol=1e-3,
        )
    finally:
        tokenizer.padding_side = original_padding_side


def test_full_sequence_return_has_no_interior_pads(model_and_tokenizer, device):
    """`return_full_sequence=True` returns the left-packed prompt, so short rows carry no interior pads.

    The prompt slice of the returned ids equals `to_left_pad(input_ids, attention_mask)` row for
    row, the layout the continuation was generated from (`[pads, prompt, continuation]`).
    """
    base_model, tokenizer = model_and_tokenizer
    model = base_model.to(device)
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    try:
        control = _cast_control(text_config(model).hidden_size, text_config(model).num_hidden_layers - 1)
        pipeline = SteeringPipeline(controls=[control], model=model, tokenizer=tokenizer)
        pipeline.steer()

        encoded = tokenizer(["a b c d e f g h", "a b c", "a b c d e"], return_tensors="pt", padding=True)
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        prompt_len = input_ids.size(1)

        returned = pipeline.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=2,
            do_sample=False,
            return_full_sequence=True,
        )

        expected_prompt, _ = to_left_pad(input_ids, attention_mask)
        assert torch.equal(returned[:, :prompt_len].cpu(), expected_prompt.cpu())
    finally:
        tokenizer.padding_side = original_padding_side
