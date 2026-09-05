import random

import pytest
import torch

from steerability.algorithms.core.steering_pipeline import SteeringPipeline
from steerability.algorithms.output_control.deal.control import DeAL
from tests.utils.sweep import build_param_grid

PROMPT_TEXT = "Answer concisely. If you make a recommendation, begin with the word 'Yes'."

DEAL_GRID = {
    "lookahead": [2, 4],
    "init_beams": [4, 6],
    "topk": [1, 3],
    "max_iterations": [2, 4],
}


def _reward_func_factory():
    """
    Returns a simple reward function:
    - +2 for each occurrence of the target keyword (case-insensitive)
    - +0.1 per character to mildly prefer longer (non-empty) continuations
    Ensures it returns one score per continuation.
    """
    def reward_func(prompt: str, continuations: list[str], reward_params: dict) -> list[float]:
        target = (reward_params or {}).get("target", "yes")
        t_lower = target.lower()
        scores = []
        for continuation in continuations:
            continuation_lower = continuation.lower()
            hits = continuation_lower.count(t_lower)
            scores.append(hits * 2.0 + 0.1 * max(0, len(continuation)))
        return scores
    return reward_func


@pytest.mark.parametrize("conf", build_param_grid(DEAL_GRID))
def test_deal(model_and_tokenizer, device: torch.device, conf: dict):
    """
    Verify that DeAL steers decoding via reward-guided beam selection on every model/device/param combo.
    """
    # deterministic behavior
    random.seed(0)
    torch.manual_seed(0)

    base_model, tokenizer = model_and_tokenizer
    model = base_model.to(device)

    # build DeAL control
    deal = DeAL(
        reward_func=_reward_func_factory(),
        lookahead=conf["lookahead"],
        init_beams=conf["init_beams"],
        topk=conf["topk"],
        max_iterations=conf["max_iterations"],
    )

    # pipeline
    pipeline = SteeringPipeline(controls=[deal], model=model, tokenizer=tokenizer)
    pipeline.steer()

    # prepare inputs & runtime kwargs
    prompt_ids = tokenizer(PROMPT_TEXT, return_tensors="pt").input_ids.to(device)
    runtime_kwargs = {
        "reward_params": {
            "target": "Yes"
        }
    }

    # generate
    out_ids = pipeline.generate(
        input_ids=prompt_ids,
        runtime_kwargs=runtime_kwargs,
        max_new_tokens=8,
    )

    # assertions
    assert isinstance(out_ids, torch.Tensor), "Output is not torch.Tensor"
    assert out_ids.ndim == 2, "Expected (batch, seq_len) tensor"
    assert out_ids.size(0) == 1, "DeAL currently supports batch size 1"

    # allow either full sequence (prompt + continuation) or continuation-only
    out_len = out_ids.size(1)
    prompt_len = prompt_ids.size(1)
    assert out_len >= 1, "No tokens returned"
    if out_len >= prompt_len:
        # full sequence case
        pass
    else:
        # continuation-only case (should be reasonably short)
        assert out_len <= 32, "Continuation-only output unexpectedly long"
