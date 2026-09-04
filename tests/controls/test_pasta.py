import logging
from types import SimpleNamespace

import pytest
import torch

from aisteer360.algorithms.core.steering_pipeline import SteeringPipeline
from aisteer360.algorithms.state_control.pasta.control import PASTA
from tests.utils.sweep import build_param_grid

PROMPT_TEXT = (
    "Answer truthfully. Therefore, when you respond: "
    "First, present your main point. "
    "Second, support it with evidence. "
    "Finally, conclude succinctly."
)

PASTA_GRID = {
    "substrings": [
        ["Therefore"],
        ["First,", "Second,", "Finally,"]
    ],
    "alpha": [0.25, 0.75],
    "scale_position": ["include", "exclude", "generation"],
    "head_config": [
        [0],
        [0, 1]
    ],
}


@pytest.mark.parametrize("conf", build_param_grid(PASTA_GRID))
def test_pasta(model_and_tokenizer, device: torch.device, conf: dict):
    """
    Verify that PASTA steers and generates on every model/device/param combo.
    """

    # move model to target device
    base_model, tokenizer = model_and_tokenizer
    model = base_model.to(device)

    # build pipeline with PASTA control
    pasta = PASTA(
        head_config=conf["head_config"],
        alpha=conf["alpha"],
        scale_position=conf["scale_position"]
    )
    pipeline = SteeringPipeline(controls=[pasta], model=model, tokenizer=tokenizer)
    pipeline.steer()

    # prepare prompt & runtime kwargs
    prompt_ids = tokenizer(PROMPT_TEXT, return_tensors="pt").input_ids.to(device)
    runtime_kwargs = {"substrings": conf["substrings"]}

    # generate
    out_ids = pipeline.generate(
        input_ids=prompt_ids,
        runtime_kwargs=runtime_kwargs,
        max_new_tokens=8,
    )

    # assertions
    assert isinstance(out_ids, torch.Tensor), "Output is not torch.Tensor"
    assert out_ids.ndim == 2, "Expected (batch, seq_len) tensor"
    assert out_ids.size(1) >= 1, "No new tokens generated"


class TestFindTokenRangeMissingSubstring:
    def test_absent_substring_returns_sentinel_and_warns(self, caplog):
        with caplog.at_level(logging.WARNING):
            start, end = PASTA._find_token_range("hello world", "absent", [(0, 5), (5, 11)])
        assert (start, end) == (0, 0)
        assert any("not found" in record.message for record in caplog.records)
        # the full input text is not dumped into the log record
        assert not any("hello world" in record.getMessage() for record in caplog.records)


class TestAttentionPreHookMissingRangeNoOp:
    """Include-mode: an item whose only range is the (0, 0) sentinel must be left untouched, while a
    sibling item with a valid range is steered as before."""

    def _make_pasta(self, num_heads: int = 2, alpha: float = 2.0) -> PASTA:
        pasta = PASTA.__new__(PASTA)
        pasta.model = SimpleNamespace(config=SimpleNamespace(num_attention_heads=num_heads))
        pasta.scale_position = "include"
        pasta._scale_constant = torch.tensor([alpha]).log()
        return pasta

    def test_empty_range_item_unchanged(self):
        pasta = self._make_pasta(num_heads=2)
        head_idx = [0, 1]
        batch_size, num_heads, seq_len = 2, 2, 6
        input_len = seq_len

        # item 0 has a valid range [1, 3]; item 1 has only the (0, 0) sentinel
        token_ranges = [
            torch.tensor([[1, 3]]),
            torch.tensor([[0, 0]]),
        ]

        attention_mask = torch.zeros(batch_size, num_heads, seq_len, seq_len)
        input_kwargs = {"attention_mask": attention_mask.clone()}
        hidden_states = torch.zeros(batch_size, seq_len, 4)

        original = attention_mask.clone()
        _, out_kwargs = pasta._attention_pre_hook(
            module=None,
            input_args=(hidden_states,),
            input_kwargs=input_kwargs,
            head_idx=head_idx,
            token_ranges=token_ranges,
            input_len=input_len,
        )
        result = out_kwargs["attention_mask"]

        # item 1 (empty range only) is a true no-op
        assert torch.equal(result[1], original[1])
        # item 0 is modified relative to the untouched baseline
        assert not torch.equal(result[0], original[0])
