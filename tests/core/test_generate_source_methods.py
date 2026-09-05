"""Parity tests for the per-source generate methods.

`generate_text`, `generate_messages`, and `generate_tokens` delegate to `generate()` with the
reserved `gen_kwargs` keys as named parameters; each must match the dispatching method's type
and shape for its source.
"""
import pytest
import torch

from steerability.algorithms.core.output import Output
from steerability.algorithms.core.steering_pipeline import SteeringPipeline

TINY_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"

GEN = {"max_new_tokens": 2, "do_sample": False}


@pytest.fixture(scope="module")
def pipeline():
    p = SteeringPipeline(model_name_or_path=TINY_MODEL)
    p.steer()
    return p


class TestGenerateTextParity:
    def test_single_matches_generate(self, pipeline):
        via_method = pipeline.generate_text("hello", **GEN)
        via_generate = pipeline.generate(text="hello", **GEN)
        assert isinstance(via_method, str)
        assert via_method == via_generate

    def test_batch_matches_generate(self, pipeline):
        via_method = pipeline.generate_text(["a", "b"], **GEN)
        via_generate = pipeline.generate(text=["a", "b"], **GEN)
        assert isinstance(via_method, list)
        assert len(via_method) == 2
        assert via_method == via_generate

    def test_return_output_returns_output(self, pipeline):
        out = pipeline.generate_text("hello", return_output=True, **GEN)
        assert isinstance(out, Output)

    def test_return_full_sequence_includes_prompt(self, pipeline):
        out = pipeline.generate_tokens(torch.tensor([[1, 2, 3]]), return_full_sequence=True, **GEN)
        assert out.shape[1] > 3


class TestGenerateMessagesParity:
    MESSAGES = [{"role": "user", "content": "hi"}]

    def test_single_matches_generate(self, pipeline):
        via_method = pipeline.generate_messages(self.MESSAGES, **GEN)
        via_generate = pipeline.generate(messages=self.MESSAGES, **GEN)
        assert isinstance(via_method, str)
        assert via_method == via_generate

    def test_batch_returns_list(self, pipeline):
        out = pipeline.generate_messages([self.MESSAGES, self.MESSAGES], **GEN)
        assert isinstance(out, list)
        assert len(out) == 2

    def test_chat_template_kwargs_named_parameter(self, pipeline):
        via_method = pipeline.generate_messages(
            self.MESSAGES, chat_template_kwargs={"enable_thinking": False}, **GEN
        )
        via_generate = pipeline.generate(
            messages=self.MESSAGES, chat_template_kwargs={"enable_thinking": False}, **GEN
        )
        assert via_method == via_generate

    def test_return_output_returns_output(self, pipeline):
        out = pipeline.generate_messages(self.MESSAGES, return_output=True, **GEN)
        assert isinstance(out, Output)


class TestGenerateTokensParity:
    def test_returns_tensor(self, pipeline):
        out = pipeline.generate_tokens(torch.tensor([[1, 2, 3]]), **GEN)
        assert isinstance(out, torch.Tensor)
        assert out.shape[0] == 1

    def test_matches_generate(self, pipeline):
        ids = torch.tensor([[1, 2, 3]])
        via_method = pipeline.generate_tokens(ids, **GEN)
        via_generate = pipeline.generate(input_ids=ids, **GEN)
        assert torch.equal(via_method, via_generate)

    def test_attention_mask_accepted(self, pipeline):
        ids = torch.tensor([[1, 2, 3]])
        mask = torch.ones_like(ids)
        out = pipeline.generate_tokens(ids, mask, **GEN)
        assert isinstance(out, torch.Tensor)

    def test_return_output_returns_output(self, pipeline):
        out = pipeline.generate_tokens(torch.tensor([1, 2, 3]), return_output=True, **GEN)
        assert isinstance(out, Output)

    def test_return_output_batch_returns_list(self, pipeline):
        out = pipeline.generate_tokens(torch.tensor([[1, 2, 3]]), return_output=True, **GEN)
        assert isinstance(out, list)
        assert all(isinstance(item, Output) for item in out)
