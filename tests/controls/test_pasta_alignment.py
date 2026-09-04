"""PASTA token-coordinate, beam-order, arch, and side-effect tests (Issues 1, 4, 6, 7).

Runs hub-free on a tiny randomly-initialized Llama with a WordLevel tokenizer whose re-encoding
is id-faithful to the real sequence (so offsets land directly in real coordinates).
"""
import copy

import pytest
import torch

from aisteer360.algorithms.core.steering_pipeline import SteeringPipeline
from aisteer360.algorithms.state_control.pasta.control import PASTA
from tests.utils.tiny_models import tiny_llama, wordlevel_tokenizer


def _pasta_pipeline(model, tokenizer, **pasta_kwargs):
    pasta = PASTA(**pasta_kwargs)
    pipeline = SteeringPipeline(controls=[pasta], model=model, tokenizer=tokenizer)
    pipeline.steer()
    return pipeline, pasta


def _capture_token_ranges(pasta, input_ids, substrings, input_len_out):
    """Run get_hooks and read back the token_ranges/input_len baked into the pre-hook partial."""
    hooks = pasta.get_hooks(input_ids, {"substrings": substrings})
    partial_fn = hooks["pre"][0]["hook_func"]
    kw = partial_fn.keywords
    input_len_out["value"] = kw["input_len"]
    return kw["token_ranges"]


class TestTokenCoordinates:
    def test_bos_prompt_range_and_input_len(self):
        """`"cat sat"` in `"the cat sat on mat"` yields range (2, 4) and input_len == 6."""
        model = tiny_llama()
        tokenizer = wordlevel_tokenizer()
        _, pasta = _pasta_pipeline(model, tokenizer, head_config=[0], alpha=2.0, scale_position="include")

        input_ids = tokenizer("the cat sat on mat", return_tensors="pt").input_ids
        assert input_ids[0].tolist() == [0, 3, 4, 5, 6, 7]  # <s> the cat sat on mat

        out = {}
        token_ranges = _capture_token_ranges(pasta, input_ids, ["cat sat"], out)

        assert token_ranges[0].tolist() == [[2, 4]]
        assert out["value"] == 6

    def test_multi_special_prefix_shifts_range(self):
        """A second prepended special token shifts the range right by one."""
        model = tiny_llama()
        tokenizer = wordlevel_tokenizer(single="<s> <s> $A")  # two BOS-like specials
        _, pasta = _pasta_pipeline(model, tokenizer, head_config=[0], alpha=2.0, scale_position="include")

        input_ids = tokenizer("the cat sat on mat", return_tensors="pt").input_ids
        # two specials prepended -> cat/sat move from tokens 2/3 to tokens 3/4
        assert input_ids[0].tolist()[:2] == [0, 0]

        out = {}
        token_ranges = _capture_token_ranges(pasta, input_ids, ["cat sat"], out)

        assert token_ranges[0].tolist() == [[3, 5]]
        assert out["value"] == input_ids.size(1)


class TestRealMaskColumns:
    def test_scaled_columns_are_the_real_span(self):
        """In the mask PASTA emits, the real span columns sit exactly log(alpha) above the rest.

        In "include" mode PASTA adds log(alpha) to the span columns and subtracts log(alpha) from
        all columns in [0, input_len); the net effect on a query's attended (causal) row is that
        span columns are log(alpha) higher than every other real key column — and the boosted
        columns are exactly the real span positions (2, 3), verifying coordinates land on the real
        sequence (which includes BOS), not a special-token-stripped re-encoding.
        """
        model = tiny_llama()
        tokenizer = wordlevel_tokenizer()
        alpha = 3.0
        _, pasta = _pasta_pipeline(model, tokenizer, head_config=[0], alpha=alpha, scale_position="include")

        input_ids = tokenizer("the cat sat on mat", return_tensors="pt").input_ids  # span (2, 4)
        seq_len = input_ids.size(1)
        assert seq_len == 6

        # capture the attention_mask PASTA's pre-hook returns (it builds a causal mask when the
        # module would otherwise receive None, then applies the scaling to it)
        attn_module = model.get_submodule(pasta._attn_module_names[0])
        captured = {}
        hooks = pasta.get_hooks(input_ids, {"substrings": ["cat sat"]})
        pasta_hook = hooks["pre"][0]["hook_func"]
        h1 = attn_module.register_forward_pre_hook(pasta_hook, with_kwargs=True)
        h2 = attn_module.register_forward_pre_hook(
            lambda m, a, k: captured.__setitem__("mask", k.get("attention_mask")), with_kwargs=True
        )
        try:
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))
        finally:
            h2.remove()
            h1.remove()

        mask = captured["mask"]
        assert mask is not None and mask.dim() == 4  # [B, H, q, k]

        log_alpha = torch.tensor(alpha).log().to(mask.dtype)
        # last query row attends causally to all 6 keys; span columns (2,3) exceed non-span real
        # columns (0,1,4,5) by exactly log(alpha)
        row = mask[0, 0, -1]  # [k]
        span_cols = [2, 3]
        non_span_cols = [0, 1, 4, 5]
        for c in span_cols:
            for nc in non_span_cols:
                torch.testing.assert_close(row[c] - row[nc], log_alpha, rtol=1e-4, atol=1e-4)
        # and the boosted set is exactly the span (nothing else is higher)
        max_val = row[: seq_len].max()
        boosted = [c for c in range(seq_len) if torch.isclose(row[c], max_val, atol=1e-4)]
        assert boosted == span_cols, f"expected span {span_cols} boosted, got {boosted}"


class TestBeamOrderIndexing:
    def test_interleave_aware_range_broadcast(self):
        """With expand=2 the ranges broadcast [r0, r0, r1, r1], matching repeat_interleave order."""
        pasta = PASTA.__new__(PASTA)
        from types import SimpleNamespace

        pasta.model = SimpleNamespace(config=SimpleNamespace(num_attention_heads=2))
        pasta.scale_position = "include"
        pasta._scale_constant = torch.tensor([2.0]).log()

        # two prompts with distinct ranges; beam expands each by 2
        r0 = torch.tensor([[1, 2]])
        r1 = torch.tensor([[3, 4]])
        token_ranges = [r0, r1]

        batch_size, num_heads, seq_len = 4, 2, 6  # 2 prompts * 2 beams
        attention_mask = torch.zeros(batch_size, num_heads, seq_len, seq_len)
        hidden = torch.zeros(batch_size, seq_len, 4)

        _, out_kwargs = pasta._attention_pre_hook(
            module=None,
            input_args=(hidden,),
            input_kwargs={"attention_mask": attention_mask.clone()},
            head_idx=[0, 1],
            token_ranges=token_ranges,
            input_len=seq_len,
        )
        result = out_kwargs["attention_mask"]

        # rows 0,1 must be identical (both use r0); rows 2,3 identical (both use r1); groups differ
        assert torch.equal(result[0], result[1])
        assert torch.equal(result[2], result[3])
        assert not torch.equal(result[0], result[2])

    def test_non_multiple_raises(self):
        from types import SimpleNamespace

        pasta = PASTA.__new__(PASTA)
        pasta.model = SimpleNamespace(config=SimpleNamespace(num_attention_heads=2))
        pasta.scale_position = "include"
        pasta._scale_constant = torch.tensor([2.0]).log()

        token_ranges = [torch.tensor([[1, 2]]), torch.tensor([[3, 4]])]
        batch_size = 3  # not a multiple of 2
        attention_mask = torch.zeros(batch_size, 2, 5, 5)
        hidden = torch.zeros(batch_size, 5, 4)

        with pytest.raises(RuntimeError, match="not a multiple"):
            pasta._attention_pre_hook(
                module=None,
                input_args=(hidden,),
                input_kwargs={"attention_mask": attention_mask},
                head_idx=[0, 1],
                token_ranges=token_ranges,
                input_len=5,
            )


class TestArchAndFailFast:
    def test_gpt2_style_resolves(self):
        """PASTA.steer() resolves modules on a GPT-2-style config and get_hooks finds them."""
        from transformers import GPT2Config, GPT2LMHeadModel

        cfg = GPT2Config(n_embd=32, n_layer=3, n_head=4, vocab_size=100, n_positions=64)
        model = GPT2LMHeadModel(cfg).eval()
        tokenizer = wordlevel_tokenizer()

        _, pasta = _pasta_pipeline(model, tokenizer, head_config=[0, 1], alpha=2.0)
        assert pasta._attn_module_names == {0: "transformer.h.0.attn", 1: "transformer.h.1.attn"}
        # every resolved module exists
        for path in pasta._attn_module_names.values():
            model.get_submodule(path)

    def test_flash_attention_fails_fast_at_steer(self):
        """A model reporting flash_attention_2 raises an informative error at steer(), not at gen."""
        model = tiny_llama()
        model.config._attn_implementation = "flash_attention_2"
        tokenizer = wordlevel_tokenizer()

        pasta = PASTA(head_config=[0], alpha=2.0)
        pipeline = SteeringPipeline(controls=[pasta], model=model, tokenizer=tokenizer)
        with pytest.raises(ValueError, match="eager"):
            pipeline.steer()


class TestSideEffects:
    def test_padding_side_unchanged(self):
        model = tiny_llama()
        tokenizer = wordlevel_tokenizer()
        tokenizer.padding_side = "right"
        _, pasta = _pasta_pipeline(model, tokenizer, head_config=[0], alpha=2.0, scale_position="include")

        input_ids = tokenizer("the cat sat on mat", return_tensors="pt").input_ids
        before = tokenizer.padding_side
        pasta.get_hooks(input_ids, {"substrings": ["cat sat"]})
        assert tokenizer.padding_side == before == "right"

    def test_runtime_kwargs_not_mutated(self):
        model = tiny_llama()
        tokenizer = wordlevel_tokenizer()
        _, pasta = _pasta_pipeline(model, tokenizer, head_config=[0], alpha=2.0, scale_position="include")

        input_ids = tokenizer("the cat sat on mat", return_tensors="pt").input_ids
        runtime_kwargs = {"substrings": [["cat sat"]]}
        snapshot = copy.deepcopy(runtime_kwargs)
        pasta.get_hooks(input_ids, runtime_kwargs)
        assert runtime_kwargs == snapshot
