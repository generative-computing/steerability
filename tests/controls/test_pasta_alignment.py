"""PASTA token-coordinate, beam-order, arch, side-effect, and decode-phase mask tests.

Runs hub-free on a tiny randomly-initialized Llama with a WordLevel tokenizer whose re-encoding
is id-faithful to the real sequence (so offsets land directly in real coordinates).
"""
import copy

import pytest
import torch

from steerability.algorithms.core.steering_pipeline import SteeringPipeline
from steerability.algorithms.state_control.pasta.control import PASTA
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
            scale_constant=torch.tensor([2.0]).log(),
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
                scale_constant=torch.tensor([2.0]).log(),
            )


class TestArchAndFailFast:
    def test_gpt2_style_resolves(self):
        """PASTA resolves every attention layer of a GPT-2-style config and steers the configured ones."""
        from transformers import GPT2Config, GPT2LMHeadModel

        cfg = GPT2Config(n_embd=32, n_layer=3, n_head=4, vocab_size=100, n_positions=64)
        model = GPT2LMHeadModel(cfg).eval()
        tokenizer = wordlevel_tokenizer()

        _, pasta = _pasta_pipeline(model, tokenizer, head_config=[0, 1], alpha=2.0)
        assert pasta._attn_module_names == {i: f"transformer.h.{i}.attn" for i in range(3)}
        assert pasta.attention_layers() == [0, 1, 2]
        assert sorted(pasta._head_map) == [0, 1]
        # every resolved module exists
        for path in pasta._attn_module_names.values():
            model.get_submodule(path)

    def test_composite_wrapper_resolves_nested_attention(self):
        """PASTA resolves every text-decoder attention layer of a composite multimodal wrapper.

        The configured layers are the ones it steers.
        """
        from tests.utils.tiny_models import tiny_gemma3_conditional

        model = tiny_gemma3_conditional(num_layers=4, hidden=32, heads=4)
        tokenizer = wordlevel_tokenizer()
        _, pasta = _pasta_pipeline(model, tokenizer, head_config=[0, 1], alpha=2.0)
        assert pasta._attn_module_names == {i: f"model.language_model.layers.{i}.self_attn" for i in range(4)}
        assert pasta.attention_layers() == [0, 1, 2, 3]
        assert sorted(pasta._head_map) == [0, 1]
        for path in pasta._attn_module_names.values():
            model.get_submodule(path)

    def test_lora_wrapper_resolves_adapted_attention(self):
        """PASTA resolves every attention layer through an unmerged LoRA wrapper and generates.

        The configured layers are the ones it steers.
        """
        from tests.utils.tiny_models import tiny_lora

        model = tiny_lora(tiny_llama(num_layers=3, hidden=32, heads=4))
        tokenizer = wordlevel_tokenizer()
        pipeline, pasta = _pasta_pipeline(model, tokenizer, head_config=[0, 1], alpha=2.0)
        assert pasta._attn_module_names == {i: f"base_model.model.model.layers.{i}.self_attn" for i in range(3)}
        assert pasta.attention_layers() == [0, 1, 2]
        assert sorted(pasta._head_map) == [0, 1]
        for path in pasta._attn_module_names.values():
            model.get_submodule(path)
        input_ids = tokenizer("the cat sat on mat", return_tensors="pt").input_ids
        out = pipeline.generate(
            input_ids=input_ids, max_new_tokens=4, runtime_kwargs={"substrings": ["cat sat"]},
        )
        assert out.size(1) >= 1

    def test_flash_attention_fails_fast_at_steer(self):
        """A model reporting flash_attention_2 raises an informative error at steer(), not at gen."""
        model = tiny_llama()
        model.config._attn_implementation = "flash_attention_2"
        tokenizer = wordlevel_tokenizer()

        pasta = PASTA(head_config=[0], alpha=2.0)
        pipeline = SteeringPipeline(controls=[pasta], model=model, tokenizer=tokenizer)
        with pytest.raises(ValueError, match="eager"):
            pipeline.steer()


class TestHeadGeometryPerLayer:
    """PASTA sizes its per-layer head map and synthesized mask from each layer's own head count."""

    def test_num_heads_by_layer_matches_config_on_uniform_model(self):
        model = tiny_llama(num_layers=4, hidden=32, heads=4)
        tokenizer = wordlevel_tokenizer()
        _, pasta = _pasta_pipeline(model, tokenizer, head_config=[0, 1, 2, 3], alpha=2.0)
        assert pasta._num_heads_by_layer == {0: 4, 1: 4, 2: 4, 3: 4}

    def test_heterogeneous_num_heads_by_layer(self):
        """On a stub whose layers alternate head dim, the head count differs across layers."""
        from tests.utils.tiny_models import heterogeneous_head_stub

        stub = heterogeneous_head_stub(num_layers=4, hidden=32)  # head_dim 4/8 -> heads 8/4
        pasta = PASTA(head_config=[0, 1], alpha=2.0)
        pasta.steer(stub, wordlevel_tokenizer())
        assert pasta._num_heads_by_layer[0] == 8
        assert pasta._num_heads_by_layer[1] == 4

    def test_head_index_rejected_per_layer(self):
        """A head index valid on a wider layer but not a narrower one is rejected for that layer."""
        from tests.utils.tiny_models import heterogeneous_head_stub

        stub = heterogeneous_head_stub(num_layers=4, hidden=32)  # layer 0 has 8 heads, layer 1 has 4
        pasta = PASTA(head_config={0: [7], 1: [7]}, alpha=2.0)  # head 7 valid on layer 0, not layer 1
        with pytest.raises(ValueError, match="out of range for layer 1"):
            pasta.steer(stub, wordlevel_tokenizer())

    def test_synthesized_mask_head_axis_follows_layer(self):
        """`_attention_pre_hook` sizes the synthesized mask's head axis by the layer's head count."""
        from tests.utils.tiny_models import heterogeneous_head_stub

        stub = heterogeneous_head_stub(num_layers=4, hidden=32)
        pasta = PASTA(head_config=[0, 1], alpha=2.0, scale_position="include")
        pasta.steer(stub, wordlevel_tokenizer())

        token_ranges = [torch.tensor([[0, 1]])]
        for layer_idx, expected_heads in ((0, 8), (1, 4)):
            hidden_states = torch.zeros(1, 3, 32)
            _, out_kwargs = pasta._attention_pre_hook(
                module=None,
                input_args=(hidden_states,),
                input_kwargs={},
                head_idx=pasta._head_map[layer_idx],
                token_ranges=token_ranges,
                input_len=3,
                layer_idx=layer_idx,
                scale_constant=torch.tensor([pasta.alpha]).log(),
            )
            assert out_kwargs["attention_mask"].shape[1] == expected_heads


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


class TestDecodePhaseMask:
    """Regression tests for the decode-phase mask under transformers v5 (no `cache_position`).

    The v5 port regression built a broadcastable (b, h, 1, 1) bias on every decode step. SDPA's
    cuda mem-efficient backend expands broadcastable biases into a stride-0 last dimension and
    rejects them ("(*bias): last dimension must be contiguous"); the cpu math kernel broadcasts
    silently, turning every decode-step edit into a softmax-invariant no-op. These tests pin the
    key-axis width, last-dim contiguity, and the persistence of the log-alpha boost during
    generation, observed at layer 1 so the per-layer cache-length derivation is exercised.
    """

    def test_decode_mask_spans_cache_and_keeps_boost(self):
        model = tiny_llama()
        tokenizer = wordlevel_tokenizer()
        alpha = 2.0
        _, pasta = _pasta_pipeline(model, tokenizer, head_config=[0, 1], alpha=alpha, scale_position="include")

        input_ids = tokenizer("the cat sat on mat", return_tensors="pt").input_ids  # span (2, 4)
        input_len = input_ids.size(1)
        hooks = pasta.get_hooks(input_ids, {"substrings": ["cat sat"]})

        # register every pasta hook, then observe at layer 1: during a decode step layer 0's cache
        # is already updated when layer 1's pre-hook runs, so this exercises the per-layer cache
        # length derivation (a layer-0 lookup would overcount by the query length)
        captured = []
        handles = [
            model.get_submodule(spec["module"]).register_forward_pre_hook(spec["hook_func"], with_kwargs=True)
            for spec in hooks["pre"]
        ]
        observed = model.get_submodule(hooks["pre"][1]["module"])
        handles.append(
            observed.register_forward_pre_hook(
                lambda module, args, kwargs: captured.append(kwargs["attention_mask"]), with_kwargs=True
            )
        )

        try:
            with torch.no_grad():
                model.generate(
                    input_ids,
                    min_new_tokens=3,  # forces three forwards even if a random weight argmaxes eos
                    max_new_tokens=3,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
        finally:
            for handle in handles:
                handle.remove()

        assert len(captured) == 3  # prefill + 2 decode steps
        log_alpha = torch.tensor(alpha).log()
        for step, mask in enumerate(captured):
            assert mask is not None and mask.dim() == 4
            expected_q = input_len if step == 0 else 1
            expected_k = input_len + step
            assert mask.shape[2] == expected_q
            # the key axis must span the cached keys plus the current token; a (.., 1, 1) decode
            # mask is the v5 regression (cuda stride-0 rejection, cpu silent steering no-op)
            assert mask.shape[3] == expected_k, f"step {step}: key axis {mask.shape[3]}, expected {expected_k}"
            assert mask.stride(-1) == 1
            # steering persists at every step: span columns (2, 3) sit exactly log(alpha) above
            # the non-span real columns on the last (current) query row
            row = mask[0, 0, -1]
            for span_col in (2, 3):
                for other_col in (0, 1, 4, 5):
                    torch.testing.assert_close(
                        row[span_col] - row[other_col], log_alpha, rtol=1e-4, atol=1e-4
                    )
