"""Tests for hidden-state capture at module boundaries (`core/internals/capture.py`).

Covers the boundary indexing contract (`location="layer_input"` key `l` equals what a pre-hook on
layer `l` observes, i.e. `hidden_states[l]`), batching consistency, and input validation.
"""
import pytest
import torch

from steerability.algorithms.core.internals.capture import layerwise_tokenwise_hidden
from tests.utils.tiny_models import tiny_llama, wordlevel_tokenizer

NUM_LAYERS = 4


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return tiny_llama(num_layers=NUM_LAYERS)


@pytest.fixture(scope="module")
def enc():
    tokenizer = wordlevel_tokenizer()
    texts = ["the cat sat on mat", "dog ran fast", "attention span on the mat dog"]
    return tokenizer(texts, return_tensors="pt", padding=True)


class TestBoundaryIndexing:
    def test_returns_one_entry_per_layer(self, model, enc):
        for location in ("layer_output", "layer_input"):
            hidden = layerwise_tokenwise_hidden(model, dict(enc), location=location)
            assert sorted(hidden.keys()) == list(range(NUM_LAYERS))
            N, T = enc["input_ids"].shape
            for states in hidden.values():
                assert states.shape[:2] == (N, T)

    def test_layer_input_is_previous_layer_output(self, model, enc):
        out = layerwise_tokenwise_hidden(model, dict(enc), location="layer_output")
        inp = layerwise_tokenwise_hidden(model, dict(enc), location="layer_input")
        for lid in range(NUM_LAYERS - 1):
            torch.testing.assert_close(inp[lid + 1], out[lid])

    def test_layer_input_matches_pre_hook_view(self, model, enc):
        captured = {}

        def make_pre_hook(layer_id):
            def hook(_module, args, kwargs):
                hidden = args[0] if args else kwargs.get("hidden_states")
                captured[layer_id] = hidden.detach().cpu()
            return hook

        handles = [
            model.model.layers[lid].register_forward_pre_hook(make_pre_hook(lid), with_kwargs=True)
            for lid in range(NUM_LAYERS)
        ]
        try:
            with torch.no_grad():
                model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], use_cache=False)
        finally:
            for handle in handles:
                handle.remove()

        hidden = layerwise_tokenwise_hidden(model, dict(enc), location="layer_input")
        # layerwise extraction zeroes rows outside the attention mask, so it matches the raw
        # pre-hook view (which carries computed pad-row values) only at in-mask positions
        in_mask = enc["attention_mask"].bool().reshape(-1)
        for lid in range(NUM_LAYERS):
            hidden_rows = hidden[lid].reshape(-1, hidden[lid].size(-1))[in_mask]
            captured_rows = captured[lid].reshape(-1, captured[lid].size(-1))[in_mask]
            torch.testing.assert_close(hidden_rows, captured_rows)


class TestBatching:
    def test_batch_size_does_not_change_results(self, model, enc):
        whole = layerwise_tokenwise_hidden(model, dict(enc), batch_size=8)
        chunked = layerwise_tokenwise_hidden(model, dict(enc), batch_size=1)
        for lid in whole:
            torch.testing.assert_close(whole[lid], chunked[lid])

    def test_on_batch_called_per_batch(self, model, enc):
        calls = {"n": 0}

        def tick():
            calls["n"] += 1

        layerwise_tokenwise_hidden(model, dict(enc), batch_size=2, on_batch=tick)
        assert calls["n"] == 2  # 3 rows at batch_size=2


class TestValidation:
    def test_unsupported_location_raises(self, model, enc):
        with pytest.raises(ValueError, match="Unsupported hidden-state location"):
            layerwise_tokenwise_hidden(model, dict(enc), location="middle")  # type: ignore[arg-type]
