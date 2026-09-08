"""Behavioral tests for `HFBackend` and `ExclusiveSession` against tiny hub-free models."""
import pytest
import torch

from steerability.algorithms.core.execution import (
    BackendSpec,
    GenerationItem,
    GenerationParams,
    HookEntry,
    InterventionEntry,
    InterventionSpec,
    PreparedPrompt,
    ScoringItem,
    StackEntry,
    UnsupportedOperationError,
)
from steerability.algorithms.core.internals.capture import layerwise_tokenwise_hidden
from steerability.algorithms.core.steering_pipeline import SteeringPipeline
from steerability.backends.huggingface import HFBackend
from tests.utils.tiny_models import tiny_gpt2, tiny_llama, wordlevel_tokenizer

HF_SPEC = BackendSpec(kind="huggingface")


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return tiny_llama(num_layers=2, hidden=16, heads=2)


@pytest.fixture(scope="module")
def tokenizer():
    return wordlevel_tokenizer()


@pytest.fixture()
def backend(model, tokenizer):
    return HFBackend.adopt(HF_SPEC, lambda: model, lambda: tokenizer)


def _text_item(text, **kwargs):
    return GenerationItem(prompt=PreparedPrompt.from_text(text), **kwargs)


class TestBackendConstruction:

    def test_requires_huggingface_kind(self, model, tokenizer):
        with pytest.raises(ValueError, match="huggingface"):
            HFBackend.adopt(BackendSpec(kind="vllm", model="m"), lambda: model, lambda: tokenizer)

    def test_requires_model_reference_or_providers(self):
        with pytest.raises(ValueError, match="model reference"):
            HFBackend(HF_SPEC)


class TestSessionLifecycle:

    def test_one_open_session_per_backend(self, backend):
        session = backend.open_session()
        with pytest.raises(RuntimeError, match="exclusive session"):
            backend.open_session()
        session.close()
        backend.open_session().close()

    def test_closed_session_raises(self, backend):
        with backend.open_session() as session:
            pass
        with pytest.raises(RuntimeError, match="closed"):
            session.generate([_text_item("the cat")], GenerationParams(max_new_tokens=1))


class TestLayout:

    def test_llama_layout(self, backend):
        with backend.open_session() as session:
            layout = session.layout
        assert layout.num_layers == 2
        assert layout.hidden_size == 16
        assert layout.num_attention_heads == 2
        assert layout.head_dim == 8
        assert layout.dtype == "float32"
        assert len(layout.model_fingerprint) == 16

    def test_gpt2_layout(self, tokenizer):
        gpt2 = tiny_gpt2(num_layers=3, hidden=32, heads=4)
        backend = HFBackend.adopt(HF_SPEC, lambda: gpt2, lambda: tokenizer)
        with backend.open_session() as session:
            layout = session.layout
        assert layout.num_layers == 3
        assert layout.hidden_size == 32
        assert layout.head_dim == 8

    def test_composite_wrapper_layout_and_capture(self, tokenizer):
        """A composite multimodal wrapper reports its text-decoder facts and captures per layer."""
        from tests.utils.tiny_models import tiny_gemma3_conditional

        gemma = tiny_gemma3_conditional(num_layers=4, hidden=32, heads=4)
        backend = HFBackend.adopt(HF_SPEC, lambda: gemma, lambda: tokenizer)
        with backend.open_session() as session:
            layout = session.layout
            captured = session.capture(
                [PreparedPrompt.from_token_ids(torch.tensor([[3, 4, 5, 6]]))],
                layers=list(range(4)),
                mode="all_tokens",
            )
        assert layout.num_layers == 4
        assert layout.hidden_size == 32
        assert layout.num_attention_heads == 4
        assert layout.head_dim == 8
        assert layout.model_type == "gemma3"
        assert set(captured.hidden) == {0, 1, 2, 3}


class TestGenerate:

    def test_matches_direct_model_generate(self, backend, model, tokenizer):
        encoded = tokenizer(["the cat sat"], return_tensors="pt", padding=True)
        expected = model.generate(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            max_new_tokens=4,
            do_sample=False,
        )
        with backend.open_session() as session:
            results = session.generate(
                [_text_item("the cat sat")],
                GenerationParams(max_new_tokens=4, greedy=True),
            )
        assert len(results) == 1
        output = results[0].output
        assert torch.equal(output.adapted_input_ids, encoded["input_ids"])
        assert torch.equal(output.output_ids, expected[:, encoded["input_ids"].size(1):])

    def test_seeded_generation_is_repeatable_and_leaves_rng_untouched(self, backend):
        params = GenerationParams(max_new_tokens=8, greedy=False, temperature=1.0)
        items = [_text_item("the cat", seed=1234)]
        rng_state = torch.get_rng_state()
        with backend.open_session() as session:
            first = session.generate(items, params)
        with backend.open_session() as session:
            second = session.generate(items, params)
        assert torch.equal(first[0].output.output_ids, second[0].output.output_ids)
        assert torch.equal(rng_state, torch.get_rng_state())

    @pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
    def test_seeded_generation_restores_mps_rng(self, tokenizer):
        mps_model = tiny_llama(num_layers=2, hidden=16, heads=2).to("mps")
        backend = HFBackend.adopt(HF_SPEC, lambda: mps_model, lambda: tokenizer)
        params = GenerationParams(max_new_tokens=4, greedy=False, temperature=1.0)
        mps_state = torch.mps.get_rng_state()
        with backend.open_session() as session:
            session.generate([_text_item("the cat", seed=7)], params)
        assert torch.equal(mps_state, torch.mps.get_rng_state())

    def test_extra_logits_processor_merges_with_item_stacks(self, backend):
        def _force_token(prefix_ids, scores):
            forced = torch.full_like(scores, float("-inf"))
            forced[:, 5] = 0.0
            return forced

        def _identity(prefix_ids, scores):
            return scores

        entry = StackEntry(logits_processors=(_identity,))
        params = GenerationParams(max_new_tokens=2, greedy=True, extra={"logits_processor": [_force_token]})
        with backend.open_session() as session:
            results = session.generate([_text_item("the cat", output_entries=(entry,))], params)
        assert results[0].output.output_ids.tolist() == [[5, 5]]

    def test_hook_entry_applies_and_unregisters(self, backend, model):
        def _shift_hidden(module, args, kwargs):
            if args:
                return (args[0] + 10.0, *args[1:]), kwargs
            kwargs["hidden_states"] = kwargs["hidden_states"] + 10.0
            return args, kwargs

        hooks = {"pre": [{"module": "model.layers.1", "hook_func": _shift_hidden}], "forward": [], "backward": []}
        params = GenerationParams(max_new_tokens=4, greedy=True)
        with backend.open_session() as session:
            plain = session.generate([_text_item("the cat sat")], params)
            hooked = session.generate(
                [_text_item("the cat sat", state_entries=(HookEntry(hooks=hooks),))], params,
            )
        assert not torch.equal(plain[0].output.output_ids, hooked[0].output.output_ids)
        assert len(model.model.layers[1]._forward_pre_hooks) == 0

    def test_stack_entry_processor_applies(self, backend):
        def _force_token(prefix_ids, scores):
            forced = torch.full_like(scores, float("-inf"))
            forced[:, 5] = 0.0
            return forced

        entry = StackEntry(logits_processors=(_force_token,))
        with backend.open_session() as session:
            results = session.generate(
                [_text_item("the cat", output_entries=(entry,))],
                GenerationParams(max_new_tokens=3, greedy=True),
            )
        assert results[0].output.output_ids.tolist() == [[5, 5, 5]]

    def test_intervention_entry_unsupported(self, backend):
        item = _text_item("the cat", state_entries=(InterventionEntry(spec=InterventionSpec()),))
        with backend.open_session() as session:
            with pytest.raises(UnsupportedOperationError, match="intervention-capable"):
                session.generate([item], GenerationParams(max_new_tokens=1))


class TestScore:

    def test_matches_pipeline_compute_logprobs(self, model, tokenizer, backend):
        pipeline = SteeringPipeline(controls=[], model=model, tokenizer=tokenizer)
        pipeline.steer()

        encoded = tokenizer(["the cat sat"], return_tensors="pt", padding=True)
        ref = torch.tensor([[4, 5, 6]])
        expected = pipeline.compute_logprobs(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            ref_output_ids=ref,
        )
        item = ScoringItem(
            prompt=PreparedPrompt.from_token_ids(encoded["input_ids"], encoded["attention_mask"]),
            ref_output_ids=ref,
        )
        with backend.open_session() as session:
            scored = session.score([item], GenerationParams())
        assert torch.allclose(scored, expected, atol=1e-5)

    def test_mismatched_ref_lengths_rejected(self, backend):
        items = [
            ScoringItem(prompt=PreparedPrompt.from_text("the cat"), ref_output_ids=torch.tensor([4, 5])),
            ScoringItem(prompt=PreparedPrompt.from_text("dog ran"), ref_output_ids=torch.tensor([4])),
        ]
        with backend.open_session() as session:
            with pytest.raises(ValueError, match="reference length"):
                session.score(items, GenerationParams())

    def test_right_padded_prompt_scores_like_unpadded(self, backend, tokenizer):
        encoded = tokenizer(["the cat sat"], return_tensors="pt", padding=True)
        pad_id = tokenizer.pad_token_id
        padded_ids = torch.cat(
            [encoded["input_ids"], torch.full((1, 3), pad_id, dtype=torch.long)], dim=1,
        )
        padded_mask = torch.cat(
            [encoded["attention_mask"], torch.zeros((1, 3), dtype=torch.long)], dim=1,
        )
        ref = torch.tensor([[4, 5]])
        with backend.open_session() as session:
            unpadded = session.score(
                [ScoringItem(
                    prompt=PreparedPrompt.from_token_ids(encoded["input_ids"], encoded["attention_mask"]),
                    ref_output_ids=ref,
                )],
                GenerationParams(),
            )
            padded = session.score(
                [ScoringItem(
                    prompt=PreparedPrompt.from_token_ids(padded_ids, padded_mask),
                    ref_output_ids=ref,
                )],
                GenerationParams(),
            )
        assert torch.allclose(padded, unpadded, atol=1e-5)

    def test_empty_items_return_empty_tensor(self, backend):
        with backend.open_session() as session:
            scored = session.score([], GenerationParams())
        assert scored.shape == (0, 0)

    def test_empty_capture_rejected(self, backend):
        with backend.open_session() as session:
            with pytest.raises(ValueError, match="at least one prompt"):
                session.capture([], layers=[0], mode="all_tokens")


class TestCapture:

    def test_matches_layerwise_extraction(self, backend, model, tokenizer):
        texts = ["the cat sat on mat", "dog ran"]
        encoded = tokenizer(texts, return_tensors="pt", padding=True)
        expected = layerwise_tokenwise_hidden(model, dict(encoded), location="layer_output")
        prompts = [PreparedPrompt.from_text(text) for text in texts]
        with backend.open_session() as session:
            captured = session.capture(prompts, layers=[0, 1], mode="all_tokens")
        assert set(captured.hidden) == {0, 1}
        for layer in (0, 1):
            assert torch.allclose(captured.hidden[layer], expected[layer], atol=1e-5)
        assert torch.equal(captured.attention_mask, encoded["attention_mask"].cpu())

    def test_last_token_mode_selects_last_real_position(self, backend, tokenizer):
        texts = ["the cat sat on mat", "dog ran"]
        prompts = [PreparedPrompt.from_text(text) for text in texts]
        with backend.open_session() as session:
            all_tokens = session.capture(prompts, layers=[1], mode="all_tokens")
            last_token = session.capture(prompts, layers=[1], mode="last_token")
        lengths = all_tokens.attention_mask.sum(dim=1)
        assert last_token.hidden[1].shape == (2, 16)
        for row, length in enumerate(lengths.tolist()):
            assert torch.allclose(
                last_token.hidden[1][row], all_tokens.hidden[1][row, length - 1], atol=1e-6,
            )

    def test_unknown_mode_rejected(self, backend):
        with backend.open_session() as session:
            with pytest.raises(ValueError, match="capture mode"):
                session.capture([PreparedPrompt.from_text("the cat")], layers=[0], mode="middle")

    def test_out_of_range_layer_rejected(self, backend):
        with backend.open_session() as session:
            with pytest.raises(ValueError, match="out of range"):
                session.capture([PreparedPrompt.from_text("the cat")], layers=[7], mode="all_tokens")


class TestPreparedPromptContract:

    def test_exactly_one_source_required(self):
        with pytest.raises(ValueError, match="exactly one"):
            PreparedPrompt(text="the cat", token_ids=torch.tensor([[1]]))
        with pytest.raises(ValueError, match="exactly one"):
            PreparedPrompt()

    def test_text_resolution_matches_tokenizer(self, tokenizer):
        prompt = PreparedPrompt.from_text("the cat sat").resolve_token_ids(tokenizer)
        encoded = tokenizer(["the cat sat"], return_tensors="pt", padding=True)
        assert torch.equal(prompt.token_ids, encoded["input_ids"])

    def test_token_form_passes_through(self, tokenizer):
        prompt = PreparedPrompt.from_token_ids([3, 4, 5])
        assert prompt.resolve_token_ids(tokenizer) is prompt
        assert prompt.token_ids.shape == (1, 3)

    def test_resolution_without_tokenizer_rejected(self):
        with pytest.raises(ValueError, match="tokenizer"):
            PreparedPrompt.from_text("the cat").resolve_token_ids(None)

    def test_multi_row_token_ids_rejected(self):
        with pytest.raises(ValueError, match="one prompt row"):
            PreparedPrompt.from_token_ids(torch.tensor([[1, 2], [3, 4]]))
