"""Structural fact derivation through `text_config` on composite and plain models.

Pins that `resolve_layout(model=...)` reads the text sub-config on a composite multimodal wrapper
(never a silent `0`), that the derived facts agree with the session's `layout`, that a config
lacking a fact raises instead of defaulting, and that a config declaring `head_dim` per layer
(Gemma 4) reports no single `head_dim` while state controls still steer.
"""
import pytest
import torch

from steerability.algorithms.core.internals.model_layout import text_config
from steerability.algorithms.state_control.common.layout_facts import resolve_layout
from tests.utils.tiny_models import gemma4_declares_per_layer_head_dim, tiny_gemma3_conditional, tiny_gemma4_causal

LAYERS = 4
HIDDEN = 32
HEADS = 4


def test_resolve_layout_reads_text_config_on_composite():
    model = tiny_gemma3_conditional(num_layers=LAYERS, hidden=HIDDEN, heads=HEADS)
    facts = resolve_layout(model=model)
    assert facts.num_layers == LAYERS
    assert facts.hidden_size == HIDDEN
    assert facts.num_attention_heads == HEADS
    assert facts.head_dim == HIDDEN // HEADS
    assert facts.model_type == "gemma3"


def test_resolve_layout_matches_session_layout():
    from steerability.algorithms.core.execution.spec import BackendSpec
    from steerability.backends.huggingface import HFBackend

    model = tiny_gemma3_conditional(num_layers=LAYERS, hidden=HIDDEN, heads=HEADS)
    backend = HFBackend.adopt(BackendSpec(kind="huggingface"), lambda: model, lambda: None)
    with backend.open_session() as session:
        session_facts = session.layout
    model_facts = resolve_layout(model=model)
    assert model_facts.num_layers == session_facts.num_layers
    assert model_facts.hidden_size == session_facts.hidden_size
    assert model_facts.num_attention_heads == session_facts.num_attention_heads
    assert model_facts.head_dim == session_facts.head_dim
    assert model_facts.model_type == session_facts.model_type


def test_missing_hidden_size_raises_rather_than_zero():
    """A composite config whose text sub-config lacks `hidden_size` raises at the read site."""

    class _TextConfig:
        def get_text_config(self):
            return self

    class _StubModel:
        config = _TextConfig()

    with pytest.raises(AttributeError):
        _ = text_config(_StubModel()).hidden_size


@pytest.mark.skipif(
    not gemma4_declares_per_layer_head_dim(),
    reason="the installed transformers does not declare Gemma 4's head_dim per layer",
)
class TestPerLayerHeadDim:
    """Gemma 4 declares `head_dim` per layer; the facts report it as None and steering proceeds."""

    def test_resolve_layout_reports_per_layer_head_dim_as_none(self):
        model = tiny_gemma4_causal(num_layers=6, hidden=HIDDEN, heads=HEADS)
        facts = resolve_layout(model=model)
        assert facts.num_layers == 6
        assert facts.hidden_size == HIDDEN
        assert facts.num_attention_heads == HEADS
        assert facts.head_dim is None
        assert facts.model_type == "gemma4_text"

    def test_session_layout_agrees(self):
        from steerability.algorithms.core.execution.spec import BackendSpec
        from steerability.backends.huggingface import HFBackend

        model = tiny_gemma4_causal(num_layers=6, hidden=HIDDEN, heads=HEADS)
        backend = HFBackend.adopt(BackendSpec(kind="huggingface"), lambda: model, lambda: None)
        with backend.open_session() as session:
            session_facts = session.layout
        assert session_facts == resolve_layout(model=model)
        assert session_facts.head_dim is None

    def test_precomputed_caa_steers_and_generates(self):
        from steerability.algorithms.core.steering_pipeline import SteeringPipeline
        from steerability.algorithms.state_control.caa.control import CAA
        from steerability.algorithms.state_control.common.steering_vector import SteeringVector
        from tests.utils.tiny_models import wordlevel_tokenizer

        model = tiny_gemma4_causal(num_layers=6, hidden=HIDDEN, heads=HEADS)
        tokenizer = wordlevel_tokenizer()
        vector = SteeringVector(model_type="gemma4_text", directions={3: torch.ones(HIDDEN)})
        pipeline = SteeringPipeline(
            model=model, tokenizer=tokenizer, controls=[CAA(steering_vector=vector, layer_id=3, multiplier=2.0)],
            device="cpu",
        )
        pipeline.steer()
        out = pipeline.generate(text="the cat sat", max_new_tokens=3, do_sample=False, return_output=True)
        assert out.output_ids.shape[0] == 1 and out.output_ids.shape[1] <= 3
