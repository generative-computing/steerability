"""Venue-matched pre-fitted artifact identity on engine layouts (model_ref and model_type)."""
import pytest
import torch

from aisteer360.algorithms.core.execution import ModelFacts
from aisteer360.algorithms.core.internals.probes.probe import Probe
from aisteer360.algorithms.core.internals.probes.probe_set import ProbeSet
from aisteer360.algorithms.output_control.routed_decoding import P, Route, RoutedDecoding, Router
from aisteer360.algorithms.output_control.routed_decoding.actions import respond
from tests.utils.tiny_models import wordlevel_tokenizer

HIDDEN = 16


class _EngineSession:
    """Engine-session double: layout facts only, no live model."""

    def __init__(self, model_ref="org/served-model", model_type="llama"):
        self._model_ref = model_ref
        self._model_type = model_type

    @property
    def layout(self) -> ModelFacts:
        return ModelFacts(
            num_layers=2, hidden_size=HIDDEN, num_attention_heads=2, head_dim=HIDDEN // 2,
            dtype="float32", model_fingerprint="c" * 16,
            model_type=self._model_type, model_ref=self._model_ref,
        )


def _probe(meta=None, model_type="llama") -> Probe:
    return Probe(
        model_type=model_type, location="layer_input", pooling="mean",
        layer_ids=[0], weights={0: torch.zeros(HIDDEN)}, bias=1e9, meta=dict(meta or {}),
    )


def _routed(probe) -> RoutedDecoding:
    return RoutedDecoding(
        probes=ProbeSet({"p": probe}),
        rules=Router(routes=[Route("r", when=P("p"), action=respond("x"))]),
    )


class TestEngineModelRefComparison:

    def test_matching_recorded_ref_accepts(self):
        control = _routed(_probe(meta={"model_ref": "org/served-model"}))
        control.steer(model=None, tokenizer=wordlevel_tokenizer(), session=_EngineSession())

    def test_unrecorded_ref_is_exempt(self):
        control = _routed(_probe(meta={"model_fingerprint": "a" * 16}))
        control.steer(model=None, tokenizer=wordlevel_tokenizer(), session=_EngineSession())

    def test_differing_recorded_ref_raises_naming_the_served_reference(self):
        control = _routed(_probe(meta={"model_ref": "org/other-model"}))
        with pytest.raises(ValueError, match="model reference.*'org/served-model'"):
            control.steer(model=None, tokenizer=wordlevel_tokenizer(), session=_EngineSession())

    def test_escape_hatch_skips_the_ref_comparison(self):
        control = _routed(_probe(meta={"model_ref": "org/other-model"}))
        control.allow_model_mismatch = True
        control.steer(model=None, tokenizer=wordlevel_tokenizer(), session=_EngineSession())

    def test_known_model_type_mismatch_raises(self):
        control = _routed(_probe(model_type="gpt2"))
        with pytest.raises(ValueError, match="model_type 'gpt2'.*serves 'llama'"):
            control.steer(model=None, tokenizer=wordlevel_tokenizer(), session=_EngineSession())

    def test_unknown_model_type_is_exempt_on_engines(self):
        control = _routed(_probe(model_type="unknown"))
        control.steer(model=None, tokenizer=wordlevel_tokenizer(), session=_EngineSession())
