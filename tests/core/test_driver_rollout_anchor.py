"""Driver-rollout anchor golden.

A decoding driver's rollouts run through the pipeline's `SteeredSession` while the in-process
session hosts the generation's hooks. Hooks are built once per logical generation and close over
the original prompt boundary, so continuation tokens re-prefilled by a later rollout are
re-steered at their original positions and a rollout's continuation distribution matches a
single steered pass.
"""
import torch
from transformers import LogitsProcessorList, StoppingCriteriaList

from aisteer360.algorithms.core.steering_pipeline import SteeringPipeline
from aisteer360.algorithms.output_control.base import DecodingDriver, session_generate
from aisteer360.algorithms.state_control.caa.control import CAA
from aisteer360.algorithms.state_control.common.steering_vector import SteeringVector
from tests.utils.runtime_helpers import RecordingTransform
from tests.utils.tiny_models import tiny_llama, wordlevel_tokenizer

HIDDEN = 32
HEADS = 4
LAYERS = 4
FIRST_LEG = 3
SECOND_LEG = 3


def _steering_vector(seed: int = 5) -> SteeringVector:
    generator = torch.Generator().manual_seed(seed)
    return SteeringVector(
        model_type="llama",
        directions={1: torch.randn(1, HIDDEN, generator=generator)},
    )


class _TwoLegDriver(DecodingDriver):
    """Generates in two rollouts: the second re-prefills the first leg's continuation."""

    Args = None
    supports_batching = True

    def decode(self, input_ids, attention_mask, model, logits_processors,
               stopping_criteria, runtime_kwargs, session=None, **gen_kwargs):
        first = session_generate(
            session, input_ids, attention_mask,
            max_new_tokens=FIRST_LEG, do_sample=False, eos_token_id=None,
        )
        return session_generate(
            session, first, torch.ones_like(first),
            max_new_tokens=SECOND_LEG, do_sample=False, eos_token_id=None,
        )


def _steered_pipeline(control, model):
    pipeline = SteeringPipeline(
        controls=[control] if not isinstance(control, list) else control,
        model=model,
        tokenizer=wordlevel_tokenizer(),
    )
    pipeline.steer()
    return pipeline


def test_two_leg_rollouts_match_single_steered_pass():
    """Greedy continuation over two rollouts equals one steered pass of the combined length."""
    torch.manual_seed(0)
    model = tiny_llama(num_layers=LAYERS, hidden=HIDDEN, heads=HEADS)
    steering_vector = _steering_vector()
    prompt = torch.arange(3, 8, dtype=torch.long).unsqueeze(0)

    single = _steered_pipeline(
        CAA(steering_vector=steering_vector, layer_id=1, multiplier=6.0, token_scope="after_prompt"),
        model,
    )
    reference = single.generate(
        input_ids=prompt, max_new_tokens=FIRST_LEG + SECOND_LEG, do_sample=False,
        eos_token_id=None, return_full_sequence=True,
    )

    driven = _steered_pipeline(
        [
            CAA(steering_vector=steering_vector, layer_id=1, multiplier=6.0, token_scope="after_prompt"),
            _TwoLegDriver(),
        ],
        model,
    )
    two_leg = driven.generate(
        input_ids=prompt, do_sample=False, eos_token_id=None, return_full_sequence=True,
    )

    assert torch.equal(reference, two_leg)


def test_second_rollout_resteers_continuation_at_original_positions():
    """The second rollout's re-prefill steers exactly the re-prefilled continuation columns."""
    torch.manual_seed(0)
    model = tiny_llama(num_layers=LAYERS, hidden=HIDDEN, heads=HEADS)
    prompt = torch.arange(3, 8, dtype=torch.long).unsqueeze(0)
    prompt_len = prompt.size(1)

    control = CAA(
        steering_vector=_steering_vector(), layer_id=1, multiplier=6.0, token_scope="after_prompt",
    )
    pipeline = _steered_pipeline([control, _TwoLegDriver()], model)

    recorder = RecordingTransform(value=0.0)
    control._transform = recorder
    pipeline.generate(input_ids=prompt, do_sample=False, eos_token_id=None)

    # the second rollout's prefill forwards [prompt; first-leg continuation] in one pass
    reprefill_masks = [m for m in recorder.masks if m.size(1) == prompt_len + FIRST_LEG]
    assert reprefill_masks, "expected a re-prefill pass covering prompt plus first-leg tokens"
    mask = reprefill_masks[0][0]
    assert not mask[:prompt_len].any()  # the user prompt stays unsteered
    assert mask[prompt_len:].all()  # re-prefilled continuation tokens re-steered at their positions


class TestWireAnchorRewrite:
    """Wire twin of the anchor golden: prompt-relative scope kinds are client-side sugar, and
    their wire form inside a driver generation is absolute."""

    def _lowered_spec(self, scope_kwargs):
        import pytest

        pytest.importorskip("vllm_hook_plugins")
        from aisteer360.algorithms.state_control.common.lowering import lower_interventions
        from aisteer360.algorithms.state_control.common.specs import Intervention, TokenScope
        from aisteer360.algorithms.state_control.common.transforms import AdditiveTransform

        intervention = Intervention(
            layers=(1,),
            transform=AdditiveTransform({1: torch.ones(1, HIDDEN)}, strength=2.0),
            scope=TokenScope(**scope_kwargs),
        )
        return lower_interventions([intervention], num_layers=LAYERS)

    def test_after_prompt_rewrites_to_absolute_anchor(self):
        from aisteer360.algorithms.core.execution.payloads import remap_prompt_relative_scopes

        spec = self._lowered_spec({"kind": "after_prompt"})
        rewritten = remap_prompt_relative_scopes(spec, anchor=7)
        assert rewritten.ops[0]["scope"] == {"kind": "from_position", "position": 7}
        # one scalar changed per op; artifact ids are untouched
        assert rewritten.artifact_ids() == spec.artifact_ids()
        # the cache salt varies with the anchor: differently anchored requests genuinely
        # compute different hidden states
        assert rewritten.salt() != spec.salt()

    def test_last_k_has_no_absolute_rollout_form(self):
        import pytest

        from aisteer360.algorithms.core.execution.payloads import remap_prompt_relative_scopes

        spec = self._lowered_spec({"kind": "last_k", "last_k": 3})
        # in-process last_k is relative to each forwarded pass, which no fixed position
        # reproduces across rollouts, so the rewrite refuses rather than misanchor
        with pytest.raises(ValueError, match="last_k"):
            remap_prompt_relative_scopes(spec, anchor=7)

    def test_absolute_scopes_pass_through_unchanged(self):
        from aisteer360.algorithms.core.execution.payloads import remap_prompt_relative_scopes

        spec = self._lowered_spec({"kind": "all"})
        assert remap_prompt_relative_scopes(spec, anchor=7) is spec

    def test_steered_session_injects_rewritten_entries_per_item(self):
        from aisteer360.algorithms.core.execution import InterventionEntry
        from aisteer360.algorithms.core.execution.backend import SteeredSession
        from aisteer360.algorithms.core.execution.payloads import (
            GenerationItem,
            PreparedPrompt,
            remap_prompt_relative_scopes,
        )

        spec = self._lowered_spec({"kind": "after_prompt"})
        entry = InterventionEntry(spec=remap_prompt_relative_scopes(spec, anchor=5))

        captured = {}

        class _ProbeSession:
            def generate(self, items, params):
                captured["items"] = items
                return []

        steered = SteeredSession(_ProbeSession(), (entry,))
        prompt = PreparedPrompt.from_token_ids(torch.ones(1, 9, dtype=torch.long), None)
        steered.generate([GenerationItem(prompt=prompt)], params=None)

        (item,) = captured["items"]
        (injected,) = item.state_entries
        assert injected is entry
        assert injected.spec.ops[0]["scope"] == {"kind": "from_position", "position": 5}
