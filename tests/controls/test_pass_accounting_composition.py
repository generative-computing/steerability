"""Integration tests for position and gate accounting under non-linear decode patterns.

Composes a real runtime-based state control with output controls that break the plain decode
pattern: a score-preserving probe that forwards the pipeline's own model per step (the SASA
shape), multi-call decoding drivers (`SearchDecoding`, `PhasedDecoding`), and a detached
variant-prompt source (`ContrastiveGuidance` over `PromptVariantSource`). Position scoping,
condition scoring, and greedy outputs must be unaffected by the extra passes.

Runs hub-free on a tiny randomly-initialized Llama.
"""
import warnings

import pytest
import torch

from aisteer360.algorithms.core.execution.access import ModelAccess
from aisteer360.algorithms.core.steering_pipeline import SteeringPipeline
from aisteer360.algorithms.output_control.base import OutputControl
from aisteer360.algorithms.output_control.common.candidate_forward import CandidateForward
from aisteer360.algorithms.output_control.common.logit_sources import PromptVariantSource
from aisteer360.algorithms.output_control.contrastive_guidance.control import ContrastiveGuidance
from aisteer360.algorithms.output_control.phased_decoding.control import PhasedDecoding
from aisteer360.algorithms.output_control.search_decoding.control import SearchDecoding
from aisteer360.algorithms.state_control.base import StateControl
from aisteer360.algorithms.state_control.common.gating import CallableReadout, Evidence, Gate
from aisteer360.algorithms.state_control.common.runtime import TransformHookRuntime
from aisteer360.algorithms.state_control.common.token_scope import compute_prompt_lens
from tests.utils.runtime_helpers import NeverCompleteRule, RecordingTransform
from tests.utils.tiny_models import tiny_llama, wordlevel_tokenizer

HIDDEN = 32
HEADS = 4
LAYERS = 4

PROMPT_IDS = torch.arange(3, 7, dtype=torch.long).unsqueeze(0)  # prompt_len 4
GEN_KWARGS = {"do_sample": False, "eos_token_id": None}


class _RecordingStateControl(StateControl):
    """Real state control on the shared runtime: records token masks and adds a constant.

    With `with_condition=True`, a condition hook on `model.layers.0` (the pass opener) feeds a
    never-complete gate through a counting readout, and the behavior hook follows as a
    non-opener.
    """

    Args = None
    supports_batching = True

    def __init__(self, hook_point="layer_output", module_path="model.layers.1", with_condition=False,
                 token_scope="after_prompt", from_position=None):
        super().__init__()
        self.hooks = {"pre": [], "forward": [], "backward": []}
        self.registered = []
        self.module_path = module_path
        self.with_condition = with_condition
        self.token_scope = token_scope
        self.from_position = from_position
        self.runtime = TransformHookRuntime(hook_point=hook_point)
        self.transform = RecordingTransform(value=0.5)
        self.scorer_calls: list[tuple] = []
        self.gate = None
        if with_condition:
            def readout(pooled, layer_id):
                self.scorer_calls.append(tuple(pooled.shape))
                return torch.zeros(pooled.size(0))

            self.gate = Gate(Evidence((0,), CallableReadout(readout)), NeverCompleteRule(open=True))

    def get_hooks(self, input_ids, runtime_kwargs, attention_mask=None, **kwargs):
        ids = input_ids if isinstance(input_ids, torch.Tensor) else input_ids["input_ids"]
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)
        self.runtime.reset(compute_prompt_lens(ids, None))
        if self.gate is not None:
            self.gate.reset(self.runtime.num_logical_rows)

        specs = {"pre": [], "forward": [], "backward": []}
        if self.with_condition:
            specs["forward"].append({
                "module": "model.layers.0",
                "hook_func": self.runtime.build_condition_hook(
                    layer_id=0, gate=self.gate, is_pass_opener=True),
            })
        behavior_hook = self.runtime.build_behavior_hook(
            layer_id=1, transform=self.transform, gate=self.gate,
            token_scope=self.token_scope, from_position=self.from_position,
            is_pass_opener=not self.with_condition)
        phase = "forward" if self.runtime.hook_point == "layer_output" else "pre"
        specs[phase].append({"module": self.module_path, "hook_func": behavior_hook})
        return specs


class _SameModelScoringControl(OutputControl):
    """Score-preserving probe: forwards the pipeline's own model per decode step (the SASA shape).

    The processor evaluates the top-3 candidates through `CandidateForward` and returns the scores
    unchanged, so any output divergence is attributable to state-control misapplication.
    """

    Args = None
    same_model_forwards = True

    def steer_access(self) -> ModelAccess:
        return ModelAccess.MODULE

    def steer(self, model, tokenizer=None, **kwargs):
        self.model = model
        return model

    def get_logits_processors(self, input_ids, runtime_kwargs, **kwargs):
        forward = CandidateForward(self.model)

        def _probe(prefix_ids, scores):
            candidate_ids = scores.topk(3, dim=-1).indices
            forward.last_hidden_states(prefix_ids, candidate_ids, None)
            return scores

        return [_probe]


def _steered_pipeline(model, tokenizer, controls) -> SteeringPipeline:
    pipeline = SteeringPipeline(controls=controls, model=model, tokenizer=tokenizer)
    pipeline.steer()
    return pipeline


def _fixtures():
    torch.manual_seed(0)
    model = tiny_llama(num_layers=LAYERS, hidden=HIDDEN, heads=HEADS)
    tokenizer = wordlevel_tokenizer()
    return model, tokenizer


RECORDER_PLACEMENTS = {
    "decoder_layer": {},
    "o_proj": {"hook_point": "layer_input", "module_path": "model.layers.1.self_attn.o_proj"},
}


class TestSameModelScoringComposition:
    @pytest.mark.parametrize("placement", list(RECORDER_PLACEMENTS))
    def test_greedy_output_parity_and_warnings(self, placement):
        """A score-preserving same-model probe leaves the steered greedy output unchanged.

        At decoder-layer hook points the composition is exact and silent; at `o_proj` (no
        `cache_position`), auxiliary transforms are skipped with exactly one warning and the
        main-pass accounting stays untouched, so parity still holds. The `from_position` scope
        makes the transform sensitive to absolute positions mid-generation, so any offset skew
        from the auxiliary passes changes the steered hidden states and thus the greedy output.
        """
        model, tokenizer = _fixtures()
        recorder_kwargs = {"token_scope": "from_position", "from_position": 6, **RECORDER_PLACEMENTS[placement]}

        recorder_a = _RecordingStateControl(**recorder_kwargs)
        pipeline_a = _steered_pipeline(model, tokenizer, [recorder_a])
        out_a = pipeline_a.generate(input_ids=PROMPT_IDS, max_new_tokens=6, **GEN_KWARGS)

        recorder_b = _RecordingStateControl(**recorder_kwargs)
        pipeline_b = _steered_pipeline(model, tokenizer, [recorder_b, _SameModelScoringControl()])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out_b = pipeline_b.generate(input_ids=PROMPT_IDS, max_new_tokens=6, **GEN_KWARGS)

        assert torch.equal(out_a, out_b)
        aux_warnings = [w for w in caught if "Auxiliary same-model passes" in str(w.message)]
        clock_warnings = [w for w in caught if "cache_position" in str(w.message)]
        if placement == "decoder_layer":
            assert not aux_warnings and not clock_warnings
        else:
            assert len(aux_warnings) == 1

    def test_gate_isolation_scorer_call_counts(self):
        """Condition scorers see the same passes with and without the same-model probe."""
        model, tokenizer = _fixtures()

        recorder_a = _RecordingStateControl(with_condition=True)
        pipeline_a = _steered_pipeline(model, tokenizer, [recorder_a])
        pipeline_a.generate(input_ids=PROMPT_IDS, max_new_tokens=6, **GEN_KWARGS)

        recorder_b = _RecordingStateControl(with_condition=True)
        pipeline_b = _steered_pipeline(model, tokenizer, [recorder_b, _SameModelScoringControl()])
        pipeline_b.generate(input_ids=PROMPT_IDS, max_new_tokens=6, **GEN_KWARGS)

        assert recorder_a.scorer_calls == recorder_b.scorer_calls


class TestMultiCallDriverComposition:
    def _assert_prompt_columns_unsteered(self, recorder, prompt_len):
        wide_masks = [m for m in recorder.transform.masks if m.size(1) >= prompt_len]
        assert wide_masks  # a later call's re-prefill pass was recorded
        for mask in wide_masks:
            assert not bool(mask[:, :prompt_len].any())

    def test_recorded_mask_hygiene_under_search_decoding(self):
        model, tokenizer = _fixtures()
        recorder = _RecordingStateControl()
        search = SearchDecoding(
            scorer=lambda prompt, continuations, params: [float(len(c)) for c in continuations],
            segment_len=2,
            num_candidates=2,
            keep_k=1,
            max_iterations=2,
            propose_mode="sample",
        )
        pipeline = _steered_pipeline(model, tokenizer, [recorder, search])
        pipeline.generate(input_ids=PROMPT_IDS, max_new_tokens=4, **GEN_KWARGS)
        self._assert_prompt_columns_unsteered(recorder, PROMPT_IDS.size(1))

    def test_recorded_mask_hygiene_under_phased_decoding(self):
        model, tokenizer = _fixtures()
        recorder = _RecordingStateControl()
        phased = PhasedDecoding(
            plan=[{"generate": {"budget": 3}}, {"fixed": " cat "}, {"generate": {"budget": 3}}],
        )
        pipeline = _steered_pipeline(model, tokenizer, [recorder, phased])
        pipeline.generate(input_ids=PROMPT_IDS, **GEN_KWARGS)
        self._assert_prompt_columns_unsteered(recorder, PROMPT_IDS.size(1))


class TestDetachedSourceComposition:
    def test_variant_prompt_end_to_end(self):
        """A zero-weight variant source neither raises, changes outputs, nor gets transformed."""
        model, tokenizer = _fixtures()

        recorder_a = _RecordingStateControl()
        pipeline_a = _steered_pipeline(model, tokenizer, [recorder_a])
        out_a = pipeline_a.generate(input_ids=PROMPT_IDS, max_new_tokens=6, **GEN_KWARGS)

        recorder_b = _RecordingStateControl()
        guidance = ContrastiveGuidance(
            sources=[PromptVariantSource(prompt_transform=lambda text: " ".join(text.split()[-1:]) or text)],
            weights=[0.0],
        )
        pipeline_b = _steered_pipeline(model, tokenizer, [recorder_b, guidance])
        out_b = pipeline_b.generate(input_ids=PROMPT_IDS, max_new_tokens=6, **GEN_KWARGS)

        assert torch.equal(out_a, out_b)
        assert len(recorder_b.transform.masks) == len(recorder_a.transform.masks)
