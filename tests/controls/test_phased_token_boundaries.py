"""Token-id phase boundaries for phased drivers (`Generated.until_token_ids`).

A special-token delimiter is stripped by `skip_special_tokens=True`, so a stop string holding one
never fires on vLLM; `until_token_ids` is the portable form that lowers to `stop_token_ids` on
every backend. These tests cover the Hugging Face stop, the session-path lowering to
`stop_token_ids` (rendered into vLLM sampling args), the both-boundaries case, plan validation, and
`BudgetForcing.end_think_token_ids`.
"""
import pytest
import torch

from steerability.algorithms.core.steering_pipeline import SteeringPipeline
from steerability.algorithms.output_control.base import OutputControl
from steerability.algorithms.output_control.budget_forcing.control import BudgetForcing
from steerability.algorithms.output_control.common.drivers.phased import Generated
from steerability.algorithms.output_control.phased_decoding.control import PhasedDecoding, _parse_phase
from steerability.backends.vllm import render_vllm_sampling_args
from tests.utils.runtime_helpers import script_session_generate
from tests.utils.tiny_models import reasoning_tag_tokenizer, tiny_llama

VOCAB = 100


def _pipeline(controls, tokenizer, model=None):
    if model is None:
        model = tiny_llama(num_layers=2, hidden=16, heads=2, vocab=VOCAB)
    pipeline = SteeringPipeline(controls=controls, model=model, tokenizer=tokenizer)
    pipeline.steer()
    return pipeline, model, tokenizer


class _ForceToken(OutputControl):
    """Force the next token to a fixed id at every step, so a phase boundary is reachable."""

    Args = None

    def __init__(self, token_id: int):
        self._token_id = token_id

    def get_logits_processors(self, input_ids, runtime_kwargs, **kwargs):
        token_id = self._token_id

        def _force(prefix_ids, scores):
            out = torch.full_like(scores, float("-inf"))
            out[:, token_id] = 0.0
            return out

        return [_force]


class TestGeneratedDataclass:
    def test_until_token_ids_normalized_to_int_tuple(self):
        phase = Generated(until_token_ids=[5, 7])
        assert phase.until_token_ids == (5, 7)
        assert all(isinstance(i, int) for i in phase.until_token_ids)


class TestHuggingFaceStop:
    def test_stops_at_the_token_and_splices_the_fixed_phase(self):
        # register a special-token delimiter and force the model to emit it
        tokenizer = reasoning_tag_tokenizer(special_tags=("<end>",))
        close_id = tokenizer.convert_tokens_to_ids("<end>")
        driver = PhasedDecoding(plan=[
            {"generate": {"until_token_ids": [close_id]}},
            {"fixed": "answer", "add_special_tokens": False},
            {"generate": {}},
        ])
        pipeline, model, tokenizer = _pipeline([_ForceToken(close_id), driver], tokenizer)
        prompt = tokenizer("R", return_tensors="pt").input_ids
        out = pipeline.generate(
            input_ids=prompt, max_new_tokens=8, do_sample=False, return_full_sequence=True,
        )
        decoded = tokenizer.decode(out[0], skip_special_tokens=False)
        # the stop token is present (so the first phase stopped at it) and the fixed phase spliced after
        assert "<end>" in decoded
        assert "answer" in decoded


class TestSessionLowering:
    def test_until_token_ids_lower_to_stop_token_ids(self, monkeypatch):
        tokenizer = reasoning_tag_tokenizer(special_tags=("<end>",))
        close_id = tokenizer.convert_tokens_to_ids("<end>")
        driver = PhasedDecoding(plan=[
            {"generate": {"until_token_ids": [close_id]}},
            {"fixed": "answer", "add_special_tokens": False},
            {"generate": {}},
        ])
        pipeline, model, tokenizer = _pipeline([driver], tokenizer)

        seen_stop_token_ids = []

        def fake_generate(**kwargs):
            seen_stop_token_ids.append(tuple(kwargs.get("stop_token_ids") or ()))
            inp = kwargs["input_ids"]
            cont = torch.tensor([[close_id]], dtype=torch.long)
            return torch.cat([inp, cont.expand(inp.size(0), -1).to(inp.device)], dim=1)

        script_session_generate(monkeypatch, fake_generate)
        prompt = tokenizer("R", return_tensors="pt").input_ids
        pipeline.generate(input_ids=prompt, max_new_tokens=8, do_sample=False, return_full_sequence=True)
        # the first (bounded) phase carried the close id as a stop token id; the answer phase did not
        assert close_id in seen_stop_token_ids[0]
        assert seen_stop_token_ids[-1] == ()

    def test_rendered_vllm_args_carry_the_stop_token(self):
        from steerability.algorithms.core.execution.params import GenerationParams

        args = render_vllm_sampling_args(GenerationParams(stop_token_ids=(41,)))
        assert args["stop_token_ids"] == [41]


class TestBothBoundaries:
    def test_stops_at_whichever_occurs_first(self, monkeypatch):
        # a plan carrying both until (a substring) and until_token_ids lowers both on the session path
        tokenizer = reasoning_tag_tokenizer(special_tags=("<end>",))
        close_id = tokenizer.convert_tokens_to_ids("<end>")
        driver = PhasedDecoding(plan=[
            {"generate": {"until": "stop", "until_token_ids": [close_id]}},
            {"generate": {}},
        ])
        pipeline, model, tokenizer = _pipeline([driver], tokenizer)

        seen = {}

        def fake_generate(**kwargs):
            seen.setdefault("stop_strings", tuple(kwargs.get("stop_strings") or ()))
            seen.setdefault("stop_token_ids", tuple(kwargs.get("stop_token_ids") or ()))
            inp = kwargs["input_ids"]
            return torch.cat([inp, torch.tensor([[close_id]]).expand(inp.size(0), -1).to(inp.device)], dim=1)

        script_session_generate(monkeypatch, fake_generate)
        prompt = tokenizer("R", return_tensors="pt").input_ids
        pipeline.generate(input_ids=prompt, max_new_tokens=8, do_sample=False, return_full_sequence=True)
        assert "stop" in seen["stop_strings"]
        assert close_id in seen["stop_token_ids"]


class TestPlanValidation:
    def test_parses_until_token_ids(self):
        phase = _parse_phase({"generate": {"until_token_ids": [3, 9]}})
        assert isinstance(phase, Generated)
        assert phase.until_token_ids == (3, 9)

    def test_rejects_non_integer_entries(self):
        with pytest.raises(ValueError, match="until_token_ids must contain only ints"):
            _parse_phase({"generate": {"until_token_ids": [3, "x"]}})

    def test_rejects_non_sequence(self):
        with pytest.raises(ValueError, match="until_token_ids must be a sequence"):
            _parse_phase({"generate": {"until_token_ids": 5}})

    def test_empty_default_is_valid(self):
        assert _parse_phase({"generate": {}}).until_token_ids == ()


class TestBudgetForcingTokenBoundary:
    def test_end_think_token_ids_end_the_thinking_phases(self):
        bf = BudgetForcing(max_thinking_tokens=8, num_extensions=1, end_think="</think>",
                           end_think_token_ids=(42,))
        plan = bf.plan("prompt", {})
        thinking_phases = [p for p in plan if isinstance(p, Generated) and p.budget == 8]
        assert thinking_phases  # the initial and each extension thinking phase
        for phase in thinking_phases:
            assert phase.until == "</think>"
            assert phase.until_token_ids == (42,)
        # the answer phase carries no boundary
        assert plan[-1].until is None and plan[-1].until_token_ids == () and plan[-1].budget is None

    def test_args_reject_string_token_ids(self):
        with pytest.raises(ValueError, match="end_think_token_ids"):
            BudgetForcing(max_thinking_tokens=8, end_think_token_ids="42")
