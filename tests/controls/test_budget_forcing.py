"""Behavior tests for BudgetForcing (output multiplicity design, P4).

Hub-free: phase generation is scripted through the session so phase splicing, the forced
closing tag, and extension rounds are asserted deterministically.
"""
import pytest
import torch

from steerability.algorithms.core.steering_pipeline import SteeringPipeline
from steerability.algorithms.output_control.budget_forcing.control import BudgetForcing
from steerability.algorithms.output_control.common.drivers.phased import Fixed, Generated
from tests.utils.runtime_helpers import script_session_generate
from tests.utils.tiny_models import tiny_llama, wordlevel_tokenizer

VOCAB = 100


def _pipeline(controls, model=None, tokenizer=None):
    if model is None:
        model = tiny_llama(num_layers=2, hidden=16, heads=2, vocab=VOCAB)
    if tokenizer is None:
        tokenizer = wordlevel_tokenizer()
    pipeline = SteeringPipeline(controls=controls, model=model, tokenizer=tokenizer)
    pipeline.steer()
    return pipeline, model, tokenizer


class TestPlan:
    def test_plan_structure_no_extensions(self):
        bf = BudgetForcing(max_thinking_tokens=16, num_extensions=0, end_think="</think>")
        plan = bf.plan("prompt", {})
        # thinking, forced tag, answer
        assert len(plan) == 3
        assert isinstance(plan[0], Generated) and plan[0].until == "</think>" and plan[0].budget == 16
        assert isinstance(plan[1], Fixed) and plan[1].text == "</think>" and plan[1].replace is False
        assert isinstance(plan[2], Generated) and plan[2].until is None and plan[2].budget is None

    def test_plan_structure_with_extensions(self):
        bf = BudgetForcing(max_thinking_tokens=8, extension_text="Wait", num_extensions=2)
        plan = bf.plan("prompt", {})
        # thinking + 2 * (Fixed + Generated) + forced tag + answer = 1 + 4 + 1 + 1 = 7
        assert len(plan) == 7
        assert isinstance(plan[1], Fixed) and plan[1].text == "Wait"
        assert isinstance(plan[2], Generated) and plan[2].budget == 8
        assert isinstance(plan[3], Fixed) and plan[3].text == "Wait"
        assert isinstance(plan[5], Fixed) and plan[5].text == "</think>"

    def test_extension_fixed_phases_are_plain_appends(self):
        bf = BudgetForcing(max_thinking_tokens=8, num_extensions=1)
        plan = bf.plan("prompt", {})
        for phase in plan:
            if isinstance(phase, Fixed):
                assert phase.replace is False
                assert phase.add_special_tokens is False


class TestConfig:
    def test_is_decoding_driver(self):
        from steerability.algorithms.output_control.base import DecodingDriver
        assert isinstance(BudgetForcing(max_thinking_tokens=8), DecodingDriver)

    def test_no_extract_rule(self):
        bf = BudgetForcing(max_thinking_tokens=8)
        assert bf.extract_after is None

    def test_rejects_bad_args(self):
        with pytest.raises(ValueError):
            BudgetForcing(max_thinking_tokens=0)
        with pytest.raises(ValueError):
            BudgetForcing(max_thinking_tokens=8, num_extensions=-1)
        with pytest.raises(ValueError):
            BudgetForcing(max_thinking_tokens=8, end_think="")


class TestEndToEnd:
    def test_forces_closing_tag_and_answer(self, monkeypatch):
        # the wordlevel test tokenizer maps out-of-vocab words to <pad>, so use an in-vocab marker
        # ("span") to make the forced closing tag observable in the decoded stream.
        model = tiny_llama(num_layers=2, hidden=16, heads=2, vocab=VOCAB)
        tokenizer = wordlevel_tokenizer()
        bf = BudgetForcing(max_thinking_tokens=4, num_extensions=0, end_think="span")
        pipeline, model, tokenizer = _pipeline([bf], model=model, tokenizer=tokenizer)

        # each phase's generate appends tokens; the thinking phase never emits the marker on its own,
        # so the forced Fixed("span") is what introduces it.
        def fake_generate(**kwargs):
            inp = kwargs["input_ids"]
            cont = tokenizer("cat mat", return_tensors="pt", add_special_tokens=False).input_ids
            return torch.cat([inp, cont.expand(inp.size(0), -1).to(inp.device)], dim=1)

        script_session_generate(monkeypatch, fake_generate)
        prompt = tokenizer("the dog", return_tensors="pt").input_ids
        out = pipeline.generate(
            input_ids=prompt,
            runtime_kwargs={},
            return_full_sequence=True,
        )
        decoded = tokenizer.decode(out[0], skip_special_tokens=False)
        # the forced closing marker is present (spliced by the Fixed phase)
        assert "span" in decoded

    def test_extension_text_spliced_between_thinking_segments(self, monkeypatch):
        model = tiny_llama(num_layers=2, hidden=16, heads=2, vocab=VOCAB)
        tokenizer = wordlevel_tokenizer()
        bf = BudgetForcing(max_thinking_tokens=3, extension_text="on", num_extensions=1, end_think="span")
        pipeline, model, tokenizer = _pipeline([bf], model=model, tokenizer=tokenizer)

        def fake_generate(**kwargs):
            inp = kwargs["input_ids"]
            cont = tokenizer("cat", return_tensors="pt", add_special_tokens=False).input_ids
            return torch.cat([inp, cont.expand(inp.size(0), -1).to(inp.device)], dim=1)

        script_session_generate(monkeypatch, fake_generate)
        prompt = tokenizer("the dog", return_tensors="pt").input_ids
        out = pipeline.generate(
            input_ids=prompt,
            runtime_kwargs={},
            return_full_sequence=True,
        )
        decoded = tokenizer.decode(out[0], skip_special_tokens=False)
        # the extension text "on" and the forced marker both appear
        assert "on" in decoded
        assert "span" in decoded

    def test_folded_stacks_reach_every_generated_phase(self, monkeypatch):
        model = tiny_llama(num_layers=2, hidden=16, heads=2, vocab=VOCAB)
        tokenizer = wordlevel_tokenizer()

        from steerability.algorithms.output_control.base import OutputControl

        class _ForceToken(OutputControl):
            Args = None

            def get_logits_processors(self, input_ids, runtime_kwargs, **kwargs):
                def _force(prefix_ids, scores):
                    out = torch.full_like(scores, float("-inf"))
                    out[:, 7] = 0.0
                    return out
                return [_force]

        saw_processor = []

        def fake_generate(**kwargs):
            saw_processor.append("logits_processor" in kwargs)
            inp = kwargs["input_ids"]
            cont = tokenizer("cat", return_tensors="pt", add_special_tokens=False).input_ids
            return torch.cat([inp, cont.expand(inp.size(0), -1).to(inp.device)], dim=1)

        bf = BudgetForcing(max_thinking_tokens=3, num_extensions=1, end_think="</think>")
        pipeline, model, tokenizer = _pipeline([_ForceToken(), bf], model=model, tokenizer=tokenizer)
        script_session_generate(monkeypatch, fake_generate)
        prompt = tokenizer("the dog", return_tensors="pt").input_ids
        pipeline.generate(input_ids=prompt, runtime_kwargs={})
        # 3 Generated phases (thinking, 1 extension, answer); each received the composed stack
        assert len(saw_processor) == 3
        assert all(saw_processor)

    def test_registered_in_registry(self):
        import steerability.algorithms.core.registry as r
        assert "budget_forcing" in r.REGISTRY["output_control"]
