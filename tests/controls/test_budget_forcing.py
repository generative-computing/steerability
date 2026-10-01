"""Behavior tests for BudgetForcing (output multiplicity design, P4).

Hub-free: phase generation is scripted through the session so phase splicing, the forced
closing tag, and extension rounds are asserted deterministically.
"""
import pytest
import torch

from steerability.algorithms.core.steering_pipeline import SteeringPipeline
from steerability.algorithms.output_control.budget_forcing.control import BudgetForcing
from steerability.algorithms.output_control.common.drivers.phased import Fixed, Generated, PlanState
from tests.utils.runtime_helpers import script_session_generate
from tests.utils.tiny_models import reasoning_tag_tokenizer, tiny_llama, wordlevel_tokenizer

VOCAB = 100


def _state(last_stop, text=""):
    return PlanState(last_stop=last_stop, appended=0, text=text)


def _budget_filling_generate(tokenizer, word):
    """A scripted generate that emits `word` until the call's `max_new_tokens` is reached."""
    word_id = tokenizer(word, add_special_tokens=False).input_ids[0]

    def fake_generate(**kwargs):
        inp = kwargs["input_ids"]
        count = kwargs.get("max_new_tokens") or 1
        cont = torch.full((inp.size(0), count), word_id, dtype=inp.dtype, device=inp.device)
        return torch.cat([inp, cont], dim=1)

    return fake_generate


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
        # thinking, closing tag, answer
        assert len(plan) == 3
        assert isinstance(plan[0], Generated) and plan[0].until == "</think>" and plan[0].budget == 16
        closing = plan[1](_state("budget"))
        assert isinstance(closing, Fixed) and closing.text == "</think>" and closing.replace is False
        assert isinstance(plan[2], Generated) and plan[2].until is None and plan[2].budget is None

    def test_plan_structure_with_extensions(self):
        bf = BudgetForcing(max_thinking_tokens=8, extension_text="Wait", num_extensions=2)
        plan = bf.plan("prompt", {})
        # thinking + 2 * (extension + thinking) + closing tag + answer = 1 + 4 + 1 + 1 = 7
        assert len(plan) == 7
        cut_off = _state("budget")
        assert isinstance(plan[1](cut_off), Fixed) and plan[1](cut_off).text == "Wait"
        assert isinstance(plan[2](cut_off), Generated) and plan[2](cut_off).budget == 8
        assert isinstance(plan[3](cut_off), Fixed) and plan[3](cut_off).text == "Wait"
        assert plan[5](cut_off).text == "</think>"

    @pytest.mark.parametrize("last_stop", ["until", "until_token", "eos", "length", None])
    def test_extension_rounds_run_only_after_a_budget_stop(self, last_stop):
        bf = BudgetForcing(max_thinking_tokens=8, num_extensions=1)
        plan = bf.plan("prompt", {})
        assert plan[1](_state(last_stop)) is None
        assert plan[2](_state(last_stop)) is None

    @pytest.mark.parametrize("last_stop", ["until", "until_token"])
    def test_no_closing_tag_after_the_model_closed_its_thinking(self, last_stop):
        bf = BudgetForcing(max_thinking_tokens=8, end_think="</think>")
        plan = bf.plan("prompt", {})
        assert plan[1](_state(last_stop, text="thought </think>")) is None

    def test_no_closing_tag_when_the_stream_ends_with_it(self):
        bf = BudgetForcing(max_thinking_tokens=8, end_think="</think>")
        assert bf.plan("prompt", {})[1](_state("eos", text="thought </think>")) is None

    def test_extension_fixed_phases_are_plain_appends(self):
        bf = BudgetForcing(max_thinking_tokens=8, num_extensions=1)
        plan = bf.plan("prompt", {})
        for entry in plan:
            phase = entry(_state("budget")) if callable(entry) else entry
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

        # every thinking segment reaches its budget, so the extension round runs
        script_session_generate(monkeypatch, _budget_filling_generate(tokenizer, "cat"))
        prompt = tokenizer("the dog", return_tensors="pt").input_ids
        out = pipeline.generate(
            input_ids=prompt,
            runtime_kwargs={},
            max_new_tokens=10,
        )
        decoded = tokenizer.decode(out[0], skip_special_tokens=False)
        # three thinking tokens, the extension, three thinking tokens, the forced marker, the answer
        assert decoded.split() == ["cat"] * 3 + ["on"] + ["cat"] * 3 + ["span", "cat", "cat"]

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
        fill_budget = _budget_filling_generate(tokenizer, "cat")

        def fake_generate(**kwargs):
            saw_processor.append("logits_processor" in kwargs)
            return fill_budget(**kwargs)

        bf = BudgetForcing(max_thinking_tokens=3, num_extensions=1, end_think="</think>")
        pipeline, model, tokenizer = _pipeline([_ForceToken(), bf], model=model, tokenizer=tokenizer)
        script_session_generate(monkeypatch, fake_generate)
        prompt = tokenizer("the dog", return_tensors="pt").input_ids
        pipeline.generate(input_ids=prompt, runtime_kwargs={}, max_new_tokens=12)
        # 3 Generated phases (thinking, 1 extension, answer); each received the composed stack
        assert len(saw_processor) == 3
        assert all(saw_processor)

    def test_registered_in_registry(self):
        import steerability.algorithms.core.registry as r
        assert "budget_forcing" in r.REGISTRY["output_control"]


class TestClosingTag:
    """A model that closes its own thinking receives no second closing tag and no extension."""

    # "x" takes id 2, the tiny Llama's eos id, and appears only in the prompt
    WORDS = ("x", "thought", "A", "Wait")

    @staticmethod
    def _closing_generate(tokenizer):
        # a thinking phase (stop string "</think>") emits "thought </think>"; the answer emits "A A"
        def fake_generate(**kwargs):
            inp = kwargs["input_ids"]
            text = "thought </think>" if "</think>" in (kwargs.get("stop_strings") or ()) else "A A"
            cont = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids
            return torch.cat([inp, cont.expand(inp.size(0), -1).to(inp.device)], dim=1)

        return fake_generate

    def _run(self, monkeypatch, fake_generate_factory, **budget_forcing_kwargs):
        tokenizer = reasoning_tag_tokenizer(ordinary_tags=("</think>",), words=self.WORDS)
        bf = BudgetForcing(end_think="</think>", extension_text="Wait", **budget_forcing_kwargs)
        pipeline, _, tokenizer = _pipeline([bf], tokenizer=tokenizer)
        script_session_generate(monkeypatch, fake_generate_factory(tokenizer))
        prompt = tokenizer("x", return_tensors="pt", add_special_tokens=False).input_ids
        out = pipeline.generate(input_ids=prompt, max_new_tokens=16)
        return tokenizer.decode(out[0], skip_special_tokens=False).split()

    @pytest.mark.parametrize("num_extensions", [0, 2])
    def test_model_closing_its_thinking_gets_one_tag(self, monkeypatch, num_extensions):
        tokens = self._run(
            monkeypatch, self._closing_generate, max_thinking_tokens=4, num_extensions=num_extensions,
        )
        assert tokens == ["thought", "</think>", "A", "A"]

    def test_budget_exhausted_thinking_still_extends(self, monkeypatch):
        tokens = self._run(
            monkeypatch, lambda tokenizer: _budget_filling_generate(tokenizer, "thought"),
            max_thinking_tokens=2, num_extensions=1,
        )
        assert tokens[:7] == ["thought", "thought", "Wait", "thought", "thought", "</think>", "thought"]
        assert tokens.count("</think>") == 1
