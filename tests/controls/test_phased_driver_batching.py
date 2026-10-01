"""`PhasedDriver` under batches and multiple candidates.

Covers pad handling on a left-padded batch, `num_return_sequences > 1`, the row order of the
returned candidates, the `max_new_tokens` ceiling across phases, state controls under several
candidates, `RoutedDecoding` through the shared driver, and `session_generate_items`.

Hub-free: a tiny randomly initialized Llama with a WordLevel tokenizer that pads on the left. Unless
a test says otherwise, every generate call disables the model's EOS and suppresses the pad token, so
each phase generates exactly its ceiling and continuation lengths are deterministic.
"""
import warnings

import pytest
import torch

from steerability.algorithms.core.execution.payloads import ItemResult
from steerability.algorithms.core.execution.session_utils import session_generate_items
from steerability.algorithms.core.internals.probes import Probe, ProbeSet
from steerability.algorithms.core.output import Output
from steerability.algorithms.core.steering_pipeline import SteeringPipeline
from steerability.algorithms.output_control.base import OutputControl
from steerability.algorithms.output_control.common.drivers.phased import Fixed, Generated
from steerability.algorithms.output_control.phased_decoding.control import PhasedDecoding
from steerability.algorithms.output_control.routed_decoding import (
    P,
    Route,
    RoutedDecoding,
    Router,
    generate,
    prefix,
    respond,
)
from steerability.algorithms.state_control.caa.control import CAA
from steerability.algorithms.state_control.common.steering_vector import SteeringVector
from steerability.algorithms.state_control.iti.control import ITI
from steerability.backends.huggingface import ExclusiveSession
from tests.utils.runtime_helpers import RecordingTransform
from tests.utils.tiny_models import tiny_llama, wordlevel_tokenizer

HIDDEN = 16
CHAT_TEMPLATE = (
    "{{ bos_token }}"
    "{% for message in messages %}{{ message['content'] }} {% endfor %}"
    "{% if add_generation_prompt %}sat {% endif %}"
)
LONG_CONVERSATION = [{"role": "user", "content": "the cat sat on the mat and the dog ran"}]
SHORT_CONVERSATION = [{"role": "user", "content": "dog ran"}]
PAD_TOKEN_ID = 2  # the WordLevel tokenizer's <pad>, which is also the tiny Llama's default eos id
FIXED_LENGTH = {"eos_token_id": None, "suppress_tokens": [PAD_TOKEN_ID]}


def _tokenizer():
    tokenizer = wordlevel_tokenizer()
    tokenizer.chat_template = CHAT_TEMPLATE
    tokenizer.padding_side = "left"
    return tokenizer


def _pipeline(controls, seed: int = 0):
    torch.manual_seed(seed)
    model = tiny_llama(num_layers=2, hidden=HIDDEN, heads=2)
    tokenizer = _tokenizer()
    pipeline = SteeringPipeline(controls=controls, model=model, tokenizer=tokenizer)
    pipeline.steer()
    return pipeline, tokenizer


def _ids(tokenizer, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def _strip_pads(row: torch.Tensor, pad_token_id: int) -> list[int]:
    values = row.tolist()
    while values and values[-1] == pad_token_id:
        values.pop()
    return values


def _count_session_calls(monkeypatch) -> list[int]:
    """Record the item count of every session generate call, delegating to the real session."""
    calls: list[int] = []
    original = ExclusiveSession.generate

    def counting(self, items, params):
        calls.append(len(items))
        return original(self, items, params)

    monkeypatch.setattr(ExclusiveSession, "generate", counting)
    return calls


class TestPaddedBatch:
    def test_shorter_prompt_matches_its_single_prompt_run(self):
        # seed 3 gives weights under which attending to the pad positions changes the greedy tokens
        pipeline, tokenizer = _pipeline([
            PhasedDecoding(plan=[{"generate": {"budget": 3}}, {"fixed": "on"}, {"generate": {"budget": 3}}]),
        ], seed=3)
        long_ids = tokenizer("the cat sat on the mat and the dog ran fast")["input_ids"]
        short_ids = tokenizer("dog ran")["input_ids"]
        pad = tokenizer.pad_token_id
        width = len(long_ids)
        batch = torch.tensor([long_ids, [pad] * (width - len(short_ids)) + short_ids])
        mask = torch.tensor([[1] * width, [0] * (width - len(short_ids)) + [1] * len(short_ids)])

        batched = pipeline.generate(
            input_ids=batch, attention_mask=mask, max_new_tokens=16, do_sample=False, **FIXED_LENGTH,
        )
        alone = pipeline.generate(
            input_ids=torch.tensor([short_ids]), max_new_tokens=16, do_sample=False, **FIXED_LENGTH,
        )
        assert _strip_pads(batched[1], pad) == _strip_pads(alone[0], pad)


class TestCandidates:
    def test_three_candidates_for_one_prompt(self):
        pipeline, tokenizer = _pipeline([
            PhasedDecoding(plan=[{"generate": {"budget": 2}}, {"fixed": "on"}, {"generate": {"budget": 2}}]),
        ])
        output = pipeline.generate(
            messages=LONG_CONVERSATION, n=3, return_output=True, max_new_tokens=16,
            do_sample=True, seed=0, **FIXED_LENGTH,
        )
        assert output.output_ids.shape == (3, 5)
        assert len(output.finish_reasons) == 3
        on_id = _ids(tokenizer, "on")[0]
        for row in output.output_ids:
            assert row[2].item() == on_id  # two generated tokens, the fixed text, two generated tokens

    def test_batch_of_two_is_row_major_candidate_minor(self):
        def marker(prompt_text, params):
            return "mat" if "cat" in prompt_text else "fast"

        pipeline, tokenizer = _pipeline([
            PhasedDecoding(plan=[{"fixed": marker}, {"generate": {"budget": 2}}]),
        ])
        outputs = pipeline.generate(
            messages=[LONG_CONVERSATION, SHORT_CONVERSATION], n=2, return_output=True,
            max_new_tokens=16, do_sample=True, seed=0, **FIXED_LENGTH,
        )
        assert len(outputs) == 4
        mat_id, fast_id = _ids(tokenizer, "mat")[0], _ids(tokenizer, "fast")[0]
        cat_id = _ids(tokenizer, "cat")[0]
        for index, output in enumerate(outputs):
            # candidates 0 and 1 belong to the first prompt, 2 and 3 to the second
            assert (cat_id in output.adapted_input_ids[0].tolist()) == (index < 2)
            assert output.output_ids[0, 0].item() == (mat_id if index < 2 else fast_id)
            assert len(_strip_pads(output.output_ids[0], PAD_TOKEN_ID)) == 3


class TestCeiling:
    def test_first_phase_exhausting_the_ceiling_skips_the_rest(self, monkeypatch):
        pipeline, tokenizer = _pipeline([
            PhasedDecoding(plan=[{"generate": {}}, {"fixed": "on"}, {"generate": {}}]),
        ])
        calls = _count_session_calls(monkeypatch)
        output = pipeline.generate(
            messages=SHORT_CONVERSATION, return_output=True, max_new_tokens=4, do_sample=False, **FIXED_LENGTH,
        )
        assert output.output_ids.size(1) == 4
        assert output.finish_reasons[0] == "length"
        assert len(calls) == 1  # the second generated phase issued no session call

    def test_ceiling_bounds_the_total_continuation(self):
        pipeline, _ = _pipeline([
            PhasedDecoding(plan=[{"generate": {"budget": 3}}, {"fixed": "on"}, {"generate": {}}]),
        ])
        output = pipeline.generate(
            messages=SHORT_CONVERSATION, return_output=True, max_new_tokens=6, do_sample=False, **FIXED_LENGTH,
        )
        assert output.output_ids.size(1) == 6  # 3 generated, 1 fixed, 2 generated under the ceiling
        assert output.finish_reasons[0] == "length"

    def test_plan_completing_under_the_ceiling(self):
        pipeline, _ = _pipeline([
            PhasedDecoding(plan=[{"generate": {"budget": 2}}, {"fixed": "on"}, {"generate": {"budget": 2}}]),
        ])
        output = pipeline.generate(
            messages=SHORT_CONVERSATION, return_output=True, max_new_tokens=10, do_sample=False, **FIXED_LENGTH,
        )
        assert output.output_ids.size(1) == 5
        assert output.finish_reasons[0] != "length"

    def test_fixed_phase_crossing_the_ceiling_is_appended_whole(self, monkeypatch):
        pipeline, tokenizer = _pipeline([
            PhasedDecoding(plan=[{"fixed": "the cat sat on"}, {"generate": {}}]),
        ])
        calls = _count_session_calls(monkeypatch)
        output = pipeline.generate(
            messages=SHORT_CONVERSATION, return_output=True, max_new_tokens=2, do_sample=False, **FIXED_LENGTH,
        )
        assert output.output_ids[0].tolist() == _ids(tokenizer, "the cat sat on")
        assert output.finish_reasons[0] == "length"
        assert calls == []  # the ceiling is reached, so the generated phase is skipped

    def test_row_at_the_ceiling_is_length_when_pad_equals_eos(self):
        # the first row's fixed text crosses the ceiling, so the second row, cut at the ceiling,
        # is right-padded with the pad token that is also the eos
        def fixed_text(prompt_text, params):
            return "the cat sat on the mat and the dog ran fast" if "cat" in prompt_text else "on"

        torch.manual_seed(0)
        model = tiny_llama(num_layers=2, hidden=HIDDEN, heads=2)
        tokenizer = _tokenizer()
        tokenizer.pad_token = tokenizer.eos_token
        pipeline = SteeringPipeline(
            controls=[PhasedDecoding(plan=[{"fixed": fixed_text}, {"generate": {}}])], model=model, tokenizer=tokenizer,
        )
        pipeline.steer()
        long_row, short_row = pipeline.generate(
            messages=[LONG_CONVERSATION, SHORT_CONVERSATION], return_output=True, max_new_tokens=4,
            do_sample=False, eos_token_id=None, suppress_tokens=[tokenizer.eos_token_id, PAD_TOKEN_ID],
        )
        assert long_row.output_ids.size(1) > 4
        assert len(_strip_pads(short_row.output_ids[0], tokenizer.pad_token_id)) == 4
        assert long_row.finish_reasons[0] == "length"
        assert short_row.finish_reasons[0] == "length"

    def test_beam_search_with_several_candidates_raises(self):
        pipeline, _ = _pipeline([PhasedDecoding(plan=[{"generate": {"budget": 2}}])])
        with pytest.raises(ValueError, match="beam search"):
            pipeline.generate(
                messages=SHORT_CONVERSATION, n=2, return_output=True, max_new_tokens=4, num_beams=2,
                **FIXED_LENGTH,
            )


class _ForceFirstCandidate(OutputControl):
    """Force `token_id` as the second generated token of the first row of a two-row call over the prompt.

    Every other position of every call has `token_id` and each id in `blocked` masked out, so the
    other candidate never stops early.
    """

    Args = None

    def __init__(self, prompt_length: int, token_id: int, blocked: tuple[int, ...] = ()):
        self._prompt_length = prompt_length
        self._token_id = token_id
        self._blocked = (token_id, *blocked)

    def get_logits_processors(self, input_ids, runtime_kwargs, **kwargs):
        def _force(prefix_ids, scores):
            scores = scores.clone()
            scores[:, list(self._blocked)] = float("-inf")
            if prefix_ids.size(0) == 2 and prefix_ids.size(1) == self._prompt_length + 1:
                scores[0] = float("-inf")
                scores[0, self._token_id] = 0.0
            return scores

        return [_force]


def _steering_vector() -> SteeringVector:
    generator = torch.Generator().manual_seed(5)
    return SteeringVector(model_type="llama", directions={1: torch.randn(1, HIDDEN, generator=generator)})


def _iti() -> ITI:
    """ITI on both heads of layer 1, steering the generated positions only."""
    generator = torch.Generator().manual_seed(7)
    vector = SteeringVector(
        model_type="llama", directions={1: 3 * torch.randn(2, HIDDEN // 2, generator=generator)},
        num_heads=2, head_dim=HIDDEN // 2,
    )
    return ITI(steering_vector=vector, selected_heads=[(1, 0), (1, 1)], alpha=20.0, token_scope="after_prompt")


class _HeadSiteRecordingTransform(RecordingTransform):
    """A `RecordingTransform` hooked at the attention output projection, like a head-additive transform."""

    wire_kind = "head_additive"


def _record_session_calls(monkeypatch, recorder: RecordingTransform) -> list[tuple[list[int], list[torch.Tensor]]]:
    """Record each session generate call's item lengths and the token masks recorded during the call."""
    calls: list[tuple[list[int], list[torch.Tensor]]] = []
    original = ExclusiveSession.generate

    def recording(self, items, params):
        start = len(recorder.masks)
        results = original(self, items, params)
        calls.append(([item.prompt.token_ids.size(-1) for item in items], recorder.masks[start:]))
        return results

    monkeypatch.setattr(ExclusiveSession, "generate", recording)
    return calls


class TestStateControls:
    def test_candidates_at_different_lengths_keep_their_prompts_unsteered(self, monkeypatch):
        # the first candidate stops on eos after two tokens, so the candidates reach the last
        # phase with different appended counts
        prompt = torch.arange(3, 10).unsqueeze(0)
        tokenizer = _tokenizer()
        control = CAA(steering_vector=_steering_vector(), layer_id=1, multiplier=1.0, token_scope="after_prompt")
        pipeline, _ = _pipeline([
            control,
            _ForceFirstCandidate(prompt.size(1), tokenizer.eos_token_id),
            PhasedDecoding(plan=[{"generate": {"budget": 4}}, {"fixed": "on"}, {"generate": {"budget": 3}}]),
        ])
        recorder = RecordingTransform(value=0.0)
        control._transform = recorder
        calls = _record_session_calls(monkeypatch, recorder)
        pipeline.generate(
            input_ids=prompt, n=2, max_new_tokens=16, do_sample=False,
            eos_token_id=tokenizer.eos_token_id, suppress_tokens=[PAD_TOKEN_ID],
        )

        prefills = 0
        for lengths, masks in calls:
            width = max(lengths)
            for mask in (mask for mask in masks if mask.size(1) == width):
                prefills += 1
                for row, length in enumerate(lengths):
                    # the call left-packs its items, so this item's prompt ends here
                    prompt_end = width - length + prompt.size(1)
                    assert not mask[row, :prompt_end].any()
        assert prefills >= 2  # the last phase re-prefills each candidate's stream

    @pytest.mark.parametrize(
        "plan",
        [[{"generate": {}}], [{"generate": {"budget": 3}}, {"fixed": "on"}, {"generate": {}}]],
        ids=["one-phase", "three-phase"],
    )
    def test_counting_fallback_candidates_match_one_candidate_under_item_seeds(self, plan):
        # ITI hooks the attention output projection, which receives no pass positions, so its
        # hooks count passes; under per-item seeds the session decodes the candidates one at a time
        pipeline, _ = _pipeline([_iti(), PhasedDecoding(plan=plan)])
        generate_kwargs = dict(
            messages=SHORT_CONVERSATION, return_output=True, max_new_tokens=6, do_sample=False, **FIXED_LENGTH,
        )
        single = pipeline.generate(n=1, **generate_kwargs)
        several = pipeline.generate(n=2, seed=1, seed_scope="item", **generate_kwargs)
        assert [row.tolist() for row in several.output_ids] == [single.output_ids[0].tolist()] * 2

    def test_counting_fallback_positions_restart_at_each_session_call(self, monkeypatch):
        prompt = torch.arange(3, 10).unsqueeze(0)
        control = _iti()
        pipeline, _ = _pipeline([
            control, PhasedDecoding(plan=[{"generate": {"budget": 3}}, {"fixed": "on"}, {"generate": {"budget": 3}}]),
        ])
        recorder = _HeadSiteRecordingTransform(value=0.0)
        control._transform = recorder
        calls = _record_session_calls(monkeypatch, recorder)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pipeline.generate(input_ids=prompt, max_new_tokens=16, do_sample=False, **FIXED_LENGTH)

        assert not [warning for warning in caught if "Multiple generate calls" in str(warning.message)]
        assert len(calls) == 2
        (length,), masks = calls[1]
        prefill = [mask for mask in masks if mask.size(1) == length]
        # the last phase re-prefills the prompt, three generated tokens, and the fixed text
        assert len(prefill) == 1
        assert prefill[0][0].tolist() == [False] * prompt.size(1) + [True] * (length - prompt.size(1))

    def test_batch_of_prompts_with_state_controls_is_refused(self):
        control = CAA(steering_vector=_steering_vector(), layer_id=1, multiplier=1.0, token_scope="after_prompt")
        pipeline, _ = _pipeline([control, PhasedDecoding(plan=[{"generate": {"budget": 2}}])])
        assert not pipeline.supports_batching
        with pytest.raises(ValueError, match="one prompt per call"):
            pipeline.generate(
                messages=[LONG_CONVERSATION, SHORT_CONVERSATION], return_output=True, max_new_tokens=4,
                **FIXED_LENGTH,
            )
        output = pipeline.generate(
            messages=SHORT_CONVERSATION, n=2, return_output=True, max_new_tokens=4, **FIXED_LENGTH,
        )
        assert output.output_ids.shape == (2, 2)

    def test_batching_is_supported_without_state_controls(self):
        pipeline, _ = _pipeline([PhasedDecoding(plan=[{"generate": {"budget": 2}}])])
        assert pipeline.supports_batching


class TestTerminalTokens:
    def test_pad_equal_eos_is_not_appended_after_another_eos(self):
        # pad and the tokenizer's eos share an id, and the model also stops on the id 10
        prompt = torch.arange(3, 10).unsqueeze(0)
        tokenizer = _tokenizer()
        tokenizer.pad_token = tokenizer.eos_token
        eos = tokenizer.eos_token_id
        torch.manual_seed(0)
        model = tiny_llama(num_layers=2, hidden=HIDDEN, heads=2)
        model.generation_config.eos_token_id = [eos, 10]
        pipeline = SteeringPipeline(
            controls=[
                _ForceFirstCandidate(prompt.size(1), 10, blocked=(eos,)),
                PhasedDecoding(plan=[{"generate": {"budget": 4}}, {"fixed": "on"}, {"generate": {"budget": 1}}]),
            ],
            model=model, tokenizer=tokenizer,
        )
        pipeline.steer()
        first, second = pipeline.generate(input_ids=prompt, n=2, max_new_tokens=16, do_sample=False, return_output=True)

        on_id = _ids(tokenizer, "on")[0]
        assert first.output_ids[0, 1].item() == 10
        assert first.output_ids[0, 2].item() == on_id  # the fixed text follows the emitted eos directly
        assert second.output_ids[0, 4].item() == on_id


def _always_probe():
    generator = torch.Generator().manual_seed(200)
    weight = torch.randn(HIDDEN, generator=generator)
    return ProbeSet({
        "always": Probe(
            model_type="llama", location="layer_input", pooling="mean", layer_ids=[1],
            weights={1: weight / weight.norm()}, bias=1e9, meta={},
        ),
    })


class TestRoutedDecoding:
    def test_candidates_through_the_shared_driver(self):
        router = RoutedDecoding(
            probes=_always_probe(),
            rules=Router(routes=[Route("note", when=P("always"), action=prefix("the"))], default_action=generate()),
        )
        pipeline, tokenizer = _pipeline([router])
        outputs = pipeline.generate(
            messages=[LONG_CONVERSATION, SHORT_CONVERSATION], n=2, return_output=True,
            max_new_tokens=3, do_sample=True, seed=0, **FIXED_LENGTH,
        )
        assert len(outputs) == 4
        assert router.latest_routes == ["note", "note"]
        the_id = _ids(tokenizer, "the")[0]
        for output in outputs:
            assert output.output_ids.size(1) == 3
            assert output.output_ids[0, 0].item() == the_id
            assert output.finish_reasons[0] == "length"

    def test_canned_response_longer_than_the_ceiling_is_returned_whole(self, monkeypatch):
        text = "the cat sat on the mat and the dog ran fast"
        router = RoutedDecoding(
            probes=_always_probe(),
            rules=Router(routes=[Route("canned", when=P("always"), action=respond(text))]),
        )
        pipeline, tokenizer = _pipeline([router])
        calls = _count_session_calls(monkeypatch)
        output = pipeline.generate(
            messages=SHORT_CONVERSATION, return_output=True, max_new_tokens=4, **FIXED_LENGTH,
        )
        assert output.output_ids[0].tolist() == _ids(tokenizer, text)
        assert output.finish_reasons[0] == "length"
        assert calls == []

    def test_max_rollouts_counts_the_longest_plan(self):
        two_phase = [Generated(budget=2), Fixed("on"), Generated()]
        rules = Router(
            routes=[
                Route("canned", when=P("always"), action=respond("the mat")),
                Route("raw", when=P("always"), action=two_phase),
            ],
            default_action=generate(),
        )
        assert RoutedDecoding(probes=_always_probe(), rules=rules).max_rollouts_per_query() == 2
        canned_only = Router(routes=[Route("canned", when=P("always"), action=respond("the mat"))])
        assert RoutedDecoding(probes=_always_probe(), rules=canned_only).max_rollouts_per_query() == 1


class _PaddingSession:
    """A session double whose results are right-padded to a common length, as a batched session returns."""

    def __init__(self, tokenizer, continuations, reasons):
        self.tokenizer = tokenizer
        self._continuations = continuations
        self._reasons = reasons
        self.seen = []

    def generate(self, items, params):
        self.seen.append((items, params))
        width = max(len(ids) for ids in self._continuations)
        pad = self.tokenizer.pad_token_id
        results = []
        for index, (ids, reason) in enumerate(zip(self._continuations, self._reasons)):
            row = torch.tensor([ids + [pad] * (width - len(ids))])
            results.append(ItemResult(index=index, output=Output(
                output_ids=row, adapted_input_ids=items[index].prompt.token_ids,
                finish_reasons=(reason,),
            )))
        return results


class TestSessionGenerateItems:
    def test_one_unpadded_item_per_row_with_one_candidate(self):
        tokenizer = wordlevel_tokenizer()
        session = _PaddingSession(tokenizer, [[5, 6], [7]], ["length", None])
        rows = [torch.tensor([0, 3, 4]), torch.tensor([0, 8])]
        results = session_generate_items(session, rows, max_new_tokens=2, num_return_sequences=4)

        items, params = session.seen[0]
        assert params.n == 1
        assert params.max_new_tokens == 2
        assert [item.prompt.token_ids[0].tolist() for item in items] == [[0, 3, 4], [0, 8]]
        assert all(bool(item.prompt.attention_mask.all()) for item in items)
        assert [(ids.tolist(), reason) for ids, reason in results] == [([5, 6], "length"), ([7], None)]

    def test_pad_equal_eos_is_kept_on_an_eos_finish(self):
        tokenizer = wordlevel_tokenizer()
        tokenizer.pad_token = tokenizer.eos_token
        eos = tokenizer.eos_token_id
        session = _PaddingSession(tokenizer, [[5, eos], [5, 6, 7, 8]], ["eos", "length"])
        results = session_generate_items(session, [torch.tensor([0, 3]), torch.tensor([0, 4])])
        assert results[0][0].tolist() == [5, eos]
        assert results[1][0].tolist() == [5, 6, 7, 8]

    def test_pad_equal_eos_is_not_kept_after_another_terminal_token(self):
        tokenizer = wordlevel_tokenizer()
        tokenizer.pad_token = tokenizer.eos_token
        session = _PaddingSession(tokenizer, [[5, 10], [5, 6, 7, 8], [5, 11]], ["eos", "length", "eos"])
        rows = [torch.tensor([0, 3]), torch.tensor([0, 4]), torch.tensor([0, 5])]
        results = session_generate_items(session, rows, eos_token_ids=(10,), stop_token_ids=(11,))
        assert [ids.tolist() for ids, _ in results] == [[5, 10], [5, 6, 7, 8], [5, 11]]

    def test_pad_that_is_a_generation_config_eos_is_kept(self):
        tokenizer = wordlevel_tokenizer()
        pad = tokenizer.pad_token_id
        assert pad != tokenizer.eos_token_id
        # the session reports no reason for a stop on an eos id the tokenizer does not know
        session = _PaddingSession(tokenizer, [[5, pad], [5, 6, 7, 8], [5, 6]], [None, "length", "stop"])
        rows = [torch.tensor([0, 3]), torch.tensor([0, 4]), torch.tensor([0, 5])]
        results = session_generate_items(session, rows, eos_token_ids=(pad,))
        assert [ids.tolist() for ids, _ in results] == [[5, pad], [5, 6, 7, 8], [5, 6]]
        # without the generation config's eos ids the pad is stripped as padding
        results = session_generate_items(session, rows)
        assert results[0][0].tolist() == [5]
