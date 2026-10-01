"""Every registered `DecodingDriver` preset under batched, multi-candidate generation.

The evaluation provider calls `pipeline.generate(..., n=k, return_output=True)` and reads one
`Output` per candidate, in row-major, candidate-minor order. Each preset runs on a batch of two
prompts of different lengths with `n=2`. A marker control pushes every generated token of a row
toward a token that identifies the row's prompt, and ends the second prompt's phases early, so the
continuations differ in length. The test checks the row count, the row order, right padding, the
ceiling, and each row's prompt.

The presets come from the registry, and every preset needs an entry in `PRESET_KWARGS` (a
minimal configuration for the tiny model), so a new preset without one fails here.

Hub-free: a tiny Llama with a WordLevel tokenizer that pads on the left and a minimal chat
template. Each continuation must start with the first token its driver appends, which a row padded
on the left of the returned batch does not.
"""
import pytest
import torch

from steerability.algorithms.core.internals.probes import Probe, ProbeSet
from steerability.algorithms.core.registry import REGISTRY
from steerability.algorithms.core.steering_pipeline import SteeringPipeline
from steerability.algorithms.output_control.base import DecodingDriver, OutputControl
from steerability.algorithms.output_control.routed_decoding import P, Route, Router, generate, prefix
from steerability.utils.rendering import encode_for_model
from tests.utils.tiny_models import tiny_llama, wordlevel_tokenizer

HIDDEN = 16
PAD_TOKEN_ID = 2  # the WordLevel tokenizer's <pad>
MAX_NEW_TOKENS = 8
NUM_CANDIDATES = 2
CHAT_TEMPLATE = (
    "{{ bos_token }}"
    "{% for message in messages %}{{ message['content'] }} {% endfor %}"
    "{% if add_generation_prompt %}sat {% endif %}"
)
CONVERSATIONS = [
    [{"role": "user", "content": "the cat sat on the mat"}],
    [{"role": "user", "content": "dog ran"}],
]
# the marker token in each prompt's continuations; neither occurs in a prompt or a fixed phase
MARKERS = ("fast", "attention")


def _marker_scorer(prompt, continuations, params):
    """Prefer continuations containing more marker tokens, so every search keeps a marked candidate."""
    return [float(sum(text.split().count(marker) for marker in MARKERS)) for text in continuations]


def _always_probe():
    generator = torch.Generator().manual_seed(0)
    weight = torch.randn(HIDDEN, generator=generator)
    return ProbeSet({
        "always": Probe(
            model_type="llama", location="layer_input", pooling="mean", layer_ids=[1],
            weights={1: weight / weight.norm()}, bias=1e9, meta={},
        ),
    })


PRESET_KWARGS = {
    "best_of_n": lambda: {"n": 2, "scorer": _marker_scorer},
    "budget_forcing": lambda: {
        "max_thinking_tokens": 3, "num_extensions": 1, "extension_text": "on", "end_think": "span",
    },
    "deal": lambda: {"reward_func": _marker_scorer, "lookahead": 2, "init_beams": 2, "topk": 1, "max_iterations": 4},
    "phased_decoding": lambda: {"plan": [{"generate": {"budget": 3}}, {"fixed": "on"}, {"generate": {}}]},
    "routed_decoding": lambda: {
        "probes": _always_probe(),
        "rules": Router(routes=[Route("note", when=P("always"), action=prefix("the"))], default_action=generate()),
    },
    "search_decoding": lambda: {
        "scorer": _marker_scorer, "segment_len": 2, "num_candidates": 2, "keep_k": 1, "max_iterations": 4,
        "propose_mode": "sample",
    },
}


# the first token a preset appends when it is not the row's marker
FIRST_APPENDED = {"routed_decoding": "the"}


def _registered_driver_presets() -> list[str]:
    """The toolkit's registered output-control methods whose control is a `DecodingDriver`."""
    return sorted(
        name for name, method in REGISTRY["output_control"].items()
        if issubclass(method.control_cls, DecodingDriver) and method.control_cls.__module__.startswith("steerability.")
    )


class _PromptMarker(OutputControl):
    """Push each row toward its prompt's marker token and away from the other prompt's marker.

    A row whose prefix contains `cat` favors the first marker. Any other row favors the second
    marker until its prefix contains one, then favors eos, so the second prompt's generated
    phases end after at most one marker.
    """

    Args = None

    def __init__(self, tokenizer):
        self._ids = {word: tokenizer.convert_tokens_to_ids(word) for word in ("cat", *MARKERS)}
        self._eos = tokenizer.eos_token_id

    def get_logits_processors(self, input_ids, runtime_kwargs, **kwargs):
        cat, first, second = self._ids["cat"], self._ids[MARKERS[0]], self._ids[MARKERS[1]]
        eos = self._eos

        def _mark(prefix_ids, scores):
            scores = scores.clone()
            for row, tokens in enumerate(prefix_ids):
                if bool((tokens == cat).any()):
                    favored, avoided = first, second
                elif bool((tokens == second).any()):
                    favored, avoided = eos, first
                else:
                    favored, avoided = second, first
                scores[row, favored] += 100.0
                scores[row, avoided] -= 100.0
            return scores

        return [_mark]


def _pipeline(driver):
    torch.manual_seed(0)
    model = tiny_llama(num_layers=2, hidden=HIDDEN, heads=2)
    tokenizer = wordlevel_tokenizer()
    tokenizer.chat_template = CHAT_TEMPLATE
    tokenizer.padding_side = "left"
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    pipeline = SteeringPipeline(controls=[_PromptMarker(tokenizer), driver], model=model, tokenizer=tokenizer)
    pipeline.steer()
    return pipeline, tokenizer


def test_every_registered_driver_preset_has_an_entry():
    assert _registered_driver_presets() == sorted(PRESET_KWARGS)


@pytest.mark.parametrize("name", _registered_driver_presets())
def test_candidates_are_row_major_right_padded_and_aligned_with_their_prompts(name):
    driver = REGISTRY["output_control"][name].control_cls(**PRESET_KWARGS[name]())
    pipeline, tokenizer = _pipeline(driver)
    outputs = pipeline.generate(
        messages=CONVERSATIONS, n=NUM_CANDIDATES, return_output=True, max_new_tokens=MAX_NEW_TOKENS,
        suppress_tokens=[PAD_TOKEN_ID],
    )

    assert len(outputs) == len(CONVERSATIONS) * NUM_CANDIDATES
    marker_ids = [tokenizer.convert_tokens_to_ids(marker) for marker in MARKERS]
    for index, output in enumerate(outputs):
        prompt_index = index // NUM_CANDIDATES
        expected_prompt = encode_for_model(tokenizer, messages=CONVERSATIONS[prompt_index])["input_ids"]
        prompt = output.adapted_input_ids[0]
        assert prompt[prompt != PAD_TOKEN_ID].tolist() == expected_prompt
        assert len(output.finish_reasons) == output.output_ids.size(0)  # one reason per returned row

        continuation = output.output_ids[0].tolist()
        length = len(continuation)
        while length and continuation[length - 1] == PAD_TOKEN_ID:
            length -= 1
        assert 0 < length <= MAX_NEW_TOKENS
        assert PAD_TOKEN_ID not in continuation[:length]  # pads only trail the continuation
        assert marker_ids[prompt_index] in continuation[:length]
        assert marker_ids[1 - prompt_index] not in continuation[:length]
        first = FIRST_APPENDED.get(name)
        expected_first = tokenizer.convert_tokens_to_ids(first) if first else marker_ids[prompt_index]
        assert continuation[0] == expected_first  # the slice at the prompt length is not shifted
