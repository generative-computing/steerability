"""Residual-norm measurement and end-to-end dosing.

Covers `measure_residual_norms` (composition over `render_for_model` + `tokenize_texts` +
`layerwise_tokenwise_hidden`) and the dosed-vector path through `CAST(behavior_vector=...)` and
`ActivationAdapter(transform=AdditiveTransform(...))`. Hub-free on tiny models.
"""
import pytest
import torch

from steerability.algorithms.core.internals.capture import layerwise_tokenwise_hidden
from steerability.algorithms.core.steering_pipeline import SteeringPipeline
from steerability.algorithms.state_control.activation_adapter.control import ActivationAdapter
from steerability.algorithms.state_control.cast.control import CAST
from steerability.algorithms.state_control.common import measure_residual_norms
from steerability.algorithms.state_control.common.steering_vector import SteeringVector
from steerability.algorithms.state_control.common.transforms import AdditiveTransform
from steerability.utils.rendering import render_for_model
from tests.utils.tiny_models import tiny_llama, wordlevel_tokenizer

HIDDEN = 32
LAYERS = 4

# a minimal chat template written over the tiny vocab words so rendered prompts tokenize cleanly
_CHAT_TEMPLATE = (
    "{{ bos_token }}"
    "{% for message in messages %}{{ message['content'] }} {% endfor %}"
    "{% if add_generation_prompt %}sat {% endif %}"
)

# prompts drawn from the tiny WordLevel vocab
_PROMPTS = ["the cat sat", "dog ran fast", "the mat"]


def _model():
    torch.manual_seed(0)
    return tiny_llama(num_layers=LAYERS, hidden=HIDDEN, heads=4)


def _chat_tokenizer():
    tok = wordlevel_tokenizer()
    tok.chat_template = _CHAT_TEMPLATE
    return tok


def _manual_norms(model, tokenizer, prompts, layer_ids, location, stat, prompt_format="chat_prompt"):
    """Reference computation via a direct output_hidden_states forward, per prompt.

    For `location="layer_output"`, the final layer's entry in `output_hidden_states` carries the
    model's final norm already applied, so the raw boundary (the one a forward hook observes and
    the one `measure_residual_norms` reports) is re-captured with a forward hook on the last
    decoder layer.
    """
    device = next(model.parameters()).device
    per_layer_values = {lid: [] for lid in layer_ids}
    for p in prompts:
        text = render_for_model(tokenizer, prompt=p, mode=prompt_format)
        template_applied = getattr(tokenizer, "chat_template", None) is not None and prompt_format != "raw"
        enc = tokenizer(text, return_tensors="pt", add_special_tokens=not template_applied).to(device)
        final_raw = []
        handle = None
        if location == "layer_output":
            handle = model.model.layers[-1].register_forward_hook(
                lambda module, args, output: final_raw.append(output[0] if isinstance(output, tuple) else output)
            )
        try:
            with torch.no_grad():
                out = model(input_ids=enc["input_ids"], attention_mask=enc.get("attention_mask"),
                            output_hidden_states=True, return_dict=True)
        finally:
            if handle is not None:
                handle.remove()
        states = list(out.hidden_states[1:]) if location == "layer_output" else list(out.hidden_states[:-1])
        if location == "layer_output":
            states[-1] = final_raw[0]  # hidden_states[-1] is post-final-norm; the hook sees the raw boundary
        for lid in layer_ids:
            norms = states[lid].to(torch.float32).norm(dim=-1).flatten()
            per_layer_values[lid].append(norms)
    result = {}
    for lid, vals in per_layer_values.items():
        cat = torch.cat(vals)
        result[lid] = float(cat.median() if stat == "median" else cat.mean())
    return result


class TestMeasureResidualNorms:
    @pytest.mark.parametrize("location", ["layer_output", "layer_input"])
    @pytest.mark.parametrize("stat", ["median", "mean"])
    def test_matches_manual_hidden_states(self, location, stat):
        model, tokenizer = _model(), _chat_tokenizer()
        layer_ids = [0, 1, 2, 3]
        got = measure_residual_norms(model, tokenizer, layer_ids, _PROMPTS, location=location, stat=stat)
        expected = _manual_norms(model, tokenizer, _PROMPTS, layer_ids, location, stat)
        assert set(got.keys()) == set(layer_ids)
        for lid in layer_ids:
            assert got[lid] == pytest.approx(expected[lid], rel=1e-4, abs=1e-4)

    def test_none_layer_ids_returns_all_and_agrees(self):
        model, tokenizer = _model(), _chat_tokenizer()
        all_norms = measure_residual_norms(model, tokenizer, None, _PROMPTS)
        assert sorted(all_norms.keys()) == list(range(LAYERS))
        explicit = measure_residual_norms(model, tokenizer, [1, 2], _PROMPTS)
        for lid in (1, 2):
            assert all_norms[lid] == pytest.approx(explicit[lid], rel=1e-5, abs=1e-6)

    def test_padding_invariance(self):
        """A batched (left-padded) measurement equals the per-prompt aggregate; pads are excluded."""
        model, tokenizer = _model(), _chat_tokenizer()
        tokenizer.padding_side = "left"
        layer_ids = [1, 2]

        batched = measure_residual_norms(model, tokenizer, layer_ids, _PROMPTS, stat="mean")

        # per-prompt reference: pool the same real-token norms one prompt at a time
        device = next(model.parameters()).device
        per_layer = {lid: [] for lid in layer_ids}
        for p in _PROMPTS:
            text = render_for_model(tokenizer, prompt=p, mode="chat_prompt")
            enc = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)
            hidden = layerwise_tokenwise_hidden(model, enc, location="layer_output")
            for lid in layer_ids:
                per_layer[lid].append(hidden[lid].to(torch.float32).norm(dim=-1).flatten())
        per_prompt = {lid: float(torch.cat(v).mean()) for lid, v in per_layer.items()}

        for lid in layer_ids:
            assert batched[lid] == pytest.approx(per_prompt[lid], rel=1e-4, abs=1e-4)

    def test_templated_tokenization_no_duplicate_bos(self):
        """A chat-templated prompt is tokenized with add_special_tokens=False (single BOS)."""
        tokenizer = _chat_tokenizer()
        text = render_for_model(tokenizer, prompt="the cat sat", mode="chat_prompt")
        contract_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        buggy_ids = tokenizer(text, add_special_tokens=True)["input_ids"]  # would double the BOS
        bos = tokenizer.bos_token_id
        assert contract_ids.count(bos) == 1
        assert buggy_ids.count(bos) == 2  # confirms the contract choice actually matters here

    def test_raw_prompt_format_path(self):
        model, tokenizer = _model(), _chat_tokenizer()
        got = measure_residual_norms(model, tokenizer, [0, 1], _PROMPTS, prompt_format="raw")
        expected = _manual_norms(model, tokenizer, _PROMPTS, [0, 1], "layer_output", "median",
                                 prompt_format="raw")
        for lid in (0, 1):
            assert got[lid] == pytest.approx(expected[lid], rel=1e-4, abs=1e-4)

    def test_out_of_range_layer_raises(self):
        model, tokenizer = _model(), _chat_tokenizer()
        with pytest.raises(ValueError, match="out of range"):
            measure_residual_norms(model, tokenizer, [LAYERS], _PROMPTS)

    def test_empty_prompts_raises(self):
        model, tokenizer = _model(), _chat_tokenizer()
        with pytest.raises(ValueError, match="at least one prompt"):
            measure_residual_norms(model, tokenizer, [0], [])

    def test_bad_stat_raises(self):
        model, tokenizer = _model(), _chat_tokenizer()
        with pytest.raises(ValueError, match="stat must be"):
            measure_residual_norms(model, tokenizer, [0], _PROMPTS, stat="max")  # type: ignore[arg-type]


class TestDosedVectorEndToEnd:
    """A vector dosed via `scaled_to_norms(measure_residual_norms(...))` steers as expected."""

    DOSE = 0.5

    def _direction_vector(self):
        g = torch.Generator().manual_seed(7)
        return SteeringVector(
            model_type="llama",
            directions={lid: torch.randn(1, HIDDEN, generator=g) * 3.0 for lid in range(LAYERS)},
        )

    def test_dosed_delta_norm_matches_dose(self):
        """The dosed vector applied additively at each behavior layer produces a per-token delta of
        norm dose * measured_norm at steered positions and leaves unsteered positions untouched —
        the exact operation CAST's behavior hook performs at token_scope='all'."""
        model, tokenizer = _model(), _chat_tokenizer()
        behavior_layers = [2, 3]

        norms = measure_residual_norms(
            model, tokenizer, behavior_layers, _PROMPTS, location="layer_input"
        )
        dosed = self._direction_vector().scaled_to_norms(norms, scale=self.DOSE)

        # capture the layer-input residual stream to steer on realistic states
        text = render_for_model(tokenizer, prompt="the cat sat", mode="chat_prompt")
        enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        ids = enc["input_ids"]

        base = {}

        def make_capture(lid):
            def pre(module, args, kwargs):
                h = args[0] if args else kwargs.get("hidden_states")
                base[lid] = h.detach().clone()
            return pre

        handles = [model.model.layers[lid].register_forward_pre_hook(make_capture(lid), with_kwargs=True)
                   for lid in behavior_layers]
        with torch.no_grad():
            model(input_ids=ids, attention_mask=enc["attention_mask"])
        for h in handles:
            h.remove()

        transform = AdditiveTransform(dosed, strength=1.0)
        for lid in behavior_layers:
            hidden = base[lid]
            T = hidden.size(1)
            # steer only the second half of positions; the rest must stay bit-identical
            token_mask = torch.zeros(1, T, dtype=torch.bool)
            token_mask[0, T // 2:] = True
            out = transform.apply(hidden, layer_id=lid, token_mask=token_mask)

            delta = (out - hidden).to(torch.float32).norm(dim=-1)[0]  # [T]
            steered = delta[token_mask[0]]
            unsteered = out[0, ~token_mask[0]]
            assert torch.allclose(
                steered, torch.full_like(steered, self.DOSE * norms[lid]), rtol=1e-3, atol=1e-3
            ), f"layer {lid}: steered delta {steered} != {self.DOSE * norms[lid]}"
            assert torch.equal(unsteered, hidden[0, ~token_mask[0]])  # unsteered bit-identical

    def test_cast_generates_with_dosed_vector(self):
        model, tokenizer = _model(), _chat_tokenizer()
        behavior_layers = [2, 3]
        norms = measure_residual_norms(model, tokenizer, behavior_layers, _PROMPTS, location="layer_input")
        dosed = self._direction_vector().scaled_to_norms(norms, scale=self.DOSE)

        control = CAST(
            behavior_vector=dosed,
            behavior_layer_ids=behavior_layers,
            behavior_vector_strength=1.0,
            token_scope="all",
        )
        pipeline = SteeringPipeline(controls=[control], model=model, tokenizer=tokenizer)
        pipeline.steer()

        out = pipeline.generate(messages=[{"role": "user", "content": "the cat sat"}], max_new_tokens=3, do_sample=False)
        assert isinstance(out, str)

    def test_activation_adapter_binds_and_generates(self):
        model, tokenizer = _model(), _chat_tokenizer()
        behavior_layers = [1, 2]
        norms = measure_residual_norms(model, tokenizer, behavior_layers, _PROMPTS, location="layer_input")
        dosed = self._direction_vector().scaled_to_norms(norms, scale=self.DOSE)

        control = ActivationAdapter(
            transform=AdditiveTransform(dosed),
            layer_ids=behavior_layers,
            hook_point="layer_input",
        )
        pipeline = SteeringPipeline(controls=[control], model=model, tokenizer=tokenizer)
        pipeline.steer()

        out = pipeline.generate(messages=[{"role": "user", "content": "the cat sat"}], max_new_tokens=3, do_sample=False)
        assert isinstance(out, str)
