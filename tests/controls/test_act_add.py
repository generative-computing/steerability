"""Tests for ActAdd.

Three tiers:

- Extraction tests pinning the fit boundary: the single-pair estimator captures at the
  layer-input boundary (the boundary the control injects at), records that location on the
  artifact, and tokenizes without special tokens.
- Hook-level window tests on a hub-free tiny Llama, measuring the injected delta as the
  difference between layer `L-1`'s output and the stream layer `L` actually receives: the
  positional window covers absolute positions `[alignment, alignment + T)` exactly, decode
  passes inside the prompt window receive nothing, and a window extending past the prompt
  injects at the covered generated positions once each.
- Integration tests (parametrized over CI models) steering via a precomputed vector and via
  the prompt-pair path, then generating.
"""
import pytest
import torch

from aisteer360.algorithms.core.execution import ModelFacts
from aisteer360.algorithms.core.internals.capture import capture_hidden
from aisteer360.algorithms.core.steering_pipeline import SteeringPipeline
from aisteer360.algorithms.state_control.act_add.control import ActAdd
from aisteer360.algorithms.state_control.common.estimators import SinglePairEstimator
from aisteer360.algorithms.state_control.common.steering_vector import SteeringVector
from aisteer360.algorithms.state_control.common.transforms import AdditiveTransform
from tests.utils.sweep import build_param_grid
from tests.utils.tiny_models import tiny_llama, wordlevel_tokenizer

HIDDEN = 32
HEADS = 4
LAYERS = 4

PROMPT_TEXT = "Give me a short set of instructions to follow when you respond."


class _LayoutOnlySession:
    def __init__(self, layout: ModelFacts):
        self.layout = layout


@pytest.fixture()
def layout_session():
    return _LayoutOnlySession(ModelFacts(
        num_layers=LAYERS,
        hidden_size=HIDDEN,
        num_attention_heads=HEADS,
        head_dim=HIDDEN // HEADS,
        dtype="float32",
        model_fingerprint="0" * 16,
    ))


def _vector(k: int, layers=range(LAYERS), seed: int = 0, meta: dict | None = None) -> SteeringVector:
    generator = torch.Generator().manual_seed(seed)
    return SteeringVector(
        model_type="llama",
        directions={lid: torch.randn(k, HIDDEN, generator=generator) for lid in layers},
        meta=meta or {},
    )


class _BoundaryDelta:
    """Measures the injected delta at layer `L`'s input, one tensor per forward pass.

    The control's pre-hook edits the arguments of layer `L`'s call, so the un-edited stream is
    layer `L-1`'s output and the edited stream is what `L`'s `input_layernorm` receives; their
    difference is exactly the injected quantity.
    """

    def __init__(self, model, layer_id: int):
        assert layer_id >= 1
        self.upstream: list[torch.Tensor] = []
        self.received: list[torch.Tensor] = []
        layers = model.model.layers
        self._handles = [
            layers[layer_id - 1].register_forward_hook(self._grab_upstream),
            layers[layer_id].input_layernorm.register_forward_pre_hook(self._grab_received),
        ]

    def _grab_upstream(self, module, args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        self.upstream.append(hidden.detach().clone())

    def _grab_received(self, module, args):
        self.received.append(args[0].detach().clone())

    def deltas(self) -> list[torch.Tensor]:
        """Per-pass injected deltas of shape `[B, seq_len, H]`."""
        assert len(self.upstream) == len(self.received)
        return [received - upstream for upstream, received in zip(self.upstream, self.received)]

    def remove(self):
        for handle in self._handles:
            handle.remove()


def _steered_deltas(control, prompt_len: int, max_new_tokens: int = 4, layer_id: int = 1):
    """Steer a tiny Llama with `control`, generate greedily, and return the per-pass deltas."""
    torch.manual_seed(0)
    model = tiny_llama(num_layers=LAYERS, hidden=HIDDEN, heads=HEADS)
    tokenizer = wordlevel_tokenizer()
    pipeline = SteeringPipeline(controls=[control], model=model, tokenizer=tokenizer)
    pipeline.steer()

    observer = _BoundaryDelta(model, layer_id)
    input_ids = torch.arange(3, 3 + prompt_len, dtype=torch.long).unsqueeze(0)
    try:
        pipeline.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=None,
        )
    finally:
        observer.remove()
    return observer.deltas()


class TestExtractionBoundary:
    """The single-pair fit reads the same boundary the control injects at."""

    def test_fit_matches_independent_layer_input_capture(self):
        torch.manual_seed(0)
        model = tiny_llama(num_layers=LAYERS, hidden=HIDDEN, heads=HEADS)
        tokenizer = wordlevel_tokenizer(single="$A")

        positive, negative = "the cat sat", "the dog ran"  # equal token lengths, no padding
        fitted = SinglePairEstimator().fit(
            model, tokenizer, positive_prompt=positive, negative_prompt=negative,
        )

        enc = tokenizer([positive, negative], return_tensors="pt", add_special_tokens=False)
        hidden, _ = capture_hidden(enc, model=model, location="layer_input")
        for layer_id, direction in fitted.directions.items():
            expected = (hidden[layer_id][0] - hidden[layer_id][1]).to(torch.float32)
            torch.testing.assert_close(direction, expected)

    def test_fit_rows_align_with_content_tokens(self):
        """No fabricated BOS row: `T` equals the pair's content-token count on a
        BOS-prepending tokenizer."""
        torch.manual_seed(0)
        model = tiny_llama(num_layers=LAYERS, hidden=HIDDEN, heads=HEADS)
        tokenizer = wordlevel_tokenizer()  # prepends <s> on ordinary encoding
        fitted = SinglePairEstimator().fit(
            model, tokenizer, positive_prompt="the cat sat", negative_prompt="the dog ran",
        )
        assert fitted.num_tokens == 3

    def test_fit_records_location_and_meta_survives_save_load(self, tmp_path):
        torch.manual_seed(0)
        model = tiny_llama(num_layers=LAYERS, hidden=HIDDEN, heads=HEADS)
        tokenizer = wordlevel_tokenizer(single="$A")
        fitted = SinglePairEstimator().fit(
            model, tokenizer, positive_prompt="cat", negative_prompt="dog",
        )
        assert fitted.meta["location"] == "layer_input"

        path = str(tmp_path / "act_add.svec")
        fitted.save(path)
        loaded = SteeringVector.load(path)
        assert loaded.meta["location"] == "layer_input"

    def test_single_token_pair_fits_t1(self):
        """A single-token pair on a tokenizer without a BOS token fits a `T = 1` vector."""
        torch.manual_seed(0)
        model = tiny_llama(num_layers=LAYERS, hidden=HIDDEN, heads=HEADS)
        tokenizer = wordlevel_tokenizer(single="$A")
        assert tokenizer("cat", add_special_tokens=False)["input_ids"] == [4]
        fitted = SinglePairEstimator().fit(
            model, tokenizer, positive_prompt="cat", negative_prompt="dog",
        )
        assert fitted.num_tokens == 1


class TestWindowGeometry:
    """Injection covers absolute positions `[alignment, alignment + T)` and nothing else."""

    def test_prefill_window_positions_and_values(self):
        vector = _vector(k=2, layers=[1], seed=5)
        control = ActAdd(steering_vector=vector, layer_id=1, multiplier=2.0, alignment=2)
        deltas = _steered_deltas(control, prompt_len=5, max_new_tokens=3)

        prefill = deltas[0]
        expected = 2.0 * vector.directions[1]
        torch.testing.assert_close(prefill[0, 2], expected[0])
        torch.testing.assert_close(prefill[0, 3], expected[1])
        for position in (0, 1, 4):
            assert float(prefill[0, position].abs().max()) == 0.0
        for decode in deltas[1:]:
            assert float(decode.abs().max()) == 0.0

    def test_decode_passes_receive_nothing_at_alignment_zero(self):
        """A nonzero row 0 with `alignment=0` steers prompt position 0 only, never the
        KV-cached decode steps."""
        vector = _vector(k=2, layers=[1], seed=7)
        assert float(vector.directions[1][0].abs().max()) > 0
        control = ActAdd(steering_vector=vector, layer_id=1, multiplier=1.0, alignment=0)
        deltas = _steered_deltas(control, prompt_len=4, max_new_tokens=4)

        prefill = deltas[0]
        torch.testing.assert_close(prefill[0, 0], vector.directions[1][0])
        torch.testing.assert_close(prefill[0, 1], vector.directions[1][1])
        assert float(prefill[0, 2:].abs().max()) == 0.0
        assert len(deltas) == 4  # one prefill pass plus one pass per further generated token
        for decode in deltas[1:]:
            assert float(decode.abs().max()) == 0.0

    def test_window_past_prompt_injects_at_generated_positions_once(self):
        """A window extending past a short prompt injects the remaining rows at exactly the
        covered generated positions."""
        vector = _vector(k=3, layers=[1], seed=9)
        control = ActAdd(steering_vector=vector, layer_id=1, multiplier=1.0, alignment=0)
        deltas = _steered_deltas(control, prompt_len=2, max_new_tokens=4)

        prefill = deltas[0]
        torch.testing.assert_close(prefill[0, 0], vector.directions[1][0])
        torch.testing.assert_close(prefill[0, 1], vector.directions[1][1])
        # absolute position 2 is the first generated token; it receives row 2
        torch.testing.assert_close(deltas[1][0, 0], vector.directions[1][2])
        for decode in deltas[2:]:
            assert float(decode.abs().max()) == 0.0

    def test_single_token_pair_steers_alignment_position_only(self):
        """A `T = 1` positional vector fitted from a single-token pair on a no-BOS tokenizer
        steers position `alignment` only, not every masked position."""
        torch.manual_seed(0)
        model = tiny_llama(num_layers=LAYERS, hidden=HIDDEN, heads=HEADS)
        tokenizer = wordlevel_tokenizer(single="$A")
        control = ActAdd(positive_prompt="cat", negative_prompt="dog", layer_id=1, multiplier=3.0)
        pipeline = SteeringPipeline(controls=[control], model=model, tokenizer=tokenizer)
        pipeline.steer()

        fitted = control._steering_vector
        assert fitted.num_tokens == 1

        observer = _BoundaryDelta(model, 1)
        input_ids = tokenizer("the cat sat on mat", return_tensors="pt", add_special_tokens=False)["input_ids"]
        assert input_ids.size(1) == 5
        try:
            pipeline.generate(input_ids=input_ids, max_new_tokens=3, do_sample=False, eos_token_id=None)
        finally:
            observer.remove()

        deltas = observer.deltas()
        prefill = deltas[0]
        torch.testing.assert_close(prefill[0, 0], 3.0 * fitted.directions[1][0])
        assert float(prefill[0, 1:].abs().max()) == 0.0
        for decode in deltas[1:]:
            assert float(decode.abs().max()) == 0.0


class TestModeValidation:

    def test_multi_row_direction_without_positional_flag_raises(self):
        with pytest.raises(ValueError, match="positional=True"):
            AdditiveTransform({1: torch.ones(3, HIDDEN)})

    def test_multi_row_source_without_positional_flag_raises_at_bind(self, layout_session):
        from aisteer360.algorithms.state_control.common.sources import _Precomputed
        from aisteer360.algorithms.state_control.common.specs import Intervention

        transform = AdditiveTransform(_Precomputed(_vector(k=3, layers=[1])))
        intervention = Intervention(layers=(1,), transform=transform)
        with pytest.raises(ValueError, match="positional=True"):
            intervention.bind(None, None, layout=layout_session.layout)

    def test_recorded_layer_output_artifact_raises_at_steer(self, layout_session):
        vector = _vector(k=2, layers=[1], meta={"location": "layer_output"})
        control = ActAdd(steering_vector=vector, layer_id=1)
        with pytest.raises(ValueError, match="extracted at 'layer_output'"):
            control.steer(model=None, session=layout_session)

    def test_recorded_layer_input_artifact_steers(self, layout_session):
        vector = _vector(k=2, layers=[1], meta={"location": "layer_input"})
        control = ActAdd(steering_vector=vector, layer_id=1)
        control.steer(model=None, session=layout_session)
        assert control._layer_id == 1

    def test_unrecorded_artifact_passes(self, layout_session):
        control = ActAdd(steering_vector=_vector(k=2, layers=[1]), layer_id=1)
        control.steer(model=None, session=layout_session)
        assert control._layer_id == 1


# integration tests over CI models

def _dims(model):
    return model.config.hidden_size, model.config.num_hidden_layers


ACT_ADD_GRID = {
    "alignment": [0, 1],
    "multiplier": [1.0, 4.0],
}


@pytest.mark.parametrize("conf", build_param_grid(ACT_ADD_GRID))
def test_act_add_precomputed_vector(model_and_tokenizer, device: torch.device, conf: dict):
    """Steer with a precomputed positional vector and confirm generation produces tokens."""
    base_model, tokenizer = model_and_tokenizer
    model = base_model.to(device)

    hidden_size, num_layers = _dims(model)
    generator = torch.Generator().manual_seed(11)
    steering_vector = SteeringVector(
        model_type=model.config.model_type,
        directions={1: torch.randn(3, hidden_size, generator=generator)},
    )

    act_add = ActAdd(
        steering_vector=steering_vector,
        layer_id=1,
        multiplier=conf["multiplier"],
        alignment=conf["alignment"],
    )
    pipeline = SteeringPipeline(controls=[act_add], device_map=device, model=model, tokenizer=tokenizer)
    pipeline.steer()

    prompt_ids = tokenizer(PROMPT_TEXT, return_tensors="pt").input_ids.to(device)
    out_ids = pipeline.generate(input_ids=prompt_ids, max_new_tokens=8)

    assert isinstance(out_ids, torch.Tensor)
    assert out_ids.ndim == 2
    assert out_ids.size(1) >= 1


def test_act_add_prompt_pair_path(model_and_tokenizer, device: torch.device):
    """Fit the positional vector from a prompt pair and confirm generation."""
    base_model, tokenizer = model_and_tokenizer
    model = base_model.to(device)

    act_add = ActAdd(
        positive_prompt="Love",
        negative_prompt="Hate",
        layer_id=1,
        multiplier=2.0,
    )
    pipeline = SteeringPipeline(controls=[act_add], device_map=device, model=model, tokenizer=tokenizer)
    pipeline.steer()

    fitted = act_add._steering_vector
    assert fitted is not None
    assert fitted.meta["location"] == "layer_input"

    prompt_ids = tokenizer(PROMPT_TEXT, return_tensors="pt").input_ids.to(device)
    out_ids = pipeline.generate(input_ids=prompt_ids, max_new_tokens=8)

    assert isinstance(out_ids, torch.Tensor)
    assert out_ids.ndim == 2
    assert out_ids.size(1) >= 1
