"""State-control multiplicity in `SteeringPipeline` (design PR 1).

Covers the relaxed one-per-category rule for the state category: `merge_controls` returns an ordered
`state_controls` list, the session registers every entry's hooks in list order, same-module hooks
chain (so composition is order-sensitive by design), a failed registration removes prior entries'
hooks, `supports_batching` is the AND across all controls, and `compute_logprobs` composes edits.

Runs hub-free on a tiny randomly-initialized Llama.
"""
import pytest
import torch

from aisteer360.algorithms.core.steering_pipeline import SteeringPipeline
from aisteer360.algorithms.core.utils.assembly import collect_state_entries
from aisteer360.algorithms.core.utils.controls import merge_controls
from aisteer360.algorithms.input_control.base import InputControl
from aisteer360.algorithms.state_control.base import HookControl
from tests.utils.tiny_models import tiny_llama, wordlevel_tokenizer


class _IdentityInputControl(InputControl):
    """Concrete input control with an identity `adapt`."""

    supports_batching = True

    def adapt(self, input_ids, runtime_kwargs=None):
        return input_ids

HIDDEN = 32
HEADS = 4
LAYERS = 4


class _ConstantAddControl(HookControl):
    """Adds a constant vector to a layer's output at every position via a forward hook.

    A minimal concrete state control (no Args) used to observe hook composition and ordering. The
    added vector is `value * ones(H)`; the hooked layer is `model.layers.{layer_id}`.
    """

    Args = None
    supports_batching = True

    def __init__(self, layer_id: int, value: float, recorder: list | None = None):
        super().__init__()
        self._layer_id = layer_id
        self._value = value
        self._recorder = recorder

    def get_hooks(self, input_ids, runtime_kwargs, **kwargs):
        def _hook(module, args, kwargs_, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if self._recorder is not None:
                # record the mean of the pre-edit hidden state this hook observed
                self._recorder.append((self._value, float(hidden.mean())))
            hidden = hidden + self._value
            return (hidden,) + output[1:] if isinstance(output, tuple) else hidden

        return {
            "pre": [],
            "forward": [{"module": f"model.layers.{self._layer_id}", "hook_func": _hook}],
            "backward": [],
        }


class _AblateControl(HookControl):
    """Zeros a layer's output at every position (a non-commuting counterpart to additive)."""

    Args = None
    supports_batching = True

    def __init__(self, layer_id: int):
        super().__init__()
        self._layer_id = layer_id

    def get_hooks(self, input_ids, runtime_kwargs, **kwargs):
        def _hook(module, args, kwargs_, output):
            hidden = output[0] if isinstance(output, tuple) else output
            hidden = hidden * 0.0
            return (hidden,) + output[1:] if isinstance(output, tuple) else hidden

        return {
            "pre": [],
            "forward": [{"module": f"model.layers.{self._layer_id}", "hook_func": _hook}],
            "backward": [],
        }


class _BadModuleControl(HookControl):
    """Names a non-existent module so the session's registration raises."""

    Args = None
    supports_batching = True

    def get_hooks(self, input_ids, runtime_kwargs, **kwargs):
        def _hook(module, args, kwargs_, output):
            return output

        return {
            "pre": [],
            "forward": [{"module": "model.layers.does_not_exist", "hook_func": _hook}],
            "backward": [],
        }


def _pipeline(controls, model=None):
    """Build a steered pipeline; pass a shared `model` to compare edits across pipelines."""
    if model is None:
        model = tiny_llama(num_layers=LAYERS, hidden=HIDDEN, heads=HEADS)
    tokenizer = wordlevel_tokenizer()
    pipeline = SteeringPipeline(controls=controls, model=model, tokenizer=tokenizer)
    pipeline.steer()
    return pipeline, model


# merge_controls
class TestMergeControlsMultiplicity:
    def test_two_state_controls_encounter_order(self):
        a, b = _ConstantAddControl(1, 0.1), _ConstantAddControl(2, 0.2)
        result = merge_controls([a, b])
        assert result["state_controls"] == [a, b]

    def test_two_input_controls_encounter_order(self):
        a, b = _IdentityInputControl(), _IdentityInputControl()
        result = merge_controls([a, b])
        assert result["input_controls"] == [a, b]

    def test_unknown_type_raises(self):
        with pytest.raises(TypeError, match="Unknown control type"):
            merge_controls([object()])

    def test_empty_yields_empty_categories(self):
        result = merge_controls([])
        assert result["state_controls"] == []
        assert result["input_controls"] == []


# hook composition + ordering
class TestHookComposition:
    def test_both_controls_fire_and_chain_at_shared_module(self):
        """Two additive controls on the same layer both fire; the second observes the first's edit."""
        recorder = []
        first = _ConstantAddControl(1, value=1.0, recorder=recorder)
        second = _ConstantAddControl(1, value=2.0, recorder=recorder)
        pipeline, _ = _pipeline([first, second])

        input_ids = torch.arange(3, 7, dtype=torch.long).unsqueeze(0)
        pipeline.generate(input_ids=input_ids, max_new_tokens=1, do_sample=False, eos_token_id=None)

        # both hooks recorded at least once
        first_obs = [obs for val, obs in recorder if val == 1.0]
        second_obs = [obs for val, obs in recorder if val == 2.0]
        assert first_obs and second_obs
        # the second hook (registered after the first) observes the +1.0 edit already applied:
        # its observed mean equals the first hook's observed mean + 1.0 within the same pass
        assert second_obs[0] == pytest.approx(first_obs[0] + 1.0, abs=1e-4)

    def test_order_sensitive_non_commuting(self):
        """add-then-ablate zeroes the output; ablate-then-add leaves a constant. Different states."""
        input_ids = torch.arange(3, 7, dtype=torch.long).unsqueeze(0)

        def _final_hidden(controls):
            pipeline, model = _pipeline(controls)
            entries = collect_state_entries(
                pipeline.state_controls, input_ids, {},
                hooks_in_process=True, lowered_state=pipeline._lowered_state, model=pipeline.model,
            )
            backend = pipeline._backend_for(pipeline._resolve_backend_spec(None))
            captured = {}

            with backend.open_session() as session, session.entries_applied(entries):
                # register the capture hook AFTER the control hooks so it observes the composed edit
                def _capture(module, args, kwargs_, output):
                    captured["h"] = (output[0] if isinstance(output, tuple) else output).detach().clone()

                handle = model.model.layers[1].register_forward_hook(_capture, with_kwargs=True)
                try:
                    with torch.no_grad():
                        model(input_ids=input_ids)
                finally:
                    handle.remove()
            return captured["h"]

        # add(+5) then ablate(*0) -> zeros; ablate(*0) then add(+5) -> constant 5
        add_then_ablate = _final_hidden([_ConstantAddControl(1, 5.0), _AblateControl(1)])
        ablate_then_add = _final_hidden([_AblateControl(1), _ConstantAddControl(1, 5.0)])

        assert not torch.allclose(add_then_ablate, ablate_then_add)
        assert torch.allclose(add_then_ablate, torch.zeros_like(add_then_ablate), atol=1e-5)
        assert torch.allclose(ablate_then_add, torch.full_like(ablate_then_add, 5.0), atol=1e-5)


# registration unwind
class TestRegistrationUnwind:
    def test_failed_registration_removes_prior_entries_hooks(self):
        good = _ConstantAddControl(1, 1.0)
        bad = _BadModuleControl()
        pipeline, model = _pipeline([good, bad])

        input_ids = torch.arange(3, 7, dtype=torch.long).unsqueeze(0)
        with pytest.raises(AttributeError):
            pipeline.generate(input_ids=input_ids, max_new_tokens=1, do_sample=False, eos_token_id=None)

        # the good control's hooks must not leak onto the model
        assert len(model.model.layers[1]._forward_hooks) == 0

        # a subsequent plain forward pass is unaffected by any leaked hook
        with torch.no_grad():
            model(input_ids=input_ids)


# supports_batching
class TestSupportsBatching:
    def test_and_across_state_controls(self):
        class _NonBatch(_ConstantAddControl):
            supports_batching = False

        pipeline_all_ok = SteeringPipeline(
            model_name_or_path="m",controls=[_ConstantAddControl(1, 1.0), _ConstantAddControl(2, 1.0)]
        )
        assert pipeline_all_ok.supports_batching is True

        pipeline_mixed = SteeringPipeline(
            model_name_or_path="m",controls=[_ConstantAddControl(1, 1.0), _NonBatch(2, 1.0)]
        )
        assert pipeline_mixed.supports_batching is False


# compute_logprobs composition
class TestComputeLogprobsComposition:
    @pytest.mark.parametrize("batched", [True, False])
    def test_two_controls_match_single_fused_edit(self, batched):
        """logprobs under [add(a), add(b)] on distinct layers == logprobs under one fused control."""
        input_ids = torch.arange(3, 7, dtype=torch.long).unsqueeze(0)
        ref = torch.tensor([[7, 8, 9]], dtype=torch.long)

        class _FusedControl(HookControl):
            Args = None
            supports_batching = True

            def get_hooks(self, ids, rk, **kw):
                def _mk(val):
                    def _hook(module, args, kwargs_, output):
                        hidden = output[0] if isinstance(output, tuple) else output
                        hidden = hidden + val
                        return (hidden,) + output[1:] if isinstance(output, tuple) else hidden
                    return _hook

                return {
                    "pre": [],
                    "forward": [
                        {"module": "model.layers.1", "hook_func": _mk(0.5)},
                        {"module": "model.layers.2", "hook_func": _mk(0.3)},
                    ],
                    "backward": [],
                }

        # share one model across both pipelines so only the control composition differs
        model = tiny_llama(num_layers=LAYERS, hidden=HIDDEN, heads=HEADS)

        # ensure both paths take the batched-vs-sequential branch as requested
        composed = [_ConstantAddControl(1, 0.5), _ConstantAddControl(2, 0.3)]
        if not batched:
            composed[1].supports_batching = False  # forces the sequential fallback for both

        p_two, _ = _pipeline(composed, model=model)
        lp_two = p_two.compute_logprobs(input_ids=input_ids, ref_output_ids=ref)

        fused = _FusedControl()
        if not batched:
            fused.supports_batching = False
        p_one, _ = _pipeline([fused], model=model)
        lp_one = p_one.compute_logprobs(input_ids=input_ids, ref_output_ids=ref)

        assert torch.allclose(lp_two, lp_one, atol=1e-4)
