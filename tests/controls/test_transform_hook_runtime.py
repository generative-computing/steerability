"""Unit tests for the shared `TransformHookRuntime`.

Exercises the runtime directly with hand-registered hooks on a tiny Llama: `cache_position`-derived
position offsets and their pass-counting fallback, pass-opener KV-offset semantics across
prefill/decode with multiple hooked layers, `after_prompt`/`last_k`/`from_position`/`all` token
scopes, auxiliary-pass marking (aligned and detached), beam-expansion alignment, tuple vs
bare-tensor outputs, the pre-hook (`layer_input`) extract/replace path, and read-only condition
hooks feeding a gate.

Runs hub-free on a tiny randomly-initialized Llama.
"""
import warnings

import pytest
import torch

from steerability.algorithms.core.utils.auxiliary_pass import auxiliary_pass
from steerability.algorithms.state_control.common.gating import CallableReadout, Evidence, Gate, PerKeyThreshold
from steerability.algorithms.state_control.common.runtime import TransformHookRuntime
from steerability.algorithms.state_control.common.token_scope import compute_prompt_lens
from tests.utils.runtime_helpers import NeverCompleteRule
from tests.utils.runtime_helpers import RecordingTransform as _RecordingTransform
from tests.utils.runtime_helpers import strip_clock
from tests.utils.tiny_models import tiny_llama

HIDDEN = 32
HEADS = 4
LAYERS = 4


def _register(model, runtime, hooks, strip=False):
    """Register `(layer_id, hook_callable)` pairs at the runtime's hook point; return handles."""
    handles = []
    for layer_id, hook in hooks:
        module = model.model.layers[layer_id]
        if strip:
            hook = strip_clock(hook)
        if runtime.hook_point == "layer_output":
            handles.append(module.register_forward_hook(hook, with_kwargs=True))
        else:
            handles.append(module.register_forward_pre_hook(hook, with_kwargs=True))
    return handles


class TestPassOpenerOffset:
    @pytest.mark.parametrize("strip", [False, True], ids=["clock", "fallback"])
    def test_offset_advances_once_per_pass_multi_layer(self, strip):
        """With three hooked layers, `after_prompt` steers every decode pass and no prefill pass."""
        model = tiny_llama(num_layers=LAYERS, hidden=HIDDEN, heads=HEADS)
        runtime = TransformHookRuntime(hook_point="layer_output")
        gate = None
        transforms = {lid: _RecordingTransform() for lid in (0, 1, 2)}

        input_ids = torch.arange(3, 7, dtype=torch.long).unsqueeze(0)  # prompt_len 4
        runtime.reset(compute_prompt_lens(input_ids, None))

        layer_ids = [0, 1, 2]
        opener = min(layer_ids)
        hooks = [
            (lid, runtime.build_behavior_hook(
                layer_id=lid, transform=transforms[lid], gate=gate,
                token_scope="after_prompt", is_pass_opener=(lid == opener)))
            for lid in layer_ids
        ]
        handles = _register(model, runtime, hooks, strip=strip)
        try:
            model.generate(input_ids=input_ids, max_new_tokens=5, do_sample=False, eos_token_id=None)
        finally:
            for h in handles:
                h.remove()

        # the runtime skips no-op applies, so the prefill pass (all positions < prompt_len)
        # records nothing; each layer's transform is called once per DECODE pass = 4
        # (the final generated token is emitted but never re-processed)
        for lid in layer_ids:
            masks = transforms[lid].masks
            assert len(masks) == 4
            # decode passes (seq_len 1, absolute position >= 4) -> steered
            for m in masks:
                assert m.shape[1] == 1 and bool(m.all())

    @pytest.mark.parametrize("strip", [False, True], ids=["clock", "fallback"])
    @pytest.mark.parametrize("prompt_len", [1, 4])
    def test_prompt_len_one_still_steers_decode(self, prompt_len, strip):
        """A length-1 prompt must not confuse prefill with decode."""
        model = tiny_llama(num_layers=LAYERS, hidden=HIDDEN, heads=HEADS)
        runtime = TransformHookRuntime(hook_point="layer_output")
        transform = _RecordingTransform()

        input_ids = torch.arange(3, 3 + prompt_len, dtype=torch.long).unsqueeze(0)
        runtime.reset(compute_prompt_lens(input_ids, None))
        hook = runtime.build_behavior_hook(
            layer_id=1, transform=transform, gate=None,
            token_scope="after_prompt", is_pass_opener=True)
        handles = _register(model, runtime, [(1, hook)], strip=strip)
        try:
            model.generate(input_ids=input_ids, max_new_tokens=5, do_sample=False, eos_token_id=None)
        finally:
            for h in handles:
                h.remove()

        steered = sum(1 for m in transform.masks if bool(m.any()))
        assert steered == 4  # max_new_tokens - 1 decode passes


class TestTokenScopes:
    def _run_single_pass(self, token_scope, seq_len, **kw):
        model = tiny_llama(num_layers=LAYERS, hidden=HIDDEN, heads=HEADS)
        runtime = TransformHookRuntime(hook_point="layer_output")
        transform = _RecordingTransform()
        input_ids = torch.arange(3, 3 + seq_len, dtype=torch.long).unsqueeze(0)
        runtime.reset(compute_prompt_lens(input_ids, None))
        hook = runtime.build_behavior_hook(
            layer_id=1, transform=transform, gate=None, token_scope=token_scope,
            is_pass_opener=True, **kw)
        handles = _register(model, runtime, [(1, hook)])
        try:
            with torch.no_grad():
                model(input_ids=input_ids)
        finally:
            for h in handles:
                h.remove()
        return transform.masks[0]

    def test_all_scope(self):
        mask = self._run_single_pass("all", seq_len=4)
        assert bool(mask.all())

    def test_last_k_scope(self):
        mask = self._run_single_pass("last_k", seq_len=4, last_k=2)
        assert mask.squeeze(0).tolist() == [False, False, True, True]

    def test_from_position_scope(self):
        mask = self._run_single_pass("from_position", seq_len=4, from_position=1)
        assert mask.squeeze(0).tolist() == [False, True, True, True]


class TestBeamExpansion:
    def test_align_mask_to_expanded_batch(self):
        """When hidden batch > prompt batch (beam search), the mask is replicated to align."""
        model = tiny_llama(num_layers=LAYERS, hidden=HIDDEN, heads=HEADS)
        runtime = TransformHookRuntime(hook_point="layer_output")
        transform = _RecordingTransform()
        input_ids = torch.arange(3, 7, dtype=torch.long).unsqueeze(0)
        runtime.reset(compute_prompt_lens(input_ids, None))
        hook = runtime.build_behavior_hook(
            layer_id=1, transform=transform, gate=None,
            token_scope="all", is_pass_opener=True)
        handles = _register(model, runtime, [(1, hook)])
        try:
            model.generate(
                input_ids=input_ids, max_new_tokens=2, do_sample=False,
                num_beams=3, eos_token_id=None,
            )
        finally:
            for h in handles:
                h.remove()
        # some recorded mask must have a batch dimension expanded to a multiple of 1 (the beams)
        assert any(m.size(0) >= 3 for m in transform.masks)


class TestBareTensorOutput:
    def test_handles_bare_tensor_layer_output(self):
        """A layer returning a bare tensor (not a tuple) is handled without error."""
        runtime = TransformHookRuntime(hook_point="layer_output")
        transform = _RecordingTransform(value=2.0)
        runtime.reset(torch.tensor([4]))
        hook = runtime.build_behavior_hook(
            layer_id=0, transform=transform, gate=None,
            token_scope="all", is_pass_opener=True)

        hidden = torch.zeros(1, 4, HIDDEN)
        out = hook(None, (), {}, hidden)  # bare-tensor output path
        assert isinstance(out, torch.Tensor)
        assert torch.allclose(out, torch.full_like(hidden, 2.0))

    def test_handles_tuple_layer_output(self):
        runtime = TransformHookRuntime(hook_point="layer_output")
        transform = _RecordingTransform(value=2.0)
        runtime.reset(torch.tensor([4]))
        hook = runtime.build_behavior_hook(
            layer_id=0, transform=transform, gate=None,
            token_scope="all", is_pass_opener=True)

        hidden = torch.zeros(1, 4, HIDDEN)
        extra = torch.tensor([1.0])
        out = hook(None, (), {}, (hidden, extra))
        assert isinstance(out, tuple)
        assert torch.allclose(out[0], torch.full_like(hidden, 2.0))
        assert out[1] is extra  # trailing elements preserved


class TestPreHookPath:
    def test_layer_input_extract_replace(self):
        """The `layer_input` pre-hook path steers via extract/replace on the layer input."""
        model = tiny_llama(num_layers=LAYERS, hidden=HIDDEN, heads=HEADS)
        runtime = TransformHookRuntime(hook_point="layer_input")
        transform = _RecordingTransform(value=3.0)
        input_ids = torch.arange(3, 7, dtype=torch.long).unsqueeze(0)
        runtime.reset(compute_prompt_lens(input_ids, None))

        captured = {}

        def _capture(module, args, kwargs):
            hidden = args[0] if args else kwargs.get("hidden_states")
            captured["h"] = hidden.detach().clone()
            return None

        hook = runtime.build_behavior_hook(
            layer_id=2, transform=transform, gate=None,
            token_scope="all", is_pass_opener=True)
        # register the steering pre-hook, then a capture pre-hook AFTER it to observe the edit
        h1 = model.model.layers[2].register_forward_pre_hook(hook, with_kwargs=True)
        h2 = model.model.layers[2].register_forward_pre_hook(_capture, with_kwargs=True)
        try:
            with torch.no_grad():
                model(input_ids=input_ids)
        finally:
            h1.remove()
            h2.remove()

        assert transform.masks  # the pre-hook fired
        assert "h" in captured  # the layer input was edited before the capture hook saw it


class TestConditionHook:
    def test_condition_hook_is_read_only_and_updates_gate(self):
        """A condition hook pools, reads out per-row values, feeds the gate, and leaves hidden
        states untouched."""
        runtime = TransformHookRuntime(hook_point="layer_output")
        seen = {}

        def _readout(pooled, layer_id):
            seen["pooled"] = pooled
            return torch.full((pooled.size(0),), 0.9)  # per-row; above threshold

        gate = Gate(
            Evidence((1,), CallableReadout(_readout)),
            PerKeyThreshold(threshold=0.5, comparator="ge"),
        )
        gate.reset(1)
        runtime.reset(torch.tensor([4]))

        hook = runtime.build_condition_hook(layer_id=1, gate=gate, is_pass_opener=True)

        hidden = torch.randn(1, 4, HIDDEN)
        out = hook(None, (), {}, hidden)
        assert out is hidden  # unmodified output returned as-is
        assert seen["pooled"].shape == (1, HIDDEN)
        assert gate.is_open()  # 0.9 >= 0.5 opens the gate


def _after_prompt_hook(runtime, transform, prompt_len=4):
    """Build an `after_prompt` opener behavior hook on a freshly reset runtime."""
    runtime.reset(torch.tensor([prompt_len]))
    return runtime.build_behavior_hook(
        layer_id=0, transform=transform, gate=None,
        token_scope="after_prompt", is_pass_opener=True)


class TestClockOffsets:
    def test_decode_position_from_cache_position(self):
        """A single-token pass is positioned by `cache_position`, not by the pass counter."""
        runtime = TransformHookRuntime(hook_point="layer_output")
        transform = _RecordingTransform()
        hook = _after_prompt_hook(runtime, transform)

        hidden = torch.zeros(1, 1, HIDDEN)
        hook(None, (), {"cache_position": torch.tensor([7])}, (hidden,))
        assert len(transform.masks) == 1 and bool(transform.masks[0].all())

        hook(None, (), {"cache_position": torch.tensor([2])}, (hidden,))
        assert len(transform.masks) == 1  # position 2 is inside the prompt; nothing recorded

    def test_restart_sequence_never_steers_prompt_columns(self):
        """A second `generate` call re-forwarding a longer frontier keeps prompt columns unsteered."""
        runtime = TransformHookRuntime(hook_point="layer_output")
        transform = _RecordingTransform()
        hook = _after_prompt_hook(runtime, transform)

        def run(seq_len, start):
            hidden = torch.zeros(1, seq_len, HIDDEN)
            hook(None, (), {"cache_position": torch.arange(start, start + seq_len)}, (hidden,))

        run(4, 0)  # first call: prefill (all prompt, nothing recorded)
        run(1, 4)  # first call: decode step
        run(6, 0)  # second call: re-prefill over the longer frontier
        run(1, 6)  # second call: decode step

        assert len(transform.masks) == 3
        assert transform.masks[0].shape == (1, 1) and bool(transform.masks[0].all())
        re_prefill = transform.masks[1]
        assert re_prefill.shape == (1, 6)
        assert not bool(re_prefill[:, :4].any())  # prompt columns stay unsteered
        assert bool(re_prefill[:, 4:].all())
        assert transform.masks[2].shape == (1, 1) and bool(transform.masks[2].all())

    def test_clock_disappearing_warns_once_and_falls_back(self):
        """A pass missing `cache_position` after the clock was observed warns once and counts."""
        runtime = TransformHookRuntime(hook_point="layer_output")
        transform = _RecordingTransform()
        hook = _after_prompt_hook(runtime, transform)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            hook(None, (), {"cache_position": torch.arange(4)}, (torch.zeros(1, 4, HIDDEN),))
            hook(None, (), {}, (torch.zeros(1, 1, HIDDEN),))
            hook(None, (), {}, (torch.zeros(1, 1, HIDDEN),))
        matches = [w for w in caught if "falling back to pass counting" in str(w.message)]
        assert len(matches) == 1
        # the fallback snapshot places both cache-less passes after the prompt
        assert len(transform.masks) == 2
        assert all(bool(m.all()) for m in transform.masks)


class TestAuxiliaryPasses:
    def _hook_and_state(self):
        runtime = TransformHookRuntime(hook_point="layer_output")
        transform = _RecordingTransform()
        hook = _after_prompt_hook(runtime, transform)
        return runtime, transform, hook

    def test_aligned_aux_with_clock_applies_and_leaves_counter(self):
        runtime, transform, hook = self._hook_and_state()
        hidden = torch.zeros(1, 1, HIDDEN)
        with auxiliary_pass(aligned=True):
            hook(None, (), {"cache_position": torch.tensor([5])}, (hidden,))
        assert len(transform.masks) == 1 and bool(transform.masks[0].all())
        assert runtime._offset == 0  # the fallback counter never sees auxiliary passes
        assert runtime._prefill_seen is False

    def test_aligned_aux_without_clock_skips_and_warns_once(self):
        runtime, transform, hook = self._hook_and_state()
        hidden = torch.zeros(1, 1, HIDDEN)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with auxiliary_pass(aligned=True):
                hook(None, (), {}, (hidden,))
                hook(None, (), {}, (hidden,))
        assert not transform.masks
        matches = [w for w in caught if "Auxiliary same-model passes" in str(w.message)]
        assert len(matches) == 1

    def test_detached_aux_skipped_in_both_modes_without_warning(self):
        runtime, transform, hook = self._hook_and_state()
        hidden = torch.zeros(1, 1, HIDDEN)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with auxiliary_pass(aligned=False):
                hook(None, (), {"cache_position": torch.tensor([5])}, (hidden,))
                hook(None, (), {}, (hidden,))
        assert not transform.masks
        assert not caught
        assert runtime._offset == 0


class TestConditionHookAuxiliary:
    def _condition_hook(self):
        runtime = TransformHookRuntime(hook_point="layer_output")
        readout_calls = []

        def readout(pooled, layer_id):
            readout_calls.append(tuple(pooled.shape))
            return torch.zeros(pooled.size(0))

        gate = Gate(Evidence((0,), CallableReadout(readout)), NeverCompleteRule(open=True))
        gate.reset(1)
        runtime.reset(torch.tensor([4]), prompt_mask=torch.ones(1, 4, dtype=torch.bool))
        hook = runtime.build_condition_hook(layer_id=0, gate=gate, is_pass_opener=True)
        return gate, readout_calls, hook

    @pytest.mark.parametrize("with_clock", [True, False], ids=["clock", "fallback"])
    @pytest.mark.parametrize("aligned", [True, False], ids=["aligned", "detached"])
    @pytest.mark.parametrize("seq_len", [2, 6])
    def test_condition_ignores_auxiliary_passes(self, aligned, with_clock, seq_len):
        """No scoring, no gate update, no accounting; a variant prompt of any length never raises."""
        gate, readout_calls, hook = self._condition_hook()
        hidden = torch.zeros(1, seq_len, HIDDEN)
        kwargs = {"cache_position": torch.arange(seq_len)} if with_clock else {}
        with auxiliary_pass(aligned=aligned):
            hook(None, (), kwargs, (hidden,))
        assert not readout_calls
        assert gate.evidence_values() == {}


class TestFallbackMultiCallHeuristic:
    def _hook(self):
        runtime = TransformHookRuntime(hook_point="layer_output")
        return _after_prompt_hook(runtime, _RecordingTransform())

    def test_two_prefill_length_passes_warn_once(self):
        hook = self._hook()
        hidden = torch.zeros(1, 4, HIDDEN)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            hook(None, (), {}, (hidden,))
            hook(None, (), {}, (hidden,))
            hook(None, (), {}, (hidden,))
        matches = [w for w in caught if "Multiple generate calls" in str(w.message)]
        assert len(matches) == 1

    def test_single_teacher_forced_pass_does_not_warn(self):
        hook = self._hook()
        hidden = torch.zeros(1, 10, HIDDEN)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            hook(None, (), {}, (hidden,))
        assert not [w for w in caught if "Multiple generate calls" in str(w.message)]
