"""Shared helpers for `TransformHookRuntime` tests.

`RecordingTransform` records the token mask seen at each apply and adds a constant, so a
mis-positioned mask changes hidden states and therefore greedy outputs. `strip_clock` wraps a
runtime hook to drop the `cache_position` kwarg, forcing the pass-counting fallback path.
"""
import torch

from aisteer360.algorithms.state_control.common.transforms.base import BaseTransform


class RecordingTransform(BaseTransform):
    """Records the token mask seen at each apply; adds a constant."""

    def __init__(self, value: float = 1.0):
        self.value = value
        self.masks: list[torch.BoolTensor] = []

    def apply(self, hidden_states, *, layer_id, token_mask, **kwargs):
        self.masks.append(token_mask.detach().clone())
        return hidden_states + self.value


class NeverCompleteRule:
    """A gate rule whose `is_complete` never reports True, so the gate re-scores every pass.

    `decide` returns all rows open or all closed per the constructor flag, regardless of
    evidence, keeping the gate's live decision constant while its readout keeps running.
    """

    wire_kind = None

    def __init__(self, open: bool = True):
        self._open = open

    def decide(self, values, num_rows):
        return torch.full((num_rows,), self._open, dtype=torch.bool)

    def is_complete(self, seen, expected):
        return False

    def export(self):
        return None


def strip_clock(hook):
    """Wrap a runtime hook so it never sees `cache_position`, forcing the pass-counting fallback.

    Only the wrapped hook is blinded; when a pre-hook returns replacement inputs, the kwarg is
    restored so the module's real call still receives it.
    """

    def stripped(module, args, kwargs, *rest):
        seen = {k: v for k, v in kwargs.items() if k != "cache_position"}
        result = hook(module, args, seen, *rest)
        if (
            not rest  # pre-hook shape; a returned (args, kwargs) pair replaces the module inputs
            and isinstance(result, tuple)
            and len(result) == 2
            and isinstance(result[1], dict)
            and "cache_position" in kwargs
            and "cache_position" not in result[1]
        ):
            return result[0], {**result[1], "cache_position": kwargs["cache_position"]}
        return result

    return stripped


class RuntimeCapture:
    """Captures each `TransformHookRuntime` that `build_hooks` constructs.

    `build_hooks` creates one fresh runtime per logical generation and discards its reference
    once the hook closures own it; tests asserting position or opener state install this via
    `capture_built_runtimes` and read `.last`.
    """

    def __init__(self):
        self.runtimes = []

    @property
    def last(self):
        return self.runtimes[-1] if self.runtimes else None


def capture_built_runtimes(monkeypatch) -> RuntimeCapture:
    """Patch the runtime module so every runtime built by `build_hooks` is recorded."""
    import aisteer360.algorithms.state_control.common.runtime as runtime_module

    capture = RuntimeCapture()
    original = runtime_module.TransformHookRuntime

    class _Recording(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            capture.runtimes.append(self)

    monkeypatch.setattr(runtime_module, "TransformHookRuntime", _Recording)
    return capture


class ScriptedSession:
    """A minimal session double whose `generate` runs items through a scripted callable.

    `fake_generate` follows the `model.generate` convention: it receives `input_ids` (and any
    generation kwargs it cares to read) and returns full sequences (prompt plus continuation).
    """

    def __init__(self, fake_generate, tokenizer=None):
        self._fake_generate = fake_generate
        self.tokenizer = tokenizer

    def generate(self, items, params):
        from aisteer360.algorithms.core.execution.payloads import ItemResult
        from aisteer360.algorithms.core.output import Output

        results = []
        gen_kwargs = params.to_gen_kwargs()
        for index, item in enumerate(items):
            prompt = item.prompt
            if prompt.token_ids is None:
                prompt = prompt.resolve_token_ids(self.tokenizer)
            ids = prompt.token_ids
            full = self._fake_generate(
                input_ids=ids, attention_mask=prompt.attention_mask, **gen_kwargs
            )
            results.append(ItemResult(index=index, output=Output(
                output_ids=full[:, ids.size(1):],
                adapted_input_ids=ids,
                finish_reason=None,
                finish_reasons=None,
            )))
        return results


def script_session_generate(monkeypatch, fake_generate):
    """Patch the in-process session so driver rollouts run through `fake_generate`.

    Drivers roll out through the pipeline's `SteeredSession`; scripting a rollout therefore
    scripts the session's `generate`.
    """
    from aisteer360.backends.huggingface import ExclusiveSession

    def generate(self, items, params):
        return ScriptedSession(fake_generate, tokenizer=self.tokenizer).generate(items, params)

    monkeypatch.setattr(ExclusiveSession, "generate", generate)
