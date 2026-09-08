"""Engine-boot environment policy for the offline vLLM backend and a launched `vllm serve`.

vLLM reads some settings from process environment variables only, and its engine core is a
spawned process that inherits the environment as it stands when `LLM(...)` builds its config.
This module computes one boot policy with two consumers: the offline backend applies it to
`os.environ` with save and restore semantics around `LLM(...)`, so the settings are live while
the engine is constructed and are cleared again immediately afterwards, and `serve_environment`
returns the same policy as a fresh mapping for a `vllm serve` process.

Two variables are governed:

- `VLLM_USE_FLASHINFER_SAMPLER` defaults to `"0"` on every boot. vLLM selects the FlashInfer
  top-k/top-p sampler when it is available, and `flashinfer-python` ships that kernel JIT-only;
  the JIT compile runs at startup warmup and invokes `nvcc` at a path derived from `CUDA_HOME`,
  so a node whose CUDA toolkit does not match the installed torch build fails at boot. The
  native PyTorch sampler decodes the same greedy tokens. This is a default: an explicit caller
  setting wins.
- `VLLM_HOOK_WORKER` is forced to `"unified"` for a `hook_plugin` boot to select the plugin's
  unified worker, and is restored to its prior value after the boot.

This module does not set `VLLM_USE_V2_MODEL_RUNNER`; the vLLM-Hook plugin owns the model-runner
constraint (it pins the legacy runner from inside the plugin, covering the `vllm serve` process
the toolkit does not launch).
"""
import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager

HOOK_WORKER_VARIABLE = "VLLM_HOOK_WORKER"
FLASHINFER_SAMPLER_VARIABLE = "VLLM_USE_FLASHINFER_SAMPLER"


def engine_boot_environment(hook_plugin: bool) -> tuple[dict[str, str], dict[str, str]]:
    """Return the forced and default environment mappings for one offline engine boot.

    Args:
        hook_plugin: Whether the boot selects the vLLM-Hook unified worker.

    Returns:
        A `(forced, defaults)` pair. `forced` variables are applied regardless of any existing
        value and restored afterwards; `defaults` are applied only when the variable is unset,
        so an explicit caller value wins. `defaults` always carries
        `VLLM_USE_FLASHINFER_SAMPLER="0"`; `forced` carries `VLLM_HOOK_WORKER="unified"` only
        when `hook_plugin` is true.
    """
    forced: dict[str, str] = {}
    defaults: dict[str, str] = {FLASHINFER_SAMPLER_VARIABLE: "0"}
    if hook_plugin:
        forced[HOOK_WORKER_VARIABLE] = "unified"
    return forced, defaults


@contextmanager
def engine_environment(forced: Mapping[str, str], defaults: Mapping[str, str]) -> Iterator[dict[str, str]]:
    """Apply the boot environment for the span of the block and restore it on exit.

    Forced variables are written regardless of any existing value; default variables are
    written only when absent from `os.environ`. Every variable written is restored on exit,
    including variables that were unset on entry (they are removed again). Restoration also
    runs when the block raises.

    Args:
        forced: Variables to set regardless of any existing value.
        defaults: Variables to set only when they are absent from the environment.

    Yields:
        The variables actually written, as a name-to-value mapping.
    """
    previous: dict[str, str | None] = {}
    applied: dict[str, str] = {name: value for name, value in defaults.items() if name not in os.environ}
    applied.update(forced)
    for name, value in applied.items():
        previous[name] = os.environ.get(name)
        os.environ[name] = value
    try:
        yield dict(applied)
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def serve_environment(hook_plugin: bool, base: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return the environment for a `vllm serve` process under the engine boot policy.

    Args:
        hook_plugin: Whether the server loads the vLLM-Hook unified worker.
        base: Environment to start from; `os.environ` when omitted. Not mutated.

    Returns:
        A copy of `base` with the boot defaults filled where absent and the forced variables applied.
    """
    forced, defaults = engine_boot_environment(hook_plugin)
    env = dict(os.environ if base is None else base)
    for name, value in defaults.items():
        env.setdefault(name, value)
    env.update(forced)
    return env
