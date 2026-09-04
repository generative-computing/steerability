"""Backend resolution and caching shared by the metrics that execute through the backend seam.

`LLMJudgeMetric` and `Perplexity` are configured by a model reference and a backend, never by
live model objects. Both resolve that configuration into a `Backend` through
`resolve_metric_backend`, which caches by `BackendSpec` so metrics configured with equal specs
share one loaded model or engine. This mirrors `SteeringPipeline._backends` and the cache-by-spec
guidance in the `Backend` docstring.

Cached backends are released and evicted by `release_metric_backends()`; the cache is otherwise
process-lifetime. The offline vLLM engine is one-per-process: a `vllm` judge next to a `vllm`
pipeline puts two engines in one process, which `VLLMBackend.release` does not support. Put one
side on `vllm-serve` or `huggingface`.
"""
from __future__ import annotations

import logging

from aisteer360.algorithms.core.execution.backend import Backend, resolve_backend_class
from aisteer360.algorithms.core.execution.spec import BackendSpec

logger = logging.getLogger(__name__)

BackendConfig = "BackendSpec | str | Backend | None"

_METRIC_BACKENDS: dict[BackendSpec, Backend] = {}


def _backend_for_spec(spec: BackendSpec) -> Backend:
    """Return the cached backend for `spec`, constructing it on first use."""
    backend = _METRIC_BACKENDS.get(spec)
    if backend is None:
        backend = resolve_backend_class(spec)(spec)
        _METRIC_BACKENDS[spec] = backend
    return backend


def release_metric_backends() -> None:
    """Release every cached metric backend and empty the cache.

    Mirrors `SteeringPipeline.release_backends()` for the metric-side cache. The cache is
    emptied first, then each formerly cached backend's `release()` runs once: engine-owning
    backends shut their engines down, and the Hugging Face and serve backends are no-ops. A
    release failure is logged and does not prevent the remaining releases. Live `Backend`
    instances passed to a metric are never cached and are not touched. Release is idempotent.
    Since metrics no longer pin their backend, emptying the cache drops the last reference to a
    Hugging Face judge, so its model memory is freed even though `release()` is a no-op there.

    Metrics resolve their backend per `compute()`, so a released metric constructs its backend
    again on next use. The offline vLLM engine's release is process-global with respect to vLLM
    distributed state and assumes no other live vLLM engine in the process (see
    `VLLMBackend.release`).
    """
    backends = list(_METRIC_BACKENDS.values())
    _METRIC_BACKENDS.clear()
    for backend in backends:
        try:
            backend.release()
        except Exception:
            logger.warning("Metric backend release failed", exc_info=True)


def resolve_metric_backend(
    model: str | None,
    backend: "BackendSpec | str | Backend | None",
) -> Backend:
    """Resolve one judge-model identity from `(model, backend)` into a `Backend`.

    Exactly one identity must emerge. The resolution rules:

    - `backend` is None or `"huggingface"`: requires `model`; resolves to an in-process
      Hugging Face backend for `BackendSpec(kind="huggingface", model=model)`, via the cache.
    - `backend` is `"vllm"`: requires `model`; resolves for `BackendSpec(kind="vllm", model=model)`,
      via the cache.
    - `backend` is the bare string `"vllm-serve"`: raises `TypeError`; a serve backend needs a
      `BackendSpec` carrying `base_url`.
    - `backend` is a `BackendSpec`: `spec.model` and `model` must agree when both are set (a new
      spec with `model` filled is used when the spec's model is unset), and at least one must be
      set; resolved via the cache.
    - `backend` is a live `Backend`: used as-is (never cached); `model` must be None.

    Model options (device placement, dtype, quantization) travel as spec options, e.g.
    `BackendSpec(kind="huggingface", model=..., options={"device_map": "cuda:1",
    "hf_model_kwargs": {"torch_dtype": "bfloat16"}})`. Option values must be plain data, since spec
    canonicalization renders live objects as strings, so dtypes are given as strings.

    Args:
        model: Model reference (hub id or local path), or None when `backend` carries the identity.
        backend: A `BackendSpec`, a backend-kind string, a live `Backend`, or None.

    Returns:
        The resolved backend.

    Raises:
        TypeError: If `backend` is the bare string `"vllm-serve"`; if a kind string requires a
            `model` and none is given; or if a `BackendSpec` and `model` are both unset.
        ValueError: If `backend` is a `BackendSpec` whose model conflicts with `model`; or if
            `backend` is a live `Backend` and `model` is also given.
    """
    if isinstance(backend, Backend):
        if model is not None:
            raise ValueError(
                "Pass either a live `Backend` or a `model` reference, not both; the backend "
                "already carries the judge-model identity."
            )
        return backend

    if isinstance(backend, BackendSpec):
        if backend.model is not None and model is not None and backend.model != model:
            raise ValueError(
                f"Conflicting judge model: `model`={model!r} and the backend spec's "
                f"model={backend.model!r} differ. Pass one, or make them equal."
            )
        if backend.model is None and model is not None:
            backend = BackendSpec(kind=backend.kind, model=model, options=backend.options_dict())
        if backend.model is None:
            raise TypeError(
                "The backend spec has no model and no `model` was given; set one so the judge has "
                "a model identity."
            )
        return _backend_for_spec(backend)

    if backend is None or backend == "huggingface":
        if model is None:
            raise TypeError("A judge on the huggingface backend requires a `model` reference.")
        return _backend_for_spec(BackendSpec(kind="huggingface", model=model))

    if backend == "vllm":
        if model is None:
            raise TypeError("A judge on the vllm backend requires a `model` reference.")
        return _backend_for_spec(BackendSpec(kind="vllm", model=model))

    if backend == "vllm-serve":
        raise TypeError(
            "A vllm-serve judge cannot be configured from the bare string 'vllm-serve'; pass a "
            "BackendSpec carrying base_url, e.g. BackendSpec(kind='vllm-serve', model=..., "
            "options={'base_url': 'http://localhost:8000'})."
        )

    raise TypeError(
        f"Unknown backend {backend!r}; pass a BackendSpec, a live Backend, or one of "
        "'huggingface' / 'vllm'."
    )
