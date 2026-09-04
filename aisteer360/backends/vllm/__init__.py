"""The vLLM backends: the offline engine (`"vllm"`) and the OpenAI-compatible server
(`"vllm-serve"`).

This package imports cleanly without vLLM installed. The capability tables (`capabilities.py`)
are static data used by `check()`; the strict parameter-rendering table and the request/response
mapping helpers (`rendering.py`) are plain functions; the two backend classes and their
backend-side helpers live in `backend.py`, and the request sessions in `session.py`. Constructing
`VLLMBackend` requires the `vllm` optional dependency (it boots an engine); `VLLMServeBackend`
needs only a reachable vLLM server.
"""
from aisteer360.backends.vllm.backend import VLLMBackend, VLLMServeBackend
from aisteer360.backends.vllm.capabilities import VLLM_BASELINE_CAPABILITIES
from aisteer360.backends.vllm.rendering import (
    extract_ref_logprobs,
    map_vllm_finish_reason,
    merge_intervention_specs,
    raise_for_spec_rejection,
    remap_spec_for_scoring,
    render_constraint_sampling_args,
    render_guided_decoding_field,
    render_vllm_sampling_args,
)
from aisteer360.backends.vllm.session import VLLMOfflineSession, VLLMServeSession

__all__ = [
    "VLLMBackend",
    "VLLMServeBackend",
    "VLLM_BASELINE_CAPABILITIES",
    "extract_ref_logprobs",
    "map_vllm_finish_reason",
    "merge_intervention_specs",
    "raise_for_spec_rejection",
    "remap_spec_for_scoring",
    "render_constraint_sampling_args",
    "render_guided_decoding_field",
    "render_vllm_sampling_args",
    "VLLMOfflineSession",
    "VLLMServeSession",
]
