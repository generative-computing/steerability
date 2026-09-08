"""The in-process Hugging Face backend and its exclusive session.

`backend.py` holds `HFBackend` and its static capability advertisement; `session.py` holds
`ExclusiveSession` (direct model access, hook scopes, and the default decode loop) and the
generation-parameter rendering helpers. Constructing `HFBackend` loads a model and tokenizer in
the current process.
"""
from steerability.backends.huggingface.backend import HF_CAPABILITIES, HFBackend
from steerability.backends.huggingface.session import ExclusiveSession, compose_stop_criteria, render_hf_gen_kwargs

__all__ = [
    "ExclusiveSession",
    "HFBackend",
    "HF_CAPABILITIES",
    "compose_stop_criteria",
    "render_hf_gen_kwargs",
]
