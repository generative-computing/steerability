"""Digests identifying the model a fitted artifact was estimated on."""
import hashlib

import torch
from transformers import PreTrainedModel


def model_fingerprint(model: PreTrainedModel) -> str:
    """Deterministic identity digest of a model's configuration, dtype, and sampled weights.

    The digest is a sha256 over (a) `model.config.to_json_string()`, (b) `str(model.dtype)`, and
    (c) for up to 8 evenly spaced named parameters, the parameter name plus its first 64 elements
    flattened and cast to float32 bytes, truncated to 16 hex characters.

    Properties:

    - The same checkpoint at the same dtype yields equal digests in any process or device
        placement, since parameter values are deterministic under `from_pretrained`.
    - Any weight edit that touches the sampled slices (SFT, DPO, merging, LoRA merge) changes the
        digest; so does attaching unmerged adapters, since parameter names change.
    - dtype is part of the identity, since a calibrated operating point is a numeric artifact and
        the same checkpoint loaded at a different dtype is a different runtime model for detection
        purposes.

    Args:
        model: The model to fingerprint.

    Returns:
        A 16-character lowercase hex digest.
    """
    digest = hashlib.sha256()
    digest.update(model.config.to_json_string().encode("utf-8"))
    digest.update(str(model.dtype).encode("utf-8"))

    named_parameters = list(model.named_parameters())
    n = len(named_parameters)
    if n > 0:
        k = min(8, n)
        indices = [0] if k == 1 else [round(i * (n - 1) / (k - 1)) for i in range(k)]
        for idx in dict.fromkeys(indices):
            name, param = named_parameters[idx]
            digest.update(name.encode("utf-8"))
            sample = param.detach().reshape(-1)[:64].to(device="cpu", dtype=torch.float32).contiguous()
            digest.update(sample.numpy().tobytes())

    return digest.hexdigest()[:16]


def session_artifact_identity(session) -> tuple[str, dict]:
    """`(model_type, meta)` recorded for a session-fitted artifact, from the session layout.

    The meta carries the layout's `model_fingerprint` and `model_ref` when present, so
    identity checks against the venue that will read the artifact stay possible without a
    live model. Returns `("unknown", {})` when no session layout is available.

    Args:
        session: The `SteeringSession` the artifact was fitted through, or None.

    Returns:
        The model type and the provenance mapping.
    """
    layout = None
    if session is not None:
        try:
            layout = session.layout
        except Exception:
            layout = None
    if layout is None:
        return "unknown", {}
    meta: dict = {}
    if layout.model_fingerprint:
        meta["model_fingerprint"] = layout.model_fingerprint
    if layout.model_ref:
        meta["model_ref"] = layout.model_ref
    return layout.model_type or "unknown", meta


def artifact_provenance_meta(model, tokenizer=None) -> dict:
    """Provenance fingerprints for a fitted steering artifact.

    Always records the toolkit model fingerprint; when `vllm_hook_plugins` is installed, adds
    the plugin's config and chat-template fingerprint recipes, which cross-check against a
    serving engine's discovery payload.

    Args:
        model: The model the artifact was fitted on.
        tokenizer: The tokenizer used during fitting, for the chat-template fingerprint.

    Returns:
        The provenance mapping. Keys: `"model_fingerprint"`, and when the plugin is installed,
        `"config_fingerprint"` and (with a tokenizer) `"chat_template_fingerprint"`.
    """
    meta = {"model_fingerprint": model_fingerprint(model)}
    try:
        from vllm_hook_plugins.core.fingerprints import chat_template_fingerprint, config_fingerprint
    except ImportError:
        return meta
    try:
        meta["config_fingerprint"] = config_fingerprint(model.config.to_dict())
    except (TypeError, ValueError):
        pass
    if tokenizer is not None:
        meta["chat_template_fingerprint"] = chat_template_fingerprint(
            getattr(tokenizer, "chat_template", None)
        )
    return meta


_ABSENT_TEMPLATE_DIGEST = hashlib.sha256(b"").hexdigest()


def is_absent_chat_template_fingerprint(fingerprint: str | None) -> bool:
    """Whether `fingerprint` denotes an absent chat template under the plugin recipe.

    A serving engine that exposes no chat template reports the fingerprint of an empty
    template, so a mismatch against such a value reflects exposure rather than divergence
    and comparisons should skip it. Uses the plugin recipe when `vllm_hook_plugins` is
    installed and falls back to the recipe's stable digest of an empty template otherwise.

    Args:
        fingerprint: The reported fingerprint, or None when none was reported.

    Returns:
        True when the fingerprint denotes an absent chat template.
    """
    if not fingerprint:
        return True
    try:
        from vllm_hook_plugins.core.fingerprints import chat_template_fingerprint
    except ImportError:
        return fingerprint in (f"sha256:{_ABSENT_TEMPLATE_DIGEST}", _ABSENT_TEMPLATE_DIGEST)
    return fingerprint in (chat_template_fingerprint(None), chat_template_fingerprint(""))
