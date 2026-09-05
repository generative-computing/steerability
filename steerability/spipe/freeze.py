"""Building an `SPipe` from a live `SteeringPipeline` (`to_spipe` orchestration).

The recipe section is the codec encoding of each control's constructor args, in pipeline
order. Freezing additionally walks each enabled control's `export_state()` and
`frozen_form()`, writes the exported artifacts into the bundle's store with provenance and
fit digests, and assembles the lock. The freeze walk over every enabled control runs before
any recipe encoding, so a recipe value with the same content as an exported artifact reuses
the artifact's record.
"""
from __future__ import annotations

import logging
import platform
import tempfile
from dataclasses import fields as dataclass_fields
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from steerability.algorithms.core.base_control import NotFreezableError
from steerability.spipe.codec import EncodeContext, digest_of, encode, encoded_size
from steerability.spipe.errors import SpipeSaveError
from steerability.spipe.format import ARTIFACTS_DIR, FORMAT_VERSION
from steerability.spipe.store import ArtifactStore

if TYPE_CHECKING:
    from steerability.algorithms.core.steering_pipeline import SteeringPipeline
    from steerability.spipe.spipe import SPipe

logger = logging.getLogger(__name__)

INLINE_LIMIT_BYTES = 1_000_000


def _toolkit_version() -> str:
    try:
        from importlib.metadata import version

        return version("steerability")
    except Exception:
        import steerability

        return getattr(steerability, "__version__", "unknown")


def _package_versions() -> dict[str, str]:
    import torch
    import transformers

    return {
        "steerability": _toolkit_version(),
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "python": platform.python_version(),
    }


def _resolve_model_ref(pipeline, model_ref: str | None) -> str:
    if model_ref is not None:
        return str(model_ref)
    if pipeline.model_name_or_path is not None:
        return str(pipeline.model_name_or_path)
    model = pipeline.model
    if model is not None:
        ref = getattr(model, "name_or_path", None) or getattr(
            getattr(model, "config", None), "_name_or_path", None
        )
        if ref:
            return str(ref)
    raise SpipeSaveError(
        "The pipeline has no resolvable model reference; pass model_ref= to to_spipe()."
    )


def _recipe_args_encoded(control, ctx: EncodeContext, path: str) -> dict:
    args = getattr(control, "args", None)
    if args is None:
        return {}
    encoded = {}
    for f in dataclass_fields(args):
        if not f.init:
            continue
        encoded[f.name] = encode(getattr(args, f.name), ctx, f"{path}.{f.name}")
    return encoded


def _artifact_class_of(value: Any) -> str:
    import torch

    from steerability.algorithms.core.internals.probes.probe import Probe
    from steerability.algorithms.core.internals.probes.probe_set import ProbeSet
    from steerability.algorithms.state_control.common.gating import Gate
    from steerability.algorithms.state_control.common.steering_vector import SteeringVector

    if isinstance(value, (Gate, Probe, ProbeSet)):
        return "calibrated"
    if isinstance(value, (SteeringVector, torch.Tensor)):
        return "direction"
    return "opaque"


def _collect_artifact_ids(encoded: Any, found: list[str]) -> None:
    if isinstance(encoded, dict):
        for key, value in encoded.items():
            if key == "$artifact" and isinstance(value, str):
                found.append(value)
            else:
                _collect_artifact_ids(value, found)
    elif isinstance(encoded, list):
        for item in encoded:
            _collect_artifact_ids(item, found)


def _freeze_control(control, ctx: EncodeContext, entry_path: str) -> Any:
    """Freeze one enabled control: encode its exported state and frozen form.

    Returns:
        The `resolved` manifest value (an object, an array of objects, or None for controls
        with nothing to pin).
    """
    from steerability.algorithms.core.registry import method_key_for

    state = control.export_state()
    fits = list(control.steer_fits())
    if not state and not fits:
        return None

    fit_identity = control.fit_identity()
    fit_digest = digest_of(fit_identity) if fit_identity is not None else None
    fit_source = fits[0][0] if fits else (type(control).__name__ if fit_digest is not None else None)

    # encode each exported state value with its metadata installed, giving artifact sidecars
    # and manifest records the fit provenance; content-equal values re-encoded later inside
    # the frozen args reuse these records (first write wins)
    artifacts: dict[str, dict] = {}
    remaining_fits = list(fits)
    for name, value in state.items():
        artifact_class = _artifact_class_of(value)
        source = None
        digest = None
        matched = next((fit for fit in remaining_fits if fit[1] == artifact_class), None)
        if matched is not None:
            remaining_fits.remove(matched)
            source, digest = matched[0], fit_digest
        elif fit_digest is not None and not fits:
            source, digest = fit_source, fit_digest
        ctx.artifact_fields = {
            "artifact_class": artifact_class,
            "source": source,
            "fit_digest": digest,
        }
        encoded_value = encode(value, ctx, f"{entry_path}.state[{name!r}]")
        ctx.artifact_fields = {}
        ids: list[str] = []
        _collect_artifact_ids(encoded_value, ids)
        if ids:
            artifacts[name] = ctx.records[ids[0]].manifest_entry()

    forms = control.frozen_form(state)
    if isinstance(forms, tuple):
        forms = [forms]

    method_key = method_key_for(type(control))
    resolved_entries = []
    for form_index, (frozen_key, frozen_args) in enumerate(forms):
        encoded_args = {
            name: encode(value, ctx, f"{entry_path}.resolved[{form_index}].{name}")
            for name, value in frozen_args.items()
        }
        origin = None
        if frozen_key != method_key:
            origin = {"method": method_key, "args": None}  # args filled in by the caller
        resolved_entries.append({
            "method": frozen_key,
            "args": encoded_args,
            "artifacts": artifacts if form_index == 0 else {},
            "origin": origin,
        })

    return resolved_entries[0] if len(resolved_entries) == 1 else resolved_entries


def build_spipe(pipeline: SteeringPipeline, *, freeze: bool | None = None, model_ref: str | None = None) -> "SPipe":
    """Build an `SPipe` from a pipeline.

    Args:
        pipeline: The `SteeringPipeline` to serialize.
        freeze: Freeze the resolution (default: the pipeline's steered state). `False` forces
            a recipe-only spipe from a steered pipeline.
        model_ref: Explicit model reference, overriding the pipeline's.

    Returns:
        The built `SPipe` (backed by a temporary store directory until saved).

    Raises:
        SpipeSaveError: If the model reference is unresolvable, a control is unregistered or
            cannot serialize, freezing is requested on an unsteered pipeline, or a control
            cannot freeze.
    """
    from steerability.algorithms.core.identity import config_descriptor_from_controls, config_digest
    from steerability.algorithms.core.registry import method_key_for
    from steerability.spipe.spipe import SPipe

    if freeze is None:
        freeze = bool(pipeline._is_steered)
    if freeze and not pipeline._is_steered:
        raise SpipeSaveError("freeze=True requires a steered pipeline; call steer() first.")

    ref = _resolve_model_ref(pipeline, model_ref)
    revision = pipeline.hf_model_kwargs.get("revision") if pipeline.hf_model_kwargs else None

    controls = [
        *pipeline.structural_controls, *pipeline.input_controls,
        *pipeline.state_controls, *pipeline.output_controls,
    ]

    temp = tempfile.TemporaryDirectory(prefix="spipe-")
    base_dir = Path(temp.name)
    store = ArtifactStore(base_dir / ARTIFACTS_DIR)

    provenance: dict[str, Any] = {}
    spec = pipeline._resolve_backend_spec(pipeline.backend)
    provenance["backend_spec_hash"] = spec.spec_hash
    model = pipeline.model
    default_model_type = None
    if model is not None:
        from steerability.algorithms.core.internals.fingerprint import artifact_provenance_meta

        meta = artifact_provenance_meta(model, pipeline.tokenizer)
        provenance["model_fingerprint"] = meta.get("model_fingerprint")
        provenance["tokenizer_fingerprint"] = meta.get("chat_template_fingerprint")
        default_model_type = getattr(model.config, "model_type", None)

    ctx = EncodeContext(store=store, provenance=provenance, default_model_type=default_model_type)

    # the freeze walk over every enabled control runs before any recipe encoding: exported
    # artifacts land in the store with their fit provenance first, and a content-equal recipe
    # value of any control reuses the metadata-rich record
    resolved_by_index: dict[int, Any] = {}
    for i, control in enumerate(controls):
        entry_path = f"controls[{i}]"
        if getattr(control, "gate_driven_externally", False):
            raise SpipeSaveError(
                f"{entry_path}: {type(control).__name__} follows an externally driven shared "
                "gate, an in-memory relationship that does not serialize; save the driving "
                "pipeline without the follower, or gate the control directly."
            )
        if freeze and control.enabled:
            try:
                resolved_by_index[i] = _freeze_control(control, ctx, entry_path)
            except NotFreezableError as exc:
                raise SpipeSaveError(
                    f"{entry_path}: {exc} The pipeline can still be saved with freeze=False."
                ) from exc

    entries = []
    for i, control in enumerate(controls):
        entry_path = f"controls[{i}]"
        resolved = resolved_by_index.get(i)
        entry: dict[str, Any] = {
            "method": method_key_for(type(control)),
            "enabled": bool(control.enabled),
            "args": _recipe_args_encoded(control, ctx, f"{entry_path}.args"),
            "resolved": resolved,
        }
        if resolved is not None:
            for item in (resolved if isinstance(resolved, list) else [resolved]):
                if item.get("origin") is not None:
                    item["origin"]["args"] = entry["args"]
        size = encoded_size(entry)
        if size > INLINE_LIMIT_BYTES:
            raise SpipeSaveError(
                f"{entry_path}: the entry's inline JSON is {size} bytes, over the "
                f"{INLINE_LIMIT_BYTES}-byte limit; convert large dataset fields to a $data "
                "reference (DataRef), or save the fitted pipeline frozen and share that."
            )
        entries.append(entry)

    descriptor = config_descriptor_from_controls(controls)
    config_id = config_digest(descriptor)
    recipe_id = config_digest({"model": ref, "controls": descriptor["controls"]})

    lock = None
    if freeze:
        lock = {
            "config_id": config_id,
            "recipe_id": recipe_id,
            "model_fingerprint": provenance.get("model_fingerprint"),
            "tokenizer_fingerprint": provenance.get("tokenizer_fingerprint"),
            "torch_dtype": str(model.dtype).removeprefix("torch.") if model is not None else None,
            "steer_backend_spec_hash": spec.spec_hash,
            "fit": pipeline.fit,
            "seed": None,
            "versions": _package_versions(),
        }

    manifest = {
        "format": FORMAT_VERSION,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "toolkit_version": _toolkit_version(),
        "code_dependent": bool(ctx.code_refs),
        "model": {"ref": ref, "revision": revision},
        "controls": entries,
        "lock": lock,
    }

    return SPipe(manifest, store=store, base_dir=base_dir, allow_code=True, _temp=temp)
