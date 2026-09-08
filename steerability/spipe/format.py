"""The `spipe/1` manifest schema and archive packing.

The canonical form of a `.spipe` is a directory holding `spipe.json` and an optional
`artifacts/` store; a `.spipe` file is a zip of that directory. Zips are written
deterministically (sorted member names, stored uncompressed, fixed timestamps) and unpacked
with a zip-slip guard, symlink rejection, and a decompressed-size cap.
"""
from __future__ import annotations

import json
import logging
import zipfile
from pathlib import Path
from typing import Any, Mapping

from steerability.spipe.errors import SpipeFormatError

logger = logging.getLogger(__name__)

FORMAT_VERSION = "spipe/1"
MANIFEST_NAME = "spipe.json"
ARTIFACTS_DIR = "artifacts"

SIZE_CAP_BYTES = 20 * 1024**3

_TOP_LEVEL_KEYS = {"format", "created_at", "toolkit_version", "code_dependent", "model", "controls", "lock"}
_MODEL_KEYS = {"ref", "revision"}
_ENTRY_KEYS = {"method", "enabled", "args", "resolved"}
_RESOLVED_KEYS = {"method", "args", "artifacts", "origin"}
_ARTIFACT_RECORD_KEYS = {"id", "encoding", "type", "artifact_class", "source", "fit_digest", "provenance"}
_LOCK_KEYS = {
    "config_id", "recipe_id", "model_fingerprint", "tokenizer_fingerprint", "torch_dtype",
    "steer_backend_spec_hash", "fit", "seed", "versions",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SpipeFormatError(message)


def _check_keys(mapping: Mapping, allowed: set[str], where: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    _require(not unknown, f"{where}: unknown key(s) {unknown}; the spipe/1 schema rejects them.")


def _validate_resolved_entry(resolved: Mapping, where: str) -> None:
    _require(isinstance(resolved, Mapping), f"{where}: resolved entry must be an object.")
    _check_keys(resolved, _RESOLVED_KEYS, where)
    _require(isinstance(resolved.get("method"), str) and resolved["method"],
             f"{where}.method: required non-empty string.")
    _require(isinstance(resolved.get("args"), Mapping), f"{where}.args: required object.")
    artifacts = resolved.get("artifacts", {})
    _require(isinstance(artifacts, Mapping), f"{where}.artifacts: must be an object.")
    for name, record in artifacts.items():
        record_where = f"{where}.artifacts[{name!r}]"
        _require(isinstance(record, Mapping), f"{record_where}: must be an object.")
        _check_keys(record, _ARTIFACT_RECORD_KEYS, record_where)
        _require(isinstance(record.get("id"), str) and record["id"].startswith("sha256:"),
                 f"{record_where}.id: required 'sha256:<hex>' string.")
        _require(record.get("encoding") in ("tensors", "tree"),
                 f"{record_where}.encoding: must be 'tensors' or 'tree'.")
        _require(record.get("artifact_class") in ("direction", "calibrated", "opaque"),
                 f"{record_where}.artifact_class: must be 'direction', 'calibrated', or 'opaque'.")
    origin = resolved.get("origin")
    if origin is not None:
        _require(isinstance(origin, Mapping) and set(origin) <= {"method", "args"},
                 f"{where}.origin: must be null or an object with 'method' and 'args'.")


def validate_manifest(manifest: Any) -> None:
    """Validate a parsed `spipe.json` against the `spipe/1` schema.

    Args:
        manifest: The parsed manifest.

    Raises:
        SpipeFormatError: On any schema violation; the message names the offending field.
            An unsupported `format` value is reported first, naming the found and supported
            versions.
    """
    _require(isinstance(manifest, Mapping), "spipe.json: manifest must be a JSON object.")
    found = manifest.get("format")
    _require(
        found == FORMAT_VERSION,
        f"Unsupported spipe format {found!r}; this toolkit supports {FORMAT_VERSION!r}.",
    )
    _check_keys(manifest, _TOP_LEVEL_KEYS, "spipe.json")
    for key in ("created_at", "toolkit_version"):
        _require(isinstance(manifest.get(key), str), f"spipe.json.{key}: required string.")
    _require(isinstance(manifest.get("code_dependent"), bool), "spipe.json.code_dependent: required boolean.")

    model = manifest.get("model")
    _require(isinstance(model, Mapping), "spipe.json.model: required object.")
    _check_keys(model, _MODEL_KEYS, "spipe.json.model")
    _require(isinstance(model.get("ref"), str) and model["ref"], "spipe.json.model.ref: required non-empty string.")
    _require(model.get("revision") is None or isinstance(model["revision"], str),
             "spipe.json.model.revision: must be a string or null.")

    controls = manifest.get("controls")
    _require(isinstance(controls, list), "spipe.json.controls: required array.")
    for i, entry in enumerate(controls):
        where = f"spipe.json.controls[{i}]"
        _require(isinstance(entry, Mapping), f"{where}: must be an object.")
        _check_keys(entry, _ENTRY_KEYS, where)
        _require(isinstance(entry.get("method"), str) and "/" in entry.get("method", ""),
                 f"{where}.method: required '<category>_control/<name>' string.")
        _require(isinstance(entry.get("enabled"), bool), f"{where}.enabled: required boolean.")
        _require(isinstance(entry.get("args"), Mapping), f"{where}.args: required object.")
        resolved = entry.get("resolved")
        if resolved is None:
            continue
        if isinstance(resolved, list):
            _require(bool(resolved), f"{where}.resolved: an array of frozen entries must be non-empty.")
            for j, item in enumerate(resolved):
                _validate_resolved_entry(item, f"{where}.resolved[{j}]")
        else:
            _validate_resolved_entry(resolved, f"{where}.resolved")

    lock = manifest.get("lock")
    if lock is not None:
        _require(isinstance(lock, Mapping), "spipe.json.lock: must be an object or null.")
        _check_keys(lock, _LOCK_KEYS, "spipe.json.lock")
        for key in ("config_id", "recipe_id"):
            _require(isinstance(lock.get(key), str) and lock[key], f"spipe.json.lock.{key}: required string.")
        _require(lock.get("fit") in ("auto", "in_process"), "spipe.json.lock.fit: must be 'auto' or 'in_process'.")
        _require(isinstance(lock.get("versions"), Mapping), "spipe.json.lock.versions: required object.")


def write_manifest(manifest: Mapping, directory: str | Path) -> None:
    """Write `spipe.json` into `directory` (sorted keys, two-space indent, UTF-8)."""
    path = Path(directory) / MANIFEST_NAME
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, sort_keys=True, indent=2)
        handle.write("\n")


def read_manifest(directory: str | Path) -> dict:
    """Read and validate `spipe.json` from `directory`.

    Raises:
        SpipeFormatError: If the manifest is absent, is not valid JSON, or violates the
            schema.
    """
    path = Path(directory) / MANIFEST_NAME
    if not path.exists():
        raise SpipeFormatError(f"{directory} contains no {MANIFEST_NAME}; not a spipe bundle.")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SpipeFormatError(f"{path} is not valid JSON: {exc}.") from exc
    validate_manifest(manifest)
    return manifest


def pack_zip(directory: str | Path, target: str | Path) -> None:
    """Write `directory` as a deterministic zip at `target`.

    Members are written in sorted arcname order, stored uncompressed, with a fixed
    `(1980, 1, 1, 0, 0, 0)` timestamp and no extra fields. Equal directories therefore
    produce byte-equal files.

    Raises:
        SpipeFormatError: If `directory` contains a symlink.
    """
    directory = Path(directory)
    members = sorted(
        (path for path in directory.rglob("*") if path.is_file() or path.is_symlink()),
        key=lambda path: path.relative_to(directory).as_posix(),
    )
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_STORED) as archive:
        for path in members:
            if path.is_symlink():
                raise SpipeFormatError(f"Refusing to pack symlink {path} into a spipe archive.")
            arcname = path.relative_to(directory).as_posix()
            info = zipfile.ZipInfo(arcname, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())


def unpack_zip(archive_path: str | Path, dest: str | Path) -> None:
    """Extract a `.spipe` zip into `dest` with the format's safety guards.

    Every member name is normalized and rejected if absolute, containing `..`, or resolving
    outside `dest`; symlink members are rejected; the total decompressed size is capped at
    20 GB.

    Raises:
        SpipeFormatError: If the archive is not a zip, a member violates the guards, or the
            size cap is exceeded.
    """
    dest = Path(dest).resolve()
    if not zipfile.is_zipfile(archive_path):
        raise SpipeFormatError(f"{archive_path} is not a zip archive.")
    with zipfile.ZipFile(archive_path) as archive:
        total = sum(info.file_size for info in archive.infolist())
        if total > SIZE_CAP_BYTES:
            raise SpipeFormatError(
                f"Archive decompresses to {total} bytes, over the {SIZE_CAP_BYTES}-byte cap."
            )
        for info in archive.infolist():
            name = info.filename
            if name.startswith(("/", "\\")) or ".." in Path(name).parts or (len(name) > 1 and name[1] == ":"):
                raise SpipeFormatError(f"Archive member {name!r} escapes the extraction root.")
            mode = (info.external_attr >> 16) & 0o170000
            if mode == 0o120000:
                raise SpipeFormatError(f"Archive member {name!r} is a symlink; symlinks are rejected.")
            target = (dest / name).resolve()
            if not target.is_relative_to(dest):
                raise SpipeFormatError(f"Archive member {name!r} escapes the extraction root.")
        for info in archive.infolist():
            if info.is_dir():
                (dest / info.filename).mkdir(parents=True, exist_ok=True)
                continue
            target = dest / info.filename
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, open(target, "wb") as sink:
                while chunk := source.read(1 << 20):
                    sink.write(chunk)
