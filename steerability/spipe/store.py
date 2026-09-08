"""Content-addressed artifact store backing frozen `.spipe` bundles.

The store is a directory of artifact directories, each named by its content id. Two encodings
exist: `"tensors"` (a single `artifact.safetensors`, id and bytes produced by `artifact_id_for`
from `state_control/common/lowering.py`, byte-compatible with the vLLM-Hook plugin registry)
and `"tree"` (a directory copied verbatim under `payload/`, id a SHA-256 over the sorted
relative paths and per-file digests). Every artifact directory carries an `artifact.json`
sidecar duplicating its manifest record plus type-specific reconstruction metadata, which
keeps a detached artifact directory self-describing. Writes are idempotent and reads verify the
content hash.
"""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import torch

from steerability.spipe.errors import SpipeIntegrityError, SpipeSaveError

logger = logging.getLogger(__name__)

TENSOR_FILE = "artifact.safetensors"
SIDECAR_FILE = "artifact.json"
PAYLOAD_DIR = "payload"


def _dir_name(artifact_id: str) -> str:
    """Directory name for an artifact id (`sha256:<hex>` becomes `sha256-<hex>`)."""
    return artifact_id.replace(":", "-", 1)


def _artifact_id_from_dir_name(name: str) -> str:
    """Artifact id for a store directory name."""
    return name.replace("-", ":", 1)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_id_for(root: Path) -> str:
    """The content id of a directory tree.

    The id is `"sha256:" + sha256` over the concatenation of
    `f"{posix_relpath}\\n{sha256_of_file_hex}\\n"` for every file, sorted by posix relative
    path. Symlinks are rejected.

    Args:
        root: Directory to hash.

    Returns:
        The `sha256:<hex>` id.

    Raises:
        SpipeSaveError: If `root` is not a directory, is empty, or contains a symlink.
    """
    root = Path(root)
    if not root.is_dir():
        raise SpipeSaveError(f"Tree artifact source {root} is not a directory.")
    entries: list[tuple[str, str]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise SpipeSaveError(f"Tree artifact source contains a symlink: {path}.")
        if path.is_file():
            entries.append((path.relative_to(root).as_posix(), _file_sha256(path)))
    if not entries:
        raise SpipeSaveError(f"Tree artifact source {root} contains no files.")
    digest = hashlib.sha256()
    for relpath, file_hash in entries:
        digest.update(f"{relpath}\n{file_hash}\n".encode("utf-8"))
    return "sha256:" + digest.hexdigest()


def tensors_payload(tensors: Mapping[str, torch.Tensor]) -> tuple[str, bytes]:
    """The content id and serialized safetensors bytes of a tensor payload.

    Uses the id algorithm of `artifact_id_for` (float32, contiguous, CPU, sorted names,
    SHA-256 over the safetensors serialization), which makes tensor artifact directories
    byte-compatible with the vLLM-Hook plugin registry.

    Args:
        tensors: Mapping from tensor name to tensor.

    Returns:
        The `sha256:<hex>` id and the serialized bytes.
    """
    import safetensors.torch

    from steerability.algorithms.state_control.common.lowering import artifact_id_for

    artifact_id, prepared = artifact_id_for(tensors)
    data = safetensors.torch.save({name: prepared[name] for name in sorted(prepared)})
    return artifact_id, data


@dataclass
class ArtifactRecord:
    """One artifact's manifest record.

    Attributes:
        id: Content-addressed artifact id (`sha256:<hex>`).
        encoding: `"tensors"` or `"tree"`.
        type: Name of the producing class (e.g. `"SteeringVector"`, `"Probe"`).
        artifact_class: `"direction"`, `"calibrated"`, or `"opaque"`.
        source: Fit-source class name, or None for recipe-supplied artifacts.
        fit_digest: 12-hex-character digest of the producing fit identity, or None.
        provenance: Producing-side fingerprints (`backend_spec_hash`, `model_fingerprint`,
            `tokenizer_fingerprint`), each possibly None.
        type_meta: Type-specific reconstruction metadata (sidecar only, excluded from the
            manifest record).
    """

    id: str
    encoding: str
    type: str
    artifact_class: str = "opaque"
    source: str | None = None
    fit_digest: str | None = None
    provenance: dict = field(default_factory=dict)
    type_meta: dict = field(default_factory=dict)

    def manifest_entry(self) -> dict:
        """The record as it appears in `spipe.json` (without `type_meta`)."""
        return {
            "id": self.id,
            "encoding": self.encoding,
            "type": self.type,
            "artifact_class": self.artifact_class,
            "source": self.source,
            "fit_digest": self.fit_digest,
            "provenance": {
                "backend_spec_hash": self.provenance.get("backend_spec_hash"),
                "model_fingerprint": self.provenance.get("model_fingerprint"),
                "tokenizer_fingerprint": self.provenance.get("tokenizer_fingerprint"),
            },
        }

    def sidecar_entry(self) -> dict:
        """The record as written to the artifact's `artifact.json` sidecar."""
        return {**self.manifest_entry(), "type_meta": self.type_meta}

    @classmethod
    def from_mapping(cls, data: Mapping) -> "ArtifactRecord":
        return cls(
            id=data["id"],
            encoding=data["encoding"],
            type=data["type"],
            artifact_class=data.get("artifact_class", "opaque"),
            source=data.get("source"),
            fit_digest=data.get("fit_digest"),
            provenance=dict(data.get("provenance") or {}),
            type_meta=dict(data.get("type_meta") or {}),
        )


class ArtifactStore:
    """Directory-backed store of content-addressed artifacts.

    Args:
        root: The store directory (`artifacts/` inside a spipe, or any external directory for
            thin exports). Created on first write.
        resolver: Optional callable mapping an artifact id to the directory holding that
            artifact, used when artifacts live outside `root` (a thin export's external
            store).
    """

    def __init__(self, root: str | Path, resolver: Callable[[str], Path] | None = None):
        self.root = Path(root)
        self._resolver = resolver

    def _dir_for(self, artifact_id: str) -> Path:
        local = self.root / _dir_name(artifact_id)
        if local.exists() or self._resolver is None:
            return local
        resolved = Path(self._resolver(artifact_id))
        return resolved if resolved.exists() else local

    def has(self, artifact_id: str) -> bool:
        """Whether the store holds `artifact_id`."""
        return (self._dir_for(artifact_id) / SIDECAR_FILE).exists()

    def ids(self) -> list[str]:
        """Sorted ids of every artifact under `root`."""
        if not self.root.is_dir():
            return []
        return sorted(
            _artifact_id_from_dir_name(child.name)
            for child in self.root.iterdir()
            if child.is_dir() and (child / SIDECAR_FILE).exists()
        )

    def _write_sidecar(self, directory: Path, record: ArtifactRecord) -> None:
        with open(directory / SIDECAR_FILE, "w", encoding="utf-8") as handle:
            json.dump(record.sidecar_entry(), handle, sort_keys=True, indent=2)

    def put_tensors(self, tensors: Mapping[str, torch.Tensor], record_fields: dict) -> ArtifactRecord:
        """Write a tensor payload as a `"tensors"` artifact, idempotently.

        Args:
            tensors: Mapping from tensor name to tensor.
            record_fields: Record fields other than `id` and `encoding` (`type`,
                `artifact_class`, `source`, `fit_digest`, `provenance`, `type_meta`).

        Returns:
            The artifact record.
        """
        artifact_id, data = tensors_payload(tensors)
        record = ArtifactRecord(id=artifact_id, encoding="tensors", **record_fields)
        directory = self.root / _dir_name(artifact_id)
        if not (directory / TENSOR_FILE).exists():
            directory.mkdir(parents=True, exist_ok=True)
            (directory / TENSOR_FILE).write_bytes(data)
            self._write_sidecar(directory, record)
        return record

    def put_tree(self, source: str | Path, record_fields: dict) -> ArtifactRecord:
        """Copy a directory as a `"tree"` artifact, idempotently.

        Args:
            source: Directory whose contents become `payload/`. Symlinks are rejected.
            record_fields: Record fields other than `id` and `encoding`.

        Returns:
            The artifact record.

        Raises:
            SpipeSaveError: If `source` is missing, empty, or contains a symlink.
        """
        source = Path(source)
        artifact_id = tree_id_for(source)
        record = ArtifactRecord(id=artifact_id, encoding="tree", **record_fields)
        directory = self.root / _dir_name(artifact_id)
        if not (directory / PAYLOAD_DIR).exists():
            directory.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, directory / PAYLOAD_DIR, symlinks=False)
            self._write_sidecar(directory, record)
        return record

    def record_for(self, artifact_id: str) -> ArtifactRecord:
        """The sidecar record of a stored artifact.

        Raises:
            SpipeIntegrityError: If the artifact is absent or the sidecar's `id` does not
                match the directory name.
        """
        directory = self._dir_for(artifact_id)
        sidecar = directory / SIDECAR_FILE
        if not sidecar.exists():
            raise SpipeIntegrityError(
                f"Artifact {artifact_id} is not in the store at {self.root}"
                + (" (no external resolver matched)" if self._resolver else
                   "; for a thin bundle, pass artifact_store= to load()")
                + "."
            )
        record = ArtifactRecord.from_mapping(json.loads(sidecar.read_text(encoding="utf-8")))
        if record.id != artifact_id:
            raise SpipeIntegrityError(
                f"Artifact sidecar at {directory} records id {record.id} but the directory "
                f"is named for {artifact_id}."
            )
        return record

    def verify(self, artifact_id: str) -> None:
        """Verify a stored artifact's bytes against its content id.

        Raises:
            SpipeIntegrityError: If the artifact is absent, the sidecar id mismatches, or the
                recomputed content hash differs from `artifact_id`.
        """
        record = self.record_for(artifact_id)
        directory = self._dir_for(artifact_id)
        if record.encoding == "tensors":
            data = (directory / TENSOR_FILE).read_bytes()
            actual = "sha256:" + hashlib.sha256(data).hexdigest()
        else:
            actual = tree_id_for(directory / PAYLOAD_DIR)
        if actual != artifact_id:
            raise SpipeIntegrityError(
                f"Artifact {artifact_id} failed integrity verification (content hashes to "
                f"{actual})."
            )

    def load_tensors(self, artifact_id: str) -> dict[str, torch.Tensor]:
        """Load and verify a `"tensors"` artifact.

        Returns:
            Mapping from tensor name to float32 CPU tensor.
        """
        import safetensors.torch

        self.verify(artifact_id)
        directory = self._dir_for(artifact_id)
        return safetensors.torch.load_file(str(directory / TENSOR_FILE))

    def payload_path(self, artifact_id: str) -> Path:
        """The verified `payload/` path of a `"tree"` artifact."""
        self.verify(artifact_id)
        return self._dir_for(artifact_id) / PAYLOAD_DIR

    def size_of(self, artifact_id: str) -> int:
        """Total on-disk bytes of an artifact's content (sidecar excluded)."""
        directory = self._dir_for(artifact_id)
        record = self.record_for(artifact_id)
        if record.encoding == "tensors":
            return (directory / TENSOR_FILE).stat().st_size
        return sum(p.stat().st_size for p in (directory / PAYLOAD_DIR).rglob("*") if p.is_file())

    def copy_into(self, dest_root: str | Path, artifact_ids: list[str]) -> None:
        """Copy the named artifacts into another store directory (fat export)."""
        dest_root = Path(dest_root)
        for artifact_id in artifact_ids:
            source = self._dir_for(artifact_id)
            dest = dest_root / _dir_name(artifact_id)
            if not dest.exists():
                dest.mkdir(parents=True, exist_ok=True)
                shutil.copytree(source, dest, symlinks=False, dirs_exist_ok=True)


def save_object_tree(save_fn: Callable[[Path], Any], store: ArtifactStore, record_fields: dict) -> ArtifactRecord:
    """Save an object into the store as a `"tree"` artifact via its own `save(path)` method.

    Args:
        save_fn: Callable writing the object into a directory (or file inside it).
        store: The destination store.
        record_fields: Record fields other than `id` and `encoding`.

    Returns:
        The artifact record.
    """
    import tempfile

    with tempfile.TemporaryDirectory(prefix="spipe-artifact-") as tmp:
        save_fn(Path(tmp))
        return store.put_tree(tmp, record_fields)
