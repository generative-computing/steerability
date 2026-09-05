"""Artifact store encodings, ids, idempotence, and integrity."""
import json

import pytest
import torch

from steerability.algorithms.state_control.common.lowering import artifact_id_for
from steerability.spipe.errors import SpipeIntegrityError, SpipeSaveError
from steerability.spipe.store import ArtifactStore, tree_id_for


@pytest.fixture
def store(tmp_path):
    return ArtifactStore(tmp_path / "artifacts")


RECORD_FIELDS = {"type": "Tensor", "artifact_class": "opaque", "source": None,
                 "fit_digest": None, "provenance": {}, "type_meta": {}}


def test_tensor_id_matches_artifact_id_for(store):
    tensors = {"1": torch.randn(1, 8, dtype=torch.bfloat16), "3": torch.randn(1, 8)}
    record = store.put_tensors(tensors, dict(RECORD_FIELDS))
    expected_id, _ = artifact_id_for(tensors)
    assert record.id == expected_id

    # stored bytes hash back to the id (byte-compatibility with the plugin registry)
    store.verify(record.id)
    loaded = store.load_tensors(record.id)
    assert loaded["1"].dtype == torch.float32
    assert torch.allclose(loaded["3"], tensors["3"])


def test_tensor_write_idempotent(store):
    tensors = {"value": torch.ones(4)}
    first = store.put_tensors(tensors, dict(RECORD_FIELDS))
    second = store.put_tensors({"value": torch.ones(4)}, dict(RECORD_FIELDS))
    assert first.id == second.id
    assert store.ids() == [first.id]


def test_tree_id_stability_and_order_independence(tmp_path, store):
    src = tmp_path / "src"
    (src / "sub").mkdir(parents=True)
    (src / "a.txt").write_text("alpha")
    (src / "sub" / "b.txt").write_text("beta")
    first = tree_id_for(src)

    other = tmp_path / "other"
    (other / "sub").mkdir(parents=True)
    (other / "sub" / "b.txt").write_text("beta")
    (other / "a.txt").write_text("alpha")
    assert tree_id_for(other) == first

    record = store.put_tree(src, {**RECORD_FIELDS, "type": "CheckpointArtifact"})
    assert record.id == first
    assert (store.payload_path(record.id) / "sub" / "b.txt").read_text() == "beta"


def test_tree_symlink_rejected(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.txt").write_text("alpha")
    (src / "link").symlink_to(src / "a.txt")
    with pytest.raises(SpipeSaveError, match="symlink"):
        tree_id_for(src)


def test_corruption_raises_integrity_error(store):
    record = store.put_tensors({"value": torch.ones(4)}, dict(RECORD_FIELDS))
    tensor_file = store.root / record.id.replace(":", "-", 1) / "artifact.safetensors"
    tensor_file.write_bytes(tensor_file.read_bytes()[:-1] + b"\x00")
    with pytest.raises(SpipeIntegrityError, match="integrity"):
        store.verify(record.id)


def test_sidecar_dirname_mismatch_raises(store):
    record = store.put_tensors({"value": torch.ones(4)}, dict(RECORD_FIELDS))
    sidecar = store.root / record.id.replace(":", "-", 1) / "artifact.json"
    data = json.loads(sidecar.read_text())
    data["id"] = "sha256:" + "0" * 64
    sidecar.write_text(json.dumps(data))
    with pytest.raises(SpipeIntegrityError, match="named for"):
        store.record_for(record.id)


def test_missing_artifact_names_thin_hint(store):
    with pytest.raises(SpipeIntegrityError, match="artifact_store"):
        store.record_for("sha256:" + "a" * 64)
