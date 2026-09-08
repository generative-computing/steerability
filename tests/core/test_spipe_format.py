"""Manifest schema validation, archive packing, and directory save and verify behavior for `spipe/1`."""
import json
import shutil
import warnings
import zipfile

import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer

from steerability.algorithms.core.steering_pipeline import SteeringPipeline
from steerability.algorithms.state_control.caa.control import CAA
from steerability.spipe import SPipe
from steerability.spipe.errors import SpipeFormatError, SpipeSaveError
from steerability.spipe.format import pack_zip, unpack_zip, validate_manifest
from steerability.spipe.freeze import _package_versions

TINY_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"
KIND = {"positives": ["kind a", "kind b"], "negatives": ["mean a", "mean b"]}
CALM = {"positives": ["calm a", "calm b"], "negatives": ["angry a", "angry b"]}


def minimal_manifest(**overrides):
    manifest = {
        "format": "spipe/1",
        "created_at": "2026-08-25T00:00:00Z",
        "toolkit_version": "0.5.0",
        "code_dependent": False,
        "model": {"ref": "org/model", "revision": None},
        "controls": [],
        "lock": None,
    }
    manifest.update(overrides)
    return manifest


def test_minimal_manifest_validates():
    validate_manifest(minimal_manifest())


def test_lock_versions_records_the_toolkit_under_its_package_name():
    versions = _package_versions()
    assert "steerability" in versions
    lock = {
        "config_id": "cfg",
        "recipe_id": "rec",
        "model_fingerprint": None,
        "tokenizer_fingerprint": None,
        "torch_dtype": None,
        "steer_backend_spec_hash": None,
        "fit": "auto",
        "seed": None,
        "versions": versions,
    }
    validate_manifest(minimal_manifest(lock=lock))


def test_version_refusal_names_versions():
    with pytest.raises(SpipeFormatError, match=r"'spipe/2'.*'spipe/1'"):
        validate_manifest(minimal_manifest(format="spipe/2"))


def test_unknown_top_level_key_rejected():
    with pytest.raises(SpipeFormatError, match="unknown key"):
        validate_manifest(minimal_manifest(extra=1))


def test_unknown_entry_key_rejected():
    entry = {"method": "state_control/caa", "enabled": True, "args": {}, "resolved": None, "extra": 1}
    with pytest.raises(SpipeFormatError, match=r"controls\[0\].*unknown key"):
        validate_manifest(minimal_manifest(controls=[entry]))


def test_resolved_object_and_array_forms():
    resolved = {"method": "state_control/caa", "args": {}, "artifacts": {}, "origin": None}
    entry = {"method": "state_control/caa", "enabled": True, "args": {}, "resolved": resolved}
    validate_manifest(minimal_manifest(controls=[entry]))
    entry_list = dict(entry, resolved=[resolved, dict(resolved)])
    validate_manifest(minimal_manifest(controls=[entry_list]))
    with pytest.raises(SpipeFormatError, match="non-empty"):
        validate_manifest(minimal_manifest(controls=[dict(entry, resolved=[])]))


def test_artifact_record_validation():
    record = {"id": "notahash", "encoding": "tensors", "type": "SteeringVector",
              "artifact_class": "direction", "source": None, "fit_digest": None, "provenance": {}}
    resolved = {"method": "state_control/caa", "args": {}, "artifacts": {"v": record}, "origin": None}
    entry = {"method": "state_control/caa", "enabled": True, "args": {}, "resolved": resolved}
    with pytest.raises(SpipeFormatError, match="sha256"):
        validate_manifest(minimal_manifest(controls=[entry]))


def test_zip_determinism(tmp_path):
    src = tmp_path / "bundle"
    (src / "artifacts").mkdir(parents=True)
    (src / "spipe.json").write_text(json.dumps(minimal_manifest()))
    (src / "artifacts" / "blob").write_bytes(b"payload")
    pack_zip(src, tmp_path / "a.spipe")
    pack_zip(src, tmp_path / "b.spipe")
    assert (tmp_path / "a.spipe").read_bytes() == (tmp_path / "b.spipe").read_bytes()


def test_zip_slip_rejected(tmp_path):
    evil = tmp_path / "evil.spipe"
    with zipfile.ZipFile(evil, "w") as archive:
        archive.writestr("../outside.txt", "boom")
    with pytest.raises(SpipeFormatError, match="escapes"):
        unpack_zip(evil, tmp_path / "dest")


def test_zip_symlink_member_rejected(tmp_path):
    evil = tmp_path / "evil.spipe"
    with zipfile.ZipFile(evil, "w") as archive:
        info = zipfile.ZipInfo("link")
        info.external_attr = (0o120777 << 16)
        archive.writestr(info, "/etc/passwd")
    with pytest.raises(SpipeFormatError, match="symlink"):
        unpack_zip(evil, tmp_path / "dest")


def test_not_a_zip_rejected(tmp_path):
    bogus = tmp_path / "bogus.spipe"
    bogus.write_text("not a zip")
    with pytest.raises(SpipeFormatError, match="not a zip"):
        unpack_zip(bogus, tmp_path / "dest")


@pytest.fixture(scope="module")
def tiny_model():
    model = AutoModelForCausalLM.from_pretrained(TINY_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(TINY_MODEL)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def frozen_spipe(model, tokenizer, *datasets):
    """A frozen spipe with one fitted CAA vector per dataset."""
    controls = [
        CAA(data=data, train_spec={"method": "mean_diff", "accumulate": "last_token"}, layer_id=1)
        for data in datasets
    ]
    pipeline = SteeringPipeline(model=model, tokenizer=tokenizer, controls=controls, model_name_or_path=TINY_MODEL)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipeline.steer()
    return pipeline.to_spipe()


def test_in_place_save_honours_artifacts(tmp_path, tiny_model):
    spipe = frozen_spipe(*tiny_model, KIND)
    thin = spipe.save(tmp_path / "thin_dir", artifacts="thin")
    external = spipe.save(tmp_path / "fat_dir") / "artifacts"
    assert not (thin / "artifacts").exists()

    # a fat save onto the thin directory embeds the artifacts the store resolves externally
    SPipe.load(thin, artifact_store=external).save(thin, artifacts="fat")
    assert (thin / "artifacts").is_dir()
    reloaded = SPipe.load(thin)
    report = reloaded.verify()
    assert report.ok
    assert not any("thin" in message for message in report.warnings)

    # a thin save onto a directory that embeds artifacts would have to delete them
    with pytest.raises(SpipeSaveError, match="thin"):
        reloaded.save(thin, artifacts="thin")
    assert (thin / "artifacts").is_dir()



def test_verify_classifies_availability_per_artifact(tmp_path, tiny_model):
    spipe = frozen_spipe(*tiny_model, KIND, CALM)
    saved = spipe.save(tmp_path / "partial")
    artifact_dirs = sorted(child for child in (saved / "artifacts").iterdir() if child.is_dir())
    assert len(artifact_dirs) == 2
    shutil.rmtree(artifact_dirs[0])
    missing_id = artifact_dirs[0].name.replace("-", ":", 1)

    report = SPipe.load(saved).verify()
    assert report.ok
    assert not report.errors
    (warning,) = [message for message in report.warnings if "thin" in message]
    assert "1 of 2" in warning
    assert missing_id in warning
