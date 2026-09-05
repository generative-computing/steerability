"""Codec round-trips and security gates for the `.spipe` value codec."""
import json
import warnings

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from steerability.algorithms.core.internals.data import ContrastivePairs
from steerability.spipe.codec import CodeRef, DataRef, DecodeContext, EncodeContext, decode, digest_of, encode
from steerability.spipe.errors import SpipeCodeRefError, SpipeSaveError
from steerability.spipe.store import ArtifactStore

TINY_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def module_level_scorer(response, row):
    return float(len(response))


@pytest.fixture
def store(tmp_path):
    return ArtifactStore(tmp_path / "artifacts")


def roundtrip(value, store, *, allow_code=False):
    ctx = EncodeContext(store=store)
    encoded = encode(value, ctx)
    return decode(encoded, DecodeContext(store=store, allow_code=allow_code)), encoded


def test_plain_values_pass_through(store):
    value = {"a": 1, "b": [1.5, "x", None, True], "c": {"nested": [1, 2]}}
    decoded, encoded = roundtrip(value, store)
    assert decoded == value
    assert encoded == value


def test_nonstring_keys_roundtrip_via_map(store):
    value = {1: [0, 2], 5: [1]}
    decoded, encoded = roundtrip(value, store)
    assert decoded == value
    assert "$map" in encoded


def test_map_encoding_and_digest_are_key_order_independent():
    forward = encode({10: 1.0, 5: 2.0}, EncodeContext(store=None))
    backward = encode({5: 2.0, 10: 1.0}, EncodeContext(store=None))
    assert forward == backward
    assert digest_of({10: 1.0, 5: 2.0}) == digest_of({5: 2.0, 10: 1.0})


def test_dataclass_roundtrip(store):
    pairs = ContrastivePairs(positives=["a", "b"], negatives=["c", "d"], prompts=["p", "q"])
    decoded, encoded = roundtrip(pairs, store)
    assert encoded["$dc"].endswith("ContrastivePairs")
    assert decoded == pairs


def test_enum_and_dtype_roundtrip(store):
    from peft import PeftType

    decoded, _ = roundtrip(PeftType.LORA, store)
    assert decoded is PeftType.LORA
    decoded, _ = roundtrip(torch.bfloat16, store)
    assert decoded is torch.bfloat16


def test_tensor_roundtrip_via_store(store):
    tensor = torch.randn(3, 4)
    decoded, encoded = roundtrip(tensor, store)
    assert "$artifact" in encoded
    assert torch.allclose(decoded, tensor)


def test_steering_vector_roundtrip(store):
    from steerability.algorithms.state_control.common.steering_vector import SteeringVector

    vector = SteeringVector(
        model_type="llama",
        directions={1: torch.randn(1, 8), 3: torch.randn(1, 8)},
        explained_variances={1: 0.5, 3: 0.25},
        meta={"location": "layer_output"},
    )
    ctx = EncodeContext(store=store)
    encoded = encode(vector, ctx)
    decoded = decode(encoded, DecodeContext(store=store))
    assert decoded.model_type == "llama"
    assert sorted(decoded.directions) == [1, 3]
    assert torch.allclose(decoded.directions[3], vector.directions[3])
    assert decoded.explained_variances == {1: 0.5, 3: 0.25}
    assert decoded.meta == {"location": "layer_output"}


def test_direction_class_artifacts_decode_verified(store):
    from steerability.algorithms.state_control.common.sources import VerifiedPrecomputed
    from steerability.algorithms.state_control.common.steering_vector import SteeringVector

    vector = SteeringVector(model_type="llama", directions={1: torch.randn(1, 8)})
    ctx = EncodeContext(store=store)
    ctx.artifact_fields = {"artifact_class": "direction", "source": "ContrastiveFit", "fit_digest": "0" * 12}
    encoded = encode(vector, ctx)
    decoded = decode(encoded, DecodeContext(store=store, verify="strict"))
    assert isinstance(decoded, VerifiedPrecomputed)
    assert decoded.artifact_class == "direction"


def test_reserved_dollar_key_rejected(store):
    with pytest.raises(SpipeSaveError, match="reserved"):
        encode({"$evil": 1}, EncodeContext(store=store))


def test_lambda_rejected_with_naming_hint(store):
    with pytest.raises(SpipeSaveError, match="module-level name"):
        encode(lambda x: x, EncodeContext(store=store))


def test_partial_and_bound_method_rejected(store):
    import functools

    with pytest.raises(SpipeSaveError, match="module-level name"):
        encode(functools.partial(module_level_scorer, "x"), EncodeContext(store=store))
    with pytest.raises(SpipeSaveError, match="module-level name"):
        encode("abc".upper, EncodeContext(store=store))


def test_ref_gating_both_directions(store):
    ctx = EncodeContext(store=store)
    encoded = encode(module_level_scorer, ctx)
    assert encoded == {"$ref": f"{__name__}:module_level_scorer"}
    assert ctx.code_refs

    with pytest.raises(SpipeCodeRefError, match="allow_code"):
        decode(encoded, DecodeContext(store=store, allow_code=False))
    decoded = decode(encoded, DecodeContext(store=store, allow_code=True))
    assert decoded is module_level_scorer
    sentinel = decode(encoded, DecodeContext(store=store, allow_code=False, code_mode="sentinel"))
    assert isinstance(sentinel, CodeRef)
    with pytest.raises(SpipeCodeRefError):
        sentinel("x", {})


def test_sentinel_ref_never_imports_under_allow_code(store):
    # the digest-only decode path (code_mode="sentinel") must not import the target even when
    # allow_code is granted, so a $ref to an absent module still yields the inert CodeRef
    encoded = {"$ref": "steerability_absent_module_xyz:some_fn"}
    decoded = decode(encoded, DecodeContext(store=store, allow_code=True, code_mode="sentinel"))
    assert decoded == CodeRef("steerability_absent_module_xyz:some_fn")
    assert encode(decoded, EncodeContext(store=None)) == encoded


def test_dc_import_gating(store):
    encoded = {"$dc": "os.path.sep", "fields": {}}
    with pytest.raises(SpipeCodeRefError, match="allow_code"):
        decode(encoded, DecodeContext(store=store, allow_code=False))


def test_live_model_refused(store):
    import torch.nn as nn

    with pytest.raises(SpipeSaveError, match="name_or_path"):
        encode(nn.Linear(2, 2), EncodeContext(store=store))


def test_data_ref_roundtrip_kept(store):
    ref = DataRef(kind="hf", repo_id="org/data", split="train")
    ctx = EncodeContext(store=store)
    encoded = encode(ref, ctx)
    assert encoded == {"$data": {"kind": "hf", "repo_id": "org/data", "split": "train"}}
    kept = decode(encoded, DecodeContext(store=store, data_mode="keep"))
    assert kept == ref


def test_hf_dataset_encodes_opaque(store):
    from datasets import Dataset

    ds = Dataset.from_dict({"text": ["a", "b"]})
    encoded = encode(ds, EncodeContext(store=store))
    assert encoded["$data"]["kind"] == "opaque"
    assert encoded["$data"]["fingerprint"] == ds._fingerprint


def test_component_transform_roundtrip(store):
    from steerability.algorithms.state_control.common.transforms import AdditiveTransform, NormPreservingTransform

    transform = NormPreservingTransform(AdditiveTransform({1: torch.randn(1, 8)}, strength=2.5))
    ctx = EncodeContext(store=store)
    encoded = encode(transform, ctx)
    assert encoded["$component"] == "norm_preserving"
    assert encoded["inner"]["$component"] == "additive"
    decoded = decode(encoded, DecodeContext(store=store))
    assert isinstance(decoded, NormPreservingTransform)
    assert decoded.inner.strength == 2.5
    assert torch.allclose(decoded.inner.directions[1], transform.inner.directions[1])


def test_component_gate_roundtrip(store):
    from steerability.algorithms.state_control.common.gating import (
        Evidence,
        Gate,
        PerKeyThreshold,
        ProjectedCosineReadout,
    )

    directions = {1: torch.randn(8)}
    gate = Gate(
        Evidence((1,), ProjectedCosineReadout(directions), pooling="mean"),
        PerKeyThreshold(threshold=0.4, comparator="ge", aggregate="any"),
    )
    ctx = EncodeContext(store=store)
    encoded = encode(gate, ctx)
    assert encoded["$component"] == "gate"
    decoded = decode(encoded, DecodeContext(store=store))
    assert isinstance(decoded, Gate)
    assert decoded.evidence.layer_ids == (1,)
    assert decoded.rule.threshold == 0.4
    pooled = torch.randn(2, 8)
    assert torch.allclose(decoded.evidence.readout(pooled, 1), gate.evidence.readout(pooled, 1))


def test_callable_readout_gate_refused(store):
    from steerability.algorithms.state_control.common.gating import CallableReadout, Evidence, Gate, SumThreshold

    gate = Gate(Evidence((1,), CallableReadout(lambda pooled, lid: pooled[:, 0])), SumThreshold())
    with pytest.raises(ValueError, match="CallableReadout"):
        encode(gate, EncodeContext(store=store))


def test_selector_roundtrip(store):
    from steerability.algorithms.state_control.common.selectors import FractionalDepthSelector

    decoded, encoded = roundtrip(FractionalDepthSelector(fraction=0.4, minimum=1), store)
    assert encoded["$component"] == "fractional_depth"
    assert decoded.fraction == 0.4 and decoded.minimum == 1


def test_as_path_decodes_to_payload_path(store, tmp_path):
    src = tmp_path / "product"
    src.mkdir()
    (src / "weights.bin").write_bytes(b"abc")
    from steerability.algorithms.core.execution.payloads import CheckpointArtifact

    encoded = encode(CheckpointArtifact(path=str(src)), EncodeContext(store=store))
    assert encoded.get("as") == "path"
    decoded = decode(encoded, DecodeContext(store=store))
    assert (pytest.importorskip("pathlib").Path(decoded) / "weights.bin").read_bytes() == b"abc"


def test_pickle_backed_memory_gated_behind_allow_code(store):
    from steerability.algorithms.input_control.common.memory.pool import PoolMemory

    pool = PoolMemory(items=["a", "b"], metadata={"score": [1.0, 2.0]})
    encoded = encode(pool, EncodeContext(store=store))
    with pytest.raises(SpipeCodeRefError, match="pickled"):
        decode(encoded, DecodeContext(store=store, allow_code=False))
    decoded = decode(encoded, DecodeContext(store=store, allow_code=True))
    assert decoded.items == ["a", "b"]


def test_digest_mode_stable_across_roundtrip(store):
    pairs = ContrastivePairs(positives=["a", "b"], negatives=["c", "d"])
    fit_input = {"data": pairs, "scorer": module_level_scorer, "tensor": torch.ones(2, 2, dtype=torch.bfloat16)}
    before = digest_of(fit_input)

    ctx = EncodeContext(store=store)
    encoded = encode(fit_input, ctx)
    decoded = decode(encoded, DecodeContext(store=store, allow_code=False, code_mode="sentinel"))
    assert before == digest_of(decoded)


def test_unhandled_object_raises_in_strict_mode(store):
    class Opaque:
        pass

    with pytest.raises(SpipeSaveError, match="no serialized form"):
        encode(Opaque(), EncodeContext(store=store))
    # digest mode reduces to a type name instead
    assert digest_of(Opaque()) == digest_of(Opaque())


@pytest.fixture(scope="module")
def frozen_caa_spipe():
    from steerability.algorithms.core.steering_pipeline import SteeringPipeline
    from steerability.algorithms.state_control.caa.control import CAA

    model = AutoModelForCausalLM.from_pretrained(TINY_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(TINY_MODEL)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    caa = CAA(
        data={"positives": ["kind a", "kind b"], "negatives": ["mean a", "mean b"]},
        train_spec={"method": "mean_diff", "accumulate": "last_token"},
        layer_id=1,
    )
    pipeline = SteeringPipeline(model=model, tokenizer=tokenizer, controls=[caa], model_name_or_path=TINY_MODEL)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipeline.steer()
    return pipeline.to_spipe()


def test_manifest_record_governs_verification_wrap(tmp_path, frozen_caa_spipe):
    from steerability.algorithms.state_control.common.sources import VerifiedPrecomputed
    from steerability.spipe import SPipe

    thin = frozen_caa_spipe.save(tmp_path / "thin_dir", artifacts="thin")
    external = frozen_caa_spipe.save(tmp_path / "fat_dir") / "artifacts"
    # a shared external store may hold a sidecar written by another bundle for the same content
    (record,) = (
        record
        for entry in frozen_caa_spipe.manifest["controls"]
        for record in entry["resolved"]["artifacts"].values()
    )
    assert record["artifact_class"] == "direction"
    sidecar_path = external / record["id"].replace(":", "-", 1) / "artifact.json"
    sidecar = json.loads(sidecar_path.read_text())
    sidecar.update(artifact_class="opaque", source=None, fit_digest=None)
    sidecar_path.write_text(json.dumps(sidecar))

    rebuilt = SPipe.load(thin, artifact_store=external).pipeline()
    source = rebuilt.state_controls[0].steering_vector
    assert isinstance(source, VerifiedPrecomputed)
    assert source.artifact_class == "direction"
