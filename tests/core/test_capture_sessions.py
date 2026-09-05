"""Tests for capture-backed fitting and probe reads: the `capture_hidden` bridge, estimator
fitting through a capture-only session, `ProbeSet.read` through a session, and provenance
stamping. A wrapper double hides the in-process model and serves `capture` only, standing in
for a remote capture-capable session."""
import pytest
import torch

from steerability.algorithms.core.execution import BackendSpec
from steerability.algorithms.core.internals.capture import capture_hidden
from steerability.algorithms.core.internals.data import ContrastivePairs
from steerability.algorithms.core.internals.probes import ProbeFitSpec, ProbeSet, fit_probe
from steerability.algorithms.state_control.caa.control import CAA
from steerability.algorithms.state_control.common.estimators import MeanDifferenceEstimator
from steerability.algorithms.state_control.common.fit_specs import VectorTrainSpec
from steerability.backends.huggingface import HFBackend
from tests.utils.tiny_models import tiny_llama, wordlevel_tokenizer

PAIRS = ContrastivePairs(
    positives=["the cat sat on the mat", "the dog ran fast"],
    negatives=["the mat sat on the cat", "the dog sat on the mat"],
)
SPEC = VectorTrainSpec(method="mean_diff", accumulate="last_token", prompt_format="raw")


class _CaptureOnlySession:
    """Session double serving only `layout` and `capture`, like a remote capture backend."""

    def __init__(self, inner):
        self._inner = inner

    @property
    def layout(self):
        return self._inner.layout

    def capture(self, prompts, layers, mode, location="layer_output"):
        return self._inner.capture(prompts, layers, mode, location=location)


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return tiny_llama(num_layers=4, hidden=32, heads=4)


@pytest.fixture(scope="module")
def tokenizer():
    return wordlevel_tokenizer()


@pytest.fixture()
def capture_session(model, tokenizer):
    backend = HFBackend.adopt(BackendSpec(kind="huggingface"), lambda: model, lambda: tokenizer)
    with backend.open_session() as inner:
        yield _CaptureOnlySession(inner)


class TestCaptureHiddenBridge:

    def test_in_process_path_matches_session_path(self, model, tokenizer, capture_session):
        enc = tokenizer(["the cat sat", "the dog ran fast"], return_tensors="pt", padding=True)
        direct, direct_mask = capture_hidden(enc, model=model, location="layer_output")
        via_session, session_mask = capture_hidden(enc, session=capture_session, location="layer_output")
        assert torch.equal(direct_mask, session_mask)
        for layer in direct:
            assert torch.allclose(direct[layer], via_session[layer], atol=1e-6)

    def test_no_model_and_no_capture_raises(self):
        enc = {"input_ids": torch.ones(1, 3, dtype=torch.long)}
        with pytest.raises(ValueError, match="capture-capable session"):
            capture_hidden(enc, session=object())


class TestRemoteFitting:

    def test_mean_difference_fit_matches_in_process(self, model, tokenizer, capture_session):
        in_process = MeanDifferenceEstimator().fit(model, tokenizer, data=PAIRS, spec=SPEC)
        remote = MeanDifferenceEstimator().fit(None, tokenizer, data=PAIRS, spec=SPEC, session=capture_session)
        for layer in in_process.directions:
            assert torch.allclose(
                in_process.directions[layer], remote.directions[layer], atol=1e-5
            )

    def test_caa_steers_from_data_through_capture_session(self, tokenizer, capture_session):
        control = CAA(data=PAIRS, train_spec=SPEC, layer_id=1)
        control.steer(model=None, tokenizer=tokenizer, session=capture_session)
        assert control._steering_vector is not None
        assert control.export_intervention_spec() is not None

    def test_fit_probe_matches_in_process(self, model, tokenizer, capture_session):
        spec = ProbeFitSpec(method="mean_diff", pooling="mean", location="layer_input",
                            prompt_format="raw", candidate_layers=[1, 2])
        in_process = fit_probe(model, tokenizer, data=PAIRS, spec=spec)
        remote = fit_probe(None, tokenizer, data=PAIRS, spec=spec, session=capture_session)
        assert remote.layer_ids == in_process.layer_ids
        for layer in in_process.weights:
            assert torch.allclose(in_process.weights[layer], remote.weights[layer], atol=1e-5)
        assert abs(remote.bias - in_process.bias) < 1e-4


class TestProbeReadThroughSession:

    def test_read_via_session_matches_in_process(self, model, tokenizer, capture_session):
        spec = ProbeFitSpec(method="mean_diff", pooling="mean", location="layer_input",
                            prompt_format="raw", candidate_layers=[1, 2])
        probe_set = ProbeSet.fit(model, tokenizer, data={"topic": PAIRS}, spec=spec)
        enc = tokenizer(["the cat sat on the mat", "the dog"], return_tensors="pt", padding=True)

        in_process = probe_set.read(model, enc["input_ids"], enc["attention_mask"])
        via_session = probe_set.read(None, enc["input_ids"], enc["attention_mask"], session=capture_session)
        assert torch.allclose(in_process.scores["topic"], via_session.scores["topic"], atol=1e-4)
        assert torch.equal(in_process.decisions["topic"], via_session.decisions["topic"])

    def test_read_without_model_or_session_raises(self, model, tokenizer):
        spec = ProbeFitSpec(method="mean_diff", pooling="mean", location="layer_input",
                            prompt_format="raw", candidate_layers=[1])
        probe_set = ProbeSet.fit(model, tokenizer, data={"topic": PAIRS}, spec=spec)
        with pytest.raises(ValueError, match="capture-capable session"):
            probe_set.read(None, torch.ones(1, 3, dtype=torch.long))


class TestProvenanceStamps:

    def test_fitted_vector_records_fingerprints(self, model, tokenizer):
        vector = MeanDifferenceEstimator().fit(model, tokenizer, data=PAIRS, spec=SPEC)
        assert vector.meta["model_fingerprint"]
        assert vector.meta["config_fingerprint"].startswith("sha256:")
        assert vector.meta["chat_template_fingerprint"].startswith("sha256:")

    def test_meta_round_trips_through_save_load(self, model, tokenizer, tmp_path):
        vector = MeanDifferenceEstimator().fit(model, tokenizer, data=PAIRS, spec=SPEC)
        path = str(tmp_path / "vector.svec")
        vector.save(path)
        loaded = type(vector).load(path)
        assert loaded.meta == vector.meta

    def test_probe_meta_records_plugin_fingerprints(self, model, tokenizer):
        spec = ProbeFitSpec(method="mean_diff", pooling="mean", location="layer_input",
                            prompt_format="raw", candidate_layers=[1])
        probe = fit_probe(model, tokenizer, data=PAIRS, spec=spec)
        assert probe.meta["config_fingerprint"].startswith("sha256:")
        assert probe.meta["chat_template_fingerprint"].startswith("sha256:")
