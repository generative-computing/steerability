"""The staged steer on engine backends: phase partition and per-phase order, the free
protocol (weights gone before the engine boots, retention raises), structural artifact
handoff, and the capture smoke-test degradation path.

Engine paths run against a fake backend registered by monkeypatching
`resolve_backend_class`, since CI has no vLLM.
"""
import os
import weakref

import pytest
import torch

from aisteer360.algorithms.core.execution import (
    BackendSpec,
    Capability,
    CaptureResult,
    CheckpointArtifact,
    ModelAccess,
    ModelFacts,
    PreparedPrompt,
    Requirements,
)
from aisteer360.algorithms.core.steering_pipeline import SteeringPipeline
from aisteer360.algorithms.input_control.base import InputControl
from aisteer360.algorithms.output_control.base import OutputControl
from aisteer360.algorithms.structural_control.base import StructuralControl
from tests.utils.tiny_models import tiny_llama, wordlevel_tokenizer

HIDDEN = 16
LAYERS = 2


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    """A saved tiny model plus tokenizer, loadable as the stage's model reference."""
    path = tmp_path_factory.mktemp("tiny-llama")
    torch.manual_seed(0)
    tiny_llama(num_layers=LAYERS, hidden=HIDDEN, heads=2).save_pretrained(path)
    wordlevel_tokenizer().save_pretrained(path)
    return str(path)


class FakeEngineSession:
    """Engine session double: layout facts, recorded capture, no live model."""

    def __init__(self, backend):
        self._backend = backend
        self.closed = False

    @property
    def tokenizer(self):
        return self._backend.tokenizer

    @property
    def layout(self) -> ModelFacts:
        return ModelFacts(
            num_layers=LAYERS, hidden_size=HIDDEN, num_attention_heads=2, head_dim=HIDDEN // 2,
            dtype="float32", model_fingerprint="0" * 16, model_type="llama",
            model_ref=self._backend.spec.model,
        )

    def close(self):
        self.closed = True

    def capture(self, prompts, layers, mode, location="layer_output"):
        type(self._backend).events.append("engine-capture")
        if type(self._backend).capture_fails:
            raise RuntimeError("capture transport down")
        generator = torch.Generator().manual_seed(0)
        n = len(prompts)
        hidden = {
            layer: torch.randn(n, 1, HIDDEN, generator=generator)
            if mode == "all_tokens" else torch.randn(n, HIDDEN, generator=generator)
            for layer in layers
        }
        return CaptureResult(
            hidden=hidden, attention_mask=torch.ones(n, 1, dtype=torch.long),
            mode=mode, location=location,
        )

    def generate(self, items, params):
        raise AssertionError("steer-phase mocks do not generate")

    def score(self, items, params):
        raise AssertionError("steer-phase mocks do not score")


class FakeEngineBackend:
    """Engine backend double recording boots, releases, and staged payloads."""

    instances: list = []
    events: list = []
    capture_fails: bool = False
    boot_observer = None

    def __init__(self, spec, artifacts=()):
        self.spec = spec
        self.artifacts = tuple(artifacts)
        self.released = False
        self.staged_payloads: dict = {}
        self.tokenizer = wordlevel_tokenizer()
        self._discovery = None
        type(self).instances.append(self)
        type(self).events.append("boot")
        if type(self).boot_observer is not None:
            type(self).boot_observer()

    @classmethod
    def capabilities_for_spec(cls, spec):
        from aisteer360.backends.vllm import VLLMBackend, VLLMServeBackend

        backend_cls = VLLMServeBackend if spec.kind == "vllm-serve" else VLLMBackend
        return backend_cls.capabilities_for_spec(spec)

    def open_session(self):
        return FakeEngineSession(self)

    def stage_artifacts(self, payloads):
        self.staged_payloads.update(payloads)

    def release(self):
        self.released = True
        type(self).events.append("release")

    @classmethod
    def reset(cls):
        cls.instances = []
        cls.events = []
        cls.capture_fails = False
        cls.boot_observer = None


@pytest.fixture
def fake_engine(monkeypatch):
    import aisteer360.algorithms.core.execution.backend as backend_module
    import aisteer360.algorithms.core.steering_pipeline as pipeline_module

    original = backend_module.resolve_backend_class

    def resolver(spec):
        if spec.kind == "huggingface":
            return original(spec)
        return FakeEngineBackend

    monkeypatch.setattr(backend_module, "resolve_backend_class", resolver)
    monkeypatch.setattr(pipeline_module, "resolve_backend_class", resolver)
    FakeEngineBackend.reset()
    yield FakeEngineBackend
    FakeEngineBackend.reset()


CALLS: list = []


class _StageStructural(StructuralControl):
    Args = None

    def artifact_capability(self):
        return Capability.SERVE_CHECKPOINT

    def export_artifact(self):
        return CheckpointArtifact(path="/tmp/ckpt")

    def steer(self, model, tokenizer=None, session=None, **kwargs):
        CALLS.append(("structural", model is not None))
        return model


class _ModuleOutput(OutputControl):
    """Module-level output control that is portable at generate and does not retain."""

    Args = None

    def requirements(self):
        return Requirements()

    def steer_access(self):
        return ModelAccess.MODULE

    def steer(self, model=None, tokenizer=None, session=None, **kwargs):
        CALLS.append(("module_output", model is not None))
        _ModuleOutput.stage_ref = weakref.ref(model)


class _SessionInput(InputControl):
    def adapt(self, input_ids, runtime_kwargs=None):
        return input_ids

    def steer(self, model=None, tokenizer=None, session=None, **kwargs):
        CALLS.append(("input", model is not None))


class _CaptureFitter(OutputControl):
    """Capture-level output control with a declared fit; portable at generate."""

    Args = None

    def __init__(self, label):
        super().__init__()
        self.label = label
        self.steer_count = 0
        self.saw_in_process = None

    def requirements(self):
        return Requirements()

    def steer_access(self):
        return ModelAccess.CAPTURE

    def steer_fits(self):
        return (("_FakeFit", "direction"),)

    def steer(self, model=None, tokenizer=None, session=None, **kwargs):
        self.steer_count += 1
        self.saw_in_process = getattr(session, "in_process", None)
        CALLS.append((self.label, model is not None))
        prompt = PreparedPrompt.from_token_ids(torch.tensor([[0]], dtype=torch.long))
        session.capture([prompt], [0], "last_token")


class _RetainingModule(OutputControl):
    """Deliberately retains the staged model while claiming a portable generate phase."""

    Args = None

    def requirements(self):
        return Requirements()

    def steer_access(self):
        return ModelAccess.MODULE

    def steer(self, model=None, tokenizer=None, session=None, **kwargs):
        self.model = model


def _engine_spec(model_dir) -> BackendSpec:
    return BackendSpec(kind="vllm", model=model_dir, options={"hook_plugin": True})


class TestPhasePartition:

    def test_stage_runs_module_steps_first_in_per_phase_global_order(self, fake_engine, model_dir):
        CALLS.clear()
        controls = [_SessionInput(), _ModuleOutput(), _StageStructural()]
        pipeline = SteeringPipeline(controls=controls, backend=_engine_spec(model_dir))
        pipeline.steer()

        # global order is structural, input, output; the stage phase (structural and the
        # module output control) runs before the session phase (the input control)
        assert CALLS == [
            ("structural", True), ("module_output", True), ("input", False),
        ]

    def test_stage_is_freed_before_the_engine_boots(self, fake_engine, model_dir):
        CALLS.clear()

        def observer():
            assert pipeline.model is None
            assert _ModuleOutput.stage_ref() is None

        fake_engine.boot_observer = observer
        pipeline = SteeringPipeline(
            controls=[_ModuleOutput()], backend=_engine_spec(model_dir),
        )
        pipeline.steer()
        assert len(fake_engine.instances) == 1
        assert pipeline.model is None

    def test_structural_artifacts_hand_off_to_the_engine(self, fake_engine, model_dir):
        CALLS.clear()
        pipeline = SteeringPipeline(
            controls=[_StageStructural()], backend=_engine_spec(model_dir),
        )
        pipeline.steer()
        (backend,) = fake_engine.instances
        (artifact,) = backend.artifacts
        assert artifact.path == "/tmp/ckpt"
        assert artifact.provenance.backend_spec_hash is not None
        assert artifact.provenance.model_fingerprint is not None


class TestFreeProtocol:

    def test_retaining_control_raises_naming_itself(self, fake_engine, model_dir):
        pipeline = SteeringPipeline(
            controls=[_RetainingModule()], backend=_engine_spec(model_dir),
        )
        with pytest.raises(RuntimeError, match="retained past the steer stage by: _RetainingModule"):
            pipeline.steer()
        assert fake_engine.instances == []  # the engine never booted


class TestSmokeTestDegradation:

    def test_capture_failure_degrades_to_the_stage_without_double_steers(self, fake_engine, model_dir):
        CALLS.clear()
        fake_engine.capture_fails = True
        fitter_a = _CaptureFitter("fitter_a")
        fitter_b = _CaptureFitter("fitter_b")
        session_input = _SessionInput()
        pipeline = SteeringPipeline(
            controls=[session_input, fitter_a, fitter_b],
            backend=_engine_spec(model_dir),
        )

        report = pipeline.check()
        assert all(step.venue == "session" for step in report.plan.steps)
        assert report.plan.stages is False

        with pytest.warns(UserWarning, match="failed at steer.*degrades to a staged in-process model"):
            pipeline.steer()

        # each fitter steered exactly once, on the stage with in-process capture
        assert fitter_a.steer_count == 1
        assert fitter_b.steer_count == 1
        assert fitter_a.saw_in_process is True
        assert fitter_b.saw_in_process is True
        # the engine was released before the stage ran, then re-booted for the rest
        assert fake_engine.events[:2] == ["boot", "engine-capture"]
        assert fake_engine.events[2] == "release"
        assert fake_engine.events[3] == "boot"
        assert len(fake_engine.instances) == 2
        assert fake_engine.instances[0].released is True
        # the input control ran through the re-booted engine session, after the stage
        assert CALLS == [("fitter_a", False), ("fitter_b", False), ("input", False)]
        assert pipeline.model is None

    def test_passing_smoke_test_keeps_fits_on_the_session(self, fake_engine, model_dir):
        CALLS.clear()
        fitter = _CaptureFitter("fitter")
        pipeline = SteeringPipeline(
            controls=[fitter], backend=_engine_spec(model_dir),
        )
        pipeline.steer()
        assert fitter.steer_count == 1
        assert fitter.saw_in_process is False
        assert len(fake_engine.instances) == 1
        # one smoke capture plus the fitter's own capture, both through the engine
        assert fake_engine.events.count("engine-capture") == 2


@pytest.mark.skipif(
    os.environ.get("RUN_TRL_SMOKE") != "1",
    reason="set RUN_TRL_SMOKE=1 to run the TRL staged-steer smoke test (trains a tiny SFT LoRA)",
)
class TestTRLStagedSteerSmoke:
    """A real SFT LoRA control, staged on an engine backend, releases the staged model and hands off
    its merged checkpoint.

    Reproduces the notebook scenario that first exposed the retention bug (merged-LoRA SFT with a
    vLLM backend) against the fake engine, so the whole staged path runs without vLLM: train on the
    staged in-process model, merge, free the stage (the retention check must pass), then boot the
    engine with the merged checkpoint as its artifact.
    """

    def test_merged_lora_sft_frees_the_stage_and_hands_off_the_checkpoint(
        self, fake_engine, model_dir, tmp_path
    ):
        from datasets import Dataset

        from aisteer360.algorithms.structural_control.wrappers.trl.sfttrainer import SFT

        tokenizer = wordlevel_tokenizer()
        encoded = tokenizer(["the cat sat on the mat", "the dog ran fast"])
        train_dataset = Dataset.from_dict(
            {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
                "labels": [list(ids) for ids in encoded["input_ids"]],
            }
        )

        merged_dir = tmp_path / "sft_merged"
        sft = SFT(
            train_dataset=train_dataset,
            use_peft=True,
            r=4,
            lora_alpha=8,
            target_modules=["q_proj", "v_proj"],
            merge_lora_after_train=True,
            merged_output_dir=str(merged_dir),
            output_dir=str(tmp_path / "sft_out"),
            load_best_model_at_end=False,
            per_device_train_batch_size=1,
            num_train_epochs=1,
            report_to="none",
            training_args={"use_cpu": True, "bf16": False, "fp16": False, "max_steps": 1},
        )
        pipeline = SteeringPipeline(controls=[sft], backend=_engine_spec(model_dir))
        pipeline.steer()

        # the stage was freed (retention check passed) and the engine booted exactly once
        assert pipeline.model is None
        assert len(fake_engine.instances) == 1
        (backend,) = fake_engine.instances

        # the merged checkpoint is the artifact handed to the engine
        (artifact,) = backend.artifacts
        assert isinstance(artifact, CheckpointArtifact)
        assert artifact.path == str(merged_dir)
        assert merged_dir.exists()
