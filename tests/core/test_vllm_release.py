"""Engine-gated tests for deterministic `VLLMBackend` release: boot->release->boot in one
process, idempotence, released-instance errors, pipeline-level release with
reconstruct-on-next-use, and end-to-end pipeline generation under a budget stop. The whole module
skips when vLLM is not installed; running it requires a GPU-capable environment with the `vllm`
extra.

This is a separate module because each test here boots and releases its own engine, so it must not
share a process-lifetime engine with module-scoped fixtures from another file mid-test.

The engine runs in its own process, so the GPU must be in the default compute mode. Under
`Exclusive_Process` the engine process is refused a CUDA context once the test process holds
one, and every test here skips.
"""
import pytest

vllm = pytest.importorskip("vllm")

from aisteer360.algorithms.core.execution import GenerationItem, GenerationParams, PreparedPrompt  # noqa: E402
from aisteer360.algorithms.core.steering_pipeline import SteeringPipeline  # noqa: E402
from aisteer360.backends.vllm import VLLMBackend  # noqa: E402

TINY_MODEL = "JackFram/llama-68m"


def _spec():
    from aisteer360.algorithms.core.execution import BackendSpec

    return BackendSpec(
        kind="vllm",
        model=TINY_MODEL,
        options={
            "engine_kwargs": {
                "enforce_eager": True,
                "max_model_len": 512,
                "gpu_memory_utilization": 0.25,
            }
        },
    )


def _boot_or_skip():
    try:
        return VLLMBackend(_spec())
    except Exception as exception:
        pytest.skip(f"Could not boot the vLLM engine: {exception}")


def _generate_once(backend) -> list:
    item = GenerationItem(prompt=PreparedPrompt.from_text("The capital of France is"))
    with backend.open_session() as session:
        return session.generate([item], GenerationParams(max_new_tokens=8, greedy=True))


def test_boot_release_boot():
    """Construct, generate, release, then construct a second engine with the same spec and
    generate again, in one process; both generations succeed."""
    first = _boot_or_skip()
    first_results = _generate_once(first)
    assert first_results[0].output.output_ids.shape[1] > 0
    first.release()

    second = VLLMBackend(_spec())
    try:
        second_results = _generate_once(second)
        assert second_results[0].output.output_ids.shape[1] > 0
    finally:
        second.release()


def test_release_idempotent():
    backend = _boot_or_skip()
    backend.release()
    backend.release()


def test_released_backend_raises():
    backend = _boot_or_skip()
    session = backend.open_session()  # opened before release
    backend.release()

    with pytest.raises(RuntimeError, match="was released"):
        backend.open_session()

    item = GenerationItem(prompt=PreparedPrompt.from_text("The capital of France is"))
    with pytest.raises(RuntimeError, match="was released"):
        session.generate([item], GenerationParams(max_new_tokens=8, greedy=True))


def test_pipeline_release_on_vllm():
    """Steer, generate, release_backends(), then generate again; reconstruct-on-next-use boots a
    fresh engine and succeeds."""
    from aisteer360.algorithms.output_control.stopping_rules.control import StoppingRules

    pipeline = SteeringPipeline(
        controls=[StoppingRules(budget=6)],
        backend=_spec(),
    )
    try:
        # the engine boots inside the guard at the steer phase's session step
        pipeline.steer()
    except Exception as exception:
        pytest.skip(f"Could not boot the vLLM engine: {exception}")
    try:
        first = pipeline.generate(text="Once upon a time", max_new_tokens=8, do_sample=False)
        assert isinstance(first, str)

        pipeline.release_backends()
        assert pipeline._backends == {}

        second = pipeline.generate(text="Once upon a time", max_new_tokens=8, do_sample=False)
        assert isinstance(second, str)
    finally:
        pipeline.release_backends()


def test_pipeline_end_to_end_with_stopping_rules():
    """Steer and generate end to end on the engine with a budget stop; the returned continuation is
    truncated to the budget."""
    from aisteer360.algorithms.output_control.stopping_rules.control import StoppingRules

    pipeline = SteeringPipeline(
        controls=[StoppingRules(budget=6)],
        backend=_spec(),
    )
    try:
        # the engine boots inside the guard at the steer phase's session step
        pipeline.steer()
    except Exception as exception:
        pytest.skip(f"Could not boot the vLLM engine: {exception}")
    try:
        out = pipeline.generate(text="Once upon a time", max_new_tokens=16, do_sample=False,
                                return_output=True)
        assert out.output_ids.shape[1] <= 6
    finally:
        pipeline.release_backends()
