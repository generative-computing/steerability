"""Engine-gated tests for the offline `VLLMBackend`: prompt-only generation, greedy HF/vLLM
parity, prompt-logprob scoring, and declarative-constraint parity. The whole module skips when
vLLM is not installed; running it requires a GPU-capable environment with the `vllm` extra.

The plugin-worker tests live in `test_vllm_plugin_engine` because that engine and this offline
engine each assume they are the only live vLLM engine in the process at release, so their
module-scoped fixtures must not coexist.

The engine runs in its own process, so the GPU must be in the default compute mode. Under
`Exclusive_Process` the engine process is refused a CUDA context once the test process holds
one, and every engine-gated test here skips.
"""
import pytest

vllm = pytest.importorskip("vllm")

# every engine boots `JackFram/llama-68m`, which is not in the CI model set
pytestmark = pytest.mark.network

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from steerability.algorithms.core.execution import (  # noqa: E402
    BackendSpec,
    GenerationItem,
    GenerationParams,
    PreparedPrompt,
    ScoringItem,
)
from steerability.algorithms.core.steering_pipeline import SteeringPipeline  # noqa: E402
from steerability.backends.vllm import VLLMBackend  # noqa: E402
from steerability.utils.tokenization import ensure_pad_token  # noqa: E402

TINY_MODEL = "JackFram/llama-68m"

# float32 matches the checkpoint's own dtype, so greedy decoding agrees token-for-token with the
# HF reference arms
ENGINE_KWARGS = {
    "enforce_eager": True,
    "max_model_len": 512,
    "dtype": "float32",
    "gpu_memory_utilization": 0.25,
}


def _tokenizer():
    """The tiny model's tokenizer with a pad token, as the pipeline's own loader would set it."""
    return ensure_pad_token(AutoTokenizer.from_pretrained(TINY_MODEL))


@pytest.fixture(scope="module")
def engine_backend():
    spec = BackendSpec(
        kind="vllm",
        model=TINY_MODEL,
        options={"engine_kwargs": dict(ENGINE_KWARGS)},
    )
    try:
        backend = VLLMBackend(spec)
    except Exception as exception:
        pytest.skip(f"Could not boot the vLLM engine: {exception}")
    try:
        yield backend
    finally:
        backend.release()


class TestOfflineEngine:

    def test_prompt_only_generation(self, engine_backend):
        item = GenerationItem(prompt=PreparedPrompt.from_text("The capital of France is"))
        with engine_backend.open_session() as session:
            results = session.generate([item], GenerationParams(max_new_tokens=8, greedy=True))
        output = results[0].output
        assert output.output_ids.shape[0] == 1
        assert output.output_ids.shape[1] > 0
        assert output.finish_reasons[0] in ("stop", "eos", "length")

    def test_greedy_parity_with_hf(self, engine_backend):
        tokenizer = _tokenizer()
        model = AutoModelForCausalLM.from_pretrained(TINY_MODEL)
        encoded = tokenizer("The sky is", return_tensors="pt")
        hf_full = model.generate(
            input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"],
            max_new_tokens=8, do_sample=False,
        )
        hf_new = hf_full[0, encoded["input_ids"].size(1):].tolist()

        item = GenerationItem(prompt=PreparedPrompt.from_token_ids(encoded["input_ids"]))
        with engine_backend.open_session() as session:
            results = session.generate([item], GenerationParams(max_new_tokens=8, greedy=True))
        vllm_new = results[0].output.output_ids[0].tolist()
        assert vllm_new[: len(hf_new)] == hf_new[: len(vllm_new)]

    def test_stop_string_semantics(self, engine_backend):
        item = GenerationItem(prompt=PreparedPrompt.from_text("a b a b a b"))
        with engine_backend.open_session() as session:
            results = session.generate(
                [item],
                GenerationParams(max_new_tokens=16, greedy=True, stop_strings=("b",)),
            )
        output = results[0].output
        decoded = engine_backend.tokenizer.decode(output.output_ids[0], skip_special_tokens=True)
        if output.finish_reasons[0] == "stop":
            assert "b" in decoded  # ids returned as generated, stop text included

    def test_prompt_logprob_scoring(self, engine_backend):
        tokenizer = engine_backend.tokenizer
        prompt_ids = tokenizer("hello world", return_tensors="pt")["input_ids"]
        ref = prompt_ids[:, -2:]
        item = ScoringItem(
            prompt=PreparedPrompt.from_token_ids(prompt_ids), ref_output_ids=ref,
        )
        with engine_backend.open_session() as session:
            scored = session.score([item], GenerationParams())
        assert scored.shape == (1, 2)
        assert torch.isfinite(scored).all()


class TestConstraintParityOnEngine:
    """Parity fixture: one declarative source constrains identically on both arms."""

    def test_json_schema_constrained_parity(self, engine_backend):
        import json

        from steerability.algorithms.output_control.constrained_decoding import ConstrainedDecoding

        schema = {
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
            "additionalProperties": False,
        }
        prompt = "Return a JSON object:"

        def run(backend_spec, backend=None):
            control = ConstrainedDecoding(json_schema=schema, include_in_scoring=False)
            pipeline = SteeringPipeline(
                controls=[control], backend=backend_spec,
                model=AutoModelForCausalLM.from_pretrained(TINY_MODEL),
                tokenizer=_tokenizer(),
            )
            if backend is not None:
                pipeline._backends[backend.spec] = backend
            pipeline.steer()
            return pipeline.generate(text=prompt, max_new_tokens=64, do_sample=False)

        hf_text = run("huggingface")
        engine_text = run(engine_backend.spec, engine_backend)

        def parsed(label: str, text: str):
            try:
                return json.loads(text)
            except json.JSONDecodeError as error:
                pytest.fail(f"{label} arm produced incomplete or invalid JSON ({error}): {text!r}")

        assert parsed("engine", engine_text) is not None
        assert parsed("hf", hf_text) is not None
        assert engine_text == hf_text


def _greedy_item(tokenizer, text: str) -> GenerationItem:
    encoded = tokenizer(text, return_tensors="pt")
    return GenerationItem(prompt=PreparedPrompt.from_token_ids(encoded["input_ids"]))


class TestPerRequestEngine:
    """Per-request admission through the engine thread: concurrent calls share the running batch, a
    short request returns before a long one, and cancellation and release abort in-flight requests."""

    def test_score_matches_the_hugging_face_reference(self, engine_backend):
        tokenizer = _tokenizer()
        model = AutoModelForCausalLM.from_pretrained(TINY_MODEL, dtype=torch.float32)
        prompt_ids = tokenizer("The capital of France is", return_tensors="pt")["input_ids"]
        ref = tokenizer(" Paris", add_special_tokens=False, return_tensors="pt")["input_ids"]
        with torch.no_grad():
            logits = model(torch.cat([prompt_ids, ref], dim=1)).logits
        positions = torch.arange(prompt_ids.size(1) - 1, prompt_ids.size(1) - 1 + ref.size(1))
        reference = torch.log_softmax(logits[0, positions], dim=-1).gather(-1, ref[0].unsqueeze(-1)).squeeze(-1)

        item = ScoringItem(prompt=PreparedPrompt.from_token_ids(prompt_ids), ref_output_ids=ref)
        with engine_backend.open_session() as session:
            scored = session.score([item], GenerationParams())
        # the engine and the in-process forward use different attention kernels, which differ by
        # about 1e-3 in float32
        assert torch.allclose(scored[0], reference, atol=1e-2)

    def test_concurrent_calls_share_the_engine_and_the_short_one_returns_first(self, engine_backend):
        import threading
        import time

        tokenizer = engine_backend.tokenizer
        short_item = _greedy_item(tokenizer, "The capital of France is")
        long_item = _greedy_item(tokenizer, "Once upon a time")
        short_params = GenerationParams(max_new_tokens=4, greedy=True)
        long_params = GenerationParams(max_new_tokens=400, min_new_tokens=400, greedy=True)

        with engine_backend.open_session() as session:
            serial_short = session.generate([short_item], short_params)[0].output.output_ids.tolist()
            serial_long = session.generate([long_item], long_params)[0].output.output_ids.tolist()

        finished: dict[str, float] = {}
        outputs: dict[str, list] = {}

        def run(name, item, params):
            with engine_backend.open_session() as session:
                outputs[name] = session.generate([item], params)[0].output.output_ids.tolist()
            finished[name] = time.monotonic()

        long_thread = threading.Thread(target=run, args=("long", long_item, long_params))
        short_thread = threading.Thread(target=run, args=("short", short_item, short_params))
        long_thread.start()
        time.sleep(0.05)
        short_thread.start()
        long_thread.join(120)
        short_thread.join(120)

        assert finished["short"] < finished["long"]
        assert outputs["short"] == serial_short
        assert outputs["long"] == serial_long

    def test_cancelling_a_token_aborts_the_engine_request(self, engine_backend):
        import threading
        import time

        from steerability.algorithms.core.execution import CancelToken, GenerationCancelled

        item = _greedy_item(engine_backend.tokenizer, "Once upon a time")
        params = GenerationParams(max_new_tokens=480, min_new_tokens=480, greedy=True)
        token = CancelToken()
        errors = []

        def run():
            try:
                with engine_backend.open_session() as session:
                    session.generate([item], params, cancel_token=token)
            except BaseException as error:
                errors.append(error)

        thread = threading.Thread(target=run)
        thread.start()
        time.sleep(0.3)
        token.cancel()
        thread.join(60)

        assert len(errors) == 1 and isinstance(errors[0], GenerationCancelled)
        deadline = time.monotonic() + 10
        while engine_backend._engine.output_processor.get_num_unfinished_requests() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert engine_backend._engine.output_processor.get_num_unfinished_requests() == 0

    def test_pipeline_generate_with_a_cancelled_token_raises(self, engine_backend):
        import threading
        import time

        from steerability.algorithms.core.execution import CancelToken, GenerationCancelled

        pipeline = SteeringPipeline(backend=engine_backend.spec, tokenizer=_tokenizer())
        pipeline._backends[engine_backend.spec] = engine_backend
        pipeline.steer()
        assert pipeline.supports_concurrent_generate
        token = CancelToken()
        errors = []

        def run():
            try:
                pipeline.generate(
                    text="Once upon a time", max_new_tokens=480, min_new_tokens=480, do_sample=False,
                    cancel_token=token,
                )
            except BaseException as error:
                errors.append(error)

        thread = threading.Thread(target=run)
        thread.start()
        time.sleep(0.3)
        token.cancel()
        thread.join(60)
        assert len(errors) == 1 and isinstance(errors[0], GenerationCancelled)
