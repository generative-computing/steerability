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

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from aisteer360.algorithms.core.execution import (  # noqa: E402
    BackendSpec,
    GenerationItem,
    GenerationParams,
    PreparedPrompt,
    ScoringItem,
)
from aisteer360.algorithms.core.steering_pipeline import SteeringPipeline  # noqa: E402
from aisteer360.backends.vllm import VLLMBackend  # noqa: E402
from aisteer360.utils.tokenization import ensure_pad_token  # noqa: E402

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
        assert output.finish_reason in ("stop", "eos", "length")

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
        if output.finish_reason == "stop":
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

        from aisteer360.algorithms.output_control.constrained_decoding import ConstrainedDecoding
        from aisteer360.backends.vllm.backend import _structured_outputs_engine_kwargs

        pytest.importorskip("xgrammar")
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
        # byte equality holds only when the engine grammar is whitespace-compact; on the legacy
        # guided-decoding surface without that switch, compare parsed structure instead
        defaults = _structured_outputs_engine_kwargs()
        if "structured_outputs_config" in defaults or "guided_decoding_disable_any_whitespace" in defaults:
            assert engine_text == hf_text
        else:
            assert parsed("engine", engine_text) == parsed("hf", hf_text)
