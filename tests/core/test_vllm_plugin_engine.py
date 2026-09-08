"""Engine-gated tests for `VLLMBackend` with the vLLM-Hook unified worker: greedy-decode parity
per exported state control, KV-salting regression, chunked-prefill exactness, and the capture and
probe paths. The whole module skips when vLLM is not installed; running it requires a GPU-capable
environment with the `vllm` extra.

This is a separate module because its plugin engine and the offline engine in `test_vllm_engine`
each assume they are the only live vLLM engine in the process at release, so their module-scoped
fixtures must not coexist.

The engine runs in its own process, so the GPU must be in the default compute mode. Under
`Exclusive_Process` the engine process is refused a CUDA context once the test process holds
one, and every test here skips.
"""
import pytest

vllm = pytest.importorskip("vllm")

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from steerability.algorithms.core.execution import (  # noqa: E402
    BackendSpec,
    GenerationItem,
    GenerationParams,
    PreparedPrompt,
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
def plugin_backend():
    """Engine with the vLLM-Hook unified worker active and prefix caching enabled."""
    spec = BackendSpec(
        kind="vllm",
        model=TINY_MODEL,
        options={
            "hook_plugin": True,
            "engine_kwargs": {**ENGINE_KWARGS, "enable_prefix_caching": True},
        },
    )
    try:
        backend = VLLMBackend(spec)
    except Exception as exception:
        pytest.skip(f"Could not boot the plugin engine: {exception}")
    if backend._discovery is None:
        backend.release()
        pytest.skip("The engine served no vLLM-Hook discovery payload.")
    try:
        yield backend
    finally:
        backend.release()


def _hf_reference(control_factory, prompt: str, max_new_tokens: int = 8):
    """Greedy continuation ids under the control's hooks on the in-process backend."""
    tokenizer = _tokenizer()
    model = AutoModelForCausalLM.from_pretrained(TINY_MODEL)
    control = control_factory()
    pipeline = SteeringPipeline(controls=[control], model=model, tokenizer=tokenizer)
    pipeline.steer()
    out = pipeline.generate(text=prompt, max_new_tokens=max_new_tokens, do_sample=False,
                            return_output=True)
    return out.output_ids[0].tolist(), control


def _steered_vector(model_ref: str, hidden: int, layers, k: int = 1, seed: int = 5):
    from steerability.algorithms.state_control.common.steering_vector import SteeringVector

    generator = torch.Generator().manual_seed(seed)
    return SteeringVector(
        model_type="llama",
        directions={lid: 4.0 * torch.randn(k, hidden, generator=generator) for lid in layers},
    )


class TestSpecParityOnEngine:
    """Greedy-decode parity per exported control (§8.2). Skips without a live plugin engine."""

    def _parity(self, plugin_backend, control_factory, prompt="The committee reviewed the plan"):
        reference_ids, _ = _hf_reference(control_factory, prompt)

        control = control_factory()
        pipeline = SteeringPipeline(
            controls=[control], backend=plugin_backend.spec,
        )
        pipeline.model = AutoModelForCausalLM.from_pretrained(TINY_MODEL)
        pipeline.tokenizer = _tokenizer()
        pipeline._backends[plugin_backend.spec] = plugin_backend
        pipeline.steer()
        out = pipeline.generate(text=prompt, max_new_tokens=8, do_sample=False, return_output=True)
        engine_ids = out.output_ids[0].tolist()
        overlap = min(len(reference_ids), len(engine_ids))
        assert engine_ids[:overlap] == reference_ids[:overlap]

    def test_caa_parity(self, plugin_backend):
        hidden = plugin_backend._layout.hidden_size
        self._parity(
            plugin_backend,
            lambda: __import__(
                "steerability.algorithms.state_control.caa.control", fromlist=["CAA"]
            ).CAA(steering_vector=_steered_vector(TINY_MODEL, hidden, [1]), layer_id=1, multiplier=6.0),
        )

    def test_directional_ablation_parity(self, plugin_backend):
        hidden = plugin_backend._layout.hidden_size
        from steerability.algorithms.state_control.directional_ablation.control import DirectionalAblation
        self._parity(
            plugin_backend,
            lambda: DirectionalAblation(
                steering_vector=_steered_vector(TINY_MODEL, hidden, [1, 2]), layer_ids=[1, 2],
            ),
        )

    def test_angular_steering_parity(self, plugin_backend):
        hidden = plugin_backend._layout.hidden_size
        from steerability.algorithms.state_control.angular_steering.control import AngularSteering
        self._parity(
            plugin_backend,
            lambda: AngularSteering(
                steering_vector=_steered_vector(TINY_MODEL, hidden, [1], k=2),
                target_degree=40.0, intervention_point="layer_output",
            ),
        )

    def test_steered_after_baseline_shared_prefix(self, plugin_backend):
        """The salting rule's regression alarm: a steered request after a baseline request over
        the same prompt must not reuse KV computed without the intervention."""
        from steerability.algorithms.core.execution import InterventionEntry
        from steerability.algorithms.state_control.common.lowering import lower_interventions
        from steerability.algorithms.state_control.common.specs import Intervention, TokenScope
        from steerability.algorithms.state_control.common.transforms import AdditiveTransform

        hidden = plugin_backend._layout.hidden_size
        vector = _steered_vector(TINY_MODEL, hidden, [1])
        spec = lower_interventions(
            [Intervention(
                layers=(1,),
                transform=AdditiveTransform(vector.directions, strength=8.0),
                scope=TokenScope("all"),
            )],
            num_layers=plugin_backend._layout.num_layers,
        )
        prompt = PreparedPrompt.from_text("The committee reviewed the proposal carefully")
        params = GenerationParams(max_new_tokens=8, greedy=True)
        with plugin_backend.open_session() as session:
            baseline_first = session.generate([GenerationItem(prompt=prompt)], params)
            steered = session.generate(
                [GenerationItem(prompt=prompt, state_entries=(InterventionEntry(spec=spec),))],
                params,
            )
            baseline_again = session.generate([GenerationItem(prompt=prompt)], params)
        assert steered[0].output.output_ids.tolist() != baseline_first[0].output.output_ids.tolist()
        assert baseline_again[0].output.output_ids.tolist() == baseline_first[0].output.output_ids.tolist()

    def test_scored_vs_generated_scope_agreement(self, plugin_backend):
        """The vLLM engine backend refuses `compute_logprobs` for a scoped intervention, since its
        prompt-logprob scoring would anchor token scopes at the request's prompt end rather than the
        control's scope; the huggingface arm scores normally."""
        from steerability.algorithms.state_control.caa.control import CAA

        hidden = plugin_backend._layout.hidden_size
        factory = lambda: CAA(
            steering_vector=_steered_vector(TINY_MODEL, hidden, [1]), layer_id=1,
            multiplier=6.0, token_scope="after_prompt",
        )
        tokenizer = _tokenizer()
        model = AutoModelForCausalLM.from_pretrained(TINY_MODEL)
        prompt_ids = tokenizer("hello world example", return_tensors="pt")["input_ids"]
        ref_ids = tokenizer(" one two", return_tensors="pt", add_special_tokens=False)["input_ids"]

        hf_pipeline = SteeringPipeline(controls=[factory()], model=model, tokenizer=tokenizer)
        hf_pipeline.steer()
        hf_scores = hf_pipeline.compute_logprobs(prompt_ids, ref_output_ids=ref_ids)

        engine_pipeline = SteeringPipeline(
            controls=[factory()], backend=plugin_backend.spec,
        )
        engine_pipeline.model = AutoModelForCausalLM.from_pretrained(TINY_MODEL)
        engine_pipeline.tokenizer = tokenizer
        engine_pipeline._backends[plugin_backend.spec] = plugin_backend
        engine_pipeline.steer()
        # the backend refuses rather than return silently mis-anchored scores
        from steerability.algorithms.core.execution.contracts import UnsupportedPipelineError

        with pytest.raises(UnsupportedPipelineError, match="unsupported at score on backend kind 'vllm'"):
            engine_pipeline.compute_logprobs(prompt_ids, ref_output_ids=ref_ids)
        assert hf_scores.shape == (1, ref_ids.shape[-1])

    def test_chunked_prefill_last_k_exactness(self, plugin_backend):
        """`last_k` selects absolute positions, so a long prompt under chunked prefill steers
        exactly the last k prompt rows plus decode rows (§3.4)."""
        from steerability.algorithms.state_control.caa.control import CAA

        hidden = plugin_backend._layout.hidden_size
        long_prompt = " ".join(["review"] * 96)
        factory = lambda: CAA(
            steering_vector=_steered_vector(TINY_MODEL, hidden, [1]), layer_id=1,
            multiplier=6.0, token_scope="last_k", last_k=3,
        )
        reference_ids, _ = _hf_reference(factory, long_prompt)

        control = factory()
        pipeline = SteeringPipeline(
            controls=[control], backend=plugin_backend.spec,
        )
        pipeline.model = AutoModelForCausalLM.from_pretrained(TINY_MODEL)
        pipeline.tokenizer = _tokenizer()
        pipeline._backends[plugin_backend.spec] = plugin_backend
        pipeline.steer()
        out = pipeline.generate(text=long_prompt, max_new_tokens=8, do_sample=False,
                                return_output=True)
        engine_ids = out.output_ids[0].tolist()
        overlap = min(len(reference_ids), len(engine_ids))
        assert engine_ids[:overlap] == reference_ids[:overlap]


class TestCaptureOnEngine:
    """Capture and probe-path fixtures. Skip without a live plugin engine."""

    @pytest.mark.parametrize("location", ["layer_output", "layer_input"])
    @pytest.mark.parametrize("mode", ["all_tokens", "last_token"])
    def test_capture_parity_with_in_process_funnel(self, plugin_backend, mode, location):
        from steerability.algorithms.core.execution import BackendSpec
        from steerability.backends.huggingface import HFBackend

        tokenizer = _tokenizer()
        model = AutoModelForCausalLM.from_pretrained(TINY_MODEL)
        prompts = [
            PreparedPrompt.from_text("The committee reviewed the proposal"),
            PreparedPrompt.from_text("A short prompt"),
        ]
        num_layers = plugin_backend._layout.num_layers
        layers = sorted({max(0, num_layers - 2), num_layers - 1})

        hf_backend = HFBackend.adopt(
            BackendSpec(kind="huggingface"), lambda: model, lambda: tokenizer,
        )
        with hf_backend.open_session() as hf_session:
            reference = hf_session.capture(prompts, layers, mode, location=location)
        with plugin_backend.open_session() as session:
            captured = session.capture(prompts, layers, mode, location=location)

        assert captured.attention_mask.tolist() == reference.attention_mask.tolist()
        for layer in layers:
            assert torch.allclose(
                captured.hidden[layer].float(), reference.hidden[layer].float(),
                atol=5e-2, rtol=5e-2,
            )

    def test_vector_fitted_on_engine_steers_in_process(self, plugin_backend):
        from steerability.algorithms.core.internals.data import ContrastivePairs
        from steerability.algorithms.state_control.common.estimators import MeanDifferenceEstimator
        from steerability.algorithms.state_control.common.fit_specs import VectorTrainSpec

        pairs = ContrastivePairs(
            positives=["the committee approved it", "they agreed at once"],
            negatives=["the committee rejected it", "they refused at once"],
        )
        spec = VectorTrainSpec(method="mean_diff", accumulate="last_token", prompt_format="raw")
        tokenizer = _tokenizer()
        model = AutoModelForCausalLM.from_pretrained(TINY_MODEL)

        with plugin_backend.open_session() as session:
            remote_vector = MeanDifferenceEstimator().fit(
                None, tokenizer, data=pairs, spec=spec, session=session,
            )
        local_vector = MeanDifferenceEstimator().fit(model, tokenizer, data=pairs, spec=spec)
        for layer in local_vector.directions:
            assert torch.allclose(
                remote_vector.directions[layer], local_vector.directions[layer],
                atol=5e-2, rtol=5e-2,
            )

    def test_conditional_gate_open_vs_closed_matches_in_process(self, plugin_backend):
        """A probe-gated adapter fires on the gate-open prompt and stays inert on the
        gate-closed prompt, matching in-process decisions."""
        from steerability.algorithms.core.internals.probes import Probe
        from steerability.algorithms.state_control.activation_adapter.control import ActivationAdapter
        from steerability.algorithms.state_control.common.transforms import AdditiveTransform

        layout = plugin_backend._layout
        hidden = layout.hidden_size
        num_layers = layout.num_layers
        cond_layer = max(0, num_layers - 2)
        intv_layer = num_layers - 1
        tokenizer = _tokenizer()
        model = AutoModelForCausalLM.from_pretrained(TINY_MODEL)

        open_prompt = "the committee approved the proposal"
        closed_prompt = "nothing to see here at all"
        enc_open = tokenizer(open_prompt, return_tensors="pt")
        enc_closed = tokenizer(closed_prompt, return_tensors="pt")

        # a probe whose weights separate the two prompts at layer 1's input
        from steerability.algorithms.core.internals.capture import layerwise_tokenwise_hidden
        hs_open = layerwise_tokenwise_hidden(model, dict(enc_open), location="layer_input")
        hs_closed = layerwise_tokenwise_hidden(model, dict(enc_closed), location="layer_input")
        weight = (hs_open[cond_layer].mean(dim=(0, 1)) - hs_closed[cond_layer].mean(dim=(0, 1))).float()
        weight = weight / weight.norm()
        score_open = float(hs_open[cond_layer].float().mean(dim=(0, 1)) @ weight)
        score_closed = float(hs_closed[cond_layer].float().mean(dim=(0, 1)) @ weight)
        bias = -(score_open + score_closed) / 2
        probe = Probe(
            model_type=getattr(model.config, "model_type", "unknown"),
            location="layer_input", pooling="mean", layer_ids=[cond_layer],
            weights={cond_layer: weight}, bias=bias, meta={},
        )

        generator = torch.Generator().manual_seed(9)
        vector = {intv_layer: 6.0 * torch.randn(1, hidden, generator=generator)}

        def factory():
            return ActivationAdapter(
                transform=AdditiveTransform(vector, strength=1.0),
                layer_ids=[intv_layer], hook_point="layer_input", token_scope="all",
                gate=probe.as_gate(allow_model_mismatch=True),
            )

        def run(backend_spec, backend=None):
            pipeline = SteeringPipeline(
                controls=[factory()], backend=backend_spec,
                model=AutoModelForCausalLM.from_pretrained(TINY_MODEL),
                tokenizer=tokenizer,
            )
            if backend is not None:
                pipeline._backends[backend.spec] = backend
            pipeline.steer()
            return [
                pipeline.generate(text=prompt, max_new_tokens=8, do_sample=False, return_output=True)
                for prompt in (open_prompt, closed_prompt)
            ]

        hf_outputs = run("huggingface")
        engine_outputs = run(plugin_backend.spec, plugin_backend)
        for hf_out, engine_out in zip(hf_outputs, engine_outputs):
            hf_ids = hf_out.output_ids[0].tolist()
            engine_ids = engine_out.output_ids[0].tolist()
            overlap = min(len(hf_ids), len(engine_ids))
            assert engine_ids[:overlap] == hf_ids[:overlap]

    def test_routed_decoding_end_to_end_on_engine(self, plugin_backend):
        from steerability.algorithms.core.internals.data import ContrastivePairs
        from steerability.algorithms.core.internals.probes import ProbeFitSpec, ProbeSetFit
        from steerability.algorithms.output_control.routed_decoding import P, Route, RoutedDecoding, Router, respond

        pairs = ContrastivePairs(
            positives=["the committee approved it"],
            negatives=["nothing to see here"],
        )
        control = RoutedDecoding(
            probes=ProbeSetFit(
                data={"topic": pairs},
                spec=ProbeFitSpec(method="mean_diff", pooling="mean", location="layer_input",
                                  prompt_format="raw", candidate_layers=[1]),
            ),
            rules=Router(routes=[Route("topic", when=P("topic"), action=respond("ROUTED"))]),
        )
        pipeline = SteeringPipeline(
            controls=[control], backend=plugin_backend.spec,
        )
        pipeline.model = AutoModelForCausalLM.from_pretrained(TINY_MODEL)
        pipeline.tokenizer = _tokenizer()
        pipeline._backends[plugin_backend.spec] = plugin_backend
        pipeline.steer()
        text = pipeline.generate(text="the committee approved it", max_new_tokens=8, do_sample=False)
        assert isinstance(text, str)
