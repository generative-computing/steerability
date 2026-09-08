"""Tests for `VLLMServeBackend` and `VLLMServeSession` against a mocked vLLM server, plus the
encoder-decoder spec rejection. No vLLM installation or live server is required."""
import logging

import pytest
import torch
from transformers import LlamaConfig, T5Config

from steerability.algorithms.core.execution import (
    BackendSpec,
    GenerationItem,
    GenerationParams,
    HookEntry,
    InterventionEntry,
    InterventionSpec,
    PartialBatchError,
    PreparedPrompt,
    ScoringItem,
    TransportError,
    UnsupportedOperationError,
)
from steerability.backends.vllm import VLLMServeBackend
from tests.utils.tiny_models import wordlevel_tokenizer


class _FakeServer:
    """Routes `_request_json` calls to canned responses and records requests."""

    def __init__(self, model_id="m", completions=None, version=True, prompt_logprobs=None):
        self.model_id = model_id
        self.completions = completions or {}
        self.version = version
        self.prompt_logprobs = prompt_logprobs
        self.discovery: dict | None = None
        self.requests: list[tuple[str, dict | None]] = []
        self.fail_prompts: dict[tuple[int, ...], int] = {}

    def handle(self, path, payload):
        self.requests.append((path, payload))
        if path == "/version":
            if self.version:
                return {"version": "0.10.0"}
            raise ValueError("HTTP 404 from /version: not found")
        if path == "/v1/hook/capabilities":
            if self.discovery is not None:
                return self.discovery
            raise ValueError("HTTP 404 from /v1/hook/capabilities: not found")
        if path == "/v1/models":
            return {"data": [{"id": self.model_id}]}
        if path == "/v1/completions":
            prompt = tuple(payload["prompt"])
            remaining = self.fail_prompts.get(prompt, 0)
            if remaining > 0:
                self.fail_prompts[prompt] = remaining - 1
                raise TransportError("connection reset")
            if self.prompt_logprobs is not None and "prompt_logprobs" in payload:
                entries = [None] + [
                    {str(token_id): {"logprob": self.prompt_logprobs}}
                    for token_id in prompt[1:]
                ]
                return {"choices": [{
                    "text": "", "finish_reason": "length", "prompt_logprobs": entries,
                }]}
            key = prompt
            choices = self.completions.get(key)
            if choices is None:
                choices = [{
                    "token_ids": [9, 1], "finish_reason": "stop", "stop_reason": None,
                }]
            return {"choices": choices}
        raise ValueError(f"HTTP 404 from {path}: not found")


@pytest.fixture()
def fake_server(monkeypatch):
    server = _FakeServer()

    def fake_request(self, path, payload, expect_json=True):
        return server.handle(path, payload)

    monkeypatch.setattr(VLLMServeBackend, "_request_json", fake_request)
    monkeypatch.setattr(
        "steerability.backends.vllm.backend._client_tokenizer",
        lambda source, trust_remote_code=False: wordlevel_tokenizer(),
    )
    monkeypatch.setattr(
        "steerability.backends.vllm.backend._config_layout",
        lambda source, trust_remote_code=False: None,
    )
    monkeypatch.setattr("steerability.backends.vllm.capabilities._DISCOVERY_CACHE", {})
    return server


def _serve_spec(**options):
    merged = {"base_url": "http://server:8000", "retry_backoff": 0.0, **options}
    return BackendSpec(kind="vllm-serve", model="m", options=merged)


class TestServeBackendConstruction:

    def test_constructs_against_vllm_server(self, fake_server):
        backend = VLLMServeBackend(_serve_spec())
        assert backend._served_model == "m"
        assert ("/version", None) in fake_server.requests

    def test_non_vllm_endpoint_rejected(self, fake_server):
        fake_server.version = False
        with pytest.raises(ValueError, match="version"):
            VLLMServeBackend(_serve_spec())

    def test_base_url_v1_suffix_normalizes(self, fake_server):
        backend = VLLMServeBackend(_serve_spec(base_url="http://server:8000/v1/"))
        assert backend._base_url == "http://server:8000"

    def test_served_model_mismatch_rejected(self, fake_server):
        fake_server.model_id = "other-model"
        with pytest.raises(ValueError, match="other-model"):
            VLLMServeBackend(_serve_spec())

    def test_missing_base_url_rejected(self):
        with pytest.raises(ValueError, match="base_url"):
            VLLMServeBackend(BackendSpec(kind="vllm-serve", model="m"))

    def test_hook_plugin_without_discovery_surface_rejected(self, fake_server):
        with pytest.raises(ValueError, match="hook"):
            VLLMServeBackend(_serve_spec(hook_plugin=True))


class TestServeFingerprintVerification:
    """The chat-template comparison against the discovery payload's fingerprint."""

    @staticmethod
    def _templated_tokenizer():
        tokenizer = wordlevel_tokenizer()
        tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"
        return tokenizer

    @pytest.fixture()
    def templated_client(self, monkeypatch):
        monkeypatch.setattr(
            "steerability.backends.vllm.backend._client_tokenizer",
            lambda source, trust_remote_code=False: self._templated_tokenizer(),
        )

    def test_absent_served_template_fingerprint_skips_comparison(
        self, fake_server, templated_client, tmp_path, caplog,
    ):
        pytest.importorskip("vllm_hook_plugins")
        from vllm_hook_plugins.core.fingerprints import chat_template_fingerprint

        payload = _discovery_payload()
        payload["model"]["chat_template_fingerprint"] = chat_template_fingerprint(None)
        fake_server.discovery = payload
        with caplog.at_level(logging.WARNING, logger="steerability.backends.vllm"):
            VLLMServeBackend(_serve_spec(hook_plugin=True, artifact_dir=str(tmp_path)))
        assert not any("differs from the served" in record.getMessage() for record in caplog.records)

    def test_differing_served_template_fingerprint_warns(
        self, fake_server, templated_client, tmp_path, caplog,
    ):
        pytest.importorskip("vllm_hook_plugins")
        from vllm_hook_plugins.core.fingerprints import chat_template_fingerprint

        payload = _discovery_payload()
        payload["model"]["chat_template_fingerprint"] = chat_template_fingerprint("{{ other }}")
        fake_server.discovery = payload
        with caplog.at_level(logging.WARNING, logger="steerability.backends.vllm"):
            VLLMServeBackend(_serve_spec(hook_plugin=True, artifact_dir=str(tmp_path)))
        assert any("differs from the served" in record.getMessage() for record in caplog.records)


class TestServeSessionGenerate:

    def _item(self, ids=(0, 3, 4)):
        return GenerationItem(prompt=PreparedPrompt.from_token_ids(list(ids)))

    def test_token_id_round_trip_and_finish_mapping(self, fake_server):
        fake_server.completions[(0, 3, 4)] = [
            {"token_ids": [5, 6], "finish_reason": "stop", "stop_reason": "sat"},
        ]
        backend = VLLMServeBackend(_serve_spec())
        with backend.open_session() as session:
            results = session.generate([self._item()], GenerationParams(max_new_tokens=4))
        output = results[0].output
        assert output.output_ids.tolist() == [[5, 6]]
        assert output.adapted_input_ids.tolist() == [[0, 3, 4]]
        assert output.finish_reason == "stop"
        body = next(p for path, p in fake_server.requests if path == "/v1/completions")
        assert body["prompt"] == [0, 3, 4]
        assert body["return_token_ids"] is True
        assert body["max_tokens"] == 4

    def test_eos_maps_from_null_stop_reason(self, fake_server):
        backend = VLLMServeBackend(_serve_spec())
        with backend.open_session() as session:
            results = session.generate([self._item()], GenerationParams())
        assert results[0].output.finish_reason == "eos"

    def test_multiple_candidates_pack_per_item(self, fake_server):
        fake_server.completions[(0, 3, 4)] = [
            {"token_ids": [5, 6, 7], "finish_reason": "length", "stop_reason": None},
            {"token_ids": [8], "finish_reason": "stop", "stop_reason": None},
        ]
        backend = VLLMServeBackend(_serve_spec())
        with backend.open_session() as session:
            results = session.generate([self._item()], GenerationParams(n=2, max_new_tokens=3))
        output = results[0].output
        assert output.output_ids.shape == (2, 3)
        assert output.finish_reasons == ("length", "eos")

    def test_server_without_token_id_return_rejected(self, fake_server):
        fake_server.completions[(0, 3, 4)] = [{"text": "hi", "finish_reason": "stop"}]
        backend = VLLMServeBackend(_serve_spec())
        with backend.open_session() as session:
            with pytest.raises(PartialBatchError) as excinfo:
                session.generate([self._item()], GenerationParams())
        assert "return_token_ids" in str(excinfo.value)

    def test_transient_transport_failure_retries_to_success(self, fake_server):
        fake_server.fail_prompts[(0, 3, 4)] = 2  # two failures, third attempt succeeds
        backend = VLLMServeBackend(_serve_spec())
        with backend.open_session() as session:
            results = session.generate([self._item()], GenerationParams())
        assert len(results) == 1

    def test_persistent_failure_surfaces_partial_batch(self, fake_server):
        fake_server.fail_prompts[(0, 3)] = 99
        backend = VLLMServeBackend(_serve_spec())
        items = [self._item((0, 3, 4)), self._item((0, 3)), self._item((0, 4, 5))]
        with backend.open_session() as session:
            with pytest.raises(PartialBatchError) as excinfo:
                session.generate(items, GenerationParams())
        error = excinfo.value
        assert error.failed_indices == (1,)
        assert len(error.results) == 2
        assert isinstance(error.failures[0][1], TransportError)

    def test_unmapped_extra_key_raises_before_any_request(self, fake_server):
        backend = VLLMServeBackend(_serve_spec())
        request_count = len(fake_server.requests)
        with backend.open_session() as session:
            with pytest.raises(ValueError, match="num_beams"):
                session.generate([self._item()], GenerationParams(extra={"num_beams": 2}))
        assert len(fake_server.requests) == request_count

    def test_shared_seed_derives_distinct_request_seeds(self, fake_server):
        backend = VLLMServeBackend(_serve_spec())
        items = [self._item((0, 3, 4)), self._item((0, 3))]
        with backend.open_session() as session:
            session.generate(items, GenerationParams(seed=42))
        seeds = [
            payload["seed"] for path, payload in fake_server.requests
            if path == "/v1/completions"
        ]
        assert len(seeds) == 2
        assert seeds[0] != seeds[1]

    def test_hook_entries_rejected(self, fake_server):
        backend = VLLMServeBackend(_serve_spec())
        item = GenerationItem(
            prompt=PreparedPrompt.from_token_ids([0, 3]),
            state_entries=(HookEntry(hooks={"pre": []}),),
        )
        with backend.open_session() as session:
            with pytest.raises(UnsupportedOperationError, match="huggingface"):
                session.generate([item], GenerationParams())

    def test_intervention_entries_require_hook_plugin(self, fake_server):
        backend = VLLMServeBackend(_serve_spec())
        item = GenerationItem(
            prompt=PreparedPrompt.from_token_ids([0, 3]),
            state_entries=(InterventionEntry(spec=InterventionSpec()),),
        )
        with backend.open_session() as session:
            with pytest.raises(UnsupportedOperationError, match="hook_plugin"):
                session.generate([item], GenerationParams())


class TestServeSessionScore:

    def test_prompt_logprob_scoring(self, fake_server):
        fake_server.prompt_logprobs = -1.25
        backend = VLLMServeBackend(_serve_spec())
        items = [
            ScoringItem(
                prompt=PreparedPrompt.from_token_ids([0, 3, 4]),
                ref_output_ids=torch.tensor([[5, 6]]),
            ),
            ScoringItem(
                prompt=PreparedPrompt.from_token_ids([0, 4]),
                ref_output_ids=torch.tensor([[7, 3]]),
            ),
        ]
        with backend.open_session() as session:
            scored = session.score(items, GenerationParams())
        assert scored.shape == (2, 2)
        assert torch.allclose(scored, torch.full((2, 2), -1.25))
        body = next(p for path, p in fake_server.requests if path == "/v1/completions")
        assert body["prompt"] == [0, 3, 4, 5, 6]
        assert body["prompt_logprobs"] == 0

    def test_mismatched_ref_lengths_rejected(self, fake_server):
        backend = VLLMServeBackend(_serve_spec())
        items = [
            ScoringItem(prompt=PreparedPrompt.from_token_ids([0, 3]), ref_output_ids=torch.tensor([[5]])),
            ScoringItem(prompt=PreparedPrompt.from_token_ids([0, 3]), ref_output_ids=torch.tensor([[5, 6]])),
        ]
        with backend.open_session() as session:
            with pytest.raises(ValueError, match="reference length"):
                session.score(items, GenerationParams())

    def test_forward_kwargs_rejected(self, fake_server):
        backend = VLLMServeBackend(_serve_spec())
        item = ScoringItem(
            prompt=PreparedPrompt.from_token_ids([0, 3]), ref_output_ids=torch.tensor([[5]]),
        )
        with backend.open_session() as session:
            with pytest.raises(ValueError, match="output_attentions"):
                session.score([item], GenerationParams(extra={"output_attentions": True}))


class TestServeSessionLifecycle:

    def test_closed_session_rejected(self, fake_server):
        backend = VLLMServeBackend(_serve_spec())
        session = backend.open_session()
        session.close()
        with pytest.raises(RuntimeError, match="closed"):
            session.generate([], GenerationParams())

    def test_capture_unsupported(self, fake_server):
        backend = VLLMServeBackend(_serve_spec())
        with backend.open_session() as session:
            with pytest.raises(UnsupportedOperationError, match="capture"):
                session.capture([PreparedPrompt.from_token_ids([0, 3])], [0], "all_tokens")

    def test_layout_unresolvable_raises_on_access(self, fake_server):
        backend = VLLMServeBackend(_serve_spec())
        with backend.open_session() as session:
            with pytest.raises(RuntimeError, match="layout"):
                _ = session.layout


class TestEncoderDecoderSpecRejection:

    def test_local_encoder_decoder_config_rejected_for_vllm_kinds(self, tmp_path):
        config_dir = tmp_path / "enc-dec"
        T5Config().save_pretrained(config_dir)
        for kind in ("vllm", "vllm-serve"):
            with pytest.raises(ValueError, match="encoder-decoder"):
                BackendSpec(kind=kind, model=str(config_dir))

    def test_huggingface_kind_unaffected(self, tmp_path):
        config_dir = tmp_path / "enc-dec"
        T5Config().save_pretrained(config_dir)
        spec = BackendSpec(kind="huggingface", model=str(config_dir))
        assert spec.model == str(config_dir)

    def test_decoder_only_config_accepted(self, tmp_path):
        config_dir = tmp_path / "decoder"
        LlamaConfig(num_hidden_layers=1, hidden_size=8, num_attention_heads=2).save_pretrained(config_dir)
        spec = BackendSpec(kind="vllm", model=str(config_dir))
        assert spec.kind == "vllm"

    def test_unresolvable_reference_passes(self):
        spec = BackendSpec(kind="vllm", model="m")
        assert spec.model == "m"


def _discovery_payload(**engine_overrides):
    return {
        "plugin_version": "0.4.0",
        "vllm_version": "0.10.0",
        "active_worker": "unified",
        "intervention_kinds": {
            "transforms": ["additive", "projection", "rotation", "head_additive"],
            "modifiers": ["norm_preserving", "alignment_adaptive"],
            "scopes": ["all", "after_prompt", "last_k", "from_position"],
            "readouts": ["affine", "cosine", "projected_cosine"],
            "rules": ["per_key_threshold", "sum_threshold"],
            "constraints": {"head_additive": "tensor_parallel_size==1"},
        },
        "processor_kinds": {"processors": []},
        "capture_kinds": {
            "kinds": ["residual"],
            "locations": ["layer_output", "layer_input"],
            "modes": ["all_tokens", "last_token"],
        },
        "artifact_transports": ["shared_fs"],
        "engine": {
            "enforce_eager": True,
            "prefix_caching": True,
            "speculative_decoding": False,
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            **engine_overrides,
        },
        "model": {"id": "m"},
    }


def _mini_spec(scope=None, kind="additive"):
    from steerability.algorithms.state_control.common.lowering import artifact_id_for

    params = {"strength": 1.0} if kind in ("additive", "head_additive") else {}
    artifact_id, prepared = artifact_id_for({"vector": torch.ones(4)})
    op = {
        "layers": [0],
        "transform": {"kind": kind, **params, "modifiers": [], "artifact": artifact_id},
        "scope": scope or {"kind": "all"},
        "gate": None,
    }
    return InterventionSpec(ops=(op,), artifacts={artifact_id: prepared})


def _spec_item(spec, prompt=(0, 3)):
    return GenerationItem(
        prompt=PreparedPrompt.from_token_ids(list(prompt)),
        state_entries=(InterventionEntry(spec=spec),),
    )


class TestServeSpecLowering:

    def _plugin_backend(self, fake_server, tmp_path, **engine_overrides):
        fake_server.discovery = _discovery_payload(**engine_overrides)
        return VLLMServeBackend(_serve_spec(hook_plugin=True, artifact_dir=str(tmp_path)))

    def test_spec_bearing_request_carries_xargs_and_salt(self, fake_server, tmp_path):
        pytest.importorskip("vllm_hook_plugins")
        backend = self._plugin_backend(fake_server, tmp_path)
        spec = _mini_spec()
        with backend.open_session() as session:
            session.generate([_spec_item(spec)], GenerationParams(max_new_tokens=2))
        body = next(p for path, p in fake_server.requests if path == "/v1/completions")
        assert body["vllm_xargs"]["intervention_spec"] == spec.canonical()
        assert body["cache_salt"] == spec.salt()

    def test_spec_artifacts_materialize_into_artifact_dir(self, fake_server, tmp_path):
        pytest.importorskip("vllm_hook_plugins")
        backend = self._plugin_backend(fake_server, tmp_path)
        spec = _mini_spec()
        with backend.open_session() as session:
            session.generate([_spec_item(spec)], GenerationParams(max_new_tokens=2))
        (artifact_id,) = spec.artifact_ids()
        sha = artifact_id.removeprefix("sha256:")
        assert (tmp_path / sha[:2] / f"{sha}.safetensors").exists()

    def test_spec_free_requests_share_constant_backend_salt(self, fake_server, tmp_path):
        pytest.importorskip("vllm_hook_plugins")
        backend = self._plugin_backend(fake_server, tmp_path)
        items = [
            GenerationItem(prompt=PreparedPrompt.from_token_ids([0, 3])),
            GenerationItem(prompt=PreparedPrompt.from_token_ids([0, 4])),
        ]
        with backend.open_session() as session:
            session.generate(items, GenerationParams(max_new_tokens=2))
        salts = {p["cache_salt"] for path, p in fake_server.requests if path == "/v1/completions"}
        assert salts == {backend._plain_salt}
        assert backend._plain_salt != _mini_spec().salt()

    def test_plugin_free_requests_carry_no_salt(self, fake_server):
        backend = VLLMServeBackend(_serve_spec())
        with backend.open_session() as session:
            session.generate(
                [GenerationItem(prompt=PreparedPrompt.from_token_ids([0, 3]))],
                GenerationParams(max_new_tokens=2),
            )
        body = next(p for path, p in fake_server.requests if path == "/v1/completions")
        assert "cache_salt" not in body

    def test_speculative_decoding_engine_refuses_specs(self, fake_server, tmp_path):
        backend = self._plugin_backend(fake_server, tmp_path, speculative_decoding=True)
        with backend.open_session() as session:
            with pytest.raises(UnsupportedOperationError, match="speculative decoding"):
                session.generate([_spec_item(_mini_spec())], GenerationParams())

    def test_non_eager_engine_refuses_specs(self, fake_server, tmp_path):
        backend = self._plugin_backend(fake_server, tmp_path, enforce_eager=False)
        with backend.open_session() as session:
            with pytest.raises(UnsupportedOperationError, match="enforce_eager"):
                session.generate([_spec_item(_mini_spec())], GenerationParams())

    def test_constrained_kind_refused_under_tensor_parallelism(self, fake_server, tmp_path):
        backend = self._plugin_backend(fake_server, tmp_path, tensor_parallel_size=2)
        with backend.open_session() as session:
            with pytest.raises(UnsupportedOperationError, match="tensor_parallel_size=2"):
                session.generate([_spec_item(_mini_spec(kind="head_additive"))], GenerationParams())

    def test_scoring_remaps_after_prompt_to_from_position(self, fake_server, tmp_path):
        pytest.importorskip("vllm_hook_plugins")
        fake_server.prompt_logprobs = -0.5
        backend = self._plugin_backend(fake_server, tmp_path)
        spec = _mini_spec(scope={"kind": "after_prompt"})
        item = ScoringItem(
            prompt=PreparedPrompt.from_token_ids([0, 3, 4]),
            ref_output_ids=torch.tensor([[5, 6]]),
            state_entries=(InterventionEntry(spec=spec),),
        )
        with backend.open_session() as session:
            session.score([item], GenerationParams())
        body = next(p for path, p in fake_server.requests if path == "/v1/completions")
        import json as json_module
        sent = json_module.loads(body["vllm_xargs"]["intervention_spec"])
        assert sent["ops"][0]["scope"] == {"kind": "from_position", "position": 3}
        assert body["cache_salt"] != spec.salt()

    def test_scoring_all_scope_travels_unchanged(self, fake_server, tmp_path):
        pytest.importorskip("vllm_hook_plugins")
        fake_server.prompt_logprobs = -0.5
        backend = self._plugin_backend(fake_server, tmp_path)
        spec = _mini_spec(scope={"kind": "all"})
        item = ScoringItem(
            prompt=PreparedPrompt.from_token_ids([0, 3, 4]),
            ref_output_ids=torch.tensor([[5, 6]]),
            state_entries=(InterventionEntry(spec=spec),),
        )
        with backend.open_session() as session:
            session.score([item], GenerationParams())
        body = next(p for path, p in fake_server.requests if path == "/v1/completions")
        assert body["cache_salt"] == spec.salt()


class TestSpecRejectionMapping:

    def test_kind_and_constraint_codes_are_support_facts(self):
        from steerability.backends.vllm import raise_for_spec_rejection

        with pytest.raises(UnsupportedOperationError, match="E_UNKNOWN_KIND"):
            raise_for_spec_rejection(
                "HTTP 400: E_UNKNOWN_KIND at ops[0].gate.kind: gate kind 'probe_sum' is not served"
            )
        with pytest.raises(UnsupportedOperationError, match="E_CONSTRAINT"):
            raise_for_spec_rejection(
                "HTTP 400: E_CONSTRAINT at ops[0].transform.kind: kind 'head_additive' requires tensor_parallel_size==1"
            )

    def test_malformed_spec_codes_raise_value_error(self):
        from steerability.backends.vllm import raise_for_spec_rejection

        with pytest.raises(ValueError, match="E_BAD_PARAM at ops\\[0\\]\\.transform\\.strength"):
            raise_for_spec_rejection(
                "HTTP 400: E_BAD_PARAM at ops[0].transform.strength: 'strength' must be a number"
            )

    def test_plain_message_does_not_raise(self):
        from steerability.backends.vllm import raise_for_spec_rejection

        raise_for_spec_rejection("HTTP 400: model not found")


class TestMergeInterventionSpecs:

    def test_ops_concatenate_and_artifacts_union(self):
        from steerability.backends.vllm import merge_intervention_specs

        first = _mini_spec()
        second = _mini_spec(scope={"kind": "after_prompt"})
        merged = merge_intervention_specs([first, second])
        assert len(merged.ops) == 2
        assert set(merged.artifacts) == set(first.artifacts) | set(second.artifacts)


class TestServeConstraintLowering:

    def test_constraint_entry_renders_guided_field(self, fake_server):
        from steerability.algorithms.core.execution import ConstraintEntry, ConstraintSource

        backend = VLLMServeBackend(_serve_spec())
        item = GenerationItem(
            prompt=PreparedPrompt.from_token_ids([0, 3]),
            output_entries=(ConstraintEntry(
                source=ConstraintSource(kind="json_schema", value={"type": "object"}),
            ),),
        )
        with backend.open_session() as session:
            session.generate([item], GenerationParams(max_new_tokens=4))
        body = next(p for path, p in fake_server.requests if path == "/v1/completions")
        assert body["guided_json"] == {"type": "object"}

    def test_choice_constraint_renders_guided_choice(self, fake_server):
        from steerability.algorithms.core.execution import ConstraintEntry, ConstraintSource

        backend = VLLMServeBackend(_serve_spec())
        item = GenerationItem(
            prompt=PreparedPrompt.from_token_ids([0, 3]),
            output_entries=(ConstraintEntry(
                source=ConstraintSource(kind="choice", value=("cat", "dog")),
            ),),
        )
        with backend.open_session() as session:
            session.generate([item], GenerationParams())
        body = next(p for path, p in fake_server.requests if path == "/v1/completions")
        assert body["guided_choice"] == ["cat", "dog"]

    def test_scoring_with_constraint_entry_refused(self, fake_server):
        from steerability.algorithms.core.execution import ConstraintEntry, ConstraintSource

        backend = VLLMServeBackend(_serve_spec())
        item = ScoringItem(
            prompt=PreparedPrompt.from_token_ids([0, 3]),
            ref_output_ids=torch.tensor([[5, 6]]),
            output_entries=(ConstraintEntry(
                source=ConstraintSource(kind="regex", value="cat"),
            ),),
        )
        with backend.open_session() as session:
            with pytest.raises(UnsupportedOperationError, match="prompt logprobs"):
                session.score([item], GenerationParams())

    def test_two_constraints_per_item_refused(self, fake_server):
        from steerability.algorithms.core.execution import ConstraintEntry, ConstraintSource

        backend = VLLMServeBackend(_serve_spec())
        item = GenerationItem(
            prompt=PreparedPrompt.from_token_ids([0, 3]),
            output_entries=(
                ConstraintEntry(source=ConstraintSource(kind="regex", value="cat")),
                ConstraintEntry(source=ConstraintSource(kind="regex", value="dog")),
            ),
        )
        with backend.open_session() as session:
            with pytest.raises(UnsupportedOperationError, match="one structured-output constraint"):
                session.generate([item], GenerationParams())

    def test_pipeline_lowers_declarative_constraint_to_serve(self, fake_server):
        from steerability.algorithms.core.steering_pipeline import SteeringPipeline
        from steerability.algorithms.output_control.constrained_decoding import ConstrainedDecoding
        from tests.utils.tiny_models import tiny_llama

        control = ConstrainedDecoding(regex="cat|dog", include_in_scoring=False)
        pipeline = SteeringPipeline(
            controls=[control], backend=_serve_spec(),
        )
        pipeline.model = tiny_llama(num_layers=2, hidden=16, heads=2)
        pipeline.tokenizer = wordlevel_tokenizer()
        pipeline.steer()
        text = pipeline.generate(text="the cat", max_new_tokens=4, do_sample=False)
        assert isinstance(text, str)
        body = next(p for path, p in fake_server.requests if path == "/v1/completions")
        assert body["guided_regex"] == "cat|dog"


class TestStageArtifacts:

    def test_stage_writes_into_configured_registry(self, fake_server, tmp_path):
        pytest.importorskip("vllm_hook_plugins")
        fake_server.discovery = _discovery_payload()
        backend = VLLMServeBackend(
            _serve_spec(hook_plugin=True, artifact_dir=str(tmp_path)),
        )
        spec = _mini_spec()
        backend.stage_artifacts(spec.artifacts)
        (artifact_id,) = spec.artifact_ids()
        sha = artifact_id.removeprefix("sha256:")
        assert (tmp_path / sha[:2] / f"{sha}.safetensors").exists()

    def test_stage_without_registry_puts_to_the_artifact_route(self, fake_server, monkeypatch):
        fake_server.discovery = _discovery_payload()
        backend = VLLMServeBackend(_serve_spec(hook_plugin=True))
        puts: list[tuple[str, int]] = []
        monkeypatch.setattr(
            VLLMServeBackend, "_put_bytes",
            lambda self, path, data: puts.append((path, len(data))),
        )
        spec = _mini_spec()
        backend.stage_artifacts(spec.artifacts)
        (artifact_id,) = spec.artifact_ids()
        assert puts == [(f"/v1/hook/artifacts/{artifact_id}", puts[0][1])]
        assert puts[0][1] > 0
        # idempotent: a second staging of the same content is a no-op
        backend.stage_artifacts(spec.artifacts)
        assert len(puts) == 1

    def test_stage_is_a_noop_without_payloads(self, fake_server):
        backend = VLLMServeBackend(_serve_spec())
        backend.stage_artifacts({})


class TestSharedFsVisibility:
    """`stage_artifacts` verifies shared_fs visibility when the server advertises its registry."""

    def _backend(self, fake_server, monkeypatch, tmp_path, registry_root):
        payload = _discovery_payload()
        if registry_root is not None:
            payload["artifact_registry_root"] = registry_root
        fake_server.discovery = payload
        monkeypatch.setattr(
            "steerability.backends.vllm.backend._ArtifactUploader.upload_payloads",
            lambda self, payloads: None,
        )
        spec = _serve_spec(hook_plugin=True, artifact_dir=str(tmp_path))
        return VLLMServeBackend(spec)

    def test_invisible_artifact_raises_with_both_roots(self, fake_server, monkeypatch, tmp_path):
        backend = self._backend(fake_server, monkeypatch, tmp_path, "/srv/registry")
        monkeypatch.setattr(VLLMServeBackend, "_head_ok", lambda self, path: False)
        with pytest.raises(ValueError, match="not visible to the server's registry"):
            backend.stage_artifacts({"sha256:" + "ab" * 32: {}})

    def test_visible_artifact_passes(self, fake_server, monkeypatch, tmp_path):
        backend = self._backend(fake_server, monkeypatch, tmp_path, "/srv/registry")
        monkeypatch.setattr(VLLMServeBackend, "_head_ok", lambda self, path: True)
        backend.stage_artifacts({"sha256:" + "ab" * 32: {}})

    def test_server_without_advertised_root_skips_probe(self, fake_server, monkeypatch, tmp_path):
        backend = self._backend(fake_server, monkeypatch, tmp_path, None)

        def _fail(self, path):
            raise AssertionError("probe must not run without an advertised registry root")

        monkeypatch.setattr(VLLMServeBackend, "_head_ok", _fail)
        backend.stage_artifacts({"sha256:" + "ab" * 32: {}})


class TestConfigLayout:
    """`_config_layout` reads the text sub-config of a composite checkpoint from disk."""

    def test_gemma3_config_dir_reports_text_facts(self, tmp_path):
        from transformers import Gemma3Config, Gemma3TextConfig, SiglipVisionConfig

        from steerability.backends.vllm.backend import _config_layout

        text = Gemma3TextConfig(
            hidden_size=32, intermediate_size=64, num_hidden_layers=4,
            num_attention_heads=4, num_key_value_heads=4, head_dim=8, vocab_size=100,
        )
        vision = SiglipVisionConfig(
            hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
            image_size=16, patch_size=8,
        )
        Gemma3Config(text_config=text, vision_config=vision, mm_tokens_per_image=4).save_pretrained(tmp_path)

        facts = _config_layout(str(tmp_path))
        assert facts is not None
        assert facts.num_layers == 4
        assert facts.hidden_size == 32
        assert facts.num_attention_heads == 4
        assert facts.head_dim == 8
        assert facts.model_type == "gemma3"
