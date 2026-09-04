"""The vLLM backends: the offline engine (`"vllm"`) and the OpenAI-compatible server
(`"vllm-serve"`).

This module holds the backend-side helpers (the artifact uploader, layout and tokenizer
resolution, discovery reconciliation) and the two backend classes. It imports cleanly without
vLLM installed: constructing `VLLMBackend` requires the `vllm` optional dependency (it boots an
engine); `VLLMServeBackend` needs only a reachable vLLM server. Every `from vllm ...` and
`from vllm_hook_plugins ...` import stays inside a function or method body.
"""
import dataclasses
import gc
import hashlib
import json
import logging
import os
import urllib.error
import urllib.request
import uuid
from collections.abc import Sequence
from typing import Any

import torch

from aisteer360.algorithms.core.execution.backend import Backend
from aisteer360.algorithms.core.execution.contracts import BackendCapabilities
from aisteer360.algorithms.core.execution.fanout import TransportError
from aisteer360.algorithms.core.execution.payloads import (
    Artifact,
    CheckpointArtifact,
    InterventionSpec,
    LoRAArtifact,
    ModelFacts,
)
from aisteer360.algorithms.core.execution.spec import BackendSpec
from aisteer360.algorithms.core.internals.fingerprint import is_absent_chat_template_fingerprint
from aisteer360.backends.vllm.capabilities import _DISCOVERY_CACHE, _reconcile_discovery, _vllm_capabilities
from aisteer360.backends.vllm.rendering import raise_for_spec_rejection
from aisteer360.backends.vllm.session import VLLMOfflineSession, VLLMServeSession
from aisteer360.utils.optional import require
from aisteer360.utils.tokenization import ensure_pad_token

logger = logging.getLogger(__name__)

_DEFAULT_REQUEST_TIMEOUT = 120.0
_DEFAULT_MAX_CONCURRENCY = 8
_DEFAULT_MAX_ATTEMPTS = 3

_STRUCTURED_OUTPUT_ENGINE_KEYS: tuple[str, ...] = (
    "structured_outputs_config",
    "guided_decoding_backend",
    "guided_decoding_disable_any_whitespace",
)


def _structured_outputs_engine_kwargs() -> dict[str, Any]:
    """Engine kwargs selecting xgrammar with compact whitespace, in the installed vLLM's vocabulary.

    vLLM renamed guided decoding to structured outputs; `EngineArgs` carries either
    `structured_outputs_config` (current surface) or `guided_decoding_backend` (legacy surface).
    The probe reads the dataclass fields rather than a version number. On the legacy surface the
    compact-whitespace switch is set only where the field exists, so json outputs may carry
    whitespace the in-process automaton does not emit. Returns an empty mapping when
    `EngineArgs` cannot be imported or inspected, in which case no default is applied.

    Returns:
        Keyword arguments for `vllm.LLM`, or an empty mapping.
    """
    try:
        from vllm import EngineArgs

        names = {field.name for field in dataclasses.fields(EngineArgs)}
    except (ImportError, TypeError):
        return {}
    if "structured_outputs_config" in names:
        return {"structured_outputs_config": {"disable_any_whitespace": True, "backend": "xgrammar"}}
    kwargs: dict[str, Any] = {"guided_decoding_backend": "xgrammar"}
    if "guided_decoding_disable_any_whitespace" in names:
        kwargs["guided_decoding_disable_any_whitespace"] = True
    return kwargs


class _ArtifactUploader:
    """Materializes spec tensor payloads into the registry root the serving engine reads."""

    def __init__(self, root: str | None):
        self._root = root
        self._registry = None
        self._written: set[str] = set()

    def upload(self, spec: InterventionSpec) -> None:
        if spec.artifacts:
            self.upload_payloads(spec.artifacts)

    def upload_payloads(self, payloads) -> None:
        """Write content-addressed payloads into the registry, verifying each id."""
        if not payloads:
            return
        if self._registry is None:
            artifacts_module = require("vllm_hook_plugins.core.artifacts")
            self._registry = artifacts_module.ArtifactRegistry(self._root)
        for artifact_id, tensors in payloads.items():
            if artifact_id in self._written:
                continue
            written_id = self._registry.write(dict(tensors))
            if written_id != artifact_id:
                raise ValueError(
                    f"Artifact registry wrote {written_id} for a payload the spec references as "
                    f"{artifact_id}; the client and registry disagree on content addressing."
                )
            self._written.add(artifact_id)


def _reject_encoder_decoder(model_ref: str, trust_remote_code: bool = False) -> None:
    """Reject encoder-decoder models for vLLM execution (in-process only per the seam)."""
    from transformers import AutoConfig

    try:
        config = AutoConfig.from_pretrained(model_ref, trust_remote_code=trust_remote_code)
    except Exception:
        return
    if getattr(config, "is_encoder_decoder", False):
        raise ValueError(
            f"Model {model_ref!r} is an encoder-decoder model; encoder-decoder execution is "
            "in-process only. Run this pipeline on the huggingface backend."
        )


def _config_layout(model_ref: str, trust_remote_code: bool = False) -> ModelFacts | None:
    """A client-side `ModelFacts` from the model config, or None when unresolvable.

    The fingerprint hashes the config JSON (volatile name/version fields removed), so it
    identifies the architecture and configuration rather than the weights.
    """
    from transformers import AutoConfig

    try:
        config = AutoConfig.from_pretrained(model_ref, trust_remote_code=trust_remote_code)
    except Exception:
        return None
    hidden_size = getattr(config, "hidden_size", None)
    num_heads = getattr(config, "num_attention_heads", None)
    head_dim = getattr(config, "head_dim", None)
    if head_dim is None and hidden_size and num_heads:
        head_dim = hidden_size // num_heads
    dtype = getattr(config, "torch_dtype", None)
    config_dict = {
        key: value for key, value in config.to_dict().items()
        if key not in ("_name_or_path", "transformers_version")
    }
    digest = hashlib.sha256(
        json.dumps(config_dict, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]
    return ModelFacts(
        num_layers=getattr(config, "num_hidden_layers", 0),
        hidden_size=hidden_size or 0,
        num_attention_heads=num_heads,
        head_dim=head_dim,
        dtype=str(dtype).removeprefix("torch.") if dtype is not None else "unknown",
        model_fingerprint=digest,
        model_type=getattr(config, "model_type", None),
        model_ref=model_ref,
    )


def _client_tokenizer(source: str, trust_remote_code: bool = False):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=trust_remote_code)
    return ensure_pad_token(tokenizer)


def _split_artifacts(artifacts: Sequence[Artifact]) -> tuple[CheckpointArtifact | None, LoRAArtifact | None]:
    checkpoint = next((a for a in artifacts if isinstance(a, CheckpointArtifact)), None)
    lora = next((a for a in artifacts if isinstance(a, LoRAArtifact)), None)
    return checkpoint, lora


class VLLMBackend(Backend):
    """The offline vLLM engine backend.

    Boots one engine per backend instance from the spec (`engine_kwargs` option forwarded to
    `vllm.LLM`); requires the `vllm` optional dependency. A `CheckpointArtifact` overrides the
    served model reference and a `LoRAArtifact` attaches as a LoRA request on every generation.
    When the spec declares `hook_plugin`, the unified worker is selected via
    `VLLM_HOOK_WORKER=unified` and the discovery payload is fetched once and cached by spec
    hash. Capability advertisement is available through `capabilities_for_spec` without
    constructing the backend.
    """

    def __init__(self, spec: BackendSpec, artifacts: Sequence[Artifact] = ()) -> None:
        if spec.kind != "vllm":
            raise ValueError(f"VLLMBackend requires a 'vllm' spec; got kind {spec.kind!r}.")
        self.spec = spec
        self._released = False
        require("vllm")
        from vllm import LLM

        checkpoint, lora = _split_artifacts(artifacts)
        model_ref = checkpoint.path if checkpoint is not None else spec.model
        if model_ref is None:
            raise ValueError("VLLMBackend needs a model reference on the spec or a checkpoint artifact.")
        trust_remote_code = bool(spec.get_option("trust_remote_code", default=False))
        _reject_encoder_decoder(model_ref, trust_remote_code)

        engine_kwargs = dict(spec.get_option("engine_kwargs", default={}) or {})
        # default to a compact xgrammar grammar so json constraints match the in-process automaton
        # (disable_any_whitespace needs an explicit backend); a caller key in either vocabulary wins
        if not any(key in engine_kwargs for key in _STRUCTURED_OUTPUT_ENGINE_KEYS):
            engine_kwargs.update(_structured_outputs_engine_kwargs())
        if lora is not None:
            engine_kwargs.setdefault("enable_lora", True)
        if trust_remote_code:
            engine_kwargs.setdefault("trust_remote_code", True)
        if spec.get_option("hook_plugin"):
            # worker hooks do not run under CUDA-graph replay; spec construction rejects an
            # explicit False, so this only fills the default
            engine_kwargs.setdefault("enforce_eager", True)

        # the worker-selection variable is scoped to this engine's boot so a later plugin-free
        # engine in the same process is unaffected
        previous_worker = os.environ.get("VLLM_HOOK_WORKER")
        if spec.get_option("hook_plugin"):
            os.environ["VLLM_HOOK_WORKER"] = "unified"
        try:
            self._llm = LLM(model=model_ref, **engine_kwargs)
        finally:
            if spec.get_option("hook_plugin"):
                if previous_worker is None:
                    os.environ.pop("VLLM_HOOK_WORKER", None)
                else:
                    os.environ["VLLM_HOOK_WORKER"] = previous_worker

        # the pipeline records a backend only when the constructor returns, so a failure from
        # here on must release the engine before propagating
        try:
            self._lora_request = None
            if lora is not None:
                from vllm.lora.request import LoRARequest

                self._lora_request = LoRARequest("steered", 1, lora.path)

            tokenizer_source = (
                spec.get_option("tokenizer_name_or_path")
                or model_ref
            )
            self.tokenizer = _client_tokenizer(tokenizer_source, trust_remote_code)
            self._layout = _config_layout(model_ref, trust_remote_code)
            self._plain_salt = uuid.uuid4().hex
            self._artifact_uploader = _ArtifactUploader(spec.get_option("artifact_dir"))
            self._discovery: dict | None = None
            if spec.get_option("hook_plugin"):
                self._discovery = self._fetch_discovery()
        except Exception:
            self.release()
            raise

    def stage_artifacts(self, payloads) -> None:
        """Write each content-addressed artifact into the plugin registry the engine reads.

        The offline engine shares the process's filesystem, so staging is a registry write
        (idempotent, verified against the content address).
        """
        self._artifact_uploader.upload_payloads(payloads)

    def _fetch_discovery(self) -> dict | None:
        cached = _DISCOVERY_CACHE.get(self.spec.spec_hash)
        if cached is not None:
            return cached
        payload = None
        for target in (self._llm, getattr(self._llm, "llm_engine", None)):
            rpc = getattr(target, "collective_rpc", None)
            if callable(rpc):
                try:
                    replies = rpc("hook_capabilities")
                except Exception as error:
                    logger.warning("vLLM-Hook discovery failed: %s", error)
                    return None
                payload = next((reply for reply in replies if reply), None)
                break
        if payload is None:
            logger.warning(
                "vLLM-Hook discovery returned no payload; is VLLM_HOOK_WORKER=unified active?"
            )
            return None
        _DISCOVERY_CACHE[self.spec.spec_hash] = payload
        _reconcile_discovery(self.spec, self.capabilities_for_spec(self.spec), payload)
        return payload

    @classmethod
    def capabilities_for_spec(cls, spec: BackendSpec) -> BackendCapabilities:
        """The capability advertisement implied by `spec`."""
        return _vllm_capabilities(spec, offline=True)

    def open_session(self) -> "VLLMOfflineSession":
        """Open a request session over the shared engine.

        Raises:
            RuntimeError: If the backend has been released.
        """
        self._require_llm()
        return VLLMOfflineSession(self)

    def _require_llm(self):
        """The live engine, or a `RuntimeError` when the backend has been released."""
        if self._llm is None:
            raise RuntimeError(
                "This VLLMBackend was released; construct a new backend (or a new "
                "SteeringPipeline operation, which reconstructs backends automatically)."
            )
        return self._llm

    def release(self) -> None:
        """Shut the engine down explicitly and mark the backend unusable.

        Release is idempotent; after it, a new backend must be constructed. The distributed-state
        teardown is process-global, so release assumes no other live vLLM engine in the process.
        Ray-based executors are out of scope. Engine-touching calls on any still-open session raise
        after release.
        """
        if self._released:
            return
        self._released = True
        llm = self._llm
        self._llm = None
        self._lora_request = None

        for resolve in (
            lambda: getattr(llm, "shutdown", None),
            lambda: getattr(getattr(llm, "llm_engine", None), "shutdown", None),
            lambda: getattr(
                getattr(getattr(llm, "llm_engine", None), "engine_core", None), "shutdown", None
            ),
        ):
            shutdown = resolve()
            if callable(shutdown):
                try:
                    shutdown()
                except Exception:
                    logger.warning("vLLM engine shutdown hop failed; continuing.", exc_info=True)
                break

        del llm
        gc.collect()

        try:
            from vllm.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel

            destroy_model_parallel()
            destroy_distributed_environment()
        except Exception:
            logger.warning("vLLM distributed-state teardown failed; continuing.", exc_info=True)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class VLLMServeBackend(Backend):
    """The vLLM OpenAI-compatible server backend.

    Targets a vLLM server rather than an arbitrary OpenAI-compatible endpoint: construction
    verifies the server's version surface (`GET /version`), fetches the plugin discovery payload
    (`GET /v1/hook/capabilities`) when the spec declares `hook_plugin`, and checks the served
    model id against the spec (or serves the pipeline's structural artifacts). Prompts submit as
    token ids on the completions endpoint with the token-id return option; the chat endpoint is
    not used. Requires no local vLLM installation.

    Spec options: `base_url` (required, the server root), `api_key`, `request_timeout`,
    `max_concurrency`, `max_retries`, `retry_backoff`, `tokenizer_name_or_path`,
    `trust_remote_code`, `hook_plugin`.
    """

    def __init__(self, spec: BackendSpec, artifacts: Sequence[Artifact] = ()) -> None:
        if spec.kind != "vllm-serve":
            raise ValueError(f"VLLMServeBackend requires a 'vllm-serve' spec; got kind {spec.kind!r}.")
        self.spec = spec
        base_url = spec.get_option("base_url")
        if not base_url:
            raise ValueError("VLLMServeBackend requires a 'base_url' option on the spec.")
        self._base_url = base_url.rstrip("/").removesuffix("/v1")
        self._api_key = spec.get_option("api_key")
        self._timeout = float(spec.get_option("request_timeout", default=_DEFAULT_REQUEST_TIMEOUT))
        self.max_concurrency = int(spec.get_option("max_concurrency", default=_DEFAULT_MAX_CONCURRENCY))
        self.max_attempts = int(spec.get_option("max_retries", default=_DEFAULT_MAX_ATTEMPTS))
        self.backoff_base = float(spec.get_option("retry_backoff", default=0.5))
        trust_remote_code = bool(spec.get_option("trust_remote_code", default=False))

        version = self._get_json("/version")
        if not isinstance(version, dict) or "version" not in version:
            raise ValueError(
                f"The endpoint at {self._base_url} does not expose the vLLM version surface; "
                "only vLLM servers are supported."
            )

        self._discovery: dict | None = None
        if spec.get_option("hook_plugin"):
            self._discovery = _DISCOVERY_CACHE.get(spec.spec_hash)
            if self._discovery is None:
                try:
                    self._discovery = self._get_json("/v1/hook/capabilities")
                except (TransportError, ValueError) as error:
                    raise ValueError(
                        f"The spec declares hook_plugin but {self._base_url} serves no "
                        f"/v1/hook/capabilities discovery surface: {error}"
                    ) from error
                _DISCOVERY_CACHE[spec.spec_hash] = self._discovery
                _reconcile_discovery(spec, self.capabilities_for_spec(spec), self._discovery)

        checkpoint, lora = _split_artifacts(artifacts)
        expected_model = checkpoint.path if checkpoint is not None else spec.model
        if lora is not None:
            self._served_model = self._load_lora_adapter(lora)
        else:
            served = self._served_model_ids()
            if expected_model is None:
                if len(served) != 1:
                    raise ValueError(
                        f"The spec names no model and the server serves {served}; set the "
                        "spec's model to disambiguate."
                    )
                self._served_model = served[0]
            elif expected_model in served:
                self._served_model = expected_model
            else:
                raise ValueError(
                    f"The server at {self._base_url} serves {served}, not the configured "
                    f"model {expected_model!r}."
                )

        tokenizer_source = (
            spec.get_option("tokenizer_name_or_path")
            or (checkpoint.path if checkpoint is not None else None)
            or (lora.base_model if lora is not None else None)
            or spec.model
        )
        if tokenizer_source is None:
            tokenizer_source = self._served_model
        self.tokenizer = _client_tokenizer(tokenizer_source, trust_remote_code)
        self._layout = _config_layout(
            (checkpoint.path if checkpoint is not None else None) or spec.model or self._served_model,
            trust_remote_code,
        )
        self._plain_salt = uuid.uuid4().hex
        # spec artifacts write to the registry root the server reads; shared_fs visibility
        # is verified against the server after writing (see stage_artifacts)
        self._artifact_uploader = _ArtifactUploader(spec.get_option("artifact_dir"))
        if self._discovery is not None:
            self._verify_fingerprints()

    def _served_model_ids(self) -> list[str]:
        payload = self._get_json("/v1/models")
        return [entry.get("id") for entry in payload.get("data", []) if isinstance(entry, dict)]

    def stage_artifacts(self, payloads) -> None:
        """Make each content-addressed artifact available to the serving engine.

        With an `artifact_dir` option the payloads are written into that registry root, which
        must be the server's registry directory (its `VLLM_HOOK_REGISTRY_DIR`) on a shared
        filesystem; visibility is verified through the server's artifact route when the
        discovery payload advertises `artifact_registry_root`. Otherwise each payload is PUT
        to the plugin's artifact route (`/v1/hook/artifacts/{id}`, body safetensors bytes, id
        verified server-side); already-exists is success.
        """
        if not payloads:
            return
        if self.spec.get_option("artifact_dir"):
            self._artifact_uploader.upload_payloads(payloads)
            self._verify_shared_fs_visibility(payloads)
            return
        import safetensors.torch

        for artifact_id, tensors in payloads.items():
            if artifact_id in self._artifact_uploader._written:
                continue
            data = safetensors.torch.save({name: tensors[name] for name in sorted(tensors)})
            self._put_bytes(f"/v1/hook/artifacts/{artifact_id}", data)
            self._artifact_uploader._written.add(artifact_id)

    def _verify_shared_fs_visibility(self, payloads) -> None:
        """Probe that shared_fs artifacts are visible to the server's registry.

        Gated on the discovery payload advertising `artifact_registry_root` (servers that
        advertise it also serve `HEAD /v1/hook/artifacts/{id}`); older servers skip the
        probe and keep the write-and-trust behavior.
        """
        server_root = (self._discovery or {}).get("artifact_registry_root")
        if not server_root:
            return
        client_root = os.path.abspath(self.spec.get_option("artifact_dir"))
        for artifact_id in payloads:
            if self._head_ok(f"/v1/hook/artifacts/{artifact_id}"):
                continue
            raise ValueError(
                f"artifact {artifact_id} written under {client_root} is not visible to the "
                f"server's registry ({server_root}); the shared_fs transport requires "
                "artifact_dir and the server's VLLM_HOOK_REGISTRY_DIR to name the same "
                "directory. Set VLLM_HOOK_REGISTRY_DIR on the server, point artifact_dir at "
                "the server's registry root, or drop artifact_dir to use the HTTP artifact "
                "route."
            )

    def _head_ok(self, path: str) -> bool:
        """HEAD a server path; True on 200, False on 404, raise otherwise."""
        request = urllib.request.Request(f"{self._base_url}{path}", method="HEAD")
        if self._api_key:
            request.add_header("Authorization", f"Bearer {self._api_key}")
        try:
            with urllib.request.urlopen(request, timeout=self._timeout):
                return True
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return False
            body = error.read().decode("utf-8", errors="replace")
            raise ValueError(f"HTTP {error.code} from {self._base_url}{path}: {body}") from error

    def _put_bytes(self, path: str, data: bytes) -> None:
        """PUT raw bytes to the server, mapping a missing route to a configuration error."""
        import urllib.error
        import urllib.request

        url = f"{self._base_url}{path}"
        request = urllib.request.Request(url, data=data, method="PUT")
        request.add_header("Content-Type", "application/octet-stream")
        if self._api_key:
            request.add_header("Authorization", f"Bearer {self._api_key}")
        try:
            with urllib.request.urlopen(request, timeout=self._timeout):
                return
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            if error.code in (404, 405):
                raise ValueError(
                    f"{self._base_url} serves no artifact route ({error.code}); update the "
                    "server's vllm_hook_plugins, or configure artifact_dir on a filesystem "
                    "shared with the server."
                ) from error
            raise_for_spec_rejection(body)
            raise ValueError(f"HTTP {error.code} from {url}: {body}") from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise TransportError(f"artifact upload to {url} failed: {error}") from error

    def _load_lora_adapter(self, lora: LoRAArtifact) -> str:
        served = self._served_model_ids()
        base = lora.base_model or self.spec.model
        if base and base not in served:
            raise ValueError(
                f"The server at {self._base_url} serves {served}, not the adapter's base "
                f"model {base!r}."
            )
        # the adapter name keys on path plus provenance, so a retrained adapter at the same
        # path loads as a new server-side adapter rather than reusing stale weights
        identity = f"{lora.path}:{lora.provenance.model_fingerprint or ''}"
        adapter_name = f"steered-{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:8]}"
        if adapter_name in served:
            return adapter_name
        try:
            self._post_json(
                "/v1/load_lora_adapter",
                {"lora_name": adapter_name, "lora_path": lora.path},
                expect_json=False,
            )
        except (TransportError, ValueError) as error:
            raise ValueError(
                f"Could not load the LoRA artifact at {lora.path!r} onto the server "
                f"(dynamic adapter loading requires VLLM_ALLOW_RUNTIME_LORA_UPDATING): {error}"
            ) from error
        return adapter_name

    def _verify_fingerprints(self) -> None:
        """Verify the client tokenizer against the discovery payload's fingerprint recipes.

        Uses the plugin's engine-free `core.fingerprints` when the `vllm_hook_plugins` package
        is installed; mismatches warn rather than raise. Without the package, verification is
        skipped with a warning. A served fingerprint equal to the absent-template digest means
        the server exposes no chat template, so the comparison is skipped since a mismatch
        against it would reflect exposure rather than divergence.
        """
        model_block = (self._discovery or {}).get("model", {})
        remote_chat = model_block.get("chat_template_fingerprint")
        if remote_chat is None:
            return
        if is_absent_chat_template_fingerprint(remote_chat):
            logger.debug(
                "The server at %s does not expose a chat template (fingerprint %s is the "
                "absent-template digest); skipping the chat template comparison.",
                self._base_url, remote_chat,
            )
            return
        try:
            from vllm_hook_plugins.core.fingerprints import chat_template_fingerprint
        except ImportError:
            logger.warning(
                "Install vllm-hook-plugins to verify the client tokenizer against the server's "
                "fingerprints; skipping verification."
            )
            return
        local_chat = chat_template_fingerprint(getattr(self.tokenizer, "chat_template", None))
        if local_chat != remote_chat:
            logger.warning(
                "Client chat template (fingerprint %s) differs from the served one (%s); "
                "templated prompts may diverge from server-side expectations.",
                local_chat, remote_chat,
            )

    def _request_json(self, path: str, payload: dict | None, expect_json: bool = True) -> dict:
        url = f"{self._base_url}{path}"
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(url, data=data, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as error:
            body = ""
            try:
                body = error.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            # 5xx, timeouts, and rate limiting are transport-level and safe to retry
            if error.code >= 500 or error.code in (408, 429):
                raise TransportError(f"HTTP {error.code} from {url}: {body}") from error
            # admission rejections carry the plugin's E_* code and JSON path verbatim
            raise_for_spec_rejection(body)
            raise ValueError(f"HTTP {error.code} from {url}: {body}") from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise TransportError(f"Request to {url} failed: {error}") from error
        if not expect_json:
            return {"text": body}
        try:
            return json.loads(body)
        except json.JSONDecodeError as error:
            raise ValueError(f"Non-JSON response from {url}: {error}") from error

    def _get_json(self, path: str) -> dict:
        return self._request_json(path, None)

    def _post_json(self, path: str, payload: dict, expect_json: bool = True) -> dict:
        return self._request_json(path, payload, expect_json=expect_json)

    @classmethod
    def capabilities_for_spec(cls, spec: BackendSpec) -> BackendCapabilities:
        """The capability advertisement implied by `spec`."""
        return _vllm_capabilities(spec, offline=False)

    def open_session(self) -> "VLLMServeSession":
        """Open a request session over the shared connection."""
        return VLLMServeSession(self)
