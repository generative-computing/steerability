"""The vLLM request sessions: the offline-engine session and the server session.

`_RequestSessionBase` holds the lifecycle and layout shared by both; `VLLMOfflineSession`
submits batched engine calls and serves plugin capture, and `VLLMServeSession` fans requests out
over the completions endpoint. This module imports cleanly without vLLM installed; the `vllm`
imports the sessions need stay method-local.
"""
import json
import uuid
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

import torch

from steerability.algorithms.core.execution.contracts import CaptureKinds, UnsupportedOperationError
from steerability.algorithms.core.execution.fanout import (
    PartialBatchError,
    derive_item_seed,
    run_bounded,
    with_transport_retries,
)
from steerability.algorithms.core.execution.params import GenerationParams
from steerability.algorithms.core.execution.payloads import (
    CaptureResult,
    ConstraintSource,
    GenerationItem,
    InterventionSpec,
    ItemResult,
    ModelFacts,
    PreparedPrompt,
    ScoringItem,
)
from steerability.algorithms.core.output import Output
from steerability.backends.vllm.capabilities import _refuse_by_constraints, _refuse_by_engine_facts
from steerability.backends.vllm.rendering import (
    _load_safetensors_bytes,
    _split_item_entries,
    extract_ref_logprobs,
    map_vllm_finish_reason,
    remap_spec_for_scoring,
    render_constraint_sampling_args,
    render_guided_decoding_field,
    render_vllm_sampling_args,
)

if TYPE_CHECKING:
    from steerability.backends.vllm.backend import VLLMBackend, VLLMServeBackend


class _RequestSessionBase:
    """Lifecycle and layout shared by the vLLM request sessions."""

    def __init__(self, backend) -> None:
        self._backend = backend
        self._closed = False
        self._generate_count = 0

    @property
    def closed(self) -> bool:
        """Whether the session has been closed."""
        return self._closed

    def close(self) -> None:
        """Close the session; further use raises `RuntimeError`."""
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("This session is closed; open a new session on the backend.")

    @property
    def tokenizer(self):
        """The backend's client-side tokenizer."""
        self._ensure_open()
        return self._backend.tokenizer

    @property
    def layout(self) -> ModelFacts:
        """Structural facts from the model config (client-side).

        Raises:
            RuntimeError: If the model config could not be resolved.
        """
        self._ensure_open()
        layout = self._backend._layout
        if layout is None:
            raise RuntimeError(
                "The model config could not be resolved client-side, so no layout is available."
            )
        return layout

    def _item_seed(self, item: GenerationItem, params: GenerationParams, index: int) -> int | None:
        if item.seed is not None:
            return item.seed
        if params.seed is not None:
            return derive_item_seed(params.seed, f"generate-{self._generate_count}", index)
        return None

    def _prepare_spec_submission(
        self,
        items: Sequence[GenerationItem | ScoringItem],
        backend_name: str,
        allow_constraints: bool = True,
    ) -> tuple[list[InterventionSpec | None], list[ConstraintSource | None], list[str] | None]:
        """Per-item intervention specs, constraint sources, and cache salts for a batch.

        Spec-bearing items salt with the reference derivation over the spec and its artifact
        ids; spec-free items through a plugin-active backend salt with the backend's constant
        salt (structural KV isolation; the worker cannot police requests that carry no
        new-surface keys). Engine-fact refusals and constraint checks run before any artifact
        is written; artifact payloads are then materialized into the registry root the engine
        reads.
        """
        backend = self._backend
        plugin_active = bool(backend.spec.get_option("hook_plugin"))
        specs, constraints = _split_item_entries(
            items, backend_name, plugin_active=plugin_active, allow_constraints=allow_constraints,
        )
        if any(spec is not None for spec in specs):
            discovery = getattr(backend, "_discovery", None)
            _refuse_by_engine_facts(discovery, "intervention")
            _refuse_by_constraints(specs, discovery, backend.intervention_kinds)
            for spec in specs:
                if spec is not None:
                    backend._artifact_uploader.upload(spec)
        salts: list[str] | None = None
        if plugin_active:
            salts = [
                spec.salt() if spec is not None else backend._plain_salt for spec in specs
            ]
        return specs, constraints, salts

    def _resolve_item_ids(self, item: GenerationItem | ScoringItem) -> list[int]:
        """The prompt's real token ids, with padding positions dropped per the attention mask,
        since a padded batch row would otherwise submit its pad tokens as prompt content."""
        resolved = item.prompt.resolve_token_ids(self.tokenizer)
        ids = resolved.token_ids[0]
        if resolved.attention_mask is not None:
            ids = ids[resolved.attention_mask[0].bool()]
        return ids.tolist()

    def _pack_output(
        self, index: int, prompt_ids: list[int], candidates: list[tuple[list[int], str | None]],
    ) -> ItemResult:
        """Build one `ItemResult` from per-candidate token ids and mapped finish reasons."""
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        max_len = max((len(ids) for ids, _ in candidates), default=0)
        rows = torch.full((len(candidates), max_len), pad_token_id, dtype=torch.long)
        reasons: list[str | None] = []
        for row, (ids, reason) in enumerate(candidates):
            if ids:
                rows[row, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            reasons.append(reason)
        return ItemResult(
            index=index,
            output=Output(
                output_ids=rows,
                adapted_input_ids=torch.tensor([prompt_ids], dtype=torch.long),
                finish_reason=reasons[0] if reasons else None,
                finish_reasons=tuple(reasons),
            ),
        )

    def capture(
        self,
        prompts: list[PreparedPrompt],
        layers: list[int],
        mode: Literal["all_tokens", "last_token"],
        location: Literal["layer_output", "layer_input"] = "layer_output",
    ) -> CaptureResult:
        """Hidden-state capture over the plugin is not implemented in this toolkit version."""
        raise UnsupportedOperationError(
            "Hidden-state capture on vLLM backends is not implemented in this toolkit version."
        )


class VLLMOfflineSession(_RequestSessionBase):
    """Request session over the offline engine.

    Token-id prompts submit as `TokensPrompt`s in one engine call with per-item sampling
    parameters; the engine schedules the batch internally, so no client-side fan-out is needed.
    """

    def capture(
        self,
        prompts: list[PreparedPrompt],
        layers: list[int],
        mode: Literal["all_tokens", "last_token"],
        location: Literal["layer_output", "layer_input"] = "layer_output",
    ) -> CaptureResult:
        """Hidden-state capture over the plugin's capture surface.

        One request per prompt carries a `capture` spec and a fresh random `cache_salt`
        (a prefix-cache hit skips forward passes, so capture cannot tolerate reused salts) with
        `max_tokens=1`; the surplus decode position is truncated by the plugin. Per-layer
        tensors are stacked and right-padded to the batch's longest prompt.

        Args:
            prompts: The prompts to capture over.
            layers: 0-based decoder-layer indices to capture.
            mode: `"all_tokens"` for every prompt position, `"last_token"` for the final real
                position per row.
            location: The residual-stream boundary, `"layer_output"` or `"layer_input"`.

        Returns:
            The capture result: `[N, T, H]` per layer for `"all_tokens"` or `[N, H]` for
            `"last_token"`, on CPU in the engine's native dtype, with the derived `[N, T]`
            attention mask.

        Raises:
            UnsupportedOperationError: If the spec declares no `hook_plugin`, the negotiated
                capture kinds lack the requested mode or location, or the engine facts refuse
                capture (speculative decoding, non-eager execution).
            ValueError: If `prompts` is empty, a layer id is out of range, or the engine
                returned no capture payload.
        """
        self._ensure_open()
        backend = self._backend
        if not backend.spec.get_option("hook_plugin"):
            raise UnsupportedOperationError(
                "Hidden-state capture requires the vLLM-Hook plugin; declare hook_plugin=True "
                "on the vllm backend spec, or run capture on the huggingface backend."
            )
        capture_kinds = backend.capture_kinds
        required = CaptureKinds(
            kinds=frozenset({"residual"}),
            locations=frozenset({location}),
            modes=frozenset({mode}),
        )
        if capture_kinds is None or not capture_kinds.contains(required):
            raise UnsupportedOperationError(
                f"The serving backend does not advertise capture mode {mode!r} at location "
                f"{location!r}; update the server's vllm_hook_plugins or run capture on the "
                "huggingface backend."
            )
        _refuse_by_engine_facts(backend._discovery, "capture")
        if not prompts:
            raise ValueError("capture() requires at least one prompt.")
        num_layers = self.layout.num_layers
        missing = sorted(int(layer) for layer in layers if not 0 <= int(layer) < num_layers)
        if missing:
            raise ValueError(
                f"Requested layer ids {missing} are out of range; the model has {num_layers} layers."
            )

        from vllm import SamplingParams, TokensPrompt

        layer_ids = [int(layer) for layer in layers]
        # the client's validator and assembly expect full prompt coverage and pool the last real
        # position themselves, so every wire capture requests all_tokens
        wire_mode = "all_tokens" if mode == "last_token" else mode
        capture_spec = {"layers": layer_ids, "mode": wire_mode, "location": location}
        engine_prompts = []
        prompt_lens: list[int] = []
        for prompt in prompts:
            resolved = prompt.resolve_token_ids(self.tokenizer)
            ids = resolved.token_ids[0]
            if resolved.attention_mask is not None:
                ids = ids[resolved.attention_mask[0].bool()]
            ids = ids.tolist()
            prompt_lens.append(len(ids))
            engine_prompt = TokensPrompt(prompt_token_ids=ids)
            engine_prompt["cache_salt"] = uuid.uuid4().hex
            engine_prompts.append(engine_prompt)
        sampling = SamplingParams(max_tokens=1, temperature=0.0, extra_args={"capture": capture_spec})

        request_outputs = self._backend._require_llm().generate(engine_prompts, sampling, use_tqdm=False)

        rows_per_layer: dict[int, list[torch.Tensor]] = {layer: [] for layer in layer_ids}
        for index, request_output in enumerate(request_outputs):
            payload = getattr(request_output, "captures", None)
            if payload is None:
                raise ValueError(
                    "The engine returned no capture payload; is the vLLM-Hook unified worker "
                    "active on this engine?"
                )
            manifest_json, data = payload
            manifest = json.loads(manifest_json)
            tensors = _load_safetensors_bytes(data)
            for layer in layer_ids:
                stacked = tensors.get(f"layer_{layer}")
                if stacked is None or stacked.size(0) < prompt_lens[index]:
                    raise ValueError(
                        f"The capture payload covers layer {layer} at "
                        f"{0 if stacked is None else stacked.size(0)} of {prompt_lens[index]} "
                        f"prompt positions for prompt {index}; positions recorded: "
                        f"{manifest.get('positions', {}).get(str(layer))}."
                    )
                rows_per_layer[layer].append(stacked[: prompt_lens[index]])

        max_len = max(prompt_lens)
        attention_mask = torch.zeros(len(prompts), max_len, dtype=torch.long)
        for index, length in enumerate(prompt_lens):
            attention_mask[index, :length] = 1

        hidden: dict[int, torch.Tensor] = {}
        for layer, rows in rows_per_layer.items():
            if mode == "last_token":
                hidden[layer] = torch.stack([row[-1] for row in rows])
            else:
                padded = torch.zeros(len(rows), max_len, rows[0].size(-1), dtype=rows[0].dtype)
                for index, row in enumerate(rows):
                    padded[index, : row.size(0)] = row
                hidden[layer] = padded
        return CaptureResult(hidden=hidden, attention_mask=attention_mask, mode=mode, location=location)

    def generate(
        self,
        items: Sequence[GenerationItem],
        params: GenerationParams,
    ) -> list[ItemResult]:
        """Generate one result per item through the engine.

        Args:
            items: The generation items; state entries lower as intervention specs on
                plugin-active backends, and no client-side hooks or live processors execute
                here.
            params: Normalized generation parameters shared by all items; unmapped `extra` keys
                raise.

        Returns:
            One `ItemResult` per item, in item order.
        """
        self._ensure_open()
        if not items:
            return []
        item_specs, item_constraints, item_salts = self._prepare_spec_submission(items, "vllm")
        base_args = render_vllm_sampling_args(params)

        from vllm import SamplingParams, TokensPrompt

        prompts = []
        sampling = []
        prompt_ids_per_item: list[list[int]] = []
        for index, item in enumerate(items):
            ids = self._resolve_item_ids(item)
            prompt_ids_per_item.append(ids)
            args = dict(base_args)
            seed = self._item_seed(item, params, index)
            if seed is not None:
                args["seed"] = seed
            if item_constraints[index] is not None:
                field, value = render_guided_decoding_field(item_constraints[index])
                args.update(render_constraint_sampling_args(field, value))
            if item_specs[index] is not None:
                args["extra_args"] = {"intervention_spec": item_specs[index].to_wire()}
            prompt = TokensPrompt(prompt_token_ids=ids)
            if item_salts is not None:
                prompt["cache_salt"] = item_salts[index]
            prompts.append(prompt)
            sampling.append(SamplingParams(**args))
        self._generate_count += 1

        generate_kwargs: dict[str, Any] = {"use_tqdm": False}
        if self._backend._lora_request is not None:
            generate_kwargs["lora_request"] = self._backend._lora_request
        request_outputs = self._backend._require_llm().generate(prompts, sampling, **generate_kwargs)

        results: list[ItemResult] = []
        for index, request_output in enumerate(request_outputs):
            candidates = [
                (
                    list(candidate.token_ids),
                    map_vllm_finish_reason(
                        candidate.finish_reason, getattr(candidate, "stop_reason", None),
                    ),
                )
                for candidate in request_output.outputs
            ]
            results.append(self._pack_output(index, prompt_ids_per_item[index], candidates))
        return results

    def score(
        self,
        items: Sequence[ScoringItem],
        params: GenerationParams,
    ) -> torch.Tensor:
        """Teacher-forced log-probabilities of each item's reference tokens via prompt logprobs.

        Each item's prompt and reference concatenate into one token-id prompt submitted with
        `prompt_logprobs=0`, and the reference positions' log-probabilities are read back.

        Args:
            items: The scoring items. Every item must carry the same reference length.
            params: Must carry no `extra` keys; forward keyword arguments have no remote
                rendering.

        Returns:
            Log probabilities of shape `[num_items, ref_len]` on CPU.

        Raises:
            ValueError: If items carry differing reference lengths or `params.extra` is
                non-empty.
        """
        self._ensure_open()
        if params.extra:
            raise ValueError(
                f"Scoring parameter(s) {sorted(params.extra)} have no vLLM rendering; remote "
                "scoring accepts no forward keyword arguments."
            )
        if not items:
            return torch.zeros((0, 0), dtype=torch.float32)
        item_specs, _, item_salts = self._prepare_spec_submission(
            items, "vllm", allow_constraints=False,
        )
        ref_lens = {item.ref_output_ids.shape[-1] for item in items}
        if len(ref_lens) > 1:
            raise ValueError(f"All scoring items must share one reference length; got {sorted(ref_lens)}.")
        ref_len = ref_lens.pop()
        if ref_len == 0:
            return torch.zeros((len(items), 0), dtype=torch.float32)

        from vllm import SamplingParams, TokensPrompt

        prompts = []
        sampling = []
        ref_ids_per_item: list[list[int]] = []
        for index, item in enumerate(items):
            prompt_ids = self._resolve_item_ids(item)
            ref_ids = item.ref_output_ids.reshape(-1).tolist()
            ref_ids_per_item.append(ref_ids)
            prompt = TokensPrompt(prompt_token_ids=[*prompt_ids, *ref_ids])
            args: dict[str, Any] = {"max_tokens": 1, "temperature": 0.0, "prompt_logprobs": 0}
            if item_specs[index] is not None:
                scoring_spec = remap_spec_for_scoring(item_specs[index], len(prompt_ids))
                args["extra_args"] = {"intervention_spec": scoring_spec.to_wire()}
                if item_salts is not None:
                    item_salts[index] = scoring_spec.salt()
            if item_salts is not None:
                prompt["cache_salt"] = item_salts[index]
            prompts.append(prompt)
            sampling.append(SamplingParams(**args))

        generate_kwargs: dict[str, Any] = {"use_tqdm": False}
        if self._backend._lora_request is not None:
            generate_kwargs["lora_request"] = self._backend._lora_request
        request_outputs = self._backend._require_llm().generate(prompts, sampling, **generate_kwargs)
        rows = [
            extract_ref_logprobs(request_output.prompt_logprobs, ref_ids)
            for request_output, ref_ids in zip(request_outputs, ref_ids_per_item)
        ]
        return torch.tensor(rows, dtype=torch.float32)


class VLLMServeSession(_RequestSessionBase):
    """Request session over a vLLM server's completions endpoint.

    Items fan out concurrently under the backend's `max_concurrency`; transport failures retry
    with exponential backoff, and a batch whose items partially fail raises `PartialBatchError`
    carrying the successes and the re-issuable failures.
    """

    def generate(
        self,
        items: Sequence[GenerationItem],
        params: GenerationParams,
    ) -> list[ItemResult]:
        """Generate one result per item through the completions endpoint.

        Args:
            items: The generation items; entries must be empty.
            params: Normalized generation parameters shared by all items; unmapped `extra` keys
                raise.

        Returns:
            One `ItemResult` per item, in item order.

        Raises:
            PartialBatchError: If some items failed after transport retries while others
                succeeded.
        """
        self._ensure_open()
        if not items:
            return []
        item_specs, item_constraints, item_salts = self._prepare_spec_submission(items, "vllm-serve")
        base_args = render_vllm_sampling_args(params)
        backend = self._backend

        item_ids = [self._resolve_item_ids(item) for item in items]
        seeds = [self._item_seed(item, params, index) for index, item in enumerate(items)]
        self._generate_count += 1

        def make_task(index: int):
            def task() -> ItemResult:
                body: dict[str, Any] = {
                    "model": backend._served_model,
                    "prompt": item_ids[index],
                    "return_token_ids": True,
                    **base_args,
                }
                if seeds[index] is not None:
                    body["seed"] = seeds[index]
                if item_constraints[index] is not None:
                    field, value = render_guided_decoding_field(item_constraints[index])
                    body[f"guided_{field}"] = value
                if item_specs[index] is not None:
                    # vllm_xargs is scalar-only, so nested specs travel as JSON strings
                    body["vllm_xargs"] = {
                        "intervention_spec": item_specs[index].canonical(),
                    }
                if item_salts is not None:
                    body["cache_salt"] = item_salts[index]
                payload = with_transport_retries(
                    lambda: backend._post_json("/v1/completions", body),
                    max_attempts=backend.max_attempts,
                    backoff_base=backend.backoff_base,
                )
                choices = payload.get("choices", [])
                if not choices:
                    raise ValueError("The completions response carries no choices.")
                candidates = []
                for choice in choices:
                    token_ids = choice.get("token_ids")
                    if token_ids is None:
                        raise ValueError(
                            "The completions response carries no token_ids; the server does "
                            "not support the token-id return option (return_token_ids)."
                        )
                    candidates.append((
                        list(token_ids),
                        map_vllm_finish_reason(choice.get("finish_reason"), choice.get("stop_reason")),
                    ))
                return self._pack_output(index, item_ids[index], candidates)
            return task

        outcomes = run_bounded([make_task(i) for i in range(len(items))], backend.max_concurrency)
        failures = [(i, outcome) for i, outcome in enumerate(outcomes) if isinstance(outcome, Exception)]
        results = [outcome for outcome in outcomes if not isinstance(outcome, Exception)]
        if failures:
            raise PartialBatchError(results, failures)
        return results

    def score(
        self,
        items: Sequence[ScoringItem],
        params: GenerationParams,
    ) -> torch.Tensor:
        """Teacher-forced log-probabilities of each item's reference tokens via prompt logprobs.

        Args:
            items: The scoring items. Every item must carry the same reference length.
            params: Must carry no `extra` keys.

        Returns:
            Log probabilities of shape `[num_items, ref_len]` on CPU.

        Raises:
            ValueError: If items carry differing reference lengths or `params.extra` is
                non-empty.
            PartialBatchError: If some items failed after transport retries while others
                succeeded.
        """
        self._ensure_open()
        if params.extra:
            raise ValueError(
                f"Scoring parameter(s) {sorted(params.extra)} have no vLLM rendering; remote "
                "scoring accepts no forward keyword arguments."
            )
        if not items:
            return torch.zeros((0, 0), dtype=torch.float32)
        item_specs, _, item_salts = self._prepare_spec_submission(
            items, "vllm-serve", allow_constraints=False,
        )
        ref_lens = {item.ref_output_ids.shape[-1] for item in items}
        if len(ref_lens) > 1:
            raise ValueError(f"All scoring items must share one reference length; got {sorted(ref_lens)}.")
        ref_len = ref_lens.pop()
        if ref_len == 0:
            return torch.zeros((len(items), 0), dtype=torch.float32)
        backend = self._backend

        prompt_ids = [self._resolve_item_ids(item) for item in items]
        ref_ids = [item.ref_output_ids.reshape(-1).tolist() for item in items]

        def make_task(index: int):
            def task() -> list[float]:
                body = {
                    "model": backend._served_model,
                    "prompt": [*prompt_ids[index], *ref_ids[index]],
                    "max_tokens": 1,
                    "temperature": 0.0,
                    "prompt_logprobs": 0,
                }
                if item_specs[index] is not None:
                    scoring_spec = remap_spec_for_scoring(item_specs[index], len(prompt_ids[index]))
                    body["vllm_xargs"] = {"intervention_spec": scoring_spec.canonical()}
                    body["cache_salt"] = scoring_spec.salt()
                elif item_salts is not None:
                    body["cache_salt"] = item_salts[index]
                payload = with_transport_retries(
                    lambda: backend._post_json("/v1/completions", body),
                    max_attempts=backend.max_attempts,
                    backoff_base=backend.backoff_base,
                )
                choices = payload.get("choices", [])
                if not choices:
                    raise ValueError("The completions response carries no choices.")
                prompt_logprobs = choices[0].get("prompt_logprobs")
                return extract_ref_logprobs(prompt_logprobs, ref_ids[index])
            return task

        outcomes = run_bounded([make_task(i) for i in range(len(items))], backend.max_concurrency)
        failures = [(i, outcome) for i, outcome in enumerate(outcomes) if isinstance(outcome, Exception)]
        if failures:
            successes = [outcome for outcome in outcomes if not isinstance(outcome, Exception)]
            raise PartialBatchError(successes, failures)
        return torch.tensor(outcomes, dtype=torch.float32)
