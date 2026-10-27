"""The vLLM request sessions: the offline-engine session and the server session.

`_RequestSessionBase` holds the lifecycle and layout shared by both. `VLLMOfflineSession` submits
one engine request per item through the backend's engine thread and serves plugin capture, and
`VLLMServeSession` fans requests out over the completions endpoint. This module imports cleanly
without vLLM installed; the `vllm` imports the sessions need stay method-local.
"""
import concurrent.futures
import json
import uuid
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

import torch

from steerability.algorithms.core.execution.cancellation import CancelToken, GenerationCancelled
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
from steerability.backends.vllm.engine_thread import EngineClosedError
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
    from steerability.backends.vllm.backend import RequestHandle, VLLMBackend, VLLMServeBackend


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

    @staticmethod
    def _await_handles(handles: Sequence["RequestHandle"], cancel_token: CancelToken | None) -> list:
        """Wait for every submitted engine request and return their outputs in submission order.

        With a token, cancelling it cancels the requests not yet done, which aborts them on the
        engine. A request that fails or is interrupted cancels the remaining requests before the
        error propagates.

        Args:
            handles: The submitted requests, in item order.
            cancel_token: The call's cancellation token, or None.

        Returns:
            One finished `RequestOutput` per handle.

        Raises:
            GenerationCancelled: If the token was cancelled before every request finished.
            EngineClosedError: If the backend was released while a request was in flight.
        """
        # the token's callback references this list, which is emptied once the call ends so that a
        # long-lived token does not keep the finished requests alive
        pending = list(handles)
        if cancel_token is not None:
            cancel_token.on_cancel(lambda: [handle.cancel() for handle in pending if not handle.done()])
        outputs = []
        try:
            for handle in handles:
                outputs.append(handle.result())
        except concurrent.futures.CancelledError as error:
            for handle in handles:
                handle.cancel()
            if cancel_token is not None and cancel_token.cancelled:
                raise GenerationCancelled("The generation was cancelled through its CancelToken.") from error
            raise EngineClosedError("The backend was released while a request was in flight.") from error
        except BaseException:
            for handle in handles:
                handle.cancel()
            raise
        finally:
            pending.clear()
        return outputs

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
                finish_reasons=tuple(reasons),
            ),
        )

    def capture(
        self,
        prompts: list[PreparedPrompt],
        layers: list[int],
        mode: Literal["all_tokens", "last_token"],
        location: Literal["layer_output", "layer_input"] = "layer_output",
        state_entries: Sequence = (),
    ) -> CaptureResult:
        """Hidden-state capture over the plugin is not implemented in this toolkit version."""
        raise UnsupportedOperationError(
            "Hidden-state capture on vLLM backends is not implemented in this toolkit version."
        )


class VLLMOfflineSession(_RequestSessionBase):
    """Request session over the offline engine.

    Each item submits as one `TokensPrompt` engine request with its own sampling parameters,
    through the backend's engine thread. The engine's scheduler admits each request as it arrives,
    so the items of one call and the items of concurrent calls on any thread share one running
    batch, and each request finishes independently.
    """

    def _submit_all(self, requests, lora_request=None) -> list["RequestHandle"]:
        """Submit `(prompt, sampling_params)` pairs in order, cancelling the submitted requests when
        a later submission fails."""
        handles: list = []
        try:
            for prompt, sampling_params in requests:
                handles.append(self._backend._submit_request(prompt, sampling_params, lora_request=lora_request))
        except BaseException:
            for handle in handles:
                handle.cancel()
            raise
        return handles

    def capture(
        self,
        prompts: list[PreparedPrompt],
        layers: list[int],
        mode: Literal["all_tokens", "last_token"],
        location: Literal["layer_output", "layer_input"] = "layer_output",
        state_entries: Sequence = (),
    ) -> CaptureResult:
        """Hidden-state capture over the plugin's capture surface.

        One request per prompt carries a `capture` spec and a fresh random `cache_salt`
        (a prefix-cache hit skips forward passes, so capture cannot tolerate reused salts) with
        `max_tokens=1`; the surplus decode position is truncated by the plugin. Per-layer
        tensors are stacked and right-padded to the batch's longest prompt. With
        `state_entries`, every request also carries their merged intervention spec, so the
        captured forward passes are steered as a generation request with the same entries is.

        Args:
            prompts: The prompts to capture over.
            layers: 0-based decoder-layer indices to capture.
            mode: `"all_tokens"` for every prompt position, `"last_token"` for the final real
                position per row.
            location: The residual-stream boundary, `"layer_output"` or `"layer_input"`.
            state_entries: `InterventionEntry` contributions applied to the captured forward
                passes, e.g., the steering of the generation that a decoding driver runs. Empty
                (default) captures the unsteered model.

        Returns:
            The capture result: `[N, T, H]` per layer for `"all_tokens"` or `[N, H]` for
            `"last_token"`, on CPU in the engine's native dtype, with the derived `[N, T]`
            attention mask.

        Raises:
            UnsupportedOperationError: If the spec declares no `hook_plugin`, the negotiated
                capture kinds lack the location or the wire mode (`"all_tokens"` for both
                requested modes), or the engine facts refuse capture (speculative decoding,
                non-eager execution).
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
        # every wire capture requests all_tokens, since the client's validator and assembly expect
        # full prompt coverage and pool the last real position themselves
        wire_mode = "all_tokens" if mode == "last_token" else mode
        capture_kinds = backend.capture_kinds
        required = CaptureKinds(
            kinds=frozenset({"residual"}),
            locations=frozenset({location}),
            modes=frozenset({wire_mode}),
        )
        if capture_kinds is None or not capture_kinds.contains(required):
            raise UnsupportedOperationError(
                f"The serving backend does not advertise capture mode {wire_mode!r} at location "
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
        capture_spec = {"layers": layer_ids, "mode": wire_mode, "location": location}
        extra_args: dict[str, Any] = {"capture": capture_spec}
        if state_entries:
            # every prompt carries the same entries, so one item gives the merged spec
            (intervention,), _, _ = self._prepare_spec_submission(
                [GenerationItem(prompt=prompts[0], state_entries=tuple(state_entries))], "vllm",
            )
            if intervention is not None:
                extra_args["intervention_spec"] = intervention.to_wire()
        requests = []
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
            requests.append((engine_prompt, SamplingParams(max_tokens=1, temperature=0.0, extra_args=dict(extra_args))))

        request_outputs = self._await_handles(self._submit_all(requests), None)

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
        *,
        cancel_token: CancelToken | None = None,
    ) -> list[ItemResult]:
        """Generate one result per item through the engine.

        Each item is submitted as one engine request, and the engine admits each request as it is
        submitted. The call returns once every item's request has finished. Cancelling
        `cancel_token` aborts the call's unfinished requests on the engine, which frees their
        slots in the running batch.

        Args:
            items: The generation items; state entries lower as intervention specs on
                plugin-active backends, and no client-side hooks or in-memory processors execute
                here.
            params: Normalized generation parameters shared by all items; unmapped `extra` keys
                raise `ValueError`.
            cancel_token: Token whose cancellation aborts the call's requests, or None.

        Returns:
            One `ItemResult` per item, in item order.

        Raises:
            GenerationCancelled: If `cancel_token` is cancelled before every request finished.
            EngineClosedError: If the backend is released while a request is in flight.
        """
        self._ensure_open()
        if cancel_token is not None:
            cancel_token.raise_if_cancelled()
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

        handles = self._submit_all(zip(prompts, sampling), lora_request=self._backend._lora_request)
        request_outputs = self._await_handles(handles, cancel_token)

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
        *,
        cancel_token: CancelToken | None = None,
    ) -> torch.Tensor:
        """Teacher-forced log-probabilities of each item's reference tokens via prompt logprobs.

        Each item's prompt and reference concatenate into one token-id prompt submitted as one
        engine request with `prompt_logprobs=0`, and the reference positions' log-probabilities
        are read back. Cancelling `cancel_token` aborts the call's unfinished requests.

        Args:
            items: The scoring items. Every item must carry the same reference length.
            params: Must carry no `extra` keys; forward keyword arguments have no remote
                rendering.
            cancel_token: Token whose cancellation aborts the call's requests, or None.

        Returns:
            Log probabilities of shape `[num_items, ref_len]` on CPU.

        Raises:
            ValueError: If items carry differing reference lengths or `params.extra` is
                non-empty.
            GenerationCancelled: If `cancel_token` is cancelled before every request finished.
            EngineClosedError: If the backend is released while a request is in flight.
        """
        self._ensure_open()
        if cancel_token is not None:
            cancel_token.raise_if_cancelled()
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

        handles = self._submit_all(zip(prompts, sampling), lora_request=self._backend._lora_request)
        request_outputs = self._await_handles(handles, cancel_token)
        rows = [
            extract_ref_logprobs(request_output.prompt_logprobs, ref_ids)
            for request_output, ref_ids in zip(request_outputs, ref_ids_per_item)
        ]
        return torch.tensor(rows, dtype=torch.float32)


class VLLMServeSession(_RequestSessionBase):
    """Request session over a vLLM server's completions endpoint.

    The items of one call fan out concurrently under the backend's `max_concurrency`, and a
    single-item call issues its one HTTP request on the caller's thread. The server batches
    requests from concurrent calls, and the caller bounds how many calls run at once. Transport
    failures retry with exponential backoff (`max_retries` applies to transport errors only), and
    a batch whose items partially fail raises `PartialBatchError` containing the successes and the
    re-issuable failures. The backend's `request_timeout` bounds each HTTP request and must
    cover one full generation.
    """

    def generate(
        self,
        items: Sequence[GenerationItem],
        params: GenerationParams,
        *,
        cancel_token: CancelToken | None = None,
    ) -> list[ItemResult]:
        """Generate one result per item through the completions endpoint.

        A cancelled `cancel_token` stops the call before each item's request starts; a request
        already sent completes.

        Args:
            items: The generation items; entries must be empty.
            params: Normalized generation parameters shared by all items; unmapped `extra` keys
                raise `ValueError`.
            cancel_token: Token checked before each item's request starts, or None.

        Returns:
            One `ItemResult` per item, in item order.

        Raises:
            GenerationCancelled: If `cancel_token` is cancelled before an item's request starts.
            PartialBatchError: If some items failed after transport retries while others
                succeeded.
        """
        self._ensure_open()
        if cancel_token is not None:
            cancel_token.raise_if_cancelled()
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
                if cancel_token is not None:
                    cancel_token.raise_if_cancelled()
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
        _raise_if_cancelled(outcomes, cancel_token)
        failures = [(i, outcome) for i, outcome in enumerate(outcomes) if isinstance(outcome, Exception)]
        results = [outcome for outcome in outcomes if not isinstance(outcome, Exception)]
        if failures:
            raise PartialBatchError(results, failures)
        return results

    def score(
        self,
        items: Sequence[ScoringItem],
        params: GenerationParams,
        *,
        cancel_token: CancelToken | None = None,
    ) -> torch.Tensor:
        """Teacher-forced log-probabilities of each item's reference tokens via prompt logprobs.

        A cancelled `cancel_token` stops the call before each item's request starts; a request
        already sent completes.

        Args:
            items: The scoring items. Every item must carry the same reference length.
            params: Must carry no `extra` keys.
            cancel_token: Token checked before each item's request starts, or None.

        Returns:
            Log probabilities of shape `[num_items, ref_len]` on CPU.

        Raises:
            ValueError: If items carry differing reference lengths or `params.extra` is
                non-empty.
            GenerationCancelled: If `cancel_token` is cancelled before an item's request starts.
            PartialBatchError: If some items failed after transport retries while others
                succeeded.
        """
        self._ensure_open()
        if cancel_token is not None:
            cancel_token.raise_if_cancelled()
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
                if cancel_token is not None:
                    cancel_token.raise_if_cancelled()
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
        _raise_if_cancelled(outcomes, cancel_token)
        failures = [(i, outcome) for i, outcome in enumerate(outcomes) if isinstance(outcome, Exception)]
        if failures:
            successes = [outcome for outcome in outcomes if not isinstance(outcome, Exception)]
            raise PartialBatchError(successes, failures)
        return torch.tensor(outcomes, dtype=torch.float32)


def _raise_if_cancelled(outcomes: list, cancel_token: CancelToken | None) -> None:
    """Raise `GenerationCancelled` when a fan-out task stopped on the call's cancelled token.

    A cancellation then surfaces as itself rather than inside a `PartialBatchError`.
    """
    if cancel_token is None or not cancel_token.cancelled:
        return
    for outcome in outcomes:
        if isinstance(outcome, GenerationCancelled):
            raise outcome
