"""The in-process exclusive session: direct model access, hook scopes, and the default decode loop."""
import contextlib
import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import torch
from transformers import LogitsProcessorList, PreTrainedModel, PreTrainedTokenizerBase, StoppingCriteriaList

from steerability.algorithms.core.execution.contracts import UnsupportedOperationError
from steerability.algorithms.core.execution.fanout import derive_item_seed
from steerability.algorithms.core.execution.params import GenerationParams
from steerability.algorithms.core.execution.payloads import (
    CaptureResult,
    GenerationItem,
    HookEntry,
    ItemResult,
    ModelFacts,
    PreparedPrompt,
    ScoringItem,
    StackEntry,
)
from steerability.algorithms.core.internals.model_layout import text_config
from steerability.algorithms.core.output import Output, infer_finish_reasons
from steerability.algorithms.output_control.base import stack_generate_kwargs
from steerability.algorithms.output_control.common.criteria import StopOnSubstring, StopOnTokens
from steerability.algorithms.state_control.common.hook_utils import get_model_layer_list
from steerability.utils.tokenization import infer_attention_mask_from_ids, to_left_pad

if TYPE_CHECKING:
    from steerability.backends.huggingface.backend import HFBackend

logger = logging.getLogger(__name__)

_CAPTURE_BATCH_SIZE = 8


def render_hf_gen_kwargs(params: GenerationParams) -> dict:
    """Render normalized generation parameters onto `model.generate` keyword arguments.

    Keys in `params.extra` pass through untouched; a normalized field always takes precedence
    over a same-named extra key. `seed` is not rendered here, since the session applies it as a
    `fork_rng`-scoped `manual_seed` around the item's decode; the stop fields are not rendered
    either, since the session composes them as prompt-anchored stop criteria
    (`compose_stop_criteria`).

    Args:
        params: The normalized parameters.

    Returns:
        Keyword arguments for `model.generate`.
    """
    gen_kwargs = dict(params.extra)
    if params.max_new_tokens is not None:
        gen_kwargs["max_new_tokens"] = params.max_new_tokens
    if params.min_new_tokens is not None:
        gen_kwargs["min_new_tokens"] = params.min_new_tokens
    if params.temperature is not None:
        gen_kwargs["temperature"] = params.temperature
    if params.top_p is not None:
        gen_kwargs["top_p"] = params.top_p
    if params.top_k is not None:
        gen_kwargs["top_k"] = params.top_k
    if params.repetition_penalty is not None:
        gen_kwargs["repetition_penalty"] = params.repetition_penalty
    if params.greedy is not None:
        gen_kwargs["do_sample"] = not params.greedy
    if params.n is not None:
        gen_kwargs["num_return_sequences"] = params.n
    return gen_kwargs


def compose_stop_criteria(
    params: GenerationParams, prompt_len: int, tokenizer: PreTrainedTokenizerBase | None
) -> list:
    """The stop criteria implied by the normalized stop fields, anchored at `prompt_len`.

    Args:
        params: The normalized parameters carrying `stop_strings` and `stop_token_ids`.
        prompt_len: Prompt length the substring criteria decode past.
        tokenizer: Tokenizer for substring decoding; required when stop strings are set.

    Returns:
        The composed criteria (possibly empty).

    Raises:
        ValueError: If stop strings are set and no tokenizer is available.
    """
    criteria: list = []
    if params.stop_strings:
        if tokenizer is None:
            raise ValueError("stop_strings require a tokenizer on the session.")
        for text in params.stop_strings:
            criteria.append(StopOnSubstring(tokenizer, text, prompt_len))
    if params.stop_token_ids:
        criteria.append(StopOnTokens(params.stop_token_ids))
    return criteria


class ExclusiveSession:
    """The in-process session: direct model access, hook scopes, and the default decode loop.

    Exposes `.model` for components whose requirements include `Capability.IN_PROCESS_TORCH`.
    The default decode delegates to `model.generate`. Items execute serially, each under its own
    hook registrations, which preserves in-process semantics for every entry combination.
    """

    def __init__(self, backend: "HFBackend") -> None:
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

    def __enter__(self) -> "ExclusiveSession":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("This session is closed; open a new session on the backend.")

    @property
    def model(self) -> PreTrainedModel:
        """The live model.

        Raises:
            RuntimeError: If the session is closed or no model is available yet.
        """
        self._ensure_open()
        model = self._backend._model_provider()
        if model is None:
            raise RuntimeError("No model is available on this session.")
        return model

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase | None:
        """The tokenizer, or None when the adopting caller has not resolved one yet."""
        self._ensure_open()
        return self._backend._tokenizer_provider()

    @property
    def layout(self) -> ModelFacts:
        """Structural facts derived from the loaded model, computed on every access so weight
        edits and model replacements are always reflected.

        `num_layers` comes from the resolved decoder layer list; `hidden_size`,
        `num_attention_heads`, and `head_dim` come from the text config (`text_config(model)`,
        the text sub-config on composite multimodal models), with a
        `hidden_size // num_attention_heads` fallback for `head_dim`; `dtype` from the model;
        `model_fingerprint` from the weight/config fingerprint; and `model_type` from the
        composite config, so a multimodal checkpoint keeps its wrapper `model_type`.
        """
        model = self.model

        from steerability.algorithms.core.internals.fingerprint import model_fingerprint

        _, layer_names = get_model_layer_list(model)
        text_cfg = text_config(model)
        hidden_size = text_cfg.hidden_size
        num_heads = getattr(text_cfg, "num_attention_heads", None)
        head_dim = getattr(text_cfg, "head_dim", None)
        if head_dim is None and num_heads:
            head_dim = hidden_size // num_heads
        return ModelFacts(
            num_layers=len(layer_names),
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            head_dim=head_dim,
            dtype=str(model.dtype).removeprefix("torch."),
            model_fingerprint=model_fingerprint(model),
            model_type=getattr(model.config, "model_type", None),
            model_ref=getattr(model, "name_or_path", None),
        )

    def _resolve_prompt_tensors(self, prompt: PreparedPrompt) -> tuple[torch.Tensor, torch.Tensor]:
        """Token ids and attention mask for one prompt, on the model device."""
        resolved = prompt.resolve_token_ids(self.tokenizer)
        device = self.model.device
        input_ids = resolved.token_ids.to(device)
        attention_mask = resolved.attention_mask
        if attention_mask is None:
            tokenizer = self.tokenizer
            if tokenizer is not None and tokenizer.pad_token_id is not None:
                attention_mask = infer_attention_mask_from_ids(input_ids, tokenizer.pad_token_id)
            else:
                attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        attention_mask = attention_mask.to(dtype=input_ids.dtype, device=device)
        return input_ids, attention_mask

    def _compose_entry_stacks(
        self, output_entries, extra_processors=(), extra_criteria=(),
    ) -> tuple[LogitsProcessorList, StoppingCriteriaList]:
        """Compose the items' stack entries, appending caller extras after entry contributions."""
        processors: list = []
        criteria: list = []
        for entry in output_entries:
            if not isinstance(entry, StackEntry):
                raise UnsupportedOperationError(
                    f"{type(entry).__name__} requires an engine-hosted processor path; the "
                    "in-process session consumes StackEntry contributions."
                )
            processors.extend(entry.logits_processors)
            criteria.extend(entry.stopping_criteria)
        processors.extend(extra_processors)
        criteria.extend(extra_criteria)
        return LogitsProcessorList(processors), StoppingCriteriaList(criteria)

    def _register_state_entries(self, model: PreTrainedModel, state_entries) -> list:
        """Attach each entry's hook specifications to `model`, returning removable handles.

        The session is the single registrar of state hooks: entries are the only carriage, and
        registration lives strictly inside the session's execution of work. Pre and forward
        hooks register with `with_kwargs=True`; backward hooks register as full backward hooks.
        If registration fails partway, handles already attached are removed before re-raising.
        """
        handles: list = []
        try:
            for entry in state_entries:
                if not isinstance(entry, HookEntry):
                    raise UnsupportedOperationError(
                        f"{type(entry).__name__} requires an intervention-capable backend; the "
                        "in-process session consumes HookEntry contributions."
                    )
                for phase in ("pre", "forward", "backward"):
                    for spec in entry.hooks.get(phase, []):
                        module = model.get_submodule(spec["module"])
                        if phase == "pre":
                            handle = module.register_forward_pre_hook(spec["hook_func"], with_kwargs=True)
                        elif phase == "forward":
                            handle = module.register_forward_hook(spec["hook_func"], with_kwargs=True)
                        else:
                            handle = module.register_full_backward_hook(spec["hook_func"])
                        handles.append(handle)
        except Exception:
            for handle in handles:
                handle.remove()
            raise
        return handles

    @contextlib.contextmanager
    def entries_applied(self, state_entries):
        """Apply state entries to the live model for the duration of the context.

        Used by the pipeline around a client-side decoding driver's `decode`, so every forward
        the driver issues on the live model, including rollouts through this session and
        auxiliary scoring passes, runs under the generation's hooks. The session owns
        registration; hooks are removed when the context exits, even on error.
        """
        handles = self._register_state_entries(self.model, state_entries)
        try:
            yield self
        finally:
            for handle in handles:
                handle.remove()

    def _seeded(self, seed: int | None):
        """A context that snapshots and restores RNG state around a seeded decode.

        The CPU generator is always covered. On CUDA models every CUDA device generator is
        covered, since sharded models may sample on a device other than the first parameter's.
        On MPS models the MPS generator is covered.
        """
        if seed is None:
            return contextlib.nullcontext()
        device = self.model.device
        if device.type == "cuda":
            return torch.random.fork_rng(devices=list(range(torch.cuda.device_count())))
        if device.type == "mps":
            return _mps_rng_fork()
        return torch.random.fork_rng(devices=[])

    def _apply_seed(self, seed: int) -> None:
        """Seed the generators covered by `_seeded` for this decode."""
        torch.default_generator.manual_seed(seed)
        device = self.model.device
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        elif device.type == "mps":
            torch.mps.manual_seed(seed)

    def _item_seeds(self, items: Sequence[GenerationItem], params: GenerationParams) -> list[int | None]:
        """Effective per-item seeds.

        An item's own seed is always honored. Otherwise, under `params.seed` with
        `seed_scope="item"`, each item derives its own seed from this call's operation id and its
        index; with `seed_scope="dispatch"`, every item takes the dispatch seed, derived at index 0
        so a single-item dispatch decodes identically under either scope. A dispatch mixing explicit
        and absent item seeds falls back to per-item derivation. Unseeded calls yield None.
        """
        operation_id = f"generate-{self._generate_count}"
        if (
            params.seed is not None
            and params.seed_scope == "dispatch"
            and all(item.seed is None for item in items)
        ):
            return [derive_item_seed(params.seed, operation_id, 0)] * len(items)
        seeds: list[int | None] = []
        for index, item in enumerate(items):
            if item.seed is not None:
                seeds.append(item.seed)
            elif params.seed is not None:
                seeds.append(derive_item_seed(params.seed, operation_id, index))
            else:
                seeds.append(None)
        return seeds

    def _report_serial_fallback(self, items: Sequence[GenerationItem]) -> None:
        """Log once per backend and reason why a multi-item dispatch decodes one item at a time.

        Reached only when `batchable` is False, so identical entries imply distinct seeds.
        """
        if not self._entries_identical(items):
            key, detail = "entries", (
                "items carry distinct state or output entries (row-scoped runtime kwargs or per-row hooks)"
            )
        else:
            key, detail = "seeds", (
                "items carry distinct seeds (seed_scope='item'); pass seed_scope='dispatch' to generate(), or "
                "set ProviderOptions.seed_scope on the evaluation provider, to decode the dispatch in one pass"
            )
        if self._backend.report_once(f"serial_fallback:{key}"):
            logger.info("Multi-item generate (%d items) decodes serially: %s.", len(items), detail)

    @staticmethod
    def _entries_identical(items: Sequence[GenerationItem | ScoringItem]) -> bool:
        """True when every item carries the same state and output entry objects."""
        first = items[0]
        for item in items[1:]:
            if len(item.state_entries) != len(first.state_entries) or any(
                a is not b for a, b in zip(item.state_entries, first.state_entries)
            ):
                return False
            if len(item.output_entries) != len(first.output_entries) or any(
                a is not b for a, b in zip(item.output_entries, first.output_entries)
            ):
                return False
        return True

    def _stack_prompt_rows(self, rows: list[tuple[torch.Tensor, torch.Tensor]]):
        """Stack resolved single-row prompts into one right-padded batch.

        The batched generate and score paths both left-pack the result via `to_left_pad` before
        the forward pass.
        """
        pad_token_id = getattr(self.tokenizer, "pad_token_id", None) or 0
        max_len = max(ids.size(1) for ids, _ in rows)
        device = rows[0][0].device
        input_ids = torch.full((len(rows), max_len), pad_token_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros((len(rows), max_len), dtype=rows[0][1].dtype, device=device)
        for row, (ids, mask) in enumerate(rows):
            length = ids.size(1)
            input_ids[row, :length] = ids[0]
            attention_mask[row, :length] = mask[0]
        return input_ids, attention_mask

    def _classify(self, new_tokens: torch.Tensor, gen_kwargs: dict, params: GenerationParams) -> list[str | None]:
        """Per-row finish reasons under the pinned precedence, from the composed stop rules."""
        tokenizer = self.tokenizer
        return infer_finish_reasons(
            new_tokens,
            gen_kwargs,
            eos_token_id=getattr(tokenizer, "eos_token_id", None),
            pad_token_id=getattr(tokenizer, "pad_token_id", None),
            stop_strings=params.stop_strings,
            stop_token_ids=params.stop_token_ids,
            tokenizer=tokenizer,
        )

    def generate(
        self,
        items: Sequence[GenerationItem],
        params: GenerationParams,
    ) -> list[ItemResult]:
        """Generate one result per item, each under its own hook registrations.

        Items sharing identical state entries, identical output entries, and identical-or-absent
        effective seeds execute in one batched `model.generate` pass (right-padded to a common
        prompt length, then left-packed together); otherwise items decode serially. Caller-supplied `logits_processor` and
        `stopping_criteria` entries in `params.extra` append after the items' own contributions,
        and the normalized stop fields compose as stop rules anchored at the prompt length. A
        seeded item decodes inside a seeded RNG fork, so seeded runs are reproducible and the
        covered generator state (CPU, plus the model device's) is restored afterwards; when
        `params.seed` is set and an item carries no seed of its own, the item's seed derives per
        index, so multi-item fan-outs sample distinct streams.

        Args:
            items: The generation items.
            params: Normalized generation parameters shared by all items.

        Returns:
            One `ItemResult` per item, in item order. Each result's `finish_reasons` carries one
            reason per candidate with the precedence stop, then eos, then length, then None.
        """
        self._ensure_open()
        model = self.model
        tokenizer = self.tokenizer
        gen_kwargs = render_hf_gen_kwargs(params)
        # default pad_token_id per call to avoid the transformers open-end-generation warning
        # without mutating the model's generation config; caller kwargs and a model-configured
        # value both take precedence over the tokenizer fallback
        if (
            "pad_token_id" not in gen_kwargs
            and model.generation_config.pad_token_id is None
            and getattr(tokenizer, "pad_token_id", None) is not None
        ):
            gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
        user_processors = tuple(gen_kwargs.pop("logits_processor", None) or ())
        user_criteria = tuple(gen_kwargs.pop("stopping_criteria", None) or ())

        if not items:
            return []
        seeds = self._item_seeds(items, params)
        self._generate_count += 1

        batchable = (
            len(items) > 1
            and self._entries_identical(items)
            and len(set(seeds)) == 1
        )
        if batchable:
            return self._generate_batched(
                items, params, gen_kwargs, user_processors, user_criteria, seeds[0],
            )
        if len(items) > 1:
            self._report_serial_fallback(items)

        results: list[ItemResult] = []
        for index, item in enumerate(items):
            input_ids, attention_mask = self._resolve_prompt_tensors(item.prompt)
            processors, criteria = self._compose_entry_stacks(
                item.output_entries, extra_processors=user_processors, extra_criteria=user_criteria,
            )
            criteria.extend(compose_stop_criteria(params, input_ids.size(1), tokenizer))
            stacks = stack_generate_kwargs(processors, criteria)
            handles = self._register_state_entries(model, item.state_entries)
            try:
                seed = seeds[index]
                with self._seeded(seed):
                    if seed is not None:
                        self._apply_seed(seed)
                    full_ids = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        **stacks,
                        **gen_kwargs,
                    )
            finally:
                for handle in handles:
                    handle.remove()

            new_tokens = full_ids[:, input_ids.size(1):]
            reasons = self._classify(new_tokens, gen_kwargs, params)
            results.append(ItemResult(
                index=index,
                output=Output(
                    output_ids=new_tokens,
                    adapted_input_ids=input_ids,
                    finish_reason=reasons[0],
                    finish_reasons=tuple(reasons),
                ),
            ))
        return results

    def _generate_batched(
        self,
        items: Sequence[GenerationItem],
        params: GenerationParams,
        gen_kwargs: dict,
        user_processors: tuple,
        user_criteria: tuple,
        seed: int | None,
    ) -> list[ItemResult]:
        """One `model.generate` pass over all items (identical entries, one shared seed).

        The stacked prompts left-pack (pad positions move before the real tokens), so every row's
        first generated token is predicted from its last real token rather than from a trailing pad.
        """
        model = self.model
        rows = [self._resolve_prompt_tensors(item.prompt) for item in items]
        input_ids, attention_mask = self._stack_prompt_rows(rows)
        input_ids, attention_mask = to_left_pad(input_ids, attention_mask)
        processors, criteria = self._compose_entry_stacks(
            items[0].output_entries, extra_processors=user_processors, extra_criteria=user_criteria,
        )
        criteria.extend(compose_stop_criteria(params, input_ids.size(1), self.tokenizer))
        stacks = stack_generate_kwargs(processors, criteria)
        handles = self._register_state_entries(model, items[0].state_entries)
        try:
            with self._seeded(seed):
                if seed is not None:
                    self._apply_seed(seed)
                full_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **stacks,
                    **gen_kwargs,
                )
        finally:
            for handle in handles:
                handle.remove()

        prompt_len = input_ids.size(1)
        new_tokens = full_ids[:, prompt_len:]
        num_candidates = params.n or 1
        results: list[ItemResult] = []
        for index in range(len(items)):
            item_rows = new_tokens[index * num_candidates:(index + 1) * num_candidates]
            reasons = self._classify(item_rows, gen_kwargs, params)
            results.append(ItemResult(
                index=index,
                output=Output(
                    output_ids=item_rows,
                    adapted_input_ids=input_ids[index:index + 1],
                    finish_reason=reasons[0],
                    finish_reasons=tuple(reasons),
                ),
            ))
        return results

    def score(
        self,
        items: Sequence[ScoringItem],
        params: GenerationParams,
    ) -> torch.Tensor:
        """Teacher-forced log-probabilities of each item's reference tokens.

        For each item, the prompt left-packs (pad positions move before the real tokens) so the
        reference follows the prompt's last real token, then prompt and reference concatenate
        into one causal forward pass under the item's hook registrations; the item's logits
        processors replay position-by-position with the same `(prefix_ids, scores)` view they
        receive during generation. Items sharing identical state and output entries score in one
        batched forward pass (prompts right-padded to a common length, then left-packed
        together); otherwise items score serially. Stopping criteria never apply. Decoder-only
        models only; the pipeline's `compute_logprobs` serves encoder-decoder models in-process.

        Args:
            items: The scoring items. Every item must carry the same reference length.
            params: `params.extra` passes through as forward keyword arguments.

        Returns:
            Log probabilities of shape `[num_items, ref_len]`.

        Raises:
            ValueError: If items carry differing reference lengths.
            UnsupportedOperationError: If the model is an encoder-decoder model.
        """
        self._ensure_open()
        model = self.model
        if getattr(model.config, "is_encoder_decoder", False):
            raise UnsupportedOperationError(
                "Session scoring supports decoder-only models; encoder-decoder scoring runs "
                "through SteeringPipeline.compute_logprobs."
            )
        device = model.device
        forward_kwargs = dict(params.extra)

        if not items:
            return torch.zeros((0, 0), device=device, dtype=torch.float32)
        ref_lens = {item.ref_output_ids.shape[-1] for item in items}
        if len(ref_lens) > 1:
            raise ValueError(f"All scoring items must share one reference length; got {sorted(ref_lens)}.")

        if len(items) > 1 and self._entries_identical(items):
            return self._score_batched(items, forward_kwargs)

        all_logprobs: list[torch.Tensor] = []
        for item in items:
            input_ids, attention_mask = self._resolve_prompt_tensors(item.prompt)
            input_ids, attention_mask = to_left_pad(input_ids, attention_mask)
            ref_output_ids = item.ref_output_ids
            if ref_output_ids.ndim == 1:
                ref_output_ids = ref_output_ids.unsqueeze(0)
            ref_output_ids = ref_output_ids.to(device)
            ref_len = ref_output_ids.size(1)
            if ref_len == 0:
                all_logprobs.append(torch.zeros((1, 0), device=device, dtype=torch.float32))
                continue

            processors, _ = self._compose_entry_stacks(item.output_entries)
            handles = self._register_state_entries(model, item.state_entries)
            try:
                with torch.no_grad():
                    combined_ids = torch.cat([input_ids, ref_output_ids], dim=1)
                    combined_mask = torch.cat([
                        attention_mask,
                        torch.ones(1, ref_len, device=device, dtype=attention_mask.dtype),
                    ], dim=1)
                    outputs = model(
                        input_ids=combined_ids,
                        attention_mask=combined_mask,
                        **forward_kwargs,
                    )
                    input_len = input_ids.size(1)
                    logits = outputs.logits[:, input_len - 1: input_len + ref_len - 1, :]
                    if len(processors):
                        for t in range(logits.size(1)):
                            prefix = torch.cat([input_ids, ref_output_ids[:, :t]], dim=1)
                            logits[:, t, :] = processors(prefix, logits[:, t, :])
                    logprobs = torch.log_softmax(logits, dim=-1)
                    all_logprobs.append(
                        logprobs.gather(dim=-1, index=ref_output_ids.unsqueeze(-1)).squeeze(-1)
                    )
            finally:
                for handle in handles:
                    handle.remove()

        return torch.cat(all_logprobs, dim=0)

    def _score_batched(self, items: Sequence[ScoringItem], forward_kwargs: dict) -> torch.Tensor:
        """One causal forward pass over all items (identical entries)."""
        model = self.model
        device = model.device
        rows = [self._resolve_prompt_tensors(item.prompt) for item in items]
        input_ids, attention_mask = self._stack_prompt_rows(rows)
        input_ids, attention_mask = to_left_pad(input_ids, attention_mask)

        refs: list[torch.Tensor] = []
        for item in items:
            ref = item.ref_output_ids
            if ref.ndim == 1:
                ref = ref.unsqueeze(0)
            refs.append(ref.to(device))
        ref_output_ids = torch.cat(refs, dim=0)
        ref_len = ref_output_ids.size(1)
        if ref_len == 0:
            return torch.zeros((len(items), 0), device=device, dtype=torch.float32)

        processors, _ = self._compose_entry_stacks(items[0].output_entries)
        handles = self._register_state_entries(model, items[0].state_entries)
        try:
            with torch.no_grad():
                combined_ids = torch.cat([input_ids, ref_output_ids], dim=1)
                combined_mask = torch.cat([
                    attention_mask,
                    torch.ones(len(items), ref_len, device=device, dtype=attention_mask.dtype),
                ], dim=1)
                outputs = model(
                    input_ids=combined_ids,
                    attention_mask=combined_mask,
                    **forward_kwargs,
                )
                input_len = input_ids.size(1)
                logits = outputs.logits[:, input_len - 1: input_len + ref_len - 1, :]
                if len(processors):
                    for t in range(logits.size(1)):
                        prefix = torch.cat([input_ids, ref_output_ids[:, :t]], dim=1)
                        logits[:, t, :] = processors(prefix, logits[:, t, :])
                logprobs = torch.log_softmax(logits, dim=-1)
                return logprobs.gather(dim=-1, index=ref_output_ids.unsqueeze(-1)).squeeze(-1)
        finally:
            for handle in handles:
                handle.remove()

    def capture(
        self,
        prompts: list[PreparedPrompt],
        layers: list[int],
        mode: Literal["all_tokens", "last_token"],
        location: Literal["layer_output", "layer_input"] = "layer_output",
    ) -> CaptureResult:
        """Capture residual-stream hidden states for `prompts` at `layers`.

        Prompts resolve to token ids, right-pad into one batch, and run through the shared
        layerwise extraction. In `"all_tokens"` mode each layer's tensor is `[N, T, H]` with rows
        outside the attention mask zeroed; in `"last_token"` mode the last real (non-pad) position
        of each row is selected, giving `[N, H]`.

        Args:
            prompts: The prompts to capture.
            layers: 0-based layer ids to keep.
            mode: `"all_tokens"` or `"last_token"`.
            location: `"layer_output"` (a layer's output boundary) or `"layer_input"` (the
                boundary a forward pre-hook observes).

        Returns:
            The captured tensors and the batch attention mask, on CPU.

        Raises:
            ValueError: If `mode` is unknown or a requested layer id is out of range.
        """
        self._ensure_open()
        if mode not in ("all_tokens", "last_token"):
            raise ValueError(f"Unknown capture mode {mode!r}; modes are 'all_tokens', 'last_token'.")
        if not prompts:
            raise ValueError("capture() requires at least one prompt.")

        from steerability.algorithms.core.internals.capture import layerwise_tokenwise_hidden
        from steerability.algorithms.core.internals.pooling import aggregate_condition_hidden

        model = self.model
        device = model.device
        tokenizer = self.tokenizer
        pad_token_id = getattr(tokenizer, "pad_token_id", None) or 0

        rows = [self._resolve_prompt_tensors(prompt) for prompt in prompts]
        max_len = max(ids.size(1) for ids, _ in rows)
        input_ids = torch.full((len(rows), max_len), pad_token_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros((len(rows), max_len), dtype=torch.long, device=device)
        for row, (ids, mask) in enumerate(rows):
            length = ids.size(1)
            input_ids[row, :length] = ids[0]
            attention_mask[row, :length] = mask[0]

        enc = {"input_ids": input_ids, "attention_mask": attention_mask}
        hidden = layerwise_tokenwise_hidden(
            model, enc, batch_size=_CAPTURE_BATCH_SIZE, location=location,
        )
        missing = [layer for layer in layers if layer not in hidden]
        if missing:
            raise ValueError(
                f"Requested layer ids {missing} are out of range; the model has {len(hidden)} layers."
            )

        mask_cpu = attention_mask.cpu()
        selected = {layer: hidden[layer] for layer in layers}
        if mode == "last_token":
            selected = {
                layer: aggregate_condition_hidden(tensor, "last", mask_cpu)
                for layer, tensor in selected.items()
            }
        return CaptureResult(
            hidden=selected, attention_mask=mask_cpu, mode=mode, location=location,
        )


@contextlib.contextmanager
def _mps_rng_fork():
    """Snapshot and restore the CPU and MPS generator states around a seeded decode."""
    cpu_state = torch.get_rng_state()
    mps_state = torch.mps.get_rng_state()
    try:
        yield
    finally:
        torch.set_rng_state(cpu_state)
        torch.mps.set_rng_state(mps_state)
