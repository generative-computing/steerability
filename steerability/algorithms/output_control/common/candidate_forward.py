"""Same-model forward of candidate continuations with prefix KV-cache reuse.

Some candidate values require forwarding the pipeline's own model mid-step to read candidate
hidden states; `CandidateForward` performs those forwards. The states it reports lie at the raw
output boundary of the final decoder layer (`location="layer_output"` in the capture utilities),
before the model's final norm. These passes are marked as auxiliary via
`auxiliary_pass(aligned=True)`, so state-control accounting keeps them out of condition
scoring and gate updates while transforms still apply at the candidates' true positions (prefix
and candidate positions lie on the generation's own coordinate axis). At hook points where the
state runtime cannot read pass positions, it skips transforming these passes and
warns once. Values scored by an auxiliary model are unaffected.
"""
from __future__ import annotations

import torch
from transformers import PreTrainedModel

from steerability.algorithms.core.internals.model_layout import resolve_model_layout
from steerability.algorithms.core.utils.auxiliary_pass import auxiliary_pass
from steerability.algorithms.output_control.common.kv_cache import extends_prefix, full_prefix_mask, repeat_cache


class CandidateForward:
    """Batched forward of `(prefix + candidate_k)` for all K candidates on a given model.

    Maintains an internal prefix KV cache keyed on the prefix ids. When a call's prefix extends the
    cached one, only the delta tokens are forwarded to extend the cache (one 1-token forward per
    decode step in the common case); any non-extension (rewind, restart, new generation, scoring
    replay from an unrelated prefix) rebuilds from scratch. The candidate evaluation then repeats a
    fresh copy of the cache across the K candidates, so the incremental cache survives the call.
    Explicit `cache_position` is passed on every forward, including the prefix rebuild (this is what
    `generate` does internally), and the model derives token positions from it, so the prefix must
    be unpadded: an attention mask containing zeros is rejected. Supports batch size 1 only.

    Args:
        model: The model to forward through (the pipeline's own model for same-model values).
    """

    def __init__(self, model: PreTrainedModel):
        self.model = model
        self._final_layer = model.get_submodule(resolve_model_layout(model).layer_names[-1])
        self._cached_ids: torch.Tensor | None = None  # [1, T_c]
        self._cached_mask: torch.Tensor | None = None  # [1, T_c]
        self._cache = None  # past_key_values covering _cached_ids

    def _sync_cache(self, prefix_ids: torch.Tensor, full_mask: torch.Tensor) -> None:
        """Bring the internal cache up to `prefix_ids` (extend by the delta, or rebuild)."""
        if not extends_prefix(self._cached_ids, prefix_ids):
            out = self.model(
                input_ids=prefix_ids, attention_mask=full_mask, use_cache=True, return_dict=True,
                cache_position=torch.arange(prefix_ids.size(1), device=prefix_ids.device),
            )
            self._cache = out.past_key_values
        else:
            cached_len = self._cached_ids.size(1)
            if prefix_ids.size(1) > cached_len:
                delta = prefix_ids[:, cached_len:]
                positions = torch.arange(cached_len, prefix_ids.size(1), device=prefix_ids.device)
                out = self.model(
                    input_ids=delta, attention_mask=full_mask, past_key_values=self._cache,
                    use_cache=True, cache_position=positions, return_dict=True,
                )
                self._cache = out.past_key_values
            # equal length + extends -> cache already current; nothing to do
        self._cached_ids = prefix_ids.detach()
        self._cached_mask = full_mask

    @torch.no_grad()
    def last_hidden_states(
        self,
        prefix_ids: torch.Tensor,
        candidate_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return final-layer hidden states at the candidate position for each candidate.

        The returned states lie at the raw output boundary of the final decoder layer
        (`location="layer_output"` in the capture utilities), before the model's final norm,
        recovered with a forward hook on that layer. The hook is registered per call, around the
        candidate forward only, so it observes the output after any session-registered
        state-control hooks on the same module.

        Args:
            prefix_ids: `[1, T]` prefix (batch size 1).
            candidate_ids: `[1, K]` candidate next tokens.
            attention_mask: Optional prefix mask; right-extended with ones to the prefix length.
                Must contain no zeros.

        Returns:
            A tensor `[K, H]` of candidate-position hidden states, one per candidate.

        Raises:
            ValueError: If the prefix batch size is not 1, or the attention mask contains zeros.
            RuntimeError: If the candidate forward does not pass through the final decoder layer
                exactly once.
        """
        if prefix_ids.size(0) != 1:
            raise ValueError("CandidateForward supports batch size 1 only.")
        if attention_mask is not None and not bool(attention_mask.all()):
            raise ValueError(
                "CandidateForward positions tokens by sequence index and requires an unpadded prefix; "
                "the attention mask contains zeros."
            )

        num = candidate_ids.size(1)
        device = prefix_ids.device
        full_mask = full_prefix_mask(prefix_ids, attention_mask)
        with auxiliary_pass(aligned=True):
            self._sync_cache(prefix_ids, full_mask)

            prefix_len = prefix_ids.size(1)
            repeated = repeat_cache(self._cache, num, preserve_input=True)
            cand_tokens = candidate_ids.reshape(num, 1)
            cand_mask = torch.cat(
                [full_mask.repeat(num, 1), torch.ones(num, 1, device=device, dtype=full_mask.dtype)],
                dim=1,
            )
            positions = torch.arange(prefix_len, prefix_len + 1, device=device)

            final_boundary: list[torch.Tensor] = []

            def _grab_final(module, args, output):
                final_boundary.append(output[0] if isinstance(output, tuple) else output)

            handle = self._final_layer.register_forward_hook(_grab_final)
            try:
                self.model(
                    input_ids=cand_tokens,
                    attention_mask=cand_mask,
                    past_key_values=repeated,
                    use_cache=True,
                    cache_position=positions,
                    return_dict=True,
                )
            finally:
                handle.remove()
        if len(final_boundary) != 1:
            raise RuntimeError(
                f"Expected exactly one final-layer forward for the candidate batch, "
                f"observed {len(final_boundary)}."
            )
        return final_boundary[0][:, -1, :]  # [K, H]
