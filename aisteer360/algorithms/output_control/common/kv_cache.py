"""KV-cache utilities for same-model candidate evaluation.

`repeat_cache` / `select_cache` handle the cache-format compatibility that same-model value
functions need: repeat a prefix cache across K candidates, then select the chosen candidate's slice.
`extends_prefix` / `full_prefix_mask` are the pure tensor helpers an incremental prefix cache needs:
whether a new prefix extends the cached one, and the full attention mask spanning a prefix.

Mutation contract: both functions may mutate the input cache
in-place on some backends (`batch_repeat_interleave`, `batch_select`, in-place key/value lists) and
return a fresh cache on others (the legacy tuple format, where `to_legacy_cache` round-trips). Treat
the input cache as consumed after the call and use only the returned handle. The exception is
`repeat_cache(..., preserve_input=True)`, which never mutates the input (it takes only the copying
paths); use it when the input cache must survive the call (e.g. an incremental prefix cache repeated
across candidates).
"""
from __future__ import annotations

import torch
from transformers.cache_utils import DynamicCache


def extends_prefix(cached_ids: torch.Tensor | None, ids: torch.Tensor) -> bool:
    """Whether `ids` extends `cached_ids` row-for-row (a shared, no-shorter prefix).

    Args:
        cached_ids: The previously cached prefix ids `[B, T_c]`, or None when nothing is cached.
        ids: The candidate new prefix ids `[B, T]`.

    Returns:
        True when `cached_ids` is not None and `ids` is at least as long and matches `cached_ids`
        over its first `T_c` positions; False otherwise (including any shorter or divergent prefix,
        which must trigger a cache rebuild).
    """
    if cached_ids is None or ids.size(1) < cached_ids.size(1):
        return False
    return bool(torch.equal(ids[:, : cached_ids.size(1)], cached_ids.to(ids.device)))


def full_prefix_mask(prefix_ids: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    """The provided mask right-extended with ones to the prefix length.

    The mask must span the full prefix; generated tokens are always real, so right-extension with
    ones is exact.

    Args:
        prefix_ids: The prefix ids `[B, T]`.
        attention_mask: The prefix mask `[B, T']` with `T' <= T`, or None for an all-ones mask.

    Returns:
        An attention mask `[B, T]`.

    Raises:
        ValueError: If `attention_mask` is longer than `prefix_ids`.
    """
    if attention_mask is None:
        return torch.ones_like(prefix_ids)
    pad = prefix_ids.size(1) - attention_mask.size(1)
    if pad < 0:
        raise ValueError("attention_mask is longer than prefix_ids.")
    if pad == 0:
        return attention_mask
    ones = torch.ones(attention_mask.size(0), pad, device=attention_mask.device, dtype=attention_mask.dtype)
    return torch.cat([attention_mask, ones], dim=1)


def repeat_cache(cache, n: int, *, preserve_input: bool = False):
    """Repeat every cache entry `n` times along the batch dimension.

    Args:
        cache: A KV cache (`DynamicCache`, legacy tuple, or key/value-list style).
        n: Number of repeats per entry.
        preserve_input: When True, never mutate `cache`; take the copying paths (the legacy-tuple
            round-trip for `DynamicCache`-style caches, the tensor-building path for raw tuples) and
            raise `TypeError` for cache types that offer only in-place repetition. The returned cache
            shares no batch-repeated storage with the input.

    Returns:
        The repeated cache (may alias the input unless `preserve_input=True`; see the module
        mutation contract).

    Raises:
        TypeError: If the cache type is unsupported, or if `preserve_input=True` and the cache offers
            only in-place repetition.
    """
    if hasattr(cache, "batch_repeat_interleave") and not preserve_input:
        cache.batch_repeat_interleave(n)
        return cache

    if hasattr(cache, "to_legacy_cache"):
        raw = cache.to_legacy_cache()
        repeated = tuple(
            tuple(t.repeat(n, 1, 1, 1) for t in layer)
            for layer in raw
        )
        return DynamicCache.from_legacy_cache(repeated)

    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        if preserve_input:
            raise TypeError(
                f"{type(cache).__name__} offers only in-place repetition; cannot preserve_input."
            )
        for i in range(len(cache.key_cache)):
            cache.key_cache[i] = cache.key_cache[i].repeat_interleave(n, dim=0)
            cache.value_cache[i] = cache.value_cache[i].repeat_interleave(n, dim=0)
        return cache

    if isinstance(cache, tuple):
        return tuple(
            tuple(t.repeat_interleave(n, dim=0) for t in layer)
            for layer in cache
        )

    raise TypeError(f"Unsupported cache type: {type(cache).__name__}")


def select_cache(cache, idx: torch.Tensor):
    """Select cache entries along the batch dimension by index.

    Args:
        cache: A KV cache (`DynamicCache`, legacy tuple, or key/value-list style).
        idx: 1-D index tensor of rows to keep.

    Returns:
        The selected cache (may alias the input; see the module mutation contract).

    Raises:
        ValueError: If `idx` is not 1-D.
        TypeError: If the cache type is unsupported.
    """
    if not torch.is_tensor(idx):
        idx = torch.as_tensor(idx)
    if idx.dtype != torch.long:
        idx = idx.long()
    if idx.dim() != 1:
        raise ValueError(f"idx must be 1D, got shape {tuple(idx.shape)}")

    if hasattr(cache, "batch_select"):
        cache.batch_select(idx)
        return cache

    if hasattr(cache, "batch_gather"):
        cache.batch_gather(idx)
        return cache

    if hasattr(cache, "to_legacy_cache"):
        raw = cache.to_legacy_cache()
        selected = tuple(
            tuple(t[idx, :, :, :] for t in layer)
            for layer in raw
        )
        return DynamicCache.from_legacy_cache(selected)

    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        for i in range(len(cache.key_cache)):
            if cache.key_cache[i] is not None:
                cache.key_cache[i] = cache.key_cache[i].index_select(dim=0, index=idx.to(cache.key_cache[i].device))
            if cache.value_cache[i] is not None:
                cache.value_cache[i] = cache.value_cache[i].index_select(
                    dim=0, index=idx.to(cache.value_cache[i].device)
                )
        return cache

    if isinstance(cache, tuple):
        return tuple(
            tuple(t.index_select(dim=0, index=idx.to(t.device)) for t in layer)
            for layer in cache
        )

    raise TypeError(f"Unsupported cache type: {type(cache).__name__}")
