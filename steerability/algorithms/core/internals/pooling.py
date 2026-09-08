"""Token, span, and position aggregation primitives for captured hidden states."""
from typing import Literal

import torch


def masked_mean(hidden: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    """Mean-pool `[N, T, H]` hidden states over real (non-pad) positions.

    Falls back to an unmasked mean over all positions when no attention mask is available.

    Args:
        hidden: Shape `[N, T, H]`.
        attention_mask: Shape `[N, T]` (1 for real tokens, 0 for pads), or None.

    Returns:
        Pooled tensor of shape `[N, H]`.
    """
    if attention_mask is None:
        return hidden.mean(dim=1)
    m = attention_mask.to(hidden.dtype).unsqueeze(-1)  # [N, T, 1]
    return (hidden * m).sum(dim=1) / m.sum(dim=1).clamp_min(1e-8)


@torch.no_grad()
def aggregate_condition_hidden(
    hidden: torch.Tensor,
    mode: Literal["mean", "last"],
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Aggregate `[B, T, H]` hidden states to `[B, H]` using non-pad tokens only.

    Args:
        hidden: Shape `[B, T, H]`.
        mode: "mean" pools over all real tokens; "last" selects the last real token per row.
        attention_mask: Shape `[B, T]` (1 for real tokens, 0 for pads), or None. When None, "mean"
            averages all positions and "last" uses the final position.

    Returns:
        Aggregated tensor of shape `[B, H]`.

    Raises:
        ValueError: If a row has no real tokens, or the mode is unsupported.
    """
    if mode == "mean":
        return masked_mean(hidden, attention_mask)

    if mode != "last":
        raise ValueError(f"Unsupported condition comparison mode: {mode!r}.")

    batch_size = hidden.size(0)
    if attention_mask is None:
        return hidden[:, -1, :]

    mask = attention_mask.to(hidden.device).bool()
    if not mask.any(dim=1).all():
        raise ValueError("aggregate_condition_hidden received a row with no real tokens.")

    sequence_length = mask.size(1)
    positions = torch.arange(sequence_length, device=hidden.device).unsqueeze(0)
    last_positions = positions.masked_fill(~mask, -1).max(dim=1).values  # [B]
    return hidden[torch.arange(batch_size, device=hidden.device), last_positions]


def select_spans(
    enc: dict[str, torch.Tensor],
    prompt_enc: dict[str, torch.Tensor] | None,
    accumulate: str,
) -> list[tuple[int, int]]:
    """Determine token spans to pool over for each sample.

    Span bounds are derived from the attention mask, so pads on either padding side are excluded.
    Under `accumulate="suffix-only"`, the per-sample prompt length (the number of real tokens in
    `prompt_enc`) is skipped at the start of each span.

    Args:
        enc: Tokenized full sequences (prompts + completions).
        prompt_enc: Tokenized prompts only (if accumulate == "suffix-only").
        accumulate: "all", "suffix-only", or "last_token".

    Returns:
        List of (start, end) tuples, one per sample.

    Raises:
        ValueError: If `accumulate` is unsupported.
    """
    if accumulate not in ("all", "suffix-only", "last_token"):
        raise ValueError(
            f"select_spans does not support accumulate='{accumulate}'. "
            "Expected one of: 'all', 'suffix-only', 'last_token'."
        )

    input_ids = enc["input_ids"]
    attention_mask = enc.get("attention_mask")
    N, T = input_ids.shape

    spans = []
    for i in range(N):
        # number of prompt tokens to skip for suffix-only pooling (a count, not an absolute index)
        if accumulate == "suffix-only" and prompt_enc is not None:
            prompt_len = (
                int(prompt_enc["attention_mask"][i].sum().item())
                if "attention_mask" in prompt_enc
                else prompt_enc["input_ids"].size(1)
            )
        else:
            prompt_len = 0

        # derive both bounds from the mask so the span excludes pads on either padding side
        if attention_mask is not None:
            non_pad = (attention_mask[i] == 1).nonzero(as_tuple=True)[0]
            if len(non_pad) > 0:
                first = int(non_pad[0].item())
                last = int(non_pad[-1].item())
            else:
                first, last = 0, T - 1
            if accumulate == "last_token":
                # one-token span at the final non-pad position (derived from the mask, so it works
                # regardless of padding side)
                start, end = last, last + 1
            else:
                start = first + prompt_len
                end = last + 1
        else:
            if accumulate == "last_token":
                start, end = T - 1, T
            else:
                start = prompt_len
                end = T

        spans.append((start, end))

    return spans


def pool_over_spans(
    hidden: torch.Tensor,
    spans: list[tuple[int, int]],
) -> torch.Tensor:
    """Mean-pool hidden states over specified spans.

    A degenerate span (`start >= end`) falls back to the sample's last token position.

    Args:
        hidden: Shape `[N, T, H]`.
        spans: List of `(start, end)` tuples.

    Returns:
        Pooled tensor of shape `[N, H]`.
    """
    N, T, H = hidden.shape
    pooled = []
    for i, (start, end) in enumerate(spans):
        if start >= end:
            # fallback: use last token
            pooled.append(hidden[i, -1, :])
        else:
            pooled.append(hidden[i, start:end, :].mean(dim=0))
    return torch.stack(pooled, dim=0)


def get_last_token_positions(
    attention_mask: torch.Tensor | None,
    seq_len: int,
    num_samples: int,
) -> torch.LongTensor:
    """Find the last non-pad token position for each sample.

    Args:
        attention_mask: Shape `[N, T]` or None.
        seq_len: Sequence length `T`.
        num_samples: Number of samples `N`.

    Returns:
        Tensor of shape `[N]` with last token positions.
    """
    if attention_mask is None:
        # no padding, last token is at seq_len - 1
        return torch.full((num_samples,), seq_len - 1, dtype=torch.long)

    # for each sample, find the last position where attention_mask == 1
    # this handles both left-padded and right-padded sequences
    positions = torch.arange(seq_len, device=attention_mask.device).unsqueeze(0).expand(num_samples, -1)
    # mask out padded positions with -1
    masked_positions = torch.where(attention_mask == 1, positions, torch.tensor(-1, device=attention_mask.device))
    return masked_positions.max(dim=1).values


def select_at_positions(
    hidden: torch.Tensor,
    positions: torch.LongTensor,
) -> torch.Tensor:
    """Select hidden states at specified positions for each sample.

    Args:
        hidden: Shape `[N, T, H]`.
        positions: Shape `[N]` with position indices.

    Returns:
        Tensor of shape `[N, H]`.
    """
    N, _, H = hidden.shape
    # gather at the specified positions
    idx = positions.view(N, 1, 1).expand(N, 1, H)
    return hidden.gather(dim=1, index=idx).squeeze(1)
