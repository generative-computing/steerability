from __future__ import annotations

import logging
from functools import partial
from typing import Sequence

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from aisteer360.algorithms.core.execution.access import ModelAccess
from aisteer360.algorithms.core.execution.contracts import Capability, Requirements, SpecConstraint, needs
from aisteer360.algorithms.core.execution.spec import BackendSpec
from aisteer360.algorithms.state_control.base import HookControl
from aisteer360.algorithms.state_control.common.model_layout import resolve_model_layout
from aisteer360.algorithms.state_control.pasta.args import PASTAArgs

logger = logging.getLogger(__name__)

SUPPORTED_ATTN_IMPLEMENTATIONS = ("eager", "sdpa")


def _attn_implementation_supported(spec: BackendSpec) -> bool:
    """True unless a huggingface spec configures an attention implementation PASTA cannot steer."""
    if spec.kind != "huggingface":
        return True
    impl = spec.get_option("hf_model_kwargs", "attn_implementation")
    return impl is None or impl in SUPPORTED_ATTN_IMPLEMENTATIONS


class PASTA(HookControl):
    """
    Implementation of PASTA (Post-hoc Attention STeering Approach) from Zhang et al., 2023.

    PASTA performs controlled text generation by dynamically modifying attention patterns during inference to amplify or
    suppress the influence of specific text spans. This allows for fine-grained steering of model behavior without
    requiring model retraining or parameter updates.

    The algorithm works by:

    1. **Substring Identification**: Locate target substrings within the input prompt using tokenizer offset mapping to
    determine precise token ranges.

    2. **Attention Modification**: Inject scaling factors into the attention mask of specified layers and heads to
    increase or decrease attention weights for the identified token ranges.

    3. **Dynamic Steering**: Apply different scaling strategies (include, exclude, or generation-focused) to control how
    the model attends to relevant spans during text generation.

    This approach enables real-time control over model focus and can be used for tasks like concept amplification, bias
    mitigation, or content filtering without architectural changes.

    Args:
        alpha (float): Multiplicative scaling factor applied to attention weights (implemented as adding log(alpha)
            to attention logits via the attention mask). Values > 1 amplify attention to the targeted span; values in
            (0, 1) suppress it. Must be > 0. Defaults to 1.0.
        head_config (dict | list): Configuration specifying which layers/heads to modify. If dict, maps layer indices
            to lists of head indices. If list, applies to all heads in specified layers.
        scale_position (str): Strategy for applying attention scaling. Options:

            - "include": Scale attention TO the target substrings
            - "exclude": Scale attention AWAY FROM the target substrings
            - "generation": Scale attention during generation phase

            Defaults to "include".

    Note:
        PASTA injects a 4D additive attention mask, which only the `"eager"` and `"sdpa"` attention
        implementations consume. Load the model with `attn_implementation="eager"` (or `"sdpa"`);
        `"flash_attention_2"` expects a 2D padding mask and silently ignores or errors on the
        injected scaling. `steer()` fails fast if the model reports an unsupported implementation.

    Reference:
    - "PASTA: Tell Your Model Where to Attend: Post-hoc Attention Steering for LLMs"
    Qingru Zhang, Chandan Singh, Liyuan Liu, Xiaodong Liu, Bin Yu, Jianfeng Gao, Tuo Zhao
    [https://arxiv.org/abs/2311.02262](https://arxiv.org/abs/2311.02262)
    """

    Args = PASTAArgs
    RUNTIME_KWARGS_SCHEMA = [
        {
            "name": "substrings",
            "type": "list[str]",
            "required": True,
            "help": "Substrings whose attention should be steered. Required at inference time.",
        },
    ]

    supports_batching: bool = True

    # placeholders
    model: PreTrainedModel | None = None
    tokenizer: PreTrainedTokenizer | None = None
    device: torch.device | str | None = None

    _head_map: dict[int, list[int]] | None = None
    _layers: list[int] | None = None
    _attn_module_names: dict[int, str] | None = None
    _scale_constant: torch.Tensor | None = None

    def requirements(self) -> Requirements:
        """Backend requirements for attention-map editing.

        PASTA writes into attention maps through torch hooks, which fused paged-attention
        kernels never materialize, so the generate phase requires `Capability.IN_PROCESS_TORCH`.
        The spec constraint reports a configured incompatible attention implementation
        (`hf_model_kwargs["attn_implementation"]` outside `"eager"`/`"sdpa"`) at `check()` time,
        before any model loads; the same condition is re-checked against the live model in
        `steer()`.

        Returns:
            The control's phase-keyed requirements.
        """
        return Requirements(
            generate=needs(Capability.IN_PROCESS_TORCH),
            spec_constraints=(
                SpecConstraint(
                    description=(
                        "PASTA requires attn_implementation 'eager' or 'sdpa' to inject a 4D "
                        "attention mask; set attn_implementation=\"eager\" in hf_model_kwargs."
                    ),
                    predicate=_attn_implementation_supported,
                ),
            ),
        )

    def steer_access(self) -> ModelAccess:
        """`ModelAccess.MODULE`; the attention module paths resolve on the live model, which
        is retained for the hook closures (the generate phase is in-process)."""
        return ModelAccess.MODULE

    def steer(
        self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer | None = None, **__
    ) -> PreTrainedModel:
        """Initialize PASTA by configuring attention head mappings and model references.

        Sets up the layer and head configurations that will be modified during generation,
        resolves the architecture-specific attention module paths, and fails fast on unsupported
        layers or attention implementations (rather than deep inside generation).

        Args:
            model (PreTrainedModel): The base language model to be steered.
            tokenizer (PreTrainedTokenizer | None): Tokenizer for substring identification.
                If None, attempts to retrieve from model attributes.
            **__: Additional arguments (unused).

        Returns:
            PreTrainedModel: The input model (unchanged).

        Raises:
            ValueError: If a configured attention module path is missing on the model, or if the
                model's attention implementation is not one of `"eager"` / `"sdpa"`.
        """
        self.model = model
        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        self.device = next(model.parameters()).device
        self._setup_head_config(self.head_config)
        self._resolve_attention_modules(model)
        self._check_attention_implementation(model)
        return model

    def _resolve_attention_modules(self, model: PreTrainedModel) -> None:
        """Resolve the per-layer attention module path from the model layout.

        The per-layer attention module paths come from `resolve_model_layout` (`.self_attn` for
        `model.layers.*`, `.attn` for `transformer.h.*`). Validates every configured layer's module
        exists so registration cannot fail mid-generation.

        Raises:
            ValueError: If the architecture is unrecognized or a configured module path is absent.
        """
        attn_names = resolve_model_layout(model).attn_names

        self._attn_module_names = {}
        for layer in self._layers:
            if layer < 0 or layer >= len(attn_names):
                raise ValueError(
                    f"PASTA layer {layer} out of range for model with {len(attn_names)} layers."
                )
            path = attn_names[layer]
            try:
                model.get_submodule(path)
            except AttributeError as error:
                raise ValueError(f"PASTA could not resolve attention module {path!r}.") from error
            self._attn_module_names[layer] = path

    @staticmethod
    def _check_attention_implementation(model: PreTrainedModel) -> None:
        """Fail fast unless the model uses an attention implementation PASTA can steer.

        Raises:
            ValueError: If `model.config._attn_implementation` is not `"eager"` or `"sdpa"`.
        """
        impl = getattr(model.config, "_attn_implementation", "eager")
        if impl not in {"eager", "sdpa"}:
            raise ValueError(
                f"PASTA requires attn_implementation 'eager' or 'sdpa' to inject a 4D attention "
                f"mask; got {impl!r}. Load the model with attn_implementation=\"eager\"."
            )

    def get_hooks(
        self,
        input_ids: torch.Tensor,
        runtime_kwargs: dict | None,
        **__,
    ) -> dict[str, list]:
        """Create attention modification hooks for specified substrings.

        Identifies token ranges corresponding to target substrings and prepares hooks that will modify attention weights
        during the forward pass.

        Args:
            input_ids (torch.Tensor): Input token IDs of shape [batch_size, seq_len].
            runtime_kwargs (dict | None): Must contain "substrings" key with target text spans:

                - str: Single substring applied to all batch items
                - list[str]: List of substrings applied to all batch items
                - list[list[str]]: Per-batch substring groups
            **__: Additional arguments (unused).

        Returns:
            dict[str, list]: Hook specifications with "pre", "forward", "backward" keys. Only "pre" hooks are populated for attention modification.

        Raises:
            ValueError: If "substrings" not in runtime_kwargs or batch size mismatch.
        """
        if not runtime_kwargs or "substrings" not in runtime_kwargs:
            raise ValueError("PASTA requires 'substrings' inside runtime_kwargs")

        substrings = runtime_kwargs["substrings"]
        batch_size = input_ids.size(0)

        # normalize to (batch, group, str) in a local copy so we never mutate the caller's list
        if isinstance(substrings, str):
            groups: list[list[str]] = [[substrings] for _ in range(batch_size)]
        elif substrings and isinstance(substrings[0], str):
            groups = [list(substrings) for _ in range(batch_size)]
        elif len(substrings) != batch_size:
            raise ValueError(
                f"Need {batch_size} substring groups (one per prompt); got {len(substrings)}"
            )
        else:
            groups = [list(group) for group in substrings]

        # decode *with* special tokens so offsets share the attention mask's coordinate system
        # (its key axis includes BOS/template tokens)
        prompts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=False)

        # round-trip the substrings through the tokenizer so they match the decoded text
        # (tokenization can subtly change text, e.g. drop spaces)
        for group_idx, group in enumerate(groups):
            try:
                groups[group_idx] = self.tokenizer.batch_decode(
                    self.tokenizer(group, return_tensors="pt", padding=True)["input_ids"],
                    skip_special_tokens=True,
                )
            except Exception as error:
                raise ValueError(
                    f"PASTA failed to re-tokenize substrings {group!r}: {error}"
                ) from error

        # per item: re-encode the decoded text (no specials) and locate token ranges. A faithful
        # fast tokenizer reproduces the real input ids, so its offsets are already in real
        # coordinates; processing per item also avoids any batch padding / padding_side handling.
        token_ranges: list[torch.Tensor] = []
        for item_idx in range(batch_size):
            row_ids = input_ids[item_idx]
            enc = self.tokenizer(
                prompts[item_idx],
                return_tensors="pt",
                return_offsets_mapping=True,
                add_special_tokens=False,
            )
            enc_ids = enc["input_ids"][0].to(row_ids.device)
            offsets = enc["offset_mapping"][0].tolist()

            if torch.equal(enc_ids, row_ids):
                # id-faithful: offsets are already in the real sequence's coordinates
                ranges = [
                    torch.tensor(self._find_token_range(prompts[item_idx], sub, offsets))
                    for sub in groups[item_idx]
                ]
            else:
                # not id-faithful (e.g. sentencepiece whitespace normalization): compute the range
                # in re-encoded space, then relocate the token-id window inside the real ids
                ranges = [
                    torch.tensor(
                        self._locate_range_by_ids(
                            prompts[item_idx], sub, offsets, enc_ids, row_ids
                        )
                    )
                    for sub in groups[item_idx]
                ]
            token_ranges.append(torch.stack(ranges))

        # input_len is the real (padded) sequence length, matching the attention mask's key axis
        input_len = input_ids.size(1)

        if self._scale_constant is None:
            self._scale_constant = torch.tensor(
                [self.alpha],
                device=self.device,
                dtype=torch.float32,
            ).log()

        hooks: dict[str, list] = {"pre": [], "forward": [], "backward": []}
        for layer in self._layers:
            hooks["pre"].append(
                {
                    "module": self._attn_module_names[layer],
                    "hook_func": partial(
                        self._attention_pre_hook,
                        head_idx=self._head_map[layer],
                        token_ranges=token_ranges,
                        input_len=input_len,
                    ),
                }
            )

        return hooks

    def _setup_head_config(self, head_config):
        """Parse and validate attention head configuration.

        Converts various configuration formats into internal layer-head mappings and validates against model architecture.

        Args:
            head_config: Configuration specifying which layers/heads to modify:

                - dict: Maps layer indices to lists of head indices
                - list: Layer indices (applies to all heads in those layers)

        Raises:
            ValueError: If configuration format invalid or heads out of range.
        """
        if isinstance(head_config, dict):
            self._head_map = {int(l): list(h) for l, h in head_config.items()}
            self._layers = sorted(self._head_map.keys())
        elif isinstance(head_config, list):
            self._layers = [int(l) for l in head_config]
            self._head_map = {
                l: list(range(self.model.config.num_attention_heads))
                for l in self._layers
            }
        else:
            raise ValueError(f"Invalid head configuration: {head_config!r}")

        num_heads = self.model.config.num_attention_heads
        for layer, heads in self._head_map.items():
            for head in heads:
                if not 0 <= head < num_heads:
                    raise ValueError(
                        f"Head {head} out of range for layer {layer} (0–{num_heads-1})"
                    )

    @staticmethod
    def _find_token_range(
        string: str,
        substring: str,
        offset_mapping: Sequence[tuple[int, int]],
        occurrence: int = 0,
    ) -> tuple[int, int]:
        """Map a substring to its token index range using offset mapping.

        Locates the character positions of a substring and converts them to token indices using the tokenizer's offset mapping.

        Args:
            string: Full text to search within.
            substring: Target substring to locate.
            offset_mapping: List of (start_char, end_char) tuples for each token.
            occurrence: Which occurrence to find if substring appears multiple times.
                Defaults to 0 (first occurrence).

        Returns:
            tuple[int, int]: Start (inclusive) and end (exclusive) token indices. If the substring is
                absent from `string`, returns the `(0, 0)` sentinel (skipped downstream) after warning.

        Raises:
            ValueError: If the substring is present in `string` but cannot be aligned to token offsets.
        """
        if substring not in string:
            logger.warning(
                "PASTA: substring %r not found in input (len=%d chars); skipping steering for this range.",
                substring, len(string),
            )
            return 0, 0

        char_index = -1
        for _ in range(occurrence + 1):
            char_index = string.index(substring, char_index + 1)
        char_start = char_index
        char_end = char_start + len(substring)

        token_start = token_end = None
        for token_idx, (start_char, end_char) in enumerate(offset_mapping):
            if token_start is None and start_char <= char_start < end_char:
                token_start = token_idx
            if token_end is None and start_char < char_end <= end_char:
                token_end = token_idx

        if token_start is None or token_end is None:
            raise ValueError("Could not map substring to token range")

        return token_start, token_end + 1

    @staticmethod
    def _locate_range_by_ids(
        text: str,
        substring: str,
        offsets: Sequence[tuple[int, int]],
        enc_ids: torch.Tensor,
        real_ids: torch.Tensor,
    ) -> tuple[int, int]:
        """Fallback range resolution when re-encoding is not id-faithful to the real sequence.

        Computes the token range in the re-encoded (special-free) space, extracts that window of
        token ids, and finds the matching contiguous id window inside the real sequence (choosing
        the occurrence nearest the naive index). Returns the `(0, 0)` skip sentinel if the
        substring is absent or its id window cannot be located in the real ids.

        Args:
            text: The decoded text the offsets were computed against.
            substring: The target substring.
            offsets: Offset mapping for the re-encoded text.
            enc_ids: Token ids of the re-encoded text (same coordinate system as `offsets`).
            real_ids: Token ids of the real (padded) sequence to relocate the window into.

        Returns:
            `(start, end)` token indices in the real sequence, or `(0, 0)` if unresolved.
        """
        naive_start, naive_end = PASTA._find_token_range(text, substring, offsets)
        if naive_start == naive_end:  # absent (already warned) or empty
            return 0, 0

        window = enc_ids[naive_start:naive_end]
        w_len = window.size(0)
        real = real_ids.tolist()
        window_list = window.tolist()

        # every contiguous position in real_ids that matches the window
        matches = [
            i for i in range(len(real) - w_len + 1)
            if real[i:i + w_len] == window_list
        ]
        if not matches:
            logger.warning(
                "PASTA: could not relocate substring %r window into the real sequence; skipping.",
                substring,
            )
            return 0, 0

        # pick the occurrence nearest the naive index
        best = min(matches, key=lambda i: abs(i - naive_start))
        return best, best + w_len

    def _attention_pre_hook(
        self,
        module,
        input_args: tuple,
        input_kwargs: dict,
        head_idx: list[int],
        token_ranges: list[torch.Tensor],
        input_len: int,
    ):
        """Modify attention mask to steer focus toward/away from target tokens.

        Pre-forward hook that adjusts attention weights by adding scaling factors to the attention mask for specified token ranges and attention heads.

        Args:
            module: The attention module being hooked.
            input_args: Positional arguments to the forward pass.
            input_kwargs: Keyword arguments to the forward pass.
            head_idx: List of attention head indices to modify.
            token_ranges: Token index ranges to apply scaling to.
            input_len: Length of input sequence (for generation positioning).

        Returns:
            Tuple of potentially modified (input_args, input_kwargs).

        Raises:
            RuntimeError: If hidden states cannot be located.
            ValueError: If scale_position is invalid.
        """
        hidden_states = (
            input_args[0] if input_args else input_kwargs.get("hidden_states")
        )
        if hidden_states is None:
            raise RuntimeError("PASTA: could not locate hidden states")

        attention_mask = input_kwargs.get("attention_mask")
        if attention_mask is None:  # build it
            batch_size, query_len, _ = hidden_states.size()
            num_heads = self.model.config.num_attention_heads

            # during decoding the query attends to the full kv cache, so the mask spans the cached key
            # positions (read from cache_position) rather than just the current query window
            cache_position = input_kwargs.get("cache_position")
            if cache_position is not None:
                key_len = int(cache_position[-1]) + 1
            else:
                key_len = query_len

            # query row i sits at absolute position (key_len - query_len + i) and attends to keys 0..position
            query_positions = torch.arange(
                key_len - query_len, key_len, device=hidden_states.device
            ).unsqueeze(1)
            key_positions = torch.arange(key_len, device=hidden_states.device).unsqueeze(0)
            causal = torch.where(
                key_positions <= query_positions,
                hidden_states.new_zeros(()),
                hidden_states.new_full((), float("-inf")),
            )
            attention_mask = causal[None, None]  # (1,1,q,k)
            attention_mask = attention_mask.expand(
                batch_size, num_heads, -1, -1
            ).contiguous()
            input_kwargs["attention_mask"] = attention_mask

        attention_mask = attention_mask.to(hidden_states.dtype).contiguous().clone()
        if attention_mask.size(1) == 1:
            attention_mask = attention_mask.expand(
                -1,
                self.model.config.num_attention_heads,
                -1,
                -1,
            ).contiguous()

        batch_size = attention_mask.size(0)

        # beam search expands the batch via repeat_interleave ([item0, item0, item1, item1]); index
        # token_ranges the same way (a modulo would misorder the beams)
        if batch_size > len(token_ranges):
            if batch_size % len(token_ranges) != 0:
                raise RuntimeError(
                    f"Hidden batch {batch_size} is not a multiple of the prompt batch "
                    f"{len(token_ranges)}; cannot align PASTA token ranges."
                )
            expand = batch_size // len(token_ranges)
            token_ranges = [token_ranges[i // expand] for i in range(batch_size)]

        for batch_index in range(batch_size):
            ranges = token_ranges[batch_index].tolist()
            has_valid_range = any(start != end for start, end in ranges)
            for start_idx, end_idx in ranges:
                if start_idx == end_idx:
                    continue
                if self.scale_position == "include":
                    attention_mask[
                        batch_index, head_idx, :, start_idx:end_idx
                    ] += self._scale_constant
                elif self.scale_position == "exclude":
                    attention_mask[
                        batch_index, head_idx, :, :start_idx
                    ] += self._scale_constant
                    attention_mask[
                        batch_index, head_idx, :, end_idx:input_len
                    ] += self._scale_constant
                elif self.scale_position == "generation":
                    attention_mask[
                        batch_index, head_idx, :, :input_len
                    ] += self._scale_constant

                else:
                    raise ValueError(f"Unknown scale_position '{self.scale_position}'")

            if self.scale_position == "include" and has_valid_range:
                attention_mask[batch_index, head_idx, :, :input_len] -= self._scale_constant

        input_kwargs["attention_mask"] = attention_mask
        return input_args, input_kwargs
