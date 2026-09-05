from __future__ import annotations

import logging
from functools import partial
from typing import TYPE_CHECKING, Sequence

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from steerability.algorithms.core.execution.access import ModelAccess
from steerability.algorithms.core.execution.contracts import Capability, Requirements, SpecConstraint, needs
from steerability.algorithms.core.execution.spec import BackendSpec
from steerability.algorithms.core.internals.model_layout import head_geometry, resolve_model_layout
from steerability.algorithms.state_control.base import HookControl
from steerability.algorithms.state_control.pasta.args import PASTAArgs
from steerability.algorithms.state_control.pasta.profiling import HeadProfile

if TYPE_CHECKING:
    from steerability.algorithms.state_control.pasta.profiling import HeadProfileResult

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

    The `head_config` argument accepts a dict (layer index to head indices), a list (layer indices,
    all heads), or a `HeadProfile` recipe. A `HeadProfile` moves the paper's one-time head-profiling
    stage into `steer()`: each candidate head is steered on its own on a task-agnostic set of
    profiling prompts, scored by a `SampleScorer`, and ranked by the paired lift over an unsteered
    baseline; the selected heads become the dict head map, and the resolution is available as
    `head_profile` (a `HeadProfileResult`). Profiling runs on the live model through the pipeline's
    session, and the resolved head map freezes into a `.spipe` as a plain-dict PASTA.

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
            "scope": "row",
            "help": (
                "Substrings whose attention should be steered. Required at inference time. A `str` "
                "broadcasts to every row; a `list[list[str]]` of batch length carries one group per "
                "row; a flat `list[str]` is accepted only at batch size 1, as that row's group."
            ),
        },
    ]

    supports_batching: bool = True

    # placeholders
    model: PreTrainedModel | None = None
    tokenizer: PreTrainedTokenizerBase | None = None
    device: torch.device | str | None = None
    head_profile: "HeadProfileResult | None" = None

    _head_map: dict[int, list[int]] | None = None
    _layers: list[int] | None = None
    _attn_module_names: dict[int, str] | None = None
    _num_heads_by_layer: dict[int, int] | None = None
    _layout = None

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
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase | None = None,
        session=None,
        **__,
    ) -> PreTrainedModel:
        """Initialize PASTA by configuring attention head mappings and model references.

        Resolves the attention module path and head count of every attention layer of the model,
        resolves a `HeadProfile` head map when `head_config` is a profiling recipe (steering each
        candidate head through the pipeline's session), sets up the resolved head map, and fails
        fast on unsupported layers or attention implementations (rather than deep inside
        generation).

        Args:
            model (PreTrainedModel): The base language model to be steered.
            tokenizer (PreTrainedTokenizerBase | None): Tokenizer for substring identification.
                If None, attempts to retrieve from model attributes.
            session: The `ScopedSession` the pipeline hands to `steer()`, scoped to
                `ModelAccess.MODULE`, through which a `HeadProfile` runs its rollouts.

        Returns:
            PreTrainedModel: The input model (unchanged).

        Raises:
            ValueError: If a configured attention module path is missing on the model, if the
                model's attention implementation is not one of `"eager"` / `"sdpa"`, or if a
                `HeadProfile` selects no heads.
        """
        self.model = model
        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        self.device = next(model.parameters()).device
        self._resolve_attention_modules(model)
        self._check_attention_implementation(model)

        if isinstance(self.head_config, HeadProfile):
            self.head_profile = self.head_config.resolve(self, model, self.tokenizer, session=session)
            head_map = self.head_profile.head_config
        else:
            head_map = self.head_config

        self._setup_layers(head_map)
        self._finalize_head_map()
        return model

    def _resolve_attention_modules(self, model: PreTrainedModel) -> None:
        """Resolve every attention layer's module path and head count from the model layout.

        The per-layer attention module paths come from `resolve_model_layout` (`.self_attn` for
        `model.layers.*`, `.attn` for `transformer.h.*`). Every attention layer of the layout is
        resolved (not only the configured layers), so a `HeadProfile` can enumerate candidates
        from the same table `get_hooks` reads. Each layer's head count is read per layer from the
        module tree, so a model whose head count varies across layers is sized per layer.

        Raises:
            ValueError: If the architecture is unrecognized, or an attention module path is
                absent.
        """
        layout = resolve_model_layout(model)
        self._layout = layout
        attn_names = layout.attn_names

        self._attn_module_names = {}
        self._num_heads_by_layer = {}
        for layer in layout.attention_layers:
            path = attn_names[layer]
            try:
                model.get_submodule(path)
            except AttributeError as error:
                raise ValueError(f"PASTA could not resolve attention module {path!r}.") from error
            self._attn_module_names[layer] = path
            self._num_heads_by_layer[layer] = head_geometry(model, layout, layer).num_heads

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
        during the forward pass, at the configured head map and the control's `alpha`.

        Args:
            input_ids (torch.Tensor): Input token IDs of shape [batch_size, seq_len].
            runtime_kwargs (dict | None): Must contain "substrings" key with target text spans:

                - str: one substring, broadcast to every batch item
                - list[list[str]]: one substring group per batch item, of batch length
                - list[str]: accepted only at batch size 1, as that row's group

        Returns:
            dict[str, list]: Hook specifications with "pre", "forward", "backward" keys. Only "pre" hooks are populated for attention modification.

        Raises:
            ValueError: If "substrings" is missing from runtime_kwargs, a flat list of strings is
                passed at batch size > 1, a group is a `str` or contains a non-`str` element, or
                the number of groups does not match the batch size.
        """
        if not runtime_kwargs or "substrings" not in runtime_kwargs:
            raise ValueError("PASTA requires 'substrings' inside runtime_kwargs")

        token_ranges, input_len = self.locate_spans(input_ids, runtime_kwargs["substrings"])
        return self.build_hooks_for(token_ranges, input_len, self._head_map, self.alpha)

    def locate_spans(self, input_ids: torch.Tensor, substrings) -> tuple[list[torch.Tensor], int]:
        """Locate the `substrings` spans in `input_ids`, in the tensor's own coordinates.

        Runs the substring-to-token-range resolution once for a prepared batch, so a caller
        steering many head maps over one batch (head profiling) locates the spans once and only
        reassembles the hook dict per candidate. The returned ranges are in the coordinate system
        of `input_ids` (its key axis includes any BOS/template and left-pad tokens).

        Args:
            input_ids: Prompt token ids of shape `[batch_size, seq_len]`.
            substrings: The `substrings` runtime-kwarg value, in any of PASTA's accepted forms.

        Returns:
            A `(token_ranges, input_len)` pair, where `token_ranges` is one `[G, 2]` tensor of
            `(start, end)` spans per row and `input_len` is the sequence length.

        Raises:
            ValueError: If `substrings` is malformed (see `_normalize_substrings`).
        """
        batch_size = input_ids.size(0)
        groups = self._normalize_substrings(substrings, batch_size)

        # decode *with* special tokens so offsets share the attention mask's coordinate system
        # (its key axis includes BOS/template tokens)
        prompts = self.tokenizer.decode(input_ids, skip_special_tokens=False)

        # round-trip the substrings through the tokenizer so they match the decoded text
        # (tokenization can subtly change text, e.g. drop spaces)
        for group_idx, group in enumerate(groups):
            try:
                groups[group_idx] = self.tokenizer.decode(
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
        return token_ranges, input_len

    def build_hooks_for(
        self,
        token_ranges: list[torch.Tensor],
        input_len: int,
        head_map: dict[int, list[int]],
        alpha: float,
    ) -> dict[str, list]:
        """Assemble the pre-hook dict for a head map at a strength, from located spans.

        The log-alpha constant is bound into each pre-hook partial, so a profiling strength and
        the control's own `alpha` never cross. Only `"pre"` hooks are populated.

        Args:
            token_ranges: Located spans, one `[G, 2]` tensor per row (from `locate_spans`).
            input_len: The sequence length the ranges were located in.
            head_map: Layer index to head indices to steer.
            alpha: The emphasis strength for this hook set.

        Returns:
            Hook specifications with `"pre"`, `"forward"`, `"backward"` keys.
        """
        scale_constant = torch.tensor([alpha], device=self.device, dtype=torch.float32).log()
        hooks: dict[str, list] = {"pre": [], "forward": [], "backward": []}
        for layer, heads in head_map.items():
            hooks["pre"].append(
                {
                    "module": self._attn_module_names[layer],
                    "hook_func": partial(
                        self._attention_pre_hook,
                        head_idx=heads,
                        token_ranges=token_ranges,
                        input_len=input_len,
                        layer_idx=layer,
                        scale_constant=scale_constant,
                    ),
                }
            )
        return hooks

    def attention_layers(self) -> list[int]:
        """The model's attention layers, ascending (available after `steer()`)."""
        return sorted(self._num_heads_by_layer)

    def num_heads_of_layer(self, layer: int) -> int:
        """The attention head count of `layer` (available after `steer()`)."""
        return self._num_heads_by_layer[layer]

    def steer_fits(self) -> tuple[tuple[str, str], ...]:
        """The fit artifacts the steer step produces, for the steer plan.

        A `HeadProfile` head map produces one `("HeadProfile", "direction")` fit (the lift grid,
        classed as a translation-robust direction). A dict or list head map produces no fit. This
        is a pure function of the args, so `check()` can read it before `steer()`.
        """
        if isinstance(self.head_config, HeadProfile):
            return (("HeadProfile", "direction"),)
        return ()

    def export_state(self) -> dict:
        """The resolved profile's lift grid under `"head_profile"` (after a profiled `steer()`).

        Empty for a dict or list head map, or before `steer()` resolves a `HeadProfile`. The full
        `HeadProfileResult` is not serialized here; it stays on `head_profile` and the caller
        writes it through `HeadProfileResult.save`.
        """
        if self.head_profile is not None:
            return {"head_profile": self.head_profile.lift}
        return {}

    def frozen_form(self, state: dict) -> tuple[str, dict]:
        """A same-class `state_control/pasta` frozen form carrying the resolved dict head map.

        Only used when `head_config` is a `HeadProfile`; a dict or list head map keeps the
        `BaseControl` default (the recipe is the frozen form). The loaded control is a plain-dict
        PASTA whose `steer()` does no rollouts.
        """
        if isinstance(self.head_config, HeadProfile):
            return "state_control/pasta", {
                "substrings": self.substrings,
                "head_config": self.head_profile.head_config,
                "alpha": self.alpha,
                "scale_position": self.scale_position,
            }
        return super().frozen_form(state)

    def fit_identity(self):
        """The fit-relevant recipe inputs for staleness detection, or None.

        For a `HeadProfile` head map, the profile's `fit_ingredients()` together with the
        control's `scale_position` (the profile is scored under it). None for a dict or list head
        map. The control's own `alpha` is an application parameter and is excluded.
        """
        if isinstance(self.head_config, HeadProfile):
            return {"profile": self.head_config.fit_ingredients(), "scale_position": self.scale_position}
        return None

    @staticmethod
    def _normalize_substrings(substrings, batch_size: int) -> list[list[str]]:
        """Normalize the `substrings` runtime kwarg to one group per batch row, in a local copy.

        Accepted forms are a `str` (one substring, broadcast to every row), a `list[list[str]]`
        (one group per row, of batch length), and a flat `list[str]` (accepted only at batch
        size 1, as that row's group). Every group must be a non-`str` sequence of `str`; a `str`
        where a group is expected raises rather than being iterated character by character.

        Args:
            substrings: The runtime-kwarg value to normalize.
            batch_size: Number of prompt rows.

        Returns:
            One list of substrings per batch row.

        Raises:
            ValueError: If a flat `list[str]` is passed at batch size > 1 (the message names the
                accepted forms and the `[[...]] * batch_size` broadcast workaround), the number of
                groups does not match the batch size, or any group is a `str` or contains a
                non-`str` element.
        """
        if isinstance(substrings, str):
            return [[substrings] for _ in range(batch_size)]
        if not isinstance(substrings, Sequence):
            raise ValueError(
                f"PASTA 'substrings' must be a str, a list[list[str]] of batch length, or a flat "
                f"list[str] at batch size 1; got {type(substrings).__name__}."
            )
        substrings = list(substrings)
        if all(isinstance(element, str) for element in substrings):
            if batch_size > 1:
                raise ValueError(
                    f"PASTA received a flat list[str] for 'substrings' with batch size {batch_size}. "
                    "Accepted forms are a str (broadcast to every row) and a list[list[str]] with one "
                    "group per row; to broadcast one group over the batch, pass "
                    "[[...]] * batch_size."
                )
            return [substrings]
        if len(substrings) != batch_size:
            raise ValueError(
                f"Need {batch_size} substring groups (one per prompt); got {len(substrings)}"
            )
        groups: list[list[str]] = []
        for group in substrings:
            if isinstance(group, str) or not isinstance(group, Sequence):
                raise ValueError(
                    f"PASTA substring groups must be non-str sequences of str; got "
                    f"{type(group).__name__} for one row."
                )
            group = list(group)
            invalid = [element for element in group if not isinstance(element, str)]
            if invalid:
                raise ValueError(
                    f"PASTA substring groups must contain only str elements; got "
                    f"{type(invalid[0]).__name__} in one row's group."
                )
            groups.append(group)
        return groups

    def _setup_layers(self, head_config) -> None:
        """Derive the configured layer set (and explicit head lists for the dict form).

        The head map is finalized in `_finalize_head_map` once the per-layer head counts are
        known; the list form defers to that step for its all-heads default.

        Args:
            head_config: Configuration specifying which layers/heads to modify:

                - dict: Maps layer indices to lists of head indices
                - list: Layer indices (applies to all heads in those layers)

        Raises:
            ValueError: If the configuration format is invalid.
        """
        if isinstance(head_config, dict):
            self._head_map = {int(layer): list(heads) for layer, heads in head_config.items()}
            self._layers = sorted(self._head_map.keys())
        elif isinstance(head_config, list):
            self._layers = [int(layer) for layer in head_config]
            self._head_map = None
        else:
            raise ValueError(f"Invalid head configuration: {head_config!r}")

    def _finalize_head_map(self) -> None:
        """Validate the configured layers, fill the all-heads default, and check head indices.

        Every configured layer must be an attention layer of the model (its head count is in
        `_num_heads_by_layer`, recorded in `_resolve_attention_modules`); a layer out of range or
        without an attention module raises. The list form of `head_config` then expands to every
        head of each configured layer, and the dict form's explicit head lists are validated,
        both against that layer's own head count. Models whose head count varies across layers are
        handled per layer.

        Raises:
            ValueError: If a configured layer is out of range or carries no attention module, or a
                head index is out of range for its layer.
        """
        num_layers = self._layout.num_layers
        for layer in self._layers:
            if layer < 0 or layer >= num_layers:
                raise ValueError(
                    f"PASTA layer {layer} out of range for model with {num_layers} layers."
                )
            if layer not in self._num_heads_by_layer:
                raise ValueError(
                    f"PASTA layer {layer} carries no attention module; attention layers of this "
                    f"model are {list(self._layout.attention_layers)}."
                )

        if self._head_map is None:  # list form: all heads of each configured layer
            self._head_map = {
                layer: list(range(self._num_heads_by_layer[layer])) for layer in self._layers
            }

        for layer, heads in self._head_map.items():
            num_heads = self._num_heads_by_layer[layer]
            for head in heads:
                if not 0 <= head < num_heads:
                    raise ValueError(
                        f"Head {head} out of range for layer {layer} (0-{num_heads - 1})"
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
        scale_constant: torch.Tensor,
        layer_idx: int = 0,
    ):
        """Modify attention mask to steer focus toward/away from target tokens.

        Pre-forward hook that adjusts attention weights by adding scaling factors to the attention mask for specified token ranges and attention heads.

        In `"include"` mode each prompt column is edited once: the highlighted span columns are
        left untouched and `scale_constant` is subtracted from the complement (the non-span prompt
        columns), the paper's Eq. 3 (non-highlighted prompt columns scaled by the coefficient).
        Overlapping spans therefore net to zero, and no column is touched twice, which removes the
        bfloat16 residue of a double touch. `"exclude"` and `"generation"` add `scale_constant` to
        the non-span columns and the whole prompt respectively.

        Args:
            module: The attention module being hooked.
            input_args: Positional arguments to the forward pass.
            input_kwargs: Keyword arguments to the forward pass.
            head_idx: List of attention head indices to modify.
            token_ranges: Token index ranges to apply scaling to.
            input_len: Length of input sequence (for generation positioning).
            scale_constant: The `log(alpha)` edit magnitude, bound into the hook so a profiling
                strength and the control's own `alpha` cannot cross.
            layer_idx: Decoder layer index of the hooked attention module, used to read this
                layer's cached key length when `cache_position` is absent (transformers v5).

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
            num_heads = self._num_heads_by_layer[layer_idx]

            # during decoding the query attends to the full kv cache, so the mask spans the cached key
            # positions rather than just the current query window. transformers v4 exposes the key axis
            # via cache_position; v5 removed that kwarg from attention calls, so fall back to this
            # layer's cache length (the pre-hook runs before the layer's cache update, so cached length
            # plus query_len is the post-update key length). an undersized mask would rely on sdpa
            # broadcasting, and the cuda mem-efficient kernel rejects the stride-0 last dimension that
            # its mask expansion produces ("(*bias): last dimension must be contiguous")
            cache_position = input_kwargs.get("cache_position")
            past_key_values = input_kwargs.get("past_key_values")
            if cache_position is not None:
                key_len = int(cache_position[-1]) + 1
            elif past_key_values is not None and callable(getattr(past_key_values, "get_seq_length", None)):
                key_len = int(past_key_values.get_seq_length(layer_idx) or 0) + query_len
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

        if attention_mask.dtype == torch.bool:
            # transformers v5 materializes boolean masks (true = attend) for sdpa; convert to the
            # additive convention before applying the log-alpha scaling edits
            attention_mask = torch.where(
                attention_mask,
                hidden_states.new_zeros(()),
                hidden_states.new_full((), torch.finfo(hidden_states.dtype).min),
            )
        attention_mask = attention_mask.to(hidden_states.dtype).contiguous().clone()
        if attention_mask.size(1) == 1:
            num_heads = self._num_heads_by_layer[layer_idx]
            attention_mask = attention_mask.expand(-1, num_heads, -1, -1).contiguous()

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

        if self.scale_position not in ("include", "exclude", "generation"):
            raise ValueError(f"Unknown scale_position '{self.scale_position}'")

        for batch_index in range(batch_size):
            ranges = token_ranges[batch_index].tolist()
            has_valid_range = any(start != end for start, end in ranges)
            if not has_valid_range:
                continue

            if self.scale_position == "include":
                # edit each prompt column once: subtract on the complement of the highlighted
                # span union, leaving the span columns untouched (overlapping spans net to zero).
                # a per-column delta applied over the [:input_len] slice keeps the write a single
                # subscription (advanced head indexing writes back only through one __setitem__)
                delta = attention_mask.new_full((input_len,), 0.0)
                delta[:] = -scale_constant
                for start_idx, end_idx in ranges:
                    if start_idx != end_idx:
                        delta[start_idx:end_idx] = 0.0
                attention_mask[batch_index, head_idx, :, :input_len] += delta
            elif self.scale_position == "exclude":
                for start_idx, end_idx in ranges:
                    if start_idx == end_idx:
                        continue
                    attention_mask[batch_index, head_idx, :, :start_idx] += scale_constant
                    attention_mask[batch_index, head_idx, :, end_idx:input_len] += scale_constant
            else:  # generation
                attention_mask[batch_index, head_idx, :, :input_len] += scale_constant

        input_kwargs["attention_mask"] = attention_mask
        return input_args, input_kwargs
