"""
Few-shot learning control for prompt adaptation.
"""
import warnings
from typing import Any, Sequence

import torch
from transformers import PreTrainedTokenizer

from aisteer360.algorithms.input_control.base import InputControl
from aisteer360.algorithms.input_control.common.formatters.few_shot_block import FewShotBlockFormatter
from aisteer360.algorithms.input_control.common.memory.pool import PoolMemory
from aisteer360.algorithms.input_control.common.memory.text import TextMemory
from aisteer360.algorithms.input_control.common.selectors.base import BaseSelector
from aisteer360.algorithms.input_control.few_shot.args import FewShotArgs
from aisteer360.algorithms.input_control.few_shot.selectors import selector_from_arg
from aisteer360.utils.rendering import has_chat_template, render_messages


class FewShot(InputControl):
    """
    Implementation of few-shot learning control for prompt adaptation.

    FewShot enables selective behavioral steering by prepending specific examples to user prompts, guiding model
    responses through demonstration.

    The method operates in two modes:

    1. **Pool-based sampling**: Maintains pools of positive and negative examples from which k examples are dynamically
        selected using configurable sampling strategies (random, semantic similarity, etc.).

    2. **Runtime injection**: Accepts examples directly at inference time through runtime_kwargs, enabling
        context-specific demonstrations without predefined pools. Useful for dynamic or user-provided examples.

    The selected examples are formatted into a system prompt with clear positive/negative labels and prepended to the
    user query using the model's chat template, allowing the model to learn the desired behavior pattern from the
    demonstrations.

    Args:
        directive (str, optional): Instruction text that precedes the examples, explaining the task or desired behavior.
            Defaults to None.
        positive_example_pool (Sequence[dict], optional): Pool of positive examples demonstrating desired behavior.
            Each dict can contain multiple key-value pairs. Defaults to None.
        negative_example_pool (Sequence[dict], optional): Pool of negative examples showing undesired behavior to avoid.
            Each dict can contain multiple key-value pairs. Defaults to None.
        k_positive (int, optional): Number of positive examples to sample from the pool per query.
            Defaults to None.
        k_negative (int, optional): Number of negative examples to sample from the pool per query.
            Defaults to None.
        selector (BaseSelector | str | None, optional): How examples are picked from pools. Accepts a
            `BaseSelector` instance, a registry name (e.g. `"random"`), or `None` (defaults to
            `RandomSelector`). Defaults to None.
        formatter (BaseFormatter, optional): Formatter that renders the example block into the chat or
            token stream. Defaults to `FewShotBlockFormatter()` when not supplied.

    Runtime keyword arguments:

    - `positive_examples` (`list[dict]`, `optional`): Positive examples to use for this specific query (overrides pool-based selection).
    - `negative_examples` (`list[dict]`, `optional`): Negative examples to use for this specific query (overrides pool-based selection).

    Notes:

    - Requires a tokenizer with chat_template support for optimal formatting
    - Examples are automatically labeled as "### Positive example" or "### Negative example"
    - When both pools and runtime examples are available, runtime examples take precedence
    - If no examples are provided, the original input is returned unchanged
    - Keys with a leading underscore in example dicts are reserved by the framework (e.g. `_polarity`);
        user fields should not start with `_`.
    """

    Args = FewShotArgs

    supports_batching: bool = True

    # placeholders (dataclass attrs from FewShotArgs override these at __init__ time)
    tokenizer: PreTrainedTokenizer | None = None
    directive: str | None = None
    positive_example_pool: Sequence[dict] | None = None
    negative_example_pool: Sequence[dict] | None = None
    k_positive: int | None = None
    k_negative: int | None = None
    selector: Any = None  # str | BaseSelector | None — resolved in steer()
    formatter: Any = None  # BaseFormatter | None — resolved in steer()

    # method-owned state populated in steer()
    pool: PoolMemory[dict] | None = None
    _selector: BaseSelector | None = None
    _formatter: FewShotBlockFormatter | None = None

    def steer(
            self,
            model=None,
            tokenizer: PreTrainedTokenizer | None = None,
            **kwargs,
    ) -> None:
        self.tokenizer = tokenizer

        # build the example pool
        self.pool = PoolMemory[dict]()
        for example in self.positive_example_pool or []:
            self.pool.add(example, polarity="pos")
        for example in self.negative_example_pool or []:
            self.pool.add(example, polarity="neg")

        # resolve selector argument (instance | name | None) into a BaseSelector[dict]
        self._selector = selector_from_arg(self.selector)

        # selectors that need offline preparation (e.g. EPR) get a chance here
        prepare = getattr(self._selector, "prepare", None)
        if callable(prepare):
            prepare(model=model, tokenizer=tokenizer, data=self.pool)

        # formatter is shared between adapt and adapt_messages
        self._formatter = self.formatter or FewShotBlockFormatter()

    def adapt(
        self,
        input_ids: list[int] | torch.Tensor,
        runtime_kwargs: dict | None = None,
    ) -> list[int] | torch.Tensor:
        """Add few-shot examples to the model's prompt and return adapted token ids.

        Both the chat-template path and the no-template fallback route through the resolved
        `BaseFormatter`, which renders the example block from the pool.

        Assumes `input_ids` represents the user's prompt before any chat templating. Pre-templated
        input will be re-templated and produce malformed output; use `adapt_messages` for chat input.

        Args:
            input_ids: The user's prompt token IDs.
            runtime_kwargs: May contain `positive_examples` / `negative_examples` to override pool sampling.

        Returns:
            The transformed token IDs.

        Raises:
            RuntimeError: If tokenizer is not set (requires calling `steer()` first)

        Warnings:
            UserWarning: Issued when no examples are configured or none remain after selection.
        """
        if self.tokenizer is None:
            raise RuntimeError("FewShot needs a tokenizer; call .steer() first.")

        # infer mode from arguments
        using_runtime_examples = (
            runtime_kwargs
            and ("positive_examples" in runtime_kwargs or "negative_examples" in runtime_kwargs)
        )
        using_pool_mode = self.positive_example_pool is not None or self.negative_example_pool is not None

        using_directive = bool(self.directive)
        if not (using_runtime_examples or using_pool_mode or using_directive):
            warnings.warn(
                "FewShot: nothing to inject (no examples, no directive). Returning input unchanged.",
                UserWarning,
            )
            return input_ids

        # determine input format
        is_tensor = isinstance(input_ids, torch.Tensor)
        original_device = input_ids.device if is_tensor else None
        original_dtype = input_ids.dtype if is_tensor else None

        # normalize to 2D list format [batch_size, seq_len]
        if is_tensor:
            if input_ids.ndim == 1:
                batch_input_ids = [input_ids.tolist()]
                single_sequence = True
            else:
                batch_input_ids = input_ids.tolist()
                single_sequence = False
        else:
            if isinstance(input_ids[0], int):
                batch_input_ids = [input_ids]
                single_sequence = True
            else:
                batch_input_ids = input_ids
                single_sequence = False

        use_chat_template = has_chat_template(self.tokenizer)

        adapted_batch: list[list[int]] = []
        for input_ids_single in batch_input_ids:
            original_text = self.tokenizer.decode(input_ids_single, skip_special_tokens=True)

            # sample or gather examples independently per item
            if using_runtime_examples:
                examples = self._gather_runtime_examples(runtime_kwargs)
            else:
                examples = self._sample_from_pools(query=original_text)

            if not examples and not self.directive:
                warnings.warn(
                    "FewShot: nothing to inject for this item. Returning input unchanged.",
                    UserWarning,
                )
                adapted_batch.append(list(input_ids_single))
                continue

            slot_memory = TextMemory(slots={
                "examples": examples,
                "directive": self.directive or "",
            })

            if use_chat_template:
                chat = [{"role": "user", "content": original_text}]
                adapted_chat = self._formatter.apply_to_messages([chat], slot_memory)[0]
                rendered = render_messages(self.tokenizer, adapted_chat, add_generation_prompt=True)
                adapted_tokens = self.tokenizer(rendered, add_special_tokens=False)["input_ids"]
            else:
                input_tensor = torch.tensor(input_ids_single, dtype=torch.long).unsqueeze(0)
                adapted_tensor = self._formatter.apply_to_ids(input_tensor, slot_memory, self.tokenizer)
                adapted_tokens = adapted_tensor[0].tolist()

            adapted_batch.append(adapted_tokens)

        # pad to uniform length for batched output
        max_len = max(len(seq) for seq in adapted_batch)
        if self.tokenizer.pad_token_id is None:
            raise RuntimeError(
                "FewShot: tokenizer has no pad_token_id; cannot pad batch sequences. "
                "Set a pad token before using FewShot with batched inputs."
            )
        pad_id = self.tokenizer.pad_token_id

        padded_batch = [seq + [pad_id] * (max_len - len(seq)) for seq in adapted_batch]

        # convert back to original format
        if is_tensor:
            result = torch.tensor(padded_batch, dtype=original_dtype, device=original_device)
            if single_sequence:
                result = result.squeeze(0)
            return result
        else:
            if single_sequence:
                return padded_batch[0]
            return padded_batch

    def _select_with_polarity(self, polarity: str, k: int, query: Any = None) -> list[dict]:
        """Run the resolved selector against the pool subset matching `polarity`."""
        if self.pool is None or self._selector is None:
            return []
        polarities = self.pool.metadata.get("polarity", [])
        items = [item for item, pol in zip(self.pool.items, polarities) if pol == polarity]
        if not items or k <= 0:
            return []
        return self._selector.select(items, query=query, k=k)

    def _sample_from_pools(self, query: Any = None) -> list[dict[str, Any]]:
        """Sample examples from the pools, attaching polarity labels for downstream formatting."""
        all_examples: list[dict[str, Any]] = []
        if self.positive_example_pool and self.k_positive and self.k_positive > 0:
            for example in self._select_with_polarity("pos", self.k_positive, query=query):
                all_examples.append({**example, "_polarity": "positive"})
        if self.negative_example_pool and self.k_negative and self.k_negative > 0:
            for example in self._select_with_polarity("neg", self.k_negative, query=query):
                all_examples.append({**example, "_polarity": "negative"})
        return all_examples

    def adapt_messages(
        self,
        messages: list[list[dict]],
        runtime_kwargs: dict | None = None,
    ) -> list[list[dict]] | None:
        """Insert a single system message containing the directive and labeled example blocks.

        Runtime examples (`positive_examples` / `negative_examples` in `runtime_kwargs`) take precedence
        over pool-based selection. If there are no examples and no directive, returns None (no change).
        """
        runtime_kwargs = runtime_kwargs or {}
        using_runtime = "positive_examples" in runtime_kwargs or "negative_examples" in runtime_kwargs
        using_pools = self.positive_example_pool is not None or self.negative_example_pool is not None
        if not (using_runtime or using_pools or self.directive):
            return None

        out: list[list[dict]] = []
        for chat in messages:
            if using_runtime:
                examples = self._gather_runtime_examples(runtime_kwargs)
            else:
                examples = self._sample_from_pools(query=chat)
            if not examples and not self.directive:
                out.append(list(chat))
                continue
            slot_memory = TextMemory(slots={
                "examples": examples,
                "directive": self.directive or "",
            })
            adapted_batch = self._formatter.apply_to_messages([chat], slot_memory)
            out.append(adapted_batch[0])
        return out

    @staticmethod
    def _gather_runtime_examples(runtime_kwargs: dict[str, Any]) -> list[dict[str, Any]]:
        """Gather examples from runtime_kwargs."""
        examples = []
        if "positive_examples" in runtime_kwargs:
            for example in runtime_kwargs["positive_examples"]:
                examples.append({**example, "_polarity": "positive"})
        if "negative_examples" in runtime_kwargs:
            for example in runtime_kwargs["negative_examples"]:
                examples.append({**example, "_polarity": "negative"})
        return examples
