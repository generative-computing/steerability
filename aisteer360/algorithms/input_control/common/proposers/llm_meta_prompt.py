"""Sample candidates from an LLM via a templated meta-prompt."""
from __future__ import annotations

from typing import Any, Callable

import torch
from transformers import PreTrainedTokenizerBase

from aisteer360.algorithms.input_control.common.proposers.base import BaseProposer
from aisteer360.algorithms.input_control.common.proposers.utils.parsing import parse_whole


class LLMMetaPromptProposer(BaseProposer):
    """Format a meta-prompt with `seed` (and optional `context`), sample `n` responses from an LLM, and
    parse each response into one or more candidates.

    Args:
        llm: A causal language model.
        tokenizer: Tokenizer paired with `llm`.
        meta_prompt_template: Python format string. Always receives `seed`; may also reference any keys
            present in the `context` dict passed to `propose()`.
        parse_fn: Maps a single decoded LLM response to a list of candidates (dropping empties).
            Defaults to `parse_whole` ("treat the whole stripped response as one candidate"). Pass
            `parse_fenced_or_whole` for long instructions (GEPA) or `parse_concise_instruction` for
            short system prompts (CPO / PRewrite). See `proposers.utils.parsing`.
        gen_kwargs: Forwarded to `llm.generate`. The default uses sampling with temperature 0.9 so that
            requesting `n > 1` does not yield identical candidates.
        use_chat_template: Controls whether the rendered meta-prompt is wrapped as a single user turn
            via the tokenizer's chat template before sampling. `None` (default) wraps iff the tokenizer
            has a chat template; `True` wraps iff a template exists (silently raw otherwise); `False`
            never wraps. Base models without a template are unaffected.
        max_attempts: Maximum number of sampling rounds in `propose`.
    """

    def __init__(
        self,
        llm,
        tokenizer: PreTrainedTokenizerBase,
        meta_prompt_template: str,
        parse_fn: Callable[[str], list[Any]] = parse_whole,
        gen_kwargs: dict | None = None,
        use_chat_template: bool | None = None,
        max_attempts: int = 4,
    ) -> None:
        self.llm = llm
        self.tokenizer = tokenizer
        self.meta_prompt_template = meta_prompt_template
        self.parse_fn = parse_fn
        self.gen_kwargs = gen_kwargs or {
            "max_new_tokens": 128,
            "do_sample": True,
            "temperature": 0.9,
            "top_p": 0.95,
        }
        self.use_chat_template = use_chat_template
        self.max_attempts = max_attempts

    def _render_meta_prompt(self, seed: Any, context: dict | None) -> str:
        kwargs: dict[str, Any] = {"seed": seed}
        if context:
            kwargs.update(context)
        return self.meta_prompt_template.format(**kwargs)

    def _chat_mode(self) -> bool:
        template_available = bool(getattr(self.tokenizer, "chat_template", None))
        if self.use_chat_template is None:
            return template_available
        return bool(self.use_chat_template) and template_available

    def _sample_responses(self, rendered: str, n: int) -> list[str]:
        chat_mode = self._chat_mode()
        if chat_mode:
            text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": rendered}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text = rendered
        encoded = self.tokenizer(
            text, return_tensors="pt", add_special_tokens=not chat_mode
        ).to(self.llm.device)
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask")

        gen_kwargs = dict(self.gen_kwargs)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        if pad_id is not None:
            gen_kwargs.setdefault("pad_token_id", pad_id)

        # greedy / beam-1 decoding does not support `num_return_sequences > 1`. In those cases call
        # `generate` sequentially. (Greedy with n > 1 is mostly useful for tests; in production callers
        # pass `do_sample=True` so the parallel path applies.)
        do_sample = gen_kwargs.get("do_sample", False)
        num_beams = gen_kwargs.get("num_beams", 1)
        parallel_ok = do_sample or num_beams > 1

        prompt_len = input_ids.size(1)
        decoded: list[str] = []

        if parallel_ok and n > 1:
            gen_kwargs.setdefault("num_return_sequences", n)
            with torch.no_grad():
                output_ids = self.llm.generate(input_ids, attention_mask=attention_mask, **gen_kwargs)
            decoded.extend(
                self.tokenizer.batch_decode(output_ids[:, prompt_len:], skip_special_tokens=True)
            )

        while len(decoded) < n:
            single_kwargs = dict(gen_kwargs)
            single_kwargs["num_return_sequences"] = 1
            with torch.no_grad():
                output_ids = self.llm.generate(input_ids, attention_mask=attention_mask, **single_kwargs)
            decoded.extend(
                self.tokenizer.batch_decode(output_ids[:, prompt_len:], skip_special_tokens=True)
            )

        return decoded[:n]

    def propose(
        self,
        seed: Any,
        n: int = 1,
        context: dict | None = None,
    ) -> list[Any]:
        rendered = self._render_meta_prompt(seed, context)

        # resample until `n` survive or `max_attempts`; return whatever was collected
        candidates: list[Any] = []
        for _ in range(max(self.max_attempts, 1)):
            if len(candidates) >= n:
                break
            for response in self._sample_responses(rendered, n):
                candidates.extend(self.parse_fn(response))
        return candidates[:n] if len(candidates) >= n else candidates
