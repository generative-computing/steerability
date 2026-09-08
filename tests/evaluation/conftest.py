"""Shared stubs for the Inspect evaluation-stack tests.

`StubSteeringPipeline` mimics the `SteeringPipeline.generate` surface the provider and collator
consume (batched `messages=`/`text=` dispatch with `return_output=True`, and the bare-conversation
multi-candidate shape), recording every call. Only the surface is mimicked; no model runs.
"""
from typing import Any

import torch

from steerability.algorithms.core.output import Output

CHAT_TEMPLATE = "{% for message in messages %}{{ message['content'] }} {% endfor %}"


class StubTokenizer:
    """Tokenizer stub: optional chat template, pad id 0, and an id-derived batch decode.

    `encode`/`decode` intern each distinct string to a reversible id sequence, so a tag encoded and
    decoded round-trips to itself and reasoning-split resolution treats the tags as ordinary
    (text mode). The batch path (a 2D tensor of `output_ids`) keeps the canned per-row decode.
    """

    def __init__(self, chat_template: str | None = CHAT_TEMPLATE, decode_texts: list[str] | None = None):
        self.chat_template = chat_template
        self.pad_token_id = 0
        self.eos_token_id = 1
        self._decode_texts = decode_texts
        self._intern: dict[str, int] = {}
        self._reverse: dict[int, str] = {}

    def encode(self, text, add_special_tokens=False):
        token_id = self._intern.get(text)
        if token_id is None:
            token_id = 1000 + len(self._intern)
            self._intern[text] = token_id
            self._reverse[token_id] = text
        return [token_id]

    def decode(self, ids, skip_special_tokens=True):
        if isinstance(ids, list) and (not ids or isinstance(ids[0], int)):
            return "".join(self._reverse.get(int(token_id), "") for token_id in ids)
        if self._decode_texts is not None:
            return [self._decode_texts[int(row[0]) % len(self._decode_texts)] for row in ids]
        return [f"row-{int(row[0])}" for row in ids]


class StubControl:
    """Minimal control stand-in carrying only what `runtime_kwargs_schema` reads.

    `runtime_kwargs_schema` consults `enabled` and `RUNTIME_KWARGS_SCHEMA` only, so a plain object
    with those two attributes is enough to give a stub pipeline a declared runtime-kwarg scope.
    """

    enabled = True

    def __init__(self, runtime_kwargs_schema: list[dict]):
        self.RUNTIME_KWARGS_SCHEMA = runtime_kwargs_schema


def make_output(row_ids: list[list[int]], prompt_ids: list[int], reasons: tuple[str | None, ...]) -> Output:
    """One `Output` with the given candidate rows, prompt ids, and per-row finish reasons."""
    return Output(
        output_ids=torch.tensor(row_ids, dtype=torch.long),
        adapted_input_ids=torch.tensor([prompt_ids], dtype=torch.long),
        finish_reason=reasons[0],
        finish_reasons=tuple(reasons),
    )


class StubSteeringPipeline:
    """Recording stand-in for a steered `SteeringPipeline`.

    Attributes:
        calls: One dict per `generate` invocation with the received arguments.
        fail_above_batch_size: When set, a dispatch with more prompts than this raises.
        gate: Optional `threading.Event` the first dispatch waits on before returning.
    """

    def __init__(
        self,
        *,
        tokenizer: StubTokenizer | None = None,
        supports_batching: bool = True,
        controls: tuple = (),
        decode_texts: list[str] | None = None,
    ):
        self._is_steered = True
        self.supports_batching = supports_batching
        self.controls = controls
        self.tokenizer = tokenizer if tokenizer is not None else StubTokenizer(decode_texts=decode_texts)
        self.calls: list[dict[str, Any]] = []
        self.fail_above_batch_size: int | None = None
        self.gate = None
        self._gate_used = False
        self._next_token = 0

    def generate(
        self,
        *,
        messages=None,
        text=None,
        runtime_kwargs=None,
        return_output=True,
        **gen_kwargs,
    ):
        self.calls.append({
            "messages": messages,
            "text": text,
            "runtime_kwargs": runtime_kwargs,
            "gen_kwargs": dict(gen_kwargs),
        })
        if self.gate is not None and not self._gate_used:
            self._gate_used = True
            self.gate.wait(10)
        num_candidates = gen_kwargs.get("n", 1)
        single_conversation = messages is not None and messages and isinstance(messages[0], dict)
        if single_conversation or isinstance(text, str):
            rows = []
            for _ in range(num_candidates):
                rows.append([self._next_token])
                self._next_token += 1
            return make_output(rows, [1, 2], ("eos",) * num_candidates)
        prompts = messages if messages is not None else text
        if self.fail_above_batch_size is not None and len(prompts) > self.fail_above_batch_size:
            raise ValueError(f"stub rejects batches larger than {self.fail_above_batch_size}")
        outputs = []
        for _ in prompts:
            outputs.append(make_output([[self._next_token]], [1, 2], ("eos",)))
            self._next_token += 1
        return outputs
