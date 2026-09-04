"""Constrained decoding: declarative structured outputs rendered per execution arm."""
from __future__ import annotations

import torch

from aisteer360.algorithms.core.execution.contracts import Capability, ConstraintKinds, Requirements, any_of, needs
from aisteer360.algorithms.core.execution.payloads import ConstraintSource
from aisteer360.algorithms.output_control.base import OutputControl
from aisteer360.algorithms.output_control.common.processors.constraint import ConstraintProcessor

from .args import ConstrainedDecodingArgs


class ConstrainedDecoding(OutputControl):
    """Constrained decoding from one declarative source, rendered per execution arm.

    A declarative constraint (JSON schema, regex, EBNF grammar, or a choice set) renders two
    ways from one source: in process it compiles into a client-side automaton driving a
    `ConstraintProcessor` (masking every logit the grammar forbids), and on vLLM backends it
    renders onto the engine's native structured-output request parameters, consumed in place of
    the live processor. Both arms compile from the same source, so shared cases under greedy
    decode produce identically constrained outputs; the masking implementation (client automaton
    or engine grammar backend) is the documented difference between the arms.

    A control constructed with a live `automaton` object has no declarative form and runs in
    process only. The in-process compilation requires the `xgrammar` optional dependency
    (`aisteer360[guided]`); a vLLM-only pipeline never compiles client-side.

    Structured outputs do not apply to prompt logprobs, so `include_in_scoring=True` requires
    the in-process backend at score; `include_in_scoring=False` opts out of scoring.
    """

    Args = ConstrainedDecodingArgs
    supports_batching = False  # one automaton per processor; the allowed set applies batch-wide

    def _configure(self) -> None:
        self.tokenizer = None
        self._compiled_automaton = None

    def requirements(self) -> Requirements:
        """In-process compilation or engine-native structured outputs at generate."""
        score = needs(Capability.IN_PROCESS_TORCH) if self.include_in_scoring else ()
        if self.source is None:
            return Requirements(
                generate=needs(
                    Capability.IN_PROCESS_TORCH,
                    hint=(
                        "a live automaton object has no declarative form; construct the control "
                        "with a ConstraintSource (or json_schema/regex/grammar/choice) or run "
                        "this pipeline on the huggingface backend"
                    ),
                ),
                score=score,
            )
        return Requirements(
            generate=any_of(
                needs(Capability.IN_PROCESS_TORCH),
                needs(
                    Capability.GUIDED_DECODING,
                    kinds=ConstraintKinds(constraints=frozenset({self.source.kind})),
                ),
            ),
            score=score,
        )

    def export_constraint(self, runtime_kwargs: dict | None = None) -> ConstraintSource | None:
        """The declarative source, or None for automaton-object configurations."""
        return self.source

    def _automaton(self):
        if self.automaton is not None:
            return self.automaton
        if self._compiled_automaton is None:
            if self.tokenizer is None:
                raise RuntimeError(
                    "ConstrainedDecoding requires a tokenizer to compile its constraint; "
                    "steer() must run first."
                )
            from .utils.automaton import compile_constraint_automaton

            self._compiled_automaton = compile_constraint_automaton(self.source, self.tokenizer)
        return self._compiled_automaton

    def get_logits_processors(
        self,
        input_ids: torch.Tensor,
        runtime_kwargs: dict | None,
        **kwargs,
    ) -> list:
        """The in-process arm: one `ConstraintProcessor` over the (compiled) automaton."""
        return [ConstraintProcessor(self._automaton())]
