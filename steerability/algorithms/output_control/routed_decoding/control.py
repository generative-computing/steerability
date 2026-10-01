"""Decoding driver that routes each row to a response strategy based on probe decisions."""
from __future__ import annotations

import warnings
from dataclasses import replace

import torch
from transformers import LogitsProcessorList, PreTrainedModel, PreTrainedTokenizerBase, StoppingCriteriaList

from steerability.algorithms.core.execution.access import ModelAccess
from steerability.algorithms.core.execution.backend import SteeringSession
from steerability.algorithms.core.execution.contracts import Capability, CaptureKinds, Requirements, any_of, needs
from steerability.algorithms.core.internals.fingerprint import model_fingerprint
from steerability.algorithms.core.internals.probes import ProbeSetFit
from steerability.algorithms.output_control.common.drivers.phased import Fixed, Generated, PhasedDriver

from .actions import Generate, Prefix, Respond
from .args import RoutedDecodingArgs


class RoutedDecoding(PhasedDriver):
    """Decoding driver that routes each prompt to a response strategy based on probe decisions.

    `RoutedDecoding` combines a `ProbeSet` with a `Router` (the `rules` argument). The probe
    set contains named, calibrated probes that are scored together in one read-only forward
    pass. The router contains an ordered list of routes, each a condition over the probe
    names. Routes are evaluated per row, and the first route that matches is used. Decoding
    proceeds in three steps:

    1. **Probe pass**: `ProbeSet.read()` runs one read-only forward pass over the prompt and
       returns each probe's signed score and decision (`score >= 0`) for every row. The probe
       set stores the readings of the most recent pass in `latest`.
    2. **Routing**: `rules.route()` matches each row to the first route its decisions
       satisfy, or to the default action when no route matches.
    3. **Execution**: each row's action is converted to a phase plan. `PhasedDriver` runs the
       plan once per candidate (`num_return_sequences` candidates per row), with
       `max_new_tokens` as the ceiling for each candidate. `Respond(text)` appends the tokens
       of `text` without generating. `Prefix(text)` appends the tokens of `text` and then
       generates. `Generate()` generates without appending text. An action can also be given
       directly as a list of `Fixed` and `Generated` phases. The composed logits processors
       and stopping criteria apply in every generated phase.

    The `probes` argument accepts a fitted `ProbeSet` or a `ProbeSetFit`, which `steer()`
    fits on the model the pipeline provides. `steer()` raises `ValueError` when a fitted
    `ProbeSet` records a model that differs from the pipeline's. With
    `allow_model_mismatch=True`, only the model type must match.

    Behavior transforms from other state controls in the pipeline apply to the probe pass when
    their scope includes prompt positions, and probe scores are measured under that steering.
    On the Hugging Face backend the pass runs on the loaded model inside
    `auxiliary_pass(aligned=True)`, under the generation's hooks, and its capture hooks are
    removed before decoding begins. The condition scorers, gates, and position counters of the
    other controls ignore the pass. On the offline vLLM engine the pass is a capture request
    that carries the generation's intervention spec, i.e., the spec the driver's rollouts carry,
    which contains no conditional gates. A row routed to `Respond` requires one forward pass
    over the prompt and no decode steps. Any other row requires one forward pass over the
    prompt more than the default driver.

    The following `runtime_kwargs` are accepted:

    - `"canned_responses"`: A dict mapping route names to replacement text. It replaces the
      `text` of the `Respond` or `Prefix` action of each listed route for the current call
      only. A key that is not the name of a `Respond` or `Prefix` route is ignored with a
      `UserWarning`.

    Attributes:
        latest_routes: The route name chosen for each row in the most recent `decode()`
            call, with `"default"` for rows that matched no route.
        tokenizer: The tokenizer attached by `steer()`, or None before `steer()` runs.
    """

    Args = RoutedDecodingArgs

    supports_batching: bool = True
    same_model_forwards: bool = True

    RUNTIME_KWARGS_SCHEMA = [
        {
            "name": "canned_responses",
            "type": "dict[str, str]",
            "scope": "call",
            "description": "Per-call override of Respond/Prefix text, keyed by route name.",
        },
    ]

    tokenizer: PreTrainedTokenizerBase | None = None

    def _configure(self) -> None:
        """Initialize the `PhasedDriver` fields that the plan runner reads."""
        self.extract_after = None
        self.tokenizer = None
        self.latest_routes: list[str] = []

    def max_rollouts_per_query(self) -> int | None:
        """Return the largest number of `Generated` phases in any route's plan.

        The default action is included. The bound applies per candidate. The probe pass is a
        read-only capture and does not count as a rollout.

        Returns:
            The bound, or None when an action cannot be converted to a phase plan.
        """
        actions = [route.action for route in self.rules.routes] + [self._default_action()]
        counts = []
        for action in actions:
            try:
                plan = self._lower(action)
            except (TypeError, ValueError):
                return None
            counts.append(sum(isinstance(phase, Generated) for phase in plan))
        return max(counts)

    def requirements(self) -> Requirements:
        """Return the backend capabilities the probe pass needs at generation time.

        The probe pass reads the prompt's hidden states. The backend must either run the
        model in process (`Capability.IN_PROCESS_TORCH`) or return residual captures at layer
        inputs for all tokens (`Capability.HIDDEN_CAPTURE`).
        """
        return Requirements(
            generate=any_of(
                needs(Capability.IN_PROCESS_TORCH),
                needs(
                    Capability.HIDDEN_CAPTURE,
                    kinds=CaptureKinds(
                        kinds=frozenset({"residual"}),
                        locations=frozenset({"layer_input"}),
                        modes=frozenset({"all_tokens"}),
                    ),
                    hint=(
                        "the probe pass needs hidden-state capture, which this backend does not "
                        "return; run on huggingface or the offline vLLM engine"
                    ),
                ),
            ),
        )

    def steer_access(self) -> ModelAccess:
        """Return the model access that `steer()` needs.

        Returns:
            `ModelAccess.CAPTURE` when `probes` is a `ProbeSetFit`, since fitting extracts
            hidden states. `ModelAccess.FACTS` when `probes` is a fitted `ProbeSet`, since the
            identity checks read only the session layout.
        """
        if isinstance(self.probes, ProbeSetFit):
            return ModelAccess.CAPTURE
        return ModelAccess.FACTS

    def steer_fits(self) -> tuple[tuple[str, str], ...]:
        """Return the fits that `steer()` performs.

        Returns:
            One `("ProbeSetFit", "calibrated")` entry when `probes` is a `ProbeSetFit`,
            otherwise an empty tuple.
        """
        if isinstance(self.probes, ProbeSetFit):
            return (("ProbeSetFit", "calibrated"),)
        return ()

    def export_state(self) -> dict:
        """Return the fitted probe set for export.

        Returns:
            A dict with the fitted `ProbeSet` under the `"probes"` key, or an empty dict when
            the probes are not yet fitted.
        """
        if self.probes is not None and not isinstance(self.probes, ProbeSetFit):
            return {"probes": self.probes}
        return {}

    def frozen_form(self, state: dict) -> tuple[str, dict]:
        """Return the frozen form of this control.

        Args:
            state: The state returned by `export_state()`.

        Returns:
            The name `"output_control/routed_decoding"` and the constructor kwargs of the
            frozen control, i.e., the fitted `ProbeSet` from `state`, the configured `rules`,
            and `allow_model_mismatch`.
        """
        return "output_control/routed_decoding", {
            "probes": state["probes"],
            "rules": self.args.rules,
            "allow_model_mismatch": self.allow_model_mismatch,
        }

    def fit_identity(self):
        """Return the `ProbeSetFit` when `probes` was given as one, otherwise None."""
        if isinstance(self.args.probes, ProbeSetFit):
            return self.args.probes
        return None

    def steer(
        self,
        model: PreTrainedModel | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        session: SteeringSession | None = None,
        **__,
    ) -> PreTrainedModel | None:
        """Attach the tokenizer, fit or check the probes, and validate the routing rules.

        A `ProbeSetFit` is fitted on the model or session the pipeline provides (its
        `StatsSpec`, when present, is estimated first). A fitted `ProbeSet` is checked
        against the pipeline's model instead. With a loaded model or an in-process session,
        a probe whose recorded `model_fingerprint` differs from the model's raises
        `ValueError`. On other backends, a probe whose recorded `model_ref` differs from the
        session layout's raises `ValueError`. Probes with no recorded value are exempt, and
        `allow_model_mismatch=True` disables both checks. The probe set's `model_type` must
        also match the model's. On backends without a loaded model, it is compared with the
        layout's `model_type` when both are known. `allow_model_mismatch` does not disable
        the `model_type` check.

        Args:
            model: The pipeline's model, or None on backends without a loaded model.
            tokenizer: Tokenizer used for splicing and padding. If None, `model.tokenizer` is
                used when present.
            session: The `SteeringSession` scoped to this control, provided by the pipeline.

        Returns:
            The input model, unchanged.

        Raises:
            ValueError: If a fitted `ProbeSet` records a model fingerprint, model reference,
                or model type that differs from the pipeline's, or if a route references a
                probe name the set does not define.
        """
        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)

        layout = None
        if model is None and session is not None:
            layout = session.layout

        if isinstance(self.probes, ProbeSetFit):
            self.probes = self.probes.fit(model, self.tokenizer, session=session)
        elif not self.allow_model_mismatch:
            live_fingerprint = None
            if model is not None:
                live_fingerprint = model_fingerprint(model)
            elif layout is not None and getattr(session, "in_process", False):
                live_fingerprint = layout.model_fingerprint
            if live_fingerprint is not None:
                mismatched = [
                    name for name, probe in self.probes.probes.items()
                    if probe.meta.get("model_fingerprint") not in (None, live_fingerprint)
                ]
                if mismatched:
                    raise ValueError(
                        "ProbeSet was fitted on a different model than this pipeline produced. "
                        "Pass a ProbeSetFit for steer-time fitting on the pipeline's final model, "
                        "or set allow_model_mismatch=True."
                    )
            elif layout is not None and layout.model_ref is not None:
                mismatched = [
                    name for name, probe in self.probes.probes.items()
                    if probe.meta.get("model_ref") not in (None, layout.model_ref)
                ]
                if mismatched:
                    raise ValueError(
                        "ProbeSet records a model reference that differs from the one this "
                        f"backend serves ({layout.model_ref!r}). Pass a ProbeSetFit for "
                        "steer-time fitting on the pipeline's final model, or set "
                        "allow_model_mismatch=True."
                    )

        if model is not None or (layout is not None and getattr(session, "in_process", False)):
            live_model_type = (
                getattr(model.config, "model_type", "unknown") if model is not None
                else (layout.model_type or "unknown")
            )
            if self.probes.model_type != live_model_type:
                raise ValueError(
                    f"ProbeSet was fitted on model_type {self.probes.model_type!r} but the "
                    f"pipeline's model is {live_model_type!r}."
                )
        elif (
            layout is not None
            and layout.model_type is not None
            and self.probes.model_type not in ("unknown", layout.model_type)
        ):
            raise ValueError(
                f"ProbeSet was fitted on model_type {self.probes.model_type!r} but this "
                f"backend serves {layout.model_type!r}."
            )

        self.rules.validate_names(set(self.probes.names))
        return model

    def decode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None,
               model: PreTrainedModel | None, logits_processors: LogitsProcessorList,
               stopping_criteria: StoppingCriteriaList, runtime_kwargs: dict | None,
               session: SteeringSession | None = None, **gen_kwargs) -> torch.Tensor:
        """Score the probes on the prompt, route each row, and run the routed phase plans.

        Pad positions are removed from each row (using the attention mask) before its plan
        runs.

        Args:
            input_ids: Prompt token ids of shape `[B, T]`. A 1-D tensor is treated as one
                row.
            attention_mask: Prompt attention mask with the same shape as `input_ids`, or
                None.
            model: The pipeline's model.
            logits_processors: The composed `LogitsProcessorList`, applied in every
                generated phase.
            stopping_criteria: The composed `StoppingCriteriaList`, applied in every
                generated phase.
            runtime_kwargs: Per-call parameters (see the class docstring).
            session: The `SteeringSession` on which the probe pass and the generated phases
                run.
            **gen_kwargs: Generation keyword arguments. `max_new_tokens` bounds each
                candidate, and `num_return_sequences` sets the number of candidates per row.

        Returns:
            A tensor of shape `[B * n, T + L]`, where `n` is `num_return_sequences` and `L`
            is the length of the longest continuation. Rows are ordered by prompt row and
            then by candidate. Each row is the padded prompt followed by one candidate's
            continuation, right-padded.

        Raises:
            RuntimeError: If no tokenizer is attached because `steer()` has not run.
            TypeError: If a matched action cannot be converted to a phase plan.
            ValueError: If no session was provided, if a matched plan contains a `Fixed`
                phase with `replace=True`, or if beam search (`num_beams > 1`) is requested
                with more than one candidate.

        Warns:
            UserWarning: If `canned_responses` contains a key that is not the name of a
                `Respond` or `Prefix` route.
        """
        self._check_ready(session)
        runtime_kwargs = runtime_kwargs or {}
        overrides = runtime_kwargs.get("canned_responses") or {}
        input_ids, attention_mask = self._as_batch(input_ids, attention_mask)

        readings = self.probes.read(model, input_ids, attention_mask, session=session)
        matched = self.rules.route(readings.decisions)
        self.latest_routes = [route.name if route is not None else "default" for route in matched]

        if overrides:
            routes_by_name = {route.name: route for route in self.rules.routes}
            unusable = [
                key for key in overrides
                if key not in routes_by_name
                or not isinstance(routes_by_name[key].action, (Respond, Prefix))
            ]
            if unusable:
                warnings.warn(
                    f"canned_responses keys {unusable} do not name a Respond/Prefix route and "
                    "are ignored.",
                    UserWarning,
                )

        plans = []
        for route in matched:
            if route is not None:
                action = route.action
                if route.name in overrides and isinstance(action, (Respond, Prefix)):
                    action = replace(action, text=overrides[route.name])
            else:
                action = self._default_action()
            plans.append(self._lower(action))

        rows = self._unpadded_rows(input_ids, attention_mask)
        prompt_texts = [self.tokenizer.decode(row, skip_special_tokens=True) for row in rows]
        return self._run_plans(
            input_ids, rows, plans, prompt_texts, [{} for _ in rows], model, session,
            logits_processors, stopping_criteria, gen_kwargs,
        )

    def _default_action(self):
        """Return the action for rows that match no route.

        Returns:
            `rules.default_action`, or `Generate()` when the router sets no default.
        """
        return self.rules.default_action if self.rules.default_action is not None else Generate()

    @staticmethod
    def _lower(action) -> list:
        """Convert an action to a phase plan.

        Args:
            action: A `Respond`, `Prefix`, or `Generate` (or any object with a `plan()`
                method), or a list or tuple of `Fixed` and `Generated` phases.

        Returns:
            The phase plan as a list.

        Raises:
            TypeError: If `action` has no `plan()` method and is not a list or tuple.
            ValueError: If the plan contains a `Fixed` phase with `replace=True`. Routed
                actions append to the row's prompt, and the layout of the returned tensor
                depends on the prompt being kept.
        """
        if hasattr(action, "plan"):
            plan = action.plan()
        elif isinstance(action, (list, tuple)):
            plan = list(action)
        else:
            raise TypeError(
                f"Cannot lower action of type {type(action).__name__} to a phase plan; use "
                "respond()/prefix()/generate() or a list of Fixed/Generated phases."
            )
        if any(isinstance(phase, Fixed) and phase.replace for phase in plan):
            raise ValueError(
                "RoutedDecoding does not support prompt-replacing Fixed phases (replace=True); "
                "routed actions append to the row's prompt."
            )
        return plan
