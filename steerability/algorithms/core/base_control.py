"""Shared base class for steering controls across all four categories."""
import copy
from abc import ABC
from dataclasses import fields
from typing import Any

from steerability.algorithms.core.base_args import BaseArgs
from steerability.algorithms.core.execution.access import ModelAccess
from steerability.algorithms.core.execution.contracts import Capability, Requirements, needs


class NotFreezableError(RuntimeError):
    """A control produces steer-time state but declares no frozen form.

    Raised by `BaseControl.frozen_form` when a control's steer step produces fits or exported
    state and the control does not override `frozen_form` to name a constructor-valid frozen
    form. Re-exported from `steerability.spipe.errors` alongside the other spipe exceptions.
    """


class BaseControl(ABC):
    """Common construction and lifecycle for steering controls.

    Subclasses set `Args` to their hyperparameter dataclass; construction validates the arguments
    and mirrors every field onto the instance. A field whose name the subclass exposes as a
    `@property` is not mirrored (e.g. `CAST.condition_point`); the raw value stays reachable via
    `self.args.<name>`. Controls with `Args = None` accept no constructor arguments.

    Attributes:
        Args: The control's hyperparameter dataclass, or None for arg-free controls.
        RUNTIME_KWARGS_SCHEMA: Declarations for the per-call parameters the pipeline maps onto the
            control at inference time. Each entry is a dict carrying `name` plus optional `type`,
            `required`, `help`, and `scope` fields. `scope` is `"row"` for a per-prompt value (in
            a batched call the control receives a sequence with one element per prompt row, in row
            order, each element one row's value) or `"call"` for one value per `generate` call
            regardless of batch size; an entry without `scope` is `"call"`. The pipeline validates
            the declarations at `steer()` and raises when two enabled controls declare one name
            with a different `scope` or `type`.
        enabled: Whether the control participates in the pipeline (identity controls set False).
        supports_batching: Whether the control processes a batched prompt in one call.
    """

    Args: type[BaseArgs] | None = None
    RUNTIME_KWARGS_SCHEMA: list[dict] = []

    enabled: bool = True
    supports_batching: bool = False

    def __init__(self, *args, **kwargs) -> None:
        if self.Args is None:  # null / arg-free control
            if args or kwargs:
                raise TypeError(f"{type(self).__name__} accepts no constructor arguments.")
            self._configure()
            return

        self.args: BaseArgs = self.Args.validate(*args, **kwargs)

        # move fields to attributes, skipping any name the control exposes as a property
        # (e.g. CAST.condition_point); the raw value stays reachable via self.args.<name>
        for field in fields(self.args):
            if isinstance(getattr(type(self), field.name, None), property):
                continue
            setattr(self, field.name, getattr(self.args, field.name))

        self._configure()

    def _configure(self) -> None:
        """Post-construction hook, called after `Args` fields are mirrored onto the instance.

        Driver presets override this to map their mirrored args onto the fields their generic base
        reads, so subclasses never bypass a parent `__init__`. Default no-op.
        """
        pass

    def requirements(self) -> Requirements:
        """Backend requirements computed from this instance's configuration, per phase.

        The default requires `Capability.IN_PROCESS_TORCH` at generate and nothing at score,
        which only the Hugging Face backend satisfies. A control with portable mechanisms
        overrides this to state weaker or alternative requirements. Configuration determines the
        result, so two configurations of one class may differ. Only enabled controls are
        consulted during support evaluation.

        Returns:
            The control's phase-keyed requirements.
        """
        return Requirements(generate=needs(Capability.IN_PROCESS_TORCH))

    def steer_access(self) -> ModelAccess:
        """The model access this instance's steer step requires, on the `ModelAccess` ladder.

        The default is `ModelAccess.FACTS` (layout and tokenizer only). A control whose steer
        step generates or scores through the session declares `ROLLOUTS`, one that captures
        hidden states declares `CAPTURE`, and one that touches the model as a live
        `torch.nn.Module` declares `MODULE`. Configuration determines the result. The pipeline
        hands `steer()` a session scoped to the declared rung, and the live model only at
        `MODULE`. A control may retain the pipeline model beyond `steer()` only if its
        generate phase requires `Capability.IN_PROCESS_TORCH`.

        Returns:
            The declared access rung.
        """
        return ModelAccess.FACTS

    def steer_fits(self) -> tuple[tuple[str, str], ...]:
        """The fit artifacts this instance's steer step will produce, for the steer plan.

        Each entry is `(artifact, artifact_class)`, where `artifact` is the fit source or
        recipe class name and `artifact_class` is `"direction"` or `"calibrated"`. The default
        is an empty tuple (no fits).

        Returns:
            The declared fit artifacts, in declaration order.
        """
        return ()

    def export_state(self) -> dict[str, Any]:
        """Steer-time state to persist when freezing, keyed by logical name.

        Values are typed artifacts (`SteeringVector`, `Probe`, `ProbeSet`, a `Memory`
        implementation, `CheckpointArtifact`, `LoRAArtifact`, `torch.Tensor`, or a `Path` to an
        on-disk product). Must be called after `steer()`. The default returns an empty mapping
        (nothing to persist).

        Returns:
            Mapping from logical name to artifact value.
        """
        return {}

    def frozen_form(self, state: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """The `(registry method key, constructor kwargs)` of this control's frozen form.

        `state` is the output of `export_state()`; artifact values inside the returned kwargs
        are replaced by store references at encode time. When `steer_fits()` is empty and
        `state` is empty, the recipe is the frozen form and the control's own key and recipe
        args are returned unchanged. A control whose freeze expands into several entries may
        return a list of `(method key, kwargs)` pairs instead.

        Args:
            state: The output of `export_state()`.

        Returns:
            The frozen method key and its constructor kwargs.

        Raises:
            NotFreezableError: If the control produces fits or state but declares no frozen
                form.
        """
        if not state and not self.steer_fits():
            from steerability.algorithms.core.registry import method_key_for

            args = getattr(self, "args", None)
            kwargs = {field.name: getattr(args, field.name) for field in fields(args) if field.init} \
                if args is not None else {}
            return method_key_for(type(self)), kwargs
        raise NotFreezableError(
            f"{type(self).__name__} produces steer-time state but declares no frozen form; "
            "override frozen_form() (and export_state()) to support freezing."
        )

    def fit_identity(self) -> Any | None:
        """The object whose canonical form defines this control's fit digest, or None.

        The digest detects recipe edits that invalidate frozen artifacts, so the returned
        object should cover exactly the inputs a re-fit would consume (training data, fit
        specs) and exclude inert application parameters (strengths, multipliers). The default
        returns None (no fits).

        Returns:
            The fit-identity object, or None.
        """
        return None

    def clone_for_call(self, seed: int | None = None) -> "BaseControl":
        """A configuration-preserving shallow clone for one generation call.

        The clone shares steer-time artifacts (memories, steering vectors, attached tokenizers)
        with the original but has its own attribute namespace, so per-call attribute mutation on
        the clone never races another call using the original. When `seed` is given and the
        control defines `reseed(seed)`, the clone's client-side RNG is re-seeded.

        Args:
            seed: Optional seed forwarded to the clone's `reseed()`.

        Returns:
            The clone.
        """
        clone = copy.copy(self)
        if seed is not None:
            reseed = getattr(clone, "reseed", None)
            if callable(reseed):
                reseed(seed)
        return clone

    def cleanup(self) -> None:
        """Release resources allocated during `steer()`.

        Override in subclasses that allocate GPU memory or other resources during steering. Default
        no-op.
        """
        pass
