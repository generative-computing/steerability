"""Shared base class for steering controls across all four categories."""
import copy
from abc import ABC
from dataclasses import fields

from aisteer360.algorithms.core.base_args import BaseArgs
from aisteer360.algorithms.core.execution.access import ModelAccess
from aisteer360.algorithms.core.execution.contracts import Capability, Requirements, needs


class BaseControl(ABC):
    """Common construction and lifecycle for steering controls.

    Subclasses set `Args` to their hyperparameter dataclass; construction validates the arguments
    and mirrors every field onto the instance. A field whose name the subclass exposes as a
    `@property` is not mirrored (e.g. `CAST.condition_point`); the raw value stays reachable via
    `self.args.<name>`. Controls with `Args = None` accept no constructor arguments.

    Attributes:
        Args: The control's hyperparameter dataclass, or None for arg-free controls.
        RUNTIME_KWARGS_SCHEMA: Declarations for the per-call parameters the pipeline maps onto the
            control at inference time.
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

    def clone_for_call(self, seed: int | None = None):
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
