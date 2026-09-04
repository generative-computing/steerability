"""RoutedDecoding argument validation."""
from __future__ import annotations

from dataclasses import dataclass

from aisteer360.algorithms.core.base_args import BaseArgs
from aisteer360.algorithms.core.internals.probes import ProbeSet, ProbeSetFit

from .routing import Router


@dataclass
class RoutedDecodingArgs(BaseArgs):
    """Arguments for `RoutedDecoding`.

    Attributes:
        probes: The probes whose decisions drive routing. Either a fitted `ProbeSet`, or a
            `ProbeSetFit` recipe the driver fits at `steer()` time on the model the pipeline
            provides.
        rules: The `Router` mapping probe decisions to actions. Probe names are validated
            here against `probes.names` (available in both forms), so misconfigured routes fail
            at construction, before any model loads.
        allow_model_mismatch: When True, a fitted `ProbeSet` whose recorded model fingerprints
            differ from the pipeline's model is accepted at `steer()` time instead of raising.
    """

    probes: ProbeSet | ProbeSetFit | None = None
    rules: Router | None = None
    allow_model_mismatch: bool = False

    def __post_init__(self):
        if not isinstance(self.probes, (ProbeSet, ProbeSetFit)):
            raise TypeError(
                f"probes must be a ProbeSet or ProbeSetFit instance; got "
                f"{type(self.probes).__name__}."
            )
        if not isinstance(self.rules, Router):
            raise TypeError(
                f"rules must be a Router instance; got "
                f"{type(self.rules).__name__}."
            )
        self.rules.validate_names(set(self.probes.names))
