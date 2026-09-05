"""Model access declarations and the steer plan.

`ModelAccess` names what a control's steer step requires of the pipeline model. The pipeline
satisfies every declaration deterministically: the bottom two rungs through the backend's
session, `CAPTURE` through session capture where advertised and a staged in-process model where
not, and `MODULE` through a staged in-process model always. `SteerPlan` records, per enabled
control and per fit artifact, where the pipeline will run each steer step for one backend
configuration.
"""
import enum
from dataclasses import dataclass
from typing import Literal


class ModelAccess(enum.IntEnum):
    """What a control's steer step requires of the pipeline model, as a cumulative ladder.

    Attributes:
        FACTS: Structural facts (`session.layout`) and a tokenizer.
        ROLLOUTS: FACTS plus generation and scoring through the session.
        CAPTURE: ROLLOUTS plus hidden-state capture through the session.
        MODULE: The model as a live `torch.nn.Module` in the client process.
    """

    FACTS = 0
    ROLLOUTS = 1
    CAPTURE = 2
    MODULE = 3


Venue = Literal["live", "session", "stage"]
ArtifactClass = Literal["direction", "calibrated"]


@dataclass(frozen=True, slots=True)
class PlannedStep:
    """One enabled control's steer step in the plan.

    Attributes:
        control: Class name of the control.
        access: The control's declared steer access.
        venue: Where the step runs. `"live"` is the persistent in-process model on the Hugging
            Face backend, `"session"` the engine session, and `"stage"` the temporary
            in-process model on engine backends.
    """

    control: str
    access: ModelAccess
    venue: Venue


@dataclass(frozen=True, slots=True)
class PlannedFit:
    """One fit artifact's venue in the plan.

    Fits execute inside their owning control's steer step, so a fit's venue is the step's.

    Attributes:
        control: Class name of the control whose steer runs the fit.
        artifact: Class name of the fit source or recipe, e.g. `"ContrastiveFit"`.
        artifact_class: `"direction"` for translation-robust fits (mean-difference or PCA
            directions) or `"calibrated"` for artifacts compared against absolute activation
            statistics (probe biases, gate thresholds).
        venue: Where the fit runs.
    """

    control: str
    artifact: str
    artifact_class: ArtifactClass
    venue: Venue


@dataclass(frozen=True, slots=True)
class SteerPlan:
    """The deterministic steer plan for one backend configuration.

    A pure function of the enabled controls' declarations and the backend spec, so the same
    configuration always yields the same plan.

    Attributes:
        steps: Every enabled control, in pipeline order.
        fits: Every fit artifact the steer phase will run, in pipeline order.
        stages: True when a staged in-process model will be constructed.
        notices: Deterministic warnings the steer phase will emit (calibration crossings).
    """

    steps: tuple[PlannedStep, ...] = ()
    fits: tuple[PlannedFit, ...] = ()
    stages: bool = False
    notices: tuple[str, ...] = ()
