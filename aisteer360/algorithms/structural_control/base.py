"""Structural control base classes.

This module provides the abstract base class for methods that create persistent changes to the model, either through
weight updates or architectural changes.

Two base classes are provided:

- `StructuralControl`: Base class for all structural control methods.

Structural controls implement steering through model weight or architecture modifications, transforming base parameters
θ to θ', resulting in generations following y ~ p_θ'(x).

Examples of structural controls:

- Fine-tuning (full or parameter-efficient like LoRA)
- Model merging (e.g., via MergeKit)
- Direct Preference Optimization (DPO)
- Adapter layers and modules
- Weight interpolation and averaging

See Also:

- `aisteer360.algorithms.structural_control`: Implementations of structural control methods
- `aisteer360.core.steering_pipeline`: Integration with steering pipeline
"""
from abc import abstractmethod

from transformers import PreTrainedModel, PreTrainedTokenizer

from aisteer360.algorithms.core.base_args import BaseArgs
from aisteer360.algorithms.core.base_control import BaseControl
from aisteer360.algorithms.core.execution.access import ModelAccess
from aisteer360.algorithms.core.execution.contracts import Capability, Requirements, any_of, needs
from aisteer360.algorithms.core.execution.payloads import Artifact


class StructuralControl(BaseControl):
    """Abstract base class for structural control steering methods.

    Modifies model parameters or architecture persistently, returning a new model instance with transformed weights.

    Methods:
        steer(model, tokenizer, **kwargs) -> PreTrainedModel: Training logic (required)
    """

    Args: type[BaseArgs] | None = None
    RUNTIME_KWARGS_SCHEMA: list[dict] = []

    enabled: bool = True
    supports_batching: bool = True

    @abstractmethod
    def steer(
            self,
            model: PreTrainedModel,
            tokenizer: PreTrainedTokenizer = None,
            session=None,
            **kwargs
    ) -> PreTrainedModel:
        """Required steering/preparation.

        `session` is a `SteeringSession` on the steering backend, provided by the pipeline.
        """
        pass

    def artifact_capability(self) -> Capability | None:
        """The serve capability implied by this configuration's steer-time artifact, or None.

        Controls whose `steer()` writes a servable product to disk return
        `Capability.SERVE_CHECKPOINT` for a full-weights checkpoint or `Capability.SERVE_LORA`
        for an adapter, so the generate phase gains a serving alternative. The default returns
        None (no on-disk artifact), which keeps the generate phase in-process only.

        Returns:
            The capability, or None.
        """
        return None

    def export_artifact(self) -> Artifact | None:
        """The steer-time artifact this control produced, or None.

        Called by the pipeline after `steer()` completes. The returned artifact must exist on
        disk and correspond to `artifact_capability()` (a `CheckpointArtifact` for
        `Capability.SERVE_CHECKPOINT`, a `LoRAArtifact` for `Capability.SERVE_LORA`). The
        default returns None.

        Returns:
            The artifact, or None.
        """
        return None

    def requirements(self) -> Requirements:
        """Backend requirements computed from this instance's configuration, per phase.

        The generate phase requires `Capability.IN_PROCESS_TORCH` for in-process adoption of
        the returned model; when the configuration produces an on-disk artifact
        (`artifact_capability()`), serving that artifact is an alternative, so a backend
        advertising the matching serve capability also supports the generate phase.

        Returns:
            The control's phase-keyed requirements.
        """
        generate = needs(Capability.IN_PROCESS_TORCH)
        capability = self.artifact_capability()
        if capability is not None:
            generate = any_of(
                generate,
                needs(capability, hint="serve the steer-time artifact on a vLLM backend"),
            )
        return Requirements(generate=generate)

    def steer_access(self) -> ModelAccess:
        """`ModelAccess.MODULE`; training happens on live weights."""
        return ModelAccess.MODULE
