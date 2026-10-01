"""LoadCheckpoint: a structural control that installs a checkpoint as the pipeline model."""
from __future__ import annotations

import logging

from transformers import AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizerBase

from steerability.algorithms.core.execution.access import ModelAccess
from steerability.algorithms.core.execution.contracts import Capability
from steerability.algorithms.core.execution.payloads import CheckpointArtifact
from steerability.algorithms.structural_control.base import StructuralControl

from .args import LoadCheckpointArgs

logger = logging.getLogger(__name__)


class LoadCheckpoint(StructuralControl):
    """Installs a saved full-weights checkpoint as the pipeline model.

    On the in-process backend, `steer()` loads the checkpoint with
    `AutoModelForCausalLM.from_pretrained` and returns it, replacing the incoming model for
    subsequent controls. On vLLM backends the checkpoint is served directly
    (`Capability.SERVE_CHECKPOINT`), exported as a `CheckpointArtifact`. The steer step declares
    `ModelAccess.FACTS`, since the checkpoint path is part of the configuration. On an engine
    backend, `steer()` then loads nothing, and the checkpoint is loaded in process only on a stage
    that a later steer step needs. The frozen form of a trained structural control (fine-tune,
    merge) is an instance of this control pointing at the trained checkpoint.
    """

    Args = LoadCheckpointArgs
    supports_batching = True

    def artifact_capability(self) -> Capability:
        """`Capability.SERVE_CHECKPOINT`; the configuration is an on-disk checkpoint."""
        return Capability.SERVE_CHECKPOINT

    def export_artifact(self) -> CheckpointArtifact:
        """The configured checkpoint directory."""
        return CheckpointArtifact(path=str(self.path))

    def steer_access(self) -> ModelAccess:
        """`ModelAccess.FACTS`; an engine backend serves the configured checkpoint without a staged model."""
        return ModelAccess.FACTS

    def steer(
        self,
        model: PreTrainedModel | None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        session=None,
        **kwargs,
    ) -> PreTrainedModel | None:
        """Load the checkpoint and return it as the new pipeline model.

        Args:
            model: The incoming pipeline model (unused; the checkpoint replaces it).
            tokenizer: The pipeline tokenizer (unused).
            session: The steering session. On a session that serves no in-process model (an
                engine backend), nothing is loaded and the engine serves the exported artifact.

        Returns:
            The loaded model, or None on a session that serves no in-process model.
        """
        if session is not None and not getattr(session, "in_process", True):
            return None
        logger.info("Loading checkpoint %s.", self.path)
        return AutoModelForCausalLM.from_pretrained(
            self.path,
            device_map=self.device_map,
            trust_remote_code=self.trust_remote_code,
            **self.hf_model_kwargs,
        )
