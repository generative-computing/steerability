"""LoadLoRA: a structural control that attaches a saved LoRA adapter to the pipeline model."""
from __future__ import annotations

import logging

from transformers import PreTrainedModel, PreTrainedTokenizerBase

from steerability.algorithms.core.execution.contracts import Capability
from steerability.algorithms.core.execution.payloads import LoRAArtifact
from steerability.algorithms.structural_control.base import StructuralControl

from .args import LoadLoRAArgs

logger = logging.getLogger(__name__)


class LoadLoRA(StructuralControl):
    """Attaches a saved LoRA adapter to the pipeline model.

    On the in-process backend, `steer()` verifies the adapter's `base_model` against the
    incoming model's reference, attaches the adapter with `peft.PeftModel.from_pretrained`,
    and merges it into the base weights when `merge=True`. On vLLM backends the adapter is
    served directly (`Capability.SERVE_LORA`), exported as a `LoRAArtifact`. The frozen form
    of an adapter-producing structural control (e.g. an SFT LoRA run) is an instance of this
    control pointing at the trained adapter.

    With `merge=False` (the default) the pipeline model stays a `PeftModel`, so a state control
    listed after this control hooks the adapted model: layout resolution peels the PEFT wrapper
    and the hooked modules are the adapter's own.
    """

    Args = LoadLoRAArgs
    supports_batching = True

    def artifact_capability(self) -> Capability:
        """`Capability.SERVE_LORA`; the configuration is an on-disk adapter."""
        return Capability.SERVE_LORA

    def export_artifact(self) -> LoRAArtifact:
        """The configured adapter directory and its base model reference."""
        return LoRAArtifact(path=str(self.path), base_model=str(self.base_model))

    def steer(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase | None = None,
        **kwargs,
    ) -> PreTrainedModel:
        """Attach the adapter to `model` and return the adapted model.

        Args:
            model: The pipeline model the adapter applies to.
            tokenizer: The pipeline tokenizer (unused).

        Returns:
            The adapted model (merged into the base weights when `merge=True`).

        Raises:
            ValueError: If `model` is None, or its reference differs from `base_model` and
                `allow_base_mismatch` is False.
        """
        from peft import PeftModel

        if model is None:
            raise ValueError("LoadLoRA requires the pipeline model; provide a base model.")
        if not self.allow_base_mismatch:
            live_ref = getattr(model, "name_or_path", None) or getattr(
                getattr(model, "config", None), "_name_or_path", None
            )
            if live_ref is not None and str(live_ref) != str(self.base_model):
                raise ValueError(
                    f"LoRA adapter was trained on base model {self.base_model!r} but the "
                    f"pipeline model is {live_ref!r}; load the matching base model, or set "
                    "allow_base_mismatch=True."
                )
        logger.info("Attaching LoRA adapter %s.", self.path)
        adapted = PeftModel.from_pretrained(model, self.path)
        if self.merge:
            adapted = adapted.merge_and_unload()
        return adapted
