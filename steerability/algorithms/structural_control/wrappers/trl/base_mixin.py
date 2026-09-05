import difflib
from dataclasses import fields
from typing import Any

import trl
from peft import PeftType
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from steerability.algorithms.core.base_control import NotFreezableError
from steerability.algorithms.core.execution.contracts import Capability
from steerability.algorithms.core.execution.payloads import Artifact, CheckpointArtifact, LoRAArtifact


def resolve_config_kwargs(config_cls: type, training_args: dict[str, Any]) -> dict[str, Any]:
    """The kwargs to construct `config_cls` from `training_args`.

    Every key must be a dataclass field of `config_cls`; `None` values are dropped so the config's
    own default applies. The toolkit forwards `training_args` verbatim to TRL, so a key the config
    does not declare would otherwise be passed to a constructor that does not accept it.

    Args:
        config_cls: A TRL config dataclass (`DPOConfig`, `SFTConfig`, `GRPOConfig`, `PPOConfig`).
        training_args: The composed training arguments.

    Returns:
        The accepted kwargs, in `training_args` order, without `None` values.

    Raises:
        ValueError: If any key is not a field of `config_cls`. The message names the config class,
            the installed TRL version, and the unknown keys, with a closest-field suggestion when
            one is close.
    """
    allowed = {field.name for field in fields(config_cls)}
    unknown = sorted(key for key in training_args if key not in allowed)
    if unknown:
        details = []
        for key in unknown:
            close = difflib.get_close_matches(key, allowed, n=1)
            details.append(f"{key!r} (did you mean {close[0]!r}?)" if close else repr(key))
        raise ValueError(
            f"{config_cls.__name__} (trl {trl.__version__}) does not accept training_args key(s): "
            f"{', '.join(details)}. training_args is forwarded verbatim to TRL; remove these keys or "
            f"use a field the installed {config_cls.__name__} declares."
        )
    return {key: value for key, value in training_args.items() if value is not None}


class TRLMixin:
    """
    Small shared helpers for TRL-based structural controls.
    """

    # populated from Args by subclasses
    base_model_name_or_path: str | None = None
    tokenizer_name_or_path: str | None = None
    hf_model_kwargs: dict[str, Any] = {}
    trust_remote_code: bool = False

    training_args: dict[str, Any] = {}
    output_dir: str | None = None
    resume_from_checkpoint: str | None = None

    use_peft: bool = False
    peft_type: Any = None
    lora_kwargs: dict[str, Any] = {}
    adapter_name: str | None = None

    merge_lora_after_train: bool = False
    merged_output_dir: str | None = None

    # resolved at runtime
    tokenizer: PreTrainedTokenizerBase | None = None
    device = None
    _resolved_base_ref: str | None = None

    def _resolve_model_tokenizer(
        self,
        model: PreTrainedModel | None,
        tokenizer: PreTrainedTokenizerBase | None,
    ) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        """Resolve the model and tokenizer, returning both as locals.

        Loads the model from `base_model_name_or_path` when `model` is None, and the tokenizer from
        `tokenizer_name_or_path`, the model's `name_or_path`, or `base_model_name_or_path` when
        `tokenizer` is None. Records the base model reference on `self._resolved_base_ref` (for
        `export_artifact`) and the model's device on `self.device`. The model is not stored on the
        instance; it is returned for the caller to thread.

        Returns:
            The resolved `(model, tokenizer)` pair.

        Raises:
            ValueError: If `model` is None and `base_model_name_or_path` is unset, or the tokenizer
                path cannot be resolved.
        """
        if model is None:
            if not self.base_model_name_or_path:
                raise ValueError("TRLMixin: model is None and `base_model_name_or_path` was not provided.")
            model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name_or_path,
                trust_remote_code=self.trust_remote_code,
                **(self.hf_model_kwargs or {}),
            )

        self._resolved_base_ref = self.base_model_name_or_path or getattr(model, "name_or_path", None)

        if tokenizer is None:
            path = (
                self.tokenizer_name_or_path
                or getattr(model, "name_or_path", None)
                or self.base_model_name_or_path
            )
            if not path:
                raise ValueError("TRLMixin: could not resolve tokenizer path.")
            self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=self.trust_remote_code)
        else:
            self.tokenizer = tokenizer

        self.device = next(model.parameters()).device
        return model, self.tokenizer

    def _post_train_freeze(self, model: PreTrainedModel) -> PreTrainedModel:
        """Put `model` in eval mode, freeze its parameters, and return it.

        The model is passed in and returned; it is not stored on the instance.
        """
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return model

    def _maybe_save_trained_artifacts(self, trainer) -> None:
        output_dir = self.training_args.get("output_dir") or self.output_dir
        if output_dir:
            trainer.save_model(output_dir)
            try:
                self.tokenizer.save_pretrained(output_dir)
            except Exception:
                pass

    def _resolved_output_dir(self) -> str | None:
        return self.training_args.get("output_dir") or self.output_dir

    def artifact_capability(self) -> Capability | None:
        """The serve capability implied by this training configuration.

        Training runs only when `train_dataset` is set, so a configuration without one produces
        no artifact. A LoRA run without merge-back saves an adapter to the output directory
        (`Capability.SERVE_LORA`); a full fine-tune saves a checkpoint there
        (`Capability.SERVE_CHECKPOINT`); a merged LoRA run yields a checkpoint only when
        `merged_output_dir` is set.
        """
        if getattr(self, "train_dataset", None) is None:
            return None
        is_lora = bool(self.use_peft) and self.peft_type == PeftType.LORA
        if is_lora and not self.merge_lora_after_train:
            return Capability.SERVE_LORA if self._resolved_output_dir() else None
        if is_lora and self.merge_lora_after_train:
            return Capability.SERVE_CHECKPOINT if self.merged_output_dir else None
        return Capability.SERVE_CHECKPOINT if self._resolved_output_dir() else None

    def export_artifact(self) -> Artifact | None:
        """The on-disk product of this configuration's `steer()`, matching
        `artifact_capability()`."""
        capability = self.artifact_capability()
        if capability is None:
            return None
        if capability == Capability.SERVE_LORA:
            base = self.base_model_name_or_path or self._resolved_base_ref or ""
            return LoRAArtifact(path=str(self._resolved_output_dir()), base_model=str(base))
        is_lora = bool(self.use_peft) and self.peft_type == PeftType.LORA
        path = self.merged_output_dir if (is_lora and self.merge_lora_after_train) else self._resolved_output_dir()
        return CheckpointArtifact(path=str(path))

    def export_state(self) -> dict[str, Any]:
        """The trained on-disk product under the `"artifact"` key, for freezing.

        Returns an empty mapping when the configuration trains nothing (`train_dataset` is
        None), so an inert wrapper's recipe is its frozen form.

        Raises:
            NotFreezableError: If the configuration trains but produces no on-disk product
                (a merged LoRA run without `merged_output_dir`).
        """
        if getattr(self, "train_dataset", None) is None:
            return {}
        artifact = self.export_artifact()
        if artifact is None:
            raise NotFreezableError(
                f"{type(self).__name__} trains but writes no on-disk product; set output_dir "
                "(or merged_output_dir for a merged LoRA run) to make the result freezable."
            )
        return {"artifact": artifact}

    def frozen_form(self, state: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """The trained artifact as a `load_lora` or `load_checkpoint` entry."""
        artifact = state["artifact"]
        if isinstance(artifact, LoRAArtifact):
            return "structural_control/load_lora", {
                "path": artifact,
                "base_model": artifact.base_model,
                "merge": False,
            }
        return "structural_control/load_checkpoint", {"path": artifact}

    def fit_identity(self) -> Any | None:
        """The training-relevant args projection, or None when nothing trains.

        Output locations (`output_dir`, `merged_output_dir`, `resume_from_checkpoint`, and
        `training_args["output_dir"]`) are excluded, since where a product is saved does not
        change what was trained.
        """
        if getattr(self, "train_dataset", None) is None:
            return None
        args = getattr(self, "args", None)
        if args is None:
            return None
        excluded = {"output_dir", "merged_output_dir", "resume_from_checkpoint"}
        payload: dict[str, Any] = {}
        for f in fields(args):
            if not f.init or f.name in excluded:
                continue
            value = getattr(args, f.name)
            if f.name == "training_args" and isinstance(value, dict):
                value = {k: v for k, v in value.items() if k not in excluded}
            payload[f.name] = value
        return payload

    def _maybe_merge_lora_in_place(self, model: PreTrainedModel) -> PreTrainedModel:
        """Optionally merge LoRA into the base weights, returning the (possibly merged) model.

        When `use_peft` and `merge_lora_after_train` are set and `model` exposes `merge_and_unload`,
        merges the adapter, refreshes `self.device` from the merged model, and saves the merged model
        and tokenizer to `merged_output_dir` when set. Returns the merged model, or `model` unchanged
        otherwise. The model is not stored on the instance.
        """
        if not (self.use_peft and self.merge_lora_after_train):
            return model

        # trainer often returns a PEFT-wrapped model; merge if possible
        if hasattr(model, "merge_and_unload"):
            model = model.merge_and_unload()
            self.device = next(model.parameters()).device

            # save if requested
            if self.merged_output_dir:
                model.save_pretrained(self.merged_output_dir)
                try:
                    self.tokenizer.save_pretrained(self.merged_output_dir)
                except Exception:
                    pass

        return model
