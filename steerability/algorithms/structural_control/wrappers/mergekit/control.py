from pathlib import Path

from steerability.utils.optional import require

require("mergekit")
import mergekit.config as mk_config
import mergekit.merge as mk_merge
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from steerability.algorithms.core.execution.contracts import Capability
from steerability.algorithms.core.execution.payloads import CheckpointArtifact
from steerability.algorithms.structural_control.base import StructuralControl
from steerability.algorithms.structural_control.wrappers.mergekit.args import MergeKitArgs


class MergeKit(StructuralControl):
    """
    Wrapper for merging models via MergeKit [https://github.com/arcee-ai/mergekit](https://github.com/arcee-ai/mergekit).

    MergeKit combines multiple language models using various merge strategies like linear interpolation, SLERP, and
    TIES. This wrapper integrates MergeKit's functionality to enable structural control through model composition.

    The process involves loading a merge configuration (from YAML or dict), executing the merge operation, and
    optionally loading the resulting merged model. Supports caching to avoid redundant operations.

    Reference:

    - "Arcee's MergeKit: A Toolkit for Merging Large Language Models"
      Charles Goddard, Shamane Siriwardhana, Malikeh Ehghaghi, Luke Meyers, Vladimir Karpukhin, Brian Benedict,
      Mark McQuade, Jacob Solawetz
      [https://aclanthology.org/2024.emnlp-industry.36](https://aclanthology.org/2024.emnlp-industry.36)
    """

    Args = MergeKitArgs

    def artifact_capability(self) -> Capability:
        """Merging always leaves a full-weights checkpoint at `out_path`."""
        return Capability.SERVE_CHECKPOINT

    def export_artifact(self) -> CheckpointArtifact:
        """The merged checkpoint directory written (or reused) by `steer()`."""
        return CheckpointArtifact(path=str(self.args.out_path))

    def export_state(self) -> dict:
        """The merged checkpoint under the `"artifact"` key, for freezing."""
        return {"artifact": self.export_artifact()}

    def frozen_form(self, state: dict) -> tuple[str, dict]:
        """The merged checkpoint as a `load_checkpoint` entry."""
        return "structural_control/load_checkpoint", {"path": state["artifact"]}

    def fit_identity(self) -> dict:
        """The merge-relevant args projection: the merge configuration and the dtype."""
        return {
            "config_path": str(self.args.config_path) if self.args.config_path else None,
            "config_dict": self.args.config_dict,
            "dtype": self.args.dtype,
        }

    def steer(
            self,
            model: PreTrainedModel,
            tokenizer: PreTrainedTokenizerBase = None,
            **_
    ) -> PreTrainedModel:
        """Execute model merging via MergeKit and optionally return the merged model.

        Performs structural steering by merging multiple models according to a configuration file or dictionary.
        Supports caching to avoid redundant merge operations and can either return the merged model or the original
        model based on configuration.

        The method follows this logic:

        1. Load merge configuration from YAML file or dictionary
        2. Check if merged model already exists (skip if `force_remerge=False`)
        3. Execute merge if needed using MergeKit
        4. Optionally load and return the merged model

        Args:
            model (PreTrainedModel): The base model (potentially unused depending on the method).
            tokenizer (PreTrainedTokenizerBase, optional): Base tokenizer (currently unused).

        Returns:
            PreTrainedModel: Either the merged model (if `load_merged=True`) or the original model. When returning
            merged model, attempts to attach a new tokenizer if one was created during merging.

        Note:

        - If out_path exists and `force_remerge=False`, skips merging and loads cached result
        - Merged model saved to `out_path` directory with full weights and config
        - If `load_merged=False`, performs merge but returns original model
        """
        args: MergeKitArgs = self.args

        if args.config_path:
            with open(args.config_path, "r", encoding="utf-8") as config_file:
                config = mk_config.MergeConfiguration.model_validate(yaml.safe_load(config_file))
        else:
            config = mk_config.MergeConfiguration.model_validate(args.config_dict)

        # find merged weights
        out_path = Path(args.out_path)
        if out_path.exists() and not args.force_remerge:
            if args.load_merged:
                merged = AutoModelForCausalLM.from_pretrained(
                    pretrained_model_name_or_path=str(out_path),
                    device_map=args.device_map,
                    trust_remote_code=args.trust_remote_code,
                    dtype=getattr(torch, args.dtype)
                )
                return merged
            return model

        # merge
        # with FileLock(str(out_path) + ".lock"):
        mk_merge.run_merge(
            merge_config=config,
            out_path=str(out_path),
            options=mk_merge.MergeOptions(
                cuda=args.allow_cuda,
                trust_remote_code=args.trust_remote_code,
                **args.extra_merge_options,
            )
        )

        # load merged checkpoint (and check if merge returned new tokenizer)
        if args.load_merged:
            merged = AutoModelForCausalLM.from_pretrained(
                out_path,
                dtype=getattr(torch, args.dtype),
                device_map=args.device_map,
                trust_remote_code=args.trust_remote_code,
            )
            try:
                merged.tokenizer = AutoTokenizer.from_pretrained(
                    out_path,
                    trust_remote_code=args.trust_remote_code
                )
            except Exception:
                pass
            return merged

        return model
