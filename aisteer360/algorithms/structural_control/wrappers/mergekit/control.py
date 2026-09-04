from pathlib import Path

from aisteer360.utils.optional import require

require("mergekit")
import mergekit.config as mk_config
import mergekit.merge as mk_merge
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from aisteer360.algorithms.core.execution.contracts import Capability
from aisteer360.algorithms.core.execution.payloads import CheckpointArtifact
from aisteer360.algorithms.structural_control.base import StructuralControl
from aisteer360.algorithms.structural_control.wrappers.mergekit.args import MergeKitArgs


class MergeKit(StructuralControl):
    """
    Wrapper for merging models via MergeKit [https://github.com/arcee-ai/mergekit](https://github.com/arcee-ai/mergekit).

    MergeKit combines multiple language models using various merge strategies like linear interpolation, SLERP, and
    TIES. This wrapper integrates MergeKit's functionality to enable structural control through model composition.

    The process involves loading a merge configuration (from YAML or dict), executing the merge operation, and
    optionally loading the resulting merged model. Supports caching to avoid redundant operations.

    Args:
        config_path (str, optional): Path to YAML merge configuration file. Defaults to None.
        config_dict (dict, optional): Dictionary merge configuration. Defaults to None.
        out_path (str): Output directory for merged model.
        load_merged (bool): Whether to load merged model after merging. Defaults to True.
        force_remerge (bool): Force remerge even if output exists. Defaults to False.
        allow_cuda (bool): Use CUDA acceleration if available. Defaults to True.
        device_map (str | dict, optional): Device mapping for model loading. Defaults to None.
        trust_remote_code (bool): Trust remote code when loading. Defaults to False.
        dtype (str): PyTorch dtype for loading. Defaults to "float16".

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

    def steer(
            self,
            model: PreTrainedModel,
            tokenizer: PreTrainedTokenizer = None,
            **_
    ):
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
            tokenizer (PreTrainedTokenizer, optional): Base tokenizer (currently unused).
            **_: Additional arguments (ignored).

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
                    torch_dtype=getattr(torch, args.dtype)
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
            )
        )

        # load merged checkpoint (and check if merge returned new tokenizer)
        if args.load_merged:
            merged = AutoModelForCausalLM.from_pretrained(
                out_path,
                torch_dtype=getattr(torch, args.dtype),
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
