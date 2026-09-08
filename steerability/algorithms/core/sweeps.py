"""Configuration sweeps over one base model: expansion, pre-flight support checks, and pipeline lifecycle.

One configuration is one concrete list of control instances. `expand_configurations` enumerates the
configurations of a pipelines mapping (fixed controls, `ControlSpec` sweeps, and the empty baseline)
as `ConfigPoint`s carrying canonical identity; `preflight` evaluates backend support per point before
any model or engine work; `PipelineFactory` builds, steers, and releases one pipeline per point with a
shared preloaded base model on the Hugging Face backend and a fingerprint tripwire over it.
"""
import gc
import itertools
import logging
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Literal, Mapping, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from steerability.algorithms.core.execution.spec import BackendSpec
from steerability.algorithms.core.identity import (
    config_descriptor_from_controls,
    config_descriptor_from_specs,
    config_digest,
)
from steerability.algorithms.core.internals.fingerprint import model_fingerprint
from steerability.algorithms.core.specs import ControlSpec
from steerability.algorithms.core.steering_pipeline import SteeringPipeline
from steerability.algorithms.structural_control.base import StructuralControl
from steerability.utils.tokenization import ensure_pad_token

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ConfigPoint:
    """One concrete configuration of one named pipeline.

    Attributes:
        pipeline_name: Name of the pipeline the point belongs to.
        config_id: Canonical configuration identifier; `"baseline"` for the empty pipeline, else
            the digest of `descriptor`.
        descriptor: JSON-serializable canonical descriptor of the configuration (control classes
            and their full constructor parameters).
        specs: The `ControlSpec` list for spec-defined pipelines, else None.
        params: Mapping from resolved spec name to that spec's resolved constructor kwargs, else
            None for fixed and baseline pipelines.
        controls_factory: Zero-argument callable building the configuration's control instances.
            Fixed pipelines return the user-supplied instances on every call; spec pipelines
            instantiate fresh controls per call, so pre-flight instances are discarded and
            execution re-instantiates.
    """

    pipeline_name: str
    config_id: str
    descriptor: Mapping[str, Any]
    specs: tuple | None
    params: Mapping[str, Mapping[str, Any]] | None
    controls_factory: Callable[[], list]


def expand_configurations(
    pipelines: Mapping[str, Sequence],
    *,
    base_model_name_or_path: str | Path,
) -> Iterator[ConfigPoint]:
    """Enumerate the concrete configurations of a pipelines mapping, in execution order.

    Each pipeline value is an empty sequence (the unsteered baseline), a list of instantiated
    controls (one fixed configuration), or a list of `ControlSpec`s (one configuration per point
    of the cartesian product of the specs' search spaces). Spec search spaces receive a context
    dict with `"pipeline_name"`, `"base_model_name_or_path"`, and, at resolution time,
    `"combo_id"`.

    Args:
        pipelines: Mapping from pipeline name to `[]`, `[Control, ...]`, or `[ControlSpec, ...]`
            (None is treated as the baseline).
        base_model_name_or_path: The base model reference, forwarded into spec contexts.

    Yields:
        One `ConfigPoint` per concrete configuration.

    Raises:
        TypeError: If a pipeline mixes `ControlSpec` and fixed controls.
        ValueError: If two `ControlSpec`s in one pipeline resolve to the same name.
    """
    for pipeline_name, pipeline in pipelines.items():
        pipeline = list(pipeline or [])
        has_specs = any(isinstance(control, ControlSpec) for control in pipeline)
        if has_specs and not all(isinstance(control, ControlSpec) for control in pipeline):
            raise TypeError(
                f"Pipeline '{pipeline_name}' mixes ControlSpec and fixed controls. Either use only fixed controls "
                "or only ControlSpecs. Wrap fixed configs in ControlSpec(vars=None) if needed."
            )
        if not pipeline:
            yield ConfigPoint(
                pipeline_name=pipeline_name,
                config_id="baseline",
                descriptor={"controls": []},
                specs=None,
                params=None,
                controls_factory=lambda: [],
            )
            continue
        if not has_specs:
            fixed = list(pipeline)
            descriptor = config_descriptor_from_controls(fixed)
            yield ConfigPoint(
                pipeline_name=pipeline_name,
                config_id=config_digest(descriptor),
                descriptor=descriptor,
                specs=None,
                params=None,
                controls_factory=lambda fixed=fixed: fixed,
            )
            continue

        resolved_names = [spec.name or spec.control_cls.__name__ for spec in pipeline]
        duplicates = sorted({name for name in resolved_names if resolved_names.count(name) > 1})
        if duplicates:
            raise ValueError(
                f"Pipeline '{pipeline_name}' has multiple ControlSpecs resolving to the same name(s): "
                f"{duplicates}. Give each spec a distinct `name=` so their parameters are tracked separately."
            )

        base_context = {
            "pipeline_name": pipeline_name,
            "base_model_name_or_path": base_model_name_or_path,
        }
        spec_points = []
        for spec in pipeline:
            points = list(spec.iter_points(base_context)) or [{}]
            spec_points.append((spec, points))
        spec_list, points_lists = zip(*spec_points)

        for combo_id, combo in enumerate(itertools.product(*points_lists)):
            context = {**base_context, "combo_id": combo_id}
            params = {
                (spec.name or spec.control_cls.__name__): spec.resolve_params(chosen=point, context=context)
                for spec, point in zip(spec_list, combo)
            }

            def controls_factory(params=params, spec_list=spec_list):
                return [
                    spec.control_cls(**params[spec.name or spec.control_cls.__name__])
                    for spec in spec_list
                ]

            descriptor = config_descriptor_from_specs(spec_list, params)
            yield ConfigPoint(
                pipeline_name=pipeline_name,
                config_id=config_digest(descriptor),
                descriptor=descriptor,
                specs=tuple(spec_list),
                params=params,
                controls_factory=controls_factory,
            )


def preflight(
    points: Iterable[ConfigPoint],
    *,
    base_model_name_or_path: str | Path,
    backend: BackendSpec | str | None,
    fit: Literal["auto", "in_process"],
) -> list[str]:
    """Evaluate backend support for every point before any model or engine work.

    Constructs one probe `SteeringPipeline` per point (construction performs no I/O), evaluates
    `check()`, and discards the probe. The empty baseline is trivially supported and produces no
    probe. Callers choose whether to raise on failures or skip the failing points.

    Args:
        points: The configuration points to check.
        base_model_name_or_path: The base model reference for the probe pipelines.
        backend: Backend forwarded to the probe pipelines.
        fit: Fit venue policy forwarded to the probe pipelines.

    Returns:
        One message per unsupported (point, control, phase), each naming the pipeline and its
        `config_id` and ending in core's stable verdict text. Empty when every point is supported.

    Raises:
        ModuleNotFoundError: If a configured backend kind requires an optional dependency that is
            not installed.
    """
    messages: list[str] = []
    for point in points:
        controls = point.controls_factory()
        if not controls:
            continue
        probe = SteeringPipeline(
            model_name_or_path=base_model_name_or_path, controls=controls,
            backend=backend, fit=fit,
        )
        report = probe.check()
        if report.ok:
            continue
        for failure in report.failures:
            messages.append(
                f"{point.pipeline_name} [{point.config_id}] {failure.control} ({failure.phase}): "
                f"{failure.message}"
            )
    return messages


def _has_structural_control(controls: Sequence[Any]) -> bool:
    """Return True if any of the controls is an enabled `StructuralControl`."""
    return any(
        isinstance(control, StructuralControl) and getattr(control, "enabled", True)
        for control in controls
    )


class PipelineFactory:
    """Build, steer, and release one `SteeringPipeline` per configuration against one base model.

    On the Hugging Face backend, configurations without an enabled structural control share one
    preloaded base model and tokenizer, loaded lazily on first use. Configurations with an enabled
    structural control, and every configuration on an engine backend, construct from
    `base_model_name_or_path` with the placement knobs; before such a configuration the factory
    drops any resident shared base, so one full model is resident at a time, and the next
    shared-base configuration reloads it.

    The shared base is expected not to be mutated by a non-structural configuration. After each
    shared-base configuration a fingerprint tripwire checks the shared model for change and, on
    detecting one, warns naming the configuration's controls and drops the shared model so the next
    configuration reloads a clean base. The fingerprint samples a bounded subset of parameters, so
    the tripwire makes the no-mutation invariant observable rather than proven.

    Args:
        base_model_name_or_path: Hugging Face model ID or local path of the base model.
        backend: Backend forwarded to each pipeline (a `BackendSpec` or a known kind name); None
            uses the in-process Hugging Face backend.
        fit: Fit venue policy forwarded to each pipeline.
        hf_model_kwargs: Extra kwargs forwarded to `AutoModelForCausalLM.from_pretrained` on
            in-process loads (the shared base and each engine arm's staged steer model).
        device_map: Device placement strategy for in-process loads.
        trust_remote_code: Trust remote code when loading tokenizers; forwarded to each pipeline.
    """

    def __init__(
        self,
        base_model_name_or_path: str | Path,
        *,
        backend: BackendSpec | str | None = None,
        fit: Literal["auto", "in_process"] = "auto",
        hf_model_kwargs: dict | None = None,
        device_map: str | dict | None = "auto",
        trust_remote_code: bool = False,
    ) -> None:
        self.base_model_name_or_path = base_model_name_or_path
        self.backend = backend
        self.fit = fit
        self.hf_model_kwargs = hf_model_kwargs or {}
        self.device_map = device_map
        self.trust_remote_code = trust_remote_code

        self._base_model: PreTrainedModel | None = None
        self._base_tokenizer: PreTrainedTokenizerBase | None = None
        self._base_fingerprint: str | None = None

    @property
    def backend_kind(self) -> str:
        """The configured backend kind (`"huggingface"`, `"vllm"`, or `"vllm-serve"`)."""
        if isinstance(self.backend, BackendSpec):
            return self.backend.kind
        return self.backend or "huggingface"

    @property
    def shared_base_fingerprint(self) -> str | None:
        """Fingerprint of the resident shared base model, or None when none is resident."""
        return self._base_fingerprint

    def _ensure_base_model(self) -> None:
        """Load the shared base model and tokenizer once (for reuse across configurations)."""
        if self._base_model is not None and self._base_tokenizer is not None:
            return
        self._base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name_or_path,
            device_map=self.device_map,
            **self.hf_model_kwargs,
        )
        self._base_tokenizer = ensure_pad_token(AutoTokenizer.from_pretrained(
            self.base_model_name_or_path, trust_remote_code=self.trust_remote_code,
        ))
        self._base_fingerprint = self._fingerprint_or_none(self._base_model)

    def _fingerprint_or_none(self, model) -> str | None:
        """Digest of the shared base model, or None (guard disabled) when fingerprinting fails."""
        if model is None:
            return None
        try:
            return model_fingerprint(model)
        except Exception:
            logger.debug("Model fingerprint unavailable; shared-model guard disabled.", exc_info=True)
            return None

    def _drop_shared_base(self) -> None:
        """Release the shared base model and tokenizer references and reclaim memory."""
        if self._base_model is None and self._base_tokenizer is None:
            return
        self._base_model = None
        self._base_tokenizer = None
        self._base_fingerprint = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _verify_shared_base_model(self, controls: Sequence[Any]) -> None:
        """Tripwire: detect shared-base mutation after a configuration, then quarantine.

        The fingerprint samples up to 8 parameters times 64 elements, so this makes the no-mutation
        invariant observable, not proven. Warn-and-quarantine (not raise) is deliberate, since
        aborting a long sweep for one misbehaving control is worse than reloading and flagging.

        Args:
            controls: The configuration's controls, named in the warning (or "baseline" when empty).
        """
        if self._base_model is None or self._base_fingerprint is None:
            return
        current = self._fingerprint_or_none(self._base_model)
        if current == self._base_fingerprint:
            return
        names = ", ".join(type(control).__name__ for control in controls) or "baseline"
        logger.warning(
            "Shared base model changed during configuration [%s] (fingerprint %s -> %s); its recorded "
            "results reflect the mutated weights. Dropping the shared model so the next configuration "
            "reloads a clean base.",
            names, self._base_fingerprint, current,
        )
        self._drop_shared_base()

    @contextmanager
    def steered(self, controls: Sequence) -> Iterator[SteeringPipeline]:
        """Build and steer a pipeline for one configuration, releasing it on exit.

        Everything the caller does inside the `with` block runs inside the protected region: on
        exit (including on error) every control's `cleanup()` runs best effort, the pipeline's
        backends are released, the shared-base fingerprint tripwire runs for shared-base
        configurations, and memory is reclaimed.

        Args:
            controls: Instantiated controls for this configuration (empty for the baseline).

        Yields:
            The steered `SteeringPipeline`.
        """
        controls = list(controls)
        uses_shared_base = (
            self.backend_kind == "huggingface" and not _has_structural_control(controls)
        )
        common: dict[str, Any] = {"controls": controls, "backend": self.backend, "fit": self.fit}
        pipeline: SteeringPipeline | None = None
        try:
            if uses_shared_base:
                self._ensure_base_model()
                pipeline = SteeringPipeline(
                    model=self._base_model, tokenizer=self._base_tokenizer, **common,
                )
            else:
                self._drop_shared_base()
                pipeline = SteeringPipeline(
                    model_name_or_path=self.base_model_name_or_path,
                    device_map=self.device_map,
                    hf_model_kwargs=self.hf_model_kwargs,
                    trust_remote_code=self.trust_remote_code,
                    **common,
                )
            pipeline.steer()
            yield pipeline
        finally:
            if pipeline is not None:
                for control in (*pipeline.structural_controls, *pipeline.input_controls,
                                *pipeline.state_controls, *pipeline.output_controls):
                    cleanup_fn = getattr(control, "cleanup", None)
                    if callable(cleanup_fn):
                        try:
                            cleanup_fn()
                        except Exception:
                            logger.warning("Control cleanup failed", exc_info=True)
                pipeline.release_backends()
                del pipeline
            if uses_shared_base:
                self._verify_shared_base_model(controls)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def release(self) -> None:
        """Drop the shared base model; the factory remains usable and reloads it on next use."""
        self._drop_shared_base()
