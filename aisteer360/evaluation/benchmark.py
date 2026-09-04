"""Benchmark runner for steering pipelines.

Provides a `Benchmark` class for evaluating one or more steering pipeline configurations on a single `UseCase`.
"""
import datetime
import gc
import itertools
import json
import logging
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

import aisteer360
from aisteer360.algorithms.core.execution.spec import KNOWN_BACKEND_KINDS, BackendSpec
from aisteer360.algorithms.core.internals.fingerprint import model_fingerprint
from aisteer360.algorithms.core.specs import ControlSpec
from aisteer360.algorithms.core.steering_pipeline import SteeringPipeline
from aisteer360.algorithms.structural_control.base import StructuralControl
from aisteer360.evaluation.metrics.backend_utils import release_metric_backends
from aisteer360.evaluation.use_cases.base import UseCase
from aisteer360.evaluation.utils.data_utils import to_jsonable
from aisteer360.evaluation.utils.identity import (
    canonical_value,
    config_descriptor_from_controls,
    config_descriptor_from_specs,
    config_digest,
    derive_trial_seed,
    qualname,
)
from aisteer360.utils.tokenization import ensure_pad_token

logger = logging.getLogger(__name__)

_CHECKPOINT_FILENAME = "checkpoint.json"
_CHECKPOINT_FORMAT = 3
_IDENTITY_META_FIELDS = (
    "format", "model", "backend", "fit", "use_case", "evaluation_data_digest", "gen_kwargs_digest",
)


class UnsupportedBenchmarkError(RuntimeError):
    """One or more sweep configurations are unsupported on the configured backends.

    Aggregates support verdicts across every unsupported sweep point so a bad sweep fails once,
    completely, before any model or engine work. Each line ends in core's stable verdict text.

    Attributes:
        failures: One line per unsupported (pipeline, config, control, phase).
    """

    def __init__(self, failures: Sequence[str]) -> None:
        self.failures = list(failures)
        super().__init__("Unsupported pipeline configuration(s):\n" + "\n".join(self.failures))


class Benchmark:
    """Benchmark functionality for comparing steering pipelines on a use case.

    A Benchmark runs one or more steering pipeline configurations on a given use case, optionally with multiple trials
    per configuration. Each trial reuses the same steered model and re-samples any generate-time randomness (e.g.,
    few-shot selection, sampling-based decoding). When ``seed`` is set, one seed is derived per (config, trial) and
    threaded through ``gen_kwargs`` into core's seed path and into use-case-side RNG, so a resumed trial samples what
    an uninterrupted trial would have. Reproduction holds on the same hardware, dtype, and torch/vLLM versions; it is
    a reproducibility handle, not a cross-version guarantee.

    When ``save_dir`` is provided, results are checkpointed to an envelope after each trial (or after each config
    when ``checkpoint_every="config"``), so a run can be interrupted and resumed. Resume is trial-granular, so a
    config completes only its missing trials and raising ``num_trials`` runs only the delta. Only an envelope whose
    identity metadata (``format`` first) matches the current configuration resumes; a well-shaped envelope produced
    under a different configuration or an earlier format is refused with an error naming the differing field, while
    anything unreadable or wrong-shaped at the checkpoint path is ignored with one warning and overwritten by the
    next save.

    The backend is forwarded to the pipelines this benchmark builds. ``device_map`` and ``hf_model_kwargs`` govern
    in-process model loading, including each engine arm's staged steer model; the shared-preloaded-model fast path
    and the fingerprint tripwire are Hugging Face features. Everything else about placement belongs on the
    ``BackendSpec``. Before any model or engine work, ``_preflight`` evaluates every sweep point's ``check()`` and
    either raises one aggregate error (``on_unsupported="raise"``) or skips the unsupported points with a warning
    (``on_unsupported="skip"``). On engine arms, each configuration whose steer plan stages loads and frees its own
    staged model; benchmark-level stage reuse is not performed.

    Non-structural Hugging Face pipelines share one preloaded base model; structural pipelines load their own model
    from ``base_model_name_or_path``. The shared base is expected not to be mutated by a non-structural configuration.
    After each shared-base configuration finishes, a fingerprint tripwire checks the shared model for change and, on
    detecting one, warns naming the configuration and drops the shared model so the next configuration reloads a clean
    base. The tripwire samples a bounded subset of parameters, so it makes the no-mutation invariant observable rather
    than proven.

    Attributes:
        use_case: Use case that defines prompt construction, generation logic, and evaluation metrics.
        base_model_name_or_path: Hugging Face model ID or local path for the base causal language model.
        steering_pipelines: Mapping from pipeline name to a list of controls or `ControlSpec` objects; empty list
            denotes a baseline (no steering).
        runtime_overrides: Optional overrides passed through to `UseCase.generate` for runtime control parameters.
            Overrides are routed by control class name over the pipeline's supplied controls, so two instances of
            the same class in one pipeline share a single override entry.
        hf_model_kwargs: Extra kwargs forwarded to `AutoModelForCausalLM.from_pretrained` on in-process loads.
        gen_kwargs: Generation kwargs forwarded to :meth:`UseCase.generate`.
        device_map: Device placement strategy used when loading in-process (Hugging Face) models.
        num_trials: Number of evaluation trials to run per concrete pipeline configuration. Not part of config
            identity; it is a completion target recorded in checkpoint metadata.
        batch_size: Generation batch size forwarded as a keyword into ``UseCase.generate``.
        save_dir: Optional directory for incremental checkpoints. When set, runs are written to a
            ``checkpoint.json`` envelope and the use case's ``export()`` is called after each pipeline finishes.
        seed: Optional benchmark-level base seed; when set, a per-(config, trial) seed is derived from it.
        backend: Backend forwarded to each pipeline (a `BackendSpec` or a known kind name); None uses the
            in-process Hugging Face backend.
        fit: Fit venue policy forwarded to each pipeline (`"auto"` or `"in_process"`). Part of checkpoint
            identity, since the fit venue affects artifacts and therefore results.
        on_unsupported: ``"raise"`` (default) fails the run with one aggregate error on any unsupported sweep point;
            ``"skip"`` runs the supported points and warns once per skipped point.
        checkpoint_every: ``"trial"`` (default) writes the checkpoint after every trial; ``"config"`` writes once per
            configuration.
    """

    def __init__(
        self,
        use_case: UseCase,
        base_model_name_or_path: str | Path,
        steering_pipelines: dict[str, list[Any]],
        runtime_overrides: dict[str, dict[str, Any]] | None = None,
        hf_model_kwargs: dict | None = None,
        gen_kwargs: dict | None = None,
        device_map: str = "auto",
        num_trials: int = 1,
        batch_size: int = 8,
        save_dir: str | Path | None = None,
        seed: int | None = None,
        backend: "BackendSpec | str | None" = None,
        fit: Literal["auto", "in_process"] = "auto",
        on_unsupported: Literal["raise", "skip"] = "raise",
        checkpoint_every: Literal["trial", "config"] = "trial",
    ) -> None:
        if not isinstance(use_case, UseCase):
            raise TypeError(f"use_case must be a UseCase instance; got {type(use_case).__name__}.")
        if not isinstance(steering_pipelines, dict):
            raise TypeError(f"steering_pipelines must be a dict; got {type(steering_pipelines).__name__}.")
        for name, pipeline in steering_pipelines.items():
            if pipeline is not None and not isinstance(pipeline, (list, tuple)):
                raise TypeError(
                    f"steering_pipelines[{name!r}] must be a list, tuple, or None; got {type(pipeline).__name__}."
                )
        self.num_trials = int(num_trials)
        if self.num_trials < 0:
            raise ValueError("num_trials must be >= 0.")
        self.batch_size = int(batch_size)
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1.")

        if backend is not None and not isinstance(backend, BackendSpec) and backend not in KNOWN_BACKEND_KINDS:
            raise TypeError(
                f"backend must be a BackendSpec or one of {', '.join(KNOWN_BACKEND_KINDS)}; got {backend!r}."
            )
        if fit not in ("auto", "in_process"):
            raise ValueError(f"fit must be 'auto' or 'in_process'; got {fit!r}.")
        if on_unsupported not in ("raise", "skip"):
            raise ValueError(f"on_unsupported must be 'raise' or 'skip'; got {on_unsupported!r}.")
        if checkpoint_every not in ("trial", "config"):
            raise ValueError(f"checkpoint_every must be 'trial' or 'config'; got {checkpoint_every!r}.")
        if seed is not None and "seed" in (gen_kwargs or {}):
            raise ValueError("Set the trial seed via Benchmark(seed=...) or via gen_kwargs['seed'], not both.")

        self.use_case = use_case
        self.base_model_name_or_path = base_model_name_or_path
        self.steering_pipelines = steering_pipelines
        self.runtime_overrides = runtime_overrides
        self.hf_model_kwargs = hf_model_kwargs or {}
        self.gen_kwargs = gen_kwargs or {}
        self.device_map = device_map
        self.save_dir = Path(save_dir) if save_dir is not None else None
        self.seed = seed
        self.backend = backend
        self.fit = fit
        self.on_unsupported = on_unsupported
        self.checkpoint_every = checkpoint_every
        self._backend_kind = (
            backend.kind if isinstance(backend, BackendSpec) else (backend or "huggingface")
        )
        self._skipped: set[tuple[str, str]] = set()

        # lazy-init shared base model/tokenizer
        self._base_model: PreTrainedModel | None = None
        self._base_tokenizer: PreTrainedTokenizerBase | None = None
        self._base_fingerprint: str | None = None

    def _ensure_base_model(self) -> None:
        """Load the base model/tokenizer once (for reuse across pipelines)."""
        if self._base_model is not None and self._base_tokenizer is not None:
            return

        self._base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name_or_path,
            device_map=self.device_map,
            **self.hf_model_kwargs,
        )
        self._base_tokenizer = AutoTokenizer.from_pretrained(self.base_model_name_or_path)
        self._base_tokenizer = ensure_pad_token(self._base_tokenizer)
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

    def _verify_shared_base_model(self, controls: Sequence[Any]) -> None:
        """Tripwire: detect shared-base mutation after a configuration, then quarantine.

        The fingerprint samples up to 8 parameters times 64 elements, so this makes the no-mutation
        invariant observable, not proven; trials after an early-trial mutation ran polluted before
        detection. Warn-and-quarantine (not raise) is deliberate, since aborting a long sweep for one
        misbehaving control is worse than reloading and flagging.

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
        self._base_model = None
        self._base_tokenizer = None
        self._base_fingerprint = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _has_structural_control(controls: Sequence[Any]) -> bool:
        """Return True if any of the controls is an enabled StructuralControl."""
        return any(
            isinstance(control, StructuralControl) and getattr(control, "enabled", True)
            for control in controls
        )

    def _backend_meta(self, value: "BackendSpec | str | None") -> dict:
        """Identity metadata for one backend argument.

        An explicit spec is user-stated identity and is recorded via its ``spec_hash``; the implicit
        Hugging Face default reduces to its kind, since its options carry ``device_map`` and
        ``hf_model_kwargs`` (placement), and moving a resume between machines must not invalidate it.

        Args:
            value: A `BackendSpec`, a known kind name, or None.

        Returns:
            A dict with a ``"kind"`` key and, for an explicit spec, a ``"spec_hash"`` key.
        """
        if isinstance(value, BackendSpec):
            return {"kind": value.kind, "spec_hash": value.spec_hash}
        return {"kind": value or "huggingface"}

    def _checkpoint_meta(self) -> dict:
        """Checkpoint envelope metadata; only ``_IDENTITY_META_FIELDS`` participate in the resume match."""
        return {
            "format": _CHECKPOINT_FORMAT,
            "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "toolkit_version": getattr(aisteer360, "__version__", "unknown"),
            "model": str(self.base_model_name_or_path),
            "backend": self._backend_meta(self.backend),
            "fit": self.fit,
            "use_case": qualname(type(self.use_case)),
            "evaluation_data_digest": config_digest(
                {"data": canonical_value(self.use_case.evaluation_data)}
            ),
            "gen_kwargs_digest": config_digest({"gen_kwargs": canonical_value(self.gen_kwargs)}),
            "num_trials": self.num_trials,
            "batch_size": self.batch_size,
        }

    def _load_checkpoint(self) -> dict[str, list[dict[str, Any]]]:
        """Load profiles from a well-shaped envelope; ignore anything else; refuse an identity
        mismatch.

        Identity is gated once, field by field with ``format`` first, so a readable envelope
        from an earlier format refuses loudly rather than being overwritten.

        Returns:
            The recorded profiles dict, or an empty dict when there is nothing to resume.

        Raises:
            ValueError: If the file is a well-shaped envelope produced under a different
                configuration or format; the message names the first differing identity field.
        """
        if self.save_dir is None:
            return {}
        path = self.save_dir / _CHECKPOINT_FILENAME
        if not path.exists():
            return {}
        try:
            with open(path, encoding="utf-8") as f:
                payload = json.load(f)
        except (json.JSONDecodeError, OSError):
            logger.warning("Could not read checkpoint file; starting fresh.", exc_info=True)
            return {}
        if not (
            isinstance(payload, dict)
            and isinstance(payload.get("meta"), dict)
            and isinstance(payload.get("profiles"), dict)
        ):
            logger.warning(
                "Checkpoint at %s is not a checkpoint envelope; ignoring it (the next save overwrites it).",
                path,
            )
            return {}
        meta = payload["meta"]
        expected = self._checkpoint_meta()
        for field in _IDENTITY_META_FIELDS:
            if meta.get(field) != expected[field]:
                raise ValueError(
                    f"Checkpoint at {path} was produced under a different configuration: {field} "
                    f"was {meta.get(field)!r}, now {expected[field]!r}. Pass a new save_dir, or "
                    "restore the original configuration."
                )
        profiles = payload["profiles"]
        n_runs = sum(len(runs) for runs in profiles.values())
        logger.info("Resumed from checkpoint: %d run(s) across %d pipeline(s)", n_runs, len(profiles))
        return profiles

    def _save_checkpoint(self, profiles: dict[str, list[dict[str, Any]]]) -> None:
        """Atomically write the current profiles to a checkpoint envelope."""
        if self.save_dir is None:
            return
        self.save_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "meta": self._checkpoint_meta(),
            "profiles": to_jsonable(profiles),
        }
        tmp = self.save_dir / f"{_CHECKPOINT_FILENAME}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        tmp.rename(self.save_dir / _CHECKPOINT_FILENAME)

    def run(self) -> dict[str, list[dict[str, Any]]]:
        """Run the benchmark on all configured steering pipelines.

        A pre-flight pass checks every sweep point's backend support before any model or engine work. Each pipeline
        configuration is then expanded into one or more control settings (via `ControlSpec` when present); for each
        configuration, the model is steered once and evaluated over the trials still missing from any resumed
        checkpoint.

        When ``save_dir`` was provided at construction time, runs are persisted incrementally to a checkpoint envelope
        and the use case's ``export()`` method is called after each pipeline finishes. A subsequent call with the same
        ``save_dir`` resumes only the missing trials of each configuration.

        When the run finishes or fails, cached metric backends (judges, ``Perplexity``) are released; metrics construct
        them again on next use.

        Returns:
            A mapping from pipeline name to a list of run dictionaries. Each run dictionary has keys:

                - `"trial_id"`: Integer trial index.
                - `"generations"`: Model generations returned by the use case.
                - `"evaluations"`: Metric results returned by the use case.
                - `"params"`: Mapping from spec name to constructor kwargs used for control, or an empty dict for
                    fixed/baseline pipelines.
                - `"config_id"`: The configuration's canonical identifier (`"baseline"` for the empty pipeline).
                - `"seed"`: The trial's derived seed, or None when no benchmark seed was set.
                - `"provenance"`: Backend kinds, model fingerprint, and toolkit version for the run.

        Raises:
            UnsupportedBenchmarkError: If any sweep point is unsupported and ``on_unsupported="raise"``.
            ValueError: If a resumable checkpoint was produced under a different configuration.
        """
        self._preflight()
        profiles = self._load_checkpoint()

        try:
            for pipeline_name, pipeline in self.steering_pipelines.items():
                logger.info("Running pipeline: %s", pipeline_name)
                pipeline_runs: list[dict[str, Any]] = list(profiles.get(pipeline_name, []))
                profiles[pipeline_name] = pipeline_runs  # live reference; record mutates it in place

                def record(run: dict[str, Any], _runs=pipeline_runs, _profiles=profiles) -> None:
                    _runs.append(run)
                    if self.checkpoint_every == "trial":
                        self._save_checkpoint(_profiles)

                for specs, params, controls_factory in self._iter_config_points(pipeline_name, pipeline):
                    controls = controls_factory()
                    config_id = self._config_id(specs=specs, params=params, controls=controls)
                    if (pipeline_name, config_id) in self._skipped:
                        continue
                    self._run_pipeline(
                        controls, specs=specs, params=params,
                        existing_runs=pipeline_runs, record=record,
                    )
                    if self.checkpoint_every == "config":
                        self._save_checkpoint(profiles)

                logger.info("Pipeline %s complete", pipeline_name)
                self._save_checkpoint(profiles)
                self._try_export(profiles)

            return profiles
        finally:
            release_metric_backends()  # judge and perplexity engines; metrics resolve them again on next use

    def _config_id(self, *, specs=None, params=None, controls=None) -> str:
        """The canonical config id for one configuration (`"baseline"` for the empty pipeline)."""
        if specs:
            return config_digest(config_descriptor_from_specs(specs, params or {}))
        if controls:
            return config_digest(config_descriptor_from_controls(controls))
        return "baseline"

    def _provenance(self) -> dict[str, Any]:
        """Backend kind, model fingerprint, and toolkit version recorded on each run dict."""
        return {
            "backend": self._backend_kind,
            "model_fingerprint": self._base_fingerprint,
            "toolkit_version": getattr(aisteer360, "__version__", "unknown"),
        }

    def _run_pipeline(
        self,
        controls: list[Any],
        *,
        specs: Sequence[Any] | None = None,
        params: dict[str, dict[str, Any]] | None = None,
        existing_runs: list[dict[str, Any]] | None = None,
        record: Callable[[dict[str, Any]], None] | None = None,
    ) -> list[dict[str, Any]]:
        """Run a concrete steering pipeline configuration for its missing trials.

        Handles baseline (no controls), fixed-control, and spec-instantiated configurations. Trials already present
        in ``existing_runs`` for this configuration are kept; only the trials in ``range(num_trials)`` not yet
        recorded are executed. When all trials are present, no model is loaded or steered. Each new run is appended
        through ``record`` (the single accumulation channel used by :meth:`run`); the return value is the full
        trial-sorted run list for direct callers and tests.

        Args:
            controls: Instantiated steering controls, or an empty list for the baseline.
            specs: The configuration's specs (spec-instantiated pipelines), or None.
            params: Mapping from spec name to full constructor kwargs, or None for fixed/baseline pipelines.
            existing_runs: Runs already loaded from a checkpoint for this pipeline.
            record: Callback invoked once per newly executed trial.

        Returns:
            The trial-sorted run list for this configuration.
        """
        config_id = self._config_id(specs=specs, params=params, controls=controls)
        existing = [run for run in (existing_runs or []) if run["config_id"] == config_id]
        done = {run["trial_id"] for run in existing}
        pending = [trial_id for trial_id in range(self.num_trials) if trial_id not in done]
        if len(done) > self.num_trials:
            logger.warning(
                "Config %s holds %d trial(s) but num_trials=%d; keeping all recorded trials.",
                config_id, len(done), self.num_trials,
            )
        if not pending:
            logger.info("Skipping config=%s (all %d trial(s) complete)", config_id, len(done))
            return existing

        uses_shared_base = (
            self._backend_kind == "huggingface" and not self._has_structural_control(controls)
        )
        pipeline: SteeringPipeline | None = None
        new_runs: list[dict[str, Any]] = []
        try:
            pipeline = self._build_config_pipeline(controls)
            tokenizer = pipeline.tokenizer

            for trial_id in pending:
                trial_seed = (
                    derive_trial_seed(self.seed, config_id, trial_id) if self.seed is not None else None
                )
                trial_gen_kwargs = dict(self.gen_kwargs)  # fresh per trial; the use case never sees the shared dict
                extra_kwargs: dict[str, Any] = {}
                if trial_seed is not None:
                    trial_gen_kwargs["seed"] = trial_seed  # -> GenerationParams.seed on every backend
                    extra_kwargs["trial_seed"] = trial_seed  # -> use-case-side rng (lands in **kwargs)
                generations = self.use_case.generate(
                    model_or_pipeline=pipeline,
                    tokenizer=tokenizer,
                    gen_kwargs=trial_gen_kwargs,
                    runtime_overrides=self.runtime_overrides,
                    batch_size=self.batch_size,
                    **extra_kwargs,
                )
                scores = self.use_case.evaluate(generations)
                run = {
                    "trial_id": trial_id,
                    "generations": generations,
                    "evaluations": scores,
                    "params": params or {},
                    "config_id": config_id,
                    "seed": trial_seed,
                    "provenance": self._provenance(),
                }
                new_runs.append(run)
                if record is not None:
                    record(run)
            return sorted(existing + new_runs, key=lambda run: run["trial_id"])
        finally:
            # cleanup controls that may hold GPU resources (e.g., reward models)
            if pipeline is not None:
                for control in (*pipeline.structural_controls, *pipeline.input_controls,
                                *pipeline.state_controls, *pipeline.output_controls):
                    cleanup_fn = getattr(control, "cleanup", None)
                    if callable(cleanup_fn):
                        try:
                            cleanup_fn()
                        except Exception:
                            logger.warning("Control cleanup failed", exc_info=True)
                pipeline.release_backends()  # deterministic engine shutdown
                del pipeline
            if uses_shared_base:
                self._verify_shared_base_model(controls)

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _build_config_pipeline(self, controls: list[Any]) -> SteeringPipeline:
        """Build and steer the pipeline for one configuration under the configured backend.

        The shared-preloaded-model fast path and the fingerprint guard are Hugging Face features; on engine kinds
        core owns model, stage, and engine lifecycle (``device_map`` and ``hf_model_kwargs`` configure the staged
        steer model through the pipeline's constructor knobs). Which
        controls run where is core's contract; unsupported arrangements were already refused by the pre-flight
        check.

        Args:
            controls: Instantiated steering controls for this configuration.

        Returns:
            The steered `SteeringPipeline`.
        """
        common: dict[str, Any] = {
            "controls": list(controls),
            "backend": self.backend,
            "fit": self.fit,
        }
        if self._backend_kind != "huggingface":
            pipeline = SteeringPipeline(
                model_name_or_path=self.base_model_name_or_path,
                device_map=self.device_map, hf_model_kwargs=self.hf_model_kwargs, **common,
            )
            pipeline.steer()
            return pipeline
        if self._has_structural_control(controls):
            pipeline = SteeringPipeline(
                model_name_or_path=self.base_model_name_or_path,
                device_map=self.device_map, hf_model_kwargs=self.hf_model_kwargs, **common,
            )
            pipeline.steer()
            return pipeline
        self._ensure_base_model()  # only shared-base configurations load the shared base
        pipeline = SteeringPipeline(model=self._base_model, tokenizer=self._base_tokenizer, **common)
        pipeline.steer()
        return pipeline

    def _iter_config_points(self, pipeline_name: str, pipeline: list[Any] | None):
        """Yield ``(specs, params, controls_factory)`` per concrete configuration, in execution order.

        Fixed pipelines yield their user-supplied instances (one factory returning the same list, matching today's
        reuse); spec pipelines yield fresh instantiations per factory call, so pre-flight instances are discarded and
        execution re-instantiates. Control instances are never shared across pipelines, and constructors are light by
        contract, so instantiating twice is acceptable.

        Args:
            pipeline_name: Name of the pipeline being enumerated.
            pipeline: The pipeline's list of controls and/or `ControlSpec`s, or None for the baseline.

        Yields:
            One ``(specs, params, controls_factory)`` triple per configuration. ``specs`` is the spec list for
            spec-instantiated configurations and None otherwise; ``params`` is the resolved per-spec kwargs mapping
            or None; ``controls_factory`` builds the configuration's control instances on call.

        Raises:
            TypeError: If the pipeline mixes `ControlSpec` and fixed controls.
            ValueError: If two `ControlSpec`s resolve to the same name.
        """
        pipeline = pipeline or []
        has_specs = any(isinstance(control, ControlSpec) for control in pipeline)
        if has_specs and not all(isinstance(control, ControlSpec) for control in pipeline):
            raise TypeError(
                f"Pipeline '{pipeline_name}' mixes ControlSpec and fixed controls. Either use only fixed controls "
                "or only ControlSpecs. Wrap fixed configs in ControlSpec(vars=None) if needed."
            )
        if not pipeline:
            yield None, None, lambda: []
            return
        if not has_specs:
            fixed = list(pipeline)
            yield None, None, lambda: fixed
            return

        resolved_names = [spec.name or spec.control_cls.__name__ for spec in pipeline]
        duplicates = sorted({name for name in resolved_names if resolved_names.count(name) > 1})
        if duplicates:
            raise ValueError(
                f"Pipeline '{pipeline_name}' has multiple ControlSpecs resolving to the same name(s): "
                f"{duplicates}. Give each spec a distinct `name=` so their parameters are tracked separately."
            )

        base_context = {
            "pipeline_name": pipeline_name,
            "base_model_name_or_path": self.base_model_name_or_path,
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

            def controls_factory(params=params):
                return [
                    spec.control_cls(**params[spec.name or spec.control_cls.__name__])
                    for spec in spec_list
                ]

            yield spec_list, params, controls_factory

    def _preflight(self) -> None:
        """Check every sweep point's backend support before any model or engine work.

        Probe pipelines never load anything (construction is cheap); ``check()`` does no work. A string backend kind whose
        optional dependency is not installed raises `ModuleNotFoundError` here, which is the intended fail-fast.
        Skipped points are not recorded in the checkpoint, so resume re-checks and re-skips (idempotent).

        Raises:
            UnsupportedBenchmarkError: If any sweep point is unsupported and ``on_unsupported="raise"``.
        """
        self._skipped.clear()
        failures: list[str] = []
        for pipeline_name, pipeline in self.steering_pipelines.items():
            for specs, params, controls_factory in self._iter_config_points(pipeline_name, pipeline):
                controls = controls_factory()
                if not controls:
                    continue  # the empty pipeline is trivially supported
                config_id = self._config_id(specs=specs, params=params, controls=controls)
                probe = SteeringPipeline(
                    model_name_or_path=self.base_model_name_or_path, controls=controls,
                    backend=self.backend, fit=self.fit,
                )
                report = probe.check()
                if report.ok:
                    continue
                for failure in report.failures:
                    failures.append(
                        f"{pipeline_name} [{config_id}] {failure.control} ({failure.phase}): "
                        f"{failure.message}"
                    )
                self._skipped.add((pipeline_name, config_id))
        if not failures:
            return
        if self.on_unsupported == "raise":
            raise UnsupportedBenchmarkError(failures)
        for line in failures:
            logger.warning("Skipping unsupported configuration: %s", line)

    def _try_export(self, profiles: dict[str, list[dict[str, Any]]]) -> None:
        """Call the use case's export method; log and swallow failures."""
        if self.save_dir is None:
            return
        try:
            self.export(profiles)
        except Exception:
            logger.warning("Incremental export failed; checkpoint is still intact.", exc_info=True)

    def export(self, profiles: dict[str, list[dict[str, Any]]], save_dir: str | Path | None = None) -> None:
        """Export benchmark results to disk.

        Sanitizes the profiles to a JSON-friendly structure. When the use case overrides `export`, its
        method is called; otherwise the sanitized profiles are written to ``profiles.json`` under
        ``save_dir``. An `export` assigned as an instance attribute (rather than a class override) is
        not detected, so the default write runs.

        Args:
            profiles: The benchmark profiles to export.
            save_dir: Directory to export into; created if absent. When omitted, falls back to the
                ``save_dir`` provided at construction.

        Raises:
            ValueError: If no ``save_dir`` is given and none was provided at construction.
        """
        if save_dir is None:
            save_dir = self.save_dir
        if save_dir is None:
            raise ValueError("No save_dir provided; pass one to export() or set save_dir at construction.")
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        safe_profiles = to_jsonable(profiles)
        if type(self.use_case).export is not UseCase.export:  # instance-attribute exports are not detected
            self.use_case.export(safe_profiles, str(save_path))
            return
        with open(save_path / "profiles.json", "w", encoding="utf-8") as f:
            json.dump(safe_profiles, f, indent=4, ensure_ascii=False)
