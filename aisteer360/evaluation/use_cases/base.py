"""Base class for all use cases.

Provides the framework for loading evaluation data, declaring use-case-specific constructor
parameters, applying metrics, and running standardized evaluations. Subclasses implement
`generate()` and `evaluate()`; they declare any extra constructor parameters as class-level
annotations rather than writing an `__init__`.
"""
import copy
import inspect
import json
import logging
import warnings
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar, NamedTuple, get_origin

from aisteer360.evaluation.metrics.base import Metric

logger = logging.getLogger(__name__)


class _DeclaredParameter(NamedTuple):
    required: bool
    default: Any


def _is_classvar(annotation: Any) -> bool:
    """True for `ClassVar` annotations, including the stringized forms."""
    if annotation is ClassVar or get_origin(annotation) is ClassVar:
        return True
    if isinstance(annotation, str):
        stripped = annotation.strip()
        return stripped.startswith("ClassVar") or stripped.startswith("typing.ClassVar")
    return False


class UseCase(ABC):
    """Base use case class.

    A subclass declares each extra constructor parameter as a class-level annotation below
    `UseCase`. A class attribute of the same name makes that parameter optional with the attribute
    as its default; a bare annotation makes it required. At construction the declared parameters are
    read from `**kwargs`: unknown keywords raise `TypeError`, missing required parameters raise
    `TypeError`, and each declared value is set as an instance attribute. Mutable class-level
    defaults (`list`, `dict`, `set`) are copied per instance so instances never share one object.

    Annotations that are underscore-prefixed, name a base `__init__` parameter, are `ClassVar`
    (including the stringized forms), or whose class value is a method or property are not treated
    as parameters. A class in the mro that does not subclass `UseCase` (a plain mixin) contributes no
    parameters. An optional parameter whose default is a callable is skipped by the callable rule, so
    callable defaults are unsupported.

    Retained evaluation instances are validated through `validate_evaluation_data` at construction,
    after shuffling and sampling, so only the instances that will run are checked.
    """

    @classmethod
    def _declared_parameters(cls) -> dict[str, _DeclaredParameter]:
        """Extra constructor parameters declared by class-level annotations below `UseCase`.

        Returns:
            A mapping from parameter name to a `_DeclaredParameter(required, default)`. A bare
            annotation (no class value) is required; an annotation with a non-callable class value is
            optional with that value as its default.
        """
        base_init_names = frozenset(
            name
            for name, parameter in inspect.signature(UseCase.__init__).parameters.items()
            if parameter.kind not in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
        ) - {"self"}

        declared: dict[str, _DeclaredParameter] = {}
        for klass in reversed(cls.__mro__):
            if klass is UseCase or not (isinstance(klass, type) and issubclass(klass, UseCase)):
                continue  # only classes strictly below UseCase in the mro declare parameters
            for name, annotation in vars(klass).get("__annotations__", {}).items():
                if name.startswith("_") or name in base_init_names or _is_classvar(annotation):
                    continue
                value = getattr(cls, name, inspect.Parameter.empty)
                if value is inspect.Parameter.empty:
                    declared[name] = _DeclaredParameter(required=True, default=None)
                    continue
                if callable(value) or isinstance(value, property):
                    continue  # an annotated method or property is not a parameter
                declared[name] = _DeclaredParameter(required=False, default=value)
        return declared

    def __init__(
        self,
        evaluation_data: list[dict] | str | Path,
        evaluation_metrics: list[Metric],
        num_samples: int = -1,
        shuffle: bool = False,
        seed: int = 555,
        **kwargs,
    ) -> None:
        """Load evaluation data, bind declared parameters, and validate the retained instances.

        Args:
            evaluation_data: A sequence of mappings (one per instance) or a path to a `.json`/
                `.jsonl` file. In-memory sequences are shallow-copied per instance so shuffling and
                sampling never mutate the caller's list.
            evaluation_metrics: Metrics used by `evaluate`. Every item must be a `Metric`.
            num_samples: Keep only the first `num_samples` instances (after shuffling) when positive;
                a non-positive value keeps all.
            shuffle: Shuffle the instances with a `random.Random(seed)` before sampling.
            seed: Seed for the shuffle.
            **kwargs: Values for the subclass's declared parameters.

        Raises:
            TypeError: If a keyword is not a declared parameter, a required declared parameter is
                missing, `evaluation_data` is neither a sequence of mappings nor a `.json`/`.jsonl`
                path, or an item of `evaluation_metrics` is not a `Metric`.
            ValueError: If a retained instance fails `validate_evaluation_data`; the message carries
                the offending `evaluation_data[<index>]` prefix.

        Warns:
            UserWarning: If the loaded evaluation data is empty, or if two metrics share a name
                (later metrics replace earlier ones in the name-keyed results).
        """
        declared = self._declared_parameters()
        unknown = sorted(set(kwargs) - set(declared))
        if unknown:
            raise TypeError(
                f"{type(self).__name__} got unexpected keyword argument(s) {unknown}; "
                f"declared parameters are {sorted(declared)}."
            )
        missing = sorted(name for name, spec in declared.items() if spec.required and name not in kwargs)
        if missing:
            raise TypeError(f"{type(self).__name__} missing required parameter(s) {missing}.")
        for name, spec in declared.items():
            value = kwargs[name] if name in kwargs else spec.default
            if name not in kwargs and isinstance(value, (list, dict, set)):
                value = copy.copy(value)  # never share a mutable class-level default across instances
            setattr(self, name, value)

        self.evaluation_data = self._load_evaluation_data(evaluation_data)
        if not self.evaluation_data:
            warnings.warn(
                "Either evaluation data was not provided, or was unable to be generated.", UserWarning
            )

        if shuffle:
            import random

            random.Random(seed).shuffle(self.evaluation_data)
        if num_samples > 0:
            self.evaluation_data = self.evaluation_data[:num_samples]

        for index, item in enumerate(self.evaluation_data):  # validate only what will run
            try:
                self.validate_evaluation_data(item)
            except ValueError as error:
                raise ValueError(f"evaluation_data[{index}]: {error}") from error

        if not all(isinstance(metric, Metric) for metric in evaluation_metrics):
            raise TypeError("All items in `evaluation_metrics` must be of type `Metric`.")
        names = [metric.name for metric in evaluation_metrics]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            warnings.warn(
                f"Duplicate metric name(s) {duplicates}; later metrics replace earlier ones in "
                "name-keyed results.",
                UserWarning,
            )
        self.evaluation_metrics = evaluation_metrics
        self._metrics_by_name = {metric.name: metric for metric in evaluation_metrics}

    @staticmethod
    def _load_evaluation_data(evaluation_data: list[dict] | str | Path) -> list[dict]:
        """Load evaluation data to a list of dicts.

        Args:
            evaluation_data: A sequence of mappings, or a path to a `.json`/`.jsonl` file.

        Returns:
            A list of dicts, one per instance. Items are shallow-copied so downstream shuffling and
            sampling never mutate the caller's list.

        Raises:
            TypeError: If `evaluation_data` is neither a non-string sequence nor a path, or the
                loaded content is not a list of mappings.
        """
        if isinstance(evaluation_data, (str, Path)):
            path = Path(evaluation_data)
            with open(path, encoding="utf-8") as f:
                loaded = (
                    [json.loads(line) for line in f if line.strip()]
                    if path.suffix == ".jsonl"
                    else json.load(f)
                )
        elif isinstance(evaluation_data, Sequence) and not isinstance(evaluation_data, (str, bytes)):
            loaded = list(evaluation_data)
        else:
            raise TypeError(
                f"evaluation_data must be a sequence of mappings or a path to .json/.jsonl; got "
                f"{type(evaluation_data).__name__}."
            )
        if not isinstance(loaded, list) or not all(isinstance(item, Mapping) for item in loaded):
            raise TypeError("evaluation_data must contain mappings (one per instance).")
        return [dict(item) for item in loaded]  # shallow copies: shuffle/sample never mutate the caller's list

    @abstractmethod
    def generate(
        self,
        model_or_pipeline,
        tokenizer,
        gen_kwargs=None,
        runtime_overrides: dict[str, dict[str, Any]] | None = None,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Required generation logic for the current use case."""
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, generations: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """Required evaluation logic for the model's generations via `evaluation_metrics`."""
        raise NotImplementedError

    def validate_evaluation_data(self, instance: Mapping[str, Any]) -> None:
        """Validate one retained instance; raise `ValueError` on schema violations. Default: no-op."""

    def export(self, profiles: dict[str, Any], save_dir: str) -> None:
        """Optional formatting and export of evaluation profiles. Default: no-op."""
        logger.debug("%s defines no export; skipping.", type(self).__name__)
