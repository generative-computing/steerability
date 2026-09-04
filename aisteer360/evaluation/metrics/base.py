from abc import ABC, abstractmethod
from typing import Any


class Metric(ABC):
    """Base class for evaluation metrics.

    A metric computes scores on model-generated responses. Subclasses implement `compute`; a metric
    may accept configuration (e.g. a judge model, a tokenizer) through constructor keyword arguments,
    stored on `self.extras`.

    Args:
        name: The metric's name. Defaults to the class name when None, so directly instantiated
            metrics can carry distinct names.
        **extras: Configuration for the metric, stored on `self.extras`.

    Attributes:
        name: The metric's name, defaulting to the class name. `UseCase.evaluate` keys its results by
            this name, so two metrics sharing a name collide in the results dict (the use case warns
            at construction).
        extras: The constructor keyword arguments.
    """

    def __init__(self, name: str | None = None, **extras: Any) -> None:
        self.name: str = name or self.__class__.__name__
        self.extras: dict[str, Any] = extras

    @abstractmethod
    def compute(
        self,
        responses: list[Any],
        prompts: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Compute the metric's scores.

        Stateless with respect to the instance: the result is a function of the arguments alone, so
        one metric instance can score many runs. Use cases legitimately pass richer per-item records
        (not only strings) as `responses`, hence the `list[Any]` type.

        Args:
            responses: The model outputs to score, one per item.
            prompts: The prompts that produced the responses, one per item, or None.
            **kwargs: Additional per-item fields the metric needs (e.g. reference answers).

        Returns:
            A mapping from result key to value.
        """
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.compute(*args, **kwargs)
