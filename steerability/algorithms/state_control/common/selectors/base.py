"""Base class for selectors that choose control sites and parameters."""
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

T = TypeVar("T")


class BaseSelector(ABC, Generic[T]):
    """Selects where and how to apply a control, given a model and artifacts.

    The type parameter T describes the selection output (e.g., a list of layer
    ids, or a named tuple of (layer_id, threshold, comparator)).
    """

    @abstractmethod
    def select(self, **kwargs) -> T:
        """Run selection logic and return the result.

        Returns:
            Selection result of type T.
        """
        ...
