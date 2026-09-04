"""Base class for estimators that learn artifacts during steer()."""
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from transformers import PreTrainedModel, PreTrainedTokenizerBase

T = TypeVar("T")


class BaseEstimator(ABC, Generic[T]):
    """Learns an artifact from a model and data during the steer() phase.

    The type parameter T is the artifact type (e.g., SteeringVector).
    """

    @abstractmethod
    def fit(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        **kwargs,
    ) -> T:
        """Fit the estimator and return the learned artifact.

        Args:
            model: The model being steered. Used to extract hidden states.
            tokenizer: Tokenizer for encoding training data.
            **kwargs: Estimator-specific arguments (e.g., data, fit specs).

        Returns:
            The learned artifact of type T.
        """
        ...
