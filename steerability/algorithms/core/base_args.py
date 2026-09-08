"""
Base argument validation for steering method configuration.
"""
from dataclasses import dataclass
from typing import Any, Mapping, TypeVar

T = TypeVar("T", bound="BaseArgs")


@dataclass
class BaseArgs:
    """Base class for all method's args classes."""

    @classmethod
    def validate(cls: type[T], _init_data: Any | None = None, **kwargs) -> T:
        """Create and validate an Args instance from dict, kwargs, or existing instance.

        Args:
            _init_data: Existing instance, dict of args, or None. Named with underscore prefix
                to avoid collision with common field names like "data".

        Returns:
            Validated instance of the Args class
        """

        if isinstance(_init_data, cls):
            return _init_data

        if isinstance(_init_data, Mapping):
            kwargs = {**_init_data, **kwargs}

        return cls(**kwargs)
