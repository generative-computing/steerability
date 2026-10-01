"""
Specification utilities for steering controls.

Provides:

- `ControlSpec`: a description of a steering control plus a hyperparameter search space.
- `Factory`: a marker for a `ControlSpec` parameter that is computed from the sweep context.
"""
import itertools
import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Literal, Mapping, Sequence

# the type of the search space for a ControlSpec:
#   - Mapping[str, Sequence[Any]]: intervals (cartesian product)
#   - Sequence[Mapping[str, Any]]: list of parameter dicts
#   - Callable[[dict], Iterable[Mapping[str, Any]]]: generates dicts given a context
Space = Mapping[str, Sequence[Any]] | Sequence[Mapping[str, Any]] | Callable[[dict], Iterable[Mapping[str, Any]]]


@dataclass(frozen=True, slots=True)
class Factory:
    """A `ControlSpec` parameter computed from the sweep context at each search point.

    `ControlSpec.resolve_params` calls `function` with the context of the search point and
    passes the return value to the control's constructor. Parameters that are not wrapped in
    `Factory`, including callables, are passed to the constructor unchanged.

    Attributes:
        function: A callable `(context) -> value`. The context contains the sweep's base context
            and the point's chosen values under `"search_params"`.
    """

    function: Callable[[dict], Any]

    def __call__(self, context: dict) -> Any:
        """Return `function(context)`."""
        return self.function(context)


@dataclass(slots=True)
class ControlSpec:
    """Specification for a parameterized steering control.

    A `ControlSpec` describes a control class plus a search space over its constructor arguments. The sweep layer
    (`core/sweeps.py`, and `SteeringEval` above it) uses it to instantiate control instances for different
    hyperparameter settings.

    Attributes:
        control_cls: The steering control class to instantiate.
        params: Fixed constructor arguments for the control. A value wrapped in `Factory` is
            computed from the context of each search point. Every other value, including a
            callable, is passed to the constructor unchanged.
        vars: Optional search space over additional constructor arguments. May be:

            - mapping (cartesian grid), whose dimensions must each contain at least one value
            - list of parameter dicts
            - callable that yields parameter dicts given a context
        name: Optional short name for this spec; defaults to `control_cls.__name__` if omitted.
        search_strategy: Strategy for traversing `vars` when it is a mapping or a sequence. Either `"grid"` (use all
            points) or `"random"` (sample a subset).
        num_samples: Number of points to sample when `search_strategy="random"` and `vars` is a mapping or sequence;
            ignored when `vars` is callable.
        seed: Optional random seed used when `search_strategy="random"`.
    """

    control_cls: type
    params: Mapping[str, Any] = field(default_factory=dict)
    vars: Space | None = None
    name: str | None = None
    search_strategy: Literal["grid", "random"] = "grid"
    num_samples: int | None = None
    seed: int | None = None

    def iter_points(self, context: dict) -> Iterable[dict[str, Any]]:
        """Iterate over local search points for this spec.

        Args:
            context: Context dictionary; passed through to functional `vars` if `vars` is callable.

        Yields:
            Parameter dictionaries (possibly empty) that will be merged into `params` when constructing a concrete
            control instance.

        Raises:
            ValueError: If `vars` is a mapping with a dimension that contains no values.
        """
        search_space = self.vars

        # no search space
        if search_space is None:
            yield {}
            return

        # forward the context
        if callable(search_space):
            yield from search_space(context)  # callable controls sampling
            return

        # Mapping[str, Sequence[Any]]: potentially large cartesian product
        if isinstance(search_space, Mapping):
            param_names = list(search_space.keys())
            param_values = [list(search_space[name]) for name in param_names]

            empty = [name for name, vals in zip(param_names, param_values) if not vals]
            if empty:
                raise ValueError(
                    f"ControlSpec for {self.control_cls.__name__} has search dimension(s) {empty} with no values; give "
                    "each dimension at least one value, or move a fixed value to `params`."
                )

            sizes = [len(vals) for vals in param_values]
            n_points = math.prod(sizes)

            # GRID SEARCH: iterate over the cartesian product
            if (
                self.search_strategy == "grid"
                or self.num_samples is None
                or self.num_samples >= n_points
            ):
                for combo in itertools.product(*param_values):
                    yield dict(zip(param_names, combo))
                return

            # RANDOM SEARCH: sample indices
            rng = random.Random(self.seed)
            k = min(self.num_samples, n_points)
            index_samples = rng.sample(range(n_points), k)

            # decode (flat) index into a combination of parameter choices
            for flat_index in index_samples:
                idx = flat_index
                indices_per_dim: list[int] = []
                for size in reversed(sizes):
                    indices_per_dim.append(idx % size)
                    idx //= size
                indices_per_dim.reverse()

                values = [param_values[dim][indices_per_dim[dim]] for dim in range(len(param_names))]
                yield dict(zip(param_names, values))
            return

        # Sequence[Mapping[str, Any]]: explicit list of parameter dicts
        combinations = [dict(param_dict) for param_dict in search_space]

        if (
            self.search_strategy == "random"
            and self.num_samples is not None
            and self.num_samples < len(combinations)
        ):
            rng = random.Random(self.seed)
            combinations = rng.sample(combinations, self.num_samples)

        for combination in combinations:
            yield combination

    def resolve_params(self, chosen: dict[str, Any], context: dict) -> dict[str, Any]:
        """Compute the full constructor kwargs for this control at one search point.

        Each `Factory` value in `params` is called with the context of the point, and every other
        value is used unchanged. The chosen values of the point then take precedence over
        `params`.

        Args:
            chosen: The values the search point assigns to the swept arguments.
            context: The sweep context. A copy with `"search_params"` set to `chosen` is passed to
                each `Factory`.

        Returns:
            The constructor kwargs.
        """
        local_context = dict(context)
        local_context["search_params"] = chosen

        resolved_params = {
            key: (value(local_context) if isinstance(value, Factory) else value)
            for key, value in self.params.items()
        }

        resolved_params.update(chosen)
        return resolved_params
