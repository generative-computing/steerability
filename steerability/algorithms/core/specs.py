"""
Specification utilities for steering controls.

Provides:

- `ControlSpec`: a description of a steering control plus a hyperparameter search space.
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


@dataclass(slots=True)
class ControlSpec:
    """Specification for a parameterized steering control.

    A `ControlSpec` describes a control class plus a search space over its constructor arguments. The sweep layer
    (`core/sweeps.py`, and `SteeringEval` above it) uses it to instantiate control instances for different
    hyperparameter settings.

    Attributes:
        control_cls: The steering control class to instantiate.
        params: Fixed constructor arguments for the control.
        vars: Optional search space over additional constructor arguments. May be:

            - mapping (cartesian grid)
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

            if any(len(vals) == 0 for vals in param_values):
                return

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
        """Compute the full kwargs for this control at a given search point.
        """
        local_context = dict(context)
        local_context["search_params"] = chosen

        resolved_params = {
            key: (value(local_context) if callable(value) else value)
            for key, value in self.params.items()
        }

        resolved_params.update(chosen)
        return resolved_params
