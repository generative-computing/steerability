"""Visualization utilities for benchmark profiles."""

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from aisteer360.utils.optional import require

require("matplotlib")
import matplotlib.pyplot as plt

_COLOR_CYCLE = [
    "#348ABD",
    "#E24A33",
    "#988ED5",
    "#777777",
    "#FBC15E",
    "#8EBA42",
    "#FFB5B8",
]

# grey color for axis labels, ticks, and secondary text
AXIS_GREY = "#555555"

# double-ring marker shapes cycled per group in plot_tradeoff_scatter
_DOUBLE_RING_SHAPES = ["o", "s", "^", "h", "D", "v"]

# markers for fixed (non-swept) reference pipelines overlaid on scatter-type plots (plot_tradeoff, plot_tradeoff_scatter)
_FIXED_PIPELINE_MARKERS = [
    {"marker": "X", "color": "black"},  # baseline
    {"marker": "s", "color": "#E24A33"},  # red square
    {"marker": "D", "color": "#348ABD"},  # blue diamond
    {"marker": "^", "color": "#988ED5"},  # purple triangle
    {"marker": "P", "color": "#8EBA42"},  # green plus
    {"marker": "v", "color": "#FBC15E"},  # amber down-triangle
]

# line styles for fixed reference pipelines overlaid on sensitivity plots
_FIXED_PIPELINE_STYLES = [
    {"color": "#555555", "linestyle": "--"},  # baseline grey
    {"color": "#E24A33", "linestyle": ":"},  # red
    {"color": "#348ABD", "linestyle": "-."},  # blue
    {"color": "#988ED5", "linestyle": ":"},  # purple
    {"color": "#8EBA42", "linestyle": "-."},  # green
    {"color": "#FBC15E", "linestyle": ":"},  # amber
]


def apply_plot_style() -> None:
    """Apply specific matplotlib rcParams for scientific style."""
    plt.rcParams.update({
        # fonts: prefer clean sans-serif
        "font.family": "sans-serif",
        "font.sans-serif": ["Roboto", "DejaVu Sans", "Arial", "sans-serif"],
        "font.size": 10,
        "axes.titlesize": "medium",
        "axes.labelsize": "medium",

        # spines & ticks: show all four sides
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.linewidth": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "lines.linewidth": 1.5,
        "lines.markersize": 6,

        # grid: subtle and in the background
        "axes.grid": False,
        "grid.color": "#cbcbcb",
        "grid.linestyle": ":",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.5,

        # colors
        "axes.prop_cycle": plt.cycler(color=_COLOR_CYCLE),
    })


def _clean_axes(ax: plt.Axes) -> None:
    """Ensure all four axis spines are visible and apply grey styling."""
    for spine in ("right", "top", "left", "bottom"):
        ax.spines[spine].set_visible(True)

    ax.xaxis.label.set_color(AXIS_GREY)
    ax.yaxis.label.set_color(AXIS_GREY)
    ax.tick_params(axis="both", colors=AXIS_GREY)


def _draw_error_bars(
    ax: plt.Axes,
    df: pd.DataFrame,
    x_mean_col: str,
    y_mean_col: str,
    x_std_col: str | None = None,
    y_std_col: str | None = None,
    **kwargs: Any,
) -> None:
    """Draw thin black error bars for every row in `df`.

    Bars are drawn as thin black lines with small end-caps, positioned behind
    the data points.

    Args:
        ax: Axes to draw on.
        df: DataFrame whose rows supply coordinates and error magnitudes.
        x_mean_col: Column name for x centre values.
        y_mean_col: Column name for y centre values.
        x_std_col: Column name for x error magnitudes (omitted when ``None``).
        y_std_col: Column name for y error magnitudes (omitted when ``None``).
        **kwargs: Overrides for the default ``ax.errorbar`` keyword arguments.
    """
    defaults: dict[str, Any] = {
        "fmt": "none",
        "ecolor": "black",
        "elinewidth": 0.5,
        "capsize": 2,
        "capthick": 0.5,
        "zorder": 2,
    }
    defaults.update(kwargs)

    for _, row in df.iterrows():
        ax.errorbar(
            row[x_mean_col],
            row[y_mean_col],
            xerr=row.get(x_std_col, 0) if x_std_col else None,
            yerr=row.get(y_std_col, 0) if y_std_col else None,
            **defaults,
        )


def _draw_double_ring(
    ax: plt.Axes,
    x: np.ndarray | Sequence[float],
    y: np.ndarray | Sequence[float],
    marker: str = "o",
    fill_color: str | np.ndarray | None = None,
    cmap: str | None = None,
    outer_s: float = 120,
    inner_s: float = 60,
    center_s: float = 50,
    label: str | None = None,
    zorder_base: int = 3,
    fill: bool = True,
    **center_kwargs: Any,
) -> plt.matplotlib.collections.PathCollection | None:
    """Draw double-ring markers at the given coordinates.

    Renders three scatter layers per point: an outer ring, an inner ring, and
    optionally a color-filled centre. All three use the same `marker` shape so
    the visual generalises from circles to squares, triangles, hexagons, etc.

    Args:
        ax: Axes to draw on.
        x: X coordinates.
        y: Y coordinates.
        marker: Marker shape.
        fill_color: Color for centre fill (ignored when ``fill=False``).
        cmap: Colormap name when `fill_color` is numeric.
        outer_s: Size of outer ring.
        inner_s: Size of inner ring.
        center_s: Size of centre marker.
        label: Legend label.
        zorder_base: Base z-order for layering.
        fill: Whether to fill the centre of the markers. When ``False``, only
            the double-ring outline is drawn.
        **center_kwargs: Extra kwargs for the centre scatter call.

    Returns:
        The centre ``PathCollection`` when `fill_color` is a numeric array
        (useful for creating a colorbar), otherwise ``None``.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    # outer ring
    ax.scatter(x, y, s=outer_s, marker=marker, facecolor="none",
               edgecolor="black", linewidth=1, zorder=zorder_base)
    # inner ring
    ax.scatter(x, y, s=inner_s, marker=marker, facecolor="none",
               edgecolor="black", linewidth=0.5, zorder=zorder_base + 1)

    if not fill:
        # draw an invisible scatter for the label only
        if label is not None:
            ax.scatter([], [], s=center_s, marker=marker, facecolor="none",
                       edgecolor="black", linewidth=1, label=label)
        return None

    # centre fill
    center_kw: dict[str, Any] = {
        "s": center_s,
        "marker": marker,
        "edgecolors": "none",
        "zorder": zorder_base + 2,
    }
    center_kw.update(center_kwargs)

    is_numeric_array = (
        isinstance(fill_color, np.ndarray)
        or (isinstance(fill_color, (list, tuple))
            and len(fill_color) > 0
            and not isinstance(fill_color[0], str))
    )

    if is_numeric_array:
        center_kw["c"] = fill_color
        if cmap is not None:
            center_kw["cmap"] = cmap
        return ax.scatter(x, y, label=label, **center_kw)
    else:
        center_kw["color"] = fill_color if fill_color is not None else _COLOR_CYCLE[0]
        ax.scatter(x, y, label=label, **center_kw)
        return None


def _style_colorbar(
    cbar: plt.colorbar,
    values: np.ndarray | None = None,
) -> None:
    """Apply the standard AXIS_GREY styling to a colorbar.

    When `values` is provided and all entries are integer-like with ≤10 unique
    values, the colorbar ticks are snapped to those discrete values.
    """
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(size=0, colors=AXIS_GREY)
    cbar.ax.yaxis.label.set_color(AXIS_GREY)

    if values is not None:
        unique_vals = np.unique(values)
        is_discrete = all(float(v).is_integer() for v in unique_vals)
        if is_discrete and len(unique_vals) <= 10:
            cbar.set_ticks(unique_vals)
            cbar.set_ticklabels([str(int(v)) for v in unique_vals])


def _build_refs_list(
    baseline: pd.DataFrame | pd.Series | None = None,
    compare_to_pipelines: list[tuple[str, pd.DataFrame]] | None = None,
    baseline_label: str = "baseline",
) -> list[tuple[str, pd.DataFrame]]:
    """Build and validate a merged list of fixed reference pipelines.

    Handles both the deprecated ``baseline`` / ``baseline_row`` parameter
    (prepended with `baseline_label`) and the newer ``compare_to_pipelines``
    list.  Each entry is validated to contain at most one configuration.

    Args:
        baseline: Optional baseline data, either a ``pd.DataFrame`` (one or more
            rows sharing a single config) or a ``pd.Series`` (single row,
            converted to a one-row DataFrame).
        compare_to_pipelines: Optional list of ``(label, summary_df)`` tuples.
        baseline_label: Legend label for the baseline entry.

    Returns:
        List of ``(label, DataFrame)`` tuples ready for rendering.

    Raises:
        ValueError: If any entry contains multiple distinct ``config_id`` values.
    """
    all_refs: list[tuple[str, pd.DataFrame]] = []

    if baseline is not None:
        if isinstance(baseline, pd.Series):
            baseline = pd.DataFrame([baseline])
        if not baseline.empty:
            all_refs.append((baseline_label, baseline))

    if compare_to_pipelines:
        for label, ref_df in compare_to_pipelines:
            if ref_df is not None and not ref_df.empty and "config_id" in ref_df.columns:
                n_configs = ref_df["config_id"].nunique()
                if n_configs > 1:
                    raise ValueError(
                        f"compare_to_pipelines entry '{label}' contains {n_configs} "
                        f"configurations. Only fixed (non-swept) pipelines are "
                        f"supported — ControlSpec sweeps with multiple configurations "
                        f"should be plotted as a separate swept series."
                    )
            all_refs.append((label, ref_df))

    return all_refs


def _draw_ref_scatter_markers(
    ax: plt.Axes,
    all_refs: list[tuple[str, pd.DataFrame]],
    x_mean_col: str,
    y_mean_col: str,
    x_std_col: str,
    y_std_col: str,
) -> None:
    """Draw fixed reference pipelines as distinct shaped markers with error bars.

    Uses ``_FIXED_PIPELINE_MARKERS`` to cycle through marker shapes and colors.
    Each reference pipeline is rendered as a single prominent marker with thin
    error bars, at a high z-order so it sits on top of the main data series.
    """
    for i, (label, ref_df) in enumerate(all_refs):
        if ref_df is None or ref_df.empty:
            continue
        style = _FIXED_PIPELINE_MARKERS[i % len(_FIXED_PIPELINE_MARKERS)]
        brow = ref_df.iloc[0]
        bx, by = brow[x_mean_col], brow[y_mean_col]
        bx_err = brow.get(x_std_col, 0)
        by_err = brow.get(y_std_col, 0)
        ax.errorbar(
            bx, by,
            xerr=bx_err, yerr=by_err,
            fmt="none", ecolor="black", elinewidth=0.5,
            capsize=2, capthick=0.5, zorder=6,
        )
        ax.scatter(
            bx, by,
            marker=style["marker"], s=100, c=style["color"],
            linewidths=1.0, edgecolors="white", zorder=7, label=label,
        )


def _draw_ref_hlines(
    ax: plt.Axes,
    all_refs: list[tuple[str, pd.DataFrame]],
    metric: str,
) -> None:
    """Draw fixed reference pipelines as horizontal lines with ±1 std bands.

    Uses ``_FIXED_PIPELINE_STYLES`` to cycle through line styles and colors.
    Intended for sensitivity-style plots where the x-axis is the swept parameter
    rather than a second metric.
    """
    for i, (label, ref_df) in enumerate(all_refs):
        if ref_df is None or ref_df.empty:
            continue
        style = _FIXED_PIPELINE_STYLES[i % len(_FIXED_PIPELINE_STYLES)]
        ref_val = ref_df[f"{metric}_mean"].iloc[0]
        ref_std = ref_df[f"{metric}_std"].iloc[0]
        ax.axhline(ref_val, linewidth=1.5, label=label, **style)
        ax.axhspan(ref_val - ref_std, ref_val + ref_std,
                    color=style["color"], alpha=0.1)


def _compute_pareto_points(
    summary: pd.DataFrame,
    x_metric: str,
    y_metric: str,
    maximize_x: bool = True,
    maximize_y: bool = True,
) -> list[tuple[float, float]]:
    """Compute Pareto-optimal points from summary data.

    A point is Pareto-optimal if no other point dominates it (i.e., no other
    point is at least as good in both dimensions and strictly better in at
    least one).

    Returns:
        List of ``(x, y)`` tuples representing the Pareto frontier, sorted by x.
    """
    x_mean = f"{x_metric}_mean"
    y_mean = f"{y_metric}_mean"

    points = [(row[x_mean], row[y_mean]) for _, row in summary.iterrows()]

    pareto_points = []
    for px, py in points:
        dominated = False
        for qx, qy in points:
            qx_better = (qx > px) if maximize_x else (qx < px)
            qy_better = (qy > py) if maximize_y else (qy < py)
            qx_ge = (qx >= px) if maximize_x else (qx <= px)
            qy_ge = (qy >= py) if maximize_y else (qy <= py)

            if qx_ge and qy_ge and (qx_better or qy_better):
                dominated = True
                break
        if not dominated:
            pareto_points.append((px, py))

    pareto_points.sort(key=lambda p: p[0])
    return pareto_points


def _overlay_pareto_frontier(
    ax: plt.Axes,
    summary: pd.DataFrame,
    x_metric: str,
    y_metric: str,
    maximize_x: bool = True,
    maximize_y: bool = True,
    label: str | None = None,
    frontier_style: dict[str, Any] | None = None,
) -> list[tuple[float, float]]:
    """Compute and draw the Pareto frontier on the given axes.

    Args:
        ax: Matplotlib axes to draw on.
        summary: DataFrame with ``{x_metric}_mean`` and ``{y_metric}_mean``.
        x_metric: Metric for x-axis.
        y_metric: Metric for y-axis.
        maximize_x: Whether higher x values are better.
        maximize_y: Whether higher y values are better.
        label: Optional legend label for the frontier line.
        frontier_style: Dict of style kwargs for the line.  Defaults to thick
            semi-transparent black.

    Returns:
        List of ``(x, y)`` tuples representing the Pareto frontier points.
    """
    if frontier_style is None:
        frontier_style = {
            "color": "black",
            "linestyle": "-",
            "linewidth": 3,
            "alpha": 0.3,
            "zorder": 2,
        }

    pareto_points = _compute_pareto_points(
        summary, x_metric, y_metric, maximize_x, maximize_y,
    )

    if pareto_points:
        pareto_x, pareto_y = zip(*pareto_points)
        ax.plot(pareto_x, pareto_y, label=label, **frontier_style)

        # midpoint "frontier" annotation
        mid_idx = len(pareto_x) // 2
        ax.annotate(
            "frontier",
            (pareto_x[mid_idx], pareto_y[mid_idx]),
            xytext=(8, -8),
            textcoords="offset points",
            fontsize=8,
            color=AXIS_GREY,
            alpha=0.8,
        )

    return pareto_points


## PUBLIC PLOTTING FUNCTIONS

def plot_metric_by_config(
    summary: pd.DataFrame,
    metric: str,
    x_col: str = "config_id",
    baseline_value: float | None = None,
    baseline_std: float | None = None,
    ax: plt.Axes | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    save_path: str | Path | None = None,
    **errorbar_kwargs: Any,
) -> plt.Axes:
    """Plot a metric with error bars across configurations."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    _clean_axes(ax)

    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"

    defaults = {
        "fmt": "o-",
        "capsize": 0,
        "elinewidth": 1,
        "markersize": 6,
        "markeredgewidth": 1,
        "markeredgecolor": "white",
        "zorder": 3,
    }
    defaults.update(errorbar_kwargs)

    ax.errorbar(summary[x_col], summary[mean_col], yerr=summary[std_col], **defaults)

    if baseline_value is not None:
        ax.axhline(baseline_value, color="#444444", linestyle="--",
                    linewidth=1, label="Baseline", zorder=1)
        if baseline_std is not None:
            ax.axhspan(baseline_value - baseline_std,
                        baseline_value + baseline_std,
                        color="#999999", alpha=0.1, edgecolor="none", zorder=0)

    ax.set_xlabel(xlabel or x_col)
    ax.set_ylabel(ylabel or metric)
    if title:
        ax.set_title(title, loc="left", fontweight="medium", fontsize=10)
    ax.grid(True, axis="y", zorder=-1)
    ax.legend(frameon=False, loc="best")

    if save_path is not None:
        ax.get_figure().savefig(save_path, bbox_inches="tight", dpi=150)

    return ax


def plot_tradeoff_scatter(
    summary: pd.DataFrame,
    x_metric: str,
    y_metric: str,
    group_col: str | None = None,
    group_order: Sequence[str] | None = None,
    color_col: str | None = None,
    label_col: str | None = None,
    label_points: str = "all",
    baseline_row: pd.Series | None = None,
    compare_to_pipelines: list[tuple[str, pd.DataFrame]] | None = None,
    per_trial_data: pd.DataFrame | None = None,
    ax: plt.Axes | None = None,
    title: str = "metric tradeoff",
    xlabel: str | None = None,
    ylabel: str | None = None,
    cmap: str = "viridis",
    show_pareto: bool = False,
    maximize_x: bool = True,
    maximize_y: bool = True,
    fill: bool = True,
    save_path: str | Path | None = None,
    **scatter_kwargs: Any,
) -> plt.Axes:
    """Plot a scatter of two metrics showing their tradeoff.

    Displays summary configurations as double-ring scatter points with thin
    black error bars.  When ``group_col`` is provided the data is split into
    groups, each rendered with a distinct double-ring marker shape (circle,
    square, triangle, hexagon, ...) and its own color from the standard color
    cycle so that different pipelines are visually distinguishable.

    Fixed (non-swept) reference pipelines can be overlaid as distinct solid
    markers using ``compare_to_pipelines`` (or the deprecated ``baseline_row``
    parameter).  A Pareto frontier can optionally be shown.

    Args:
        summary: DataFrame with metric columns (``{metric}_mean``, ``{metric}_std``).
        x_metric: Metric for x-axis.
        y_metric: Metric for y-axis.
        group_col: Optional column used to split ``summary`` into groups.  Each
            group receives a unique double-ring marker shape and color, and
            appears in the legend.  When ``color_col`` is also provided, the
            color fill comes from the colormap instead but shapes still
            differentiate groups.
        group_order: Optional sequence specifying the order of groups in the
            legend.  If ``None``, groups appear in their first-occurrence order
            in the DataFrame.  Values in ``group_order`` that are not present
            in the data are silently ignored.
        color_col: Optional column for color-coding points via a colormap.
        label_col: Optional column for text annotations next to each point.
        label_points: Which points to label when ``label_col`` is provided.
            ``"all"`` labels every point; ``"frontier"`` labels only Pareto-
            optimal points.  Defaults to ``"all"``.
        baseline_row: Deprecated.  A single ``pd.Series`` to plot as a
            reference marker.  Prefer ``compare_to_pipelines``.
        compare_to_pipelines: Optional list of ``(label, summary_df)`` tuples
            for fixed reference pipelines rendered with distinct shaped markers.
        per_trial_data: Optional DataFrame with per-trial values for a small-dot
            scatter overlay.
        ax: Matplotlib axes to plot on.  If ``None``, a new figure is created.
        title: Plot title.
        xlabel: Label for x-axis.  Defaults to ``x_metric``.
        ylabel: Label for y-axis.  Defaults to ``y_metric``.
        cmap: Colormap name when ``color_col`` is used.
        show_pareto: Whether to overlay the Pareto frontier.
        maximize_x: Whether higher x values are better (for Pareto).
        maximize_y: Whether higher y values are better (for Pareto).
        fill: Whether to fill the centre of the double-ring markers. When
            ``False``, only the outline rings are drawn.
        save_path: Optional path to save the figure (150 dpi).
        **scatter_kwargs: Extra kwargs forwarded to the centre scatter call of
            each double-ring group.

    Returns:
        The matplotlib axes with the plot.

    Raises:
        ValueError: If a ``compare_to_pipelines`` entry has multiple configs.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    _clean_axes(ax)

    x_mean = f"{x_metric}_mean"
    y_mean = f"{y_metric}_mean"
    x_std = f"{x_metric}_std"
    y_std = f"{y_metric}_std"

    # determine groups
    if group_col is not None and group_col in summary.columns:
        if group_order is not None:
            # use specified order, filtering to values present in data
            present = set(summary[group_col].unique())
            group_keys = [g for g in group_order if g in present]
        else:
            group_keys = list(summary[group_col].unique())
    else:
        group_keys = [None]

    # error bars (behind everything)
    _draw_error_bars(ax, summary, x_mean, y_mean, x_std, y_std)

    # per-trial scatter overlay (50 % opacity)
    if per_trial_data is not None and x_metric in per_trial_data.columns and y_metric in per_trial_data.columns:
        for gi, gkey in enumerate(group_keys):
            if gkey is not None and group_col is not None and group_col in per_trial_data.columns:
                mask = per_trial_data[group_col] == gkey
            else:
                mask = pd.Series(True, index=per_trial_data.index)
            ax.scatter(
                per_trial_data.loc[mask, x_metric].values,
                per_trial_data.loc[mask, y_metric].values,
                s=12, color=_COLOR_CYCLE[gi % len(_COLOR_CYCLE)],
                alpha=0.5, zorder=2,
            )

    # double-ring markers per group
    colorbar_scatter = None

    for gi, gkey in enumerate(group_keys):
        grp = summary[summary[group_col] == gkey] if gkey is not None else summary
        gx = grp[x_mean].values
        gy = grp[y_mean].values
        marker = _DOUBLE_RING_SHAPES[gi % len(_DOUBLE_RING_SHAPES)]
        group_color = _COLOR_CYCLE[gi % len(_COLOR_CYCLE)]
        grp_label = str(gkey) if gkey is not None else None

        if color_col is not None and color_col in grp.columns:
            sc = _draw_double_ring(
                ax, gx, gy, marker=marker,
                fill_color=grp[color_col].values, cmap=cmap,
                label=grp_label, fill=fill, **scatter_kwargs,
            )
            if sc is not None and colorbar_scatter is None:
                colorbar_scatter = sc
        else:
            _draw_double_ring(
                ax, gx, gy, marker=marker,
                fill_color=group_color, label=grp_label, fill=fill, **scatter_kwargs,
            )

    # colorbar
    if colorbar_scatter is not None:
        cbar = plt.colorbar(colorbar_scatter, ax=ax, label=color_col)
        _style_colorbar(cbar, values=summary[color_col].values)

    # fixed reference pipelines
    all_refs = _build_refs_list(baseline_row, compare_to_pipelines)
    _draw_ref_scatter_markers(ax, all_refs, x_mean, y_mean, x_std, y_std)

    # axes dressing
    ax.set_xlabel(xlabel or x_metric)
    ax.set_ylabel(ylabel or y_metric)
    ax.set_title(title, loc="left", fontweight="medium", fontsize=10)
    ax.grid(True, zorder=-1)

    # Pareto frontier
    pareto_points: list[tuple[float, float]] = []
    if show_pareto or label_points == "frontier":
        pareto_parts = [summary] + [
            ref_df for _, ref_df in all_refs if ref_df is not None and not ref_df.empty
        ]
        pareto_points = _compute_pareto_points(
            pd.concat(pareto_parts, ignore_index=True),
            x_metric, y_metric,
            maximize_x=maximize_x, maximize_y=maximize_y,
        )
    if show_pareto and pareto_points:
        frontier_style = {
            "color": "black",
            "linestyle": "-",
            "linewidth": 3,
            "alpha": 0.3,
            "zorder": 2,
        }
        pareto_x, pareto_y = zip(*pareto_points)
        ax.plot(pareto_x, pareto_y, **frontier_style)
        mid_idx = len(pareto_x) // 2
        ax.annotate(
            "frontier",
            (pareto_x[mid_idx], pareto_y[mid_idx]),
            xytext=(8, -8),
            textcoords="offset points",
            fontsize=8,
            color=AXIS_GREY,
            alpha=0.8,
        )

    # text labels
    if label_col is not None and label_col in summary.columns:
        pareto_set = set(pareto_points)
        for _, row in summary.iterrows():
            point = (row[x_mean], row[y_mean])
            if label_points == "frontier" and point not in pareto_set:
                continue
            ax.annotate(
                str(row[label_col]), point,
                textcoords="offset points", xytext=(5, 5),
                fontsize=8, color="#333333", alpha=0.8,
            )

    if len(group_keys) > 1 or all_refs:
        handles, labels = ax.get_legend_handles_labels()
        if group_order is not None and handles:
            # reorder legend entries according to group_order
            label_to_handle = dict(zip(labels, handles))
            ordered_handles = []
            ordered_labels = []
            for name in group_order:
                if name in label_to_handle:
                    ordered_handles.append(label_to_handle.pop(name))
                    ordered_labels.append(name)
            # append any remaining entries not in group_order
            for lbl, hdl in label_to_handle.items():
                ordered_handles.append(hdl)
                ordered_labels.append(lbl)
            ax.legend(ordered_handles, ordered_labels, frameon=False, loc="best", fontsize=8)
        else:
            ax.legend(frameon=False, loc="best", fontsize=8)

    if save_path is not None:
        ax.get_figure().savefig(save_path, bbox_inches="tight", dpi=150)

    return ax


def plot_metric_heatmap(
    pivot_df: pd.DataFrame,
    ax: plt.Axes | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    annot: bool = True,
    fmt: str = ".2f",
    cmap: str = "RdYlGn",
    vmin: float | None = None,
    vmax: float | None = None,
    cbar_label: str | None = None,
    square: bool = False,
    col_label_decimals: int | None = 2,
    save_path: str | Path | None = None,
) -> plt.Axes:
    """Plot a heatmap from a pivoted DataFrame.

    Args:
        pivot_df: Pivoted DataFrame with values to plot.
        ax: Matplotlib axes to plot on.  If ``None``, a new figure is created.
        title: Title for the plot.
        xlabel: Label for x-axis.
        ylabel: Label for y-axis.
        annot: Whether to annotate cells with values.
        fmt: Format string for annotations.
        cmap: Colormap name.
        vmin: Minimum value for colormap.
        vmax: Maximum value for colormap.
        cbar_label: Label for colorbar.
        square: Whether to enforce square cells.
        col_label_decimals: Decimal places for rounding column labels (``None``
            to disable).
        save_path: Optional path to save the figure (150 dpi).

    Returns:
        The matplotlib axes with the heatmap.
    """
    sns = require("seaborn")

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    plot_df = pivot_df.copy()
    if col_label_decimals is not None:
        plot_df.columns = [
            round(c, col_label_decimals) if isinstance(c, float) else c
            for c in plot_df.columns
        ]

    cbar_kws = {"label": cbar_label} if cbar_label else {}

    sns.heatmap(
        plot_df, annot=annot, fmt=fmt, cmap=cmap, ax=ax,
        vmin=vmin, vmax=vmax, cbar_kws=cbar_kws,
        square=square, linewidths=0.5, linecolor="white",
    )

    if title:
        ax.set_title(title, loc="left", fontweight="medium", fontsize=10)
    if xlabel:
        ax.set_xlabel(xlabel, color=AXIS_GREY)
    if ylabel:
        ax.set_ylabel(ylabel, color=AXIS_GREY)
    ax.tick_params(axis="both", colors=AXIS_GREY)

    if ax.collections:
        _style_colorbar(ax.collections[0].colorbar)

    if save_path is not None:
        ax.get_figure().savefig(save_path, bbox_inches="tight", dpi=150)

    return ax


def plot_comparison_bars(
    comparison_df: pd.DataFrame,
    metric_cols: Sequence[str],
    group_col: str,
    ax: plt.Axes | None = None,
    title: str | None = None,
    ylabel: str = "Value",
    colors: Sequence[str] | None = None,
    bar_width: float = 0.35,
    save_path: str | Path | None = None,
) -> plt.Axes:
    """Plot grouped bar chart comparing metrics across groups."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    _clean_axes(ax)

    n_groups = len(comparison_df)
    n_metrics = len(metric_cols)
    x = np.arange(n_groups)

    if colors is None:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    total_width = bar_width * n_metrics
    offsets = np.linspace(
        -total_width / 2 + bar_width / 2,
        total_width / 2 - bar_width / 2,
        n_metrics,
    )

    for i, (col, offset) in enumerate(zip(metric_cols, offsets)):
        ax.bar(
            x + offset, comparison_df[col], bar_width,
            label=col, color=colors[i % len(colors)], edgecolor="none", zorder=3,
        )

    ax.axhline(0, color="#333333", linewidth=1, zorder=4)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df[group_col], rotation=0, ha="center")

    if title:
        ax.set_title(title, loc="left", fontweight="medium", fontsize=10)
    ax.grid(True, axis="y", zorder=0)
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1, 1.1), ncol=n_metrics)

    if save_path is not None:
        ax.get_figure().savefig(save_path, bbox_inches="tight", dpi=150)

    return ax


def plot_sensitivity(
    swept: pd.DataFrame,
    metric: str,
    sweep_col: str,
    baseline: pd.DataFrame | None = None,
    compare_to_pipelines: list[tuple[str, pd.DataFrame]] | None = None,
    per_trial_data: pd.DataFrame | None = None,
    ax: plt.Axes | None = None,
    metric_label: str | None = None,
    sweep_label: str | None = None,
    title: str | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    save_path: str | Path | None = None,
) -> plt.Axes:
    """Plot a single metric's sensitivity to a swept parameter.

    Displays the swept configurations as connected points with error bars and
    optional per-trial scatter.  Fixed (non-swept) pipelines are overlaid as
    horizontal reference lines with ±1 std shaded bands.

    Args:
        swept: DataFrame of swept configurations with ``{metric}_mean`` and
            ``{metric}_std`` columns.
        metric: Metric name (used to find ``{metric}_mean`` / ``{metric}_std``).
        sweep_col: Column name for the swept parameter (x-axis).
        baseline: Optional baseline DataFrame.  Deprecated in favour of
            ``compare_to_pipelines``.
        compare_to_pipelines: Optional list of ``(label, summary_df)`` tuples
            for fixed pipelines to overlay as horizontal reference lines.
        per_trial_data: Optional per-trial DataFrame for scatter overlay.
        ax: Matplotlib axes.  If ``None``, a new figure is created.
        metric_label: Y-axis label.  Defaults to `metric`.
        sweep_label: X-axis label.  Defaults to `sweep_col`.
        title: Plot title.  Defaults to ``"{metric_label} sensitivity"``.
        xlim: Optional ``(min, max)`` for x-axis limits.
        ylim: Optional ``(min, max)`` for y-axis limits.
        save_path: Optional path to save the figure (150 dpi).

    Returns:
        The matplotlib axes with the plot.

    Raises:
        ValueError: If a ``compare_to_pipelines`` entry has multiple configs.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))

    _clean_axes(ax)

    metric_label = metric_label or metric
    sweep_label = sweep_label or sweep_col
    title = title or f"{metric_label} sensitivity"

    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    x_vals = swept[sweep_col].values
    y_vals = swept[mean_col].values
    y_err = swept[std_col].values

    # thin black dashed line connecting points
    ax.plot(x_vals, y_vals, linestyle="--", linewidth=0.5, color="black", zorder=2)

    # per-trial scatter
    if per_trial_data is not None and metric in per_trial_data.columns:
        for x_val in x_vals:
            trial_vals = per_trial_data.loc[per_trial_data[sweep_col] == x_val, metric].values
            ax.scatter(
                np.full(len(trial_vals), x_val), trial_vals,
                s=12, color="black", zorder=2, alpha=0.5,
            )

    # error bars
    ax.errorbar(
        x_vals, y_vals, yerr=y_err,
        fmt="none", ecolor="black", elinewidth=0.5,
        capsize=2, capthick=0.5, zorder=3,
    )

    # double-ring markers
    _draw_double_ring(
        ax, x_vals, y_vals, marker="o",
        fill_color="white", zorder_base=4,
    )

    # fixed reference pipelines
    all_refs = _build_refs_list(baseline, compare_to_pipelines)
    _draw_ref_hlines(ax, all_refs, metric)

    ax.set_xlabel(sweep_label)
    ax.set_ylabel(metric_label)
    ax.set_title(title, loc="left", fontweight="medium", fontsize=10)
    ax.grid(True, axis="both", zorder=-1)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if all_refs:
        ax.legend(frameon=False, loc="best", fontsize=8)

    if save_path is not None:
        ax.get_figure().savefig(save_path, bbox_inches="tight", dpi=150)

    return ax


def plot_tradeoff(
    swept: pd.DataFrame,
    x_metric: str,
    y_metric: str,
    sweep_col: str,
    baseline: pd.DataFrame | None = None,
    compare_to_pipelines: list[tuple[str, pd.DataFrame]] | None = None,
    per_trial_data: pd.DataFrame | None = None,
    ax: plt.Axes | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    sweep_label: str | None = None,
    title: str = "tradeoff",
    cmap: str = "magma",
    show_pareto: bool = True,
    maximize_x: bool = True,
    maximize_y: bool = True,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    save_path: str | Path | None = None,
) -> plt.Axes:
    """Plot a tradeoff scatter with optional Pareto frontier overlay.

    Displays swept configurations as color-coded scatter points with error
    bars.  Fixed (non-swept) pipelines are overlaid as distinct markers with
    error bars and included in the Pareto frontier computation.

    Args:
        swept: DataFrame of swept configurations with metric columns.
        x_metric: Metric for x-axis (``{x_metric}_mean`` / ``{x_metric}_std``).
        y_metric: Metric for y-axis (``{y_metric}_mean`` / ``{y_metric}_std``).
        sweep_col: Column for color-coding points.
        baseline: Optional baseline DataFrame.  Deprecated in favour of
            ``compare_to_pipelines``.
        compare_to_pipelines: Optional ``(label, summary_df)`` list for fixed
            pipelines to overlay as distinct markers.
        per_trial_data: Optional per-trial DataFrame for scatter overlay.
        ax: Matplotlib axes.  If ``None``, a new figure is created.
        x_label: X-axis label.  Defaults to `x_metric`.
        y_label: Y-axis label.  Defaults to `y_metric`.
        sweep_label: Colorbar label.  Defaults to `sweep_col`.
        title: Plot title.
        cmap: Colormap for scatter points.
        show_pareto: Whether to overlay the Pareto frontier.
        maximize_x: Whether higher x values are better (for Pareto).
        maximize_y: Whether higher y values are better (for Pareto).
        xlim: Optional ``(min, max)`` for x-axis.
        ylim: Optional ``(min, max)`` for y-axis.
        save_path: Optional path to save the figure (150 dpi).

    Returns:
        The matplotlib axes with the plot.

    Raises:
        ValueError: If a ``compare_to_pipelines`` entry has multiple configs.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    _clean_axes(ax)

    x_label = x_label or x_metric
    y_label = y_label or y_metric
    sweep_label = sweep_label or sweep_col

    x_mean = f"{x_metric}_mean"
    y_mean = f"{y_metric}_mean"
    x_std = f"{x_metric}_std"
    y_std = f"{y_metric}_std"
    x_vals = swept[x_mean].values
    y_vals = swept[y_mean].values
    c_vals = swept[sweep_col].values

    # error bars
    _draw_error_bars(ax, swept, x_mean, y_mean, x_std, y_std)

    # per-trial scatter overlay
    if per_trial_data is not None and x_metric in per_trial_data.columns and y_metric in per_trial_data.columns:
        cmap_obj = plt.get_cmap(cmap)
        norm = plt.Normalize(vmin=c_vals.min(), vmax=c_vals.max())
        for sweep_val in c_vals:
            mask = per_trial_data[sweep_col] == sweep_val
            ax.scatter(
                per_trial_data.loc[mask, x_metric].values,
                per_trial_data.loc[mask, y_metric].values,
                s=12, color=cmap_obj(norm(sweep_val)), zorder=2, alpha=0.5,
            )

    # double-ring markers with colormapped fill
    scatter = _draw_double_ring(
        ax, x_vals, y_vals, marker="o",
        fill_color=c_vals, cmap=cmap,
    )

    # fixed reference pipelines
    all_refs = _build_refs_list(baseline, compare_to_pipelines)
    _draw_ref_scatter_markers(ax, all_refs, x_mean, y_mean, x_std, y_std)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title, loc="left", fontweight="medium", fontsize=10)
    ax.grid(True, zorder=-1)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    # colorbar
    if scatter is not None:
        cbar = plt.colorbar(scatter, ax=ax, label=sweep_label)
        _style_colorbar(cbar, values=c_vals)

    # Pareto frontier
    if show_pareto:
        pareto_parts = [swept] + [
            ref_df for _, ref_df in all_refs if ref_df is not None and not ref_df.empty
        ]
        _overlay_pareto_frontier(
            ax, pd.concat(pareto_parts, ignore_index=True),
            x_metric, y_metric,
            maximize_x=maximize_x, maximize_y=maximize_y,
        )

    if all_refs:
        ax.legend(frameon=False, loc="best", fontsize=8)

    if save_path is not None:
        ax.get_figure().savefig(save_path, bbox_inches="tight", dpi=150)

    return ax


def create_tradeoff_figure(
    summary: pd.DataFrame,
    x_metric: str,
    y_metric: str,
    sweep_col: str,
    baseline_pipeline: str = "baseline",
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    sweep_label: str | None = None,
    figsize: tuple[float, float] = (12, 4),
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Create a multi-panel tradeoff figure with sensitivity and tradeoff plots.

    Creates a figure with three panels:
        1. Left: x_metric sensitivity to sweep_col
        2. Centre: y_metric sensitivity to sweep_col
        3. Right: x_metric vs y_metric tradeoff scatter

    Args:
        summary: DataFrame with metric columns and `sweep_col`.
        x_metric: Metric for x-axis of tradeoff and first sensitivity panel.
        y_metric: Metric for y-axis of tradeoff and second sensitivity panel.
        sweep_col: Column name for the swept parameter.
        baseline_pipeline: Name of the baseline pipeline in the ``"pipeline"``
            column.
        title: Optional overall figure title.
        x_label: Label for `x_metric`.
        y_label: Label for `y_metric`.
        sweep_label: Label for `sweep_col`.
        figsize: Figure size as ``(width, height)``.
        save_path: Optional path to save the figure (150 dpi).

    Returns:
        The matplotlib Figure with three panels.
    """
    x_label = x_label or x_metric
    y_label = y_label or y_metric
    sweep_label = sweep_label or sweep_col

    baseline = (
        summary[summary["pipeline"] == baseline_pipeline]
        if "pipeline" in summary.columns else pd.DataFrame()
    )
    swept = (
        summary[summary["pipeline"] != baseline_pipeline]
        if "pipeline" in summary.columns else summary
    )

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    plot_sensitivity(
        swept, metric=x_metric, sweep_col=sweep_col, baseline=baseline,
        ax=axes[0], metric_label=x_label, sweep_label=sweep_label,
        title=f"{x_label} sensitivity",
    )
    plot_sensitivity(
        swept, metric=y_metric, sweep_col=sweep_col, baseline=baseline,
        ax=axes[1], metric_label=y_label, sweep_label=sweep_label,
        title=f"{y_label} sensitivity",
    )
    plot_tradeoff(
        swept, x_metric=x_metric, y_metric=y_metric, sweep_col=sweep_col,
        baseline=baseline, ax=axes[2], x_label=x_label, y_label=y_label,
        sweep_label=sweep_label, title="tradeoff",
    )

    if title:
        fig.suptitle(title, fontweight="medium", fontsize=12)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)

    return fig


def plot_pareto_frontier(
    summary: pd.DataFrame,
    x_metric: str,
    y_metric: str,
    ax: plt.Axes | None = None,
    maximize_x: bool = True,
    maximize_y: bool = True,
    frontier_style: dict[str, Any] | None = None,
    save_path: str | Path | None = None,
) -> tuple[plt.Axes, list[tuple[float, float]]]:
    """Overlay Pareto frontier on an existing or new scatter plot.

    Computes the Pareto-optimal points and draws the frontier line with a
    midpoint ``"frontier"`` annotation.

    Args:
        summary: DataFrame with ``{x_metric}_mean`` and ``{y_metric}_mean``.
        x_metric: Metric for x-axis.
        y_metric: Metric for y-axis.
        ax: Matplotlib axes.  If ``None``, a new figure is created.
        maximize_x: Whether higher x values are better.
        maximize_y: Whether higher y values are better.
        frontier_style: Style kwargs for the frontier line.
        save_path: Optional path to save the figure (150 dpi).

    Returns:
        Tuple of ``(axes, list of Pareto frontier points as (x, y) tuples)``.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
        _clean_axes(ax)

    pareto_points = _overlay_pareto_frontier(
        ax, summary, x_metric, y_metric,
        maximize_x=maximize_x, maximize_y=maximize_y,
        label="pareto frontier", frontier_style=frontier_style,
    )

    if pareto_points:
        ax.legend(frameon=False, loc="best")

    if save_path is not None:
        ax.get_figure().savefig(save_path, bbox_inches="tight", dpi=150)

    return ax, pareto_points
