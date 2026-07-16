"""Helpers for legacy DataStore / scarf.plots callers.

These do not change legacy public behavior by default. They provide safe
argument handling and an explicit eligibility check for opt-in bridging.
"""

import logging
from copy import deepcopy
from typing import Any

logger = logging.getLogger("scarf")


def copy_plot_mutables(
    *,
    color_key: dict[Any, Any] | None = None,
    mask_values: list[Any] | None = None,
    scatter_kwargs: dict[str, Any] | None = None,
) -> tuple[dict[Any, Any] | None, list[Any] | None, dict[str, Any] | None]:
    """Return deep copies so legacy helpers never mutate caller inputs."""
    return (
        None if color_key is None else dict(color_key),
        None if mask_values is None else list(mask_values),
        None if scatter_kwargs is None else deepcopy(scatter_kwargs),
    )


def plot_layout_bridge_blockers(
    *,
    layout_key: str | list[str] | None,
    color_by: str | list[str] | None,
    do_shading: bool,
    mask_values: list[Any] | None,
    subset_by: str | None,
    shuffle_df: bool,
    legend_ondata: bool,
    legend_onside: bool,
    force_ints_as_cats: bool,
    clip_fraction: float,
    ax: Any,
    use_plotting: bool = False,
    title: str | list[str] | None = None,
    scatter_kwargs: dict[str, Any] | None = None,
) -> tuple[str, ...]:
    """Return reasons ``plot_layout`` must stay on the legacy renderer.

    With ``use_plotting=False`` (default), bridging is always blocked.
    With ``use_plotting=True``, only hard incompatibilities block the bridge.
    Soft differences (legacy on-data legends, integer-as-category defaults)
    are allowed; ``scarf.plotting.embedding`` uses its own legend and type rules.
    """
    if not use_plotting:
        return ("bridge_not_enabled",)

    blockers: list[str] = []
    if do_shading:
        blockers.append("do_shading")
    if isinstance(layout_key, list) and len(layout_key) > 1:
        blockers.append("multiple_layout_keys")
    if mask_values:
        blockers.append("mask_values")
    if shuffle_df:
        blockers.append("shuffle_df")
    if title is not None:
        blockers.append("title")
    if scatter_kwargs:
        blockers.append("scatter_kwargs")
    # subset_by, clip_fraction, and ax are supported by embedding.
    _ = (subset_by, legend_ondata, legend_onside, force_ints_as_cats, clip_fraction, ax)
    return tuple(dict.fromkeys(blockers))


def embedding_kwargs_from_plot_layout(
    *,
    layout_key: str,
    color_by: Any,
    cell_key: str,
    from_assay: str | None,
    point_size: float,
    point_sizes: Any,
    sort_values: bool,
    cmap: str | None,
    show_fig: bool,
    clip_fraction: float = 0.0,
    subset_by: str | None = None,
    default_color: str = "steelblue",
    missing_color: str = "k",
    target: Any | None = None,
    figsize: tuple[float, float] | None = None,
    n_columns: int | None = None,
    color_key: dict[Any, Any] | None = None,
    legend_loc: str = "auto",
) -> dict[str, Any]:
    """Map a subset of ``plot_layout`` args to ``embedding`` kwargs."""
    from ._contracts import CategoricalScale, ColorScale

    return {
        "layout_key": layout_key if isinstance(layout_key, str) else layout_key[0],
        "color_by": color_by,
        "cell_key": cell_key,
        "from_assay": from_assay,
        "point_size": point_size,
        "point_sizes": point_sizes,
        "sort_values": sort_values,
        "color_scale": ColorScale(cmap=cmap) if cmap is not None else None,
        "categorical_scale": (
            CategoricalScale(palette=dict(color_key)) if color_key is not None else None
        ),
        "clip_fraction": clip_fraction,
        "subset_by": subset_by,
        "default_color": default_color,
        "missing_color": missing_color,
        "target": target,
        "figsize": figsize,
        "n_columns": n_columns,
        "legend_loc": legend_loc,
        "show": show_fig,
    }


def try_bridge_plot_layout(
    store: Any,
    *,
    use_plotting: bool,
    layout_key: str | list[str] | None,
    color_by: str | list[str] | None,
    do_shading: bool,
    mask_values: list[Any] | None,
    subselection_key: str | None,
    shuffle_df: bool,
    legend_ondata: bool,
    legend_onside: bool,
    force_ints_as_cats: bool,
    clip_fraction: float,
    ax: Any,
    cell_key: str,
    from_assay: str | None,
    point_size: float,
    size_vals: Any,
    sort_values: bool,
    cmap: str | None,
    default_color: str,
    mask_color: str,
    width: float,
    height: float,
    n_columns: int,
    show_fig: bool,
    savename: str | None,
    save_dpi: int,
    color_key: dict[Any, Any] | None = None,
    title: str | list[str] | None = None,
    scatter_kwargs: dict[str, Any] | None = None,
) -> tuple[bool, Any]:
    """Attempt opt-in bridge to ``scarf.plotting.embedding``.

    Returns ``(handled, result)``. When ``handled`` is False, callers should use
    the legacy renderer. When True, ``result`` is a ``PlotResult`` or ``None``
    (after ``show_fig``).
    """
    blockers = plot_layout_bridge_blockers(
        layout_key=layout_key,
        color_by=color_by,
        do_shading=do_shading,
        mask_values=mask_values,
        subset_by=subselection_key,
        shuffle_df=shuffle_df,
        legend_ondata=legend_ondata,
        legend_onside=legend_onside,
        force_ints_as_cats=force_ints_as_cats,
        clip_fraction=clip_fraction,
        ax=ax,
        use_plotting=use_plotting,
        title=title,
        scatter_kwargs=scatter_kwargs,
    )
    if blockers:
        if use_plotting:
            logger.warning(
                "plot_layout(use_plotting=True) fell back to the legacy renderer "
                "because: %s",
                ", ".join(blockers),
            )
        return False, None

    if layout_key is None:
        raise ValueError("Please provide a value for `layout_key` parameter.")
    if isinstance(layout_key, list):
        layout_key = layout_key[0]

    color_names = [color_by] if isinstance(color_by, str) else color_by or []
    if cmap is not None and color_key is None:
        categorical_cmap = False
        for color in color_names:
            if color not in store.cells.columns:
                continue
            dtype_kind = getattr(store.cells.get_dtype(color), "kind", None)
            if dtype_kind in ("b", "O", "S", "U", "T") or (
                dtype_kind in ("i", "u") and force_ints_as_cats
            ):
                categorical_cmap = True
                break
        if categorical_cmap:
            logger.warning(
                "plot_layout(use_plotting=True) fell back to the legacy renderer "
                "because categorical cmap translation is not exact"
            )
            return False, None

    if legend_ondata and legend_onside:
        legend_loc = "auto"
    elif legend_ondata:
        legend_loc = "on_data"
    elif legend_onside:
        legend_loc = "right"
    else:
        legend_loc = "auto"

    from .embedding import embedding

    from ._contracts import CellField

    if isinstance(color_by, str):
        bridged_colors: Any = [color_by]
        unwrap_single = True
    elif color_by is None:
        bridged_colors = None
        unwrap_single = False
    else:
        bridged_colors = list(color_by)
        unwrap_single = False
    if bridged_colors is not None:
        for index, color in enumerate(bridged_colors):
            if color not in store.cells.columns:
                continue
            dtype_kind = getattr(store.cells.get_dtype(color), "kind", None)
            if dtype_kind in ("i", "u"):
                bridged_colors[index] = CellField(
                    color,
                    kind=("categorical" if force_ints_as_cats else "continuous"),
                )
        if unwrap_single:
            bridged_colors = bridged_colors[0]

    kwargs = embedding_kwargs_from_plot_layout(
        layout_key=layout_key,
        color_by=bridged_colors,
        cell_key=cell_key,
        from_assay=from_assay,
        point_size=point_size,
        point_sizes=size_vals,
        sort_values=sort_values,
        cmap=cmap,
        show_fig=False,
        clip_fraction=clip_fraction,
        subset_by=subselection_key,
        default_color=default_color,
        missing_color=mask_color,
        target=ax,
        figsize=None if ax is not None else (width, height),
        n_columns=n_columns,
        color_key=color_key,
        legend_loc=legend_loc,
    )
    result = embedding(store, **kwargs)
    if savename is not None:
        result.save(savename, dpi=save_dpi)
    if show_fig:
        result.show()
        return True, None
    return True, result
