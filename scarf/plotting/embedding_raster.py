"""Blockwise raster embedding for large cell counts."""

from dataclasses import replace
from typing import Any, Hashable

import numpy as np

from ._contracts import CellField, ColorScale, PlotProvenance
from ._deps import require_matplotlib
from ._figure import LegendSpec, PlotResult, normalize_axes_target
from ._raster import draw_raster_canvas, raster_from_metadata
from ._style import apply_figure_chrome, square_axis_limits, theme_context


def _is_categorical_column(
    cells: Any,
    *,
    key: str,
    cell_key: str,
    block_rows: int | None,
    kind: str,
    max_categories: int = 100,
) -> bool:
    if kind == "categorical":
        return True
    if kind == "continuous":
        return False
    dtype = cells.get_dtype(key)
    dtype_kind = getattr(dtype, "kind", None)
    if dtype_kind in ("b", "O", "S", "U", "T"):
        return True
    if dtype_kind not in ("i", "u"):
        return False

    categories: set[Any] = set()
    for block in cells.iter_row_blocks(
        cell_key=cell_key,
        columns=[key],
        block_rows=block_rows,
    ):
        categories.update(np.unique(block.values[key]).tolist())
        if len(categories) > max_categories:
            return False
    return True


def embedding_raster(
    store: Any,
    *,
    layout_key: str,
    color_by: str | CellField | None = None,
    cell_key: str = "I",
    pixels: int = 400,
    block_rows: int | None = None,
    color_scale: ColorScale | None = None,
    missing_color: str = "white",
    subset_by: str | None = None,
    target: Any | None = None,
    figsize: tuple[float, float] | None = None,
    theme: str = "notebook",
    seed: int = 0,
    show: bool = True,
) -> PlotResult:
    """Draw a layout as a pixel image for large cell counts.

    Prefer this over :func:`embedding` when you only need a continuous
    cell-metadata color (for example ``RNA_nCounts``) and the dataset is large
    enough that a full scatter is slow or memory-heavy. Values are read in row
    blocks, so full columns are not loaded at once.

    ``color_by`` must be continuous metadata, not a gene. Pass
    ``CellField(key, kind="continuous")`` when an integer column should be
    treated as continuous. Categorical colors are not supported here.
    Empty pixels use a white background by default (override with
    ``missing_color``). Pass ``subset_by`` to keep cells marked ``True`` in a
    boolean cell-metadata column.
    """
    require_matplotlib()
    color_scale = color_scale or ColorScale(cmap="viridis", quantiles=(0.01, 0.99))
    if color_scale.scale != "linear":
        raise NotImplementedError(
            "embedding_raster currently supports only linear color scales"
        )
    x_key = f"{layout_key}1"
    y_key = f"{layout_key}2"
    for key in (x_key, y_key):
        if key not in store.cells.columns:
            raise KeyError(f"Layout column {key!r} not found in cell metadata")

    color_key: str | None
    color_label: str | None
    if isinstance(color_by, CellField):
        color_key = color_by.key
        color_label = color_by.label or color_by.key
        color_kind = color_by.kind
    else:
        color_key = color_by
        color_label = color_by
        color_kind = "auto"
    if color_key is not None and color_key not in store.cells.columns:
        raise KeyError(
            f"color_by {color_key!r} must be a cell-metadata column for "
            "embedding_raster (gene coloring uses embedding() for now)"
        )
    if color_key is not None and _is_categorical_column(
        store.cells,
        key=color_key,
        cell_key=cell_key,
        block_rows=block_rows,
        kind=color_kind,
    ):
        raise NotImplementedError(
            "embedding_raster supports continuous color values only; use "
            "embedding() for categorical cell metadata"
        )

    if subset_by is not None and subset_by not in store.cells.columns:
        raise KeyError(f"subset_by {subset_by!r} not found in cell metadata")

    quantiles = color_scale.quantiles
    canvas = raster_from_metadata(
        store.cells,
        x_key=x_key,
        y_key=y_key,
        color_key=color_key,
        cell_key=cell_key,
        subset_by=subset_by,
        pixels=pixels,
        block_rows=block_rows,
        quantiles=quantiles,
        seed=seed,
    )
    if color_scale.vmin is not None or color_scale.vmax is not None:
        canvas = replace(
            canvas,
            vmin=(
                float(color_scale.vmin) if color_scale.vmin is not None else canvas.vmin
            ),
            vmax=(
                float(color_scale.vmax) if color_scale.vmax is not None else canvas.vmax
            ),
        )
        if canvas.vmax <= canvas.vmin:
            raise ValueError("Color limits must satisfy vmin < vmax")

    panel_key: Hashable = color_label or layout_key
    resolved_figsize = figsize
    if resolved_figsize is None and target is None:
        resolved_figsize = (5.2, 5.0)
    fig, axes, owns = normalize_axes_target(
        target,
        panel_keys=[panel_key],
        figsize=resolved_figsize,
    )
    ax = axes[panel_key]
    with theme_context(theme):
        # Square the data window first, then draw into that extent so the
        # image fills the axes (no floating grey rectangle in empty margins).
        xlim, ylim = square_axis_limits(
            (canvas.extent[0], canvas.extent[1]),
            (canvas.extent[2], canvas.extent[3]),
        )
        squared = replace(
            canvas,
            extent=(xlim[0], xlim[1], ylim[0], ylim[1]),
        )
        im = draw_raster_canvas(
            ax,
            squared,
            cmap=color_scale.cmap or "viridis",
            missing_color=missing_color,
            vcenter=color_scale.vcenter,
        )
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_box_aspect(1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(f"{layout_key}1")
        ax.set_ylabel(f"{layout_key}2")
        cb = fig.colorbar(
            im,
            ax=ax,
            location="top",
            orientation="horizontal",
            shrink=0.8,
            fraction=0.06,
            pad=0.08,
        )
        cb.set_label(color_label or "log1p cell count")
        apply_figure_chrome(fig, theme)

    result = PlotResult(
        figure=fig,
        axes=axes,
        tables={},
        legends=(
            LegendSpec(
                kind="colorbar",
                label=color_label or "log1p cell count",
            ),
        ),
        scales=(color_scale,),
        provenance=PlotProvenance(
            cell_key=cell_key,
            n_cells=canvas.n_cells,
            renderer="matplotlib-raster",
            notes=(
                "embedding_raster",
                "two_pass",
                f"layout={layout_key}",
                *(
                    ("approximate_quantiles",)
                    if color_key is not None and quantiles is not None
                    else ()
                ),
            ),
            extras={
                "pixels": pixels,
                "block_rows": block_rows,
                "n_blocks": canvas.n_blocks,
                "vmin": canvas.vmin,
                "vmax": canvas.vmax,
                "color_by": color_key,
                "color_mode": "continuous" if color_key is not None else "density",
                "subset_by": subset_by,
                "missing_color": missing_color,
            },
        ),
        owns_figure=owns,
        theme=theme,
    )
    if show:
        result.show()
    return result
