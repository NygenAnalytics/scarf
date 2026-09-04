"""Blockwise raster embedding for large cell counts."""

from collections.abc import Iterable, Iterator
from dataclasses import replace
from typing import Any, Hashable, cast

import numpy as np

from ..metadata import MetaDataRowBlock
from ..metadata.rows import (
    read_metadata_missing_rows_chunkwise,
    read_metadata_rows_chunkwise,
)
from ..storage import ArtifactRef
from ..storage.artifacts import artifact_group
from ..storage.selections import (
    iter_stored_selection_blocks,
    validate_stored_selection_integrity,
)
from ..storage.types import as_zarr_array
from ._contracts import CellField, ColorScale, PlotProvenance
from ._data import _validated_embedding_selection
from ._deps import require_matplotlib
from ._display import stored_display_metadata
from ._figure import LegendSpec, PlotResult, normalize_axes_target
from ._raster import (
    _apply_raster_missing_mask,
    draw_raster_canvas,
    raster_from_metadata,
)
from ._style import apply_figure_chrome, theme_context


_ARTIFACT_X = "__scarf_artifact_embedding_x"
_ARTIFACT_Y = "__scarf_artifact_embedding_y"


class _ArtifactRasterCells:
    """Present artifact coordinates and aligned live fields as bounded blocks."""

    __slots__ = (
        "_cells",
        "_coordinates",
        "_root",
        "_selection",
    )

    def __init__(
        self,
        root: Any,
        cells: Any,
        coordinates: Any,
        selection: ArtifactRef,
    ) -> None:
        self._root = root
        self._cells = cells
        self._coordinates = coordinates
        self._selection = selection

    @property
    def columns(self) -> list[str]:
        return [*self._cells.columns, _ARTIFACT_X, _ARTIFACT_Y]

    @property
    def field_source(self) -> str:
        return (
            "frozen_run"
            if callable(getattr(self._cells, "_iter_selected_blocks", None))
            else "live_metadata"
        )

    def get_dtype(self, column: str) -> np.dtype[Any]:
        if column in {_ARTIFACT_X, _ARTIFACT_Y}:
            return cast(np.dtype[Any], np.dtype(self._coordinates.dtype))
        get_dtype = getattr(self._cells, "get_dtype", None)
        if callable(get_dtype):
            return cast(np.dtype[Any], np.dtype(get_dtype(column)))
        field_dtype = getattr(self._cells, "_field_dtype", None)
        if not callable(field_dtype):
            raise TypeError("Raster cell view cannot report field dtypes")
        return cast(np.dtype[Any], np.dtype(field_dtype(column)))

    def iter_row_blocks(
        self,
        *,
        cell_key: str = "I",
        columns: Iterable[str] | None = None,
        block_rows: int | None = None,
    ) -> Iterator[MetaDataRowBlock]:
        if cell_key != "I":
            raise ValueError(
                "cell_key cannot override an artifact's stored cell selection"
            )
        requested = list(columns or ())
        unknown = [column for column in requested if column not in self.columns]
        if unknown:
            raise KeyError(f"Raster fields were not found: {unknown!r}")
        resolved_rows = None if block_rows is None else int(block_rows)
        if resolved_rows is not None and resolved_rows < 1:
            raise ValueError("block_rows must be >= 1")

        metadata_columns = [
            column for column in requested if column not in {_ARTIFACT_X, _ARTIFACT_Y}
        ]
        frozen_iterator = getattr(self._cells, "_iter_selected_blocks", None)
        if callable(frozen_iterator):
            source_blocks = frozen_iterator(metadata_columns, resolved_rows)
        else:
            source_blocks = iter_stored_selection_blocks(
                self._root,
                self._selection,
                kind="cell_selection",
                scope="datastore",
                assay=None,
                table_path="cellData",
                block_rows=resolved_rows,
            )

        compact_start = 0
        for block in source_blocks:
            if isinstance(block, MetaDataRowBlock):
                row_indices = block.active_global_indices.astype(
                    np.int64,
                    copy=False,
                )
            else:
                row_indices = block.selected_indices.astype(
                    np.int64,
                    copy=False,
                )
            compact_stop = compact_start + len(row_indices)
            coordinates = np.asarray(self._coordinates[compact_start:compact_stop])
            if coordinates.shape != (len(row_indices), 2):
                raise ValueError(
                    "Embedding coordinates changed while they were being read"
                )
            if not np.isfinite(coordinates).all():
                raise ValueError("Embedding coordinates must be finite")
            values: dict[str, np.ndarray] = {}
            for column in requested:
                if column == _ARTIFACT_X:
                    values[column] = coordinates[:, 0]
                elif column == _ARTIFACT_Y:
                    values[column] = coordinates[:, 1]
                elif isinstance(block, MetaDataRowBlock):
                    values[column] = block.values[column]
                else:
                    live_values = read_metadata_rows_chunkwise(
                        self._cells,
                        column,
                        row_indices,
                    )
                    missing = read_metadata_missing_rows_chunkwise(
                        self._cells,
                        column,
                        row_indices,
                    )
                    values[column] = _apply_raster_missing_mask(
                        live_values,
                        missing,
                    )
            yield MetaDataRowBlock(
                start=block.start,
                stop=block.stop,
                active_global_indices=row_indices,
                values=values,
            )
            compact_start = compact_stop


def _resolve_artifact_raster_cells(
    store: Any,
    layout: ArtifactRef,
) -> tuple[_ArtifactRasterCells, ArtifactRef]:
    """Validate one embedding artifact without materializing its payload."""
    selection = _validated_embedding_selection(store, layout)
    view_selection = getattr(store.cells, "_selection_ref", selection)
    if view_selection != selection:
        raise ValueError(
            "Frozen run fields and embedding must share the same cell selection"
        )
    validated = validate_stored_selection_integrity(
        store.zw,
        selection,
        kind="cell_selection",
        scope="datastore",
        assay=None,
        table_path="cellData",
    )
    group = artifact_group(store.zw, layout)
    if "values" not in group:
        raise ValueError("Embedding artifact has no canonical values array")
    try:
        coordinates = as_zarr_array(group["values"], name="values")
    except TypeError as exc:
        raise ValueError("Embedding artifact values are malformed") from exc
    if coordinates.shape != (validated.selected_count, 2):
        raise ValueError(
            "Embedding must have two columns and one row per selected cell"
        )
    if np.dtype(coordinates.dtype).kind not in {"f", "i", "u"}:
        raise TypeError("Embedding coordinates must be numeric")
    return (
        _ArtifactRasterCells(
            store.zw,
            store.cells,
            coordinates,
            selection,
        ),
        selection,
    )


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
    layout_key: str | None = None,
    layout: ArtifactRef | None = None,
    color_by: str | CellField | None = None,
    cell_key: str = "I",
    pixels: int = 400,
    block_rows: int | None = None,
    color_scale: ColorScale | None = None,
    missing_color: str | None = None,
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

    Provide exactly one coordinate source. ``layout`` consumes an immutable
    embedding artifact and its stored cell selection. ``layout_key`` reads two
    explicit live metadata columns named ``{layout_key}1`` and
    ``{layout_key}2``.

    ``color_by`` must be continuous metadata, not a gene. Pass
    ``CellField(key, kind="continuous")`` when an integer column should be
    treated as continuous. Categorical colors are not supported here.
    Empty pixels use a white background by default. Set
    ``ColorScale.missing_color`` or use ``missing_color`` as an explicit
    override. Pass ``subset_by`` to keep cells marked ``True`` in a boolean
    cell-metadata column.
    """
    require_matplotlib()
    if (layout_key is None) == (layout is None):
        raise ValueError("Provide exactly one of layout_key or layout")
    layout_selection: ArtifactRef | None = None
    if layout is None:
        assert layout_key is not None
        raster_cells = store.cells
        x_key = f"{layout_key}1"
        y_key = f"{layout_key}2"
        layout_name = layout_key
        layout_source = "live_metadata"
        field_source = "live_metadata" if color_by is not None or subset_by else None
        for key in (x_key, y_key):
            if key not in raster_cells.columns:
                raise KeyError(f"Layout column {key!r} not found in cell metadata")
    else:
        if cell_key != "I":
            raise ValueError(
                "cell_key cannot override an artifact's stored cell selection"
            )
        raster_cells, layout_selection = _resolve_artifact_raster_cells(
            store,
            layout,
        )
        x_key = _ARTIFACT_X
        y_key = _ARTIFACT_Y
        layout_name = "Embedding"
        layout_source = "artifact"
        field_source = (
            raster_cells.field_source if color_by is not None or subset_by else None
        )

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
    if color_key is not None and color_key not in raster_cells.columns:
        raise KeyError(
            f"color_by {color_key!r} must be a cell-metadata column for "
            "embedding_raster (gene coloring uses embedding() for now)"
        )
    stored_display = (
        None if color_key is None else stored_display_metadata(store, color_key)
    )
    if stored_display is not None and color_kind == "auto":
        color_kind = (
            "continuous" if stored_display["kind"] == "continuous" else "categorical"
        )
    if color_key is not None and _is_categorical_column(
        raster_cells,
        key=color_key,
        cell_key=cell_key,
        block_rows=block_rows,
        kind=color_kind,
    ):
        raise NotImplementedError(
            "embedding_raster supports continuous color values only; use "
            "embedding() for categorical cell metadata"
        )

    if color_scale is None:
        if stored_display is not None and stored_display["kind"] == "continuous":
            minimum = stored_display["minimum"]
            maximum = stored_display["maximum"]
            fixed_limits = (
                minimum is not None
                and maximum is not None
                and float(maximum) > float(minimum)
            )
            color_scale = ColorScale(
                cmap=str(stored_display["colormap"]),
                vmin=float(minimum) if fixed_limits else None,
                vmax=float(maximum) if fixed_limits else None,
                missing_color="white",
                scale=str(stored_display["scale"]),  # type: ignore[arg-type]
            )
        else:
            color_scale = ColorScale(
                cmap="viridis",
                quantiles=(0.01, 0.99),
                missing_color="white",
            )
    if missing_color is not None:
        color_scale = replace(color_scale, missing_color=missing_color)
    effective_missing_color = color_scale.missing_color
    if color_scale.scale != "linear":
        raise NotImplementedError(
            "embedding_raster currently supports only linear color scales"
        )

    if subset_by is not None and subset_by not in raster_cells.columns:
        raise KeyError(f"subset_by {subset_by!r} not found in cell metadata")

    quantiles = color_scale.quantiles
    canvas = raster_from_metadata(
        raster_cells,
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

    panel_key: Hashable = color_label or layout_name
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
        xlim = (canvas.extent[0], canvas.extent[1])
        ylim = (canvas.extent[2], canvas.extent[3])
        im = draw_raster_canvas(
            ax,
            canvas,
            cmap=color_scale.cmap or "viridis",
            missing_color=effective_missing_color,
            vcenter=color_scale.vcenter,
        )
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_box_aspect(1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(f"{layout_name}1")
        ax.set_ylabel(f"{layout_name}2")
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
            assay=None if layout is None else layout.assay,
            cell_key=cell_key if layout is None else None,
            n_cells=canvas.n_cells,
            renderer="matplotlib-raster",
            notes=(
                "embedding_raster",
                "two_pass",
                f"{layout_source}_layout",
                *((f"{field_source}_fields",) if field_source is not None else ()),
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
                "layout": None if layout is None else layout.to_dict(),
                "cell_selection": (
                    None if layout_selection is None else layout_selection.to_dict()
                ),
                "vmin": canvas.vmin,
                "vmax": canvas.vmax,
                "color_by": color_key,
                "color_mode": "continuous" if color_key is not None else "density",
                "subset_by": subset_by,
                "layout_source": layout_source,
                "field_source": field_source,
                "missing_color": effective_missing_color,
            },
        ),
        owns_figure=owns,
        theme=theme,
    )
    if show:
        result.show()
    return result
