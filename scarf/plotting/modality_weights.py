"""WNN modality-weight plots over an explicit embedding."""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from ..storage.artifacts import ArtifactRef, artifact_group, inspect_artifact
from ..storage.types import as_zarr_array
from ._contracts import ColorScale, PlotProvenance
from ._data import _artifact_cell_selection, _resolve_layout
from ._deps import require_matplotlib
from ._figure import LegendSpec, PlotResult, normalize_axes_target
from ._style import (
    DEFAULT_PANEL_INCHES,
    DEFAULT_RASTERIZE_THRESHOLD,
    FrameStyle,
    apply_figure_chrome,
    default_point_edgewidth,
    default_point_size,
    finish_embedding_axes,
    scatter_edgecolor,
    square_axis_limits,
    theme_context,
)


def _artifact_ref(value: Any, *, label: str) -> ArtifactRef:
    if not isinstance(value, Mapping):
        raise ValueError(f"WNN graph has no valid {label} input")
    try:
        return ArtifactRef.from_dict(value)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"WNN graph has no valid {label} input") from exc


def _wnn_assays(
    store: Any,
    graph: ArtifactRef,
) -> tuple[list[str], ArtifactRef]:
    if not isinstance(graph, ArtifactRef):
        raise TypeError("graph must be an ArtifactRef")
    if graph.scope != "datastore" or graph.kind != "integrated_graph":
        raise ValueError("graph must identify a datastore integrated_graph artifact")
    status = inspect_artifact(store.zw, graph)
    if not status.complete:
        raise ValueError("Integrated graph artifact is unavailable or incomplete")
    if status.operation != "integrate_assays":
        raise ValueError("graph must be produced by integrate_assays")
    parameters = status.parameters or {}
    if parameters.get("method") != "wnn":
        raise ValueError("graph must be a WNN integrated graph")

    raw_assays = parameters.get("assays")
    if not isinstance(raw_assays, Sequence) or isinstance(raw_assays, str | bytes):
        raise ValueError("WNN graph provenance has no ordered assay list")
    assays = list(raw_assays)
    if (
        len(assays) < 2
        or any(not isinstance(assay, str) or not assay for assay in assays)
        or len(set(assays)) != len(assays)
    ):
        raise ValueError("WNN graph provenance has an invalid ordered assay list")

    inputs = status.inputs or {}
    expected_source_keys = {f"source_{index}" for index in range(len(assays))}
    source_keys = {key for key in inputs if key.startswith("source_")}
    if source_keys != expected_source_keys:
        raise ValueError("WNN graph sources do not match its ordered assay list")
    for index, assay in enumerate(assays):
        raw_source = inputs[f"source_{index}"]
        if not isinstance(raw_source, Mapping):
            raise ValueError("WNN graph source provenance is malformed")
        neighbors = _artifact_ref(
            raw_source.get("neighbors"),
            label=f"source_{index}.neighbors",
        )
        coordinates = _artifact_ref(
            raw_source.get("coordinates"),
            label=f"source_{index}.coordinates",
        )
        if neighbors.kind != "neighbors" or neighbors.assay != assay:
            raise ValueError("WNN neighbor source order does not match its assays")
        if coordinates.kind not in {"reduction", "batch_correction"}:
            raise ValueError("WNN coordinate source has an invalid artifact kind")
        if coordinates.assay != assay:
            raise ValueError("WNN coordinate source order does not match its assays")

    group = artifact_group(store.zw, graph)
    stored_assays = group.attrs.get("assays")
    if not isinstance(stored_assays, Sequence) or isinstance(
        stored_assays, str | bytes
    ):
        raise ValueError("WNN graph payload has no ordered assay list")
    if list(stored_assays) != assays:
        raise ValueError("WNN graph payload assay order disagrees with provenance")
    return assays, _artifact_cell_selection(store, graph, label="WNN graph")


def _load_weights(
    store: Any,
    graph: ArtifactRef,
    *,
    n_cells: int,
    n_assays: int,
) -> np.ndarray:
    group = artifact_group(store.zw, graph)
    if "modality_weights" not in group:
        raise ValueError("WNN graph has no modality_weights array")
    array = as_zarr_array(group["modality_weights"], name="modality_weights")
    if array.shape != (n_cells, n_assays):
        raise ValueError(
            "WNN modality weights must have one row per selected cell and "
            "one column per assay"
        )
    if np.dtype(array.dtype).kind != "f":
        raise TypeError("WNN modality weights must use a floating-point dtype")
    weights = np.asarray(array[:], dtype=np.float64)
    if not np.isfinite(weights).all():
        raise ValueError("WNN modality weights must be finite")
    if (weights < 0).any():
        raise ValueError("WNN modality weights must be non-negative")
    row_sums = weights.sum(axis=1)
    if not np.allclose(row_sums, 1.0, rtol=1e-6, atol=1e-6):
        raise ValueError("Every WNN modality-weight row must sum to one")
    return weights


def _layout_limits(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[tuple[float, float], tuple[float, float]]:
    x_span = float(x.max() - x.min())
    y_span = float(y.max() - y.min())
    x_pad = 0.05 * (x_span if x_span > 0 else 1.0)
    y_pad = 0.05 * (y_span if y_span > 0 else 1.0)
    return square_axis_limits(
        (float(x.min() - x_pad), float(x.max() + x_pad)),
        (float(y.min() - y_pad), float(y.max() + y_pad)),
    )


def modality_weights(
    store: Any,
    *,
    graph: ArtifactRef,
    layout: ArtifactRef,
    point_size: float | None = None,
    point_alpha: float = 1.0,
    cmap: str = "viridis",
    n_columns: int | None = None,
    target: Any | None = None,
    figsize: tuple[float, float] | None = None,
    theme: str = "notebook",
    frame: FrameStyle = "minimal",
    rasterize_threshold: int = DEFAULT_RASTERIZE_THRESHOLD,
    show: bool = True,
) -> PlotResult:
    """Plot each assay's WNN contribution over an explicit embedding."""
    if point_size is not None and (not np.isfinite(point_size) or point_size <= 0):
        raise ValueError("point_size must be positive and finite")
    if not np.isfinite(point_alpha) or not 0 <= point_alpha <= 1:
        raise ValueError("point_alpha must be between zero and one")
    if not isinstance(cmap, str) or not cmap:
        raise TypeError("cmap must be a non-empty string")
    if (
        isinstance(rasterize_threshold, bool)
        or not isinstance(rasterize_threshold, int)
        or rasterize_threshold < 1
    ):
        raise ValueError("rasterize_threshold must be a positive integer")

    assays, graph_selection = _wnn_assays(store, graph)
    coordinates, cell_indices, layout_selection = _resolve_layout(store, layout)
    if graph_selection != layout_selection:
        raise ValueError(
            "WNN graph and layout must share the exact cell-selection artifact"
        )
    weights = _load_weights(
        store,
        graph,
        n_cells=len(cell_indices),
        n_assays=len(assays),
    )

    x = coordinates[:, 0]
    y = coordinates[:, 1]
    xlim, ylim = _layout_limits(x, y)
    resolved_columns = len(assays) if n_columns is None else n_columns
    if (
        isinstance(resolved_columns, bool)
        or not isinstance(resolved_columns, int)
        or resolved_columns < 1
    ):
        raise ValueError("n_columns must be a positive integer")
    resolved_columns = min(resolved_columns, len(assays))
    if figsize is None and target is None:
        rows = int(np.ceil(len(assays) / resolved_columns))
        figsize = (
            DEFAULT_PANEL_INCHES * resolved_columns,
            DEFAULT_PANEL_INCHES * rows,
        )

    plt, _ = require_matplotlib()
    with theme_context(theme):
        figure, axes, owns_figure = normalize_axes_target(
            target,
            panel_keys=assays,
            figsize=figsize,
            n_columns=resolved_columns,
        )
        resolved_size = (
            default_point_size(len(cell_indices))
            if point_size is None
            else float(point_size)
        )
        edgewidth = default_point_edgewidth(
            len(cell_indices),
            point_size=resolved_size,
        )
        edgecolors = "none" if edgewidth == 0 else scatter_edgecolor(theme)
        rasterized = len(cell_indices) >= rasterize_threshold
        for index, assay in enumerate(assays):
            axis = axes[assay]
            artist = axis.scatter(
                x,
                y,
                c=weights[:, index],
                cmap=plt.get_cmap(cmap),
                vmin=0.0,
                vmax=1.0,
                s=resolved_size,
                alpha=point_alpha,
                linewidths=edgewidth,
                edgecolors=edgecolors,
                rasterized=rasterized,
            )
            colorbar = figure.colorbar(
                artist,
                ax=axis,
                location="top",
                orientation="horizontal",
                shrink=0.8,
                fraction=0.06,
                pad=0.02,
            )
            colorbar.set_label(f"{assay} weight")
            finish_embedding_axes(
                axis,
                xlim=xlim,
                ylim=ylim,
                title=assay,
                frame=frame,
            )
        apply_figure_chrome(figure, theme)

    table = pd.DataFrame(
        weights,
        index=pd.Index(cell_indices, name="cell_index"),
        columns=assays,
    )
    result = PlotResult(
        figure=figure,
        axes=axes,
        tables={"weights": table},
        legends=(
            LegendSpec(
                kind="colorbar",
                label="Modality weight",
                scale_key="modality_weight",
                extras={"assays": assays},
            ),
        ),
        scales=(ColorScale(cmap=cmap, vmin=0.0, vmax=1.0, scope="shared"),),
        provenance=PlotProvenance(
            n_cells=len(cell_indices),
            renderer="matplotlib",
            notes=("modality_weights", "artifact"),
            extras={
                "graph": graph.to_dict(),
                "layout": layout.to_dict(),
                "cell_selection": graph_selection.to_dict(),
                "assays": assays,
                "rasterize_threshold": rasterize_threshold,
            },
        ),
        owns_figure=owns_figure,
        theme=theme,
    )
    if show:
        result.show()
    return result
