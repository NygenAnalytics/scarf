"""Shared ordering and annotation helpers for heatmaps."""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from ._contracts import CategoricalScale
from ._style import category_label, resolve_category_scale


def _explicit_order(
    labels: Sequence[Any],
    requested: Sequence[Any] | None,
    *,
    axis_name: str,
) -> list[Any] | None:
    if requested is None:
        return None
    order = list(requested)
    if len(order) != len(set(order)):
        raise ValueError(f"{axis_name}_order cannot contain duplicates")
    observed = list(labels)
    missing = [label for label in observed if label not in order]
    unexpected = [label for label in order if label not in observed]
    if missing or unexpected:
        details = []
        if missing:
            details.append("missing: " + ", ".join(map(str, missing[:10])))
        if unexpected:
            details.append("unexpected: " + ", ".join(map(str, unexpected[:10])))
        raise ValueError(
            f"{axis_name}_order must contain every observed label ("
            + "; ".join(details)
            + ")"
        )
    return order


# SciPy's hierarchical clustering methods; the last three need Euclidean data.
_LINKAGE_METHODS = (
    "single",
    "complete",
    "average",
    "weighted",
    "centroid",
    "median",
    "ward",
)
_EUCLIDEAN_LINKAGE_METHODS = frozenset({"centroid", "median", "ward"})


def validate_linkage(method: str, metric: str) -> None:
    """Reject a clustering method that is undefined for the distance metric."""
    if method not in _LINKAGE_METHODS:
        raise ValueError(
            "cluster_method must be one of "
            + ", ".join(repr(name) for name in _LINKAGE_METHODS)
            + f", not {method!r}"
        )
    if method in _EUCLIDEAN_LINKAGE_METHODS and metric != "euclidean":
        raise ValueError(
            f"cluster_method={method!r} requires cluster_metric='euclidean', "
            f"not {metric!r}. Use cluster_method='average' or 'complete' with "
            f"{metric!r} distances"
        )


def _finite_linkage_values(values: np.ndarray) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64).copy()
    if np.isfinite(data).all():
        return data
    column_means = np.nanmean(data, axis=0)
    column_means = np.nan_to_num(column_means, nan=0.0)
    missing_row, missing_column = np.where(~np.isfinite(data))
    data[missing_row, missing_column] = column_means[missing_column]
    return data


def _linkage(values: np.ndarray, *, method: str, metric: str) -> np.ndarray:
    """Hierarchically cluster rows, treating undefined distances as the largest.

    Missing values are imputed, but infinite values are rejected. Correlation
    and cosine distances are undefined for constant or all-zero rows, for
    example a feature expressed in no group.
    """
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import pdist

    validate_linkage(method, metric)
    data = np.asarray(values, dtype=np.float64)
    if np.isinf(data).any():
        raise ValueError("Heatmap clustering requires finite values")
    distances = pdist(_finite_linkage_values(data), metric=metric)
    undefined = np.isnan(distances)
    if undefined.any():
        defined = distances[~undefined]
        distances[undefined] = float(defined.max()) if defined.size else 1.0
    return np.asarray(linkage(distances, method=method, optimal_ordering=True))


def order_heatmap(
    matrix: pd.DataFrame,
    *,
    row_order: Sequence[Any] | None,
    column_order: Sequence[Any] | None,
    cluster_rows: bool,
    cluster_columns: bool,
    method: str,
    metric: str,
) -> tuple[pd.DataFrame, np.ndarray | None, np.ndarray | None]:
    from scipy.cluster.hierarchy import leaves_list

    explicit_rows = _explicit_order(
        list(matrix.index),
        row_order,
        axis_name="row",
    )
    explicit_columns = _explicit_order(
        list(matrix.columns),
        column_order,
        axis_name="column",
    )
    row_linkage = None
    column_linkage = None
    resolved_rows = list(matrix.index)
    resolved_columns = list(matrix.columns)
    if explicit_rows is not None:
        resolved_rows = explicit_rows
    elif cluster_rows and matrix.shape[0] > 1:
        row_linkage = _linkage(matrix.to_numpy(), method=method, metric=metric)
        resolved_rows = [matrix.index[index] for index in leaves_list(row_linkage)]
    if explicit_columns is not None:
        resolved_columns = explicit_columns
    elif cluster_columns and matrix.shape[1] > 1:
        column_linkage = _linkage(matrix.to_numpy().T, method=method, metric=metric)
        resolved_columns = [
            matrix.columns[index] for index in leaves_list(column_linkage)
        ]
    return (
        matrix.reindex(index=resolved_rows, columns=resolved_columns),
        row_linkage,
        column_linkage,
    )


def normalize_annotations(
    labels: Sequence[Any],
    annotations: Mapping[
        str,
        Mapping[Any, Any] | Sequence[Any],
    ]
    | None,
    *,
    axis_name: str,
) -> pd.DataFrame:
    index = pd.Index(labels)
    if annotations is None:
        return pd.DataFrame(index=index)
    columns: dict[str, list[Any]] = {}
    for name, values in annotations.items():
        if isinstance(values, Mapping):
            missing = [label for label in index if label not in values]
            if missing:
                raise ValueError(
                    f"{axis_name} annotation {name!r} is missing labels: "
                    + ", ".join(map(str, missing[:10]))
                )
            columns[name] = [values[label] for label in index]
        else:
            resolved = list(values)
            if len(resolved) != len(index):
                raise ValueError(
                    f"{axis_name} annotation {name!r} must have {len(index)} values"
                )
            columns[name] = resolved
    return pd.DataFrame(columns, index=index)


def annotation_colors(
    annotations: pd.DataFrame,
    scales: Mapping[str, CategoricalScale] | None,
) -> tuple[pd.DataFrame, list[CategoricalScale]]:
    colors = pd.DataFrame(index=annotations.index)
    resolved_scales: list[CategoricalScale] = []
    for name in annotations:
        values = annotations[name].to_numpy(dtype=object)
        resolved = resolve_category_scale(
            values,
            scales.get(name) if scales is not None else None,
            context=f"annotation_scales[{name!r}]",
        )
        palette = resolved.palette or {}
        colors[name] = [
            resolved.missing_color if pd.isna(value) else palette[value]
            for value in values
        ]
        resolved_scales.append(resolved)
    return colors, resolved_scales


def annotation_legend_handles(
    mpl: Any,
    names: Sequence[str],
    scales: Sequence[CategoricalScale],
) -> list[Any]:
    """Square legend handles naming each annotation value and its color."""
    handles: list[Any] = []
    for name, scale in zip(names, scales, strict=True):
        if scale.order is None or scale.palette is None:
            continue
        handles.extend(
            mpl.lines.Line2D(
                [],
                [],
                marker="s",
                linestyle="",
                markerfacecolor=scale.palette[value],
                markeredgecolor="none",
                markersize=5,
                label=f"{name}: {category_label(scale, value)}",
            )
            for value in scale.order
        )
    return handles


def draw_annotation_strips(
    ax: Any,
    *,
    row_colors: pd.DataFrame,
    column_colors: pd.DataFrame,
    n_rows: int,
    n_columns: int,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Draw annotation strips and return the expanded data limits they require."""
    from matplotlib.patches import Rectangle

    tick_labels = ax.get_yticklabels() or ax.get_xticklabels()
    label_size = tick_labels[0].get_fontsize() if tick_labels else 8.0
    row_width = max(0.18, n_columns * 0.025)
    column_height = max(0.18, n_rows * 0.025)
    row_label_y = -0.5 - len(column_colors.columns) * column_height - 0.08
    for annotation_index, name in enumerate(row_colors):
        left = -0.5 - (annotation_index + 1) * row_width
        for row_index, color in enumerate(row_colors[name]):
            ax.add_patch(
                Rectangle(
                    (left, row_index - 0.5),
                    row_width,
                    1,
                    facecolor=color,
                    edgecolor="none",
                    clip_on=False,
                )
            )
        ax.text(
            left + row_width / 2,
            row_label_y,
            name,
            rotation=90,
            ha="center",
            va="bottom",
            fontsize=label_size * 0.9,
            clip_on=False,
        )
    for annotation_index, name in enumerate(column_colors):
        top = -0.5 - (annotation_index + 1) * column_height
        for column_index, color in enumerate(column_colors[name]):
            ax.add_patch(
                Rectangle(
                    (column_index - 0.5, top),
                    1,
                    column_height,
                    facecolor=color,
                    edgecolor="none",
                    clip_on=False,
                )
            )
        ax.text(
            n_columns - 0.4,
            top + column_height / 2,
            name,
            ha="left",
            va="center",
            fontsize=label_size * 0.9,
            clip_on=False,
        )
    return (
        (-0.5 - len(row_colors.columns) * row_width, n_columns - 0.5),
        (
            n_rows - 0.5,
            -0.5 - len(column_colors.columns) * column_height,
        ),
    )
