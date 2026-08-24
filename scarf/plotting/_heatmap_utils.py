"""Shared ordering and annotation helpers for heatmaps."""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from ._contracts import CategoricalScale
from ._style import categorical_color_map, sort_categories


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


def _finite_linkage_values(values: np.ndarray) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64).copy()
    if np.isfinite(data).all():
        return data
    column_means = np.nanmean(data, axis=0)
    column_means = np.nan_to_num(column_means, nan=0.0)
    missing_row, missing_column = np.where(~np.isfinite(data))
    data[missing_row, missing_column] = column_means[missing_column]
    return data


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
    from scipy.cluster.hierarchy import leaves_list, linkage

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
        row_linkage = linkage(
            _finite_linkage_values(matrix.to_numpy()),
            method=method,
            metric=metric,
            optimal_ordering=True,
        )
        resolved_rows = [matrix.index[index] for index in leaves_list(row_linkage)]
    if explicit_columns is not None:
        resolved_columns = explicit_columns
    elif cluster_columns and matrix.shape[1] > 1:
        column_linkage = linkage(
            _finite_linkage_values(matrix.to_numpy().T),
            method=method,
            metric=metric,
            optimal_ordering=True,
        )
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
        observed = [value for value in pd.unique(values) if pd.notna(value)]
        scale = scales.get(name) if scales is not None else None
        if scale is not None and scale.order is not None:
            missing = [value for value in observed if value not in scale.order]
            if missing:
                raise ValueError(
                    f"annotation scale {name!r} is missing values: "
                    + ", ".join(map(str, missing[:10]))
                )
            order = [value for value in scale.order if value in set(observed)]
        else:
            order = sort_categories(observed)
        palette = categorical_color_map(
            order,
            palette=scale.palette if scale is not None else None,
            palette_name=scale.palette_name if scale is not None else "default",
        )
        missing_color = scale.missing_color if scale is not None else "#bdbdbd"
        colors[name] = [
            missing_color if pd.isna(value) else palette[value] for value in values
        ]
        resolved_scales.append(
            CategoricalScale(
                order=tuple(order),
                palette=palette,
                labels=scale.labels if scale is not None else None,
                missing_color=missing_color,
                missing_label=scale.missing_label if scale is not None else "NA",
                palette_name=(scale.palette_name if scale is not None else "default"),
            )
        )
    return colors, resolved_scales


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
