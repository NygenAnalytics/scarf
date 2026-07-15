"""Diagnostic plot compatibility wrappers."""

from typing import Any

import numpy as np
import pandas as pd


def qc(
    data: pd.DataFrame,
    color: str = "steelblue",
    cmap: str = "tab20",
    figsize: tuple[float, float] | None = None,
    label_size: float = 10.0,
    title_size: float = 10,
    sup_title: str | None = None,
    sup_title_size: float = 12,
    scatter_size: float = 1.0,
    max_points: int = 10_000,
    show_on_single_row: bool = True,
    show: bool = True,
) -> Any | None:
    """Facade for ``scarf.plots.plot_qc``."""
    from ..plots import plot_qc

    return plot_qc(
        data=data,
        color=color,
        cmap=cmap,
        fig_size=figsize,
        label_size=label_size,
        title_size=title_size,
        sup_title=sup_title,
        sup_title_size=sup_title_size,
        scatter_size=scatter_size,
        max_points=max_points,
        show_on_single_row=show_on_single_row,
        show_fig=show,
    )


def elbow(
    variance_explained: np.ndarray | list[float],
    figsize: tuple[float | None, float] = (None, 2),
) -> None:
    """Facade for ``scarf.plots.plot_elbow``."""
    from ..plots import plot_elbow

    return plot_elbow(variance_explained, figsize=figsize)


def graph_qc(graph: Any) -> None:
    """Facade for ``scarf.plots.plot_graph_qc``."""
    from ..plots import plot_graph_qc

    return plot_graph_qc(graph)


def highly_variable_features(
    mean_nonzero: np.ndarray,
    corrected_variance: np.ndarray,
    n_cells: np.ndarray,
    selected: np.ndarray,
    *,
    label_size: float = 12,
    figsize: tuple[float, float] = (4.5, 4.0),
    point_sizes: tuple[float, float] = (3, 30),
    colormaps: tuple[str, str] = ("winter", "magma_r"),
) -> None:
    """Plot diagnostics for highly variable feature selection."""
    from ..plots import plot_mean_var

    plot_mean_var(
        nzm=mean_nonzero,
        fv=corrected_variance,
        n_cells=n_cells,
        hvg=selected,
        ax_label_fs=label_size,
        fig_size=figsize,
        ss=point_sizes,
        cmaps=colormaps,
    )
