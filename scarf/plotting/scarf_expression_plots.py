"""Expression violin plots and faceted box-plot grids for Scarf DataStores.

These two entry points are thin, opinionated facades over the same scarf-native
data layer used by the rest of :mod:`scarf.plotting`. They build a long-form
pandas DataFrame straight from a :class:`~scarf.datastore.datastore.DataStore`
and render it with matplotlib/seaborn. No scanpy, anndata, or Seurat objects
are involved anywhere in the pipeline.

- :func:`violinplot` draws one violin panel per gene, coloured by a grouping
  metadata column, with jittered points overlaid and a full-width dark title
  banner across the top of the figure.
- :func:`boxplot` draws one gene block per gene; each block is a facet grid
  over a second metadata column (for example datasets or conditions) holding
  side-by-side box plots per group, a lettered label beneath every facet, and
  a single consolidated legend outside the axes on the right.
- :func:`stacked_violin` stacks one slim violin row per gene with zero spacing
  between rows; each violin's colour encodes the mean expression of that gene
  within the group (shared scale, colorbar on the right), gene names sit in a
  left-hand column, and the group categories line up down the stack, in a
  minimalist Scanpy-like style.
"""

from collections.abc import Hashable, Mapping, Sequence
from typing import Any, cast

import numpy as np
import pandas as pd

from ._contracts import (
    CategoricalScale,
    ColorScale,
    FeatureRef,
    NormalizationSpec,
    PlotProvenance,
)
from ._data import (
    fetch_normalized_feature_matrix,
    resolve_cell_selection,
    resolve_feature,
)
from ._deps import require_matplotlib, require_seaborn
from ._figure import LegendSpec, PlotResult
from ._style import (
    apply_figure_chrome,
    categorical_color_map,
    sort_categories,
    theme_context,
)


def _scarf_version() -> str:
    try:
        from importlib.metadata import version

        return version("scarf")
    except Exception:
        return "unknown"


def _coerce_normalization(
    normalization: NormalizationSpec | Sequence[str] | None,
) -> NormalizationSpec:
    if normalization is None:
        return NormalizationSpec()
    if isinstance(normalization, NormalizationSpec):
        return normalization
    try:
        source, transform = normalization  # type: ignore[misc]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "normalization must be a NormalizationSpec or a (source, transform) pair"
        ) from exc
    return NormalizationSpec(source=cast(Any, source), transform=cast(Any, transform))


def _resolve_genes(
    store: Any,
    features: str | FeatureRef | Sequence[str | FeatureRef],
    *,
    from_assay: str | None,
) -> list[Any]:
    if isinstance(features, (str, FeatureRef)):
        feature_list = [features]
    else:
        feature_list = list(features)
    if not feature_list:
        raise ValueError("features must be non-empty")
    resolved = [resolve_feature(store, f, from_assay=from_assay) for f in feature_list]
    return resolved


def _fetch_column(store: Any, column: str, cell_key: str | None) -> np.ndarray:
    """Fetch a cell-metadata column, with a helpful error when it is missing."""
    try:
        if cell_key is None:
            return np.asarray(store.cells.fetch_all(column))
        return np.asarray(store.cells.fetch(column, key=cell_key))
    except KeyError as exc:
        available = list(store.cells.columns)
        shown = ", ".join(map(str, available[:20]))
        if len(available) > 20:
            shown += f", ... ({len(available) - 20} more)"
        raise KeyError(
            f"'{column}' is not a metadata column in this DataStore. "
            f"Available metadata columns: {shown}"
        ) from exc


def _expression_long_frame(
    store: Any,
    resolved: Sequence[Any],
    *,
    group_by: str,
    facet_by: str | None = None,
    cell_key: str | None = "I",
    from_assay: str | None = None,
    normalization: NormalizationSpec | None = None,
    groups: Sequence[Any] | None = None,
    subset_by: str | None = None,
) -> tuple[list[pd.DataFrame], list[Any]]:
    """Build long-form expression frames, one per resolved feature.

    Returns ``(frames, group_order)`` where ``frames[j]`` is aligned with
    ``resolved[j]`` and holds ``value`` (float), ``group`` (categorical), and
    ``facet`` columns when ``facet_by`` is set.
    """
    if cell_key is None:
        cell_idx = np.arange(store.cells.N, dtype=np.int64)
    else:
        cell_idx = np.asarray(store.cells.active_index(cell_key), dtype=np.int64)

    n = len(cell_idx)
    group_vals = _fetch_column(store, group_by, cell_key)
    if len(group_vals) != n:
        raise ValueError("group_by length does not match selected cells")

    facet_vals = None
    if facet_by is not None:
        facet_vals = _fetch_column(store, facet_by, cell_key)
        if len(facet_vals) != n:
            raise ValueError("facet_by length does not match selected cells")

    subset_vals = (
        _fetch_column(store, subset_by, cell_key) if subset_by is not None else None
    )
    mask, group_order = resolve_cell_selection(
        n,
        subset=subset_vals,
        subset_name=subset_by,
        category_values=group_vals,
        groups=groups,
    )
    assert group_order is not None
    expr = fetch_normalized_feature_matrix(
        store,
        list(resolved),
        cell_idx,
        normalization=normalization,
    )
    if expr.shape[0] != n:
        raise ValueError("expression matrix does not match selected cells")
    expr = expr[mask]
    group_vals = group_vals[mask]
    if facet_vals is not None:
        facet_vals = facet_vals[mask]

    frames: list[pd.DataFrame] = []
    for j in range(len(resolved)):
        part = pd.DataFrame(
            {
                "value": np.asarray(expr[:, j], dtype=np.float64),
                "group": group_vals,
            }
        )
        if facet_vals is not None:
            part["facet"] = facet_vals
        frames.append(part)
    return frames, group_order


def _group_palette(
    order: Sequence[Any],
    palette: Mapping[Any, str] | Sequence[str] | str | None,
) -> dict[Any, str]:
    categories = list(order)
    if isinstance(palette, Mapping):
        return categorical_color_map(categories, palette=dict(palette))
    if palette is None:
        return categorical_color_map(categories)
    if isinstance(palette, str):
        sns = require_seaborn()
        colors = sns.color_palette(palette, n_colors=len(categories)).as_hex()
        return dict(zip(categories, colors))
    colors = list(palette)
    if len(colors) < len(categories):
        raise ValueError(
            f"palette has {len(colors)} colors but {len(categories)} groups require "
            f"at least that many"
        )
    return dict(zip(categories, colors))


def _categorical_scale(order: Sequence[Any], palette: dict[Any, str]) -> CategoricalScale:
    return CategoricalScale(order=tuple(order), palette=palette)


def _subsample(frame: pd.DataFrame, max_points: int, rng: np.random.Generator) -> pd.DataFrame:
    if max_points <= 0 or len(frame) <= max_points:
        return frame
    idx = rng.choice(len(frame), size=max_points, replace=False)
    return frame.iloc[np.sort(idx)]


def _panel_tables(
    resolved: Sequence[Any],
    frames: Sequence[pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    labels = [r.label for r in resolved]
    counts = pd.Series(labels).value_counts()
    tables: dict[str, pd.DataFrame] = {}
    for index, label in enumerate(labels):
        name = label if counts[label] == 1 else f"{index}:{label}"
        tables[name] = frames[index].reset_index(drop=True)
    return tables


def _resolve_group_order(
    frames: Sequence[pd.DataFrame],
    group_order: list[Any],
) -> list[Any]:
    if group_order:
        return list(group_order)
    observed = pd.unique(pd.concat(frames)["group"]).tolist()
    return sort_categories(observed)


def violinplot(
    store: Any,
    features: str | FeatureRef | Sequence[str | FeatureRef],
    *,
    group_by: str,
    cell_key: str | None = "I",
    from_assay: str | None = None,
    normalization: NormalizationSpec | Sequence[str] | None = None,
    groups: Sequence[Any] | None = None,
    subset_by: str | None = None,
    title: str | None = "Expression violin plot",
    x_label_rotation: float = 45,
    palette: Mapping[Any, str] | Sequence[str] | str | None = None,
    jitter: bool = True,
    jitter_size: float = 1.5,
    jitter_amount: float = 0.3,
    point_color: str = "black",
    point_alpha: float = 0.7,
    violin_inner: str | None = None,
    violin_alpha: float = 0.85,
    violin_linewidth: float = 0.8,
    title_bg_color: str = "#1F2933",
    title_text_color: str = "white",
    grid: bool = True,
    grid_alpha: float = 0.3,
    max_points: int = 20000,
    seed: int = 0,
    share_y: bool = True,
    figsize: tuple[float, float] | None = None,
    theme: str = "notebook",
    target: Any | None = None,
    show: bool = True,
) -> PlotResult:
    """Plot per-gene expression violins grouped by a cell metadata column.

    One violin panel is drawn per gene with the ``group_by`` categories on the
    x-axis. Violin bodies are filled with distinct colours mapped to each
    group, and individual cells are overlaid as jittered points. A dark,
    full-width header block holds the figure title in white text.

    Data is read natively from a Scarf ``DataStore``: cell metadata comes from
    ``store.cells`` and expression values from the assay feature matrix, so no
    AnnData, Scanpy, or Seurat objects are required.

    Args:
        store: A Scarf DataStore.
        features: Gene names, ids, indices, ``FeatureRef`` objects, or a list.
        group_by: Cell metadata column used to group and colour the violins.
        cell_key: Boolean metadata column selecting cells (default ``"I"``).
            Pass ``None`` to include every cell.
        from_assay: Assay to read expression from (defaults to store default).
        normalization: ``NormalizationSpec`` or a ``(source, transform)`` pair,
            for example ``("assay", "log1p")``.
        groups: Keep (and order) only these ``group_by`` categories.
        subset_by: Boolean metadata column; keep only ``True`` cells.
        title: Header text. Pass ``None`` to omit the header block.
        x_label_rotation: Rotation of the group labels (degrees).
        palette: Color map (dict), palette name, or sequence of colors.
        jitter: Overlay individual cells as jittered points.
        jitter_size / jitter_amount: Point size and jitter width for the overlay.
        point_color / point_alpha: Colour and opacity of the overlay points.
        violin_inner: Seaborn ``inner`` style (``None`` keeps the body clean).
        violin_alpha / violin_linewidth: Fill opacity and edge width.
        title_bg_color / title_text_color: Header block fill and text colour.
        grid / grid_alpha: Toggle the subtle horizontal gridlines.
        max_points: Cap on the number of overlay points drawn (``0`` disables).
        seed: Random seed for point subsampling and jitter.
        share_y: Share one value scale across all gene panels.
        figsize: Figure size in inches; auto-sized when omitted.
        theme: Scarf plot theme.
        target: ``None`` to create an owned figure, or a matplotlib ``Figure``
            to draw into.
        show: Display the figure before returning.
    """
    plt, _ = require_matplotlib()
    from matplotlib.figure import Figure as _Figure

    sns = require_seaborn()
    normalization = _coerce_normalization(normalization)
    resolved = _resolve_genes(store, features, from_assay=from_assay)
    frames, group_order = _expression_long_frame(
        store,
        resolved,
        group_by=group_by,
        cell_key=cell_key,
        from_assay=from_assay,
        normalization=normalization,
        groups=groups,
        subset_by=subset_by,
    )
    order = _resolve_group_order(frames, group_order)
    palette_map = _group_palette(order, palette)

    n_panels = len(resolved)
    if figsize is None:
        panel_width = max(2.8, 0.55 * max(len(order), 1) + 1.4)
        figsize = (panel_width * n_panels, 4.8)

    if target is None:
        fig = plt.figure(figsize=figsize)
        owns = True
    else:
        if not isinstance(target, _Figure):
            raise TypeError("target must be None or a matplotlib Figure")
        fig = target
        owns = False

    header_ax = None
    if title is not None:
        header_ax = fig.add_axes([0.0, 0.93, 1.0, 0.07])
        header_ax.set_axis_off()
        header_ax.set_facecolor(title_bg_color)
        header_ax.text(
            0.5,
            0.5,
            title,
            transform=header_ax.transAxes,
            ha="center",
            va="center",
            color=title_text_color,
            fontsize=13,
            fontweight="bold",
        )
        top = 0.90
    else:
        top = 0.94
    panel_gs = fig.add_gridspec(1, n_panels, top=top, bottom=0.08, wspace=0.22)
    axes = [fig.add_subplot(panel_gs[0, j]) for j in range(n_panels)]

    rng = np.random.default_rng(seed)
    any_subsampled = False
    y_limits: list[tuple[float, float]] = []
    with theme_context(theme):
        for j, (res, ax) in enumerate(zip(resolved, axes)):
            sub = frames[j]
            sns.violinplot(
                data=sub,
                x="group",
                y="value",
                hue="group",
                hue_order=order,
                order=order,
                palette=palette_map,
                dodge=False,
                legend=False,
                inner=violin_inner,
                cut=0,
                linewidth=violin_linewidth,
                saturation=0.9,
                ax=ax,
            )
            for collection in ax.collections:
                collection.set_alpha(violin_alpha)
            if jitter:
                points = _subsample(sub, max_points, rng)
                if len(points) < len(sub):
                    any_subsampled = True
                sns.stripplot(
                    data=points,
                    x="group",
                    y="value",
                    order=order,
                    color=point_color,
                    size=jitter_size,
                    jitter=jitter_amount,
                    alpha=point_alpha,
                    linewidth=0,
                    ax=ax,
                )
            ax.set_title(res.label, fontstyle="italic", fontsize=11)
            ax.set_xlabel("")
            ax.set_ylabel("expression")
            ax.tick_params(axis="x", labelrotation=x_label_rotation)
            for tick in ax.get_xticklabels():
                tick.set_ha("right")
            if grid:
                ax.grid(axis="y", which="major", alpha=grid_alpha, linewidth=0.6)
                ax.set_axisbelow(True)
            finite = sub["value"].to_numpy(dtype=np.float64)
            finite = finite[np.isfinite(finite)]
            if len(finite):
                y_limits.append((float(finite.min()), float(finite.max())))

        if share_y and len(y_limits) > 1:
            ymin = min(lo for lo, _ in y_limits)
            ymax = max(hi for _, hi in y_limits)
            if ymax > ymin:
                pad = 0.05 * (ymax - ymin)
                for ax in axes:
                    ax.set_ylim(ymin - pad, ymax + pad)

        apply_figure_chrome(fig, theme)
        if header_ax is not None:
            header_ax.set_axis_off()
            header_ax.set_facecolor(title_bg_color)

    panel_keys = [res.label for res in resolved]
    if len(set(panel_keys)) != len(panel_keys):
        panel_keys = list(range(n_panels))
    axes_map = dict(zip(panel_keys, axes))

    result = PlotResult(
        figure=fig,
        axes=axes_map,
        tables=_panel_tables(resolved, frames),
        legends=(LegendSpec(kind="categorical", label=group_by),),
        scales=(_categorical_scale(order, palette_map),),
        provenance=PlotProvenance(
            scarf_version=_scarf_version(),
            assay=(resolved[0].assay if len({r.assay for r in resolved}) == 1 else None),
            cell_key=cell_key,
            n_cells=len(frames[0]),
            renderer="matplotlib",
            notes=("violin",),
            extras={
                "group_by": group_by,
                "groups": None if groups is None else list(groups),
                "subset_by": subset_by,
                "x_label_rotation": x_label_rotation,
                "jitter": jitter,
                "jitter_size": jitter_size,
                "title": title,
                "share_y": share_y,
                "approximate": any_subsampled,
                "normalization": {
                    "source": normalization.source,
                    "transform": normalization.transform,
                },
            },
        ),
        owns_figure=owns,
        theme=theme,
    )
    if show:
        result.show()
    return result


def _draw_boxes(
    ax: Any,
    sns: Any,
    sub: pd.DataFrame,
    *,
    order: Sequence[Any],
    palette_map: dict[Any, str],
    show_outliers: bool,
    flier_size: float,
    box_linewidth: float,
    median_linewidth: float,
) -> None:
    """Draw side-by-side group boxes with a widened, dark median line."""
    sns.boxplot(
        data=sub,
        x="group",
        y="value",
        order=order,
        hue="group",
        hue_order=order,
        palette=palette_map,
        dodge=False,
        legend=False,
        showfliers=show_outliers,
        fliersize=flier_size,
        linewidth=box_linewidth,
        ax=ax,
    )
    for line in ax.lines:
        x_data, y_data = line.get_xdata(), line.get_ydata()
        if (
            len(x_data) >= 2
            and len(y_data) >= 2
            and x_data[0] != x_data[-1]
            and y_data[0] == y_data[-1]
        ):
            line.set_linewidth(median_linewidth)
            line.set_color("#111111")


def _add_group_legend(
    fig: Any,
    order: Sequence[Any],
    palette_map: dict[Any, str],
    *,
    legend_loc: str,
    legend_title: str | None,
    group_by: str,
) -> None:
    from matplotlib.patches import Patch

    handles = [
        Patch(
            facecolor=palette_map[value],
            edgecolor="#333333",
            linewidth=0.6,
            label=str(value),
        )
        for value in order
    ]
    resolved_loc = "outside right center" if legend_loc == "outside right" else legend_loc
    fig.legend(
        handles=handles,
        title=legend_title or group_by,
        loc=resolved_loc,
        frameon=False,
        fontsize=9,
    )


def _boxplot_simple(
    plt: Any,
    sns: Any,
    figure_type: Any,
    store: Any,
    resolved: Sequence[Any],
    frames: Sequence[pd.DataFrame],
    order: Sequence[Any],
    palette_map: dict[Any, str],
    *,
    group_by: str,
    cell_key: str | None,
    groups: Sequence[Any] | None,
    subset_by: str | None,
    normalization: NormalizationSpec,
    x_label_rotation: float,
    show_outliers: bool,
    flier_size: float,
    box_linewidth: float,
    median_linewidth: float,
    show_legend: bool,
    legend_loc: str,
    legend_title: str | None,
    share_y: bool,
    figsize: tuple[float, float] | None,
    title: str | None,
    theme: str,
    target: Any | None,
    show: bool,
) -> PlotResult:
    """Draw a single box plot per gene: one box per ``group_by`` category."""
    n_genes = len(resolved)
    reserve_legend = show_legend and legend_loc in (
        "outside right",
        "outside right upper",
        "outside right center",
        "outside right lower",
    )
    if figsize is None:
        panel_width = max(2.8, 0.55 * max(len(order), 1) + 1.4)
        width = panel_width * n_genes + (1.6 if reserve_legend else 0.4)
        figsize = (width, 4.6)

    if target is None:
        fig = plt.figure(figsize=figsize)
        owns = True
    else:
        if not isinstance(target, figure_type):
            raise TypeError("target must be None or a matplotlib Figure")
        fig = target
        owns = False

    right_margin = 0.82 if reserve_legend else 0.97
    panel_gs = fig.add_gridspec(
        1,
        n_genes,
        wspace=0.25,
        top=0.90 if title is None else 0.88,
        bottom=0.12,
        right=right_margin,
        left=0.10,
    )
    axes = [fig.add_subplot(panel_gs[0, j]) for j in range(n_genes)]

    panel_keys: list[Hashable] = [res.label for res in resolved]
    if len(set(panel_keys)) != len(panel_keys):
        panel_keys = list(range(n_genes))
    axes_map = dict(zip(panel_keys, axes))

    tables: dict[str, pd.DataFrame] = {}
    y_limits: list[tuple[float, float]] = []
    with theme_context(theme):
        for j, (res, gene_frame) in enumerate(zip(resolved, frames)):
            ax = axes[j]
            _draw_boxes(
                ax,
                sns,
                gene_frame,
                order=order,
                palette_map=palette_map,
                show_outliers=show_outliers,
                flier_size=flier_size,
                box_linewidth=box_linewidth,
                median_linewidth=median_linewidth,
            )
            ax.tick_params(axis="x", labelrotation=x_label_rotation)
            for tick in ax.get_xticklabels():
                tick.set_ha("right")
            ax.set_xlabel(group_by)
            ax.set_ylabel("expression")
            ax.set_title(res.label, fontstyle="italic", fontsize=11)
            finite = gene_frame["value"].to_numpy(dtype=np.float64)
            finite = finite[np.isfinite(finite)]
            if len(finite):
                y_limits.append((float(finite.min()), float(finite.max())))
            tables[res.label] = gene_frame.reset_index(drop=True)
        if share_y and len(y_limits) > 1:
            ymin = min(lo for lo, _ in y_limits)
            ymax = max(hi for _, hi in y_limits)
            if ymax > ymin:
                pad = 0.05 * (ymax - ymin)
                for ax in axes:
                    ax.set_ylim(ymin - pad, ymax + pad)
        if title is not None:
            fig.suptitle(title)
        apply_figure_chrome(fig, theme)
        if show_legend:
            _add_group_legend(
                fig,
                order,
                palette_map,
                legend_loc=legend_loc,
                legend_title=legend_title,
                group_by=group_by,
            )

    n_cells = len(frames[0])
    return PlotResult(
        figure=fig,
        axes=axes_map,
        tables=tables,
        legends=(
            (LegendSpec(kind="categorical", label=group_by),) if show_legend else ()
        ),
        scales=(_categorical_scale(order, palette_map),),
        provenance=PlotProvenance(
            scarf_version=_scarf_version(),
            assay=(resolved[0].assay if len({r.assay for r in resolved}) == 1 else None),
            cell_key=cell_key,
            n_cells=n_cells,
            renderer="matplotlib",
            notes=("boxplot",),
            extras={
                "group_by": group_by,
                "facet_by": None,
                "groups": None if groups is None else list(groups),
                "subset_by": subset_by,
                "x_label_rotation": x_label_rotation,
                "show_outliers": show_outliers,
                "show_legend": show_legend,
                "legend_loc": legend_loc,
                "title": title,
                "share_y": share_y,
                "normalization": {
                    "source": normalization.source,
                    "transform": normalization.transform,
                },
            },
        ),
        owns_figure=owns,
        theme=theme,
    )


def boxplot(
    store: Any,
    features: str | FeatureRef | Sequence[str | FeatureRef],
    *,
    group_by: str,
    facet_by: str | None = None,
    facet_order: Sequence[Any] | None = None,
    facet_titles: Mapping[Any, str] | None = None,
    panel_label_format: str = "({letter}) {title}",
    cell_key: str | None = "I",
    from_assay: str | None = None,
    normalization: NormalizationSpec | Sequence[str] | None = None,
    groups: Sequence[Any] | None = None,
    subset_by: str | None = None,
    n_cols: int | None = None,
    x_label_rotation: float = 90,
    show_outliers: bool = True,
    flier_size: float = 3,
    flier_alpha: float = 0.6,
    box_linewidth: float = 0.8,
    median_linewidth: float = 2.5,
    palette: Mapping[Any, str] | Sequence[str] | str | None = None,
    show_legend: bool = False,
    legend_loc: str = "outside right",
    legend_title: str | None = None,
    share_y: bool = True,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
    theme: str = "notebook",
    target: Any | None = None,
    show: bool = True,
) -> PlotResult:
    """Plot expression box plots for one or more genes.

    Without ``facet_by`` this is a single box plot per gene: the
    ``group_by`` categories sit on the x-axis and each gets one box with a
    thickened median line, whiskers, and optional outlier dots. Pass
    ``facet_by`` to split each gene into a grid of panels, one per category of
    that column (for example datasets or conditions). Every facet panel holds
    side-by-side box plots for the ``group_by`` categories and carries a
    lettered label such as ``(a) Dataset 1`` beneath it. The ``group_by``
    categories are already named on the x-axis, so no legend is drawn by
    default; pass ``show_legend=True`` to add a consolidated legend outside the
    axes on the right.

    Args:
        store: A Scarf DataStore.
        features: Gene names, ids, indices, ``FeatureRef`` objects, or a list.
        group_by: Cell metadata column holding the side-by-side boxes.
        facet_by: Optional cell metadata column whose categories become facets.
            When ``None``, one single box plot is drawn per gene.
        facet_order: Order of the facets; natural order when omitted.
        facet_titles: Optional display names for facet values.
        panel_label_format: Template with ``{letter}``, ``{title}``, ``{index}``
            and ``{facet}`` fields for the label beneath each facet.
        cell_key: Boolean metadata column selecting cells (default ``"I"``).
        from_assay: Assay to read expression from (defaults to store default).
        normalization: ``NormalizationSpec`` or a ``(source, transform)`` pair.
        groups: Keep (and order) only these ``group_by`` categories.
        subset_by: Boolean metadata column; keep only ``True`` cells.
        n_cols: Facet panels per row inside each gene block.
        x_label_rotation: Rotation of the group labels (degrees).
        show_outliers: Draw individual outlier dots past the whiskers.
        flier_size / flier_alpha: Size and opacity of outlier dots.
        box_linewidth / median_linewidth: Edge width and median line width.
        palette: Color map (dict), palette name, or sequence of colors.
        show_legend: Draw a consolidated categorical legend. Off by default
            because the ``group_by`` categories are already labelled on the
            x-axis.
        legend_loc: Placement of the legend when ``show_legend=True``.
        legend_title: Legend title when ``show_legend=True`` (defaults to
            ``group_by``).
        share_y: Share one value scale across the facets of each gene block.
        figsize: Figure size in inches; auto-sized when omitted.
        title: Optional figure super-title.
        theme: Scarf plot theme.
        target: ``None`` to create an owned figure, or a matplotlib ``Figure``.
        show: Display the figure before returning.
    """
    plt, _ = require_matplotlib()
    from matplotlib.figure import Figure as _Figure

    sns = require_seaborn()
    normalization = _coerce_normalization(normalization)
    resolved = _resolve_genes(store, features, from_assay=from_assay)
    frames, group_order = _expression_long_frame(
        store,
        resolved,
        group_by=group_by,
        facet_by=facet_by,
        cell_key=cell_key,
        from_assay=from_assay,
        normalization=normalization,
        groups=groups,
        subset_by=subset_by,
    )
    order = _resolve_group_order(frames, group_order)
    palette_map = _group_palette(order, palette)

    n_genes = len(resolved)
    reserve_legend = show_legend and legend_loc in (
        "outside right",
        "outside right upper",
        "outside right center",
        "outside right lower",
    )

    if facet_by is None:
        return _boxplot_simple(
            plt,
            sns,
            _Figure,
            store,
            resolved,
            frames,
            order,
            palette_map,
            group_by=group_by,
            cell_key=cell_key,
            groups=groups,
            subset_by=subset_by,
            normalization=normalization,
            x_label_rotation=x_label_rotation,
            show_outliers=show_outliers,
            flier_size=flier_size,
            box_linewidth=box_linewidth,
            median_linewidth=median_linewidth,
            show_legend=show_legend,
            legend_loc=legend_loc,
            legend_title=legend_title,
            share_y=share_y,
            figsize=figsize,
            title=title,
            theme=theme,
            target=target,
            show=show,
        )

    observed_facets = pd.unique(pd.concat(frames)["facet"]).tolist()
    if facet_order is not None:
        facet_list = list(facet_order)
        missing = [value for value in facet_list if value not in observed_facets]
        if missing:
            raise ValueError(
                "facet_order contains labels not present in the data: "
                + ", ".join(map(str, missing[:10]))
            )
    else:
        facet_list = sort_categories(observed_facets)
    if not facet_list:
        raise ValueError("No cells remain after facet selection")
    n_facets = len(facet_list)

    n_cols = n_cols if n_cols is not None else min(n_facets, 4)
    n_cols = max(1, min(n_cols, n_facets))
    n_facet_rows = int(np.ceil(n_facets / n_cols))

    if figsize is None:
        panel_width = max(2.6, 0.5 * max(len(order), 1) + 1.2)
        width = panel_width * n_cols + (1.6 if reserve_legend else 0.4)
        height = 2.1 * n_facet_rows * n_genes + 0.6
        figsize = (width, height)

    if target is None:
        fig = plt.figure(figsize=figsize)
        owns = True
    else:
        if not isinstance(target, _Figure):
            raise TypeError("target must be None or a matplotlib Figure")
        fig = target
        owns = False

    right_margin = 0.82 if reserve_legend else 0.97
    outer = fig.add_gridspec(
        n_genes,
        1,
        hspace=0.35,
        top=0.90 if title is None else 0.88,
        bottom=0.08,
        right=right_margin,
        left=0.10,
    )

    axes_map: dict[Hashable, Any] = {}
    tables: dict[str, pd.DataFrame] = {}
    letters = [chr(ord("a") + i) if i < 26 else str(i) for i in range(n_facets)]

    with theme_context(theme):
        for gene_index, (res, gene_frame) in enumerate(zip(resolved, frames)):
            block = outer[gene_index, 0]
            inner = block.subgridspec(n_facet_rows, n_cols, wspace=0.25, hspace=0.5)
            facet_axes: list[Any] = []
            y_limits: list[tuple[float, float]] = []
            for facet_index, facet_value in enumerate(facet_list):
                row = facet_index // n_cols
                col = facet_index % n_cols
                ax = fig.add_subplot(inner[row, col])
                facet_axes.append(ax)
                axes_map[(res.label, facet_value)] = ax
                sub = gene_frame[gene_frame["facet"] == facet_value]
                _draw_boxes(
                    ax,
                    sns,
                    sub,
                    order=order,
                    palette_map=palette_map,
                    show_outliers=show_outliers,
                    flier_size=flier_size,
                    box_linewidth=box_linewidth,
                    median_linewidth=median_linewidth,
                )
                ax.tick_params(axis="x", labelrotation=x_label_rotation)
                for tick in ax.get_xticklabels():
                    tick.set_ha("right")
                facet_title = (
                    facet_titles[facet_value]
                    if facet_titles is not None and facet_value in facet_titles
                    else str(facet_value)
                )
                label = panel_label_format.format(
                    letter=letters[facet_index],
                    title=facet_title,
                    index=facet_index,
                    facet=str(facet_value),
                )
                ax.text(
                    0.5,
                    -0.32,
                    label,
                    transform=ax.transAxes,
                    ha="center",
                    va="top",
                    fontsize=9,
                    color="#333333",
                )
                if gene_index == n_genes - 1:
                    ax.set_xlabel(group_by)
                else:
                    ax.set_xlabel("")
                finite = sub["value"].to_numpy(dtype=np.float64)
                finite = finite[np.isfinite(finite)]
                if len(finite):
                    y_limits.append((float(finite.min()), float(finite.max())))
                if facet_index == 0:
                    ax.set_ylabel(res.label, fontstyle="italic")
                else:
                    ax.set_ylabel("")
            if share_y and len(y_limits) > 1:
                ymin = min(lo for lo, _ in y_limits)
                ymax = max(hi for _, hi in y_limits)
                if ymax > ymin:
                    pad = 0.05 * (ymax - ymin)
                    for ax in facet_axes:
                        ax.set_ylim(ymin - pad, ymax + pad)
            gene_table = gene_frame.reset_index(drop=True)
            tables[res.label] = gene_table

        if title is not None:
            fig.suptitle(title)
        apply_figure_chrome(fig, theme)
        if show_legend:
            _add_group_legend(
                fig,
                order,
                palette_map,
                legend_loc=legend_loc,
                legend_title=legend_title,
                group_by=group_by,
            )

    n_cells = len(frames[0])
    result = PlotResult(
        figure=fig,
        axes=axes_map,
        tables=tables,
        legends=(
            (LegendSpec(kind="categorical", label=group_by),) if show_legend else ()
        ),
        scales=(_categorical_scale(order, palette_map),),
        provenance=PlotProvenance(
            scarf_version=_scarf_version(),
            assay=(resolved[0].assay if len({r.assay for r in resolved}) == 1 else None),
            cell_key=cell_key,
            n_cells=n_cells,
            renderer="matplotlib",
            notes=("boxplot", "faceted"),
            extras={
                "group_by": group_by,
                "facet_by": facet_by,
                "facet_order": [str(value) for value in facet_list],
                "groups": None if groups is None else list(groups),
                "subset_by": subset_by,
                "n_cols": n_cols,
                "x_label_rotation": x_label_rotation,
                "show_outliers": show_outliers,
                "show_legend": show_legend,
                "legend_loc": legend_loc,
                "title": title,
                "share_y": share_y,
                "normalization": {
                    "source": normalization.source,
                    "transform": normalization.transform,
                },
            },
        ),
        owns_figure=owns,
        theme=theme,
    )
    if show:
        result.show()
    return result


def _stacked_group_palette(
    order: Sequence[Any],
    palette: Mapping[Any, str] | Sequence[str] | str | None,
    cmap: str | None,
) -> dict[Any, str]:
    """Resolve group colours, sampling a colormap when ``palette`` is unset."""
    if cmap is None or palette is not None:
        return _group_palette(order, palette)
    from matplotlib import colormaps
    from matplotlib.colors import to_hex

    if cmap not in colormaps:
        raise ValueError(f"Unknown colormap {cmap!r}")
    colors = colormaps[cmap](np.linspace(0, 1, max(len(order), 1)))
    return dict(zip(order, (to_hex(color) for color in colors)))


def _violin_scale_kwargs(scale: str) -> dict[str, str]:
    """Map the public ``scale`` argument onto seaborn's current kwarg.

    Seaborn renamed ``scale`` to ``density_norm`` in 0.13; the new name is
    accepted there and on every later release without a deprecation warning.
    """
    return {"density_norm": scale}


def _mean_expression_row_colors(
    frames: Sequence[pd.DataFrame],
    order: Sequence[Any],
    *,
    cmap: str,
    vmin: float | None,
    vmax: float | None,
) -> tuple[list[dict[Any, str]], float, float]:
    """Per-row (per-gene) group colours by mean expression on a shared scale.

    ``row_color_maps[i]`` maps each group to a colour for ``frames[i]``, where
    a group's colour encodes the mean expression of that gene within the group.
    The colour scale is shared across all rows and bounded by ``(lo, hi)``
    (explicit ``vmin``/``vmax``, or the min/max of every gene-by-group mean).
    """
    from matplotlib import colormaps
    from matplotlib.colors import to_hex

    if cmap not in colormaps:
        raise ValueError(f"Unknown colormap {cmap!r}")
    row_means: list[dict[Any, float]] = []
    for frame in frames:
        means: dict[Any, float] = {}
        for group in order:
            values = frame.loc[frame["group"] == group, "value"].to_numpy(
                dtype=np.float64
            )
            values = values[np.isfinite(values)]
            means[group] = float(values.mean()) if values.size else np.nan
        row_means.append(means)
    finite_values = [
        mean for row in row_means for mean in row.values() if np.isfinite(mean)
    ]
    if not finite_values:
        raise ValueError("No finite expression values to colour by")
    if vmin is None:
        vmin = min(finite_values)
    if vmax is None:
        vmax = max(finite_values)
    if vmax < vmin:
        raise ValueError("vmax must be greater than or equal to vmin")
    span = vmax - vmin
    row_color_maps: list[dict[Any, str]] = []
    for means in row_means:
        colors: dict[Any, str] = {}
        for group in order:
            mean = means[group]
            t = (
                0.5
                if not np.isfinite(mean) or span == 0
                else float(np.clip((mean - vmin) / span, 0, 1))
            )
            colors[group] = to_hex(colormaps[cmap](t))
        row_color_maps.append(colors)
    return row_color_maps, vmin, vmax


def stacked_violin(
    store: Any,
    features: str | FeatureRef | Sequence[str | FeatureRef],
    *,
    group_by: str,
    cell_key: str | None = "I",
    from_assay: str | None = None,
    normalization: NormalizationSpec | Sequence[str] | None = None,
    groups: Sequence[Any] | None = None,
    subset_by: str | None = None,
    color_by: str = "mean",
    vmin: float | None = None,
    vmax: float | None = None,
    row_standardize: bool = False,
    x_label_rotation: float = 90,
    row_height: float = 0.8,
    width: float | None = None,
    palette: Mapping[Any, str] | Sequence[str] | str | None = None,
    cmap: str | None = None,
    violin_linewidth: float = 0.5,
    scale: str = "width",
    violin_alpha: float = 0.9,
    legend_loc: str = "outside right",
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
    theme: str = "notebook",
    target: Any | None = None,
    show: bool = True,
) -> PlotResult:
    """Plot a minimalist stacked violin plot, one row per gene.

    Genes become a vertical stack of slim subplots with zero vertical spacing
    (``hspace=0``). Inside every row a violin is drawn for each ``group_by``
    category, so the groups line up as columns down the whole stack. By default
    each violin's fill colour encodes the mean expression of that gene within
    the group, on a shared scale across all rows with a continuous colorbar on
    the right. Pass ``color_by="group"`` to colour each column with a single
    shared group colour instead (using ``palette`` or ``cmap``). Numeric y-ticks
    are removed and the gene name is rendered as a horizontal label on the
    left, forming a neat column of gene names. X-axis ticks appear only on the
    bottom-most row, rotated by ``x_label_rotation``. Only the left spine is
    kept (plus the bottom spine on the lowest row) for a floating, seamless
    look.

    Data is read natively from a Scarf ``DataStore``: cell metadata comes from
    ``store.cells`` and expression values from the assay feature matrix, so no
    AnnData, Scanpy, or Seurat objects are required.

    Args:
        store: A Scarf DataStore.
        features: Gene names, ids, indices, ``FeatureRef`` objects, or a list.
        group_by: Cell metadata column whose categories become the x-axis.
        cell_key: Boolean metadata column selecting cells (default ``"I"``).
            Pass ``None`` to include every cell.
        from_assay: Assay to read expression from (defaults to store default).
        normalization: ``NormalizationSpec`` or a ``(source, transform)`` pair,
            for example ``("assay", "log1p")``.
        groups: Keep (and order) only these ``group_by`` categories.
        subset_by: Boolean metadata column; keep only ``True`` cells.
        color_by: ``"mean"`` colours each violin by mean expression of the gene
            within the group; ``"group"`` uses the categorical ``palette``.
        vmin / vmax: Optional fixed bounds for the mean-expression colour scale.
        row_standardize: Z-score each gene row so different genes share a
            comparable value scale.
        x_label_rotation: Rotation of the group labels (degrees).
        row_height: Height in inches allotted to each gene row; the default
            figure height is ``n_genes * row_height``.
        width: Figure width in inches (auto-sized when omitted).
        palette: Color map (dict), palette name, or sequence of colors, used
            when ``color_by="group"``.
        cmap: Colormap used when ``color_by="mean"`` (default ``"viridis"``),
            or to derive one shared colour per group when ``color_by="group"``
            and ``palette`` is ``None``.
        violin_linewidth: Border width of the violin shapes.
        scale: Seaborn violin sizing: ``"area"`` or ``"width"``.
        violin_alpha: Fill opacity of the violins.
        legend_loc: Placement of the categorical legend (``color_by="group"``).
        figsize: Figure size in inches; overrides the auto-computed size.
        title: Optional figure super-title.
        theme: Scarf plot theme.
        target: ``None`` to create an owned figure, or a matplotlib ``Figure``.
        show: Display the figure before returning.
    """
    plt, _ = require_matplotlib()
    from matplotlib.figure import Figure as _Figure

    sns = require_seaborn()
    normalization = _coerce_normalization(normalization)
    resolved = _resolve_genes(store, features, from_assay=from_assay)
    frames, group_order = _expression_long_frame(
        store,
        resolved,
        group_by=group_by,
        cell_key=cell_key,
        from_assay=from_assay,
        normalization=normalization,
        groups=groups,
        subset_by=subset_by,
    )
    order = _resolve_group_order(frames, group_order)
    if color_by not in ("mean", "group"):
        raise ValueError("color_by must be 'mean' or 'group'")
    color_scale_bounds: tuple[float, float] | None = None
    row_color_maps: list[dict[Any, str]] | None = None
    resolved_cmap = cmap or "viridis"
    if color_by == "mean":
        row_color_maps, color_lo, color_hi = _mean_expression_row_colors(
            frames,
            order,
            cmap=resolved_cmap,
            vmin=vmin,
            vmax=vmax,
        )
        color_scale_bounds = (color_lo, color_hi)
    else:
        palette_map = _stacked_group_palette(order, palette, cmap)

    if scale not in ("area", "width", "count"):
        raise ValueError("scale must be 'area', 'width', or 'count'")
    if row_height <= 0:
        raise ValueError("row_height must be positive")

    n_genes = len(resolved)
    reserve_legend = legend_loc in (
        "outside right",
        "outside right upper",
        "outside right center",
        "outside right lower",
    )
    if figsize is None:
        if width is None:
            width = max(4.0, 0.5 * max(len(order), 1) + 2.0)
        colorbar_extra = 1.4 if color_scale_bounds is not None else 0.0
        width = width + (1.6 if reserve_legend else 0.0) + colorbar_extra
        figsize = (width, n_genes * row_height)

    if target is None:
        fig = plt.figure(figsize=figsize)
        owns = True
    else:
        if not isinstance(target, _Figure):
            raise TypeError("target must be None or a matplotlib Figure")
        fig = target
        owns = False

    right_margin = 0.82 if reserve_legend else 0.97
    gs = fig.add_gridspec(
        n_genes,
        1,
        hspace=0.0,
        top=0.90 if title is None else 0.86,
        bottom=0.10,
        left=0.18,
        right=right_margin,
    )
    axes = [fig.add_subplot(gs[i, 0]) for i in range(n_genes)]

    panel_keys: list[Hashable] = [res.label for res in resolved]
    if len(set(panel_keys)) != len(panel_keys):
        panel_keys = list(range(n_genes))
    axes_map = dict(zip(panel_keys, axes))

    tables: dict[str, pd.DataFrame] = {}
    with theme_context(theme):
        for i, (res, gene_frame) in enumerate(zip(resolved, frames)):
            ax = axes[i]
            display = gene_frame
            if row_standardize:
                values = display["value"].to_numpy(dtype=np.float64)
                finite = values[np.isfinite(values)]
                mean = float(finite.mean()) if finite.size else 0.0
                std = float(finite.std()) if finite.size else 0.0
                values = (values - mean) / std if std > 0 else values
                display = display.assign(value=values)
            sns.violinplot(
                data=display,
                x="group",
                y="value",
                hue="group",
                hue_order=order,
                order=order,
                palette=row_color_maps[i] if row_color_maps is not None else palette_map,
                dodge=False,
                legend=False,
                inner=None,
                cut=0,
                **_violin_scale_kwargs(scale),
                linewidth=violin_linewidth,
                saturation=0.9,
                ax=ax,
            )
            for collection in ax.collections:
                collection.set_alpha(violin_alpha)
            ax.set_yticks([])
            ax.set_ylabel(
                res.label,
                rotation=0,
                fontstyle="italic",
                ha="right",
                va="center",
                labelpad=8,
            )
            if i < n_genes - 1:
                ax.set_xticks([])
                ax.set_xlabel("")
            else:
                ax.tick_params(axis="x", labelrotation=x_label_rotation)
                for tick in ax.get_xticklabels():
                    tick.set_ha("right")
                ax.set_xlabel(group_by)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            if i < n_genes - 1:
                ax.spines["bottom"].set_visible(False)
            tables[res.label] = display.reset_index(drop=True)

        if title is not None:
            fig.suptitle(title)
        apply_figure_chrome(fig, theme)
        if color_scale_bounds is not None:
            color_lo, color_hi = color_scale_bounds
            if color_hi > color_lo:
                from matplotlib.colors import Normalize

                mappable = plt.cm.ScalarMappable(
                    cmap=resolved_cmap,
                    norm=Normalize(vmin=color_lo, vmax=color_hi),
                )
                mappable.set_array([])
                colorbar = fig.colorbar(
                    mappable,
                    ax=axes,
                    location="right",
                    shrink=0.8,
                    fraction=0.04,
                    pad=0.02,
                )
                colorbar.set_label("mean expression")
        else:
            _add_group_legend(
                fig,
                order,
                palette_map,
                legend_loc=legend_loc,
                legend_title=None,
                group_by=group_by,
            )

    if color_scale_bounds is not None:
        color_lo, color_hi = color_scale_bounds
        legend_specs = (
            LegendSpec(
                kind="colorbar",
                label="mean expression",
                extras={"vmin": color_lo, "vmax": color_hi},
            ),
        )
        scale_specs: tuple[Any, ...] = (
            ColorScale(cmap=resolved_cmap, vmin=color_lo, vmax=color_hi, scope="shared"),
        )
    else:
        legend_specs = (LegendSpec(kind="categorical", label=group_by),)
        scale_specs = (_categorical_scale(order, palette_map),)

    result = PlotResult(
        figure=fig,
        axes=axes_map,
        tables=tables,
        legends=legend_specs,
        scales=scale_specs,
        provenance=PlotProvenance(
            scarf_version=_scarf_version(),
            assay=(resolved[0].assay if len({r.assay for r in resolved}) == 1 else None),
            cell_key=cell_key,
            n_cells=len(frames[0]),
            renderer="matplotlib",
            notes=("stacked_violin",),
            extras={
                "group_by": group_by,
                "groups": None if groups is None else list(groups),
                "subset_by": subset_by,
                "color_by": color_by,
                "cmap": resolved_cmap if color_by == "mean" else cmap,
                "vmin": color_scale_bounds[0] if color_scale_bounds else None,
                "vmax": color_scale_bounds[1] if color_scale_bounds else None,
                "row_standardize": row_standardize,
                "x_label_rotation": x_label_rotation,
                "row_height": row_height,
                "scale": scale,
                "legend_loc": legend_loc,
                "title": title,
                "normalization": {
                    "source": normalization.source,
                    "transform": normalization.transform,
                },
            },
        ),
        owns_figure=owns,
        theme=theme,
    )
    if show:
        result.show()
    return result
