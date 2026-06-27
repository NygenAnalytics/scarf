"""Contains the code for plotting in Scarf."""

from collections.abc import Iterator
from typing import Any

import matplotlib as mpl  # type: ignore[import-not-found]
import matplotlib.pyplot as plt  # type: ignore[import-not-found]
import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns
from cmocean import cm  # type: ignore[import-not-found]

from .utils import logger

plt.rcParams["svg.fonttype"] = "none"


# These palettes were lifted from scanpy.plotting.palettes
custom_palettes = {
    10: [
        "#1f77b4",
        "#ff7f0e",
        "#279e68",
        "#d62728",
        "#aa40fc",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#b5bd61",
        "#17becf",
    ],
    20: [
        "#1f77b4",
        "#aec7e8",
        "#ff7f0e",
        "#ffbb78",
        "#2ca02c",
        "#98df8a",
        "#d62728",
        "#ff9896",
        "#9467bd",
        "#c5b0d5",
        "#8c564b",
        "#c49c94",
        "#e377c2",
        "#f7b6d2",
        "#7f7f7f",
        "#c7c7c7",
        "#bcbd22",
        "#dbdb8d",
        "#17becf",
        "#9edae5",
    ],
    28: [
        "#023fa5",
        "#7d87b9",
        "#bec1d4",
        "#d6bcc0",
        "#bb7784",
        "#8e063b",
        "#4a6fe3",
        "#8595e1",
        "#b5bbe3",
        "#e6afb9",
        "#e07b91",
        "#d33f6a",
        "#11c638",
        "#8dd593",
        "#c6dec7",
        "#ead3c6",
        "#f0b98d",
        "#ef9708",
        "#0fcfc0",
        "#9cded6",
        "#d5eae7",
        "#f3e1eb",
        "#f6c4e1",
        "#f79cd4",
        "#7f7f7f",
        "#c7c7c7",
        "#1CE6FF",
        "#336600",
    ],
    102: [
        "#FFFF00",
        "#1CE6FF",
        "#FF34FF",
        "#FF4A46",
        "#008941",
        "#006FA6",
        "#A30059",
        "#FFDBE5",
        "#7A4900",
        "#0000A6",
        "#63FFAC",
        "#B79762",
        "#004D43",
        "#8FB0FF",
        "#997D87",
        "#5A0007",
        "#809693",
        "#6A3A4C",
        "#1B4400",
        "#4FC601",
        "#3B5DFF",
        "#4A3B53",
        "#FF2F80",
        "#61615A",
        "#BA0900",
        "#6B7900",
        "#00C2A0",
        "#FFAA92",
        "#FF90C9",
        "#B903AA",
        "#D16100",
        "#DDEFFF",
        "#000035",
        "#7B4F4B",
        "#A1C299",
        "#300018",
        "#0AA6D8",
        "#013349",
        "#00846F",
        "#372101",
        "#FFB500",
        "#C2FFED",
        "#A079BF",
        "#CC0744",
        "#C0B9B2",
        "#C2FF99",
        "#001E09",
        "#00489C",
        "#6F0062",
        "#0CBD66",
        "#EEC3FF",
        "#456D75",
        "#B77B68",
        "#7A87A1",
        "#788D66",
        "#885578",
        "#FAD09F",
        "#FF8A9A",
        "#D157A0",
        "#BEC459",
        "#456648",
        "#0086ED",
        "#886F4C",
        "#34362D",
        "#B4A8BD",
        "#00A6AA",
        "#452C2C",
        "#636375",
        "#A3C8C9",
        "#FF913F",
        "#938A81",
        "#575329",
        "#00FECF",
        "#B05B6F",
        "#8CD0FF",
        "#3B9700",
        "#04F757",
        "#C8A1A1",
        "#1E6E00",
        "#7900D7",
        "#A77500",
        "#6367A9",
        "#A05837",
        "#6B002C",
        "#772600",
        "#D790FF",
        "#9B9700",
        "#549E79",
        "#FFF69F",
        "#201625",
        "#72418F",
        "#BC23FF",
        "#99ADC0",
        "#3A2465",
        "#922329",
        "#5B4534",
        "#FDE8DC",
        "#404E55",
        "#0089A3",
        "#CB7E98",
        "#A4E804",
        "#324E72",
    ],
}


def clean_axis(ax: Any, ts: int = 11, ga: float = 0.4) -> bool:
    """Clean matplotlib axis spines and add a light grid.

    Args:
        ax: Matplotlib axis.
        ts: Tick label font size.
        ga: Grid line alpha.

    Returns:
        True
    """
    ax.xaxis.set_tick_params(labelsize=ts)
    ax.yaxis.set_tick_params(labelsize=ts)
    for i in ["top", "bottom", "left", "right"]:
        ax.spines[i].set_visible(False)
    ax.grid(which="major", linestyle="--", alpha=ga)
    ax.figure.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    return True


def plot_graph_qc(g: Any) -> None:
    """Plot KNN graph QC: node degree distribution and edge weight histogram.

    Args:
        g: Sparse adjacency matrix (CSR or COO) for the KNN graph.
    """
    _, axis = plt.subplots(1, 2, figsize=(12, 4))
    ax = axis[0]
    x = np.array((g != 0).sum(axis=0))[0]
    y = pd.Series(x).value_counts().sort_index()
    ax.bar(y.index, y.values, width=0.5)
    xlim = np.percentile(x, 99.5) + 5
    ax.set_xlim((0, xlim))
    ax.set_xlabel("Node degree")
    ax.set_ylabel("Frequency")
    ax.text(
        xlim,
        y.values.max(),
        f"plot is clipped (max degree: {y.index.max()})",
        ha="right",
        fontsize=9,
    )
    clean_axis(ax)
    ax = axis[1]
    ax.hist(g.data, bins=30)
    ax.set_xlabel("Edge weight")
    ax.set_ylabel("Frequency")
    clean_axis(ax)
    plt.tight_layout()
    plt.show()


def plot_qc(
    data: pd.DataFrame,
    color: str = "steelblue",
    cmap: str = "tab20",
    fig_size: tuple | None = None,
    label_size: float = 10.0,
    title_size: float = 10,
    sup_title: str | None = None,
    sup_title_size: float = 12,
    scatter_size: float = 1.0,
    max_points: int = 10000,
    show_on_single_row: bool = True,
    show_fig: bool = True,
) -> Any | None:
    """Plot per-metric QC violin plots grouped by ``groups`` column.

    Args:
        data: DataFrame with a ``groups`` column and numeric metric columns.
        color: Violin fill when a single group is shown.
        cmap: Colormap when multiple groups are shown.
        fig_size: Figure size tuple (auto if None).
        label_size: Axis label font size.
        title_size: Subplot title font size.
        sup_title: Figure suptitle.
        sup_title_size: Suptitle font size.
        scatter_size: Unused (kept for API compatibility).
        max_points: Unused (kept for API compatibility).
        show_on_single_row: Lay out metrics in one row vs one column.
        show_fig: Call ``plt.show()`` when True; otherwise return the figure.
    """
    n_plots = data.shape[1] - 1
    n_groups = data["groups"].nunique()
    if n_groups > 5 and show_on_single_row is True:
        logger.info(
            "Too many groups in the plot. If you think that plot is too wide then consider turning "
            "`show_on_single_row` parameter to False"
        )
    if show_on_single_row is True:
        n_rows = 1
        n_cols = n_plots
    else:
        n_rows = n_plots
        n_cols = 1
    if fig_size is None:
        fig_width = min(15, n_groups + (2 * n_cols))
        fig_height = 1 + 2.5 * n_rows
        fig_size = (fig_width, fig_height)
    fig = plt.figure(figsize=fig_size)
    grouped = data.groupby("groups", observed=False)
    for i in range(n_plots):
        if data.columns[i] == "groups":
            continue
        vals_raw: dict[str, list[Any]] = {"g": [], "v": []}
        for j in sorted(data["groups"].unique()):
            val = grouped.get_group(j)[data.columns[i]].values
            vals_raw["g"].extend([j for _ in range(len(val))])
            vals_raw["v"].extend(list(val))
        vals = pd.DataFrame(vals_raw)
        ax = fig.add_subplot(n_rows, n_cols, i + 1)
        if n_groups == 1:
            sns.violinplot(
                y="v",
                x="g",
                data=vals,
                linewidth=1,
                orient="v",
                alpha=0.6,
                inner=None,
                cut=0,
                color=color,
            )
        else:
            sns.violinplot(
                y="v",
                x="g",
                data=vals,
                linewidth=1,
                orient="v",
                alpha=0.6,
                inner=None,
                cut=0,
                palette=cmap,
            )
        if len(vals) > max_points:
            sns.stripplot(
                x="g",
                y="v",
                data=vals.sample(n=max_points),
                jitter=0.4,
                ax=ax,
                orient="v",
                s=scatter_size,
                color="k",
                alpha=0.4,
            )
        else:
            sns.stripplot(
                x="g",
                y="v",
                data=vals,
                jitter=0.4,
                ax=ax,
                orient="v",
                s=scatter_size,
                color="k",
                alpha=0.4,
            )
        ax.set_ylabel(data.columns[i], fontsize=label_size)
        ax.set_xlabel("")
        if n_groups == 1:
            ax.set_xticks([])
            ax.set_xticklabels([])
        if data["groups"].nunique() == 1:
            ax.set_title(
                "Median: %.1f" % (int(np.median(vals["v"]))), fontsize=title_size
            )
        # clean_axis(ax)
        ax.figure.patch.set_alpha(0)
        ax.patch.set_alpha(0)
    if sup_title is not None:
        fig.suptitle(sup_title, fontsize=sup_title_size)
    plt.tight_layout()
    if show_fig:
        plt.show()
        return None
    return fig


def plot_mean_var(
    nzm: np.ndarray,
    fv: np.ndarray,
    n_cells: np.ndarray,
    hvg: np.ndarray,
    ax_label_fs: float = 12,
    fig_size: tuple[float, float] = (4.5, 4.0),
    ss: tuple[float, float] = (3, 30),
    cmaps: tuple[str, str] = ("winter", "magma_r"),
) -> None:
    """Show a mean-variance scatter plot with HVGs highlighted.

    Args:
        nzm: Non-zero mean expression per feature.
        fv: Variance (or corrected variance) per feature.
        n_cells: Number of cells expressing each feature (for coloring).
        hvg: Boolean mask of highly variable features.
        ax_label_fs: Axis label font size.
        fig_size: Figure size.
        ss: Scatter sizes for non-HVG and HVG points.
        cmaps: Colormaps for non-HVG and HVG points.
    """
    _, ax = plt.subplots(1, 1, figsize=fig_size)
    nzm = np.log2(nzm)
    fv = np.log2(fv)
    ax.scatter(nzm[~hvg], fv[~hvg], alpha=0.6, c=n_cells[~hvg], cmap=cmaps[0], s=ss[0])
    ax.scatter(
        nzm[hvg],
        fv[hvg],
        alpha=0.8,
        c=n_cells[hvg],
        cmap=cmaps[1],
        s=ss[1],
        edgecolor="k",
        lw=0.5,
    )
    ax.set_xlabel("Log mean non-zero expression", fontsize=ax_label_fs)
    ax.set_ylabel("Log corrected variance", fontsize=ax_label_fs)
    clean_axis(ax)
    plt.tight_layout()
    plt.show()


def plot_elbow(
    var_exp: npt.NDArray[Any] | list[float],
    figsize: tuple[float | None, float] = (None, 2),
) -> None:
    """Plot PCA variance explained with an automatic elbow marker.

    Args:
        var_exp: Percent variance explained per component.
        figsize: Figure size; width auto-scales with component count if None.
    """
    from kneed import KneeLocator  # type: ignore[import-not-found]

    x = range(len(var_exp))
    kneedle = KneeLocator(x, var_exp, S=1.0, curve="convex", direction="decreasing")
    if figsize[0] is None:
        figsize = (0.25 * len(var_exp), figsize[1])
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(x, var_exp, lw=1)
    ax.set_xticks(x)
    ax.axvline(kneedle.elbow, lw=1, c="r", label="Elbow")
    ax.set_ylabel("% Variance explained", fontsize=9)
    ax.set_xlabel("Principal components", fontsize=9)
    clean_axis(ax, ts=8)
    ax.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    plt.show()


def plot_heatmap(
    cdf: pd.DataFrame,
    fontsize: float = 10,
    width_factor: float = 0.03,
    height_factor: float = 0.02,
    cmap: Any = cm.matter_r,
    savename: str | None = None,
    save_dpi: int = 300,
    figsize: tuple[float, float] | None = None,
    show_fig: bool = True,
    **heatmap_kwargs: Any,
) -> Any:
    """Show a clustered heatmap of the input DataFrame.

    Args:
        cdf: Data to cluster and plot.
        fontsize: Base font size for auto figsize.
        width_factor: Width scaling per column.
        height_factor: Height scaling per row.
        cmap: Colormap name or object.
        savename: If set, save figure to this path.
        save_dpi: DPI for saved figure.
        figsize: Explicit figure size (auto if None).
        show_fig: Show interactively when True.
        **heatmap_kwargs: Extra kwargs passed to ``sns.clustermap``.
    """
    if figsize is None:
        figsize = (
            cdf.shape[1] * fontsize * width_factor,
            fontsize * cdf.shape[0] * height_factor,
        )
    cgx = sns.clustermap(
        cdf,
        yticklabels=cdf.index,
        xticklabels=cdf.columns,
        method="ward",
        figsize=figsize,
        cmap=cmap,
        rasterized=True,
        **heatmap_kwargs,
    )
    cgx.ax_heatmap.set_yticklabels(
        cdf.index[cgx.dendrogram_row.reordered_ind], fontsize=fontsize
    )
    cgx.ax_heatmap.set_xticklabels(
        cdf.columns[cgx.dendrogram_col.reordered_ind], fontsize=fontsize
    )
    cgx.ax_heatmap.figure.patch.set_alpha(0)
    cgx.ax_heatmap.patch.set_alpha(0)
    if savename:
        plt.savefig(savename, dpi=save_dpi)
    if show_fig:
        plt.show()
        return None
    return cgx


def _scatter_fix_type(v: pd.Series, ints_as_cats: bool) -> pd.Series:
    vt = v.dtype
    if v.nunique() == 1:
        return pd.Series(np.ones(len(v)), index=v.index).astype(np.float64)
    if vt in [bool, np.bool]:
        # converting first to int to handle bool
        return v.astype(int).astype("category")
    if (
        vt in [str, object]
        or vt.name in ("category", "string")
        or pd.api.types.is_string_dtype(v)
    ):
        return v.astype("category")
    elif np.issubdtype(vt.type, np.integer) and ints_as_cats:
        if v.nunique() > 100:
            logger.warning("Too many categories. set force_ints_as_cats to false")
        return v.astype(int).astype("category")
    else:
        return v.astype(np.float64)


def _scatter_fix_mask(
    v: pd.Series, mask_vals: list[Any] | None, mask_name: str
) -> pd.Series:
    if mask_vals is None:
        mask_vals = []
    mask_vals += [np.nan]
    iscat = False
    if v.dtype.name == "category":
        iscat = True
        v = v.astype(object)
    # There is a bug in pandas which causes failure above 1M rows
    # v[v.isin(mask_vals)] = mask_name
    v[np.isin(v, mask_vals)] = mask_name
    if iscat:
        v = v.astype("category")
    return v


def _scatter_make_colors(
    v: pd.Series,
    cmap: Any,
    color_key: dict[Any, Any] | None,
    mask_color: str,
    mask_name: str,
) -> tuple[Any | None, dict[Any, Any] | None]:
    from matplotlib.pyplot import get_cmap  # type: ignore[import-not-found]

    na_idx = v == mask_name
    uv = v[~na_idx].unique()

    if v.dtype.name != "category":
        if cmap is None:
            return cm.deep, None
        else:
            return get_cmap(cmap), None
    else:
        if cmap is None:
            cmap = "custom"

    if color_key is not None:
        for i in uv:
            if i not in color_key:
                raise KeyError(f"ERROR: key {i} missing in `color_key`")
        if na_idx.sum() > 0:
            if mask_name not in color_key:
                color_key[mask_name] = mpl.colors.to_hex(mask_color)
        return None, color_key
    else:
        if cmap == "custom" and len(uv) <= 102:
            if len(uv) <= 10:
                pal = custom_palettes[10]
            elif len(uv) <= 20:
                pal = custom_palettes[20]
            elif len(uv) <= 28:
                pal = custom_palettes[28]
            else:
                pal = custom_palettes[102]
        else:
            if cmap == "custom":
                cmap = None
            pal = sns.color_palette(palette=cmap, n_colors=len(uv)).as_hex()
        color_key = dict(zip(sorted(uv), pal))
        if na_idx.sum() > 0:
            color_key[mask_name] = mpl.colors.to_hex(mask_color)
        return None, color_key


def _scatter_cleanup(ax: Any, sw: float, sc: str, ds: tuple[str, ...]) -> None:
    for i in ["bottom", "left", "top", "right"]:
        spine = ax.spines[i]
        if i in ds:
            spine.set_visible(True)
            spine.set_linewidth(sw)
            spine.set_edgecolor(sc)
        else:
            spine.set_visible(False)
    ax.figure.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    ax.set_aspect("auto")
    return None


def _scatter_label_axis(df: pd.DataFrame, ax: Any, fs: float, fo: float) -> None:
    x, y = df.columns[:2]
    ax.set_xlabel(x, fontsize=fs)
    ax.set_ylabel(y, fontsize=fs)
    vmin, vmax = df[x].min(), df[x].max()
    ax.set_xlim((vmin - abs(vmin * fo), vmax + abs(vmax * fo)))
    vmin, vmax = df[y].min(), df[y].max()
    ax.set_ylim((vmin - abs(vmin * fo), vmax + abs(vmax * fo)))
    ax.set_xticks([])
    ax.set_yticks([])
    return None


def _scatter_legends(
    df: pd.DataFrame,
    ax: Any,
    cmap: Any,
    ck: dict[Any, Any] | None,
    ondata: bool,
    onside: bool,
    fontsize: float,
    title: str | None,
    title_fontsize: float,
    hide_title: bool,
    n_per_col: int,
    scale: float,
    ls: float,
    cs: float,
    cbs: float,
) -> None:
    """

    Args:
        df: dataframe
        ax: axis object
        cmap: color map
        ck: color key
        ondata: display legend over scatter plot?
        onside: display legend on side?
        fontsize: fontsize of legend text
        title: Title of subplot/axes
        hide_title: Whether to hide the title
        n_per_col: number of legends per column
        scale: scale legend marker size
        ls: line spacing
        cs: column spacing
        cbs: Cbar shrink factor

    Returns:

    """
    from matplotlib.colors import Normalize  # type: ignore[import-not-found]
    from matplotlib.colorbar import ColorbarBase, make_axes_gridspec  # type: ignore[import-not-found]

    x, y, vc = df.columns[:3]
    v = df[vc]
    cax = make_axes_gridspec(ax, location="top", shrink=cbs, aspect=25, fraction=0.1)[0]
    if v.nunique() <= 1:
        cax.set_axis_off()
        return None
    if v.dtype.name == "category":
        if hide_title is False:
            if title is not None:
                ax.title.set_text(title)
            else:
                ax.title.set_text(vc)
            ax.title.set_fontsize(title_fontsize)
        centers = df[[x, y, vc]].groupby(vc, observed=False).median().T
        for i in centers:
            if ondata:
                ax.text(
                    centers[i][x],
                    centers[i][y],
                    i,
                    fontsize=fontsize,
                    ha="center",
                    va="center",
                )
            if onside:
                assert ck is not None
                ax.scatter(
                    [float(centers[i][x])],
                    [float(centers[i][y])],
                    c=ck[i],
                    label=i,
                    alpha=1,
                    s=0.01,
                )
        if onside:
            n_cols = v.nunique() // n_per_col
            if v.nunique() % n_per_col > 0:
                n_cols += 1
            ax.legend(
                ncol=n_cols,
                loc=(1, 0),
                frameon=False,
                fontsize=fontsize,
                markerscale=scale,
                labelspacing=ls,
                columnspacing=cs,
            )
        cax.set_axis_off()
    else:
        norm = Normalize(vmin=v.min(), vmax=v.max())
        cb = ColorbarBase(cax, cmap=cmap, norm=norm, orientation="horizontal")
        if hide_title is False:
            if title is not None:
                cb.set_label(title, fontsize=title_fontsize)
            else:
                cb.set_label(vc, fontsize=title_fontsize)
        cb.ax.xaxis.set_label_position("bottom")
        cb.ax.xaxis.set_ticks_position("top")
        outline = cb.ax.spines.get("outline")
        if outline is not None:
            outline.set_visible(False)
    return None


def _make_grid(
    width: float,
    height: float,
    w_pad: float | None,
    h_pad: float | None,
    n_panels: int,
    n_columns: int,
) -> tuple[Any, npt.NDArray[Any]]:
    n_columns = np.minimum(n_panels, n_columns)
    n_rows = np.ceil(n_panels / n_columns).astype(int)
    if w_pad is None and h_pad is None:
        constrained = True
    else:
        constrained = False
    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(width * n_columns, height * n_rows),
        squeeze=False,
        constrained_layout=constrained,
    )
    diff = (n_rows * n_columns) - n_panels
    while diff > 0:
        fig.delaxes(axes[n_rows - 1, n_columns - diff])
        diff -= 1
    if not constrained:
        plt.tight_layout(w_pad=w_pad, h_pad=h_pad)

    return fig, axes


def _create_axes(
    dfs: list[pd.DataFrame],
    in_ax: npt.NDArray[Any] | Any | None,
    width: float,
    height: float,
    w_pad: float | None,
    h_pad: float | None,
    n_columns: int,
) -> npt.NDArray[Any]:
    if len(dfs) > 1:
        if in_ax is not None:
            logger.warning(
                "'in_ax' will not be used as multiple attributes will be plotted. Using internal grid"
                "layout"
            )
        _, axs = _make_grid(width, height, w_pad, h_pad, len(dfs), n_columns)
    else:
        if in_ax is None:
            _, axs = plt.subplots(1, 1, figsize=(width, height), squeeze=False)
        else:
            axs = in_ax
    return axs


def _iter_dataframes(
    dfs: list[pd.DataFrame],
    mask_values: list[Any] | None,
    mask_name: str,
    force_ints_as_cats: bool,
) -> Iterator[tuple[int, pd.DataFrame]]:
    for n, df in enumerate(dfs):
        vc = df.columns[2]
        v = _scatter_fix_mask(df[vc].copy(), mask_values, mask_name)
        df[vc] = _scatter_fix_type(v, force_ints_as_cats)
        yield n, df


def _handle_titles_type(titles: str | list[str] | None, n_df: int) -> list[str] | None:
    if titles is not None:
        if n_df > 1:
            if len(titles) != n_df or not isinstance(titles, list):
                logger.warning(
                    "Number of titles is not same as the the number of titles. Provided titles cannot be used"
                )
                titles = None
        else:
            if isinstance(titles, str):
                titles = [titles]
    return titles


def plot_scatter(
    dfs: list[pd.DataFrame],
    in_ax: npt.NDArray[Any] | Any | None = None,
    width: float = 6,
    height: float = 6,
    default_color: str = "steelblue",
    color_map: Any | None = None,
    color_key: dict[Any, Any] | None = None,
    mask_values: list[Any] | None = None,
    mask_name: str = "NA",
    mask_color: str = "k",
    point_size: float = 10,
    ax_label_size: float = 12,
    frame_offset: float = 0.05,
    spine_width: float = 0.5,
    spine_color: str = "k",
    displayed_sides: tuple[str, ...] = ("bottom", "left"),
    legend_ondata: bool = True,
    legend_onside: bool = True,
    legend_size: float = 12,
    legends_per_col: int = 20,
    titles: str | list[str] | None = None,
    title_size: int = 12,
    hide_title: bool = False,
    cbar_shrink: float = 0.6,
    marker_scale: float = 70,
    lspacing: float = 0.1,
    cspacing: float = 1,
    savename: str | None = None,
    dpi: int = 300,
    force_ints_as_cats: bool = True,
    n_columns: int = 4,
    w_pad: float = 1,
    h_pad: float = 1,
    show_fig: bool = True,
    scatter_kwargs: dict[str, Any] | None = None,
) -> npt.NDArray[Any] | None:
    """Show one or more 2D scatter plots from annotation DataFrames.

    Each DataFrame must contain x, y, and value columns. When multiple
    DataFrames are provided, plots are arranged in a grid.

    Args:
        dfs: List of DataFrames with columns [x, y, value].
        in_ax: Existing axes array to draw into (optional).
        width: Subplot width in inches.
        height: Subplot height in inches.
        default_color: Color for continuous values without a colormap.
        color_map: Matplotlib colormap name or object.
        color_key: Dict mapping category values to colors.
        mask_values: Values to render with ``mask_color``.
        mask_name: Label for masked values in legend.
        mask_color: Color for masked points.
        point_size: Scatter marker size.
        ax_label_size: Axis label font size.
        frame_offset: Axis limit padding fraction.
        spine_width: Spine line width.
        spine_color: Spine color.
        displayed_sides: Tuple of spines to show.
        legend_ondata: Draw category labels on data points.
        legend_onside: Draw legend table beside the plot.
        legend_size: Legend font size.
        legends_per_col: Max legend entries per column.
        titles: Subplot title or list of titles.
        title_size: Title font size.
        hide_title: Suppress titles when True.
        cbar_shrink: Colorbar shrink factor.
        marker_scale: Scale for on-data legend markers.
        lspacing: Legend label spacing.
        cspacing: Legend column spacing.
        savename: Save path (optional).
        dpi: Save DPI.
        force_ints_as_cats: Treat integer columns as categorical.
        n_columns: Grid columns when plotting multiple panels.
        w_pad: Width padding between subplots.
        h_pad: Height padding between subplots.
        show_fig: Show interactively when True.
        scatter_kwargs: Extra kwargs passed to ``ax.scatter`` (not ``c`` or ``s``).
    """
    from matplotlib.colors import to_hex  # type: ignore[import-not-found]

    def _handle_scatter_kwargs(sk: dict[str, Any] | None) -> dict[str, Any]:
        if sk is None:
            sk = {}
        if "c" in sk:
            logger.warning("scatter_kwarg value `c` will be ignored")
            del sk["c"]
        if "s" in sk:
            logger.warning("scatter_kwarg value `s` will be ignored")
            del sk["s"]
        if "lw" not in sk:
            sk["lw"] = 0.1
        if "edgecolors" not in sk:
            sk["edgecolors"] = "k"
        return sk

    titles = _handle_titles_type(titles, len(dfs))

    axs = _create_axes(dfs, in_ax, width, height, w_pad, h_pad, n_columns)
    for n, df in _iter_dataframes(dfs, mask_values, mask_name, force_ints_as_cats):
        v = df[df.columns[2]]
        col_map, col_key = _scatter_make_colors(
            v, color_map, color_key, mask_color, mask_name
        )
        if v.dtype.name == "category":
            assert col_key is not None
            df["c"] = [col_key[x] for x in v]
        else:
            if v.nunique() == 1:
                df["c"] = [default_color for _ in v]
            else:
                v = v.copy().fillna(0)
                mmv = (v - v.min()) / (v.max() - v.min())
                assert col_map is not None
                df["c"] = [to_hex(col_map(x)) for x in mmv]
        if "s" not in df:
            df["s"] = [point_size for _ in df.index]
        scatter_kwargs = _handle_scatter_kwargs(sk=scatter_kwargs)
        ax = axs[int(n / n_columns), n % n_columns]
        ax.scatter(
            df.values[:, 0],
            df.values[:, 1],
            c=df["c"].values,
            s=df["s"].values,
            rasterized=True,
            **scatter_kwargs,
        )
        _scatter_label_axis(df, ax, ax_label_size, frame_offset)
        _scatter_cleanup(ax, spine_width, spine_color, displayed_sides)
        if titles is not None:
            title = titles[n]
        else:
            title = None
        _scatter_legends(
            df,
            ax,
            col_map,
            col_key,
            legend_ondata,
            legend_onside,
            legend_size,
            title,
            title_size,
            hide_title,
            legends_per_col,
            marker_scale,
            lspacing,
            cspacing,
            cbar_shrink,
        )

    if savename:
        plt.savefig(savename, dpi=dpi, bbox_inches="tight")
    if show_fig:
        plt.show()
        return None
    return axs


def shade_scatter(
    dfs: list[pd.DataFrame],
    in_ax: npt.NDArray[Any] | Any | None = None,
    figsize: float = 6,
    pixels: int = 1000,
    spread_px: int = 1,
    spread_threshold: float = 0.2,
    min_alpha: int = 10,
    color_map: Any | None = None,
    color_key: dict[Any, Any] | None = None,
    mask_values: list[Any] | None = None,
    mask_name: str = "NA",
    mask_color: str = "k",
    ax_label_size: float = 12,
    frame_offset: float = 0.05,
    spine_width: float = 0.5,
    spine_color: str = "k",
    displayed_sides: tuple[str, ...] = ("bottom", "left"),
    legend_ondata: bool = True,
    legend_onside: bool = True,
    legend_size: float = 12,
    legends_per_col: int = 20,
    titles: str | list[str] | None = None,
    title_size: int = 12,
    hide_title: bool = False,
    cbar_shrink: float = 0.6,
    marker_scale: float = 70,
    lspacing: float = 0.1,
    cspacing: float = 1,
    savename: str | None = None,
    dpi: int = 300,
    force_ints_as_cats: bool = True,
    n_columns: int = 4,
    w_pad: float | None = None,
    h_pad: float | None = None,
    show_fig: bool = True,
) -> npt.NDArray[Any] | None:
    """Show datashader-density scatter plots for large cell embeddings.

    Args:
        dfs: List of DataFrames with columns [x, y, value].
        in_ax: Existing axes to draw into.
        figsize: Subplot width and height in inches.
        pixels: Canvas resolution for datashader aggregation.
        spread_px: Pixel spread for categorical shading.
        spread_threshold: Minimum fraction of pixels to apply spread.
        min_alpha: Minimum alpha for rendered pixels.
        color_map: Colormap for continuous values.
        color_key: Dict mapping categories to colors.
        mask_values: Values to treat as masked.
        mask_name: Label for masked category.
        mask_color: Color for masked values.
        ax_label_size: Axis label font size.
        frame_offset: Axis limit padding.
        spine_width: Spine line width.
        spine_color: Spine color.
        displayed_sides: Visible spines.
        legend_ondata: Draw labels on data.
        legend_onside: Draw side legend table.
        legend_size: Legend font size.
        legends_per_col: Legend entries per column.
        titles: Subplot title(s).
        title_size: Title font size.
        hide_title: Suppress titles.
        cbar_shrink: Colorbar shrink factor.
        marker_scale: On-data legend marker scale.
        lspacing: Legend label spacing.
        cspacing: Legend column spacing.
        savename: Save path (optional).
        dpi: Save DPI.
        force_ints_as_cats: Treat integers as categorical.
        n_columns: Grid columns for multiple panels.
        w_pad: Width padding between subplots.
        h_pad: Height padding between subplots.
        show_fig: Show interactively when True.
    """
    import datashader as dsh  # type: ignore[import-not-found]
    from datashader.mpl_ext import dsshow  # type: ignore[import-not-found]
    import datashader.transfer_functions as tf  # type: ignore[import-not-found]
    from functools import partial

    titles = _handle_titles_type(titles, len(dfs))
    axs = _create_axes(dfs, in_ax, figsize, figsize, w_pad, h_pad, n_columns)
    for n, df in _iter_dataframes(dfs, mask_values, mask_name, force_ints_as_cats):
        dim1, dim2, vc = df.columns[:3]
        v = df[vc]
        col_map, col_key = _scatter_make_colors(
            v, color_map, color_key, mask_color, mask_name
        )
        if v.dtype.name == "category":
            agg = dsh.count_cat(vc)
        else:
            if v.nunique() == 1:
                agg = dsh.count(vc)
            else:
                agg = dsh.mean(vc)

        ax = axs[int(n / n_columns), n % n_columns]
        dsshow(
            df,
            dsh.Point(dim1, dim2),
            aggregator=agg,
            norm="eq_hist",
            color_key=col_key,
            cmap=col_map,
            alpha_range=(min_alpha, 255),
            shade_hook=partial(
                tf.dynspread, threshold=spread_threshold, max_px=spread_px
            ),
            plot_height=pixels,
            plot_width=pixels,
            aspect="equal",
            width_scale=1,
            height_scale=1,
            ax=ax,
        )

        _scatter_label_axis(df, ax, ax_label_size, frame_offset)
        _scatter_cleanup(ax, spine_width, spine_color, displayed_sides)
        if titles is not None:
            title = titles[n]
        else:
            title = None
        _scatter_legends(
            df,
            ax,
            col_map,
            col_key,
            legend_ondata,
            legend_onside,
            legend_size,
            title,
            title_size,
            hide_title,
            legends_per_col,
            marker_scale,
            lspacing,
            cspacing,
            cbar_shrink,
        )

    if savename:
        plt.savefig(savename, dpi=dpi, bbox_inches="tight")
    if show_fig:
        plt.show()
        return None
    return axs


def _draw_pie(
    ax: Any,
    dist: npt.NDArray[Any],
    colors: list[Any],
    xpos: float,
    ypos: float,
    size: float,
) -> None:
    # https://stackoverflow.com/questions/56337732/how-to-plot-scatter-pie-chart-using-matplotlib
    cumsum = np.cumsum(dist)
    cumsum = cumsum / cumsum[-1]
    pie = [0] + cumsum.tolist()
    for r1, r2, c in zip(pie[:-1], pie[1:], colors):
        angles = np.linspace(2 * np.pi * r1, 2 * np.pi * r2)
        x = [0] + np.cos(angles).tolist()
        y = [0] + np.sin(angles).tolist()
        xy = np.column_stack([x, y])
        ax.scatter([xpos], [ypos], marker=xy, s=size, c=c)


def hierarchy_pos(
    g: Any,
    root: Any | None = None,
    width: float = 1.0,
    vert_gap: float = 0.2,
    vert_loc: float = 0,
    leaf_vs_root_factor: float = 0.5,
) -> dict[Any, tuple[float, float]]:
    """This function was lifted from here: https://github.com/springer-
    math/Mathematics-of-Epidemics-on-
    Networks/blob/80c8accbe0c6b7710c0a189df17529696ac31bf9/EoN/auxiliary.py.

    If the graph is a tree this will return the positions to plot this in a
    hierarchical layout.

    Based on Joel's answer at https://stackoverflow.com/a/29597209/2966723,
    but with some modifications.
    We include this because it may be useful for plotting transmission trees,
    and there is currently no networkx equivalent (though it may be coming soon).

    There are two basic approaches we think of to allocate the horizontal
    location of a node.

    - Top down: we allocate horizontal space to a node.  Then its ``k``
      descendants split up that horizontal space equally.  This tends to result
      in overlapping nodes when some have many descendants.
    - Bottom up: we allocate horizontal space to each leaf node.  A node at a
      higher level gets the entire space allocated to its descendant leaves.
      Based on this, leaf nodes at higher levels get the same space as leaf
      nodes very deep in the tree.

    We use use both of these approaches simultaneously with ``leaf_vs_root_factor``
    determining how much of the horizontal space is based on the bottom up
    or top down approaches.  ``0`` gives pure bottom up, while 1 gives pure top
    down.

    Args:
        g: the graph (must be a tree)
        root: the root node of the tree
              - if the tree is directed and this is not given, the root will be found and used
              - if the tree is directed and this is given, then the positions will be just for the descendants of
               this node.
              - if the tree is undirected and not given, then a random choice will be used.
        width: horizontal space allocated for this branch - avoids overlap with other branches
        vert_gap: gap between levels of hierarchy
        vert_loc: vertical location of root
        leaf_vs_root_factor: leaf_vs_root_factor
        xcenter: horizontal location of root
    """

    import networkx as nx

    if not nx.is_tree(g):
        raise TypeError("cannot use hierarchy_pos on a graph that is not a tree")

    if root is None:
        if isinstance(g, nx.DiGraph):
            root = next(
                iter(nx.topological_sort(g))
            )  # allows back compatibility with nx version 1.11
        else:
            root = np.random.choice(list(g.nodes))

    def _hierarchy_pos(
        g: Any,
        root: Any,
        leftmost: float,
        width: float,
        leafdx: float = 0.2,
        vert_gap: float = 0.2,
        vert_loc: float = 0,
        xcenter: float = 0.5,
        rootpos: dict[Any, tuple[float, float]] | None = None,
        leafpos: dict[Any, tuple[float, float]] | None = None,
        parent: Any | None = None,
    ) -> tuple[dict[Any, tuple[float, float]], dict[Any, tuple[float, float]], int]:
        """
        see hierarchy_pos docstring for most arguments
        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed
        """

        if rootpos is None:
            rootpos = {root: (xcenter, vert_loc)}
        else:
            rootpos[root] = (xcenter, vert_loc)
        if leafpos is None:
            leafpos = {}
        children = list(g.neighbors(root))
        leaf_count = 0
        if not isinstance(g, nx.DiGraph) and parent is not None:
            children.remove(parent)
        if len(children) != 0:
            rootdx = width / len(children)
            nextx = xcenter - width / 2 - rootdx / 2
            for child in children:
                nextx += rootdx
                rootpos, leafpos, newleaves = _hierarchy_pos(
                    g,
                    child,
                    leftmost + leaf_count * leafdx,
                    width=rootdx,
                    leafdx=leafdx,
                    vert_gap=vert_gap,
                    vert_loc=vert_loc - vert_gap,
                    xcenter=nextx,
                    rootpos=rootpos,
                    leafpos=leafpos,
                    parent=root,
                )
                leaf_count += newleaves

            leftmostchild = min((x for x, y in [leafpos[child] for child in children]))
            rightmostchild = max((x for x, y in [leafpos[child] for child in children]))
            leafpos[root] = ((leftmostchild + rightmostchild) / 2, vert_loc)
        else:
            leaf_count = 1
            leafpos[root] = (leftmost, vert_loc)
        #        pos[root] = (leftmost + (leaf_count-1)*dx/2., vert_loc)
        #        print(leaf_count)
        return rootpos, leafpos, leaf_count

    xcenter = width / 2.0
    if isinstance(g, nx.DiGraph):
        leafcount = len(
            [node for node in nx.descendants(g, root) if g.out_degree(node) == 0]
        )
    elif isinstance(g, nx.Graph):
        leafcount = len(
            [
                node
                for node in nx.node_connected_component(g, root)
                if g.degree(node) == 1 and node != root
            ]
        )
    rootpos, leafpos, leaf_count = _hierarchy_pos(
        g,
        root,
        0,
        width,
        leafdx=width * 1.0 / leafcount,
        vert_gap=vert_gap,
        vert_loc=vert_loc,
        xcenter=xcenter,
    )
    pos = {}
    for node in rootpos:
        pos[node] = (
            leaf_vs_root_factor * leafpos[node][0]
            + (1 - leaf_vs_root_factor) * rootpos[node][0],
            leafpos[node][1],
        )
    xmax = max(x for x, y in pos.values())
    for node in pos:
        pos[node] = (pos[node][0] * width / xmax, pos[node][1])
    return pos


def plot_cluster_hierarchy(
    sg: Any,
    clusts: Any,
    color_values: Any | None = None,
    force_ints_as_cats: bool = True,
    width: float = 2,
    lvr_factor: float = 0.5,
    vert_gap: float = 0.2,
    min_node_size: float = 10,
    node_size_multiplier: float = 1e4,
    node_power: float = 1,
    root_size: float = 100,
    non_leaf_size: float = 10,
    show_labels: bool = False,
    fontsize: float = 10,
    root_color: str = "#C0C0C0",
    non_leaf_color: str = "k",
    cmap: str | None = None,
    color_key: dict[Any, Any] | None = None,
    edgecolors: str = "k",
    edgewidth: float = 1,
    alpha: float = 0.7,
    figsize: tuple[float, float] = (5, 5),
    ax: Any | None = None,
    show_fig: bool = True,
    savename: str | None = None,
    save_dpi: int = 300,
) -> Any | None:
    """Plot a cluster hierarchy tree with colored leaf nodes.

    Args:
        sg: NetworkX graph of the cluster hierarchy.
        clusts: Cluster id per leaf cell.
        color_values: Values for leaf coloring (default: cluster ids).
        force_ints_as_cats: Treat integer color values as categorical.
        width: Horizontal layout width for hierarchy positioning.
        lvr_factor: Blend between leaf-only and root-based layout (0-1).
        vert_gap: Vertical gap between hierarchy levels.
        min_node_size: Minimum node marker size.
        node_size_multiplier: Scale factor for node sizes.
        node_power: Exponent applied to node leaf counts for sizing.
        root_size: Marker size for root node.
        non_leaf_size: Marker size for internal nodes.
        show_labels: Draw cluster labels on internal nodes.
        fontsize: Label font size.
        root_color: Root node color.
        non_leaf_color: Internal node color.
        cmap: Colormap for continuous color values.
        color_key: Dict mapping categories to colors.
        edgecolors: Node edge color.
        edgewidth: Node edge width.
        alpha: Node alpha.
        figsize: Figure size when ``ax`` is None.
        ax: Existing axis to draw into.
        show_fig: Show interactively when True.
        savename: Save path (optional).
        save_dpi: Save DPI.

    Returns:
        Matplotlib Axes when ``show_fig`` is False.
    """
    import networkx as nx
    import math
    from matplotlib.colors import to_hex  # type: ignore[import-not-found]

    if color_values is None:
        color_values = pd.Series(clusts)
        using_clust_for_colors = True
    else:
        color_values = pd.Series(color_values)
        using_clust_for_colors = False
    color_values = _scatter_fix_type(color_values, force_ints_as_cats)
    colormap, color_key = _scatter_make_colors(
        color_values, cmap, color_key, "k", "longdummyvaluesofh3489hfpiqehdcbla"
    )
    pos = hierarchy_pos(
        sg, width=width * math.pi, leaf_vs_root_factor=lvr_factor, vert_gap=vert_gap
    )
    new_pos = {
        u: (r * math.cos(theta), r * math.sin(theta)) for u, (theta, r) in pos.items()
    }

    if color_key is None:
        cluster_values = (
            pd.DataFrame({"clusters": clusts, "v": color_values})
            .groupby("clusters", observed=False)
            .mean()["v"]
        )
        mmv: pd.Series = (cluster_values - cluster_values.min()) / (
            cluster_values.max() - cluster_values.min()
        )
        assert colormap is not None
        color_key = {k: to_hex(colormap(v)) for k, v in mmv.to_dict().items()}
    else:
        cluster_values = None

    assert color_key is not None

    cs = pd.Series(clusts).value_counts()
    cs = (node_size_multiplier * ((cs / cs.sum()) ** node_power)).to_dict()
    nc, ns = [], []

    for i in sg.nodes():
        if "partition_id" in sg.nodes[i]:
            clust_id = sg.nodes[i]["partition_id"]
            if cluster_values is not None or using_clust_for_colors:
                nc.append(color_key[clust_id])
                ns.append(max(cs[clust_id], min_node_size))
            else:
                nc.append("white")
                ns.append(0)
        else:
            if sg.nodes[i]["nleaves"] == len(clusts):
                nc.append(root_color)
                ns.append(root_size)
            else:
                nc.append(non_leaf_color)
                ns.append(non_leaf_size)
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    nx.draw(
        sg,
        pos=new_pos,
        node_size=ns,
        node_color=nc,
        ax=ax,
        edgecolors=edgecolors,
        alpha=alpha,
        linewidths=edgewidth,
    )

    if cluster_values is None and using_clust_for_colors is False:
        for i in sg.nodes():
            if "partition_id" in sg.nodes[i]:
                clust_id = sg.nodes[i]["partition_id"]
                idx = clusts == clust_id
                counts = color_values[idx].value_counts()
                _draw_pie(
                    ax,
                    counts.values,
                    [color_key[x] for x in counts.index],
                    new_pos[i][0],
                    new_pos[i][1],
                    max(cs[clust_id], min_node_size),
                )

    if show_labels:
        for i in sg.nodes():
            if "partition_id" in sg.nodes[i]:
                clust_id = sg.nodes[i]["partition_id"]
                ax.text(
                    new_pos[i][0],
                    new_pos[i][1],
                    clust_id,
                    fontsize=fontsize,
                    ha="center",
                    va="center",
                )
    if savename:
        plt.savefig(savename, dpi=save_dpi)
    if show_fig:
        plt.show()
        return None
    return ax


def plot_annotated_heatmap(
    df: npt.NDArray[Any],
    xbar_values: npt.NDArray[Any],
    ybar_values: npt.NDArray[Any],
    display_row_labels: list[str] | None = None,
    row_labels: list[str] | None = None,
    width: int = 5,
    height: int = 10,
    vmin: float = -2.0,
    vmax: float = 2.0,
    heatmap_cmap: str | None = None,
    xbar_cmap: Any | None = None,
    ybar_cmap: str | None = None,
    tick_fontsize: int = 10,
    axis_fontsize: int = 12,
    row_label_fontsize: int = 12,
    savename: str | None = None,
    save_dpi: int = 300,
    show_fig: bool = True,
) -> None:
    """Plot a heatmap with pseudotime and cluster annotation bars.

    Args:
        df: 2D expression matrix (features x ordering bins).
        xbar_values: Values for the bottom pseudotime bar.
        ybar_values: Values for the right cluster color bar.
        display_row_labels: Subset of row labels to show on the heatmap.
        row_labels: Labels for all rows (default: index strings).
        width: Figure width in inches.
        height: Figure height in inches.
        vmin: Heatmap color scale minimum.
        vmax: Heatmap color scale maximum.
        heatmap_cmap: Colormap for the main heatmap.
        xbar_cmap: Colormap for the pseudotime bar.
        ybar_cmap: Colormap for the cluster bar.
        tick_fontsize: Colorbar tick font size.
        axis_fontsize: Title font size.
        row_label_fontsize: Row label font size.
        savename: Save path (optional).
        save_dpi: Save DPI.
        show_fig: Show interactively when True.
    """
    import matplotlib.ticker as mticker  # type: ignore[import-not-found]

    if display_row_labels is None:
        display_row_labels = []
    if row_labels is None:
        row_labels = list(map(str, range(df.shape[0])))
    else:
        if len(row_labels) != df.shape[0]:
            raise ValueError(
                "ERROR: Number of provided feature labels and size of the data array does not match"
            )

    whr = height / width
    fig = plt.figure(constrained_layout=False, figsize=(width, height))
    gs = fig.add_gridspec(nrows=int(20 * whr), ncols=20, wspace=0, hspace=0)
    heatmap_ax = fig.add_subplot(gs[:-2, 1:16])
    clustbar_ax = fig.add_subplot(gs[:-2, 17:18])
    cbar_ax = fig.add_subplot(gs[round(7 * whr) : round(12 * whr), -1:])
    ptime_ax = fig.add_subplot(gs[-1:, 1:16])

    if heatmap_cmap is None:
        heatmap_cmap = "coolwarm"
    sns.heatmap(
        df,
        ax=heatmap_ax,
        cbar_ax=cbar_ax,
        xticklabels=[],
        yticklabels=[],
        vmin=vmin,
        vmax=vmax,
        cmap=heatmap_cmap,
    )

    if len(display_row_labels) > 0:
        row_label_index = {x.lower(): n for n, x in enumerate(row_labels)}
        display_row_labels = [
            x for x in display_row_labels if x.lower() in row_label_index
        ]
        heatmap_ax.set_yticks([row_label_index[x.lower()] for x in display_row_labels])
        heatmap_ax.set_yticklabels(display_row_labels, fontsize=row_label_fontsize)

    heatmap_ax.set_title(f"{df.shape[0]} features", fontsize=axis_fontsize)
    ticks_loc = cbar_ax.get_yticks().tolist()
    cbar_ax.yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
    cbar_ax.set_yticklabels([x for x in ticks_loc], fontsize=tick_fontsize)

    if ybar_cmap is None:
        ybar_cmap = "tab20"
    clustbar_ax.imshow(ybar_values.reshape(-1, 1), aspect="auto", cmap=ybar_cmap)
    clustbar_ax.set_xticks([])
    clustbar_ax.set_yticks([])

    for i in set(ybar_values):
        y = np.where(ybar_values == i)[0].mean()
        clustbar_ax.text(
            0,
            y,
            i,
            fontsize=axis_fontsize,
            ha="center",
            va="center",
        )

    binned_ptime = [x.mean() for x in np.array_split(sorted(xbar_values), df.shape[1])]
    if xbar_cmap is None:
        xbar_cmap = cm.deep
    ptime_ax.imshow([binned_ptime], aspect="auto", cmap=xbar_cmap)
    ptime_ax.set_xticks([])
    ptime_ax.set_yticks([])
    ptime_ax.set_xlabel("------ Pseudotime----->", fontsize=axis_fontsize)
    if savename is not None:
        plt.savefig(savename, dpi=save_dpi)
    if show_fig:
        plt.show()
