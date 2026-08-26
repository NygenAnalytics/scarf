"""Distribution plots for metadata and feature values."""

import warnings
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any, Hashable, Literal

import numpy as np
import pandas as pd

from ._contracts import (
    CategoricalScale,
    CellField,
    ColorScale,
    DistKind,
    FeatureRef,
    NormalizationSpec,
    PlotProvenance,
    StudyDesign,
)
from ._data import (
    fetch_normalized_feature_matrix,
    resolve_cell_selection,
    resolve_feature,
)
from ._deps import require_matplotlib, require_seaborn
from ._display import resolve_categorical_scale
from ._figure import LegendSpec, PlotResult, normalize_axes_target
from ._style import (
    MAX_FIGURE_WIDTH_INCHES,
    apply_figure_chrome,
    capped_figsize,
    categorical_color_map,
    continuous_norm,
    sort_categories,
    theme_context,
)


def _cell_index(store: Any, cell_key: str | None) -> np.ndarray:
    if cell_key is None:
        return np.arange(store.cells.N, dtype=np.int64)
    return np.asarray(store.cells.active_index(cell_key))


def _fetch_cell_column(store: Any, column: str, cell_key: str | None) -> np.ndarray:
    if cell_key is None:
        return np.asarray(store.cells.fetch_all(column))
    return np.asarray(store.cells.fetch(column, key=cell_key))


def _fetch_series(
    store: Any,
    key: str | CellField | FeatureRef,
    *,
    cell_key: str | None,
    from_assay: str | None,
    normalization: NormalizationSpec,
) -> tuple[np.ndarray, str, bool]:
    """Return (values, label, is_feature)."""
    if isinstance(key, CellField):
        return (
            _fetch_cell_column(store, key.key, cell_key),
            key.label or key.key,
            False,
        )
    if isinstance(key, FeatureRef) or (
        isinstance(key, str) and key not in store.cells.columns
    ):
        resolved = resolve_feature(store, key, from_assay=from_assay)
        cell_idx = _cell_index(store, cell_key)
        mat = fetch_normalized_feature_matrix(
            store,
            [resolved],
            cell_idx,
            normalization=normalization,
        )
        return mat[:, 0], resolved.label, True
    return _fetch_cell_column(store, str(key), cell_key), str(key), False


def _subsample_frame(
    df: pd.DataFrame,
    *,
    max_points: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, bool]:
    """Return (frame, was_subsampled)."""
    if max_points <= 0 or len(df) <= max_points:
        return df, False
    idx = rng.choice(len(df), size=max_points, replace=False)
    return df.iloc[np.sort(idx)], True


def _group_palette(
    order: list[Any],
    categorical_scale: CategoricalScale | None,
) -> dict[Any, str]:
    return categorical_color_map(
        order,
        palette=categorical_scale.palette if categorical_scale else None,
        palette_name=(
            categorical_scale.palette_name if categorical_scale else "default"
        ),
        missing_label=None,
    )


@contextmanager
def _seed_legacy_numpy(rng: np.random.Generator) -> Iterator[None]:
    state = np.random.get_state()
    np.random.seed(int(rng.integers(0, np.iinfo(np.uint32).max)))
    try:
        yield
    finally:
        np.random.set_state(state)


def _draw_violin_or_box(
    ax: Any,
    sns: Any,
    df: pd.DataFrame,
    *,
    kind: Literal["violin", "box"],
    color: str,
    max_points: int,
    point_size: float,
    rng: np.random.Generator,
    order: list[Any] | None = None,
    palette: dict[Any, str] | None = None,
    orientation: Literal["vertical", "horizontal"] = "vertical",
    violin_inner: str | None = "quartile",
    linewidth: float = 0.8,
    fill_alpha: float = 0.9,
    point_alpha: float = 0.28,
    split_order: list[Any] | None = None,
    split_palette: dict[Any, str] | None = None,
    show_legend: bool = True,
) -> bool:
    plot_kw: dict[str, Any] = {}
    if order is not None:
        plot_kw["order"] = order
    split = (
        split_order is not None and split_palette is not None and len(split_order) == 2
    )
    grouped = order is not None and palette is not None and len(order) > 0
    if split:
        plot_kw["hue"] = "split"
        plot_kw["hue_order"] = split_order
        plot_kw["palette"] = split_palette
        plot_kw["legend"] = show_legend
    elif grouped:
        plot_kw["hue"] = "group"
        plot_kw["hue_order"] = order
        plot_kw["palette"] = palette
        plot_kw["legend"] = False
        plot_kw["dodge"] = False

    x_key = "group" if orientation == "vertical" else "value"
    y_key = "value" if orientation == "vertical" else "group"
    collections_before = len(ax.collections)
    if kind == "violin":
        sns.violinplot(
            data=df,
            x=x_key,
            y=y_key,
            ax=ax,
            color=None if grouped else color,
            inner=violin_inner,
            cut=0,
            linewidth=linewidth,
            saturation=0.9,
            split=split,
            **plot_kw,
        )
    else:
        sns.boxplot(
            data=df,
            x=x_key,
            y=y_key,
            ax=ax,
            color=None if grouped else color,
            showfliers=max_points <= 0,
            linewidth=linewidth,
            fliersize=2,
            **plot_kw,
        )
    for collection in ax.collections[collections_before:]:
        collection.set_alpha(fill_alpha)

    subsampled = False
    if max_points > 0 and len(df) > 0:
        pts, subsampled = _subsample_frame(df, max_points=max_points, rng=rng)
        strip_kw = {k: v for k, v in plot_kw.items() if k != "legend"}
        if split:
            strip_kw["legend"] = False
            strip_kw["dodge"] = True
        elif "palette" in strip_kw:
            strip_kw.pop("palette", None)
            strip_kw.pop("hue", None)
        with _seed_legacy_numpy(rng):
            sns.stripplot(
                data=pts,
                x=x_key,
                y=y_key,
                ax=ax,
                color=None if split else "0.15",
                size=point_size,
                jitter=0.28,
                alpha=point_alpha,
                **(
                    strip_kw
                    if split
                    else {k: v for k, v in strip_kw.items() if k == "order"}
                ),
            )
    return subsampled


def _sample_aggregate(
    frame: pd.DataFrame,
    *,
    statistic: Literal["mean", "median", "fraction"],
    expression_cutoff: float,
    split: bool,
) -> pd.DataFrame:
    valid_sample = pd.notna(frame["sample"]) & (frame["sample"].astype(str) != "")
    frame = frame.loc[valid_sample].copy()
    if frame.empty:
        raise ValueError("No selected cells have a valid sample value")
    columns = ["sample", "group"]
    if split:
        columns.append("split")
    grouped = frame.groupby(columns, observed=False, dropna=False)["raw_value"]
    if statistic == "mean":
        values = grouped.mean()
    elif statistic == "median":
        values = grouped.median()
    else:
        values = grouped.apply(
            lambda value: float(
                np.mean(value.to_numpy(dtype=np.float64) > expression_cutoff)
            )
        )
    counts = grouped.size().rename("nCells")
    return pd.concat(
        (values.rename("raw_value"), counts),
        axis=1,
    ).reset_index()


def _draw_hist(
    ax: Any,
    df: pd.DataFrame,
    *,
    color: str,
    bins: int,
    group_by: str | None,
    order: list[Any] | None,
    palette: dict[Any, str] | None,
    show_legend: bool,
) -> None:
    if group_by is None:
        vals = df["value"].to_numpy(dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        ax.hist(vals, bins=bins, color=color, edgecolor="white", linewidth=0.4)
        return
    all_values = df["value"].to_numpy(dtype=np.float64)
    all_values = all_values[np.isfinite(all_values)]
    shared_bins = (
        np.histogram_bin_edges(all_values, bins=bins) if len(all_values) else bins
    )
    levels = order if order is not None else list(pd.unique(df["group"]))
    for g in levels:
        sub = df[df["group"] == g]
        vals = sub["value"].to_numpy(dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        face = palette[g] if palette is not None and g in palette else None
        ax.hist(
            vals,
            bins=shared_bins,
            alpha=0.45,
            label=str(g),
            color=face,
            edgecolor="white",
            linewidth=0.3,
        )
    if show_legend:
        ax.legend(frameon=False, fontsize=8)


def _ecdf_xy(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    v = np.sort(values[np.isfinite(values)])
    if len(v) == 0:
        return np.array([]), np.array([])
    y = np.arange(1, len(v) + 1, dtype=np.float64) / len(v)
    return v, y


def _draw_ecdf(
    ax: Any,
    df: pd.DataFrame,
    *,
    color: str,
    max_points: int,
    rng: np.random.Generator,
    group_by: str | None,
    order: list[Any] | None,
    palette: dict[Any, str] | None,
    show_legend: bool,
) -> bool:
    """Draw ECDF. May subsample values for display; returns whether subsampled."""
    subsampled = False
    if group_by is None:
        vals = df["value"].to_numpy(dtype=np.float64)
        if max_points > 0 and len(vals) > max_points:
            vals = vals[rng.choice(len(vals), size=max_points, replace=False)]
            subsampled = True
        x, y = _ecdf_xy(vals)
        if len(x):
            ax.step(x, y, where="post", color=color, linewidth=1.5)
        return subsampled

    levels = order if order is not None else list(pd.unique(df["group"]))
    for g in levels:
        sub = df[df["group"] == g]
        vals = sub["value"].to_numpy(dtype=np.float64)
        if max_points > 0 and len(vals) > max_points:
            vals = vals[rng.choice(len(vals), size=max_points, replace=False)]
            subsampled = True
        x, y = _ecdf_xy(vals)
        if len(x):
            line_color = palette[g] if palette is not None and g in palette else None
            ax.step(
                x,
                y,
                where="post",
                linewidth=1.2,
                label=str(g),
                color=line_color,
            )
    if show_legend:
        ax.legend(frameon=False, fontsize=8)
    ax.set_ylim(-0.02, 1.02)
    return subsampled


def _panel_display_frame(
    values: np.ndarray,
    groups_arr: np.ndarray,
    *,
    split_arr: np.ndarray | None,
    sample_arr: np.ndarray | None,
    sample_stat: Literal["mean", "median", "fraction"],
    expression_cutoff: float,
    row_standardize: bool,
) -> pd.DataFrame:
    """Build the per-panel display frame (value, group[, split][, sample])."""
    cell_frame = pd.DataFrame(
        {
            "raw_value": np.asarray(values, dtype=np.float64),
            "group": groups_arr,
        }
    )
    if split_arr is not None:
        cell_frame["split"] = split_arr
    if sample_arr is not None:
        cell_frame["sample"] = sample_arr
        frame = _sample_aggregate(
            cell_frame,
            statistic=sample_stat,
            expression_cutoff=expression_cutoff,
            split=split_arr is not None,
        )
    else:
        frame = cell_frame
    display_values = frame["raw_value"].to_numpy(dtype=np.float64)
    if row_standardize:
        finite = np.isfinite(display_values)
        mean = float(np.mean(display_values[finite])) if finite.any() else 0.0
        std = float(np.std(display_values[finite])) if finite.any() else 0.0
        display_values = (
            (display_values - mean) / std if std > 0 else np.zeros_like(display_values)
        )
    frame["value"] = display_values
    return frame


def _panel_group_means(frame: pd.DataFrame) -> pd.Series:
    """Per-group mean of the display values in one panel."""
    return frame.groupby("group", observed=False)["value"].mean()


def _mean_color_limits(
    means_by_panel: Sequence[pd.Series],
    color_scale: ColorScale,
) -> tuple[list[tuple[float, float]], tuple[float, float]]:
    """Resolve per-panel and reference colour limits from group means.

    ``scope="shared"`` gives every stacked row the same limits derived from all
    pooled group means, so the rows share one continuous scale. Explicit
    ``vmin`` / ``vmax`` override quantile or observed bounds; a ``vcenter``
    pivot extends derived bounds so diverging maps work on one-sided data.

    ``scope="panel"`` rescales each row independently by the strict min/max of
    that row's own group means, so the lowest mean maps to 0 and the highest to
    1. Quantiles, bounds, and pivots are not applied per panel; the shared
    reference for the unit colorbar is ``(0, 1)``.
    """
    if color_scale.scope not in ("shared", "panel"):
        raise ValueError(
            "color_scale.scope must be 'shared' or 'panel' for mean coloring; "
            f"got {color_scale.scope!r}"
        )

    arrays = [
        np.asarray(means.to_numpy(dtype=np.float64), dtype=np.float64)
        for means in means_by_panel
    ]
    pooled = np.concatenate(arrays) if arrays else np.array([], dtype=np.float64)
    if pooled[np.isfinite(pooled)].size == 0:
        raise ValueError("No finite expression values to colour by")

    if color_scale.scope == "panel":
        limits: list[tuple[float, float]] = []
        for array in arrays:
            finite = array[np.isfinite(array)]
            if finite.size == 0:
                limits.append((0.0, 1.0))
            else:
                limits.append((float(np.nanmin(finite)), float(np.nanmax(finite))))
        return limits, (0.0, 1.0)

    def resolve(values: np.ndarray) -> tuple[float, float]:
        finite = np.asarray(values, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            return 0.0, 1.0
        if color_scale.quantiles is not None:
            q0, q1 = color_scale.quantiles
            lo = float(np.nanquantile(finite, q0))
            hi = float(np.nanquantile(finite, q1))
        else:
            lo = float(np.nanmin(finite))
            hi = float(np.nanmax(finite))
        if color_scale.vmin is not None:
            lo = color_scale.vmin
        if color_scale.vmax is not None:
            hi = color_scale.vmax
        if hi < lo:
            raise ValueError("vmax must be greater than or equal to vmin")
        if color_scale.vcenter is not None:
            if not lo < color_scale.vcenter < hi:
                if color_scale.vmin is not None or color_scale.vmax is not None:
                    raise ValueError(
                        "vcenter must be strictly between the resolved color limits "
                        "when vmin/vmax are explicit"
                    )
                lo = min(lo, color_scale.vcenter)
                hi = max(hi, color_scale.vcenter)
                span = hi - lo
                eps = max(span * 1e-6, 1e-9)
                if color_scale.vcenter <= lo:
                    lo = color_scale.vcenter - eps
                if color_scale.vcenter >= hi:
                    hi = color_scale.vcenter + eps
        return lo, hi

    reference = resolve(pooled)
    return [reference] * len(arrays), reference


def _format_stat_p_text(p_value: float, show_p_value: bool) -> str:
    if not show_p_value:
        if p_value <= 0.001:
            return "***"
        if p_value <= 0.01:
            return "**"
        if p_value <= 0.05:
            return "*"
        return "ns"
    if p_value < 0.001:
        return f"p={p_value:.1e}"
    return f"p={p_value:.2g}"


def _annotate_distribution_stats(
    ax: Any,
    frame: pd.DataFrame,
    *,
    method: str,
    group_order: list[Any],
    orientation: Literal["vertical", "horizontal"],
    bracket_height: float,
    show_p_value: bool,
) -> bool:
    """Draw significance brackets over an existing violin/box panel.

    Pure matplotlib: ``ax.plot`` plus ``ax.text`` on positions derived from
    the seaborn category order, so the seaborn backend stays untouched.
    Pairwise tables provide their own groups; a one-way ANOVA omnibus table
    spans every displayed group. Returns whether anything was drawn.
    """
    p_column = "p_value_adjusted" if "p_value_adjusted" in frame.columns else "p_value"
    if p_column not in frame.columns or frame.empty:
        return False
    grouped_categories = ["group_1", "group_2"]
    pairwise = all(column in frame.columns for column in grouped_categories)
    if pairwise:
        rows: list[tuple[int, int, float]] = []
        for _, row in frame.iterrows():
            try:
                pos_1 = group_order.index(row["group_1"])
                pos_2 = group_order.index(row["group_2"])
            except ValueError:
                continue
            p_value = float(row[p_column])
            if np.isfinite(p_value):
                rows.append((pos_1, pos_2, p_value))
    elif method == "one_way_anova":
        p_value = float(frame.iloc[0][p_column])
        rows = (
            [(0, len(group_order) - 1, p_value)]
            if len(group_order) > 1 and np.isfinite(p_value)
            else []
        )
    else:
        return False
    if not rows:
        return False
    step = bracket_height * 1.6
    text_offset = bracket_height / 3
    any_drawn = False
    if orientation == "vertical":
        value_bottom, base_top = ax.get_ylim()
        highest = base_top
        for level, (pos_1, pos_2, p_value) in enumerate(rows):
            y = base_top + level * step
            elbow = y + bracket_height / 3
            ax.plot(
                [pos_1, pos_1, pos_2, pos_2],
                [y, elbow, elbow, y],
                color="black",
                lw=0.9,
                clip_on=False,
            )
            ax.text(
                (pos_1 + pos_2) / 2,
                elbow + text_offset,
                _format_stat_p_text(p_value, show_p_value),
                ha="center",
                va="bottom",
                fontsize=7,
            )
            highest = max(highest, elbow + step)
        ax.set_ylim(value_bottom, max(float(base_top), highest))
        any_drawn = True
    else:
        value_left, base_right = ax.get_xlim()
        farthest = base_right
        for level, (pos_1, pos_2, p_value) in enumerate(rows):
            x = base_right + level * step
            elbow = x + bracket_height / 3
            ax.plot(
                [x, elbow, elbow, x],
                [pos_1, pos_1, pos_2, pos_2],
                color="black",
                lw=0.9,
                clip_on=False,
            )
            ax.text(
                elbow + text_offset,
                (pos_1 + pos_2) / 2,
                _format_stat_p_text(p_value, show_p_value),
                ha="left",
                va="center",
                fontsize=7,
            )
            farthest = max(farthest, elbow + step)
        ax.set_xlim(float(value_left), max(float(base_right), farthest))
        any_drawn = True
    return any_drawn


def _render_color_limits(lo: float, hi: float) -> tuple[float, float]:
    """Return colourbar limits, padding degenerate scales so a bar renders."""
    if hi > lo:
        return lo, hi
    pad = max(0.5, abs(lo) * 0.05)
    return lo - pad, hi + pad


def _mean_colorbar_label(color_scale: ColorScale, row_standardize: bool) -> str:
    """Label for the mean-expression colourbar, adapted to the value scale."""
    if color_scale.scope == "panel":
        return "Relative Expression Per Gene"
    return "mean standardized value" if row_standardize else "mean expression"


def _mean_group_palette(
    means: pd.Series,
    order: Sequence[Any],
    *,
    color_scale: ColorScale,
    lo: float,
    hi: float,
) -> dict[Any, str]:
    """Map each group to a colour by its mean expression in ``means``."""
    from matplotlib import colormaps
    from matplotlib.colors import to_hex

    _, mpl = require_matplotlib()
    cmap = color_scale.cmap or "viridis"
    if cmap not in colormaps:
        raise ValueError(f"Unknown colormap {cmap!r}")
    span = hi - lo
    norm = continuous_norm(mpl, vmin=lo, vmax=hi, vcenter=color_scale.vcenter)
    palette_map: dict[Any, str] = {}
    for group in order:
        mean = means.get(group)
        if mean is None or not np.isfinite(float(mean)):
            palette_map[group] = color_scale.missing_color
        else:
            t = 0.5 if span == 0 else float(norm(float(mean), clip=True))
            palette_map[group] = to_hex(colormaps[cmap](t))
    return palette_map


def distribution(
    store: Any,
    keys: str | CellField | FeatureRef | Sequence[str | CellField | FeatureRef],
    *,
    group_by: str | None = None,
    groups: Sequence[Any] | None = None,
    split_by: str | None = None,
    sample_by: str | None = None,
    study_design: StudyDesign | None = None,
    sample_stat: Literal["mean", "median", "fraction"] = "mean",
    expression_cutoff: float = 0.0,
    subset_by: str | None = None,
    cell_key: str | None = "I",
    from_assay: str | None = None,
    normalization: NormalizationSpec | None = None,
    categorical_scale: CategoricalScale | None = None,
    split_scale: CategoricalScale | None = None,
    kind: DistKind = "violin",
    bins: int = 40,
    max_points: int | None = None,
    point_size: float = 0.8,
    point_alpha: float = 0.28,
    seed: int = 0,
    color: str = "steelblue",
    color_by: Literal["group", "mean"] = "group",
    color_scale: ColorScale | None = None,
    orientation: Literal["vertical", "horizontal"] = "vertical",
    row_standardize: bool = False,
    share_y: bool | None = None,
    violin_inner: str | None = "quartile",
    violin_linewidth: float = 0.8,
    violin_alpha: float = 0.9,
    italicize_features: bool = False,
    target: Any | None = None,
    figsize: tuple[float, float] | None = None,
    max_figure_width: float | None = MAX_FIGURE_WIDTH_INCHES,
    title: str | None = None,
    theme: str = "notebook",
    show_legend: bool = True,
    stats_results: Any = None,
    stats_keys: Sequence[str] | None = None,
    stats_method: str | None = None,
    stats_bracket_height: float | None = None,
    stats_show_p: bool = True,
    show: bool = True,
) -> PlotResult:
    """Compare value distributions for QC metrics or genes.

    ``keys`` may be cell-metadata columns (for example ``RNA_nCounts``) or gene
    names. ``kind`` selects the display: ``"violin"``, ``"box"``, ``"hist"``,
    or ``"ecdf"``. With ``group_by``, each category gets its own distribution
    along the x-axis and a distinct color.

    Cell selection (same knobs as embeddings):

    - ``cell_key``: boolean metadata column selecting cells (default ``"I"``).
      Pass ``None`` to include every cell, including those marked inactive.
    - ``subset_by``: boolean metadata column; keep ``True`` cells
    - ``groups``: keep / order these ``group_by`` categories

    For violins and boxes, Scarf can overlay a subsample of cells as points.
    ``max_points`` limits how many points are drawn (``0`` disables points;
    ``None`` selects the kind default). Stacked violins default to no overlay
    (``0``), while other kinds default to ``10000``. Histograms always use
    every selected cell. Regular multi-gene violin and box panels share their
    value scale by default. Stacked violins keep independent scales unless
    ``share_y=True``. With horizontal orientation, the same option shares the
    x-axis value scale.

    Set ``sample_by`` or ``study_design`` to summarize cells within biological
    samples before plotting. ``split_by`` draws two violin halves for a second
    categorical variable.

    Pass ``stats_results`` (a single
    :class:`~scarf.features.statistical.StatisticalTestResult`, or a mapping
    from panel label to one) to annotate significance brackets over the drawn
    violins or boxes. Brackets are pure matplotlib overlays positioned by the
    displayed category order; pairwise tables place one bracket per row and a
    one-way ANOVA omnibus result spans every group. The annotation prefers
    ``p_value_adjusted`` when present. Nothing is recomputed here; when
    ``stats_results`` does not match the plotted selection, panels are skipped
    with a warning instead of raising.

    For ``kind="stacked_violin"``, pass ``color_by="mean"`` to colour each
    stacked violin by its group mean expression on a continuous scale. Pass a
    ``ColorScale`` to control the mapping: ``cmap`` chooses the colormap,
    ``vmin``/``vmax`` fix explicit limits, ``quantiles`` clips extreme means,
    and ``vcenter`` enables diverging maps. The color scale follows
    ``share_y``: ``scope="shared"`` when ``share_y=True`` and ``scope="panel"``
    otherwise, unless an explicit ``color_scale.scope`` overrides it. A
    ``scope="shared"`` scale is drawn as a colorbar on the right;
    ``scope="panel"`` rescales each row independently and draws a single
    reference colorbar. On caller-supplied axes the colorbar is not drawn;
    its limits are exposed through ``PlotResult.legends``.
    ``color_by="mean"`` cannot be combined with ``split_by``.
    """
    plt, mpl = require_matplotlib()
    normalization = normalization or NormalizationSpec()
    color_scale_was_explicit = color_scale is not None
    color_scale = color_scale or ColorScale(cmap="viridis")
    if color_scale_was_explicit and color_by != "mean":
        raise ValueError("color_scale applies only when color_by='mean'")
    if color_by not in ("group", "mean"):
        raise ValueError("color_by must be 'group' or 'mean'")
    if color_by == "mean" and kind != "stacked_violin":
        raise ValueError("color_by='mean' is available only for stacked_violin")
    if color_by == "mean" and group_by is None:
        raise ValueError("color_by='mean' requires group_by")
    resolved_max_points = (
        0
        if max_points is None and kind == "stacked_violin"
        else 10000
        if max_points is None
        else max_points
    )
    if color_by == "mean":
        if not color_scale_was_explicit:
            color_scale = ColorScale(
                cmap="viridis",
                scope="shared" if share_y is True else "panel",
            )
        if color_scale.scale != "linear":
            raise NotImplementedError(
                "distribution mean coloring currently supports only linear color scales"
            )
        if color_scale.scope not in ("shared", "panel"):
            raise ValueError(
                "color_scale.scope must be 'shared' or 'panel' for mean coloring; "
                f"got {color_scale.scope!r}"
            )
        if color_scale.scope == "panel" and (
            color_scale.quantiles is not None
            or color_scale.vmin is not None
            or color_scale.vmax is not None
            or color_scale.vcenter is not None
        ):
            raise ValueError(
                "quantiles/vmin/vmax/vcenter apply only to scope='shared' for "
                "mean coloring; panel scope uses strict per-gene 0-to-1 scaling"
            )
    if kind not in ("violin", "stacked_violin", "box", "hist", "ecdf"):
        raise ValueError(
            "kind must be 'violin', 'stacked_violin', 'box', 'hist', or 'ecdf'"
        )
    if orientation not in ("vertical", "horizontal"):
        raise ValueError("orientation must be 'vertical' or 'horizontal'")
    if kind in ("hist", "ecdf") and orientation != "vertical":
        raise ValueError("orientation applies only to violin and box plots")
    if row_standardize and kind != "stacked_violin":
        raise ValueError("row_standardize is available only for stacked_violin")
    if share_y is not None and kind not in ("violin", "stacked_violin", "box"):
        raise ValueError("share_y applies only to violin and box plots")
    if stats_results is not None:
        if group_by is None:
            raise ValueError("stats_results requires group_by")
        if kind not in ("violin", "stacked_violin", "box"):
            raise ValueError(
                "stats annotation applies only to violin, stacked_violin, and box plots"
            )
        if stats_bracket_height is not None and stats_bracket_height <= 0:
            raise ValueError("stats_bracket_height must be positive when provided")
    if violin_linewidth < 0:
        raise ValueError("violin_linewidth must be non-negative")
    if not 0 <= violin_alpha <= 1 or not 0 <= point_alpha <= 1:
        raise ValueError("violin and point alpha values must be between 0 and 1")
    if groups is not None and group_by is None:
        raise ValueError("groups requires group_by")
    if split_by is not None and group_by is None:
        raise ValueError("split_by requires group_by")
    if split_by is not None and kind not in ("violin", "stacked_violin"):
        raise ValueError("split_by is available only for violin plots")
    if split_by is not None and color_by == "mean":
        raise ValueError("color_by='mean' cannot be combined with split_by")
    if split_by == group_by and split_by is not None:
        raise ValueError("split_by and group_by must refer to different columns")
    if sample_stat not in ("mean", "median", "fraction"):
        raise ValueError("sample_stat must be 'mean', 'median', or 'fraction'")
    if study_design is not None:
        if sample_by is not None and sample_by != study_design.sample_by:
            raise ValueError("sample_by conflicts with study_design.sample_by")
        sample_by = study_design.sample_by
    if kind in ("violin", "stacked_violin", "box"):
        sns = require_seaborn()
    else:
        sns = None
    if bins < 1:
        raise ValueError("bins must be >= 1")

    if isinstance(keys, (str, CellField, FeatureRef)):
        key_list: list[str | CellField | FeatureRef] = [keys]
    else:
        key_list = list(keys)
    if not key_list:
        raise ValueError("keys must be non-empty")
    if group_by is not None:
        categorical_scale = resolve_categorical_scale(
            store,
            group_by,
            categorical_scale,
        )
    if split_by is not None:
        split_scale = resolve_categorical_scale(
            store,
            split_by,
            split_scale,
        )
    feature_assays: set[str] = set()
    for key in key_list:
        if not (
            isinstance(key, FeatureRef)
            or (isinstance(key, str) and key not in store.cells.columns)
        ):
            continue
        assay_name = (
            key.assay
            if isinstance(key, FeatureRef) and key.assay is not None
            else from_assay or store._defaultAssay
        )
        feature_assays.add(assay_name)

    series_list = [
        _fetch_series(
            store,
            k,
            cell_key=cell_key,
            from_assay=from_assay,
            normalization=normalization,
        )
        for k in key_list
    ]
    n = len(series_list[0][0])
    if group_by is not None:
        groups_arr = _fetch_cell_column(store, group_by, cell_key)
        if len(groups_arr) != n:
            raise ValueError("group_by length does not match selected cells")
    else:
        groups_arr = np.zeros(n, dtype=int)
    split_arr = (
        _fetch_cell_column(store, split_by, cell_key) if split_by is not None else None
    )
    sample_arr = (
        _fetch_cell_column(store, sample_by, cell_key)
        if sample_by is not None
        else None
    )
    if split_arr is not None and len(split_arr) != n:
        raise ValueError("split_by length does not match selected cells")
    if sample_arr is not None and len(sample_arr) != n:
        raise ValueError("sample_by length does not match selected cells")

    subset_vals = (
        _fetch_cell_column(store, subset_by, cell_key)
        if subset_by is not None
        else None
    )
    selection_mask, group_order = resolve_cell_selection(
        n,
        subset=subset_vals,
        subset_name=subset_by,
        category_values=groups_arr if group_by is not None else None,
        groups=groups,
    )
    dropped_sample_cells = 0
    if sample_arr is not None:
        valid_sample = pd.notna(sample_arr) & (np.asarray(sample_arr, dtype=str) != "")
        dropped_sample_cells = int((selection_mask & ~valid_sample).sum())
        selection_mask &= valid_sample
    dropped_split_cells = 0
    if split_arr is not None:
        valid_split = pd.notna(split_arr) & (np.asarray(split_arr, dtype=str) != "")
        dropped_split_cells = int((selection_mask & ~valid_split).sum())
        selection_mask &= valid_split
    if not selection_mask.any():
        raise ValueError("No cells remain after distribution selections")
    if group_by is None:
        group_order = None
    elif groups is None:
        observed_values = list(pd.unique(groups_arr[selection_mask]))
        if categorical_scale is not None and categorical_scale.order is not None:
            observed = set(observed_values)
            missing = [
                value for value in observed if value not in categorical_scale.order
            ]
            if missing:
                raise ValueError(
                    "categorical_scale.order is missing observed values: "
                    + ", ".join(map(str, missing[:10]))
                )
            group_order = [
                value for value in categorical_scale.order if value in observed
            ]
        else:
            group_order = sort_categories(observed_values)

    series_list = [
        (np.asarray(vals)[selection_mask], label, is_feature)
        for vals, label, is_feature in series_list
    ]
    groups_arr = groups_arr[selection_mask]
    if split_arr is not None:
        split_arr = split_arr[selection_mask]
    if sample_arr is not None:
        sample_arr = sample_arr[selection_mask]
    n = int(selection_mask.sum())
    any_feature = any(is_feature for _, _, is_feature in series_list)

    panel_keys: list[Hashable] = [label for _, label, _ in series_list]
    if len(set(panel_keys)) != len(panel_keys):
        panel_keys = list(range(len(panel_keys)))

    n_groups = 1 if group_by is None else len(group_order or [])
    n_panels = len(panel_keys)
    palette = (
        _group_palette(list(group_order), categorical_scale)
        if group_by is not None and group_order is not None
        else None
    )
    split_order: list[Any] | None = None
    split_palette: dict[Any, str] | None = None
    if split_arr is not None:
        observed_split = list(pd.unique(split_arr))
        if split_scale is not None and split_scale.order is not None:
            missing_split = [
                value for value in observed_split if value not in split_scale.order
            ]
            if missing_split:
                raise ValueError(
                    "split_scale.order is missing observed values: "
                    + ", ".join(map(str, missing_split[:10]))
                )
            split_order = [
                value for value in split_scale.order if value in set(observed_split)
            ]
        else:
            split_order = sort_categories(observed_split)
        if len(split_order) != 2:
            raise ValueError(
                "split_by must contain exactly two observed categories; "
                f"found {len(split_order)}"
            )
        split_palette = _group_palette(split_order, split_scale)
    # Width scales with category count so rotated labels stay readable; wrap
    # to extra rows before exceeding the page width.
    width_cap = (
        MAX_FIGURE_WIDTH_INCHES if max_figure_width is None else float(max_figure_width)
    )
    if width_cap <= 0:
        raise ValueError("max_figure_width must be positive or None")
    panel_width = min(width_cap, max(3.6, 0.55 * max(n_groups, 1) + 1.8))
    if figsize is None and target is None:
        n_columns = (
            1
            if kind == "stacked_violin"
            else max(1, min(n_panels, int(width_cap // panel_width) or 1))
        )
        n_rows = int(np.ceil(n_panels / n_columns))
        row_height = 1.3 if kind == "stacked_violin" else 4.0
        figsize = capped_figsize(
            panel_width * n_columns,
            max(2.4, row_height * n_rows + 0.8),
            max_width=max_figure_width,
        )
    else:
        n_columns = n_panels

    # Build the per-panel display frames once; the group means that feed the
    # mean-expression colour scale are derived only when ``color_by="mean"``.
    panel_display_frames = [
        _panel_display_frame(
            np.asarray(vals),
            groups_arr,
            split_arr=split_arr,
            sample_arr=sample_arr,
            sample_stat=sample_stat,
            expression_cutoff=expression_cutoff,
            row_standardize=row_standardize,
        )
        for vals, _label, _is_feature in series_list
    ]
    panel_group_means: list[pd.Series] | None = None
    mean_limits: list[tuple[float, float]] | None = None
    if color_by == "mean":
        panel_group_means = [
            _panel_group_means(frame) for frame in panel_display_frames
        ]
        mean_limits, _reference_limits = _mean_color_limits(
            panel_group_means,
            color_scale,
        )
    # Limits actually drawn on the colourbar. Degenerate scales (all group
    # means equal) are padded symmetrically so the key is always visible.
    render_limits: tuple[float, float] | None = None
    if mean_limits is not None:
        if color_scale.scope == "shared":
            base = mean_limits[0]
        else:
            # Panel scope colours each row by its own 0-to-1 relative scale, so
            # the single colorbar shows the unit range.
            base = (0.0, 1.0)
        render_limits = _render_color_limits(*base)

    fig, axes, owns = normalize_axes_target(
        target,
        panel_keys=panel_keys,
        figsize=figsize,
        n_columns=n_columns,
    )

    rng = np.random.default_rng(seed)
    any_subsampled = False
    y_limits: list[tuple[float, float]] = []
    panel_tables: list[tuple[str, pd.DataFrame]] = []
    with theme_context(theme):
        for panel_index, ((vals, label, is_feature), panel_key) in enumerate(
            zip(series_list, panel_keys)
        ):
            ax = axes[panel_key]
            df = panel_display_frames[panel_index]
            table = df.rename(
                columns={
                    "raw_value": "value",
                    "value": "display_value",
                }
            )
            panel_tables.append((label, table))
            if sample_arr is None:
                value_axis_label = label
            elif sample_stat == "fraction":
                value_axis_label = f"Sample fraction > {expression_cutoff:g}"
            else:
                value_axis_label = f"Sample {sample_stat} {label}"
            if kind in ("violin", "stacked_violin", "box"):
                assert sns is not None
                panel_palette = palette
                if mean_limits is not None:
                    assert panel_group_means is not None
                    lo, hi = mean_limits[panel_index]
                    panel_palette = _mean_group_palette(
                        panel_group_means[panel_index],
                        list(group_order or []),
                        color_scale=color_scale,
                        lo=lo,
                        hi=hi,
                    )
                subsampled = _draw_violin_or_box(
                    ax,
                    sns,
                    df,
                    kind=("violin" if kind == "stacked_violin" else kind),  # type: ignore[arg-type]
                    color=color,
                    max_points=resolved_max_points,
                    point_size=point_size,
                    rng=rng,
                    order=None if group_by is None else list(group_order or []),
                    palette=panel_palette,
                    orientation=orientation,
                    violin_inner=violin_inner,
                    linewidth=violin_linewidth,
                    fill_alpha=violin_alpha,
                    point_alpha=point_alpha,
                    split_order=split_order,
                    split_palette=split_palette,
                    show_legend=(
                        show_legend
                        and split_order is not None
                        and panel_index == n_panels - 1
                    ),
                )
                any_subsampled = any_subsampled or subsampled
                if split_order is not None and show_legend:
                    legend = ax.get_legend()
                    if legend is not None:
                        handles, labels = ax.get_legend_handles_labels()
                        legend.remove()
                        ax.legend(
                            handles,
                            labels,
                            title=split_by or "split",
                            frameon=False,
                            loc="upper left",
                            bbox_to_anchor=(1.02, 1),
                            borderaxespad=0,
                        )
                if group_by is None:
                    if orientation == "vertical":
                        ax.set_xticks([])
                    else:
                        ax.set_yticks([])
                else:
                    category_axis = "x" if orientation == "vertical" else "y"
                    ax.tick_params(axis=category_axis, labelrotation=45)
                    ticks = (
                        ax.get_xticklabels()
                        if orientation == "vertical"
                        else ax.get_yticklabels()
                    )
                    for tick in ticks:
                        tick.set_ha("right")
            elif kind == "hist":
                _draw_hist(
                    ax,
                    df,
                    color=color,
                    bins=bins,
                    group_by=group_by,
                    order=None if group_by is None else list(group_order or []),
                    palette=palette,
                    show_legend=show_legend,
                )
            else:
                subsampled = _draw_ecdf(
                    ax,
                    df,
                    color=color,
                    max_points=resolved_max_points,
                    rng=rng,
                    group_by=group_by,
                    order=None if group_by is None else list(group_order or []),
                    palette=palette,
                    show_legend=show_legend,
                )
                any_subsampled = any_subsampled or subsampled
                ax.set_ylabel("ECDF")

            ax.set_title(
                "" if kind == "stacked_violin" and orientation == "vertical" else label,
                fontstyle=("italic" if italicize_features and is_feature else "normal"),
                loc=("left" if kind == "stacked_violin" else "center"),
            )
            if kind in ("hist", "ecdf"):
                ax.set_xlabel(value_axis_label)
                if kind == "hist":
                    ax.set_ylabel("count")
            else:
                # Panel titles already name the key, so the value axis only
                # repeats it when the title is suppressed or the values were
                # aggregated per sample.
                keeps_key_on_axis = kind == "stacked_violin" or sample_arr is not None
                measurement_label = (
                    "standardized value"
                    if row_standardize
                    else value_axis_label
                    if keeps_key_on_axis
                    else "value"
                )
                if orientation == "vertical":
                    ax.set_xlabel(group_by or "")
                    ax.set_ylabel(
                        measurement_label,
                        fontstyle=(
                            "italic"
                            if italicize_features
                            and is_feature
                            and not row_standardize
                            and keeps_key_on_axis
                            else "normal"
                        ),
                    )
                else:
                    ax.set_xlabel(
                        measurement_label,
                        fontstyle=(
                            "italic"
                            if italicize_features
                            and is_feature
                            and not row_standardize
                            and keeps_key_on_axis
                            else "normal"
                        ),
                    )
                    ax.set_ylabel(group_by or "")
            if kind == "stacked_violin" and panel_index < n_panels - 1:
                if orientation == "vertical":
                    ax.tick_params(axis="x", labelbottom=False)
                    ax.set_xlabel("")
                else:
                    ax.tick_params(axis="y", labelleft=False)
                    ax.set_ylabel("")
            finite = df["value"].to_numpy(dtype=np.float64)
            finite = finite[np.isfinite(finite)]
            if len(finite):
                y_limits.append((float(finite.min()), float(finite.max())))

        resolved_share_y = (
            any_feature and kind in ("violin", "box") if share_y is None else share_y
        )
        if resolved_share_y and len(y_limits) > 1:
            ymin = min(lo for lo, _ in y_limits)
            ymax = max(hi for _, hi in y_limits)
            if ymax > ymin:
                pad = 0.05 * (ymax - ymin)
                for ax in axes.values():
                    if orientation == "vertical":
                        ax.set_ylim(ymin - pad, ymax + pad)
                    else:
                        ax.set_xlim(ymin - pad, ymax + pad)

        stats_annotated_any = False
        stats_annotated_keys: list[str] = []
        stats_methods: set[str] = set()
        stats_adjustments: set[str] = set()
        if stats_results is not None:
            display_order = list(group_order or [])
            allowed_stats_keys = (
                None if stats_keys is None else {str(value) for value in stats_keys}
            )
            warned_validation: set[str] = set()

            def _warn_once(reason_key: str, message: str) -> None:
                if reason_key not in warned_validation:
                    warnings.warn(message, UserWarning, stacklevel=2)
                    warned_validation.add(reason_key)

            for index, (_vals, label, _is_feature) in enumerate(series_list):
                str_label = str(panel_keys[index])
                if allowed_stats_keys is not None and str_label not in (
                    allowed_stats_keys
                ):
                    continue
                result_for_panel = (
                    stats_results.get(str_label)
                    if isinstance(stats_results, Mapping)
                    else stats_results
                )
                if result_for_panel is None:
                    continue
                if result_for_panel.group_key != group_by:
                    _warn_once(
                        "group_by",
                        "stats_results.group_key does not match group_by; "
                        "skipping statistical annotations",
                    )
                    continue
                if result_for_panel.cell_key != cell_key:
                    _warn_once(
                        "cell_key",
                        "stats_results.cell_key does not match cell_key; "
                        "skipping statistical annotations",
                    )
                    continue
                if result_for_panel.n_cells != n:
                    _warn_once(
                        "n_cells",
                        f"stats_results was computed on {result_for_panel.n_cells} "
                        f"cells but the plot shows {n}; skipping statistical "
                        "annotations",
                    )
                    continue
                table_to_annotate = result_for_panel.tables.get(str_label)
                if table_to_annotate is None:
                    continue
                ax_panel = axes[panel_keys[index]]
                value_lo, value_hi = (
                    ax_panel.get_ylim()
                    if orientation == "vertical"
                    else ax_panel.get_xlim()
                )
                span = float(value_hi - value_lo)
                resolved_height = (
                    stats_bracket_height
                    if stats_bracket_height is not None
                    else max(0.06 * span, 1e-12)
                )
                if _annotate_distribution_stats(
                    ax_panel,
                    table_to_annotate,
                    method=result_for_panel.method,
                    group_order=display_order,
                    orientation=orientation,
                    bracket_height=float(resolved_height),
                    show_p_value=stats_show_p,
                ):
                    stats_annotated_any = True
                    stats_annotated_keys.append(str_label)
                    stats_methods.add(result_for_panel.method)
                    stats_adjustments.add(result_for_panel.adjustment_method)

        if title is not None:
            fig.suptitle(title)
        apply_figure_chrome(fig, theme)
        # The colorbar is only drawn on figures Scarf owns; caller-supplied
        # axes keep their layout and the limits are exposed through
        # ``PlotResult.legends`` / ``scales`` instead.
        if render_limits is not None and owns:
            lo, hi = render_limits
            mappable = plt.cm.ScalarMappable(
                cmap=color_scale.cmap or "viridis",
                norm=continuous_norm(
                    mpl,
                    vmin=lo,
                    vmax=hi,
                    vcenter=color_scale.vcenter,
                ),
            )
            mappable.set_array([])
            colorbar = fig.colorbar(
                mappable,
                ax=list(axes.values()),
                location="right",
                shrink=0.8,
                fraction=0.04,
                pad=0.02,
            )
            colorbar.set_label(_mean_colorbar_label(color_scale, row_standardize))

    label_counts = pd.Series([label for _, label, _ in series_list]).value_counts()
    tables = {}
    for index, (label, table) in enumerate(panel_tables):
        table_name = label if label_counts[label] == 1 else f"{index}:{label}"
        tables[table_name] = table
    notes = ["distribution", kind]
    if any_subsampled:
        notes.append("subsampled_display")

    resolved_color_scale: ColorScale | None = None
    if render_limits is not None:
        lo, hi = render_limits
        resolved_color_scale = ColorScale(
            cmap=color_scale.cmap,
            vmin=lo,
            vmax=hi,
            vcenter=color_scale.vcenter,
            quantiles=color_scale.quantiles,
            missing_color=color_scale.missing_color,
            scope=color_scale.scope,
            scale=color_scale.scale,
        )

    if resolved_color_scale is not None:
        assert render_limits is not None
        lo, hi = render_limits
        legend_specs: tuple[LegendSpec, ...] = (
            LegendSpec(
                kind="colorbar",
                label=_mean_colorbar_label(color_scale, row_standardize),
                extras={"vmin": lo, "vmax": hi},
            ),
        )
        scale_specs: tuple[Any, ...] = (resolved_color_scale,)
    elif split_order is not None:
        legend_specs = (LegendSpec(kind="categorical", label=split_by or "split"),)
        scale_specs = (
            CategoricalScale(
                order=tuple(split_order),
                palette=split_palette,
                labels=(split_scale.labels if split_scale is not None else None),
                missing_color=(
                    split_scale.missing_color if split_scale is not None else "#bdbdbd"
                ),
                missing_label=(
                    split_scale.missing_label if split_scale is not None else "NA"
                ),
                palette_name=(
                    split_scale.palette_name if split_scale is not None else "default"
                ),
            ),
        )
    else:
        legend_specs = (LegendSpec(kind="distribution", label=kind),)
        scale_specs = () if categorical_scale is None else (categorical_scale,)

    result = PlotResult(
        figure=fig,
        axes=axes,
        tables=tables,
        legends=legend_specs,
        scales=scale_specs,
        provenance=PlotProvenance(
            assay=(next(iter(feature_assays)) if len(feature_assays) == 1 else None),
            cell_key=cell_key,
            n_cells=n,
            n_samples=(
                int(pd.Series(sample_arr).nunique()) if sample_arr is not None else None
            ),
            renderer="matplotlib",
            notes=tuple(notes),
            extras={
                "max_points": resolved_max_points,
                "seed": seed,
                "group_by": group_by,
                "groups": None if groups is None else list(groups),
                "split_by": split_by,
                "split_order": split_order,
                "sample_by": sample_by,
                "sample_stat": sample_stat if sample_by is not None else None,
                "expression_cutoff": (
                    expression_cutoff
                    if sample_by is not None and sample_stat == "fraction"
                    else None
                ),
                "dropped_sample_cells": dropped_sample_cells,
                "dropped_split_cells": dropped_split_cells,
                "subset_by": subset_by,
                "bins": bins if kind == "hist" else None,
                "title": title,
                "orientation": orientation,
                "row_standardize": row_standardize,
                "share_y": resolved_share_y,
                "color_by": color_by,
                "color_scale_scope": (
                    color_scale.scope if resolved_color_scale is not None else None
                ),
                "cmap": (
                    resolved_color_scale.cmap
                    if resolved_color_scale is not None
                    else None
                ),
                "vmin": render_limits[0] if render_limits is not None else None,
                "vmax": render_limits[1] if render_limits is not None else None,
                "violin_inner": violin_inner,
                "italicize_features": italicize_features,
                "approximate": any_subsampled,
                "normalization": {
                    "source": normalization.source,
                    "transform": normalization.transform,
                },
                "assays": sorted(feature_assays),
                **(
                    {
                        "stats_method": sorted(stats_methods),
                        "stats_adjustment": sorted(stats_adjustments),
                        "stats_annotated": stats_annotated_any,
                        "stats_annotated_keys": stats_annotated_keys,
                    }
                    if stats_results is not None
                    else {}
                ),
            },
        ),
        owns_figure=owns,
        theme=theme,
    )
    if show:
        result.show()
    return result
