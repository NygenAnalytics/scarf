"""Distribution plots for metadata and feature values."""

from collections.abc import Sequence
from typing import Any, Hashable, Literal

import numpy as np
import pandas as pd

from ._contracts import (
    CategoricalScale,
    CellField,
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
from ._figure import LegendSpec, PlotResult, normalize_axes_target
from ._style import (
    MAX_FIGURE_WIDTH_INCHES,
    apply_figure_chrome,
    capped_figsize,
    categorical_color_map,
    theme_context,
)

DistKind = Literal["violin", "box", "hist", "ecdf"]


def _scarf_version() -> str:
    try:
        from importlib.metadata import version

        return version("scarf")
    except Exception:
        return "unknown"


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
        missing_label=None,
    )


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
) -> bool:
    plot_kw: dict[str, Any] = {}
    if order is not None:
        plot_kw["order"] = order
    grouped = order is not None and palette is not None and len(order) > 0
    if grouped:
        plot_kw["hue"] = "group"
        plot_kw["hue_order"] = order
        plot_kw["palette"] = palette
        plot_kw["legend"] = False
        plot_kw["dodge"] = False

    if kind == "violin":
        sns.violinplot(
            data=df,
            x="group",
            y="value",
            ax=ax,
            color=None if grouped else color,
            inner="quartile",
            cut=0,
            linewidth=0.8,
            saturation=0.9,
            **plot_kw,
        )
    else:
        sns.boxplot(
            data=df,
            x="group",
            y="value",
            ax=ax,
            color=None if grouped else color,
            showfliers=max_points <= 0,
            linewidth=0.8,
            fliersize=2,
            **plot_kw,
        )

    subsampled = False
    if max_points > 0 and len(df) > 0:
        pts, subsampled = _subsample_frame(df, max_points=max_points, rng=rng)
        strip_kw = {k: v for k, v in plot_kw.items() if k != "legend"}
        if "palette" in strip_kw:
            strip_kw.pop("palette", None)
            strip_kw.pop("hue", None)
        sns.stripplot(
            data=pts,
            x="group",
            y="value",
            ax=ax,
            color="0.15",
            size=point_size,
            jitter=0.28,
            alpha=0.28,
            **{k: v for k, v in strip_kw.items() if k == "order"},
        )
    return subsampled


def _draw_hist(
    ax: Any,
    df: pd.DataFrame,
    *,
    color: str,
    bins: int,
    group_by: str | None,
    order: list[Any] | None,
    palette: dict[Any, str] | None,
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
    ax.legend(frameon=False, fontsize=8)
    ax.set_ylim(-0.02, 1.02)
    return subsampled


def distribution(
    store: Any,
    keys: str | CellField | FeatureRef | Sequence[str | CellField | FeatureRef],
    *,
    group_by: str | None = None,
    groups: Sequence[Any] | None = None,
    subset_by: str | None = None,
    cell_key: str | None = "I",
    from_assay: str | None = None,
    normalization: NormalizationSpec | None = None,
    categorical_scale: CategoricalScale | None = None,
    kind: DistKind = "violin",
    bins: int = 40,
    max_points: int = 10000,
    point_size: float = 0.8,
    seed: int = 0,
    color: str = "steelblue",
    target: Any | None = None,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
    theme: str = "notebook",
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
    ``max_points`` limits how many points are drawn (``0`` disables points).
    Histograms always use every selected cell. When several gene keys are
    passed, panels share a y-axis scale. Several keys wrap into a grid instead
    of growing into a very wide figure.
    """
    require_matplotlib()
    normalization = normalization or NormalizationSpec()
    if kind not in ("violin", "box", "hist", "ecdf"):
        raise ValueError("kind must be 'violin', 'box', 'hist', or 'ecdf'")
    if groups is not None and group_by is None:
        raise ValueError("groups requires group_by")
    if kind in ("violin", "box"):
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
    if group_by is None:
        group_order = None

    series_list = [
        (np.asarray(vals)[selection_mask], label, is_feature)
        for vals, label, is_feature in series_list
    ]
    groups_arr = groups_arr[selection_mask]
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
    # Width scales with category count so rotated labels stay readable; wrap
    # to extra rows before exceeding the page width.
    panel_width = min(MAX_FIGURE_WIDTH_INCHES, max(3.6, 0.55 * max(n_groups, 1) + 1.8))
    if figsize is None and target is None:
        n_columns = max(
            1,
            min(n_panels, int(MAX_FIGURE_WIDTH_INCHES // panel_width) or 1),
        )
        n_rows = int(np.ceil(n_panels / n_columns))
        figsize = capped_figsize(panel_width * n_columns, 4.0 * n_rows)
    else:
        n_columns = n_panels

    fig, axes, owns = normalize_axes_target(
        target,
        panel_keys=panel_keys,
        figsize=figsize,
        n_columns=n_columns,
    )

    rng = np.random.default_rng(seed)
    any_subsampled = False
    y_limits: list[tuple[float, float]] = []
    with theme_context(theme):
        for (vals, label, _), panel_key in zip(series_list, panel_keys):
            ax = axes[panel_key]
            df = pd.DataFrame({"value": vals, "group": groups_arr})
            if kind in ("violin", "box"):
                assert sns is not None
                subsampled = _draw_violin_or_box(
                    ax,
                    sns,
                    df,
                    kind=kind,  # type: ignore[arg-type]
                    color=color,
                    max_points=max_points,
                    point_size=point_size,
                    rng=rng,
                    order=None if group_by is None else list(group_order or []),
                    palette=palette,
                )
                any_subsampled = any_subsampled or subsampled
                if group_by is None:
                    ax.set_xticks([])
                else:
                    ax.tick_params(axis="x", labelrotation=45)
                    for tick in ax.get_xticklabels():
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
                )
            else:
                subsampled = _draw_ecdf(
                    ax,
                    df,
                    color=color,
                    max_points=max_points,
                    rng=rng,
                    group_by=group_by,
                    order=None if group_by is None else list(group_order or []),
                    palette=palette,
                )
                any_subsampled = any_subsampled or subsampled
                ax.set_ylabel("ECDF")

            ax.set_title(label)
            if kind in ("hist", "ecdf"):
                ax.set_xlabel(label)
                if kind == "hist":
                    ax.set_ylabel("count")
            else:
                ax.set_xlabel(group_by or "")
                ax.set_ylabel(label)
            finite = vals[np.isfinite(vals)]
            if len(finite):
                y_limits.append((float(finite.min()), float(finite.max())))

        if any_feature and kind in ("violin", "box") and len(y_limits) > 1:
            ymin = min(lo for lo, _ in y_limits)
            ymax = max(hi for _, hi in y_limits)
            if ymax > ymin:
                pad = 0.05 * (ymax - ymin)
                for ax in axes.values():
                    ax.set_ylim(ymin - pad, ymax + pad)

        if title is not None:
            fig.suptitle(title)
        apply_figure_chrome(fig, theme)

    label_counts = pd.Series([label for _, label, _ in series_list]).value_counts()
    tables = {}
    for index, (vals, label, _) in enumerate(series_list):
        table_name = label if label_counts[label] == 1 else f"{index}:{label}"
        tables[table_name] = pd.DataFrame({"value": vals, "group": groups_arr})
    notes = ["distribution", kind]
    if any_subsampled:
        notes.append("subsampled_display")

    result = PlotResult(
        figure=fig,
        axes=axes,
        tables=tables,
        legends=(LegendSpec(kind="distribution", label=kind),),
        scales=(() if categorical_scale is None else (categorical_scale,)),
        provenance=PlotProvenance(
            scarf_version=_scarf_version(),
            assay=(next(iter(feature_assays)) if len(feature_assays) == 1 else None),
            cell_key=cell_key,
            n_cells=n,
            renderer="matplotlib",
            notes=tuple(notes),
            extras={
                "max_points": max_points,
                "seed": seed,
                "group_by": group_by,
                "groups": None if groups is None else list(groups),
                "subset_by": subset_by,
                "bins": bins if kind == "hist" else None,
                "title": title,
                "approximate": any_subsampled,
                "normalization": {
                    "source": normalization.source,
                    "transform": normalization.transform,
                },
                "assays": sorted(feature_assays),
            },
        ),
        owns_figure=owns,
        theme=theme,
    )
    if show:
        result.show()
    return result
