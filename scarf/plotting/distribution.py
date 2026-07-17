"""Distribution plots for metadata and feature values."""

from collections.abc import Sequence
from typing import Any, Hashable, Literal

import numpy as np
import pandas as pd

from ._contracts import CellField, FeatureRef, NormalizationSpec, PlotProvenance
from ._data import fetch_normalized_feature_matrix, resolve_feature
from ._deps import require_matplotlib, require_seaborn
from ._figure import LegendSpec, PlotResult, normalize_axes_target
from ._style import (
    MAX_FIGURE_WIDTH_INCHES,
    apply_figure_chrome,
    capped_figsize,
    sort_categories,
    theme_context,
)

DistKind = Literal["violin", "box", "hist", "ecdf"]


def _scarf_version() -> str:
    try:
        from importlib.metadata import version

        return version("scarf")
    except Exception:
        return "unknown"


def _fetch_series(
    store: Any,
    key: str | CellField | FeatureRef,
    *,
    cell_key: str,
    from_assay: str | None,
    normalization: NormalizationSpec,
) -> tuple[np.ndarray, str]:
    if isinstance(key, CellField):
        return np.asarray(
            store.cells.fetch(key.key, key=cell_key)
        ), key.label or key.key
    if isinstance(key, FeatureRef) or (
        isinstance(key, str) and key not in store.cells.columns
    ):
        resolved = resolve_feature(store, key, from_assay=from_assay)
        cell_idx = store.cells.active_index(cell_key)
        mat = fetch_normalized_feature_matrix(
            store,
            [resolved],
            cell_idx,
            normalization=normalization,
        )
        return mat[:, 0], resolved.label
    return np.asarray(store.cells.fetch(str(key), key=cell_key)), str(key)


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
) -> bool:
    plot_kw: dict[str, Any] = {}
    if order is not None:
        plot_kw["order"] = order
    if kind == "violin":
        sns.violinplot(
            data=df,
            x="group",
            y="value",
            ax=ax,
            color=color,
            inner=None,
            cut=0,
            linewidth=1,
            **plot_kw,
        )
    else:
        sns.boxplot(data=df, x="group", y="value", ax=ax, color=color, **plot_kw)

    subsampled = False
    if max_points > 0 and len(df) > 0:
        pts, subsampled = _subsample_frame(df, max_points=max_points, rng=rng)
        sns.stripplot(
            data=pts,
            x="group",
            y="value",
            ax=ax,
            color="k",
            size=point_size,
            jitter=0.35,
            alpha=0.4,
            **plot_kw,
        )
    return subsampled


def _draw_hist(
    ax: Any,
    df: pd.DataFrame,
    *,
    color: str,
    bins: int,
    group_by: str | None,
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
    for g, sub in df.groupby("group", observed=False, sort=True):
        vals = sub["value"].to_numpy(dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        ax.hist(
            vals,
            bins=shared_bins,
            alpha=0.45,
            label=str(g),
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

    for g, sub in df.groupby("group", observed=False, sort=True):
        vals = sub["value"].to_numpy(dtype=np.float64)
        if max_points > 0 and len(vals) > max_points:
            vals = vals[rng.choice(len(vals), size=max_points, replace=False)]
            subsampled = True
        x, y = _ecdf_xy(vals)
        if len(x):
            ax.step(x, y, where="post", linewidth=1.2, label=str(g))
    ax.legend(frameon=False, fontsize=8)
    ax.set_ylim(-0.02, 1.02)
    return subsampled


def distribution(
    store: Any,
    keys: str | CellField | FeatureRef | Sequence[str | CellField | FeatureRef],
    *,
    group_by: str | None = None,
    cell_key: str = "I",
    from_assay: str | None = None,
    normalization: NormalizationSpec | None = None,
    kind: DistKind = "violin",
    bins: int = 40,
    max_points: int = 10000,
    point_size: float = 1.0,
    seed: int = 0,
    color: str = "steelblue",
    target: Any | None = None,
    figsize: tuple[float, float] | None = None,
    theme: str = "notebook",
    show: bool = False,
) -> PlotResult:
    """Compare value distributions for QC metrics or genes.

    ``keys`` may be cell-metadata columns (for example ``RNA_nCounts``) or gene
    names. ``kind`` selects the display: ``"violin"``, ``"box"``, ``"hist"``,
    or ``"ecdf"``. With ``group_by``, each category gets its own distribution
    along the x-axis.

    For violins and boxes, Scarf can overlay a subsample of cells as points.
    ``max_points`` limits how many points are drawn so large datasets stay
    responsive; the subsample is deterministic when you set ``seed``.
    Histograms always use every selected cell. When several keys are passed,
    panels wrap into a grid instead of growing into a very wide figure.
    """
    require_matplotlib()
    normalization = normalization or NormalizationSpec()
    if kind not in ("violin", "box", "hist", "ecdf"):
        raise ValueError("kind must be 'violin', 'box', 'hist', or 'ecdf'")
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
        groups = np.asarray(store.cells.fetch(group_by, key=cell_key))
        if len(groups) != n:
            raise ValueError("group_by length does not match selected cells")
    else:
        groups = np.zeros(n, dtype=int)

    panel_keys: list[Hashable] = [label for _, label in series_list]
    if len(set(panel_keys)) != len(panel_keys):
        panel_keys = list(range(len(panel_keys)))

    n_groups = int(pd.Series(groups).nunique())
    n_panels = len(panel_keys)
    group_order = sort_categories(list(pd.unique(groups)))
    # Width scales with category count so rotated labels stay readable; wrap
    # to extra rows before exceeding the page width.
    panel_width = min(MAX_FIGURE_WIDTH_INCHES, max(3.6, 0.55 * n_groups + 1.8))
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
    with theme_context(theme):
        for (vals, label), panel_key in zip(series_list, panel_keys):
            ax = axes[panel_key]
            df = pd.DataFrame({"value": vals, "group": groups})
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
                    order=None if group_by is None else group_order,
                )
                any_subsampled = any_subsampled or subsampled
                if group_by is None:
                    ax.set_xticks([])
                else:
                    ax.tick_params(axis="x", labelrotation=45)
                    for tick in ax.get_xticklabels():
                        tick.set_ha("right")
            elif kind == "hist":
                _draw_hist(ax, df, color=color, bins=bins, group_by=group_by)
            else:
                subsampled = _draw_ecdf(
                    ax,
                    df,
                    color=color,
                    max_points=max_points,
                    rng=rng,
                    group_by=group_by,
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
        apply_figure_chrome(fig, theme)

    label_counts = pd.Series([label for _, label in series_list]).value_counts()
    tables = {}
    for index, (vals, label) in enumerate(series_list):
        table_name = label if label_counts[label] == 1 else f"{index}:{label}"
        tables[table_name] = pd.DataFrame({"value": vals, "group": groups})
    notes = ["distribution", kind]
    if any_subsampled:
        notes.append("subsampled_display")

    result = PlotResult(
        figure=fig,
        axes=axes,
        tables=tables,
        legends=(LegendSpec(kind="distribution", label=kind),),
        scales=(),
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
                "bins": bins if kind == "hist" else None,
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
