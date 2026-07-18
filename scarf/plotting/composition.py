"""Sample-level composition figures, including paired subject lines."""

from typing import Any, Hashable, Literal

import numpy as np
import pandas as pd

from ._contracts import CategoricalScale, PlotProvenance, StudyDesign
from ._deps import require_matplotlib
from ._figure import LegendSpec, PlotResult, normalize_axes_target
from ._style import (
    apply_figure_chrome,
    capped_figsize,
    categorical_color_map,
    scatter_edgecolor,
    sort_categories,
    theme_context,
)


def _constant_within_sample(
    samples: np.ndarray,
    values: np.ndarray,
    *,
    field_name: str,
) -> pd.Series:
    """Return one value per sample; raise if a sample has multiple values."""
    df = pd.DataFrame({"sample": samples, "value": values})
    nunique = df.groupby("sample", observed=False)["value"].nunique(dropna=False)
    bad = nunique[nunique > 1]
    if len(bad):
        raise ValueError(
            f"{field_name} is not constant within sample(s): "
            + ", ".join(map(str, list(bad.index[:10])))
        )
    return df.groupby("sample", observed=False)["value"].first()


def _attach_sample_meta(
    per_sample: pd.DataFrame,
    *,
    samples: np.ndarray,
    subject_vals: np.ndarray | None,
    pair_vals: np.ndarray | None,
    condition_vals: np.ndarray | None,
) -> pd.DataFrame:
    meta_parts: list[pd.Series] = []
    if subject_vals is not None:
        meta_parts.append(
            _constant_within_sample(
                samples, subject_vals, field_name="subject_by"
            ).rename("subject")
        )
    if pair_vals is not None:
        meta_parts.append(
            _constant_within_sample(samples, pair_vals, field_name="pair_by").rename(
                "pair"
            )
        )
    if condition_vals is not None:
        meta_parts.append(
            _constant_within_sample(
                samples, condition_vals, field_name="condition_by"
            ).rename("condition")
        )
    if not meta_parts:
        return per_sample
    meta = pd.concat(meta_parts, axis=1).reset_index()
    return per_sample.merge(meta, on="sample", how="left")


def _sort_conditions(values: list[Any]) -> list[Any]:
    """Order conditions for paired plots with a light before/after preference."""
    preferred = (
        "before",
        "pre",
        "baseline",
        "control",
        "ctrl",
        "untreated",
        "after",
        "post",
        "treated",
        "stimulated",
        "stim",
    )

    def key(value: Any) -> tuple[int, str]:
        text = str(value).lower()
        try:
            return preferred.index(text), text
        except ValueError:
            return len(preferred), text

    return sorted(values, key=key)


def _draw_pair_lines(
    ax: Any,
    per_sample: pd.DataFrame,
    cat_order: list[Any],
    condition_order: list[Any],
    pair_col: str,
) -> int:
    """Draw one line per category and pair across conditions."""
    n_lines = 0
    positions = {
        (cat, condition): float(index)
        for index, (cat, condition) in enumerate(
            (cat, condition) for cat in cat_order for condition in condition_order
        )
    }
    for cat in cat_order:
        category_rows = per_sample[per_sample["category"] == cat]
        valid_pair = pd.notna(category_rows[pair_col]) & (
            category_rows[pair_col].astype(str) != ""
        )
        category_rows = category_rows[valid_pair]
        for _, group in category_rows.groupby(
            pair_col,
            observed=False,
            dropna=True,
        ):
            xs: list[float] = []
            ys: list[float] = []
            for condition in condition_order:
                rows = group[group["condition"] == condition]
                values = rows["proportion"].to_numpy(dtype=np.float64)
                finite = values[np.isfinite(values)]
                if len(finite) == 0:
                    continue
                xs.append(positions[(cat, condition)])
                ys.append(float(np.mean(finite)))
            if len(xs) >= 2:
                ax.plot(
                    xs,
                    ys,
                    color="#757575",
                    alpha=0.45,
                    linewidth=0.9,
                    zorder=1,
                    solid_capstyle="round",
                )
                n_lines += 1
    return n_lines


def composition(
    store: Any,
    *,
    category_by: str,
    cell_key: str = "I",
    sample_by: str | None = None,
    subject_by: str | None = None,
    pair_by: str | None = None,
    condition_by: str | None = None,
    study_design: StudyDesign | None = None,
    kind: Literal["stacked", "per_sample"] = "stacked",
    categorical_scale: CategoricalScale | None = None,
    target: Any | None = None,
    figsize: tuple[float, float] | None = None,
    theme: str = "notebook",
    show: bool = True,
) -> PlotResult:
    """Show how cell categories (clusters, cell types) vary across samples.

    ``kind="stacked"`` draws stacked bars. With ``sample_by``, there is one bar
    per sample; without it, one bar for the whole dataset.
    ``kind="per_sample"`` draws one point per sample in each category (requires
    ``sample_by`` or a :class:`StudyDesign`).

    For paired before/after designs, pass ``condition_by`` together with
    ``subject_by`` or ``pair_by`` (directly or via ``study_design``). Scarf
    then connects the same subject across conditions inside each category.
    When a subject has several samples in one condition, those proportions are
    averaged for the connecting line.

    Figure width is capped so large category lists stay page-sized.
    """
    _, mpl = require_matplotlib()
    if kind not in ("stacked", "per_sample"):
        raise ValueError("kind must be 'stacked' or 'per_sample'")
    if study_design is not None:
        sample_by = study_design.sample_by
        subject_by = study_design.subject_by or subject_by
        pair_by = study_design.pair_by or pair_by
        condition_by = study_design.condition_by or condition_by
    if (subject_by is not None or pair_by is not None) and condition_by is None:
        raise ValueError(
            "Paired composition requires condition_by together with "
            "subject_by or pair_by"
        )

    cats = np.asarray(
        store.cells.fetch(category_by, key=cell_key),
        dtype=object,
    ).copy()
    if len(cats) == 0:
        raise ValueError(f"No cells selected by cell_key {cell_key!r}")
    missing_label = (
        categorical_scale.missing_label if categorical_scale is not None else "NA"
    )
    cats[pd.isna(cats)] = missing_label
    observed_categories = list(pd.unique(cats))
    if categorical_scale and categorical_scale.order is not None:
        cat_order = list(categorical_scale.order)
        unlisted = [
            category for category in observed_categories if category not in cat_order
        ]
        if unlisted:
            raise ValueError(
                "categorical_scale.order is missing observed values: "
                + ", ".join(map(str, unlisted[:10]))
            )
    else:
        cat_order = sort_categories(observed_categories)

    pair_col: str | None = None
    n_pair_lines = 0
    n_unpaired_samples = 0
    dropped_sample_cells = 0

    if sample_by is None:
        if kind == "per_sample":
            raise ValueError("kind='per_sample' requires sample_by or StudyDesign")
        if subject_by is not None or pair_by is not None:
            raise ValueError("subject_by / pair_by require sample_by or StudyDesign")
        counts = pd.Series(cats).value_counts()
        props = counts.reindex(cat_order).fillna(0) / max(counts.sum(), 1)
        aggregate = pd.DataFrame(
            {
                "category": cat_order,
                "proportion": [props.get(c, 0.0) for c in cat_order],
            }
        )
        per_sample = None
        props_mat = None
    else:
        samples = store.cells.fetch(sample_by, key=cell_key)
        valid = pd.notna(samples) & (np.asarray(samples, dtype=object) != "")
        if not valid.any():
            raise ValueError(f"No cells have valid values for sample_by {sample_by!r}")
        dropped_sample_cells = int((~valid).sum())
        sample_vals_valid = np.asarray(samples)[valid]
        df = pd.DataFrame(
            {
                "sample": sample_vals_valid,
                "category": np.asarray(cats)[valid],
            }
        )
        ct = pd.crosstab(df["sample"], df["category"])
        ct = ct.reindex(columns=cat_order, fill_value=0)
        props_mat = ct.div(ct.sum(axis=1).replace(0, np.nan), axis=0)
        cell_counts = ct.stack()
        cell_counts.index = cell_counts.index.set_names(["sample", "category"])
        cell_counts = cell_counts.rename("n_cells").reset_index()
        per_sample = props_mat.reset_index().melt(
            id_vars="sample", var_name="category", value_name="proportion"
        )
        per_sample = per_sample.merge(
            cell_counts, on=["sample", "category"], how="left"
        )

        subject_vals = (
            np.asarray(store.cells.fetch(subject_by, key=cell_key))[valid]
            if subject_by is not None
            else None
        )
        pair_vals = (
            np.asarray(store.cells.fetch(pair_by, key=cell_key))[valid]
            if pair_by is not None
            else None
        )
        condition_vals = (
            np.asarray(store.cells.fetch(condition_by, key=cell_key))[valid]
            if condition_by is not None
            else None
        )
        per_sample = _attach_sample_meta(
            per_sample,
            samples=sample_vals_valid,
            subject_vals=subject_vals,
            pair_vals=pair_vals,
            condition_vals=condition_vals,
        )
        if pair_by is not None:
            pair_col = "pair"
        elif subject_by is not None:
            pair_col = "subject"
        if pair_col is not None:
            sample_pairs = per_sample[["sample", pair_col]].drop_duplicates()
            missing_pair = pd.isna(sample_pairs[pair_col]) | (
                sample_pairs[pair_col].astype(str) == ""
            )
            n_unpaired_samples = int(missing_pair.sum())

        aggregate = (
            per_sample.groupby("category", observed=False)["proportion"]
            .mean()
            .reindex(cat_order)
            .reset_index()
        )

    palette = categorical_color_map(
        cat_order,
        palette=categorical_scale.palette if categorical_scale else None,
    )

    panel_key: Hashable = "composition"
    resolved_figsize = figsize
    if resolved_figsize is None and target is None:
        if kind == "per_sample" and pair_col is not None:
            n_conditions = max(
                2,
                int(per_sample["condition"].nunique()) if per_sample is not None else 2,
            )
            resolved_figsize = capped_figsize(
                max(5.5, 0.42 * len(cat_order) * n_conditions + 2.0),
                3.8,
            )
        elif kind == "stacked" and sample_by is not None and per_sample is not None:
            n_samples = int(per_sample["sample"].nunique())
            resolved_figsize = capped_figsize(
                max(4.5, 0.28 * n_samples + 1.8),
                3.6,
            )
        else:
            resolved_figsize = capped_figsize(
                max(4.5, 0.35 * len(cat_order) + 1.8),
                3.6,
            )
    edgecolor = scatter_edgecolor(theme)

    with theme_context(theme):
        fig, axes, owns = normalize_axes_target(
            target,
            panel_keys=[panel_key],
            figsize=resolved_figsize,
        )
        ax = axes[panel_key]
        if kind == "stacked":
            if sample_by is None:
                bottom = 0.0
                for cat in cat_order:
                    val = float(
                        aggregate.loc[aggregate["category"] == cat, "proportion"].iloc[
                            0
                        ]
                    )
                    ax.bar(
                        0,
                        val,
                        bottom=bottom,
                        color=palette[cat],
                        label=str(cat),
                        width=0.6,
                    )
                    bottom += val
                ax.set_xticks([])
                ax.set_ylabel("proportion")
            else:
                assert per_sample is not None and props_mat is not None
                samples_order = list(props_mat.index)
                bottoms = np.zeros(len(samples_order))
                x = np.arange(len(samples_order))
                for cat in cat_order:
                    heights = props_mat[cat].to_numpy(dtype=np.float64)
                    heights = np.nan_to_num(heights, nan=0.0)
                    ax.bar(
                        x,
                        heights,
                        bottom=bottoms,
                        color=palette[cat],
                        label=str(cat),
                        width=0.8,
                    )
                    bottoms += heights
                ax.set_xticks(x)
                ax.set_xticklabels(
                    [str(s) for s in samples_order], rotation=45, ha="right"
                )
                ax.set_ylabel("proportion")
            fig.legend(
                frameon=False,
                loc="outside right center",
                title=category_by,
            )
        else:
            assert per_sample is not None
            if pair_col is not None:
                valid_conditions = per_sample["condition"].dropna()
                valid_conditions = valid_conditions[valid_conditions.astype(str) != ""]
                condition_order = _sort_conditions(list(valid_conditions.unique()))
                if len(condition_order) < 2:
                    raise ValueError(
                        "Paired composition requires at least two condition values"
                    )
                n_pair_lines = _draw_pair_lines(
                    ax,
                    per_sample,
                    cat_order,
                    condition_order,
                    pair_col,
                )
                plot_groups = [
                    (category, condition)
                    for category in cat_order
                    for condition in condition_order
                ]
                markers = ["o", "s", "^", "D", "v", "P"]
                for index, (category, condition) in enumerate(plot_groups):
                    rows = per_sample[
                        (per_sample["category"] == category)
                        & (per_sample["condition"] == condition)
                    ]
                    jitter = (np.arange(len(rows)) - (len(rows) - 1) / 2) * 0.02
                    condition_index = list(condition_order).index(condition)
                    ax.scatter(
                        np.full(len(rows), index) + jitter,
                        rows["proportion"],
                        c=palette[category],
                        s=40,
                        marker=markers[condition_index % len(markers)],
                        edgecolors=edgecolor,
                        linewidths=0.3,
                        zorder=2,
                    )
                # One tick per category (block center); conditions use marker shape.
                n_conditions = len(condition_order)
                centers = [
                    i * n_conditions + (n_conditions - 1) / 2
                    for i in range(len(cat_order))
                ]
                ax.set_xticks(centers)
                ax.set_xticklabels(
                    [str(category) for category in cat_order],
                    rotation=45,
                    ha="right",
                )
                for boundary in range(
                    n_conditions,
                    len(plot_groups),
                    n_conditions,
                ):
                    ax.axvline(boundary - 0.5, color="#bdbdbd", linewidth=0.6, zorder=0)
                handles = [
                    mpl.lines.Line2D(
                        [],
                        [],
                        marker="o",
                        linestyle="",
                        markerfacecolor=palette[cat],
                        markeredgecolor=edgecolor,
                        markersize=6,
                        label=str(cat),
                    )
                    for cat in cat_order
                ]
                condition_handles = [
                    mpl.lines.Line2D(
                        [],
                        [],
                        marker=markers[index % len(markers)],
                        linestyle="",
                        markerfacecolor="#9e9e9e",
                        markeredgecolor=edgecolor,
                        markersize=6,
                        label=str(condition),
                    )
                    for index, condition in enumerate(condition_order)
                ]
                legend_cats = fig.legend(
                    handles=handles,
                    frameon=False,
                    loc="outside right upper",
                    title=category_by,
                )
                fig.add_artist(legend_cats)
                fig.legend(
                    handles=condition_handles,
                    frameon=False,
                    loc="outside right lower",
                    title="condition",
                )
            else:
                for i, cat in enumerate(cat_order):
                    sub = per_sample[per_sample["category"] == cat]
                    jitter = (np.arange(len(sub)) - (len(sub) - 1) / 2) * 0.02
                    ax.scatter(
                        np.full(len(sub), i) + jitter,
                        sub["proportion"],
                        c=palette[cat],
                        s=40,
                        edgecolors=edgecolor,
                        linewidths=0.3,
                        label=str(cat),
                        zorder=2,
                    )
                ax.set_xticks(range(len(cat_order)))
                ax.set_xticklabels(
                    [str(c) for c in cat_order],
                    rotation=45,
                    ha="right",
                )
                fig.legend(
                    frameon=False,
                    loc="outside right center",
                    title=category_by,
                )
            ax.set_ylabel("proportion")
            ymax = float(np.nanmax(per_sample["proportion"].to_numpy(dtype=np.float64)))
            ax.set_ylim(-0.02, min(1.02, max(0.2, ymax * 1.15)))
        apply_figure_chrome(fig, theme)

    tables = {"aggregate": aggregate}
    if per_sample is not None:
        tables["per_sample"] = per_sample

    from importlib.metadata import version

    try:
        scarf_version = version("scarf")
    except Exception:
        scarf_version = "unknown"

    notes = ["composition", kind]
    if pair_col is not None:
        notes.append(f"paired_by={pair_col}")

    result = PlotResult(
        figure=fig,
        axes=axes,
        tables=tables,
        legends=(LegendSpec(kind="categorical", label=category_by),),
        scales=(CategoricalScale(order=tuple(cat_order), palette=palette),),
        provenance=PlotProvenance(
            scarf_version=scarf_version,
            cell_key=cell_key,
            n_cells=int(len(cats)),
            n_samples=int(per_sample["sample"].nunique())
            if per_sample is not None
            else None,
            renderer="matplotlib",
            notes=tuple(notes),
            extras={
                "subject_by": subject_by,
                "pair_by": pair_by,
                "condition_by": condition_by,
                "n_pair_lines": n_pair_lines,
                "n_unpaired_samples": n_unpaired_samples,
                "dropped_sample_cells": dropped_sample_cells,
            },
        ),
        owns_figure=owns,
        theme=theme,
    )
    if show:
        result.show()
    return result
