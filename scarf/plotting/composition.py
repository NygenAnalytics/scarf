"""Sample-level composition figures, including paired subject lines."""

from typing import Any, Hashable, Literal

import numpy as np
import pandas as pd

from ._contracts import CategoricalScale, PlotProvenance, StudyDesign
from ._deps import require_matplotlib
from ._display import resolve_categorical_scale
from ._figure import (
    LegendSpec,
    PlotResult,
    _place_legend_blocks,
    normalize_axes_target,
)
from ._style import (
    apply_figure_chrome,
    capped_figsize,
    categorical_color_map,
    foreground_color,
    scatter_edgecolor,
    sort_categories,
    theme_context,
)

_MISSING_CATEGORY = object()


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


def _summarize_proportions(
    per_sample: pd.DataFrame,
    *,
    by_condition: bool,
    uncertainty: Literal["none", "sd", "se", "ci95"],
) -> pd.DataFrame:
    columns = ["category"]
    if by_condition:
        columns.append("condition")
    grouped = per_sample.groupby(
        columns,
        observed=False,
        dropna=False,
    )["proportion"]
    summary = grouped.agg(["mean", "std", "count"]).reset_index()
    summary = summary.rename(
        columns={
            "mean": "mean_proportion",
            "count": "n_samples",
        }
    )
    spread = np.zeros(len(summary), dtype=np.float64)
    standard_deviation = summary["std"].fillna(0).to_numpy(dtype=np.float64)
    sample_count = summary["n_samples"].to_numpy(dtype=np.float64)
    if uncertainty == "sd":
        spread = standard_deviation
    elif uncertainty in ("se", "ci95"):
        standard_error = np.divide(
            standard_deviation,
            np.sqrt(sample_count),
            out=np.zeros_like(standard_deviation),
            where=sample_count > 1,
        )
        if uncertainty == "se":
            spread = standard_error
        else:
            from scipy.stats import t

            critical = np.ones_like(sample_count)
            valid = sample_count > 1
            critical[valid] = t.ppf(0.975, sample_count[valid] - 1)
            critical[~valid] = 0
            spread = standard_error * critical
    mean = summary["mean_proportion"].to_numpy(dtype=np.float64)
    summary["lower"] = np.clip(mean - spread, 0, 1)
    summary["upper"] = np.clip(mean + spread, 0, 1)
    return summary.drop(columns="std")


def _summary_marker_handle(
    mpl: Any,
    *,
    edgecolor: str,
    uncertainty: str,
) -> Any:
    spread_label = {
        "sd": "mean +/- SD",
        "se": "mean +/- SE",
        "ci95": "mean with 95% CI",
    }.get(uncertainty, "mean")
    return mpl.lines.Line2D(
        [],
        [],
        marker="D",
        linestyle="",
        markerfacecolor="#9e9e9e",
        markeredgecolor=edgecolor,
        markersize=5,
        label=spread_label,
    )


def _place_axis_legend_blocks(
    ax: Any,
    blocks: list[tuple[str, list[Any]]],
) -> None:
    slots = {
        1: (("upper left", (1.02, 1.0)),),
        2: (("upper left", (1.02, 1.0)), ("lower left", (1.02, 0.0))),
        3: (
            ("upper left", (1.02, 1.0)),
            ("center left", (1.02, 0.5)),
            ("lower left", (1.02, 0.0)),
        ),
    }
    legends: list[Any] = []
    for (location, anchor), (title, handles) in zip(slots[len(blocks)], blocks):
        if legends:
            ax.add_artist(legends[-1])
        legends.append(
            ax.legend(
                handles=handles,
                frameon=False,
                loc=location,
                bbox_to_anchor=anchor,
                title=title,
            )
        )


def _draw_summary_markers(
    ax: Any,
    summary: pd.DataFrame,
    *,
    positions: dict[tuple[Any, Any | None], float],
    palette: dict[Any, str],
    edgecolor: str,
) -> None:
    for row in summary.itertuples(index=False):
        condition = getattr(row, "condition", None)
        position = positions[(row.category, condition)]
        lower = float(row.lower)
        upper = float(row.upper)
        mean = float(row.mean_proportion)
        ax.errorbar(
            position,
            mean,
            yerr=np.asarray([[mean - lower], [upper - mean]]),
            fmt="D",
            markersize=4,
            markerfacecolor=palette[row.category],
            markeredgecolor=edgecolor,
            markeredgewidth=0.5,
            ecolor=edgecolor,
            elinewidth=1.0,
            capsize=2,
            zorder=4,
        )


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
    show_summary: bool = True,
    uncertainty: Literal["none", "sd", "se", "ci95"] | None = None,
    categorical_scale: CategoricalScale | None = None,
    bar_width: float = 0.82,
    bar_gap: float = 0.12,
    segment_edgecolor: str | None = None,
    segment_linewidth: float = 0.5,
    show_percent_labels: bool = False,
    label_min_fraction: float = 0.08,
    percent_format: str = "{:.0%}",
    target: Any | None = None,
    figsize: tuple[float, float] | None = None,
    max_figure_width: float | None = 7.5,
    theme: str = "notebook",
    show_legend: bool = True,
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
    Per-sample plots show sample means with 95% confidence intervals by
    default. Use ``uncertainty`` to select standard deviation, standard error,
    or no interval.

    Figure width is capped so large category lists stay page-sized.
    """
    _, mpl = require_matplotlib()
    if kind not in ("stacked", "per_sample"):
        raise ValueError("kind must be 'stacked' or 'per_sample'")
    if uncertainty not in (None, "none", "sd", "se", "ci95"):
        raise ValueError("uncertainty must be 'none', 'sd', 'se', 'ci95', or None")
    if kind == "stacked" and uncertainty not in (None, "none"):
        raise ValueError("uncertainty is available only for kind='per_sample'")
    resolved_uncertainty: Literal["none", "sd", "se", "ci95"] = (
        "ci95"
        if kind == "per_sample" and uncertainty is None
        else uncertainty or "none"
    )
    if bar_width <= 0 or bar_gap < 0:
        raise ValueError("bar_width must be positive and bar_gap non-negative")
    if segment_linewidth < 0:
        raise ValueError("segment_linewidth must be non-negative")
    if not 0 <= label_min_fraction <= 1:
        raise ValueError("label_min_fraction must be between 0 and 1")
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
    categorical_scale = resolve_categorical_scale(
        store,
        category_by,
        categorical_scale,
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
    missing_mask = pd.isna(cats)
    cats[missing_mask] = _MISSING_CATEGORY
    observed_categories = [
        value for value in pd.unique(cats) if value is not _MISSING_CATEGORY
    ]
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
    if missing_mask.any():
        cat_order.append(_MISSING_CATEGORY)

    pair_col: str | None = None
    n_pair_lines = 0
    n_unpaired_samples = 0
    dropped_sample_cells = 0
    summary_table: pd.DataFrame | None = None
    condition_order: list[Any] | None = None

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
        if kind == "per_sample" and show_summary:
            summary_source = per_sample
            if condition_by is not None:
                valid_condition = pd.notna(per_sample["condition"]) & (
                    per_sample["condition"].astype(str) != ""
                )
                summary_source = per_sample.loc[valid_condition]
            summary_table = _summarize_proportions(
                summary_source,
                by_condition=condition_by is not None,
                uncertainty=resolved_uncertainty,
            )

    nonmissing_order = [
        category for category in cat_order if category is not _MISSING_CATEGORY
    ]
    palette = categorical_color_map(
        nonmissing_order,
        palette=categorical_scale.palette if categorical_scale else None,
        palette_name=(
            categorical_scale.palette_name if categorical_scale else "default"
        ),
    )
    resolved_missing_color = (
        categorical_scale.missing_color if categorical_scale is not None else "#bdbdbd"
    )
    if missing_mask.any():
        palette[_MISSING_CATEGORY] = resolved_missing_color
    display_labels = (
        categorical_scale.labels
        if categorical_scale is not None and categorical_scale.labels is not None
        else {}
    )

    def category_label(value: Any) -> str:
        if value is _MISSING_CATEGORY:
            return missing_label
        return display_labels.get(value, str(value))

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
                max_width=max_figure_width,
            )
        elif kind == "stacked" and sample_by is not None and per_sample is not None:
            n_samples = int(per_sample["sample"].nunique())
            resolved_figsize = capped_figsize(
                max(4.5, 0.28 * n_samples + 1.8),
                3.6,
                max_width=max_figure_width,
            )
        else:
            resolved_figsize = capped_figsize(
                max(4.5, 0.35 * len(cat_order) + 1.8),
                3.6,
                max_width=max_figure_width,
            )
    edgecolor = scatter_edgecolor(theme)
    segment_border = segment_edgecolor or foreground_color(theme)

    with theme_context(theme):
        fig, axes, owns = normalize_axes_target(
            target,
            panel_keys=[panel_key],
            figsize=resolved_figsize,
        )
        ax = axes[panel_key]
        if kind == "stacked":

            def segment_text_color(face: str) -> str:
                rgb = mpl.colors.to_rgb(face)
                luminance = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
                return "#111111" if luminance > 0.58 else "#ffffff"

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
                        label=category_label(cat),
                        width=bar_width,
                        edgecolor=segment_border,
                        linewidth=segment_linewidth,
                    )
                    if show_percent_labels and val >= label_min_fraction:
                        ax.text(
                            0,
                            bottom + val / 2,
                            percent_format.format(val),
                            ha="center",
                            va="center",
                            color=segment_text_color(palette[cat]),
                            fontsize=7,
                            clip_on=True,
                        )
                    bottom += val
                ax.set_xticks([])
                ax.set_ylabel("proportion")
            else:
                assert per_sample is not None and props_mat is not None
                samples_order = list(props_mat.index)
                bottoms = np.zeros(len(samples_order))
                x = np.arange(len(samples_order)) * (bar_width + bar_gap)
                for cat in cat_order:
                    heights = props_mat[cat].to_numpy(dtype=np.float64)
                    heights = np.nan_to_num(heights, nan=0.0)
                    ax.bar(
                        x,
                        heights,
                        bottom=bottoms,
                        color=palette[cat],
                        label=category_label(cat),
                        width=bar_width,
                        edgecolor=segment_border,
                        linewidth=segment_linewidth,
                    )
                    if show_percent_labels:
                        for xpos, bottom, height in zip(x, bottoms, heights):
                            if height < label_min_fraction:
                                continue
                            ax.text(
                                xpos,
                                bottom + height / 2,
                                percent_format.format(height),
                                ha="center",
                                va="center",
                                color=segment_text_color(palette[cat]),
                                fontsize=6.5,
                                clip_on=True,
                            )
                    bottoms += heights
                ax.set_xticks(x)
                ax.set_xticklabels(
                    [str(s) for s in samples_order], rotation=45, ha="right"
                )
                ax.set_xlabel(sample_by)
                ax.set_ylabel("proportion")
            if show_legend:
                legend_kwargs = {
                    "frameon": False,
                    "title": category_by,
                    "ncols": max(1, int(np.ceil(len(cat_order) / 20))),
                    "columnspacing": 0.8,
                }
                if owns:
                    fig.legend(loc="outside right center", **legend_kwargs)
                else:
                    ax.legend(
                        loc="upper left",
                        bbox_to_anchor=(1.02, 1),
                        borderaxespad=0,
                        **legend_kwargs,
                    )
        else:
            assert per_sample is not None
            if condition_by is not None:
                valid_conditions = per_sample["condition"].dropna()
                valid_conditions = valid_conditions[valid_conditions.astype(str) != ""]
                condition_order = _sort_conditions(list(valid_conditions.unique()))
                if not condition_order:
                    raise ValueError("condition_by has no valid sample values")
                if pair_col is not None and len(condition_order) < 2:
                    raise ValueError(
                        "Paired composition requires at least two condition values"
                    )
                if pair_col is not None:
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
                summary_positions = {
                    (category, condition): float(index)
                    for index, (category, condition) in enumerate(plot_groups)
                }
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
                if summary_table is not None:
                    _draw_summary_markers(
                        ax,
                        summary_table,
                        positions=summary_positions,
                        palette=palette,
                        edgecolor=edgecolor,
                    )
                # One tick per category (block center); conditions use marker shape.
                n_conditions = len(condition_order)
                centers = [
                    i * n_conditions + (n_conditions - 1) / 2
                    for i in range(len(cat_order))
                ]
                ax.set_xticks(centers)
                ax.set_xticklabels(
                    [category_label(category) for category in cat_order],
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
                        label=category_label(cat),
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
                summary_handle = (
                    _summary_marker_handle(
                        mpl,
                        edgecolor=edgecolor,
                        uncertainty=resolved_uncertainty,
                    )
                    if summary_table is not None
                    else None
                )
                if show_legend:
                    blocks = [
                        (category_by, handles),
                        ("Condition", condition_handles),
                    ]
                    if summary_handle is not None:
                        blocks.append(("Summary", [summary_handle]))
                    if owns:
                        _place_legend_blocks(
                            fig,
                            [
                                (
                                    title,
                                    block_handles,
                                    [
                                        str(handle.get_label())
                                        for handle in block_handles
                                    ],
                                )
                                for title, block_handles in blocks
                            ],
                        )
                    else:
                        _place_axis_legend_blocks(ax, blocks)
            else:
                summary_positions = {
                    (category, None): float(index)
                    for index, category in enumerate(cat_order)
                }
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
                        label=category_label(cat),
                        zorder=2,
                    )
                if summary_table is not None:
                    _draw_summary_markers(
                        ax,
                        summary_table,
                        positions=summary_positions,
                        palette=palette,
                        edgecolor=edgecolor,
                    )
                ax.set_xticks(range(len(cat_order)))
                ax.set_xticklabels(
                    [category_label(c) for c in cat_order],
                    rotation=45,
                    ha="right",
                )
                summary_handle = (
                    _summary_marker_handle(
                        mpl,
                        edgecolor=edgecolor,
                        uncertainty=resolved_uncertainty,
                    )
                    if summary_table is not None
                    else None
                )
                if show_legend:
                    blocks = [(category_by, list(ax.get_legend_handles_labels()[0]))]
                    if summary_handle is not None:
                        blocks.append(("Summary", [summary_handle]))
                    if owns:
                        _place_legend_blocks(
                            fig,
                            [
                                (
                                    title,
                                    block_handles,
                                    [
                                        str(handle.get_label())
                                        for handle in block_handles
                                    ],
                                )
                                for title, block_handles in blocks
                            ],
                        )
                    else:
                        _place_axis_legend_blocks(ax, blocks)
            ax.set_xlabel(category_by)
            ax.set_ylabel("proportion")
            ymax = float(np.nanmax(per_sample["proportion"].to_numpy(dtype=np.float64)))
            if summary_table is not None:
                ymax = max(
                    ymax,
                    float(np.nanmax(summary_table["upper"].to_numpy(dtype=np.float64))),
                )
            ax.set_ylim(-0.02, min(1.02, max(0.2, ymax * 1.15)))
        apply_figure_chrome(fig, theme)

    aggregate_table = aggregate.copy()
    aggregate_table["category"] = aggregate_table["category"].mask(
        aggregate_table["category"] == _MISSING_CATEGORY,
        None,
    )
    tables = {"aggregate": aggregate_table}
    if per_sample is not None:
        per_sample_table = per_sample.copy()
        per_sample_table["category"] = per_sample_table["category"].mask(
            per_sample_table["category"] == _MISSING_CATEGORY,
            None,
        )
        tables["per_sample"] = per_sample_table
    if summary_table is not None:
        summary_output = summary_table.copy()
        summary_output["category"] = summary_output["category"].mask(
            summary_output["category"] == _MISSING_CATEGORY,
            None,
        )
        tables["summary"] = summary_output

    notes = ["composition", kind]
    if pair_col is not None:
        notes.append(f"paired_by={pair_col}")

    legends = [LegendSpec(kind="categorical", label=category_by)]
    if condition_order is not None:
        legends.append(
            LegendSpec(
                kind="marker",
                label=condition_by,
                extras={
                    "values": list(condition_order),
                    "markers": [
                        ["o", "s", "^", "D", "v", "P"][index % 6]
                        for index in range(len(condition_order))
                    ],
                },
            )
        )

    result = PlotResult(
        figure=fig,
        axes=axes,
        tables=tables,
        legends=tuple(legends),
        scales=(
            CategoricalScale(
                order=tuple(nonmissing_order),
                palette={category: palette[category] for category in nonmissing_order},
                labels=(dict(display_labels) if display_labels else None),
                missing_color=resolved_missing_color,
                missing_label=(
                    categorical_scale.missing_label
                    if categorical_scale is not None
                    else "NA"
                ),
                palette_name=(
                    categorical_scale.palette_name
                    if categorical_scale is not None
                    else "default"
                ),
            ),
        ),
        provenance=PlotProvenance(
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
                "show_summary": show_summary if kind == "per_sample" else False,
                "uncertainty": (
                    resolved_uncertainty
                    if kind == "per_sample" and show_summary
                    else "none"
                ),
                "n_pair_lines": n_pair_lines,
                "n_unpaired_samples": n_unpaired_samples,
                "dropped_sample_cells": dropped_sample_cells,
                "bar_width": bar_width,
                "bar_gap": bar_gap,
                "segment_edgecolor": segment_border,
                "segment_linewidth": segment_linewidth,
                "show_percent_labels": show_percent_labels,
                "label_min_fraction": label_min_fraction,
            },
        ),
        owns_figure=owns,
        theme=theme,
    )
    if show:
        result.show()
    return result
