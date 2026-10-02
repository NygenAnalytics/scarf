"""Agent plots derived from exact, saved numerical artifacts."""

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from ..plotting import LegendSpec, PlotProvenance, PlotResult, theme_context
from ..utils.arrays import sort_categories


def marker_dotplot(
    store: Any, run: Any, *, top_n: int = 2, max_genes: int = 40, show: bool = False
) -> PlotResult:
    """Plot saved normalized means and expressing fractions without reading counts.

    Select up to ``top_n`` markers per cluster, with score >= 0.25 and expressing
    fraction >= 0.2. Take successive ranks across clusters up to ``max_genes``.
    Dot area shows expressing fraction; color shows log1p of the saved normalized
    mean, using one shared scale. Missing statistics remain missing.
    """
    for name, value, bound in (("top_n", top_n, 10), ("max_genes", max_genes, 60)):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 1 <= value <= bound
        ):
            raise ValueError(f"{name} must be an integer between 1 and {bound}")
    labels = run.cells.fetch("clusters")
    groups = sort_categories({str(label) for label in labels})
    ranks: list[dict[int, str]] = [{} for _ in range(top_n)]
    marker = run["markers"]
    for group in groups:
        frame = store.get_markers(
            marker=marker, group_id=group, min_score=-1, min_frac_exp=-1
        )
        candidates = frame.loc[(frame.score >= 0.25) & (frame.frac_exp >= 0.2)]
        for rank, row in enumerate(candidates.head(top_n).itertuples(index=False)):
            if len(ranks[rank]) < max_genes:
                ranks[rank].setdefault(int(row.feature_index), str(row.feature_name))
        del frame, candidates
    selected: dict[int, str] = {}
    for ranked in ranks:
        for index, name in ranked.items():
            if len(selected) < max_genes:
                selected.setdefault(index, name)
    if not selected:
        raise ValueError("No markers meet the score and expressing-fraction thresholds")
    pieces = []
    columns = ["group_id", "feature_index", "score", "mean", "frac_exp"]
    for group in groups:
        frame = store.get_markers(
            marker=marker, group_id=group, min_score=-1, min_frac_exp=-1
        )
        pieces.append(frame.loc[frame.feature_index.isin(selected), columns].copy())
        del frame
    index = pd.MultiIndex.from_product([groups, selected], names=columns[:2])
    table = pd.concat(pieces).set_index(columns[:2]).reindex(index).reset_index()
    measured = table["mean"].notna() | table["frac_exp"].notna()
    values = table.loc[measured, ["mean", "frac_exp"]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or (values[:, 1] > 1).any():
        raise ValueError(
            "Marker means and expressing fractions must be finite and valid"
        )
    counts = Counter(selected.values())
    names = {
        index: f"{name} [{index}]" if counts[name] > 1 else name
        for index, name in selected.items()
    }
    table["feature_name"] = table.feature_index.map(selected)
    table["feature_label"] = table.feature_index.map(names)
    table["log1p_mean"] = np.log1p(table["mean"])
    present = table.loc[measured]
    from matplotlib import pyplot as plt

    figure = None
    try:
        with theme_context("notebook"):
            figure, ax = plt.subplots(
                figsize=(
                    min(24, max(7, len(selected) * 0.32 + 3)),
                    min(24, max(4, len(groups) * 0.3 + 2)),
                ),
                dpi=200,
                layout="constrained",
            )
            dots = ax.scatter(
                present.feature_index.map({key: i for i, key in enumerate(selected)}),
                present.group_id.map({key: i for i, key in enumerate(groups)}),
                s=present.frac_exp * 140,
                c=present.log1p_mean,
                cmap="viridis",
                vmin=0,
                edgecolors="none",
            )
            ax.set_xticks(
                range(len(selected)), labels=list(names.values()), rotation=90
            )
            ax.set_yticks(range(len(groups)), labels=groups)
            ax.tick_params(labelsize=8)
            ax.set(
                xlim=(-0.6, len(selected) - 0.4),
                ylim=(len(groups) - 0.4, -0.6),
                xlabel="Marker gene",
                ylabel="Cluster",
                title="Marker expression across final clusters",
            )
            ax.set_axisbelow(True)
            ax.grid(color="#e8e8e8", linewidth=0.5)
            for spine in ax.spines.values():
                spine.set_visible(False)
            color_label = "log(1 + mean normalized expression)"
            figure.colorbar(
                dots,
                ax=ax,
                pad=0.025,
                shrink=0.65,
                location="bottom",
                label=color_label,
            )
            handles = [
                ax.scatter(
                    [],
                    [],
                    s=fraction * 140,
                    color="#777777",
                    edgecolors="none",
                    label=f"{fraction:.0%}",
                )
                for fraction in (0.25, 0.5, 0.75, 1)
            ]
            if (~measured).any():
                missing = table.loc[~measured]
                handles.append(
                    ax.scatter(
                        missing.feature_index.map(
                            {key: i for i, key in enumerate(selected)}
                        ),
                        missing.group_id.map({key: i for i, key in enumerate(groups)}),
                        s=12,
                        marker="x",
                        color="#888888",
                        linewidths=0.6,
                        label="Not measured",
                    )
                )
            ax.legend(
                handles=handles,
                title="Cells expressing",
                loc="upper left",
                bbox_to_anchor=(1.02, 1),
                frameon=False,
                fontsize=8,
                title_fontsize=8,
            )
            result = PlotResult(
                figure=figure,
                axes={"markers": ax},
                tables={"markers": table},
                legends=(
                    LegendSpec("colorbar", color_label),
                    LegendSpec("size", "Cells expressing"),
                ),
                scales=(),
                owns_figure=True,
                provenance=PlotProvenance(
                    assay=run.assay,
                    n_cells=len(labels),
                    notes=(
                        "Dot area is expressing fraction; color uses one shared log1p "
                        "normalized-mean scale. Missing statistics are not zero.",
                    ),
                    extras={
                        "runId": run.run_id,
                        "markers": marker.to_dict(),
                        "clusters": run["clusters"].to_dict(),
                        "topN": top_n,
                        "maxGenes": max_genes,
                        "minScore": 0.25,
                        "minFracExp": 0.2,
                        "featureIndices": list(selected),
                        "groupOrder": groups,
                    },
                ),
            )
            if show:
                result.show()
            return result
    except BaseException:
        if figure is not None:
            plt.close(figure)
        raise
