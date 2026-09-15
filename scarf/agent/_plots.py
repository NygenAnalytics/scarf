"""Bounded views of the exact artifacts selected by an agent analysis."""

import hashlib
from collections import Counter
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

from ..graph.feature_projection import graph_cell_selection
from ..storage.refs import ArtifactRef
from ..storage.selections import validate_stored_selection_integrity
from ..storage.types import as_zarr_array

if TYPE_CHECKING:
    from ..datastore.datastore import DataStore
    from ..plotting import PlotResult


DISPLAY_CELLS = 50_000
DISPLAY_BLOCK_ROWS = 100_000


def cluster_counts(store: "DataStore", clusters: ArtifactRef) -> dict[str, int]:
    """Count every saved cluster label while retaining only one row block."""
    values = as_zarr_array(store.load_artifact(clusters)["values"], name="values")
    if values.ndim != 1:
        raise ValueError("Cluster labels must be one-dimensional")
    counts: Counter[str] = Counter()
    for start in range(0, values.shape[0], DISPLAY_BLOCK_ROWS):
        labels, frequencies = np.unique(
            np.asarray(values[start : start + DISPLAY_BLOCK_ROWS]).astype(str),
            return_counts=True,
        )
        counts.update(dict(zip(labels.tolist(), frequencies.tolist(), strict=True)))
    return dict(sorted(counts.items()))


def _selection_input(store: "DataStore", ref: ArtifactRef) -> ArtifactRef:
    status = store.inspect_artifact(ref)
    if not status.complete:
        raise ValueError("The selected analysis artifact is incomplete")
    raw = (status.inputs or {}).get("cell_selection")
    if not isinstance(raw, Mapping):
        raise ValueError("The selected artifact has no frozen cell selection")
    return ArtifactRef.from_dict(raw)


def _sample_quotas(counts: Mapping[str, int], maximum: int) -> dict[str, int]:
    """Allocate a proportional display, keeping at least one cell per cluster."""
    total = sum(counts.values())
    if total <= maximum:
        return dict(counts)
    if len(counts) > maximum:
        raise RuntimeError("There are more clusters than the display cell limit")
    remaining = maximum - len(counts)
    available = total - len(counts)
    shares = {
        label: remaining * (count - 1) / available for label, count in counts.items()
    }
    quotas = {label: 1 + int(share) for label, share in shares.items()}
    remainder = maximum - sum(quotas.values())
    for label in sorted(shares, key=lambda key: (-(shares[key] % 1), key))[:remainder]:
        quotas[label] += 1
    return quotas


def _sample_cluster_rows(
    values: Any,
    counts: Mapping[str, int],
    *,
    maximum: int,
    seed: int,
    block_rows: int = DISPLAY_BLOCK_ROWS,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep bounded per-cluster priority samples, independent of read boundaries."""
    quotas = _sample_quotas(counts, maximum)
    rng = np.random.Generator(np.random.PCG64(seed))
    priorities = {label: np.empty(0, dtype=np.float64) for label in counts}
    rows = {label: np.empty(0, dtype=np.int64) for label in counts}
    for start in range(0, int(values.shape[0]), block_rows):
        labels = np.asarray(values[start : start + block_rows]).astype(str)
        keys = rng.random(len(labels))
        for label in np.unique(labels):
            positions = np.flatnonzero(labels == label)
            combined_keys = np.concatenate((priorities[label], keys[positions]))
            combined_rows = np.concatenate((rows[label], positions + start))
            quota = quotas[label]
            if len(combined_rows) > quota:
                keep = np.argpartition(combined_keys, quota - 1)[:quota]
                combined_keys = combined_keys[keep]
                combined_rows = combined_rows[keep]
            priorities[label] = combined_keys
            rows[label] = combined_rows
    selected = np.concatenate(list(rows.values()))
    labels = np.concatenate(
        [np.full(len(rows[label]), label, dtype=object) for label in counts]
    )
    order = np.argsort(selected)
    return selected[order], labels[order]


def plot_final_umap(
    store: "DataStore",
    *,
    umap: ArtifactRef,
    clusters: ArtifactRef,
    cell_selection: ArtifactRef,
    graph: ArtifactRef,
    max_points: int = DISPLAY_CELLS,
    seed: int = 0,
    figsize: tuple[float, float] = (9, 6),
    show: bool = True,
) -> "PlotResult":
    """Draw a bounded categorical map from one completed analysis.

    Population counts always include every cell. Display sampling changes no
    saved selection, coordinates, labels, markers, or other analysis result.
    """
    from ..plotting import CategoricalScale, LegendSpec, PlotProvenance, PlotResult

    if (
        isinstance(max_points, bool)
        or not isinstance(max_points, int)
        or not 1 <= max_points <= DISPLAY_CELLS
    ):
        raise ValueError(f"max_points must be an integer from 1 to {DISPLAY_CELLS}")
    if umap.kind != "embedding" or clusters.kind != "cluster_labels":
        raise ValueError("The final map requires embedding and cluster-label artifacts")
    if umap.assay != clusters.assay or umap.scope != clusters.scope:
        raise ValueError("The final UMAP and clusters must belong to the same assay")
    for ref in (umap, clusters):
        if _selection_input(store, ref) != cell_selection:
            raise ValueError(
                "Final artifacts must share the exact frozen cell selection"
            )
    if graph_cell_selection(store.zw, graph) != cell_selection:
        raise ValueError("The final graph must use the exact frozen cell selection")
    embedding_status = store.inspect_artifact(umap)
    if embedding_status.operation != "run_umap":
        raise ValueError("The final UMAP must be a saved run_umap artifact")
    for ref in (umap, clusters):
        raw_graph = (store.inspect_artifact(ref).inputs or {}).get("graph")
        if (
            not isinstance(raw_graph, Mapping)
            or ArtifactRef.from_dict(raw_graph) != graph
        ):
            raise ValueError("The final UMAP and clusters must use the selected graph")
    selection = validate_stored_selection_integrity(
        store.zw,
        cell_selection,
        kind="cell_selection",
        scope="datastore",
        assay=None,
        table_path="cellData",
    )
    coordinates = as_zarr_array(store.load_artifact(umap)["values"], name="values")
    labels = as_zarr_array(store.load_artifact(clusters)["values"], name="values")
    total = selection.selected_count
    if coordinates.shape != (total, 2) or labels.shape != (total,):
        raise ValueError(
            "Final coordinates and labels must align with the frozen selection"
        )
    if np.dtype(coordinates.dtype).kind not in "fiu":
        raise TypeError("Final UMAP coordinates must be numeric")
    if total == 0:
        raise ValueError("The final selection contains no cells")
    counts = cluster_counts(store, clusters)
    rows, sampled_labels = _sample_cluster_rows(
        labels, counts, maximum=max_points, seed=seed
    )
    sampled_coordinates = np.empty((len(rows), 2), dtype=np.float64)
    for start in range(0, total, DISPLAY_BLOCK_ROWS):
        left, right = np.searchsorted(rows, [start, start + DISPLAY_BLOCK_ROWS])
        if left == right:
            continue
        block = np.asarray(coordinates[start : start + DISPLAY_BLOCK_ROWS])
        if not np.isfinite(block).all():
            raise ValueError("Final UMAP coordinates must be finite")
        sampled_coordinates[left:right] = block[rows[left:right] - start]

    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.colors import to_hex
    from matplotlib.lines import Line2D

    categories = tuple(counts)
    cmap = plt.get_cmap(
        "tab10"
        if len(categories) <= 10
        else "tab20"
        if len(categories) <= 20
        else "gist_rainbow"
    )
    palette = {
        label: to_hex(
            cmap(index if len(categories) <= 20 else index / (len(categories) - 1))
        )
        for index, label in enumerate(categories)
    }
    figure, axis = plt.subplots(figsize=figsize, constrained_layout=True)
    try:
        order = np.random.Generator(np.random.PCG64(seed)).permutation(len(rows))
        axis.scatter(
            sampled_coordinates[order, 0],
            sampled_coordinates[order, 1],
            color=[palette[label] for label in sampled_labels[order]],
            s=3,
            alpha=0.65,
            linewidths=0,
            rasterized=True,
        )
        axis.set(xlabel="UMAP 1", ylabel="UMAP 2", xticks=[], yticks=[])
        axis.spines[["top", "right"]].set_visible(False)
        if len(categories) <= 40:
            axis.legend(
                handles=[
                    Line2D(
                        [],
                        [],
                        color=palette[label],
                        marker="o",
                        linestyle="",
                        label=f"{label} ({counts[label]:,})",
                    )
                    for label in categories
                ],
                title="Cluster (all cells)",
                loc="center left",
                bbox_to_anchor=(1, 0.5),
                frameon=False,
                fontsize=8,
            )
        else:
            for label in categories:
                center = np.median(sampled_coordinates[sampled_labels == label], axis=0)
                axis.annotate(str(label), center, fontsize=7)
        caption = (
            f"{len(rows):,} of {total:,} cells shown; cluster counts use all cells."
        )
        axis.set_title(caption, fontsize=10)
        result = PlotResult(
            figure=figure,
            axes={"clusters": axis},
            tables={
                "cluster_counts": pd.DataFrame(
                    {"cluster": list(counts), "cells": list(counts.values())}
                )
            },
            legends=(LegendSpec(kind="categorical", label="Cluster (all cells)"),),
            scales=(CategoricalScale(order=categories, palette=palette),),
            provenance=PlotProvenance(
                assay=umap.assay,
                n_cells=len(rows),
                notes=("Final saved UMAP; display sampling only.",),
                extras={
                    "layout": umap.to_dict(),
                    "color_artifacts": [clusters.to_dict()],
                    "cell_selection": cell_selection.to_dict(),
                    "graph": graph.to_dict(),
                    "workspace": store.workspace,
                    "input_n_cells": total,
                    "cluster_counts": counts,
                    "display_sampling": {
                        "method": "proportional_clusters_with_minimum_one",
                        "max_points": max_points,
                        "seed": seed,
                        "generator": "PCG64",
                        "row_sha256": hashlib.sha256(
                            rows.astype("<i8").tobytes()
                        ).hexdigest(),
                    },
                },
            ),
            owns_figure=True,
        )
        if show:
            result.show()
        return result
    except BaseException:
        plt.close(figure)
        raise
