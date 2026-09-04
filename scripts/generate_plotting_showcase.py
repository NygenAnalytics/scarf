"""Generate publication plotting acceptance figures from the 1K PBMC fixture."""

import argparse
from collections.abc import Hashable, Sequence
from hashlib import file_digest
from pathlib import Path
import shutil
import tarfile
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scarf import DataStore
from scarf.embeddings import write_imported_embedding
import scarf.plotting as splt
from scarf.storage.artifacts import ArtifactRef, fingerprint_array
from scarf.storage.selections import read_stored_selection_indices
from scarf.storage.types import as_zarr_array


_FIXTURE = Path("tests/datasets/1K_pbmc_citeseq.zarr.tar.gz")
_LAYOUT_FIXTURE = Path("tests/visual/showcase/plotting_showcase_layout.npz")
_MARKER_SETS = {
    "T cells": ("CD3D", "IL7R", "LTB"),
    "B cells": ("MS4A1", "CD79A", "CD37"),
    "Myeloid": ("LST1", "FCER1G", "CTSS"),
    "Cytotoxic": ("NKG7", "GNLY", "GZMB"),
}
_CLUSTER_LABEL = "Leiden cluster"
_CELL_CYCLE_LABEL = "Cell-cycle phase"
# These internal legend identities preserve the committed composite geometry.
# They are replaced with the friendly labels before saving and are never metadata keys.
_CLUSTER_LEGEND_ID = "RNA_leiden_cluster"
_CELL_CYCLE_LEGEND_ID = "RNA_cell_cycle_phase"
_SHOWCASE_PALETTE = (
    "#4e79a7",
    "#f28e2b",
    "#e15759",
    "#76b7b2",
    "#59a14f",
    "#edc948",
    "#b07aa1",
    "#ff9da7",
    "#9c755f",
    "#bab0ac",
)


def _zarr_root(directory: Path) -> Path:
    if (directory / "zarr.json").exists() or (directory / ".zgroup").exists():
        return directory
    candidates = [
        path
        for path in directory.iterdir()
        if path.is_dir()
        and ((path / "zarr.json").exists() or (path / ".zgroup").exists())
    ]
    if len(candidates) != 1:
        raise ValueError(f"Could not identify one Zarr store under {directory}")
    return candidates[0]


def _copy_fixture(source: Path, destination: Path) -> Path:
    if source.is_dir():
        target = destination / "showcase.zarr"
        shutil.copytree(source, target)
        return target
    if not source.exists():
        raise FileNotFoundError(
            f"Fixture not found at {source}. Download the existing test fixtures first."
        )
    with tarfile.open(source, "r:gz") as archive:
        archive.extractall(destination, filter="data")
    return _zarr_root(destination)


def _build_graph(
    store: DataStore,
    cell_selection: ArtifactRef,
    features: ArtifactRef,
) -> ArtifactRef:
    normalized = store.run_normalization(
        cell_selection,
        features,
    )
    reduction = store.run_pca(
        normalized,
        dims=11,
        local_cache=False,
    )
    ann_index = store.build_ann_index(
        reduction,
        ann_efc=50,
        ann_ef=50,
        ann_m=48,
        rand_state=4466,
    )
    neighbors = store.query_neighbors(
        ann_index,
        coordinates=reduction,
        k=11,
    )
    return store.build_connectivity_map(neighbors)


def _import_fixed_layout(
    store: DataStore,
    cell_selection: ArtifactRef,
    source: Path,
) -> ArtifactRef:
    with np.load(source, allow_pickle=False) as fixture:
        cell_ids = fixture["cellIds"].astype(str)
        umap = fixture["umap"]

    fixture_index = pd.Index(cell_ids)
    selected = read_stored_selection_indices(
        store.zw,
        cell_selection,
        kind="cell_selection",
        scope="datastore",
        assay=None,
        table_path="cellData",
    )
    selected_ids = np.asarray(store.cells.fetch_all("ids")).astype(str)[selected]
    positions = fixture_index.get_indexer(selected_ids)
    if (
        not fixture_index.is_unique
        or len(fixture_index) != len(selected_ids)
        or np.any(positions < 0)
    ):
        raise ValueError(
            f"Layout fixture at {source} does not match the selected datastore cells"
        )
    if umap.shape != (len(fixture_index), 2):
        raise ValueError(f"Layout fixture at {source} must contain two UMAP columns")
    coordinates = np.asarray(umap[positions], dtype=np.float32)
    with source.open("rb") as stream:
        source_digest = file_digest(stream, "sha256").digest()
    return write_imported_embedding(
        store.zw,
        assay="RNA",
        dimreduc_key="plotting_showcase_umap",
        role="umap",
        coordinates=coordinates,
        source_digest=source_digest,
        payload_fingerprints={"values": fingerprint_array(coordinates)},
        source_cell_ids=selected_ids,
        cell_selection=cell_selection,
    )


def _prepare_store(
    source: Path,
    work_directory: Path,
    layout_fixture: Path = _LAYOUT_FIXTURE,
) -> tuple[DataStore, dict[str, ArtifactRef]]:
    store = DataStore(
        str(_copy_fixture(source, work_directory)),
        default_assay="RNA",
        nthreads=2,
    )
    cell_selection = store.auto_filter_cells()
    features = store.select_hvgs(
        cell_selection,
        top_n=100,
        show_plot=False,
        bin_strategy="fixed",
        min_cells=int(0.01 * store.cells.N),
        max_cells=np.inf,
        blacklist="^MT-|^RPS|^RPL|^MRPS|^MRPL|^CCN|^HLA-|^H2-|^HIST",
    )
    marker_features = store.set_feature_selection(
        from_assay="RNA",
        mask=np.ones(store.get_assay("RNA").feats.N, dtype=bool),
    )
    graph = _build_graph(store, cell_selection, features)
    clusters = store.run_leiden_clustering(graph)
    cell_cycle = store.run_cell_cycle_scoring(cell_selection)
    layout = _import_fixed_layout(store, cell_selection, layout_fixture)
    return store, {
        "cell_selection": cell_selection,
        "features": marker_features,
        "graph": graph,
        "clusters": clusters,
        "cell_cycle": cell_cycle,
        "layout": layout,
    }


def _marker_sets(store: DataStore) -> dict[str, list[str]]:
    features = store._get_assay("RNA").feats
    available = {str(value) for value in features.fetch_all("names")}
    selected = {
        group: [gene for gene in genes if gene in available]
        for group, genes in _MARKER_SETS.items()
    }
    selected = {group: genes for group, genes in selected.items() if genes}
    if selected:
        return selected
    fallback = [str(value) for value in features.fetch_all("names")[:8]]
    midpoint = max(1, len(fallback) // 2)
    return {"Set 1": fallback[:midpoint], "Set 2": fallback[midpoint:]}


def _first_continuous_feature(store: DataStore, markers: dict[str, list[str]]) -> str:
    for genes in markers.values():
        if genes:
            return genes[0]
    return "RNA_nCounts"


def _save(result: splt.PlotResult, path: Path) -> Path:
    output = result.save(path, dpi=180, exact_size=True)
    result.close()
    return output


def _normalize_svg(path: Path) -> Path:
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text(
        "\n".join(line.rstrip() for line in lines) + "\n",
        encoding="utf-8",
    )
    return path


def _label_categorical_legend(result: splt.PlotResult, label: str) -> None:
    for legend in result.legends:
        if legend.kind == "categorical":
            legend.label = label


def _artifact_values(
    store: DataStore,
    artifact: ArtifactRef,
    value_name: str,
) -> np.ndarray:
    group = store.load_artifact(artifact)
    return np.asarray(as_zarr_array(group[value_name], name=value_name)[:])


def _showcase_categorical_scale(values: np.ndarray) -> splt.CategoricalScale:
    order = tuple(sorted(pd.unique(values).tolist()))
    if len(order) > len(_SHOWCASE_PALETTE):
        return splt.CategoricalScale(order=order)
    return splt.CategoricalScale(
        order=order,
        palette=dict(zip(order, _SHOWCASE_PALETTE, strict=False)),
    )


def _relative_cycling_share_per_cluster(
    store: DataStore,
    clusters: ArtifactRef,
    cell_cycle: ArtifactRef,
) -> dict[object, str]:
    """Group clusters by their relative share of cells outside G1."""
    cluster_values = _artifact_values(store, clusters, "values")
    phase_values = _artifact_values(store, cell_cycle, "phase").astype(str)
    if cluster_values.shape != phase_values.shape:
        raise ValueError("Cluster and cell-cycle artifacts do not align")
    share = (pd.Series(phase_values) != "G1").groupby(cluster_values).mean()
    n_groups = min(3, len(share))
    labels = ("low", "medium", "high")[:n_groups]
    bins = pd.qcut(
        share.rank(method="first"),
        q=n_groups,
        labels=labels,
    )
    return {label: str(value) for label, value in bins.items()}


def _expression_matrix_figures(
    store: DataStore,
    output_directory: Path,
    markers: dict[str, list[str]],
    *,
    clusters: ArtifactRef,
    cell_cycle: ArtifactRef,
    features: ArtifactRef,
) -> list[Path]:
    outputs: list[Path] = []
    marker_table = store.run_marker_search(
        clusters,
        features=features,
    )
    cycling_share = _relative_cycling_share_per_cluster(
        store,
        clusters,
        cell_cycle,
    )
    cycling_scale = splt.CategoricalScale(order=("low", "medium", "high"))
    heatmap = splt.marker_heatmap(
        store,
        marker=marker_table,
        topn=4,
        figsize=(6.0, 7.5),
        fontsize=8,
        cluster_columns=True,
        color_scale=splt.ColorScale(
            cmap="RdBu_r",
            vmin=-1.0,
            vcenter=0.0,
            vmax=2.0,
        ),
        column_annotations={"relative cycling share": cycling_share},
        annotation_scales={"relative cycling share": cycling_scale},
        theme="paper",
        show=False,
    )
    heatmap.axes["heatmap"].set_xlabel(_CLUSTER_LABEL)
    outputs.append(_save(heatmap, output_directory / "marker_heatmap.png"))

    matrix = splt.matrixplot(
        store,
        features=markers,
        groups=clusters,
        normalization=splt.NormalizationSpec(transform="log1p"),
        standardize="feature",
        cluster_groups=True,
        color_scale=splt.ColorScale(cmap="RdBu_r", vcenter=0.0),
        column_annotations={
            "relative cycling share": {
                str(label): value for label, value in cycling_share.items()
            }
        },
        annotation_scales={"relative cycling share": cycling_scale},
        figsize=(6.0, 5.0),
        theme="paper",
        show=False,
    )
    outputs.append(_save(matrix, output_directory / "matrix_plot.png"))
    return outputs


def _additional_real_data_figures(
    store: DataStore,
    output_directory: Path,
    *,
    layout: ArtifactRef,
    clusters: ArtifactRef,
    cell_cycle: ArtifactRef,
    cell_cycle_scale: splt.CategoricalScale,
) -> list[Path]:
    outputs: list[Path] = []
    boxes = splt.distribution(
        store,
        keys=cell_cycle,
        grouping=cell_cycle,
        categorical_scale=cell_cycle_scale,
        kind="box",
        share_y=True,
        figsize=(7.2, 3.8),
        theme="paper",
        show=False,
    )
    for axis, title in zip(
        boxes.axes.values(),
        ("S-phase score", "G2M-phase score"),
        strict=True,
    ):
        axis.set_title(title)
        axis.set_xlabel(_CELL_CYCLE_LABEL)
        axis.set_ylabel("score")
    outputs.append(_save(boxes, output_directory / "cell_cycle_scores.png"))

    cluster_values = _artifact_values(store, clusters, "values")
    highlighted_group = pd.Series(cluster_values).value_counts().index[0]
    highlighted_indices = tuple(
        np.flatnonzero(cluster_values == highlighted_group).tolist()
    )
    highlighted = splt.embedding(
        store,
        layout=layout,
        color_by=None,
        default_color="#bdbdbd",
        point_alpha=0.4,
        highlight=splt.Highlight(
            indices=highlighted_indices,
            color="#d62728",
            dim_alpha=0.12,
            size_multiplier=1.35,
            halo_width=0.4,
        ),
        show_titles=False,
        theme="paper",
        show=False,
    )
    next(iter(highlighted.axes.values())).set_title(
        f"Cluster {highlighted_group} highlighted"
    )
    outputs.append(_save(highlighted, output_directory / "highlighted_embedding.png"))
    return outputs


def _dark_theme_figure(
    store: DataStore,
    output_directory: Path,
    feature: str,
    *,
    layout: ArtifactRef,
    clusters: ArtifactRef,
    cluster_scale: splt.CategoricalScale,
) -> Path:
    dark = splt.embedding(
        store,
        layout=layout,
        color_by=[clusters, feature],
        legend_loc="on_data",
        categorical_scale=cluster_scale,
        color_scale=splt.ColorScale(cmap="magma", quantiles=(0, 0.99)),
        theme="dark",
        show=False,
    )
    next(iter(dark.axes.values())).set_title("Leiden clusters")
    return _save(dark, output_directory / "dark_embedding.png")


def generate_showcase(
    store: DataStore,
    output_directory: Path,
    *,
    layout: ArtifactRef,
    graph: ArtifactRef,
    clusters: ArtifactRef,
    cell_cycle: ArtifactRef,
    features: ArtifactRef,
) -> list[Path]:
    output_directory.mkdir(parents=True, exist_ok=True)
    markers = _marker_sets(store)
    feature = _first_continuous_feature(store, markers)
    cluster_scale = _showcase_categorical_scale(
        _artifact_values(store, clusters, "values")
    )
    cell_cycle_scale = _showcase_categorical_scale(
        _artifact_values(store, cell_cycle, "phase").astype(str)
    )
    outputs: list[Path] = []
    categorical = splt.embedding(
        store,
        layout=layout,
        color_by=clusters,
        categorical_scale=cluster_scale,
        legend_loc="on_data",
        theme="paper",
        show=False,
    )
    next(iter(categorical.axes.values())).set_title("Leiden clusters")
    outputs.append(_save(categorical, output_directory / "categorical_embedding.png"))

    continuous = splt.embedding(
        store,
        layout=layout,
        color_by=feature,
        sort_values=True,
        color_scale=splt.ColorScale(cmap="magma", quantiles=(0, 0.99)),
        density_overlay=splt.DensityOverlay(
            statistic="mean",
            pixels=72,
            sigma=5.0,
            min_support=0.35,
            levels=(0.9,),
            max_hotspots=1,
            color="#303030",
            alpha=0.9,
            linewidth=1.2,
        ),
        show_titles=False,
        theme="paper",
        show=False,
    )
    outputs.append(_save(continuous, output_directory / "continuous_embedding.png"))

    connectivity = splt.cluster_connectivity(
        store,
        groups=clusters,
        layout=layout,
        graph=graph,
        categorical_scale=cluster_scale,
        show_cells=True,
        cell_alpha=0.3,
        theme="paper",
        show=False,
    )
    outputs.append(_save(connectivity, output_directory / "cluster_connectivity.png"))

    dotplot = splt.dotplot(
        store,
        features=markers,
        groups=clusters,
        normalization=splt.NormalizationSpec(transform="log1p"),
        theme="paper",
        show=False,
    )
    dotplot.axes["dotplot"].set_xlabel(_CLUSTER_LABEL)
    outputs.append(_save(dotplot, output_directory / "grouped_dotplot.png"))

    violin = splt.distribution(
        store,
        keys=[gene for genes in markers.values() for gene in genes[:1]],
        grouping=clusters,
        categorical_scale=cluster_scale,
        normalization=splt.NormalizationSpec(transform="log1p"),
        kind="stacked_violin",
        share_y=True,
        max_points=600,
        point_alpha=0.18,
        theme="paper",
        show=False,
    )
    for axis in violin.axes.values():
        if axis.get_xlabel():
            axis.set_xlabel(_CLUSTER_LABEL)
    outputs.append(_save(violin, output_directory / "stacked_violin.png"))

    composition = splt.composition(
        store,
        categories=cell_cycle,
        grouping=clusters,
        categorical_scale=cell_cycle_scale,
        segment_linewidth=0.7,
        show_percent_labels=True,
        label_min_fraction=0.06,
        theme="paper",
        show=False,
    )
    composition.axes["composition"].set_xlabel(_CLUSTER_LABEL)
    for legend in composition.figure.legends:
        legend.set_title(_CELL_CYCLE_LABEL)
    outputs.append(_save(composition, output_directory / "composition.png"))

    outputs.extend(
        _expression_matrix_figures(
            store,
            output_directory,
            markers,
            clusters=clusters,
            cell_cycle=cell_cycle,
            features=features,
        )
    )
    outputs.extend(
        _additional_real_data_figures(
            store,
            output_directory,
            layout=layout,
            clusters=clusters,
            cell_cycle=cell_cycle,
            cell_cycle_scale=cell_cycle_scale,
        )
    )
    outputs.append(
        _dark_theme_figure(
            store,
            output_directory,
            feature,
            layout=layout,
            clusters=clusters,
            cluster_scale=cluster_scale,
        )
    )

    figure, axes = plt.subplot_mosaic(
        [
            ["embedding", "continuous", "connectivity"],
            ["dotplot", "dotplot", "composition"],
        ],
        figsize=(11, 6.5),
        gridspec_kw={"width_ratios": [1.0, 1.0, 0.9], "height_ratios": [1, 0.9]},
        layout="constrained",
    )
    children: dict[Hashable, splt.PlotResult] = {
        "embedding": splt.embedding(
            store,
            layout=layout,
            color_by=clusters,
            categorical_scale=cluster_scale,
            legend_loc="on_data",
            show_legend=False,
            show_titles=False,
            target=axes["embedding"],
            theme="paper",
            show=False,
        ),
        "continuous": splt.embedding(
            store,
            layout=layout,
            color_by=feature,
            color_scale=splt.ColorScale(cmap="magma", quantiles=(0, 0.99)),
            density_overlay=splt.DensityOverlay(
                statistic="mean",
                pixels=60,
                sigma=4.2,
                min_support=0.35,
                levels=(0.9,),
                max_hotspots=1,
                color="#303030",
                alpha=0.9,
                linewidth=0.95,
            ),
            show_legend=False,
            show_titles=False,
            target=axes["continuous"],
            theme="paper",
            show=False,
        ),
        "connectivity": splt.cluster_connectivity(
            store,
            groups=clusters,
            layout=layout,
            graph=graph,
            categorical_scale=cluster_scale,
            show_cells=True,
            cell_alpha=0.3,
            target=axes["connectivity"],
            theme="paper",
            show=False,
        ),
        "dotplot": splt.dotplot(
            store,
            features=markers,
            groups=clusters,
            normalization=splt.NormalizationSpec(transform="log1p"),
            swap_axes=True,
            show_legend=False,
            target=axes["dotplot"],
            theme="paper",
            show=False,
        ),
        "composition": splt.composition(
            store,
            categories=cell_cycle,
            grouping=clusters,
            categorical_scale=cell_cycle_scale,
            segment_linewidth=0.7,
            show_percent_labels=False,
            label_min_fraction=0.06,
            show_legend=False,
            target=axes["composition"],
            theme="paper",
            show=False,
        ),
    }
    for name in ("embedding", "connectivity"):
        _label_categorical_legend(children[name], _CLUSTER_LEGEND_ID)
    _label_categorical_legend(children["composition"], _CELL_CYCLE_LEGEND_ID)
    axes["dotplot"].set_ylabel(_CLUSTER_LABEL)
    axes["composition"].set_xlabel(_CLUSTER_LABEL)
    composite = splt.compose_results(figure, children, shared_legends=True)
    for legend in figure.legends:
        title = legend.get_title().get_text()
        if title == _CLUSTER_LEGEND_ID:
            legend.set_title("Leiden clusters")
        elif title == _CELL_CYCLE_LEGEND_ID:
            legend.set_title(_CELL_CYCLE_LABEL)
    outputs.append(
        composite.save(
            output_directory / "publication_composite.png",
            dpi=220,
            exact_size=True,
        )
    )
    outputs.append(
        _normalize_svg(
            composite.save(
                output_directory / "publication_composite.svg",
                exact_size=True,
            )
        )
    )
    plt.close(figure)
    return outputs


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, default=_FIXTURE)
    parser.add_argument(
        "--layout-fixture",
        type=Path,
        default=_LAYOUT_FIXTURE,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plotting_showcase"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    with tempfile.TemporaryDirectory(prefix="scarf_plot_showcase_") as temporary:
        store, artifacts = _prepare_store(
            arguments.fixture,
            Path(temporary),
            arguments.layout_fixture,
        )
        outputs = generate_showcase(
            store,
            arguments.output_dir,
            layout=artifacts["layout"],
            graph=artifacts["graph"],
            clusters=artifacts["clusters"],
            cell_cycle=artifacts["cell_cycle"],
            features=artifacts["features"],
        )
    print("\n".join(str(path) for path in outputs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
