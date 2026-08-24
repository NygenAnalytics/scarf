"""Generate publication plotting acceptance figures from the 1K PBMC fixture."""

import argparse
from collections.abc import Hashable, Sequence
from pathlib import Path
import shutil
import tarfile
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scarf import DataStore
import scarf.plotting as splt


_FIXTURE = Path("tests/datasets/1K_pbmc_citeseq.zarr.tar.gz")
_LAYOUT_FIXTURE = Path("tests/visual/showcase/plotting_showcase_layout.npz")
_MARKER_SETS = {
    "T cells": ("CD3D", "IL7R", "LTB"),
    "B cells": ("MS4A1", "CD79A", "CD37"),
    "Myeloid": ("LST1", "FCER1G", "CTSS"),
    "Cytotoxic": ("NKG7", "GNLY", "GZMB"),
}
_CLUSTER_KEY = "RNA_leiden_cluster"
_CELL_CYCLE_KEY = "RNA_cell_cycle_phase"


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


def _build_graph(store: DataStore) -> None:
    normalized = store.run_normalization(
        from_assay="RNA",
        cell_key="I",
        feat_key="hvgs",
        update_state=False,
    )
    reduction = store.run_pca(
        normalized,
        dims=11,
        pca_cell_key="I",
        update_state=False,
        local_cache=False,
    )
    ann_index = store.build_ann_index(
        reduction,
        ann_efc=50,
        ann_ef=50,
        ann_m=48,
        rand_state=4466,
        update_state=False,
    )
    store.build_embedding_initialization(
        reduction,
        n_centroids=1000,
        rand_state=4466,
    )
    neighbors = store.query_neighbors(
        ann_index,
        coordinates=reduction,
        k=11,
        update_state=False,
    )
    store.build_connectivity_map(neighbors)


def _apply_fixed_layout(store: DataStore, source: Path) -> None:
    with np.load(source, allow_pickle=False) as fixture:
        cell_ids = fixture["cellIds"].astype(str)
        umap = fixture["umap"]

    fixture_index = pd.Index(cell_ids)
    current_ids = np.asarray(store.cells.fetch("ids")).astype(str)
    positions = fixture_index.get_indexer(current_ids)
    if (
        not fixture_index.is_unique
        or len(fixture_index) != len(current_ids)
        or np.any(positions < 0)
    ):
        raise ValueError(
            f"Layout fixture at {source} does not match the selected datastore cells"
        )
    if umap.shape != (len(fixture_index), 2):
        raise ValueError(f"Layout fixture at {source} must contain two UMAP columns")

    store.cells.insert(
        "RNA_UMAP1",
        umap[positions, 0],
        key="I",
        overwrite=True,
    )
    store.cells.insert(
        "RNA_UMAP2",
        umap[positions, 1],
        key="I",
        overwrite=True,
    )


def _prepare_store(
    source: Path,
    work_directory: Path,
    layout_fixture: Path = _LAYOUT_FIXTURE,
) -> DataStore:
    store = DataStore(
        str(_copy_fixture(source, work_directory)),
        default_assay="RNA",
        nthreads=2,
    )
    store.auto_filter_cells(show_qc_plots=False)
    if "hvgs" not in store._get_assay("RNA").feats.columns:
        store.mark_hvgs(
            top_n=100,
            show_plot=False,
            bin_strategy="fixed",
            min_cells=int(0.01 * store.cells.N),
            max_cells=np.inf,
            blacklist="^MT-|^RPS|^RPL|^MRPS|^MRPL|^CCN|^HLA-|^H2-|^HIST",
        )
    try:
        store.get_latest_graph_loc(
            from_assay="RNA",
            cell_key="I",
            feat_key="hvgs",
        )
    except KeyError:
        _build_graph(store)
    if _CLUSTER_KEY not in store.cells.columns:
        store.run_leiden_clustering()
    _apply_fixed_layout(store, layout_fixture)
    return store


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


def _relative_cycling_share_per_cluster(store: DataStore) -> dict[object, str]:
    """Group clusters by their relative share of cells outside G1."""
    clusters = pd.Series(np.asarray(store.cells.fetch(_CLUSTER_KEY)))
    phases = pd.Series(np.asarray(store.cells.fetch(_CELL_CYCLE_KEY)).astype(str))
    share = (phases != "G1").groupby(clusters).mean()
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
) -> list[Path]:
    outputs: list[Path] = []
    store.run_marker_search(group_key=_CLUSTER_KEY)
    cycling_share = _relative_cycling_share_per_cluster(store)
    cycling_scale = splt.CategoricalScale(order=("low", "medium", "high"))
    heatmap = splt.marker_heatmap(
        store,
        group_key=_CLUSTER_KEY,
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
    heatmap.axes["heatmap"].set_xlabel("Leiden cluster")
    outputs.append(_save(heatmap, output_directory / "marker_heatmap.png"))

    matrix = splt.matrixplot(
        store,
        features=markers,
        group_by=_CLUSTER_KEY,
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
) -> list[Path]:
    outputs: list[Path] = []
    boxes = splt.distribution(
        store,
        keys=("RNA_S_score", "RNA_G2M_score"),
        group_by=_CELL_CYCLE_KEY,
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
        axis.set_xlabel("Cell-cycle phase")
        axis.set_ylabel("score")
    outputs.append(_save(boxes, output_directory / "cell_cycle_scores.png"))

    clusters = np.asarray(store.cells.fetch(_CLUSTER_KEY))
    highlighted_group = pd.Series(clusters).value_counts().index[0]
    highlighted = splt.embedding(
        store,
        layout_key="RNA_UMAP",
        color_by=None,
        default_color="#bdbdbd",
        point_alpha=0.4,
        highlight=splt.Highlight(
            by=_CLUSTER_KEY,
            groups=(highlighted_group,),
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
) -> Path:
    dark = splt.embedding(
        store,
        layout_key="RNA_UMAP",
        color_by=[_CLUSTER_KEY, feature],
        legend_loc="on_data",
        color_scale=splt.ColorScale(cmap="magma", quantiles=(0, 0.99)),
        theme="dark",
        show=False,
    )
    next(iter(dark.axes.values())).set_title("Leiden clusters")
    return _save(dark, output_directory / "dark_embedding.png")


def generate_showcase(store: DataStore, output_directory: Path) -> list[Path]:
    output_directory.mkdir(parents=True, exist_ok=True)
    markers = _marker_sets(store)
    feature = _first_continuous_feature(store, markers)
    cluster_key = _CLUSTER_KEY
    cell_cycle_key = _CELL_CYCLE_KEY
    if cell_cycle_key not in store.cells.columns:
        store.run_cell_cycle_scoring()

    outputs: list[Path] = []
    categorical = splt.embedding(
        store,
        layout_key="RNA_UMAP",
        color_by=cluster_key,
        legend_loc="on_data",
        theme="paper",
        show=False,
    )
    next(iter(categorical.axes.values())).set_title("Leiden clusters")
    outputs.append(_save(categorical, output_directory / "categorical_embedding.png"))

    continuous = splt.embedding(
        store,
        layout_key="RNA_UMAP",
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
        group_by=cluster_key,
        layout_key="RNA_UMAP",
        feat_key="hvgs",
        show_cells=True,
        cell_alpha=0.3,
        theme="paper",
        show=False,
    )
    outputs.append(_save(connectivity, output_directory / "cluster_connectivity.png"))

    dotplot = splt.dotplot(
        store,
        features=markers,
        group_by=cluster_key,
        normalization=splt.NormalizationSpec(transform="log1p"),
        theme="paper",
        show=False,
    )
    dotplot.axes["dotplot"].set_xlabel("Leiden cluster")
    outputs.append(_save(dotplot, output_directory / "grouped_dotplot.png"))

    violin = splt.distribution(
        store,
        keys=[gene for genes in markers.values() for gene in genes[:1]],
        group_by=cluster_key,
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
            axis.set_xlabel("Leiden cluster")
    outputs.append(_save(violin, output_directory / "stacked_violin.png"))

    composition = splt.composition(
        store,
        category_by=cell_cycle_key,
        sample_by=cluster_key,
        segment_linewidth=0.7,
        show_percent_labels=True,
        label_min_fraction=0.06,
        theme="paper",
        show=False,
    )
    composition.axes["composition"].set_xlabel("Leiden cluster")
    for legend in composition.figure.legends:
        if legend.get_title().get_text() == cell_cycle_key:
            legend.set_title("Cell-cycle phase")
    outputs.append(_save(composition, output_directory / "composition.png"))

    outputs.extend(_expression_matrix_figures(store, output_directory, markers))
    outputs.extend(_additional_real_data_figures(store, output_directory))
    outputs.append(_dark_theme_figure(store, output_directory, feature))

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
            layout_key="RNA_UMAP",
            color_by=cluster_key,
            legend_loc="on_data",
            show_legend=False,
            show_titles=False,
            target=axes["embedding"],
            theme="paper",
            show=False,
        ),
        "continuous": splt.embedding(
            store,
            layout_key="RNA_UMAP",
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
            group_by=cluster_key,
            layout_key="RNA_UMAP",
            feat_key="hvgs",
            show_cells=True,
            cell_alpha=0.3,
            target=axes["connectivity"],
            theme="paper",
            show=False,
        ),
        "dotplot": splt.dotplot(
            store,
            features=markers,
            group_by=cluster_key,
            normalization=splt.NormalizationSpec(transform="log1p"),
            swap_axes=True,
            show_legend=False,
            target=axes["dotplot"],
            theme="paper",
            show=False,
        ),
        "composition": splt.composition(
            store,
            category_by=cell_cycle_key,
            sample_by=cluster_key,
            segment_linewidth=0.7,
            show_percent_labels=False,
            label_min_fraction=0.06,
            show_legend=False,
            target=axes["composition"],
            theme="paper",
            show=False,
        ),
    }
    axes["dotplot"].set_ylabel("Leiden cluster")
    axes["composition"].set_xlabel("Leiden cluster")
    composite = splt.compose_results(figure, children, shared_legends=True)
    for legend in figure.legends:
        title = legend.get_title().get_text()
        if title == cluster_key:
            legend.set_title("Leiden clusters")
        elif title == cell_cycle_key:
            legend.set_title("Cell-cycle phase")
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
        store = _prepare_store(
            arguments.fixture,
            Path(temporary),
            arguments.layout_fixture,
        )
        outputs = generate_showcase(store, arguments.output_dir)
    print("\n".join(str(path) for path in outputs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
