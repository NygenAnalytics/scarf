import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import pytest

from scarf.storage.artifacts import ArtifactRef

from . import full_path, remove, chunked_total_sum


def build_neighbourhood_graph(
    datastore,
    *,
    from_assay: str | None = None,
    cell_key: str = "I",
    features: ArtifactRef | str,
    reduction_method: str = "pca",
    dims: int = 11,
    pca_cell_key: str | None = None,
    k: int = 11,
    ann_metric: str = "l2",
    ann_efc: int | None = None,
    ann_ef: int | None = None,
    ann_m: int | None = None,
    ann_parallel: bool = False,
    rand_state: int = 4466,
    n_centroids: int = 1000,
    batch_size: int | None = None,
    log_transform: bool | None = None,
    renormalize_subset: bool | None = None,
    local_connectivity: float = 1.0,
    bandwidth: float = 1.5,
    feat_scaling: bool = True,
    lsi_skip_first: bool = True,
    harmonize: bool = False,
    batch_columns: list[str] | None = None,
    harmony_params: dict | None = None,
    local_cache: bool | str = "auto",
    update_state: bool = True,
    invalidate_cache: bool = False,
) -> ArtifactRef:
    normalized = datastore.run_normalization(
        from_assay=from_assay,
        cell_key=cell_key,
        features=features,
        log_transform=log_transform,
        renormalize_subset=renormalize_subset,
        update_state=False,
        invalidate_cache=invalidate_cache,
    )
    if reduction_method == "lsi":
        reduction = datastore.run_lsi(
            normalized,
            dims=dims,
            skip_first=lsi_skip_first,
            batch_size=batch_size,
            local_cache=local_cache,
            update_state=False,
            invalidate_cache=invalidate_cache,
        )
    elif reduction_method == "pca":
        reduction = datastore.run_pca(
            normalized,
            dims=dims,
            pca_cell_key=pca_cell_key or cell_key,
            feat_scaling=feat_scaling,
            batch_size=batch_size,
            local_cache=local_cache,
            update_state=False,
            invalidate_cache=invalidate_cache,
        )
    else:
        raise ValueError(f"Unsupported test reduction method: {reduction_method}")
    coordinates = reduction
    if harmonize:
        coordinates = datastore.run_harmony(
            batch_columns or [],
            reduction,
            harmony_params=harmony_params,
            batch_size=batch_size,
            update_state=False,
            invalidate_cache=invalidate_cache,
        )
    effective_ann_efc = ann_efc or min(100, max(k * 3, 50))
    effective_ann_ef = ann_ef or min(100, max(k * 3, 50))
    effective_ann_m = ann_m or min(max(48, int(dims * 1.5)), 64)
    ann_index = datastore.build_ann_index(
        coordinates,
        ann_metric=ann_metric,
        ann_efc=effective_ann_efc,
        ann_ef=effective_ann_ef,
        ann_m=effective_ann_m,
        ann_parallel=ann_parallel,
        rand_state=rand_state,
        batch_size=batch_size,
        update_state=False,
        invalidate_cache=invalidate_cache,
    )
    datastore.build_embedding_initialization(
        reduction,
        n_centroids=n_centroids,
        rand_state=rand_state,
        batch_size=batch_size,
        update_state=update_state,
        invalidate_cache=invalidate_cache,
    )
    neighbors = datastore.query_neighbors(
        ann_index,
        coordinates=coordinates,
        k=k,
        batch_size=batch_size,
        update_state=False,
        invalidate_cache=invalidate_cache,
    )
    connectivity = datastore.build_connectivity_map(
        neighbors,
        local_connectivity=local_connectivity,
        bandwidth=bandwidth,
        update_state=update_state,
        invalidate_cache=invalidate_cache,
    )
    return connectivity


def _extract_zarr_fixture(tar_path: str, prefix: str) -> tuple[str, str]:
    import tarfile

    temp_dir = tempfile.mkdtemp(prefix=prefix)
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(temp_dir, filter="data")
    if os.path.isfile(os.path.join(temp_dir, ".zgroup")) or os.path.isfile(
        os.path.join(temp_dir, "zarr.json")
    ):
        return temp_dir, temp_dir
    for name in sorted(os.listdir(temp_dir)):
        candidate = os.path.join(temp_dir, name)
        if os.path.isdir(candidate) and (
            os.path.isfile(os.path.join(candidate, ".zgroup"))
            or os.path.isfile(os.path.join(candidate, "zarr.json"))
        ):
            return temp_dir, candidate
    return temp_dir, temp_dir


def _datastore_tar_path() -> str:
    return full_path("1K_pbmc_citeseq.zarr.tar.gz")


def _has_graph(datastore) -> bool:
    state = datastore.get_assay_state("RNA")
    return state is not None and state.connectivity_map is not None


def _cell_has(datastore, column: str) -> bool:
    return column in datastore.cells.columns


@pytest.fixture(scope="session")
def toy_crdir_writer(toy_crdir_reader, tmp_path_factory):
    from scarf.writers import CrToZarr

    out_fn = tmp_path_factory.mktemp("toy_crdir") / "toy_crdir.zarr"
    writer = CrToZarr(toy_crdir_reader, str(out_fn))
    writer.dump()
    yield str(out_fn)


@pytest.fixture(scope="session")
def toy_crdir_ds(toy_crdir_writer):
    from scarf.datastore.datastore import DataStore

    yield DataStore(toy_crdir_writer, default_assay="RNA")


@pytest.fixture(scope="session")
def datastore_zarr_root():
    temp_dir, zarr_root = _extract_zarr_fixture(
        _datastore_tar_path(), "scarf_session_1K_pbmc_"
    )
    yield zarr_root
    remove(temp_dir)


@pytest.fixture(scope="session")
def datastore(datastore_zarr_root):
    from scarf.datastore.datastore import DataStore

    temp_dir = tempfile.mkdtemp(prefix="scarf_session_working_1K_pbmc_")
    shutil.copytree(datastore_zarr_root, temp_dir, dirs_exist_ok=True)
    yield DataStore(temp_dir, default_assay="RNA")
    remove(temp_dir)


@pytest.fixture(scope="session")
def rna_raw_total(datastore):
    return chunked_total_sum(datastore.RNA.rawData)


@pytest.fixture(scope="session")
def assay2_raw_total(datastore):
    return chunked_total_sum(datastore.assay2.rawData)


@pytest.fixture
def datastore_ephemeral(datastore_zarr_root):
    from scarf.datastore.datastore import DataStore

    temp_dir = tempfile.mkdtemp(prefix="scarf_ephemeral_1K_pbmc_")
    shutil.copytree(datastore_zarr_root, temp_dir, dirs_exist_ok=True)
    yield DataStore(temp_dir, default_assay="RNA")
    remove(temp_dir)


@pytest.fixture(scope="session")
def analyzed_datastore_zarr_root(datastore_zarr_root, tmp_path_factory):
    from scarf.datastore.datastore import DataStore

    zarr_root = tmp_path_factory.mktemp("scarf_analyzed_1K_pbmc_") / "data.zarr"
    shutil.copytree(datastore_zarr_root, zarr_root)
    datastore = DataStore(str(zarr_root), default_assay="RNA")
    datastore.auto_filter_cells(show_qc_plots=False)
    hvg_ref = datastore.mark_hvgs(
        top_n=100,
        show_plot=False,
        bin_strategy="fixed",
        min_cells=max(20, int(0.01 * datastore.cells.N)),
        max_cells=np.inf,
        blacklist="^MT-|^RPS|^RPL|^MRPS|^MRPL|^CCN|^HLA-|^H2-|^HIST",
    )
    build_neighbourhood_graph(
        datastore,
        features=hvg_ref,
        local_cache=False,
    )
    state = datastore.get_assay_state("RNA")
    assert state is not None and state.connectivity_map is not None
    return str(zarr_root)


@pytest.fixture
def analyzed_datastore_ephemeral(analyzed_datastore_zarr_root):
    from scarf.datastore.datastore import DataStore

    temp_dir = tempfile.mkdtemp(prefix="scarf_ephemeral_analyzed_1K_pbmc_")
    shutil.copytree(analyzed_datastore_zarr_root, temp_dir, dirs_exist_ok=True)
    yield DataStore(temp_dir, default_assay="RNA")
    remove(temp_dir)


@pytest.fixture(scope="session")
def auto_filter_cells(datastore):
    datastore.auto_filter_cells(show_qc_plots=False)


@pytest.fixture(scope="session")
def mark_hvgs(auto_filter_cells, datastore):
    return datastore.mark_hvgs(
        top_n=100,
        show_plot=False,
        bin_strategy="fixed",
        min_cells=max(20, int(0.01 * datastore.cells.N)),
        max_cells=np.inf,
        blacklist="^MT-|^RPS|^RPL|^MRPS|^MRPL|^CCN|^HLA-|^H2-|^HIST",
    )


@pytest.fixture(scope="session")
def detected_features(auto_filter_cells, datastore):
    del auto_filter_cells
    return datastore.select_detected_features(
        cell_key="I",
        min_cells=20,
        label="detected_features",
    )


@pytest.fixture(scope="session")
def graph_artifacts(mark_hvgs, datastore):
    build_neighbourhood_graph(datastore, features=mark_hvgs)
    state = datastore.get_assay_state("RNA")
    assert state is not None and state.neighbors is not None
    yield datastore.inspect_artifact(state.neighbors).path


@pytest.fixture(scope="session")
def leiden_clustering(graph_artifacts, datastore):
    if not _cell_has(datastore, "RNA_leiden_cluster"):
        datastore.run_leiden_clustering()
    yield datastore.cells.fetch("RNA_leiden_cluster")


@pytest.fixture(scope="session")
def legacy_leiden_clustering(graph_artifacts, datastore):
    label = "legacy_leiden_cluster"
    column = f"RNA_{label}"
    if not _cell_has(datastore, column):
        datastore.run_leiden_clustering(backend="leidenalg", label=label)
    yield datastore.cells.fetch(column)


@pytest.fixture(scope="session")
def paris_clustering(graph_artifacts, datastore):
    if not _cell_has(datastore, "RNA_cluster"):
        datastore.run_paris_clustering(n_clusters=10, label="cluster")
    yield datastore.cells.fetch("RNA_cluster")


@pytest.fixture(scope="session")
def paris_clustering_auto(graph_artifacts, datastore):
    if not _cell_has(datastore, "RNA_adaptive_clusters"):
        datastore.run_paris_clustering(
            n_clusters="auto",
            min_cluster_size=10,
            label="adaptive_clusters",
        )
    yield datastore.cells.fetch("RNA_adaptive_clusters")


@pytest.fixture(scope="session")
def umap(graph_artifacts, datastore):
    if not _cell_has(datastore, "RNA_UMAP1"):
        datastore.run_umap(n_epochs=50)
    yield np.array(
        [datastore.cells.fetch("RNA_UMAP1"), datastore.cells.fetch("RNA_UMAP2")]
    ).T


@pytest.fixture(scope="session")
def marker_search(datastore, paris_clustering, detected_features):
    return datastore.run_marker_search(
        group_key="RNA_cluster",
        features=detected_features,
    )


@pytest.fixture(scope="session")
def pseudotime_scoring(datastore, legacy_leiden_clustering):
    if not _cell_has(datastore, "RNA_pseudotime"):
        datastore.run_pseudotime_scoring(
            source_sink_key="RNA_legacy_leiden_cluster",
            sources=[6],
            sinks=[3],
        )
    yield datastore.cells.fetch("RNA_pseudotime")


@pytest.fixture(scope="session")
def pseudotime_markers(datastore, pseudotime_scoring, detected_features):
    result = datastore.run_pseudotime_marker_search(
        pseudotime_key="RNA_pseudotime",
        features=detected_features,
    )
    yield result


@pytest.fixture(scope="session")
def pseudotime_aggregation(datastore, pseudotime_scoring, detected_features):
    result = datastore.run_pseudotime_aggregation(
        features=detected_features,
        pseudotime_key="RNA_pseudotime",
        cluster_label="pseudotime_clusters",
        n_clusters=15,
        window_size=50,
        chunk_size=10,
    )
    yield result


@pytest.fixture(scope="session")
def grouped_assay(datastore, pseudotime_aggregation):
    datastore.add_grouped_assay(
        group_key="pseudotime_clusters", assay_label="PTIME_MODULES"
    )


@pytest.fixture(scope="session")
def cell_cycle_scoring(auto_filter_cells, datastore):
    del auto_filter_cells
    if not _cell_has(datastore, "RNA_cell_cycle_phase"):
        datastore.run_cell_cycle_scoring()
    return datastore.cells.fetch("RNA_cell_cycle_phase")


@pytest.fixture(scope="session")
def topacedo_sampler(paris_clustering, datastore):
    import importlib.util

    if importlib.util.find_spec("topacedo") is None:
        pytest.skip("topacedo package not installed")
    if not _cell_has(datastore, "RNA_sketched"):
        datastore.run_topacedo_sampler(cluster_key="RNA_cluster")
    return datastore.cells.fetch("RNA_sketched")


@pytest.fixture(scope="session")
def cell_attrs():
    return pd.read_csv(full_path("cell_attributes.csv"), index_col=0)


@pytest.fixture(scope="session")
def atac_datastore():
    from scarf.datastore.datastore import DataStore

    fn = full_path("500_pbmc_atac.zarr.tar.gz")
    temp_dir, zarr_root = _extract_zarr_fixture(fn, "scarf_session_atac_")
    yield DataStore(zarr_root)
    remove(temp_dir)


@pytest.fixture(scope="session")
def mark_prevalent_peaks(atac_datastore):
    return atac_datastore.mark_prevalent_peaks(top_n=5000)


@pytest.fixture(scope="session")
def make_atac_graph(mark_prevalent_peaks, atac_datastore):
    build_neighbourhood_graph(
        atac_datastore,
        features=mark_prevalent_peaks,
        reduction_method="lsi",
        feat_scaling=False,
    )
    state = atac_datastore.get_assay_state("ATAC")
    assert state is not None and state.neighbors is not None
    yield atac_datastore.inspect_artifact(state.neighbors).path
