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
    cell_selection: ArtifactRef | None = None,
    features: ArtifactRef,
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
    invalidate_cache: bool = False,
) -> ArtifactRef:
    assay_name = from_assay or datastore._defaultAssay
    cell_selection = (
        datastore.snapshot_cell_selection(cell_key)
        if cell_selection is None
        else cell_selection
    )
    feature_selection = datastore.resolve_features(assay_name, features)
    normalized = datastore.run_normalization(
        cell_selection,
        feature_selection,
        log_transform=log_transform,
        renormalize_subset=renormalize_subset,
        invalidate_cache=invalidate_cache,
    )
    pca_selection = (
        datastore.snapshot_cell_selection(pca_cell_key)
        if pca_cell_key is not None
        else None
    )
    if reduction_method == "lsi":
        reduction = datastore.run_lsi(
            normalized,
            dims=dims,
            skip_first=lsi_skip_first,
            batch_size=batch_size,
            local_cache=local_cache,
            invalidate_cache=invalidate_cache,
        )
    elif reduction_method == "pca":
        reduction = datastore.run_pca(
            normalized,
            dims=dims,
            pca_cell_selection=pca_selection,
            feat_scaling=feat_scaling,
            batch_size=batch_size,
            local_cache=local_cache,
            invalidate_cache=invalidate_cache,
        )
    else:
        raise ValueError(f"Unsupported test reduction method: {reduction_method}")
    coordinates = reduction
    if harmonize:
        coordinates = datastore.run_harmony(
            reduction,
            batch_columns or [],
            harmony_params=harmony_params,
            batch_size=batch_size,
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
        invalidate_cache=invalidate_cache,
    )
    datastore.build_embedding_initialization(
        reduction,
        n_centroids=n_centroids,
        rand_state=rand_state,
        batch_size=batch_size,
        invalidate_cache=invalidate_cache,
    )
    neighbors = datastore.query_neighbors(
        ann_index,
        coordinates=coordinates,
        k=k,
        batch_size=batch_size,
        invalidate_cache=invalidate_cache,
    )
    connectivity = datastore.build_connectivity_map(
        neighbors,
        local_connectivity=local_connectivity,
        bandwidth=bandwidth,
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


def _input_ref(datastore, ref: ArtifactRef, name: str) -> ArtifactRef:
    raw = (datastore.inspect_artifact(ref).inputs or {}).get(name)
    if not isinstance(raw, dict):
        raise AssertionError(f"{ref.kind} fixture artifact has no {name!r} input")
    return ArtifactRef.from_dict(raw)


def _graph_coordinates(datastore, graph: ArtifactRef) -> ArtifactRef:
    neighbors = _input_ref(datastore, graph, "neighbors")
    return _input_ref(datastore, neighbors, "coordinates")


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
    filtered = datastore.auto_filter_cells()
    hvg_ref = datastore.select_hvgs(
        filtered,
        top_n=100,
        show_plot=False,
        bin_strategy="fixed",
        min_cells=max(20, int(0.01 * datastore.cells.N)),
        max_cells=np.inf,
        blacklist="^MT-|^RPS|^RPL|^MRPS|^MRPL|^CCN|^HLA-|^H2-|^HIST",
    )
    graph = build_neighbourhood_graph(
        datastore,
        cell_selection=filtered,
        features=hvg_ref,
        local_cache=False,
    )
    assert graph.kind == "connectivity_map"
    assert datastore.inspect_artifact(graph).complete
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
    return datastore.auto_filter_cells()


@pytest.fixture(scope="session")
def mark_hvgs(auto_filter_cells, datastore):
    return datastore.select_hvgs(
        auto_filter_cells,
        top_n=100,
        show_plot=False,
        bin_strategy="fixed",
        min_cells=max(20, int(0.01 * datastore.cells.N)),
        max_cells=np.inf,
        blacklist="^MT-|^RPS|^RPL|^MRPS|^MRPL|^CCN|^HLA-|^H2-|^HIST",
    )


@pytest.fixture(scope="session")
def detected_features(auto_filter_cells, datastore):
    return datastore.select_detected_features(
        auto_filter_cells,
        min_cells=20,
    )


@pytest.fixture(scope="session")
def connectivity_graph(auto_filter_cells, mark_hvgs, datastore):
    return build_neighbourhood_graph(
        datastore,
        cell_selection=auto_filter_cells,
        features=mark_hvgs,
    )


@pytest.fixture(scope="session")
def graph_artifacts(connectivity_graph, datastore):
    neighbors = _input_ref(datastore, connectivity_graph, "neighbors")
    yield datastore.inspect_artifact(neighbors).path


@pytest.fixture(scope="session")
def leiden_clustering(connectivity_graph, datastore):
    yield datastore.run_leiden_clustering(connectivity_graph)


@pytest.fixture(scope="session")
def legacy_leiden_clustering(connectivity_graph, datastore):
    yield datastore.run_leiden_clustering(
        connectivity_graph,
        backend="leidenalg",
    )


@pytest.fixture(scope="session")
def paris_clustering(connectivity_graph, datastore):
    yield datastore.run_paris_clustering(
        connectivity_graph,
        n_clusters=10,
    )


@pytest.fixture(scope="session")
def paris_clustering_auto(connectivity_graph, datastore):
    yield datastore.run_paris_clustering(
        connectivity_graph,
        n_clusters="auto",
        min_cluster_size=10,
    )


@pytest.fixture(scope="session")
def umap(connectivity_graph, datastore):
    initialization = datastore.build_embedding_initialization(
        _graph_coordinates(datastore, connectivity_graph)
    )
    yield datastore.run_umap(
        connectivity_graph,
        initialization,
        n_epochs=50,
    )


@pytest.fixture(scope="session")
def marker_search(datastore, paris_clustering, detected_features):
    return datastore.run_marker_search(
        paris_clustering,
        features=detected_features,
    )


@pytest.fixture(scope="session")
def pseudotime_scoring(datastore, connectivity_graph, legacy_leiden_clustering):
    yield datastore.run_pseudotime_scoring(
        connectivity_graph,
        source_sink=legacy_leiden_clustering,
        sources=[6],
        sinks=[3],
    )


@pytest.fixture(scope="session")
def pseudotime_markers(datastore, pseudotime_scoring, detected_features):
    result = datastore.run_pseudotime_marker_search(
        pseudotime_scoring,
        features=detected_features,
    )
    yield result


@pytest.fixture(scope="session")
def pseudotime_aggregation(datastore, pseudotime_scoring, detected_features):
    result = datastore.run_pseudotime_aggregation(
        pseudotime_scoring,
        features=detected_features,
        n_clusters=15,
        window_size=50,
        chunk_size=10,
    )
    yield result


@pytest.fixture(scope="session")
def grouped_assay(datastore, pseudotime_aggregation):
    datastore.add_grouped_assay(
        pseudotime_aggregation,
        assay_label="PTIME_MODULES",
    )
    return pseudotime_aggregation


@pytest.fixture(scope="session")
def cell_cycle_scoring(auto_filter_cells, datastore):
    return datastore.run_cell_cycle_scoring(auto_filter_cells)


@pytest.fixture(scope="session")
def topacedo_sampler(paris_clustering, connectivity_graph, datastore):
    import importlib.util

    if importlib.util.find_spec("topacedo") is None:
        pytest.skip("topacedo package not installed")
    return datastore.run_topacedo_sampler(
        connectivity_graph,
        paris_clustering,
    )


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
    return atac_datastore.select_prevalent_peaks(
        atac_datastore.snapshot_cell_selection(),
        top_n=5000,
    )


@pytest.fixture(scope="session")
def atac_connectivity_graph(mark_prevalent_peaks, atac_datastore):
    return build_neighbourhood_graph(
        atac_datastore,
        features=mark_prevalent_peaks,
        reduction_method="lsi",
        feat_scaling=False,
    )


@pytest.fixture(scope="session")
def make_atac_graph(atac_connectivity_graph, atac_datastore):
    neighbors = _input_ref(atac_datastore, atac_connectivity_graph, "neighbors")
    yield atac_datastore.inspect_artifact(neighbors).path
