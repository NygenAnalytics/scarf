import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import pytest

from . import full_path, remove, dask_total_sum


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
    analyzed = full_path("1K_pbmc_citeseq_analyzed.zarr.tar.gz")
    if os.path.isfile(analyzed):
        return analyzed
    return full_path("1K_pbmc_citeseq.zarr.tar.gz")


def _has_graph(datastore) -> bool:
    try:
        datastore._get_latest_graph_loc(from_assay="RNA", cell_key="I", feat_key="hvgs")
        return True
    except KeyError:
        return False


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

    yield DataStore(datastore_zarr_root, default_assay="RNA")


@pytest.fixture(scope="session")
def rna_raw_total(datastore):
    return dask_total_sum(datastore.RNA.rawData)


@pytest.fixture(scope="session")
def assay2_raw_total(datastore):
    return dask_total_sum(datastore.assay2.rawData)


@pytest.fixture
def datastore_ephemeral(datastore_zarr_root):
    from scarf.datastore.datastore import DataStore

    temp_dir = tempfile.mkdtemp(prefix="scarf_ephemeral_1K_pbmc_")
    shutil.copytree(datastore_zarr_root, temp_dir, dirs_exist_ok=True)
    yield DataStore(temp_dir, default_assay="RNA")
    remove(temp_dir)


@pytest.fixture(scope="session")
def auto_filter_cells(datastore):
    if not _has_graph(datastore):
        datastore.auto_filter_cells(show_qc_plots=False)


@pytest.fixture(scope="session")
def mark_hvgs(auto_filter_cells, datastore):
    if not _has_graph(datastore):
        datastore.mark_hvgs(top_n=100, show_plot=False)


@pytest.fixture(scope="session")
def make_graph(mark_hvgs, datastore):
    if not _has_graph(datastore):
        datastore.make_graph(feat_key="hvgs")
    graph_loc = datastore._get_latest_graph_loc(
        from_assay="RNA", cell_key="I", feat_key="hvgs"
    )
    yield graph_loc.rsplit("/", 1)[0]


@pytest.fixture(scope="session")
def leiden_clustering(make_graph, datastore):
    if not _cell_has(datastore, "RNA_leiden_cluster"):
        datastore.run_leiden_clustering()
    yield datastore.cells.fetch("RNA_leiden_cluster")


@pytest.fixture(scope="session")
def paris_clustering(make_graph, datastore):
    if not _cell_has(datastore, "RNA_cluster"):
        datastore.run_clustering(n_clusters=10)
    yield datastore.cells.fetch("RNA_cluster")


@pytest.fixture(scope="session")
def paris_clustering_balanced(make_graph, datastore):
    if not _cell_has(datastore, "RNA_balanced_clusters"):
        datastore.run_clustering(
            balanced_cut=True, max_size=100, min_size=10, label="balanced_clusters"
        )
    yield datastore.cells.fetch("RNA_balanced_clusters")


@pytest.fixture(scope="session")
def umap(make_graph, datastore):
    if not _cell_has(datastore, "RNA_UMAP1"):
        datastore.run_umap(n_epochs=50)
    yield np.array(
        [datastore.cells.fetch("RNA_UMAP1"), datastore.cells.fetch("RNA_UMAP2")]
    ).T


@pytest.fixture(scope="session")
def marker_search(datastore, paris_clustering):
    if (
        "markers" not in datastore.z["RNA"]
        or "I__RNA_cluster" not in datastore.z["RNA"]["markers"]
    ):
        datastore.run_marker_search(group_key="RNA_cluster")


@pytest.fixture(scope="session")
def pseudotime_scoring(datastore, leiden_clustering):
    if not _cell_has(datastore, "RNA_pseudotime"):
        datastore.run_pseudotime_scoring(
            source_sink_key="RNA_leiden_cluster", sources=[6], sinks=[3]
        )
    yield datastore.cells.fetch("RNA_pseudotime")


@pytest.fixture(scope="session")
def pseudotime_markers(datastore, pseudotime_scoring):
    if "I__RNA_pseudotime__r" not in datastore.RNA.feats.columns:
        datastore.run_pseudotime_marker_search(pseudotime_key="RNA_pseudotime")
    df = datastore.RNA.feats.to_pandas_dataframe(
        ["names", "I__RNA_pseudotime__r"], key="I"
    )
    yield df


@pytest.fixture(scope="session")
def pseudotime_aggregation(datastore, pseudotime_scoring):
    result = datastore.run_pseudotime_aggregation(
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
def run_mapping(make_graph, datastore):
    projections = datastore.z["RNA"].get("projections", None)
    if projections is None or "selfmap" not in projections:
        datastore.run_mapping(
            target_assay=datastore.RNA,
            target_name="selfmap",
            target_feat_key="hvgs_self",
            save_k=3,
        )


@pytest.fixture(scope="session")
def run_mapping_coral(make_graph, datastore):
    projections = datastore.z["RNA"].get("projections", None)
    if projections is None or "selfmap_coral" not in projections:
        datastore.run_mapping(
            target_assay=datastore.RNA,
            target_name="selfmap_coral",
            target_feat_key="hvgs_self2",
            save_k=3,
            run_coral=True,
        )


@pytest.fixture(scope="session")
def run_unified_umap(run_mapping, datastore):
    projections = datastore.z["RNA"].get("projections", None)
    if projections is None or "unified_UMAP" not in projections:
        datastore.run_unified_umap(target_names=["selfmap"])


@pytest.fixture(scope="session")
def cell_cycle_scoring(datastore):
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
    atac_datastore.mark_prevalent_peaks(top_n=5000)


@pytest.fixture(scope="session")
def make_atac_graph(mark_prevalent_peaks, atac_datastore):
    atac_datastore.make_graph(feat_key="prevalent_peaks")
    graph_loc = atac_datastore._get_latest_graph_loc(
        from_assay="ATAC", cell_key="I", feat_key="prevalent_peaks"
    )
    yield graph_loc.rsplit("/", 1)[0]
