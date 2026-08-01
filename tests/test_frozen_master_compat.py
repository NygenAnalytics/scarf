"""Master-format compatibility against a real frozen Scarf store.

These tests download a genuine pre-1.0 (master-format, Zarr v2) store from the
public cytebase bucket instead of hand-building synthetic hybrids. They pin the
read paths that a released store must keep working: encoded graph-chain lookup,
graph loading, initial embedding, and Paris recompute from a legacy dendrogram.
They also assert that a read-only open does not mutate the store on disk.
They are marked ``integration`` because they require network access, but the
project's default test command includes integration tests.

The corpus lives under a ``_legacy_master`` dataset. Cytebase publishes the
current-format store under the plain dataset name, and the archive it replaced
is preserved alongside it so this frozen layout stays reachable. See
``scripts/publish_docs_datasets.py``.
"""

import hashlib
import os
import shutil

import numpy as np
import pytest

_FROZEN_DATASET = "tenx_5K_pbmc_rnaseq_legacy_master"
_ASSAY = "RNA"
_CELL_KEY = "I"
_FEAT_KEY = "hvgs"
_EXPECTED_GRAPH_LOC = (
    "RNA/normed__I__hvgs/reduction__pca__15__I/"
    "ann__l2__50__50__48__4466/knn__11/graph__1.0__1.5"
)
_EXPECTED_KNN_LOC = _EXPECTED_GRAPH_LOC.rsplit("/", 1)[0]
_EXPECTED_NORMED_LOC = "RNA/normed__I__hvgs"


def _tree_digest(root: str) -> str:
    """Order-independent digest of every file path, size, and content."""
    digest = hashlib.blake2b(digest_size=32)
    for dirpath, _dirnames, filenames in sorted(os.walk(root)):
        for name in sorted(filenames):
            path = os.path.join(dirpath, name)
            rel = os.path.relpath(path, root)
            with open(path, "rb") as handle:
                payload = handle.read()
            digest.update(rel.encode("utf-8"))
            digest.update(len(payload).to_bytes(8, "little"))
            digest.update(payload)
    return digest.hexdigest()


def _array_digest(array: np.ndarray) -> str:
    values = np.asarray(array)
    digest = hashlib.blake2b(digest_size=32)
    digest.update(str(values.shape).encode())
    digest.update(str(values.dtype).encode())
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


@pytest.fixture(scope="module")
def frozen_master_store(tmp_path_factory) -> str:
    from scarf import cytebase

    dest = tmp_path_factory.mktemp("frozen_master")
    repository = cytebase.connect("scarf_docs")
    dataset_path = repository.download_dataset(_FROZEN_DATASET, dest, zarr=True)
    store_path = os.path.join(str(dataset_path), "data.zarr")
    assert os.path.isdir(store_path)
    # A master-format store is Zarr v2 (uses .zgroup, not zarr.json).
    assert os.path.isfile(os.path.join(store_path, ".zgroup"))
    return store_path


@pytest.mark.integration
def test_frozen_master_store_is_v2_master_layout(frozen_master_store: str) -> None:
    # The legacy graph chain and per-cluster markers are the master layout that
    # later phases must not silently break.
    graph_leaf = os.path.join(
        frozen_master_store,
        "RNA/normed__I__hvgs/reduction__pca__15__I/"
        "ann__l2__50__50__48__4466/knn__11/graph__1.0__1.5",
    )
    assert os.path.isdir(graph_leaf)
    dendrogram = os.path.join(graph_leaf, "dendrogram")
    assert os.path.isdir(dendrogram)
    markers = os.path.join(frozen_master_store, "RNA/markers/I__RNA_cluster/1")
    assert os.path.isdir(markers)


@pytest.mark.integration
def test_frozen_master_graph_lookup_and_load(frozen_master_store: str) -> None:
    from scarf.datastore.datastore import DataStore
    from scarf.graph.state import validate_legacy_graph_selection

    datastore = DataStore(frozen_master_store, default_assay=_ASSAY, zarr_mode="r")

    assert (
        datastore.get_latest_graph_loc(_ASSAY, _CELL_KEY, _FEAT_KEY)
        == _EXPECTED_GRAPH_LOC
    )
    assert (
        datastore.get_normalized_group_path(_ASSAY, _CELL_KEY, _FEAT_KEY)
        == _EXPECTED_NORMED_LOC
    )
    stored = datastore._lookup_stored_graph(_ASSAY, _CELL_KEY, _FEAT_KEY)
    assert stored.paths.cell_graph_group_path == _EXPECTED_GRAPH_LOC
    assert stored.ann_metric == "l2"
    assert stored.k == 11

    latest_graph = datastore.load_graph(
        from_assay=_ASSAY, cell_key=_CELL_KEY, feat_key=_FEAT_KEY
    )
    explicit_graph = datastore.load_graph(graph_loc=_EXPECTED_GRAPH_LOC)
    assert latest_graph.shape == (3940, 3940)
    assert latest_graph.nnz == 43340
    assert (latest_graph != explicit_graph).nnz == 0
    ann_stream = datastore._load_ann_stream(
        _ASSAY,
        _CELL_KEY,
        _FEAT_KEY,
        knn_loc=_EXPECTED_KNN_LOC,
    )
    assert ann_stream.k == 11
    assert ann_stream.annIdx is not None
    validate_legacy_graph_selection(
        datastore,
        _EXPECTED_KNN_LOC,
        _ASSAY,
        _CELL_KEY,
        _FEAT_KEY,
    )
    assert datastore._keys_from_knn_path(_ASSAY, _EXPECTED_KNN_LOC) == (
        _CELL_KEY,
        _FEAT_KEY,
    )
    graph_group = datastore.zw[_EXPECTED_GRAPH_LOC]
    assert (
        _array_digest(graph_group["edges"][:])
        == "33e3daf56516e5ea4869bf312e2033891b0cdd78232cfd8232addd07b863b36a"
    )
    assert (
        _array_digest(graph_group["weights"][:])
        == "a29ccc26aac9d7024327b824afb3f8ca0c030c3ecf5428b6ef3970ba6ff9f06a"
    )
    assert (
        _array_digest(graph_group["dendrogram"][:])
        == "f30c9ba746100c1217ee1b093c6cc71775e7f2eff0e3f0d06b73ab21257b2236"
    )


@pytest.mark.integration
def test_frozen_master_initial_embedding_is_finite(
    frozen_master_store: str,
) -> None:
    from scarf.datastore.datastore import DataStore

    datastore = DataStore(frozen_master_store, default_assay=_ASSAY, zarr_mode="r")
    graph = datastore.load_graph(
        from_assay=_ASSAY, cell_key=_CELL_KEY, feat_key=_FEAT_KEY
    )
    ini_embed = datastore._get_ini_embed(_ASSAY, _CELL_KEY, _FEAT_KEY, 2)
    assert ini_embed.shape == (graph.shape[0], 2)
    assert np.isfinite(ini_embed).all()


@pytest.mark.integration
def test_frozen_master_read_only_open_does_not_mutate(
    frozen_master_store: str,
) -> None:
    from scarf.datastore.datastore import DataStore

    before = _tree_digest(frozen_master_store)
    datastore = DataStore(frozen_master_store, default_assay=_ASSAY, zarr_mode="r")
    datastore.get_normalized_group_path(_ASSAY, _CELL_KEY, _FEAT_KEY)
    datastore.load_graph(graph_loc=_EXPECTED_GRAPH_LOC)
    datastore._lookup_stored_graph(_ASSAY, _CELL_KEY, _FEAT_KEY)
    datastore._load_ann_stream(
        _ASSAY,
        _CELL_KEY,
        _FEAT_KEY,
        knn_loc=_EXPECTED_KNN_LOC,
    )
    datastore._get_ini_embed(_ASSAY, _CELL_KEY, _FEAT_KEY, 2)
    after = _tree_digest(frozen_master_store)
    assert before == after


@pytest.mark.integration
def test_frozen_master_reads_and_recomputes_markers_without_mutation(
    frozen_master_store: str,
) -> None:
    from scarf.datastore.datastore import DataStore

    before = _tree_digest(frozen_master_store)
    datastore = DataStore(frozen_master_store, default_assay=_ASSAY, zarr_mode="r")
    stored = datastore.get_markers(
        from_assay=_ASSAY,
        cell_key=_CELL_KEY,
        group_key="RNA_cluster",
        group_id=1,
        min_score=0,
        min_frac_exp=0,
    )
    computed = datastore.run_marker_search(
        from_assay=_ASSAY,
        cell_key=_CELL_KEY,
        feat_key="I__hvgs",
        group_key="RNA_cluster",
        gene_batch_size=100,
        n_threads=1,
        skip_save=True,
    )

    assert not stored.empty
    assert computed
    assert all(frame.shape[0] > 0 for frame in computed.values())
    assert _tree_digest(frozen_master_store) == before


@pytest.mark.integration
def test_frozen_master_writable_copy_publishes_canonical_markers_additively(
    frozen_master_store: str,
    tmp_path,
) -> None:
    from scarf.datastore.datastore import DataStore
    from scarf.features.markers.table import MARKER_STAT_COLUMNS

    working_copy = str(tmp_path / "data.zarr")
    shutil.copytree(frozen_master_store, working_copy)
    legacy_subtree = os.path.join(
        working_copy,
        "RNA/markers/I__RNA_cluster",
    )
    legacy_before = _tree_digest(legacy_subtree)
    datastore = DataStore(working_copy, default_assay=_ASSAY, zarr_mode="r+")

    datastore.run_marker_search(
        from_assay=_ASSAY,
        cell_key=_CELL_KEY,
        feat_key="I__hvgs",
        group_key="RNA_cluster",
        gene_batch_size=100,
        n_threads=1,
    )

    assert _tree_digest(legacy_subtree) == legacy_before
    published = datastore._resolve_marker_group(
        _ASSAY,
        _CELL_KEY,
        "RNA_cluster",
    )
    assert "schema_version" not in published.attrs
    assert list(published.attrs["stat_columns"]) == list(MARKER_STAT_COLUMNS)
    assert published.attrs["method"] == "mannwhitneyu"
    assert published.attrs["alternative"] == "two-sided"
    assert published.attrs["adjustment_method"] == "fdr_bh"
    assert published.attrs["adjustment_scope"] == ("within_group_all_tested_features")
    assert all(
        published[group].attrs["n_group"] >= 2
        and published[group].attrs["n_reference"] >= 2
        for group in published.group_keys()
    )


@pytest.mark.integration
def test_frozen_master_legacy_dendrogram_triggers_hierarchy_rebuild(
    frozen_master_store: str,
    tmp_path,
) -> None:
    from scarf.datastore.datastore import DataStore
    from scarf.storage.artifacts import list_artifacts

    working_copy = str(tmp_path / "data.zarr")
    shutil.copytree(frozen_master_store, working_copy)

    # The legacy dendrogram signals that the current hierarchy must be refit
    # from the stored graph. Its values are not reused as a current generation.
    hierarchy_root = f"{_EXPECTED_GRAPH_LOC}/paris_hierarchy"
    datastore = DataStore(working_copy, default_assay=_ASSAY, zarr_mode="r+")
    assert hierarchy_root not in datastore.zw

    result = datastore.run_paris_clustering(
        from_assay=_ASSAY,
        cell_key=_CELL_KEY,
        feat_key=_FEAT_KEY,
        n_clusters=8,
        label="frozen_paris",
    )

    labels = datastore.cells.fetch("RNA_frozen_paris")
    assert len(set(labels.tolist())) == 8
    np.testing.assert_array_equal(labels, result.labels)
    # The released graph remains untouched; the new hierarchy is independent.
    assert hierarchy_root not in datastore.zw
    hierarchies = list_artifacts(
        datastore.zw,
        scope="assay",
        assay=_ASSAY,
        kind="cluster_hierarchy",
    )
    assert result.hierarchy_generation_id in {ref.artifact_id for ref in hierarchies}
