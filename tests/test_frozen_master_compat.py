"""Hard-break migration gates against a frozen pre-1.0 master-format store.

These tests download a genuine master-format (Zarr v2) RNA store from the
public cytebase bucket. Opening that archive as RNA must fail closed and ask
for a rebuild. A repacked copy must preserve graph and marker layout while the
untouched legacy archive digest stays identical.

The corpus lives under a ``_legacy_master`` dataset. Cytebase publishes the
current-format store under the plain dataset name, and the archive it replaced
is preserved alongside it. See ``scripts/publish_docs_datasets.py``.
"""

import hashlib
import os

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
def test_frozen_master_rna_open_hard_breaks_without_strip_counts_t(
    frozen_master_store: str,
) -> None:
    from scarf.datastore.datastore import DataStore
    from scarf.storage.counts_t_contract import inspect_counts_t
    from scarf.storage.stores import load_zarr

    root = load_zarr(frozen_master_store, mode="r")
    inspected = inspect_counts_t(root, _ASSAY)
    assert inspected.status in {
        "missing",
        "zarr-v2",
        "unsupported-layout",
        "incomplete",
        "shape-dtype-mismatch",
        "missing-layout-metadata",
        "layout-mismatch",
    }

    with pytest.raises(ValueError, match="countsT|Zarr v3|Rebuild|repack"):
        DataStore(frozen_master_store, default_assay=_ASSAY, zarr_mode="r")


@pytest.mark.integration
def test_frozen_master_repack_preserves_graph_schema_and_legacy_digest(
    frozen_master_store: str,
    tmp_path,
) -> None:
    from scarf.datastore.datastore import DataStore
    from scarf.storage.counts_t_contract import inspect_counts_t
    from scarf.storage.stores import load_zarr
    from scarf.tools.repack_zarr import repack_store

    before = _tree_digest(frozen_master_store)
    repacked = str(tmp_path / "repacked.zarr")
    repack_store(frozen_master_store, repacked, nthreads=2)

    assert _tree_digest(frozen_master_store) == before

    root = load_zarr(repacked, mode="r")
    inspected = inspect_counts_t(root, _ASSAY)
    assert inspected.status == "ready"

    datastore = DataStore(repacked, default_assay=_ASSAY, zarr_mode="r")
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
    assert datastore._keys_from_knn_path(_ASSAY, _EXPECTED_KNN_LOC) == (
        _CELL_KEY,
        _FEAT_KEY,
    )

    markers = datastore.get_markers(
        from_assay=_ASSAY,
        cell_key=_CELL_KEY,
        group_key="RNA_cluster",
        group_id=1,
        min_score=0,
        min_frac_exp=0,
    )
    assert not markers.empty
    assert _tree_digest(frozen_master_store) == before
