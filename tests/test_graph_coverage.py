import shutil
import sys
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import zarr
from scipy.sparse import csr_matrix
from zarr.storage import MemoryStore

from scarf.assay import ATACassay
from scarf.datastore.datastore import DataStore
from scarf.datastore.graph_datastore import GraphDataStore


class _MemoryGraphStore(GraphDataStore):
    @property
    def assay_names(self) -> list[str]:
        return self._assay_names


@pytest.fixture
def isolated_toy_datastore(toy_crdir_writer: str, tmp_path: Path) -> DataStore:
    zarr_path = tmp_path / "toy.zarr"
    shutil.copytree(toy_crdir_writer, zarr_path)
    return DataStore(
        str(zarr_path),
        default_assay="RNA",
        min_features_per_cell=0,
        min_cells_per_feature=0,
        nthreads=1,
    )


def _memory_graph_store(
    assay_names: list[str] | None = None,
) -> _MemoryGraphStore:
    store = _MemoryGraphStore.__new__(_MemoryGraphStore)
    store.z = zarr.open_group(store=MemoryStore(), mode="w")
    store.workspace = None
    store.zarr_mode = "r+"
    store._defaultAssay = "RNA"
    store._assay_names = assay_names or []
    store._integratedGraphsLoc = "integratedGraphs"
    store._cachedMagicOperator = None
    store._cachedMagicOperatorLoc = None
    store.nthreads = 1
    return store


def _add_test_graph(store: _MemoryGraphStore, label: str = "graph") -> str:
    graph_loc = f"integratedGraphs/{label}"
    graph_group = store.zw.create_group(graph_loc)
    graph_group.attrs["n_cells"] = 3
    graph_group.attrs["n_neighbors"] = 2
    graph_group.create_array(
        "edges",
        data=np.array(
            [
                [0, 1],
                [0, 2],
                [1, 0],
                [1, 2],
                [2, 0],
                [2, 1],
            ],
            dtype=np.uint64,
        ),
    )
    graph_group.create_array(
        "weights",
        data=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
    )
    return graph_loc


@pytest.mark.parametrize(
    ("symmetric", "upper_only", "use_k", "expected"),
    [
        (
            False,
            False,
            None,
            np.array(
                [
                    [0.0, 0.1, 0.2],
                    [0.3, 0.0, 0.4],
                    [0.5, 0.6, 0.0],
                ]
            ),
        ),
        (
            False,
            True,
            1,
            np.array(
                [
                    [0.0, 0.1, 0.0],
                    [0.3, 0.0, 0.0],
                    [0.5, 0.0, 0.0],
                ]
            ),
        ),
        (
            False,
            False,
            0,
            np.array(
                [
                    [0.0, 0.1, 0.0],
                    [0.3, 0.0, 0.0],
                    [0.5, 0.0, 0.0],
                ]
            ),
        ),
        (
            False,
            False,
            99,
            np.array(
                [
                    [0.0, 0.1, 0.2],
                    [0.3, 0.0, 0.4],
                    [0.5, 0.6, 0.0],
                ]
            ),
        ),
        (
            True,
            False,
            None,
            np.array(
                [
                    [0.0, 0.37, 0.6],
                    [0.37, 0.0, 0.76],
                    [0.6, 0.76, 0.0],
                ]
            ),
        ),
        (
            True,
            True,
            None,
            np.array(
                [
                    [0.0, 0.37, 0.6],
                    [0.0, 0.0, 0.76],
                    [0.0, 0.0, 0.0],
                ]
            ),
        ),
    ],
)
def test_load_graph_option_matrix(
    symmetric: bool,
    upper_only: bool,
    use_k: int | None,
    expected: np.ndarray,
) -> None:
    store = _memory_graph_store()
    graph_loc = _add_test_graph(store)

    graph = store.load_graph(
        from_assay="RNA",
        cell_key="I",
        feat_key="I",
        symmetric=symmetric,
        upper_only=upper_only,
        use_k=use_k,
        graph_loc=graph_loc,
    )

    np.testing.assert_allclose(graph.toarray(), expected)


def test_load_graph_latest_location_formats_and_errors() -> None:
    store = _memory_graph_store()
    graph_loc = _add_test_graph(store)
    normed_loc = "RNA/normed__I__I"
    reduction_loc = f"{normed_loc}/reduction__pca__2__I"
    ann_loc = f"{reduction_loc}/ann__l2__50__50__48__1"
    knn_loc = f"{ann_loc}/knn__2"

    normed_group = store.zw.create_group(normed_loc)
    reduction_group = store.zw.create_group(reduction_loc)
    ann_group = store.zw.create_group(ann_loc)
    knn_group = store.zw.create_group(knn_loc)
    normed_group.attrs["latest_reduction"] = reduction_loc
    reduction_group.attrs["latest_ann"] = ann_loc
    ann_group.attrs["latest_knn"] = knn_loc
    knn_group.attrs["latest_graph"] = graph_loc

    store._get_latest_cell_key = Mock(return_value="I")
    store._get_latest_feat_key = Mock(return_value="I")
    graph = store.load_graph()
    np.testing.assert_allclose(
        graph.toarray(),
        np.array(
            [
                [0.0, 0.1, 0.2],
                [0.3, 0.0, 0.4],
                [0.5, 0.6, 0.0],
            ]
        ),
    )
    store._get_latest_cell_key.assert_called_once_with("RNA")
    store._get_latest_feat_key.assert_called_once_with("RNA")

    n_cells, graph_coo = store._store_to_sparse(graph_loc, sparse_format="coo", use_k=1)
    assert n_cells == 3
    assert graph_coo.getformat() == "coo"
    assert graph_coo.nnz == 3

    with pytest.raises(ValueError, match="not found in zarr location"):
        store.load_graph(
            from_assay="RNA",
            cell_key="I",
            feat_key="I",
            graph_loc="integratedGraphs/missing",
        )

    empty_store = _memory_graph_store()
    with pytest.raises(KeyError):
        empty_store.load_graph(
            from_assay="RNA",
            cell_key="I",
            feat_key="I",
        )


def test_set_graph_params_defaults_and_reduction_validation(
    isolated_toy_datastore: DataStore,
) -> None:
    store = isolated_toy_datastore

    params = store._set_graph_params(
        from_assay="RNA",
        cell_key="I",
        feat_key="coverageFresh",
        reduction_method="AUTO",
    )
    assert params == (
        True,
        True,
        "pca",
        11,
        "I",
        "l2",
        50,
        50,
        48,
        4466,
        11,
        1000,
        1.0,
        1.5,
    )

    assert store._choose_reduction_method(store.RNA, "PCA") == "pca"
    atac_assay = ATACassay.__new__(ATACassay)
    assert store._choose_reduction_method(atac_assay, "auto") == "lsi"
    assert store._choose_reduction_method(store.RNA, "custom") == "custom"

    with pytest.raises(ValueError, match="Please choose"):
        store._choose_reduction_method(store.RNA, "invalid")

    with pytest.raises(ValueError, match="does not exist"):
        store._set_graph_params(
            from_assay="RNA",
            cell_key="I",
            feat_key="coverageFresh",
            reduction_method="pca",
            pca_cell_key="missing",
        )

    with pytest.raises(TypeError, match="should be `bool`"):
        store._set_graph_params(
            from_assay="RNA",
            cell_key="I",
            feat_key="coverageFresh",
            reduction_method="pca",
            pca_cell_key="ids",
        )


def test_set_graph_params_reuses_cached_hierarchy(
    isolated_toy_datastore: DataStore,
) -> None:
    store = isolated_toy_datastore
    normed_loc = "RNA/normed__I__coverageCached"
    reduction_loc = f"{normed_loc}/reduction__pca__7__I"
    ann_loc = f"{reduction_loc}/ann__cosine__70__60__32__123"
    knn_loc = f"{ann_loc}/knn__5"
    kmeans_loc = f"{reduction_loc}/kmeans__9__123"
    graph_loc = f"{knn_loc}/graph__2.0__3.0"

    normed_group = store.zw.create_group(normed_loc)
    reduction_group = store.zw.create_group(reduction_loc)
    ann_group = store.zw.create_group(ann_loc)
    knn_group = store.zw.create_group(knn_loc)
    store.zw.create_group(kmeans_loc)
    store.zw.create_group(graph_loc)
    normed_group.attrs["subset_params"] = {
        "log_transform": False,
        "renormalize_subset": False,
    }
    normed_group.attrs["latest_reduction"] = reduction_loc
    reduction_group.attrs["latest_ann"] = ann_loc
    reduction_group.attrs["latest_kmeans"] = kmeans_loc
    ann_group.attrs["latest_knn"] = knn_loc
    knn_group.attrs["latest_graph"] = graph_loc

    params = store._set_graph_params(
        from_assay="RNA",
        cell_key="I",
        feat_key="coverageCached",
        reduction_method="pca",
    )

    assert params == (
        False,
        False,
        "pca",
        7,
        "I",
        "cosine",
        70,
        60,
        32,
        123,
        5,
        9,
        2.0,
        3.0,
    )


def test_set_graph_params_uses_missing_metadata_fallbacks(
    isolated_toy_datastore: DataStore,
) -> None:
    store = isolated_toy_datastore
    empty_normed_loc = "RNA/normed__I__coverageEmpty"
    store.zw.create_group(empty_normed_loc)

    defaults = store._set_graph_params(
        from_assay="RNA",
        cell_key="I",
        feat_key="coverageEmpty",
        reduction_method="pca",
    )
    assert defaults == (
        True,
        True,
        "pca",
        11,
        "I",
        "l2",
        50,
        50,
        48,
        4466,
        11,
        1000,
        1.0,
        1.5,
    )

    normed_loc = "RNA/normed__I__coveragePartial"
    reduction_loc = f"{normed_loc}/reduction__pca__6__I"
    ann_loc = f"{reduction_loc}/ann__l2__50__50__48__4466"
    knn_loc = f"{ann_loc}/knn__11"
    normed_group = store.zw.create_group(normed_loc)
    normed_group.attrs["subset_params"] = {
        "log_transform": True,
        "renormalize_subset": True,
    }
    normed_group.attrs["latest_reduction"] = reduction_loc
    store.zw.create_group(reduction_loc)
    store.zw.create_group(knn_loc)

    fallbacks = store._set_graph_params(
        from_assay="RNA",
        cell_key="I",
        feat_key="coveragePartial",
        reduction_method="pca",
    )
    assert fallbacks == (
        True,
        True,
        "pca",
        6,
        "I",
        "l2",
        50,
        50,
        48,
        4466,
        11,
        1000,
        1.0,
        1.5,
    )


def test_latest_knn_and_ann_stream_cache_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _memory_graph_store(["RNA"])
    assay_group = store.zw.create_group("RNA")
    assay_group.attrs["latest_cell_key"] = "I"
    assay_group.attrs["latest_feat_key"] = "I"
    normed_loc = "RNA/normed__I__I"
    reduction_loc = f"{normed_loc}/reduction__pca__2__I"
    ann_loc = f"{reduction_loc}/ann__l2__50__50__48__1"
    knn_loc = f"{ann_loc}/knn__2"
    normed_group = store.zw.create_group(normed_loc)
    reduction_group = store.zw.create_group(reduction_loc)
    ann_group = store.zw.create_group(ann_loc)
    store.zw.create_group(knn_loc)
    normed_group.attrs["latest_reduction"] = reduction_loc
    reduction_group.attrs["latest_ann"] = ann_loc
    ann_group.attrs["latest_knn"] = knn_loc
    reduction_group.create_array("reduction", data=np.eye(2))
    normed_group.create_array("data", data=np.eye(3, 2))
    store._load_default_assay = Mock(return_value="RNA")

    assert store._get_latest_knn_loc() == knn_loc
    store._load_default_assay.assert_called_once_with()

    with pytest.raises(ValueError, match="Assay missing does not exist"):
        store._get_latest_knn_loc("missing")

    del reduction_group["reduction"]
    with pytest.raises(ValueError, match="PCA Reduction not found"):
        store._get_latest_knn_loc("RNA")
    reduction_group.create_array("reduction", data=np.eye(2))

    assert store._has_ann_stream_cache("RNA", "I", "I") is True
    assert store._has_ann_stream_cache("RNA", "I", "I", knn_loc=knn_loc) is True
    assert store._has_ann_stream_cache("RNA", "I", "missing") is False
    assert (
        store._has_ann_stream_cache("RNA", "I", "I", knn_loc=f"{ann_loc}/knn__99")
        is False
    )

    store.zw.create_group("RNA/normed__I__broken")
    assert store._has_ann_stream_cache("RNA", "I", "broken") is False

    legacy_path = tmp_path / "ann_idx"
    legacy_path.write_bytes(b"index")
    monkeypatch.setattr(
        "scarf.datastore.graph_datastore.legacy_ann_index_path",
        lambda *_: str(legacy_path),
    )
    assert (
        store._ann_stream_recoverable(
            "missing/ann", "missing/reduction", "missing/normed"
        )
        is True
    )

    monkeypatch.setattr(
        "scarf.datastore.graph_datastore.legacy_ann_index_path",
        lambda *_: None,
    )
    assert (
        store._ann_stream_recoverable(
            "missing/ann", "missing/reduction", "missing/normed"
        )
        is False
    )


def test_ann_index_resolution_and_persistence_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _memory_graph_store()
    ann_loc = "RNA/normed/reduction/ann__l2__50__50__48__1"
    ann_group = store.zw.create_group(ann_loc)
    ann_group.create_array("ann_idx_bytes", data=np.array([1, 2, 3], dtype=np.uint8))
    stored_index = object()
    load_stored = Mock(return_value=stored_index)
    monkeypatch.setattr("scarf.datastore.graph_datastore.load_ann_index", load_stored)
    monkeypatch.setattr(
        "scarf.datastore.graph_datastore.legacy_ann_index_path",
        lambda *_: None,
    )

    assert store._resolve_ann_index(ann_loc, "l2", 3) is stored_index
    load_stored.assert_called_once_with(ann_group, "l2", 3)

    del ann_group["ann_idx_bytes"]
    custom_path = tmp_path / "custom.idx"
    legacy_path = tmp_path / "legacy.idx"
    custom_path.write_bytes(b"custom")
    legacy_path.write_bytes(b"legacy")
    custom_index = object()
    legacy_index = object()
    load_path = Mock(
        side_effect=lambda path, *_: (
            custom_index if path == str(custom_path) else legacy_index
        )
    )
    monkeypatch.setattr(
        "scarf.datastore.graph_datastore.load_ann_index_from_path",
        load_path,
    )

    assert (
        store._resolve_ann_index(
            ann_loc,
            "cosine",
            4,
            ann_index_fetcher=lambda _: str(custom_path),
        )
        is custom_index
    )

    failing_fetcher = Mock(side_effect=RuntimeError("fetch failed"))
    assert (
        store._resolve_ann_index(
            ann_loc,
            "l2",
            3,
            ann_index_fetcher=failing_fetcher,
        )
        is None
    )

    save_index = Mock()
    monkeypatch.setattr("scarf.datastore.graph_datastore.save_ann_index", save_index)
    monkeypatch.setattr(
        "scarf.datastore.graph_datastore.legacy_ann_index_path",
        lambda *_: str(legacy_path),
    )
    assert store._resolve_ann_index(ann_loc, "l2", 3) is legacy_index
    save_index.assert_called_once_with(ann_group, legacy_index)

    custom_saver = Mock()
    custom_only_loc = "custom/ann"
    store._persist_ann_index(
        custom_only_loc, custom_index, ann_index_saver=custom_saver
    )
    custom_saver.assert_called_once_with(custom_index, custom_only_loc)
    assert custom_only_loc not in store.zw

    fallback_loc = "fallback/ann"
    failing_saver = Mock(side_effect=RuntimeError("save failed"))
    store._persist_ann_index(fallback_loc, legacy_index, ann_index_saver=failing_saver)
    assert fallback_loc in store.zw
    save_index.assert_called_with(store.zw[fallback_loc], legacy_index)

    store.zarr_mode = "r"
    read_only_loc = "readonly/ann"
    calls_before = save_index.call_count
    store._persist_ann_index(read_only_loc, object())
    assert read_only_loc in store.zw
    assert save_index.call_count == calls_before


def test_normalized_cache_validation_paths(
    isolated_toy_datastore: DataStore,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assay = isolated_toy_datastore.RNA
    location = "coverageNormedCache"
    group = assay.z.create_group(location)
    cell_idx, feat_idx = assay._get_cell_feat_idx("I", "I")
    subset_hash = assay._create_subset_hash(cell_idx, feat_idx)
    subset_params = {
        "log_transform": False,
        "renormalize_subset": True,
    }
    group.attrs["subset_hash"] = subset_hash
    group.attrs["subset_params"] = subset_params

    assert (
        GraphDataStore._normed_data_cached(assay, "I", "I", location, False, True)
        is True
    )
    group.attrs["subset_params"] = {
        "log_transform": True,
        "renormalize_subset": True,
    }
    assert (
        GraphDataStore._normed_data_cached(assay, "I", "I", location, False, True)
        is False
    )
    del group.attrs["subset_hash"]
    assert (
        GraphDataStore._normed_data_cached(assay, "I", "I", location, False, True)
        is False
    )

    missing_path = tmp_path / "missing.zarr"
    assert (
        GraphDataStore._staged_normed_cached(
            str(missing_path), subset_hash, subset_params
        )
        is False
    )

    staged_path = tmp_path / "staged.zarr"
    staged_root = zarr.open_group(str(staged_path), mode="w")
    assert (
        GraphDataStore._staged_normed_cached(
            str(staged_path), subset_hash, subset_params
        )
        is False
    )
    staged = staged_root.create_array("data", data=np.eye(2))
    staged.attrs["staged_subset_hash"] = subset_hash
    staged.attrs["staged_subset_params"] = subset_params
    staged.attrs["staged_complete"] = True
    assert (
        GraphDataStore._staged_normed_cached(
            str(staged_path), subset_hash, subset_params
        )
        is True
    )
    staged.attrs["staged_complete"] = False
    assert (
        GraphDataStore._staged_normed_cached(
            str(staged_path), subset_hash, subset_params
        )
        is False
    )

    monkeypatch.setattr(
        "scarf.datastore.graph_datastore.zarr.open_group",
        Mock(side_effect=RuntimeError("broken cache")),
    )
    assert (
        GraphDataStore._staged_normed_cached(
            str(staged_path), subset_hash, subset_params
        )
        is False
    )


def test_partial_normalization_statistics_cache_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _memory_graph_store()
    data = Mock()
    data.mean.return_value = np.array([2.0, 4.0])
    data.std.return_value = np.array([1.5, 2.5])
    monkeypatch.setattr(
        "scarf.datastore.graph_datastore.show_dask_progress",
        lambda values, *_: values,
    )

    missing_mu = store.zw.create_group("missingMu")
    missing_mu.create_array("sigma", data=np.array([3.0, 5.0]))
    mu, sigma = store._load_or_compute_norm_stats("missingMu", data, "pca")
    np.testing.assert_allclose(mu, [2.0, 4.0])
    np.testing.assert_allclose(sigma, [3.0, 5.0])
    np.testing.assert_allclose(missing_mu["mu"][:], [2.0, 4.0])

    missing_sigma = store.zw.create_group("missingSigma")
    missing_sigma.create_array("mu", data=np.array([6.0, 8.0]))
    mu, sigma = store._load_or_compute_norm_stats("missingSigma", data, "pca")
    np.testing.assert_allclose(mu, [6.0, 8.0])
    np.testing.assert_allclose(sigma, [1.5, 2.5])
    np.testing.assert_allclose(missing_sigma["sigma"][:], [1.5, 2.5])

    store.zarr_mode = "r"
    read_only_mu = store.zw.create_group("readOnlyMu")
    read_only_mu.create_array("sigma", data=np.array([3.0, 5.0]))
    mu, sigma = store._load_or_compute_norm_stats("readOnlyMu", data, "pca")
    np.testing.assert_allclose(mu, [2.0, 4.0])
    np.testing.assert_allclose(sigma, [3.0, 5.0])
    assert "mu" not in read_only_mu

    read_only_sigma = store.zw.create_group("readOnlySigma")
    read_only_sigma.create_array("mu", data=np.array([6.0, 8.0]))
    mu, sigma = store._load_or_compute_norm_stats("readOnlySigma", data, "pca")
    np.testing.assert_allclose(mu, [6.0, 8.0])
    np.testing.assert_allclose(sigma, [1.5, 2.5])
    assert "sigma" not in read_only_sigma


def test_remote_cache_plan_auto_and_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _memory_graph_store()
    monkeypatch.setattr(
        "scarf.datastore.graph_datastore.is_remote_datastore",
        lambda *_: True,
    )

    enabled, cache_path, remove = store._resolve_local_cache_plan(
        "s3://bucket/store", store.z, "auto"
    )
    assert enabled is True
    assert cache_path is not None
    assert Path(cache_path).is_dir()
    assert remove is True
    shutil.rmtree(cache_path)

    with pytest.raises(TypeError, match="local_cache must be"):
        store._resolve_local_cache_plan("s3://bucket/store", store.z, object())


def test_get_imputed_creates_loads_and_reuses_operator() -> None:
    store = _memory_graph_store()
    graph_loc = _add_test_graph(store)
    values = np.array([1.0, 2.0, 4.0])
    graph = csr_matrix(
        np.array(
            [
                [0.0, 1.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
            ]
        )
    )
    store._get_latest_keys = Mock(return_value=("RNA", "I", "I"))
    store.get_cell_vals = Mock(return_value=values)
    store._get_latest_graph_loc = Mock(return_value=graph_loc)
    store.load_graph = Mock(return_value=graph)

    with pytest.raises(ValueError, match="name for the feature"):
        store.get_imputed(feature_name=None)
    store.get_cell_vals.assert_not_called()

    first = store.get_imputed(feature_name="gene", t=1)
    np.testing.assert_allclose(first, np.array([3.0, 2.5, 1.5]))
    assert f"{graph_loc}/magic_1" in store.zw
    assert store._cachedMagicOperatorLoc == f"{graph_loc}/magic_1"
    assert store._cachedMagicOperator is not None
    store.load_graph.assert_called_once_with(
        from_assay="RNA",
        cell_key="I",
        feat_key="I",
        symmetric=True,
        upper_only=False,
    )

    cached_operator = store._cachedMagicOperator
    second = store.get_imputed(feature_name="gene", t=1)
    np.testing.assert_allclose(second, first)
    assert store._cachedMagicOperator is cached_operator
    assert store.load_graph.call_count == 1

    store._cachedMagicOperator = None
    store._cachedMagicOperatorLoc = None
    loaded = store.get_imputed(feature_name="gene", t=1, cache_operator=True)
    np.testing.assert_allclose(loaded, first)
    assert store._cachedMagicOperator is not None
    assert store._cachedMagicOperatorLoc == f"{graph_loc}/magic_1"

    store._cachedMagicOperator = None
    store._cachedMagicOperatorLoc = None
    uncached = store.get_imputed(feature_name="gene", t=1, cache_operator=False)
    np.testing.assert_allclose(uncached, first)
    assert store._cachedMagicOperator is None
    assert store._cachedMagicOperatorLoc is None

    squared = store.get_imputed(feature_name="gene", t=2, cache_operator=False)
    np.testing.assert_allclose(squared, np.array([2.0, 2.25, 2.75]))
    assert f"{graph_loc}/magic_2" in store.zw
    assert store.load_graph.call_count == 2
    assert store._cachedMagicOperator is None
    assert store._cachedMagicOperatorLoc is None


def test_filter_cells_open_bounds_reset_and_boundaries(
    isolated_toy_datastore: DataStore,
) -> None:
    store = isolated_toy_datastore
    attr = "RNA_nCounts"
    values = store.cells.fetch_all(attr)
    lower = float(values.min())
    upper = float(values.max())

    store.cells.reset_key("I")
    store.filter_cells(
        attrs=[attr],
        lows=[lower],
        highs=[None],
        reset_previous=True,
    )
    expected = values > lower
    np.testing.assert_array_equal(store.cells.fetch_all("I"), expected)

    store.filter_cells(
        attrs=[attr],
        lows=[None],
        highs=[upper],
    )
    expected &= values < upper
    np.testing.assert_array_equal(store.cells.fetch_all("I"), expected)

    store.filter_cells(
        attrs=[attr, "missing"],
        lows=[None, None],
        highs=[None, None],
        reset_previous=True,
    )
    assert store.cells.fetch_all("I").all()

    store.filter_cells(
        attrs=[attr],
        lows=[lower],
        highs=[upper],
        reset_previous=True,
        keep_bounds=True,
    )
    assert store.cells.fetch_all("I").all()


def test_run_marker_search_skip_save_and_errors(
    isolated_toy_datastore: DataStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = isolated_toy_datastore
    monkeypatch.setattr(store, "_get_latest_feat_key", lambda _: "I")
    markers = {"cluster": object()}
    finder = Mock(return_value=markers)
    monkeypatch.setattr("scarf.markers.find_markers_by_rank", finder)
    prenormed_group = store.zw.create_group("coveragePrenormed")

    with pytest.raises(ValueError, match="group_key"):
        store.run_marker_search(group_key=None)

    result = store.run_marker_search(
        group_key="ids",
        prenormed_store="coveragePrenormed",
        skip_save=True,
    )
    assert result is markers
    first_call = finder.call_args.kwargs
    assert first_call["assay"] is store.RNA
    assert first_call["group_key"] == "ids"
    assert first_call["cell_key"] == "I"
    assert first_call["feat_key"] == "I"
    assert first_call["batch_size"] >= 1
    assert first_call["n_threads"] == store.nthreads
    assert first_call["prenormed_store"].path == "coveragePrenormed"
    assert "I__ids" not in store.zw["RNA/markers"]

    finder.reset_mock()
    result = store.run_marker_search(
        group_key="ids",
        cell_key="I",
        feat_key="I",
        gene_batch_size=2,
        use_prenormed=True,
        prenormed_store=prenormed_group,
        n_threads=3,
        skip_save=True,
        log_transform=False,
    )
    assert result is markers
    second_call = finder.call_args.kwargs
    assert second_call["batch_size"] == 2
    assert second_call["use_prenormed"] is True
    assert second_call["prenormed_store"] is prenormed_group
    assert second_call["n_threads"] == 3
    assert second_call["log_transform"] is False

    with pytest.raises(KeyError):
        store.run_marker_search(
            group_key="ids",
            prenormed_store="missing",
            skip_save=True,
        )

    finder.side_effect = RuntimeError("marker failure")
    with pytest.raises(RuntimeError, match="marker failure"):
        store.run_marker_search(group_key="ids", skip_save=True)


def test_make_graph_harmony_input_validation(
    isolated_toy_datastore: DataStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = isolated_toy_datastore
    graph_params = (
        True,
        True,
        "pca",
        2,
        "I",
        "l2",
        50,
        50,
        48,
        1,
        2,
        2,
        1.0,
        1.5,
    )
    set_params = Mock(return_value=graph_params)
    monkeypatch.setattr(store, "_set_graph_params", set_params)

    with pytest.raises(ValueError, match="no batches provided"):
        store.make_graph(
            from_assay="RNA",
            cell_key="I",
            feat_key="I",
            harmonize=True,
            batch_columns=None,
        )

    with pytest.raises(ValueError, match="batches must be a list"):
        store.make_graph(
            from_assay="RNA",
            cell_key="I",
            feat_key="I",
            harmonize=True,
            batch_columns="ids",
        )

    assert set_params.call_count == 2


def test_run_tsne_orchestration_and_error_paths(
    isolated_toy_datastore: DataStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = isolated_toy_datastore
    graph = csr_matrix(
        np.array(
            [
                [0.0, 1.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
            ]
        )
    )
    initial = np.array(
        [
            [0.0, 0.0],
            [0.5, 0.5],
            [1.0, 1.0],
        ]
    )
    embedding = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
    )
    latest_keys = Mock(return_value=("RNA", "I", "I"))
    load_graph = Mock(return_value=graph)
    get_initial = Mock(return_value=initial)
    runner = Mock(return_value=embedding)
    monkeypatch.setattr(store, "_get_latest_keys", latest_keys)
    monkeypatch.setattr(store, "load_graph", load_graph)
    monkeypatch.setattr(store, "_get_ini_embed", get_initial)
    monkeypatch.setattr("scarf.knn_utils.run_sgtsne", runner)

    store.run_tsne(
        symmetric_graph=True,
        graph_upper_only=True,
        parallel=True,
        nthreads=None,
        max_iter=20,
        label="coverageTsne",
    )
    get_initial.assert_called_once_with("RNA", "I", "I", 2)
    first_call = runner.call_args
    assert first_call.args[0] is graph
    np.testing.assert_array_equal(first_call.args[1], initial)
    assert first_call.kwargs["parallel"] is True
    assert first_call.kwargs["nthreads"] == store.nthreads
    assert first_call.kwargs["max_iter"] == 20
    np.testing.assert_allclose(store.cells.fetch("RNA_coverageTsne1"), embedding[0])
    np.testing.assert_allclose(store.cells.fetch("RNA_coverageTsne2"), embedding[1])

    store.run_tsne(
        from_assay="RNA",
        cell_key="I",
        feat_key="I",
        ini_embed=initial,
        parallel=False,
        label="serialTsne",
    )
    assert runner.call_args.kwargs["nthreads"] == 1
    assert runner.call_args.kwargs["parallel"] is False

    with pytest.raises(ValueError, match="required shape"):
        store.run_tsne(
            ini_embed=np.zeros((2, 2)),
            tsne_dims=2,
        )

    runner.side_effect = FileNotFoundError("sgtsne missing")
    store.run_tsne(
        ini_embed=initial,
        parallel=True,
        nthreads=2,
        label="missingTsne",
    )
    assert runner.call_args.kwargs["nthreads"] == 2

    latest_calls = latest_keys.call_count
    monkeypatch.setattr(sys, "platform", "win32")
    assert store.run_tsne() is None
    assert latest_keys.call_count == latest_calls


def test_integrate_assays_snn_writes_and_overwrites_graph() -> None:
    store = _memory_graph_store(["RNA", "ADT"])
    graphs = {
        "RNA": csr_matrix(
            np.array(
                [
                    [0.0, 1.0, 2.0],
                    [3.0, 0.0, 4.0],
                    [5.0, 6.0, 0.0],
                ]
            )
        ),
        "ADT": csr_matrix(
            np.array(
                [
                    [0.0, 7.0, 8.0],
                    [9.0, 0.0, 10.0],
                    [11.0, 12.0, 0.0],
                ]
            )
        ),
    }
    load_graph = Mock(side_effect=lambda **kwargs: graphs[kwargs["from_assay"]])
    store.load_graph = load_graph

    store.integrate_assays(
        assays=["RNA", "ADT"],
        label="joint",
        method="snn",
        chunk_size=2,
    )
    store.integrate_assays(
        assays=["RNA", "ADT"],
        label="joint",
        method="snn",
        chunk_size=2,
    )

    integrated_group = store.zw["integratedGraphs/joint"]
    assert integrated_group.attrs["n_cells"] == 3
    assert integrated_group.attrs["n_neighbors"] == 2
    assert integrated_group["edges"].shape == (6, 2)
    assert integrated_group["weights"].shape == (6,)
    assert load_graph.call_count == 4

    integrated = GraphDataStore.load_graph(
        store,
        from_assay="RNA",
        cell_key="I",
        feat_key="I",
        graph_loc="integratedGraphs/joint",
    )
    assert integrated.shape == (3, 3)
    np.testing.assert_array_equal(np.diff(integrated.indptr), [2, 2, 2])
    np.testing.assert_allclose(
        np.sort(integrated.data),
        np.array([7.0, 8.0, 9.0, 10.0, 11.0, 12.0]),
    )


def test_integrate_assays_validation_errors() -> None:
    store = _memory_graph_store(["RNA", "ADT"])
    graph = csr_matrix(
        np.array(
            [
                [0.0, 1.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
            ]
        )
    )
    store.load_graph = Mock(return_value=graph)

    with pytest.raises(ValueError, match="missing was not found"):
        store.integrate_assays(
            assays=["RNA", "missing"],
            label="invalid",
            method="snn",
        )

    with pytest.raises(ValueError, match="only two assays"):
        store.integrate_assays(
            assays=["RNA"],
            label="invalid",
            method="wnn",
        )

    with pytest.raises(ValueError, match="Method unknown not supported"):
        store.integrate_assays(
            assays=["RNA", "ADT"],
            label="invalid",
            method="unknown",
        )
