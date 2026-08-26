import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import zarr
from scipy.sparse import coo_matrix, csr_matrix
from zarr.storage import MemoryStore

from scarf.datastore.datastore import DataStore
from scarf.datastore.graph_datastore import GraphDataStore
from scarf.graph.state import GraphSelection, integrated_graph_label
from scarf.storage.artifacts import (
    ArtifactRef,
    artifact_path,
    list_artifacts,
    make_provenance,
    new_artifact_id,
)
from scarf.storage.errors import ArtifactResolutionError


class _MemoryGraphStore(GraphDataStore):
    @property
    def assay_names(self) -> list[str]:
        return self._assay_names


class _CoordinateBlocks:
    data = None

    def __init__(self, blocks: list[np.ndarray]) -> None:
        self.blocks = blocks

    def iter_coordinate_blocks(self, _message: str):
        yield from self.blocks


@pytest.fixture
def isolated_toy_datastore(toy_crdir_writer: str, tmp_path: Path) -> DataStore:
    zarr_path = tmp_path / "toy.zarr"
    shutil.copytree(toy_crdir_writer, zarr_path)
    return DataStore(
        str(zarr_path),
        default_assay="RNA",
        min_features_per_cell=0,
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
    store.storageProfile = "fast_local"
    return store


def _add_test_graph(
    store: _MemoryGraphStore,
    label: str = "graph",
) -> ArtifactRef:
    graph = ArtifactRef(
        scope="datastore",
        kind="integrated_graph",
        artifact_id=new_artifact_id(),
    )
    graph_group = store.zw.create_group(artifact_path(graph))
    graph_group.attrs.update(
        {
            "artifact_id": graph.artifact_id,
            "kind": graph.kind,
            "provenance": make_provenance(
                operation="test_graph",
                parameters={"label": label},
                inputs={},
            ),
            "execution_options": {},
            "complete": True,
        }
    )
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
    return graph


def _add_complete_artifact(
    store: _MemoryGraphStore,
    kind: str,
    *,
    assay: str | None = "RNA",
    inputs: dict[str, object] | None = None,
    parameters: dict[str, object] | None = None,
    arrays: dict[str, np.ndarray] | None = None,
) -> ArtifactRef:
    ref = ArtifactRef(
        scope="assay" if assay is not None else "datastore",
        assay=assay,
        kind=kind,
        artifact_id=new_artifact_id(),
    )
    group = store.zw.create_group(artifact_path(ref))
    group.attrs.update(
        {
            "artifact_id": ref.artifact_id,
            "kind": kind,
            "provenance": make_provenance(
                operation=f"test_{kind}",
                parameters=parameters or {},
                inputs=inputs or {},
            ),
            "execution_options": {},
            "complete": True,
        }
    )
    for name, values in (arrays or {}).items():
        group.create_array(name, data=values)
    return ref


def _patch_trajectory_graph_resolution(
    monkeypatch: pytest.MonkeyPatch,
    graph: ArtifactRef,
) -> None:
    def resolve(
        _store: object,
        selected: ArtifactRef | None,
        *,
        from_assay: str | None,
        cell_key: str | None,
    ) -> GraphSelection:
        assert selected is None or selected == graph
        return GraphSelection(
            graph_loc=artifact_path(graph),
            graph_ref=graph,
            from_assay=from_assay or "RNA",
            cell_key=cell_key or "I",
            integrated_label="graph",
        )

    monkeypatch.setattr(
        "scarf.datastore._operations.trajectory.resolve_graph_selection",
        resolve,
    )


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
    graph_ref = _add_test_graph(store)

    graph = store._load_graph_artifact(
        graph_ref,
        symmetric=symmetric,
        upper_only=upper_only,
        use_k=use_k,
    )

    np.testing.assert_allclose(graph.toarray(), expected)


def test_graph_memory_cache_is_keyed_and_scoped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _memory_graph_store()
    graph_ref = _add_test_graph(store)
    original = store._store_to_sparse
    reads = 0

    def counted_store_to_sparse(
        location: str,
        sparse_format: str = "csr",
        use_k: int | None = None,
    ):
        nonlocal reads
        reads += 1
        return original(location, sparse_format, use_k)

    monkeypatch.setattr(store, "_store_to_sparse", counted_store_to_sparse)

    with store._graph_memory_cache_scope():
        raw = store._load_graph_artifact(
            graph_ref,
            symmetric=None,
            upper_only=None,
            use_k=None,
        )
        equivalent = store._load_graph_artifact(
            graph_ref,
            symmetric=False,
            upper_only=True,
            use_k=None,
        )
        symmetric = store._load_graph_artifact(
            graph_ref,
            symmetric=True,
            upper_only=None,
            use_k=None,
        )
        reduced = store._load_graph_artifact(
            graph_ref,
            symmetric=None,
            upper_only=None,
            use_k=1,
        )

        assert raw is equivalent
        assert raw is not symmetric
        assert raw is not reduced
        assert reads == 3
        with store._graph_memory_cache_scope():
            nested = store._load_graph_artifact(
                graph_ref,
                symmetric=None,
                upper_only=None,
                use_k=None,
            )
            assert nested is raw
            assert reads == 3

    assert store._graphMemoryCache is None
    uncached = store._load_graph_artifact(
        graph_ref,
        symmetric=None,
        upper_only=None,
        use_k=None,
    )
    assert uncached is not raw
    assert reads == 4


def test_corrupt_zarr_ann_bytes_raise_artifact_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _memory_graph_store()
    ann = _add_complete_artifact(
        store,
        "ann_index",
        arrays={"ann_idx_bytes": np.array([1, 2, 3], dtype=np.uint8)},
    )
    monkeypatch.setattr(
        "scarf.datastore._operations.graph.load_ann_index",
        Mock(side_effect=RuntimeError("corrupt Zarr ANN bytes")),
    )

    with pytest.raises(ArtifactResolutionError) as caught:
        store._resolve_ann_index(ann, "l2", 3)

    assert caught.value.code == "corrupt_payload"
    assert caught.value.context["artifact_id"] == ann.artifact_id
    assert isinstance(caught.value.__cause__, RuntimeError)


def test_legacy_filesystem_ann_is_not_loaded_without_zarr_bytes(
    tmp_path: Path,
) -> None:
    import hnswlib

    store = _memory_graph_store()
    store_path = tmp_path / "store.zarr"
    store.z = zarr.open_group(str(store_path), mode="w")
    ann = _add_complete_artifact(store, "ann_index")
    ann_group = store.zw[artifact_path(ann)]
    before_attrs = dict(ann_group.attrs)
    data = np.random.default_rng(9).random((20, 3), dtype=np.float32)
    source = hnswlib.Index(space="l2", dim=3)
    source.init_index(max_elements=len(data), ef_construction=50, M=16)
    source.add_items(data)
    legacy_path = store_path / artifact_path(ann) / "ann_idx"
    source.save_index(str(legacy_path))

    with pytest.raises(ArtifactResolutionError) as caught:
        store._resolve_ann_index(ann, "l2", 3, expected_count=len(data))

    assert caught.value.code == "corrupt_payload"
    assert "ann_idx_bytes" not in ann_group
    assert dict(ann_group.attrs) == before_attrs
    assert legacy_path.exists()


def test_partial_normalization_statistics_cache_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _memory_graph_store()
    data = Mock()
    data.mean.return_value = np.array([2.0, 4.0])
    data.std.return_value = np.array([1.5, 2.5])
    monkeypatch.setattr(
        "scarf.datastore._operations.graph.compute_with_progress",
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
        "scarf.datastore._operations.graph.is_remote_datastore",
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


def test_get_imputed_creates_loads_and_reuses_operator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _memory_graph_store()
    graph_ref = _add_test_graph(store)
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
    _patch_trajectory_graph_resolution(monkeypatch, graph_ref)
    store.get_cell_vals = Mock(return_value=values)
    store.load_graph = Mock(return_value=graph)

    operator = store.get_diffusion_operator(t=1)
    assert isinstance(operator, coo_matrix)
    store.get_cell_vals.assert_not_called()

    first = store.get_imputed(feature_name="gene", t=1)
    np.testing.assert_allclose(first, np.array([3.0, 2.5, 1.5]))
    diffusion_refs = list_artifacts(
        store.zw,
        scope="datastore",
        kind="diffusion_operator",
    )
    assert len(diffusion_refs) == 1
    assert artifact_path(diffusion_refs[0]) in store.zw
    assert store._cachedMagicOperatorLoc == diffusion_refs[0].artifact_id
    assert store._cachedMagicOperator is not None
    store.load_graph.assert_called_once_with(
        graph_ref,
        from_assay="RNA",
        cell_key="I",
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
    assert store._cachedMagicOperatorLoc == diffusion_refs[0].artifact_id

    store._cachedMagicOperator = None
    store._cachedMagicOperatorLoc = None
    uncached = store.get_imputed(feature_name="gene", t=1, cache_operator=False)
    np.testing.assert_allclose(uncached, first)
    assert store._cachedMagicOperator is None
    assert store._cachedMagicOperatorLoc is None

    squared = store.get_imputed(feature_name="gene", t=2, cache_operator=False)
    np.testing.assert_allclose(squared, np.array([2.0, 2.25, 2.75]))
    assert (
        len(
            list_artifacts(
                store.zw,
                scope="datastore",
                kind="diffusion_operator",
            )
        )
        == 2
    )
    assert store.load_graph.call_count == 2
    assert store._cachedMagicOperator is None
    assert store._cachedMagicOperatorLoc is None


def test_get_diffusion_operator_discards_stale_in_memory_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _memory_graph_store()
    graph_ref = _add_test_graph(store)
    first_graph = csr_matrix(
        np.array(
            [
                [0.0, 1.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
            ]
        )
    )
    replacement_graph = csr_matrix(np.eye(3))
    _patch_trajectory_graph_resolution(monkeypatch, graph_ref)
    store.load_graph = Mock(side_effect=[first_graph, replacement_graph])

    first = store.get_diffusion_operator(t=1)
    first_ref = list_artifacts(
        store.zw,
        scope="datastore",
        kind="diffusion_operator",
    )[0]
    del store.zw[artifact_path(first_ref)]
    replacement = store.get_diffusion_operator(t=1)

    assert store.load_graph.call_count == 2
    assert replacement is not first
    np.testing.assert_allclose(replacement.toarray(), np.eye(3))


def test_read_only_diffusion_operator_reuses_memory_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _memory_graph_store()
    graph_ref = _add_test_graph(store)
    graph = csr_matrix(
        np.array(
            [
                [0.0, 1.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
            ]
        )
    )
    store.zarr_mode = "r"
    _patch_trajectory_graph_resolution(monkeypatch, graph_ref)
    store.load_graph = Mock(return_value=graph)

    first = store.get_diffusion_operator(t=1, cache_operator=True)
    second = store.get_diffusion_operator(t=1, cache_operator=True)

    assert second is first
    assert store.load_graph.call_count == 1
    assert store._cachedMagicOperatorLoc is not None
    assert store._cachedMagicOperatorLoc.startswith("read_only:")

    invalidated = store.get_diffusion_operator(
        t=1,
        cache_operator=True,
        invalidate_cache=True,
    )
    assert invalidated is not first
    assert store.load_graph.call_count == 2

    store.get_diffusion_operator(t=1, cache_operator=False)
    assert store.load_graph.call_count == 3
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
    all_features = store._ensure_all_features("RNA")
    markers = {"cluster": object()}
    finder = Mock(return_value=markers)
    monkeypatch.setattr("scarf.features.markers.find_markers_by_rank", finder)

    with pytest.raises(ValueError, match="group_key"):
        store.run_marker_search(group_key=None, features=all_features)

    result = store.run_marker_search(
        group_key="ids",
        features=all_features,
        skip_save=True,
    )
    assert result is markers
    first_call = finder.call_args.kwargs
    assert first_call["assay"] is store.RNA
    np.testing.assert_array_equal(first_call["groups"], store.cells.fetch("ids"))
    np.testing.assert_array_equal(
        first_call["cell_idx"],
        store.cells.active_index("I"),
    )
    np.testing.assert_array_equal(
        first_call["feat_idx"],
        np.arange(store.RNA.feats.N),
    )
    assert "batch_size" not in first_call
    assert first_call["nthreads"] == store.nthreads
    assert "markers" not in store.zw["RNA"]

    finder.reset_mock()
    result = store.run_marker_search(
        group_key="ids",
        cell_key="I",
        features=all_features,
        nthreads=3,
        skip_save=True,
        log_transform=False,
    )
    assert result is markers
    second_call = finder.call_args.kwargs
    assert "batch_size" not in second_call
    assert second_call["nthreads"] == 3
    assert second_call["log_transform"] is False

    finder.side_effect = RuntimeError("marker failure")
    with pytest.raises(RuntimeError, match="marker failure"):
        store.run_marker_search(
            group_key="ids",
            features=all_features,
            skip_save=True,
        )


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
    load_graph = Mock(return_value=graph)
    graph_ref = ArtifactRef(
        scope="assay",
        assay="RNA",
        kind="connectivity_map",
        artifact_id="1" * 64,
    )
    initialization_ref = ArtifactRef(
        scope="assay",
        assay="RNA",
        kind="embedding_initialization",
        artifact_id="2" * 64,
    )
    get_initial = Mock(return_value=(initial, initialization_ref))
    runner = Mock(return_value=embedding)
    selection_ref = store._ensure_cell_selection("I")
    graph_selection = GraphSelection(
        graph_loc=artifact_path(graph_ref),
        graph_ref=graph_ref,
        from_assay="RNA",
        cell_key="I",
        integrated_label=None,
    )
    monkeypatch.setattr(
        "scarf.datastore._operations.embeddings.resolve_graph_selection",
        lambda *_args, **_kwargs: graph_selection,
    )
    monkeypatch.setattr(store, "load_graph", load_graph)
    monkeypatch.setattr(store, "_get_ini_embed", get_initial)
    monkeypatch.setattr(
        store, "_graph_cell_selection", Mock(return_value=selection_ref)
    )
    monkeypatch.setattr("scarf.embeddings.sgtsne.run_sgtsne", runner)

    tsne_ref = store.run_tsne(
        symmetric_graph=True,
        graph_upper_only=True,
        parallel=True,
        nthreads=None,
        max_iter=20,
        label="coverageTsne",
    )
    get_initial.assert_called_once_with("RNA", "I", graph_ref, 2)
    first_call = runner.call_args
    assert first_call.args[0] is graph
    np.testing.assert_array_equal(first_call.args[1], initial)
    assert first_call.kwargs["parallel"] is True
    assert first_call.kwargs["nthreads"] == store.nthreads
    assert first_call.kwargs["max_iter"] == 20
    np.testing.assert_allclose(store.cells.fetch("RNA_coverageTsne1"), embedding[0])
    np.testing.assert_allclose(store.cells.fetch("RNA_coverageTsne2"), embedding[1])
    first_ref = ArtifactRef.from_dict(
        store.zw["cellData/RNA_coverageTsne1"].attrs["source_artifact"]
    )
    assert first_ref == tsne_ref
    assert first_ref.kind == "embedding"
    assert (
        ArtifactRef.from_dict(
            store.zw["cellData/RNA_coverageTsne2"].attrs["source_artifact"]
        )
        == first_ref
    )
    np.testing.assert_allclose(
        store.load_artifact(first_ref)["values"][:],
        embedding.T,
    )

    store.run_tsne(
        graph_ref,
        from_assay="RNA",
        cell_key="I",
        ini_embed=initial,
        parallel=False,
        label="serialTsne",
        invalidate_cache=True,
    )
    assert runner.call_args.kwargs["nthreads"] == 1
    assert runner.call_args.kwargs["parallel"] is False

    with pytest.raises(ValueError, match="invalid shape"):
        store.run_tsne(
            ini_embed=np.zeros((2, 2)),
            tsne_dims=2,
        )

    runner.side_effect = FileNotFoundError("sgtsne missing")
    with pytest.raises(RuntimeError, match="SG-tSNE failed"):
        store.run_tsne(
            ini_embed=initial,
            parallel=True,
            nthreads=2,
            label="missingTsne",
        )
    assert runner.call_args.kwargs["nthreads"] == 2

    runner_calls = runner.call_count
    monkeypatch.setattr(sys, "platform", "win32")
    assert (
        store.run_tsne(
            symmetric_graph=True,
            graph_upper_only=True,
            parallel=True,
            nthreads=None,
            max_iter=20,
            label="cachedTsne",
        )
        == first_ref
    )
    assert runner.call_count == runner_calls
    np.testing.assert_allclose(
        store.cells.fetch("RNA_cachedTsne1"),
        embedding[0],
    )


def test_integrate_assays_snn_writes_and_overwrites_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _memory_graph_store(["RNA", "ADT"])
    monkeypatch.setattr(
        "scarf.datastore._operations.graph.validate_cell_selection_artifact",
        lambda *_args, **_kwargs: None,
    )
    selection_ref = ArtifactRef(
        scope="datastore",
        kind="cell_selection",
        artifact_id="a" * 64,
    )
    store._graph_cell_selection = Mock(return_value=selection_ref)
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
    sources = {
        assay: ArtifactRef(
            scope="assay",
            assay=assay,
            kind="connectivity_map",
            artifact_id=new_artifact_id(),
        )
        for assay in graphs
    }
    states = {
        assay: SimpleNamespace(
            neighbors=None,
            connectivity_map=sources[assay],
            cell_key="I",
        )
        for assay in graphs
    }
    monkeypatch.setattr(
        "scarf.datastore._operations.graph.read_assay_state",
        lambda _root, assay: states[assay],
    )
    monkeypatch.setattr(
        "scarf.datastore._operations.graph.resolve_native_graph_inputs",
        lambda *_args: SimpleNamespace(cell_selection=selection_ref),
    )
    monkeypatch.setattr(
        "scarf.datastore._operations.graph._validate_integration_source_payload",
        lambda *_args: 3,
    )
    load_captured = Mock(side_effect=lambda ref, **_kwargs: graphs[ref.assay])
    store._load_graph_artifact = load_captured

    first_ref = store.integrate_assays(
        assays=["RNA", "ADT"],
        label="joint",
        method="snn",
        chunk_size=2,
    )
    second_ref = store.integrate_assays(
        assays=["RNA", "ADT"],
        label="joint",
        method="snn",
        chunk_size=2,
    )

    assert first_ref.kind == "integrated_graph"
    assert second_ref == first_ref
    integrated_path = artifact_path(first_ref)
    registered = store.zw["integratedGraphs"].attrs["artifacts"]
    assert ArtifactRef.from_dict(registered["joint"]) == first_ref
    integrated_group = store.zw[integrated_path]
    assert integrated_group.attrs["n_cells"] == 3
    assert integrated_group.attrs["n_neighbors"] == 2
    assert integrated_group["edges"].shape == (6, 2)
    assert integrated_group["weights"].shape == (6,)
    assert load_captured.call_count == 2
    status = store.inspect_artifact(first_ref)
    assert set(status.inputs or {}) == {
        "cell_selection",
        "source_0",
        "source_1",
    }
    assert ArtifactRef.from_dict(status.inputs["source_0"]) == sources["RNA"]
    assert ArtifactRef.from_dict(status.inputs["source_1"]) == sources["ADT"]

    loaded = GraphDataStore._load_graph_artifact(
        store,
        first_ref,
        symmetric=None,
        upper_only=None,
        use_k=None,
    )
    assert loaded.shape == (3, 3)
    np.testing.assert_array_equal(np.diff(loaded.indptr), [2, 2, 2])
    np.testing.assert_allclose(
        np.sort(loaded.data),
        np.array([7.0, 8.0, 9.0, 10.0, 11.0, 12.0]),
    )


@pytest.mark.parametrize("method", ["snn", "wnn"])
def test_integrate_assays_uses_sources_captured_before_planning(
    monkeypatch: pytest.MonkeyPatch,
    method: str,
) -> None:
    import scarf.datastore._operations.graph as graph_operations

    store = _memory_graph_store(["RNA", "ADT"])
    cell_data = store.zw.create_group("cellData")
    cell_data.create_array("I", data=np.ones(3, dtype=bool))

    def insert_cell_column(
        column: str,
        values: np.ndarray,
        *,
        overwrite: bool,
        key: str,
    ) -> None:
        assert overwrite is True
        assert key == "I"
        cell_data.create_array(column, data=np.asarray(values), overwrite=True)

    store.cells = SimpleNamespace(insert=insert_cell_column)
    selection = _add_complete_artifact(store, "cell_selection", assay=None)
    neighbor_indices = np.array(
        [[1, 2], [0, 2], [0, 1]],
        dtype=np.uint32,
    )
    replacement_indices = np.array(
        [[2, 1], [2, 0], [1, 0]],
        dtype=np.uint32,
    )
    captured_sources: dict[str, ArtifactRef] = {}
    replacement_sources: dict[str, ArtifactRef] = {}
    captured_coordinates: dict[str, ArtifactRef] = {}
    replacement_coordinates: dict[str, ArtifactRef] = {}
    coordinate_by_source: dict[ArtifactRef, ArtifactRef] = {}
    for assay in ("RNA", "ADT"):
        source_kind = "connectivity_map" if method == "snn" else "neighbors"
        captured_sources[assay] = _add_complete_artifact(
            store,
            source_kind,
            assay=assay,
            arrays={"indices": neighbor_indices} if method == "wnn" else None,
        )
        replacement_sources[assay] = _add_complete_artifact(
            store,
            source_kind,
            assay=assay,
            arrays={"indices": replacement_indices} if method == "wnn" else None,
        )
        captured_coordinates[assay] = _add_complete_artifact(
            store,
            "reduction",
            assay=assay,
        )
        replacement_coordinates[assay] = _add_complete_artifact(
            store,
            "reduction",
            assay=assay,
        )
        coordinate_by_source[captured_sources[assay]] = captured_coordinates[assay]
        coordinate_by_source[replacement_sources[assay]] = replacement_coordinates[
            assay
        ]

    active_sources = dict(captured_sources)

    def current_state(_root: zarr.Group, assay: str) -> SimpleNamespace:
        source = active_sources[assay]
        return SimpleNamespace(
            neighbors=source if method == "wnn" else None,
            connectivity_map=source if method == "snn" else None,
            cell_key="I",
        )

    read_state = Mock(side_effect=current_state)
    monkeypatch.setattr(graph_operations, "read_assay_state", read_state)
    monkeypatch.setattr(
        graph_operations,
        "resolve_native_graph_inputs",
        lambda _root, source: SimpleNamespace(
            coordinates=coordinate_by_source[source],
            cell_selection=selection,
        ),
    )
    monkeypatch.setattr(
        graph_operations,
        "validate_cell_selection_artifact",
        Mock(),
    )
    validate_source = Mock(return_value=3)
    monkeypatch.setattr(
        graph_operations,
        "_validate_integration_source_payload",
        validate_source,
    )

    real_plan_artifact = graph_operations.plan_artifact

    def plan_and_change_current_state(*args: object, **kwargs: object):
        planned = real_plan_artifact(*args, **kwargs)
        active_sources.update(replacement_sources)
        return planned

    monkeypatch.setattr(
        graph_operations,
        "plan_artifact",
        plan_and_change_current_state,
    )

    merged = coo_matrix(
        (
            np.arange(1, 7, dtype=np.float32),
            (
                np.array([0, 0, 1, 1, 2, 2]),
                np.array([1, 2, 0, 2, 0, 1]),
            ),
        ),
        shape=(3, 3),
    )
    if method == "snn":
        graphs = {
            ref: csr_matrix(np.full((3, 3), index + 1, dtype=np.float32))
            for index, ref in enumerate(captured_sources.values())
        }
        load_graph = Mock(side_effect=lambda ref, **_kwargs: graphs[ref])
        store._load_graph_artifact = load_graph
        merge_graphs = Mock(return_value=merged)
        monkeypatch.setattr("scarf.neighbors.graph.merge_graphs", merge_graphs)
    else:
        coordinate_values = {
            ref: np.full((3, 2), index + 1, dtype=np.float32)
            for index, ref in enumerate(captured_coordinates.values())
        }

        def coordinate_source(
            ref: ArtifactRef,
            *,
            batch_size: int | None,
        ) -> tuple[_CoordinateBlocks, int, int]:
            assert batch_size is None
            return _CoordinateBlocks([coordinate_values[ref]]), 3, 2

        store._coordinate_source = Mock(side_effect=coordinate_source)
        modality_weights = np.array(
            [[0.6, 0.4], [0.5, 0.5], [0.4, 0.6]],
            dtype=np.float32,
        )
        integrate_wnn = Mock(return_value=(merged, modality_weights))
        monkeypatch.setattr(
            "scarf.neighbors.integration._wnn_integration_many",
            integrate_wnn,
        )

    integrated = store.integrate_assays(
        ["RNA", "ADT"],
        label=f"captured_{method}",
        method=method,
    )

    assert store.inspect_artifact(integrated).complete
    assert [call.args for call in read_state.call_args_list] == [
        (store.zw, "RNA"),
        (store.zw, "ADT"),
    ]
    assert [call.args[1] for call in validate_source.call_args_list] == list(
        captured_sources.values()
    )
    status = store.inspect_artifact(integrated)
    for index, assay in enumerate(("RNA", "ADT")):
        stored_source = status.inputs[f"source_{index}"]
        if method == "snn":
            assert ArtifactRef.from_dict(stored_source) == captured_sources[assay]
        else:
            stored_neighbors = ArtifactRef.from_dict(stored_source["neighbors"])
            stored_coordinates = ArtifactRef.from_dict(stored_source["coordinates"])
            assert stored_neighbors == captured_sources[assay]
            assert stored_coordinates == captured_coordinates[assay]

    if method == "snn":
        assert [call.args[0] for call in load_graph.call_args_list] == list(
            captured_sources.values()
        )
    else:
        assert [
            call.args[0] for call in store._coordinate_source.call_args_list
        ] == list(captured_coordinates.values())
        modalities = integrate_wnn.call_args.args[0]
        assert [modality[0] for modality in modalities] == ["RNA", "ADT"]
        for _assay, indices, _coordinates in modalities:
            np.testing.assert_array_equal(indices, neighbor_indices)


def test_integrate_assays_validation_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _memory_graph_store(["RNA", "ADT"])

    with pytest.raises(ValueError, match="missing was not found"):
        store.integrate_assays(
            assays=["missing", "RNA"],
            label="invalid",
            method="snn",
        )

    with pytest.raises(ValueError, match="at least two assays"):
        store.integrate_assays(
            assays=["RNA"],
            label="invalid",
            method="wnn",
        )

    with pytest.raises(ValueError, match="unique assay names"):
        store.integrate_assays(
            assays=["RNA", "RNA"],
            label="invalid",
            method="wnn",
        )

    with pytest.raises(TypeError, match="l2_normalize must be a boolean"):
        store.integrate_assays(
            assays=["RNA", "ADT"],
            label="invalid",
            method="wnn",
            l2_normalize="yes",
        )

    with pytest.raises(ValueError, match="Method unknown not supported"):
        store.integrate_assays(
            assays=["RNA", "ADT"],
            label="invalid",
            method="unknown",
        )

    monkeypatch.setattr(
        "scarf.datastore._operations.graph.read_assay_state",
        lambda *_args: None,
    )
    with pytest.raises(ArtifactResolutionError) as missing_graph:
        store.integrate_assays(
            assays=["RNA", "ADT"],
            label="invalid",
            method="snn",
        )
    assert missing_graph.value.code == "missing_current_graph"
    assert missing_graph.value.context == {"assay": "RNA"}

    with pytest.raises(ArtifactResolutionError) as missing_neighbors:
        store.integrate_assays(
            assays=["RNA", "ADT"],
            label="invalid",
            method="wnn",
        )
    assert missing_neighbors.value.code == "missing_current_neighbors"
    assert missing_neighbors.value.context == {"assay": "RNA"}


@pytest.mark.parametrize(
    ("method", "source_kind"),
    [("snn", "connectivity_map"), ("wnn", "neighbors")],
)
def test_integrate_assays_rejects_corrupt_sources_before_planning(
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    source_kind: str,
) -> None:
    store = _memory_graph_store(["RNA", "ADT"])
    source = _add_complete_artifact(store, source_kind, assay="RNA")
    selection = ArtifactRef(
        scope="datastore",
        kind="cell_selection",
        artifact_id="a" * 64,
    )
    coordinates = ArtifactRef(
        scope="assay",
        assay="RNA",
        kind="reduction",
        artifact_id="b" * 64,
    )
    state = SimpleNamespace(
        neighbors=source if method == "wnn" else None,
        connectivity_map=source if method == "snn" else None,
        cell_key="I",
    )
    monkeypatch.setattr(
        "scarf.datastore._operations.graph.read_assay_state",
        lambda *_args: state,
    )
    monkeypatch.setattr(
        "scarf.datastore._operations.graph.resolve_native_graph_inputs",
        lambda *_args: SimpleNamespace(
            coordinates=coordinates,
            cell_selection=selection,
        ),
    )
    monkeypatch.setattr(
        "scarf.datastore._operations.graph.validate_cell_selection_artifact",
        lambda *_args, **_kwargs: None,
    )
    plan = Mock(side_effect=AssertionError("artifact planning must not start"))
    monkeypatch.setattr(
        "scarf.datastore._operations.graph.plan_artifact",
        plan,
    )
    before = (
        list_artifacts(store.zw, scope="assay", assay="RNA"),
        list_artifacts(store.zw, scope="datastore"),
        tuple(sorted(store.zw.group_keys())),
    )

    with pytest.raises(ArtifactResolutionError) as caught:
        store.integrate_assays(
            ["RNA", "ADT"],
            label="corrupt_source",
            method=method,
        )

    assert caught.value.code == "corrupt_payload"
    assert caught.value.context["artifact_id"] == source.artifact_id
    plan.assert_not_called()
    assert (
        list_artifacts(store.zw, scope="assay", assay="RNA"),
        list_artifacts(store.zw, scope="datastore"),
        tuple(sorted(store.zw.group_keys())),
    ) == before
    assert "integratedGraphs" not in store.zw


def test_integrated_graph_index_rejects_corrupt_entries() -> None:
    store = _memory_graph_store()
    index = store.zw.create_group("integratedGraphs")
    graph = ArtifactRef(
        scope="datastore",
        kind="integrated_graph",
        artifact_id="8" * 64,
    )

    index.attrs["artifacts"] = []
    with pytest.raises(RuntimeError, match="artifact index is invalid"):
        integrated_graph_label(store, graph)

    index.attrs["artifacts"] = {"joint": "not-a-reference"}
    with pytest.raises(RuntimeError, match="index for 'joint' is invalid"):
        integrated_graph_label(store, graph)

    other = ArtifactRef(
        scope="datastore",
        kind="integrated_graph",
        artifact_id="9" * 64,
    )
    index.attrs["artifacts"] = {"joint": other.to_dict()}
    with pytest.raises(KeyError, match="not registered"):
        integrated_graph_label(store, graph)


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("invalid_index", "ANN query returned an invalid cell index"),
        ("short_stream", "Coordinate source contains 2 rows, expected 3"),
    ],
)
def test_query_neighbors_guards_ann_indices_and_coordinate_row_count(
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
    message: str,
) -> None:
    store = _memory_graph_store()
    coordinates = ArtifactRef(
        scope="assay",
        assay="RNA",
        kind="reduction",
        artifact_id="9" * 64,
    )
    ann = ArtifactRef(
        scope="assay",
        assay="RNA",
        kind="ann_index",
        artifact_id="a" * 64,
    )
    result = ArtifactRef(
        scope="assay",
        assay="RNA",
        kind="neighbors",
        artifact_id="b" * 64,
    )
    blocks = [
        np.zeros(
            (3 if failure == "invalid_index" else 2, 2),
            dtype=np.float32,
        )
    ]
    store._coordinate_source = Mock(return_value=(_CoordinateBlocks(blocks), 3, 2))

    def require(ref, _kind, **_kwargs):
        if ref == ann:
            return SimpleNamespace(
                inputs={"coordinates": coordinates.to_dict()},
                parameters={
                    "ann_metric": "l2",
                    "ann_ef": 50,
                    "parallel_threads": 1,
                },
                path="ann",
            )
        return SimpleNamespace(inputs={}, parameters={}, path="coordinates")

    store._require_complete_artifact = Mock(side_effect=require)
    store._plan_assay_artifact = Mock(
        return_value=SimpleNamespace(ref=result, reused=False)
    )
    store._resolve_ann_index = Mock(return_value=object())
    monkeypatch.setattr(
        "scarf.datastore._operations.graph.AnnIndexStage.configure",
        Mock(return_value=object()),
    )

    class InvalidQuery:
        def __init__(self, *_args):
            pass

        def query(self, block, *, self_indices):
            if failure == "invalid_index":
                indices = np.full((len(block), 1), 3, dtype=np.int64)
            else:
                indices = ((self_indices + 1) % 3).reshape(-1, 1)
            return (
                indices,
                np.zeros((len(block), 1), dtype=np.float32),
                0,
            )

    monkeypatch.setattr(
        "scarf.datastore._operations.graph.NeighborQueryStage",
        InvalidQuery,
    )

    with pytest.raises(ValueError, match=message):
        store.query_neighbors(
            ann,
            k=1,
            update_state=False,
        )


def test_reused_graph_stages_skip_expensive_compute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _memory_graph_store()
    coordinates = ArtifactRef(
        scope="assay",
        assay="RNA",
        kind="reduction",
        artifact_id="c" * 64,
    )
    ann = ArtifactRef(
        scope="assay",
        assay="RNA",
        kind="ann_index",
        artifact_id="d" * 64,
    )
    neighbors = ArtifactRef(
        scope="assay",
        assay="RNA",
        kind="neighbors",
        artifact_id="e" * 64,
    )
    connectivity = ArtifactRef(
        scope="assay",
        assay="RNA",
        kind="connectivity_map",
        artifact_id="f" * 64,
    )
    coordinate_source = _CoordinateBlocks([np.zeros((3, 2), dtype=np.float32)])
    store._coordinate_source = Mock(return_value=(coordinate_source, 3, 2))
    store._publish_current_artifact = Mock()
    fit_ann = Mock(side_effect=AssertionError("ANN fit must be skipped"))
    monkeypatch.setattr(
        "scarf.datastore._operations.graph.AnnIndexStage.fit",
        fit_ann,
    )

    store._plan_assay_artifact = Mock(
        return_value=SimpleNamespace(ref=ann, reused=True)
    )
    assert (
        store.build_ann_index(
            coordinates,
            update_state=False,
        )
        == ann
    )
    fit_ann.assert_not_called()

    store._require_complete_artifact = Mock(
        return_value=SimpleNamespace(
            inputs={"coordinates": coordinates.to_dict()},
            parameters={"ann_metric": "l2"},
            path="ann",
        )
    )
    store._plan_assay_artifact = Mock(
        return_value=SimpleNamespace(ref=neighbors, reused=True)
    )
    store._resolve_ann_index = Mock(
        side_effect=AssertionError("ANN index must not be loaded")
    )
    assert (
        store.query_neighbors(
            ann,
            k=2,
            update_state=False,
        )
        == neighbors
    )
    store._resolve_ann_index.assert_not_called()

    neighbor_path = "neighbor_source"
    neighbor_group = store.zw.create_group(neighbor_path)
    neighbor_group.create_array(
        "indices",
        data=np.array([[1, 2], [0, 2], [0, 1]], dtype=np.uint32),
    )
    store._require_complete_artifact = Mock(
        return_value=SimpleNamespace(path=neighbor_path)
    )
    store._plan_assay_artifact = Mock(
        return_value=SimpleNamespace(ref=connectivity, reused=True)
    )
    monkeypatch.setattr(
        "scarf.datastore._operations.graph.validate_distance_provenance",
        Mock(),
    )
    build_connectivity = Mock(
        side_effect=AssertionError("connectivity build must be skipped")
    )
    monkeypatch.setattr(
        "scarf.neighbors.graph.build_connectivity_arrays",
        build_connectivity,
    )

    assert (
        store.build_connectivity_map(
            neighbors,
            update_state=False,
        )
        == connectivity
    )
    build_connectivity.assert_not_called()


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("non_matrix", "WNN coordinate blocks must be matrices"),
        ("short_stream", "WNN coordinate stream did not cover every cell"),
        (
            "neighbor_count",
            "WNN neighbors and coordinates for RNA contain different cell counts",
        ),
        ("imported", "WNN coordinates must be reduction or batch_correction"),
    ],
)
def test_wnn_input_helpers_fail_before_integration_compute(
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
    message: str,
) -> None:
    store = _memory_graph_store(["RNA", "ADT"])
    selection = _add_complete_artifact(
        store,
        "cell_selection",
        assay=None,
    )
    states = {}
    coordinates_by_assay = {}
    for assay in ("RNA", "ADT"):
        coordinates = _add_complete_artifact(
            store,
            "imported_coordinates"
            if failure == "imported" and assay == "RNA"
            else "reduction",
            assay=assay,
        )
        coordinates_by_assay[assay] = coordinates
        indices = np.array([[1], [0]], dtype=np.uint32)
        if failure != "neighbor_count":
            indices = np.array([[1], [2], [0]], dtype=np.uint32)
        neighbors = _add_complete_artifact(
            store,
            "neighbors",
            assay=assay,
            arrays={"indices": indices},
        )
        states[assay] = SimpleNamespace(
            neighbors=neighbors,
            connectivity_map=None,
            cell_key="I",
        )

    monkeypatch.setattr(
        "scarf.datastore._operations.graph.read_assay_state",
        lambda _root, assay: states[assay],
    )
    monkeypatch.setattr(
        "scarf.datastore._operations.graph.resolve_native_graph_inputs",
        lambda _root, source: SimpleNamespace(
            coordinates=coordinates_by_assay[source.assay],
            cell_selection=selection,
        ),
    )
    monkeypatch.setattr(
        "scarf.datastore._operations.graph._validate_integration_source_payload",
        lambda *_args: 3,
    )
    monkeypatch.setattr(
        "scarf.datastore._operations.graph.validate_cell_selection_artifact",
        Mock(),
    )
    store._selection_artifacts_match = Mock(return_value=True)
    monkeypatch.setattr(
        "scarf.datastore._operations.graph.column_display",
        Mock(return_value=None),
    )
    integrated = ArtifactRef(
        scope="datastore",
        kind="integrated_graph",
        artifact_id="0" * 64,
    )
    plan_artifact = Mock(return_value=SimpleNamespace(ref=integrated, reused=False))
    monkeypatch.setattr(
        "scarf.datastore._operations.graph.plan_artifact",
        plan_artifact,
    )

    if failure == "non_matrix":
        blocks = [np.zeros(3, dtype=np.float32)]
    elif failure == "short_stream":
        blocks = [np.zeros((2, 2), dtype=np.float32)]
    else:
        blocks = [np.zeros((3, 2), dtype=np.float32)]
    store._coordinate_source = Mock(return_value=(_CoordinateBlocks(blocks), 3, 2))
    integrate = Mock(side_effect=AssertionError("WNN integration must not run"))
    monkeypatch.setattr(
        "scarf.neighbors.integration._wnn_integration_many",
        integrate,
    )

    def stored_paths(group: zarr.Group, prefix: str = "") -> tuple[str, ...]:
        paths: list[str] = []
        for name in group.keys():
            path = f"{prefix}/{name}" if prefix else name
            paths.append(path)
            child = group[name]
            if isinstance(child, zarr.Group):
                paths.extend(stored_paths(child, path))
        return tuple(sorted(paths))

    before = (
        list_artifacts(store.zw, scope="assay", assay="RNA"),
        list_artifacts(store.zw, scope="assay", assay="ADT"),
        list_artifacts(store.zw, scope="datastore"),
        stored_paths(store.zw),
    )

    with pytest.raises(ValueError, match=message):
        store.integrate_assays(
            ["RNA", "ADT"],
            label="invalid_wnn",
            method="wnn",
        )
    integrate.assert_not_called()
    if failure == "imported":
        plan_artifact.assert_not_called()
        assert (
            list_artifacts(store.zw, scope="assay", assay="RNA"),
            list_artifacts(store.zw, scope="assay", assay="ADT"),
            list_artifacts(store.zw, scope="datastore"),
            stored_paths(store.zw),
        ) == before
        assert "integratedGraphs" not in store.zw


def test_artifact_ann_stream_rejects_wrong_assay_before_lineage_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _memory_graph_store()
    neighbors = ArtifactRef(
        scope="assay",
        assay="ADT",
        kind="neighbors",
        artifact_id="1" * 64,
    )
    resolve_lineage = Mock(side_effect=AssertionError("lineage must not load"))
    monkeypatch.setattr(
        "scarf.datastore._operations.graph.resolve_native_graph_inputs",
        resolve_lineage,
    )
    with pytest.raises(ValueError, match="does not belong to from_assay"):
        store._load_artifact_ann_stream(
            "RNA",
            "I",
            True,
            neighbors,
        )
    resolve_lineage.assert_not_called()


def test_ann_storage_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _memory_graph_store()
    missing = ArtifactRef(
        scope="assay",
        assay="RNA",
        kind="ann_index",
        artifact_id="1" * 64,
    )
    with pytest.raises(ArtifactResolutionError) as caught:
        store._resolve_ann_index(missing, "l2", 2)
    assert caught.value.code == "missing_artifact"

    save_index = Mock()
    monkeypatch.setattr(
        "scarf.datastore._operations.graph.save_ann_index",
        save_index,
    )
    store.zarr_mode = "r"
    store._persist_ann_index(
        "read_only_ann",
        object(),
        ann_metric="l2",
        dimensions=2,
        element_count=3,
    )
    save_index.assert_not_called()

    store.zarr_mode = "r+"
    store._persist_ann_index(
        "writable_ann",
        object(),
        ann_metric="l2",
        dimensions=2,
        element_count=3,
    )
    assert "writable_ann" in store.zw
    save_index.assert_called_once()


def test_norm_statistics_populates_a_fully_empty_cache() -> None:
    store = _memory_graph_store()
    store.zw.create_group("stats")
    data = Mock()
    data.mean_and_std.return_value = (
        np.asarray([2.0, np.nan]),
        np.asarray([3.0, 0.0]),
    )

    mean, scale = store._load_or_compute_norm_stats("stats", data, "pca")

    np.testing.assert_allclose(mean, [2.0, 0.0])
    np.testing.assert_allclose(scale, [3.0, 1.0])
    np.testing.assert_allclose(store.zw["stats/mu"][:], mean)
    np.testing.assert_allclose(store.zw["stats/sigma"][:], scale)


def test_normalized_local_cache_cleans_up_after_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _memory_graph_store()
    store.zarr_loc = "remote"
    store.resources = None
    normalized = _add_complete_artifact(
        store,
        "normalized",
        arrays={"data": np.arange(6, dtype=np.float32).reshape(3, 2)},
    )

    existing = object()
    store._normalizedArtifactCache = {normalized: existing}
    store._resolve_local_cache_plan = Mock(
        side_effect=AssertionError("an existing cache must be reused")
    )
    with store._cache_normalized_artifact(normalized, True, 2):
        assert store._normalizedArtifactCache[normalized] is existing

    store._normalizedArtifactCache = {}
    store._resolve_local_cache_plan = Mock(return_value=(True, None, False))
    with pytest.raises(RuntimeError, match="Local cache path is missing"):
        with store._cache_normalized_artifact(normalized, True, 2):
            pass

    cache_base = tmp_path / "normalized_cache"
    cache_base.mkdir()
    staged_root = zarr.open_group(store=MemoryStore(), mode="w")
    staged = staged_root.create_array(
        "data",
        shape=(3, 2),
        dtype=np.float32,
        chunks=(2, 2),
    )
    store._resolve_local_cache_plan = Mock(return_value=(True, str(cache_base), True))
    monkeypatch.setattr(
        "scarf.datastore._operations.graph.create_or_open_staged_normed_array",
        Mock(return_value=staged),
    )

    def copy_array(source, target, **_kwargs):
        target[:, :] = source[:, :]

    monkeypatch.setattr(
        "scarf.datastore._operations.graph.copy_zarr_array",
        copy_array,
    )

    with pytest.raises(RuntimeError, match="downstream failure"):
        with store._cache_normalized_artifact(normalized, True, 2):
            assert normalized in store._normalizedArtifactCache
            raise RuntimeError("downstream failure")

    assert normalized not in store._normalizedArtifactCache
    assert not cache_base.exists()
