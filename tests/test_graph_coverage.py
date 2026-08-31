import hashlib
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
from scarf.embeddings.imported import write_imported_coordinates
from scarf.metadata import MetaData
from scarf.storage.artifacts import (
    ArtifactRef,
    artifact_path,
    fingerprint_array,
    list_artifacts,
    make_provenance,
    new_artifact_id,
)
from scarf.storage.errors import ArtifactResolutionError
from scarf.storage.selections import (
    read_stored_selection_mask,
    resolve_stored_selection_artifact,
)


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


def _metadata_snapshot(table: MetaData) -> dict[str, np.ndarray]:
    return {
        column: np.asarray(table.fetch_all(column)).copy() for column in table.columns
    }


def _assert_metadata_unchanged(
    table: MetaData,
    before: dict[str, np.ndarray],
) -> None:
    assert set(table.columns) == set(before)
    for column, values in before.items():
        np.testing.assert_array_equal(table.fetch_all(column), values)


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


def _add_test_cell_selection(
    store: _MemoryGraphStore,
    *,
    feature_values: np.ndarray | None = None,
) -> ArtifactRef:
    values = np.ones(3, dtype=bool)
    cells = store.zw.create_group("cellData")
    cells.create_array("ids", data=np.asarray(["c0", "c1", "c2"]))
    cells.create_array("I", data=values)
    if feature_values is not None:
        cells.create_array("gene", data=np.asarray(feature_values))
    store.cells = MetaData(cells)
    return resolve_stored_selection_artifact(
        store.zw,
        table_path="cellData",
        id_column="ids",
        source_column="I",
        scope="datastore",
        kind="cell_selection",
        operation="test_cell_selection",
        parameters={},
        inputs={},
    )


def _patch_trajectory_graph_resolution(
    monkeypatch: pytest.MonkeyPatch,
    graph: ArtifactRef,
    selection: ArtifactRef | None = None,
) -> None:
    strict_selection = selection is not None
    if selection is None:
        selection = ArtifactRef(
            scope="datastore",
            kind="cell_selection",
            artifact_id="0" * 64,
        )

    monkeypatch.setattr(
        "scarf.datastore._operations.trajectory.graph_cell_selection",
        lambda _root, selected: selection if selected == graph else None,
    )
    if not strict_selection:
        monkeypatch.setattr(
            "scarf.datastore._operations.trajectory.validate_stored_selection_integrity",
            Mock(),
        )
    monkeypatch.setattr(
        "scarf.datastore._operations.trajectory.resolve_graph_source_assay",
        lambda _root, selected, requested, **_kwargs: requested or "RNA",
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


def test_diffusion_operator_round_trip_and_explicit_imputation(
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
    selection = _add_test_cell_selection(store, feature_values=values)
    _patch_trajectory_graph_resolution(monkeypatch, graph_ref, selection)
    store.load_graph = Mock(return_value=graph)

    first_ref = store.run_diffusion_operator(graph_ref, t=1)
    operator = store.load_diffusion_operator(first_ref)
    assert isinstance(operator, coo_matrix)
    assert first_ref.kind == "diffusion_operator"
    status = store.inspect_artifact(first_ref)
    assert status.operation == "run_diffusion_operator"
    assert status.parameters == {"t": 1}
    assert set(status.inputs or {}) == {"connectivity_map", "cell_selection"}
    assert ArtifactRef.from_dict(status.inputs["connectivity_map"]) == graph_ref
    assert ArtifactRef.from_dict(status.inputs["cell_selection"]) == selection
    first = store.get_imputed(
        feature_name="gene",
        diffusion=first_ref,
    )
    np.testing.assert_allclose(first, np.array([3.0, 2.5, 1.5]))
    store.load_graph.assert_called_once_with(
        graph_ref,
        symmetric=True,
        upper_only=False,
    )

    reused_ref = store.run_diffusion_operator(graph_ref, t=1)
    second = store.get_imputed(feature_name="gene", diffusion=reused_ref)
    assert reused_ref == first_ref
    np.testing.assert_allclose(second, first)
    assert store.load_graph.call_count == 1

    squared_ref = store.run_diffusion_operator(graph_ref, t=2)
    squared = store.get_imputed(
        feature_name="gene",
        diffusion=squared_ref,
    )
    np.testing.assert_allclose(squared, np.array([2.0, 2.25, 2.75]))
    assert squared_ref != first_ref
    assert store.load_graph.call_count == 2

    invalidated_ref = store.run_diffusion_operator(
        graph_ref,
        t=1,
        invalidate_cache=True,
    )
    assert invalidated_ref not in {first_ref, squared_ref}
    assert store.load_graph.call_count == 3
    assert (
        len(
            list_artifacts(
                store.zw,
                scope="datastore",
                kind="diffusion_operator",
            )
        )
        == 3
    )


def test_diffusion_operator_loader_rejects_mismatched_lineage_and_payload(
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
    selection = _add_test_cell_selection(store)
    _patch_trajectory_graph_resolution(monkeypatch, graph_ref, selection)
    store.load_graph = Mock(return_value=graph)

    diffusion = store.run_diffusion_operator(graph_ref, t=1)
    group = store.zw[artifact_path(diffusion)]
    other_selection = ArtifactRef(
        scope="datastore",
        kind="cell_selection",
        artifact_id="f" * 64,
    )
    group.attrs["provenance"] = make_provenance(
        operation="run_diffusion_operator",
        parameters={"t": 1},
        inputs={
            "connectivity_map": graph_ref,
            "cell_selection": other_selection,
        },
    )
    with pytest.raises(ValueError, match="does not match its graph lineage"):
        store.load_diffusion_operator(diffusion)

    group.attrs["provenance"] = make_provenance(
        operation="run_diffusion_operator",
        parameters={"t": 1},
        inputs={"connectivity_map": graph_ref, "cell_selection": selection},
    )
    group["row"][0] = 3
    with pytest.raises(ValueError, match="sparse payload is malformed"):
        store.load_diffusion_operator(diffusion)


def test_diffusion_operator_content_tamper_is_rejected_and_not_reused(
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
    selection = _add_test_cell_selection(store)
    _patch_trajectory_graph_resolution(monkeypatch, graph_ref, selection)
    store.load_graph = Mock(return_value=graph)

    first = store.run_diffusion_operator(graph_ref, t=1)
    group = store.zw[artifact_path(first)]
    group["data"][0] = float(group["data"][0]) / 2.0

    with pytest.raises(ValueError, match="sparse payload is malformed"):
        store.load_diffusion_operator(first)
    replacement = store.run_diffusion_operator(graph_ref, t=1)
    assert replacement != first
    assert store.load_diffusion_operator(replacement).shape == (3, 3)


def test_read_only_diffusion_operator_only_reuses_persisted_artifacts(
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
    selection = _add_test_cell_selection(store)
    _patch_trajectory_graph_resolution(monkeypatch, graph_ref, selection)
    store.load_graph = Mock(return_value=graph)

    persisted = store.run_diffusion_operator(graph_ref, t=1)
    store.zarr_mode = "r"
    reused = store.run_diffusion_operator(graph_ref, t=1)
    assert reused == persisted
    assert store.load_diffusion_operator(reused).shape == (3, 3)
    assert store.load_graph.call_count == 1

    with pytest.raises(PermissionError, match=r"zarr_mode='r\+'"):
        store.run_diffusion_operator(graph_ref, t=2)
    with pytest.raises(PermissionError, match=r"zarr_mode='r\+'"):
        store.run_diffusion_operator(graph_ref, t=1, invalidate_cache=True)
    assert store.load_graph.call_count == 1


def test_filter_cells_open_bounds_composition_and_boundaries(
    isolated_toy_datastore: DataStore,
) -> None:
    store = isolated_toy_datastore
    attr = "RNA_nCounts"
    values = store.cells.fetch_all(attr)
    lower = float(values.min())
    upper = float(values.max())

    store.cells.reset_key("I")
    live_before = np.asarray(store.cells.fetch_all("I"), dtype=bool).copy()
    first = store.filter_cells(
        attrs=[attr],
        lows=[lower],
        highs=[None],
    )
    expected = values > lower
    np.testing.assert_array_equal(
        read_stored_selection_mask(
            store.zw,
            first,
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        ),
        expected,
    )
    np.testing.assert_array_equal(store.cells.fetch_all("I"), live_before)

    second = store.filter_cells(
        attrs=[attr],
        lows=[None],
        highs=[upper],
        cell_selection=first,
    )
    expected &= values < upper
    np.testing.assert_array_equal(
        read_stored_selection_mask(
            store.zw,
            second,
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        ),
        expected,
    )

    open_bounds = store.filter_cells(
        attrs=[attr],
        lows=[None],
        highs=[None],
    )
    assert read_stored_selection_mask(
        store.zw,
        open_bounds,
        kind="cell_selection",
        scope="datastore",
        assay=None,
        table_path="cellData",
    ).all()

    inclusive = store.filter_cells(
        attrs=[attr],
        lows=[lower],
        highs=[upper],
        keep_bounds=True,
    )
    assert read_stored_selection_mask(
        store.zw,
        inclusive,
        kind="cell_selection",
        scope="datastore",
        assay=None,
        table_path="cellData",
    ).all()


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
    selection_ref = store.snapshot_cell_selection("I")
    monkeypatch.setattr(
        "scarf.datastore._operations.embeddings.graph_cell_selection",
        lambda _root, selected: selection_ref if selected == graph_ref else None,
    )
    monkeypatch.setattr(store, "load_graph", load_graph)
    monkeypatch.setattr(store, "_get_ini_embed", get_initial)
    monkeypatch.setattr(
        store, "_graph_cell_selection", Mock(return_value=selection_ref)
    )
    monkeypatch.setattr("scarf.embeddings.sgtsne.run_sgtsne", runner)
    metadata_before = _metadata_snapshot(store.cells)

    tsne_ref = store.run_tsne(
        graph_ref,
        initialization_ref,
        symmetric_graph=True,
        graph_upper_only=True,
        parallel=True,
        nthreads=None,
        max_iter=20,
    )
    get_initial.assert_called_once_with(initialization_ref, graph_ref, 2)
    first_call = runner.call_args
    assert first_call.args[0] is graph
    np.testing.assert_array_equal(first_call.args[1], initial)
    assert first_call.kwargs["parallel"] is True
    assert first_call.kwargs["nthreads"] == store.nthreads
    assert first_call.kwargs["max_iter"] == 20
    first_ref = tsne_ref
    assert first_ref.kind == "embedding"
    np.testing.assert_allclose(
        store.load_artifact(first_ref)["values"][:],
        embedding.T,
    )
    _assert_metadata_unchanged(store.cells, metadata_before)

    store.run_tsne(
        graph_ref,
        initial,
        parallel=False,
        invalidate_cache=True,
    )
    assert runner.call_args.kwargs["nthreads"] == 1
    assert runner.call_args.kwargs["parallel"] is False

    with pytest.raises(ValueError, match="invalid shape"):
        store.run_tsne(
            graph_ref,
            np.zeros((2, 2)),
            tsne_dims=2,
        )

    runner.side_effect = FileNotFoundError("sgtsne missing")
    with pytest.raises(RuntimeError, match="SG-tSNE failed"):
        store.run_tsne(
            graph_ref,
            initial,
            parallel=True,
            nthreads=2,
            invalidate_cache=True,
        )
    assert runner.call_args.kwargs["nthreads"] == 2

    runner_calls = runner.call_count
    monkeypatch.setattr(sys, "platform", "win32")
    assert (
        store.run_tsne(
            graph_ref,
            initialization_ref,
            symmetric_graph=True,
            graph_upper_only=True,
            parallel=True,
            nthreads=None,
            max_iter=20,
        )
        == first_ref
    )
    assert runner.call_count == runner_calls
    _assert_metadata_unchanged(store.cells, metadata_before)


def test_integrate_assays_snn_writes_and_reuses_exact_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _memory_graph_store(["RNA", "ADT"])
    monkeypatch.setattr(
        "scarf.datastore._operations.graph.validate_stored_selection_integrity",
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
        list(sources.values()),
        method="snn",
        chunk_size=2,
    )
    second_ref = store.integrate_assays(
        list(sources.values()),
        method="snn",
        chunk_size=2,
    )

    assert first_ref.kind == "integrated_graph"
    assert second_ref == first_ref
    integrated_path = artifact_path(first_ref)
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
def test_integrate_assays_persists_exact_sources(
    monkeypatch: pytest.MonkeyPatch,
    method: str,
) -> None:
    import scarf.datastore._operations.graph as graph_operations

    assays = ("RNA", "ADT", "ATAC")
    store = _memory_graph_store(list(assays))
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
    captured_sources: dict[str, ArtifactRef] = {}
    captured_coordinates: dict[str, ArtifactRef] = {}
    coordinate_by_source: dict[ArtifactRef, ArtifactRef] = {}
    for assay in assays:
        source_kind = "connectivity_map" if method == "snn" else "neighbors"
        captured_sources[assay] = _add_complete_artifact(
            store,
            source_kind,
            assay=assay,
            arrays={"indices": neighbor_indices} if method == "wnn" else None,
        )
        captured_coordinates[assay] = _add_complete_artifact(
            store,
            "reduction",
            assay=assay,
        )
        coordinate_by_source[captured_sources[assay]] = captured_coordinates[assay]
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
        "validate_stored_selection_integrity",
        Mock(),
    )
    validate_source = Mock(return_value=3)
    monkeypatch.setattr(
        graph_operations,
        "_validate_integration_source_payload",
        validate_source,
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
            [
                [0.5, 0.3, 0.2],
                [0.2, 0.5, 0.3],
                [0.3, 0.2, 0.5],
            ],
            dtype=np.float32,
        )
        integrate_wnn = Mock(return_value=(merged, modality_weights))
        monkeypatch.setattr(
            "scarf.neighbors.integration._wnn_integration_many",
            integrate_wnn,
        )

    if method == "wnn":
        integrated = store.integrate_assays(list(captured_sources.values()))
    else:
        integrated = store.integrate_assays(
            list(captured_sources.values()),
            method=method,
        )
    reused = store.integrate_assays(
        list(captured_sources.values()),
        method=method,
    )

    assert store.inspect_artifact(integrated).complete
    assert reused == integrated
    assert [call.args[1] for call in validate_source.call_args_list] == list(
        captured_sources.values()
    ) * 2
    status = store.inspect_artifact(integrated)
    for index, assay in enumerate(assays):
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
        assert [modality[0] for modality in modalities] == list(assays)
        for _assay, indices, _coordinates in modalities:
            np.testing.assert_array_equal(indices, neighbor_indices)
        group = store.load_artifact(integrated)
        assert group.attrs["assays"] == list(assays)
        np.testing.assert_allclose(
            group["modality_weights"][:],
            modality_weights,
        )
        assert integrate_wnn.call_count == 1


def test_integrate_assays_validation_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _memory_graph_store(["RNA", "ADT"])
    selection = ArtifactRef(
        scope="datastore",
        kind="cell_selection",
        artifact_id="a" * 64,
    )
    sources = [
        ArtifactRef(
            scope="assay",
            assay=assay,
            kind="connectivity_map",
            artifact_id=artifact_id * 64,
        )
        for assay, artifact_id in (("RNA", "b"), ("ADT", "c"))
    ]

    with pytest.raises(ValueError, match="at least two assays"):
        store.integrate_assays(sources[:1], method="snn")

    with pytest.raises(TypeError, match="only ArtifactRef"):
        store.integrate_assays(["RNA", "ADT"], method="snn")

    with pytest.raises(TypeError, match="l2_normalize must be a boolean"):
        store.integrate_assays(
            sources,
            method="wnn",
            l2_normalize="yes",
        )

    with pytest.raises(ValueError, match="Method unknown not supported"):
        store.integrate_assays(sources, method="unknown")

    monkeypatch.setattr(
        "scarf.datastore._operations.graph.resolve_native_graph_inputs",
        lambda *_args: SimpleNamespace(cell_selection=selection),
    )
    monkeypatch.setattr(
        "scarf.datastore._operations.graph.validate_stored_selection_integrity",
        Mock(),
    )
    monkeypatch.setattr(
        "scarf.datastore._operations.graph._validate_integration_source_payload",
        Mock(return_value=3),
    )
    duplicate = ArtifactRef(
        scope="assay",
        assay="RNA",
        kind="connectivity_map",
        artifact_id="d" * 64,
    )
    with pytest.raises(ValueError, match="unique assay sources"):
        store.integrate_assays([sources[0], duplicate], method="snn")

    with pytest.raises(ArtifactResolutionError) as wrong_kind:
        store.integrate_assays(sources, method="wnn")
    assert wrong_kind.value.code == "wrong_kind"


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
    other_source = _add_complete_artifact(store, source_kind, assay="ADT")
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
    monkeypatch.setattr(
        "scarf.datastore._operations.graph.resolve_native_graph_inputs",
        lambda *_args: SimpleNamespace(
            coordinates=coordinates,
            cell_selection=selection,
        ),
    )
    monkeypatch.setattr(
        "scarf.datastore._operations.graph.validate_stored_selection_integrity",
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
            [source, other_source],
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
        )


def test_reused_graph_stages_skip_expensive_compute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _memory_graph_store()
    selection = _add_test_cell_selection(store)
    coordinate_values = np.zeros((3, 2), dtype=np.float32)
    coordinates = write_imported_coordinates(
        store.zw,
        assay="RNA",
        dimreduc_key="pca",
        role="pca",
        coordinates=coordinate_values,
        source_digest=hashlib.sha256(b"reused-graph-stage").digest(),
        payload_fingerprints={"data": fingerprint_array(coordinate_values)},
        source_cell_ids=np.asarray(store.zw["cellData/ids"][:]),
        cell_selection=selection,
        block_rows=2,
    )
    ann = _add_complete_artifact(
        store,
        "ann_index",
        inputs={"coordinates": coordinates},
    )
    neighbor_indices = np.array(
        [[1, 2], [0, 2], [0, 1]],
        dtype=np.uint32,
    )
    neighbors = _add_complete_artifact(
        store,
        "neighbors",
        inputs={"ann_index": ann, "coordinates": coordinates},
        arrays={"indices": neighbor_indices},
    )
    connectivity = ArtifactRef(
        scope="assay",
        assay="RNA",
        kind="connectivity_map",
        artifact_id="f" * 64,
    )
    coordinate_source = _CoordinateBlocks([np.zeros((3, 2), dtype=np.float32)])
    store._coordinate_source = Mock(return_value=(coordinate_source, 3, 2))
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
        )
        == neighbors
    )
    store._resolve_ann_index.assert_not_called()

    store._require_complete_artifact = Mock(
        return_value=SimpleNamespace(path=artifact_path(neighbors))
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
    neighbors_by_assay = {}
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
        neighbors_by_assay[assay] = neighbors
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
        "scarf.datastore._operations.graph.validate_stored_selection_integrity",
        Mock(),
    )
    store._selection_artifacts_match = Mock(return_value=True)
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
            list(neighbors_by_assay.values()),
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


def test_artifact_ann_stream_rejects_detached_ref_before_lineage_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _memory_graph_store()
    neighbors = ArtifactRef(
        scope="datastore",
        kind="neighbors",
        artifact_id="1" * 64,
    )
    resolve_lineage = Mock(side_effect=AssertionError("lineage must not load"))
    monkeypatch.setattr(
        "scarf.datastore._operations.graph.resolve_native_graph_inputs",
        resolve_lineage,
    )
    with pytest.raises(ValueError, match="neighbors must be assay-scoped"):
        store._load_artifact_ann_stream(neighbors, True)
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
