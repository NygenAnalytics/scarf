"""Regression tests for DataStore presentation helpers."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import zarr
from zarr.storage import MemoryStore

from scarf.datastore._operations.presentation import _PresentationOperationsMixin
from scarf.matrix import ChunkedArray
from scarf.storage.artifacts import (
    ArtifactRef,
    artifact_path,
    make_provenance,
    new_artifact_id,
)
from scarf.storage.selections import resolve_generated_selection_artifact


class _PresentationStore(_PresentationOperationsMixin):
    def __init__(self, root: zarr.Group) -> None:
        self.zw = root


def _presentation_store() -> tuple[_PresentationStore, MemoryStore]:
    backing = MemoryStore()
    root = zarr.open_group(store=backing, mode="w")
    root.attrs["title"] = "small store"
    root.create_array(
        "root_values",
        data=np.asarray([True, False], dtype=np.bool_),
        chunks=(1,),
    )
    branch = root.create_group("branch")
    branch.attrs.update({"label": "selected", "rank": 1})
    values = branch.create_array(
        "values",
        data=np.arange(6, dtype=np.float32).reshape(2, 3),
        chunks=(1, 3),
    )
    values.attrs["units"] = "counts"
    nested = branch.create_group("nested")
    nested.create_array(
        "deep",
        data=np.arange(4, dtype=np.int16),
        chunks=(2,),
    )
    root.create_group("sibling").create_array(
        "hidden",
        data=np.asarray([1], dtype=np.uint8),
    )
    return _PresentationStore(root), backing


def _write_complete_artifact(
    root: zarr.Group,
    kind: str,
    *,
    assay: str | None = "RNA",
    inputs: dict[str, object] | None = None,
    arrays: dict[str, np.ndarray] | None = None,
    operation: str | None = None,
) -> ArtifactRef:
    ref = ArtifactRef(
        scope="assay" if assay is not None else "datastore",
        assay=assay,
        kind=kind,
        artifact_id=new_artifact_id(),
    )
    group = root.create_group(artifact_path(ref))
    group.attrs.update(
        {
            "artifact_id": ref.artifact_id,
            "kind": kind,
            "provenance": make_provenance(
                operation=operation or f"test_{kind}",
                parameters={},
                inputs=inputs or {},
            ),
            "execution_options": {},
            "complete": True,
        }
    )
    for name, values in (arrays or {}).items():
        group.create_array(name, data=values)
    return ref


def _write_cell_selection(
    root: zarr.Group,
    values: np.ndarray,
    *,
    operation: str = "test_cell_selection",
) -> ArtifactRef:
    if "cellData" not in root:
        cell_data = root.create_group("cellData")
        cell_data.create_array(
            "ids",
            data=np.asarray([f"cell_{index}" for index in range(len(values))]),
        )
    cell_data = root["cellData"]
    return resolve_generated_selection_artifact(
        root,
        scope="datastore",
        kind="cell_selection",
        values=np.asarray(values, dtype=bool),
        row_ids=np.asarray(cell_data["ids"][:]),
        operation=operation,
        parameters={},
        inputs={},
        source_column="I",
    )[0]


def _patch_graph_resolution(
    monkeypatch: pytest.MonkeyPatch,
    graph_ref: ArtifactRef,
    *,
    selection: ArtifactRef | None = None,
) -> None:
    selection_ref = selection or ArtifactRef(
        scope="datastore",
        kind="cell_selection",
        artifact_id="0" * 64,
    )
    monkeypatch.setattr(
        "scarf.datastore._operations.presentation.resolve_graph_source_assay",
        lambda _root, graph, requested, **_kwargs: (
            requested or "RNA" if graph == graph_ref else "RNA"
        ),
    )
    monkeypatch.setattr(
        "scarf.datastore._operations.presentation.graph_cell_selection",
        lambda _root, graph: selection_ref if graph == graph_ref else None,
    )


@pytest.mark.parametrize("start", ["branch", "/branch", "branch/", "/branch/"])
def test_show_zarr_tree_normalizes_path_and_filters_to_subtree(
    start: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    store, _backing = _presentation_store()

    store.show_zarr_tree(start=start, depth=0)
    captured = capsys.readouterr().out

    assert "/branch" in captured
    assert "nested" in captured
    assert "values" in captured
    assert "deep" not in captured
    assert "root_values" not in captured
    assert "sibling" not in captured


def test_show_zarr_tree_depth_controls_nested_output(
    capsys: pytest.CaptureFixture[str],
) -> None:
    store, _backing = _presentation_store()

    store.show_zarr_tree(start="branch", depth=0)
    shallow = capsys.readouterr().out
    store.show_zarr_tree(start="branch", depth=1)
    deep = capsys.readouterr().out

    assert "deep" not in shallow
    assert "deep (4,) int16" in deep


def test_show_zarr_tree_formats_arrays_without_mutating_attributes(
    capsys: pytest.CaptureFixture[str],
) -> None:
    store, _backing = _presentation_store()
    branch_attrs = dict(store.zw["branch"].attrs)
    array_attrs = dict(store.zw["branch/values"].attrs)

    store.show_zarr_tree(start="branch", depth=1)
    captured = capsys.readouterr().out

    assert "values (2, 3) float32" in captured
    assert "values: shape=(2, 3), dtype=float32, chunks=(1, 3)" in captured
    assert dict(store.zw["branch"].attrs) == branch_attrs
    assert dict(store.zw["branch/values"].attrs) == array_attrs


@pytest.mark.parametrize(
    ("start", "error_type"),
    [
        ("does_not_exist", KeyError),
        ("branch/values", TypeError),
    ],
)
def test_show_zarr_tree_rejects_invalid_start_path(
    start: str,
    error_type: type[Exception],
) -> None:
    store, _backing = _presentation_store()

    with pytest.raises(error_type):
        store.show_zarr_tree(start=start, depth=1)


@pytest.mark.parametrize(
    ("depth", "error_type"),
    [
        (-1, ValueError),
        ("one", TypeError),
    ],
)
def test_show_zarr_tree_rejects_invalid_depth(
    depth: object,
    error_type: type[Exception],
) -> None:
    store, _backing = _presentation_store()

    with pytest.raises(error_type):
        store.show_zarr_tree(depth=depth)


def test_show_zarr_tree_operates_on_read_only_memory_store(
    capsys: pytest.CaptureFixture[str],
) -> None:
    _writable, backing = _presentation_store()
    root = zarr.open_group(store=backing.with_read_only(True), mode="r")
    store = _PresentationStore(root)
    before_root_attrs = dict(root.attrs)
    before_branch_attrs = dict(root["branch"].attrs)

    store.show_zarr_tree(start="/", depth=1)
    captured = capsys.readouterr().out

    assert {"branch", "root_values", "sibling"} <= set(captured.split())
    assert dict(root.attrs) == before_root_attrs
    assert dict(root["branch"].attrs) == before_branch_attrs


def test_to_anndata_exports_an_empty_feature_selection() -> None:
    store, _backing = _presentation_store()
    features = SimpleNamespace(
        N=2,
        columns=["ids", "names"],
        to_pandas_dataframe=Mock(
            return_value=pd.DataFrame({"ids": ["f0", "f1"], "names": ["g0", "g1"]})
        ),
        fetch_all=Mock(return_value=np.asarray(["f0", "f1"])),
    )
    assay = SimpleNamespace(
        feats=features,
        rawData=ChunkedArray(np.asarray([[1, 2], [3, 4]], dtype=np.uint32)),
        nthreads=1,
        name="RNA",
    )
    store._get_assay = Mock(return_value=assay)
    store.cells = SimpleNamespace(
        columns=["ids"],
        active_index=Mock(return_value=np.asarray([0, 1])),
        to_pandas_dataframe=Mock(return_value=pd.DataFrame({"ids": ["c0", "c1"]})),
    )

    exported = store.to_anndata(feature_indexes=[])

    assert exported.shape == (2, 0)
    assert list(exported.obs_names) == ["c0", "c1"]
    store.cells.active_index.assert_called_once_with("I")


@pytest.mark.parametrize("label_kind", ["integer", "string", "float_with_nan"])
def test_membership_strength_matches_reference_counts(
    monkeypatch: pytest.MonkeyPatch,
    label_kind: str,
) -> None:
    import scarf.datastore._operations.presentation as presentation

    rng = np.random.default_rng(17)
    n_cells, k = 61, 7
    store, _backing = _presentation_store()
    selection = _write_cell_selection(store.zw, np.ones(n_cells, dtype=bool))
    neighbours = rng.integers(0, n_cells, size=(n_cells, k))
    edges = np.stack(
        (np.repeat(np.arange(n_cells), k), neighbours.ravel()),
        axis=1,
    ).astype(np.uint64)
    graph_ref = _write_complete_artifact(
        store.zw,
        "connectivity_map",
        arrays={"edges": edges},
    )
    codes = rng.integers(0, 4, size=n_cells)
    labels = {
        "integer": codes.astype(np.int64),
        "string": np.asarray(["a", "b", "c", "d"])[codes],
        "float_with_nan": np.asarray([0.5, 1.5, np.nan, np.nan])[codes],
    }[label_kind]
    clusters = _write_complete_artifact(
        store.zw,
        "cluster_labels",
        inputs={"cell_selection": selection},
        arrays={"values": labels},
    )
    store._get_graph_ncells_k = Mock(return_value=(n_cells, k))
    _patch_graph_resolution(monkeypatch, graph_ref, selection=selection)
    # Small blocks exercise several edge blocks and a partial final block.
    monkeypatch.setattr(presentation, "_MEMBERSHIP_BLOCK_EDGES", 3 * k + 2)

    ref = store.calc_membership_strength(clusters, graph_ref)

    stored = np.asarray(store.zw[artifact_path(ref)]["values"][:])
    reference = np.asarray(
        [
            pd.Series(row).value_counts(dropna=False).iloc[0] / k
            for row in labels[neighbours]
        ],
        dtype=np.float64,
    ).round(3)
    assert stored.dtype == np.float64
    assert stored.tobytes() == reference.tobytes()


def test_membership_strength_rejects_a_different_graph_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, _backing = _presentation_store()
    graph_ref = _write_complete_artifact(store.zw, "connectivity_map")
    graph_selection = _write_cell_selection(
        store.zw,
        np.asarray([True, True]),
        operation="graph_selection",
    )
    cluster_selection = _write_cell_selection(
        store.zw,
        np.asarray([True, True]),
        operation="cluster_selection",
    )
    clusters = _write_complete_artifact(
        store.zw,
        "cluster_labels",
        inputs={"cell_selection": cluster_selection},
        arrays={"values": np.asarray([0, 1])},
    )
    store._get_graph_ncells_k = Mock(return_value=(2, 1))
    _patch_graph_resolution(monkeypatch, graph_ref, selection=graph_selection)

    with pytest.raises(ValueError, match="do not match"):
        store.calc_membership_strength(
            clusters,
            graph_ref,
        )


def test_smart_label_returns_an_artifact_and_handles_unmatched_base_labels() -> None:
    store, _backing = _presentation_store()
    selection = _write_cell_selection(store.zw, np.asarray([True, True, True]))
    clusters = _write_complete_artifact(
        store.zw,
        "cluster_labels",
        assay=None,
        inputs={"cell_selection": selection},
        arrays={"values": np.asarray(["a", "a", "a"])},
    )
    base = _write_complete_artifact(
        store.zw,
        "hto_identity",
        assay=None,
        inputs={"cell_selection": selection},
        arrays={"values": np.asarray(["X", "X", "Y"])},
    )

    first = store.smart_label(clusters, base)
    second = store.smart_label(clusters, base)

    assert first == second
    assert first.kind == "smart_label"
    assert store.zw[artifact_path(first)]["values"][:].tolist() == [
        "X-Ya",
        "X-Ya",
        "X-Ya",
    ]
    assert set(store.zw["cellData"].array_keys()) == {"ids"}


def test_prepare_cluster_tree_rejects_unresolved_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, _backing = _presentation_store()
    graph_ref = ArtifactRef(
        scope="assay",
        assay="RNA",
        kind="connectivity_map",
        artifact_id="7" * 64,
    )
    _patch_graph_resolution(monkeypatch, graph_ref)
    with pytest.raises(TypeError, match="clusters must be an ArtifactRef"):
        store._prepare_cluster_tree(
            graph=graph_ref,
            clusters="clusters",
        )
    wrong_clusters = ArtifactRef(
        scope="assay",
        assay="RNA",
        kind="cluster_labels",
        artifact_id="6" * 64,
    )
    with pytest.raises(ValueError, match="cluster_cut artifact"):
        store._prepare_cluster_tree(graph=graph_ref, clusters=wrong_clusters)


def test_artifact_cluster_tree_requires_cut_and_hierarchy_provenance() -> None:
    store, _backing = _presentation_store()
    graph_ref = ArtifactRef(
        scope="assay",
        assay="RNA",
        kind="connectivity_map",
        artifact_id="8" * 64,
    )

    wrong_clusters = ArtifactRef(
        scope="assay",
        assay="RNA",
        kind="cluster_labels",
        artifact_id="9" * 64,
    )
    with pytest.raises(ValueError, match="cluster_cut artifact"):
        store._prepare_artifact_cluster_tree(
            graph_ref=graph_ref,
            clusters_ref=wrong_clusters,
            from_assay="RNA",
            fill_by_value=None,
            invalidate_cache=False,
        )

    cut_ref = _write_complete_artifact(
        store.zw,
        "cluster_cut",
        inputs={"connectivity_map": graph_ref},
        arrays={"labels": np.asarray([0, 1])},
        operation="cut_paris_hierarchy",
    )
    with pytest.raises(ValueError, match="no hierarchy input"):
        store._prepare_artifact_cluster_tree(
            graph_ref=graph_ref,
            clusters_ref=cut_ref,
            from_assay="RNA",
            fill_by_value=None,
            invalidate_cache=False,
        )
