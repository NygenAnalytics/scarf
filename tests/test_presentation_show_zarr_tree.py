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
    inspect_artifact,
    make_provenance,
    new_artifact_id,
)
from scarf.storage.selections import resolve_generated_selection_artifact


class _PresentationStore(_PresentationOperationsMixin):
    def __init__(self, root: zarr.Group) -> None:
        self.zw = root

    def _require_writable(self, operation: str) -> None:
        if self.zw.read_only:
            raise PermissionError(
                f"{operation} requires a DataStore opened with zarr_mode='r+'"
            )


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
    parameters: dict[str, object] | None = None,
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


_BRANCH_DEPTH_0 = (
    "/branch\n"
    "├── nested\n"
    "└── values (2, 3) float32\n"
    "\n"
    "  values: shape=(2, 3), dtype=float32, chunks=(1, 3)\n"
)
_BRANCH_DEPTH_1 = (
    "/branch\n"
    "├── nested\n"
    "│   └── deep (4,) int16\n"
    "└── values (2, 3) float32\n"
    "\n"
    "  values: shape=(2, 3), dtype=float32, chunks=(1, 3)\n"
)
_ROOT_DEPTH_1 = (
    "/\n"
    "├── branch\n"
    "│   ├── nested\n"
    "│   └── values (2, 3) float32\n"
    "├── root_values (2,) bool\n"
    "└── sibling\n"
    "    └── hidden (1,) uint8\n"
    "\n"
    "  root_values: shape=(2,), dtype=bool, chunks=(1,)\n"
)


@pytest.mark.parametrize("start", ["branch", "/branch", "branch/", "/branch/"])
def test_show_zarr_tree_normalizes_path_and_filters_to_subtree(
    start: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    store, _backing = _presentation_store()

    store.show_zarr_tree(start=start, depth=0)

    assert capsys.readouterr().out == _BRANCH_DEPTH_0


def test_show_zarr_tree_depth_controls_nested_output(
    capsys: pytest.CaptureFixture[str],
) -> None:
    store, _backing = _presentation_store()

    store.show_zarr_tree(start="branch", depth=1)

    assert capsys.readouterr().out == _BRANCH_DEPTH_1


def test_show_zarr_tree_formats_arrays_without_mutating_attributes(
    capsys: pytest.CaptureFixture[str],
) -> None:
    store, _backing = _presentation_store()
    branch_attrs = dict(store.zw["branch"].attrs)
    array_attrs = dict(store.zw["branch/values"].attrs)

    store.show_zarr_tree(start="branch", depth=1)

    assert capsys.readouterr().out == _BRANCH_DEPTH_1
    assert dict(store.zw["branch"].attrs) == branch_attrs
    assert dict(store.zw["branch/values"].attrs) == array_attrs


@pytest.mark.parametrize(
    ("start", "error_type", "message"),
    [
        ("does_not_exist", KeyError, "does_not_exist"),
        (
            "branch/values",
            TypeError,
            "Expected Zarr group at 'branch/values', got Array",
        ),
    ],
)
def test_show_zarr_tree_rejects_invalid_start_path(
    start: str,
    error_type: type[Exception],
    message: str,
) -> None:
    store, _backing = _presentation_store()

    with pytest.raises(error_type, match=message):
        store.show_zarr_tree(start=start, depth=1)


def test_show_zarr_tree_rejects_a_negative_depth() -> None:
    store, _backing = _presentation_store()

    with pytest.raises(ValueError, match="max_depth must be None or >= 0"):
        store.show_zarr_tree(depth=-1)


def test_show_zarr_tree_operates_on_read_only_memory_store(
    capsys: pytest.CaptureFixture[str],
) -> None:
    _writable, backing = _presentation_store()
    root = zarr.open_group(store=backing.with_read_only(True), mode="r")
    store = _PresentationStore(root)
    before_root_attrs = dict(root.attrs)
    before_branch_attrs = dict(root["branch"].attrs)

    store.show_zarr_tree(start="/", depth=1)

    assert capsys.readouterr().out == _ROOT_DEPTH_1
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
        # Export reads column attributes to declare membership columns.
        _get_array=Mock(return_value=SimpleNamespace(attrs={})),
    )

    exported = store.to_anndata(feature_indexes=[])

    assert exported.shape == (2, 0)
    assert list(exported.obs_names) == ["c0", "c1"]
    assert "scarf" not in exported.uns
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
    agreement = presentation.neighbor_label_agreement

    def small_blocks(edges, label_codes, *, k):
        # Small blocks exercise several edge blocks and a partial final block.
        return agreement(edges, label_codes, k=k, block_edges=3 * k + 2)

    monkeypatch.setattr(presentation, "neighbor_label_agreement", small_blocks)

    ref = store.calc_membership_strength(clusters, graph_ref)

    stored = np.asarray(store.zw[artifact_path(ref)]["values"][:])
    # The share of each cell's neighbours that carry its own label; NaN
    # labels count as one label.
    neighbour_labels = labels[neighbours]
    own = labels[:, None]
    shared = (neighbour_labels == own) | (pd.isna(neighbour_labels) & pd.isna(own))
    reference = (np.count_nonzero(shared, axis=1) / k).round(3)
    assert stored.dtype == np.float64
    assert stored.tobytes() == reference.tobytes()
    status = inspect_artifact(store.zw, ref)
    assert status.revision == 2
    assert status.parameters == {"algorithm_version": 2, "decimals": 3}


def test_membership_strength_needs_a_writable_store_only_for_new_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, backing = _presentation_store()
    selection = _write_cell_selection(store.zw, np.ones(4, dtype=bool))
    edges = np.asarray([[0, 1], [1, 0], [2, 3], [3, 2]], dtype=np.uint64)
    graph_ref = _write_complete_artifact(
        store.zw, "connectivity_map", arrays={"edges": edges}
    )
    clusters, other_clusters = (
        _write_complete_artifact(
            store.zw,
            "cluster_labels",
            inputs={"cell_selection": selection},
            arrays={"values": values},
        )
        for values in (np.asarray([0, 0, 1, 1]), np.asarray([0, 1, 1, 1]))
    )
    _patch_graph_resolution(monkeypatch, graph_ref, selection=selection)
    store._get_graph_ncells_k = Mock(return_value=(4, 1))
    ref = store.calc_membership_strength(clusters, graph_ref)
    read_only = _PresentationStore(zarr.open_group(store=backing, mode="r"))
    read_only._get_graph_ncells_k = Mock(return_value=(4, 1))

    assert read_only.calc_membership_strength(clusters, graph_ref) == ref
    with pytest.raises(PermissionError, match="calc_membership_strength requires"):
        read_only.calc_membership_strength(other_clusters, graph_ref)


def test_membership_strength_rejects_non_graph_kinds_before_reuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scarf.datastore._operations.presentation as presentation

    store, _backing = _presentation_store()
    neighbors = _write_complete_artifact(store.zw, "neighbors")
    clusters = _write_complete_artifact(store.zw, "cluster_labels")

    def refuse_lookup(*_args, **_kwargs):
        raise AssertionError("a non-graph input must not be inspected or planned")

    monkeypatch.setattr(presentation, "inspect_artifact", refuse_lookup)
    monkeypatch.setattr(presentation, "plan_cell_data_artifact", refuse_lookup)
    with pytest.raises(ValueError, match="connectivity_map or integrated_graph"):
        store.calc_membership_strength(clusters, neighbors)


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

    with pytest.raises(
        ValueError, match="Cluster labels do not match the graph cell selection"
    ):
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


def _smart_label_inputs(
    store: _PresentationStore,
    to_relabel: np.ndarray,
    base_labels: np.ndarray,
) -> tuple[ArtifactRef, ArtifactRef]:
    selection = _write_cell_selection(store.zw, np.ones(len(to_relabel), dtype=bool))
    clusters = _write_complete_artifact(
        store.zw,
        "cluster_labels",
        assay=None,
        inputs={"cell_selection": selection},
        arrays={"values": to_relabel},
    )
    base = _write_complete_artifact(
        store.zw,
        "hto_identity",
        assay=None,
        inputs={"cell_selection": selection},
        arrays={"values": base_labels},
    )
    return clusters, base


def test_smart_label_suffixes_continue_past_z_without_merging_labels() -> None:
    store, _backing = _presentation_store()
    # Cluster c holds c + 1 cells, so clusters rank by size from 39 down to 0.
    to_relabel = np.repeat(np.arange(40), np.arange(1, 41))
    clusters, base = _smart_label_inputs(
        store, to_relabel, np.full(len(to_relabel), "T")
    )

    ref = store.smart_label(clusters, base)

    names = store.zw[artifact_path(ref)]["values"][:].astype(str)
    by_label = dict(zip(to_relabel.tolist(), names.tolist(), strict=True))
    letters = [chr(ord("a") + index) for index in range(26)]
    suffixes = letters + [f"a{letter}" for letter in letters[:14]]
    # Larger clusters take earlier suffixes; z continues as aa, ab, ...
    assert by_label == {39 - rank: f"T{suffix}" for rank, suffix in enumerate(suffixes)}
    # The frozen version parameter of earlier releases and how it names labels.
    assert inspect_artifact(store.zw, ref).parameters == {
        "algorithm_version": 3,
        "suffix_style": "lowercase_letter",
    }


def test_smart_label_rejects_hyphen_joined_names_that_collide() -> None:
    store, _backing = _presentation_store()
    # Cluster 1 absorbs base label X and becomes T-Xa, the name of cluster 2.
    clusters, base = _smart_label_inputs(
        store,
        np.asarray([1] * 10 + [2] * 10),
        np.asarray(["T"] * 8 + ["X"] * 2 + ["T-X"] * 10),
    )

    with pytest.raises(ValueError, match="'T-Xa'"):
        store.smart_label(clusters, base)


def test_smart_label_of_an_earlier_release_is_reused() -> None:
    store, backing = _presentation_store()
    clusters, base = _smart_label_inputs(
        store, np.asarray([0, 0, 1, 1]), np.asarray(["A", "A", "B", "B"])
    )
    selection = inspect_artifact(store.zw, clusters).input_ref("cell_selection")
    # What releases before 1.0.0 stored for the same labels, such as imported
    # labels or label snapshots, whose identities this release keeps. The
    # values are unchanged and smart_label has no revision, so the label is
    # an exact match.
    earlier = _write_complete_artifact(
        store.zw,
        "smart_label",
        assay=None,
        operation="smart_label",
        parameters={"algorithm_version": 3, "suffix_style": "lowercase_letter"},
        inputs={"values": clusters, "base_labels": base, "cell_selection": selection},
        arrays={"values": np.asarray(["Aa", "Aa", "Ba", "Ba"])},
    )
    read_only = _PresentationStore(zarr.open_group(store=backing, mode="r"))

    assert read_only.smart_label(clusters, base) == earlier
    assert store.smart_label(clusters, base) == earlier
    assert inspect_artifact(store.zw, earlier).is_current


def test_smart_label_needs_a_writable_store_only_for_new_results() -> None:
    store, backing = _presentation_store()
    clusters, base = _smart_label_inputs(
        store, np.asarray([0, 0, 1, 1]), np.asarray(["A", "A", "B", "B"])
    )
    other_clusters, _ = _smart_label_inputs(
        store, np.asarray([0, 1, 1, 1]), np.asarray(["A", "A", "B", "B"])
    )
    ref = store.smart_label(clusters, base)
    read_only = _PresentationStore(zarr.open_group(store=backing, mode="r"))

    assert read_only.smart_label(clusters, base) == ref
    with pytest.raises(PermissionError, match="smart_label requires"):
        read_only.smart_label(other_clusters, base)


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
