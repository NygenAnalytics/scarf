import copy
import hashlib
from typing import Any

import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

from scarf.datastore.datastore import DataStore
from scarf.embeddings.imported import (
    validate_imported_embedding_artifact,
    write_imported_coordinates,
    write_imported_embedding,
)
from scarf.graph.state import (
    AssayState,
    read_assay_state,
    validate_imported_coordinates_artifact,
    validate_neighbors_artifact_selection,
)
from scarf.graph.errors import IncompatibleAnalysisStateError
from scarf.storage.errors import ArtifactResolutionError
from scarf.storage.artifacts import (
    artifact_group,
    artifact_path,
    fingerprint_array,
    fingerprint_strings,
    list_artifacts,
)
from scarf.storage.budget import ResourceBudget
from scarf.storage.refs import ArtifactRef
from scarf.storage.selections import resolve_selection_artifact

_SOURCE_DIGEST = hashlib.sha256(b"source-rds").digest()


def _root_with_selection(
    mask: np.ndarray | None = None,
    *,
    store: Any | None = None,
) -> tuple[zarr.Group, ArtifactRef, np.ndarray, np.ndarray]:
    root = zarr.open_group(
        store=MemoryStore() if store is None else store,
        mode="w",
    )
    root.create_group("RNA")
    cell_ids = np.array([f"cell_{index}" for index in range(8)])
    selection = (
        np.ones(len(cell_ids), dtype=bool)
        if mask is None
        else np.asarray(mask, dtype=bool)
    )
    cell_data = root.create_group("cellData")
    cell_data.create_array("ids", data=cell_ids)
    cell_data.create_array("names", data=cell_ids)
    cell_data.create_array("I", data=selection)
    ref = resolve_selection_artifact(
        root,
        scope="datastore",
        kind="cell_selection",
        values=selection,
        row_ids=cell_ids,
        operation="manual_selection",
        parameters={},
        inputs={},
        source_column="I",
    )
    return root, ref, cell_ids, selection


def _fingerprints(
    *,
    data: np.ndarray,
    loadings: np.ndarray | None = None,
    feature_ids: np.ndarray | None = None,
    stdev: np.ndarray | None = None,
) -> dict[str, str]:
    values = {"data": fingerprint_array(data)}
    if loadings is not None:
        values["loadings"] = fingerprint_array(loadings)
    if feature_ids is not None:
        values["feature_ids"] = fingerprint_strings(feature_ids)
    if stdev is not None:
        values["stdev"] = fingerprint_array(stdev)
    return values


def _graph_store(root: zarr.Group) -> DataStore:
    store = DataStore.__new__(DataStore)
    store.z = root
    store.workspace = None
    store.zarr_mode = "r+"
    store.storageProfile = "fast_local"
    store.resources = ResourceBudget(memoryBytes=64 * 1024 * 1024, workers=1)
    store.nthreads = 1
    store._defaultAssay = "RNA"
    return store


def _write_coordinate_fixture(
    root: zarr.Group,
    selection: ArtifactRef,
    cell_ids: np.ndarray,
    mask: np.ndarray,
    *,
    include_optional: bool = False,
) -> tuple[ArtifactRef, np.ndarray]:
    coordinates = np.arange(
        int(mask.sum()) * 3,
        dtype=np.float32,
    ).reshape(int(mask.sum()), 3)
    kwargs: dict[str, Any] = {}
    if include_optional:
        loadings = np.arange(12, dtype=np.float64).reshape(4, 3)
        feature_ids = np.asarray([f"gene_{index}" for index in range(4)])
        stdev = np.asarray([3.0, 2.0, 1.0], dtype=np.float64)
        kwargs.update(
            {
                "loadings": loadings,
                "feature_ids": feature_ids,
                "stdev": stdev,
                "payload_fingerprints": _fingerprints(
                    data=coordinates,
                    loadings=loadings,
                    feature_ids=feature_ids,
                    stdev=stdev,
                ),
            }
        )
    else:
        kwargs["payload_fingerprints"] = {"data": fingerprint_array(coordinates)}
    ref = write_imported_coordinates(
        root,
        assay="RNA",
        dimreduc_key="pca",
        role="pca",
        coordinates=coordinates,
        source_digest=_SOURCE_DIGEST,
        source_cell_ids=cell_ids[mask],
        cell_selection=selection,
        cell_key="I",
        block_rows=2,
        **kwargs,
    )
    return ref, coordinates


def _write_embedding_fixture(
    root: zarr.Group,
    selection: ArtifactRef,
    cell_ids: np.ndarray,
    mask: np.ndarray,
) -> tuple[ArtifactRef, np.ndarray]:
    coordinates = np.arange(
        int(mask.sum()) * 2,
        dtype=np.float32,
    ).reshape(int(mask.sum()), 2)
    ref = write_imported_embedding(
        root,
        assay="RNA",
        dimreduc_key="umap",
        role="umap",
        coordinates=coordinates,
        source_digest=_SOURCE_DIGEST,
        payload_fingerprints={"values": fingerprint_array(coordinates)},
        source_cell_ids=cell_ids[mask],
        cell_selection=selection,
        cell_key="I",
        block_rows=2,
    )
    return ref, coordinates


def _tamper_artifact_attribute(
    root: zarr.Group,
    ref: ArtifactRef,
    attribute: str,
    path: tuple[str, ...],
    value: Any,
) -> None:
    group = artifact_group(root, ref)
    payload = copy.deepcopy(group.attrs[attribute])
    target = payload
    for name in path[:-1]:
        target = target[name]
    target[path[-1]] = value
    group.attrs[attribute] = payload


def test_imported_coordinates_write_blockwise_with_honest_provenance() -> None:
    root, selection, cell_ids, mask = _root_with_selection()
    selected_ids = cell_ids[mask]
    coordinates = np.arange(24, dtype=np.float32).reshape(8, 3)
    loadings = np.arange(15, dtype=np.float64).reshape(5, 3)
    feature_ids = np.array([f"gene_{index}" for index in range(5)])
    stdev = np.array([3.0, 2.0, 1.0], dtype=np.float64)
    pulls: list[int] = []

    def coordinate_blocks():
        for start in range(0, len(coordinates), 3):
            pulls.append(start)
            yield coordinates[start : start + 3]

    ref = write_imported_coordinates(
        root,
        assay="RNA",
        dimreduc_key="pca",
        role="pca",
        coordinates=coordinate_blocks,
        coordinate_shape=coordinates.shape,
        coordinate_dtype=coordinates.dtype,
        source_digest=_SOURCE_DIGEST,
        payload_fingerprints=_fingerprints(
            data=coordinates,
            loadings=loadings,
            feature_ids=feature_ids,
            stdev=stdev,
        ),
        source_cell_ids=selected_ids,
        cell_selection=selection,
        cell_key="I",
        loadings=loadings,
        feature_ids=feature_ids,
        stdev=stdev,
        named_result="seurat_pca",
        block_rows=2,
    )

    assert pulls == [0, 3, 6]
    assert ref.kind == "imported_coordinates"
    status = root[artifact_path(ref)].attrs["provenance"]
    assert status["operation"] == "import_dimreduc"
    assert status["inputs"]["source_digest"] == {"bytes_hex": _SOURCE_DIGEST.hex()}
    assert status["inputs"]["cell_selection"] == selection.to_dict()
    assert status["inputs"]["ordered_cell_ids_fingerprint"] == fingerprint_strings(
        selected_ids
    )
    group = artifact_group(root, ref)
    np.testing.assert_array_equal(group["data"][:], coordinates)
    np.testing.assert_array_equal(group["loadings"][:], loadings)
    np.testing.assert_array_equal(group["feature_ids"][:], feature_ids)
    np.testing.assert_array_equal(group["stdev"][:], stdev)
    validate_imported_coordinates_artifact(root, ref, cell_key="I")
    state = read_assay_state(root, "RNA")
    assert state is not None
    assert state.normalized is None
    assert state.reduction is None
    assert not hasattr(state, "feat_key")
    assert state.named_results["seurat_pca"] == ref


def test_imported_coordinates_replace_an_unavailable_current_graph() -> None:
    root, selection, cell_ids, mask = _root_with_selection()
    missing_graph = ArtifactRef(
        scope="assay",
        assay="RNA",
        kind="connectivity_map",
        artifact_id="f" * 64,
    )
    state_group = root["RNA"].create_group("state")
    state_group.attrs["state"] = AssayState(
        assay="RNA",
        cell_key="I",
        connectivity_map=missing_graph,
    ).to_dict()

    coordinates = np.arange(24, dtype=np.float32).reshape(8, 3)
    ref = write_imported_coordinates(
        root,
        assay="RNA",
        dimreduc_key="pca",
        role="pca",
        coordinates=coordinates,
        source_digest=_SOURCE_DIGEST,
        payload_fingerprints={"data": fingerprint_array(coordinates)},
        source_cell_ids=cell_ids[mask],
        cell_selection=selection,
        cell_key="I",
        named_result="seurat_pca",
    )

    state = read_assay_state(root, "RNA")
    assert state is not None
    assert state.connectivity_map is None
    assert state.named_results == {"seurat_pca": ref}


def test_imported_coordinates_reject_legacy_state_before_writing() -> None:
    root, selection, cell_ids, mask = _root_with_selection()
    legacy_state = AssayState(assay="RNA", cell_key="I").to_dict()
    legacy_state["feat_key"] = "I"
    state_group = root["RNA"].create_group("state")
    state_group.attrs["state"] = legacy_state
    coordinates = np.arange(24, dtype=np.float32).reshape(8, 3)

    with pytest.raises(IncompatibleAnalysisStateError) as caught:
        write_imported_coordinates(
            root,
            assay="RNA",
            dimreduc_key="pca",
            role="pca",
            coordinates=coordinates,
            source_digest=_SOURCE_DIGEST,
            payload_fingerprints={"data": fingerprint_array(coordinates)},
            source_cell_ids=cell_ids[mask],
            cell_selection=selection,
            cell_key="I",
            named_result="seurat_pca",
        )

    assert caught.value.code == "legacy_feature_contract"
    assert (
        list_artifacts(
            root,
            scope="assay",
            assay="RNA",
            kind="imported_coordinates",
        )
        == []
    )


def test_imported_coordinates_validate_alignment_before_artifact_creation() -> None:
    root, selection, cell_ids, mask = _root_with_selection()
    coordinates = np.arange(24, dtype=np.float32).reshape(8, 3)

    with pytest.raises(ArtifactResolutionError) as caught:
        write_imported_coordinates(
            root,
            assay="RNA",
            dimreduc_key="pca",
            role="pca",
            coordinates=coordinates,
            source_digest=_SOURCE_DIGEST,
            payload_fingerprints={"data": fingerprint_array(coordinates)},
            source_cell_ids=cell_ids[mask][::-1],
            cell_selection=selection,
            cell_key="I",
        )

    assert caught.value.code == "dimreduc_cell_identity_mismatch"
    assert (
        list_artifacts(
            root,
            scope="assay",
            assay="RNA",
            kind="imported_coordinates",
        )
        == []
    )


def test_imported_coordinate_alignment_reads_cell_ids_in_blocks() -> None:
    root, selection, cell_ids, mask = _root_with_selection()
    coordinates = np.arange(24, dtype=np.float32).reshape(8, 3)

    class TrackedIds:
        shape = cell_ids.shape

        def __init__(self) -> None:
            self.reads: list[tuple[int, int]] = []

        def __getitem__(self, key: slice) -> np.ndarray:
            assert isinstance(key, slice)
            assert key.start is not None and key.stop is not None
            self.reads.append((key.start, key.stop))
            return cell_ids[key]

    tracked: Any = TrackedIds()
    write_imported_coordinates(
        root,
        assay="RNA",
        dimreduc_key="pca",
        role="pca",
        coordinates=coordinates,
        source_digest=_SOURCE_DIGEST,
        payload_fingerprints={"data": fingerprint_array(coordinates)},
        source_cell_ids=tracked,
        cell_selection=selection,
        cell_key="I",
        block_rows=3,
    )

    assert tracked.reads == [(0, 3), (3, 6), (6, 8)]


def test_imported_coordinate_loadings_stream_feature_ids_twice() -> None:
    root, selection, cell_ids, mask = _root_with_selection()
    coordinates = np.arange(24, dtype=np.float32).reshape(8, 3)
    loadings = np.arange(18, dtype=np.float32).reshape(6, 3)
    feature_ids = np.array([f"gene-{index}" for index in range(6)])

    class TrackedFeatureIds:
        def __init__(self) -> None:
            self.reads: list[tuple[int, int]] = []

        def __len__(self) -> int:
            return len(feature_ids)

        def __getitem__(self, key: slice) -> np.ndarray:
            assert isinstance(key, slice)
            assert key.start is not None and key.stop is not None
            self.reads.append((key.start, key.stop))
            return feature_ids[key]

    tracked: Any = TrackedFeatureIds()
    ref = write_imported_coordinates(
        root,
        assay="RNA",
        dimreduc_key="pca",
        role="pca",
        coordinates=coordinates,
        source_digest=_SOURCE_DIGEST,
        payload_fingerprints={
            "data": fingerprint_array(coordinates),
            "loadings": fingerprint_array(loadings),
            "feature_ids": fingerprint_strings(feature_ids),
        },
        source_cell_ids=cell_ids[mask],
        cell_selection=selection,
        cell_key="I",
        loadings=loadings,
        feature_ids=tracked,
        block_rows=3,
    )

    assert artifact_group(root, ref)["feature_ids"][:].tolist() == feature_ids.tolist()
    assert tracked.reads == [(0, 3), (3, 6), (0, 3), (3, 6)]


def test_imported_coordinates_require_a_fixed_size_source_digest() -> None:
    root, selection, cell_ids, mask = _root_with_selection()
    coordinates = np.arange(24, dtype=np.float32).reshape(8, 3)

    with pytest.raises(TypeError, match="exactly 32 bytes"):
        write_imported_coordinates(
            root,
            assay="RNA",
            dimreduc_key="pca",
            role="pca",
            coordinates=coordinates,
            source_digest=b"short",
            payload_fingerprints={"data": fingerprint_array(coordinates)},
            source_cell_ids=cell_ids[mask],
            cell_selection=selection,
            cell_key="I",
        )


def test_imported_coordinates_reject_unstored_payload_fingerprint() -> None:
    root, selection, cell_ids, mask = _root_with_selection()
    coordinates = np.arange(24, dtype=np.float32).reshape(8, 3)

    with pytest.raises(ValueError, match="Unexpected payload fingerprints"):
        write_imported_coordinates(
            root,
            assay="RNA",
            dimreduc_key="pca",
            role="pca",
            coordinates=coordinates,
            source_digest=_SOURCE_DIGEST,
            payload_fingerprints={
                "data": fingerprint_array(coordinates),
                "normalized": fingerprint_array(coordinates),
            },
            source_cell_ids=cell_ids[mask],
            cell_selection=selection,
            cell_key="I",
        )

    assert (
        list_artifacts(
            root,
            scope="assay",
            assay="RNA",
            kind="imported_coordinates",
        )
        == []
    )


def test_imported_coordinate_validation_detects_payload_tampering() -> None:
    root, selection, cell_ids, mask = _root_with_selection()
    coordinates = np.arange(24, dtype=np.float32).reshape(8, 3)
    ref = write_imported_coordinates(
        root,
        assay="RNA",
        dimreduc_key="pca",
        role="pca",
        coordinates=coordinates,
        source_digest=_SOURCE_DIGEST,
        payload_fingerprints={"data": fingerprint_array(coordinates)},
        source_cell_ids=cell_ids[mask],
        cell_selection=selection,
        cell_key="I",
    )

    artifact_group(root, ref)["data"][0, 0] = -1

    with pytest.raises(ValueError, match="fingerprint.*does not match"):
        validate_imported_coordinates_artifact(root, ref)


def test_imported_embedding_writes_values_and_links_metadata_columns() -> None:
    mask = np.array([True, False, True, False, True, True, False, True])
    root, selection, cell_ids, mask = _root_with_selection(mask)
    coordinates = np.arange(10, dtype=np.float32).reshape(5, 2)
    ref = write_imported_embedding(
        root,
        assay="RNA",
        dimreduc_key="umap",
        role="umap",
        coordinates=(coordinates[:2], coordinates[2:]),
        coordinate_shape=coordinates.shape,
        coordinate_dtype=coordinates.dtype,
        source_digest=_SOURCE_DIGEST,
        payload_fingerprints={"values": fingerprint_array(coordinates)},
        source_cell_ids=cell_ids[mask],
        cell_selection=selection,
        cell_key="I",
        block_rows=2,
    )

    assert ref.kind == "embedding"
    assert (
        root[artifact_path(ref)].attrs["provenance"]["operation"] == "import_dimreduc"
    )
    validate_imported_embedding_artifact(root, ref, cell_key="I")
    for index, column_name in enumerate(("RNA_UMAP1", "RNA_UMAP2")):
        column = root["cellData"][column_name]
        np.testing.assert_array_equal(column[:][mask], coordinates[:, index])
        assert np.isnan(column[:][~mask]).all()
        assert column.attrs["source_artifact"] == ref.to_dict()
        assert column.attrs["source_value"] == "values"
        assert column.attrs["value_index"] == index


def test_ann_and_neighbor_query_accept_detached_imported_coordinates() -> None:
    root, selection, cell_ids, mask = _root_with_selection()
    rng = np.random.default_rng(42)
    coordinates = rng.normal(size=(8, 3)).astype(np.float32)
    imported = write_imported_coordinates(
        root,
        assay="RNA",
        dimreduc_key="pca",
        role="pca",
        coordinates=coordinates,
        source_digest=_SOURCE_DIGEST,
        payload_fingerprints={"data": fingerprint_array(coordinates)},
        source_cell_ids=cell_ids[mask],
        cell_selection=selection,
        cell_key="I",
    )
    store = _graph_store(root)

    ann = store.build_ann_index(
        imported,
        ann_efc=10,
        ann_ef=10,
        ann_m=4,
        update_state=True,
    )
    state = read_assay_state(root, "RNA")
    assert state is not None
    assert state.normalized is None
    assert state.ann_index == ann

    neighbors = store.query_neighbors(ann, k=3, update_state=True)
    state = read_assay_state(root, "RNA")
    assert state is not None
    assert state.normalized is None
    assert state.ann_index == ann
    assert state.neighbors == neighbors
    validate_neighbors_artifact_selection(root, neighbors, "I")
    group = artifact_group(root, neighbors)
    assert group["indices"].shape == (8, 3)
    assert group["distances"].shape == (8, 3)


def test_imported_coordinates_kind_is_assay_scoped() -> None:
    with pytest.raises(ValueError, match="must be assay-scoped"):
        ArtifactRef(
            scope="datastore",
            kind="imported_coordinates",
            artifact_id="a" * 64,
        )


def test_imported_coordinates_reject_invalid_block_rows_and_fingerprints() -> None:
    root, selection, cell_ids, mask = _root_with_selection()
    coordinates = np.arange(24, dtype=np.float32).reshape(8, 3)
    fingerprints = {"data": fingerprint_array(coordinates[mask])}
    common = dict(
        assay="RNA",
        cell_selection=selection,
        cell_key="I",
        source_cell_ids=cell_ids[mask],
        coordinates=coordinates[mask],
        dimreduc_key="pca",
        role="pca",
        source_digest=_SOURCE_DIGEST,
        payload_fingerprints=fingerprints,
    )

    with pytest.raises(TypeError, match="block_rows must be a positive integer"):
        write_imported_coordinates(root, block_rows=0.5, **common)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="block_rows must be greater than zero"):
        write_imported_coordinates(root, block_rows=0, **common)
    with pytest.raises(ValueError, match="64-character lowercase hex"):
        write_imported_coordinates(
            root,
            assay="RNA",
            cell_selection=selection,
            cell_key="I",
            source_cell_ids=cell_ids[mask],
            coordinates=coordinates[mask],
            dimreduc_key="pca",
            role="pca",
            source_digest=_SOURCE_DIGEST,
            payload_fingerprints={"data": "not-a-fingerprint"},
        )


def test_imported_coordinates_reject_invalid_cell_ids_and_nonfinite_values() -> None:
    root, selection, cell_ids, mask = _root_with_selection()
    coordinates = np.arange(24, dtype=np.float32).reshape(8, 3)
    selected = coordinates[mask].copy()
    bad_ids = cell_ids[mask].copy()
    bad_ids[0] = ""

    with pytest.raises(ValueError, match="invalid identifier"):
        write_imported_coordinates(
            root,
            assay="RNA",
            cell_selection=selection,
            cell_key="I",
            source_cell_ids=bad_ids,
            coordinates=selected,
            dimreduc_key="pca",
            role="pca",
            source_digest=_SOURCE_DIGEST,
            payload_fingerprints={"data": fingerprint_array(selected)},
        )

    selected[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        write_imported_coordinates(
            root,
            assay="RNA",
            cell_selection=selection,
            cell_key="I",
            source_cell_ids=cell_ids[mask],
            coordinates=selected,
            dimreduc_key="pca",
            role="pca",
            source_digest=_SOURCE_DIGEST,
            payload_fingerprints={"data": fingerprint_array(np.nan_to_num(selected))},
        )


def test_imported_coordinates_reject_mismatched_loadings_shape() -> None:
    root, selection, cell_ids, mask = _root_with_selection()
    coordinates = np.arange(24, dtype=np.float32).reshape(8, 3)
    loadings = np.arange(10, dtype=np.float64).reshape(5, 2)
    feature_ids = np.array([f"gene_{index}" for index in range(5)])

    with pytest.raises(ValueError, match="loadings dimensions must match"):
        write_imported_coordinates(
            root,
            assay="RNA",
            cell_selection=selection,
            cell_key="I",
            source_cell_ids=cell_ids[mask],
            coordinates=coordinates[mask],
            dimreduc_key="pca",
            role="pca",
            source_digest=_SOURCE_DIGEST,
            payload_fingerprints=_fingerprints(
                data=coordinates[mask],
                loadings=loadings,
                feature_ids=feature_ids,
            ),
            loadings=loadings,
            feature_ids=feature_ids,
        )


@pytest.mark.parametrize(
    ("coordinates", "error_type", "message"),
    [
        (
            np.arange(8, dtype=np.float32),
            ValueError,
            "must have 2 non-empty dimensions",
        ),
        (
            np.empty((8, 0), dtype=np.float32),
            ValueError,
            "must have 2 non-empty dimensions",
        ),
        (
            np.arange(24, dtype=np.int32).reshape(8, 3),
            TypeError,
            "floating-point dtype",
        ),
    ],
)
def test_imported_coordinates_reject_invalid_matrix_shape_and_dtype(
    coordinates,
    error_type,
    message,
) -> None:
    root, selection, cell_ids, mask = _root_with_selection()

    with pytest.raises(error_type, match=message):
        write_imported_coordinates(
            root,
            assay="RNA",
            dimreduc_key="pca",
            role="pca",
            coordinates=coordinates,
            source_digest=_SOURCE_DIGEST,
            payload_fingerprints={"data": "a" * 64},
            source_cell_ids=cell_ids[mask],
            cell_selection=selection,
            cell_key="I",
        )


def test_imported_coordinates_report_selected_row_count_mismatch() -> None:
    root, selection, cell_ids, _mask = _root_with_selection()
    coordinates = np.arange(21, dtype=np.float32).reshape(7, 3)

    with pytest.raises(ArtifactResolutionError) as caught:
        write_imported_coordinates(
            root,
            assay="RNA",
            dimreduc_key="pca",
            role="pca",
            coordinates=coordinates,
            source_digest=_SOURCE_DIGEST,
            payload_fingerprints={"data": fingerprint_array(coordinates)},
            source_cell_ids=cell_ids[:7],
            cell_selection=selection,
            cell_key="I",
            block_rows=3,
        )

    assert caught.value.code == "dimreduc_row_count_mismatch"
    assert caught.value.context["coordinate_rows"] == 7
    assert caught.value.context["source_cell_count"] == 7


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("shape", "invalid shape or dtype"),
        ("dtype", "invalid shape or dtype"),
        ("short", "contains 7 rows, expected 8"),
        ("long", "exceeds its declared row count"),
    ],
)
def test_imported_coordinate_stream_validates_declared_shape_and_dtype(
    case,
    message,
) -> None:
    root, selection, cell_ids, mask = _root_with_selection()
    coordinates = np.arange(24, dtype=np.float32).reshape(8, 3)
    if case == "shape":
        block = np.arange(16, dtype=np.float32).reshape(8, 2)
    elif case == "dtype":
        block = coordinates.astype(np.float64)
    elif case == "short":
        block = coordinates[:7]
    else:
        block = np.arange(27, dtype=np.float32).reshape(9, 3)

    def blocks():
        yield block

    with pytest.raises(ValueError, match=message):
        write_imported_coordinates(
            root,
            assay="RNA",
            dimreduc_key="pca",
            role="pca",
            coordinates=blocks,
            coordinate_shape=coordinates.shape,
            coordinate_dtype=coordinates.dtype,
            source_digest=_SOURCE_DIGEST,
            payload_fingerprints={"data": fingerprint_array(coordinates)},
            source_cell_ids=cell_ids[mask],
            cell_selection=selection,
            cell_key="I",
            block_rows=3,
        )


@pytest.mark.parametrize(
    ("attribute", "path", "value", "message"),
    [
        (
            "provenance",
            ("operation",),
            "run_pca",
            "operation must be 'import_dimreduc'",
        ),
        (
            "execution_options",
            ("cell_key",),
            "",
            "has no cell selection key",
        ),
        (
            "execution_options",
            ("block_rows",),
            0,
            "block_rows is invalid",
        ),
        (
            "provenance",
            ("parameters", "dimreduc_key"),
            "",
            "source key is missing",
        ),
        (
            "provenance",
            ("parameters", "dims"),
            2,
            "dimensions do not match data",
        ),
        (
            "provenance",
            ("parameters", "role"),
            "umap",
            "role is invalid",
        ),
        (
            "provenance",
            ("inputs", "source_digest"),
            {"bytes_hex": "g" * 64},
            "source digest is not hexadecimal",
        ),
        (
            "provenance",
            ("inputs", "payload_fingerprints"),
            {"data": "short"},
            "payload fingerprints are malformed",
        ),
        (
            "provenance",
            ("inputs", "ordered_cell_ids_fingerprint"),
            "0" * 64,
            "cell IDs do not match",
        ),
    ],
)
def test_imported_coordinate_validator_rejects_provenance_tampering(
    attribute,
    path,
    value,
    message,
) -> None:
    root, selection, cell_ids, mask = _root_with_selection()
    ref, _coordinates = _write_coordinate_fixture(
        root,
        selection,
        cell_ids,
        mask,
    )
    _tamper_artifact_attribute(root, ref, attribute, path, value)

    with pytest.raises(ArtifactResolutionError, match=message) as caught:
        validate_imported_coordinates_artifact(root, ref)
    expected_code = (
        "dimreduc_cell_identity_mismatch"
        if path == ("inputs", "ordered_cell_ids_fingerprint")
        else "corrupt_payload"
    )
    assert caught.value.code == expected_code


def test_imported_coordinate_validator_rejects_scope_and_kind_mismatches() -> None:
    root, selection, cell_ids, mask = _root_with_selection()
    ref, _coordinates = _write_coordinate_fixture(
        root,
        selection,
        cell_ids,
        mask,
    )
    wrong_kind = ArtifactRef(
        scope="assay",
        assay="RNA",
        kind="embedding",
        artifact_id=ref.artifact_id,
    )
    wrong_scope = ArtifactRef(
        scope="datastore",
        kind="embedding",
        artifact_id=ref.artifact_id,
    )

    with pytest.raises(ArtifactResolutionError) as caught:
        validate_imported_coordinates_artifact(root, wrong_kind)
    assert caught.value.code == "artifact_reference_mismatch"
    with pytest.raises(ArtifactResolutionError) as caught:
        validate_imported_coordinates_artifact(root, wrong_scope)
    assert caught.value.code == "artifact_reference_mismatch"


@pytest.mark.parametrize(
    ("payload_name", "replacement", "message"),
    [
        (
            "data",
            np.arange(24, dtype=np.int32).reshape(8, 3),
            "data must be a floating-point matrix",
        ),
        (
            "loadings",
            np.arange(8, dtype=np.float64).reshape(4, 2),
            "loadings and feature IDs are misaligned",
        ),
        (
            "feature_ids",
            np.arange(4, dtype=np.int32),
            "loadings and feature IDs are misaligned",
        ),
        (
            "stdev",
            np.asarray([2.0, 1.0], dtype=np.float64),
            "stdev does not match dimensions",
        ),
    ],
)
def test_imported_coordinate_validator_rejects_payload_shape_and_dtype_tampering(
    payload_name,
    replacement,
    message,
) -> None:
    root, selection, cell_ids, mask = _root_with_selection()
    ref, _coordinates = _write_coordinate_fixture(
        root,
        selection,
        cell_ids,
        mask,
        include_optional=True,
    )
    group = artifact_group(root, ref)
    group.create_array(
        payload_name,
        data=replacement,
        overwrite=True,
    )

    with pytest.raises(ArtifactResolutionError, match=message) as caught:
        validate_imported_coordinates_artifact(root, ref)
    assert caught.value.code == "corrupt_payload"


def test_imported_coordinate_validator_checks_optional_flags_and_fingerprints() -> None:
    root, selection, cell_ids, mask = _root_with_selection()
    ref, _coordinates = _write_coordinate_fixture(
        root,
        selection,
        cell_ids,
        mask,
        include_optional=True,
    )
    _tamper_artifact_attribute(
        root,
        ref,
        "provenance",
        ("parameters", "loadings_stored"),
        False,
    )
    with pytest.raises(
        ArtifactResolutionError,
        match="storage flag does not match payload",
    ) as caught:
        validate_imported_coordinates_artifact(root, ref)
    assert caught.value.code == "corrupt_payload"

    _tamper_artifact_attribute(
        root,
        ref,
        "provenance",
        ("parameters", "loadings_stored"),
        True,
    )
    artifact_group(root, ref)["loadings"][0, 0] = -1.0
    with pytest.raises(
        ArtifactResolutionError,
        match="fingerprint for 'loadings' does not match",
    ) as caught:
        validate_imported_coordinates_artifact(root, ref)
    assert caught.value.code == "corrupt_payload"


def test_imported_coordinate_validation_detects_selection_and_cell_key_changes() -> (
    None
):
    root, selection, cell_ids, mask = _root_with_selection()
    ref, _coordinates = _write_coordinate_fixture(
        root,
        selection,
        cell_ids,
        mask,
    )

    with pytest.raises(
        ArtifactResolutionError,
        match="does not match imported coordinates",
    ) as caught:
        validate_imported_coordinates_artifact(root, ref, cell_key="other")
    assert caught.value.code == "row_mismatch"

    root["cellData"]["I"][0] = False
    with pytest.raises(ArtifactResolutionError) as caught:
        validate_imported_coordinates_artifact(root, ref)
    assert caught.value.code == "selection_values_changed"


def test_imported_coordinate_validation_rechecks_exact_selection_size() -> None:
    root, selection, cell_ids, mask = _root_with_selection()
    ref, _coordinates = _write_coordinate_fixture(
        root,
        selection,
        cell_ids,
        mask,
    )
    replacement = np.asarray(artifact_group(root, ref)["data"][:-1])
    artifact_group(root, ref).create_array(
        "data",
        data=replacement,
        overwrite=True,
    )
    _tamper_artifact_attribute(
        root,
        ref,
        "provenance",
        ("inputs", "payload_fingerprints"),
        {"data": fingerprint_array(replacement)},
    )

    with pytest.raises(
        ArtifactResolutionError,
        match="rows do not match the exact cell selection",
    ) as caught:
        validate_imported_coordinates_artifact(root, ref)
    assert caught.value.code == "dimreduc_row_count_mismatch"


def test_imported_coordinates_reuse_without_consuming_coordinate_blocks() -> None:
    root, selection, cell_ids, mask = _root_with_selection()
    first, coordinates = _write_coordinate_fixture(
        root,
        selection,
        cell_ids,
        mask,
    )
    pulls = []

    def coordinate_blocks():
        pulls.append("consumed")
        yield coordinates

    reused = write_imported_coordinates(
        root,
        assay="RNA",
        dimreduc_key="pca",
        role="pca",
        coordinates=coordinate_blocks,
        coordinate_shape=coordinates.shape,
        coordinate_dtype=coordinates.dtype,
        source_digest=_SOURCE_DIGEST,
        payload_fingerprints={"data": fingerprint_array(coordinates)},
        source_cell_ids=cell_ids[mask],
        cell_selection=selection,
        cell_key="I",
        block_rows=2,
    )

    assert reused == first
    assert pulls == []
    assert (
        len(
            list_artifacts(
                root,
                scope="assay",
                assay="RNA",
                kind="imported_coordinates",
            )
        )
        == 1
    )


def test_imported_coordinate_reuse_rejects_tampered_candidate() -> None:
    root, selection, cell_ids, mask = _root_with_selection()
    first, coordinates = _write_coordinate_fixture(
        root,
        selection,
        cell_ids,
        mask,
    )
    artifact_group(root, first)["data"][0, 0] = -1.0

    replacement = write_imported_coordinates(
        root,
        assay="RNA",
        dimreduc_key="pca",
        role="pca",
        coordinates=coordinates,
        source_digest=_SOURCE_DIGEST,
        payload_fingerprints={"data": fingerprint_array(coordinates)},
        source_cell_ids=cell_ids[mask],
        cell_selection=selection,
        cell_key="I",
        block_rows=2,
    )

    assert replacement != first
    validate_imported_coordinates_artifact(root, replacement)
    assert (
        len(
            list_artifacts(
                root,
                scope="assay",
                assay="RNA",
                kind="imported_coordinates",
            )
        )
        == 2
    )


@pytest.mark.parametrize(
    ("role", "dimreduc_key", "error_type", "message"),
    [
        (3, "umap", TypeError, "role must be a string"),
        ("pca", "umap", ValueError, "role 'umap' or 'tsne'"),
        ("umap", "", ValueError, "dimreduc_key must be a non-empty string"),
    ],
)
def test_imported_embedding_validates_role_and_source_key(
    role,
    dimreduc_key,
    error_type,
    message,
) -> None:
    root, selection, cell_ids, mask = _root_with_selection()
    coordinates = np.arange(16, dtype=np.float32).reshape(8, 2)

    with pytest.raises(error_type, match=message):
        write_imported_embedding(
            root,
            assay="RNA",
            dimreduc_key=dimreduc_key,
            role=role,
            coordinates=coordinates,
            source_digest=_SOURCE_DIGEST,
            payload_fingerprints={"values": fingerprint_array(coordinates)},
            source_cell_ids=cell_ids[mask],
            cell_selection=selection,
            cell_key="I",
        )


@pytest.mark.parametrize(
    ("metadata_columns", "message"),
    [
        (("only_one",), "one name per embedding dimension"),
        (("duplicate", "duplicate"), "invalid or duplicate"),
        (("I", "second"), "invalid or duplicate"),
    ],
)
def test_imported_embedding_validates_metadata_column_contract(
    metadata_columns,
    message,
) -> None:
    root, selection, cell_ids, mask = _root_with_selection()
    coordinates = np.arange(16, dtype=np.float32).reshape(8, 2)

    with pytest.raises(ValueError, match=message):
        write_imported_embedding(
            root,
            assay="RNA",
            dimreduc_key="umap",
            role="umap",
            coordinates=coordinates,
            source_digest=_SOURCE_DIGEST,
            payload_fingerprints={"values": fingerprint_array(coordinates)},
            source_cell_ids=cell_ids[mask],
            cell_selection=selection,
            cell_key="I",
            metadata_columns=metadata_columns,
        )


@pytest.mark.parametrize(
    ("attribute", "path", "value", "message"),
    [
        (
            "provenance",
            ("operation",),
            "run_umap",
            "operation must be 'import_dimreduc'",
        ),
        (
            "execution_options",
            ("cell_key",),
            "",
            "has no cell selection key",
        ),
        (
            "execution_options",
            ("block_rows",),
            False,
            "block_rows is invalid",
        ),
        (
            "provenance",
            ("parameters", "dimreduc_key"),
            "",
            "source key is missing",
        ),
        (
            "provenance",
            ("parameters", "dims"),
            3,
            "payload is malformed",
        ),
        (
            "provenance",
            ("parameters", "role"),
            "pca",
            "payload is malformed",
        ),
        (
            "provenance",
            ("inputs", "cell_selection"),
            None,
            "has no cell selection input",
        ),
        (
            "provenance",
            ("inputs", "ordered_cell_ids_fingerprint"),
            "0" * 64,
            "cell IDs are out of order",
        ),
        (
            "provenance",
            ("inputs", "source_digest"),
            {"bytes_hex": "g" * 64},
            "source digest is not hexadecimal",
        ),
        (
            "provenance",
            ("inputs", "payload_fingerprints"),
            {"values": "f" * 64},
            "payload fingerprint does not match",
        ),
    ],
)
def test_imported_embedding_validator_rejects_provenance_tampering(
    attribute,
    path,
    value,
    message,
) -> None:
    root, selection, cell_ids, mask = _root_with_selection()
    ref, _coordinates = _write_embedding_fixture(
        root,
        selection,
        cell_ids,
        mask,
    )
    _tamper_artifact_attribute(root, ref, attribute, path, value)

    with pytest.raises(ValueError, match=message):
        validate_imported_embedding_artifact(root, ref)


def test_imported_embedding_validator_rejects_scope_and_kind_mismatches() -> None:
    root, selection, cell_ids, mask = _root_with_selection()
    ref, _coordinates = _write_embedding_fixture(
        root,
        selection,
        cell_ids,
        mask,
    )
    wrong_kind = ArtifactRef(
        scope="assay",
        assay="RNA",
        kind="imported_coordinates",
        artifact_id=ref.artifact_id,
    )
    wrong_scope = ArtifactRef(
        scope="datastore",
        kind="embedding",
        artifact_id=ref.artifact_id,
    )

    with pytest.raises(ValueError, match="assay-scoped embedding"):
        validate_imported_embedding_artifact(root, wrong_kind)
    with pytest.raises(ValueError, match="assay-scoped embedding"):
        validate_imported_embedding_artifact(root, wrong_scope)


@pytest.mark.parametrize(
    "replacement",
    [
        np.arange(16, dtype=np.int32).reshape(8, 2),
        np.arange(8, dtype=np.float32),
        np.arange(24, dtype=np.float32).reshape(8, 3),
    ],
)
def test_imported_embedding_validator_rejects_payload_shape_and_dtype(
    replacement,
) -> None:
    root, selection, cell_ids, mask = _root_with_selection()
    ref, _coordinates = _write_embedding_fixture(
        root,
        selection,
        cell_ids,
        mask,
    )
    artifact_group(root, ref).create_array(
        "values",
        data=replacement,
        overwrite=True,
    )

    with pytest.raises(ValueError, match="payload is malformed"):
        validate_imported_embedding_artifact(root, ref)


def test_imported_embedding_validation_rechecks_selection_and_cell_key() -> None:
    root, selection, cell_ids, mask = _root_with_selection()
    ref, _coordinates = _write_embedding_fixture(
        root,
        selection,
        cell_ids,
        mask,
    )
    with pytest.raises(ValueError, match="cell_key does not match"):
        validate_imported_embedding_artifact(root, ref, cell_key="other")

    replacement = np.asarray(artifact_group(root, ref)["values"][:-1])
    artifact_group(root, ref).create_array(
        "values",
        data=replacement,
        overwrite=True,
    )
    _tamper_artifact_attribute(
        root,
        ref,
        "provenance",
        ("inputs", "payload_fingerprints"),
        {"values": fingerprint_array(replacement)},
    )
    with pytest.raises(ValueError, match="rows do not match its cell selection"):
        validate_imported_embedding_artifact(root, ref)


def test_imported_embedding_reuses_payload_without_consuming_blocks() -> None:
    root, selection, cell_ids, mask = _root_with_selection()
    first, coordinates = _write_embedding_fixture(
        root,
        selection,
        cell_ids,
        mask,
    )
    pulls = []

    def coordinate_blocks():
        pulls.append("consumed")
        yield coordinates

    reused = write_imported_embedding(
        root,
        assay="RNA",
        dimreduc_key="umap",
        role="umap",
        coordinates=coordinate_blocks,
        coordinate_shape=coordinates.shape,
        coordinate_dtype=coordinates.dtype,
        source_digest=_SOURCE_DIGEST,
        payload_fingerprints={"values": fingerprint_array(coordinates)},
        source_cell_ids=cell_ids[mask],
        cell_selection=selection,
        cell_key="I",
        block_rows=2,
    )

    assert reused == first
    assert pulls == []
    for index, column_name in enumerate(("RNA_UMAP1", "RNA_UMAP2")):
        np.testing.assert_array_equal(
            root["cellData"][column_name][:],
            coordinates[:, index],
        )


def test_imported_artifacts_validate_and_coordinates_reuse_read_only(tmp_path) -> None:
    store_path = tmp_path / "imported.zarr"
    root, selection, cell_ids, mask = _root_with_selection(
        store=str(store_path),
    )
    coordinate_ref, coordinates = _write_coordinate_fixture(
        root,
        selection,
        cell_ids,
        mask,
    )
    embedding_ref, _embedding = _write_embedding_fixture(
        root,
        selection,
        cell_ids,
        mask,
    )
    del root

    read_only = zarr.open_group(store=str(store_path), mode="r")
    columns_before = tuple(sorted(read_only["cellData"].keys()))
    validate_imported_coordinates_artifact(read_only, coordinate_ref, cell_key="I")
    validate_imported_embedding_artifact(read_only, embedding_ref, cell_key="I")
    pulls = []

    def coordinate_blocks():
        pulls.append("consumed")
        yield coordinates

    reused = write_imported_coordinates(
        read_only,
        assay="RNA",
        dimreduc_key="pca",
        role="pca",
        coordinates=coordinate_blocks,
        coordinate_shape=coordinates.shape,
        coordinate_dtype=coordinates.dtype,
        source_digest=_SOURCE_DIGEST,
        payload_fingerprints={"data": fingerprint_array(coordinates)},
        source_cell_ids=cell_ids[mask],
        cell_selection=selection,
        cell_key="I",
        block_rows=2,
    )

    assert reused == coordinate_ref
    assert pulls == []
    assert tuple(sorted(read_only["cellData"].keys())) == columns_before
