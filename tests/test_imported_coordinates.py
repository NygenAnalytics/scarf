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
    ArtifactSelectionError,
    read_assay_state,
    validate_imported_coordinates_artifact,
    validate_neighbors_artifact_selection,
)
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
) -> tuple[zarr.Group, ArtifactRef, np.ndarray, np.ndarray]:
    root = zarr.open_group(store=MemoryStore(), mode="w")
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
        feat_key="I",
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
    assert state.reduction is None
    assert state.named_results["seurat_pca"] == ref


def test_imported_coordinates_validate_alignment_before_artifact_creation() -> None:
    root, selection, cell_ids, mask = _root_with_selection()
    coordinates = np.arange(24, dtype=np.float32).reshape(8, 3)

    with pytest.raises(ArtifactSelectionError) as caught:
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

    with pytest.raises(ValueError, match="pass update_state=False"):
        store.build_ann_index(imported, update_state=True)
    assert read_assay_state(root, "RNA") is None

    ann = store.build_ann_index(
        imported,
        ann_efc=10,
        ann_ef=10,
        ann_m=4,
        update_state=False,
    )
    with pytest.raises(ValueError, match="pass update_state=False"):
        store.query_neighbors(ann, k=3, update_state=True)
    assert read_assay_state(root, "RNA") is None

    neighbors = store.query_neighbors(ann, k=3, update_state=False)
    validate_neighbors_artifact_selection(root, neighbors, "I", "unused")
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
