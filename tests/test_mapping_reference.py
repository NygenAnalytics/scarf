import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

import scarf.mapping as mapping
from scarf.datastore.datastore import DataStore
from scarf.metadata.artifacts import plan_cell_data_artifact, write_cell_data_artifact
from scarf.storage.artifacts import ArtifactRef, artifact_group, artifact_path
from scarf.storage.selections import read_stored_selection_indices
from tests.storage_helpers import write_count_store

_COMMON_ARRAYS = {
    "feature_ids",
    "feature_means",
    "feature_scales",
    "center",
    "loadings",
    "reference_distance_quantiles",
    "reference_distance_values",
}
_SYMPHONY_ARRAYS = {
    "centroids",
    "raw_centroids",
    "corrected_centroids",
    "cluster_mass",
    "sigma",
}

# The reference keeps all but two cells and the first 26 of 30 genes, so its
# cell and feature selections differ from the full store axes.
MAPPING_N_CELLS = 64
MAPPING_N_GENES = 30
MAPPING_REFERENCE_GENES = 26
MAPPING_EXCLUDED_CELLS = (5, 22)


@dataclass(frozen=True)
class MappingStoreSource:
    """A built reference store that each writing test copies first.

    Four cell programs of sixteen cells each raise their own five genes, so
    the PCA, the neighbor graph and the Harmony clusters have structure. The
    store also holds a three-protein ADT assay. ``reference`` and
    ``symphony_reference`` are set when the source was built with them.
    """

    path: Path
    cells: ArtifactRef
    features: ArtifactRef
    normalized: ArtifactRef
    reduction: ArtifactRef
    other_reduction: ArtifactRef
    neighbors: ArtifactRef
    correction: ArtifactRef
    symphony_neighbors: ArtifactRef
    reference: ArtifactRef | None = None
    symphony_reference: ArtifactRef | None = None

    def open_copy(self, path: Path, *, zarr_mode: str = "r+") -> DataStore:
        shutil.copytree(self.path, path)
        return DataStore(str(path), default_assay="RNA", zarr_mode=zarr_mode)


def mapping_counts(seed: int = 17) -> np.ndarray:
    """Return the reference counts: every cell measures every gene."""
    rng = np.random.default_rng(seed)
    programs = np.arange(MAPPING_N_CELLS) % 4
    counts = rng.poisson(4.0, size=(MAPPING_N_CELLS, MAPPING_N_GENES)) + 1
    for program in range(4):
        counts[programs == program, 5 * program : 5 * program + 5] += 30
    return counts


def build_mapping_store(path: Path, *, with_references: bool) -> MappingStoreSource:
    """Write and analyze the small reference store described above."""
    rng = np.random.default_rng(23)
    write_count_store(
        str(path),
        {
            "RNA": mapping_counts(),
            "ADT": rng.integers(1, 20, size=(MAPPING_N_CELLS, 3)),
        },
        "uint32",
    )
    store = DataStore(str(path), default_assay="RNA", min_features_per_cell=0)
    keep = np.ones(MAPPING_N_CELLS, dtype=bool)
    keep[list(MAPPING_EXCLUDED_CELLS)] = False
    store.cells.insert("reference_cells", keep)
    store.cells.insert(
        "mapping_batch",
        np.where(np.arange(MAPPING_N_CELLS) % 2, "a", "b"),
    )
    cells = store.snapshot_cell_selection("reference_cells")
    features = store.set_feature_selection(
        from_assay="RNA",
        feature_indexes=list(range(MAPPING_REFERENCE_GENES)),
    )
    normalized = store.run_normalization(cells, features)
    reduction = store.run_pca(normalized, dims=4, local_cache=False)
    other_reduction = store.run_pca(normalized, dims=3, local_cache=False)
    neighbors = store.query_neighbors(
        store.build_ann_index(reduction),
        coordinates=reduction,
        k=5,
    )
    correction = store.run_harmony(
        reduction,
        ["mapping_batch"],
        harmony_params={"nclust": 3},
    )
    symphony_neighbors = store.query_neighbors(
        store.build_ann_index(correction),
        coordinates=correction,
        k=5,
    )
    references = {}
    if with_references:
        references = {
            "reference": store.build_mapping_reference(neighbors),
            "symphony_reference": store.build_mapping_reference(symphony_neighbors),
        }
    return MappingStoreSource(
        path=path,
        cells=cells,
        features=features,
        normalized=normalized,
        reduction=reduction,
        other_reduction=other_reduction,
        neighbors=neighbors,
        correction=correction,
        symphony_neighbors=symphony_neighbors,
        **references,
    )


def build_mapping_query(path: Path) -> Path:
    """Write a query store with the RNA counts of the reference cells.

    It measures every reference gene but holds no artifacts.
    """
    write_count_store(str(path), {"RNA": mapping_counts()}, "uint32")
    DataStore(str(path), default_assay="RNA", min_features_per_cell=0)
    return path


@pytest.fixture(scope="module")
def mapping_source(tmp_path_factory) -> MappingStoreSource:
    return build_mapping_store(
        tmp_path_factory.mktemp("mapping_reference") / "reference.zarr",
        with_references=False,
    )


@pytest.fixture
def reference_store(mapping_source, tmp_path) -> DataStore:
    return mapping_source.open_copy(tmp_path / "reference.zarr")


def _edit_provenance(group, section, key, value) -> None:
    provenance = dict(group.attrs["provenance"])
    if section is None:
        provenance[key] = value
    else:
        provenance[section] = dict(provenance[section]) | {key: value}
    group.attrs["provenance"] = provenance


def _input_ref(datastore, ref: ArtifactRef, name: str) -> ArtifactRef:
    return ArtifactRef.from_dict(datastore.inspect_artifact(ref).inputs[name])


def test_embedded_mapping_reference_helpers_are_not_public():
    for name in (
        "LATEST_MAPPING_REFERENCE_ATTRIBUTE",
        "MAPPING_REFERENCE_GROUP",
        "MAPPING_REFERENCES_GROUP",
        "load_mapping_reference",
        "mapping_reference_hash",
        "persist_mapping_reference",
        "resolve_mapping_reference_group",
        "validate_mapping_reference_artifact",
    ):
        assert name not in mapping.__all__
        assert not hasattr(mapping, name)
    assert not hasattr(mapping.MappingReference, "map_query")


def _mapping_reference_source_paths(datastore, neighbors):
    neighbor_inputs = datastore.inspect_artifact(neighbors).inputs
    coordinates = ArtifactRef.from_dict(neighbor_inputs["coordinates"])
    ann_index = ArtifactRef.from_dict(neighbor_inputs["ann_index"])
    paths = {
        f"{artifact_path(neighbors)}/indices",
        f"{artifact_path(neighbors)}/distances",
        f"{artifact_path(ann_index)}/ann_idx_bytes",
        f"{artifact_path(coordinates)}/data",
    }
    if coordinates.kind == "batch_correction":
        reduction = ArtifactRef.from_dict(
            datastore.inspect_artifact(coordinates).inputs["reduction"]
        )
        paths.update(
            {
                f"{artifact_path(coordinates)}/{name}"
                for name in (
                    "centroids",
                    "raw_centroids",
                    "corrected_centroids",
                    "cluster_mass",
                    "sigma",
                )
            }
        )
    else:
        reduction = coordinates
    reduction_inputs = datastore.inspect_artifact(reduction).inputs
    scaling = ArtifactRef.from_dict(reduction_inputs["feature_scaling"])
    normalized = ArtifactRef.from_dict(reduction_inputs["normalized"])
    normalized_inputs = datastore.inspect_artifact(normalized).inputs
    cell_selection = ArtifactRef.from_dict(normalized_inputs["cell_selection"])
    feature_selection = ArtifactRef.from_dict(normalized_inputs["feature_selection"])
    paths.update(
        {
            f"{artifact_path(reduction)}/data",
            f"{artifact_path(reduction)}/loadings",
            f"{artifact_path(reduction)}/center",
            f"{artifact_path(scaling)}/mean",
            f"{artifact_path(scaling)}/scale",
            f"{artifact_path(cell_selection)}/values",
            f"{artifact_path(feature_selection)}/values",
            f"{neighbors.assay}/featureData/ids",
        }
    )
    return paths


def _reject_full_array_reads(monkeypatch, paths):
    original_getitem = zarr.Array.__getitem__
    original_array = zarr.Array.__array__
    row_spans = {path: [] for path in paths}

    def axis_span(item, length):
        if isinstance(item, slice):
            start, stop, step = item.indices(length)
            return len(range(start, stop, step))
        if isinstance(item, (int, np.integer)):
            return 1
        return length

    def guarded_getitem(array, key):
        if array.path in paths:
            if key is Ellipsis:
                raise AssertionError(
                    f"mapping-reference publication materialized {array.path}"
                )
            first_axis = key[0] if isinstance(key, tuple) else key
            span = axis_span(first_axis, int(array.shape[0]))
            row_spans[array.path].append(span)
            if array.ndim == 2 and int(array.shape[0]) > 1_000 and span > 1_000:
                raise AssertionError(
                    f"mapping-reference publication read {span} rows from "
                    f"{array.path} in one block"
                )
        return original_getitem(array, key)

    def guarded_array(array, dtype=None, copy=None):
        if array.path in paths:
            raise AssertionError(
                f"mapping-reference publication implicitly materialized {array.path}"
            )
        return original_array(array, dtype=dtype, copy=copy)

    monkeypatch.setattr(zarr.Array, "__getitem__", guarded_getitem)
    monkeypatch.setattr(zarr.Array, "__array__", guarded_array)
    return row_spans


def test_plain_mapping_reference_packages_and_loads_existing_chain(
    mapping_source,
    reference_store,
):
    datastore = reference_store
    neighbors = mapping_source.neighbors
    before = set(datastore.list_artifacts(from_assay="RNA"))
    reference_ref = datastore.build_mapping_reference(neighbors)
    assert isinstance(reference_ref, ArtifactRef)
    reference = datastore.get_mapping_reference(reference_ref)

    assert reference.method == "pca"
    assert reference.symphony_state is None
    assert reference.neighbors == neighbors
    assert reference.reduction == mapping_source.reduction
    assert reference.cell_selection == mapping_source.cells
    assert reference.feature_selection == mapping_source.features
    assert not hasattr(reference, "feature_key")
    assert reference.dataset_fingerprint == datastore._ensure_dataset_fingerprint("RNA")
    selected_cells = read_stored_selection_indices(
        datastore.zw,
        reference.cell_selection,
        kind="cell_selection",
        scope="datastore",
        assay=None,
        table_path="cellData",
    )
    np.testing.assert_array_equal(
        selected_cells,
        np.setdiff1d(np.arange(MAPPING_N_CELLS), MAPPING_EXCLUDED_CELLS),
    )
    assert reference.selected_cell_count == len(selected_cells)
    assert reference.ann_metric == "l2"
    np.testing.assert_array_equal(
        reference.feature_ids,
        [f"RNA{index}" for index in range(MAPPING_REFERENCE_GENES)],
    )
    # The model is an exact copy of the feature scaling and PCA it packages.
    normalized_status = datastore.inspect_artifact(mapping_source.normalized)
    assert reference.normalization_parameters == normalized_status.parameters
    assert reference.size_factor == normalized_status.parameters["size_factor"]
    pca = artifact_group(datastore.zw, mapping_source.reduction)
    scaling = artifact_group(
        datastore.zw,
        _input_ref(datastore, mapping_source.reduction, "feature_scaling"),
    )
    np.testing.assert_array_equal(reference.model.feature_means, scaling["mean"][:])
    np.testing.assert_array_equal(reference.model.feature_scales, scaling["scale"][:])
    np.testing.assert_array_equal(reference.model.center, pca["center"][:])
    np.testing.assert_array_equal(reference.model.loadings, pca["loadings"][:])
    # The distance summary holds quantiles of each cell's first-neighbor
    # distance, one quantile per reference cell when cells are few.
    first_distances = np.asarray(
        artifact_group(datastore.zw, neighbors)["distances"][:, 0],
        dtype=np.float64,
    )
    quantiles = np.linspace(0.0, 1.0, len(selected_cells))
    np.testing.assert_allclose(reference.reference_distance_quantiles, quantiles)
    np.testing.assert_allclose(
        reference.reference_distance_values,
        np.quantile(first_distances, quantiles),
    )

    group = artifact_group(datastore.zw, reference.ref)
    assert "feature_key" not in group.attrs["reference_metadata"]
    assert set(group.array_keys()) == _COMMON_ARRAYS
    status = datastore.inspect_artifact(reference.ref)
    assert status.parameters == {"method": "pca"}
    assert set((status.inputs or {})) == {
        "reduction",
        "ann_index",
        "neighbors",
        "cell_selection",
        "feature_selection",
    }
    created = set(datastore.list_artifacts(from_assay="RNA")) - before
    assert created == {reference.ref}

    assert datastore.get_mapping_reference(reference.ref).ref == reference.ref
    assert datastore.build_mapping_reference(neighbors) == reference.ref

    expected_layout = np.column_stack(
        (
            np.arange(len(selected_cells), dtype=np.float64),
            -np.arange(len(selected_cells), dtype=np.float64),
        )
    )
    planned_layout = plan_cell_data_artifact(
        datastore.zw,
        scope="assay",
        assay="RNA",
        kind="embedding",
        operation="manual_reference_embedding",
        parameters={},
        inputs={},
        execution_options={},
        cell_selection=reference.cell_selection,
        arrays={"values": (expected_layout.shape, "f")},
    )
    write_cell_data_artifact(
        datastore.zw,
        planned_layout,
        {"values": expected_layout},
    )
    np.testing.assert_array_equal(
        reference.fetch_cell_column("ids"),
        np.asarray(datastore.cells.fetch_all("ids"))[selected_cells],
    )
    np.testing.assert_array_equal(
        reference._fetch_layout(planned_layout.ref),
        expected_layout,
    )


def test_symphony_mapping_reference_has_conditional_state_and_read_only_reload(
    mapping_source,
    reference_store,
    monkeypatch,
):
    datastore = reference_store
    neighbors = mapping_source.symphony_neighbors
    _reject_full_array_reads(
        monkeypatch,
        _mapping_reference_source_paths(datastore, neighbors),
    )
    reference_ref = datastore.build_mapping_reference(neighbors)
    reference = datastore.get_mapping_reference(reference_ref)

    assert reference.method == "symphony"
    assert reference.symphony_state is not None
    assert reference.batch_correction == mapping_source.correction
    assert reference.reduction == mapping_source.reduction
    assert reference.symphony_state.n_dims == reference.model.n_dims
    assert reference.symphony_state.n_clusters == 3
    assert reference.metadata["batch_columns"] == ["mapping_batch"]
    assert reference.metadata["harmony_parameters"]["nclust"] == 3
    # The first selected cell is in batch b, but Harmony encodes the plain
    # labels in sorted order and the reference copies that order.
    assert reference.metadata["batch_levels"] == [["a", "b"]]
    # Harmony stores centroids as dimensions by clusters; the reference keeps
    # them as clusters by dimensions, like its other centroid arrays.
    harmony = artifact_group(datastore.zw, mapping_source.correction)
    assert reference.metadata["batch_levels"] == harmony.attrs["batch_levels"]
    np.testing.assert_array_equal(
        reference.symphony_state.centroids,
        np.asarray(harmony["centroids"][:]).T,
    )
    for name in ("raw_centroids", "corrected_centroids", "cluster_mass", "sigma"):
        np.testing.assert_array_equal(
            getattr(reference.symphony_state, name), harmony[name][:]
        )

    group = artifact_group(datastore.zw, reference.ref)
    assert set(group.array_keys()) == _COMMON_ARRAYS | _SYMPHONY_ARRAYS
    assert set(group.attrs) == {
        "artifact_id",
        "kind",
        "provenance",
        "execution_options",
        "created_at_ns",
        "scarf_version",
        "complete",
        "reference_metadata",
        "payload_fingerprint",
    }
    status = datastore.inspect_artifact(reference.ref)
    assert status.parameters == {"method": "symphony"}
    assert set((status.inputs or {})) == {
        "reduction",
        "batch_correction",
        "ann_index",
        "neighbors",
        "cell_selection",
        "feature_selection",
    }

    read_only = DataStore(
        datastore.zarr_loc,
        default_assay="RNA",
        zarr_mode="r",
    )
    loaded = read_only.get_mapping_reference(reference.ref)
    assert loaded.ref == reference.ref
    assert loaded.symphony_state is not None
    np.testing.assert_array_equal(loaded.feature_ids, reference.feature_ids)
    np.testing.assert_array_equal(
        loaded.symphony_state.corrected_centroids,
        reference.symphony_state.corrected_centroids,
    )


def test_symphony_mapping_reference_requires_recorded_batch_levels(
    mapping_source,
    reference_store,
):
    datastore = reference_store
    del artifact_group(datastore.zw, mapping_source.correction).attrs["batch_levels"]

    with pytest.raises(ValueError, match="Re-run run_harmony"):
        datastore.build_mapping_reference(mapping_source.symphony_neighbors)


def test_symphony_mapping_reference_load_rejects_each_tampered_record(
    mapping_source,
    reference_store,
):
    from scarf.mapping.artifact import load_artifact_mapping_reference

    datastore = reference_store
    reference = datastore.get_mapping_reference(
        datastore.build_mapping_reference(mapping_source.symphony_neighbors)
    )
    group = artifact_group(datastore.zw, reference.ref)
    correction = artifact_group(datastore.zw, reference.batch_correction)
    original = correction.attrs["provenance"]
    for section, key, value, message in (
        (None, "operation", "run_scanorama", "correction is not Harmony"),
        ("inputs", "reduction", reference.feature_selection.to_dict(), "different PCA"),
    ):
        provenance = dict(original)
        if section is None:
            provenance[key] = value
        else:
            provenance[section] = dict(provenance[section]) | {key: value}
        correction.attrs["provenance"] = provenance
        with pytest.raises(ValueError, match=message):
            load_artifact_mapping_reference(datastore, reference.ref)
        correction.attrs["provenance"] = original

    for name, index, value, message in (
        ("sigma", 0, 0.0, "Symphony correction model is invalid"),
        ("centroids", (0, 0), None, "Symphony model changed from its input"),
    ):
        array = group[name]
        stored = array[index]
        array[index] = stored + 1.0 if value is None else value
        with pytest.raises(ValueError, match=message):
            load_artifact_mapping_reference(datastore, reference.ref)
        array[index] = stored

    metadata = dict(group.attrs["reference_metadata"])
    group.attrs["reference_metadata"] = metadata | {"batch_columns": ["other"]}
    with pytest.raises(ValueError, match="Symphony metadata does not match"):
        load_artifact_mapping_reference(datastore, reference.ref)
    group.attrs["reference_metadata"] = metadata
    assert load_artifact_mapping_reference(datastore, reference.ref).ref == (
        reference.ref
    )

    data = correction["data"]
    data.resize((data.shape[0], data.shape[1] + 1))
    with pytest.raises(ValueError, match="Harmony coordinates do not match"):
        load_artifact_mapping_reference(datastore, reference.ref)


@pytest.mark.parametrize("missing", ["batch_levels", "batch_columns"])
def test_symphony_mapping_reference_load_requires_recorded_batch_metadata(
    mapping_source,
    reference_store,
    missing,
):
    datastore = reference_store
    reference_ref = datastore.build_mapping_reference(mapping_source.symphony_neighbors)
    group = artifact_group(datastore.zw, mapping_source.correction)
    if missing == "batch_levels":
        del group.attrs["batch_levels"]
    else:
        provenance = dict(group.attrs["provenance"])
        parameters = dict(provenance["parameters"])
        del parameters["batch_columns"]
        group.attrs["provenance"] = {**provenance, "parameters": parameters}

    with pytest.raises(ValueError, match="Re-run run_harmony"):
        datastore.get_mapping_reference(reference_ref)


def test_symphony_mapping_reference_of_an_earlier_harmony_record_still_loads(
    mapping_source,
    reference_store,
):
    datastore = reference_store
    # Record the correction as releases before 1.0.0 did: without a revision,
    # and with the levels in order of first appearance among the selected
    # cells. Those releases recorded the same parameters, the frozen
    # algorithm_version included.
    correction = artifact_group(datastore.zw, mapping_source.correction)
    provenance = {
        key: value
        for key, value in correction.attrs["provenance"].items()
        if key != "revision"
    }
    assert provenance["parameters"]["algorithm_version"] == "centroid_snapshot_v2"
    correction.attrs["provenance"] = provenance
    correction.attrs["batch_levels"] = [["b", "a"]]
    status = datastore.inspect_artifact(mapping_source.correction)
    assert status.revision == 1
    assert not status.is_current

    reference = datastore.get_mapping_reference(
        datastore.build_mapping_reference(mapping_source.symphony_neighbors)
    )

    # A reference keeps the record of its own correction, so a superseded
    # correction still builds and loads its reference.
    assert reference.batch_correction == mapping_source.correction
    assert reference.metadata["batch_levels"] == [["b", "a"]]
    assert reference.metadata["harmony_parameters"] == {"nclust": 3}


def test_loaded_mapping_reference_is_deeply_immutable(
    mapping_source,
    reference_store,
):
    datastore = reference_store
    reference = datastore.get_mapping_reference(
        datastore.build_mapping_reference(mapping_source.symphony_neighbors)
    )
    assert reference.symphony_state is not None

    arrays = (
        reference.model.feature_means,
        reference.model.feature_scales,
        reference.model.center,
        reference.model.loadings,
        reference.symphony_state.centroids,
        reference.symphony_state.corrected_centroids,
        reference.feature_ids,
        reference.reference_distance_quantiles,
        reference.reference_distance_values,
    )
    for values in arrays:
        assert not values.flags.writeable
        with pytest.raises(ValueError, match="cannot set WRITEABLE flag"):
            values.flags.writeable = True

    with pytest.raises(TypeError, match="does not support item assignment"):
        reference.metadata["method"] = "pca"  # type: ignore[index]
    with pytest.raises(TypeError, match="does not support item assignment"):
        reference.metadata["harmony_parameters"]["nclust"] = 9  # type: ignore[index]
    # Frozen lists keep list equality in both directions.
    batch_columns = reference.metadata["batch_columns"]
    with pytest.raises(AttributeError, match="append"):
        batch_columns.append("other")  # type: ignore[attr-defined]
    assert batch_columns == ["mapping_batch"]
    assert not batch_columns != ["mapping_batch"]
    assert ["mapping_batch"] == batch_columns
    assert not ["mapping_batch"] != batch_columns
    assert batch_columns != ["other"]
    normalization = reference.normalization_parameters
    normalization["size_factor"] = -1
    assert reference.size_factor == 1000.0
    assert reference.current_model_digest() == reference.model_digest


def test_mapping_reference_binding_and_publication_validation_are_blockwise(
    mapping_source,
    reference_store,
    monkeypatch,
):
    import scarf.mapping.artifact as mapping_artifact

    datastore = reference_store
    neighbors = mapping_source.neighbors
    reference_ref = datastore.build_mapping_reference(neighbors)
    reference = datastore.get_mapping_reference(reference_ref)
    _reject_full_array_reads(
        monkeypatch,
        _mapping_reference_source_paths(datastore, neighbors),
    )

    def fail_materialization(*args, **kwargs):
        raise AssertionError("mapping-reference validation materialized an array")

    monkeypatch.setattr(mapping_artifact, "_values", fail_materialization)
    assert mapping_artifact.validate_mapping_reference_binding(reference) is reference

    replacement = datastore.build_mapping_reference(
        neighbors,
        invalidate_cache=True,
    )
    assert replacement != reference_ref
    assert datastore.inspect_artifact(replacement).complete


def test_mapping_reference_source_streaming_uses_bounded_explicit_slices(
    monkeypatch,
):
    import scarf.mapping.artifact as mapping_artifact

    root = zarr.open_group(store=MemoryStore(), mode="w")
    source_group = root.create_group("sources")
    target_group = root.create_group("reference")
    n_features = 10_001
    n_dims = 2
    feature_means = source_group.create_array(
        "feature_means",
        data=np.linspace(0.0, 1.0, n_features, dtype=np.float64),
        chunks=(1_000,),
    )
    feature_scales = source_group.create_array(
        "feature_scales",
        data=np.ones(n_features, dtype=np.float64),
        chunks=(1_000,),
    )
    center = source_group.create_array(
        "center",
        data=np.linspace(-1.0, 1.0, n_features, dtype=np.float64),
        chunks=(1_000,),
    )
    loadings = source_group.create_array(
        "loadings",
        data=np.ones((n_features, n_dims), dtype=np.float64),
        chunks=(1_000, n_dims),
    )
    sources = {
        feature_means.path,
        feature_scales.path,
        center.path,
        loadings.path,
    }
    row_spans = _reject_full_array_reads(monkeypatch, sources)
    feature_ids = np.arange(n_features).astype(str)
    metadata = {"method": "pca"}
    quantiles = np.asarray([0.0, 1.0], dtype=np.float64)
    distance_values = np.asarray([0.5, 1.5], dtype=np.float64)

    assert mapping_artifact.validate_mapping_reference_sources(
        feature_means=feature_means,
        feature_scales=feature_scales,
        center=center,
        loadings=loadings,
        symphony_sources=None,
    ) == (n_features, n_dims)
    source_fingerprint = mapping_artifact.mapping_reference_source_fingerprint(
        feature_means=feature_means,
        feature_scales=feature_scales,
        center=center,
        loadings=loadings,
        symphony_sources=None,
    )
    mapping_artifact.write_artifact_mapping_reference_from_sources(
        target_group,
        feature_means=feature_means,
        feature_scales=feature_scales,
        center=center,
        loadings=loadings,
        symphony_sources=None,
        feature_ids=feature_ids,
        metadata=metadata,
        reference_distance_quantiles=quantiles,
        reference_distance_values=distance_values,
    )
    assert mapping_artifact.mapping_reference_payload_matches_sources(
        target_group,
        feature_means=feature_means,
        feature_scales=feature_scales,
        center=center,
        loadings=loadings,
        symphony_sources=None,
        feature_ids=feature_ids,
        metadata=metadata,
        reference_distance_quantiles=quantiles,
        reference_distance_values=distance_values,
        expected_source_fingerprint=source_fingerprint,
    )
    assert row_spans[feature_means.path]
    assert max(row_spans[feature_means.path]) <= 10_000
    assert row_spans[feature_scales.path]
    assert max(row_spans[feature_scales.path]) <= 10_000
    assert row_spans[center.path]
    assert max(row_spans[center.path]) <= 10_000
    assert row_spans[loadings.path]
    assert max(row_spans[loadings.path]) <= 1_000
    monkeypatch.undo()
    np.testing.assert_array_equal(target_group["loadings"][:], loadings[:])
    np.testing.assert_array_equal(target_group["center"][:], center[:])


@pytest.mark.parametrize("array_name", ["loadings", "center"])
def test_mapping_reference_rejects_valid_shaped_payload_tampering(
    mapping_source,
    reference_store,
    array_name,
):
    datastore = reference_store
    reference = datastore.get_mapping_reference(
        datastore.build_mapping_reference(mapping_source.neighbors)
    )
    group = artifact_group(datastore.zw, reference.ref)
    values = group[array_name]
    index = (0, 0) if array_name == "loadings" else 0
    values[index] = float(values[index]) + 0.25

    with pytest.raises(ValueError, match="PCA model changed from its inputs"):
        datastore.get_mapping_reference(reference.ref)

    replacement = datastore.build_mapping_reference(reference.neighbors)
    assert replacement != reference.ref


def test_mapping_reference_binds_distance_summary_to_neighbor_input(
    mapping_source,
    reference_store,
):
    import scarf.mapping.artifact as mapping_artifact

    datastore = reference_store
    reference_ref = datastore.build_mapping_reference(mapping_source.neighbors)
    group = artifact_group(datastore.zw, reference_ref)
    values = group["reference_distance_values"]
    values[:] = np.asarray(values[:], dtype=np.float64) + 0.25
    metadata = dict(group.attrs["reference_metadata"])
    group.attrs["payload_fingerprint"] = mapping_artifact._payload_fingerprint(
        group,
        metadata["method"],
        metadata,
    )

    with pytest.raises(ValueError, match="changed from its neighbor input"):
        datastore.get_mapping_reference(reference_ref)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("duplicate", "feature IDs must be unique"),
        ("drop", "Selected reference features do not match PCA loadings"),
    ],
)
def test_mapping_reference_rejects_selected_feature_ids_unlike_the_loadings(
    mapping_source,
    reference_store,
    monkeypatch,
    change,
    message,
):
    import scarf.datastore._operations.mapping_reference as reference_operations

    datastore = reference_store
    original = reference_operations._selected_feature_ids

    def changed_feature_ids(*args, **kwargs):
        feature_ids = np.array(original(*args, **kwargs), copy=True)
        if change == "drop":
            return feature_ids[:-1]
        feature_ids[1] = feature_ids[0]
        return feature_ids

    monkeypatch.setattr(
        reference_operations,
        "_selected_feature_ids",
        changed_feature_ids,
    )
    with pytest.raises(ValueError, match=message):
        datastore.build_mapping_reference(mapping_source.neighbors)
    assert datastore.list_artifacts(kind="mapping_reference", from_assay="RNA") == []


def test_mapping_reference_validates_payload_before_finish(
    mapping_source,
    reference_store,
    monkeypatch,
):
    import scarf.datastore._operations.mapping_reference as reference_operations

    datastore = reference_store
    neighbors = mapping_source.neighbors
    before = set(
        datastore.list_artifacts(
            kind="mapping_reference",
            from_assay="RNA",
        )
    )
    original_writer = reference_operations.write_artifact_mapping_reference_from_sources

    def corrupt_written_reference(group, *args, **kwargs):
        original_writer(group, *args, **kwargs)
        scales = group["feature_scales"]
        scales[0] = float(scales[0]) + 0.25

    monkeypatch.setattr(
        reference_operations,
        "write_artifact_mapping_reference_from_sources",
        corrupt_written_reference,
    )
    with pytest.raises(ValueError, match="reuse contract"):
        datastore.build_mapping_reference(neighbors, invalidate_cache=True)

    created = (
        set(
            datastore.list_artifacts(
                kind="mapping_reference",
                from_assay="RNA",
            )
        )
        - before
    )
    # A failed write deletes its incomplete slot.
    assert created == set()


@pytest.mark.parametrize("array_name", ["loadings", "center"])
def test_mapping_reference_rejects_source_mutation_during_publication(
    mapping_source,
    reference_store,
    monkeypatch,
    array_name,
):
    import scarf.datastore._operations.mapping_reference as reference_operations

    datastore = reference_store
    neighbors = mapping_source.neighbors
    before = set(
        datastore.list_artifacts(
            kind="mapping_reference",
            from_assay="RNA",
        )
    )
    original_writer = reference_operations.write_artifact_mapping_reference_from_sources

    def mutate_source_then_write(group, *args, **kwargs):
        source = kwargs[array_name]
        index = (0, 0) if array_name == "loadings" else 0
        source[index] = float(source[index]) + 0.25
        original_writer(group, *args, **kwargs)

    monkeypatch.setattr(
        reference_operations,
        "write_artifact_mapping_reference_from_sources",
        mutate_source_then_write,
    )

    with pytest.raises(ValueError, match="reuse contract"):
        datastore.build_mapping_reference(neighbors, invalidate_cache=True)

    created = (
        set(
            datastore.list_artifacts(
                kind="mapping_reference",
                from_assay="RNA",
            )
        )
        - before
    )
    # A failed write deletes its incomplete slot.
    assert created == set()


def _flatten_neighbor_distances(group: zarr.Group) -> None:
    n_cells = int(group["indices"].shape[0])
    del group["distances"]
    group.create_array(
        "distances",
        data=np.zeros(n_cells, dtype=np.float32),
        chunks=(max(1, min(n_cells, 10)),),
    )


def _drop_last_neighbor_row(group: zarr.Group) -> None:
    """Store a valid neighbor graph over one cell fewer than the selection."""
    n_cells = int(group["indices"].shape[0]) - 1
    n_neighbors = int(group["indices"].shape[1])
    rows = np.arange(n_cells)[:, np.newaxis]
    indices = (rows + 1 + np.arange(n_neighbors)) % n_cells
    distances = np.tile(
        np.linspace(1.0, 2.0, n_neighbors, dtype=np.float32), (n_cells, 1)
    )
    for name, values in (
        ("indices", indices.astype(np.uint32)),
        ("distances", distances),
    ):
        del group[name]
        group.create_array(name, data=values)
    group.attrs["n_cells"] = n_cells


@pytest.mark.parametrize(
    ("tamper", "load_cause", "build_message"),
    [
        (_flatten_neighbor_distances, "stored dimensions", "stored dimensions"),
        (
            _drop_last_neighbor_row,
            "Neighbors do not match the selected reference cell count",
            "Neighbor rows must match the selected reference cells",
        ),
    ],
    ids=["flat-distances", "fewer-rows"],
)
def test_mapping_reference_rejects_corrupt_neighbor_payload_on_build_and_load(
    mapping_source,
    reference_store,
    tamper,
    load_cause,
    build_message,
):
    datastore = reference_store
    neighbors = mapping_source.neighbors
    reference_ref = datastore.build_mapping_reference(neighbors)
    tamper(artifact_group(datastore.zw, neighbors))

    with pytest.raises(ValueError, match="neighbor distances are invalid") as caught:
        datastore.get_mapping_reference(reference_ref)
    assert load_cause in str(caught.value.__cause__)
    with pytest.raises(ValueError, match=build_message):
        datastore.build_mapping_reference(neighbors, invalidate_cache=True)


def test_mapping_reference_rejects_corrupt_ann_payload_on_build_and_load(
    mapping_source,
    reference_store,
):
    datastore = reference_store
    neighbors = mapping_source.neighbors
    reference = datastore.get_mapping_reference(
        datastore.build_mapping_reference(neighbors)
    )
    ann_group = artifact_group(datastore.zw, reference.ann_index)
    payload = ann_group["ann_idx_bytes"]
    payload[0] = np.uint8(int(payload[0]) ^ 1)

    with pytest.raises(ValueError, match="ANN index payload is invalid"):
        datastore.get_mapping_reference(reference.ref)
    with pytest.raises(ValueError, match="payload digest"):
        datastore.build_mapping_reference(neighbors, invalidate_cache=True)


def test_get_mapping_reference_requires_a_mapping_reference_artifact(
    mapping_source,
    reference_store,
):
    datastore = reference_store

    with pytest.raises(TypeError, match="reference must be an ArtifactRef"):
        datastore.get_mapping_reference("reference")
    for ref in (
        mapping_source.neighbors,
        ArtifactRef(
            scope="datastore",
            kind="mapping_reference",
            artifact_id="a" * 64,
        ),
    ):
        with pytest.raises(ValueError, match="assay-scoped mapping_reference"):
            datastore.get_mapping_reference(ref)


def test_build_mapping_reference_validates_arguments_before_reading(
    mapping_source,
    reference_store,
):
    datastore = reference_store
    read_only = DataStore(datastore.zarr_loc, default_assay="RNA", zarr_mode="r")
    adt_neighbors = ArtifactRef(
        scope="assay",
        assay="ADT",
        kind="neighbors",
        artifact_id="b" * 64,
    )

    with pytest.raises(ValueError, match="requires a read-write store"):
        read_only.build_mapping_reference(mapping_source.neighbors)
    for arguments, error, message in (
        ({"neighbors": "neighbors"}, TypeError, "neighbors must be an ArtifactRef"),
        (
            {"neighbors": mapping_source.neighbors, "invalidate_cache": 1},
            TypeError,
            "invalidate_cache must be a boolean",
        ),
        (
            {"neighbors": mapping_source.reduction},
            ValueError,
            "neighbors must identify an assay-scoped neighbors artifact",
        ),
        ({"neighbors": adt_neighbors}, TypeError, "support RNA assays only"),
    ):
        with pytest.raises(error, match=message):
            datastore.build_mapping_reference(**arguments)
    assert datastore.list_artifacts(kind="mapping_reference", from_assay="RNA") == []


def _source_groups(datastore, source: MappingStoreSource) -> dict[str, zarr.Group]:
    normalized = source.normalized
    return {
        "neighbors": artifact_group(datastore.zw, source.neighbors),
        "ann_index": artifact_group(
            datastore.zw, _input_ref(datastore, source.neighbors, "ann_index")
        ),
        "reduction": artifact_group(datastore.zw, source.reduction),
        "normalized": artifact_group(datastore.zw, normalized),
        "scaling": artifact_group(
            datastore.zw,
            _input_ref(datastore, source.reduction, "feature_scaling"),
        ),
        "correction": artifact_group(datastore.zw, source.correction),
    }


def test_build_mapping_reference_rejects_each_inconsistent_chain_record(
    mapping_source,
    reference_store,
):
    datastore = reference_store
    groups = _source_groups(datastore, mapping_source)
    datastore.cells.insert(
        "no_cells", np.zeros(MAPPING_N_CELLS, dtype=bool), overwrite=True
    )
    empty_cells = datastore.snapshot_cell_selection("no_cells").to_dict()
    other_kind = mapping_source.features.to_dict()
    other_reduction = mapping_source.other_reduction.to_dict()
    plain = mapping_source.neighbors
    symphony = mapping_source.symphony_neighbors
    for neighbors, target, section, key, value, message in (
        (plain, "neighbors", None, "operation", "old_knn", "require query_neighbors"),
        (plain, "ann_index", None, "operation", "old_ann", "require build_ann_index"),
        (
            plain,
            "neighbors",
            "inputs",
            "coordinates",
            other_kind,
            "must be a PCA reduction or batch correction",
        ),
        (
            plain,
            "ann_index",
            "inputs",
            "coordinates",
            other_reduction,
            "Neighbors and ANN index use different coordinates",
        ),
        (
            symphony,
            "correction",
            None,
            "operation",
            "run_scanorama",
            "require run_harmony correction",
        ),
        (plain, "reduction", None, "operation", "run_svd", "require a run_pca"),
        (
            plain,
            "reduction",
            "parameters",
            "feat_scaling",
            False,
            "require PCA with feature scaling",
        ),
        (
            plain,
            "normalized",
            None,
            "operation",
            "old_norm",
            "require a run_normalization artifact",
        ),
        (
            plain,
            "normalized",
            "inputs",
            "dataset_fingerprint",
            "",
            "Normalized artifact is missing its dataset fingerprint",
        ),
        (
            plain,
            "scaling",
            "parameters",
            "enabled",
            False,
            "require enabled scaling for the same normalized data",
        ),
        (plain, "ann_index", "parameters", "ann_metric", "ip", "only l2 and cosine"),
        (
            plain,
            "neighbors",
            "parameters",
            "distance_metric",
            "cosine",
            "Neighbor and ANN distance metrics do not match",
        ),
        (
            plain,
            "normalized",
            "inputs",
            "cell_selection",
            empty_cells,
            "require at least one selected cell",
        ),
    ):
        group = groups[target]
        original = group.attrs["provenance"]
        _edit_provenance(group, section, key, value)
        try:
            with pytest.raises(ValueError, match=message):
                datastore.build_mapping_reference(neighbors)
        finally:
            group.attrs["provenance"] = original
    assert datastore.list_artifacts(kind="mapping_reference", from_assay="RNA") == []
    # The restored records build both references.
    for neighbors in (plain, symphony):
        assert datastore.inspect_artifact(
            datastore.build_mapping_reference(neighbors)
        ).complete


def _resize_columns(name: str, change: int):
    def resize(group: zarr.Group) -> None:
        array = group[name]
        array.resize((array.shape[0], array.shape[1] + change))

    return resize


def _replace_values(name: str, shape_change: tuple[int, ...]):
    def replace(group: zarr.Group) -> None:
        values = np.asarray(group[name][:])
        del group[name]
        group.create_array(
            name,
            data=np.ones(
                tuple(
                    size + change
                    for size, change in zip(values.shape, shape_change, strict=True)
                ),
                dtype=values.dtype,
            ),
        )

    return replace


def _flatten_loadings(group: zarr.Group) -> None:
    values = np.asarray(group["loadings"][:])
    del group["loadings"]
    group.create_array("loadings", data=values[:, 0].copy())


@pytest.mark.parametrize(
    ("method", "target", "tamper", "message"),
    [
        (
            "pca",
            "reduction",
            _flatten_loadings,
            "Reference PCA loadings have incompatible dimensions",
        ),
        (
            "pca",
            "reduction",
            _resize_columns("data", 1),
            "PCA rows must match the selected reference cells and dimensions",
        ),
        (
            "symphony",
            "correction",
            _resize_columns("data", 1),
            "Harmony coordinates do not match the reference PCA dimensions",
        ),
        (
            "symphony",
            "correction",
            _replace_values("centroids", (1, 0)),
            "Harmony correction dimensions do not match PCA loadings",
        ),
        (
            "symphony",
            "correction",
            _replace_values("sigma", (1,)),
            "Harmony correction arrays have incompatible dimensions",
        ),
    ],
    ids=[
        "flat-loadings",
        "pca-columns",
        "harmony-columns",
        "harmony-centroid-rows",
        "harmony-sigma-length",
    ],
)
def test_build_mapping_reference_rejects_misshapen_source_payloads(
    mapping_source,
    reference_store,
    method,
    target,
    tamper,
    message,
):
    datastore = reference_store
    neighbors = (
        mapping_source.neighbors
        if method == "pca"
        else mapping_source.symphony_neighbors
    )
    tamper(_source_groups(datastore, mapping_source)[target])

    with pytest.raises(ValueError, match=message):
        datastore.build_mapping_reference(neighbors)
    assert datastore.list_artifacts(kind="mapping_reference", from_assay="RNA") == []


def test_mapping_reference_rejects_array_attributes(
    mapping_source,
    reference_store,
):
    datastore = reference_store
    reference_ref = datastore.build_mapping_reference(mapping_source.neighbors)
    artifact_group(datastore.zw, reference_ref)["loadings"].attrs["schema_version"] = 1

    with pytest.raises(ValueError, match="array attributes"):
        datastore.get_mapping_reference(reference_ref)


def test_mapping_reference_rejects_unsupported_normalization_at_build_time(
    mapping_source,
    reference_store,
):
    datastore = reference_store
    group = artifact_group(datastore.zw, mapping_source.normalized)
    _edit_provenance(group, "parameters", "normalization_method", "unsupported")

    with pytest.raises(ValueError, match="Unsupported reference normalization"):
        datastore.build_mapping_reference(mapping_source.neighbors)


def test_mapping_reference_rejects_incomplete_contract(
    mapping_source,
    reference_store,
):
    datastore = reference_store
    reference_ref = datastore.build_mapping_reference(mapping_source.neighbors)
    group = artifact_group(datastore.zw, reference_ref)
    del group["feature_scales"]
    with pytest.raises(ValueError, match="build_mapping_reference\\(neighbors\\)"):
        datastore.get_mapping_reference(reference_ref)


def test_mapping_reference_validates_live_dataset_fingerprint(
    mapping_source,
    reference_store,
):
    datastore = reference_store
    reference_ref = datastore.build_mapping_reference(mapping_source.neighbors)
    # Normalized data recorded for another dataset cannot become a reference.
    normalized = artifact_group(datastore.zw, mapping_source.normalized)
    original = normalized.attrs["provenance"]
    _edit_provenance(normalized, "inputs", "dataset_fingerprint", "stale-dataset")
    with pytest.raises(
        ValueError,
        match="Normalized artifact does not match the current reference dataset",
    ):
        datastore.build_mapping_reference(
            mapping_source.neighbors, invalidate_cache=True
        )
    normalized.attrs["provenance"] = original

    datastore.RNA.attrs["dataset_fingerprint"] = "changed"
    with pytest.raises(ValueError, match="dataset fingerprint"):
        datastore.get_mapping_reference(reference_ref)


def test_mapping_reference_rejects_nonmonotonic_distance_summary(
    mapping_source,
    reference_store,
):
    datastore = reference_store
    reference_ref = datastore.build_mapping_reference(mapping_source.neighbors)
    values = artifact_group(datastore.zw, reference_ref)["reference_distance_values"]
    values[:] = np.linspace(1.0, 0.0, values.shape[0])

    with pytest.raises(ValueError, match="distance summary"):
        datastore.get_mapping_reference(reference_ref)
