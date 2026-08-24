from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

import scarf.mapping.projection as projection_storage
from scarf.mapping.models import ScaledPCAProjectionModel
from scarf.mapping.projection import (
    NO_QUERY_BATCH_FINGERPRINT,
    ProjectionWriter,
    load_projection,
    plan_projection,
    resolve_projection,
)
from scarf.mapping.reference import MappingReference
from scarf.storage.artifact_writer import (
    finish_artifact,
    plan_artifact,
    start_artifact,
)
from scarf.storage.artifacts import (
    ArtifactRef,
    ExternalArtifactRef,
    artifact_group,
    inspect_artifact,
    list_artifacts,
)


def _selection(
    root: zarr.Group,
    *,
    kind: str,
    values: np.ndarray,
    assay: str | None,
) -> ArtifactRef:
    scope = "datastore" if assay is None else "assay"
    planned = plan_artifact(
        root,
        scope=scope,
        assay=assay,
        kind=kind,
        operation="manual_selection",
        parameters={},
        inputs={},
        execution_options={},
    )
    group = start_artifact(root, planned)
    group.create_array(
        "values",
        data=np.asarray(values, dtype=bool),
        chunks=(len(values),),
    )
    finish_artifact(group, planned)
    return planned.ref


def _query_inputs(
    *,
    n_cells: int = 4,
) -> tuple[zarr.Group, ArtifactRef, ArtifactRef]:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    cell_selection = _selection(
        root,
        kind="cell_selection",
        values=np.ones(n_cells, dtype=bool),
        assay=None,
    )
    feature_selection = _selection(
        root,
        kind="feature_selection",
        values=np.array([True, False, True]),
        assay="RNA",
    )
    return root, cell_selection, feature_selection


def _artifact_ref(
    kind: str,
    token: str,
    *,
    assay: str | None = "RNA",
) -> ArtifactRef:
    return ArtifactRef(
        scope="datastore" if assay is None else "assay",
        assay=assay,
        kind=kind,
        artifact_id=token * 64,
    )


class _ReferenceDatastore:
    def __init__(self, root: zarr.Group) -> None:
        self.zw = root

    def _get_assay(self, name: str) -> SimpleNamespace:
        return SimpleNamespace(attrs=self.zw[name].attrs)


def _mapping_reference(
    *,
    fingerprint: str = "reference-dataset",
    token: str = "a",
) -> tuple[MappingReference, zarr.Group]:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    assay = root.create_group("RNA")
    assay.attrs["dataset_fingerprint"] = fingerprint
    reference = MappingReference(
        datastore=_ReferenceDatastore(root),
        ref=_artifact_ref("mapping_reference", token),
        assay_name="RNA",
        cell_key="I",
        feature_key="I",
        reduction=_artifact_ref("reduction", "1"),
        ann_index=_artifact_ref("ann_index", "2"),
        neighbors=_artifact_ref("neighbors", "3"),
        cell_selection=_artifact_ref("cell_selection", "4", assay=None),
        feature_selection=_artifact_ref("feature_selection", "5"),
        batch_correction=None,
        dataset_fingerprint=fingerprint,
        selected_cell_count=3,
        model=ScaledPCAProjectionModel(
            feature_means=np.zeros(1),
            feature_scales=np.ones(1),
            loadings=np.ones((1, 1)),
        ),
        symphony_state=None,
        feature_ids=np.array(["gene"]),
        metadata={
            "method": "pca",
            "ann_metric": "l2",
            "normalization_parameters": {"size_factor": 1.0},
        },
        reference_distance_quantiles=np.array([0.5]),
        reference_distance_values=np.array([1.0]),
    )
    return reference, root


def _plan(
    root: zarr.Group,
    cell_selection: ArtifactRef,
    feature_selection: ArtifactRef,
    external: ExternalArtifactRef,
    *,
    mapping_name: str = "atlas",
    n_cells: int = 4,
    save_k: int = 2,
    missing_feature_policy: str = "reference_mean",
    correction_method: str = "none",
    invalidate_cache: bool = False,
):
    return plan_projection(
        root,
        query_assay="RNA",
        mapping_name=mapping_name,
        n_cells=n_cells,
        save_k=save_k,
        missing_feature_policy=missing_feature_policy,
        correction_method=correction_method,
        cell_selection=cell_selection,
        feature_selection=feature_selection,
        selected_expression_fingerprint="e" * 64,
        query_batch_fingerprint=NO_QUERY_BATCH_FINGERPRINT,
        mapping_reference=external,
        reference_cell_count=3,
        invalidate_cache=invalidate_cache,
    )


def _blocks() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.array([[0, 1], [1, 2], [2, 1], [0, 2]], dtype=np.uint32),
        np.array(
            [[0.1, 0.2], [0.0, 1.0], [2.0, 3.0], [0.5, 0.75]],
            dtype=np.float32,
        ),
        np.array([False, True, False, True]),
    )


def _diagnostics(*, zero_norm_cell_count: int = 2) -> dict[str, object]:
    return {
        "featureCoverage": 0.75,
        "queryBatchCount": 1,
        "algorithmVariant": "scaled_pca",
        "zeroNormCellCount": zero_norm_cell_count,
        "queryScaledDispersion": 1.0,
    }


def _write(
    root: zarr.Group,
    plan,
    *,
    created_at_ns: int | None = None,
) -> ArtifactRef:
    indices, distances, uninformative = _blocks()
    writer = ProjectionWriter(root, plan, chunk_rows=2)
    writer.write_block(0, indices[:2], distances[:2], uninformative[:2])
    writer.write_block(2, indices[2:], distances[2:], uninformative[2:])
    ref = writer.finish(_diagnostics())
    if created_at_ns is not None:
        artifact_group(root, ref).attrs["created_at_ns"] = created_at_ns
    return ref


def _replace_array(
    group: zarr.Group,
    name: str,
    values: np.ndarray,
) -> None:
    del group[name]
    chunks = tuple(max(1, min(int(size), 2)) for size in values.shape)
    group.create_array(name, data=values, chunks=chunks)


def test_projection_writer_persists_exact_contract_and_loads_copies() -> None:
    root, cell_selection, feature_selection = _query_inputs()
    reference, reference_root = _mapping_reference()
    external = reference.external_ref
    before_reference_attrs = dict(reference_root["RNA"].attrs)
    plan = _plan(root, cell_selection, feature_selection, external)

    assert not plan.reused
    assert not inspect_artifact(root, plan.ref).exists
    indices, distances, uninformative = _blocks()
    writer = ProjectionWriter(root, plan, chunk_rows=2)
    assert writer.ref == plan.ref
    assert writer.next_row == 0
    assert not inspect_artifact(root, plan.ref).complete
    writer.write_block(0, indices[:1], distances[:1], uninformative[:1])
    writer.write_block(1, indices[1:3], distances[1:3], uninformative[1:3])
    writer.write_block(3, indices[3:], distances[3:], uninformative[3:])
    assert writer.next_row == 4
    assert writer.finish(_diagnostics()) == plan.ref
    assert writer.finished
    with pytest.raises(RuntimeError, match="already finished"):
        writer.write_block(4, indices[:1], distances[:1], uninformative[:1])
    with pytest.raises(RuntimeError, match="cannot be aborted"):
        writer.abort()

    status = inspect_artifact(root, plan.ref)
    assert status.complete
    assert status.operation == "map_query"
    assert status.parameters == {
        "mapping_name": "atlas",
        "save_k": 2,
        "missing_feature_policy": "reference_mean",
        "correction_method": "none",
    }
    assert set(status.inputs or {}) == {
        "cell_selection",
        "feature_selection",
        "selected_expression_fingerprint",
        "query_batch_fingerprint",
        "mapping_reference",
    }
    assert (status.inputs or {})["mapping_reference"] == external.to_dict()
    group = artifact_group(root, plan.ref)
    assert set(group.array_keys()) == {"indices", "distances", "uninformative"}
    assert set(group.group_keys()) == set()
    assert group.attrs["diagnostics"] == _diagnostics()
    assert "ann_metric" not in (status.parameters or {})
    assert "reference_feature_indices" not in group

    metadata = load_projection(root, plan.ref)
    assert metadata.ref == plan.ref
    assert metadata.mapping_name == "atlas"
    assert metadata.n_cells == 4
    assert metadata.correction_method == "none"
    assert metadata.diagnostics == _diagnostics()
    assert metadata.indices is None
    assert metadata.distances is None
    assert metadata.uninformative is None
    assert metadata.reference is None

    loaded = load_projection(
        root,
        plan.ref,
        load_arrays=True,
        reference=reference,
    )
    np.testing.assert_array_equal(loaded.indices, indices)
    np.testing.assert_allclose(loaded.distances, distances)
    np.testing.assert_array_equal(loaded.uninformative, uninformative)
    assert loaded.reference is reference
    assert replace(metadata, reference=reference) == metadata
    assert "MappingReference" not in repr(loaded)
    assert "[[" not in repr(loaded)
    assert dict(reference_root["RNA"].attrs) == before_reference_attrs


@pytest.mark.parametrize(
    ("indices", "distances", "uninformative", "message"),
    [
        (
            np.ones((2, 2), dtype=np.int64),
            np.ones((2, 2), dtype=np.float64),
            np.zeros(2, dtype=bool),
            "unsigned",
        ),
        (
            np.ones((2, 2), dtype=np.uint32),
            np.ones((2, 2), dtype=np.int64),
            np.zeros(2, dtype=bool),
            "floating",
        ),
        (
            np.ones((2, 2), dtype=np.uint32),
            np.ones((2, 2), dtype=np.float64),
            np.zeros(2, dtype=np.uint8),
            "boolean",
        ),
        (
            np.ones((2, 2), dtype=np.uint32),
            np.ones((2, 1), dtype=np.float64),
            np.zeros(2, dtype=bool),
            "match the index block shape",
        ),
        (
            np.ones((2, 2), dtype=np.uint32),
            np.ones((2, 2), dtype=np.float64),
            np.zeros((2, 1), dtype=bool),
            "one value per row",
        ),
        (
            np.ones((0, 2), dtype=np.uint32),
            np.ones((0, 2), dtype=np.float64),
            np.zeros(0, dtype=bool),
            "cannot be empty",
        ),
        (
            np.zeros((5, 2), dtype=np.uint32),
            np.ones((5, 2), dtype=np.float64),
            np.zeros(5, dtype=bool),
            "exceeds the declared cell count",
        ),
        (
            np.ones((2, 2), dtype=np.uint32),
            np.array([[0.0, np.nan], [1.0, 2.0]]),
            np.zeros(2, dtype=bool),
            "finite",
        ),
        (
            np.ones((2, 2), dtype=np.uint32),
            np.array([[0.0, -1.0], [1.0, 2.0]]),
            np.zeros(2, dtype=bool),
            "non-negative",
        ),
        (
            np.ones((2, 1), dtype=np.uint32),
            np.ones((2, 1), dtype=np.float64),
            np.zeros(2, dtype=bool),
            "shape",
        ),
        (
            np.array([[0, 3], [1, 2]], dtype=np.uint32),
            np.ones((2, 2), dtype=np.float64),
            np.zeros(2, dtype=bool),
            "selected reference cells",
        ),
    ],
)
def test_projection_writer_aborts_after_invalid_block(
    indices: np.ndarray,
    distances: np.ndarray,
    uninformative: np.ndarray,
    message: str,
) -> None:
    root, cell_selection, feature_selection = _query_inputs()
    reference, _ = _mapping_reference()
    plan = _plan(root, cell_selection, feature_selection, reference.external_ref)
    writer = ProjectionWriter(root, plan, chunk_rows=2)

    with pytest.raises((TypeError, ValueError), match=message):
        writer.write_block(0, indices, distances, uninformative)

    assert writer.aborted
    assert not inspect_artifact(root, plan.ref).complete
    with pytest.raises(RuntimeError, match="aborted"):
        writer.finish(_diagnostics())


def test_projection_writer_constructor_and_start_failures_leave_incomplete_artifacts(
    monkeypatch,
) -> None:
    root, cell_selection, feature_selection = _query_inputs()
    reference, _ = _mapping_reference()
    first = _plan(root, cell_selection, feature_selection, reference.external_ref)
    original_create = projection_storage.create_zarr_dataset

    def fail_distances(group, name, *args, **kwargs):
        if name == "distances":
            raise RuntimeError("injected array creation failure")
        return original_create(group, name, *args, **kwargs)

    monkeypatch.setattr(
        projection_storage,
        "create_zarr_dataset",
        fail_distances,
    )
    with pytest.raises(RuntimeError, match="injected array creation failure"):
        ProjectionWriter(root, first, chunk_rows=2)
    assert not inspect_artifact(root, first.ref).complete

    monkeypatch.undo()
    second = _plan(
        root,
        cell_selection,
        feature_selection,
        reference.external_ref,
        invalidate_cache=True,
    )
    writer = ProjectionWriter(root, second, chunk_rows=2)
    indices, distances, uninformative = _blocks()
    with pytest.raises(TypeError, match="start must be an integer"):
        writer.write_block(True, indices[:1], distances[:1], uninformative[:1])
    assert writer.aborted
    assert not inspect_artifact(root, second.ref).complete


def test_projection_writer_requires_contiguous_complete_coverage_and_can_abort() -> (
    None
):
    root, cell_selection, feature_selection = _query_inputs()
    reference, _ = _mapping_reference()
    plan = _plan(root, cell_selection, feature_selection, reference.external_ref)
    indices, distances, uninformative = _blocks()
    writer = ProjectionWriter(root, plan, chunk_rows=2)

    writer.write_block(0, indices[:2], distances[:2], uninformative[:2])
    with pytest.raises(ValueError, match="wrote 2 of 4"):
        writer.finish(_diagnostics())
    assert writer.aborted
    assert not inspect_artifact(root, plan.ref).complete

    second_plan = _plan(
        root,
        cell_selection,
        feature_selection,
        reference.external_ref,
        invalidate_cache=True,
    )
    second = ProjectionWriter(root, second_plan, chunk_rows=2)
    second.abort()
    assert second.aborted
    assert not inspect_artifact(root, second_plan.ref).complete
    with pytest.raises(RuntimeError, match="aborted"):
        second.write_block(0, indices, distances, uninformative)

    third_plan = _plan(
        root,
        cell_selection,
        feature_selection,
        reference.external_ref,
        invalidate_cache=True,
    )
    third = ProjectionWriter(root, third_plan, chunk_rows=2)
    with pytest.raises(ValueError, match="expected 0, received 1"):
        third.write_block(1, indices[:1], distances[:1], uninformative[:1])
    assert third.aborted


def test_projection_writer_reuses_only_a_valid_complete_artifact() -> None:
    root, cell_selection, feature_selection = _query_inputs()
    reference, _ = _mapping_reference()
    first = _plan(root, cell_selection, feature_selection, reference.external_ref)
    ref = _write(root, first)

    reused = _plan(root, cell_selection, feature_selection, reference.external_ref)
    assert reused.reused
    assert reused.ref == ref
    assert load_projection(root, ref).diagnostics["zeroNormCellCount"] == 2
    with pytest.raises(ValueError, match="without a writer"):
        ProjectionWriter(root, reused, chunk_rows=2)

    artifact_group(root, ref).create_array(
        "extra",
        data=np.ones(1),
        chunks=(1,),
    )
    replacement = _plan(
        root,
        cell_selection,
        feature_selection,
        reference.external_ref,
    )
    assert not replacement.reused
    assert replacement.ref != ref


@pytest.mark.parametrize(
    "tamper",
    ["distance_shape", "index_dtype", "distance_value", "diagnostics"],
)
def test_projection_plan_rejects_tampered_cached_payload(tamper: str) -> None:
    root, cell_selection, feature_selection = _query_inputs()
    reference, _ = _mapping_reference()
    first = _plan(root, cell_selection, feature_selection, reference.external_ref)
    ref = _write(root, first)
    group = artifact_group(root, ref)
    if tamper == "distance_shape":
        _replace_array(
            group,
            "distances",
            np.ones((4, 1), dtype=np.float64),
        )
    elif tamper == "index_dtype":
        _replace_array(
            group,
            "indices",
            np.asarray(group["indices"][:], dtype=np.int64),
        )
    elif tamper == "distance_value":
        group["distances"][0, 0] = np.nan
    else:
        diagnostics = dict(group.attrs["diagnostics"])
        diagnostics["featureCoverage"] = 0.0
        group.attrs["diagnostics"] = diagnostics

    replacement = _plan(
        root,
        cell_selection,
        feature_selection,
        reference.external_ref,
    )

    assert not replacement.reused
    assert replacement.ref != ref


def test_projection_plan_rejects_out_of_range_cached_neighbors() -> None:
    root, cell_selection, feature_selection = _query_inputs()
    reference, _ = _mapping_reference()
    first = _plan(root, cell_selection, feature_selection, reference.external_ref)
    ref = _write(root, first)
    artifact_group(root, ref)["indices"][0, 0] = reference.selected_cell_count

    replacement = _plan(
        root,
        cell_selection,
        feature_selection,
        reference.external_ref,
    )

    assert not replacement.reused
    assert replacement.ref != ref


def test_projection_finish_rejects_diagnostics_inconsistent_with_rows() -> None:
    root, cell_selection, feature_selection = _query_inputs()
    reference, _ = _mapping_reference()
    plan = _plan(root, cell_selection, feature_selection, reference.external_ref)
    indices, distances, uninformative = _blocks()
    writer = ProjectionWriter(root, plan, chunk_rows=4)
    writer.write_block(0, indices, distances, uninformative)

    with pytest.raises(ValueError, match="number of uninformative"):
        writer.finish(_diagnostics(zero_norm_cell_count=1))

    assert not inspect_artifact(root, plan.ref).complete


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("featureCoverage", 0.0, "featureCoverage"),
        ("queryBatchCount", 5, "cannot exceed"),
        ("zeroNormCellCount", 5, "cannot exceed"),
        ("queryScaledDispersion", -1.0, "queryScaledDispersion"),
        ("unexpected", 1, "exactly"),
    ],
)
def test_projection_finish_rejects_invalid_diagnostics_and_aborts(
    field: str,
    value: object,
    message: str,
) -> None:
    root, cell_selection, feature_selection = _query_inputs()
    reference, _ = _mapping_reference()
    plan = _plan(root, cell_selection, feature_selection, reference.external_ref)
    indices, distances, uninformative = _blocks()
    writer = ProjectionWriter(root, plan, chunk_rows=4)
    writer.write_block(0, indices, distances, uninformative)
    diagnostics = _diagnostics()
    diagnostics[field] = value

    with pytest.raises((TypeError, ValueError), match=message):
        writer.finish(diagnostics)

    assert writer.aborted
    assert not inspect_artifact(root, plan.ref).complete


def _manual_projection(
    root: zarr.Group,
    cell_selection: ArtifactRef,
    feature_selection: ArtifactRef,
    mapping_reference: ArtifactRef | ExternalArtifactRef,
    *,
    operation: str = "map_query",
) -> ArtifactRef:
    planned = plan_artifact(
        root,
        scope="assay",
        assay="RNA",
        kind="projection",
        operation=operation,
        parameters={
            "mapping_name": "atlas",
            "save_k": 2,
            "missing_feature_policy": "reference_mean",
            "correction_method": "none",
        },
        inputs={
            "cell_selection": cell_selection,
            "feature_selection": feature_selection,
            "selected_expression_fingerprint": "e" * 64,
            "query_batch_fingerprint": NO_QUERY_BATCH_FINGERPRINT,
            "mapping_reference": mapping_reference,
        },
        execution_options={},
    )
    group = start_artifact(root, planned)
    indices, distances, uninformative = _blocks()
    group.create_array("indices", data=indices, chunks=(2, 2))
    group.create_array("distances", data=distances, chunks=(2, 2))
    group.create_array("uninformative", data=uninformative, chunks=(2,))
    group.attrs["diagnostics"] = _diagnostics()
    finish_artifact(group, planned)
    return planned.ref


def test_projection_loader_rejects_old_and_local_reference_contracts() -> None:
    root, cell_selection, feature_selection = _query_inputs()
    reference, _ = _mapping_reference()
    old = _manual_projection(
        root,
        cell_selection,
        feature_selection,
        reference.external_ref,
        operation="map_with_reference",
    )
    local = _manual_projection(
        root,
        cell_selection,
        feature_selection,
        reference.ref,
    )

    for ref in (old, local):
        with pytest.raises(ValueError, match="run_mapping"):
            load_projection(root, ref)


@pytest.mark.parametrize(
    "malformation",
    ["missing", "array", "group", "attribute", "diagnostics"],
)
def test_projection_loader_rejects_malformed_or_extra_payload(
    malformation: str,
) -> None:
    root, cell_selection, feature_selection = _query_inputs()
    reference, _ = _mapping_reference()
    ref = _write(
        root,
        _plan(root, cell_selection, feature_selection, reference.external_ref),
    )
    group = artifact_group(root, ref)
    if malformation == "missing":
        del group["distances"]
    elif malformation == "array":
        group.create_array("extra", data=np.ones(1), chunks=(1,))
    elif malformation == "group":
        group.create_group("extra")
    elif malformation == "attribute":
        group.attrs["extra"] = "invalid"
    else:
        diagnostics = dict(group.attrs["diagnostics"])
        diagnostics["zeroNormCellCount"] = 1
        group.attrs["diagnostics"] = diagnostics

    with pytest.raises(ValueError, match="run_mapping"):
        load_projection(root, ref)


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        ("index_dtype", "unsigned integer"),
        ("distance_dtype", "floating matrix"),
        ("distance_shape", "floating matrix"),
        ("uninformative_dtype", "boolean row vector"),
        ("uninformative_shape", "boolean row vector"),
        ("distance_nan", "finite"),
        ("distance_negative", "non-negative"),
        ("selection_count", "stored query cell selection"),
        ("diagnostics_type", "diagnostics must be a mapping"),
        ("created_at", "created_at_ns"),
    ],
)
def test_projection_loader_rejects_shape_dtype_and_value_tampering(
    tamper: str,
    message: str,
) -> None:
    root, cell_selection, feature_selection = _query_inputs()
    reference, _ = _mapping_reference()
    ref = _write(
        root,
        _plan(root, cell_selection, feature_selection, reference.external_ref),
    )
    group = artifact_group(root, ref)
    if tamper == "index_dtype":
        _replace_array(
            group,
            "indices",
            np.asarray(group["indices"][:], dtype=np.int64),
        )
    elif tamper == "distance_dtype":
        _replace_array(
            group,
            "distances",
            np.asarray(group["distances"][:], dtype=np.int64),
        )
    elif tamper == "distance_shape":
        _replace_array(
            group,
            "distances",
            np.ones((4, 1), dtype=np.float64),
        )
    elif tamper == "uninformative_dtype":
        _replace_array(
            group,
            "uninformative",
            np.asarray(group["uninformative"][:], dtype=np.uint8),
        )
    elif tamper == "uninformative_shape":
        _replace_array(
            group,
            "uninformative",
            np.asarray(group["uninformative"][:], dtype=bool)[:, np.newaxis],
        )
    elif tamper == "distance_nan":
        group["distances"][0, 0] = np.nan
    elif tamper == "distance_negative":
        group["distances"][0, 0] = -1.0
    elif tamper == "selection_count":
        artifact_group(root, cell_selection)["values"][0] = False
    elif tamper == "diagnostics_type":
        group.attrs["diagnostics"] = "invalid"
    else:
        group.attrs["created_at_ns"] = 0

    with pytest.raises(ValueError, match=message):
        load_projection(root, ref, reference=reference)


def test_projection_written_without_the_dispersion_diagnostic_is_rejected() -> None:
    root, cell_selection, feature_selection = _query_inputs()
    reference, _ = _mapping_reference()
    ref = _write(
        root,
        _plan(root, cell_selection, feature_selection, reference.external_ref),
    )
    group = artifact_group(root, ref)
    diagnostics = dict(group.attrs["diagnostics"])
    del diagnostics["queryScaledDispersion"]
    group.attrs["diagnostics"] = diagnostics

    # Projections predating the diagnostic are not silently migrated. They are
    # cheap to recompute, so the loader names the gap instead of guessing a value.
    with pytest.raises(ValueError, match="is missing queryScaledDispersion"):
        load_projection(root, ref)


def test_projection_loader_validates_and_matches_provided_reference() -> None:
    root, cell_selection, feature_selection = _query_inputs()
    reference, _ = _mapping_reference()
    ref = _write(
        root,
        _plan(root, cell_selection, feature_selection, reference.external_ref),
    )
    other_artifact, _ = _mapping_reference(
        fingerprint="reference-dataset",
        token="b",
    )
    other_fingerprint, _ = _mapping_reference(
        fingerprint="other-dataset",
        token="a",
    )

    for mismatched in (other_artifact, other_fingerprint):
        with pytest.raises(ValueError, match="run_mapping"):
            load_projection(root, ref, reference=mismatched)

    reference.datastore.zw["RNA"].attrs["dataset_fingerprint"] = "changed"
    with pytest.raises(ValueError, match="run_mapping"):
        load_projection(root, ref, reference=reference)


def test_projection_loader_rejects_invalid_call_and_artifact_handles() -> None:
    root, cell_selection, feature_selection = _query_inputs()
    reference, _ = _mapping_reference()
    ref = _write(
        root,
        _plan(root, cell_selection, feature_selection, reference.external_ref),
    )

    with pytest.raises(TypeError, match="load_arrays must be a boolean"):
        load_projection(root, ref, load_arrays=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="reference must be a MappingReference"):
        load_projection(root, ref, reference=object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="assay-scoped projection"):
        load_projection(root, cell_selection)
    with pytest.raises(ValueError, match="missing or incomplete"):
        load_projection(root, _artifact_ref("projection", "d"))


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        ("parameters", "parameters do not match"),
        ("policy", "missing_feature_policy is unsupported"),
        ("inputs", "inputs do not match"),
        ("fingerprint", "selected_expression_fingerprint"),
        ("external", "mapping_reference input is malformed"),
        ("external_kind", "identify a mapping_reference"),
        ("selection_scope", "wrong kind or scope"),
    ],
)
def test_projection_loader_rejects_malformed_provenance(
    tamper: str,
    message: str,
) -> None:
    root, cell_selection, feature_selection = _query_inputs()
    reference, _ = _mapping_reference()
    ref = _write(
        root,
        _plan(root, cell_selection, feature_selection, reference.external_ref),
    )
    group = artifact_group(root, ref)
    provenance = dict(group.attrs["provenance"])
    parameters = dict(provenance["parameters"])
    inputs = dict(provenance["inputs"])
    if tamper == "parameters":
        parameters["extra"] = True
    elif tamper == "policy":
        parameters["missing_feature_policy"] = "guess"
    elif tamper == "inputs":
        inputs.pop("query_batch_fingerprint")
    elif tamper == "fingerprint":
        inputs["selected_expression_fingerprint"] = ""
    elif tamper == "external":
        inputs["mapping_reference"] = "invalid"
    elif tamper == "external_kind":
        external = reference.external_ref.to_dict()
        external["ref"] = _artifact_ref("projection", "f").to_dict()
        inputs["mapping_reference"] = external
    else:
        inputs["cell_selection"] = _artifact_ref(
            "reduction",
            "f",
            assay=None,
        ).to_dict()
    provenance["parameters"] = parameters
    provenance["inputs"] = inputs
    group.attrs["provenance"] = provenance

    with pytest.raises(ValueError, match=message):
        load_projection(root, ref, reference=reference)


def test_projection_loader_metadata_only_rejects_out_of_range_neighbor() -> None:
    root, cell_selection, feature_selection = _query_inputs()
    reference, _ = _mapping_reference()
    ref = _write(
        root,
        _plan(root, cell_selection, feature_selection, reference.external_ref),
    )
    artifact_group(root, ref)["indices"][2, 1] = reference.selected_cell_count
    mismatched, _ = _mapping_reference(
        fingerprint="reference-dataset",
        token="b",
    )

    with pytest.raises(
        ValueError,
        match="does not match the projection input.*run_mapping",
    ):
        load_projection(root, ref, reference=mismatched)

    with pytest.raises(
        ValueError,
        match="outside the selected reference cell range.*run_mapping",
    ):
        load_projection(root, ref, reference=reference)


def test_projection_resolver_scopes_names_by_reference_and_uses_newest() -> None:
    root, cell_selection, feature_selection = _query_inputs()
    first_reference, _ = _mapping_reference(token="a")
    second_reference, _ = _mapping_reference(
        fingerprint="second-dataset",
        token="b",
    )
    first_old = _write(
        root,
        _plan(
            root,
            cell_selection,
            feature_selection,
            first_reference.external_ref,
            mapping_name="shared",
        ),
        created_at_ns=10,
    )
    first_new = _write(
        root,
        _plan(
            root,
            cell_selection,
            feature_selection,
            first_reference.external_ref,
            mapping_name="shared",
            invalidate_cache=True,
        ),
        created_at_ns=30,
    )
    second = _write(
        root,
        _plan(
            root,
            cell_selection,
            feature_selection,
            second_reference.external_ref,
            mapping_name="shared",
        ),
        created_at_ns=20,
    )

    assert first_old != first_new
    assert (
        resolve_projection(
            root,
            query_assay="RNA",
            mapping_name="shared",
            mapping_reference=first_reference.external_ref,
        )
        == first_new
    )
    assert (
        resolve_projection(
            root,
            query_assay="RNA",
            mapping_name="shared",
            mapping_reference=second_reference.external_ref,
        )
        == second
    )
    assert (
        len(
            list_artifacts(
                root,
                scope="assay",
                assay="RNA",
                kind="projection",
                complete_only=True,
            )
        )
        == 3
    )


def test_plan_projection_rejects_empty_string_arguments() -> None:
    root, cell_selection, feature_selection = _query_inputs()
    reference, _ = _mapping_reference()

    with pytest.raises(TypeError, match="query_assay must be a non-empty string"):
        plan_projection(
            root,
            query_assay=" ",
            mapping_name="atlas",
            n_cells=4,
            save_k=2,
            missing_feature_policy="reference_mean",
            correction_method="none",
            cell_selection=cell_selection,
            feature_selection=feature_selection,
            selected_expression_fingerprint="e" * 64,
            query_batch_fingerprint=NO_QUERY_BATCH_FINGERPRINT,
            mapping_reference=reference.external_ref,
            reference_cell_count=3,
        )
    with pytest.raises(TypeError, match="mapping_name must be a non-empty string"):
        plan_projection(
            root,
            query_assay="RNA",
            mapping_name="",
            n_cells=4,
            save_k=2,
            missing_feature_policy="reference_mean",
            correction_method="none",
            cell_selection=cell_selection,
            feature_selection=feature_selection,
            selected_expression_fingerprint="e" * 64,
            query_batch_fingerprint=NO_QUERY_BATCH_FINGERPRINT,
            mapping_reference=reference.external_ref,
            reference_cell_count=3,
        )
    with pytest.raises(ValueError, match="save_k must be positive"):
        plan_projection(
            root,
            query_assay="RNA",
            mapping_name="atlas",
            n_cells=4,
            save_k=0,
            missing_feature_policy="reference_mean",
            correction_method="none",
            cell_selection=cell_selection,
            feature_selection=feature_selection,
            selected_expression_fingerprint="e" * 64,
            query_batch_fingerprint=NO_QUERY_BATCH_FINGERPRINT,
            mapping_reference=reference.external_ref,
            reference_cell_count=3,
        )


def test_plan_projection_rejects_selection_reference_and_count_mismatches() -> None:
    root, cell_selection, feature_selection = _query_inputs()
    reference, _ = _mapping_reference()

    with pytest.raises(ValueError, match="selected row count"):
        _plan(
            root,
            cell_selection,
            feature_selection,
            reference.external_ref,
            n_cells=3,
        )
    with pytest.raises(ValueError, match="missing_feature_policy"):
        _plan(
            root,
            cell_selection,
            feature_selection,
            reference.external_ref,
            missing_feature_policy="guess",
        )
    with pytest.raises(
        ValueError, match="cell_selection.*wrong artifact kind or scope"
    ):
        _plan(
            root,
            feature_selection,
            feature_selection,
            reference.external_ref,
        )
    with pytest.raises(
        ValueError, match="feature_selection.*wrong artifact kind or scope"
    ):
        _plan(
            root,
            cell_selection,
            cell_selection,
            reference.external_ref,
        )
    with pytest.raises(TypeError, match="ExternalArtifactRef"):
        _plan(
            root,
            cell_selection,
            feature_selection,
            reference.ref,  # type: ignore[arg-type]
        )
    wrong_external = ExternalArtifactRef(
        dataset_fingerprint="reference-dataset",
        ref=_artifact_ref("projection", "f"),
    )
    with pytest.raises(ValueError, match="mapping_reference artifact"):
        _plan(
            root,
            cell_selection,
            feature_selection,
            wrong_external,
        )


def test_resolve_projection_rejects_empty_mapping_name() -> None:
    root, cell_selection, feature_selection = _query_inputs()
    reference, _ = _mapping_reference()
    _write(
        root,
        _plan(
            root,
            cell_selection,
            feature_selection,
            reference.external_ref,
        ),
    )
    with pytest.raises(ValueError, match="mapping_name must be a non-empty string"):
        resolve_projection(
            root,
            query_assay="RNA",
            mapping_name="",
            mapping_reference=reference.external_ref,
        )


def test_resolve_projection_rejects_missing_and_malformed_candidates() -> None:
    root, cell_selection, feature_selection = _query_inputs()
    reference, _ = _mapping_reference()
    with pytest.raises(ValueError, match="No complete projection"):
        resolve_projection(
            root,
            query_assay="RNA",
            mapping_name="missing",
            mapping_reference=reference.external_ref,
        )

    ref = _write(
        root,
        _plan(
            root,
            cell_selection,
            feature_selection,
            reference.external_ref,
            mapping_name="malformed",
        ),
    )
    group = artifact_group(root, ref)
    provenance = dict(group.attrs["provenance"])
    parameters = dict(provenance["parameters"])
    parameters["extra"] = True
    provenance["parameters"] = parameters
    group.attrs["provenance"] = provenance

    with pytest.raises(ValueError, match="malformed map_query parameters"):
        resolve_projection(
            root,
            query_assay="RNA",
            mapping_name="malformed",
            mapping_reference=reference.external_ref,
        )
