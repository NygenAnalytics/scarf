from dataclasses import replace
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest

import scarf.datastore._operations.mapping as mapping_operations
import scarf.mapping.projection as projection_storage
from scarf.datastore.datastore import DataStore
from scarf.mapping.confidence import conformal_prediction_sets, distance_weights
from scarf.mapping.projection import (
    NO_QUERY_BATCH_FINGERPRINT,
    ProjectionWriter,
    plan_projection,
)
from scarf.metadata.artifacts import (
    plan_cell_data_artifact,
    write_cell_data_artifact,
)
from scarf.storage.artifact_writer import (
    finish_artifact,
    plan_artifact,
    start_artifact,
)
from scarf.storage.artifacts import (
    ArtifactRef,
    fingerprint_array,
)
from scarf.storage.feature_selection import (
    _feature_selection_plan,
    _ordered_feature_ids_fingerprint,
    _write_feature_selection,
)
from scarf.storage.selections import (
    read_stored_selection_indices,
    resolve_selection_artifact,
)
from scarf.storage.types import as_zarr_array as checked_zarr_array


class _RecordingProjectionArray:
    def __init__(self, array, name: str, reads: list[tuple[str, object]]) -> None:
        self._array = array
        self._name = name
        self._reads = reads

    def __getattr__(self, name: str):
        return getattr(self._array, name)

    def __getitem__(self, key):
        if key == slice(None, None, None) or key is Ellipsis:
            raise AssertionError(f"{self._name} was materialized with a full slice")
        self._reads.append((self._name, key))
        return self._array[key]

    def __array__(self, dtype=None, copy=None):
        raise AssertionError(f"{self._name} was materialized as a complete array")


def _record_projection_reads(monkeypatch, reads: list[tuple[str, object]]) -> None:
    def recording_array(node, *, name: str = ""):
        array = checked_zarr_array(node, name=name)
        if name in {"indices", "distances", "uninformative"}:
            return _RecordingProjectionArray(array, name, reads)
        return array

    monkeypatch.setattr(mapping_operations, "as_zarr_array", recording_array)
    monkeypatch.setattr(projection_storage, "as_zarr_array", recording_array)


def _plain_reference(datastore):
    graphs = datastore.list_artifacts(
        kind="connectivity_map",
        from_assay="RNA",
        scope="assay",
        complete_only=True,
    )
    assert len(graphs) == 1
    neighbors = ArtifactRef.from_dict(
        datastore.inspect_artifact(graphs[0]).inputs["neighbors"]
    )
    reference_ref = datastore.build_mapping_reference(neighbors)
    return datastore.get_mapping_reference(reference_ref)


def _copied_query(datastore, path: Path, *, zarr_mode: str = "r+") -> DataStore:
    shutil.copytree(datastore.zarr_loc, path)
    return DataStore(
        str(path),
        default_assay="RNA",
        zarr_mode=zarr_mode,
    )


def _snapshot_store(path: str) -> dict[str, bytes]:
    root = Path(path)
    return {
        str(file.relative_to(root)): file.read_bytes()
        for file in root.rglob("*")
        if file.is_file()
    }


def _write_projection(
    query,
    reference,
    *,
    indices: np.ndarray,
    distances: np.ndarray,
    uninformative: np.ndarray,
    cell_key: str = "mapping_cells",
    feature_coverage: float = 1.0,
) -> ArtifactRef:
    index_values = np.asarray(indices, dtype=np.uint64)
    distance_values = np.asarray(distances, dtype=np.float64)
    uninformative_values = np.asarray(uninformative, dtype=bool)
    n_cells = len(index_values)
    if cell_key == "I":
        cell_mask = np.asarray(query.cells.fetch_all("I"), dtype=bool)
        assert int(cell_mask.sum()) == n_cells
    else:
        cell_mask = np.zeros(query.cells.N, dtype=bool)
        cell_mask[:n_cells] = True
        query.cells.insert(cell_key, cell_mask, overwrite=True)
    cell_selection = resolve_selection_artifact(
        query.zw,
        scope="datastore",
        kind="cell_selection",
        values=cell_mask,
        row_ids=np.asarray(query.cells.fetch_all("ids")),
        operation="manual_selection",
        parameters={},
        inputs={},
        source_column=cell_key,
    )
    all_features = query.select_all_features(from_assay="RNA")
    query_feature_ids = np.asarray(query.RNA.feats.fetch_all("ids")).astype(str)
    reference_feature_ids = np.asarray(reference.feature_ids).astype(str)
    feature_mask = np.isin(query_feature_ids, reference_feature_ids)
    feature_ids_fingerprint = _ordered_feature_ids_fingerprint(query.RNA)
    feature_plan = _feature_selection_plan(
        query.zw,
        assay="RNA",
        n_features=query.RNA.feats.N,
        ordered_feature_ids_fingerprint=feature_ids_fingerprint,
        operation="select_mapping_overlap",
        parameters={},
        inputs={
            "mapping_reference": reference.external_ref,
            "all_features": all_features,
        },
        execution_options={},
        expected_payload_fingerprint=fingerprint_array(feature_mask),
    )
    _write_feature_selection(
        query.zw,
        feature_plan,
        ordered_feature_ids_fingerprint=feature_ids_fingerprint,
        payload={"values": feature_mask},
    )
    feature_selection = feature_plan.ref
    planned = plan_projection(
        query.zw,
        query_assay="RNA",
        n_cells=n_cells,
        save_k=index_values.shape[1],
        missing_feature_policy="reference_mean",
        correction_method="none",
        cell_selection=cell_selection,
        feature_selection=feature_selection,
        selected_expression_fingerprint="e" * 64,
        query_batch_fingerprint=NO_QUERY_BATCH_FINGERPRINT,
        query_batch_count=1,
        mapping_reference=reference.external_ref,
        reference=reference,
        reference_cell_count=reference.selected_cell_count,
    )
    writer = ProjectionWriter(
        query.zw,
        planned,
        chunk_rows=max(1, min(n_cells, 2)),
    )
    writer.write_block(
        0,
        index_values,
        distance_values,
        uninformative_values,
    )
    ref = writer.finish(
        {
            "featureCoverage": float(feature_coverage),
            "queryBatchCount": 1,
            "algorithmVariant": "scaled_pca",
            "zeroNormCellCount": int(np.count_nonzero(uninformative_values)),
            "queryScaledDispersion": 1.0,
        }
    )
    return ref


def _write_reference_labels(reference, name: str = "reference_labels") -> np.ndarray:
    labels = np.full(reference.selected_cell_count, "other", dtype=object)
    labels[:2] = ["winner", "runner_up"]
    _write_reference_column(reference, name, labels)
    return labels


def _reference_cell_indices(reference) -> np.ndarray:
    return read_stored_selection_indices(
        reference.datastore.zw,
        reference.cell_selection,
        kind="cell_selection",
        scope="datastore",
        assay=None,
        table_path="cellData",
    )


def _write_reference_column(reference, name: str, values: np.ndarray) -> None:
    compact = np.asarray(values)
    indices = _reference_cell_indices(reference)
    if compact.ndim != 1 or len(compact) != len(indices):
        raise ValueError("Reference metadata must have one value per selected cell")
    full = np.empty(reference.datastore.cells.N, dtype=compact.dtype)
    if compact.dtype.kind in {"O", "S", "U"}:
        full[:] = ""
    else:
        full[:] = 0
    full[indices] = compact
    reference.datastore.cells.insert(name, full, overwrite=True)


def _attach_reference_missing_mask(
    reference,
    column: str,
    compact_missing: np.ndarray,
) -> None:
    indices = _reference_cell_indices(reference)
    missing = np.zeros(reference.datastore.cells.N, dtype=bool)
    missing[indices] = np.asarray(compact_missing, dtype=bool)
    group = reference.datastore.zw["cellData"]
    mask_name = f"__scarf_missing__{column}"
    group.create_array(mask_name, data=missing, chunks=(len(missing),))
    reference.datastore.cells._get_array(column).attrs["missing_mask"] = mask_name


def _write_reference_layout(
    reference,
    *,
    name: str,
) -> tuple[np.ndarray, ArtifactRef]:
    first = np.arange(reference.selected_cell_count, dtype=np.float64) * 10
    layout = np.column_stack((first, first + 10))
    planned = plan_cell_data_artifact(
        reference.datastore.zw,
        scope="assay",
        assay=reference.assay_name,
        kind="embedding",
        operation="manual_reference_embedding",
        parameters={"name": name},
        inputs={},
        execution_options={},
        cell_selection=reference.cell_selection,
        arrays={"values": (layout.shape, "f")},
    )
    write_cell_data_artifact(
        reference.datastore.zw,
        planned,
        {"values": layout},
    )
    return layout, planned.ref


@pytest.fixture
def mapping_consumer_context(analyzed_datastore_ephemeral, tmp_path):
    reference_store = analyzed_datastore_ephemeral
    reference = _plain_reference(reference_store)
    query = _copied_query(reference_store, tmp_path / "query.zarr")
    return reference_store, reference, query


def test_mapping_result_requires_explicit_ref_and_reference_after_cold_reopen(
    mapping_consumer_context,
):
    reference_store, reference, query = mapping_consumer_context
    result = _write_projection(
        query,
        reference,
        indices=np.array([[0, 1], [1, 0]]),
        distances=np.array([[1.0, 9.0], [9.0, 1.0]]),
        uninformative=np.array([False, False]),
    )

    reopened_reference_store = DataStore(
        reference_store.zarr_loc,
        default_assay="RNA",
        zarr_mode="r",
    )
    reopened_reference = reopened_reference_store.get_mapping_reference(reference.ref)
    reopened_query = DataStore(
        query.zarr_loc,
        default_assay="RNA",
        zarr_mode="r",
    )
    loaded = reopened_query.get_mapping_result(
        result,
        reference=reopened_reference,
        load_arrays=True,
    )

    assert loaded.ref == result
    assert loaded.reference is reopened_reference
    assert loaded.indices is not None
    with pytest.raises(TypeError, match="required keyword-only.*reference"):
        query.get_mapping_result(result)
    with pytest.raises(TypeError, match="result must be an ArtifactRef"):
        query.get_mapping_result(
            "atlas",
            reference=reference,
        )
    with pytest.raises(TypeError, match="result must be an ArtifactRef"):
        query.get_mapping_result(loaded, reference=reference)

    mismatched = replace(
        reference,
        ref=ArtifactRef(
            scope="assay",
            assay=reference.assay_name,
            kind="mapping_reference",
            artifact_id="0" * 64,
        ),
    )
    with pytest.raises(ValueError, match="Mapping reference artifact is missing"):
        query.get_mapping_result(result, reference=mismatched)


def test_mapping_scores_exclude_uninformative_rows_and_preserve_groups(
    mapping_consumer_context,
):
    _, reference, query = mapping_consumer_context
    result = _write_projection(
        query,
        reference,
        indices=np.array([[0, 1], [0, 1], [1, 2], [2, 3]]),
        distances=np.array([[1.0, 9.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]),
        uninformative=np.array([False, True, True, True]),
    )
    groups = np.array(["informative", "informative", "empty", "empty"])

    weighted = list(
        query.get_mapping_score(
            result,
            target_groups=groups,
            reference=reference,
            log_transform=False,
            multiplier=1.0,
        )
    )
    unweighted = list(
        query.get_mapping_score(
            result,
            target_groups=groups,
            reference=reference,
            log_transform=False,
            multiplier=1.0,
            weighted=False,
            fixed_weight=0.2,
        )
    )

    assert [group for group, _ in weighted] == ["informative", "empty"]
    assert [group for group, _ in unweighted] == ["informative", "empty"]
    assert weighted[0][1].shape == (reference.selected_cell_count,)
    # Published weight, 1 / (log(distance + 1) + 1), divided by one informative
    # query cell times two neighbors.
    expected = 1.0 / (np.log1p(np.array([1.0, 9.0])) + 1.0) / 2.0
    np.testing.assert_allclose(weighted[0][1][:2], expected)
    np.testing.assert_array_equal(weighted[1][1], 0.0)
    np.testing.assert_allclose(unweighted[0][1][:2], [0.1, 0.1])
    np.testing.assert_array_equal(unweighted[1][1], 0.0)

    with pytest.raises(ValueError, match="one value per projected query cell"):
        list(
            query.get_mapping_score(
                result,
                target_groups=np.array(["short"]),
                reference=reference,
            )
        )
    with pytest.raises(ValueError, match="fixed_weight"):
        list(query.get_mapping_score(result, reference=reference, fixed_weight=0.0))
    with pytest.raises(TypeError, match="weighted"):
        list(query.get_mapping_score(result, reference=reference, weighted=1))


def test_mapping_scores_keep_missing_target_groups_distinct(
    mapping_consumer_context,
):
    _, reference, query = mapping_consumer_context
    result = _write_projection(
        query,
        reference,
        indices=np.array([[0, 1], [2, 0], [1, 2]]),
        distances=np.ones((3, 2)),
        uninformative=np.zeros(3, dtype=bool),
    )
    groups = pd.Categorical(
        ["present", None, "present"],
        categories=["present", "unused"],
    )

    scores = list(
        query.get_mapping_score(
            result,
            target_groups=np.asarray(groups),
            reference=reference,
            log_transform=False,
            multiplier=1.0,
            weighted=False,
            fixed_weight=1.0,
        )
    )

    assert len(scores) == 2
    assert scores[0][0] == "present"
    assert pd.isna(scores[1][0])
    np.testing.assert_allclose(scores[0][1][:3], [0.25, 0.5, 0.25])
    np.testing.assert_allclose(scores[1][1][:3], [0.5, 0.0, 0.5])
    np.testing.assert_array_equal(scores[0][1][3:], 0.0)
    np.testing.assert_array_equal(scores[1][1][3:], 0.0)


def test_labels_and_evidence_abstain_without_fabricating_metrics(
    mapping_consumer_context,
):
    _, reference, query = mapping_consumer_context
    _write_reference_labels(reference)
    result = _write_projection(
        query,
        reference,
        indices=np.array([[0, 1], [0, 1], [1, 0]]),
        distances=np.array([[1.0, 9.0], [1.0, 9.0], [1.0, 9.0]]),
        uninformative=np.array([False, True, False]),
        feature_coverage=1.0,
    )

    labels = query.get_target_classes(
        result,
        reference_class_group="reference_labels",
        reference=reference,
        threshold_fraction=0.75,
    )
    evidence = query.get_target_label_evidence(
        result,
        reference=reference,
        reference_class_group="reference_labels",
        threshold_fraction=0.75,
        calibration_nonconformity=np.array([0.1, 0.2, 0.3]),
        conformal_alpha=0.2,
    )

    assert isinstance(labels, pd.Series)
    assert labels.tolist() == ["winner", "NA", "runner_up"]
    assert evidence["label"].tolist() == ["winner", "NA", "runner_up"]
    assert evidence["isUnknown"].tolist() == [False, True, False]
    assert evidence["featureCoverage"].tolist() == [1.0] * 3
    for column in (
        "voteFraction",
        "voteEntropy",
        "topTwoMargin",
        "referenceDistancePercentile",
    ):
        assert np.isnan(evidence.loc[1, column])
        assert np.isfinite(evidence.loc[[0, 2], column]).all()
    np.testing.assert_allclose(evidence.loc[[0, 2], "voteFraction"], [0.9, 0.9])
    assert evidence.loc[1, "predictionSet"] == ()
    assert "reference_labels" not in query.cells.columns


def test_label_transfer_handles_ties_thresholds_distance_and_subsets(
    mapping_consumer_context,
):
    _, reference, query = mapping_consumer_context
    _write_reference_labels(reference)
    result = _write_projection(
        query,
        reference,
        indices=np.array([[0, 1], [0, 1], [1, 0]]),
        distances=np.array([[1.0, 1.0], [0.0, 4.0], [1.0, 3.0]]),
        uninformative=np.zeros(3, dtype=bool),
    )

    labels = query.get_target_classes(
        result,
        reference_class_group="reference_labels",
        reference=reference,
        threshold_fraction=0.75,
    )
    subset = query.get_target_classes(
        result,
        reference_class_group="reference_labels",
        reference=reference,
        threshold_fraction=0.75,
        target_subset=[2],
    )
    empty = query.get_target_classes(
        result,
        reference_class_group="reference_labels",
        reference=reference,
        target_subset=[],
    )
    evidence = query.get_target_label_evidence(
        result,
        reference_class_group="reference_labels",
        reference=reference,
        threshold_fraction=0.75,
        max_distance=0.5,
    )

    assert labels.tolist() == ["NA", "winner", "runner_up"]
    assert subset.index.tolist() == [2]
    assert subset.tolist() == ["runner_up"]
    assert empty.empty
    assert evidence["label"].tolist() == ["NA", "winner", "NA"]
    assert evidence["isUnknown"].tolist() == [True, False, True]
    assert evidence.loc[0, "voteFraction"] == pytest.approx(0.5)
    assert evidence.loc[0, "voteEntropy"] == pytest.approx(np.log(2.0))
    assert evidence.loc[0, "topTwoMargin"] == pytest.approx(0.0)
    assert evidence.loc[2, "voteFraction"] == pytest.approx(0.75)
    assert np.isfinite(evidence.loc[2, "referenceDistancePercentile"])

    with pytest.raises(TypeError, match="target_subset must be a list"):
        query.get_target_classes(
            result,
            reference_class_group="reference_labels",
            reference=reference,
            target_subset=(0,),  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="entries must be integers"):
        query.get_target_classes(
            result,
            reference_class_group="reference_labels",
            reference=reference,
            target_subset=[True],
        )
    with pytest.raises(ValueError, match="out-of-range"):
        query.get_target_classes(
            result,
            reference_class_group="reference_labels",
            reference=reference,
            target_subset=[3],
        )
    with pytest.raises(TypeError, match="reference_class_group"):
        query.get_target_classes(
            result,
            reference_class_group="",
            reference=reference,
        )
    with pytest.raises(TypeError, match="na_val"):
        query.get_target_label_evidence(
            result,
            reference_class_group="reference_labels",
            reference=reference,
            na_val=None,  # type: ignore[arg-type]
        )


def test_label_transfer_excludes_explicitly_missing_reference_labels(
    mapping_consumer_context,
):
    _, reference, query = mapping_consumer_context
    labels = np.arange(reference.selected_cell_count, dtype=np.int64)
    _write_reference_column(reference, "nullable_labels", labels)
    missing = np.zeros(reference.selected_cell_count, dtype=bool)
    missing[0] = True
    _attach_reference_missing_mask(reference, "nullable_labels", missing)
    result = _write_projection(
        query,
        reference,
        indices=np.array([[0, 1], [0, 1]]),
        distances=np.array([[1.0, 9.0], [1.0, 9.0]]),
        uninformative=np.array([False, False]),
    )

    transferred = query.get_target_classes(
        result,
        reference_class_group="nullable_labels",
        reference=reference,
        threshold_fraction=0.5,
    )
    evidence = query.get_target_label_evidence(
        result,
        reference_class_group="nullable_labels",
        reference=reference,
        threshold_fraction=0.5,
        calibration_nonconformity=np.array([0.1, 0.2]),
    )

    assert transferred.tolist() == ["NA", "NA"]
    assert evidence["label"].tolist() == ["NA", "NA"]
    assert evidence["voteFraction"].max() < 0.5
    assert all(0 not in values for values in evidence["predictionSet"])


def test_label_transfer_excludes_nan_reference_labels(
    mapping_consumer_context,
):
    _, reference, query = mapping_consumer_context
    labels = np.ones(reference.selected_cell_count, dtype=np.float64)
    labels[:2] = np.nan
    _write_reference_column(reference, "sparse_labels", labels)
    result = _write_projection(
        query,
        reference,
        indices=np.array([[0, 1]]),
        distances=np.array([[1.0, 2.0]]),
        uninformative=np.array([False]),
    )

    evidence = query.get_target_label_evidence(
        result,
        reference_class_group="sparse_labels",
        reference=reference,
        calibration_nonconformity=np.array([0.1, 0.2]),
    )

    assert evidence.loc[0, "label"] == "NA"
    assert evidence.loc[0, "voteFraction"] == 0.0
    assert evidence.loc[0, "predictionSet"] == ()


def test_label_transfer_excludes_blank_byte_reference_labels(
    mapping_consumer_context,
):
    _, reference, query = mapping_consumer_context
    labels = np.full(reference.selected_cell_count, b"known", dtype="S8")
    labels[:2] = [b"", b"  "]
    _write_reference_column(reference, "byte_labels", labels)
    result = _write_projection(
        query,
        reference,
        indices=np.array([[0, 1]]),
        distances=np.array([[1.0, 2.0]]),
        uninformative=np.array([False]),
    )

    evidence = query.get_target_label_evidence(
        result,
        reference_class_group="byte_labels",
        reference=reference,
    )

    assert evidence.loc[0, "label"] == "NA"
    assert evidence.loc[0, "voteFraction"] == 0.0


def test_vote_entropy_is_conditional_on_available_labels() -> None:
    _, top_vote, entropy, _, unknown, _ = DataStore._label_vote_decision(
        np.array(["known", "missing"], dtype=object),
        np.array([0, 1]),
        np.array([0.01, 0.99]),
        0.5,
        "NA",
        reference_label_valid=np.array([True, False]),
    )

    assert top_vote == pytest.approx(0.01)
    assert entropy == pytest.approx(0.0)
    assert unknown


def test_label_transfer_calibration_is_deterministic_and_validated() -> None:
    calibrated = DataStore.calibrate_label_transfer_threshold(
        vote_fractions=np.array([0.2, 0.6, 0.8, 0.9]),
        correct=np.array([False, True, True, False]),
        target_coverage=0.5,
    )

    assert calibrated == {
        "voteThreshold": pytest.approx(0.7),
        "validationCoverage": pytest.approx(0.5),
        "validationAccuracy": pytest.approx(0.5),
    }

    with pytest.raises(ValueError, match="matching vectors"):
        DataStore.calibrate_label_transfer_threshold(
            np.ones((1, 2)),
            np.ones(2, dtype=bool),
        )
    for coverage in (0.0, 1.1):
        with pytest.raises(ValueError, match="target_coverage"):
            DataStore.calibrate_label_transfer_threshold(
                np.array([0.5]),
                np.array([True]),
                target_coverage=coverage,
            )
    with pytest.raises(ValueError, match="correct held-out prediction"):
        DataStore.calibrate_label_transfer_threshold(
            np.array([0.2, 0.8]),
            np.array([False, False]),
        )
    for invalid in (np.nan, np.inf, -0.1, 1.1):
        with pytest.raises(ValueError, match=r"finite values in \[0, 1\]"):
            DataStore.calibrate_label_transfer_threshold(
                np.array([0.5, invalid]),
                np.array([True, False]),
            )
    with pytest.raises(ValueError, match="real numeric"):
        DataStore.calibrate_label_transfer_threshold(
            np.array([True, False]),
            np.array([True, False]),
        )
    with pytest.raises(ValueError, match="boolean vector"):
        DataStore.calibrate_label_transfer_threshold(
            np.array([0.5, 0.6]),
            np.array([1, 0]),
        )
    for invalid_coverage in (True, np.nan):
        with pytest.raises(ValueError, match="target_coverage"):
            DataStore.calibrate_label_transfer_threshold(
                np.array([0.5]),
                np.array([True]),
                target_coverage=invalid_coverage,
            )


def test_confidence_helpers_reject_invalid_shapes_calibration_and_alpha() -> None:
    with pytest.raises(ValueError, match="two-dimensional distance"):
        distance_weights(np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="two-dimensional array"):
        conformal_prediction_sets(
            np.array([0.8, 0.2]),
            np.array([0.1]),
        )
    with pytest.raises(ValueError, match="non-empty vector"):
        conformal_prediction_sets(
            np.array([[0.8, 0.2]]),
            np.array([]),
        )
    with pytest.raises(ValueError, match="strictly between"):
        conformal_prediction_sets(
            np.array([[0.8, 0.2]]),
            np.array([0.1]),
            alpha=1.0,
        )
    with pytest.raises(ValueError, match="finite"):
        conformal_prediction_sets(
            np.array([[np.nan, 0.2]]),
            np.array([0.1]),
        )
    with pytest.raises(ValueError, match="scores must be in"):
        conformal_prediction_sets(
            np.array([[1.1, 0.2]]),
            np.array([0.1]),
        )
    with pytest.raises(ValueError, match="nonconformity must be in"):
        conformal_prediction_sets(
            np.array([[0.8, 0.2]]),
            np.array([-0.1]),
        )

    calibration = np.r_[np.zeros(7), np.ones(14)]
    exact_boundary = conformal_prediction_sets(
        np.array([[0.0]]),
        calibration,
        alpha=15 / 22,
    )
    assert not exact_boundary[0, 0]


def test_mapping_consumers_stream_projection_arrays(
    mapping_consumer_context,
    monkeypatch,
):
    _, reference, query = mapping_consumer_context
    _write_reference_labels(reference)
    result = _write_projection(
        query,
        reference,
        indices=np.array([[0, 1], [0, 1], [1, 0], [1, 0]]),
        distances=np.array([[1.0, 9.0], [2.0, 3.0], [1.0, 9.0], [4.0, 5.0]]),
        uninformative=np.array([False, True, False, False]),
    )
    reads: list[tuple[str, object]] = []
    _record_projection_reads(monkeypatch, reads)

    consumers = (
        lambda: list(query.get_mapping_score(result, reference=reference)),
        lambda: query.get_target_classes(
            result,
            reference_class_group="reference_labels",
            reference=reference,
        ),
        lambda: query.get_target_label_evidence(
            result,
            reference_class_group="reference_labels",
            reference=reference,
        ),
    )
    for consume in consumers:
        reads.clear()
        consume()
        assert {name for name, _ in reads} == {
            "indices",
            "distances",
            "uninformative",
        }
        for _, key in reads:
            assert isinstance(key, slice)
            assert key.start is not None
            assert key.stop is not None
            assert 0 < key.stop - key.start <= 2


def test_conformal_label_scores_are_allocated_one_row_at_a_time(
    mapping_consumer_context,
    monkeypatch,
):
    _, reference, query = mapping_consumer_context
    labels = _write_reference_labels(reference)
    result = _write_projection(
        query,
        reference,
        indices=np.array([[0, 1], [1, 0]]),
        distances=np.array([[1.0, 9.0], [1.0, 9.0]]),
        uninformative=np.array([False, False]),
    )
    allocations: list[object] = []

    class _NumpyProxy:
        def __getattr__(self, name: str):
            return getattr(np, name)

        def zeros(self, shape, *args, **kwargs):
            allocations.append(shape)
            return np.zeros(shape, *args, **kwargs)

    monkeypatch.setattr(mapping_operations, "np", _NumpyProxy())
    score_shape = (2, len(pd.unique(labels)))
    row_score_shape = len(pd.unique(labels))

    query.get_target_label_evidence(
        result,
        reference_class_group="reference_labels",
        reference=reference,
    )
    assert row_score_shape not in allocations

    allocations.clear()
    query.get_target_label_evidence(
        result,
        reference_class_group="reference_labels",
        reference=reference,
        calibration_nonconformity=np.array([0.1, 0.2]),
    )
    assert score_shape not in allocations
    assert allocations.count(row_score_shape) == 2


def test_reference_layout_requires_an_explicit_complete_embedding(
    mapping_consumer_context,
):
    _, reference, _ = mapping_consumer_context
    _, layout = _write_reference_layout(
        reference,
        name="explicit_layout",
    )
    with pytest.raises(TypeError, match="layout must be an ArtifactRef"):
        reference.fetch_layout("explicit_layout")
    wrong = ArtifactRef(
        scope="assay",
        assay=reference.assay_name,
        kind="cluster_labels",
        artifact_id=layout.artifact_id,
    )
    with pytest.raises(ValueError, match="embedding artifact"):
        reference.fetch_layout(wrong)


def test_reference_layout_reads_explicit_immutable_artifact(
    mapping_consumer_context,
):
    _, reference, _ = mapping_consumer_context
    expected, layout = _write_reference_layout(
        reference,
        name="immutable_layout",
    )
    reference.datastore.cells.insert(
        "unrelated_layout1",
        np.full(reference.datastore.cells.N, -999.0),
        overwrite=True,
    )

    np.testing.assert_array_equal(reference.fetch_layout(layout), expected)


def test_every_mapping_consumer_rejects_old_projection_artifacts(
    mapping_consumer_context,
):
    _, reference, query = mapping_consumer_context
    planned = plan_artifact(
        query.zw,
        scope="assay",
        assay="RNA",
        kind="projection",
        operation="map_with_reference",
        parameters={},
        inputs={},
        execution_options={},
    )
    group = start_artifact(query.zw, planned)
    finish_artifact(group, planned)
    old = planned.ref
    consumers = (
        lambda: query.get_mapping_result(old, reference=reference),
        lambda: list(query.get_mapping_score(old, reference=reference)),
        lambda: query.get_target_classes(
            old,
            reference_class_group="ids",
            reference=reference,
        ),
        lambda: query.get_target_label_evidence(
            old,
            reference_class_group="ids",
            reference=reference,
        ),
    )

    for consumer in consumers:
        with pytest.raises(ValueError, match="Re-run run_mapping"):
            consumer()
