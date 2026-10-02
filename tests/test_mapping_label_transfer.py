from collections import Counter
from dataclasses import replace
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest

import scarf.datastore._operations.mapping as mapping_operations
import scarf.mapping.models as mapping_models
import scarf.mapping.projection as projection_storage
from scarf.datastore.datastore import DataStore
from scarf.mapping.confidence import (
    _conformal_membership,
    _validated_conformal_calibration,
    distance_weights,
)
from scarf.mapping.models import LabelTransferResult
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
    ExternalArtifactRef,
    fingerprint_array,
)
from scarf.storage.feature_selection import (
    _feature_selection_plan,
    _ordered_feature_ids_fingerprint,
    _write_feature_selection,
)
from scarf.storage.selections import (
    read_stored_selection_indices,
    resolve_generated_selection_artifact,
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
    cell_selection = resolve_generated_selection_artifact(
        query.zw,
        scope="datastore",
        kind="cell_selection",
        values=cell_mask,
        row_ids=np.asarray(query.cells.fetch_all("ids")),
        operation="manual_selection",
        parameters={},
        inputs={},
        source_column=cell_key,
    )[0]
    all_features = query.select_all_features(from_assay="RNA")
    query_feature_ids = np.asarray(query.RNA.feats.fetch_all("ids")).astype(str)
    reference_feature_ids = np.asarray(reference.feature_ids).astype(str)
    feature_mask = np.isin(query_feature_ids, reference_feature_ids)
    feature_ids_fingerprint = _ordered_feature_ids_fingerprint(query.RNA.z)
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
        query_dataset_fingerprint=query._ensure_dataset_fingerprint("RNA"),
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
            "uninformativeCellCount": int(np.count_nonzero(uninformative_values)),
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


@pytest.mark.parametrize("weighted", [True, False])
def test_mapping_score_data_reads_projection_once(
    mapping_consumer_context,
    monkeypatch,
    weighted,
):
    _, reference, query = mapping_consumer_context
    indices = np.array([[0, 1], [2, 3], [1, 2], [3, 0], [0, 2], [1, 3], [2, 1]])
    distances = np.arange(1, 15, dtype=np.float64).reshape(7, 2)
    uninformative = np.array([False, False, True, False, False, False, False])
    result = _write_projection(
        query,
        reference,
        indices=indices,
        distances=distances,
        uninformative=uninformative,
    )
    # None and NaN are one missing group in both the generator and the table.
    groups = np.array(["b", "a", "a", np.nan, "c", "b", None], dtype=object)
    options = {
        "target_groups": groups,
        "log_transform": True,
        "multiplier": 10.0,
        "weighted": weighted,
        "fixed_weight": 0.3,
    }
    expected = list(query.get_mapping_score(result, reference=reference, **options))
    reads: list[tuple[str, object]] = []
    _record_projection_reads(monkeypatch, reads)

    loaded, scores, classes, coordinates = query._mapping_score_data(
        result,
        reference=reference,
        **options,
    )

    assert loaded.ref == result
    assert classes is None and coordinates is None
    assert [str(group) for group, _ in scores] == ["b", "a", "nan", "c"]
    for (group, values), (expected_group, expected_values) in zip(
        scores, expected, strict=True
    ):
        assert str(group) == str(expected_group)
        np.testing.assert_array_equal(values, expected_values)
    rows_read = Counter()
    for name, key in reads:
        assert isinstance(key, slice)
        rows_read[name] += key.stop - key.start
    # Loading validates each payload array once, then one pass scores all four
    # groups. Unweighted scores never read distances.
    n_cells = len(indices)
    assert rows_read["indices"] == 2 * n_cells
    assert rows_read["uninformative"] == 2 * n_cells
    assert rows_read["distances"] == (2 if weighted else 1) * n_cells


def _transfer(query, result, reference, labels="reference_labels", **options):
    ref = query.run_label_transfer(
        result,
        reference=reference,
        reference_labels=labels,
        **options,
    )
    return ref, query.get_label_transfer(ref, load_votes=True)


def test_label_transfer_saves_labels_frozen_inputs_and_evidence(
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

    transfer, loaded = _transfer(query, result, reference, threshold_fraction=0.75)

    assert transfer.kind == "label_transfer" and transfer.assay == "RNA"
    assert isinstance(loaded, LabelTransferResult)
    assert loaded.ref == transfer
    assert loaded.projection == result
    assert loaded.threshold_fraction == 0.75
    assert loaded.max_distance is None
    assert loaded.reference_label_source == "reference_labels"
    assert loaded.labels.tolist() == ["winner", None, "runner_up"]
    evidence = loaded.evidence
    assert evidence["candidateLabel"].tolist() == ["winner", None, "runner_up"]
    assert evidence["abstained"].tolist() == [False, True, False]
    assert evidence["abstentionReason"].tolist() == [None, "uninformative_cell", None]
    for column in (
        "voteFraction",
        "voteEntropy",
        "topTwoMargin",
        "nearestDistance",
        "referenceDistancePercentile",
    ):
        assert np.isnan(evidence.loc[1, column])
        assert np.isfinite(evidence.loc[[0, 2], column]).all()
    np.testing.assert_allclose(evidence.loc[[0, 2], "voteFraction"], [0.9, 0.9])
    np.testing.assert_allclose(evidence.loc[[0, 2], "nearestDistance"], [1.0, 1.0])
    np.testing.assert_array_equal(loaded.cell_idx, [0, 1, 2])
    assert not loaded.cell_idx.flags.writeable
    assert loaded.prediction_sets(np.array([0.1, 0.2, 0.3]), alpha=0.2)[1] == ()
    assert "reference_labels" not in query.cells.columns

    status = query.inspect_artifact(transfer)
    assert status.operation == "transfer_labels"
    assert status.parameters == {
        "threshold_fraction": 0.75,
        "max_distance": None,
        "algorithm_version": 1,
    }
    assert ArtifactRef.from_dict(status.inputs["projection"]) == result
    frozen = query.inspect_artifact(loaded.reference_labels)
    assert frozen.operation == "freeze_reference_labels"
    assert frozen.parameters == {"source_column": "reference_labels"}
    assert set(frozen.inputs) == {"mapping_reference", "labels_fingerprint"}
    assert frozen.inputs["mapping_reference"] == reference.external_ref.to_dict()
    np.testing.assert_array_equal(loaded.categories, ["winner", "runner_up", "other"])


def test_label_transfer_handles_ties_thresholds_distance_and_arguments(
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

    _, thresholded = _transfer(query, result, reference, threshold_fraction=0.75)
    _, bounded = _transfer(
        query,
        result,
        reference,
        threshold_fraction=0.75,
        max_distance=0.5,
    )

    assert thresholded.labels.tolist() == [None, "winner", "runner_up"]
    assert thresholded.evidence["abstentionReason"].tolist() == [
        "tied_vote",
        None,
        None,
    ]
    assert bounded.labels.tolist() == [None, "winner", None]
    assert bounded.evidence["abstentionReason"].tolist() == [
        "tied_vote",
        None,
        "beyond_max_distance",
    ]
    assert bounded.evidence["candidateLabel"].tolist() == [None, "winner", "runner_up"]
    assert bounded.max_distance == 0.5
    evidence = bounded.evidence
    assert evidence.loc[0, "voteFraction"] == pytest.approx(0.5)
    assert evidence.loc[0, "voteEntropy"] == pytest.approx(np.log(2.0))
    assert evidence.loc[0, "topTwoMargin"] == pytest.approx(0.0)
    assert evidence.loc[2, "voteFraction"] == pytest.approx(0.75)
    assert np.isfinite(evidence.loc[2, "referenceDistancePercentile"])
    assert thresholded.ref != bounded.ref
    assert thresholded.reference_labels == bounded.reference_labels

    def run(**options):
        arguments = {
            "reference": reference,
            "reference_labels": "reference_labels",
            **options,
        }
        return query.run_label_transfer(
            arguments.pop("projection", result), **arguments
        )

    with pytest.raises(TypeError, match="projection must be an ArtifactRef"):
        run(projection="projection")
    with pytest.raises(TypeError, match="reference must be a MappingReference"):
        run(reference=object())
    for labels in ("", 3, None):
        with pytest.raises(TypeError, match="reference_labels must be"):
            run(reference_labels=labels)
    for threshold in (1.5, -0.1, True, np.nan):
        with pytest.raises(ValueError, match="threshold_fraction"):
            run(threshold_fraction=threshold)
    for distance in (-1.0, np.inf, False):
        with pytest.raises(ValueError, match="max_distance"):
            run(max_distance=distance)
    with pytest.raises(TypeError, match="invalidate_cache"):
        run(invalidate_cache=1)
    layout = _write_reference_layout(reference, name="not_labels")[1]
    with pytest.raises(ValueError, match="must hold cell labels"):
        run(reference_labels=layout)
    with pytest.raises(TypeError, match="transfer must be an ArtifactRef"):
        query.get_label_transfer("transfer")
    with pytest.raises(ValueError, match="label_transfer artifact"):
        query.get_label_transfer(result)


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

    _, loaded = _transfer(
        query, result, reference, "nullable_labels", threshold_fraction=0.5
    )
    _, permissive = _transfer(
        query, result, reference, "nullable_labels", threshold_fraction=0.05
    )

    assert loaded.labels.tolist() == [None, None]
    assert loaded.evidence["abstentionReason"].tolist() == ["below_threshold"] * 2
    assert loaded.evidence["voteFraction"].max() < 0.5
    assert 0 not in loaded.categories.tolist()
    assert loaded.categories.dtype == np.int64
    assert all(
        0 not in values for values in loaded.prediction_sets(np.array([0.1, 0.2]))
    )
    # Integer reference labels stay integers.
    assert permissive.labels.tolist() == [1, 1]
    np.testing.assert_allclose(permissive.label_vote_shares([1, 1]), [0.1, 0.1])
    with pytest.raises(ValueError, match="only as text"):
        permissive.label_vote_shares(["1", "1"])


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

    _, loaded = _transfer(query, result, reference, "sparse_labels")

    assert loaded.labels.tolist() == [None]
    assert loaded.evidence.loc[0, "abstentionReason"] == "no_labeled_neighbors"
    assert loaded.evidence.loc[0, "voteFraction"] == 0.0
    assert loaded.prediction_sets(np.array([0.1, 0.2]))[0] == ()


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

    _, loaded = _transfer(query, result, reference, "byte_labels")

    assert loaded.labels.tolist() == [None]
    assert loaded.evidence.loc[0, "voteFraction"] == 0.0
    np.testing.assert_array_equal(loaded.categories, ["known"])


def test_label_transfer_keeps_a_real_na_class_apart_from_abstention(
    mapping_consumer_context,
):
    _, reference, query = mapping_consumer_context
    labels = np.full(reference.selected_cell_count, "other", dtype=object)
    labels[:2] = ["NA", "T"]
    _write_reference_column(reference, "labels_with_na", labels)
    result = _write_projection(
        query,
        reference,
        indices=np.array([[0, 1], [0, 1], [1, 0]]),
        distances=np.array([[1.0, 9.0], [1.0, 1.0], [1.0, 9.0]]),
        uninformative=np.zeros(3, dtype=bool),
    )

    _, loaded = _transfer(query, result, reference, "labels_with_na")

    assert loaded.labels.tolist() == ["NA", None, "T"]
    assert loaded.evidence["abstained"].tolist() == [False, True, False]


def test_saved_label_transfer_survives_reference_reannotation(
    mapping_consumer_context,
    monkeypatch,
):
    reference_store, reference, query = mapping_consumer_context
    original = _write_reference_labels(reference)
    result = _write_projection(
        query,
        reference,
        indices=np.array([[0, 1], [1, 0]]),
        distances=np.array([[1.0, 9.0], [1.0, 9.0]]),
        uninformative=np.array([False, False]),
    )
    first, before = _transfer(query, result, reference)

    revised = original.copy()
    revised[:2] = ["curated_winner", "curated_runner_up"]
    _write_reference_column(reference, "reference_labels", revised)
    second, after = _transfer(query, result, reference)

    assert second != first
    assert after.reference_labels != before.reference_labels
    assert after.labels.tolist() == ["curated_winner", "curated_runner_up"]
    reloaded = query.get_label_transfer(first)
    pd.testing.assert_frame_equal(reloaded.evidence, before.evidence)
    assert reloaded.labels.tolist() == ["winner", "runner_up"]

    # Restoring the labels reuses the first frozen copy and transfer.
    _write_reference_column(reference, "reference_labels", original)
    assert _transfer(query, result, reference)[0] == first

    # Loading a saved transfer reads nothing from the reference datastore.
    reference_backing_store = reference_store.zw.store
    keys: list[str] = []
    store_type = type(reference_backing_store)
    original_get = store_type.get

    async def recording_get(store, key, prototype, byte_range=None):
        if store is reference_backing_store:
            keys.append(key)
        return await original_get(store, key, prototype, byte_range)

    monkeypatch.setattr(store_type, "get", recording_get)
    reopened = DataStore(query.zarr_loc, default_assay="RNA", zarr_mode="r")
    keys.clear()
    assert reopened.get_label_transfer(first).labels.tolist() == ["winner", "runner_up"]
    assert keys == []


def test_label_transfer_reuses_matches_and_refuses_read_only_writes(
    mapping_consumer_context,
):
    _, reference, query = mapping_consumer_context
    _write_reference_labels(reference)
    result = _write_projection(
        query,
        reference,
        indices=np.array([[0, 1], [1, 0]]),
        distances=np.array([[1.0, 9.0], [1.0, 9.0]]),
        uninformative=np.array([False, False]),
    )
    first, first_result = _transfer(query, result, reference)
    assert _transfer(query, result, reference)[0] == first
    refreshed, refreshed_result = _transfer(
        query, result, reference, invalidate_cache=True
    )
    assert refreshed != first
    # The frozen labels are an exact copy, so a forced recompute reuses them.
    assert refreshed_result.reference_labels == first_result.reference_labels
    assert len(query.list_artifacts(kind="reference_labels", scope="datastore")) == 1

    read_only = DataStore(query.zarr_loc, default_assay="RNA", zarr_mode="r")
    assert read_only.run_label_transfer(
        result,
        reference=reference,
        reference_labels="reference_labels",
    ) in {first, refreshed}
    before = _snapshot_store(query.zarr_loc)
    with pytest.raises(PermissionError, match="run_label_transfer requires"):
        read_only.run_label_transfer(
            result,
            reference=reference,
            reference_labels="reference_labels",
            threshold_fraction=0.9,
        )
    assert _snapshot_store(query.zarr_loc) == before


def _write_reference_label_artifact(
    reference,
    labels: np.ndarray,
    *,
    kind: str,
    scope: str,
) -> ArtifactRef:
    planned = plan_cell_data_artifact(
        reference.datastore.zw,
        scope=scope,
        assay=reference.assay_name if scope == "assay" else None,
        kind=kind,
        operation="curate_reference_labels",
        parameters={"kind": kind},
        inputs={},
        execution_options={},
        cell_selection=reference.cell_selection,
        arrays={"values": (labels.shape, None)},
    )
    write_cell_data_artifact(reference.datastore.zw, planned, {"values": labels})
    return planned.ref


@pytest.mark.parametrize(
    ("kind", "scope", "anchor_assay"),
    [("cluster_labels", "assay", None), ("smart_label", "datastore", "RNA")],
)
def test_label_transfer_reads_reference_label_artifacts(
    mapping_consumer_context,
    kind,
    scope,
    anchor_assay,
):
    reference_store, reference, query = mapping_consumer_context
    labels = np.full(reference.selected_cell_count, "other", dtype=object)
    labels[:2] = ["winner", "runner_up"]
    source = _write_reference_label_artifact(
        reference,
        labels.astype(str),
        kind=kind,
        scope=scope,
    )
    result = _write_projection(
        query,
        reference,
        indices=np.array([[0, 1], [1, 0]]),
        distances=np.array([[1.0, 9.0], [1.0, 9.0]]),
        uninformative=np.array([False, False]),
    )

    transfer, loaded = _transfer(query, result, reference, source)

    assert loaded.labels.tolist() == ["winner", "runner_up"]
    expected_source = ExternalArtifactRef(
        dataset_fingerprint=reference.dataset_fingerprint,
        ref=source,
        anchor_assay=anchor_assay,
    )
    assert loaded.reference_label_source == expected_source
    frozen = query.inspect_artifact(loaded.reference_labels)
    assert frozen.parameters == {}
    assert frozen.inputs["source_labels"] == expected_source.to_dict()

    lineage = query.lineage(transfer, references=reference)
    status = lineage.graph.nodes[expected_source]["status"]
    assert status.complete and status.operation == "curate_reference_labels"
    assert lineage.graph.has_edge(expected_source, loaded.reference_labels)
    assert "unresolved external" not in lineage.to_markdown()


def test_label_transfer_is_a_cell_label_artifact(mapping_consumer_context):
    from scarf.metadata.selection import resolve_complete_labels, resolve_grouping

    _, reference, query = mapping_consumer_context
    _write_reference_labels(reference)
    result = _write_projection(
        query,
        reference,
        indices=np.array([[0, 1], [0, 1], [1, 0]]),
        distances=np.array([[1.0, 9.0], [1.0, 9.0], [1.0, 9.0]]),
        uninformative=np.array([False, True, False]),
    )
    transfer, loaded = _transfer(query, result, reference)

    grouping = resolve_grouping(query.zw, query.cells, transfer)
    np.testing.assert_array_equal(grouping.cell_idx, loaded.cell_idx)
    assert grouping.missing_mask is not None
    assert grouping.missing_mask.tolist() == [False, True, False]
    assert grouping.labels[[0, 2]].tolist() == ["winner", "runner_up"]
    with pytest.raises(ValueError, match="contains missing labels"):
        resolve_complete_labels(query.zw, transfer, name="transfer")

    winners = query.select_cells(transfer, include=["winner", "runner_up"])
    selected = read_stored_selection_indices(
        query.zw,
        winners,
        kind="cell_selection",
        scope="datastore",
        assay=None,
        table_path="cellData",
    )
    # The abstained cell is never selected.
    np.testing.assert_array_equal(selected, loaded.cell_idx[[0, 2]])


def test_label_transfer_rejects_a_changed_payload(mapping_consumer_context):
    from scarf.storage.artifacts import artifact_group

    _, reference, query = mapping_consumer_context
    _write_reference_labels(reference)
    result = _write_projection(
        query,
        reference,
        indices=np.array([[0, 1], [1, 0]]),
        distances=np.array([[1.0, 9.0], [1.0, 9.0]]),
        uninformative=np.array([False, False]),
    )
    transfer, _ = _transfer(query, result, reference)
    artifact_group(query.zw, transfer)["vote_fraction"][0] = 0.1

    with pytest.raises(ValueError, match="Re-run run_label_transfer"):
        query.get_label_transfer(transfer)
    # A changed payload is never reused.
    assert _transfer(query, result, reference)[0] != transfer


def test_label_vote_shares_calibrate_prediction_sets(mapping_consumer_context):
    _, reference, query = mapping_consumer_context
    _write_reference_labels(reference)
    result = _write_projection(
        query,
        reference,
        indices=np.array([[0, 1], [0, 1], [1, 0]]),
        distances=np.array([[1.0, 9.0], [1.0, 9.0], [1.0, 9.0]]),
        uninformative=np.array([False, True, False]),
    )
    _, loaded = _transfer(query, result, reference)

    shares = loaded.label_vote_shares(["winner", "winner", "absent"])
    np.testing.assert_allclose(shares[[0, 2]], [0.9, 0.0])
    assert np.isnan(shares[1])
    assert np.isnan(loaded.label_vote_shares([None, "winner", "winner"])[0])
    with pytest.raises(ValueError, match="one value per"):
        loaded.label_vote_shares(["winner"])

    sets = loaded.prediction_sets(1.0 - shares[[0, 2]], alpha=0.4)
    assert sets.name == "predictionSet"
    assert "winner" in sets[0]
    assert sets[1] == ()


def test_label_transfer_reads_its_payload_once(mapping_consumer_context, monkeypatch):
    _, reference, query = mapping_consumer_context
    _write_reference_labels(reference)
    result = _write_projection(
        query,
        reference,
        indices=np.array([[0, 1], [0, 1], [1, 0]]),
        distances=np.array([[1.0, 9.0], [1.0, 9.0], [1.0, 9.0]]),
        uninformative=np.array([False, True, False]),
    )
    query_backing_store = query.zw.store
    keys: list[str] = []
    store_type = type(query_backing_store)
    original_get = store_type.get

    async def recording_get(store, key, prototype, byte_range=None):
        if store is query_backing_store:
            keys.append(key)
        return await original_get(store, key, prototype, byte_range)

    monkeypatch.setattr(store_type, "get", recording_get)
    metadata = ("zarr.json", ".zarray", ".zattrs", ".zgroup")

    def payload_reads() -> Counter:
        return Counter(
            key
            for key in keys
            if "/artifacts/label_transfer/" in key and not key.endswith(metadata)
        )

    transfer = query.run_label_transfer(
        result,
        reference=reference,
        reference_labels="reference_labels",
    )
    written = payload_reads()
    # Blocks are digested as they are written, so only the completion check
    # reads the payload back, once.
    assert written and set(written.values()) == {1}
    vote_chunks = {
        key
        for key in written
        if "/vote_class_codes/" in key or "/vote_class_fractions/" in key
    }
    assert vote_chunks

    # A default load reads every chunk except the votes, once each.
    keys.clear()
    query.get_label_transfer(transfer)
    loaded = payload_reads()
    assert set(loaded) == set(written) - vote_chunks
    assert set(loaded.values()) == {1}

    keys.clear()
    query.get_label_transfer(transfer, load_votes=True)
    with_votes = payload_reads()
    assert set(with_votes) == set(written)
    assert set(with_votes.values()) == {1}


def test_label_transfer_loads_votes_only_on_request_without_copies(
    mapping_consumer_context,
    monkeypatch,
):
    _, reference, query = mapping_consumer_context
    _write_reference_labels(reference)
    result = _write_projection(
        query,
        reference,
        indices=np.array([[0, 1], [1, 0]]),
        distances=np.array([[1.0, 9.0], [1.0, 9.0]]),
        uninformative=np.array([False, False]),
    )
    transfer = query.run_label_transfer(
        result,
        reference=reference,
        reference_labels="reference_labels",
    )

    without_votes = query.get_label_transfer(transfer)
    assert without_votes.vote_class_codes is None
    assert without_votes.vote_class_fractions is None
    assert without_votes.evidence["candidateLabel"].tolist() == ["winner", "runner_up"]
    for use_votes in (
        lambda: without_votes.prediction_sets(np.array([0.1, 0.2])),
        lambda: without_votes.label_vote_shares(["winner", "winner"]),
    ):
        with pytest.raises(ValueError, match="load_votes=True"):
            use_votes()
    with pytest.raises(TypeError, match="load_votes must be a boolean"):
        query.get_label_transfer(transfer, load_votes=1)

    # The loader freezes the arrays it reads, so the result keeps them as
    # read-only views instead of copying them.
    def no_copy(*_args, **_kwargs):
        raise AssertionError("a loaded array was copied")

    monkeypatch.setattr(mapping_models, "read_only_copy", no_copy)
    with_votes = query.get_label_transfer(transfer, load_votes=True)
    for array in (
        with_votes.vote_class_codes,
        with_votes.vote_class_fractions,
        with_votes.cell_idx,
        with_votes.categories,
    ):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.setflags(write=True)
    pd.testing.assert_frame_equal(with_votes.evidence, without_votes.evidence)


def test_label_transfer_result_copies_arrays_that_callers_can_change(
    mapping_consumer_context,
):
    from dataclasses import replace as replace_fields

    _, reference, query = mapping_consumer_context
    _write_reference_labels(reference)
    result = _write_projection(
        query,
        reference,
        indices=np.array([[0, 1], [1, 0]]),
        distances=np.array([[1.0, 9.0], [1.0, 9.0]]),
        uninformative=np.array([False, False]),
    )
    _, loaded = _transfer(query, result, reference)
    codes = np.array(loaded.vote_class_codes)
    fractions = np.array(loaded.vote_class_fractions)

    rebuilt = replace_fields(
        loaded,
        vote_class_codes=codes,
        vote_class_fractions=fractions,
    )
    codes[:] = -1

    assert not np.shares_memory(rebuilt.vote_class_codes, codes)
    assert (rebuilt.vote_class_codes == loaded.vote_class_codes).all()
    with pytest.raises(ValueError, match="loaded together"):
        replace_fields(loaded, vote_class_codes=None)


def test_label_transfer_writes_with_the_datastore_storage_profile(
    mapping_consumer_context,
    monkeypatch,
):
    import scarf.mapping.label_transfer as label_transfer

    _, reference, query = mapping_consumer_context
    _write_reference_labels(reference)
    result = _write_projection(
        query,
        reference,
        indices=np.array([[0, 1], [1, 0]]),
        distances=np.array([[1.0, 9.0], [1.0, 9.0]]),
        uninformative=np.array([False, False]),
    )
    assert query.storageProfile is not None
    profiles: list[object] = []
    for name in ("create_metadata_column", "create_zarr_dataset"):
        original = getattr(label_transfer, name)

        def recording(*args, _original=original, **kwargs):
            profiles.append(kwargs.get("profile"))
            return _original(*args, **kwargs)

        monkeypatch.setattr(label_transfer, name, recording)

    query.run_label_transfer(
        result,
        reference=reference,
        reference_labels="reference_labels",
    )

    # Frozen reference labels and the transfer share the datastore's profile.
    assert len(profiles) > 2
    assert all(profile is query.storageProfile for profile in profiles)


def test_unsigned_reference_labels_keep_their_values() -> None:
    from scarf.mapping.label_transfer import encode_reference_labels

    values = np.array([2**64 - 1, 7, 2**64 - 1], dtype=np.uint64)
    categories, codes = encode_reference_labels(
        values,
        np.array([True, True, False]),
    )

    assert categories.dtype == np.uint64
    assert categories.tolist() == [2**64 - 1, 7]
    assert codes.tolist() == [0, 1, -1]


def test_vote_entropy_is_conditional_on_available_labels() -> None:
    from scarf.mapping.confidence import _label_vote_block

    votes = _label_vote_block(
        np.array([[0, -1]]),
        np.array([[0.01, 0.99]]),
    )

    assert votes.vote_fraction[0] == pytest.approx(0.01)
    assert votes.vote_entropy[0] == pytest.approx(0.0)
    assert votes.has_labeled_votes[0]
    assert not votes.is_tied[0]


def test_confidence_helpers_reject_invalid_shapes_calibration_and_alpha() -> None:
    with pytest.raises(ValueError, match="two-dimensional distance"):
        distance_weights(np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="non-empty vector"):
        _validated_conformal_calibration(np.array([]), 0.1)
    with pytest.raises(ValueError, match="strictly between"):
        _validated_conformal_calibration(np.array([0.1]), 1.0)
    with pytest.raises(ValueError, match="finite"):
        _validated_conformal_calibration(np.array([np.nan]), 0.1)
    with pytest.raises(ValueError, match="nonconformity must be in"):
        _validated_conformal_calibration(np.array([-0.1]), 0.1)

    calibration, alpha = _validated_conformal_calibration(
        np.r_[np.zeros(7), np.ones(14)],
        15 / 22,
    )
    exact_boundary = _conformal_membership(np.array([[0.0]]), calibration, alpha)
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
        lambda: query.run_label_transfer(
            result,
            reference=reference,
            reference_labels="reference_labels",
            invalidate_cache=True,
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

    _, loaded = _transfer(query, result, reference)
    monkeypatch.setattr(mapping_models, "np", _NumpyProxy())
    score_shape = (2, len(pd.unique(labels)))
    row_score_shape = len(pd.unique(labels))

    loaded.prediction_sets(np.array([0.1, 0.2]))
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
        reference._fetch_layout("explicit_layout")
    wrong = ArtifactRef(
        scope="assay",
        assay=reference.assay_name,
        kind="cluster_labels",
        artifact_id=layout.artifact_id,
    )
    with pytest.raises(ValueError, match="embedding artifact"):
        reference._fetch_layout(wrong)


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

    np.testing.assert_array_equal(reference._fetch_layout(layout), expected)


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
        lambda: query.run_label_transfer(
            old,
            reference=reference,
            reference_labels="ids",
        ),
    )

    for consumer in consumers:
        with pytest.raises(ValueError, match="Re-run run_mapping"):
            consumer()


def test_bound_reference_repeats_label_transfer_without_rereading_its_payload(
    mapping_consumer_context,
    monkeypatch,
):

    from scarf.storage.artifacts import artifact_group

    reference_store, reference, query = mapping_consumer_context
    _write_reference_labels(reference)
    result = _write_projection(
        query,
        reference,
        indices=np.array([[0, 1], [1, 0]]),
        distances=np.array([[1.0, 9.0], [9.0, 1.0]]),
        uninformative=np.array([False, False]),
    )
    first = query.run_label_transfer(
        result,
        reference=reference,
        reference_labels="reference_labels",
        threshold_fraction=0.5,
    )

    payload_paths = {
        artifact_group(reference_store.zw, ref).path
        for ref in (
            reference.ref,
            reference.reduction,
            reference.ann_index,
            reference.neighbors,
        )
    }
    reference_backing_store = reference_store.zw.store
    keys: list[str] = []
    store_type = type(reference_backing_store)
    original_get = store_type.get

    async def recording_get(store, key, prototype, byte_range=None):
        if store is reference_backing_store:
            keys.append(key)
        return await original_get(store, key, prototype, byte_range)

    monkeypatch.setattr(store_type, "get", recording_get)
    second = query.run_label_transfer(
        result,
        reference=reference,
        reference_labels="reference_labels",
        threshold_fraction=0.5,
    )
    assert second == first
    chunk_keys = [key for key in keys if not key.endswith("zarr.json")]
    assert any(key.startswith("cellData/reference_labels/") for key in chunk_keys)
    payload_keys = [
        key
        for key in chunk_keys
        if any(key.startswith(f"{prefix}/") for prefix in payload_paths)
    ]
    assert payload_keys == []
    cell_id_reads = Counter(
        key for key in chunk_keys if key.startswith("cellData/ids/")
    )
    # Binding and label reads share one row-identity validation per call.
    assert cell_id_reads and set(cell_id_reads.values()) == {1}
    assert not any(key.startswith("cellData/I/") for key in chunk_keys)


def test_binding_rejects_a_handle_whose_arrays_were_replaced(mapping_consumer_context):
    from scarf.mapping.models import ScaledPCAProjectionModel

    _, reference, query = mapping_consumer_context
    result = _write_projection(
        query,
        reference,
        indices=np.array([[0, 1], [1, 0]]),
        distances=np.array([[1.0, 9.0], [9.0, 1.0]]),
        uninformative=np.array([False, False]),
    )
    model = reference.model
    forged = replace(
        reference,
        model=ScaledPCAProjectionModel(
            feature_means=np.asarray(model.feature_means),
            feature_scales=np.asarray(model.feature_scales),
            center=np.asarray(model.center),
            loadings=np.asarray(model.loadings) * 2.0,
        ),
    )
    with pytest.raises(ValueError, match="does not match its stored artifact"):
        query.get_mapping_result(result, reference=forged)
    assert query.get_mapping_result(result, reference=reference).n_cells == 2


@pytest.mark.parametrize("stored_fingerprint", [False, True])
@pytest.mark.parametrize(
    "consumer",
    [
        "label_transfer",
        "column",
        "layout",
        "result",
        "result_arrays",
        "score",
        "lineage",
    ],
)
def test_reused_reference_rejects_reordered_cells(
    mapping_consumer_context, stored_fingerprint, consumer
):
    reference_store, reference, query = mapping_consumer_context
    _write_reference_labels(reference)
    selected = _reference_cell_indices(reference)[:2]
    result = _write_projection(
        query,
        reference,
        indices=np.array([[0, 1], [1, 0]]),
        distances=np.array([[1.0, 9.0], [9.0, 1.0]]),
        uninformative=np.array([False, False]),
    )
    _, layout = _write_reference_layout(reference, name="reference_layout")
    consume = {
        "label_transfer": lambda: query.run_label_transfer(
            result,
            reference=reference,
            reference_labels="reference_labels",
        ),
        "column": lambda: reference.fetch_cell_column("reference_labels"),
        "layout": lambda: query._mapping_score_data(
            result, reference=reference, layout=layout
        ),
        "result": lambda: query.get_mapping_result(result, reference=reference),
        "result_arrays": lambda: query.get_mapping_result(
            result, reference=reference, load_arrays=True
        ),
        "score": lambda: list(query.get_mapping_score(result, reference=reference)),
        "lineage": lambda: query.lineage(result, references=reference),
    }[consumer]
    if not stored_fingerprint:
        reference_store.RNA.z.attrs.pop("dataset_fingerprint")
        with pytest.raises(ValueError, match="inconsistent dataset identity"):
            consume()
        return
    consume()

    for column in ("ids", "reference_labels"):
        array = reference_store.zw[f"cellData/{column}"]
        values = np.asarray(array[:])
        values[selected] = values[selected[::-1]]
        array[:] = values

    with pytest.raises(ValueError, match="dataset fingerprint mismatch|row identity"):
        consume()


def test_reused_reference_requires_persisted_dataset_identity(mapping_consumer_context):
    reference_store, reference, query = mapping_consumer_context
    result = _write_projection(
        query,
        reference,
        indices=np.array([[0, 1], [1, 0]]),
        distances=np.array([[1.0, 9.0], [9.0, 1.0]]),
        uninformative=np.array([False, False]),
    )
    query.get_mapping_result(result, reference=reference)
    reference_store.RNA.z.attrs.pop("dataset_fingerprint")
    with pytest.raises(ValueError, match="inconsistent dataset identity"):
        query.get_mapping_result(result, reference=reference)


@pytest.mark.parametrize("attribute", ["dtype", "shape"])
def test_reused_reference_rejects_changed_array_metadata(
    mapping_consumer_context, attribute
):
    _, reference, query = mapping_consumer_context
    _write_reference_labels(reference)
    result = _write_projection(
        query,
        reference,
        indices=np.array([[0, 1], [1, 0]]),
        distances=np.array([[1.0, 9.0], [9.0, 1.0]]),
        uninformative=np.array([False, False]),
    )
    query_cells = query.snapshot_cell_selection("I")
    query.get_mapping_result(result, reference=reference)
    loadings = reference.model.loadings
    assert not loadings.flags.writeable
    # NumPy 2.5 deprecates assigning dtype or shape, so swap in a reinterpreted
    # view of the same bytes instead of editing the array in place.
    changed = (
        loadings.view(np.dtype(f"u{loadings.dtype.itemsize}"))
        if attribute == "dtype"
        else loadings.reshape(loadings.size)
    )
    object.__setattr__(reference.model, "loadings", changed)
    try:
        consumers = (
            lambda: query.get_mapping_result(result, reference=reference),
            lambda: query.run_label_transfer(
                result,
                reference=reference,
                reference_labels="reference_labels",
            ),
            lambda: query.run_mapping(reference, query_cells),
        )
        for consume in consumers:
            with pytest.raises(ValueError, match="does not match its stored artifact"):
                consume()
    finally:
        object.__setattr__(reference.model, "loadings", loadings)
    assert query.get_mapping_result(result, reference=reference).n_cells == 2
