import numpy as np
import pandas as pd
import pytest
import zarr
from scipy.sparse import csr_matrix
from zarr.storage import MemoryStore

from scarf.assay import Assay
from scarf.storage.budget import ResourceBudget
from scarf.trajectory.feature_dynamics import (
    aggregate_feature_profiles,
    knn_clustering,
)
from scarf.trajectory.feature_dynamics import (
    scatter_feature_clusters as _scatter_feature_clusters,
    validate_pseudotime_regressor as _validated_pseudotime_regressor,
)
from scarf.trajectory.pseudotime import (
    make_source_sink_vector as _make_source_sink_vector,
    random_walk_laplacian_transpose as _random_walk_laplacian_transpose,
    select_pseudotime_component as _select_pseudotime_component,
    truncated_pba_potential as _truncated_pba_potential,
    validate_source_sink_labels as _validate_source_sink_labels,
    validate_source_sink_vector as _validate_source_sink_vector,
)
from scarf.trajectory.parameters import AGGREGATION_ANN_DEFAULTS
from scarf.trajectory.results import PseudotimeMarkerResult


def _metadata_values(table):
    return {
        column: np.asarray(table.fetch_all(column)).copy() for column in table.columns
    }


def _assert_metadata_unchanged(table, before):
    assert set(table.columns) == set(before)
    for column, values in before.items():
        np.testing.assert_array_equal(table.fetch_all(column), values)


def test_source_sink_vector_supports_one_sided_supervision():
    labels = pd.Series(["source", "source", "other", "other"])
    vector = _make_source_sink_vector(labels, ["source"], [])

    assert np.array_equal(vector, [-1.0, -1.0, 1.0, 1.0])
    assert vector.sum() == pytest.approx(0.0)


def test_source_sink_labels_reject_missing_and_overlap():
    labels = pd.Series(["a", "b", "c"])

    with pytest.raises(ValueError, match="overlap"):
        _validate_source_sink_labels(labels, ["a"], ["a"], "test cells")
    with pytest.raises(ValueError, match="Missing sources"):
        _validate_source_sink_labels(labels, ["missing"], [], "test cells")


def test_source_sink_vector_rejects_unbalanced_and_nonfinite_values():
    for values in (
        np.array([-2.0, 1.0]),
        np.array([-1.0, 2.0]),
        np.array([-1.0, np.nan]),
        np.array([-1.0, np.inf]),
    ):
        with pytest.raises(ValueError):
            _validate_source_sink_vector(values, 2, "ss_vec")

    column = _validate_source_sink_vector(
        np.array([[-1.0], [1.0]]),
        2,
        "ss_vec",
    )
    assert column.shape == (2,)


def test_source_sink_vector_rejects_wrong_shapes():
    for values in (
        np.zeros((2, 2)),
        np.zeros((1, 2)),
        np.zeros(3),
    ):
        with pytest.raises(ValueError):
            _validate_source_sink_vector(values, 2, "ss_vec")


def test_raw_source_sink_scoring_round_trips_and_validates_snapshot(
    datastore,
    connectivity_graph,
):
    from scipy.sparse.csgraph import connected_components

    from scarf.storage.artifacts import ArtifactRef, artifact_group

    graph = datastore.load_graph(
        connectivity_graph,
        symmetric=True,
        upper_only=False,
    )
    _, components = connected_components(graph, directed=False)
    component_sizes = np.bincount(components)
    retained = np.flatnonzero(components == int(np.argmax(component_sizes)))
    source_sink = np.zeros(graph.shape[0], dtype=np.float64)
    source_sink[retained[0]] = -1.0
    source_sink[retained[-1]] = 1.0

    with pytest.raises(TypeError, match="min_max_norm_ptime"):
        datastore.run_pseudotime_scoring(
            connectivity_graph,
            ss_vec=source_sink,
            min_max_norm_ptime=1,
        )

    ref = datastore.run_pseudotime_scoring(
        connectivity_graph,
        ss_vec=source_sink,
        n_singular_vals=10,
    )
    result = datastore.load_pseudotime_scoring(ref)
    assert result.graph == connectivity_graph
    np.testing.assert_array_equal(
        result.valid, components == np.argmax(component_sizes)
    )

    status = datastore.inspect_artifact(ref)
    snapshot = ArtifactRef.from_dict(status.inputs["source_sink"])
    snapshot_status = datastore.inspect_artifact(snapshot)
    assert snapshot_status.operation == "snapshot_pseudotime_source_sink"
    assert snapshot_status.parameters == {"shape": [graph.shape[0]]}
    np.testing.assert_array_equal(
        artifact_group(datastore.zw, snapshot)["values"][:],
        source_sink,
    )

    snapshot_group = artifact_group(datastore.zw, snapshot)
    original_provenance = dict(snapshot_group.attrs["provenance"])
    corrupted = dict(original_provenance)
    corrupted["parameters"] = {"shape": [graph.shape[0], 1]}
    snapshot_group.attrs["provenance"] = corrupted
    try:
        with pytest.raises(ValueError, match="snapshot is invalid"):
            datastore.load_pseudotime_scoring(ref)
    finally:
        snapshot_group.attrs["provenance"] = original_provenance

    datastore.load_pseudotime_scoring(ref)


def test_raw_source_sink_scoring_owns_the_vector_before_validation_and_snapshot(
    datastore,
    connectivity_graph,
    monkeypatch,
):
    import scarf.datastore._operations.trajectory as trajectory_operations
    from scipy.sparse.csgraph import connected_components

    graph = datastore.load_graph(
        connectivity_graph,
        symmetric=True,
        upper_only=False,
    )
    _, components = connected_components(graph, directed=False)
    component_sizes = np.bincount(components)
    retained = np.flatnonzero(components == int(np.argmax(component_sizes)))
    source_sink = np.zeros(graph.shape[0], dtype=np.float64)
    source_sink[retained[0]] = -1.0
    source_sink[retained[-1]] = 1.0
    expected = source_sink.copy()
    original_validator = trajectory_operations._validate_source_sink_vector_impl
    original_potential = trajectory_operations._truncated_pba_potential_impl
    received: list[np.ndarray] = []

    def mutate_caller_after_validation(values, n_cells, context):
        validated = original_validator(values, n_cells, context)
        if context == "ss_vec":
            source_sink[:] *= -1.0
        return validated

    def capture_source_sink(laplacian, n_values, seed, values):
        received.append(np.array(values, copy=True))
        return original_potential(laplacian, n_values, seed, values)

    monkeypatch.setattr(
        trajectory_operations,
        "_validate_source_sink_vector_impl",
        mutate_caller_after_validation,
    )
    monkeypatch.setattr(
        trajectory_operations,
        "_truncated_pba_potential_impl",
        capture_source_sink,
    )
    ref = datastore.run_pseudotime_scoring(
        connectivity_graph,
        ss_vec=source_sink,
        n_singular_vals=10,
        invalidate_cache=True,
    )
    result = datastore.load_pseudotime_scoring(ref)

    assert len(received) == 1
    np.testing.assert_array_equal(received[0], expected[result.valid])
    np.testing.assert_array_equal(source_sink, -expected)


def test_source_sink_vector_rejects_fully_labelled_cells():
    labels = pd.Series(["source", "sink"])
    with pytest.raises(ValueError, match="All selected cells"):
        _make_source_sink_vector(labels, ["source"], ["sink"])


def test_largest_component_selection_is_deterministic_on_ties():
    adjacency = np.zeros((6, 6), dtype=float)
    for start in (0, 3):
        adjacency[start, start + 1] = adjacency[start + 1, start] = 1.0
        adjacency[start + 1, start + 2] = adjacency[start + 2, start + 1] = 1.0
    selected_indices = np.array([10, 11, 12, 1, 2, 3])

    retained, sizes = _select_pseudotime_component(
        csr_matrix(adjacency),
        selected_indices,
        "largest",
    )

    assert sizes == [3, 3]
    assert np.array_equal(retained, [False, False, False, True, True, True])
    with pytest.raises(ValueError, match="connected components"):
        _select_pseudotime_component(
            csr_matrix(adjacency),
            selected_indices,
            "error",
        )


def test_truncated_pba_operator_matches_dense_svd_reference():
    adjacency = np.zeros((8, 8), dtype=float)
    weights = [1.0, 2.0, 1.5, 3.0, 0.75, 2.5, 1.25]
    for index, weight in enumerate(weights):
        adjacency[index, index + 1] = weight
        adjacency[index + 1, index] = weight
    laplacian_transpose = _random_walk_laplacian_transpose(csr_matrix(adjacency))
    source_sink = np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    k = 5

    actual = _truncated_pba_potential(
        laplacian_transpose,
        k,
        17,
        source_sink,
    )

    left, singular_values, right_t = np.linalg.svd(
        laplacian_transpose.toarray(),
        full_matrices=False,
    )
    retained_modes = np.argsort(singular_values)[:k]
    retained_modes = retained_modes[np.argsort(singular_values[retained_modes])]
    nonzero_modes = retained_modes[1:]
    expected = left[:, nonzero_modes] @ (
        (1.0 / singular_values[nonzero_modes])
        * (right_t[nonzero_modes, :] @ source_sink)
    )

    assert np.allclose(actual, expected, rtol=1e-6, atol=1e-8)


def test_truncated_pba_logs_svd_stage_start():
    from loguru import logger

    adjacency = np.zeros((6, 6), dtype=float)
    for index in range(5):
        adjacency[index, index + 1] = 1.0
        adjacency[index + 1, index] = 1.0
    laplacian_transpose = _random_walk_laplacian_transpose(csr_matrix(adjacency))
    source_sink = np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    messages: list[str] = []
    sink = logger.add(
        lambda message: messages.append(message.record["message"]),
        level="DEBUG",
    )
    try:
        _truncated_pba_potential(laplacian_transpose, 3, 7, source_sink)
    finally:
        logger.remove(sink)

    assert any(
        "Pseudotime scoring: calculating SVD" in msg
        and "k=3" in msg
        and "shape=(6, 6)" in msg
        for msg in messages
    )


def test_dense_pba_reference_orders_a_path_from_source_to_sink():
    adjacency = np.zeros((7, 7), dtype=float)
    for index in range(6):
        adjacency[index, index + 1] = 1.0
        adjacency[index + 1, index] = 1.0
    laplacian_transpose = _random_walk_laplacian_transpose(csr_matrix(adjacency))
    source_sink = np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

    potential = np.linalg.pinv(laplacian_transpose.toarray().T) @ source_sink

    assert potential[0] < potential[-1]
    assert np.all(np.diff(potential) > 0)


def test_marker_search_returns_an_explicit_artifact_without_feature_writes(
    datastore,
    pseudotime_scoring,
    detected_features,
):
    from scarf.storage.artifacts import (
        ArtifactRef,
        artifact_path,
        fingerprint_stored_strings,
    )

    feature_metadata_before = _metadata_values(datastore.RNA.feats)
    ref = datastore.run_pseudotime_marker_search(
        pseudotime_scoring,
        features=detected_features,
        min_cells=1,
    )
    result = datastore.load_pseudotime_markers(ref)

    assert isinstance(ref, ArtifactRef)
    assert isinstance(result, PseudotimeMarkerResult)
    assert result.ref == ref
    assert result.pseudotime == pseudotime_scoring
    assert result.feature_selection == detected_features
    _assert_metadata_unchanged(datastore.RNA.feats, feature_metadata_before)
    group = datastore.zw[artifact_path(ref)]
    assert set(group.array_keys()) == {
        "feature_ids",
        "feature_names",
        "p_value",
        "p_value_adjusted",
        "r_value",
    }
    inputs = datastore.inspect_artifact(ref).inputs or {}
    assert inputs["ordered_feature_ids_fingerprint"] == fingerprint_stored_strings(
        group["feature_ids"]
    )
    assert inputs["ordered_feature_names_fingerprint"] == (
        fingerprint_stored_strings(group["feature_names"])
    )
    assert "p_value_adjusted" in result.table.columns


def test_trajectory_feature_selection_indices_are_read_blockwise(
    datastore,
    detected_features,
    monkeypatch,
):
    import scarf.datastore._operations.trajectory as trajectory_operations
    from scarf.storage.artifacts import artifact_path

    selection_values_path = f"{artifact_path(detected_features)}/values"
    original_getitem = zarr.Array.__getitem__

    def reject_full_selection_read(array, key):
        if array.path == selection_values_path and (
            key is Ellipsis
            or (
                isinstance(key, slice)
                and key.start is None
                and key.stop is None
                and key.step is None
            )
        ):
            raise AssertionError("trajectory materialized the full feature selection")
        return original_getitem(array, key)

    monkeypatch.setattr(zarr.Array, "__getitem__", reject_full_selection_read)

    resolved, indices = trajectory_operations._resolve_feature_indices(
        datastore,
        datastore.RNA,
        detected_features,
    )

    assert resolved == detected_features
    assert indices.dtype == np.int64
    assert len(indices) > 0


def test_marker_loader_uses_frozen_feature_names(
    datastore,
    pseudotime_markers,
):
    expected = datastore.load_pseudotime_markers(
        pseudotime_markers
    ).table.feature_name.to_numpy(copy=True)
    live_names = datastore.RNA.feats._get_array("names")
    original = np.asarray(live_names[:]).copy()
    try:
        renamed = original.astype(str)
        renamed[:] = [f"renamed_{index}" for index in range(len(renamed))]
        live_names[:] = renamed
        loaded = datastore.load_pseudotime_markers(pseudotime_markers)
        np.testing.assert_array_equal(
            loaded.table.feature_name.to_numpy(),
            expected,
        )
    finally:
        live_names[:] = original


def test_trajectory_loaders_do_not_depend_on_live_normalization_settings(
    datastore,
    pseudotime_markers,
    pseudotime_aggregation,
    pseudotime_scoring,
    detected_features,
    monkeypatch,
):
    marker_before = datastore.load_pseudotime_markers(pseudotime_markers)
    aggregation_before = datastore.load_pseudotime_aggregation(pseudotime_aggregation)
    monkeypatch.setattr(datastore.RNA, "sf", float(datastore.RNA.sf) * 2.0)

    marker_after = datastore.load_pseudotime_markers(pseudotime_markers)
    aggregation_after = datastore.load_pseudotime_aggregation(pseudotime_aggregation)
    pd.testing.assert_frame_equal(marker_after.table, marker_before.table)
    np.testing.assert_array_equal(
        aggregation_after.feature_indices,
        aggregation_before.feature_indices,
    )
    np.testing.assert_array_equal(
        aggregation_after.feature_clusters,
        aggregation_before.feature_clusters,
    )

    rerun = datastore.run_pseudotime_marker_search(
        pseudotime_scoring,
        features=detected_features,
    )
    assert rerun != pseudotime_markers


def test_aggregation_provenance_records_resolved_ann_defaults(
    datastore,
    pseudotime_aggregation,
):
    status = datastore.inspect_artifact(pseudotime_aggregation)
    expected = dict(AGGREGATION_ANN_DEFAULTS)
    expected["dim"] = 10
    assert (status.parameters or {})["ann_params"] == expected

    result = datastore.load_pseudotime_aggregation(pseudotime_aggregation)
    arguments = {
        "pseudotime": result.pseudotime,
        "features": result.feature_selection,
        "n_clusters": 15,
        "window_size": 50,
        "chunk_size": 10,
        "ann_params": {"random_seed": 445},
    }
    overridden = datastore.run_pseudotime_aggregation(**arguments)
    assert overridden != pseudotime_aggregation
    overridden_parameters = datastore.inspect_artifact(overridden).parameters or {}
    assert overridden_parameters["ann_params"] == {
        **expected,
        "random_seed": 445,
    }
    assert datastore.run_pseudotime_aggregation(**arguments) == overridden


@pytest.mark.parametrize(
    ("batch_size", "error_type"),
    [
        (True, TypeError),
        (False, TypeError),
        (0, ValueError),
        (-1, ValueError),
        (1.5, TypeError),
    ],
)
def test_trajectory_feature_batch_sizes_are_strictly_positive_integers(
    datastore,
    pseudotime_scoring,
    detected_features,
    batch_size,
    error_type,
):
    with pytest.raises(error_type, match="gene_batch_size"):
        datastore.run_pseudotime_marker_search(
            pseudotime_scoring,
            features=detected_features,
            gene_batch_size=batch_size,
        )
    with pytest.raises(error_type, match="batch_size"):
        datastore.run_pseudotime_aggregation(
            pseudotime_scoring,
            features=detected_features,
            batch_size=batch_size,
        )


def test_aggregation_rejects_nonmapping_ann_parameters(
    datastore,
    pseudotime_scoring,
    detected_features,
):
    with pytest.raises(TypeError, match="ann_params must be a mapping"):
        datastore.run_pseudotime_aggregation(
            pseudotime_scoring,
            features=detected_features,
            ann_params=[("M", 20)],  # type: ignore[arg-type]
        )


def test_trajectory_operations_require_boolean_cache_invalidation(
    datastore,
    connectivity_graph,
    pseudotime_scoring,
    detected_features,
):
    with pytest.raises(TypeError, match="invalidate_cache"):
        datastore.run_pseudotime_scoring(
            connectivity_graph,
            invalidate_cache=1,
        )
    with pytest.raises(TypeError, match="invalidate_cache"):
        datastore.run_fate_mapping(
            pseudotime_scoring,
            pseudotime_scoring,
            sinks=[1],
            invalidate_cache=1,
        )
    with pytest.raises(TypeError, match="invalidate_cache"):
        datastore.run_pseudotime_marker_search(
            pseudotime_scoring,
            features=detected_features,
            invalidate_cache=1,
        )
    with pytest.raises(TypeError, match="invalidate_cache"):
        datastore.run_pseudotime_aggregation(
            pseudotime_scoring,
            features=detected_features,
            invalidate_cache=1,
        )


def test_marker_identity_change_during_computation_leaves_artifact_incomplete(
    datastore,
    pseudotime_scoring,
    detected_features,
    monkeypatch,
):
    import scarf.features.markers as marker_algorithms

    before = set(
        datastore.list_artifacts(
            kind="pseudotime_markers",
            from_assay="RNA",
        )
    )
    live_names = datastore.RNA.feats._get_array("names")
    original_names = np.asarray(live_names[:]).copy()
    renamed = original_names.astype(str)
    renamed[:] = [f"changed_{index}" for index in range(len(renamed))]
    original_search = marker_algorithms.find_markers_by_regression

    def mutate_after_computation(*args, **kwargs):
        result = original_search(*args, **kwargs)
        live_names[:] = renamed
        return result

    monkeypatch.setattr(
        marker_algorithms,
        "find_markers_by_regression",
        mutate_after_computation,
    )
    try:
        with pytest.raises(ValueError, match="identities changed"):
            datastore.run_pseudotime_marker_search(
                pseudotime_scoring,
                features=detected_features,
                min_cells=1,
                invalidate_cache=True,
            )
    finally:
        live_names[:] = original_names

    created = (
        set(
            datastore.list_artifacts(
                kind="pseudotime_markers",
                from_assay="RNA",
            )
        )
        - before
    )
    assert len(created) == 1
    assert not datastore.inspect_artifact(created.pop()).complete


def test_marker_normalization_change_during_computation_leaves_artifact_incomplete(
    datastore,
    pseudotime_scoring,
    detected_features,
    monkeypatch,
):
    import scarf.features.markers as marker_algorithms

    before = set(
        datastore.list_artifacts(
            kind="pseudotime_markers",
            from_assay="RNA",
        )
    )
    original_size_factor = datastore.RNA.sf
    assert original_size_factor is not None
    original_search = marker_algorithms.find_markers_by_regression

    def mutate_after_computation(*args, **kwargs):
        result = original_search(*args, **kwargs)
        datastore.RNA.sf = original_size_factor * 2
        return result

    monkeypatch.setattr(
        marker_algorithms,
        "find_markers_by_regression",
        mutate_after_computation,
    )
    try:
        with pytest.raises(ValueError, match="normalization settings changed"):
            datastore.run_pseudotime_marker_search(
                pseudotime_scoring,
                features=detected_features,
                min_cells=1,
                invalidate_cache=True,
            )
    finally:
        datastore.RNA.sf = original_size_factor

    created = (
        set(
            datastore.list_artifacts(
                kind="pseudotime_markers",
                from_assay="RNA",
            )
        )
        - before
    )
    assert len(created) == 1
    assert not datastore.inspect_artifact(created.pop()).complete


def test_aggregation_normalization_change_leaves_artifact_incomplete(
    datastore,
    pseudotime_scoring,
    detected_features,
    monkeypatch,
):
    before = set(
        datastore.list_artifacts(
            kind="pseudotime_aggregation",
            from_assay="RNA",
        )
    )
    original_size_factor = datastore.RNA.sf
    assert original_size_factor is not None
    original_writer = datastore.RNA._write_aggregated_ordering_group

    def mutate_after_aggregation(*args, **kwargs):
        result = original_writer(*args, **kwargs)
        datastore.RNA.sf = original_size_factor * 2
        return result

    monkeypatch.setattr(
        datastore.RNA,
        "_write_aggregated_ordering_group",
        mutate_after_aggregation,
    )
    try:
        with pytest.raises(ValueError, match="normalization settings changed"):
            datastore.run_pseudotime_aggregation(
                pseudotime_scoring,
                features=detected_features,
                window_size=10,
                chunk_size=5,
                n_neighbours=3,
                n_clusters=2,
                invalidate_cache=True,
            )
    finally:
        datastore.RNA.sf = original_size_factor

    created = (
        set(
            datastore.list_artifacts(
                kind="pseudotime_aggregation",
                from_assay="RNA",
            )
        )
        - before
    )
    assert len(created) == 1
    assert not datastore.inspect_artifact(created.pop()).complete


def test_incomplete_pseudotime_marker_artifact_is_recomputed(
    datastore,
    pseudotime_scoring,
    detected_features,
    monkeypatch,
):
    import scarf.features.markers as marker_algorithms
    from scarf.storage.artifacts import artifact_path

    arguments = {
        "pseudotime": pseudotime_scoring,
        "features": detected_features,
        "min_cells": 1,
        "gene_batch_size": 8,
    }
    first = datastore.run_pseudotime_marker_search(**arguments)
    old_artifact = datastore.zw[artifact_path(first)]
    del old_artifact["p_value_adjusted"]
    feature_metadata_before = _metadata_values(datastore.RNA.feats)

    original = marker_algorithms.find_markers_by_regression
    calls = 0

    def tracked_marker_search(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        marker_algorithms,
        "find_markers_by_regression",
        tracked_marker_search,
    )
    second = datastore.run_pseudotime_marker_search(**arguments)

    assert calls == 1
    assert second != first
    assert "p_value_adjusted" not in old_artifact
    assert "p_value_adjusted" in datastore.zw[artifact_path(second)]
    _assert_metadata_unchanged(datastore.RNA.feats, feature_metadata_before)


def test_regressor_validation_points_to_component_validity_key():
    with pytest.raises(ValueError, match="ptime__valid"):
        _validated_pseudotime_regressor(
            np.array([0.0, np.nan]),
            2,
            "ptime",
            "I",
            has_validity_column=True,
        )


@pytest.mark.parametrize(
    ("values", "expected_size", "error_type", "match"),
    [
        (["early", "late"], 2, TypeError, "must be numeric"),
        (np.array([[0.0, 1.0]]), 1, ValueError, "one-dimensional"),
        (np.array([0.0]), 2, ValueError, "selects 2 cells"),
        (np.array([0.0, np.inf]), 2, ValueError, "non-finite"),
        (np.array([0.5, 0.5]), 2, ValueError, "distinct values"),
    ],
)
def test_regressor_validation_rejects_invalid_columns(
    values,
    expected_size,
    error_type,
    match,
):
    with pytest.raises(error_type, match=match):
        _validated_pseudotime_regressor(
            values,
            expected_size,
            "ptime",
            "I",
            has_validity_column=False,
        )


class _AggregationCells:
    def __init__(self, ordering: np.ndarray):
        self.ordering = ordering
        self.N = len(ordering)

    def fetch(self, _ordering_key: str, key: str) -> np.ndarray:
        assert key == "I"
        return self.ordering


class _AggregationAssay:
    def __init__(self, expression: np.ndarray, ordering: np.ndarray):
        self.expression = expression
        self.cells = _AggregationCells(ordering)
        self.feats = type("Features", (), {"N": expression.shape[1]})()
        self.z = zarr.open_group(store=MemoryStore(), mode="w")
        self.nthreads = 1
        self.resources = ResourceBudget(1024**3, 1)

    def iter_normed_feature_wise(self, *_args, **_kwargs):
        yield pd.DataFrame(
            self.expression,
            columns=np.arange(self.expression.shape[1]),
        )


@pytest.mark.parametrize(
    ("ordering", "window_size", "chunk_size", "error_type", "match"),
    [
        (np.array([[0.0, 1.0]]), 2, 2, ValueError, "one-dimensional"),
        (np.array([0.0, np.nan]), 2, 2, ValueError, "finite values"),
        (np.array([0.0, 1.0]), True, 2, TypeError, "window_size"),
        (np.array([0.0, 1.0]), 2, True, TypeError, "chunk_size"),
        (np.array([0.0, 1.0]), 0, 2, ValueError, "window_size"),
        (np.array([0.0, 1.0]), 2, 0, ValueError, "chunk_size"),
    ],
)
def test_aggregation_rejects_invalid_ordering_and_sizes(
    ordering,
    window_size,
    chunk_size,
    error_type,
    match,
):
    assay = _AggregationAssay(
        np.ones((ordering.shape[0], 2)),
        ordering,
    )

    with pytest.raises(error_type, match=match):
        Assay._prepare_aggregated_ordering(
            assay,
            np.arange(ordering.shape[0]),
            np.arange(assay.expression.shape[1]),
            ordering,
            min_exp=0.0,
            window_size=window_size,
            chunk_size=chunk_size,
            smoothen=False,
            z_scale=False,
            norm_params={},
        )


class _ShapeOnlyArray:
    def __init__(self, shape: tuple[int, ...]):
        self.shape = shape


def test_knn_clustering_rejects_infeasible_parameters():
    with pytest.raises(ValueError, match="two-dimensional"):
        knn_clustering(_ShapeOnlyArray((3,)), 1, 1, 1)
    with pytest.raises(ValueError, match="At least two"):
        knn_clustering(_ShapeOnlyArray((1, 3)), 1, 1, 1)
    with pytest.raises(TypeError, match="n_neighbours"):
        knn_clustering(_ShapeOnlyArray((3, 3)), True, 2, 1)
    with pytest.raises(ValueError, match="n_neighbours"):
        knn_clustering(_ShapeOnlyArray((3, 3)), np.int64(3), np.int32(2), 1)
    with pytest.raises(ValueError, match="n_clusters"):
        knn_clustering(_ShapeOnlyArray((3, 3)), np.int32(1), np.int64(4), 1)
    with pytest.raises(TypeError, match="n_clusters"):
        knn_clustering(_ShapeOnlyArray((3, 3)), 1, np.bool_(True), 1)
    with pytest.raises(ValueError, match="effective bin count"):
        knn_clustering(
            _ShapeOnlyArray((3, 3)),
            1,
            2,
            1,
            {"dim": 2},
        )
    with pytest.raises(ValueError, match="smaller than the feature count"):
        knn_clustering(
            _ShapeOnlyArray((3, 3)),
            1,
            2,
            1,
            {"max_elements": 2},
        )


def test_feature_cluster_scatter_honors_custom_unassigned_value():
    values = _scatter_feature_clusters(
        5,
        np.array([1, 3]),
        np.array([1, 2]),
        0,
    )

    assert np.array_equal(values, [0, 1, 0, 2, 0])
    with pytest.raises(ValueError, match="misaligned"):
        _scatter_feature_clusters(3, np.array([0, 1]), np.array([1]), 0)
    with pytest.raises(ValueError, match="conflicts"):
        _scatter_feature_clusters(3, np.array([0]), np.array([1]), 1)


def test_aggregate_feature_profiles_orders_filters_and_bins():
    values = np.array(
        [
            [3.0, 5.0, 0.01],
            [1.0, 5.0, 0.01],
            [2.0, 5.0, 0.01],
            [0.0, 5.0, 0.01],
        ],
        dtype=float,
    )
    ordering = np.array([3, 1, 2, 0])
    feature_indices = np.array([10, 11, 12])

    binned, valid = aggregate_feature_profiles(
        values,
        ordering,
        feature_indices,
        min_expression=0.1,
        window_size=2,
        n_bins=2,
        smooth=False,
        z_scale=False,
    )

    np.testing.assert_array_equal(valid, [True, False, False])
    assert binned.shape == (3, 2)
    np.testing.assert_allclose(binned[0], [0.5, 2.5])
    np.testing.assert_array_equal(binned[1], 0.0)
    np.testing.assert_array_equal(binned[2], 0.0)

    z_binned, z_valid = aggregate_feature_profiles(
        values,
        ordering,
        feature_indices,
        min_expression=0.1,
        window_size=2,
        n_bins=2,
        smooth=False,
        z_scale=True,
    )
    np.testing.assert_array_equal(z_valid, valid)
    np.testing.assert_allclose(z_binned[0].mean(), 0.0, atol=1e-12)
    assert z_binned[0, 0] < 0 < z_binned[0, 1]

    with pytest.raises(ValueError, match="non-finite values"):
        aggregate_feature_profiles(
            np.array([[1.0, np.nan], [2.0, 3.0]]),
            np.array([0, 1]),
            np.array([0, 1]),
            min_expression=0.0,
            window_size=1,
            n_bins=1,
            smooth=False,
            z_scale=False,
        )
