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
    from scarf.storage.artifacts import ArtifactRef, artifact_path

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
    assert set(datastore.zw[artifact_path(ref)].array_keys()) == {
        "p_value",
        "p_value_adjusted",
        "r_value",
    }
    assert "p_value_adjusted" in result.table.columns


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
