import numpy as np
import pandas as pd
import pytest
import zarr
from scipy.sparse import csr_matrix
from zarr.storage import MemoryStore

from scarf.assay import (
    PSEUDOTIME_AGGREGATION_SCHEMA_VERSION,
    Assay,
)
from scarf.datastore.datastore import (
    DataStore,
    _scatter_feature_clusters,
    _validated_pseudotime_regressor,
)
from scarf.datastore.graph_datastore import (
    _make_source_sink_vector,
    _random_walk_laplacian_transpose,
    _select_pseudotime_component,
    _truncated_pba_potential,
    _validate_source_sink_labels,
    _validate_source_sink_vector,
)
from scarf.markers import knn_clustering


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


class _FakeCells:
    def __init__(self, values: np.ndarray, columns: list[str] | None = None):
        self.values = values
        self.columns = ["ptime"] if columns is None else columns

    def fetch(self, _column: str, key: str) -> np.ndarray:
        assert key
        return self.values

    def active_index(self, _key: str) -> np.ndarray:
        return np.arange(self.values.shape[0])


class _FakeFeatures:
    def __init__(self):
        self.insertions: list[tuple[str, np.ndarray, str]] = []

    @staticmethod
    def active_index(key: str) -> np.ndarray:
        assert key == "subset"
        return np.array([1, 3])

    def insert(
        self,
        column_name: str,
        values: np.ndarray,
        *,
        key: str,
        overwrite: bool,
    ) -> None:
        assert overwrite
        self.insertions.append((column_name, values, key))


class _MarkerAssay:
    def __init__(self):
        self.cells = _FakeCells(np.array([0.0, 1.0, 2.0, 3.0]))
        self.feats = _FakeFeatures()

    @staticmethod
    def iter_normed_feature_wise(**_kwargs):
        yield pd.DataFrame(
            {
                1: [0.0, 1.0, 2.0, 3.0],
                3: [3.0, 2.0, 1.0, 0.0],
            }
        )


class _MarkerStore:
    def __init__(self):
        self.assay = _MarkerAssay()

    def _get_assay(self, _from_assay: str):
        return self.assay


def test_marker_search_writes_strict_feature_subset_with_subset_key():
    store = _MarkerStore()

    DataStore.run_pseudotime_marker_search(
        store,
        from_assay="RNA",
        cell_key="I",
        feat_key="subset",
        pseudotime_key="ptime",
        min_cells=1,
    )

    assert [item[2] for item in store.assay.feats.insertions] == [
        "subset",
        "subset",
    ]
    assert all(item[1].shape == (2,) for item in store.assay.feats.insertions)


def test_regressor_validation_points_to_component_validity_key():
    assay = type(
        "FakeAssay",
        (),
        {
            "cells": _FakeCells(
                np.array([0.0, np.nan]),
                ["ptime", "ptime__valid"],
            )
        },
    )()

    with pytest.raises(ValueError, match="ptime__valid"):
        _validated_pseudotime_regressor(assay, "I", "ptime")


class _AggregationCells:
    def __init__(self, ordering: np.ndarray):
        self.ordering = ordering

    def fetch(self, _ordering_key: str, key: str) -> np.ndarray:
        assert key == "I"
        return self.ordering


class _AggregationAssay:
    def __init__(self, expression: np.ndarray, ordering: np.ndarray):
        self.expression = expression
        self.cells = _AggregationCells(ordering)
        self.z = zarr.open_group(store=MemoryStore(), mode="w")
        self.nthreads = 1

    def _get_cell_feat_idx(
        self,
        _cell_key: str,
        _feat_key: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        return np.arange(self.expression.shape[0]), np.arange(self.expression.shape[1])

    def iter_normed_feature_wise(self, *_args, **_kwargs):
        yield pd.DataFrame(
            self.expression,
            columns=np.arange(self.expression.shape[1]),
        )


def test_aggregation_orders_without_smoothing_and_filters_constant_profiles():
    expression = np.array(
        [
            [10.0, 5.0],
            [20.0, 5.0],
            [30.0, 5.0],
            [40.0, 5.0],
        ]
    )
    assay = _AggregationAssay(expression, np.array([2.0, 0.0, 3.0, 1.0]))

    aggregated, feature_indices = Assay.save_aggregated_ordering(
        assay,
        cell_key="I",
        feat_key="I",
        ordering_key="ptime",
        min_exp=0.0,
        smoothen=False,
        z_scale=False,
        window_size=20,
        chunk_size=20,
        batch_size=2,
    )

    assert np.array_equal(feature_indices, [0])
    assert np.array_equal(
        aggregated.compute(),
        np.array([[20.0, 40.0, 10.0, 30.0]]),
    )
    group = assay.z["aggregated_I_I_ptime"]
    assert np.array_equal(group["valid_features"][:], [True, False])
    assert np.isfinite(group["data"][:]).all()
    assert group.attrs["schema_version"] == PSEUDOTIME_AGGREGATION_SCHEMA_VERSION
    assert all(isinstance(value, str) for value in group.attrs["hashes"])


def test_old_aggregation_schema_is_rebuilt():
    expression = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])
    assay = _AggregationAssay(expression, np.arange(4, dtype=float))
    kwargs = {
        "cell_key": "I",
        "feat_key": "I",
        "ordering_key": "ptime",
        "smoothen": False,
        "z_scale": False,
        "window_size": 2,
        "chunk_size": 2,
        "batch_size": 2,
    }

    Assay.save_aggregated_ordering(assay, **kwargs)
    group = assay.z["aggregated_I_I_ptime"]
    group.attrs["schema_version"] = 1
    group["data"][:] = 999.0

    aggregated, _ = Assay.save_aggregated_ordering(assay, **kwargs)

    assert not np.all(aggregated.compute() == 999.0)
    assert (
        assay.z["aggregated_I_I_ptime"].attrs["schema_version"]
        == PSEUDOTIME_AGGREGATION_SCHEMA_VERSION
    )


class _ShapeOnlyArray:
    def __init__(self, shape: tuple[int, int]):
        self.shape = shape


def test_knn_clustering_rejects_infeasible_parameters():
    with pytest.raises(ValueError, match="At least two"):
        knn_clustering(_ShapeOnlyArray((1, 3)), 1, 1, 1)
    with pytest.raises(ValueError, match="n_neighbours"):
        knn_clustering(_ShapeOnlyArray((3, 3)), 3, 2, 1)
    with pytest.raises(ValueError, match="n_clusters"):
        knn_clustering(_ShapeOnlyArray((3, 3)), 1, 4, 1)


def test_feature_cluster_scatter_honors_custom_unassigned_value():
    values = _scatter_feature_clusters(
        5,
        np.array([1, 3]),
        np.array([1, 2]),
        0,
    )

    assert np.array_equal(values, [0, 1, 0, 2, 0])
    with pytest.raises(ValueError, match="conflicts"):
        _scatter_feature_clusters(3, np.array([0]), np.array([1]), 1)
