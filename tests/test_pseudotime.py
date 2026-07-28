import numpy as np
import pandas as pd
import pytest
import zarr
from scipy.sparse import csr_matrix
from zarr.storage import MemoryStore

from scarf.assay import Assay
from scarf.storage.budget import ResourceBudget
from scarf.trajectory.feature_dynamics import knn_clustering
from scarf.datastore.datastore import DataStore
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
        level="INFO",
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

    @staticmethod
    def fetch_all(key: str) -> np.ndarray:
        assert key == "names"
        return np.array(["zero", "one", "two", "three"])


class _MarkerAssay:
    def __init__(self):
        self.name = "RNA"
        self.cells = _FakeCells(np.array([0.0, 1.0, 2.0, 3.0]))
        self.feats = _FakeFeatures()
        self.resources = ResourceBudget(1024**3, 1)

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
        self.resources = ResourceBudget(1024**3, 1)

    def _get_assay(self, _from_assay: str):
        return self.assay


def test_marker_search_writes_strict_feature_subset_with_subset_key():
    store = _MarkerStore()

    result = DataStore.run_pseudotime_marker_search(
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
    assert isinstance(result, PseudotimeMarkerResult)
    assert result.correlation_key == "I__ptime__r"
    assert result.p_value_key == "I__ptime__p"
    assert result.table["feature_index"].tolist() == [1, 3]
    assert result.table["feature_name"].tolist() == ["one", "three"]


def test_regressor_validation_points_to_component_validity_key():
    with pytest.raises(ValueError, match="ptime__valid"):
        _validated_pseudotime_regressor(
            np.array([0.0, np.nan]),
            2,
            "ptime",
            "I",
            has_validity_column=True,
        )


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
        self.resources = ResourceBudget(1024**3, 1)

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
    assert "schema_version" not in group.attrs
    assert all(isinstance(value, str) for value in group.attrs["hashes"])


def test_incomplete_aggregation_cache_is_rebuilt():
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
    del group["valid_features"]
    group["data"][:] = 999.0

    aggregated, _ = Assay.save_aggregated_ordering(assay, **kwargs)

    assert not np.all(aggregated.compute() == 999.0)
    assert "valid_features" in assay.z["aggregated_I_I_ptime"]
    assert "schema_version" not in assay.z["aggregated_I_I_ptime"].attrs


class _ShapeOnlyArray:
    def __init__(self, shape: tuple[int, int]):
        self.shape = shape


def test_knn_clustering_rejects_infeasible_parameters():
    with pytest.raises(ValueError, match="At least two"):
        knn_clustering(_ShapeOnlyArray((1, 3)), 1, 1, 1)
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
    with pytest.raises(ValueError, match="conflicts"):
        _scatter_feature_clusters(3, np.array([0]), np.array([1]), 1)
