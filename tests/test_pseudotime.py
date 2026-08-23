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
        "subset",
    ]
    assert all(item[1].shape == (2,) for item in store.assay.feats.insertions)
    assert isinstance(result, PseudotimeMarkerResult)
    assert result.correlation_key == "I__ptime__r"
    assert result.p_value_key == "I__ptime__p"
    assert result.p_value_adjusted_key == "I__ptime__padj"
    assert result.table["feature_index"].tolist() == [1, 3]
    assert result.table["feature_name"].tolist() == ["one", "three"]
    assert "p_value_adjusted" in result.table.columns


def test_incomplete_current_pseudotime_marker_artifact_is_recomputed(
    datastore_ephemeral,
    monkeypatch,
):
    import scarf.features.markers as marker_algorithms
    from scarf.storage.artifacts import ArtifactRef, artifact_path

    assay = datastore_ephemeral.RNA
    datastore_ephemeral.cells.insert(
        "current_ptime",
        np.linspace(0.0, 1.0, datastore_ephemeral.cells.N),
        overwrite=True,
    )
    feature_mask = np.zeros(assay.feats.N, dtype=bool)
    feature_mask[:16] = True
    assay.feats.insert(
        "current_ptime_features",
        feature_mask,
        overwrite=True,
    )
    arguments = {
        "from_assay": "RNA",
        "cell_key": "I",
        "feat_key": "current_ptime_features",
        "pseudotime_key": "current_ptime",
        "min_cells": 1,
        "gene_batch_size": 8,
    }
    first = datastore_ephemeral.run_pseudotime_marker_search(**arguments)
    old_ref = ArtifactRef.from_dict(
        assay.z["featureData"][first.correlation_key].attrs["source_artifact"]
    )
    old_artifact = datastore_ephemeral.zw[artifact_path(old_ref)]
    del old_artifact["p_value_adjusted"]

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
    second = datastore_ephemeral.run_pseudotime_marker_search(**arguments)

    new_ref = ArtifactRef.from_dict(
        assay.z["featureData"][second.correlation_key].attrs["source_artifact"]
    )
    assert calls == 1
    assert new_ref != old_ref
    assert "p_value_adjusted" not in old_artifact
    assert "p_value_adjusted" in datastore_ephemeral.zw[artifact_path(new_ref)]


def _prepare_legacy_pseudotime_marker_artifact(datastore_ephemeral):
    from scarf.storage.artifacts import ArtifactRef, artifact_path

    assay = datastore_ephemeral.RNA
    n_cells = datastore_ephemeral.cells.N
    datastore_ephemeral.cells.insert(
        "legacy_ptime",
        np.linspace(0.0, 1.0, n_cells),
        overwrite=True,
    )
    feature_mask = np.zeros(assay.feats.N, dtype=bool)
    feature_mask[:16] = True
    assay.feats.insert(
        "legacy_ptime_features",
        feature_mask,
        overwrite=True,
    )
    arguments = {
        "from_assay": "RNA",
        "cell_key": "I",
        "feat_key": "legacy_ptime_features",
        "pseudotime_key": "legacy_ptime",
        "min_cells": 1,
        "gene_batch_size": 8,
    }
    first = datastore_ephemeral.run_pseudotime_marker_search(**arguments)
    ref = ArtifactRef.from_dict(
        assay.z["featureData"][first.correlation_key].attrs["source_artifact"]
    )
    artifact = datastore_ephemeral.zw[artifact_path(ref)]
    provenance = dict(artifact.attrs["provenance"])
    parameters = dict(provenance["parameters"])
    for field_name in (
        "association_method",
        "p_value_method",
        "adjustment_method",
        "adjustment_scope",
    ):
        parameters.pop(field_name)
    provenance["parameters"] = parameters
    artifact.attrs["provenance"] = provenance
    raw_p_values = np.asarray(artifact["p_value"][:]).copy()
    raw_r_values = np.asarray(artifact["r_value"][:]).copy()
    assert np.isfinite(raw_p_values).any()
    del artifact["p_value_adjusted"]
    assay.feats.drop(first.p_value_adjusted_key)
    return (
        arguments,
        first,
        ref,
        artifact,
        raw_r_values,
        raw_p_values,
    )


def test_legacy_pseudotime_marker_artifact_is_upgraded_additively(
    datastore_ephemeral,
    monkeypatch,
):
    import scarf.features.markers as marker_algorithms
    from scarf.features.markers.correction import _bh_adjusted_pvalues
    from scarf.storage.artifacts import ArtifactRef, artifact_path

    (
        arguments,
        first,
        ref,
        artifact,
        raw_r_values,
        raw_p_values,
    ) = _prepare_legacy_pseudotime_marker_artifact(datastore_ephemeral)
    attrs_before = dict(artifact.attrs)
    arrays_before = set(artifact.array_keys())

    def fail_if_recomputed(*_args, **_kwargs):
        raise AssertionError("legacy pseudotime markers should be reused")

    monkeypatch.setattr(
        marker_algorithms,
        "find_markers_by_regression",
        fail_if_recomputed,
    )
    reused = datastore_ephemeral.run_pseudotime_marker_search(**arguments)

    reused_ref = ArtifactRef.from_dict(
        datastore_ephemeral.RNA.z["featureData"][reused.correlation_key].attrs[
            "source_artifact"
        ]
    )
    published = datastore_ephemeral.zw[artifact_path(reused_ref)]
    assert reused_ref != ref
    assert set(artifact.array_keys()) == arrays_before
    assert "p_value_adjusted" not in artifact
    assert dict(artifact.attrs) == attrs_before
    np.testing.assert_array_equal(artifact["r_value"][:], raw_r_values)
    np.testing.assert_array_equal(artifact["p_value"][:], raw_p_values)
    assert set(published.array_keys()) == {
        "p_value",
        "p_value_adjusted",
        "r_value",
    }
    np.testing.assert_allclose(
        reused.table["p_value_adjusted"].to_numpy(),
        _bh_adjusted_pvalues(raw_p_values),
        equal_nan=True,
    )
    adjusted_column = datastore_ephemeral.RNA.z["featureData"][
        first.p_value_adjusted_key
    ]
    adjusted_ref = ArtifactRef.from_dict(adjusted_column.attrs["source_artifact"])
    assert adjusted_ref == reused_ref
    assert adjusted_column.attrs["source_value"] == "p_value_adjusted"
    assert "p_value_adjusted" in published
    status = datastore_ephemeral.inspect_artifact(reused_ref)
    assert status.parameters["association_method"] == "pearson"
    assert status.parameters["p_value_method"] == "student_t"
    assert status.parameters["adjustment_method"] == "fdr_bh"
    assert status.parameters["adjustment_scope"] == "tested_features"
    assert "algorithm_version" not in status.parameters
    assert "schema_version" not in status.parameters
    assert "correction_method" not in status.parameters


def test_read_only_legacy_pseudotime_marker_reuse_is_non_mutating(
    datastore_ephemeral,
    monkeypatch,
):
    import scarf.features.markers as marker_algorithms
    from scarf.datastore.datastore import DataStore
    from scarf.features.markers.correction import _bh_adjusted_pvalues

    (
        arguments,
        first,
        _ref,
        artifact,
        raw_r_values,
        raw_p_values,
    ) = _prepare_legacy_pseudotime_marker_artifact(datastore_ephemeral)
    attrs_before = dict(artifact.attrs)
    arrays_before = {
        name: np.asarray(artifact[name][:]).copy() for name in artifact.array_keys()
    }
    feature_columns_before = set(datastore_ephemeral.RNA.z["featureData"].array_keys())

    def fail_if_recomputed(*_args, **_kwargs):
        raise AssertionError("legacy pseudotime markers should be reused")

    monkeypatch.setattr(
        marker_algorithms,
        "find_markers_by_regression",
        fail_if_recomputed,
    )
    read_only = DataStore(
        datastore_ephemeral.zarr_loc,
        default_assay="RNA",
        zarr_mode="r",
    )
    reused = read_only.run_pseudotime_marker_search(**arguments)

    np.testing.assert_allclose(
        reused.table["p_value_adjusted"].to_numpy(),
        _bh_adjusted_pvalues(raw_p_values),
        equal_nan=True,
    )
    assert first.p_value_adjusted_key not in read_only.RNA.feats.columns
    assert set(read_only.RNA.z["featureData"].array_keys()) == feature_columns_before
    assert dict(artifact.attrs) == attrs_before
    assert set(artifact.array_keys()) == set(arrays_before)
    for name, expected in arrays_before.items():
        np.testing.assert_array_equal(artifact[name][:], expected)
    np.testing.assert_array_equal(artifact["r_value"][:], raw_r_values)


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
            "I",
            "I",
            "ptime",
            min_exp=0.0,
            window_size=window_size,
            chunk_size=chunk_size,
            smoothen=False,
            z_scale=False,
            norm_params={},
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
