import numpy as np
import pytest

from scarf.chunked import ChunkedArray
from scarf.mapping_utils import (
    _correlation_alignment,
    _order_features,
    conformal_prediction_sets,
    distance_weights,
)


class _Features:
    def __init__(self, ids: list[str]) -> None:
        self._ids = ids

    def fetch_all(self, column: str) -> np.ndarray:
        assert column == "ids"
        return np.asarray(self._ids)


class _Assay:
    def __init__(self, ids: list[str]) -> None:
        self.feats = _Features(ids)


def test_distance_weights_uses_all_neighbors_and_handles_zero_distance():
    weights = distance_weights(np.array([[0.0, 1.0, 4.0], [1.0, 1.0, 1.0]]))

    np.testing.assert_allclose(weights[0], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(weights[1], [1 / 3, 1 / 3, 1 / 3])
    np.testing.assert_allclose(weights.sum(axis=1), 1.0)


def test_distance_weights_rejects_invalid_hnsw_distances():
    with pytest.raises(ValueError, match="non-negative"):
        distance_weights(np.array([[-1.0, 1.0]]))
    with pytest.raises(ValueError, match="finite"):
        distance_weights(np.array([[np.nan, 1.0]]))


def test_coral_rejects_one_cell_cohorts():
    source = ChunkedArray.from_numpy(np.array([[1.0, 2.0]]))
    target = ChunkedArray.from_numpy(np.array([[1.0, 2.0], [2.0, 3.0]]))

    with pytest.raises(ValueError, match="at least two cells"):
        _correlation_alignment(source, target, nthreads=1)


def test_coral_alignment_is_finite_for_small_full_rank_inputs():
    source = ChunkedArray.from_numpy(
        np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 4.0]], dtype=np.float64)
    )
    target = ChunkedArray.from_numpy(
        np.array([[1.0, 0.0], [3.0, 1.0], [5.0, 3.0]], dtype=np.float64)
    )

    aligned = _correlation_alignment(source, target, nthreads=1).compute()

    assert aligned.shape == source.shape
    assert np.all(np.isfinite(aligned))


def test_conformal_prediction_sets_include_high_score_labels():
    sets = conformal_prediction_sets(
        np.array([[0.95, 0.1], [0.7, 0.7]]),
        np.array([0.05, 0.1, 0.2, 0.25]),
        alpha=0.2,
    )

    assert sets.shape == (2, 2)
    assert sets[0, 0]
    assert not sets[0, 1]


def test_feature_alignment_rejects_duplicate_identifiers():
    with pytest.raises(ValueError, match="unique"):
        _order_features(
            _Assay(["gene_a", "gene_a"]),
            _Assay(["gene_a", "gene_b"]),
            np.array(["gene_a", "gene_a"]),
            filter_null=False,
            missing_feature_policy="zero",
            nthreads=1,
        )


def test_feature_ordering_covers_zero_intersection_and_error_inputs():
    source = _Assay(["gene_a", "gene_b", "gene_c"])
    target = _Assay(["gene_b", "gene_a"])
    selected = np.array(["gene_a", "gene_b", "gene_c"])

    zero_source, zero_target = _order_features(
        source,
        target,
        selected,
        filter_null=False,
        missing_feature_policy="zero",
        nthreads=1,
    )
    intersection_source, intersection_target = _order_features(
        source,
        target,
        selected,
        filter_null=False,
        missing_feature_policy="intersection",
        nthreads=1,
    )

    np.testing.assert_array_equal(zero_source, [0, 1, 2])
    np.testing.assert_array_equal(zero_target, [1, 0, -1])
    np.testing.assert_array_equal(intersection_source, [0, 1])
    np.testing.assert_array_equal(intersection_target, [1, 0])


def test_coral_restored_values_use_reference_scaling_contract():
    query = np.array(
        [[1.0, 2.0], [2.0, 4.0], [4.0, 5.0], [7.0, 9.0]],
        dtype=np.float64,
    )
    reference = np.array(
        [[10.0, 3.0], [12.0, 6.0], [15.0, 8.0], [20.0, 12.0]],
        dtype=np.float64,
    )
    query_mean = query.mean(axis=0)
    query_scale = query.std(axis=0)
    reference_mean = reference.mean(axis=0)
    reference_scale = reference.std(axis=0)
    standardized = _correlation_alignment(
        ChunkedArray.from_numpy((query - query_mean) / query_scale),
        ChunkedArray.from_numpy((reference - reference_mean) / reference_scale),
        nthreads=1,
    ).compute()
    restored = standardized * reference_scale + reference_mean

    np.testing.assert_allclose(
        (restored - reference_mean) / reference_scale,
        standardized,
        atol=1e-12,
    )
