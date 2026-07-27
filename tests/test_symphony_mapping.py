import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from scarf.embeddings.harmony import fit_harmony
from scarf.mapping.models import SymphonyReferenceModel
from scarf.mapping.symphony import (
    accumulate_sufficient_statistics,
    apply_query_correction,
    initialize_sufficient_statistics,
    project_pca,
    soft_cluster_assignments,
    solve_query_correction,
    weighted_centroids,
    zero_norm_rows,
)


def _single_cluster_reference() -> SymphonyReferenceModel:
    return SymphonyReferenceModel(
        feature_means=np.zeros(2),
        feature_scales=np.ones(2),
        loadings=np.eye(2),
        centroids=np.array([[1.0, 0.0]]),
        raw_centroids=np.zeros((1, 2)),
        corrected_centroids=np.zeros((1, 2)),
        cluster_mass=np.array([4.0]),
        sigma=np.array([0.1]),
        correction_ridge=0.0,
    )


def _correct(
    model: SymphonyReferenceModel, query: np.ndarray, batch_codes: np.ndarray
) -> np.ndarray:
    projected = project_pca(query, model)
    assignments = soft_cluster_assignments(projected, model)
    counts, sums = initialize_sufficient_statistics(batch_codes.max() + 1, model)
    accumulate_sufficient_statistics(counts, sums, projected, assignments, batch_codes)
    correction = solve_query_correction(counts, sums, model)
    return apply_query_correction(
        projected, assignments, batch_codes, model, correction
    )


def test_symphony_closed_form_affine_shift_fixture():
    model = _single_cluster_reference()
    unshifted = np.array([[-1.0, 0.0], [1.0, 0.0]])
    shifted = unshifted + np.array([3.0, -2.0])

    corrected = _correct(model, shifted, np.zeros(2, dtype=np.int64))

    np.testing.assert_allclose(
        corrected,
        np.array([[2 / 7, -6 / 7], [16 / 7, -6 / 7]]),
        atol=1e-12,
    )


def test_symphony_no_shift_control_is_identity_for_centered_query():
    model = _single_cluster_reference()
    query = np.array([[-2.0, 1.0], [2.0, -1.0]])

    corrected = _correct(model, query, np.zeros(2, dtype=np.int64))

    np.testing.assert_allclose(corrected, query, atol=1e-12)


def test_symphony_handles_empty_cluster_batch_statistics_without_nan():
    model = _single_cluster_reference()
    counts, sums = initialize_sufficient_statistics(2, model)
    counts[0, 0] = 2.0
    sums[0, 0] = np.array([4.0, -2.0])

    correction = solve_query_correction(counts, sums, model)

    assert np.all(np.isfinite(correction.batch_offsets))
    np.testing.assert_array_equal(correction.batch_offsets[1], np.zeros((1, 2)))


def test_symphony_no_shift_composition_subset_stays_stable():
    model = SymphonyReferenceModel(
        feature_means=np.zeros(2),
        feature_scales=np.ones(2),
        loadings=np.eye(2),
        centroids=np.array([[-2.0, 0.0], [2.0, 0.0]]),
        raw_centroids=np.array([[-2.0, 0.0], [2.0, 0.0]]),
        corrected_centroids=np.array([[-2.0, 0.0], [2.0, 0.0]]),
        cluster_mass=np.array([100.0, 100.0]),
        sigma=np.array([0.1, 0.1]),
        correction_ridge=0.0,
    )
    cluster_zero_only = np.array([[-3.0, 0.0], [-1.0, 0.0]])

    corrected = _correct(model, cluster_zero_only, np.zeros(2, dtype=np.int64))

    np.testing.assert_allclose(corrected, cluster_zero_only, atol=1e-12)


def test_symphony_composition_imbalance_preserves_distinct_populations():
    model = SymphonyReferenceModel(
        feature_means=np.zeros(2),
        feature_scales=np.ones(2),
        loadings=np.eye(2),
        centroids=np.array([[-1.0, 0.0], [1.0, 0.0]]),
        raw_centroids=np.array([[-2.0, 0.0], [2.0, 0.0]]),
        corrected_centroids=np.array([[-2.0, 0.0], [2.0, 0.0]]),
        cluster_mass=np.array([100.0, 100.0]),
        sigma=np.array([0.01, 0.01]),
        correction_ridge=0.0,
    )
    query = np.array(
        [
            [-3.0, 0.0],
            [-2.5, 0.0],
            [-1.5, 0.0],
            [-1.0, 0.0],
            [1.5, 0.0],
            [2.5, 0.0],
        ]
    )
    populations = np.array([0, 0, 0, 0, 1, 1])

    corrected = _correct(model, query, np.zeros(len(query), dtype=np.int64))
    separation = corrected[populations == 1].mean(axis=0) - corrected[
        populations == 0
    ].mean(axis=0)

    np.testing.assert_allclose(corrected, query, atol=1e-12)
    assert separation[0] == 4.0


def test_symphony_joint_ridge_handles_independent_query_batches():
    model = _single_cluster_reference()
    centered = np.array([[-1.0, 0.0], [1.0, 0.0], [-2.0, 1.0], [2.0, -1.0]])
    batches = np.array([0, 0, 1, 1], dtype=np.int64)
    shifted = centered.copy()
    shifted[batches == 0] += np.array([3.0, -2.0])
    shifted[batches == 1] += np.array([-4.0, 5.0])

    corrected = _correct(model, shifted, batches)

    np.testing.assert_allclose(
        corrected,
        np.array(
            [
                [-1 / 12, -5 / 12],
                [23 / 12, -5 / 12],
                [-41 / 12, 35 / 12],
                [7 / 12, 11 / 12],
            ]
        ),
        atol=1e-12,
    )


def test_symphony_joint_ridge_shrinks_query_shift():
    model = _single_cluster_reference()
    reference = np.array([[-2.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    shifted = reference + np.array([8.0, -3.0])
    corrected = _correct(model, shifted, np.zeros(4, dtype=np.int64))

    assert np.linalg.norm(corrected.mean(axis=0)) < np.linalg.norm(shifted.mean(axis=0))


def test_symphony_correction_is_row_order_invariant():
    model = _single_cluster_reference()
    query = np.array([[2.0, -1.0], [4.0, -1.0], [-3.0, 2.0], [-1.0, 2.0]])
    batches = np.array([0, 0, 1, 1], dtype=np.int64)
    order = np.array([2, 0, 3, 1])

    expected = _correct(model, query, batches)
    reordered = _correct(model, query[order], batches[order])
    restored = np.empty_like(reordered)
    restored[order] = reordered

    np.testing.assert_allclose(restored, expected, atol=1e-12)


def test_symphony_sufficient_statistics_are_chunk_invariant():
    model = _single_cluster_reference()
    query = np.array([[2.0, -1.0], [4.0, -1.0], [-3.0, 2.0], [-1.0, 2.0]])
    batches = np.array([0, 0, 1, 1], dtype=np.int64)
    coordinates = project_pca(query, model)
    assignments = soft_cluster_assignments(coordinates, model)

    expected_counts, expected_sums = initialize_sufficient_statistics(2, model)
    accumulate_sufficient_statistics(
        expected_counts,
        expected_sums,
        coordinates,
        assignments,
        batches,
    )
    chunked_counts, chunked_sums = initialize_sufficient_statistics(2, model)
    for start, stop in ((0, 1), (1, 3), (3, 4)):
        accumulate_sufficient_statistics(
            chunked_counts,
            chunked_sums,
            coordinates[start:stop],
            assignments[start:stop],
            batches[start:stop],
        )

    np.testing.assert_allclose(chunked_counts, expected_counts, atol=1e-12)
    np.testing.assert_allclose(chunked_sums, expected_sums, atol=1e-12)


def test_symphony_query_ridge_matches_upstream_fixed_penalty():
    unregularized = _single_cluster_reference()
    regularized = SymphonyReferenceModel(
        feature_means=unregularized.feature_means,
        feature_scales=unregularized.feature_scales,
        loadings=unregularized.loadings,
        centroids=unregularized.centroids,
        raw_centroids=unregularized.raw_centroids,
        corrected_centroids=unregularized.corrected_centroids,
        cluster_mass=unregularized.cluster_mass,
        sigma=unregularized.sigma,
        correction_ridge=10.0,
    )
    counts = np.array([[1.0]])
    sums = np.array([[[5.0, 0.0]]])

    raw = solve_query_correction(counts, sums, unregularized)
    shrunk = solve_query_correction(counts, sums, regularized)

    np.testing.assert_array_equal(shrunk.batch_offsets, raw.batch_offsets)


def test_zero_norm_rows_are_identified_without_nonfinite_assignments():
    model = _single_cluster_reference()
    coordinates = np.array([[0.0, 0.0], [1.0, 0.0]])

    mask = zero_norm_rows(coordinates)
    assignments = soft_cluster_assignments(coordinates, model)

    assert mask.tolist() == [True, False]
    assert np.all(np.isfinite(assignments))


def test_harmony_accepts_numpy_scalar_sigma():
    embedding = np.array([[-1.0, -0.8, 0.8, 1.0], [0.0, 0.2, -0.2, 0.0]])
    metadata = pd.DataFrame({"batch": ["a", "a", "b", "b"]})

    result = fit_harmony(
        embedding,
        metadata,
        sigma=np.float64(0.1),
        theta=np.float64(2.0),
        lamb=np.int64(3),
        nclust=2,
        max_iter_harmony=2,
        max_iter_kmeans=2,
    )

    assert result.sigma.shape == (2,)
    assert result.parameters["sigma"] == [0.1, 0.1]
    assert result.parameters["theta"] == [2.0, 2.0]
    assert result.parameters["lambda"] == [3.0, 3.0]
    assert result.parameters["clusterBackend"] == "sklearn.cluster.KMeans"


def test_weighted_centroids_reject_empty_reference_clusters():
    coordinates = np.array([[0.0, 0.0], [1.0, 1.0]])
    assignments = np.array([[1.0, 1.0], [0.0, 0.0]])

    with np.testing.assert_raises_regex(ValueError, "empty cluster"):
        weighted_centroids(coordinates, assignments)
    with np.testing.assert_raises_regex(ValueError, "masses must be positive"):
        replace(_single_cluster_reference(), cluster_mass=np.array([0.0]))


def test_symphony_r_0_1_3_static_golden_fixture():
    fixture = json.loads(
        (Path(__file__).parent / "symphony_r_0_1_3_golden.json").read_text()
    )
    reference = fixture["reference"]
    model = SymphonyReferenceModel(
        feature_means=np.asarray(reference["featureMeans"]),
        feature_scales=np.asarray(reference["featureScales"]),
        loadings=np.asarray(reference["loadings"]),
        centroids=np.asarray(reference["centroids"]),
        raw_centroids=np.asarray(reference["rawCentroids"]),
        corrected_centroids=np.asarray(reference["correctedCentroids"]),
        cluster_mass=np.asarray(reference["clusterMass"]),
        sigma=np.asarray(reference["sigma"]),
        correction_ridge=float(reference["correctionRidge"]),
    )
    query = np.asarray(fixture["query"])
    batch_codes = np.asarray(fixture["batchCodes"], dtype=np.int64)
    projected = project_pca(query, model)
    assignments = soft_cluster_assignments(projected, model)
    corrected = _correct(model, query, batch_codes)

    assert fixture["provenance"]["packageVersion"] == "0.1.3"
    np.testing.assert_allclose(
        projected,
        fixture["expectedProjected"],
        rtol=0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        assignments,
        fixture["expectedAssignments"],
        rtol=1e-12,
        atol=1e-18,
    )
    np.testing.assert_allclose(
        corrected,
        fixture["expectedCorrected"],
        rtol=0,
        atol=1e-12,
    )

    nonzero = fixture["nonzeroCorrection"]
    nonzero_reference = nonzero["reference"]
    nonzero_model = SymphonyReferenceModel(
        feature_means=np.asarray(nonzero_reference["featureMeans"]),
        feature_scales=np.asarray(nonzero_reference["featureScales"]),
        loadings=np.asarray(nonzero_reference["loadings"]),
        centroids=np.asarray(nonzero_reference["centroids"]),
        raw_centroids=np.asarray(nonzero_reference["rawCentroids"]),
        corrected_centroids=np.asarray(nonzero_reference["correctedCentroids"]),
        cluster_mass=np.asarray(nonzero_reference["clusterMass"]),
        sigma=np.asarray(nonzero_reference["sigma"]),
        correction_ridge=float(nonzero_reference["correctionRidge"]),
    )
    nonzero_query = np.asarray(nonzero["query"])
    nonzero_batches = np.asarray(nonzero["batchCodes"], dtype=np.int64)
    nonzero_projected = project_pca(nonzero_query, nonzero_model)
    nonzero_assignments = soft_cluster_assignments(nonzero_projected, nonzero_model)
    nonzero_corrected = _correct(nonzero_model, nonzero_query, nonzero_batches)

    np.testing.assert_allclose(
        nonzero_assignments,
        nonzero["expectedAssignments"],
        rtol=0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        nonzero_corrected,
        nonzero["expectedCorrected"],
        rtol=0,
        atol=1e-12,
    )
    assert not np.allclose(nonzero_corrected, nonzero_query)
