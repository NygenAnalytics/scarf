import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scarf.embeddings.harmony import fit_harmony
from scarf.mapping.models import (
    MappingResult,
    QueryCorrection,
    ScaledPCAProjectionModel,
    SymphonyCorrectionModel,
)
from scarf.mapping.symphony import (
    accumulate_sufficient_statistics,
    apply_query_correction,
    initialize_sufficient_statistics,
    project_pca,
    scaled_dispersion_sum,
    soft_cluster_assignments,
    solve_query_correction,
    weighted_centroids,
    zero_norm_rows,
)
from scarf.storage import ArtifactRef


def _single_cluster_reference() -> tuple[
    ScaledPCAProjectionModel,
    SymphonyCorrectionModel,
]:
    return (
        ScaledPCAProjectionModel(
            feature_means=np.zeros(2),
            feature_scales=np.ones(2),
            loadings=np.eye(2),
        ),
        SymphonyCorrectionModel(
            centroids=np.array([[1.0, 0.0]]),
            raw_centroids=np.zeros((1, 2)),
            corrected_centroids=np.zeros((1, 2)),
            cluster_mass=np.array([4.0]),
            sigma=np.array([0.1]),
        ),
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"loadings": np.ones(2)}, "two-dimensional"),
        ({"feature_means": np.ones(3)}, "means have incompatible"),
        ({"feature_scales": np.ones(3)}, "scales have incompatible"),
        ({"feature_scales": np.array([1.0, 0.0])}, "scales must be positive"),
        ({"feature_means": np.array([0.0, np.nan])}, "non-finite"),
    ],
)
def test_scaled_pca_projection_model_rejects_invalid_payloads(overrides, message):
    values = {
        "feature_means": np.zeros(2),
        "feature_scales": np.ones(2),
        "loadings": np.eye(2),
    }
    values.update(overrides)

    with pytest.raises(ValueError, match=message):
        ScaledPCAProjectionModel(**values)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"centroids": np.ones(2)}, "centroids must be two-dimensional"),
        ({"raw_centroids": np.ones((2, 2))}, "raw centroids have incompatible"),
        (
            {"corrected_centroids": np.ones((2, 2))},
            "corrected centroids have incompatible",
        ),
        ({"cluster_mass": np.array([0.0])}, "cluster masses must be positive"),
        ({"sigma": np.array([0.0])}, "kernel widths must be positive"),
        ({"raw_centroids": np.array([[np.nan, 0.0]])}, "non-finite"),
    ],
)
def test_symphony_correction_model_rejects_invalid_payloads(overrides, message):
    values = {
        "centroids": np.array([[1.0, 0.0]]),
        "raw_centroids": np.zeros((1, 2)),
        "corrected_centroids": np.zeros((1, 2)),
        "cluster_mass": np.ones(1),
        "sigma": np.ones(1),
    }
    values.update(overrides)

    with pytest.raises(ValueError, match=message):
        SymphonyCorrectionModel(**values)


@pytest.mark.parametrize(
    ("offsets", "counts", "message"),
    [
        (np.zeros((1, 2)), np.zeros((1, 2)), "batch, cluster, and dimension"),
        (np.zeros((1, 1, 2)), np.zeros((2, 1)), "counts must match"),
        (
            np.array([[[np.nan, 0.0]]]),
            np.zeros((1, 1)),
            "offsets contain non-finite",
        ),
    ],
)
def test_query_correction_rejects_invalid_payloads(offsets, counts, message):
    with pytest.raises(ValueError, match=message):
        QueryCorrection(offsets, counts)


def test_mapping_result_requires_loaded_axes_for_selection_properties():
    result = MappingResult(
        ref=ArtifactRef(
            scope="assay",
            assay="RNA",
            kind="projection",
            artifact_id="a" * 64,
        ),
        n_cells=2,
        correction_method="none",
        diagnostics={},
        reference=object(),
    )

    with pytest.raises(RuntimeError, match="DataStore.get_mapping_result"):
        result.cell_selection
    with pytest.raises(RuntimeError, match="DataStore.get_mapping_result"):
        result.feature_selection


def test_symphony_primitives_reject_misaligned_and_nonfinite_inputs():
    projection, model = _single_cluster_reference()
    counts, sums = initialize_sufficient_statistics(1, model)
    correction = QueryCorrection(np.zeros((1, 1, 2)), np.zeros((1, 1)))

    with pytest.raises(ValueError, match="Expected query matrix"):
        project_pca(np.ones((2, 1)), projection)
    with pytest.raises(ValueError, match="non-finite"):
        project_pca(np.array([[np.nan, 0.0]]), projection)
    with pytest.raises(ValueError, match="incompatible dimensions"):
        soft_cluster_assignments(np.ones((2, 1)), model)
    with np.errstate(invalid="ignore"):
        with pytest.raises(ValueError, match="non-finite"):
            soft_cluster_assignments(np.array([[np.inf, 0.0]]), model)
    with pytest.raises(ValueError, match="two-dimensional"):
        zero_norm_rows(np.ones(2))
    with pytest.raises(ValueError, match="At least one query batch"):
        initialize_sufficient_statistics(0, model)

    with pytest.raises(ValueError, match="batch codes must match"):
        accumulate_sufficient_statistics(
            counts, sums, np.zeros((2, 2)), np.ones((1, 1)), np.array([0])
        )
    with pytest.raises(ValueError, match="Assignment count"):
        accumulate_sufficient_statistics(
            counts, sums, np.zeros((1, 2)), np.ones((1, 2)), np.array([0])
        )
    with pytest.raises(ValueError, match="out of range"):
        accumulate_sufficient_statistics(
            counts, sums, np.zeros((1, 2)), np.ones((1, 1)), np.array([1])
        )
    with pytest.raises(ValueError, match="count statistics"):
        solve_query_correction(np.zeros(1), sums, model)
    with pytest.raises(ValueError, match="sum statistics"):
        solve_query_correction(counts, np.zeros((1, 1, 1)), model)

    with pytest.raises(ValueError, match="rows must agree"):
        apply_query_correction(
            np.zeros((2, 2)), np.ones((1, 1)), np.array([0]), model, correction
        )
    with pytest.raises(ValueError, match="cluster count"):
        apply_query_correction(
            np.zeros((1, 2)), np.ones((1, 2)), np.array([0]), model, correction
        )
    incompatible = QueryCorrection(np.zeros((1, 1, 3)), np.zeros((1, 1)))
    with pytest.raises(ValueError, match="reference dimensions"):
        apply_query_correction(
            np.zeros((1, 2)), np.ones((1, 1)), np.array([0]), model, incompatible
        )
    with pytest.raises(ValueError, match="non-finite coordinates"):
        apply_query_correction(
            np.array([[np.nan, 0.0]]),
            np.ones((1, 1)),
            np.array([0]),
            model,
            correction,
        )
    with pytest.raises(ValueError, match="dimensions do not agree"):
        weighted_centroids(np.zeros((2, 2)), np.ones((2, 3)))


def test_scaled_dispersion_reads_one_for_data_matching_the_reference():
    rng = np.random.default_rng(0)
    reference = rng.normal(size=(400, 6))
    model = ScaledPCAProjectionModel(
        feature_means=reference.mean(axis=0),
        feature_scales=reference.std(axis=0),
        loadings=np.eye(6),
    )
    n_features = reference.shape[1]

    def dispersion(values: np.ndarray) -> float:
        return scaled_dispersion_sum(values, model) / (len(values) * n_features)

    # The reference is z-scored by construction, so it must read exactly 1.
    assert dispersion(reference) == pytest.approx(1.0)
    # A query holding a quarter of the reference spread reads a sixteenth, since
    # the statistic is a variance rather than a standard deviation.
    compressed = model.feature_means + 0.25 * (reference - model.feature_means)
    assert dispersion(compressed) == pytest.approx(0.0625)
    widened = model.feature_means + 2.0 * (reference - model.feature_means)
    assert dispersion(widened) == pytest.approx(4.0)
    # An offset query is not narrow, so a pure shift must not look compressed.
    assert dispersion(reference + 3.0 * model.feature_scales) == pytest.approx(10.0)

    with pytest.raises(ValueError, match="Expected query matrix"):
        scaled_dispersion_sum(reference[:, :2], model)


def _correct(
    projection: ScaledPCAProjectionModel,
    correction_model: SymphonyCorrectionModel,
    query: np.ndarray,
    batch_codes: np.ndarray,
) -> np.ndarray:
    projected = project_pca(query, projection)
    assignments = soft_cluster_assignments(projected, correction_model)
    counts, sums = initialize_sufficient_statistics(
        batch_codes.max() + 1,
        correction_model,
    )
    accumulate_sufficient_statistics(counts, sums, projected, assignments, batch_codes)
    correction = solve_query_correction(counts, sums, correction_model)
    return apply_query_correction(
        projected,
        assignments,
        batch_codes,
        correction_model,
        correction,
    )


def test_symphony_closed_form_affine_shift_fixture():
    projection, correction_model = _single_cluster_reference()
    unshifted = np.array([[-1.0, 0.0], [1.0, 0.0]])
    shifted = unshifted + np.array([3.0, -2.0])

    corrected = _correct(
        projection,
        correction_model,
        shifted,
        np.zeros(2, dtype=np.int64),
    )

    np.testing.assert_allclose(
        corrected,
        np.array([[2 / 7, -6 / 7], [16 / 7, -6 / 7]]),
        atol=1e-12,
    )


def test_symphony_no_shift_control_is_identity_for_centered_query():
    projection, correction_model = _single_cluster_reference()
    query = np.array([[-2.0, 1.0], [2.0, -1.0]])

    corrected = _correct(
        projection,
        correction_model,
        query,
        np.zeros(2, dtype=np.int64),
    )

    np.testing.assert_allclose(corrected, query, atol=1e-12)


def test_symphony_handles_empty_cluster_batch_statistics_without_nan():
    _, correction_model = _single_cluster_reference()
    counts, sums = initialize_sufficient_statistics(2, correction_model)
    counts[0, 0] = 2.0
    sums[0, 0] = np.array([4.0, -2.0])

    correction = solve_query_correction(counts, sums, correction_model)

    assert np.all(np.isfinite(correction.batch_offsets))
    np.testing.assert_array_equal(correction.batch_offsets[1], np.zeros((1, 2)))


def test_symphony_no_shift_composition_subset_stays_stable():
    projection = ScaledPCAProjectionModel(
        feature_means=np.zeros(2),
        feature_scales=np.ones(2),
        loadings=np.eye(2),
    )
    correction_model = SymphonyCorrectionModel(
        centroids=np.array([[-2.0, 0.0], [2.0, 0.0]]),
        raw_centroids=np.array([[-2.0, 0.0], [2.0, 0.0]]),
        corrected_centroids=np.array([[-2.0, 0.0], [2.0, 0.0]]),
        cluster_mass=np.array([100.0, 100.0]),
        sigma=np.array([0.1, 0.1]),
    )
    cluster_zero_only = np.array([[-3.0, 0.0], [-1.0, 0.0]])

    corrected = _correct(
        projection,
        correction_model,
        cluster_zero_only,
        np.zeros(2, dtype=np.int64),
    )

    np.testing.assert_allclose(corrected, cluster_zero_only, atol=1e-12)


def test_symphony_composition_imbalance_preserves_distinct_populations():
    projection = ScaledPCAProjectionModel(
        feature_means=np.zeros(2),
        feature_scales=np.ones(2),
        loadings=np.eye(2),
    )
    correction_model = SymphonyCorrectionModel(
        centroids=np.array([[-1.0, 0.0], [1.0, 0.0]]),
        raw_centroids=np.array([[-2.0, 0.0], [2.0, 0.0]]),
        corrected_centroids=np.array([[-2.0, 0.0], [2.0, 0.0]]),
        cluster_mass=np.array([100.0, 100.0]),
        sigma=np.array([0.01, 0.01]),
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

    corrected = _correct(
        projection,
        correction_model,
        query,
        np.zeros(len(query), dtype=np.int64),
    )
    separation = corrected[populations == 1].mean(axis=0) - corrected[
        populations == 0
    ].mean(axis=0)

    np.testing.assert_allclose(corrected, query, atol=1e-12)
    assert separation[0] == 4.0


def test_symphony_joint_ridge_handles_independent_query_batches():
    projection, correction_model = _single_cluster_reference()
    centered = np.array([[-1.0, 0.0], [1.0, 0.0], [-2.0, 1.0], [2.0, -1.0]])
    batches = np.array([0, 0, 1, 1], dtype=np.int64)
    shifted = centered.copy()
    shifted[batches == 0] += np.array([3.0, -2.0])
    shifted[batches == 1] += np.array([-4.0, 5.0])

    corrected = _correct(projection, correction_model, shifted, batches)

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
    projection, correction_model = _single_cluster_reference()
    reference = np.array([[-2.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    shifted = reference + np.array([8.0, -3.0])
    corrected = _correct(
        projection,
        correction_model,
        shifted,
        np.zeros(4, dtype=np.int64),
    )

    assert np.linalg.norm(corrected.mean(axis=0)) < np.linalg.norm(shifted.mean(axis=0))


def test_symphony_correction_is_row_order_invariant():
    projection, correction_model = _single_cluster_reference()
    query = np.array([[2.0, -1.0], [4.0, -1.0], [-3.0, 2.0], [-1.0, 2.0]])
    batches = np.array([0, 0, 1, 1], dtype=np.int64)
    order = np.array([2, 0, 3, 1])

    expected = _correct(projection, correction_model, query, batches)
    reordered = _correct(
        projection,
        correction_model,
        query[order],
        batches[order],
    )
    restored = np.empty_like(reordered)
    restored[order] = reordered

    np.testing.assert_allclose(restored, expected, atol=1e-12)


def test_symphony_sufficient_statistics_are_chunk_invariant():
    projection, correction_model = _single_cluster_reference()
    query = np.array([[2.0, -1.0], [4.0, -1.0], [-3.0, 2.0], [-1.0, 2.0]])
    batches = np.array([0, 0, 1, 1], dtype=np.int64)
    coordinates = project_pca(query, projection)
    assignments = soft_cluster_assignments(coordinates, correction_model)

    expected_counts, expected_sums = initialize_sufficient_statistics(
        2,
        correction_model,
    )
    accumulate_sufficient_statistics(
        expected_counts,
        expected_sums,
        coordinates,
        assignments,
        batches,
    )
    chunked_counts, chunked_sums = initialize_sufficient_statistics(
        2,
        correction_model,
    )
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


def test_symphony_query_correction_applies_the_upstream_unit_penalty():
    _, correction_model = _single_cluster_reference()
    counts = np.array([[1.0]])
    sums = np.array([[[5.0, 0.0]]])

    correction = solve_query_correction(counts, sums, correction_model)

    # One cluster of mass 4 centred on the origin and one query batch holding a
    # single cell at (5, 0). Upstream Symphony penalises each batch term by a
    # fixed unit ridge, so the design cross-product is [[5, 1], [1, 2]] and the
    # batch offset solves to 20 / 9. Any other penalty moves this number.
    np.testing.assert_allclose(
        correction.batch_offsets,
        np.array([[[20.0 / 9.0, 0.0]]]),
        rtol=0,
        atol=1e-12,
    )


def test_zero_norm_rows_are_identified_without_nonfinite_assignments():
    _, correction_model = _single_cluster_reference()
    coordinates = np.array([[0.0, 0.0], [1.0, 0.0]])

    mask = zero_norm_rows(coordinates)
    assignments = soft_cluster_assignments(coordinates, correction_model)

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
        _, correction_model = _single_cluster_reference()
        replace(correction_model, cluster_mass=np.array([0.0]))


def test_symphony_r_0_1_3_static_golden_fixture():
    fixture = json.loads(
        (Path(__file__).parent / "symphony_r_0_1_3_golden.json").read_text()
    )
    reference = fixture["reference"]
    projection = ScaledPCAProjectionModel(
        feature_means=np.asarray(reference["featureMeans"]),
        feature_scales=np.asarray(reference["featureScales"]),
        loadings=np.asarray(reference["loadings"]),
    )
    correction_model = SymphonyCorrectionModel(
        centroids=np.asarray(reference["centroids"]),
        raw_centroids=np.asarray(reference["rawCentroids"]),
        corrected_centroids=np.asarray(reference["correctedCentroids"]),
        cluster_mass=np.asarray(reference["clusterMass"]),
        sigma=np.asarray(reference["sigma"]),
    )
    query = np.asarray(fixture["query"])
    batch_codes = np.asarray(fixture["batchCodes"], dtype=np.int64)
    projected = project_pca(query, projection)
    assignments = soft_cluster_assignments(projected, correction_model)
    corrected = _correct(projection, correction_model, query, batch_codes)

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
    nonzero_projection = ScaledPCAProjectionModel(
        feature_means=np.asarray(nonzero_reference["featureMeans"]),
        feature_scales=np.asarray(nonzero_reference["featureScales"]),
        loadings=np.asarray(nonzero_reference["loadings"]),
    )
    nonzero_correction_model = SymphonyCorrectionModel(
        centroids=np.asarray(nonzero_reference["centroids"]),
        raw_centroids=np.asarray(nonzero_reference["rawCentroids"]),
        corrected_centroids=np.asarray(nonzero_reference["correctedCentroids"]),
        cluster_mass=np.asarray(nonzero_reference["clusterMass"]),
        sigma=np.asarray(nonzero_reference["sigma"]),
    )
    nonzero_query = np.asarray(nonzero["query"])
    nonzero_batches = np.asarray(nonzero["batchCodes"], dtype=np.int64)
    nonzero_projected = project_pca(nonzero_query, nonzero_projection)
    nonzero_assignments = soft_cluster_assignments(
        nonzero_projected,
        nonzero_correction_model,
    )
    nonzero_corrected = _correct(
        nonzero_projection,
        nonzero_correction_model,
        nonzero_query,
        nonzero_batches,
    )

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
