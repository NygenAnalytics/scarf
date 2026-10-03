import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

from scarf.mapping.confidence import (
    _conformal_membership,
    _distance_quantile_summary,
    _validated_conformal_calibration,
    add_mapping_scores,
    distance_weights,
    finish_mapping_scores,
    mapping_score_weights,
)
from scarf.mapping.features import _feature_ids


def test_mapping_score_kernel_skips_rows_and_scales_each_group():
    scores = np.zeros((3, 4))
    scored_rows = np.zeros(3, dtype=np.int64)
    # The second block row is skipped, so the neighbors cover rows 0 and 2.
    weights = mapping_score_weights(np.array([[0.0, 1.0], [0.0, 0.0]]))
    add_mapping_scores(
        scores,
        scored_rows,
        np.array([[0, 1], [2, 3]]),
        weights,
        skip=np.array([False, True, False]),
        groups=np.array([0, 0, 1]),
    )
    finish_mapping_scores(
        scores,
        scored_rows,
        n_neighbors=2,
        multiplier=1000.0,
        log_transform=False,
    )

    assert scored_rows.tolist() == [1, 1, 0]
    np.testing.assert_allclose(scores[0], [500.0, 500.0 / (np.log(2) + 1), 0, 0])
    np.testing.assert_allclose(scores[1], [0.0, 0.0, 500.0, 500.0])
    np.testing.assert_array_equal(scores[2], 0.0)

    single = np.zeros((1, 2))
    single_rows = np.zeros(1, dtype=np.int64)
    add_mapping_scores(
        single,
        single_rows,
        np.array([[1, 1]]),
        np.ones((1, 2)),
        skip=np.array([True, False]),
    )
    finish_mapping_scores(
        single,
        single_rows,
        n_neighbors=2,
        multiplier=1000,
        log_transform=True,
    )
    np.testing.assert_allclose(single, np.log1p([[0.0, 1000.0]]))
    with pytest.raises(ValueError, match="one row per query row"):
        add_mapping_scores(
            single,
            single_rows,
            np.array([[0, 1]]),
            np.ones((1, 2)),
            skip=np.zeros(2, dtype=bool),
        )


def test_mapping_score_weights_stay_absolute_across_query_cells():
    near = np.full((1, 6), 1.0)
    far = np.full((1, 6), 100.0)

    near_weights = mapping_score_weights(near)
    far_weights = mapping_score_weights(far)

    np.testing.assert_allclose(near_weights, 1.0 / (np.log1p(1.0) + 1.0))
    np.testing.assert_allclose(far_weights, 1.0 / (np.log1p(100.0) + 1.0))
    # A query cell far from the reference must deposit less total weight than one
    # that lands on it. Row normalization would make both sums equal and reduce
    # the mapping score to a neighbor count.
    assert far_weights.sum() < 0.4 * near_weights.sum()
    assert mapping_score_weights(np.zeros((1, 3))).tolist() == [[1.0, 1.0, 1.0]]

    with pytest.raises(ValueError, match="non-negative"):
        mapping_score_weights(np.array([[-1.0, 1.0]]))
    with pytest.raises(ValueError, match="finite"):
        mapping_score_weights(np.array([[np.nan, 1.0]]))
    with pytest.raises(ValueError, match="two-dimensional"):
        mapping_score_weights(np.array([1.0, 2.0]))


def test_distance_weights_use_metric_distances_and_split_zero_ties():
    weights = distance_weights(
        np.array(
            [
                [1.0, 9.0, 9.0],
                [0.0, 0.0, 4.0],
                [1.0, 1.0, 1.0],
            ]
        )
    )

    np.testing.assert_allclose(weights[0], [0.8181818181818182, 1 / 11, 1 / 11])
    np.testing.assert_allclose(weights[1], [0.5, 0.5, 0.0])
    np.testing.assert_allclose(weights[2], [1 / 3, 1 / 3, 1 / 3])
    np.testing.assert_allclose(weights.sum(axis=1), 1.0)


def test_distance_weights_keep_subnormal_positive_rows_finite():
    smallest = np.nextafter(0.0, 1.0)
    next_smallest = np.nextafter(smallest, 1.0)

    weights = distance_weights(
        np.array(
            [
                [1.0, 9.0],
                [smallest, next_smallest],
                [smallest, np.finfo(np.float64).tiny],
            ]
        )
    )

    np.testing.assert_allclose(weights[0], [0.9, 0.1])
    np.testing.assert_allclose(weights[1], [2 / 3, 1 / 3])
    assert np.all(np.isfinite(weights))
    np.testing.assert_allclose(weights.sum(axis=1), 1.0)


def test_distance_weights_reject_invalid_metric_distances():
    with pytest.raises(ValueError, match="non-negative"):
        distance_weights(np.array([[-1.0, 1.0]]))
    with pytest.raises(ValueError, match="finite"):
        distance_weights(np.array([[np.nan, 1.0]]))


def test_distance_quantile_summary_handles_vectors_and_neighbor_matrices():
    first_neighbors = np.array([0.0, 1.0, 4.0, 9.0, 16.0])
    neighbor_matrix = np.column_stack((first_neighbors, first_neighbors + 1))

    vector_summary = _distance_quantile_summary(
        first_neighbors,
        max_samples=3,
        n_quantiles=3,
    )
    matrix_summary = _distance_quantile_summary(
        neighbor_matrix,
        max_samples=3,
        n_quantiles=3,
    )

    np.testing.assert_allclose(vector_summary[0], [0.0, 0.5, 1.0])
    np.testing.assert_allclose(vector_summary[1], [0.0, 4.0, 16.0])
    np.testing.assert_allclose(matrix_summary[0], vector_summary[0])
    np.testing.assert_allclose(matrix_summary[1], vector_summary[1])


def test_distance_quantile_summary_samples_stride_rows_across_chunks():
    # Ten rows read in chunks of four and sampled at most four times keep
    # every third row, counted from the first row rather than per chunk.
    rng = np.random.default_rng(5)
    values = np.sort(rng.uniform(0.0, 5.0, size=(10, 3)), axis=1)
    root = zarr.open_group(store=MemoryStore(), mode="w")
    stored = root.create_array("distances", data=values, chunks=(4, 3))

    quantiles, summary = _distance_quantile_summary(
        stored,
        max_samples=4,
        n_quantiles=11,
    )

    sampled = values[[0, 3, 6, 9], 0]
    np.testing.assert_allclose(quantiles, np.linspace(0.0, 1.0, 4))
    np.testing.assert_allclose(summary, np.quantile(sampled, quantiles))
    # Few rows cap the quantile count at the number of samples.
    capped = _distance_quantile_summary(values[:2], n_quantiles=1_001)
    np.testing.assert_allclose(capped[0], [0.0, 1.0])
    np.testing.assert_allclose(capped[1], np.sort(values[:2, 0]))


@pytest.mark.parametrize(
    ("distances", "options", "message"),
    [
        (np.zeros((2, 2, 2)), {}, "one- or two-dimensional"),
        (np.zeros((0, 3)), {}, "Neighbor distances are empty"),
        (np.zeros(0), {}, "Neighbor distances are empty"),
        (np.zeros((3, 0)), {}, "do not contain any neighbors"),
        (np.ones(3), {"max_samples": 0}, "counts must be positive"),
        (np.ones(3), {"n_quantiles": 0}, "counts must be positive"),
    ],
)
def test_distance_quantile_summary_rejects_unusable_inputs(
    distances,
    options,
    message,
):
    with pytest.raises(ValueError, match=message):
        _distance_quantile_summary(distances, **options)


def _reference_conformal_membership(scores, calibration, alpha):
    """Split-conformal p-value with ties counted as exceedances."""
    exceedances = (
        np.asarray(calibration)[np.newaxis, np.newaxis, :]
        >= (1.0 - np.asarray(scores))[..., np.newaxis]
    ).sum(axis=-1)
    return (exceedances + 1) / (len(calibration) + 1) > alpha


@pytest.mark.parametrize("alpha", [0.2, 0.5])
def test_conformal_membership_includes_high_score_labels(alpha):
    raw_calibration = np.array([0.25, 0.05, 0.2, 0.1])
    calibration, resolved_alpha = _validated_conformal_calibration(
        raw_calibration,
        alpha,
    )
    # The nonconformity of 0.75 equals the calibration value 0.25 exactly, so
    # it decides whether ties count as exceedances; 0.74 falls just short.
    scores = np.array([[0.95, 0.1], [0.7, 0.7], [0.75, 0.74], [0.8, 0.79]])
    sets = _conformal_membership(scores, calibration, resolved_alpha)

    np.testing.assert_array_equal(calibration, np.sort(raw_calibration))
    assert resolved_alpha == alpha
    np.testing.assert_array_equal(
        sets,
        _reference_conformal_membership(scores, raw_calibration, alpha),
    )
    expected = (
        [[True, False], [False, False], [True, False], [True, True]]
        if alpha == 0.2
        else [[True, False], [False, False], [False, False], [True, False]]
    )
    np.testing.assert_array_equal(sets, expected)


def test_feature_identifier_order_is_preserved():
    identifiers = _feature_ids(
        np.array(["gene_b", "gene_a", "gene_c"]),
        name="Reference feature identifiers",
    )

    np.testing.assert_array_equal(identifiers, ["gene_b", "gene_a", "gene_c"])
    assert not identifiers.flags.writeable


def test_feature_alignment_rejects_duplicate_identifiers():
    with pytest.raises(ValueError, match="unique"):
        _feature_ids(
            np.array(["gene_a", "gene_a"]),
            name="Reference feature identifiers",
        )


def _decided(codes, weights, threshold):
    """Return the vote of a block and whether each row keeps its label."""
    from scarf.mapping.confidence import _label_vote_block
    from scarf.mapping.label_transfer import ASSIGNED, _abstention_reasons

    votes = _label_vote_block(codes, weights)
    reasons = _abstention_reasons(
        votes,
        np.zeros(len(codes)),
        threshold_fraction=threshold,
        max_distance=None,
    )
    return votes, reasons == ASSIGNED


@pytest.mark.parametrize("n_neighbors", [1, 3, 11, 100])
@pytest.mark.parametrize("threshold", [0.0, 0.5, 1.0])
def test_block_label_votes_match_scalar_categorical_votes(n_neighbors, threshold):
    rng = np.random.default_rng(91)
    codes = rng.integers(-1, 9, (80, n_neighbors))
    weights = rng.uniform(size=codes.shape)
    weights[rng.random(codes.shape) < 0.2] = 0
    codes[0] = -1
    weights[1] = 0
    actual, assigned = _decided(codes, weights, threshold)
    for row in range(len(codes)):
        mass = {}
        for code, weight in zip(codes[row], weights[row], strict=True):
            if code >= 0:
                mass[code] = mass.get(code, 0.0) + float(weight)
        labeled_total = sum(mass.values())
        if labeled_total <= 0:
            assert not actual.has_labeled_votes[row]
            assert not assigned[row]
            assert (
                actual.vote_fraction[row]
                == actual.vote_entropy[row]
                == actual.top_two_margin[row]
                == 0
            )
            continue
        fractions = {
            code: value / float(weights[row].sum()) for code, value in mass.items()
        }
        ordered = sorted(fractions.items(), key=lambda item: item[1], reverse=True)
        top = ordered[0][1]
        winners = [code for code, fraction in ordered if np.isclose(fraction, top)]
        entropy = -sum(
            (value / labeled_total) * np.log(value / labeled_total)
            for value in mass.values()
            if value > 0
        )
        margin = top - (ordered[1][1] if len(ordered) > 1 else 0)
        assert actual.has_labeled_votes[row]
        assert actual.is_tied[row] == (len(winners) != 1)
        assert assigned[row] == (top >= threshold and len(winners) == 1)
        if not actual.is_tied[row]:
            assert actual.winner_codes[row] == winners[0]
        assert actual.vote_fraction[row] == top
        assert actual.top_two_margin[row] == margin
        assert actual.vote_entropy[row] == pytest.approx(entropy, rel=0, abs=1e-12)


def test_label_votes_preserve_ties_thresholds_and_large_class_codes():
    codes = np.array(
        [[1000000, 2000000, -1], [1000000, 1000000, -1], [1000000, 2000000, -1]]
    )
    weights = np.array(
        [[0.500001, 0.499999, 0], [0.1, 0.2, 0.7], [1e-12, 0, 1 - 1e-12]]
    )
    votes, assigned = _decided(codes, weights, 0.0)
    assert votes.fractions.shape == codes.shape
    assert votes.is_tied.tolist() == [True, False, True]
    assert assigned.tolist() == [False, True, False]
    assert votes.winner_codes[1] == 1000000
    _, at_threshold = _decided(codes[1:2], weights[1:2], votes.vote_fraction[1])
    _, above_threshold = _decided(
        codes[1:2], weights[1:2], np.nextafter(votes.vote_fraction[1], np.inf)
    )
    assert at_threshold[0]
    assert not above_threshold[0]


def test_same_physical_store_matches_normalized_and_nested_locations(tmp_path):
    from types import SimpleNamespace

    from scarf.datastore._operations.mapping import _same_physical_store

    def datastore(location, zarr_loc=None):
        root = zarr.open_group(str(location), mode="a")
        return SimpleNamespace(z=root, zarr_loc=zarr_loc)

    reference_path = tmp_path / "reference.zarr"
    reference = SimpleNamespace(datastore=datastore(reference_path))
    for query in (
        datastore(reference_path, zarr_loc=f"file://{reference_path}/"),
        datastore(reference_path / "nested.zarr"),
    ):
        assert _same_physical_store(query, reference)
    assert not _same_physical_store(datastore(tmp_path / "query.zarr"), reference)
    shared = SimpleNamespace(z=zarr.open_group(store=MemoryStore(), mode="w"))
    assert _same_physical_store(shared, SimpleNamespace(datastore=shared))
    with pytest.raises(
        TypeError, match="reference.datastore must be an open DataStore"
    ):
        _same_physical_store(shared, SimpleNamespace(datastore=object()))


def test_projected_coordinate_replay_is_contiguous_and_rejects_truncation():
    from tempfile import TemporaryFile

    from scarf.datastore._operations.mapping import _read_projected_blocks

    coordinates = np.arange(10, dtype=np.float64).reshape(5, 2) / 4.0
    uninformative = np.array([False, True, False, False, True])

    with TemporaryFile() as handle:
        coordinates.tofile(handle)
        blocks = list(
            _read_projected_blocks(handle, uninformative, n_dims=2, block_rows=2)
        )
    assert [start for start, _, _ in blocks] == [0, 2, 4]
    assert [len(values) for _, values, _ in blocks] == [2, 2, 1]
    np.testing.assert_array_equal(
        np.vstack([values for _, values, _ in blocks]), coordinates
    )
    np.testing.assert_array_equal(
        np.concatenate([flags for _, _, flags in blocks]), uninformative
    )

    # A replay file one row short must fail instead of yielding a short block.
    with TemporaryFile() as handle:
        coordinates[:4].tofile(handle)
        replay = _read_projected_blocks(handle, uninformative, n_dims=2, block_rows=2)
        assert next(replay)[0] == 0
        assert next(replay)[0] == 2
        with pytest.raises(RuntimeError, match="coordinates are incomplete"):
            next(replay)
