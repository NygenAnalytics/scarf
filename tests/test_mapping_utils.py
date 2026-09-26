import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

from scarf.mapping.confidence import (
    _distance_quantile_summary,
    conformal_prediction_sets,
    distance_weights,
    mapping_score_weights,
)
from scarf.mapping.features import _feature_ids
from scarf.mapping.hashing import array_hash, array_store_hash


def test_mapping_array_hashes_match_golden_values():
    # The length-prefixed encoding replaced separator-joined identifiers, so
    # these values intentionally differ from the previous release.
    numeric = np.array([[1.5, -2.0], [0.0, 4.25]], dtype=np.float64)
    identifiers = np.array(["gene_a", "gene_b"], dtype="<U6")

    assert (
        array_hash(numeric)
        == "cd8f2c493797778faf739739e7e7e5fd535f0d772d001c0e92099a81d9399faf"
    )
    assert (
        array_store_hash(numeric)
        == "cd8f2c493797778faf739739e7e7e5fd535f0d772d001c0e92099a81d9399faf"
    )
    assert (
        array_hash(identifiers)
        == "5c78c9ed0b0ce3f452f5592d9a6be9de0456ab14aa165d5738dddc074c6316a7"
    )
    assert (
        array_store_hash(identifiers)
        == "5c78c9ed0b0ce3f452f5592d9a6be9de0456ab14aa165d5738dddc074c6316a7"
    )


def test_mapping_identifier_hashes_are_unambiguous():
    assert array_hash(["a\x1fb", "c"]) != array_hash(["a", "b\x1fc"])
    assert array_hash(["ab", ""]) != array_hash(["a", "b"])
    assert array_hash(["a", "b"]) != array_hash([["a", "b"]])


def test_mapping_identifier_hashes_agree_across_string_storage():
    identifiers = ["gene_a", "gene_bb", "g"]
    expected = array_hash(identifiers)
    root = zarr.open_group(store=MemoryStore(), mode="w")
    stored = root.create_array(
        "ids",
        shape=(len(identifiers),),
        dtype=np.dtypes.StringDType(),
        chunks=(2,),
    )
    stored[:] = np.array(identifiers, dtype=np.dtypes.StringDType())
    assert np.dtype(stored.dtype).kind == "T"

    assert array_store_hash(stored) == expected
    for dtype in (object, "U", "S", np.dtypes.StringDType()):
        values = np.array(identifiers, dtype=dtype)
        assert array_hash(values) == expected
        assert array_store_hash(values) == expected


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


def test_conformal_prediction_sets_include_high_score_labels():
    sets = conformal_prediction_sets(
        np.array([[0.95, 0.1], [0.7, 0.7]]),
        np.array([0.05, 0.1, 0.2, 0.25]),
        alpha=0.2,
    )

    assert sets.shape == (2, 2)
    assert sets[0, 0]
    assert not sets[0, 1]


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


@pytest.mark.parametrize("n_neighbors", [1, 3, 11, 100])
@pytest.mark.parametrize("threshold", [0.0, 0.5, 1.0])
def test_block_label_votes_match_scalar_categorical_votes(n_neighbors, threshold):
    from scarf.mapping.confidence import _label_vote_block

    rng = np.random.default_rng(91)
    codes = rng.integers(-1, 9, (80, n_neighbors))
    weights = rng.uniform(size=codes.shape)
    weights[rng.random(codes.shape) < 0.2] = 0
    codes[0] = -1
    weights[1] = 0
    actual = _label_vote_block(codes, weights, threshold)
    for row in range(len(codes)):
        mass = {}
        for code, weight in zip(codes[row], weights[row], strict=True):
            if code >= 0:
                mass[code] = mass.get(code, 0.0) + float(weight)
        labeled_total = sum(mass.values())
        if labeled_total <= 0:
            assert actual.is_unknown[row]
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
        assert actual.is_unknown[row] == (top < threshold or len(winners) != 1)
        if not actual.is_unknown[row]:
            assert actual.prediction_codes[row] == winners[0]
        assert actual.vote_fraction[row] == top
        assert actual.top_two_margin[row] == margin
        assert actual.vote_entropy[row] == pytest.approx(entropy, rel=0, abs=1e-12)


def test_label_votes_preserve_ties_thresholds_and_large_class_codes():
    from scarf.mapping.confidence import _label_vote_block

    codes = np.array(
        [[1000000, 2000000, -1], [1000000, 1000000, -1], [1000000, 2000000, -1]]
    )
    weights = np.array(
        [[0.500001, 0.499999, 0], [0.1, 0.2, 0.7], [1e-12, 0, 1 - 1e-12]]
    )
    votes = _label_vote_block(codes, weights, 0.0)
    assert votes.fractions.shape == codes.shape
    assert votes.is_unknown.tolist() == [True, False, True]
    at_threshold = _label_vote_block(codes[1:2], weights[1:2], votes.vote_fraction[1])
    above_threshold = _label_vote_block(
        codes[1:2], weights[1:2], np.nextafter(votes.vote_fraction[1], np.inf)
    )
    assert not at_threshold.is_unknown[0]
    assert above_threshold.is_unknown[0]
