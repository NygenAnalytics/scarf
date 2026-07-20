import numba
import numpy as np
import pandas as pd
import pytest

from scarf.features.enrichment.aucell import (
    build_gene_set_index,
    make_rank_permutation,
    resolve_n_up,
    score_aucell_block,
)
from scarf.features.enrichment.net import prepare_network
from scarf.features.enrichment.waggr import build_waggr_model, score_waggr_block


def _prepared_network(*, weighted: bool):
    return prepare_network(
        pd.DataFrame(
            {
                "source": ["Alpha", "Alpha", "Beta", "Beta"],
                "target": ["g0", "g2", "g1", "g3"],
                "weight": [-1.0, 2.0, 3.0, -1.0],
            }
        ),
        active_feature_names=np.array(["g0", "g1", "g2", "g3"]),
        active_feature_index=np.arange(4),
        tmin=2,
        weighted=weighted,
    )


def test_waggr_matches_frozen_decoupler_2_2_reference():
    model = build_waggr_model(_prepared_network(weighted=True))
    values = np.array([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]])

    sums = score_waggr_block(values, model, mode="wsum")
    means = score_waggr_block(values, model, mode="wmean")

    np.testing.assert_allclose(sums, [[5.0, 2.0], [0.0, 8.0]])
    np.testing.assert_allclose(means, [[5.0 / 3.0, 0.5], [0.0, 2.0]])


def test_waggr_is_invariant_to_row_block_boundaries():
    model = build_waggr_model(_prepared_network(weighted=True))
    values = np.arange(28, dtype=np.float64).reshape(7, 4)

    whole = score_waggr_block(values, model, mode="wmean")
    split = np.vstack(
        [
            score_waggr_block(values[:2], model, mode="wmean"),
            score_waggr_block(values[2:5], model, mode="wmean"),
            score_waggr_block(values[5:], model, mode="wmean"),
        ]
    )

    np.testing.assert_array_equal(whole, split)


def test_waggr_rejects_invalid_values_and_modes():
    model = build_waggr_model(_prepared_network(weighted=True))

    with pytest.raises(ValueError, match="mode"):
        score_waggr_block(np.ones((1, 4)), model, mode="median")  # type: ignore
    with pytest.raises(ValueError, match="finite"):
        score_waggr_block(
            np.array([[1.0, np.nan, 2.0, 3.0]]),
            model,
            mode="wmean",
        )


def test_aucell_matches_frozen_decoupler_2_2_reference():
    network = _prepared_network(weighted=False)
    permutation = make_rank_permutation(4, 0)
    rank_feature_index = np.arange(4)[permutation]
    sets = build_gene_set_index(network, rank_feature_index)
    values = np.array(
        [
            [4, 3, 2, 1],
            [1, 4, 3, 2],
        ],
        dtype=np.uint32,
    )

    scores = score_aucell_block(values, permutation, sets, n_up=3)

    np.testing.assert_allclose(
        scores,
        [
            [2.0 / 3.0, 1.0 / 3.0],
            [1.0 / 3.0, 2.0 / 3.0],
        ],
    )


def test_aucell_uses_one_global_seeded_tie_permutation():
    network = _prepared_network(weighted=False)
    permutation = make_rank_permutation(4, 0)
    sets = build_gene_set_index(network, np.arange(4)[permutation])
    tied = np.ones((3, 4), dtype=np.float64)

    scores = score_aucell_block(tied, permutation, sets, n_up=2)

    np.testing.assert_array_equal(permutation, [2, 0, 1, 3])
    np.testing.assert_array_equal(scores, np.array([[1.0, 0.0]] * 3))
    np.testing.assert_array_equal(
        score_aucell_block(np.zeros((1, 4)), permutation, sets, n_up=2),
        [[0.0, 0.0]],
    )


def test_aucell_is_block_and_thread_count_invariant():
    network = _prepared_network(weighted=False)
    permutation = make_rank_permutation(4, 11)
    sets = build_gene_set_index(network, np.arange(4)[permutation])
    values = np.array(
        [[9, 3, 1, 0], [2, 4, 8, 1], [0, 1, 0, 2], [7, 7, 7, 7]],
        dtype=np.float64,
    )
    original_threads = numba.get_num_threads()
    try:
        numba.set_num_threads(1)
        single_thread = score_aucell_block(values, permutation, sets, n_up=3)
        numba.set_num_threads(min(2, numba.config.NUMBA_NUM_THREADS))
        two_threads = score_aucell_block(values, permutation, sets, n_up=3)
    finally:
        numba.set_num_threads(original_threads)
    blocked = np.vstack(
        [
            score_aucell_block(values[:1], permutation, sets, n_up=3),
            score_aucell_block(values[1:], permutation, sets, n_up=3),
        ]
    )

    np.testing.assert_array_equal(single_thread, two_threads)
    np.testing.assert_array_equal(single_thread, blocked)


def test_resolve_n_up_validates_the_ranking_universe():
    assert resolve_n_up(100, None) == 5
    assert resolve_n_up(3, None) == 2

    with pytest.raises(ValueError, match="at least two"):
        resolve_n_up(1, None)
    with pytest.raises(ValueError, match="greater than 1"):
        resolve_n_up(4, 1)
    with pytest.raises(ValueError, match="at most 4"):
        resolve_n_up(4, 5)
