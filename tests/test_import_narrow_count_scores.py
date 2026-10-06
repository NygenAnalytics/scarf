"""Scores of assays without a normalization do not depend on the count dtype.

Imports store small integral counts as uint8 or uint16, in which NumPy would
take logarithms in float16 or float32.
"""

import numpy as np
import pytest

from scarf import DataStore
from tests.storage_helpers import write_count_store

# Guide counts of 4,000 cells whose float16 logarithms sum to wrong feature
# averages, which bin some control features differently.
_COUNTS = (
    np.random.default_rng(0)
    .poisson(np.linspace(0.3, 4.0, 40), size=(4_000, 40))
    .astype(np.int64)
)
_FEATURES = [f"CRISPR{index}" for index in range(0, 40, 8)]


def _store(tmp_path, dtype: str) -> DataStore:
    path = str(tmp_path / f"{dtype}.zarr")
    write_count_store(path, {"RNA": _COUNTS, "CRISPR": _COUNTS}, dtype)
    return DataStore(path, default_assay="RNA", min_features_per_cell=0, nthreads=1)


def _scores(store: DataStore, **options) -> np.ndarray:
    return store.CRISPR.score_features(_FEATURES, "I", 3, 4, 1, **options)


@pytest.fixture(scope="module")
def uint32_store(tmp_path_factory) -> DataStore:
    """Counts stored as uint32, whose logarithms are float64."""
    return _store(tmp_path_factory.mktemp("uint32"), "uint32")


@pytest.fixture(scope="module")
def uint32_scores(uint32_store) -> np.ndarray:
    return _scores(uint32_store, log_transform=True)


@pytest.mark.parametrize("dtype", ["uint8", "uint16", "float32"])
def test_generic_assay_log_scores_match_every_storage_dtype(
    tmp_path, uint32_scores, dtype
):
    scores = _scores(_store(tmp_path, dtype), log_transform=True)
    assert scores.dtype == np.float64
    np.testing.assert_array_equal(scores, uint32_scores)


def test_assays_without_a_normalization_score_their_counts_by_default(
    uint32_store, uint32_scores
):
    unlogged = _scores(uint32_store)
    assert unlogged.dtype == np.float64
    assert unlogged.shape == uint32_scores.shape == (uint32_store.cells.N,)
    assert not np.allclose(unlogged, uint32_scores)


def test_log_scores_bin_control_features_by_their_mean_logs(tmp_path):
    # One large count makes the third feature the highest by mean count but
    # the lowest by mean log, which gives the target another control feature.
    counts = np.tile([2, 3, 0, 1], (10, 1))
    counts[0, 2] = 40
    path = str(tmp_path / "skewed.zarr")
    write_count_store(path, {"RNA": counts, "CRISPR": counts}, "uint32")
    store = DataStore(path, default_assay="RNA", min_features_per_cell=0, nthreads=1)

    scores = store.CRISPR.score_features(["CRISPR0"], "I", 4, 3, 0, log_transform=True)

    np.testing.assert_allclose(scores, np.full(10, np.log1p(2.0) - np.log1p(1.0)))
