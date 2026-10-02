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


def _scores(tmp_path, dtype: str) -> np.ndarray:
    path = str(tmp_path / f"{dtype}.zarr")
    write_count_store(path, {"RNA": _COUNTS, "CRISPR": _COUNTS}, dtype)
    store = DataStore(path, default_assay="RNA", min_features_per_cell=0, nthreads=1)
    features = [f"CRISPR{index}" for index in range(0, 40, 8)]
    return store.CRISPR.score_features(features, "I", 3, 4, 1, log_transform=True)


@pytest.mark.parametrize("dtype", ["uint8", "uint16", "float32"])
def test_generic_assay_log_scores_match_every_storage_dtype(tmp_path, dtype):
    expected = _scores(tmp_path, "uint32")
    scores = _scores(tmp_path, dtype)
    assert scores.dtype == np.float64
    np.testing.assert_array_equal(scores, expected)
