"""Unit tests for feature-operation helpers and marker write guards."""

import numpy as np
import pandas as pd
import pytest
import zarr
from zarr.storage import MemoryStore

from scarf.datastore._operations.features import (
    _MARKER_STAT_COLUMNS,
    _group_assignment_digest,
    _marker_stats_matrix,
    _shared_marker_feature_index,
)
from scarf.datastore.datastore import DataStore
from scarf.features.markers.table import MARKER_STAT_COLUMNS


def _marker_frame(index: list[int], *, score: float = 1.0) -> pd.DataFrame:
    n = len(index)
    data = {
        column: np.full(n, score, dtype=np.float64) for column in MARKER_STAT_COLUMNS
    }
    return pd.DataFrame(data, index=np.asarray(index, dtype=np.int64))


def test_shared_marker_feature_index_sorts_and_casts():
    markers = {0: _marker_frame([3, 1], score=0.5)}
    shared = _shared_marker_feature_index(markers)
    np.testing.assert_array_equal(shared, np.array([1, 3], dtype=np.int32))
    assert shared.dtype == np.int32


def test_shared_marker_feature_index_rejects_invalid_groups():
    with pytest.raises(ValueError, match="unique as strings"):
        _shared_marker_feature_index(
            {
                1: _marker_frame([0, 1]),
                "1": _marker_frame([0, 1]),
            }
        )
    bad_index = _marker_frame([0, 1])
    bad_index.index = pd.Index(["a", "b"])
    with pytest.raises(ValueError, match="one-dimensional integer index"):
        _shared_marker_feature_index({0: bad_index})
    duplicate = _marker_frame([0, 0])
    with pytest.raises(ValueError, match="unique within each group"):
        _shared_marker_feature_index({0: duplicate})
    oversized = _marker_frame([0, int(np.iinfo(np.int32).max) + 1])
    with pytest.raises(ValueError, match="non-negative int32"):
        _shared_marker_feature_index({0: oversized})
    negative = _marker_frame([-1, 0])
    with pytest.raises(ValueError, match="non-negative int32"):
        _shared_marker_feature_index({0: negative})
    mismatched = {
        0: _marker_frame([0, 1]),
        1: _marker_frame([0, 2]),
    }
    with pytest.raises(ValueError, match="identical feature index sets"):
        _shared_marker_feature_index(mismatched)
    with pytest.raises(ValueError, match="Cannot save empty"):
        _shared_marker_feature_index(
            {
                0: _marker_frame([]),
                1: _marker_frame([]),
            }
        )


def test_marker_stats_matrix_requires_finite_aligned_values():
    frame = _marker_frame([0, 2], score=1.5)
    with pytest.raises(ValueError, match="must all be finite"):
        _marker_stats_matrix(frame, np.array([0, 1, 2], dtype=np.int32))

    finite = _marker_frame([0, 1], score=2.0)
    matrix = _marker_stats_matrix(finite, np.array([0, 1], dtype=np.int32))
    assert matrix.shape == (2, len(_MARKER_STAT_COLUMNS))
    np.testing.assert_allclose(matrix, 2.0)

    poisoned = finite.copy()
    poisoned.loc[0, "score"] = np.nan
    with pytest.raises(ValueError, match="must all be finite"):
        _marker_stats_matrix(poisoned, np.array([0, 1], dtype=np.int32))


def test_group_assignment_digest_is_deterministic():
    values = np.array([1, 2, 1])
    first = _group_assignment_digest(values)
    second = _group_assignment_digest(values.copy())
    assert first == second
    assert first != _group_assignment_digest(np.array([1, 2, 3]))


def test_write_marker_slot_requires_valid_counts_and_skips_empty_groups():
    root = zarr.open_group(store=MemoryStore(), mode="w")
    slot = root.create_group("markers")
    markers = {
        0: _marker_frame([1, 0], score=0.25),
        1: _marker_frame([]),
    }
    feature_names = np.array(["first", "second"])
    feature_ids = np.array(["id-1", "id-2"])

    with pytest.raises(ValueError, match="target and reference counts"):
        DataStore._write_marker_slot(
            slot,
            markers,
            group_cell_counts={},
            feature_names=feature_names,
            feature_ids=feature_ids,
        )
    with pytest.raises(ValueError, match="integers >= 2"):
        DataStore._write_marker_slot(
            slot,
            markers,
            group_cell_counts={0: (1, 5)},
            feature_names=feature_names,
            feature_ids=feature_ids,
        )

    DataStore._write_marker_slot(
        slot,
        markers,
        group_cell_counts={0: (4, 8)},
        feature_names=feature_names,
        feature_ids=feature_ids,
    )
    np.testing.assert_array_equal(
        slot["feature_index"][:], np.array([0, 1], dtype=np.int32)
    )
    assert "0" in slot
    assert "1" not in slot
    np.testing.assert_array_equal(slot["feature_names"][:], feature_names)
    np.testing.assert_array_equal(slot["feature_ids"][:], feature_ids)
    assert slot["0"].attrs["n_group"] == 4
    assert slot["0"].attrs["n_reference"] == 8
    np.testing.assert_allclose(slot["0"]["stats"][:], 0.25)
