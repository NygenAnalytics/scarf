"""Validation coverage for fate-mapping helpers."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from scarf.trajectory.fate import (
    _normalize_pseudotime,
    _validate_graph,
    _validate_sink_groups,
)


def test_validate_sink_groups_accepts_and_rejects():
    labels = np.array(["A", "B", "A", "C"])
    sinks, groups = _validate_sink_groups(labels, ["A", "B"])
    assert sinks == ("A", "B")
    np.testing.assert_array_equal(groups, [0, 1, 0, -1])

    with pytest.raises(TypeError, match="must be a list"):
        _validate_sink_groups(labels, ("A",))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="At least one sink"):
        _validate_sink_groups(labels, [])
    with pytest.raises(ValueError, match="unique"):
        _validate_sink_groups(labels, ["A", "A"])
    with pytest.raises(ValueError, match="were not found"):
        _validate_sink_groups(labels, ["Z"])
    with pytest.raises(TypeError, match="hashable scalar"):
        _validate_sink_groups(labels, [["A"]])  # type: ignore[list-item]


def test_validate_graph_and_normalize_pseudotime_edges():
    good = csr_matrix(np.array([[0.0, 1.0], [1.0, 0.0]]))
    assert _validate_graph(good, 2) is None

    with pytest.raises(TypeError, match="csr_matrix"):
        _validate_graph(np.eye(2), 2)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="does not match"):
        _validate_graph(good, 3)
    with pytest.raises(ValueError, match="non-negative"):
        _validate_graph(csr_matrix(np.array([[0.0, -1.0], [-1.0, 0.0]])), 2)
    with pytest.raises(ValueError, match="finite"):
        _validate_graph(csr_matrix(np.array([[0.0, np.nan], [np.nan, 0.0]])), 2)

    np.testing.assert_allclose(
        _normalize_pseudotime(np.array([2.0, 4.0, 6.0])),
        [0.0, 0.5, 1.0],
    )
    with pytest.raises(TypeError, match="numeric"):
        _normalize_pseudotime(np.array(["a", "b"]))
    with pytest.raises(ValueError, match="one-dimensional"):
        _normalize_pseudotime(np.array([[1.0, 2.0]]))
    with pytest.raises(ValueError, match="at least two distinct"):
        _normalize_pseudotime(np.array([1.0, 1.0, 1.0]))
