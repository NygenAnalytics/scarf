"""Unit tests for feature-operation helpers."""

import numpy as np

from scarf.datastore._operations.features import _group_assignment_digest


def test_group_assignment_digest_follows_each_feature_label():
    values = np.array([1, 2, 1])
    first = _group_assignment_digest(values)

    assert first == _group_assignment_digest(values.copy())
    assert first != _group_assignment_digest(np.array([1, 2, 3]))
    # Each feature keeps its own label, so reordering changes the digest.
    assert first != _group_assignment_digest(np.array([2, 1, 1]))
    assert first != _group_assignment_digest(np.array([1, 2]))
