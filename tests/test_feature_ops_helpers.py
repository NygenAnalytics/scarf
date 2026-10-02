"""Unit tests for feature-operation helpers."""

import numpy as np

from scarf.datastore._operations.features import _group_assignment_digest


def test_group_assignment_digest_is_deterministic():
    values = np.array([1, 2, 1])
    first = _group_assignment_digest(values)
    second = _group_assignment_digest(values.copy())
    assert first == second
    assert first != _group_assignment_digest(np.array([1, 2, 3]))
