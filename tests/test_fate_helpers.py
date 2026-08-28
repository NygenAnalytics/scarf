"""Validation coverage for fate-mapping helpers."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from scarf.trajectory import fate as fate_module
from scarf.trajectory.fate import (
    _make_transition,
    _normalize_pseudotime,
    _validate_graph,
    _validate_sink_groups,
    compute_fate_probabilities,
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


def test_validate_sink_groups_rejects_unhashable_scalar_values():
    with pytest.raises(TypeError, match="hashable scalar"):
        _validate_sink_groups(
            np.array(["A"], dtype=object),
            [{"sink": "A"}],  # type: ignore[list-item]
        )


def test_validate_sink_groups_rejects_invalid_comparisons():
    class UncomparableSink:
        def __hash__(self):
            return 1

        def __eq__(self, other):
            raise ValueError("comparison is undefined")

    with pytest.raises(TypeError, match="comparable scalar"):
        _validate_sink_groups(
            np.array(["A"], dtype=object),
            [UncomparableSink()],
        )
    with pytest.raises(TypeError, match="comparable scalar"):
        _validate_sink_groups(np.array([["A"]]), ["A"])


def test_validate_sink_groups_rejects_overlapping_custom_labels():
    class WildcardLabel:
        def __eq__(self, other):
            return other in {"A", "B"}

    labels = np.empty(1, dtype=object)
    labels[0] = WildcardLabel()

    with pytest.raises(ValueError, match="disjoint"):
        _validate_sink_groups(labels, ["A", "B"])


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


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf, -np.inf])
def test_normalize_pseudotime_rejects_nonfinite_values(invalid_value):
    with pytest.raises(ValueError, match="finite"):
        _normalize_pseudotime(np.array([0.0, invalid_value]))


def test_transition_rejects_isolated_transient_cells():
    graph = csr_matrix(np.eye(2))

    with pytest.raises(ValueError, match="1 isolated transient cells"):
        _make_transition(
            graph,
            np.array([0.0, 1.0]),
            np.array([True, False]),
            beta=1.0,
        )


def test_duplicate_graph_weights_cannot_overflow_during_canonicalization():
    maximum = np.finfo(np.float64).max
    graph = csr_matrix(
        (
            np.array([maximum, maximum, maximum, maximum]),
            np.array([1, 1, 0, 0]),
            np.array([0, 2, 4]),
        ),
        shape=(2, 2),
    )

    with pytest.raises(ValueError, match="became non-finite"):
        compute_fate_probabilities(
            graph,
            np.array([0.0, 1.0]),
            np.array(["root", "A"]),
            ["A"],
        )


def test_transition_rejects_weights_that_underflow_during_float_conversion():
    longdouble_min = np.nextafter(np.longdouble(0), np.longdouble(1))
    float64_min = np.nextafter(np.float64(0), np.float64(1))
    if longdouble_min >= np.longdouble(float64_min):
        pytest.skip(
            "Platform longdouble has no wider exponent range than float64, so "
            "an underflowing weight cannot be constructed here"
        )
    tiny = longdouble_min
    graph = csr_matrix(
        (
            np.array([tiny, tiny], dtype=np.longdouble),
            (np.array([0, 1]), np.array([1, 0])),
        ),
        shape=(2, 2),
    )

    with pytest.raises(ValueError, match="remain positive"):
        _make_transition(
            graph,
            np.array([0.0, 1.0]),
            np.array([False, True]),
            beta=1.0,
        )


def test_fate_solver_rejects_nonfinite_backend_output(monkeypatch):
    graph = csr_matrix(
        np.array(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ]
        )
    )

    def nonfinite_solution(operator, *_args, **_kwargs):
        return np.full(operator.shape[0], np.nan), 0

    monkeypatch.setattr(fate_module, "gmres", nonfinite_solution)

    with pytest.raises(RuntimeError, match="produced non-finite values"):
        compute_fate_probabilities(
            graph,
            np.array([0.0, 0.5, 1.0]),
            np.array(["A", "middle", "B"]),
            ["A", "B"],
        )


def test_fate_solver_rejects_float32_overflow(monkeypatch):
    graph = csr_matrix(
        np.array(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ]
        )
    )

    def overflowing_solution(operator, *_args, **_kwargs):
        return np.full(operator.shape[0], np.finfo(np.float64).max), 0

    monkeypatch.setattr(fate_module, "gmres", overflowing_solution)

    with np.errstate(over="ignore"):
        with pytest.raises(
            RuntimeError, match="calculation produced non-finite values"
        ):
            compute_fate_probabilities(
                graph,
                np.array([0.0, 0.5, 1.0]),
                np.array(["A", "middle", "B"]),
                ["A", "B"],
            )
