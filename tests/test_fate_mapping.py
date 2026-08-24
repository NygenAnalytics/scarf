from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from scipy.sparse import csr_matrix, diags

from scarf.datastore.graph_datastore import GraphDataStore
from scarf.trajectory.fate import (
    _make_transition,
    _normalize_pseudotime,
    compute_fate_probabilities,
    make_sink_tokens,
)
from scarf.trajectory.results import FateMappingResult


def _y_graph() -> tuple[csr_matrix, np.ndarray, np.ndarray]:
    adjacency = np.zeros((5, 5), dtype=np.float64)
    for first, second in ((0, 1), (1, 2), (0, 3), (3, 4)):
        adjacency[first, second] = adjacency[second, first] = 1.0
    pseudotime = np.array([0.0, 0.5, 1.0, 0.5, 1.0])
    labels = np.array(["root", "a-mid", "A", "b-mid", "B"])
    return csr_matrix(adjacency), pseudotime, labels


def _dense_dirichlet_reference(
    transition: csr_matrix,
    labels: np.ndarray,
    sinks: list[str],
) -> np.ndarray:
    matrix = np.eye(transition.shape[0]) - transition.toarray()
    absorbing = np.isin(labels, sinks)
    matrix[absorbing] = 0.0
    matrix[np.flatnonzero(absorbing), np.flatnonzero(absorbing)] = 1.0
    probabilities = np.empty((transition.shape[0], len(sinks)), dtype=np.float64)
    for group, sink in enumerate(sinks):
        boundary = np.asarray(labels == sink, dtype=np.float64)
        probabilities[:, group] = np.linalg.solve(matrix, boundary)
    return probabilities


def test_soft_transition_preserves_support_and_normalizes_rows():
    graph, pseudotime, _ = _y_graph()
    graph.setdiag(2.0)
    expected_support = graph.toarray() > 0
    np.fill_diagonal(expected_support, False)

    transition = _make_transition(
        graph,
        _normalize_pseudotime(pseudotime),
        np.zeros(graph.shape[0], dtype=bool),
        beta=10.0,
    )

    np.testing.assert_array_equal(transition.toarray() > 0, expected_support)
    np.testing.assert_allclose(np.asarray(transition.sum(axis=1)).ravel(), 1.0)
    assert transition.dtype == np.float64
    assert transition[1, 0] < transition[1, 2]
    assert transition[0, 1] == pytest.approx(transition[0, 3])


def test_soft_transition_is_affine_invariant_and_beta_zero_is_unbiased():
    graph, pseudotime, _ = _y_graph()
    transformed = (pseudotime * 23.0) - 7.0

    first = _make_transition(
        graph.copy(),
        _normalize_pseudotime(pseudotime),
        np.zeros(graph.shape[0], dtype=bool),
        beta=10.0,
    )
    second = _make_transition(
        graph.copy(),
        _normalize_pseudotime(transformed),
        np.zeros(graph.shape[0], dtype=bool),
        beta=10.0,
    )
    unbiased = _make_transition(
        graph.copy(),
        _normalize_pseudotime(pseudotime),
        np.zeros(graph.shape[0], dtype=bool),
        beta=0.0,
    )
    expected_unbiased = graph.toarray()
    expected_unbiased /= expected_unbiased.sum(axis=1, keepdims=True)

    np.testing.assert_allclose(first.toarray(), second.toarray())
    np.testing.assert_allclose(unbiased.toarray(), expected_unbiased)


def test_soft_transition_preserves_extreme_penalty_ratios():
    adjacency = np.zeros((3, 3), dtype=np.float64)
    adjacency[0, 1] = adjacency[1, 0] = 1.0
    adjacency[0, 2] = adjacency[2, 0] = 1.0

    transition = _make_transition(
        csr_matrix(adjacency),
        np.array([1.0, 0.2, 0.0]),
        np.zeros(3, dtype=bool),
        beta=1000.0,
    )

    assert transition[0, 1] > 1.0 - 1e-12
    assert 0.0 < transition[0, 2] < 1e-80
    np.testing.assert_allclose(np.asarray(transition.sum(axis=1)).ravel(), 1.0)


def test_soft_transition_normalizes_maximum_finite_weights():
    adjacency = np.full((3, 3), np.finfo(np.float64).max)
    np.fill_diagonal(adjacency, 0.0)

    transition = _make_transition(
        csr_matrix(adjacency),
        np.array([0.0, 0.5, 1.0]),
        np.zeros(3, dtype=bool),
        beta=0.0,
    )

    assert np.isfinite(transition.data).all()
    np.testing.assert_allclose(transition.toarray().sum(axis=1), 1.0)
    np.testing.assert_allclose(transition.data, 0.5)


def test_pseudotime_normalization_handles_extreme_finite_values():
    normalized = _normalize_pseudotime(np.array([-1e308, 0.0, 1e308], dtype=np.float64))

    np.testing.assert_allclose(normalized, [0.0, 0.5, 1.0])


def test_single_sink_assigns_probability_one_without_solver(
    monkeypatch: pytest.MonkeyPatch,
):
    from scarf.trajectory import fate as fate_module

    graph, pseudotime, labels = _y_graph()
    monkeypatch.setattr(
        fate_module,
        "gmres",
        lambda *_args, **_kwargs: pytest.fail("GMRES must not run for one sink"),
    )

    probabilities, valid, sink_labels = compute_fate_probabilities(
        graph,
        pseudotime,
        labels,
        ["A"],
    )

    assert sink_labels == ("A",)
    np.testing.assert_array_equal(valid, np.ones(graph.shape[0], dtype=bool))
    np.testing.assert_array_equal(probabilities, np.ones((graph.shape[0], 1)))


def test_two_sink_branch_matches_dense_dirichlet_reference():
    graph, pseudotime, labels = _y_graph()
    reference_transition = _make_transition(
        graph.copy(),
        _normalize_pseudotime(pseudotime),
        np.isin(labels, ["A", "B"]),
        beta=10.0,
    )
    expected = _dense_dirichlet_reference(
        reference_transition,
        labels,
        ["A", "B"],
    )

    probabilities, valid, _ = compute_fate_probabilities(
        graph,
        pseudotime,
        labels,
        ["A", "B"],
    )

    np.testing.assert_array_equal(valid, np.ones(graph.shape[0], dtype=bool))
    assert probabilities.dtype == np.float32
    np.testing.assert_allclose(probabilities, expected, rtol=1e-5, atol=1e-7)
    np.testing.assert_array_equal(probabilities[2], [1.0, 0.0])
    np.testing.assert_array_equal(probabilities[4], [0.0, 1.0])
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)


def test_three_sink_branch_matches_dense_dirichlet_reference():
    adjacency = np.zeros((7, 7), dtype=np.float64)
    for first, second in ((0, 1), (1, 2), (0, 3), (3, 4), (0, 5), (5, 6)):
        adjacency[first, second] = adjacency[second, first] = 1.0
    graph = csr_matrix(adjacency)
    pseudotime = np.array([0.0, 0.5, 1.0, 0.5, 1.0, 0.5, 1.0])
    labels = np.array(["root", "a-mid", "A", "b-mid", "B", "c-mid", "C"])
    sinks = ["A", "B", "C"]
    reference_transition = _make_transition(
        graph.copy(),
        _normalize_pseudotime(pseudotime),
        np.isin(labels, sinks),
        beta=10.0,
    )
    expected = _dense_dirichlet_reference(reference_transition, labels, sinks)

    probabilities, valid, _ = compute_fate_probabilities(
        graph,
        pseudotime,
        labels,
        sinks,
    )

    np.testing.assert_array_equal(valid, np.ones(graph.shape[0], dtype=bool))
    np.testing.assert_allclose(probabilities, expected, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)


def test_computation_does_not_mutate_input_graph():
    graph, pseudotime, labels = _y_graph()
    graph = graph.astype(np.float32)
    original = graph.copy()

    first, _, _ = compute_fate_probabilities(
        graph,
        pseudotime,
        labels,
        ["A", "B"],
    )
    second, _, _ = compute_fate_probabilities(
        graph,
        pseudotime,
        labels,
        ["A", "B"],
    )

    np.testing.assert_array_equal(graph.indptr, original.indptr)
    np.testing.assert_array_equal(graph.indices, original.indices)
    np.testing.assert_array_equal(graph.data, original.data)
    np.testing.assert_array_equal(first, second)


def test_sinkless_components_are_invalid_and_other_components_remain_valid():
    adjacency = np.zeros((8, 8), dtype=np.float64)
    for first, second in ((0, 1), (1, 2), (3, 4), (4, 5), (6, 7)):
        adjacency[first, second] = adjacency[second, first] = 1.0
    pseudotime = np.array([0.0, 0.5, 1.0, 0.0, 0.5, 1.0, 0.2, 0.8])
    labels = np.array(["root", "mid", "A", "root", "mid", "B", "other", "other"])

    probabilities, valid, _ = compute_fate_probabilities(
        csr_matrix(adjacency),
        pseudotime,
        labels,
        ["A", "B"],
    )

    np.testing.assert_array_equal(
        valid,
        np.array([True, True, True, True, True, True, False, False]),
    )
    np.testing.assert_allclose(probabilities[:3], [[1.0, 0.0]] * 3, atol=1e-7)
    np.testing.assert_allclose(probabilities[3:6], [[0.0, 1.0]] * 3, atol=1e-7)
    assert np.isnan(probabilities[6:]).all()


def test_explicit_zero_weight_does_not_connect_components():
    graph = csr_matrix(
        (
            np.array([1.0, 1.0, 0.0]),
            np.array([1, 0, 2]),
            np.array([0, 1, 3, 3]),
        ),
        shape=(3, 3),
    )

    probabilities, valid, _ = compute_fate_probabilities(
        graph,
        np.array([0.0, 1.0, 2.0]),
        np.array(["root", "A", "other"]),
        ["A"],
    )

    np.testing.assert_array_equal(valid, [True, True, False])
    np.testing.assert_array_equal(probabilities[:2], [[1.0], [1.0]])
    assert np.isnan(probabilities[2]).all()
    assert graph.nnz == 3


def test_nonfinite_pseudotime_in_sinkless_component_is_rejected():
    adjacency = np.zeros((4, 4), dtype=np.float64)
    adjacency[0, 1] = adjacency[1, 0] = 1.0
    adjacency[2, 3] = adjacency[3, 2] = 1.0

    with pytest.raises(ValueError, match="finite"):
        compute_fate_probabilities(
            csr_matrix(adjacency),
            np.array([0.0, 1.0, np.nan, 1.0]),
            np.array(["root", "A", "other", "other"]),
            ["A"],
        )


def test_complex_graph_weights_are_rejected_clearly():
    graph, pseudotime, labels = _y_graph()

    with pytest.raises(TypeError, match="real numeric"):
        compute_fate_probabilities(
            graph.astype(np.complex128),
            pseudotime,
            labels,
            ["A"],
        )


def test_asymmetric_graph_support_is_rejected():
    graph = csr_matrix(
        (
            np.array([1.0]),
            np.array([1]),
            np.array([0, 1, 1]),
        ),
        shape=(2, 2),
    )

    with pytest.raises(ValueError, match="symmetric"):
        compute_fate_probabilities(
            graph,
            np.array([0.0, 1.0]),
            np.array(["root", "A"]),
            ["A"],
        )


def test_compute_fate_rejects_invalid_parameters_and_inputs():
    graph, pseudotime, labels = _y_graph()

    with pytest.raises(TypeError, match="beta must be numeric"):
        compute_fate_probabilities(graph, pseudotime, labels, ["A"], beta="fast")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="beta must be finite"):
        compute_fate_probabilities(graph, pseudotime, labels, ["A"], beta=-1.0)
    with pytest.raises(ValueError, match="beta must be finite"):
        compute_fate_probabilities(graph, pseudotime, labels, ["A"], beta=np.nan)
    with pytest.raises(TypeError, match="solver_tol must be numeric"):
        compute_fate_probabilities(
            graph,
            pseudotime,
            labels,
            ["A"],
            solver_tol=None,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="solver_tol must be finite"):
        compute_fate_probabilities(graph, pseudotime, labels, ["A"], solver_tol=0.0)
    with pytest.raises(ValueError, match="solver_tol must be finite"):
        compute_fate_probabilities(graph, pseudotime, labels, ["A"], solver_tol=1.0)
    with pytest.raises(TypeError, match="max_iterations must be an integer"):
        compute_fate_probabilities(
            graph,
            pseudotime,
            labels,
            ["A"],
            max_iterations=1.5,  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="max_iterations must be an integer"):
        compute_fate_probabilities(
            graph,
            pseudotime,
            labels,
            ["A"],
            max_iterations=True,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="at least 1"):
        compute_fate_probabilities(graph, pseudotime, labels, ["A"], max_iterations=0)

    with pytest.raises(TypeError, match="csr_matrix"):
        compute_fate_probabilities(
            np.asarray(graph.toarray()),  # type: ignore[arg-type]
            pseudotime,
            labels,
            ["A"],
        )
    with pytest.raises(ValueError, match="does not match"):
        compute_fate_probabilities(graph, pseudotime, labels[:3], ["A"])
    with pytest.raises(ValueError, match="non-negative"):
        negative = graph.copy()
        negative.data[0] = -1.0
        compute_fate_probabilities(negative, pseudotime, labels, ["A"])
    with pytest.raises(ValueError, match="No cells were selected"):
        compute_fate_probabilities(
            csr_matrix((0, 0), dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=object),
            ["A"],
        )
    with pytest.raises(ValueError, match="one-dimensional"):
        compute_fate_probabilities(
            graph,
            pseudotime,
            labels.reshape(-1, 1),
            ["A"],
        )
    with pytest.raises(ValueError, match="align with the selected cells"):
        compute_fate_probabilities(graph, pseudotime[:2], labels, ["A"])
    with pytest.raises(TypeError, match="Pseudotime values must be numeric"):
        compute_fate_probabilities(
            graph,
            np.array(["a", "b", "c", "d", "e"], dtype=object),
            labels,
            ["A"],
        )
    with pytest.raises(ValueError, match="at least two distinct"):
        compute_fate_probabilities(
            graph,
            np.zeros(labels.shape[0], dtype=np.float64),
            labels,
            ["A"],
        )


def test_make_sink_tokens_fallback_and_collision_suffixes():
    assert make_sink_tokens(("!!!", "!!!", "A")) == ("sink_1", "sink_2", "A")
    assert make_sink_tokens(("A!", "A_", "A")) == ("A", "A_2", "A_3")


def test_malformed_csr_structure_is_rejected():
    graph, pseudotime, labels = _y_graph()
    graph.indices[0] = graph.shape[0]

    with pytest.raises(ValueError, match="invalid CSR"):
        compute_fate_probabilities(
            graph,
            pseudotime,
            labels,
            ["A", "B"],
        )


def test_weights_that_overflow_float64_are_rejected():
    if np.finfo(np.longdouble).max <= np.finfo(np.float64).max:
        pytest.skip("long double does not exceed float64 on this platform")
    graph, pseudotime, labels = _y_graph()
    graph = graph.astype(np.longdouble)
    graph.data[0] = np.finfo(np.longdouble).max

    with pytest.raises(ValueError, match="converted to float64"):
        compute_fate_probabilities(
            graph,
            pseudotime,
            labels,
            ["A", "B"],
        )


def test_reordering_sinks_only_reorders_probability_columns():
    graph, pseudotime, labels = _y_graph()
    forward, _, _ = compute_fate_probabilities(
        graph.copy(),
        pseudotime,
        labels,
        ["A", "B"],
    )
    reverse, _, _ = compute_fate_probabilities(
        graph.copy(),
        pseudotime,
        labels,
        ["B", "A"],
    )

    np.testing.assert_allclose(forward, reverse[:, ::-1], rtol=1e-5, atol=1e-7)


def test_sink_tokens_remain_unique_after_sanitization():
    assert make_sink_tokens(("A", "A_2", "A!")) == ("A", "A_2", "A_3")


@pytest.mark.parametrize(
    ("sinks", "error", "message"),
    [
        ([], ValueError, "At least one"),
        (["A", "A"], ValueError, "unique"),
        (["missing"], ValueError, "not found"),
        ([("A", "B")], TypeError, "scalar"),
        (("A",), TypeError, "list"),
    ],
)
def test_invalid_sink_definitions_fail_clearly(
    sinks: Any,
    error: type[Exception],
    message: str,
):
    graph, pseudotime, labels = _y_graph()

    with pytest.raises(error, match=message):
        compute_fate_probabilities(
            graph,
            pseudotime,
            labels,
            sinks,
        )


def test_max_iterations_limits_gmres_inner_iterations():
    n_cells = 80
    graph = diags(
        [np.ones(n_cells - 1), np.ones(n_cells - 1)],
        [-1, 1],
        shape=(n_cells, n_cells),
        format="csr",
    )
    labels = np.full(n_cells, "other", dtype=object)
    labels[0] = "A"
    labels[-1] = "B"

    with pytest.raises(RuntimeError, match="after 1 iteration"):
        compute_fate_probabilities(
            graph,
            np.linspace(0.0, 1.0, n_cells),
            labels,
            ["A", "B"],
            beta=0.0,
            solver_tol=1e-12,
            max_iterations=1,
        )


def test_loose_solver_tolerance_does_not_bypass_output_validation(
    monkeypatch: pytest.MonkeyPatch,
):
    from scarf.trajectory import fate as fate_module

    graph, pseudotime, labels = _y_graph()

    def invalid_solution(operator: Any, *_args: Any, **_kwargs: Any):
        return np.full(operator.shape[0], 2.0), 0

    monkeypatch.setattr(fate_module, "gmres", invalid_solution)
    with pytest.raises(RuntimeError, match="numerical bounds"):
        compute_fate_probabilities(
            graph,
            pseudotime,
            labels,
            ["A", "B"],
            solver_tol=0.9,
        )


def test_localized_solver_error_fails_residual_validation(
    monkeypatch: pytest.MonkeyPatch,
):
    from scarf.trajectory import fate as fate_module

    n_cells = 203
    adjacency = np.zeros((n_cells, n_cells), dtype=np.float64)
    adjacency[0, 1] = adjacency[1, 0] = 1.0
    adjacency[1, 2] = adjacency[2, 1] = 1.0
    labels = np.empty(n_cells, dtype=object)
    labels[:3] = ["A", "middle", "B"]
    labels[3:103] = "A"
    labels[103:] = "B"
    pseudotime = np.ones(n_cells, dtype=np.float64)
    pseudotime[1] = 0.0
    original_gmres = fate_module.gmres

    def localized_error(*args: Any, **kwargs: Any):
        solution, info = original_gmres(*args, **kwargs)
        solution[1] += 0.01
        return solution, info

    monkeypatch.setattr(fate_module, "gmres", localized_error)
    with pytest.raises(RuntimeError, match="residual"):
        compute_fate_probabilities(
            csr_matrix(adjacency),
            pseudotime,
            labels,
            ["A", "B"],
            beta=0.0,
            solver_tol=1e-3,
        )


class _FateCells:
    def __init__(self, pseudotime: np.ndarray, labels: np.ndarray):
        self.values: dict[str, np.ndarray] = {
            "I": np.ones(pseudotime.shape[0], dtype=bool),
            "ptime": pseudotime,
            "state": labels,
        }
        self.insertions: list[str] = []

    @property
    def columns(self) -> list[str]:
        return list(self.values)

    def fetch(self, column: str, key: str = "I") -> np.ndarray:
        return np.asarray(self.values[column][self.values[key].astype(bool)])

    def fetch_all(self, column: str) -> np.ndarray:
        return np.asarray(self.values[column])

    def active_index(self, key: str) -> np.ndarray:
        return np.flatnonzero(self.values[key])

    def insert(
        self,
        column_name: str,
        values: np.ndarray,
        fill_value: Any = np.nan,
        key: str = "I",
        overwrite: bool = False,
    ) -> None:
        assert overwrite
        selected = self.values[key].astype(bool)
        if np.asarray(values).dtype.kind == "b":
            filled = np.full(selected.shape[0], bool(fill_value), dtype=bool)
        else:
            filled = np.full(selected.shape[0], fill_value, dtype=np.float32)
        filled[selected] = values
        self.values[column_name] = filled
        self.insertions.append(column_name)


class _FateStore:
    def __init__(self):
        self.graph, pseudotime, labels = _y_graph()
        self.cells = _FateCells(pseudotime, labels)
        self.assay = SimpleNamespace(name="RNA", cells=self.cells)
        self.graph_loads = 0

    @staticmethod
    def _get_latest_keys(
        _from_assay: str | None,
        _cell_key: str | None,
        _feat_key: str | None,
    ) -> tuple[str, str, str]:
        return "RNA", "I", "I"

    def _get_assay(self, _from_assay: str) -> SimpleNamespace:
        return self.assay

    def load_graph(self, **_kwargs: Any) -> csr_matrix:
        self.graph_loads += 1
        return self.graph.copy()

    @staticmethod
    def _col_renamer(from_assay: str, cell_key: str, suffix: str) -> str:
        if cell_key == "I":
            return f"{from_assay}_{suffix}"
        return f"{from_assay}_{cell_key}_{suffix}"


def test_datastore_method_writes_aligned_fate_and_validity_columns():
    store = _FateStore()

    result = GraphDataStore.run_fate_mapping(
        store,
        pseudotime_key="ptime",
        sink_key="state",
        sinks=["A", "B"],
    )

    assert isinstance(result, FateMappingResult)
    assert result.fate_keys == ("RNA_fate_A", "RNA_fate_B")
    assert result.validity_key == "RNA_fate__valid"
    assert result.sink_labels == ("A", "B")
    assert store.cells.insertions == [
        "RNA_fate_A",
        "RNA_fate_B",
        "RNA_fate__valid",
    ]
    np.testing.assert_allclose(
        np.column_stack([store.cells.fetch_all(key) for key in result.fate_keys]),
        result.values,
    )
    np.testing.assert_array_equal(
        store.cells.fetch_all(result.validity_key),
        result.valid,
    )


def test_datastore_method_aligns_subset_outputs_to_all_cells():
    store = _FateStore()
    store.cells.values["subset"] = np.array([True, True, True, False, False])
    store.cells.values["ptime"][3:] = np.nan

    result = GraphDataStore.run_fate_mapping(
        store,
        subset_cell_key="subset",
        pseudotime_key="ptime",
        sink_key="state",
        sinks=["A"],
    )

    assert result.result_cell_key == "subset"
    assert result.values.shape == (3, 1)
    stored = store.cells.fetch_all(result.fate_keys[0])
    np.testing.assert_array_equal(stored[:3], np.ones(3))
    assert np.isnan(stored[3:]).all()
    np.testing.assert_array_equal(
        store.cells.fetch_all(result.validity_key),
        [True, True, True, False, False],
    )


@pytest.mark.parametrize("sinks", [[], ("A",)])
def test_datastore_rejects_invalid_sink_container_before_loading_graph(
    sinks: Any,
):
    store = _FateStore()

    with pytest.raises((TypeError, ValueError)):
        GraphDataStore.run_fate_mapping(
            store,
            pseudotime_key="ptime",
            sink_key="state",
            sinks=sinks,
        )

    assert store.graph_loads == 0


def test_failed_solver_writes_no_metadata(monkeypatch: pytest.MonkeyPatch):
    from scarf.datastore._operations import trajectory as trajectory_operations

    store = _FateStore()

    def fail(*_args: Any, **_kwargs: Any):
        raise RuntimeError("forced non-convergence")

    monkeypatch.setattr(
        trajectory_operations,
        "_compute_fate_probabilities_impl",
        fail,
    )
    with pytest.raises(RuntimeError, match="forced non-convergence"):
        GraphDataStore.run_fate_mapping(
            store,
            pseudotime_key="ptime",
            sink_key="state",
            sinks=["A", "B"],
        )

    assert store.cells.insertions == []
