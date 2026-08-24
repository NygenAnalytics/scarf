from dataclasses import fields
from importlib.util import find_spec
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from scarf.datastore._operations import trajectory as trajectory_operations
from scarf.datastore._operations.trajectory import (
    _TrajectoryOperationsMixin,
    _group_assignment_digest,
    _validate_assay_pseudotime,
)
from scarf.embeddings.sgtsne import run_sgtsne
from scarf.neighbors.stream import AnnStream
from scarf.trajectory.feature_dynamics import validate_pseudotime_regressor
from scarf.trajectory.results import PseudotimeScoreResult
from tests.signature_contracts import signature_digest


def test_embedding_and_trajectory_entry_point_signatures_are_stable():
    methods = {
        "AnnStream.__init__": AnnStream.__init__,
        "run_sgtsne": run_sgtsne,
        "validate_pseudotime_regressor": validate_pseudotime_regressor,
    }
    assert signature_digest(methods) == (
        "b963e7139e72eea4843182060350a8c907e973014df48a84a13806ec6b435dfd"
    )
    assert [field.name for field in fields(PseudotimeScoreResult)] == [
        "pseudotime_key",
        "validity_key",
        "assay",
        "graph_cell_key",
        "result_cell_key",
        "feature_key",
        "values",
        "valid",
    ]


def test_moved_symbols_are_absent_from_old_hybrid_modules():
    from scarf.features import markers
    from scarf.datastore import datastore, graph_datastore

    assert find_spec("scarf.knn_utils") is None
    retired = {
        markers: {"knn_clustering"},
        datastore: {
            "_scatter_feature_clusters",
            "_validated_pseudotime_regressor",
        },
        graph_datastore: {
            "_make_source_sink_vector",
            "_random_walk_laplacian_transpose",
            "_select_pseudotime_component",
            "_truncated_pba_potential",
            "_validate_source_sink_labels",
            "_validate_source_sink_vector",
        },
    }
    for module, names in retired.items():
        assert names.isdisjoint(vars(module))


class _TrajectoryCells:
    def __init__(self, values):
        self.values = {name: np.asarray(value) for name, value in values.items()}
        self.insertions = []

    @property
    def columns(self):
        return list(self.values)

    def fetch(self, column, key="I"):
        return self.values[column][self.values[key].astype(bool)]

    def fetch_all(self, column):
        return self.values[column]

    def active_index(self, key):
        return np.flatnonzero(self.values[key])

    def insert(self, column, values, **_kwargs):
        self.values[column] = np.asarray(values)
        self.insertions.append(column)


class _TrajectoryValidationStore:
    def __init__(self, values, graph=None):
        self.cells = _TrajectoryCells(values)
        self.assay = SimpleNamespace(cells=self.cells)
        self.graph = graph

    @staticmethod
    def _get_latest_keys(_from_assay, _cell_key, _feat_key):
        return "RNA", "I", "I"

    def _get_assay(self, _from_assay):
        return self.assay

    def load_graph(self, **_kwargs):
        return self.graph.copy()

    @staticmethod
    def _col_renamer(from_assay, cell_key, suffix):
        return f"{from_assay}_{cell_key}_{suffix}"


def _chain_graph(size):
    graph = np.zeros((size, size), dtype=float)
    for index in range(size - 1):
        graph[index, index + 1] = 1.0
        graph[index + 1, index] = 1.0
    return csr_matrix(graph)


def test_trajectory_group_digest_is_deterministic_and_order_sensitive():
    assignments = np.array(["group-1", "group-2"])

    assert _group_assignment_digest(assignments) == _group_assignment_digest(
        assignments.copy()
    )
    assert _group_assignment_digest(assignments) != _group_assignment_digest(
        assignments[::-1]
    )


def test_assay_pseudotime_wraps_metadata_conversion_errors():
    class InvalidCells:
        columns = []

        @staticmethod
        def fetch(_column, key):
            raise ValueError(f"invalid key {key}")

    assay = SimpleNamespace(cells=InvalidCells())
    with pytest.raises(TypeError, match="'ptime' must be numeric"):
        _validate_assay_pseudotime(assay, "I", "ptime")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"sink_key": "state", "sinks": ["A"]}, "pseudotime_key"),
        ({"pseudotime_key": "ptime", "sinks": ["A"]}, "sink_key"),
        ({"pseudotime_key": "ptime", "sink_key": "state"}, "sinks"),
        (
            {
                "pseudotime_key": "ptime",
                "sink_key": "state",
                "sinks": ["A"],
                "label": 1,
            },
            "label must be a string",
        ),
        (
            {
                "pseudotime_key": "ptime",
                "sink_key": "state",
                "sinks": ["A"],
                "label": "",
            },
            "label must not be empty",
        ),
    ],
)
def test_fate_operation_rejects_missing_or_invalid_output_arguments(kwargs, message):
    with pytest.raises((TypeError, ValueError), match=message):
        _TrajectoryOperationsMixin.run_fate_mapping(object(), **kwargs)


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (
            {
                "I": np.array([True, True]),
                "subset": np.array([1, 0]),
            },
            "boolean values",
        ),
        (
            {
                "I": np.array([True, False, True]),
                "subset": np.array([True, True, False]),
            },
            "complete subset",
        ),
        (
            {
                "I": np.array([True, True]),
                "subset": np.array([False, False]),
            },
            "No cells were selected",
        ),
    ],
)
def test_fate_operation_validates_subset_before_loading_assay(values, message):
    store = _TrajectoryValidationStore(values)

    with pytest.raises((TypeError, ValueError), match=message):
        _TrajectoryOperationsMixin.run_fate_mapping(
            store,
            subset_cell_key="subset",
            pseudotime_key="ptime",
            sink_key="state",
            sinks=["A"],
        )


def test_pseudotime_operation_rejects_incomplete_subset_before_graph_loading():
    store = _TrajectoryValidationStore(
        {
            "I": np.array([True, False, True]),
            "subset": np.array([True, True, False]),
        }
    )

    with pytest.raises(ValueError, match="complete subset"):
        _TrajectoryOperationsMixin.run_pseudotime_scoring(
            store,
            subset_cell_key="subset",
        )


@pytest.mark.parametrize(
    ("size", "kwargs", "message"),
    [
        (0, {}, "No cells were selected"),
        (4, {"n_singular_vals": True}, "must be an integer"),
        (4, {"n_singular_vals": 1}, "at least 2"),
        (3, {"n_singular_vals": 2}, "at least 4 cells"),
        (
            4,
            {"n_singular_vals": 2, "sources": ["A"]},
            "source_sink_key is required",
        ),
        (
            4,
            {
                "n_singular_vals": 2,
                "source_sink_key": "state",
                "sources": ["A"],
                "ss_vec": np.array([-1.0, 0.0, 0.0, 1.0]),
            },
            "either ss_vec",
        ),
        (
            4,
            {"n_singular_vals": 2},
            "Provide source/sink labels",
        ),
    ],
)
def test_pseudotime_operation_validates_small_graph_and_source_inputs(
    monkeypatch,
    size,
    kwargs,
    message,
):
    values = {
        "I": np.ones(size, dtype=bool),
        "state": np.array(["A", "middle", "middle", "B"][:size]),
    }
    store = _TrajectoryValidationStore(values, graph=_chain_graph(size))
    monkeypatch.setattr(
        trajectory_operations,
        "_stored_graph_input",
        lambda *_args: ("graph", object()),
    )
    monkeypatch.setattr(
        trajectory_operations,
        "validate_legacy_graph_selection",
        lambda *_args: None,
    )

    with pytest.raises((TypeError, ValueError), match=message):
        _TrajectoryOperationsMixin.run_pseudotime_scoring(store, **kwargs)


def test_fate_operation_rejects_backend_label_mismatch_before_writes(monkeypatch):
    size = 3
    values = {
        "I": np.ones(size, dtype=bool),
        "ptime": np.linspace(0.0, 1.0, size),
        "state": np.array(["A", "middle", "A"]),
    }
    store = _TrajectoryValidationStore(values, graph=_chain_graph(size))
    monkeypatch.setattr(
        trajectory_operations,
        "_compute_fate_probabilities_impl",
        lambda *_args, **_kwargs: (
            np.ones((size, 1), dtype=np.float32),
            np.ones(size, dtype=bool),
            ("different",),
        ),
    )

    with pytest.raises(ValueError, match="do not match requested"):
        _TrajectoryOperationsMixin.run_fate_mapping(
            store,
            pseudotime_key="ptime",
            sink_key="state",
            sinks=["A"],
        )

    assert store.cells.insertions == []
