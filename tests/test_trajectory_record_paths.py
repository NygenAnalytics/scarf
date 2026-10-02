"""Persisted trajectory records are validated before they are loaded or reused.

Most tests store a small trajectory, change one persisted fact or pass one
inconsistent input, and check that the public operation, or the payload
validator it relies on, refuses it instead of loading, reusing or writing it.
The others pin the record and payload forms that validation must accept.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
import zarr
from loguru import logger
from scipy.sparse import block_diag, csr_matrix
from zarr.storage import MemoryStore

from scarf.metadata import MetaData
from scarf.metadata.artifacts import plan_cell_data_artifact, write_cell_data_artifact
from scarf.storage.artifacts import (
    ArtifactRef,
    artifact_group,
    fingerprint_stored_arrays,
    fingerprint_stored_strings,
    list_artifacts,
    new_artifact_id,
)
from scarf.storage.selections import (
    resolve_metadata_snapshot,
    resolve_stored_selection_artifact,
)
from scarf.trajectory.artifacts import (
    AGGREGATION_PAYLOAD,
    FATE_PAYLOAD,
    MARKER_PAYLOAD,
    PSEUDOTIME_PAYLOAD,
    aggregation_payload_is_valid,
    fate_payload_is_valid,
    marker_payload_is_valid,
    pseudotime_payload_is_valid,
    true_array_indices,
    validate_aggregation_parameters,
    validate_fate_parameters,
    validate_marker_parameters,
)
from scarf.trajectory.parameters import resolve_aggregation_ann_params

from .test_graph_coverage import _memory_graph_store
from .test_graph_feature_projection import _native_chain

_Y_EDGES = ((0, 1), (1, 2), (0, 3), (3, 4))
_Y_LABELS = np.array(["root", "a-mid", "A", "b-mid", "B"])
_Y_SOURCE_SINK = np.array([-1.0, 0.0, 0.6, 0.0, 0.4])


def _adjacency(n_cells: int, edges: Any) -> np.ndarray:
    adjacency = np.zeros((n_cells, n_cells), dtype=np.float64)
    for first, second in edges:
        adjacency[first, second] = adjacency[second, first] = 1.0
    return adjacency


def _y_graph() -> np.ndarray:
    return _adjacency(5, _Y_EDGES)


def _path(n_cells: int) -> np.ndarray:
    return _adjacency(n_cells, [(cell, cell + 1) for cell in range(n_cells - 1)])


def _cell_selection(store: Any, operation: str) -> ArtifactRef:
    return resolve_stored_selection_artifact(
        store.zw,
        table_path="cellData",
        id_column="ids",
        source_column="I",
        scope="datastore",
        kind="cell_selection",
        operation=operation,
        parameters={},
        inputs={},
    )


def _write_graph_payload(store: Any, graph: ArtifactRef, adjacency: np.ndarray) -> None:
    rows, cols = np.nonzero(adjacency)
    group = artifact_group(store.zw, graph)
    group.create_array("edges", data=np.column_stack((rows, cols)).astype(np.uint64))
    group.create_array("weights", data=adjacency[rows, cols])
    group.attrs.update({"n_cells": adjacency.shape[0], "n_neighbors": 2})


@dataclass(frozen=True)
class _Trajectory:
    store: Any
    cell_ids: np.ndarray
    selection: ArtifactRef
    other_selection: ArtifactRef
    graphs: dict[str, ArtifactRef]

    @property
    def graph(self) -> ArtifactRef:
        return self.graphs["RNA"]

    def labels(
        self,
        values: Any,
        *,
        selection: ArtifactRef | None = None,
    ) -> ArtifactRef:
        return resolve_metadata_snapshot(
            self.store.zw,
            values=np.asarray(values),
            row_ids=self.cell_ids,
            operation="test_labels",
            parameters={},
            inputs={"cell_selection": selection or self.selection},
            source_columns=["label"],
        )

    def artifacts(self, kind: str) -> list[ArtifactRef]:
        return list_artifacts(self.store.zw, scope="assay", assay="RNA", kind=kind)


def _trajectory_store(
    adjacency: np.ndarray,
    *,
    assays: tuple[str, ...] = ("RNA",),
) -> _Trajectory:
    """Store one native graph per assay over one shared cell selection.

    A second selection holds the same cells under a different identity, so
    an artifact aligned to it has the right length but the wrong lineage.
    """
    n_cells = adjacency.shape[0]
    store = _memory_graph_store()
    table = store.z.create_group("cellData")
    cell_ids = np.array([f"c{cell}" for cell in range(n_cells)])
    table.create_array("ids", data=cell_ids)
    table.create_array("I", data=np.ones(n_cells, dtype=bool))
    store.cells = MetaData(table)
    selection = _cell_selection(store, "test_selection")
    graphs: dict[str, ArtifactRef] = {}
    for assay in assays:
        graph, _neighbors, _coordinates = _native_chain(
            store.zw,
            assay,
            cell_selection=selection,
        )
        _write_graph_payload(store, graph, adjacency)
        graphs[assay] = graph
    return _Trajectory(
        store=store,
        cell_ids=cell_ids,
        selection=selection,
        other_selection=_cell_selection(store, "test_other_selection"),
        graphs=graphs,
    )


def _embedding(trajectory: _Trajectory) -> ArtifactRef:
    """Store a two-dimensional embedding, laid out as UMAP and t-SNE store theirs."""
    n_cells = len(trajectory.cell_ids)
    planned = plan_cell_data_artifact(
        trajectory.store.zw,
        scope="assay",
        assay="RNA",
        kind="embedding",
        operation="test_embedding",
        parameters={},
        inputs={},
        execution_options={},
        cell_selection=trajectory.selection,
        arrays={"values": ((n_cells, 2), "f")},
    )
    write_cell_data_artifact(
        trajectory.store.zw,
        planned,
        {"values": np.zeros((n_cells, 2))},
    )
    return planned.ref


def _missing(kind: str) -> ArtifactRef:
    return ArtifactRef(
        scope="assay",
        assay="RNA",
        kind=kind,
        artifact_id=new_artifact_id(),
    )


def _rewrite_record(store: Any, ref: ArtifactRef, **fields: Any) -> None:
    group = artifact_group(store.zw, ref)
    group.attrs["provenance"] = {**group.attrs["provenance"], **fields}


def _rewrite_parameters(store: Any, ref: ArtifactRef, **parameters: Any) -> None:
    recorded = artifact_group(store.zw, ref).attrs["provenance"]["parameters"]
    _rewrite_record(store, ref, parameters={**recorded, **parameters})


def _rewrite_inputs(store: Any, ref: ArtifactRef, **inputs: ArtifactRef) -> None:
    recorded = artifact_group(store.zw, ref).attrs["provenance"]["inputs"]
    _rewrite_record(
        store,
        ref,
        inputs={
            **recorded,
            **{name: value.to_dict() for name, value in inputs.items()},
        },
    )


def _resize_graph(trajectory: _Trajectory, n_cells: int) -> None:
    artifact_group(trajectory.store.zw, trajectory.graph).attrs["n_cells"] = n_cells


def _replace_array(group: zarr.Group, name: str, values: np.ndarray) -> None:
    del group[name]
    group.create_array(name, data=values)


# Diffusion operator


def test_diffusion_operator_refuses_graphs_it_cannot_align() -> None:
    trajectory = _trajectory_store(_y_graph())
    store = trajectory.store

    with pytest.raises(TypeError, match="graph must be an ArtifactRef"):
        store.run_diffusion_operator("graph")
    with pytest.raises(ValueError, match="Graph artifact is unavailable or incomplete"):
        store.run_diffusion_operator(_missing("connectivity_map"))
    _resize_graph(trajectory, 6)
    with pytest.raises(
        ValueError,
        match="Graph cell count does not match its stored cell selection",
    ):
        store.run_diffusion_operator(trajectory.graph)

    assert trajectory.artifacts("diffusion_operator") == []


def test_diffusion_loader_requires_a_complete_diffusion_artifact() -> None:
    trajectory = _trajectory_store(_y_graph())
    store = trajectory.store

    with pytest.raises(TypeError, match="diffusion must be an ArtifactRef"):
        store.load_diffusion_operator("diffusion")
    with pytest.raises(ValueError, match="must be a diffusion_operator artifact"):
        store.load_diffusion_operator(trajectory.graph)
    with pytest.raises(ValueError, match="artifact is unavailable or incomplete"):
        store.load_diffusion_operator(_missing("diffusion_operator"))


_DIFFUSION_RECORD_TAMPERS: dict[
    str,
    tuple[Callable[[_Trajectory, ArtifactRef], None], str],
] = {
    "foreign-operation": (
        lambda trajectory, ref: _rewrite_record(
            trajectory.store, ref, operation="run_imputation"
        ),
        "was not produced by run_diffusion_operator",
    ),
    "extra-parameter": (
        lambda trajectory, ref: _rewrite_parameters(trajectory.store, ref, steps=1),
        "Diffusion-operator parameters are malformed",
    ),
    "zero-power": (
        lambda trajectory, ref: _rewrite_parameters(trajectory.store, ref, t=0),
        "Diffusion-operator power is malformed",
    ),
    "missing-selection-input": (
        lambda trajectory, ref: _rewrite_record(
            trajectory.store,
            ref,
            inputs={"connectivity_map": trajectory.graph.to_dict()},
        ),
        "Diffusion-operator lineage inputs are malformed",
    ),
    "graph-of-another-assay": (
        lambda trajectory, ref: _rewrite_inputs(
            trajectory.store, ref, connectivity_map=trajectory.graphs["ADT"]
        ),
        "scope does not match its graph input",
    ),
    "deleted-graph": (
        lambda trajectory, ref: _rewrite_inputs(
            trajectory.store, ref, connectivity_map=_missing("connectivity_map")
        ),
        "Diffusion-operator graph is unavailable or incomplete",
    ),
    "resized-graph": (
        lambda trajectory, _ref: _resize_graph(trajectory, 6),
        "graph count does not match its cell selection",
    ),
}


@pytest.mark.parametrize("tamper", sorted(_DIFFUSION_RECORD_TAMPERS))
def test_diffusion_loader_rejects_each_tampered_record(tamper: str) -> None:
    trajectory = _trajectory_store(_y_graph(), assays=("RNA", "ADT"))
    diffusion = trajectory.store.run_diffusion_operator(trajectory.graph, t=1)
    change, message = _DIFFUSION_RECORD_TAMPERS[tamper]
    change(trajectory, diffusion)

    with pytest.raises(ValueError, match=message):
        trajectory.store.load_diffusion_operator(diffusion)


def _narrow_diffusion_values(group: zarr.Group) -> None:
    _replace_array(group, "data", np.asarray(group["data"][:], dtype=np.float32))


_DIFFUSION_PAYLOAD_TAMPERS: dict[str, Callable[[zarr.Group], None]] = {
    "unknown-attribute": lambda group: group.attrs.update({"schema_version": 1}),
    "blank-fingerprint": lambda group: group.attrs.update({"payload_fingerprint": ""}),
    "annotated-array": lambda group: group["row"].attrs.update({"unit": "cell"}),
    "float32-values": _narrow_diffusion_values,
}


@pytest.mark.parametrize("tamper", sorted(_DIFFUSION_PAYLOAD_TAMPERS))
def test_tampered_diffusion_payload_is_neither_loaded_nor_reused(tamper: str) -> None:
    trajectory = _trajectory_store(_y_graph())
    store = trajectory.store
    diffusion = store.run_diffusion_operator(trajectory.graph, t=1)
    expected = store.load_diffusion_operator(diffusion).toarray()
    _DIFFUSION_PAYLOAD_TAMPERS[tamper](artifact_group(store.zw, diffusion))

    with pytest.raises(ValueError, match="sparse payload is malformed"):
        store.load_diffusion_operator(diffusion)
    recomputed = store.run_diffusion_operator(trajectory.graph, t=1)
    assert recomputed != diffusion
    np.testing.assert_array_equal(
        store.load_diffusion_operator(recomputed).toarray(),
        expected,
    )


# Pseudotime scoring


def _labels_argument(trajectory: _Trajectory) -> ArtifactRef:
    return trajectory.labels(_Y_LABELS)


_PSEUDOTIME_ARGUMENT_ERRORS: dict[
    str,
    tuple[Callable[[_Trajectory], dict[str, Any]], type[Exception], str],
] = {
    "no-source-sink": (
        lambda trajectory: {},
        ValueError,
        "Provide source/sink labels or a custom zero-sum ss_vec",
    ),
    "labels-without-artifact": (
        lambda trajectory: {"sources": ["root"]},
        ValueError,
        "source_sink is required when sources or sinks are provided",
    ),
    "vector-and-labels": (
        lambda trajectory: {"ss_vec": _Y_SOURCE_SINK, "sinks": ["A"]},
        ValueError,
        "Provide either ss_vec or source_sink",
    ),
    "artifact-without-labels": (
        lambda trajectory: {"source_sink": _labels_argument(trajectory)},
        ValueError,
        "At least one source or sink label must be provided",
    ),
    "tuple-sources": (
        lambda trajectory: {
            "source_sink": _labels_argument(trajectory),
            "sources": ("root",),
        },
        TypeError,
        "sources must be a list",
    ),
    "tuple-sinks": (
        lambda trajectory: {
            "source_sink": _labels_argument(trajectory),
            "sinks": ("A",),
        },
        TypeError,
        "sinks must be a list",
    ),
    "unknown-component-policy": (
        lambda trajectory: {"ss_vec": _Y_SOURCE_SINK, "component_policy": "smallest"},
        ValueError,
        "component_policy must be 'largest' or 'error'",
    ),
}


@pytest.mark.parametrize("case", sorted(_PSEUDOTIME_ARGUMENT_ERRORS))
def test_pseudotime_scoring_rejects_ambiguous_source_sink_arguments(case: str) -> None:
    trajectory = _trajectory_store(_y_graph())
    arguments, error, message = _PSEUDOTIME_ARGUMENT_ERRORS[case]

    with pytest.raises(error, match=message):
        trajectory.store.run_pseudotime_scoring(
            trajectory.graph,
            **arguments(trajectory),
        )
    assert trajectory.artifacts("pseudotime") == []


def _truncated_labels(trajectory: _Trajectory) -> ArtifactRef:
    labels = trajectory.labels(_Y_LABELS)
    group = artifact_group(trajectory.store.zw, labels)
    _replace_array(group, "values", np.asarray(group["values"][:-1]))
    return labels


_SOURCE_SINK_INPUTS: dict[
    str,
    tuple[Callable[[_Trajectory], Any], type[Exception], str],
] = {
    "column-name": (
        lambda trajectory: "clusters",
        TypeError,
        "cell data input must be an ArtifactRef",
    ),
    "missing-artifact": (
        lambda trajectory: _missing("cluster_labels"),
        ValueError,
        "Cell-data artifact is unavailable or incomplete",
    ),
    "diffusion-operator": (
        lambda trajectory: trajectory.store.run_diffusion_operator(
            trajectory.graph, t=1
        ),
        ValueError,
        "diffusion_operator artifact has no 'values' cell-data array",
    ),
    "truncated-labels": (
        _truncated_labels,
        ValueError,
        "Cell-data artifact values do not match their selection",
    ),
    "labels-of-another-selection": (
        lambda trajectory: trajectory.labels(
            _Y_LABELS, selection=trajectory.other_selection
        ),
        ValueError,
        "Source/sink labels do not match the graph selection",
    ),
    "two-values-per-cell": (
        _embedding,
        ValueError,
        "Source/sink labels do not align with graph rows",
    ),
}


@pytest.mark.parametrize("case", sorted(_SOURCE_SINK_INPUTS))
def test_pseudotime_scoring_requires_one_label_per_graph_cell(case: str) -> None:
    trajectory = _trajectory_store(_y_graph())
    source_sink, error, message = _SOURCE_SINK_INPUTS[case]

    with pytest.raises(error, match=message):
        trajectory.store.run_pseudotime_scoring(
            trajectory.graph,
            source_sink=source_sink(trajectory),
            sources=["root"],
            sinks=["A", "B"],
        )
    assert trajectory.artifacts("pseudotime") == []


def _empty_graph_payload(trajectory: _Trajectory) -> None:
    group = artifact_group(trajectory.store.zw, trajectory.graph)
    _replace_array(group, "edges", np.empty((0, 2), dtype=np.uint64))
    _replace_array(group, "weights", np.empty(0, dtype=np.float64))
    _resize_graph(trajectory, 0)


def test_pseudotime_scoring_requires_a_graph_of_at_least_four_cells() -> None:
    triangle = _trajectory_store(_adjacency(3, [(0, 1), (1, 2), (0, 2)]))
    with pytest.raises(ValueError, match="must contain at least 4 cells"):
        triangle.store.run_pseudotime_scoring(
            triangle.graph,
            ss_vec=np.array([-1.0, 0.0, 1.0]),
        )
    assert triangle.artifacts("pseudotime") == []

    # A graph payload emptied after its lineage was written has no cells left.
    emptied = _trajectory_store(_y_graph())
    _empty_graph_payload(emptied)
    with pytest.raises(ValueError, match="No cells were selected for pseudotime"):
        emptied.store.run_pseudotime_scoring(emptied.graph, ss_vec=_Y_SOURCE_SINK)
    with pytest.raises(TypeError, match="graph must be an ArtifactRef"):
        emptied.store.run_pseudotime_scoring("graph", ss_vec=_Y_SOURCE_SINK)
    assert emptied.artifacts("pseudotime") == []


def test_pseudotime_scoring_refuses_a_graph_larger_than_its_selection() -> None:
    trajectory = _trajectory_store(_y_graph())
    # A sixth, isolated graph cell has no row in the five-cell selection.
    _resize_graph(trajectory, 6)

    with pytest.raises(
        ValueError,
        match="Graph cell count does not match its stored cell selection",
    ):
        trajectory.store.run_pseudotime_scoring(
            trajectory.graph,
            ss_vec=_Y_SOURCE_SINK,
        )
    assert trajectory.artifacts("pseudotime") == []


def test_pseudotime_scoring_keeps_requested_modes_and_raw_potential() -> None:
    trajectory = _trajectory_store(_y_graph())
    store = trajectory.store
    messages: list[str] = []
    sink = logger.add(
        lambda message: messages.append(message.record["message"]),
        level="WARNING",
    )
    try:
        normalized = store.run_pseudotime_scoring(
            trajectory.graph,
            ss_vec=_Y_SOURCE_SINK,
        )
        raw = store.run_pseudotime_scoring(
            trajectory.graph,
            ss_vec=_Y_SOURCE_SINK,
            min_max_norm_ptime=False,
        )
    finally:
        logger.remove(sink)

    # Five cells support three singular modes, so the default 30 is reduced.
    assert "Reducing n_singular_vals from 30 to 3 for the retained graph size" in (
        messages
    )
    parameters = store.inspect_artifact(raw).parameters or {}
    assert parameters["n_singular_vals"] == 30
    assert parameters["min_max_norm_ptime"] is False
    assert raw != normalized
    raw_values = store.load_pseudotime_scoring(raw).values
    # The raw potential is centred, so it spans zero instead of [0, 1].
    assert raw_values.min() < 0.0 < raw_values.max()
    np.testing.assert_allclose(
        (raw_values - raw_values.min()) / np.ptp(raw_values),
        store.load_pseudotime_scoring(normalized).values,
        rtol=0.0,
        atol=1e-12,
    )


_TWO_COMPONENTS = block_diag((csr_matrix(_y_graph()), csr_matrix(_path(2)))).toarray()


@pytest.mark.parametrize(
    ("adjacency", "ss_vec", "message"),
    [
        pytest.param(
            _TWO_COMPONENTS,
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0]),
            "Pseudotime calculation produced a constant potential",
            id="source-sink-outside-retained-component",
        ),
        # The source/sink validator accepts any finite zero-sum vector, so a
        # potential beyond the float64 range must be refused before writing.
        pytest.param(
            _path(4),
            np.array([-1e308, 0.0, 0.0, 1e308]),
            "Pseudotime calculation produced non-finite values",
            id="overflowing-potential",
        ),
        pytest.param(
            _path(4),
            np.array([-7.5e307, 0.0, 0.0, 7.5e307]),
            "Pseudotime normalization produced non-finite values",
            id="overflowing-potential-range",
        ),
    ],
)
def test_pseudotime_scoring_refuses_degenerate_potentials(
    adjacency: np.ndarray,
    ss_vec: np.ndarray,
    message: str,
) -> None:
    trajectory = _trajectory_store(adjacency)

    with (
        np.errstate(over="ignore", invalid="ignore"),
        pytest.raises(ValueError, match=message),
    ):
        trajectory.store.run_pseudotime_scoring(
            trajectory.graph,
            ss_vec=ss_vec,
            n_singular_vals=2,
        )
    assert trajectory.artifacts("pseudotime") == []


# Pseudotime loading


@dataclass(frozen=True)
class _ScoredTrajectory:
    trajectory: _Trajectory
    labels: ArtifactRef
    from_labels: ArtifactRef
    from_vector: ArtifactRef

    @property
    def store(self) -> Any:
        return self.trajectory.store


def _scored_trajectory() -> _ScoredTrajectory:
    trajectory = _trajectory_store(_y_graph(), assays=("RNA", "ADT"))
    labels = trajectory.labels(_Y_LABELS)
    return _ScoredTrajectory(
        trajectory=trajectory,
        labels=labels,
        from_labels=trajectory.store.run_pseudotime_scoring(
            trajectory.graph,
            source_sink=labels,
            sources=["root"],
            sinks=["A", "B"],
            n_singular_vals=3,
        ),
        from_vector=trajectory.store.run_pseudotime_scoring(
            trajectory.graph,
            ss_vec=_Y_SOURCE_SINK,
            n_singular_vals=3,
        ),
    )


def test_pseudotime_loader_requires_a_complete_pseudotime_artifact() -> None:
    scored = _scored_trajectory()

    with pytest.raises(TypeError, match="ref must be an ArtifactRef"):
        scored.store.load_pseudotime_scoring("pseudotime")
    with pytest.raises(ValueError, match="ref must be a pseudotime artifact"):
        scored.store.load_pseudotime_scoring(scored.labels)
    with pytest.raises(ValueError, match="Pseudotime artifact is unavailable"):
        scored.store.load_pseudotime_scoring(_missing("pseudotime"))


def _reverse_source_sink_snapshot(scored: _ScoredTrajectory) -> None:
    inputs = scored.store.inspect_artifact(scored.from_vector).inputs or {}
    snapshot = artifact_group(
        scored.store.zw,
        ArtifactRef.from_dict(inputs["source_sink"]),
    )
    snapshot["values"][:] = -np.asarray(snapshot["values"][:])


def _alter_pseudotime_values(scored: _ScoredTrajectory) -> None:
    values = artifact_group(scored.store.zw, scored.from_labels)["pseudotime"]
    values[0] = float(values[0]) + 0.25


_PSEUDOTIME_RECORD_TAMPERS: dict[
    str,
    tuple[str, Callable[[_ScoredTrajectory], None], str],
] = {
    "malformed-parameters": (
        "from_labels",
        lambda scored: _rewrite_parameters(
            scored.store, scored.from_labels, n_singular_vals=1
        ),
        "Pseudotime parameters are malformed",
    ),
    "other-cell-selection": (
        "from_labels",
        lambda scored: _rewrite_inputs(
            scored.store,
            scored.from_labels,
            cell_selection=scored.trajectory.other_selection,
        ),
        "Pseudotime graph and cell selection do not match",
    ),
    "graph-of-another-assay": (
        "from_labels",
        lambda scored: _rewrite_inputs(
            scored.store,
            scored.from_labels,
            connectivity_map=scored.trajectory.graphs["ADT"],
        ),
        "Pseudotime artifact does not share its graph scope",
    ),
    "resized-graph": (
        "from_labels",
        lambda scored: _resize_graph(scored.trajectory, 6),
        "Pseudotime graph does not match its stored selection",
    ),
    "altered-values": (
        "from_labels",
        _alter_pseudotime_values,
        "Pseudotime artifact payload is invalid",
    ),
    "labels-of-another-selection": (
        "from_labels",
        lambda scored: _rewrite_inputs(
            scored.store,
            scored.from_labels,
            source_sink=scored.trajectory.labels(
                _Y_LABELS, selection=scored.trajectory.other_selection
            ),
        ),
        "Pseudotime source/sink input uses a different selection",
    ),
    "reversed-source-sink-vector": (
        "from_vector",
        _reverse_source_sink_snapshot,
        "Pseudotime source/sink snapshot is invalid",
    ),
}


@pytest.mark.parametrize("tamper", sorted(_PSEUDOTIME_RECORD_TAMPERS))
def test_pseudotime_loader_rejects_each_tampered_record(tamper: str) -> None:
    scored = _scored_trajectory()
    name, change, message = _PSEUDOTIME_RECORD_TAMPERS[tamper]
    change(scored)

    with pytest.raises(ValueError, match=message):
        scored.store.load_pseudotime_scoring(getattr(scored, name))


# Fate mapping


@dataclass(frozen=True)
class _MappedTrajectory:
    scored: _ScoredTrajectory
    fate: ArtifactRef

    @property
    def store(self) -> Any:
        return self.scored.store

    @property
    def trajectory(self) -> _Trajectory:
        return self.scored.trajectory


def _mapped_trajectory() -> _MappedTrajectory:
    scored = _scored_trajectory()
    return _MappedTrajectory(
        scored=scored,
        fate=scored.store.run_fate_mapping(
            scored.from_vector,
            scored.labels,
            sinks=["A", "B"],
        ),
    )


def test_fate_mapping_requires_sink_labels_for_the_pseudotime_cells() -> None:
    scored = _scored_trajectory()
    store = scored.store
    pseudotime = scored.from_vector

    with pytest.raises(ValueError, match="sinks must be provided"):
        store.run_fate_mapping(pseudotime, scored.labels)
    with pytest.raises(TypeError, match="pseudotime must be an ArtifactRef"):
        store.run_fate_mapping("pseudotime", scored.labels, sinks=["A"])
    with pytest.raises(TypeError, match="sink_labels must be an ArtifactRef"):
        store.run_fate_mapping(pseudotime, "labels", sinks=["A"])
    with pytest.raises(
        ValueError,
        match="Sink labels do not match the pseudotime cell selection",
    ):
        store.run_fate_mapping(
            pseudotime,
            scored.trajectory.labels(
                _Y_LABELS, selection=scored.trajectory.other_selection
            ),
            sinks=["A"],
        )
    with pytest.raises(ValueError, match="Sink labels must be one-dimensional"):
        store.run_fate_mapping(pseudotime, _embedding(scored.trajectory), sinks=[0.0])
    assert scored.trajectory.artifacts("fate_map") == []


def test_fate_loader_requires_a_complete_fate_artifact() -> None:
    mapped = _mapped_trajectory()

    with pytest.raises(TypeError, match="ref must be an ArtifactRef"):
        mapped.store.load_fate_mapping("fate")
    with pytest.raises(ValueError, match="ref must be a fate_map artifact"):
        mapped.store.load_fate_mapping(mapped.scored.from_vector)
    with pytest.raises(ValueError, match="Fate-map artifact is unavailable"):
        mapped.store.load_fate_mapping(_missing("fate_map"))


def _move_fate_to_another_assay(mapped: _MappedTrajectory) -> None:
    other_graph = mapped.trajectory.graphs["ADT"]
    other_pseudotime = mapped.store.run_pseudotime_scoring(
        other_graph,
        ss_vec=_Y_SOURCE_SINK,
        n_singular_vals=3,
    )
    _rewrite_inputs(
        mapped.store,
        mapped.fate,
        connectivity_map=other_graph,
        pseudotime=other_pseudotime,
    )


def _alter_fate_probabilities(mapped: _MappedTrajectory) -> None:
    probabilities = artifact_group(mapped.store.zw, mapped.fate)["probabilities"]
    probabilities[1] = np.array([0.25, 0.75], dtype=probabilities.dtype)


_FATE_RECORD_TAMPERS: dict[str, tuple[Callable[[_MappedTrajectory], None], str]] = {
    "malformed-parameters": (
        lambda mapped: _rewrite_parameters(mapped.store, mapped.fate, beta=-1.0),
        "Fate-map parameters are malformed",
    ),
    "graph-of-another-assay": (
        lambda mapped: _rewrite_inputs(
            mapped.store,
            mapped.fate,
            connectivity_map=mapped.trajectory.graphs["ADT"],
        ),
        "Fate-map lineage does not match its pseudotime",
    ),
    "lineage-of-another-assay": (
        _move_fate_to_another_assay,
        "Fate-map artifact does not share its graph scope",
    ),
    "labels-of-another-selection": (
        lambda mapped: _rewrite_inputs(
            mapped.store,
            mapped.fate,
            sink_labels=mapped.trajectory.labels(
                _Y_LABELS, selection=mapped.trajectory.other_selection
            ),
        ),
        "Fate-map sink labels use a different cell selection",
    ),
    "unknown-sink": (
        lambda mapped: _rewrite_parameters(mapped.store, mapped.fate, sinks=["A", "Z"]),
        "Fate-map sink labels are malformed",
    ),
    "altered-probabilities": (
        _alter_fate_probabilities,
        "Fate-map artifact payload is invalid",
    ),
}


@pytest.mark.parametrize("tamper", sorted(_FATE_RECORD_TAMPERS))
def test_fate_loader_rejects_each_tampered_record(tamper: str) -> None:
    mapped = _mapped_trajectory()
    change, message = _FATE_RECORD_TAMPERS[tamper]
    change(mapped)

    with pytest.raises(ValueError, match=message):
        mapped.store.load_fate_mapping(mapped.fate)


# Pseudotime feature analyses


def test_marker_loader_rejects_altered_correlations(
    datastore,
    pseudotime_markers,
) -> None:
    r_values = artifact_group(datastore.zw, pseudotime_markers)["r_value"]
    tested = int(np.flatnonzero(np.isfinite(r_values[:]))[0])
    original = float(r_values[tested])
    try:
        r_values[tested] = 0.25 if original == 0.5 else 0.5
        with pytest.raises(
            ValueError,
            match="Pseudotime-marker artifact payload is invalid",
        ):
            datastore.load_pseudotime_markers(pseudotime_markers)
    finally:
        r_values[tested] = original
    assert datastore.load_pseudotime_markers(pseudotime_markers).ref == (
        pseudotime_markers
    )


def test_marker_search_stores_no_correlation_that_is_not_finite(
    datastore,
    pseudotime_scoring,
    detected_features,
    monkeypatch,
) -> None:
    import scarf.features.markers as marker_algorithms

    original_search = marker_algorithms.find_markers_by_regression

    def overflowing_search(*args: Any, **kwargs: Any) -> Any:
        # A raw pseudotime near the float64 limit overflows the Pearson sums,
        # and the kernel then reports NaN for the feature.
        markers = original_search(*args, **kwargs)
        markers.iloc[0, markers.columns.get_loc("r_value")] = np.nan
        return markers

    monkeypatch.setattr(
        marker_algorithms,
        "find_markers_by_regression",
        overflowing_search,
    )
    before = set(datastore.list_artifacts(kind="pseudotime_markers", from_assay="RNA"))

    with pytest.raises(
        ValueError,
        match="Pseudotime marker correlations are missing or not finite",
    ):
        datastore.run_pseudotime_marker_search(
            pseudotime_scoring,
            features=detected_features,
            invalidate_cache=True,
        )
    assert (
        set(datastore.list_artifacts(kind="pseudotime_markers", from_assay="RNA"))
        == before
    )


def test_aggregation_loader_rejects_unresolved_ann_parameters(
    datastore,
    pseudotime_aggregation,
) -> None:
    group = artifact_group(datastore.zw, pseudotime_aggregation)
    original = group.attrs["provenance"]
    try:
        # Records always hold the resolved HNSW settings, never an empty request.
        group.attrs["provenance"] = {
            **original,
            "parameters": {**original["parameters"], "ann_params": {}},
        }
        with pytest.raises(
            ValueError,
            match="Pseudotime-aggregation ANN parameters are malformed",
        ):
            datastore.load_pseudotime_aggregation(pseudotime_aggregation)
    finally:
        group.attrs["provenance"] = original
    assert datastore.load_pseudotime_aggregation(pseudotime_aggregation).ref == (
        pseudotime_aggregation
    )


def test_aggregation_needs_more_features_than_its_feature_graph_uses(
    datastore,
    pseudotime_scoring,
) -> None:
    single = datastore.set_feature_selection(from_assay="RNA", feature_indexes=[0])
    triple = datastore.set_feature_selection(
        from_assay="RNA",
        feature_indexes=[0, 1, 2],
    )
    before = set(
        datastore.list_artifacts(kind="pseudotime_aggregation", from_assay="RNA")
    )

    for features, options, message in (
        (single, {}, "At least two selected features are required"),
        (
            triple,
            {"n_neighbours": 3},
            "n_neighbours must be smaller than the selected feature count",
        ),
        (
            triple,
            {"n_neighbours": 2, "n_clusters": 4},
            "n_clusters cannot exceed the selected feature count",
        ),
    ):
        with pytest.raises(ValueError, match=message):
            datastore.run_pseudotime_aggregation(
                pseudotime_scoring,
                features=features,
                **options,
            )
    assert (
        set(datastore.list_artifacts(kind="pseudotime_aggregation", from_assay="RNA"))
        == before
    )


# Persisted parameter records


_FATE_RECORD = {
    "sinks": ["A"],
    "beta": 10.0,
    "solver_tol": 1e-6,
    "max_iterations": 1000,
}
_MARKER_RECORD = {
    "normalization": {"log_transform": False, "renormalize_subset": False},
    "normalization_method": {
        "module": "scarf.assay.normalization",
        "qualname": "norm_lib_size",
    },
    "size_factor": 1000.0,
    "association_method": "pearson",
    "p_value_method": "student_t",
    "adjustment_method": "fdr_bh",
    "adjustment_scope": "tested_features",
    "min_cells": 10,
}
_AGGREGATION_RECORD = {
    "normalization": _MARKER_RECORD["normalization"],
    "normalization_method": _MARKER_RECORD["normalization_method"],
    "size_factor": 1000.0,
    "min_exp": 1e-3,
    "window_size": 20,
    "chunk_size": 10,
    "smoothen": True,
    "z_scale": True,
    "n_neighbours": 2,
    "n_clusters": 3,
    "ann_params": {},
    "nan_cluster_value": -1,
}


@pytest.mark.parametrize(
    ("changes", "error", "message"),
    [
        pytest.param(
            {"beta": "fast"}, TypeError, "beta must be a real number", id="text-beta"
        ),
        pytest.param(
            {"beta": np.inf}, ValueError, "beta must be finite", id="infinite-beta"
        ),
        pytest.param(
            {"solver_tol": 1.0},
            ValueError,
            "solver_tol must be less than 1.0",
            id="unit-tolerance",
        ),
        pytest.param(
            {"sinks": "A"}, TypeError, "sinks must be a list", id="text-sinks"
        ),
        pytest.param(
            {"sinks": [np.nan]},
            ValueError,
            "sinks must contain finite scalar labels",
            id="nan-sink",
        ),
        pytest.param(
            {"sinks": [None]},
            TypeError,
            "sinks must contain scalar labels",
            id="none-sink",
        ),
        pytest.param(
            {"sinks": []},
            ValueError,
            "sinks must contain at least one label",
            id="no-sinks",
        ),
    ],
)
def test_fate_record_rejects_each_malformed_parameter(
    changes: dict[str, Any],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        validate_fate_parameters({**_FATE_RECORD, **changes})


@pytest.mark.parametrize(
    ("changes", "error", "message"),
    [
        pytest.param(
            {"normalization": {"log_transform": False}},
            ValueError,
            "normalization parameters are malformed",
            id="partial-normalization",
        ),
        pytest.param(
            {"normalization_method": "norm_lib_size"},
            TypeError,
            "normalization_method must be a mapping",
            id="method-name",
        ),
        pytest.param(
            {"normalization_method": {"module": "scarf.assay.normalization"}},
            ValueError,
            "normalization_method is malformed",
            id="method-without-qualname",
        ),
        pytest.param(
            {"normalization_method": {"identity": ""}},
            ValueError,
            "normalization_method is malformed",
            id="blank-identity",
        ),
    ],
)
def test_marker_record_rejects_malformed_normalization(
    changes: dict[str, Any],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        validate_marker_parameters({**_MARKER_RECORD, **changes})


def test_marker_record_of_an_assay_without_size_factor_is_valid() -> None:
    # Only RNA assays set a size factor; other assays record None.
    validated = validate_marker_parameters({**_MARKER_RECORD, "size_factor": None})

    assert validated["size_factor"] is None


def test_aggregation_record_accepts_only_supported_ann_spaces() -> None:
    cosine = validate_aggregation_parameters(
        {**_AGGREGATION_RECORD, "ann_params": {"space": "cosine", "M": 16}}
    )

    assert cosine["ann_params"] == {"space": "cosine", "M": 16}
    with pytest.raises(ValueError, match="ann_params.space is unsupported"):
        validate_aggregation_parameters(
            {**_AGGREGATION_RECORD, "ann_params": {"space": "hamming"}}
        )


# Stored payloads


_RECORD_ATTRIBUTES = {
    "artifact_id": "a" * 64,
    "kind": "test_payload",
    "provenance": {},
    "execution_options": {},
    "created_at_ns": 1,
    "scarf_version": "test",
    "complete": True,
}


def _refresh_fingerprint(group: zarr.Group, names: tuple[str, ...]) -> None:
    group.attrs["payload_fingerprint"] = fingerprint_stored_arrays(group, names)


def _payload_group(
    payload: dict[str, Any],
    names: tuple[str, ...],
    *,
    chunk_rows: int | None = None,
    **attributes: Any,
) -> zarr.Group:
    """Store a complete candidate payload fingerprinted over ``names``."""
    group = zarr.open_group(store=MemoryStore(), mode="w").create_group("candidate")
    for name, raw in payload.items():
        values = np.asarray(raw)
        chunks = "auto" if chunk_rows is None else (chunk_rows, *values.shape[1:])
        group.create_array(name, data=values, chunks=chunks)
    group.attrs.update({**_RECORD_ATTRIBUTES, **attributes})
    _refresh_fingerprint(group, names)
    return group


def test_true_indices_are_gathered_across_blocks_without_true_values() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    valid = root.create_array(
        "valid",
        data=np.array([True, False, False, False, True]),
        chunks=(2,),
    )

    indices = true_array_indices(valid)

    np.testing.assert_array_equal(indices, [0, 4])
    assert indices.dtype == np.int64


def test_pseudotime_payload_validation_spans_blocks_without_scored_cells() -> None:
    valid = np.array([True, True, False, False, True, True])
    group = _payload_group(
        {
            "pseudotime": np.array([0.0, 0.4, np.nan, np.nan, 0.6, 1.0]),
            "valid": valid,
        },
        PSEUDOTIME_PAYLOAD,
        chunk_rows=2,
    )
    arguments = {"n_cells": 6, "min_max_normalized": True, "expected_valid": valid}
    assert pseudotime_payload_is_valid(group, **arguments)

    # An unscored cell must keep its NaN placeholder.
    group["pseudotime"][2] = 0.5
    _refresh_fingerprint(group, PSEUDOTIME_PAYLOAD)
    assert not pseudotime_payload_is_valid(group, **arguments)


def _replace_and_refresh(
    names: tuple[str, ...],
    name: str,
    values: np.ndarray,
) -> Callable[[zarr.Group], None]:
    def replace(group: zarr.Group) -> None:
        _replace_array(group, name, values)
        _refresh_fingerprint(group, names)

    return replace


def _set_and_refresh(
    names: tuple[str, ...],
    name: str,
    index: int,
    value: Any,
) -> Callable[[zarr.Group], None]:
    def change(group: zarr.Group) -> None:
        group[name][index] = value
        _refresh_fingerprint(group, names)

    return change


def _delete_validity(group: zarr.Group) -> None:
    del group["valid"]


def _validity_as_group(group: zarr.Group) -> None:
    del group["valid"]
    group.create_group("valid")


_PSEUDOTIME_ARRAY_TAMPERS: dict[str, Callable[[zarr.Group], None]] = {
    "missing-validity": _delete_validity,
    "validity-group": _validity_as_group,
    "short-values": _replace_and_refresh(
        PSEUDOTIME_PAYLOAD, "pseudotime", np.array([0.0, 1.0])
    ),
    "integer-validity": _replace_and_refresh(
        PSEUDOTIME_PAYLOAD, "valid", np.ones(3, dtype=np.int8)
    ),
}


@pytest.mark.parametrize("tamper", sorted(_PSEUDOTIME_ARRAY_TAMPERS))
def test_pseudotime_payload_requires_its_exact_arrays(tamper: str) -> None:
    valid = np.ones(3, dtype=bool)
    group = _payload_group(
        {"pseudotime": np.array([0.0, 0.5, 1.0]), "valid": valid},
        PSEUDOTIME_PAYLOAD,
    )
    arguments = {"n_cells": 3, "min_max_normalized": True, "expected_valid": valid}
    assert pseudotime_payload_is_valid(group, **arguments)

    _PSEUDOTIME_ARRAY_TAMPERS[tamper](group)
    assert not pseudotime_payload_is_valid(group, **arguments)


_FATE_LABELS = np.array(["A", "mid", "B", "mid", "A", "B"], dtype=object)
_FATE_VALID = np.ones(6, dtype=bool)
_FATE_PROBABILITIES = np.array(
    [[1.0, 0.0], [0.4, 0.6], [0.0, 1.0], [0.7, 0.3], [1.0, 0.0], [0.0, 1.0]],
    dtype=np.float32,
)


def _fate_group(*, chunk_rows: int | None = None) -> zarr.Group:
    return _payload_group(
        {"probabilities": _FATE_PROBABILITIES, "valid": _FATE_VALID},
        FATE_PAYLOAD,
        chunk_rows=chunk_rows,
    )


def _fate_is_valid(group: zarr.Group, sink_values: np.ndarray = _FATE_LABELS) -> bool:
    return fate_payload_is_valid(
        group,
        n_cells=6,
        n_sinks=2,
        pseudotime_valid=_FATE_VALID,
        sink_values=sink_values,
        sink_labels=["A", "B"],
    )


def test_fate_payload_validation_spans_blocks_without_boundary_cells() -> None:
    # The first two-row block holds no B cell and the second no A cell.
    assert _fate_is_valid(_fate_group(chunk_rows=2))


def test_fate_payload_requires_a_valid_boundary_cell_for_every_sink() -> None:
    relabelled = np.where(_FATE_LABELS == "B", "C", _FATE_LABELS)

    assert not _fate_is_valid(_fate_group(), sink_values=relabelled)


_FATE_PAYLOAD_TAMPERS: dict[str, Callable[[zarr.Group], None]] = {
    "unknown-attribute": lambda group: group.attrs.update({"schema_version": 1}),
    "extra-array": lambda group: group.create_array("notes", data=np.zeros(6)),
    # A scalar array cannot be fingerprinted, so the comparison fails closed.
    "scalar-validity": lambda group: _replace_array(group, "valid", np.array(True)),
    "third-sink-column": _replace_and_refresh(
        FATE_PAYLOAD,
        "probabilities",
        np.hstack([_FATE_PROBABILITIES, np.zeros((6, 1), dtype=np.float32)]),
    ),
    "shifted-validity": _replace_and_refresh(
        FATE_PAYLOAD, "valid", np.array([True] * 5 + [False])
    ),
    "unnormalized-row": _set_and_refresh(
        FATE_PAYLOAD, "probabilities", 1, np.array([0.4, 0.7], dtype=np.float32)
    ),
}


@pytest.mark.parametrize("tamper", sorted(_FATE_PAYLOAD_TAMPERS))
def test_fate_payload_rejects_each_tampered_array(tamper: str) -> None:
    group = _fate_group()
    assert _fate_is_valid(group)

    _FATE_PAYLOAD_TAMPERS[tamper](group)
    assert not _fate_is_valid(group)


def _marker_payload() -> tuple[zarr.Group, Callable[[], bool]]:
    group = _payload_group(
        {
            "r_value": np.array([0.5, np.nan, -0.4]),
            "p_value": np.array([0.01, np.nan, 0.02]),
            "p_value_adjusted": np.array([0.02, np.nan, 0.03]),
            "feature_names": np.array(["g0", "g1", "g2"]),
            "feature_ids": np.array(["id0", "id1", "id2"]),
        },
        MARKER_PAYLOAD,
    )
    ids = fingerprint_stored_strings(group["feature_ids"])
    names = fingerprint_stored_strings(group["feature_names"])

    def is_valid() -> bool:
        return marker_payload_is_valid(
            group,
            n_features=3,
            selected_features=np.array([0, 2]),
            expected_feature_ids_fingerprint=ids,
            expected_feature_names_fingerprint=names,
        )

    return group, is_valid


_MARKER_PAYLOAD_TAMPERS: dict[str, Callable[[zarr.Group], None]] = {
    "unknown-attribute": lambda group: group.attrs.update({"schema_version": 1}),
    "integer-correlations": _replace_and_refresh(
        MARKER_PAYLOAD, "r_value", np.array([1, 0, -1])
    ),
    "undecodable-ids": _replace_and_refresh(
        MARKER_PAYLOAD,
        "feature_ids",
        np.array([b"\xff", b"id1", b"id2"], dtype="S3"),
    ),
    "correlation-above-one": _set_and_refresh(MARKER_PAYLOAD, "r_value", 0, 1.5),
    "p-value-above-one": _set_and_refresh(MARKER_PAYLOAD, "p_value", 2, 1.5),
}


# Byte-string identities exercise decoding, which Zarr warns has no V3 spec.
@pytest.mark.filterwarnings("ignore::zarr.errors.UnstableSpecificationWarning")
@pytest.mark.parametrize("tamper", sorted(_MARKER_PAYLOAD_TAMPERS))
def test_marker_payload_rejects_each_tampered_array(tamper: str) -> None:
    group, is_valid = _marker_payload()
    assert is_valid()

    _MARKER_PAYLOAD_TAMPERS[tamper](group)
    assert not is_valid()


def _aggregation_payload(
    *,
    chunk_rows: int | None = None,
) -> tuple[zarr.Group, Callable[[], bool]]:
    """Store features 0, 2 and 3 of four, with feature 2 excluded from modules."""
    group = _payload_group(
        {
            "data": np.array([[0.0, 1.0], [0.0, 0.0], [1.0, 0.0]]),
            "feature_indices": np.array([0, 2, 3], dtype=np.uint64),
            "valid_features": np.array([True, False, True]),
            "feature_clusters": np.array([1, -1, 2]),
            "cluster_values": np.array([1, -1, -1, 2]),
            "feature_names": np.array(["g0", "g1", "g2", "g3"]),
            "feature_ids": np.array(["id0", "id1", "id2", "id3"]),
        },
        AGGREGATION_PAYLOAD,
        chunk_rows=chunk_rows,
        input_fingerprints=["cells", "features", "ordering"],
        nan_cluster_value=-1,
        effective_window=2,
        effective_bins=2,
    )
    ids = fingerprint_stored_strings(group["feature_ids"])
    names = fingerprint_stored_strings(group["feature_names"])

    def is_valid() -> bool:
        return aggregation_payload_is_valid(
            group,
            n_features=4,
            selected_features=np.array([0, 2, 3]),
            n_bins=2,
            n_clusters=2,
            n_neighbours=1,
            nan_cluster_value=-1,
            ann_params=resolve_aggregation_ann_params(None, dim=2),
            expected_input_fingerprints=["cells", "features", "ordering"],
            expected_feature_ids_fingerprint=ids,
            expected_feature_names_fingerprint=names,
            effective_window=2,
        )

    return group, is_valid


def test_aggregation_payload_validation_spans_unselected_feature_blocks() -> None:
    # One-row blocks put the unselected feature 1 in a block of its own.
    _group, is_valid = _aggregation_payload(chunk_rows=1)

    assert is_valid()


def _keep_one_valid_feature(group: zarr.Group) -> None:
    group["valid_features"][2] = False
    group["data"][2] = np.zeros(2)
    group["feature_clusters"][2] = -1
    group["cluster_values"][3] = -1
    _refresh_fingerprint(group, AGGREGATION_PAYLOAD)


_AGGREGATION_PAYLOAD_TAMPERS: dict[str, Callable[[zarr.Group], None]] = {
    "unknown-attribute": lambda group: group.attrs.update({"schema_version": 1}),
    "signed-feature-indices": _replace_and_refresh(
        AGGREGATION_PAYLOAD, "feature_indices", np.array([0, 2, 3], dtype=np.int64)
    ),
    "other-input-fingerprints": lambda group: group.attrs.update(
        {"input_fingerprints": ["cells", "features", "reordered"]}
    ),
    "other-unassigned-value": lambda group: group.attrs.update(
        {"nan_cluster_value": -2}
    ),
    "other-window": lambda group: group.attrs.update({"effective_window": 3}),
    "other-bins": lambda group: group.attrs.update({"effective_bins": 3}),
    "undecodable-names": _replace_and_refresh(
        AGGREGATION_PAYLOAD,
        "feature_names",
        np.array([b"g0", b"\xff", b"g2", b"g3"], dtype="S2"),
    ),
    "renamed-feature": _replace_and_refresh(
        AGGREGATION_PAYLOAD,
        "feature_names",
        np.array(["g0", "renamed", "g2", "g3"]),
    ),
    "values-for-excluded-feature": _set_and_refresh(
        AGGREGATION_PAYLOAD, "data", 1, np.array([0.5, 0.0])
    ),
    "one-valid-feature": _keep_one_valid_feature,
    "relabelled-module": _set_and_refresh(AGGREGATION_PAYLOAD, "cluster_values", 0, 2),
}


@pytest.mark.filterwarnings("ignore::zarr.errors.UnstableSpecificationWarning")
@pytest.mark.parametrize("tamper", sorted(_AGGREGATION_PAYLOAD_TAMPERS))
def test_aggregation_payload_rejects_each_tampered_record(tamper: str) -> None:
    group, is_valid = _aggregation_payload()
    assert is_valid()

    _AGGREGATION_PAYLOAD_TAMPERS[tamper](group)
    assert not is_valid()
