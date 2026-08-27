from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix

from ...assay import Assay
from ...assay.normalization import reject_unknown_normalization_params
from ...graph.feature_projection import (
    graph_cell_selection,
    resolve_graph_source_assay,
)
from ...matrix import ChunkedArray
from ...metadata.rows import read_metadata_rows_chunkwise
from ...metadata.arguments import (
    FateMappingArguments,
    PseudotimeAggregationArguments,
    PseudotimeMarkerArguments,
    PseudotimeScoringArguments,
)
from ...metadata.artifacts import plan_cell_data_artifact, write_cell_data_artifact
from ...storage.artifact_writer import (
    ArrayRequirement,
    AttributeRequirement,
    finish_artifact,
    plan_artifact,
    reused_artifact_group,
    start_artifact,
)
from ...storage.artifacts import (
    ArtifactRef,
    artifact_path,
    callable_identity,
    inspect_artifact,
)
from ...storage.types import as_zarr_array, as_zarr_group
from ...storage.selections import (
    read_stored_selection_indices,
    validate_stored_selection_integrity,
)
from ...storage.arrays import create_zarr_dataset
from ...trajectory.feature_dynamics import (
    scatter_feature_clusters as _scatter_feature_clusters_impl,
    validate_pseudotime_regressor,
)
from ...trajectory.fate import (
    compute_fate_probabilities as _compute_fate_probabilities_impl,
)
from ...trajectory.pseudotime import (
    make_source_sink_vector as _make_source_sink_vector_impl,
    random_walk_laplacian_transpose as _random_walk_laplacian_transpose_impl,
    select_pseudotime_component as _select_pseudotime_component_impl,
    truncated_pba_potential as _truncated_pba_potential_impl,
    validate_source_sink_labels as _validate_source_sink_labels_impl,
    validate_source_sink_vector as _validate_source_sink_vector_impl,
)
from ...trajectory.results import (
    FateMappingResult,
    PseudotimeAggregationResult,
    PseudotimeMarkerResult,
    PseudotimeScoreResult,
)
from ...utils.compute import controlled_compute
from ...utils.logging import logger

if TYPE_CHECKING:
    from ..mapping_datastore import (
        MappingDatastore as _TrajectoryFeatureOperationsBase,
    )
    from .graph import _GraphOperationsMixin as _TrajectoryOperationsBase
else:
    _TrajectoryFeatureOperationsBase = object
    _TrajectoryOperationsBase = object


_CELL_VALUE_NAMES = {
    "cell_cycle": "phase",
    "cluster_cut": "labels",
    "pseudotime": "pseudotime",
}


def _artifact_ref_input(raw: Any, label: str) -> ArtifactRef:
    if not isinstance(raw, dict):
        raise ValueError(f"{label} artifact reference is malformed")
    return ArtifactRef.from_dict(raw)


def _diffusion_power(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int | np.integer):
        raise TypeError("t must be a positive integer")
    power = int(value)
    if power < 1:
        raise ValueError("t must be a positive integer")
    return power


def _load_cell_artifact_values(
    root: Any,
    ref: ArtifactRef,
    *,
    value_name: str | None = None,
) -> tuple[np.ndarray, ArtifactRef]:
    if not isinstance(ref, ArtifactRef):
        raise TypeError("cell data input must be an ArtifactRef")
    status = inspect_artifact(root, ref)
    if not status.complete:
        raise ValueError("Cell-data artifact is unavailable or incomplete")
    raw_selection = (status.inputs or {}).get("cell_selection")
    if not isinstance(raw_selection, dict):
        raise ValueError("Cell-data artifact has no cell-selection input")
    selection = ArtifactRef.from_dict(raw_selection)
    validate_stored_selection_integrity(
        root,
        selection,
        kind="cell_selection",
        scope="datastore",
        assay=None,
        table_path="cellData",
    )
    canonical_name = value_name or _CELL_VALUE_NAMES.get(ref.kind, "values")
    group = as_zarr_group(root[status.path], name=status.path)
    if canonical_name not in group:
        raise ValueError(
            f"{ref.kind} artifact has no {canonical_name!r} cell-data array"
        )
    values = np.asarray(as_zarr_array(group[canonical_name], name=canonical_name)[:])
    if values.ndim < 1:
        raise ValueError("Cell-data artifact values must have a row axis")
    return values, selection


def _resolve_feature_indices(
    store: Any,
    assay: Assay,
    features: ArtifactRef,
) -> tuple[ArtifactRef, np.ndarray]:
    if not isinstance(features, ArtifactRef):
        raise TypeError("features must be an ArtifactRef")
    feature_selection = store.resolve_features(assay.name, features)
    selection_group = as_zarr_group(
        store.zw[artifact_path(feature_selection)],
        name=artifact_path(feature_selection),
    )
    values = np.asarray(
        as_zarr_array(selection_group["values"], name="values")[:],
        dtype=bool,
    )
    feature_indices = np.flatnonzero(values).astype(np.int64, copy=False)
    if len(feature_indices) == 0:
        raise ValueError("Feature selection contains no active features")
    return feature_selection, feature_indices


class _TrajectoryOperationsMixin(_TrajectoryOperationsBase):
    def run_diffusion_operator(
        self,
        graph: ArtifactRef,
        *,
        t: int = 2,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Compute and persist a sparse MAGIC diffusion operator.

        The returned artifact is aligned to the graph's exact stored cell
        selection. Repeating the same graph and diffusion power reuses the
        complete artifact unless ``invalidate_cache`` is true.
        """
        from ...neighbors.diffusion import diffusion_operator

        if not isinstance(graph, ArtifactRef):
            raise TypeError("graph must be an ArtifactRef")
        power = _diffusion_power(t)
        if not isinstance(invalidate_cache, bool):
            raise TypeError("invalidate_cache must be a boolean")
        graph_ref = graph
        graph_status = inspect_artifact(self.zw, graph_ref)
        if not graph_status.complete:
            raise ValueError("Graph artifact is unavailable or incomplete")
        selection = graph_cell_selection(self.zw, graph_ref)
        validated_selection = validate_stored_selection_integrity(
            self.zw,
            selection,
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        )
        n_cells, _ = self._get_graph_ncells_k(graph_status.path)
        if n_cells != validated_selection.selected_count:
            raise ValueError(
                "Graph cell count does not match its stored cell selection"
            )
        planned = plan_artifact(
            self.zw,
            scope=graph_ref.scope,
            assay=graph_ref.assay,
            kind="diffusion_operator",
            operation="run_diffusion_operator",
            parameters={"t": power},
            inputs={
                "connectivity_map": graph_ref,
                "cell_selection": selection,
            },
            execution_options={"invalidate_cache": invalidate_cache},
            invalidate_cache=invalidate_cache,
            required_arrays=(
                ArrayRequirement("row", shape=(None,), dtype=np.uint64),
                ArrayRequirement("col", shape=(None,), dtype=np.uint64),
                ArrayRequirement("data", shape=(None,), dtype=np.float64),
            ),
            required_attributes=(
                AttributeRequirement(
                    "n_cells",
                    expected_types=(int, np.integer),
                    predicate=lambda value: (
                        not isinstance(value, bool) and int(value) == n_cells
                    ),
                ),
            ),
        )
        if planned.reused:
            reused_artifact_group(self.zw, planned)
            return planned.ref
        if self.zarr_mode != "r+":
            raise PermissionError(
                "run_diffusion_operator requires a DataStore opened with "
                "zarr_mode='r+' unless a matching artifact already exists"
            )

        graph_matrix = self.load_graph(
            graph_ref,
            symmetric=True,
            upper_only=False,
        )
        if graph_matrix.shape != (n_cells, n_cells):
            raise ValueError(
                "Loaded graph shape does not match its stored cell selection"
            )
        diff_op = diffusion_operator(graph_matrix, power).tocoo()
        shape = (int(diff_op.nnz),)
        store = start_artifact(self.zw, planned)
        for name, dtype in zip(
            ("row", "col", "data"),
            (np.uint64, np.uint64, np.float64),
            strict=True,
        ):
            array = create_zarr_dataset(store, name, (1000000,), dtype, shape)
            array[:] = np.asarray(getattr(diff_op, name), dtype=dtype)
        store.attrs["n_cells"] = n_cells
        finish_artifact(store, planned)
        return planned.ref

    def _load_diffusion_operator_with_lineage(
        self,
        diffusion: ArtifactRef,
    ) -> tuple[coo_matrix, ArtifactRef, ArtifactRef]:
        if not isinstance(diffusion, ArtifactRef):
            raise TypeError("diffusion must be an ArtifactRef")
        if diffusion.kind != "diffusion_operator":
            raise ValueError("diffusion must be a diffusion_operator artifact")
        status = inspect_artifact(self.zw, diffusion)
        if not status.complete:
            raise ValueError("Diffusion-operator artifact is unavailable or incomplete")
        if status.operation != "run_diffusion_operator":
            raise ValueError(
                "Diffusion-operator artifact was not produced by run_diffusion_operator"
            )
        parameters = status.parameters or {}
        if set(parameters) != {"t"}:
            raise ValueError("Diffusion-operator parameters are malformed")
        try:
            _diffusion_power(parameters["t"])
        except (TypeError, ValueError) as error:
            raise ValueError("Diffusion-operator power is malformed") from error
        inputs = status.inputs or {}
        if set(inputs) != {"connectivity_map", "cell_selection"}:
            raise ValueError("Diffusion-operator lineage inputs are malformed")
        graph = _artifact_ref_input(inputs["connectivity_map"], "Graph")
        selection = _artifact_ref_input(inputs["cell_selection"], "Cell-selection")
        if diffusion.scope != graph.scope or diffusion.assay != graph.assay:
            raise ValueError("Diffusion-operator scope does not match its graph input")
        graph_status = inspect_artifact(self.zw, graph)
        if not graph_status.complete:
            raise ValueError("Diffusion-operator graph is unavailable or incomplete")
        graph_selection = graph_cell_selection(self.zw, graph)
        if selection != graph_selection:
            raise ValueError(
                "Diffusion-operator cell selection does not match its graph lineage"
            )
        validated_selection = validate_stored_selection_integrity(
            self.zw,
            selection,
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        )
        graph_n_cells, _ = self._get_graph_ncells_k(graph_status.path)
        if graph_n_cells != validated_selection.selected_count:
            raise ValueError(
                "Diffusion-operator graph count does not match its cell selection"
            )

        group = as_zarr_group(self.zw[status.path], name=status.path)
        raw_n_cells = group.attrs.get("n_cells")
        if (
            isinstance(raw_n_cells, bool)
            or not isinstance(raw_n_cells, int | np.integer)
            or int(raw_n_cells) != graph_n_cells
        ):
            raise ValueError(
                "Diffusion-operator cell count does not match its graph lineage"
            )
        arrays = {
            name: as_zarr_array(group[name], name=name) if name in group else None
            for name in ("row", "col", "data")
        }
        if any(array is None for array in arrays.values()):
            raise ValueError("Diffusion-operator sparse payload is incomplete")
        row_array = arrays["row"]
        col_array = arrays["col"]
        data_array = arrays["data"]
        assert (
            row_array is not None and col_array is not None and data_array is not None
        )
        if (
            row_array.ndim != 1
            or col_array.ndim != 1
            or data_array.ndim != 1
            or row_array.shape != col_array.shape
            or row_array.shape != data_array.shape
            or np.dtype(row_array.dtype) != np.dtype(np.uint64)
            or np.dtype(col_array.dtype) != np.dtype(np.uint64)
            or np.dtype(data_array.dtype) != np.dtype(np.float64)
        ):
            raise ValueError("Diffusion-operator sparse payload is malformed")
        rows = np.asarray(row_array[:], dtype=np.uint64)
        cols = np.asarray(col_array[:], dtype=np.uint64)
        data = np.asarray(data_array[:], dtype=np.float64)
        if (
            not np.all(np.isfinite(data))
            or np.any(rows >= graph_n_cells)
            or np.any(cols >= graph_n_cells)
        ):
            raise ValueError("Diffusion-operator sparse payload is malformed")
        operator = coo_matrix(
            (data, (rows, cols)),
            shape=(graph_n_cells, graph_n_cells),
        )
        return operator, graph, selection

    def load_diffusion_operator(self, diffusion: ArtifactRef) -> coo_matrix:
        """Load a validated diffusion-operator artifact as a COO matrix."""
        operator, _graph, _selection = self._load_diffusion_operator_with_lineage(
            diffusion
        )
        return operator

    def get_imputed(
        self,
        feature_name: str,
        diffusion: ArtifactRef,
        *,
        from_assay: str | None = None,
    ) -> np.ndarray:
        """Impute feature values by diffusing along the KNN graph (MAGIC-style).

        Args:
            from_assay: Name of assay to be used. If no value is provided then the default assay will be used.
            feature_name: Name of the feature to be imputed.
            diffusion: Explicit diffusion-operator artifact returned by
                ``run_diffusion_operator``.

        Returns:
            An array of imputed values for the given feature

        """

        diff_op, graph, selection = self._load_diffusion_operator_with_lineage(
            diffusion
        )
        assay_name = resolve_graph_source_assay(
            self.zw,
            graph,
            from_assay,
            parameter_name="from_assay",
        )
        cell_indices = read_stored_selection_indices(
            self.zw,
            selection,
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        )
        if feature_name in self.cells.columns:
            data = read_metadata_rows_chunkwise(
                self.cells,
                feature_name,
                cell_indices,
            )
        else:
            assay = self._get_assay(assay_name)
            feature_indices = assay.feats.get_index_by([feature_name], "names")
            if len(feature_indices) == 0:
                raise ValueError(
                    f"ERROR: {feature_name} not found in {assay_name} assay."
                )
            if len(feature_indices) > 1:
                logger.warning(
                    f"Imputing the mean of {len(feature_indices)} features because "
                    f"{feature_name} is not unique."
                )
            data = controlled_compute(
                assay.normed(cell_indices, feature_indices).mean(axis=1),
                self.nthreads,
            ).astype(np.float64)
        return cast(np.ndarray, diff_op.dot(data))

    def run_pseudotime_scoring(
        self,
        graph: ArtifactRef,
        *,
        n_singular_vals: int = 30,
        source_sink: ArtifactRef | None = None,
        sources: list[Any] | None = None,
        sinks: list[Any] | None = None,
        ss_vec: np.ndarray | None = None,
        min_max_norm_ptime: bool = True,
        random_seed: int = 4444,
        component_policy: Literal["largest", "error"] = "largest",
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Calculate pseudotime and return an immutable cell artifact.

        This is a reimplementation of the population balance analysis approach
        from Weinreb et al. 2018. The graph's stored cell selection defines the
        output axis. Disconnected cells excluded by ``component_policy`` remain
        represented with NaN values and a false ``valid`` entry.

        Args:
            graph: Explicit connectivity-map or integrated-graph artifact.
            n_singular_vals: Number of the smallest singular values to save.
            source_sink: Explicit axis-aligned label artifact. Required with
                ``sources`` or ``sinks`` and omitted when ``ss_vec`` is supplied.
            sources: A list of group/cluster ids from ``source_sink`` to be treated as sources. Sources are
                     usually progenitor/precursor or other actively dividing cell states.
            sinks: A list of group/cluster ids from ``source_sink`` to be treated as sinks. Sinks are usually
                   more differentiated (or terminally differentiated) cell states.
            ss_vec: Custom source-sink value for each selected cell. It must sum
                to zero, with negative values for sources and positive values for
                sinks. This is mutually exclusive with label-based arguments.
            min_max_norm_ptime: Whether to perform min-max normalization on the final pseudotime values so that values
                                are in 0 to 1 range. (Default: True)
            random_seed: A random seed for svds (Default: 4444)
            component_policy: How to handle a disconnected selected graph. ``'largest'`` scores the largest connected
                              component and marks other selected cells as unscored. ``'error'`` raises instead.

        Returns:
            Reference to an artifact containing ``pseudotime`` and ``valid``.
        """

        if not isinstance(graph, ArtifactRef):
            raise TypeError("graph must be an ArtifactRef")
        graph_ref = graph
        stored_selection = graph_cell_selection(self.zw, graph_ref)
        validate_stored_selection_integrity(
            self.zw,
            stored_selection,
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        )

        logger.info(f"Pseudotime scoring: loading graph {graph_ref.artifact_id}")
        graph_matrix = self.load_graph(
            graph_ref,
            symmetric=True,
            upper_only=False,
        )

        if graph_matrix.shape[0] == 0:
            raise ValueError("No cells were selected for pseudotime scoring")
        selected_cell_indices = read_stored_selection_indices(
            self.zw,
            stored_selection,
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        )
        retained_mask, component_sizes = _select_pseudotime_component_impl(
            graph_matrix,
            selected_cell_indices,
            component_policy,
        )
        retained_graph = graph_matrix[retained_mask][:, retained_mask].tocsr()
        if len(component_sizes) > 1:
            logger.warning(
                f"Selected graph components have sizes {component_sizes}. "
                f"Scoring the largest component with {retained_graph.shape[0]} cells"
            )

        retained_n_cells = retained_graph.shape[0]
        if not isinstance(n_singular_vals, int) or isinstance(n_singular_vals, bool):
            raise TypeError("n_singular_vals must be an integer")
        if n_singular_vals < 2:
            raise ValueError("n_singular_vals must be at least 2")
        if retained_n_cells < 4:
            raise ValueError(
                "The retained graph must contain at least 4 cells for pseudotime scoring"
            )
        effective_k = min(n_singular_vals, retained_n_cells - 2)
        if effective_k != n_singular_vals:
            logger.warning(
                f"Reducing n_singular_vals from {n_singular_vals} to {effective_k} "
                "for the retained graph size"
            )

        label_arguments_supplied = (
            source_sink is not None or sources is not None or sinks is not None
        )
        if ss_vec is not None and label_arguments_supplied:
            raise ValueError(
                "Provide either ss_vec or source_sink with source/sink labels, not both"
            )
        if ss_vec is None and source_sink is None:
            if sources is not None or sinks is not None:
                raise ValueError(
                    "source_sink is required when sources or sinks are provided"
                )
            raise ValueError("Provide source/sink labels or a custom zero-sum ss_vec")

        if ss_vec is not None:
            full_source_sink = _validate_source_sink_vector_impl(
                ss_vec,
                graph_matrix.shape[0],
                "ss_vec",
            )
            source_sink_input: ArtifactRef | np.ndarray = full_source_sink
            retained_source_sink = _validate_source_sink_vector_impl(
                full_source_sink[retained_mask],
                retained_n_cells,
                "ss_vec restricted to the retained component",
            )
        else:
            if sources is not None and not isinstance(sources, list):
                raise TypeError("sources must be a list")
            if sinks is not None and not isinstance(sinks, list):
                raise TypeError("sinks must be a list")
            source_labels = [] if sources is None else sources
            sink_labels = [] if sinks is None else sinks
            if not source_labels and not sink_labels:
                raise ValueError("At least one source or sink label must be provided")

            assert source_sink is not None
            raw_labels, label_selection = _load_cell_artifact_values(
                self.zw,
                source_sink,
            )
            if label_selection != stored_selection:
                raise ValueError("Source/sink labels do not match the graph selection")
            if raw_labels.ndim != 1 or raw_labels.shape[0] != graph_matrix.shape[0]:
                raise ValueError("Source/sink labels do not align with graph rows")
            selected_labels = pd.Series(raw_labels)
            _validate_source_sink_labels_impl(
                selected_labels,
                source_labels,
                sink_labels,
                "the selected cells",
            )
            retained_labels = selected_labels.iloc[
                np.flatnonzero(retained_mask)
            ].reset_index(drop=True)
            _validate_source_sink_labels_impl(
                retained_labels,
                source_labels,
                sink_labels,
                "the retained connected component",
            )
            retained_source_sink = _validate_source_sink_vector_impl(
                _make_source_sink_vector_impl(
                    retained_labels,
                    source_labels,
                    sink_labels,
                ),
                retained_n_cells,
                "generated source/sink vector",
            )
            source_sink_input = source_sink

        arguments = PseudotimeScoringArguments(
            connectivity_map=graph_ref,
            source_sink=source_sink_input,
            cell_selection=stored_selection,
            n_singular_vals=n_singular_vals,
            sources=tuple([] if sources is None else sources),
            sinks=tuple([] if sinks is None else sinks),
            min_max_norm_ptime=min_max_norm_ptime,
            random_seed=random_seed,
            component_policy=component_policy,
            invalidate_cache=invalidate_cache,
        )
        record = arguments.to_record()
        artifact_scope = graph_ref.scope
        planned = plan_cell_data_artifact(
            self.zw,
            scope=artifact_scope,
            assay=graph_ref.assay,
            kind=arguments.artifact_kind,
            operation=arguments.operation,
            parameters=record.parameters,
            inputs=record.inputs,
            execution_options=record.execution_options,
            cell_selection=stored_selection,
            arrays={
                "pseudotime": ((graph_matrix.shape[0],), "f"),
                "valid": ((graph_matrix.shape[0],), "b"),
            },
            invalidate_cache=invalidate_cache,
        )
        if planned.reused:
            return planned.ref

        logger.debug("Pseudotime scoring: constructing Laplacian")
        laplacian_transpose = _random_walk_laplacian_transpose_impl(retained_graph)
        retained_ptime = _truncated_pba_potential_impl(
            laplacian_transpose,
            effective_k,
            random_seed,
            retained_source_sink,
        )
        if not np.isfinite(retained_ptime).all():
            raise ValueError("Pseudotime calculation produced non-finite values")
        value_range = float(np.ptp(retained_ptime))
        potential_scale = max(1.0, float(np.abs(retained_ptime).max()))
        if value_range <= np.finfo(float).eps * potential_scale:
            raise ValueError("Pseudotime calculation produced a constant potential")
        if min_max_norm_ptime:
            retained_ptime = (retained_ptime - retained_ptime.min()) / value_range
            retained_ptime = np.clip(retained_ptime, 0.0, 1.0)
            if not np.isfinite(retained_ptime).all():
                raise ValueError("Pseudotime normalization produced non-finite values")

        ptime = np.full(graph_matrix.shape[0], np.nan, dtype=float)
        ptime[retained_mask] = retained_ptime
        write_cell_data_artifact(
            self.zw,
            planned,
            {
                "pseudotime": ptime,
                "valid": retained_mask,
            },
        )
        if not retained_mask.all():
            logger.warning("Unscored cells contain NaN pseudotime in the artifact")
        logger.info(
            f"Stored pseudotime scores for "
            f"{int(retained_mask.sum())}/{len(retained_mask)} cells"
        )
        return planned.ref

    def load_pseudotime_scoring(
        self,
        ref: ArtifactRef,
    ) -> PseudotimeScoreResult:
        """Load pseudotime values from an explicit completed artifact."""
        if not isinstance(ref, ArtifactRef):
            raise TypeError("ref must be an ArtifactRef")
        if ref.kind != "pseudotime":
            raise ValueError("ref must be a pseudotime artifact")
        status = inspect_artifact(self.zw, ref)
        if not status.complete or status.operation != "run_pseudotime_scoring":
            raise ValueError("Pseudotime artifact is unavailable or invalid")
        raw_graph = (status.inputs or {}).get("connectivity_map")
        raw_selection = (status.inputs or {}).get("cell_selection")
        if not isinstance(raw_graph, dict) or not isinstance(raw_selection, dict):
            raise ValueError("Pseudotime artifact lineage is malformed")
        graph = ArtifactRef.from_dict(raw_graph)
        selection = ArtifactRef.from_dict(raw_selection)
        values, loaded_selection = _load_cell_artifact_values(
            self.zw,
            ref,
            value_name="pseudotime",
        )
        valid, valid_selection = _load_cell_artifact_values(
            self.zw,
            ref,
            value_name="valid",
        )
        if loaded_selection != selection or valid_selection != selection:
            raise ValueError("Pseudotime arrays do not match their stored selection")
        return PseudotimeScoreResult(
            ref=ref,
            graph=graph,
            cell_selection=selection,
            values=np.asarray(values, dtype=np.float64),
            valid=np.asarray(valid, dtype=bool),
        )

    def run_fate_mapping(
        self,
        pseudotime: ArtifactRef,
        sink_labels: ArtifactRef,
        *,
        sinks: list[Any] | None = None,
        beta: float = 10.0,
        solver_tol: float = 1e-6,
        max_iterations: int = 1000,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Compute fate probabilities from explicit pseudotime and labels.

        Args:
            pseudotime: Explicit pseudotime artifact. Its graph lineage is used.
            sink_labels: Explicit axis-aligned cell-label artifact.
            sinks: Ordered sink labels. Every matching selected cell becomes a
                   fate boundary.
            beta: Strength of the penalty applied to backward graph edges.
            solver_tol: Relative tolerance used by the GMRES solver.
            max_iterations: Maximum number of GMRES inner iterations per sink.
                            GMRES restart is fixed at 20.

        Returns:
            Reference to an artifact containing ``probabilities`` and ``valid``.
        """
        if sinks is None:
            raise ValueError("sinks must be provided")
        if not isinstance(sinks, list):
            raise TypeError("sinks must be a list")
        if not sinks:
            raise ValueError("At least one sink label must be provided")
        if not isinstance(pseudotime, ArtifactRef):
            raise TypeError("pseudotime must be an ArtifactRef")
        if not isinstance(sink_labels, ArtifactRef):
            raise TypeError("sink_labels must be an ArtifactRef")
        pseudotime_result = self.load_pseudotime_scoring(pseudotime)
        graph_ref = pseudotime_result.graph
        stored_selection = pseudotime_result.cell_selection
        sink_values, sink_selection = _load_cell_artifact_values(
            self.zw,
            sink_labels,
        )
        if sink_values.ndim != 1:
            raise ValueError("Sink labels must be one-dimensional")
        if sink_selection != stored_selection:
            raise ValueError("Sink labels do not match the pseudotime cell selection")
        ptime = pseudotime_result.values
        ptime_valid = pseudotime_result.valid
        if ptime.shape != ptime_valid.shape or sink_values.shape != ptime.shape:
            raise ValueError("Pseudotime and sink labels must align")
        if not ptime_valid.any():
            raise ValueError("No cells were selected for fate mapping")

        logger.info(f"Fate mapping: loading graph {graph_ref.artifact_id}")
        graph_matrix = self.load_graph(
            graph_ref,
            symmetric=True,
            upper_only=False,
        )
        if graph_matrix.shape[0] != len(ptime_valid):
            raise ValueError("Pseudotime does not align with its graph")
        retained_graph = graph_matrix[ptime_valid][:, ptime_valid].tocsr()
        retained_ptime = ptime[ptime_valid]
        retained_sink_values = sink_values[ptime_valid]

        requested_sink_labels = tuple(sinks)
        arguments = FateMappingArguments(
            connectivity_map=graph_ref,
            pseudotime=pseudotime,
            sink_labels=sink_labels,
            cell_selection=stored_selection,
            sinks=requested_sink_labels,
            beta=beta,
            solver_tol=solver_tol,
            max_iterations=max_iterations,
            invalidate_cache=invalidate_cache,
        )
        record = arguments.to_record()
        planned = plan_cell_data_artifact(
            self.zw,
            scope=graph_ref.scope,
            assay=graph_ref.assay,
            kind=arguments.artifact_kind,
            operation=arguments.operation,
            parameters=record.parameters,
            inputs=record.inputs,
            execution_options=record.execution_options,
            cell_selection=stored_selection,
            arrays={
                "probabilities": (
                    (len(ptime), len(requested_sink_labels)),
                    "f",
                ),
                "valid": ((len(ptime),), "b"),
            },
            invalidate_cache=invalidate_cache,
        )
        if planned.reused:
            return planned.ref
        retained_probabilities, retained_valid, computed_sink_labels = (
            _compute_fate_probabilities_impl(
                retained_graph,
                retained_ptime,
                retained_sink_values,
                sinks,
                beta=beta,
                solver_tol=solver_tol,
                max_iterations=max_iterations,
                _copy_graph=False,
            )
        )
        if computed_sink_labels != requested_sink_labels:
            raise ValueError("Computed fate labels do not match requested sinks")
        probabilities = np.full(
            (len(ptime), len(requested_sink_labels)),
            np.nan,
            dtype=retained_probabilities.dtype,
        )
        valid = np.zeros(len(ptime), dtype=bool)
        probabilities[ptime_valid] = retained_probabilities
        valid[ptime_valid] = retained_valid
        write_cell_data_artifact(
            self.zw,
            planned,
            {
                "probabilities": probabilities,
                "valid": valid,
            },
        )
        logger.info(
            f"Stored fate probabilities for {len(requested_sink_labels)} sinks "
            f"across {int(valid.sum())}/{len(valid)} cells"
        )
        return planned.ref

    def load_fate_mapping(
        self,
        ref: ArtifactRef,
    ) -> FateMappingResult:
        """Load fate probabilities from an explicit completed artifact."""
        if not isinstance(ref, ArtifactRef):
            raise TypeError("ref must be an ArtifactRef")
        if ref.kind != "fate_map":
            raise ValueError("ref must be a fate_map artifact")
        status = inspect_artifact(self.zw, ref)
        if not status.complete or status.operation != "run_fate_mapping":
            raise ValueError("Fate-map artifact is unavailable or invalid")
        inputs = status.inputs or {}
        raw_graph = inputs.get("connectivity_map")
        raw_pseudotime = inputs.get("pseudotime")
        raw_sink_labels = inputs.get("sink_labels")
        raw_selection = inputs.get("cell_selection")
        graph = _artifact_ref_input(raw_graph, "Fate-map graph")
        pseudotime = _artifact_ref_input(raw_pseudotime, "Fate-map pseudotime")
        sink_labels = _artifact_ref_input(raw_sink_labels, "Fate-map sink labels")
        selection = _artifact_ref_input(raw_selection, "Fate-map cell selection")
        parameters = status.parameters or {}
        raw_sinks = parameters.get("sinks")
        if not isinstance(raw_sinks, list | tuple):
            raise ValueError("Fate-map sink labels are malformed")
        probabilities, loaded_selection = _load_cell_artifact_values(
            self.zw,
            ref,
            value_name="probabilities",
        )
        valid, valid_selection = _load_cell_artifact_values(
            self.zw,
            ref,
            value_name="valid",
        )
        if loaded_selection != selection or valid_selection != selection:
            raise ValueError("Fate-map arrays do not match their stored selection")
        return FateMappingResult(
            ref=ref,
            graph=graph,
            pseudotime=pseudotime,
            sink_labels_artifact=sink_labels,
            cell_selection=selection,
            sink_labels=tuple(raw_sinks),
            values=np.asarray(probabilities),
            valid=np.asarray(valid, dtype=bool),
        )


class _TrajectoryFeatureOperationsMixin(_TrajectoryFeatureOperationsBase):
    def run_pseudotime_marker_search(
        self,
        pseudotime: ArtifactRef,
        *,
        features: ArtifactRef,
        min_cells: int = 10,
        gene_batch_size: int | None = None,
        invalidate_cache: bool = False,
        **norm_params: Any,
    ) -> ArtifactRef:
        """Store feature correlations with an explicit pseudotime artifact.

        Args:
            pseudotime: Explicit pseudotime artifact defining cells and ordering.
            features: Explicit assay-scoped feature-selection artifact.
            min_cells: Minimum number of expressing cells required for a feature.
            gene_batch_size: Number of features loaded per batch. When None,
                selected features are grouped into chunk-aligned blocks that
                fit the operation memory budget.
            **norm_params: Extra keyword arguments forwarded to normalized expression.

        Returns:
            Reference to the immutable pseudotime-marker artifact.
        """
        from ...features.markers import find_markers_by_regression

        reject_unknown_normalization_params(
            norm_params,
            caller="run_pseudotime_marker_search",
        )
        if not isinstance(pseudotime, ArtifactRef):
            raise TypeError("pseudotime must be an ArtifactRef")
        if not isinstance(features, ArtifactRef):
            raise TypeError("features must be an ArtifactRef")
        if features.scope != "assay" or features.assay is None:
            raise ValueError("features must be an assay-scoped feature selection")
        assay = self._get_assay(features.assay)
        feature_selection, feature_index = _resolve_feature_indices(
            self,
            assay,
            features,
        )
        ptime_result = self.load_pseudotime_scoring(pseudotime)
        selected_cell_indices = read_stored_selection_indices(
            self.zw,
            ptime_result.cell_selection,
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        )
        cell_index = np.asarray(
            selected_cell_indices[ptime_result.valid],
            dtype=np.int64,
        )
        ptime = validate_pseudotime_regressor(
            ptime_result.values[ptime_result.valid],
            len(cell_index),
            "pseudotime artifact",
            "pseudotime validity mask",
            has_validity_column=True,
        )
        resolved_norm_params = {
            **norm_params,
            "log_transform": norm_params.get("log_transform", False),
            "renormalize_subset": norm_params.get(
                "renormalize_subset",
                False,
            ),
        }
        n_cells = len(cell_index)
        n_feats = len(feature_index)
        logger.info(
            f"Pseudotime markers: correlating features "
            f"(cells={n_cells}, features={n_feats}, "
            f"batch_size={gene_batch_size if gene_batch_size is not None else 'auto'})"
        )
        arguments = PseudotimeMarkerArguments(
            cell_selection=ptime_result.cell_selection,
            feature_selection=feature_selection,
            pseudotime=pseudotime,
            normalization=resolved_norm_params,
            normalization_method=callable_identity(assay.normMethod),
            size_factor=getattr(assay, "sf", None),
            association_method="pearson",
            p_value_method="student_t",
            adjustment_method="fdr_bh",
            adjustment_scope="tested_features",
            min_cells=min_cells,
            gene_batch_size=gene_batch_size,
            nthreads=int(
                getattr(
                    assay,
                    "nthreads",
                    getattr(self, "nthreads", 1),
                )
            ),
            invalidate_cache=invalidate_cache,
        )
        planned = arguments.plan(
            self.zw,
            scope="assay",
            assay=assay.name,
            invalidate_cache=invalidate_cache,
            required_arrays=tuple(
                ArrayRequirement(
                    name,
                    shape=(assay.feats.N,),
                    dtype_kind="f",
                )
                for name in ("r_value", "p_value", "p_value_adjusted")
            ),
        )
        if planned.reused:
            return planned.ref
        if getattr(self, "zarr_mode", "r+") != "r+":
            raise ValueError(
                "Pseudotime marker search requires a DataStore opened with "
                "zarr_mode='r+' when no reusable artifact exists"
            )
        markers = find_markers_by_regression(
            assay=assay,
            cell_idx=cell_index,
            feat_idx=feature_index,
            regressor=ptime,
            min_cells=min_cells,
            batch_size=gene_batch_size,
            **resolved_norm_params,
        ).reindex(feature_index)
        if markers["r_value"].isna().any():
            raise ValueError(
                "Pseudotime marker results are not aligned to feature selection"
            )
        full_r_values = np.full(assay.feats.N, np.nan, dtype=np.float64)
        full_p_values = np.full(assay.feats.N, np.nan, dtype=np.float64)
        full_p_values_adjusted = np.full(
            assay.feats.N,
            np.nan,
            dtype=np.float64,
        )
        full_r_values[feature_index] = np.asarray(markers["r_value"].values)
        full_p_values[feature_index] = np.asarray(markers["p_value"].values)
        full_p_values_adjusted[feature_index] = np.asarray(
            markers["p_value_adjusted"].values
        )
        write_cell_data_artifact(
            self.zw,
            planned,
            {
                "r_value": full_r_values,
                "p_value": full_p_values,
                "p_value_adjusted": full_p_values_adjusted,
            },
        )
        logger.info(f"Stored pseudotime marker scores for {len(markers)} features")
        return planned.ref

    def load_pseudotime_markers(
        self,
        ref: ArtifactRef,
    ) -> PseudotimeMarkerResult:
        """Load a pseudotime-marker table from an explicit artifact."""
        if not isinstance(ref, ArtifactRef):
            raise TypeError("ref must be an ArtifactRef")
        if ref.kind != "pseudotime_markers" or ref.scope != "assay":
            raise ValueError("ref must be an assay-scoped pseudotime_markers artifact")
        status = inspect_artifact(self.zw, ref)
        if not status.complete or status.operation != "run_pseudotime_marker_search":
            raise ValueError("Pseudotime-marker artifact is unavailable or invalid")
        inputs = status.inputs or {}
        raw_cell_selection = inputs.get("cell_selection")
        raw_feature_selection = inputs.get("feature_selection")
        raw_pseudotime = inputs.get("pseudotime")
        cell_selection = _artifact_ref_input(
            raw_cell_selection,
            "Pseudotime-marker cell selection",
        )
        feature_selection = _artifact_ref_input(
            raw_feature_selection,
            "Pseudotime-marker feature selection",
        )
        pseudotime = _artifact_ref_input(
            raw_pseudotime,
            "Pseudotime-marker pseudotime",
        )
        assay = self._get_assay(ref.assay)
        _, feature_indices = _resolve_feature_indices(
            self,
            assay,
            feature_selection,
        )
        group = as_zarr_group(self.zw[status.path], name=status.path)
        table = pd.DataFrame(
            {
                "feature_index": feature_indices,
                "feature_name": np.asarray(
                    assay.feats.fetch_all("names"),
                    dtype=object,
                )[feature_indices],
                "r_value": np.asarray(
                    as_zarr_array(group["r_value"], name="r_value")[:]
                )[feature_indices],
                "p_value": np.asarray(
                    as_zarr_array(group["p_value"], name="p_value")[:]
                )[feature_indices],
                "p_value_adjusted": np.asarray(
                    as_zarr_array(
                        group["p_value_adjusted"],
                        name="p_value_adjusted",
                    )[:]
                )[feature_indices],
            }
        )
        return PseudotimeMarkerResult(
            ref=ref,
            table=table,
            assay=assay.name,
            cell_selection=cell_selection,
            feature_selection=feature_selection,
            pseudotime=pseudotime,
        )

    def run_pseudotime_aggregation(
        self,
        pseudotime: ArtifactRef,
        *,
        features: ArtifactRef,
        min_exp: float = 1e-3,
        window_size: int = 200,
        chunk_size: int = 50,
        smoothen: bool = True,
        z_scale: bool = True,
        n_neighbours: int = 11,
        n_clusters: int = 10,
        batch_size: int | None = None,
        ann_params: dict | None = None,
        nan_cluster_value: int = -1,
        invalidate_cache: bool = False,
        **norm_params: Any,
    ) -> ArtifactRef:
        """Cluster features by pseudotime and store one immutable artifact.

        Args:
            pseudotime: Explicit pseudotime artifact defining cells and ordering.
            features: Explicit assay-scoped feature-selection artifact.
            min_exp: Minimum mean normalized expression required for clustering.
            window_size: Rolling window size used to smooth feature values.
            chunk_size: Number of pseudotime bins to create.
            smoothen: Whether to smooth expression along pseudotime.
            z_scale: Whether to standardize each retained feature.
            n_neighbours: Number of neighbors in the feature graph.
            n_clusters: Number of feature modules to create.
            batch_size: Number of features processed per batch. When None,
                selected features are grouped into chunk-aligned blocks that
                fit the operation memory budget.
            ann_params: Parameters forwarded to the HNSW index.
            nan_cluster_value: Value assigned to features excluded from clustering.
            **norm_params: Extra keyword arguments forwarded to normalized expression.

        Returns:
            Reference to the immutable pseudotime-aggregation artifact.
        """
        from ...trajectory.feature_dynamics import knn_clustering

        reject_unknown_normalization_params(
            norm_params,
            caller="run_pseudotime_aggregation",
        )
        if not isinstance(pseudotime, ArtifactRef):
            raise TypeError("pseudotime must be an ArtifactRef")
        if not isinstance(features, ArtifactRef):
            raise TypeError("features must be an ArtifactRef")
        if features.scope != "assay" or features.assay is None:
            raise ValueError("features must be an assay-scoped feature selection")
        assay = self._get_assay(features.assay)
        feature_selection, feature_indices = _resolve_feature_indices(
            self,
            assay,
            features,
        )
        ptime_result = self.load_pseudotime_scoring(pseudotime)
        selected_cell_indices = read_stored_selection_indices(
            self.zw,
            ptime_result.cell_selection,
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        )
        cell_indices = np.asarray(
            selected_cell_indices[ptime_result.valid],
            dtype=np.int64,
        )
        if not isinstance(nan_cluster_value, (int, np.integer)) or isinstance(
            nan_cluster_value,
            (bool, np.bool_),
        ):
            raise TypeError("nan_cluster_value must be an integer")
        nan_cluster_value = int(nan_cluster_value)
        cell_ordering = validate_pseudotime_regressor(
            ptime_result.values[ptime_result.valid],
            len(cell_indices),
            "pseudotime artifact",
            "pseudotime validity mask",
            has_validity_column=True,
        )
        resolved_norm_params = {
            **norm_params,
            "log_transform": norm_params.get("log_transform", False),
            "renormalize_subset": norm_params.get(
                "renormalize_subset",
                False,
            ),
        }
        resolved_ann_params = {} if ann_params is None else dict(ann_params)
        (
            cell_ordering,
            cell_indices,
            feature_indices,
            effective_window,
            effective_bins,
            input_fingerprints,
            _legacy_parameters,
        ) = assay._prepare_aggregated_ordering(
            cell_indices,
            feature_indices,
            cell_ordering,
            min_exp=min_exp,
            window_size=window_size,
            chunk_size=chunk_size,
            smoothen=smoothen,
            z_scale=z_scale,
            norm_params=resolved_norm_params,
        )
        arguments = PseudotimeAggregationArguments(
            cell_selection=ptime_result.cell_selection,
            feature_selection=feature_selection,
            pseudotime=pseudotime,
            normalization=resolved_norm_params,
            normalization_method=callable_identity(assay.normMethod),
            size_factor=getattr(assay, "sf", None),
            min_exp=min_exp,
            window_size=window_size,
            chunk_size=chunk_size,
            smoothen=smoothen,
            z_scale=z_scale,
            n_neighbours=n_neighbours,
            n_clusters=n_clusters,
            ann_params=resolved_ann_params,
            nan_cluster_value=nan_cluster_value,
            batch_size=batch_size,
            nthreads=self.nthreads,
            invalidate_cache=invalidate_cache,
        )
        planned = arguments.plan(
            self.zw,
            scope="assay",
            assay=assay.name,
            invalidate_cache=invalidate_cache,
            required_arrays=(
                ArrayRequirement(
                    "data",
                    shape=(len(feature_indices), effective_bins),
                    dtype_kind="f",
                ),
                ArrayRequirement(
                    "feature_indices",
                    shape=(len(feature_indices),),
                    dtype_kind="u",
                ),
                ArrayRequirement(
                    "valid_features",
                    shape=(len(feature_indices),),
                    dtype_kind="b",
                ),
                ArrayRequirement(
                    "feature_clusters",
                    shape=(len(feature_indices),),
                    dtype_kind="i",
                ),
                ArrayRequirement(
                    "cluster_values",
                    shape=(assay.feats.N,),
                    dtype_kind="i",
                ),
            ),
            required_attributes=(
                AttributeRequirement(
                    "input_fingerprints",
                    expected_types=(list, tuple),
                ),
                AttributeRequirement(
                    "nan_cluster_value",
                    expected_types=(int,),
                ),
            ),
        )
        if planned.reused:
            return planned.ref
        logger.info("Pseudotime modules: aggregating feature profiles")
        aggregation_group = start_artifact(self.zw, planned)
        full_data, stored_feature_indices, valid_features = (
            assay._write_aggregated_ordering_group(
                aggregation_group,
                cell_idx=cell_indices,
                cell_ordering=cell_ordering,
                feat_idx=feature_indices,
                min_exp=min_exp,
                effective_window=effective_window,
                effective_bins=effective_bins,
                smoothen=smoothen,
                z_scale=z_scale,
                batch_size=batch_size,
                norm_params=resolved_norm_params,
            )
        )
        valid_data = full_data[valid_features]
        valid_feature_indices = stored_feature_indices[valid_features]
        clusts = knn_clustering(
            d_array=valid_data,
            n_neighbours=n_neighbours,
            n_clusters=n_clusters,
            nthreads=self.nthreads,
            ann_params=resolved_ann_params,
        )
        stored_feature_clusters = np.full(
            len(stored_feature_indices),
            nan_cluster_value,
            dtype=np.int64,
        )
        stored_feature_clusters[valid_features] = clusts
        cluster_values = _scatter_feature_clusters_impl(
            assay.feats.N,
            valid_feature_indices,
            clusts,
            nan_cluster_value,
        )
        for name, values in (
            ("feature_clusters", stored_feature_clusters),
            ("cluster_values", cluster_values),
        ):
            output = create_zarr_dataset(
                aggregation_group,
                name,
                (max(len(values), 1),),
                values.dtype,
                values.shape,
            )
            output[:] = values
        aggregation_group.attrs["input_fingerprints"] = input_fingerprints
        aggregation_group.attrs["nan_cluster_value"] = nan_cluster_value
        aggregation_group.attrs["effective_window"] = effective_window
        aggregation_group.attrs["effective_bins"] = effective_bins
        finish_artifact(aggregation_group, planned)
        logger.info(f"Stored {np.unique(clusts).size} pseudotime modules")
        return planned.ref

    def load_pseudotime_aggregation(
        self,
        ref: ArtifactRef,
    ) -> PseudotimeAggregationResult:
        """Load a lazy pseudotime aggregation from an explicit artifact."""
        if not isinstance(ref, ArtifactRef):
            raise TypeError("ref must be an ArtifactRef")
        if ref.kind != "pseudotime_aggregation" or ref.scope != "assay":
            raise ValueError(
                "ref must be an assay-scoped pseudotime_aggregation artifact"
            )
        status = inspect_artifact(self.zw, ref)
        if not status.complete or status.operation != "run_pseudotime_aggregation":
            raise ValueError(
                "Pseudotime-aggregation artifact is unavailable or invalid"
            )
        inputs = status.inputs or {}
        raw_cell_selection = inputs.get("cell_selection")
        raw_feature_selection = inputs.get("feature_selection")
        raw_pseudotime = inputs.get("pseudotime")
        cell_selection = _artifact_ref_input(
            raw_cell_selection,
            "Pseudotime-aggregation cell selection",
        )
        feature_selection = _artifact_ref_input(
            raw_feature_selection,
            "Pseudotime-aggregation feature selection",
        )
        pseudotime = _artifact_ref_input(
            raw_pseudotime,
            "Pseudotime-aggregation pseudotime",
        )
        group = as_zarr_group(self.zw[status.path], name=status.path)
        full_data = ChunkedArray(
            as_zarr_array(group["data"], name="data"),
            nthreads=self.nthreads,
        )
        feature_indices = np.asarray(
            as_zarr_array(group["feature_indices"], name="feature_indices")[:]
        )
        valid_features = np.asarray(
            as_zarr_array(group["valid_features"], name="valid_features")[:],
            dtype=bool,
        )
        feature_clusters = np.asarray(
            as_zarr_array(group["feature_clusters"], name="feature_clusters")[:],
            dtype=np.int64,
        )
        return PseudotimeAggregationResult(
            ref=ref,
            data=full_data[valid_features],
            feature_indices=feature_indices[valid_features],
            feature_clusters=feature_clusters[valid_features],
            assay=cast(str, ref.assay),
            cell_selection=cell_selection,
            feature_selection=feature_selection,
            pseudotime=pseudotime,
        )
