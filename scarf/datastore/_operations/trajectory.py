from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix

from ...assay import Assay
from ...assay.normalization import reject_unknown_normalization_params
from ...graph.state import resolve_stored_graph_input, validate_legacy_graph_selection
from ...matrix import ChunkedArray
from ...metadata.arguments import (
    FateMappingArguments,
    PseudotimeAggregationArguments,
    PseudotimeMarkerArguments,
    PseudotimeScoringArguments,
)
from ...metadata.artifacts import (
    artifact_values,
    categorical_display,
    column_display,
    continuous_display,
    feature_column_display,
    link_cell_data_column,
    link_feature_data_column,
    plan_cell_data_artifact,
    write_cell_data_artifact,
)
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
    provenance_hash,
)
from ...storage.types import as_zarr_array, as_zarr_group
from ...storage.arrays import create_zarr_dataset
from ...trajectory.feature_dynamics import (
    scatter_feature_clusters as _scatter_feature_clusters_impl,
    validate_pseudotime_regressor,
)
from ...trajectory.fate import (
    compute_fate_probabilities as _compute_fate_probabilities_impl,
    make_sink_tokens as _make_sink_tokens_impl,
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
from ...utils.arrays import array_digest
from ...utils.logging import logger

if TYPE_CHECKING:
    from ..mapping_datastore import (
        MappingDatastore as _TrajectoryFeatureOperationsBase,
    )
    from .graph import _GraphOperationsMixin as _TrajectoryOperationsBase
else:
    _TrajectoryFeatureOperationsBase = object
    _TrajectoryOperationsBase = object


def _group_assignment_digest(values: np.ndarray) -> str:
    return array_digest(np.asarray(values).astype(str))


def _validate_assay_pseudotime(
    assay: Assay,
    cell_key: str,
    pseudotime_key: str,
) -> np.ndarray:
    try:
        values = assay.cells.fetch(pseudotime_key, key=cell_key)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"Pseudotime column '{pseudotime_key}' must be numeric"
        ) from exc
    expected_size = assay.cells.active_index(cell_key).shape[0]
    return validate_pseudotime_regressor(
        values,
        expected_size,
        pseudotime_key,
        cell_key,
        has_validity_column=f"{pseudotime_key}__valid" in assay.cells.columns,
    )


def _stored_graph_input(
    store: Any,
    from_assay: str,
    cell_key: str,
    feat_key: str,
) -> tuple[str, object]:
    graph_loc = store.get_latest_graph_loc(from_assay, cell_key, feat_key)
    return graph_loc, resolve_stored_graph_input(store.zw, graph_loc)


class _TrajectoryOperationsMixin(_TrajectoryOperationsBase):
    def get_diffusion_operator(
        self,
        from_assay: str | None = None,
        cell_key: str | None = None,
        feat_key: str | None = None,
        t: int = 2,
        *,
        cache_operator: bool = True,
        invalidate_cache: bool = False,
    ) -> coo_matrix:
        """Load or calculate the sparse MAGIC diffusion operator."""
        from ...neighbors.diffusion import diffusion_operator

        from_assay, cell_key, feat_key = self._get_latest_keys(
            from_assay,
            cell_key,
            feat_key,
        )
        graph_loc = self.get_latest_graph_loc(from_assay, cell_key, feat_key)
        graph_input: object = resolve_stored_graph_input(self.zw, graph_loc)
        planned = plan_artifact(
            self.zw,
            scope="assay",
            assay=from_assay,
            kind="diffusion_operator",
            operation="get_diffusion_operator",
            parameters={"t": t},
            inputs={"connectivity_map": graph_input},
            execution_options={
                "cache_operator": cache_operator,
                "invalidate_cache": invalidate_cache,
            },
            invalidate_cache=invalidate_cache,
            required_arrays=(
                ArrayRequirement("row", dtype_kind="u"),
                ArrayRequirement("col", dtype_kind="u"),
                ArrayRequirement("data", dtype_kind="f"),
            ),
        )
        magic_loc = planned.ref.artifact_id
        read_only_magic_loc = f"read_only:{provenance_hash(planned.provenance)}"
        if not cache_operator:
            self._cachedMagicOperator = None
            self._cachedMagicOperatorLoc = None
        cache_lookup_loc = (
            read_only_magic_loc
            if self.zarr_mode != "r+" and not planned.reused
            else magic_loc
        )
        cached = (
            self._cachedMagicOperator
            if self._cachedMagicOperatorLoc == cache_lookup_loc
            else None
        )
        if self.zarr_mode != "r+" and not planned.reused:
            if cache_operator and not invalidate_cache and cached is not None:
                return cast(coo_matrix, cached)
            legacy_path = f"{graph_loc}/magic_{t}"
            if not invalidate_cache and legacy_path in self.zw:
                legacy = as_zarr_group(self.zw[legacy_path], name=legacy_path)
                n_cells, _ = self._get_graph_ncells_k(graph_loc)
                diff_op = coo_matrix(
                    (
                        np.asarray(as_zarr_array(legacy["data"], name="data")[:]),
                        (
                            np.asarray(as_zarr_array(legacy["row"], name="row")[:]),
                            np.asarray(as_zarr_array(legacy["col"], name="col")[:]),
                        ),
                    ),
                    shape=(n_cells, n_cells),
                )
            else:
                graph = self.load_graph(
                    from_assay=from_assay,
                    cell_key=cell_key,
                    feat_key=feat_key,
                    symmetric=True,
                    upper_only=False,
                )
                diff_op = diffusion_operator(graph, t)
            if cache_operator:
                self._cachedMagicOperator = diff_op
                self._cachedMagicOperatorLoc = read_only_magic_loc  # type: ignore[assignment]
            return diff_op
        if planned.reused:
            store = reused_artifact_group(self.zw, planned)
            if cached is not None:
                diff_op = cast(coo_matrix, cached)
            else:
                logger.debug("Using existing MAGIC diffusion operator")
                n_cells, _ = self._get_graph_ncells_k(graph_loc)
                diff_op = coo_matrix(
                    (
                        np.asarray(as_zarr_array(store["data"], name="data")[:]),
                        (
                            np.asarray(as_zarr_array(store["row"], name="row")[:]),
                            np.asarray(as_zarr_array(store["col"], name="col")[:]),
                        ),
                    ),
                    shape=(n_cells, n_cells),
                )
        else:
            graph = self.load_graph(
                from_assay=from_assay,
                cell_key=cell_key,
                feat_key=feat_key,
                symmetric=True,
                upper_only=False,
            )
            diff_op = diffusion_operator(graph, t)
            shape = diff_op.data.shape
            store = start_artifact(self.zw, planned)
            for name, dtype in zip(
                ("row", "col", "data"),
                ("uint32", "uint32", "float64"),
                strict=True,
            ):
                array = create_zarr_dataset(store, name, (1000000,), dtype, shape)
                array[:] = getattr(diff_op, name)
            store.attrs["n_cells"] = int(graph.shape[0])
            finish_artifact(store, planned)

        if cache_operator:
            self._cachedMagicOperator = diff_op
            self._cachedMagicOperatorLoc = magic_loc  # type: ignore[assignment]
        else:
            self._cachedMagicOperator = None
            self._cachedMagicOperatorLoc = None
        return diff_op

    def get_imputed(
        self,
        from_assay: str | None = None,
        cell_key: str | None = None,
        feature_name: str | None = None,
        feat_key: str | None = None,
        t: int = 2,
        cache_operator: bool = True,
        invalidate_cache: bool = False,
    ) -> np.ndarray:
        """Impute feature values by diffusing along the KNN graph (MAGIC-style).

        Args:
            from_assay: Name of assay to be used. If no value is provided then the default assay will be used.
            cell_key: Cell key. Should be same as the one that was used in the desired graph. (Default value: 'I')
            feature_name: Name of the feature to be imputed
            feat_key: Feature key. Should be same as the one that was used in the desired graph. By default, the latest
                       used feature for the given assay will be used.
            t: Same as the t parameter in MAGIC. Higher values lead to larger diffusion of values. Too large values
               can slow down the algorithm and cause over-smoothening. (Default value: 2)
            cache_operator: Whether to keep the diffusion operator in memory after the method returns. Can be useful
                            to set to True if many features are to imputed in a batch but can lead to increased memory
                            usage. (Default value: True)

        Returns:
            An array of imputed values for the given feature

        """

        from_assay, cell_key, feat_key = self._get_latest_keys(
            from_assay, cell_key, feat_key
        )
        if feature_name is None:
            raise ValueError(
                "ERROR: Please provide name for the feature to be imputed. It can, for example, "
                "be a gene name."
            )
        data = self.get_cell_vals(
            from_assay=from_assay, cell_key=cell_key, k=feature_name
        )
        diff_op = self.get_diffusion_operator(
            from_assay=from_assay,
            cell_key=cell_key,
            feat_key=feat_key,
            t=t,
            cache_operator=cache_operator,
            invalidate_cache=invalidate_cache,
        )
        return cast(np.ndarray, diff_op.dot(data))

    def run_pseudotime_scoring(
        self,
        from_assay: str | None = None,
        cell_key: str | None = None,
        subset_cell_key: str | None = None,
        feat_key: str | None = None,
        n_singular_vals: int = 30,
        source_sink_key: str | None = None,
        sources: list[Any] | None = None,
        sinks: list[Any] | None = None,
        ss_vec: np.ndarray | None = None,
        min_max_norm_ptime: bool = True,
        random_seed: int = 4444,
        label: str = "pseudotime",
        component_policy: Literal["largest", "error"] = "largest",
        invalidate_cache: bool = False,
    ) -> PseudotimeScoreResult:
        """
        Calculate differentiation potential of cells. This function is a reimplementation of population balance
        analysis (PBA) approach published in Weinreb et al. 2017, PNAS. This function computes the random walk
        normalized Laplacian matrix of the reference graph, L_rw = I-A/D and then calculates a Moore-Penrose
        pseudoinverse of L_rw.

        Args:
            from_assay: Name of assay to be used. If no value is provided then the default assay will be used.
            cell_key: Cell key. Should be same as the one that was used in the desired graph. (Default value: 'I')
            subset_cell_key: Cell key for the subset of cells for which pseudotime scoring is to be performed.
            feat_key: Feature key. Should be same as the one that was used in the desired graph. By default, the latest
                        used feature for the given assay will be used.
            n_singular_vals: Number of the smallest singular values to save.
            source_sink_key: Cell metadata column containing source and sink
                group labels. Required with ``sources`` or ``sinks`` and omitted
                when ``ss_vec`` is supplied.
            sources: A list of group/clusters ids from `source_sink_key` column to be treated as sources. Sources are
                     usually progenitor/precursor or other actively dividing cell states.
            sinks: A list of group/clusters ids from `source_sink_key` column to be treated as sinks. Sinks are usually
                   more differentiated (or terminally differentiated) cell states.
            ss_vec: Custom source-sink value for each selected cell. It must sum
                to zero, with negative values for sources and positive values for
                sinks. This is mutually exclusive with label-based arguments.
            min_max_norm_ptime: Whether to perform min-max normalization on the final pseudotime values so that values
                                are in 0 to 1 range. (Default: True)
            random_seed: A random seed for svds (Default: 4444)
            label: Base label for the pseudotime cell metadata column.
            component_policy: How to handle a disconnected selected graph. ``'largest'`` scores the largest connected
                              component and marks other selected cells as unscored. ``'error'`` raises instead.

        Returns:
            Pseudotime values, validity mask, and their saved metadata keys.
        """

        from_assay, cell_key, feat_key = self._get_latest_keys(
            from_assay, cell_key, feat_key
        )
        if subset_cell_key is None:
            subset_cell_key = cell_key
            cell_idx = self.cells.fetch(subset_cell_key, key=cell_key)
        else:
            cell_idx = self.cells.fetch(subset_cell_key, key=cell_key)
            if cell_idx.sum() != self.cells.fetch_all(subset_cell_key).sum():
                raise ValueError("subset_cell_key is not a complete subset of cell_key")

        logger.info(
            f"Pseudotime scoring: loading graph "
            f"({from_assay}, cell_key={cell_key}, feat_key={feat_key})"
        )
        graph_loc, graph_input = _stored_graph_input(
            self,
            from_assay,
            cell_key,
            feat_key,
        )
        if not isinstance(graph_input, ArtifactRef):
            validate_legacy_graph_selection(
                self,
                graph_loc,
                from_assay,
                cell_key,
                feat_key,
            )
        graph = self.load_graph(
            from_assay=from_assay,
            cell_key=cell_key,
            feat_key=feat_key,
            symmetric=True,
            upper_only=False,
        )
        if cell_idx.shape[0] != cell_idx.sum():
            graph = graph[cell_idx][:, cell_idx]

        if graph.shape[0] == 0:
            raise ValueError("No cells were selected for pseudotime scoring")
        parent_cell_indices = self.cells.active_index(cell_key)
        selected_cell_indices = parent_cell_indices[np.asarray(cell_idx, dtype=bool)]
        retained_mask, component_sizes = _select_pseudotime_component_impl(
            graph,
            selected_cell_indices,
            component_policy,
        )
        retained_graph = graph[retained_mask][:, retained_mask].tocsr()
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
            source_sink_key is not None or sources is not None or sinks is not None
        )
        if ss_vec is not None and label_arguments_supplied:
            raise ValueError(
                "Provide either ss_vec or source_sink_key with source/sink labels, not both"
            )
        if ss_vec is None and source_sink_key is None:
            if sources is not None or sinks is not None:
                raise ValueError(
                    "source_sink_key is required when sources or sinks are provided"
                )
            raise ValueError("Provide source/sink labels or a custom zero-sum ss_vec")

        if ss_vec is not None:
            full_source_sink = _validate_source_sink_vector_impl(
                ss_vec,
                graph.shape[0],
                "ss_vec",
            )
            source_sink_input: object = full_source_sink
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

            selected_labels = pd.Series(
                self.cells.fetch(cast(str, source_sink_key), key=cell_key)[cell_idx]
            )
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
            source_sink_input = self._resolve_cell_data_provenance_input(
                cast(str, source_sink_key),
                cell_key=cell_key,
            )

        graph_selection = self._ensure_cell_selection(cell_key)
        if isinstance(graph_input, ArtifactRef):
            stored_selection = self._graph_cell_selection(graph_input)
            if not self._selection_artifacts_match(
                stored_selection,
                graph_selection,
            ):
                raise ValueError("cell_key does not match the graph cell selection")
        result_selection = self._ensure_cell_selection(subset_cell_key)
        arguments = PseudotimeScoringArguments(
            connectivity_map=graph_input,
            source_sink=source_sink_input,
            cell_selection=result_selection,
            n_singular_vals=n_singular_vals,
            sources=tuple([] if sources is None else sources),
            sinks=tuple([] if sinks is None else sinks),
            min_max_norm_ptime=min_max_norm_ptime,
            random_seed=random_seed,
            component_policy=component_policy,
            from_assay=from_assay,
            cell_key=cell_key,
            subset_cell_key=subset_cell_key,
            feat_key=feat_key,
            label=label,
            invalidate_cache=invalidate_cache,
        )
        record = arguments.to_record()
        artifact_scope = (
            graph_input.scope if isinstance(graph_input, ArtifactRef) else "assay"
        )
        planned = plan_cell_data_artifact(
            self.zw,
            scope=artifact_scope,
            assay=(
                graph_input.assay
                if isinstance(graph_input, ArtifactRef) and graph_input.scope == "assay"
                else from_assay
                if artifact_scope == "assay"
                else None
            ),
            kind=arguments.artifact_kind,
            operation=arguments.operation,
            parameters=record.parameters,
            inputs=record.inputs,
            execution_options=record.execution_options,
            cell_selection=result_selection,
            arrays={
                "pseudotime": ((graph.shape[0],), "f"),
                "valid": ((graph.shape[0],), "b"),
            },
            invalidate_cache=invalidate_cache,
        )
        output_column = self._col_renamer(from_assay, subset_cell_key, label)
        validity_column = f"{output_column}__valid"
        preserved_displays = {
            output_column: column_display(self.zw, output_column),
            validity_column: column_display(self.zw, validity_column),
        }

        def publish_result(
            values: np.ndarray,
            valid: np.ndarray,
        ) -> PseudotimeScoreResult:
            self.cells.insert(
                output_column,
                values,
                key=subset_cell_key,
                overwrite=True,
            )
            self.cells.insert(
                validity_column,
                valid,
                fill_value=False,
                key=subset_cell_key,
                overwrite=True,
            )
            link_cell_data_column(
                self.zw,
                output_column,
                planned.ref,
                value_name="pseudotime",
                default_display=continuous_display(values),
                preserved_display=preserved_displays[output_column],
            )
            link_cell_data_column(
                self.zw,
                validity_column,
                planned.ref,
                value_name="valid",
                default_display=categorical_display(valid),
                preserved_display=preserved_displays[validity_column],
            )
            if not valid.all():
                logger.warning(
                    f"Unscored cells contain NaN pseudotime. Use cell key "
                    f"'{validity_column}' for downstream analysis"
                )
            logger.info(
                f"Stored pseudotime scores for {int(valid.sum())}/{len(valid)} cells"
            )
            return PseudotimeScoreResult(
                pseudotime_key=output_column,
                validity_key=validity_column,
                assay=from_assay,
                graph_cell_key=cell_key,
                result_cell_key=subset_cell_key,
                feature_key=feat_key,
                values=values,
                valid=valid,
            )

        if planned.reused:
            artifact_group = reused_artifact_group(self.zw, planned)
            return publish_result(
                artifact_values(artifact_group, "pseudotime"),
                artifact_values(artifact_group, "valid").astype(bool),
            )

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

        ptime = np.full(graph.shape[0], np.nan, dtype=float)
        ptime[retained_mask] = retained_ptime
        write_cell_data_artifact(
            self.zw,
            planned,
            {
                "pseudotime": ptime,
                "valid": retained_mask,
            },
        )
        logger.debug("Pseudotime scoring: saving pseudotime")
        return publish_result(ptime, retained_mask)

    def run_fate_mapping(
        self,
        from_assay: str | None = None,
        cell_key: str | None = None,
        subset_cell_key: str | None = None,
        feat_key: str | None = None,
        pseudotime_key: str | None = None,
        sink_key: str | None = None,
        sinks: list[Any] | None = None,
        beta: float = 10.0,
        solver_tol: float = 1e-6,
        max_iterations: int = 1000,
        label: str = "fate",
        invalidate_cache: bool = False,
    ) -> FateMappingResult:
        """Compute cell fate probabilities toward user-provided sink groups.

        Args:
            from_assay: Assay whose neighborhood graph should be used.
            cell_key: Cell key used to create the neighborhood graph.
            subset_cell_key: Cell key restricting the graph for this calculation.
                             Use the pseudotime validity key when unscored cells
                             contain non-finite pseudotime.
            feat_key: Feature key used to create the neighborhood graph.
            pseudotime_key: Numeric cell metadata column containing pseudotime.
            sink_key: Cell metadata column containing the sink labels.
            sinks: Ordered sink labels. Every matching selected cell becomes a
                   fate boundary.
            beta: Strength of the penalty applied to backward graph edges.
            solver_tol: Relative tolerance used by the GMRES solver.
            max_iterations: Maximum number of GMRES inner iterations per sink.
                            GMRES restart is fixed at 20.
            label: Base label used for saved fate probability columns.

        Returns:
            Fate probabilities, validity mask, sink labels, and saved metadata keys.
        """
        if pseudotime_key is None:
            raise ValueError("pseudotime_key must be provided")
        if sink_key is None:
            raise ValueError("sink_key must be provided")
        if sinks is None:
            raise ValueError("sinks must be provided")
        if not isinstance(sinks, list):
            raise TypeError("sinks must be a list")
        if not sinks:
            raise ValueError("At least one sink label must be provided")
        if not isinstance(label, str):
            raise TypeError("label must be a string")
        if not label:
            raise ValueError("label must not be empty")

        from_assay, cell_key, feat_key = self._get_latest_keys(
            from_assay,
            cell_key,
            feat_key,
        )
        if subset_cell_key is None:
            subset_cell_key = cell_key
        cell_idx = np.asarray(self.cells.fetch(subset_cell_key, key=cell_key))
        if cell_idx.ndim != 1 or cell_idx.dtype.kind != "b":
            raise TypeError("subset_cell_key must select cells with boolean values")
        if int(cell_idx.sum()) != int(self.cells.fetch_all(subset_cell_key).sum()):
            raise ValueError("subset_cell_key is not a complete subset of cell_key")
        if not cell_idx.any():
            raise ValueError("No cells were selected for fate mapping")

        assay = self._get_assay(from_assay)
        pseudotime = _validate_assay_pseudotime(
            assay,
            subset_cell_key,
            pseudotime_key,
        )
        sink_values = np.asarray(assay.cells.fetch(sink_key, key=subset_cell_key))
        storage_backed = all(
            hasattr(self, name)
            for name in (
                "zw",
                "get_latest_graph_loc",
                "_ensure_cell_selection",
                "_resolve_cell_data_provenance_input",
            )
        )
        if storage_backed:
            graph_loc, graph_input = _stored_graph_input(
                self,
                from_assay,
                cell_key,
                feat_key,
            )
            if not isinstance(graph_input, ArtifactRef):
                validate_legacy_graph_selection(
                    self,
                    graph_loc,
                    from_assay,
                    cell_key,
                    feat_key,
                )
            pseudotime_input = self._resolve_cell_data_provenance_input(
                pseudotime_key,
                cell_key=subset_cell_key,
            )
            sink_input = self._resolve_cell_data_provenance_input(
                sink_key,
                cell_key=subset_cell_key,
            )

        logger.info(
            f"Fate mapping: loading graph "
            f"({from_assay}, cell_key={cell_key}, feat_key={feat_key})"
        )
        graph = self.load_graph(
            from_assay=from_assay,
            cell_key=cell_key,
            feat_key=feat_key,
            symmetric=True,
            upper_only=False,
        )
        if cell_idx.shape[0] != int(cell_idx.sum()):
            graph = graph[cell_idx][:, cell_idx].tocsr()

        sink_labels = tuple(sinks)
        output_base = self._col_renamer(from_assay, subset_cell_key, label)
        fate_keys = tuple(
            f"{output_base}_{token}" for token in _make_sink_tokens_impl(sink_labels)
        )
        validity_key = f"{output_base}__valid"
        planned = None
        preserved_displays: dict[str, dict[str, Any] | None] = {}
        if storage_backed:
            graph_selection = self._ensure_cell_selection(cell_key)
            if isinstance(graph_input, ArtifactRef):
                stored_selection = self._graph_cell_selection(graph_input)
                if not self._selection_artifacts_match(
                    stored_selection,
                    graph_selection,
                ):
                    raise ValueError("cell_key does not match the graph cell selection")
            result_selection = self._ensure_cell_selection(subset_cell_key)
            arguments = FateMappingArguments(
                connectivity_map=graph_input,
                pseudotime=pseudotime_input,
                sink_labels=sink_input,
                cell_selection=result_selection,
                sinks=sink_labels,
                beta=beta,
                solver_tol=solver_tol,
                max_iterations=max_iterations,
                from_assay=from_assay,
                cell_key=cell_key,
                subset_cell_key=subset_cell_key,
                feat_key=feat_key,
                pseudotime_key=pseudotime_key,
                sink_key=sink_key,
                label=label,
                invalidate_cache=invalidate_cache,
            )
            record = arguments.to_record()
            artifact_scope = (
                graph_input.scope if isinstance(graph_input, ArtifactRef) else "assay"
            )
            planned = plan_cell_data_artifact(
                self.zw,
                scope=artifact_scope,
                assay=(
                    graph_input.assay
                    if isinstance(graph_input, ArtifactRef)
                    and graph_input.scope == "assay"
                    else from_assay
                    if artifact_scope == "assay"
                    else None
                ),
                kind=arguments.artifact_kind,
                operation=arguments.operation,
                parameters=record.parameters,
                inputs=record.inputs,
                execution_options=record.execution_options,
                cell_selection=result_selection,
                arrays={
                    "probabilities": (
                        (graph.shape[0], len(sink_labels)),
                        "f",
                    ),
                    "valid": ((graph.shape[0],), "b"),
                },
                invalidate_cache=invalidate_cache,
            )
            preserved_displays = {
                column: column_display(self.zw, column)
                for column in (*fate_keys, validity_key)
            }
        if planned is not None and planned.reused:
            artifact_group = reused_artifact_group(self.zw, planned)
            probabilities = artifact_values(artifact_group, "probabilities")
            valid = artifact_values(artifact_group, "valid").astype(bool)
        else:
            probabilities, valid, computed_sink_labels = (
                _compute_fate_probabilities_impl(
                    graph,
                    pseudotime,
                    sink_values,
                    sinks,
                    beta=beta,
                    solver_tol=solver_tol,
                    max_iterations=max_iterations,
                    _copy_graph=False,
                )
            )
            if computed_sink_labels != sink_labels:
                raise ValueError("Computed fate labels do not match requested sinks")
            if planned is not None:
                write_cell_data_artifact(
                    self.zw,
                    planned,
                    {
                        "probabilities": probabilities,
                        "valid": valid,
                    },
                )

        logger.debug("Fate mapping: saving probabilities")
        for index, fate_key in enumerate(fate_keys):
            self.cells.insert(
                fate_key,
                probabilities[:, index],
                key=subset_cell_key,
                overwrite=True,
            )
        self.cells.insert(
            validity_key,
            valid,
            fill_value=False,
            key=subset_cell_key,
            overwrite=True,
        )
        if planned is not None:
            for index, fate_key in enumerate(fate_keys):
                link_cell_data_column(
                    self.zw,
                    fate_key,
                    planned.ref,
                    value_name="probabilities",
                    value_index=index,
                    default_display=continuous_display(probabilities[:, index]),
                    preserved_display=preserved_displays[fate_key],
                )
            link_cell_data_column(
                self.zw,
                validity_key,
                planned.ref,
                value_name="valid",
                default_display=categorical_display(valid),
                preserved_display=preserved_displays[validity_key],
            )
        logger.info(
            f"Stored fate probabilities for {len(fate_keys)} sinks "
            f"across {int(valid.sum())}/{len(valid)} cells"
        )
        return FateMappingResult(
            fate_keys=fate_keys,
            validity_key=validity_key,
            sink_labels=sink_labels,
            assay=from_assay,
            graph_cell_key=cell_key,
            result_cell_key=subset_cell_key,
            feature_key=feat_key,
            pseudotime_key=pseudotime_key,
            sink_key=sink_key,
            values=probabilities,
            valid=valid,
        )


class _TrajectoryFeatureOperationsMixin(_TrajectoryFeatureOperationsBase):
    def run_pseudotime_marker_search(
        self,
        from_assay: str | None = None,
        cell_key: str | None = None,
        feat_key: str | None = None,
        pseudotime_key: str | None = None,
        min_cells: int = 10,
        gene_batch_size: int | None = None,
        invalidate_cache: bool = False,
        **norm_params: Any,
    ) -> PseudotimeMarkerResult:
        """Identify genes correlated with a pseudotime ordering of cells.

        Args:
            from_assay: Name of the assay to use. The default assay is used when omitted.
            cell_key: Boolean cell metadata column selecting cells.
            feat_key: Boolean feature metadata column selecting features.
            pseudotime_key: Numeric cell metadata column containing pseudotime values.
            min_cells: Minimum number of expressing cells required for a feature.
            gene_batch_size: Number of features loaded per batch. When None,
                selected features are grouped into chunk-aligned blocks that
                fit the operation memory budget.
            **norm_params: Extra keyword arguments forwarded to normalized expression.

        Returns:
            Correlation table and the feature metadata keys where it was saved.
        """
        from ...features.markers import find_markers_by_regression

        reject_unknown_normalization_params(
            norm_params,
            caller="run_pseudotime_marker_search",
        )
        if pseudotime_key is None:
            raise ValueError(
                "ERROR: Please provide a value for `pseudotime_key`. This should be the name of a column from "
                "cell metadata object where pseudotime values are stored. If you ran `run_pseudotime_scoring` then "
                "the values are stored under `RNA_pseudotime` by default."
            )
        if cell_key is None:
            cell_key = "I"
        if feat_key is None:
            feat_key = "I"
        assay = self._get_assay(from_assay)
        ptime = _validate_assay_pseudotime(
            assay,
            cell_key,
            pseudotime_key,
        )
        resolved_norm_params = {
            **norm_params,
            "log_transform": norm_params.get("log_transform", False),
            "renormalize_subset": norm_params.get(
                "renormalize_subset",
                False,
            ),
        }
        n_cells = len(assay.cells.active_index(cell_key))
        feature_index = assay.feats.active_index(feat_key)
        n_feats = len(feature_index)
        logger.info(
            f"Pseudotime markers: correlating features "
            f"(cells={n_cells}, features={n_feats}, "
            f"batch_size={gene_batch_size if gene_batch_size is not None else 'auto'})"
        )
        correlation_key = f"{cell_key}__{pseudotime_key}__r"
        p_value_key = f"{cell_key}__{pseudotime_key}__p"
        p_value_adjusted_key = f"{cell_key}__{pseudotime_key}__padj"
        storage_backed = all(
            hasattr(self, name)
            for name in (
                "zw",
                "_ensure_cell_selection",
                "_resolve_cell_data_provenance_input",
            )
        ) and hasattr(assay, "z")
        planned = None
        legacy_planned = None
        preserved_displays: dict[str, dict[str, Any] | None] = {}
        if storage_backed:
            cell_selection = self._ensure_cell_selection(cell_key)
            pseudotime_input = self._resolve_cell_data_provenance_input(
                pseudotime_key,
                cell_key=cell_key,
            )
            feature_values = np.asarray(
                assay.feats.fetch_all(feat_key),
                dtype=bool,
            )
            feature_selection = self._resolve_selection_input(
                metadata_group=as_zarr_group(
                    assay.z["featureData"],
                    name="featureData",
                ),
                column=feat_key,
                values=feature_values,
                row_ids=np.asarray(assay.feats.fetch_all("ids")),
                scope="assay",
                kind="feature_selection",
                assay=assay.name,
                invalidate_cache=False,
            )
            arguments = PseudotimeMarkerArguments(
                cell_selection=cell_selection,
                feature_selection=feature_selection,
                pseudotime=pseudotime_input,
                normalization=resolved_norm_params,
                normalization_method=callable_identity(assay.normMethod),
                size_factor=getattr(assay, "sf", None),
                association_method="pearson",
                p_value_method="student_t",
                adjustment_method="fdr_bh",
                adjustment_scope="tested_features",
                min_cells=min_cells,
                from_assay=assay.name,
                cell_key=cell_key,
                feat_key=feat_key,
                pseudotime_key=pseudotime_key,
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
            record = arguments.to_record()
            planned = arguments.plan(
                self.zw,
                scope="assay",
                assay=assay.name,
                invalidate_cache=invalidate_cache,
                required_arrays=(
                    ArrayRequirement(
                        "r_value",
                        shape=(n_feats,),
                        dtype_kind="f",
                    ),
                    ArrayRequirement(
                        "p_value",
                        shape=(n_feats,),
                        dtype_kind="f",
                    ),
                    ArrayRequirement(
                        "p_value_adjusted",
                        shape=(n_feats,),
                        dtype_kind="f",
                    ),
                ),
            )
            if not planned.reused and not invalidate_cache:
                legacy_parameters = dict(record.parameters)
                for field_name in (
                    "association_method",
                    "p_value_method",
                    "adjustment_method",
                    "adjustment_scope",
                ):
                    legacy_parameters.pop(field_name)
                legacy_planned = plan_artifact(
                    self.zw,
                    scope="assay",
                    assay=assay.name,
                    kind=arguments.artifact_kind,
                    operation=arguments.operation,
                    parameters=legacy_parameters,
                    inputs=record.inputs,
                    execution_options=record.execution_options,
                    required_arrays=(
                        ArrayRequirement(
                            "r_value",
                            shape=(n_feats,),
                            dtype_kind="f",
                        ),
                        ArrayRequirement(
                            "p_value",
                            shape=(n_feats,),
                            dtype_kind="f",
                        ),
                    ),
                )
                if not legacy_planned.reused:
                    legacy_planned = None
            else:
                legacy_planned = None
            preserved_displays = {
                correlation_key: feature_column_display(
                    assay.z,
                    correlation_key,
                ),
                p_value_key: feature_column_display(
                    assay.z,
                    p_value_key,
                ),
                p_value_adjusted_key: feature_column_display(
                    assay.z,
                    p_value_adjusted_key,
                ),
            }
        if planned is not None and planned.reused:
            artifact_group = reused_artifact_group(self.zw, planned)
            r_values = artifact_values(artifact_group, "r_value")
            p_values = artifact_values(artifact_group, "p_value")
            p_values_adjusted = artifact_values(
                artifact_group,
                "p_value_adjusted",
            )
            markers = pd.DataFrame(
                {
                    "r_value": r_values,
                    "p_value": p_values,
                    "p_value_adjusted": p_values_adjusted,
                },
                index=feature_index,
            )
        elif legacy_planned is not None and legacy_planned.reused:
            from ...features.markers.correction import _bh_adjusted_pvalues

            artifact_group = reused_artifact_group(self.zw, legacy_planned)
            r_values = artifact_values(artifact_group, "r_value")
            p_values = artifact_values(artifact_group, "p_value")
            p_values_adjusted = _bh_adjusted_pvalues(p_values)
            markers = pd.DataFrame(
                {
                    "r_value": r_values,
                    "p_value": p_values,
                    "p_value_adjusted": p_values_adjusted,
                },
                index=feature_index,
            )
            if planned is not None and getattr(self, "zarr_mode", "r+") == "r+":
                write_cell_data_artifact(
                    self.zw,
                    planned,
                    {
                        "r_value": r_values,
                        "p_value": p_values,
                        "p_value_adjusted": p_values_adjusted,
                    },
                )
        else:
            markers = find_markers_by_regression(
                assay=assay,
                cell_key=cell_key,
                feat_key=feat_key,
                regressor=ptime,
                min_cells=min_cells,
                batch_size=gene_batch_size,
                **resolved_norm_params,
            )
            markers = markers.reindex(feature_index)
            if markers["r_value"].isna().any():
                raise ValueError(
                    "Pseudotime marker results are not aligned to feat_key"
                )
            r_values = np.asarray(markers["r_value"].values)
            p_values = np.asarray(markers["p_value"].values)
            p_values_adjusted = np.asarray(markers["p_value_adjusted"].values)
            if planned is not None and getattr(self, "zarr_mode", "r+") == "r+":
                write_cell_data_artifact(
                    self.zw,
                    planned,
                    {
                        "r_value": r_values,
                        "p_value": p_values,
                        "p_value_adjusted": p_values_adjusted,
                    },
                )
        publish_metadata = (
            not storage_backed
            or getattr(
                self,
                "zarr_mode",
                "r+",
            )
            == "r+"
        )
        if publish_metadata:
            logger.debug("Pseudotime markers: saving marker scores")
            assay.feats.insert(
                correlation_key,
                r_values,
                key=feat_key,
                overwrite=True,
            )
            assay.feats.insert(
                p_value_key,
                p_values,
                key=feat_key,
                overwrite=True,
            )
            assay.feats.insert(
                p_value_adjusted_key,
                p_values_adjusted,
                key=feat_key,
                overwrite=True,
            )
            if planned is not None:
                link_feature_data_column(
                    assay.z,
                    correlation_key,
                    planned.ref,
                    value_name="r_value",
                    default_display=continuous_display(r_values),
                    preserved_display=preserved_displays[correlation_key],
                )
                link_feature_data_column(
                    assay.z,
                    p_value_key,
                    planned.ref,
                    value_name="p_value",
                    default_display=continuous_display(p_values),
                    preserved_display=preserved_displays[p_value_key],
                )
                link_feature_data_column(
                    assay.z,
                    p_value_adjusted_key,
                    planned.ref,
                    value_name="p_value_adjusted",
                    default_display=continuous_display(p_values_adjusted),
                    preserved_display=preserved_displays[p_value_adjusted_key],
                )
        table = markers.rename_axis("feature_index").reset_index()
        feature_names = np.asarray(assay.feats.fetch_all("names"), dtype=object)
        table.insert(
            1,
            "feature_name",
            feature_names[table["feature_index"].to_numpy(dtype=np.int64)],
        )
        if publish_metadata:
            logger.info(f"Stored pseudotime marker scores for {len(table)} features")
        else:
            logger.info(
                f"Loaded pseudotime marker scores for {len(table)} features "
                "without modifying the read-only store"
            )
        return PseudotimeMarkerResult(
            table=table,
            correlation_key=correlation_key,
            p_value_key=p_value_key,
            assay=assay.name,
            cell_key=cell_key,
            feature_key=feat_key,
            pseudotime_key=pseudotime_key,
        )

    def run_pseudotime_aggregation(
        self,
        from_assay: str | None = None,
        cell_key: str | None = None,
        feat_key: str | None = None,
        pseudotime_key: str | None = None,
        cluster_label: str | None = None,
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
    ) -> PseudotimeAggregationResult:
        """Cluster features by their pseudotime-ordered expression profiles.

        Args:
            from_assay: Name of the assay to use. The default assay is used when omitted.
            cell_key: Boolean cell metadata column selecting cells.
            feat_key: Boolean feature metadata column selecting features.
            pseudotime_key: Required numeric cell metadata column containing
                pseudotime values.
            cluster_label: Required new or existing feature metadata column where
                module identities are saved. Existing values are overwritten.
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
            Lazy aggregated matrix with aligned feature indices and clusters.
        """
        from ...trajectory.feature_dynamics import knn_clustering

        reject_unknown_normalization_params(
            norm_params,
            caller="run_pseudotime_aggregation",
        )
        feat_key = feat_key or "I"
        from_assay, cell_key, _ = self._get_latest_keys(
            from_assay,
            cell_key,
            feat_key,
        )
        assay = self._get_assay(from_assay)

        if pseudotime_key is None:
            raise ValueError(
                "ERROR: Please provide a value for `pseudotime_key` parameter. This is the column in "
                "the cell attribute table that contains the pseudotime values."
            )
        if cluster_label is None:
            raise ValueError(
                "ERROR: Please provide a value for cluster_label. "
                "It will be used to create new column in feature attribute table. The module identity "
                "of each feature will be saved under this column name. If this column already exists "
                "then it will be overwritten."
            )
        if not isinstance(nan_cluster_value, (int, np.integer)) or isinstance(
            nan_cluster_value,
            (bool, np.bool_),
        ):
            raise TypeError("nan_cluster_value must be an integer")
        nan_cluster_value = int(nan_cluster_value)
        _validate_assay_pseudotime(assay, cell_key, pseudotime_key)
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
            _cell_indices,
            feature_indices,
            effective_window,
            effective_bins,
            input_fingerprints,
            _legacy_parameters,
        ) = assay._prepare_aggregated_ordering(
            cell_key,
            feat_key,
            pseudotime_key,
            min_exp=min_exp,
            window_size=window_size,
            chunk_size=chunk_size,
            smoothen=smoothen,
            z_scale=z_scale,
            norm_params=resolved_norm_params,
        )
        cell_selection = self._ensure_cell_selection(cell_key)
        pseudotime_input = self._resolve_cell_data_provenance_input(
            pseudotime_key,
            cell_key=cell_key,
        )
        feature_values = np.asarray(
            assay.feats.fetch_all(feat_key),
            dtype=bool,
        )
        feature_selection = self._resolve_selection_input(
            metadata_group=as_zarr_group(
                assay.z["featureData"],
                name="featureData",
            ),
            column=feat_key,
            values=feature_values,
            row_ids=np.asarray(assay.feats.fetch_all("ids")),
            scope="assay",
            kind="feature_selection",
            assay=assay.name,
            invalidate_cache=False,
        )
        arguments = PseudotimeAggregationArguments(
            cell_selection=cell_selection,
            feature_selection=feature_selection,
            pseudotime=pseudotime_input,
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
            from_assay=assay.name,
            cell_key=cell_key,
            feat_key=feat_key,
            pseudotime_key=pseudotime_key,
            cluster_label=cluster_label,
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
        preserved_display = feature_column_display(assay.z, cluster_label)
        if planned.reused:
            aggregation_group = reused_artifact_group(self.zw, planned)
            full_data = ChunkedArray(
                as_zarr_array(aggregation_group["data"], name="data"),
                nthreads=self.nthreads,
            )
            stored_feature_indices = np.asarray(
                as_zarr_array(
                    aggregation_group["feature_indices"],
                    name="feature_indices",
                )[:]
            )
            valid_features = np.asarray(
                as_zarr_array(
                    aggregation_group["valid_features"],
                    name="valid_features",
                )[:],
                dtype=bool,
            )
            stored_feature_clusters = np.asarray(
                as_zarr_array(
                    aggregation_group["feature_clusters"],
                    name="feature_clusters",
                )[:],
                dtype=np.int64,
            )
            cluster_values = np.asarray(
                as_zarr_array(
                    aggregation_group["cluster_values"],
                    name="cluster_values",
                )[:],
                dtype=np.int64,
            )
        else:
            logger.info("Pseudotime modules: aggregating feature profiles")
            aggregation_group = start_artifact(self.zw, planned)
            full_data, stored_feature_indices, valid_features = (
                assay._write_aggregated_ordering_group(
                    aggregation_group,
                    cell_key=cell_key,
                    feat_key=feat_key,
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
        df = full_data[valid_features]
        feat_ids = stored_feature_indices[valid_features]
        clusts = stored_feature_clusters[valid_features]
        logger.debug("Pseudotime modules: saving module labels")
        assay.feats.insert(
            cluster_label,
            cluster_values,
            fill_value=nan_cluster_value,
            overwrite=True,
        )
        link_feature_data_column(
            assay.z,
            cluster_label,
            planned.ref,
            value_name="cluster_values",
            default_display=categorical_display(cluster_values),
            preserved_display=preserved_display,
        )

        legacy_cluster_digest: str | None = None
        for assay_name in self.assay_names:
            grouped_assay = self._get_assay(assay_name)
            if (
                grouped_assay.attrs.get("grouped_from_assay") != assay.name
                or grouped_assay.attrs.get("grouped_group_key") != cluster_label
            ):
                continue
            raw_group_artifact = grouped_assay.attrs.get("grouped_group_artifact")
            if isinstance(raw_group_artifact, dict):
                stale = raw_group_artifact != planned.ref.to_dict()
            else:
                if legacy_cluster_digest is None:
                    legacy_cluster_digest = _group_assignment_digest(cluster_values)
                stale = (
                    grouped_assay.attrs.get("grouped_group_digest")
                    != legacy_cluster_digest
                )
            if stale:
                logger.warning(
                    f"Grouped assay '{assay_name}' is stale after updating "
                    f"feature groups in '{cluster_label}'. Rerun add_grouped_assay"
                )
        logger.info(f"Stored {np.unique(clusts).size} pseudotime modules")
        return PseudotimeAggregationResult(
            data=df,
            feature_indices=np.asarray(feat_ids),
            feature_clusters=np.asarray(clusts),
            cluster_key=cluster_label,
            storage_path=(
                f"{str(self.zw.path).strip('/')}/{artifact_path(planned.ref)}"
                if str(self.zw.path).strip("/")
                else artifact_path(planned.ref)
            ),
            assay=assay.name,
            cell_key=cell_key,
            feature_key=feat_key,
            pseudotime_key=pseudotime_key,
        )
