from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import pandas as pd

from ...graph.errors import IncompatibleAnalysisStateError
from ...graph.state import resolve_graph_selection
from ...metadata.artifacts import (
    artifact_values,
    categorical_display,
    column_display,
    continuous_display,
    link_cell_data_column,
    plan_cell_data_artifact,
    write_cell_data_artifact,
)
from ...metadata.arguments import LeidenArguments, TopacedoArguments
from ...storage.artifacts import (
    ArtifactRef,
    artifact_path,
    inspect_artifact,
)
from ...storage.artifact_writer import (
    ArrayRequirement,
    PlannedArtifact,
    finish_artifact,
    plan_artifact,
    reused_artifact_group,
    start_artifact,
)
from ...storage.arrays import create_zarr_dataset
from ...storage.types import as_zarr_array, as_zarr_group
from ...utils.logging import logger

if TYPE_CHECKING:
    from ...clustering.paris_multiscale import ParisClusteringResult
    from .graph import _GraphOperationsMixin as _ClusteringOperationsBase
else:
    _ClusteringOperationsBase = object


@dataclass(frozen=True, slots=True)
class _PreparedLeidenClustering:
    planned: PlannedArtifact
    graph: ArtifactRef
    graph_loc: str
    from_assay: str
    label_assay: str
    cell_key: str
    resolution: float
    backend: Literal["igraph", "leidenalg"]
    symmetric_graph: bool
    graph_upper_only: bool
    random_seed: int
    label: str
    n_cells: int

    @property
    def graph_key(self) -> tuple[str, bool, bool]:
        return (
            self.graph_loc,
            self.symmetric_graph,
            self.graph_upper_only,
        )


class _ClusteringOperationsMixin(_ClusteringOperationsBase):
    def _run_paris_from_artifacts(
        self,
        *,
        graph_ref: ArtifactRef,
        graph_loc: str,
        from_assay: str,
        label_assay: str,
        cell_key: str,
        fixed_cluster_count: int | None,
        effective_min_cluster_size: int | None,
        label: str,
        force_recalc: bool,
    ) -> "ParisClusteringResult":
        from ...clustering._paris_modularity import modularity_split_gains
        from ...clustering.paris import (
            fit_paris_hierarchy,
            hierarchy_to_dendrogram,
            straight_cut,
        )
        from ...clustering.paris_multiscale import (
            ParisClusterDiagnostic,
            ParisClusteringResult,
            adaptive_cut,
            collapse_equal_height_plateaus,
        )
        from .paris_persistence import (
            load_hierarchy_group,
            preflight_hierarchy_artifact_cut,
            preflight_paris_adaptive_cut,
            preflight_paris_fit,
            write_hierarchy_group,
        )

        artifact_scope = graph_ref.scope
        artifact_assay = graph_ref.assay
        n_cells, _effective_k = self._get_graph_ncells_k(graph_loc)
        cut_mode: Literal["adaptive", "fixed"] = (
            "fixed" if fixed_cluster_count is not None else "adaptive"
        )
        graph_group = as_zarr_group(self.zw[graph_loc], name=graph_loc)
        budget = self.resources
        hierarchy_plan = plan_artifact(
            self.zw,
            scope=artifact_scope,
            assay=artifact_assay,
            kind="cluster_hierarchy",
            operation="fit_paris_hierarchy",
            parameters={},
            inputs={"connectivity_map": graph_ref},
            execution_options={"invalidate_cache": force_recalc},
            invalidate_cache=force_recalc,
            required_arrays=(
                ArrayRequirement("children"),
                ArrayRequirement("heights"),
                ArrayRequirement("sizes"),
                ArrayRequirement("component_roots"),
                ArrayRequirement("synthetic_joins"),
            ),
        )
        hierarchy = plateau_forest = None
        fitted_graph = None
        if hierarchy_plan.reused:
            try:
                hierarchy_group = reused_artifact_group(
                    self.zw,
                    hierarchy_plan,
                )
                preflight_hierarchy_artifact_cut(
                    hierarchy_group,
                    cut_mode,
                    budget,
                )
                hierarchy, plateau_forest = load_hierarchy_group(
                    hierarchy_group,
                    hierarchy_plan.ref.artifact_id,
                )
            except (KeyError, TypeError, ValueError):
                hierarchy_plan = plan_artifact(
                    self.zw,
                    scope=artifact_scope,
                    assay=artifact_assay,
                    kind="cluster_hierarchy",
                    operation="fit_paris_hierarchy",
                    parameters={},
                    inputs={"connectivity_map": graph_ref},
                    execution_options={"invalidate_cache": force_recalc},
                    invalidate_cache=True,
                )
        if hierarchy is None or plateau_forest is None:
            estimated_peak_bytes = preflight_paris_fit(
                graph_group,
                n_cells,
                budget,
            )
            fitted_graph = self.load_graph(
                graph_ref,
                from_assay=from_assay,
                cell_key=cell_key,
                symmetric=False,
                upper_only=False,
            )
            hierarchy = fit_paris_hierarchy(
                fitted_graph,
                nthreads=budget.workers,
            )
            plateau_forest = collapse_equal_height_plateaus(hierarchy)
            hierarchy_group = start_artifact(self.zw, hierarchy_plan)
            write_hierarchy_group(hierarchy_group, hierarchy, plateau_forest)
            hierarchy_group.attrs["estimated_peak_bytes"] = estimated_peak_bytes
            finish_artifact(hierarchy_group, hierarchy_plan)
        if hierarchy.n_leaves != n_cells:
            raise ValueError("Paris hierarchy size does not match graph")

        mode = "fixed" if fixed_cluster_count is not None else "auto"
        cut_parameters = (
            {"mode": mode, "n_clusters": fixed_cluster_count}
            if fixed_cluster_count is not None
            else {
                "mode": mode,
                "min_cluster_size": effective_min_cluster_size,
            }
        )
        current_selection = self._ensure_cell_selection(cell_key)
        cell_selection = self._graph_cell_selection(graph_ref)
        if not self._selection_artifacts_match(
            cell_selection,
            current_selection,
        ):
            raise ValueError("cell_key does not match the graph cell selection")
        cut_inputs = {
            "cluster_hierarchy": hierarchy_plan.ref,
            "connectivity_map": graph_ref,
            "cell_selection": cell_selection,
        }
        cut_plan = plan_artifact(
            self.zw,
            scope=artifact_scope,
            assay=artifact_assay,
            kind="cluster_cut",
            operation="cut_paris_hierarchy",
            parameters=cut_parameters,
            inputs=cut_inputs,
            execution_options={
                "label": label,
                "invalidate_cache": force_recalc,
            },
            invalidate_cache=force_recalc,
            required_arrays=(
                ArrayRequirement("labels", shape=(n_cells,), dtype_kind="i"),
            ),
        )
        result = None
        if cut_plan.reused:
            cut_group = reused_artifact_group(self.zw, cut_plan)
            try:
                raw_diagnostics = cut_group.attrs.get("diagnostics", [])
                if not isinstance(raw_diagnostics, list) or any(
                    not isinstance(diagnostic, dict) for diagnostic in raw_diagnostics
                ):
                    raise TypeError("Paris diagnostics must be mappings")
                diagnostics = tuple(
                    ParisClusterDiagnostic(**diagnostic)
                    for diagnostic in raw_diagnostics
                )
                labels = np.asarray(
                    as_zarr_array(cut_group["labels"], name="labels")[:],
                    dtype=np.int32,
                )
                result = ParisClusteringResult(
                    labels=labels,
                    mode=cast(Literal["auto", "fixed"], mode),
                    n_clusters=int(
                        cast(
                            int | float | str,
                            cut_group.attrs["n_clusters"],
                        )
                    ),
                    diagnostics=diagnostics,
                    min_cluster_size=effective_min_cluster_size,
                )
            except (KeyError, TypeError, ValueError):
                cut_plan = plan_artifact(
                    self.zw,
                    scope=artifact_scope,
                    assay=artifact_assay,
                    kind="cluster_cut",
                    operation="cut_paris_hierarchy",
                    parameters=cut_parameters,
                    inputs=cut_inputs,
                    execution_options={
                        "label": label,
                        "invalidate_cache": True,
                    },
                    invalidate_cache=True,
                )
        if result is None:
            if fixed_cluster_count is None:
                assert effective_min_cluster_size is not None
                if fitted_graph is None:
                    preflight_paris_adaptive_cut(
                        graph_group,
                        n_cells,
                        budget,
                    )
                    fitted_graph = self.load_graph(
                        graph_ref,
                        from_assay=from_assay,
                        cell_key=cell_key,
                    )
                split_gate = modularity_split_gains(
                    hierarchy,
                    plateau_forest,
                    fitted_graph,
                )
                result = adaptive_cut(
                    hierarchy,
                    effective_min_cluster_size,
                    plateau_forest=plateau_forest,
                    split_gate=split_gate,
                )
            else:
                dendrogram = hierarchy_to_dendrogram(hierarchy)
                labels = straight_cut(dendrogram, fixed_cluster_count).astype(
                    np.int32,
                    copy=False,
                )
                result = ParisClusteringResult(
                    labels=labels,
                    mode="fixed",
                    n_clusters=int(np.unique(labels).size),
                )
            cut_group = start_artifact(self.zw, cut_plan)
            labels_array = create_zarr_dataset(
                cut_group,
                "labels",
                (min(max(n_cells, 1), 100_000),),
                "i4",
                result.labels.shape,
            )
            labels_array[:] = result.labels
            cut_group.attrs["n_clusters"] = int(result.n_clusters)
            cut_group.attrs["diagnostics"] = [
                asdict(diagnostic) for diagnostic in result.diagnostics
            ]
            finish_artifact(cut_group, cut_plan)

        if fixed_cluster_count is not None:
            dendrogram_plan = plan_artifact(
                self.zw,
                scope=artifact_scope,
                assay=artifact_assay,
                kind="dendrogram",
                operation="materialize_paris_dendrogram",
                parameters={"compatibility": True},
                inputs={"cluster_hierarchy": hierarchy_plan.ref},
                execution_options={},
                required_arrays=(ArrayRequirement("data", dtype_kind="f"),),
            )
            if not dendrogram_plan.reused:
                dendrogram = hierarchy_to_dendrogram(
                    hierarchy,
                    compatibility=True,
                )
                dendrogram_group = start_artifact(self.zw, dendrogram_plan)
                dendrogram_array = create_zarr_dataset(
                    dendrogram_group,
                    "data",
                    (min(max(dendrogram.shape[0], 1), 5000), 4),
                    "f8",
                    dendrogram.shape,
                )
                dendrogram_array[:] = dendrogram
                finish_artifact(dendrogram_group, dendrogram_plan)

        final_label_key = self._col_renamer(label_assay, cell_key, label)
        preserved_display = column_display(self.zw, final_label_key)
        self.cells.insert(
            final_label_key,
            result.labels,
            fill_value=-1,
            key=cell_key,
            overwrite=True,
        )
        link_cell_data_column(
            self.zw,
            final_label_key,
            cut_plan.ref,
            value_name="labels",
            default_display=categorical_display(result.labels),
            preserved_display=preserved_display,
        )
        action = "Reused" if cut_plan.reused else "Stored"
        logger.info(f"{action} Paris clustering with {result.n_clusters} clusters")
        return replace(
            result,
            label_key=final_label_key,
            hierarchy_generation_id=hierarchy_plan.ref.artifact_id,
            ref=cut_plan.ref,
        )

    def _prepare_leiden_clustering(
        self,
        graph: ArtifactRef | None = None,
        *,
        from_assay: str | None = None,
        cell_key: str | None = None,
        resolution: float = 1.0,
        backend: Literal["igraph", "leidenalg"] = "igraph",
        symmetric_graph: bool = False,
        graph_upper_only: bool = False,
        label: str = "leiden_cluster",
        random_seed: int = 4444,
        invalidate_cache: bool = False,
    ) -> _PreparedLeidenClustering:
        if backend not in {"igraph", "leidenalg"}:
            raise ValueError("backend must be 'igraph' or 'leidenalg'")
        graph_selection = resolve_graph_selection(
            self,
            graph,
            from_assay=from_assay,
            cell_key=cell_key,
        )
        from_assay = graph_selection.from_assay
        cell_key = graph_selection.cell_key
        n_cells, _effective_k = self._get_graph_ncells_k(graph_selection.graph_loc)
        graph_input = graph_selection.graph_ref
        artifact_scope = graph_input.scope
        selection = self._ensure_cell_selection(cell_key)
        graph_cell_selection = self._graph_cell_selection(graph_input)
        if not self._selection_artifacts_match(graph_cell_selection, selection):
            raise ValueError("cell_key does not match the graph cell selection")
        selection = graph_cell_selection
        arguments = LeidenArguments(
            graph=graph_input,
            resolution=resolution,
            backend=backend,
            symmetric_graph=symmetric_graph,
            graph_upper_only=graph_upper_only,
            random_seed=random_seed,
            label=label,
            from_assay=from_assay,
            cell_key=cell_key,
            invalidate_cache=invalidate_cache,
        )
        record = arguments.to_record()
        planned = plan_cell_data_artifact(
            self.zw,
            scope=artifact_scope,
            assay=(
                graph_input.assay
                if graph_input.scope == "assay"
                else from_assay
                if artifact_scope == "assay"
                else None
            ),
            kind=arguments.artifact_kind,
            operation=arguments.operation,
            parameters=record.parameters,
            inputs=record.inputs,
            execution_options=record.execution_options,
            cell_selection=selection,
            arrays={"values": ((n_cells,), "i")},
            invalidate_cache=invalidate_cache,
        )
        return _PreparedLeidenClustering(
            planned=planned,
            graph=graph_input,
            graph_loc=graph_selection.graph_loc,
            from_assay=from_assay,
            label_assay=graph_selection.output_assay,
            cell_key=cell_key,
            resolution=resolution,
            backend=backend,
            symmetric_graph=symmetric_graph,
            graph_upper_only=graph_upper_only,
            random_seed=random_seed,
            label=label,
            n_cells=n_cells,
        )

    def _load_prepared_leiden_graph(
        self,
        prepared: _PreparedLeidenClustering,
    ) -> Any:
        graph = self.load_graph(
            prepared.graph,
            from_assay=prepared.from_assay,
            cell_key=prepared.cell_key,
            symmetric=prepared.symmetric_graph,
            upper_only=prepared.graph_upper_only,
        )
        return graph.tocsr()

    @staticmethod
    def _compute_prepared_leiden(
        prepared: _PreparedLeidenClustering,
        graph: Any,
    ) -> np.ndarray:
        from ...clustering.leiden import leiden_membership

        if prepared.planned.reused:
            raise ValueError("Cannot recompute a reusable Leiden artifact")
        return leiden_membership(
            graph,
            prepared.resolution,
            prepared.random_seed,
            backend=prepared.backend,
        )

    def _publish_prepared_leiden(
        self,
        prepared: _PreparedLeidenClustering,
        membership: np.ndarray | None,
    ) -> tuple[str, ArtifactRef]:
        if prepared.planned.reused:
            artifact_group = as_zarr_group(
                self.zw[artifact_path(prepared.planned.ref)],
                name=prepared.planned.ref.artifact_id,
            )
            membership = artifact_values(artifact_group, "values")
        else:
            if membership is None:
                raise ValueError("Leiden membership is required for a new artifact")
            membership = np.asarray(membership)
            if membership.shape != (prepared.n_cells,):
                raise ValueError(
                    "Leiden membership must contain one label per graph cell"
                )
            if membership.dtype.kind not in {"i", "u"}:
                raise TypeError("Leiden membership must contain integer labels")
            write_cell_data_artifact(
                self.zw,
                prepared.planned,
                {"values": membership},
            )
        column = self._col_renamer(
            prepared.label_assay,
            prepared.cell_key,
            prepared.label,
        )
        preserved_display = column_display(self.zw, column)
        self.cells.insert(
            column,
            membership,
            fill_value=-1,
            key=prepared.cell_key,
            overwrite=True,
        )
        link_cell_data_column(
            self.zw,
            column,
            prepared.planned.ref,
            value_name="values",
            default_display=categorical_display(membership),
            preserved_display=preserved_display,
        )
        action = "Reused" if prepared.planned.reused else "Stored"
        logger.info(
            f"{action} Leiden clustering with {np.unique(membership).size} clusters"
        )
        return column, prepared.planned.ref

    def run_leiden_clustering(
        self,
        graph: ArtifactRef | None = None,
        *,
        from_assay: str | None = None,
        cell_key: str | None = None,
        resolution: float = 1.0,
        backend: Literal["igraph", "leidenalg"] = "igraph",
        symmetric_graph: bool = False,
        graph_upper_only: bool = False,
        label: str = "leiden_cluster",
        random_seed: int = 4444,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Execute Leiden clustering and save identities in cell metadata.

        Args:
            graph: Connectivity map or integrated graph to partition. The
                current analysis chain of the assay is used when omitted.
            from_assay: Assay whose current graph should be used.
            cell_key: Cell key of the graph.
            resolution: Leiden resolution parameter.
            backend: Leiden implementation. Native igraph is the default.
            symmetric_graph: Forwarded to `load_graph`.
            graph_upper_only: Forwarded to `load_graph`.
            label: Base name of the cell-metadata column that receives labels.
            random_seed: Seed for the Leiden optimizer.
            invalidate_cache: Force a new cluster-labels artifact.

        Returns:
            Reference to the cluster-labels artifact backing the label column.
        """
        prepared = self._prepare_leiden_clustering(
            graph,
            from_assay=from_assay,
            cell_key=cell_key,
            resolution=resolution,
            backend=backend,
            symmetric_graph=symmetric_graph,
            graph_upper_only=graph_upper_only,
            label=label,
            random_seed=random_seed,
            invalidate_cache=invalidate_cache,
        )
        membership = None
        if not prepared.planned.reused:
            graph_matrix = self._load_prepared_leiden_graph(prepared)
            membership = self._compute_prepared_leiden(prepared, graph_matrix)
        _column, ref = self._publish_prepared_leiden(prepared, membership)
        return ref

    def run_paris_clustering(
        self,
        graph: ArtifactRef | None = None,
        *,
        from_assay: str | None = None,
        cell_key: str | None = None,
        n_clusters: int | Literal["auto"] = "auto",
        min_cluster_size: int | None = None,
        force_recalc: bool = False,
        invalidate_cache: bool = False,
        label: str = "paris_cluster",
    ) -> "ParisClusteringResult":
        """Fit the canonical Paris hierarchy and write a fixed or adaptive cut.

        Pass ``graph`` to partition an explicit connectivity map or integrated
        graph. The returned result carries the cut artifact in its ``ref``
        field alongside the labels and diagnostics.
        """
        if isinstance(n_clusters, (bool, np.bool_)):
            raise TypeError("n_clusters must be an integer or 'auto'")
        if isinstance(n_clusters, str):
            if n_clusters != "auto":
                raise ValueError("n_clusters must be an integer or 'auto'")
            fixed_cluster_count = None
        elif isinstance(n_clusters, (int, np.integer)):
            if n_clusters < 1:
                raise ValueError("n_clusters must be positive")
            fixed_cluster_count = int(n_clusters)
        else:
            raise TypeError("n_clusters must be an integer or 'auto'")
        if fixed_cluster_count is not None and min_cluster_size is not None:
            raise ValueError("min_cluster_size is only valid when n_clusters='auto'")
        invalidate_artifacts = force_recalc or invalidate_cache

        graph_selection = resolve_graph_selection(
            self,
            graph,
            from_assay=from_assay,
            cell_key=cell_key,
        )
        from_assay = graph_selection.from_assay
        cell_key = graph_selection.cell_key
        graph_loc = graph_selection.graph_loc
        n_cells, effective_k = self._get_graph_ncells_k(graph_loc)
        active_cell_count = int(np.count_nonzero(self.cells.fetch_all(cell_key)))
        if active_cell_count != n_cells:
            raise ValueError(
                f"cell_key {cell_key!r} selects {active_cell_count} cells, "
                f"but the graph contains {n_cells}"
            )
        if fixed_cluster_count is not None and fixed_cluster_count > n_cells:
            raise ValueError(f"n_clusters must not exceed the graph size ({n_cells})")

        if fixed_cluster_count is None:
            if min_cluster_size is None:
                effective_min_cluster_size = effective_k + 1
            else:
                if isinstance(min_cluster_size, (bool, np.bool_)) or not isinstance(
                    min_cluster_size,
                    (int, np.integer),
                ):
                    raise TypeError("min_cluster_size must be an integer")
                if min_cluster_size < 2:
                    raise ValueError("min_cluster_size must be at least 2")
                effective_min_cluster_size = int(min_cluster_size)
        else:
            effective_min_cluster_size = None

        return self._run_paris_from_artifacts(
            graph_ref=graph_selection.graph_ref,
            graph_loc=graph_loc,
            from_assay=from_assay,
            label_assay=graph_selection.output_assay,
            cell_key=cell_key,
            fixed_cluster_count=fixed_cluster_count,
            effective_min_cluster_size=effective_min_cluster_size,
            label=label,
            force_recalc=invalidate_artifacts,
        )

    def run_topacedo_sampler(
        self,
        graph: ArtifactRef | None = None,
        *,
        from_assay: str | None = None,
        cell_key: str | None = None,
        cluster_key: str | None = None,
        use_k: int | None = None,
        density_depth: int = 2,
        density_bandwidth: float = 5.0,
        max_sampling_rate: float = 0.05,
        min_sampling_rate: float = 0.01,
        min_cells_per_group: int = 3,
        snn_bandwidth: float = 5.0,
        seed_reward: float = 3.0,
        non_seed_reward: float = 0,
        edge_cost_multiplier: float = 1.0,
        edge_cost_bandwidth: float = 10.0,
        save_sampling_key: str = "sketched",
        save_density_key: str = "cell_density",
        save_mean_snn_key: str = "snn_value",
        save_seeds_key: str = "sketch_seeds",
        rand_state: int = 4466,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Perform sub-sampling (aka sketching) of cells using TopACeDo
        algorithm. Sub-sampling required that cells are partitioned in cluster
        already. Since, sub-sampling is dependent on cluster information,
        having, large number of homogeneous and even sized cluster improves
        sub-sampling results.

        Args:
            graph: Connectivity map or integrated graph to sample. The current
                   analysis chain of the assay is used when omitted.
            from_assay: Name of assay to be used. If no value is provided then the default assay will be used.
            cell_key: Cell key. Should be same as the one that was used in the desired graph. (Default value: 'I')
            cluster_key: Name of the column in cell metadata table where cluster information is stored.
            use_k: Number of top k-nearest neighbours to retain in the graph over which downsampling is performed.
                   BY default all neighbours are used. (Default value: None)
            density_depth: Same as 'search_depth' parameter in `calc_neighbourhood_density`. (Default value: 2)
            density_bandwidth: This value is used to scale the penalty affected by neighbourhood density. Higher values
                               will lead to a larger penalty. (Default value: 5.0)
            max_sampling_rate: Maximum fraction of cells to sample from each group. The effective sampling rate is lower
                               than this value depending on the neighbourhood degree and SNN density of cells.
                               Should be greater than 0 and less than 1. (Default value: 0.05)
            min_sampling_rate: Minimum sampling rate. Effective sampling rate is not allowed to be lower than this
                               value. Should be greater than 0 and less than 1. (Default value: 0.01)
            min_cells_per_group: Minimum number of cells to sample from each group. (Default value: 3)
            snn_bandwidth: Bandwidth for the shared nearest neighbour award. Clusters with higher mean SNN values get
                           lower sampling penalty. This value, is raised to mean SNN value of the cluster to obtain
                           sampling reward of the cluster. (Default value: 5.0)
            seed_reward: Reward/prize value for seed nodes. (Default value: 3.0)
            non_seed_reward: Reward/prize for non-seed nodes. (Default value: 0)
            edge_cost_multiplier: This value is multiplier to each edge's cost. Higher values will make graph traversal
                                  costly and might lead to removal of poorly connected nodes (Default value: 1.0)
            edge_cost_bandwidth: This value is raised to edge cost to get an adjusted edge cost (Default value: 10.0)
            save_sampling_key: base label for marking the cells that were sampled into a cell metadata column
                               (Default value: 'sketched')
            save_density_key: base label for saving the cell neighbourhood densities into a cell metadata column
                              (Default value: 'cell_density')
            save_mean_snn_key: base label for saving the SNN value for each cell (identified by topacedo sampler) into
                               a cell metadata column (Default value: 'snn_value')
            save_seeds_key: base label for saving the seed cells (identified by topacedo sampler) into a cell
                            metadata column (Default value: 'sketch_seeds')
            rand_state: A random values to set seed while sampling cells from a cluster randomly. (Default value: 4466)

        Returns:
            Reference to the sampling artifact. Open it with ``load_artifact``
            to read the ``edges`` array of Steiner tree edges over graph row
            indices.
        """

        graph_selection = resolve_graph_selection(
            self,
            graph,
            from_assay=from_assay,
            cell_key=cell_key,
        )
        from_assay = graph_selection.from_assay
        cell_key = graph_selection.cell_key
        output_assay = graph_selection.output_assay
        if cluster_key is None:
            raise ValueError("ERROR: Please provide a value for cluster key")
        clusters = pd.Series(self.cells.fetch(cluster_key, key=cell_key))
        graph_input = graph_selection.graph_ref
        cluster_column = as_zarr_array(
            as_zarr_group(self.zw["cellData"], name="cellData")[cluster_key],
            name=cluster_key,
        )
        raw_cut_ref = cluster_column.attrs.get("source_artifact")
        if isinstance(raw_cut_ref, dict):
            try:
                cut_ref = ArtifactRef.from_dict(raw_cut_ref)
            except (TypeError, ValueError) as error:
                raise IncompatibleAnalysisStateError(
                    "TopACeDo cluster state has a malformed artifact reference",
                    code="invalid_analysis_state",
                    context={"cluster_key": cluster_key},
                ) from error
        else:
            cut_ref = None
        if cut_ref is not None and cut_ref.kind == "cluster_cut":
            cluster_input: object = cut_ref
            cut_inputs = inspect_artifact(self.zw, cut_ref).inputs or {}
            raw_hierarchy_ref = cut_inputs.get("cluster_hierarchy")
            current_graph_input = graph_input.to_dict()
            if cut_inputs.get("connectivity_map") != current_graph_input:
                raise ValueError("Cluster cut does not belong to the requested graph")
            cut_group = as_zarr_group(
                self.zw[inspect_artifact(self.zw, cut_ref).path],
                name=cut_ref.artifact_id,
            )
            clusters = pd.Series(
                np.asarray(as_zarr_array(cut_group["labels"], name="labels")[:])
            )
        else:
            raise IncompatibleAnalysisStateError(
                "TopACeDo requires a cluster_cut artifact from Paris clustering",
                code="invalid_analysis_state",
                context={
                    "cluster_key": cluster_key,
                    "artifact_kind": None if cut_ref is None else cut_ref.kind,
                },
            )
        if isinstance(raw_hierarchy_ref, dict):
            from ...clustering.paris import hierarchy_to_dendrogram
            from .paris_persistence import load_hierarchy_group

            hierarchy_ref = ArtifactRef.from_dict(raw_hierarchy_ref)
            dendrogram_plan = plan_artifact(
                self.zw,
                scope=hierarchy_ref.scope,
                assay=hierarchy_ref.assay,
                kind="dendrogram",
                operation="materialize_paris_dendrogram",
                parameters={"compatibility": True},
                inputs={"cluster_hierarchy": hierarchy_ref},
                execution_options={},
                required_arrays=(ArrayRequirement("data", dtype_kind="f"),),
            )
            if not dendrogram_plan.reused:
                hierarchy_group = as_zarr_group(
                    self.zw[inspect_artifact(self.zw, hierarchy_ref).path],
                    name=hierarchy_ref.artifact_id,
                )
                hierarchy, _plateau = load_hierarchy_group(
                    hierarchy_group,
                    hierarchy_ref.artifact_id,
                )
                dendrogram = hierarchy_to_dendrogram(
                    hierarchy,
                    compatibility=True,
                )
                dendrogram_group = start_artifact(self.zw, dendrogram_plan)
                output = create_zarr_dataset(
                    dendrogram_group,
                    "data",
                    (min(max(dendrogram.shape[0], 1), 5000), 4),
                    "f8",
                    dendrogram.shape,
                )
                output[:] = dendrogram
                finish_artifact(dendrogram_group, dendrogram_plan)
            else:
                dendrogram_group = reused_artifact_group(self.zw, dendrogram_plan)
            dendrogram = np.asarray(
                as_zarr_array(dendrogram_group["data"], name="data")[:]
            )
            dendrogram_input: object = dendrogram_plan.ref
        else:
            raise IncompatibleAnalysisStateError(
                "TopACeDo cluster state does not name its Paris hierarchy",
                code="invalid_analysis_state",
                context={"cluster_key": cluster_key},
            )

        graph_matrix = self.load_graph(
            graph_input,
            from_assay=from_assay,
            cell_key=cell_key,
            symmetric=False,
            upper_only=False,
            use_k=use_k,
        )

        if len(clusters) != graph_matrix.shape[0]:
            raise ValueError(
                f"ERROR: cluster information exists for {len(clusters)} cells while graph has "
                f"{graph_matrix.shape[0]} cells."
            )
        selection = self._ensure_cell_selection(cell_key)
        graph_cell_selection = self._graph_cell_selection(graph_input)
        if not self._selection_artifacts_match(graph_cell_selection, selection):
            raise ValueError("cell_key does not match the graph cell selection")
        selection = graph_cell_selection
        artifact_scope = graph_input.scope
        arguments = TopacedoArguments(
            graph=graph_input,
            clusters=cluster_input,
            dendrogram=dendrogram_input,
            cell_selection=selection,
            use_k=use_k,
            density_depth=density_depth,
            density_bandwidth=density_bandwidth,
            max_sampling_rate=max_sampling_rate,
            min_sampling_rate=min_sampling_rate,
            min_cells_per_group=min_cells_per_group,
            snn_bandwidth=snn_bandwidth,
            seed_reward=seed_reward,
            non_seed_reward=non_seed_reward,
            edge_cost_multiplier=edge_cost_multiplier,
            edge_cost_bandwidth=edge_cost_bandwidth,
            rand_state=rand_state,
            from_assay=from_assay,
            cell_key=cell_key,
            cluster_key=cluster_key,
            save_sampling_key=save_sampling_key,
            save_density_key=save_density_key,
            save_mean_snn_key=save_mean_snn_key,
            save_seeds_key=save_seeds_key,
            invalidate_cache=invalidate_cache,
        )
        planned = arguments.plan(
            self.zw,
            scope=artifact_scope,
            assay=(
                graph_input.assay
                if graph_input.scope == "assay"
                else from_assay
                if artifact_scope == "assay"
                else None
            ),
            invalidate_cache=invalidate_cache,
            required_arrays=(
                ArrayRequirement(
                    "sampled",
                    shape=(graph_matrix.shape[0],),
                    dtype_kind="b",
                ),
                ArrayRequirement(
                    "density",
                    shape=(graph_matrix.shape[0],),
                    dtype_kind="f",
                ),
                ArrayRequirement(
                    "mean_snn",
                    shape=(graph_matrix.shape[0],),
                    dtype_kind="f",
                ),
                ArrayRequirement(
                    "seeds",
                    shape=(graph_matrix.shape[0],),
                    dtype_kind="b",
                ),
                ArrayRequirement(
                    "edges",
                    shape=(None, 2),
                    dtype_kind="i",
                ),
            ),
        )
        columns = {
            "sampled": self._col_renamer(
                output_assay,
                cell_key,
                save_sampling_key,
            ),
            "density": self._col_renamer(
                output_assay,
                cell_key,
                save_density_key,
            ),
            "mean_snn": self._col_renamer(
                output_assay,
                cell_key,
                save_mean_snn_key,
            ),
            "seeds": self._col_renamer(
                output_assay,
                cell_key,
                save_seeds_key,
            ),
        }
        preserved_displays = {
            name: column_display(self.zw, column) for name, column in columns.items()
        }
        if planned.reused:
            artifact_group = reused_artifact_group(self.zw, planned)
            sampled = artifact_values(artifact_group, "sampled").astype(bool)
            density = artifact_values(artifact_group, "density")
            mean_snn = artifact_values(artifact_group, "mean_snn")
            seeds = artifact_values(artifact_group, "seeds").astype(bool)
        else:
            try:
                from topacedo import TopacedoSampler
            except ImportError as error:
                raise ImportError("Could not find topacedo package") from error
            sampler = TopacedoSampler(
                graph_matrix,
                clusters.values,
                dendrogram,
                density_depth,
                density_bandwidth,
                max_sampling_rate,
                min_sampling_rate,
                min_cells_per_group,
                snn_bandwidth,
                seed_reward,
                non_seed_reward,
                edge_cost_multiplier,
                edge_cost_bandwidth,
                rand_state,
            )
            nodes, edges = sampler.run()
            raw_node_indices = np.asarray(nodes)
            if raw_node_indices.dtype.kind not in {"i", "u"}:
                raise ValueError("TopACeDo returned non-integer sampled-cell indices")
            node_indices = raw_node_indices.astype(np.int64, copy=False)
            if node_indices.ndim != 1 or np.any(
                (node_indices < 0) | (node_indices >= graph_matrix.shape[0])
            ):
                raise ValueError("TopACeDo returned invalid sampled-cell indices")
            sampled = np.zeros(graph_matrix.shape[0], dtype=bool)
            sampled[node_indices] = True
            density = np.asarray(sampler.densities, dtype=np.float64)
            mean_snn = np.asarray(sampler.meanSnn, dtype=np.float64)
            if density.shape != (graph_matrix.shape[0],):
                raise ValueError("TopACeDo returned invalid cell-density values")
            if mean_snn.shape != (graph_matrix.shape[0],):
                raise ValueError("TopACeDo returned invalid mean-SNN values")
            raw_seed_indices = np.asarray(sampler.seeds)
            if raw_seed_indices.dtype.kind not in {"i", "u"}:
                raise ValueError("TopACeDo returned non-integer seed-cell indices")
            seed_indices = raw_seed_indices.astype(np.int64, copy=False)
            if seed_indices.ndim != 1 or np.any(
                (seed_indices < 0) | (seed_indices >= graph_matrix.shape[0])
            ):
                raise ValueError("TopACeDo returned invalid seed-cell indices")
            seeds = np.zeros(graph_matrix.shape[0], dtype=bool)
            seeds[seed_indices] = True
            raw_edge_values = np.asarray(edges)
            if raw_edge_values.size and raw_edge_values.dtype.kind not in {"i", "u"}:
                raise ValueError("TopACeDo returned non-integer edge pairs")
            edge_values = raw_edge_values.astype(np.int64, copy=False)
            if edge_values.size == 0:
                edge_values = edge_values.reshape(0, 2)
            elif edge_values.ndim != 2 or edge_values.shape[1] != 2:
                raise ValueError("TopACeDo returned invalid edge pairs")
            if np.any((edge_values < 0) | (edge_values >= graph_matrix.shape[0])):
                raise ValueError("TopACeDo returned out-of-range edge endpoints")
            write_cell_data_artifact(
                self.zw,
                planned,
                {
                    "sampled": sampled,
                    "density": density,
                    "mean_snn": mean_snn,
                    "seeds": seeds,
                    "edges": edge_values,
                },
            )
        values = {
            "sampled": sampled,
            "density": density,
            "mean_snn": mean_snn,
            "seeds": seeds,
        }
        defaults = {
            "sampled": categorical_display(sampled),
            "density": continuous_display(density),
            "mean_snn": continuous_display(mean_snn),
            "seeds": categorical_display(seeds),
        }
        for name, column in columns.items():
            self.cells.insert(
                column,
                values[name],
                fill_value=False if name in {"sampled", "seeds"} else np.nan,
                key=cell_key,
                overwrite=True,
            )
        for name, column in columns.items():
            link_cell_data_column(
                self.zw,
                column,
                planned.ref,
                value_name=name,
                default_display=defaults[name],
                preserved_display=preserved_displays[name],
            )
        logger.debug(f"Sketched cells saved under column '{columns['sampled']}'")
        logger.debug(
            f"Cell neighbourhood densities saved under column: '{columns['density']}'"
        )
        logger.debug(f"Mean SNN values saved under column: '{columns['mean_snn']}'")
        logger.debug(f"Seed cells saved under column: '{columns['seeds']}'")
        return planned.ref
