from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np

from ...graph.feature_projection import graph_cell_selection
from ...graph.kinds import require_graph_kind
from ...metadata.artifacts import (
    artifact_values,
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
    AttributeRequirement,
    PlannedArtifact,
    finish_artifact,
    plan_artifact,
    reused_artifact_group,
    start_artifact,
)
from ...storage.arrays import create_zarr_dataset
from ...storage.types import as_zarr_array, as_zarr_group
from ...storage.errors import ArtifactResolutionError
from ...utils.logging import logger
from ...utils.shutdown import shutdown_checkpoint

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
    resolution: float
    backend: Literal["igraph", "leidenalg"]
    symmetric_graph: bool
    graph_upper_only: bool
    random_seed: int
    n_cells: int

    @property
    def graph_key(self) -> tuple[str, bool, bool]:
        return (
            self.graph_loc,
            self.symmetric_graph,
            self.graph_upper_only,
        )


class _ClusteringOperationsMixin(_ClusteringOperationsBase):
    def _clustering_graph(self, graph: ArtifactRef) -> tuple[str, int, int]:
        """Return a complete clustering graph's location, cell count, and k."""
        if not isinstance(graph, ArtifactRef):
            raise TypeError("graph must be an ArtifactRef")
        require_graph_kind(graph)
        status = inspect_artifact(self.zw, graph)
        if not status.complete:
            raise ValueError("Graph artifact is unavailable or incomplete")
        n_cells, k = self._get_graph_ncells_k(status.path)
        return status.path, n_cells, k

    def _run_paris_from_artifacts(
        self,
        *,
        graph_ref: ArtifactRef,
        graph_loc: str,
        fixed_cluster_count: int | None,
        effective_min_cluster_size: int | None,
        invalidate_cache: bool,
    ) -> "ParisClusteringResult":
        from ...clustering._paris_core import ParisHierarchy
        from ...clustering._paris_modularity import modularity_split_gains
        from ...clustering.paris import (
            fit_paris_hierarchy,
            fixed_cut,
            hierarchy_to_dendrogram,
        )
        from ...clustering.paris_multiscale import (
            ParisClusteringResult,
            PlateauForest,
            adaptive_cut,
            collapse_equal_height_plateaus,
        )
        from .paris_persistence import (
            hierarchy_array_requirements,
            hierarchy_attribute_requirements,
            load_hierarchy_group,
            plan_paris_dendrogram,
            preflight_hierarchy_artifact_cut,
            preflight_paris_adaptive_cut,
            preflight_paris_fit,
            read_paris_cut_diagnostics,
            write_hierarchy_group,
            write_paris_dendrogram,
        )

        artifact_scope = graph_ref.scope
        artifact_assay = graph_ref.assay
        cell_selection = graph_cell_selection(self.zw, graph_ref)
        n_cells, _effective_k = self._get_graph_ncells_k(graph_loc)
        cut_mode: Literal["adaptive", "fixed"] = (
            "fixed" if fixed_cluster_count is not None else "adaptive"
        )
        mode: Literal["auto", "fixed"] = (
            "fixed" if fixed_cluster_count is not None else "auto"
        )
        graph_group = as_zarr_group(self.zw[graph_loc], name=graph_loc)
        budget = self.resources
        # Structurally incomplete hierarchies are skipped here and refitted.
        hierarchy_plan = plan_artifact(
            self.zw,
            scope=artifact_scope,
            assay=artifact_assay,
            kind="cluster_hierarchy",
            operation="fit_paris_hierarchy",
            parameters={},
            inputs={"connectivity_map": graph_ref},
            execution_options={"invalidate_cache": invalidate_cache},
            invalidate_cache=invalidate_cache,
            required_arrays=hierarchy_array_requirements(),
            required_attributes=hierarchy_attribute_requirements(n_cells),
        )
        loaded: tuple[ParisHierarchy, PlateauForest] | None = None
        fitted_graph = None

        def hierarchy_payload() -> tuple[ParisHierarchy, PlateauForest]:
            # Load or fit the hierarchy only when a cut or dendrogram needs it.
            nonlocal loaded, fitted_graph
            if loaded is not None:
                return loaded
            if hierarchy_plan.reused:
                hierarchy_group = reused_artifact_group(self.zw, hierarchy_plan)
                preflight_hierarchy_artifact_cut(
                    hierarchy_group,
                    cut_mode,
                    budget,
                )
                try:
                    loaded = load_hierarchy_group(
                        hierarchy_group,
                        hierarchy_plan.ref.artifact_id,
                    )
                except (KeyError, TypeError, ValueError) as error:
                    raise ArtifactResolutionError(
                        f"Paris hierarchy artifact {hierarchy_plan.ref.artifact_id} "
                        "is unreadable. Rerun with invalidate_cache=True to "
                        "recompute it.",
                        code="corrupt_payload",
                        context={"artifact_id": hierarchy_plan.ref.artifact_id},
                    ) from error
            else:
                estimated_peak_bytes = preflight_paris_fit(
                    graph_group,
                    n_cells,
                    budget,
                )
                fitted_graph = self._load_graph_artifact(
                    graph_ref,
                    symmetric=False,
                    upper_only=False,
                    use_k=None,
                )
                shutdown_checkpoint()
                hierarchy = fit_paris_hierarchy(
                    fitted_graph,
                    nthreads=budget.workers,
                )
                shutdown_checkpoint()
                plateau_forest = collapse_equal_height_plateaus(hierarchy)
                hierarchy_group = start_artifact(self.zw, hierarchy_plan)
                write_hierarchy_group(hierarchy_group, hierarchy, plateau_forest)
                hierarchy_group.attrs["estimated_peak_bytes"] = estimated_peak_bytes
                finish_artifact(hierarchy_group, hierarchy_plan)
                loaded = hierarchy, plateau_forest
            if loaded[0].n_leaves != n_cells:
                raise ValueError("Paris hierarchy size does not match graph")
            return loaded

        cut_parameters = (
            {"mode": mode, "n_clusters": fixed_cluster_count}
            if fixed_cluster_count is not None
            else {
                "mode": mode,
                "min_cluster_size": effective_min_cluster_size,
            }
        )
        cut_inputs = {
            "cluster_hierarchy": hierarchy_plan.ref,
            "connectivity_map": graph_ref,
            "cell_selection": cell_selection,
        }

        def valid_cluster_count(value: object) -> bool:
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                return False
            return fixed_cluster_count is None or value == fixed_cluster_count

        def readable_cut(_ref: ArtifactRef, group: Any) -> bool:
            try:
                read_paris_cut_diagnostics(group, mode)
            except (TypeError, ValueError):
                return False
            return True

        # A new hierarchy has a fresh identity, so its cut is never reused.
        cut_plan = plan_artifact(
            self.zw,
            scope=artifact_scope,
            assay=artifact_assay,
            kind="cluster_cut",
            operation="cut_paris_hierarchy",
            parameters=cut_parameters,
            inputs=cut_inputs,
            execution_options={"invalidate_cache": invalidate_cache},
            invalidate_cache=invalidate_cache,
            required_arrays=(
                ArrayRequirement("labels", shape=(n_cells,), dtype_kind="i"),
            ),
            required_attributes=(
                AttributeRequirement("n_clusters", predicate=valid_cluster_count),
            ),
            reuse_validator=readable_cut,
        )
        if cut_plan.reused:
            cut_group = reused_artifact_group(self.zw, cut_plan)
            result = ParisClusteringResult(
                labels=np.asarray(
                    as_zarr_array(cut_group["labels"], name="labels")[:],
                    dtype=np.int32,
                ),
                mode=mode,
                n_clusters=int(cast(int, cut_group.attrs["n_clusters"])),
                diagnostics=read_paris_cut_diagnostics(cut_group, mode),
                min_cluster_size=effective_min_cluster_size,
            )
        else:
            hierarchy, plateau_forest = hierarchy_payload()
            if fixed_cluster_count is None:
                assert effective_min_cluster_size is not None
                if fitted_graph is None:
                    preflight_paris_adaptive_cut(
                        graph_group,
                        n_cells,
                        budget,
                    )
                    fitted_graph = self._load_graph_artifact(
                        graph_ref,
                        symmetric=None,
                        upper_only=None,
                        use_k=None,
                    )
                split_gate = modularity_split_gains(
                    hierarchy,
                    plateau_forest,
                    fitted_graph,
                )
                shutdown_checkpoint()
                result = adaptive_cut(
                    hierarchy,
                    effective_min_cluster_size,
                    plateau_forest=plateau_forest,
                    split_gate=split_gate,
                )
                shutdown_checkpoint()
            else:
                n_components = len(hierarchy.component_roots)
                if 1 < fixed_cluster_count < n_components:
                    raise ValueError(
                        f"The graph has {n_components} connected components, so a "
                        f"fixed Paris cut cannot produce {fixed_cluster_count} "
                        f"clusters. Request n_clusters=1, at least "
                        f"{n_components} clusters, or n_clusters='auto'."
                    )
                labels = fixed_cut(hierarchy, fixed_cluster_count).astype(
                    np.int32,
                    copy=False,
                )
                shutdown_checkpoint()
                result = ParisClusteringResult(
                    labels=labels,
                    mode="fixed",
                    n_clusters=fixed_cluster_count,
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
            dendrogram_plan = plan_paris_dendrogram(self.zw, hierarchy_plan.ref)
            if not dendrogram_plan.reused:
                write_paris_dendrogram(
                    self.zw,
                    dendrogram_plan,
                    hierarchy_to_dendrogram(
                        hierarchy_payload()[0],
                        compatibility=True,
                    ),
                )

        action = "Reused" if cut_plan.reused else "Stored"
        logger.info(f"{action} Paris clustering with {result.n_clusters} clusters")
        return replace(
            result,
            hierarchy_artifact_id=hierarchy_plan.ref.artifact_id,
            ref=cut_plan.ref,
        )

    def _prepare_leiden_clustering(
        self,
        graph: ArtifactRef,
        *,
        resolution: float = 1.0,
        backend: Literal["igraph", "leidenalg"] = "igraph",
        symmetric_graph: bool = False,
        graph_upper_only: bool = False,
        random_seed: int = 4444,
        invalidate_cache: bool = False,
    ) -> _PreparedLeidenClustering:
        from ...clustering.leiden import canonical_random_seed, canonical_resolution

        if backend not in {"igraph", "leidenalg"}:
            raise ValueError("backend must be 'igraph' or 'leidenalg'")
        resolution = canonical_resolution(resolution)
        random_seed = canonical_random_seed(random_seed)
        graph_loc, n_cells, _k = self._clustering_graph(graph)
        graph_input = graph
        artifact_scope = graph_input.scope
        selection = graph_cell_selection(self.zw, graph_input)
        arguments = LeidenArguments(
            graph=graph_input,
            resolution=resolution,
            backend=backend,
            edge_weighting="graph",
            symmetric_graph=symmetric_graph,
            graph_upper_only=graph_upper_only,
            random_seed=random_seed,
            invalidate_cache=invalidate_cache,
        )
        record = arguments.to_record()
        planned = plan_cell_data_artifact(
            self.zw,
            scope=artifact_scope,
            assay=(graph_input.assay if graph_input.scope == "assay" else None),
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
            graph_loc=graph_loc,
            resolution=resolution,
            backend=backend,
            symmetric_graph=symmetric_graph,
            graph_upper_only=graph_upper_only,
            random_seed=random_seed,
            n_cells=n_cells,
        )

    def _load_prepared_leiden_graph(
        self,
        prepared: _PreparedLeidenClustering,
    ) -> Any:
        graph = self._load_graph_artifact(
            prepared.graph,
            symmetric=prepared.symmetric_graph,
            upper_only=prepared.graph_upper_only,
            use_k=None,
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
        shutdown_checkpoint()
        membership = leiden_membership(
            graph,
            prepared.resolution,
            prepared.random_seed,
            backend=prepared.backend,
        )
        shutdown_checkpoint()
        return membership

    def _finish_prepared_leiden(
        self,
        prepared: _PreparedLeidenClustering,
        membership: np.ndarray | None,
    ) -> tuple[np.ndarray, ArtifactRef]:
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
        action = "Reused" if prepared.planned.reused else "Stored"
        logger.info(
            f"{action} Leiden clustering with {np.unique(membership).size} clusters"
        )
        return membership, prepared.planned.ref

    def _run_leiden_artifact(
        self,
        graph: ArtifactRef,
        *,
        resolution: float = 1.0,
        backend: Literal["igraph", "leidenalg"] = "igraph",
        symmetric_graph: bool = False,
        graph_upper_only: bool = False,
        random_seed: int = 4444,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Execute Leiden clustering and return its immutable artifact.

        Args:
            graph: Explicit connectivity map or integrated graph to partition.
            resolution: Finite positive Leiden resolution, recorded as a float.
            backend: Leiden implementation. Native igraph is the default.
            symmetric_graph: Forwarded to `load_graph`.
            graph_upper_only: Forwarded to `load_graph`.
            random_seed: Non-negative integer seed for the Leiden optimizer.
            invalidate_cache: Force a new cluster-labels artifact.

        Returns:
            Reference to the cluster-labels artifact.
        """
        prepared = self._prepare_leiden_clustering(
            graph,
            resolution=resolution,
            backend=backend,
            symmetric_graph=symmetric_graph,
            graph_upper_only=graph_upper_only,
            random_seed=random_seed,
            invalidate_cache=invalidate_cache,
        )
        membership = None
        if not prepared.planned.reused:
            graph_matrix = self._load_prepared_leiden_graph(prepared)
            membership = self._compute_prepared_leiden(prepared, graph_matrix)
        _membership, ref = self._finish_prepared_leiden(prepared, membership)
        return ref

    def run_leiden_clustering(
        self,
        graph: ArtifactRef,
        *,
        resolution: float = 1.0,
        backend: Literal["igraph", "leidenalg"] = "igraph",
        symmetric_graph: bool = False,
        graph_upper_only: bool = False,
        random_seed: int = 4444,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Build and return immutable Leiden cluster labels.

        ``resolution`` must be a finite positive number and is recorded as a
        float, so ``1`` and ``1.0`` identify the same artifact. ``random_seed``
        must be a non-negative integer because stored labels are reused.
        """
        return self._run_leiden_artifact(
            graph,
            resolution=resolution,
            backend=backend,
            symmetric_graph=symmetric_graph,
            graph_upper_only=graph_upper_only,
            random_seed=random_seed,
            invalidate_cache=invalidate_cache,
        )

    def _run_paris_artifact(
        self,
        graph: ArtifactRef,
        *,
        n_clusters: int | Literal["auto"] = "auto",
        min_cluster_size: int | None = None,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Fit the canonical Paris hierarchy and write a fixed or adaptive cut.

        Pass ``graph`` to partition an explicit connectivity map or integrated
        graph. The returned reference identifies the immutable cut artifact;
        ``load_paris_clustering`` reconstructs labels and diagnostics.
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
        graph_loc, n_cells, effective_k = self._clustering_graph(graph)
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

        result = self._run_paris_from_artifacts(
            graph_ref=graph,
            graph_loc=graph_loc,
            fixed_cluster_count=fixed_cluster_count,
            effective_min_cluster_size=effective_min_cluster_size,
            invalidate_cache=invalidate_cache,
        )
        if result.ref is None:
            raise RuntimeError("Paris clustering did not produce an artifact")
        return result.ref

    def _load_paris_artifact_result(
        self,
        ref: ArtifactRef,
    ) -> "ParisClusteringResult":
        from ...clustering.paris_multiscale import ParisClusteringResult
        from .paris_persistence import read_paris_cut_diagnostics

        status = inspect_artifact(self.zw, ref)
        if not status.complete or status.operation != "cut_paris_hierarchy":
            raise ValueError("Paris cut artifact is unavailable or invalid")
        group = as_zarr_group(self.zw[status.path], name=status.path)
        labels = np.asarray(
            as_zarr_array(group["labels"], name="labels")[:],
            dtype=np.int32,
        )
        parameters = status.parameters or {}
        mode = parameters.get("mode")
        if mode not in {"auto", "fixed"}:
            raise ValueError("Paris cut mode is invalid")
        try:
            diagnostics = read_paris_cut_diagnostics(
                group,
                cast(Literal["auto", "fixed"], mode),
            )
        except (TypeError, ValueError) as error:
            raise ArtifactResolutionError(
                f"Paris cut artifact {ref.artifact_id} does not match the current "
                "diagnostics schema. Recompute it with run_paris_clustering("
                "invalidate_cache=True).",
                code="corrupt_payload",
                context={"artifact_id": ref.artifact_id},
            ) from error
        raw_hierarchy = (status.inputs or {}).get("cluster_hierarchy")
        hierarchy_id = (
            ArtifactRef.from_dict(raw_hierarchy).artifact_id
            if isinstance(raw_hierarchy, dict)
            else None
        )
        return ParisClusteringResult(
            labels=labels,
            mode=cast(Literal["auto", "fixed"], mode),
            n_clusters=int(cast(int | float | str, group.attrs["n_clusters"])),
            diagnostics=diagnostics,
            min_cluster_size=(
                int(parameters["min_cluster_size"])
                if mode == "auto" and parameters.get("min_cluster_size") is not None
                else None
            ),
            hierarchy_artifact_id=hierarchy_id,
            ref=ref,
        )

    def load_paris_clustering(
        self,
        ref: ArtifactRef,
    ) -> "ParisClusteringResult":
        """Load a Paris result from an explicit completed cut artifact."""
        if not isinstance(ref, ArtifactRef):
            raise TypeError("ref must be an ArtifactRef")
        if ref.kind != "cluster_cut":
            raise ValueError("ref must be a cluster_cut artifact")
        return self._load_paris_artifact_result(ref)

    def run_paris_clustering(
        self,
        graph: ArtifactRef,
        *,
        n_clusters: int | Literal["auto"] = "auto",
        min_cluster_size: int | None = None,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Build and return an immutable Paris cut artifact.

        A fixed integer ``n_clusters`` must be 1 or at least the number of
        connected components in ``graph``, and the hierarchy must split into
        exactly that many clusters at one height. Otherwise a ``ValueError``
        names the alternatives, such as ``n_clusters='auto'``.
        """
        return self._run_paris_artifact(
            graph,
            n_clusters=n_clusters,
            min_cluster_size=min_cluster_size,
            invalidate_cache=invalidate_cache,
        )

    def run_topacedo_sampler(
        self,
        graph: ArtifactRef,
        clusters: ArtifactRef,
        *,
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
        rand_state: int = 4466,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Perform sub-sampling (aka sketching) of cells using TopACeDo
        algorithm. Sub-sampling required that cells are partitioned in cluster
        already. Since, sub-sampling is dependent on cluster information,
        having, large number of homogeneous and even sized cluster improves
        sub-sampling results.

        Args:
            graph: Explicit connectivity map or integrated graph to sample.
            clusters: Explicit Paris ``cluster_cut`` artifact for this graph.
            use_k: Number of top k-nearest neighbours to retain in the graph over which downsampling is performed.
                   Must be an integer from 2 to the graph's k. By default all neighbours are used. (Default value: None)
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
            rand_state: A random values to set seed while sampling cells from a cluster randomly. (Default value: 4466)

        Returns:
            Reference to the sampling artifact. Open it with ``load_artifact``
            to read the ``edges`` array of Steiner tree edges over graph row
            indices.
        """
        from .paris_persistence import (
            load_hierarchy_group,
            plan_paris_dendrogram,
            write_paris_dendrogram,
        )

        _graph_loc, n_cells, graph_k = self._clustering_graph(graph)
        if not isinstance(clusters, ArtifactRef):
            raise TypeError("clusters must be an ArtifactRef")
        if clusters.kind != "cluster_cut":
            raise ValueError("clusters must be a Paris cluster_cut artifact")
        graph_input = graph
        # Validate before any plan or write; out-of-range values fail or alias.
        if use_k is not None:
            if isinstance(use_k, bool | np.bool_) or not isinstance(
                use_k,
                int | np.integer,
            ):
                raise TypeError("use_k must be an integer or None")
            if not 2 <= use_k <= graph_k:
                raise ValueError(
                    f"use_k must be between 2 and the graph's k ({graph_k}), "
                    "or None to use every neighbour"
                )
            # Every neighbour is the default, so use_k=k shares its identity.
            use_k = None if use_k == graph_k else int(use_k)
        selection = graph_cell_selection(self.zw, graph_input)
        cut_status = inspect_artifact(self.zw, clusters)
        if not cut_status.complete or cut_status.operation != "cut_paris_hierarchy":
            raise ArtifactResolutionError(
                "TopACeDo requires a complete Paris cluster_cut artifact",
                code="corrupt_payload",
                context={"artifact_id": clusters.artifact_id},
            )
        cluster_input = clusters
        cut_inputs = cut_status.inputs or {}
        raw_hierarchy_ref = cut_inputs.get("cluster_hierarchy")
        if cut_inputs.get("connectivity_map") != graph_input.to_dict():
            raise ValueError("Cluster cut does not belong to the requested graph")
        if cut_inputs.get("cell_selection") != selection.to_dict():
            raise ValueError("Cluster cut does not match the graph cell selection")
        if not isinstance(raw_hierarchy_ref, dict):
            raise ArtifactResolutionError(
                "TopACeDo cluster cut does not name its Paris hierarchy",
                code="corrupt_payload",
                context={"artifact_id": clusters.artifact_id},
            )
        cut_group = as_zarr_group(
            self.zw[cut_status.path],
            name=clusters.artifact_id,
        )
        cluster_labels = as_zarr_array(cut_group["labels"], name="labels")
        if tuple(cluster_labels.shape) != (n_cells,):
            raise ValueError(
                f"Cluster labels contain {cluster_labels.shape[0]} cells while "
                f"graph has {n_cells} cells."
            )
        hierarchy_ref = ArtifactRef.from_dict(raw_hierarchy_ref)
        dendrogram_plan = plan_paris_dendrogram(self.zw, hierarchy_ref)
        artifact_scope = graph_input.scope
        arguments = TopacedoArguments(
            graph=graph_input,
            clusters=cluster_input,
            dendrogram=dendrogram_plan.ref,
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
            invalidate_cache=invalidate_cache,
        )
        planned = arguments.plan(
            self.zw,
            scope=artifact_scope,
            assay=(graph_input.assay if graph_input.scope == "assay" else None),
            invalidate_cache=invalidate_cache,
            required_arrays=(
                ArrayRequirement("sampled", shape=(n_cells,), dtype_kind="b"),
                ArrayRequirement("density", shape=(n_cells,), dtype_kind="f"),
                ArrayRequirement("mean_snn", shape=(n_cells,), dtype_kind="f"),
                ArrayRequirement("seeds", shape=(n_cells,), dtype_kind="b"),
                ArrayRequirement("edges", shape=(None, 2), dtype_kind="i"),
            ),
        )
        if planned.reused:
            logger.info("Reused TopACeDo sampling artifact")
            return planned.ref
        try:
            from topacedo import TopacedoSampler
        except ImportError as error:
            raise ImportError("Could not find topacedo package") from error

        if dendrogram_plan.reused:
            dendrogram_group = reused_artifact_group(self.zw, dendrogram_plan)
            dendrogram = np.asarray(
                as_zarr_array(dendrogram_group["data"], name="data")[:]
            )
        else:
            from ...clustering.paris import hierarchy_to_dendrogram

            hierarchy_status = inspect_artifact(self.zw, hierarchy_ref)
            if (
                not hierarchy_status.complete
                or hierarchy_status.operation != "fit_paris_hierarchy"
            ):
                raise ArtifactResolutionError(
                    "TopACeDo requires the complete Paris hierarchy named by "
                    "the cluster cut",
                    code="corrupt_payload",
                    context={"artifact_id": hierarchy_ref.artifact_id},
                )
            hierarchy_group = as_zarr_group(
                self.zw[hierarchy_status.path],
                name=hierarchy_ref.artifact_id,
            )
            hierarchy, _plateau = load_hierarchy_group(
                hierarchy_group,
                hierarchy_ref.artifact_id,
            )
            dendrogram = hierarchy_to_dendrogram(hierarchy, compatibility=True)
            write_paris_dendrogram(self.zw, dendrogram_plan, dendrogram)
        cluster_values = np.asarray(cluster_labels[:])
        graph_matrix = self._load_graph_artifact(
            graph_input,
            symmetric=False,
            upper_only=False,
            use_k=use_k,
        ).copy()
        graph_matrix.eliminate_zeros()

        sampler = TopacedoSampler(
            graph_matrix,
            cluster_values,
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
            (node_indices < 0) | (node_indices >= n_cells)
        ):
            raise ValueError("TopACeDo returned invalid sampled-cell indices")
        sampled = np.zeros(n_cells, dtype=bool)
        sampled[node_indices] = True
        density = np.asarray(sampler.densities, dtype=np.float64)
        mean_snn = np.asarray(sampler.meanSnn, dtype=np.float64)
        if density.shape != (n_cells,):
            raise ValueError("TopACeDo returned invalid cell-density values")
        if mean_snn.shape != (n_cells,):
            raise ValueError("TopACeDo returned invalid mean-SNN values")
        raw_seed_indices = np.asarray(sampler.seeds)
        if raw_seed_indices.dtype.kind not in {"i", "u"}:
            raise ValueError("TopACeDo returned non-integer seed-cell indices")
        seed_indices = raw_seed_indices.astype(np.int64, copy=False)
        if seed_indices.ndim != 1 or np.any(
            (seed_indices < 0) | (seed_indices >= n_cells)
        ):
            raise ValueError("TopACeDo returned invalid seed-cell indices")
        seeds = np.zeros(n_cells, dtype=bool)
        seeds[seed_indices] = True
        raw_edge_values = np.asarray(edges)
        if raw_edge_values.size and raw_edge_values.dtype.kind not in {"i", "u"}:
            raise ValueError("TopACeDo returned non-integer edge pairs")
        edge_values = raw_edge_values.astype(np.int64, copy=False)
        if edge_values.size == 0:
            edge_values = edge_values.reshape(0, 2)
        elif edge_values.ndim != 2 or edge_values.shape[1] != 2:
            raise ValueError("TopACeDo returned invalid edge pairs")
        if np.any((edge_values < 0) | (edge_values >= n_cells)):
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
        logger.info(f"Stored TopACeDo sampling artifact for {int(sampled.sum())} cells")
        return planned.ref
