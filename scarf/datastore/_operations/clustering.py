from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import pandas as pd

from ...storage.types import as_zarr_array, as_zarr_group
from ...utils.logging import logger

if TYPE_CHECKING:
    from scipy.sparse import csr_matrix

    from ...clustering._paris_core import ParisHierarchy
    from ...clustering.paris_multiscale import ParisClusteringResult, PlateauForest
    from .graph import _GraphOperationsMixin as _ClusteringOperationsBase
else:
    _ClusteringOperationsBase = object


class _ClusteringOperationsMixin(_ClusteringOperationsBase):
    def _resolve_paris_hierarchy(
        self,
        *,
        graph_loc: str,
        from_assay: str,
        cell_key: str,
        feat_key: str,
        force_recalc: bool,
        cut_mode: Literal["adaptive", "fixed"],
    ) -> tuple[
        str,
        "ParisHierarchy",
        "PlateauForest",
        bool,
        int | None,
        "csr_matrix | None",
    ]:
        import warnings

        from ...clustering.paris_multiscale import collapse_equal_height_plateaus
        from ...clustering.paris import fit_paris_hierarchy
        from ...storage.budget import get_resource_budget
        from .paris_persistence import (
            LATEST_PARIS_GENERATION,
            generation_location,
            load_hierarchy_generation,
            preflight_cached_paris_cut,
            preflight_paris_fit,
            write_hierarchy_generation,
        )

        graph_group = as_zarr_group(self.zw[graph_loc], name=graph_loc)
        generation_value = graph_group.attrs.get(LATEST_PARIS_GENERATION)
        budget = get_resource_budget()
        if generation_value is not None and not force_recalc:
            generation_id = str(generation_value)
            preflight_cached_paris_cut(
                self.zw,
                graph_loc,
                generation_id,
                cut_mode,
                budget,
            )
            hierarchy, plateau_forest = load_hierarchy_generation(
                self.zw,
                graph_loc,
                generation_id,
            )
            logger.info(
                f"Using Paris hierarchy generation "
                f"{generation_location(graph_loc, generation_id)}"
            )
            return generation_id, hierarchy, plateau_forest, True, None, None

        legacy_dendrogram = f"{graph_loc}/dendrogram"
        if generation_value is None and (
            legacy_dendrogram in self.zw or "latest_dendrogram" in graph_group.attrs
        ):
            warnings.warn(
                "The cached Paris hierarchy predates canonical additive graphs and "
                "will be rebuilt. Existing Paris cluster labels can change even "
                "when the requested integer cluster count is unchanged.",
                UserWarning,
                stacklevel=3,
            )

        n_cells, _effective_k = self._get_graph_ncells_k(graph_loc)
        estimated_peak_bytes = preflight_paris_fit(
            graph_group,
            n_cells,
            budget,
        )
        graph = self.load_graph(
            from_assay=from_assay,
            cell_key=cell_key,
            feat_key=feat_key,
            symmetric=False,
            upper_only=False,
            graph_loc=graph_loc,
        )
        hierarchy = fit_paris_hierarchy(graph, n_threads=budget.workers)
        plateau_forest = collapse_equal_height_plateaus(hierarchy)
        generation_id, location = write_hierarchy_generation(
            self.zw,
            graph_loc,
            hierarchy,
            plateau_forest,
        )
        generation_group = as_zarr_group(self.zw[location], name=location)
        generation_group.attrs["estimated_peak_bytes"] = estimated_peak_bytes
        graph_group.attrs[LATEST_PARIS_GENERATION] = generation_id
        if "latest_dendrogram" in graph_group.attrs:
            del graph_group.attrs["latest_dendrogram"]
        return (
            generation_id,
            hierarchy,
            plateau_forest,
            False,
            estimated_peak_bytes,
            graph,
        )

    def run_leiden_clustering(
        self,
        from_assay: str | None = None,
        cell_key: str | None = None,
        feat_key: str | None = None,
        resolution: float = 1.0,
        integrated_graph: str | None = None,
        symmetric_graph: bool = False,
        graph_upper_only: bool = False,
        label: str = "leiden_cluster",
        random_seed: int = 4444,
    ) -> None:
        """Executes Leiden graph clustering algorithm on the cell-neighbourhood
        graph and saves cluster identities in the cell metadata column.

        Args:
            from_assay: Name of assay to be used. If no value is provided then the default assay will be used.
            cell_key: Cell key. Should be same as the one that was used in the desired graph. (Default value: 'I')
            feat_key:  Feature key. Should be same as the one that was used in the desired graph. By default, the latest
                       used feature for the given assay will be used.
            resolution: Resolution parameter for `RBConfigurationVertexPartition` configuration
            integrated_graph:
            symmetric_graph: This parameter is forwarded to `load_graph` and is same as there. (Default value: True)
            graph_upper_only: This parameter is forwarded to `load_graph` and is same as there. (Default value: True)
            label: base label for cluster identity in the cell metadata column (Default value: 'leiden_cluster')
            random_seed: (Default value: 4444)

        Returns:
        """
        from ...clustering.leiden import leiden_membership

        from_assay, cell_key, feat_key = self._get_latest_keys(
            from_assay, cell_key, feat_key
        )
        graph_loc = None
        if integrated_graph is not None:
            graph_loc = f"{self._integratedGraphsLoc}/{integrated_graph}"
            if graph_loc not in self.zw:
                raise KeyError(
                    f"ERROR: An integrated graph with label: {integrated_graph} does not exist"
                )
        graph = self.load_graph(
            from_assay=from_assay,
            cell_key=cell_key,
            feat_key=feat_key,
            symmetric=symmetric_graph,
            upper_only=graph_upper_only,
            graph_loc=graph_loc,
        )
        if integrated_graph is not None:
            from_assay = integrated_graph
        membership = leiden_membership(graph, resolution, random_seed)
        self.cells.insert(
            self._col_renamer(from_assay, cell_key, label),
            membership,
            fill_value=-1,
            key=cell_key,
            overwrite=True,
        )
        return None

    def run_paris_clustering(
        self,
        from_assay: str | None = None,
        cell_key: str | None = None,
        feat_key: str | None = None,
        n_clusters: int | Literal["auto"] = "auto",
        integrated_graph: str | None = None,
        min_cluster_size: int | None = None,
        force_recalc: bool = False,
        label: str = "paris_cluster",
    ) -> "ParisClusteringResult":
        """Fit the canonical Paris hierarchy and write a fixed or adaptive cut."""
        from dataclasses import replace
        from time import perf_counter

        from ...clustering._paris_modularity import modularity_split_gains
        from ...clustering.paris_multiscale import (
            ParisClusteringResult,
            adaptive_cut,
        )
        from ...clustering.paris import hierarchy_to_dendrogram, straight_cut
        from ...storage.budget import get_resource_budget
        from .paris_persistence import (
            activate_adaptive_result,
            adaptive_config_digest,
            clear_active_adaptive_result,
            ensure_compatibility_dendrogram,
            garbage_collect_hierarchy_generations,
            load_adaptive_result,
            preflight_paris_adaptive_cut,
            persist_adaptive_result,
        )

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

        if integrated_graph is not None:
            graph_loc = f"{self._integratedGraphsLoc}/{integrated_graph}"
            if graph_loc not in self.zw:
                raise KeyError(
                    f"An integrated graph with label {integrated_graph!r} does not exist"
                )
        else:
            graph_loc = None

        from_assay, cell_key, feat_key = self._get_latest_keys(
            from_assay,
            cell_key,
            feat_key,
        )
        if graph_loc is None:
            graph_loc = self._get_latest_graph_loc(
                from_assay,
                cell_key,
                feat_key,
            )

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

        (
            generation_id,
            hierarchy,
            plateau_forest,
            hierarchy_cache_hit,
            _estimated_peak_bytes,
            fitted_graph,
        ) = self._resolve_paris_hierarchy(
            graph_loc=graph_loc,
            from_assay=from_assay,
            cell_key=cell_key,
            feat_key=feat_key,
            force_recalc=force_recalc,
            cut_mode="adaptive" if fixed_cluster_count is None else "fixed",
        )
        if hierarchy.n_leaves != n_cells:
            raise ValueError("Cached Paris hierarchy size does not match the graph")

        label_assay = integrated_graph if integrated_graph is not None else from_assay
        final_label_key = self._col_renamer(label_assay, cell_key, label)
        adaptive_digest: str | None = None
        if fixed_cluster_count is None:
            assert effective_min_cluster_size is not None
            adaptive_digest = adaptive_config_digest(
                generation_id,
                effective_min_cluster_size,
            )
            cut_start = perf_counter()
            result = load_adaptive_result(
                self.zw,
                graph_loc,
                label,
                adaptive_digest,
                hierarchy,
            )
            if result is None:
                if fitted_graph is None:
                    graph_group = as_zarr_group(self.zw[graph_loc], name=graph_loc)
                    preflight_paris_adaptive_cut(
                        graph_group,
                        n_cells,
                        get_resource_budget(),
                    )
                    guard_graph = self.load_graph(
                        from_assay=from_assay,
                        cell_key=cell_key,
                        feat_key=feat_key,
                        symmetric=False,
                        upper_only=False,
                        graph_loc=graph_loc,
                    )
                else:
                    guard_graph = fitted_graph
                split_gate = modularity_split_gains(
                    hierarchy,
                    plateau_forest,
                    guard_graph,
                )
                result = adaptive_cut(
                    hierarchy,
                    effective_min_cluster_size,
                    plateau_forest=plateau_forest,
                    split_gate=split_gate,
                )
                cut_seconds = perf_counter() - cut_start
                persist_adaptive_result(
                    self.zw,
                    graph_loc,
                    label,
                    adaptive_digest,
                    result,
                    generation_id=generation_id,
                    final_label_key=final_label_key,
                    hierarchy_cache_hit=hierarchy_cache_hit,
                    cut_seconds=cut_seconds,
                )
            else:
                cut_seconds = perf_counter() - cut_start
        else:
            ensure_compatibility_dendrogram(
                self.zw,
                graph_loc,
                generation_id,
                hierarchy,
            )
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

        metadata_start = perf_counter()
        self.cells.insert(
            final_label_key,
            result.labels,
            fill_value=-1,
            key=cell_key,
            overwrite=True,
        )
        metadata_write_seconds = perf_counter() - metadata_start

        if adaptive_digest is None:
            clear_active_adaptive_result(
                self.zw,
                graph_loc,
                label,
            )
        else:
            activate_adaptive_result(
                self.zw,
                graph_loc,
                label,
                adaptive_digest,
                metadata_write_seconds=metadata_write_seconds,
            )
        garbage_collect_hierarchy_generations(self.zw, graph_loc)
        return replace(
            result,
            label_key=final_label_key,
            hierarchy_generation_id=generation_id,
        )

    def run_clustering(
        self,
        from_assay: str | None = None,
        cell_key: str | None = None,
        feat_key: str | None = None,
        n_clusters: int | Literal["auto"] | None = None,
        integrated_graph: str | None = None,
        symmetric_graph: bool = False,
        graph_upper_only: bool = False,
        balanced_cut: bool = False,
        max_size: int | None = None,
        min_size: int | None = None,
        max_distance_fc: float = 2,
        force_recalc: bool = False,
        label: str = "cluster",
    ) -> None:
        """Deprecated forwarding shim for `run_paris_clustering`."""
        import warnings

        warnings.warn(
            "run_clustering is deprecated and will be removed in the next major "
            "release. Use run_paris_clustering instead.",
            FutureWarning,
            stacklevel=2,
        )
        if (
            balanced_cut
            or max_size is not None
            or min_size is not None
            or max_distance_fc != 2
        ):
            raise ValueError(
                "The DataStore balanced-cut mode has been removed. Use "
                "run_paris_clustering(n_clusters='auto') and optionally set "
                "min_cluster_size."
            )
        if n_clusters is None:
            raise ValueError(
                "n_clusters=None is no longer valid. Pass an integer or 'auto'."
            )
        if symmetric_graph or graph_upper_only:
            warnings.warn(
                "symmetric_graph and graph_upper_only are deprecated and ignored. "
                "Paris always uses the canonical additive graph.",
                FutureWarning,
                stacklevel=2,
            )
        self.run_paris_clustering(
            from_assay=from_assay,
            cell_key=cell_key,
            feat_key=feat_key,
            n_clusters=n_clusters,
            integrated_graph=integrated_graph,
            force_recalc=force_recalc,
            label=label,
        )
        return None

    def run_topacedo_sampler(
        self,
        from_assay: str | None = None,
        cell_key: str | None = None,
        feat_key: str | None = None,
        integrated_graph: str | None = None,
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
        return_edges: bool = False,
    ) -> None | list[Any]:
        """Perform sub-sampling (aka sketching) of cells using TopACeDo
        algorithm. Sub-sampling required that cells are partitioned in cluster
        already. Since, sub-sampling is dependent on cluster information,
        having, large number of homogeneous and even sized cluster improves
        sub-sampling results.

        Args:
            from_assay: Name of assay to be used. If no value is provided then the default assay will be used.
            cell_key: Cell key. Should be same as the one that was used in the desired graph. (Default value: 'I')
            feat_key: Feature key. Should be same as the one that was used in the desired graph. By default, the latest
                       used feature for the given assay will be used.
            integrated_graph: Integrated graph label. By default, use the latest assay graph.
            cluster_key: Name of the column in cell metadata table where cluster information is stored.
            use_k: Number of top k-nearest neighbours to retain in the graph over which downsampling is performed.
                   BY default all neighbours are used. (Default value: None)
            density_depth: Same as 'search_depth' parameter in `calc_neighbourhood_density`. (Default value: 2)
            density_bandwidth: This value is used to scale the penalty affected by neighbourhood density. Higher values
                               will lead to a larger penalty. (Default value: 5.0)
            max_sampling_rate: Maximum fraction of cells to sample from each group. The effective sampling rate is lower
                               than this value depending on the neighbourhood degree and SNN density of cells.
                               Should be greater than 0 and less than 1. (Default value: 0.1)
            min_sampling_rate: Minimum sampling rate. Effective sampling rate is not allowed to be lower than this
                               value. Should be greater than 0 and less than 1. (Default value: 0.01)
            min_cells_per_group: Minimum number of cells to sample from each group. (Default value: 3)
            snn_bandwidth: Bandwidth for the shared nearest neighbour award. Clusters with higher mean SNN values get
                           lower sampling penalty. This value, is raised to mean SNN value of the cluster to obtain
                           sampling reward of the cluster. (Default value: 5.0)
            seed_reward: Reward/prize value for seed nodes. (Default value: 3.0)
            non_seed_reward: Reward/prize for non-seed nodes. (Default value: 0.1)
            edge_cost_multiplier: This value is multiplier to each edge's cost. Higher values will make graph traversal
                                  costly and might lead to removal of poorly connected nodes (Default value: 1.0)
            edge_cost_bandwidth: This value is raised to edge cost to get an adjusted edge cost (Default value: 1.0)
            save_sampling_key: base label for marking the cells that were sampled into a cell metadata column
                               (Default value: 'sketched')
            save_density_key: base label for saving the cell neighbourhood densities into a cell metadata column
                              (Default value: 'cell_density')
            save_mean_snn_key: base label for saving the SNN value for each cell (identified by topacedo sampler) into
                               a cell metadata column (Default value: 'snn_value')
            save_seeds_key: base label for saving the seed cells (identified by topacedo sampler) into a cell
                            metadata column (Default value: 'sketch_seeds')
            rand_state: A random values to set seed while sampling cells from a cluster randomly. (Default value: 4466)
            return_edges: If True, then steiner nodes and edges are returned. (Default value: False)

        Returns:
        """

        try:
            from topacedo import TopacedoSampler
        except ImportError:
            logger.error("Could not find topacedo package")
            return None

        from_assay, cell_key, feat_key = self._get_latest_keys(
            from_assay, cell_key, feat_key
        )
        if cluster_key is None:
            raise ValueError("ERROR: Please provide a value for cluster key")
        clusters = pd.Series(self.cells.fetch(cluster_key, key=cell_key))
        if integrated_graph is None:
            graph_loc = self._get_latest_graph_loc(from_assay, cell_key, feat_key)
            output_assay = from_assay
        else:
            graph_loc = f"{self._integratedGraphsLoc}/{integrated_graph}"
            if graph_loc not in self.zw:
                raise KeyError(
                    f"An integrated graph with label {integrated_graph!r} does not exist"
                )
            output_assay = integrated_graph
        graph = self.load_graph(
            from_assay=from_assay,
            cell_key=cell_key,
            feat_key=feat_key,
            symmetric=False,
            upper_only=False,
            use_k=use_k,
            graph_loc=graph_loc,
        )
        from .paris_persistence import resolve_compatibility_dendrogram

        try:
            dendrogram_loc, _generation_id = resolve_compatibility_dendrogram(
                self.zw,
                graph_loc,
                final_label_key=cluster_key,
            )
            dendrogram = np.asarray(
                as_zarr_array(
                    self.zw[dendrogram_loc],
                    name=dendrogram_loc,
                )[:]
            )
        except KeyError:
            raise KeyError(
                "ERROR: Couldn't find the dendrogram for clustering. Please note that "
                "TopACeDo requires a dendrogram from Paris clustering."
            )

        if len(clusters) != graph.shape[0]:
            raise ValueError(
                f"ERROR: cluster information exists for {len(clusters)} cells while graph has "
                f"{graph.shape[0]} cells."
            )
        sampler = TopacedoSampler(
            graph,
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
        a = np.zeros(self.cells.fetch_all(cell_key).sum()).astype(bool)
        a[nodes] = True
        key = self._col_renamer(output_assay, cell_key, save_sampling_key)
        self.cells.insert(key, a, fill_value=False, key=cell_key, overwrite=True)
        logger.debug(f"Sketched cells saved under column '{key}'")

        key = self._col_renamer(output_assay, cell_key, save_density_key)
        self.cells.insert(key, sampler.densities, key=cell_key, overwrite=True)
        logger.debug(f"Cell neighbourhood densities saved under column: '{key}'")

        key = self._col_renamer(output_assay, cell_key, save_mean_snn_key)
        self.cells.insert(key, sampler.meanSnn, key=cell_key, overwrite=True)
        logger.debug(f"Mean SNN values saved under column: '{key}'")

        a = np.zeros(self.cells.fetch_all(cell_key).sum()).astype(bool)
        a[sampler.seeds] = True
        key = self._col_renamer(output_assay, cell_key, save_seeds_key)
        self.cells.insert(key, a, fill_value=False, key=cell_key, overwrite=True)
        logger.debug(f"Seed cells saved under column: '{key}'")

        if return_edges:
            return cast(list[Any], edges)
        return None
