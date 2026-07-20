from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd

from ...storage.types import as_zarr_array, as_zarr_group
from ...storage.arrays import create_zarr_dataset
from ...utils.logging import logger

if TYPE_CHECKING:
    from .graph import _GraphOperationsMixin as _ClusteringOperationsBase
else:
    _ClusteringOperationsBase = object


class _ClusteringOperationsMixin(_ClusteringOperationsBase):
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
        print(
            "[run_leiden_clustering] ENTER load_graph "
            f"assay={from_assay} cell_key={cell_key} feat_key={feat_key}",
            flush=True,
        )
        graph = self.load_graph(
            from_assay=from_assay,
            cell_key=cell_key,
            feat_key=feat_key,
            symmetric=symmetric_graph,
            upper_only=graph_upper_only,
            graph_loc=graph_loc,
        )
        print(
            f"[run_leiden_clustering] load_graph DONE shape={graph.shape} "
            f"nnz={graph.nnz}; ENTER leiden_membership",
            flush=True,
        )
        if integrated_graph is not None:
            from_assay = integrated_graph
        membership = leiden_membership(graph, resolution, random_seed)
        print(
            f"[run_leiden_clustering] leiden_membership DONE "
            f"n_labels={membership.size}",
            flush=True,
        )
        self.cells.insert(
            self._col_renamer(from_assay, cell_key, label),
            membership,
            fill_value=-1,
            key=cell_key,
            overwrite=True,
        )
        return None

    def run_clustering(
        self,
        from_assay: str | None = None,
        cell_key: str | None = None,
        feat_key: str | None = None,
        n_clusters: int | None = None,
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
        """Executes Paris clustering algorithm
        (https://arxiv.org/pdf/1806.01664.pdf) on the cell-neighbourhood graph.
        The algorithm captures the multiscale structure of the graph in to an
        ordinary dendrogram structure. The distances in the dendrogram are
        based on probability of sampling node (aka cell) pairs. These methods
        create this dendrogram if it doesn't already exist for the graph and
        induces either a straight cut or balanced cut to obtain clusters of
        cells.

        Args:
            from_assay: Name of assay to be used. If no value is provided then the default assay will be used.
            cell_key: Cell key. Should be same as the one that was used in the desired graph. (Default value: 'I')
            feat_key:  Feature key. Should be same as the one that was used in the desired graph. By default, the latest
                       used feature for the given assay will be used.
            n_clusters: Number of desired clusters (required if balanced_cut is False)
            integrated_graph:
            symmetric_graph: This parameter is forwarded to `load_graph` and is same as there. (Default value: True)
            graph_upper_only: This parameter is forwarded to `load_graph` and is same as there. (Default value: True)
            balanced_cut: If True, then uses the balanced cut algorithm as implemented in ``BalancedCut`` to obtain
                          clusters (Default value: False)
            max_size: Same as `max_size` in ``BalancedCut``. The limit for a maximum number of cells in a cluster.
                      This parameter value is required if `balanced_cut` is True.
            min_size: Same as `min_size` in ``BalancedCut``. The limit for a minimum number of cells in a cluster.
                      This parameter value is required if `balanced_cut` is True.
            max_distance_fc:  Same as `max_distance_fc` in ``BalancedCut``. The threshold of ratio of distance between
                              two clusters beyond which they will not be merged. (Default value: 2)
            force_recalc: Forces recalculation of dendrogram even if one already exists for the graph
            label: Base label for cluster identity in the cell metadata column (Default value: 'cluster')

        Returns:
            None
        """
        from ...clustering.paris import (
            balanced_cut as find_balanced_cut,
            paris_dendrogram,
            straight_cut,
        )

        if balanced_cut is False:
            if n_clusters is None:
                raise ValueError(
                    "ERROR: Please provide a value for n_clusters parameter. We are working on making "
                    "this parameter free"
                )
        else:
            if n_clusters is not None:
                logger.info(
                    "Using balanced cut method for cutting dendrogram. `n_clusters` will be ignored."
                )
            if max_size is None or min_size is None:
                raise ValueError(
                    "ERROR: Please provide value for max_size and min_size"
                )

        from_assay, cell_key, feat_key = self._get_latest_keys(
            from_assay, cell_key, feat_key
        )

        graph_loc = self._get_latest_graph_loc(from_assay, cell_key, feat_key)
        if integrated_graph is not None:
            graph_loc = f"{self._integratedGraphsLoc}/{integrated_graph}"
            if graph_loc not in self.zw:
                raise KeyError(
                    f"ERROR: An integrated graph with label: {integrated_graph} does not exist"
                )

        dendrogram_loc = f"{graph_loc}/dendrogram"
        # tuple are changed to list when saved as zarr attrs
        if dendrogram_loc in self.zw and force_recalc is False:
            dendrogram = np.asarray(
                as_zarr_array(self.zw[dendrogram_loc], name=dendrogram_loc)[:]
            )
            logger.info("Using existing dendrogram")
        else:
            graph = self.load_graph(
                from_assay=from_assay,
                cell_key=cell_key,
                feat_key=feat_key,
                symmetric=symmetric_graph,
                upper_only=graph_upper_only,
                graph_loc=graph_loc,
            )
            dendrogram = paris_dendrogram(graph)
            graph_grp = as_zarr_group(self.zw[graph_loc], name=graph_loc)
            g = create_zarr_dataset(
                graph_grp,
                dendrogram_loc.rsplit("/", 1)[1],
                (5000,),
                "f8",
                (graph.shape[0] - 1, 4),
            )
            g[:] = dendrogram
        as_zarr_group(self.zw[graph_loc], name=graph_loc).attrs["latest_dendrogram"] = (
            dendrogram_loc
        )

        if balanced_cut:
            assert max_size is not None and min_size is not None
            labels = find_balanced_cut(
                dendrogram,
                max_size,
                min_size,
                max_distance_fc,
            )
            logger.info(f"{len(set(labels))} clusters found")
        else:
            assert n_clusters is not None
            labels = straight_cut(dendrogram, n_clusters)

        if integrated_graph is not None:
            from_assay = integrated_graph
        self.cells.insert(
            self._col_renamer(from_assay, cell_key, label),
            labels,
            fill_value=-1,
            key=cell_key,
            overwrite=True,
        )
        return None

    def run_topacedo_sampler(
        self,
        from_assay: str | None = None,
        cell_key: str | None = None,
        feat_key: str | None = None,
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
        graph = self.load_graph(
            from_assay=from_assay,
            cell_key=cell_key,
            feat_key=feat_key,
            symmetric=False,
            upper_only=False,
            use_k=use_k,
        )
        graph_loc = self._get_latest_graph_loc(from_assay, cell_key, feat_key)
        try:
            dendrogram = np.asarray(
                as_zarr_array(
                    self.zw[f"{graph_loc}/dendrogram"],
                    name=f"{graph_loc}/dendrogram",
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
        key = self._col_renamer(from_assay, cell_key, save_sampling_key)
        self.cells.insert(key, a, fill_value=False, key=cell_key, overwrite=True)
        logger.debug(f"Sketched cells saved under column '{key}'")

        key = self._col_renamer(from_assay, cell_key, save_density_key)
        self.cells.insert(key, sampler.densities, key=cell_key, overwrite=True)
        logger.debug(f"Cell neighbourhood densities saved under column: '{key}'")

        key = self._col_renamer(from_assay, cell_key, save_mean_snn_key)
        self.cells.insert(key, sampler.meanSnn, key=cell_key, overwrite=True)
        logger.debug(f"Mean SNN values saved under column: '{key}'")

        a = np.zeros(self.cells.fetch_all(cell_key).sum()).astype(bool)
        a[sampler.seeds] = True
        key = self._col_renamer(from_assay, cell_key, save_seeds_key)
        self.cells.insert(key, a, fill_value=False, key=cell_key, overwrite=True)
        logger.debug(f"Seed cells saved under column: '{key}'")

        if return_edges:
            return cast(list[Any], edges)
        return None
