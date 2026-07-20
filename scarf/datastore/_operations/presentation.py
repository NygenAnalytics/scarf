from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any, Literal, cast
import warnings

import numpy as np
import pandas as pd
import zarr

from ...storage.types import as_zarr_array, as_zarr_group
from ...storage.arrays import create_zarr_dataset
from ...utils.logging import logger

if TYPE_CHECKING:
    from ..mapping_datastore import MappingDatastore as _PresentationOperationsBase
else:
    _PresentationOperationsBase = object


class _PresentationOperationsMixin(_PresentationOperationsBase):
    def to_anndata(
        self,
        from_assay: str | None = None,
        cell_key: str | None = None,
        layers: dict[str, str] | None = None,
    ) -> Any:
        """Return an assay as an in-memory AnnData object.

        Cell and feature metadata are copied to ``obs`` and ``var``. Layout
        coordinates remain ordinary ``obs`` columns; this method does not
        populate ``obsm``.

        Args:
            from_assay: Name of assay to be used. If no value is provided then the default assay will be used.
            cell_key: Name of column from cell metadata that has boolean values. This is used to subset cells
            layers: A mapping of layer names to assay names. Ex. {'spliced': 'RNA', 'unspliced': 'URNA'}. The raw data
                    from the assays will be stored as sparse arrays in the corresponding layer in anndata.

        Returns:
            An AnnData object, or ``None`` when ``anndata`` is unavailable.
        """
        try:
            # noinspection PyPackageRequirements
            from anndata import AnnData  # type: ignore
        except ImportError:
            logger.error(
                "Package anndata is not installed because its an optional dependency. "
                "Install via `pip install anndata` or `conda install anndata -c conda-forge`"
            )
            return None

        if cell_key is None:
            cell_key = "I"
        assay = self._get_assay(from_assay)
        df = self.cells.to_pandas_dataframe(self.cells.columns, key=cell_key)
        obs = df.reset_index(drop=True).set_index("ids")
        df = assay.feats.to_pandas_dataframe(assay.feats.columns)
        var = df.rename(columns={"ids": "gene_ids"}).set_index("gene_ids")
        adata = AnnData(assay.to_raw_sparse(cell_key), obs=obs, var=var)
        if layers is not None:
            for layer, assay_name in layers.items():
                adata.layers[layer] = self._get_assay(assay_name).to_raw_sparse(
                    cell_key
                )
        return adata

    def show_zarr_tree(self, start: str = "/", depth: int = 2) -> None:
        """Prints the Zarr hierarchy of the DataStore.

        Args:
            start: Location in Zarr hierarchy to be used as the root for display
            depth: Depth of Zarr hierarchy to be displayed.

        Returns:
            None
        """
        from ...storage.layout import array_info

        root = start.strip("/")
        node: zarr.Group = (
            self.zw if root == "" else as_zarr_group(self.zw[root], name=root)
        )
        print(node.tree(level=depth))
        for key in node.array_keys():
            print(f"  {key}: {array_info(as_zarr_array(node[key], name=key))}")

    def calc_membership_strength(
        self, from_assay: str, cell_key: str, feat_key: str, clust_key: str
    ) -> None:
        """Store per-cell cluster membership strength from the latest KNN graph.

        For each cell, computes the fraction of KNN neighbors sharing the most
        common cluster label and saves it in cell metadata.

        Args:
            from_assay: Assay used to locate the KNN graph.
            cell_key: Boolean column selecting cells.
            feat_key: Feature key used when the graph was built.
            clust_key: Cell metadata column with cluster assignments.

        Returns:
            None
        """
        loc = self._get_latest_graph_loc(
            from_assay=from_assay, cell_key=cell_key, feat_key=feat_key
        )
        n_cells, k = self._get_graph_ncells_k(graph_loc=loc)
        clusts = self.cells.fetch(clust_key, key=cell_key)
        graph_grp = as_zarr_group(self.zw[loc], name=loc)
        edges = np.asarray(as_zarr_array(graph_grp["edges"], name="edges")[:])
        v = pd.DataFrame(clusts[edges[:, 1].reshape(k, n_cells)])
        x = np.array([v[x].value_counts().index[0] for x in v])
        self.cells.insert(
            f"{from_assay}_{cell_key}_cluster_membership_strength",
            (np.array((v == x).sum().values) / k).round(3),
            key=cell_key,
            overwrite=True,
        )
        return None

    def smart_label(
        self,
        to_relabel: str,
        base_label: str,
        cell_key: str = "I",
        new_col_name: str | None = None,
    ) -> None | list[str]:
        """A convenience function to relabel the values in a cell attribute
        column (A) based on the values in another cell attribute column (B).
        For each unique value in A, the most frequently occurring value in B is
        found. If two or more values in A have maximum overlap with the same
        value in B, then they all get the same label as B along with different
        suffixes like, 'a', 'b', etc. The suffixes are ordered based on where
        the largest fraction of the B label lies. If one label from A takes up
        multiple labels from B then all the labels from B are included, and they
        are delimited by hyphens.

        Args:
            to_relabel: Cell attributes column to relabel
            base_label: Cell attributes column to relabel
            cell_key: Cell key fetching column values
            new_col_name: Name of new column where relabeled values will be saved. If None then values
                          are returned and not saved in cell attributes table

        Returns: None or a list of relabelled values
        """
        df = pd.crosstab(
            self.cells.fetch(base_label, key=cell_key),
            self.cells.fetch(to_relabel, key=cell_key),
        )
        normed_frac = df.divide(df.sum(axis=1), axis="index")
        idxmax = df.idxmax()
        new_names = {}
        for i in sorted(idxmax.unique()):
            j = normed_frac[idxmax[idxmax == i].index].loc[i]
            j = j.sort_values(ascending=False).index
            for n, k in enumerate(j, start=1):
                a = chr(ord("@") + n)
                new_names[k] = f"{i}{a.lower()}"

        missing_vals = list(set(df.index).difference(idxmax.unique()))
        if len(missing_vals) > 0:
            miss_idxmax = df.loc[missing_vals].idxmax(axis=1).to_dict()
            for k, v in miss_idxmax.items():
                new_names[v] = f"{new_names[v][:-1]}-{k}{new_names[v][-1]}"

        ret_val = [new_names[x] for x in self.cells.fetch(to_relabel, key=cell_key)]
        if new_col_name is None:
            return ret_val
        else:
            self.cells.insert(new_col_name, ret_val, overwrite=True)
            return None

    def _prepare_cluster_tree(
        self,
        *,
        from_assay: str | None = None,
        cell_key: str | None = None,
        feat_key: str | None = None,
        cluster_key: str | None = None,
        fill_by_value: str | None = None,
    ) -> dict[str, Any]:
        from networkx import DiGraph, to_pandas_edgelist

        from ...clustering.hierarchy import CoalesceTree, make_digraph

        from_assay, cell_key, feat_key = self._get_latest_keys(
            from_assay, cell_key, feat_key
        )
        if cluster_key is None:
            raise ValueError(
                "ERROR: Please provide a value for `cluster_key` parameter"
            )

        clusters = np.asarray(self.cells.fetch(cluster_key, key=cell_key))
        graph_loc = self._get_latest_graph_loc(from_assay, cell_key, feat_key)
        graph_grp = as_zarr_group(self.zw[graph_loc], name=graph_loc)
        dendrogram_loc = cast(str, graph_grp.attrs["latest_dendrogram"])
        coalesced_loc = dendrogram_loc + f"_coalesced_{len(set(clusters))}"

        if coalesced_loc in self.zw:
            subgraph = DiGraph()
            subgraph.add_edges_from(
                np.asarray(
                    as_zarr_array(
                        self.zw[coalesced_loc + "/edgelist"],
                        name=f"{coalesced_loc}/edgelist",
                    )[:]
                )
            )
            nodelist = np.asarray(
                as_zarr_array(
                    self.zw[coalesced_loc + "/nodelist"],
                    name=f"{coalesced_loc}/nodelist",
                )[:]
            )
            partition_ids = np.asarray(
                as_zarr_array(
                    self.zw[coalesced_loc + "/partition_id"],
                    name=f"{coalesced_loc}/partition_id",
                )[:]
            )
            cluster_labels = {str(value): value for value in set(clusters)}
            for node_data, partition_id in zip(nodelist, partition_ids):
                node = int(node_data[0])
                subgraph.nodes[node]["nleaves"] = int(node_data[1])
                partition_text = str(partition_id)
                if partition_text != "-1":
                    subgraph.nodes[node]["partition_id"] = cluster_labels.get(
                        partition_text, partition_id
                    )
        else:
            dendrogram = np.asarray(
                as_zarr_array(self.zw[dendrogram_loc], name=dendrogram_loc)[:]
            )
            subgraph = CoalesceTree(make_digraph(dendrogram), clusters)
            edge_list = to_pandas_edgelist(subgraph).values
            store = create_zarr_dataset(
                self.zw,
                f"{coalesced_loc}/edgelist",
                (100000,),
                "u8",
                edge_list.shape,
            )
            store[:] = edge_list

            node_list = []
            partition_id_list = []
            for node in subgraph.nodes():
                node_data = subgraph.nodes[node]
                partition_id = node_data.get("partition_id", -1)
                node_list.append((node, node_data["nleaves"]))
                partition_id_list.append(str(partition_id))

            node_list_arr = np.asarray(node_list)
            store = create_zarr_dataset(
                self.zw,
                f"{coalesced_loc}/nodelist",
                (100000,),
                node_list_arr.dtype,
                node_list_arr.shape,
            )
            store[:] = node_list_arr

            store = create_zarr_dataset(
                self.zw,
                f"{coalesced_loc}/partition_id",
                (100000,),
                str,
                (len(partition_id_list),),
            )
            store[:] = partition_id_list

        color_values = (
            self.get_cell_vals(
                from_assay=from_assay,
                cell_key=cell_key,
                k=fill_by_value,
            )
            if fill_by_value is not None
            else None
        )
        return {
            "graph": subgraph,
            "clusters": clusters,
            "color_values": color_values,
            "from_assay": from_assay,
            "cell_key": cell_key,
            "feat_key": feat_key,
            "cluster_key": cluster_key,
            "coalesced_location": coalesced_loc,
        }

    def _load_metric_knn(
        self,
        use_latest_knn: bool,
        from_assay: str | None,
        knn_loc: str | None,
    ) -> tuple[str, str, str, zarr.Array, zarr.Array]:
        if from_assay is None:
            from_assay = self._load_default_assay()

        if use_latest_knn and knn_loc is None:
            resolved_knn_loc = self._get_latest_knn_loc(from_assay)
            logger.info(f"Using the latest knn graph at location: {resolved_knn_loc}")
        else:
            if knn_loc is None:
                raise ValueError("Please provide values for the KNN graph location.")
            if knn_loc not in self.zw:
                raise ValueError(f"Could not find the knn graph at location: {knn_loc}")
            resolved_knn_loc = knn_loc
            logger.info(f"Using the knn graph at location: {resolved_knn_loc}")

        normed_part = resolved_knn_loc.split("/")[1]
        _, cell_key, _ = normed_part.split("__")
        knn_grp = as_zarr_group(
            self.zw[resolved_knn_loc],
            name=resolved_knn_loc,
        )
        distances = as_zarr_array(knn_grp["distances"], name="distances")
        indices = as_zarr_array(knn_grp["indices"], name="indices")
        return from_assay, resolved_knn_loc, cell_key, distances, indices

    def metric_lisi(
        self,
        label_colnames: Iterable[str],
        use_latest_knn: bool = True,
        from_assay: str | None = None,
        knn_loc: str | None = None,
        save_result: bool = False,
        return_lisi: bool = True,
        perplexity: float = 30,
    ) -> list[tuple[str, np.ndarray]] | None:
        """Calculate Local Inverse Simpson Index (LISI) scores for cell populations.

        LISI measures how well mixed different cell populations are in the local neighborhood
        of each cell. Higher scores indicate better mixing of different populations.

        Args:
            label_colnames: Column names from cell metadata containing population labels
            use_latest_knn: Whether to use the most recent KNN graph (default: True)
            from_assay: Name of assay to use if not using latest KNN
            knn_loc: Location of KNN graph if not using latest (default: None)
            save_result: Whether to save LISI scores to cell metadata (default: False)
            return_lisi: Whether to return LISI scores (default: True)
            perplexity: Effective neighborhood size used by LISI. It is reduced
                with a warning when the graph has fewer than three times this
                many neighbors.

        Returns:
            If return_lisi is True, returns list of tuples containing:

            - Label column name
            - numpy array of LISI scores for that label

            If return_lisi is False, returns None

        Raises:
            ValueError: If KNN inputs, perplexity, or labels are invalid
            KeyError: If label columns not found in cell metadata

        Notes:
            LISI scores are computed for each label column separately.
            Scores near 1 indicate cells grouped with similar labels.
            Higher scores indicate more mixing between different labels.
        """

        label_cols = list(label_colnames)
        if from_assay is None:
            from_assay = self._load_default_assay()

        if use_latest_knn and knn_loc is None:
            knn_loc = self._get_latest_knn_loc(from_assay)
            logger.info(f"Using the latest knn graph at location: {knn_loc}")

        else:
            if knn_loc is None:
                raise ValueError("Please provide values for the KNN graph location.")
            if knn_loc not in self.zw:
                raise ValueError(f"Could not find the knn graph at location: {knn_loc}")

            logger.info(f"Using the knn graph at location: {knn_loc}")

        normed_part = knn_loc.split("/")[1]
        _, cell_key, _ = normed_part.split("__")
        knn_grp = as_zarr_group(self.zw[knn_loc], name=knn_loc)

        distances = as_zarr_array(knn_grp["distances"], name="distances")
        indices = as_zarr_array(knn_grp["indices"], name="indices")

        try:
            metadata = self.cells.to_pandas_dataframe(columns=label_cols + [cell_key])
            metadata = metadata[metadata[cell_key]]
        except KeyError:
            raise KeyError(
                f"Could not find the column(s) {label_cols} in the cell metadata table."
            )

        from ...metrics import compute_lisi

        lisi_scores = compute_lisi(
            distances,
            indices,
            metadata,
            label_cols,
            perplexity=perplexity,
        )
        # lisi_scores Shape -> (n_cells, n_labels)
        if save_result:
            for col, vals in zip(label_cols, lisi_scores.T):
                col_name = f"lisi__{col}__{knn_loc.split('/')[-1]}"
                self.cells.insert(
                    column_name=col_name, values=vals, overwrite=True, key=cell_key
                )

        if return_lisi:
            return list(zip(label_cols, lisi_scores.T))
        else:
            return None

    def metric_ilisi(
        self,
        batch_colname: str,
        use_latest_knn: bool = True,
        from_assay: str | None = None,
        knn_loc: str | None = None,
        perplexity: float | None = None,
        scale: bool = True,
    ) -> float:
        """Compute scIB integration LISI on a persisted KNN graph.

        Args:
            batch_colname: Cell metadata column containing batch labels.
            use_latest_knn: Use the latest KNN graph when ``knn_loc`` is not
                provided.
            from_assay: Assay used to resolve the latest KNN graph.
            knn_loc: Explicit persisted KNN location.
            perplexity: Effective neighborhood size. ``None`` uses
                ``floor(k / 3)``.
            scale: Scale the median LISI by the number of observed batches.

        Returns:
            Median iLISI, scaled so higher values indicate better batch mixing
            when ``scale`` is true.

        Notes:
            Scarf persisted KNN graphs exclude self-neighbors, as required by
            this metric.
        """
        from ...metrics import ilisi_knn

        _, _, cell_key, distances, indices = self._load_metric_knn(
            use_latest_knn,
            from_assay,
            knn_loc,
        )
        batch_labels = self.cells.fetch(batch_colname, key=cell_key)
        return ilisi_knn(
            distances,
            indices,
            batch_labels,
            perplexity=perplexity,
            scale=scale,
        )

    def metric_clisi(
        self,
        label_colname: str,
        use_latest_knn: bool = True,
        from_assay: str | None = None,
        knn_loc: str | None = None,
        perplexity: float | None = None,
        scale: bool = True,
    ) -> float:
        """Compute scIB cell-type LISI on a persisted KNN graph.

        Args:
            label_colname: Cell metadata column containing biological labels.
            use_latest_knn: Use the latest KNN graph when ``knn_loc`` is not
                provided.
            from_assay: Assay used to resolve the latest KNN graph.
            knn_loc: Explicit persisted KNN location.
            perplexity: Effective neighborhood size. ``None`` uses
                ``floor(k / 3)``.
            scale: Invert and scale the median LISI by the number of observed
                labels.

        Returns:
            Median cLISI, scaled so higher values indicate better label
            conservation when ``scale`` is true.

        Notes:
            Scarf persisted KNN graphs exclude self-neighbors, as required by
            this metric.
        """
        from ...metrics import clisi_knn

        _, _, cell_key, distances, indices = self._load_metric_knn(
            use_latest_knn,
            from_assay,
            knn_loc,
        )
        cell_labels = self.cells.fetch(label_colname, key=cell_key)
        return clisi_knn(
            distances,
            indices,
            cell_labels,
            perplexity=perplexity,
            scale=scale,
        )

    def metric_graph_connectivity(
        self,
        label_colname: str,
        from_assay: str | None = None,
        cell_key: str | None = None,
        feat_key: str | None = None,
        graph_loc: str | None = None,
    ) -> float:
        """Score label connectivity on a persisted, symmetrized assay graph.

        Args:
            label_colname: Cell metadata column containing biological labels.
            from_assay: Assay used to resolve the latest graph.
            cell_key: Cell-selection key used to resolve the latest graph.
            feat_key: Feature-selection key used to resolve the latest graph.
            graph_loc: Explicit persisted standard-assay graph location.

        Returns:
            Mean fraction of cells retained in the largest connected component
            for each label.

        Notes:
            Persisted directed edges are treated as undirected. This follows
            the original scIB symmetrized-graph definition and intentionally
            differs from the directed strong-component calculation currently
            used by YosefLab ``scib-metrics``. Integrated graph locations are
            rejected because they do not preserve safe cell-key provenance.
        """
        from ...metrics import graph_connectivity

        if graph_loc is None:
            from_assay, cell_key, feat_key = self._get_latest_keys(
                from_assay,
                cell_key,
                feat_key,
            )
            graph_loc = self._get_latest_graph_loc(
                from_assay,
                cell_key,
                feat_key,
            )
        else:
            if graph_loc.startswith(self._integratedGraphsLoc):
                raise ValueError(
                    "Integrated graph connectivity is unavailable because the "
                    "graph does not record its cell-key provenance"
                )
            if graph_loc not in self.zw:
                raise ValueError(f"Could not find the graph at location: {graph_loc}")

            path_parts = graph_loc.split("/")
            if len(path_parts) < 2:
                raise ValueError(
                    f"Could not determine graph provenance from location: {graph_loc}"
                )
            normed_parts = path_parts[1].split("__")
            if len(normed_parts) != 3 or normed_parts[0] != "normed":
                raise ValueError(
                    f"Could not determine graph provenance from location: {graph_loc}"
                )

            path_assay = path_parts[0]
            _, path_cell_key, path_feat_key = normed_parts
            if from_assay is not None and from_assay != path_assay:
                raise ValueError("from_assay does not match the graph location")
            if cell_key is not None and cell_key != path_cell_key:
                raise ValueError("cell_key does not match the graph location")
            if feat_key is not None and feat_key != path_feat_key:
                raise ValueError("feat_key does not match the graph location")
            from_assay = path_assay
            cell_key = path_cell_key
            feat_key = path_feat_key

        n_cells, _ = self._get_graph_ncells_k(graph_loc)
        labels = self.cells.fetch(label_colname, key=cell_key)
        if len(labels) != n_cells:
            raise ValueError("Graph labels must match the number of cells in the graph")

        graph_grp = as_zarr_group(self.zw[graph_loc], name=graph_loc)
        edges = as_zarr_array(graph_grp["edges"], name="edges")
        return graph_connectivity(edges, labels)

    def metric_graph_silhouette(
        self,
        use_latest_knn: bool = True,
        res_label: str = "leiden_cluster",
        from_assay: str | None = None,
        knn_loc: str | None = None,
        random_seed: int = 4444,
        sample_size: int = 11,
    ) -> np.ndarray | None:
        """Calculate modified silhouette scores for evaluating cluster separation.

        This implements a graph-based silhouette score that measures how similar cells
        are to their own cluster compared to the nearest neighboring cluster.

        Args:
            use_latest_knn: Whether to use most recent KNN graph (default: True)
            res_label: Base or full column name containing cluster labels
                (default: "leiden_cluster")
            from_assay: Name of assay to use if not using latest KNN (default: None)
            knn_loc: Location of KNN graph if not using latest (default: None)
            random_seed: Seed used for cluster sampling.
            sample_size: Maximum size of each sampled cluster group.

        Returns:
            numpy array of silhouette scores for each cluster, or None if computation fails

        Raises:
            ValueError: If graph, labels, sampling, or embedding data are invalid

        Notes:
            Scores range from -1 to 1:
            - Near 1: Cluster is well-separated from neighboring clusters
            - Near 0: Cluster overlaps with neighboring clusters
            - Near -1: Cluster may be incorrectly assigned

            Implementation uses sampling for efficiency with large datasets.
            NaN values indicate clusters that couldn't be scored due to size constraints.
        """

        if from_assay is None:
            from_assay = self._load_default_assay()

        if use_latest_knn and knn_loc is None:
            knn_loc = self._get_latest_knn_loc(from_assay)
            logger.info(
                f"Using the latest knn graph at location: {knn_loc} for assay: {from_assay}"
            )

        else:
            if knn_loc is None:
                raise ValueError("Please provide values for the KNN graph location.")
            if knn_loc not in self.zw:
                raise ValueError(f"Could not find the knn graph at location: {knn_loc}")
            logger.info(f"Using the knn graph at location: {knn_loc}")

        from ...metrics import silhouette_scoring

        normed_part = knn_loc.split("/")[1]
        _, cell_key, feat_key_parsed = normed_part.split("__")
        ann_obj = self._load_ann_stream(
            from_assay=from_assay,
            cell_key=cell_key,
            feat_key=feat_key_parsed,
            knn_loc=knn_loc,
        )

        knn_grp = as_zarr_group(self.z[knn_loc], name=knn_loc)
        neighbor_indices = as_zarr_array(knn_grp["indices"], name="indices")
        neighbor_distances = as_zarr_array(knn_grp["distances"], name="distances")
        if ann_obj.harmonizedData is not None:
            metric_data = ann_obj.harmonizedData
            data_is_reduced = True
        else:
            if ann_obj.harmonize:
                raise ValueError("Harmony coordinates are missing for this KNN graph")
            metric_data = ann_obj.data
            data_is_reduced = False
        scores = silhouette_scoring(
            self,  # type: ignore[arg-type]
            ann_obj,
            None,
            metric_data,
            from_assay,
            res_label,
            cell_key=cell_key,
            random_seed=random_seed,
            sample_size=sample_size,
            data_is_reduced=data_is_reduced,
            distance_metric=cast(Any, ann_obj.annMetric),
            neighbor_indices=neighbor_indices,
            neighbor_distances=neighbor_distances,
        )
        return scores

    def metric_silhouette(
        self,
        use_latest_knn: bool = True,
        res_label: str = "leiden_cluster",
        from_assay: str | None = None,
        knn_loc: str | None = None,
        random_seed: int = 4444,
        sample_size: int = 11,
    ) -> np.ndarray | None:
        """Deprecated alias for :meth:`metric_graph_silhouette`."""
        warnings.warn(
            "metric_silhouette is deprecated and will be removed in Scarf 2.0. "
            "Use metric_graph_silhouette instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.metric_graph_silhouette(
            use_latest_knn=use_latest_knn,
            res_label=res_label,
            from_assay=from_assay,
            knn_loc=knn_loc,
            random_seed=random_seed,
            sample_size=sample_size,
        )

    def metric_label_concordance(
        self,
        label_columns: Sequence[str],
        metric: Literal["ari", "nmi"] = "ari",
    ) -> float:
        """Compare two metadata label partitions using ARI or NMI.

        This measures whether two labelings of the same cells agree, for
        example predicted clusters against imported reference annotations. It
        does not measure batch mixing; use :meth:`metric_ilisi`,
        :meth:`metric_proportional_batch_mixing`, or :meth:`metric_lisi`
        for that.

        Args:
            label_columns: Exactly two cell metadata column names to compare.
            metric: ``"ari"`` for the adjusted Rand index or ``"nmi"`` for
                normalized mutual information.

        Returns:
            Agreement between the two partitions. ARI ranges from -1 to 1 and
            NMI from 0 to 1, with higher values meaning stronger agreement.

        Raises:
            ValueError: If the number of columns or the metric name is invalid.
        """
        from ...metrics import label_concordance_score

        label_values = [
            np.asarray(self.cells.fetch_all(column)) for column in label_columns
        ]
        return label_concordance_score(label_values, metric)

    def metric_integration(
        self,
        batch_labels: list[str],
        metric: Literal["ari", "nmi"] = "ari",
    ) -> float:
        """Backward-compatible alias for :meth:`metric_label_concordance`.

        This method compares label agreement and does not measure neighborhood
        mixing. Use :meth:`metric_ilisi` or
        :meth:`metric_proportional_batch_mixing` to evaluate batch integration.
        """
        warnings.warn(
            "metric_integration is deprecated and will be removed in Scarf 2.0. "
            "Use metric_label_concordance for ARI/NMI, metric_ilisi for scIB "
            "iLISI, or metric_proportional_batch_mixing for Scarf's "
            "proportion-aware batch mixing score.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.metric_label_concordance(batch_labels, metric)

    def metric_proportional_batch_mixing(
        self,
        label_colname: str,
        use_latest_knn: bool = True,
        from_assay: str | None = None,
        knn_loc: str | None = None,
        perplexity: float = 30,
    ) -> float:
        """Summarize batch LISI as a normalized neighborhood-mixing score.

        This computes batch LISI on the current KNN graph and rescales its mean
        against the mixing that perfectly integrated data would reach given the
        dataset's batch sizes. Unlike raw LISI, the result is bounded in
        ``[0, 1]``, which makes it easier to compare across graphs and datasets.

        Args:
            label_colname: Cell metadata column holding the batch assignment.
            use_latest_knn: Whether to use the most recent KNN graph
                (default: True).
            from_assay: Name of assay to use if not using the latest KNN.
            knn_loc: Location of the KNN graph if not using the latest.
            perplexity: Effective neighborhood size passed to LISI.

        Returns:
            A value in ``[0, 1]``. Scores near 1 indicate that neighborhoods mix
            batches as well as the global composition allows, and scores near 0
            indicate poorly mixed batches.

        Raises:
            ValueError: If KNN inputs are invalid or the column has fewer than
                two batches.
        """
        from ...metrics import lisi_batch_mixing_score

        if from_assay is None:
            from_assay = self._load_default_assay()
        resolved_knn_loc = knn_loc
        if use_latest_knn and resolved_knn_loc is None:
            resolved_knn_loc = self._get_latest_knn_loc(from_assay)
        if resolved_knn_loc is None:
            raise ValueError("Please provide values for the KNN graph location.")

        lisi_result = self.metric_lisi(
            label_colnames=[label_colname],
            use_latest_knn=use_latest_knn,
            from_assay=from_assay,
            knn_loc=resolved_knn_loc,
            save_result=False,
            return_lisi=True,
            perplexity=perplexity,
        )
        if lisi_result is None:
            raise RuntimeError("LISI computation did not return scores")

        normed_part = resolved_knn_loc.split("/")[1]
        _, cell_key, _ = normed_part.split("__")
        batch_labels = self.cells.fetch(label_colname, key=cell_key)
        return lisi_batch_mixing_score(lisi_result[0][1], batch_labels)

    def metric_batch_mixing(
        self,
        label_colname: str,
        use_latest_knn: bool = True,
        from_assay: str | None = None,
        knn_loc: str | None = None,
        perplexity: float = 30,
    ) -> float:
        """Deprecated alias for :meth:`metric_proportional_batch_mixing`."""
        warnings.warn(
            "metric_batch_mixing is deprecated and will be removed in Scarf 2.0. "
            "Use metric_proportional_batch_mixing instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.metric_proportional_batch_mixing(
            label_colname=label_colname,
            use_latest_knn=use_latest_knn,
            from_assay=from_assay,
            knn_loc=knn_loc,
            perplexity=perplexity,
        )
