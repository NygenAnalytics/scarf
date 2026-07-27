from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
import zarr
from scipy.sparse import csr_matrix, vstack

from ...storage.types import as_zarr_array, as_zarr_group
from ...storage.arrays import create_zarr_dataset
from ...storage.artifacts import (
    ArtifactRef,
    artifact_path,
    fingerprint_stored_arrays,
    inspect_artifact,
    parse_artifact_path,
)
from ...storage.artifact_writer import (
    ArrayRequirement,
    finish_artifact,
    plan_artifact,
    reused_artifact_group,
    start_artifact,
)
from ...graph.state import (
    resolve_stored_graph_input,
    validate_legacy_graph_selection,
)
from ...metadata.arguments import (
    MembershipStrengthArguments,
    SmartLabelArguments,
)
from ...metadata.artifacts import (
    artifact_values,
    categorical_display,
    column_display,
    continuous_display,
    link_cell_data_column,
    plan_cell_data_artifact,
    write_cell_data_artifact,
)
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
        *,
        matrix: Literal["raw", "normed"] = "raw",
        feature_indexes: Sequence[int] | None = None,
        feature_names: Sequence[str] | None = None,
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
            matrix: Whether ``X`` contains raw counts or normalized values.
            feature_indexes: Global feature rows to export, in the requested order.
            feature_names: Feature names to export, in the requested order.

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
        if matrix not in ("raw", "normed"):
            raise ValueError("matrix must be either 'raw' or 'normed'")
        if feature_indexes is not None and feature_names is not None:
            raise ValueError("feature_indexes and feature_names are mutually exclusive")

        if feature_indexes is not None:
            if isinstance(feature_indexes, str):
                raise TypeError(
                    "feature_indexes must be a sequence of integer feature indexes"
                )
            feat_idx = np.asarray(feature_indexes)
            if feat_idx.ndim != 1:
                raise ValueError("feature_indexes must be one-dimensional")
            if feat_idx.size == 0:
                feat_idx = np.empty(0, dtype=np.int64)
            elif not np.issubdtype(feat_idx.dtype, np.integer):
                raise TypeError("feature_indexes must contain only integers")
            else:
                feat_idx = feat_idx.astype(np.int64, copy=False)
            if np.unique(feat_idx).size != feat_idx.size:
                raise ValueError("feature_indexes must contain unique indexes")
            if np.any(feat_idx < 0) or np.any(feat_idx >= assay.feats.N):
                raise IndexError("feature_indexes contains an out-of-range index")
        elif feature_names is not None:
            if isinstance(feature_names, str):
                raise TypeError(
                    "feature_names must be a sequence of feature names, not a string"
                )
            requested_names = list(feature_names)
            if not all(isinstance(name, str) for name in requested_names):
                raise TypeError("feature_names must contain only strings")
            if len(set(requested_names)) != len(requested_names):
                raise ValueError("feature_names must contain unique names")
            name_positions: dict[str, list[int]] = {}
            for index, name in enumerate(assay.feats.fetch_all("names").astype(str)):
                name_positions.setdefault(name, []).append(index)
            missing = [name for name in requested_names if name not in name_positions]
            if missing:
                raise KeyError("Feature names not found: " + ", ".join(missing))
            ambiguous = [
                name for name in requested_names if len(name_positions[name]) != 1
            ]
            if ambiguous:
                raise ValueError(
                    "Feature names are not unique in the assay: " + ", ".join(ambiguous)
                )
            feat_idx = np.asarray(
                [name_positions[name][0] for name in requested_names],
                dtype=np.int64,
            )
        else:
            feat_idx = np.arange(assay.feats.N, dtype=np.int64)

        cell_idx = self.cells.active_index(cell_key)
        df = self.cells.to_pandas_dataframe(self.cells.columns, key=cell_key)
        obs = df.reset_index(drop=True).set_index("ids")
        df = assay.feats.to_pandas_dataframe(assay.feats.columns).iloc[feat_idx]
        var = df.rename(columns={"ids": "gene_ids"}).set_index("gene_ids")
        if matrix == "raw":
            x = assay.to_raw_sparse(cell_key)[:, feat_idx].tocsr()
        else:
            normed = assay.normed(cell_idx=cell_idx, feat_idx=feat_idx)
            blocks = [csr_matrix(block) for block in normed.stream_blocks()]
            x = (
                vstack(blocks, format="csr")
                if blocks
                else csr_matrix((len(cell_idx), len(feat_idx)))
            )
        adata = AnnData(x, obs=obs, var=var)
        if layers is not None:
            selected_ids = assay.feats.fetch_all("ids").astype(str)[feat_idx]
            if np.unique(selected_ids).size != selected_ids.size:
                raise ValueError(
                    "Selected feature IDs must be unique when exporting layers"
                )
            for layer, assay_name in layers.items():
                layer_assay = self._get_assay(assay_name)
                layer_matrix = layer_assay.to_raw_sparse(cell_key)
                layer_id_positions: dict[str, list[int]] = {}
                for index, feature_id in enumerate(
                    layer_assay.feats.fetch_all("ids").astype(str)
                ):
                    layer_id_positions.setdefault(feature_id, []).append(index)
                missing_ids = [
                    feature_id
                    for feature_id in selected_ids
                    if feature_id not in layer_id_positions
                ]
                ambiguous_ids = [
                    feature_id
                    for feature_id in selected_ids
                    if len(layer_id_positions.get(feature_id, ())) > 1
                ]
                if missing_ids or ambiguous_ids:
                    details = []
                    if missing_ids:
                        details.append("missing: " + ", ".join(missing_ids))
                    if ambiguous_ids:
                        details.append("ambiguous: " + ", ".join(ambiguous_ids))
                    raise ValueError(
                        f"Layer {layer!r} cannot align selected feature IDs ("
                        + "; ".join(details)
                        + ")"
                    )
                layer_feat_idx = np.asarray(
                    [layer_id_positions[feature_id][0] for feature_id in selected_ids],
                    dtype=np.int64,
                )
                adata.layers[layer] = layer_matrix[:, layer_feat_idx].tocsr()
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
        self,
        from_assay: str,
        cell_key: str,
        feat_key: str,
        clust_key: str,
        invalidate_cache: bool = False,
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
        loc = self.get_latest_graph_loc(
            from_assay=from_assay, cell_key=cell_key, feat_key=feat_key
        )
        n_cells, k = self._get_graph_ncells_k(graph_loc=loc)
        selection = self._ensure_cell_selection(cell_key)
        graph_input: object = resolve_stored_graph_input(self.zw, loc)
        if isinstance(graph_input, ArtifactRef):
            graph_selection = self._graph_cell_selection(graph_input)
            if not self._selection_artifacts_match(graph_selection, selection):
                raise ValueError("cell_key does not match the graph cell selection")
        else:
            validate_legacy_graph_selection(
                self,
                loc,
                from_assay,
                cell_key,
                feat_key,
            )
        cluster_input = self._resolve_cell_data_provenance_input(
            clust_key,
            cell_key=cell_key,
        )
        output_key = f"{from_assay}_{cell_key}_cluster_membership_strength"
        arguments = MembershipStrengthArguments(
            connectivity_map=graph_input,
            clusters=cluster_input,
            cell_selection=selection,
            algorithm_version=2,
            decimals=3,
            from_assay=from_assay,
            cell_key=cell_key,
            feat_key=feat_key,
            clust_key=clust_key,
            output_key=output_key,
            invalidate_cache=invalidate_cache,
        )
        record = arguments.to_record()
        planned = plan_cell_data_artifact(
            self.zw,
            scope="assay",
            assay=from_assay,
            kind=arguments.artifact_kind,
            operation=arguments.operation,
            parameters=record.parameters,
            inputs=record.inputs,
            execution_options=record.execution_options,
            cell_selection=selection,
            arrays={"values": ((n_cells,), "f")},
            invalidate_cache=invalidate_cache,
        )
        preserved_display = column_display(self.zw, output_key)
        if planned.reused:
            artifact_group = as_zarr_group(
                self.zw[artifact_path(planned.ref)],
                name=planned.ref.artifact_id,
            )
            values = artifact_values(artifact_group, "values")
            self.cells.insert(
                output_key,
                values,
                key=cell_key,
                overwrite=True,
            )
            link_cell_data_column(
                self.zw,
                output_key,
                planned.ref,
                value_name="values",
                default_display={
                    **continuous_display(values),
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                preserved_display=preserved_display,
            )
            return None
        clusts = self.cells.fetch(clust_key, key=cell_key)
        graph_grp = as_zarr_group(self.zw[loc], name=loc)
        edges = np.asarray(as_zarr_array(graph_grp["edges"], name="edges")[:])
        if edges.shape != (n_cells * k, 2):
            raise ValueError(
                "Graph edges do not match the stored cell and k dimensions"
            )
        edge_rows = edges.reshape(n_cells, k, 2)
        expected_sources = np.broadcast_to(
            np.arange(n_cells, dtype=edge_rows.dtype)[:, None],
            (n_cells, k),
        )
        if not np.array_equal(edge_rows[:, :, 0], expected_sources):
            raise ValueError("Graph edges are not stored in cell-major order")
        neighbor_clusters = np.asarray(clusts)[edge_rows[:, :, 1]]
        values = np.asarray(
            [
                pd.Series(row).value_counts(dropna=False).iloc[0] / k
                for row in neighbor_clusters
            ],
            dtype=np.float64,
        ).round(3)
        write_cell_data_artifact(
            self.zw,
            planned,
            {"values": values},
        )
        self.cells.insert(
            output_key,
            values,
            key=cell_key,
            overwrite=True,
        )
        link_cell_data_column(
            self.zw,
            output_key,
            planned.ref,
            value_name="values",
            default_display={
                **continuous_display(values),
                "minimum": 0.0,
                "maximum": 1.0,
            },
            preserved_display=preserved_display,
        )
        return None

    def smart_label(
        self,
        to_relabel: str,
        base_label: str,
        cell_key: str = "I",
        new_col_name: str | None = None,
        invalidate_cache: bool = False,
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
        values_to_relabel = np.asarray(self.cells.fetch(to_relabel, key=cell_key))
        base_values = np.asarray(self.cells.fetch(base_label, key=cell_key))
        if len(values_to_relabel) == 0:
            if new_col_name is None:
                return []
            raise ValueError(f"cell_key {cell_key!r} selects no cells")
        planned = None
        preserved_display = None
        if new_col_name is not None:
            selection = self._ensure_cell_selection(cell_key)
            arguments = SmartLabelArguments(
                values=self._resolve_cell_data_provenance_input(
                    to_relabel,
                    cell_key=cell_key,
                ),
                base_labels=self._resolve_cell_data_provenance_input(
                    base_label,
                    cell_key=cell_key,
                ),
                cell_selection=selection,
                algorithm_version=2,
                suffix_style="lowercase_letter",
                to_relabel=to_relabel,
                base_label=base_label,
                cell_key=cell_key,
                new_col_name=new_col_name,
                invalidate_cache=invalidate_cache,
            )
            record = arguments.to_record()
            planned = plan_cell_data_artifact(
                self.zw,
                scope="datastore",
                kind=arguments.artifact_kind,
                operation=arguments.operation,
                parameters=record.parameters,
                inputs=record.inputs,
                execution_options=record.execution_options,
                cell_selection=selection,
                arrays={"values": ((len(self.cells.active_index(cell_key)),), None)},
                invalidate_cache=invalidate_cache,
            )
            preserved_display = column_display(self.zw, new_col_name)
            if planned.reused:
                artifact_group = as_zarr_group(
                    self.zw[artifact_path(planned.ref)],
                    name=planned.ref.artifact_id,
                )
                values = artifact_values(artifact_group, "values")
                self.cells.insert(
                    new_col_name,
                    values,
                    key=cell_key,
                    overwrite=True,
                )
                link_cell_data_column(
                    self.zw,
                    new_col_name,
                    planned.ref,
                    value_name="values",
                    default_display=categorical_display(values),
                    preserved_display=preserved_display,
                )
                return None

        df = pd.crosstab(
            base_values,
            values_to_relabel,
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

        missing_vals = df.index.difference(
            pd.Index(idxmax.unique()),
            sort=False,
        ).tolist()
        if len(missing_vals) > 0:
            miss_idxmax = df.loc[missing_vals].idxmax(axis=1).to_dict()
            for k, v in miss_idxmax.items():
                new_names[v] = f"{new_names[v][:-1]}-{k}{new_names[v][-1]}"

        ret_val = [new_names[x] for x in values_to_relabel]
        if new_col_name is None:
            return ret_val
        assert planned is not None
        values = np.asarray(ret_val)
        write_cell_data_artifact(
            self.zw,
            planned,
            {"values": values},
        )
        self.cells.insert(
            new_col_name,
            values,
            key=cell_key,
            overwrite=True,
        )
        link_cell_data_column(
            self.zw,
            new_col_name,
            planned.ref,
            value_name="values",
            default_display=categorical_display(values),
            preserved_display=preserved_display,
        )
        return None

    def _prepare_artifact_cluster_tree(
        self,
        *,
        graph_ref: ArtifactRef | dict[str, Any],
        graph_loc: str,
        from_assay: str,
        cell_key: str,
        feat_key: str,
        integrated_graph: str | None,
        cluster_key: str,
        fill_by_value: str | None,
        invalidate_cache: bool,
    ) -> dict[str, Any]:
        from networkx import DiGraph, to_pandas_edgelist

        from ...clustering.cluster_tree import CoalesceTree, make_digraph
        from ...clustering.paris import hierarchy_to_dendrogram
        from .paris_persistence import load_hierarchy_group

        cell_data = as_zarr_group(self.zw["cellData"], name="cellData")
        cluster_column = as_zarr_array(cell_data[cluster_key], name=cluster_key)
        raw_cut_ref = cluster_column.attrs.get("source_artifact")
        if not isinstance(raw_cut_ref, dict):
            raise ValueError("Cluster column has no source artifact")
        cut_ref = ArtifactRef.from_dict(raw_cut_ref)
        cut_inputs = inspect_artifact(self.zw, cut_ref).inputs or {}
        raw_graph_ref = cut_inputs.get("connectivity_map")
        expected_graph_input = (
            graph_ref.to_dict() if isinstance(graph_ref, ArtifactRef) else graph_ref
        )
        if raw_graph_ref != expected_graph_input:
            raise ValueError("Cluster cut does not belong to the requested graph")
        raw_hierarchy_ref = cut_inputs.get("cluster_hierarchy")
        if not isinstance(raw_hierarchy_ref, dict):
            raise ValueError("Cluster cut has no hierarchy input")
        hierarchy_ref = ArtifactRef.from_dict(raw_hierarchy_ref)
        hierarchy_group = as_zarr_group(
            self.zw[inspect_artifact(self.zw, hierarchy_ref).path],
            name=hierarchy_ref.artifact_id,
        )
        hierarchy, _plateau = load_hierarchy_group(
            hierarchy_group,
            hierarchy_ref.artifact_id,
        )
        dendrogram_plan = plan_artifact(
            self.zw,
            scope=hierarchy_ref.scope,
            assay=hierarchy_ref.assay,
            kind="dendrogram",
            operation="materialize_paris_dendrogram",
            parameters={"compatibility": True},
            inputs={"cluster_hierarchy": hierarchy_ref},
            execution_options={},
            invalidate_cache=invalidate_cache,
            required_arrays=(ArrayRequirement("data", dtype_kind="f"),),
        )
        if dendrogram_plan.reused:
            dendrogram_group = reused_artifact_group(self.zw, dendrogram_plan)
        else:
            dendrogram = hierarchy_to_dendrogram(hierarchy, compatibility=True)
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
        dendrogram = np.asarray(as_zarr_array(dendrogram_group["data"], name="data")[:])
        cut_group = as_zarr_group(
            self.zw[artifact_path(cut_ref)],
            name=artifact_path(cut_ref),
        )
        clusters = np.asarray(as_zarr_array(cut_group["labels"], name="labels")[:])
        coalesced_plan = plan_artifact(
            self.zw,
            scope=cut_ref.scope,
            assay=cut_ref.assay,
            kind="coalesced_tree",
            operation="coalesce_cluster_tree",
            parameters={},
            inputs={
                "dendrogram": dendrogram_plan.ref,
                "cluster_cut": cut_ref,
            },
            execution_options={"cluster_key": cluster_key},
            invalidate_cache=invalidate_cache,
            required_arrays=(
                ArrayRequirement("edgelist"),
                ArrayRequirement("nodelist"),
                ArrayRequirement("partition_id"),
            ),
        )
        if coalesced_plan.reused:
            coalesced_group = reused_artifact_group(self.zw, coalesced_plan)
            subgraph = DiGraph()
            subgraph.add_edges_from(
                np.asarray(
                    as_zarr_array(
                        coalesced_group["edgelist"],
                        name="edgelist",
                    )[:]
                )
            )
            nodelist = np.asarray(
                as_zarr_array(coalesced_group["nodelist"], name="nodelist")[:]
            )
            partition_ids = np.asarray(
                as_zarr_array(
                    coalesced_group["partition_id"],
                    name="partition_id",
                )[:]
            )
            cluster_labels = {str(value): value for value in set(clusters)}
            for node_data, partition_id in zip(
                nodelist,
                partition_ids,
                strict=True,
            ):
                node = int(node_data[0])
                subgraph.nodes[node]["nleaves"] = int(node_data[1])
                if str(partition_id) != "-1":
                    subgraph.nodes[node]["partition_id"] = cluster_labels.get(
                        str(partition_id),
                        partition_id,
                    )
        else:
            subgraph = CoalesceTree(make_digraph(dendrogram), clusters)
            edge_list = to_pandas_edgelist(subgraph).values
            coalesced_group = start_artifact(self.zw, coalesced_plan)
            edge_array = create_zarr_dataset(
                coalesced_group,
                "edgelist",
                (100000,),
                "u8",
                edge_list.shape,
            )
            edge_array[:] = edge_list
            node_list = []
            partition_id_values = []
            for node in subgraph.nodes():
                node_data = subgraph.nodes[node]
                node_list.append((node, node_data["nleaves"]))
                partition_id_values.append(str(node_data.get("partition_id", -1)))
            node_values = np.asarray(node_list)
            node_array = create_zarr_dataset(
                coalesced_group,
                "nodelist",
                (100000,),
                node_values.dtype,
                node_values.shape,
            )
            node_array[:] = node_values
            partition_array = create_zarr_dataset(
                coalesced_group,
                "partition_id",
                (100000,),
                str,
                (len(partition_id_values),),
            )
            partition_array[:] = partition_id_values
            finish_artifact(coalesced_group, coalesced_plan)
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
            "integrated_graph": integrated_graph,
            "cluster_key": cluster_key,
            "coalesced_location": inspect_artifact(
                self.zw,
                coalesced_plan.ref,
            ).path,
        }

    def _prepare_cluster_tree(
        self,
        *,
        from_assay: str | None = None,
        cell_key: str | None = None,
        feat_key: str | None = None,
        integrated_graph: str | None = None,
        cluster_key: str | None = None,
        fill_by_value: str | None = None,
        invalidate_cache: bool = False,
    ) -> dict[str, Any]:
        from networkx import DiGraph, to_pandas_edgelist

        from ...clustering.cluster_tree import CoalesceTree, make_digraph
        from ...utils.arrays import array_digest
        from .paris_persistence import resolve_compatibility_dendrogram

        from_assay, cell_key, feat_key = self._get_latest_keys(
            from_assay, cell_key, feat_key
        )
        if cluster_key is None:
            raise ValueError(
                "ERROR: Please provide a value for `cluster_key` parameter"
            )

        if integrated_graph is None:
            graph_loc = self.get_latest_graph_loc(from_assay, cell_key, feat_key)
        else:
            graph_loc = self._resolve_integrated_graph_path(integrated_graph)
            if graph_loc is None:
                raise KeyError(
                    f"An integrated graph with label {integrated_graph!r} does not exist"
                )
        cell_data = as_zarr_group(self.zw["cellData"], name="cellData")
        cluster_column = as_zarr_array(cell_data[cluster_key], name=cluster_key)
        raw_cut_ref = cluster_column.attrs.get("source_artifact")
        if integrated_graph is None and isinstance(raw_cut_ref, dict):
            cut_ref = ArtifactRef.from_dict(raw_cut_ref)
            if cut_ref.kind == "cluster_cut":
                cut_inputs = inspect_artifact(self.zw, cut_ref).inputs or {}
                raw_graph_ref = cut_inputs.get("connectivity_map")
                if isinstance(raw_graph_ref, dict):
                    try:
                        selected_graph_ref = ArtifactRef.from_dict(raw_graph_ref)
                    except (KeyError, TypeError, ValueError):
                        pass
                    else:
                        graph_loc = artifact_path(selected_graph_ref)
        try:
            graph_ref: ArtifactRef | dict[str, str] | None = parse_artifact_path(
                graph_loc
            )
        except ValueError:
            if isinstance(raw_cut_ref, dict):
                graph_group = as_zarr_group(self.zw[graph_loc], name=graph_loc)
                graph_ref = {
                    "legacy_graph_fingerprint": fingerprint_stored_arrays(
                        graph_group,
                        ("edges", "weights"),
                    )
                }
            else:
                graph_ref = None
        if isinstance(raw_cut_ref, dict):
            assert graph_ref is not None
            return self._prepare_artifact_cluster_tree(
                graph_ref=graph_ref,
                graph_loc=graph_loc,
                from_assay=from_assay,
                cell_key=cell_key,
                feat_key=feat_key,
                integrated_graph=integrated_graph,
                cluster_key=cluster_key,
                fill_by_value=fill_by_value,
                invalidate_cache=invalidate_cache,
            )
        if isinstance(graph_ref, ArtifactRef):
            raise ValueError("Cluster column has no source artifact for this graph")
        clusters = np.asarray(self.cells.fetch(cluster_key, key=cell_key))
        dendrogram_loc, generation_id = resolve_compatibility_dendrogram(
            self.zw,
            graph_loc,
            self.resources,
        )
        if clusters.dtype.hasobject:
            hashed_clusters = pd.util.hash_pandas_object(
                pd.Series(clusters),
                index=False,
                categorize=True,
            ).to_numpy(dtype=np.uint64)
            cluster_digest = array_digest(hashed_clusters)
        else:
            cluster_digest = array_digest(clusters)
        coalesced_loc = f"{dendrogram_loc}_coalesced_{cluster_digest}"
        cache_hit = False
        if coalesced_loc in self.zw:
            coalesced_group = as_zarr_group(
                self.zw[coalesced_loc],
                name=coalesced_loc,
            )
            cache_hit = (
                coalesced_group.attrs.get("complete") is True
                and coalesced_group.attrs.get("cluster_digest") == cluster_digest
                and coalesced_group.attrs.get("hierarchy_generation_id")
                == (generation_id or "legacy")
            )

        if cache_hit:
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
            for node_data, partition_id in zip(
                nodelist,
                partition_ids,
                strict=True,
            ):
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
            coalesced_group = self.zw.create_group(coalesced_loc, overwrite=True)
            coalesced_group.attrs.update(
                {
                    "complete": False,
                    "cluster_digest": cluster_digest,
                    "hierarchy_generation_id": generation_id or "legacy",
                    "cluster_key": cluster_key,
                }
            )
            store = create_zarr_dataset(
                coalesced_group,
                "edgelist",
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
                coalesced_group,
                "nodelist",
                (100000,),
                node_list_arr.dtype,
                node_list_arr.shape,
            )
            store[:] = node_list_arr

            store = create_zarr_dataset(
                coalesced_group,
                "partition_id",
                (100000,),
                str,
                (len(partition_id_list),),
            )
            store[:] = partition_id_list
            coalesced_group.attrs["complete"] = True

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
            "integrated_graph": integrated_graph,
            "cluster_key": cluster_key,
            "coalesced_location": coalesced_loc,
        }
