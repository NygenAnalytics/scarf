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
    inspect_artifact,
)
from ...storage.artifact_writer import (
    ArrayRequirement,
    finish_artifact,
    plan_artifact,
    reused_artifact_group,
    start_artifact,
)
from ...graph.feature_projection import (
    graph_cell_selection,
    resolve_graph_source_assay,
)
from ...metadata.arguments import (
    MembershipStrengthArguments,
    SmartLabelArguments,
)
from ...metadata.artifacts import (
    plan_cell_data_artifact,
    write_cell_data_artifact,
)
from ...utils.logging import logger
from ...utils.compute import controlled_compute
from ...storage.selections import (
    read_stored_selection_indices,
    validate_stored_selection_integrity,
)

if TYPE_CHECKING:
    from ..mapping_datastore import MappingDatastore as _PresentationOperationsBase
    from ..pipeline_run import PipelineRun
else:
    _PresentationOperationsBase = object


_CELL_LABEL_VALUE_NAMES = {
    "cell_cycle": "phase",
    "cluster_cut": "labels",
}


def _load_cell_label_artifact(
    root: zarr.Group,
    ref: ArtifactRef,
) -> tuple[np.ndarray, ArtifactRef]:
    if not isinstance(ref, ArtifactRef):
        raise TypeError("label input must be an ArtifactRef")
    status = inspect_artifact(root, ref)
    if not status.complete:
        raise ValueError("Label artifact is unavailable or incomplete")
    raw_selection = (status.inputs or {}).get("cell_selection")
    if not isinstance(raw_selection, dict):
        raise ValueError("Label artifact has no cell-selection input")
    selection = ArtifactRef.from_dict(raw_selection)
    validate_stored_selection_integrity(
        root,
        selection,
        kind="cell_selection",
        scope="datastore",
        assay=None,
        table_path="cellData",
    )
    value_name = _CELL_LABEL_VALUE_NAMES.get(ref.kind, "values")
    group = as_zarr_group(root[status.path], name=status.path)
    if value_name not in group:
        raise ValueError(
            f"{ref.kind} artifact has no canonical {value_name!r} label array"
        )
    values = np.asarray(as_zarr_array(group[value_name], name=value_name)[:])
    if values.ndim != 1:
        raise ValueError("Label artifact values must be one-dimensional")
    return values, selection


def _raw_sparse_for_indices(
    assay: Any,
    cell_idx: np.ndarray,
    feat_idx: np.ndarray,
) -> csr_matrix:
    """Materialize one explicitly selected raw matrix in bounded row blocks."""
    selected = assay.rawData[:, feat_idx][cell_idx, :]
    blocks = [
        csr_matrix(block)
        for block in selected.stream_blocks(
            nthreads=assay.nthreads,
            msg=f"Converting {assay.name} raw data to CSR",
        )
    ]
    if blocks:
        return vstack(blocks, format="csr")
    return csr_matrix(
        (len(cell_idx), len(feat_idx)),
        dtype=assay.rawData.dtype,
    )


def _lift_frozen_umap_to_obsm(adata: Any) -> None:
    umap_columns: dict[int, str] = {}
    for column in adata.obs.columns:
        prefix, separator, suffix = str(column).rpartition("_")
        if prefix == "umap" and separator and suffix.isdigit():
            component = int(suffix)
            if component > 0:
                umap_columns[component] = str(column)
    if not umap_columns:
        return
    expected = list(range(1, max(umap_columns) + 1))
    if sorted(umap_columns) != expected:
        raise ValueError("Frozen UMAP fields must be consecutively numbered")
    ordered_columns = [umap_columns[index] for index in expected]
    adata.obsm["X_umap"] = adata.obs[ordered_columns].to_numpy(copy=True)
    adata.obs.drop(columns=ordered_columns, inplace=True)


class _PresentationOperationsMixin(_PresentationOperationsBase):
    def to_anndata(
        self,
        from_assay: str | None = None,
        cell_key: str | None = None,
        layers: dict[str, str] | None = None,
        *,
        run: "PipelineRun | None" = None,
        matrix: Literal["raw", "normed"] = "raw",
        feature_indexes: Sequence[int] | None = None,
        feature_names: Sequence[str] | None = None,
    ) -> Any:
        """Return an assay as an in-memory AnnData object.

        Cell and feature metadata are copied to ``obs`` and ``var``. Without
        ``run``, layout coordinates remain ordinary ``obs`` columns and this
        method does not populate ``obsm``. With ``run``, consecutive frozen
        ``umap_*`` fields are written to ``obsm["X_umap"]`` and removed from
        ``obs``. Cluster and QC labels stay in ``obs``.

        Args:
            from_assay: Name of assay to be used. If no value is provided then the default assay will be used.
            cell_key: Name of column from cell metadata that has boolean values. This is used to subset cells
            layers: A mapping of layer names to assay names. Ex. {'spliced': 'RNA', 'unspliced': 'URNA'}. The raw data
                    from the assays will be stored as sparse arrays in the corresponding layer in anndata.
            run: A completed pipeline run opened from this datastore. When provided,
                 export uses its frozen cell and feature selections and metadata.
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

        if matrix not in ("raw", "normed"):
            raise ValueError("matrix must be either 'raw' or 'normed'")
        if feature_indexes is not None and feature_names is not None:
            raise ValueError("feature_indexes and feature_names are mutually exclusive")

        run_cells = None
        run_features = None
        if run is not None:
            from ..pipeline_run import PipelineRun

            if not isinstance(run, PipelineRun):
                raise TypeError("run must be a PipelineRun")
            if run._owner is not self:
                raise ValueError("run must be opened from this datastore")
            if from_assay is not None or cell_key is not None:
                raise ValueError(
                    "Run-aware export uses the frozen run selection and assay"
                )
            if feature_indexes is not None or feature_names is not None:
                raise ValueError(
                    "Run-aware export uses the frozen run feature selection"
                )
            assay = self._get_assay(run.assay)
            run_cells = run.cells
            run_features = run.features
            cell_idx = np.flatnonzero(run_cells.fetch_all("I")).astype(
                np.int64,
                copy=False,
            )
            feat_idx = np.flatnonzero(run_features.fetch_all("I")).astype(
                np.int64,
                copy=False,
            )
            obs = (
                run_cells.to_pandas_dataframe(run_cells.columns)
                .reset_index(drop=True)
                .set_index("ids")
            )
            var = (
                run_features.to_pandas_dataframe(run_features.columns)
                .rename(columns={"ids": "gene_ids"})
                .set_index("gene_ids")
            )
        else:
            if cell_key is None:
                cell_key = "I"
            assay = self._get_assay(from_assay)

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
                for index, name in enumerate(
                    assay.feats.fetch_all("names").astype(str)
                ):
                    name_positions.setdefault(name, []).append(index)
                missing = [
                    name for name in requested_names if name not in name_positions
                ]
                if missing:
                    raise KeyError("Feature names not found: " + ", ".join(missing))
                ambiguous = [
                    name for name in requested_names if len(name_positions[name]) != 1
                ]
                if ambiguous:
                    raise ValueError(
                        "Feature names are not unique in the assay: "
                        + ", ".join(ambiguous)
                    )
                feat_idx = np.asarray(
                    [name_positions[name][0] for name in requested_names],
                    dtype=np.int64,
                )
            else:
                feat_idx = np.arange(assay.feats.N, dtype=np.int64)

            cell_idx = self.cells.active_index(cell_key)
            obs = (
                self.cells.to_pandas_dataframe(self.cells.columns, key=cell_key)
                .reset_index(drop=True)
                .set_index("ids")
            )
            var = (
                assay.feats.to_pandas_dataframe(assay.feats.columns)
                .iloc[feat_idx]
                .rename(columns={"ids": "gene_ids"})
                .set_index("gene_ids")
            )

        if matrix == "raw":
            if run is None:
                assert cell_key is not None
                x = assay.to_raw_sparse(cell_key)[:, feat_idx].tocsr()
            else:
                x = _raw_sparse_for_indices(assay, cell_idx, feat_idx)
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
            if run_features is None:
                selected_ids = assay.feats.fetch_all("ids").astype(str)[feat_idx]
            else:
                selected_ids = run_features.fetch("ids").astype(str)
            if np.unique(selected_ids).size != selected_ids.size:
                raise ValueError(
                    "Selected feature IDs must be unique when exporting layers"
                )
            for layer, assay_name in layers.items():
                layer_assay = self._get_assay(assay_name)
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
                if run is None:
                    assert cell_key is not None
                    layer_matrix = layer_assay.to_raw_sparse(cell_key)
                    adata.layers[layer] = layer_matrix[:, layer_feat_idx].tocsr()
                else:
                    adata.layers[layer] = _raw_sparse_for_indices(
                        layer_assay,
                        cell_idx,
                        layer_feat_idx,
                    )
        if run is not None:
            _lift_frozen_umap_to_obsm(adata)
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
        clusters: ArtifactRef,
        graph: ArtifactRef,
        *,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Store per-cell cluster membership strength as an artifact.

        For each cell, computes the fraction of KNN neighbors sharing the most
        common cluster label.

        Args:
            clusters: Explicit axis-aligned cluster-label artifact.
            graph: Explicit connectivity-map or integrated-graph artifact.

        Returns:
            Reference to the immutable membership-strength artifact.
        """
        if not isinstance(graph, ArtifactRef):
            raise TypeError("graph must be an ArtifactRef")
        graph_ref = graph
        status = inspect_artifact(self.zw, graph_ref)
        if not status.complete:
            raise ValueError("Graph artifact is unavailable or incomplete")
        loc = status.path
        n_cells, k = self._get_graph_ncells_k(graph_loc=loc)
        selection = graph_cell_selection(self.zw, graph_ref)
        validate_stored_selection_integrity(
            self.zw,
            selection,
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        )
        cluster_values, cluster_selection = _load_cell_label_artifact(
            self.zw,
            clusters,
        )
        if cluster_selection != selection:
            raise ValueError("Cluster labels do not match the graph cell selection")
        if cluster_values.shape != (n_cells,):
            raise ValueError("Cluster labels do not align with graph rows")
        arguments = MembershipStrengthArguments(
            connectivity_map=graph_ref,
            clusters=clusters,
            cell_selection=selection,
            algorithm_version=2,
            decimals=3,
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
            cell_selection=selection,
            arrays={"values": ((n_cells,), "f")},
            invalidate_cache=invalidate_cache,
        )
        if planned.reused:
            return planned.ref
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
        neighbor_clusters = cluster_values[edge_rows[:, :, 1]]
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
        return planned.ref

    def smart_label(
        self,
        to_relabel: ArtifactRef,
        base_label: ArtifactRef,
        *,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Relabel one cell-label artifact using another label artifact.

        Values in artifact A are relabeled from their overlap with artifact B.
        For each unique value in A, the most frequently occurring value in B is
        found. If two or more values in A have maximum overlap with the same
        value in B, then they all get the same label as B along with different
        suffixes like, 'a', 'b', etc. The suffixes are ordered based on where
        the largest fraction of the B label lies. If one label from A takes up
        multiple labels from B then all the labels from B are included, and they
        are delimited by hyphens.

        Args:
            to_relabel: Explicit axis-aligned label artifact to relabel.
            base_label: Explicit axis-aligned base-label artifact.

        Returns:
            Reference to the immutable relabeled-values artifact.
        """
        values_to_relabel, selection = _load_cell_label_artifact(
            self.zw,
            to_relabel,
        )
        base_values, base_selection = _load_cell_label_artifact(
            self.zw,
            base_label,
        )
        if base_selection != selection:
            raise ValueError("Label artifacts must share one cell selection")
        if base_values.shape != values_to_relabel.shape:
            raise ValueError(
                "Label artifacts must have matching one-dimensional shapes"
            )
        arguments = SmartLabelArguments(
            values=to_relabel,
            base_labels=base_label,
            cell_selection=selection,
            algorithm_version=2,
            suffix_style="lowercase_letter",
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
            arrays={"values": (values_to_relabel.shape, None)},
            invalidate_cache=invalidate_cache,
        )
        if planned.reused:
            return planned.ref
        if len(values_to_relabel) == 0:
            write_cell_data_artifact(
                self.zw,
                planned,
                {"values": np.asarray([], dtype=str)},
            )
            return planned.ref

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

        values = np.asarray([new_names[x] for x in values_to_relabel])
        write_cell_data_artifact(
            self.zw,
            planned,
            {"values": values},
        )
        return planned.ref

    def _prepare_artifact_cluster_tree(
        self,
        *,
        graph_ref: ArtifactRef,
        clusters_ref: ArtifactRef,
        from_assay: str,
        fill_by_value: str | None,
        invalidate_cache: bool,
    ) -> dict[str, Any]:
        from networkx import DiGraph, to_pandas_edgelist

        from ...clustering.cluster_tree import CoalesceTree, make_digraph
        from ...clustering.paris import hierarchy_to_dendrogram
        from .paris_persistence import load_hierarchy_group

        if (
            clusters_ref.scope != "assay"
            or clusters_ref.assay != from_assay
            or clusters_ref.kind != "cluster_cut"
        ):
            raise ValueError(
                "clusters must identify an assay-scoped cluster_cut artifact "
                "for the graph assay"
            )
        cut_status = inspect_artifact(self.zw, clusters_ref)
        if not cut_status.complete or cut_status.operation != "cut_paris_hierarchy":
            raise ValueError(
                "clusters must identify a complete Paris cluster-cut artifact"
            )
        cut_inputs = cut_status.inputs or {}
        raw_graph_ref = cut_inputs.get("connectivity_map")
        expected_graph_input = graph_ref.to_dict()
        if raw_graph_ref != expected_graph_input:
            raise ValueError("Cluster cut does not belong to the requested graph")
        raw_hierarchy_ref = cut_inputs.get("cluster_hierarchy")
        if not isinstance(raw_hierarchy_ref, dict):
            raise ValueError("Cluster cut has no hierarchy input")
        hierarchy_ref = ArtifactRef.from_dict(raw_hierarchy_ref)
        hierarchy_status = inspect_artifact(self.zw, hierarchy_ref)
        if (
            not hierarchy_status.complete
            or hierarchy_status.operation != "fit_paris_hierarchy"
            or (hierarchy_status.inputs or {}).get("connectivity_map")
            != expected_graph_input
        ):
            raise ValueError(
                "Cluster cut does not have a complete hierarchy for the requested graph"
            )
        hierarchy_group = as_zarr_group(
            self.zw[hierarchy_status.path],
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
            self.zw[artifact_path(clusters_ref)],
            name=artifact_path(clusters_ref),
        )
        clusters = np.asarray(as_zarr_array(cut_group["labels"], name="labels")[:])
        raw_selection = cut_inputs.get("cell_selection")
        if not isinstance(raw_selection, dict):
            raise ValueError("Cluster cut has no cell-selection input")
        selection_ref = ArtifactRef.from_dict(raw_selection)
        cell_indices = read_stored_selection_indices(
            self.zw,
            selection_ref,
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        )
        if clusters.shape != (len(cell_indices),):
            raise ValueError(
                "Cluster labels do not align with the stored cell selection"
            )
        coalesced_plan = plan_artifact(
            self.zw,
            scope=clusters_ref.scope,
            assay=clusters_ref.assay,
            kind="coalesced_tree",
            operation="coalesce_cluster_tree",
            parameters={},
            inputs={
                "dendrogram": dendrogram_plan.ref,
                "cluster_cut": clusters_ref,
            },
            execution_options={},
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
        color_values = None
        if fill_by_value is not None:
            if fill_by_value in self.cells.columns:
                color_values = np.asarray(self.cells.fetch_all(fill_by_value))[
                    cell_indices
                ]
            else:
                assay = self._get_assay(from_assay)
                feature_indices = assay.feats.get_index_by(
                    [fill_by_value],
                    "names",
                )
                if len(feature_indices) == 0:
                    raise ValueError(
                        f"ERROR: {fill_by_value} not found in {from_assay} assay."
                    )
                if len(feature_indices) > 1:
                    logger.warning(
                        f"Plotting mean of {len(feature_indices)} features because "
                        f"{fill_by_value} is not unique."
                    )
                color_values = controlled_compute(
                    assay.normed(cell_indices, feature_indices).mean(axis=1),
                    self.nthreads,
                ).astype(np.float64)
        return {
            "graph": subgraph,
            "clusters": clusters,
            "color_values": color_values,
            "from_assay": from_assay,
            "graph_ref": graph_ref,
            "clusters_ref": clusters_ref,
            "cell_selection": selection_ref,
            "coalesced_location": inspect_artifact(
                self.zw,
                coalesced_plan.ref,
            ).path,
        }

    def _prepare_cluster_tree(
        self,
        *,
        graph: ArtifactRef,
        clusters: ArtifactRef,
        from_assay: str | None = None,
        fill_by_value: str | None = None,
        invalidate_cache: bool = False,
    ) -> dict[str, Any]:
        """Prepare an artifact-backed cluster tree for one exact graph."""
        if not isinstance(graph, ArtifactRef):
            raise TypeError("graph must be an ArtifactRef")
        if not isinstance(clusters, ArtifactRef):
            raise TypeError("clusters must be an ArtifactRef")
        assay_name = resolve_graph_source_assay(
            self.zw,
            graph,
            from_assay,
            parameter_name="from_assay",
        )
        return self._prepare_artifact_cluster_tree(
            graph_ref=graph,
            clusters_ref=clusters,
            from_assay=assay_name,
            fill_by_value=fill_by_value,
            invalidate_cache=invalidate_cache,
        )
