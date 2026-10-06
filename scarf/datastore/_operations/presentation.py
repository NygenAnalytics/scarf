from collections.abc import Iterator, Mapping, Sequence
from functools import partial
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
import zarr
from scipy.sparse import csr_matrix, vstack

from ...assay.base import raw_csr
from ...features.values import measured_feature_means
from ...storage.types import as_zarr_array, as_zarr_group
from ...storage.arrays import create_zarr_dataset
from ...storage.artifacts import (
    ArtifactRef,
    artifact_path,
    inspect_artifact,
)
from ...storage.artifact_writer import (
    ArrayRequirement,
    artifact_transaction,
    plan_artifact,
    reused_artifact_group,
)
from ...graph.feature_projection import (
    graph_cell_selection,
    resolve_graph_source_assay,
)
from ...graph.kinds import require_graph_kind
from ...metadata.arguments import (
    MEMBERSHIP_STRENGTH_ALGORITHM_VERSION,
    SMART_LABEL_ALGORITHM_VERSION,
    MembershipStrengthArguments,
    SmartLabelArguments,
)
from ...metadata.membership import exported_membership, membership_declaration
from ...metadata.rows import apply_missing_mask, read_metadata_missing_rows
from ...metadata.selection import resolve_complete_labels
from ...metadata.artifacts import (
    plan_cell_data_artifact,
    write_cell_data_artifact,
)
from ...metrics.graph import neighbor_label_agreement
from ...utils.logging import logger
from ...storage.selections import (
    read_stored_selection_indices,
    validate_stored_selection_integrity,
)

if TYPE_CHECKING:
    from ...writers.export import H5adExportPlan, H5adMatrix
    from ..mapping_datastore import MappingDatastore as _PresentationOperationsBase
    from ..pipeline_run import PipelineAxisView, PipelineRun
else:
    _PresentationOperationsBase = object


def _letter_suffix(position: int) -> str:
    """Return a lowercase base-26 suffix: 1 is ``a``, 26 is ``z``, 27 is ``aa``."""
    letters = ""
    while position > 0:
        position, remainder = divmod(position - 1, 26)
        letters = chr(ord("a") + remainder) + letters
    return letters


def _frozen_umap_fields(columns: Sequence[str]) -> list[str]:
    """Return a run's frozen ``umap_<k>`` fields in component order.

    A run export moves these fields from ``obs`` to ``obsm["X_umap"]``.
    Fields whose component is not positive stay in ``obs``.
    """
    umap_columns: dict[int, str] = {}
    for column in columns:
        prefix, separator, suffix = str(column).rpartition("_")
        if prefix == "umap" and separator and suffix.isdigit():
            component = int(suffix)
            if component > 0:
                umap_columns[component] = str(column)
    if not umap_columns:
        return []
    expected = list(range(1, max(umap_columns) + 1))
    if sorted(umap_columns) != expected:
        raise ValueError("Frozen UMAP fields must be consecutively numbered")
    return [umap_columns[index] for index in expected]


def _frozen_field(
    view: "PipelineAxisView",
    column: str,
    rows: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Read a frozen run field at its selected rows, or at ``rows`` of them."""
    values, missing = view._selected_field(column)
    if rows is None:
        return values, missing
    return values[rows], None if missing is None else missing[rows]


def _frozen_coordinates(
    view: "PipelineAxisView",
    columns: tuple[str, ...],
) -> np.ndarray:
    """Stack frozen coordinate fields into a cells-by-components array."""
    return np.column_stack(
        [apply_missing_mask(*view._selected_field(column)) for column in columns]
    )


def _raw_count_blocks(
    assay: Any,
    cell_idx: np.ndarray,
    feat_idx: np.ndarray,
) -> Iterator[np.ndarray]:
    """Stream the raw counts of the selected cells and features by row block.

    The export's conversion of each block to CSR is charged as resident.
    """
    from ...writers.export import h5ad_conversion_bytes, largest_block_rows

    selected = assay.rawData[:, feat_idx][cell_idx, :]
    blocks: Iterator[np.ndarray] = selected._stream_blocks(
        nthreads=assay.nthreads,
        msg=f"Exporting {assay.name} raw counts",
        prefetch=None,
        row_mask=None,
        resident_bytes=h5ad_conversion_bytes(
            selected.dtype, selected.shape[1], largest_block_rows(selected)
        ),
    )
    return blocks


def _raw_count_matrix(
    assay: Any,
    cell_idx: np.ndarray,
    feat_idx: np.ndarray,
) -> "H5adMatrix":
    """Plan the export of the raw counts of selected cells and features."""
    from ...writers.export import H5adMatrix

    return H5adMatrix(
        shape=(len(cell_idx), len(feat_idx)),
        dtype=np.dtype(assay.rawData.dtype),
        blocks=partial(_raw_count_blocks, assay, cell_idx, feat_idx),
    )


def _stored_row_blocks(
    data: zarr.Array,
    nthreads: int,
    resources: Any,
    msg: str,
) -> Iterator[np.ndarray]:
    """Stream a stored dense matrix by row block, as stored.

    The export's conversion of each block to CSR is charged as resident.
    """
    from ...matrix import ChunkedArray
    from ...writers.export import h5ad_conversion_bytes, largest_block_rows

    matrix = ChunkedArray(data, nthreads=nthreads, resources=resources)
    blocks: Iterator[np.ndarray] = matrix._stream_blocks(
        nthreads=None,
        msg=msg,
        prefetch=None,
        row_mask=None,
        resident_bytes=h5ad_conversion_bytes(
            data.dtype, int(data.shape[1]), largest_block_rows(matrix)
        ),
    )
    return blocks


def _require_unique_layer_ids(selected_ids: np.ndarray) -> None:
    if np.unique(selected_ids).size != selected_ids.size:
        raise ValueError("Selected feature IDs must be unique when exporting layers")


def _anndata_from_plan(anndata: Any, plan: "H5adExportPlan") -> Any:
    """Materialize an export plan as an in-memory AnnData object.

    ``obs`` and ``var`` are the tables of ``scarf.writers.export.h5ad_frame``,
    so their categoricals are the ones that the file of the plan stores.
    """
    from ...writers.export import h5ad_frame, materialize_h5ad_matrix

    n_obs, n_vars = plan.x.shape
    adata = anndata(
        materialize_h5ad_matrix(plan.x),
        obs=h5ad_frame(plan.obs_index, plan.obs, n_obs),
        var=h5ad_frame(plan.var_index, plan.var, n_vars),
        uns=membership_declaration(dict(plan.membership)),
    )
    for name, read in plan.obsm.items():
        adata.obsm[name] = read()
    for name, layer in plan.layers.items():
        adata.layers[name] = materialize_h5ad_matrix(layer, f"Layer {name!r}")
    return adata


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
        ``obs``. Cluster and QC labels stay in ``obs``. Rows that a nullable
        column's linked missing mask flags are missing values in ``obs`` and
        ``var``, with or without ``run``.

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
            An AnnData object.

        Raises:
            ImportError: If ``anndata`` is not installed.
            UnmeasuredCellsError: If normed or other-assay data has an unmeasured cell.
        """
        try:
            # noinspection PyPackageRequirements
            from anndata import AnnData  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "DataStore.to_anndata requires anndata. "
                "Install it with: pip install 'scarf[extra]'"
            ) from exc

        if matrix not in ("raw", "normed"):
            raise ValueError("matrix must be either 'raw' or 'normed'")
        if feature_indexes is not None and feature_names is not None:
            raise ValueError("feature_indexes and feature_names are mutually exclusive")

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
            return _anndata_from_plan(
                AnnData,
                self._h5ad_run_plan(run, matrix=matrix, layers=layers),
            )

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
        if matrix == "normed":
            self._require_measured_cells(
                assay.name, cell_key, operation="to_anndata", remedy="cell_key"
            )
        layer_features: dict[str, tuple[Any, np.ndarray]] = {}
        if layers is not None:
            selected_ids = assay.feats.fetch_all("ids").astype(str)[feat_idx]
            _require_unique_layer_ids(selected_ids)
            for layer, assay_name in layers.items():
                layer_features[layer] = self._layer_features(
                    layer, assay_name, selected_ids
                )
                if layer_features[layer][0].name != assay.name:
                    # Only the exported assay declares its membership.
                    self._require_measured_cells(
                        assay_name,
                        cell_key,
                        operation="to_anndata",
                        remedy="cell_key",
                    )
        declared, omitted = exported_membership(self.cells, assay.name)
        obs = (
            self.cells.to_pandas_dataframe(
                [column for column in self.cells.columns if column not in omitted],
                key=cell_key,
            )
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
            x = raw_csr(assay, cell_idx, feat_idx)
        else:
            normed = assay.normed(cell_idx=cell_idx, feat_idx=feat_idx)
            blocks = [csr_matrix(block) for block in normed.stream_blocks()]
            x = (
                vstack(blocks, format="csr")
                if blocks
                else csr_matrix((len(cell_idx), len(feat_idx)))
            )
        adata = AnnData(x, obs=obs, var=var, uns=membership_declaration(declared))
        for layer, (layer_assay, layer_feat_idx) in layer_features.items():
            adata.layers[layer] = raw_csr(layer_assay, cell_idx, layer_feat_idx)
        return adata

    def _layer_features(
        self,
        layer: str,
        assay_name: str,
        selected_ids: np.ndarray,
    ) -> tuple[Any, np.ndarray]:
        """Return a layer's assay and its feature rows of ``selected_ids``."""
        layer_assay = self._get_assay(assay_name)
        positions: dict[str, list[int]] = {}
        for index, feature_id in enumerate(
            layer_assay.feats.fetch_all("ids").astype(str)
        ):
            positions.setdefault(feature_id, []).append(index)
        missing_ids = [
            feature_id for feature_id in selected_ids if feature_id not in positions
        ]
        ambiguous_ids = [
            feature_id
            for feature_id in selected_ids
            if len(positions.get(feature_id, ())) > 1
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
        return layer_assay, np.asarray(
            [positions[feature_id][0] for feature_id in selected_ids],
            dtype=np.int64,
        )

    def _h5ad_run_plan(
        self,
        run: "PipelineRun",
        *,
        matrix: Literal["raw", "normed"],
        layers: Mapping[str, str] | None = None,
    ) -> "H5adExportPlan":
        """Resolve which rows, columns, and values a run export holds.

        ``to_anndata(run=...)`` materializes this plan and
        ``scarf.writers.to_h5ad(..., run=...)`` streams it into a file, so
        the two exports cannot differ. Rows are the run's cells. With
        ``matrix="raw"``, columns are the run's feature universe and values
        its raw counts; with ``"normed"``, columns are the features of the
        run's ``normalized`` artifact, which must be exactly its highly
        variable features, and values are that artifact's stored float32
        values, read row band by row band. Cell and feature fields are the
        run's frozen fields, with masked rows missing, encoded as
        ``"categorical"`` columns: the plan decides which text fields are
        categoricals and in what order their categories are, for the file
        and for the object alike. Consecutive ``umap_<k>`` fields become
        ``obsm["X_umap"]``. ``layers`` maps layer names to assays whose raw
        counts are aligned to the exported features by ID.
        """
        from ...writers.export import H5adColumn, H5adExportPlan, H5adMatrix
        from ..pipeline_run import PipelineRun

        if matrix not in ("raw", "normed"):
            raise ValueError("matrix must be either 'raw' or 'normed'")
        if not isinstance(run, PipelineRun):
            raise TypeError("run must be a PipelineRun")
        if run._owner is not self:
            raise ValueError("run must be opened from this datastore")
        assay = self._get_assay(run.assay)
        cells = run.cells
        features = run.features
        cell_idx = np.flatnonzero(cells.fetch_all("I")).astype(np.int64, copy=False)
        universe = np.asarray(features.fetch_all("I"), dtype=bool)
        # Exported rows among the selected rows of the feature view.
        feature_rows: np.ndarray | None = None
        if matrix == "raw":
            universe_idx = np.flatnonzero(universe).astype(np.int64, copy=False)
            x = _raw_count_matrix(assay, cell_idx, universe_idx)
        else:
            x, normalized_features = self._run_normalized_values(run)
            if x.shape[0] != len(cell_idx):
                raise ValueError(
                    f"The normalized artifact of pipeline run {run.run_id} does "
                    "not cover the run's cells"
                )
            if np.any(normalized_features & ~universe):
                raise ValueError(
                    f"The normalized artifact of pipeline run {run.run_id} holds "
                    "features outside the run's feature universe"
                )
            feature_rows = np.flatnonzero(normalized_features[universe])

        def frozen(
            view: "PipelineAxisView",
            column: str,
            name: str | None = None,
            rows: np.ndarray | None = None,
        ) -> H5adColumn:
            return H5adColumn(
                column if name is None else name,
                partial(_frozen_field, view, column, rows),
                "categorical",
            )

        cell_fields = [column for column in cells.columns if column != "ids"]
        umap = _frozen_umap_fields(cell_fields)
        plan_layers: dict[str, H5adMatrix] = {}
        if layers is not None:
            selected_ids = np.asarray(_frozen_field(features, "ids", feature_rows)[0])
            selected_ids = selected_ids.astype(str)
            _require_unique_layer_ids(selected_ids)
            for layer, assay_name in layers.items():
                layer_assay, layer_feat_idx = self._layer_features(
                    layer, assay_name, selected_ids
                )
                if layer_assay.name != assay.name:
                    # A run export declares no membership, and the run checked
                    # only its own assay over its cells.
                    self._require_measured_cells(
                        assay_name, cell_idx, operation="to_anndata", remedy="export"
                    )
                plan_layers[layer] = _raw_count_matrix(
                    layer_assay, cell_idx, layer_feat_idx
                )
        return H5adExportPlan(
            x=x,
            obs_index=frozen(cells, "ids"),
            obs=tuple(
                frozen(cells, column) for column in cell_fields if column not in umap
            ),
            var_index=frozen(features, "ids", "gene_ids", feature_rows),
            var=tuple(
                frozen(features, column, rows=feature_rows)
                for column in features.columns
                if column != "ids"
            ),
            obsm=(
                {"X_umap": partial(_frozen_coordinates, cells, tuple(umap))}
                if umap
                else {}
            ),
            layers=plan_layers,
        )

    def _run_normalized_values(
        self,
        run: "PipelineRun",
    ) -> tuple["H5adMatrix", np.ndarray]:
        """Plan the export of a run's normalized values.

        Returns the values of the run's ``normalized`` artifact, streamed in
        row bands as stored, and the mask of their features over the assay.
        The artifact's cell selection must be the run's cell selection, and
        its feature selection the run's highly variable features.
        """
        from ...assay.normalization import load_normalized_inputs
        from ...writers.export import H5adMatrix

        if "normalized" not in run:
            raise ValueError(
                f"Pipeline run {run.run_id} has no normalized output, so it has "
                "no normalized values to export; export raw counts with "
                "matrix='raw'"
            )
        normalized = run["normalized"]
        if (
            normalized.kind != "normalized"
            or normalized.scope != "assay"
            or normalized.assay != run.assay
        ):
            raise ValueError(
                f"The normalized output of pipeline run {run.run_id} is not a "
                f"normalized artifact of assay {run.assay!r}"
            )
        group, selections = load_normalized_inputs(self.zw, normalized)
        if selections.cells.ref != run.cells._selection_ref:
            raise ValueError(
                f"The normalized artifact of pipeline run {run.run_id} does not "
                "cover the run's cells"
            )
        if (
            "highly_variable_features" not in run
            or selections.features != run["highly_variable_features"]
        ):
            raise ValueError(
                f"The normalized artifact of pipeline run {run.run_id} does not "
                "cover the run's highly variable features"
            )
        data = as_zarr_array(group["data"], name="data")
        values = H5adMatrix(
            shape=(int(data.shape[0]), int(data.shape[1])),
            dtype=np.dtype(np.float32),
            blocks=partial(
                _stored_row_blocks,
                data,
                self.nthreads,
                self.resources,
                f"Exporting {run.assay} normalized values",
            ),
        )
        return values, np.asarray(selections.featureMask, dtype=bool)

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

        For each cell, computes the fraction of KNN neighbors sharing its own
        cluster label.

        Args:
            clusters: Explicit axis-aligned cluster-label artifact over the
                cell selection of ``graph``, with a label for every cell.
                Labels that its linked missing mask flags raise
                ``ValueError``.
            graph: Explicit connectivity-map or integrated-graph artifact.

        Returns:
            Reference to the immutable membership-strength artifact.

        Raises:
            ValueError: If ``graph`` is not a connectivity map or integrated
                graph.
            PermissionError: If no matching result exists and the store is
                not opened with ``zarr_mode='r+'``.
        """
        if not isinstance(graph, ArtifactRef):
            raise TypeError("graph must be an ArtifactRef")
        require_graph_kind(graph)
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
        resolved_clusters = resolve_complete_labels(
            self.zw,
            clusters,
            name="clusters",
            remedy=(
                "Freeze the labels over the graph's cell selection with "
                "snapshot_cluster_labels(clusters, cell_selection=...). If some "
                "graph cells have no label, first build the graph over "
                "select_cells(clusters, include=[...])"
            ),
        )
        cluster_values = resolved_clusters.values
        if resolved_clusters.source_cell_selection != selection:
            raise ValueError("Cluster labels do not match the graph cell selection")
        arguments = MembershipStrengthArguments(
            connectivity_map=graph_ref,
            clusters=clusters,
            cell_selection=selection,
            algorithm_version=MEMBERSHIP_STRENGTH_ALGORITHM_VERSION,
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
        self._require_writable("calc_membership_strength")
        graph_grp = as_zarr_group(self.zw[loc], name=loc)
        edges = as_zarr_array(graph_grp["edges"], name="edges")
        if tuple(edges.shape) != (n_cells * k, 2):
            raise ValueError(
                "Graph edges do not match the stored cell and k dimensions"
            )
        if k < 1:
            raise ValueError("Graph must record at least one neighbour per cell")
        # NaN labels share one integer code, so cells labelled NaN agree.
        cluster_codes, _uniques = pd.factorize(cluster_values, use_na_sentinel=False)
        values = neighbor_label_agreement(edges, cluster_codes, k=k)
        write_cell_data_artifact(
            self.zw,
            planned,
            {"values": values.round(arguments.decimals)},
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
        suffixes: 'a' to 'z', then 'aa', 'ab', and so on. The suffixes are
        ordered based on where the largest fraction of the B label lies. If one
        label from A takes up multiple labels from B then all the labels from B
        are included, and they are delimited by hyphens. Both artifacts need
        one cell selection and a label for every cell; labels that a linked
        missing mask flags raise ``ValueError``.

        Args:
            to_relabel: Explicit axis-aligned label artifact to relabel.
            base_label: Explicit axis-aligned base-label artifact.

        Returns:
            Reference to the immutable relabeled-values artifact.

        Raises:
            ValueError: If two labels of A would receive the same new name,
                which hyphen-joined base labels can cause.
            PermissionError: If no matching result exists and the store is
                not opened with ``zarr_mode='r+'``.
        """
        remedy = (
            "Select the cells labelled in both artifacts with select_cells(..., "
            "include=[...]) and freeze both over that selection with "
            "snapshot_cluster_labels(..., cell_selection=...)"
        )
        relabelled = resolve_complete_labels(
            self.zw, to_relabel, name="to_relabel", remedy=remedy
        )
        base = resolve_complete_labels(
            self.zw, base_label, name="base_label", remedy=remedy
        )
        values_to_relabel = relabelled.values
        base_values = base.values
        selection = relabelled.source_cell_selection
        if base.source_cell_selection != selection:
            raise ValueError("Label artifacts must share one cell selection")
        arguments = SmartLabelArguments(
            values=to_relabel,
            base_labels=base_label,
            cell_selection=selection,
            algorithm_version=SMART_LABEL_ALGORITHM_VERSION,
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
        self._require_writable("smart_label")
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
        bases: dict[Any, str] = {}
        suffixes: dict[Any, str] = {}
        for i in sorted(idxmax.unique()):
            j = normed_frac[idxmax[idxmax == i].index].loc[i]
            j = j.sort_values(ascending=False).index
            for n, k in enumerate(j, start=1):
                bases[k] = str(i)
                suffixes[k] = _letter_suffix(n)

        missing_vals = df.index.difference(
            pd.Index(idxmax.unique()),
            sort=False,
        ).tolist()
        if len(missing_vals) > 0:
            miss_idxmax = df.loc[missing_vals].idxmax(axis=1).to_dict()
            for k, v in miss_idxmax.items():
                bases[v] = f"{bases[v]}-{k}"

        new_names = {label: bases[label] + suffixes[label] for label in bases}
        labels_by_name: dict[str, list[Any]] = {}
        for label, name in new_names.items():
            labels_by_name.setdefault(name, []).append(label)
        for name, labels in labels_by_name.items():
            if len(labels) > 1:
                raise ValueError(
                    f"smart_label would name labels {labels!r} alike as {name!r}. "
                    "Rename the base labels so that hyphen-joined names stay "
                    "distinct."
                )

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
        from .paris_persistence import (
            load_hierarchy_group,
            plan_paris_dendrogram,
            write_paris_dendrogram,
        )

        if clusters_ref.kind != "cluster_cut":
            raise ValueError("clusters must identify a cluster_cut artifact")
        # A cut shares its graph's scope and assay; integrated graphs are
        # datastore-scoped, and from_assay only supplies the fill values.
        if (
            clusters_ref.scope != graph_ref.scope
            or clusters_ref.assay != graph_ref.assay
        ):
            raise ValueError("Cluster cut does not belong to the requested graph")
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
        dendrogram_plan = plan_paris_dendrogram(
            self.zw,
            hierarchy_ref,
            invalidate_cache=invalidate_cache,
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
        # A read-only store computes a cache miss in memory and persists nothing.
        persist = not self.zw.read_only
        if coalesced_plan.reused:
            coalesced_group = reused_artifact_group(self.zw, coalesced_plan)
            nodelist = np.asarray(
                as_zarr_array(coalesced_group["nodelist"], name="nodelist")[:]
            )
            partition_ids = np.asarray(
                as_zarr_array(
                    coalesced_group["partition_id"],
                    name="partition_id",
                )[:]
            )
            subgraph = DiGraph()
            # Add nodes first so a single-node tree without edges is complete.
            subgraph.add_nodes_from(int(node) for node in nodelist[:, 0])
            subgraph.add_edges_from(
                np.asarray(
                    as_zarr_array(
                        coalesced_group["edgelist"],
                        name="edgelist",
                    )[:]
                )
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
            if dendrogram_plan.reused:
                dendrogram_group = reused_artifact_group(self.zw, dendrogram_plan)
                dendrogram = np.asarray(
                    as_zarr_array(dendrogram_group["data"], name="data")[:]
                )
            else:
                hierarchy_group = as_zarr_group(
                    self.zw[hierarchy_status.path],
                    name=hierarchy_ref.artifact_id,
                )
                hierarchy, _plateau = load_hierarchy_group(
                    hierarchy_group,
                    hierarchy_ref.artifact_id,
                )
                dendrogram = hierarchy_to_dendrogram(hierarchy, compatibility=True)
                if persist:
                    write_paris_dendrogram(self.zw, dendrogram_plan, dendrogram)
            subgraph = CoalesceTree(make_digraph(dendrogram), clusters)
            if persist:
                edge_list = to_pandas_edgelist(subgraph).values
                node_list = []
                partition_id_values = []
                for node in subgraph.nodes():
                    node_data = subgraph.nodes[node]
                    node_list.append((node, node_data["nleaves"]))
                    partition_id_values.append(str(node_data.get("partition_id", -1)))
                node_values = np.asarray(node_list)
                with artifact_transaction(self.zw, coalesced_plan) as coalesced_group:
                    edge_array = create_zarr_dataset(
                        coalesced_group,
                        "edgelist",
                        (100000,),
                        "u8",
                        edge_list.shape,
                    )
                    edge_array[:] = edge_list
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
        color_values = None
        color_missing = None
        if fill_by_value is not None:
            if fill_by_value in self.cells.columns:
                color_values = np.asarray(self.cells.fetch_all(fill_by_value))[
                    cell_indices
                ]
                color_missing = read_metadata_missing_rows(
                    self.cells,
                    fill_by_value,
                    cell_indices,
                )
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
                # A cell that the assay did not measure has no fill value, and
                # only measured cells are normalized, as display reads do.
                color_values = measured_feature_means(
                    assay, feature_indices, cell_indices, nthreads=self.nthreads
                )
        return {
            "graph": subgraph,
            "clusters": clusters,
            "color_values": color_values,
            "color_missing": color_missing,
            "from_assay": from_assay,
            "graph_ref": graph_ref,
            "clusters_ref": clusters_ref,
            "cell_selection": selection_ref,
            "coalesced_location": (
                artifact_path(coalesced_plan.ref)
                if coalesced_plan.reused or persist
                else None
            ),
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
