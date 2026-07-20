from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix

from ...assay import Assay
from ...storage.types import as_zarr_array, as_zarr_group
from ...storage.arrays import create_zarr_dataset
from ...trajectory.feature_dynamics import (
    scatter_feature_clusters as _scatter_feature_clusters_impl,
    validate_pseudotime_regressor,
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


class _TrajectoryOperationsMixin(_TrajectoryOperationsBase):
    def get_imputed(
        self,
        from_assay: str | None = None,
        cell_key: str | None = None,
        feature_name: str | None = None,
        feat_key: str | None = None,
        t: int = 2,
        cache_operator: bool = True,
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

        from ...neighbors.diffusion import diffusion_operator

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

        graph_loc = self._get_latest_graph_loc(from_assay, cell_key, feat_key)
        magic_loc = f"{graph_loc}/magic_{t}"
        if magic_loc in self.zw:
            logger.info("Using existing MAGIC diffusion operator")
            if self._cachedMagicOperatorLoc == magic_loc:
                diff_op = cast(coo_matrix, self._cachedMagicOperator)
            else:
                n_cells, _ = self._get_graph_ncells_k(graph_loc)
                store = as_zarr_group(self.zw[magic_loc], name=magic_loc)
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
                if cache_operator:
                    self._cachedMagicOperator = diff_op
                    self._cachedMagicOperatorLoc = magic_loc  # type: ignore[assignment]
                else:
                    self._cachedMagicOperator = None
                    self._cachedMagicOperatorLoc = None
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
            store = self.zw.create_group(magic_loc, overwrite=True)
            for i, j in zip(["row", "col", "data"], ["uint32", "uint32", "float32"]):
                zg = create_zarr_dataset(store, i, (1000000,), j, shape)
                zg[:] = getattr(diff_op, i)
            as_zarr_group(self.zw[graph_loc], name=graph_loc).attrs["latest_magic"] = (
                magic_loc
            )
            if cache_operator:
                self._cachedMagicOperator = diff_op
                self._cachedMagicOperatorLoc = magic_loc  # type: ignore[assignment]
            else:
                self._cachedMagicOperator = None
                self._cachedMagicOperatorLoc = None
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

        logger.info("Pseudotime scoring: constructing Laplacian")
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
        output_column = self._col_renamer(from_assay, subset_cell_key, label)
        validity_column = f"{output_column}__valid"

        logger.info("Pseudotime scoring: saving pseudotime")
        self.cells.insert(
            output_column,
            ptime,
            key=subset_cell_key,
            overwrite=True,
        )
        self.cells.insert(
            validity_column,
            retained_mask,
            fill_value=False,
            key=subset_cell_key,
            overwrite=True,
        )
        if not retained_mask.all():
            logger.warning(
                f"Unscored cells contain NaN pseudotime. Use cell key "
                f"'{validity_column}' for downstream analysis"
            )
        return PseudotimeScoreResult(
            pseudotime_key=output_column,
            validity_key=validity_column,
            assay=from_assay,
            graph_cell_key=cell_key,
            result_cell_key=subset_cell_key,
            feature_key=feat_key,
            values=ptime,
            valid=retained_mask,
        )


class _TrajectoryFeatureOperationsMixin(_TrajectoryFeatureOperationsBase):
    def run_pseudotime_marker_search(
        self,
        from_assay: str | None = None,
        cell_key: str | None = None,
        feat_key: str | None = None,
        pseudotime_key: str | None = None,
        min_cells: int = 10,
        gene_batch_size: int = 50,
        **norm_params: Any,
    ) -> PseudotimeMarkerResult:
        """Identify genes correlated with a pseudotime ordering of cells.

        Args:
            from_assay: Name of the assay to use. The default assay is used when omitted.
            cell_key: Boolean cell metadata column selecting cells.
            feat_key: Boolean feature metadata column selecting features.
            pseudotime_key: Numeric cell metadata column containing pseudotime values.
            min_cells: Minimum number of expressing cells required for a feature.
            gene_batch_size: Number of features loaded per batch.
            **norm_params: Extra keyword arguments forwarded to normalized expression.

        Returns:
            Correlation table and the feature metadata keys where it was saved.
        """
        from ...features.markers import find_markers_by_regression

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
        n_cells = len(assay.cells.active_index(cell_key))
        n_feats = len(assay.feats.active_index(feat_key))
        logger.info(
            f"Pseudotime markers: correlating features "
            f"(cells={n_cells}, features={n_feats}, batch_size={gene_batch_size})"
        )
        markers = find_markers_by_regression(
            assay=assay,
            cell_key=cell_key,
            feat_key=feat_key,
            regressor=ptime,
            min_cells=min_cells,
            batch_size=gene_batch_size,
            **norm_params,
        )
        feature_index = assay.feats.active_index(feat_key)
        markers = markers.reindex(feature_index)
        if markers.isna().any(axis=None):
            raise ValueError("Pseudotime marker results are not aligned to feat_key")
        logger.info("Pseudotime markers: saving marker scores")
        correlation_key = f"{cell_key}__{pseudotime_key}__r"
        p_value_key = f"{cell_key}__{pseudotime_key}__p"
        assay.feats.insert(
            correlation_key,
            np.array(markers["r_value"].values),
            key=feat_key,
            overwrite=True,
        )
        assay.feats.insert(
            p_value_key,
            np.array(markers["p_value"].values),
            key=feat_key,
            overwrite=True,
        )
        table = markers.rename_axis("feature_index").reset_index()
        feature_names = np.asarray(assay.feats.fetch_all("names"), dtype=object)
        table.insert(
            1,
            "feature_name",
            feature_names[table["feature_index"].to_numpy(dtype=np.int64)],
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
        batch_size: int = 100,
        ann_params: dict | None = None,
        nan_cluster_value: int = -1,
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
            batch_size: Number of features processed per batch.
            ann_params: Parameters forwarded to the HNSW index.
            nan_cluster_value: Value assigned to features excluded from clustering.
            **norm_params: Extra keyword arguments forwarded to normalized expression.

        Returns:
            Lazy aggregated matrix with aligned feature indices and clusters.
        """
        from ...trajectory.feature_dynamics import knn_clustering

        from_assay, cell_key, _ = self._get_latest_keys(from_assay, cell_key, feat_key)
        if feat_key is None:
            feat_key = "I"
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

        logger.info("Pseudotime modules: aggregating feature profiles")
        df, feat_ids = assay.save_aggregated_ordering(
            cell_key=cell_key,
            feat_key=feat_key,
            ordering_key=pseudotime_key,
            min_exp=min_exp,
            window_size=window_size,
            chunk_size=chunk_size,
            smoothen=smoothen,
            z_scale=z_scale,
            batch_size=batch_size,
            **norm_params,
        )
        if ann_params is None:
            ann_params = {}
        clusts = knn_clustering(
            d_array=df,
            n_neighbours=n_neighbours,
            n_clusters=n_clusters,
            n_threads=self.nthreads,
            ann_params=ann_params,
        )
        cluster_values = _scatter_feature_clusters_impl(
            assay.feats.N,
            feat_ids,
            clusts,
            nan_cluster_value,
        )
        logger.info("Pseudotime modules: saving module labels")
        assay.feats.insert(
            cluster_label,
            cluster_values,
            fill_value=nan_cluster_value,
            overwrite=True,
        )

        location = f"aggregated_{cell_key}_{feat_key}_{pseudotime_key}"
        aggregation_group = as_zarr_group(assay.z[location], name=location)
        cluster_digest = _group_assignment_digest(cluster_values)
        aggregation_group.attrs["cluster_label"] = cluster_label
        aggregation_group.attrs["cluster_digest"] = cluster_digest
        aggregation_group.attrs["nan_cluster_value"] = nan_cluster_value

        for assay_name in self.assay_names:
            grouped_assay = self._get_assay(assay_name)
            if (
                grouped_assay.attrs.get("grouped_from_assay") == assay.name
                and grouped_assay.attrs.get("grouped_group_key") == cluster_label
                and grouped_assay.attrs.get("grouped_group_digest") != cluster_digest
            ):
                logger.warning(
                    f"Grouped assay '{assay_name}' is stale after updating "
                    f"feature groups in '{cluster_label}'. Rerun add_grouped_assay"
                )
        return PseudotimeAggregationResult(
            data=df,
            feature_indices=np.asarray(feat_ids),
            feature_clusters=np.asarray(clusts),
            cluster_key=cluster_label,
            storage_path=str(aggregation_group.path),
            assay=assay.name,
            cell_key=cell_key,
            feature_key=feat_key,
            pseudotime_key=pseudotime_key,
        )
