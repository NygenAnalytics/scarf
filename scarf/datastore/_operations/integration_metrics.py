from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import zarr

from ...graph.encoded_paths import (
    is_integrated_graph_path,
    parse_assay_keys_from_nearest_neighbors_path,
)
from ...graph.distances import validate_distance_provenance
from ...graph.paths import StoredAssayGraph
from ...graph.state import (
    read_assay_state,
    validate_legacy_graph_selection,
    validate_normalized_artifact_selection,
)
from ...storage.artifacts import (
    ArtifactRef,
    group_at,
    inspect_artifact,
    parse_artifact_path,
)
from ...storage.types import as_zarr_array, as_zarr_group
from ...utils.logging import logger

if TYPE_CHECKING:
    from ...metrics import ClusterSeparabilityResult
    from ..mapping_datastore import MappingDatastore as _IntegrationMetricsBase
else:
    _IntegrationMetricsBase = object


class _IntegrationMetricsOperationsMixin(_IntegrationMetricsBase):
    def _keys_from_knn_path(
        self,
        from_assay: str,
        knn_loc: str,
    ) -> tuple[str, str]:
        try:
            ref = parse_artifact_path(knn_loc)
        except ValueError:
            (
                path_assay,
                cell_key,
                feat_key,
            ) = parse_assay_keys_from_nearest_neighbors_path(knn_loc)
            if path_assay != from_assay:
                raise ValueError(
                    f"KNN path belongs to assay {path_assay!r}, not {from_assay!r}"
                )
            validate_legacy_graph_selection(
                self,
                knn_loc,
                from_assay,
                cell_key,
                feat_key,
            )
            return cell_key, feat_key
        if ref.kind != "neighbors":
            raise ValueError(f"Not a neighbors artifact: {knn_loc}")
        if ref.scope != "assay" or ref.assay != from_assay:
            raise ValueError(
                f"KNN artifact belongs to assay {ref.assay!r}, not {from_assay!r}"
            )
        state = read_assay_state(self.zw, from_assay)

        def require_artifact(
            candidate: ArtifactRef,
            expected_kind: str,
        ) -> Any:
            status = inspect_artifact(self.zw, candidate)
            if (
                candidate.kind != expected_kind
                or candidate.scope != "assay"
                or candidate.assay != from_assay
                or not status.exists
                or not status.complete
            ):
                raise ValueError(f"{expected_kind} artifact is incomplete or invalid")
            return status

        def input_ref(status: Any, owner_kind: str, name: str) -> ArtifactRef:
            value = (status.inputs or {}).get(name)
            if not isinstance(value, dict):
                raise ValueError(f"{owner_kind} has no {name!r} input")
            return ArtifactRef.from_dict(value)

        neighbors_status = require_artifact(ref, "neighbors")
        coordinates = input_ref(neighbors_status, "neighbors", "coordinates")
        ann_index = input_ref(neighbors_status, "neighbors", "ann_index")
        ann_status = require_artifact(ann_index, "ann_index")
        ann_coordinates = input_ref(ann_status, "ann_index", "coordinates")
        if ann_coordinates != coordinates:
            raise ValueError(
                "Neighbors and ANN artifacts reference different coordinates"
            )
        if coordinates.kind not in {"reduction", "batch_correction"}:
            raise ValueError("Neighbors coordinates have an invalid artifact kind")
        coordinates_status = require_artifact(coordinates, coordinates.kind)
        reduction = (
            input_ref(coordinates_status, "batch_correction", "reduction")
            if coordinates.kind == "batch_correction"
            else coordinates
        )
        reduction_status = require_artifact(reduction, "reduction")
        normalized = input_ref(reduction_status, "reduction", "normalized")
        normalized_status = require_artifact(normalized, "normalized")
        selected_cell_key: object
        selected_feature_key: object
        if state is not None and state.neighbors == ref:
            selected_cell_key = state.cell_key
            selected_feature_key = state.feat_key
        else:
            normalized_execution = normalized_status.execution_options or {}
            selected_cell_key = normalized_execution.get("cell_key")
            selected_feature_key = normalized_execution.get("feat_key")
        if not isinstance(selected_cell_key, str) or not isinstance(
            selected_feature_key,
            str,
        ):
            raise ValueError("Selection source columns are missing")
        validate_normalized_artifact_selection(
            self.zw,
            normalized,
            selected_cell_key,
            selected_feature_key,
        )
        return selected_cell_key, selected_feature_key

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
            logger.debug(f"Using the latest KNN graph at {resolved_knn_loc}")
        else:
            if knn_loc is None:
                raise ValueError("Please provide values for the KNN graph location.")
            if knn_loc not in self.zw:
                raise ValueError(f"Could not find the knn graph at location: {knn_loc}")
            resolved_knn_loc = knn_loc
            logger.debug(f"Using the KNN graph at {resolved_knn_loc}")

        cell_key, _ = self._keys_from_knn_path(from_assay, resolved_knn_loc)
        validate_distance_provenance(self.zw, resolved_knn_loc)
        knn_grp = as_zarr_group(
            self.zw[resolved_knn_loc],
            name=resolved_knn_loc,
        )
        distances = as_zarr_array(knn_grp["distances"], name="distances")
        indices = as_zarr_array(knn_grp["indices"], name="indices")
        return from_assay, resolved_knn_loc, cell_key, distances, indices

    def metric_lisi(
        self,
        label_columns: Sequence[str],
        use_latest_knn: bool = True,
        from_assay: str | None = None,
        knn_loc: str | None = None,
        perplexity: float = 30,
    ) -> dict[str, np.ndarray]:
        """Calculate Local Inverse Simpson Index (LISI) scores for cell populations.

        LISI measures how well mixed different cell populations are in the local neighborhood
        of each cell. Higher scores indicate better mixing of different populations.

        Args:
            label_columns: Column names from cell metadata containing population labels
            use_latest_knn: Whether to use the most recent KNN graph (default: True)
            from_assay: Name of assay to use if not using latest KNN
            knn_loc: Location of KNN graph if not using latest (default: None)
            perplexity: Effective neighborhood size used by LISI. It is reduced
                with a warning when the graph has fewer than three times this
                many neighbors.

        Returns:
            A mapping from each label column to its per-cell LISI scores.

        Raises:
            ValueError: If KNN inputs, perplexity, or labels are invalid
            KeyError: If label columns not found in cell metadata

        Notes:
            LISI scores are computed for each label column separately.
            Scores near 1 indicate cells grouped with similar labels.
            Higher scores indicate more mixing between different labels.
        """

        if isinstance(label_columns, str):
            raise TypeError("label_columns must be a sequence of column names")
        label_cols = list(label_columns)
        if not label_cols:
            raise ValueError("label_columns must be non-empty")
        if not all(isinstance(column, str) for column in label_cols):
            raise TypeError("label_columns must contain only strings")
        if len(set(label_cols)) != len(label_cols):
            raise ValueError("label_columns contains duplicate names")

        _, _, cell_key, distances, indices = self._load_metric_knn(
            use_latest_knn,
            from_assay,
            knn_loc,
        )
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
        return {
            column: scores
            for column, scores in zip(label_cols, lisi_scores.T, strict=True)
        }

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
            graph_loc = self.get_latest_graph_loc(
                from_assay,
                cell_key,
                feat_key,
            )
        else:
            try:
                explicit_ref = parse_artifact_path(graph_loc)
            except ValueError:
                explicit_ref = None
            if is_integrated_graph_path(
                graph_loc,
                self._integratedGraphsLoc,
            ) or (explicit_ref is not None and explicit_ref.kind == "integrated_graph"):
                raise ValueError(
                    "Integrated graph connectivity is unavailable because the "
                    "graph does not record its cell-key provenance"
                )
            if graph_loc not in self.zw:
                raise ValueError(f"Could not find the graph at location: {graph_loc}")

            stored_graph = self._lookup_stored_graph(graph_loc=graph_loc)
            if not isinstance(stored_graph, StoredAssayGraph):
                raise ValueError("Graph connectivity requires an assay graph")

            path_assay = stored_graph.from_assay
            path_cell_key = stored_graph.cell_key
            path_feat_key = stored_graph.feat_key
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
            logger.debug(
                f"Using the latest knn graph at location: {knn_loc} for assay: {from_assay}"
            )

        else:
            if knn_loc is None:
                raise ValueError("Please provide values for the KNN graph location.")
            if knn_loc not in self.zw:
                raise ValueError(f"Could not find the knn graph at location: {knn_loc}")
            logger.debug(f"Using the KNN graph at {knn_loc}")

        from ...metrics import silhouette_scoring

        cell_key, feat_key_parsed = self._keys_from_knn_path(from_assay, knn_loc)
        ann_obj = self._load_ann_stream(
            from_assay=from_assay,
            cell_key=cell_key,
            feat_key=feat_key_parsed,
            knn_loc=knn_loc,
        )

        validate_distance_provenance(self.zw, knn_loc)
        knn_grp = as_zarr_group(self.zw[knn_loc], name=knn_loc)
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

    def _validate_reduction_cell_selection(
        self,
        reduction: ArtifactRef,
        cell_key: str,
    ) -> None:
        normalized = self._artifact_input_ref(reduction, "normalized", "normalized")
        execution = (
            self._require_complete_artifact(normalized, "normalized").execution_options
            or {}
        )
        selected_cell_key = execution.get("cell_key")
        selected_feat_key = execution.get("feat_key")
        state = (
            read_assay_state(self.zw, reduction.assay)
            if reduction.assay is not None
            else None
        )
        if state is not None and state.normalized == normalized:
            selected_cell_key = state.cell_key
            selected_feat_key = state.feat_key
        if not isinstance(selected_cell_key, str) or not isinstance(
            selected_feat_key,
            str,
        ):
            raise ValueError("Reduction selection source columns are missing")
        if cell_key != selected_cell_key:
            raise ValueError(
                f"cell_key {cell_key!r} does not match the cell selection "
                f"{selected_cell_key!r} used to build the reduction"
            )
        validate_normalized_artifact_selection(
            self.zw,
            normalized,
            selected_cell_key,
            selected_feat_key,
        )

    def metric_cluster_separability(
        self,
        pca: ArtifactRef,
        cluster_columns: Sequence[str],
        *,
        cell_key: str = "I",
        n_folds: int = 5,
        max_sample_cells: int = 50_000,
        max_silhouette_cells: int = 10_000,
        random_seed: int = 4444,
        svm_c: float = 1.0,
        svm_max_iter: int = 10_000,
    ) -> "ClusterSeparabilityResult":
        """Evaluate cluster-label separability in PCA coordinates."""
        from ...metrics import evaluate_cluster_separability

        if isinstance(cluster_columns, str):
            raise TypeError("cluster_columns must be a sequence of column names")
        columns = list(cluster_columns)
        if not columns:
            raise ValueError("cluster_columns must be non-empty")
        if not all(isinstance(column, str) and column for column in columns):
            raise TypeError("cluster_columns must contain non-empty strings")
        if len(set(columns)) != len(columns):
            raise ValueError("cluster_columns contains duplicate names")

        status = self._require_complete_artifact(pca, "reduction")
        if status.operation != "run_pca":
            raise ValueError("pca must reference a PCA reduction artifact")
        self._validate_reduction_cell_selection(pca, cell_key)
        group = group_at(self.zw, status.path)
        if "data" not in group:
            raise ValueError("PCA reduction coordinates are missing")
        coordinates = as_zarr_array(group["data"], name="PCA coordinates")
        clusterings = {
            column: np.asarray(self.cells.fetch(column, key=cell_key))
            for column in columns
        }
        for column, labels in clusterings.items():
            if len(labels) != coordinates.shape[0]:
                raise ValueError(
                    f"Cluster column {column!r} does not align with PCA rows"
                )

        return evaluate_cluster_separability(
            coordinates,
            clusterings,
            n_folds=n_folds,
            max_sample_cells=max_sample_cells,
            max_silhouette_cells=max_silhouette_cells,
            random_seed=random_seed,
            svm_c=svm_c,
            svm_max_iter=svm_max_iter,
        )

    def metric_label_concordance(
        self,
        label_columns: Sequence[str],
        metric: Literal["ari", "nmi"] = "ari",
        *,
        cell_key: str = "I",
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
            cell_key: Boolean cell metadata column selecting rows to compare.

        Returns:
            Agreement between the two partitions. ARI ranges from -1 to 1 and
            NMI from 0 to 1, with higher values meaning stronger agreement.

        Raises:
            ValueError: If the number of columns or the metric name is invalid.
        """
        from ...metrics import label_concordance_score

        label_values = [
            np.asarray(self.cells.fetch(column, key=cell_key))
            for column in label_columns
        ]
        return label_concordance_score(label_values, metric)

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
            label_columns=[label_colname],
            use_latest_knn=use_latest_knn,
            from_assay=from_assay,
            knn_loc=resolved_knn_loc,
            perplexity=perplexity,
        )

        cell_key, _ = self._keys_from_knn_path(
            from_assay,
            resolved_knn_loc,
        )
        batch_labels = self.cells.fetch(label_colname, key=cell_key)
        return lisi_batch_mixing_score(lisi_result[label_colname], batch_labels)
