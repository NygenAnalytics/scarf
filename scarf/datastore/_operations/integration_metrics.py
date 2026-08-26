from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import zarr

from ...graph.distances import validate_distance_provenance
from ...graph.errors import IncompatibleAnalysisStateError
from ...graph.feature_projection import resolve_native_graph_inputs
from ...graph.state import (
    read_assay_state,
    read_assay_state_document,
    resolve_graph_selection,
    validate_cell_selection_artifact,
)
from ...storage.artifacts import (
    ArtifactRef,
    group_at,
    inspect_artifact,
)
from ...storage.errors import ArtifactResolutionError
from ...storage.feature_selection import resolve_feature_selection
from ...storage.types import as_zarr_array, as_zarr_group

if TYPE_CHECKING:
    from ...metrics import ClusterSeparabilityResult
    from ..mapping_datastore import MappingDatastore as _IntegrationMetricsBase
else:
    _IntegrationMetricsBase = object


class _IntegrationMetricsOperationsMixin(_IntegrationMetricsBase):
    def _resolve_metric_neighbors(
        self,
        neighbors: ArtifactRef | None,
        *,
        from_assay: str | None,
        cell_key: str | None,
    ) -> tuple[ArtifactRef, str, str]:
        explicit_neighbors = neighbors is not None
        if neighbors is None:
            assay = from_assay or self._load_default_assay()
            state = read_assay_state(self.zw, assay)
            if state is None or state.neighbors is None:
                raise ArtifactResolutionError(
                    f"Assay {assay!r} has no current neighbors artifact",
                    code="missing_current_neighbors",
                    context={"assay": assay},
                )
            neighbors = state.neighbors
        elif not isinstance(neighbors, ArtifactRef):
            raise TypeError("neighbors must be an artifact reference")
        else:
            assay = neighbors.assay or ""
            if from_assay is not None and from_assay != assay:
                raise ArtifactResolutionError(
                    "neighbors belongs to a different assay",
                    code="wrong_assay",
                    context={
                        "assay": assay,
                        "expected_assay": from_assay,
                        "artifact_id": neighbors.artifact_id,
                    },
                )
        if neighbors.kind != "neighbors":
            raise ArtifactResolutionError(
                "neighbors must reference a neighbors artifact",
                code="wrong_kind",
                context={
                    "assay": neighbors.assay,
                    "artifact_id": neighbors.artifact_id,
                    "actual_kind": neighbors.kind,
                    "expected_kind": "neighbors",
                },
            )
        if neighbors.scope != "assay":
            raise ArtifactResolutionError(
                "neighbors must be assay-scoped",
                code="wrong_scope",
                context={
                    "artifact_id": neighbors.artifact_id,
                    "actual_scope": neighbors.scope,
                    "expected_scope": "assay",
                },
            )
        if neighbors.assay != assay:
            raise ArtifactResolutionError(
                "neighbors belongs to a different assay",
                code="wrong_assay",
                context={
                    "assay": neighbors.assay,
                    "expected_assay": assay,
                    "artifact_id": neighbors.artifact_id,
                },
            )
        if explicit_neighbors:
            read_assay_state_document(self.zw, assay)
        lineage = resolve_native_graph_inputs(self.zw, neighbors)
        selection_status = inspect_artifact(self.zw, lineage.cell_selection)
        source_column = (selection_status.execution_options or {}).get("source_column")
        if not isinstance(source_column, str) or not source_column:
            raise ArtifactResolutionError(
                "Neighbor cell selection has no source column",
                code="corrupt_payload",
                context={
                    "assay": assay,
                    "artifact_id": lineage.cell_selection.artifact_id,
                },
            )
        validate_cell_selection_artifact(
            self.zw,
            lineage.cell_selection,
            source_column,
        )
        if cell_key is not None and cell_key != source_column:
            raise ArtifactResolutionError(
                "cell_key does not match the neighbors cell selection",
                code="row_mismatch",
                context={
                    "assay": assay,
                    "cell_key": cell_key,
                    "expected_cell_key": source_column,
                },
            )
        return neighbors, assay, source_column

    def _load_metric_knn(
        self,
        neighbors: ArtifactRef | None,
        *,
        from_assay: str | None,
        cell_key: str | None,
    ) -> tuple[ArtifactRef, str, str, zarr.Array, zarr.Array]:
        ref, assay, selected_cell_key = self._resolve_metric_neighbors(
            neighbors,
            from_assay=from_assay,
            cell_key=cell_key,
        )
        status = inspect_artifact(self.zw, ref)
        validate_distance_provenance(self.zw, ref)
        knn_grp = as_zarr_group(
            self.zw[status.path],
            name=status.path,
        )
        distances = as_zarr_array(knn_grp["distances"], name="distances")
        indices = as_zarr_array(knn_grp["indices"], name="indices")
        return ref, assay, selected_cell_key, distances, indices

    def metric_lisi(
        self,
        label_columns: Sequence[str],
        neighbors: ArtifactRef | None = None,
        *,
        from_assay: str | None = None,
        cell_key: str | None = None,
        perplexity: float = 30,
    ) -> dict[str, np.ndarray]:
        """Calculate Local Inverse Simpson Index (LISI) scores for cell populations.

        LISI measures how well mixed different cell populations are in the local neighborhood
        of each cell. Higher scores indicate better mixing of different populations.

        Args:
            label_columns: Column names from cell metadata containing population labels
            neighbors: Neighbor artifact to score. The assay's current
                neighbors are used when omitted.
            from_assay: Assay used to resolve current neighbors.
            cell_key: Optional cell-selection key, validated against neighbors.
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
            neighbors,
            from_assay=from_assay,
            cell_key=cell_key,
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
        neighbors: ArtifactRef | None = None,
        *,
        from_assay: str | None = None,
        cell_key: str | None = None,
        perplexity: float | None = None,
        scale: bool = True,
    ) -> float:
        """Compute scIB integration LISI on a persisted KNN graph.

        Args:
            batch_colname: Cell metadata column containing batch labels.
            neighbors: Neighbor artifact to score. The assay's current
                neighbors are used when omitted.
            from_assay: Assay used to resolve current neighbors.
            cell_key: Optional cell-selection key, validated against neighbors.
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
            neighbors,
            from_assay=from_assay,
            cell_key=cell_key,
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
        neighbors: ArtifactRef | None = None,
        *,
        from_assay: str | None = None,
        cell_key: str | None = None,
        perplexity: float | None = None,
        scale: bool = True,
    ) -> float:
        """Compute scIB cell-type LISI on a persisted KNN graph.

        Args:
            label_colname: Cell metadata column containing biological labels.
            neighbors: Neighbor artifact to score. The assay's current
                neighbors are used when omitted.
            from_assay: Assay used to resolve current neighbors.
            cell_key: Optional cell-selection key, validated against neighbors.
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
            neighbors,
            from_assay=from_assay,
            cell_key=cell_key,
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
        graph: ArtifactRef | None = None,
        *,
        from_assay: str | None = None,
        cell_key: str | None = None,
    ) -> float:
        """Score label connectivity on a persisted, symmetrized assay graph.

        Args:
            label_colname: Cell metadata column containing biological labels.
            graph: Connectivity-map or integrated-graph artifact. The assay's
                current connectivity map is used when omitted.
            from_assay: Assay used to resolve the current graph.
            cell_key: Optional cell-selection key, validated against the graph.

        Returns:
            Mean fraction of cells retained in the largest connected component
            for each label.

        Notes:
            Persisted directed edges are treated as undirected. This follows
            the original scIB symmetrized-graph definition and intentionally
            differs from the directed strong-component calculation currently
            used by YosefLab ``scib-metrics``.
        """
        from ...metrics import graph_connectivity

        selection = resolve_graph_selection(
            self,
            graph,
            from_assay=from_assay,
            cell_key=cell_key,
        )
        n_cells, _ = self._get_graph_ncells_k(selection.graph_loc)
        labels = self.cells.fetch(label_colname, key=selection.cell_key)
        if len(labels) != n_cells:
            raise ValueError("Graph labels must match the number of cells in the graph")

        graph_grp = as_zarr_group(
            self.zw[selection.graph_loc],
            name=selection.graph_loc,
        )
        edges = as_zarr_array(graph_grp["edges"], name="edges")
        return graph_connectivity(edges, labels)

    def metric_graph_silhouette(
        self,
        res_label: str = "leiden_cluster",
        neighbors: ArtifactRef | None = None,
        *,
        from_assay: str | None = None,
        cell_key: str | None = None,
        random_seed: int = 4444,
        sample_size: int = 11,
    ) -> np.ndarray | None:
        """Calculate modified silhouette scores for evaluating cluster separation.

        This implements a graph-based silhouette score that measures how similar cells
        are to their own cluster compared to the nearest neighboring cluster.

        Args:
            res_label: Base or full column name containing cluster labels
                (default: "leiden_cluster")
            neighbors: Neighbor artifact to score. The assay's current
                neighbors are used when omitted.
            from_assay: Assay used to resolve current neighbors.
            cell_key: Optional cell-selection key, validated against neighbors.
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

        from ...metrics import silhouette_scoring

        neighbors, from_assay, cell_key, neighbor_distances, neighbor_indices = (
            self._load_metric_knn(
                neighbors,
                from_assay=from_assay,
                cell_key=cell_key,
            )
        )
        lineage = resolve_native_graph_inputs(self.zw, neighbors)
        coordinate_status = inspect_artifact(self.zw, lineage.coordinates)
        coordinate_group = group_at(self.zw, coordinate_status.path)
        ann_metric = str(
            (inspect_artifact(self.zw, lineage.ann_index).parameters or {}).get(
                "ann_metric",
                "l2",
            )
        )
        metric_data: Any
        if "data" in coordinate_group:
            metric_data = as_zarr_array(
                coordinate_group["data"],
                name="coordinates",
            )
            data_is_reduced = True
            ann_obj = cast(Any, SimpleNamespace(annMetric=ann_metric))
        else:
            ann_obj = self._load_artifact_ann_stream(
                from_assay,
                cell_key,
                True,
                neighbors,
            )
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
            distance_metric=cast(Any, ann_metric),
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
        normalized_status = self._require_complete_artifact(normalized, "normalized")
        execution = normalized_status.execution_options or {}
        selected_cell_key = execution.get("cell_key")
        state = (
            read_assay_state_document(self.zw, reduction.assay)
            if reduction.assay is not None
            else None
        )
        if state is not None and state.normalized == normalized:
            selected_cell_key = state.cell_key
        if not isinstance(selected_cell_key, str):
            raise ValueError("Reduction selection source columns are missing")
        if cell_key != selected_cell_key:
            raise ValueError(
                f"cell_key {cell_key!r} does not match the cell selection "
                f"{selected_cell_key!r} used to build the reduction"
            )
        raw_feature_selection = (normalized_status.inputs or {}).get(
            "feature_selection"
        )
        legacy_context = {
            "assay": normalized.assay,
            "artifact_id": normalized.artifact_id,
            "artifact_kind": normalized.kind,
            "input_name": "feature_selection",
        }
        if not isinstance(raw_feature_selection, Mapping):
            raise IncompatibleAnalysisStateError(
                "Normalized artifact uses the removed feature-selection contract",
                code="legacy_feature_contract",
                context=legacy_context,
            )
        try:
            feature_selection = ArtifactRef.from_dict(raw_feature_selection)
        except (TypeError, ValueError) as error:
            raise IncompatibleAnalysisStateError(
                "Normalized artifact has a malformed feature_selection input",
                code="legacy_feature_contract",
                context=legacy_context,
            ) from error
        if set(raw_feature_selection) != set(feature_selection.to_dict()):
            raise IncompatibleAnalysisStateError(
                "Normalized artifact has a malformed feature_selection input",
                code="legacy_feature_contract",
                context=legacy_context,
            )
        if normalized.assay is None:
            raise ArtifactResolutionError(
                "Normalized feature selection has no assay",
                code="wrong_scope",
                context={
                    "artifact_id": normalized.artifact_id,
                    "actual_scope": normalized.scope,
                    "expected_scope": "assay",
                },
            )
        resolve_feature_selection(
            self.zw,
            normalized.assay,
            feature_selection,
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
        neighbors: ArtifactRef | None = None,
        *,
        from_assay: str | None = None,
        cell_key: str | None = None,
        perplexity: float = 30,
    ) -> float:
        """Summarize batch LISI as a normalized neighborhood-mixing score.

        This computes batch LISI on the current KNN graph and rescales its mean
        against the mixing that perfectly integrated data would reach given the
        dataset's batch sizes. Unlike raw LISI, the result is bounded in
        ``[0, 1]``, which makes it easier to compare across graphs and datasets.

        Args:
            label_colname: Cell metadata column holding the batch assignment.
            neighbors: Neighbor artifact to score. The assay's current
                neighbors are used when omitted.
            from_assay: Assay used to resolve current neighbors.
            cell_key: Optional cell-selection key, validated against neighbors.
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

        neighbors, from_assay, cell_key = self._resolve_metric_neighbors(
            neighbors,
            from_assay=from_assay,
            cell_key=cell_key,
        )
        lisi_result = self.metric_lisi(
            label_columns=[label_colname],
            neighbors=neighbors,
            from_assay=from_assay,
            cell_key=cell_key,
            perplexity=perplexity,
        )
        batch_labels = self.cells.fetch(label_colname, key=cell_key)
        return lisi_batch_mixing_score(lisi_result[label_colname], batch_labels)
