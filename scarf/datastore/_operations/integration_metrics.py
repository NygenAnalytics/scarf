from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import pandas as pd
import zarr

from ...graph.distances import validate_distance_provenance
from ...graph.feature_projection import (
    graph_cell_selection,
    resolve_native_graph_inputs,
)
from ...metadata.artifacts import (
    artifact_values,
    plan_cell_data_artifact,
    write_cell_data_artifact,
)
from ...metadata.rows import read_metadata_rows_chunkwise
from ...storage.artifacts import (
    ArtifactRef,
    group_at,
    inspect_artifact,
)
from ...storage.errors import ArtifactResolutionError
from ...storage.feature_selection import resolve_feature_selection
from ...storage.selections import (
    read_stored_selection_indices,
    resolve_metadata_snapshot,
    validate_stored_selection_integrity,
)
from ...storage.types import as_zarr_array, as_zarr_group

if TYPE_CHECKING:
    from ...metrics import ClusterSeparabilityResult
    from ..mapping_datastore import MappingDatastore as _IntegrationMetricsBase
else:
    _IntegrationMetricsBase = object


class _IntegrationMetricsOperationsMixin(_IntegrationMetricsBase):
    def _resolve_metric_neighbors(
        self,
        neighbors: ArtifactRef,
    ) -> tuple[ArtifactRef, str, np.ndarray]:
        if not isinstance(neighbors, ArtifactRef):
            raise TypeError("neighbors must be an artifact reference")
        assay = neighbors.assay or ""
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
        lineage = resolve_native_graph_inputs(self.zw, neighbors)
        cell_indices = read_stored_selection_indices(
            self.zw,
            lineage.cell_selection,
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        )
        return neighbors, assay, cell_indices

    def _load_metric_knn(
        self,
        neighbors: ArtifactRef,
    ) -> tuple[ArtifactRef, str, np.ndarray, zarr.Array, zarr.Array]:
        ref, assay, cell_indices = self._resolve_metric_neighbors(neighbors)
        status = inspect_artifact(self.zw, ref)
        validate_distance_provenance(self.zw, ref)
        knn_grp = as_zarr_group(
            self.zw[status.path],
            name=status.path,
        )
        distances = as_zarr_array(knn_grp["distances"], name="distances")
        indices = as_zarr_array(knn_grp["indices"], name="indices")
        return ref, assay, cell_indices, distances, indices

    def metric_lisi(
        self,
        label_columns: Sequence[str],
        neighbors: ArtifactRef,
        *,
        perplexity: float = 30,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Calculate Local Inverse Simpson Index (LISI) scores for cell populations.

        LISI measures how well mixed different cell populations are in the local neighborhood
        of each cell. Higher scores indicate better mixing of different populations.

        Args:
            label_columns: Column names from cell metadata containing population labels
            neighbors: Explicit neighbor artifact to score.
            perplexity: Effective neighborhood size used by LISI. It is reduced
                with a warning when the graph has fewer than three times this
                many neighbors.
            invalidate_cache: Force creation of a new metric artifact.

        Returns:
            An immutable quality-metric artifact. Pass it to
            :meth:`load_metric_lisi` to read the per-cell scores.

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

        neighbors, assay, cell_indices, distances, indices = self._load_metric_knn(
            neighbors
        )
        try:
            labels = {
                column: read_metadata_rows_chunkwise(
                    self.cells,
                    column,
                    cell_indices,
                )
                for column in label_cols
            }
        except KeyError:
            raise KeyError(
                f"Could not find the column(s) {label_cols} in the cell metadata table."
            )

        row_ids = read_metadata_rows_chunkwise(
            self.cells,
            "ids",
            cell_indices,
        )
        label_snapshots = [
            {
                "column": column,
                "artifact": resolve_metadata_snapshot(
                    self.zw,
                    values=np.asarray(labels[column]),
                    row_ids=np.asarray(row_ids),
                    operation="snapshot_metric_label",
                    parameters={"column": column},
                    inputs={"neighbors": neighbors},
                    source_columns=[column],
                    invalidate_cache=invalidate_cache,
                ),
            }
            for column in label_cols
        ]
        selection = resolve_native_graph_inputs(self.zw, neighbors).cell_selection
        planned = plan_cell_data_artifact(
            self.zw,
            scope="assay",
            assay=assay,
            kind="quality_metric",
            operation="metric_lisi",
            parameters={
                "label_columns": label_cols,
                "perplexity": float(perplexity),
            },
            inputs={
                "neighbors": neighbors,
                "label_snapshots": label_snapshots,
            },
            execution_options={},
            cell_selection=selection,
            arrays={"values": ((len(cell_indices), len(label_cols)), "f")},
            invalidate_cache=invalidate_cache,
        )
        if not planned.reused:
            from ...metrics import compute_lisi

            lisi_scores = compute_lisi(
                distances,
                indices,
                pd.DataFrame(labels),
                label_cols,
                perplexity=perplexity,
            )
            write_cell_data_artifact(
                self.zw,
                planned,
                {"values": lisi_scores},
            )
        return planned.ref

    def load_metric_lisi(
        self,
        metric: ArtifactRef,
    ) -> dict[str, np.ndarray]:
        """Load per-cell LISI scores from an explicit metric artifact."""
        if not isinstance(metric, ArtifactRef):
            raise TypeError("metric must be an ArtifactRef")
        status = self._require_complete_artifact(metric, "quality_metric")
        if status.operation != "metric_lisi":
            raise ValueError("metric must reference a LISI quality-metric artifact")
        parameters = status.parameters or {}
        raw_columns = parameters.get("label_columns")
        if not isinstance(raw_columns, list) or not all(
            isinstance(column, str) and column for column in raw_columns
        ):
            raise ValueError("LISI metric label columns are malformed")
        label_columns = list(raw_columns)
        if len(set(label_columns)) != len(label_columns):
            raise ValueError("LISI metric label columns are malformed")
        values = artifact_values(group_at(self.zw, status.path), "values")
        if values.ndim != 2 or values.shape[1] != len(label_columns):
            raise ValueError("LISI metric values are malformed")
        return {
            column: scores.copy()
            for column, scores in zip(label_columns, values.T, strict=True)
        }

    def metric_ilisi(
        self,
        batch_colname: str,
        neighbors: ArtifactRef,
        *,
        perplexity: float | None = None,
        scale: bool = True,
    ) -> float:
        """Compute scIB integration LISI on a persisted KNN graph.

        Args:
            batch_colname: Cell metadata column containing batch labels.
            neighbors: Explicit neighbor artifact to score.
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

        _, _, cell_indices, distances, indices = self._load_metric_knn(neighbors)
        batch_labels = read_metadata_rows_chunkwise(
            self.cells,
            batch_colname,
            cell_indices,
        )
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
        neighbors: ArtifactRef,
        *,
        perplexity: float | None = None,
        scale: bool = True,
    ) -> float:
        """Compute scIB cell-type LISI on a persisted KNN graph.

        Args:
            label_colname: Cell metadata column containing biological labels.
            neighbors: Explicit neighbor artifact to score.
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

        _, _, cell_indices, distances, indices = self._load_metric_knn(neighbors)
        cell_labels = read_metadata_rows_chunkwise(
            self.cells,
            label_colname,
            cell_indices,
        )
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
        graph: ArtifactRef,
    ) -> float:
        """Score label connectivity on a persisted, symmetrized assay graph.

        Args:
            label_colname: Cell metadata column containing biological labels.
            graph: Explicit connectivity-map or integrated-graph artifact.
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

        if not isinstance(graph, ArtifactRef):
            raise TypeError("graph must be an ArtifactRef")
        if graph.kind not in {"connectivity_map", "integrated_graph"}:
            raise ValueError(
                "graph must reference a connectivity map or an integrated graph"
            )
        selection = graph_cell_selection(self.zw, graph)
        cell_indices = read_stored_selection_indices(
            self.zw,
            selection,
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        )
        status = inspect_artifact(self.zw, graph)
        n_cells, _ = self._get_graph_ncells_k(status.path)
        labels = read_metadata_rows_chunkwise(
            self.cells,
            label_colname,
            cell_indices,
        )
        if len(labels) != n_cells:
            raise ValueError("Graph labels must match the number of cells in the graph")

        graph_grp = as_zarr_group(
            self.zw[status.path],
            name=status.path,
        )
        edges = as_zarr_array(graph_grp["edges"], name="edges")
        return graph_connectivity(edges, labels)

    def metric_graph_silhouette(
        self,
        neighbors: ArtifactRef,
        clusters: ArtifactRef,
        *,
        random_seed: int = 4444,
        sample_size: int = 11,
    ) -> np.ndarray | None:
        """Calculate modified silhouette scores for evaluating cluster separation.

        This implements a graph-based silhouette score that measures how similar cells
        are to their own cluster compared to the nearest neighboring cluster.

        Args:
            neighbors: Explicit neighbor artifact to score.
            clusters: Explicit cluster-label or Paris cluster-cut artifact.
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

        neighbors, from_assay, cell_indices, neighbor_distances, neighbor_indices = (
            self._load_metric_knn(neighbors)
        )
        lineage = resolve_native_graph_inputs(self.zw, neighbors)
        if not isinstance(clusters, ArtifactRef):
            raise TypeError("clusters must be an ArtifactRef")
        if clusters.kind not in {"cluster_labels", "cluster_cut"}:
            raise ValueError(
                "clusters must reference cluster labels or a Paris cluster cut"
            )
        cluster_status = inspect_artifact(self.zw, clusters)
        if not cluster_status.complete:
            raise ValueError("Cluster artifact is unavailable or incomplete")
        raw_cluster_selection = (cluster_status.inputs or {}).get("cell_selection")
        if not isinstance(raw_cluster_selection, Mapping):
            raise ValueError("Cluster artifact has no cell-selection input")
        cluster_selection = ArtifactRef.from_dict(raw_cluster_selection)
        if cluster_selection != lineage.cell_selection:
            raise ValueError(
                "Cluster labels and neighbors use different cell selections"
            )
        cluster_group = group_at(self.zw, cluster_status.path)
        value_name = "labels" if clusters.kind == "cluster_cut" else "values"
        if value_name not in cluster_group:
            raise ValueError(
                f"Cluster artifact has no canonical {value_name!r} label array"
            )
        cluster_labels = np.asarray(
            as_zarr_array(cluster_group[value_name], name=value_name)[:]
        )
        if cluster_labels.ndim != 1 or len(cluster_labels) != len(cell_indices):
            raise ValueError(
                "Cluster labels must contain one value per selected graph cell"
            )
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
                neighbors,
                True,
            )
            if ann_obj.harmonize:
                raise ValueError("Harmony coordinates are missing for this KNN graph")
            metric_data = ann_obj.data
            data_is_reduced = False
        selected_cells = SimpleNamespace(
            columns=("clusters",),
            fetch=lambda column, key="I": cluster_labels,
        )
        scores = silhouette_scoring(
            SimpleNamespace(cells=selected_cells),  # type: ignore[arg-type]
            ann_obj,
            None,
            metric_data,
            from_assay,
            "clusters",
            cell_key="I",
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
    ) -> ArtifactRef:
        normalized = self._artifact_input_ref(reduction, "normalized", "normalized")
        normalized_status = self._require_complete_artifact(normalized, "normalized")
        cell_selection = self._artifact_input_ref(
            normalized,
            "cell_selection",
            "cell_selection",
        )
        validate_stored_selection_integrity(
            self.zw,
            cell_selection,
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        )
        raw_feature_selection = (normalized_status.inputs or {}).get(
            "feature_selection"
        )
        context = {
            "assay": normalized.assay,
            "artifact_id": normalized.artifact_id,
            "artifact_kind": normalized.kind,
            "input_name": "feature_selection",
        }
        if not isinstance(raw_feature_selection, Mapping):
            raise ArtifactResolutionError(
                "Normalized artifact is missing its feature_selection input",
                code="corrupt_payload",
                context=context,
            )
        try:
            feature_selection = ArtifactRef.from_dict(raw_feature_selection)
        except (TypeError, ValueError) as error:
            raise ArtifactResolutionError(
                "Normalized artifact has a malformed feature_selection input",
                code="corrupt_payload",
                context=context,
            ) from error
        if set(raw_feature_selection) != set(feature_selection.to_dict()):
            raise ArtifactResolutionError(
                "Normalized artifact has a malformed feature_selection input",
                code="corrupt_payload",
                context=context,
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
        return cell_selection

    def metric_cluster_separability(
        self,
        pca: ArtifactRef,
        clusters: Mapping[str, ArtifactRef],
        *,
        n_folds: int = 5,
        max_sample_cells: int = 50_000,
        max_silhouette_cells: int = 10_000,
        random_seed: int = 4444,
        svm_c: float = 1.0,
        svm_max_iter: int = 10_000,
    ) -> "ClusterSeparabilityResult":
        """Evaluate cluster-label separability in PCA coordinates."""
        from ...metrics import evaluate_cluster_separability

        if not isinstance(clusters, Mapping) or not clusters:
            raise TypeError("clusters must be a non-empty mapping of names to refs")
        if not all(isinstance(name, str) and name for name in clusters):
            raise TypeError("cluster names must be non-empty strings")
        if not all(isinstance(ref, ArtifactRef) for ref in clusters.values()):
            raise TypeError("cluster values must be ArtifactRefs")

        status = self._require_complete_artifact(pca, "reduction")
        if status.operation != "run_pca":
            raise ValueError("pca must reference a PCA reduction artifact")
        cell_selection = self._validate_reduction_cell_selection(pca)
        group = group_at(self.zw, status.path)
        if "data" not in group:
            raise ValueError("PCA reduction coordinates are missing")
        coordinates = as_zarr_array(group["data"], name="PCA coordinates")
        clusterings: dict[str, np.ndarray] = {}
        for name, ref in clusters.items():
            if (
                ref.kind not in {"cluster_labels", "cluster_cut"}
                or ref.scope != "assay"
                or ref.assay != pca.assay
            ):
                raise ValueError(
                    f"Cluster {name!r} must be an assay-scoped clustering "
                    "artifact for the PCA assay"
                )
            cluster_status = self._require_complete_artifact(
                ref,
                ref.kind,
                assay=pca.assay,
            )
            raw_selection = (cluster_status.inputs or {}).get("cell_selection")
            if not isinstance(raw_selection, Mapping):
                raise ValueError(f"Cluster {name!r} has no cell-selection input")
            try:
                cluster_selection = ArtifactRef.from_dict(raw_selection)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Cluster {name!r} has a malformed cell-selection input"
                ) from error
            if cluster_selection != cell_selection:
                raise ValueError(
                    f"Cluster {name!r} does not use the PCA cell selection"
                )
            value_name = "labels" if ref.kind == "cluster_cut" else "values"
            cluster_group = group_at(self.zw, cluster_status.path)
            if value_name not in cluster_group:
                raise ValueError(f"Cluster {name!r} has no label values")
            labels = np.asarray(
                as_zarr_array(cluster_group[value_name], name=value_name)[:]
            )
            if labels.ndim != 1:
                raise ValueError(f"Cluster {name!r} labels must be one-dimensional")
            if len(labels) != coordinates.shape[0]:
                raise ValueError(f"Cluster {name!r} does not align with PCA rows")
            clusterings[name] = labels

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
        neighbors: ArtifactRef,
        *,
        perplexity: float = 30,
    ) -> float:
        """Summarize batch LISI as a normalized neighborhood-mixing score.

        This computes batch LISI on the supplied KNN graph and rescales its mean
        against the mixing that perfectly integrated data would reach given the
        dataset's batch sizes. Unlike raw LISI, the result is bounded in
        ``[0, 1]``, which makes it easier to compare across graphs and datasets.

        Args:
            label_colname: Cell metadata column holding the batch assignment.
            neighbors: Explicit neighbor artifact to score.
            perplexity: Effective neighborhood size passed to LISI.

        Returns:
            A value in ``[0, 1]``. Scores near 1 indicate that neighborhoods mix
            batches as well as the global composition allows, and scores near 0
            indicate poorly mixed batches.

        Raises:
            ValueError: If KNN inputs are invalid or the column has fewer than
                two batches.
        """
        from ...metrics import compute_lisi, lisi_batch_mixing_score

        _, _, cell_indices, distances, indices = self._load_metric_knn(neighbors)
        batch_labels = read_metadata_rows_chunkwise(
            self.cells,
            label_colname,
            cell_indices,
        )
        lisi_scores = compute_lisi(
            distances,
            indices,
            pd.DataFrame({label_colname: batch_labels}),
            [label_colname],
            perplexity=perplexity,
        )[:, 0]
        return lisi_batch_mixing_score(lisi_scores, batch_labels)
