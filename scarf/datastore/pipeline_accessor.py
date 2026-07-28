from typing import Any, Literal

from ..storage.artifacts import ArtifactRef
from ..storage.types import as_zarr_array, as_zarr_group


type StepOptions = dict[str, Any] | Literal[False] | None


class PipelineAccessor:
    __slots__ = ("_store",)

    def __init__(self, store: Any) -> None:
        self._store = store

    def _column_ref(self, column: str) -> ArtifactRef:
        cell_data = as_zarr_group(
            self._store.zw["cellData"],
            name="cellData",
        )
        raw_ref = as_zarr_array(
            cell_data[column],
            name=column,
        ).attrs.get("source_artifact")
        if not isinstance(raw_ref, dict):
            raise RuntimeError(f"Pipeline output column {column!r} has no artifact ref")
        return ArtifactRef.from_dict(raw_ref)

    def _feature_ref(self, assay_name: str, column: str) -> ArtifactRef:
        assay = self._store._get_assay(assay_name)
        feature_data = as_zarr_group(
            assay.z["featureData"],
            name="featureData",
        )
        raw_ref = as_zarr_array(
            feature_data[column],
            name=column,
        ).attrs.get("source_artifact")
        if not isinstance(raw_ref, dict):
            raise RuntimeError(
                f"Pipeline feature column {column!r} has no artifact ref"
            )
        return ArtifactRef.from_dict(raw_ref)

    def _marker_ref(
        self,
        assay_name: str,
        cell_key: str,
        group_key: str,
    ) -> ArtifactRef:
        assay = self._store._get_assay(assay_name)
        markers = as_zarr_group(assay.z["markers"], name="markers")
        raw_artifacts = markers.attrs.get("artifacts", {})
        if not isinstance(raw_artifacts, dict):
            raise RuntimeError("Marker artifact index is invalid")
        raw_ref = raw_artifacts.get(f"{cell_key}__{group_key}")
        if not isinstance(raw_ref, dict):
            raise RuntimeError("Marker search did not publish an artifact")
        return ArtifactRef.from_dict(raw_ref)

    @staticmethod
    def _options(value: StepOptions) -> dict[str, Any]:
        return {} if value is None or value is False else dict(value)

    @staticmethod
    def _resolution_label(resolution: float) -> str:
        return f"leiden_{resolution}"

    @staticmethod
    def _cluster_recipe_key(value: Any) -> str:
        if value == "paris":
            return "paris"
        if isinstance(value, dict) and set(value) == {"leiden"}:
            return PipelineAccessor._resolution_label(float(value["leiden"]))
        if isinstance(value, int | float):
            return PipelineAccessor._resolution_label(float(value))
        if isinstance(value, str) and value.startswith("leiden_"):
            return value
        raise ValueError(
            "clusters must be 'paris', a Leiden resolution, or {'leiden': resolution}"
        )

    def run(
        self,
        pipeline_id: str = "basic_rna_analysis",
        *,
        from_assay: str | None = None,
        cell_key: str = "I",
        filtering: StepOptions = None,
        cell_cycle_scoring: StepOptions = None,
        highly_variable_features: StepOptions = None,
        normalization: StepOptions = None,
        pca: StepOptions = None,
        harmony: dict[str, Any] | None = None,
        ann_index: StepOptions = None,
        neighbors: StepOptions = None,
        connectivity: StepOptions = None,
        umap: StepOptions = None,
        leiden: dict[float, dict[str, Any]] | None = None,
        paris: StepOptions = None,
        doublet_scoring: StepOptions = None,
        markers: StepOptions = None,
    ) -> dict[str, ArtifactRef]:
        """Run the standard provenance-backed RNA analysis recipe.

        Most step options accept ``None`` to run with defaults, ``False`` to
        skip, or a dictionary forwarded to the underlying operation. Harmony
        is skipped when omitted and requires a dictionary containing
        ``batch_columns``. Leiden defaults to one run at resolution 1.0.

        Args:
            pipeline_id: Recipe identifier. Only ``basic_rna_analysis`` is
                currently available.
            from_assay: RNA assay to analyze. Uses the default assay when
                omitted.
            cell_key: Boolean cell selection used throughout the recipe.
            filtering: Filtering options, including ``method="auto"`` or
                ``method="manual"``.
            cell_cycle_scoring: Cell-cycle scoring options or ``False``.
            highly_variable_features: HVG selection options or ``False``.
            normalization: Normalization options.
            pca: PCA options. ``n_centroids`` is consumed by embedding
                initialization.
            harmony: Harmony options with a non-empty ``batch_columns`` list.
            ann_index: ANN-index construction options.
            neighbors: Neighbor-query options.
            connectivity: Connectivity-map options.
            umap: UMAP options or ``False``.
            leiden: Mapping from resolution to Leiden options. Use an empty
                mapping to run no Leiden clustering.
            paris: Paris clustering options or ``False``.
            doublet_scoring: Doublet-scoring options or ``False``.
            markers: Marker-search options or ``False``.

        Returns:
            Artifact references keyed by pipeline result name.

        Raises:
            ValueError: If the recipe identifier or dependent step options are
                invalid.
            RuntimeError: If a step does not publish its expected artifact.
        """
        if pipeline_id != "basic_rna_analysis":
            raise ValueError(
                f"Unknown pipeline_id {pipeline_id!r}; "
                "available pipelines: basic_rna_analysis"
            )
        store = self._store
        assay_name = from_assay or store._defaultAssay
        if assay_name is None:
            raise ValueError("No assay was provided and no default is configured")
        if filtering is not False and cell_key != "I":
            raise ValueError(
                "basic_rna_analysis filtering currently requires cell_key='I'"
            )
        if isinstance(markers, dict) and markers.get("skip_save") is True:
            raise ValueError("basic_rna_analysis markers cannot use skip_save=True")
        artifacts: dict[str, ArtifactRef] = {}

        if filtering is not False:
            options = self._options(filtering)
            method = options.pop("method", "auto")
            if method == "auto":
                options.setdefault("show_qc_plots", False)
                if "attrs" not in options:
                    options["attrs"] = [
                        column
                        for suffix in (
                            "nCounts",
                            "nFeatures",
                            "percentMito",
                            "percentRibo",
                        )
                        if (column := f"{assay_name}_{suffix}") in store.cells.columns
                    ]
                store.auto_filter_cells(**options)
            elif method == "manual":
                store.filter_cells(**options)
            else:
                raise ValueError("filtering method must be 'auto' or 'manual'")
            artifacts["cell_selection"] = store._ensure_cell_selection(cell_key)

        if cell_cycle_scoring is not False:
            options = self._options(cell_cycle_scoring)
            store.run_cell_cycle_scoring(
                from_assay=assay_name,
                cell_key=cell_key,
                **options,
            )
            phase_label = options.get("phase_label", "cell_cycle_phase")
            phase_column = store._col_renamer(
                assay_name,
                cell_key,
                phase_label,
            )
            artifacts["cell_cycle"] = self._column_ref(phase_column)

        hvg_options = self._options(highly_variable_features)
        hvg_name = str(hvg_options.get("hvg_key_name", "hvgs"))
        if highly_variable_features is not False:
            hvg_options.setdefault("show_plot", False)
            store.mark_hvgs(
                from_assay=assay_name,
                cell_key=cell_key,
                **hvg_options,
            )
            feature_column = f"{cell_key}__{hvg_name}"
            artifacts["highly_variable_features"] = self._feature_ref(
                assay_name,
                feature_column,
            )

        normalization_options = self._options(normalization)
        normalization_options.setdefault("log_transform", True)
        normalization_options.setdefault("renormalize_subset", True)
        normalized = store.run_normalization(
            from_assay=assay_name,
            cell_key=cell_key,
            feat_key=hvg_name,
            update_state=False,
            **normalization_options,
        )
        artifacts["normalized"] = normalized

        pca_options = self._options(pca)
        n_centroids = int(pca_options.pop("n_centroids", 1000))
        initialization_rand_state = int(pca_options.pop("rand_state", 4466))
        reduction = store.run_pca(
            normalized,
            update_state=False,
            **pca_options,
        )
        artifacts["pca"] = reduction

        coordinates = reduction
        if harmony is not None:
            harmony_options = dict(harmony)
            batch_columns = harmony_options.pop("batch_columns", None)
            if not isinstance(batch_columns, list) or not batch_columns:
                raise ValueError("harmony requires a non-empty batch_columns list")
            coordinates = store.run_harmony(
                batch_columns,
                reduction,
                update_state=False,
                **harmony_options,
            )
            artifacts["harmony"] = coordinates

        ann = store.build_ann_index(
            coordinates,
            update_state=False,
            **self._options(ann_index),
        )
        artifacts["ann_index"] = ann
        neighbor_ref = store.query_neighbors(
            ann,
            coordinates=coordinates,
            update_state=False,
            **self._options(neighbors),
        )
        artifacts["neighbors"] = neighbor_ref
        graph = store.build_connectivity_map(
            neighbor_ref,
            update_state=False,
            **self._options(connectivity),
        )
        artifacts["connectivity_map"] = graph

        initialization = store._build_embedding_initialization(
            reduction,
            n_centroids=n_centroids,
            rand_state=initialization_rand_state,
            batch_size=None,
            invalidate_cache=False,
        )
        store._publish_current_artifact(
            graph,
            update_state=True,
            embedding_initialization=initialization,
        )
        artifacts["embedding_initialization"] = initialization

        if umap is not False:
            umap_options = self._options(umap)
            umap_label = str(umap_options.get("label", "UMAP"))
            store.run_umap(
                from_assay=assay_name,
                cell_key=cell_key,
                feat_key=hvg_name,
                **umap_options,
            )
            umap_column = store._col_renamer(
                assay_name,
                cell_key,
                f"{umap_label}1",
            )
            artifacts["umap"] = self._column_ref(umap_column)

        cluster_columns: dict[str, str] = {}
        leiden_options = {1.0: {}} if leiden is None else dict(leiden)
        for raw_resolution, raw_options in leiden_options.items():
            resolution = float(raw_resolution)
            options = dict(raw_options)
            recipe_key = self._resolution_label(resolution)
            label = str(options.pop("label", recipe_key))
            store.run_leiden_clustering(
                from_assay=assay_name,
                cell_key=cell_key,
                feat_key=hvg_name,
                resolution=resolution,
                label=label,
                **options,
            )
            column = store._col_renamer(
                assay_name,
                cell_key,
                label,
            )
            cluster_columns[recipe_key] = column
            artifacts[recipe_key] = self._column_ref(column)

        if paris is not False:
            paris_options = self._options(paris)
            paris_label = str(paris_options.pop("label", "paris_cluster"))
            result = store.run_paris_clustering(
                from_assay=assay_name,
                cell_key=cell_key,
                feat_key=hvg_name,
                label=paris_label,
                **paris_options,
            )
            if result.label_key is None:
                raise RuntimeError("Paris clustering did not publish labels")
            cluster_columns["paris"] = result.label_key
            artifacts["paris"] = self._column_ref(result.label_key)

        if doublet_scoring is not False:
            options = self._options(doublet_scoring)
            selector = options.pop(
                "clusters",
                "paris"
                if "paris" in cluster_columns
                else max(
                    cluster_columns,
                    key=lambda key: (
                        float(key.removeprefix("leiden_"))
                        if key.startswith("leiden_")
                        else float("-inf")
                    ),
                ),
            )
            recipe_key = self._cluster_recipe_key(selector)
            if recipe_key not in cluster_columns:
                raise ValueError(
                    f"Doublet cluster result {recipe_key!r} is unavailable"
                )
            score_column = store.run_doublet_detection(
                cluster_key=cluster_columns[recipe_key],
                from_assay=assay_name,
                cell_key=cell_key,
                feat_key=hvg_name,
                **options,
            )
            artifacts["doublets"] = self._column_ref(score_column)

        if markers is not False:
            options = self._options(markers)
            default_marker_clusters: Any = (
                "paris"
                if "paris" in cluster_columns
                else max(
                    (key for key in cluster_columns if key.startswith("leiden_")),
                    key=lambda key: float(key.removeprefix("leiden_")),
                )
            )
            selector = options.pop(
                "clusters",
                default_marker_clusters,
            )
            recipe_key = self._cluster_recipe_key(selector)
            if recipe_key not in cluster_columns:
                raise ValueError(f"Marker cluster result {recipe_key!r} is unavailable")
            group_key = cluster_columns[recipe_key]
            options.setdefault("feat_key", "I")
            store.run_marker_search(
                from_assay=assay_name,
                cell_key=cell_key,
                group_key=group_key,
                **options,
            )
            artifacts["markers"] = self._marker_ref(
                assay_name,
                cell_key,
                group_key,
            )
        return artifacts
