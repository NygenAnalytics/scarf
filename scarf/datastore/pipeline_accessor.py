from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from ..metadata.artifacts import column_display, link_cell_data_column
from ..storage.artifacts import ArtifactRef, group_at
from ..storage.types import as_zarr_array, as_zarr_group
from ..utils.logging import logger


type StepOptions = dict[str, Any] | Literal[False] | None
type PipelineEventKind = Literal[
    "stage_started",
    "stage_completed",
    "stage_failed",
]


@dataclass(frozen=True, slots=True)
class PipelineEvent:
    kind: PipelineEventKind
    stage: str
    error: Exception | None = None


type PipelineCallback = Callable[[PipelineEvent], None]


class _PipelineEventEmitter:
    __slots__ = ("_callback",)

    def __init__(self, callback: PipelineCallback | None) -> None:
        self._callback = callback

    def emit(
        self,
        kind: PipelineEventKind,
        stage: str,
        error: Exception | None = None,
    ) -> None:
        if self._callback is None:
            return
        event = PipelineEvent(kind=kind, stage=stage, error=error)
        try:
            self._callback(event)
        except Exception:
            logger.exception(
                f"Pipeline callback failed while handling {kind} for {stage}"
            )

    def start(self, stage: str) -> None:
        logger.info(f"Running pipeline stage: {stage.replace('_', ' ')}")
        self.emit("stage_started", stage)

    def complete(self, stage: str) -> None:
        self.emit("stage_completed", stage)
        logger.info(f"Completed pipeline stage: {stage.replace('_', ' ')}")

    @contextmanager
    def stage(self, stage: str) -> Iterator[None]:
        self.start(stage)
        try:
            yield
        except Exception as error:
            self.emit("stage_failed", stage, error)
            raise
        self.complete(stage)


_DEFAULT_LEIDEN: dict[float, dict[str, Any]] = {
    0.5: {},
    0.75: {},
    1.0: {},
    1.25: {},
}

_SELECTED_CLUSTER_LABEL = "clusters"


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

    def _column_source_value(self, column: str) -> str:
        cell_data = as_zarr_group(
            self._store.zw["cellData"],
            name="cellData",
        )
        value_name = as_zarr_array(
            cell_data[column],
            name=column,
        ).attrs.get("source_value")
        if not isinstance(value_name, str):
            raise RuntimeError(
                f"Cluster column {column!r} is not linked to an artifact value"
            )
        return value_name

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
            raise RuntimeError("Marker search did not write an artifact")
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

    def _load_pca_coordinates(self, reduction: ArtifactRef) -> np.ndarray:
        status = self._store.inspect_artifact(reduction)
        group = group_at(self._store.zw, status.path)
        if "data" not in group:
            source, _n_cells, _dims = self._store._coordinate_source(
                reduction,
                batch_size=None,
            )
            blocks = list(source.iter_coordinate_blocks("Loading PCA for silhouette"))
            if not blocks:
                raise RuntimeError("PCA reduction produced no coordinate blocks")
            return np.vstack(blocks)
        return np.asarray(as_zarr_array(group["data"], name="data")[:], dtype=float)

    def _select_clusters_by_pca_silhouette(
        self,
        *,
        reduction: ArtifactRef,
        cell_key: str,
        cluster_columns: dict[str, str],
    ) -> str:
        from sklearn.metrics import silhouette_score

        if not cluster_columns:
            raise ValueError(
                "Silhouette cluster selection requires at least one clustering result"
            )
        coordinates = self._load_pca_coordinates(reduction)
        sample_size = min(10_000, coordinates.shape[0])
        best_key: str | None = None
        best_score = float("-inf")
        scores: dict[str, float] = {}
        for recipe_key, column in cluster_columns.items():
            labels = np.asarray(self._store.cells.fetch(column, key=cell_key))
            if labels.shape[0] != coordinates.shape[0]:
                raise RuntimeError(
                    "PCA coordinates and cluster labels cover different cells"
                )
            n_labels = len(np.unique(labels))
            if n_labels < 2:
                logger.warning(
                    f"Skipping silhouette for {recipe_key}: fewer than two clusters"
                )
                continue
            score = float(
                silhouette_score(
                    coordinates,
                    labels,
                    sample_size=sample_size if sample_size < labels.shape[0] else None,
                    random_state=4466,
                )
            )
            scores[recipe_key] = score
            if score > best_score:
                best_score = score
                best_key = recipe_key
        if best_key is None:
            raise RuntimeError(
                "Could not score any clustering partition with silhouette"
            )
        logger.info(
            "Cluster silhouette scores on PCA: "
            + ", ".join(f"{key}={value:.4f}" for key, value in scores.items())
            + f"; selected {best_key}"
        )
        return best_key

    def _publish_selected_clusters(
        self,
        *,
        assay_name: str,
        cell_key: str,
        source_column: str,
        ref: ArtifactRef,
    ) -> str:
        store = self._store
        column: str = store._col_renamer(assay_name, cell_key, _SELECTED_CLUSTER_LABEL)
        if column == source_column:
            return column
        labels = np.asarray(store.cells.fetch(source_column, key=cell_key))
        store.cells.insert(
            column,
            labels,
            fill_value=-1,
            key=cell_key,
            overwrite=True,
        )
        link_cell_data_column(
            store.zw,
            column,
            ref,
            value_name=self._column_source_value(source_column),
            default_display=column_display(store.zw, source_column),
        )
        logger.info(f"Selected {source_column} as {column}")
        return column

    def _run_clustering_jobs(
        self,
        *,
        graph: ArtifactRef,
        assay_name: str,
        cell_key: str,
        feat_key: str,
        leiden_options: dict[float, dict[str, Any]],
        paris_options: dict[str, Any] | None,
        clustering_concurrency: int,
        events: _PipelineEventEmitter,
    ) -> tuple[dict[str, str], dict[str, ArtifactRef]]:
        store = self._store
        cluster_columns: dict[str, str] = {}
        artifacts: dict[str, ArtifactRef] = {}
        if not leiden_options and paris_options is None:
            return cluster_columns, artifacts

        job_order: list[str] = []
        prepared_leiden: dict[str, Any] = {}
        started_jobs: set[str] = set()
        terminal_jobs: set[str] = set()

        def start_job(recipe_key: str) -> None:
            if recipe_key not in started_jobs:
                events.start(recipe_key)
                started_jobs.add(recipe_key)

        def complete_job(recipe_key: str) -> None:
            events.complete(recipe_key)
            terminal_jobs.add(recipe_key)

        def fail_job(recipe_key: str, error: Exception) -> None:
            start_job(recipe_key)
            events.emit("stage_failed", recipe_key, error)
            terminal_jobs.add(recipe_key)

        def fail_unpublished_jobs() -> None:
            for recipe_key in job_order:
                if recipe_key in started_jobs and recipe_key not in terminal_jobs:
                    fail_job(
                        recipe_key,
                        RuntimeError(
                            f"Clustering job {recipe_key} was not written because "
                            "another clustering job failed"
                        ),
                    )

        for raw_resolution, raw_options in leiden_options.items():
            resolution = float(raw_resolution)
            options = dict(raw_options)
            recipe_key = self._resolution_label(resolution)
            if recipe_key in prepared_leiden:
                raise ValueError(f"Duplicate Leiden resolution {resolution}")
            label = str(options.pop("label", recipe_key))
            job_order.append(recipe_key)
            try:
                prepared_leiden[recipe_key] = store._prepare_leiden_clustering(
                    graph,
                    from_assay=assay_name,
                    cell_key=cell_key,
                    feat_key=feat_key,
                    resolution=resolution,
                    label=label,
                    **options,
                )
            except Exception as error:
                fail_job(recipe_key, error)
                raise

        paris_job: dict[str, Any] | None = None
        if paris_options is not None:
            options = dict(paris_options)
            paris_label = str(options.pop("label", "paris_cluster"))
            job_order.append("paris")
            paris_job = {
                "label": paris_label,
                "options": options,
            }

        graph_cache: dict[tuple[str, bool, bool], Any] = {}
        for recipe_key, prepared in prepared_leiden.items():
            if prepared.planned.reused or prepared.graph_key in graph_cache:
                continue
            try:
                graph_cache[prepared.graph_key] = store._load_prepared_leiden_graph(
                    prepared
                )
            except Exception as error:
                fail_job(recipe_key, error)
                raise

        compute_results: dict[str, np.ndarray] = {}
        completed: dict[str, str] = {}
        artifact_refs: dict[str, ArtifactRef] = {}
        first_error: Exception | None = None
        runnable_jobs = sum(
            not prepared.planned.reused for prepared in prepared_leiden.values()
        ) + int(paris_job is not None)
        if runnable_jobs:
            workers = max(1, min(clustering_concurrency, runnable_jobs))
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures: dict[Any, str] = {}
                for recipe_key in job_order:
                    if recipe_key == "paris":
                        assert paris_job is not None
                        start_job(recipe_key)
                        try:
                            future = executor.submit(
                                store.run_paris_clustering,
                                graph,
                                from_assay=assay_name,
                                cell_key=cell_key,
                                feat_key=feat_key,
                                label=paris_job["label"],
                                **paris_job["options"],
                            )
                        except Exception as error:
                            fail_job(recipe_key, error)
                            if first_error is None:
                                first_error = error
                            continue
                    else:
                        prepared = prepared_leiden[recipe_key]
                        if prepared.planned.reused:
                            continue
                        start_job(recipe_key)
                        try:
                            future = executor.submit(
                                store._compute_prepared_leiden,
                                prepared,
                                graph_cache[prepared.graph_key],
                            )
                        except Exception as error:
                            fail_job(recipe_key, error)
                            if first_error is None:
                                first_error = error
                            continue
                    futures[future] = recipe_key
                for future in as_completed(futures):
                    recipe_key = futures[future]
                    try:
                        result = future.result()
                        if recipe_key == "paris":
                            if result is None or result.label_key is None:
                                raise RuntimeError(
                                    "Paris clustering did not write labels"
                                )
                            if result.ref is None:
                                raise RuntimeError(
                                    "Paris clustering did not record an artifact"
                                )
                            completed[recipe_key] = result.label_key
                            artifact_refs[recipe_key] = result.ref
                            complete_job(recipe_key)
                        else:
                            compute_results[recipe_key] = result
                    except Exception as error:
                        fail_job(recipe_key, error)
                        if first_error is None:
                            first_error = error

        if first_error is not None:
            fail_unpublished_jobs()
            raise first_error

        for recipe_key, prepared in prepared_leiden.items():
            start_job(recipe_key)
            try:
                column, ref = store._publish_prepared_leiden(
                    prepared,
                    compute_results.get(recipe_key),
                )
                completed[recipe_key] = column
                artifact_refs[recipe_key] = ref
            except Exception as error:
                fail_job(recipe_key, error)
                fail_unpublished_jobs()
                raise
            complete_job(recipe_key)

        for recipe_key in job_order:
            column = completed[recipe_key]
            cluster_columns[recipe_key] = column
            artifacts[recipe_key] = artifact_refs[recipe_key]
        return cluster_columns, artifacts

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
        clustering_concurrency: int = 2,
        doublet_scoring: StepOptions = None,
        markers: StepOptions = None,
        callback: PipelineCallback | None = None,
    ) -> dict[str, ArtifactRef]:
        """Run the standard provenance-backed RNA analysis recipe.

        Most step options accept ``None`` to run with defaults, ``False`` to
        skip, or a dictionary forwarded to the underlying operation. Harmony
        is skipped when omitted and requires a dictionary containing
        ``batch_columns``. Leiden defaults to resolutions 0.5, 0.75, 1.0, and
        1.25. Leiden and Paris membership work can overlap under
        ``clustering_concurrency`` while store writes stay serialized. When
        more than one partition is available, the one with the highest
        silhouette score on PCA coordinates is selected. Its labels are copied
        to ``{assay}_clusters``, linked to the same artifact, and used by
        doublet scoring and marker search unless those steps name a partition
        through ``clusters``. Highly variable feature selection is mandatory.
        When provided, ``callback`` receives
        serialized stage events on the calling thread. Callback errors are
        logged without interrupting the pipeline. Stable stage names are
        ``filtering``, ``cell_cycle_scoring``, ``highly_variable_features``,
        ``normalization``, ``pca``, ``harmony``, ``ann_index``, ``neighbors``,
        ``connectivity``, ``embedding_initialization``, ``umap``,
        ``cluster_selection``, ``doublet_scoring``, and ``markers``. Clustering
        jobs use ``leiden_<resolution>`` and ``paris``. Skipped stages emit no
        events. ``stage_completed`` means the expected output has finished
        writing and is available. If one clustering job fails, any started
        sibling that cannot finish writing
        emits ``stage_failed`` with an abort error; the original job error is
        re-raised.

        Args:
            pipeline_id: Recipe identifier. Only ``basic_rna_analysis`` is
                currently available.
            from_assay: RNA assay to analyze. Uses the default assay when
                omitted.
            cell_key: Boolean cell selection used throughout the recipe.
            filtering: Filtering options, including ``method="auto"`` or
                ``method="manual"``.
            cell_cycle_scoring: Cell-cycle scoring options or ``False``.
            highly_variable_features: HVG selection options. Cannot be
                ``False``.
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
            clustering_concurrency: Maximum number of Leiden/Paris jobs that
                may run at once. Membership compute can overlap; store writes
                are serialized. Use ``1`` for a fully serial path.
            doublet_scoring: Doublet-scoring options or ``False``.
            markers: Marker-search options or ``False``.
            callback: Optional callable receiving ``PipelineEvent`` values.

        Returns:
            Artifact references keyed by pipeline result name.

        Raises:
            ValueError: If the recipe identifier or dependent step options are
                invalid.
            RuntimeError: If a step does not write its expected artifact.
        """
        if pipeline_id != "basic_rna_analysis":
            raise ValueError(
                f"Unknown pipeline_id {pipeline_id!r}; "
                "available pipelines: basic_rna_analysis"
            )
        if (
            isinstance(clustering_concurrency, bool)
            or not isinstance(clustering_concurrency, int)
            or clustering_concurrency < 1
        ):
            raise ValueError("clustering_concurrency must be an integer >= 1")
        if callback is not None and not callable(callback):
            raise TypeError("callback must be callable")
        store = self._store
        assay_name = from_assay or store._defaultAssay
        if assay_name is None:
            raise ValueError("No assay was provided and no default is configured")
        if filtering is not False and cell_key != "I":
            raise ValueError(
                "basic_rna_analysis filtering currently requires cell_key='I'"
            )
        if highly_variable_features is False:
            raise ValueError(
                "basic_rna_analysis requires highly_variable_features; "
                "pass options or omit the argument to use defaults"
            )
        if isinstance(markers, dict) and markers.get("skip_save") is True:
            raise ValueError("basic_rna_analysis markers cannot use skip_save=True")
        artifacts: dict[str, ArtifactRef] = {}
        events = _PipelineEventEmitter(callback)

        if filtering is not False:
            with events.stage("filtering"):
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
                            if (column := f"{assay_name}_{suffix}")
                            in store.cells.columns
                        ]
                    store.auto_filter_cells(**options)
                elif method == "manual":
                    store.filter_cells(**options)
                else:
                    raise ValueError("filtering method must be 'auto' or 'manual'")
                artifacts["cell_selection"] = store._ensure_cell_selection(cell_key)

        if cell_cycle_scoring is not False:
            with events.stage("cell_cycle_scoring"):
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

        with events.stage("highly_variable_features"):
            hvg_options = self._options(highly_variable_features)
            hvg_name = str(hvg_options.get("hvg_key_name", "hvgs"))
            hvg_options.setdefault("show_plot", False)
            hvg_options.setdefault("top_n", 1000)
            hvg_options.setdefault("min_cells", 20)
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

        with events.stage("normalization"):
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

        with events.stage("pca"):
            pca_options = self._options(pca)
            n_centroids = int(pca_options.pop("n_centroids", 1000))
            initialization_rand_state = int(pca_options.pop("rand_state", 4466))
            pca_options.setdefault("dims", 21)
            reduction = store.run_pca(
                normalized,
                update_state=False,
                **pca_options,
            )
            artifacts["pca"] = reduction

        coordinates = reduction
        if harmony is not None:
            with events.stage("harmony"):
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

        with events.stage("ann_index"):
            ann = store.build_ann_index(
                coordinates,
                update_state=False,
                **self._options(ann_index),
            )
            artifacts["ann_index"] = ann

        with events.stage("neighbors"):
            neighbor_options = self._options(neighbors)
            neighbor_options.setdefault("k", 11)
            neighbor_ref = store.query_neighbors(
                ann,
                coordinates=coordinates,
                update_state=False,
                **neighbor_options,
            )
            artifacts["neighbors"] = neighbor_ref

        with events.stage("connectivity"):
            graph = store.build_connectivity_map(
                neighbor_ref,
                update_state=False,
                **self._options(connectivity),
            )
            artifacts["connectivity_map"] = graph

        with events.stage("embedding_initialization"):
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

        with store._graph_memory_cache_scope():
            if umap is not False:
                with events.stage("umap"):
                    umap_options = self._options(umap)
                    artifacts["umap"] = store.run_umap(
                        graph,
                        from_assay=assay_name,
                        cell_key=cell_key,
                        feat_key=hvg_name,
                        **umap_options,
                    )

            leiden_options = dict(_DEFAULT_LEIDEN) if leiden is None else dict(leiden)
            paris_options = None if paris is False else self._options(paris)
            cluster_columns, cluster_artifacts = self._run_clustering_jobs(
                graph=graph,
                assay_name=assay_name,
                cell_key=cell_key,
                feat_key=hvg_name,
                leiden_options=leiden_options,
                paris_options=paris_options,
                clustering_concurrency=clustering_concurrency,
                events=events,
            )
        artifacts.update(cluster_artifacts)

        doublet_options = (
            None if doublet_scoring is False else self._options(doublet_scoring)
        )
        marker_options = None if markers is False else self._options(markers)
        selected_column: str | None = None
        if cluster_columns:
            with events.stage("cluster_selection"):
                if len(cluster_columns) == 1:
                    selected_recipe_key = next(iter(cluster_columns))
                else:
                    selected_recipe_key = self._select_clusters_by_pca_silhouette(
                        reduction=reduction,
                        cell_key=cell_key,
                        cluster_columns=cluster_columns,
                    )
                artifacts["selected_clusters"] = artifacts[selected_recipe_key]
                selected_column = self._publish_selected_clusters(
                    assay_name=assay_name,
                    cell_key=cell_key,
                    source_column=cluster_columns[selected_recipe_key],
                    ref=artifacts[selected_recipe_key],
                )

        def _cluster_column(options: dict[str, Any], step: str) -> str:
            if "clusters" in options:
                recipe_key = self._cluster_recipe_key(options.pop("clusters"))
                if recipe_key not in cluster_columns:
                    raise ValueError(
                        f"{step} cluster result {recipe_key!r} is unavailable"
                    )
                return cluster_columns[recipe_key]
            if selected_column is None:
                raise ValueError(
                    f"{step} needs a clustering result; run Leiden or Paris, "
                    "or disable this step"
                )
            return selected_column

        if doublet_options is not None:
            with events.stage("doublet_scoring"):
                options = dict(doublet_options)
                score_column = store.run_doublet_detection(
                    cluster_key=_cluster_column(options, "Doublet"),
                    from_assay=assay_name,
                    cell_key=cell_key,
                    feat_key=hvg_name,
                    **options,
                )
                artifacts["doublets"] = self._column_ref(score_column)

        if marker_options is not None:
            with events.stage("markers"):
                options = dict(marker_options)
                group_key = _cluster_column(options, "Marker")
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
        logger.info(f"Pipeline completed with {len(artifacts)} artifacts")
        return artifacts
