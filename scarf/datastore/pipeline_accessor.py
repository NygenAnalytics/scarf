from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ..storage.artifacts import ArtifactRef, artifact_group
from ..storage.pipeline_runs import (
    PipelineOutputRecord,
    abandon_pipeline_label_claim,
    complete_pipeline_run_record,
    create_pipeline_run_record,
    fail_pipeline_run_record,
    interrupt_pipeline_run_record,
    load_pipeline_run_record,
)
from ..storage.selections import (
    resolve_stored_selection_artifact,
    snapshot_run_metadata,
)
from ..storage.types import as_zarr_array
from ..utils.logging import logger
from ..utils.shutdown import (
    ShutdownToken,
    TemporarySignalGuard,
    shutdown_checkpoint,
    shutdown_scope,
)
from ._pipeline_cluster_selection import (
    cluster_label_values,
    run_cluster_selection,
)
from ._pipeline_fields import build_pipeline_fields
from ._pipeline_filtering import filter_pipeline_selection
from ._pipeline_ledger import (
    PipelineCallback,
    PipelineEvent as PipelineEvent,
    PipelineEventKind as PipelineEventKind,
    PipelineEventEmitter,
    RunLedger,
    interruption_record,
)
from ._pipeline_recipe import (
    ResolvedPipelineRecipe,
    resolve_pipeline_recipe,
)
from .pipeline_run import (
    PipelineExecutionError,
    PipelineRun,
    list_pipeline_runs,
    open_pipeline_run,
)

# Preserve the documented event type's import and pickle identity after moving
# its implementation into the ledger module.
PipelineEvent.__module__ = __name__


class PipelineAccessor:
    """Store-bound entry point for the durable basic RNA pipeline."""

    __slots__ = ("_store",)

    def __init__(self, store: Any) -> None:
        self._store = store

    def open(
        self,
        *,
        run_id: str | None = None,
        label: str | None = None,
    ) -> PipelineRun:
        """Open one durable run by ID or immutable label."""
        if (run_id is None) == (label is None):
            raise ValueError("Provide exactly one of run_id or label")
        return open_pipeline_run(self._store, run_id=run_id, label=label)

    def list_runs(
        self,
        *,
        status: str | Sequence[str] | None = None,
        limit: int = 20,
    ) -> tuple[PipelineRun, ...]:
        """List recent runs, optionally filtered by their terminal status."""
        return list_pipeline_runs(self._store, status=status, limit=limit)

    def abandon_label_claim(
        self,
        *,
        label: str,
        run_id: str,
        reason: str,
    ) -> PipelineRun:
        """Mark an exact stopped finalizer interrupted so its label can be retried.

        The caller must first confirm that the process executing ``run_id`` has
        stopped. Scarf does not infer abandonment from elapsed time because a
        slow live finalizer must never lose its label claim.
        """

        if self._store.zarr_mode != "r+":
            raise PermissionError("Label-claim recovery requires zarr_mode='r+'")
        record = abandon_pipeline_label_claim(
            self._store.zw,
            label=label,
            run_id=run_id,
            reason=reason,
        )
        return PipelineRun(self._store, record)

    def run(
        self,
        *,
        assay: str | None = None,
        label: str | None = None,
        cell_key: str = "I",
        filtering: bool | Mapping[str, object] = True,
        harmony_batch_columns: Sequence[str] | None = None,
        hvg_count: int = 1000,
        pca_dims: int = 21,
        neighbors_k: int = 11,
        umap: bool = True,
        leiden: Mapping[str, object] | bool = True,
        cell_cycle: bool = True,
        paris: bool = True,
        doublets: bool = True,
        markers: bool = True,
        snapshot_columns: Sequence[str] = (),
        callback: PipelineCallback | None = None,
    ) -> PipelineRun:
        """Run the validated rich RNA recipe and return its durable handle."""
        if callback is not None and not callable(callback):
            raise TypeError("callback must be callable")
        store = self._store
        if store.zarr_mode != "r+":
            raise PermissionError("Pipeline execution requires zarr_mode='r+'")
        recipe = resolve_pipeline_recipe(
            store,
            assay=assay,
            label=label,
            cell_key=cell_key,
            filtering=filtering,
            harmony_batch_columns=harmony_batch_columns,
            hvg_count=hvg_count,
            pca_dims=pca_dims,
            neighbors_k=neighbors_k,
            umap=umap,
            leiden=leiden,
            cell_cycle=cell_cycle,
            paris=paris,
            doublets=doublets,
            markers=markers,
            snapshot_columns=snapshot_columns,
        )
        return self._run_recipe(recipe, callback)

    def _run_recipe(
        self,
        recipe: ResolvedPipelineRecipe,
        callback: PipelineCallback | None,
    ) -> PipelineRun:
        token = ShutdownToken()
        active_run_id: list[str] = []
        result: PipelineRun | None = None
        try:
            with TemporarySignalGuard(token) as guard, shutdown_scope(token):
                try:
                    result = self._execute_recipe(
                        recipe,
                        callback,
                        signal_guard_available=guard.available,
                        signal_guard_unavailable_reason=guard.unavailable_reason,
                        active_run_id=active_run_id,
                    )
                except BaseException as error:
                    interruption = interruption_record(error)
                    if interruption is not None and active_run_id:
                        current = load_pipeline_run_record(
                            self._store.zw,
                            active_run_id[0],
                        )
                        if not current.complete:
                            interrupt_pipeline_run_record(
                                self._store.zw,
                                run_id=active_run_id[0],
                                interruption=interruption,
                            )
                            PipelineEventEmitter(callback).emit(
                                "pipeline_interrupted",
                                "between_stages",
                                error,
                            )
                    raise
        finally:
            if token.requested:
                token.propagate()
        assert result is not None
        return result

    def _execute_recipe(
        self,
        recipe: ResolvedPipelineRecipe,
        callback: PipelineCallback | None,
        *,
        signal_guard_available: bool,
        signal_guard_unavailable_reason: str | None,
        active_run_id: list[str],
    ) -> PipelineRun:
        store = self._store
        shutdown_checkpoint()
        from .. import __version__

        assay_obj = store._get_assay(recipe.assay)
        config = recipe.to_config()
        config["shutdown"] = {
            "signalGuardAvailable": signal_guard_available,
            "unavailableReason": signal_guard_unavailable_reason,
        }
        record = create_pipeline_run_record(
            store.zw,
            recipe="basic_rna_analysis",
            requested_label=recipe.label,
            assay=recipe.assay,
            config=config,
            stage_order=recipe.stage_order,
            scarf_version=__version__,
        )
        active_run_id.append(record.run_id)
        ledger = RunLedger(store.zw, record.run_id, callback)
        artifacts: dict[str, ArtifactRef] = {}
        cell_snapshot: ArtifactRef
        feature_snapshot: ArtifactRef
        frozen_feature_names: np.ndarray
        all_features: ArtifactRef

        def input_snapshot_stage() -> Sequence[tuple[str, ArtifactRef]]:
            nonlocal cell_snapshot, feature_snapshot, frozen_feature_names, all_features
            input_selection = resolve_stored_selection_artifact(
                store.zw,
                table_path="cellData",
                id_column="ids",
                source_column=recipe.cell_key,
                scope="datastore",
                kind="cell_selection",
                operation="snapshot_pipeline_input_selection",
                parameters={"assay": recipe.assay},
                inputs={},
            )
            cell_snapshot = snapshot_run_metadata(
                store.zw,
                table_path="cellData",
                id_column="ids",
                columns=recipe.cell_snapshot_columns,
                axis="cell",
            )
            feature_snapshot = snapshot_run_metadata(
                store.zw,
                table_path=f"{recipe.assay}/featureData",
                id_column="ids",
                columns=("names",),
                axis="feature",
                assay=recipe.assay,
            )
            all_features = store.select_all_features(from_assay=assay_obj.name)
            frozen_feature_names = np.asarray(
                as_zarr_array(
                    artifact_group(store.zw, feature_snapshot)["names"],
                    name="names",
                )[:]
            )
            artifacts["input_cell_selection"] = input_selection
            artifacts["feature_universe"] = all_features
            return (
                ("input_cell_selection", input_selection),
                ("cell_snapshot", cell_snapshot),
                ("feature_snapshot", feature_snapshot),
                ("feature_universe", all_features),
            )

        ledger.run("input_snapshot", input_snapshot_stage)

        if recipe.filtering["enabled"]:

            def filtering_stage() -> Sequence[tuple[str, ArtifactRef]]:
                ref = filter_pipeline_selection(
                    store,
                    recipe=recipe,
                    input_selection=artifacts["input_cell_selection"],
                    cell_snapshot=cell_snapshot,
                )
                artifacts["analysis_cell_selection"] = ref
                return (("analysis_cell_selection", ref),)

            ledger.run("filtering", filtering_stage)
        else:
            artifacts["analysis_cell_selection"] = artifacts["input_cell_selection"]
            ledger.skip("filtering")
        analysis_selection = artifacts["analysis_cell_selection"]

        if recipe.cell_cycle:

            def cell_cycle_stage() -> Sequence[tuple[str, ArtifactRef]]:
                ref = store._run_cell_cycle_scoring_artifact(
                    assay=assay_obj,
                    cell_selection=analysis_selection,
                    feature_names=frozen_feature_names,
                    feature_snapshot=feature_snapshot,
                )
                artifacts["cell_cycle"] = ref
                return (("cell_cycle", ref),)

            ledger.run("cell_cycle", cell_cycle_stage)
        else:
            ledger.skip("cell_cycle")

        def hvg_stage() -> Sequence[tuple[str, ArtifactRef]]:
            hvg = store._select_hvgs_artifact(
                assay=assay_obj,
                cell_selection=analysis_selection,
                feature_names=frozen_feature_names,
                feature_snapshot=feature_snapshot,
                top_n=recipe.hvg_count,
                show_plot=False,
            )
            artifacts["highly_variable_features"] = hvg
            return (("highly_variable_features", hvg),)

        ledger.run("highly_variable_features", hvg_stage)

        def normalization_stage() -> Sequence[tuple[str, ArtifactRef]]:
            ref = store.run_normalization(
                analysis_selection,
                artifacts["highly_variable_features"],
            )
            artifacts["normalized"] = ref
            return (("normalized", ref),)

        ledger.run("normalization", normalization_stage)

        def pca_stage() -> Sequence[tuple[str, ArtifactRef]]:
            ref = store.run_pca(artifacts["normalized"], dims=recipe.pca_dims)
            artifacts["pca"] = ref
            return (("pca", ref),)

        ledger.run("pca", pca_stage)
        coordinates = artifacts["pca"]
        if recipe.harmony_batch_columns:

            def harmony_stage() -> Sequence[tuple[str, ArtifactRef]]:
                ref = store._run_harmony_artifact(
                    artifacts["pca"],
                    cell_snapshot,
                    list(recipe.harmony_batch_columns),
                )
                artifacts["harmony"] = ref
                return (("harmony", ref),)

            ledger.run("harmony", harmony_stage)
            coordinates = artifacts["harmony"]
        else:
            ledger.skip("harmony")

        def ann_stage() -> Sequence[tuple[str, ArtifactRef]]:
            ref = store.build_ann_index(coordinates)
            artifacts["ann_index"] = ref
            return (("ann_index", ref),)

        ledger.run("ann_index", ann_stage)

        def neighbors_stage() -> Sequence[tuple[str, ArtifactRef]]:
            ref = store.query_neighbors(
                artifacts["ann_index"],
                coordinates=coordinates,
                k=recipe.neighbors_k,
            )
            artifacts["neighbors"] = ref
            return (("neighbors", ref),)

        ledger.run("neighbors", neighbors_stage)

        def connectivity_stage() -> Sequence[tuple[str, ArtifactRef]]:
            ref = store.build_connectivity_map(artifacts["neighbors"])
            artifacts["connectivity_map"] = ref
            return (("connectivity_map", ref),)

        ledger.run("connectivity", connectivity_stage)

        if recipe.umap:

            def initialization_stage() -> Sequence[tuple[str, ArtifactRef]]:
                ref = store.build_embedding_initialization(coordinates)
                artifacts["embedding_initialization"] = ref
                return (("embedding_initialization", ref),)

            ledger.run("embedding_initialization", initialization_stage)

            def umap_stage() -> Sequence[tuple[str, ArtifactRef]]:
                ref = store._run_umap_artifact(
                    artifacts["connectivity_map"],
                    artifacts["embedding_initialization"],
                )
                artifacts["umap"] = ref
                return (("umap", ref),)

            ledger.run("umap", umap_stage)
        else:
            ledger.skip("embedding_initialization")
            ledger.skip("umap")

        for key, resolution in recipe.leiden_partitions:
            output_key = f"leiden_{key}"

            def leiden_stage(
                output_key: str = output_key,
                resolution: float = resolution,
            ) -> Sequence[tuple[str, ArtifactRef]]:
                ref = store._run_leiden_artifact(
                    artifacts["connectivity_map"],
                    resolution=resolution,
                )
                artifacts[output_key] = ref
                return ((output_key, ref),)

            ledger.run(output_key, leiden_stage)

        if recipe.paris:

            def paris_stage() -> Sequence[tuple[str, ArtifactRef]]:
                ref = store._run_paris_artifact(artifacts["connectivity_map"])
                artifacts["paris"] = ref
                return (("paris", ref),)

            ledger.run("paris", paris_stage)
        else:
            ledger.skip("paris")

        clustering_candidates = [
            (f"leiden_{key}", artifacts[f"leiden_{key}"])
            for key, _resolution in recipe.leiden_partitions
        ]
        if clustering_candidates:

            def cluster_selection_stage() -> Sequence[tuple[str, ArtifactRef]]:
                decision, selected_key, selected_ref = run_cluster_selection(
                    store,
                    coordinates=coordinates,
                    connectivity_map=artifacts["connectivity_map"],
                    cell_selection=analysis_selection,
                    candidates=clustering_candidates,
                )
                artifacts["cluster_selection"] = decision
                artifacts["clusters"] = selected_ref
                logger.info(f"Selected clustering candidate: {selected_key}")
                return (("cluster_selection", decision),)

            ledger.run("cluster_selection", cluster_selection_stage)
        else:
            ledger.skip("cluster_selection")

        doublet_graph = artifacts["connectivity_map"]
        if recipe.doublets and recipe.harmony_batch_columns:

            def doublet_graph_stage() -> Sequence[tuple[str, ArtifactRef]]:
                nonlocal doublet_graph
                ann = store.build_ann_index(artifacts["pca"])
                neighbors = store.query_neighbors(
                    ann,
                    coordinates=artifacts["pca"],
                    k=recipe.neighbors_k,
                )
                graph = store.build_connectivity_map(neighbors)
                doublet_graph = graph
                return (
                    ("uncorrected_ann_index", ann),
                    ("uncorrected_neighbors", neighbors),
                    ("uncorrected_connectivity_map", graph),
                )

            ledger.run("doublet_graph", doublet_graph_stage)
        else:
            ledger.skip("doublet_graph")

        if recipe.doublets:

            def doublet_stage() -> Sequence[tuple[str, ArtifactRef]]:
                clusters = artifacts["clusters"]
                ref = store._run_doublet_detection_artifact(
                    source_assay=assay_obj,
                    cell_selection=analysis_selection,
                    clusters=clusters,
                    cluster_values=cluster_label_values(store.zw, clusters),
                    connectivity=doublet_graph,
                    feature_names=frozen_feature_names,
                    feature_snapshot=feature_snapshot,
                )
                artifacts["doublets"] = ref
                return (("doublets", ref),)

            ledger.run("doublets", doublet_stage)
        else:
            ledger.skip("doublets")

        if recipe.markers:

            def marker_stage() -> Sequence[tuple[str, ArtifactRef]]:
                clusters = artifacts["clusters"]
                ref = store._run_marker_search_artifact(
                    assay=assay_obj,
                    cell_selection=analysis_selection,
                    clusters=clusters,
                    cluster_values=cluster_label_values(store.zw, clusters),
                    feature_selection=all_features,
                    feature_names=frozen_feature_names,
                    feature_snapshot=feature_snapshot,
                )
                artifacts["markers"] = ref
                return (("markers", ref),)

            ledger.run("markers", marker_stage)
        else:
            ledger.skip("markers")

        ordered_keys = (
            "input_cell_selection",
            "analysis_cell_selection",
            "feature_universe",
            "cell_cycle",
            "highly_variable_features",
            "normalized",
            "pca",
            "harmony",
            "ann_index",
            "neighbors",
            "connectivity_map",
            "embedding_initialization",
            "umap",
            *(f"leiden_{key}" for key, _value in recipe.leiden_partitions),
            "paris",
            "cluster_selection",
            "clusters",
            "doublets",
            "markers",
        )
        outputs = tuple(
            PipelineOutputRecord(key=key, artifact=artifacts[key])
            for key in ordered_keys
            if key in artifacts
        )
        try:
            fields = build_pipeline_fields(
                store,
                recipe,
                artifacts,
                cell_snapshot=cell_snapshot,
                feature_snapshot=feature_snapshot,
            )
            shutdown_checkpoint()
            complete_pipeline_run_record(
                store.zw,
                run_id=record.run_id,
                outputs=outputs,
                fields=fields,
            )
            shutdown_checkpoint()
        except Exception as error:
            current = load_pipeline_run_record(store.zw, record.run_id)
            if not current.complete:
                fail_pipeline_run_record(
                    store.zw,
                    run_id=record.run_id,
                    error=error,
                )
            raise PipelineExecutionError(record.run_id, "finalize", error) from error
        return open_pipeline_run(store, run_id=record.run_id)
