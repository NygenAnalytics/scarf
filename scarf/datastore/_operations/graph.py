import os
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast
from weakref import WeakKeyDictionary

import numpy as np
import pandas as pd
import zarr
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, csr_matrix

from ...embeddings.harmony import HarmonyResult
from ...storage.types import as_zarr_array, as_zarr_group
from ...assay import Assay, RNAassay
from ...graph.build import (
    GraphBuildPlan,
    GraphDataInputs,
    GraphExecutionOptions,
    ResolvedGraphParameters,
)
from ...graph.arguments import (
    AnnIndexArguments,
    ConnectivityMapArguments,
    CustomReductionArguments,
    EmbeddingInitializationArguments,
    FeatureScalingArguments,
    HarmonyArguments,
    LsiArguments,
    NeighborQueryArguments,
    NormalizationArguments,
    PcaArguments,
)
from ...graph.encoded_paths import (
    is_integrated_graph_path,
    lookup_latest_assay_graph,
    lookup_latest_cell_graph_group_path,
    lookup_latest_kmeans_group_path,
    lookup_latest_neighbor_index_group_path,
    lookup_latest_nearest_neighbor_paths,
    lookup_latest_nearest_neighbors_group_path,
    lookup_latest_reduction_group_path,
    lookup_stored_integrated_graph,
    make_integrated_graph_path,
    make_nearest_neighbors_group_path,
    make_neighbor_index_group_path,
    make_normalized_group_path,
    make_reduction_group_path,
    nearest_neighbor_paths_from_loc,
    nearest_neighbors_group_path_from_cell_graph,
    parse_assay_graph_paths,
    parse_cell_graph_group_path,
    parse_kmeans_group_path,
    parse_nearest_neighbors_group_path,
    parse_neighbor_index_group_path,
    parse_reduction_group_path,
)
from ...graph.paths import StoredAssayGraph, StoredGraph
from ...graph.state import (
    AssayState,
    normalized_path_from_state,
    read_assay_state,
    stored_assay_graph_from_ref,
    stored_assay_graph_from_state,
    validate_artifact_graph_selection,
    validate_cell_selection_artifact,
    validate_legacy_graph_selection,
    validate_normalized_artifact_selection,
    write_assay_state,
)
from ...matrix import ChunkedArray
from ...mapping.reference import MappingReference
from ...metadata.artifacts import (
    categorical_display,
    link_feature_data_column,
)
from ...neighbors.stages import (
    AnnIndexStage,
    BatchCorrectionStage,
    ChunkedCoordinateStream,
    KMeansInitializationStage,
    LazyTransformStream,
    NeighborQueryStage,
    ReductionTransform,
)
from ...neighbors.stream import AnnStream
from ...storage.ann_index import (
    has_ann_index,
    legacy_ann_index_path,
    load_ann_index,
    load_ann_index_from_path,
    save_ann_index,
)
from ...storage.arrays import create_zarr_dataset
from ...storage.artifact_writer import (
    ArrayRequirement,
    AttributeRequirement,
    PlannedArtifact,
    finish_artifact,
    plan_artifact,
    reused_artifact_group,
    start_artifact,
)
from ...storage.artifacts import (
    ArtifactRef,
    ArtifactScope,
    artifact_path,
    fingerprint_array,
    fingerprint_stored_arrays,
    fingerprint_strings,
    inspect_artifact,
    parse_artifact_path,
)
from ...storage.copy import (
    copy_zarr_array,
    create_or_open_staged_normed_array,
)
from ...storage.stores import is_remote_datastore, zarr_root_path
from ...storage.selections import (
    resolve_metadata_snapshot,
    resolve_selection_artifact,
)
from ...utils.arrays import clean_array
from ...utils.compute import show_dask_progress
from ...utils.logging import logger

if TYPE_CHECKING:
    from ..base_datastore import BaseDataStore as _GraphOperationsBase
else:
    _GraphOperationsBase = object


EMBEDDING_CACHE_MAX_BYTES = 256 * 1024 * 1024


class _GraphOperationsMixin(_GraphOperationsBase):
    _annStreamPaths: WeakKeyDictionary[AnnStream, str]
    _annStreamNeighborPaths: WeakKeyDictionary[AnnStream, str]
    _normalizedArtifactCache: dict[ArtifactRef, ChunkedArray]
    _artifactExecutionContext: dict[str, Any]

    def _remember_ann_stream_path(self, ann_obj: AnnStream, path: str) -> None:
        try:
            paths = self._annStreamPaths
        except AttributeError:
            paths = WeakKeyDictionary()
            self._annStreamPaths = paths
        paths[ann_obj] = path

    def _ann_stream_path(self, ann_obj: object) -> str | None:
        try:
            paths = self._annStreamPaths
        except AttributeError:
            paths = None
        path = (
            paths.get(ann_obj)
            if paths is not None and isinstance(ann_obj, AnnStream)
            else None
        )
        if path is not None:
            return path
        legacy_path: object = getattr(ann_obj, "annPath", None)
        return legacy_path if isinstance(legacy_path, str) else None

    def _remember_ann_stream_neighbors(
        self,
        ann_obj: AnnStream,
        path: str,
    ) -> None:
        try:
            paths = self._annStreamNeighborPaths
        except AttributeError:
            paths = WeakKeyDictionary()
            self._annStreamNeighborPaths = paths
        paths[ann_obj] = path

    def _ann_stream_neighbors_path(self, ann_obj: object) -> str | None:
        try:
            paths = self._annStreamNeighborPaths
        except AttributeError:
            return None
        return paths.get(ann_obj) if isinstance(ann_obj, AnnStream) else None

    def _mapping_reference_metadata(
        self,
        assay: Assay,
        from_assay: str,
        cell_key: str,
        feat_key: str,
        reduction_loc: str,
        ann_loc: str,
        batch_columns: list[str],
    ) -> dict[str, Any]:
        from ...mapping.hashing import array_hash, array_store_hash

        normed_loc = make_normalized_group_path(from_assay, cell_key, feat_key)
        normed_group = as_zarr_group(self.zw[normed_loc], name=normed_loc)
        reduction_group = as_zarr_group(self.zw[reduction_loc], name=reduction_loc)
        feature_key = f"{cell_key}__{feat_key}" if feat_key != "I" else "I"
        feature_ids = assay.feats.fetch("ids", key=feature_key)
        batch_values = (
            np.column_stack(
                [self.cells.fetch(column, key=cell_key) for column in batch_columns]
            )
            if batch_columns
            else np.empty((len(self.cells.active_index(cell_key)), 0), dtype=str)
        )
        loadings = np.asarray(
            as_zarr_array(reduction_group["reduction"], name="reduction")[:]
        )
        corrected_hash = ""
        if "harmonizedData" in reduction_group:
            corrected_hash = array_store_hash(
                as_zarr_array(reduction_group["harmonizedData"], name="harmonizedData")
            )
        ann_group = as_zarr_group(self.zw[ann_loc], name=ann_loc)
        (
            ann_metric,
            ann_efc,
            ann_ef,
            ann_m,
            ann_rand_state,
            _,
            _,
        ) = parse_neighbor_index_group_path(ann_loc)
        return {
            "assay": from_assay,
            "cellKey": cell_key,
            "featureKey": feat_key,
            "reductionPath": reduction_loc,
            "annPath": ann_loc,
            "featureHash": array_hash(feature_ids),
            "cellHash": array_hash(self.cells.fetch("ids", key=cell_key)),
            "batchValueHash": array_hash(batch_values),
            "batchColumns": batch_columns,
            "subsetHash": normed_group.attrs["subset_hash"],
            "subsetParams": normed_group.attrs["subset_params"],
            "loadingsHash": array_hash(loadings),
            "correctedCoordinatesHash": corrected_hash,
            "reductionMethod": "pca",
            "annContract": {
                "metric": ann_metric,
                "efConstruction": ann_efc,
                "ef": ann_ef,
                "m": ann_m,
                "randomState": ann_rand_state,
                "featureScaling": bool(ann_group.attrs.get("featureScaling", True)),
            },
        }

    def _persist_mapping_reference(
        self,
        assay: Assay,
        from_assay: str,
        cell_key: str,
        feat_key: str,
        reduction_loc: str,
        ann_loc: str,
        knn_loc: str,
        batch_columns: list[str],
        ann_obj: AnnStream,
        mapping_artifact: PlannedArtifact,
        normalized_ref: ArtifactRef,
    ) -> None:
        from ...mapping.artifact import write_artifact_mapping_reference
        from ...mapping.confidence import _distance_quantile_summary
        from ...mapping.models import SymphonyReferenceModel
        from ...mapping.symphony import weighted_centroids

        if ann_obj.harmonyResult is None:
            return
        if ann_obj.loadings is None:
            raise RuntimeError(
                "Cannot persist a mapping reference without PCA loadings"
            )
        harmony = ann_obj.harmonyResult
        try:
            cluster_mass, raw_centroids = weighted_centroids(
                harmony.original.T, harmony.assignments
            )
            _, corrected_centroids = weighted_centroids(
                harmony.corrected.T, harmony.assignments
            )
        except ValueError as exc:
            if "empty cluster" not in str(exc):
                raise
            raise ValueError(
                "Harmony produced an empty reference cluster. Rebuild with a "
                "smaller harmony_params['nclust'] value."
            ) from exc
        ridge_values = np.diag(harmony.ridge)[1:]
        correction_ridge = (
            float(np.mean(ridge_values[ridge_values > 0]))
            if np.any(ridge_values > 0)
            else 1.0
        )
        model = SymphonyReferenceModel(
            feature_means=ann_obj.mu,
            feature_scales=ann_obj.sigma,
            loadings=ann_obj.loadings,
            centroids=harmony.centroids.T,
            raw_centroids=raw_centroids,
            corrected_centroids=corrected_centroids,
            cluster_mass=cluster_mass,
            sigma=harmony.sigma,
            correction_ridge=correction_ridge,
        )
        feature_key = f"{cell_key}__{feat_key}" if feat_key != "I" else "I"
        feature_ids = assay.feats.fetch("ids", key=feature_key)
        metadata = {
            "assay": from_assay,
            "cell_key": cell_key,
            "feature_key": feat_key,
            "batch_columns": batch_columns,
            "harmony_parameters": harmony.parameters,
            "batch_levels": [list(levels) for levels in harmony.batch_levels],
            "method": "symphony",
            "normalization_parameters": (
                inspect_artifact(self.zw, normalized_ref).parameters or {}
            ),
        }
        knn_group = as_zarr_group(self.zw[knn_loc], name=knn_loc)
        distance_quantiles, distance_values = _distance_quantile_summary(
            as_zarr_array(knn_group["distances"], name="distances")
        )
        group = start_artifact(self.zw, mapping_artifact)
        write_artifact_mapping_reference(
            group,
            model,
            feature_ids,
            metadata,
            reference_distance_quantiles=distance_quantiles,
            reference_distance_values=distance_values,
        )
        finish_artifact(group, mapping_artifact)

    def _restore_harmony_result(
        self,
        ann_obj: AnnStream,
        correction_ref: ArtifactRef,
    ) -> None:
        if ann_obj.harmonyResult is not None:
            return
        if ann_obj.harmonizedData is None:
            raise RuntimeError("Stored Harmony coordinates are missing")
        group = as_zarr_group(
            self.zw[artifact_path(correction_ref)],
            name=artifact_path(correction_ref),
        )
        required = ("assignments", "centroids", "sigma", "ridge")
        if any(name not in group for name in required):
            raise RuntimeError("Stored Harmony model metadata is incomplete")
        status = inspect_artifact(self.zw, correction_ref)
        parameters = status.parameters or {}
        batch_columns = tuple(
            str(column) for column in parameters.get("batch_columns", [])
        )
        batch_levels_raw = group.attrs.get("batch_levels", [])
        batch_levels = tuple(
            tuple(str(value) for value in levels)
            for levels in cast(list[list[Any]], batch_levels_raw)
        )
        ann_obj.harmonyResult = HarmonyResult(
            original=np.vstack(
                ann_obj._reduced_blocks(
                    "Restoring uncorrected latent dimensions",
                )
            ).T,
            corrected=ann_obj.harmonizedData.compute().T,
            assignments=np.asarray(
                as_zarr_array(group["assignments"], name="assignments")[:]
            ),
            centroids=np.asarray(
                as_zarr_array(group["centroids"], name="centroids")[:]
            ),
            sigma=np.asarray(as_zarr_array(group["sigma"], name="sigma")[:]),
            ridge=np.asarray(as_zarr_array(group["ridge"], name="ridge")[:]),
            batch_columns=batch_columns,
            batch_levels=batch_levels,
            parameters=dict(parameters.get("harmony_parameters", {})),
        )

    def get_mapping_reference(
        self,
        from_assay: str | None = None,
        cell_key: str | None = None,
        feat_key: str | None = None,
    ) -> MappingReference:
        """Load a validated immutable RNA/PCA Symphony-style mapping reference."""
        from ...mapping.artifact import (
            load_mapping_reference,
            resolve_mapping_reference_group,
        )

        from_assay, cell_key, feat_key = self._get_latest_keys(
            from_assay, cell_key, feat_key
        )
        assay = self._get_assay(from_assay)
        if not isinstance(assay, RNAassay):
            raise TypeError("Mapping references currently support RNA assays only")
        state = read_assay_state(self.zw, from_assay)
        if state is not None and state.matches(cell_key, feat_key):
            mapping_ref = state.named_results.get("mapping_reference")
            if mapping_ref is None:
                raise KeyError(
                    "AssayState has no mapping reference for the selected graph"
                )
            from ...mapping.artifact import load_artifact_mapping_reference

            return load_artifact_mapping_reference(
                self,
                mapping_ref,
                from_assay,
                cell_key,
                feat_key,
            )
        normed_loc = make_normalized_group_path(from_assay, cell_key, feat_key)
        if normed_loc not in self.zw:
            raise KeyError("No normalized reference data exists for the requested keys")
        try:
            reduction_loc = lookup_latest_reduction_group_path(self.zw, normed_loc)
        except KeyError:
            reduction_loc = ""
        if not reduction_loc or reduction_loc not in self.zw:
            raise KeyError("No reduction exists for the requested reference")
        reduction_group = as_zarr_group(self.zw[reduction_loc], name=reduction_loc)
        try:
            ann_loc = lookup_latest_neighbor_index_group_path(self.zw, reduction_loc)
        except KeyError:
            ann_loc = ""
        if not ann_loc or ann_loc not in self.zw:
            raise KeyError("No ANN index exists for the requested reference")
        try:
            artifact, _, is_legacy = resolve_mapping_reference_group(reduction_group)
        except KeyError:
            raise ValueError(
                "This harmonized graph predates the mapping-reference artifact. "
                "Rebuild it with build_mapping_reference."
            ) from None
        artifact_ann_path = artifact.attrs.get("annPath")
        if isinstance(artifact_ann_path, str):
            ann_loc = artifact_ann_path
        if ann_loc not in self.zw:
            raise ValueError(
                "The mapping-reference ANN index is missing. Rebuild the reference."
            )
        ann_group = as_zarr_group(self.zw[ann_loc], name=ann_loc)
        if not bool(ann_group.attrs.get("isHarmonized", False)):
            raise ValueError(
                "The requested graph is not harmonized. Build a harmonized mapping reference first."
            )
        batch_columns = [
            str(column) for column in cast(list[Any], artifact.attrs["batchColumns"])
        ]
        expected = self._mapping_reference_metadata(
            assay,
            from_assay,
            cell_key,
            feat_key,
            reduction_loc,
            ann_loc,
            batch_columns,
        )
        legacy_omissions = {"correctedCoordinatesHash", "annContract"}
        for key, value in expected.items():
            if is_legacy and key in legacy_omissions:
                continue
            if artifact.attrs.get(key) != value:
                raise ValueError(
                    f"Mapping reference is stale because {key} no longer matches. "
                    "Rebuild it with build_mapping_reference."
                )
        return load_mapping_reference(
            self, from_assay, cell_key, feat_key, reduction_loc, ann_loc
        )

    def build_mapping_reference(
        self,
        from_assay: str | None = None,
        cell_key: str | None = None,
        feat_key: str | None = None,
        batch_columns: list[str] | None = None,
        **graph_kwargs: Any,
    ) -> MappingReference:
        """Build and return an RNA/PCA Symphony mapping reference."""
        if self.zarr_mode != "r+":
            raise ValueError("Building a mapping reference requires a read-write store")
        if batch_columns is None:
            raise ValueError("batch_columns is required to build a mapping reference")
        if graph_kwargs.get("feat_scaling", True) is False:
            raise ValueError(
                "Mapping references require feat_scaling=True because query "
                "projection uses the stored reference mean and scale."
            )
        reduction_method = graph_kwargs.get("reduction_method", "pca")
        if reduction_method not in {"auto", "pca"}:
            raise ValueError("Mapping references require PCA reduction")
        force_harmony_refit = False
        try:
            current = self.get_mapping_reference(from_assay, cell_key, feat_key)
        except (KeyError, ValueError):
            force_harmony_refit = True
        else:
            current_columns = [
                str(column)
                for column in cast(
                    list[Any],
                    current.metadata.get(
                        "batch_columns",
                        current.metadata.get("batchColumns", []),
                    ),
                )
            ]
            force_harmony_refit = (
                current_columns != batch_columns
                or not bool(current.metadata.get("complete", False))
                or (
                    "artifact_id" not in current.metadata
                    and "artifactHash" not in current.metadata
                )
            )
        graph_kwargs["reduction_method"] = "pca"
        graph_kwargs["feat_scaling"] = True
        plan = self._resolve_graph_plan(
            from_assay=from_assay,
            cell_key=cell_key,
            feat_key=feat_key,
            harmonize=True,
            batch_columns=batch_columns,
            force_harmony_refit=force_harmony_refit,
            **graph_kwargs,
        )
        self._run_resolved_graph_plan(plan)
        reference = self.get_mapping_reference(from_assay, cell_key, feat_key)
        if not bool(reference.metadata.get("complete", False)):
            raise RuntimeError(
                "Mapping reference build did not produce a complete artifact"
            )
        return reference

    @staticmethod
    def _choose_reduction_method(assay: Assay, reduction_method: str) -> str:
        """This is a convenience function to determine the linear dimension
        reduction method to be used for a given assay. It is uses a
        predetermined rule to make this determination.

        Args:
            assay: Assay object.
            reduction_method: Name of reduction method to use. It can be one from either: 'pca', 'lsi', 'auto'.

        Returns:
            The name of dimension reduction method to be used. Either 'pca' or 'lsi'

        Raises:
            ValueError: If `reduction_method` is not one of either 'pca', 'lsi', 'auto'
        """
        reduction_method = reduction_method.lower()
        if reduction_method not in ["pca", "lsi", "auto", "custom"]:
            raise ValueError(
                "ERROR: Please choose either 'pca' or 'lsi' as reduction method"
            )
        if reduction_method == "auto":
            assay_type = str(assay.__class__).split(".")[-1][:-2]
            if assay_type == "ATACassay":
                logger.debug("Using LSI for dimension reduction")
                reduction_method = "lsi"
            else:
                logger.debug("Using PCA for dimension reduction")
                reduction_method = "pca"
        return reduction_method

    def _resolve_graph_parameters(
        self,
        from_assay: str,
        cell_key: str,
        feat_key: str,
        log_transform: bool | None = None,
        renormalize_subset: bool | None = None,
        reduction_method: str = "auto",
        dims: int | None = None,
        pca_cell_key: str | None = None,
        ann_metric: str | None = None,
        ann_efc: int | None = None,
        ann_ef: int | None = None,
        ann_m: int | None = None,
        rand_state: int | None = None,
        k: int | None = None,
        n_centroids: int | None = None,
        local_connectivity: float | None = None,
        bandwidth: float | None = None,
        feat_scaling: bool = True,
        lsi_skip_first: bool = True,
        harmonize: bool = False,
        batch_columns: list[str] | None = None,
        harmony_params: dict[str, Any] | None = None,
    ) -> ResolvedGraphParameters:
        """Resolve graph parameters from explicit, cached, and default values.

        Args:
            from_assay: Same as from_assay in make_graph
            cell_key: Same as cell_key in make_graph
            feat_key: Same as feat_key in make_graph
            log_transform: Same as log_transform in make_graph
            renormalize_subset: Same as renormalize_subset in make_graph
            reduction_method: Same as reduction_method in make_graph
            dims: Same as dims in make_graph
            pca_cell_key: Same as pca_cell_key in make_graph
            ann_metric: Same as ann_metric in make_graph
            ann_efc: Same as ann_efc in make_graph
            ann_ef: Same as ann_ef in make_graph
            ann_m: Same as ann_m in make_graph
            rand_state: Same as rand_state in make_graph
            k: Same as k in make_graph
            n_centroids: Same as n_centroids in make_graph
            local_connectivity: Same as local_connectivity in make_graph
            bandwidth: Same as bandwidth in make_graph

        Returns:
            The complete resolved parameter set.
        """

        def log_message(
            category: str,
            name: str,
            value: Any,
            custom_msg: str | None = None,
        ) -> bool:
            """Convenience function to log variable usage messages for
            make_graph.

            Args:
                category:
                name:
                value:
                custom_msg:

            Returns:
            """
            msg = f"No value provided for parameter `{name}`. "
            if category == "default":
                msg += f"Will use default value: {value}"
                logger.debug(msg)
            elif category == "cached":
                msg += f"Will use previously used value: {value}"
                logger.debug(msg)
            else:
                if custom_msg is None:
                    return False
                else:
                    logger.info(custom_msg)
            return True

        default_values: dict[str, Any] = {
            "log_transform": True,
            "renormalize_subset": True,
            "dims": 11,
            "ann_metric": "l2",
            "rand_state": 4466,
            "k": 11,
            "n_centroids": 1000,
            "local_connectivity": 1.0,
            "bandwidth": 1.5,
        }
        state = read_assay_state(self.zw, from_assay)
        if state is not None and state.matches(cell_key, feat_key):
            reduction_status = (
                inspect_artifact(self.zw, state.reduction)
                if state.reduction is not None
                else None
            )
            state_normalized = (
                inspect_artifact(self.zw, state.normalized).parameters or {}
                if state.normalized is not None
                else {}
            )
            state_reduction = (
                reduction_status.parameters or {}
                if reduction_status is not None
                else {}
            )
            state_reduction_execution = (
                reduction_status.execution_options or {}
                if reduction_status is not None
                else {}
            )
            state_ann = (
                inspect_artifact(self.zw, state.ann_index).parameters or {}
                if state.ann_index is not None
                else {}
            )
            state_neighbors = (
                inspect_artifact(self.zw, state.neighbors).parameters or {}
                if state.neighbors is not None
                else {}
            )
            state_connectivity = (
                inspect_artifact(self.zw, state.connectivity_map).parameters or {}
                if state.connectivity_map is not None
                else {}
            )
            state_initialization = (
                inspect_artifact(
                    self.zw,
                    state.embedding_initialization,
                ).parameters
                or {}
                if state.embedding_initialization is not None
                else {}
            )
            state_correction = (
                inspect_artifact(self.zw, state.batch_correction).parameters or {}
                if state.batch_correction is not None
                else {}
            )
        else:
            state_normalized = {}
            state_reduction = {}
            state_reduction_execution = {}
            state_ann = {}
            state_neighbors = {}
            state_connectivity = {}
            state_initialization = {}
            state_correction = {}

        normed_loc = make_normalized_group_path(from_assay, cell_key, feat_key)
        if log_transform is None or renormalize_subset is None:
            if state_normalized:
                c_log_transform = state_normalized.get("log_transform")
                c_renormalize_subset = state_normalized.get("renormalize_subset")
            elif normed_loc in self.zw:
                normed_grp = as_zarr_group(self.zw[normed_loc], name=normed_loc)
                if "subset_params" in normed_grp.attrs:
                    # This works in coordination with save_normalized_data
                    subset_params = cast(
                        dict[str, Any], normed_grp.attrs["subset_params"]
                    )
                    c_log_transform, c_renormalize_subset = (
                        subset_params["log_transform"],
                        subset_params["renormalize_subset"],
                    )
                else:
                    c_log_transform, c_renormalize_subset = None, None
            else:
                c_log_transform, c_renormalize_subset = None, None
            if log_transform is None:
                if c_log_transform is not None:
                    log_transform = bool(c_log_transform)
                    log_message("cached", "log_transform", log_transform)
                else:
                    log_transform = default_values["log_transform"]
                    log_message("default", "log_transform", log_transform)
            if renormalize_subset is None:
                if c_renormalize_subset is not None:
                    renormalize_subset = bool(c_renormalize_subset)
                    log_message("cached", "renormalize_subset", renormalize_subset)
                else:
                    renormalize_subset = default_values["renormalize_subset"]
                    log_message("default", "renormalize_subset", renormalize_subset)
        log_transform = bool(log_transform)
        renormalize_subset = bool(renormalize_subset)

        if dims is None or pca_cell_key is None:
            if state_reduction:
                c_dims = state_reduction.get("dims")
                c_pca_cell_key = state_reduction_execution.get("pca_cell_key")
            elif normed_loc in self.zw:
                try:
                    reduction_loc_attr = lookup_latest_reduction_group_path(
                        self.zw, normed_loc
                    )
                except KeyError:
                    reduction_loc_attr = None
                if reduction_loc_attr is not None:
                    try:
                        _, c_dims, c_pca_cell_key = parse_reduction_group_path(
                            reduction_loc_attr
                        )
                    except ValueError:
                        c_dims, c_pca_cell_key = None, None
                else:
                    c_dims, c_pca_cell_key = None, None
            else:
                c_dims, c_pca_cell_key = None, None
            if dims is None:
                if c_dims is not None:
                    dims = int(c_dims)
                    log_message("cached", "dims", dims)
                else:
                    dims = default_values["dims"]
                    log_message("default", "dims", dims)
            if pca_cell_key is None:
                if c_pca_cell_key is not None:
                    pca_cell_key = c_pca_cell_key
                    log_message("cached", "pca_cell_key", pca_cell_key)
                else:
                    pca_cell_key = cell_key
                    log_message("default", "pca_cell_key", pca_cell_key)
            else:
                if pca_cell_key not in self.cells.columns:
                    raise ValueError(
                        f"ERROR: `pca_use_cell_key` {pca_cell_key} does not exist in cell metadata"
                    )
                if self.cells.get_dtype(pca_cell_key) != bool:  # noqa: E721
                    raise TypeError(
                        "ERROR: Type of `pca_use_cell_key` column in cell metadata should be `bool`"
                    )
        dims = int(dims)
        if reduction_method == "auto" and state_reduction.get("reduction_method"):
            reduction_method = str(state_reduction["reduction_method"])
        reduction_method = self._choose_reduction_method(
            self._get_assay(from_assay), reduction_method
        )
        reduction_loc = make_reduction_group_path(
            normed_loc, reduction_method, dims, pca_cell_key
        )

        latest_ann_loc: str | None = None
        c_ann_metric: str | None = None
        c_ann_efc: int | None = None
        c_ann_ef: int | None = None
        c_ann_m: int | None = None
        c_rand_state: int | None = None
        if state_ann:
            c_ann_metric = cast(str | None, state_ann.get("ann_metric"))
            c_ann_efc = cast(int | None, state_ann.get("ann_efc"))
            c_ann_ef = cast(int | None, state_ann.get("ann_ef"))
            c_ann_m = cast(int | None, state_ann.get("ann_m"))
            c_rand_state = cast(int | None, state_ann.get("rand_state"))
        elif reduction_loc in self.zw:
            try:
                latest_ann_loc = lookup_latest_neighbor_index_group_path(
                    self.zw, reduction_loc
                )
            except KeyError:
                latest_ann_loc = None
            if latest_ann_loc is not None:
                try:
                    (
                        c_ann_metric,
                        c_ann_efc,
                        c_ann_ef,
                        c_ann_m,
                        c_rand_state,
                        _,
                        _,
                    ) = parse_neighbor_index_group_path(latest_ann_loc)
                except ValueError:
                    c_ann_metric = None
                    c_ann_efc = None
                    c_ann_ef = None
                    c_ann_m = None
                    c_rand_state = None

        if (
            ann_metric is None
            or ann_efc is None
            or ann_ef is None
            or ann_m is None
            or rand_state is None
        ):
            if ann_metric is None:
                if c_ann_metric is not None:
                    ann_metric = c_ann_metric
                    log_message("cached", "ann_metric", ann_metric)
                else:
                    ann_metric = default_values["ann_metric"]
                    log_message("default", "ann_metric", ann_metric)
            if ann_efc is None:
                if c_ann_efc is not None:
                    ann_efc = int(c_ann_efc)
                    log_message("cached", "ann_efc", ann_efc)
                else:
                    ann_efc = None  # Will be set after value for k is determined
                    log_message("default", "ann_efc", "min(100, max(k * 3, 50))")
            if ann_ef is None:
                if c_ann_ef is not None:
                    ann_ef = int(c_ann_ef)
                    log_message("cached", "ann_ef", ann_ef)
                else:
                    ann_ef = None  # Will be set after value for k is determined
                    log_message("default", "ann_ef", "min(100, max(k * 3, 50))")
            if ann_m is None:
                if c_ann_m is not None:
                    ann_m = int(c_ann_m)
                    log_message("cached", "ann_m", ann_m)
                else:
                    ann_m = min(max(48, int(dims * 1.5)), 64)
                    log_message("default", "ann_m", ann_m)
            if rand_state is None:
                if c_rand_state is not None:
                    rand_state = int(c_rand_state)
                    log_message("cached", "rand_state", rand_state)
                else:
                    rand_state = default_values["rand_state"]
                    log_message("default", "rand_state", rand_state)
        ann_metric = str(ann_metric)
        ann_m = int(ann_m)
        rand_state = int(rand_state)

        latest_ann_matches_parameters = (
            c_ann_metric == ann_metric
            and c_ann_efc == ann_efc
            and c_ann_ef == ann_ef
            and c_ann_m == ann_m
            and c_rand_state == rand_state
        )
        latest_knn_loc: str | None = None
        if latest_ann_loc is not None:
            try:
                latest_knn_loc = lookup_latest_nearest_neighbors_group_path(
                    self.zw, latest_ann_loc
                )
            except KeyError:
                latest_knn_loc = None

        if k is None:
            if state_neighbors.get("k") is not None:
                k = int(cast(int | float | str, state_neighbors["k"]))
                log_message("cached", "k", k)
            elif latest_knn_loc is not None:
                try:
                    k = parse_nearest_neighbors_group_path(latest_knn_loc)
                    log_message("cached", "k", k)
                except ValueError:
                    k = default_values["k"]
                    log_message("default", "k", k)
            else:
                k = default_values["k"]
                log_message("default", "k", k)
        k = int(k)
        if ann_ef is None:
            ann_ef = min(100, max(k * 3, 50))
        ann_ef = int(ann_ef)
        if ann_efc is None:
            ann_efc = min(100, max(k * 3, 50))
        ann_efc = int(ann_efc)
        # Intermediate path used only to read cached graph params; suffixes for
        # scaling/Harmony are applied later in make_graph.
        ann_loc = make_neighbor_index_group_path(
            reduction_loc,
            ann_metric,
            ann_efc,
            ann_ef,
            ann_m,
            rand_state,
        )
        knn_loc = make_nearest_neighbors_group_path(ann_loc, k)
        graph_params_knn_loc = knn_loc
        if latest_ann_matches_parameters and latest_knn_loc is not None:
            try:
                if parse_nearest_neighbors_group_path(latest_knn_loc) == k:
                    graph_params_knn_loc = latest_knn_loc
            except ValueError:
                pass

        if n_centroids is None:
            if state_initialization.get("n_centroids") is not None:
                n_centroids = int(
                    cast(int | float | str, state_initialization["n_centroids"])
                )
                log_message("cached", "n_centroids", n_centroids)
            elif reduction_loc in self.zw:
                kmeans_loc_attr = lookup_latest_kmeans_group_path(
                    self.zw, reduction_loc
                )
                if kmeans_loc_attr is not None:
                    try:
                        n_centroids, _ = parse_kmeans_group_path(kmeans_loc_attr)
                        log_message("default", "n_centroids", n_centroids)
                    except ValueError:
                        n_centroids = default_values["n_centroids"]
                        log_message("default", "n_centroids", n_centroids)
                else:
                    n_centroids = default_values["n_centroids"]
                    log_message("default", "n_centroids", n_centroids)
            else:
                n_centroids = default_values["n_centroids"]
                log_message("default", "n_centroids", n_centroids)
        n_centroids = int(n_centroids)

        if local_connectivity is None or bandwidth is None:
            if state_connectivity:
                c_local_connectivity = state_connectivity.get("local_connectivity")
                c_bandwidth = state_connectivity.get("bandwidth")
            elif graph_params_knn_loc in self.zw:
                try:
                    graph_loc_attr = lookup_latest_cell_graph_group_path(
                        self.zw, graph_params_knn_loc
                    )
                except KeyError:
                    graph_loc_attr = None
                if graph_loc_attr is not None:
                    try:
                        c_local_connectivity, c_bandwidth = parse_cell_graph_group_path(
                            graph_loc_attr
                        )
                    except ValueError:
                        c_local_connectivity, c_bandwidth = None, None
                else:
                    c_local_connectivity, c_bandwidth = None, None
            else:
                c_local_connectivity, c_bandwidth = None, None
            if local_connectivity is None:
                if c_local_connectivity is not None:
                    local_connectivity = c_local_connectivity
                    log_message("cached", "local_connectivity", local_connectivity)
                else:
                    local_connectivity = default_values["local_connectivity"]
                    log_message("default", "local_connectivity", local_connectivity)
            if bandwidth is None:
                if c_bandwidth is not None:
                    bandwidth = c_bandwidth
                    log_message("cached", "bandwidth", bandwidth)
                else:
                    bandwidth = default_values["bandwidth"]
                    log_message("default", "bandwidth", bandwidth)
        local_connectivity = float(local_connectivity)
        bandwidth = float(bandwidth)
        if feat_scaling is None:
            feat_scaling = bool(state_reduction.get("feat_scaling", True))
        if lsi_skip_first is None:
            lsi_skip_first = bool(state_reduction.get("lsi_skip_first", True))
        if harmonize is None:
            harmonize = bool(state_correction)
        if harmonize and state_correction:
            if batch_columns is None:
                stored_columns = state_correction.get("batch_columns")
                if isinstance(stored_columns, list):
                    batch_columns = [str(column) for column in stored_columns]
            if harmony_params is None:
                stored_parameters = state_correction.get("harmony_parameters")
                if isinstance(stored_parameters, dict):
                    harmony_params = dict(stored_parameters)

        return ResolvedGraphParameters(
            log_transform=log_transform,
            renormalize_subset=renormalize_subset,
            reduction_method=reduction_method,
            dims=dims,
            pca_cell_key=pca_cell_key,
            ann_metric=ann_metric,
            ann_efc=ann_efc,
            ann_ef=ann_ef,
            ann_m=ann_m,
            rand_state=rand_state,
            k=k,
            n_centroids=n_centroids,
            local_connectivity=local_connectivity,
            bandwidth=bandwidth,
            feat_scaling=feat_scaling,
            lsi_skip_first=lsi_skip_first,
            harmonize=harmonize,
            batch_columns=batch_columns,
            harmony_params=harmony_params,
        )

    def _get_latest_keys(
        self,
        from_assay: str | None,
        cell_key: str | None,
        feat_key: str | None,
    ) -> tuple[str, str, str]:
        if from_assay is None:
            from_assay = self._defaultAssay
        if from_assay is None:
            raise ValueError("No default assay is configured")
        if cell_key is None:
            cell_key = self._get_latest_cell_key(from_assay)
        if feat_key is None:
            feat_key = self._get_latest_feat_key(from_assay)
        return from_assay, cell_key, feat_key

    def get_normalized_group_path(
        self, from_assay: str, cell_key: str, feat_key: str
    ) -> str:
        """Return the selected normalized artifact or released-layout path.

        Args:
            from_assay: Name of the assay.
            cell_key: Cell key used (or to be used) for the graph.
            feat_key: Feature key used (or to be used) for the graph.

        Explicit keys use matching assay state when present, then the historical
        ``normed__`` convention.
        """
        state_path = normalized_path_from_state(
            self.zw,
            from_assay,
            cell_key,
            feat_key,
        )
        if state_path is not None:
            return state_path
        return make_normalized_group_path(from_assay, cell_key, feat_key)

    def get_latest_graph_loc(
        self, from_assay: str, cell_key: str, feat_key: str
    ) -> str:
        """Return the location of the latest graph in the Zarr hierarchy.

        Args:
            from_assay: Name of the assay.
            cell_key: Cell key used to create the graph.
            feat_key: Feature key used to create the graph.

        Returns:
            Path of graph in the Zarr hierarchy
        """
        stored = self._lookup_stored_graph(from_assay, cell_key, feat_key)
        if not isinstance(stored, StoredAssayGraph):
            raise TypeError("Latest assay graph lookup returned a non-assay graph")
        return stored.paths.cell_graph_group_path

    def _get_latest_graph_loc(
        self, from_assay: str, cell_key: str, feat_key: str
    ) -> str:
        """Compatibility alias for :meth:`get_latest_graph_loc`."""
        return self.get_latest_graph_loc(from_assay, cell_key, feat_key)

    def _resolve_integrated_graph_path(self, label: str) -> str:
        index_path = self._integratedGraphsLoc
        if index_path in self.zw:
            index_group = as_zarr_group(self.zw[index_path], name=index_path)
            raw_artifacts = index_group.attrs.get("artifacts", {})
            if "artifacts" in index_group.attrs and not isinstance(
                raw_artifacts,
                dict,
            ):
                raise RuntimeError("Integrated graph artifact index is invalid")
            if isinstance(raw_artifacts, dict):
                raw_ref = raw_artifacts.get(label)
                if label in raw_artifacts and not isinstance(raw_ref, dict):
                    raise RuntimeError(
                        f"Integrated graph index for {label!r} is invalid"
                    )
                if isinstance(raw_ref, dict):
                    try:
                        ref = ArtifactRef.from_dict(raw_ref)
                        if ref.scope != "datastore" or ref.kind != "integrated_graph":
                            raise ValueError(
                                "Integrated graph index has an invalid ref"
                            )
                        status = inspect_artifact(self.zw, ref)
                    except (KeyError, TypeError, ValueError) as exc:
                        raise RuntimeError(
                            f"Integrated graph index for {label!r} is invalid"
                        ) from exc
                    if not status.exists or not status.complete:
                        raise RuntimeError(
                            f"Integrated graph index for {label!r} is incomplete"
                        )
                    return status.path
        return make_integrated_graph_path(index_path, label)

    def _lookup_stored_graph(
        self,
        from_assay: str | None = None,
        cell_key: str | None = None,
        feat_key: str | None = None,
        graph_loc: str | None = None,
    ) -> StoredGraph:
        """Return a stored assay or integrated graph without mutating the store.

        When ``graph_loc`` is omitted, resolves assay state before released
        ``latest_*`` pointers. Explicit logical and encoded locations remain
        readable without changing state.
        """
        if graph_loc is not None:
            if is_integrated_graph_path(graph_loc, self._integratedGraphsLoc):
                return lookup_stored_integrated_graph(self.zw, graph_loc)
            try:
                ref = parse_artifact_path(graph_loc)
            except ValueError:
                explicit_stored = parse_assay_graph_paths(graph_loc)
                validate_legacy_graph_selection(
                    self,
                    graph_loc,
                    explicit_stored.from_assay,
                    explicit_stored.cell_key,
                    explicit_stored.feat_key,
                )
                return explicit_stored
            if ref.scope == "datastore" and ref.kind == "integrated_graph":
                status = inspect_artifact(self.zw, ref)
                if not status.exists or not status.complete:
                    raise RuntimeError(
                        f"Integrated graph artifact is incomplete: {graph_loc}"
                    )
                return lookup_stored_integrated_graph(self.zw, graph_loc)
            if ref.scope != "assay" or ref.kind != "connectivity_map":
                raise ValueError(f"Not an assay connectivity-map artifact: {graph_loc}")
            return stored_assay_graph_from_ref(self.zw, ref)

        state_assay = from_assay or self._defaultAssay
        if state_assay is not None:
            state = read_assay_state(self.zw, state_assay)
            if (
                state is not None
                and (cell_key is None or cell_key == state.cell_key)
                and (feat_key is None or feat_key == state.feat_key)
            ):
                selected_cell_key = cell_key or state.cell_key
                selected_feat_key = feat_key or state.feat_key
                state_stored = stored_assay_graph_from_state(
                    self.zw,
                    state_assay,
                    selected_cell_key,
                    selected_feat_key,
                )
                if state_stored is not None:
                    return state_stored
        from_assay, cell_key, feat_key = self._get_latest_keys(
            from_assay,
            cell_key,
            feat_key,
        )
        legacy_stored = lookup_latest_assay_graph(
            self.zw,
            from_assay,
            cell_key,
            feat_key,
        )
        validate_legacy_graph_selection(
            self,
            legacy_stored.paths.cell_graph_group_path,
            from_assay,
            cell_key,
            feat_key,
        )
        return legacy_stored

    def _get_latest_knn_loc(self, from_assay: str | None = None) -> str:
        """Convenience function to identify location of the latest KNN graph in
        the Zarr hierarchy.

        Args:
            from_assay: Name of the assay.

        Returns:
            Path of KNN graph in the Zarr hierarchy
        """
        if from_assay is None:
            logger.info("Using default assay for KNN graph.")
            from_assay = self._load_default_assay()

        if from_assay not in self.assay_names:
            raise ValueError(f"ERROR: Assay {from_assay} does not exist")

        state = read_assay_state(self.zw, from_assay)
        if state is not None:
            if state.neighbors is None:
                raise RuntimeError("AssayState has no neighbors artifact")
            status = inspect_artifact(self.zw, state.neighbors)
            if not status.exists or not status.complete:
                raise RuntimeError("AssayState selects incomplete neighbors")
            return status.path

        latest_cell_key = cast(
            str,
            as_zarr_group(self.zw[from_assay], name=from_assay).attrs[
                "latest_cell_key"
            ],
        )
        latest_feat_key = cast(
            str,
            as_zarr_group(self.zw[from_assay], name=from_assay).attrs[
                "latest_feat_key"
            ],
        )
        paths = lookup_latest_nearest_neighbor_paths(
            self.zw,
            from_assay,
            latest_cell_key,
            latest_feat_key,
        )
        reduction_loc = paths.reduction_group_path
        reduction_grp = as_zarr_group(self.zw[reduction_loc], name=reduction_loc)
        if "reduction" not in reduction_grp:
            raise ValueError(f"ERROR: PCA Reduction not found in {reduction_loc}")
        return paths.nearest_neighbors_group_path

    def _ann_stream_recoverable(
        self,
        ann_loc: str,
        reduction_loc: str,
        normed_loc: str,
    ) -> bool:
        """Return True when an ANN stream can be loaded or rebuilt without a full make_graph."""
        if ann_loc in self.zw and has_ann_index(
            as_zarr_group(self.zw[ann_loc], name=ann_loc)
        ):
            return True
        legacy = legacy_ann_index_path(zarr_root_path(self.zw), ann_loc)
        if legacy is not None and os.path.exists(legacy):
            return True
        if reduction_loc in self.zw:
            reduction_grp = as_zarr_group(self.zw[reduction_loc], name=reduction_loc)
            if "reduction" in reduction_grp:
                if normed_loc in self.zw:
                    normed_grp = as_zarr_group(self.zw[normed_loc], name=normed_loc)
                    if "data" in normed_grp:
                        return True
        return False

    def _resolve_ann_index(
        self,
        ann_loc: str,
        ann_metric: str,
        dim: int,
        ann_index_fetcher: Callable | None = None,
    ) -> Any:
        """Load ANN index from zarr, legacy file, custom fetcher, or return None to rebuild."""
        ann_group: zarr.Group | None = (
            as_zarr_group(self.zw[ann_loc], name=ann_loc)
            if ann_loc in self.zw
            else None
        )

        if ann_group is not None and has_ann_index(ann_group):
            return load_ann_index(ann_group, ann_metric, dim)

        if ann_index_fetcher is not None:
            try:
                ann_index_fn = ann_index_fetcher(ann_loc)
            except Exception:
                ann_index_fn = None
                logger.warning("Custom `ann_index_fetcher` failed")
            if ann_index_fn is not None and os.path.exists(ann_index_fn):
                return load_ann_index_from_path(ann_index_fn, ann_metric, dim)

        legacy = legacy_ann_index_path(zarr_root_path(self.zw), ann_loc)
        if legacy is not None and os.path.exists(legacy):
            return load_ann_index_from_path(legacy, ann_metric, dim)

        logger.info(
            "ANN index not found in store; will rebuild from normalized data and loadings"
        )
        return None

    def _persist_ann_index(
        self,
        ann_loc: str,
        ann_idx: Any,
        ann_index_saver: Callable | None = None,
    ) -> None:
        """Save an hnswlib index into the zarr hierarchy or via a custom saver."""
        if ann_index_saver is not None:
            try:
                ann_index_saver(ann_idx, ann_loc)
            except Exception:
                logger.warning("Custom `ann_index_saver` failed")
        if ann_loc not in self.zw:
            self.zw.create_group(ann_loc, overwrite=True)
        if self.zarr_mode != "r+":
            logger.debug("Skipping ANN index persistence on read-only store")
            return
        save_ann_index(as_zarr_group(self.zw[ann_loc], name=ann_loc), ann_idx)

    def _has_ann_stream_cache(
        self,
        from_assay: str,
        cell_key: str,
        feat_key: str,
        knn_loc: str | None = None,
        feat_scaling: bool | None = None,
    ) -> bool:
        state = read_assay_state(self.zw, from_assay)
        if state is not None and state.matches(cell_key, feat_key):
            if state.normalized is None:
                return False
            try:
                validate_normalized_artifact_selection(
                    self.zw,
                    state.normalized,
                    cell_key,
                    feat_key,
                )
            except (KeyError, RuntimeError, TypeError, ValueError):
                return False
            required = (
                state.normalized,
                state.feature_scaling,
                state.reduction,
                state.ann_index,
                state.neighbors,
            )
            if any(ref is None for ref in required):
                return False
            assert state.ann_index is not None
            ann_group = as_zarr_group(
                self.zw[artifact_path(state.ann_index)],
                name=artifact_path(state.ann_index),
            )
            if not has_ann_index(ann_group):
                return False
            if feat_scaling is not None and state.reduction is not None:
                reduction_status = inspect_artifact(self.zw, state.reduction)
                cached_scaling = bool(
                    (reduction_status.parameters or {}).get("feat_scaling", True)
                )
                if cached_scaling != feat_scaling:
                    return False
            return True
        try:
            if knn_loc is None:
                chain = lookup_latest_nearest_neighbor_paths(
                    self.zw,
                    from_assay,
                    cell_key,
                    feat_key,
                )
                knn_loc = chain.nearest_neighbors_group_path
                ann_loc = chain.neighbor_index_group_path
                reduction_loc = chain.reduction_group_path
                normed_loc = chain.normalized_group_path
            else:
                chain = nearest_neighbor_paths_from_loc(knn_loc)
                ann_loc = chain.neighbor_index_group_path
                reduction_loc = chain.reduction_group_path
                normed_loc = chain.normalized_group_path

            if knn_loc not in self.zw:
                return False
            if ann_loc not in self.zw:
                return False
            try:
                validate_legacy_graph_selection(
                    self,
                    knn_loc,
                    from_assay,
                    cell_key,
                    feat_key,
                )
            except (KeyError, RuntimeError, TypeError, ValueError):
                return False
            if feat_scaling is not None:
                ann_grp = as_zarr_group(self.zw[ann_loc], name=ann_loc)
                cached_scaling = bool(ann_grp.attrs.get("featureScaling", True))
                if cached_scaling != feat_scaling:
                    return False
            return self._ann_stream_recoverable(ann_loc, reduction_loc, normed_loc)
        except KeyError:
            return False

    def _load_or_compute_norm_stats(
        self,
        normed_loc: str,
        data: ChunkedArray,
        reduction_method: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        mu, sigma = np.ndarray([]), np.ndarray([])
        if reduction_method not in ["pca", "manual"]:
            return mu, sigma
        normed_grp = as_zarr_group(self.zw[normed_loc], name=normed_loc)
        need_mu = "mu" not in normed_grp
        need_sigma = "sigma" not in normed_grp
        if not need_mu:
            mu = np.asarray(as_zarr_array(normed_grp["mu"], name="mu")[:])
        if not need_sigma:
            sigma = np.asarray(as_zarr_array(normed_grp["sigma"], name="sigma")[:])
        if need_mu and need_sigma:
            mu_raw, sigma_raw = data.mean_and_std(
                nthreads=self.nthreads,
                msg="Calculating mean and std. dev. of norm. data",
            )
            mu = clean_array(mu_raw)
            sigma = clean_array(sigma_raw, 1)
            if self.zarr_mode == "r+":
                g = create_zarr_dataset(normed_grp, "mu", (100000,), "f8", mu.shape)
                g[:] = mu
                g = create_zarr_dataset(
                    normed_grp, "sigma", (100000,), "f8", sigma.shape
                )
                g[:] = sigma
            else:
                logger.debug("Skipping mu/sigma persistence on read-only store")
        elif need_mu:
            mu = clean_array(
                show_dask_progress(
                    data.mean(axis=0),
                    "Calculating mean of norm. data",
                    self.nthreads,
                )
            )
            if self.zarr_mode == "r+":
                g = create_zarr_dataset(normed_grp, "mu", (100000,), "f8", mu.shape)
                g[:] = mu
            else:
                logger.debug("Skipping mu persistence on read-only store")
        elif need_sigma:
            sigma = clean_array(
                show_dask_progress(
                    data.std(axis=0),
                    "Calculating std. dev. of norm. data",
                    self.nthreads,
                ),
                1,
            )
            if self.zarr_mode == "r+":
                g = create_zarr_dataset(
                    normed_grp, "sigma", (100000,), "f8", sigma.shape
                )
                g[:] = sigma
            else:
                logger.debug("Skipping sigma persistence on read-only store")
        return mu, sigma

    def _load_artifact_ann_stream(
        self,
        from_assay: str,
        cell_key: str,
        feat_key: str,
        feat_scaling: bool,
        neighbors_ref: ArtifactRef | None = None,
        fit_kmeans: bool = False,
        n_centroids: int = 2,
    ) -> AnnStream | None:
        def input_ref(owner: ArtifactRef, name: str) -> ArtifactRef:
            value = (inspect_artifact(self.zw, owner).inputs or {}).get(name)
            if not isinstance(value, dict):
                raise ValueError(f"{owner.kind} has no {name!r} artifact input")
            return ArtifactRef.from_dict(value)

        correction_ref = None
        if neighbors_ref is None:
            state = read_assay_state(self.zw, from_assay)
            if state is None or not state.matches(cell_key, feat_key):
                return None
            if (
                state.normalized is None
                or state.feature_scaling is None
                or state.reduction is None
                or state.ann_index is None
                or state.neighbors is None
            ):
                raise KeyError("AssayState has no complete ANN stream")
            normalized_ref = state.normalized
            scaling_ref = state.feature_scaling
            reduction_ref = state.reduction
            ann_ref = state.ann_index
            neighbors_ref = state.neighbors
            correction_ref = state.batch_correction
        else:
            ann_ref = input_ref(neighbors_ref, "ann_index")
            coordinates_ref = input_ref(neighbors_ref, "coordinates")
            if coordinates_ref.kind == "batch_correction":
                correction_ref = coordinates_ref
                reduction_ref = input_ref(correction_ref, "reduction")
            elif coordinates_ref.kind == "reduction":
                reduction_ref = coordinates_ref
            else:
                raise ValueError("Unsupported neighbor coordinate artifact")
            normalized_ref = input_ref(reduction_ref, "normalized")
            scaling_ref = input_ref(reduction_ref, "feature_scaling")
        normalized_group = as_zarr_group(
            self.zw[artifact_path(normalized_ref)],
            name=artifact_path(normalized_ref),
        )
        scaling_group = as_zarr_group(
            self.zw[artifact_path(scaling_ref)],
            name=artifact_path(scaling_ref),
        )
        reduction_group = as_zarr_group(
            self.zw[artifact_path(reduction_ref)],
            name=artifact_path(reduction_ref),
        )
        neighbors_group = as_zarr_group(
            self.zw[artifact_path(neighbors_ref)],
            name=artifact_path(neighbors_ref),
        )
        data = ChunkedArray(
            as_zarr_array(normalized_group["data"], name="data"),
            nthreads=self.nthreads,
        )
        mu = np.asarray(as_zarr_array(scaling_group["mean"], name="mean")[:])
        sigma = np.asarray(as_zarr_array(scaling_group["scale"], name="scale")[:])
        loadings = (
            np.asarray(as_zarr_array(reduction_group["loadings"], name="loadings")[:])
            if "loadings" in reduction_group
            else None
        )
        reduction_status = inspect_artifact(self.zw, reduction_ref)
        reduction_params = reduction_status.parameters or {}
        reduction_execution = reduction_status.execution_options or {}
        operation = reduction_status.operation
        reduction_method = (
            {
                "run_pca": "pca",
                "run_lsi": "lsi",
                "run_custom_reduction": "custom",
            }.get(operation)
            if operation is not None
            else None
        )
        if reduction_method is None:
            reduction_method = str(reduction_params.get("reduction_method", "pca"))
        ann_params = inspect_artifact(self.zw, ann_ref).parameters or {}
        neighbor_params = inspect_artifact(self.zw, neighbors_ref).parameters or {}
        cached_scaling = bool(
            reduction_params.get(
                "feat_scaling",
                reduction_method == "pca",
            )
        )
        if cached_scaling != feat_scaling:
            raise ValueError(
                f"Reduction was built with feat_scaling={cached_scaling}, "
                f"not {feat_scaling}"
            )
        corrected = None
        if correction_ref is not None:
            correction_group = as_zarr_group(
                self.zw[artifact_path(correction_ref)],
                name=artifact_path(correction_ref),
            )
            corrected = ChunkedArray(
                as_zarr_array(correction_group["data"], name="data"),
                nthreads=self.nthreads,
            )
        ann_metric = str(ann_params.get("ann_metric", "l2"))
        raw_dims = reduction_params.get("dims")
        dims = int(
            raw_dims
            if raw_dims is not None
            else loadings.shape[1]
            if loadings is not None
            else data.shape[1]
        )
        ann_idx = self._resolve_ann_index(
            artifact_path(ann_ref),
            ann_metric,
            dims if dims > 0 else data.shape[1],
            ann_index_fetcher=None,
        )
        if ann_idx is None:
            raise RuntimeError("Selected ANN artifact has no readable index")
        neighbor_indices = as_zarr_array(
            neighbors_group["indices"],
            name="indices",
        )
        k_value = neighbor_params.get("k")
        ann_obj = AnnStream(
            data=data,
            k=int(neighbor_indices.shape[1] if k_value is None else k_value),
            n_cluster=n_centroids,
            reduction_method=reduction_method,
            dims=dims,
            loadings=loadings,
            use_for_pca=(
                self.cells.fetch(
                    str(reduction_execution.get("pca_cell_key", cell_key)),
                    key=cell_key,
                )
                if reduction_method == "pca"
                else np.ones(data.shape[0], dtype=bool)
            ),
            mu=mu,
            sigma=sigma,
            ann_metric=ann_metric,
            ann_efc=int(ann_params.get("ann_efc", 50)),
            ann_ef=int(ann_params.get("ann_ef", 50)),
            ann_m=int(ann_params.get("ann_m", 48)),
            nthreads=self.nthreads,
            ann_parallel=bool(ann_params.get("ann_parallel", False)),
            rand_state=int(ann_params.get("rand_state", 4466)),
            do_kmeans_fit=fit_kmeans,
            disable_scaling=not cached_scaling,
            ann_idx=ann_idx,
            lsi_skip_first=bool(
                reduction_params.get(
                    "skip_first",
                    reduction_params.get("lsi_skip_first", True),
                )
            ),
            lsi_params={},
            harmonize=correction_ref is not None,
            harmonized_data=corrected,
            batches=None,
            cache_embeddings=False,
        )
        persisted_ann_threads = int(ann_params.get("parallel_threads") or 1)
        AnnIndexStage.configure(
            ann_obj.annIdx,
            ef=int(ann_params.get("ann_ef", 50)),
            threads=persisted_ann_threads,
        )
        ann_obj.annThreads = persisted_ann_threads
        self._remember_ann_stream_path(
            ann_obj,
            artifact_path(ann_ref),
        )
        assert neighbors_ref is not None
        self._remember_ann_stream_neighbors(
            ann_obj,
            artifact_path(neighbors_ref),
        )
        return ann_obj

    def _load_ann_stream(
        self,
        from_assay: str,
        cell_key: str,
        feat_key: str,
        feat_scaling: bool = True,
        knn_loc: str | None = None,
    ) -> AnnStream:
        """Load an AnnStream from an existing graph without recomputing KNN."""

        artifact_neighbors = None
        if knn_loc is not None:
            try:
                candidate = parse_artifact_path(knn_loc)
            except ValueError:
                pass
            else:
                if candidate.kind == "neighbors":
                    artifact_neighbors = candidate
                    self._artifact_chain_state(
                        candidate,
                        cell_key_override=cell_key,
                        feat_key_override=feat_key,
                    )
        if knn_loc is None:
            state = read_assay_state(self.zw, from_assay)
            if (
                state is not None
                and state.matches(cell_key, feat_key)
                and state.normalized is not None
            ):
                validate_normalized_artifact_selection(
                    self.zw,
                    state.normalized,
                    cell_key,
                    feat_key,
                )
        if knn_loc is None or artifact_neighbors is not None:
            artifact_stream = self._load_artifact_ann_stream(
                from_assay,
                cell_key,
                feat_key,
                feat_scaling,
                neighbors_ref=artifact_neighbors,
            )
            if artifact_stream is not None:
                return artifact_stream
        if knn_loc is None:
            chain = lookup_latest_nearest_neighbor_paths(
                self.zw,
                from_assay,
                cell_key,
                feat_key,
            )
            normed_loc = chain.normalized_group_path
            if normed_loc not in self.zw:
                raise KeyError(f"No normalized data at {normed_loc}")
            reduction_loc = chain.reduction_group_path
            ann_loc = chain.neighbor_index_group_path
            knn_loc = chain.nearest_neighbors_group_path
        else:
            chain = nearest_neighbor_paths_from_loc(knn_loc)
            validate_legacy_graph_selection(
                self,
                knn_loc,
                from_assay,
                cell_key,
                feat_key,
            )
            ann_loc = chain.neighbor_index_group_path
            reduction_loc = chain.reduction_group_path
            normed_loc = chain.normalized_group_path

        if knn_loc not in self.zw:
            raise KeyError(f"KNN graph not found at {knn_loc}")

        (
            ann_metric,
            ann_efc,
            ann_ef,
            ann_m,
            rand_state,
            _,
            _,
        ) = parse_neighbor_index_group_path(ann_loc)
        reduction_method, dims, pca_cell_key = parse_reduction_group_path(reduction_loc)
        k = parse_nearest_neighbors_group_path(knn_loc)

        data = ChunkedArray(
            as_zarr_array(
                as_zarr_group(self.zw[normed_loc], name=normed_loc)["data"],
                name="data",
            ),
            nthreads=self.nthreads,
        )
        mu, sigma = self._load_or_compute_norm_stats(normed_loc, data, reduction_method)

        loadings: NDArray[Any] | None = None
        reduction_grp = as_zarr_group(self.zw[reduction_loc], name=reduction_loc)
        if "reduction" in reduction_grp:
            loadings = np.asarray(
                as_zarr_array(reduction_grp["reduction"], name="reduction")[:]
            )

        ann_grp = as_zarr_group(self.zw[ann_loc], name=ann_loc)
        cached_scaling = bool(ann_grp.attrs.get("featureScaling", True))
        if cached_scaling != feat_scaling:
            raise ValueError(
                f"ANN index at {ann_loc} was built with featureScaling="
                f"{cached_scaling}, not {feat_scaling}. Rebuild the graph."
            )
        harmonize = cast(bool, ann_grp.attrs.get("isHarmonized", False))
        harmonized_data = None
        batches = None
        if harmonize and "harmonizedData" in reduction_grp:
            harmonized_arr = as_zarr_array(
                reduction_grp["harmonizedData"], name="harmonizedData"
            )
            harmonized_data = ChunkedArray(harmonized_arr, nthreads=self.nthreads)
            batch_columns = cast(list[str] | None, harmonized_arr.attrs.get("batches"))
            if batch_columns:
                batches = pd.DataFrame(
                    {
                        x: self.cells.fetch(x, key=cell_key).astype(object)
                        for x in batch_columns
                    }
                )

        temp_dim = dims if dims > 0 else data.shape[1]
        ann_idx = self._resolve_ann_index(
            ann_loc,
            ann_metric,
            temp_dim,
            ann_index_fetcher=None,
        )
        rebuilt_ann = ann_idx is None

        use_for_pca = self.cells.fetch(pca_cell_key, key=cell_key)
        logger.info(f"Loaded existing ANN stream from {ann_loc}")
        ann_obj = AnnStream(
            data=data,
            k=k,
            n_cluster=2,
            reduction_method=reduction_method,
            dims=dims,
            loadings=loadings,
            use_for_pca=use_for_pca,
            mu=mu,
            sigma=sigma,
            ann_metric=ann_metric,
            ann_efc=ann_efc,
            ann_ef=ann_ef,
            ann_m=ann_m,
            nthreads=self.nthreads,
            ann_parallel=False,
            rand_state=rand_state,
            do_kmeans_fit=False,
            disable_scaling=not feat_scaling,
            ann_idx=ann_idx,
            lsi_skip_first=True,
            lsi_params={},
            harmonize=harmonize,
            harmonized_data=harmonized_data,
            batches=batches,
            cache_embeddings=False,
        )
        self._remember_ann_stream_path(ann_obj, ann_loc)
        self._remember_ann_stream_neighbors(ann_obj, knn_loc)
        if rebuilt_ann and self.zarr_mode == "r+":
            self._persist_ann_index(ann_loc, ann_obj.annIdx)
        return ann_obj

    def _get_graph_ncells_k(self, graph_loc: str) -> tuple[int, int]:
        """

        Args:
            graph_loc:

        Returns:

        """
        if is_integrated_graph_path(graph_loc, self._integratedGraphsLoc):
            stored = lookup_stored_integrated_graph(self.zw, graph_loc)
            if stored.n_cells is None or stored.n_neighbors is None:
                raise KeyError(
                    f"Integrated graph at {graph_loc} is missing n_cells/n_neighbors"
                )
            return stored.n_cells, stored.n_neighbors
        graph_group = as_zarr_group(self.zw[graph_loc], name=graph_loc)
        if "n_cells" in graph_group.attrs and "n_neighbors" in graph_group.attrs:
            return (
                int(cast(int | float | str, graph_group.attrs["n_cells"])),
                int(cast(int | float | str, graph_group.attrs["n_neighbors"])),
            )
        knn_loc = nearest_neighbors_group_path_from_cell_graph(graph_loc)
        knn_grp = as_zarr_group(self.zw[knn_loc], name=knn_loc)
        indices = as_zarr_array(knn_grp["indices"], name="indices")
        return indices.shape[0], indices.shape[1]

    def _store_to_sparse(
        self, graph_loc: str, sparse_format: str = "csr", use_k: int | None = None
    ) -> tuple[int, csr_matrix | coo_matrix]:
        """

        Args:
            graph_loc:
            sparse_format:
            use_k:

        Returns:

        """
        logger.debug(f"Loading graph from location: {graph_loc}")
        store = as_zarr_group(self.zw[graph_loc], name=graph_loc)
        n_cells, k = self._get_graph_ncells_k(graph_loc)
        # TODO: can we have a progress bar for graph loading. Append to coo matrix?
        if use_k is None:
            use_k = k
        if use_k > k:
            use_k = k
        if use_k < 1:
            use_k = 1
        if use_k != k:
            indexer = np.tile([True] * use_k + [False] * (k - use_k), n_cells)
        else:
            indexer = None
        w = np.asarray(as_zarr_array(store["weights"], name="weights")[:])
        e = np.asarray(as_zarr_array(store["edges"], name="edges")[:])
        if indexer is not None:
            w, e = w[indexer], e[indexer]
        if sparse_format == "csr":
            return n_cells, csr_matrix(
                (w, (e[:, 0], e[:, 1])), shape=(n_cells, n_cells)
            )
        else:
            return n_cells, coo_matrix(
                (w, (e[:, 0], e[:, 1])), shape=(n_cells, n_cells)
            )

    @staticmethod
    def _resolve_local_cache_plan(
        zarr_loc: Any,
        group: zarr.Group,
        local_cache: bool | str,
    ) -> tuple[bool, str | None, bool]:
        """Return whether to stage, cache base directory, and remove-on-success flag."""
        if local_cache is False or not is_remote_datastore(zarr_loc, group):
            return False, None, False
        if local_cache is True or local_cache == "auto":
            return True, tempfile.mkdtemp(prefix="scarf_local_cache_"), True
        if isinstance(local_cache, str):
            os.makedirs(local_cache, exist_ok=True)
            return True, local_cache, False
        raise TypeError(
            f"local_cache must be 'auto', True, False, or a path string, got {local_cache!r}"
        )

    def _require_complete_artifact(
        self,
        ref: ArtifactRef,
        kind: str,
        *,
        assay: str | None = None,
    ) -> Any:
        if ref.kind != kind:
            raise ValueError(f"Expected {kind!r} artifact, got {ref.kind!r}")
        if assay is not None and (ref.scope != "assay" or ref.assay != assay):
            raise ValueError(f"Artifact must belong to assay {assay!r}")
        status = inspect_artifact(self.zw, ref)
        if not status.exists:
            raise KeyError(f"Artifact does not exist: {status.path}")
        if not status.complete:
            raise RuntimeError(f"Artifact is incomplete: {status.path}")
        return status

    def _artifact_input_ref(
        self,
        ref: ArtifactRef,
        name: str,
        kind: str,
    ) -> ArtifactRef:
        status = self._require_complete_artifact(ref, ref.kind)
        raw_ref = (status.inputs or {}).get(name)
        if not isinstance(raw_ref, dict):
            raise ValueError(f"{ref.kind} artifact has no {name!r} input")
        input_ref = ArtifactRef.from_dict(raw_ref)
        self._require_complete_artifact(input_ref, kind)
        return input_ref

    def _resolve_selection_input(
        self,
        *,
        metadata_group: zarr.Group,
        column: str,
        values: np.ndarray,
        row_ids: np.ndarray,
        scope: ArtifactScope,
        kind: str,
        assay: str | None,
        invalidate_cache: bool,
    ) -> ArtifactRef:
        ref = None
        if not invalidate_cache:
            column_array = as_zarr_array(
                metadata_group[column],
                name=column,
            )
            raw_ref = column_array.attrs.get("source_artifact")
            if isinstance(raw_ref, dict):
                try:
                    candidate = ArtifactRef.from_dict(raw_ref)
                    status = inspect_artifact(self.zw, candidate)
                except (KeyError, TypeError, ValueError):
                    pass
                else:
                    if (
                        candidate.scope == scope
                        and candidate.kind == kind
                        and candidate.assay == assay
                        and status.complete
                        and (status.inputs or {}).get("ordered_row_ids_fingerprint")
                        == fingerprint_strings(row_ids)
                    ):
                        group = as_zarr_group(
                            self.zw[status.path],
                            name=status.path,
                        )
                        if "values" in group:
                            stored = np.asarray(
                                as_zarr_array(
                                    group["values"],
                                    name="values",
                                )[:]
                            )
                            if (
                                stored.ndim == 1
                                and stored.dtype == np.dtype(bool)
                                and values.ndim == 1
                                and values.dtype == np.dtype(bool)
                                and stored.shape == values.shape
                                and np.array_equal(stored, values)
                            ):
                                ref = candidate
        if ref is None:
            ref = resolve_selection_artifact(
                self.zw,
                scope=scope,
                assay=assay,
                kind=kind,
                values=values,
                row_ids=row_ids,
                operation="manual_selection",
                parameters={},
                inputs={},
                source_column=column,
                invalidate_cache=invalidate_cache,
            )
        if scope == "assay" and assay is not None:
            link_feature_data_column(
                self._get_assay(assay).z,
                column,
                ref,
                value_name="values",
                default_display=categorical_display(values),
            )
        else:
            target = as_zarr_array(
                metadata_group[column],
                name=column,
            )
            target.attrs["source_artifact"] = ref.to_dict()
            target.attrs["source_value"] = "values"
        return ref

    def _selected_artifact(
        self,
        from_assay: str | None,
        field_name: str,
        kind: str,
    ) -> ArtifactRef:
        assay = from_assay or self._defaultAssay
        if assay is None:
            raise ValueError("No assay was provided and no default is configured")
        state = read_assay_state(self.zw, assay)
        if state is None:
            raise KeyError(f"Assay {assay!r} has no selected artifact state")
        ref = getattr(state, field_name)
        if not isinstance(ref, ArtifactRef):
            raise KeyError(f"AssayState has no selected {field_name} artifact")
        self._require_complete_artifact(ref, kind, assay=assay)
        return ref

    def _artifact_chain_state(
        self,
        terminal: ArtifactRef,
        *,
        embedding_initialization: ArtifactRef | None = None,
        named_results: dict[str, ArtifactRef] | None = None,
        cell_key_override: str | None = None,
        feat_key_override: str | None = None,
    ) -> AssayState:
        normalized = feature_scaling = reduction = None
        batch_correction = ann_index = neighbors = connectivity_map = None
        current = terminal
        if current.kind == "connectivity_map":
            connectivity_map = current
            current = self._artifact_input_ref(
                current,
                "neighbors",
                "neighbors",
            )
        if current.kind == "neighbors":
            neighbors = current
            ann_index = self._artifact_input_ref(
                current,
                "ann_index",
                "ann_index",
            )
            neighbor_status = self._require_complete_artifact(
                current,
                "neighbors",
            )
            raw_coordinates = (neighbor_status.inputs or {}).get("coordinates")
            if not isinstance(raw_coordinates, dict):
                raise ValueError("Neighbors artifact has no coordinates input")
            current = ArtifactRef.from_dict(raw_coordinates)
            if current.kind not in {"reduction", "batch_correction"}:
                raise ValueError(
                    "Neighbor coordinates must be reduction or batch_correction"
                )
            self._require_complete_artifact(current, current.kind)
            ann_coordinates = self._artifact_input_ref(
                ann_index,
                "coordinates",
                current.kind,
            )
            if ann_coordinates != current:
                raise ValueError("Neighbors and ANN index use different coordinates")
        elif current.kind == "ann_index":
            ann_index = current
            status = self._require_complete_artifact(current, "ann_index")
            raw_coordinates = (status.inputs or {}).get("coordinates")
            if not isinstance(raw_coordinates, dict):
                raise ValueError("ANN artifact has no coordinates input")
            current = ArtifactRef.from_dict(raw_coordinates)
        if current.kind == "batch_correction":
            batch_correction = current
            current = self._artifact_input_ref(
                current,
                "reduction",
                "reduction",
            )
        if current.kind == "reduction":
            reduction = current
            normalized = self._artifact_input_ref(
                current,
                "normalized",
                "normalized",
            )
            feature_scaling = self._artifact_input_ref(
                current,
                "feature_scaling",
                "feature_scaling",
            )
        elif current.kind == "normalized":
            normalized = current
        else:
            raise ValueError(f"Cannot publish graph state from {terminal.kind!r}")
        assert normalized is not None
        normalized_status = self._require_complete_artifact(
            normalized,
            "normalized",
        )
        execution = normalized_status.execution_options or {}
        cell_key = execution.get("cell_key")
        feat_key = execution.get("feat_key")
        if not isinstance(cell_key, str) or not isinstance(feat_key, str):
            raise ValueError("Normalized artifact is missing cell_key or feat_key")
        if normalized.assay is None:
            raise ValueError("Normalized artifact has no assay")
        previous = read_assay_state(self.zw, normalized.assay)
        if (cell_key_override is None) != (feat_key_override is None):
            raise ValueError(
                "cell_key and feat_key overrides must be provided together"
            )
        if cell_key_override is not None and feat_key_override is not None:
            cell_key = cell_key_override
            feat_key = feat_key_override
        elif previous is not None and previous.normalized == normalized:
            cell_key = previous.cell_key
            feat_key = previous.feat_key
        validate_normalized_artifact_selection(
            self.zw,
            normalized,
            cell_key,
            feat_key,
        )
        if embedding_initialization is None and reduction is not None:
            if previous is not None and previous.embedding_initialization is not None:
                previous_reduction = self._artifact_input_ref(
                    previous.embedding_initialization,
                    "reduction",
                    "reduction",
                )
                if previous_reduction == reduction:
                    embedding_initialization = previous.embedding_initialization
        return AssayState(
            assay=normalized.assay,
            cell_key=cell_key,
            feat_key=feat_key,
            normalized=normalized,
            feature_scaling=feature_scaling,
            reduction=reduction,
            batch_correction=batch_correction,
            ann_index=ann_index,
            embedding_initialization=embedding_initialization,
            neighbors=neighbors,
            connectivity_map=connectivity_map,
            named_results=named_results or {},
        )

    def _publish_current_artifact(
        self,
        ref: ArtifactRef,
        *,
        update_state: bool,
        embedding_initialization: ArtifactRef | None = None,
        named_results: dict[str, ArtifactRef] | None = None,
        cell_key_override: str | None = None,
        feat_key_override: str | None = None,
    ) -> None:
        if not update_state:
            return
        candidate = self._artifact_chain_state(
            ref,
            embedding_initialization=embedding_initialization,
            named_results=named_results,
            cell_key_override=cell_key_override,
            feat_key_override=feat_key_override,
        )
        previous = read_assay_state(self.zw, candidate.assay)
        field_name = {
            "normalized": "normalized",
            "reduction": "reduction",
            "batch_correction": "batch_correction",
            "ann_index": "ann_index",
            "neighbors": "neighbors",
            "connectivity_map": "connectivity_map",
        }.get(ref.kind)
        if (
            previous is not None
            and previous.matches(candidate.cell_key, candidate.feat_key)
            and field_name is not None
            and getattr(previous, field_name) == ref
            and embedding_initialization is None
            and named_results is None
        ):
            candidate = previous
        write_assay_state(
            self.zw,
            candidate,
        )

    def _graph_cell_selection(
        self,
        graph_ref: ArtifactRef,
    ) -> ArtifactRef:
        if graph_ref.kind == "connectivity_map":
            state = self._artifact_chain_state(graph_ref)
            if state.normalized is None:
                raise ValueError("Graph has no normalized input")
            return self._artifact_input_ref(
                state.normalized,
                "cell_selection",
                "cell_selection",
            )
        if graph_ref.kind == "integrated_graph":
            status = self._require_complete_artifact(
                graph_ref,
                "integrated_graph",
            )
            raw_selection = (status.inputs or {}).get("cell_selection")
            if not isinstance(raw_selection, dict):
                raise ValueError("Integrated graph has no shared cell selection")
            selection = ArtifactRef.from_dict(raw_selection)
            self._require_complete_artifact(
                selection,
                "cell_selection",
            )
            return selection
        raise ValueError("Graph ref must be connectivity_map or integrated_graph")

    def _load_normalized_artifact(
        self,
        ref: ArtifactRef,
        *,
        batch_size: int,
    ) -> ChunkedArray:
        try:
            cached = self._normalizedArtifactCache.get(ref)
        except AttributeError:
            cached = None
        if cached is not None:
            return cached
        status = self._require_complete_artifact(ref, "normalized")
        group = as_zarr_group(self.zw[status.path], name=status.path)
        return ChunkedArray(
            as_zarr_array(group["data"], name="data"),
            block_size=batch_size,
            nthreads=self.nthreads,
        )

    @contextmanager
    def _cache_normalized_artifact(
        self,
        ref: ArtifactRef,
        local_cache: bool | str,
        batch_size: int,
    ) -> Iterator[None]:
        enabled, cache_base, remove_on_success = self._resolve_local_cache_plan(
            self.zarr_loc,
            self.z,
            local_cache,
        )
        if not enabled:
            yield
            return
        if cache_base is None:
            raise RuntimeError("Local cache path is missing")
        status = self._require_complete_artifact(ref, "normalized")
        source_group = as_zarr_group(
            self.zw[status.path],
            name=status.path,
        )
        source = as_zarr_array(source_group["data"], name="data")
        cache_path = os.path.join(
            cache_base,
            ref.artifact_id,
            "normed.zarr",
        )
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        staged = create_or_open_staged_normed_array(
            cache_path,
            (int(source.shape[0]), int(source.shape[1])),
        )
        if not bool(staged.attrs.get("complete", False)):
            copy_zarr_array(
                source,
                staged,
                msg="Staging normalized data locally",
            )
            staged.attrs["complete"] = True
        try:
            cache = self._normalizedArtifactCache
        except AttributeError:
            cache = {}
            self._normalizedArtifactCache = cache
        cache[ref] = ChunkedArray(
            staged,
            block_size=batch_size,
            nthreads=self.nthreads,
        )
        try:
            yield
        finally:
            cache.pop(ref, None)
            if remove_on_success:
                import shutil

                shutil.rmtree(cache_base, ignore_errors=True)

    def _load_reduction_stream(
        self,
        reduction_ref: ArtifactRef,
        *,
        batch_size: int,
    ) -> tuple[ReductionTransform, LazyTransformStream]:
        status = self._require_complete_artifact(
            reduction_ref,
            "reduction",
        )
        parameters = status.parameters or {}
        operation = status.operation
        method = (
            {
                "run_pca": "pca",
                "run_lsi": "lsi",
                "run_custom_reduction": "custom",
            }.get(operation)
            if operation is not None
            else None
        )
        if method is None:
            method = str(parameters.get("reduction_method", "pca"))
        normalized_ref = self._artifact_input_ref(
            reduction_ref,
            "normalized",
            "normalized",
        )
        scaling_ref = self._artifact_input_ref(
            reduction_ref,
            "feature_scaling",
            "feature_scaling",
        )
        normalized = self._load_normalized_artifact(
            normalized_ref,
            batch_size=batch_size,
        )
        scaling_group = as_zarr_group(
            self.zw[artifact_path(scaling_ref)],
            name=artifact_path(scaling_ref),
        )
        mu = np.asarray(as_zarr_array(scaling_group["mean"], name="mean")[:])
        sigma = np.asarray(as_zarr_array(scaling_group["scale"], name="scale")[:])
        reduction_group = as_zarr_group(
            self.zw[status.path],
            name=status.path,
        )
        loadings = (
            np.asarray(
                as_zarr_array(
                    reduction_group["loadings"],
                    name="loadings",
                )[:]
            )
            if "loadings" in reduction_group
            else None
        )
        raw_dims = parameters.get("dims")
        dims = int(
            raw_dims
            if raw_dims is not None
            else loadings.shape[1]
            if loadings is not None
            else 0
        )
        transform = ReductionTransform(
            data=normalized,
            method=method,
            dims=dims,
            loadings=loadings,
            use_for_pca=np.ones(normalized.shape[0], dtype=bool),
            mu=mu,
            sigma=sigma,
            batch_size=batch_size,
            nthreads=self.nthreads,
            rand_state=int(parameters.get("rand_state", 4466)),
            disable_scaling=not bool(parameters.get("feat_scaling", method == "pca")),
            lsi_skip_first=bool(
                parameters.get(
                    "skip_first",
                    parameters.get("lsi_skip_first", True),
                )
            ),
            lsi_params={},
        )
        stream = LazyTransformStream(
            data=normalized,
            transform=transform.transform,
            nthreads=self.nthreads,
            batch_size=batch_size,
        )
        return transform, stream

    def _coordinate_source(
        self,
        coordinates: ArtifactRef,
        *,
        batch_size: int,
    ) -> tuple[Any, int, int]:
        if coordinates.kind == "batch_correction":
            status = self._require_complete_artifact(
                coordinates,
                "batch_correction",
            )
            group = as_zarr_group(self.zw[status.path], name=status.path)
            data = ChunkedArray(
                as_zarr_array(group["data"], name="data"),
                block_size=batch_size,
                nthreads=self.nthreads,
            )
            return (
                ChunkedCoordinateStream(data, self.nthreads),
                int(data.shape[0]),
                int(data.shape[1]),
            )
        if coordinates.kind == "reduction":
            transform, stream = self._load_reduction_stream(
                coordinates,
                batch_size=batch_size,
            )
            dims = (
                int(transform.dims)
                if transform.dims is not None and transform.dims > 0
                else int(stream.data.shape[1])
            )
            return stream, int(stream.data.shape[0]), dims
        raise ValueError("Coordinates must reference reduction or batch_correction")

    def _plan_assay_artifact(
        self,
        assay: str,
        arguments: Any,
        *,
        required_arrays: tuple[str | ArrayRequirement, ...] = (),
        invalidate_cache: bool,
        reuse_validator: Callable[[ArtifactRef, zarr.Group], bool] | None = None,
    ) -> PlannedArtifact:
        record = arguments.to_record()
        execution_options = dict(record.execution_options)
        try:
            execution_options.update(self._artifactExecutionContext)
        except AttributeError:
            pass
        return plan_artifact(
            self.zw,
            scope="assay",
            assay=assay,
            kind=arguments.artifact_kind,
            operation=arguments.operation,
            parameters=record.parameters,
            inputs=record.inputs,
            execution_options=execution_options,
            invalidate_cache=invalidate_cache,
            required_arrays=required_arrays,
            reuse_validator=reuse_validator,
        )

    @contextmanager
    def _artifact_execution_context(
        self,
        options: dict[str, Any],
    ) -> Iterator[None]:
        try:
            previous = self._artifactExecutionContext
        except AttributeError:
            previous = {}
        self._artifactExecutionContext = {**previous, **options}
        try:
            yield
        finally:
            self._artifactExecutionContext = previous

    def run_normalization(
        self,
        from_assay: str | None = None,
        cell_key: str = "I",
        feat_key: str | None = None,
        *,
        log_transform: bool | None = None,
        renormalize_subset: bool | None = None,
        batch_size: int | None = None,
        update_state: bool = True,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        assay_name = from_assay or self._defaultAssay
        if assay_name is None:
            raise ValueError("No assay was provided and no default is configured")
        if feat_key is None:
            raise ValueError("feat_key is required for normalization")
        assay = self._get_assay(assay_name)
        stored_feat_key = feat_key if feat_key == "I" else f"{cell_key}__{feat_key}"
        if stored_feat_key not in assay.feats.columns:
            raise KeyError(
                f"Feature selection column {stored_feat_key!r} does not exist"
            )
        cell_values = np.asarray(self.cells.fetch_all(cell_key))
        feature_values = np.asarray(assay.feats.fetch_all(stored_feat_key))
        if cell_values.dtype != bool or feature_values.dtype != bool:
            raise TypeError("Cell and feature selections must be boolean")
        n_cells = int(cell_values.sum())
        n_features = int(feature_values.sum())
        if n_cells < 1 or n_features < 1:
            raise ValueError("Normalization requires selected cells and features")
        effective_batch_size = min(
            int(batch_size or assay.rawData.chunksize[0]),
            n_cells,
        )
        state = read_assay_state(self.zw, assay_name)
        stored_parameters: dict[str, Any] = {}
        if (
            state is not None
            and state.matches(cell_key, feat_key)
            and state.normalized is not None
        ):
            stored_parameters = (
                inspect_artifact(self.zw, state.normalized).parameters or {}
            )
        if log_transform is None:
            log_transform = bool(stored_parameters.get("log_transform", True))
        if renormalize_subset is None:
            renormalize_subset = bool(stored_parameters.get("renormalize_subset", True))
        cell_data = as_zarr_group(self.zw["cellData"], name="cellData")
        feature_data = as_zarr_group(
            assay.z["featureData"],
            name="featureData",
        )
        cell_selection = self._resolve_selection_input(
            metadata_group=cell_data,
            column=cell_key,
            values=cell_values,
            row_ids=np.asarray(self.cells.fetch_all("ids")),
            scope="datastore",
            kind="cell_selection",
            assay=None,
            invalidate_cache=invalidate_cache,
        )
        feature_selection = self._resolve_selection_input(
            metadata_group=feature_data,
            column=stored_feat_key,
            values=feature_values,
            row_ids=np.asarray(assay.feats.fetch_all("ids")),
            scope="assay",
            kind="feature_selection",
            assay=assay_name,
            invalidate_cache=invalidate_cache,
        )
        normalization_method = assay.normMethod
        if callable(normalization_method):
            method_qualname = str(getattr(normalization_method, "__qualname__", ""))
            if (
                "<locals>" in method_qualname or "<lambda>" in method_qualname
            ) and getattr(normalization_method, "artifact_identity", None) is None:
                raise ValueError(
                    "Dynamic normalization callables must define "
                    "artifact_identity for provenance"
                )
        raw_size_factor = getattr(assay, "sf", None)
        arguments = NormalizationArguments(
            from_assay=assay_name,
            cell_key=cell_key,
            feat_key=feat_key,
            cell_selection=cell_selection,
            feature_selection=feature_selection,
            normalization_method=normalization_method,
            size_factor=(
                float(cast(int | float, raw_size_factor))
                if raw_size_factor is not None
                else None
            ),
            log_transform=log_transform,
            renormalize_subset=renormalize_subset,
            batch_size=effective_batch_size,
            update_state=update_state,
            local_cache=False,
            invalidate_cache=invalidate_cache,
        )
        planned = self._plan_assay_artifact(
            assay_name,
            arguments,
            required_arrays=(
                ArrayRequirement(
                    "data",
                    shape=(n_cells, n_features),
                    dtype_kind="f",
                ),
            ),
            invalidate_cache=invalidate_cache,
        )
        self._ensure_dataset_fingerprint(assay_name)
        if not planned.reused:
            group = start_artifact(self.zw, planned)
            relative_path = artifact_path(planned.ref).removeprefix(f"{assay_name}/")
            assay.save_normalized_data(
                cell_key,
                feat_key,
                effective_batch_size,
                relative_path,
                log_transform,
                renormalize_subset,
                False,
                artifact_mode=True,
            )
            finish_artifact(group, planned)
        self._publish_current_artifact(
            planned.ref,
            update_state=update_state,
            cell_key_override=cell_key,
            feat_key_override=feat_key,
        )
        return planned.ref

    def _run_reduction_artifact(
        self,
        *,
        method: str,
        normalized: ArtifactRef | None,
        from_assay: str | None,
        dims: int,
        pca_cell_key: str | None,
        feat_scaling: bool,
        lsi_skip_first: bool,
        custom_loadings: np.ndarray | None,
        rand_state: int,
        batch_size: int | None,
        show_elbow_plot: bool,
        update_state: bool,
        invalidate_cache: bool,
    ) -> ArtifactRef:
        normalized_ref = normalized or self._selected_artifact(
            from_assay,
            "normalized",
            "normalized",
        )
        normalized_status = self._require_complete_artifact(
            normalized_ref,
            "normalized",
        )
        if normalized_ref.assay is None:
            raise ValueError("Normalized artifact has no assay")
        assay_name = normalized_ref.assay
        execution = normalized_status.execution_options or {}
        cell_key = execution.get("cell_key")
        feat_key = execution.get("feat_key")
        if not isinstance(cell_key, str) or not isinstance(feat_key, str):
            raise ValueError("Normalized artifact has no cell_key or feat_key")
        state = read_assay_state(self.zw, assay_name)
        if state is not None and state.normalized == normalized_ref:
            cell_key = state.cell_key
            feat_key = state.feat_key
        validate_normalized_artifact_selection(
            self.zw,
            normalized_ref,
            cell_key,
            feat_key,
        )
        data_group = as_zarr_group(
            self.zw[normalized_status.path],
            name=normalized_status.path,
        )
        data_array = as_zarr_array(data_group["data"], name="data")
        n_cells, n_features = map(int, data_array.shape)
        normalized_execution = normalized_status.execution_options or {}
        stored_batch_size = normalized_execution.get("batch_size")
        effective_batch_size = min(
            int(batch_size or stored_batch_size or data_array.chunks[0]),
            n_cells,
        )
        effective_dims = min(int(dims), n_cells)
        if effective_dims >= effective_batch_size:
            effective_dims = max(effective_batch_size - 1, 0)
        if custom_loadings is not None:
            if custom_loadings.shape[0] != n_features:
                raise ValueError("Custom loadings rows must match normalized features")
            effective_dims = int(custom_loadings.shape[1])
        pca_key = pca_cell_key or cell_key
        pca_selection = None
        if method == "pca":
            pca_values = np.asarray(self.cells.fetch_all(pca_key))
            if pca_values.dtype != bool:
                raise TypeError("pca_cell_key must reference a boolean column")
            cell_data = as_zarr_group(
                self.zw["cellData"],
                name="cellData",
            )
            pca_selection = self._resolve_selection_input(
                metadata_group=cell_data,
                column=pca_key,
                values=pca_values,
                row_ids=np.asarray(self.cells.fetch_all("ids")),
                scope="datastore",
                kind="cell_selection",
                assay=None,
                invalidate_cache=invalidate_cache,
            )
        enabled_scaling = method == "pca"
        scaling_arguments = FeatureScalingArguments(
            normalized=normalized_ref,
            enabled=enabled_scaling,
            calculation_batch_size=(effective_batch_size if enabled_scaling else None),
            batch_size=effective_batch_size,
            invalidate_cache=invalidate_cache,
        )
        scaling_shape = n_features if enabled_scaling else 0
        scaling_plan = self._plan_assay_artifact(
            assay_name,
            scaling_arguments,
            required_arrays=(
                ArrayRequirement(
                    "mean",
                    shape=(scaling_shape,),
                    dtype_kind="f",
                ),
                ArrayRequirement(
                    "scale",
                    shape=(scaling_shape,),
                    dtype_kind="f",
                ),
            ),
            invalidate_cache=invalidate_cache,
        )
        normalized_data = ChunkedArray(
            data_array,
            block_size=effective_batch_size,
            nthreads=self.nthreads,
        )
        if scaling_plan.reused:
            scaling_group = reused_artifact_group(
                self.zw,
                scaling_plan,
            )
            mu = np.asarray(as_zarr_array(scaling_group["mean"], name="mean")[:])
            sigma = np.asarray(as_zarr_array(scaling_group["scale"], name="scale")[:])
        else:
            if enabled_scaling:
                mu_raw, sigma_raw = normalized_data.mean_and_std(
                    nthreads=self.nthreads,
                    msg="Calculating normalization statistics",
                )
                mu = clean_array(mu_raw)
                sigma = clean_array(sigma_raw, 1)
            else:
                mu = np.array([], dtype=np.float64)
                sigma = np.array([], dtype=np.float64)
            scaling_group = start_artifact(self.zw, scaling_plan)
            mean_array = create_zarr_dataset(
                scaling_group,
                "mean",
                (100000,),
                "f8",
                mu.shape,
            )
            mean_array[:] = mu
            scale_array = create_zarr_dataset(
                scaling_group,
                "scale",
                (100000,),
                "f8",
                sigma.shape,
            )
            scale_array[:] = sigma
            finish_artifact(scaling_group, scaling_plan)
        if method == "pca":
            assert pca_selection is not None
            arguments: Any = PcaArguments(
                normalized=normalized_ref,
                feature_scaling=scaling_plan.ref,
                pca_cell_selection=pca_selection,
                pca_cell_key=pca_key,
                dims=effective_dims,
                feat_scaling=feat_scaling,
                batch_size=effective_batch_size,
                show_elbow_plot=show_elbow_plot,
                update_state=update_state,
                invalidate_cache=invalidate_cache,
            )
        elif method == "lsi":
            arguments = LsiArguments(
                normalized=normalized_ref,
                feature_scaling=scaling_plan.ref,
                dims=effective_dims,
                skip_first=lsi_skip_first,
                rand_state=rand_state,
                batch_size=effective_batch_size,
                update_state=update_state,
                invalidate_cache=invalidate_cache,
            )
        else:
            if custom_loadings is None:
                raise ValueError("Custom reduction requires loadings")
            arguments = CustomReductionArguments(
                normalized=normalized_ref,
                feature_scaling=scaling_plan.ref,
                loadings=custom_loadings,
                update_state=update_state,
                invalidate_cache=invalidate_cache,
            )
        required_arrays: tuple[str | ArrayRequirement, ...] = ()
        if effective_dims > 0:
            required_arrays = (
                ArrayRequirement(
                    "loadings",
                    shape=(n_features, effective_dims),
                    dtype_kind="f",
                ),
            )
        planned = self._plan_assay_artifact(
            assay_name,
            arguments,
            required_arrays=required_arrays,
            invalidate_cache=invalidate_cache,
        )
        transform = None
        if not planned.reused:
            use_for_pca = (
                self.cells.fetch(pca_key, key=cell_key)
                if method == "pca"
                else np.ones(n_cells, dtype=bool)
            )
            transform = ReductionTransform(
                data=normalized_data,
                method=method,
                dims=effective_dims,
                loadings=custom_loadings,
                use_for_pca=use_for_pca,
                mu=mu,
                sigma=sigma,
                batch_size=effective_batch_size,
                nthreads=self.nthreads,
                rand_state=rand_state,
                disable_scaling=not feat_scaling,
                lsi_skip_first=lsi_skip_first,
                lsi_params={},
            )
            reduction_group = start_artifact(self.zw, planned)
            if transform.loadings is not None:
                output = create_zarr_dataset(
                    reduction_group,
                    "loadings",
                    normalized_data.chunksize,
                    "f8",
                    transform.loadings.shape,
                )
                output[:, :] = transform.loadings
            finish_artifact(reduction_group, planned)
        if show_elbow_plot and method == "pca":
            from ...plotting import elbow

            if transform is None or transform.pca is None:
                logger.warning("PCA was not fitted so no elbow plot is available")
            else:
                elbow(
                    variance_explained=(100 * transform.pca.explained_variance_ratio_),
                    show=True,
                )
        self._publish_current_artifact(
            planned.ref,
            update_state=update_state,
        )
        return planned.ref

    def run_pca(
        self,
        normalized: ArtifactRef | None = None,
        *,
        from_assay: str | None = None,
        dims: int = 11,
        pca_cell_key: str | None = None,
        feat_scaling: bool = True,
        batch_size: int | None = None,
        show_elbow_plot: bool = False,
        update_state: bool = True,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        return self._run_reduction_artifact(
            method="pca",
            normalized=normalized,
            from_assay=from_assay,
            dims=dims,
            pca_cell_key=pca_cell_key,
            feat_scaling=feat_scaling,
            lsi_skip_first=False,
            custom_loadings=None,
            rand_state=4466,
            batch_size=batch_size,
            show_elbow_plot=show_elbow_plot,
            update_state=update_state,
            invalidate_cache=invalidate_cache,
        )

    def run_lsi(
        self,
        normalized: ArtifactRef | None = None,
        *,
        from_assay: str | None = None,
        dims: int = 11,
        skip_first: bool = True,
        rand_state: int = 4466,
        batch_size: int | None = None,
        update_state: bool = True,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        return self._run_reduction_artifact(
            method="lsi",
            normalized=normalized,
            from_assay=from_assay,
            dims=dims,
            pca_cell_key=None,
            feat_scaling=False,
            lsi_skip_first=skip_first,
            custom_loadings=None,
            rand_state=rand_state,
            batch_size=batch_size,
            show_elbow_plot=False,
            update_state=update_state,
            invalidate_cache=invalidate_cache,
        )

    def run_custom_reduction(
        self,
        loadings: np.ndarray,
        normalized: ArtifactRef | None = None,
        *,
        from_assay: str | None = None,
        batch_size: int | None = None,
        update_state: bool = True,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        return self._run_reduction_artifact(
            method="custom",
            normalized=normalized,
            from_assay=from_assay,
            dims=int(loadings.shape[1]),
            pca_cell_key=None,
            feat_scaling=False,
            lsi_skip_first=False,
            custom_loadings=np.asarray(loadings),
            rand_state=4466,
            batch_size=batch_size,
            show_elbow_plot=False,
            update_state=update_state,
            invalidate_cache=invalidate_cache,
        )

    def run_harmony(
        self,
        batch_columns: list[str],
        reduction: ArtifactRef | None = None,
        *,
        from_assay: str | None = None,
        harmony_params: dict[str, Any] | None = None,
        batch_size: int | None = None,
        update_state: bool = True,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        reduction_ref = reduction or self._selected_artifact(
            from_assay,
            "reduction",
            "reduction",
        )
        reduction_status = self._require_complete_artifact(
            reduction_ref,
            "reduction",
        )
        if reduction_ref.assay is None:
            raise ValueError("Reduction artifact has no assay")
        normalized_ref = self._artifact_input_ref(
            reduction_ref,
            "normalized",
            "normalized",
        )
        normalized_status = self._require_complete_artifact(
            normalized_ref,
            "normalized",
        )
        execution = normalized_status.execution_options or {}
        cell_key = execution.get("cell_key")
        feat_key = execution.get("feat_key")
        if not isinstance(cell_key, str) or not isinstance(feat_key, str):
            raise ValueError("Normalized artifact has no cell_key or feat_key")
        state = read_assay_state(self.zw, reduction_ref.assay)
        if state is not None and state.normalized == normalized_ref:
            cell_key = state.cell_key
            feat_key = state.feat_key
        validate_normalized_artifact_selection(
            self.zw,
            normalized_ref,
            cell_key,
            feat_key,
        )
        if not isinstance(batch_columns, list) or not batch_columns:
            raise ValueError("batch_columns must be a non-empty list")
        batches = pd.DataFrame(
            {
                column: self.cells.fetch(column, key=cell_key).astype(object)
                for column in batch_columns
            }
        )
        reduction_parameters = reduction_status.parameters or {}
        effective_batch_size = int(
            batch_size or reduction_parameters.get("batch_size", len(batches))
        )
        effective_batch_size = min(effective_batch_size, len(batches))
        cell_selection = self._artifact_input_ref(
            normalized_ref,
            "cell_selection",
            "cell_selection",
        )
        batch_values = resolve_metadata_snapshot(
            self.zw,
            values=batches.to_numpy(),
            row_ids=np.asarray(self.cells.fetch("ids", key=cell_key)),
            operation="snapshot_metadata",
            parameters={"columns": batch_columns},
            inputs={"cell_selection": cell_selection},
            source_columns=batch_columns,
            invalidate_cache=invalidate_cache,
        )
        arguments = HarmonyArguments(
            reduction=reduction_ref,
            batch_values=batch_values,
            batch_columns=tuple(batch_columns),
            harmony_parameters=harmony_params or {},
            batch_size=effective_batch_size,
            force_refit=False,
            invalidate_cache=invalidate_cache,
        )
        _source, n_cells, dims = self._coordinate_source(
            reduction_ref,
            batch_size=effective_batch_size,
        )
        planned = self._plan_assay_artifact(
            reduction_ref.assay,
            arguments,
            required_arrays=(
                ArrayRequirement(
                    "data",
                    shape=(n_cells, dims),
                    dtype_kind="f",
                ),
                ArrayRequirement("assignments", dtype_kind="f"),
                ArrayRequirement("centroids", dtype_kind="f"),
                ArrayRequirement("sigma", dtype_kind="f"),
                ArrayRequirement("ridge", dtype_kind="f"),
            ),
            invalidate_cache=invalidate_cache,
        )
        if not planned.reused:
            _transform, stream = self._load_reduction_stream(
                reduction_ref,
                batch_size=effective_batch_size,
            )
            correction = BatchCorrectionStage(
                stream=stream,
                batches=batches,
                parameters=harmony_params or {},
                corrected_data=None,
                nthreads=self.nthreads,
            )
            corrected = correction.ensure_corrected()
            result = correction.result
            if result is None:
                raise RuntimeError("Harmony did not return fit metadata")
            group = start_artifact(self.zw, planned)
            output = create_zarr_dataset(
                group,
                "data",
                corrected.chunksize,
                "f8",
                corrected.shape,
            )
            start = 0
            for block in corrected.blocks:
                values = np.asarray(block.compute())
                stop = start + values.shape[0]
                output[start:stop, :] = values
                start = stop
            for name, values in (
                ("assignments", result.assignments),
                ("centroids", result.centroids),
                ("sigma", result.sigma),
                ("ridge", result.ridge),
            ):
                result_array = create_zarr_dataset(
                    group,
                    name,
                    tuple(max(int(size), 1) for size in values.shape),
                    "f8",
                    values.shape,
                )
                result_array[...] = values
            group.attrs["batch_levels"] = [
                list(levels) for levels in result.batch_levels
            ]
            finish_artifact(group, planned)
        self._publish_current_artifact(
            planned.ref,
            update_state=update_state,
        )
        return planned.ref

    def _build_embedding_initialization(
        self,
        reduction: ArtifactRef,
        *,
        n_centroids: int,
        rand_state: int,
        batch_size: int,
        invalidate_cache: bool,
    ) -> ArtifactRef:
        if reduction.assay is None:
            raise ValueError("Reduction artifact has no assay")
        _transform, stream = self._load_reduction_stream(
            reduction,
            batch_size=batch_size,
        )
        n_cells = int(stream.data.shape[0])
        coordinate_dims = (
            int(_transform.dims)
            if _transform.dims is not None and _transform.dims > 0
            else int(stream.data.shape[1])
        )
        effective_clusters = min(
            max(int(n_centroids), 2),
            batch_size,
            n_cells,
        )
        if (
            n_cells * coordinate_dims * np.dtype(np.float64).itemsize
            <= EMBEDDING_CACHE_MAX_BYTES
        ):
            stream.cache("Building cell embeddings")
        arguments = EmbeddingInitializationArguments(
            reduction=reduction,
            n_centroids=effective_clusters,
            rand_state=rand_state,
            batch_size=batch_size,
            invalidate_cache=invalidate_cache,
        )
        planned = self._plan_assay_artifact(
            reduction.assay,
            arguments,
            required_arrays=(
                ArrayRequirement(
                    "cluster_centers",
                    shape=(effective_clusters, coordinate_dims),
                    dtype_kind="f",
                ),
                ArrayRequirement(
                    "cluster_labels",
                    shape=(n_cells,),
                    dtype_kind="f",
                ),
            ),
            invalidate_cache=invalidate_cache,
        )
        if not planned.reused:
            initialization = KMeansInitializationStage.fit(
                stream=stream,
                n_clusters=effective_clusters,
                rand_state=rand_state,
                nthreads=self.nthreads,
                enabled=True,
            )
            if initialization.model is None:
                raise RuntimeError("K-means initialization did not fit")
            group = start_artifact(self.zw, planned)
            centers = create_zarr_dataset(
                group,
                "cluster_centers",
                (1000, 1000),
                "f8",
                initialization.model.cluster_centers_.shape,
            )
            centers[:, :] = initialization.model.cluster_centers_
            labels = create_zarr_dataset(
                group,
                "cluster_labels",
                (100000,),
                "f8",
                initialization.labels.shape,
            )
            labels[:] = initialization.labels
            finish_artifact(group, planned)
        return planned.ref

    def _build_mapping_reference_artifact(
        self,
        *,
        reduction: ArtifactRef,
        batch_correction: ArtifactRef,
        ann_index: ArtifactRef,
        neighbors: ArtifactRef,
        invalidate_cache: bool,
    ) -> ArtifactRef:
        from ...mapping.artifact import write_artifact_mapping_reference
        from ...mapping.confidence import _distance_quantile_summary
        from ...mapping.models import SymphonyReferenceModel
        from ...mapping.symphony import weighted_centroids

        if reduction.assay is None:
            raise ValueError("Reduction artifact has no assay")
        reduction_status = self._require_complete_artifact(
            reduction,
            "reduction",
        )
        if reduction_status.operation != "run_pca" or not bool(
            (reduction_status.parameters or {}).get(
                "feat_scaling",
                False,
            )
        ):
            raise ValueError("Mapping references require PCA with feature scaling")
        if (
            self._artifact_input_ref(
                batch_correction,
                "reduction",
                "reduction",
            )
            != reduction
        ):
            raise ValueError("Batch correction uses another reduction")
        if (
            self._artifact_input_ref(
                ann_index,
                "coordinates",
                "batch_correction",
            )
            != batch_correction
        ):
            raise ValueError("ANN index uses another batch correction")
        if (
            self._artifact_input_ref(
                neighbors,
                "ann_index",
                "ann_index",
            )
            != ann_index
            or self._artifact_input_ref(
                neighbors,
                "coordinates",
                "batch_correction",
            )
            != batch_correction
        ):
            raise ValueError("Neighbors use another corrected graph chain")
        normalized = self._artifact_input_ref(
            reduction,
            "normalized",
            "normalized",
        )
        cell_selection = self._artifact_input_ref(
            normalized,
            "cell_selection",
            "cell_selection",
        )
        feature_selection = self._artifact_input_ref(
            normalized,
            "feature_selection",
            "feature_selection",
        )
        planned = plan_artifact(
            self.zw,
            scope="assay",
            assay=reduction.assay,
            kind="mapping_reference",
            operation="build_mapping_reference",
            parameters={"method": "symphony"},
            inputs={
                "reduction": reduction,
                "batch_correction": batch_correction,
                "ann_index": ann_index,
                "neighbors": neighbors,
                "cell_selection": cell_selection,
                "feature_selection": feature_selection,
            },
            execution_options=dict(getattr(self, "_artifactExecutionContext", {})),
            invalidate_cache=invalidate_cache,
            required_arrays=(
                "feature_ids",
                "feature_means",
                "feature_scales",
                "loadings",
                "centroids",
                "raw_centroids",
                "corrected_centroids",
                "cluster_mass",
                "sigma",
                "reference_distance_quantiles",
                "reference_distance_values",
            ),
            required_attributes=(
                AttributeRequirement(
                    "correction_ridge",
                    expected_types=(int, float),
                    predicate=lambda value: (
                        not isinstance(value, bool) and np.isfinite(value)
                    ),
                ),
                AttributeRequirement(
                    "reference_metadata",
                    expected_types=(dict,),
                ),
            ),
        )
        if planned.reused:
            return planned.ref
        reduction_parameters = reduction_status.parameters or {}
        batch_size = int(reduction_parameters.get("batch_size", 1000))
        transform, stream = self._load_reduction_stream(
            reduction,
            batch_size=batch_size,
        )
        if transform.loadings is None:
            raise RuntimeError("Mapping reference requires persisted PCA loadings")
        original = np.vstack(
            stream.parallel_blocks(
                "Loading reference coordinates",
            )
        )
        correction_status = self._require_complete_artifact(
            batch_correction,
            "batch_correction",
        )
        correction_group = as_zarr_group(
            self.zw[correction_status.path],
            name=correction_status.path,
        )
        corrected = np.asarray(as_zarr_array(correction_group["data"], name="data")[:])
        assignments = np.asarray(
            as_zarr_array(
                correction_group["assignments"],
                name="assignments",
            )[:]
        )
        cluster_mass, raw_centroids = weighted_centroids(
            original,
            assignments,
        )
        _, corrected_centroids = weighted_centroids(
            corrected,
            assignments,
        )
        ridge = np.asarray(as_zarr_array(correction_group["ridge"], name="ridge")[:])
        ridge_values = np.diag(ridge)[1:]
        correction_ridge = (
            float(np.mean(ridge_values[ridge_values > 0]))
            if np.any(ridge_values > 0)
            else 1.0
        )
        scaling = self._artifact_input_ref(
            reduction,
            "feature_scaling",
            "feature_scaling",
        )
        scaling_group = as_zarr_group(
            self.zw[artifact_path(scaling)],
            name=artifact_path(scaling),
        )
        model = SymphonyReferenceModel(
            feature_means=np.asarray(
                as_zarr_array(scaling_group["mean"], name="mean")[:]
            ),
            feature_scales=np.asarray(
                as_zarr_array(scaling_group["scale"], name="scale")[:]
            ),
            loadings=transform.loadings,
            centroids=np.asarray(
                as_zarr_array(
                    correction_group["centroids"],
                    name="centroids",
                )[:]
            ).T,
            raw_centroids=raw_centroids,
            corrected_centroids=corrected_centroids,
            cluster_mass=cluster_mass,
            sigma=np.asarray(
                as_zarr_array(
                    correction_group["sigma"],
                    name="sigma",
                )[:]
            ),
            correction_ridge=correction_ridge,
        )
        normalized_execution = (
            inspect_artifact(self.zw, normalized).execution_options or {}
        )
        feature_column = (
            normalized_execution.get("feat_key")
            if normalized_execution.get("feat_key") == "I"
            else (
                f"{normalized_execution.get('cell_key')}__"
                f"{normalized_execution.get('feat_key')}"
            )
        )
        assay = self._get_assay(reduction.assay)
        feature_ids = assay.feats.fetch("ids", key=str(feature_column))
        correction_parameters = correction_status.parameters or {}
        metadata = {
            "assay": reduction.assay,
            "cell_key": normalized_execution.get("cell_key"),
            "feature_key": normalized_execution.get("feat_key"),
            "batch_columns": correction_parameters.get(
                "batch_columns",
                [],
            ),
            "harmony_parameters": correction_parameters.get(
                "harmony_parameters",
                {},
            ),
            "batch_levels": correction_group.attrs.get(
                "batch_levels",
                [],
            ),
            "method": "symphony",
            "normalization_parameters": (
                inspect_artifact(self.zw, normalized).parameters or {}
            ),
        }
        neighbors_group = as_zarr_group(
            self.zw[artifact_path(neighbors)],
            name=artifact_path(neighbors),
        )
        distance_quantiles, distance_values = _distance_quantile_summary(
            as_zarr_array(
                neighbors_group["distances"],
                name="distances",
            )
        )
        group = start_artifact(self.zw, planned)
        write_artifact_mapping_reference(
            group,
            model,
            feature_ids,
            metadata,
            distance_quantiles,
            distance_values,
        )
        finish_artifact(group, planned)
        return planned.ref

    def build_ann_index(
        self,
        coordinates: ArtifactRef | None = None,
        *,
        from_assay: str | None = None,
        ann_metric: str = "l2",
        ann_efc: int = 50,
        ann_ef: int = 50,
        ann_m: int = 48,
        ann_parallel: bool = False,
        rand_state: int = 4466,
        batch_size: int | None = None,
        ann_index_fetcher: Callable | None = None,
        ann_index_saver: Callable | None = None,
        update_state: bool = True,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        if coordinates is None:
            assay = from_assay or self._defaultAssay
            if assay is None:
                raise ValueError("No assay was provided and no default is configured")
            state = read_assay_state(self.zw, assay)
            if state is None or state.reduction is None:
                raise KeyError("AssayState has no selected reduction")
            coordinates = (
                state.batch_correction
                if state.batch_correction is not None
                else state.reduction
            )
        if coordinates.assay is None:
            raise ValueError("Coordinate artifact has no assay")
        reduction_ref = (
            self._artifact_input_ref(
                coordinates,
                "reduction",
                "reduction",
            )
            if coordinates.kind == "batch_correction"
            else coordinates
        )
        reduction_parameters = inspect_artifact(self.zw, reduction_ref).parameters or {}
        requested_batch_size = int(
            batch_size or reduction_parameters.get("batch_size") or 1000
        )
        coordinate_source, n_cells, dims = self._coordinate_source(
            coordinates,
            batch_size=requested_batch_size,
        )
        effective_batch_size = min(
            requested_batch_size,
            n_cells,
        )
        parallel_threads = self.nthreads if ann_parallel else None
        arguments = AnnIndexArguments(
            coordinates=coordinates,
            ann_metric=ann_metric,
            ann_efc=ann_efc,
            ann_ef=ann_ef,
            ann_m=ann_m,
            rand_state=rand_state,
            ann_parallel=ann_parallel,
            parallel_threads=parallel_threads,
            batch_size=effective_batch_size,
            ann_index_fetcher=ann_index_fetcher,
            ann_index_saver=ann_index_saver,
            local_cache=False,
            invalidate_cache=invalidate_cache,
        )

        def valid_ann_artifact(
            _ref: ArtifactRef,
            group: zarr.Group,
        ) -> bool:
            try:
                load_ann_index(group, ann_metric, dims)
            except (FileNotFoundError, RuntimeError, ValueError):
                return False
            return True

        planned = self._plan_assay_artifact(
            coordinates.assay,
            arguments,
            required_arrays=(ArrayRequirement("ann_idx_bytes", dtype_kind="u"),),
            invalidate_cache=invalidate_cache,
            reuse_validator=valid_ann_artifact,
        )
        if not planned.reused:
            ann_idx = None
            if ann_index_fetcher is not None:
                ann_idx = self._resolve_ann_index(
                    artifact_path(planned.ref),
                    ann_metric,
                    dims,
                    ann_index_fetcher=ann_index_fetcher,
                )
            if ann_idx is None:
                if (
                    isinstance(coordinate_source, LazyTransformStream)
                    and n_cells * dims * np.dtype(np.float64).itemsize
                    <= EMBEDDING_CACHE_MAX_BYTES
                ):
                    coordinate_source.cache("Building cell embeddings")
                ann_idx = AnnIndexStage.fit(
                    coordinates=coordinate_source,
                    metric=ann_metric,
                    dims=dims,
                    n_cells=n_cells,
                    ef_construction=ann_efc,
                    ef=ann_ef,
                    m=ann_m,
                    rand_state=rand_state,
                    ann_threads=(self.nthreads if ann_parallel else 1),
                )
            else:
                ann_idx = AnnIndexStage.configure(
                    ann_idx,
                    ef=ann_ef,
                    threads=(self.nthreads if ann_parallel else 1),
                )
            group = start_artifact(self.zw, planned)
            self._persist_ann_index(
                artifact_path(planned.ref),
                ann_idx,
                ann_index_saver=ann_index_saver,
            )
            finish_artifact(group, planned)
        self._publish_current_artifact(
            planned.ref,
            update_state=update_state,
        )
        return planned.ref

    def query_neighbors(
        self,
        ann_index: ArtifactRef | None = None,
        *,
        from_assay: str | None = None,
        coordinates: ArtifactRef | None = None,
        k: int = 11,
        batch_size: int | None = None,
        update_state: bool = True,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        ann_ref = ann_index or self._selected_artifact(
            from_assay,
            "ann_index",
            "ann_index",
        )
        ann_status = self._require_complete_artifact(
            ann_ref,
            "ann_index",
        )
        if ann_ref.assay is None:
            raise ValueError("ANN artifact has no assay")
        raw_coordinates = (ann_status.inputs or {}).get("coordinates")
        if not isinstance(raw_coordinates, dict):
            raise ValueError("ANN artifact has no coordinates input")
        stored_coordinates = ArtifactRef.from_dict(raw_coordinates)
        if stored_coordinates.kind not in {
            "reduction",
            "batch_correction",
        }:
            raise ValueError("ANN coordinates must be reduction or batch_correction")
        self._require_complete_artifact(
            stored_coordinates,
            stored_coordinates.kind,
        )
        if coordinates is not None and coordinates != stored_coordinates:
            raise ValueError("coordinates do not match the ANN artifact input")
        ann_execution = ann_status.execution_options or {}
        requested_batch_size = int(
            batch_size or ann_execution.get("batch_size") or 1000
        )
        coordinate_source, n_cells, dims = self._coordinate_source(
            stored_coordinates,
            batch_size=requested_batch_size,
        )
        effective_k = min(int(k), n_cells - 1)
        if effective_k < 1:
            raise ValueError("Neighbor queries require at least two cells")
        effective_batch_size = min(
            requested_batch_size,
            n_cells,
        )
        arguments = NeighborQueryArguments(
            ann_index=ann_ref,
            coordinates=stored_coordinates,
            k=effective_k,
            batch_size=effective_batch_size,
            invalidate_cache=invalidate_cache,
        )
        planned = self._plan_assay_artifact(
            ann_ref.assay,
            arguments,
            required_arrays=(
                ArrayRequirement(
                    "indices",
                    shape=(n_cells, effective_k),
                    dtype_kind="u",
                ),
                ArrayRequirement(
                    "distances",
                    shape=(n_cells, effective_k),
                    dtype_kind="f",
                ),
            ),
            invalidate_cache=invalidate_cache,
        )
        if not planned.reused:
            if (
                isinstance(coordinate_source, LazyTransformStream)
                and n_cells * dims * np.dtype(np.float64).itemsize
                <= EMBEDDING_CACHE_MAX_BYTES
            ):
                coordinate_source.cache("Building cell embeddings")
            ann_parameters = ann_status.parameters or {}
            ann_idx = self._resolve_ann_index(
                ann_status.path,
                str(ann_parameters.get("ann_metric", "l2")),
                dims,
            )
            if ann_idx is None:
                raise RuntimeError("ANN artifact has no readable index")
            ann_idx = AnnIndexStage.configure(
                ann_idx,
                ef=int(ann_parameters.get("ann_ef", 50)),
                threads=(int(ann_parameters.get("parallel_threads") or 1)),
            )
            query = NeighborQueryStage(ann_idx, effective_k)
            group = start_artifact(self.zw, planned)
            indices_array = create_zarr_dataset(
                group,
                "indices",
                (effective_batch_size,),
                "u8",
                (n_cells, effective_k),
            )
            distances_array = create_zarr_dataset(
                group,
                "distances",
                (effective_batch_size,),
                "f8",
                (n_cells, effective_k),
            )
            start = 0
            missed_recall = 0
            for block in coordinate_source.iter_coordinate_blocks(
                "Identifying neighbors"
            ):
                stop = start + len(block)
                result = query.query(
                    block,
                    self_indices=np.arange(start, stop),
                )
                block_indices, block_distances, missed = cast(
                    tuple[np.ndarray, np.ndarray, int],
                    result,
                )
                indices_array[start:stop, :] = block_indices
                distances_array[start:stop, :] = block_distances
                missed_recall += missed
                start = stop
            group.attrs["n_cells"] = n_cells
            group.attrs["n_neighbors"] = effective_k
            group.attrs["recall"] = 100.0 * (n_cells - missed_recall) / n_cells
            finish_artifact(group, planned)
        self._publish_current_artifact(
            planned.ref,
            update_state=update_state,
        )
        return planned.ref

    def build_connectivity_map(
        self,
        neighbors: ArtifactRef | None = None,
        *,
        from_assay: str | None = None,
        local_connectivity: float = 1.0,
        bandwidth: float = 1.5,
        batch_size: int | None = None,
        update_state: bool = True,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        neighbors_ref = neighbors or self._selected_artifact(
            from_assay,
            "neighbors",
            "neighbors",
        )
        status = self._require_complete_artifact(
            neighbors_ref,
            "neighbors",
        )
        if neighbors_ref.assay is None:
            raise ValueError("Neighbors artifact has no assay")
        group = as_zarr_group(self.zw[status.path], name=status.path)
        indices = as_zarr_array(group["indices"], name="indices")
        n_cells, n_neighbors = map(int, indices.shape)
        neighbor_execution = status.execution_options or {}
        effective_batch_size = min(
            int(batch_size or neighbor_execution.get("batch_size") or 1000),
            n_cells,
        )
        arguments = ConnectivityMapArguments(
            neighbors=neighbors_ref,
            local_connectivity=local_connectivity,
            bandwidth=bandwidth,
            batch_size=effective_batch_size,
            invalidate_cache=invalidate_cache,
        )
        planned = self._plan_assay_artifact(
            neighbors_ref.assay,
            arguments,
            required_arrays=(
                ArrayRequirement(
                    "edges",
                    shape=(n_cells * n_neighbors, 2),
                    dtype_kind="u",
                ),
                ArrayRequirement(
                    "weights",
                    shape=(n_cells * n_neighbors,),
                    dtype_kind="f",
                ),
            ),
            invalidate_cache=invalidate_cache,
        )
        if not planned.reused:
            from ...neighbors.graph_store import smoothen_dists

            output = start_artifact(self.zw, planned)
            smoothen_dists(
                output,
                indices,
                as_zarr_array(group["distances"], name="distances"),
                local_connectivity,
                bandwidth,
                effective_batch_size,
            )
            output.attrs["n_cells"] = n_cells
            output.attrs["n_neighbors"] = n_neighbors
            finish_artifact(output, planned)
        self._publish_current_artifact(
            planned.ref,
            update_state=update_state,
        )
        return planned.ref

    def _resolve_graph_plan(
        self,
        from_assay: str | None = None,
        cell_key: str | None = None,
        feat_key: str | None = None,
        pca_cell_key: str | None = None,
        reduction_method: str = "auto",
        dims: int | None = None,
        k: int | None = None,
        ann_metric: str | None = None,
        ann_efc: int | None = None,
        ann_ef: int | None = None,
        ann_m: int | None = None,
        ann_parallel: bool = False,
        rand_state: int | None = None,
        n_centroids: int | None = None,
        batch_size: int | None = None,
        log_transform: bool | None = None,
        renormalize_subset: bool | None = None,
        local_connectivity: float | None = None,
        bandwidth: float | None = None,
        update_keys: bool = True,
        return_ann_object: bool = False,
        custom_loadings: np.ndarray | None = None,
        feat_scaling: bool = True,
        lsi_skip_first: bool = True,
        harmonize: bool = False,
        batch_columns: list[str] | None = None,
        show_elbow_plot: bool = False,
        ann_index_fetcher: Callable | None = None,
        ann_index_saver: Callable | None = None,
        local_cache: bool | str = "auto",
        harmony_params: dict[str, Any] | None = None,
        force_harmony_refit: bool = False,
        invalidate_cache: bool = False,
    ) -> GraphBuildPlan:
        if from_assay is None:
            from_assay = self._defaultAssay
        assay = self._get_assay(from_assay)
        if cell_key is None:
            cell_key = "I"
        if feat_key is None:
            bool_col_parts = [
                x.split("__", 1)
                for x in assay.feats.columns
                if assay.feats.get_dtype(x) == bool and x != "I"  # noqa: E721
            ]
            bool_cols_msg = " ".join(f"{part[1]}({part[0]})" for part in bool_col_parts)
            raise ValueError(
                "ERROR: You have to choose which features that should be used for graph construction. "
                "Ideally you should have performed a feature selection step before making this graph. "
                "Feature selection step adds a column to your feature table. \n"
                "You have following boolean columns in the feature "
                f"metadata of assay {from_assay} which you can choose from: {bool_cols_msg}\n The values in "
                f"brackets indicate the cell_key for which the feat_key is available. Choosing 'I' "
                f"as `feat_key` means that you will use all the genes for graph creation."
            )
        if batch_size is None:
            state = read_assay_state(self.zw, from_assay)
            if (
                state is not None
                and state.matches(cell_key, feat_key)
                and state.reduction is not None
            ):
                state_reduction = (
                    inspect_artifact(self.zw, state.reduction).parameters or {}
                )
                stored_batch_size = state_reduction.get("batch_size")
                batch_size = (
                    int(cast(int | float | str, stored_batch_size))
                    if stored_batch_size is not None
                    else None
                )
            if batch_size is None:
                batch_size = assay.rawData.chunksize[0]
        if custom_loadings is not None:
            reduction_method = "custom"
            dims = custom_loadings.shape[1]
            logger.info(
                f"`dims` parameter and its default value ignored as using custom loadings "
                f"with {dims} dims"
            )

        parameters = self._resolve_graph_parameters(
            from_assay=from_assay,
            cell_key=cell_key,
            feat_key=feat_key,
            log_transform=log_transform,
            renormalize_subset=renormalize_subset,
            reduction_method=reduction_method,
            dims=dims,
            pca_cell_key=pca_cell_key,
            ann_metric=ann_metric,
            ann_efc=ann_efc,
            ann_ef=ann_ef,
            ann_m=ann_m,
            rand_state=rand_state,
            k=k,
            n_centroids=n_centroids,
            local_connectivity=local_connectivity,
            bandwidth=bandwidth,
            feat_scaling=feat_scaling,
            lsi_skip_first=lsi_skip_first,
            harmonize=harmonize,
            batch_columns=batch_columns,
            harmony_params=harmony_params,
        )
        n_active_cells = len(self.cells.active_index(cell_key))
        effective_batch_size = min(int(batch_size), n_active_cells)
        if effective_batch_size < 1:
            raise ValueError("Graph construction requires at least one active cell")
        if n_active_cells < 2:
            raise ValueError("Graph construction requires at least two active cells")
        effective_dims = parameters.dims
        if custom_loadings is None:
            effective_dims = min(effective_dims, len(self.cells.active_index(cell_key)))
            if effective_dims >= effective_batch_size:
                effective_dims = max(effective_batch_size - 1, 0)
        effective_centroids = min(
            max(parameters.n_centroids, 2),
            effective_batch_size,
        )
        parameters = replace(
            parameters,
            dims=effective_dims,
            n_centroids=effective_centroids,
            k=min(parameters.k, n_active_cells - 1),
        )
        if parameters.harmonize:
            if parameters.batch_columns is None:
                raise ValueError("Harmonization requested but no batches provided")
            if isinstance(parameters.batch_columns, list) is False:
                raise ValueError(
                    "batches must be a list of columns in cell metadata that represent batches"
                )
            for column in parameters.batch_columns:
                self.cells.fetch(column, key=cell_key)
        return GraphBuildPlan(
            data_inputs=GraphDataInputs(
                assay=assay,
                from_assay=from_assay,
                cell_key=cell_key,
                feat_key=feat_key,
                custom_loadings=custom_loadings,
            ),
            parameters=parameters,
            options=GraphExecutionOptions(
                batch_size=effective_batch_size,
                update_keys=update_keys,
                return_ann_object=return_ann_object,
                show_elbow_plot=show_elbow_plot,
                ann_parallel=ann_parallel,
                ann_index_fetcher=ann_index_fetcher,
                ann_index_saver=ann_index_saver,
                local_cache=local_cache,
                force_harmony_refit=force_harmony_refit,
                invalidate_cache=invalidate_cache,
            ),
        )

    def _execute_atomic_graph_plan(
        self,
        plan: GraphBuildPlan,
        normalized: ArtifactRef,
    ) -> AnnStream | None:
        params = plan.parameters
        inputs = plan.data_inputs
        options = plan.options
        if params.reduction_method == "pca":
            reduction = self.run_pca(
                normalized,
                dims=params.dims,
                pca_cell_key=params.pca_cell_key,
                feat_scaling=params.feat_scaling,
                batch_size=options.batch_size,
                show_elbow_plot=options.show_elbow_plot,
                update_state=False,
                invalidate_cache=options.invalidate_cache,
            )
        elif params.reduction_method == "lsi":
            reduction = self.run_lsi(
                normalized,
                dims=params.dims,
                skip_first=params.lsi_skip_first,
                rand_state=params.rand_state,
                batch_size=options.batch_size,
                update_state=False,
                invalidate_cache=options.invalidate_cache,
            )
        elif params.reduction_method == "custom":
            if inputs.custom_loadings is None:
                raise ValueError("Custom reduction requires custom_loadings")
            reduction = self.run_custom_reduction(
                inputs.custom_loadings,
                normalized,
                batch_size=options.batch_size,
                update_state=False,
                invalidate_cache=options.invalidate_cache,
            )
        else:
            raise ValueError(
                f"Unsupported reduction method {params.reduction_method!r}"
            )
        coordinates = reduction
        if params.harmonize:
            coordinates = self.run_harmony(
                params.batch_columns or [],
                reduction,
                harmony_params=params.harmony_params,
                batch_size=options.batch_size,
                update_state=False,
                invalidate_cache=(
                    options.invalidate_cache or options.force_harmony_refit
                ),
            )
        ann_index_ref = self.build_ann_index(
            coordinates,
            ann_metric=params.ann_metric,
            ann_efc=params.ann_efc,
            ann_ef=params.ann_ef,
            ann_m=params.ann_m,
            ann_parallel=options.ann_parallel,
            rand_state=params.rand_state,
            batch_size=options.batch_size,
            ann_index_fetcher=options.ann_index_fetcher,
            ann_index_saver=options.ann_index_saver,
            update_state=False,
            invalidate_cache=options.invalidate_cache,
        )
        initialization = self._build_embedding_initialization(
            reduction,
            n_centroids=params.n_centroids,
            rand_state=params.rand_state,
            batch_size=options.batch_size,
            invalidate_cache=options.invalidate_cache,
        )
        neighbors_ref = self.query_neighbors(
            ann_index_ref,
            coordinates=coordinates,
            k=params.k,
            batch_size=options.batch_size,
            update_state=False,
            invalidate_cache=options.invalidate_cache,
        )
        connectivity = self.build_connectivity_map(
            neighbors_ref,
            local_connectivity=params.local_connectivity,
            bandwidth=params.bandwidth,
            batch_size=options.batch_size,
            update_state=False,
            invalidate_cache=options.invalidate_cache,
        )
        named_results: dict[str, ArtifactRef] = {}
        if (
            params.harmonize
            and params.reduction_method == "pca"
            and params.feat_scaling
        ):
            named_results["mapping_reference"] = self._build_mapping_reference_artifact(
                reduction=reduction,
                batch_correction=coordinates,
                ann_index=ann_index_ref,
                neighbors=neighbors_ref,
                invalidate_cache=(
                    options.invalidate_cache or options.force_harmony_refit
                ),
            )
        self._publish_current_artifact(
            connectivity,
            update_state=options.update_keys,
            embedding_initialization=initialization,
            named_results=named_results,
        )
        if not options.return_ann_object:
            return None
        ann_stream = self._load_artifact_ann_stream(
            inputs.from_assay,
            inputs.cell_key,
            inputs.feat_key,
            params.feat_scaling if params.reduction_method == "pca" else False,
            neighbors_ref=neighbors_ref,
            fit_kmeans=True,
            n_centroids=params.n_centroids,
        )
        if ann_stream is None:
            raise RuntimeError("Could not load completed graph artifacts")
        return ann_stream

    def _run_resolved_graph_plan(
        self,
        plan: GraphBuildPlan,
    ) -> AnnStream | None:
        params = plan.parameters
        inputs = plan.data_inputs
        options = plan.options
        with self._artifact_execution_context({"local_cache": options.local_cache}):
            normalized = self.run_normalization(
                from_assay=inputs.from_assay,
                cell_key=inputs.cell_key,
                feat_key=inputs.feat_key,
                log_transform=params.log_transform,
                renormalize_subset=params.renormalize_subset,
                batch_size=options.batch_size,
                update_state=False,
                invalidate_cache=options.invalidate_cache,
            )
            with self._cache_normalized_artifact(
                normalized,
                options.local_cache,
                options.batch_size,
            ):
                return self._execute_atomic_graph_plan(
                    plan,
                    normalized,
                )

    def make_graph(
        self,
        from_assay: str | None = None,
        cell_key: str | None = None,
        feat_key: str | None = None,
        pca_cell_key: str | None = None,
        reduction_method: str = "auto",
        dims: int | None = None,
        k: int | None = None,
        ann_metric: str | None = None,
        ann_efc: int | None = None,
        ann_ef: int | None = None,
        ann_m: int | None = None,
        ann_parallel: bool = False,
        rand_state: int | None = None,
        n_centroids: int | None = None,
        batch_size: int | None = None,
        log_transform: bool | None = None,
        renormalize_subset: bool | None = None,
        local_connectivity: float | None = None,
        bandwidth: float | None = None,
        update_keys: bool = True,
        return_ann_object: bool = False,
        custom_loadings: np.ndarray | None = None,
        feat_scaling: bool = True,
        lsi_skip_first: bool = True,
        harmonize: bool = False,
        batch_columns: list[str] | None = None,
        show_elbow_plot: bool = False,
        ann_index_fetcher: Callable | None = None,
        ann_index_saver: Callable | None = None,
        local_cache: bool | str = "auto",
        harmony_params: dict[str, Any] | None = None,
        _force_harmony_refit: bool = False,
    ) -> AnnStream | None:
        """Compatibility facade for the atomic graph operations.

        - Normalizes the data calling the `save_normalized_data` for the assay
        - runs reduction and optional Harmony correction
        - builds the ANN index and embedding initialization
        - queries ANN index for nearest neighbours and saves the distances and indices of the neighbours
        - recalculates distances into graph weights
        - publishes the completed refs through `AssayState`

        Args:
            from_assay: Assay to use for graph creation. If no value is provided then `defaultAssay` will be used
            cell_key: Cells to use for graph creation. By default all cells with True value in 'I' will be used.
                      The provided value for `cell_key` should be a column in cell metadata table with boolean values.
            feat_key: Features to use for graph creation. It is a required parameter. We have chosen not to set this
                      to 'I' by default because this might lead to usage of too many features and may lead to poor
                      results. The value for `feat_key` should be a column in feature metadata from the `from_assay`
                      assay and should be boolean type.
            pca_cell_key: Name of a column from cell metadata table. This column should be boolean type. If no value is
                          provided then the value is set to same as `cell_key` which means all the cells in the
                          normalized data will be used for fitting the pca. This parameter, hence, basically provides a
                          mechanism to subset the normalized data only for PCA fitting step. This parameter can be
                          useful, for example, the data has cells from multiple replicates which wont merge together, in
                          which case the `pca_cell_key` can be used to fit PCA on cells from only one of the replicate.
            reduction_method: Method to use for linear dimension reduction. Could be either 'pca', 'lsi' or 'auto'. In
                              case of 'auto' `_choose_reduction_method` will be used to determine the best reduction
                              type for the assay.
            dims: Number of top reduced dimensions to use (Default value: 11)
            k: Number of nearest neighbours to query for each cell (Default value: 11)
            ann_metric: Refer to HNSWlib link above (Default value: 'l2')
            ann_efc: Refer to HNSWlib link above (Default value: min(100, max(k * 3, 50)))
            ann_ef: Refer to HNSWlib link above (Default value: min(100, max(k * 3, 50)))
            ann_m: Refer to HNSWlib link above (Default value: min(max(48, int(dims * 1.5)), 64) )
            ann_parallel: If True, then ANN graph is created in parallel mode using DataStore.nthreads number of
                          threads. Results obtained in parallel mode will not be reproducible. (Default: False)
            rand_state: Random seed number (Default value: 4466)
            n_centroids: Number of centroids for Kmeans clustering. As a general indication, have a value of 1+ for
                         every 100 cells. Small (<2000 cells) and very small (<500 cells) use a ballpark number for max
                         expected number of clusters (Default value: 500). The results of kmeans clustering are only
                         used to provide initial embedding for UMAP and tSNE. (Default value: 500)
            batch_size: Number of cells in a batch. This number is guided by number of features being used and the
                        amount of available free memory. Though the full data is already divided into chunks, however,
                        if only a fraction of features is being used in the normalized dataset, then the chunk size
                        can be increased to speed up the computation (i.e. PCA fitting and ANN index building).
                        (Default value: 1000)
            log_transform: If True, then the normalized data is log-transformed (only affects RNAassay type assays).
                           (Default value: True)
            renormalize_subset: If True, then the data is normalized using only those features that are True in
                                `feat_key` column rather using total expression of all features in a cell (only affects
                                RNAassay type assays). (Default value: True)
            local_connectivity: This parameter is forwarded to `smooth_knn_dist` function from UMAP package. Higher
                                value will push distribution of edge weights towards terminal values (binary like).
                                Lower values will accumulate edge weights around the mean produced by `bandwidth`
                                parameter. (Default value: 1.0)
            bandwidth: This parameter is forwarded to `smooth_knn_dist` function from UMAP package. Higher value will
                       push the mean of distribution of graph edge weights towards right.  (Default value: 1.5). Read
                       more about `smooth_knn_dist` function here:
                       https://umap-learn.readthedocs.io/en/latest/api.html#umap.umap_.smooth_knn_dist
            update_keys: If True, publish the completed chain through
                         `AssayState`. The name is retained for compatibility.
            return_ann_object: If True then returns the ANNStream object. This allows one to directly interact with the
                               PCA transformer and HNSWlib index. Check out ANNStream documentation to know more.
                               (Default: False)
            custom_loadings: Custom loadings/transformer for linear dimension reduction. If provided, should have a form
                             (d x p) where d is same the number of active features in feat_key and p is the number of
                             reduced dimensions. `dims` parameter is ignored when this is provided.
                             (Default value: None)
            feat_scaling: If True (default) then the feature will be z-scaled otherwise not. It is highly recommended
                          that this is kept as True unless you know what you are doing.
            lsi_skip_first: Whether to remove the first LSI dimension when using ATAC-Seq data.
            harmonize: If True, run Harmony batch correction on the PCA embedding before
                       building the KNN graph. Requires ``batch_columns``.
            batch_columns: Cell metadata columns defining batch variables for Harmony.
            harmony_params: Optional keyword arguments forwarded to ``fit_harmony``.
            show_elbow_plot: If True, then an elbow plot is shown when PCA is fitted to the data. Not shown when using
                            existing PCA loadings or custom loadings. (Default value: False)
            ann_index_fetcher: Optional callable to load a pre-built ANN index instead of fitting one.
            ann_index_saver: Optional callable to persist a fitted ANN index for reuse.
            local_cache: When ``'auto'`` or ``True``, remote stores copy the normalized
                         matrix to a local scratch Zarr before PCA/ANN/kmeans/KNN so
                         multi-pass reads hit local disk instead of object storage.
                         A string value is treated as a persistent scratch base path
                         keyed by artifact ID (~8 GB for 1M cells x 2000 HVGs in
                         float32). ``False`` disables staging.

        Returns:
            Either None or `AnnStream` object
        """
        import warnings

        warnings.warn(
            "make_graph is deprecated; call the atomic graph methods instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        plan = self._resolve_graph_plan(
            from_assay=from_assay,
            cell_key=cell_key,
            feat_key=feat_key,
            pca_cell_key=pca_cell_key,
            reduction_method=reduction_method,
            dims=dims,
            k=k,
            ann_metric=ann_metric,
            ann_efc=ann_efc,
            ann_ef=ann_ef,
            ann_m=ann_m,
            ann_parallel=ann_parallel,
            rand_state=rand_state,
            n_centroids=n_centroids,
            batch_size=batch_size,
            log_transform=log_transform,
            renormalize_subset=renormalize_subset,
            local_connectivity=local_connectivity,
            bandwidth=bandwidth,
            update_keys=update_keys,
            return_ann_object=return_ann_object,
            custom_loadings=custom_loadings,
            feat_scaling=feat_scaling,
            lsi_skip_first=lsi_skip_first,
            harmonize=harmonize,
            batch_columns=batch_columns,
            show_elbow_plot=show_elbow_plot,
            ann_index_fetcher=ann_index_fetcher,
            ann_index_saver=ann_index_saver,
            local_cache=local_cache,
            harmony_params=harmony_params,
            force_harmony_refit=_force_harmony_refit,
            invalidate_cache=False,
        )
        return self._run_resolved_graph_plan(plan)

    def load_graph(
        self,
        from_assay: str | None = None,
        cell_key: str | None = None,
        feat_key: str | None = None,
        symmetric: bool | None = None,
        upper_only: bool | None = None,
        use_k: int | None = None,
        graph_loc: str | None = None,
    ) -> csr_matrix:
        """Load the cell neighbourhood as a scipy sparse matrix.

        Args:
            from_assay: Name of the assay. If None then the default assay is used.
            cell_key: Cell key used to create the graph. If None then the latest feature key used for creating a
                      KNN graph is used.
            feat_key: Feature key used to create the graph. If None then the latest feature key used for creating a
                      KNN graph is used.
            symmetric: If True, makes the graph symmetric by adding it to its transpose.
            upper_only: If True, then only the values from upper triangular of the matrix are returned. This is only
                       used when symmetric is True.
            use_k: Number of top k-nearest neighbours to keep in the graph. This value must be greater than 0 and less
                   the parameter k used. By default, all neighbours are used. (Default value: None)
            graph_loc: Zarr hierarchy where the graph is stored. If no value is provided then graph location is
                       obtained from `_get_latest_graph_loc` method.

        Returns:
            A scipy sparse matrix representing cell neighbourhood graph.
        """

        def symmetrize(g: csr_matrix) -> csr_matrix:
            t = g + g.T
            t = t - g.multiply(g.T)
            return t

        from scipy.sparse import triu

        if graph_loc is None:
            stored = self._lookup_stored_graph(from_assay, cell_key, feat_key)
            if not isinstance(stored, StoredAssayGraph):
                raise TypeError("Expected an assay graph for load_graph lookup")
            graph_loc = stored.paths.cell_graph_group_path
        try:
            explicit_ref = parse_artifact_path(graph_loc)
        except ValueError:
            if not is_integrated_graph_path(
                graph_loc,
                self._integratedGraphsLoc,
            ):
                stored = self._lookup_stored_graph(graph_loc=graph_loc)
                if not isinstance(stored, StoredAssayGraph):
                    raise ValueError("Expected an assay graph")
                if from_assay is not None and from_assay != stored.from_assay:
                    raise ValueError("from_assay does not match the graph location")
                if cell_key is not None and cell_key != stored.cell_key:
                    raise ValueError("cell_key does not match the graph location")
                if feat_key is not None and feat_key != stored.feat_key:
                    raise ValueError("feat_key does not match the graph location")
        else:
            status = inspect_artifact(self.zw, explicit_ref)
            if not status.exists or not status.complete:
                raise RuntimeError(f"Graph artifact is incomplete: {graph_loc}")
            if (
                explicit_ref.scope == "assay"
                and explicit_ref.kind == "connectivity_map"
            ):
                stored = stored_assay_graph_from_ref(self.zw, explicit_ref)
                if from_assay is not None and from_assay != stored.from_assay:
                    raise ValueError("from_assay does not match the graph artifact")
                if cell_key is not None and cell_key != stored.cell_key:
                    raise ValueError("cell_key does not match the graph artifact")
                if feat_key is not None and feat_key != stored.feat_key:
                    raise ValueError("feat_key does not match the graph artifact")
            elif (
                explicit_ref.scope == "datastore"
                and explicit_ref.kind == "integrated_graph"
            ):
                selection = self._graph_cell_selection(explicit_ref)
                source_column = (
                    inspect_artifact(self.zw, selection).execution_options or {}
                ).get("source_column")
                selected_cell_key = cell_key or source_column
                if not isinstance(selected_cell_key, str):
                    raise ValueError(
                        "Integrated graph cell selection key is unavailable"
                    )
                validate_cell_selection_artifact(
                    self.zw,
                    selection,
                    selected_cell_key,
                )
            else:
                raise ValueError(f"Not a graph artifact: {graph_loc}")
        if graph_loc not in self.zw:
            raise ValueError(
                f"{graph_loc} not found in zarr location. "
                f"Run `make_graph` for assay {from_assay}"
            )
        n_cells, graph = self._store_to_sparse(graph_loc, "csr", use_k)
        if symmetric is True:
            graph = symmetrize(graph)
            if upper_only is True:
                graph = triu(graph)
        return graph

    def integrate_assays(
        self,
        assays: list[str],
        label: str,
        method: str = "snn",
        chunk_size: int = 10000,
        invalidate_cache: bool = False,
    ) -> None:
        """Merges KNN graphs of two or more assays from within the same
        DataStore. The input KNN graphs should have been constructed on the
        same set of cells and should each have been constructed with equal
        number of neighbours (parameter: k) The merged KNN graph has the same
        size and shape as the input graphs.

        Args:
            assays: Name of the input assays. The latest constructed graph from each assay is used.
            label: Label for integrated graph
            method: Choose a method for modality integration. Available options: 'snn': Shared nearest neighbour
                    approach and 'wnn': Weighted nearest neighbor approach based on Hao et. alm Cell 2022.
            chunk_size: number of cells to be loaded at a time while reading and writing the graph

        Returns: None
        """
        from ...neighbors.graph import merge_graphs
        from ...neighbors.integration import wnn_integration

        if method not in {"snn", "wnn"}:
            raise ValueError(
                f"Method {method} not supported, choose one of these: 'snn', 'wnn'"
            )
        if method == "wnn" and len(assays) != 2:
            raise ValueError(
                "WNN integration in Scarf can currently be performed using only two assays"
            )
        source_inputs: dict[str, Any] = {}
        legacy_wnn_coordinates: dict[str, np.ndarray] = {}
        shared_selection: ArtifactRef | None = None
        for assay_name in assays:
            if assay_name not in self.assay_names:
                raise ValueError(f"ERROR: Assay {assay_name} was not found.")
            state = read_assay_state(self.zw, assay_name)
            if state is not None and state.connectivity_map is not None:
                if state.normalized is None:
                    raise ValueError(
                        f"Assay {assay_name!r} has no normalized graph input"
                    )
                validate_artifact_graph_selection(
                    self.zw,
                    state.connectivity_map,
                    state.cell_key,
                    state.feat_key,
                )
                selection = self._artifact_input_ref(
                    state.normalized,
                    "cell_selection",
                    "cell_selection",
                )
                if method == "wnn":
                    coordinates_ref = (
                        state.batch_correction
                        if state.batch_correction is not None
                        else state.reduction
                    )
                    if coordinates_ref is None:
                        raise ValueError(
                            f"Assay {assay_name!r} has no selected coordinates"
                        )
                    source_inputs[assay_name] = {
                        "connectivity_map": state.connectivity_map,
                        "coordinates": coordinates_ref,
                    }
                else:
                    source_inputs[assay_name] = state.connectivity_map
            else:
                legacy_graph_path = self.get_latest_graph_loc(
                    assay_name,
                    self._get_latest_cell_key(assay_name),
                    self._get_latest_feat_key(assay_name),
                )
                legacy_graph = as_zarr_group(
                    self.zw[legacy_graph_path],
                    name=legacy_graph_path,
                )
                legacy_input: dict[str, Any] = {
                    "legacy_graph_fingerprint": fingerprint_stored_arrays(
                        legacy_graph,
                        ("edges", "weights"),
                    ),
                }
                selection = self._ensure_cell_selection(
                    self._get_latest_cell_key(assay_name)
                )
                if method == "wnn":
                    cell_key = self._get_latest_cell_key(assay_name)
                    feat_key = self._get_latest_feat_key(assay_name)
                    ann = self._load_ann_stream(
                        assay_name,
                        cell_key,
                        feat_key,
                    )
                    if ann.harmonizedData is not None:
                        coordinates = np.vstack(
                            [
                                np.asarray(block.compute())
                                for block in ann.harmonizedData.blocks
                            ]
                        )
                    else:
                        coordinates = np.vstack(
                            [
                                ann.reducer(block)
                                for block in ann.iter_blocks(
                                    f"Loading {assay_name} coordinates"
                                )
                            ]
                        )
                    legacy_wnn_coordinates[assay_name] = coordinates
                    legacy_input["legacy_coordinates_fingerprint"] = fingerprint_array(
                        coordinates
                    )
                source_inputs[assay_name] = legacy_input
            if shared_selection is None:
                shared_selection = selection
            elif not self._selection_artifacts_match(
                shared_selection,
                selection,
            ):
                raise ValueError("Integrated graphs require one shared cell selection")
        if shared_selection is None:
            raise ValueError("No assay cell selection was resolved")
        source_inputs["cell_selection"] = shared_selection
        integrated_plan = plan_artifact(
            self.zw,
            scope="datastore",
            kind="integrated_graph",
            operation="integrate_assays",
            parameters={"method": method, "assays": assays},
            inputs=source_inputs,
            execution_options={"label": label, "chunk_size": chunk_size},
            invalidate_cache=invalidate_cache,
            required_arrays=(
                ArrayRequirement("edges"),
                ArrayRequirement("weights", dtype_kind="f"),
            ),
        )

        def select_integrated_artifact() -> None:
            index_path = self._integratedGraphsLoc
            index_group = (
                as_zarr_group(self.zw[index_path], name=index_path)
                if index_path in self.zw
                else self.zw.create_group(index_path)
            )
            raw_artifacts = index_group.attrs.get("artifacts", {})
            if "artifacts" in index_group.attrs and not isinstance(
                raw_artifacts,
                dict,
            ):
                raise RuntimeError("Integrated graph artifact index is invalid")
            artifacts = dict(raw_artifacts) if isinstance(raw_artifacts, dict) else {}
            artifacts[label] = integrated_plan.ref.to_dict()
            index_group.attrs["artifacts"] = artifacts

        if integrated_plan.reused:
            select_integrated_artifact()
            return None

        def load_pca_knn(assay_name: str) -> tuple[csr_matrix, NDArray[Any]]:
            state = read_assay_state(self.zw, assay_name)
            if (
                state is not None
                and state.connectivity_map is not None
                and state.reduction is not None
            ):
                graph = self.load_graph(
                    graph_loc=artifact_path(state.connectivity_map),
                    symmetric=False,
                    upper_only=False,
                ).tocsr()
                coordinates_ref = (
                    state.batch_correction
                    if state.batch_correction is not None
                    else state.reduction
                )
                coordinate_status = inspect_artifact(
                    self.zw,
                    coordinates_ref,
                )
                coordinate_parameters = coordinate_status.parameters or {}
                batch_size = int(
                    coordinate_parameters.get("batch_size")
                    or (coordinate_status.execution_options or {}).get("batch_size")
                    or 1000
                )
                coordinate_source, _n_cells, _dims = self._coordinate_source(
                    coordinates_ref,
                    batch_size=batch_size,
                )
                coordinates = np.vstack(
                    list(
                        coordinate_source.iter_coordinate_blocks(
                            f"Loading {assay_name} coordinates",
                        )
                    )
                )
                return graph, coordinates
            cell_key = self._get_latest_cell_key(assay_name)
            feat_key = self._get_latest_feat_key(assay_name)
            graph = self.load_graph(
                from_assay=assay_name,
                cell_key=cell_key,
                feat_key=feat_key,
                symmetric=False,
                upper_only=False,
            ).tocsr()
            return graph, legacy_wnn_coordinates[assay_name]

        if method == "snn":
            graphs: list[csr_matrix] = []
            for assay in assays:
                if assay not in self.assay_names:
                    raise ValueError(f"ERROR: Assay {assay} was not found.")
                graphs.append(
                    self.load_graph(
                        from_assay=assay,
                        cell_key=None,
                        feat_key=None,
                        symmetric=False,
                        upper_only=False,
                    ).tocsr()
                )
            merged_graph = merge_graphs(graphs)
        elif method == "wnn":
            g1, ld1 = load_pca_knn(assays[0])
            g2, ld2 = load_pca_knn(assays[1])
            merged_graph = wnn_integration(
                assays[0], g1, ld1, assays[1], g2, ld2, self.nthreads
            )
        n_cells = merged_graph.shape[0]
        n_neighbors = int(merged_graph.size / n_cells)

        store = start_artifact(self.zw, integrated_plan)
        store.attrs["n_cells"] = n_cells
        store.attrs["n_neighbors"] = n_neighbors

        edge_chunk = chunk_size * n_neighbors
        zge = create_zarr_dataset(
            store,
            "edges",
            (edge_chunk,),
            ("u8", "u8"),
            (n_cells * n_neighbors, 2),
        )
        zgw = create_zarr_dataset(
            store, "weights", (edge_chunk,), "f8", (n_cells * n_neighbors,)
        )

        zge[:, 0] = merged_graph.row
        zge[:, 1] = merged_graph.col
        zgw[:] = merged_graph.data
        finish_artifact(store, integrated_plan)
        select_integrated_artifact()
