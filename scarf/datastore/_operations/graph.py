import hashlib
import json
import os
import shutil
import tempfile
import time
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

from ...storage.types import as_zarr_array, as_zarr_group
from ...assay import Assay, RNAassay
from ...graph.build import (
    CellGraph,
    FeatureMeansAndScales,
    GraphBuildPlan,
    GraphDataInputs,
    GraphExecutionOptions,
    NearestNeighbors,
    NormalizedMatrix,
    ResolvedGraphParameters,
    _GraphBuildOutcome,
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
    make_assay_graph_paths,
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
from ...matrix import ChunkedArray
from ...mapping.reference import MappingReference
from ...neighbors.stream import AnnStream
from ...storage.ann_index import (
    has_ann_index,
    legacy_ann_index_path,
    load_ann_index,
    load_ann_index_from_path,
    save_ann_index,
)
from ...storage.arrays import create_zarr_dataset
from ...storage.copy import (
    copy_zarr_array,
    create_or_open_staged_normed_array,
    open_or_create_staged_normed_array,
)
from ...storage.stores import is_remote_datastore, zarr_root_path
from ...utils.arrays import clean_array
from ...utils.compute import show_dask_progress
from ...utils.logging import logger

if TYPE_CHECKING:
    from ..base_datastore import BaseDataStore as _GraphOperationsBase
else:
    _GraphOperationsBase = object


EMBEDDING_CACHE_MAX_BYTES = 256 * 1024 * 1024


class _GraphBuildProgress:
    """Step-wise progress logging and timing for ``make_graph``."""

    def __init__(self, total: int) -> None:
        self._total = total
        self._step = 0
        self._started = time.perf_counter()
        self._records: list[tuple[int, str, float, str]] = []

    @contextmanager
    def step(self, name: str, *, cached: bool = False) -> Iterator[None]:
        self._step += 1
        mode = "reusing cached" if cached else "computing"
        label = f"{self._step}/{self._total}"
        logger.info(f"make_graph step {label}: {name} ({mode})")
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - t0
            self._records.append((self._step, name, elapsed, mode))
            logger.info(
                f"make_graph step {label}: {name} finished in {elapsed:.1f}s ({mode})"
            )

    def finish(self) -> None:
        total_elapsed = time.perf_counter() - self._started
        accounted = sum(r[2] for r in self._records)
        logger.info(
            f"make_graph finished in {total_elapsed:.1f}s "
            f"({self._step}/{self._total} steps, {accounted:.1f}s in logged steps)"
        )


class _GraphOperationsMixin(_GraphOperationsBase):
    _annStreamPaths: WeakKeyDictionary[AnnStream, str]

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

    @staticmethod
    def _should_cache_ann_embeddings(
        data: ChunkedArray,
        dims: int | None,
        loadings: np.ndarray | None,
        *,
        needed: bool,
        harmonize: bool,
    ) -> bool:
        if not needed or harmonize:
            return False
        if loadings is not None and len(loadings) > 0:
            n_dims = int(loadings.shape[1])
        elif dims is None or dims < 1:
            n_dims = int(data.shape[1])
        else:
            n_dims = min(
                dims,
                int(data.shape[0]),
                max(int(data.chunksize[0]) - 1, 1),
            )
        cache_bytes = int(data.shape[0]) * n_dims * np.dtype(np.float64).itemsize
        return cache_bytes <= EMBEDDING_CACHE_MAX_BYTES

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
    ) -> None:
        from ...mapping.artifact import persist_mapping_reference
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
        reduction_group = as_zarr_group(self.zw[reduction_loc], name=reduction_loc)
        feature_key = f"{cell_key}__{feat_key}" if feat_key != "I" else "I"
        feature_ids = assay.feats.fetch("ids", key=feature_key)
        metadata = self._mapping_reference_metadata(
            assay,
            from_assay,
            cell_key,
            feat_key,
            reduction_loc,
            ann_loc,
            batch_columns,
        )
        metadata["harmonyParameters"] = harmony.parameters
        metadata["batchLevels"] = [list(levels) for levels in harmony.batch_levels]
        knn_group = as_zarr_group(self.zw[knn_loc], name=knn_loc)
        distance_quantiles, distance_values = _distance_quantile_summary(
            as_zarr_array(knn_group["distances"], name="distances")
        )
        persist_mapping_reference(
            reduction_group,
            model,
            feature_ids,
            metadata,
            reference_distance_quantiles=distance_quantiles,
            reference_distance_values=distance_values,
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
        """Build and return a versioned RNA/PCA Symphony-style mapping reference."""
        if self.zarr_mode != "r+":
            raise ValueError("Building a mapping reference requires a read-write store")
        if batch_columns is None:
            raise ValueError("batch_columns is required to build a mapping reference")
        if graph_kwargs.get("feat_scaling", True) is False:
            raise ValueError(
                "Mapping references require feat_scaling=True because query "
                "projection uses the stored reference mean and scale."
            )
        force_harmony_refit = bool(graph_kwargs)
        if not force_harmony_refit:
            try:
                current = self.get_mapping_reference(from_assay, cell_key, feat_key)
            except (KeyError, ValueError):
                force_harmony_refit = True
            else:
                current_columns = [
                    str(column)
                    for column in cast(
                        list[Any], current.metadata.get("batchColumns", [])
                    )
                ]
                force_harmony_refit = (
                    current_columns != batch_columns
                    or not bool(current.metadata.get("complete", False))
                    or "artifactHash" not in current.metadata
                )
        self.make_graph(
            from_assay=from_assay,
            cell_key=cell_key,
            feat_key=feat_key,
            harmonize=True,
            batch_columns=batch_columns,
            _force_harmony_refit=force_harmony_refit,
            **graph_kwargs,
        )
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
        harmony_contract_hash: str | None = None,
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

        normed_loc = make_normalized_group_path(from_assay, cell_key, feat_key)
        if log_transform is None or renormalize_subset is None:
            if normed_loc in self.zw:
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
            if normed_loc in self.zw:
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
        if reduction_loc in self.zw:
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
            if latest_knn_loc is not None:
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
            if reduction_loc in self.zw:
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
            if graph_params_knn_loc in self.zw:
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
            harmony_contract_hash=harmony_contract_hash,
        )

    def _get_latest_keys(
        self,
        from_assay: str | None,
        cell_key: str | None,
        feat_key: str | None,
    ) -> tuple[str, str, str]:
        if from_assay is None:
            from_assay = self._defaultAssay
        if cell_key is None:
            cell_key = self._get_latest_cell_key(from_assay)
        if feat_key is None:
            feat_key = self._get_latest_feat_key(from_assay)
        return from_assay, cell_key, feat_key

    def get_normalized_group_path(
        self, from_assay: str, cell_key: str, feat_key: str
    ) -> str:
        """Return the normalized-group path for assay, cell, and feature keys.

        The path is deterministic and does not require a graph to exist yet.

        Args:
            from_assay: Name of the assay.
            cell_key: Cell key used (or to be used) for the graph.
            feat_key: Feature key used (or to be used) for the graph.

        Returns:
            ``{from_assay}/normed__{cell_key}__{feat_key}``
        """
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

    def _lookup_stored_graph(
        self,
        from_assay: str | None = None,
        cell_key: str | None = None,
        feat_key: str | None = None,
        graph_loc: str | None = None,
    ) -> StoredGraph:
        """Return a stored assay or integrated graph without mutating the store.

        When ``graph_loc`` is omitted, follows ``latest_*`` pointers for the
        assay keys. When ``graph_loc`` points under the integrated-graphs slot,
        returns integrated metadata only. Encoded assay ``graph_loc`` values are
        parsed; callers that pass an explicit location to ``load_graph`` may
        still use that path directly.
        """
        if graph_loc is not None:
            if is_integrated_graph_path(graph_loc, self._integratedGraphsLoc):
                return lookup_stored_integrated_graph(self.zw, graph_loc)
            return parse_assay_graph_paths(graph_loc)

        from_assay, cell_key, feat_key = self._get_latest_keys(
            from_assay, cell_key, feat_key
        )
        return lookup_latest_assay_graph(self.zw, from_assay, cell_key, feat_key)

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

        if ann_index_fetcher is not None:
            try:
                ann_index_fn = ann_index_fetcher(ann_loc)
            except Exception:
                ann_index_fn = None
                logger.warning("Custom `ann_index_fetcher` failed")
            if ann_index_fn is not None and os.path.exists(ann_index_fn):
                return load_ann_index_from_path(ann_index_fn, ann_metric, dim)

        if ann_group is not None and has_ann_index(ann_group):
            return load_ann_index(ann_group, ann_metric, dim)

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
                return
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

    @staticmethod
    def _normed_data_cached(
        assay: Assay,
        cell_key: str,
        feat_key: str,
        location: str,
        log_transform: bool,
        renormalize_subset: bool,
    ) -> bool:
        resolved_feat = feat_key if feat_key == "I" else f"{cell_key}__{feat_key}"
        if location not in assay.z:
            return False
        try:
            cell_idx, feat_idx = assay._get_cell_feat_idx(cell_key, resolved_feat)
            subset_hash = assay._create_subset_hash(cell_idx, feat_idx)
            subset_params = {
                "log_transform": log_transform,
                "renormalize_subset": renormalize_subset,
            }
            grp = assay.z[location]
            return (
                subset_hash == grp.attrs["subset_hash"]
                and subset_params == grp.attrs["subset_params"]
            )
        except (KeyError, ValueError, AttributeError):
            return False

    @staticmethod
    def _staged_normed_cached(
        cache_path: str,
        subset_hash: str,
        subset_params: dict[str, Any],
    ) -> bool:
        if not os.path.isfile(os.path.join(cache_path, "zarr.json")):
            return False
        try:
            root = zarr.open_group(cache_path, mode="r")
            if "data" not in root:
                return False
            staged = root["data"]
            return (
                staged.attrs.get("staged_subset_hash") == subset_hash
                and staged.attrs.get("staged_subset_params") == subset_params
                and bool(staged.attrs.get("staged_complete"))
            )
        except Exception:
            return False

    def _load_ann_stream(
        self,
        from_assay: str,
        cell_key: str,
        feat_key: str,
        feat_scaling: bool = True,
        knn_loc: str | None = None,
    ) -> AnnStream:
        """Load an AnnStream from an existing graph without recomputing KNN."""

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

    def _normed_cache_key(self, subset_hash: str, subset_params: dict[str, Any]) -> str:
        import hashlib
        import json

        payload = json.dumps(
            {"subset_hash": subset_hash, "subset_params": subset_params},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:32]

    def _stage_normed_data(
        self,
        remote_array: zarr.Array,
        subset_hash: str,
        subset_params: dict[str, Any],
        cache_base: str,
    ) -> ChunkedArray:
        """Copy normalized data to a local scratch Zarr for multi-pass graph building."""
        cache_key = self._normed_cache_key(subset_hash, subset_params)
        cache_path = os.path.join(cache_base, cache_key, "normed.zarr")
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        staged = open_or_create_staged_normed_array(cache_path, remote_array)
        if (
            staged.attrs.get("staged_subset_hash") == subset_hash
            and staged.attrs.get("staged_subset_params") == subset_params
            and staged.attrs.get("staged_complete")
        ):
            logger.info(f"Reusing staged normalized data at {cache_path}")
        else:
            logger.info(f"Staging normalized data locally at {cache_path}")
            copy_zarr_array(
                remote_array,
                staged,
                msg="Staging normalized data locally",
            )
            staged.attrs["staged_subset_hash"] = subset_hash
            staged.attrs["staged_subset_params"] = subset_params
            staged.attrs["staged_complete"] = True
        return ChunkedArray(staged, nthreads=self.nthreads)

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
    ) -> GraphBuildPlan:
        if from_assay is None:
            from_assay = self._defaultAssay
        assay = self._get_assay(from_assay)
        if batch_size is None:
            batch_size = assay.rawData.chunksize[0]
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
        batches = None
        harmony_contract_hash: str | None = None
        if harmonize:
            if batch_columns is None:
                raise ValueError("Harmonization requested but no batches provided")
            else:
                if isinstance(batch_columns, list) is False:
                    raise ValueError(
                        "batches must be a list of columns in cell metadata that represent batches"
                    )
                batches = pd.DataFrame(
                    {
                        x: self.cells.fetch(x, key=cell_key).astype(object)
                        for x in batch_columns
                    }
                )
            if batches is None:
                raise ValueError("Harmony requires batch metadata")
            from ...mapping.hashing import array_hash

            harmony_contract = {
                "batchColumns": batch_columns,
                "batchValueHash": array_hash(
                    batches.astype(str).to_numpy().reshape(-1)
                ),
                "parameters": harmony_params or {},
            }
            harmony_contract_hash = hashlib.sha256(
                json.dumps(
                    harmony_contract,
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                ).encode()
            ).hexdigest()[:16]

        parameters = replace(
            parameters,
            harmony_contract_hash=harmony_contract_hash,
        )
        paths = make_assay_graph_paths(
            from_assay=from_assay,
            cell_key=cell_key,
            feat_key=feat_key,
            reduction_method=parameters.reduction_method,
            dims=parameters.dims,
            pca_cell_key=parameters.pca_cell_key,
            ann_metric=parameters.ann_metric,
            ann_efc=parameters.ann_efc,
            ann_ef=parameters.ann_ef,
            ann_m=parameters.ann_m,
            rand_state=parameters.rand_state,
            k=parameters.k,
            local_connectivity=parameters.local_connectivity,
            bandwidth=parameters.bandwidth,
            n_centroids=parameters.n_centroids,
            feat_scaling=parameters.feat_scaling,
            harmony_contract_hash=harmony_contract_hash,
        )
        return GraphBuildPlan(
            data_inputs=GraphDataInputs(
                assay=assay,
                from_assay=from_assay,
                cell_key=cell_key,
                feat_key=feat_key,
                pca_cell_key=parameters.pca_cell_key,
                batches=batches,
                custom_loadings=custom_loadings,
            ),
            parameters=parameters,
            options=GraphExecutionOptions(
                batch_size=batch_size,
                update_keys=update_keys,
                return_ann_object=return_ann_object,
                show_elbow_plot=show_elbow_plot,
                ann_parallel=ann_parallel,
                ann_index_fetcher=ann_index_fetcher,
                ann_index_saver=ann_index_saver,
                local_cache=local_cache,
                force_harmony_refit=force_harmony_refit,
            ),
            paths=paths,
        )

    def _load_or_create_normalized_matrix(
        self,
        plan: GraphBuildPlan,
        progress: _GraphBuildProgress,
        *,
        cache_enabled: bool,
        cache_base: str | None,
    ) -> NormalizedMatrix:
        assay = plan.data_inputs.assay
        cell_key = plan.data_inputs.cell_key
        feat_key = plan.data_inputs.feat_key
        params = plan.parameters
        options = plan.options
        normed_loc = plan.paths.normalized_group_path
        normed_short = normed_loc.split("/")[-1]
        cached_normed = self._normed_data_cached(
            assay,
            cell_key,
            feat_key,
            normed_short,
            params.log_transform,
            params.renormalize_subset,
        )

        # When caching locally and the normalized matrix must be recomputed,
        # mirror each normalized band straight into the local staging cache
        # during the write pass so it never has to be downloaded back from the
        # remote store afterwards.
        staged_mirror = None
        if cache_enabled and not cached_normed and cache_base is not None:
            resolved_feat = feat_key if feat_key == "I" else f"{cell_key}__{feat_key}"
            pre_cell_idx, pre_feat_idx = assay._get_cell_feat_idx(
                cell_key, resolved_feat
            )
            pre_hash = assay._create_subset_hash(pre_cell_idx, pre_feat_idx)
            pre_params = {
                "log_transform": params.log_transform,
                "renormalize_subset": params.renormalize_subset,
            }
            pre_cache_key = self._normed_cache_key(pre_hash, pre_params)
            pre_cache_path = os.path.join(cache_base, pre_cache_key, "normed.zarr")
            os.makedirs(os.path.dirname(pre_cache_path), exist_ok=True)
            staged_mirror = create_or_open_staged_normed_array(
                pre_cache_path, (len(pre_cell_idx), len(pre_feat_idx))
            )

        with progress.step("normalize expression matrix", cached=cached_normed):
            data = assay.save_normalized_data(
                cell_key,
                feat_key,
                options.batch_size,
                normed_short,
                params.log_transform,
                params.renormalize_subset,
                options.update_keys,
                mirror=staged_mirror,
            )
        normed_grp = as_zarr_group(self.zw[normed_loc], name=normed_loc)
        subset_hash = normed_grp.attrs.get("subset_hash")
        subset_params = normed_grp.attrs.get("subset_params")
        if subset_hash is None or subset_params is None:
            raise RuntimeError(
                f"Normalized matrix at {normed_loc} is missing subset metadata; "
                "delete the partial group and retry make_graph"
            )
        subset_hash = cast(str, subset_hash)
        subset_params = cast(dict[str, Any], subset_params)
        if cache_enabled:
            if cache_base is None:
                raise RuntimeError("cache_base must be set when cache_enabled is True")
            cache_key = self._normed_cache_key(subset_hash, subset_params)
            cache_path = os.path.join(cache_base, cache_key, "normed.zarr")
            staged_cached = self._staged_normed_cached(
                cache_path, subset_hash, subset_params
            )
            with progress.step("stage normalized matrix locally", cached=staged_cached):
                normed_grp = as_zarr_group(self.zw[normed_loc], name=normed_loc)
                data = self._stage_normed_data(
                    as_zarr_array(normed_grp["data"], name="data"),
                    subset_hash,
                    subset_params,
                    cache_base,
                )
        return NormalizedMatrix(data=data)

    def _load_or_compute_feature_means_and_scales(
        self,
        normalized: NormalizedMatrix,
        plan: GraphBuildPlan,
        progress: _GraphBuildProgress,
    ) -> FeatureMeansAndScales:
        normed_loc = plan.paths.normalized_group_path
        reduction_method = plan.parameters.reduction_method
        data = normalized.data
        normed_grp = as_zarr_group(self.zw[normed_loc], name=normed_loc)
        cached_stats = (
            reduction_method in ("pca", "manual")
            and "mu" in normed_grp
            and "sigma" in normed_grp
        )
        with progress.step("normalization statistics", cached=cached_stats):
            mu, sigma = self._load_or_compute_norm_stats(
                normed_loc, data, reduction_method
            )
        return FeatureMeansAndScales(mu=mu, sigma=sigma)

    def _load_or_create_ann_stream(
        self,
        normalized: NormalizedMatrix,
        means_and_scales: FeatureMeansAndScales,
        plan: GraphBuildPlan,
        progress: _GraphBuildProgress,
    ) -> AnnStream:
        data = normalized.data
        custom_loadings = plan.data_inputs.custom_loadings
        if custom_loadings is not None and data.shape[1] != custom_loadings.shape[0]:
            raise ValueError(
                f"Provided custom loadings has {custom_loadings.shape[0]} features while the data "
                f"has {data.shape[1]} features."
            )

        params = plan.parameters
        options = plan.options
        paths = plan.paths
        reduction_loc = paths.reduction_group_path
        ann_loc = paths.neighbor_index_group_path
        knn_loc = paths.nearest_neighbors_group_path
        kmeans_loc = cast(str, paths.kmeans_initialization_group_path)
        graph_loc = paths.cell_graph_group_path
        mu = means_and_scales.mu
        sigma = means_and_scales.sigma
        dims = params.dims
        reduction_method = params.reduction_method
        harmonize = params.harmonize
        batch_columns = params.batch_columns
        batches = plan.data_inputs.batches
        feat_scaling = params.feat_scaling
        force_harmony_refit = options.force_harmony_refit
        ann_metric = params.ann_metric
        ann_index_fetcher = options.ann_index_fetcher
        ann_index_saver = options.ann_index_saver

        loadings: NDArray[Any] | None = None
        fit_kmeans = True
        use_for_pca = self.cells.fetch(
            params.pca_cell_key, key=plan.data_inputs.cell_key
        )
        harmonized_data = None
        if reduction_loc in self.zw:
            reduction_grp = as_zarr_group(self.zw[reduction_loc], name=reduction_loc)
            if "reduction" in reduction_grp:
                loadings = np.asarray(
                    as_zarr_array(reduction_grp["reduction"], name="reduction")[:]
                )
                if data.shape[1] != loadings.shape[0]:
                    logger.warning(
                        "Consistency breached in loading pre-cached loadings. Will perform fresh reduction."
                    )
                    loadings = None
                    del self.zw[reduction_loc]
            if (
                harmonize
                and not force_harmony_refit
                and "harmonizedData" in reduction_grp
            ):
                harmonized_arr = as_zarr_array(
                    reduction_grp["harmonizedData"], name="harmonizedData"
                )
                if "batches" in harmonized_arr.attrs:
                    if harmonized_arr.attrs["batches"] == batch_columns:
                        harmonized_data = ChunkedArray(
                            harmonized_arr,
                            nthreads=self.nthreads,
                        )

        if custom_loadings is None:
            if loadings is not None:
                logger.info(
                    f"Using existing loadings for {reduction_method} with {dims} dims"
                )
        else:
            if loadings is not None and np.array_equal(loadings, custom_loadings):
                logger.info("Custom loadings same as used before. Loading from cache")
            else:
                loadings = custom_loadings
                logger.info(
                    f"Using custom loadings with {dims} dims. Will overwrite any "
                    f"previously used custom loadings"
                )
                if reduction_loc in self.zw:
                    del self.zw[reduction_loc]
        if harmonized_data is not None:
            logger.info(f"Using existing harmonized data with {dims} dims")

        ann_idx = None
        had_cached_ann_idx = False
        replace_ann_after_fit = False
        if ann_loc in self.zw:
            ann_grp = as_zarr_group(self.zw[ann_loc], name=ann_loc)
            reset_ann = False
            cached_scaling = ann_grp.attrs.get("featureScaling")
            if cached_scaling is not None and bool(cached_scaling) != feat_scaling:
                reset_ann = True
            elif cached_scaling is None and feat_scaling is False:
                reset_ann = True
            if "isHarmonized" in ann_grp.attrs:
                if cast(bool, ann_grp.attrs["isHarmonized"]):
                    if harmonize is False:
                        reset_ann = True
                    if harmonized_data is None:
                        reset_ann = True
                else:
                    if harmonize:
                        reset_ann = True
            else:
                if harmonize:  # Mostly for backward compatibility
                    reset_ann = True
            if force_harmony_refit:
                reset_ann = True

            if reset_ann:
                replace_ann_after_fit = True
            else:
                temp = dims if dims > 0 else data.shape[1]
                ann_idx = self._resolve_ann_index(
                    ann_loc,
                    ann_metric,
                    temp,
                    ann_index_fetcher=ann_index_fetcher,
                )
                if ann_idx is not None:
                    had_cached_ann_idx = True
                    logger.info("Using existing ANN index")

        if kmeans_loc in self.zw:
            fit_kmeans = False
            logger.info("using existing kmeans cluster centers")
        disable_scaling = True if feat_scaling is False else False

        cached_ann_stream = (
            loadings is not None and had_cached_ann_idx and not fit_kmeans
        )
        need_embeddings = (
            ann_idx is None
            or fit_kmeans
            or knn_loc not in self.zw
            or graph_loc not in self.zw
        )
        cache_embeddings = self._should_cache_ann_embeddings(
            data,
            dims,
            loadings,
            needed=need_embeddings,
            harmonize=harmonize,
        )

        # TODO: expose LSImodel parameters
        with progress.step(
            "dimension reduction, ANN index, and kmeans",
            cached=cached_ann_stream,
        ):
            ann_obj = AnnStream(
                data=data,
                k=params.k,
                n_cluster=params.n_centroids,
                reduction_method=reduction_method,
                dims=dims,
                loadings=loadings,
                use_for_pca=use_for_pca,
                mu=mu,
                sigma=sigma,
                ann_metric=ann_metric,
                ann_efc=params.ann_efc,
                ann_ef=params.ann_ef,
                ann_m=params.ann_m,
                nthreads=self.nthreads,
                ann_parallel=options.ann_parallel,
                rand_state=params.rand_state,
                do_kmeans_fit=fit_kmeans,
                disable_scaling=disable_scaling,
                ann_idx=ann_idx,
                lsi_skip_first=params.lsi_skip_first,
                lsi_params={},
                harmonize=harmonize,
                harmonized_data=harmonized_data,
                batches=batches,
                harmony_params=params.harmony_params,
                cache_embeddings=cache_embeddings,
            )
        self._remember_ann_stream_path(ann_obj, ann_loc)
        if (
            harmonize
            and reduction_method == "pca"
            and ann_obj.featureScaling
            and ann_obj.harmonyResult is not None
        ):
            from ...mapping.symphony import weighted_centroids

            try:
                weighted_centroids(
                    ann_obj.harmonyResult.original.T,
                    ann_obj.harmonyResult.assignments,
                )
                weighted_centroids(
                    ann_obj.harmonyResult.corrected.T,
                    ann_obj.harmonyResult.assignments,
                )
            except ValueError as exc:
                if "empty cluster" not in str(exc):
                    raise
                raise ValueError(
                    "Harmony produced an empty reference cluster. Rebuild with a "
                    "smaller harmony_params['nclust'] value."
                ) from exc

        if replace_ann_after_fit and ann_loc in self.zw:
            del self.zw[ann_loc]
        save_loadings = reduction_loc not in self.zw
        save_harmonized = (
            harmonize and harmonized_data is None and ann_obj.harmonizedData is not None
        )
        save_ann = ann_idx is None
        save_kmeans = fit_kmeans
        cached_persist = not (
            save_loadings or save_harmonized or save_ann or save_kmeans
        )
        with progress.step("persist graph artifacts to Zarr", cached=cached_persist):
            if save_loadings:
                logger.debug(f"Saving loadings to {reduction_loc}")
                self.zw.create_group(reduction_loc, overwrite=True)
                reduction_grp = as_zarr_group(
                    self.zw[reduction_loc], name=reduction_loc
                )
                if ann_obj.loadings is not None:
                    g = create_zarr_dataset(
                        reduction_grp,
                        "reduction",
                        data.chunksize,
                        "f8",
                        ann_obj.loadings.shape,
                    )
                    g[:, :] = ann_obj.loadings
            if save_harmonized:
                reduction_grp = as_zarr_group(
                    self.zw[reduction_loc], name=reduction_loc
                )
                if ann_obj.harmonizedData is not None:
                    harmonized = ann_obj.harmonizedData
                    g = create_zarr_dataset(
                        reduction_grp,
                        "harmonizedData",
                        harmonized.chunksize,
                        "f8",
                        harmonized.shape,
                    )
                    start = 0
                    for block in harmonized.blocks:
                        values = np.asarray(block.compute())
                        stop = start + values.shape[0]
                        g[start:stop, :] = values
                        start = stop
                    g.attrs["batches"] = batch_columns
            if save_ann:
                if ann_loc not in self.zw:
                    logger.debug(f"Saving ANN index to {ann_loc}")
                    self.zw.create_group(ann_loc, overwrite=True)
                self._persist_ann_index(
                    ann_loc,
                    ann_obj.annIdx,
                    ann_index_saver=ann_index_saver,
                )
            if save_kmeans:
                if ann_obj.kmeans is None:
                    raise RuntimeError("kmeans model missing despite fit_kmeans=True")
                logger.debug(f"Saving kmeans clusters to {kmeans_loc}")
                self.zw.create_group(kmeans_loc, overwrite=True)
                kmeans_grp = as_zarr_group(self.zw[kmeans_loc], name=kmeans_loc)
                g = create_zarr_dataset(
                    kmeans_grp,
                    "cluster_centers",
                    (1000, 1000),
                    "f8",
                    ann_obj.kmeans.cluster_centers_.shape,
                )
                g[:, :] = ann_obj.kmeans.cluster_centers_
                g = create_zarr_dataset(
                    kmeans_grp,
                    "cluster_labels",
                    (100000,),
                    "f8",
                    ann_obj.clusterLabels.shape,
                )
                g[:] = ann_obj.clusterLabels
        return ann_obj

    def _load_or_query_nearest_neighbors(
        self,
        ann_stream: AnnStream,
        plan: GraphBuildPlan,
        progress: _GraphBuildProgress,
    ) -> NearestNeighbors:
        knn_loc = plan.paths.nearest_neighbors_group_path
        graph_loc = plan.paths.cell_graph_group_path
        knn_existed = knn_loc in self.zw
        graph_already_complete = knn_existed and graph_loc in self.zw
        recall: str | None = None

        if graph_already_complete:
            with progress.step("KNN neighbor search", cached=True):
                pass
        else:
            from ...neighbors.graph_store import self_query_knn

            if not knn_existed:
                with progress.step("KNN neighbor search", cached=False):
                    recall_val = self_query_knn(
                        ann_obj=ann_stream,
                        store=self.zw.create_group(knn_loc, overwrite=True),
                        chunk_size=plan.options.batch_size,
                        nthreads=self.nthreads,
                    )
                    recall = "%.2f" % recall_val
            else:
                with progress.step("KNN neighbor search", cached=True):
                    pass
        return NearestNeighbors(
            knn_group_path=knn_loc,
            recall=recall,
            graph_already_complete=graph_already_complete,
        )

    def _load_or_build_cell_graph(
        self,
        nearest_neighbors: NearestNeighbors,
        plan: GraphBuildPlan,
        progress: _GraphBuildProgress,
    ) -> CellGraph:
        knn_loc = nearest_neighbors.knn_group_path
        graph_loc = plan.paths.cell_graph_group_path
        params = plan.parameters

        if nearest_neighbors.graph_already_complete:
            with progress.step("smooth KNN distances into graph", cached=True):
                logger.info("KNN graph already exists will not recompute.")
        else:
            from ...neighbors.graph_store import smoothen_dists

            knn_grp = as_zarr_group(self.zw[knn_loc], name=knn_loc)
            with progress.step("smooth KNN distances into graph", cached=False):
                smoothen_dists(
                    self.zw.create_group(graph_loc, overwrite=True),
                    as_zarr_array(knn_grp["indices"], name="indices"),
                    as_zarr_array(knn_grp["distances"], name="distances"),
                    params.local_connectivity,
                    params.bandwidth,
                    plan.options.batch_size,
                )
            if nearest_neighbors.recall is not None:
                logger.info(f"ANN recall: {nearest_neighbors.recall}%")
        return CellGraph(cell_graph_group_path=graph_loc)

    def _finalize_graph_build(
        self,
        plan: GraphBuildPlan,
        ann_stream: AnnStream,
        cell_graph: CellGraph,
        progress: _GraphBuildProgress,
    ) -> _GraphBuildOutcome:
        params = plan.parameters
        options = plan.options
        inputs = plan.data_inputs
        paths = plan.paths
        normed_loc = paths.normalized_group_path
        reduction_loc = paths.reduction_group_path
        ann_loc = paths.neighbor_index_group_path
        knn_loc = paths.nearest_neighbors_group_path
        kmeans_loc = cast(str, paths.kmeans_initialization_group_path)
        graph_loc = cell_graph.cell_graph_group_path
        harmonize = params.harmonize
        feat_scaling = params.feat_scaling
        fresh_batch_correction = harmonize and ann_stream.harmonyResult is not None

        with progress.step("finalize graph metadata", cached=False):
            normed_grp = as_zarr_group(self.zw[normed_loc], name=normed_loc)
            reduction_grp = as_zarr_group(self.zw[reduction_loc], name=reduction_loc)
            ann_grp = as_zarr_group(self.zw[ann_loc], name=ann_loc)
            knn_grp = as_zarr_group(self.zw[knn_loc], name=knn_loc)
            normed_grp.attrs["latest_reduction"] = reduction_loc
            reduction_grp.attrs["latest_ann"] = ann_loc
            reduction_grp.attrs["latest_kmeans"] = kmeans_loc
            ann_grp.attrs["isHarmonized"] = harmonize
            ann_grp.attrs["featureScaling"] = feat_scaling
            ann_grp.attrs["latest_knn"] = knn_loc
            knn_grp.attrs["latest_graph"] = graph_loc
        if fresh_batch_correction and params.reduction_method == "pca":
            if ann_stream.featureScaling:
                self._persist_mapping_reference(
                    inputs.assay,
                    inputs.from_assay,
                    inputs.cell_key,
                    inputs.feat_key,
                    reduction_loc,
                    ann_loc,
                    knn_loc,
                    params.batch_columns or [],
                    ann_stream,
                )
            else:
                logger.warning(
                    "Skipping mapping-reference persistence because "
                    "feat_scaling=False is incompatible with query projection."
                )
        if options.return_ann_object:
            return _GraphBuildOutcome(
                plan=plan,
                ann_stream=ann_stream,
                cell_graph_group_path=graph_loc,
                fresh_batch_correction=fresh_batch_correction,
            )
        if options.show_elbow_plot:
            from ...plotting import elbow

            try:
                var_exp = 100 * ann_stream._pca.explained_variance_ratio_
            except AttributeError:
                logger.warning("PCA was not fitted so not showing an Elbow plot")
            else:
                elbow(variance_explained=var_exp, show=True)
        return _GraphBuildOutcome(
            plan=plan,
            ann_stream=None,
            cell_graph_group_path=graph_loc,
            fresh_batch_correction=fresh_batch_correction,
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
        """Creates a cell neighbourhood graph. Performs following steps in the
        process:

        - Normalizes the data calling the `save_normalized_data` for the assay
        - instantiates ANNStream class which perform dimension reduction, feature scaling (optional) and fits ANN index
        - queries ANN index for nearest neighbours and saves the distances and indices of the neighbours
        - recalculates distances into graph weights
        - saves the indices and distances in sparse graph friendly form
        - fits a MiniBatch kmeans on the data

        The data for all the steps is saved in the Zarr in the following hierarchy which is organized based on data
        dependency. Parameter values for each step are incorporated into group names in the hierarchy::

            RNA
            ├── normed__I__hvgs
            │   ├── data (7648, 2000) float64                 # Normalized data
            │   ├── mu (2000,) float64                    # Means of normalized feature values
            │   ├── sigma (2000,) float64                 # Std dev. of normalized feature values
            │   └── reduction__pca__31__I                     # Dimension reduction group
            │       ├── reduction (2000, 31) float64          # PCA loadings matrix
            │       ├── ann__l2__63__63__48__4466             # ANN group named with ANN parameters
            │       │   └── knn__21                           # KNN group with value of k in name
            │       │       ├── distances (7648, 21) float64  # Raw distance matrix for k neighbours
            │       │       ├── indices (7648, 21) uint64     # Indices for k neighbours
            │       │       └── graph__1.0__1.5               # sparse graph with continuous form distance values
            │       │           ├── edges (160608, 2) uint64
            │       │           └── weights (160608,) float64
            │       └── kmeans__100__4466                     # Kmeans groups
            │           ├── cluster_centers (100, 31) float64 # Centroid matrix
            │           └── cluster_labels (7648,) float64    # Cluster labels for cells
            ...

        The most recent child of each hierarchy node is noted for quick retrieval and in cases where multiple child
        nodes exist. Parameters starting with `ann` are forwarded to HNSWlib. More details about these parameters can
        be found here: https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md

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
            update_keys: If True (default) then `latest_feat_key` zarr attribute of the assay will be updated.
                         Choose False if you are experimenting with a `feat_key` do not want to override existing
                         `latest_feat_key` and by extension `latest_graph`.
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
                         keyed by ``subset_hash`` (~8 GB for 1M cells x 2000 HVGs in
                         float32). ``False`` disables staging.

        Returns:
            Either None or `AnnStream` object
        """
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
        )
        cache_enabled, cache_base, remove_on_success = self._resolve_local_cache_plan(
            self.zarr_loc, self.z, plan.options.local_cache
        )
        planned = 7
        if cache_enabled:
            planned += 1
        progress = _GraphBuildProgress(planned)

        graph_succeeded = False
        try:
            normalized = self._load_or_create_normalized_matrix(
                plan,
                progress,
                cache_enabled=cache_enabled,
                cache_base=cache_base,
            )
            means_and_scales = self._load_or_compute_feature_means_and_scales(
                normalized, plan, progress
            )
            ann_stream = self._load_or_create_ann_stream(
                normalized, means_and_scales, plan, progress
            )
            nearest_neighbors = self._load_or_query_nearest_neighbors(
                ann_stream, plan, progress
            )
            cell_graph = self._load_or_build_cell_graph(
                nearest_neighbors, plan, progress
            )
            outcome = self._finalize_graph_build(plan, ann_stream, cell_graph, progress)
            graph_succeeded = True
            return outcome.ann_stream
        finally:
            progress.finish()
            if cache_enabled and cache_base is not None:
                if remove_on_success:
                    if graph_succeeded and not plan.options.return_ann_object:
                        shutil.rmtree(cache_base, ignore_errors=True)
                    elif not graph_succeeded:
                        logger.warning(
                            f"Graph build failed; local cache scratch retained at {cache_base}"
                        )
                elif not graph_succeeded:
                    logger.warning(
                        f"Graph build failed; local cache retained at {cache_base}"
                    )

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

        from_assay, cell_key, feat_key = self._get_latest_keys(
            from_assay, cell_key, feat_key
        )

        if graph_loc is None:
            stored = self._lookup_stored_graph(from_assay, cell_key, feat_key)
            if not isinstance(stored, StoredAssayGraph):
                raise TypeError("Expected an assay graph for load_graph lookup")
            graph_loc = stored.paths.cell_graph_group_path
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

        def load_pca_knn(assay_name: str) -> tuple[csr_matrix, NDArray[Any]]:
            g = self.load_graph(
                from_assay=assay_name,
                cell_key=None,
                feat_key=None,
                symmetric=False,
                upper_only=False,
            ).tocsr()
            ao = self.make_graph(
                from_assay=assay_name,
                feat_key=self._get_latest_feat_key(assay_name),
                return_ann_object=True,
                update_keys=False,
            )
            if ao is None:
                raise RuntimeError(
                    f"make_graph did not return AnnStream for {assay_name}"
                )
            if ao.loadings is None:
                logger.warning(
                    f"No dimension reduction was used for {assay_name} data. "
                    f"Memory consumption may be high."
                )
                return g, ao.data.compute()
            return g, ao.data.dot(ao.loadings).compute()

        if method == "snn":
            merged_graph = []
            for assay in assays:
                if assay not in self.assay_names:
                    raise ValueError(f"ERROR: Assay {assay} was not found.")
                merged_graph.append(
                    self.load_graph(
                        from_assay=assay,
                        cell_key=None,
                        feat_key=None,
                        symmetric=False,
                        upper_only=False,
                    ).tocsr()
                )
            merged_graph = merge_graphs(merged_graph)
        elif method == "wnn":
            if len(assays) != 2:
                raise ValueError(
                    "WNN integration in Scarf can currently be performed using only two assays"
                )
            g1, ld1 = load_pca_knn(assays[0])
            g2, ld2 = load_pca_knn(assays[1])
            merged_graph = wnn_integration(
                assays[0], g1, ld1, assays[1], g2, ld2, self.nthreads
            )
        else:
            raise ValueError(
                f"Method {method} not supported, choose one of these: 'snn', 'wnn'"
            )

        n_cells = merged_graph.shape[0]
        n_neighbors = int(merged_graph.size / n_cells)

        ig_loc = self._integratedGraphsLoc
        if ig_loc not in self.zw:
            self.zw.create_group(ig_loc)
        ig_grp = as_zarr_group(self.zw[ig_loc], name=ig_loc)
        integrated_graph_loc = make_integrated_graph_path(ig_loc, label)
        if label in ig_grp:
            del self.zw[integrated_graph_loc]
        store = self.zw.create_group(integrated_graph_loc)
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
