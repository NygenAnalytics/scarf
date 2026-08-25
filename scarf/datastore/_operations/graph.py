import math
import os
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import replace
from threading import RLock
from typing import TYPE_CHECKING, Any, Literal, cast
from weakref import WeakKeyDictionary

import numpy as np
import pandas as pd
import zarr
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, csr_matrix

from ...embeddings.reduction import _streaming_lsi_accumulator_bytes
from ...storage.types import as_zarr_array, as_zarr_group
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
    _positive_integer,
)
from ...graph.distances import validate_distance_provenance
from ...graph.feature_projection import (
    graph_cell_selection,
    resolve_native_graph_inputs,
)
from ...graph.state import (
    AssayState,
    named_result_mismatch,
    read_assay_state,
    read_assay_state_document,
    validate_cell_selection_artifact,
    validate_imported_coordinates_artifact,
    validate_normalized_artifact_selection,
    validate_assay_state,
    write_assay_state,
)
from ...matrix import ChunkedArray
from ...metadata.artifacts import (
    categorical_display,
    column_display,
    continuous_display,
    link_cell_data_column,
    link_feature_data_column,
)
from ...neighbors.stages import (
    AnnIndexStage,
    BatchCorrectionStage,
    ChunkedCoordinateStream,
    CoordinateSource,
    KMeansInitializationStage,
    LazyTransformStream,
    NeighborQueryStage,
    ReductionTransform,
)
from ...neighbors.stream import AnnStream
from ...storage.ann_index import (
    has_ann_index,
    load_ann_index,
    save_ann_index,
)
from ...storage.arrays import create_numeric_array, create_zarr_dataset
from ...storage.artifact_writer import (
    ArrayRequirement,
    PlannedArtifact,
    finish_artifact,
    plan_artifact,
    reused_artifact_group,
    start_artifact,
)
from ...storage.artifacts import (
    ArtifactRef,
    ArtifactScope,
    artifact_group,
    artifact_path,
    fingerprint_strings,
    group_at,
    inspect_artifact,
    require_complete_artifact,
    serialize_artifact_value,
)
from ...storage.errors import ArtifactResolutionError
from ...storage.copy import (
    copy_zarr_array,
    create_or_open_staged_normed_array,
)
from ...storage.budget import ResourceBudget
from ...storage.geometry import array_geometry
from ...storage.layout import (
    _group_zarr_format,
    array_shard_rows,
    iter_shard_row_slices,
    row_sharded_array_spec,
)
from ...storage.profiles import resolve_storage_profile
from ...storage.sharding import write_dense_from_row_batches
from ...storage.stores import is_remote_datastore
from ...storage.selections import (
    resolve_metadata_snapshot,
    resolve_selection_artifact,
)
from ...utils.arrays import clean_array
from ...utils.compute import compute_with_progress
from ...utils.logging import logger

if TYPE_CHECKING:
    from ..base_datastore import BaseDataStore as _GraphOperationsBase
else:
    _GraphOperationsBase = object


def _row_block(
    array: zarr.Array,
    requested: int | None,
    *,
    minimum: int | None = None,
) -> int:
    """Resolve a row block, defaulting to the array's own on-disk row band.

    An explicit value that is not a whole number of row bands makes every
    block straddle a band boundary, so each band is fetched and decoded more
    than once.
    """
    band = max(1, array_shard_rows(array))
    n_rows = max(1, int(array.shape[0]))
    if requested is None:
        resolved = min(band, n_rows)
    else:
        resolved = min(_positive_integer(requested, "batch_size"), n_rows)
    if minimum is not None and resolved < minimum:
        aligned = min(n_rows, ((minimum + band - 1) // band) * band)
        if requested is not None:
            logger.warning(
                f"batch_size {resolved} is below the required minimum of "
                f"{minimum}; using the aligned batch_size {aligned}."
            )
        resolved = aligned
    if resolved % band and resolved < n_rows:
        logger.warning(
            f"batch_size {resolved} is not a multiple of the {band}-row band "
            f"of {array.name}; blocks will straddle band boundaries and reread "
            "them. Leave batch_size unset to follow the stored layout."
        )
    return resolved


def _integration_payload_error(
    ref: ArtifactRef,
    message: str,
    *,
    payload: str | None = None,
) -> ArtifactResolutionError:
    context: dict[str, Any] = {
        "assay": ref.assay,
        "artifact_id": ref.artifact_id,
        "actual_kind": ref.kind,
    }
    if payload is not None:
        context["payload"] = payload
    return ArtifactResolutionError(
        message,
        code="corrupt_payload",
        context=context,
    )


def _integration_array_block_rows(array: zarr.Array) -> int:
    chunks = array.chunks
    if chunks and chunks[0]:
        return max(1, int(chunks[0]))
    return max(1, min(int(array.shape[0]), 100_000))


def _integration_payload_dimensions(
    group: zarr.Group,
    ref: ArtifactRef,
) -> tuple[int, int]:
    raw_cells = group.attrs.get("n_cells")
    raw_neighbors = group.attrs.get("n_neighbors")
    if (
        isinstance(raw_cells, bool)
        or not isinstance(raw_cells, int | np.integer)
        or int(raw_cells) < 1
        or isinstance(raw_neighbors, bool)
        or not isinstance(raw_neighbors, int | np.integer)
        or int(raw_neighbors) < 1
    ):
        raise _integration_payload_error(
            ref,
            f"{ref.kind} artifact has invalid n_cells or n_neighbors metadata",
        )
    n_cells = int(raw_cells)
    n_neighbors = int(raw_neighbors)
    if n_neighbors >= n_cells:
        raise _integration_payload_error(
            ref,
            f"{ref.kind} artifact has an invalid neighbor count",
        )
    return n_cells, n_neighbors


def _validate_integration_connectivity_payload(
    root: zarr.Group,
    ref: ArtifactRef,
) -> int:
    try:
        group = artifact_group(root, ref)
        edges = as_zarr_array(group["edges"], name="edges")
        weights = as_zarr_array(group["weights"], name="weights")
    except Exception as error:
        raise _integration_payload_error(
            ref,
            "Connectivity-map artifact payload is unreadable",
        ) from error
    n_cells, n_neighbors = _integration_payload_dimensions(group, ref)
    expected_edges = n_cells * n_neighbors
    if (
        edges.ndim != 2
        or tuple(map(int, edges.shape)) != (expected_edges, 2)
        or np.dtype(edges.dtype) != np.dtype(np.uint32)
        or weights.ndim != 1
        or tuple(map(int, weights.shape)) != (expected_edges,)
        or np.dtype(weights.dtype) != np.dtype(np.float32)
    ):
        raise _integration_payload_error(
            ref,
            "Connectivity-map arrays do not match their stored dimensions",
        )

    row_counts = np.zeros(n_cells, dtype=np.uint64)
    block_rows = _integration_array_block_rows(edges)
    for start in range(0, expected_edges, block_rows):
        stop = min(start + block_rows, expected_edges)
        try:
            edge_block = np.asarray(edges[start:stop])
            weight_block = np.asarray(weights[start:stop])
        except Exception as error:
            raise _integration_payload_error(
                ref,
                "Connectivity-map arrays are unreadable",
            ) from error
        if (
            np.any(edge_block >= n_cells)
            or not np.all(np.isfinite(weight_block))
            or np.any(weight_block < 0)
        ):
            raise _integration_payload_error(
                ref,
                "Connectivity-map arrays contain invalid edge or weight values",
            )
        row_counts += np.bincount(
            edge_block[:, 0],
            minlength=n_cells,
        ).astype(np.uint64, copy=False)
    if np.any(row_counts != n_neighbors):
        raise _integration_payload_error(
            ref,
            "Connectivity-map rows do not match n_neighbors",
        )
    return n_cells


def _validate_integration_neighbors_payload(
    root: zarr.Group,
    ref: ArtifactRef,
) -> int:
    try:
        group = artifact_group(root, ref)
        indices = as_zarr_array(group["indices"], name="indices")
        distances = as_zarr_array(group["distances"], name="distances")
    except Exception as error:
        raise _integration_payload_error(
            ref,
            "Neighbors artifact payload is unreadable",
        ) from error
    n_cells, n_neighbors = _integration_payload_dimensions(group, ref)
    expected_shape = (n_cells, n_neighbors)
    raw_self_hit_rate = group.attrs.get("self_hit_rate")
    if (
        isinstance(raw_self_hit_rate, bool)
        or not isinstance(raw_self_hit_rate, int | float | np.integer | np.floating)
        or not math.isfinite(float(raw_self_hit_rate))
        or not 0 <= float(raw_self_hit_rate) <= 100
        or indices.ndim != 2
        or tuple(map(int, indices.shape)) != expected_shape
        or np.dtype(indices.dtype) != np.dtype(np.uint32)
        or distances.ndim != 2
        or tuple(map(int, distances.shape)) != expected_shape
        or np.dtype(distances.dtype) != np.dtype(np.float32)
    ):
        raise _integration_payload_error(
            ref,
            "Neighbors arrays or metadata do not match their stored dimensions",
        )

    block_rows = _integration_array_block_rows(indices)
    for start in range(0, n_cells, block_rows):
        stop = min(start + block_rows, n_cells)
        try:
            index_block = np.asarray(indices[start:stop])
            distance_block = np.asarray(distances[start:stop])
        except Exception as error:
            raise _integration_payload_error(
                ref,
                "Neighbors arrays are unreadable",
            ) from error
        row_ids = np.arange(start, stop, dtype=np.uint32)[:, None]
        if (
            np.any(index_block >= n_cells)
            or np.any(index_block == row_ids)
            or not np.all(np.isfinite(distance_block))
            or np.any(distance_block < 0)
        ):
            raise _integration_payload_error(
                ref,
                "Neighbors arrays contain invalid indices or distances",
            )
    return n_cells


def _validate_integration_source_payload(
    root: zarr.Group,
    ref: ArtifactRef,
) -> int:
    if ref.kind == "connectivity_map":
        return _validate_integration_connectivity_payload(root, ref)
    if ref.kind == "neighbors":
        return _validate_integration_neighbors_payload(root, ref)
    raise _integration_payload_error(
        ref,
        "Integration source has an unsupported artifact kind",
    )


def _streaming_lsi_block_rows(
    array: zarr.Array,
    resources: ResourceBudget,
    *,
    n_components: int,
    n_oversamples: int,
) -> int:
    n_rows, n_features = map(int, array.shape)
    width = min(n_rows, n_features, n_components + n_oversamples)
    accumulator_bytes = _streaming_lsi_accumulator_bytes(n_features, width)
    geometry = array_geometry(array)
    decode_bytes = 0 if geometry is None else geometry.nominalChunkBytes()
    available = resources.memoryBytes - accumulator_bytes - decode_bytes
    input_itemsize = max(int(np.dtype(array.dtype).itemsize), 1)
    row_bytes = (
        n_features * (input_itemsize + np.dtype(np.float64).itemsize)
        + width * np.dtype(np.float64).itemsize
    )
    if available < row_bytes:
        required = accumulator_bytes + decode_bytes + row_bytes
        raise MemoryError(
            f"Streaming LSI needs about {required} bytes for one row block, "
            f"but the operation limit is {resources.memoryBytes} bytes"
        )
    return max(1, min(n_rows, available // row_bytes))


def _sampling_fraction(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a number")
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        raise TypeError(f"{name} must be a number") from None
    if not math.isfinite(resolved) or not 0 < resolved <= 1:
        raise ValueError(f"{name} must be greater than 0 and at most 1")
    return resolved


class _GraphOperationsMixin(_GraphOperationsBase):
    _annStreamPaths: WeakKeyDictionary[AnnStream, str]
    _annStreamNeighborPaths: WeakKeyDictionary[AnnStream, str]
    _normalizedArtifactCache: dict[ArtifactRef, ChunkedArray]
    _artifactExecutionContext: dict[str, Any]
    _graphMemoryCache: dict[tuple[str, bool, bool, int | None], csr_matrix] | None
    _graphMemoryCacheLock: Any

    @contextmanager
    def _graph_memory_cache_scope(self) -> Iterator[None]:
        """Bound graph reuse to one product pipeline section."""
        existing = getattr(self, "_graphMemoryCache", None)
        if existing is not None:
            yield
            return

        cache: dict[tuple[str, bool, bool, int | None], csr_matrix] = {}
        lock = getattr(self, "_graphMemoryCacheLock", None)
        if lock is None:
            lock = RLock()
            self._graphMemoryCacheLock = lock
        self._graphMemoryCache = cache
        try:
            yield
        finally:
            with lock:
                cache.clear()
                if self._graphMemoryCache is cache:
                    self._graphMemoryCache = None

    if TYPE_CHECKING:

        def _ensure_all_features(self, assay: Any) -> ArtifactRef: ...

        def resolve_features(
            self,
            assay: str,
            features: ArtifactRef | str,
        ) -> ArtifactRef: ...

        def _build_mapping_reference_artifact(
            self,
            *,
            reduction: ArtifactRef,
            batch_correction: ArtifactRef,
            ann_index: ArtifactRef,
            neighbors: ArtifactRef,
            invalidate_cache: bool,
        ) -> ArtifactRef: ...

    def _remember_ann_stream_path(self, ann_obj: AnnStream, path: str) -> None:
        try:
            paths = self._annStreamPaths
        except AttributeError:
            paths = WeakKeyDictionary()
            self._annStreamPaths = paths
        paths[ann_obj] = path

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

    def _resolve_ann_index(
        self,
        ann_ref: ArtifactRef,
        ann_metric: str,
        dim: int,
        expected_count: int | None = None,
    ) -> Any:
        """Load the persisted Zarr bytes for a complete ANN artifact."""
        context = {
            "assay": ann_ref.assay,
            "artifact_id": ann_ref.artifact_id,
            "actual_kind": ann_ref.kind,
        }
        if ann_ref.kind != "ann_index":
            raise ArtifactResolutionError(
                "Expected an ann_index artifact",
                code="wrong_kind",
                context={**context, "expected_kind": "ann_index"},
            )
        if ann_ref.scope != "assay":
            raise ArtifactResolutionError(
                "ANN index artifact must be assay-scoped",
                code="wrong_scope",
                context={**context, "expected_scope": "assay"},
            )
        try:
            status = inspect_artifact(self.zw, ann_ref)
        except (KeyError, TypeError, ValueError) as error:
            raise ArtifactResolutionError(
                "ANN artifact record is malformed",
                code="corrupt_payload",
                context=context,
            ) from error
        if not status.exists:
            raise ArtifactResolutionError(
                "ANN artifact does not exist",
                code="missing_artifact",
                context=context,
            )
        if not status.complete:
            raise ArtifactResolutionError(
                "ANN artifact is incomplete",
                code="incomplete_artifact",
                context=context,
            )
        try:
            ann_group = as_zarr_group(self.zw[status.path], name=status.path)
        except (KeyError, TypeError, ValueError) as error:
            raise ArtifactResolutionError(
                "ANN artifact payload is malformed",
                code="corrupt_payload",
                context=context,
            ) from error
        if not has_ann_index(ann_group):
            raise ArtifactResolutionError(
                "ANN artifact has no persisted Zarr index bytes",
                code="corrupt_payload",
                context=context,
            )
        try:
            return load_ann_index(
                ann_group,
                ann_metric,
                dim,
                expected_count=expected_count,
            )
        except (
            FileNotFoundError,
            KeyError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as error:
            raise ArtifactResolutionError(
                "ANN artifact has unreadable Zarr index bytes",
                code="corrupt_payload",
                context=context,
            ) from error

    def _persist_ann_index(
        self,
        ann_loc: str,
        ann_idx: Any,
        *,
        ann_metric: str,
        dimensions: int,
        element_count: int,
    ) -> None:
        """Save an hnswlib index into the Zarr hierarchy."""
        if self.zarr_mode != "r+":
            logger.debug("Skipping ANN index persistence on read-only store")
            return
        if ann_loc not in self.zw:
            self.zw.create_group(ann_loc, overwrite=True)
        save_ann_index(
            as_zarr_group(self.zw[ann_loc], name=ann_loc),
            ann_idx,
            profile=self.storageProfile,
            metric=ann_metric,
            dimensions=dimensions,
            element_count=element_count,
        )

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
                msg="Calculating normalization statistics",
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
                compute_with_progress(
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
                compute_with_progress(
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
        feat_scaling: bool,
        neighbors_ref: ArtifactRef,
    ) -> AnnStream:
        if neighbors_ref.scope != "assay" or neighbors_ref.assay != from_assay:
            raise ValueError("neighbors does not belong to from_assay")
        lineage = resolve_native_graph_inputs(self.zw, neighbors_ref)
        if lineage.normalized is None or lineage.reduction is None:
            raise ValueError(
                "Graph silhouette requires neighbors built from normalized data"
            )
        correction_ref = None
        ann_ref = lineage.ann_index
        reduction_ref = lineage.reduction
        normalized_ref = lineage.normalized
        coordinates_ref = lineage.coordinates
        if coordinates_ref.kind == "batch_correction":
            correction_ref = coordinates_ref
        reduction_status = inspect_artifact(self.zw, reduction_ref)
        raw_scaling = (reduction_status.inputs or {}).get("feature_scaling")
        if not isinstance(raw_scaling, dict):
            raise ValueError("reduction has no 'feature_scaling' artifact input")
        scaling_ref = ArtifactRef.from_dict(raw_scaling)
        normalized_group = artifact_group(self.zw, normalized_ref)
        scaling_group = artifact_group(self.zw, scaling_ref)
        reduction_group = artifact_group(self.zw, reduction_ref)
        neighbors_group = artifact_group(self.zw, neighbors_ref)
        data = ChunkedArray(
            as_zarr_array(normalized_group["data"], name="data"),
            nthreads=self.nthreads,
            resources=self.resources,
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
            correction_group = artifact_group(self.zw, correction_ref)
            corrected = ChunkedArray(
                as_zarr_array(correction_group["data"], name="data"),
                nthreads=self.nthreads,
                resources=self.resources,
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
            ann_ref,
            ann_metric,
            dims if dims > 0 else data.shape[1],
            expected_count=int(data.shape[0]),
        )
        neighbor_indices = as_zarr_array(
            neighbors_group["indices"],
            name="indices",
        )
        k_value = neighbor_params.get("k")
        ann_obj = AnnStream(
            data=data,
            k=int(neighbor_indices.shape[1] if k_value is None else k_value),
            n_cluster=2,
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
            do_kmeans_fit=False,
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
        self._remember_ann_stream_neighbors(
            ann_obj,
            artifact_path(neighbors_ref),
        )
        return ann_obj

    def _get_graph_ncells_k(self, graph_loc: str) -> tuple[int, int]:
        """

        Args:
            graph_loc:

        Returns:

        """
        graph_group = as_zarr_group(self.zw[graph_loc], name=graph_loc)
        if "n_cells" not in graph_group.attrs or "n_neighbors" not in graph_group.attrs:
            raise ValueError("Graph artifact is missing n_cells or n_neighbors")
        return (
            int(cast(int | float | str, graph_group.attrs["n_cells"])),
            int(cast(int | float | str, graph_group.attrs["n_neighbors"])),
        )

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
        if use_k is None:
            use_k = k
        if use_k > k:
            use_k = k
        if use_k < 1:
            use_k = 1
        w = np.asarray(as_zarr_array(store["weights"], name="weights")[:])
        e = np.asarray(as_zarr_array(store["edges"], name="edges")[:])
        if use_k != k:
            from ...neighbors.graph import take_nearest_per_row

            w, e = take_nearest_per_row(w, e, n_cells, use_k)
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
        return require_complete_artifact(self.zw, ref)

    def _artifact_input_ref(
        self,
        ref: ArtifactRef,
        name: str,
        kind: str,
        *,
        require_input_complete: bool = True,
    ) -> ArtifactRef:
        status = self._require_complete_artifact(ref, ref.kind)
        raw_ref = (status.inputs or {}).get(name)
        if not isinstance(raw_ref, dict):
            raise ValueError(f"{ref.kind} artifact has no {name!r} input")
        input_ref = ArtifactRef.from_dict(raw_ref)
        if require_input_complete:
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
                        group = group_at(self.zw, status.path)
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
            if getattr(self, "zarr_mode", "r+") != "r+":
                raise PermissionError(
                    f"Selection provenance for {column!r} is unavailable "
                    "in the read-only store"
                )
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
        if getattr(self, "zarr_mode", "r+") != "r+":
            return ref
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
            if current.kind not in {
                "reduction",
                "batch_correction",
                "imported_coordinates",
            }:
                raise ValueError(
                    "Neighbor coordinates must be reduction, batch_correction, "
                    "or imported_coordinates"
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
        if current.kind == "imported_coordinates":
            imported_status = self._require_complete_artifact(
                current,
                "imported_coordinates",
            )
            if current.assay is None:
                raise ValueError("Imported-coordinate artifact has no assay")
            execution = imported_status.execution_options or {}
            cell_key = execution.get("cell_key")
            if not isinstance(cell_key, str):
                raise ValueError("Imported-coordinate artifact has no cell_key")
            if cell_key_override is not None:
                cell_key = cell_key_override
            validate_imported_coordinates_artifact(
                self.zw,
                current,
                cell_key=cell_key,
            )
            imported_state = AssayState(
                assay=current.assay,
                cell_key=cell_key,
                ann_index=ann_index,
                neighbors=neighbors,
                connectivity_map=connectivity_map,
                named_results=named_results or {},
            )
            previous = read_assay_state_document(self.zw, current.assay)
            if named_results is None and previous is not None:
                carried = {}
                for name, ref in previous.named_results.items():
                    try:
                        fits = (
                            named_result_mismatch(
                                self.zw,
                                name,
                                ref,
                                imported_state,
                            )
                            is None
                        )
                    except (KeyError, RuntimeError, TypeError, ValueError):
                        continue
                    if fits:
                        carried[name] = ref
                if carried:
                    imported_state = replace(
                        imported_state,
                        named_results=carried,
                    )
            return imported_state
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
            reduction_status = self._require_complete_artifact(
                current,
                "reduction",
            )
            raw_scaling = (reduction_status.inputs or {}).get("feature_scaling")
            if raw_scaling is not None:
                if not isinstance(raw_scaling, dict):
                    raise ValueError(
                        "Reduction feature_scaling input is not an artifact ref"
                    )
                feature_scaling = ArtifactRef.from_dict(raw_scaling)
                self._require_complete_artifact(
                    feature_scaling,
                    "feature_scaling",
                    assay=current.assay,
                )
        elif current.kind == "normalized":
            normalized = current
        else:
            raise ValueError(f"Cannot select graph state from {terminal.kind!r}")
        assert normalized is not None
        normalized_status = self._require_complete_artifact(
            normalized,
            "normalized",
        )
        execution = normalized_status.execution_options or {}
        cell_key = execution.get("cell_key")
        if not isinstance(cell_key, str):
            raise ValueError("Normalized artifact is missing cell_key")
        if normalized.assay is None:
            raise ValueError("Normalized artifact has no assay")
        previous = read_assay_state_document(self.zw, normalized.assay)
        if cell_key_override is not None:
            cell_key = cell_key_override
        elif previous is not None and previous.normalized == normalized:
            cell_key = previous.cell_key
        validate_normalized_artifact_selection(
            self.zw,
            normalized,
            cell_key,
        )
        if embedding_initialization is None and reduction is not None:
            if previous is not None and previous.embedding_initialization is not None:
                try:
                    previous_reduction = self._artifact_input_ref(
                        previous.embedding_initialization,
                        "reduction",
                        "reduction",
                    )
                except (KeyError, RuntimeError, TypeError, ValueError):
                    previous_reduction = None
                if previous_reduction == reduction:
                    embedding_initialization = previous.embedding_initialization
        if connectivity_map is None and neighbors is not None:
            if previous is not None and previous.connectivity_map is not None:
                try:
                    previous_neighbors = self._artifact_input_ref(
                        previous.connectivity_map,
                        "neighbors",
                        "neighbors",
                    )
                except (KeyError, RuntimeError, ValueError):
                    # Publishing a new chain is how a store recovers from a graph
                    # artifact that was removed, so a stale ref cannot block it.
                    previous_neighbors = None
                if previous_neighbors == neighbors:
                    connectivity_map = previous.connectivity_map
        state = AssayState(
            assay=normalized.assay,
            cell_key=cell_key,
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
        if named_results is None and previous is not None and previous.named_results:
            carried = {}
            for name, ref in previous.named_results.items():
                try:
                    fits = named_result_mismatch(self.zw, name, ref, state) is None
                except (KeyError, RuntimeError, TypeError, ValueError):
                    continue
                if fits:
                    carried[name] = ref
            if carried:
                state = replace(state, named_results=carried)
        return state

    def _publish_current_artifact(
        self,
        ref: ArtifactRef,
        *,
        update_state: bool,
        embedding_initialization: ArtifactRef | None = None,
        named_results: dict[str, ArtifactRef] | None = None,
        named_result_updates: dict[str, ArtifactRef] | None = None,
        cell_key_override: str | None = None,
    ) -> None:
        if not update_state:
            return
        if named_results is not None and named_result_updates is not None:
            raise ValueError(
                "named_results and named_result_updates cannot both be provided"
            )
        candidate = self._artifact_chain_state(
            ref,
            embedding_initialization=embedding_initialization,
            named_results=named_results,
            cell_key_override=cell_key_override,
        )
        if named_result_updates is not None:
            merged_results = dict(candidate.named_results)
            merged_results.update(named_result_updates)
            candidate = replace(candidate, named_results=merged_results)
        previous = read_assay_state_document(self.zw, candidate.assay)
        field_name = {
            "normalized": "normalized",
            "embedding_initialization": "embedding_initialization",
            "reduction": "reduction",
            "batch_correction": "batch_correction",
            "ann_index": "ann_index",
            "neighbors": "neighbors",
            "connectivity_map": "connectivity_map",
        }.get(ref.kind)
        if (
            previous is not None
            and previous.matches(candidate.cell_key)
            and field_name is not None
            and getattr(previous, field_name) == ref
            and embedding_initialization is None
            and named_results is None
            and named_result_updates is None
        ):
            try:
                validate_assay_state(self.zw, previous)
            except ArtifactResolutionError:
                # An unavailable artifact in the previous current chain must
                # not prevent this complete chain from becoming current.
                pass
            else:
                candidate = previous
        write_assay_state(
            self.zw,
            candidate,
        )

    def _graph_cell_selection(
        self,
        graph_ref: ArtifactRef,
    ) -> ArtifactRef:
        return graph_cell_selection(self.zw, graph_ref)

    def _load_normalized_artifact(
        self,
        ref: ArtifactRef,
        *,
        batch_size: int | None,
    ) -> ChunkedArray:
        try:
            cached = self._normalizedArtifactCache.get(ref)
        except AttributeError:
            cached = None
        if cached is not None:
            return cached
        status = self._require_complete_artifact(ref, "normalized")
        group = group_at(self.zw, status.path)
        backing = as_zarr_array(group["data"], name="data")
        return ChunkedArray(
            backing,
            block_size=_row_block(backing, batch_size),
            nthreads=self.nthreads,
            resources=self.resources,
        )

    @contextmanager
    def _cache_normalized_artifact(
        self,
        ref: ArtifactRef,
        local_cache: bool | str,
        batch_size: int | None,
    ) -> Iterator[None]:
        try:
            already_cached = ref in self._normalizedArtifactCache
        except AttributeError:
            already_cached = False
        if already_cached:
            yield
            return
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
        source_group = group_at(self.zw, status.path)
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
                resources=self.resources,
            )
            staged.attrs["complete"] = True
        try:
            cache = self._normalizedArtifactCache
        except AttributeError:
            cache = {}
            self._normalizedArtifactCache = cache
        cache[ref] = ChunkedArray(
            staged,
            block_size=_row_block(staged, batch_size),
            nthreads=self.nthreads,
            resources=self.resources,
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
        batch_size: int | None,
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
        block_rows = int(normalized.chunksize[0])
        scaling_group = artifact_group(self.zw, scaling_ref)
        mu = np.asarray(as_zarr_array(scaling_group["mean"], name="mean")[:])
        sigma = np.asarray(as_zarr_array(scaling_group["scale"], name="scale")[:])
        reduction_group = group_at(self.zw, status.path)
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
            batch_size=block_rows,
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
            batch_size=block_rows,
        )
        return transform, stream

    def _coordinate_source(
        self,
        coordinates: ArtifactRef,
        *,
        batch_size: int | None,
    ) -> tuple[CoordinateSource, int, int]:
        if coordinates.kind == "imported_coordinates":
            status = self._require_complete_artifact(
                coordinates,
                "imported_coordinates",
            )
            execution = status.execution_options or {}
            cell_key = execution.get("cell_key")
            if not isinstance(cell_key, str) or not cell_key:
                raise ValueError(
                    "Imported-coordinate artifact has no cell selection key"
                )
            validate_imported_coordinates_artifact(
                self.zw,
                coordinates,
                cell_key=cell_key,
            )
            group = group_at(self.zw, status.path)
            backing = as_zarr_array(group["data"], name="data")
            data = ChunkedArray(
                backing,
                block_size=_row_block(backing, batch_size),
                nthreads=self.nthreads,
                resources=self.resources,
            )
            logger.debug(
                f"Using imported coordinates from {status.path} "
                f"(shape={data.shape[0]} x {data.shape[1]})"
            )
            return (
                ChunkedCoordinateStream(data, self.nthreads),
                int(data.shape[0]),
                int(data.shape[1]),
            )
        if coordinates.kind == "batch_correction":
            status = self._require_complete_artifact(
                coordinates,
                "batch_correction",
            )
            group = group_at(self.zw, status.path)
            backing = as_zarr_array(group["data"], name="data")
            data = ChunkedArray(
                backing,
                block_size=_row_block(backing, batch_size),
                nthreads=self.nthreads,
                resources=self.resources,
            )
            logger.debug(
                f"Using materialized batch-correction coordinates from "
                f"{status.path} (shape={data.shape[0]} x {data.shape[1]})"
            )
            return (
                ChunkedCoordinateStream(data, self.nthreads),
                int(data.shape[0]),
                int(data.shape[1]),
            )
        if coordinates.kind == "reduction":
            status = self._require_complete_artifact(
                coordinates,
                "reduction",
            )
            group = group_at(self.zw, status.path)
            method_label = {
                "run_pca": "PCA",
                "run_lsi": "LSI",
                "run_custom_reduction": "custom reduction",
            }.get(
                status.operation or "",
                str((status.parameters or {}).get("reduction_method", "reduction")),
            )
            if "data" in group:
                backing = as_zarr_array(group["data"], name="data")
                data = ChunkedArray(
                    backing,
                    block_size=_row_block(backing, batch_size),
                    nthreads=self.nthreads,
                    resources=self.resources,
                )
                logger.debug(
                    f"Using materialized {method_label} coordinates from "
                    f"{status.path} (shape={data.shape[0]} x {data.shape[1]})"
                )
                return (
                    ChunkedCoordinateStream(data, self.nthreads),
                    int(data.shape[0]),
                    int(data.shape[1]),
                )
            logger.warning(
                f"Materialized {method_label} coordinates missing at "
                f"{status.path} (no 'data' array); falling back to on-the-fly "
                f"projection from normalized data. Downstream ANN, neighbors, "
                f"and embedding init will re-read normalized features instead "
                f"of stored reduced coordinates."
            )
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
        raise ValueError(
            "Coordinates must reference reduction, batch_correction, "
            "or imported_coordinates"
        )

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
        *,
        features: ArtifactRef | str,
        log_transform: bool | None = None,
        renormalize_subset: bool | None = None,
        update_state: bool = True,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Normalize one cell and feature selection into an artifact.

        An identical operation reuses a complete artifact unless
        ``invalidate_cache`` is true. The returned reference can be passed
        directly to a reduction method.

        Args:
            from_assay: Assay to normalize. Uses the default assay when omitted.
            cell_key: Boolean cell metadata column selecting rows.
            features: Exact published feature label or feature-selection ref.
            log_transform: Whether to apply the assay log transform. When
                omitted, reuse the selected artifact setting or default to
                true. ATAC defaults to false and rejects true.
            renormalize_subset: Whether to recompute the normalization
                denominator from selected features. When omitted, reuse the
                selected setting. ATAC defaults to false; other assays default
                to true.
            update_state: Select the result as the assay's current normalized
                artifact.
            invalidate_cache: Force a new artifact instead of reusing an
                identical complete result.

        Returns:
            Reference to the normalized artifact.

        Raises:
            ArtifactResolutionError: If the feature selection is invalid.
            TypeError: If a selection is not boolean.
            ValueError: If the assay or selected data is invalid.
        """
        assay_name = from_assay or self._defaultAssay
        if assay_name is None:
            raise ValueError("No assay was provided and no default is configured")
        assay = self._get_assay(assay_name)
        state = read_assay_state_document(self.zw, assay_name)
        self._ensure_all_features(assay)
        feature_selection = self.resolve_features(assay_name, features)
        cell_values = np.asarray(self.cells.fetch_all(cell_key))
        feature_group = artifact_group(self.zw, feature_selection)
        feature_values = np.asarray(
            as_zarr_array(feature_group["values"], name="values")[:],
            dtype=bool,
        )
        if cell_values.dtype != bool or feature_values.dtype != bool:
            raise TypeError("Cell and feature selections must be boolean")
        n_cells = int(cell_values.sum())
        n_features = int(feature_values.sum())
        if n_cells < 1 or n_features < 1:
            raise ValueError("Normalization requires selected cells and features")
        cell_selection = self._ensure_cell_selection(cell_key)
        stored_parameters: dict[str, Any] = {}
        if state is not None and state.cell_key == cell_key and state.normalized:
            candidate_parameters: dict[str, Any] = {}
            try:
                normalized_status = inspect_artifact(self.zw, state.normalized)
                if not normalized_status.exists or not normalized_status.complete:
                    raise ValueError("Current normalized artifact is unavailable")
                validate_normalized_artifact_selection(
                    self.zw,
                    state.normalized,
                    cell_key,
                )
                normalized_inputs = normalized_status.inputs or {}
                stored_cell_selection = ArtifactRef.from_dict(
                    cast(dict[str, Any], normalized_inputs["cell_selection"])
                )
                stored_feature_selection = ArtifactRef.from_dict(
                    cast(dict[str, Any], normalized_inputs["feature_selection"])
                )
                candidate_parameters = normalized_status.parameters or {}
            except (KeyError, RuntimeError, TypeError, ValueError):
                stored_cell_selection = None
                stored_feature_selection = None
            if (
                stored_cell_selection == cell_selection
                and stored_feature_selection == feature_selection
            ):
                stored_parameters = candidate_parameters
        from ...assay import ATACassay

        if isinstance(assay, ATACassay):
            stored_method_is_current = stored_parameters.get(
                "normalization_method"
            ) == serialize_artifact_value(assay.normMethod)
            if log_transform is None:
                log_transform = (
                    bool(stored_parameters.get("log_transform", False))
                    if stored_method_is_current
                    else False
                )
            elif not isinstance(log_transform, (bool, np.bool_)):
                raise TypeError("log_transform must be a boolean")
            if log_transform:
                raise ValueError(
                    "ATAC TF-IDF does not support log_transform; use False"
                )
            else:
                log_transform = False
            if renormalize_subset is None:
                renormalize_subset = (
                    bool(stored_parameters.get("renormalize_subset", False))
                    if stored_method_is_current
                    else False
                )
            elif not isinstance(renormalize_subset, (bool, np.bool_)):
                raise TypeError("renormalize_subset must be a boolean")
            else:
                renormalize_subset = bool(renormalize_subset)
        else:
            if log_transform is None:
                log_transform = bool(stored_parameters.get("log_transform", True))
            if renormalize_subset is None:
                renormalize_subset = bool(
                    stored_parameters.get("renormalize_subset", True)
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
            update_state=update_state,
            invalidate_cache=invalidate_cache,
        )
        planned = self._plan_assay_artifact(
            assay_name,
            arguments,
            required_arrays=(
                ArrayRequirement(
                    "data",
                    shape=(n_cells, n_features),
                    dtype=np.float32,
                ),
                ArrayRequirement(
                    "feature_sum",
                    shape=(n_features,),
                    dtype=np.float64,
                ),
                ArrayRequirement(
                    "feature_squared_sum",
                    shape=(n_features,),
                    dtype=np.float64,
                ),
            ),
            invalidate_cache=invalidate_cache,
        )
        self._ensure_dataset_fingerprint(assay_name)
        if not planned.reused:
            group = start_artifact(self.zw, planned)
            relative_path = artifact_path(planned.ref).removeprefix(f"{assay_name}/")
            assay.save_normalized_data(
                np.flatnonzero(cell_values),
                np.flatnonzero(feature_values),
                relative_path,
                log_transform=log_transform,
                renormalize_subset=renormalize_subset,
            )
            finish_artifact(group, planned)
        self._publish_current_artifact(
            planned.ref,
            update_state=update_state,
            cell_key_override=cell_key,
        )
        action = "Reused" if planned.reused else "Stored"
        logger.info(
            f"{action} normalized data for {n_cells} cells and {n_features} features"
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
        local_cache: bool | str,
        show_elbow_plot: bool,
        update_state: bool,
        invalidate_cache: bool,
        lsi_solver: Literal["streaming", "materialized"] = "streaming",
        lsi_n_iter: int = 5,
        lsi_n_oversamples: int = 10,
    ) -> ArtifactRef:
        requested_dims = _positive_integer(dims, "dims")
        if batch_size is not None:
            _positive_integer(batch_size, "batch_size")
        normalized_ref = normalized or self._selected_artifact(
            from_assay,
            "normalized",
            "normalized",
        )
        status = self._require_complete_artifact(normalized_ref, "normalized")
        if normalized_ref.assay is None:
            raise ValueError("Normalized artifact has no assay")
        read_assay_state_document(self.zw, normalized_ref.assay)
        group = group_at(self.zw, status.path)
        data = as_zarr_array(group["data"], name="data")
        effective_batch_size = _row_block(
            data,
            batch_size,
            minimum=(requested_dims + 1 if method == "pca" else None),
        )
        if method == "lsi" and lsi_solver == "streaming":
            memory_limited_rows = _streaming_lsi_block_rows(
                data,
                self.resources,
                n_components=requested_dims + int(lsi_skip_first),
                n_oversamples=lsi_n_oversamples,
            )
            if effective_batch_size > memory_limited_rows:
                logger.warning(
                    f"Reducing LSI batch_size from {effective_batch_size} to "
                    f"{memory_limited_rows} rows to honor the memory budget"
                )
                effective_batch_size = memory_limited_rows
        with self._artifact_execution_context({"local_cache": local_cache}):
            with self._cache_normalized_artifact(
                normalized_ref,
                local_cache,
                effective_batch_size,
            ):
                return self._run_reduction_artifact_impl(
                    method=method,
                    normalized=normalized_ref,
                    from_assay=from_assay,
                    dims=requested_dims,
                    pca_cell_key=pca_cell_key,
                    feat_scaling=feat_scaling,
                    lsi_skip_first=lsi_skip_first,
                    custom_loadings=custom_loadings,
                    rand_state=rand_state,
                    batch_size=effective_batch_size,
                    show_elbow_plot=show_elbow_plot,
                    update_state=update_state,
                    invalidate_cache=invalidate_cache,
                    lsi_solver=lsi_solver,
                    lsi_n_iter=lsi_n_iter,
                    lsi_n_oversamples=lsi_n_oversamples,
                )

    def _run_reduction_artifact_impl(
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
        lsi_solver: Literal["streaming", "materialized"] = "streaming",
        lsi_n_iter: int = 5,
        lsi_n_oversamples: int = 10,
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
        if not isinstance(cell_key, str):
            raise ValueError("Normalized artifact has no cell_key")
        state = read_assay_state_document(self.zw, assay_name)
        if state is not None and state.normalized == normalized_ref:
            cell_key = state.cell_key
        validate_normalized_artifact_selection(
            self.zw,
            normalized_ref,
            cell_key,
        )
        data_group = as_zarr_group(
            self.zw[normalized_status.path],
            name=normalized_status.path,
        )
        data_array = as_zarr_array(data_group["data"], name="data")
        n_cells, n_features = map(int, data_array.shape)
        effective_batch_size = min(
            _positive_integer(batch_size, "batch_size"),
            n_cells,
        )
        effective_dims = _positive_integer(dims, "dims")
        if custom_loadings is not None:
            if custom_loadings.ndim != 2:
                raise ValueError("Custom loadings must be a two-dimensional matrix")
            if custom_loadings.shape[0] != n_features:
                raise ValueError("Custom loadings rows must match normalized features")
            effective_dims = int(custom_loadings.shape[1])
            if effective_dims < 1:
                raise ValueError("Custom loadings must contain at least one dimension")
        pca_key = pca_cell_key or cell_key
        pca_selection = None
        pca_use_values: np.ndarray | None = None
        if method == "pca":
            pca_use_values = np.asarray(self.cells.fetch(pca_key, key=cell_key))
            if pca_use_values.dtype != bool or pca_use_values.shape != (n_cells,):
                raise TypeError(
                    "pca_cell_key must select one boolean value per normalized cell"
                )
            selected_pca_cells = int(pca_use_values.sum())
            if selected_pca_cells < effective_dims + 1:
                raise ValueError("PCA requires at least dims + 1 selected cells")
            if n_features < effective_dims + 1:
                raise ValueError("PCA requires at least dims + 1 selected features")
            if effective_batch_size < effective_dims + 1:
                raise ValueError("PCA batch_size must be at least dims + 1")
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
        elif method == "lsi":
            required_rank = effective_dims + int(lsi_skip_first)
            if required_rank > min(n_cells, n_features):
                raise ValueError(
                    "LSI dimensions, including the skipped component, exceed "
                    "the normalized matrix rank"
                )
        enabled_scaling = method == "pca" and feat_scaling
        scaling_arguments = FeatureScalingArguments(
            normalized=normalized_ref,
            enabled=enabled_scaling,
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
                    dtype=np.float64,
                ),
                ArrayRequirement(
                    "scale",
                    shape=(scaling_shape,),
                    dtype=np.float64,
                ),
            ),
            invalidate_cache=invalidate_cache,
        )
        normalized_data = self._load_normalized_artifact(
            normalized_ref,
            batch_size=effective_batch_size,
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
                if "feature_sum" in data_group and "feature_squared_sum" in data_group:
                    total = np.asarray(
                        as_zarr_array(
                            data_group["feature_sum"],
                            name="feature_sum",
                        )[:],
                        dtype=np.float64,
                    )
                    squared_total = np.asarray(
                        as_zarr_array(
                            data_group["feature_squared_sum"],
                            name="feature_squared_sum",
                        )[:],
                        dtype=np.float64,
                    )
                    mu_raw = total / n_cells
                    variance = squared_total / n_cells - np.square(mu_raw)
                    sigma_raw = np.sqrt(np.clip(variance, 0, None))
                else:
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
                solver=lsi_solver,
                n_iter=lsi_n_iter,
                n_oversamples=lsi_n_oversamples,
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
                dims=effective_dims,
                feat_scaling=feat_scaling,
                update_state=update_state,
                invalidate_cache=invalidate_cache,
            )
        required_arrays: tuple[str | ArrayRequirement, ...] = (
            ArrayRequirement(
                "loadings",
                shape=(n_features, effective_dims),
                dtype=np.float64,
            ),
            ArrayRequirement(
                "data",
                shape=(n_cells, effective_dims),
                dtype=np.float32,
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
                pca_use_values if method == "pca" else np.ones(n_cells, dtype=bool)
            )
            assert use_for_pca is not None
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
                lsi_params={
                    "solver": lsi_solver,
                    "n_iter": lsi_n_iter,
                    "n_oversamples": lsi_n_oversamples,
                },
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
                score_spec = row_sharded_array_spec(
                    (n_cells, effective_dims),
                    np.float32,
                    profile=resolve_storage_profile(reduction_group.store),
                    band_rows=min(n_cells, 1_000_000),
                    zarr_format=_group_zarr_format(reduction_group),
                    fill_value=0.0,
                )
                scores = create_numeric_array(
                    reduction_group,
                    "data",
                    score_spec,
                )

                def score_blocks() -> Iterator[np.ndarray]:
                    for block in normalized_data.stream_blocks(
                        nthreads=self.nthreads,
                        msg="Calculating reduced coordinates",
                    ):
                        yield np.asarray(
                            transform.transform(block),
                            dtype=np.float32,
                        )

                write_dense_from_row_batches(
                    scores,
                    score_blocks(),
                    dtype=np.float32,
                    msg="Writing reduced coordinates",
                    resources=self.resources,
                    io=self.storageIo,
                )
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
        action = "Reused" if planned.reused else "Stored"
        logger.info(
            f"{action} {method.upper()} reduction for {n_cells} cells "
            f"with {effective_dims} dimensions"
        )
        return planned.ref

    def run_pca(
        self,
        normalized: ArtifactRef | None = None,
        *,
        from_assay: str | None = None,
        dims: int = 21,
        pca_cell_key: str | None = None,
        feat_scaling: bool = True,
        batch_size: int | None = None,
        local_cache: bool | str = "auto",
        show_elbow_plot: bool = False,
        update_state: bool = True,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Fit or reuse PCA for a normalized artifact.

        Args:
            normalized: Normalized artifact to reduce. Uses the selected assay
                state when omitted.
            from_assay: Assay used to resolve the selected normalized artifact.
            dims: Requested number of principal components. (Default: 21)
            pca_cell_key: Optional boolean cell column used to fit PCA while
                projecting every selected cell.
            feat_scaling: Whether to standardize features before fitting PCA.
            batch_size: Number of selected cells processed per block. When
                omitted, whole stored row bands are combined as needed to fit
                at least ``dims + 1`` rows. An explicit smaller value is
                expanded to that aligned minimum with a warning.
            local_cache: Local staging policy for normalized data on remote
                stores.
            show_elbow_plot: Whether to display explained variance after a new
                PCA fit.
            update_state: Select the result as the current reduction.
            invalidate_cache: Force a new reduction artifact.

        Returns:
            Reference to the PCA reduction artifact.
        """
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
            local_cache=local_cache,
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
        solver: Literal["streaming", "materialized"] = "streaming",
        n_iter: int = 5,
        n_oversamples: int = 10,
        batch_size: int | None = None,
        local_cache: bool | str = "auto",
        update_state: bool = True,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Fit or reuse latent semantic indexing for normalized data.

        Args:
            normalized: Normalized artifact to reduce. Uses selected assay
                state when omitted.
            from_assay: Assay used to resolve selected normalized data.
            dims: Requested number of retained LSI dimensions.
            skip_first: Whether to omit the first singular component.
            rand_state: Seed used by the randomized decomposition.
            solver: Memory-bounded streaming solver or materialized compatibility
                solver.
            n_iter: Power iterations used by randomized LSI.
            n_oversamples: Extra random vectors used to stabilize the fitted
                singular subspace.
            batch_size: Number of selected cells processed per block.
            local_cache: Local staging policy for normalized data on remote
                stores.
            update_state: Select the result as the current reduction.
            invalidate_cache: Force a new reduction artifact.

        Returns:
            Reference to the LSI reduction artifact.
        """
        if solver not in {"streaming", "materialized"}:
            raise ValueError("solver must be 'streaming' or 'materialized'")
        if isinstance(n_iter, bool) or not isinstance(n_iter, (int, np.integer)):
            raise TypeError("n_iter must be an integer")
        if isinstance(n_oversamples, bool) or not isinstance(
            n_oversamples,
            (int, np.integer),
        ):
            raise TypeError("n_oversamples must be an integer")
        if n_iter < 0:
            raise ValueError("n_iter must be nonnegative")
        if n_oversamples < 0:
            raise ValueError("n_oversamples must be nonnegative")
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
            local_cache=local_cache,
            show_elbow_plot=False,
            update_state=update_state,
            invalidate_cache=invalidate_cache,
            lsi_solver=solver,
            lsi_n_iter=int(n_iter),
            lsi_n_oversamples=int(n_oversamples),
        )

    def run_custom_reduction(
        self,
        loadings: np.ndarray,
        normalized: ArtifactRef | None = None,
        *,
        from_assay: str | None = None,
        batch_size: int | None = None,
        local_cache: bool | str = "auto",
        update_state: bool = True,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Register custom feature loadings as a reusable reduction.

        Args:
            loadings: Two-dimensional feature-by-dimension loading matrix. Its
                row count must match the normalized feature selection.
            normalized: Normalized artifact associated with the loadings. Uses
                selected assay state when omitted.
            from_assay: Assay used to resolve selected normalized data.
            batch_size: Number of selected cells processed per block.
            local_cache: Local staging policy for normalized data on remote
                stores.
            update_state: Select the result as the current reduction.
            invalidate_cache: Force a new reduction artifact.

        Returns:
            Reference to the custom reduction artifact.
        """
        loading_values = np.asarray(loadings)
        if loading_values.ndim != 2 or loading_values.shape[1] < 1:
            raise ValueError(
                "Custom loadings must be a two-dimensional matrix with columns"
            )
        return self._run_reduction_artifact(
            method="custom",
            normalized=normalized,
            from_assay=from_assay,
            dims=int(loading_values.shape[1]),
            pca_cell_key=None,
            feat_scaling=False,
            lsi_skip_first=False,
            custom_loadings=loading_values,
            rand_state=4466,
            batch_size=batch_size,
            local_cache=local_cache,
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
        """Fit or reuse Harmony correction for a reduction artifact."""
        reduction_ref = reduction or self._selected_artifact(
            from_assay,
            "reduction",
            "reduction",
        )
        self._require_complete_artifact(
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
        if not isinstance(cell_key, str):
            raise ValueError("Normalized artifact has no cell_key")
        state = read_assay_state_document(self.zw, reduction_ref.assay)
        if state is not None and state.normalized == normalized_ref:
            cell_key = state.cell_key
        validate_normalized_artifact_selection(
            self.zw,
            normalized_ref,
            cell_key,
        )
        if not isinstance(batch_columns, list) or not batch_columns:
            raise ValueError("batch_columns must be a non-empty list")
        if len(set(batch_columns)) != len(batch_columns):
            raise ValueError("batch_columns must be unique")
        requested_batch_size = (
            None if batch_size is None else _positive_integer(batch_size, "batch_size")
        )
        batches = pd.DataFrame(
            {
                column: self.cells.fetch(column, key=cell_key).astype(object)
                for column in batch_columns
            }
        )
        source, n_cells, dims = self._coordinate_source(
            reduction_ref,
            batch_size=requested_batch_size,
        )
        source_data = getattr(source, "data", None)
        source_batch_size = (
            int(source_data.chunksize[0]) if source_data is not None else n_cells
        )
        effective_batch_size = min(
            source_batch_size if requested_batch_size is None else requested_batch_size,
            n_cells,
        )
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
            algorithm_version="centroid_snapshot_v2",
            batch_size=effective_batch_size,
            invalidate_cache=invalidate_cache,
        )
        planned = self._plan_assay_artifact(
            reduction_ref.assay,
            arguments,
            required_arrays=(
                ArrayRequirement(
                    "data",
                    shape=(n_cells, dims),
                    dtype=np.float32,
                ),
                ArrayRequirement("cluster_mass", dtype=np.float64),
                ArrayRequirement("raw_centroids", dtype=np.float64),
                ArrayRequirement("corrected_centroids", dtype=np.float64),
                ArrayRequirement("centroids", dtype=np.float64),
                ArrayRequirement("sigma", dtype=np.float64),
                ArrayRequirement("ridge", dtype=np.float64),
            ),
            invalidate_cache=invalidate_cache,
        )
        if not planned.reused:
            correction = BatchCorrectionStage(
                stream=source,
                n_cells=n_cells,
                dims=dims,
                batch_size=effective_batch_size,
                batches=batches,
                parameters=harmony_params or {},
                corrected_data=None,
                nthreads=self.nthreads,
            )
            corrected = correction.ensure_corrected()
            result = correction.result
            if result is None:
                raise RuntimeError("Harmony did not return fit metadata")
            from ...mapping.symphony import weighted_centroids

            cluster_mass, raw_centroids = weighted_centroids(
                result.original.T,
                result.assignments,
            )
            _, corrected_centroids = weighted_centroids(
                result.corrected.T,
                result.assignments,
            )
            group = start_artifact(self.zw, planned)
            output = create_numeric_array(
                group,
                "data",
                row_sharded_array_spec(
                    corrected.shape,
                    np.float32,
                    profile=resolve_storage_profile(group.store),
                    band_rows=min(n_cells, 1_000_000),
                    zarr_format=_group_zarr_format(group),
                    fill_value=0.0,
                ),
            )
            for start, stop in iter_shard_row_slices(
                n_cells,
                array_shard_rows(output),
            ):
                output[start:stop, :] = np.asarray(
                    result.corrected[:, start:stop].T,
                    dtype=np.float32,
                )
            for name, values in (
                ("cluster_mass", cluster_mass),
                ("raw_centroids", raw_centroids),
                ("corrected_centroids", corrected_centroids),
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
        action = "Reused" if planned.reused else "Stored"
        logger.info(
            f"{action} Harmony coordinates for {n_cells} cells with {dims} dimensions"
        )
        return planned.ref

    def _build_embedding_initialization(
        self,
        reduction: ArtifactRef,
        *,
        n_centroids: int,
        rand_state: int,
        batch_size: int | None,
        invalidate_cache: bool,
        kmeans_sampling: float = 0.1,
        kmeans_batch_size: int = 10_000,
        algorithm_version: str = "minibatch_kmeans_v2",
    ) -> ArtifactRef:
        if reduction.assay is None:
            raise ValueError("Reduction artifact has no assay")
        resolved_batch_size = (
            None if batch_size is None else _positive_integer(batch_size, "batch_size")
        )
        requested_clusters = _positive_integer(n_centroids, "n_centroids")
        resolved_rand_state = _positive_integer(rand_state, "rand_state")
        resolved_kmeans_sampling = _sampling_fraction(
            kmeans_sampling,
            "kmeans_sampling",
        )
        requested_kmeans_batch_size = _positive_integer(
            kmeans_batch_size,
            "kmeans_batch_size",
        )
        stream, n_cells, coordinate_dims = self._coordinate_source(
            reduction,
            batch_size=resolved_batch_size,
        )
        source_data = getattr(stream, "data", None)
        source_batch_size = (
            int(source_data.chunksize[0]) if source_data is not None else n_cells
        )
        requested_batch_size = (
            source_batch_size if resolved_batch_size is None else resolved_batch_size
        )
        effective_batch_size = min(int(requested_batch_size), n_cells)
        if requested_clusters < 2 or n_cells < 2:
            raise ValueError(
                "Embedding initialization requires at least two cells and centroids"
            )
        effective_clusters = min(
            requested_clusters,
            n_cells,
        )
        effective_kmeans_batch_size = min(
            n_cells,
            max(requested_kmeans_batch_size, effective_clusters),
        )
        arguments = EmbeddingInitializationArguments(
            reduction=reduction,
            n_centroids=effective_clusters,
            rand_state=resolved_rand_state,
            batch_size=effective_batch_size,
            kmeans_sampling=resolved_kmeans_sampling,
            kmeans_batch_size=effective_kmeans_batch_size,
            algorithm_version=algorithm_version,
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
                    dtype=np.uint32,
                ),
            ),
            invalidate_cache=invalidate_cache,
        )
        if not planned.reused:
            initialization = KMeansInitializationStage.fit(
                stream=stream,
                n_rows=n_cells,
                batch_size=effective_batch_size,
                n_clusters=effective_clusters,
                rand_state=resolved_rand_state,
                nthreads=self.nthreads,
                enabled=True,
                kmeans_sampling=resolved_kmeans_sampling,
                kmeans_batch_size=effective_kmeans_batch_size,
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
                np.uint32,
                initialization.labels.shape,
            )
            labels[:] = initialization.labels
            finish_artifact(group, planned)
        action = "Reused" if planned.reused else "Stored"
        logger.info(
            f"{action} embedding initialization with {effective_clusters} centroids"
        )
        return planned.ref

    def build_embedding_initialization(
        self,
        reduction: ArtifactRef | None = None,
        *,
        from_assay: str | None = None,
        n_centroids: int = 1000,
        rand_state: int = 4466,
        batch_size: int | None = None,
        kmeans_sampling: float = 0.1,
        kmeans_batch_size: int = 10_000,
        update_state: bool = True,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Build or reuse K-means embedding initialization for a reduction.

        Downstream UMAP reads this artifact from ``AssayState`` unless you pass
        ``ini_embed`` explicitly.

        Args:
            reduction: Reduction artifact to cluster. Uses the selected assay
                reduction when omitted.
            from_assay: Assay used to resolve the selected reduction.
            n_centroids: Requested number of K-means centroids.
            rand_state: K-means random seed.
            batch_size: Number of cells processed per block.
            kmeans_sampling: Fraction of cells considered during centroid seeding.
            kmeans_batch_size: Number of cells per internal K-means update.
            update_state: Select the result as the current embedding
                initialization.
            invalidate_cache: Force a new initialization artifact.

        Returns:
            Reference to the embedding-initialization artifact.
        """
        if reduction is None:
            assay = from_assay or self._defaultAssay
            if assay is None:
                raise ValueError("No assay was provided and no default is configured")
            state = read_assay_state(self.zw, assay)
            if state is None or state.reduction is None:
                raise KeyError(f"AssayState for {assay!r} has no selected reduction")
            reduction = state.reduction
        elif reduction.assay is not None:
            read_assay_state_document(self.zw, reduction.assay)
        initialization = self._build_embedding_initialization(
            reduction,
            n_centroids=n_centroids,
            rand_state=rand_state,
            batch_size=batch_size,
            invalidate_cache=invalidate_cache,
            kmeans_sampling=kmeans_sampling,
            kmeans_batch_size=kmeans_batch_size,
        )
        if update_state:
            if reduction.assay is None:
                raise ValueError("Reduction artifact has no assay")
            state_document = read_assay_state_document(self.zw, reduction.assay)
            state = None
            if state_document is not None and state_document.reduction == reduction:
                try:
                    validate_assay_state(self.zw, state_document)
                except ArtifactResolutionError:
                    pass
                else:
                    state = state_document
            if state is not None:
                write_assay_state(
                    self.zw,
                    replace(
                        state,
                        embedding_initialization=initialization,
                    ),
                )
            else:
                self._publish_current_artifact(
                    reduction,
                    update_state=True,
                    embedding_initialization=initialization,
                )
        return initialization

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
        update_state: bool = True,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Build or reuse an approximate nearest-neighbor index."""
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
        read_assay_state_document(self.zw, coordinates.assay)
        if ann_metric not in {"l2", "cosine"}:
            raise ValueError("ann_metric must be one of: l2, cosine")
        resolved_ann_efc = _positive_integer(ann_efc, "ann_efc")
        resolved_ann_ef = _positive_integer(ann_ef, "ann_ef")
        resolved_ann_m = _positive_integer(ann_m, "ann_m")
        resolved_rand_state = _positive_integer(rand_state, "rand_state")
        if resolved_ann_m < 2:
            raise ValueError("ann_m must be at least two")
        if not isinstance(ann_parallel, bool):
            raise TypeError("ann_parallel must be a boolean")
        resolved_batch_size = (
            None if batch_size is None else _positive_integer(batch_size, "batch_size")
        )
        coordinate_source, n_cells, dims = self._coordinate_source(
            coordinates,
            batch_size=resolved_batch_size,
        )
        source_data = getattr(coordinate_source, "data", None)
        source_batch_size = (
            int(source_data.chunksize[0]) if source_data is not None else n_cells
        )
        requested_batch_size = (
            source_batch_size if resolved_batch_size is None else resolved_batch_size
        )
        effective_batch_size = min(int(requested_batch_size), n_cells)
        parallel_threads = self.nthreads if ann_parallel else None
        arguments = AnnIndexArguments(
            coordinates=coordinates,
            ann_metric=ann_metric,
            ann_efc=resolved_ann_efc,
            ann_ef=resolved_ann_ef,
            ann_m=resolved_ann_m,
            rand_state=resolved_rand_state,
            ann_parallel=ann_parallel,
            parallel_threads=parallel_threads,
            batch_size=effective_batch_size,
            invalidate_cache=invalidate_cache,
        )

        def valid_ann_artifact(
            _ref: ArtifactRef,
            group: zarr.Group,
        ) -> bool:
            try:
                load_ann_index(
                    group,
                    ann_metric,
                    dims,
                    expected_count=n_cells,
                )
            except (FileNotFoundError, RuntimeError, ValueError):
                return False
            return True

        planned = self._plan_assay_artifact(
            coordinates.assay,
            arguments,
            required_arrays=(ArrayRequirement("ann_idx_bytes", dtype=np.uint8),),
            invalidate_cache=invalidate_cache,
            reuse_validator=valid_ann_artifact,
        )
        if not planned.reused:
            ann_idx = AnnIndexStage.fit(
                coordinates=coordinate_source,
                metric=ann_metric,
                dims=dims,
                n_cells=n_cells,
                ef_construction=resolved_ann_efc,
                ef=resolved_ann_ef,
                m=resolved_ann_m,
                rand_state=resolved_rand_state,
                nthreads=(self.nthreads if ann_parallel else 1),
            )
            group = start_artifact(self.zw, planned)
            self._persist_ann_index(
                artifact_path(planned.ref),
                ann_idx,
                ann_metric=ann_metric,
                dimensions=dims,
                element_count=n_cells,
            )
            finish_artifact(group, planned)
        self._publish_current_artifact(
            planned.ref,
            update_state=update_state,
        )
        action = "Reused" if planned.reused else "Stored"
        logger.info(f"{action} ANN index for {n_cells} cells")
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
        """Query an ANN artifact and persist compact neighbor matrices."""
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
        read_assay_state_document(self.zw, ann_ref.assay)
        raw_coordinates = (ann_status.inputs or {}).get("coordinates")
        if not isinstance(raw_coordinates, dict):
            raise ValueError("ANN artifact has no coordinates input")
        stored_coordinates = ArtifactRef.from_dict(raw_coordinates)
        if stored_coordinates.kind not in {
            "reduction",
            "batch_correction",
            "imported_coordinates",
        }:
            raise ValueError(
                "ANN coordinates must be reduction, batch_correction, "
                "or imported_coordinates"
            )
        self._require_complete_artifact(
            stored_coordinates,
            stored_coordinates.kind,
        )
        if coordinates is not None and coordinates != stored_coordinates:
            raise ValueError("coordinates do not match the ANN artifact input")
        requested_k = _positive_integer(k, "k")
        resolved_batch_size = (
            None if batch_size is None else _positive_integer(batch_size, "batch_size")
        )
        coordinate_source, n_cells, dims = self._coordinate_source(
            stored_coordinates,
            batch_size=resolved_batch_size,
        )
        if n_cells < 2:
            raise ValueError("Neighbor queries require at least two cells")
        effective_k = min(requested_k, n_cells - 1)
        if n_cells - 1 > np.iinfo(np.uint32).max:
            raise ValueError("Neighbor indices require fewer than 2**32 cells")
        source_data = getattr(coordinate_source, "data", None)
        source_batch_size = (
            int(source_data.chunksize[0]) if source_data is not None else n_cells
        )
        requested_batch_size = (
            source_batch_size if resolved_batch_size is None else resolved_batch_size
        )
        effective_batch_size = min(int(requested_batch_size), n_cells)
        ann_parameters = ann_status.parameters or {}
        ann_metric = ann_parameters.get("ann_metric")
        if ann_metric not in {"l2", "cosine"}:
            raise ValueError("ANN artifact has no supported distance metric")
        arguments = NeighborQueryArguments(
            ann_index=ann_ref,
            coordinates=stored_coordinates,
            k=effective_k,
            distance_metric=str(ann_metric),
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
                    dtype=np.uint32,
                ),
                ArrayRequirement(
                    "distances",
                    shape=(n_cells, effective_k),
                    dtype=np.float32,
                ),
            ),
            invalidate_cache=invalidate_cache,
        )
        if not planned.reused:
            ann_idx = self._resolve_ann_index(
                ann_ref,
                str(ann_metric),
                dims,
                expected_count=n_cells,
            )
            ann_idx = AnnIndexStage.configure(
                ann_idx,
                ef=int(ann_parameters.get("ann_ef", 50)),
                threads=(int(ann_parameters.get("parallel_threads") or 1)),
            )
            query = NeighborQueryStage(
                ann_idx,
                effective_k,
                str(ann_metric),
            )
            indices = np.empty((n_cells, effective_k), dtype=np.uint32)
            distances = np.empty((n_cells, effective_k), dtype=np.float32)
            start = 0
            missed_self_hits = 0
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
                if np.any(block_indices < 0) or np.any(block_indices >= n_cells):
                    raise ValueError("ANN query returned an invalid cell index")
                indices[start:stop, :] = block_indices
                distances[start:stop, :] = block_distances
                missed_self_hits += missed
                start = stop
            if start != n_cells:
                raise ValueError(
                    f"Coordinate source contains {start} rows, expected {n_cells}"
                )
            group = start_artifact(self.zw, planned)
            array_profile = resolve_storage_profile(group.store)
            zarr_format = _group_zarr_format(group)
            indices_array = create_numeric_array(
                group,
                "indices",
                row_sharded_array_spec(
                    indices.shape,
                    np.uint32,
                    profile=array_profile,
                    band_rows=min(n_cells, 1_000_000),
                    zarr_format=zarr_format,
                ),
            )
            distances_array = create_numeric_array(
                group,
                "distances",
                row_sharded_array_spec(
                    distances.shape,
                    np.float32,
                    profile=array_profile,
                    band_rows=min(n_cells, 1_000_000),
                    zarr_format=zarr_format,
                    fill_value=0.0,
                ),
            )
            indices_array[:, :] = indices
            distances_array[:, :] = distances
            group.attrs["n_cells"] = n_cells
            group.attrs["n_neighbors"] = effective_k
            group.attrs["self_hit_rate"] = (
                100.0 * (n_cells - missed_self_hits) / n_cells
            )
            finish_artifact(group, planned)
        self._publish_current_artifact(
            planned.ref,
            update_state=update_state,
        )
        action = "Reused" if planned.reused else "Stored"
        logger.info(f"{action} {effective_k} neighbors for each of {n_cells} cells")
        return planned.ref

    def build_connectivity_map(
        self,
        neighbors: ArtifactRef | None = None,
        *,
        from_assay: str | None = None,
        local_connectivity: float = 1.0,
        bandwidth: float = 1.5,
        update_state: bool = True,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Convert persisted neighbors into a weighted connectivity graph.

        Args:
            neighbors: Neighbors artifact. Uses current assay state when
                omitted.
            from_assay: Assay used to resolve current neighbors.
            local_connectivity: UMAP-style local-connectivity adjustment.
            bandwidth: Distance-kernel bandwidth multiplier.
            update_state: Select the result as the current connectivity map.
            invalidate_cache: Force a new connectivity artifact.

        Returns:
            Reference to the connectivity-map artifact.
        """
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
        read_assay_state_document(self.zw, neighbors_ref.assay)
        group = group_at(self.zw, status.path)
        indices = as_zarr_array(group["indices"], name="indices")
        n_cells, n_neighbors = map(int, indices.shape)
        validate_distance_provenance(self.zw, neighbors_ref)
        arguments = ConnectivityMapArguments(
            neighbors=neighbors_ref,
            local_connectivity=local_connectivity,
            bandwidth=bandwidth,
            invalidate_cache=invalidate_cache,
        )
        planned = self._plan_assay_artifact(
            neighbors_ref.assay,
            arguments,
            required_arrays=(
                ArrayRequirement(
                    "edges",
                    shape=(None, 2),
                    dtype=np.uint32,
                ),
                ArrayRequirement(
                    "weights",
                    shape=(None,),
                    dtype=np.float32,
                ),
            ),
            invalidate_cache=invalidate_cache,
        )
        if not planned.reused:
            from ...neighbors.graph import build_connectivity_arrays

            distance_values = np.asarray(
                as_zarr_array(
                    group["distances"],
                    name="distances",
                )[:]
            )
            edge_values, weight_values = build_connectivity_arrays(
                np.asarray(indices[:]),
                distance_values,
                local_connectivity=local_connectivity,
                bandwidth=bandwidth,
            )
            output = start_artifact(self.zw, planned)
            profile = resolve_storage_profile(output.store)
            zarr_format = _group_zarr_format(output)
            edge_band_rows = min(n_cells, 1_000_000) * n_neighbors
            edges = create_numeric_array(
                output,
                "edges",
                row_sharded_array_spec(
                    edge_values.shape,
                    np.uint32,
                    profile=profile,
                    band_rows=edge_band_rows,
                    zarr_format=zarr_format,
                ),
            )
            weights = create_numeric_array(
                output,
                "weights",
                row_sharded_array_spec(
                    weight_values.shape,
                    np.float32,
                    profile=profile,
                    band_rows=edge_band_rows,
                    zarr_format=zarr_format,
                    fill_value=0.0,
                ),
            )
            edges[:, :] = edge_values
            weights[:] = weight_values
            output.attrs["n_cells"] = n_cells
            output.attrs["n_neighbors"] = n_neighbors
            finish_artifact(output, planned)
        self._publish_current_artifact(
            planned.ref,
            update_state=update_state,
        )
        action = "Reused" if planned.reused else "Stored"
        logger.info(f"{action} connectivity map for {n_cells} cells")
        return planned.ref

    def _load_graph_artifact(
        self,
        graph: ArtifactRef,
        *,
        symmetric: bool | None,
        upper_only: bool | None,
        use_k: int | None,
    ) -> csr_matrix:
        """Load one already captured and validated graph reference."""

        def symmetrize(g: csr_matrix) -> csr_matrix:
            t = g + g.T
            t = t - g.multiply(g.T)
            return t

        from scipy.sparse import triu

        if graph.kind not in {"connectivity_map", "integrated_graph"}:
            raise ValueError(
                "Graph reference must be connectivity_map or integrated_graph"
            )
        graph_loc = require_complete_artifact(self.zw, graph).path
        cache_key = (
            graph_loc,
            symmetric is True,
            symmetric is True and upper_only is True,
            use_k,
        )
        cache = getattr(self, "_graphMemoryCache", None)
        if cache is not None:
            with self._graphMemoryCacheLock:
                cached = cache.get(cache_key)
            if cached is not None:
                return cached
        _n_cells, matrix = self._store_to_sparse(graph_loc, "csr", use_k)
        assert isinstance(matrix, csr_matrix)
        if symmetric is True:
            matrix = symmetrize(matrix)
            if upper_only is True:
                matrix = triu(matrix).tocsr()
        if cache is not None:
            with self._graphMemoryCacheLock:
                active_cache = getattr(self, "_graphMemoryCache", None)
                if active_cache is cache:
                    matrix = active_cache.setdefault(cache_key, matrix)
        return matrix

    def load_graph(
        self,
        graph: ArtifactRef | None = None,
        *,
        from_assay: str | None = None,
        cell_key: str | None = None,
        symmetric: bool | None = None,
        upper_only: bool | None = None,
        use_k: int | None = None,
    ) -> csr_matrix:
        """Load the cell neighbourhood as a scipy sparse matrix.

        Args:
            graph: Connectivity-map or integrated-graph artifact. The current
                assay graph is used when omitted.
            from_assay: Name of the assay used for current-state resolution.
            cell_key: Optional cell key, validated against the graph lineage.
            symmetric: If True, makes the graph symmetric by adding it to its transpose.
            upper_only: If True, then only the values from upper triangular of the matrix are returned. This is only
                       used when symmetric is True.
            use_k: Number of top k-nearest neighbours to keep in the graph. This value must be greater than 0 and less
                   the parameter k used. By default, all neighbours are used. (Default value: None)

        Returns:
            A scipy sparse matrix representing cell neighbourhood graph.
        """

        from ...graph.state import resolve_graph_selection

        selection = resolve_graph_selection(
            self,
            graph,
            from_assay=from_assay,
            cell_key=cell_key,
        )
        return self._load_graph_artifact(
            selection.graph_ref,
            symmetric=symmetric,
            upper_only=upper_only,
            use_k=use_k,
        )

    def integrate_assays(
        self,
        assays: list[str],
        label: str,
        method: str = "snn",
        chunk_size: int = 10000,
        invalidate_cache: bool = False,
        l2_normalize: bool = True,
    ) -> ArtifactRef:
        """Integrate the current state-selected graphs for selected assays.

        SNN combines shared edge support across two or more assays. WNN accepts
        two or more assays and uses Hao-inspired per-cell modality weights.
        Scarf WNN scores only the union of the existing self-free KNN rows and
        uses the distance span from the nearest to the k-th neighbour as its
        bandwidth, so it is not bit-identical to Seurat's default wider search
        and SNN-far bandwidth.

        Args:
            assays: Input assay names. Each assay's current state-selected graph
                or neighbors artifact is captured once.
            label: Label for integrated graph
            method: Choose a method for modality integration. Available options: 'snn': Shared nearest neighbour
                    approach and 'wnn': Hao-inspired weighted nearest neighbor integration.
            chunk_size: number of cells to be loaded at a time while reading and writing the graph
            invalidate_cache: Force a new integrated-graph artifact.
            l2_normalize: L2-normalize modality coordinates during WNN scoring.
                This algorithmic setting is stored in artifact provenance.

        Returns:
            Reference to the integrated-graph artifact. Pass it to `run_umap`,
            `run_tsne`, or the clustering methods as their ``graph`` argument.

        WNN stores one modality-weight column per assay, named
        ``{label}_{assay}_weight`` in cell metadata.
        """
        from ...neighbors.graph import merge_graphs
        from ...neighbors.integration import _wnn_integration_many

        assays = list(assays)
        if method not in {"snn", "wnn"}:
            raise ValueError(
                f"Method {method} not supported, choose one of these: 'snn', 'wnn'"
            )
        if len(assays) < 2:
            raise ValueError("Assay integration requires at least two assays")
        if len(set(assays)) != len(assays):
            raise ValueError("Assay integration requires unique assay names")
        if method == "wnn" and not isinstance(l2_normalize, bool | np.bool_):
            raise TypeError("l2_normalize must be a boolean")

        def materialize_coordinate_blocks(
            blocks: Iterator[np.ndarray],
            n_cells: int,
            *,
            expected_dims: int | None = None,
        ) -> np.ndarray:
            coordinates: np.ndarray | None = None
            start = 0
            for values in blocks:
                block = np.asarray(values)
                if block.ndim != 2:
                    raise ValueError("WNN coordinate blocks must be matrices")
                if expected_dims is not None and block.shape[1] != expected_dims:
                    raise ValueError("WNN coordinate dimensions changed between blocks")
                if coordinates is None:
                    coordinates = np.empty(
                        (n_cells, block.shape[1]),
                        dtype=block.dtype,
                    )
                stop = start + len(block)
                if stop > n_cells:
                    raise ValueError("WNN coordinate stream exceeded its cell count")
                coordinates[start:stop] = block
                start = stop
            if coordinates is None or start != n_cells:
                raise ValueError("WNN coordinate stream did not cover every cell")
            return coordinates

        source_inputs: dict[str, Any] = {}
        captured_sources: list[ArtifactRef] = []
        captured_coordinates: list[ArtifactRef | None] = []
        shared_selection: ArtifactRef | None = None
        shared_cell_key: str | None = None
        shared_source_n_cells: int | None = None
        for index, assay_name in enumerate(assays):
            if assay_name not in self.assay_names:
                raise ValueError(f"ERROR: Assay {assay_name} was not found.")
            state = read_assay_state(self.zw, assay_name)
            source = (
                None
                if state is None
                else (state.neighbors if method == "wnn" else state.connectivity_map)
            )
            if source is None:
                code = (
                    "missing_current_neighbors"
                    if method == "wnn"
                    else "missing_current_graph"
                )
                noun = "neighbors" if method == "wnn" else "connectivity map"
                raise ArtifactResolutionError(
                    f"Assay {assay_name!r} has no current {noun}",
                    code=code,
                    context={"assay": assay_name},
                )
            assert state is not None
            ancestry = resolve_native_graph_inputs(self.zw, source)
            if method == "wnn" and ancestry.coordinates.kind not in {
                "reduction",
                "batch_correction",
            }:
                raise ArtifactResolutionError(
                    "WNN coordinates must be reduction or batch_correction",
                    code="wrong_kind",
                    context={
                        "assay": assay_name,
                        "artifact_id": ancestry.coordinates.artifact_id,
                        "actual_kind": ancestry.coordinates.kind,
                        "expected_kind": "reduction,batch_correction",
                    },
                )
            validate_cell_selection_artifact(
                self.zw,
                ancestry.cell_selection,
                state.cell_key,
            )
            source_n_cells = _validate_integration_source_payload(self.zw, source)
            if shared_source_n_cells is None:
                shared_source_n_cells = source_n_cells
            elif source_n_cells != shared_source_n_cells:
                raise _integration_payload_error(
                    source,
                    "Integration sources contain different cell counts",
                )
            selection = ancestry.cell_selection
            captured_sources.append(source)
            captured_coordinates.append(
                ancestry.coordinates if method == "wnn" else None
            )
            source_inputs[f"source_{index}"] = (
                {
                    "neighbors": source,
                    "coordinates": ancestry.coordinates,
                }
                if method == "wnn"
                else source
            )
            cell_key = state.cell_key
            if shared_selection is None:
                shared_selection = selection
                shared_cell_key = cell_key
            elif shared_selection != selection:
                raise ValueError(
                    "Integrated graphs require one exact shared cell selection"
                )
        if shared_selection is None:
            raise ValueError("No assay cell selection was resolved")
        if shared_cell_key is None:
            raise RuntimeError("No cell key was resolved for assay integration")
        source_inputs["cell_selection"] = shared_selection
        parameters: dict[str, Any] = {"method": method, "assays": assays}
        required_arrays: list[ArrayRequirement] = [
            ArrayRequirement("edges"),
            ArrayRequirement("weights", dtype_kind="f"),
        ]
        if method == "wnn":
            parameters["l2_normalize"] = bool(l2_normalize)
            required_arrays.append(
                ArrayRequirement(
                    "modality_weights",
                    shape=(None, len(assays)),
                    dtype=np.float32,
                )
            )
        integrated_plan = plan_artifact(
            self.zw,
            scope="datastore",
            kind="integrated_graph",
            operation="integrate_assays",
            parameters=parameters,
            inputs=source_inputs,
            execution_options={"label": label, "chunk_size": chunk_size},
            invalidate_cache=invalidate_cache,
            required_arrays=tuple(required_arrays),
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

        weight_columns = (
            [f"{label}_{assay_name}_weight" for assay_name in assays]
            if method == "wnn"
            else []
        )
        preserved_weight_displays = {
            column: column_display(self.zw, column) for column in weight_columns
        }

        def publish_modality_weights() -> None:
            if method != "wnn":
                return
            group = artifact_group(self.zw, integrated_plan.ref)
            stored_assays = group.attrs.get("assays")
            if not isinstance(stored_assays, list) or stored_assays != assays:
                raise RuntimeError("Stored WNN modality assay order is invalid")
            try:
                stored_n_cells = _positive_integer(
                    group.attrs.get("n_cells"),
                    "stored n_cells",
                )
            except (TypeError, ValueError) as error:
                raise RuntimeError(
                    "Stored WNN modality weights have invalid cell metadata"
                ) from error
            values = np.asarray(
                as_zarr_array(
                    group["modality_weights"],
                    name="modality_weights",
                )[:],
                dtype=np.float32,
            )
            if values.shape != (stored_n_cells, len(assays)):
                raise RuntimeError("Stored WNN modality weights have an invalid shape")
            for index, column in enumerate(weight_columns):
                column_values = values[:, index]
                self.cells.insert(
                    column,
                    column_values,
                    overwrite=True,
                    key=shared_cell_key,
                )
                link_cell_data_column(
                    self.zw,
                    column,
                    integrated_plan.ref,
                    value_name="modality_weights",
                    value_index=index,
                    default_display=continuous_display(column_values),
                    preserved_display=preserved_weight_displays[column],
                )

        if integrated_plan.reused:
            select_integrated_artifact()
            publish_modality_weights()
            return integrated_plan.ref

        def load_wnn_inputs(
            index: int,
            assay_name: str,
        ) -> tuple[np.ndarray, NDArray[Any]]:
            neighbors = captured_sources[index]
            coordinates_ref = captured_coordinates[index]
            if neighbors.kind != "neighbors" or coordinates_ref is None:
                raise RuntimeError("Captured WNN source is invalid")
            neighbors_group = artifact_group(self.zw, neighbors)
            indices = np.asarray(
                as_zarr_array(
                    neighbors_group["indices"],
                    name="indices",
                )[:]
            )
            coordinate_source, n_cells, dims = self._coordinate_source(
                coordinates_ref,
                batch_size=None,
            )
            coordinates = materialize_coordinate_blocks(
                (
                    np.asarray(block)
                    for block in coordinate_source.iter_coordinate_blocks(
                        f"Loading {assay_name} coordinates",
                    )
                ),
                n_cells,
                expected_dims=dims,
            )
            if indices.shape[0] != n_cells:
                raise ValueError(
                    f"WNN neighbors and coordinates for {assay_name} "
                    "contain different cell counts"
                )
            return indices, coordinates

        modality_weights: np.ndarray | None = None
        if method == "snn":
            graphs = [
                self._load_graph_artifact(
                    source,
                    symmetric=None,
                    upper_only=None,
                    use_k=None,
                ).tocsr()
                for source in captured_sources
            ]
            merged_graph = merge_graphs(graphs)
        elif method == "wnn":
            modalities = [
                (assay, *load_wnn_inputs(index, assay))
                for index, assay in enumerate(assays)
            ]
            merged_graph, modality_weights = _wnn_integration_many(
                modalities,
                self.nthreads,
                l2_normalize=l2_normalize,
            )
        n_cells = merged_graph.shape[0]
        n_neighbors = int(merged_graph.size / n_cells)

        store = start_artifact(self.zw, integrated_plan)
        store.attrs["n_cells"] = n_cells
        store.attrs["n_neighbors"] = n_neighbors
        store.attrs["assays"] = list(assays)

        edge_chunk = chunk_size * n_neighbors
        zge = create_zarr_dataset(
            store,
            "edges",
            (edge_chunk,),
            np.uint32,
            (n_cells * n_neighbors, 2),
        )
        zgw = create_zarr_dataset(
            store,
            "weights",
            (edge_chunk,),
            np.float32,
            (n_cells * n_neighbors,),
        )

        zge[:, 0] = merged_graph.row
        zge[:, 1] = merged_graph.col
        zgw[:] = merged_graph.data
        if modality_weights is not None:
            stored_modality_weights = create_zarr_dataset(
                store,
                "modality_weights",
                (min(chunk_size, n_cells), len(assays)),
                np.float32,
                modality_weights.shape,
            )
            stored_modality_weights[:, :] = modality_weights
        finish_artifact(store, integrated_plan)
        select_integrated_artifact()
        publish_modality_weights()
        return integrated_plan.ref
