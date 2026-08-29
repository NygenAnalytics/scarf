import math
import os
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from threading import RLock
from typing import TYPE_CHECKING, Any, Literal, cast

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
from ...graph.distances import (
    validate_distance_provenance,
    validate_neighbors_payload,
)
from ...graph.feature_projection import (
    graph_cell_selection,
    resolve_coordinate_inputs,
    resolve_native_graph_inputs,
)
from ...embeddings.imported_storage import (
    validate_imported_coordinates_artifact,
)
from ...matrix import ChunkedArray
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
    group_at,
    inspect_artifact,
    require_complete_artifact,
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
    ValidatedStoredSelection,
    iter_selected_axis_selection_blocks,
    read_stored_selection_mask,
    resolve_selection_artifact,
    snapshot_run_metadata,
    validate_run_metadata_snapshot,
    validate_stored_selection_integrity,
)
from ...utils.arrays import clean_array
from ...utils.logging import logger
from ...utils.shutdown import shutdown_checkpoint

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


def _validate_integration_source_payload(
    root: zarr.Group,
    ref: ArtifactRef,
) -> int:
    if ref.kind == "connectivity_map":
        return _validate_integration_connectivity_payload(root, ref)
    if ref.kind == "neighbors":
        return validate_neighbors_payload(root, ref).n_cells
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

        def select_all_features(
            self,
            *,
            from_assay: str | None = None,
        ) -> ArtifactRef: ...

        def resolve_features(
            self,
            assay: str,
            features: ArtifactRef,
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

    def _load_artifact_ann_stream(
        self,
        neighbors_ref: ArtifactRef,
        feat_scaling: bool,
    ) -> AnnStream:
        if neighbors_ref.scope != "assay" or neighbors_ref.assay is None:
            raise ValueError("neighbors must be assay-scoped")
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
                read_stored_selection_mask(
                    self.zw,
                    self._artifact_input_ref(
                        reduction_ref,
                        "pca_cell_selection",
                        "cell_selection",
                    ),
                    kind="cell_selection",
                    scope="datastore",
                    assay=None,
                    table_path="cellData",
                )[
                    read_stored_selection_mask(
                        self.zw,
                        lineage.cell_selection,
                        kind="cell_selection",
                        scope="datastore",
                        assay=None,
                        table_path="cellData",
                    )
                ]
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
        del metadata_group
        if getattr(self, "zarr_mode", "r+") != "r+":
            raise PermissionError(
                f"Snapshotting selection column {column!r} requires a writable store"
            )
        return resolve_selection_artifact(
            self.zw,
            scope=scope,
            assay=assay,
            kind=kind,
            values=values,
            row_ids=row_ids,
            operation="snapshot_manual_selection",
            parameters={},
            inputs={},
            source_column=column,
            invalidate_cache=invalidate_cache,
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
        resolve_coordinate_inputs(self.zw, coordinates)
        if coordinates.kind == "imported_coordinates":
            status = self._require_complete_artifact(
                coordinates,
                "imported_coordinates",
            )
            validate_imported_coordinates_artifact(self.zw, coordinates)
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
        cell_selection: ArtifactRef,
        features: ArtifactRef,
        *,
        log_transform: bool | None = None,
        renormalize_subset: bool | None = None,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Normalize explicit immutable cell and feature selections."""
        if not isinstance(cell_selection, ArtifactRef):
            raise TypeError("cell_selection must be an ArtifactRef")
        if not isinstance(features, ArtifactRef):
            raise TypeError("features must be an ArtifactRef")
        assay_name = features.assay
        if assay_name is None:
            raise ValueError("Feature-selection artifact has no assay")
        assay = self._get_assay(assay_name)
        feature_selection = self.resolve_features(assay_name, features)
        validated_cells = validate_stored_selection_integrity(
            self.zw,
            cell_selection,
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        )
        cell_values = np.asarray(validated_cells.values[:], dtype=bool)
        feature_group = artifact_group(self.zw, feature_selection)
        feature_values = np.asarray(
            as_zarr_array(feature_group["values"], name="values")[:],
            dtype=bool,
        )
        n_cells = validated_cells.selected_count
        n_features = int(feature_values.sum())
        if n_cells < 1 or n_features < 1:
            raise ValueError("Normalization requires selected cells and features")
        from ...assay import ATACassay

        if isinstance(assay, ATACassay):
            if log_transform is None:
                log_transform = False
            elif not isinstance(log_transform, bool | np.bool_):
                raise TypeError("log_transform must be a boolean")
            if log_transform:
                raise ValueError(
                    "ATAC TF-IDF does not support log_transform; use False"
                )
            if renormalize_subset is None:
                renormalize_subset = False
            elif not isinstance(renormalize_subset, bool | np.bool_):
                raise TypeError("renormalize_subset must be a boolean")
        else:
            if log_transform is None:
                log_transform = True
            elif not isinstance(log_transform, bool | np.bool_):
                raise TypeError("log_transform must be a boolean")
            if renormalize_subset is None:
                renormalize_subset = True
            elif not isinstance(renormalize_subset, bool | np.bool_):
                raise TypeError("renormalize_subset must be a boolean")
        log_transform = bool(log_transform)
        renormalize_subset = bool(renormalize_subset)
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
        raw_dataset_fingerprint = assay.attrs.get("dataset_fingerprint")
        dataset_fingerprint = (
            raw_dataset_fingerprint
            if isinstance(raw_dataset_fingerprint, str) and raw_dataset_fingerprint
            else self._calculate_dataset_fingerprint(assay_name)
        )
        arguments = NormalizationArguments(
            cell_selection=cell_selection,
            feature_selection=feature_selection,
            dataset_fingerprint=dataset_fingerprint,
            normalization_method=normalization_method,
            size_factor=(
                float(cast(int | float, raw_size_factor))
                if raw_size_factor is not None
                else None
            ),
            log_transform=log_transform,
            renormalize_subset=renormalize_subset,
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
        if not planned.reused:
            group = start_artifact(self.zw, planned)
            relative_path = artifact_path(planned.ref).removeprefix(f"{assay_name}/")
            assay._write_normalized_payload(
                np.flatnonzero(cell_values),
                np.flatnonzero(feature_values),
                relative_path,
                log_transform=log_transform,
                renormalize_subset=renormalize_subset,
            )
            finish_artifact(group, planned)
        action = "Reused" if planned.reused else "Stored"
        logger.info(
            f"{action} normalized data for {n_cells} cells and {n_features} features"
        )
        return planned.ref

    def _run_reduction_artifact(
        self,
        *,
        method: str,
        normalized: ArtifactRef,
        dims: int,
        pca_cell_selection: ArtifactRef | None,
        feat_scaling: bool,
        lsi_skip_first: bool,
        custom_loadings: np.ndarray | None,
        rand_state: int,
        batch_size: int | None,
        local_cache: bool | str,
        show_elbow_plot: bool,
        invalidate_cache: bool,
        lsi_solver: Literal["streaming", "materialized"] = "streaming",
        lsi_n_iter: int = 5,
        lsi_n_oversamples: int = 10,
    ) -> ArtifactRef:
        requested_dims = _positive_integer(dims, "dims")
        if batch_size is not None:
            _positive_integer(batch_size, "batch_size")
        if not isinstance(normalized, ArtifactRef):
            raise TypeError("normalized must be an ArtifactRef")
        normalized_ref = normalized
        status = self._require_complete_artifact(normalized_ref, "normalized")
        if normalized_ref.assay is None:
            raise ValueError("Normalized artifact has no assay")
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
                    dims=requested_dims,
                    pca_cell_selection=pca_cell_selection,
                    feat_scaling=feat_scaling,
                    lsi_skip_first=lsi_skip_first,
                    custom_loadings=custom_loadings,
                    rand_state=rand_state,
                    batch_size=effective_batch_size,
                    show_elbow_plot=show_elbow_plot,
                    invalidate_cache=invalidate_cache,
                    lsi_solver=lsi_solver,
                    lsi_n_iter=lsi_n_iter,
                    lsi_n_oversamples=lsi_n_oversamples,
                )

    def _run_reduction_artifact_impl(
        self,
        *,
        method: str,
        normalized: ArtifactRef,
        dims: int,
        pca_cell_selection: ArtifactRef | None,
        feat_scaling: bool,
        lsi_skip_first: bool,
        custom_loadings: np.ndarray | None,
        rand_state: int,
        batch_size: int | None,
        show_elbow_plot: bool,
        invalidate_cache: bool,
        lsi_solver: Literal["streaming", "materialized"] = "streaming",
        lsi_n_iter: int = 5,
        lsi_n_oversamples: int = 10,
    ) -> ArtifactRef:
        normalized_ref = normalized
        normalized_status = self._require_complete_artifact(
            normalized_ref,
            "normalized",
        )
        if normalized_ref.assay is None:
            raise ValueError("Normalized artifact has no assay")
        assay_name = normalized_ref.assay
        normalized_cell_selection = self._artifact_input_ref(
            normalized_ref,
            "cell_selection",
            "cell_selection",
        )
        normalized_feature_selection = self._artifact_input_ref(
            normalized_ref,
            "feature_selection",
            "feature_selection",
        )
        normalized_feature_selection = self.resolve_features(
            assay_name,
            normalized_feature_selection,
        )
        validated_normalized_cells = validate_stored_selection_integrity(
            self.zw,
            normalized_cell_selection,
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        )
        data_group = as_zarr_group(
            self.zw[normalized_status.path],
            name=normalized_status.path,
        )
        data_array = as_zarr_array(data_group["data"], name="data")
        n_cells, n_features = map(int, data_array.shape)
        feature_group = artifact_group(self.zw, normalized_feature_selection)
        selected_features = int(
            np.count_nonzero(
                np.asarray(
                    as_zarr_array(feature_group["values"], name="values")[:],
                    dtype=bool,
                )
            )
        )
        if selected_features != n_features:
            raise ArtifactResolutionError(
                "Normalized columns do not match its feature selection",
                code="column_mismatch",
                context={
                    "assay": assay_name,
                    "artifact_id": normalized_ref.artifact_id,
                    "normalized_columns": n_features,
                    "selected_count": selected_features,
                },
            )
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
        if validated_normalized_cells.selected_count != n_cells:
            raise ArtifactResolutionError(
                "Normalized rows do not match its cell selection",
                code="row_mismatch",
                context={
                    "assay": assay_name,
                    "artifact_id": normalized_ref.artifact_id,
                    "normalized_rows": n_cells,
                    "selected_count": validated_normalized_cells.selected_count,
                },
            )
        pca_selection = pca_cell_selection or normalized_cell_selection
        pca_use_values: np.ndarray | None = None
        if method == "pca":
            normalized_mask = read_stored_selection_mask(
                self.zw,
                normalized_cell_selection,
                kind="cell_selection",
                scope="datastore",
                assay=None,
                table_path="cellData",
            )
            pca_mask = (
                normalized_mask
                if pca_cell_selection is None
                else read_stored_selection_mask(
                    self.zw,
                    pca_cell_selection,
                    kind="cell_selection",
                    scope="datastore",
                    assay=None,
                    table_path="cellData",
                )
            )
            if np.any(pca_mask & ~normalized_mask):
                raise ArtifactResolutionError(
                    "PCA cell selection must be a subset of normalized cells",
                    code="row_mismatch",
                    context={
                        "assay": assay_name,
                        "artifact_id": pca_selection.artifact_id,
                    },
                )
            pca_use_values = pca_mask[normalized_mask]
            selected_pca_cells = int(pca_use_values.sum())
            if selected_pca_cells < effective_dims + 1:
                raise ValueError("PCA requires at least dims + 1 selected cells")
            if n_features < effective_dims + 1:
                raise ValueError("PCA requires at least dims + 1 selected features")
            if effective_batch_size < effective_dims + 1:
                raise ValueError("PCA batch_size must be at least dims + 1")
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
                dims=effective_dims,
                feat_scaling=feat_scaling,
                batch_size=effective_batch_size,
                show_elbow_plot=show_elbow_plot,
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
        action = "Reused" if planned.reused else "Stored"
        logger.info(
            f"{action} {method.upper()} reduction for {n_cells} cells "
            f"with {effective_dims} dimensions"
        )
        return planned.ref

    def run_pca(
        self,
        normalized: ArtifactRef,
        *,
        dims: int = 21,
        pca_cell_selection: ArtifactRef | None = None,
        feat_scaling: bool = True,
        batch_size: int | None = None,
        local_cache: bool | str = "auto",
        show_elbow_plot: bool = False,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Fit or reuse PCA for a normalized artifact.

        Args:
            normalized: Normalized artifact to reduce.
            dims: Requested number of principal components. (Default: 21)
            pca_cell_selection: Optional stored cell-selection artifact used to
                fit PCA while projecting every cell in ``normalized``.
            feat_scaling: Whether to standardize features before fitting PCA.
            batch_size: Number of selected cells processed per block. When
                omitted, whole stored row bands are combined as needed to fit
                at least ``dims + 1`` rows. An explicit smaller value is
                expanded to that aligned minimum with a warning.
            local_cache: Local staging policy for normalized data on remote
                stores.
            show_elbow_plot: Whether to display explained variance after a new
                PCA fit.
            invalidate_cache: Force a new reduction artifact.

        Returns:
            Reference to the PCA reduction artifact.
        """
        return self._run_reduction_artifact(
            method="pca",
            normalized=normalized,
            dims=dims,
            pca_cell_selection=pca_cell_selection,
            feat_scaling=feat_scaling,
            lsi_skip_first=False,
            custom_loadings=None,
            rand_state=4466,
            batch_size=batch_size,
            local_cache=local_cache,
            show_elbow_plot=show_elbow_plot,
            invalidate_cache=invalidate_cache,
        )

    def run_lsi(
        self,
        normalized: ArtifactRef,
        *,
        dims: int = 11,
        skip_first: bool = True,
        rand_state: int = 4466,
        solver: Literal["streaming", "materialized"] = "streaming",
        n_iter: int = 5,
        n_oversamples: int = 10,
        batch_size: int | None = None,
        local_cache: bool | str = "auto",
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Fit or reuse latent semantic indexing for normalized data.

        Args:
            normalized: Normalized artifact to reduce.
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
            dims=dims,
            pca_cell_selection=None,
            feat_scaling=False,
            lsi_skip_first=skip_first,
            custom_loadings=None,
            rand_state=rand_state,
            batch_size=batch_size,
            local_cache=local_cache,
            show_elbow_plot=False,
            invalidate_cache=invalidate_cache,
            lsi_solver=solver,
            lsi_n_iter=int(n_iter),
            lsi_n_oversamples=int(n_oversamples),
        )

    def run_custom_reduction(
        self,
        loadings: np.ndarray,
        normalized: ArtifactRef,
        *,
        batch_size: int | None = None,
        local_cache: bool | str = "auto",
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Register custom feature loadings as a reusable reduction.

        Args:
            loadings: Two-dimensional feature-by-dimension loading matrix. Its
                row count must match the normalized feature selection.
            normalized: Normalized artifact associated with the loadings.
            batch_size: Number of selected cells processed per block.
            local_cache: Local staging policy for normalized data on remote
                stores.
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
            dims=int(loading_values.shape[1]),
            pca_cell_selection=None,
            feat_scaling=False,
            lsi_skip_first=False,
            custom_loadings=loading_values,
            rand_state=4466,
            batch_size=batch_size,
            local_cache=local_cache,
            show_elbow_plot=False,
            invalidate_cache=invalidate_cache,
        )

    def run_harmony(
        self,
        reduction: ArtifactRef,
        batch_columns: list[str],
        *,
        harmony_params: dict[str, Any] | None = None,
        batch_size: int | None = None,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Snapshot live batch columns, then fit or reuse Harmony correction."""
        self._resolve_harmony_reduction(reduction)
        if not isinstance(batch_columns, list) or not batch_columns:
            raise ValueError("batch_columns must be a non-empty list")
        if any(not isinstance(column, str) or not column for column in batch_columns):
            raise ValueError("batch_columns must contain non-empty strings")
        if len(set(batch_columns)) != len(batch_columns):
            raise ValueError("batch_columns must be unique")
        batch_snapshot = snapshot_run_metadata(
            self.zw,
            table_path="cellData",
            id_column="ids",
            columns=batch_columns,
            axis="cell",
            invalidate_cache=invalidate_cache,
        )
        return self._run_harmony_artifact(
            reduction,
            batch_snapshot,
            batch_columns,
            harmony_params=harmony_params,
            batch_size=batch_size,
            invalidate_cache=invalidate_cache,
        )

    def _resolve_harmony_reduction(
        self,
        reduction: ArtifactRef,
    ) -> tuple[str, ArtifactRef, ValidatedStoredSelection]:
        if not isinstance(reduction, ArtifactRef):
            raise TypeError("reduction must be an ArtifactRef")
        resolve_coordinate_inputs(self.zw, reduction)
        reduction_ref = reduction
        self._require_complete_artifact(reduction_ref, "reduction")
        if reduction_ref.assay is None:
            raise ValueError("Reduction artifact has no assay")
        normalized_ref = self._artifact_input_ref(
            reduction_ref,
            "normalized",
            "normalized",
        )
        self._require_complete_artifact(normalized_ref, "normalized")
        cell_selection = self._artifact_input_ref(
            normalized_ref,
            "cell_selection",
            "cell_selection",
        )
        validated_cells = validate_stored_selection_integrity(
            self.zw,
            cell_selection,
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        )
        return reduction_ref.assay, cell_selection, validated_cells

    def _run_harmony_artifact(
        self,
        reduction: ArtifactRef,
        batch_snapshot: ArtifactRef,
        batch_columns: list[str],
        *,
        harmony_params: dict[str, Any] | None = None,
        batch_size: int | None = None,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Fit Harmony from an explicit immutable metadata snapshot."""
        if not isinstance(batch_snapshot, ArtifactRef):
            raise TypeError("batch_snapshot must be an ArtifactRef")
        reduction_ref = reduction
        reduction_assay, cell_selection, validated_cells = (
            self._resolve_harmony_reduction(reduction_ref)
        )
        if not isinstance(batch_columns, list) or not batch_columns:
            raise ValueError("batch_columns must be a non-empty list")
        if any(not isinstance(column, str) or not column for column in batch_columns):
            raise ValueError("batch_columns must contain non-empty strings")
        if len(set(batch_columns)) != len(batch_columns):
            raise ValueError("batch_columns must be unique")
        requested_batch_size = (
            None if batch_size is None else _positive_integer(batch_size, "batch_size")
        )
        snapshot = validate_run_metadata_snapshot(
            self.zw,
            batch_snapshot,
            axis="cell",
            assay=None,
            table_path="cellData",
            ordered_columns=None,
        )
        missing_columns = [column for column in batch_columns if column not in snapshot]
        if missing_columns:
            raise ArtifactResolutionError(
                "Harmony batch columns are absent from the metadata snapshot",
                code="snapshot_contract_mismatch",
                context={
                    "artifact_id": batch_snapshot.artifact_id,
                    "missing_columns": ",".join(missing_columns),
                },
            )

        def selected_snapshot_values(column: str) -> np.ndarray:
            values = as_zarr_array(snapshot[column], name=column)
            selected_blocks = tuple(
                np.asarray(block.values)
                for block in iter_selected_axis_selection_blocks(
                    self.zw,
                    cell_selection,
                    values,
                    kind="cell_selection",
                    scope="datastore",
                    assay=None,
                    table_path="cellData",
                    block_rows=requested_batch_size,
                )
            )
            selected = np.concatenate(selected_blocks)
            missing_name = values.attrs.get("missing_mask")
            if isinstance(missing_name, str):
                missing_values = as_zarr_array(
                    snapshot[missing_name], name=missing_name
                )
                missing = np.concatenate(
                    tuple(
                        np.asarray(block.values, dtype=bool)
                        for block in iter_selected_axis_selection_blocks(
                            self.zw,
                            cell_selection,
                            missing_values,
                            kind="cell_selection",
                            scope="datastore",
                            assay=None,
                            table_path="cellData",
                            block_rows=requested_batch_size,
                        )
                    )
                )
                selected = selected.astype(object)
                selected[missing] = None
            return selected.astype(object)

        batches = pd.DataFrame(
            {column: selected_snapshot_values(column) for column in batch_columns}
        )
        source, n_cells, dims = self._coordinate_source(
            reduction_ref,
            batch_size=requested_batch_size,
        )
        if n_cells != validated_cells.selected_count:
            raise ArtifactResolutionError(
                "Reduction rows do not match its cell selection",
                code="row_mismatch",
                context={
                    "assay": reduction_ref.assay,
                    "artifact_id": reduction_ref.artifact_id,
                    "coordinate_rows": n_cells,
                    "selected_count": validated_cells.selected_count,
                },
            )
        source_data = getattr(source, "data", None)
        source_batch_size = (
            int(source_data.chunksize[0]) if source_data is not None else n_cells
        )
        effective_batch_size = min(
            source_batch_size if requested_batch_size is None else requested_batch_size,
            n_cells,
        )
        arguments = HarmonyArguments(
            reduction=reduction_ref,
            batch_snapshot=batch_snapshot,
            batch_columns=tuple(batch_columns),
            harmony_parameters=harmony_params or {},
            algorithm_version="centroid_snapshot_v2",
            batch_size=effective_batch_size,
            invalidate_cache=invalidate_cache,
        )
        planned = self._plan_assay_artifact(
            reduction_assay,
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
                shutdown_checkpoint()
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
        action = "Reused" if planned.reused else "Stored"
        logger.info(
            f"{action} Harmony coordinates for {n_cells} cells with {dims} dimensions"
        )
        return planned.ref

    def _build_embedding_initialization(
        self,
        coordinates: ArtifactRef,
        *,
        n_centroids: int,
        rand_state: int,
        batch_size: int | None,
        invalidate_cache: bool,
        kmeans_sampling: float = 0.1,
        kmeans_batch_size: int = 10_000,
        algorithm_version: str = "minibatch_kmeans_v2",
    ) -> ArtifactRef:
        if coordinates.assay is None:
            raise ValueError("Coordinate artifact has no assay")
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
            coordinates,
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
            coordinates=coordinates,
            n_centroids=effective_clusters,
            rand_state=resolved_rand_state,
            batch_size=effective_batch_size,
            kmeans_sampling=resolved_kmeans_sampling,
            kmeans_batch_size=effective_kmeans_batch_size,
            algorithm_version=algorithm_version,
            invalidate_cache=invalidate_cache,
        )
        planned = self._plan_assay_artifact(
            coordinates.assay,
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
        coordinates: ArtifactRef,
        *,
        n_centroids: int = 1000,
        rand_state: int = 4466,
        batch_size: int | None = None,
        kmeans_sampling: float = 0.1,
        kmeans_batch_size: int = 10_000,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Build or reuse K-means initialization for explicit coordinates.

        Pass the returned reference explicitly to an embedding operation.

        Args:
            coordinates: Reduction or batch-correction artifact to cluster.
            n_centroids: Requested number of K-means centroids.
            rand_state: K-means random seed.
            batch_size: Number of cells processed per block.
            kmeans_sampling: Fraction of cells considered during centroid seeding.
            kmeans_batch_size: Number of cells per internal K-means update.
            invalidate_cache: Force a new initialization artifact.

        Returns:
            Reference to the embedding-initialization artifact.
        """
        if not isinstance(coordinates, ArtifactRef):
            raise TypeError("coordinates must be an ArtifactRef")
        return self._build_embedding_initialization(
            coordinates,
            n_centroids=n_centroids,
            rand_state=rand_state,
            batch_size=batch_size,
            invalidate_cache=invalidate_cache,
            kmeans_sampling=kmeans_sampling,
            kmeans_batch_size=kmeans_batch_size,
        )

    def build_ann_index(
        self,
        coordinates: ArtifactRef,
        *,
        ann_metric: str = "l2",
        ann_efc: int = 50,
        ann_ef: int = 50,
        ann_m: int = 48,
        ann_parallel: bool = False,
        rand_state: int = 4466,
        batch_size: int | None = None,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Build or reuse an approximate nearest-neighbor index."""
        if not isinstance(coordinates, ArtifactRef):
            raise TypeError("coordinates must be an ArtifactRef")
        if coordinates.assay is None:
            raise ValueError("Coordinate artifact has no assay")
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
        action = "Reused" if planned.reused else "Stored"
        logger.info(f"{action} ANN index for {n_cells} cells")
        return planned.ref

    def query_neighbors(
        self,
        ann_index: ArtifactRef,
        *,
        coordinates: ArtifactRef | None = None,
        k: int = 11,
        batch_size: int | None = None,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Query an ANN artifact and persist compact neighbor matrices."""
        if not isinstance(ann_index, ArtifactRef):
            raise TypeError("ann_index must be an ArtifactRef")
        ann_ref = ann_index
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
        action = "Reused" if planned.reused else "Stored"
        logger.info(f"{action} {effective_k} neighbors for each of {n_cells} cells")
        return planned.ref

    def build_connectivity_map(
        self,
        neighbors: ArtifactRef,
        *,
        local_connectivity: float = 1.0,
        bandwidth: float = 1.5,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Convert persisted neighbors into a weighted connectivity graph.

        Args:
            neighbors: Neighbors artifact.
            local_connectivity: UMAP-style local-connectivity adjustment.
            bandwidth: Distance-kernel bandwidth multiplier.
            invalidate_cache: Force a new connectivity artifact.

        Returns:
            Reference to the connectivity-map artifact.
        """
        if not isinstance(neighbors, ArtifactRef):
            raise TypeError("neighbors must be an ArtifactRef")
        neighbors_ref = neighbors
        status = self._require_complete_artifact(
            neighbors_ref,
            "neighbors",
        )
        resolve_native_graph_inputs(self.zw, neighbors_ref)
        if neighbors_ref.assay is None:
            raise ValueError("Neighbors artifact has no assay")
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
        graph: ArtifactRef,
        *,
        symmetric: bool | None = None,
        upper_only: bool | None = None,
        use_k: int | None = None,
    ) -> csr_matrix:
        """Load the cell neighbourhood as a scipy sparse matrix.

        Args:
            graph: Connectivity-map or integrated-graph artifact.
            symmetric: If True, makes the graph symmetric by adding it to its transpose.
            upper_only: If True, then only the values from upper triangular of the matrix are returned. This is only
                       used when symmetric is True.
            use_k: Number of top k-nearest neighbours to keep in the graph. This value must be greater than 0 and less
                   the parameter k used. By default, all neighbours are used. (Default value: None)

        Returns:
            A scipy sparse matrix representing cell neighbourhood graph.
        """

        if not isinstance(graph, ArtifactRef):
            raise TypeError("graph must be an ArtifactRef")
        selection = graph_cell_selection(self.zw, graph)
        validate_stored_selection_integrity(
            self.zw,
            selection,
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        )
        return self._load_graph_artifact(
            graph,
            symmetric=symmetric,
            upper_only=upper_only,
            use_k=use_k,
        )

    def integrate_assays(
        self,
        sources: list[ArtifactRef],
        method: str = "snn",
        chunk_size: int = 10000,
        invalidate_cache: bool = False,
        l2_normalize: bool = True,
    ) -> ArtifactRef:
        """Integrate explicit graph or neighbor artifacts across assays.

        SNN combines shared edge support across two or more assays. WNN accepts
        two or more assays and uses Hao-inspired per-cell modality weights.
        Scarf WNN scores only the union of the existing self-free KNN rows and
        uses the distance span from the nearest to the k-th neighbour as its
        bandwidth, so it is not bit-identical to Seurat's default wider search
        and SNN-far bandwidth.

        Args:
            sources: Connectivity-map refs for SNN or neighbor refs for WNN.
            method: Choose a method for modality integration. Available options: 'snn': Shared nearest neighbour
                    approach and 'wnn': Hao-inspired weighted nearest neighbor integration.
            chunk_size: number of cells to be loaded at a time while reading and writing the graph
            invalidate_cache: Force a new integrated-graph artifact.
            l2_normalize: L2-normalize modality coordinates during WNN scoring.
                This algorithmic setting is stored in artifact provenance.

        Returns:
            Reference to the integrated-graph artifact. Pass it to `run_umap`,
            `run_tsne`, or the clustering methods as their ``graph`` argument.

        WNN modality weights remain in the returned immutable artifact.
        """
        from ...neighbors.graph import merge_graphs
        from ...neighbors.integration import _wnn_integration_many

        sources = list(sources)
        if method not in {"snn", "wnn"}:
            raise ValueError(
                f"Method {method} not supported, choose one of these: 'snn', 'wnn'"
            )
        if len(sources) < 2:
            raise ValueError("Assay integration requires at least two assays")
        if not all(isinstance(source, ArtifactRef) for source in sources):
            raise TypeError("sources must contain only ArtifactRef values")
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
        shared_source_n_cells: int | None = None
        assays: list[str] = []
        expected_kind = "neighbors" if method == "wnn" else "connectivity_map"
        for index, source in enumerate(sources):
            if source.kind != expected_kind:
                raise ArtifactResolutionError(
                    f"{method.upper()} integration requires {expected_kind} artifacts",
                    code="wrong_kind",
                    context={
                        "artifact_id": source.artifact_id,
                        "actual_kind": source.kind,
                        "expected_kind": expected_kind,
                    },
                )
            assay_name = source.assay
            if assay_name is None:
                raise ArtifactResolutionError(
                    "Integration source has no assay",
                    code="wrong_scope",
                    context={"artifact_id": source.artifact_id},
                )
            assays.append(assay_name)
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
            validate_stored_selection_integrity(
                self.zw,
                ancestry.cell_selection,
                kind="cell_selection",
                scope="datastore",
                assay=None,
                table_path="cellData",
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
            if shared_selection is None:
                shared_selection = selection
            elif shared_selection != selection:
                raise ValueError(
                    "Integrated graphs require one exact shared cell selection"
                )
        if shared_selection is None:
            raise ValueError("No assay cell selection was resolved")
        if len(set(assays)) != len(assays):
            raise ValueError("Assay integration requires unique assay sources")
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
            execution_options={"chunk_size": chunk_size},
            invalidate_cache=invalidate_cache,
            required_arrays=tuple(required_arrays),
        )

        if integrated_plan.reused:
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
        return integrated_plan.ref
