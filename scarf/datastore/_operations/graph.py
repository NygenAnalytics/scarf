import math
import os
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import pandas as pd
import zarr
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from ...embeddings.reduction import (
    _gram_pca_dispatch,
    _streaming_lsi_accumulator_bytes,
    require_materialized_lsi_budget,
)
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
)
from ...graph.distances import (
    payload_error,
    validate_distance_provenance,
    validate_integration_source_payload,
)
from ...graph.feature_projection import (
    graph_cell_selection,
    resolve_coordinate_inputs,
    resolve_native_graph_inputs,
)
from ...graph.kinds import require_graph_kind
from ...matrix import ChunkedArray
from ...metadata.rows import apply_missing_mask
from ...neighbors.stages import (
    AnnIndexStage,
    BatchCorrectionStage,
    ChunkedCoordinateStream,
    CoordinateSource,
    KMeansInitializationStage,
    NeighborQueryStage,
    ReductionTransform,
)
from ...storage.ann_index import (
    load_ann_index,
    save_ann_index,
)
from ...storage.arrays import (
    create_numeric_array,
    create_zarr_dataset,
    linked_missing_mask,
)
from ...storage.artifact_writer import (
    ArrayRequirement,
    PlannedArtifact,
    artifact_transaction,
    plan_artifact,
    reused_artifact_group,
)
from ...storage.artifacts import (
    ArtifactRef,
    ArtifactStatus,
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
    ZarrArraySpec,
    _group_zarr_format,
    array_shard_rows,
    iter_shard_row_slices,
    row_sharded_array_spec,
)
from ...storage.profiles import resolve_storage_profile
from ...storage.sharding import write_dense_from_row_batches
from ...storage.stores import is_remote_datastore
from ...storage.selections import (
    iter_selected_axis_selection_blocks,
    read_stored_selection_mask,
    snapshot_run_metadata,
    validate_run_metadata_snapshot,
)
from ...utils.arrays import clean_array
from ...utils.arguments import integer_argument
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
        resolved = min(integer_argument(requested, "batch_size", minimum=1), n_rows)
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


def _reduction_write_bytes(
    spec: ZarrArraySpec,
    data: ChunkedArray,
    resources: ResourceBudget,
    *,
    dims: int,
    transform_bytes: int,
) -> tuple[int, int]:
    """Plan the reduced-coordinate write and return producer and writer bytes.

    Raises MemoryError when the budget cannot hold the streamed input, the
    fitted transform, and one written band together.
    """
    from ...storage.io_policy import StorageIoPolicy
    from ...storage.sharding import plan_dense_write

    producer_bytes = (
        data._resident_bytes()
        + 3 * data._block_task_bytes()
        + data.chunksize[0] * dims * 4
        + transform_bytes
    )
    writer_plan = plan_dense_write(
        spec,
        resources,
        1,
        io=StorageIoPolicy(readWorkers=1),
        residentBytes=producer_bytes,
    )
    return producer_bytes, writer_plan.reservedBytes - producer_bytes


def _coordinate_spec(root: zarr.Group, n_cells: int, dims: int) -> ZarrArraySpec:
    """Return the layout of a reduction's float32 coordinate array."""
    return row_sharded_array_spec(
        (n_cells, dims),
        np.float32,
        profile=resolve_storage_profile(root.store),
        band_rows=min(n_cells, 1_000_000),
        zarr_format=_group_zarr_format(root),
        fill_value=0.0,
    )


def _lsi_write_block_rows(
    array: zarr.Array,
    root: zarr.Group,
    resources: ResourceBudget,
    *,
    dims: int,
    limit: int,
) -> int:
    """Return the most rows, up to ``limit``, per block of the LSI coordinate write.

    The write holds three input blocks at once, so it can need smaller blocks
    than the fit. When no block fits, ``limit`` is returned unchanged and the
    write plan raises MemoryError before the fit.
    """
    spec = _coordinate_spec(root, int(array.shape[0]), dims)
    # The fitted LSI loadings are float64.
    transform_bytes = int(array.shape[1]) * dims * np.dtype(np.float64).itemsize

    def fits(rows: int) -> bool:
        try:
            _reduction_write_bytes(
                spec,
                ChunkedArray(array, block_size=rows),
                resources,
                dims=dims,
                transform_bytes=transform_bytes,
            )
        except MemoryError:
            return False
        return True

    if fits(limit) or not fits(1):
        return limit
    # Larger blocks need more memory, so the largest fitting block is bisected.
    low, high = 1, limit
    while high - low > 1:
        middle = (low + high) // 2
        if fits(middle):
            low = middle
        else:
            high = middle
    return low


def _read_pca_center(group: zarr.Group) -> np.ndarray:
    if "center" not in group:
        raise ValueError("PCA artifact has no fitted center. Re-run run_pca.")
    loadings = as_zarr_array(group["loadings"], name="loadings")
    center = as_zarr_array(group["center"], name="center")
    if (
        loadings.ndim != 2
        or center.shape != (loadings.shape[0],)
        or np.dtype(center.dtype) != np.dtype(np.float64)
    ):
        raise ValueError("PCA center has incompatible shape or dtype. Re-run run_pca.")
    values = np.asarray(center[:])
    if not np.all(np.isfinite(values)):
        raise ValueError("PCA center contains non-finite values. Re-run run_pca.")
    return values


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


def _validated_harmony_request(
    batch_columns: Any,
    harmony_params: dict[str, Any] | None,
    batch_size: int | None,
) -> tuple[dict[str, Any], int | None]:
    """Return resolved Harmony parameters and the optional block size."""
    from ...embeddings.harmony.api import validate_harmony_parameters

    if not isinstance(batch_columns, list) or not batch_columns:
        raise ValueError("batch_columns must be a non-empty list")
    if any(not isinstance(column, str) or not column for column in batch_columns):
        raise ValueError("batch_columns must contain non-empty strings")
    if len(set(batch_columns)) != len(batch_columns):
        raise ValueError("batch_columns must be unique")
    requested_batch_size = (
        None
        if batch_size is None
        else integer_argument(batch_size, "batch_size", minimum=1)
    )
    return validate_harmony_parameters(harmony_params), requested_batch_size


def _requested_block_rows(
    source: CoordinateSource,
    requested: int | None,
    n_cells: int,
) -> int:
    """Return the requested block size, or the coordinate chunk rows, capped."""
    source_data = getattr(source, "data", None)
    source_rows = int(source_data.chunksize[0]) if source_data is not None else n_cells
    return min(source_rows if requested is None else int(requested), n_cells)


class _GraphOperationsMixin(_GraphOperationsBase):
    _normalizedArtifactCache: dict[ArtifactRef, ChunkedArray]
    _artifactExecutionContext: dict[str, Any]

    def _resolve_ann_index(
        self,
        ann_ref: ArtifactRef,
        ann_metric: str,
        dim: int,
        expected_count: int | None = None,
    ) -> Any:
        """Load the Zarr index bytes of a complete ANN artifact the caller resolved."""
        ann_group = artifact_group(self.zw, ann_ref)
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
                context={
                    "assay": ann_ref.assay,
                    "artifact_id": ann_ref.artifact_id,
                    "actual_kind": ann_ref.kind,
                },
            ) from error

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

    def _store_to_sparse(self, graph_loc: str, use_k: int | None) -> csr_matrix:
        """Read a stored graph as CSR, keeping each cell's ``use_k`` nearest edges.

        Callers validate ``use_k`` against the graph's ``k``; None keeps every edge.
        """
        logger.debug(f"Loading graph from location: {graph_loc}")
        store = as_zarr_group(self.zw[graph_loc], name=graph_loc)
        n_cells, k = self._get_graph_ncells_k(graph_loc)
        w = np.asarray(as_zarr_array(store["weights"], name="weights")[:])
        e = np.asarray(as_zarr_array(store["edges"], name="edges")[:])
        if use_k is not None and use_k != k:
            from ...neighbors.graph import take_nearest_per_row

            w, e = take_nearest_per_row(w, e, n_cells, use_k)
        return csr_matrix((w, (e[:, 0], e[:, 1])), shape=(n_cells, n_cells))

    @staticmethod
    def _resolve_local_cache_plan(
        zarr_loc: Any,
        group: zarr.Group,
        local_cache: bool | str,
    ) -> tuple[bool, str | None, bool]:
        """Return the staging flag, cache directory, and delete-after-use flag."""
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
    ) -> ArtifactStatus:
        if ref.kind != kind:
            raise ValueError(f"Expected {kind!r} artifact, got {ref.kind!r}")
        if assay is not None and (ref.scope != "assay" or ref.assay != assay):
            raise ValueError(f"Artifact must belong to assay {assay!r}")
        return require_complete_artifact(self.zw, ref)

    def _artifact_input_ref(
        self,
        ref: ArtifactRef,
        name: str,
        kind: str | None,
    ) -> ArtifactRef:
        """Return the complete ``name`` input of a complete artifact.

        ``kind`` is the required input kind; None accepts the recorded kind.
        """
        status = self._require_complete_artifact(ref, ref.kind)
        input_ref = status.input_ref(name)
        self._require_complete_artifact(
            input_ref,
            input_ref.kind if kind is None else kind,
        )
        return input_ref

    def _load_normalized_artifact(
        self,
        ref: ArtifactRef,
        *,
        batch_size: int,
    ) -> ChunkedArray:
        """Stream normalized data in blocks of an already resolved ``batch_size``."""
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
            block_size=batch_size,
            nthreads=self.nthreads,
            resources=self.resources,
        )

    @contextmanager
    def _cache_normalized_artifact(
        self,
        ref: ArtifactRef,
        local_cache: bool | str,
        batch_size: int,
    ) -> Iterator[None]:
        """Stage remote normalized data locally, read in ``batch_size`` blocks."""
        try:
            already_cached = ref in self._normalizedArtifactCache
        except AttributeError:
            already_cached = False
        if already_cached:
            yield
            return
        status = self._require_complete_artifact(ref, "normalized")
        artifact = group_at(self.zw, status.path)
        # The store that holds the artifact decides, so a mount stages a
        # normalized artifact that it reuses from a remote source.
        enabled, cache_base, remove_after_use = self._resolve_local_cache_plan(
            self.zarr_loc,
            artifact,
            local_cache,
        )
        if not enabled:
            yield
            return
        try:
            cache = self._normalizedArtifactCache
        except AttributeError:
            cache = {}
            self._normalizedArtifactCache = cache
        # A staging copy that fails or is interrupted is removed with the rest
        # of a temporary cache directory.
        try:
            if cache_base is None:
                raise RuntimeError("Local cache path is missing")
            source = as_zarr_array(artifact["data"], name="data")
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
            cache[ref] = ChunkedArray(
                staged,
                block_size=batch_size,
                nthreads=self.nthreads,
                resources=self.resources,
            )
            yield
        finally:
            cache.pop(ref, None)
            if remove_after_use and cache_base is not None:
                import shutil

                shutil.rmtree(cache_base, ignore_errors=True)

    def _coordinate_source(
        self,
        coordinates: ArtifactRef,
        *,
        batch_size: int | None,
    ) -> tuple[CoordinateSource, int, int]:
        lineage = resolve_coordinate_inputs(self.zw, coordinates)
        if lineage.reduction is not None:
            reduction_status = inspect_artifact(self.zw, lineage.reduction)
            if reduction_status.operation == "run_pca":
                _read_pca_center(artifact_group(self.zw, lineage.reduction))
        group = artifact_group(self.zw, coordinates)
        backing = as_zarr_array(group["data"], name="data")
        data = ChunkedArray(
            backing,
            block_size=_row_block(backing, batch_size),
            nthreads=self.nthreads,
            resources=self.resources,
        )
        return (
            ChunkedCoordinateStream(data, self.nthreads),
            int(data.shape[0]),
            int(data.shape[1]),
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
        feature_selection = features
        self._require_complete_artifact(features, "feature_selection", assay=assay_name)
        self._require_complete_artifact(cell_selection, "cell_selection")
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
        method_qualname = str(getattr(normalization_method, "__qualname__", ""))
        dynamic = "<locals>" in method_qualname or "<lambda>" in method_qualname
        if dynamic and getattr(normalization_method, "artifact_identity", None) is None:
            raise ValueError(
                "Dynamic normalization callables must define "
                "artifact_identity for provenance"
            )
        raw_size_factor = getattr(assay, "sf", None)
        size_factor = (
            float(cast(int | float, raw_size_factor))
            if raw_size_factor is not None
            else None
        )
        from ...assay.normalization import load_normalization_selections

        dataset_fingerprint = self._ensure_dataset_fingerprint(assay_name)
        selections = load_normalization_selections(
            self.zw, assay_name, cell_selection, features
        )
        n_cells = selections.cells.selected_count
        n_features = int(selections.featureMask.sum())
        arguments = NormalizationArguments(
            cell_selection=cell_selection,
            feature_selection=feature_selection,
            dataset_fingerprint=dataset_fingerprint,
            normalization_method=normalization_method,
            size_factor=size_factor,
            log_transform=log_transform,
            renormalize_subset=renormalize_subset,
            invalidate_cache=invalidate_cache,
        )

        def valid_shape(_ref: ArtifactRef, group: zarr.Group) -> bool:
            data = as_zarr_array(group["data"], name="data")
            return data.shape == (n_cells, n_features) and all(
                as_zarr_array(group[name], name=name).shape == (data.shape[1],)
                for name in ("feature_sum", "feature_squared_sum")
            )

        planned = self._plan_assay_artifact(
            assay_name,
            arguments,
            required_arrays=(
                ArrayRequirement(
                    "data",
                    shape=(None, None),
                    dtype=np.float32,
                ),
                ArrayRequirement(
                    "feature_sum",
                    shape=(None,),
                    dtype=np.float64,
                ),
                ArrayRequirement(
                    "feature_squared_sum",
                    shape=(None,),
                    dtype=np.float64,
                ),
            ),
            invalidate_cache=invalidate_cache,
            reuse_validator=valid_shape,
        )
        if not planned.reused:
            self._require_writable("run_normalization")
            cell_values = np.asarray(selections.cells.values[:], dtype=bool)
            with artifact_transaction(self.zw, planned):
                relative_path = artifact_path(planned.ref).removeprefix(
                    f"{assay_name}/"
                )
                assay._write_normalized_payload(
                    np.flatnonzero(cell_values),
                    np.flatnonzero(selections.featureMask),
                    relative_path,
                    log_transform=log_transform,
                    renormalize_subset=renormalize_subset,
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
        requested_dims = integer_argument(dims, "dims", minimum=1)
        if batch_size is not None:
            integer_argument(batch_size, "batch_size", minimum=1)
        if not isinstance(normalized, ArtifactRef):
            raise TypeError("normalized must be an ArtifactRef")
        normalized_ref = normalized
        status = self._require_complete_artifact(normalized_ref, "normalized")
        group = group_at(self.zw, status.path)
        data = as_zarr_array(group["data"], name="data")
        effective_batch_size = _row_block(
            data,
            batch_size,
            minimum=(requested_dims + 1 if method == "pca" else None),
        )
        if method == "lsi" and lsi_solver == "streaming":
            fit_rows = _streaming_lsi_block_rows(
                data,
                self.resources,
                n_components=requested_dims + int(lsi_skip_first),
                n_oversamples=lsi_n_oversamples,
            )
            memory_limited_rows = _lsi_write_block_rows(
                data,
                self.zw,
                self.resources,
                dims=requested_dims,
                limit=min(effective_batch_size, fit_rows),
            )
            if effective_batch_size > memory_limited_rows:
                logger.warning(
                    f"Reducing LSI batch_size from {effective_batch_size} to "
                    f"{memory_limited_rows} rows to honor the memory budget"
                )
                effective_batch_size = memory_limited_rows
        with self._artifact_execution_context({"local_cache": local_cache}):
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
                local_cache=local_cache,
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
        local_cache: bool | str = "auto",
        lsi_solver: Literal["streaming", "materialized"] = "streaming",
        lsi_n_iter: int = 5,
        lsi_n_oversamples: int = 10,
    ) -> ArtifactRef:
        normalized_ref = normalized
        from ...assay.normalization import load_normalized_inputs

        if normalized_ref.assay is None:
            raise ValueError("Normalized artifact has no assay")
        assay_name = normalized_ref.assay
        self._ensure_dataset_fingerprint(assay_name)
        data_group, selections = load_normalized_inputs(self.zw, normalized_ref)
        normalized_cell_selection = selections.cells.ref
        data_array = as_zarr_array(data_group["data"], name="data")
        n_cells, n_features = map(int, data_array.shape)
        effective_batch_size = min(
            integer_argument(batch_size, "batch_size", minimum=1),
            n_cells,
        )
        effective_dims = integer_argument(dims, "dims", minimum=1)
        if custom_loadings is not None:
            if custom_loadings.shape[0] != n_features:
                raise ValueError("Custom loadings rows must match normalized features")
            effective_dims = int(custom_loadings.shape[1])
        pca_selection = pca_cell_selection or normalized_cell_selection
        pca_use_values: np.ndarray | None = None
        if method == "pca":
            normalized_mask = np.asarray(selections.cells.values[:], dtype=bool)
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
        if method == "pca":
            assert pca_selection is not None
            n_blocks = -(-n_cells // effective_batch_size)
            use_gram, _reason = _gram_pca_dispatch(
                n_features,
                effective_batch_size,
                n_blocks,
            )
            arguments: Any = PcaArguments(
                normalized=normalized_ref,
                feature_scaling=scaling_plan.ref,
                pca_cell_selection=pca_selection,
                dims=effective_dims,
                feat_scaling=feat_scaling,
                # IncrementalPCA over several blocks depends on the block size.
                incremental_block_rows=(
                    effective_batch_size if n_blocks > 1 and not use_gram else None
                ),
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
            # Only run_custom_reduction selects this method, with loadings.
            assert custom_loadings is not None
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
        if method == "pca":
            required_arrays += (
                ArrayRequirement("center", shape=(n_features,), dtype=np.float64),
            )
        planned = self._plan_assay_artifact(
            assay_name,
            arguments,
            required_arrays=required_arrays,
            invalidate_cache=invalidate_cache,
        )
        if planned.reused:
            if show_elbow_plot and method == "pca":
                logger.warning("PCA was not fitted so no elbow plot is available")
            logger.info(
                f"Reused {method.upper()} reduction for {n_cells} cells "
                f"with {effective_dims} dimensions"
            )
            return planned.ref
        self._require_writable(
            {"pca": "run_pca", "lsi": "run_lsi"}.get(method, "run_custom_reduction")
        )
        if method == "lsi" and lsi_solver == "materialized":
            require_materialized_lsi_budget(
                n_rows=n_cells,
                n_features=n_features,
                itemsize=int(np.dtype(data_array.dtype).itemsize),
                n_components=effective_dims + int(lsi_skip_first),
                n_oversamples=lsi_n_oversamples,
                memory_bytes=self.resources.memoryBytes,
            )

        with self._cache_normalized_artifact(
            normalized_ref, local_cache, effective_batch_size
        ):
            normalized_data = self._load_normalized_artifact(
                normalized_ref,
                batch_size=effective_batch_size,
            )
            score_spec = _coordinate_spec(self.zw, n_cells, effective_dims)
            # Plan the coordinate write against an upper bound on the fitted
            # transform, so a budget that cannot hold it fails before the fit.
            float_bytes = np.dtype(np.float64).itemsize
            producer_bytes, write_bytes = _reduction_write_bytes(
                score_spec,
                normalized_data,
                self.resources,
                dims=effective_dims,
                transform_bytes=(
                    custom_loadings.nbytes
                    if custom_loadings is not None
                    else n_features * effective_dims * float_bytes
                )
                + (2 * n_features * float_bytes if enabled_scaling else 0)
                + (n_features * float_bytes if method == "pca" else 0),
            )
            if scaling_plan.reused:
                scaling_group = reused_artifact_group(
                    self.zw,
                    scaling_plan,
                )
                mu = np.asarray(as_zarr_array(scaling_group["mean"], name="mean")[:])
                sigma = np.asarray(
                    as_zarr_array(scaling_group["scale"], name="scale")[:]
                )
            else:
                if enabled_scaling:
                    if (
                        "feature_sum" in data_group
                        and "feature_squared_sum" in data_group
                    ):
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
                with artifact_transaction(self.zw, scaling_plan) as scaling_group:
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
            loadings = transform.loadings
            if loadings is None or loadings.shape != (n_features, effective_dims):
                raise ValueError(
                    f"{method.upper()} loadings have shape "
                    f"{None if loadings is None else loadings.shape}; expected "
                    f"{(n_features, effective_dims)}"
                )
            with artifact_transaction(self.zw, planned) as reduction_group:
                if method == "pca":
                    assert transform.center is not None
                    center_array = create_zarr_dataset(
                        reduction_group,
                        "center",
                        (n_features,),
                        "f8",
                        (n_features,),
                    )
                    center_array[:] = transform.center
                output = create_zarr_dataset(
                    reduction_group,
                    "loadings",
                    normalized_data.chunksize,
                    "f8",
                    loadings.shape,
                )
                output[:, :] = loadings
                scores = create_numeric_array(
                    reduction_group,
                    "data",
                    score_spec,
                )

                def score_blocks() -> Iterator[np.ndarray]:
                    for block in normalized_data._stream_blocks(
                        nthreads=self.nthreads,
                        msg="Calculating reduced coordinates",
                        prefetch=1,
                        row_mask=None,
                        resident_bytes=write_bytes
                        + producer_bytes
                        - normalized_data._block_task_bytes(),
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
                    producerReserveBytes=producer_bytes,
                )
        if show_elbow_plot and method == "pca":
            from ...plotting import elbow

            # A reused fit returned above, so PCA was fitted here.
            assert transform.pca is not None
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
                expanded to that aligned minimum with a warning. Several
                blocks narrower than the selected features, or more than 4096
                features, are fitted with IncrementalPCA, whose result depends
                on the block size, so the block size then joins the artifact
                identity.
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
            rand_state: Non-negative integer seed used by the randomized
                decomposition.
            solver: Memory-bounded streaming solver or materialized compatibility
                solver. The materialized solver holds the whole matrix and raises
                MemoryError before reading it when that exceeds the memory budget.
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
        n_iter = integer_argument(n_iter, "n_iter", minimum=0)
        n_oversamples = integer_argument(n_oversamples, "n_oversamples", minimum=0)
        if not isinstance(skip_first, bool | np.bool_):
            raise TypeError("skip_first must be a boolean")
        rand_state = integer_argument(rand_state, "rand_state", minimum=0)
        return self._run_reduction_artifact(
            method="lsi",
            normalized=normalized,
            dims=dims,
            pca_cell_selection=None,
            feat_scaling=False,
            lsi_skip_first=bool(skip_first),
            custom_loadings=None,
            rand_state=rand_state,
            batch_size=batch_size,
            local_cache=local_cache,
            show_elbow_plot=False,
            invalidate_cache=invalidate_cache,
            lsi_solver=solver,
            lsi_n_iter=n_iter,
            lsi_n_oversamples=n_oversamples,
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
            loadings: Two-dimensional feature-by-dimension matrix of finite real
                loadings. Its row count must match the normalized feature
                selection.
            normalized: Normalized artifact associated with the loadings.
            batch_size: Number of selected cells processed per block.
            local_cache: Local staging policy for normalized data on remote
                stores.
            invalidate_cache: Force a new reduction artifact.

        Returns:
            Reference to the custom reduction artifact.
        """
        loading_values = np.asarray(loadings)
        if loading_values.dtype.kind not in "iuf":
            raise TypeError("Custom loadings must contain real numbers")
        if loading_values.ndim != 2 or loading_values.shape[1] < 1:
            raise ValueError(
                "Custom loadings must be a two-dimensional matrix with columns"
            )
        if not np.all(np.isfinite(loading_values)):
            raise ValueError("Custom loadings must contain only finite values")
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
        # Validate before the snapshot is written; the fit validates again
        # because the pipeline calls it with its own snapshot.
        self._resolve_harmony_reduction(reduction)
        _validated_harmony_request(batch_columns, harmony_params, batch_size)
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
    ) -> tuple[str, ArtifactRef]:
        if not isinstance(reduction, ArtifactRef):
            raise TypeError("reduction must be an ArtifactRef")
        # This validates the reduction and the selections of its normalized
        # input.
        resolve_coordinate_inputs(self.zw, reduction)
        reduction_ref = reduction
        self._require_complete_artifact(reduction_ref, "reduction")
        # resolve_coordinate_inputs rejected coordinates without an assay.
        assert reduction_ref.assay is not None
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
        return reduction_ref.assay, cell_selection

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
        reduction_ref = reduction
        reduction_assay, cell_selection = self._resolve_harmony_reduction(reduction_ref)
        resolved_harmony_params, requested_batch_size = _validated_harmony_request(
            batch_columns,
            harmony_params,
            batch_size,
        )
        snapshot = validate_run_metadata_snapshot(
            self.zw,
            batch_snapshot,
            axis="cell",
            assay=None,
            table_path="cellData",
            ordered_columns=None,
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
            missing_values = linked_missing_mask(snapshot, column, values=values)
            if missing_values is not None:
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
                selected = apply_missing_mask(selected, missing, labels=True)
            return selected.astype(object)

        batches = pd.DataFrame(
            {column: selected_snapshot_values(column) for column in batch_columns}
        )
        # The coordinate source has one row per selected cell.
        source, n_cells, dims = self._coordinate_source(
            reduction_ref,
            batch_size=requested_batch_size,
        )
        effective_batch_size = _requested_block_rows(
            source,
            requested_batch_size,
            n_cells,
        )
        arguments = HarmonyArguments(
            reduction=reduction_ref,
            batch_snapshot=batch_snapshot,
            batch_columns=tuple(batch_columns),
            harmony_parameters=resolved_harmony_params,
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
            self._require_writable("run_harmony")
            correction = BatchCorrectionStage(
                stream=source,
                n_cells=n_cells,
                dims=dims,
                batch_size=effective_batch_size,
                batches=batches,
                parameters=resolved_harmony_params,
                corrected_data=None,
                nthreads=self.nthreads,
            )
            corrected = correction.ensure_corrected()
            result = correction.result
            # Without corrected data, ensure_corrected fits Harmony.
            assert result is not None
            from ...mapping.symphony import weighted_centroids

            cluster_mass, raw_centroids = weighted_centroids(
                result.original.T,
                result.assignments,
            )
            _, corrected_centroids = weighted_centroids(
                result.corrected.T,
                result.assignments,
            )
            with artifact_transaction(self.zw, planned) as group:
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
        algorithm_version: str = "minibatch_kmeans_v3",
    ) -> ArtifactRef:
        if coordinates.assay is None:
            raise ValueError("Coordinate artifact has no assay")
        resolved_batch_size = (
            None
            if batch_size is None
            else integer_argument(batch_size, "batch_size", minimum=1)
        )
        requested_clusters = integer_argument(n_centroids, "n_centroids", minimum=1)
        resolved_rand_state = integer_argument(rand_state, "rand_state", minimum=1)
        resolved_kmeans_sampling = _sampling_fraction(
            kmeans_sampling,
            "kmeans_sampling",
        )
        requested_kmeans_batch_size = integer_argument(
            kmeans_batch_size, "kmeans_batch_size", minimum=1
        )
        stream, n_cells, coordinate_dims = self._coordinate_source(
            coordinates,
            batch_size=resolved_batch_size,
        )
        effective_batch_size = _requested_block_rows(
            stream,
            resolved_batch_size,
            n_cells,
        )
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
            self._require_writable("build_embedding_initialization")
            initialization = KMeansInitializationStage.fit(
                stream=stream,
                n_rows=n_cells,
                batch_size=effective_batch_size,
                n_clusters=effective_clusters,
                rand_state=resolved_rand_state,
                nthreads=self.nthreads,
                kmeans_sampling=resolved_kmeans_sampling,
                kmeans_batch_size=effective_kmeans_batch_size,
            )
            with artifact_transaction(self.zw, planned) as group:
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
        resolved_ann_efc = integer_argument(ann_efc, "ann_efc", minimum=1)
        resolved_ann_ef = integer_argument(ann_ef, "ann_ef", minimum=1)
        resolved_ann_m = integer_argument(ann_m, "ann_m", minimum=1)
        resolved_rand_state = integer_argument(rand_state, "rand_state", minimum=1)
        if resolved_ann_m < 2:
            raise ValueError("ann_m must be at least two")
        if not isinstance(ann_parallel, bool):
            raise TypeError("ann_parallel must be a boolean")
        resolved_batch_size = (
            None
            if batch_size is None
            else integer_argument(batch_size, "batch_size", minimum=1)
        )
        if coordinates.kind not in {
            "reduction",
            "batch_correction",
            "imported_coordinates",
        }:
            raise ValueError(
                "Coordinates must reference reduction, batch_correction, or imported_coordinates"
            )
        self._ensure_dataset_fingerprint(coordinates.assay)
        coordinate_source, n_cells, dims = self._coordinate_source(
            coordinates,
            batch_size=resolved_batch_size,
        )
        effective_batch_size = _requested_block_rows(
            coordinate_source,
            resolved_batch_size,
            n_cells,
        )
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
            from ...storage.ann_index import validate_ann_index_contract

            try:
                validate_ann_index_contract(
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
            self._require_writable("build_ann_index")
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
            with artifact_transaction(self.zw, planned) as group:
                save_ann_index(
                    group,
                    ann_idx,
                    profile=self.storageProfile,
                    metric=ann_metric,
                    dimensions=dims,
                    element_count=n_cells,
                )
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
        # _coordinate_source below validates the coordinate kind and lineage.
        stored_coordinates = self._artifact_input_ref(ann_ref, "coordinates", None)
        if coordinates is not None and coordinates != stored_coordinates:
            raise ValueError("coordinates do not match the ANN artifact input")
        requested_k = integer_argument(k, "k", minimum=1)
        resolved_batch_size = (
            None
            if batch_size is None
            else integer_argument(batch_size, "batch_size", minimum=1)
        )
        coordinate_source, n_cells, dims = self._coordinate_source(
            stored_coordinates,
            batch_size=resolved_batch_size,
        )
        if n_cells < 2:
            raise ValueError("Neighbor queries require at least two cells")
        effective_k = min(requested_k, n_cells - 1)
        # Stored graph payloads must record n_cells within the uint32 range.
        if n_cells > np.iinfo(np.uint32).max:
            raise ValueError("Neighbor indices require fewer than 2**32 cells")
        effective_batch_size = _requested_block_rows(
            coordinate_source,
            resolved_batch_size,
            n_cells,
        )
        ann_parameters = ann_status.parameters or {}
        ann_metric = ann_parameters.get("ann_metric")
        if ann_metric not in {"l2", "cosine"}:
            raise ValueError("ANN artifact has no supported distance metric")
        ann_ef = ann_parameters.get("ann_ef")
        if isinstance(ann_ef, bool) or not isinstance(ann_ef, int) or ann_ef < 1:
            raise ValueError(
                "ANN artifact has no valid ann_ef search depth. Re-run build_ann_index."
            )
        if "parallel_threads" not in ann_parameters:
            raise ValueError(
                "ANN artifact has no parallel_threads record. Re-run build_ann_index."
            )
        from ...storage.ann_index import validate_ann_index_contract

        validate_ann_index_contract(
            artifact_group(self.zw, ann_ref),
            str(ann_metric),
            dims,
            expected_count=n_cells,
        )
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
            self._require_writable("query_neighbors")
            ann_idx = self._resolve_ann_index(
                ann_ref,
                str(ann_metric),
                dims,
                expected_count=n_cells,
            )
            ann_idx = AnnIndexStage.configure(
                ann_idx,
                ef=ann_ef,
                threads=int(ann_parameters["parallel_threads"] or 1),
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
            with artifact_transaction(self.zw, planned) as group:
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
        # resolve_native_graph_inputs rejected neighbors without an assay.
        assert neighbors_ref.assay is not None
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
                    shape=(n_cells * n_neighbors, 2),
                    dtype=np.uint32,
                ),
                ArrayRequirement(
                    "weights",
                    shape=(n_cells * n_neighbors,),
                    dtype=np.float32,
                ),
            ),
            invalidate_cache=invalidate_cache,
        )
        if not planned.reused:
            from ...neighbors.graph import build_connectivity_arrays

            self._require_writable("build_connectivity_map")
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
            with artifact_transaction(self.zw, planned) as output:
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
        action = "Reused" if planned.reused else "Stored"
        logger.info(f"{action} connectivity map for {n_cells} cells")
        return planned.ref

    def _graph_location(self, graph: ArtifactRef) -> str:
        """Return a complete graph's location."""
        require_graph_kind(graph)
        return require_complete_artifact(self.zw, graph).path

    def _load_graph_artifact(
        self,
        graph: ArtifactRef,
        *,
        symmetric: bool | None,
        upper_only: bool | None,
        use_k: int | None,
    ) -> csr_matrix:
        """Load one already captured and validated graph reference."""
        from scipy.sparse import triu

        matrix = self._store_to_sparse(self._graph_location(graph), use_k)
        if symmetric:
            # Fuzzy union of the directed edge weights.
            matrix = (matrix + matrix.T) - matrix.multiply(matrix.T)
            if upper_only:
                matrix = triu(matrix).tocsr()
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
            use_k: Number of top k-nearest neighbours to keep in the graph. It must be an integer from 1 to the
                   graph's k. By default, all neighbours are used. (Default value: None)

        Returns:
            A scipy sparse matrix representing cell neighbourhood graph.
        """

        if not isinstance(graph, ArtifactRef):
            raise TypeError("graph must be an ArtifactRef")
        for name, flag in (("symmetric", symmetric), ("upper_only", upper_only)):
            if flag is not None and not isinstance(flag, bool | np.bool_):
                raise TypeError(f"{name} must be a boolean or None")
        if use_k is not None and (
            isinstance(use_k, bool) or not isinstance(use_k, int | np.integer)
        ):
            raise TypeError("use_k must be an integer or None")
        graph_cell_selection(self.zw, graph)
        if use_k is not None:
            k = self._get_graph_ncells_k(self._graph_location(graph))[1]
            if not 1 <= use_k <= k:
                raise ValueError(
                    f"use_k must be between 1 and the graph's k ({k}), "
                    "or None to use every neighbour"
                )
        return self._load_graph_artifact(
            graph,
            symmetric=bool(symmetric),
            upper_only=bool(upper_only),
            use_k=use_k,
        )

    def integrate_assays(
        self,
        sources: list[ArtifactRef],
        method: str = "wnn",
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
            method: Choose a method for modality integration. Available options:
                'wnn': Hao-inspired weighted nearest neighbor integration and
                'snn': shared nearest neighbour integration.
            chunk_size: Number of cells per stored chunk of the integrated edge,
                weight, and modality-weight arrays. It does not bound memory:
                integration holds every source graph, and for WNN every
                source's coordinates, in memory.
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
        chunk_size = integer_argument(chunk_size, "chunk_size", minimum=1)
        if len(sources) < 2:
            raise ValueError("Assay integration requires at least two assays")
        if not all(isinstance(source, ArtifactRef) for source in sources):
            raise TypeError("sources must contain only ArtifactRef values")
        if method == "wnn" and not isinstance(l2_normalize, bool | np.bool_):
            raise TypeError("l2_normalize must be a boolean")

        def materialize_coordinate_blocks(
            blocks: Iterator[np.ndarray],
            n_cells: int,
        ) -> np.ndarray:
            # The blocks are row bands of one stored array, so they share its
            # width and never exceed its rows.
            coordinates: np.ndarray | None = None
            start = 0
            for values in blocks:
                block = np.asarray(values)
                if block.ndim != 2:
                    raise ValueError("WNN coordinate blocks must be matrices")
                if coordinates is None:
                    coordinates = np.empty(
                        (n_cells, block.shape[1]),
                        dtype=block.dtype,
                    )
                stop = start + len(block)
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
            source_n_cells = validate_integration_source_payload(self.zw, source)
            if method == "wnn" and ancestry.reduction is not None:
                reduction_status = inspect_artifact(self.zw, ancestry.reduction)
                if reduction_status.operation == "run_pca":
                    _read_pca_center(artifact_group(self.zw, ancestry.reduction))
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
            # Sources over one selection cover the same cells, so differing
            # counts mean a payload disagrees with that selection.
            if shared_source_n_cells is None:
                shared_source_n_cells = source_n_cells
            elif source_n_cells != shared_source_n_cells:
                raise payload_error(
                    source,
                    "Integration sources contain different cell counts",
                )
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
        self._require_writable("integrate_assays")

        def load_wnn_inputs(
            index: int,
            assay_name: str,
        ) -> tuple[np.ndarray, NDArray[Any]]:
            neighbors = captured_sources[index]
            coordinates_ref = captured_coordinates[index]
            # Every WNN source is a neighbors artifact with captured coordinates.
            assert coordinates_ref is not None
            neighbors_group = artifact_group(self.zw, neighbors)
            indices = np.asarray(
                as_zarr_array(
                    neighbors_group["indices"],
                    name="indices",
                )[:]
            )
            coordinate_source, n_cells, _ = self._coordinate_source(
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
        else:
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

        with artifact_transaction(self.zw, integrated_plan) as store:
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
        return integrated_plan.ref
