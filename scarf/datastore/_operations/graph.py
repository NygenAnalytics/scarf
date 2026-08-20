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
from ...graph.encoded_paths import (
    is_integrated_graph_path,
    lookup_latest_assay_graph,
    lookup_latest_nearest_neighbor_paths,
    lookup_stored_integrated_graph,
    make_integrated_graph_path,
    make_normalized_group_path,
    nearest_neighbor_paths_from_loc,
    nearest_neighbors_group_path_from_cell_graph,
    parse_assay_graph_paths,
    parse_nearest_neighbors_group_path,
    parse_neighbor_index_group_path,
    parse_reduction_group_path,
)
from ...graph.distances import validate_distance_provenance
from ...graph.paths import StoredAssayGraph, StoredGraph
from ...graph.state import (
    AssayState,
    named_result_mismatch,
    normalized_path_from_state,
    read_assay_state,
    stored_assay_graph_from_ref,
    stored_assay_graph_from_state,
    validate_artifact_graph_selection,
    validate_cell_selection_artifact,
    validate_imported_coordinates_artifact,
    validate_legacy_graph_selection,
    validate_neighbors_artifact_selection,
    validate_normalized_artifact_selection,
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
    legacy_ann_index_path,
    load_ann_index,
    load_ann_index_from_path,
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
    fingerprint_array,
    fingerprint_stored_arrays,
    fingerprint_strings,
    group_at,
    inspect_artifact,
    parse_artifact_path,
    require_complete_artifact,
    serialize_artifact_value,
)
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
            logger.debug("Using the default assay for the KNN graph")
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

    def _resolve_ann_index(
        self,
        ann_loc: str,
        ann_metric: str,
        dim: int,
        expected_count: int | None = None,
    ) -> Any:
        """Load an ANN index from Zarr or a legacy file."""
        ann_group: zarr.Group | None = (
            as_zarr_group(self.zw[ann_loc], name=ann_loc)
            if ann_loc in self.zw
            else None
        )

        if ann_group is not None and has_ann_index(ann_group):
            return load_ann_index(
                ann_group,
                ann_metric,
                dim,
                expected_count=expected_count,
            )

        legacy = legacy_ann_index_path(zarr_root_path(self.zw), ann_loc)
        if legacy is not None and os.path.exists(legacy):
            return load_ann_index_from_path(
                legacy,
                ann_metric,
                dim,
                expected_count=expected_count,
            )

        logger.debug(
            "ANN index not found in store; will rebuild from normalized data and loadings"
        )
        return None

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
            artifact_path(ann_ref),
            ann_metric,
            dims if dims > 0 else data.shape[1],
            expected_count=int(data.shape[0]),
        )
        rebuilt_ann = ann_idx is None
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
        assert neighbors_ref is not None
        self._remember_ann_stream_neighbors(
            ann_obj,
            artifact_path(neighbors_ref),
        )
        if rebuilt_ann and self.zarr_mode == "r+":
            self._persist_ann_index(
                artifact_path(ann_ref),
                ann_obj.annIdx,
                ann_metric=ann_metric,
                dimensions=dims if dims > 0 else data.shape[1],
                element_count=int(data.shape[0]),
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
            resources=self.resources,
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
            harmonized_data = ChunkedArray(
                harmonized_arr,
                nthreads=self.nthreads,
                resources=self.resources,
            )
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
            expected_count=int(data.shape[0]),
        )
        rebuilt_ann = ann_idx is None

        use_for_pca = self.cells.fetch(pca_cell_key, key=cell_key)
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
        )
        self._remember_ann_stream_path(ann_obj, ann_loc)
        self._remember_ann_stream_neighbors(ann_obj, knn_loc)
        if rebuilt_ann and self.zarr_mode == "r+":
            self._persist_ann_index(
                ann_loc,
                ann_obj.annIdx,
                ann_metric=ann_metric,
                dimensions=int(temp_dim),
                element_count=int(data.shape[0]),
            )
        if rebuilt_ann:
            logger.info(f"Built ANN index for {data.shape[0]} cells")
        else:
            logger.info(f"Reused stored ANN index for {data.shape[0]} cells")
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
            raise ValueError(
                "Imported coordinates are not part of the normalized AssayState "
                "graph chain; pass update_state=False"
            )
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
            raise ValueError(f"Cannot select graph state from {terminal.kind!r}")
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
            selection = self._artifact_input_ref(
                state.normalized,
                "cell_selection",
                "cell_selection",
                require_input_complete=False,
            )
            validate_cell_selection_artifact(self.zw, selection, state.cell_key)
            return selection
        if graph_ref.kind == "integrated_graph":
            status = self._require_complete_artifact(
                graph_ref,
                "integrated_graph",
            )
            raw_selection = (status.inputs or {}).get("cell_selection")
            if not isinstance(raw_selection, dict):
                raise ValueError("Integrated graph has no shared cell selection")
            selection = ArtifactRef.from_dict(raw_selection)
            selection_status = inspect_artifact(self.zw, selection)
            source_column = (selection_status.execution_options or {}).get(
                "source_column"
            )
            if not isinstance(source_column, str):
                if not selection_status.exists or not selection_status.complete:
                    validate_cell_selection_artifact(self.zw, selection, "")
                raise ValueError("Integrated graph cell selection key is unavailable")
            validate_cell_selection_artifact(
                self.zw,
                selection,
                source_column,
            )
            return selection
        raise ValueError("Graph ref must be connectivity_map or integrated_graph")

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
        feat_key: str | None = None,
        *,
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
            feat_key: Boolean feature metadata key selecting columns. For a
                non-``I`` key, the stored feature column is
                ``{cell_key}__{feat_key}``.
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
            KeyError: If the feature selection does not exist.
            TypeError: If a selection is not boolean.
            ValueError: If the assay, feature key, or selected data is invalid.
        """
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
                cell_key,
                feat_key,
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
        dims: int = 25,
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
            dims: Requested number of principal components.
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
            state = read_assay_state(self.zw, reduction.assay)
            if state is not None and state.reduction == reduction:
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
        if coordinates.kind == "imported_coordinates" and update_state:
            raise ValueError(
                "Imported coordinates cannot activate AssayState; "
                "pass update_state=False"
            )
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
        k: int = 17,
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
        if stored_coordinates.kind == "imported_coordinates" and update_state:
            raise ValueError(
                "Neighbors from imported coordinates cannot activate AssayState; "
                "pass update_state=False"
            )
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
                ann_status.path,
                str(ann_metric),
                dims,
                expected_count=n_cells,
            )
            if ann_idx is None:
                raise RuntimeError("ANN artifact has no readable index")
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
        group = group_at(self.zw, status.path)
        indices = as_zarr_array(group["indices"], name="indices")
        n_cells, n_neighbors = map(int, indices.shape)
        validate_distance_provenance(self.zw, status.path)
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
                       obtained from `get_latest_graph_loc` method.

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
                f"Build graph artifacts for assay {from_assay}"
            )
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
        n_cells, graph = self._store_to_sparse(graph_loc, "csr", use_k)
        if symmetric is True:
            graph = symmetrize(graph)
            if upper_only is True:
                graph = triu(graph)
        if cache is not None:
            with self._graphMemoryCacheLock:
                active_cache = getattr(self, "_graphMemoryCache", None)
                if active_cache is cache:
                    graph = active_cache.setdefault(cache_key, graph)
        return graph

    def integrate_assays(
        self,
        assays: list[str],
        label: str,
        method: str = "snn",
        chunk_size: int = 10000,
        invalidate_cache: bool = False,
        l2_normalize: bool = True,
    ) -> ArtifactRef:
        """Integrate the latest neighbourhood graphs for selected assays.

        SNN combines shared edge support across two or more assays. WNN accepts
        two or more assays and uses Hao-inspired per-cell modality weights.
        Scarf WNN scores only the union of the existing self-free KNN rows and
        uses the distance span from the nearest to the k-th neighbour as its
        bandwidth, so it is not bit-identical to Seurat's default wider search
        and SNN-far bandwidth.

        Args:
            assays: Name of the input assays. The latest constructed graph from each assay is used.
            label: Label for integrated graph
            method: Choose a method for modality integration. Available options: 'snn': Shared nearest neighbour
                    approach and 'wnn': Hao-inspired weighted nearest neighbor integration.
            chunk_size: number of cells to be loaded at a time while reading and writing the graph
            invalidate_cache: Force a new integrated-graph artifact.
            l2_normalize: L2-normalize modality coordinates during WNN scoring.
                This algorithmic setting is stored in artifact provenance.

        Returns:
            Reference to the integrated-graph artifact. Pass it to `run_umap`,
            `run_tsne`, or the clustering methods as their ``graph`` argument,
            or keep using ``integrated_graph=label``.

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
        if method == "wnn" and len(assays) < 2:
            raise ValueError("WNN integration requires at least two assays")
        if method == "wnn" and len(set(assays)) != len(assays):
            raise ValueError("WNN integration requires unique assay names")
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

        def neighbor_coordinates_ref(neighbors: ArtifactRef) -> ArtifactRef:
            raw_coordinates = (inspect_artifact(self.zw, neighbors).inputs or {}).get(
                "coordinates"
            )
            if not isinstance(raw_coordinates, dict):
                raise ValueError("Neighbors artifact has no coordinates input")
            coordinates = ArtifactRef.from_dict(raw_coordinates)
            if coordinates.kind not in {"reduction", "batch_correction"}:
                raise ValueError(
                    "Neighbor coordinates must be reduction or batch_correction"
                )
            self._require_complete_artifact(coordinates, coordinates.kind)
            return coordinates

        source_inputs: dict[str, Any] = {}
        legacy_wnn_coordinates: dict[str, np.ndarray] = {}
        legacy_wnn_neighbor_paths: dict[str, str] = {}
        shared_selection: ArtifactRef | None = None
        shared_cell_key: str | None = None
        for assay_name in assays:
            if assay_name not in self.assay_names:
                raise ValueError(f"ERROR: Assay {assay_name} was not found.")
            state = read_assay_state(self.zw, assay_name)
            artifact_source = state is not None and (
                (method == "wnn" and state.neighbors is not None)
                or (method == "snn" and state.connectivity_map is not None)
            )
            if artifact_source:
                assert state is not None
                if state.normalized is None:
                    raise ValueError(
                        f"Assay {assay_name!r} has no normalized graph input"
                    )
                if method == "wnn":
                    assert state.neighbors is not None
                    self._require_complete_artifact(state.neighbors, "neighbors")
                    validate_neighbors_artifact_selection(
                        self.zw,
                        state.neighbors,
                        state.cell_key,
                        state.feat_key,
                    )
                else:
                    assert state.connectivity_map is not None
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
                    assert state.neighbors is not None
                    coordinates_ref = neighbor_coordinates_ref(state.neighbors)
                    source_inputs[assay_name] = {
                        "neighbors": state.neighbors,
                        "coordinates": coordinates_ref,
                    }
                else:
                    assert state.connectivity_map is not None
                    source_inputs[assay_name] = state.connectivity_map
                cell_key = state.cell_key
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
                cell_key = self._get_latest_cell_key(assay_name)
                selection = self._ensure_cell_selection(cell_key)
                if method == "wnn":
                    neighbors_path = nearest_neighbors_group_path_from_cell_graph(
                        legacy_graph_path
                    )
                    neighbors_group = as_zarr_group(
                        self.zw[neighbors_path],
                        name=neighbors_path,
                    )
                    legacy_wnn_neighbor_paths[assay_name] = neighbors_path
                    legacy_input: dict[str, Any] = {
                        "legacy_graph_fingerprint": fingerprint_stored_arrays(
                            neighbors_group,
                            ("indices",),
                        ),
                    }
                    feat_key = self._get_latest_feat_key(assay_name)
                    ann = self._load_ann_stream(
                        assay_name,
                        cell_key,
                        feat_key,
                    )
                    if ann.harmonizedData is not None:
                        coordinates = materialize_coordinate_blocks(
                            (
                                np.asarray(block.compute())
                                for block in ann.harmonizedData.blocks
                            ),
                            ann.nCells,
                        )
                    else:
                        coordinates = materialize_coordinate_blocks(
                            (
                                ann.reducer(block)
                                for block in ann.iter_blocks(
                                    f"Loading {assay_name} coordinates"
                                )
                            ),
                            ann.nCells,
                        )
                    legacy_wnn_coordinates[assay_name] = coordinates
                    legacy_input["legacy_coordinates_fingerprint"] = fingerprint_array(
                        coordinates
                    )
                else:
                    legacy_input = {
                        "legacy_graph_fingerprint": fingerprint_stored_arrays(
                            legacy_graph,
                            ("edges", "weights"),
                        ),
                    }
                source_inputs[assay_name] = legacy_input
            if shared_selection is None:
                shared_selection = selection
                shared_cell_key = cell_key
            elif not self._selection_artifacts_match(
                shared_selection,
                selection,
            ):
                raise ValueError("Integrated graphs require one shared cell selection")
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

        def load_wnn_inputs(assay_name: str) -> tuple[np.ndarray, NDArray[Any]]:
            state = read_assay_state(self.zw, assay_name)
            if state is not None and state.neighbors is not None:
                neighbors_group = artifact_group(self.zw, state.neighbors)
                indices = np.asarray(
                    as_zarr_array(
                        neighbors_group["indices"],
                        name="indices",
                    )[:]
                )
                coordinates_ref = neighbor_coordinates_ref(state.neighbors)
                coordinate_source, _n_cells, _dims = self._coordinate_source(
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
                    _n_cells,
                    expected_dims=_dims,
                )
                if indices.shape[0] != _n_cells:
                    raise ValueError(
                        f"WNN neighbors and coordinates for {assay_name} "
                        "contain different cell counts"
                    )
                return indices, coordinates
            neighbors_path = legacy_wnn_neighbor_paths[assay_name]
            neighbors_group = as_zarr_group(
                self.zw[neighbors_path],
                name=neighbors_path,
            )
            indices = np.asarray(
                as_zarr_array(
                    neighbors_group["indices"],
                    name="indices",
                )[:]
            )
            return indices, legacy_wnn_coordinates[assay_name]

        modality_weights: np.ndarray | None = None
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
            modalities = [(assay, *load_wnn_inputs(assay)) for assay in assays]
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
