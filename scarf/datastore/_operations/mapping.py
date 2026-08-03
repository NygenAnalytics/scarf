import os
from collections.abc import Generator
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
import zarr

from ...storage.artifacts import (
    ArtifactRef,
    ValueFingerprintBuilder,
    artifact_group,
    canonical_bytes,
    inspect_artifact,
)
from ...storage.types import as_zarr_array
from ...assay import RNAassay
from ...mapping.features import AlignedFeatureStream
from ...mapping.models import MappingResult
from ...mapping.projection import (
    NO_QUERY_BATCH_FINGERPRINT,
    ProjectionWriter,
    load_projection,
    plan_projection,
    resolve_projection,
)
from ...mapping.reference import MappingReference
from ...mapping.symphony import (
    SYMPHONY_ALGORITHM,
    accumulate_sufficient_statistics,
    apply_query_correction,
    initialize_sufficient_statistics,
    project_pca,
    scaled_dispersion_sum,
    soft_cluster_assignments,
    solve_query_correction,
    zero_norm_rows,
)
from ...neighbors.stages import (
    AnnIndexStage,
    NeighborQueryStage,
)
from ...storage.ann_index import has_ann_index, load_ann_index
from ...storage.geometry import array_geometry
from ...storage.partition import row_band
from ...storage.selections import resolve_selection_artifact
from ...storage.stores import zarr_root_path
from ...utils.logging import logger

if TYPE_CHECKING:
    from ..graph_datastore import GraphDataStore as _MappingOperationsBase
else:
    _MappingOperationsBase = object


def _finite_in_range(
    value: Any,
    message: str,
    *,
    low: float,
    high: float | None = None,
    low_open: bool = False,
) -> float:
    """Coerce one numeric argument, rejecting bools and out-of-range values."""
    if isinstance(value, bool | np.bool_) or not isinstance(
        value,
        int | float | np.integer | np.floating,
    ):
        raise ValueError(message)
    resolved = float(value)
    if not np.isfinite(resolved) or resolved < low:
        raise ValueError(message)
    if low_open and resolved == low:
        raise ValueError(message)
    if high is not None and resolved > high:
        raise ValueError(message)
    return resolved


def _normalized_store_location(value: object) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    location = value.rstrip("/")
    if location.startswith("file://"):
        location = location[7:]
    if "://" in location:
        return location
    return os.path.realpath(os.path.abspath(os.path.expanduser(location)))


def _physical_store_tokens(datastore: Any) -> set[tuple[str, str | int]]:
    root = datastore.z
    store = root.store
    tokens: set[tuple[str, str | int]] = {("object", id(store))}
    root_path = _normalized_store_location(zarr_root_path(root))
    if root_path is not None:
        tokens.add(("root", root_path))
    location = _normalized_store_location(getattr(datastore, "zarr_loc", None))
    if location is not None:
        tokens.add(("location", location))
    store_root = _normalized_store_location(str(getattr(store, "root", "")))
    if store_root is not None:
        tokens.add(("root", store_root))
    return tokens


def _same_physical_store(query: Any, reference: MappingReference) -> bool:
    reference_datastore = reference.datastore
    if not hasattr(reference_datastore, "z"):
        raise TypeError("reference.datastore must be an open DataStore")
    return bool(
        _physical_store_tokens(query) & _physical_store_tokens(reference_datastore)
    )


def _query_batch_fingerprint(query_batches: pd.DataFrame) -> str:
    columns = [
        {
            "type": f"{type(column).__module__}.{type(column).__qualname__}",
            "value": repr(column),
        }
        for column in query_batches.columns
    ]
    row_hashes = pd.util.hash_pandas_object(
        query_batches,
        index=False,
        categorize=True,
    ).to_numpy(dtype=np.uint64, copy=True)
    builder = ValueFingerprintBuilder()
    builder.update_bytes(
        "query_batch_schema",
        canonical_bytes(
            {
                "columns": columns,
                "dtypes": [str(dtype) for dtype in query_batches.dtypes],
            }
        ),
    )
    builder.update_array("query_batch_rows", row_hashes)
    return builder.hexdigest()


def _mapping_memory_reservations(
    reference: MappingReference,
    *,
    n_batches: int,
    save_k: int,
    batch_codes: np.ndarray | None,
) -> tuple[int, int]:
    float_bytes = np.dtype(np.float64).itemsize
    model_arrays = (
        reference.model.feature_means,
        reference.model.feature_scales,
        reference.model.loadings,
        reference.feature_ids,
    )
    resident = sum(np.asarray(values).nbytes for values in model_arrays)
    if batch_codes is not None:
        resident += batch_codes.nbytes

    n_features = reference.model.n_features
    n_dims = reference.model.n_dims
    per_row = (
        2 * n_features * float_bytes
        + 2 * n_dims * float_bytes
        + save_k * (np.dtype(np.uint64).itemsize + float_bytes)
        + np.dtype(bool).itemsize
    )

    symphony = reference.symphony_state
    if symphony is None:
        return resident, per_row

    correction_arrays = (
        symphony.centroids,
        symphony.raw_centroids,
        symphony.corrected_centroids,
        symphony.cluster_mass,
        symphony.sigma,
    )
    resident += sum(np.asarray(values).nbytes for values in correction_arrays)
    count_bytes = n_batches * symphony.n_clusters * float_bytes
    sum_bytes = count_bytes * symphony.n_dims
    solve_bytes = (n_batches + 1) ** 2 * float_bytes + 2 * (
        n_batches + 1
    ) * symphony.n_dims * float_bytes
    resident += 2 * count_bytes + 2 * sum_bytes + solve_bytes
    per_row += (
        3 * symphony.n_clusters + 3 * symphony.n_dims
    ) * float_bytes + symphony.n_clusters * symphony.n_dims * float_bytes
    return resident, per_row


def _reference_available_k(reference: MappingReference) -> int:
    root = reference.datastore.zw
    rebuild = "Rebuild it with build_mapping_reference(neighbors)."
    if reference.ref.assay != reference.assay_name:
        raise ValueError(f"Mapping reference assay identity is inconsistent. {rebuild}")
    if reference.model.n_features != len(reference.feature_ids):
        raise ValueError(
            f"Mapping reference feature dimensions are inconsistent. {rebuild}"
        )
    if reference.method == "pca":
        if reference.symphony_state is not None:
            raise ValueError(f"Plain mapping reference has Symphony state. {rebuild}")
    elif reference.method == "symphony":
        if reference.symphony_state is None:
            raise ValueError(
                f"Symphony mapping reference has no correction state. {rebuild}"
            )
    else:
        raise ValueError(f"Mapping reference method is unsupported. {rebuild}")

    expected = (
        (reference.ref, "build_mapping_reference"),
        (reference.reduction, "run_pca"),
        (reference.ann_index, "build_ann_index"),
        (reference.neighbors, "query_neighbors"),
    )
    statuses = {}
    for ref, operation in expected:
        status = inspect_artifact(root, ref)
        if not status.exists or not status.complete or status.operation != operation:
            raise ValueError(f"Mapping reference graph chain is incomplete. {rebuild}")
        statuses[ref] = status

    ann_status = statuses[reference.ann_index]
    ann_parameters = ann_status.parameters or {}
    if ann_parameters.get("ann_metric") != reference.ann_metric:
        raise ValueError(f"Mapping reference ANN metric is inconsistent. {rebuild}")
    ann_ef = ann_parameters.get("ann_ef", 50)
    if isinstance(ann_ef, bool) or not isinstance(ann_ef, int) or ann_ef < 1:
        raise ValueError(f"Mapping reference ANN search depth is invalid. {rebuild}")

    neighbors_status = statuses[reference.neighbors]
    raw_ann = (neighbors_status.inputs or {}).get("ann_index")
    if (
        not isinstance(raw_ann, dict)
        or ArtifactRef.from_dict(raw_ann) != reference.ann_index
    ):
        raise ValueError(
            f"Mapping reference neighbors use another ANN index. {rebuild}"
        )
    if (neighbors_status.parameters or {}).get(
        "distance_metric"
    ) != reference.ann_metric:
        raise ValueError(
            f"Mapping reference neighbor metric is inconsistent. {rebuild}"
        )

    neighbors_group = artifact_group(root, reference.neighbors)
    indices = as_zarr_array(neighbors_group["indices"], name="indices")
    distances = as_zarr_array(neighbors_group["distances"], name="distances")
    if (
        indices.ndim != 2
        or distances.shape != indices.shape
        or int(indices.shape[0]) != reference.selected_cell_count
        or int(indices.shape[1]) < 1
    ):
        raise ValueError(f"Mapping reference neighbor payload is invalid. {rebuild}")
    ann_group = artifact_group(root, reference.ann_index)
    if not has_ann_index(ann_group):
        raise ValueError(f"Mapping reference ANN index is missing. {rebuild}")
    return int(indices.shape[1])


def _load_reference_neighbor_query(
    reference: MappingReference,
    *,
    save_k: int,
    workers: int,
) -> NeighborQueryStage:
    root = reference.datastore.zw
    ann_status = inspect_artifact(root, reference.ann_index)
    parameters = ann_status.parameters or {}
    index = load_ann_index(
        artifact_group(root, reference.ann_index),
        reference.ann_metric,
        reference.model.n_dims,
        expected_count=reference.selected_cell_count,
    )
    configured = AnnIndexStage.configure(
        index,
        ef=int(parameters.get("ann_ef", 50)),
        threads=workers,
    )
    return NeighborQueryStage(configured, save_k, reference.ann_metric)


class _MappingOperationsMixin(_MappingOperationsBase):
    @staticmethod
    def _projection_block_size(indices: Any) -> int:
        return row_band(
            array_geometry(indices),
            unit="chunk",
            fallback=min(max(int(indices.shape[0]), 1), 10_000),
        )

    def _projection_arrays(
        self,
        ref: ArtifactRef,
    ) -> tuple[zarr.Array, zarr.Array, zarr.Array]:
        projection = artifact_group(self.zw, ref)
        return (
            as_zarr_array(projection["indices"], name="indices"),
            as_zarr_array(projection["distances"], name="distances"),
            as_zarr_array(projection["uninformative"], name="uninformative"),
        )

    def run_mapping(
        self,
        reference: MappingReference,
        mapping_name: str,
        *,
        query_assay: str | None = None,
        cell_key: str = "I",
        save_k: int = 3,
        missing_feature_policy: str = "reference_mean",
        query_batches: pd.DataFrame | None = None,
        invalidate_cache: bool = False,
    ) -> MappingResult:
        """Map selected query cells into an immutable prepared reference."""
        if not isinstance(reference, MappingReference):
            raise TypeError("reference must be a MappingReference")
        if not isinstance(mapping_name, str) or not mapping_name.strip():
            raise TypeError("mapping_name must be a non-empty string")
        if query_assay is not None and (
            not isinstance(query_assay, str) or not query_assay.strip()
        ):
            raise TypeError("query_assay must be a non-empty string or None")
        if not isinstance(cell_key, str) or not cell_key.strip():
            raise TypeError("cell_key must be a non-empty string")
        if isinstance(save_k, bool) or not isinstance(save_k, int | np.integer):
            raise TypeError("save_k must be an integer")
        save_k = int(save_k)
        if save_k < 1:
            raise ValueError("save_k must be positive")
        if not isinstance(missing_feature_policy, str):
            raise TypeError("missing_feature_policy must be a string")
        if missing_feature_policy not in {"reference_mean", "zero", "error"}:
            raise ValueError(
                "missing_feature_policy must be 'reference_mean', 'zero', or 'error'"
            )
        if query_batches is not None and not isinstance(query_batches, pd.DataFrame):
            raise TypeError("query_batches must be a pandas DataFrame or None")
        if not isinstance(invalidate_cache, bool):
            raise TypeError("invalidate_cache must be a boolean")
        if self.zarr_mode != "r+":
            raise ValueError("Mapping requires a read-write query datastore")
        if _same_physical_store(self, reference):
            raise ValueError(
                "Query and reference cannot use the same physical Zarr store. "
                "Mount the query into a separate writable datastore."
            )

        reference.validate_dataset_fingerprint()
        assay_name = query_assay or self._defaultAssay
        if assay_name is None:
            raise ValueError("No query assay was provided and no default is configured")
        if assay_name not in self.assay_names:
            raise ValueError(f"Query assay {assay_name!r} was not found")
        assay = self._get_assay(assay_name)
        if not isinstance(assay, RNAassay):
            raise TypeError("Mapping currently supports RNA query assays only")

        cell_mask = np.asarray(self.cells.fetch_all(cell_key))
        if cell_mask.ndim != 1 or cell_mask.dtype != np.dtype(bool):
            raise TypeError("cell_key must identify a one-dimensional boolean column")
        if len(cell_mask) != self.cells.N:
            raise ValueError("cell_key does not align with query cell metadata")
        query_cell_indices = np.flatnonzero(cell_mask).astype(np.int64, copy=False)
        n_cells = len(query_cell_indices)
        if n_cells < 1:
            raise ValueError("cell_key must select at least one query cell")

        symphony_state = reference.symphony_state
        if symphony_state is None:
            if query_batches is not None:
                raise ValueError(
                    "query_batches are only supported by Symphony references"
                )
            batch_codes = None
            n_batches = 1
            query_batch_fingerprint = NO_QUERY_BATCH_FINGERPRINT
            correction_method = "none"
            algorithm_variant = "scaled_pca"
        else:
            if query_batches is None:
                batch_codes = np.zeros(n_cells, dtype=np.int64)
                n_batches = 1
                query_batch_fingerprint = NO_QUERY_BATCH_FINGERPRINT
            else:
                batch_codes, n_batches = self._query_batch_codes(
                    query_batches,
                    n_cells,
                )
                query_batch_fingerprint = _query_batch_fingerprint(query_batches)
            correction_method = "symphony"
            algorithm_variant = SYMPHONY_ALGORITHM

        available_k = _reference_available_k(reference)
        if save_k > available_k:
            logger.warning(f"`save_k` was decreased to {available_k}")
            save_k = available_k

        reserved_resident, reserved_per_row = _mapping_memory_reservations(
            reference,
            n_batches=n_batches,
            save_k=save_k,
            batch_codes=batch_codes,
        )
        stream = AlignedFeatureStream(
            query_assay=assay,
            query_cell_indices=query_cell_indices,
            reference_feature_ids=reference.feature_ids,
            reference_normalized_means=reference.model.feature_means,
            reference_normalization_parameters=reference.normalization_parameters,
            missing_feature_policy=missing_feature_policy,
            resources=self.resources,
            reserved_resident_bytes=reserved_resident,
            reserved_per_row_bytes=reserved_per_row,
        )
        selected_expression_fingerprint = stream.raw_expression_fingerprint

        cell_selection = resolve_selection_artifact(
            self.zw,
            scope="datastore",
            kind="cell_selection",
            values=cell_mask,
            row_ids=np.asarray(self.cells.fetch_all("ids")),
            operation="manual_selection",
            parameters={},
            inputs={},
            source_column=cell_key,
        )
        feature_mask = np.zeros(assay.feats.N, dtype=bool)
        feature_mask[stream.query_feature_indices] = True
        feature_selection = resolve_selection_artifact(
            self.zw,
            scope="assay",
            assay=assay_name,
            kind="feature_selection",
            values=feature_mask,
            row_ids=np.asarray(assay.feats.fetch_all("ids")),
            operation="map_query_feature_selection",
            parameters={},
            inputs={
                "alignment_map_hash": stream.alignment_map_hash,
                "mapping_reference": reference.external_ref,
            },
            source_column="mapping_reference_overlap",
        )
        projection_plan = plan_projection(
            self.zw,
            query_assay=assay_name,
            mapping_name=mapping_name,
            n_cells=n_cells,
            save_k=save_k,
            missing_feature_policy=missing_feature_policy,
            correction_method=correction_method,
            cell_selection=cell_selection,
            feature_selection=feature_selection,
            selected_expression_fingerprint=selected_expression_fingerprint,
            query_batch_fingerprint=query_batch_fingerprint,
            mapping_reference=reference.external_ref,
            reference_cell_count=reference.selected_cell_count,
            invalidate_cache=invalidate_cache,
        )
        if projection_plan.reused:
            return load_projection(
                self.zw,
                projection_plan.ref,
                reference=reference,
            )

        writer = ProjectionWriter(
            self.zw,
            projection_plan,
            chunk_rows=stream.row_geometry.block_rows,
            profile=self.storageProfile,
        )
        try:
            neighbor_query = _load_reference_neighbor_query(
                reference,
                save_k=save_k,
                workers=self.resources.workers,
            )
            if symphony_state is not None:
                assert batch_codes is not None
                counts, sums = initialize_sufficient_statistics(
                    n_batches,
                    symphony_state,
                )
                expected_start = 0
                for block in stream:
                    if block.row_offset != expected_start:
                        raise RuntimeError("Aligned query blocks are not contiguous")
                    coordinates = project_pca(block.values, reference.model)
                    assignments = soft_cluster_assignments(
                        coordinates,
                        symphony_state,
                    )
                    stop = block.row_offset + len(coordinates)
                    informative = ~zero_norm_rows(coordinates)
                    if informative.any():
                        accumulate_sufficient_statistics(
                            counts,
                            sums,
                            coordinates[informative],
                            assignments[informative],
                            batch_codes[block.row_offset : stop][informative],
                        )
                    expected_start = stop
                if expected_start != n_cells:
                    raise RuntimeError(
                        "Symphony statistics did not cover all query cells"
                    )
                correction = solve_query_correction(
                    counts,
                    sums,
                    symphony_state,
                )

            zero_norm_count = 0
            expected_start = 0
            dispersion_total = 0.0
            informative_total = 0
            for block in stream:
                if block.row_offset != expected_start:
                    raise RuntimeError("Aligned query blocks are not contiguous")
                coordinates = project_pca(block.values, reference.model)
                uninformative = zero_norm_rows(coordinates)
                zero_norm_count += int(np.count_nonzero(uninformative))
                informative = ~uninformative
                if informative.any():
                    dispersion_total += scaled_dispersion_sum(
                        block.values[informative],
                        reference.model,
                    )
                    informative_total += int(np.count_nonzero(informative))
                query_coordinates = coordinates
                if symphony_state is not None:
                    assert batch_codes is not None
                    assignments = soft_cluster_assignments(
                        coordinates,
                        symphony_state,
                    )
                    stop = block.row_offset + len(coordinates)
                    query_coordinates = apply_query_correction(
                        coordinates,
                        assignments,
                        batch_codes[block.row_offset : stop],
                        symphony_state,
                        correction,
                    )
                    query_coordinates[uninformative] = coordinates[uninformative]
                queried = cast(
                    tuple[np.ndarray, np.ndarray],
                    neighbor_query.query(query_coordinates),
                )
                indices, distances = queried
                writer.write_block(
                    block.row_offset,
                    np.asarray(indices, dtype=np.uint64),
                    np.asarray(distances, dtype=np.float64),
                    np.asarray(uninformative, dtype=bool),
                )
                expected_start = block.row_offset + len(coordinates)
            if expected_start != n_cells:
                raise RuntimeError("Mapping did not cover all selected query cells")
            denominator = informative_total * reference.model.n_features
            writer.finish(
                {
                    "featureCoverage": stream.feature_coverage,
                    "queryBatchCount": n_batches,
                    "algorithmVariant": algorithm_variant,
                    "zeroNormCellCount": zero_norm_count,
                    "queryScaledDispersion": (
                        dispersion_total / denominator if denominator else 0.0
                    ),
                }
            )
        except BaseException:
            if not writer.finished:
                writer.abort()
            raise
        return load_projection(
            self.zw,
            projection_plan.ref,
            reference=reference,
        )

    @staticmethod
    def _query_batch_codes(
        query_batches: pd.DataFrame, n_cells: int
    ) -> tuple[np.ndarray, int]:
        if len(query_batches) != n_cells:
            raise ValueError("query_batches must have one row per selected query cell")
        if query_batches.shape[1] == 0:
            raise ValueError("query_batches must include at least one column")
        if query_batches.columns.duplicated().any():
            raise ValueError("query_batches column names must be unique")
        if query_batches.isna().any().any():
            raise ValueError("query_batches cannot contain missing values")
        rows = pd.MultiIndex.from_frame(query_batches)
        codes, levels = pd.factorize(rows, sort=False)
        resolved = np.asarray(codes, dtype=np.int64)
        if np.any(resolved < 0):
            raise ValueError("query_batches contain an unencodable row")
        return resolved, len(levels)

    @staticmethod
    def _label_vote_decision(
        reference_labels: np.ndarray,
        neighbors: np.ndarray,
        weights: np.ndarray,
        threshold_fraction: float,
        na_val: str,
        force_unknown: bool = False,
    ) -> tuple[Any, float, float, float, bool, dict[Any, float]]:
        votes: dict[Any, float] = {}
        for neighbor, weight in zip(neighbors, weights):
            label = reference_labels[neighbor]
            votes[label] = votes.get(label, 0.0) + float(weight)
        total = float(sum(votes.values()))
        if total <= 0 or not votes:
            return na_val, 0.0, 0.0, 0.0, True, {}
        votes = {label: value / total for label, value in votes.items()}
        ordered = sorted(votes.items(), key=lambda item: item[1], reverse=True)
        top_vote = ordered[0][1]
        second_vote = ordered[1][1] if len(ordered) > 1 else 0.0
        winners = [label for label, vote in ordered if np.isclose(vote, top_vote)]
        is_unknown = force_unknown or top_vote < threshold_fraction or len(winners) != 1
        entropy = -sum(value * np.log(value) for _, value in ordered if value > 0)
        prediction = na_val if is_unknown else winners[0]
        return (
            prediction,
            top_vote,
            float(entropy),
            top_vote - second_vote,
            is_unknown,
            votes,
        )

    def get_mapping_result(
        self,
        result: MappingResult | ArtifactRef | str,
        *,
        reference: MappingReference | None = None,
        query_assay: str | None = None,
        load_arrays: bool = False,
    ) -> MappingResult:
        """Load one complete query-owned mapping projection."""
        if reference is not None and not isinstance(reference, MappingReference):
            raise TypeError("reference must be a MappingReference or None")
        if query_assay is not None and not isinstance(result, str):
            raise ValueError("query_assay is only valid when result is a string")

        projection_ref: ArtifactRef
        session_reference: MappingReference | None = None
        if isinstance(result, MappingResult):
            projection_ref = result.ref
            session_reference = result.reference
        elif isinstance(result, ArtifactRef):
            projection_ref = result
        elif isinstance(result, str):
            if not result.strip():
                raise TypeError("result must be a non-empty mapping name")
        else:
            raise TypeError(
                "result must be a MappingResult, ArtifactRef, or mapping name"
            )

        if reference is not None and session_reference is not None:
            if reference.external_ref != session_reference.external_ref:
                raise ValueError(
                    "Explicit reference does not match the MappingResult "
                    "reference handle"
                )
        resolved_reference = reference or session_reference
        if resolved_reference is None:
            raise ValueError(
                "A MappingReference is required unless result is an in-session "
                "MappingResult carrying its reference"
            )

        if isinstance(result, str):
            assay_name = query_assay or self._defaultAssay
            if assay_name is None:
                raise ValueError(
                    "query_assay is required when the query store has no default assay"
                )
            projection_ref = resolve_projection(
                self.zw,
                query_assay=assay_name,
                mapping_name=result,
                mapping_reference=resolved_reference.external_ref,
            )

        return load_projection(
            self.zw,
            projection_ref,
            load_arrays=load_arrays,
            reference=resolved_reference,
        )

    def get_mapping_score(
        self,
        result: MappingResult | ArtifactRef | str,
        target_groups: np.ndarray | None = None,
        *,
        reference: MappingReference | None = None,
        query_assay: str | None = None,
        log_transform: bool = True,
        multiplier: float = 1000,
        weighted: bool = True,
        fixed_weight: float = 0.1,
    ) -> Generator[tuple[Any, np.ndarray], None, None]:
        """Yield reference-sized mapping scores for each requested query group."""
        if not isinstance(log_transform, bool):
            raise TypeError("log_transform must be a boolean")
        if not isinstance(weighted, bool):
            raise TypeError("weighted must be a boolean")
        scale = _finite_in_range(
            multiplier,
            "multiplier must be finite and non-negative",
            low=0.0,
        )
        weight = _finite_in_range(
            fixed_weight,
            "fixed_weight must be finite and positive",
            low=0.0,
            low_open=True,
        )

        loaded = self.get_mapping_result(
            result,
            reference=reference,
            query_assay=query_assay,
            load_arrays=False,
        )
        assert loaded.reference is not None
        indices, distances, uninformative = self._projection_arrays(loaded.ref)
        n_cells = loaded.n_cells
        n_k = int(indices.shape[1])

        if target_groups is None:
            groups = np.zeros(n_cells, dtype=np.uint8)
        else:
            groups = np.asarray(target_groups)
            if groups.shape != (n_cells,):
                raise ValueError(
                    "target_groups must contain one value per projected query cell"
                )
        requested_groups = pd.unique(groups)

        from ...mapping.confidence import mapping_score_weights

        for group in requested_groups:
            score = np.zeros(
                loaded.reference.selected_cell_count,
                dtype=np.float64,
            )
            informative_count = 0
            block_size = self._projection_block_size(indices)
            for start in range(0, n_cells, block_size):
                stop = min(start + block_size, n_cells)
                block_groups = groups[start:stop]
                block_uninformative = np.asarray(
                    uninformative[start:stop],
                    dtype=bool,
                )
                if bool(pd.isna(group)):
                    group_mask = np.asarray(pd.isna(block_groups), dtype=bool)
                else:
                    group_mask = np.asarray(block_groups == group, dtype=bool)
                informative_mask = group_mask & ~block_uninformative
                if not informative_mask.any():
                    continue
                block_indices = np.asarray(indices[start:stop])[informative_mask]
                if weighted:
                    block_weights = mapping_score_weights(
                        np.asarray(distances[start:stop])[informative_mask]
                    )
                else:
                    block_weights = np.full(
                        block_indices.shape,
                        weight,
                        dtype=np.float64,
                    )
                np.add.at(
                    score,
                    block_indices.reshape(-1),
                    block_weights.reshape(-1),
                )
                informative_count += int(np.count_nonzero(informative_mask))
            if informative_count:
                score *= scale / (informative_count * n_k)
            if log_transform:
                score = np.log1p(score)
            yield group, score

    def get_target_classes(
        self,
        result: MappingResult | ArtifactRef | str,
        reference_class_group: str,
        *,
        reference: MappingReference | None = None,
        query_assay: str | None = None,
        threshold_fraction: float = 0.5,
        target_subset: list[int] | None = None,
        na_val: str = "NA",
    ) -> pd.Series:
        """Transfer one reference label column to projected query cells."""
        if not isinstance(reference_class_group, str) or not reference_class_group:
            raise TypeError("reference_class_group must be a non-empty string")
        threshold = _finite_in_range(
            threshold_fraction,
            "threshold_fraction must be between zero and one",
            low=0.0,
            high=1.0,
        )
        if not isinstance(na_val, str):
            raise TypeError("na_val must be a string")

        loaded = self.get_mapping_result(
            result,
            reference=reference,
            query_assay=query_assay,
            load_arrays=False,
        )
        assert loaded.reference is not None
        indices, distances, uninformative = self._projection_arrays(loaded.ref)
        reference_labels = loaded.reference.fetch_cell_column(reference_class_group)

        target_subset_set: dict[int, None] | None = None
        if target_subset is not None:
            if not isinstance(target_subset, list):
                raise TypeError("target_subset must be a list or None")
            target_subset_set = {}
            for index in target_subset:
                if isinstance(index, bool | np.bool_) or not isinstance(
                    index,
                    int | np.integer,
                ):
                    raise TypeError("target_subset entries must be integers")
                resolved_index = int(index)
                if not 0 <= resolved_index < loaded.n_cells:
                    raise ValueError("target_subset contains an out-of-range index")
                target_subset_set[resolved_index] = None

        from ...mapping.confidence import distance_weights

        preds: list[Any] = []
        prediction_indices: list[int] = []
        block_size = self._projection_block_size(indices)
        for start in range(0, loaded.n_cells, block_size):
            stop = min(start + block_size, loaded.n_cells)
            if target_subset_set is None:
                selected_offsets = np.arange(stop - start, dtype=np.intp)
            else:
                selected_offsets = np.asarray(
                    [
                        offset
                        for offset in range(stop - start)
                        if start + offset in target_subset_set
                    ],
                    dtype=np.intp,
                )
            if not selected_offsets.size:
                continue
            block_uninformative = np.asarray(
                uninformative[start:stop],
                dtype=bool,
            )
            selected_uninformative = block_uninformative[selected_offsets]
            informative_offsets = selected_offsets[~selected_uninformative]
            informative_neighbors: np.ndarray | None = None
            informative_weights: np.ndarray | None = None
            if informative_offsets.size:
                informative_neighbors = np.asarray(indices[start:stop])[
                    informative_offsets
                ]
                informative_weights = distance_weights(
                    np.asarray(distances[start:stop])[informative_offsets]
                )
            informative_position = 0
            for selected_position, offset in enumerate(selected_offsets):
                row_index = start + int(offset)
                if selected_uninformative[selected_position]:
                    prediction = na_val
                else:
                    assert informative_neighbors is not None
                    assert informative_weights is not None
                    prediction, _, _, _, _, _ = self._label_vote_decision(
                        reference_labels,
                        informative_neighbors[informative_position],
                        informative_weights[informative_position],
                        threshold,
                        na_val,
                    )
                    informative_position += 1
                preds.append(prediction)
                prediction_indices.append(row_index)
        return pd.Series(preds, index=prediction_indices)

    def get_target_label_evidence(
        self,
        result: MappingResult | ArtifactRef | str,
        reference_class_group: str,
        *,
        reference: MappingReference | None = None,
        query_assay: str | None = None,
        threshold_fraction: float = 0.5,
        na_val: str = "NA",
        max_distance: float | None = None,
        calibration_nonconformity: np.ndarray | None = None,
        conformal_alpha: float = 0.1,
    ) -> pd.DataFrame:
        """Return neighbor-vote evidence, novelty context, and unknown assignments.

        ``calibration_nonconformity`` optionally adds split-conformal prediction
        sets. Its calibration rows must be exchangeable with future queries.
        """
        if not isinstance(reference_class_group, str) or not reference_class_group:
            raise TypeError("reference_class_group must be a non-empty string")
        threshold = _finite_in_range(
            threshold_fraction,
            "threshold_fraction must be between zero and one",
            low=0.0,
            high=1.0,
        )
        if not isinstance(na_val, str):
            raise TypeError("na_val must be a string")
        distance_limit = (
            None
            if max_distance is None
            else _finite_in_range(
                max_distance,
                "max_distance must be finite and non-negative",
                low=0.0,
            )
        )

        loaded = self.get_mapping_result(
            result,
            reference=reference,
            query_assay=query_assay,
            load_arrays=False,
        )
        assert loaded.reference is not None
        indices, distances, uninformative = self._projection_arrays(loaded.ref)
        reference_labels = loaded.reference.fetch_cell_column(reference_class_group)
        class_labels = np.asarray(pd.unique(reference_labels), dtype=object)
        class_positions = {
            label: position for position, label in enumerate(class_labels)
        }

        from ...mapping.confidence import distance_weights

        predictions = np.full(loaded.n_cells, na_val, dtype=object)
        vote_fraction = np.full(loaded.n_cells, np.nan, dtype=np.float64)
        vote_entropy = np.full(loaded.n_cells, np.nan, dtype=np.float64)
        top_two_margin = np.full(loaded.n_cells, np.nan, dtype=np.float64)
        best_distances = np.full(loaded.n_cells, np.nan, dtype=np.float64)
        label_scores = (
            np.zeros(
                (loaded.n_cells, len(class_labels)),
                dtype=np.float64,
            )
            if calibration_nonconformity is not None
            else None
        )
        is_unknown = np.ones(loaded.n_cells, dtype=bool)
        block_size = self._projection_block_size(indices)
        for start in range(0, loaded.n_cells, block_size):
            stop = min(start + block_size, loaded.n_cells)
            block_uninformative = np.asarray(
                uninformative[start:stop],
                dtype=bool,
            )
            informative_offsets = np.flatnonzero(~block_uninformative)
            if not informative_offsets.size:
                continue
            block_indices = np.asarray(indices[start:stop])[informative_offsets]
            block_distances = np.asarray(distances[start:stop])[informative_offsets]
            block_weights = distance_weights(block_distances)
            for position, offset in enumerate(informative_offsets):
                row_index = start + int(offset)
                (
                    prediction,
                    top_vote,
                    entropy,
                    margin,
                    row_unknown,
                    votes,
                ) = self._label_vote_decision(
                    reference_labels,
                    block_indices[position],
                    block_weights[position],
                    threshold,
                    na_val,
                )
                best_distance = float(block_distances[position, 0])
                if distance_limit is not None and best_distance > distance_limit:
                    prediction = na_val
                    row_unknown = True
                predictions[row_index] = prediction
                vote_fraction[row_index] = top_vote
                vote_entropy[row_index] = entropy
                top_two_margin[row_index] = margin
                best_distances[row_index] = best_distance
                is_unknown[row_index] = row_unknown
                if label_scores is not None:
                    for label, score in votes.items():
                        label_scores[row_index, class_positions[label]] = score

        distance_quantiles = loaded.reference.reference_distance_quantiles
        distance_values = loaded.reference.reference_distance_values
        unique_distance_values = np.unique(distance_values)
        right_indices = (
            np.searchsorted(distance_values, unique_distance_values, side="right") - 1
        )
        unique_distance_quantiles = distance_quantiles[right_indices]
        distance_percentile = np.full(
            loaded.n_cells,
            np.nan,
            dtype=np.float64,
        )
        informative = np.isfinite(best_distances)
        if informative.any():
            distance_percentile[informative] = np.interp(
                best_distances[informative],
                unique_distance_values,
                unique_distance_quantiles,
                left=0.0,
                right=1.0,
            )
        feature_coverage = float(loaded.diagnostics["featureCoverage"])
        evidence = pd.DataFrame(
            {
                "label": predictions,
                "voteFraction": vote_fraction,
                "voteEntropy": vote_entropy,
                "topTwoMargin": top_two_margin,
                "featureCoverage": feature_coverage,
                "queryScaledDispersion": float(
                    loaded.diagnostics["queryScaledDispersion"]
                ),
                "referenceDistancePercentile": distance_percentile,
                "isUnknown": is_unknown,
            }
        )
        if calibration_nonconformity is not None:
            from ...mapping.confidence import conformal_prediction_sets

            assert label_scores is not None
            prediction_masks = conformal_prediction_sets(
                label_scores,
                calibration_nonconformity,
                alpha=conformal_alpha,
            )
            evidence["predictionSet"] = [
                tuple(class_labels[mask].tolist()) if informative[row_index] else ()
                for row_index, mask in enumerate(prediction_masks)
            ]
        return evidence

    @staticmethod
    def calibrate_label_transfer_threshold(
        vote_fractions: np.ndarray,
        correct: np.ndarray,
        target_coverage: float = 0.9,
    ) -> dict[str, float]:
        """Choose a vote threshold on held-out, donor-level validation data."""
        fractions = np.asarray(vote_fractions, dtype=np.float64)
        correct = np.asarray(correct, dtype=bool)
        if fractions.ndim != 1 or correct.shape != fractions.shape:
            raise ValueError("vote_fractions and correct must be matching vectors")
        if not 0 < target_coverage <= 1:
            raise ValueError("target_coverage must be in (0, 1]")
        valid = fractions[correct]
        if valid.size == 0:
            raise ValueError("At least one correct held-out prediction is required")
        threshold = float(np.quantile(valid, 1 - target_coverage))
        selected = fractions >= threshold
        accuracy = float(correct[selected].mean()) if selected.any() else 0.0
        return {
            "voteThreshold": threshold,
            "validationCoverage": float(selected.mean()),
            "validationAccuracy": accuracy,
        }
