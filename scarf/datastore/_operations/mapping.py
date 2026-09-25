import os
from collections.abc import Generator
from tempfile import TemporaryFile
from typing import TYPE_CHECKING, Any, BinaryIO, cast

import numpy as np
import pandas as pd
import zarr
from scipy.sparse import csr_matrix

from ...storage.artifacts import (
    ArtifactRef,
    ValueFingerprintBuilder,
    artifact_group,
    canonical_bytes,
    fingerprint_array,
)
from ...storage.types import as_zarr_array
from ...assay import RNAassay
from ...mapping.artifact import (
    _load_reference_neighbor_query,
    _reference_available_k,
    validate_mapping_reference_binding,
)
from ...mapping.features import AlignedFeatureStream
from ...mapping.confidence import _LabelVotes, _label_vote_block, distance_weights
from ...mapping.models import MappingResult
from ...mapping.projection import (
    NO_QUERY_BATCH_FINGERPRINT,
    ProjectionWriter,
    load_projection,
    plan_projection,
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
from ...storage.geometry import array_geometry
from ...storage.partition import row_band
from ...storage.selections import (
    read_stored_selection_indices,
    validate_stored_selection_integrity,
)
from ...storage.stores import zarr_root_path
from ...utils.logging import logger
from ...storage.feature_selection import (
    _feature_selection_plan,
    _ordered_feature_ids_fingerprint,
    _write_feature_selection,
)

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
    batch_design: csr_matrix | None,
) -> tuple[int, int]:
    float_bytes = np.dtype(np.float64).itemsize
    model_arrays = (
        reference.model.feature_means,
        reference.model.feature_scales,
        reference.model.center,
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
    n_terms = n_batches if batch_design is None else batch_design.shape[1]
    if batch_design is not None:
        resident += 4 * (
            batch_design.data.nbytes
            + batch_design.indices.nbytes
            + batch_design.indptr.nbytes
        )
    solve_bytes = (
        4 * (n_terms + 1) ** 2 * float_bytes
        + 2 * (n_terms + 1) * symphony.n_dims * float_bytes
    )
    # Statistics, fitted offsets, and immutable output buffers overlap in the solve.
    resident += 2 * count_bytes + 3 * sum_bytes + solve_bytes
    per_row += (
        3 * symphony.n_clusters + 3 * symphony.n_dims
    ) * float_bytes + symphony.n_clusters * symphony.n_dims * float_bytes
    return resident, per_row


def _read_projected_blocks(
    coordinates_file: BinaryIO,
    *,
    n_cells: int,
    n_dims: int,
    block_rows: int,
) -> Generator[tuple[int, np.ndarray], None, None]:
    coordinates_file.seek(0)
    for start in range(0, n_cells, block_rows):
        n_rows = min(block_rows, n_cells - start)
        values = np.fromfile(coordinates_file, dtype=np.float64, count=n_rows * n_dims)
        if values.size != n_rows * n_dims:
            raise RuntimeError("Temporary mapping coordinates are incomplete")
        yield start, values.reshape(n_rows, n_dims)


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
        cell_selection: ArtifactRef,
        *,
        query_assay: str | None = None,
        save_k: int = 3,
        missing_feature_policy: str = "reference_mean",
        query_batches: pd.DataFrame | None = None,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Map selected query cells into an immutable prepared reference."""
        if not isinstance(reference, MappingReference):
            raise TypeError("reference must be a MappingReference")
        reference = validate_mapping_reference_binding(reference)
        if not isinstance(cell_selection, ArtifactRef):
            raise TypeError("cell_selection must be an ArtifactRef")
        if query_assay is not None and (
            not isinstance(query_assay, str) or not query_assay.strip()
        ):
            raise TypeError("query_assay must be a non-empty string or None")
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
        validated_cells = validate_stored_selection_integrity(
            self.zw,
            cell_selection,
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        )
        query_cell_indices = read_stored_selection_indices(
            self.zw,
            cell_selection,
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        ).astype(np.int64, copy=False)
        n_cells = validated_cells.selected_count
        if n_cells < 1:
            raise ValueError("cell_selection must select at least one query cell")

        symphony_state = reference.symphony_state
        batch_design = None
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
                query_batches = query_batches.copy(deep=True)
                batch_codes, batch_design = self._query_batch_design(
                    query_batches,
                    n_cells,
                )
                n_batches = batch_design.shape[0]
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
            batch_design=batch_design,
        )
        feature_ids_fingerprint = _ordered_feature_ids_fingerprint(assay.z)
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
        if _ordered_feature_ids_fingerprint(assay.z) != feature_ids_fingerprint:
            raise ValueError("Query feature identities changed during mapping setup")
        selected_expression_fingerprint = stream.raw_expression_fingerprint

        all_features = cast(Any, self).select_all_features(
            from_assay=assay.name,
        )
        feature_mask = np.zeros(assay.feats.N, dtype=bool)
        feature_mask[stream.query_feature_indices] = True
        selection_plan = _feature_selection_plan(
            self.zw,
            assay=assay_name,
            n_features=assay.feats.N,
            ordered_feature_ids_fingerprint=feature_ids_fingerprint,
            operation="select_mapping_overlap",
            parameters={},
            inputs={
                "mapping_reference": reference.external_ref,
                "all_features": all_features,
            },
            execution_options={},
            expected_payload_fingerprint=fingerprint_array(feature_mask),
        )
        _write_feature_selection(
            self.zw,
            selection_plan,
            ordered_feature_ids_fingerprint=feature_ids_fingerprint,
            payload={"values": feature_mask},
        )
        feature_selection = selection_plan.ref
        projection_plan = plan_projection(
            self.zw,
            query_assay=assay_name,
            n_cells=n_cells,
            save_k=save_k,
            missing_feature_policy=missing_feature_policy,
            correction_method=correction_method,
            cell_selection=cell_selection,
            feature_selection=feature_selection,
            selected_expression_fingerprint=selected_expression_fingerprint,
            query_batch_fingerprint=query_batch_fingerprint,
            query_batch_count=n_batches,
            mapping_reference=reference.external_ref,
            reference=reference,
            reference_cell_count=reference.selected_cell_count,
            invalidate_cache=invalidate_cache,
        )
        if projection_plan.reused:
            return projection_plan.ref

        writer = ProjectionWriter(
            self.zw,
            projection_plan,
            chunk_rows=stream.row_geometry.block_rows,
            profile=self.storageProfile,
        )
        dispersion_total = 0.0
        informative_total = 0
        zero_norm_count = 0

        def projected_blocks() -> Generator[tuple[int, np.ndarray], None, None]:
            nonlocal dispersion_total, informative_total, zero_norm_count
            expected_start = 0
            for block in stream:
                if block.row_offset != expected_start:
                    raise RuntimeError("Aligned query blocks are not contiguous")
                coordinates = project_pca(block.values, reference.model)
                uninformative = zero_norm_rows(coordinates)
                zero_norm_count += int(np.count_nonzero(uninformative))
                informative = ~uninformative
                if informative.any():
                    dispersion_total += scaled_dispersion_sum(
                        block.values[informative], reference.model
                    )
                    informative_total += int(np.count_nonzero(informative))
                expected_start += len(coordinates)
                yield block.row_offset, coordinates
            if expected_start != n_cells:
                raise RuntimeError("Mapping did not cover all selected query cells")

        coordinates_file: BinaryIO | None = None
        try:
            neighbor_query = _load_reference_neighbor_query(
                reference,
                save_k=save_k,
                workers=self.resources.workers,
            )
            coordinate_blocks = projected_blocks()
            if symphony_state is not None:
                assert batch_codes is not None
                coordinates_file = TemporaryFile()
                counts, sums = initialize_sufficient_statistics(
                    n_batches,
                    symphony_state,
                )
                for row_offset, coordinates in coordinate_blocks:
                    coordinates.tofile(coordinates_file)
                    assignments = soft_cluster_assignments(
                        coordinates,
                        symphony_state,
                    )
                    stop = row_offset + len(coordinates)
                    informative = ~zero_norm_rows(coordinates)
                    if informative.any():
                        accumulate_sufficient_statistics(
                            counts,
                            sums,
                            coordinates[informative],
                            assignments[informative],
                            batch_codes[row_offset:stop][informative],
                        )
                correction = solve_query_correction(
                    counts,
                    sums,
                    symphony_state,
                    batch_design=batch_design,
                )
                coordinate_blocks = _read_projected_blocks(
                    coordinates_file,
                    n_cells=n_cells,
                    n_dims=reference.model.n_dims,
                    block_rows=stream.row_geometry.block_rows,
                )

            expected_start = 0
            for row_offset, coordinates in coordinate_blocks:
                if row_offset != expected_start:
                    raise RuntimeError("Projected query blocks are not contiguous")
                uninformative = zero_norm_rows(coordinates)
                query_coordinates = coordinates
                if symphony_state is not None:
                    assert batch_codes is not None
                    assignments = soft_cluster_assignments(
                        coordinates,
                        symphony_state,
                    )
                    stop = row_offset + len(coordinates)
                    query_coordinates = apply_query_correction(
                        coordinates,
                        assignments,
                        batch_codes[row_offset:stop],
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
                    row_offset,
                    np.asarray(indices, dtype=np.uint64),
                    np.asarray(distances, dtype=np.float64),
                    np.asarray(uninformative, dtype=bool),
                )
                expected_start = row_offset + len(coordinates)
            if expected_start != n_cells:
                raise RuntimeError("Mapping did not cover all selected query cells")
            if (
                stream.fingerprint_live_raw_expression()
                != selected_expression_fingerprint
            ):
                raise ValueError("Query expression changed during mapping")
            if _ordered_feature_ids_fingerprint(assay.z) != feature_ids_fingerprint:
                raise ValueError("Query feature identities changed during mapping")
            final_cells = validate_stored_selection_integrity(
                self.zw,
                cell_selection,
                kind="cell_selection",
                scope="datastore",
                assay=None,
                table_path="cellData",
            )
            if final_cells.selected_count != n_cells:
                raise ValueError("Query cell selection changed during mapping")
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
        finally:
            if coordinates_file is not None:
                coordinates_file.close()
        return projection_plan.ref

    @staticmethod
    def _query_batch_design(
        query_batches: pd.DataFrame, n_cells: int
    ) -> tuple[np.ndarray, csr_matrix]:
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
        n_groups = len(levels)
        n_variables = query_batches.shape[1]
        columns = np.empty((n_groups, n_variables), dtype=np.int64)
        n_terms = 0
        for variable in range(n_variables):
            variable_codes, categories = pd.factorize(
                levels.get_level_values(variable), sort=False
            )
            columns[:, variable] = variable_codes + n_terms
            n_terms += len(categories)
        design = csr_matrix(
            (
                np.ones(columns.size, dtype=np.float64),
                columns.ravel(),
                np.arange(n_groups + 1, dtype=np.int64) * n_variables,
            ),
            shape=(n_groups, n_terms),
        )
        return resolved, design

    def get_mapping_result(
        self,
        result: ArtifactRef,
        *,
        reference: MappingReference,
        load_arrays: bool = False,
    ) -> MappingResult:
        """Load one complete query-owned mapping projection."""
        if not isinstance(result, ArtifactRef):
            raise TypeError("result must be an ArtifactRef")
        if not isinstance(reference, MappingReference):
            raise TypeError("reference must be a MappingReference")
        reference = validate_mapping_reference_binding(reference)

        return load_projection(
            self.zw,
            result,
            load_arrays=load_arrays,
            reference=reference,
        )

    def get_mapping_score(
        self,
        result: ArtifactRef,
        target_groups: np.ndarray | None = None,
        *,
        reference: MappingReference,
        log_transform: bool = True,
        multiplier: float = 1000,
        weighted: bool = True,
        fixed_weight: float = 0.1,
    ) -> Generator[tuple[Any, np.ndarray], None, None]:
        """Yield reference-sized mapping scores for each requested query group."""
        loaded = self.get_mapping_result(result, reference=reference, load_arrays=False)
        yield from self._mapping_scores(
            loaded,
            target_groups=target_groups,
            log_transform=log_transform,
            multiplier=multiplier,
            weighted=weighted,
            fixed_weight=fixed_weight,
        )

    def _mapping_score_data(
        self,
        result: ArtifactRef,
        *,
        reference: MappingReference,
        target_groups: np.ndarray | None = None,
        layout: ArtifactRef | None = None,
        reference_class_group: str | None = None,
        log_transform: bool = True,
        multiplier: float = 1000,
        weighted: bool = True,
        fixed_weight: float = 0.1,
    ) -> tuple[
        MappingResult,
        list[tuple[Any, np.ndarray]],
        np.ndarray | None,
        np.ndarray | None,
    ]:
        loaded = self.get_mapping_result(result, reference=reference, load_arrays=False)
        scores = list(
            self._mapping_scores(
                loaded,
                target_groups=target_groups,
                log_transform=log_transform,
                multiplier=multiplier,
                weighted=weighted,
                fixed_weight=fixed_weight,
            )
        )
        classes = None
        if reference_class_group is not None:
            classes, _ = loaded.reference._selected_cell_values(
                reference_class_group, validate_binding=False
            )
        coordinates = None if layout is None else loaded.reference._fetch_layout(layout)
        return loaded, scores, classes, coordinates

    def _mapping_scores(
        self,
        loaded: MappingResult,
        *,
        target_groups: np.ndarray | None,
        log_transform: bool,
        multiplier: float,
        weighted: bool,
        fixed_weight: float,
    ) -> Generator[tuple[Any, np.ndarray], None, None]:
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

    @staticmethod
    def _reference_label_codes(
        reference: MappingReference, column: str
    ) -> tuple[np.ndarray, np.ndarray]:
        labels, valid = reference._fetch_cell_labels(column)
        codes = np.full(len(labels), -1, dtype=np.int64)
        valid_codes, categories = pd.factorize(labels[valid], sort=False)
        codes[valid] = valid_codes
        return np.asarray(categories, dtype=object), codes

    def _iter_label_votes(
        self,
        loaded: MappingResult,
        reference_codes: np.ndarray,
        threshold: float,
        selected_rows: np.ndarray | None = None,
    ) -> Generator[tuple[np.ndarray, np.ndarray, _LabelVotes], None, None]:
        indices, distances, uninformative = self._projection_arrays(loaded.ref)
        block_size = self._projection_block_size(indices)
        for start in range(0, loaded.n_cells, block_size):
            stop = min(start + block_size, loaded.n_cells)
            if selected_rows is None:
                offsets = np.arange(stop - start)
            else:
                left, right = np.searchsorted(selected_rows, (start, stop))
                offsets = selected_rows[left:right] - start
            if not offsets.size:
                continue
            block_uninformative = np.asarray(uninformative[start:stop], dtype=bool)
            offsets = offsets[~block_uninformative[offsets]]
            if not offsets.size:
                continue
            block_indices = np.asarray(indices[start:stop])[offsets]
            block_distances = np.asarray(distances[start:stop])[offsets]
            votes = _label_vote_block(
                reference_codes[block_indices],
                distance_weights(block_distances),
                threshold,
            )
            yield start + offsets, block_distances[:, 0], votes

    def get_target_classes(
        self,
        result: ArtifactRef,
        reference_class_group: str,
        *,
        reference: MappingReference,
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
            load_arrays=False,
        )
        class_labels, reference_codes = self._reference_label_codes(
            loaded.reference, reference_class_group
        )

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

        selected_rows = (
            np.arange(loaded.n_cells, dtype=np.int64)
            if target_subset_set is None
            else np.asarray(sorted(target_subset_set), dtype=np.int64)
        )
        predictions = np.full(len(selected_rows), na_val, dtype=object)
        for rows, _distances, votes in self._iter_label_votes(
            loaded, reference_codes, threshold, selected_rows
        ):
            known = ~votes.is_unknown
            positions = np.searchsorted(selected_rows, rows[known])
            predictions[positions] = class_labels[votes.prediction_codes[known]]
        return pd.Series(predictions.tolist(), index=selected_rows)

    def get_target_label_evidence(
        self,
        result: ArtifactRef,
        reference_class_group: str,
        *,
        reference: MappingReference,
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
            load_arrays=False,
        )
        class_labels, reference_codes = self._reference_label_codes(
            loaded.reference, reference_class_group
        )

        from ...mapping.confidence import (
            _conformal_membership,
            _validated_conformal_calibration,
        )

        prepared_calibration: np.ndarray | None = None
        resolved_conformal_alpha = 0.0
        if calibration_nonconformity is not None:
            prepared_calibration, resolved_conformal_alpha = (
                _validated_conformal_calibration(
                    calibration_nonconformity,
                    conformal_alpha,
                )
            )

        predictions = np.full(loaded.n_cells, na_val, dtype=object)
        vote_fraction = np.full(loaded.n_cells, np.nan, dtype=np.float64)
        vote_entropy = np.full(loaded.n_cells, np.nan, dtype=np.float64)
        top_two_margin = np.full(loaded.n_cells, np.nan, dtype=np.float64)
        best_distances = np.full(loaded.n_cells, np.nan, dtype=np.float64)
        prediction_sets: list[tuple[Any, ...]] | None = (
            [()] * loaded.n_cells if prepared_calibration is not None else None
        )
        is_unknown = np.ones(loaded.n_cells, dtype=bool)
        for rows, distances, votes in self._iter_label_votes(
            loaded, reference_codes, threshold
        ):
            unknown = votes.is_unknown.copy()
            if distance_limit is not None:
                unknown |= distances > distance_limit
            known = ~unknown
            predictions[rows[known]] = class_labels[votes.prediction_codes[known]]
            vote_fraction[rows] = votes.vote_fraction
            vote_entropy[rows] = votes.vote_entropy
            top_two_margin[rows] = votes.top_two_margin
            best_distances[rows] = distances
            is_unknown[rows] = unknown
            if prediction_sets is not None:
                assert prepared_calibration is not None
                for position in np.flatnonzero(votes.vote_fraction > 0):
                    label_scores = np.zeros(len(class_labels), dtype=np.float64)
                    valid = votes.class_codes[position] >= 0
                    label_scores[votes.class_codes[position, valid]] = votes.fractions[
                        position, valid
                    ]
                    prediction_mask = _conformal_membership(
                        label_scores, prepared_calibration, resolved_conformal_alpha
                    )
                    prediction_sets[int(rows[position])] = tuple(
                        class_labels[prediction_mask].tolist()
                    )

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
        if prediction_sets is not None:
            evidence["predictionSet"] = prediction_sets
        return evidence

    @staticmethod
    def calibrate_label_transfer_threshold(
        vote_fractions: np.ndarray,
        correct: np.ndarray,
        target_coverage: float = 0.9,
    ) -> dict[str, float]:
        """Choose a vote threshold on held-out, donor-level validation data."""
        raw_fractions = np.asarray(vote_fractions)
        raw_correct = np.asarray(correct)
        if raw_fractions.ndim != 1 or raw_correct.shape != raw_fractions.shape:
            raise ValueError("vote_fractions and correct must be matching vectors")
        if raw_fractions.dtype.kind not in {"i", "u", "f"}:
            raise ValueError("vote_fractions must be real numeric values in [0, 1]")
        if raw_correct.dtype != np.dtype(bool):
            raise ValueError("correct must be a boolean vector")
        fractions = np.asarray(raw_fractions, dtype=np.float64)
        if (
            not np.all(np.isfinite(fractions))
            or np.any(fractions < 0)
            or np.any(fractions > 1)
        ):
            raise ValueError("vote_fractions must be finite values in [0, 1]")
        coverage = _finite_in_range(
            target_coverage,
            "target_coverage must be in (0, 1]",
            low=0.0,
            high=1.0,
            low_open=True,
        )
        correct_values = np.asarray(raw_correct, dtype=bool)
        valid = fractions[correct_values]
        if valid.size == 0:
            raise ValueError("At least one correct held-out prediction is required")
        threshold = float(np.quantile(valid, 1 - coverage))
        selected = fractions >= threshold
        accuracy = float(correct_values[selected].mean()) if selected.any() else 0.0
        return {
            "voteThreshold": threshold,
            "validationCoverage": float(selected.mean()),
            "validationAccuracy": accuracy,
        }
