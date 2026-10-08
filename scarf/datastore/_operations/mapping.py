from collections.abc import Generator
from contextlib import closing
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
from ...mapping.label_transfer import (
    LabelTransferBlock,
    ReferenceDistancePercentiles,
    load_label_transfer,
    plan_label_transfer,
    plan_reference_labels,
    read_reference_labels,
    transfer_label_block,
    validate_reference_label_source,
    write_label_transfer,
    write_reference_labels,
)
from ...mapping.models import LabelTransferResult, MappingResult
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
)
from ...storage.geometry import array_geometry
from ...storage.partition import row_band
from ...storage.selections import (
    read_stored_selection_indices,
    validate_stored_selection_integrity,
)
from ...storage.stores import locations_overlap, zarr_root_path
from ...utils.arrays import sparse_matrix_bytes
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


def _store_locations(datastore: Any) -> set[str]:
    candidates = (zarr_root_path(datastore.z), getattr(datastore, "zarr_loc", None))
    return {value for value in candidates if isinstance(value, str) and value}


def _same_physical_store(query: Any, reference: MappingReference) -> bool:
    """Return whether the two datastores share or nest one Zarr store."""
    reference_datastore = reference.datastore
    if not hasattr(reference_datastore, "z"):
        raise TypeError("reference.datastore must be an open DataStore")
    if query.z.store is reference_datastore.z.store:
        return True
    return any(
        locations_overlap(first, second)
        for first in _store_locations(query)
        for second in _store_locations(reference_datastore)
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

    if batch_codes is not None:
        # The second pass replays one uninformative flag per selected cell.
        resident += len(batch_codes) * np.dtype(bool).itemsize
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
        resident += 4 * sparse_matrix_bytes(batch_design)
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
    uninformative: np.ndarray,
    *,
    n_dims: int,
    block_rows: int,
) -> Generator[tuple[int, np.ndarray, np.ndarray], None, None]:
    n_cells = len(uninformative)
    coordinates_file.seek(0)
    for start in range(0, n_cells, block_rows):
        n_rows = min(block_rows, n_cells - start)
        values = np.fromfile(coordinates_file, dtype=np.float64, count=n_rows * n_dims)
        if values.size != n_rows * n_dims:
            raise RuntimeError("Temporary mapping coordinates are incomplete")
        yield (
            start,
            values.reshape(n_rows, n_dims),
            uninformative[start : start + n_rows],
        )


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
        """Map selected query cells into an immutable prepared reference.

        A query cell whose raw counts are zero in every reference feature that
        the query measured is recorded as uninformative. It keeps a projection
        row but is excluded from label transfer, mapping scores, Symphony
        query-batch statistics, and ``queryScaledDispersion``.

        Raises:
            UnmeasuredCellsError: If the query assay did not measure a selected cell.
        """
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
        # After the argument and assay checks, and before the write check, so
        # a read-only store refuses unmeasured cells as a writable one does.
        self._require_measured_cells(
            assay_name, cell_selection, operation="run_mapping"
        )
        if self.zarr_mode != "r+":
            raise ValueError("Mapping requires a read-write query datastore")
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
        query_dataset_fingerprint = self._ensure_dataset_fingerprint(assay_name)
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
            query_dataset_fingerprint=query_dataset_fingerprint,
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
        overlap_features = stream.reference_index_map
        dispersion_total = 0.0
        informative_total = 0

        def projected_blocks() -> Generator[
            tuple[int, np.ndarray, np.ndarray], None, None
        ]:
            nonlocal dispersion_total, informative_total
            expected_start = 0
            with closing(stream.iter_blocks()) as blocks:
                for block in blocks:
                    if block.row_offset != expected_start:
                        raise RuntimeError("Aligned query blocks are not contiguous")
                    coordinates = project_pca(block.values, reference.model)
                    informative = block.observed
                    if informative.any():
                        dispersion_total += scaled_dispersion_sum(
                            block.values[informative],
                            reference.model,
                            features=overlap_features,
                        )
                        informative_total += int(np.count_nonzero(informative))
                    expected_start += len(coordinates)
                    yield block.row_offset, coordinates, ~informative
            if expected_start != n_cells:
                raise RuntimeError("Mapping did not cover all selected query cells")

        coordinates_file: BinaryIO | None = None
        coordinate_blocks = projected_blocks()
        try:
            neighbor_query = _load_reference_neighbor_query(
                reference,
                save_k=save_k,
                workers=self.resources.workers,
            )
            if symphony_state is not None:
                assert batch_codes is not None
                coordinates_file = TemporaryFile()
                uninformative_rows = np.empty(n_cells, dtype=bool)
                counts, sums = initialize_sufficient_statistics(
                    n_batches,
                    symphony_state,
                )
                for row_offset, coordinates, uninformative in coordinate_blocks:
                    coordinates.tofile(coordinates_file)
                    stop = row_offset + len(coordinates)
                    uninformative_rows[row_offset:stop] = uninformative
                    informative = ~uninformative
                    if informative.any():
                        assignments = soft_cluster_assignments(
                            coordinates[informative],
                            symphony_state,
                        )
                        accumulate_sufficient_statistics(
                            counts,
                            sums,
                            coordinates[informative],
                            assignments,
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
                    uninformative_rows,
                    n_dims=reference.model.n_dims,
                    block_rows=stream.row_geometry.block_rows,
                )

            uninformative_count = 0
            expected_start = 0
            for row_offset, coordinates, uninformative in coordinate_blocks:
                if row_offset != expected_start:
                    raise RuntimeError("Projected query blocks are not contiguous")
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
                uninformative_count += int(np.count_nonzero(uninformative))
                expected_start = row_offset + len(coordinates)
            if expected_start != n_cells:
                raise RuntimeError("Mapping did not cover all selected query cells")
            if _ordered_feature_ids_fingerprint(assay.z) != feature_ids_fingerprint:
                raise ValueError("Query feature identities changed during mapping")
            denominator = informative_total * len(overlap_features)
            writer.finish(
                {
                    "featureCoverage": stream.feature_coverage,
                    "queryBatchCount": n_batches,
                    "algorithmVariant": algorithm_variant,
                    "uninformativeCellCount": uninformative_count,
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
            # A failed query leaves the block stream suspended while it holds
            # Zarr's process-wide I/O limit; release it now, not at collection.
            coordinate_blocks.close()
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
        for name, column in query_batches.items():
            # The batch fingerprint hashes values by their text, so values
            # such as 1 and "1" would name different batches with one digest.
            values = pd.unique(column.to_numpy(dtype=object))
            texts = {
                value if isinstance(value, bytes) else str(value).encode()
                for value in values
            }
            if len(texts) != len(values):
                raise ValueError(
                    f"query_batches column {name!r} has distinct values with the "
                    "same text, such as 1 and '1'; use one value type per column"
                )
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
        reference_labels: str | ArtifactRef | None = None,
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
        """Return mapping scores and optional reference classes and layout.

        Reference classes come from ``reference_labels``, a reference column or
        label artifact; a reference cell without a usable label has class None.
        """
        if reference_labels is not None:
            reference_labels = validate_reference_label_source(reference_labels)
        loaded = self.get_mapping_result(result, reference=reference, load_arrays=False)
        scores = list(
            self._mapping_scores(
                loaded,
                target_groups=target_groups,
                log_transform=log_transform,
                multiplier=multiplier,
                weighted=weighted,
                fixed_weight=fixed_weight,
                one_pass=True,
            )
        )
        classes = None
        if reference_labels is not None:
            values, usable = read_reference_labels(loaded.reference, reference_labels)
            classes = np.asarray(values, dtype=object)
            classes[~usable] = None
        coordinates = None if layout is None else loaded.reference._fetch_layout(layout)
        return loaded, scores, classes, coordinates

    @staticmethod
    def _mapping_score_settings(
        loaded: MappingResult,
        target_groups: np.ndarray | None,
        *,
        log_transform: bool,
        multiplier: float,
        weighted: bool,
        fixed_weight: float,
    ) -> tuple[np.ndarray, float, float]:
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
        if target_groups is None:
            groups = np.zeros(loaded.n_cells, dtype=np.uint8)
        else:
            groups = np.asarray(target_groups)
            if groups.shape != (loaded.n_cells,):
                raise ValueError(
                    "target_groups must contain one value per projected query cell"
                )
        return groups, scale, weight

    def _mapping_scores(
        self,
        loaded: MappingResult,
        *,
        target_groups: np.ndarray | None,
        log_transform: bool,
        multiplier: float,
        weighted: bool,
        fixed_weight: float,
        one_pass: bool = False,
    ) -> Generator[tuple[Any, np.ndarray], None, None]:
        """Yield one reference-sized score row per query group.

        Missing group values form one group. By default each group is scored in
        its own pass over the projection, so one row is held at a time;
        ``one_pass`` scores every group in a single pass instead.
        """
        groups, scale, weight = self._mapping_score_settings(
            loaded,
            target_groups,
            log_transform=log_transform,
            multiplier=multiplier,
            weighted=weighted,
            fixed_weight=fixed_weight,
        )
        codes, labels = pd.factorize(groups, use_na_sentinel=False)
        batches = (
            [np.arange(len(labels))]
            if one_pass
            else [np.array([code]) for code in range(len(labels))]
        )
        indices, distances, uninformative = self._projection_arrays(loaded.ref)
        n_k = int(indices.shape[1])
        block_size = self._projection_block_size(indices)

        from ...mapping.confidence import (
            add_mapping_scores,
            finish_mapping_scores,
            mapping_score_weights,
        )

        for batch in batches:
            rows = np.full(len(labels), -1, dtype=np.int64)
            rows[batch] = np.arange(len(batch))
            scores = np.zeros(
                (len(batch), loaded.reference.selected_cell_count),
                dtype=np.float64,
            )
            scored_rows = np.zeros(len(batch), dtype=np.int64)
            for start in range(0, loaded.n_cells, block_size):
                stop = min(start + block_size, loaded.n_cells)
                block_rows = rows[codes[start:stop]]
                # Rows of other groups and uninformative query cells add nothing.
                skip = (block_rows < 0) | np.asarray(
                    uninformative[start:stop], dtype=bool
                )
                if skip.all():
                    continue
                keep = ~skip
                block_indices = np.asarray(indices[start:stop])[keep]
                if weighted:
                    block_weights = mapping_score_weights(
                        np.asarray(distances[start:stop])[keep]
                    )
                else:
                    block_weights = np.full(
                        block_indices.shape, weight, dtype=np.float64
                    )
                add_mapping_scores(
                    scores,
                    scored_rows,
                    block_indices,
                    block_weights,
                    skip=skip,
                    groups=block_rows,
                )
            finish_mapping_scores(
                scores,
                scored_rows,
                n_neighbors=n_k,
                multiplier=scale,
                log_transform=log_transform,
            )
            yield from zip(labels[batch], scores, strict=True)

    def run_label_transfer(
        self,
        projection: ArtifactRef,
        *,
        reference: MappingReference,
        reference_labels: str | ArtifactRef,
        threshold_fraction: float = 0.5,
        max_distance: float | None = None,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Transfer reference labels to projected query cells and save the result.

        The reference labels are first frozen into this query datastore as a
        ``reference_labels`` artifact, so a later change to the reference
        cannot change a saved transfer. Each query cell then takes the label
        with the largest share of its neighbors' inverse-distance weight. A
        cell abstains, and its label is missing, when it is uninformative, no
        neighbor has a usable label, two labels tie, the winning share is
        below ``threshold_fraction``, or its nearest reference neighbor is
        farther than ``max_distance``. A high vote share means that the
        neighbors agree; it is not a calibrated probability.

        A complete transfer with the same projection, reference labels, and
        decision rule is reused.

        Args:
            projection: Query projection returned by :meth:`run_mapping`.
            reference: The mapping reference that the projection used.
            reference_labels: A cell-metadata column of the reference
                datastore, or a cell-label artifact in it, such as
                ``cluster_labels``, ``cluster_cut``, or ``smart_label``.
            threshold_fraction: Smallest winning vote share that assigns a
                label.
            max_distance: Largest distance to the nearest reference neighbor
                that assigns a label. ``None`` sets no distance limit.
            invalidate_cache: Compute a new transfer even when a matching one
                exists. Frozen reference labels with the same fingerprint are
                still reused, because they are an exact copy.

        Returns:
            Reference to the immutable ``label_transfer`` artifact. Load it
            with :meth:`get_label_transfer`, or use it wherever a cell-label
            artifact is accepted, such as ``color_by`` of an embedding plot.

        Raises:
            PermissionError: If no matching transfer exists and the query
                datastore is not opened with ``zarr_mode='r+'``.
        """
        if not isinstance(projection, ArtifactRef):
            raise TypeError("projection must be an ArtifactRef")
        if not isinstance(reference, MappingReference):
            raise TypeError("reference must be a MappingReference")
        source = validate_reference_label_source(reference_labels)
        threshold = _finite_in_range(
            threshold_fraction,
            "threshold_fraction must be between zero and one",
            low=0.0,
            high=1.0,
        )
        distance_limit = (
            None
            if max_distance is None
            else _finite_in_range(
                max_distance,
                "max_distance must be finite and non-negative",
                low=0.0,
            )
        )
        if not isinstance(invalidate_cache, bool):
            raise TypeError("invalidate_cache must be a boolean")
        loaded = self.get_mapping_result(
            projection,
            reference=reference,
            load_arrays=False,
        )
        frozen_labels = plan_reference_labels(self.zw, loaded.reference, source)
        indices, distances, uninformative = self._projection_arrays(loaded.ref)
        transfer = plan_label_transfer(
            self.zw,
            projection=loaded.ref,
            cell_selection=loaded.cell_selection,
            reference_labels=frozen_labels.ref,
            categories=frozen_labels.categories,
            n_cells=loaded.n_cells,
            n_neighbors=int(indices.shape[1]),
            threshold_fraction=threshold,
            max_distance=distance_limit,
            invalidate_cache=invalidate_cache,
        )
        if transfer.reused:
            return transfer.ref
        self._require_writable("run_label_transfer")
        write_reference_labels(self.zw, frozen_labels, profile=self.storageProfile)
        reference_codes = frozen_labels.codes
        distance_percentiles = ReferenceDistancePercentiles.from_reference(
            loaded.reference
        )
        block_size = self._projection_block_size(indices)

        def blocks() -> Generator[tuple[int, LabelTransferBlock], None, None]:
            for start in range(0, loaded.n_cells, block_size):
                stop = min(start + block_size, loaded.n_cells)
                yield (
                    start,
                    transfer_label_block(
                        reference_codes[np.asarray(indices[start:stop])],
                        np.asarray(distances[start:stop]),
                        np.asarray(uninformative[start:stop], dtype=bool),
                        threshold_fraction=threshold,
                        max_distance=distance_limit,
                        distance_percentiles=distance_percentiles,
                    ),
                )

        return write_label_transfer(
            self.zw,
            transfer,
            blocks(),
            chunk_rows=block_size,
            profile=self.storageProfile,
        )

    def get_label_transfer(
        self,
        transfer: ArtifactRef,
        *,
        load_votes: bool = False,
    ) -> LabelTransferResult:
        """Load one complete label transfer from this query datastore.

        Loading reads only this datastore, so the reference datastore is not
        needed and later changes to it do not change the result.

        Args:
            transfer: Label transfer returned by :meth:`run_label_transfer`.
            load_votes: Also load each cell's neighbor votes, which
                ``label_vote_shares`` and ``prediction_sets`` need. The vote
                matrices hold one column per saved neighbor, so they are
                skipped by default.

        Returns:
            The transferred labels, their evidence, and the exact inputs.
        """
        if not isinstance(transfer, ArtifactRef):
            raise TypeError("transfer must be an ArtifactRef")
        return load_label_transfer(self.zw, transfer, load_votes=load_votes)
