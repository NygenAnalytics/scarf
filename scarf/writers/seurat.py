from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
import zarr
from scipy.sparse import coo_matrix, issparse

from ..readers import SeuratReader
from ..readers.seurat import (
    SeuratAssay,
    SeuratMetadata,
    SeuratMetadataColumn,
    SeuratMembership,
    SeuratNotice,
    SeuratNumericVector,
    SeuratRMatrix,
    SeuratReduction,
)
from ..storage.count_matrix import CountMatrixPolicy
from ..storage.artifact_writer import (
    ArrayRequirement,
    finish_artifact,
    plan_artifact,
    start_artifact,
)
from ..storage.io_policy import StorageIoPolicy
from ..storage.profiles import StorageProfile, ZarrLocation
from ..storage.refs import ArtifactRef


_RESERVED_METADATA_COLUMNS = frozenset({"I", "ids", "names"})
_DEFAULT_BLOCK_ROWS = 65_536


@dataclass(frozen=True, slots=True)
class SeuratImportResult:
    """Result of writing a Seurat object into a Scarf Zarr store.

    Attributes:
        assayNames: Assay groups written to the destination store.
        defaultAssay: Active assay selected from the Seurat object.
        cellSelection: Artifact for the imported cell filter column.
        activeIdentity: Imported Seurat active identity as immutable cluster labels.
        reductionArtifacts: Imported reductions keyed by result name.
        notices: Non-fatal import notices collected from the reader.
    """

    assayNames: tuple[str, ...]
    defaultAssay: str
    cellSelection: ArtifactRef
    activeIdentity: ArtifactRef
    reductionArtifacts: Mapping[str, ArtifactRef]
    notices: tuple[SeuratNotice, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "reductionArtifacts",
            MappingProxyType(dict(self.reductionArtifacts)),
        )

    @property
    def artifactRefs(self) -> tuple[ArtifactRef, ...]:
        return (
            self.cellSelection,
            self.activeIdentity,
            *self.reductionArtifacts.values(),
        )


def _string_blocks(
    values: Sequence[str],
    block_rows: int,
) -> Iterator[tuple[str, ...]]:
    read_block = getattr(values, "read_block", None)
    for start in range(0, len(values), block_rows):
        stop = min(start + block_rows, len(values))
        block = read_block(start, stop) if callable(read_block) else values[start:stop]
        yield tuple(_decode_text(value) for value in block)


def _bounded_string_dtype(
    values: Sequence[str],
    block_rows: int,
) -> np.dtype[Any]:
    maximum = 1
    for block in _string_blocks(values, block_rows):
        maximum = max(maximum, max((len(value) for value in block), default=1))
    return np.dtype(f"U{maximum}")


def _decode_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


class SeuratToZarr:
    """Convert a serialized Seurat object into a Scarf Zarr store.

    Args:
        reader: Open ``SeuratReader`` for the source ``.rds`` file.
        zarr_loc: Destination Zarr path or store.
        workspace: Workspace name in the destination store. None uses the
                   legacy layout without a workspace group.
        storage_options: Backend options passed when opening the Zarr store.
        mem_budget: Memory available to the conversion. Accepts bytes, a
                    suffixed size (e.g. '8G'), or a fraction of total system
                    memory (e.g. '0.6'). When None, auto-detected.
        nthreads: Worker count for write-time concurrency. When None,
                  auto-detected.
        profile: Zarr encoding profile (``fast_local`` or ``cloud``). When
                 None, chosen from the destination location.
        policy: Count-matrix geometry policy. When None, the default
                unitBytes and chunkBytes plan is used.
        io: Optional explicit read, compute, and write widths. Unset values
            stay under automatic planning.
    """

    def __init__(
        self,
        reader: SeuratReader,
        zarr_loc: ZarrLocation,
        workspace: str | None = None,
        storage_options: dict[str, Any] | None = None,
        mem_budget: int | str | None = None,
        nthreads: int | None = None,
        profile: StorageProfile | None = None,
        policy: CountMatrixPolicy | None = None,
        io: StorageIoPolicy | None = None,
    ) -> None:
        from ..storage.budget import resolve_budget
        from ..storage.schema import (
            create_empty_cell_data,
            create_empty_zarr_count_assay,
            validate_assay_name,
        )
        from ..storage.stores import load_zarr

        resources = resolve_budget(mem_budget, nthreads)
        inspection = reader.inspection
        assay_names = tuple(reader.assayNames)
        for assay_name in assay_names:
            validate_assay_name(assay_name)
        if inspection.activeAssay not in assay_names:
            raise ValueError(
                f"Active assay {inspection.activeAssay!r} is not selected for import"
            )

        assays = tuple(reader.get_assay(name) for name in assay_names)
        blocked_reductions = tuple(
            item for item in inspection.reductions if not item.importable
        )
        if blocked_reductions:
            details = ", ".join(
                f"{item.name} "
                f"({item.blockingDiagnostic.code if item.blockingDiagnostic is not None else 'unsupported'})"
                for item in blocked_reductions
            )
            raise ValueError(f"Selected reductions cannot be imported: {details}")
        reductions = tuple(
            reader.get_reduction(item.name) for item in inspection.reductions
        )
        imported_assays = set(assay_names)
        missing_reduction_assays = tuple(
            reduction.name
            for reduction in reductions
            if reduction.assayUsed not in imported_assays
        )
        if missing_reduction_assays:
            raise ValueError(
                "Selected reductions reference assays that are not selected for "
                f"import: {', '.join(missing_reduction_assays)}"
            )
        active_identity = reader.activeIdentity
        self._validate_metadata_names(reader.cellMetadata, "cell")
        self._validate_metadata_name(active_identity.name, "cell")
        membership_names = {
            f"{assay.name}_I"
            for assay in assays
            if not assay.cellMembership.allIncluded
        }
        cell_names = set(reader.cellMetadata.columnNames)
        conflicts = sorted(cell_names.intersection(membership_names))
        if conflicts:
            raise ValueError(
                "Assay membership columns conflict with cell metadata: "
                + ", ".join(conflicts)
            )
        for assay in assays:
            self._validate_metadata_names(
                assay.featureMetadata,
                f"{assay.name} feature",
            )
            dtype = np.dtype(assay.counts.dtype)
            if dtype.kind not in "biuf":
                raise TypeError(
                    f"Assay {assay.name!r} counts use unsupported dtype {dtype}"
                )
            expected_shape = (len(assay.featureIds), len(reader.cellIds))
            if tuple(assay.counts.shape) != expected_shape:
                raise ValueError(
                    f"Assay {assay.name!r} has shape {assay.counts.shape}, "
                    f"expected {expected_shape}"
                )

        source_digest = bytes.fromhex(reader.document.source.source_sha256)
        if len(source_digest) != 32:
            raise ValueError("Seurat source SHA-256 digest must contain 32 bytes")
        string_block_rows = max(
            1,
            min(
                _DEFAULT_BLOCK_ROWS,
                int(resources.memoryBytes) // (8 * 64),
            ),
        )
        cell_dtype = _bounded_string_dtype(reader.cellIds, string_block_rows)
        feature_dtypes = {
            assay.name: _bounded_string_dtype(assay.featureIds, string_block_rows)
            for assay in assays
        }

        self.reader = reader
        self.workspace = workspace
        self.storageOptions = storage_options
        self.resources = resources
        from ..storage.profiles import resolve_storage_profile

        self.profile = resolve_storage_profile(zarr_loc, profile)
        self.policy = policy
        self.io = io
        self.assayNames = assay_names
        self.defaultAssay = inspection.activeAssay
        self._assays = assays
        self._reductions = reductions
        self._activeIdentity = active_identity
        self._sourceDigest = source_digest
        self._notices = self._collect_notices(inspection.notices, assays, reductions)
        self._residentSourceBytes = sum(
            max(0, int(assay.counts.resident_bytes)) for assay in assays
        )
        self._lastImportPlans: dict[str, Any] = {}
        self._lastDenseBatchRows: dict[str, int] = {}

        self.z = load_zarr(
            zarr_loc=zarr_loc,
            mode="w",
            storage_options=storage_options,
        )
        self.root = (
            self.z
            if workspace is None
            else self.z.create_group(workspace, overwrite=True)
        )
        self.root.attrs["complete"] = False
        self.root.attrs["scarf:import_source"] = "seurat"
        self.root.attrs["scarf:import_complete"] = False
        self.root.attrs["scarf:import_source_sha256"] = (
            reader.document.source.source_sha256
        )
        self.root.attrs["scarf:import_payload_sha256"] = (
            reader.document.source.payload_sha256
        )

        self.cellData = create_empty_cell_data(
            self.z,
            workspace,
            len(reader.cellIds),
            cell_dtype,
            cell_dtype,
            profile=self.profile,
        )
        self.counts: dict[str, zarr.Array] = {}
        self.featureData: dict[str, zarr.Group] = {}
        for assay in assays:
            feature_dtype = feature_dtypes[assay.name]
            counts, feature_data = create_empty_zarr_count_assay(
                self.z,
                assay.name,
                workspace,
                len(reader.cellIds),
                len(assay.featureIds),
                feature_dtype,
                feature_dtype,
                dtype=assay.counts.dtype,
                profile=self.profile,
                policy=policy,
            )
            self.counts[assay.name] = counts
            self.featureData[assay.name] = feature_data
        self.root.attrs["defaultAssay"] = self.defaultAssay

    @staticmethod
    def _validate_metadata_name(name: str, axis: str) -> None:
        if name in _RESERVED_METADATA_COLUMNS:
            raise ValueError(f"{axis} metadata column {name!r} is reserved")
        if name.startswith("__scarf_missing__"):
            raise ValueError(
                f"{axis} metadata column {name!r} uses Scarf's internal prefix"
            )

    @classmethod
    def _validate_metadata_names(cls, metadata: SeuratMetadata, axis: str) -> None:
        names = metadata.columnNames
        if len(set(names)) != len(names):
            raise ValueError(f"{axis} metadata contains duplicate column names")
        for name in names:
            cls._validate_metadata_name(name, axis)
        generated_masks = {f"__scarf_missing__{name}" for name in names}
        overlap = generated_masks.intersection(names)
        if overlap:
            raise ValueError(
                f"{axis} metadata conflicts with generated missing masks: "
                + ", ".join(sorted(overlap))
            )

    @staticmethod
    def _collect_notices(
        root_notices: tuple[SeuratNotice, ...],
        assays: tuple[SeuratAssay, ...],
        reductions: tuple[SeuratReduction, ...],
    ) -> tuple[SeuratNotice, ...]:
        return (
            *root_notices,
            *(notice for assay in assays for notice in assay.notices),
            *(notice for reduction in reductions for notice in reduction.notices),
        )

    def dump(self, batch_size: int | None = None) -> SeuratImportResult:
        """Write assays, RNA ``countsT``, and importable reductions.

        Args:
            batch_size: Number of source cells per batch. By default, a
                        destination-aligned value is selected within the
                        memory budget.

        Returns:
            Imported assay names, cell selection, and reduction artifacts.
        """
        if batch_size is not None and (
            isinstance(batch_size, bool) or int(batch_size) <= 0
        ):
            raise ValueError("batch_size must be positive")
        requested_rows = None if batch_size is None else int(batch_size)

        self.reader.inspection
        self.root.attrs["complete"] = False
        self.root.attrs["scarf:import_complete"] = False
        try:
            metadata_rows = self._bounded_block_rows(
                requested_rows,
                row_bytes=64,
            )
            self._write_cell_data(metadata_rows)
            for assay in self._assays:
                self._write_feature_data(assay, metadata_rows)
            for assay in self._assays:
                self._write_counts(assay, requested_rows)
            from .counts_t import finalize_writer_counts_t

            for assay in self._assays:
                finalize_writer_counts_t(
                    self.z,
                    assay.name,
                    self.workspace,
                    assay_type=assay.name,
                    resources=self.resources,
                    profile=self.profile,
                    policy=self.policy,
                    io=self.io,
                )
            cell_selection = self._write_cell_selection()
            active_identity = self._write_active_identity(
                cell_selection,
                metadata_rows,
            )
            reduction_artifacts = self._write_reductions(
                cell_selection,
                requested_rows,
            )
        except BaseException:
            self.root.attrs["complete"] = False
            self.root.attrs["scarf:import_complete"] = False
            raise
        self.root.attrs["complete"] = True
        self.root.attrs["scarf:import_complete"] = True
        return SeuratImportResult(
            assayNames=self.assayNames,
            defaultAssay=self.defaultAssay,
            cellSelection=cell_selection,
            activeIdentity=active_identity,
            reductionArtifacts=reduction_artifacts,
            notices=self._notices,
        )

    def _bounded_block_rows(
        self,
        requested: int | None,
        *,
        row_bytes: int,
    ) -> int:
        bytes_per_row = max(1, int(row_bytes))
        memory_rows = max(
            1,
            int(self.resources.memoryBytes) // (8 * bytes_per_row),
        )
        preferred = _DEFAULT_BLOCK_ROWS if requested is None else requested
        return int(max(1, min(int(preferred), memory_rows)))

    def _write_cell_data(self, block_rows: int) -> None:
        self._write_string_axis(
            self.cellData["ids"],
            self.cellData["names"],
            self.reader.cellIds,
            block_rows,
        )
        for column in self.reader.cellMetadata.columns:
            self._write_metadata_column(self.cellData, column, block_rows)
        for assay in self._assays:
            if assay.cellMembership.allIncluded:
                continue
            column_name = f"{assay.name}_I"
            output = self._create_boolean_column(
                self.cellData,
                column_name,
                assay.cellMembership,
                block_rows,
            )
            output.attrs["assay"] = assay.name
            output.attrs["role"] = "assay_membership"

    def _write_feature_data(self, assay: SeuratAssay, block_rows: int) -> None:
        group = self.featureData[assay.name]
        self._write_string_axis(
            group["ids"],
            group["names"],
            assay.featureIds,
            block_rows,
        )
        for column in assay.featureMetadata.columns:
            self._write_metadata_column(group, column, block_rows)

    @staticmethod
    def _write_string_axis(
        ids: Any,
        names: Any,
        values: Sequence[str],
        block_rows: int,
    ) -> None:
        if int(ids.shape[0]) != len(values) or int(names.shape[0]) != len(values):
            raise ValueError("String axis length does not match its destination")
        start = 0
        for values_block in _string_blocks(values, block_rows):
            stop = start + len(values_block)
            block = np.asarray(values_block, dtype=ids.dtype)
            ids[start:stop] = block
            names[start:stop] = block.astype(names.dtype, copy=False)
            start = stop

    def _metadata_dtype(
        self,
        column: SeuratMetadataColumn,
        block_rows: int,
    ) -> np.dtype[Any]:
        if column.kind == "logical":
            return np.dtype(bool)
        if column.kind == "integer":
            return np.dtype(np.int64)
        if column.kind == "real":
            return np.dtype(np.float64)
        if column.kind == "factor":
            return _bounded_string_dtype(column.levels, block_rows)
        if column.kind != "character":
            raise TypeError(f"Unsupported Seurat metadata kind {column.kind!r}")
        maximum = 1
        for start in range(0, column.length, block_rows):
            stop = min(start + block_rows, column.length)
            block = column.read_block(start, stop)
            values = block.values
            if not isinstance(values, tuple):
                raise TypeError("Character metadata did not return string values")
            maximum = max(
                maximum,
                max((len(_decode_text(value)) for value in values), default=1),
            )
        return np.dtype(f"U{maximum}")

    def _metadata_blocks(
        self,
        column: SeuratMetadataColumn,
        dtype: np.dtype[Any],
        block_rows: int,
    ) -> Iterator[Any]:
        from ..storage.arrays import MetadataBlock

        for start in range(0, column.length, block_rows):
            stop = min(start + block_rows, column.length)
            block = column.read_block(start, stop)
            missing = np.asarray(block.missing, dtype=bool)
            if column.kind == "character":
                if not isinstance(block.values, tuple):
                    raise TypeError("Character metadata did not return string values")
                values = np.asarray(
                    [_decode_text(value) for value in block.values],
                    dtype=dtype,
                )
            elif column.kind == "factor":
                codes = np.asarray(block.values)
                values = np.asarray(
                    [
                        "" if missing[index] else column.levels[int(code) - 1]
                        for index, code in enumerate(codes)
                    ],
                    dtype=dtype,
                )
            else:
                values = np.asarray(block.values, dtype=dtype)
                if column.kind == "logical":
                    values[missing] = False
                elif column.kind == "integer":
                    values[missing] = 0
                elif column.kind == "real":
                    values[missing] = np.nan
            yield MetadataBlock(start, values, missing)

    def _write_metadata_column(
        self,
        group: zarr.Group,
        column: SeuratMetadataColumn,
        block_rows: int,
        *,
        name: str | None = None,
    ) -> zarr.Array:
        from ..storage.arrays import create_streamed_metadata_column

        dtype = self._metadata_dtype(column, block_rows)
        output = create_streamed_metadata_column(
            group,
            column.name if name is None else name,
            shape=column.length,
            dtype=dtype,
            blocks=self._metadata_blocks(column, dtype, block_rows),
            overwrite=True,
            chunkSize=min(_DEFAULT_BLOCK_ROWS, max(1, column.length)),
            hasMissing=True,
            profile=self.profile,
        )
        if column.kind == "factor":
            output.attrs["levels"] = list(column.levels)
            output.attrs["ordered"] = bool(column.ordered)
        return output

    def _write_active_identity(
        self,
        cell_selection: ArtifactRef,
        block_rows: int,
    ) -> ArtifactRef:
        """Store Seurat's analytical active identity without a live column."""
        column = self._activeIdentity
        dtype = self._metadata_dtype(column, block_rows)
        missing_name = "__scarf_missing__values"
        planned = plan_artifact(
            self.root,
            scope="assay",
            assay=self.defaultAssay,
            kind="cluster_labels",
            operation="import_active_identity",
            parameters={
                "source": "seurat",
                "source_key": "active.ident",
                "levels": list(column.levels),
                "ordered": bool(column.ordered),
            },
            inputs={
                "source_digest": self._sourceDigest,
                "cell_selection": cell_selection,
            },
            execution_options={"block_rows": block_rows},
            required_arrays=(
                ArrayRequirement("values", shape=(column.length,), dtype=dtype),
                ArrayRequirement(
                    missing_name,
                    shape=(column.length,),
                    dtype=bool,
                ),
            ),
        )
        if planned.reused:
            return planned.ref
        group = start_artifact(self.root, planned)
        values = self._write_metadata_column(
            group,
            column,
            block_rows,
            name="values",
        )
        if values.attrs.get("missing_mask") != missing_name:
            raise RuntimeError("Active identity missing-mask link is malformed")
        finish_artifact(group, planned)
        return planned.ref

    def _create_boolean_column(
        self,
        group: zarr.Group,
        name: str,
        values: SeuratMembership,
        block_rows: int,
    ) -> zarr.Array:
        from ..storage.arrays import MetadataBlock, create_streamed_metadata_column

        read_block = values.read_block
        return create_streamed_metadata_column(
            group,
            name,
            shape=len(values),
            dtype=bool,
            blocks=(
                MetadataBlock(
                    start,
                    read_block(start, min(start + block_rows, len(values))),
                )
                for start in range(0, len(values), block_rows)
            ),
            overwrite=True,
            chunkSize=min(_DEFAULT_BLOCK_ROWS, max(1, len(values))),
            profile=self.profile,
        )

    def _source_staging_peak(
        self,
        source: Any,
        rows: int,
    ) -> int:
        n_cells = int(source.shape[1])
        width = max(1, min(int(rows), max(1, n_cells)))
        peak = 0
        for start in range(0, n_cells, width):
            stop = min(start + width, n_cells)
            estimate = source.estimate_read_memory(start, stop)
            peak = max(
                peak,
                max(0, int(estimate.workingBytes)) + max(0, int(estimate.outputBytes)),
            )
        return peak

    def _write_counts(
        self,
        assay: SeuratAssay,
        requested_rows: int | None,
    ) -> None:
        source = assay.counts
        destination = self.counts[assay.name]
        n_cells = len(self.reader.cellIds)
        if n_cells == 0:
            return
        if source.is_sparse:
            self._write_sparse_counts(
                assay.name,
                source,
                destination,
                requested_rows,
            )
        else:
            self._write_dense_counts(
                assay.name,
                source,
                destination,
                requested_rows,
            )

    def _write_sparse_counts(
        self,
        assay_name: str,
        source: Any,
        destination: zarr.Array,
        requested_rows: int | None,
    ) -> None:
        from ..storage.sharding import (
            accumulate_sparse_to_shards,
            resolve_sparse_import_batch,
        )

        n_cells = int(source.shape[1])
        n_features = int(source.shape[0])
        staging_cache: dict[int, int] = {}

        def staging(rows: int) -> int:
            width = max(1, min(int(rows), max(1, n_cells)))
            if width not in staging_cache:
                staging_cache[width] = self._source_staging_peak(source, width)
            return staging_cache[width]

        def max_window_nnz(rows: int) -> int:
            width = max(0, min(int(rows), n_cells))
            if width == 0:
                return 0
            dense_bound = width * n_features
            estimated_values = (
                staging(width) + max(1, source.dtype.itemsize) - 1
            ) // max(1, source.dtype.itemsize)
            return int(min(dense_bound, max(0, estimated_values)))

        plan = resolve_sparse_import_batch(
            (destination,),
            nRows=n_cells,
            resources=self.resources,
            maxWindowNnz=max_window_nnz,
            sourceDtype=source.dtype,
            batchRows=requested_rows,
            residentBytes=self._residentSourceBytes,
            producerStagingBytes=staging,
        )
        self._lastImportPlans[assay_name] = plan
        self._lastImportPlan = plan

        def batches() -> Iterator[coo_matrix]:
            for start in range(0, n_cells, plan.batchRows):
                stop = min(start + plan.batchRows, n_cells)
                raw = source.read_cells(start, stop)
                block = (
                    raw.tocoo(copy=False)
                    if issparse(raw)
                    else coo_matrix(np.asarray(raw))
                )
                expected = (stop - start, int(destination.shape[1]))
                if block.shape != expected:
                    raise ValueError(
                        f"Assay {assay_name!r} source returned shape "
                        f"{block.shape}, expected {expected}"
                    )
                yield block

        rows = accumulate_sparse_to_shards(
            destination,
            batches(),
            resources=self.resources,
            residentBytes=self._residentSourceBytes,
            producerReserveBytes=plan.producerReserveBytes,
            msg=f"Writing {assay_name} counts",
            io=self.io,
        )
        if rows != n_cells:
            raise ValueError(
                f"Assay {assay_name!r} wrote {rows} count rows, expected {n_cells}"
            )

    @staticmethod
    def _dense_write_reserve(destination: zarr.Array) -> int:
        from ..storage.layout import array_shard_rows

        rows = min(int(destination.shape[0]), array_shard_rows(destination))
        columns = max(1, int(destination.shape[1]))
        itemsize = max(1, int(np.dtype(destination.dtype).itemsize))
        chunk_rows = min(max(1, rows), int(destination.chunks[0]))
        chunk_columns = min(columns, int(destination.chunks[1]))
        chunks = (
            (max(1, rows) + chunk_rows - 1)
            // chunk_rows
            * ((columns + chunk_columns - 1) // chunk_columns)
        )
        band_bytes = max(1, rows) * columns * itemsize
        chunk_bytes = chunk_rows * chunk_columns * itemsize
        encoded_chunk = chunk_bytes + chunk_bytes // 128 + 1024
        return int(
            band_bytes + chunk_bytes + 2 * chunks * encoded_chunk + chunks * 16 + 1024
        )

    def _resolve_dense_batch_rows(
        self,
        source: Any,
        destination: zarr.Array,
        requested_rows: int | None,
    ) -> tuple[int, int]:
        from ..storage.layout import array_shard_rows
        from ..storage.partition import affordable_width

        n_cells = int(source.shape[1])
        task_reserve = self._dense_write_reserve(destination)
        staging_cache: dict[int, int] = {}

        def staging(rows: int) -> int:
            width = max(1, min(int(rows), n_cells))
            if width not in staging_cache:
                staging_cache[width] = self._source_staging_peak(source, width)
            return staging_cache[width]

        def fits(rows: int) -> bool:
            required = self._residentSourceBytes + staging(rows) + task_reserve
            return bool(required <= int(self.resources.memoryBytes))

        if requested_rows is not None:
            rows = min(requested_rows, n_cells)
            if not fits(rows):
                raise MemoryError(
                    "Dense Seurat import cannot fit the requested source batch and "
                    "one destination row band within mem_budget"
                )
        else:
            preferred = min(n_cells, array_shard_rows(destination))
            rows = affordable_width(fits, preferred)
            if rows < 1:
                raise MemoryError(
                    "Dense Seurat import cannot fit one source row and one "
                    "destination row band within mem_budget"
                )
        return rows, staging(rows)

    def _write_dense_counts(
        self,
        assay_name: str,
        source: Any,
        destination: zarr.Array,
        requested_rows: int | None,
    ) -> None:
        from ..storage.budget import ResourceBudget
        from ..storage.sharding import write_dense_from_row_batches

        n_cells = int(source.shape[1])
        rows, producer_reserve = self._resolve_dense_batch_rows(
            source,
            destination,
            requested_rows,
        )
        self._lastDenseBatchRows[assay_name] = rows
        writer_memory = (
            self.resources.memoryBytes - self._residentSourceBytes - producer_reserve
        )
        if writer_memory < 1:
            raise MemoryError("Dense Seurat import has no memory left for Zarr writes")
        writer_resources = ResourceBudget(
            memoryBytes=writer_memory,
            workers=self.resources.workers,
        )

        def batches() -> Iterator[np.ndarray]:
            for start in range(0, n_cells, rows):
                stop = min(start + rows, n_cells)
                raw = source.read_cells(start, stop)
                block = raw.toarray() if issparse(raw) else np.asarray(raw)
                expected = (stop - start, int(destination.shape[1]))
                if block.shape != expected:
                    raise ValueError(
                        f"Assay {assay_name!r} source returned shape "
                        f"{block.shape}, expected {expected}"
                    )
                yield np.ascontiguousarray(block, dtype=destination.dtype)

        written = write_dense_from_row_batches(
            destination,
            batches(),
            dtype=destination.dtype,
            msg=f"Writing {assay_name} counts",
            resources=writer_resources,
            io=self.io,
        )
        if written != n_cells:
            raise ValueError(
                f"Assay {assay_name!r} wrote {written} count rows, expected {n_cells}"
            )

    def _write_cell_selection(self) -> ArtifactRef:
        from ..storage.selections import resolve_stored_selection_artifact

        ref = resolve_stored_selection_artifact(
            self.root,
            table_path="cellData",
            id_column="ids",
            source_column="I",
            scope="datastore",
            kind="cell_selection",
            operation="import_cell_selection",
            parameters={"source": "seurat"},
            inputs={"source_digest": self._sourceDigest},
        )
        return ref

    @staticmethod
    def _floating_dtype(dtype: Any) -> np.dtype[Any]:
        source: np.dtype[Any] = np.dtype(dtype)
        if source.kind == "f":
            return np.dtype(source.str)
        if source.kind in "biu":
            return np.dtype(np.float64)
        raise TypeError(f"Reduction payload uses unsupported dtype {source}")

    @staticmethod
    def _matrix_blocks(
        matrix: SeuratRMatrix,
        block_rows: int,
        dtype: np.dtype[Any],
    ) -> Iterator[np.ndarray]:
        for start in range(0, matrix.shape[0], block_rows):
            stop = min(start + block_rows, matrix.shape[0])
            yield np.asarray(matrix.read_rows(start, stop), dtype=dtype)

    @staticmethod
    def _vector_blocks(
        vector: SeuratNumericVector,
        block_rows: int,
        dtype: np.dtype[Any],
    ) -> Iterator[np.ndarray]:
        for start in range(0, vector.length, block_rows):
            stop = min(start + block_rows, vector.length)
            yield np.asarray(vector.read_block(start, stop), dtype=dtype)

    @classmethod
    def _fingerprint_matrix(
        cls,
        matrix: SeuratRMatrix,
        block_rows: int,
        dtype: np.dtype[Any],
    ) -> str:
        from ..storage.artifacts import ValueFingerprintBuilder

        builder = ValueFingerprintBuilder()
        builder.begin_array("values", matrix.shape, dtype)
        start = 0
        for block in cls._matrix_blocks(matrix, block_rows, dtype):
            builder.update_array_block("values", (start, 0), block)
            start += int(block.shape[0])
        builder.end_array("values")
        return str(builder.hexdigest())

    @classmethod
    def _fingerprint_vector(
        cls,
        vector: SeuratNumericVector,
        block_rows: int,
        dtype: np.dtype[Any],
    ) -> str:
        from ..storage.artifacts import ValueFingerprintBuilder

        builder = ValueFingerprintBuilder()
        builder.begin_array("values", (vector.length,), dtype)
        start = 0
        for block in cls._vector_blocks(vector, block_rows, dtype):
            builder.update_array_block("values", (start,), block)
            start += int(block.shape[0])
        builder.end_array("values")
        return str(builder.hexdigest())

    @staticmethod
    def _fingerprint_feature_ids(
        values: Sequence[str],
        block_rows: int,
    ) -> str:
        from ..storage.artifacts import ValueFingerprintBuilder

        dtype = _bounded_string_dtype(values, block_rows)
        builder = ValueFingerprintBuilder()
        builder.begin_array("values", (len(values),), dtype)
        for start in range(0, len(values), block_rows):
            stop = min(start + block_rows, len(values))
            builder.update_array_block(
                "values",
                (start,),
                np.asarray(values[start:stop], dtype=dtype),
            )
        builder.end_array("values")
        return str(builder.hexdigest())

    def _reduction_block_rows(
        self,
        reduction: SeuratReduction,
        requested_rows: int | None,
    ) -> int:
        row_bytes = max(1, reduction.cellEmbeddings.shape[1]) * max(
            1, reduction.cellEmbeddings.dtype.itemsize
        )
        return self._bounded_block_rows(requested_rows, row_bytes=row_bytes)

    def _write_reductions(
        self,
        cell_selection: ArtifactRef,
        requested_rows: int | None,
    ) -> dict[str, ArtifactRef]:
        from ..embeddings.imported import (
            write_imported_coordinates,
            write_imported_embedding,
        )

        artifacts: dict[str, ArtifactRef] = {}
        for reduction in self._reductions:
            block_rows = self._reduction_block_rows(reduction, requested_rows)
            coordinate_dtype = self._floating_dtype(reduction.cellEmbeddings.dtype)
            coordinate_fingerprint = self._fingerprint_matrix(
                reduction.cellEmbeddings,
                block_rows,
                coordinate_dtype,
            )

            def coordinate_blocks(
                matrix: SeuratRMatrix = reduction.cellEmbeddings,
                rows: int = block_rows,
                dtype: np.dtype[Any] = coordinate_dtype,
            ) -> Iterator[np.ndarray]:
                return self._matrix_blocks(matrix, rows, dtype)

            normalized_name = reduction.name.casefold()
            if normalized_name in {"umap", "tsne", "t-sne"}:
                role = "umap" if normalized_name == "umap" else "tsne"
                ref = write_imported_embedding(
                    self.root,
                    assay=reduction.assayUsed,
                    dimreduc_key=reduction.name,
                    role=role,
                    coordinates=coordinate_blocks,
                    coordinate_shape=reduction.cellEmbeddings.shape,
                    coordinate_dtype=coordinate_dtype,
                    source_digest=self._sourceDigest,
                    payload_fingerprints={"values": coordinate_fingerprint},
                    source_cell_ids=self.reader.cellIds,
                    cell_selection=cell_selection,
                    block_rows=block_rows,
                )
            else:
                payload_fingerprints = {"data": coordinate_fingerprint}
                loadings = reduction.featureLoadings
                loading_blocks = None
                loading_shape = None
                loading_dtype = None
                feature_ids: Sequence[str] | None = None
                if loadings is not None:
                    loading_dtype = self._floating_dtype(loadings.dtype)
                    payload_fingerprints["loadings"] = self._fingerprint_matrix(
                        loadings,
                        block_rows,
                        loading_dtype,
                    )
                    feature_ids = loadings.rowIds
                    payload_fingerprints["feature_ids"] = self._fingerprint_feature_ids(
                        feature_ids, block_rows
                    )
                    loading_shape = loadings.shape

                    def loading_blocks(
                        matrix: SeuratRMatrix = loadings,
                        rows: int = block_rows,
                        dtype: np.dtype[Any] = loading_dtype,
                    ) -> Iterator[np.ndarray]:
                        return self._matrix_blocks(matrix, rows, dtype)

                stdev = reduction.stdev
                stdev_blocks = None
                stdev_shape = None
                stdev_dtype = None
                if stdev is not None:
                    stdev_dtype = self._floating_dtype(stdev.dtype)
                    payload_fingerprints["stdev"] = self._fingerprint_vector(
                        stdev,
                        block_rows,
                        stdev_dtype,
                    )
                    stdev_shape = (stdev.length,)

                    def stdev_blocks(
                        vector: SeuratNumericVector = stdev,
                        rows: int = block_rows,
                        dtype: np.dtype[Any] = stdev_dtype,
                    ) -> Iterator[np.ndarray]:
                        return self._vector_blocks(vector, rows, dtype)

                ref = write_imported_coordinates(
                    self.root,
                    assay=reduction.assayUsed,
                    dimreduc_key=reduction.name,
                    role=reduction.role,
                    coordinates=coordinate_blocks,
                    coordinate_shape=reduction.cellEmbeddings.shape,
                    coordinate_dtype=coordinate_dtype,
                    source_digest=self._sourceDigest,
                    payload_fingerprints=payload_fingerprints,
                    source_cell_ids=self.reader.cellIds,
                    cell_selection=cell_selection,
                    loadings=loading_blocks,
                    loadings_shape=loading_shape,
                    loadings_dtype=loading_dtype,
                    feature_ids=feature_ids,
                    stdev=stdev_blocks,
                    stdev_shape=stdev_shape,
                    stdev_dtype=stdev_dtype,
                    block_rows=block_rows,
                )
            artifacts[reduction.name] = ref
        return artifacts


__all__ = ["SeuratImportResult", "SeuratToZarr"]
