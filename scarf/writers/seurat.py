from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

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
from ..storage.arrays import MISSING_MASK_PREFIX
from ..storage.count_dtype import count_storage_dtype
from ..storage.count_matrix import CountMatrixPolicy
from ..storage.artifact_writer import (
    ArrayRequirement,
    artifact_transaction,
    plan_artifact,
)
from ..storage.io_policy import StorageIoPolicy
from ..storage.metadata_keys import (
    RESERVED_METADATA_COLUMNS,
    is_reserved_metadata_name,
    metadata_column_key,
    metadata_column_keys,
)
from ..storage.profiles import StorageProfile, ZarrLocation
from ..storage.refs import ArtifactRef
from ..utils.arrays import canonicalize_sparse
from ..utils.count_values import CountValueRange
from ._store import (
    DEFAULT_IMPORT_BLOCK_ROWS,
    bounded_block_rows,
    decode_text,
    fingerprint_row_blocks,
    floating_payload_dtype,
    keyed_metadata_columns,
    resolve_import_cell_selection,
)

if TYPE_CHECKING:
    from ..storage.identity import CountSummary


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
    return "" if value is None else decode_text(value)


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
        policy: Count-matrix geometry policy, used exactly for every assay.
                When None, each assay uses the default policy with unitBytes
                and chunkBytes halved together until its counts write and
                countsT transpose fit ``mem_budget``. Either way, an import
                that does not fit raises MemoryError before the destination
                is created.
        io: Optional explicit read, compute, and write widths. Unset values
            stay under automatic planning.

    Construction prepares every selected assay's counts and reads them once.
    Each assay stores its counts in the dtype that
    :func:`~scarf.storage.count_dtype.count_storage_dtype` resolves from their
    canonical values, and a count that the stored dtype cannot hold raises
    instead of wrapping.
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
            # Matrix sources admit complex values, which are not counts. The
            # reader sizes every source from its feature and cell IDs.
            dtype = np.dtype(assay.counts.dtype)
            if dtype.kind not in "biuf":
                raise TypeError(
                    f"Assay {assay.name!r} counts use unsupported dtype {dtype}"
                )
        # Zarr nests a name with '/' or '\\', so such a column is stored with
        # '_' in their place; a valid name keeps its exact key.
        cell_metadata_keys = metadata_column_keys(
            reader.cellMetadata.columnNames,
            taken={*RESERVED_METADATA_COLUMNS, *membership_names},
        )
        feature_metadata_keys = {
            assay.name: metadata_column_keys(
                assay.featureMetadata.columnNames,
                taken=RESERVED_METADATA_COLUMNS,
            )
            for assay in assays
        }

        source_digest = bytes.fromhex(reader.document.source.source_sha256)
        string_block_rows = max(
            1,
            min(
                DEFAULT_IMPORT_BLOCK_ROWS,
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
        self.io = io
        self.assayNames = assay_names
        self.defaultAssay = inspection.activeAssay
        self._assays = assays
        self._reductions = reductions
        self._activeIdentity = active_identity
        self._cellMetadataKeys = cell_metadata_keys
        self._featureMetadataKeys = feature_metadata_keys
        self._sourceDigest = source_digest
        self._notices = self._collect_notices(inspection.notices, assays, reductions)
        self._lastImportPlans: dict[str, Any] = {}
        self._lastDenseBatchRows: dict[str, int] = {}
        # Every assay's dtype and layout are resolved, and a layout that does
        # not fit fails, before the destination exists.
        count_dtypes = {assay.name: self._prepare_counts(assay) for assay in assays}
        self._residentSourceBytes = self._source_resident_bytes()
        layouts = {
            assay.name: self._fit_count_layout(assay, count_dtypes[assay.name], policy)
            for assay in assays
        }

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
                dtype=count_dtypes[assay.name],
                profile=self.profile,
                policy=layouts[assay.name],
            )
            self.counts[assay.name] = counts
            self.featureData[assay.name] = feature_data
        self.root.attrs["defaultAssay"] = self.defaultAssay

    @staticmethod
    def _validate_metadata_name(name: str, axis: str) -> None:
        if name in {"", ".", ".."}:
            raise ValueError(
                f"{axis} metadata column {name!r} cannot name a Zarr array"
            )
        if not is_reserved_metadata_name(metadata_column_key(name)):
            return
        if name in RESERVED_METADATA_COLUMNS:
            raise ValueError(f"{axis} metadata column {name!r} is reserved")
        raise ValueError(
            f"{axis} metadata column {name!r} uses Scarf's internal prefix"
        )

    @classmethod
    def _validate_metadata_names(cls, metadata: SeuratMetadata, axis: str) -> None:
        # The reader rejects repeated names, and a name with the missing-mask
        # prefix fails here, so no column can collide with a generated mask.
        for name in metadata.columnNames:
            cls._validate_metadata_name(name, axis)

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
            batch_size: Number of source cells per batch, at most one
                        destination row band, which is the default.

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
        return bounded_block_rows(
            requested,
            row_bytes=row_bytes,
            memory_bytes=int(self.resources.memoryBytes),
        )

    def _write_cell_data(self, block_rows: int) -> None:
        self._write_string_axis(
            self.cellData["ids"],
            self.cellData["names"],
            self.reader.cellIds,
            block_rows,
        )
        for key, column in keyed_metadata_columns(
            ((column.name, column) for column in self.reader.cellMetadata.columns),
            self._cellMetadataKeys,
            "cell",
        ):
            self._write_metadata_column(self.cellData, column, block_rows, name=key)
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
        for key, column in keyed_metadata_columns(
            ((column.name, column) for column in assay.featureMetadata.columns),
            self._featureMetadataKeys[assay.name],
            f"{assay.name} feature",
        ):
            self._write_metadata_column(group, column, block_rows, name=key)

    @staticmethod
    def _write_string_axis(
        ids: Any,
        names: Any,
        values: Sequence[str],
        block_rows: int,
    ) -> None:
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
        # The reader's remaining kind is character, whose blocks hold strings.
        maximum = 1
        for start in range(0, column.length, block_rows):
            stop = min(start + block_rows, column.length)
            block = column.read_block(start, stop)
            maximum = max(
                maximum,
                max((len(_decode_text(value)) for value in block.values), default=1),
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
                else:
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
            chunkSize=min(DEFAULT_IMPORT_BLOCK_ROWS, max(1, column.length)),
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
        missing_name = f"{MISSING_MASK_PREFIX}values"
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
        with artifact_transaction(self.root, planned) as group:
            self._write_metadata_column(
                group,
                column,
                block_rows,
                name="values",
            )
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
            chunkSize=min(DEFAULT_IMPORT_BLOCK_ROWS, max(1, len(values))),
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

    def _source_resident_bytes(self) -> int:
        """Return the bytes that the count sources of every assay hold."""
        return sum(max(0, int(assay.counts.resident_bytes)) for assay in self._assays)

    def _source_estimates(
        self, source: Any
    ) -> tuple[Callable[[int], int], Callable[[int], int]]:
        """Return the read staging and the stored values of a window of cells.

        The staging of a width costs one source estimate per window of that
        width, and is computed once; the values are bounded from it.
        """
        n_cells = int(source.shape[1])
        n_features = int(source.shape[0])
        itemsize = max(1, source.dtype.itemsize)
        staging_cache: dict[int, int] = {}

        def staging(rows: int) -> int:
            width = max(1, min(int(rows), max(1, n_cells)))
            if width not in staging_cache:
                staging_cache[width] = self._source_staging_peak(source, width)
            return staging_cache[width]

        def window_values(rows: int) -> int:
            width = max(0, min(int(rows), n_cells))
            if width == 0:
                return 0
            estimated = (staging(width) + itemsize - 1) // itemsize
            return int(min(width * n_features, max(0, estimated)))

        return staging, window_values

    def _prepare_counts(self, assay: SeuratAssay) -> np.dtype[Any]:
        """Prepare one assay's count source and return its storage dtype.

        Every count is read once, in blocks, and duplicate coordinates of
        sparse blocks are summed, as the write stores them.

        Raises:
            MemoryError: If the preparation, or reading one cell, does not fit
                ``mem_budget``.
            ValueError: If a count is NaN or infinite.
        """
        from ..storage.identity import CountSummary
        from ..storage.partition import affordable_width
        from ..storage.sharding import sparse_producer_peak_bytes

        source = assay.counts
        n_features, n_cells = (int(value) for value in source.shape)
        other_sources = sum(
            max(0, int(item.counts.resident_bytes))
            for item in self._assays
            if item is not assay
        )
        # The count summary is allocated before the write and stays resident
        # through it, so preparation must fit beside it.
        self.reader._prepare_assay(
            assay.name,
            max_bytes=int(self.resources.memoryBytes)
            - other_sources
            - CountSummary.nbytes_for(n_cells, n_features),
        )
        value_range = CountValueRange()
        if n_cells:
            staging, window_values = self._source_estimates(source)
            itemsize = max(1, source.dtype.itemsize)
            available = int(self.resources.memoryBytes) - self._source_resident_bytes()

            def fits(rows: int) -> bool:
                if source.is_sparse:
                    # Summing a block's duplicates costs one unbuffered pull.
                    scan = sparse_producer_peak_bytes(0, window_values(rows), itemsize)
                else:
                    # The integrality check holds a truncated copy and masks.
                    scan = rows * n_features * (itemsize + 2)
                return staging(rows) + scan <= available

            rows = affordable_width(fits, n_cells)
            if rows < 1:
                raise MemoryError(
                    f"Assay {assay.name!r} counts cannot be read one cell at a "
                    "time within mem_budget"
                )
            for start in range(0, n_cells, rows):
                raw = source.read_cells(start, min(start + rows, n_cells))
                value_range.update(
                    canonicalize_sparse(coo_matrix(raw)).data if issparse(raw) else raw
                )
        return count_storage_dtype(source.dtype, value_range)

    def _fit_count_layout(
        self,
        assay: SeuratAssay,
        storage_dtype: Any,
        requested: CountMatrixPolicy | None,
    ) -> CountMatrixPolicy:
        """Return the layout whose counts write and ``countsT`` fit the budget.

        Both writes read source batches of one destination row band, so the
        fit estimates the source once per band, not once per cell.
        """
        from ..storage.identity import CountSummary
        from ..storage.sharding import (
            dense_counts_admission,
            fit_count_layout,
            sparse_counts_admission,
        )
        from .counts_t import counts_t_assays

        source = assay.counts
        n_features, n_cells = (int(value) for value in source.shape)
        resident = CountSummary.nbytes_for(n_cells, n_features) + (
            self._residentSourceBytes
        )
        staging, window_values = self._source_estimates(source)
        return fit_count_layout(
            {assay.name: (n_features, storage_dtype)},
            nCells=n_cells,
            profile=self.profile,
            memoryBytes=self.resources.memoryBytes,
            transposed=counts_t_assays((assay.name,), {assay.name: assay.name}),
            admitCounts=(
                sparse_counts_admission(
                    nRows=n_cells,
                    maxWindowNnz=window_values,
                    sourceDtype=source.dtype,
                    residentBytes=resident,
                    producerStagingBytes=staging,
                )
                if source.is_sparse
                else dense_counts_admission(resident, staging)
            ),
            requested=requested,
        )

    def _write_counts(
        self,
        assay: SeuratAssay,
        requested_rows: int | None,
    ) -> None:
        from ..storage.identity import CountSummary, finalize_counts

        source = assay.counts
        destination = self.counts[assay.name]
        summary = CountSummary(destination)
        n_cells = len(self.reader.cellIds)
        if n_cells == 0:
            finalize_counts(destination, summary=summary)
            return
        # Construction prepared every source.
        self._residentSourceBytes = summary.nbytes + self._source_resident_bytes()
        if source.is_sparse:
            self._write_sparse_counts(
                assay.name,
                source,
                destination,
                requested_rows,
                summary,
            )
        else:
            self._write_dense_counts(
                assay.name,
                source,
                destination,
                requested_rows,
                summary,
            )
        finalize_counts(destination, summary=summary)

    def _write_sparse_counts(
        self,
        assay_name: str,
        source: Any,
        destination: zarr.Array,
        requested_rows: int | None,
        summary: "CountSummary",
    ) -> None:
        from ..storage.sharding import (
            accumulate_sparse_to_shards,
            resolve_sparse_import_batch,
        )

        n_cells = int(source.shape[1])
        staging, window_values = self._source_estimates(source)
        plan = resolve_sparse_import_batch(
            (destination,),
            nRows=n_cells,
            resources=self.resources,
            maxWindowNnz=window_values,
            sourceDtype=source.dtype,
            batchRows=requested_rows,
            residentBytes=self._residentSourceBytes,
            producerStagingBytes=staging,
        )
        self._lastImportPlans[assay_name] = plan
        self._lastImportPlan = plan

        # A source returns one row per requested cell; the shard writer
        # rejects a batch of another width and a stream of another length.
        def batches() -> Iterator[coo_matrix]:
            for start in range(0, n_cells, plan.batchRows):
                stop = min(start + plan.batchRows, n_cells)
                raw = source.read_cells(start, stop)
                yield (
                    raw.tocoo(copy=False)
                    if issparse(raw)
                    else coo_matrix(np.asarray(raw))
                )

        accumulate_sparse_to_shards(
            destination,
            batches(),
            resources=self.resources,
            residentBytes=self._residentSourceBytes,
            producerReserveBytes=plan.producerReserveBytes,
            msg=f"Writing {assay_name} counts",
            io=self.io,
            countSummary=summary,
        )

    def _resolve_dense_batch_rows(
        self,
        source: Any,
        destination: zarr.Array,
        requested_rows: int | None,
    ) -> tuple[int, int]:
        """Return the rows of each source batch and their read staging.

        The layout fit admitted batches of one destination row band, so a
        batch holds one band, or fewer rows on request; the writer admits the
        batch again when it plans the write.
        """
        from ..storage.layout import array_shard_rows

        rows = min(int(source.shape[1]), array_shard_rows(destination))
        if requested_rows is not None:
            rows = min(requested_rows, rows)
        staging, _window_values = self._source_estimates(source)
        return rows, staging(rows)

    def _write_dense_counts(
        self,
        assay_name: str,
        source: Any,
        destination: zarr.Array,
        requested_rows: int | None,
        summary: "CountSummary",
    ) -> None:
        from ..storage.sharding import write_dense_from_row_batches

        n_cells = int(source.shape[1])
        rows, producer_reserve = self._resolve_dense_batch_rows(
            source,
            destination,
            requested_rows,
        )
        self._lastDenseBatchRows[assay_name] = rows

        def batches() -> Iterator[np.ndarray]:
            for start in range(0, n_cells, rows):
                raw = source.read_cells(start, min(start + rows, n_cells))
                # The writer casts to the stored dtype and rejects a count
                # that dtype cannot hold, a batch of another width, and a
                # stream of another length.
                yield raw.toarray() if issparse(raw) else np.asarray(raw)

        write_dense_from_row_batches(
            destination,
            batches(),
            msg=f"Writing {assay_name} counts",
            resources=self.resources,
            residentBytes=self._residentSourceBytes,
            producerReserveBytes=producer_reserve,
            io=self.io,
            countSummary=summary,
        )

    def _write_cell_selection(self) -> ArtifactRef:
        return resolve_import_cell_selection(
            self.root,
            source="seurat",
            inputs={"source_digest": self._sourceDigest},
        )

    @staticmethod
    def _floating_dtype(dtype: Any) -> np.dtype[Any]:
        return floating_payload_dtype(dtype, "Reduction payload")

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
        return fingerprint_row_blocks(
            cls._matrix_blocks(matrix, block_rows, dtype),
            tuple(matrix.shape),
            dtype,
            label="Reduction payload",
        )

    @classmethod
    def _fingerprint_vector(
        cls,
        vector: SeuratNumericVector,
        block_rows: int,
        dtype: np.dtype[Any],
    ) -> str:
        return fingerprint_row_blocks(
            cls._vector_blocks(vector, block_rows, dtype),
            (vector.length,),
            dtype,
            label="Reduction payload",
        )

    @staticmethod
    def _fingerprint_feature_ids(
        values: Sequence[str],
        block_rows: int,
    ) -> str:
        dtype = _bounded_string_dtype(values, block_rows)
        return fingerprint_row_blocks(
            (
                np.asarray(block, dtype=dtype)
                for block in _string_blocks(values, block_rows)
            ),
            (len(values),),
            dtype,
            label="Reduction feature IDs",
        )

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

            if reduction.role == "displayEmbedding":
                ref = write_imported_embedding(
                    self.root,
                    assay=reduction.assayUsed,
                    dimreduc_key=reduction.name,
                    role="umap" if reduction.name.casefold() == "umap" else "tsne",
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
