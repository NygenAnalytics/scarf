from collections.abc import Iterator
from typing import Any

import numpy as np
import pandas as pd

from ..readers import CrReader
from ..storage.count_matrix import CountMatrixPolicy
from ..storage.io_policy import StorageIoPolicy
from ..storage.profiles import (
    StorageProfile,
    ZarrLocation,
    resolve_storage_profile,
)
from ..utils.logging import logger
from ..utils.progress import iter_progress


class CrToZarr:
    """A class for converting data in the Cellranger format to a Zarr
    hierarchy.

    Args:
        cr: A CrReader object, containing the Cellranger data.
        zarr_loc: The file name for the Zarr hierarchy or a store
        dtype: the dtype of the data.
        workspace: Workspace name in the destination store. None uses the
                   legacy layout without a workspace group.
        storage_options: Backend options passed when opening the Zarr store.
        mem_budget: Memory available to the conversion. Accepts bytes, a
                    suffixed size (e.g. '8G'), or a fraction of total system memory (e.g. '0.6').
        nthreads: Worker count for write-time concurrency. When None, auto-detected.
        profile: Zarr encoding profile (``fast_local`` or ``cloud``). When
                 None, chosen from the destination location.
        policy: Count-matrix geometry policy. When None, the default
                unitBytes and chunkBytes plan is used.
        io: Optional explicit read, compute, and write widths. Unset values
            stay under automatic planning.

    Attributes:
        cr: A CrReader object, containing the Cellranger data.
        z: The Zarr hierarchy (array or group).
    """

    def __init__(
        self,
        cr: CrReader,
        zarr_loc: ZarrLocation,
        dtype: str = "uint32",
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
            create_cell_data,
            create_zarr_count_assay,
            validate_assay_name,
        )
        from ..storage.stores import load_zarr

        self.resources = resolve_budget(mem_budget, nthreads)
        self.profile = resolve_storage_profile(zarr_loc, profile)
        self.policy = policy
        self.io = io
        self.cr = cr
        mark_schema_captured = getattr(self.cr, "_mark_schema_captured", None)
        if callable(mark_schema_captured):
            mark_schema_captured()
        self.workspace = workspace
        self.storage_options = storage_options
        assay_names = tuple(dict.fromkeys(self.cr.assayFeats.columns))
        for assay_name in assay_names:
            validate_assay_name(assay_name)
        self.z = load_zarr(zarr_loc=zarr_loc, mode="w", storage_options=storage_options)
        cell_group = create_cell_data(
            root=self.z,
            workspace=self.workspace,
            ids=np.array(self.cr.cell_names()),
            names=np.array(self.cr.cell_names()),
            profile=self.profile,
        )
        for assay_name in assay_names:
            create_zarr_count_assay(
                z=self.z,
                assay_name=assay_name,
                workspace=workspace,
                n_cells=self.cr.nCells,
                feat_ids=self.cr.feature_ids(assay_name),
                feat_names=self.cr.feature_names(assay_name),
                dtype=dtype,
                profile=self.profile,
                policy=policy,
            )
        self._write_reader_metadata(cell_group, assay_names)

    def _write_reader_metadata(
        self,
        cell_group: Any,
        assay_names: tuple[str, ...],
    ) -> None:
        from ..storage.arrays import create_zarr_obj_array
        from ..storage.types import as_zarr_group

        cell_columns = getattr(self.cr, "get_cell_columns", None)
        if callable(cell_columns):
            for name, raw_values in cell_columns():
                if name in {"I", "ids", "names"} or name in cell_group:
                    continue
                values = np.asarray(raw_values)
                if values.ndim != 1 or values.size != self.cr.nCells:
                    raise ValueError(
                        f"Cell metadata column {name!r} has shape {values.shape}; "
                        f"expected ({self.cr.nCells},)"
                    )
                create_zarr_obj_array(
                    cell_group,
                    name,
                    values,
                    values.dtype,
                    profile=self.profile,
                )

        feature_columns = getattr(self.cr, "get_feature_columns", None)
        if not callable(feature_columns):
            return
        ranges = self._prep_assay_input_ranges(self.cr.assayFeats)
        targets: dict[str, tuple[Any, np.ndarray]] = {}
        for assay_name in assay_names:
            indexes = np.concatenate(
                [
                    np.arange(start, end, dtype=np.int64)
                    for start, end in ranges[assay_name]
                ]
            )
            group_path = (
                f"{assay_name}/featureData"
                if self.workspace is None
                else f"{self.workspace}/{assay_name}/featureData"
            )
            targets[assay_name] = (
                as_zarr_group(self.z[group_path], name=group_path),
                indexes,
            )
        for name, raw_values in feature_columns():
            values = np.asarray(raw_values)
            if values.ndim != 1 or values.size != self.cr.nFeatures:
                raise ValueError(
                    f"Feature metadata column {name!r} has shape {values.shape}; "
                    f"expected ({self.cr.nFeatures},)"
                )
            for group, indexes in targets.values():
                if name in {"I", "ids", "names"} or name in group:
                    continue
                selected = values[indexes]
                create_zarr_obj_array(
                    group,
                    name,
                    selected,
                    selected.dtype,
                    profile=self.profile,
                )

    @staticmethod
    def _prep_assay_input_ranges(af: pd.DataFrame) -> dict[str, list[list[int]]]:
        assay_order = (
            af.T.nFeatures.groupby(af.columns).sum().sort_values(ascending=False).index
        )
        ranges = {}
        for assay in assay_order:
            temp = []
            if len(af[assay].shape) == 2:
                for i in af[assay].values[1:3].T:
                    temp.append([i[0], i[1]])
            else:
                idx = af[assay]
                temp = [[idx.start, idx.end]]
            ranges[assay] = temp
        return ranges

    @staticmethod
    def _prep_feat_index_offset(
        ranges: dict[str, list[list[int]]],
    ) -> dict[str, list[int]]:
        feat_offset: dict[str, list[int]] = {}
        for i in ranges:
            feat_offset[i] = []
            lv = 0
            for j in ranges[i]:
                feat_offset[i].append(-j[0] + lv)
                lv += j[1] - j[0]
        return feat_offset

    def dump(
        self,
        batch_size: int | None = None,
        lines_in_mem: int = 100000,
    ) -> None:
        """Writes the count values into the Zarr matrix.

        Args:
            batch_size: Number of source cells per batch. By default, a
                        destination-aligned value is selected within the memory budget.
            lines_in_mem: Number of lines to read at a time from MTX file (only used for CrDirReader)
                          (Default value: 100000)

        Raises:
            AssertionError: Catches eventual bugs in the class, if number of cells does not match after transformation.

        Returns:
            None
        """
        from scipy.sparse import coo_matrix

        from ..storage.schema import load_count_array
        from ..storage.sharding import (
            SparseShardBuffer,
            SparseWriteBand,
            resolve_sparse_import_batch,
            sparse_matrix_bytes,
            write_sparse_bands,
        )

        if batch_size is not None and batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if lines_in_mem <= 0:
            raise ValueError("lines_in_mem must be positive")
        input_ranges = self._prep_assay_input_ranges(self.cr.assayFeats)
        stores = {
            assay: load_count_array(self.z, assay, self.workspace)
            for assay in input_ranges
        }
        buffers = {assay: SparseShardBuffer(store) for assay, store in stores.items()}
        feat_offset = self._prep_feat_index_offset(input_ranges)
        configure_lines = getattr(
            self.cr,
            "_set_sparse_import_lines_in_mem",
            None,
        )
        prepare = getattr(self.cr, "_prepare_sparse_import", None)
        release = getattr(self.cr, "_release_sparse_import", None)

        try:
            if callable(configure_lines):
                configure_lines(lines_in_mem)
            if callable(prepare):
                prepare()
            resident_reader_bytes = 0
            reader_resident = getattr(
                self.cr,
                "_sparse_import_resident_bytes",
                None,
            )
            if callable(reader_resident):
                resident_reader_bytes = max(0, int(reader_resident()))
            projection_value_bytes = max(
                self.cr.matrix_dtype.itemsize,
                *(store.dtype.itemsize for store in stores.values()),
            )

            def projection_staging_bytes(rows: int) -> int:
                source_values = max(0, int(self.cr.max_window_nnz(rows)))
                return source_values * (
                    projection_value_bytes
                    + 3 * np.dtype(np.int64).itemsize
                    + 2 * np.dtype(np.bool_).itemsize
                )

            plan = resolve_sparse_import_batch(
                tuple(stores.values()),
                nRows=self.cr.nCells,
                resources=self.resources,
                maxWindowNnz=self.cr.max_window_nnz,
                sourceDtype=self.cr.matrix_dtype,
                batchRows=batch_size,
                residentBytes=resident_reader_bytes,
                producerStagingBytes=lambda rows: self.cr.producer_staging_bytes(
                    rows,
                    lines_in_mem,
                ),
                extraProducerBytes=projection_staging_bytes,
            )
            self._lastImportPlan = plan
            resolved_batch_rows = plan.batchRows
            logger.debug(
                f"Resolved Cell Ranger source batch rows={resolved_batch_rows} "
                f"write_tasks={plan.writeTasks}"
            )

            def writes() -> Iterator[SparseWriteBand]:
                n_chunks = (
                    self.cr.nCells + resolved_batch_rows - 1
                ) // resolved_batch_rows
                source = iter_progress(
                    self.cr.consume(resolved_batch_rows, lines_in_mem),
                    total=n_chunks,
                    desc="Writing counts",
                )
                for matrix in source:
                    chunk = matrix.tocoo(copy=False)
                    source_bytes = sparse_matrix_bytes(matrix, chunk)
                    for assay in input_ranges:
                        selected = np.zeros(chunk.col.shape[0], dtype=bool)
                        columns = chunk.col.copy()
                        for bounds, offset in zip(
                            input_ranges[assay],
                            feat_offset[assay],
                            strict=True,
                        ):
                            inside = (chunk.col >= bounds[0]) & (chunk.col < bounds[1])
                            columns[inside] += offset
                            selected |= inside
                        projected = coo_matrix(
                            (
                                chunk.data[selected],
                                (chunk.row[selected], columns[selected]),
                            ),
                            shape=(chunk.shape[0], buffers[assay].nColumns),
                        )
                        for band in buffers[assay].add(projected):
                            producer_bytes = (
                                source_bytes
                                + columns.nbytes
                                + 2 * selected.nbytes
                                + sparse_matrix_bytes(projected)
                                + sum(item.residentBytes for item in buffers.values())
                            )
                            yield SparseWriteBand(
                                stores[assay],
                                band,
                                producer_bytes,
                            )
                if self.cr.nCells:
                    del matrix, chunk, selected, columns, inside, projected
                for assay, buffer in buffers.items():
                    for band in buffer.finish():
                        producer_bytes = sum(
                            item.residentBytes for item in buffers.values()
                        )
                        yield SparseWriteBand(
                            stores[assay],
                            band,
                            producer_bytes,
                        )

            write_sparse_bands(
                writes(),
                resources=self.resources,
                residentBytes=resident_reader_bytes,
                producerReserveBytes=plan.producerReserveBytes,
                total=plan.writeTasks,
                io=self.io,
            )
            if any(buffer.rows != self.cr.nCells for buffer in buffers.values()):
                raise AssertionError(
                    "Cell Ranger conversion did not write every source row"
                )
            logger.info(
                f"Wrote {self.cr.nCells} cells and "
                f"{sum(buffer.nColumns for buffer in buffers.values())} features "
                f"from Cell Ranger to {len(stores)} assay(s)"
            )
            from .counts_t import finalize_writer_counts_t_many

            finalize_writer_counts_t_many(
                self.z,
                tuple(stores),
                self.workspace,
                resources=self.resources,
                profile=self.profile,
                policy=self.policy,
                io=self.io,
            )
        finally:
            if callable(release):
                release()


MtxToZarr = CrToZarr
