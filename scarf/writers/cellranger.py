from collections.abc import Iterator
from typing import Any

import numpy as np
import pandas as pd

from ..readers import CrReader
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
        mem_budget: Memory available to the conversion. Accepts bytes, a
                    suffixed size (e.g. '8G'), or a fraction of total system memory (e.g. '0.6').
        nthreads: Worker count for write-time concurrency. When None, auto-detected.

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
        targetChunkBytes: int | None = None,
        targetShardBytes: int | None = None,
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
        create_cell_data(
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
                targetChunkBytes=targetChunkBytes,
                targetShardBytes=targetShardBytes,
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

    def dump(self, batch_size: int = 1000, lines_in_mem: int = 100000) -> None:
        """Writes the count values into the Zarr matrix.

        Args:
            batch_size: Number of cells to save at a time. (Default value: 1000)
            lines_in_mem: Number of lines to read at a time from MTX file (only used for CrDirReader)
                          (Default value: 100000)

        Raises:
            AssertionError: Catches eventual bugs in the class, if number of cells does not match after transformation.

        Returns:
            None
        """
        from scipy.sparse import coo_matrix

        from ..storage.budget import admitted_worker_count
        from ..storage.schema import load_count_array
        from ..storage.sharding import (
            SparseShardBuffer,
            SparseWriteBand,
            sparse_matrix_bytes,
            sparse_producer_peak_bytes,
            write_sparse_bands,
        )

        input_ranges = self._prep_assay_input_ranges(self.cr.assayFeats)
        stores = {
            assay: load_count_array(self.z, assay, self.workspace)
            for assay in input_ranges
        }
        buffers = {assay: SparseShardBuffer(store) for assay, store in stores.items()}
        feat_offset = self._prep_feat_index_offset(input_ranges)

        def writes() -> Iterator[SparseWriteBand]:
            n_chunks = (self.cr.nCells + batch_size - 1) // batch_size
            source = iter_progress(
                self.cr.consume(batch_size, lines_in_mem),
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
                            + selected.nbytes
                            + columns.nbytes
                            + inside.nbytes
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

        staging_bytes = self.cr.producer_staging_bytes(
            batch_size,
            lines_in_mem,
        )
        admitted_worker_count(
            self.resources,
            taskBytes=1,
            residentBytes=staging_bytes,
            requested=1,
        )
        source_nnz = self.cr.max_window_nnz(batch_size)
        producer_rows = batch_size + max(
            buffer.shardRows for buffer in buffers.values()
        )
        buffered_nnz = self.cr.max_window_nnz(producer_rows)
        value_bytes = max(
            np.dtype(self.cr.matrix_dtype).itemsize,
            *(np.dtype(store.dtype).itemsize for store in stores.values()),
        )
        producer_reserve_bytes = (
            sparse_producer_peak_bytes(
                buffered_nnz,
                source_nnz,
                value_bytes,
            )
            + staging_bytes
        )
        write_sparse_bands(
            writes(),
            resources=self.resources,
            producerReserveBytes=producer_reserve_bytes,
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
