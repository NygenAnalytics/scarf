from typing import Any

import numpy as np
import pandas as pd

from ..readers import CrReader
from ..storage.stores import ZARRLOC
from ..utils.logging import logger
from ..utils.progress import tqdmbar


class CrToZarr:
    """A class for converting data in the Cellranger format to a Zarr
    hierarchy.

    Args:
        cr: A CrReader object, containing the Cellranger data.
        zarr_loc: The file name for the Zarr hierarchy or a store
        chunk_size: The requested size of chunks to load into memory and process.
        dtype: the dtype of the data.
        mem_budget: Memory budget driving write-time chunk and shard geometry. Accepts bytes, a
                    suffixed size (e.g. '8G'), or a fraction of total system memory (e.g. '0.6').
                    Set it to simulate writing on a machine with a different memory size.
        nthreads: Worker count for write-time concurrency. When None, auto-detected.
        working_copies: Number of concurrent in-memory working copies the memory budget is divided
                        across. When None, uses SCARF_WORKING_COPIES env var or the default.

    Attributes:
        cr: A CrReader object, containing the Cellranger data.
        chunkSizes: The requested size of chunks to load into memory and process.
        z: The Zarr hierarchy (array or group).
    """

    def __init__(
        self,
        cr: CrReader,
        zarr_loc: ZARRLOC,
        chunk_size: tuple[int, int] = (1000, 1000),
        dtype: str = "uint32",
        workspace: str | None = None,
        storage_options: dict[str, Any] | None = None,
        mem_budget: int | str | None = None,
        nthreads: int | None = None,
        working_copies: int | None = None,
    ) -> None:
        from . import (
            _apply_budget_override,
            create_cell_data,
            create_zarr_count_assay,
            load_zarr,
        )

        _apply_budget_override(mem_budget, nthreads, working_copies)
        self.cr = cr
        self.chunkSizes = chunk_size
        self.workspace = workspace
        self.storage_options = storage_options
        self.z = load_zarr(zarr_loc=zarr_loc, mode="w", storage_options=storage_options)
        create_cell_data(
            z=self.z,
            workspace=self.workspace,
            ids=np.array(self.cr.cell_names()),
            names=np.array(self.cr.cell_names()),
        )
        for assay_name in self.cr.assayFeats.columns:
            create_zarr_count_assay(
                z=self.z,
                assay_name=assay_name,
                workspace=workspace,
                chunk_size=chunk_size,
                n_cells=self.cr.nCells,
                feat_ids=self.cr.feature_ids(assay_name),
                feat_names=self.cr.feature_names(assay_name),
                dtype=dtype,
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
        from . import finalize_writer_counts, load_count_store

        input_ranges = self._prep_assay_input_ranges(self.cr.assayFeats)
        stores = {x: load_count_store(self.z, x, self.workspace) for x in input_ranges}
        feat_offset = self._prep_feat_index_offset(input_ranges)
        s = 0
        n_chunks = self.cr.nCells // batch_size + 1
        for a in tqdmbar(self.cr.consume(batch_size, lines_in_mem), total=n_chunks):
            for assay in input_ranges:
                idx = np.zeros(a.col.shape[0]).astype(bool)
                feat_coords = a.col.copy()
                for r, of in zip(input_ranges[assay], feat_offset[assay]):
                    temp = (a.col >= r[0]) & (a.col < r[1])
                    if of != 0:
                        feat_coords[temp] = (
                            feat_coords[temp] + of
                        )  # of is already a negative value
                    idx = idx | temp
                if idx.sum() > 0:
                    stores[assay].set_coordinate_selection(
                        (s + a.row[idx], feat_coords[idx]), a.data[idx]
                    )
                else:
                    logger.warning(
                        f"No feature captured from chunk {s} to {s + a.shape[0]} for assay: {assay}"
                    )
            s += a.shape[0]
        if s != self.cr.nCells:
            raise AssertionError(
                "ERROR: This is a bug in CrToZarr. All cells might not have been successfully "
                "written into the zarr file. Please report this issue"
            )
        for assay in input_ranges:
            finalize_writer_counts(self.z, assay, self.workspace)
