from collections.abc import Iterator
from typing import Any

import numpy as np
import pandas as pd

from ..storage.types import as_zarr_group
from ..readers import CSVReader
from ..storage.count_dtype import count_storage_dtype
from ..storage.count_matrix import DEFAULT_COUNT_MATRIX_POLICY, CountMatrixPolicy
from ..storage.io_policy import StorageIoPolicy
from ..storage.profiles import (
    StorageProfile,
    ZarrLocation,
    resolve_storage_profile,
)
from ..storage.sharding import write_dense_from_row_batches
from ..utils.logging import logger


class CSVtoZarr:
    """A class for converting data from CSV format to a Zarr hierarchy.

    Args:
        cr: A CSVReader object
        zarr_loc: The file name for the Zarr hierarchy.
        assay_name: A label for the assay. Ex. "RNA" or "ATAC"
        workspace: Workspace name in the destination store. None uses the
                   legacy layout without a workspace group.
        storage_options: Backend options passed when opening the Zarr store.
        mem_budget: Memory available to the conversion. Accepts bytes, a
                    suffixed size (e.g. '8G'), or a fraction of total system memory (e.g. '0.6').
        nthreads: Worker count for write-time concurrency. When None, auto-detected.
        profile: Zarr encoding profile (``fast_local`` or ``cloud``). When
                 None, chosen from the destination location.
        policy: Count-matrix geometry policy, used exactly. When None, the
                default policy is used. An import that does not fit
                ``mem_budget`` raises MemoryError before the destination is
                created, naming the largest smaller policy that fits.
        io: Optional explicit read, compute, and write widths. Unset values
            stay under automatic planning.
        assay_type: Preset assay type, such as ``RNA``, for an assay whose name
                    is not a preset. When None, the assay name decides the type.

    The counts are stored in the dtype that
    :func:`~scarf.storage.count_dtype.count_storage_dtype` resolves from the
    range the reader found in its first pass over every row. A count that
    the stored dtype cannot hold raises instead of wrapping.

    Attributes:
        csvr: A CSVReader object
        fn: The file name for the Zarr hierarchy.
        z: The Zarr hierarchy (array or group).
    """

    def __init__(
        self,
        cr: CSVReader,
        zarr_loc: ZarrLocation,
        assay_name: str,
        workspace: str | None = None,
        storage_options: dict[str, Any] | None = None,
        mem_budget: int | str | None = None,
        nthreads: int | None = None,
        profile: StorageProfile | None = None,
        policy: CountMatrixPolicy | None = None,
        io: StorageIoPolicy | None = None,
        assay_type: str | None = None,
    ) -> None:
        from ..storage.budget import resolve_budget
        from ..storage.schema import (
            create_cell_data,
            create_zarr_count_assay,
            validate_assay_name,
        )
        from ..storage.stores import load_zarr
        from .counts_t import validate_assay_type

        self.csvr = cr
        self.assayName = assay_name
        validate_assay_name(self.assayName)
        validate_assay_type(assay_type)
        self.assayType = assay_type
        self.resources = resolve_budget(mem_budget, nthreads)
        self.profile = resolve_storage_profile(zarr_loc, profile)
        self.io = io
        self.workspace = workspace
        self.storage_options = storage_options
        cell_ids = self.csvr.cell_ids()
        storage_dtype = count_storage_dtype(self.csvr.countDtype, self.csvr.countRange)
        # A layout that does not fit fails here, before the destination exists.
        layout = self._fit_count_layout(storage_dtype, policy)
        self.z = load_zarr(zarr_loc, mode="w", storage_options=storage_options)
        _ = create_cell_data(
            root=self.z,
            workspace=workspace,
            ids=cell_ids,
            names=cell_ids,
            profile=self.profile,
        )
        create_zarr_count_assay(
            z=self.z,
            assay_name=self.assayName,
            workspace=workspace,
            n_cells=self.csvr.nCells,
            feat_ids=self.csvr.feature_ids(),
            feat_names=self.csvr.feature_ids(),
            dtype=storage_dtype,
            profile=self.profile,
            policy=layout,
        )

    def _count_import_requirements(self) -> tuple[int, int]:
        """Return the resident and source batch bytes of the counts write.

        A batch is one reader chunk of every count column, held with its
        DataFrame. The layout fit and the write plan the same import from
        these.
        """
        from ..storage.identity import CountSummary

        rows = min(int(self.csvr.pandas_kwargs["chunksize"]), self.csvr.nCells)
        batch_bytes = rows * self.csvr.nFeatures * self.csvr.countDtype.itemsize
        resident = CountSummary.nbytes_for(self.csvr.nCells, self.csvr.nFeatures)
        return resident, 2 * batch_bytes

    def _fit_count_layout(
        self, storage_dtype: Any, requested: CountMatrixPolicy | None
    ) -> CountMatrixPolicy:
        """Return the requested or default count layout if its writes fit the budget."""
        from ..storage.sharding import dense_counts_admission, fit_count_layout
        from .counts_t import counts_t_assays

        resident, producer_reserve = self._count_import_requirements()
        assay_types = {} if self.assayType is None else {self.assayName: self.assayType}
        return fit_count_layout(
            {self.assayName: (self.csvr.nFeatures, storage_dtype)},
            nCells=self.csvr.nCells,
            profile=self.profile,
            memoryBytes=self.resources.memoryBytes,
            transposed=counts_t_assays((self.assayName,), assay_types),
            admitCounts=dense_counts_admission(resident + producer_reserve),
            requested=DEFAULT_COUNT_MATRIX_POLICY if requested is None else requested,
        )

    def dump(self) -> None:
        """Writes the count values into the Zarr matrix.

        Raises:
            OverflowError: If a count no longer fits the stored dtype because
                the file changed after the reader's pass.
            ValueError: If the file no longer holds the rows that the
                reader's pass counted.

        Returns:
            None
        """
        from ..storage.identity import CountSummary, finalize_counts
        from ..storage.schema import load_count_array
        from ..storage.metadata_keys import metadata_column_keys
        from ._store import keyed_metadata_columns, write_metadata_column

        store = load_count_array(self.z, self.assayName, self.workspace)
        summary = CountSummary(store)
        resident, producer_reserve = self._count_import_requirements()
        cell_data_path = (
            "cellData" if self.workspace is None else f"{self.workspace}/cellData"
        )
        cell_data_grp = as_zarr_group(
            self.z[cell_data_path],
            name=cell_data_path,
        )
        # Each entry pairs a column's position in the reader payload with its
        # dtype across every row, so skipped columns keep the mapping.
        metadata = list(
            keyed_metadata_columns(
                zip(
                    self.csvr.cellDataCols,
                    enumerate(self.csvr.cellDataDtypes or []),
                ),
                metadata_column_keys(
                    self.csvr.cellDataCols,
                    taken=cell_data_grp.keys(),
                ),
                "cell",
            )
        )
        parts: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {
            position: [] for _name, (position, _dtype) in metadata
        }

        def count_batches() -> Iterator[np.ndarray]:
            for counts, cell_values in self.csvr.consume():
                if cell_values is not None:
                    for _name, (position, dtype) in metadata:
                        parts[position].append(
                            _metadata_part(cell_values[:, position], dtype)
                        )
                # The writer casts to the stored dtype and rejects a count
                # that dtype cannot hold.
                yield counts

        write_dense_from_row_batches(
            store,
            count_batches(),
            msg="Writing CSV counts",
            resources=self.resources,
            io=self.io,
            producerReserveBytes=producer_reserve,
            residentBytes=resident,
            countSummary=summary,
        )
        for name, (position, _dtype) in metadata:
            values = np.concatenate([values for values, _missing in parts[position]])
            missing = np.concatenate([missing for _values, missing in parts[position]])
            write_metadata_column(
                cell_data_grp,
                name,
                values,
                missing,
                profile=self.profile,
            )
        logger.info(
            f"Wrote {self.csvr.nCells} cells and {self.csvr.nFeatures} features "
            f"from CSV to assay {self.assayName}"
        )
        finalize_counts(store, summary=summary)
        from .counts_t import finalize_writer_counts_t

        finalize_writer_counts_t(
            self.z,
            self.assayName,
            self.workspace,
            assay_type=self.assayType,
            resources=self.resources,
            profile=self.profile,
            io=self.io,
        )


def _metadata_part(
    values: np.ndarray,
    dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert one chunk of a CSV metadata column to its dtype for every row.

    Blank text cells are missing; blank numeric cells stay NaN.
    """
    if dtype.kind != "O":
        converted = np.asarray(values, dtype=dtype)
        return converted, np.zeros(converted.shape, dtype=bool)
    missing = np.asarray(pd.isna(values), dtype=bool)
    text = np.asarray(
        ["" if absent else str(value) for value, absent in zip(values, missing)],
        dtype=str,
    )
    return text, missing
