import time
from collections.abc import Iterator
from typing import Any

import numpy as np
from scipy.sparse import coo_matrix

from ..storage.types import as_zarr_group
from ..readers import H5adReader
from ..readers.h5ad import _H5adAssayFeatures
from ..storage.profiles import (
    StorageProfile,
    ZarrLocation,
    resolve_storage_profile,
)
from ..utils.logging import logger
from ..utils.progress import tqdmbar

# Root group names the datastore layout reserves for cell metadata, matrices,
# and plots. Assays must not collide with these or the store is corrupted.
_RESERVED_ASSAY_NAMES = frozenset({"cellData", "matrices", "plots"})


def _validate_assay_names(names: tuple[str, ...]) -> None:
    for name in names:
        if not name or not name.strip():
            raise ValueError("Assay names must be non-empty")
        if "/" in name or "\\" in name:
            raise ValueError(f"Assay name {name!r} must not contain path separators")
        if name in _RESERVED_ASSAY_NAMES:
            raise ValueError(
                f"Assay name {name!r} is reserved by the datastore layout. "
                "Choose another name or provide a different assay_name_map."
            )


class H5adToZarr:
    """A class for converting data in anndata's H5ad format to Zarr hierarchy.

    Args:
        h5ad: Reader for the source H5AD file.
        zarr_loc: The file name for the Zarr hierarchy or a store
        assay_name: the name of the assay (e. g. 'RNA')
        assay_split_key: A var column used to split features into assays.
        assay_name_map: Feature type to assay name overrides.
        workspace: An optional workspace id.
        mem_budget: Memory available to the conversion. Accepts bytes, a
                    suffixed size (e.g. '8G'), or a fraction of total system memory (e.g. '0.6').
        nthreads: Worker count for write-time concurrency. When None, auto-detected.

    Attributes:
        h5ad: A h5ad object (h5 file with added AnnData structure).
        assayName: The Zarr hierarchy (array or group).
        z: The Zarr hierarchy (array or group).
    """

    def __init__(
        self,
        h5ad: H5adReader,
        zarr_loc: ZarrLocation,
        assay_name: str | None = None,
        workspace: str | None = None,
        storage_options: dict[str, Any] | None = None,
        mem_budget: int | str | None = None,
        nthreads: int | None = None,
        profile: StorageProfile | None = None,
        targetChunkBytes: int | None = None,
        targetShardBytes: int | None = None,
        assay_split_key: str | None = None,
        assay_name_map: dict[str, str] | None = None,
    ) -> None:
        from ..storage.budget import resolve_budget
        from ..storage.schema import create_zarr_count_assay
        from ..storage.stores import load_zarr

        self.resources = resolve_budget(mem_budget, nthreads)
        self.profile = resolve_storage_profile(zarr_loc, profile)
        self.h5ad = h5ad
        self.h5ad.infer_storage_dtype(self.resources.memoryBytes)
        csc_peak = self.h5ad.csc_conversion_peak_bytes()
        if csc_peak > self.resources.memoryBytes:
            raise MemoryError(
                f"CSC to CSR conversion needs about {csc_peak} bytes, but the "
                f"conversion memory limit is {self.resources.memoryBytes} bytes"
            )
        if csc_peak:
            self.h5ad.materialize_csc()
        self.storageDtype = getattr(
            self.h5ad,
            "storageDtype",
            self.h5ad.matrixDtype,
        )
        self.workspace = workspace
        self.storage_options = storage_options
        self.assaySplitKey = assay_split_key
        self.assayNameMap = assay_name_map
        self.assayFeatures: dict[str, _H5adAssayFeatures] | None
        if assay_split_key is not None:
            if assay_name is not None:
                logger.warning(
                    "`assay_name` is ignored when `assay_split_key` is provided"
                )
            self.assayName = None
            self.assayFeatures = self.h5ad.assay_feature_slices(
                assay_split_key,
                assay_name_map,
            )
            self.assayNames = tuple(self.assayFeatures)
        elif assay_name is None:
            logger.info(
                "No value provided for assay names. Will use default value: 'RNA'"
            )
            self.assayName = "RNA"
            self.assayFeatures = None
            self.assayNames = (self.assayName,)
        else:
            self.assayName = assay_name
            self.assayFeatures = None
            self.assayNames = (self.assayName,)
        _validate_assay_names(self.assayNames)
        self.z = load_zarr(zarr_loc=zarr_loc, mode="w", storage_options=storage_options)
        self._ini_cell_data()
        for resolved_assay_name in self.assayNames:
            if self.assayFeatures is None:
                feature_ids = self.h5ad.feat_ids()
                feature_names = self.h5ad.feat_names()
            else:
                features = self.assayFeatures[resolved_assay_name]
                feature_ids = features.featureIds
                feature_names = features.featureNames
            create_zarr_count_assay(
                z=self.z,
                assay_name=resolved_assay_name,
                workspace=workspace,
                n_cells=self.h5ad.nCells,
                feat_ids=feature_ids,
                feat_names=feature_names,
                dtype=self.storageDtype,
                profile=self.profile,
                targetChunkBytes=targetChunkBytes,
                targetShardBytes=targetShardBytes,
            )
        self._ini_feature_data()

    def _ini_cell_data(self) -> None:
        from ..storage.arrays import create_zarr_obj_array
        from ..storage.schema import create_cell_data

        ids = self.h5ad.cell_ids()
        g = create_cell_data(
            root=self.z,
            workspace=self.workspace,
            ids=ids,
            names=ids,
            profile=self.profile,
        )
        for i, j in self.h5ad.get_cell_columns():
            create_zarr_obj_array(g, i, j, j.dtype, profile=self.profile)

    def _ini_feature_data(self) -> None:
        from ..storage.arrays import create_zarr_obj_array

        targets: list[tuple[Any, np.ndarray | None]] = []
        for assay_name in self.assayNames:
            if self.workspace is None:
                group_path = f"{assay_name}/featureData"
            else:
                group_path = f"{self.workspace}/{assay_name}/featureData"
            feat_group = as_zarr_group(self.z[group_path], name=group_path)
            feature_indexes = (
                None
                if self.assayFeatures is None
                else self.assayFeatures[assay_name].featureIndexes
            )
            targets.append((feat_group, feature_indexes))

        # Stream one column at a time so a single decoded var column is held in
        # memory rather than every column for the full feature axis at once.
        for column_name, values in self.h5ad.get_feat_columns():
            for feat_group, feature_indexes in targets:
                if column_name in feat_group:
                    continue
                selected = (
                    values if feature_indexes is None else values[feature_indexes]
                )
                create_zarr_obj_array(
                    feat_group,
                    column_name,
                    selected,
                    selected.dtype,
                    profile=self.profile,
                )

    def dump(self, batch_size: int = 1000) -> None:
        """Write h5ad matrix data into the Zarr counts array.

        Args:
            batch_size: Number of cells read from the source per batch.

        Raises:
            AssertionError: If written cell count does not match expected nCells.

        Returns:
            None
        """
        from ..storage.budget import admitted_worker_count
        from ..storage.sharding import (
            SparseShardBuffer,
            sparse_producer_peak_bytes,
            write_sparse_bands,
        )
        from ..storage.schema import load_count_array

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        destinations = {
            assay_name: load_count_array(self.z, assay_name, self.workspace)
            for assay_name in self.assayNames
        }
        buffers = {
            assay_name: SparseShardBuffer(destination)
            for assay_name, destination in destinations.items()
        }
        logger.info(
            f"Writing counts with up to {self.resources.workers} row-band writer(s)"
        )
        started = time.perf_counter()
        batch_rows = min(batch_size, self.h5ad.nCells)
        resident_source_bytes = self.h5ad.materialized_csr_bytes()
        if self.assayFeatures is None:
            feature_index_bytes = 0
            projection_bytes = 0
            projection_scratch_bytes = 0
        else:
            feature_index_bytes = sum(
                assay.featureIndexes.nbytes for assay in self.assayFeatures.values()
            )
            projection_bytes = 2 * self.h5ad.nFeatures * np.dtype(np.int64).itemsize
            projection_scratch_bytes = (
                max(assay.featureIndexes.size for assay in self.assayFeatures.values())
                * np.dtype(np.int64).itemsize
            )
        admitted_worker_count(
            self.resources,
            taskBytes=max(
                1,
                self.h5ad.max_batch_nnz_peak_bytes(),
                projection_scratch_bytes,
            ),
            residentBytes=(
                resident_source_bytes + feature_index_bytes + projection_bytes
            ),
            requested=1,
        )
        source_nnz = self.h5ad.max_batch_nnz(batch_size)
        producer_rows = batch_size + max(
            buffer.shardRows for buffer in buffers.values()
        )
        buffered_nnz = self.h5ad.max_batch_nnz(producer_rows)
        value_bytes = max(
            np.dtype(self.h5ad.sourceMatrixDtype).itemsize,
            np.dtype(self.h5ad.matrixDtype).itemsize,
            np.dtype(self.storageDtype).itemsize,
        )
        producer_reserve_bytes = sparse_producer_peak_bytes(
            buffered_nnz,
            source_nnz,
            value_bytes,
        ) + self.h5ad.producer_batch_staging_bytes(batch_size)
        if self.h5ad.matrixOrientation == "dense":
            producer_reserve_bytes += (
                batch_rows
                * self.h5ad.nFeatures
                * np.dtype(self.h5ad.sourceMatrixDtype).itemsize
            )
        projection = (
            None if self.assayFeatures is None else self._assay_feature_projection()
        )
        if projection is not None:
            resident_source_bytes += sum(array.nbytes for array in projection)
        resident_source_bytes += feature_index_bytes
        write_sparse_bands(
            self._count_shard_tasks(
                batch_size,
                buffers,
                destinations,
                projection,
            ),
            resources=self.resources,
            residentBytes=resident_source_bytes,
            producerReserveBytes=producer_reserve_bytes,
        )
        counts_seconds = time.perf_counter() - started

        for assay_name, buffer in buffers.items():
            if buffer.rows != self.h5ad.nCells:
                raise AssertionError(
                    "ERROR: This is a bug in H5adToZarr. All cells might not have been "
                    f"successfully written into the {assay_name} counts array. "
                    "Please report this issue"
                )
        # counts is the durable physical orientation for H5AD imports. Assay
        # readers use it directly when the optional derived countsT is absent.
        logger.info(f"Counts written in {counts_seconds:.1f}s")

    def _count_shard_tasks(
        self,
        batch_size: int,
        buffers: dict[str, Any],
        destinations: dict[str, Any],
        projection: tuple[np.ndarray, np.ndarray] | None,
    ) -> Iterator[Any]:
        """Yield complete row-band writes from one serial pass over the source."""
        from ..storage.sharding import SparseWriteBand, sparse_matrix_bytes

        n_batches = (self.h5ad.nCells + batch_size - 1) // batch_size
        stream = tqdmbar(
            self.h5ad.consume(batch_size),
            total=n_batches,
            desc="Writing counts",
        )
        if self.assayFeatures is None:
            buffer = buffers[self.assayNames[0]]
            destination = destinations[self.assayNames[0]]
            for matrix in stream:
                chunk = matrix.tocoo(copy=False)
                source_bytes = sparse_matrix_bytes(matrix, chunk)
                for band in buffer.add(chunk):
                    producer_bytes = source_bytes + sum(
                        item.residentBytes for item in buffers.values()
                    )
                    yield SparseWriteBand(destination, band, producer_bytes)
            if self.h5ad.nCells:
                del matrix, chunk
        else:
            if projection is None:
                raise RuntimeError("Multi-assay projection was not initialized")
            codes, columns = projection
            for matrix in stream:
                chunk = matrix.tocoo(copy=False)
                source_bytes = sparse_matrix_bytes(matrix, chunk)
                batch_codes = codes[chunk.col]
                batch_columns = columns[chunk.col]
                source_bytes += batch_codes.nbytes + batch_columns.nbytes
                for code, assay_name in enumerate(self.assayNames):
                    buffer = buffers[assay_name]
                    selected = batch_codes == code
                    # Every assay sees every batch, including one with no
                    # values, so all buffers share the same row offsets.
                    projected = coo_matrix(
                        (
                            chunk.data[selected],
                            (chunk.row[selected], batch_columns[selected]),
                        ),
                        shape=(chunk.shape[0], buffer.nColumns),
                    )
                    for band in buffer.add(projected):
                        producer_bytes = (
                            source_bytes
                            + selected.nbytes
                            + sparse_matrix_bytes(projected)
                            + sum(item.residentBytes for item in buffers.values())
                        )
                        yield SparseWriteBand(
                            destinations[assay_name],
                            band,
                            producer_bytes,
                        )
            if self.h5ad.nCells:
                del matrix, chunk, batch_codes, batch_columns, selected, projected
        for assay_name, buffer in buffers.items():
            for band in buffer.finish():
                producer_bytes = sum(item.residentBytes for item in buffers.values())
                yield SparseWriteBand(
                    destinations[assay_name],
                    band,
                    producer_bytes,
                )

    def _assay_feature_projection(self) -> tuple[np.ndarray, np.ndarray]:
        """Map each source feature to its assay code and assay-local column."""
        if self.assayFeatures is None:
            raise RuntimeError("Multi-assay features have not been initialized")
        codes = np.full(int(self.h5ad.nFeatures), -1, dtype=np.int64)
        columns = np.zeros(int(self.h5ad.nFeatures), dtype=np.int64)
        for code, assay_name in enumerate(self.assayNames):
            indexes = self.assayFeatures[assay_name].featureIndexes
            codes[indexes] = code
            columns[indexes] = np.arange(indexes.size, dtype=np.int64)
        return codes, columns
