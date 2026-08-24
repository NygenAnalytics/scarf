import math
import time
from collections.abc import Iterator
from typing import Any, cast

import numpy as np
from scipy.sparse import coo_matrix

from ..storage.types import as_zarr_group
from ..readers import H5adReader
from ..readers.h5ad import _H5adAssayFeatures
from ..storage.budget import ResourceBudget
from ..storage.count_matrix import CountMatrixPolicy
from ..storage.io_policy import StorageIoPolicy
from ..storage.profiles import (
    StorageProfile,
    ZarrLocation,
    resolve_storage_profile,
)
from ..utils.logging import logger
from ..utils.progress import iter_progress


def _count_matrix_bands(
    matrix: Any,
    buffers: dict[str, Any],
    assay_names: tuple[str, ...],
    projection: tuple[np.ndarray, np.ndarray] | None,
) -> Iterator[tuple[str, Any, int]]:
    from ..storage.sharding import sparse_matrix_bytes

    chunk = matrix.tocoo(copy=False)
    source_bytes = sparse_matrix_bytes(matrix, chunk)
    if projection is None:
        if len(assay_names) != 1:
            raise RuntimeError("Multi-assay projection was not initialized")
        assay_name = assay_names[0]
        buffer = buffers[assay_name]
        for band in buffer.add(chunk):
            producer_bytes = source_bytes + sum(
                item.residentBytes for item in buffers.values()
            )
            yield assay_name, band, producer_bytes
        return

    codes, columns = projection
    batch_codes = codes[chunk.col]
    batch_columns = columns[chunk.col]
    source_bytes += batch_codes.nbytes + batch_columns.nbytes
    for code, assay_name in enumerate(assay_names):
        buffer = buffers[assay_name]
        selected = batch_codes == code
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
            yield assay_name, band, producer_bytes


def _finished_count_bands(
    buffers: dict[str, Any],
) -> Iterator[tuple[str, Any, int]]:
    for assay_name, buffer in buffers.items():
        for band in buffer.finish():
            producer_bytes = sum(item.residentBytes for item in buffers.values())
            yield assay_name, band, producer_bytes


def _read_h5ad_process_window(
    reader_kwargs: dict[str, Any],
    batch_size: int,
    row_start: int,
    row_end: int,
    destination_specs: dict[str, Any],
    assay_names: tuple[str, ...],
    projection: tuple[np.ndarray, np.ndarray] | None,
    connection: Any,
    stop: Any,
) -> None:
    from ..storage.sharding import SparseShardBuffer

    reader: H5adReader | None = None

    def send_band(item: tuple[str, Any, int]) -> bool:
        if stop.is_set():
            return False
        connection.send(("band", item))
        return not stop.is_set()

    try:
        reader = H5adReader(**reader_kwargs)
        buffers = {
            assay_name: SparseShardBuffer(
                destination,
                startRow=row_start,
                endRow=row_end,
            )
            for assay_name, destination in destination_specs.items()
        }
        for matrix in reader.consume_row_range(batch_size, row_start, row_end):
            if stop.is_set():
                return
            for item in _count_matrix_bands(
                matrix,
                buffers,
                assay_names,
                projection,
            ):
                if not send_band(item):
                    return
        for item in _finished_count_bands(buffers):
            if not send_band(item):
                return
        connection.send(("done", None))
    except BaseException as exc:
        try:
            connection.send(("error", f"{type(exc).__name__}: {exc}"))
        except BaseException:
            pass
    finally:
        if reader is not None:
            reader.h5.close()
        connection.close()


def _write_h5ad_process_window(
    reader_kwargs: dict[str, Any],
    batch_size: int,
    row_start: int,
    row_end: int,
    zarr_location: str,
    storage_options: dict[str, Any] | None,
    workspace: str | None,
    assay_names: tuple[str, ...],
    projection: tuple[np.ndarray, np.ndarray] | None,
    resources: ResourceBudget,
    resident_bytes: int,
    producer_reserve_bytes: int,
    io: StorageIoPolicy | None,
    connection: Any,
    stop: Any,
) -> None:
    from ..storage.execution import execution_report_scope
    from ..storage.schema import load_count_array
    from ..storage.sharding import (
        SparseShardBuffer,
        SparseWriteBand,
        write_sparse_bands,
    )
    from ..storage.stores import load_zarr

    reader: H5adReader | None = None
    try:
        reader = H5adReader(**reader_kwargs)
        assert reader is not None
        root = load_zarr(
            zarr_location,
            mode="a",
            storage_options=storage_options,
        )
        destinations = {
            assay_name: load_count_array(root, assay_name, workspace)
            for assay_name in assay_names
        }
        buffers = {
            assay_name: SparseShardBuffer(
                destination,
                startRow=row_start,
                endRow=row_end,
            )
            for assay_name, destination in destinations.items()
        }

        def writes() -> Iterator[SparseWriteBand]:
            for matrix in reader.consume_row_range(batch_size, row_start, row_end):
                if stop.is_set():
                    return
                for assay_name, band, producer_bytes in _count_matrix_bands(
                    matrix,
                    buffers,
                    assay_names,
                    projection,
                ):
                    yield SparseWriteBand(
                        destinations[assay_name],
                        band,
                        producer_bytes,
                    )
            if stop.is_set():
                return
            for assay_name, band, producer_bytes in _finished_count_bands(buffers):
                yield SparseWriteBand(
                    destinations[assay_name],
                    band,
                    producer_bytes,
                )

        with execution_report_scope() as reports:
            write_sparse_bands(
                writes(),
                resources=resources,
                residentBytes=resident_bytes,
                producerReserveBytes=producer_reserve_bytes,
                io=io,
            )
        if not stop.is_set():
            connection.send(("done", reports))
    except BaseException as exc:
        try:
            connection.send(("error", f"{type(exc).__name__}: {exc}"))
        except BaseException:
            pass
    finally:
        if reader is not None:
            reader.h5.close()
        connection.close()


def _validate_assay_names(names: tuple[str, ...]) -> None:
    from ..storage.schema import validate_assay_name

    for name in names:
        validate_assay_name(name)


class H5adToZarr:
    """A class for converting data in anndata's H5ad format to Zarr hierarchy.

    Args:
        h5ad: Reader for the source H5AD file.
        zarr_loc: The file name for the Zarr hierarchy or a store
        assay_name: the name of the assay (e. g. 'RNA')
        assay_split_key: A var column used to split features into assays.
        assay_name_map: Feature type to assay name overrides.
        workspace: An optional workspace id. None uses the legacy layout
                   without a workspace group.
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
        policy: CountMatrixPolicy | None = None,
        io: StorageIoPolicy | None = None,
        assay_split_key: str | None = None,
        assay_name_map: dict[str, str] | None = None,
    ) -> None:
        from ..storage.budget import resolve_budget
        from ..storage.schema import create_zarr_count_assay
        from ..storage.stores import load_zarr

        self.h5ad = h5ad
        self.workspace = workspace
        self.storage_options = storage_options
        self._parallelWriteLocation = zarr_loc if isinstance(zarr_loc, str) else None
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
            logger.debug("Using RNA as the default assay name")
            self.assayName = "RNA"
            self.assayFeatures = None
            self.assayNames = (self.assayName,)
        else:
            self.assayName = assay_name
            self.assayFeatures = None
            self.assayNames = (self.assayName,)
        _validate_assay_names(self.assayNames)
        self.resources = resolve_budget(mem_budget, nthreads)
        self.profile = resolve_storage_profile(zarr_loc, profile)
        self.policy = policy
        self.io = io
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
                policy=policy,
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

    def dump(self, batch_size: int | None = None) -> None:
        """Write h5ad matrix data into Zarr ``counts`` and RNA ``countsT``.

        Args:
            batch_size: Number of source cells per batch. By default, a
                        destination-aligned value is selected within the memory budget.

        Raises:
            AssertionError: If written cell count does not match expected nCells.

        Returns:
            None
        """
        self._write_counts(batch_size=batch_size)
        from .counts_t import finalize_writer_counts_t_many

        finalize_writer_counts_t_many(
            self.z,
            self.assayNames,
            self.workspace,
            resources=self.resources,
            profile=self.profile,
            policy=self.policy,
            io=self.io,
        )

    def _write_counts(self, batch_size: int | None = None) -> None:
        """Write cell-major ``counts`` only (profiling stage split helper)."""
        from ..storage.layout import array_shard_rows
        from ..storage.sharding import (
            SparseShardBuffer,
            aligned_row_windows,
            resolve_sparse_import_batch,
            write_sparse_bands,
        )
        from ..storage.schema import load_count_array

        if batch_size is not None and batch_size <= 0:
            raise ValueError("batch_size must be positive")

        destinations = {
            assay_name: load_count_array(self.z, assay_name, self.workspace)
            for assay_name in self.assayNames
        }
        buffers = {
            assay_name: SparseShardBuffer(destination)
            for assay_name, destination in destinations.items()
        }
        logger.debug(
            f"Writing counts with up to {self.resources.workers} row-band writer(s)"
        )
        started = time.perf_counter()
        resident_source_bytes = self.h5ad.materialized_csr_bytes()
        if self.assayFeatures is None:
            feature_index_bytes = 0
        else:
            feature_index_bytes = sum(
                assay.featureIndexes.nbytes for assay in self.assayFeatures.values()
            )
        projection = (
            None if self.assayFeatures is None else self._assay_feature_projection()
        )
        if projection is not None:
            resident_source_bytes += sum(array.nbytes for array in projection)
        resident_source_bytes += feature_index_bytes
        prepare = getattr(self.h5ad, "_prepare_sparse_import", None)
        if callable(prepare):
            prepare()
        reader_resident = getattr(
            self.h5ad,
            "_sparse_import_resident_bytes",
            None,
        )
        if callable(reader_resident):
            resident_source_bytes += max(0, int(reader_resident()))
        source_itemsize = np.dtype(self.h5ad.sourceMatrixDtype).itemsize
        projection_value_bytes = max(
            source_itemsize,
            *(destination.dtype.itemsize for destination in destinations.values()),
        )

        def extra_producer_bytes(rows: int) -> int:
            source_rows = min(rows, self.h5ad.nCells)
            extra = (
                source_rows * self.h5ad.nFeatures * source_itemsize
                if self.h5ad.matrixOrientation == "dense"
                else 0
            )
            if self.assayFeatures is not None:
                source_values = max(0, int(self.h5ad.max_batch_nnz(rows)))
                extra += source_values * (
                    projection_value_bytes
                    + 4 * np.dtype(np.int64).itemsize
                    + np.dtype(np.bool_).itemsize
                )
            return extra

        plan = resolve_sparse_import_batch(
            tuple(destinations.values()),
            nRows=self.h5ad.nCells,
            resources=self.resources,
            maxWindowNnz=self.h5ad.max_batch_nnz,
            sourceDtype=self.h5ad.sourceMatrixDtype,
            batchRows=batch_size,
            residentBytes=resident_source_bytes,
            producerStagingBytes=self.h5ad.producer_batch_staging_bytes,
            extraProducerBytes=extra_producer_bytes,
        )
        self._lastImportPlan = plan
        resolved_batch_rows = plan.batchRows
        shard_rows = math.lcm(
            *(array_shard_rows(destination) for destination in destinations.values())
        )
        producer_worker_limit = max(1, (int(self.resources.workers) + 1) // 2)
        if self.io is not None and self.io.readWorkers is not None:
            producer_worker_limit = min(
                producer_worker_limit,
                max(1, int(self.io.readWorkers)),
            )
        if self.h5ad.matrixOrientation == "csc":
            producer_worker_limit = 1
        available_windows = max(
            1,
            min(
                producer_worker_limit,
                (self.h5ad.nCells + shard_rows - 1) // shard_rows,
            ),
        )
        n_producers = 1
        process_resources: ResourceBudget | None = None
        process_plan = None
        for candidate in range(available_windows, 0, -1):
            direct_candidate = candidate > 1 and isinstance(
                self._parallelWriteLocation, str
            )
            if direct_candidate:
                available_memory = self.resources.memoryBytes - resident_source_bytes
                if available_memory < candidate:
                    continue
                candidate_resources = ResourceBudget(
                    available_memory // candidate,
                    max(1, int(self.resources.workers) // candidate),
                )
                candidate_resident = resident_source_bytes
                candidate_batch_rows = None
            else:
                candidate_resources = self.resources
                candidate_resident = resident_source_bytes + max(
                    0, candidate - 1
                ) * int(plan.producerReserveBytes)
                candidate_batch_rows = resolved_batch_rows
            try:
                candidate_plan = resolve_sparse_import_batch(
                    tuple(destinations.values()),
                    nRows=self.h5ad.nCells,
                    resources=candidate_resources,
                    maxWindowNnz=self.h5ad.max_batch_nnz,
                    sourceDtype=self.h5ad.sourceMatrixDtype,
                    batchRows=candidate_batch_rows,
                    residentBytes=candidate_resident,
                    producerStagingBytes=self.h5ad.producer_batch_staging_bytes,
                    extraProducerBytes=extra_producer_bytes,
                )
                if (
                    direct_candidate
                    and batch_size is not None
                    and candidate_plan.batchRows > resolved_batch_rows
                ):
                    candidate_plan = resolve_sparse_import_batch(
                        tuple(destinations.values()),
                        nRows=self.h5ad.nCells,
                        resources=candidate_resources,
                        maxWindowNnz=self.h5ad.max_batch_nnz,
                        sourceDtype=self.h5ad.sourceMatrixDtype,
                        batchRows=resolved_batch_rows,
                        residentBytes=candidate_resident,
                        producerStagingBytes=self.h5ad.producer_batch_staging_bytes,
                        extraProducerBytes=extra_producer_bytes,
                    )
            except MemoryError:
                continue
            n_producers = candidate
            process_resources = candidate_resources if direct_candidate else None
            process_plan = candidate_plan if direct_candidate else None
            break
        windows = aligned_row_windows(
            self.h5ad.nCells,
            shard_rows,
            n_producers,
        )
        n_producers = max(1, len(windows))
        self._lastImportProducerCount = n_producers
        direct_process_writes = (
            n_producers > 1
            and process_resources is not None
            and isinstance(self._parallelWriteLocation, str)
        )
        if direct_process_writes:
            assert process_resources is not None
            assert process_plan is not None
            self._lastImportPlan = process_plan
            resolved_batch_rows = process_plan.batchRows
            workers_per_process = process_resources.workers
            write_workers = n_producers * workers_per_process
        else:
            workers_per_process = None
            write_workers = (
                int(self.resources.workers)
                if n_producers == 1
                else max(1, int(self.resources.workers) - n_producers)
            )
        self._lastImportWorkersPerProcess = workers_per_process
        self._lastImportWriteWorkers = write_workers
        write_resources = ResourceBudget(self.resources.memoryBytes, write_workers)
        logger.info(
            f"Resolved H5AD source batch rows={resolved_batch_rows} "
            f"write_tasks={plan.writeTasks} producers={n_producers} "
            f"writers={write_workers} workers_per_process={workers_per_process}"
        )
        if direct_process_writes:
            assert process_resources is not None
            self._write_parallel_count_windows(
                resolved_batch_rows,
                projection,
                windows,
                resources=process_resources,
                residentBytes=resident_source_bytes,
                producerReserveBytes=self._lastImportPlan.producerReserveBytes,
            )
        else:
            extra_producer_resident = max(0, n_producers - 1) * int(
                plan.producerReserveBytes
            )
            write_sparse_bands(
                self._count_shard_tasks(
                    resolved_batch_rows,
                    buffers,
                    destinations,
                    projection,
                    windows=windows,
                ),
                resources=write_resources,
                residentBytes=resident_source_bytes + extra_producer_resident,
                producerReserveBytes=plan.producerReserveBytes,
                total=plan.writeTasks,
                io=self.io,
            )
        counts_seconds = time.perf_counter() - started

        covered = sum(end - start for start, end in windows)
        if n_producers <= 1:
            for assay_name, buffer in buffers.items():
                if buffer.rows != self.h5ad.nCells:
                    raise AssertionError(
                        "ERROR: This is a bug in H5adToZarr. All cells might not have "
                        f"been successfully written into the {assay_name} counts "
                        "array. Please report this issue"
                    )
        elif covered != self.h5ad.nCells:
            raise AssertionError(
                "ERROR: This is a bug in H5adToZarr. Parallel producers covered "
                f"{covered} cells, expected {self.h5ad.nCells}"
            )
        # counts is the durable physical orientation for H5AD imports. Public
        # dump() always finalizes paired RNA countsT after this step.
        logger.debug(f"Counts written in {counts_seconds:.1f}s")
        logger.info(
            f"Wrote {self.h5ad.nCells} cells and {self.h5ad.nFeatures} features "
            f"from H5AD to {len(destinations)} assay(s)"
        )

    def _write_parallel_count_windows(
        self,
        batch_size: int,
        projection: tuple[np.ndarray, np.ndarray] | None,
        windows: list[tuple[int, int]],
        *,
        resources: ResourceBudget,
        residentBytes: int,
        producerReserveBytes: int,
    ) -> None:
        from multiprocessing import get_context
        from multiprocessing.connection import wait

        from ..storage.execution import record_execution_report

        if not isinstance(self._parallelWriteLocation, str):
            raise RuntimeError("Parallel H5AD writes require a reopenable location")
        context = get_context("spawn")
        stop = context.Event()
        connections: dict[Any, int] = {}
        workers: list[Any] = []
        try:
            for index, (row_start, row_end) in enumerate(windows):
                parent_connection, child_connection = context.Pipe(duplex=False)
                worker = context.Process(
                    target=_write_h5ad_process_window,
                    args=(
                        self.h5ad._clone_kwargs(),
                        batch_size,
                        row_start,
                        row_end,
                        self._parallelWriteLocation,
                        self.storage_options,
                        self.workspace,
                        self.assayNames,
                        projection,
                        resources,
                        residentBytes,
                        producerReserveBytes,
                        self.io,
                        child_connection,
                        stop,
                    ),
                    name=f"h5ad-writer-{index}",
                    daemon=True,
                )
                worker.start()
                child_connection.close()
                connections[parent_connection] = index
                workers.append(worker)

            active = dict(connections)
            while active:
                ready = cast(list[Any], wait(tuple(active), timeout=0.5))
                if not ready:
                    failed = [
                        workers[index]
                        for index in active.values()
                        if workers[index].exitcode is not None
                    ]
                    if failed:
                        details = ", ".join(
                            f"{worker.name} exitcode={worker.exitcode}"
                            for worker in failed
                        )
                        raise RuntimeError(f"H5AD writer process failed: {details}")
                    continue
                for connection in ready:
                    index = active[connection]
                    try:
                        kind, payload = connection.recv()
                    except EOFError as exc:
                        raise RuntimeError(
                            f"H5AD writer {index} closed without a result"
                        ) from exc
                    if kind == "error":
                        raise RuntimeError(f"H5AD writer {index} failed: {payload}")
                    if kind != "done":
                        raise RuntimeError(
                            f"H5AD writer {index} sent an unknown message"
                        )
                    for report in payload:
                        record_execution_report(report)
                    active.pop(connection)
                    connection.close()
        finally:
            stop.set()
            deadline = time.monotonic() + 5.0
            for worker in workers:
                worker.join(timeout=max(0.0, deadline - time.monotonic()))
            for worker in workers:
                if worker.is_alive():
                    worker.terminate()
            for worker in workers:
                worker.join(timeout=1.0)
                if worker.is_alive():
                    worker.kill()
                    worker.join()
            for connection in connections:
                connection.close()

    def _count_shard_tasks(
        self,
        batch_size: int,
        buffers: dict[str, Any],
        destinations: dict[str, Any],
        projection: tuple[np.ndarray, np.ndarray] | None,
        windows: list[tuple[int, int]] | None = None,
    ) -> Iterator[Any]:
        """Yield complete row-band writes from disjoint H5AD producers."""
        ranges = windows or [(0, self.h5ad.nCells)]
        if len(ranges) <= 1:
            yield from self._emit_count_bands(
                self.h5ad,
                batch_size,
                buffers,
                destinations,
                projection,
                row_start=ranges[0][0] if ranges else 0,
                row_end=ranges[0][1] if ranges else 0,
            )
            return
        yield from self._parallel_count_bands(
            batch_size,
            destinations,
            projection,
            ranges,
        )

    def _parallel_count_bands(
        self,
        batch_size: int,
        destinations: dict[str, Any],
        projection: tuple[np.ndarray, np.ndarray] | None,
        windows: list[tuple[int, int]],
    ) -> Iterator[Any]:
        from multiprocessing import get_context
        from multiprocessing.connection import wait

        from ..storage.geometry import array_geometry
        from ..storage.layout import ZarrArraySpec
        from ..storage.sharding import SparseWriteBand

        context = get_context("spawn")
        stop = context.Event()
        connections: dict[Any, int] = {}
        child_connections: list[Any] = []
        workers: list[Any] = []
        destination_specs: dict[str, ZarrArraySpec] = {}
        for assay_name, destination in destinations.items():
            geometry = array_geometry(destination)
            if geometry is None:
                raise ValueError("H5AD destination has no stored chunk geometry")
            destination_specs[assay_name] = ZarrArraySpec(
                shape=geometry.shape,
                chunks=geometry.chunks,
                shards=geometry.shards,
                dtype=destination.dtype,
                compressors=(),
            )
        try:
            for index, (row_start, row_end) in enumerate(windows):
                parent_connection, child_connection = context.Pipe(duplex=True)
                worker = context.Process(
                    target=_read_h5ad_process_window,
                    args=(
                        self.h5ad._clone_kwargs(),
                        batch_size,
                        row_start,
                        row_end,
                        destination_specs,
                        self.assayNames,
                        projection,
                        child_connection,
                        stop,
                    ),
                    name=f"h5ad-producer-{index}",
                    daemon=True,
                )
                worker.start()
                child_connection.close()
                connections[parent_connection] = index
                child_connections.append(parent_connection)
                workers.append(worker)

            active = dict(connections)
            while active:
                ready = cast(list[Any], wait(tuple(active), timeout=0.5))
                if not ready:
                    failed = [
                        workers[index]
                        for index in active.values()
                        if workers[index].exitcode is not None
                    ]
                    if failed:
                        details = ", ".join(
                            f"{worker.name} exitcode={worker.exitcode}"
                            for worker in failed
                        )
                        raise RuntimeError(f"H5AD producer process failed: {details}")
                    continue
                for connection in ready:
                    index = active[connection]
                    try:
                        kind, payload = connection.recv()
                    except EOFError as exc:
                        raise RuntimeError(
                            f"H5AD producer {index} closed without a result"
                        ) from exc
                    if kind == "error":
                        raise RuntimeError(f"H5AD producer {index} failed: {payload}")
                    if kind == "done":
                        active.pop(connection)
                        connection.close()
                        continue
                    if kind != "band":
                        raise RuntimeError(
                            f"H5AD producer {index} sent an unknown message"
                        )
                    assay_name, band, producer_bytes = payload
                    yield SparseWriteBand(
                        destinations[assay_name],
                        band,
                        producer_bytes,
                    )
        finally:
            stop.set()
            deadline = time.monotonic() + 5.0
            while (
                any(worker.is_alive() for worker in workers)
                and time.monotonic() < deadline
            ):
                open_connections = [
                    connection
                    for connection in child_connections
                    if not connection.closed
                ]
                ready = cast(
                    list[Any],
                    wait(tuple(open_connections), timeout=0.05)
                    if open_connections
                    else [],
                )
                for connection in ready:
                    try:
                        connection.recv()
                    except (EOFError, OSError):
                        pass
                for worker in workers:
                    worker.join(timeout=0)
            for worker in workers:
                if worker.is_alive():
                    worker.terminate()
            for worker in workers:
                worker.join(timeout=1.0)
                if worker.is_alive():
                    worker.kill()
                    worker.join()
            for connection in child_connections:
                connection.close()

    def _emit_count_bands(
        self,
        reader: Any,
        batch_size: int,
        buffers: dict[str, Any],
        destinations: dict[str, Any],
        projection: tuple[np.ndarray, np.ndarray] | None,
        *,
        row_start: int,
        row_end: int,
    ) -> Iterator[Any]:
        """Yield complete row-band writes from one reader over a row window."""
        n_rows = max(0, row_end - row_start)
        n_batches = (n_rows + batch_size - 1) // batch_size if n_rows else 0
        stream = iter_progress(
            reader.consume_row_range(batch_size, row_start, row_end),
            total=n_batches,
            desc="Writing counts",
        )
        for matrix in stream:
            yield from self._emit_count_matrix_bands(
                matrix,
                buffers,
                destinations,
                projection,
            )
        yield from self._finish_count_buffers(buffers, destinations)

    def _emit_count_matrix_bands(
        self,
        matrix: Any,
        buffers: dict[str, Any],
        destinations: dict[str, Any],
        projection: tuple[np.ndarray, np.ndarray] | None,
    ) -> Iterator[Any]:
        from ..storage.sharding import SparseWriteBand

        for assay_name, band, producer_bytes in _count_matrix_bands(
            matrix,
            buffers,
            self.assayNames,
            projection,
        ):
            yield SparseWriteBand(
                destinations[assay_name],
                band,
                producer_bytes,
            )

    def _finish_count_buffers(
        self,
        buffers: dict[str, Any],
        destinations: dict[str, Any],
    ) -> Iterator[Any]:
        from ..storage.sharding import SparseWriteBand

        for assay_name, band, producer_bytes in _finished_count_bands(buffers):
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
