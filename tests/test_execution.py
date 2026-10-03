import numpy as np
import pytest

from scarf.storage.budget import ResourceBudget
from scarf.storage.execution import (
    ExecutionReport,
    WorkShape,
    auto_read_width,
    execution_report_scope,
    plan_operation,
)
from scarf.storage.io_policy import StorageIoPolicy


def test_plan_operation_reserves_inner_reads() -> None:
    resources = ResourceBudget(16 * 1024 * 1024, 8)
    plan = plan_operation(
        resources,
        WorkShape(
            nUnits=8,
            unitBytes=256 * 1024,
            innerReadBytes=64 * 1024,
            chunksPerShard=4,
            writes=True,
        ),
    )
    # Eight units read at once, each with one inner read per chunk of its shard.
    assert (plan.readWorkers, plan.innerReads, plan.writeWorkers) == (8, 4, 8)
    assert plan.reservedBytes == 8 * 256 * 1024 + 8 * 4 * 64 * 1024
    assert plan.reservedBytes <= resources.memoryBytes


def test_plan_operation_caps_inner_reads() -> None:
    resources = ResourceBudget(64 * 1024 * 1024, 8)
    plan = plan_operation(
        resources,
        WorkShape(
            nUnits=8,
            unitBytes=1024 * 1024,
            innerReadBytes=256 * 1024,
            maxInnerReads=2,
            chunksPerShard=8,
        ),
        policy=StorageIoPolicy(readWorkers=4),
    )

    assert plan.readWorkers == 4
    assert plan.innerReads == 2
    assert plan.reservedBytes == 6 * 1024 * 1024


def test_units_in_flight_share_the_requested_read_width() -> None:
    shape = WorkShape(
        nUnits=8,
        unitBytes=100,
        innerReadBytes=10,
        maxInnerReads=16,
        maxUnitsInFlight=2,
        ordered=True,
    )
    request = StorageIoPolicy(readWorkers=32)

    # The units in flight stop at two, and their inner reads make up the
    # requested width instead of being capped at the unit count.
    full = plan_operation(ResourceBudget(10_000, 4), shape, policy=request)
    assert (full.readWorkers, full.innerReads) == (2, 16)
    assert full.reservedBytes == 2 * 100 + 2 * 16 * 10
    assert full.reductionReason is None

    tight = plan_operation(ResourceBudget(400, 4), shape, policy=request)
    assert (tight.readWorkers, tight.innerReads) == (2, 10)
    assert tight.reservedBytes <= 400
    assert tight.reductionReason == (
        "20 reads used because each unit needs 100 bytes and each read 10 bytes"
    )


def test_plan_operation_couples_writers_to_memory_bounded_inner_reads() -> None:
    resources = ResourceBudget(10 * 1024 * 1024, 8)
    plan = plan_operation(
        resources,
        WorkShape(
            nUnits=64,
            unitBytes=3 * 1024 * 1024,
            innerReadBytes=1024 * 1024,
            writes=True,
        ),
        policy=StorageIoPolicy(readWorkers=64),
    )

    assert plan.readWorkers == 2
    assert plan.writeWorkers == 2
    assert plan.innerReads == 1
    assert plan.reservedBytes <= resources.memoryBytes


def test_plan_operation_is_deterministic() -> None:
    resources = ResourceBudget(64 * 1024 * 1024, 8)
    shape = WorkShape(nUnits=16, unitBytes=1024 * 1024, ordered=True)
    first = plan_operation(resources, shape)
    second = plan_operation(resources, shape)
    assert first == second
    assert first.computeWorkers == 1
    assert first.readWorkers == 16
    assert first.reservedBytes <= resources.memoryBytes


def test_plan_operation_read_width_exceeds_compute_workers() -> None:
    resources = ResourceBudget(64 * 1024 * 1024, 8)
    plan = plan_operation(
        resources,
        WorkShape(nUnits=64, unitBytes=256 * 1024),
    )
    assert auto_read_width(8) == 64
    assert plan.readWorkers == 64
    assert plan.computeWorkers == 8
    assert plan.ioConcurrency == 1
    assert plan.readWorkers > plan.requestedWorkers


def test_auto_read_width_scales_past_sixty_four_lanes() -> None:
    resources = ResourceBudget(256 * 1024 * 1024, 16)
    plan = plan_operation(
        resources,
        WorkShape(nUnits=256, unitBytes=1024 * 1024),
    )

    assert auto_read_width(16) == 128
    assert plan.readWorkers == 128
    assert plan.computeWorkers == 16
    assert plan.reservedBytes <= resources.memoryBytes


def test_nested_reads_use_memory_beyond_compute_width() -> None:
    resources = ResourceBudget(100, 4)
    plan = plan_operation(
        resources,
        WorkShape(
            nUnits=32,
            unitBytes=10,
            innerReadBytes=2,
            chunksPerShard=4,
        ),
    )

    assert plan.readWorkers == 5
    assert plan.innerReads == 4
    assert plan.readWorkers > plan.computeWorkers
    assert plan.reservedBytes == 90


def test_plan_operation_respects_ceilings() -> None:
    resources = ResourceBudget(64 * 1024 * 1024, 8)
    plan = plan_operation(
        resources,
        WorkShape(nUnits=16, unitBytes=1024 * 1024, writes=True),
        policy=StorageIoPolicy(readWorkers=3, computeWorkers=2, writeWorkers=2),
    )
    # Memory admits all 16 units, so the requested widths bind, and the eight
    # workers split their threads between two compute workers.
    assert (plan.readWorkers, plan.computeWorkers, plan.writeWorkers) == (3, 2, 2)
    assert plan.threadsPerComputeWorker == 4
    assert plan.reservedBytes == 3 * 1024 * 1024
    assert plan.reductionReason is None


def test_plan_operation_reduces_when_memory_is_tight() -> None:
    resources = ResourceBudget(8 * 1024 * 1024, 8)
    plan = plan_operation(
        resources,
        WorkShape(nUnits=16, unitBytes=3 * 1024 * 1024),
        policy=StorageIoPolicy(readWorkers=8),
    )
    assert plan.readWorkers == 2
    assert plan.reservedBytes == 2 * 3 * 1024 * 1024
    assert plan.reductionReason == (
        "2 readers used because each unit needs 3145728 bytes; "
        "2 compute workers used because each unit needs 3145728 bytes"
    )


def test_plan_operation_rejects_one_unit_over_budget() -> None:
    with pytest.raises(
        MemoryError,
        match="One unit needs about 2048 bytes in addition to 100 reserved bytes, "
        "but the operation limit is 1024 bytes",
    ):
        plan_operation(
            ResourceBudget(1024, 2),
            WorkShape(nUnits=4, unitBytes=2048, residentBytes=100),
        )


def test_map_feature_process_is_not_on_event_loop_thread() -> None:
    import asyncio

    import zarr
    from zarr.storage import MemoryStore

    from scarf.storage.count_matrix import (
        CountMatrixPolicy,
        persist_count_matrix_plan,
        plan_count_matrix_pair,
    )
    from scarf.storage.feature_stream import map_feature_read_groups

    values = np.arange(40 * 80, dtype=np.uint16).reshape(40, 80)
    plan = plan_count_matrix_pair(
        values.shape[0],
        values.shape[1],
        values.dtype,
        policy=CountMatrixPolicy(unitBytes=2_000, chunkBytes=200),
    )
    root = zarr.open_group(store=MemoryStore(), mode="w")
    group = root.create_group("RNA")
    counts_t = group.create_array(
        "countsT",
        shape=plan.countsT.shape,
        chunks=plan.countsT.chunks,
        shards=plan.countsT.shards,
        dtype=values.dtype,
        overwrite=True,
    )
    counts_t[:] = values.T
    persist_count_matrix_plan(group, plan)
    persist_count_matrix_plan(counts_t, plan)
    on_loop: list[bool] = []

    def process(group):  # type: ignore[no-untyped-def]
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            on_loop.append(False)
        else:
            on_loop.append(True)
        return group.featStart

    with execution_report_scope() as reports:
        starts = list(
            map_feature_read_groups(
                counts_t,
                process,
                resources=ResourceBudget(8 * 1024 * 1024, 2),
                io=StorageIoPolicy(readWorkers=2),
            )
        )
    # One read group per chunk of features, each processed off the loop.
    feature_chunk = int(counts_t.chunks[0])
    assert starts == list(range(0, 80, feature_chunk))
    assert on_loop == [False] * len(starts)
    report = reports[-1]
    assert report.unitKind == "countsTReadGroup"
    assert report.unitsCompleted == len(starts)


def test_plan_operation_write_workers_fit_memory() -> None:
    resources = ResourceBudget(8 * 1024 * 1024, 8)
    plan = plan_operation(
        resources,
        WorkShape(nUnits=16, unitBytes=3 * 1024 * 1024, writes=True),
    )
    assert plan.writeWorkers == 2
    assert plan.reservedBytes == 2 * 3 * 1024 * 1024
    assert plan.reductionReason == (
        "2 compute workers used because each unit needs 3145728 bytes; "
        "2 writers used because each unit needs 3145728 bytes"
    )


def test_plan_operation_reports_reason_for_few_units() -> None:
    plan = plan_operation(
        ResourceBudget(64 * 1024 * 1024, 8),
        WorkShape(nUnits=2, unitBytes=1024),
    )
    assert plan.readWorkers == 2
    assert plan.computeWorkers == 2
    assert plan.reductionReason == "2 compute workers used because there are 2 units"


def test_plan_operation_is_independent_of_call_order() -> None:
    small = WorkShape(nUnits=2, unitBytes=1024, ordered=True)
    large = WorkShape(nUnits=32, unitBytes=1024 * 1024, writes=True)
    resources = ResourceBudget(64 * 1024 * 1024, 8)
    first_small = plan_operation(resources, small)
    first_large = plan_operation(resources, large)
    second_large = plan_operation(resources, large)
    second_small = plan_operation(resources, small)
    assert first_small == second_small
    assert first_large == second_large
    assert first_large.readWorkers >= first_small.readWorkers


def test_detect_external_thread_caps(monkeypatch: pytest.MonkeyPatch) -> None:
    from scarf.storage.execution import detect_external_thread_caps

    monkeypatch.setenv("NUMBA_NUM_THREADS", "3")
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    caps = detect_external_thread_caps()
    assert caps["NUMBA_NUM_THREADS"] == 3
    monkeypatch.setenv("OMP_NUM_THREADS", "not-an-int")
    caps = detect_external_thread_caps()
    assert "OMP_NUM_THREADS" not in caps
    monkeypatch.setenv("OMP_NUM_THREADS", "0")
    assert "OMP_NUM_THREADS" not in detect_external_thread_caps()


def test_write_counts_t_records_execution_report() -> None:
    import zarr
    from zarr.storage import MemoryStore

    from scarf.storage.count_matrix import (
        CountMatrixPolicy,
        persist_count_matrix_plan,
        plan_count_matrix_pair,
    )
    from scarf.storage.sharding import write_counts_t

    values = np.arange(16 * 16, dtype=np.uint16).reshape(16, 16)
    policy = CountMatrixPolicy(unitBytes=2_000, chunkBytes=200)
    layout = plan_count_matrix_pair(
        values.shape[0],
        values.shape[1],
        values.dtype,
        policy=policy,
    )
    root = zarr.open_group(store=MemoryStore(), mode="w")
    group = root.create_group("RNA")
    counts = group.create_array(
        "counts",
        shape=layout.counts.shape,
        chunks=layout.counts.chunks,
        shards=layout.counts.shards,
        dtype=values.dtype,
        overwrite=True,
    )
    counts[:] = values
    persist_count_matrix_plan(group, layout)
    persist_count_matrix_plan(counts, layout)
    from tests.storage_helpers import finalize_test_counts

    finalize_test_counts(counts)
    with execution_report_scope() as reports:
        write_counts_t(
            counts,
            group,
            resources=ResourceBudget(16 * 1024 * 1024, 4),
        )
    assert reports
    report = reports[-1]
    assert report.unitKind == "countsRowShard"
    assert report.plan.reservedBytes <= 16 * 1024 * 1024
    np.testing.assert_array_equal(np.asarray(group["countsT"][:]), values.T)


def test_plan_active_work_never_exceeds_requested_workers() -> None:
    resources = ResourceBudget(32 * 1024 * 1024, 6)
    plan = plan_operation(
        resources,
        WorkShape(nUnits=32, unitBytes=256 * 1024, writes=True),
        policy=StorageIoPolicy(readWorkers=5, computeWorkers=4, writeWorkers=4),
    )
    # Every requested width fits six workers, so each worker gets one thread.
    assert (plan.readWorkers, plan.computeWorkers, plan.writeWorkers) == (5, 4, 4)
    assert plan.threadsPerComputeWorker == 1
    assert plan.reservedBytes == 5 * 256 * 1024
    assert plan.requestedReadWorkers == 5
    assert plan.requestedComputeWorkers == 4
    assert plan.requestedWriteWorkers == 4


def test_storage_io_policy_is_only_three_widths() -> None:
    fields = set(StorageIoPolicy.__dataclass_fields__)
    assert fields == {"readWorkers", "computeWorkers", "writeWorkers"}


def test_recorded_report_includes_external_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scarf.storage.execution import (
        ExecutionReport,
        record_execution_report,
    )

    monkeypatch.setenv("NUMBA_NUM_THREADS", "2")
    plan = plan_operation(
        ResourceBudget(8 * 1024 * 1024, 4),
        WorkShape(nUnits=8, unitBytes=1024),
    )
    with execution_report_scope() as reports:
        report = record_execution_report(
            ExecutionReport(
                plan=plan,
                unitKind="countsRowBlock",
                actualReadWorkers=plan.readWorkers,
                actualComputeWorkers=plan.computeWorkers,
                actualWriteWorkers=1,
            )
        )
    assert report.extra["externalLimits"]["NUMBA_NUM_THREADS"] == 2
    assert reports == [report]


def test_write_sparse_bands_records_plan_and_respects_write_ceiling() -> None:
    import zarr
    from zarr.storage import MemoryStore

    from scarf.storage.sharding import (
        SparseRowBand,
        SparseWriteBand,
        write_sparse_bands,
    )

    root = zarr.open_group(store=MemoryStore(), mode="w")
    destination = root.create_array(
        "counts",
        shape=(8, 3),
        chunks=(2, 3),
        shards=(4, 3),
        dtype=np.uint16,
        fill_value=0,
    )
    expected = np.arange(1, 25, dtype=np.uint16).reshape(8, 3)

    def writes():  # type: ignore[no-untyped-def]
        for start in (0, 4):
            yield SparseWriteBand(
                destination=destination,
                band=SparseRowBand(
                    start=start,
                    end=start + 4,
                    nColumns=3,
                    row=np.repeat(np.arange(4, dtype=np.int64), 3),
                    column=np.tile(np.arange(3, dtype=np.int64), 4),
                    data=expected[start : start + 4].ravel(),
                    dtype=np.uint16,
                ),
            )

    with execution_report_scope() as reports:
        write_sparse_bands(
            writes(),
            resources=ResourceBudget(2 * 1024 * 1024, 4),
            io=StorageIoPolicy(writeWorkers=1),
        )
    import_reports = [r for r in reports if r.unitKind == "countsImportBand"]
    assert len(import_reports) == 1
    report = import_reports[0]
    assert report.plan.writeWorkers == 1
    assert report.actualWriteWorkers <= 1
    # One writer admits one band per batch, and the import reports them once.
    assert report.unitsCompleted == 2
    assert report.extra["batches"] == 2
    np.testing.assert_array_equal(destination[:], expected)


def test_chunked_row_stream_uses_shared_plan() -> None:
    import zarr
    from zarr.storage import MemoryStore

    from scarf.matrix import ChunkedArray

    values = np.arange(20 * 6, dtype=np.uint16).reshape(20, 6)
    root = zarr.open_group(store=MemoryStore(), mode="w")
    array = root.create_array(
        "counts",
        shape=values.shape,
        chunks=(5, 6),
        dtype=values.dtype,
        fill_value=0,
    )
    array[:] = values
    matrix = ChunkedArray(
        array,
        nthreads=8,
        resources=ResourceBudget(4 * 1024 * 1024, 8),
    )
    matrix._io = StorageIoPolicy(readWorkers=2, computeWorkers=1)
    with execution_report_scope() as reports:
        blocks = list(matrix.stream_blocks())
    assert blocks
    np.testing.assert_array_equal(np.vstack(blocks), values)
    assert reports
    report = reports[-1]
    assert report.unitKind == "countsRowBlock"
    assert report.plan.readWorkers <= 2
    assert report.plan.computeWorkers == 1
    assert report.plan.reservedBytes <= 4 * 1024 * 1024


def _write_h5ad(path, values: np.ndarray):  # type: ignore[no-untyped-def]
    """Write a minimal CSR AnnData file with ``feature_name`` names."""
    import h5py
    from scipy.sparse import csr_matrix

    matrix = csr_matrix(values)
    with h5py.File(path, "w") as h5:
        group = h5.create_group("X")
        group.attrs["encoding-type"] = "csr_matrix"
        group.attrs["shape"] = values.shape
        group.create_dataset("data", data=matrix.data)
        group.create_dataset("indices", data=matrix.indices)
        group.create_dataset("indptr", data=matrix.indptr)
        h5.create_group("obs").create_dataset(
            "_index", data=np.array([f"c{i}".encode() for i in range(values.shape[0])])
        )
        names = np.array([f"g{i}".encode() for i in range(values.shape[1])])
        var = h5.create_group("var")
        var.create_dataset("_index", data=names)
        var.create_dataset("feature_name", data=names)
        h5.create_group("obsm")
    return path


def test_h5ad_import_records_execution_report(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from scarf.readers import H5adReader
    from scarf.writers import H5adToZarr

    values = np.arange(1, 25, dtype=np.uint16).reshape(8, 3)
    path = _write_h5ad(tmp_path / "execution.h5ad", values)
    reader = H5adReader(str(path), feature_name_key="feature_name")
    try:
        writer = H5adToZarr(
            reader,
            zarr_loc=str(tmp_path / "store.zarr"),
            mem_budget="16M",
            nthreads=2,
            io=StorageIoPolicy(writeWorkers=1),
        )
        with execution_report_scope() as reports:
            writer.dump()
    finally:
        reader.h5.close()
    # The band import and the countsT transpose each honor the one writer.
    assert [report.unitKind for report in reports] == [
        "countsImportBand",
        "countsRowShard",
    ]
    for report in reports:
        assert report.plan.writeWorkers == report.actualWriteWorkers == 1
        assert report.unitsCompleted == 1


def _dummy_report(unitKind: str) -> ExecutionReport:
    plan = plan_operation(
        ResourceBudget(8 * 1024 * 1024, 4),
        WorkShape(nUnits=4, unitBytes=1024),
    )
    return ExecutionReport(
        plan=plan,
        unitKind=unitKind,
        actualReadWorkers=plan.readWorkers,
        actualComputeWorkers=plan.computeWorkers,
        actualWriteWorkers=1,
    )


def test_execution_reports_collect_by_kind_and_scope() -> None:
    from scarf.storage.execution import (
        execution_reports_by_kind,
        record_execution_report,
    )

    first = record_execution_report(_dummy_report("countsTCellBand"))
    with execution_report_scope() as scoped:
        second = record_execution_report(_dummy_report("countsTReadGroup"))
        third = record_execution_report(_dummy_report("countsTCellBand"))
    assert scoped == [second, third]
    grouped = execution_reports_by_kind([first, *scoped])
    assert [item["unitKind"] for item in grouped["countsTCellBand"]] == [
        "countsTCellBand",
        "countsTCellBand",
    ]
    assert len(grouped["countsTReadGroup"]) == 1


def test_feature_consume_details_uses_matching_kind_after_later_reports() -> None:
    from profiling.stages import _feature_consume_details
    from profiling.config import StageResources

    payload = _feature_consume_details(
        StageResources(
            modalMemoryRequestMb=1024,
            modalMemoryLimitMb=1024,
            modalCpuRequest=1.0,
            modalCpuLimit=1.0,
            scarfMemoryBudget=1024**3,
            workers=4,
            timeoutSeconds=60,
            ephemeralDiskMb=1024,
        ),
        [_dummy_report("countsTCellBand"), _dummy_report("countsRowBlock")],
        unitKind="countsTCellBand",
    )
    assert payload["unitKind"] == "countsTCellBand"


def test_pairwise_merge_tree_is_independent_of_completion_order() -> None:
    from scarf.utils.compute import add_stat_arrays, pairwise_merge_tree

    # A merge that records its association shows the fixed tree: neighbours
    # first, then neighbouring pairs, with an odd leftover promoted unchanged.
    def merge(left: str, right: str) -> str:
        return f"({left}+{right})"

    assert pairwise_merge_tree(list("abcde"), merge) == "(((a+b)+(c+d))+e)"
    assert pairwise_merge_tree(list("abcdef"), merge) == "(((a+b)+(c+d))+(e+f))"
    assert pairwise_merge_tree(["a"], merge) == "a"

    left = (np.array([1.0, 2.0]), np.array([3.0, 4.0]))
    right = (np.array([5.0, 6.0]), np.array([7.0, 8.0]))
    third = (np.array([9.0, 10.0]), np.array([11.0, 12.0]))
    merged = pairwise_merge_tree([left, right, third], add_stat_arrays)
    np.testing.assert_array_equal(merged[0], [15.0, 18.0])
    np.testing.assert_array_equal(merged[1], [21.0, 24.0])


def test_pairwise_merge_tree_rejects_empty_and_mismatched_stats() -> None:
    from scarf.utils.compute import add_stat_arrays, pairwise_merge_tree

    with pytest.raises(ValueError, match="at least one value"):
        pairwise_merge_tree([], lambda left, right: left + right)
    assert pairwise_merge_tree([1, 2, 3], lambda left, right: left + right) == 6
    with pytest.raises(ValueError, match="same length"):
        add_stat_arrays((np.ones(2),), (np.ones(2), np.ones(2)))


def test_plan_operation_error_and_reason_branches() -> None:
    from types import SimpleNamespace

    with pytest.raises(ValueError, match="must be positive"):
        plan_operation(
            ResourceBudget(1024 * 1024, 2),
            WorkShape(nUnits=2, unitBytes=64, writes=True),
            policy=StorageIoPolicy(readWorkers=0),
        )
    with pytest.raises(ValueError, match="must be positive"):
        plan_operation(
            ResourceBudget(1024 * 1024, 2),
            WorkShape(nUnits=2, unitBytes=64),
            policy=SimpleNamespace(
                readWorkers=0, computeWorkers=None, writeWorkers=None
            ),
        )
    with pytest.raises(
        MemoryError,
        match="One live unit plus one inner read needs about 32 bytes, "
        "but only 20 bytes are available",
    ):
        plan_operation(
            ResourceBudget(20, 2),
            WorkShape(
                nUnits=4,
                unitBytes=16,
                innerReadBytes=16,
                chunksPerShard=4,
            ),
        )
    plan = plan_operation(
        ResourceBudget(512, 8),
        WorkShape(nUnits=2, unitBytes=32, writes=False),
        policy=StorageIoPolicy(readWorkers=8, computeWorkers=8),
    )
    assert (plan.readWorkers, plan.computeWorkers) == (2, 2)
    assert plan.reductionReason == (
        "2 readers used because there are 2 units; "
        "2 compute workers used because there are 2 units"
    )
    with pytest.raises(MemoryError, match="Resident data"):
        plan_operation(
            ResourceBudget(8, 2),
            WorkShape(nUnits=2, unitBytes=4, residentBytes=16),
        )
    ordered = plan_operation(
        ResourceBudget(1024, 8),
        WorkShape(nUnits=2, unitBytes=32, ordered=True),
        policy=StorageIoPolicy(computeWorkers=8),
    )
    assert ordered.reductionReason is not None
    assert "accumulate in order" in ordered.reductionReason
    compute_capped = plan_operation(
        ResourceBudget(1024, 2),
        WorkShape(nUnits=32, unitBytes=16),
        policy=StorageIoPolicy(computeWorkers=8),
    )
    assert compute_capped.reductionReason is not None
    assert "compute workers used" in compute_capped.reductionReason


def test_execution_report_scopes_close_exactly_and_follow_the_context() -> None:
    import threading

    from scarf.storage.execution import execution_report_scope, record_execution_report

    outer_scope = execution_report_scope()
    outer = outer_scope.__enter__()
    inner_scope = execution_report_scope()
    inner = inner_scope.__enter__()
    # Scopes may close out of order; each removes only itself.
    outer_scope.__exit__(None, None, None)
    report = record_execution_report(_dummy_report("countsRowBlock"))
    inner_scope.__exit__(None, None, None)
    record_execution_report(_dummy_report("countsRowBlock"))
    assert outer == []
    assert inner == [report]

    with execution_report_scope() as scoped:
        thread = threading.Thread(
            target=record_execution_report, args=(_dummy_report("countsRowBlock"),)
        )
        thread.start()
        thread.join()
    assert scoped == []
