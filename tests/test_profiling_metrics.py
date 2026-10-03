import json
import threading
from dataclasses import asdict, fields

from profiling.metrics import (
    ResourceMeasurement,
    ResourceSampler,
    StageTimer,
    child_cpu_seconds,
    read_process_tree_rss_bytes,
    stage_utilization,
)


def _write_status(proc_root, pid, parent_pid, rss_kib):
    process = proc_root / str(pid)
    process.mkdir(parents=True, exist_ok=True)
    (process / "status").write_text(
        f"Name:\tprocess-{pid}\nPPid:\t{parent_pid}\nVmRSS:\t{rss_kib} kB\n"
    )


class _SampleWatch:
    """List /proc pids like the sampler does and count how often it samples."""

    def __init__(self) -> None:
        self.samples = 0
        self._changed = threading.Condition()

    def __call__(self, proc_root):
        with self._changed:
            self.samples += 1
            self._changed.notify_all()
        return [
            int(entry.name) for entry in proc_root.iterdir() if entry.name.isdigit()
        ]

    def wait_for_fresh_sample(self) -> None:
        """Return once a sample that started after this call has completed."""
        with self._changed:
            target = self.samples + 2
            assert self._changed.wait_for(lambda: self.samples >= target, timeout=5)


def _write_cgroup(cgroup, *, current=100, peak=900):
    cgroup.mkdir(parents=True, exist_ok=True)
    (cgroup / "memory.current").write_text(str(current))
    (cgroup / "memory.peak").write_text(str(peak))
    (cgroup / "memory.max").write_text("4096")
    (cgroup / "memory.events").write_text(
        "low 0\nhigh 1\nmax 2\noom 0\noom_kill 0\noom_group_kill 0\n"
    )
    (cgroup / "cpu.max").write_text("200000 100000")


def test_process_tree_rss_aggregates_descendants_by_parent_pid(tmp_path):
    proc_root = tmp_path / "proc"
    _write_status(proc_root, 100, 1, 10)
    _write_status(proc_root, 101, 100, 20)
    _write_status(proc_root, 102, 101, 30)
    _write_status(proc_root, 200, 1, 400)
    (proc_root / "103").mkdir()

    assert read_process_tree_rss_bytes(100, procRoot=proc_root) == (10 + 20 + 30) * 1024


def test_sampler_captures_events_limits_cpu_quota_and_operation_peak(tmp_path):
    proc_root = tmp_path / "proc"
    cgroup = tmp_path / "cgroup"
    _write_status(proc_root, 100, 1, 10)
    _write_status(proc_root, 101, 100, 20)
    _write_cgroup(cgroup)

    watch = _SampleWatch()
    sampler = ResourceSampler(
        sampleIntervalSeconds=0.005,
        rootPid=100,
        procRoot=proc_root,
        cgroupPath=cgroup,
        listPids=watch,
    )
    sampler.start()

    assert (cgroup / "memory.peak").read_text() == "0"
    _write_status(proc_root, 100, 1, 30)
    _write_status(proc_root, 101, 100, 40)
    (cgroup / "memory.current").write_text("350")
    (cgroup / "memory.peak").write_text("420")
    watch.wait_for_fresh_sample()

    _write_status(proc_root, 100, 1, 15)
    _write_status(proc_root, 101, 100, 10)
    (cgroup / "memory.current").write_text("200")
    (cgroup / "memory.events").write_text(
        "low 0\nhigh 3\nmax 5\noom 1\noom_kill 1\noom_group_kill 1\n"
    )
    result = sampler.stop()

    assert result.processTreeRssBaselineBytes == 30 * 1024
    assert result.processTreeRssPeakBytes == 70 * 1024
    assert result.processTreeRssIncrementalPeakBytes == 40 * 1024
    assert result.processTreeRssAfterBytes == 25 * 1024
    assert result.cgroupMemoryCurrentBaselineBytes == 100
    assert result.cgroupMemoryCurrentPeakBytes == 350
    assert result.cgroupMemoryCurrentAfterBytes == 200
    assert result.cgroupMemoryPeakBytes == 420
    assert result.cgroupMemoryPeakScope == "operation"
    assert result.operationBaselineBytes == 100
    assert result.operationPeakBytes == 420
    assert result.operationIncrementalPeakBytes == 320
    assert result.operationPeakSource == "cgroupMemoryPeak"
    assert result.memoryEventsDelta == {
        "high": 2,
        "low": 0,
        "max": 3,
        "oom": 1,
        "oomGroupKill": 1,
        "oomKill": 1,
    }
    assert result.memoryMaxBytes == 4096
    assert result.cpuQuotaCores == 2.0
    assert all("_" not in field.name for field in fields(ResourceMeasurement))
    json.dumps(asdict(result))


def test_unresettable_cgroup_peak_is_labeled_container_lifetime(tmp_path):
    proc_root = tmp_path / "proc"
    cgroup = tmp_path / "cgroup"
    _write_status(proc_root, 100, 1, 10)
    _write_cgroup(cgroup, current=100, peak=1000)

    watch = _SampleWatch()
    sampler = ResourceSampler(
        sampleIntervalSeconds=0.005,
        rootPid=100,
        procRoot=proc_root,
        cgroupPath=cgroup,
        writeText=lambda _path, _value: None,
        listPids=watch,
    ).start()
    (cgroup / "memory.current").write_text("300")
    watch.wait_for_fresh_sample()
    (cgroup / "memory.current").write_text("200")
    result = sampler.stop()

    assert result.cgroupMemoryPeakScope == "containerLifetime"
    assert result.cgroupMemoryPeakBytes == 1000
    assert result.operationPeakBytes == 300
    assert result.operationPeakSource == "cgroupMemoryCurrent"
    assert result.operationIncrementalPeakBytes == 200


def test_sampler_can_preserve_an_outer_cgroup_peak(tmp_path):
    proc_root = tmp_path / "proc"
    cgroup = tmp_path / "cgroup"
    _write_status(proc_root, 100, 1, 10)
    _write_cgroup(cgroup, current=100, peak=1000)

    watch = _SampleWatch()
    sampler = ResourceSampler(
        sampleIntervalSeconds=0.005,
        rootPid=100,
        procRoot=proc_root,
        cgroupPath=cgroup,
        resetCgroupPeak=False,
        listPids=watch,
    ).start()

    assert (cgroup / "memory.peak").read_text() == "1000"
    (cgroup / "memory.current").write_text("300")
    watch.wait_for_fresh_sample()
    (cgroup / "memory.current").write_text("200")
    result = sampler.stop()

    assert result.cgroupMemoryPeakScope == "containerLifetime"
    assert result.cgroupMemoryPeakBytes == 1000
    assert result.operationPeakBytes == 300
    assert result.operationPeakSource == "cgroupMemoryCurrent"


def test_sampler_reads_cgroup_v1_memory_files(tmp_path):
    proc_root = tmp_path / "proc"
    cgroup_root = tmp_path / "cgroup"
    memory = cgroup_root / "memory" / "job-1"
    memory.mkdir(parents=True)
    (memory / "memory.usage_in_bytes").write_text("150")
    (memory / "memory.max_usage_in_bytes").write_text("900")
    (memory / "memory.limit_in_bytes").write_text("4096")
    proc = proc_root / "100"
    proc.mkdir(parents=True)
    (proc / "status").write_text("Name:\tpython\nPPid:\t1\nVmRSS:\t20 kB\n")
    (proc / "cgroup").write_text("6:memory:/job-1\n")

    watch = _SampleWatch()
    sampler = ResourceSampler(
        sampleIntervalSeconds=0.005,
        rootPid=100,
        procRoot=proc_root,
        cgroupRoot=cgroup_root,
        listPids=watch,
    ).start()
    (memory / "memory.usage_in_bytes").write_text("400")
    watch.wait_for_fresh_sample()
    (memory / "memory.usage_in_bytes").write_text("200")
    result = sampler.stop()

    assert result.cgroupMemoryCurrentBaselineBytes == 150
    assert result.cgroupMemoryCurrentPeakBytes == 400
    assert result.memoryMaxBytes == 4096
    assert result.cgroupMemoryPeakScope == "containerLifetime"
    assert result.cgroupMemoryPeakBytes == 900
    assert result.operationPeakSource == "cgroupMemoryCurrent"
    assert result.operationPeakBytes == 400


def test_sampler_falls_back_to_flat_cgroup_v1_memory(tmp_path):
    proc_root = tmp_path / "proc"
    cgroup_root = tmp_path / "cgroup"
    flat = cgroup_root / "memory"
    flat.mkdir(parents=True)
    (flat / "memory.usage_in_bytes").write_text("120")
    (flat / "memory.max_usage_in_bytes").write_text("500")
    (flat / "memory.limit_in_bytes").write_text("4096")
    proc = proc_root / "100"
    proc.mkdir(parents=True)
    (proc / "status").write_text("Name:\tpython\nPPid:\t1\nVmRSS:\t20 kB\n")
    (proc / "cgroup").write_text("6:memory:/ta-missing-nested\n")

    watch = _SampleWatch()
    sampler = ResourceSampler(
        sampleIntervalSeconds=0.005,
        rootPid=100,
        procRoot=proc_root,
        cgroupRoot=cgroup_root,
        listPids=watch,
    ).start()
    (flat / "memory.usage_in_bytes").write_text("300")
    watch.wait_for_fresh_sample()
    (flat / "memory.usage_in_bytes").write_text("150")
    result = sampler.stop()

    assert result.cgroupMemoryCurrentBaselineBytes == 120
    assert result.cgroupMemoryCurrentPeakBytes == 300
    assert result.operationPeakSource == "cgroupMemoryCurrent"
    assert result.memoryEventsDelta is None


def test_background_sampler_observes_process_and_cgroup_peaks(tmp_path):
    proc_root = tmp_path / "proc"
    cgroup = tmp_path / "cgroup"
    _write_status(proc_root, 100, 1, 10)
    _write_status(proc_root, 101, 100, 10)
    cgroup.mkdir()
    (cgroup / "memory.current").write_text("100")

    watch = _SampleWatch()
    sampler = ResourceSampler(
        sampleIntervalSeconds=0.005,
        rootPid=100,
        procRoot=proc_root,
        cgroupPath=cgroup,
        listPids=watch,
    ).start()
    _write_status(proc_root, 100, 1, 50)
    _write_status(proc_root, 101, 100, 70)
    (cgroup / "memory.current").write_text("800")
    watch.wait_for_fresh_sample()

    _write_status(proc_root, 100, 1, 5)
    _write_status(proc_root, 101, 100, 5)
    (cgroup / "memory.current").write_text("50")
    result = sampler.stop()

    assert result.processTreeRssPeakBytes == 120 * 1024
    assert result.cgroupMemoryCurrentPeakBytes == 800
    assert result.operationPeakBytes == 800
    assert result.cgroupMemoryPeakScope == "unavailable"


def test_unavailable_and_protected_metrics_are_empty_not_errors(tmp_path):
    listings = threading.Semaphore(0)

    def unavailable(*_args):
        raise PermissionError("protected")

    def protected_pids(*_args):
        listings.release()
        raise PermissionError("protected")

    sampler = ResourceSampler(
        sampleIntervalSeconds=0.002,
        rootPid=100,
        procRoot=tmp_path / "proc",
        cgroupPath=tmp_path / "cgroup",
        readText=unavailable,
        writeText=unavailable,
        listPids=protected_pids,
    ).start()
    # The baseline sample, then at least one background sample, hit the errors.
    assert listings.acquire(timeout=5)
    assert listings.acquire(timeout=5)
    result = sampler.stop()

    assert result.processTreeRssPeakBytes is None
    assert result.cgroupMemoryCurrentPeakBytes is None
    assert result.cgroupMemoryPeakBytes is None
    assert result.cgroupMemoryPeakScope == "unavailable"
    assert result.memoryEventsDelta is None
    assert result.memoryMaxBytes is None
    assert result.cpuQuotaCores is None
    json.dumps(asdict(result))


def test_stage_timer_records_four_independent_scopes():
    ticks = iter((0.0, 1.0, 3.0, 4.0, 9.0, 10.0, 12.0, 15.0))
    timer = StageTimer(clock=lambda: next(ticks))

    with timer:
        with timer.inputSetup():
            pass
        with timer.operation():
            pass
        with timer.validationPersistence():
            pass

    assert timer.result.inputSetupSeconds == 2.0
    assert timer.result.measuredOperationSeconds == 5.0
    assert timer.result.validationPersistenceSeconds == 2.0
    assert timer.result.wholeFunctionSeconds == 15.0
    assert asdict(timer.result) == {
        "inputSetupSeconds": 2.0,
        "measuredOperationSeconds": 5.0,
        "validationPersistenceSeconds": 2.0,
        "wholeFunctionSeconds": 15.0,
    }


def test_child_cpu_seconds_counts_finished_children() -> None:
    import subprocess
    import sys

    before = child_cpu_seconds()
    completed = subprocess.run(
        [sys.executable, "-c", "sum(range(2_000_000))"],
        check=True,
    )
    assert completed.returncode == 0
    after = child_cpu_seconds()
    assert after >= before
    assert after - before > 0.0


def test_stage_utilization_relates_cpu_and_peaks_to_limits() -> None:
    gib = 1024**3
    result = {
        "processCpuSeconds": 30.0,
        "childCpuSeconds": 10.0,
        "wholeFunctionSeconds": 10.0,
        "modalCpuLimit": 8.0,
        "workers": 4,
        "modalMemoryMb": 4096,
        "scarfMemoryBudget": 2 * gib,
        "peakCgroupBytes": 3 * gib,
        "peakRssBytes": gib,
    }

    assert stage_utilization(result) == {
        "cpuCores": 4.0,
        "cpuPercentOfLimit": 50.0,
        "cpuPercentOfWorkers": 100.0,
        "peakCgroupPercentOfContainer": 75.0,
        "peakCgroupPercentOfBudget": 150.0,
        "peakRssPercentOfContainer": 25.0,
        "peakRssPercentOfBudget": 50.0,
    }


def test_stage_utilization_leaves_unrecorded_inputs_empty() -> None:
    summary = stage_utilization(
        {
            "processCpuSeconds": 2.0,
            "childCpuSeconds": None,
            "wholeFunctionSeconds": 4.0,
            "modalCpuLimit": 1.0,
            "workers": None,
            "modalMemoryMb": 1024,
            "scarfMemoryBudget": None,
            "peakCgroupBytes": None,
            "peakRssBytes": 256 * 1024**2,
        }
    )

    assert summary["cpuCores"] == 0.5
    assert summary["cpuPercentOfLimit"] == 50.0
    assert summary["cpuPercentOfWorkers"] is None
    assert summary["peakCgroupPercentOfContainer"] is None
    assert summary["peakRssPercentOfContainer"] == 25.0
    assert summary["peakRssPercentOfBudget"] is None
    assert set(stage_utilization({}).values()) == {None}


def test_store_probe_keeps_totals_and_resets() -> None:
    from profiling.recording_store import StoreProbe

    probe = StoreProbe()
    probe.enter("get", "a/key", requestedBytes=12)
    probe.record_transfer("get", "a/key", 12)
    probe.leave("get")
    probe.enter("set", "b/key", requestedBytes=8)
    probe.record_transfer("set", "b/key", 8)
    probe.leave("set")

    assert probe.to_json() == {
        "gets": 1,
        "sets": 1,
        "deletes": 0,
        "deleteDirs": 0,
        "sizeQueries": 0,
        "rangeGets": 0,
        "partialGets": 0,
        "requestedBytes": 20,
        "transferredBytes": 20,
        "readRequestedBytes": 12,
        "readTransferredBytes": 12,
        "writeTransferredBytes": 8,
        "maxInFlight": 1,
    }
    probe.reset()
    assert set(probe.to_json().values()) == {0}


def test_store_probe_reset_keeps_operations_in_flight() -> None:
    from profiling.recording_store import StoreProbe

    probe = StoreProbe()
    probe.enter("get", "slow/key")
    probe.reset()
    probe.enter("get", "other/key")
    probe.leave("get")
    probe.leave("get")

    assert probe.to_json()["maxInFlight"] == 2
    assert probe.to_json()["gets"] == 1


def test_recording_wrapper_forwards_store_specific_operations() -> None:
    """Profiling must take the wrapped store's own path, not zarr's generic one."""
    import asyncio

    from obstore.store import MemoryStore as ObjectMemory
    from zarr.core.buffer import default_buffer_prototype
    from zarr.storage import ObjectStore

    from profiling.recording_store import StoreProbe, wrap_recording_store

    class Spy(ObjectStore):
        calls: list[str] = []

        async def getsize(self, key):
            self.calls.append("getsize")
            return await super().getsize(key)

        async def getsize_prefix(self, prefix):
            self.calls.append("getsize_prefix")
            return await super().getsize_prefix(prefix)

        async def set_if_not_exists(self, key, value):
            self.calls.append("set_if_not_exists")
            await super().set_if_not_exists(key, value)

        async def delete_dir(self, prefix):
            self.calls.append("delete_dir")
            await super().delete_dir(prefix)

    buffer = default_buffer_prototype().buffer
    inner = Spy(ObjectMemory())
    probe = StoreProbe()
    wrapped = wrap_recording_store(inner, probe=probe)

    async def exercise() -> None:
        await inner.set("big", buffer.from_bytes(b"x" * 1000))
        assert await wrapped.getsize("big") == 1000
        assert await wrapped.getsize_prefix("") == 1000
        await wrapped.set_if_not_exists("big", buffer.from_bytes(b"y"))
        await inner.set("dir/a", buffer.from_bytes(b"a"))
        await wrapped.delete_dir("dir")

    asyncio.run(exercise())

    assert Spy.calls == [
        "getsize",
        "getsize_prefix",
        "set_if_not_exists",
        "delete_dir",
    ]
    counts = probe.to_json()
    # Size queries never download the object they measure.
    assert counts["gets"] == 0
    assert counts["readTransferredBytes"] == 0
    assert counts["sizeQueries"] == 2
    assert counts["sets"] == 1
    assert counts["deleteDirs"] == 1
