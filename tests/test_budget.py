import builtins

import pytest

import scarf.storage.budget as budget_module
from scarf.storage.budget import (
    ResourceBudget,
    detect_total_memory_bytes,
    detect_workers,
    resolve_budget,
)
from scarf.storage.execution import admitted_worker_split


def _serve_files(monkeypatch: pytest.MonkeyPatch, files: dict[str, str]) -> None:
    """Serve ``Path.read_text`` from ``files``; every other path is missing."""

    def read_text(path, *_args, **_kwargs):
        try:
            return files[str(path)]
        except KeyError:
            raise OSError(f"missing {path}") from None

    monkeypatch.setattr("pathlib.Path.read_text", read_text)


def _without_meminfo(monkeypatch: pytest.MonkeyPatch) -> None:
    real_open = builtins.open

    def fake_open(path, *args, **kwargs):
        if str(path) == "/proc/meminfo":
            raise OSError("no meminfo")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", fake_open)
    # Physical memory alone decides the total, whatever cgroup this host runs in.
    monkeypatch.setattr(budget_module, "_cgroup_memory_bytes", lambda: None)


def test_resolve_budget_parses_suffix(monkeypatch):
    monkeypatch.setenv("SCARF_MEM_BUDGET", "8G")
    monkeypatch.setenv("SCARF_WORKERS", "3")
    budget = resolve_budget()
    assert budget.memoryBytes == 8 * 1024**3
    assert budget.workers == 3


def test_resolve_budget_raw_bytes_and_explicit_workers(monkeypatch):
    monkeypatch.delenv("SCARF_MEM_BUDGET", raising=False)
    budget = resolve_budget(memory=12345678, workers=2)
    assert budget.memoryBytes == 12345678
    assert budget.workers == 2


@pytest.mark.parametrize(
    ("spec", "expected"),
    [("0.5", 5_000_000_000), (0.25, 2_500_000_000), ("0.999", 9_990_000_000)],
)
def test_resolve_budget_takes_fractions_of_the_detected_total(
    monkeypatch, spec, expected
):
    monkeypatch.delenv("SCARF_MEM_BUDGET", raising=False)
    monkeypatch.setattr(budget_module, "detect_total_memory_bytes", lambda: 10**10)
    assert resolve_budget(memory=spec, workers=1).memoryBytes == expected
    monkeypatch.setenv("SCARF_MEM_BUDGET", str(spec))
    assert resolve_budget(workers=1).memoryBytes == expected


def test_resolve_budget_defaults_to_detected_memory_and_workers(monkeypatch):
    monkeypatch.delenv("SCARF_MEM_BUDGET", raising=False)
    monkeypatch.delenv("SCARF_WORKERS", raising=False)
    monkeypatch.setattr(budget_module, "detect_total_memory_bytes", lambda: 123_456)
    monkeypatch.setattr(budget_module, "detect_workers", lambda: 7)
    assert resolve_budget() == ResourceBudget(memoryBytes=123_456, workers=7)
    # Explicit values win over detection, and are kept at least one.
    assert resolve_budget(memory="2M", workers=0) == ResourceBudget(2 * 1024**2, 1)


def test_detect_memory_fallback_when_meminfo_and_sysconf_fail(monkeypatch):
    _without_meminfo(monkeypatch)
    monkeypatch.setattr("os.sysconf", lambda name: -1)
    assert detect_total_memory_bytes() == 8 * 1024**3

    def unsupported(name):
        raise ValueError(f"unrecognized configuration name {name}")

    monkeypatch.setattr("os.sysconf", unsupported)
    assert detect_total_memory_bytes() == 8 * 1024**3


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        (8 * 1024**3, 8 * 1024**3),
        ("8G", 8 * 1024**3),
        (" 512m ", 512 * 1024**2),
        ("1.5G", 3 * 1024**3 // 2),
        ("2T", 2 * 1024**4),
        (12_345_678, 12_345_678),
        # A bare number of at least 1 MiB is read as bytes.
        ("2097152", 2_097_152),
    ],
)
def test_memory_spec_valid(spec, expected, monkeypatch):
    monkeypatch.delenv("SCARF_MEM_BUDGET", raising=False)
    assert resolve_budget(memory=spec, workers=1).memoryBytes == expected


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ("1.0", "Ambiguous memory spec '1.0'"),
        ("100", "Ambiguous memory spec '100'"),
        ("abc", "Invalid memory spec: 'abc'"),
        ("8Q", "Invalid memory spec: '8Q'"),
        ("abcG", "Invalid memory spec: 'abcG'"),
        (True, "Invalid memory spec: True"),
        ("-5", "Memory budget must be positive, got '-5'"),
        ("0", "Memory budget must be positive, got '0'"),
        ("0G", "Memory budget must be positive, got '0G'"),
        ("-1G", "Memory budget must be positive, got '-1G'"),
        (0, "Memory budget must be positive, got 0"),
        (-5, "Memory budget must be positive, got -5"),
        ("", "Empty memory spec"),
        ("   ", "Empty memory spec"),
    ],
)
def test_memory_spec_invalid_or_ambiguous_rejected(spec, message, monkeypatch):
    monkeypatch.delenv("SCARF_MEM_BUDGET", raising=False)
    with pytest.raises(ValueError, match=f"^{message}"):
        resolve_budget(memory=spec, workers=1)


def test_detect_memory_uses_sysconf_when_meminfo_unavailable(monkeypatch):
    _without_meminfo(monkeypatch)

    def fake_sysconf(name):
        if name == "SC_PAGE_SIZE":
            return 4096
        if name == "SC_PHYS_PAGES":
            return 1024
        return -1

    monkeypatch.setattr("os.sysconf", fake_sysconf)
    assert detect_total_memory_bytes() == 4096 * 1024


def test_detect_memory_uses_process_cgroup_path(monkeypatch):
    nested_limit = 2 * 1024**3
    monkeypatch.setattr(
        budget_module,
        "_process_cgroup_path",
        lambda controller: "batch/job.scope" if controller == "" else None,
    )
    monkeypatch.setattr(
        budget_module,
        "_read_int",
        lambda path: (
            nested_limit if path == "/sys/fs/cgroup/batch/memory.max" else None
        ),
    )
    monkeypatch.setattr(budget_module, "_physical_memory_bytes", lambda: 16 * 1024**3)

    assert detect_total_memory_bytes() == nested_limit


def test_cgroup_memory_ignores_unlimited_and_unreadable_limits(monkeypatch):
    _serve_files(
        monkeypatch,
        {
            "/proc/self/cgroup": "malformed\n0::/batch/job.scope\n4:memory:/legacy/job\n",
            "/sys/fs/cgroup/batch/job.scope/memory.max": "not-a-number\n",
            # Values at or above 2**60 are how cgroup v1 spells "no limit".
            "/sys/fs/cgroup/batch/memory.max": str(1 << 62),
            "/sys/fs/cgroup/memory.max": "max\n",
            "/sys/fs/cgroup/memory/legacy/job/memory.limit_in_bytes": "0\n",
            "/sys/fs/cgroup/memory/legacy/memory.limit_in_bytes": str(3 * 1024**3),
        },
    )
    monkeypatch.setattr(budget_module, "_physical_memory_bytes", lambda: 16 * 1024**3)

    # Only the legacy parent directory holds a usable limit.
    assert detect_total_memory_bytes() == 3 * 1024**3
    # Physical memory hides an unlimited value behind the minimum, so read
    # the cgroup limit alone: an unlimited value is no limit at all.
    _serve_files(
        monkeypatch,
        {
            "/proc/self/cgroup": "0::/batch\n",
            "/sys/fs/cgroup/batch/memory.max": str(1 << 60),
        },
    )
    assert budget_module._cgroup_memory_bytes() is None


def test_process_cgroup_path_parses_unified_and_legacy_entries(monkeypatch):
    content = "garbage\n0::/batch/job.scope\n5:cpu,cpuacct:/legacy/job.scope\n"
    _serve_files(monkeypatch, {"/proc/self/cgroup": content})

    assert budget_module._process_cgroup_path("") == "batch/job.scope"
    assert budget_module._process_cgroup_path("cpu") == "legacy/job.scope"
    assert budget_module._process_cgroup_entry("cpuacct") == (
        ["cpu", "cpuacct"],
        "legacy/job.scope",
    )
    assert budget_module._process_cgroup_path("memory") is None
    _serve_files(monkeypatch, {})
    assert budget_module._process_cgroup_path("") is None


def test_detect_workers_uses_process_cgroup_path(monkeypatch):
    monkeypatch.setattr(
        budget_module,
        "_process_cgroup_path",
        lambda controller: "batch/job.scope" if controller == "" else None,
    )

    def read_text(path):
        if str(path) == "/sys/fs/cgroup/batch/job.scope/cpu.max":
            return "250000 100000"
        raise OSError("missing")

    monkeypatch.setattr("pathlib.Path.read_text", read_text)
    monkeypatch.setattr("os.cpu_count", lambda: 16)
    monkeypatch.setattr(
        "os.sched_getaffinity",
        lambda pid: set(range(16)),
        raising=False,
    )

    assert detect_workers() == 2


def test_detect_workers_skips_unreadable_quotas_and_affinity(monkeypatch):
    _serve_files(
        monkeypatch,
        {
            "/proc/self/cgroup": "0::/batch/job.scope\n",
            "/sys/fs/cgroup/batch/job.scope/cpu.max": "not-a-number 100000\n",
            "/sys/fs/cgroup/batch/cpu.max": "max 100000\n",
            "/sys/fs/cgroup/cpu.max": "300000 100000\n",
        },
    )
    monkeypatch.setattr("os.cpu_count", lambda: 16)

    def no_affinity(_pid):
        raise OSError("affinity unavailable")

    monkeypatch.setattr("os.sched_getaffinity", no_affinity, raising=False)

    # The root quota of three CPUs binds; the others carry no usable limit.
    assert detect_workers() == 3


def test_detect_workers_uses_combined_legacy_controller_mount(monkeypatch):
    monkeypatch.setattr(
        budget_module,
        "_process_cgroup_entry",
        lambda controller: (
            (["cpu", "cpuacct"], "batch/job.scope") if controller == "cpu" else None
        ),
    )

    def read_text(path):
        values = {
            "/sys/fs/cgroup/cpu,cpuacct/batch/cpu.cfs_quota_us": "200000",
            "/sys/fs/cgroup/cpu,cpuacct/batch/cpu.cfs_period_us": "100000",
        }
        try:
            return values[str(path)]
        except KeyError:
            raise OSError("missing") from None

    monkeypatch.setattr("pathlib.Path.read_text", read_text)
    monkeypatch.setattr("os.cpu_count", lambda: 16)
    monkeypatch.setattr(
        "os.sched_getaffinity",
        lambda pid: set(range(16)),
        raising=False,
    )

    assert detect_workers() == 2


def test_invalid_workers_env_rejected(monkeypatch):
    monkeypatch.delenv("SCARF_MEM_BUDGET", raising=False)
    monkeypatch.setenv("SCARF_WORKERS", "not-a-number")
    with pytest.raises(
        ValueError, match="Invalid SCARF_WORKERS='not-a-number'; expected an integer"
    ):
        resolve_budget(memory="8G")


def test_admitted_worker_split_bounds_outer_inner_and_resident_bytes():
    resources = ResourceBudget(memoryBytes=500, workers=8)
    outer, inner = admitted_worker_split(
        resources,
        nTasks=20,
        taskBytes=lambda concurrency: 100 + 10 * concurrency,
        residentBytes=100,
    )
    assert (outer, inner) == (3, 2)
    assert outer * inner <= resources.workers
    assert 100 + outer * (100 + 10 * inner) <= resources.memoryBytes


def test_admitted_worker_split_rejects_resident_data_at_limit():
    resources = ResourceBudget(memoryBytes=500, workers=8)
    with pytest.raises(MemoryError, match="Resident data"):
        admitted_worker_split(
            resources,
            nTasks=1,
            taskBytes=lambda _: 1,
            residentBytes=500,
        )


def test_admitted_worker_split_can_reduce_inner_concurrency_to_fit():
    resources = ResourceBudget(memoryBytes=150, workers=8)
    outer, inner = admitted_worker_split(
        resources,
        nTasks=1,
        taskBytes=lambda concurrency: 100 + 20 * concurrency,
    )
    assert (outer, inner) == (1, 2)
