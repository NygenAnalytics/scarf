import builtins

import pytest

import scarf.storage.budget as budget_module
from scarf.storage.budget import (
    ResourceBudget,
    admitted_worker_count,
    admitted_worker_split,
    detect_total_memory_bytes,
    detect_workers,
    resolve_budget,
)


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


def test_resolve_budget_fraction_of_total(monkeypatch):
    monkeypatch.delenv("SCARF_MEM_BUDGET", raising=False)
    total = detect_total_memory_bytes()
    budget = resolve_budget(memory="0.5", workers=1)
    assert abs(budget.memoryBytes - int(total * 0.5)) <= total * 0.01


def test_resolve_budget_default_is_total_memory(monkeypatch):
    monkeypatch.delenv("SCARF_MEM_BUDGET", raising=False)
    monkeypatch.delenv("SCARF_WORKERS", raising=False)
    budget = resolve_budget()
    assert budget.memoryBytes == detect_total_memory_bytes()
    assert budget.workers >= 1


def test_detect_memory_fallback_when_meminfo_absent(monkeypatch):
    real_open = builtins.open

    def fake_open(path, *args, **kwargs):
        if str(path) == "/proc/meminfo":
            raise OSError("no meminfo")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", fake_open)
    monkeypatch.setattr("os.sysconf", lambda name: -1)
    assert detect_total_memory_bytes() == 8 * 1024 * 1024 * 1024


@pytest.mark.parametrize(
    "spec, expected",
    [
        (8 * 1024**3, 8 * 1024**3),
        ("8G", 8 * 1024**3),
        ("512M", 512 * 1024**2),
        (12_345_678, 12_345_678),
    ],
)
def test_memory_spec_valid(spec, expected, monkeypatch):
    monkeypatch.delenv("SCARF_MEM_BUDGET", raising=False)
    assert resolve_budget(memory=spec, workers=1).memoryBytes == expected


@pytest.mark.parametrize("spec", ["1.0", "100", "abc", "-5", "0", "8Q", ""])
def test_memory_spec_invalid_or_ambiguous_rejected(spec, monkeypatch):
    monkeypatch.delenv("SCARF_MEM_BUDGET", raising=False)
    with pytest.raises(ValueError):
        resolve_budget(memory=spec, workers=1)


def test_memory_spec_bool_rejected(monkeypatch):
    monkeypatch.delenv("SCARF_MEM_BUDGET", raising=False)
    with pytest.raises(ValueError, match="Invalid memory spec"):
        resolve_budget(memory=True, workers=1)


def test_detect_memory_uses_sysconf_when_meminfo_unavailable(monkeypatch):
    real_open = builtins.open

    def fake_open(path, *args, **kwargs):
        if str(path) == "/proc/meminfo":
            raise OSError("no meminfo")
        return real_open(path, *args, **kwargs)

    def fake_sysconf(name):
        if name == "SC_PAGE_SIZE":
            return 4096
        if name == "SC_PHYS_PAGES":
            return 1024
        return -1

    monkeypatch.setattr(builtins, "open", fake_open)
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


def test_process_cgroup_path_parses_unified_and_legacy_entries(monkeypatch):
    content = "0::/batch/job.scope\n5:cpu,cpuacct:/legacy/job.scope\n"
    monkeypatch.setattr("pathlib.Path.read_text", lambda path: content)

    assert budget_module._process_cgroup_path("") == "batch/job.scope"
    assert budget_module._process_cgroup_path("cpu") == "legacy/job.scope"
    assert budget_module._process_cgroup_path("memory") is None


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
    monkeypatch.setattr("os.sched_getaffinity", lambda pid: set(range(16)))

    assert detect_workers() == 2


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
    monkeypatch.setattr("os.sched_getaffinity", lambda pid: set(range(16)))

    assert detect_workers() == 2


def test_invalid_workers_env_rejected(monkeypatch):
    monkeypatch.delenv("SCARF_MEM_BUDGET", raising=False)
    monkeypatch.setenv("SCARF_WORKERS", "not-a-number")
    with pytest.raises(ValueError):
        resolve_budget(memory="8G")


def test_fraction_uses_total_memory(monkeypatch):
    monkeypatch.delenv("SCARF_MEM_BUDGET", raising=False)
    total = detect_total_memory_bytes()
    got = resolve_budget(memory="0.25", workers=1).memoryBytes
    assert abs(got - int(total * 0.25)) <= total * 0.01


def test_admitted_worker_count_respects_cpu_memory_and_resident_bytes():
    resources = ResourceBudget(memoryBytes=4 * 1024**3, workers=8)
    assert admitted_worker_count(resources, taskBytes=1024**3) == 4
    assert (
        admitted_worker_count(
            resources,
            taskBytes=1024**3,
            residentBytes=1024**3,
        )
        == 3
    )
    assert (
        admitted_worker_count(
            resources,
            taskBytes=1024**3,
            requested=2,
        )
        == 2
    )


def test_admitted_worker_count_rejects_oversized_task():
    resources = ResourceBudget(memoryBytes=1024, workers=8)
    with pytest.raises(MemoryError):
        admitted_worker_count(resources, taskBytes=2048)


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
