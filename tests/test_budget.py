import builtins

import pytest

from scarf.storage.budget import (
    READ_AHEAD,
    ResourceBudget,
    detect_total_memory_bytes,
    resolve_budget,
    worker_prefetch_depth,
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


def test_worker_prefetch_depth_capped_by_read_ahead():
    budget = ResourceBudget(memoryBytes=4 * 1024**3, workers=4, workingCopies=8)
    assert worker_prefetch_depth(budget=budget) == READ_AHEAD
    assert worker_prefetch_depth(requested=1, budget=budget) == 1
    assert worker_prefetch_depth(requested=10, budget=budget) == READ_AHEAD
    assert worker_prefetch_depth(requested=0, budget=budget) == 1


def test_worker_prefetch_depth_capped_by_working_copies():
    budget = ResourceBudget(memoryBytes=4 * 1024**3, workers=4, workingCopies=1)
    assert worker_prefetch_depth(budget=budget) == 1


def test_working_copies_from_env(monkeypatch):
    monkeypatch.delenv("SCARF_MEM_BUDGET", raising=False)
    monkeypatch.setenv("SCARF_WORKING_COPIES", "8")
    budget = resolve_budget(memory="4G", workers=1)
    assert budget.workingCopies == 8


def test_layout_geometry_independent_of_workers():
    from scarf.storage.zarr_store import matrix_layout

    one = ResourceBudget(memoryBytes=8 * 1024**3, workers=1, workingCopies=4)
    eight = ResourceBudget(memoryBytes=8 * 1024**3, workers=8, workingCopies=4)
    c1, s1 = matrix_layout(1_000_000, 50_000, budget=one, itemsize=4)
    c8, s8 = matrix_layout(1_000_000, 50_000, budget=eight, itemsize=4)
    assert c1 == c8
    assert s1 == s8
