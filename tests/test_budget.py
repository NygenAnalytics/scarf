import builtins

import pytest

from scarf.storage.budget import (
    ResourceBudget,
    bounded_prefetch,
    detect_available_memory_bytes,
    resolve_budget,
    tile_rows_for_width,
)


def test_per_worker_bytes_scales_with_workers():
    one = ResourceBudget(memoryBytes=16 * 1024**3, workers=1)
    four = ResourceBudget(memoryBytes=16 * 1024**3, workers=4)
    assert one.perWorkerBytes == 16 * 1024**3
    assert four.perWorkerBytes == one.perWorkerBytes // 4


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


def test_resolve_budget_fraction_of_available(monkeypatch):
    monkeypatch.delenv("SCARF_MEM_BUDGET", raising=False)
    avail = detect_available_memory_bytes()
    budget = resolve_budget(memory="0.5", workers=1)
    assert abs(budget.memoryBytes - int(avail * 0.5)) <= avail * 0.01


def test_resolve_budget_default_fraction(monkeypatch):
    monkeypatch.delenv("SCARF_MEM_BUDGET", raising=False)
    monkeypatch.delenv("SCARF_WORKERS", raising=False)
    budget = resolve_budget()
    assert budget.memoryBytes > 0
    assert budget.workers >= 1


def test_detect_memory_fallback_when_meminfo_absent(monkeypatch):
    real_open = builtins.open

    def fake_open(path, *args, **kwargs):
        if str(path) == "/proc/meminfo":
            raise OSError("no meminfo")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", fake_open)
    monkeypatch.setattr("os.sysconf", lambda name: -1)
    assert detect_available_memory_bytes() == 8 * 1024 * 1024 * 1024


def test_tile_rows_alignment_and_caps():
    budget = ResourceBudget(memoryBytes=1 * 1024**3, workers=1)
    rows = tile_rows_for_width(
        n_cols=50_000, itemsize=8, budget=budget, chunk_rows=256, n_rows=100_000
    )
    assert rows % 256 == 0
    assert rows >= 256
    assert rows <= 100_000

    tiny = ResourceBudget(memoryBytes=1024, workers=1)
    rows_tiny = tile_rows_for_width(
        n_cols=50_000, itemsize=8, budget=tiny, chunk_rows=256, n_rows=100_000
    )
    assert rows_tiny == 256

    rows_capped = tile_rows_for_width(
        n_cols=10, itemsize=8, budget=budget, chunk_rows=1, n_rows=50
    )
    assert rows_capped == 50


def test_tile_rows_shrinks_with_more_workers():
    wide = ResourceBudget(memoryBytes=8 * 1024**3, workers=1)
    narrow = ResourceBudget(memoryBytes=8 * 1024**3, workers=8)
    rows_wide = tile_rows_for_width(50_000, 8, wide, chunk_rows=1)
    rows_narrow = tile_rows_for_width(50_000, 8, narrow, chunk_rows=1)
    assert rows_narrow < rows_wide


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
        if name == "SC_AVPHYS_PAGES":
            return 1024
        return -1

    monkeypatch.setattr(builtins, "open", fake_open)
    monkeypatch.setattr("os.sysconf", fake_sysconf)
    assert detect_available_memory_bytes() == 4096 * 1024


def test_invalid_workers_env_rejected(monkeypatch):
    monkeypatch.delenv("SCARF_MEM_BUDGET", raising=False)
    monkeypatch.setenv("SCARF_WORKERS", "not-a-number")
    with pytest.raises(ValueError):
        resolve_budget(memory="8G")


def test_fraction_uses_available_memory(monkeypatch):
    monkeypatch.delenv("SCARF_MEM_BUDGET", raising=False)
    avail = detect_available_memory_bytes()
    got = resolve_budget(memory="0.25", workers=1).memoryBytes
    assert abs(got - int(avail * 0.25)) <= avail * 0.01


def test_bounded_prefetch_caps_by_workers_and_fit():
    budget = ResourceBudget(memoryBytes=4 * 1024**3, workers=4)
    # Tiny band: capped by workers, not by fit.
    assert bounded_prefetch(1024, budget) == 4
    # Band equal to the whole per-worker slice: only one in flight.
    assert bounded_prefetch(budget.perWorkerBytes, budget) == 1
    # Requested below the worker ceiling is honored.
    assert bounded_prefetch(1024, budget, requested=2) == 2
    # Band larger than per-worker slice still allows at least one.
    assert bounded_prefetch(budget.perWorkerBytes * 10, budget) == 1
