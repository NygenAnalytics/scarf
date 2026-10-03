"""Memory reservations and thread schedules of the rank marker search.

These tests run ``find_markers_by_rank`` over stores whose counts sit in small
read groups, so each search streams several read groups and plans its kernel
threads within the assay's memory budget. Each store is written once per
module and opened afresh by every test that searches it.
"""

import textwrap
from pathlib import Path

import numpy as np
import pytest
import zarr

from scarf.datastore.datastore import DataStore
from scarf.features.markers import find_markers_by_rank

# The memory-fit store, and the store whose thread schedules are recorded.
_FIT_SHAPE = (20_000, 256)
_SCHEDULE_SHAPE = (20_000, 64)


def _rebuild_interned_strings() -> None:
    """Make CPython rebuild its interned-string table before a trace starts.

    Pathlib interns every path part it parses, and those strings soon die;
    their slots fill the table until an insertion rebuilds it. A rebuild
    inside a trace adds a block the size of the table, because the table it
    replaces predates the trace. A rebuilt table holds far more insertions
    than a marker search and its writes make.
    """
    import sys
    import tracemalloc

    chunk = 1 << 12
    for start in range(0, 1 << 21, chunk):
        # Strings made before the trace leave a rebuilt table as the only
        # sizeable block that the trace sees.
        names = [
            f"scarf-test-interned-{index}" for index in range(start, start + chunk)
        ]
        tracemalloc.start()
        try:
            for name in names:
                sys.intern(name)
            current, _ = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        if current > 1 << 16:
            return


def _write_layout_store(directory: Path, values: np.ndarray) -> Path:
    """Write uint16 RNA counts in small read groups and return the store path."""
    from scarf.storage.count_matrix import CountMatrixPolicy
    from scarf.storage.schema import create_cell_data, create_zarr_count_assay
    from scarf.writers.counts_t import finalize_writer_counts_t
    from tests.storage_helpers import finalize_test_counts

    path = Path(directory) / "layout.zarr"
    root = zarr.open_group(str(path), mode="w")
    n_cells, n_features = values.shape
    cell_ids = np.array([f"c{index}" for index in range(n_cells)])
    feature_ids = np.array([f"f{index}" for index in range(n_features)])
    create_cell_data(root, None, ids=cell_ids, names=cell_ids)
    # Read groups of 32 features over sharded bands of 2,500 cells.
    counts = create_zarr_count_assay(
        root,
        "RNA",
        None,
        n_cells,
        feature_ids,
        feature_ids,
        dtype="uint16",
        policy=CountMatrixPolicy(unitBytes=n_cells * 32 * 2, chunkBytes=32 * 2_500 * 2),
    )
    counts[:] = values
    finalize_test_counts(counts)
    finalize_writer_counts_t(root, "RNA", None)
    return path


def _open_layout_store(path: Path, **datastore_options) -> DataStore:
    return DataStore(
        str(path),
        default_assay="RNA",
        min_features_per_cell=0,
        mito_pattern="",
        ribo_pattern="",
        **datastore_options,
    )


def _layout_store(
    directory: Path, values: np.ndarray, **datastore_options
) -> DataStore:
    """Write uint16 RNA counts in small read groups and open them."""
    return _open_layout_store(
        _write_layout_store(directory, values), **datastore_options
    )


def _shared_layout_store(directory: Path, values: np.ndarray) -> Path:
    # The first open adds the cell totals, so later opens only read the store.
    path = _write_layout_store(directory, values)
    _open_layout_store(path)
    return path


@pytest.fixture(scope="module")
def fit_store(tmp_path_factory) -> Path:
    values = np.random.default_rng(0).poisson(0.15, size=_FIT_SHAPE)
    return _shared_layout_store(
        tmp_path_factory.mktemp("fit"), values.astype(np.uint16)
    )


@pytest.fixture(scope="module")
def schedule_store(tmp_path_factory) -> Path:
    values = np.random.default_rng(3).poisson(0.3, size=_SCHEDULE_SHAPE)
    return _shared_layout_store(
        tmp_path_factory.mktemp("schedule"), values.astype(np.uint16)
    )


@pytest.mark.parametrize("method_name", ["norm_lib_size", "norm_dummy"])
@pytest.mark.parametrize("n_groups", [20, 400])
def test_marker_search_and_write_fit_the_bytes_the_search_reserves(
    fit_store, tmp_path, method_name, n_groups
) -> None:
    import tracemalloc

    import scarf.assay.normalization as normalization
    from scarf.storage.execution import execution_report_scope

    store = _open_layout_store(fit_store, nthreads=4, mem_budget="24M")
    store.RNA.normMethod = getattr(normalization, method_name)
    n_cells, n_features = _FIT_SHAPE
    labels = np.random.default_rng(n_groups).integers(0, n_groups, size=n_cells)
    cells = np.arange(n_cells)
    features = np.arange(n_features)
    names = np.asarray(store.RNA.feats.fetch_all("names"))
    # Load the kernels, codecs, and pools, and import what finishes a table,
    # so that the trace holds only what the search and its writes allocate.
    # Two groups load them as well as many groups do, at a fraction of the
    # writes.
    DataStore._write_marker_slot(
        zarr.open_group(str(tmp_path / "warm.zarr"), mode="w"),
        find_markers_by_rank(store.RNA, cells % 2, cells, features[:4], writers=2),
        workers=2,
        feature_names=names,
        feature_ids=names,
    )
    slot = zarr.open_group(str(tmp_path / "markers.zarr"), mode="w")
    _rebuild_interned_strings()

    with execution_report_scope() as reports:
        tracemalloc.start()
        try:
            base, _ = tracemalloc.get_traced_memory()
            result = find_markers_by_rank(store.RNA, labels, cells, features, writers=2)
            DataStore._write_marker_slot(
                slot, result, workers=2, feature_names=names, feature_ids=names
            )
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

    (plan,) = [
        report.as_metrics()
        for report in reports
        if report.unitKind == "countsTReadGroup"
    ]
    # NumPy, Zarr, and Numba buffers are traced. The search and the writes
    # that follow it hold no more than the search reserved.
    assert peak - base <= int(plan["reservedBytes"]) <= 24 * 1024**2
    assert len(result.group_ids) == n_groups
    assert sorted(slot.group_keys()) == sorted(str(group) for group in result.group_ids)


def test_marker_search_fits_its_threads_to_the_memory_budget(
    schedule_store, monkeypatch
) -> None:
    import re

    from scarf.features.markers import search
    from scarf.features.markers.rank import gene_major_rank_scratch_bytes
    from scarf.storage.budget import ResourceBudget

    store = _open_layout_store(schedule_store, nthreads=8)
    n_cells, n_features = _SCHEDULE_SHAPE
    labels = np.random.default_rng(3).integers(0, 6, size=n_cells)
    cells = np.arange(n_cells)
    features = np.arange(n_features)
    roomy = store.RNA.resources
    expected = find_markers_by_rank(store.RNA, labels, cells, features).statistics
    schedules: list[tuple[int, int]] = []
    schedule = search._gene_major_schedule

    def recorded(*args, **kwargs):
        schedules.append(schedule(*args, **kwargs))
        return schedules[-1]

    monkeypatch.setattr(search, "_gene_major_schedule", recorded)

    def rank(memory_bytes: int, nthreads: int = 8) -> np.ndarray:
        store.RNA.resources = ResourceBudget(memory_bytes, roomy.workers)
        return find_markers_by_rank(
            store.RNA, labels, cells, features, nthreads=nthreads
        ).statistics

    with pytest.raises(MemoryError, match="needs at least") as refused:
        rank(1)
    (minimum,) = re.findall(r"needs at least (\d+) bytes", str(refused.value))
    minimum = int(minimum)
    with pytest.raises(MemoryError, match="needs at least"):
        rank(minimum - 1)
    # One thread fits exactly at the minimum, and a few more just above it.
    np.testing.assert_array_equal(rank(minimum), expected)
    assert schedules[-1] == (1, 1)
    per_thread = gene_major_rank_scratch_bytes(
        n_cells=len(cells), n_groups=6, n_features=0, nthreads=2
    ) - gene_major_rank_scratch_bytes(
        n_cells=len(cells), n_groups=6, n_features=0, nthreads=1
    )
    np.testing.assert_array_equal(rank(minimum + 2 * per_thread), expected)
    threads, calls = schedules[-1]
    assert 1 < threads * calls < roomy.workers
    # nthreads caps the threads that a roomy budget would allow.
    np.testing.assert_array_equal(rank(roomy.memoryBytes, nthreads=2), expected)
    threads, calls = schedules[-1]
    assert threads * calls == 2


def test_marker_search_ranks_narrow_read_groups_at_once(
    schedule_store, monkeypatch
) -> None:
    from scarf.features.markers import search
    from scarf.storage.feature_stream import persisted_read_group, read_group_rows

    store = _open_layout_store(schedule_store, nthreads=8)
    n_cells, n_features = _SCHEDULE_SHAPE
    labels = np.random.default_rng(4).integers(0, 6, size=n_cells)
    cells = np.arange(n_cells)
    counts_t = store.RNA.rawDataT
    # One feature of each read group leaves one row per kernel call, so whole
    # groups run at once, one serial kernel each.
    features = np.arange(0, n_features, persisted_read_group(counts_t)[0])
    groups = len(read_group_rows(counts_t, features))
    assert groups > 1
    schedules: list[tuple[int, int]] = []
    schedule = search._gene_major_schedule

    def recorded(*args, **kwargs):
        schedules.append(schedule(*args, **kwargs))
        return schedules[-1]

    monkeypatch.setattr(search, "_gene_major_schedule", recorded)
    expected = find_markers_by_rank(store.RNA, labels, cells, features).statistics
    observed = find_markers_by_rank(
        store.RNA, labels, cells, features, nthreads=8
    ).statistics

    assert schedules == [(1, 1), (1, groups)]
    np.testing.assert_array_equal(observed, expected)


_SCRATCH_CHILD = textwrap.dedent(
    """
    import json
    import sys
    import threading
    from pathlib import Path

    import numba
    import numpy as np

    from scarf.features.markers import search
    from scarf.features.markers.rank import gene_major_rank_scratch_bytes
    from scarf.storage import feature_stream
    from tests.test_markers_search import _layout_store

    directory = Path(sys.argv[1])
    values = np.load(directory / "values.npy")
    n_cells, n_features = values.shape
    store = _layout_store(directory, values, nthreads=8)
    lock = threading.Lock()
    active = 0
    calls = []
    charged = []
    original_kernel = search._marker_stats_gene_major
    original_map = feature_stream.map_feature_read_groups

    def kernel(*args):
        global active
        with lock:
            active += 1
            calls.append((args[8], numba.get_num_threads(), active))
        try:
            return original_kernel(*args)
        finally:
            with lock:
                active -= 1

    def stream(*args, **kwargs):
        charged.append(kwargs["scratchBytes"])
        return original_map(*args, **kwargs)

    original_schedule = search._gene_major_schedule
    schedules = []

    def schedule(*args, **kwargs):
        planned = original_schedule(*args, **kwargs)
        schedules.append(planned)
        return planned

    search._marker_stats_gene_major = kernel
    search._gene_major_schedule = schedule
    feature_stream.map_feature_read_groups = stream
    counts_t = store.RNA.rawDataT
    group_width = feature_stream.persisted_read_group(counts_t)[0]
    # The features of one read group, then every feature.
    for features in (np.arange(group_width), np.arange(n_features)):
        calls.clear()
        charged.clear()
        schedules.clear()
        result = search.find_markers_by_rank(
            store.RNA,
            np.arange(n_cells) % 5,
            np.arange(n_cells),
            features,
            nthreads=8,
        )
        ((slots, planned_calls),) = schedules
        print("SCRATCH:" + json.dumps({
            "features": len(features),
            "readGroups": len(feature_stream.read_group_rows(counts_t, features)),
            "workers": store.RNA.resources.workers,
            "slots": slots,
            "plannedCalls": planned_calls,
            "threads": sorted({call[0] for call in calls}),
            "numbaThreads": sorted({call[1] for call in calls}),
            "concurrentCalls": max(call[2] for call in calls),
            "charged": charged[0] - result.statistics.nbytes,
            "kernelScratch": planned_calls * gene_major_rank_scratch_bytes(
                n_cells=n_cells,
                n_groups=5,
                n_features=group_width,
                nthreads=slots,
            ),
        }))
    """
)


def test_marker_kernel_scratch_covers_the_kernel_threads_that_run(tmp_path) -> None:
    import json
    import os
    import subprocess
    import sys

    values = np.random.default_rng(1).poisson(0.3, size=(12_000, 128))
    np.save(tmp_path / "values.npy", values.astype(np.uint16))
    # Numba fixes its thread count when it loads, so a child process runs
    # the search with two Numba threads and eight Scarf workers.
    env = {**os.environ, "NUMBA_NUM_THREADS": "2", "SCARF_WORKERS": "8"}

    completed = subprocess.run(
        [sys.executable, "-c", _SCRATCH_CHILD, str(tmp_path)],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    observed = {
        entry["features"]: entry
        for entry in (
            json.loads(line.removeprefix("SCRATCH:"))
            for line in completed.stdout.splitlines()
            if line.startswith("SCRATCH:")
        )
    }
    one_group, every_group = observed[min(observed)], observed[max(observed)]
    # Eight workers split one read group over the two Numba threads of one
    # kernel call at a time.
    assert one_group["workers"] == 8
    assert one_group["readGroups"] == 1
    assert (one_group["slots"], one_group["plannedCalls"]) == (2, 1)
    assert one_group["threads"] == one_group["numbaThreads"] == [2]
    assert one_group["concurrentCalls"] == 1
    # Narrow read groups run at once with one slot each, on no more compute
    # workers than were planned for them.
    groups = every_group["readGroups"]
    assert 1 < groups <= 8
    assert (every_group["slots"], every_group["plannedCalls"]) == (1, groups)
    assert every_group["threads"] == [1]
    assert every_group["concurrentCalls"] <= groups
    # The search charges the scratch of every kernel call that can run.
    for entry in (one_group, every_group):
        assert entry["charged"] >= entry["kernelScratch"]
