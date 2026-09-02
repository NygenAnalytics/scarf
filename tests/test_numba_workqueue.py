import os
import subprocess
import sys
import textwrap


_WORKQUEUE_CHILD = textwrap.dedent(
    """
    import numpy as np
    import pandas as pd
    import zarr
    from numba import threading_layer
    from zarr.storage import MemoryStore

    from scarf import DataStore, configure_output
    from scarf.assay import norm_dummy, norm_lib_size
    from scarf.features.markers.search import find_markers_by_rank
    from scarf.storage.budget import ResourceBudget
    from scarf.storage.count_matrix import CountMatrixPolicy
    from scarf.storage.schema import create_cell_data, create_zarr_count_assay
    from scarf.writers.counts_t import finalize_writer_counts_t

    configure_output(progress=False)

    n_cells = 512
    n_features = 1_024
    rng = np.random.default_rng(41)
    values = rng.integers(
        0,
        6,
        size=(n_cells, n_features),
        dtype=np.uint32,
    )
    values[rng.random(values.shape) < 0.7] = 0

    store = MemoryStore()
    root = zarr.open_group(store=store, mode="w")
    cell_ids = np.array([f"c{i}" for i in range(n_cells)])
    feature_ids = np.array([f"f{i}" for i in range(n_features)])
    create_cell_data(
        root,
        None,
        ids=cell_ids,
        names=cell_ids,
        profile="fast_local",
    )
    policy = CountMatrixPolicy(unitBytes=262_144, chunkBytes=65_536)
    counts = create_zarr_count_assay(
        root,
        "RNA",
        None,
        n_cells,
        feature_ids,
        feature_ids,
        profile="fast_local",
        policy=policy,
    )
    counts[:] = values
    finalize_writer_counts_t(
        root,
        "RNA",
        None,
        resources=ResourceBudget(128 * 1024**2, 4),
        profile="fast_local",
        policy=policy,
    )

    def open_store(workers):
        return DataStore(
            store,
            assay_types={"RNA": "RNA"},
            default_assay="RNA",
            min_features_per_cell=0,
            mito_pattern="",
            ribo_pattern="",
            nthreads=workers,
            mem_budget=128 * 1024**2,
            zarrProfile="fast_local",
        )

    single = open_store(1)
    parallel = open_store(4)
    cell_idx = np.arange(n_cells, dtype=np.int64)
    feat_idx = np.arange(n_features, dtype=np.int64)

    expected_stats = single.RNA._streaming_feature_stats(cell_idx, feat_idx)
    actual_stats = parallel.RNA._streaming_feature_stats(cell_idx, feat_idx)
    for name, expected in expected_stats.items():
        np.testing.assert_array_equal(actual_stats[name], expected)

    groups = np.repeat(np.array(["a", "b"]), n_cells // 2)

    def marker_results(datastore, method, workers):
        datastore.RNA.normMethod = method
        return find_markers_by_rank(
            datastore.RNA,
            groups,
            cell_idx,
            feat_idx,
            nthreads=workers,
        )

    for method in (norm_lib_size, norm_dummy):
        expected_markers = marker_results(single, method, 1)
        actual_markers = marker_results(parallel, method, 4)
        for group, expected in expected_markers.items():
            pd.testing.assert_frame_equal(actual_markers[group], expected)

    assert threading_layer() == "workqueue"
    print("WORKQUEUE_OK")
    """
)


def test_numba_workqueue_feature_and_marker_streams_do_not_abort() -> None:
    env = os.environ.copy()
    env.update(
        {
            "NUMBA_NUM_THREADS": "4",
            "NUMBA_THREADING_LAYER": "workqueue",
            "SCARF_WORKERS": "4",
        }
    )
    completed = subprocess.run(
        [sys.executable, "-c", _WORKQUEUE_CHILD],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert completed.returncode == 0, (
        f"workqueue child exited with {completed.returncode}\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )
    assert "WORKQUEUE_OK" in completed.stdout
