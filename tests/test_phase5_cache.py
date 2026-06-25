import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

from scarf.datastore.graph_datastore import GraphDataStore
from scarf.knn_utils import _patch_null_weights, smoothen_dists
from scarf.storage.zarr_store import (
    copy_zarr_array,
    create_numeric_array,
    is_remote_datastore,
    normed_array_spec,
    open_or_create_staged_normed_array,
)
from scarf.writers import create_zarr_dataset


def _memory_group():
    return zarr.open_group(store=MemoryStore(), mode="w")


def test_copy_zarr_array_round_trip(tmp_path):
    src_root = zarr.open_group(str(tmp_path / "src.zarr"), mode="w")
    spec = normed_array_spec(64, 8, profile="fast_local")
    src = create_numeric_array(src_root, "data", spec)
    expected = np.random.rand(64, 8).astype(np.float32)
    src[:] = expected

    dst_root = zarr.open_group(str(tmp_path / "dst.zarr"), mode="w")
    dst = create_numeric_array(dst_root, "data", spec)
    copy_zarr_array(src, dst, block_rows=16)
    np.testing.assert_allclose(dst[:], expected, rtol=1e-6)


def test_open_or_create_staged_normed_array_reuses_shape(tmp_path):
    src_root = zarr.open_group(str(tmp_path / "src.zarr"), mode="w")
    spec = normed_array_spec(32, 4, profile="cloud")
    src = create_numeric_array(src_root, "data", spec)
    src[:] = np.ones((32, 4), dtype=np.float32)

    cache_path = str(tmp_path / "cache" / "abc123" / "normed.zarr")
    staged = open_or_create_staged_normed_array(cache_path, src)
    copy_zarr_array(src, staged, block_rows=8)
    staged.attrs["staged_subset_hash"] = "abc123"
    staged.attrs["staged_complete"] = True

    reopened = open_or_create_staged_normed_array(cache_path, src)
    assert reopened.shape == src.shape
    np.testing.assert_allclose(reopened[:], np.ones((32, 4), dtype=np.float32))


def test_is_remote_datastore():
    local_root = _memory_group()
    assert is_remote_datastore("/tmp/foo.zarr", local_root) is False
    assert is_remote_datastore("s3://bucket/path", local_root) is True


def test_resolve_local_cache_plan(tmp_path):
    local_root = _memory_group()
    enabled, base, remove = GraphDataStore._resolve_local_cache_plan(
        "/tmp/local.zarr", local_root, "auto"
    )
    assert enabled is False

    enabled, base, remove = GraphDataStore._resolve_local_cache_plan(
        "s3://bucket/path", local_root, False
    )
    assert enabled is False

    enabled, base, remove = GraphDataStore._resolve_local_cache_plan(
        "s3://bucket/path", local_root, str(tmp_path / "cache")
    )
    assert enabled is True
    assert base == str(tmp_path / "cache")
    assert remove is False


def test_stage_normed_data_skips_repeat_copy(toy_crdir_ds, tmp_path, monkeypatch):
    rna = toy_crdir_ds.RNA
    cell_idx = np.arange(rna.cells.N)
    feat_idx = np.array([0, 1, 3])
    loc = "stage_src"
    rna.z.create_group(loc, overwrite=True)
    from scarf.writers import write_renorm_subset_to_zarr

    write_renorm_subset_to_zarr(
        rna, cell_idx, feat_idx, rna.z, f"{loc}/data", rna.nthreads
    )
    remote = rna.z[f"{loc}/data"]
    subset_params = {"log_transform": False, "renormalize_subset": True}
    calls = {"n": 0}
    orig_copy = copy_zarr_array

    def counting_copy(*args, **kwargs):
        calls["n"] += 1
        return orig_copy(*args, **kwargs)

    monkeypatch.setattr(
        "scarf.datastore.graph_datastore.copy_zarr_array", counting_copy
    )
    store = GraphDataStore.__new__(GraphDataStore)
    store.nthreads = rna.nthreads
    cache_base = str(tmp_path / "scratch")
    store._stage_normed_data(remote, "hash1", subset_params, cache_base)
    store._stage_normed_data(remote, "hash1", subset_params, cache_base)
    assert calls["n"] == 1


def test_stage_normed_data_recopies_when_subset_params_change(
    toy_crdir_ds, tmp_path, monkeypatch
):
    rna = toy_crdir_ds.RNA
    cell_idx = np.arange(rna.cells.N)
    feat_idx = np.array([0, 1, 3])
    loc = "stage_src_params"
    rna.z.create_group(loc, overwrite=True)
    from scarf.writers import write_renorm_subset_to_zarr

    write_renorm_subset_to_zarr(
        rna, cell_idx, feat_idx, rna.z, f"{loc}/data", rna.nthreads
    )
    remote = rna.z[f"{loc}/data"]
    calls = {"n": 0}
    orig_copy = copy_zarr_array

    def counting_copy(*args, **kwargs):
        calls["n"] += 1
        return orig_copy(*args, **kwargs)

    monkeypatch.setattr(
        "scarf.datastore.graph_datastore.copy_zarr_array", counting_copy
    )
    store = GraphDataStore.__new__(GraphDataStore)
    store.nthreads = rna.nthreads
    cache_base = str(tmp_path / "scratch_params")
    store._stage_normed_data(
        remote, "hash1", {"log_transform": False, "renormalize_subset": True}, cache_base
    )
    store._stage_normed_data(
        remote, "hash1", {"log_transform": True, "renormalize_subset": True}, cache_base
    )
    assert calls["n"] == 2


def test_patch_null_weights_matches_full_rewrite(tmp_path):
    weights = np.array([0.0, 0.2, 0.0, 0.5, 0.0, 0.3], dtype=np.float64)
    null_positions = np.flatnonzero(weights == 0).tolist()
    fill = 0.15

    expected = weights.copy()
    expected[null_positions] = fill

    root = zarr.open_group(str(tmp_path / "weights.zarr"), mode="w")
    zgw = create_zarr_dataset(root, "weights", (2,), "f8", weights.shape)
    zgw[:] = weights
    _patch_null_weights(zgw, null_positions, fill, patch_chunk=2)
    np.testing.assert_allclose(zgw[:], expected)


def test_smoothen_dists_runs(tmp_path):
    pytest.importorskip("umap")
    n_cells, n_neighbors = 24, 5
    chunk_size = 8
    rng = np.random.default_rng(0)
    dist = rng.random((n_cells, n_neighbors)).astype(np.float64)
    dist[:, 0] = 0.0
    idx = np.tile(np.arange(n_cells), (n_cells, 1)) % n_cells

    root = zarr.open_group(str(tmp_path / "graph.zarr"), mode="w")
    knn = root.create_group("knn")
    z_idx = create_zarr_dataset(knn, "indices", (chunk_size,), "u8", idx.shape)
    z_dist = create_zarr_dataset(knn, "distances", (chunk_size,), "f8", dist.shape)
    z_idx[:] = idx
    z_dist[:] = dist
    graph = root.create_group("graph")
    smoothen_dists(graph, z_idx, z_dist, lc=1.0, bw=1.5, chunk_size=chunk_size)
    assert graph["weights"].shape[0] == graph["edges"].shape[0]
    assert graph["weights"].shape[0] > 0
