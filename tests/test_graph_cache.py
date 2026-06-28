import numpy as np
import zarr
from zarr.storage import MemoryStore

from scarf.datastore.graph_datastore import GraphDataStore
from scarf.storage.zarr_store import copy_zarr_array


def _memory_group():
    return zarr.open_group(store=MemoryStore(), mode="w")


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
        remote,
        "hash1",
        {"log_transform": False, "renormalize_subset": True},
        cache_base,
    )
    store._stage_normed_data(
        remote, "hash1", {"log_transform": True, "renormalize_subset": True}, cache_base
    )
    assert calls["n"] == 2
