import os
from dataclasses import fields

import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

from scarf.datastore._operations.graph import _GraphBuildProgress
from scarf.datastore.graph_datastore import GraphDataStore
from scarf.graph.build import _GraphBuildOutcome
from scarf.storage.copy import copy_zarr_array, create_or_open_staged_normed_array


def _memory_group():
    return zarr.open_group(store=MemoryStore(), mode="w")


def test_graph_progress_tracks_steps_and_propagates_errors() -> None:
    progress = _GraphBuildProgress(2)
    with progress.step("reuse graph", cached=True):
        pass
    with pytest.raises(RuntimeError, match="graph failed"):
        with progress.step("build graph"):
            raise RuntimeError("graph failed")
    progress.finish()

    assert progress._step == 2
    assert [record[1] for record in progress._records] == ["reuse graph", "build graph"]


def test_graph_build_outcome_carries_finalize_state() -> None:
    assert tuple(field.name for field in fields(_GraphBuildOutcome)) == (
        "plan",
        "ann_stream",
        "cell_graph_group_path",
        "fresh_batch_correction",
    )
    assert not hasattr(GraphDataStore, "_set_graph_params")


def test_get_latest_graph_loc_is_public_with_private_compatibility_alias():
    root = _memory_group()
    normed = root.create_group("RNA/normed__I__hvgs")
    reduction = root.create_group(f"{normed.path}/reduction__pca__10__I")
    ann = root.create_group(f"{reduction.path}/ann__l2__50__50__16__1")
    knn = root.create_group(f"{ann.path}/knn__11")
    graph = root.create_group(f"{knn.path}/graph__1.0__1.5")
    normed.attrs["latest_reduction"] = reduction.path
    reduction.attrs["latest_ann"] = ann.path
    ann.attrs["latest_knn"] = knn.path
    knn.attrs["latest_graph"] = graph.path

    store = GraphDataStore.__new__(GraphDataStore)
    store.z = root
    store.workspace = None

    assert store.get_normalized_group_path("RNA", "I", "hvgs") == "RNA/normed__I__hvgs"
    assert store.get_latest_graph_loc("RNA", "I", "hvgs") == graph.path
    assert store._get_latest_graph_loc("RNA", "I", "hvgs") == graph.path

    del knn.attrs["latest_graph"]
    with pytest.raises(KeyError):
        store.get_latest_graph_loc("RNA", "I", "hvgs")


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
        "scarf.datastore._operations.graph.copy_zarr_array", counting_copy
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
        "scarf.datastore._operations.graph.copy_zarr_array", counting_copy
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


def test_mirror_write_lets_staging_skip_copy(toy_crdir_ds, tmp_path, monkeypatch):
    from scarf.assay import Assay
    from scarf.writers import write_renorm_subset_to_zarr

    rna = toy_crdir_ds.RNA
    cell_idx = np.arange(rna.cells.N)
    feat_idx = np.array([0, 1, 3])
    subset_hash = "mirror_hash"
    subset_params = {"log_transform": False, "renormalize_subset": True}

    store = GraphDataStore.__new__(GraphDataStore)
    store.nthreads = rna.nthreads
    cache_base = str(tmp_path / "cache")
    cache_key = store._normed_cache_key(subset_hash, subset_params)
    cache_path = os.path.join(cache_base, cache_key, "normed.zarr")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    mirror = create_or_open_staged_normed_array(
        cache_path, (len(cell_idx), len(feat_idx))
    )

    loc = "mirror_src"
    rna.z.create_group(loc, overwrite=True)
    write_renorm_subset_to_zarr(
        rna, cell_idx, feat_idx, rna.z, f"{loc}/data", rna.nthreads, mirror=mirror
    )
    remote = rna.z[f"{loc}/data"]
    Assay._finalize_staged_mirror(mirror, subset_hash, subset_params)

    assert mirror.attrs["staged_complete"] is True
    assert mirror.attrs["staged_subset_hash"] == subset_hash
    assert np.array_equal(np.asarray(mirror[:]), np.asarray(remote[:]))

    calls = {"n": 0}
    orig_copy = copy_zarr_array

    def counting_copy(*args, **kwargs):
        calls["n"] += 1
        return orig_copy(*args, **kwargs)

    monkeypatch.setattr(
        "scarf.datastore._operations.graph.copy_zarr_array", counting_copy
    )
    staged = store._stage_normed_data(remote, subset_hash, subset_params, cache_base)
    assert calls["n"] == 0
    assert np.array_equal(np.asarray(staged.compute()), np.asarray(remote[:]))
