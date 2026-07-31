import numpy as np

from scarf.assay import RNAassay
from scarf.writers import write_renorm_subset_to_zarr


def _subset_indices(rna):
    cell_idx = np.arange(rna.cells.N)
    feat_idx = np.array([0, 1, 3])
    return cell_idx, feat_idx


def _reference_renorm_subset(rna, cell_idx, feat_idx, log_transform=False):
    raw = rna.rawData[cell_idx, :][:, feat_idx].compute()
    row_sum = raw.sum(axis=1)
    row_sum[row_sum == 0] = 1
    out = rna.sf * raw / row_sum[:, np.newaxis]
    if log_transform:
        out = np.log1p(out)
    return out.astype(np.float32)


def test_write_renorm_subset_matches_reference(toy_crdir_ds):
    rna = toy_crdir_ds.RNA
    cell_idx, feat_idx = _subset_indices(rna)
    loc = "fused_normed"

    rna.z.create_group(loc, overwrite=True)
    write_renorm_subset_to_zarr(
        rna, cell_idx, feat_idx, rna.z, f"{loc}/data", rna.nthreads
    )
    expected = _reference_renorm_subset(rna, cell_idx, feat_idx)
    written = rna.z[f"{loc}/data"][:]
    np.testing.assert_allclose(written, expected, rtol=1e-5)


def test_write_renorm_subset_log_transform(toy_crdir_ds):
    rna = toy_crdir_ds.RNA
    cell_idx, feat_idx = _subset_indices(rna)
    loc = "fused_normed_log"

    rna.z.create_group(loc, overwrite=True)
    write_renorm_subset_to_zarr(
        rna,
        cell_idx,
        feat_idx,
        rna.z,
        f"{loc}/data",
        rna.nthreads,
        log_transform=True,
    )
    expected = _reference_renorm_subset(rna, cell_idx, feat_idx, log_transform=True)
    written = rna.z[f"{loc}/data"][:]
    np.testing.assert_allclose(written, expected, rtol=1e-5)


def test_save_normalized_data_renorm_uses_fused_path(toy_crdir_ds, monkeypatch):
    rna = toy_crdir_ds.RNA
    called = {"normed": 0}
    orig_normed = RNAassay.normed

    def fake_normed(self, *args, **kwargs):
        called["normed"] += 1
        return orig_normed(self, *args, **kwargs)

    monkeypatch.setattr(RNAassay, "normed", fake_normed)
    monkeypatch.setattr(
        rna,
        "_get_cell_feat_idx",
        lambda cell_key, feat_key: _subset_indices(rna),
    )

    rna.save_normalized_data(
        cell_key="I",
        feat_key="I",
        location="normed_fused_test",
        log_transform=False,
        renormalize_subset=True,
        update_keys=False,
    )
    assert called["normed"] == 0
    cell_idx, feat_idx = _subset_indices(rna)
    expected = _reference_renorm_subset(rna, cell_idx, feat_idx)
    np.testing.assert_allclose(rna.z["normed_fused_test/data"][:], expected, rtol=1e-5)


def test_save_normalized_data_without_renorm_still_uses_normed(
    toy_crdir_ds, monkeypatch
):
    rna = toy_crdir_ds.RNA
    called = {"normed": 0, "fused": 0}
    orig_normed = RNAassay.normed

    def fake_normed(self, *args, **kwargs):
        called["normed"] += 1
        return orig_normed(self, *args, **kwargs)

    def fake_fused(*args, **kwargs):
        called["fused"] += 1
        return write_renorm_subset_to_zarr(*args, **kwargs)

    monkeypatch.setattr(RNAassay, "normed", fake_normed)
    monkeypatch.setattr(
        "scarf.storage.materialize.write_renorm_subset_to_zarr",
        fake_fused,
    )
    monkeypatch.setattr(
        rna,
        "_get_cell_feat_idx",
        lambda cell_key, feat_key: _subset_indices(rna),
    )

    rna.save_normalized_data(
        cell_key="I",
        feat_key="I",
        location="normed_standard_test",
        log_transform=False,
        renormalize_subset=False,
        update_keys=False,
    )
    assert called["normed"] == 1
    assert called["fused"] == 0
    assert (
        "normalization_identity"
        not in rna.z["normed_standard_test"].attrs["subset_params"]
    )


def test_save_normalized_data_renorm_cache_hit(toy_crdir_ds, monkeypatch):
    from scarf.storage.materialize import (
        write_renorm_subset_to_zarr as materialize_renorm_subset,
    )

    rna = toy_crdir_ds.RNA
    called = {"fused": 0}
    orig_fused = materialize_renorm_subset

    def counting_fused(*args, **kwargs):
        called["fused"] += 1
        return orig_fused(*args, **kwargs)

    monkeypatch.setattr(
        "scarf.storage.materialize.write_renorm_subset_to_zarr",
        counting_fused,
    )
    monkeypatch.setattr(
        rna,
        "_get_cell_feat_idx",
        lambda cell_key, feat_key: _subset_indices(rna),
    )

    kwargs = dict(
        cell_key="I",
        feat_key="I",
        location="normed_cache_test",
        log_transform=False,
        renormalize_subset=True,
        update_keys=False,
    )
    rna.save_normalized_data(**kwargs)
    rna.save_normalized_data(**kwargs)
    assert called["fused"] == 1


def test_atac_legacy_cache_rejects_missing_normalizer_identity(
    atac_datastore,
    monkeypatch,
):
    assay = atac_datastore.ATAC
    cell_idx = np.arange(8, dtype=np.int64)
    feat_idx = assay.feats.active_index("I")[:4]
    monkeypatch.setattr(
        assay,
        "_get_cell_feat_idx",
        lambda cell_key, feat_key: (cell_idx, feat_idx),
    )

    original = assay.normed
    calls = 0

    def counting_normed(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(assay, "normed", counting_normed)
    kwargs = dict(
        cell_key="I",
        feat_key="I",
        location="atac_normalizer_identity_cache_test",
        log_transform=False,
        renormalize_subset=False,
        update_keys=False,
    )

    assay.save_normalized_data(**kwargs)
    group = assay.z["atac_normalizer_identity_cache_test"]
    current_params = dict(group.attrs["subset_params"])
    assert current_params["normalization_identity"] == getattr(
        assay.normMethod,
        "artifact_identity",
    )

    legacy_params = dict(current_params)
    del legacy_params["normalization_identity"]
    group.attrs["subset_params"] = legacy_params
    assay.save_normalized_data(**kwargs)
    assay.save_normalized_data(**kwargs)

    assert calls == 2
