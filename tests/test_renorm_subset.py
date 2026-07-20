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
        batch_size=256,
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
        batch_size=256,
        location="normed_standard_test",
        log_transform=False,
        renormalize_subset=False,
        update_keys=False,
    )
    assert called["normed"] == 1
    assert called["fused"] == 0


def test_save_normalized_data_renorm_cache_hit(toy_crdir_ds, monkeypatch):
    rna = toy_crdir_ds.RNA
    called = {"fused": 0}
    orig_fused = write_renorm_subset_to_zarr

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
        batch_size=256,
        location="normed_cache_test",
        log_transform=False,
        renormalize_subset=True,
        update_keys=False,
    )
    rna.save_normalized_data(**kwargs)
    rna.save_normalized_data(**kwargs)
    assert called["fused"] == 1
