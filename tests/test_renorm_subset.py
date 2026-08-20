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


def test_feature_major_normalization_rejects_missing_or_unsorted_cells() -> None:
    from types import SimpleNamespace

    import pytest
    import zarr
    from zarr.storage import MemoryStore

    from scarf.storage.budget import ResourceBudget
    from scarf.storage.feature_stream import FeatureCellBand
    from scarf.storage.materialize import (
        _counts_t_renormalized_batches,
        write_renorm_subset_to_zarr,
    )
    from tests.test_feature_stream import _counts_t_with_plan

    values = np.arange(6 * 4, dtype=np.uint16).reshape(6, 4)
    counts_t = _counts_t_with_plan(values)
    assay = SimpleNamespace(rawDataT=None)
    with pytest.raises(ValueError, match="requires countsT"):
        list(
            _counts_t_renormalized_batches(
                assay,
                np.arange(6),
                np.arange(4),
                scaleFactor=1.0,
                logTransform=False,
            )
        )
    assay = SimpleNamespace(rawDataT=counts_t)
    with pytest.raises(ValueError, match="sorted unique cells"):
        list(
            _counts_t_renormalized_batches(
                assay,
                np.array([2, 0, 1]),
                np.arange(4),
                scaleFactor=1.0,
                logTransform=False,
            )
        )

    import scarf.storage.feature_stream as feature_stream_module

    ready = SimpleNamespace(
        rawDataT=counts_t,
        resources=ResourceBudget(8 * 1024 * 1024, 1),
        storageIo=None,
    )
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        feature_stream_module, "map_feature_cell_bands", lambda *_a, **_k: iter(())
    )
    try:
        with pytest.raises(RuntimeError, match="did not cover every selected cell"):
            list(
                _counts_t_renormalized_batches(
                    ready,
                    np.arange(6),
                    np.arange(4),
                    scaleFactor=1.0,
                    logTransform=False,
                )
            )
    finally:
        monkeypatch.undo()

    def empty_feature_group(_counts_t, process, **_kwargs):
        yield process(
            FeatureCellBand(
                featStart=0,
                featEnd=1,
                cellStart=0,
                cellEnd=2,
                values=np.zeros((1, 2), dtype=np.uint16),
                selectedLocal=np.array([0, 1], dtype=np.int64),
                selectedDestinations=np.array([0, 1], dtype=np.int64),
                readSec=0.0,
                blockBytes=1,
            )
        )

    monkeypatch.setattr(
        feature_stream_module, "map_feature_cell_bands", empty_feature_group
    )
    try:
        with pytest.raises(RuntimeError, match="did not contain selected features"):
            list(
                _counts_t_renormalized_batches(
                    SimpleNamespace(
                        rawDataT=counts_t,
                        resources=ResourceBudget(8 * 1024 * 1024, 1),
                        storageIo=None,
                    ),
                    np.arange(6),
                    np.array([2, 3], dtype=np.int64),
                    scaleFactor=1.0,
                    logTransform=False,
                )
            )
    finally:
        monkeypatch.undo()

    def split_cells(_counts_t, process, **_kwargs):
        yield process(
            FeatureCellBand(
                featStart=0,
                featEnd=4,
                cellStart=0,
                cellEnd=2,
                values=np.zeros((4, 2), dtype=np.uint16),
                selectedLocal=np.array([0, 1], dtype=np.int64),
                selectedDestinations=np.array([0, 2], dtype=np.int64),
                readSec=0.0,
                blockBytes=1,
            )
        )

    monkeypatch.setattr(feature_stream_module, "map_feature_cell_bands", split_cells)
    try:
        with pytest.raises(ValueError, match="contiguous selected-cell bands"):
            list(
                _counts_t_renormalized_batches(
                    ready,
                    np.arange(6),
                    np.arange(4),
                    scaleFactor=1.0,
                    logTransform=False,
                )
            )
    finally:
        monkeypatch.undo()

    root = zarr.open_group(store=MemoryStore(), mode="w")
    raw = np.arange(12, dtype=np.float32).reshape(3, 4)
    missing_sf = SimpleNamespace(rawData=raw, rawDataT=None, sf=None)
    with pytest.raises(ValueError, match="size factor"):
        write_renorm_subset_to_zarr(
            missing_sf,
            np.arange(3),
            np.arange(4),
            root,
            "missing_sf",
            1,
        )
    write_renorm_subset_to_zarr(
        SimpleNamespace(
            rawData=raw,
            rawDataT=None,
            sf=1000.0,
            resources=ResourceBudget(8 * 1024 * 1024, 1),
            storageIo=None,
        ),
        np.arange(3),
        np.arange(4),
        root,
        "from_counts",
        1,
    )
    assert "from_counts" in root
