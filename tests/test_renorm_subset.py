import numpy as np

from scarf.assay import RNAassay
from scarf.storage.artifacts import artifact_group
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


def _normalization_inputs(store, assay_name, feat_idx):
    assay = getattr(store, assay_name)
    mask = np.zeros(assay.feats.N, dtype=bool)
    mask[feat_idx] = True
    return (
        store.snapshot_cell_selection(),
        store.set_feature_selection(from_assay=assay_name, mask=mask),
    )


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


def test_run_normalization_renorm_uses_fused_path(toy_crdir_ds, monkeypatch):
    rna = toy_crdir_ds.RNA
    called = {"normed": 0}
    orig_normed = RNAassay.normed

    def fake_normed(self, *args, **kwargs):
        called["normed"] += 1
        return orig_normed(self, *args, **kwargs)

    monkeypatch.setattr(RNAassay, "normed", fake_normed)
    cell_idx, feat_idx = _subset_indices(rna)
    cells, features = _normalization_inputs(toy_crdir_ds, "RNA", feat_idx)
    normalized = toy_crdir_ds.run_normalization(
        cells,
        features,
        log_transform=False,
        renormalize_subset=True,
        invalidate_cache=True,
    )
    assert called["normed"] == 0
    expected = _reference_renorm_subset(rna, cell_idx, feat_idx)
    np.testing.assert_allclose(
        artifact_group(toy_crdir_ds.zw, normalized)["data"][:],
        expected,
        rtol=1e-5,
    )


def test_run_normalization_without_renorm_still_uses_normed(toy_crdir_ds, monkeypatch):
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
    _, feat_idx = _subset_indices(rna)
    cells, features = _normalization_inputs(toy_crdir_ds, "RNA", feat_idx)
    normalized = toy_crdir_ds.run_normalization(
        cells,
        features,
        log_transform=False,
        renormalize_subset=False,
        invalidate_cache=True,
    )
    assert called["normed"] == 1
    assert called["fused"] == 0
    assert "subset_params" not in artifact_group(toy_crdir_ds.zw, normalized).attrs


def test_run_normalization_renorm_cache_hit(toy_crdir_ds, monkeypatch):
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
    _, feat_idx = _subset_indices(rna)
    cells, features = _normalization_inputs(toy_crdir_ds, "RNA", feat_idx)
    kwargs = dict(
        cell_selection=cells,
        features=features,
        log_transform=False,
        renormalize_subset=True,
    )
    created = toy_crdir_ds.run_normalization(**kwargs, invalidate_cache=True)
    reused = toy_crdir_ds.run_normalization(**kwargs)
    assert called["fused"] == 1
    assert reused == created


def test_atac_run_normalization_reuses_complete_artifact(
    atac_datastore,
    monkeypatch,
):
    assay = atac_datastore.ATAC
    feat_idx = np.arange(min(4, assay.feats.N), dtype=np.int64)
    original = assay.normed
    calls = 0

    def counting_normed(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(assay, "normed", counting_normed)
    cells, features = _normalization_inputs(atac_datastore, "ATAC", feat_idx)
    kwargs = dict(
        cell_selection=cells,
        features=features,
        log_transform=False,
        renormalize_subset=False,
    )

    created = atac_datastore.run_normalization(**kwargs, invalidate_cache=True)
    group = artifact_group(atac_datastore.zw, created)
    assert "subset_params" not in group.attrs
    reused = atac_datastore.run_normalization(**kwargs)
    assert atac_datastore.run_normalization(**kwargs) == reused

    assert calls == 1
    assert reused == created


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
