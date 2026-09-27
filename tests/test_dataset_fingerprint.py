import pytest
import zarr

from scarf.storage.artifacts import ValueFingerprintBuilder


def test_dataset_fingerprint_is_read_without_recomputing(
    datastore_ephemeral, monkeypatch
):
    fingerprint = datastore_ephemeral._ensure_dataset_fingerprint("RNA")

    def fail_if_recomputed(*_args, **_kwargs):
        raise AssertionError("prepared identity must not be recomputed")

    monkeypatch.setattr(ValueFingerprintBuilder, "update_array", fail_if_recomputed)
    assert datastore_ephemeral._ensure_dataset_fingerprint("RNA") == fingerprint


@pytest.mark.parametrize("mode", ["r", "r+"])
def test_missing_prepared_identity_requires_explicit_rebuilding(
    datastore_ephemeral, mode
):
    store = datastore_ephemeral
    root = zarr.open_group(store=store.zw.store, mode="r+")
    del root["RNA"].attrs["dataset_fingerprint"]
    store.zarr_mode = mode
    with pytest.raises(ValueError, match="inconsistent dataset identity"):
        store._ensure_dataset_fingerprint("RNA")
    assert "dataset_fingerprint" not in root["RNA"].attrs
