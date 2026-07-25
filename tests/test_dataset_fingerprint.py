import pytest

from scarf.storage.artifacts import ValueFingerprintBuilder


def test_dataset_fingerprint_is_created_once_and_left_dormant(
    datastore_ephemeral,
    monkeypatch,
) -> None:
    assay = datastore_ephemeral.get_assay("RNA")
    if "dataset_fingerprint" in assay.attrs:
        del assay.attrs["dataset_fingerprint"]

    fingerprint = datastore_ephemeral._ensure_dataset_fingerprint("RNA")
    assert assay.attrs["dataset_fingerprint"] == fingerprint

    def fail_if_recomputed(*_args, **_kwargs):
        raise AssertionError("dataset_fingerprint must remain dormant")

    monkeypatch.setattr(ValueFingerprintBuilder, "update_array", fail_if_recomputed)
    assert datastore_ephemeral._ensure_dataset_fingerprint("RNA") == fingerprint


def test_read_only_store_does_not_create_dataset_fingerprint(
    datastore_ephemeral,
) -> None:
    assay = datastore_ephemeral.get_assay("RNA")
    if "dataset_fingerprint" in assay.attrs:
        del assay.attrs["dataset_fingerprint"]
    datastore_ephemeral.zarr_mode = "r"

    with pytest.raises(PermissionError, match="cannot be stored read-only"):
        datastore_ephemeral._ensure_dataset_fingerprint("RNA")
    assert "dataset_fingerprint" not in assay.attrs
