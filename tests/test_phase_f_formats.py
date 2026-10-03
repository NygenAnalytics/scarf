"""Phase F format locks: ANN byte storage."""

import hashlib

import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

from scarf.storage.ann_index import save_ann_index, validate_ann_index_payload
from scarf.storage.arrays import create_zarr_dataset


def test_save_ann_index_writes_exact_zarr_bytes() -> None:
    class _FakeIndex:
        def get_current_count(self) -> int:
            return 3

        def save_index(self, path: str) -> None:
            with open(path, "wb") as handle:
                handle.write(b"hnsw-bytes")

    root = zarr.open_group(store=MemoryStore(), mode="w")
    group = root.create_group("ann")
    save_ann_index(
        group,
        _FakeIndex(),
        profile="fast_local",
        metric="l2",
        dimensions=4,
        element_count=3,
    )

    assert "ann_idx_bytes" in group
    np.testing.assert_array_equal(
        group["ann_idx_bytes"][:],
        np.frombuffer(b"hnsw-bytes", dtype=np.uint8),
    )
    assert group["ann_idx_bytes"].attrs["byte_length"] == len(b"hnsw-bytes")
    assert group["ann_idx_bytes"].attrs["metric"] == "l2"
    assert group["ann_idx_bytes"].attrs["dimensions"] == 4
    assert group["ann_idx_bytes"].attrs["element_count"] == 3
    assert (
        group["ann_idx_bytes"].attrs["payload_sha256"]
        == hashlib.sha256(b"hnsw-bytes").hexdigest()
    )


@pytest.mark.parametrize("attribute", ["byte_length", "payload_sha256", "metric"])
def test_ann_index_payload_requires_its_complete_metadata(attribute: str) -> None:
    class _FakeIndex:
        def get_current_count(self) -> int:
            return 3

        def save_index(self, path: str) -> None:
            with open(path, "wb") as handle:
                handle.write(b"hnsw-bytes")

    group = zarr.open_group(store=MemoryStore(), mode="w").create_group("ann")
    save_ann_index(
        group,
        _FakeIndex(),
        profile="fast_local",
        metric="l2",
        dimensions=4,
        element_count=3,
    )
    validate_ann_index_payload(group, "l2", 4, 3)
    del group["ann_idx_bytes"].attrs[attribute]
    with pytest.raises(ValueError, match=f"metadata is missing: {attribute}"):
        validate_ann_index_payload(group, "l2", 4, 3)


def test_empty_array_uses_nonzero_chunk_dimensions() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")

    output = create_zarr_dataset(
        root,
        "empty_edges",
        (1, 2),
        np.int64,
        (0, 2),
    )

    assert output.shape == (0, 2)
    assert output.chunks == (1, 2)
