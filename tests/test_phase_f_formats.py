"""Phase F format locks: stable subset identity and ANN byte storage."""

import numpy as np
import zarr
from zarr.storage import MemoryStore

from scarf.assay.base import Assay
from scarf.storage.ann_index import save_ann_index
from scarf.storage.arrays import create_zarr_dataset


def test_subset_hash_is_stable_string_digest() -> None:
    cells = np.array([0, 2, 5], dtype=np.int64)
    feats = np.array([1, 3, 4, 7], dtype=np.int64)
    first = Assay._create_subset_hash(cells, feats)
    second = Assay._create_subset_hash(cells.copy(), feats.copy())
    assert isinstance(first, str)
    assert first == second
    assert first != Assay._create_subset_hash(cells[::-1], feats)


def test_subset_hash_matches_pinned_golden_value() -> None:
    # Pin the exact digest so the persisted subset identity cannot change
    # silently. Changing this value breaks reuse of stored normalized data.
    assert (
        Assay._create_subset_hash(
            np.array([0, 2, 5], dtype=np.int64),
            np.array([1, 3, 4, 7], dtype=np.int64),
        )
        == "2e67c2c409b18b438d0451d9ab0f3e43"
    )


def test_subset_hash_encodes_cell_feature_boundary() -> None:
    # Same concatenated indices but a different cell/feature split must not
    # collide, otherwise a stale normalized cache could be silently reused.
    left = Assay._create_subset_hash(
        np.array([0, 1], dtype=np.int64), np.array([2, 3], dtype=np.int64)
    )
    right = Assay._create_subset_hash(
        np.array([0, 1, 2], dtype=np.int64), np.array([3], dtype=np.int64)
    )
    assert left != right


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
    assert len(group["ann_idx_bytes"].attrs["payload_sha256"]) == 64


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
