"""Persisted ANN indexes load back only when their payload and record agree."""

import hashlib
from pathlib import Path

import hnswlib
import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

from scarf.storage.ann_index import (
    ANN_INDEX_ARRAY,
    load_ann_index,
    save_ann_index,
    validate_ann_index_payload,
)


def _index(n_items: int = 20, dim: int = 3) -> tuple[hnswlib.Index, np.ndarray]:
    data = np.random.default_rng(0).random((n_items, dim), dtype=np.float32)
    index = hnswlib.Index(space="l2", dim=dim)
    index.init_index(max_elements=n_items, ef_construction=50, M=8, random_seed=1)
    index.add_items(data, np.arange(n_items))
    return index, data


def _saved(index: hnswlib.Index, element_count: int = 20) -> zarr.Group:
    group = zarr.open_group(store=MemoryStore(), mode="w")
    save_ann_index(
        group,
        index,
        profile="fast_local",
        metric="l2",
        dimensions=3,
        element_count=element_count,
    )
    return group


def test_saved_index_holds_the_hnswlib_file_and_loads_the_same_neighbors(
    tmp_path: Path,
) -> None:
    index, data = _index()
    index.save_index(str(tmp_path / "reference.bin"))
    reference = (tmp_path / "reference.bin").read_bytes()

    group = _saved(index)

    stored = group[ANN_INDEX_ARRAY]
    assert np.asarray(stored[:]).tobytes() == reference
    assert stored.attrs.asdict() == {
        "byte_length": len(reference),
        "ann_index_format_version": 1,
        "metric": "l2",
        "dimensions": 3,
        "element_count": 20,
        "payload_sha256": hashlib.sha256(reference).hexdigest(),
    }
    loaded = load_ann_index(group, "l2", 3, expected_count=20)
    assert loaded.get_current_count() == 20
    labels, distances = loaded.knn_query(data, k=3)
    expected_labels, expected_distances = index.knn_query(data, k=3)
    np.testing.assert_array_equal(labels, expected_labels)
    np.testing.assert_array_equal(distances, expected_distances)


def test_save_refuses_a_count_that_differs_from_the_index() -> None:
    index, _data = _index()
    group = zarr.open_group(store=MemoryStore(), mode="w")

    with pytest.raises(
        ValueError, match="element count does not match its coordinates"
    ):
        save_ann_index(
            group,
            index,
            profile="fast_local",
            metric="l2",
            dimensions=3,
            element_count=21,
        )
    assert ANN_INDEX_ARRAY not in group


def test_load_refuses_a_tampered_payload_or_count() -> None:
    index, _data = _index()
    group = _saved(index)
    stored = group[ANN_INDEX_ARRAY]
    original = int(stored[0])

    stored[0] = original ^ 0xFF
    for check in (load_ann_index, validate_ann_index_payload):
        with pytest.raises(
            ValueError, match="ANN index payload digest does not match its metadata"
        ):
            check(group, "l2", 3)

    # An intact payload whose record claims one more element than it holds.
    stored[0] = original
    stored.attrs["element_count"] = 21
    validate_ann_index_payload(group, "l2", 3)
    with pytest.raises(
        ValueError, match="ANN index element count does not match coordinates"
    ):
        load_ann_index(group, "l2", 3)
