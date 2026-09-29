"""Tests for the stored ANN index metadata contract."""

import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

from scarf.storage.ann_index import (
    ANN_INDEX_ARRAY,
    ANN_INDEX_FORMAT_VERSION,
    validate_ann_index_contract,
)


def _ann_group(**overrides: object) -> zarr.Group:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    payload = root.create_array(ANN_INDEX_ARRAY, data=np.zeros(8, dtype=np.uint8))
    payload.attrs.update(
        {
            "byte_length": 8,
            "ann_index_format_version": ANN_INDEX_FORMAT_VERSION,
            "metric": "l2",
            "dimensions": 3,
            "element_count": 5,
            "payload_sha256": "0" * 64,
        }
        | overrides
    )
    return root


def test_ann_index_contract_accepts_a_complete_record() -> None:
    validated = validate_ann_index_contract(_ann_group(), "l2", 3, expected_count=5)

    assert (validated.stored_count, validated.stored_digest) == (5, "0" * 64)
    with pytest.raises(ValueError, match="does not match coordinates"):
        validate_ann_index_contract(_ann_group(), "l2", 3, expected_count=6)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"byte_length": 7}, "byte length does not match"),
        ({"byte_length": True}, "byte length does not match"),
        (
            {"ann_index_format_version": ANN_INDEX_FORMAT_VERSION + 1},
            "format version is unsupported",
        ),
        ({"ann_index_format_version": True}, "format version is unsupported"),
        ({"metric": "cosine"}, "metric does not match"),
        ({"dimensions": 2}, "dimensions do not match"),
        ({"element_count": True}, "element count is invalid"),
        ({"element_count": 5.0}, "element count is invalid"),
        ({"payload_sha256": 1}, "payload digest is invalid"),
    ],
)
def test_ann_index_contract_rejects_each_malformed_attribute(
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_ann_index_contract(_ann_group(**overrides), "l2", 3)


def test_ann_index_contract_rejects_missing_and_misshapen_payloads() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    with pytest.raises(FileNotFoundError, match="not found"):
        validate_ann_index_contract(root, "l2", 3)
    root.create_array(ANN_INDEX_ARRAY, data=np.zeros(4, dtype=np.int8))
    with pytest.raises(ValueError, match="one-dimensional uint8"):
        validate_ann_index_contract(root, "l2", 3)
