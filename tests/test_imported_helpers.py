"""Validation helpers for imported coordinate/embedding adapters."""

import numpy as np
import pytest

from scarf.embeddings.imported import (
    _positive_block_rows,
    _required_payload_fingerprints,
    _resolve_source,
    _string_block,
    _string_source_length,
    _validate_fingerprint,
    _validate_numeric_source,
    _validate_source_digest,
)


def test_positive_block_rows_and_fingerprint_validators():
    assert _positive_block_rows(4) == 4
    with pytest.raises(TypeError, match="positive integer"):
        _positive_block_rows(True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="greater than zero"):
        _positive_block_rows(0)

    good = "a" * 64
    assert _validate_fingerprint(good, "fp") == good
    with pytest.raises(ValueError, match="64-character lowercase"):
        _validate_fingerprint("A" * 64, "fp")
    with pytest.raises(ValueError, match="64-character lowercase"):
        _validate_fingerprint("a" * 63, "fp")
    with pytest.raises(ValueError, match="hexadecimal"):
        _validate_fingerprint("g" * 64, "fp")

    digest = b"x" * 32
    assert _validate_source_digest(digest) == digest
    assert _validate_source_digest(digest) is digest
    with pytest.raises(TypeError, match="32 bytes"):
        _validate_source_digest(b"short")
    with pytest.raises(TypeError, match="32 bytes"):
        _validate_source_digest(bytearray(b"x" * 32))


def test_required_payload_fingerprints_checks_membership():
    fingerprints = {"coords": "a" * 64, "ids": "b" * 64}
    resolved = _required_payload_fingerprints(fingerprints, {"coords", "ids"})
    assert resolved == fingerprints
    assert resolved is not fingerprints
    with pytest.raises(TypeError, match="mapping"):
        _required_payload_fingerprints(["coords"], {"coords"})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Missing payload fingerprints"):
        _required_payload_fingerprints({"coords": "a" * 64}, {"coords", "ids"})
    with pytest.raises(ValueError, match="Unexpected payload fingerprints"):
        _required_payload_fingerprints(fingerprints, {"coords"})


def test_resolve_source_rejects_stream_without_shape_and_one_shot_reuse():
    with pytest.raises(ValueError, match="required for streamed blocks"):
        _resolve_source(
            iter([np.ones((2, 2))]),
            name="coords",
            shape=None,
            dtype=None,
            block_rows=2,
        )

    source = _resolve_source(
        iter([np.ones((2, 3), dtype=np.float32)]),
        name="coords",
        shape=(2, 3),
        dtype=np.float32,
        block_rows=2,
    )
    assert source.reusable is False
    first = list(source.produce())
    assert first[0].shape == (2, 3)
    np.testing.assert_array_equal(first[0], np.ones((2, 3), dtype=np.float32))
    with pytest.raises(RuntimeError, match="only be consumed once"):
        list(source.produce())

    values = np.arange(6, dtype=np.float64).reshape(3, 2)
    array_source = _resolve_source(
        values,
        name="coords",
        shape=None,
        dtype=None,
        block_rows=2,
    )
    blocks = list(array_source.produce())
    assert [block.shape[0] for block in blocks] == [2, 1]
    np.testing.assert_array_equal(np.vstack(blocks), values)
    assert array_source.reusable is True
    np.testing.assert_array_equal(
        np.vstack(list(array_source.produce())),
        values,
    )
    with pytest.raises(TypeError, match="floating-point"):
        _validate_numeric_source(
            _resolve_source(
                np.arange(4, dtype=np.int32).reshape(2, 2),
                name="coords",
                shape=None,
                dtype=None,
                block_rows=2,
            ),
            "coords",
            2,
        )


def test_string_block_validates_identifiers_and_length():
    assert _string_source_length(["a", "b"]) == 2
    with pytest.raises(ValueError, match="one-dimensional"):
        _string_source_length(np.array([["a", "b"]]))

    values = np.array(["cell-1", "cell-2"])
    np.testing.assert_array_equal(_string_block(values, 0, 2), ["cell-1", "cell-2"])
    with pytest.raises(ValueError, match="invalid identifier"):
        _string_block(np.array(["", "x"]), 0, 2)
    with pytest.raises(TypeError, match="must contain strings"):
        _string_block(np.array([1, 2]), 0, 2)
    with pytest.raises(ValueError, match="invalid UTF-8"):
        _string_block(np.array([b"\xff"], dtype=object), 0, 1)
