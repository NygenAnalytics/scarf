"""Tests for the bounded vector readers behind the Seurat importers."""

import numpy as np
import pytest

from scarf.readers._seurat.errors import MatrixSourceError
from scarf.readers._seurat.values import (
    class_names,
    decode_text,
    decode_text_values,
    read_window,
    scalar_value,
    vector_length,
)


class _Shaped:
    """A lazy vector that reports a shape but no length."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape


class _Blocks:
    """A vector whose block reads return a fixed payload."""

    def __init__(self, length: int, payload: object) -> None:
        self.length = length
        self.payload = payload

    def __len__(self) -> int:
        return self.length

    def read_block(self, start: int, stop: int) -> object:
        return self.payload


def test_vector_length_falls_back_to_a_one_dimensional_shape() -> None:
    assert vector_length(_Shaped((4,)), "v") == 4
    with pytest.raises(MatrixSourceError, match="must be one-dimensional"):
        vector_length(_Shaped((2, 2)), "v")
    with pytest.raises(TypeError, match="no bounded length"):
        vector_length(object(), "v")


def test_read_window_flattens_columns_and_checks_the_returned_size() -> None:
    column = np.arange(4).reshape(4, 1)
    window = read_window(column, 1, 3, dtype=np.float32)
    assert window.dtype == np.float32
    np.testing.assert_array_equal(window, [1.0, 2.0])
    with pytest.raises(MatrixSourceError, match="returned 2 values; expected 3"):
        read_window(_Blocks(3, [1, 2]), 0, 3, object_path="v")
    with pytest.raises(MatrixSourceError, match="does not support bounded slicing"):
        read_window({1, 2}, 0, 1, object_path="v")


def test_scalar_value_requires_one_element() -> None:
    assert scalar_value(_Blocks(1, [np.int32(5)]), "v") == 5
    assert scalar_value(_Blocks(1, np.array([7])), "v") == 7
    with pytest.raises(MatrixSourceError, match="must be scalar"):
        scalar_value(_Blocks(1, np.array([1, 2])), "v")
    with pytest.raises(MatrixSourceError, match="must be scalar"):
        scalar_value([1, 2], "v")


def test_class_names_normalize_scalars_and_sequences() -> None:
    assert class_names(None) == ()
    assert class_names("dgCMatrix") == ("dgCMatrix",)
    assert class_names(["Assay5", "KeyMixin"]) == ("Assay5", "KeyMixin")
    assert class_names(3) == ("3",)
    with pytest.raises(MatrixSourceError, match="cannot be empty"):
        class_names([])


def test_decode_text_rejects_invalid_utf8_nul_and_non_text() -> None:
    assert decode_text(np.bytes_(b"caf\xc3\xa9"), "v") == "café"
    for value in (b"\xff", "\ud800"):
        with pytest.raises(MatrixSourceError, match="not valid UTF-8"):
            decode_text(value, "v")
    with pytest.raises(MatrixSourceError, match="contains NUL"):
        decode_text("a\x00b", "v")
    with pytest.raises(TypeError, match="must contain strings"):
        decode_text(1, "v")


def test_decode_text_values_checks_the_returned_count() -> None:
    assert decode_text_values(
        np.array([b"a", b"bc"]), object_path="v", max_bytes=1024
    ) == ("a", "bc")
    with pytest.raises(MatrixSourceError, match="returned 2 values; expected 3"):
        decode_text_values(_Blocks(3, ["a", "b"]), object_path="v", max_bytes=1024)
    with pytest.raises(TypeError, match="must be a sequence"):
        decode_text_values("abc", object_path="v", max_bytes=1024)
