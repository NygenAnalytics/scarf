"""Offline tests for Cytebase H5AD inspection, count selection and manifest checks."""

import copy
import dataclasses
import json
import re
from collections.abc import Callable
from typing import Any

import h5py
import numpy as np
import pytest
from scipy.sparse import csr_matrix

from tests.fixtures_cytebase import (
    CITATION,
    COLLECTION_ID,
    COUNTS,
    full_manifest,
    write_categorical,
    write_h5ad,
)

pytest.importorskip("pydantic")

from pydantic import ValidationError

from scarf.cytebase.pipeline import build
from scarf.cytebase.pipeline.models import Manifest

pytestmark = pytest.mark.usefixtures("cytebase_offline")

CSR = csr_matrix(COUNTS)
NORMALIZED = COUNTS / 2.0
RAW_COUNTS = np.hstack([COUNTS, COUNTS[:, :2]])
OMITTED = "Array exceeds the metadata export size limit"
INDEX_ARRAYS = "Sparse indices and indptr must be one-dimensional integer arrays"
INDPTR_ENDPOINTS = (
    "Sparse indptr endpoints or length do not match the matrix dimensions"
)
UNUSABLE_VALUES = "Stored matrix values are missing or have unsupported dimensions"
SAMPLED_CHECK = "Conversion requires the current sampled-row counts check"
DISAGREEMENT = "Selected matrix dimensions or feature table disagree with the manifest"
NEEDS_INPUT_SHAPE = "selectionNeedsInput must include a question and options"
METADATA_MESSAGE = "Reading feature metadata, cell annotations and source embeddings"


@pytest.fixture
def h5(tmp_path):
    """An HDF5 file held in memory and never written to disk."""
    with h5py.File(
        tmp_path / "memory.h5", "w", driver="core", backing_store=False
    ) as file:
        yield file


@pytest.fixture
def manifest(tmp_path) -> dict:
    """The complete manifest conversion receives for the default H5AD."""
    return full_manifest(write_h5ad(tmp_path / "source.h5ad"))


def _replaced(values: np.ndarray, position: int, value: float) -> np.ndarray:
    changed = values.copy()
    changed[position] = value
    return changed


def _frame(parent: h5py.Group, name: str, length: int) -> None:
    frame = parent.create_group(name)
    frame.attrs["_index"] = "_index"
    names = [f"{name}{i}".encode() for i in range(length)]
    frame.create_dataset("_index", data=np.array(names, dtype="S16"))


def _csr(
    parent: h5py.Group,
    name: str,
    counts: np.ndarray,
    *,
    encoding: str = "csr_matrix",
    shape: bool = True,
    **parts: np.ndarray | None,
) -> None:
    """Write a CSR group; ``parts`` replace data, indices or indptr, or omit them."""
    matrix = csr_matrix(counts)
    group = parent.create_group(name)
    group.attrs["encoding-type"] = encoding
    if shape:
        group.attrs["shape"] = counts.shape
    arrays = {"data": matrix.data, "indices": matrix.indices, "indptr": matrix.indptr}
    for part, values in (arrays | parts).items():
        if values is not None:
            group.create_dataset(part, data=values)


def _layout(
    h5: h5py.File,
    counts: np.ndarray = COUNTS,
    *,
    dense: bool = False,
    cells: int | None = 6,
    genes: int | None = 5,
    **matrix: Any,
) -> None:
    """Write obs, var and X; a table size of None leaves that table out."""
    if cells is not None:
        _frame(h5, "obs", cells)
    if genes is not None:
        _frame(h5, "var", genes)
    if dense:
        h5.create_dataset("X", data=counts)
    else:
        _csr(h5, "X", counts, **matrix)


def _validate(
    h5: h5py.File, key: str = "X", progress: Callable | None = None
) -> dict[str, Any]:
    return build._validate_candidate(h5, key, build._matrix_info(h5[key]), progress)


def _recorder() -> tuple[list[tuple[str, dict]], Callable]:
    events: list[tuple[str, dict]] = []

    def progress(stage: str, **fields: Any) -> None:
        events.append((stage, fields))

    return events, progress


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param(b"caf\xc3\xa9", "café", id="utf8-bytes"),
        pytest.param(b"bad\xff", "bad\ufffd", id="invalid-bytes"),
        pytest.param(np.array([[1, 2], [3, 4]]), [[1, 2], [3, 4]], id="array"),
        pytest.param(np.int64(3), 3, id="numpy-integer"),
        pytest.param(np.bool_(True), True, id="numpy-bool"),
        pytest.param(np.float32("nan"), None, id="numpy-nan"),
        pytest.param(float("-inf"), None, id="infinity"),
        pytest.param(
            {b"key": np.float64(1.5), 2: (1, float("inf"))},
            {"key": 1.5, "2": [1, None]},
            id="mapping",
        ),
        pytest.param(None, None, id="none"),
        pytest.param("text", "text", id="text"),
        pytest.param(2.5, 2.5, id="float"),
        pytest.param(complex(1, 2), "(1+2j)", id="other"),
    ],
)
def test_json_safe_converts_hdf5_values_to_plain_json(value, expected):
    result = build._json_safe(value)
    assert result == expected
    assert type(result) is type(expected)
    json.dumps(result, allow_nan=False)


def test_uns_value_exports_nested_metadata_and_omits_large_arrays(h5, monkeypatch):
    monkeypatch.setattr(build, "_UNS_MAX_VALUES", 3)
    monkeypatch.setattr(build, "_UNS_MAX_BYTES", 16)
    uns = h5.create_group("uns")
    uns.create_dataset("title", data="Lung atlas")
    uns.create_group("colors").create_dataset(
        "cell_type", data=np.array([b"#ffffff", b"#000000"])
    )
    uns.create_dataset("at_value_limit", data=np.arange(3, dtype=np.int8))
    uns.create_dataset("at_byte_limit", data=np.array([0.5, 1.5]))
    uns.create_dataset("too_many", data=np.arange(4, dtype=np.int8))
    uns.create_dataset("too_large", data=np.zeros(3))

    assert build._uns_value(uns) == {
        "at_byte_limit": [0.5, 1.5],
        "at_value_limit": [0, 1, 2],
        "colors": {"cell_type": ["#ffffff", "#000000"]},
        "title": "Lung atlas",
        "too_large": {
            "omitted": True,
            "reason": OMITTED,
            "shape": [3],
            "dtype": "float64",
        },
        "too_many": {"omitted": True, "reason": OMITTED, "shape": [4], "dtype": "int8"},
    }


def test_matrix_info_reads_structure_without_values(h5):
    h5.create_dataset("dense", data=COUNTS)
    _csr(h5, "csr", COUNTS)
    legacy = h5.create_group("legacy")
    legacy.attrs["h5sparse_format"] = np.bytes_(b"csc")
    legacy.attrs["h5sparse_shape"] = (6, 5)
    legacy.create_dataset("data", data=np.ones(3, dtype=np.float32))
    stored = h5.create_group("stored_shape")
    stored.create_dataset("shape", data=np.array([6, 5]))
    stored.create_dataset("data", data=np.ones(2, dtype=np.uint16))
    h5.create_group("empty")

    assert {name: build._matrix_info(node) for name, node in h5.items()} == {
        "csr": {"encoding": "csr_matrix", "shape": [6, 5], "dtype": "int32", "nnz": 22},
        "dense": {"encoding": "array", "shape": [6, 5], "dtype": "int32", "nnz": None},
        "empty": {"encoding": "csr", "shape": None, "dtype": None, "nnz": None},
        "legacy": {"encoding": "csc", "shape": [6, 5], "dtype": "float32", "nnz": 3},
        "stored_shape": {
            "encoding": "csr",
            "shape": [6, 5],
            "dtype": "uint16",
            "nnz": 2,
        },
    }


def test_column_length_reads_datasets_and_encoded_columns(h5):
    h5.create_dataset("plain", data=np.arange(4))
    h5.create_dataset("matrix", data=np.zeros((4, 2)))
    write_categorical(h5, "categorical", ["a", None, "b"])
    nullable = h5.create_group("nullable")
    nullable.create_dataset("values", data=np.arange(5))
    nullable.create_dataset("mask", data=np.zeros(5, dtype=bool))
    h5.create_group("nested_codes").create_group("codes")
    h5.create_group("empty")

    assert {name: build._column_length(node) for name, node in h5.items()} == {
        "categorical": 3,
        "empty": None,
        "matrix": None,
        "nested_codes": None,
        "nullable": 5,
        "plain": 4,
    }


def test_table_length_prefers_the_declared_index(h5):
    indexed = h5.create_group("indexed")
    indexed.attrs["_index"] = np.bytes_(b"cell_id")
    indexed.create_dataset("aaa", data=np.arange(3))
    indexed.create_dataset("cell_id", data=np.arange(6))
    default = h5.create_group("default_index")
    default.create_dataset("A_first", data=np.arange(2))
    default.create_dataset("_index", data=np.arange(4))
    fallback = h5.create_group("fallback")
    fallback.attrs["_index"] = "missing"
    fallback.create_dataset("a_matrix", data=np.zeros((3, 2)))
    write_categorical(fallback, "b_codes", ["x", "y", "x", "y", "x"])
    h5.create_group("no_columns").create_dataset("matrix", data=np.zeros((2, 2)))
    h5.create_dataset("dataset", data=np.arange(3))

    assert build._table_length(h5["indexed"]) == 6
    assert build._table_length(h5["default_index"]) == 4
    assert build._table_length(h5["fallback"]) == 5
    assert build._table_length(h5["no_columns"]) is None
    assert build._table_length(h5["dataset"]) is None
    assert build._table_length(None) is None


def test_sample_row_blocks_reads_rows_in_bounded_blocks(h5, monkeypatch):
    monkeypatch.setattr(build, "_BLOCK_VALUES", 2)
    h5.create_dataset("dense", data=COUNTS)
    with_empty_row = COUNTS.copy()
    with_empty_row[2] = 0
    _csr(h5, "sparse", with_empty_row)
    sparse = h5["sparse"]

    def sparse_row(row: int) -> list[list[int]]:
        blocks = build._sample_row_blocks(
            sparse["data"], row, 5, sparse["indices"], sparse["indptr"]
        )
        return [block.tolist() for block in blocks]

    dense_blocks = build._sample_row_blocks(h5["dense"], 0, 5, None, None)
    assert [block.tolist() for block in dense_blocks] == [[1, 0], [2, 5], [3]]
    assert sparse_row(1) == [[4, 6], [1]]
    assert sparse_row(2) == []


@pytest.mark.parametrize(
    "indptr",
    [[0, 3, 2], [0, 3, 9], [0, -1, 3]],
    ids=["decreasing", "past-stored-values", "negative"],
)
def test_sample_row_blocks_rejects_invalid_row_boundaries(h5, indptr):
    h5.create_dataset("data", data=np.ones(4))
    h5.create_dataset("indices", data=np.zeros(4, dtype=np.int32))
    h5.create_dataset("indptr", data=np.array(indptr))
    blocks = build._sample_row_blocks(h5["data"], 1, 5, h5["indices"], h5["indptr"])
    with pytest.raises(ValueError, match="Sparse row 1 has invalid indptr boundaries"):
        list(blocks)


def test_validate_candidate_accepts_csr_counts(h5):
    _layout(h5)
    assert _validate(h5) == {
        "encoding": "csr_matrix",
        "shape": [6, 5],
        "dtype": "int32",
        "nnz": 22,
        "present": True,
        "validCounts": True,
        "validationComplete": True,
        "validationMode": "sampled_rows",
        "fullMatrixValidated": False,
        "reasons": [],
        "featureAttrsKey": "var",
        "expectedShape": [6, 5],
        "formatSupported": True,
        "sampledRows": [0, 1, 2, 3, 4, 5],
        "columnsChecked": 5,
        "rowsChecked": 6,
        "valuesChecked": 22,
        "finite": True,
        "nonnegative": True,
        "integerLike": True,
        "sampleMax": 7,
    }


def test_validate_candidate_accepts_integral_dense_floats(h5):
    _layout(h5, COUNTS.astype(np.float32), dense=True)
    result = _validate(h5)
    assert result["validCounts"] is True
    assert result["encoding"] == "array"
    assert result["formatSupported"] is True
    assert result["dtype"] == "float32"
    assert result["valuesChecked"] == 30
    assert result["integerLike"] is True
    assert result["sampleMax"] == 7.0


def test_validate_candidate_checks_raw_counts_against_raw_features(h5):
    _frame(h5, "obs", 6)
    _frame(h5, "var", 3)
    raw = h5.create_group("raw")
    _frame(raw, "var", 5)
    _csr(raw, "X", COUNTS)
    result = _validate(h5, "raw/X")
    assert result["featureAttrsKey"] == "raw/var"
    assert result["expectedShape"] == [6, 5]
    assert result["validCounts"] is True


@pytest.mark.parametrize(
    ("layout", "reason"),
    [
        pytest.param(
            {"shape": False},
            "Matrix shape None does not match obs and var dimensions [6, 5]",
            id="missing-shape",
        ),
        pytest.param(
            {"counts": np.arange(6), "dense": True},
            "Matrix shape [6] does not match obs and var dimensions [6, 5]",
            id="one-dimensional-dense",
        ),
        pytest.param(
            {"cells": 4},
            "Matrix shape [6, 5] does not match obs and var dimensions [4, 5]",
            id="obs-length-mismatch",
        ),
        pytest.param(
            {"cells": None},
            "Matrix shape [6, 5] does not match obs and var dimensions [None, 5]",
            id="missing-obs",
        ),
        pytest.param(
            {"genes": None},
            "Matrix shape [6, 5] does not match obs and var dimensions [6, None]",
            id="missing-var",
        ),
        pytest.param(
            {"counts": np.zeros((0, 5), dtype=np.int32), "cells": 0},
            "Matrix shape [0, 5] does not match obs and var dimensions [0, 5]",
            id="zero-cells",
        ),
        pytest.param({"data": None}, UNUSABLE_VALUES, id="missing-data"),
        pytest.param(
            {"data": CSR.data.reshape(11, 2)},
            UNUSABLE_VALUES,
            id="two-dimensional-data",
        ),
        pytest.param(
            {"data": CSR.data > 0},
            "Expected real numeric counts; got bool",
            id="boolean-data",
        ),
        pytest.param(
            {"encoding": "csc_matrix"},
            "Sparse encoding 'csc_matrix' cannot be sampled by cell for bounded "
            "conversion. Scarf materializes CSC inputs. Provide CSR or dense counts.",
            id="csc",
        ),
        pytest.param(
            {"encoding": "COO"},
            "Sparse encoding 'coo' cannot be sampled by cell",
            id="unknown-encoding",
        ),
        pytest.param(
            {"indices": CSR.indices.astype(np.float64)},
            INDEX_ARRAYS,
            id="float-indices",
        ),
        pytest.param(
            {"indptr": CSR.indptr.reshape(-1, 1)},
            INDEX_ARRAYS,
            id="two-dimensional-indptr",
        ),
        pytest.param({"indptr": None}, INDEX_ARRAYS, id="missing-indptr"),
        pytest.param(
            {"indices": np.append(CSR.indices, 0)},
            "Sparse indices and values have different lengths",
            id="indices-length",
        ),
        pytest.param({"indptr": CSR.indptr[:-1]}, INDPTR_ENDPOINTS, id="short-indptr"),
        pytest.param(
            {"indptr": _replaced(CSR.indptr, 0, 1)}, INDPTR_ENDPOINTS, id="indptr-start"
        ),
        pytest.param(
            {"indptr": _replaced(CSR.indptr, -1, 21)}, INDPTR_ENDPOINTS, id="indptr-end"
        ),
        pytest.param(
            {"indptr": _replaced(CSR.indptr, 1, 8)},
            "Sparse row 1 has invalid indptr boundaries",
            id="decreasing-indptr",
        ),
        pytest.param(
            {"indices": _replaced(CSR.indices, 0, 5)},
            "Sparse indices in sampled row 0 exceed gene dimensions",
            id="index-past-genes",
        ),
        pytest.param(
            {"indices": _replaced(CSR.indices, 4, -1)},
            "Sparse indices in sampled row 1 exceed gene dimensions",
            id="negative-index",
        ),
    ],
)
def test_validate_candidate_rejects_unusable_structure(h5, layout, reason):
    _layout(h5, **layout)
    result = _validate(h5)
    [message] = result["reasons"]
    assert reason in message
    assert result["present"] is True
    assert result["validCounts"] is False
    assert result["validationComplete"] is False


@pytest.mark.parametrize("dense", [True, False], ids=["dense", "csr"])
@pytest.mark.parametrize(
    ("value", "reason", "integer_like"),
    [
        pytest.param(
            np.inf, "Sampled rows include NaN or infinity", False, id="infinity"
        ),
        pytest.param(-1.0, "Sampled rows include negative counts", True, id="negative"),
        pytest.param(
            0.5, "Sampled rows include non-integer counts", False, id="fraction"
        ),
    ],
)
def test_validate_candidate_reports_invalid_sampled_values(
    h5, dense, value, reason, integer_like
):
    counts = COUNTS.astype(np.float64)
    counts[3, 2] = value
    _layout(h5, counts, dense=dense)
    result = _validate(h5)
    assert result["reasons"] == [reason]
    assert result["validationComplete"] is True
    assert result["validCounts"] is False
    assert result["integerLike"] is integer_like
    assert result["sampleMax"] == 7.0


def test_validate_candidate_reports_nan_and_ignores_it_in_sample_max(h5):
    counts = COUNTS.astype(np.float64)
    counts[2] = np.nan  # The only row that holds the maximum count of 7.
    _layout(h5, counts, dense=True)
    result = _validate(h5)
    assert "Sampled rows include NaN or infinity" in result["reasons"]
    assert result["validationComplete"] is True
    assert result["validCounts"] is False
    assert result["finite"] is False
    assert result["integerLike"] is False
    assert result["sampleMax"] == 6.0


def test_validate_candidate_checks_evenly_spaced_rows_only(h5, monkeypatch):
    monkeypatch.setattr(build, "_VALIDATION_ROWS", 3)
    counts = COUNTS.copy()
    counts[1, 0] = -1  # Row 1 is not among the sampled rows.
    _layout(h5, counts)
    result = _validate(h5)
    assert result["sampledRows"] == [0, 2, 5]
    assert result["rowsChecked"] == 3
    assert result["valuesChecked"] == 12
    assert result["validCounts"] is True


@pytest.mark.parametrize("dense", [True, False], ids=["dense", "csr"])
def test_validate_candidate_checks_every_block_of_a_row(h5, monkeypatch, dense):
    monkeypatch.setattr(build, "_BLOCK_VALUES", 2)
    counts = COUNTS.astype(np.float64)
    counts[5, 4] = 1.5  # Stored in the final block of the final row.
    _layout(h5, counts, dense=dense)
    result = _validate(h5)
    assert result["reasons"] == ["Sampled rows include non-integer counts"]
    assert result["valuesChecked"] == (30 if dense else 22)


def test_validate_candidate_reports_row_progress(h5):
    _layout(h5)
    events, progress = _recorder()
    _validate(h5, progress=progress)
    assert events[0] == (
        "validating_counts",
        {
            "completed": 0,
            "total": 6,
            "unit": "rows",
            "message": "X: sampling 6 cells across all 5 genes",
        },
    )
    assert events[-1] == (
        "validating_counts",
        {
            "completed": 6,
            "total": 6,
            "unit": "rows",
            "storedValuesChecked": 22,
            "message": "X: sampling cells across all 5 genes",
        },
    )
    checked = [fields["storedValuesChecked"] for _, fields in events[1:]]
    assert checked == [4, 7, 11, 14, 18, 22]


def _metadata(h5: h5py.File) -> h5py.Group:
    obs = h5.create_group("obs")
    write_categorical(obs, "cell_type", ["T cell", None, "B cell", "T cell"])
    counts = obs.create_group("n_counts")
    counts.attrs["encoding-type"] = "nullable-integer"
    counts.create_dataset("values", data=np.array([3, 0, 5, 7], dtype=np.int32))
    counts.create_dataset("mask", data=np.array([False, True, False, False]))
    doublet = obs.create_group("is_doublet")
    doublet.attrs["encoding-type"] = "nullable-boolean"
    doublet.create_dataset("values", data=np.array([True, False, True, False]))
    doublet.create_dataset("mask", data=np.array([False, False, False, True]))
    obs.create_dataset("score", data=np.array([1.0, np.nan, 3.0, np.inf]))
    obs.create_dataset("is_primary_data", data=np.array([True, True, False, True]))
    obs.create_group("unsupported").create_dataset("x", data=np.arange(4))
    return obs


def test_column_info_describes_storage(h5):
    obs = _metadata(h5)
    assert {name: build._column_info(node) for name, node in obs.items()} == {
        "cell_type": {"dtype": "|S6", "encoding": "categorical", "categoryCount": 2},
        "is_doublet": {
            "dtype": "bool",
            "encoding": "nullable-boolean",
            "categoryCount": None,
        },
        "is_primary_data": {"dtype": "bool", "encoding": None, "categoryCount": None},
        "n_counts": {
            "dtype": "int32",
            "encoding": "nullable-integer",
            "categoryCount": None,
        },
        "score": {"dtype": "float64", "encoding": None, "categoryCount": None},
        "unsupported": {
            "dtype": "unsupported",
            "encoding": None,
            "categoryCount": None,
        },
    }


def test_column_summary_counts_labels_and_summarizes_numbers(h5):
    obs = _metadata(h5)
    names = ("cell_type", "is_doublet", "is_primary_data", "n_counts", "score")
    assert {name: build._column_summary(obs[name]) for name in names} == {
        "cell_type": {
            "uniqueCount": 2,
            "topValues": [
                {"value": "T cell", "count": 2},
                {"value": "B cell", "count": 1},
            ],
            "missing": 1,
        },
        "is_doublet": {
            "uniqueCount": 2,
            "topValues": [{"value": True, "count": 2}, {"value": False, "count": 1}],
            "missing": 1,
        },
        "is_primary_data": {
            "uniqueCount": 2,
            "topValues": [{"value": True, "count": 3}, {"value": False, "count": 1}],
            "missing": 0,
        },
        "n_counts": {"min": 3.0, "max": 7.0, "mean": 5.0, "missing": 1},
        "score": {"min": 1.0, "max": 3.0, "mean": 2.0, "missing": 2},
    }


def test_column_summary_keeps_the_fifty_most_common_values(h5):
    barcodes = [f"cell{i:02d}".encode() for i in range(60)]
    h5.create_dataset("barcode", data=np.array(barcodes))
    summary = build._column_summary(h5["barcode"])
    assert summary["uniqueCount"] == 60
    assert summary["missing"] == 0
    values = [item["value"] for item in summary["topValues"]]
    assert values == [f"cell{i:02d}" for i in range(50)]


def test_column_summary_without_finite_numbers(h5):
    h5.create_dataset("score", data=np.full(3, np.nan))
    assert build._column_summary(h5["score"]) == {
        "min": None,
        "max": None,
        "mean": None,
        "missing": 3,
    }


def test_column_summary_counts_numeric_categories_as_labels(h5):
    batch = h5.create_group("batch")
    batch.create_dataset("codes", data=np.array([0, 1, 1], dtype=np.int8))
    batch.create_dataset("categories", data=np.array([10, 20]))
    assert build._column_summary(batch) == {
        "uniqueCount": 2,
        "topValues": [{"value": 20, "count": 2}, {"value": 10, "count": 1}],
        "missing": 0,
    }


@pytest.mark.parametrize("code", [-2, 2], ids=["below-missing", "past-categories"])
def test_column_values_rejects_invalid_categorical_codes(h5, code):
    column = h5.create_group("obs").create_group("donor")
    column.create_dataset("codes", data=np.array([0, code], dtype=np.int8))
    column.create_dataset("categories", data=np.array([b"D1", b"D2"]))
    with pytest.raises(ValueError, match="Invalid categorical codes in /obs/donor"):
        build._column_values(column)


def test_column_summary_rejects_unsupported_groups(h5):
    obs = _metadata(h5)
    with pytest.raises(
        ValueError, match="Unsupported metadata column encoding: /obs/unsupported"
    ):
        build._column_summary(obs["unsupported"])


def test_inspect_file_selects_counts_and_exports_metadata(tmp_path):
    path = write_h5ad(tmp_path / "source.h5ad")
    with h5py.File(path, "r+") as h5:
        joinids = np.array([f"j{i}".encode() for i in range(6)])
        h5["obs"].create_dataset("observation_joinid", data=joinids)
    events, progress = _recorder()

    result = build.inspect_file(path, progress=progress)

    json.dumps(result, allow_nan=False)  # Records store the inspection as JSON.
    manifest = result["manifest"]
    diagnostics = manifest.pop("selectionDiagnostics")
    assert manifest == {
        "title": "Healthy lung scRNA-seq atlas",
        "citation": CITATION,
        "schemaVersion": "5.3.0",
        "organism": None,
        "nObs": 6,
        "nVars": 5,
        "countsLocation": "X",
        "apiRawDataLocation": None,
        "countsSelectionSource": "inspection",
        "countsDtype": "int32",
        "countsIntegerLike": True,
        "countsMax": None,
        "countsSampleMax": 7,
        "countsValidationMode": "sampled_rows",
        "isPrimaryDataCounts": {"true": 4, "false": 2},
        "featureIdKey": "_index",
        "featureNameKey": "feature_name",
        "featureAttrsKey": "var",
        "selectionNeedsInput": None,
        "embeddings": ["X_umap"],
    }
    assert diagnostics["X"]["validCounts"] is True
    assert diagnostics["raw/X"]["reasons"] == ["Matrix is absent"]
    keys = result["h5ad_keys"]
    assert keys["matrices"] == {"X": diagnostics["X"]}
    assert keys["apiRawDataLocation"] is None
    assert keys["scarfInspection"]["matrixKey"] == "X"
    assert keys["scarfInspection"]["featureNameKey"] == "feature_name"
    assert "h5adFn" not in keys["scarfInspection"]
    assert keys["obs"]["cell_type"] == {
        "dtype": "|S8",
        "encoding": "categorical",
        "categoryCount": 3,
    }
    assert "observation_joinid" in keys["obs"]
    assert list(keys["var"]) == ["_index", "feature_name"]
    assert "raw/var" not in keys
    assert {name: keys[name] for name in ("obsm", "varm", "obsp", "uns")} == {
        "obsm": ["X_umap"],
        "varm": [],
        "obsp": [],
        "uns": ["citation", "schema_version", "title"],
    }
    summary = result["obs_summary"]
    # Summaries skip the per-cell join IDs that the column listing still reports.
    assert sorted(summary) == [
        "_index",
        "cell_type",
        "donor_id",
        "is_primary_data",
        "n_genes",
    ]
    assert summary["cell_type"]["topValues"] == [
        {"value": "T cell", "count": 2},
        {"value": "B cell", "count": 2},
        {"value": "monocyte", "count": 2},
    ]
    assert summary["n_genes"] == {"min": 0.0, "max": 5.0, "mean": 2.5, "missing": 0}
    assert result["uns"] == {
        "citation": CITATION,
        "schema_version": "5.3.0",
        "title": "Healthy lung scRNA-seq atlas",
    }
    stages = [stage for stage, _ in events]
    assert stages == ["validating_counts"] * 7 + ["inspecting_metadata"] * 2
    started, finished = events[-2:]
    assert started[1] == {"message": METADATA_MESSAGE}
    assert finished[1] == {}


@pytest.mark.parametrize(
    ("h5ad", "raw_data_location", "location", "not_evaluated"),
    [
        pytest.param({"raw_counts": COUNTS}, None, "raw/X", ["X"], id="prefers-raw"),
        pytest.param({"raw_counts": NORMALIZED}, None, "X", [], id="skips-invalid-raw"),
        pytest.param(
            {"counts": NORMALIZED, "layers": {"counts": COUNTS}},
            None,
            "layers/counts",
            [],
            id="counts-layer",
        ),
        pytest.param(
            {"counts": NORMALIZED, "layers": {"raw_counts": COUNTS}},
            None,
            "layers/raw_counts",
            [],
            id="raw-counts-layer",
        ),
        pytest.param({"raw_counts": COUNTS}, "X", "X", ["raw/X"], id="declared-x"),
        pytest.param(
            {"raw_counts": COUNTS}, "raw.X", "raw/X", ["X"], id="declared-raw"
        ),
    ],
)
def test_inspect_file_selects_one_counts_matrix(
    tmp_path, h5ad, raw_data_location, location, not_evaluated
):
    path = write_h5ad(tmp_path / "source.h5ad", **h5ad)
    result = build.inspect_file(path, raw_data_location)
    manifest = result["manifest"]
    assert manifest["countsLocation"] == location
    assert manifest["selectionNeedsInput"] is None
    assert manifest["apiRawDataLocation"] == raw_data_location
    assert manifest["countsSelectionSource"] == (
        "inspection" if raw_data_location is None else "curation_api"
    )
    assert manifest["featureAttrsKey"] == ("raw/var" if location == "raw/X" else "var")
    assert result["h5ad_keys"]["scarfInspection"]["matrixKey"] == location
    diagnostics = manifest["selectionDiagnostics"]
    assert diagnostics[location]["validCounts"] is True
    for key in not_evaluated:
        assert diagnostics[key]["reasons"] == ["Not evaluated"]


@pytest.mark.parametrize(
    ("h5ad", "raw_data_location", "options", "question", "not_evaluated"),
    [
        pytest.param(
            {"counts": NORMALIZED, "layers": {"counts": COUNTS, "raw_counts": COUNTS}},
            None,
            ["layers/counts", "layers/raw_counts"],
            "Both counts layers pass the sampled counts check.",
            [],
            id="both-layers",
        ),
        pytest.param(
            {
                "counts": NORMALIZED,
                "raw_counts": NORMALIZED,
                "layers": {"counts": NORMALIZED},
            },
            None,
            ["raw/X", "X", "layers/counts"],
            "No supported candidate passes the sampled counts and dimension checks.",
            [],
            id="no-valid-candidate",
        ),
        pytest.param(
            {"encoding": "csc", "raw_counts": COUNTS},
            None,
            ["CSR H5AD", "Dense H5AD"],
            "Preferred matrix raw/X is unsupported: Sparse encoding 'csc_matrix'",
            ["X"],
            id="csc-raw",
        ),
        pytest.param(
            {"encoding": "csc"},
            None,
            ["CSR H5AD", "Dense H5AD"],
            "Preferred matrix X is unsupported: Sparse encoding 'csc_matrix'",
            [],
            id="csc-x",
        ),
        pytest.param(
            {"counts": NORMALIZED, "layers": {"counts": COUNTS}},
            "X",
            ["X"],
            "CELLxGENE specifies X for raw counts, but X cannot be used: "
            "Sampled rows include non-integer counts. Correct the source or metadata",
            ["layers/counts"],
            id="declared-x-invalid",
        ),
        pytest.param(
            {},
            "raw.X",
            ["raw/X"],
            "CELLxGENE specifies raw.X for raw counts, but raw/X cannot be used: "
            "Matrix is absent.",
            ["X"],
            id="declared-raw-absent",
        ),
        pytest.param(
            {"raw_counts": COUNTS},
            "layers/counts",
            ["X", "raw.X"],
            "CELLxGENE raw_data_location is 'layers/counts'; expected 'X' or 'raw.X'.",
            ["raw/X", "X"],
            id="declared-unsupported-location",
        ),
    ],
)
def test_inspect_file_asks_for_input_without_one_valid_matrix(
    tmp_path, h5ad, raw_data_location, options, question, not_evaluated
):
    path = write_h5ad(tmp_path / "source.h5ad", **h5ad)
    result = build.inspect_file(path, raw_data_location)
    manifest = result["manifest"]
    needs_input = manifest["selectionNeedsInput"]
    assert manifest["countsLocation"] == "none"
    assert needs_input["options"] == options
    assert question in needs_input["question"]
    assert needs_input["candidates"] == manifest["selectionDiagnostics"]
    for key in not_evaluated:
        assert manifest["selectionDiagnostics"][key]["reasons"] == ["Not evaluated"]
    assert manifest["featureAttrsKey"] is None
    assert result["h5ad_keys"]["scarfInspection"] is None


def test_inspect_file_reads_declared_raw_counts_with_raw_features(tmp_path):
    path = write_h5ad(tmp_path / "source.h5ad", raw_counts=RAW_COUNTS)
    result = build.inspect_file(path, "raw.X")
    manifest = result["manifest"]
    assert (manifest["nObs"], manifest["nVars"]) == (6, 7)
    assert manifest["featureAttrsKey"] == "raw/var"
    assert (manifest["featureIdKey"], manifest["featureNameKey"]) == (
        "_index",
        "feature_name",
    )
    keys = result["h5ad_keys"]
    assert keys["apiRawDataLocation"] == "raw.X"
    assert keys["scarfInspection"]["nFeatures"] == 7
    assert list(keys["raw/var"]) == ["_index", "feature_name"]


def test_inspect_file_describes_matrices_it_does_not_validate(tmp_path):
    path = write_h5ad(
        tmp_path / "source.h5ad",
        NORMALIZED,
        layers={"raw_counts": COUNTS, "normalized": NORMALIZED},
    )
    matrices = build.inspect_file(path)["h5ad_keys"]["matrices"]
    assert set(matrices) == {"X", "layers/normalized", "layers/raw_counts"}
    assert matrices["layers/normalized"] == {
        "encoding": "csr_matrix",
        "shape": [6, 5],
        "dtype": "float64",
        "nnz": 22,
    }
    assert matrices["layers/raw_counts"]["validCounts"] is True


def test_inspect_file_without_selection_reports_table_dimensions(tmp_path):
    path = write_h5ad(tmp_path / "source.h5ad", NORMALIZED)
    result = build.inspect_file(path)
    manifest = result["manifest"]
    fields = (
        "countsLocation",
        "nObs",
        "nVars",
        "countsDtype",
        "countsIntegerLike",
        "countsSampleMax",
        "countsValidationMode",
        "featureAttrsKey",
        "featureIdKey",
        "featureNameKey",
        "isPrimaryDataCounts",
    )
    assert {field: manifest[field] for field in fields} == {
        "countsLocation": "none",
        "nObs": 6,
        "nVars": 5,
        "countsDtype": None,
        "countsIntegerLike": None,
        "countsSampleMax": None,
        "countsValidationMode": None,
        "featureAttrsKey": None,
        "featureIdKey": "_index",
        "featureNameKey": "_index",
        "isPrimaryDataCounts": {"true": 4, "false": 2},
    }
    assert result["h5ad_keys"]["scarfInspection"] is None


def test_inspect_file_without_usable_tables_reports_zero_dimensions(tmp_path):
    path = write_h5ad(
        tmp_path / "source.h5ad", annotations=False, umap=False, uns=False
    )
    with h5py.File(path, "r+") as h5:
        del h5["obs/_index"]
        del h5["var"]
    result = build.inspect_file(path)
    manifest = result["manifest"]
    assert manifest["selectionDiagnostics"]["X"]["reasons"] == [
        "Matrix shape [6, 5] does not match obs and var dimensions [None, None]"
    ]
    assert (manifest["nObs"], manifest["nVars"]) == (0, 0)
    assert (manifest["featureIdKey"], manifest["featureNameKey"]) == (
        "_index",
        "_index",
    )
    assert manifest["isPrimaryDataCounts"] == {"true": 0, "false": 0}
    assert manifest["embeddings"] == []
    assert manifest["title"] is None
    assert result["uns"] == {}
    assert result["obs_summary"] == {}
    assert "var" not in result["h5ad_keys"]


def test_inspect_file_asks_for_input_when_scarf_picks_another_feature_table(
    tmp_path, monkeypatch
):
    path = write_h5ad(tmp_path / "source.h5ad")
    inspect_h5ad = build.inspect_h5ad

    def other_feature_table(h5ad_fn: str, *, matrix_key: str):
        inspection = inspect_h5ad(h5ad_fn, matrix_key=matrix_key)
        return dataclasses.replace(inspection, featureAttrsKey="raw/var")

    monkeypatch.setattr(build, "inspect_h5ad", other_feature_table)
    result = build.inspect_file(path)
    manifest = result["manifest"]
    assert manifest["selectionNeedsInput"]["question"] == (
        "Scarf cannot read selected counts X: Scarf selected raw/var instead of var. "
        "Confirm the matrix and feature metadata before conversion."
    )
    assert manifest["selectionNeedsInput"]["options"] == ["X"]
    assert manifest["countsLocation"] == "X"
    assert (manifest["nObs"], manifest["nVars"]) == (6, 5)
    assert (manifest["featureIdKey"], manifest["featureNameKey"]) == (
        "_index",
        "_index",
    )
    assert result["h5ad_keys"]["scarfInspection"] is None


def test_inspect_file_asks_for_input_for_short_feature_names(tmp_path):
    path = write_h5ad(tmp_path / "source.h5ad")
    with h5py.File(path, "r+") as h5:
        del h5["var/feature_name"]
        h5["var"].create_dataset("feature_name", data=np.array([b"CD3E", b"LYZ"]))
    manifest = build.inspect_file(path)["manifest"]
    needs_input = manifest["selectionNeedsInput"]
    assert needs_input["question"] == (
        "Scarf cannot read selected counts X: Feature-name column 'feature_name' has "
        "unsupported dimensions. Confirm the matrix and feature metadata before "
        "conversion."
    )
    assert needs_input["options"] == ["X"]


def test_inspect_file_asks_for_input_without_feature_ids(tmp_path):
    path = write_h5ad(tmp_path / "source.h5ad")
    with h5py.File(path, "r+") as h5:
        del h5["var"]
        h5.create_group("var").create_dataset("n_cells", data=np.arange(5))
    manifest = build.inspect_file(path)["manifest"]
    needs_input = manifest["selectionNeedsInput"]
    assert needs_input["question"].startswith("Scarf cannot read selected counts X:")
    assert needs_input["options"] == ["X"]
    assert manifest["countsLocation"] == "X"


def test_inspect_file_uses_scarf_feature_names_without_feature_name(tmp_path):
    path = write_h5ad(tmp_path / "source.h5ad")
    with h5py.File(path, "r+") as h5:
        h5["var"].move("feature_name", "gene_symbols")
    manifest = build.inspect_file(path)["manifest"]
    assert manifest["selectionNeedsInput"] is None
    assert (manifest["featureIdKey"], manifest["featureNameKey"]) == (
        "_index",
        "gene_symbols",
    )


def test_needs_input_marks_a_copy_of_the_record():
    record = {"cytebaseId": "lung", "status": "processing"}
    assert build._needs_input(record, "Which matrix?", ["X", "raw/X"]) == {
        "cytebaseId": "lung",
        "status": "needsInput",
        "needsInput": {"question": "Which matrix?", "options": ["X", "raw/X"]},
    }
    assert record == {"cytebaseId": "lung", "status": "processing"}


@pytest.mark.parametrize(
    ("h5ad", "raw_data_location", "location"),
    [
        pytest.param({}, None, "X", id="inspected-x"),
        pytest.param({"raw_counts": RAW_COUNTS}, "raw.X", "raw/X", id="declared-raw"),
        pytest.param({"counts": NORMALIZED}, None, "none", id="needs-input"),
        pytest.param({"counts": NORMALIZED}, "X", "none", id="declared-needs-input"),
    ],
)
def test_validated_manifest_accepts_current_inspections(
    tmp_path, h5ad, raw_data_location, location
):
    source = write_h5ad(tmp_path / "source.h5ad", **h5ad)
    manifest = full_manifest(source, raw_data_location)
    manifest["collectionId"] = manifest["collectionId"].upper()
    result = build._validated_manifest(manifest)
    assert result["countsLocation"] == location
    assert result["collectionId"] == COLLECTION_ID
    assert result == Manifest.model_validate(manifest).model_dump(mode="json")


def _update(**fields: Any) -> Callable[[dict], None]:
    return lambda manifest: manifest.update(fields)


def _update_diagnostics(**fields: Any) -> Callable[[dict], None]:
    return lambda manifest: manifest["selectionDiagnostics"]["X"].update(fields)


def _drop(*keys: str) -> Callable[[dict], None]:
    def mutate(manifest: dict) -> None:
        for key in keys:
            del manifest[key]

    return mutate


def _select_none_with_passing_diagnostics(manifest: dict) -> None:
    diagnostics = manifest["selectionDiagnostics"]
    diagnostics["none"] = copy.deepcopy(diagnostics["X"])
    manifest["countsLocation"] = "none"


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        pytest.param(
            _drop("countsValidationMode", "apiRawDataLocation"),
            "Conversion requires a current inspection manifest; missing "
            "apiRawDataLocation, countsValidationMode. Inspect the source",
            id="missing-inspection-fields",
        ),
        pytest.param(
            _update(countsSelectionSource="curation_api"),
            "Counts selection provenance disagrees with apiRawDataLocation",
            id="curation-source-without-api-location",
        ),
        pytest.param(
            _update(apiRawDataLocation="X"),
            "Counts selection provenance disagrees with apiRawDataLocation",
            id="api-location-with-inspection-source",
        ),
        pytest.param(
            _update(apiRawDataLocation="raw.X", countsSelectionSource="curation_api"),
            "Selected counts disagree with apiRawDataLocation",
            id="api-location-disagrees",
        ),
        pytest.param(
            _update(
                apiRawDataLocation="layers/counts", countsSelectionSource="curation_api"
            ),
            "Selected counts disagree with apiRawDataLocation",
            id="unsupported-api-location",
        ),
        pytest.param(
            _update(selectionNeedsInput={"options": ["X"]}),
            NEEDS_INPUT_SHAPE,
            id="needs-input-without-question",
        ),
        pytest.param(
            _update(selectionNeedsInput={"question": "Which?", "options": "X"}),
            NEEDS_INPUT_SHAPE,
            id="needs-input-options-not-a-list",
        ),
        pytest.param(
            _update_diagnostics(validationComplete=False),
            SAMPLED_CHECK,
            id="validation-incomplete",
        ),
        pytest.param(
            _update_diagnostics(validCounts=False), SAMPLED_CHECK, id="invalid-counts"
        ),
        pytest.param(
            _update_diagnostics(formatSupported=None),
            SAMPLED_CHECK,
            id="format-unchecked",
        ),
        pytest.param(
            _update_diagnostics(validationMode="full_matrix"),
            SAMPLED_CHECK,
            id="other-diagnostic-mode",
        ),
        pytest.param(
            _update(countsValidationMode=None), SAMPLED_CHECK, id="manifest-mode-unset"
        ),
        pytest.param(
            _update(countsLocation="layers/counts"),
            SAMPLED_CHECK,
            id="selection-without-diagnostics",
        ),
        pytest.param(
            _update(countsLocation="none"),
            SAMPLED_CHECK,
            id="no-selection-without-needs-input",
        ),
        pytest.param(_update(nObs=7), DISAGREEMENT, id="cell-count"),
        pytest.param(
            _update(featureAttrsKey="raw/var"),
            DISAGREEMENT,
            id="manifest-feature-table",
        ),
        pytest.param(
            _update_diagnostics(featureAttrsKey="raw/var"),
            DISAGREEMENT,
            id="diagnostic-feature-table",
        ),
        pytest.param(
            _select_none_with_passing_diagnostics,
            DISAGREEMENT,
            id="none-with-passing-diagnostics",
        ),
    ],
)
def test_validated_manifest_rejects_stale_or_inconsistent_inspections(
    manifest, mutate, message
):
    mutate(manifest)
    with pytest.raises(ValueError, match=re.escape(message)):
        build._validated_manifest(manifest)


@pytest.mark.parametrize(
    "fields",
    [
        {"countsLocation": "layers/normalized"},
        {"nObs": "six"},
        {"countsValidationMode": "full_matrix"},
    ],
    ids=["unknown-location", "non-integer-cells", "unknown-mode"],
)
def test_validated_manifest_rejects_invalid_manifest_fields(manifest, fields):
    manifest.update(fields)
    with pytest.raises(ValidationError, match=next(iter(fields))):
        build._validated_manifest(manifest)
