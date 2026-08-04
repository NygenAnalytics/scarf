"""Tests for scarf.agent.ingest."""

from pathlib import Path

import h5py
import numpy as np
import pytest
from scipy.sparse import csr_matrix

from scarf.agent import detect_format, ingest
from scarf.agent.types import Decision
from scarf.readers import inspect_h5ad


def _write_sparse_group(
    h5: h5py.File | h5py.Group, key: str, values: np.ndarray
) -> None:
    matrix = csr_matrix(values)
    group = h5.create_group(key)
    group.attrs["encoding-type"] = "csr_matrix"
    group.attrs["shape"] = values.shape
    group.create_dataset("data", data=matrix.data)
    group.create_dataset("indices", data=matrix.indices)
    group.create_dataset("indptr", data=matrix.indptr)


def _write_h5ad(
    path: Path,
    values: np.ndarray,
    *,
    feature_types: list[bytes] | None = None,
    feature_names: list[bytes] | None = None,
    raw_values: np.ndarray | None = None,
) -> None:
    n_cells, n_feats = values.shape
    with h5py.File(path, mode="w") as h5:
        _write_sparse_group(h5, "X", values)
        if raw_values is not None:
            _write_sparse_group(h5, "raw/X", raw_values)
            raw_var = h5.create_group("raw/var")
            raw_n = raw_values.shape[1]
            raw_var.create_dataset(
                "_index",
                data=np.array([f"rf{i}".encode() for i in range(raw_n)]),
            )
            raw_var.create_dataset(
                "feature_name",
                data=np.array(
                    feature_names
                    if feature_names is not None
                    else [f"g{i}".encode() for i in range(raw_n)]
                ),
            )
            if feature_types is not None:
                raw_var.create_dataset("feature_types", data=np.array(feature_types))

        obs = h5.create_group("obs")
        obs.create_dataset(
            "_index",
            data=np.array([f"c{i}".encode() for i in range(n_cells)]),
        )
        var = h5.create_group("var")
        var.create_dataset(
            "_index",
            data=np.array([f"f{i}".encode() for i in range(n_feats)]),
        )
        var.create_dataset(
            "feature_name",
            data=np.array(
                feature_names
                if feature_names is not None and raw_values is None
                else [f"g{i}".encode() for i in range(n_feats)]
            ),
        )
        if feature_types is not None and raw_values is None:
            var.create_dataset("feature_types", data=np.array(feature_types))


def test_detect_format_by_suffix(tmp_path: Path) -> None:
    assert detect_format(tmp_path / "a.h5ad") == "h5ad"
    assert detect_format(tmp_path / "a.loom") == "loom"
    assert detect_format(tmp_path / "a.rds") == "seurat"
    assert detect_format(tmp_path / "a.csv") == "csv"
    assert detect_format(tmp_path / "a.zarr") == "zarr"


def test_ingest_h5ad_prefers_raw_integer_matrix(tmp_path: Path) -> None:
    path = tmp_path / "counts.h5ad"
    _write_h5ad(
        path,
        np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
        raw_values=np.array([[1, 0, 3], [0, 2, 4]], dtype=np.uint16),
        feature_types=[b"Gene Expression", b"Gene Expression", b"Gene Expression"],
        feature_names=[b"g1", b"g2", b"g3"],
    )
    inspection = inspect_h5ad(str(path))
    assert inspection.matrixKey == "raw/X"
    assert inspection.integerLike is True

    result = ingest(path=path, zarrPath=tmp_path / "out.zarr")
    assert result.status == "done"
    assert result.format == "h5ad"
    assert result.zarrPath is not None
    assert "RNA" in result.assayNames
    assert result.summary is not None
    assert result.acceptedActions
    assert result.acceptedActions[-1]["op"] == "DataStore"


def test_ingest_h5ad_stops_on_prenormalized_only(tmp_path: Path) -> None:
    path = tmp_path / "prenorm.h5ad"
    _write_h5ad(
        path,
        np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
    )
    result = ingest(path=path, zarrPath=tmp_path / "out.zarr")
    assert result.status == "needsInput"
    assert result.needsInput is not None
    assert (
        "integer-like" in result.needsInput.question.lower()
        or "raw" in result.needsInput.question.lower()
    )
    assert "matrixKey" in result.needsInput.question


def test_ingest_h5ad_force_matrix_key_allows_prenorm(tmp_path: Path) -> None:
    path = tmp_path / "prenorm.h5ad"
    _write_h5ad(
        path,
        np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
    )
    result = ingest(
        path=path,
        zarrPath=tmp_path / "forced.zarr",
        directions={"matrixKey": "X"},
    )
    assert result.status == "done"
    assert result.format == "h5ad"
    assert "RNA" in result.assayNames
    assert any("Forced matrix X" in note for note in result.notes)


def test_ingest_h5ad_hto_digit_names_need_input(tmp_path: Path) -> None:
    path = tmp_path / "hto_digits.h5ad"
    _write_h5ad(
        path,
        np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint16),
        feature_types=[b"Gene Expression", b"Antibody Capture", b"Antibody Capture"],
        feature_names=[b"GENE1", b"HTO1", b"HTO2"],
    )
    result = ingest(path=path, zarrPath=tmp_path / "out.zarr")
    assert result.status == "needsInput"
    assert result.needsInput is not None
    assert set(result.needsInput.options) == {"ADT", "HTO"}


def test_antibody_names_look_like_hto_digit_suffix() -> None:
    from scarf.agent.ingest.common import antibody_names_look_like_hto

    assert antibody_names_look_like_hto(["HTO1", "HTO2"])
    assert antibody_names_look_like_hto(["Hashtag1"])
    assert not antibody_names_look_like_hto(["CD3", "CD19"])


def test_ingest_h5ad_ambiguous_hto_without_model_needs_input(tmp_path: Path) -> None:
    path = tmp_path / "hto.h5ad"
    _write_h5ad(
        path,
        np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint16),
        feature_types=[b"Gene Expression", b"Antibody Capture", b"Antibody Capture"],
        feature_names=[b"GENE1", b"Hashtag1", b"TotalSeq-Hashtag2"],
    )
    result = ingest(path=path, zarrPath=tmp_path / "out.zarr")
    assert result.status == "needsInput"
    assert result.needsInput is not None
    assert set(result.needsInput.options) == {"ADT", "HTO"}


def test_ingest_h5ad_modality_choice_via_directions(tmp_path: Path) -> None:
    path = tmp_path / "hto.h5ad"
    _write_h5ad(
        path,
        np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint16),
        feature_types=[b"Gene Expression", b"Antibody Capture", b"Antibody Capture"],
        feature_names=[b"GENE1", b"Hashtag1", b"TotalSeq-Hashtag2"],
    )
    result = ingest(
        path=path,
        zarrPath=tmp_path / "out.zarr",
        directions={"modalityChoice": "HTO"},
    )
    assert result.status == "done"
    assert "HTO" in result.assayNames
    assert "ADT" not in result.assayNames


def test_ingest_h5ad_modality_choice_via_function_model(tmp_path: Path) -> None:
    from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
    from pydantic_ai.models.function import AgentInfo, FunctionModel

    path = tmp_path / "hto.h5ad"
    _write_h5ad(
        path,
        np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint16),
        feature_types=[b"Gene Expression", b"Antibody Capture", b"Antibody Capture"],
        feature_names=[b"GENE1", b"Hashtag1", b"TotalSeq-Hashtag2"],
    )

    expected = Decision(
        selectedId="modality:HTO",
        rationale="hashtag names",
        evidenceIds=["modality:HTO"],
    )

    def reply(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        tool = info.output_tools[0]
        return ModelResponse(
            parts=[ToolCallPart(tool_name=tool.name, args=expected.model_dump())]
        )

    result = ingest(
        path=path,
        zarrPath=tmp_path / "out.zarr",
        model=FunctionModel(reply),
    )
    assert result.status == "done"
    assert result.decision is not None
    assert result.decision.selectedId == "modality:HTO"
    assert "HTO" in result.assayNames


def test_ingest_csv_needs_input(tmp_path: Path) -> None:
    path = tmp_path / "table.csv"
    path.write_text("a,b\n1,2\n", encoding="utf-8")
    result = ingest(path=path, zarrPath=tmp_path / "out.zarr")
    assert result.status == "needsInput"
    assert result.format == "csv"


def test_ingest_10x_h5(tmp_path: Path) -> None:
    from tests import full_path

    fixture = Path(full_path("1K_pbmc_citeseq.h5"))
    if not fixture.is_file():
        pytest.skip("10x H5 fixture not downloaded")
    result = ingest(path=fixture, zarrPath=tmp_path / "pbmc.zarr")
    assert result.status == "done"
    assert result.format == "10x_h5"
    assert "RNA" in result.assayNames
    assert result.acceptedActions
    assert result.acceptedActions[-1]["op"] == "DataStore"
