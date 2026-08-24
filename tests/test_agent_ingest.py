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


def _patch_ingest_summary(
    monkeypatch: pytest.MonkeyPatch,
    *,
    assay_name: str = "RNA",
) -> None:
    import importlib

    ingest_common = importlib.import_module("scarf.agent.ingest.common")

    def summarize(
        _zarr_path: str,
        *,
        default_assay: str | None = None,
    ) -> tuple[list[str], str, dict[str, object]]:
        resolved = default_assay or assay_name
        return (
            [assay_name],
            resolved,
            {"default_assay": resolved, "total_cells": 2},
        )

    monkeypatch.setattr(ingest_common, "open_summary", summarize)


def test_detect_format_by_suffix(tmp_path: Path) -> None:
    assert detect_format(tmp_path / "a.h5ad") == "h5ad"
    assert detect_format(tmp_path / "a.loom") == "loom"
    assert detect_format(tmp_path / "a.rds") == "seurat"
    assert detect_format(tmp_path / "a.csv") == "csv"
    assert detect_format(tmp_path / "a.zarr") == "zarr"


def test_detect_format_directory_layout_branches(tmp_path: Path) -> None:
    zarr_path = tmp_path / "store"
    zarr_path.mkdir()
    (zarr_path / "zarr.json").write_text("{}", encoding="utf-8")
    for name in ("matrix.mtx", "barcodes.tsv", "features.tsv"):
        (zarr_path / name).write_text("", encoding="utf-8")
    assert detect_format(zarr_path) == "zarr"

    tenx_path = tmp_path / "tenx"
    tenx_path.mkdir()
    for name in ("matrix.mtx.gz", "barcodes.tsv.gz", "features.tsv.gz"):
        (tenx_path / name).write_text("", encoding="utf-8")
    assert detect_format(tenx_path) == "10x_dir"

    generic_path = tmp_path / "matrix-market"
    generic_path.mkdir()
    assert detect_format(generic_path) == "mtx"


def test_detect_format_file_layout_branches(tmp_path: Path) -> None:
    h5_path = tmp_path / "counts.h5"
    h5_path.write_bytes(b"")
    assert detect_format(h5_path) == "10x_h5"

    mtx_path = tmp_path / "counts.mtx.gz"
    mtx_path.write_bytes(b"")
    assert detect_format(mtx_path) == "mtx"

    unknown_path = tmp_path / "counts.bin"
    unknown_path.write_bytes(b"")
    assert detect_format(unknown_path) == "unknown"
    assert detect_format(tmp_path / "missing.h5") == "unknown"


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


def test_ingest_missing_source_returns_structured_failure(tmp_path: Path) -> None:
    destination = tmp_path / "out.zarr"
    result = ingest(path=tmp_path / "missing.h5ad", zarrPath=destination)

    assert result.status == "failed"
    assert result.format is None
    assert result.actions == []
    assert result.acceptedActions == []
    assert not destination.exists()


def test_ingest_unknown_file_returns_structured_failure(tmp_path: Path) -> None:
    path = tmp_path / "counts.bin"
    path.write_bytes(b"unknown")
    destination = tmp_path / "out.zarr"
    result = ingest(path=path, zarrPath=destination)

    assert result.status == "failed"
    assert result.format == "unknown"
    assert result.actions == []
    assert result.acceptedActions == []
    assert not destination.exists()


def test_ingest_10x_h5(tmp_path: Path) -> None:
    from tests import full_path

    fixture = Path(full_path("1K_pbmc_citeseq.h5"))
    if not fixture.is_file():
        pytest.skip("10x H5 fixture not downloaded")
    result = ingest(path=fixture, zarrPath=tmp_path / "pbmc.zarr")
    assert result.status == "done", result.notes
    assert result.format == "10x_h5"
    assert "RNA" in result.assayNames
    assert result.acceptedActions
    assert result.acceptedActions[-1]["op"] == "DataStore"


def _file_snapshot(location: Path) -> dict[str, tuple[int, int]]:
    return {
        str(path.relative_to(location)): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in location.rglob("*")
        if path.is_file()
    }


def test_ingest_overwrite_required_before_reader(tmp_path: Path, monkeypatch) -> None:
    import importlib

    path = tmp_path / "counts.h5ad"
    values = np.array([[1, 0], [0, 2]], dtype=np.uint16)
    _write_h5ad(path, values)
    dest = tmp_path / "out.zarr"
    dest.mkdir()
    (dest / "sentinel").write_bytes(b"preserve me")
    before = _file_snapshot(dest)

    calls: list[str] = []

    def boom(*_args, **_kwargs):
        calls.append("inspect")
        raise AssertionError("inspect must not run when overwrite is missing")

    monkeypatch.setattr(
        importlib.import_module("scarf.readers._h5ad_inspect"),
        "inspect_h5ad",
        boom,
    )
    blocked = ingest(path=path, zarrPath=dest)
    assert blocked.status == "failed"
    assert calls == []
    assert 'directions={"overwrite": true}' in " ".join(blocked.notes)
    assert _file_snapshot(dest) == before


@pytest.mark.parametrize("bad_overwrite", [1, "true", "True", 0, "yes"])
def test_ingest_overwrite_rejects_non_boolean(
    tmp_path: Path, bad_overwrite: object
) -> None:
    path = tmp_path / "counts.h5ad"
    _write_h5ad(path, np.array([[1, 0], [0, 2]], dtype=np.uint16))
    dest = tmp_path / "out.zarr"
    result = ingest(
        path=path,
        zarrPath=dest,
        directions={"overwrite": bad_overwrite},
    )
    assert result.status == "failed"
    assert "overwrite must be boolean true" in " ".join(result.notes)


def test_ingest_overwrite_true_replaces_destination(tmp_path: Path) -> None:
    path = tmp_path / "counts.h5ad"
    _write_h5ad(path, np.array([[1, 0], [0, 2]], dtype=np.uint16))
    dest = tmp_path / "out.zarr"
    dest.mkdir()
    sentinel = dest / "sentinel"
    sentinel.write_bytes(b"remove me")
    second = ingest(
        path=path,
        zarrPath=dest,
        directions={"overwrite": True},
    )
    assert second.status == "done"
    convert = next(
        action for action in second.acceptedActions if action["op"] == "H5adToZarr"
    )
    assert convert["overwrite"] is True
    assert any("Overwrite authorized" in note for note in second.notes)
    assert not sentinel.exists()


def test_ingest_missing_destination_fails(tmp_path: Path) -> None:
    path = tmp_path / "counts.h5ad"
    _write_h5ad(path, np.array([[1, 0], [0, 2]], dtype=np.uint16))
    result = ingest(path=path)
    assert result.status == "failed"
    assert result.format == "h5ad"
    assert any("zarrPath is required" in note for note in result.notes)


def test_ingest_overlapping_source_destination_fails(tmp_path: Path) -> None:
    nested = tmp_path / "bundle"
    nested.mkdir()
    path = nested / "counts.h5ad"
    _write_h5ad(path, np.array([[1, 0], [0, 2]], dtype=np.uint16))
    result = ingest(path=path, zarrPath=nested)
    assert result.status == "failed"
    assert any("must not equal or nest" in note for note in result.notes)


def test_ingest_uri_probe_failure_fails_closed(tmp_path: Path, monkeypatch) -> None:
    import importlib

    ingest_common = importlib.import_module("scarf.agent.ingest.common")
    path = tmp_path / "counts.h5ad"
    _write_h5ad(path, np.array([[1, 0], [0, 2]], dtype=np.uint16))

    def boom(_location, **_kwargs):
        raise RuntimeError("probe unavailable")

    monkeypatch.setattr(ingest_common, "zarr_location_has_content", boom)
    result = ingest(path=path, zarrPath=tmp_path / "out.zarr")
    assert result.status == "failed"
    assert any(
        "Destination existence could not be verified" in note for note in result.notes
    )


def test_ingest_corrupt_h5ad_returns_structured_failure(tmp_path: Path) -> None:
    path = tmp_path / "broken.h5ad"
    path.write_bytes(b"not-an-h5ad")
    result = ingest(path=path, zarrPath=tmp_path / "out.zarr")
    assert result.status == "failed"
    assert result.format == "h5ad"
    assert any("inspect_h5ad" in note for note in result.notes)


def test_ingest_mtx_rejects_invalid_index(tmp_path: Path) -> None:
    from scarf.agent.ingest.mtx import _resolve_mtx_index

    for bad in (True, False, 1.5, -1, "x", object()):
        result = _resolve_mtx_index(bad, n_candidates=2)
        assert getattr(result, "status", None) == "failed"

    out_of_range = _resolve_mtx_index(3, n_candidates=2)
    assert out_of_range.status == "failed"
    assert any("out of range" in note for note in out_of_range.notes)


def test_ingest_mtx_selected_candidate_happy_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib

    readers = importlib.import_module("scarf.readers.mtx")
    writers = importlib.import_module("scarf.writers.cellranger")
    source = tmp_path / "matrix-market"
    source.mkdir()
    destination = tmp_path / "out.zarr"
    candidates = (object(), object())

    class FakeReader:
        def __init__(self, candidate: object) -> None:
            self.candidate = candidate
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    class FakeWriter:
        def __init__(self, reader: FakeReader, *, zarr_loc: str) -> None:
            self.reader = reader
            self.zarr_loc = zarr_loc
            self.dump_calls = 0

        def dump(self) -> None:
            self.dump_calls += 1
            Path(self.zarr_loc).mkdir()

    created_readers: list[FakeReader] = []
    created_writers: list[FakeWriter] = []

    def inspect(path: Path) -> tuple[object, object]:
        assert path == source
        return candidates

    def make_reader(candidate: object) -> FakeReader:
        reader = FakeReader(candidate)
        created_readers.append(reader)
        return reader

    def make_writer(reader: FakeReader, *, zarr_loc: str) -> FakeWriter:
        writer = FakeWriter(reader, zarr_loc=zarr_loc)
        created_writers.append(writer)
        return writer

    monkeypatch.setattr(readers, "inspect_mtx", inspect)
    monkeypatch.setattr(readers, "MtxReader", make_reader)
    monkeypatch.setattr(writers, "MtxToZarr", make_writer)
    _patch_ingest_summary(monkeypatch)

    result = ingest(
        path=source,
        zarrPath=destination,
        directions={"mtxIndex": 1},
    )

    assert result.status == "done"
    assert result.format == "mtx"
    assert result.assayNames == ["RNA"]
    assert len(created_readers) == 1
    assert created_readers[0].candidate is candidates[1]
    assert created_readers[0].close_calls == 1
    assert len(created_writers) == 1
    assert created_writers[0].dump_calls == 1
    assert result.acceptedActions[0]["op"] == "MtxToZarr"
    assert result.acceptedActions[0]["mtxIndex"] == 1
    assert result.acceptedActions[-1]["op"] == "DataStore"
    assert destination.is_dir()


def test_ingest_mtx_multiple_candidates_needs_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib

    readers = importlib.import_module("scarf.readers.mtx")
    writers = importlib.import_module("scarf.writers.cellranger")
    source = tmp_path / "matrix-market"
    source.mkdir()
    destination = tmp_path / "out.zarr"

    class UnexpectedAdapter:
        def __init__(self, *_args, **_kwargs) -> None:
            raise AssertionError("reader and writer must not run before MTX selection")

    monkeypatch.setattr(
        readers,
        "inspect_mtx",
        lambda _path: (object(), object()),
    )
    monkeypatch.setattr(readers, "MtxReader", UnexpectedAdapter)
    monkeypatch.setattr(writers, "MtxToZarr", UnexpectedAdapter)

    result = ingest(path=source, zarrPath=destination)

    assert result.status == "needsInput"
    assert result.format == "mtx"
    assert result.needsInput is not None
    assert result.needsInput.options == ["0", "1"]
    assert result.actions == []
    assert result.acceptedActions == []
    assert not destination.exists()


def test_ingest_mtx_inspection_failure_does_not_start_conversion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib

    readers = importlib.import_module("scarf.readers.mtx")
    writers = importlib.import_module("scarf.writers.cellranger")
    source = tmp_path / "matrix-market"
    source.mkdir()
    destination = tmp_path / "out.zarr"

    def fail_inspection(_path: Path) -> tuple[object, ...]:
        raise ValueError("invalid matrix layout")

    class UnexpectedAdapter:
        def __init__(self, *_args, **_kwargs) -> None:
            raise AssertionError("conversion must not run after inspection failure")

    monkeypatch.setattr(readers, "inspect_mtx", fail_inspection)
    monkeypatch.setattr(readers, "MtxReader", UnexpectedAdapter)
    monkeypatch.setattr(writers, "MtxToZarr", UnexpectedAdapter)

    result = ingest(path=source, zarrPath=destination)

    assert result.status == "failed"
    assert result.format == "mtx"
    assert result.zarrPath == str(destination)
    assert result.actions == []
    assert result.acceptedActions == []
    assert any("inspect_mtx" in note for note in result.notes)
    assert not destination.exists()


def test_ingest_mtx_conversion_failure_closes_reader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib

    readers = importlib.import_module("scarf.readers.mtx")
    writers = importlib.import_module("scarf.writers.cellranger")
    source = tmp_path / "matrix-market"
    source.mkdir()
    destination = tmp_path / "out.zarr"

    class FakeReader:
        def __init__(self, _candidate: object) -> None:
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    class FailingWriter:
        def __init__(self, _reader: FakeReader, *, zarr_loc: str) -> None:
            self.zarr_loc = zarr_loc

        def dump(self) -> None:
            Path(self.zarr_loc).mkdir()
            raise ValueError("invalid matrix values")

    created_readers: list[FakeReader] = []

    def make_reader(candidate: object) -> FakeReader:
        reader = FakeReader(candidate)
        created_readers.append(reader)
        return reader

    monkeypatch.setattr(readers, "inspect_mtx", lambda _path: (object(),))
    monkeypatch.setattr(readers, "MtxReader", make_reader)
    monkeypatch.setattr(writers, "MtxToZarr", FailingWriter)

    result = ingest(path=source, zarrPath=destination)

    assert result.status == "failed"
    assert result.format == "mtx"
    assert result.zarrPath == str(destination)
    assert result.actions == []
    assert result.acceptedActions == []
    assert len(created_readers) == 1
    assert created_readers[0].close_calls == 1
    assert any("partial store" in note for note in result.notes)
    assert destination.is_dir()


def test_ingest_seurat_success_closes_reader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib

    readers = importlib.import_module("scarf.readers.seurat")
    writers = importlib.import_module("scarf.writers.seurat")
    source = tmp_path / "object.rds"
    source.write_bytes(b"stub")
    destination = tmp_path / "out.zarr"

    class FakeReader:
        def __init__(self, path: str) -> None:
            self.path = path
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    class FakeWriter:
        def __init__(self, reader: FakeReader, *, zarr_loc: str) -> None:
            self.reader = reader
            self.zarr_loc = zarr_loc
            self.dump_calls = 0

        def dump(self) -> None:
            self.dump_calls += 1
            Path(self.zarr_loc).mkdir()

    created_readers: list[FakeReader] = []
    created_writers: list[FakeWriter] = []

    def make_reader(path: str) -> FakeReader:
        reader = FakeReader(path)
        created_readers.append(reader)
        return reader

    def make_writer(reader: FakeReader, *, zarr_loc: str) -> FakeWriter:
        writer = FakeWriter(reader, zarr_loc=zarr_loc)
        created_writers.append(writer)
        return writer

    monkeypatch.setattr(readers, "SeuratReader", make_reader)
    monkeypatch.setattr(writers, "SeuratToZarr", make_writer)
    _patch_ingest_summary(monkeypatch)

    result = ingest(path=source, zarrPath=destination)

    assert result.status == "done"
    assert result.format == "seurat"
    assert len(created_readers) == 1
    assert created_readers[0].path == str(source)
    assert created_readers[0].close_calls == 1
    assert len(created_writers) == 1
    assert created_writers[0].reader is created_readers[0]
    assert created_writers[0].dump_calls == 1
    assert result.acceptedActions[0]["op"] == "SeuratToZarr"
    assert result.acceptedActions[-1]["op"] == "DataStore"
    assert destination.is_dir()


def test_ingest_seurat_failure_closes_reader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib

    readers = importlib.import_module("scarf.readers.seurat")
    writers = importlib.import_module("scarf.writers.seurat")
    source = tmp_path / "object.rds"
    source.write_bytes(b"stub")
    destination = tmp_path / "out.zarr"

    class FakeReader:
        def __init__(self, _path: str) -> None:
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    class FailingWriter:
        def __init__(self, _reader: FakeReader, *, zarr_loc: str) -> None:
            self.zarr_loc = zarr_loc

        def dump(self) -> None:
            Path(self.zarr_loc).mkdir()
            raise RuntimeError("conversion failed")

    created_readers: list[FakeReader] = []

    def make_reader(path: str) -> FakeReader:
        reader = FakeReader(path)
        created_readers.append(reader)
        return reader

    monkeypatch.setattr(readers, "SeuratReader", make_reader)
    monkeypatch.setattr(writers, "SeuratToZarr", FailingWriter)

    result = ingest(path=source, zarrPath=destination)

    assert result.status == "failed"
    assert result.format == "seurat"
    assert result.zarrPath == str(destination)
    assert result.actions == []
    assert result.acceptedActions == []
    assert len(created_readers) == 1
    assert created_readers[0].close_calls == 1
    assert any("partial store" in note for note in result.notes)
    assert destination.is_dir()


def test_ingest_loom_success_forwards_reader_options_and_closes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib

    readers = importlib.import_module("scarf.readers.loom")
    writers = importlib.import_module("scarf.writers.loom")
    source = tmp_path / "counts.loom"
    source.write_bytes(b"stub")
    destination = tmp_path / "out.zarr"

    class FakeHandle:
        def __init__(self) -> None:
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    class FakeReader:
        def __init__(self, path: str, **kwargs: str) -> None:
            self.path = path
            self.kwargs = kwargs
            self.h5 = FakeHandle()

    class FakeWriter:
        def __init__(
            self,
            reader: FakeReader,
            *,
            zarr_loc: str,
            assay_name: str,
        ) -> None:
            self.reader = reader
            self.zarr_loc = zarr_loc
            self.assay_name = assay_name
            self.dump_calls = 0

        def dump(self) -> None:
            self.dump_calls += 1
            Path(self.zarr_loc).mkdir()

    created_readers: list[FakeReader] = []
    created_writers: list[FakeWriter] = []

    def make_reader(path: str, **kwargs: str) -> FakeReader:
        reader = FakeReader(path, **kwargs)
        created_readers.append(reader)
        return reader

    def make_writer(
        reader: FakeReader,
        *,
        zarr_loc: str,
        assay_name: str,
    ) -> FakeWriter:
        writer = FakeWriter(
            reader,
            zarr_loc=zarr_loc,
            assay_name=assay_name,
        )
        created_writers.append(writer)
        return writer

    monkeypatch.setattr(readers, "LoomReader", make_reader)
    monkeypatch.setattr(writers, "LoomToZarr", make_writer)
    _patch_ingest_summary(monkeypatch, assay_name="ADT")

    result = ingest(
        path=source,
        zarrPath=destination,
        directions={
            "cellNamesKey": "cells",
            "featureNamesKey": "genes",
            "assayName": "ADT",
            "defaultAssay": "ADT",
        },
    )

    assert result.status == "done"
    assert result.format == "loom"
    assert result.assayNames == ["ADT"]
    assert len(created_readers) == 1
    assert created_readers[0].path == str(source)
    assert created_readers[0].kwargs == {
        "cell_names_key": "cells",
        "feature_names_key": "genes",
    }
    assert created_readers[0].h5.close_calls == 1
    assert len(created_writers) == 1
    assert created_writers[0].reader is created_readers[0]
    assert created_writers[0].assay_name == "ADT"
    assert created_writers[0].dump_calls == 1
    assert result.acceptedActions[0]["op"] == "LoomToZarr"
    assert result.acceptedActions[-1]["op"] == "DataStore"
    assert result.acceptedActions[-1]["defaultAssay"] == "ADT"
    assert destination.is_dir()


def test_ingest_loom_failure_closes_reader_handle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib

    readers = importlib.import_module("scarf.readers.loom")
    writers = importlib.import_module("scarf.writers.loom")
    source = tmp_path / "counts.loom"
    source.write_bytes(b"stub")
    destination = tmp_path / "out.zarr"

    class FakeHandle:
        def __init__(self) -> None:
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    class FakeReader:
        def __init__(self, _path: str, **_kwargs: str) -> None:
            self.h5 = FakeHandle()

    class FailingWriter:
        def __init__(
            self,
            _reader: FakeReader,
            *,
            zarr_loc: str,
            assay_name: str,
        ) -> None:
            del assay_name
            self.zarr_loc = zarr_loc

        def dump(self) -> None:
            Path(self.zarr_loc).mkdir()
            raise OSError("conversion failed")

    created_readers: list[FakeReader] = []

    def make_reader(path: str, **kwargs: str) -> FakeReader:
        reader = FakeReader(path, **kwargs)
        created_readers.append(reader)
        return reader

    monkeypatch.setattr(readers, "LoomReader", make_reader)
    monkeypatch.setattr(writers, "LoomToZarr", FailingWriter)

    result = ingest(path=source, zarrPath=destination)

    assert result.status == "failed"
    assert result.format == "loom"
    assert result.zarrPath == str(destination)
    assert result.actions == []
    assert result.acceptedActions == []
    assert len(created_readers) == 1
    assert created_readers[0].h5.close_calls == 1
    assert any("partial store" in note for note in result.notes)
    assert destination.is_dir()


def test_ingest_h5ad_post_write_failure_reports_partial(
    tmp_path: Path, monkeypatch
) -> None:
    import importlib

    ingest_common = importlib.import_module("scarf.agent.ingest.common")
    path = tmp_path / "counts.h5ad"
    _write_h5ad(path, np.array([[1, 0], [0, 2]], dtype=np.uint16))
    dest = tmp_path / "out.zarr"

    def boom(*_args, **_kwargs):
        raise OSError("open failed after write")

    monkeypatch.setattr(ingest_common, "open_summary", boom)
    result = ingest(path=path, zarrPath=dest)
    assert result.status == "failed"
    assert result.zarrPath == str(dest)
    assert any(
        "partial store" in note or "converted store" in note for note in result.notes
    )
    assert dest.exists()


def test_ingest_summary_programmer_error_surfaces(tmp_path: Path, monkeypatch) -> None:
    import importlib

    ingest_common = importlib.import_module("scarf.agent.ingest.common")

    def boom(*_args, **_kwargs):
        raise RuntimeError("unexpected summary defect")

    monkeypatch.setattr(ingest_common, "open_summary", boom)
    with pytest.raises(RuntimeError, match="unexpected summary defect"):
        ingest_common.finish(
            format_name="h5ad",
            zarr_path=str(tmp_path / "out.zarr"),
            notes=[],
            convert_actions=[],
            action_labels=[],
        )


@pytest.mark.parametrize(
    ("outcome", "assays"),
    [
        ("needsInput", ["RNA", "ADT"]),
        ("failed", ["RNA"]),
        ("memoryError", ["RNA"]),
    ],
)
def test_ingest_cellranger_closes_reader(
    tmp_path: Path,
    monkeypatch,
    outcome: str,
    assays: list[str],
) -> None:
    import importlib

    from scarf.agent.ingest.result import needs_input

    cellranger_mod = importlib.import_module("scarf.agent.ingest.cellranger")
    readers = importlib.import_module("scarf.readers.cellranger")
    writers = importlib.import_module("scarf.writers.cellranger")
    ingest_cellranger = cellranger_mod.ingest_cellranger

    class FakeReader:
        def __init__(self, *_args, **_kwargs):
            self.closed = False
            self.assayFeats = type("Feats", (), {"columns": assays})()

        def feature_names(self, assay: str) -> list[str]:
            del assay
            return ["HTO-1", "HTO-2"]

        def close(self) -> None:
            self.closed = True

    created: list[FakeReader] = []

    def make_reader(*_args, **_kwargs):
        reader = FakeReader()
        created.append(reader)
        return reader

    monkeypatch.setattr(readers, "CrH5Reader", make_reader)
    if outcome == "needsInput":
        monkeypatch.setattr(
            cellranger_mod,
            "resolve_modality_choice",
            lambda **_kwargs: (
                None,
                None,
                needs_input(
                    format_name="10x_h5",
                    question="choose",
                    options=["ADT", "HTO"],
                    evidence_ids=["modality:ADT", "modality:HTO"],
                    notes=["need choice"],
                ),
            ),
        )
    else:

        class BoomWriter:
            def __init__(self, *_args, **_kwargs):
                if outcome == "memoryError":
                    raise MemoryError("writer memory exhausted")
                raise ValueError("writer open failed")

        monkeypatch.setattr(
            writers,
            "CrToZarr",
            BoomWriter,
        )

    def run():
        return ingest_cellranger(
            tmp_path / "fake.h5",
            format_name="10x_h5",
            reader_class_name="CrH5Reader",
            zarrPath=tmp_path / "out.zarr",
            model=None,
            directions={},
            notes=[],
        )

    if outcome == "memoryError":
        with pytest.raises(MemoryError, match="writer memory exhausted"):
            run()
    else:
        result = run()
        assert result.status == outcome
    assert len(created) == 1
    assert created[0].closed is True


def test_ingest_invalid_modality_decision_needs_input(tmp_path: Path) -> None:
    from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
    from pydantic_ai.models.function import AgentInfo, FunctionModel

    from scarf.agent.types import Decision

    path = tmp_path / "hto.h5ad"
    values = np.array([[1, 0, 2], [0, 3, 1]], dtype=np.uint16)
    _write_h5ad(
        path,
        values,
        feature_types=[
            b"Gene Expression",
            b"Antibody Capture",
            b"Antibody Capture",
        ],
        feature_names=[b"g1", b"HTO-1", b"HTO-2"],
    )

    def reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        tool = info.output_tools[0]
        bad = Decision(
            selectedId="modality:WRONG",
            rationale="bad",
            evidenceIds=["modality:ADT"],
        )
        return ModelResponse(
            parts=[ToolCallPart(tool_name=tool.name, args=bad.model_dump())]
        )

    result = ingest(
        path=path,
        zarrPath=tmp_path / "out.zarr",
        model=FunctionModel(reply),
    )
    assert result.status == "needsInput"
    assert result.format == "h5ad"
    assert any("modalityChoice" in note for note in result.notes)


def test_ingest_existing_zarr_readonly_does_not_mutate(tmp_path: Path) -> None:
    from scarf.datastore.datastore import DataStore
    from scarf.writers import SparseToZarr

    location = tmp_path / "existing.zarr"
    writer = SparseToZarr(
        csr_matrix(np.array([[1, 0], [0, 2]], dtype=np.uint16)),
        str(location),
        cell_ids=["c0", "c1"],
        feature_ids=["g0", "g1"],
        mem_budget=64 * 1024 * 1024,
        nthreads=1,
    )
    writer.dump()
    # Raw writer output: ingest as existing zarr without DataStore init.
    before = _file_snapshot(location)
    result = ingest(path=location)
    assert result.status == "done"
    assert result.format == "zarr"
    assert result.acceptedActions[-1]["op"] == "summarizeZarr"
    assert result.acceptedActions[-1]["zarrMode"] == "r"
    assert _file_snapshot(location) == before

    # Initialized store also stays byte-stable and agrees with DataStore.summary.
    store = DataStore(
        str(location),
        default_assay="RNA",
        min_features_per_cell=0,
        min_cells_per_feature=0,
        nthreads=1,
        mem_budget=64 * 1024 * 1024,
    )
    ds_summary = store.summary().to_dict()
    after_init = _file_snapshot(location)
    again = ingest(path=location)
    assert again.status == "done"
    assert _file_snapshot(location) == after_init
    readonly = dict(again.summary)
    for key in ("zarr_mode", "resources"):
        ds_summary.pop(key, None)
        readonly.pop(key, None)
    assert readonly == ds_summary
    assert again.summary["default_assay"] == ds_summary["default_assay"]
    assert again.summary["total_cells"] == store.cells.N


def test_ingest_h5ad_still_initializes_qc(tmp_path: Path) -> None:
    path = tmp_path / "counts.h5ad"
    _write_h5ad(path, np.array([[1, 0], [0, 2], [3, 1]], dtype=np.uint16))
    dest = tmp_path / "out.zarr"
    result = ingest(path=path, zarrPath=dest)
    assert result.status == "done"
    assert result.acceptedActions[-1]["op"] == "DataStore"
    assert result.acceptedActions[-1]["zarrMode"] == "r+"
    from scarf.datastore.datastore import DataStore

    store = DataStore(str(dest), default_assay="RNA", nthreads=1)
    assert "RNA_nCounts" in store.cells.columns
    assert "I" in store.cells.columns
