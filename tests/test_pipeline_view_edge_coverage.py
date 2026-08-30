from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import scarf.datastore.pipeline_run as pipeline_run_module
from scarf.datastore.pipeline_run import (
    PipelineAxisView,
    PipelineExecutionError,
    PipelineRun,
)
from scarf.storage.errors import ArtifactResolutionError
from scarf.storage.pipeline_runs import (
    PipelineErrorRecord,
    PipelineFieldDescriptor,
    PipelineInterruptionRecord,
    create_pipeline_run_record,
    interrupt_pipeline_run_record,
)
from scarf.storage.refs import ArtifactRef
from tests.test_pipeline_run_foundation import _Owner, _completed_run, _root


class _Array:
    chunks = None

    def __init__(self, values: Any) -> None:
        self.values = np.asarray(values)
        self.shape = self.values.shape
        self.ndim = self.values.ndim
        self.dtype = self.values.dtype

    def __getitem__(self, key: Any) -> np.ndarray:
        return self.values[key]


def _descriptor(view: PipelineAxisView, key: str) -> PipelineFieldDescriptor:
    return view._descriptor_by_key[key]


def test_pipeline_run_and_view_constructor_guards() -> None:
    with pytest.raises(TypeError, match="run_id"):
        PipelineExecutionError("", "stage", ValueError("bad"))
    with pytest.raises(TypeError, match="stage"):
        PipelineExecutionError("run", "", ValueError("bad"))
    with pytest.raises(TypeError, match="cause"):
        PipelineExecutionError("run", "stage", "bad")  # type: ignore[arg-type]

    root = _root()
    run = _completed_run(root)
    with pytest.raises(TypeError, match="owner"):
        PipelineRun(object(), run._record)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="record"):
        PipelineRun(run._owner, object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="axis"):
        PipelineAxisView(run._owner, run._record, axis="rows")  # type: ignore[arg-type]

    failed = replace(
        run._record,
        label=None,
        status="failed",
        complete=True,
        outputs=(),
        fields=(),
        error=PipelineErrorRecord("ValueError", "bad"),
    )
    with pytest.raises(RuntimeError, match="not completed"):
        PipelineAxisView(run._owner, failed, axis="cells")
    failed_run = PipelineRun(run._owner, failed)
    for operation in (
        lambda: failed_run["out"],
        lambda: iter(failed_run),
        lambda: len(failed_run),
        lambda: failed_run.cells,
        lambda: failed_run.features,
    ):
        with pytest.raises(RuntimeError, match="requires a completed run"):
            operation()

    incomplete_fields = tuple(
        field
        for field in run._record.fields
        if not (field.axis == "cells" and field.key == "names")
    )
    incomplete = replace(run._record, fields=incomplete_fields)
    with pytest.raises(ArtifactResolutionError) as caught:
        PipelineAxisView(run._owner, incomplete, axis="cells")
    assert caught.value.code == "pipeline_view_required_fields_missing"


def test_pipeline_complete_group_and_source_array_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _completed_run(_root())
    cells = run.cells
    features = run.features
    cell_names = _descriptor(cells, "names")
    feature_names = _descriptor(features, "names")

    wrong_assay = replace(
        cell_names,
        artifact=ArtifactRef("assay", "metadata_snapshot", "e" * 64, assay="ADT"),
    )
    with pytest.raises(ArtifactResolutionError) as caught:
        cells._complete_group(wrong_assay)
    assert caught.value.code == "pipeline_field_axis_mismatch"
    wrong_scope = replace(
        feature_names,
        artifact=ArtifactRef("datastore", "metadata_snapshot", "f" * 64),
    )
    with pytest.raises(ArtifactResolutionError) as caught:
        features._complete_group(wrong_scope)
    assert caught.value.code == "pipeline_field_axis_mismatch"

    original_inspect = pipeline_run_module.inspect_artifact
    monkeypatch.setattr(
        pipeline_run_module,
        "inspect_artifact",
        lambda *_args: (_ for _ in ()).throw(ValueError("bad")),
    )
    with pytest.raises(ArtifactResolutionError) as caught:
        cells._complete_group(cell_names)
    assert caught.value.code == "pipeline_field_artifact_malformed"

    monkeypatch.setattr(
        pipeline_run_module,
        "inspect_artifact",
        lambda *_args: SimpleNamespace(exists=False, complete=False),
    )
    with pytest.raises(ArtifactResolutionError) as caught:
        cells._complete_group(cell_names)
    assert caught.value.code == "artifact_missing"
    monkeypatch.setattr(
        pipeline_run_module,
        "inspect_artifact",
        lambda *_args: SimpleNamespace(exists=True, complete=False),
    )
    with pytest.raises(ArtifactResolutionError) as caught:
        cells._complete_group(cell_names)
    assert caught.value.code == "artifact_incomplete"

    monkeypatch.setattr(pipeline_run_module, "inspect_artifact", original_inspect)
    absent = replace(cell_names, source_value="absent")
    with pytest.raises(ArtifactResolutionError) as caught:
        cells._source_array(absent)
    assert caught.value.code == "pipeline_field_payload_missing"
    with pytest.raises(ArtifactResolutionError) as caught:
        cells._source_array(cell_names, missing=True)
    assert caught.value.code == "pipeline_field_payload_missing"

    monkeypatch.setattr(
        pipeline_run_module,
        "as_zarr_array",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(TypeError("bad")),
    )
    with pytest.raises(ArtifactResolutionError) as caught:
        cells._source_array(cell_names)
    assert caught.value.code == "pipeline_field_payload_malformed"


def test_pipeline_array_shape_dtype_and_identity_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _completed_run(_root())
    cells = run.cells
    names = _descriptor(cells, "names")
    umap = _descriptor(cells, "umap_1")

    with pytest.raises(ArtifactResolutionError) as caught:
        cells._resolved_array_shape(names, _Array(np.ones((4, 2))))  # type: ignore[arg-type]
    assert caught.value.code == "pipeline_field_shape_mismatch"
    with pytest.raises(ArtifactResolutionError) as caught:
        cells._resolved_array_shape(
            replace(umap, value_index=5),
            _Array(np.ones((2, 2))),  # type: ignore[arg-type]
        )
    assert caught.value.code == "pipeline_field_shape_mismatch"
    assert cells._resolved_array_shape(
        umap,
        _Array(np.ones(2, dtype=bool)),  # type: ignore[arg-type]
        missing=True,
    ) == (2,)
    with pytest.raises(ArtifactResolutionError) as caught:
        cells._expected_dtype(replace(names, dtype="not-a-numpy-dtype"))
    assert caught.value.code == "pipeline_field_dtype_mismatch"

    monkeypatch.setattr(
        PipelineAxisView,
        "_source_array",
        lambda self, descriptor, missing=False: _Array(np.arange(4)),
    )
    with pytest.raises(ArtifactResolutionError) as caught:
        cells._selection_array()
    assert caught.value.code == "pipeline_view_selection_malformed"
    monkeypatch.setattr(
        PipelineAxisView,
        "_source_array",
        lambda self, descriptor, missing=False: _Array(np.asarray([True, False, True])),
    )
    with pytest.raises(ArtifactResolutionError) as caught:
        cells._selection_array()
    assert caught.value.code == "pipeline_field_shape_mismatch"

    monkeypatch.undo()
    monkeypatch.setattr(
        pipeline_run_module,
        "inspect_artifact",
        lambda *_args: (_ for _ in ()).throw(ValueError("bad")),
    )
    with pytest.raises(ArtifactResolutionError) as caught:
        cells._expected_row_fingerprint()
    assert caught.value.code == "pipeline_view_selection_malformed"
    monkeypatch.setattr(
        pipeline_run_module,
        "inspect_artifact",
        lambda *_args: SimpleNamespace(inputs={}),
    )
    with pytest.raises(ArtifactResolutionError) as caught:
        cells._expected_row_fingerprint()
    assert caught.value.code == "row_identity_fingerprint_missing"

    monkeypatch.undo()
    table_type = type(cells._live_table)
    monkeypatch.setattr(
        table_type,
        "_get_array",
        lambda *_args: (_ for _ in ()).throw(KeyError("ids")),
    )
    with pytest.raises(ArtifactResolutionError) as caught:
        cells._validate_row_identity()
    assert caught.value.code == "row_identity_mismatch"


def test_pipeline_descriptor_contract_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    run = _completed_run(_root())
    cells = run.cells
    ids = _descriptor(cells, "ids")
    names = _descriptor(cells, "names")
    clusters = _descriptor(cells, "clusters")
    nullable = _descriptor(cells, "nullable_score")

    with pytest.raises(AssertionError, match="axis"):
        cells._validate_descriptor(replace(names, axis="features"), selected_count=2)
    with pytest.raises(ArtifactResolutionError) as caught:
        cells._validate_descriptor(replace(ids, source_value="other"), selected_count=2)
    assert caught.value.code == "pipeline_field_shape_mismatch"
    with pytest.raises(ArtifactResolutionError) as caught:
        cells._validate_descriptor(
            replace(ids, dtype=np.dtype(np.int64).str), selected_count=2
        )
    assert caught.value.code == "pipeline_field_dtype_mismatch"
    with pytest.raises(ArtifactResolutionError) as caught:
        cells._validate_descriptor(
            replace(ids, missing_mask="missing"), selected_count=2
        )
    assert caught.value.code == "pipeline_field_missing_mask_mismatch"
    with pytest.raises(ArtifactResolutionError) as caught:
        cells._validate_descriptor(
            replace(names, dtype=np.dtype(np.int64).str), selected_count=2
        )
    assert caught.value.code == "pipeline_field_dtype_mismatch"

    monkeypatch.setattr(
        PipelineAxisView,
        "_source_array",
        lambda self, descriptor, missing=False: _Array(np.asarray(["a"])),
    )
    with pytest.raises(ArtifactResolutionError) as caught:
        cells._validate_descriptor(names, selected_count=2)
    assert caught.value.code == "pipeline_field_shape_mismatch"
    monkeypatch.setattr(
        PipelineAxisView,
        "_source_array",
        lambda self, descriptor, missing=False: _Array(np.asarray(["a", "b"])),
    )
    with pytest.raises(ArtifactResolutionError) as caught:
        cells._validate_descriptor(names, selected_count=2)
    assert caught.value.code == "pipeline_field_shape_mismatch"

    monkeypatch.setattr(
        PipelineAxisView,
        "_source_array",
        lambda self, descriptor, missing=False: _Array(
            np.asarray([False, False, False])
            if missing
            else np.arange(4, dtype=np.int32)
        ),
    )
    with pytest.raises(ArtifactResolutionError) as caught:
        cells._validate_descriptor(nullable, selected_count=2)
    assert caught.value.code == "pipeline_field_missing_mask_mismatch"
    monkeypatch.setattr(
        PipelineAxisView,
        "_source_array",
        lambda self, descriptor, missing=False: _Array(np.arange(4, dtype=np.int32)),
    )
    with pytest.raises(ArtifactResolutionError) as caught:
        cells._validate_descriptor(nullable, selected_count=2)
    assert caught.value.code == "pipeline_field_missing_mask_mismatch"

    monkeypatch.undo()
    with pytest.raises(ArtifactResolutionError) as caught:
        cells._validate_descriptor(
            replace(clusters, fill="not-an-integer"), selected_count=2
        )
    assert caught.value.code == "pipeline_field_fill_mismatch"


def test_pipeline_component_and_selected_block_helpers() -> None:
    run = _completed_run(_root())
    cells = run.cells
    features = run.features
    array = _Array(np.arange(12).reshape(4, 3))
    np.testing.assert_array_equal(
        PipelineAxisView._read_component_rows(
            array,  # type: ignore[arg-type]
            np.asarray([3, 1]),
            2,
        ),
        np.asarray([11, 5]),
    )
    assert (
        PipelineAxisView._read_component_rows(
            array,  # type: ignore[arg-type]
            np.asarray([], dtype=np.int64),
            2,
        ).size
        == 0
    )

    feature_blocks = list(features._iter_selection_blocks(block_rows=1))
    assert len(feature_blocks) == 3
    with pytest.raises(ValueError, match="block_rows"):
        list(features._iter_selection_blocks(block_rows=0))

    for columns, error_type in (
        ("ids", TypeError),
        (("",), TypeError),
        (("ids", "ids"), ValueError),
        (("missing",), KeyError),
    ):
        with pytest.raises(error_type):
            list(cells._iter_selected_blocks(columns))  # type: ignore[arg-type]
    blocks = list(cells._iter_selected_blocks(cells.columns, block_rows=1))
    assert sum(len(block.active_global_indices) for block in blocks) == 2


def test_pipeline_plot_dataframe_and_head_edge_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _completed_run(_root())
    cells = run.cells
    assert cells._field_dtype("I") == np.dtype(bool)
    assert cells._field_dtype("ids") == np.dtype(
        cells._live_table._get_array("ids").dtype
    )
    with pytest.raises(KeyError):
        cells._field_dtype("missing")
    assert cells._field_display("I") is None
    with pytest.raises(KeyError):
        cells._field_display("missing")

    np.testing.assert_array_equal(
        PipelineAxisView._apply_plot_missing(
            np.asarray([True, True]), np.asarray([False, False])
        ),
        np.asarray([True, True]),
    )
    np.testing.assert_array_equal(
        PipelineAxisView._apply_plot_missing(
            np.asarray([True, True]), np.asarray([False, True])
        ),
        np.asarray([True, False]),
    )
    numeric = PipelineAxisView._apply_plot_missing(
        np.asarray([1, 2]), np.asarray([False, True])
    )
    assert np.isnan(numeric[1])
    text = PipelineAxisView._apply_plot_missing(
        np.asarray(["a", "b"]), np.asarray([False, True])
    )
    assert text[1] is None

    with monkeypatch.context() as patch:
        patch.setattr(
            PipelineAxisView,
            "fetch_all",
            lambda self, column: np.arange(self._live_table.N),
        )
        patch.setattr(
            PipelineAxisView,
            "fetch",
            lambda self, column: np.arange(self._selected_count()),
        )
        ids_descriptor = cells._descriptor_by_key.pop("ids")
        try:
            assert len(cells._plot_fetch_all("ids")) == 4
            assert len(cells._plot_fetch_selected("ids")) == 2
        finally:
            cells._descriptor_by_key["ids"] = ids_descriptor
    assert cells._selected_prefix_indices(0).size == 0

    for operation in (
        lambda: cells.fetch_all(""),
        lambda: cells.fetch(""),
        lambda: cells.fetch_all("missing"),
        lambda: cells.fetch("missing"),
        lambda: cells.to_pandas_dataframe("ids"),
        lambda: cells.to_pandas_dataframe(("",)),
        lambda: cells.to_pandas_dataframe(("ids", "ids")),
        lambda: cells.to_pandas_dataframe(("missing",)),
        lambda: cells.head(-1),
    ):
        with pytest.raises((TypeError, ValueError, KeyError)):
            operation()
    frame = cells.head(4)
    assert list(frame["nullable_score"].isna()) == [False, True]
    assert "PipelineAxisView" in repr(cells)
    assert run.recipe == "basic_rna_analysis"
    assert run.started_at_ns == 100
    assert run.finished_at_ns == 130
    with pytest.raises(ValueError, match="format"):
        run.report(format="text")  # type: ignore[arg-type]


def test_pipeline_interruption_markdown_report() -> None:
    root = _root()
    record = create_pipeline_run_record(
        root,
        recipe="basic",
        requested_label=None,
        assay="RNA",
        config={},
        stage_order=("one",),
        scarf_version="1.0",
        started_at_ns=10,
    )
    interrupted = interrupt_pipeline_run_record(
        root,
        run_id=record.run_id,
        interruption=PipelineInterruptionRecord("shutdown", "stop", 11),
        finished_at_ns=20,
    )
    run = PipelineRun(_Owner(root), interrupted)
    report = run.report(format="markdown")
    assert "## Interruption" in report
    assert "No completed outputs" in report
