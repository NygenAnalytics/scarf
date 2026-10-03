from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

import scarf.datastore.pipeline_run as pipeline_run_module
from scarf.datastore.pipeline_run import (
    PipelineAxisView,
    PipelineExecutionError,
    PipelineRun,
    open_pipeline_run,
)
from scarf.storage.artifacts import artifact_group
from scarf.storage.errors import ArtifactResolutionError
from scarf.storage.pipeline_runs import (
    PipelineErrorRecord,
    PipelineFieldDescriptor,
    PipelineInterruptionRecord,
    create_pipeline_run_record,
    interrupt_pipeline_run_record,
)
from scarf.storage.refs import ArtifactRef
from scarf.storage.selections import resolve_generated_selection_artifact
from tests.test_pipeline_run_foundation import (
    _Owner,
    _artifact,
    _completed_run,
    _root,
)


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

    # Every feature is in the run's feature universe.
    assert [
        (
            block.start,
            block.stop,
            block.selected_indices.tolist(),
            block.compact_start,
            block.compact_stop,
        )
        for block in features._iter_selection_blocks(block_rows=1)
    ] == [(0, 1, [0], 0, 1), (1, 2, [1], 1, 2), (2, 3, [2], 2, 3)]
    with pytest.raises(ValueError, match="^block_rows must be >= 1$"):
        list(features._iter_selection_blocks(block_rows=0))

    for columns, error_type, message in (
        ("ids", TypeError, "^columns must be a sequence of field names$"),
        (("",), TypeError, "^columns must contain non-empty strings$"),
        (("ids", "ids"), ValueError, "^columns must not contain duplicates$"),
        (("missing",), KeyError, r"were not captured: \['missing'\]"),
    ):
        with pytest.raises(error_type, match=message):
            list(cells._iter_selected_blocks(columns))  # type: ignore[arg-type]
    blocks = list(
        cells._iter_selected_blocks(("I", "ids", "batch", "clusters"), block_rows=1)
    )
    assert [block.active_global_indices.tolist() for block in blocks] == [
        [0],
        [],
        [2],
        [],
    ]
    assert [
        {column: values.tolist() for column, values in block.values.items()}
        for block in blocks
    ] == [
        {"I": [True], "ids": ["c1"], "batch": ["x"], "clusters": [0]},
        {"I": [], "ids": [], "batch": [], "clusters": []},
        {"I": [True], "ids": ["c3"], "batch": ["x"], "clusters": [2]},
        {"I": [], "ids": [], "batch": [], "clusters": []},
    ]


def test_component_rows_read_one_value_column_across_stored_chunks() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    values = np.arange(14, dtype=np.float32).reshape(7, 2)
    array = root.create_array("values", data=values, chunks=(3, 2))
    mask = root.create_array("mask", data=np.arange(7) % 3 == 0, chunks=(3,))
    # Rows out of order and spread over every chunk keep their requested order.
    rows = np.asarray([6, 0, 4, 3])

    np.testing.assert_array_equal(
        PipelineAxisView._read_component_rows(array, rows, 1),
        values[rows, 1],
    )
    # One mask describes every component of a multi-value field.
    np.testing.assert_array_equal(
        PipelineAxisView._read_component_rows(mask, rows, 1),
        [True, True, False, True],
    )
    assert PipelineAxisView._read_component_rows(array, rows[:0], 0).shape == (0,)


def test_pipeline_plot_dataframe_and_head_edge_paths() -> None:
    run = _completed_run(_root())
    cells = run.cells
    assert cells._field_dtype("I") == np.dtype(bool)
    assert cells._field_dtype("ids") == np.dtype(
        cells._live_table._get_array("ids").dtype
    )
    assert cells._field_dtype("clusters") == np.dtype(np.int32)
    assert cells._field_dtype("umap_1") == np.dtype(np.float32)
    with pytest.raises(KeyError, match="'missing' was not captured"):
        cells._field_dtype("missing")
    assert cells._field_display("I") is None
    assert cells._field_display("clusters") == {"kind": "categorical"}
    with pytest.raises(KeyError, match="'missing' was not captured"):
        cells._field_display("missing")

    # Fields without a missing mask reach plots unchanged.
    np.testing.assert_array_equal(cells._plot_fetch_all("clusters"), [0, -1, 2, -1])
    np.testing.assert_array_equal(cells._plot_fetch_selected("ids"), ["c1", "c3"])
    assert cells._selected_prefix_indices(0).size == 0

    for operation, error_type, message in (
        (lambda: cells.fetch_all(""), TypeError, "^column must be a non-empty string$"),
        (lambda: cells.fetch(""), TypeError, "^column must be a non-empty string$"),
        (
            lambda: cells.fetch_all("missing"),
            KeyError,
            "'missing' was not captured on the cells axis",
        ),
        (
            lambda: cells.fetch("missing"),
            KeyError,
            "'missing' was not captured on the cells axis",
        ),
        (
            lambda: cells.to_pandas_dataframe("ids"),
            TypeError,
            "^columns must be a sequence of field names$",
        ),
        (
            lambda: cells.to_pandas_dataframe(("",)),
            TypeError,
            "^columns must contain non-empty strings$",
        ),
        (
            lambda: cells.to_pandas_dataframe(("ids", "ids")),
            ValueError,
            "^columns must not contain duplicates$",
        ),
        (
            lambda: cells.to_pandas_dataframe(("missing",)),
            KeyError,
            r"were not captured: \['missing'\]",
        ),
        (lambda: cells.head(-1), ValueError, "^n must be a non-negative integer$"),
    ):
        with pytest.raises(error_type, match=message):
            operation()
    frame = cells.head(4)
    assert list(frame["nullable_score"].isna()) == [False, True]
    assert repr(cells) == (
        f"PipelineAxisView(axis='cells', columns=7, run_id='{run.run_id[:12]}')"
    )
    assert run.recipe == "basic_rna_analysis"
    assert run.started_at_ns == 100
    assert run.finished_at_ns == 130
    with pytest.raises(ValueError, match="^format must be 'dict' or 'markdown'$"):
        run.report(format="text")  # type: ignore[arg-type]


def test_feature_head_reads_the_first_universe_rows_from_frozen_fields() -> None:
    root = _root()
    run = _completed_run(root)

    # The run froze feature names A0 to C0 before the live names changed.
    assert run.features.head(2).to_dict(orient="list") == {
        "I": [True, True],
        "ids": ["g1", "g2"],
        "names": ["A0", "B0"],
        "highly_variable_features": [True, False],
    }


def _append_cell_field(
    root: zarr.Group,
    run: PipelineRun,
    descriptor: PipelineFieldDescriptor,
) -> PipelineRun:
    """Persist one more cell field on a completed run and reopen the run."""
    run_group = root[f"pipeline/runs/{run.run_id}"]
    run_group.attrs["fields"] = [*run_group.attrs["fields"], descriptor.to_dict()]
    return open_pipeline_run(_Owner(root), run_id=run.run_id)


def test_compact_field_missing_mask_applies_to_selected_rows_only() -> None:
    root = _root()
    run = _completed_run(root)
    # The compact score covers the two selected cells; the second is missing.
    score = _artifact(
        root,
        scope="assay",
        assay="RNA",
        kind="doublet_score",
        values={
            "values": np.asarray([0.25, 0.75], dtype=np.float32),
            "values_missing": np.asarray([False, True]),
        },
        inputs={"cell_selection": run["analysis_cell_selection"].to_dict()},
    )
    reopened = _append_cell_field(
        root,
        run,
        PipelineFieldDescriptor(
            key="score",
            axis="cells",
            artifact=score,
            source_value="values",
            value_index=None,
            dtype=np.dtype(np.float32).str,
            fill="nan",
            missing_mask="values_missing",
            display=None,
        ),
    )
    cells = reopened.cells

    # Stored placeholders stay visible to raw reads.
    np.testing.assert_array_equal(cells.fetch("score"), [0.25, 0.75])
    np.testing.assert_array_equal(
        cells.fetch_all("score"), [0.25, np.nan, 0.75, np.nan]
    )
    # Tables and plots show the masked selected row as missing.
    frame = cells.to_pandas_dataframe(["ids", "score"])
    assert frame["ids"].tolist() == ["c1", "c3"]
    np.testing.assert_array_equal(frame["score"].to_numpy(), [0.25, np.nan])
    np.testing.assert_array_equal(cells.head(5)["score"].to_numpy(), [0.25, np.nan])
    np.testing.assert_array_equal(
        cells._plot_fetch_all("score"), [0.25, np.nan, np.nan, np.nan]
    )
    np.testing.assert_array_equal(cells._plot_fetch_selected("score"), [0.25, np.nan])
    blocks = list(cells._iter_selected_blocks(("score",), block_rows=1))
    np.testing.assert_array_equal(
        np.concatenate([block.values["score"] for block in blocks]), [0.25, np.nan]
    )
    assert [len(block.values["score"]) for block in blocks] == [1, 0, 1, 0]


def test_cached_view_rejects_a_compact_field_whose_rows_changed() -> None:
    root = _root()
    run = _completed_run(root)
    cells = run.cells
    np.testing.assert_array_equal(cells.fetch_all("clusters"), [0, -1, 2, -1])

    # A third row no longer matches the two cells the view validated.
    artifact_group(root, run["clusters"]).create_array(
        "values",
        data=np.asarray([0, 2, 5], dtype=np.int32),
        overwrite=True,
    )

    with pytest.raises(
        ArtifactResolutionError, match="no longer aligns to view I"
    ) as caught:
        cells.fetch_all("clusters")
    assert caught.value.code == "pipeline_field_shape_mismatch"
    assert caught.value.context["field"] == "clusters"


@pytest.mark.parametrize("axis", ["cells", "features"])
@pytest.mark.parametrize(
    "change",
    [{"sourceValue": "ids"}, {"valueIndex": 0}, {"missingMask": "values"}],
    ids=["source", "component", "mask"],
)
def test_run_selection_descriptor_must_read_plain_selection_values(
    axis: str,
    change: dict[str, object],
) -> None:
    root = _root()
    run = _completed_run(root)
    run_group = root[f"pipeline/runs/{run.run_id}"]
    run_group.attrs["fields"] = [
        {**field, **change} if (field["axis"], field["key"]) == (axis, "I") else field
        for field in run_group.attrs["fields"]
    ]
    reopened = open_pipeline_run(_Owner(root), run_id=run.run_id)

    with pytest.raises(
        ArtifactResolutionError,
        match=f"^Pipeline {axis} I descriptor does not identify selection values$",
    ) as caught:
        _ = getattr(reopened, axis)
    assert caught.value.code == "pipeline_view_selection_malformed"


def test_run_with_an_empty_cell_selection_has_empty_cell_views() -> None:
    root = _root()
    run = _completed_run(root)
    empty = resolve_generated_selection_artifact(
        root,
        scope="datastore",
        kind="cell_selection",
        values=np.zeros(4, dtype=bool),
        row_ids=np.asarray(["c1", "c2", "c3", "c4"]),
        operation="test_empty_pipeline_selection",
        parameters={},
        inputs={},
        source_column="I",
    )[0]
    run_group = root[f"pipeline/runs/{run.run_id}"]
    # Only full-axis snapshot fields remain valid for an empty selection.
    run_group.attrs["fields"] = [
        {**field, "artifact": empty.to_dict()}
        if (field["axis"], field["key"]) == ("cells", "I")
        else field
        for field in run_group.attrs["fields"]
        if (field["axis"], field["key"])
        not in {("cells", "umap_1"), ("cells", "clusters")}
    ]
    cells = open_pipeline_run(_Owner(root), run_id=run.run_id).cells

    head = cells.head(3)
    assert list(head.columns) == ["I", "ids", "names", "batch", "nullable_score"]
    assert len(head) == 0
    np.testing.assert_array_equal(cells.fetch_all("I"), [False, False, False, False])
    assert cells.fetch("names").size == 0
    assert cells.to_pandas_dataframe(["ids"]).empty


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
    assert report.endswith(
        "## Outputs\n\nNo completed outputs.\n\n"
        "## Interruption\n\n- Kind: `shutdown`\n- Message: stop\n"
    )
    assert "- Label: none\n" in report
    assert "- Finished at ns: `20`\n" in report
    assert "## Failure" not in report
    # An unlabeled run omits the label from its representation.
    assert repr(run) == (
        f"PipelineRun(run_id='{record.run_id[:12]}...', status='interrupted', "
        "assay='RNA')"
    )


def test_labeled_run_representation_names_its_label() -> None:
    run = _completed_run(_root())

    assert repr(run) == (
        f"PipelineRun(run_id='{run.run_id[:12]}...', status='completed', "
        "assay='RNA', label='baseline')"
    )


def test_pipeline_view_calls_validate_row_identity_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cells = _completed_run(_root()).cells
    fingerprint = pipeline_run_module.fingerprint_stored_strings
    calls = 0

    def counting(array: Any) -> str:
        nonlocal calls
        calls += 1
        return fingerprint(array)

    monkeypatch.setattr(pipeline_run_module, "fingerprint_stored_strings", counting)
    operations = (
        lambda: cells.to_pandas_dataframe(cells.columns),
        lambda: cells.fetch("clusters"),
        lambda: cells.fetch("ids"),
        lambda: cells.fetch_all("clusters"),
        lambda: cells._plot_fetch_all("nullable_score"),
        lambda: cells._plot_fetch_selected("nullable_score"),
    )
    for operation in operations:
        calls = 0
        operation()
        assert calls == 1
    frame = cells.to_pandas_dataframe(["ids", "clusters", "nullable_score"])
    assert list(frame["nullable_score"].isna()) == [False, True]
