import asyncio
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

import scarf.datastore._pipeline_ledger as ledger_module
import scarf.datastore._pipeline_recipe as recipe_module
import scarf.storage.pipeline_runs as run_storage
from scarf.datastore._pipeline_ledger import RunLedger
from scarf.datastore.pipeline_run import PipelineExecutionError
from scarf.storage.artifact_writer import ArtifactPlanReceipt
from scarf.storage.pipeline_runs import (
    PipelineErrorRecord,
    PipelineFieldDescriptor,
    PipelineInterruptionRecord,
    PipelineOutputRecord,
    PipelinePlanRecord,
    PipelineRunRecord,
    PipelineStageMetrics,
    PipelineStageOutputRecord,
    PipelineStageRecord,
)
from scarf.storage.refs import ArtifactRef
from scarf.utils.process import ProcessTreeRssMeasurement


def _ref(value: str = "a") -> ArtifactRef:
    return ArtifactRef(
        scope="datastore",
        kind="cell_selection",
        artifact_id=value * 64,
    )


def _metrics() -> PipelineStageMetrics:
    return PipelineStageMetrics(
        wall_seconds=0.1,
        rss_baseline_bytes=10,
        rss_peak_bytes=20,
        rss_incremental_peak_bytes=10,
        sample_interval_seconds=0.1,
        sample_count=1,
        sampling_error_count=0,
        rss_unavailable_reason=None,
    )


def _running_stage(**changes: Any) -> PipelineStageRecord:
    values: dict[str, Any] = {
        "stage": "one",
        "ordinal": 0,
        "started_at_ns": 10,
        "finished_at_ns": None,
        "status": "running",
        "complete": False,
        "outputs": (),
        "plans": (),
        "metrics": None,
        "error": None,
        "interruption": None,
    }
    values.update(changes)
    return PipelineStageRecord(**values)


def _terminal_stage(**changes: Any) -> PipelineStageRecord:
    values: dict[str, Any] = {
        "stage": "one",
        "ordinal": 0,
        "started_at_ns": 10,
        "finished_at_ns": 20,
        "status": "completed",
        "complete": True,
        "outputs": (),
        "plans": (),
        "metrics": _metrics(),
        "error": None,
        "interruption": None,
    }
    values.update(changes)
    return PipelineStageRecord(**values)


def _running_run(**changes: Any) -> PipelineRunRecord:
    values: dict[str, Any] = {
        "run_id": "1" * 64,
        "recipe": "basic",
        "requested_label": None,
        "label": None,
        "assay": "RNA",
        "started_at_ns": 10,
        "finished_at_ns": None,
        "status": "running",
        "complete": False,
        "scarf_version": "1.0",
        "config": {},
        "stage_order": ("one",),
        "outputs": (),
        "fields": (),
        "error": None,
        "interruption": None,
    }
    values.update(changes)
    return PipelineRunRecord(**values)


def _terminal_run(**changes: Any) -> PipelineRunRecord:
    values: dict[str, Any] = {
        "run_id": "1" * 64,
        "recipe": "basic",
        "requested_label": None,
        "label": None,
        "assay": "RNA",
        "started_at_ns": 10,
        "finished_at_ns": 20,
        "status": "completed",
        "complete": True,
        "scarf_version": "1.0",
        "config": {},
        "stage_order": ("one",),
        "outputs": (),
        "fields": (),
        "error": None,
        "interruption": None,
    }
    values.update(changes)
    return PipelineRunRecord(**values)


def test_pipeline_scalar_and_json_contract_rejections() -> None:
    invalid_calls = (
        (run_storage._validate_bool, (1, "value"), TypeError),
        (run_storage._validate_non_negative_int, (True, "value"), TypeError),
        (run_storage._validate_positive_int, (0, "value"), TypeError),
        (run_storage._validate_non_negative_float, (object(), "value"), TypeError),
        (run_storage._validate_non_negative_float, (float("inf"), "value"), ValueError),
        (run_storage._validate_positive_float, (0, "value"), ValueError),
        (run_storage._mapping, ([], "value"), TypeError),
    )
    for function, arguments, error_type in invalid_calls:
        with pytest.raises(error_type):
            function(*arguments)

    assert run_storage._validate_nullable_non_negative_int(None, "value") is None
    assert run_storage._validate_nullable_positive_int(None, "value") is None
    with pytest.raises(ValueError, match="missing.*extra"):
        run_storage._exact_mapping({"extra": 1}, frozenset({"needed"}), "value")
    with pytest.raises(ValueError, match="non-finite"):
        run_storage._json_value(float("nan"), "value")
    with pytest.raises(TypeError, match="keys"):
        run_storage._json_value({1: "bad"}, "value")
    with pytest.raises(TypeError, match="unsupported"):
        run_storage._json_value({1, 2}, "value")
    with pytest.raises(ValueError, match="ArtifactRef"):
        run_storage._artifact_ref({}, "value")
    with pytest.raises(TypeError, match="raised directly"):
        run_storage._raise_type("raised directly")


def test_pipeline_leaf_records_reject_malformed_contracts() -> None:
    with pytest.raises(TypeError, match="message"):
        PipelineErrorRecord("ValueError", 3)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="128"):
        PipelineErrorRecord("x" * 129, "message")
    with pytest.raises(ValueError, match="512"):
        PipelineErrorRecord("ValueError", "x" * 513)
    truncated = PipelineErrorRecord.from_exception(ValueError("x" * 600))
    assert len(truncated.message) == 512
    with pytest.raises(TypeError, match="message"):
        PipelineErrorRecord.from_dict({"type": "ValueError", "message": 1})

    with pytest.raises(ValueError, match="appear together"):
        PipelineInterruptionRecord("signal", "stop", 1, signal_number=2)
    with pytest.raises(ValueError, match="too long"):
        PipelineInterruptionRecord("x" * 129, "stop", 1)

    with pytest.raises(TypeError, match="output artifact"):
        PipelineOutputRecord("out", object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="stage output artifact"):
        PipelineStageOutputRecord("out", object(), False)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="reused"):
        PipelineStageOutputRecord("out", _ref(), 1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="plan ref"):
        PipelinePlanRecord("op", object(), "created")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="disposition"):
        PipelinePlanRecord("op", _ref(), "invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="disposition"):
        PipelinePlanRecord.from_dict(
            {"operation": "op", "ref": _ref().to_dict(), "disposition": "invalid"}
        )


def test_pipeline_metrics_and_field_contract_rejections() -> None:
    base = _metrics()
    with pytest.raises(ValueError, match="baseline"):
        replace(base, rss_peak_bytes=None, rss_unavailable_reason="missing")
    with pytest.raises(ValueError, match="explicit reason"):
        replace(
            base,
            rss_baseline_bytes=None,
            rss_peak_bytes=None,
            rss_incremental_peak_bytes=None,
        )
    with pytest.raises(ValueError, match="unavailable reason"):
        replace(base, rss_unavailable_reason="unexpected")

    valid = PipelineFieldDescriptor(
        key="score",
        axis="cells",
        artifact=_ref(),
        source_value="values",
        value_index=None,
        dtype="float64",
        fill="nan",
        missing_mask=None,
        display={"limits": [0, 1]},
    )
    assert np.isnan(valid.fill_value)
    with pytest.raises(ValueError, match="axis"):
        replace(valid, axis="rows")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="artifact"):
        replace(valid, artifact=object())
    with pytest.raises(ValueError, match="string 'nan'"):
        replace(valid, fill=float("nan"))
    with pytest.raises(TypeError, match="JSON scalar"):
        replace(valid, fill=[])

    raw = valid.to_dict()
    raw["axis"] = "rows"
    with pytest.raises(ValueError, match="axis"):
        PipelineFieldDescriptor.from_dict(raw)
    raw = valid.to_dict()
    raw["fill"] = []
    with pytest.raises(TypeError, match="JSON scalar"):
        PipelineFieldDescriptor.from_dict(raw)


def test_pipeline_stage_record_rejects_inconsistent_states() -> None:
    output = PipelineStageOutputRecord("out", _ref(), False)
    plan = PipelinePlanRecord("op", _ref(), "created")
    interruption = PipelineInterruptionRecord("shutdown", "stop", 1)
    error = PipelineErrorRecord("ValueError", "bad")

    invalid = (
        ({"finished_at_ns": 9}, ValueError, "precede"),
        ({"status": "unknown"}, ValueError, "status"),
        ({"outputs": [output]}, TypeError, "outputs"),
        ({"plans": [plan]}, TypeError, "plans"),
        ({"complete": True}, ValueError, "running stages"),
        ({"metrics": _metrics()}, ValueError, "receipts"),
        ({"error": error}, ValueError, "terminal details"),
    )
    for changes, error_type, message in invalid:
        with pytest.raises(error_type, match=message):
            _running_stage(**changes)

    terminal_invalid = (
        ({"finished_at_ns": None}, "finish time"),
        ({"status": "skipped", "outputs": (output,)}, "skipped"),
        ({"status": "failed"}, "failed stages"),
        (
            {
                "status": "interrupted",
                "interruption": interruption,
                "outputs": (output,),
            },
            "outputs",
        ),
        ({"status": "interrupted"}, "interrupted stages"),
        ({"error": error}, "successful stages"),
        ({"outputs": (output, output)}, "unique"),
    )
    for changes, message in terminal_invalid:
        with pytest.raises(ValueError, match=message):
            _terminal_stage(**changes)

    raw = _running_stage().to_dict()
    raw["status"] = "unknown"
    with pytest.raises(ValueError, match="status"):
        PipelineStageRecord.from_dict(raw)
    raw = _running_stage().to_dict()
    raw["outputs"] = "bad"
    with pytest.raises(TypeError, match="outputs"):
        PipelineStageRecord.from_dict(raw)
    raw = _running_stage().to_dict()
    raw["plans"] = "bad"
    with pytest.raises(TypeError, match="plans"):
        PipelineStageRecord.from_dict(raw)


def test_pipeline_run_record_rejects_inconsistent_states() -> None:
    output = PipelineOutputRecord("out", _ref())
    field = PipelineFieldDescriptor(
        key="I",
        axis="cells",
        artifact=_ref(),
        source_value="values",
        value_index=None,
        dtype="bool",
        fill=False,
        missing_mask=None,
        display=None,
    )
    error = PipelineErrorRecord("ValueError", "bad")
    interruption = PipelineInterruptionRecord("shutdown", "stop", 1)

    running_invalid = (
        ({"finished_at_ns": 9}, ValueError, "precede"),
        ({"status": "unknown"}, ValueError, "status"),
        ({"stage_order": ()}, TypeError, "stage_order"),
        ({"stage_order": ("one", "one")}, ValueError, "unique"),
        ({"outputs": [output]}, TypeError, "outputs"),
        ({"fields": [field]}, TypeError, "fields"),
        ({"complete": True}, ValueError, "running runs"),
        ({"outputs": (output,)}, ValueError, "terminal results"),
        ({"error": error}, ValueError, "terminal details"),
    )
    for changes, error_type, message in running_invalid:
        with pytest.raises(error_type, match=message):
            _running_run(**changes)

    terminal_invalid = (
        ({"finished_at_ns": None}, "finished_at_ns"),
        ({"error": error}, "completed runs"),
        ({"requested_label": "wanted", "label": None}, "label"),
        ({"status": "failed", "error": error, "label": "bad"}, "expose"),
        ({"status": "failed"}, "failed runs require"),
        (
            {
                "status": "interrupted",
                "interruption": interruption,
                "outputs": (output,),
            },
            "expose",
        ),
        ({"status": "interrupted"}, "interrupted runs require"),
        ({"outputs": (output, output)}, "output keys"),
        ({"fields": (field, field)}, "field keys"),
    )
    for changes, message in terminal_invalid:
        with pytest.raises(ValueError, match=message):
            _terminal_run(**changes)

    for key in ("stageOrder", "outputs", "fields"):
        raw = _running_run().to_dict()
        raw[key] = "bad"
        with pytest.raises(TypeError):
            PipelineRunRecord.from_dict(raw)
    raw = _running_run().to_dict()
    raw["status"] = "unknown"
    with pytest.raises(ValueError, match="status"):
        PipelineRunRecord.from_dict(raw)
    raw = _running_run().to_dict()
    raw["runId"] = 1
    with pytest.raises(TypeError, match="runId"):
        PipelineRunRecord.from_dict(raw)


class _Cells:
    def __init__(self) -> None:
        self.columns = {
            "I",
            "RNA_nCounts",
            "RNA_nFeatures",
            "sample",
            "batch",
            "note",
            "numeric",
        }

    def get_dtype(self, column: str) -> np.dtype[Any]:
        return np.dtype(bool if column == "I" else np.int64)


class _FakeRNA:
    pass


class _RecipeStore:
    def __init__(self) -> None:
        self._defaultAssay: str | None = "RNA"
        self.cells = _Cells()
        self.assay: Any = _FakeRNA()

    def _get_assay(self, name: str) -> Any:
        return self.assay


def _recipe_kwargs() -> dict[str, Any]:
    return {
        "assay": "RNA",
        "label": None,
        "cell_key": "I",
        "filtering": False,
        "harmony_batch_columns": None,
        "hvg_count": 10,
        "pca_dims": 5,
        "neighbors_k": 3,
        "umap": False,
        "leiden": True,
        "cell_cycle": False,
        "paris": False,
        "doublets": False,
        "markers": False,
        "snapshot_columns": (),
    }


def test_pipeline_recipe_helper_validation() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        recipe_module._positive_int(True, "count")
    for value in ("name", ["ok", ""], ["same", "same"]):
        with pytest.raises((TypeError, ValueError)):
            recipe_module._column_sequence(value, "columns")
    for value in (True, "one", float("inf"), 0):
        with pytest.raises((TypeError, ValueError)):
            recipe_module._canonical_resolution(value)
    for value in (1, {}, {"partitions": "one"}, {"partitions": []}):
        with pytest.raises((TypeError, ValueError)):
            recipe_module._resolve_leiden(value)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="duplicate"):
        recipe_module._resolve_leiden({"partitions": [1, 1.0]})
    for value in (True, "bad", float("inf")):
        with pytest.raises((TypeError, ValueError)):
            recipe_module._manual_bound(value, "bounds")
    for value in (False, "bad", float("nan")):
        with pytest.raises((TypeError, ValueError)):
            recipe_module._finite_real(value, "value")


def test_pipeline_filtering_contract_errors() -> None:
    store = _RecipeStore()
    invalid = (
        (object(), TypeError, "mapping or bool"),
        ({"method": "other"}, ValueError, "method"),
        ({"attrs": ["missing"]}, KeyError, "not found"),
        ({"attrs": []}, ValueError, "no QC"),
        (
            {"method": "manual", "attrs": ["RNA_nCounts"], "unknown": 1},
            ValueError,
            "Unknown manual",
        ),
        (
            {"method": "manual", "attrs": ["RNA_nCounts"]},
            ValueError,
            "requires lows",
        ),
        (
            {
                "method": "manual",
                "attrs": ["RNA_nCounts"],
                "lows": [0, 1],
                "highs": [2],
            },
            ValueError,
            "align",
        ),
        (
            {
                "method": "manual",
                "attrs": ["RNA_nCounts"],
                "lows": [0],
                "highs": [2],
                "keep_bounds": 1,
            },
            TypeError,
            "keep_bounds",
        ),
        ({"attrs": ["RNA_nCounts"], "unknown": 1}, ValueError, "Unknown automatic"),
        ({"attrs": ["RNA_nCounts"], "min_p": 0.999}, ValueError, "0 < min_p"),
        ({"attrs": ["RNA_nCounts"], "sample_column": ""}, TypeError, "sample_column"),
        (
            {"attrs": ["RNA_nCounts"], "sample_column": "missing"},
            KeyError,
            "Sample column",
        ),
        ({"attrs": ["RNA_nCounts"], "n_mads": 0}, ValueError, "positive"),
        (
            {"attrs": ["RNA_nCounts"], "min_cells_per_sample": 1},
            ValueError,
            "at least 2",
        ),
        (
            {
                "attrs": ["RNA_nCounts"],
                "sample_column": "sample",
                "min_p": 0.02,
            },
            ValueError,
            "cannot be changed",
        ),
    )
    for value, error_type, message in invalid:
        with pytest.raises(error_type, match=message):
            recipe_module._resolve_filtering(store, "RNA", value)  # type: ignore[arg-type]

    manual = recipe_module._resolve_filtering(
        store,
        "RNA",
        {
            "method": "manual",
            "attrs": ["RNA_nCounts"],
            "lows": [None],
            "highs": [10],
            "keep_bounds": True,
        },
    )
    assert manual["lows"] == [None]
    assert manual["keepBounds"] is True


def test_resolve_pipeline_recipe_contract_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(recipe_module, "RNAassay", _FakeRNA)
    store = _RecipeStore()
    kwargs = _recipe_kwargs()
    recipe = recipe_module.resolve_pipeline_recipe(store, **kwargs)
    assert recipe.to_config()["hvgCount"] == 10

    cases: list[tuple[dict[str, Any], type[BaseException], str]] = [
        ({"label": ""}, TypeError, "label"),
        ({"cell_key": ""}, TypeError, "cell_key"),
        ({"cell_key": "missing"}, KeyError, "selection column"),
        ({"cell_key": "numeric"}, TypeError, "boolean"),
        ({"umap": 1}, TypeError, "umap"),
        ({"leiden": False, "doublets": True}, ValueError, "require"),
        ({"snapshot_columns": ["I"]}, ValueError, "reserved"),
        ({"snapshot_columns": ["missing"]}, KeyError, "Snapshot columns"),
        ({"harmony_batch_columns": []}, ValueError, "must not be empty"),
        ({"harmony_batch_columns": ["missing"]}, KeyError, "Harmony columns"),
        ({"hvg_count": 0}, ValueError, "positive integer"),
    ]
    for changes, error_type, message in cases:
        with pytest.raises(error_type, match=message):
            recipe_module.resolve_pipeline_recipe(store, **{**kwargs, **changes})

    store._defaultAssay = None
    with pytest.raises(ValueError, match="No assay"):
        recipe_module.resolve_pipeline_recipe(store, **{**kwargs, "assay": None})
    store._defaultAssay = "RNA"
    store.assay = object()
    with pytest.raises(TypeError, match="RNA assay"):
        recipe_module.resolve_pipeline_recipe(store, **kwargs)


@contextmanager
def _sample_rss() -> Any:
    measurement = ProcessTreeRssMeasurement(
        baseline_bytes=10,
        peak_bytes=20,
        incremental_peak_bytes=10,
        sample_interval_seconds=0.1,
        sample_count=1,
        sampling_error_count=0,
        unavailable_reason=None,
    )
    yield lambda: measurement


def test_run_ledger_records_and_interruption_helpers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = RunLedger(object(), "run", None)
    with pytest.raises(RuntimeError, match="not observed"):
        ledger._records((("out", _ref()),), ())
    plans = (
        ArtifactPlanReceipt("op", _ref(), "reused"),
        ArtifactPlanReceipt("op", _ref(), "created"),
    )
    assert ledger._records((("out", _ref()),), plans)[0].reused is False
    assert ledger._plans(plans)[0].operation == "op"
    assert ledger_module.interruption_record(asyncio.CancelledError()).kind == (
        "asyncio_cancelled"
    )
    with pytest.raises(TypeError, match="handled"):
        ledger.interrupt_pending(ValueError("bad"), "stage")
    with pytest.raises(TypeError, match="handled"):
        ledger._finish_interrupted(
            stage="stage",
            error=ValueError("bad"),
            metrics=_metrics(),
        )

    events = []
    ledger = RunLedger(object(), "run", events.append)
    monkeypatch.setattr(
        ledger_module,
        "load_pipeline_run_record",
        lambda *_args: SimpleNamespace(complete=False),
    )
    interrupted: list[Any] = []
    monkeypatch.setattr(
        ledger_module,
        "interrupt_pipeline_run_record",
        lambda *args, **kwargs: interrupted.append((args, kwargs)),
    )
    ledger.interrupt_pending(KeyboardInterrupt(), "stage")
    assert interrupted and events[-1].kind == "pipeline_interrupted"
    assert ledger._fallback_metrics(0).sample_count >= 1


def test_run_ledger_skip_failure_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ledger_module, "sample_process_tree_rss", _sample_rss)
    monkeypatch.setattr(
        ledger_module, "start_pipeline_stage_record", lambda *a, **k: None
    )
    monkeypatch.setattr(
        ledger_module, "finish_pipeline_stage_record", lambda *a, **k: None
    )
    monkeypatch.setattr(ledger_module, "fail_pipeline_run_record", lambda *a, **k: None)

    calls = 0

    def interrupt_after_start() -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt("stop")

    monkeypatch.setattr(ledger_module, "shutdown_checkpoint", interrupt_after_start)
    interrupted: list[str] = []
    monkeypatch.setattr(
        RunLedger,
        "_finish_interrupted",
        lambda self, **kwargs: interrupted.append(kwargs["stage"]),
    )
    with pytest.raises(KeyboardInterrupt):
        RunLedger(object(), "run", None).skip("stage")
    assert interrupted == ["stage"]

    monkeypatch.setattr(
        ledger_module,
        "start_pipeline_stage_record",
        lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt("stop")),
    )
    pending: list[str] = []
    monkeypatch.setattr(
        RunLedger,
        "interrupt_pending",
        lambda self, error, stage: pending.append(stage),
    )
    monkeypatch.setattr(ledger_module, "shutdown_checkpoint", lambda: None)
    with pytest.raises(KeyboardInterrupt):
        RunLedger(object(), "run", None).skip("pending")
    assert pending == ["pending"]

    class Fatal(BaseException):
        pass

    monkeypatch.setattr(
        ledger_module,
        "start_pipeline_stage_record",
        lambda *a, **k: (_ for _ in ()).throw(Fatal()),
    )
    with pytest.raises(Fatal):
        RunLedger(object(), "run", None).skip("fatal")

    monkeypatch.setattr(
        ledger_module,
        "start_pipeline_stage_record",
        lambda *a, **k: (_ for _ in ()).throw(ValueError("bad")),
    )
    with pytest.raises(PipelineExecutionError):
        RunLedger(object(), "run", None).skip("failed")


def test_run_ledger_run_failure_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ledger_module, "sample_process_tree_rss", _sample_rss)
    monkeypatch.setattr(ledger_module, "shutdown_checkpoint", lambda: None)
    monkeypatch.setattr(ledger_module, "fail_pipeline_run_record", lambda *a, **k: None)
    monkeypatch.setattr(
        ledger_module,
        "start_pipeline_stage_record",
        lambda *a, **k: (_ for _ in ()).throw(ValueError("start")),
    )
    with pytest.raises(PipelineExecutionError, match="start"):
        RunLedger(object(), "run", None).run("stage", lambda: ())

    monkeypatch.setattr(
        ledger_module, "start_pipeline_stage_record", lambda *a, **k: None
    )
    monkeypatch.setattr(
        ledger_module, "finish_pipeline_stage_record", lambda *a, **k: None
    )
    failures: list[BaseException] = []
    monkeypatch.setattr(
        RunLedger,
        "_finish_failed",
        lambda self, **kwargs: failures.append(kwargs["error"]),
    )
    with pytest.raises(PipelineExecutionError, match="not observed"):
        RunLedger(object(), "run", None).run(
            "unplanned",
            lambda: (("out", _ref()),),
        )
    assert isinstance(failures[-1], RuntimeError)

    def fail_finish(*_args: Any, **_kwargs: Any) -> None:
        raise ValueError("finish")

    monkeypatch.setattr(ledger_module, "finish_pipeline_stage_record", fail_finish)
    with pytest.raises(PipelineExecutionError, match="finish"):
        RunLedger(object(), "run", None).run("finish", lambda: ())

    monkeypatch.setattr(
        ledger_module, "finish_pipeline_stage_record", lambda *a, **k: None
    )
    interrupted: list[str] = []
    monkeypatch.setattr(
        RunLedger,
        "_finish_interrupted",
        lambda self, **kwargs: interrupted.append(kwargs["stage"]),
    )
    with pytest.raises(asyncio.CancelledError):
        RunLedger(object(), "run", None).run(
            "cancelled",
            lambda: (_ for _ in ()).throw(asyncio.CancelledError()),
        )
    assert interrupted == ["cancelled"]

    class Fatal(BaseException):
        pass

    with pytest.raises(Fatal):
        RunLedger(object(), "run", None).run(
            "fatal",
            lambda: (_ for _ in ()).throw(Fatal()),
        )


def test_pipeline_record_storage_rejects_torn_and_invalid_queries() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    record = run_storage.create_pipeline_run_record(
        root,
        recipe="basic",
        requested_label=None,
        assay="RNA",
        config={},
        stage_order=("one",),
        scarf_version="1.0",
        run_id="1" * 64,
        started_at_ns=10,
    )
    with pytest.raises(FileExistsError):
        run_storage.create_pipeline_run_record(
            root,
            recipe="basic",
            requested_label=None,
            assay="RNA",
            config={},
            stage_order=("one",),
            scarf_version="1.0",
            run_id=record.run_id,
            started_at_ns=10,
        )
    with pytest.raises(ValueError, match="complete=True"):
        run_storage._write_terminal_attrs(root, {"complete": False})
    with pytest.raises(ValueError, match="positive integer"):
        run_storage.list_pipeline_run_records(root, limit=0)
    with pytest.raises(TypeError, match="status"):
        run_storage.list_pipeline_run_records(root, status=object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least one"):
        run_storage.list_pipeline_run_records(root, status=[])
    with pytest.raises(ValueError, match="Unknown"):
        run_storage.list_pipeline_run_records(root, status="other")
    with pytest.raises(ValueError, match="exactly one"):
        run_storage.open_pipeline_run_record(root)
    with pytest.raises(ValueError, match="exactly one"):
        run_storage.open_pipeline_run_record(root, run_id=record.run_id, label="x")
    with pytest.raises(KeyError, match="No completed"):
        run_storage.open_pipeline_run_record(root, label="missing")

    stages = root[run_storage.pipeline_run_path(record.run_id) + "/stages"]
    stages.create_group("bad")
    with pytest.raises(ValueError, match="child name"):
        run_storage.load_pipeline_stage_records(root, record.run_id)
