"""Pipeline stages run one at a time and end with exactly one durable outcome."""

import pytest
import zarr
from zarr.storage import MemoryStore

from scarf.datastore._pipeline_ledger import PipelineEvent, RunLedger
from scarf.datastore.pipeline_run import PipelineExecutionError
from scarf.storage.pipeline_runs import (
    create_pipeline_run_record,
    finish_pipeline_stage_record,
    load_pipeline_run_record,
    load_pipeline_stage_records,
    start_pipeline_stage_record,
)
from scarf.utils.shutdown import ShutdownRequested, ShutdownToken, shutdown_scope
from tests.test_pipeline_contract_edge_coverage import _metrics


def _ledger(*stages: str) -> tuple[zarr.Group, RunLedger, list[PipelineEvent]]:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    run = create_pipeline_run_record(
        root,
        recipe="basic_rna_analysis",
        requested_label=None,
        assay="RNA",
        config={},
        stage_order=stages,
        scarf_version="1.0.0",
    )
    events: list[PipelineEvent] = []
    return root, RunLedger(root, run.run_id, events.append), events


def _statuses(root: zarr.Group, ledger: RunLedger) -> dict[str, str]:
    return {
        record.stage: record.status
        for record in load_pipeline_stage_records(root, ledger.run_id)
    }


def test_grouped_interruption_ends_the_stage_as_that_interruption() -> None:
    root, ledger, events = _ledger("first", "second")
    interruption = KeyboardInterrupt("stop")

    def first_stage() -> tuple[()]:
        raise BaseExceptionGroup(
            "workers failed",
            [ValueError("worker"), BaseExceptionGroup("nested", [interruption])],
        )

    with pytest.raises(KeyboardInterrupt) as caught:
        ledger.run("first", first_stage)

    assert caught.value is interruption
    assert load_pipeline_run_record(root, ledger.run_id).status == "interrupted"
    assert _statuses(root, ledger) == {"first": "interrupted"}
    assert [event.kind for event in events][-2:] == [
        "stage_interrupted",
        "pipeline_interrupted",
    ]


def test_grouped_failure_without_interruption_fails_the_stage() -> None:
    root, ledger, _events = _ledger("first")
    group = ExceptionGroup("workers failed", [ValueError("a"), ValueError("b")])

    def first_stage() -> tuple[()]:
        raise group

    with pytest.raises(PipelineExecutionError) as caught:
        ledger.run("first", first_stage)

    assert caught.value.__cause__ is group
    assert load_pipeline_run_record(root, ledger.run_id).status == "failed"
    assert _statuses(root, ledger) == {"first": "failed"}


def test_interruption_after_a_completed_action_keeps_the_stage_completed() -> None:
    root, ledger, events = _ledger("first", "second")
    token = ShutdownToken()

    def first_stage() -> tuple[()]:
        token.request(reason="stop after this stage")
        return ()

    with shutdown_scope(token), pytest.raises(ShutdownRequested):
        ledger.run("first", first_stage)

    assert load_pipeline_run_record(root, ledger.run_id).status == "interrupted"
    assert _statuses(root, ledger) == {"first": "completed"}
    assert [(event.kind, event.stage) for event in events] == [
        ("stage_started", "first"),
        ("stage_completed", "first"),
        ("pipeline_interrupted", "first"),
    ]


def test_stages_start_only_after_every_earlier_stage_succeeded() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    run = create_pipeline_run_record(
        root,
        recipe="basic_rna_analysis",
        requested_label=None,
        assay="RNA",
        config={},
        stage_order=("a", "b", "c", "d"),
        scarf_version="1.0.0",
    )
    start_pipeline_stage_record(root, run_id=run.run_id, ordinal=0, stage="a")
    with pytest.raises(ValueError, match="sequentially"):
        start_pipeline_stage_record(root, run_id=run.run_id, ordinal=2, stage="c")
    # "a" is still running, so no later stage may start beside it.
    with pytest.raises(ValueError, match="sequentially"):
        start_pipeline_stage_record(root, run_id=run.run_id, ordinal=1, stage="b")
    finish_pipeline_stage_record(
        root, run_id=run.run_id, ordinal=0, status="completed", metrics=_metrics()
    )
    start_pipeline_stage_record(root, run_id=run.run_id, ordinal=1, stage="b")
    finish_pipeline_stage_record(
        root, run_id=run.run_id, ordinal=1, status="skipped", metrics=_metrics()
    )
    start_pipeline_stage_record(root, run_id=run.run_id, ordinal=2, stage="c")
    finish_pipeline_stage_record(
        root,
        run_id=run.run_id,
        ordinal=2,
        status="failed",
        metrics=_metrics(),
        error=RuntimeError("deliberate failure"),
    )
    with pytest.raises(ValueError, match="sequentially"):
        start_pipeline_stage_record(root, run_id=run.run_id, ordinal=3, stage="d")
