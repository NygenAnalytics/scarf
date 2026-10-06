"""Compact results remain immutable while external histories can be relocated."""

import asyncio
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import zarr

from scarf.agent import AnalysisConfig, Study, analyze_rna, open_analysis
from scarf.agent import api, workflow
from scarf.agent.compact_result import publish_result
from scarf.agent.records import RecordError, RunRecords, procedure_identity
from scarf.agent.result import AnalysisRun


def _completed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    run_id: Any = "agent-storage-test",
    assay: str = "RNA",
    workspace: str | None = None,
    missing_stage: str | None = None,
    selection_decision: bool = True,
) -> Any:
    source = tmp_path / "source.zarr"
    root = zarr.open_group(str(source), mode="w")
    root.attrs["unrelated"] = "preserved"
    config = {
        "hvgCount": 2000,
        "pcaDims": 30,
        "neighborsK": 11,
        "harmonyBatchColumns": [],
        "leiden": {"selected": 0.75, "partitions": [0.75]},
        "umap": True,
        "markers": True,
    }
    final_run = SimpleNamespace(
        run_id="final-core-run",
        assay=assay,
        report=lambda: {"run": {"config": config}, "stages": ["not copied"]},
    )
    records = RunRecords.create(
        tmp_path / "history",
        {
            "runId": run_id,
            "source": "../source.zarr",
            "study": Study(
                context="Published RNA", objective="Describe cells"
            ).model_dump(),
            "config": AnalysisConfig(workspace=workspace).model_dump(),
            "procedureIdentity": procedure_identity(),
        },
    )
    for stage, evidence in {
        "preprocess": {"fingerprint": "frozen-source-fingerprint", "assay": assay},
        "explore": {
            "partitions": {"c2:r0.75": {"candidateId": "c2", "resolution": 0.75}},
            "summaries": [{"candidateId": "c2", "actualHvgCount": 1987}],
        },
        "finalize": {"runId": final_run.run_id, "selected": "c2:r0.75"},
    }.items():
        if stage == missing_stage:
            continue
        path = records.write_json(f"evidence/{stage}.json", evidence)
        records.append("stageCompleted", stage=stage, evidence=path)
    if selection_decision:
        records.append(
            "decisionAccepted",
            stage="finalists",
            output={"rationale": "Coherent markers at the measured resolution"},
        )
    records.append("status", status="completed")
    store = SimpleNamespace(workspace=None)
    bound = Mock(return_value=(store, final_run))
    monkeypatch.setattr(AnalysisRun, "_bound_pipeline", bound)
    result = AnalysisRun(records.path)
    return SimpleNamespace(
        result=result,
        records=result._records,
        source=source,
        root=root,
        config=config,
        run=final_run,
        store=store,
        bound=bound,
    )


@pytest.fixture
def completed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    return _completed(tmp_path, monkeypatch)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("explicit", [False, True])
def test_run_directory_default_uses_manifest_id_and_explicit_paths_still_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, asynchronous: bool, explicit: bool
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    monkeypatch.chdir(tmp_path)

    async def inspect_only(records: Any, *args: Any) -> None:
        records.append("status", status="needsInput", questions=[])

    monkeypatch.setattr(workflow, "run_workflow", inspect_only)
    monkeypatch.setattr(api, "_report", Mock())
    options = {
        "model": object(),
        "study": Study(context="Prepared RNA", objective="Describe cells"),
    }
    if explicit:
        options["run_dir"] = tmp_path / "selected-history"
    result = (
        asyncio.run(api.analyze_rna_async(source, **options))
        if asynchronous
        else analyze_rna(source, **options)
    )
    records = RunRecords(result.run_dir)
    expected = (
        tmp_path / "selected-history"
        if explicit
        else tmp_path / "agent_runs" / records.manifest["runId"]
    )
    assert result.run_dir == expected
    assert result.status == "needsInput"
    assert (expected / "run.json").is_file()
    assert (expected / "events").is_dir()
    assert list(source.iterdir()) == []


def test_compact_result_contains_verified_core_config_without_full_history(
    completed: Any,
) -> None:
    publish_result(completed.result)
    result = completed.result.compact_result
    assert result["agentRunId"] == completed.records.manifest["runId"]
    assert result["finalPipelineRunId"] == completed.run.run_id
    assert result["sourceFingerprint"] == "frozen-source-fingerprint"
    assert result["assay"] == "RNA"
    assert result["workspace"] is None
    assert result["externalRunLocator"] == "../history"
    assert result["selectedParameters"] == {
        "pipelineConfig": completed.config,
        "candidateId": "c2",
        "resolution": 0.75,
        "requestedHvgCount": 2000,
        "actualHvgCount": 1987,
        "pcaDims": 30,
        "neighborsK": 11,
        "useHarmony": False,
    }
    assert result["selectionRationale"] == "Coherent markers at the measured resolution"
    assert (
        result["procedureIdentity"] == completed.records.manifest["procedureIdentity"]
    )
    assert completed.bound.call_count == 2
    root = zarr.open_group(str(completed.source), mode="r")
    assert root.attrs["unrelated"] == "preserved"
    assert root["agent_results"].attrs["owner"] == "scarf.agent"
    assert root[f"agent_results/{result['agentRunId']}"].attrs["result"] == result
    assert list(root["agent_results"].group_keys()) == [result["agentRunId"]]
    assert not {"stages", "calls", "prompts", "credentials", "provider"} & set(result)


def test_compact_record_identifies_the_actual_pipeline_workspace(
    completed: Any,
) -> None:
    # A mounted store may supply its workspace while config.workspace is None.
    completed.store.workspace = "analysis-workspace"
    publish_result(completed.result)
    assert completed.result.compact_result["workspace"] == "analysis-workspace"


def test_default_history_symlink_into_source_is_rejected_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.zarr"
    source.mkdir()
    (tmp_path / "agent_runs").symlink_to(source, target_is_directory=True)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="run_dir must be outside"):
        analyze_rna(
            source,
            model=object(),
            study=Study(context="Prepared RNA", objective="Describe cells"),
        )
    assert list(source.iterdir()) == []


def test_reading_absent_or_existing_summary_never_changes_store_or_history(
    completed: Any,
) -> None:
    def snapshot() -> Any:
        return {
            str(path): path.read_bytes()
            for base in (completed.source, completed.records.path)
            for path in base.rglob("*")
            if path.is_file()
        }

    before = snapshot()
    assert completed.result.compact_result is None
    assert snapshot() == before
    completed.bound.assert_not_called()
    publish_result(completed.result)
    before = snapshot()
    assert open_analysis(completed.records.path).compact_result is not None
    assert snapshot() == before


@pytest.mark.parametrize(
    ("fault", "error", "message"),
    [
        ("source", ValueError, "Source fingerprint changed"),
        ("runId", RecordError, "does not match the verified final pipeline"),
        ("assay", RecordError, "does not match the verified final pipeline"),
        ("resolution", RecordError, "resolution differs from the accepted selection"),
    ],
)
def test_invalid_final_lineage_prevents_store_publication(
    completed: Any, fault: str, error: type[Exception], message: str
) -> None:
    if fault == "source":
        completed.bound.side_effect = ValueError("Source fingerprint changed")
    elif fault == "runId":
        completed.run.run_id = "different-core-run"
    elif fault == "assay":
        completed.run.assay = "different-assay"
    else:
        completed.config["leiden"]["selected"] = 1.25
    with pytest.raises(error, match=message):
        publish_result(completed.result)
    assert "agent_results" not in completed.root
    assert completed.records.latest("resultPublished") is None


@pytest.mark.parametrize("collision", ["group", "array", "assay", "workspace", "child"])
def test_namespace_collision_never_replaces_existing_data(
    completed: Any, collision: str
) -> None:
    if collision == "array":
        completed.root.create_array("agent_results", shape=(1,), dtype="i4")
    else:
        group = completed.root.create_group("agent_results")
        if collision == "assay":
            group.attrs.update(owner="scarf.agent", is_assay=True)
        elif collision == "workspace":
            group.attrs["owner"] = "scarf.agent"
            group.create_group("cellData")
        elif collision == "child":
            group.attrs["owner"] = "scarf.agent"
            group.create_group(completed.records.manifest["runId"])
    before = {
        path.relative_to(completed.source): path.read_bytes()
        for path in completed.source.rglob("*")
        if path.is_file()
    }
    with pytest.raises(RecordError, match="collid"):
        publish_result(completed.result)
    after = {
        path.relative_to(completed.source): path.read_bytes()
        for path in completed.source.rglob("*")
        if path.is_file()
    }
    assert before == after


def test_publication_is_idempotent_and_conflicting_payload_stops(
    completed: Any,
) -> None:
    publish_result(completed.result)
    before = completed.records.events()
    publish_result(completed.result)
    assert completed.records.events() == before
    completed.config["pcaDims"] = 10
    with pytest.raises(RecordError, match="differs from verified"):
        publish_result(completed.result)
    assert completed.records.events() == before
    stored = completed.root["agent_results/agent-storage-test"].attrs["result"]
    assert stored["selectedParameters"]["pcaDims"] == 30


@pytest.mark.parametrize("level", ["namespace", "result"])
def test_result_symlinks_cannot_write_outside_the_selected_store(
    completed: Any, tmp_path: Path, level: str
) -> None:
    outside = tmp_path / "unrelated.zarr"
    outside_root = zarr.open_group(str(outside), mode="w")
    owned = outside_root.create_group(
        "agent_results", attributes={"owner": "scarf.agent"}
    )
    if level == "namespace":
        (completed.source / "agent_results").symlink_to(
            outside / "agent_results", target_is_directory=True
        )
    else:
        completed.root.create_group(
            "agent_results", attributes={"owner": "scarf.agent"}
        )
        owned.create_group(
            "agent-storage-test", attributes={"result": {"unrelated": True}}
        )
        (completed.source / "agent_results" / "agent-storage-test").symlink_to(
            outside / "agent_results" / "agent-storage-test", target_is_directory=True
        )
    before = {
        path.relative_to(outside): path.read_bytes()
        for path in outside.rglob("*")
        if path.is_file()
    }
    with pytest.raises(RecordError, match="symlinks"):
        publish_result(completed.result)
    with pytest.raises(RecordError, match="symlinks"):
        _ = completed.result.compact_result
    assert before == {
        path.relative_to(outside): path.read_bytes()
        for path in outside.rglob("*")
        if path.is_file()
    }
    assert completed.records.latest("resultPublished") is None


@pytest.mark.parametrize("level", ["namespace", "result"])
def test_unowned_plain_directories_are_never_claimed_as_result_groups(
    completed: Any, level: str
) -> None:
    target = completed.source / "agent_results"
    if level == "result":
        completed.root.create_group(
            "agent_results", attributes={"owner": "scarf.agent"}
        )
        target = target / "agent-storage-test"
    target.mkdir()
    (target / "unrelated.txt").write_text("Preserve this data")
    before = {
        path.relative_to(target): path.read_bytes()
        for path in target.rglob("*")
        if path.is_file()
    }
    with pytest.raises(RecordError, match="unowned filesystem path"):
        publish_result(completed.result)
    assert before == {
        path.relative_to(target): path.read_bytes()
        for path in target.rglob("*")
        if path.is_file()
    }


def test_crash_after_zarr_write_recovers_without_overwriting_summary(
    completed: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = completed.records.append

    def interrupted(kind: str, **values: Any) -> Any:
        if kind == "resultPublished":
            raise RuntimeError("Interrupted after result write")
        return original(kind, **values)

    monkeypatch.setattr(completed.records, "append", interrupted)
    with pytest.raises(RuntimeError, match="Interrupted"):
        publish_result(completed.result)
    stored = completed.result.compact_result
    monkeypatch.setattr(completed.records, "append", original)
    publish_result(completed.result)
    assert completed.result.compact_result == stored
    assert completed.records.latest("resultPublished")["recovered"] is True


def test_relocated_history_retains_original_locator_and_records_staleness(
    completed: Any, tmp_path: Path
) -> None:
    publish_result(completed.result)
    stored = completed.result.compact_result
    replacement = tmp_path / "moved-history"
    shutil.move(completed.records.path, replacement)
    reopened = open_analysis(replacement, source=completed.source)
    assert reopened.compact_result == stored
    publish_result(reopened)
    assert reopened.compact_result == stored
    records = RunRecords(replacement)
    event = records.latest("resultLocatorStale")
    assert event["storedLocator"] == "../history"
    assert event["currentLocator"] == "../moved-history"
    before = records.events()
    publish_result(reopened)
    assert records.events() == before


def test_relocated_store_keeps_immutable_summary_and_original_history_locator(
    completed: Any, tmp_path: Path
) -> None:
    publish_result(completed.result)
    stored = completed.result.compact_result
    # Zarr would read a string path as a URL and end it at '#'.
    destination = tmp_path / "relocated#1" / "source.zarr"
    destination.parent.mkdir()
    shutil.move(completed.source, destination)
    relocated = open_analysis(completed.records.path, source=destination)
    assert relocated.compact_result == stored
    publish_result(relocated)
    assert relocated.compact_result == stored
    assert (
        completed.records.latest("resultLocatorStale")["currentLocator"]
        == "../../history"
    )


@pytest.mark.parametrize("remove_namespace", [False, True])
def test_deleted_published_result_is_not_silently_recreated(
    completed: Any, remove_namespace: bool
) -> None:
    publish_result(completed.result)
    target = "agent_results" if remove_namespace else "agent_results/agent-storage-test"
    del completed.root[target]
    before = completed.records.events()
    with pytest.raises(
        RecordError, match="Previously published compact result is missing"
    ):
        publish_result(completed.result)
    assert target not in completed.root
    assert completed.records.events() == before


def test_reading_detects_an_edited_original_locator_without_repairing_it(
    completed: Any,
) -> None:
    publish_result(completed.result)
    child = completed.root["agent_results/agent-storage-test"]
    payload = child.attrs["result"]
    payload["externalRunLocator"] = "../unrelated-history"
    child.attrs["result"] = payload
    before = completed.records.events()
    with pytest.raises(RecordError, match="differs from its publication record"):
        _ = completed.result.compact_result
    # Publishing again cannot adopt the edited summary as the recorded result.
    with pytest.raises(RecordError, match="Recorded compact publication differs"):
        publish_result(completed.result)
    assert child.attrs["result"] == payload
    assert completed.records.events() == before


def test_completed_resume_retries_publication_without_model_or_numerical_work(
    completed: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scarf.agent import compact_result, evidence

    original = compact_result.publish_result
    monkeypatch.setattr(
        compact_result, "publish_result", Mock(side_effect=OSError("disk full"))
    )
    api._publish_result(completed.result, completed.records)
    assert completed.result.status == "completed"
    assert completed.records.latest("resultPublicationError")["errorType"] == "OSError"
    monkeypatch.setattr(compact_result, "publish_result", original)
    monkeypatch.setattr(evidence, "open_store", Mock())
    monkeypatch.setattr(workflow, "validate_final", Mock())
    monkeypatch.setattr(
        workflow, "run_workflow", Mock(side_effect=AssertionError("No rerun"))
    )
    monkeypatch.setattr(api, "_report", Mock())
    resumed = api.resume_rna(completed.records.path, model=None)
    assert resumed.status == "completed"
    assert resumed.compact_result is not None
    assert completed.records.latest("resultPublished") is not None


def test_reporting_failure_does_not_remove_published_completed_result(
    completed: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    api._publish_result(completed.result, completed.records)
    monkeypatch.setattr(
        completed.result, "save_plots", Mock(side_effect=OSError("plot failed"))
    )
    monkeypatch.setattr(
        completed.result, "report", Mock(side_effect=OSError("report failed"))
    )
    api._report(completed.result, completed.records)
    assert completed.result.status == "completed"
    assert completed.result.compact_result is not None
    assert (
        len([e for e in completed.records.events() if e["kind"] == "reportError"]) == 2
    )


def test_publication_errors_are_visible_in_completed_report_summary(
    completed: Any,
) -> None:
    from scarf.agent.rendering import summary

    completed.root.create_group("agent_results")
    api._publish_result(completed.result, completed.records)
    reported = summary(completed.records)
    assert completed.result.status == "completed"
    assert any(row["kind"] == "resultPublicationError" for row in reported["errors"])


def test_incomplete_science_cannot_publish_a_result(completed: Any) -> None:
    completed.records.append("status", status="failed")
    with pytest.raises(RecordError, match="Only completed analyses"):
        publish_result(completed.result)
    assert "agent_results" not in completed.root


def test_lenient_selection_rationale_identifies_the_recorded_resolution_rule(
    completed: Any,
) -> None:
    decision = completed.records.latest("decisionAccepted")
    completed.records.append(
        "decisionResolved",
        acceptedSequence=decision["sequence"],
        rule="orderedAcceptableOption",
    )
    publish_result(completed.result)
    assert (
        "Automatic resolution: orderedAcceptableOption"
        in completed.result.compact_result["selectionRationale"]
    )


def test_malformed_external_locator_is_rejected_without_rewriting(
    completed: Any,
) -> None:
    publish_result(completed.result)
    child = completed.root["agent_results/agent-storage-test"]
    payload = child.attrs["result"]
    payload["externalRunLocator"] = "/absolute/unsupported-history"
    child.attrs["result"] = payload
    with pytest.raises(RecordError, match="valid relative history locator"):
        _ = completed.result.compact_result
    assert child.attrs["result"] == payload


@pytest.mark.parametrize("run_id", ["../escape", "nested/run", "", 7])
def test_invalid_run_identifier_cannot_name_a_result_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, run_id: Any
) -> None:
    completed = _completed(tmp_path, monkeypatch, run_id=run_id)
    for operation in (
        lambda: publish_result(completed.result),
        lambda: completed.result.compact_result,
    ):
        with pytest.raises(RecordError, match="not a valid local result group name"):
            operation()
    assert "agent_results" not in completed.root
    assert completed.records.latest("resultPublished") is None


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        ("preprocess", "missing frozen selection evidence"),
        ("explore", "missing frozen selection evidence"),
        ("finalize", "missing frozen selection evidence"),
        ("decision", "missing its final selection decision"),
    ],
)
def test_completed_status_without_frozen_selection_cannot_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str, message: str
) -> None:
    completed = _completed(
        tmp_path,
        monkeypatch,
        missing_stage=None if damage == "decision" else damage,
        selection_decision=damage != "decision",
    )
    with pytest.raises(RecordError, match=message):
        publish_result(completed.result)
    assert "agent_results" not in completed.root
    assert completed.records.latest("resultPublished") is None


@pytest.mark.parametrize("field", ["assay", "workspace"])
def test_result_namespace_cannot_be_the_selected_assay_or_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    completed = _completed(tmp_path, monkeypatch, **{field: "agent_results"})
    with pytest.raises(RecordError, match="already the selected assay or workspace"):
        publish_result(completed.result)
    assert "agent_results" not in completed.root
    assert completed.records.latest("resultPublished") is None


def test_stored_result_must_be_a_json_object(completed: Any) -> None:
    group = completed.root.create_group(
        "agent_results", attributes={"owner": "scarf.agent"}
    )
    group.create_group("agent-storage-test", attributes={"result": ["not", "a", "map"]})
    for operation in (
        lambda: publish_result(completed.result),
        lambda: completed.result.compact_result,
    ):
        with pytest.raises(RecordError, match="stored compact result is not a JSON"):
            operation()
    assert completed.root["agent_results/agent-storage-test"].attrs["result"] == [
        "not",
        "a",
        "map",
    ]
    assert completed.records.latest("resultPublished") is None
