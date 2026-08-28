"""Tests for immutable sidecar persistence of the four agent reports."""

import hashlib
import re
from pathlib import Path

import numpy as np
import pytest
import zarr
from pydantic import ValidationError

import scarf.agent as agent_api
from scarf.agent.biological_interpretation import BiologicalInterpretationReport
from scarf.agent.data_enrichment import DataEnrichmentReport
from scarf.agent.experimental_context import ExperimentalContextResult
from scarf.agent.parameter_tuning import ParameterTuningReport
from scarf.agent.persistence import (
    AgentReportReference,
    AgentWorkflowRun,
    create_agent_workflow,
    list_agent_reports,
    list_agent_workflows,
    load_agent_report,
    load_agent_workflow,
    save_agent_report,
)
from scarf.agent.types import AgentDataModel


REPORT_CASES = (
    (DataEnrichmentReport, "data_enrichment"),
    (ExperimentalContextResult, "experimental_context"),
    (ParameterTuningReport, "parameter_tuning"),
    (BiologicalInterpretationReport, "biological_interpretation"),
)


def _report(report_type: type[AgentDataModel], execution_run_id: str) -> AgentDataModel:
    report = report_type.get_example()
    return report.model_copy(
        update={
            "runInfo": report.runInfo.model_copy(
                update={
                    "agentName": "intentionally_misleading",
                    "modelName": "modèle-α",
                    "runId": execution_run_id,
                }
            )
        }
    )


def _record_group(
    path: Path,
    reference: AgentReportReference,
) -> zarr.Group:
    root = zarr.open_group(str(path), mode="r+")
    node = root[
        f"runs/{reference.workflowRunId}/{reference.agentName}/{reference.agentRunId}"
    ]
    assert isinstance(node, zarr.Group)
    return node


def test_persistence_models_have_factories_and_camelcase_fields() -> None:
    for model_type in (AgentReportReference, AgentWorkflowRun):
        assert isinstance(model_type.get_blank(), model_type)
        assert isinstance(model_type.get_example(), model_type)
        assert all("_" not in field_name for field_name in model_type.model_fields)
    assert agent_api.create_agent_workflow is create_agent_workflow
    assert agent_api.save_agent_report is save_agent_report
    assert agent_api.load_agent_report is load_agent_report
    assert agent_api.list_agent_reports is list_agent_reports
    assert agent_api.list_agent_workflows is list_agent_workflows
    assert agent_api.load_agent_workflow is load_agent_workflow


def test_persistence_models_reject_invalid_nonblank_metadata() -> None:
    with pytest.raises(ValidationError):
        AgentReportReference(workflowRunId="../workflow")
    with pytest.raises(ValidationError):
        AgentReportReference(createdAtNs=-1)
    with pytest.raises(ValidationError):
        AgentReportReference(complete=1)  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        AgentReportReference(reportType="UnknownReport")  # type: ignore[arg-type]


@pytest.mark.parametrize(("report_type", "agent_name"), REPORT_CASES)
def test_all_agent_reports_round_trip_in_exact_hierarchy(
    tmp_path: Path,
    report_type: type[AgentDataModel],
    agent_name: str,
) -> None:
    path = tmp_path / "analysis.agents.zarr"
    workflow = create_agent_workflow(
        path,
        workflow_run_id="workflow-1",
        analysis_store="analysis.zarr",
        dataset_fingerprints={"RNA": "dataset-1"},
    )
    report = _report(report_type, f"provider-{agent_name}")

    reference = save_agent_report(
        path,
        workflow.workflowRunId,
        report,  # type: ignore[arg-type]
        agent_run_id=f"{agent_name}-run",
    )
    loaded = load_agent_report(path, reference)

    assert reference.agentName == agent_name
    assert reference.executionRunId == f"provider-{agent_name}"
    assert type(loaded) is report_type
    assert loaded.model_dump(mode="json") == report.model_dump(mode="json")
    root = zarr.open_group(str(path), mode="r")
    assert root.attrs["format"] == "scarf_agent_reports"
    assert root.attrs["format_version"] == 1
    record_path = f"runs/workflow-1/{agent_name}/{agent_name}-run"
    assert record_path in root
    assert f"{record_path}/report_json" in root
    assert root[record_path].attrs["complete"] is True


def test_workflow_load_and_report_listing_are_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "analysis.agents.zarr"
    create_agent_workflow(
        path,
        workflow_run_id="workflow-1",
        analysis_store="analysis.zarr",
        dataset_fingerprints={"RNA": "dataset-1"},
    )
    for index, (report_type, _agent_name) in enumerate(reversed(REPORT_CASES)):
        save_agent_report(
            path,
            "workflow-1",
            _report(report_type, f"provider-{index}"),  # type: ignore[arg-type]
            agent_run_id=f"run-{index}",
        )

    first = list_agent_reports(path, "workflow-1")
    second = list_agent_reports(path, "workflow-1")
    workflow = load_agent_workflow(path, "workflow-1")

    assert first == second
    assert first == sorted(
        first,
        key=lambda item: (item.createdAtNs, item.agentName, item.agentRunId),
    )
    assert workflow.reports == first
    assert workflow.analysisStore == "analysis.zarr"
    assert workflow.datasetFingerprints == {"RNA": "dataset-1"}
    assert {item.agentName for item in first} == {
        agent_name for _report_type, agent_name in REPORT_CASES
    }
    filtered = list_agent_reports(
        path,
        "workflow-1",
        agent_name="parameter_tuning",
    )
    assert [item.agentName for item in filtered] == ["parameter_tuning"]


def test_generated_workflow_and_agent_run_ids_are_safe(tmp_path: Path) -> None:
    path = tmp_path / "analysis.agents.zarr"
    workflow = create_agent_workflow(path)
    reference = save_agent_report(
        path,
        workflow.workflowRunId,
        DataEnrichmentReport.get_example(),
    )

    assert re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,127}", workflow.workflowRunId)
    assert re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,127}", reference.agentRunId)


def test_existing_complete_or_incomplete_report_is_never_overwritten(
    tmp_path: Path,
) -> None:
    path = tmp_path / "analysis.agents.zarr"
    create_agent_workflow(path, workflow_run_id="workflow-1")
    report = DataEnrichmentReport.get_example()
    reference = save_agent_report(
        path,
        "workflow-1",
        report,
        agent_run_id="run-1",
    )

    with pytest.raises(FileExistsError, match="already exists"):
        save_agent_report(
            path,
            "workflow-1",
            report,
            agent_run_id="run-1",
        )

    group = _record_group(path, reference)
    group.attrs["complete"] = False
    with pytest.raises(FileExistsError, match="already exists"):
        save_agent_report(
            path,
            "workflow-1",
            report,
            agent_run_id="run-1",
        )
    with pytest.raises(RuntimeError, match="incomplete"):
        load_agent_report(path, reference)
    assert list_agent_reports(path, "workflow-1") == []
    incomplete = list_agent_reports(
        path,
        "workflow-1",
        include_incomplete=True,
    )
    assert len(incomplete) == 1
    assert incomplete[0].complete is False


def test_workflow_ids_are_immutable_and_workflows_are_isolated(tmp_path: Path) -> None:
    path = tmp_path / "analysis.agents.zarr"
    create_agent_workflow(path, workflow_run_id="workflow-1")
    create_agent_workflow(path, workflow_run_id="workflow-2")
    with pytest.raises(FileExistsError, match="already exists"):
        create_agent_workflow(path, workflow_run_id="workflow-1")

    first_report = _report(DataEnrichmentReport, "provider-1")
    second_report = _report(DataEnrichmentReport, "provider-2")
    first = save_agent_report(
        path,
        "workflow-1",
        first_report,  # type: ignore[arg-type]
        agent_run_id="shared-run",
    )
    second = save_agent_report(
        path,
        "workflow-2",
        second_report,  # type: ignore[arg-type]
        agent_run_id="shared-run",
    )

    assert list_agent_reports(path, "workflow-1") == [first]
    assert list_agent_reports(path, "workflow-2") == [second]
    assert load_agent_report(path, first).runInfo.runId == "provider-1"
    assert load_agent_report(path, second).runInfo.runId == "provider-2"
    workflows = list_agent_workflows(path)
    assert {item.workflowRunId for item in workflows} == {"workflow-1", "workflow-2"}
    assert [item.workflowRunId for item in workflows] == [
        item.workflowRunId
        for item in sorted(
            workflows,
            key=lambda item: (item.createdAtNs, item.workflowRunId),
        )
    ]


def test_save_requires_an_existing_sidecar_and_workflow(tmp_path: Path) -> None:
    path = tmp_path / "analysis.agents.zarr"
    report = DataEnrichmentReport.get_example()

    with pytest.raises(FileNotFoundError):
        save_agent_report(path, "workflow-1", report)
    assert not path.exists()

    create_agent_workflow(path, workflow_run_id="workflow-1")
    with pytest.raises(KeyError, match="Unknown agent workflow"):
        save_agent_report(path, "workflow-2", report)
    root = zarr.open_group(str(path), mode="r")
    assert set(root["runs"].group_keys()) == {"workflow-1"}


def test_load_rejects_payload_corruption(tmp_path: Path) -> None:
    path = tmp_path / "analysis.agents.zarr"
    create_agent_workflow(path, workflow_run_id="workflow-1")
    reference = save_agent_report(
        path,
        "workflow-1",
        DataEnrichmentReport.get_example(),
        agent_run_id="run-1",
    )
    group = _record_group(path, reference)
    payload = group["report_json"]
    payload[0] = (int(payload[0]) + 1) % 256

    with pytest.raises(ValueError, match="checksum"):
        load_agent_report(path, reference)


def test_load_rejects_schema_invalid_payload_with_valid_checksum(
    tmp_path: Path,
) -> None:
    path = tmp_path / "analysis.agents.zarr"
    create_agent_workflow(path, workflow_run_id="workflow-1")
    reference = save_agent_report(
        path,
        "workflow-1",
        DataEnrichmentReport.get_example(),
        agent_run_id="run-1",
    )
    group = _record_group(path, reference)
    payload = group["report_json"]
    invalid_json = b"{}" + (b" " * (payload.shape[0] - 2))
    payload[:] = np.frombuffer(invalid_json, dtype=np.uint8)
    group.attrs["payload_sha256"] = hashlib.sha256(invalid_json).hexdigest()

    with pytest.raises(ValueError, match="Pydantic schema"):
        load_agent_report(path, reference)


def test_load_rejects_report_identity_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "analysis.agents.zarr"
    create_agent_workflow(path, workflow_run_id="workflow-1")
    reference = save_agent_report(
        path,
        "workflow-1",
        DataEnrichmentReport.get_example(),
        agent_run_id="run-1",
    )
    group = _record_group(path, reference)
    group.attrs["agent_name"] = "parameter_tuning"

    with pytest.raises(ValueError, match="name identity mismatch"):
        load_agent_report(path, reference)


def test_load_rejects_missing_payload_and_type_mismatches(tmp_path: Path) -> None:
    path = tmp_path / "analysis.agents.zarr"
    create_agent_workflow(path, workflow_run_id="workflow-1")
    reference = save_agent_report(
        path,
        "workflow-1",
        DataEnrichmentReport.get_example(),
        agent_run_id="run-1",
    )
    group = _record_group(path, reference)

    group.attrs["report_type"] = "ParameterTuningReport"
    with pytest.raises(ValueError, match="report type"):
        load_agent_report(path, reference)
    group.attrs["report_type"] = "DataEnrichmentReport"

    mismatched_reference = reference.model_copy(
        update={"reportType": "ParameterTuningReport"}
    )
    with pytest.raises(ValueError, match="reference type"):
        load_agent_report(path, mismatched_reference)

    group.attrs["execution_run_id"] = "different-execution"
    with pytest.raises(ValueError, match="reference execution ID"):
        load_agent_report(path, reference)
    minimal_reference = AgentReportReference(
        workflowRunId=reference.workflowRunId,
        agentName=reference.agentName,
        agentRunId=reference.agentRunId,
    )
    with pytest.raises(ValueError, match="JSON payload"):
        load_agent_report(path, minimal_reference)
    group.attrs["execution_run_id"] = reference.executionRunId

    del group["report_json"]
    with pytest.raises(ValueError, match="missing its JSON payload"):
        load_agent_report(path, reference)


@pytest.mark.parametrize(
    "invalid_id",
    ["", ".", "..", "../x", "a/b", r"a\b", "UPPER", " leading", "x\x00"],
)
def test_run_ids_must_be_safe_single_path_components(
    tmp_path: Path,
    invalid_id: str,
) -> None:
    path = tmp_path / "analysis.agents.zarr"
    with pytest.raises(ValueError, match="safe path component"):
        create_agent_workflow(path, workflow_run_id=invalid_id)
    assert not path.exists()


@pytest.mark.parametrize("invalid_id", ["", "../x", "a/b", "UPPER"])
def test_agent_run_ids_are_validated_before_writing(
    tmp_path: Path,
    invalid_id: str,
) -> None:
    path = tmp_path / "analysis.agents.zarr"
    create_agent_workflow(path, workflow_run_id="workflow-1")

    with pytest.raises(ValueError, match="safe path component"):
        save_agent_report(
            path,
            "workflow-1",
            DataEnrichmentReport.get_example(),
            agent_run_id=invalid_id,
        )

    root = zarr.open_group(str(path), mode="r")
    workflow = root["runs/workflow-1"]
    assert isinstance(workflow, zarr.Group)
    assert list(workflow.group_keys()) == []


def test_sidecar_suffix_prevents_writing_to_analysis_store(tmp_path: Path) -> None:
    analysis_path = tmp_path / "analysis.zarr"

    with pytest.raises(ValueError, match="agents.zarr"):
        create_agent_workflow(analysis_path, workflow_run_id="workflow-1")

    assert not analysis_path.exists()

    zarr.open_group(str(analysis_path), mode="w", zarr_format=3)
    nested_path = analysis_path / "results.agents.zarr"
    with pytest.raises(ValueError, match="inside another Zarr store"):
        create_agent_workflow(nested_path, workflow_run_id="workflow-1")
    assert not nested_path.exists()


def test_local_sidecar_does_not_adopt_an_ordinary_directory(tmp_path: Path) -> None:
    path = tmp_path / "analysis.agents.zarr"
    path.mkdir()
    sentinel = path / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(ValueError, match="non-Zarr directory"):
        create_agent_workflow(path, workflow_run_id="workflow-1")

    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert not (path / "zarr.json").exists()


def test_remote_sidecar_urls_are_rejected() -> None:
    with pytest.raises(ValueError, match="local paths only"):
        create_agent_workflow(
            "s3://example/analysis.agents.zarr",
            workflow_run_id="workflow-1",
        )


def test_foreign_or_unknown_version_store_is_rejected(tmp_path: Path) -> None:
    foreign_path = tmp_path / "foreign.agents.zarr"
    foreign = zarr.open_group(str(foreign_path), mode="w", zarr_format=3)
    foreign.create_group("unrelated")
    with pytest.raises(ValueError, match="non-empty"):
        create_agent_workflow(foreign_path, workflow_run_id="workflow-1")

    path = tmp_path / "analysis.agents.zarr"
    create_agent_workflow(path, workflow_run_id="workflow-1")
    root = zarr.open_group(str(path), mode="r+")
    root.attrs["format_version"] = 2
    with pytest.raises(ValueError, match="only version 1"):
        load_agent_workflow(path, "workflow-1")


def test_boolean_format_versions_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "analysis.agents.zarr"
    create_agent_workflow(path, workflow_run_id="workflow-1")
    root = zarr.open_group(str(path), mode="r+")
    root.attrs["format_version"] = True

    with pytest.raises(ValueError, match="only version 1"):
        load_agent_workflow(path, "workflow-1")


def test_missing_store_reads_do_not_create_a_sidecar(tmp_path: Path) -> None:
    path = tmp_path / "analysis.agents.zarr"

    with pytest.raises(FileNotFoundError):
        list_agent_reports(path, "workflow-1")
    with pytest.raises(FileNotFoundError):
        load_agent_workflow(path, "workflow-1")
    assert not path.exists()


def test_unknown_agent_filter_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "analysis.agents.zarr"
    create_agent_workflow(path, workflow_run_id="workflow-1")

    with pytest.raises(ValueError, match="Unknown agent name"):
        list_agent_reports(
            path,
            "workflow-1",
            agent_name="unknown",  # type: ignore[arg-type]
        )


def test_default_listing_skips_an_interrupted_bare_report(tmp_path: Path) -> None:
    path = tmp_path / "analysis.agents.zarr"
    create_agent_workflow(path, workflow_run_id="workflow-1")
    root = zarr.open_group(str(path), mode="r+")
    root.create_group("runs/workflow-1/data_enrichment/interrupted")

    assert list_agent_reports(path, "workflow-1") == []
    with pytest.raises(ValueError, match="Malformed agent report"):
        list_agent_reports(
            path,
            "workflow-1",
            include_incomplete=True,
        )


def test_save_rejects_unknown_report_model(tmp_path: Path) -> None:
    path = tmp_path / "analysis.agents.zarr"
    create_agent_workflow(path, workflow_run_id="workflow-1")

    with pytest.raises(TypeError, match="four Scarf agent report models"):
        save_agent_report(
            path,
            "workflow-1",
            AnyReport.get_example(),  # type: ignore[arg-type]
        )


class AnyReport(AgentDataModel):
    value: str = ""

    @classmethod
    def get_example(cls) -> "AnyReport":
        return cls(value="not an agent report")
